"""Multigrid stages of the CITRUS loop: coarse grids and binned data first, each stage starting from the last.

A schedule with ``stages`` (``Schedule.stages``) runs the CITRUS loop once per :class:`Stage`. A stage takes every n-th
view, averages the counts over blocks of detector rows and channels and reconstructs on a grid with larger voxels
within the same extent; it starts from the previous stage's volume, interpolated to its grid, and keeps the per-view
scatter state of the previous stage when the computed views and the transport grid are the same. The last stage is at
full resolution. The design is in docs/source/method.rst, "Multigrid stages".
"""

from __future__ import annotations

import copy
import math
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .scatter_solver import Grid, ScatterSolver

__all__ = ["Stage"]

# the MBIRTORCH settings of a given model that every stage's model takes (the user's MBIR settings)
_MODEL_SETTINGS = ("sharpness", "snr_db", "T", "q", "p", "positivity_flag", "qggmrf_nbr_wts")


@dataclass(frozen=True)
class Stage:
    """One level of a multigrid schedule, relative to the full-resolution grid and data.

    Attributes:
        voxel_factor (float): voxel size over that of the reconstruction grid, a number >= 1 (e.g. 0.5 / 0.0947 for
            5 mm voxels over a 0.947 mm grid); each axis gets floor(n / voxel_factor) voxels, so that the stage's grid
            lies within the reconstruction grid's extent. Defaults to 1.
        row_binning (int): detector rows averaged into one. Defaults to 1.
        channel_binning (int): detector channels averaged into one. Defaults to 1.
        view_stride (int): every n-th view is used. Defaults to 1.
        num_outer_iterations (int): outer iterations of the stage. Defaults to 1.
    """

    voxel_factor: float = 1
    row_binning: int = 1
    channel_binning: int = 1
    view_stride: int = 1
    num_outer_iterations: int = 1

    @property
    def full_resolution(self) -> bool:
        """True when the stage uses the reconstruction grid and every view and pixel."""
        return self.voxel_factor == self.row_binning == self.channel_binning == self.view_stride == 1

    def validate(self) -> None:
        """Check the settings.

        Raises:
            ValueError: if the voxel factor is not a number >= 1 or another factor not an integer >= 1.
        """
        factor = self.voxel_factor
        if isinstance(factor, bool) or not isinstance(factor, (int, float, np.integer, np.floating)) or not (
                np.isfinite(float(factor)) and float(factor) >= 1):
            raise ValueError(f"Stage.voxel_factor must be a number >= 1, got {factor!r}")
        for name in ("row_binning", "channel_binning", "view_stride", "num_outer_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"Stage.{name} must be an integer >= 1, got {value!r}")


def _bin(x: Tensor, row_binning: int, channel_binning: int) -> Tensor:
    """Mean of x [..., R, C] over blocks of rows and channels (R and C divisible by the binning)."""
    *lead, R, C = x.shape
    return x.reshape(*lead, R // row_binning, row_binning, C // channel_binning, channel_binning).mean(dim=(-3, -1))


@dataclass
class _Panel:
    """The rows and channels a stage keeps (a remainder of the binning is trimmed at both ends) and its binning."""

    rows: slice
    channels: slice
    row_binning: int
    channel_binning: int
    view_stride: int

    def image(self, x: Tensor) -> Tensor:
        """A [..., R, C] image, trimmed and binned; an axis of length 1 (broadcast over the panel) is kept."""
        rows, row_binning = (slice(None), 1) if x.shape[-2] == 1 else (self.rows, self.row_binning)
        channels, channel_binning = (slice(None), 1) if x.shape[-1] == 1 else (self.channels, self.channel_binning)
        return _bin(x[..., rows, channels], row_binning, channel_binning)

    def views(self, x: Tensor) -> Tensor:
        """A [V, R, C] stack: every n-th view, trimmed and binned."""
        return self.image(x[::self.view_stride])

    def channel_profile(self, x: Tensor) -> Tensor:
        """A [C] profile, trimmed and binned along the channels."""
        if x.shape[-1] == 1:
            return x
        x = x[self.channels]
        return x.reshape(-1, self.channel_binning).mean(dim=1)

    def any_shape(self, x: Tensor, num_views: int) -> Tensor:
        """An array of the measurement shapes [V, R, C], [R, C] or [C], binned like the measurements."""
        if x.dim() == 3:
            return self.views(x) if x.shape[0] == num_views else self.image(x)
        if x.dim() == 2:
            return self.image(x)
        return self.channel_profile(x)


def _stage_weights(panel: _Panel, weights: Tensor, num_views: int) -> Tensor:
    """Measurement weights binned like the measurements; a block with a weight-0 pixel (a dead pixel) gets weight 0."""
    return panel.any_shape(weights, num_views) * (panel.any_shape((weights > 0).float(), num_views) == 1)


def _stage_panel(geometry, stage: Stage):
    """The trimmed and binned panel of a stage and its geometry.

    Args:
        geometry (ConeGeometry or ParallelGeometry): full-resolution geometry.
        stage (Stage): the stage.

    Returns:
        (_Panel, geometry): the kept pixels and the stage's geometry, with the panel centre moved by the trim.
    """
    R, C = int(geometry.num_det_rows), int(geometry.num_det_channels)
    rb, cb = int(stage.row_binning), int(stage.channel_binning)
    if R < rb or C < cb:
        raise ValueError(f"a stage bins {rb} x {cb} pixels of a {R} x {C} panel")
    top, left = (R % rb) // 2, (C % cb) // 2
    bottom, right = R % rb - top, C % cb - left
    panel = _Panel(slice(top, R - bottom), slice(left, C - right), rb, cb, int(stage.view_stride))
    stage_geometry = replace(geometry, num_det_rows=(R - top - bottom) // rb,
                             num_det_channels=(C - left - right) // cb,
                             delta_det_row=geometry.delta_det_row * rb,
                             delta_det_channel=geometry.delta_det_channel * cb,
                             det_row_offset=geometry.det_row_offset - 0.5 * (top - bottom) * geometry.delta_det_row,
                             det_channel_offset=geometry.det_channel_offset
                             - 0.5 * (left - right) * geometry.delta_det_channel)
    return panel, stage_geometry


def _stage_grid(grid: Grid, stage: Stage) -> Grid:
    """The grid of a stage: voxels ``voxel_factor`` times larger, floor(n / voxel_factor) of them per axis."""
    factor = float(stage.voxel_factor)
    return replace(grid, shape=tuple(max(1, int(math.floor(int(n) / factor + 1e-6))) for n in grid.shape),
                   delta_voxel=grid.delta_voxel * factor)


def _resample(volume: Tensor, shape) -> Tensor:
    """A volume [R, C, S] resampled to another shape, the extents matched: averaged over the fine voxels of each coarse
    voxel when the shape is smaller along every axis (thin objects keep their line integrals), else trilinear."""
    if tuple(volume.shape) == tuple(shape):
        return volume
    if all(int(n) <= int(m) for n, m in zip(shape, volume.shape)):
        return F.adaptive_avg_pool3d(volume[None, None].float(), tuple(int(n) for n in shape))[0, 0]
    return F.interpolate(volume[None, None].float(), size=tuple(shape), mode="trilinear", align_corners=False)[0, 0]


def _stage_solver(original: ScatterSolver, template: ScatterSolver, geometry,
                  panel: Optional[_Panel]) -> ScatterSolver:
    """A solver for a stage's panel: the original's source binned like the data, the template's library and settings.

    Args:
        original (ScatterSolver): the scanner's solver for the full-resolution panel.
        template (ScatterSolver): the solver of the previous stage (its built library and settings are shared), or the
            original.
        geometry (ConeGeometry or ParallelGeometry): the stage's geometry.
        panel (_Panel or None): the stage's kept and binned pixels; None for the full-resolution panel.

    Returns:
        ScatterSolver: the stage's solver.
    """
    solver = copy.copy(template)
    solver.geometry = geometry
    _clear_caches(solver)
    source = original.source
    solver.source = source
    if panel is None:
        return solver
    full = original.geometry
    changes: dict = {}
    trimmed_rows = panel.rows.start + (full.num_det_rows - panel.rows.stop)
    trimmed_channels = panel.channels.start + (full.num_det_channels - panel.channels.stop)
    if trimmed_rows:  # the field keeps its edges where trimmed rows or channels left the panel
        changes["field_margin_rows_cm"] = source.field_margin_rows_cm + 0.5 * trimmed_rows * full.delta_det_row
    if trimmed_channels:
        changes["field_margin_channels_cm"] = (source.field_margin_channels_cm
                                               + 0.5 * trimmed_channels * full.delta_det_channel)
    if source.emission_profile is not None:
        profile = torch.as_tensor(source.emission_profile, dtype=torch.float64)
        if profile.dim() == 3 and source.emission_profile_angles is not None:
            changes["emission_profile_angles"] = np.asarray(source.emission_profile_angles)
        changes["emission_profile"] = panel.image(profile)
    if source.filter_thickness_cm is not None:
        thickness = torch.as_tensor(source.filter_thickness_cm, dtype=torch.float64)
        changes["filter_thickness_cm"] = panel.image(thickness) if thickness.dim() == 2 else panel.channel_profile(
            thickness)
    solver.source = replace(source, **changes) if changes else source
    return solver


def _to_host(value):
    """A copy of ``value`` (nested dicts, lists and tuples) with every tensor moved to the CPU."""
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_host(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_host(item) for item in value)
    return value


def _clear_caches(solver: ScatterSolver) -> None:
    """Drop a stage solver's per-grid and per-direction caches (its library stays, shared with the next stage)."""
    solver._sweep_contexts, solver._transport_grids, solver._cone_quadratures = {}, {}, {}
    solver._direction_contexts = OrderedDict()


def _stage_model(make_model: Callable, grid: Grid, geometry, angles: np.ndarray, device, model) -> tuple:
    """The MBIRTORCH model of a stage with the given model's MBIR settings and its region resampled.

    Args:
        make_model (callable): ``recon.mbir_bridge.make_model``.
        grid (Grid): the stage's grid.
        geometry (ConeGeometry or ParallelGeometry): the stage's geometry.
        angles (ndarray): [V] the stage's view angles in rad.
        device (torch.device): device.
        model (ConeBeamModel or None): the user's full-resolution model, or None.

    Returns:
        (model, Tensor or None): the stage's model and its region [R, C, 1] (None for every voxel).
    """
    stage_model = make_model(grid, geometry, angles, device=device)
    region = None
    if model is not None:
        settings = {}
        for name in _MODEL_SETTINGS:
            try:
                settings[name] = model.get_params(name)
            except Exception:  # a setting MBIRTORCH does not have
                pass
        if settings:
            stage_model.set_params(no_warning=True, **settings)
        ror = model.get_params("use_ror_mask")
        if not isinstance(ror, (bool, np.bool_)):
            full = torch.as_tensor(np.asarray(ror), dtype=torch.float32, device=device)
            full = full[..., None] if full.dim() == 2 else full
            shape = (grid.shape[0], grid.shape[1], full.shape[2] if full.shape[2] == 1 else grid.shape[2])
            resampled = F.interpolate(full[None, None], size=shape, mode="nearest")[0, 0] > 0.5
            mask = resampled[:, :, 0] if resampled.shape[2] == 1 else resampled
            stage_model.set_params(use_ror_mask=mask.cpu().numpy())
            region = resampled.float()
    return stage_model, region


def multigrid_recon(citrus_single: Callable, Y, I0, angles, grid: Grid, geometry, solver: ScatterSolver, config,
                    schedule, *, regularization, init_recon, model, callback, mbir_callback, detector_gain,
                    make_model: Callable, close_store: Callable) -> tuple[Tensor, dict]:
    """The CITRUS loop over the stages of a schedule (``citrus_recon`` with ``Schedule.stages``).

    Args:
        citrus_single (callable): the single-resolution loop, with ``stage_carry`` and ``view_ids`` keywords.
        Y (Tensor or ndarray): [V, Rd, Cd] measured counts.
        I0 (float, Tensor or ndarray): flat field (a number, [Rd, Cd] or [V, Rd, Cd]).
        angles (ndarray): [V] view angles in rad.
        grid (Grid): full-resolution reconstruction grid.
        geometry (ConeGeometry or ParallelGeometry): full-resolution geometry.
        solver (ScatterSolver): the scanner's solver for the full-resolution panel.
        config (CitrusConfig): loop settings; its weights and known objects are binned and resampled per stage.
        schedule (Schedule): the schedule with ``stages``; its other settings apply to every stage.
        regularization, init_recon, model, callback, mbir_callback, detector_gain: as for ``citrus_recon``; the
            model's MBIR settings and region apply to every stage, the callbacks' records get ``"stage"``.
        make_model (callable): ``recon.mbir_bridge.make_model``.
        close_store (callable): closes a stage's state store.

    Returns:
        (Tensor, dict): the last stage's volume and info, with ``info["stages"]`` the info of every stage.
    """
    device = solver.device
    Y = torch.as_tensor(Y, device=device, dtype=torch.float32)
    angles = np.asarray(angles, dtype=np.float64).reshape(-1)
    V = len(angles)
    flat = I0 if isinstance(I0, (int, float, np.floating, np.integer)) else torch.as_tensor(
        I0, device=device, dtype=torch.float32)
    gain = None if detector_gain is None else torch.as_tensor(detector_gain, device=device, dtype=torch.float32)
    weights = None if config.measurement_weights is None else torch.as_tensor(
        config.measurement_weights, device=device, dtype=torch.float32)
    known = None if config.known_objects is None else torch.as_tensor(
        config.known_objects, device=device, dtype=torch.float32)
    stages = list(schedule.stages)
    carry: dict = {}
    infos: list[dict] = []
    template = solver
    X: Optional[Tensor] = None
    try:
        for s, stage in enumerate(stages):
            last = s == len(stages) - 1
            stride = int(stage.view_stride)
            view_ids = np.arange(V)[::stride]
            if stage.full_resolution:
                stage_geometry, stage_grid = geometry, grid
                stage_solver = solver if template is solver else _stage_solver(solver, template, geometry, None)
                stage_Y, stage_flat, stage_gain, stage_weights, stage_known = Y, flat, gain, weights, known
                stage_angles = angles
                stage_model = model if model is not None else make_model(grid, geometry, angles, device=device)
                region = None
                if model is not None:
                    ror = model.get_params("use_ror_mask")
                    if not isinstance(ror, (bool, np.bool_)):
                        region = torch.as_tensor(np.asarray(ror), dtype=torch.float32, device=device)
                        region = region[..., None] if region.dim() == 2 else region
            else:
                panel, stage_geometry = _stage_panel(geometry, stage)
                stage_grid = _stage_grid(grid, stage)
                stage_angles = angles[::stride]
                stage_Y = panel.views(Y)
                stage_flat = flat if not isinstance(flat, Tensor) else panel.any_shape(flat, V)
                stage_gain = None if gain is None else panel.image(gain)
                stage_weights = None if weights is None else _stage_weights(panel, weights, V)
                stage_known = None if known is None else _resample(known, stage_grid.shape)
                stage_solver = _stage_solver(solver, template, stage_geometry, panel)
                stage_model, region = _stage_model(make_model, stage_grid, stage_geometry, stage_angles, device,
                                                   model)
            if X is None:
                start = None if init_recon is None else _resample(
                    torch.as_tensor(init_recon, device=device, dtype=torch.float32), stage_grid.shape)
            else:  # the previous stage's volume on this grid
                start = _resample(X, stage_grid.shape)
                if region is not None:
                    start = start * region
            stage_config = replace(config, num_outer_iterations=None, measurement_weights=stage_weights,
                                   known_objects=stage_known,
                                   scatter_view_stride=max(1, int(config.scatter_view_stride) // stride))
            stage_schedule = replace(schedule, stages=None, num_outer_iterations=int(stage.num_outer_iterations))

            def tagged(function, index=s + 1):
                if function is None:
                    return None
                return lambda k, volume, record: function(k, volume, {**record, "stage": index})

            X, info = citrus_single(stage_Y, stage_flat, stage_angles, stage_grid, stage_geometry, stage_solver,
                                    stage_config, schedule=stage_schedule, regularization=regularization,
                                    init_recon=start, model=stage_model, callback=tagged(callback),
                                    mbir_callback=tagged(mbir_callback), detector_gain=stage_gain,
                                    stage_carry=carry, view_ids=view_ids)
            info.update(stage=s + 1, grid=stage_grid, geometry=stage_geometry, num_views=len(stage_angles))
            if not last:  # the earlier stages' info on the host, their solver's caches and model freed
                info = _to_host(info)
                if stage_solver is not solver:
                    _clear_caches(stage_solver)
                del stage_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            infos.append(info)
            template = stage_solver
    finally:
        if carry.get("store") is not None:
            close_store(carry["store"])
    out = dict(infos[-1])
    out["stages"] = infos
    return X, out
