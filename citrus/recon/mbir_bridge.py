"""Bridge between the scatter solver's geometry (Grid, ConeGeometry) and MBIRTORCH's ConeBeamModel.

MBIRTORCH and CITRUS use the same frames (docs/source/conventions.rst adopts MBIRTORCH's verbatim), so the bridge is
a parameter copy; verified against mbirtorch 0.1.2 `cone_beam.py` (tests/test_mbir_bridge.py):

    recon [rows, cols, slices], col -> x, row -> y, slice -> z, voxel centers at (index - (n - 1)/2) * pitch
    (+ recon_slice_offset along z); view frame x_v = cos(a) x - sin(a) y, y_v = sin(a) x + cos(a) y;
    per-voxel magnification 1 / (1/M - y_v/SDD) = SDD / (SID - y_v)  (source on +y_v);
    channel n = (u + det_channel_offset) / delta_det_channel + (num_channels - 1)/2 with u = mag * x_v,
    row     m = (v + det_row_offset) / delta_det_row + (num_rows - 1)/2 with v = mag * z,
    i.e. exactly our `ConeGeometry.pixel_uv` (u = (c - (Nc-1)/2) dc - det_channel_offset, same for v).

The ConeBeamModel projector is a separable-footprint (distance-driven-like) model of the pixel-averaged line
integral (units: value x ALU; with ALU = cm and volumes in 1/cm the sinogram is the dimensionless line integral).

MBIRTORCH's default region-of-reconstruction mask is the ellipse inscribed in the recon (x, y) box; the WLAB grid
tiles the prism exactly, so the mask would drop the prism corners. `make_cone_model` therefore sets
use_ror_mask=False by default (all voxels projected and updated).

Device handling: MBIRTORCH accepts torch tensors (any device) and places them as float32 on the model device with
`torch.as_tensor` (no copy when they already are float32 on that device). All functions here return torch tensors on
the model device (`output_sharded=True` on a single-device model), never numpy.

Projections in chunks of views (`ViewChunkProjector`): a ConeBeamModel projects a volume onto ALL its views at once.
To project onto a subset of consecutive views (e.g. the CITRUS loop's per-material primary path lengths, which must
not grow to [M', V, Rd, Cd]), a sub-model over those angles is built with `make_cone_model` (measured on the WLAB
grids: about 1 ms to build, 0.1 s for its first projection, which compiles; then the same time per view as the full
model). The projections of a sub-model equal the matching views of the full model to float rounding (~1e-5 absolute
in line integrals of order 10).

Resuming MBIR (`mbir_iterations(..., error_sinogram=e, return_checkpoint=True)`): MBIRTORCH's public `recon` (0.1.2)
recomputes the error sinogram e = y - A x_init on every call (one forward projection) and does not forward the
`init_error_sinogram` / `return_checkpoint` arguments of its engine `_vcd_recon`. With them, `mbir_iterations`
replicates `recon` (the same `initialize_recon` call: partitions, partition sequence from `first_iteration`, input
checks, regularization; then `_vcd_recon` under torch.no_grad, `close_log_file` and the same recon_dict) and passes a
caller-kept error sinogram consistent with init_recon (trusted, not checked; MBIRTORCH updates it in place, and
init_recon too). Without those arguments the plain `recon` path runs.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor

from ..scatter_solver.geometry import ConeGeometry, Grid

__all__ = [
    "make_cone_model",
    "forward_project",
    "back_project",
    "fdk",
    "scan_range",
    "is_short_scan",
    "short_scan_weights",
    "mbir_iterations",
    "freeze_regularization",
    "ViewChunkProjector",
    "mbirtorch_version",
]

DeviceLike = Union[torch.device, str, None]


def mbirtorch_version() -> str:
    import mbirtorch

    return str(mbirtorch.__version__)


def _angles_array(angles) -> np.ndarray:
    if isinstance(angles, Tensor):
        angles = angles.detach().cpu().numpy()
    return np.asarray(angles, dtype=np.float64).reshape(-1)


def make_cone_model(grid: Grid, geom: ConeGeometry, angles: Union[Sequence[float], np.ndarray, Tensor], *,
                    device: DeviceLike = None, use_ror_mask: bool = False, compile_mode: Optional[str] = None,
                    verbose: int = 0, **params):
    """MBIRTORCH `ConeBeamModel` whose forward projection of a volume [R, C, S] on `grid` lands on the panel of
    `geom` with our pixel indexing (rows along +z, channels along +x_view, same offsets), one view per angle (rad).

    Sets: sinogram_shape (V, num_det_rows, num_det_channels), source_iso_dist, source_detector_dist (constructor);
    delta_det_channel, delta_det_row, det_channel_offset, det_row_offset, recon_shape = grid.shape,
    delta_voxel, voxel_row_aspect, voxel_slice_aspect, recon_slice_offset (set_params), use_ror_mask (default False:
    every voxel of the grid is projected / updated), verbose (default 0: no per-iteration logging).
    device: the model's single device (default: cuda if available, else cpu); pinned with
    `configure_devices(devices=[device])`, which also switches off MBIRTORCH's automatic multi-GPU layout.
    compile_mode: None -> 'auto' (torch.compile) on CUDA, 'off' on CPU (torch.compile on CPU needs a C++ toolchain).
    **params: further `set_params` entries (e.g. snr_db, sharpness, positivity_flag, granularity,
    partition_sequence).
    """
    import mbirtorch as mt

    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if compile_mode is None:
        compile_mode = "auto" if dev.type == "cuda" else "off"
    ang = _angles_array(angles)
    model = mt.ConeBeamModel((len(ang), geom.num_det_rows, geom.num_det_channels), ang,
                             source_detector_dist=float(geom.source_detector_dist),
                             source_iso_dist=float(geom.source_iso_dist), compile_mode=compile_mode)
    model.set_params(no_warning=True,
                     delta_det_channel=float(geom.delta_det_channel), delta_det_row=float(geom.delta_det_row),
                     det_channel_offset=float(geom.det_channel_offset), det_row_offset=float(geom.det_row_offset),
                     recon_shape=tuple(int(n) for n in grid.shape), delta_voxel=float(grid.delta_voxel),
                     voxel_row_aspect=float(grid.voxel_row_aspect),
                     voxel_slice_aspect=float(grid.voxel_slice_aspect),
                     recon_slice_offset=float(grid.recon_slice_offset), use_ror_mask=use_ror_mask, verbose=int(verbose))
    if params:
        model.set_params(**params)
    model.configure_devices(devices=[str(dev)])
    return model


def forward_project(model, volume: Tensor) -> Tensor:
    """MBIRTORCH forward projection A x: volume [R, C, S] (1/cm) -> sinogram [views, rows, channels] (line
    integrals), float32 torch tensor on the model device."""
    return model.forward_project(volume, output_sharded=True)


def back_project(model, sinogram: Tensor) -> Tensor:
    """MBIRTORCH back projection A^T y: sinogram [views, rows, channels] -> volume [R, C, S] on the model device."""
    return model.back_project(sinogram, output_sharded=True)


def scan_range(angles) -> float:
    """Angular range (rad) covered by views evenly spaced over [start, end) with the endpoint excluded:
    (a[-1] - a[0]) V / (V - 1) (2 pi for a full rotation of V views; 0 for a single view)."""
    a = _angles_array(angles)
    if a.size < 2:
        return 0.0
    return float(abs(a[-1] - a[0]) * a.size / (a.size - 1))


def is_short_scan(angles, tol: float = 1e-6) -> bool:
    """True when the evenly spaced views cover less than a full rotation (`scan_range` < 2 pi)."""
    return scan_range(angles) < 2.0 * np.pi - tol


def short_scan_weights(geom: ConeGeometry, angles, device: DeviceLike = None) -> Optional[Tensor]:
    """FDK redundancy weights [V, 1, Cd] (float32) for views evenly spaced over a range Lambda < 2 pi, or None for a
    full rotation (MBIRTORCH's FDK is then exact as is).

    MBIRTORCH's `recon_fdk` applies no short-scan weighting: it scales every view by pi / V, the full-scan weight
    (1/2) (2 pi / V). For a short scan this method multiplies the line integrals before the FDK by
    Lambda / pi * w(beta, gamma) with the Parker weights in the generalized form of Wesarg et al. (Med. Phys. 29, 2002;
    an overscan delta = (Lambda - pi) / 2 >= gamma_m allowed), beta = (i + 1/2) Lambda / V the view position in the
    arc and gamma = -atan(u / SDD) the in-plane fan angle of channel u (the sign of MBIRTORCH's frame, tested):

        w = sin^2(pi/4 beta / (delta - gamma))              0 <= beta < 2 (delta - gamma)
            1                                               2 (delta - gamma) <= beta < pi - 2 gamma
            sin^2(pi/4 (pi + 2 delta - beta) / (delta + gamma))     pi - 2 gamma <= beta <= pi + 2 delta

    so that the weights of the two measurements of every in-plane line sum to 1. The weights need delta >= gamma_m,
    i.e. Lambda >= pi + fan angle; for a shorter range a RuntimeWarning is issued and None returned (plain MBIRTORCH
    FDK: incomplete data, shading). The weights act per channel (the in-plane Parker weighting of the cone-beam FDK,
    approximate off the central plane like FDK itself)."""
    a = _angles_array(angles)
    V = a.size
    lam = scan_range(a)
    if V < 2 or lam >= 2.0 * np.pi - 1e-6:
        return None
    sdd = float(geom.source_detector_dist)
    u = (np.arange(geom.num_det_channels) - (geom.num_det_channels - 1) / 2.0) * float(geom.delta_det_channel) \
        - float(geom.det_channel_offset)
    gamma = _PARKER_SIGN * np.arctan(u / sdd)  # [Cd]
    gm = float(np.abs(gamma).max())
    delta = 0.5 * (lam - np.pi)
    if delta < gm - 1e-9:
        import warnings

        warnings.warn(f"scan range {np.degrees(lam):.1f} deg is shorter than a short scan (180 deg + fan angle "
                      f"{np.degrees(2 * gm):.1f} deg): no redundancy weighting, the FDK is incomplete", RuntimeWarning)
        return None
    beta = (np.arange(V) + 0.5) * lam / V
    if a[-1] < a[0]:
        beta = beta[::-1].copy()
    B, Gm = beta[:, None], gamma[None, :]
    w = np.ones((V, gamma.size))
    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = B < 2.0 * (delta - Gm)
        w = np.where(r1, np.sin(np.pi / 4.0 * B / np.where(r1, delta - Gm, 1.0)) ** 2, w)
        r3 = B >= np.pi - 2.0 * Gm
        w = np.where(r3, np.sin(np.pi / 4.0 * np.clip(np.pi + 2.0 * delta - B, 0.0, None)
                                / np.where(r3, delta + Gm, 1.0)) ** 2, w)
    w = w * (lam / np.pi)
    dev = torch.device(device) if device is not None else None
    return torch.as_tensor(w[:, None, :].astype(np.float32), device=dev)


_PARKER_SIGN = -1.0  # gamma = -atan(u / SDD): orientation relative to the rotation in MBIRTORCH's frame (measured:
# the other sign leaves a 13 % error on an off-center test object, this one the full-rotation FDK error; tests)


def fdk(model, sinogram: Tensor, filter_name: str = "ramp", *, geom: Optional[ConeGeometry] = None,
        angles=None) -> Tensor:
    """FDK reconstruction (`ConeBeamModel.recon_fdk` = `recon_direct`: ramp filter with the FDK cosine pre-weight,
    then the adjoint of the forward projector) of line integrals [views, rows, channels] -> volume [R, C, S] (1/cm)
    on the model device. MBIRTORCH's FDK assumes a full rotation (no short-scan weighting); given the scan `geom` and
    `angles` of a short scan (`is_short_scan`), the line integrals are first multiplied by the Parker weights of
    `short_scan_weights` (without them a half-rotation FDK carries the usual short-scan shading and a wrong scale)."""
    if geom is not None and angles is not None:
        w = short_scan_weights(geom, angles, device=sinogram.device if isinstance(sinogram, Tensor) else None)
        if w is not None:
            sinogram = torch.as_tensor(sinogram, dtype=torch.float32, device=w.device) * w
    return model.recon_fdk(sinogram, filter_name=filter_name, output_sharded=True)


def mbir_iterations(model, sinogram: Tensor, init_recon: Union[Tensor, np.ndarray], first_iteration: int,
                    num_iterations: int = 1, weights: Optional[Tensor] = None,
                    stop_threshold_change_pct: float = 0.0, *, error_sinogram: Optional[Tensor] = None,
                    hessian: Optional[Tensor] = None, return_checkpoint: bool = False):
    """`num_iterations` warm-started VCD iterations from `init_recon` at schedule position `first_iteration`
    (the recipe verified in tests/test_mbir_bridge.py):

        model.recon(sinogram, weights, init_recon=init_recon, max_iterations=first_iteration + num_iterations,
                    first_iteration=first_iteration, stop_threshold_change_pct=0, ...)

    `recon` builds the full partition sequence of length max_iterations and DROPS its first `first_iteration`
    entries, so `max_iterations` must be the END position (first_iteration + num_iterations): `max_iterations=1,
    first_iteration=k >= 1` runs ZERO iterations. The error sinogram is recomputed as sinogram - A init_recon (no
    rescaling of a user init), the Hessian diagonal from `weights`, and the regularization is re-estimated from
    `sinogram` unless `freeze_regularization(model)` was called. The partitions are redrawn from numpy's global RNG
    at every call (the granularity of iteration k is the k-th of the default schedule; the random subsets differ
    from those of one continuous run). init_recon is not modified.

    sinogram, weights: [views, rows, channels] line integrals and weights (None = 1). Returns (recon [R, C, S]
    torch on the model device, recon_dict with recon_params['partition_sequence'], ['num_iterations'], ...).

    Resume (module docstring): error_sinogram [views, rows, channels] = sinogram - A init_recon, kept by the caller
    from an earlier call (its checkpoint), skips the initializing forward projection; MBIRTORCH trusts the pair and
    updates BOTH error_sinogram and init_recon in place (pass a copy of init_recon to keep it). hessian: the
    forward-model Hessian diagonal of an earlier call with the same weights and model (None: computed, one back
    projection). return_checkpoint: also return {'error_sinogram': e = sinogram - A recon, 'fm_hessian': ...}, the
    loop's own device tensors (no copy), as a third element. Any of the three switches runs MBIRTORCH's engine as
    `recon` would (same partitions and partition sequence for the same numpy RNG state, same regularization), so a
    resumed call equals the plain one up to the float rounding of the carried error sinogram.
    """
    k = int(first_iteration)
    n = int(num_iterations)
    if k < 0 or n < 1:
        raise ValueError("need first_iteration >= 0 and num_iterations >= 1")
    if isinstance(init_recon, Tensor):
        init_recon = init_recon.detach()
    if error_sinogram is None and hessian is None and not return_checkpoint:
        with torch.no_grad():
            recon, recon_dict = model.recon(sinogram, weights=weights, init_recon=init_recon, max_iterations=k + n,
                                            first_iteration=k, stop_threshold_change_pct=stop_threshold_change_pct,
                                            logfile_path=None, print_logs=False, output_sharded=True)
        return recon, recon_dict
    if error_sinogram is not None and init_recon is None:
        raise ValueError("error_sinogram needs init_recon (the pair must be consistent)")
    recon, recon_dict, checkpoint = _vcd_recon_resumable(model, sinogram, weights, init_recon, k + n, k,
                                                         stop_threshold_change_pct, error_sinogram, hessian)
    if return_checkpoint:
        return recon, recon_dict, checkpoint
    return recon, recon_dict


def _vcd_recon_resumable(model, sinogram, weights, init_recon, max_iterations: int, first_iteration: int,
                         stop_threshold_change_pct: float, error_sinogram, hessian):
    """MBIRTORCH 0.1.2 `TomographyModel.recon` (tomography_model.py) step by step, with `_vcd_recon`'s
    init_error_sinogram / fm_hessian / return_checkpoint arguments, which `recon` does not forward. Returns (recon
    [R, C, S] on the model device, recon_dict as `recon` builds it, checkpoint {'error_sinogram', 'fm_hessian'})."""
    import datetime

    from mbirtorch import _sharding
    from mbirtorch._utils import recon_param_names

    _sharding.reject_shards('recon', init_recon=init_recon)
    (sinogram, weights, init_recon, partitions, partition_sequence, granularity,
     regularization_params) = model.initialize_recon(sinogram, weights, init_recon, max_iterations, first_iteration,
                                                     logfile_path=None, print_logs=False, rng=None)
    with torch.no_grad():
        recon, loss_vectors, checkpoint = model._vcd_recon(
            sinogram, partitions, partition_sequence, stop_threshold_change_pct, weights=weights,
            init_recon=init_recon, first_iteration=first_iteration, init_error_sinogram=error_sinogram,
            fm_hessian=hessian, return_checkpoint=True, rng=None)
    partition_sequence = [int(val) for val in partition_sequence]
    fm_rmse = [float(val) for val in loss_vectors[0]]
    recon_params = dict(zip(recon_param_names,
                            [len(fm_rmse), granularity, partition_sequence, fm_rmse, [0], regularization_params,
                             [100 * float(val) for val in loss_vectors[2]], [float(val) for val in loss_vectors[3]],
                             [[float(v) for v in row] for row in loss_vectors[4]]]))
    for h in list(model.logger.handlers):
        h.flush()
    model.close_log_file()
    recon_dict = model.get_recon_dict(recon_params, notes=f"Reconstruction completed: {datetime.datetime.now()}\n\n")
    return recon, recon_dict, checkpoint


def freeze_regularization(model) -> dict[str, float]:
    """Keep the current sigma_y, sigma_x, sigma_prox in every later `recon` call (auto_regularize_flag=False; by
    default every `recon` call re-estimates them from the sinogram it is given). Returns the frozen values."""
    model.set_params(auto_regularize_flag=False)
    names = ["sigma_y", "sigma_x", "sigma_prox"]
    return {k: float(v) for k, v in zip(names, model.get_params(names))}


class ViewChunkProjector:
    """MBIRTORCH forward projections of volumes on `grid` onto the views of a scan (`geom`, `angles` [V] rad), in
    chunks of consecutive views (module docstring): `project(volume, start, stop)` -> [stop - start, Rd, Cd] line
    integrals (value x cm). The views [0, V) use `model` when it is given (the reconstruction's own ConeBeamModel over
    all angles); any other range uses a sub-model over angles[start:stop] (`make_cone_model`, built on first use and
    kept in an LRU of `max_models` entries). `chunks(n)` lists the (start, stop) ranges of n views each."""

    def __init__(self, grid: Grid, geom: ConeGeometry, angles, *, model=None, device: DeviceLike = None,
                 max_models: int = 8) -> None:
        self.grid = grid
        self.geom = geom
        self.angles = _angles_array(angles)
        self.model = model
        self.device = device
        self.max_models = max(1, int(max_models))
        self._models: "OrderedDict[tuple[int, int], object]" = OrderedDict()

    @property
    def num_views(self) -> int:
        return int(self.angles.size)

    def chunks(self, views_per_chunk: int) -> list[tuple[int, int]]:
        """[(start, stop), ...] covering [0, V) in runs of `views_per_chunk` views (the last one shorter)."""
        n = max(1, int(views_per_chunk))
        return [(a, min(a + n, self.num_views)) for a in range(0, self.num_views, n)]

    def model_for(self, start: int, stop: int):
        """The ConeBeamModel of the views [start, stop)."""
        start, stop = int(start), int(stop)
        if not 0 <= start < stop <= self.num_views:
            raise ValueError(f"view range [{start}, {stop}) outside [0, {self.num_views})")
        if (start, stop) == (0, self.num_views) and self.model is not None:
            return self.model
        key = (start, stop)
        m = self._models.get(key)
        if m is None:
            m = make_cone_model(self.grid, self.geom, self.angles[start:stop], device=self.device)
            self._models[key] = m
            while len(self._models) > self.max_models:
                self._models.popitem(last=False)
        else:
            self._models.move_to_end(key)
        return m

    def project(self, volume: Tensor, start: int = 0, stop: Optional[int] = None) -> Tensor:
        """Forward projection of volume [R, C, S] onto the views [start, stop) (default all): [stop - start, Rd, Cd]
        float32 on the model device."""
        stop = self.num_views if stop is None else int(stop)
        return forward_project(self.model_for(start, stop), volume)
