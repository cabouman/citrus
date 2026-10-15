"""Bridge between the scatter solver's geometry (Grid, ConeGeometry) and MBIRTORCH's ConeBeamModel.

MBIRTORCH and CITRUS use the same frames (docs/source/conventions.rst adopts MBIRTORCH's verbatim), so the bridge is
a parameter copy; verified against mbirtorch 0.0.2 `cone_beam.py` (tests/test_mbir_bridge.py):

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
"""

from __future__ import annotations

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
    "mbir_iterations",
    "freeze_regularization",
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


def fdk(model, sinogram: Tensor, filter_name: str = "ramp") -> Tensor:
    """FDK reconstruction (`ConeBeamModel.recon_fdk` = `recon_direct`: ramp filter with the FDK cosine pre-weight,
    then the adjoint of the forward projector; full-scan, no short-scan weighting) of line integrals
    [views, rows, channels] -> volume [R, C, S] (1/cm) on the model device."""
    return model.recon_fdk(sinogram, filter_name=filter_name, output_sharded=True)


def mbir_iterations(model, sinogram: Tensor, init_recon: Union[Tensor, np.ndarray], first_iteration: int,
                    num_iterations: int = 1, weights: Optional[Tensor] = None,
                    stop_threshold_change_pct: float = 0.0) -> tuple[Tensor, dict]:
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
    """
    k = int(first_iteration)
    n = int(num_iterations)
    if k < 0 or n < 1:
        raise ValueError("need first_iteration >= 0 and num_iterations >= 1")
    if isinstance(init_recon, Tensor):
        init_recon = init_recon.detach()
    with torch.no_grad():
        recon, recon_dict = model.recon(sinogram, weights=weights, init_recon=init_recon, max_iterations=k + n,
                                        first_iteration=k, stop_threshold_change_pct=stop_threshold_change_pct,
                                        logfile_path=None, print_logs=False, output_sharded=True)
    return recon, recon_dict


def freeze_regularization(model) -> dict[str, float]:
    """Keep the current sigma_y, sigma_x, sigma_prox in every later `recon` call (auto_regularize_flag=False; by
    default every `recon` call re-estimates them from the sinogram it is given). Returns the frozen values."""
    model.set_params(auto_regularize_flag=False)
    names = ["sigma_y", "sigma_x", "sigma_prox"]
    return {k: float(v) for k, v in zip(names, model.get_params(names))}
