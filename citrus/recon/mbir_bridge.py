"""Bridge between the solver's geometry (``Grid``, ``ConeGeometry``, ``ParallelGeometry``) and MBIRTORCH's models.

It builds the MBIRTORCH model of a scan (``ConeBeamModel`` or ``ParallelBeamModel``), projects and back-projects with
it, runs FDK or the filtered back projection of a parallel beam with the redundancy weights of a range that is not
whole rotations (Parker weights for a short scan, ramps for an overscan), runs warm-started and resumable MBIR
iterations, and projects volumes onto chunks of views. MBIRTORCH is imported
lazily, and the functions return torch tensors on the model device. The coupling is described in
docs/source/design_notes.rst, "MBIRTORCH coupling", and the shared frames in docs/source/conventions.rst.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Sequence, Union

import warnings

import numpy as np
import torch
from torch import Tensor

from ..scatter_solver.geometry import ConeGeometry, Grid, ParallelGeometry

__all__ = [
    "make_cone_model",
    "make_parallel_model",
    "make_model",
    "forward_project",
    "back_project",
    "fdk",
    "direct_reconstruction",
    "scan_range",
    "is_short_scan",
    "redundancy_weights",
    "redundancy_weight_function",
    "short_scan_weights",
    "mbir_iterations",
    "freeze_regularization",
    "ViewChunkProjector",
    "mbirtorch_version",
]

DeviceLike = Union[torch.device, str, None]
ScanGeometry = Union[ConeGeometry, ParallelGeometry]


def mbirtorch_version() -> str:
    """Version of the installed MBIRTORCH.

    Returns:
        str: ``mbirtorch.__version__``.
    """
    import mbirtorch

    return str(mbirtorch.__version__)


def _angles_array(angles) -> np.ndarray:
    """View angles as a flat float64 array.

    Args:
        angles (Sequence[float], ndarray or Tensor): [V] view angles in rad.

    Returns:
        ndarray: [V] view angles in rad.
    """
    if isinstance(angles, Tensor):
        angles = angles.detach().cpu().numpy()
    return np.asarray(angles, dtype=np.float64).reshape(-1)


def make_cone_model(grid: Grid, geometry: ConeGeometry, angles: Union[Sequence[float], np.ndarray, Tensor], *,
                    device: DeviceLike = None, use_ror_mask: bool = False, compile_mode: Optional[str] = None,
                    verbose: int = 0, **params):
    """MBIRTORCH ``ConeBeamModel`` whose projections of a volume on ``grid`` land on the panel of ``geometry``.

    The model copies the distances, pixel pitches and offsets, the panel shape (``use_curved_detector``) and the z
    shifts of the views (``geometry.z_shifts(angles)`` as MBIRTORCH's ``helical_z_shifts``) of ``geometry`` and the
    shape, voxel size, aspects and slice offset of ``grid``, with one view per angle, and is pinned to one device.

    Args:
        grid (Grid): reconstruction grid of the volumes [R, C, S].
        geometry (ConeGeometry): cone-beam geometry of the scan.
        angles (Sequence[float], ndarray or Tensor): [V] view angles in rad.
        device (torch.device or str, optional): the model's single device. Defaults to None (CUDA if available, else
            CPU).
        use_ror_mask (bool, optional): MBIRTORCH's region-of-reconstruction mask; False projects and updates every
            voxel. Defaults to False.
        compile_mode (str, optional): MBIRTORCH's ``compile_mode``. Defaults to None (``"auto"`` on CUDA, ``"off"`` on
            CPU).
        verbose (int, optional): MBIRTORCH's verbosity. Defaults to 0.
        **params: passed to ``ConeBeamModel.set_params`` (e.g. ``snr_db``, ``sharpness``, ``positivity_flag``).

    Returns:
        ConeBeamModel: the MBIRTORCH model.
    """
    import mbirtorch

    device = (torch.device(device) if device is not None
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if compile_mode is None:
        compile_mode = "auto" if device.type == "cuda" else "off"
    angle_array = _angles_array(angles)
    # the constructor arguments of the panel shape and the helical views, left out for the defaults
    view_arguments = {}
    if geometry.use_curved_detector:
        view_arguments["use_curved_detector"] = True
    if geometry.helical_z_shifts:
        view_arguments["helical_z_shifts"] = geometry.z_shifts(angle_array)
    # the constructor checks the cone angle with its default pixel pitch of 1 (a tall panel of small pixels trips
    # it); the check is repeated below with the real pitch
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Cone angle is more than 45 degrees")
        model = mbirtorch.ConeBeamModel((len(angle_array), geometry.num_det_rows, geometry.num_det_channels),
                                        angle_array, source_detector_dist=float(geometry.source_detector_dist),
                                        source_iso_dist=float(geometry.source_iso_dist), compile_mode=compile_mode,
                                        **view_arguments)
    model.set_params(no_warning=True,
                     delta_det_channel=float(geometry.delta_det_channel),
                     delta_det_row=float(geometry.delta_det_row),
                     det_channel_offset=float(geometry.det_channel_offset),
                     det_row_offset=float(geometry.det_row_offset),
                     recon_shape=tuple(int(n) for n in grid.shape), delta_voxel=float(grid.delta_voxel),
                     voxel_row_aspect=float(grid.voxel_row_aspect),
                     voxel_slice_aspect=float(grid.voxel_slice_aspect),
                     recon_slice_offset=float(grid.recon_slice_offset), use_ror_mask=use_ror_mask, verbose=int(verbose))
    if params:
        model.set_params(**params)
    model.verify_valid_params()  # the cone-angle check with the real panel
    model.configure_devices(devices=[str(device)])
    return model


def make_parallel_model(grid: Grid, geometry: ParallelGeometry, angles: Union[Sequence[float], np.ndarray, Tensor], *,
                        device: DeviceLike = None, use_ror_mask: bool = False, compile_mode: Optional[str] = None,
                        verbose: int = 0, **params):
    """MBIRTORCH ``ParallelBeamModel`` whose projections of a volume on ``grid`` land on the panel of ``geometry``.

    MBIRTORCH gives detector row r the slice r of the volume, so the rows of ``geometry`` must be the slices of
    ``grid`` (``ParallelGeometry.from_grid``): as many rows as slices, the row pitch ``delta_voxel``, row r at the z of
    slice r and ``voxel_slice_aspect`` 1. The channel pitch and offset, the voxel size and the row aspect are copied.

    Args:
        grid (Grid): reconstruction grid of the volumes [R, C, S].
        geometry (ParallelGeometry): parallel-beam geometry of the scan.
        angles (Sequence[float], ndarray or Tensor): [V] view angles in rad.
        device (torch.device or str, optional): the model's single device. Defaults to None (CUDA if available, else
            CPU).
        use_ror_mask (bool, optional): MBIRTORCH's region-of-reconstruction mask. Defaults to False.
        compile_mode (str, optional): MBIRTORCH's ``compile_mode``. Defaults to None (``"auto"`` on CUDA, ``"off"`` on
            CPU).
        verbose (int, optional): MBIRTORCH's verbosity. Defaults to 0.
        **params: passed to ``ParallelBeamModel.set_params`` (e.g. ``snr_db``, ``sharpness``, ``positivity_flag``).

    Returns:
        ParallelBeamModel: the MBIRTORCH model.

    Raises:
        ValueError: if the detector rows are not the slices of ``grid``.
    """
    import mbirtorch

    pitch = float(grid.delta_voxel)
    mismatches = []
    if int(geometry.num_det_rows) != int(grid.num_slices):
        mismatches.append(f"{geometry.num_det_rows} rows for {grid.num_slices} slices")
    if float(grid.voxel_slice_aspect) != 1.0:
        mismatches.append(f"voxel_slice_aspect {grid.voxel_slice_aspect} (must be 1)")
    if abs(float(geometry.delta_det_row) - pitch) > 1e-9 * pitch:
        mismatches.append(f"row pitch {geometry.delta_det_row} cm for slices of {pitch} cm")
    if abs(float(geometry.det_row_offset) + float(grid.recon_slice_offset)) > 1e-9 * max(1.0, pitch):
        mismatches.append(f"det_row_offset {geometry.det_row_offset} cm for recon_slice_offset "
                          f"{grid.recon_slice_offset} cm (must be its negative)")
    if mismatches:
        raise ValueError("MBIRTORCH's parallel beam gives detector row r the slice r of the volume; the geometry does "
                         "not match the grid (" + "; ".join(mismatches) + "): build it with ParallelGeometry.from_grid")
    device = (torch.device(device) if device is not None
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if compile_mode is None:
        compile_mode = "auto" if device.type == "cuda" else "off"
    angle_array = _angles_array(angles)
    model = mbirtorch.ParallelBeamModel((len(angle_array), geometry.num_det_rows, geometry.num_det_channels),
                                        angle_array, compile_mode=compile_mode)
    model.set_params(no_warning=True,
                     delta_det_channel=float(geometry.delta_det_channel), delta_det_row=pitch,
                     det_channel_offset=float(geometry.det_channel_offset),
                     recon_shape=tuple(int(n) for n in grid.shape), delta_voxel=pitch,
                     voxel_row_aspect=float(grid.voxel_row_aspect), use_ror_mask=use_ror_mask, verbose=int(verbose))
    if params:
        model.set_params(**params)
    model.verify_valid_params()
    model.configure_devices(devices=[str(device)])
    return model


def make_model(grid: Grid, geometry: ScanGeometry, angles: Union[Sequence[float], np.ndarray, Tensor], **kwargs):
    """MBIRTORCH model of a scan: ``make_parallel_model`` for a ``ParallelGeometry``, else ``make_cone_model``.

    Args:
        grid (Grid): reconstruction grid of the volumes [R, C, S].
        geometry (ConeGeometry or ParallelGeometry): geometry of the scan.
        angles (Sequence[float], ndarray or Tensor): [V] view angles in rad.
        **kwargs: keyword arguments of ``make_cone_model`` / ``make_parallel_model`` (``device``, ``compile_mode``,
            model parameters, ...).

    Returns:
        ConeBeamModel or ParallelBeamModel: the MBIRTORCH model.
    """
    if geometry.is_parallel:
        return make_parallel_model(grid, geometry, angles, **kwargs)
    return make_cone_model(grid, geometry, angles, **kwargs)


def forward_project(model, volume: Tensor) -> Tensor:
    """MBIRTORCH forward projection ``A x`` of a volume.

    Args:
        model (ConeBeamModel or ParallelBeamModel): MBIRTORCH model.
        volume (Tensor): [R, C, S] volume, e.g. attenuation in 1/cm.

    Returns:
        Tensor: [V, Rd, Cd] float32 line integrals on the model device.
    """
    return model.forward_project(volume, output_sharded=True)


def back_project(model, sinogram: Tensor) -> Tensor:
    """MBIRTORCH back projection ``A^T y`` of a sinogram.

    Args:
        model (ConeBeamModel or ParallelBeamModel): MBIRTORCH model.
        sinogram (Tensor): [V, Rd, Cd] sinogram.

    Returns:
        Tensor: [R, C, S] float32 volume on the model device.
    """
    return model.back_project(sinogram, output_sharded=True)


def scan_range(angles) -> float:
    """Angular range of views evenly spaced over [start, end): ``abs(a[-1] - a[0]) V / (V - 1)``.

    Args:
        angles (Sequence[float], ndarray or Tensor): [V] view angles in rad.

    Returns:
        float: range in rad (2 pi for a full rotation, 0 for fewer than two views).
    """
    a = _angles_array(angles)
    if a.size < 2:
        return 0.0
    return float(abs(a[-1] - a[0]) * a.size / (a.size - 1))


def is_short_scan(angles, tolerance: float = 1e-6) -> bool:
    """Whether evenly spaced views cover less than a full rotation (``scan_range`` below 2 pi).

    Args:
        angles (Sequence[float], ndarray or Tensor): [V] view angles in rad.
        tolerance (float, optional): tolerance in rad. Defaults to 1e-6.

    Returns:
        bool: True for a short scan.
    """
    return scan_range(angles) < 2.0 * np.pi - tolerance


_FULL_ROTATION_TOLERANCE = 1e-6  # rad: a range within this of whole rotations needs no weights
_PARKER_SIGN = -1.0  # gamma = -atan(u / SDD): the fan angle with MBIRTORCH's orientation (a line measured at beta
# and channel u is measured again at beta + pi + 2 gamma and channel -u)


def _whole_rotations(arc: float) -> Optional[int]:
    """Number of whole rotations of an angular range, or None when the range is not a whole number of rotations.

    Args:
        arc (float): range in rad.

    Returns:
        int or None: the number of rotations.
    """
    rotations = arc / (2.0 * np.pi)
    if abs(rotations - round(rotations)) * 2.0 * np.pi < _FULL_ROTATION_TOLERANCE:
        return int(round(rotations))
    return None


def redundancy_weight_function(beta, gamma, arc: float) -> np.ndarray:
    """Redundancy weights of a scan over part of a rotation or a number of rotations that is not whole.

    The weights make the measurements of each line add up to one: below a full rotation they are the Parker weights
    in the form of Wesarg, Ebert and Bortfeld (Med. Phys. 29, 2002), above it they ramp over the excess beyond the
    whole rotations at both ends (as in cabouman/mbirtorch PR 30). A parallel beam is the case ``gamma = 0``.

    Args:
        beta (ndarray): position of each view in the scan, from 0 to ``arc``, broadcastable with ``gamma``.
        gamma (ndarray): fan angle of each channel with MBIRTORCH's orientation (``_PARKER_SIGN`` times the angle).
        arc (float): angular range of the scan in rad.

    Returns:
        ndarray: the weights in the broadcast shape of ``beta`` and ``gamma``, without the filter factor.
    """
    beta, gamma = np.broadcast_arrays(np.asarray(beta, dtype=np.float64), np.asarray(gamma, dtype=np.float64))
    weights = np.ones(beta.shape, dtype=np.float64)
    if arc < 2.0 * np.pi - _FULL_ROTATION_TOLERANCE:
        delta = 0.5 * (arc - np.pi)  # the overscan beyond 180 degrees, split between the two ends
        with np.errstate(divide="ignore", invalid="ignore"):
            start = beta < 2.0 * (delta - gamma)
            weights = np.where(start, np.sin(np.pi / 4.0 * beta / np.where(start, delta - gamma, 1.0)) ** 2, weights)
            end = beta >= np.pi - 2.0 * gamma
            end_arg = np.clip(np.pi + 2.0 * delta - beta, 0.0, None)
            weights = np.where(end, np.sin(np.pi / 4.0 * end_arg / np.where(end, delta + gamma, 1.0)) ** 2, weights)
        return weights
    rotations = int(np.floor((arc + _FULL_ROTATION_TOLERANCE) / (2.0 * np.pi)))
    excess = arc - 2.0 * np.pi * rotations
    if excess > _FULL_ROTATION_TOLERANCE:
        start = beta < excess
        weights = np.where(start, np.sin(0.5 * np.pi * beta / excess) ** 2, weights)
        end = beta > 2.0 * np.pi * rotations
        weights = np.where(end, np.sin(0.5 * np.pi * np.clip(arc - beta, 0.0, None) / excess) ** 2, weights)
    return weights / (2.0 * rotations)  # every line is measured twice per rotation


def _channel_fan_angles(geometry: ScanGeometry) -> np.ndarray:
    """Fan angle of each channel with MBIRTORCH's orientation (0 for a parallel beam).

    Args:
        geometry (ConeGeometry or ParallelGeometry): scan geometry.

    Returns:
        ndarray: [Cd] angles in rad.
    """
    if geometry.is_parallel:
        return np.zeros(geometry.num_det_channels)
    u = ((np.arange(geometry.num_det_channels) - (geometry.num_det_channels - 1) / 2.0)
         * float(geometry.delta_det_channel) - float(geometry.det_channel_offset))
    sdd = float(geometry.source_detector_dist)
    return _PARKER_SIGN * (u / sdd if geometry.use_curved_detector else np.arctan(u / sdd))


def _range_too_short(arc: float, gamma: np.ndarray) -> bool:
    """Whether a range is shorter than 180 degrees plus the fan angle, with a RuntimeWarning if it is.

    Args:
        arc (float): range in rad.
        gamma (ndarray): [Cd] fan angles in rad.

    Returns:
        bool: True when the range does not measure every line.
    """
    largest = float(np.abs(gamma).max()) if gamma.size else 0.0
    if 0.5 * (arc - np.pi) < largest - _FULL_ROTATION_TOLERANCE:
        import warnings
        warnings.warn(f"scan range {np.degrees(arc):.1f} deg is shorter than 180 deg plus the fan angle "
                      f"{np.degrees(2 * largest):.1f} deg: no redundancy weighting, the direct reconstruction is "
                      f"incomplete", RuntimeWarning)
        return True
    return False


def redundancy_weights(geometry: ScanGeometry, angles, device: DeviceLike = None) -> Optional[Tensor]:
    """Redundancy weights of a circular cone-beam or a parallel-beam scan for FDK or filtered back projection.

    The weights of ``redundancy_weight_function`` at each view (the centre of its bin over the range) and channel,
    times ``range / pi`` for MBIRTORCH's filter scale ``pi / num_views``. A helical scan gets none (``fdk``).

    Args:
        geometry (ConeGeometry or ParallelGeometry): scan geometry.
        angles (Sequence[float], ndarray or Tensor): [V] view angles in rad, evenly spaced.
        device (torch.device or str, optional): device of the weights. Defaults to None (torch's default device).

    Returns:
        Tensor or None: [V, 1, Cd] float32 weights, or None for a whole number of rotations, a helical scan or a
            range shorter than 180 degrees plus the fan angle (with a RuntimeWarning).
    """
    a = _angles_array(angles)
    V = a.size
    arc = scan_range(a)
    if V < 2 or geometry.is_helical or _whole_rotations(arc) is not None:
        return None
    gamma = _channel_fan_angles(geometry)
    if _range_too_short(arc, gamma):
        return None
    beta = (np.arange(V) + 0.5) * arc / V
    if a[-1] < a[0]:
        beta = beta[::-1].copy()
    w = redundancy_weight_function(beta[:, None], gamma[None, :], arc) * (arc / np.pi)
    device = torch.device(device) if device is not None else None
    return torch.as_tensor(w[:, None, :].astype(np.float32), device=device)


def short_scan_weights(geometry: ScanGeometry, angles, device: DeviceLike = None) -> Optional[Tensor]:
    """Former name of ``redundancy_weights``, kept for existing scripts.

    Args:
        geometry (ConeGeometry or ParallelGeometry): scan geometry.
        angles (Sequence[float], ndarray or Tensor): [V] view angles in rad.
        device (torch.device or str, optional): device of the weights. Defaults to None.

    Returns:
        Tensor or None: as ``redundancy_weights``.
    """
    return redundancy_weights(geometry, angles, device)


def _weights_in_mbirtorch(method) -> bool:
    """Whether an MBIRTORCH direct reconstruction applies the redundancy weights itself (MBIRTORCH with PR 30).

    Args:
        method (callable): ``model.recon_fdk`` or ``model.recon_fbp``.

    Returns:
        bool: True when it takes ``redundancy_weighting``.
    """
    import inspect

    try:
        return "redundancy_weighting" in inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False


def fdk(model, sinogram: Tensor, filter_name: str = "ramp", *, geometry: Optional[ConeGeometry] = None,
        angles=None) -> Tensor:
    """FDK reconstruction of line integrals with MBIRTORCH and the redundancy weights of a circular scan.

    A circular scan over a range that is not whole rotations gets ``redundancy_weights`` (short scans and overscans),
    or MBIRTORCH's own weights when its ``recon_fdk`` applies them. A helical scan keeps MBIRTORCH's FDK without
    redundancy weights, which scales every slice by its view coverage: the per-row helical weights give a worse
    starting volume near the ends of the helix.

    Args:
        model (ConeBeamModel): MBIRTORCH model of the scan.
        sinogram (Tensor): [V, Rd, Cd] line integrals.
        filter_name (str, optional): MBIRTORCH's FDK filter. Defaults to "ramp".
        geometry (ConeGeometry, optional): scan geometry; with ``angles`` a circular scan is weighted. Defaults to None.
        angles (Sequence[float] or ndarray, optional): [V] view angles in rad. Defaults to None.

    Returns:
        Tensor: [R, C, S] volume in 1/cm on the model device.
    """
    weights_in_mbirtorch = _weights_in_mbirtorch(model.recon_fdk)
    if geometry is not None and geometry.is_helical:
        options = {"redundancy_weighting": False} if weights_in_mbirtorch else {}
        return model.recon_fdk(sinogram, filter_name=filter_name, output_sharded=True, **options)
    if geometry is None or angles is None or weights_in_mbirtorch:
        return model.recon_fdk(sinogram, filter_name=filter_name, output_sharded=True)
    w = redundancy_weights(geometry, angles, device=sinogram.device if isinstance(sinogram, Tensor) else None)
    if w is not None:
        sinogram = torch.as_tensor(sinogram, dtype=torch.float32, device=w.device) * w
    return model.recon_fdk(sinogram, filter_name=filter_name, output_sharded=True)


def direct_reconstruction(model, sinogram: Tensor, filter_name: str = "ramp", *,
                          geometry: Optional[ScanGeometry] = None, angles=None) -> Tensor:
    """Direct reconstruction of line integrals: FDK for a cone beam, filtered back projection for a parallel beam.

    Both get the redundancy weights of the scan (``fdk``; ``redundancy_weights`` with fan angle 0 for a parallel
    beam), unless MBIRTORCH applies them itself.

    Args:
        model (ConeBeamModel or ParallelBeamModel): MBIRTORCH model of the scan.
        sinogram (Tensor): [V, Rd, Cd] line integrals.
        filter_name (str, optional): MBIRTORCH's filter. Defaults to "ramp".
        geometry (ConeGeometry or ParallelGeometry, optional): scan geometry; with ``angles`` the scan is weighted.
            Defaults to None (the model decides the kind).
        angles (Sequence[float] or ndarray, optional): [V] view angles in rad. Defaults to None.

    Returns:
        Tensor: [R, C, S] volume in 1/cm on the model device.
    """
    parallel = geometry.is_parallel if geometry is not None else not hasattr(model, "recon_fdk")
    if not parallel:
        return fdk(model, sinogram, filter_name, geometry=geometry, angles=angles)
    if geometry is not None and angles is not None and not _weights_in_mbirtorch(model.recon_fbp):
        w = redundancy_weights(geometry, angles, device=sinogram.device if isinstance(sinogram, Tensor) else None)
        if w is not None:
            sinogram = torch.as_tensor(sinogram, dtype=torch.float32, device=w.device) * w
    return model.recon_fbp(sinogram, filter_name=filter_name, output_sharded=True)


def mbir_iterations(model, sinogram: Tensor, init_recon: Union[Tensor, np.ndarray], first_iteration: int,
                    num_iterations: int = 1, weights: Optional[Tensor] = None,
                    stop_threshold_change_pct: float = 0.0, *, error_sinogram: Optional[Tensor] = None,
                    hessian: Optional[Tensor] = None, return_checkpoint: bool = False):
    """Warm-started VCD iterations of MBIRTORCH from ``init_recon`` at position ``first_iteration`` of the schedule.

    It runs ``model.recon`` with ``max_iterations = first_iteration + num_iterations``; with ``error_sinogram``,
    ``hessian`` or ``return_checkpoint`` it runs MBIRTORCH's engine directly, resuming from the given error sinogram.

    Args:
        model (ConeBeamModel or ParallelBeamModel): MBIRTORCH model.
        sinogram (Tensor): [V, Rd, Cd] line integrals.
        init_recon (Tensor or ndarray): [R, C, S] initial volume in 1/cm; updated in place by the engine path (pass a
            copy).
        first_iteration (int): position k >= 0 of the first iteration in the partition schedule.
        num_iterations (int, optional): number of iterations. Defaults to 1.
        weights (Tensor, optional): [V, Rd, Cd] weights. Defaults to None (all ones).
        stop_threshold_change_pct (float, optional): MBIRTORCH's stopping threshold in percent. Defaults to 0.0.
        error_sinogram (Tensor, optional): [V, Rd, Cd] error sinogram ``sinogram - A init_recon`` of an earlier call,
            trusted and updated in place. Defaults to None (recomputed).
        hessian (Tensor, optional): forward-model Hessian diagonal of an earlier call with the same weights and model.
            Defaults to None (recomputed).
        return_checkpoint (bool, optional): also return the checkpoint ``{"error_sinogram", "fm_hessian"}``. Defaults
            to False.

    Returns:
        tuple: reconstruction [R, C, S] on the model device, MBIRTORCH's recon_dict and, with ``return_checkpoint``,
            the checkpoint dict of the final error sinogram and Hessian.

    Raises:
        ValueError: if ``first_iteration`` < 0, ``num_iterations`` < 1, or ``error_sinogram`` comes without
            ``init_recon``.
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
    """MBIRTORCH's ``TomographyModel.recon`` step by step, passing the error sinogram and Hessian to its engine.

    Args:
        model (ConeBeamModel or ParallelBeamModel): MBIRTORCH model.
        sinogram (Tensor): [V, Rd, Cd] line integrals.
        weights (Tensor or None): [V, Rd, Cd] weights; None for all ones.
        init_recon (Tensor or ndarray): [R, C, S] initial volume, updated in place.
        max_iterations (int): end position in the partition schedule.
        first_iteration (int): start position in the partition schedule.
        stop_threshold_change_pct (float): MBIRTORCH's stopping threshold in percent.
        error_sinogram (Tensor or None): [V, Rd, Cd] error sinogram ``sinogram - A init_recon``; None recomputes it.
        hessian (Tensor or None): forward-model Hessian diagonal; None recomputes it.

    Returns:
        (Tensor, dict, dict): reconstruction [R, C, S], recon_dict as ``recon`` builds it, and the checkpoint
            ``{"error_sinogram", "fm_hessian"}``.
    """
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
    for handler in list(model.logger.handlers):
        handler.flush()
    model.close_log_file()
    recon_dict = model.get_recon_dict(recon_params, notes=f"Reconstruction completed: {datetime.datetime.now()}\n\n")
    return recon, recon_dict, checkpoint


def freeze_regularization(model) -> dict[str, float]:
    """Freeze the model's current regularization for every later ``recon`` call (``auto_regularize_flag`` False).

    Args:
        model (ConeBeamModel or ParallelBeamModel): MBIRTORCH model.

    Returns:
        dict[str, float]: the frozen ``sigma_y``, ``sigma_x`` and ``sigma_prox``.
    """
    model.set_params(auto_regularize_flag=False)
    names = ["sigma_y", "sigma_x", "sigma_prox"]
    return {k: float(v) for k, v in zip(names, model.get_params(names))}


class ViewChunkProjector:
    """MBIRTORCH forward projections of volumes on ``grid`` onto chunks of consecutive views of a scan.

    The full view range uses ``model`` when it is given; any other range uses a sub-model over its angles
    (``make_model``), kept in a least-recently-used cache.

    Args:
        grid (Grid): reconstruction grid.
        geometry (ConeGeometry or ParallelGeometry): geometry of the scan.
        angles (Sequence[float], ndarray or Tensor): [V] view angles in rad.
        model (ConeBeamModel or ParallelBeamModel, optional): the model over all angles. Defaults to None.
        device (torch.device or str, optional): device of the sub-models. Defaults to None (CUDA if available).
        max_models (int, optional): number of sub-models kept. Defaults to 8.
    """

    def __init__(self, grid: Grid, geometry: ConeGeometry, angles, *, model=None, device: DeviceLike = None,
                 max_models: int = 8) -> None:
        self.grid = grid
        self.geometry = geometry
        self.angles = _angles_array(angles)
        self.model = model
        self.device = device
        self.max_models = max(1, int(max_models))
        self._models: "OrderedDict[tuple[int, int], object]" = OrderedDict()

    @property
    def num_views(self) -> int:
        """Number of views V."""
        return int(self.angles.size)

    def chunks(self, views_per_chunk: int) -> list[tuple[int, int]]:
        """Ranges (start, stop) covering the views [0, V) in runs of ``views_per_chunk`` views.

        Args:
            views_per_chunk (int): views per chunk (at least 1); the last chunk may be shorter.

        Returns:
            list[tuple[int, int]]: the (start, stop) ranges.
        """
        n = max(1, int(views_per_chunk))
        return [(a, min(a + n, self.num_views)) for a in range(0, self.num_views, n)]

    def model_for(self, start: int, stop: int):
        """MBIRTORCH model of the views [start, stop).

        Args:
            start (int): first view.
            stop (int): end view (excluded).

        Returns:
            ConeBeamModel or ParallelBeamModel: ``model`` for all views, else the cached or new sub-model.

        Raises:
            ValueError: if the range is empty or outside [0, V).
        """
        start, stop = int(start), int(stop)
        if not 0 <= start < stop <= self.num_views:
            raise ValueError(f"view range [{start}, {stop}) outside [0, {self.num_views})")
        if (start, stop) == (0, self.num_views) and self.model is not None:
            return self.model
        key = (start, stop)
        model = self._models.get(key)
        if model is None:
            model = make_model(self.grid, self.geometry, self.angles[start:stop], device=self.device)
            self._models[key] = model
            while len(self._models) > self.max_models:
                self._models.popitem(last=False)
        else:
            self._models.move_to_end(key)
        return model

    def project(self, volume: Tensor, start: int = 0, stop: Optional[int] = None) -> Tensor:
        """Forward projection of a volume onto the views [start, stop).

        Args:
            volume (Tensor): [R, C, S] volume.
            start (int, optional): first view. Defaults to 0.
            stop (int, optional): end view (excluded). Defaults to None (all views).

        Returns:
            Tensor: [stop - start, Rd, Cd] float32 line integrals on the model device.
        """
        stop = self.num_views if stop is None else int(stop)
        return forward_project(self.model_for(start, stop), volume)
