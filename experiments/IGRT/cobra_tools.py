"""Shared code of the COBRA2026 IGRT experiments (COBRA_TrueBeam.py, COBRA_Halcyon.py).

The scripts read as five sections: parameters, data, preprocessing, reconstruction and figure. This module holds what
they share: the downloader of the COBRA2026 cases, the readers of the Varian files (Scan.xml, geometry.xml, .xim
images, the calibration XML files), the preprocessing of the counts and flat fields, the couch model, the half-fan
weights and the figure.

Frames: a COBRA2026 case is in LPS with the origin at the isocenter. RTK puts the source at +z for gantry angle 0 and
rotates it toward +x; CITRUS puts it at +y for view angle 0 and rotates it toward +x, so CITRUS's (x, y, z) are LPS
(L, A, I) and a CITRUS volume [rows, cols, slices] indexes (anterior, left, inferior). The view angle is the RTK
gantry angle (the kV source angle). The air scans carry the treatment gantry angle, 90 degrees ahead of it. RTK's
ProjectionOffsetX/Y are the panel centre's u and v, which CITRUS subtracts, so the offsets change sign.
"""

from __future__ import annotations

import glob
import json
import os
import re
import struct
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import SimpleITK as sitk
import yaml
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.ndimage import shift as shift_image

from citrus.scatter_solver import ConeGeometry, Grid

# ======================================================================================================================
# Download
# ======================================================================================================================

API = "https://tomoradio-warehouse.creatis.insa-lyon.fr/api/v1"   # the public Girder API of the COBRA2026 data
COLLECTION_ID = "696f75f74c0b0d3d4bc7019d"                         # the COBRA2026 collection
SPLIT = "COBRA2026_Train"
# the case files the scripts read; Calibrations/ is downloaded whole
CASE_FILES = ("projections.mha", "geometry.xml", "Scan.xml", "metadata.yaml", "cbct_clinical.mha", "ct_def_masked.mha",
              "fov_cbct.mha", "fov_cbct_nocouch.mha")
# case files left out of the download (unused here): the simulated projections, the CT before registration, the
# full-field deformed CT and the dataset's FDK, about 1.6 GB per case
SKIPPED_FILES = ("projections_simulated.mha", "ct_original.mha", "ct_def.mha", "cbct_rtk.mha")


def missing_case_files(case_dir: str) -> list[str]:
    """The files the scripts need that a case folder lacks.

    Args:
        case_dir (str): case folder, e.g. data/TrueBeam/G008.

    Returns:
        list[str]: the missing files and calibrations (empty when the case is complete).
    """
    paths = {name: os.path.join(case_dir, name) for name in CASE_FILES}
    missing = [name for name, path in paths.items() if not os.path.isfile(path) or os.path.getsize(path) == 0]
    calibrations = os.path.join(case_dir, "Calibrations")
    for label, pattern in (("air scans", os.path.join("AIR*Bowtie*", "Current", "FilterBowtie_*.xim")),
                           ("grid calibration", os.path.join("SC*", "*", "Calibration.xml")),
                           ("couch calibration", os.path.join("COUCH", "*", "Calibration.xml"))):
        if not glob.glob(os.path.join(calibrations, pattern)):
            missing.append(f"Calibrations ({label})")
    return missing


def _get_json(endpoint: str, **params):
    with urllib.request.urlopen(f"{API}/{endpoint}?{urllib.parse.urlencode(params)}", timeout=60) as response:
        return json.load(response)


def _list_all(endpoint: str, **params) -> list:
    """Every entry of a Girder listing, page by page."""
    entries: list = []
    while True:
        page = _get_json(endpoint, limit=100, offset=len(entries), sort="_id", sortdir=1, **params)
        entries.extend(page)
        if len(page) < 100:
            return entries


def _child_folder(parent_id: str, parent_type: str, name: str) -> dict:
    folders = _get_json("folder", parentId=parent_id, parentType=parent_type, name=name, limit=2)
    if len(folders) != 1:
        raise ValueError(f"folder {name!r} not found once under {parent_id}")
    return folders[0]


def _safe_name(name: str) -> str:
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError(f"unexpected file name on the server: {name!r}")
    return name


def remote_case_files(case_id: str) -> list[tuple[str, str, int]]:
    """The files of a COBRA2026 training case on the server.

    Args:
        case_id (str): case name, e.g. "G008" (its first letter is the centre).

    Returns:
        list[tuple[str, str, int]]: (path relative to the case folder, Girder file id, size in bytes) of every file.
    """
    train = _child_folder(COLLECTION_ID, "collection", SPLIT)
    centre = _child_folder(train["_id"], "folder", case_id[0])
    case = _child_folder(centre["_id"], "folder", case_id)

    def walk(folder_id: str, prefix: str) -> list[tuple[str, str, int]]:
        out = []
        for item in _list_all("item", folderId=folder_id):
            files = _list_all(f"item/{item['_id']}/files")
            # a single file named like its item sits directly in the folder
            item_prefix = prefix if (len(files) == 1 and files[0]["name"] == item["name"]) else (
                prefix + _safe_name(item["name"]) + "/")
            out += [(item_prefix + _safe_name(f["name"]), f["_id"], int(f["size"])) for f in files]
        for child in _list_all("folder", parentType="folder", parentId=folder_id):
            out += walk(child["_id"], prefix + _safe_name(child["name"]) + "/")
        return out

    return walk(case["_id"], "")


def download_file(file_id: str, size: int, path: str) -> None:
    """Download one file of the server to a path, through a .part file, and check its size.

    Args:
        file_id (str): Girder file id.
        size (int): expected size in bytes.
        path (str): destination.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    partial = path + ".part"
    received = 0
    with urllib.request.urlopen(f"{API}/file/{file_id}/download", timeout=120) as response, open(partial, "wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
            received += len(chunk)
    if received != size:
        os.remove(partial)
        raise IOError(f"{path}: received {received} bytes of {size}")
    os.replace(partial, path)


def download_case(case_id: str, scanner_dir: str) -> str:
    """Download a COBRA2026 case into scanner_dir/case_id unless the folder already holds what the scripts need.

    Files already present with the server's size are kept; the files in ``SKIPPED_FILES`` are not downloaded.

    Args:
        case_id (str): case name, e.g. "G008".
        scanner_dir (str): folder of the scanner's cases, e.g. data/TrueBeam.

    Returns:
        str: the case folder.
    """
    case_dir = os.path.join(scanner_dir, case_id)
    missing = missing_case_files(case_dir)
    if not missing:
        print(f"[{case_id}] found in {case_dir}", flush=True)
        return case_dir
    print(f"[{case_id}] missing {missing}: downloading from {API} ({SPLIT})", flush=True)
    files = [f for f in remote_case_files(case_id) if f[0] not in SKIPPED_FILES]
    total = sum(size for _, _, size in files)
    done, t0 = 0, time.time()
    for relative, file_id, size in files:
        path = os.path.join(case_dir, *relative.split("/"))
        if not (os.path.isfile(path) and os.path.getsize(path) == size):
            download_file(file_id, size, path)
        done += size
        print(f"\r[{case_id}] {done / 1e6:.0f} of {total / 1e6:.0f} MB ({time.time() - t0:.0f} s)", end="", flush=True)
    print(flush=True)
    missing = missing_case_files(case_dir)
    if missing:
        raise FileNotFoundError(f"{case_dir} still lacks {missing} after the download")
    return case_dir


# ======================================================================================================================
# Varian files
# ======================================================================================================================

def _decode_xml(path: str) -> str:
    raw = open(path, "rb").read()
    return raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig", "replace")


def read_scan_xml(case_dir: str) -> dict[str, str]:
    """The text of every element of a Varian Scan.xml, by tag (the first element of each tag).

    Args:
        case_dir (str): case folder.

    Returns:
        dict[str, str]: tag -> text.
    """
    root = ET.fromstring(_decode_xml(os.path.join(case_dir, "Scan.xml")))
    values: dict[str, str] = {}
    for element in root.iter():
        tag = element.tag.split("}", 1)[-1]
        if tag not in values and element.text is not None and element.text.strip():
            values[tag] = element.text.strip()
    return values


def read_rtk_geometry(path: str) -> dict:
    """An RTK ThreeDCircularProjectionGeometry XML file.

    Args:
        path (str): geometry.xml.

    Returns:
        dict: ``sid_mm``, ``sdd_mm``, ``radius_mm`` and per view ``gantry_deg``, ``offset_x_mm``, ``offset_y_mm``.
    """
    root = ET.parse(path).getroot()
    views = list(root.iter("Projection"))

    def scalar(name, default=0.0):
        node = root.find(name)
        return float(node.text) if node is not None else default

    def per_view(name):
        return np.array([float(p.find(name).text) if p.find(name) is not None else 0.0 for p in views])

    return {"sid_mm": scalar("SourceToIsocenterDistance"), "sdd_mm": scalar("SourceToDetectorDistance"),
            "radius_mm": scalar("RadiusCylindricalDetector"), "gantry_deg": per_view("GantryAngle"),
            "offset_x_mm": per_view("ProjectionOffsetX"), "offset_y_mm": per_view("ProjectionOffsetY")}


def read_xim(path: str) -> tuple[np.ndarray, dict]:
    """A Varian .xim image (the layout read by RTK's XimImageIO: a lookup table of byte counts and the pixel
    differences to the left, upper and upper-left neighbours) and its properties.

    Args:
        path (str): .xim file.

    Returns:
        (ndarray, dict): image [rows, cols] float32 and the properties (e.g. ``GantryRtn``, ``KVMilliAmperes``).
    """
    data = open(path, "rb").read()
    _, cols, rows, _, bytes_per_pixel, compressed = struct.unpack_from("<6i", data, 8)
    pos = 32
    n = rows * cols
    if compressed == 1:
        lut_size = struct.unpack_from("<i", data, pos)[0]
        lut = np.frombuffer(data, np.uint8, lut_size, pos + 4)
        pos += 4 + lut_size
        buffer_size = struct.unpack_from("<i", data, pos)[0]
        start = pos + 4
        first = np.frombuffer(data, "<i4", cols + 1, start).astype(np.int64)   # buf[0..cols], buf[0] a sentinel
        codes = np.stack([(lut >> s) & 3 for s in (0, 2, 4, 6)], axis=1).ravel()
        if (codes == 3).any():
            raise ValueError(f"{path}: 8-byte pixel differences are not supported")
        sizes = np.array([1, 2, 4], np.int64)[codes][: n - cols]   # one difference per pixel after the first row
        raw = np.frombuffer(data, np.uint8, start + buffer_size - (start + 4 * (cols + 1)), start + 4 * (cols + 1))
        raw = np.concatenate([raw, np.zeros(4, np.uint8)]).astype(np.int64)
        offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        diff = np.empty(sizes.size, np.int64)
        for size in (1, 2, 4):
            o = offsets[sizes == size]
            value = sum(raw[o + b] << (8 * b) for b in range(size))
            diff[sizes == size] = np.where(value >= 1 << (8 * size - 1), value - (1 << (8 * size)), value)
        # buf[k] = d_k + buf[k-1] + buf[k-cols] - buf[k-cols-1]: e_k = buf[k] - buf[k-cols] is a running sum of d,
        # and every column a running sum of e
        e = (first[cols] - first[0]) + np.cumsum(diff)
        image = np.concatenate([first[None, 1:], first[None, 1:] + np.cumsum(e.reshape(rows - 1, cols), axis=0)])
        pos = start + buffer_size
        pos += 4 + 0   # uncompressed buffer size
    else:
        size = struct.unpack_from("<i", data, pos)[0]
        dtype = {1: "u1", 2: "<u2", 4: "<i4"}[bytes_per_pixel]
        image = np.frombuffer(data, dtype, n, pos + 4).reshape(rows, cols).astype(np.int64)
        pos += 4 + size
    bins = struct.unpack_from("<i", data, pos)[0]
    pos += 4 + 4 * bins
    count = struct.unpack_from("<i", data, pos)[0]
    pos += 4
    properties: dict = {}
    for _ in range(count):
        length = struct.unpack_from("<i", data, pos)[0]
        name = data[pos + 4: pos + 4 + length].split(b"\x00", 1)[0].decode("ascii", "ignore")
        kind = struct.unpack_from("<i", data, pos + 4 + length)[0]
        pos += 8 + length
        if kind == 0:
            properties[name] = struct.unpack_from("<i", data, pos)[0]
            pos += 4
        elif kind == 1:
            properties[name] = struct.unpack_from("<d", data, pos)[0]
            pos += 8
        elif kind in (2, 4, 5):
            size = struct.unpack_from("<i", data, pos)[0]
            chunk = data[pos + 4: pos + 4 + size]
            properties[name] = (chunk.split(b"\x00", 1)[0].decode("ascii", "replace") if kind == 2 else
                                list(struct.unpack(f"<{size // 8}d" if kind == 4 else f"<{size // 4}I", chunk)))
            pos += 4 + size
        else:
            raise ValueError(f"{path}: property {name!r} of unknown type {kind}")
    return image.reshape(rows, cols).astype(np.float32), properties


def grid_calibration(case_dir: str) -> tuple[float, str]:
    """The anti-scatter grid's ratio and lamella orientation from the scatter calibration (Calibrations/SC*).

    Args:
        case_dir (str): case folder.

    Returns:
        (float, str): grid ratio and orientation (e.g. 10.0, "TransAxialLamella").
    """
    text = _decode_xml(sorted(glob.glob(os.path.join(case_dir, "Calibrations", "SC*", "*", "Calibration.xml")))[0])
    orientation = re.search(r"<GridOrientation>(\w+)</GridOrientation>", text)
    return float(re.search(r"<GridRatio>([\d.]+)</GridRatio>", text).group(1)), (
        orientation.group(1) if orientation else "TransAxialLamella")


def couch_contours(case_dir: str, name: str) -> tuple[np.ndarray, np.ndarray]:
    """The outer (CouchSurface) and inner (CouchInterior) contours of a couch top of the couch calibration.

    Args:
        case_dir (str): case folder.
        name (str): couch top, e.g. "IGRT" (Scan.xml CouchTopInstalled).

    Returns:
        (ndarray, ndarray): [N, 2] points (X, Y) in mm of each contour, Y = 0 at the top surface, increasing downward.
    """
    text = _decode_xml(glob.glob(os.path.join(case_dir, "Calibrations", "COUCH", "*", "Calibration.xml"))[0])
    top = next(t for t in text.split("<CouchTop>")[1:] if f"<Name>{name}</Name>" in t)

    def contour(structure):
        part = next(c for c in top.split("<CutStructure>")[1:] if f"<StructureName>{structure}</StructureName>" in c)
        return np.array([[float(a), float(b)] for a, b in re.findall(r'<Point X="([-\d.]+)" Y="([-\d.]+)"', part)])

    return contour("CouchSurface"), contour("CouchInterior")


# ======================================================================================================================
# Case parameters read from the metadata
# ======================================================================================================================

@dataclass
class CaseInfo:
    """The parameters of a case read from its metadata (metadata.yaml, Scan.xml, Calibrations).

    Attributes:
        case_id (str): case name.
        case_dir (str): case folder.
        kvp (float): tube voltage in kV.
        tube_current_ma (float): tube current in mA.
        pulse_ms (float): pulse length in ms.
        scintillator_cm (float): CsI thickness in cm.
        grid_ratio (float): anti-scatter grid ratio.
        grid_orientation (str): orientation of the grid's lamellae.
        bowtie (str): bowtie filter name.
        couch_top (str): couch top installed (Scan.xml CouchTopInstalled).
        couch_lateral_mm (float): lateral couch position (Scan.xml CouchLat), x of the couch model's centre in mm.
        couch_vertical_mm (float): vertical couch position (Scan.xml CouchVrt) in mm.
        blades_mm (tuple): axial collimator blades (BladeY1, BladeY2) at the isocenter in mm.
        metadata (dict): the ``cbct`` entries of metadata.yaml.
    """

    case_id: str
    case_dir: str
    kvp: float
    tube_current_ma: float
    pulse_ms: float
    scintillator_cm: float
    grid_ratio: float
    grid_orientation: str
    bowtie: str
    couch_top: str
    couch_lateral_mm: float
    couch_vertical_mm: float
    blades_mm: tuple[float, float]
    metadata: dict = field(default_factory=dict)


def read_case_info(case_dir: str) -> CaseInfo:
    """The parameters of a Varian case that its metadata gives.

    Args:
        case_dir (str): case folder.

    Returns:
        CaseInfo: the parameters.
    """
    with open(os.path.join(case_dir, "metadata.yaml"), encoding="utf-8") as f:
        cbct = yaml.safe_load(f)["cbct"]
    if str(cbct["Manufacturer"]).lower() != "varian":
        raise ValueError(f"{case_dir} is not a Varian scan")
    scan = read_scan_xml(case_dir)
    ratio, orientation = grid_calibration(case_dir)
    thickness = re.search(r"(\d+)\s*um", str(cbct.get("Scintillator", "")))
    return CaseInfo(case_id=os.path.basename(os.path.normpath(case_dir)), case_dir=case_dir,
                    kvp=float(scan["Voltage"]), tube_current_ma=float(scan["Current"]),
                    pulse_ms=float(scan["PulseLength"]),
                    scintillator_cm=float(thickness.group(1)) * 1e-4 if thickness else 0.06,
                    grid_ratio=ratio, grid_orientation=orientation, bowtie=str(cbct.get("Bowtie", "")),
                    couch_top=scan.get("CouchTopInstalled", "IGRT"), couch_lateral_mm=float(scan["CouchLat"]),
                    couch_vertical_mm=float(scan["CouchVrt"]),
                    blades_mm=(float(scan["BladeY1"]), float(scan["BladeY2"])), metadata=cbct)


# ======================================================================================================================
# Counts and flat fields
# ======================================================================================================================

@dataclass
class CobraScan:
    """A preprocessed scan in CITRUS conventions (cm, rad).

    Attributes:
        case_id (str): case name.
        counts (ndarray): [V, Rd, Cd] float32 measured counts, drift corrected and binned.
        flat_field (ndarray): [V, Rd, Cd] float32 air scans at each view's angle, scaled to the scan's mAs.
        angles (ndarray): [V] view angles in rad, ascending.
        geometry (ConeGeometry): geometry of the binned panel.
        dead (ndarray or None): [V, Rd, Cd] bool pixels with a zero or non-finite count or flat field (set to the
            view's median flat field; they need weight 0).
        frames (ndarray): [V] index of each view in the acquisition (the drift model's frame index).
        lit_rows (tuple or None): raw panel rows (first, end) kept inside the axial blades, None when all are kept.
        half_fan (bool): whether the panel is offset (half fan).
        off_focal (tuple or None): (distances in cm past the lit rows' edges, emission there over the lit level).
    """

    case_id: str
    counts: np.ndarray
    flat_field: np.ndarray
    angles: np.ndarray
    geometry: ConeGeometry
    dead: Optional[np.ndarray]
    frames: np.ndarray
    lit_rows: Optional[tuple[int, int]]
    half_fan: bool
    off_focal: Optional[tuple[np.ndarray, np.ndarray]] = None


def bin_image(image: np.ndarray, row_bin: int, channel_bin: int) -> np.ndarray:
    """Mean over row_bin x channel_bin blocks of the last two axes (a remainder at the end is dropped).

    Args:
        image (ndarray): [..., rows, cols] image.
        row_bin (int): rows per bin.
        channel_bin (int): channels per bin.

    Returns:
        ndarray: [..., rows // row_bin, cols // channel_bin] binned image.
    """
    rows, cols = image.shape[-2] // row_bin * row_bin, image.shape[-1] // channel_bin * channel_bin
    image = image[..., :rows, :cols]
    return image.reshape(image.shape[:-2] + (rows // row_bin, row_bin, cols // channel_bin, channel_bin)).mean(
        axis=(-3, -1))


def air_scans(case_dir: str, clockwise: bool) -> tuple[np.ndarray, np.ndarray, float]:
    """The per-angle bowtie air scans of the acquisition (Calibrations/AIR*Bowtie*/Current).

    Args:
        case_dir (str): case folder.
        clockwise (bool): gantry direction of the scan (the CW or CC set when both exist).

    Returns:
        (ndarray, ndarray, float): images [N, rows, cols], their treatment gantry angles in degrees and their mAs.
    """
    folder = sorted(d for d in glob.glob(os.path.join(case_dir, "Calibrations", "AIR*Bowtie*", "Current"))
                    if os.path.isdir(d))[0]
    files = sorted(glob.glob(os.path.join(folder, f"FilterBowtie_{'CW' if clockwise else 'CC'}_*.xim")))
    if not files:
        files = sorted(f for f in glob.glob(os.path.join(folder, "FilterBowtie_*.xim")) if re.search(r"_\d+\.xim$", f))
    images, angles, mas = [], [], []
    for f in files:
        image, properties = read_xim(f)
        images.append(image)
        angles.append(float(properties["GantryRtn"]))
        mas.append(float(properties["KVMilliAmperes"]) * float(properties["KVMilliSeconds"]))
    if len(set(round(m, 6) for m in mas)) != 1:
        raise ValueError(f"air scans with different mAs: {mas}")
    return np.stack(images), np.array(angles), mas[0]


def lit_rows(info: CaseInfo, num_rows: int, pitch_v_mm: float, sid_mm: float, sdd_mm: float,
             margin: int = 4) -> Optional[tuple[int, int]]:
    """The panel rows inside the axial collimator blades (Scan.xml BladeY1/Y2), less a margin for the penumbra.

    The rows under the blades read blade leakage while the air scans were taken with the blades open, so they are
    left out. The row index runs opposite to the blade coordinate.

    Args:
        info (CaseInfo): case parameters.
        num_rows (int): panel rows.
        pitch_v_mm (float): row pitch in mm.
        sid_mm (float): source-isocenter distance in mm.
        sdd_mm (float): source-detector distance in mm.
        margin (int): rows left out inside each blade edge. Defaults to 4.

    Returns:
        tuple or None: (first, end) rows, None when the blades clear the panel.
    """
    centre = (num_rows - 1) / 2
    rows = sorted(centre - y * sdd_mm / sid_mm / pitch_v_mm for y in info.blades_mm)
    first, last = int(np.ceil(rows[0])) + margin, int(np.floor(rows[1])) - margin
    if first <= 0 and last >= num_rows - 1:
        return None
    return max(first, 0), min(last, num_rows - 1) + 1


def blade_field_margin_cm(info: CaseInfo, geometry: ConeGeometry) -> float:
    """How far past the panel's row edges the axial blades open, in cm at the panel (0 when they do not).

    Args:
        info (CaseInfo): case parameters.
        geometry (ConeGeometry): scan geometry.

    Returns:
        float: the margin in cm.
    """
    blade_cm = min(abs(y) for y in info.blades_mm) / 10.0 * geometry.source_detector_dist / geometry.source_iso_dist
    return max(0.0, blade_cm - 0.5 * geometry.num_det_rows * geometry.delta_det_row)


def drift_correction(frame: np.ndarray, k: int, drift: dict) -> np.ndarray:
    """One frame of counts with the panel's global drift removed.

    The model is y = (1 + B (1 - e^{-k / tau_g})) x + A (1 - e^{-k / tau_o}) on every pixel, k the frame's index in
    the acquisition (every frame counted); it is inverted for x.

    Args:
        frame (ndarray): [rows, cols] counts y of the frame.
        k (int): frame index.
        drift (dict): ``gain_buildup`` B, ``gain_time_constant_frames`` tau_g, ``offset_buildup_counts`` A and
            ``offset_time_constant_frames`` tau_o.

    Returns:
        ndarray: [rows, cols] counts x.
    """
    offset = drift["offset_buildup_counts"] * (1.0 - np.exp(-k / drift["offset_time_constant_frames"]))
    gain = 1.0 + drift["gain_buildup"] * (1.0 - np.exp(-k / drift["gain_time_constant_frames"]))
    return (frame - offset) / gain


def load_scan(info: CaseInfo, row_bin: int, channel_bin: int, view_stride: int = 1, drift: Optional[dict] = None,
              crop_to_lit_rows: bool = True, off_focal_table: bool = False) -> CobraScan:
    """Read and preprocess the projections and air scans of a Varian case.

    Every kept view is, in order: drift corrected (``drift_correction``, on the raw frame index), paired with the air
    scans interpolated to its angle (the treatment gantry angle is the view angle + 90 degrees) and scaled to the
    scan's mAs, shifted to the mean panel offset (the flex map), cropped to the lit rows and binned. Views whose angle
    repeats the previous one (the gantry starting up) are dropped before every view_stride-th view is kept.

    Args:
        info (CaseInfo): case parameters.
        row_bin (int): rows per binned row.
        channel_bin (int): channels per binned channel.
        view_stride (int): keep every n-th view. Defaults to 1.
        drift (dict, optional): global drift model of the panel (``drift_correction``). Defaults to None (none).
        crop_to_lit_rows (bool): keep only the rows inside the axial blades. Defaults to True.
        off_focal_table (bool): also measure the off-focal emission past the lit rows (``off_focal_profile``) from
            the air columns of the uncropped panel. Defaults to False.

    Returns:
        CobraScan: the scan.
    """
    case_dir = info.case_dir
    image = sitk.ReadImage(os.path.join(case_dir, "projections.mha"))
    raw = sitk.GetArrayViewFromImage(image)                       # [V, rows, cols] uint16
    pitch_u, pitch_v = image.GetSpacing()[0], image.GetSpacing()[1]
    geom = read_rtk_geometry(os.path.join(case_dir, "geometry.xml"))
    V, Rd, Cd = raw.shape
    if len(geom["gantry_deg"]) != V or geom["radius_mm"] != 0.0:
        raise ValueError("geometry.xml does not match the projections (or a curved panel)")
    unwrapped = np.degrees(np.unwrap(np.radians(geom["gantry_deg"])))
    clockwise = bool(unwrapped[-1] > unwrapped[0])

    # views in angle order, the repeated angles of a starting gantry dropped, then every view_stride-th
    order = np.argsort(unwrapped)
    step, last, keep = np.median(np.diff(unwrapped[order])), unwrapped[order[0]], [order[0]]
    for k in order[1:]:
        if unwrapped[k] - last >= 0.25 * step:
            keep.append(k)
            last = unwrapped[k]
    frames = np.array(keep[::view_stride])

    # air scans: per view, the two neighbours in gantry angle blended linearly
    air, air_deg, air_mas = air_scans(case_dir, clockwise)
    if air.shape[1:] != (Rd, Cd):
        raise ValueError(f"air scans {air.shape[1:]} do not match the projections {(Rd, Cd)}")
    sorted_air = np.argsort(air_deg)
    air, air_deg = air[sorted_air], air_deg[sorted_air]
    mas_scale = info.tube_current_ma * info.pulse_ms / air_mas

    lit = lit_rows(info, Rd, pitch_v, geom["sid_mm"], geom["sdd_mm"]) if crop_to_lit_rows else None
    r0, r1 = (0, Rd) if lit is None else (lit[0], lit[1] - (lit[1] - lit[0]) % row_bin)
    mean_x, mean_y = float(geom["offset_x_mm"].mean()), float(geom["offset_y_mm"].mean())
    counts = np.empty((len(frames), (r1 - r0) // row_bin, Cd // channel_bin), np.float32)
    flat = np.empty_like(counts)
    profiles = []
    for i, k in enumerate(frames):
        y = raw[k].astype(np.float32)
        if drift is not None:
            y = drift_correction(y, int(k), drift)
        g = (unwrapped[k] + 90.0) % 360.0
        j = int(np.searchsorted(air_deg, g, side="right"))
        lower, upper = (j - 1) % len(air_deg), j % len(air_deg)
        span = (air_deg[upper] - air_deg[lower]) % 360.0
        w = ((g - air_deg[lower]) % 360.0) / span if span > 0 else 0.0
        i0 = ((1.0 - w) * air[lower] + w * air[upper]) * mas_scale
        du = (geom["offset_x_mm"][k] - mean_x) / pitch_u
        dv = (geom["offset_y_mm"][k] - mean_y) / pitch_v
        if abs(du) > 1e-3 or abs(dv) > 1e-3:
            y = shift_image(y, (dv, du), order=1, mode="nearest")
            i0 = shift_image(i0, (dv, du), order=1, mode="nearest")
        if off_focal_table:
            profile = air_column_profile(bin_image(y, row_bin, channel_bin) / bin_image(i0, row_bin, channel_bin),
                                         r0 // row_bin, (r1 - r0) // row_bin)
            if profile is not None:
                profiles.append(profile)
        counts[i] = bin_image(y[r0:r1], row_bin, channel_bin)
        flat[i] = bin_image(i0[r0:r1], row_bin, channel_bin)
    del raw, image

    dead = ~np.isfinite(flat) | (flat <= 0) | ~np.isfinite(counts) | (counts <= 0)
    if dead.any():
        for v in np.flatnonzero(dead.any(axis=(1, 2))):
            open_beam = np.median(flat[v][~dead[v]])
            counts[v][dead[v]] = open_beam
            flat[v][dead[v]] = open_beam
        print(f"[{info.case_id}] {int(dead.sum())} pixels with a zero or non-finite count or flat field", flush=True)
    else:
        dead = None

    # the kept rows' centre is the panel's centre
    centre_y = mean_y + ((r0 + r1 - 1) / 2 - (Rd - 1) / 2) * pitch_v
    geometry = ConeGeometry(source_iso_dist=geom["sid_mm"] / 10.0, source_detector_dist=geom["sdd_mm"] / 10.0,
                            num_det_rows=counts.shape[1], num_det_channels=counts.shape[2],
                            delta_det_row=pitch_v * row_bin / 10.0, delta_det_channel=pitch_u * channel_bin / 10.0,
                            det_row_offset=-centre_y / 10.0, det_channel_offset=-mean_x / 10.0)
    off_focal = None
    if off_focal_table:
        off_focal = off_focal_from_profiles(profiles, r0 // row_bin, (r1 - r0) // row_bin, pitch_v * row_bin / 10.0)
    return CobraScan(case_id=info.case_id, counts=counts, flat_field=flat, angles=np.radians(unwrapped[frames]),
                     geometry=geometry, dead=dead, frames=frames, lit_rows=lit,
                     half_fan=abs(mean_x) > 0.25 * Cd * pitch_u, off_focal=off_focal)


def air_column_profile(transmission: np.ndarray, first_lit_row: int, num_lit_rows: int) -> Optional[np.ndarray]:
    """One view's transmission per row of the whole panel, the median over its air columns.

    The air columns are those whose lit rows (5 rows in from each edge) read more than 0.9 of the flat field.

    Args:
        transmission (ndarray): [rows, cols] counts over flat field of the uncropped, binned panel.
        first_lit_row (int): first lit row of the binned panel.
        num_lit_rows (int): number of lit rows.

    Returns:
        ndarray or None: [rows] profile, None when fewer than 10 air columns.
    """
    air = np.median(transmission[first_lit_row + 5: first_lit_row + num_lit_rows - 5], axis=0) > 0.9
    return np.median(transmission[:, air], axis=1) if air.sum() >= 10 else None


def off_focal_from_profiles(profiles: list, first_lit_row: int, num_lit_rows: int,
                            row_pitch_cm: float) -> tuple[np.ndarray, np.ndarray]:
    """The off-focal emission past the lit rows' edges from the air-column profiles of the views.

    The median profile over the views is divided by its lit level and averaged over the two sides of the lit rows by
    distance past their edges (the panel's last row left out).

    Args:
        profiles (list): [rows] air-column profiles (``air_column_profile``).
        first_lit_row (int): first lit row of the binned panel.
        num_lit_rows (int): number of lit rows.
        row_pitch_cm (float): binned row pitch in cm at the panel.

    Returns:
        (ndarray, ndarray): distances in cm past the edge (0 first) and the emission there over the lit level.
    """
    profile = np.median(np.array(profiles), axis=0)
    lit = float(np.median(profile[first_lit_row + 5: first_lit_row + num_lit_rows - 5]))
    below = profile[:first_lit_row][::-1] / lit
    above = profile[first_lit_row + num_lit_rows:] / lit
    n = min(len(below), len(above)) - 1
    distance = (np.arange(n) + 0.5) * row_pitch_cm
    fraction = (below[:n] + above[:n]) / 2
    return np.r_[0.0, distance], np.r_[fraction[0], fraction]


def bowtie_transmission(case_dir: str, channel_bin: int, smooth_channels: float = 2.0) -> tuple[np.ndarray, str]:
    """The bowtie's transmission per channel from a Varian air pair (FilterBowtie.xim over Filter.xim).

    The scan's own pair comes first (AIR-Half-Bowtie* for a half fan, AIR-Bowtie* for the Halcyon), else a factory
    pair. The ratio is averaged over the middle half of the rows, binned like the projections and lightly smoothed;
    it is not normalized, so its maximum is the transmission of the bowtie's thinnest part.

    Args:
        case_dir (str): case folder.
        channel_bin (int): channels per binned channel.
        smooth_channels (float): Gaussian sigma in binned channels. Defaults to 2.0.

    Returns:
        (ndarray, str): [Cd] transmission and the calibration folder it comes from.
    """
    width = sitk.ImageFileReader()
    width.SetFileName(os.path.join(case_dir, "projections.mha"))
    width.ReadImageInformation()
    width = width.GetSize()[0]
    calibrations = os.path.join(case_dir, "Calibrations")
    folders = (sorted(glob.glob(os.path.join(calibrations, "AIR-Half-Bowtie*", "Current")))
               + sorted(glob.glob(os.path.join(calibrations, "AIR-Bowtie*", "Current")))
               + sorted(glob.glob(os.path.join(calibrations, "AIR*", "Factory"))))
    for folder in folders:
        pair = [os.path.join(folder, name) for name in ("Filter.xim", "FilterBowtie.xim")]
        if not all(os.path.exists(f) for f in pair):
            continue
        flat, bowtie = (read_xim(f)[0].astype(np.float64) for f in pair)
        if flat.shape != bowtie.shape or flat.shape[1] != width:
            continue
        rows = slice(flat.shape[0] // 4, -(flat.shape[0] // 4))
        ratio = bowtie[rows].mean(axis=0) / np.maximum(flat[rows].mean(axis=0), 1.0)
        ratio = ratio[: len(ratio) // channel_bin * channel_bin].reshape(-1, channel_bin).mean(axis=1)
        if smooth_channels > 0:
            ratio = gaussian_filter1d(ratio, smooth_channels)
        return np.clip(ratio, 1e-3, 1.0).astype(np.float32), os.path.relpath(folder, case_dir)
    raise FileNotFoundError(f"no bowtie air pair in {calibrations}")


def half_fan_weights(geometry: ConeGeometry) -> np.ndarray:
    """Redundancy weights of an offset panel over a whole rotation (Wang, Med. Phys. 29, 2002), per channel.

    In the overlap |u| <= s around the central ray, s the panel's short side, the weight falls from 1 to 0 as
    sin^2(pi/4 (1 - g / s)), g the channel's position toward the short side, so that the two measurements of each
    line add up to 1; the long side beyond the overlap has weight 1.

    Args:
        geometry (ConeGeometry): scan geometry.

    Returns:
        ndarray: [Cd] float32 weights.
    """
    u = (np.arange(geometry.num_det_channels) - (geometry.num_det_channels - 1) / 2) * geometry.delta_det_channel \
        - geometry.det_channel_offset
    short = min(-u.min(), u.max())
    g = u if -u.min() > u.max() else -u          # the short side at positive g
    return np.where(np.abs(g) <= short, np.sin(np.pi / 4 * (1 - g / short)) ** 2, np.where(g < 0, 1.0, 0.0)).astype(
        np.float32)


# ======================================================================================================================
# Couch
# ======================================================================================================================

def couch_attenuation(case_dir: str, name: str, shape: tuple[int, int], voxel_cm: float, top_mm: float,
                      lateral_mm: float, shell_per_cm: float, interior_per_cm: float) -> tuple[np.ndarray, np.ndarray]:
    """Varian's couch-top model (Calibrations/COUCH) on an axial plane of a CITRUS grid.

    The shell between CouchSurface and CouchInterior and the interior get one attenuation each. The top surface sits
    at y = top_mm (CITRUS y, anterior positive) and the model's centre at x = lateral_mm.

    Args:
        case_dir (str): case folder.
        name (str): couch top, e.g. "IGRT".
        shape (tuple[int, int]): (rows, cols) of the grid's axial plane (rows along y, cols along x).
        voxel_cm (float): voxel size in cm.
        top_mm (float): y of the couch's top surface in mm.
        lateral_mm (float): x of the couch's centre in mm.
        shell_per_cm (float): attenuation of the shell in 1/cm.
        interior_per_cm (float): attenuation of the interior in 1/cm.

    Returns:
        (ndarray, ndarray): [rows, cols] float32 attenuation in 1/cm and bool mask of the couch.
    """
    from matplotlib.path import Path

    surface, interior = couch_contours(case_dir, name)
    rows, cols = shape
    d = voxel_cm * 10.0
    y, x = np.meshgrid((np.arange(rows) - (rows - 1) / 2) * d, (np.arange(cols) - (cols - 1) / 2) * d, indexing="ij")
    points = np.stack([x.ravel() - lateral_mm, top_mm - y.ravel()], axis=1)
    outer = Path(surface).contains_points(points).reshape(rows, cols)
    inner = Path(interior).contains_points(points).reshape(rows, cols)
    mu = np.where(inner, interior_per_cm, np.where(outer, shell_per_cm, 0.0)).astype(np.float32)
    return mu, outer | inner


# ======================================================================================================================
# Initial volume and resampling
# ======================================================================================================================

def rebin_half_fan(p, angles: np.ndarray, geometry: ConeGeometry, gamma_sign: float = 1.0):
    """Rebin a whole-rotation half-fan sinogram into full-fan views on a centred panel.

    Each virtual channel u takes the measured ray where the panel covers u, the conjugate ray (the view at
    beta + pi + 2 gamma, channel -u, interpolated linearly in angle and channel) where it does not, and a blend of
    the two with ``half_fan_weights`` in the overlap. ``gamma_sign`` is the sense of the 2 gamma shift; the returned
    mismatch (mean |direct - conjugate| in the overlap) tells the right one.

    Args:
        p (Tensor): [V, Rd, Cd] line integrals.
        angles (ndarray): [V] view angles in rad, ascending, one rotation.
        geometry (ConeGeometry): geometry of the offset panel.
        gamma_sign (float): sense of the conjugate shift, 1 or -1. Defaults to 1.

    Returns:
        (Tensor, ConeGeometry, float): rebinned sinogram, the centred panel's geometry and the overlap mismatch.
    """
    import dataclasses

    import torch

    V, Rd, Cd = p.shape
    device = p.device
    du, sdd = geometry.delta_det_channel, geometry.source_detector_dist
    u_meas = (np.arange(Cd) - (Cd - 1) / 2) * du - geometry.det_channel_offset
    short = min(abs(u_meas.min()), abs(u_meas.max()))
    n_full = 2 * int(np.ceil(max(abs(u_meas.min()), abs(u_meas.max())) / du))
    u_full = (np.arange(n_full) - (n_full - 1) / 2) * du
    long_negative = abs(u_meas.min()) > abs(u_meas.max())
    gamma = np.arctan(u_full / sdd)

    def channel(u_target):
        c = np.clip((u_target + geometry.det_channel_offset) / du + (Cd - 1) / 2, 0, Cd - 1 - 1e-6)
        k = np.floor(c).astype(int)
        return torch.as_tensor(k, device=device), torch.as_tensor((c - k).astype(np.float32), device=device)

    def sample(k, f):
        return p[:, :, k] * (1 - f) + p[:, :, torch.clamp(k + 1, max=Cd - 1)] * f

    direct, mirrored = sample(*channel(u_full)), sample(*channel(-u_full))
    covered = torch.as_tensor((u_full >= u_meas.min() - 0.5 * du) & (u_full <= u_meas.max() + 0.5 * du), device=device)
    a = np.asarray(angles, dtype=np.float64)
    conjugate = torch.empty_like(direct)
    for j in range(n_full):
        target = (a + np.pi + gamma_sign * 2 * gamma[j] - a[0]) % (2 * np.pi) + a[0]
        idx = np.searchsorted(a, target)
        lo, hi = (idx - 1) % V, idx % V
        width = (a[hi] - a[lo]) % (2 * np.pi)
        w = np.where(width > 0, ((target - a[lo]) % (2 * np.pi)) / np.where(width > 0, width, 1.0), 0.0)
        w = torch.as_tensor(w.astype(np.float32), device=device)[:, None]
        conjugate[:, :, j] = (mirrored[torch.as_tensor(lo, device=device), :, j] * (1 - w)
                              + mirrored[torch.as_tensor(hi, device=device), :, j] * w)
    g = torch.as_tensor(((1.0 if long_negative else -1.0) * u_full).astype(np.float32), device=device)
    w_direct = torch.where(g.abs() <= short, torch.sin(np.pi / 4 * (1 - g / short)) ** 2,
                           torch.where(g < 0, torch.ones_like(g), torch.zeros_like(g)))
    w_direct = torch.where(covered, w_direct, torch.zeros_like(w_direct))
    full = direct * w_direct + conjugate * (1 - w_direct)
    overlap = covered & (g.abs() <= 0.9 * short)
    mismatch = float((direct[:, :, overlap] - conjugate[:, :, overlap]).abs().mean())
    return full, dataclasses.replace(geometry, num_det_channels=n_full, det_channel_offset=0.0), mismatch


def initial_volume(scan: CobraScan, model, grid: Grid, device="cuda"):
    """FDK of the uncorrected line integrals, clamped at 0: the start of the CITRUS loop.

    A whole-rotation half fan is rebinned to full-fan views first (``rebin_half_fan``, the sense of the conjugate
    shift with the smaller overlap mismatch); other scans get the library's redundancy-weighted FDK.

    Args:
        scan (CobraScan): the scan.
        model (ConeBeamModel): MBIRTORCH model of the loop.
        grid (Grid): the loop grid.
        device (str): device. Defaults to "cuda".

    Returns:
        Tensor: [R, C, S] volume in 1/cm.
    """
    import torch

    from citrus.recon.mbir_bridge import fdk, make_cone_model, scan_range

    p = -torch.log(torch.as_tensor(scan.counts, device=device).clamp_min(1.0)
                   / torch.as_tensor(scan.flat_field, device=device))
    if scan.half_fan and abs(scan_range(scan.angles) - 2 * np.pi) < np.radians(3.0):
        candidates = [rebin_half_fan(p, scan.angles, scan.geometry, sign) for sign in (1.0, -1.0)]
        full, geometry_full, _ = min(candidates, key=lambda c: c[2])
        del candidates, p
        model_full = make_cone_model(grid, geometry_full, scan.angles, device=device)
        X = model_full.recon_fdk(full, filter_name="ramp", output_sharded=True)
        del model_full, full
    else:
        X = fdk(model, p, geometry=scan.geometry, angles=scan.angles)
    return torch.as_tensor(X, device=device).clamp_min(0.0)


def read_volume(path: str) -> tuple[np.ndarray, tuple, tuple]:
    """A MetaImage volume as [z, y, x] float32 with its origin and spacing in mm.

    Args:
        path (str): .mha file.

    Returns:
        (ndarray, tuple, tuple): the volume, its origin and its spacing.
    """
    image = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(image).astype(np.float32), tuple(image.GetOrigin()), tuple(image.GetSpacing())


def write_volume(volume: np.ndarray, origin: tuple, spacing: tuple, path: str) -> None:
    """Write a [z, y, x] volume as MetaImage with an origin and spacing in mm.

    Args:
        volume (ndarray): [z, y, x] volume.
        origin (tuple): origin in mm.
        spacing (tuple): spacing in mm.
        path (str): .mha file.
    """
    image = sitk.GetImageFromArray(volume.astype(np.float32))
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    sitk.WriteImage(image, path, useCompression=True)


def citrus_to_lps(volume: np.ndarray, grid: Grid, shape: tuple, origin: tuple, spacing: tuple) -> np.ndarray:
    """Resample a CITRUS volume [rows, cols, slices] onto an LPS grid [z, y, x] (CITRUS x = L, y = -P, z = -S).

    Args:
        volume (ndarray): [R, C, S] volume.
        grid (Grid): its grid (MBIRTORCH places slice k at z = dz (k - (S - 1) / 2) + recon_slice_offset, whatever
            the panel's row offset).
        shape (tuple): (nz, ny, nx) of the LPS grid.
        origin (tuple): its origin in mm.
        spacing (tuple): its spacing in mm.

    Returns:
        ndarray: [nz, ny, nx] float32 volume; points outside the CITRUS grid take the volume's minimum.
    """
    nz, ny, nx = shape
    z, y, x = (origin[i] + spacing[i] * np.arange(n) for i, n in ((2, nz), (1, ny), (0, nx)))
    R, C, S = volume.shape
    rows = (-y / 10.0 / grid.delta_voxel + (R - 1) / 2.0)
    cols = x / 10.0 / grid.delta_voxel + (C - 1) / 2.0
    slices = (-z / 10.0 - grid.recon_slice_offset) / grid.delta_voxel + (S - 1) / 2.0
    zz, yy, xx = np.meshgrid(slices, rows, cols, indexing="ij")
    out = map_coordinates(volume, [yy.ravel(), xx.ravel(), zz.ravel()], order=1, mode="constant",
                          cval=float(volume.min()))
    return out.reshape(nz, ny, nx).astype(np.float32)


# ======================================================================================================================
# Figure
# ======================================================================================================================

def comparison_figure(recon_hu: np.ndarray, clinical_hu: np.ndarray, ct_hu: np.ndarray, fov: np.ndarray,
                      spacing: tuple, path: str, title: str, window=(-400.0, 400.0),
                      difference_window: float = 200.0) -> None:
    """Axial and coronal slices of the deformed CT, the clinical CBCT and the CITRUS reconstruction, with the
    differences of the two CBCTs against the CT inside the field of view.

    The images share the CT's HU window and the differences a symmetric window. The axial slice is the middle one;
    the coronal slice passes through the body's centre on it.

    Args:
        recon_hu (ndarray): [z, y, x] CITRUS reconstruction in HU on the CT's grid.
        clinical_hu (ndarray): [z, y, x] clinical CBCT in HU.
        ct_hu (ndarray): [z, y, x] deformed CT in HU.
        fov (ndarray): [z, y, x] field-of-view mask of the differences.
        spacing (tuple): (x, y, z) spacing in mm.
        path (str): output image.
        title (str): figure title.
        window (tuple): HU window of the images. Defaults to (-400, 400).
        difference_window (float): the differences are shown in [-w, w] HU. Defaults to 200.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nz = ct_hu.shape[0]
    kz = nz // 2
    body = ct_hu[kz] > -300
    ky = int(round(np.nonzero(body)[0].mean())) if body.any() else ct_hu.shape[1] // 2
    inside = fov > 0
    columns = [("deformed CT", ct_hu, False), ("clinical CBCT", clinical_hu, False), ("CITRUS", recon_hu, False),
               ("clinical CBCT - CT", clinical_hu - ct_hu, True), ("CITRUS - CT", recon_hu - ct_hu, True)]
    views = [("axial", lambda v: v[kz], "upper", spacing[1] / spacing[0]),
             ("coronal", lambda v: v[:, ky], "lower", spacing[2] / spacing[0])]
    nz, ny, nx = ct_hu.shape
    heights = [ny * spacing[1] / (nx * spacing[0]), nz * spacing[2] / (nx * spacing[0])]   # per unit panel width
    fig, axes = plt.subplots(2, len(columns), figsize=(3.4 * len(columns), 3.2 * sum(heights) + 1.6),
                             gridspec_kw={"height_ratios": heights}, constrained_layout=True)
    for i, (view, cut, origin, aspect) in enumerate(views):
        for j, (name, volume, difference) in enumerate(columns):
            ax = axes[i, j]
            if difference:
                image = np.where(cut(inside), cut(volume), np.nan)
                mae = float(np.abs(volume[inside & (ct_hu > -1000)]).mean())
                shown = ax.imshow(image, cmap="RdBu_r", vmin=-difference_window, vmax=difference_window,
                                  origin=origin, aspect=aspect, interpolation="nearest")
                ax.set_title(f"{name}\nMAE in the FOV {mae:.0f} HU" if i == 0 else name, fontsize=10)
            else:
                shown = ax.imshow(cut(volume), cmap="gray", vmin=window[0], vmax=window[1], origin=origin,
                                  aspect=aspect, interpolation="nearest")
                ax.set_title(name, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(view)
        fig.colorbar(axes[i, 2].images[0], ax=axes[i, :3], shrink=0.8, label="HU")
        fig.colorbar(shown, ax=axes[i, 3:], shrink=0.8, label="HU difference")
    fig.suptitle(title)
    fig.savefig(path, dpi=120)
    plt.close(fig)
