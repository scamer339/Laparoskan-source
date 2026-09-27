"""DICOM series discovery and selected-series volume loading.

Discovery reads headers only.  Pixel data are loaded only for the series the
user selects, through SimpleITK/GDCM; no patient files are copied into the app.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

from .case import Case
from .geometry import ImageGeometry


class DicomSeriesError(ValueError):
    """A DICOM series is incomplete, irregular, or cannot be reconstructed."""


@dataclass(frozen=True)
class _SliceHeader:
    path: str
    series_uid: str
    modality: str
    description: str
    rows: int
    columns: int
    pixel_spacing_row_col: tuple[float, float]
    orientation_lps: tuple[float, ...]
    position_lps: tuple[float, float, float]


@dataclass(frozen=True)
class DicomSeries:
    series_uid: str
    description: str
    modality: str
    files: tuple[str, ...]
    rows: int
    columns: int
    spacing_xy_mm: tuple[float, float]
    slice_spacing_mm: float
    orientation_lps: tuple[float, ...]
    origin_lps: tuple[float, float, float]
    warnings: tuple[str, ...] = ()
    rejection_reason: str = ""

    @property
    def is_loadable(self) -> bool:
        return not self.rejection_reason and self.slice_count >= 2

    @property
    def slice_count(self) -> int:
        return len(self.files)

    @property
    def summary(self) -> str:
        state = "Ready" if self.is_loadable else f"Unavailable: {self.rejection_reason}"
        return (
            f"{self.modality or 'DICOM'} · {self.description or 'Unnamed series'} · "
            f"{self.columns} × {self.rows} × {self.slice_count} · "
            f"{self.spacing_xy_mm[0]:.3g} × {self.spacing_xy_mm[1]:.3g} × "
            f"{self.slice_spacing_mm:.3g} mm · {state}"
        )


def _read_header(path: Path) -> _SliceHeader | None:
    if path.name.casefold() == "dicomdir":
        return None
    try:
        with path.open("rb") as source:
            prefix = source.read(132)
        has_part10_marker = len(prefix) >= 132 and prefix[128:132] == b"DICM"
        has_meta_group = len(prefix) >= 6 and prefix[:2] in (b"\x02\x00", b"\x00\x02")
        extension_allows_raw = path.suffix.casefold() in {".dcm", ".dicom"}
        if not (has_part10_marker or has_meta_group or extension_allows_raw):
            return None
        if has_part10_marker or has_meta_group:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=False)
        else:
            # Force parsing only for conventionally named legacy DICOM files;
            # arbitrary extensionless assets are never fed to pydicom.
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        uid = str(getattr(ds, "SeriesInstanceUID", "")).strip()
        modality = str(getattr(ds, "Modality", "")).strip().upper()
        position = tuple(float(v) for v in ds.ImagePositionPatient)
        orientation = tuple(float(v) for v in ds.ImageOrientationPatient)
        spacing = tuple(float(v) for v in ds.PixelSpacing)
        rows, columns = int(ds.Rows), int(ds.Columns)
        if (
            not uid or len(position) != 3 or len(orientation) != 6 or len(spacing) != 2
            or rows < 1 or columns < 1
            or not all(np.isfinite(v) for v in (*position, *orientation, *spacing))
            or min(spacing) <= 0
        ):
            return None
        if modality not in {"CT", "MR"}:
            return None
        return _SliceHeader(
            path=str(path),
            series_uid=uid,
            modality=modality,
            description=str(getattr(ds, "SeriesDescription", "")).strip(),
            rows=rows,
            columns=columns,
            pixel_spacing_row_col=(spacing[0], spacing[1]),
            orientation_lps=orientation,
            position_lps=position,  # type: ignore[arg-type]
        )
    except (InvalidDicomError, OSError, ValueError, TypeError, AttributeError, KeyError):
        return None


def _build_series(headers: Sequence[_SliceHeader]) -> DicomSeries:
    if not headers:
        raise DicomSeriesError("No readable image slices were found")
    first = headers[0]
    reason = ""
    warnings: list[str] = []
    if any(h.series_uid != first.series_uid for h in headers):
        reason = "Selected files contain more than one DICOM series"

    if any((h.rows, h.columns) != (first.rows, first.columns) for h in headers):
        reason = reason or "Image dimensions change within this series"
    if any(h.modality != first.modality for h in headers):
        reason = reason or "Modality changes within this series"
    orientation = np.asarray(first.orientation_lps, dtype=np.float64)
    if any(not np.allclose(h.orientation_lps, orientation, atol=1e-4, rtol=1e-4) for h in headers):
        reason = reason or "Image orientation changes within this series"
    pixel_spacing = np.asarray(first.pixel_spacing_row_col, dtype=np.float64)
    if any(not np.allclose(h.pixel_spacing_row_col, pixel_spacing, atol=1e-3, rtol=1e-4) for h in headers):
        reason = reason or "Pixel spacing changes within this series"

    orientation_valid = bool(
        orientation.shape == (6,)
        and np.all(np.isfinite(orientation))
        and np.all(np.abs(orientation) <= 1.0001)
    )
    row_direction = orientation[:3] if orientation_valid else np.asarray((1.0, 0.0, 0.0))
    column_direction = orientation[3:] if orientation_valid else np.asarray((0.0, 1.0, 0.0))
    row_norm, col_norm = np.linalg.norm(row_direction), np.linalg.norm(column_direction)
    if (not orientation_valid or abs(row_norm - 1.0) > 1e-4
            or abs(col_norm - 1.0) > 1e-4
            or abs(float(row_direction @ column_direction)) > 1e-4):
        reason = reason or "Image orientation is not orthonormal"
    normal = np.cross(row_direction, column_direction)
    normal_norm = float(np.linalg.norm(normal))
    if not np.isfinite(normal_norm) or normal_norm <= 1e-8:
        reason = reason or "Image orientation does not define a slice direction"
        normal = np.asarray((0.0, 0.0, 1.0))
    else:
        normal /= normal_norm

    positions = np.asarray([h.position_lps for h in headers], dtype=np.float64)
    if not np.all(np.isfinite(positions)) or np.any(np.abs(positions) > 1e7):
        reason = reason or "Image positions contain invalid patient-space coordinates"
        projections = np.arange(len(headers), dtype=np.float64)
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            projections = np.einsum("ij,j->i", positions, normal)
        if not np.all(np.isfinite(projections)):
            reason = reason or "Image positions cannot be projected to finite physical coordinates"
            projections = np.arange(len(headers), dtype=np.float64)
    order = np.argsort(projections, kind="stable")
    sorted_positions = positions[order]
    sorted_projections = projections[order]
    sorted_headers = [headers[int(index)] for index in order]
    diffs = np.diff(sorted_projections)
    if len(diffs) == 0:
        step = 0.0
        reason = reason or "At least two slices are required to derive spacing from image positions"
    elif not np.all(np.isfinite(diffs)):
        step = 0.0
        reason = reason or "Slice position differences are not finite"
    else:
        if np.any(diffs <= 0.01):
            reason = reason or "Duplicate or reversed slice positions were found"
        step = float(median(float(v) for v in diffs))
        tolerance = max(0.05, step * 0.02)
        if any(abs(float(value) - step) > tolerance for value in diffs):
            reason = reason or (
                f"Irregular slice spacing: position steps range from "
                f"{float(np.min(diffs)):.3g} to {float(np.max(diffs)):.3g} mm"
            )
        if len(sorted_positions) > 1:
            position_steps = np.diff(sorted_positions, axis=0)
            in_plane_steps = position_steps - diffs[:, None] * normal[None, :]
            if any(float(np.linalg.norm(value)) > tolerance for value in in_plane_steps):
                reason = reason or (
                    "Slice positions shift within the image plane; this sheared or "
                    "gantry-tilted stack is not supported"
                )
    if step > 0 and len(diffs) and float(np.max(diffs)) > step * 1.5:
        reason = reason or "A large gap between slice positions makes this series irregular"

    return DicomSeries(
        series_uid=first.series_uid,
        description=first.description,
        modality=first.modality,
        files=tuple(h.path for h in sorted_headers),
        rows=first.rows,
        columns=first.columns,
        spacing_xy_mm=(float(pixel_spacing[1]), float(pixel_spacing[0])),
        slice_spacing_mm=step,
        orientation_lps=tuple(float(v) for v in orientation),
        origin_lps=tuple(float(v) for v in sorted_positions[0]),
        warnings=tuple(warnings),
        rejection_reason=reason,
    )


def discover_series(root: str | Path) -> list[DicomSeries]:
    """Scan a directory recursively for CT/MR headers, including extensionless files."""
    folder = Path(root).expanduser()
    if not folder.exists() or not folder.is_dir():
        raise DicomSeriesError(f"DICOM folder does not exist: {folder}")
    grouped: dict[str, list[_SliceHeader]] = {}
    excluded_directory_names = {
        ".git", ".venv", ".venv-dicom", "venv", "venv3", "__pycache__",
        "node_modules", "build", "dist", ".pytest_cache", ".ruff_cache",
    }
    for path in folder.rglob("*"):
        try:
            try:
                relative_parts = path.relative_to(folder).parts
            except ValueError:
                continue
            if any(part.casefold() in excluded_directory_names for part in relative_parts[:-1]):
                continue
            if not path.is_file() or path.is_symlink() or path.stat().st_size < 132:
                continue
        except OSError:
            continue
        header = _read_header(path)
        if header is not None:
            grouped.setdefault(header.series_uid, []).append(header)

    series = [_build_series(items) for items in grouped.values()]
    return sorted(series, key=lambda item: (item.modality, item.description.casefold(), item.series_uid))


def load_series(series: DicomSeries | Sequence[str | Path]) -> Case:
    """Load one validated series via GDCM, retaining its SimpleITK buffer."""
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError("SimpleITK is required to load DICOM pixel data") from exc
    if isinstance(series, DicomSeries):
        candidate = series
    else:
        headers = [header for path in series if (header := _read_header(Path(path))) is not None]
        candidate = _build_series(headers)
    if not candidate.is_loadable:
        raise DicomSeriesError(candidate.rejection_reason or "Series is not loadable")

    reader = sitk.ImageSeriesReader()
    reader.SetImageIO("GDCMImageIO")
    reader.SetFileNames(list(candidate.files))
    try:
        image = reader.Execute()
    except RuntimeError as exc:
        raise DicomSeriesError(f"GDCM could not read the selected series: {exc}") from exc

    size = tuple(int(v) for v in image.GetSize())
    if size != (candidate.columns, candidate.rows, candidate.slice_count):
        raise DicomSeriesError(
            f"Loaded volume size {size} does not match DICOM headers "
            f"({candidate.columns}, {candidate.rows}, {candidate.slice_count})"
        )

    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    expected_first = np.asarray(candidate.origin_lps)
    expected_last = np.asarray(
        pydicom.dcmread(candidate.files[-1], stop_before_pixels=True, force=True).ImagePositionPatient,
        dtype=np.float64,
    )
    if np.allclose(origin, expected_first, atol=0.05):
        expected_normal = np.cross(candidate.orientation_lps[:3], candidate.orientation_lps[3:])
    elif np.allclose(origin, expected_last, atol=0.05):
        expected_normal = -np.cross(candidate.orientation_lps[:3], candidate.orientation_lps[3:])
    else:
        raise DicomSeriesError("Loaded volume origin does not match the ordered slice positions")
    expected_x = np.asarray(candidate.orientation_lps[:3])
    expected_y = np.asarray(candidate.orientation_lps[3:])
    if not np.allclose(direction[:, 0], expected_x, atol=1e-4) or not np.allclose(direction[:, 1], expected_y, atol=1e-4):
        raise DicomSeriesError("Loaded volume directions do not match ImageOrientationPatient")
    if not np.allclose(direction[:, 2], expected_normal, atol=1e-4):
        raise DicomSeriesError("Loaded volume slice direction does not match image positions")

    spacing = tuple(float(v) for v in image.GetSpacing())
    expected_x_spacing, expected_y_spacing = candidate.spacing_xy_mm
    if abs(spacing[0] - expected_x_spacing) > 1e-3 or abs(spacing[1] - expected_y_spacing) > 1e-3:
        raise DicomSeriesError("Loaded in-plane spacing does not match PixelSpacing")
    # SimpleITK/GDCM may use SliceThickness.  Replace only Z spacing with the
    # robust median step derived from sorted ImagePositionPatient coordinates.
    image.SetSpacing((expected_x_spacing, expected_y_spacing, candidate.slice_spacing_mm))
    final_spacing = tuple(float(v) for v in image.GetSpacing())
    final_origin = tuple(float(v) for v in image.GetOrigin())
    final_direction = tuple(float(v) for v in image.GetDirection())
    geometry = ImageGeometry(size, final_spacing, final_origin, final_direction)
    volume = sitk.GetArrayViewFromImage(image)
    return Case(
        name=candidate.description or f"{candidate.modality} study",
        volume=volume,
        geometry=geometry,
        metadata={
            "modality": candidate.modality,
            "series_description": candidate.description,
            "slice_count": candidate.slice_count,
            "synthetic": False,
            "spacing_source": "ImagePositionPatient projected distances",
            "warnings": list(candidate.warnings),
        },
        source=str(Path(candidate.files[0]).parent),
        native_image=image,
    )
