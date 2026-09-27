"""Physical-space multiplanar reslicing for arbitrary orthonormal DICOM grids."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .case import Case


@dataclass(frozen=True)
class MPRSlice:
    plane: str
    pixels: np.ndarray
    valid: np.ndarray
    origin_lps: tuple[float, float, float]
    column_axis_lps: tuple[float, float, float]
    row_axis_lps: tuple[float, float, float]
    normal_lps: tuple[float, float, float]
    pixel_spacing_mm: float

    def pixel_to_lps(self, column: float, row: float) -> tuple[float, float, float]:
        point = (
            np.asarray(self.origin_lps)
            + float(column) * self.pixel_spacing_mm * np.asarray(self.column_axis_lps)
            + float(row) * self.pixel_spacing_mm * np.asarray(self.row_axis_lps)
        )
        return tuple(float(value) for value in point)


_PLANE_AXES = {
    # Column and row directions are stated in LPS.  Rows increase down-screen.
    "axial": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "sagittal": ((0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
    "coronal": ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
}


def _sample_linear(volume_zyx: np.ndarray, geometry, physical_grid_lps: np.ndarray,
                   outside_value: float) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized trilinear sampling; geometry remains in patient LPS space."""
    continuous = geometry.physical_to_index(physical_grid_lps)
    x, y, z = (continuous[..., index] for index in range(3))
    size_x, size_y, size_z = geometry.size_xyz
    valid = (
        (x >= 0.0) & (x <= size_x - 1)
        & (y >= 0.0) & (y <= size_y - 1)
        & (z >= 0.0) & (z <= size_z - 1)
    )

    x0 = np.floor(x).astype(np.intp)
    y0 = np.floor(y).astype(np.intp)
    z0 = np.floor(z).astype(np.intp)
    x1 = np.minimum(x0 + 1, size_x - 1)
    y1 = np.minimum(y0 + 1, size_y - 1)
    z1 = np.minimum(z0 + 1, size_z - 1)
    x0 = np.clip(x0, 0, size_x - 1)
    y0 = np.clip(y0, 0, size_y - 1)
    z0 = np.clip(z0, 0, size_z - 1)

    dx = (x - x0).astype(np.float32)
    dy = (y - y0).astype(np.float32)
    dz = (z - z0).astype(np.float32)
    source = np.asarray(volume_zyx)
    c000 = source[z0, y0, x0].astype(np.float32, copy=False)
    c100 = source[z0, y0, x1].astype(np.float32, copy=False)
    c010 = source[z0, y1, x0].astype(np.float32, copy=False)
    c110 = source[z0, y1, x1].astype(np.float32, copy=False)
    c001 = source[z1, y0, x0].astype(np.float32, copy=False)
    c101 = source[z1, y0, x1].astype(np.float32, copy=False)
    c011 = source[z1, y1, x0].astype(np.float32, copy=False)
    c111 = source[z1, y1, x1].astype(np.float32, copy=False)

    c00 = c000 * (1.0 - dx) + c100 * dx
    c10 = c010 * (1.0 - dx) + c110 * dx
    c01 = c001 * (1.0 - dx) + c101 * dx
    c11 = c011 * (1.0 - dx) + c111 * dx
    c0 = c00 * (1.0 - dy) + c10 * dy
    c1 = c01 * (1.0 - dy) + c11 * dy
    sampled = c0 * (1.0 - dz) + c1 * dz
    sampled = np.where(valid, sampled, np.float32(outside_value)).astype(np.float32, copy=False)
    return sampled, valid


def reslice_plane(
    case: Case,
    plane: str,
    center_lps: tuple[float, float, float],
    pixel_spacing_mm: float = 1.0,
    max_size: int = 768,
) -> MPRSlice:
    """Reslice axial, sagittal, or coronal data on a patient-axis plane.

    Sampling is done in physical LPS coordinates and therefore supports oblique
    and flipped source directions.  Intensity pixels are returned in source
    units (for CT, Hounsfield units); display windowing belongs to the UI.
    """
    if plane not in _PLANE_AXES:
        raise ValueError(f"unsupported MPR plane: {plane}")
    if not math.isfinite(pixel_spacing_mm) or pixel_spacing_mm <= 0:
        raise ValueError("pixel_spacing_mm must be a positive finite value")
    if max_size < 2:
        raise ValueError("max_size must be at least two pixels")

    column = np.asarray(_PLANE_AXES[plane][0], dtype=np.float64)
    row = np.asarray(_PLANE_AXES[plane][1], dtype=np.float64)
    normal = np.cross(column, row)
    normal /= np.linalg.norm(normal)
    center = np.asarray(center_lps, dtype=np.float64)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("center_lps must contain three finite LPS coordinates")
    slice_offset = float(np.einsum("i,i->", center, normal, optimize=False))

    corners = case.geometry.boundary_corners_lps
    col_projection = np.einsum("ij,j->i", corners, column, optimize=False)
    row_projection = np.einsum("ij,j->i", corners, row, optimize=False)
    if not np.all(np.isfinite(col_projection)) or not np.all(np.isfinite(row_projection)):
        raise ValueError("MPR physical extent calculation produced non-finite values")
    min_col, max_col = float(col_projection.min()), float(col_projection.max())
    min_row, max_row = float(row_projection.min()), float(row_projection.max())
    extent_col, extent_row = max_col - min_col, max_row - min_row

    actual_spacing = max(
        float(pixel_spacing_mm),
        extent_col / (max_size - 1),
        extent_row / (max_size - 1),
    )
    width = max(2, int(math.ceil(extent_col / actual_spacing)) + 1)
    height = max(2, int(math.ceil(extent_row / actual_spacing)) + 1)
    col_values = min_col + np.arange(width, dtype=np.float64) * actual_spacing
    row_values = min_row + np.arange(height, dtype=np.float64) * actual_spacing
    grid = (
        col_values[None, :, None] * column
        + row_values[:, None, None] * row
        + slice_offset * normal
    )
    outside = -1024.0 if str(case.metadata.get("modality", "CT")).upper() == "CT" else 0.0
    pixels, valid = _sample_linear(case.volume, case.geometry, grid, outside)
    origin = min_col * column + min_row * row + slice_offset * normal

    return MPRSlice(
        plane=plane,
        pixels=pixels,
        valid=valid,
        origin_lps=tuple(float(v) for v in origin),
        column_axis_lps=tuple(float(v) for v in column),
        row_axis_lps=tuple(float(v) for v in row),
        normal_lps=tuple(float(v) for v in normal),
        pixel_spacing_mm=actual_spacing,
    )


def sample_labelmap_nearest(case: Case, physical_grid_lps: np.ndarray) -> np.ndarray:
    """Nearest-neighbour label sampling for a display overlay on any MPR plane."""
    assert case.segmentation is not None
    continuous = case.geometry.physical_to_index(physical_grid_lps)
    nearest = np.rint(continuous).astype(np.intp)
    size_x, size_y, size_z = case.geometry.size_xyz
    valid = (
        (nearest[..., 0] >= 0) & (nearest[..., 0] < size_x)
        & (nearest[..., 1] >= 0) & (nearest[..., 1] < size_y)
        & (nearest[..., 2] >= 0) & (nearest[..., 2] < size_z)
    )
    nearest[..., 0] = np.clip(nearest[..., 0], 0, size_x - 1)
    nearest[..., 1] = np.clip(nearest[..., 1], 0, size_y - 1)
    nearest[..., 2] = np.clip(nearest[..., 2], 0, size_z - 1)
    values = case.segmentation[nearest[..., 2], nearest[..., 1], nearest[..., 0]]
    return np.where(valid, values, 0).astype(np.uint8, copy=False)
