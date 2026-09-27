"""Patient-space geometry primitives (DICOM LPS, millimetres)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Iterable

import numpy as np


def _triple(values: Iterable[float], name: str) -> tuple[float, float, float]:
    result = tuple(float(v) for v in values)
    if len(result) != 3 or not all(math.isfinite(v) for v in result):
        raise ValueError(f"{name} must contain three finite values")
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class ImageGeometry:
    """Mapping between XYZ voxel indices and DICOM patient LPS millimetres.

    ``direction_lps`` is a row-major 3x3 matrix whose columns are the physical
    directions of increasing voxel X, Y, and Z.  A determinant of -1 is valid:
    it represents an axis flip while preserving an orthonormal image grid.
    Arrays stored by :class:`laparoskan.case.Case` use Z,Y,X order.
    """

    size_xyz: tuple[int, int, int]
    spacing_xyz: tuple[float, float, float]
    origin_lps: tuple[float, float, float]
    direction_lps: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.size_xyz) != 3 or any(int(v) < 1 for v in self.size_xyz):
            raise ValueError("size_xyz must contain three positive dimensions")
        spacing = _triple(self.spacing_xyz, "spacing_xyz")
        _triple(self.origin_lps, "origin_lps")
        if any(v <= 0 for v in spacing):
            raise ValueError("spacing_xyz values must be positive")
        direction = np.asarray(self.direction_lps, dtype=np.float64)
        if direction.size != 9 or not np.all(np.isfinite(direction)):
            raise ValueError("direction_lps must contain nine finite values")
        direction = direction.reshape(3, 3)
        gram = direction.T @ direction
        if not np.allclose(gram, np.eye(3), atol=1e-4, rtol=1e-4):
            raise ValueError("direction_lps must be an orthonormal matrix")
        if not math.isclose(abs(float(np.linalg.det(direction))), 1.0, abs_tol=1e-4):
            raise ValueError("direction_lps must be invertible and orthonormal")

    @property
    def direction_matrix(self) -> np.ndarray:
        return np.asarray(self.direction_lps, dtype=np.float64).reshape(3, 3)

    def index_to_physical(self, index_xyz: Iterable[float] | np.ndarray) -> np.ndarray:
        """Convert one or many continuous XYZ indices to LPS millimetres."""
        indices = np.asarray(index_xyz, dtype=np.float64)
        if indices.shape == () or indices.shape[-1] != 3:
            raise ValueError("indices must have a final dimension of three (XYZ)")
        if not np.all(np.isfinite(indices)):
            raise ValueError("indices must be finite")
        scaled = indices * np.asarray(self.spacing_xyz, dtype=np.float64)
        physical = np.einsum("...j,ij->...i", scaled, self.direction_matrix, optimize=False)
        physical = physical + np.asarray(self.origin_lps, dtype=np.float64)
        if not np.all(np.isfinite(physical)):
            raise ValueError("index coordinates exceed finite physical LPS range")
        return physical

    def physical_to_index(self, point_lps: Iterable[float] | np.ndarray) -> np.ndarray:
        """Convert one or many LPS points to continuous XYZ voxel indices."""
        points = np.asarray(point_lps, dtype=np.float64)
        if points.shape == () or points.shape[-1] != 3:
            raise ValueError("points must have a final dimension of three (LPS)")
        if not np.all(np.isfinite(points)):
            raise ValueError("physical LPS points must be finite")
        delta = points - np.asarray(self.origin_lps, dtype=np.float64)
        # Direction is orthonormal, so projections onto its columns are the
        # inverse transform. Explicit sums avoid BLAS warnings on batched
        # short vectors and make invalid results straightforward to reject.
        local = np.einsum("...i,ij->...j", delta, self.direction_matrix, optimize=False)
        indices = local / np.asarray(self.spacing_xyz, dtype=np.float64)
        if not np.all(np.isfinite(indices)):
            raise ValueError("physical LPS coordinates exceed finite index range")
        return indices

    @property
    def boundary_corners_lps(self) -> np.ndarray:
        """Eight voxel-cell boundary corners in physical coordinates."""
        extents = [(-0.5, float(size) - 0.5) for size in self.size_xyz]
        return self.index_to_physical(np.asarray(list(product(*extents))))

    @property
    def voxel_diagonal_mm(self) -> float:
        return math.sqrt(sum(value * value for value in self.spacing_xyz))

    @property
    def voxel_count(self) -> int:
        return int(np.prod(self.size_xyz, dtype=np.int64))


def as_lps_point(point: Iterable[float]) -> tuple[float, float, float]:
    """Validate and normalize a physical point to an immutable XYZ tuple."""
    return _triple(point, "point_lps")
