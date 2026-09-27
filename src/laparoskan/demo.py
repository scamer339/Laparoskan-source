"""Deterministic synthetic abdominal phantom for training and smoke checks."""

from __future__ import annotations

import math

import numpy as np

from .case import Case
from .geometry import ImageGeometry


def create_demo_case() -> Case:
    """Create an explicitly synthetic, non-isotropic oblique CT-like phantom."""
    size_x, size_y, size_z = 128, 128, 96
    spacing = (0.82, 0.82, 1.2)
    angle = math.radians(11.0)
    c, s = math.cos(angle), math.sin(angle)
    direction = ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))
    geometry = ImageGeometry(
        size_xyz=(size_x, size_y, size_z),
        spacing_xyz=spacing,
        origin_lps=(-52.48, -52.48, -57.0),
        direction_lps=tuple(value for row in direction for value in row),
    )

    z, y, x = np.ogrid[:size_z, :size_y, :size_x]
    cx, cy, cz = (size_x - 1) / 2.0, (size_y - 1) / 2.0, (size_z - 1) / 2.0
    # All synthetic geometry is authored in voxel coordinates, then carries
    # the oblique DICOM transform through the shared ImageGeometry object.
    body_q = ((x - cx) / 58.0) ** 2 + ((y - cy) / 54.0) ** 2 + ((z - cz) / 40.0) ** 2
    body = body_q <= 1.0
    volume = np.full((size_z, size_y, size_x), -1000, dtype=np.int16)
    volume[body] = 28

    labels = np.zeros_like(volume, dtype=np.uint8)
    skin_shell = body & (body_q >= 0.90)
    labels[skin_shell] = 5

    def ellipsoid(center: tuple[float, float, float], radii: tuple[float, float, float]):
        ex, ey, ez = center
        rx, ry, rz = radii
        return (
            ((x - ex) / rx) ** 2
            + ((y - ey) / ry) ** 2
            + ((z - ez) / rz) ** 2
            <= 1.0
        )

    liver = ellipsoid((46, 52, 48), (25, 21, 17))
    volume[liver] = 95
    labels[liver] = 3

    other_organs = (
        ellipsoid((86, 56, 48), (11, 16, 13))
        | ellipsoid((42, 84, 48), (9, 14, 9))
        | ellipsoid((87, 84, 48), (9, 14, 9))
        | ellipsoid((67, 79, 50), (18, 9, 8))
    )
    volume[other_organs] = 58
    labels[other_organs] = 4

    # A segmented vessel tube sits near, but does not cross, the target.
    vessel = (
        ((x - 68) / 3.2) ** 2 + ((y - 66) / 3.2) ** 2 <= 1.0
    ) & (z >= 29) & (z <= 70)
    volume[vessel] = 180
    labels[vessel] = 2

    abscess = ellipsoid((58, 48, 49), (9.0, 8.0, 7.0))
    volume[abscess] = 22
    labels[abscess] = 1

    bones = (
        ellipsoid((64, 103, 49), (9, 6, 8))
        | ellipsoid((20, 51, 47), (4, 3, 23))
        | ellipsoid((106, 51, 47), (4, 3, 23))
    )
    volume[bones] = 700
    labels[bones] = 6

    reference_entry = geometry.index_to_physical((58.0, 48.0, 85.0))
    reference_target = geometry.index_to_physical((58.0, 48.0, 49.0))

    return Case(
        name="Synthetic abdominal phantom",
        volume=volume,
        geometry=geometry,
        segmentation=labels,
        metadata={
            "modality": "CT",
            "synthetic": True,
            "case_kind": "training_demo",
            "description": "Generated geometric phantom; not a patient scan or diagnosis.",
            "reference_entry_lps": [float(value) for value in reference_entry],
            "reference_target_lps": [float(value) for value in reference_target],
            "reference_is_synthetic": True,
            "window_center": 40,
            "window_width": 400,
        },
        source="Synthetic demo",
    )
