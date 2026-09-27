import math
import unittest

import numpy as np

from laparoskan.case import Case
from laparoskan.geometry import ImageGeometry
from laparoskan.mpr import reslice_plane


def oblique_geometry(flipped=False):
    angle = math.radians(23)
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.asarray(((c, -s, 0), (s, c, 0), (0, 0, 1.0)))
    if flipped:
        rotation = rotation @ np.diag((-1.0, 1.0, 1.0))
    return ImageGeometry(
        size_xyz=(9, 7, 5),
        spacing_xyz=(0.7, 1.3, 2.4),
        origin_lps=(31.0, -44.0, 12.5),
        direction_lps=tuple(rotation.reshape(-1)),
    )


class ImageGeometryTests(unittest.TestCase):
    def test_non_isotropic_oblique_round_trip(self):
        geometry = oblique_geometry()
        indices = np.asarray(((0, 0, 0), (3.25, 5.5, 2.0), (8, 6, 4)), dtype=float)
        physical = geometry.index_to_physical(indices)
        np.testing.assert_allclose(geometry.physical_to_index(physical), indices, atol=1e-10)

    def test_flipped_direction_round_trip(self):
        geometry = oblique_geometry(flipped=True)
        self.assertLess(np.linalg.det(geometry.direction_matrix), 0.0)
        index = np.asarray((1.5, 2.0, 3.5))
        np.testing.assert_allclose(
            geometry.physical_to_index(geometry.index_to_physical(index)), index, atol=1e-10
        )

    def test_rejects_non_orthonormal_direction(self):
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            ImageGeometry((3, 3, 3), (1, 1, 1), (0, 0, 0), (1, 0, 0, 0.2, 1, 0, 0, 0, 1))

    def test_mpr_uses_physical_grid_for_all_planes(self):
        geometry = oblique_geometry(flipped=True)
        z, y, x = np.indices((5, 7, 9))
        volume = (x + 2 * y + 3 * z).astype(np.int16)
        case = Case("linear", volume, geometry)
        center = tuple(geometry.index_to_physical((4.0, 3.0, 2.0)))
        for plane in ("axial", "sagittal", "coronal"):
            with self.subTest(plane=plane):
                view = reslice_plane(case, plane, center, pixel_spacing_mm=0.5)
                self.assertEqual(view.pixels.ndim, 2)
                self.assertTrue(view.valid.any())
                col, row = view.pixels.shape[1] // 2, view.pixels.shape[0] // 2
                index = geometry.physical_to_index(view.pixel_to_lps(col, row))
                if np.all(index >= 0) and np.all(index <= np.asarray(geometry.size_xyz) - 1):
                    expected = index[0] + 2 * index[1] + 3 * index[2]
                    self.assertAlmostEqual(float(view.pixels[row, col]), float(expected), places=4)


if __name__ == "__main__":
    unittest.main()
