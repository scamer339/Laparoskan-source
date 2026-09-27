import math
import unittest
import warnings

import numpy as np

from laparoskan.case import DEFAULT_SEGMENTS, Case, SegmentInfo
from laparoskan.geometry import ImageGeometry
from laparoskan.demo import create_demo_case
from laparoskan.planning import evaluate_training, plan_trajectory, suggest_trajectories


IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


class PlanningTests(unittest.TestCase):
    def test_depth_target_hit_and_collision_are_physical(self):
        geometry = ImageGeometry((12, 10, 8), (2.0, 0.5, 3.0), (10.0, -5.0, 20.0), IDENTITY)
        labels = np.zeros((8, 10, 12), dtype=np.uint8)
        labels[3, 2, 8] = 1
        labels[3, 2, 4] = 2
        case = Case("known", np.zeros_like(labels, dtype=np.int16), geometry, labels)
        entry = tuple(geometry.index_to_physical((1, 2, 3)))
        target = tuple(geometry.index_to_physical((8, 2, 3)))
        plan = plan_trajectory(case, entry, target)
        self.assertAlmostEqual(plan.depth_mm, 14.0, places=7)
        self.assertTrue(plan.target_hit)
        self.assertTrue(plan.collided)
        self.assertIn("Vessels", plan.collision_labels)
        self.assertAlmostEqual(plan.angle_degrees, 0.0, places=7)
        metrics = evaluate_training(case, plan, entry, target)
        self.assertTrue(metrics["reference_errors_available"])
        self.assertAlmostEqual(metrics["depth_error_mm"], 0.0, places=7)
        self.assertAlmostEqual(metrics["angular_error_degrees"], 0.0, places=7)

    def test_supercover_detects_line_grazing_voxel_boundary(self):
        geometry = ImageGeometry((10, 8, 6), (1, 1, 1), (0, 0, 0), IDENTITY)
        labels = np.zeros((6, 8, 10), dtype=np.uint8)
        # The path lies exactly on the boundary between y=1 and y=2 cells.
        # Vessel voxel y=2 is touched even though the centerline is not inside it.
        labels[2, 2, 4] = 2
        labels[2, 2, 8] = 1
        case = Case("grazing", np.zeros_like(labels, dtype=np.int16), geometry, labels)
        plan = plan_trajectory(case, (0.0, 1.5, 2.0), (8.0, 1.5, 2.0))
        self.assertTrue(plan.collided)
        self.assertIn("Vessels", plan.collision_labels)
        self.assertTrue(plan.target_hit)

    def test_oblique_flipped_volume_preserves_physical_distance(self):
        angle = math.radians(31)
        rotation = np.asarray(
            ((math.cos(angle), -math.sin(angle), 0),
             (math.sin(angle), math.cos(angle), 0),
             (0, 0, 1.0))
        ) @ np.diag((-1.0, 1.0, 1.0))
        geometry = ImageGeometry(
            (12, 9, 7), (1.6, 0.7, 2.3), (70.0, -12.0, 35.0), tuple(rotation.reshape(-1))
        )
        labels = np.zeros((7, 9, 12), dtype=np.uint8)
        labels[3, 4, 8] = 1
        labels[3, 4, 5] = 2
        case = Case("oblique", np.zeros_like(labels, dtype=np.int16), geometry, labels)
        entry = geometry.index_to_physical((1, 4, 3))
        target = geometry.index_to_physical((8, 4, 3))
        plan = plan_trajectory(case, tuple(entry), tuple(target))
        self.assertAlmostEqual(plan.depth_mm, 7 * 1.6, places=6)
        self.assertTrue(plan.target_hit)
        self.assertTrue(plan.collided)

    def test_clearance_is_a_conservative_lower_bound(self):
        geometry = ImageGeometry((12, 9, 6), (1, 1, 1), (0, 0, 0), IDENTITY)
        labels = np.zeros((6, 9, 12), dtype=np.uint8)
        labels[2, 5, 5] = 2
        labels[2, 2, 9] = 1
        case = Case("clearance", np.zeros_like(labels, dtype=np.int16), geometry, labels)
        plan = plan_trajectory(case, (1.0, 2.0, 2.0), (9.0, 2.0, 2.0))
        # Exact point-to-voxel-box clearance is 2.5 mm; subtracting half the
        # voxel diagonal intentionally returns a conservative lower bound.
        self.assertGreaterEqual(plan.clearances_mm["Vessels"], 0.0)
        self.assertLessEqual(plan.clearances_mm["Vessels"], 2.5)
        self.assertGreater(plan.clearances_mm["Vessels"], 1.0)

    def test_entry_validation_distinguishes_surface_interior_and_missing_mask(self):
        geometry = ImageGeometry((12, 10, 8), (1.0, 1.0, 1.0), (0, 0, 0), IDENTITY)
        labels = np.zeros((8, 10, 12), dtype=np.uint8)
        labels[3, 4, 2] = 5
        labels[3, 4, 10] = 1
        case = Case("surface entry", np.zeros_like(labels, dtype=np.int16), geometry, labels)
        surface_entry = tuple(geometry.index_to_physical((2, 4, 3)))
        interior_entry = tuple(geometry.index_to_physical((6, 4, 3)))
        target = tuple(geometry.index_to_physical((10, 4, 3)))

        self.assertIs(plan_trajectory(case, surface_entry, target).entry_on_body_surface, True)
        self.assertIs(plan_trajectory(case, interior_entry, target).entry_on_body_surface, False)

        case.segmentation[3, 4, 2] = 0
        case.touch_segmentation()
        self.assertIsNone(plan_trajectory(case, surface_entry, target).entry_on_body_surface)

    def test_entry_validation_uses_exposed_boundary_not_any_solid_body_voxel(self):
        geometry = ImageGeometry((16, 16, 12), (1.0, 1.0, 1.0), (0, 0, 0), IDENTITY)
        labels = np.zeros((12, 16, 16), dtype=np.uint8)
        # Simulate a threshold-filled body mask, plus one voxel at the image
        # boundary to verify out-of-volume neighbors count as exterior.
        labels[1:11, 1:11, 1:11] = 5
        labels[0, 13, 10] = 5
        labels[6, 6, 6] = 1  # Internal target overwrites one body-mask voxel.
        segments = tuple(
            SegmentInfo(item.id, "Body surface" if item.id == 5 else item.name, item.color_rgb)
            for item in DEFAULT_SEGMENTS
        )
        case = Case("solid body mask", np.zeros_like(labels, dtype=np.int16), geometry, labels,
                    segments=segments)
        outer_boundary = tuple(geometry.index_to_physical((1, 6, 6)))
        solid_interior = tuple(geometry.index_to_physical((6, 6, 5)))
        volume_edge = tuple(geometry.index_to_physical((10, 13, 0)))
        target = tuple(geometry.index_to_physical((6, 6, 6)))

        plan = plan_trajectory(case, outer_boundary, target)
        self.assertIs(plan.entry_on_body_surface, True)
        self.assertNotIn("Body surface", plan.collision_labels)
        self.assertNotIn("Body surface", plan.unmarked_critical_labels)
        self.assertNotIn("Body surface", plan.clearances_mm)
        self.assertIs(plan_trajectory(case, solid_interior, target).entry_on_body_surface, False)
        self.assertIs(plan_trajectory(case, volume_edge, target).entry_on_body_surface, True)
        candidates = suggest_trajectories(
            case, target, max_candidates=4, max_entry_points=16, search_radius_mm=5.0
        )
        self.assertGreaterEqual(len(candidates), 2)
        self.assertTrue(all(item.plan.entry_on_body_surface is True for item in candidates))

    def test_cropped_solid_body_at_image_edge_is_not_verified_skin(self):
        geometry = ImageGeometry((16, 16, 12), (1.0, 1.0, 1.0), (0, 0, 0), IDENTITY)
        labels = np.full((12, 16, 16), 5, dtype=np.uint8)
        labels[6, 8, 8] = 1  # A valid target, while the body label is cropped by the FOV.
        case = Case("cropped body", np.zeros_like(labels, dtype=np.int16), geometry, labels)
        edge_entry = tuple(geometry.index_to_physical((0, 8, 6)))
        target = tuple(geometry.index_to_physical((8, 8, 6)))

        plan = plan_trajectory(case, edge_entry, target)

        self.assertIs(plan.entry_on_body_surface, False)
        candidates = suggest_trajectories(
            case, target, max_candidates=4, max_entry_points=16, search_radius_mm=5.0
        )
        self.assertEqual(candidates, ())

    def test_unmarked_critical_labels_never_receive_clearance_claims(self):
        geometry = ImageGeometry((12, 10, 8), (1.0, 1.0, 1.0), (0, 0, 0), IDENTITY)
        labels = np.zeros((8, 10, 12), dtype=np.uint8)
        labels[3, 4, 10] = 1
        case = Case("unmarked critical", np.zeros_like(labels, dtype=np.int16), geometry, labels)
        entry = tuple(geometry.index_to_physical((1, 4, 3)))
        target = tuple(geometry.index_to_physical((10, 4, 3)))

        plan = plan_trajectory(case, entry, target, critical_labels=("Vessels",))

        self.assertIsNone(plan.closest_clearance_mm)
        self.assertEqual(plan.clearances_mm, {})
        self.assertEqual(plan.unmarked_critical_labels, ("Vessels",))
        self.assertFalse(plan.clearance_complete)

    def test_candidate_suggestions_are_deterministic_bounded_and_warning_free(self):
        case = create_demo_case()
        target = tuple(case.metadata["reference_target_lps"])

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            first = suggest_trajectories(
                case, target, max_candidates=3, max_entry_points=8, search_radius_mm=10.0
            )
            second = suggest_trajectories(
                case, target, max_candidates=3, max_entry_points=8, search_radius_mm=10.0
            )

        self.assertGreaterEqual(len(first), 2)
        self.assertLessEqual(len(first), 3)
        self.assertEqual([item.rank for item in first], [1, 2, 3][:len(first)])
        self.assertEqual([item.plan.entry_lps for item in first],
                         [item.plan.entry_lps for item in second])
        self.assertIn("not a clinical optimum", first[0].ranking_summary)
        outside_target = tuple(case.geometry.index_to_physical((5, 5, 5)))
        with self.assertRaisesRegex(ValueError, "inside the marked Abscess"):
            suggest_trajectories(case, outside_target, max_entry_points=8)
        no_abscess = case.segmentation.copy()
        no_abscess[no_abscess == 1] = 0
        unmarked_case = Case(
            "no marked target", case.volume.copy(), case.geometry, no_abscess,
            segments=case.segments,
        )
        with self.assertRaisesRegex(ValueError, "requires a marked Abscess target"):
            suggest_trajectories(unmarked_case, target, max_entry_points=8)

    def test_batch_geometry_transform_rejects_nonfinite_values(self):
        geometry = ImageGeometry((8, 7, 6), (1.0, 1.5, 2.0), (2, -3, 4), IDENTITY)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            points = geometry.index_to_physical(np.asarray([[1, 2, 3], [4, 5, 1]]))
            indices = geometry.physical_to_index(points)
        np.testing.assert_allclose(indices, ((1, 2, 3), (4, 5, 1)), atol=1e-12)
        with self.assertRaises(ValueError):
            geometry.index_to_physical((np.inf, 0, 0))


if __name__ == "__main__":
    unittest.main()
