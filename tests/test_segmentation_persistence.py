import json
import struct
import tempfile
import unittest
from pathlib import Path
import zipfile
from unittest.mock import patch

import numpy as np

from laparoskan.case import Case, SegmentInfo
from laparoskan.demo import create_demo_case
from laparoskan.geometry import ImageGeometry
from laparoskan.persistence import _MAX_VOLUME_MEMBER_BYTES, load_case, save_case
from laparoskan.segmentation import (
    RegionGrowLimitError,
    clear_segment,
    create_segment,
    paint_voxel,
    region_grow_segment,
    threshold_segment,
)


IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


class SegmentationTests(unittest.TestCase):
    def test_physical_brush_and_threshold_region_edit_labelmap(self):
        geometry = ImageGeometry((12, 10, 8), (0.5, 1.0, 2.0), (0, 0, 0), IDENTITY)
        volume = np.zeros((8, 10, 12), dtype=np.int16)
        volume[2:4, 3:6, 4:8] = 200
        case = Case("segmentation", volume, geometry)
        segment = create_segment(case, "Test structure", (10, 20, 30))
        point = tuple(geometry.index_to_physical((5, 4, 3)))
        changed = paint_voxel(case, point, segment.id, radius_mm=1.0)
        self.assertGreater(changed, 1)
        self.assertEqual(int(case.segmentation[3, 4, 5]), segment.id)
        revision = case.segmentation_revision
        thresholded = threshold_segment(
            case, segment.id, 200, 200,
            region=(slice(2, 4), slice(3, 6), slice(4, 8)),
        )
        self.assertGreater(thresholded, 0)
        self.assertGreater(case.segmentation_revision, revision)
        self.assertGreater(clear_segment(case, segment.id), 0)
        self.assertFalse(np.any(case.segmentation == segment.id))

    def test_region_grow_is_6_connected_and_respects_other_labels(self):
        geometry = ImageGeometry((8, 7, 6), (1, 1, 1), (0, 0, 0), IDENTITY)
        volume = np.zeros((6, 7, 8), dtype=np.int16)
        volume[2, 2, 1:4] = 100
        volume[2, 3, 1] = 100
        labels = np.zeros_like(volume, dtype=np.uint8)
        labels[2, 2, 2] = 2  # A marked structure blocks the flood in X.
        case = Case("region grow", volume, geometry, labels)
        segment = create_segment(case, "Grown region")
        seed = tuple(geometry.index_to_physical((1, 2, 2)))

        changed = region_grow_segment(case, segment.id, seed, tolerance_hu=0, max_voxels=8)

        self.assertEqual(changed, 2)
        self.assertEqual(int(case.segmentation[2, 2, 1]), segment.id)
        self.assertEqual(int(case.segmentation[2, 3, 1]), segment.id)
        self.assertEqual(int(case.segmentation[2, 2, 2]), 2)
        self.assertEqual(int(case.segmentation[2, 2, 3]), 0)

    def test_region_grow_cap_raises_without_partial_mask_write(self):
        geometry = ImageGeometry((8, 7, 6), (1, 1, 1), (0, 0, 0), IDENTITY)
        volume = np.zeros((6, 7, 8), dtype=np.int16)
        volume[2, 2, 1:5] = 100
        case = Case("bounded region grow", volume, geometry)
        segment = create_segment(case, "Bounded region")
        seed = tuple(geometry.index_to_physical((1, 2, 2)))
        before = case.segmentation.copy()
        revision = case.segmentation_revision

        with self.assertRaises(RegionGrowLimitError):
            region_grow_segment(case, segment.id, seed, tolerance_hu=0, max_voxels=3)

        np.testing.assert_array_equal(case.segmentation, before)
        self.assertEqual(case.segmentation_revision, revision)

    def test_case_rejects_orphaned_nonzero_label_ids(self):
        geometry = ImageGeometry((4, 3, 2), (1, 1, 1), (0, 0, 0), IDENTITY)
        labels = np.zeros((2, 3, 4), dtype=np.uint8)
        labels[0, 1, 2] = 2
        segments = (
            SegmentInfo(1, "First", (10, 20, 30)),
            SegmentInfo(3, "Third", (30, 20, 10)),
        )
        with self.assertRaisesRegex(ValueError, "undefined segment id"):
            Case("orphan label", np.zeros_like(labels, dtype=np.int16), geometry,
                 labels, segments=segments)


class CasePersistenceTests(unittest.TestCase):
    def test_round_trip_preserves_volume_masks_geometry_and_metadata(self):
        case = create_demo_case()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "demo.lapcase"
            self.assertEqual(save_case(case, path), str(path))
            reopened = load_case(path)
            self.assertEqual(reopened.name, case.name)
            self.assertEqual(reopened.case_id, case.case_id)
            self.assertEqual(reopened.geometry, case.geometry)
            self.assertEqual(reopened.metadata["reference_is_synthetic"], True)
            np.testing.assert_array_equal(reopened.volume, case.volume)
            np.testing.assert_array_equal(reopened.segmentation, case.segmentation)
            self.assertTrue(reopened.segmentation.flags.writeable)
            added = create_segment(reopened, "Post-open edit", (30, 200, 160))
            point = tuple(reopened.geometry.index_to_physical((8, 8, 8)))
            # Allow a small sub-voxel radius so floating-point LPS/index round
            # trips do not make a nominal zero-radius brush miss the center.
            self.assertEqual(
                paint_voxel(reopened, point, added.id, radius_mm=0.1 * min(reopened.geometry.spacing_xyz)),
                1,
            )
            self.assertEqual(int(reopened.segmentation[8, 8, 8]), added.id)
            reopened.close()

    def test_invalid_case_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "broken.lapcase"
            path.write_bytes(b"not a zip archive")
            with self.assertRaises(Exception):
                load_case(path)

    def test_declared_npy_shape_mismatch_is_rejected_before_extraction(self):
        geometry = ImageGeometry((4, 3, 2), (1, 1, 1), (0, 0, 0), IDENTITY)
        case = Case("small", np.zeros((2, 3, 4), dtype=np.int16), geometry)
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.lapcase"
            corrupt = Path(temp) / "shape-mismatch.lapcase"
            save_case(case, source)
            with zipfile.ZipFile(source, "r") as archive:
                members = {name: archive.read(name) for name in archive.namelist()}
            manifest = json.loads(members["manifest.json"])
            manifest["geometry"]["size_xyz"][0] = 5
            members["manifest.json"] = json.dumps(manifest).encode("utf-8")
            with zipfile.ZipFile(corrupt, "w", compression=zipfile.ZIP_STORED) as archive:
                for name, content in members.items():
                    archive.writestr(name, content)

            with patch("laparoskan.persistence.tempfile.TemporaryDirectory") as create_temp:
                with self.assertRaisesRegex(ValueError, "shape or dtype"):
                    load_case(corrupt)
                create_temp.assert_not_called()

    def test_oversized_declared_zip_member_is_rejected_before_extraction(self):
        geometry = ImageGeometry((4, 3, 2), (1, 1, 1), (0, 0, 0), IDENTITY)
        case = Case("small", np.zeros((2, 3, 4), dtype=np.int16), geometry)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "oversized.lapcase"
            save_case(case, path)
            data = bytearray(path.read_bytes())
            cursor = 0
            patched = False
            signature = b"PK\x01\x02"
            while True:
                offset = data.find(signature, cursor)
                if offset < 0:
                    break
                name_size, extra_size, comment_size = struct.unpack_from("<HHH", data, offset + 28)
                name_start = offset + 46
                name = bytes(data[name_start:name_start + name_size])
                if name == b"volume.npy":
                    struct.pack_into("<I", data, offset + 24, _MAX_VOLUME_MEMBER_BYTES + 1)
                    patched = True
                    break
                cursor = name_start + name_size + extra_size + comment_size
            self.assertTrue(patched, "test archive did not contain a volume.npy central-directory entry")
            path.write_bytes(data)

            with patch("laparoskan.persistence.tempfile.TemporaryDirectory") as create_temp:
                with self.assertRaisesRegex(ValueError, "volume member exceeds"):
                    load_case(path)
                create_temp.assert_not_called()


if __name__ == "__main__":
    unittest.main()
