import tempfile
import unittest
from pathlib import Path
import warnings
import importlib.util

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from laparoskan.dicom import discover_series, load_series


def write_ct_slice(path: Path, series_uid: str, z: float, instance: int,
                   pixel_spacing=(0.8, 0.8), orientation=(1, 0, 0, 0, 1, 0),
                   x_shift: float = 0.0, study_uid: str | None = None):
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    sop_uid = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = sop_uid
    dataset.StudyInstanceUID = study_uid or generate_uid()
    dataset.SeriesInstanceUID = series_uid
    dataset.Modality = "CT"
    dataset.SeriesDescription = "Synthetic geometry fixture"
    dataset.Rows = 2
    dataset.Columns = 3
    dataset.PixelSpacing = list(pixel_spacing)
    dataset.ImagePositionPatient = [10.0 + x_shift, -20.0, float(z)]
    dataset.ImageOrientationPatient = list(orientation)
    dataset.SliceThickness = 1.0
    dataset.SpacingBetweenSlices = 0.8
    dataset.InstanceNumber = instance
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.RescaleIntercept = -1024
    dataset.RescaleSlope = 1
    pixels = np.full((2, 3), instance, dtype="<i2")
    dataset.PixelData = pixels.tobytes()
    dataset.save_as(path, write_like_original=False)


class DicomSeriesTests(unittest.TestCase):
    def _write_series(self, root: Path, positions, **overrides):
        uid = generate_uid()
        study_uid = generate_uid()
        for index, z in enumerate(positions):
            subfolder = root / "nested"
            subfolder.mkdir(exist_ok=True)
            # No extension: discovery must inspect DICOM headers, not filenames.
            write_ct_slice(subfolder / f"slice-{index}", uid, z, index + 1,
                           study_uid=study_uid, **overrides)
        return uid

    def test_extensionless_discovery_derives_slice_spacing_from_positions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_series(root, (0.0, 0.8, 1.6, 2.4))
            series = discover_series(root)
            self.assertEqual(len(series), 1)
            self.assertTrue(series[0].is_loadable, series[0].rejection_reason)
            self.assertEqual(series[0].slice_count, 4)
            self.assertEqual(series[0].spacing_xy_mm, (0.8, 0.8))
            self.assertAlmostEqual(series[0].slice_spacing_mm, 0.8, places=6)

    def test_auxiliary_files_are_skipped_without_pydicom_warnings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_series(root, (0.0, 0.8, 1.6))
            (root / "DICOMDIR").write_bytes(b"\0" * 512)
            (root / "unrelated-extensionless-data").write_bytes(b"\0" * 512)
            hidden = root / ".venv-dicom" / "nested"
            hidden.mkdir(parents=True)
            write_ct_slice(hidden / "fixture", generate_uid(), 0.0, 1)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                series = discover_series(root)
            self.assertEqual(len(series), 1)
            self.assertEqual(series[0].slice_count, 3)
            self.assertEqual(captured, [])

    def test_irregular_large_gap_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_series(root, (0.0, 0.8, 1.6, 58.4, 59.2))
            candidate = discover_series(root)[0]
            self.assertFalse(candidate.is_loadable)
            self.assertIn("Irregular slice spacing", candidate.rejection_reason)

    def test_duplicate_positions_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_series(root, (0.0, 0.8, 0.8, 1.6))
            candidate = discover_series(root)[0]
            self.assertFalse(candidate.is_loadable)
            self.assertIn("Duplicate", candidate.rejection_reason)

    def test_orientation_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            uid = generate_uid()
            folder = root / "nested"
            folder.mkdir()
            write_ct_slice(folder / "slice-a", uid, 0.0, 1)
            write_ct_slice(folder / "slice-b", uid, 0.8, 2,
                           orientation=(0, 1, 0, -1, 0, 0))
            candidate = discover_series(root)[0]
            self.assertFalse(candidate.is_loadable)
            self.assertIn("orientation changes", candidate.rejection_reason)

    def test_in_plane_slice_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            uid, study_uid = generate_uid(), generate_uid()
            folder = root / "nested"
            folder.mkdir()
            for index, z in enumerate((0.0, 0.8, 1.6)):
                write_ct_slice(folder / f"slice-{index}", uid, z, index + 1,
                               x_shift=0.2 * index, study_uid=study_uid)
            candidate = discover_series(root)[0]
            self.assertFalse(candidate.is_loadable)
            self.assertIn("shift within the image plane", candidate.rejection_reason)

    @unittest.skipIf(importlib.util.find_spec("SimpleITK") is None,
                     "SimpleITK is required for selected-series pixel loading")
    def test_selected_series_load_uses_projected_spacing_not_slice_thickness(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_series(root, (0.0, 0.8, 1.6))
            case = load_series(discover_series(root)[0])
            self.assertEqual(case.volume.shape, (3, 2, 3))
            self.assertAlmostEqual(case.geometry.spacing_xyz[2], 0.8, places=5)
            self.assertAlmostEqual(case.geometry.spacing_xyz[0], 0.8, places=5)
            self.assertEqual(case.metadata["spacing_source"], "ImagePositionPatient projected distances")
            self.assertEqual(case.volume.dtype, np.int16)


if __name__ == "__main__":
    unittest.main()
