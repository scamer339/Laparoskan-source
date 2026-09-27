"""Application entry point and headless runtime smoke check."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
import tempfile
from pathlib import Path

from . import __version__


def _version_of(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def smoke_test() -> int:
    """Exercise synthetic geometry, MPR, VTK extraction and local case I/O."""
    try:
        import numpy as np
        import pydicom
        import SimpleITK as sitk
        import vtk
        from PySide6 import QtCore, QtGui, QtWidgets

        from .demo import create_demo_case
        from .mpr import reslice_plane
        from .persistence import load_case, save_case
        from .planning import evaluate_training, plan_trajectory
        from .segmentation import build_surface

        case = create_demo_case()
        center = case.geometry.index_to_physical(
            ((case.geometry.size_xyz[0] - 1) / 2,
             (case.geometry.size_xyz[1] - 1) / 2,
             (case.geometry.size_xyz[2] - 1) / 2)
        )
        mpr = {plane: reslice_plane(case, plane, tuple(center))
               for plane in ("axial", "sagittal", "coronal")}
        if any(view.pixels.ndim != 2 or not np.any(view.valid) for view in mpr.values()):
            raise RuntimeError("MPR reslicing returned an empty plane")

        abscess = case.segment_by_name("Abscess")
        surface = build_surface(case, abscess.id)
        if surface.GetNumberOfPoints() == 0 or surface.GetNumberOfCells() == 0:
            raise RuntimeError("VTK did not produce an abscess surface")

        entry = tuple(case.metadata["reference_entry_lps"])
        target = tuple(case.metadata["reference_target_lps"])
        plan = plan_trajectory(case, entry, target)
        if not plan.target_hit:
            raise RuntimeError("synthetic reference target does not hit the abscess label")
        exercise = evaluate_training(
            case, plan,
            reference_entry_lps=tuple(case.metadata["reference_entry_lps"]),
            reference_target_lps=tuple(case.metadata["reference_target_lps"]),
        )
        if not exercise["reference_errors_available"]:
            raise RuntimeError("synthetic reference exercise metrics are unavailable")

        with tempfile.TemporaryDirectory(prefix="laparoskan-smoke-") as temp:
            path = Path(temp) / "smoke.lapcase"
            save_case(case, path)
            reopened = load_case(path)
            if reopened.geometry != case.geometry:
                raise RuntimeError("patient-space geometry changed during save/reopen")
            if not np.array_equal(reopened.volume, case.volume):
                raise RuntimeError("volume changed during save/reopen")
            if not np.array_equal(reopened.segmentation, case.segmentation):
                raise RuntimeError("segmentation changed during save/reopen")

        print(f"Laparoskan {__version__} smoke test: PASS")
        print(
            "Runtime: "
            f"PySide6 {QtCore.__version__}; SimpleITK {sitk.Version_VersionString()}; "
            f"VTK {vtk.vtkVersion.GetVTKVersion()}; pydicom {pydicom.__version__}"
        )
        print(
            "Synthetic case: "
            f"{case.geometry.size_xyz[0]} × {case.geometry.size_xyz[1]} × "
            f"{case.geometry.size_xyz[2]} voxels; axial/sagittal/coronal MPR OK; "
            f"3D surface {surface.GetNumberOfPoints()} points"
        )
        print(f"Planning: target hit; depth {plan.depth_mm:.2f} mm; local save/reopen OK")
        return 0
    except Exception as exc:
        print(f"Laparoskan smoke test: FAIL ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Laparoskan educational CT planning application")
    parser.add_argument("--smoke-test", action="store_true",
                        help="check imaging dependencies, geometry, MPR, 3D extraction and case I/O")
    parser.add_argument("--version", action="version", version=f"Laparoskan {__version__}")
    args = parser.parse_args(argv)
    if args.smoke_test:
        return smoke_test()

    try:
        from PySide6.QtWidgets import QApplication
        from .demo import create_demo_case
        from .ui import MainWindow
    except Exception as exc:
        print(f"Laparoskan could not initialize its desktop interface: {exc}", file=sys.stderr)
        return 2

    application = QApplication.instance() or QApplication(sys.argv[:1])
    application.setApplicationName("Laparoskan")
    application.setOrganizationName("Laparoskan Research")
    window = MainWindow(case=create_demo_case())
    window.show()
    return int(application.exec())
