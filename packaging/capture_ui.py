"""Capture the synthetic planning screen without loading patient data."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: capture_ui.py OUTPUT.png", file=sys.stderr)
        return 2
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from laparoskan.demo import create_demo_case
    from laparoskan.ui import MainWindow

    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = MainWindow(case=create_demo_case())
    window._navigate(2, 1, 1)
    window.show()
    output = Path(sys.argv[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"saved": False}

    def capture() -> None:
        result["saved"] = window.grab().save(str(output), "PNG")
        window.close()
        app.quit()

    QTimer.singleShot(1200, capture)
    QTimer.singleShot(20000, app.quit)
    app.exec()
    if not result["saved"] or not output.is_file():
        raise RuntimeError("Qt did not capture the synthetic planning screen")
    print(f"Synthetic planning screenshot: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
