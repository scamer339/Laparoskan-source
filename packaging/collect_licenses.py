"""Copy available installed-package license texts into a portable bundle."""

from __future__ import annotations

import importlib.metadata as metadata
from pathlib import Path
import shutil
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: collect_licenses.py BUNDLE_DIRECTORY", file=sys.stderr)
        return 2
    destination = Path(sys.argv[1]) / "licenses"
    destination.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for distribution in sorted(metadata.distributions(), key=lambda item: item.metadata.get("Name", "")):
        name = distribution.metadata.get("Name", "unnamed")
        version = distribution.version
        copied = 0
        for packaged_file in distribution.files or ():
            leaf = Path(str(packaged_file)).name.lower()
            if not leaf.startswith(("license", "copying", "notice", "copyright",
                                    "lgpl", "gpl", "apache", "bsd", "mit")):
                continue
            source = Path(distribution.locate_file(packaged_file))
            if not source.is_file() or source.stat().st_size > 5 * 1024 * 1024:
                continue
            package_dir = destination / f"{name}-{version}"
            package_dir.mkdir(parents=True, exist_ok=True)
            target = package_dir / f"{copied:02d}-{source.name}"
            shutil.copyfile(source, target)
            copied += 1
        lines.append(f"{name} {version}: {copied} text file(s)")
    (destination / "INDEX.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Collected available license texts from {len(lines)} installed distributions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
