"""In-memory imaging case and editable integer segmentation labelmap."""

from __future__ import annotations

from dataclasses import dataclass, field
import uuid
from typing import Any

import numpy as np

from .geometry import ImageGeometry


@dataclass(frozen=True)
class SegmentInfo:
    id: int
    name: str
    color_rgb: tuple[int, int, int]

    def __post_init__(self) -> None:
        if not 1 <= int(self.id) <= 255:
            raise ValueError("segment id must be between 1 and 255")
        if not self.name.strip():
            raise ValueError("segment name cannot be empty")
        if len(self.color_rgb) != 3 or any(not 0 <= int(c) <= 255 for c in self.color_rgb):
            raise ValueError("color_rgb must contain three byte values")


DEFAULT_SEGMENTS = (
    SegmentInfo(1, "Abscess", (245, 168, 43)),
    SegmentInfo(2, "Vessels", (221, 69, 78)),
    SegmentInfo(3, "Liver", (168, 91, 77)),
    SegmentInfo(4, "Other organs", (201, 131, 112)),
    SegmentInfo(5, "Body contour", (167, 184, 202)),
    SegmentInfo(6, "Bones", (231, 220, 194)),
)


@dataclass
class Case:
    """A volume, patient-space geometry, and editable Z,Y,X labelmap.

    The medical labelmap is never replaced by a smoothed display surface.
    ``native_image`` may retain a SimpleITK image backing ``volume`` without
    making a second copy of a large selected DICOM series.
    """

    name: str
    volume: np.ndarray
    geometry: ImageGeometry
    segmentation: np.ndarray | None = None
    segments: tuple[SegmentInfo, ...] = DEFAULT_SEGMENTS
    case_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    native_image: Any | None = field(default=None, repr=False, compare=False)
    segmentation_revision: int = 0
    _segment_presence_cache: tuple[int, frozenset[int]] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        self.volume = np.asarray(self.volume)
        expected_zyx = tuple(reversed(self.geometry.size_xyz))
        if self.volume.ndim != 3 or self.volume.shape != expected_zyx:
            raise ValueError(
                f"volume must have Z,Y,X shape {expected_zyx}; got {self.volume.shape}"
            )
        segmentation_was_empty_default = self.segmentation is None
        if segmentation_was_empty_default:
            self.segmentation = np.zeros(expected_zyx, dtype=np.uint8)
        else:
            labels = np.asarray(self.segmentation)
            if labels.shape != expected_zyx:
                raise ValueError("segmentation shape must match volume Z,Y,X shape")
            if not np.issubdtype(labels.dtype, np.integer):
                raise ValueError("segmentation values must use an integer label dtype")
            if labels.size and (int(labels.min()) < 0 or int(labels.max()) > 255):
                raise ValueError("segmentation values must fit the uint8 labelmap")
            self.segmentation = labels.astype(np.uint8, copy=False)

        ids = [segment.id for segment in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("segment ids must be unique")
        names = [segment.name for segment in self.segments]
        if len(names) != len(set(names)):
            raise ValueError("segment names must be unique")
        defined_ids = set(ids)
        if self.segmentation.size and not segmentation_was_empty_default:
            for start in range(0, self.segmentation.shape[0], 32):
                values = np.unique(self.segmentation[start:start + 32])
                orphaned = [int(value) for value in values if value != 0 and int(value) not in defined_ids]
                if orphaned:
                    raise ValueError(
                        "labelmap references undefined segment id(s): "
                        + ", ".join(str(value) for value in orphaned)
                    )

    def segment_by_id(self, segment_id: int) -> SegmentInfo:
        for segment in self.segments:
            if segment.id == segment_id:
                return segment
        raise KeyError(f"unknown segment id: {segment_id}")

    def segment_by_name(self, name: str) -> SegmentInfo:
        for segment in self.segments:
            if segment.name.casefold() == name.casefold():
                return segment
        raise KeyError(f"unknown segment name: {name}")

    def touch_segmentation(self) -> None:
        """Advance the revision after a labelmap edit (for derived-data caches)."""
        self.segmentation_revision += 1
        self._segment_presence_cache = None

    def close(self) -> None:
        """Release memory maps and temporary storage owned by a loaded case.

        A closed case must not be used again. DICOM-backed SimpleITK images do
        not expose temporary-directory cleanup and remain managed by Python.
        """
        owner = self.native_image
        cleanup = getattr(owner, "cleanup", None)
        if not callable(cleanup):
            return
        maps = []
        for array in (self.volume, self.segmentation):
            current = array
            while current is not None:
                memory_map = getattr(current, "_mmap", None)
                if memory_map is not None:
                    if all(memory_map is not existing for existing in maps):
                        maps.append(memory_map)
                    break
                current = getattr(current, "base", None)
        for memory_map in maps:
            memory_map.close()
        cleanup()
        self.native_image = None

    @property
    def present_segment_ids(self) -> frozenset[int]:
        """Segment IDs with voxels, scanned in bounded Z chunks and cached by revision."""
        if (self._segment_presence_cache is not None
                and self._segment_presence_cache[0] == self.segmentation_revision):
            return self._segment_presence_cache[1]
        assert self.segmentation is not None
        found: set[int] = set()
        expected = {segment.id for segment in self.segments}
        for start in range(0, self.segmentation.shape[0], 32):
            stop = min(self.segmentation.shape[0], start + 32)
            found.update(int(value) for value in np.unique(self.segmentation[start:stop]))
            found.discard(0)
            if found >= expected:
                break
        result = frozenset(found)
        self._segment_presence_cache = (self.segmentation_revision, result)
        return result

    @property
    def has_segmentation(self) -> bool:
        return bool(self.segmentation is not None and np.any(self.segmentation))
