"""Local single-file .lapcase persistence with geometry-preserving arrays."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile
import zipfile

import numpy as np

from .case import Case, SegmentInfo
from .geometry import ImageGeometry


_FORMAT_VERSION = 1
_REQUIRED_MEMBERS = {"manifest.json", "volume.npy", "segmentation.npy"}
_MIB = 1024 * 1024
_MAX_MANIFEST_BYTES = 1 * _MIB
_MAX_VOLUME_MEMBER_BYTES = 768 * _MIB
_MAX_SEGMENTATION_MEMBER_BYTES = 384 * _MIB
_MAX_TOTAL_UNCOMPRESSED_BYTES = _MAX_MANIFEST_BYTES + _MAX_VOLUME_MEMBER_BYTES + _MAX_SEGMENTATION_MEMBER_BYTES
_MAX_ARCHIVE_FILE_BYTES = _MAX_TOTAL_UNCOMPRESSED_BYTES + 16 * _MIB
_MAX_CENTRAL_DIRECTORY_BYTES = 1 * _MIB
_MAX_NPY_HEADER_BYTES = 64 * 1024
_MAX_DIMENSION = 16_384
_MAX_CASE_VOXELS = 380_000_000
_ALLOWED_VOLUME_DTYPES = {
    ("i", 1), ("u", 1), ("i", 2), ("u", 2), ("i", 4), ("u", 4),
    ("f", 4), ("f", 8),
}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.name
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_array(archive: zipfile.ZipFile, member: str, array: np.ndarray) -> None:
    with archive.open(member, "w") as output:
        np.lib.format.write_array(output, np.asarray(array), allow_pickle=False)


def _validate_volume_dtype(value) -> np.dtype:
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("case volume dtype is invalid") from exc
    if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
        raise ValueError("case volume dtype must be a plain numeric scalar dtype")
    if (dtype.kind, dtype.itemsize) not in _ALLOWED_VOLUME_DTYPES:
        raise ValueError(f"case volume dtype is unsupported: {dtype}")
    return dtype


def _read_bounded_manifest(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> dict:
    if info.file_size < 2 or info.file_size > _MAX_MANIFEST_BYTES:
        raise ValueError("case manifest exceeds the allowed size")
    with archive.open(info, "r") as stream:
        raw = stream.read(_MAX_MANIFEST_BYTES + 1)
    if len(raw) != info.file_size or len(raw) > _MAX_MANIFEST_BYTES:
        raise ValueError("case manifest size does not match its archive header")
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("case manifest is not valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("case manifest must be a JSON object")
    return manifest


def _validate_manifest(manifest: dict):
    if manifest.get("format") != "laparoskan-case":
        raise ValueError("file is not a Laparoskan case")
    if int(manifest.get("format_version", 0)) != _FORMAT_VERSION:
        raise ValueError(f"unsupported case format version: {manifest.get('format_version')}")
    geometry_data = manifest.get("geometry")
    if not isinstance(geometry_data, dict):
        raise ValueError("case manifest has no valid geometry object")
    size_data = geometry_data.get("size_xyz")
    if (not isinstance(size_data, list) or len(size_data) != 3
            or any(type(value) is not int or value < 1 or value > _MAX_DIMENSION
                   for value in size_data)):
        raise ValueError("case volume dimensions are invalid or exceed limits")
    size_xyz = tuple(size_data)
    voxel_count = math.prod(size_xyz)
    if voxel_count > _MAX_CASE_VOXELS:
        raise ValueError("case volume exceeds the allowed voxel count")
    geometry = ImageGeometry(
        size_xyz=size_xyz,
        spacing_xyz=tuple(float(v) for v in geometry_data["spacing_xyz"]),
        origin_lps=tuple(float(v) for v in geometry_data["origin_lps"]),
        direction_lps=tuple(float(v) for v in geometry_data["direction_lps"]),
    )
    volume_dtype = _validate_volume_dtype(manifest.get("volume_dtype"))
    if manifest.get("segmentation_dtype") != "uint8":
        raise ValueError("case segmentation dtype must be uint8")
    segments_data = manifest.get("segments")
    if not isinstance(segments_data, list) or len(segments_data) > 255:
        raise ValueError("case segment list is invalid or exceeds the label limit")
    segments = tuple(
        SegmentInfo(
            id=int(item["id"]),
            name=str(item["name"]),
            color_rgb=tuple(int(v) for v in item["color_rgb"]),
        )
        for item in segments_data
        if isinstance(item, dict)
    )
    if len(segments) != len(segments_data):
        raise ValueError("case segment list contains an invalid entry")
    if len({item.id for item in segments}) != len(segments):
        raise ValueError("case segment IDs must be unique")
    if len({item.name for item in segments}) != len(segments):
        raise ValueError("case segment names must be unique")
    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("case metadata must be a JSON object")
    name = manifest.get("name")
    case_id = manifest.get("case_id")
    if not isinstance(name, str) or not name or not isinstance(case_id, str) or not case_id:
        raise ValueError("case name and ID must be non-empty strings")
    return geometry, segments, metadata, volume_dtype, name, case_id


def _inspect_npy_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    expected_shape: tuple[int, int, int],
    expected_dtype: np.dtype,
    max_member_bytes: int,
) -> None:
    if info.file_size <= 0 or info.file_size > max_member_bytes:
        raise ValueError(f"{info.filename} exceeds the allowed uncompressed size")
    try:
        with archive.open(info, "r") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(
                    stream, max_header_size=_MAX_NPY_HEADER_BYTES
                )
            elif version == (2, 0):
                shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(
                    stream, max_header_size=_MAX_NPY_HEADER_BYTES
                )
            else:
                raise ValueError(f"unsupported NPY format version: {version}")
            header_bytes = stream.tell()
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ValueError(f"{info.filename} has an invalid NPY header") from exc
    dtype = np.dtype(dtype)
    if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
        raise ValueError(f"{info.filename} uses an unsafe NPY dtype")
    if tuple(shape) != expected_shape or dtype != expected_dtype:
        raise ValueError(f"{info.filename} shape or dtype does not match the case manifest")
    expected_size = header_bytes + math.prod(expected_shape) * dtype.itemsize
    if expected_size != info.file_size:
        raise ValueError(f"{info.filename} byte size does not match its NPY shape")


def _copy_member_bounded(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo, destination: Path, max_member_bytes: int
) -> None:
    copied = 0
    with archive.open(info, "r") as source, destination.open("wb") as output:
        while True:
            chunk = source.read(8 * _MIB)
            if not chunk:
                break
            copied += len(chunk)
            if copied > max_member_bytes or copied > info.file_size:
                raise ValueError(f"{info.filename} expanded beyond its allowed size")
            output.write(chunk)
    if copied != info.file_size:
        raise ValueError(f"{info.filename} uncompressed size does not match its archive header")


def _preflight_zip_directory(source: Path, archive_bytes: int) -> None:
    """Reject oversized ZIP directories/member counts before ZipFile allocates entries."""
    tail_bytes = min(archive_bytes, 22 + 65_535)
    with source.open("rb") as stream:
        stream.seek(archive_bytes - tail_bytes)
        tail = stream.read(tail_bytes)
    end_offset = tail.rfind(b"PK\x05\x06")
    while end_offset >= 0:
        if end_offset + 22 <= len(tail):
            (signature, disk_number, directory_disk, disk_entries, total_entries,
             directory_size, directory_offset, comment_size) = struct.unpack_from(
                "<4s4H2LH", tail, end_offset
            )
            if (signature == b"PK\x05\x06"
                    and end_offset + 22 + comment_size == len(tail)):
                if disk_number != 0 or directory_disk != 0 or disk_entries != total_entries:
                    raise ValueError("multi-disk case archives are not supported")
                if total_entries != len(_REQUIRED_MEMBERS):
                    raise ValueError("case archive must contain exactly three members")
                if directory_size > _MAX_CENTRAL_DIRECTORY_BYTES:
                    raise ValueError("case archive directory exceeds the allowed size")
                if directory_offset + directory_size > archive_bytes:
                    raise ValueError("case archive directory points outside the file")
                return
        end_offset = tail.rfind(b"PK\x05\x06", 0, end_offset)
    raise ValueError("case archive has no valid ZIP end record")


def _close_array_mapping(array) -> None:
    current = array
    while current is not None:
        memory_map = getattr(current, "_mmap", None)
        if memory_map is not None:
            memory_map.close()
            return
        current = getattr(current, "base", None)


def save_case(case: Case, path: str | Path) -> str:
    """Atomically save the full volume, segmentation, and patient-space geometry."""
    destination = Path(path).expanduser()
    if destination.suffix.lower() != ".lapcase":
        destination = destination.with_suffix(".lapcase")
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert case.segmentation is not None
    if case.volume.ndim != 3 or case.volume.size > _MAX_CASE_VOXELS:
        raise ValueError("case volume dimensions exceed the .lapcase limits")
    _validate_volume_dtype(case.volume.dtype)
    if case.volume.nbytes + _MAX_NPY_HEADER_BYTES > _MAX_VOLUME_MEMBER_BYTES:
        raise ValueError("case volume exceeds the .lapcase size limit")
    if case.segmentation.nbytes + _MAX_NPY_HEADER_BYTES > _MAX_SEGMENTATION_MEMBER_BYTES:
        raise ValueError("case segmentation exceeds the .lapcase size limit")
    manifest = {
        "format": "laparoskan-case",
        "format_version": _FORMAT_VERSION,
        "case_id": case.case_id,
        "name": case.name,
        "geometry": {
            "size_xyz": list(case.geometry.size_xyz),
            "spacing_xyz": list(case.geometry.spacing_xyz),
            "origin_lps": list(case.geometry.origin_lps),
            "direction_lps": list(case.geometry.direction_lps),
        },
        "volume_dtype": str(case.volume.dtype),
        "segmentation_dtype": "uint8",
        "segments": [
            {"id": segment.id, "name": segment.name, "color_rgb": list(segment.color_rgb)}
            for segment in case.segments
        ],
        "metadata": _json_safe(case.metadata),
    }
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp",
                                             dir=destination.parent)
    os.close(descriptor)
    try:
        with zipfile.ZipFile(temp_name, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, allow_nan=False))
            _write_array(archive, "volume.npy", case.volume)
            _write_array(archive, "segmentation.npy", case.segmentation)
        os.replace(temp_name, destination)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    return str(destination)


def load_case(path: str | Path) -> Case:
    """Open a bounded `.lapcase` after validating archive metadata and NPY headers."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    archive_bytes = source.stat().st_size
    if archive_bytes <= 0 or archive_bytes > _MAX_ARCHIVE_FILE_BYTES:
        raise ValueError("case archive exceeds the allowed file size")
    _preflight_zip_directory(source, archive_bytes)
    temp_directory = None
    try:
        with zipfile.ZipFile(source, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(infos) != len(_REQUIRED_MEMBERS) or set(names) != _REQUIRED_MEMBERS:
                raise ValueError("case archive must contain exactly one manifest and two array members")
            if len(set(names)) != len(names):
                raise ValueError("case archive contains duplicate member names")
            by_name = {info.filename: info for info in infos}
            for info in infos:
                if info.flag_bits & 0x1:
                    raise ValueError("encrypted case archive members are not supported")
                if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                    raise ValueError("case archive uses an unsupported compression method")
                if info.file_size < 0 or info.compress_size < 0:
                    raise ValueError("case archive contains invalid member sizes")
            manifest_info = by_name["manifest.json"]
            volume_info = by_name["volume.npy"]
            segmentation_info = by_name["segmentation.npy"]
            if manifest_info.file_size > _MAX_MANIFEST_BYTES:
                raise ValueError("case manifest exceeds the allowed size")
            if volume_info.file_size > _MAX_VOLUME_MEMBER_BYTES:
                raise ValueError("case volume member exceeds the allowed size")
            if segmentation_info.file_size > _MAX_SEGMENTATION_MEMBER_BYTES:
                raise ValueError("case segmentation member exceeds the allowed size")
            total_uncompressed = sum(info.file_size for info in infos)
            if total_uncompressed > _MAX_TOTAL_UNCOMPRESSED_BYTES:
                raise ValueError("case archive exceeds the total uncompressed size limit")

            manifest = _read_bounded_manifest(archive, manifest_info)
            geometry, segments, metadata, volume_dtype, name, case_id = _validate_manifest(manifest)
            shape_zyx = tuple(reversed(geometry.size_xyz))
            expected_segmentation_dtype = np.dtype(np.uint8)
            _inspect_npy_member(
                archive, volume_info, shape_zyx, volume_dtype, _MAX_VOLUME_MEMBER_BYTES
            )
            _inspect_npy_member(
                archive, segmentation_info, shape_zyx, expected_segmentation_dtype,
                _MAX_SEGMENTATION_MEMBER_BYTES,
            )

            extraction_bytes = volume_info.file_size + segmentation_info.file_size
            required_free = extraction_bytes + 16 * _MIB
            if shutil.disk_usage(tempfile.gettempdir()).free < required_free:
                raise OSError("insufficient temporary disk space to open this case")
            temp_directory = tempfile.TemporaryDirectory(prefix="laparoskan-case-")
            output_root = Path(temp_directory.name)
            _copy_member_bounded(
                archive, volume_info, output_root / "volume.npy", _MAX_VOLUME_MEMBER_BYTES
            )
            _copy_member_bounded(
                archive, segmentation_info, output_root / "segmentation.npy",
                _MAX_SEGMENTATION_MEMBER_BYTES,
            )

        volume = np.load(output_root / "volume.npy", mmap_mode="r", allow_pickle=False)
        # Copy-on-write keeps the labelmap editable without eagerly duplicating
        # a large array; writes remain local until the next case save.
        segmentation = np.load(
            output_root / "segmentation.npy", mmap_mode="c", allow_pickle=False
        )
        return Case(
            name=name,
            volume=volume,
            geometry=geometry,
            segmentation=segmentation,
            segments=segments,
            case_id=case_id,
            metadata=metadata,
            source=str(source),
            native_image=temp_directory,
        )
    except Exception:
        if temp_directory is not None:
            _close_array_mapping(locals().get("segmentation"))
            _close_array_mapping(locals().get("volume"))
            temp_directory.cleanup()
        raise
