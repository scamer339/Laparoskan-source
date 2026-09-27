"""Editable medical labelmap operations kept separate from display meshes."""

from __future__ import annotations

import numpy as np
from collections import deque

from .case import Case, SegmentInfo


class RegionGrowLimitError(ValueError):
    """The connected component exceeded the configured voxel safety cap."""


def create_segment(
    case: Case,
    name: str,
    color_rgb: tuple[int, int, int] = (64, 150, 235),
) -> SegmentInfo:
    if any(segment.name.casefold() == name.strip().casefold() for segment in case.segments):
        raise ValueError(f"segment already exists: {name}")
    used = {segment.id for segment in case.segments}
    available = next((segment_id for segment_id in range(1, 256) if segment_id not in used), None)
    if available is None:
        raise ValueError("the labelmap already contains the maximum of 255 structures")
    info = SegmentInfo(available, name.strip(), color_rgb)
    case.segments = (*case.segments, info)
    return info


def paint_voxel(
    case: Case,
    physical_lps: tuple[float, float, float],
    label_id: int,
    radius_mm: float = 4.0,
    erase: bool = False,
) -> int:
    """Paint or erase a physical-mm spherical brush in the medical labelmap."""
    if not np.isfinite(radius_mm) or radius_mm < 0:
        raise ValueError("brush radius must be a finite non-negative value")
    if not erase:
        case.segment_by_id(label_id)
    assert case.segmentation is not None
    center_xyz = case.geometry.physical_to_index(physical_lps)
    size_x, size_y, size_z = case.geometry.size_xyz
    radius_index = np.ceil(radius_mm / np.asarray(case.geometry.spacing_xyz)).astype(int)
    x0 = max(0, int(np.floor(center_xyz[0])) - radius_index[0])
    x1 = min(size_x, int(np.ceil(center_xyz[0])) + radius_index[0] + 1)
    y0 = max(0, int(np.floor(center_xyz[1])) - radius_index[1])
    y1 = min(size_y, int(np.ceil(center_xyz[1])) + radius_index[1] + 1)
    z0 = max(0, int(np.floor(center_xyz[2])) - radius_index[2])
    z1 = min(size_z, int(np.ceil(center_xyz[2])) + radius_index[2] + 1)
    if x0 >= x1 or y0 >= y1 or z0 >= z1:
        return 0

    sx, sy, sz = case.geometry.spacing_xyz
    zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
    brush = (
        ((xx - center_xyz[0]) * sx) ** 2
        + ((yy - center_xyz[1]) * sy) ** 2
        + ((zz - center_xyz[2]) * sz) ** 2
        <= radius_mm * radius_mm
    )
    local = case.segmentation[z0:z1, y0:y1, x0:x1]
    replacement = 0 if erase else label_id
    changed = brush & (local != replacement)
    count = int(np.count_nonzero(changed))
    if count:
        local[changed] = replacement
        case.touch_segmentation()
    return count


def threshold_segment(
    case: Case,
    label_id: int,
    lower_hu: float,
    upper_hu: float,
    region: tuple[slice, slice, slice] | None = None,
) -> int:
    """Assign user-selected CT intensity range to a segment, optionally in ZYX ROI."""
    case.segment_by_id(label_id)
    if not np.isfinite(lower_hu) or not np.isfinite(upper_hu) or lower_hu > upper_hu:
        raise ValueError("intensity range must be finite and lower <= upper")
    assert case.segmentation is not None
    if region is None:
        region = (slice(None), slice(None), slice(None))
    if len(region) != 3 or not all(isinstance(value, slice) for value in region):
        raise ValueError("region must be a Z,Y,X tuple of slices")
    volume_view = case.volume[region]
    label_view = case.segmentation[region]
    selected = (volume_view >= lower_hu) & (volume_view <= upper_hu)
    changed = selected & (label_view != label_id)
    count = int(np.count_nonzero(changed))
    if count:
        label_view[changed] = label_id
        case.touch_segmentation()
    return count


def region_grow_segment(
    case: Case,
    label_id: int,
    seed_lps: tuple[float, float, float],
    tolerance_hu: float = 35.0,
    max_voxels: int = 250_000,
) -> int:
    """Flood-fill a 6-connected CT component within seed HU ± tolerance.

    Growth is bounded by ``max_voxels`` and commits only after the entire
    component fits under that cap. Voxels assigned to another segment block
    growth and are never overwritten; background and the selected label may be
    traversed. The returned count is the number of newly assigned voxels.
    """
    case.segment_by_id(label_id)
    if str(case.metadata.get("modality", "CT")).upper() != "CT":
        raise ValueError("HU region growing is available for CT cases only")
    if not np.isfinite(tolerance_hu) or tolerance_hu < 0:
        raise ValueError("tolerance_hu must be finite and non-negative")
    try:
        voxel_limit = int(max_voxels)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("max_voxels must be a positive integer") from exc
    if isinstance(max_voxels, bool) or voxel_limit != max_voxels or voxel_limit < 1:
        raise ValueError("max_voxels must be a positive integer")
    assert case.segmentation is not None
    seed_index = np.rint(case.geometry.physical_to_index(seed_lps)).astype(np.int64)
    size_x, size_y, size_z = case.geometry.size_xyz
    if (seed_index[0] < 0 or seed_index[0] >= size_x
            or seed_index[1] < 0 or seed_index[1] >= size_y
            or seed_index[2] < 0 or seed_index[2] >= size_z):
        raise ValueError("region-growing seed lies outside the volume")

    volume_flat = case.volume.ravel(order="C")
    labels_flat = case.segmentation.ravel(order="C")
    seed_flat = (int(seed_index[2]) * size_y + int(seed_index[1])) * size_x + int(seed_index[0])
    current_label = int(labels_flat[seed_flat])
    if current_label not in (0, label_id):
        raise ValueError("region-growing seed lies inside another marked structure")
    seed_value = float(volume_flat[seed_flat])
    if not np.isfinite(seed_value):
        raise ValueError("region-growing seed has no finite image intensity")
    low, high = seed_value - float(tolerance_hu), seed_value + float(tolerance_hu)
    plane_size = size_x * size_y
    accepted = {seed_flat}
    frontier = deque((seed_flat,))

    while frontier:
        flat_index = frontier.popleft()
        z, remainder = divmod(flat_index, plane_size)
        y, x = divmod(remainder, size_x)
        neighbours = []
        if x > 0:
            neighbours.append(flat_index - 1)
        if x + 1 < size_x:
            neighbours.append(flat_index + 1)
        if y > 0:
            neighbours.append(flat_index - size_x)
        if y + 1 < size_y:
            neighbours.append(flat_index + size_x)
        if z > 0:
            neighbours.append(flat_index - plane_size)
        if z + 1 < size_z:
            neighbours.append(flat_index + plane_size)
        for neighbour in neighbours:
            if neighbour in accepted:
                continue
            neighbor_label = int(labels_flat[neighbour])
            if neighbor_label not in (0, label_id):
                continue
            intensity = float(volume_flat[neighbour])
            if not np.isfinite(intensity) or intensity < low or intensity > high:
                continue
            accepted.add(neighbour)
            if len(accepted) > voxel_limit:
                raise RegionGrowLimitError(
                    f"Connected region exceeded the {voxel_limit:,}-voxel limit; "
                    "no voxels were changed. Narrow the HU tolerance or choose another seed."
                )
            frontier.append(neighbour)

    flat_indices = np.fromiter(accepted, dtype=np.int64)
    new_indices = flat_indices[labels_flat[flat_indices] == 0]
    count = int(new_indices.size)
    if count:
        # Assign through the original array: ravel() can return a copy for a
        # non-contiguous segmentation supplied by a caller.
        indices_zyx = np.unravel_index(new_indices, case.segmentation.shape, order="C")
        case.segmentation[indices_zyx] = label_id
        case.touch_segmentation()
    return count


def clear_segment(case: Case, label_id: int) -> int:
    case.segment_by_id(label_id)
    assert case.segmentation is not None
    selected = case.segmentation == label_id
    count = int(np.count_nonzero(selected))
    if count:
        case.segmentation[selected] = 0
        case.touch_segmentation()
    return count


def segment_voxel_counts(case: Case) -> dict[int, int]:
    assert case.segmentation is not None
    values, counts = np.unique(case.segmentation, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts) if value != 0}


def build_surface(case: Case, label_id: int):
    """Build a smoothed display-only VTK surface from one medical label.

    Physical measurements must use the unchanged labelmap and ImageGeometry;
    this mesh is only for the interactive 3D display.
    """
    case.segment_by_id(label_id)
    assert case.segmentation is not None
    try:
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk
    except ImportError as exc:
        raise RuntimeError("VTK is required to reconstruct a 3D surface") from exc

    if not np.any(case.segmentation == label_id):
        return vtk.vtkPolyData()

    image = vtk.vtkImageData()
    image.SetDimensions(*case.geometry.size_xyz)
    image.SetOrigin(*case.geometry.origin_lps)
    image.SetSpacing(*case.geometry.spacing_xyz)
    matrix = vtk.vtkMatrix3x3()
    direction = case.geometry.direction_matrix
    for row in range(3):
        for column in range(3):
            matrix.SetElement(row, column, float(direction[row, column]))
    image.SetDirectionMatrix(matrix)
    flat = np.ravel(case.segmentation, order="C")
    scalars = numpy_to_vtk(flat, deep=False, array_type=vtk.VTK_UNSIGNED_CHAR)
    scalars.SetName("Medical segmentation labels")
    image.GetPointData().SetScalars(scalars)
    # numpy_to_vtk(deep=False) shares memory. Keep both backing objects alive
    # for as long as the VTK image is retained by the contour pipeline.
    image._laparoskan_numpy_reference = flat
    image._laparoskan_scalar_reference = scalars

    contour = vtk.vtkDiscreteFlyingEdges3D()
    contour.SetInputData(image)
    contour.SetNumberOfContours(1)
    contour.SetValue(0, int(label_id))
    contour.ComputeNormalsOff()
    contour.Update()
    if contour.GetOutput().GetNumberOfPoints() == 0:
        return vtk.vtkPolyData()

    smooth = vtk.vtkWindowedSincPolyDataFilter()
    smooth.SetInputConnection(contour.GetOutputPort())
    smooth.SetNumberOfIterations(12)
    smooth.SetPassBand(0.12)
    smooth.BoundarySmoothingOff()
    smooth.FeatureEdgeSmoothingOff()
    smooth.NonManifoldSmoothingOn()
    smooth.NormalizeCoordinatesOn()
    smooth.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(smooth.GetOutputPort())
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()

    output = vtk.vtkPolyData()
    output.DeepCopy(normals.GetOutput())
    return output
