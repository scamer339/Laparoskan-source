"""Trajectory measurements using only volume geometry and medical label voxels."""

from __future__ import annotations

from dataclasses import dataclass
import math
from itertools import product

import numpy as np

from .case import Case


_BODY_SURFACE_NAMES = frozenset({"body contour", "skin", "body surface", "skin surface"})


def _is_body_surface_segment(segment) -> bool:
    return segment.name.strip().casefold() in _BODY_SURFACE_NAMES


def _externally_exposed_body_voxel(case: Case, xyz: tuple[int, int, int]) -> bool:
    """Conservatively recognize surface voxels with a zero ray to image edge.

    Any different nonzero label blocks a ray. This intentionally rejects some
    concave surfaces and interior cavities. Touching the image edge alone does
    not establish skin: a candidate still needs an interior face-neighbor
    background voxel whose axis ray stays background to the image boundary.
    """
    assert case.segmentation is not None
    x, y, z = xyz
    size_x, size_y, size_z = case.geometry.size_xyz
    labels = case.segmentation
    rays = (
        labels[z, y, :x], labels[z, y, x + 1:],
        labels[z, :y, x], labels[z, y + 1:, x],
        labels[:z, y, x], labels[z + 1:, y, x],
    )
    return any(ray.size > 0 and bool(np.all(ray == 0)) for ray in rays)


@dataclass(frozen=True)
class NeedlePlan:
    entry_lps: tuple[float, float, float]
    target_lps: tuple[float, float, float]
    direction_lps: tuple[float, float, float]
    depth_mm: float
    angle_degrees: float
    target_hit: bool
    collided: bool
    collision_labels: tuple[str, ...]
    clearances_mm: dict[str, float]
    closest_clearance_mm: float | None
    clearance_complete: bool
    clearance_limit_mm: float
    entry_on_body_surface: bool | None
    unmarked_critical_labels: tuple[str, ...]


@dataclass(frozen=True)
class TrajectoryCandidate:
    """One deterministic educational suggestion, with its measured plan."""

    rank: int
    plan: NeedlePlan
    clearance_lower_bound_mm: float | None
    ranking_summary: str


def _point_lps(point) -> np.ndarray:
    result = np.asarray(point, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("trajectory points must be three finite LPS coordinates")
    return result


def _label_at_lps(case: Case, point: np.ndarray) -> int:
    assert case.segmentation is not None
    index = np.rint(case.geometry.physical_to_index(point)).astype(np.int64)
    size_x, size_y, size_z = case.geometry.size_xyz
    if np.any(index < 0) or index[0] >= size_x or index[1] >= size_y or index[2] >= size_z:
        return 0
    return int(case.segmentation[index[2], index[1], index[0]])


def _entry_is_near_segmented_surface(case: Case, point: np.ndarray) -> bool | None:
    """Check proximity to exposed body-mask voxel cells.

    A voxel is exposed only when a face-neighbor is background connected to
    the image boundary by an all-background axis ray. Image edges alone do not
    verify skin. A point is accepted within one voxel diagonal of an exposed
    cell. This remains a conservative discrete labelmap check, not a continuous
    skin-surface reconstruction.
    """
    surface_segment = next(
        (segment for segment in case.segments if _is_body_surface_segment(segment)),
        None,
    )
    if surface_segment is None or surface_segment.id not in case.present_segment_ids:
        return None
    assert case.segmentation is not None
    index_xyz = case.geometry.physical_to_index(point)
    tolerance_mm = case.geometry.voxel_diagonal_mm
    spacing = np.asarray(case.geometry.spacing_xyz, dtype=np.float64)
    # The halo lets us test neighbors around every queried voxel without
    # mistaking the local search-ROI edge for an image-volume edge.
    radius_xyz = np.ceil(tolerance_mm / spacing + 0.5).astype(int) + 1
    center = np.rint(index_xyz).astype(int)
    size_x, size_y, size_z = case.geometry.size_xyz
    x0, y0, z0 = np.maximum(center - radius_xyz, (0, 0, 0))
    x1, y1, z1 = np.minimum(center + radius_xyz + 1, (size_x, size_y, size_z))
    if x0 >= x1 or y0 >= y1 or z0 >= z1:
        return False
    local = case.segmentation[z0:z1, y0:y1, x0:x1]
    candidates_zyx = np.argwhere(local == surface_segment.id)
    if not len(candidates_zyx):
        return False
    candidates_xyz = candidates_zyx[:, ::-1].astype(np.int64)
    candidates_xyz += np.asarray((x0, y0, z0), dtype=np.int64)
    size_xyz = np.asarray((size_x, size_y, size_z), dtype=np.int64)
    exposed = np.zeros(len(candidates_xyz), dtype=bool)
    for delta in (
        (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)
    ):
        neighbors_xyz = candidates_xyz + np.asarray(delta, dtype=np.int64)
        inside = np.all((neighbors_xyz >= 0) & (neighbors_xyz < size_xyz[None, :]), axis=1)
        exposed |= ~inside
        if np.any(inside):
            neighbor_labels = case.segmentation[
                neighbors_xyz[inside, 2], neighbors_xyz[inside, 1], neighbors_xyz[inside, 0]
            ]
            exposed[inside] |= neighbor_labels != surface_segment.id
    candidates_xyz = candidates_xyz[exposed]
    exterior = np.fromiter(
        (_externally_exposed_body_voxel(case, tuple(int(v) for v in xyz))
         for xyz in candidates_xyz),
        dtype=bool,
        count=len(candidates_xyz),
    )
    candidates_xyz = candidates_xyz[exterior].astype(np.float64)
    if not len(candidates_xyz):
        return False
    # Distance from a continuous point to each closed physical voxel cell.
    outside = np.maximum(np.abs(candidates_xyz - index_xyz[None, :]) - 0.5, 0.0)
    distances = np.linalg.norm(outside * spacing[None, :], axis=1)
    return bool(float(distances.min()) <= tolerance_mm + 1e-9)


def _cells_touching(point_xyz: np.ndarray, size_xyz: tuple[int, int, int]):
    """Return all closed voxel cells containing a continuous XYZ point."""
    choices = []
    for coordinate, size in zip(point_xyz, size_xyz):
        shifted = float(coordinate) + 0.5
        boundary = round(shifted)
        if math.isclose(shifted, boundary, rel_tol=0.0, abs_tol=1e-9):
            axis = (int(boundary) - 1, int(boundary))
        else:
            axis = (int(math.floor(shifted)),)
        choices.append(tuple(index for index in axis if 0 <= index < size))
    if any(not axis for axis in choices):
        return ()
    return tuple(product(*choices))


def _segment_voxel_supercover(start_xyz: np.ndarray, end_xyz: np.ndarray,
                              size_xyz: tuple[int, int, int]) -> set[tuple[int, int, int]]:
    """Enumerate every closed voxel cell touched by a segment in index space.

    Event parameters are generated at every half-integer grid plane; cells at
    each event and in each open interval are included.  This handles corner
    grazing and lines lying along voxel boundaries without sampling gaps.
    """
    start = np.asarray(start_xyz, dtype=np.float64)
    end = np.asarray(end_xyz, dtype=np.float64)
    delta = end - start
    low, high = np.full(3, -0.5), np.asarray(size_xyz, dtype=np.float64) - 0.5
    t_min, t_max = 0.0, 1.0
    for axis in range(3):
        if abs(delta[axis]) <= 1e-15:
            if start[axis] < low[axis] or start[axis] > high[axis]:
                return set()
            continue
        t0 = (low[axis] - start[axis]) / delta[axis]
        t1 = (high[axis] - start[axis]) / delta[axis]
        t_min = max(t_min, min(t0, t1))
        t_max = min(t_max, max(t0, t1))
        if t_min > t_max:
            return set()

    clipped_start = start + t_min * delta
    clipped_end = start + t_max * delta
    clipped_delta = clipped_end - clipped_start
    events = {0.0, 1.0}
    for axis in range(3):
        if abs(clipped_delta[axis]) <= 1e-15:
            continue
        first = math.floor(min(clipped_start[axis], clipped_end[axis]) + 0.5) - 1
        last = math.ceil(max(clipped_start[axis], clipped_end[axis]) + 0.5) + 1
        for plane_index in range(first, last + 1):
            boundary = plane_index - 0.5
            t = (boundary - clipped_start[axis]) / clipped_delta[axis]
            if 1e-12 < t < 1.0 - 1e-12:
                events.add(float(t))

    ordered = sorted(events)
    cells: set[tuple[int, int, int]] = set()
    for event in ordered:
        point = clipped_start + event * clipped_delta
        cells.update(_cells_touching(point, size_xyz))
    for left, right in zip(ordered, ordered[1:]):
        if right - left <= 1e-12:
            continue
        point = clipped_start + ((left + right) * 0.5) * clipped_delta
        cells.update(_cells_touching(point, size_xyz))
    return cells


def _critical_segments(case: Case, requested) -> tuple:
    if requested is None:
        return tuple(
            segment for segment in case.segments
            if segment.name.strip().casefold() != "abscess" and not _is_body_surface_segment(segment)
        )
    result = []
    for value in requested:
        result.append(case.segment_by_id(value) if isinstance(value, int) else case.segment_by_name(str(value)))
    return tuple(result)


def _segment_distance_candidates(case: Case, label_ids: set[int], entry: np.ndarray,
                                 target: np.ndarray, limit_mm: float):
    """Yield patient-space centers of marked voxels in a bounded search region."""
    assert case.segmentation is not None
    direction = target - entry
    # Sample a capsule's axis to bound an LPS box around the complete trajectory.
    length = float(np.linalg.norm(direction))
    count = max(2, int(math.ceil(length / max(limit_mm * 0.5, 1.0))) + 1)
    axis_points = entry[None, :] + np.linspace(0.0, 1.0, count)[:, None] * direction[None, :]
    # Include centers whose voxel bounding sphere could yield a surface lower
    # bound within the reporting radius.
    padding = limit_mm + case.geometry.voxel_diagonal_mm * 0.5
    lower_lps = axis_points.min(axis=0) - padding
    upper_lps = axis_points.max(axis=0) + padding
    world_corners = np.asarray(list(product(*zip(lower_lps, upper_lps))), dtype=np.float64)
    index_corners = case.geometry.physical_to_index(world_corners)
    size_x, size_y, size_z = case.geometry.size_xyz
    low = np.floor(index_corners.min(axis=0)).astype(int)
    high = np.ceil(index_corners.max(axis=0)).astype(int) + 1
    x0, y0, z0 = np.maximum(low, (0, 0, 0))
    x1, y1, z1 = np.minimum(high, (size_x, size_y, size_z))
    if x0 >= x1 or y0 >= y1 or z0 >= z1:
        return np.empty((0, 3), dtype=np.float64), True

    roi = case.segmentation[z0:z1, y0:y1, x0:x1]
    selected = np.isin(roi, np.fromiter(label_ids, dtype=np.uint8))
    local_zyx = np.argwhere(selected)
    if not len(local_zyx):
        # No voxels within the bounded search region; the actual clearance is
        # at least the search radius and is reported as a lower bound.
        return np.empty((0, 3), dtype=np.float64), False
    absolute_xyz = local_zyx[:, ::-1] + np.asarray((x0, y0, z0), dtype=np.int64)
    points_lps = case.geometry.index_to_physical(absolute_xyz)
    labels = roi[selected]
    complete = True
    return np.column_stack((points_lps, labels.astype(np.float64))), complete


def plan_trajectory(
    case: Case,
    entry_lps: tuple[float, float, float],
    target_lps: tuple[float, float, float],
    critical_labels=None,
    search_radius_mm: float = 30.0,
) -> NeedlePlan:
    """Measure a line in LPS mm and query intersections/clearances to label voxels.

    Collision checks use a continuous-index voxel supercover, including cells
    touched only at edges or corners. Clearance searches use segment voxel
    centres in a bounded physical neighbourhood and subtract half the voxel
    diagonal to report a conservative surface-distance lower bound. A value at
    the search limit is unresolved beyond that radius. Entry validation is
    available only when a body-contour/skin label has voxels.
    """
    if not np.isfinite(search_radius_mm) or search_radius_mm <= 0:
        raise ValueError("search_radius_mm must be positive and finite")
    entry, target = _point_lps(entry_lps), _point_lps(target_lps)
    vector = target - entry
    depth = float(np.linalg.norm(vector))
    if depth <= 1e-8:
        raise ValueError("entry and target must be distinct points")
    direction = vector / depth
    # Elevation relative to the axial plane.  This is geometric orientation,
    # not an assessment of clinical suitability.
    angle = math.degrees(math.asin(min(1.0, abs(float(direction[2])))))

    target_segment = next(
        (segment for segment in case.segments if segment.name.casefold() == "abscess"),
        case.segments[0] if case.segments else None,
    )
    target_hit = bool(target_segment and _label_at_lps(case, target) == target_segment.id)
    critical = tuple(segment for segment in _critical_segments(case, critical_labels)
                     if target_segment is None or segment.id != target_segment.id)
    present_ids = case.present_segment_ids
    unmarked_critical = tuple(segment.name for segment in critical
                              if segment.id not in present_ids)
    marked_critical = tuple(segment for segment in critical if segment.id in present_ids)
    assert case.segmentation is not None

    index_start = case.geometry.physical_to_index(entry)
    index_end = case.geometry.physical_to_index(target)
    touched_cells = _segment_voxel_supercover(index_start, index_end, case.geometry.size_xyz)
    collision_ids = {
        int(case.segmentation[z, y, x])
        for x, y, z in touched_cells
        if int(case.segmentation[z, y, x]) != 0
    }
    collision_ids.intersection_update(segment.id for segment in marked_critical)
    by_id = {segment.id: segment for segment in marked_critical}
    collision_names = tuple(by_id[value].name for value in sorted(collision_ids))

    clearances: dict[str, float] = {}
    closest = None
    complete = True
    critical_ids = {segment.id for segment in marked_critical}
    voxel_radius = case.geometry.voxel_diagonal_mm * 0.5
    if critical_ids:
        candidates, found_complete = _segment_distance_candidates(
            case, critical_ids, entry, target, float(search_radius_mm)
        )
        complete = found_complete and not bool(unmarked_critical)
        if candidates.size:
            candidate_points = candidates[:, :3]
            candidate_labels = candidates[:, 3].astype(np.uint8)
            displacement = candidate_points - entry[None, :]
            projection_mm = np.einsum("ij,j->i", displacement, vector, optimize=False)
            if not np.all(np.isfinite(projection_mm)):
                raise ValueError("clearance projection produced non-finite values")
            projection = np.clip(projection_mm / (depth * depth), 0.0, 1.0)
            closest_points = entry[None, :] + projection[:, None] * vector[None, :]
            center_distances = np.linalg.norm(candidate_points - closest_points, axis=1)
            for segment in marked_critical:
                selected = candidate_labels == segment.id
                if np.any(selected):
                    center_distance = float(center_distances[selected].min())
                    if center_distance <= search_radius_mm + voxel_radius:
                        clearances[segment.name] = max(0.0, center_distance - voxel_radius)
                    else:
                        clearances[segment.name] = float(search_radius_mm)
                        complete = False
                else:
                    clearances[segment.name] = float(search_radius_mm)
                    complete = False
        else:
            clearances = {segment.name: float(search_radius_mm) for segment in marked_critical}
        if clearances:
            closest = min(clearances.values())
    else:
        complete = not bool(unmarked_critical)

    return NeedlePlan(
        entry_lps=tuple(float(v) for v in entry),
        target_lps=tuple(float(v) for v in target),
        direction_lps=tuple(float(v) for v in direction),
        depth_mm=depth,
        angle_degrees=angle,
        target_hit=target_hit,
        collided=bool(collision_names),
        collision_labels=collision_names,
        clearances_mm=clearances,
        closest_clearance_mm=closest,
        clearance_complete=complete,
        clearance_limit_mm=float(search_radius_mm),
        entry_on_body_surface=_entry_is_near_segmented_surface(case, entry),
        unmarked_critical_labels=unmarked_critical,
    )


def _sample_surface_entries(case: Case, max_entry_points: int) -> list[tuple[float, float, float]]:
    """Take a deterministic, memory-bounded sample from a body surface label."""
    if max_entry_points < 1:
        raise ValueError("max_entry_points must be positive")
    surface_segment = next(
        (segment for segment in case.segments if _is_body_surface_segment(segment)),
        None,
    )
    if surface_segment is None or surface_segment.id not in case.present_segment_ids:
        return []
    assert case.segmentation is not None
    size_z = case.geometry.size_xyz[2]
    present_slices: list[int] = []
    # Scan in small Z blocks so sampling remains bounded in temporary memory.
    for start in range(0, size_z, 16):
        block = case.segmentation[start:min(size_z, start + 16)]
        present_slices.extend(
            start + int(offset)
            for offset in np.flatnonzero(np.any(block == surface_segment.id, axis=(1, 2)))
        )
    if not present_slices:
        return []
    # Roughly four XY samples per selected slice spread candidates around the
    # shell while bounding each temporary coordinate array to one 2D slice.
    z_count = min(len(present_slices), max(1, int(math.ceil(max_entry_points / 4))))
    z_indices = np.unique(np.rint(np.linspace(0, len(present_slices) - 1, z_count)).astype(int))
    z_values = [present_slices[int(index)] for index in z_indices]
    quota = max(1, int(math.ceil(max_entry_points / len(z_values))))
    sampled_xyz: list[tuple[int, int, int]] = []
    for z in z_values:
        z = int(z)
        mask = case.segmentation[z] == surface_segment.id
        boundary = np.zeros_like(mask, dtype=bool)
        # A body-mask voxel is entry-eligible when any face-neighbor is
        # outside that same mask. Out-of-volume neighbors count as exterior.
        boundary[:, 0] = mask[:, 0]
        boundary[:, -1] |= mask[:, -1]
        boundary[0, :] |= mask[0, :]
        boundary[-1, :] |= mask[-1, :]
        boundary[1:, :] |= mask[1:, :] & ~mask[:-1, :]
        boundary[:-1, :] |= mask[:-1, :] & ~mask[1:, :]
        boundary[:, 1:] |= mask[:, 1:] & ~mask[:, :-1]
        boundary[:, :-1] |= mask[:, :-1] & ~mask[:, 1:]
        if z == 0:
            boundary |= mask
        else:
            boundary |= mask & (case.segmentation[z - 1] != surface_segment.id)
        if z + 1 >= size_z:
            boundary |= mask
        else:
            boundary |= mask & (case.segmentation[z + 1] != surface_segment.id)
        coords_yx = np.argwhere(boundary)
        if not len(coords_yx):
            continue
        eligible_yx = [
            (int(y), int(x)) for y, x in coords_yx
            if _externally_exposed_body_voxel(case, (int(x), int(y), z))
        ]
        if not eligible_yx:
            continue
        coords_yx = np.asarray(eligible_yx, dtype=np.int64)
        selected_indices = np.unique(np.rint(np.linspace(
            0, len(coords_yx) - 1, min(quota, len(coords_yx))
        )).astype(int))
        for selected in selected_indices:
            y, x = coords_yx[int(selected)]
            sampled_xyz.append((int(x), int(y), int(z)))
            if len(sampled_xyz) >= max_entry_points:
                break
        if len(sampled_xyz) >= max_entry_points:
            break
    return [tuple(float(value) for value in case.geometry.index_to_physical(index))
            for index in sampled_xyz]


def suggest_trajectories(
    case: Case,
    target_lps: tuple[float, float, float],
    critical_labels=None,
    max_candidates: int = 5,
    max_entry_points: int = 32,
    search_radius_mm: float = 30.0,
) -> tuple[TrajectoryCandidate, ...]:
    """Return bounded deterministic trajectory suggestions for a target.

    Entry points are sampled from the existing body-contour/skin label. Plans
    sort by fewer marked critical-structure crossings, higher conservative
    clearance lower bound, shorter path, then LPS coordinates. This is a
    transparent geometric ordering for education, not a clinical optimum.
    Missing surface labels produce an empty tuple rather than invented entries.
    Suggestions require a marked Abscess mask and a target point inside it.
    """
    if not 1 <= int(max_candidates) <= 32:
        raise ValueError("max_candidates must be between 1 and 32")
    if not 1 <= int(max_entry_points) <= 128:
        raise ValueError("max_entry_points must be between 1 and 128")
    if not np.isfinite(search_radius_mm) or search_radius_mm <= 0:
        raise ValueError("search_radius_mm must be positive and finite")
    target = _point_lps(target_lps)
    abscess = next(
        (segment for segment in case.segments if segment.name.strip().casefold() == "abscess"),
        None,
    )
    if abscess is None or abscess.id not in case.present_segment_ids:
        raise ValueError("candidate generation requires a marked Abscess target")
    if _label_at_lps(case, target) != abscess.id:
        raise ValueError("candidate target must lie inside the marked Abscess region")
    entries = _sample_surface_entries(case, int(max_entry_points))
    measured: list[NeedlePlan] = []
    for entry in entries:
        if np.linalg.norm(np.asarray(entry) - target) <= 1e-8:
            continue
        plan = plan_trajectory(
            case, entry, tuple(target), critical_labels, search_radius_mm
        )
        if plan.entry_on_body_surface is True:
            measured.append(plan)

    def rank_key(plan: NeedlePlan):
        clearance = (plan.closest_clearance_mm
                     if plan.closest_clearance_mm is not None else 0.0)
        return (len(plan.collision_labels), -clearance, plan.depth_mm, plan.entry_lps)

    measured.sort(key=rank_key)
    summary = (
        "Sorted by fewer marked critical-structure crossings, higher conservative "
        "clearance lower bound, shorter depth, then LPS entry coordinates. "
        "Educational geometric ordering; not a clinical optimum."
    )
    return tuple(
        TrajectoryCandidate(
            rank=index + 1,
            plan=plan,
            clearance_lower_bound_mm=plan.closest_clearance_mm,
            ranking_summary=summary,
        )
        for index, plan in enumerate(measured[:int(max_candidates)])
    )


def evaluate_training(
    case: Case,
    plan: NeedlePlan,
    reference_entry_lps: tuple[float, float, float] | None = None,
    reference_target_lps: tuple[float, float, float] | None = None,
) -> dict[str, object]:
    """Return transparent exercise facts; no opaque aggregate score is invented."""
    result: dict[str, object] = {
        "target_hit": plan.target_hit,
        "depth_mm": plan.depth_mm,
        "angle_to_axial_plane_degrees": plan.angle_degrees,
        "collided": plan.collided,
        "collision_labels": list(plan.collision_labels),
        "clearances_mm": dict(plan.clearances_mm),
        "clearance_complete": plan.clearance_complete,
        "clearance_limit_mm": plan.clearance_limit_mm,
        "entry_on_body_surface": plan.entry_on_body_surface,
        "unmarked_critical_labels": list(plan.unmarked_critical_labels),
        "reference_errors_available": False,
    }
    if reference_entry_lps is not None and reference_target_lps is not None:
        reference_entry = _point_lps(reference_entry_lps)
        reference_target = _point_lps(reference_target_lps)
        ref_vector = reference_target - reference_entry
        ref_depth = float(np.linalg.norm(ref_vector))
        if ref_depth <= 1e-8:
            raise ValueError("reference entry and target must be distinct")
        ref_direction = ref_vector / ref_depth
        direction = np.asarray(plan.direction_lps)
        direction_cosine = float(np.einsum("i,i->", direction, ref_direction, optimize=False))
        if not math.isfinite(direction_cosine):
            raise ValueError("training angular comparison produced a non-finite value")
        angular_error = math.degrees(
            math.acos(float(np.clip(direction_cosine, -1.0, 1.0)))
        )
        result.update({
            "reference_errors_available": True,
            "entry_error_mm": float(np.linalg.norm(np.asarray(plan.entry_lps) - reference_entry)),
            "target_error_mm": float(np.linalg.norm(np.asarray(plan.target_lps) - reference_target)),
            "depth_error_mm": abs(plan.depth_mm - ref_depth),
            "angular_error_degrees": angular_error,
        })
    return result
