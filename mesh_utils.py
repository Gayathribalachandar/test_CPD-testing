import math

import numpy as np
from scipy.spatial import Delaunay, cKDTree
from shapely.geometry import Point
from shapely.ops import nearest_points


def _as_points_array(points):
    if points is None:
        return np.empty((0, 2), dtype=float)
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] < 2:
            return np.empty((0, 2), dtype=float)
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.empty((0, 2), dtype=float)
    return np.asarray(arr[:, :2], dtype=float)


def dedupe_min_distance(points, min_distance):
    pts = _as_points_array(points)
    if pts.size == 0:
        return pts
    try:
        min_distance = float(min_distance)
    except Exception:
        min_distance = 0.0
    if min_distance <= 1e-12 or len(pts) <= 1:
        return pts
    cell = max(min_distance, 1e-9)
    min_sq = float(min_distance) * float(min_distance)
    grid = {}
    kept = []

    def _grid_key(point):
        return (int(math.floor(point[0] / cell)), int(math.floor(point[1] / cell)))

    for point in pts:
        key = _grid_key(point)
        valid = True
        for gx in range(key[0] - 1, key[0] + 2):
            for gy in range(key[1] - 1, key[1] + 2):
                for other in grid.get((gx, gy), []):
                    dx = float(point[0]) - float(other[0])
                    dy = float(point[1]) - float(other[1])
                    if dx * dx + dy * dy < min_sq:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            continue
        grid.setdefault(key, []).append(point)
        kept.append(point)
    return _as_points_array(kept)


def snap_points_to_boundary(points, boundary, snap_distance, min_distance=None):
    pts = _as_points_array(points)
    if pts.size == 0 or boundary is None or getattr(boundary, "is_empty", True):
        return pts
    try:
        snap_distance = float(snap_distance)
    except Exception:
        snap_distance = 0.0
    if snap_distance <= 0.0:
        return pts
    snapped = np.array(pts, copy=True)
    min_distance = float(min_distance) if min_distance not in (None, "") else 0.0
    for idx, point in enumerate(snapped):
        probe = Point(float(point[0]), float(point[1]))
        try:
            if float(boundary.distance(probe)) > snap_distance:
                continue
            _, nearest = nearest_points(probe, boundary)
        except Exception:
            continue
        candidate = np.array([float(nearest.x), float(nearest.y)], dtype=float)
        if min_distance > 1e-12 and len(snapped) > 1:
            delta = snapped - candidate
            dist_sq = np.sum(delta * delta, axis=1)
            dist_sq[idx] = np.inf
            if np.any(dist_sq < (min_distance * min_distance)):
                continue
        snapped[idx] = candidate
    return snapped


def prune_isolated_points(points, radius, *, min_neighbors=1, search_radius_factor=2.2):
    pts = _as_points_array(points)
    if pts.size == 0 or len(pts) <= max(3, int(min_neighbors) + 1):
        return pts
    try:
        radius = float(radius)
    except Exception:
        radius = 0.0
    if radius <= 0.0:
        return pts
    query_radius = max(radius, radius * float(search_radius_factor))
    tree = cKDTree(pts)
    neighbors = tree.query_ball_point(pts, r=query_radius)
    keep_indices = [
        idx for idx, ids in enumerate(neighbors)
        if len(ids) - 1 >= int(min_neighbors)
    ]
    if len(keep_indices) < 3:
        return pts
    return np.asarray(pts[np.asarray(keep_indices, dtype=int)], dtype=float)


def stabilize_particle_cloud(points, radius, *, geometry=None, boundary=None):
    pts = _as_points_array(points)
    if pts.size == 0:
        return pts
    try:
        radius = float(radius)
    except Exception:
        radius = 0.0
    if radius <= 0.0:
        return pts
    pts = dedupe_min_distance(pts, radius * 0.98)
    if boundary is not None and not getattr(boundary, "is_empty", True):
        pts = snap_points_to_boundary(pts, boundary, radius * 0.5, min_distance=radius * 0.8)
        pts = dedupe_min_distance(pts, radius * 0.98)
    if geometry is not None and not getattr(geometry, "is_empty", True):
        mask = []
        for point in pts:
            probe = Point(float(point[0]), float(point[1]))
            try:
                mask.append(bool(geometry.covers(probe)))
            except Exception:
                mask.append(bool(geometry.contains(probe)))
        if mask:
            pts = pts[np.asarray(mask, dtype=bool)]
    pts = prune_isolated_points(pts, radius, min_neighbors=1, search_radius_factor=2.2)
    return _as_points_array(pts)


def triangle_quality_metrics(triangle_points):
    pts = _as_points_array(triangle_points)
    if len(pts) != 3:
        return {
            "area": 0.0,
            "min_angle_deg": 0.0,
            "aspect_ratio": float("inf"),
        }
    a = float(np.linalg.norm(pts[1] - pts[0]))
    b = float(np.linalg.norm(pts[2] - pts[1]))
    c = float(np.linalg.norm(pts[0] - pts[2]))
    lengths = [a, b, c]
    perimeter = a + b + c
    semiperimeter = perimeter * 0.5
    area_sq = max(
        semiperimeter
        * max(semiperimeter - a, 0.0)
        * max(semiperimeter - b, 0.0)
        * max(semiperimeter - c, 0.0),
        0.0,
    )
    area = math.sqrt(area_sq)
    min_angle = 0.0
    if area > 1e-14:
        for left, right, opposite in ((a, b, c), (b, c, a), (c, a, b)):
            denom = max(2.0 * left * right, 1e-14)
            cosine = (left * left + right * right - opposite * opposite) / denom
            cosine = max(-1.0, min(1.0, cosine))
            min_angle = min_angle or math.degrees(math.acos(cosine))
            min_angle = min(min_angle, math.degrees(math.acos(cosine)))
    shortest = max(min(lengths), 1e-14)
    longest = max(lengths)
    aspect_ratio = longest / shortest if shortest > 0.0 else float("inf")
    return {
        "area": float(area),
        "min_angle_deg": float(min_angle),
        "aspect_ratio": float(aspect_ratio),
    }


def triangle_quality_ok(triangle_points, *, min_angle_deg=15.0, max_aspect_ratio=5.0, min_area=0.0):
    metrics = triangle_quality_metrics(triangle_points)
    return (
        metrics["area"] >= float(min_area)
        and metrics["min_angle_deg"] >= float(min_angle_deg)
        and metrics["aspect_ratio"] <= float(max_aspect_ratio)
    )


POISSON_DEFAULT_SEED = 42


def poisson_sample(poly, r, k=30, seed=POISSON_DEFAULT_SEED):
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = poly.bounds
    cell = r / math.sqrt(2)
    grid = {}

    def grid_key(p):
        return (int(p[0] // cell), int(p[1] // cell))

    def valid(p):
        gx, gy = grid_key(p)
        for i in range(gx - 2, gx + 3):
            for j in range(gy - 2, gy + 3):
                if (i, j) in grid:
                    for q in grid[(i, j)]:
                        if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < r * r:
                            return False
        try:
            return poly.covers(Point(p))
        except Exception:
            return poly.contains(Point(p))

    pts = []
    active = []
    for _ in range(200):
        p = (rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if poly.contains(Point(p)):
            pts.append(p)
            active.append(p)
            grid.setdefault(grid_key(p), []).append(p)
            break
    if not active:
        return np.array([])
    while active:
        idx = int(rng.integers(len(active)))
        p = active[idx]
        found = False
        for _ in range(k):
            ang = rng.uniform(0, 2 * math.pi)
            rad = rng.uniform(r, 2 * r)
            q = (p[0] + rad * math.cos(ang), p[1] + rad * math.sin(ang))
            if minx <= q[0] <= maxx and miny <= q[1] <= maxy and valid(q):
                pts.append(q)
                active.append(q)
                grid.setdefault(grid_key(q), []).append(q)
                found = True
                break
        if not found:
            active.pop(idx)
    return np.array(pts)


def square_lattice_sample(poly, spacing, origin=None):
    if spacing <= 0:
        return np.array([])
    minx, miny, maxx, maxy = poly.bounds
    if origin is None:
        origin = (minx, miny)
    ox, oy = origin
    covers_fn = getattr(poly, "covers", None)
    touches_fn = getattr(poly, "touches", None)

    def _start(min_val, origin_val):
        return origin_val + math.ceil((min_val - origin_val) / spacing) * spacing

    start_x = _start(minx, ox)
    start_y = _start(miny, oy)
    xs = np.arange(start_x, maxx + spacing * 0.5, spacing)
    ys = np.arange(start_y, maxy + spacing * 0.5, spacing)
    pts = []
    for x in xs:
        for y in ys:
            p = Point(x, y)
            if covers_fn:
                if covers_fn(p):
                    pts.append((float(x), float(y)))
            elif poly.contains(p) or (touches_fn and touches_fn(p)):
                pts.append((float(x), float(y)))
    return np.array(pts)


def sample_ring(ring, dx):
    c = np.array(ring.coords)
    if len(c) < 2:
        return []
    try:
        dx = float(dx)
    except Exception:
        dx = 0.0
    if dx <= 1e-12:
        return []
    seg = np.diff(c, axis=0)
    L = np.sqrt((seg ** 2).sum(axis=1))
    s = np.insert(np.cumsum(L), 0, 0)
    total = s[-1]
    if total == 0:
        return []
    closed = False
    if len(c) > 2:
        try:
            closed = bool(np.linalg.norm(c[0] - c[-1]) <= 1e-9)
        except Exception:
            closed = False

    # Use an evenly distributed step based on the full curve length so we do not leave
    # a tiny remainder segment at the loop closure (or at the end of an open polyline).
    if closed:
        n_steps = max(3, int(round(float(total) / dx)))
        step = float(total) / float(n_steps)
        d_values = [step * i for i in range(n_steps)]
    else:
        n_steps = max(1, int(round(float(total) / dx)))
        step = float(total) / float(n_steps)
        d_values = [step * i for i in range(n_steps + 1)]
    pts = []
    for d in d_values:
        i = np.searchsorted(s, d) - 1
        i = max(0, min(i, len(L) - 1))
        if L[i] == 0:
            continue
        t = (d - s[i]) / L[i]
        pts.append(((1 - t) * c[i] + t * c[i + 1]).tolist())
    return pts


def generate_mesh(points, polygon):
    if len(points) < 3:
        return np.array([])
    tri = Delaunay(points)
    valid = []
    for s in tri.simplices:
        centroid = np.mean(points[s], axis=0)
        if polygon.contains(Point(centroid)):
            valid.append(s)
    return np.array(valid)


def part_tagged_triangles_to_element_map(
    triangle_part_ids,
    part_material_lookup=None,
):
    """Convert a per-triangle part_id array into the element_part_map list.

    Schema matches what sketch_view's worker emits: each entry is
    {'element_idx', 'part_id', 'material_id'}. material_id is filled from the
    optional part_material_lookup mapping (part_id -> material_id), else None.

    Returns a list of length len(triangle_part_ids).
    """
    out = []
    if triangle_part_ids is None:
        return out
    lookup = dict(part_material_lookup or {})
    for idx, pid in enumerate(triangle_part_ids):
        try:
            pid_int = int(pid)
        except Exception:
            pid_int = None
        out.append(
            {
                "element_idx": int(idx),
                "part_id": pid_int,
                "material_id": lookup.get(pid_int) if pid_int is not None else None,
            }
        )
    return out


def map_geometry_to_nodes(nodes, bc_list, tol):
    mapped_data = []
    if nodes is None or len(nodes) == 0:
        return mapped_data
    nodes = np.asarray(nodes)

    def _is_point(value):
        if isinstance(value, (tuple, list, np.ndarray)) and len(value) == 2:
            try:
                float(value[0])
                float(value[1])
                return True
            except (TypeError, ValueError):
                return False
        return False

    def _is_edge(value):
        return (
            isinstance(value, (tuple, list, np.ndarray))
            and len(value) == 2
            and _is_point(value[0])
            and _is_point(value[1])
        )

    for bc in bc_list:
        target = bc["coords"]
        if _is_point(target):
            dists = np.linalg.norm(nodes - np.array(target), axis=1)
            min_idx = np.argmin(dists)
            if dists[min_idx] < tol:
                mapped_data.append({"node_id": int(min_idx), "bc": bc})
        elif _is_edge(target):
            p1, p2 = np.array(target[0], dtype=float), np.array(target[1], dtype=float)
            edge_vec = p2 - p1
            len_sq = np.dot(edge_vec, edge_vec)
            if len_sq == 0:
                continue
            node_vecs = nodes - p1
            t = np.sum(node_vecs * edge_vec, axis=1) / len_sq
            on_segment = (t >= -1e-6) & (t <= 1 + 1e-6)
            projections = p1 + np.outer(t, edge_vec)
            dists = np.linalg.norm(nodes - projections, axis=1)
            indices = np.where(on_segment & (dists < tol))[0]
            for idx in indices:
                mapped_data.append(
                    {"node_id": int(idx), "bc": bc, "node_count": len(indices)}
                )
    return mapped_data
