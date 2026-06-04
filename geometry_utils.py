import math

from shapely.geometry import LineString, Polygon
from shapely.ops import split as _shapely_split


def dist(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def point_line_dist(pt, a, b):
    ax, ay = a
    bx, by = b
    px, py = pt
    dx, dy = bx - ax, by - ay
    if dx == dy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    nx, ny = ax + t * dx, ay + t * dy
    return math.hypot(px - nx, py - ny)


def split_polygon_by_line(polygon, line_start, line_end, extend=1e6):
    """Split a Shapely Polygon by a straight line into 2+ sub-polygons.

    The line is extended beyond `line_start` / `line_end` by `extend` so the
    user only has to click two points roughly indicating direction — the
    actual cut spans the full polygon. Returns a list of Polygon pieces.
    If the line doesn't actually cross the polygon, returns [polygon] (no
    change).

    The returned polygons preserve any interior rings (holes) of the input
    that fall entirely inside them. Interior rings that the cut line bisects
    are not currently supported (we drop them); the caller should warn.
    """
    if polygon is None or polygon.is_empty:
        return []
    sx, sy = float(line_start[0]), float(line_start[1])
    ex, ey = float(line_end[0]), float(line_end[1])
    dx, dy = ex - sx, ey - sy
    seg_len = math.hypot(dx, dy)
    if seg_len < 1e-12:
        return [polygon]
    # Extend the line in both directions so it fully crosses the polygon.
    ux, uy = dx / seg_len, dy / seg_len
    p0 = (sx - ux * extend, sy - uy * extend)
    p1 = (ex + ux * extend, ey + uy * extend)
    cutter = LineString([p0, p1])
    try:
        result = _shapely_split(polygon, cutter)
    except Exception:
        return [polygon]
    pieces = []
    for g in getattr(result, "geoms", [result]):
        if isinstance(g, Polygon) and not g.is_empty and g.area > 1e-12:
            pieces.append(g)
    if len(pieces) <= 1:
        return [polygon]
    return pieces


def get_solid_features(geom):
    if geom is None or geom.is_empty:
        return [], []
    geoms = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
    verts = set()
    edges = []
    for g in geoms:
        coords = list(g.exterior.coords)
        for i in range(len(coords) - 1):
            verts.add(coords[i])
            edges.append((coords[i], coords[i + 1]))
        for interior in g.interiors:
            coords = list(interior.coords)
            for i in range(len(coords) - 1):
                verts.add(coords[i])
                edges.append((coords[i], coords[i + 1]))
    return list(verts), edges
