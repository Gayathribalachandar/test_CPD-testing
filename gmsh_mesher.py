import math

import numpy as np


class GmshError(RuntimeError):
    pass


def verify_gmsh_available():
    try:
        import gmsh as _gmsh  # noqa: F401
        return True
    except Exception:
        return False


def _ensure_list(dimtags):
    if dimtags is None:
        return []
    return dimtags if isinstance(dimtags, list) else list(dimtags)


def _hex_to_tets(hex_nodes):
    n0, n1, n2, n3, n4, n5, n6, n7 = hex_nodes
    return [
        (n0, n1, n3, n4),
        (n1, n2, n3, n6),
        (n1, n3, n4, n6),
        (n1, n4, n5, n6),
        (n3, n4, n6, n7),
    ]


def _apply_rotation(gmsh, dimtags, center, rotation):
    rx, ry, rz = rotation
    cx, cy, cz = center
    if abs(rx) > 1e-9:
        gmsh.model.occ.rotate(dimtags, cx, cy, cz, 1, 0, 0, math.radians(rx))
    if abs(ry) > 1e-9:
        gmsh.model.occ.rotate(dimtags, cx, cy, cz, 0, 1, 0, math.radians(ry))
    if abs(rz) > 1e-9:
        gmsh.model.occ.rotate(dimtags, cx, cy, cz, 0, 0, 1, math.radians(rz))


def _build_primitive(gmsh, prim):
    ptype = prim.get("type")
    params = prim.get("params", {})
    transform = prim.get("transform", {})
    cx = float(transform.get("tx", 0.0))
    cy = float(transform.get("ty", 0.0))
    cz = float(transform.get("tz", 0.0))
    rotation = (
        float(transform.get("rx", 0.0)),
        float(transform.get("ry", 0.0)),
        float(transform.get("rz", 0.0)),
    )

    tag = None
    if ptype == "box":
        w = float(params.get("width", 1.0))
        d = float(params.get("depth", 1.0))
        h = float(params.get("height", 1.0))
        tag = gmsh.model.occ.addBox(cx - w * 0.5, cy - d * 0.5, cz - h * 0.5, w, d, h)
    elif ptype == "cylinder":
        r = float(params.get("radius", 1.0))
        h = float(params.get("height", 1.0))
        tag = gmsh.model.occ.addCylinder(cx, cy, cz - h * 0.5, 0, 0, h, r)
    elif ptype == "sphere":
        r = float(params.get("radius", 1.0))
        tag = gmsh.model.occ.addSphere(cx, cy, cz, r)
    elif ptype == "cone":
        r1 = float(params.get("radius_base", 1.0))
        r2 = float(params.get("radius_top", 0.0))
        h = float(params.get("height", 1.0))
        tag = gmsh.model.occ.addCone(cx, cy, cz - h * 0.5, 0, 0, h, r1, r2)
    elif ptype == "ring":
        r1 = float(params.get("major_radius", 1.0))
        r2 = float(params.get("tube_radius", 0.2))
        tag = gmsh.model.occ.addTorus(cx, cy, cz, r1, r2)
    elif ptype == "extrude":
        w = float(params.get("profile_width", 1.0))
        d = float(params.get("profile_depth", 1.0))
        h = float(params.get("height", 1.0))
        tag = gmsh.model.occ.addBox(cx - w * 0.5, cy - d * 0.5, cz - h * 0.5, w, d, h)
    elif ptype == "revolve":
        r = float(params.get("profile_radius", 1.0))
        h = float(params.get("height", 1.0))
        tag = gmsh.model.occ.addCylinder(cx, cy, cz - h * 0.5, 0, 0, h, r)

    if tag is None:
        return []
    dimtags = [(3, tag)]
    if any(abs(v) > 1e-9 for v in rotation):
        _apply_rotation(gmsh, dimtags, (cx, cy, cz), rotation)
    return dimtags


def _build_occ_model(gmsh, model3d):
    shape_map = {}
    for prim in model3d.get("primitives", []):
        dimtags = _build_primitive(gmsh, prim)
        if dimtags:
            shape_map[int(prim.get("id"))] = dimtags

    for op in model3d.get("operations", []):
        a = shape_map.get(int(op.get("a", -1)))
        b = shape_map.get(int(op.get("b", -1)))
        if not a or not b:
            continue
        op_type = str(op.get("op", "")).lower()
        if op_type in ("union", "add", "fuse"):
            out, _ = gmsh.model.occ.fuse(a, b, removeObject=False, removeTool=False)
        elif op_type in ("cut", "subtract", "difference"):
            out, _ = gmsh.model.occ.cut(a, b, removeObject=False, removeTool=False)
        elif op_type in ("intersect", "common"):
            out, _ = gmsh.model.occ.intersect(a, b, removeObject=False, removeTool=False)
        else:
            continue
        out = _ensure_list(out)
        if out:
            shape_map[int(op.get("id"))] = out

    if model3d.get("operations"):
        last_id = int(model3d["operations"][-1].get("id"))
        return shape_map.get(last_id, [])

    all_vols = []
    for dimtags in shape_map.values():
        all_vols.extend(dimtags)
    return all_vols


def _mesh_elements(gmsh, allow_hex):
    elem_types, _, elem_nodes = gmsh.model.mesh.getElements(3)
    tets = []
    hexes = []
    unsupported = False
    for etype, conn in zip(elem_types, elem_nodes):
        if etype == 4:  # 4-node tetra
            conn = np.asarray(conn, dtype=int).reshape(-1, 4)
            tets.extend(conn.tolist())
        elif etype == 5 and allow_hex:  # 8-node hex
            conn = np.asarray(conn, dtype=int).reshape(-1, 8)
            hexes.extend(conn.tolist())
        else:
            unsupported = True
    return tets, hexes, unsupported


def _generate_mesh(gmsh, mesh_size, mesh_type, fallback_tetra):
    mesh_type = str(mesh_type).lower()
    if mesh_type not in ("tetra", "hex-dominant"):
        mesh_type = "tetra"

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", float(mesh_size))
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", float(mesh_size))
    gmsh.option.setNumber("Mesh.ElementOrder", 1)

    def _generate_tetra():
        gmsh.option.setNumber("Mesh.RecombineAll", 0)
        gmsh.model.mesh.generate(3)

    def _generate_hex_dominant():
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
        gmsh.model.mesh.generate(3)

    used_type = "tetra"
    if mesh_type == "hex-dominant":
        try:
            _generate_hex_dominant()
            used_type = "hex-dominant"
        except Exception:
            if not fallback_tetra:
                raise
            gmsh.model.mesh.clear()
            _generate_tetra()
            used_type = "tetra (fallback)"
    else:
        _generate_tetra()
        used_type = "tetra"

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    if len(node_tags) == 0:
        raise GmshError("No particles generated.")
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    tag_to_index = {int(tag): idx for idx, tag in enumerate(node_tags)}

    allow_hex = mesh_type == "hex-dominant"
    tets, hexes, unsupported = _mesh_elements(gmsh, allow_hex=allow_hex)
    if unsupported and mesh_type == "hex-dominant" and fallback_tetra:
        gmsh.model.mesh.clear()
        _generate_tetra()
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(coords, dtype=float).reshape(-1, 3)
        tag_to_index = {int(tag): idx for idx, tag in enumerate(node_tags)}
        tets, hexes, _ = _mesh_elements(gmsh, allow_hex=False)
        used_type = "tetra (fallback)"

    tet_elements = []
    for tet in tets:
        tet_elements.append([tag_to_index[int(n)] for n in tet])

    if hexes:
        for hex_elem in hexes:
            mapped = [tag_to_index[int(n)] for n in hex_elem]
            for tet in _hex_to_tets(mapped):
                tet_elements.append(list(tet))
        if used_type == "hex-dominant":
            used_type = "hex-dominant (displayed as tetra)"

    if not tet_elements:
        raise GmshError("No tetrahedral elements generated.")

    return coords, np.asarray(tet_elements, dtype=int), used_type


def generate_volume_mesh(model3d, mesh_size, mesh_type="tetra", fallback_tetra=True):
    if not verify_gmsh_available():
        raise GmshError(
            "Gmsh Python API not available. Open Dependency Check and install 'gmsh'."
        )
    try:
        import gmsh
    except Exception as exc:
        raise GmshError(
            "Gmsh Python API not available. Install with 'sudo apt install python3-gmsh' "
            "or 'pip install gmsh'."
        ) from exc

    mesh_type = str(mesh_type).lower()
    if mesh_type not in ("tetra", "hex-dominant"):
        mesh_type = "tetra"

    try:
        if hasattr(gmsh, "isInitialized") and gmsh.isInitialized():
            try:
                gmsh.finalize()
            except Exception:
                pass
    except Exception:
        pass
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cpd_simstudio")
    try:
        volumes = _build_occ_model(gmsh, model3d)
        if not volumes:
            raise GmshError("No 3D volumes to generate connections.")
        all_vols = gmsh.model.occ.getEntities(3)
        if all_vols:
            keep = set(tuple(v) for v in volumes)
            remove = [v for v in all_vols if tuple(v) not in keep]
            if remove:
                gmsh.model.occ.remove(remove, recursive=True)
        gmsh.model.occ.synchronize()
        return _generate_mesh(gmsh, mesh_size, mesh_type, fallback_tetra)
    finally:
        gmsh.finalize()


# ---------------------------------------------------------------------------
# 2D heterogeneous multi-part surface meshing
# ---------------------------------------------------------------------------

import signal as _signal_mod
import threading as _threading_mod
from dataclasses import dataclass, field as _dc_field
from typing import Any


class _SuppressSignalInWorker:
    """Skip gmsh's SIGINT-handler install when initializing from a Qt worker.

    gmsh.initialize() calls signal.signal(SIGINT, ...) which raises
    "signal only works in main thread of the main interpreter" if invoked
    from a non-main Python thread. We monkey-patch signal.signal to a no-op
    for the duration of the gmsh init/finalize calls when we're off-thread.
    """

    def __enter__(self):
        self._patched = _threading_mod.current_thread() is not _threading_mod.main_thread()
        if self._patched:
            self._orig = _signal_mod.signal
            _signal_mod.signal = lambda *a, **kw: None
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._patched:
            _signal_mod.signal = self._orig
        return False


@dataclass
class GmshPartSpec:
    """Mesh-build description of a single part for the 2D gmsh pipeline.

    Loops are lists of (x, y) tuples. The first and last points must NOT
    repeat; the loop is closed implicitly. The mesher rounds endpoints and
    deduplicates shared curves so neighbouring parts share node ids.
    """

    id: int
    name: str
    material_id: Any
    outer_loop: list
    inner_loops: list = _dc_field(default_factory=list)


def _round_key(pt, tol):
    return (round(float(pt[0]) / tol) * tol, round(float(pt[1]) / tol) * tol)


def _build_gmsh_2d_geometry(gmsh, parts, tol):
    """OCC + fragment build. Handles overlap, corner-only contact and clean shared edges.

    Each part is built as an OpenCASCADE plane surface, then `occ.fragment`
    splits overlaps and welds shared edges. For overlap regions (a fragment
    sub-surface that came from multiple input parts), the *smaller* original
    part wins — matching the matrix/inclusion convention.

    Returns:
        surfaces: dict[part_id -> list[surface_tag]]
        curve_to_surfaces: dict[curve_tag -> set(part_id)]   (post-fragment topology)
        hole_curves: set[curve_tag]                          (inner-loop boundaries)
    """
    point_cache = {}
    line_cache = {}

    def _get_point(pt):
        key = _round_key(pt, tol)
        tag = point_cache.get(key)
        if tag is None:
            tag = gmsh.model.occ.addPoint(float(key[0]), float(key[1]), 0.0)
            point_cache[key] = tag
        return tag

    def _get_line(a_pt, b_pt):
        a_key = _round_key(a_pt, tol)
        b_key = _round_key(b_pt, tol)
        if a_key == b_key:
            return None
        edge = tuple(sorted((a_key, b_key)))
        cached = line_cache.get(edge)
        if cached is not None:
            return cached
        a_tag = _get_point(a_pt)
        b_tag = _get_point(b_pt)
        tag = gmsh.model.occ.addLine(a_tag, b_tag)
        line_cache[edge] = tag
        return tag

    def _make_loop(coords):
        if len(coords) < 3:
            raise GmshError("Loop needs at least 3 distinct points.")
        if (
            len(coords) >= 2
            and abs(float(coords[0][0]) - float(coords[-1][0])) < tol * 0.5
            and abs(float(coords[0][1]) - float(coords[-1][1])) < tol * 0.5
        ):
            coords = list(coords)[:-1]
        line_tags = []
        n = len(coords)
        for i in range(n):
            tag = _get_line(coords[i], coords[(i + 1) % n])
            if tag is not None:
                line_tags.append(tag)
        if len(line_tags) < 3:
            raise GmshError("Loop has too few non-degenerate edges.")
        return gmsh.model.occ.addCurveLoop(line_tags), line_tags

    # Build one OCC surface per part (and one per polygon piece for MultiPolygon).
    # part_surfaces_in is a flat list aligned with the indices we hand to fragment.
    part_surfaces_in = []  # [(part_id, name, area, surface_tag, hole_loop_line_tags)]
    for part in parts:
        outer_loop_tag, _outer_lines = _make_loop(part.outer_loop)
        inner_loop_tags = []
        hole_lines_for_part = []
        for inner in (part.inner_loops or []):
            try:
                in_tag, in_lines = _make_loop(inner)
            except GmshError:
                continue
            inner_loop_tags.append(in_tag)
            hole_lines_for_part.extend(in_lines)
        surf = gmsh.model.occ.addPlaneSurface([outer_loop_tag, *inner_loop_tags])
        # Cheap polygon area for overlap-tiebreaking. Negative = clockwise; we just want |area|.
        area = 0.0
        coords = list(part.outer_loop)
        n = len(coords)
        for i in range(n):
            x0, y0 = coords[i][0], coords[i][1]
            x1, y1 = coords[(i + 1) % n][0], coords[(i + 1) % n][1]
            area += float(x0) * float(y1) - float(x1) * float(y0)
        area = abs(area) * 0.5
        part_surfaces_in.append(
            (int(part.id), str(part.name or f"part_{part.id}"), area, surf, set(hole_lines_for_part))
        )

    if not part_surfaces_in:
        return {}, {}, set()

    # Run fragment to split overlaps and weld shared edges. fragment requires
    # at least one tool entity, so the single-part case bypasses it.
    if len(part_surfaces_in) == 1:
        gmsh.model.occ.synchronize()
        pid, _name, _area, surf, hole_lines = part_surfaces_in[0]
        # Build curve_to_surfaces from the single surface's boundary curves.
        bnd = gmsh.model.getBoundary([(2, surf)], oriented=False, recursive=False)
        curve_to_surfaces = {}
        for dim, ctag in bnd:
            if dim == 1:
                curve_to_surfaces.setdefault(int(ctag), set()).add(pid)
        return {pid: [surf]}, curve_to_surfaces, set(hole_lines)

    object_dimtags = [(2, part_surfaces_in[0][3])]
    tool_dimtags = [(2, ps[3]) for ps in part_surfaces_in[1:]]
    _out_dimtags, out_map = gmsh.model.occ.fragment(object_dimtags, tool_dimtags)
    gmsh.model.occ.synchronize()

    # out_map[i] = list of new (dim, tag) entities derived from input i.
    # Apply smaller-area-wins for overlap: if multiple inputs claim the same
    # fragment sub-surface, the one with the smallest original area takes it.
    fragment_owners = {}  # new_tag -> list of (area, part_idx_in_inputs)
    for input_idx, derived in enumerate(out_map):
        _pid, _name, area, _surf, _hole = part_surfaces_in[input_idx]
        for dim, tag in derived:
            if dim != 2:
                continue
            fragment_owners.setdefault(int(tag), []).append((area, input_idx))

    new_surface_to_part_idx = {}
    for new_tag, claims in fragment_owners.items():
        # Smallest-area input wins. Stable secondary key = input_idx (later wins on tie).
        claims.sort(key=lambda c: (c[0], -c[1]))
        new_surface_to_part_idx[new_tag] = claims[0][1]

    surfaces = {}
    for new_tag, input_idx in new_surface_to_part_idx.items():
        pid = part_surfaces_in[input_idx][0]
        surfaces.setdefault(pid, []).append(new_tag)

    # Detect interface curves from post-fragment boundary topology.
    # An interface curve is one that bounds 2+ surfaces with different part_ids.
    curve_to_surfaces = {}
    for new_tag, input_idx in new_surface_to_part_idx.items():
        pid = part_surfaces_in[input_idx][0]
        bnd = gmsh.model.getBoundary([(2, new_tag)], oriented=False, recursive=False)
        for dim, ctag in bnd:
            if dim == 1:
                curve_to_surfaces.setdefault(int(ctag), set()).add(pid)

    # Hole curves: inner-loop lines whose tags survived fragment unchanged.
    # Best-effort — fragment can rewrite them, in which case the hole won't
    # be tagged here, but the interface logic above still catches their
    # post-fragment counterparts when relevant.
    hole_curves = set()
    for _pid, _name, _area, _surf, hole_lines in part_surfaces_in:
        for c in hole_lines:
            if int(c) in curve_to_surfaces:
                hole_curves.add(int(c))

    return surfaces, curve_to_surfaces, hole_curves


def _classify_feature_curves(curve_to_surfaces, hole_curves):
    """Split curves into interface curves (shared by ≥2 surfaces), hole
    curves (inner-loop boundaries), and outer boundary curves (touching
    exactly one surface and not a hole). All three are candidates for
    Distance-field refinement.
    """
    interface_curves = sorted(
        c for c, surf_set in curve_to_surfaces.items() if len(surf_set) >= 2
    )
    holes = sorted(int(c) for c in hole_curves)
    hole_set = set(holes)
    boundary_curves = sorted(
        int(c)
        for c, surf_set in curve_to_surfaces.items()
        if len(surf_set) == 1 and int(c) not in hole_set
    )
    return interface_curves, holes, boundary_curves


def _polygon_convex_hull_ccw(points):
    """Return the convex hull of (x, y) points in CCW order.

    Andrew's monotone chain algorithm — handles degenerate inputs by returning
    a list with fewer than 3 points.
    """
    pts = sorted({(float(x), float(y)) for x, y in points})
    if len(pts) < 3:
        return pts

    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _polygon_exterior_distance_expression(hull_ccw):
    """gmsh MathEval expression approximating distance from (x,y) to the
    polygon's exterior: 0 inside, positive outside, smoothly increasing.

    Uses the normalized signed-distance form (a*x + b*y + c) per edge, then
    takes max(0, -min(signed_distances)). For convex polygons this is the
    exact perpendicular distance to the nearest violating edge, which equals
    the distance to the polygon boundary in the half-space the point lies in.
    """
    if len(hull_ccw) < 3:
        return "0"
    import math as _math
    sd_terms = []
    n = len(hull_ccw)
    for i in range(n):
        x1, y1 = hull_ccw[i]
        x2, y2 = hull_ccw[(i + 1) % n]
        a_raw = -(y2 - y1)
        b_raw = (x2 - x1)
        edge_len = _math.hypot(a_raw, b_raw)
        if edge_len < 1e-12:
            continue
        a = a_raw / edge_len
        b = b_raw / edge_len
        c = -(a_raw * x1 + b_raw * y1) / edge_len
        sd_terms.append(f"(({a:.10g})*x+({b:.10g})*y+({c:.10g}))")
    if not sd_terms:
        return "0"
    min_expr = sd_terms[0]
    for term in sd_terms[1:]:
        min_expr = f"min({min_expr},{term})"
    return f"max(0,(-({min_expr})))"


def _polygon_inside_expression(hull_ccw, transition_scale=1.0):
    """Build a gmsh MathEval expression that approximates 1 inside the CCW
    convex polygon and 0 outside. AND of half-plane tests, with each test
    encoded as a sigmoid because gmsh's MathEval parser does not support
    comparison operators (>=, <=).

    For a CCW convex polygon, the signed distance to edge (p1 -> p2) is
        d = ( -(y2-y1)*(x-x1) + (x2-x1)*(y-y1) ) / edge_length
    Positive d ⇒ inside the half-plane; negative ⇒ outside.

    Sigmoid step: 1/(1+exp(-k*d)) where k = 1/transition_scale. A point at
    distance `transition_scale` outside the edge is mapped to sigmoid ≈ 0.27;
    well-inside points → 1; well-outside points → 0.
    """
    if len(hull_ccw) < 3:
        return ""
    import math as _math
    k = 1.0 / max(float(transition_scale), 1e-9)
    terms = []
    n = len(hull_ccw)
    for i in range(n):
        x1, y1 = hull_ccw[i]
        x2, y2 = hull_ccw[(i + 1) % n]
        # Normalized half-plane equation so a*x + b*y + c is signed distance.
        a_raw = -(y2 - y1)
        b_raw = (x2 - x1)
        edge_len = _math.hypot(a_raw, b_raw)
        if edge_len < 1e-12:
            continue
        a = a_raw / edge_len
        b = b_raw / edge_len
        c = (a_raw * x1 + b_raw * y1) / edge_len * -1.0
        # Note: a*x + b*y + c = signed distance to the edge (positive inside).
        terms.append(
            f"(1/(1+exp(-({k:.10g})*(({a:.10g})*x+({b:.10g})*y+({c:.10g})))))"
        )
    if not terms:
        return ""
    return "*".join(terms)


def _curve_endpoint_map(gmsh):
    """Cache: gmsh curve tag -> (pt1_tag, (x1, y1), pt2_tag, (x2, y2)).

    Used by edge-seed resolution to match user-picked endpoints (which were
    recorded in part coordinates at edge-pick time) to live gmsh curve tags
    (which are re-issued every meshing run). Coordinate proximity is the
    only stable identifier we have across meshes.
    """
    out = {}
    for dim, ctag in gmsh.model.getEntities(dim=1):
        bnd = gmsh.model.getBoundary([(1, ctag)], oriented=False)
        pt_tags = [int(t) for d, t in bnd if d == 0]
        if len(pt_tags) != 2:
            continue
        coords = []
        for p in pt_tags:
            try:
                v = gmsh.model.getValue(0, p, [])
            except Exception:
                v = (0.0, 0.0, 0.0)
            coords.append((float(v[0]), float(v[1])))
        out[int(ctag)] = (pt_tags[0], coords[0], pt_tags[1], coords[1])
    return out


def _resolve_seed(seed, curve_map, tol):
    """Return list of (curve_tag, fine_endpoint_tag, edge_length) for each
    edge_ref in `seed` that matches a curve. The fine endpoint honors
    seed.flip_bias: by default the user-recorded `start` is the fine end.

    When seed.propagate_to_neighbors is True, additionally append every curve
    that shares an endpoint with any directly-matched curve. Propagated curves
    don't have a meaningful "fine end" — single-bias semantics are best-effort.
    """
    import math as _math
    matched = []
    matched_curve_tags = set()
    matched_point_tags = set()  # endpoints of directly-matched curves
    for ref in getattr(seed, "edge_refs", []) or []:
        start = ref.get("start")
        end = ref.get("end")
        if start is None or end is None:
            continue
        for ctag, (p1_tag, p1_xy, p2_tag, p2_xy) in curve_map.items():
            d11 = _math.hypot(p1_xy[0] - start[0], p1_xy[1] - start[1])
            d22 = _math.hypot(p2_xy[0] - end[0], p2_xy[1] - end[1])
            d12 = _math.hypot(p1_xy[0] - end[0], p1_xy[1] - end[1])
            d21 = _math.hypot(p2_xy[0] - start[0], p2_xy[1] - start[1])
            forward = d11 < tol and d22 < tol
            reverse = d12 < tol and d21 < tol
            if not (forward or reverse):
                continue
            edge_len = _math.hypot(p2_xy[0] - p1_xy[0], p2_xy[1] - p1_xy[1])
            if edge_len < 1e-12:
                continue
            # Fine end: ref.start by default, ref.end when flipped.
            # If matched in reverse orientation, the gmsh "start" point is p2.
            start_pt_tag = p1_tag if forward else p2_tag
            end_pt_tag = p2_tag if forward else p1_tag
            fine_pt_tag = end_pt_tag if seed.flip_bias else start_pt_tag
            matched.append((int(ctag), int(fine_pt_tag), float(edge_len)))
            matched_curve_tags.add(int(ctag))
            matched_point_tags.add(int(p1_tag))
            matched_point_tags.add(int(p2_tag))
            break  # first matching curve per ref

    # Propagation: include every other curve that shares an endpoint with any
    # directly-matched curve. The propagated curves inherit the seed's size
    # field but don't have a unique "fine end" — we use their own p1 as a
    # placeholder for single-bias (best-effort; the natural curve orientation
    # decides which end becomes fine).
    if getattr(seed, "propagate_to_neighbors", False) and matched_curve_tags:
        for ctag, (p1_tag, p1_xy, p2_tag, p2_xy) in curve_map.items():
            if int(ctag) in matched_curve_tags:
                continue
            if int(p1_tag) in matched_point_tags or int(p2_tag) in matched_point_tags:
                edge_len = _math.hypot(p2_xy[0] - p1_xy[0], p2_xy[1] - p1_xy[1])
                if edge_len < 1e-12:
                    continue
                matched.append((int(ctag), int(p1_tag), float(edge_len)))
                matched_curve_tags.add(int(ctag))
    return matched


def _resolve_vertex_seed(seed, curve_map, tol):
    """Map a VertexSeed's world coordinate to the closest gmsh point tag.

    Returns the matched point tag or None if no point lies within `tol`.
    """
    import math as _math
    seen = set()
    best_tag = None
    best_d = float("inf")
    tx, ty = float(seed.point[0]), float(seed.point[1])
    for ctag, (p1_tag, p1_xy, p2_tag, p2_xy) in curve_map.items():
        for pt_tag, pt_xy in ((p1_tag, p1_xy), (p2_tag, p2_xy)):
            if pt_tag in seen:
                continue
            seen.add(pt_tag)
            d = _math.hypot(pt_xy[0] - tx, pt_xy[1] - ty)
            if d < best_d:
                best_d = d
                best_tag = pt_tag
    if best_tag is not None and best_d <= tol:
        return int(best_tag)
    return None


def _install_size_field(
    gmsh,
    sizing,
    interface_curves,
    hole_curves,
    custom_zones=None,
    boundary_curves=None,
    edge_seeds=None,
    part_mesh_overrides=None,
    surfaces=None,
    vertex_seeds=None,
    boundary_layer_seeds=None,
):
    """Install combined Distance+MathEval size fields. Skip empty curve lists.

    custom_zones: optional iterable of CustomMeshZone-like objects.
    boundary_curves: optional outer boundary curves (touching one surface) —
        treated identically to interfaces so the mesh gradient-refines
        toward the model's outer edge.
    edge_seeds: optional iterable of EdgeSeed-like objects (Abaqus-style
        local edge seeds).
    part_mesh_overrides: optional {part_id: {"h_bulk": float}} mapping. Each
        override produces a gmsh Constant field scoped to that part's surface
        tags. Combined with all other fields via Min, the override wins
        whenever it is finer than the rest — the common per-part refinement
        case. To coarsen a single part relative to the global bulk, increase
        the global size and use per-part overrides to refine.
    surfaces: dict[part_id -> list[surface_tag]] from _build_gmsh_2d_geometry.
        Required to scope per-part overrides; ignored if part_mesh_overrides
        is empty.
    """
    if sizing is None:
        return None

    # Disable competing size sources; otherwise the size field is silently overridden.
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    threshold_ids = []

    def _add_distance_threshold(curve_list):
        if not curve_list:
            return  # Gmsh errors on empty CurvesList — skip silently.
        dist_id = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist_id, "CurvesList", [int(c) for c in curve_list])
        # Higher Sampling resolves long boundary curves more accurately so the
        # size field doesn't undersample and produce visual artifacts.
        gmsh.model.mesh.field.setNumber(dist_id, "Sampling", 200)

        # MathEval gives a true asymptotic exponential approach from h_feature
        # to h_bulk — no clamping, no perceptible "edge" where refinement ends.
        # Formula:  size(d) = h_feature + (h_bulk - h_feature) * (1 - exp(-d / scale))
        #   d = 0           -> h_feature                (full refinement at boundary)
        #   d = scale       -> h_feature + 0.63*(h_bulk-h_feature)
        #   d = 3*scale     -> h_feature + 0.95*(h_bulk-h_feature) (essentially h_bulk)
        # scale = transition_width / 3 calibrates so that at d=transition_width
        # the size has reached ~95% of h_bulk, matching the user's intent for
        # "the gradient zone" without a hard cutoff.
        h_feat = float(sizing.h_feature)
        h_bulk = float(sizing.h_bulk)
        delta = h_bulk - h_feat
        scale = max(float(sizing.transition_width) / 3.0, 1e-9)
        math_id = gmsh.model.mesh.field.add("MathEval")
        expr = f"{h_feat} + ({delta})*(1.0 - exp(-F{dist_id}/{scale}))"
        gmsh.model.mesh.field.setString(math_id, "F", expr)
        threshold_ids.append(math_id)

    # Bulk + gradient refinement.
    # ----------------------------------------------------------------------
    # Two paths:
    #
    # (a) Per-part Restrict-scoped refinement — used when we have `surfaces`
    #     and the per-part architecture is feasible. Each part gets its own
    #     bulk size (override or global) AND its own gradient (h_feature →
    #     part_h_bulk), Restrict'd to the part's surfaces. This is the only
    #     way to make per-part h_bulk COARSER than the global value: the
    #     global asymptote is replaced by the per-part asymptote inside the
    #     part, so no global field "leaks" a finer value into the override
    #     region. Interface curves (shared between parts) get refined twice
    #     — once per neighboring part — and Min picks whichever h_feature is
    #     smaller, so the shared edge stays correctly fine.
    #
    # (b) Global refinement — fallback for the rare case where surfaces is
    #     empty. Lumps interface, hole, and outer boundary curves into one
    #     Distance + MathEval that grows from h_feature to global h_bulk.
    #
    # Path (a) was added to fix the "override only honored when finer than
    # global" limitation.
    use_per_part = bool(surfaces)
    if use_per_part:
        # Map each refinable boundary curve to the part_id(s) it bounds.
        # Outer-boundary curves (touching only one part AND not a hole) are
        # deliberately excluded here so the outermost edges of the geometry
        # default to BULK-sized mesh. Holes and multi-part interfaces are
        # still refined to h_feature. Users who want explicit refinement on
        # an outer edge can use Edge Seeds (per-edge, opt-in).
        all_curves = set()
        for cl in (interface_curves or [], hole_curves or []):
            for c in cl:
                all_curves.add(int(c))
        surf_to_part = {}
        for pid, slist in surfaces.items():
            for s in slist:
                surf_to_part[int(s)] = int(pid)
        curve_to_parts = {}
        for c in all_curves:
            try:
                up_surfs, _ = gmsh.model.getAdjacencies(1, int(c))
            except Exception:
                up_surfs = []
            for s in up_surfs:
                pid = surf_to_part.get(int(s))
                if pid is not None:
                    curve_to_parts.setdefault(int(c), set()).add(int(pid))

        global_h_feat = float(sizing.h_feature)
        scale = max(float(sizing.transition_width) / 3.0, 1e-9)
        for pid, surf_tags in surfaces.items():
            if not surf_tags:
                continue
            override = (part_mesh_overrides or {}).get(int(pid)) or {}
            try:
                h_part_bulk = float(override.get("h_bulk") or sizing.h_bulk)
            except Exception:
                h_part_bulk = float(sizing.h_bulk)
            try:
                h_part_feat = float(override.get("h_feature") or global_h_feat)
            except Exception:
                h_part_feat = global_h_feat
            if h_part_feat <= 0 or h_part_feat > h_part_bulk:
                h_part_feat = min(global_h_feat, h_part_bulk)

            part_curves = [c for c, parts in curve_to_parts.items() if int(pid) in parts]
            this_part_fields = []

            # Bulk Constant (always present for a part)
            const_id = gmsh.model.mesh.field.add("Constant")
            gmsh.model.mesh.field.setNumber(const_id, "VIn", h_part_bulk)
            # VOut high so Restrict outside this part doesn't contribute via Min.
            gmsh.model.mesh.field.setNumber(const_id, "VOut", float(sizing.h_bulk) * 1e6)
            gmsh.model.mesh.field.setNumbers(const_id, "SurfacesList", [int(s) for s in surf_tags])
            this_part_fields.append(const_id)

            # Gradient if the part has refined boundaries
            if part_curves and h_part_feat < h_part_bulk:
                dist_id = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(dist_id, "CurvesList", part_curves)
                gmsh.model.mesh.field.setNumber(dist_id, "Sampling", 200)
                math_id = gmsh.model.mesh.field.add("MathEval")
                delta = h_part_bulk - h_part_feat
                expr = f"{h_part_feat:.10g} + ({delta:.10g})*(1.0 - exp(-F{dist_id}/{scale:.10g}))"
                gmsh.model.mesh.field.setString(math_id, "F", expr)
                this_part_fields.append(math_id)

            # Combine bulk + gradient into one per-part field, then Restrict.
            if len(this_part_fields) == 1:
                inner = this_part_fields[0]
            else:
                inner = gmsh.model.mesh.field.add("Min")
                gmsh.model.mesh.field.setNumbers(inner, "FieldsList", this_part_fields)
            restrict_id = gmsh.model.mesh.field.add("Restrict")
            gmsh.model.mesh.field.setNumber(restrict_id, "InField", inner)
            gmsh.model.mesh.field.setNumbers(restrict_id, "SurfacesList", [int(s) for s in surf_tags])
            try:
                gmsh.model.mesh.field.setNumber(restrict_id, "IncludeBoundary", 1)
            except Exception:
                pass
            threshold_ids.append(restrict_id)
    else:
        _add_distance_threshold(interface_curves)
        _add_distance_threshold(hole_curves)
        _add_distance_threshold(boundary_curves or [])

    # Zone-based local refinement: each user-drawn polygon contributes a
    # MathEval size field whose value is zone_size INSIDE the polygon and
    # h_bulk OUTSIDE. Membership is computed from the polygon's convex hull
    # via an AND of half-plane tests (one per hull edge). This honors the
    # actual polygon shape rather than its bbox, so refinement does not bleed
    # into rectangular margins. For non-convex polygons the convex hull is a
    # conservative superset — a small over-refinement near concavities, but
    # never the box-shaped bleed seen with axis-aligned Box fields.
    if custom_zones:
        import math as _math
        h_bulk_v = float(sizing.h_bulk)
        for zone in custom_zones:
            try:
                pts = list(getattr(zone, "points", []) or [])
                zone_size = float(zone.derived_mesh_size())
                poly_area = float(getattr(zone, "polygon_area", lambda: 0.0)())
            except Exception:
                pts = []
                zone_size = 0.0
                poly_area = 0.0
            if len(pts) < 3 or zone_size <= 0.0:
                continue
            hull = _polygon_convex_hull_ccw(pts)
            if len(hull) < 3:
                continue
            # Sharp inside-classifier: ≈1 for points inside the polygon,
            # ≈0 outside. Transition_scale narrow so even points 1 element
            # inside the boundary read as fully refined.
            char_len = _math.sqrt(max(poly_area, 1e-12))
            sharp_scale = max(min(h_bulk_v, char_len / 4.0) / 5.0, 1e-6)
            inside_expr = _polygon_inside_expression(hull, transition_scale=sharp_scale)
            if not inside_expr:
                continue
            # Smooth seeding-bias OUTSIDE: an exponential growth from zone_size
            # at the polygon boundary out to h_bulk far away. Scale controls
            # how wide the gradient ring is — bigger scale = more gradual.
            # Use the bigger of (h_bulk × 3) or (char_len × 0.5) so the
            # gradient is always at least a few bulk elements wide.
            gradient_scale = max(h_bulk_v * 3.0, char_len * 0.5, 1e-6)
            ext_dist_expr = _polygon_exterior_distance_expression(hull)
            # size(x, y) = zone_size + (1 - inside) * (h_bulk - zone_size)
            #                              * (1 - exp(-d_exterior / scale))
            #   inside the polygon (inside ≈ 1)             → zone_size
            #   at the polygon boundary (d_ext = 0)         → zone_size
            #   well outside the polygon (d_ext >> scale)   → h_bulk
            #   transition zone:                              smooth exp ramp
            zone_field_id = gmsh.model.mesh.field.add("MathEval")
            expr = (
                f"{zone_size:.10g} + (1.0 - ({inside_expr}))*"
                f"({h_bulk_v - zone_size:.10g})*"
                f"(1.0 - exp(-({ext_dist_expr})/({gradient_scale:.10g})))"
            )
            gmsh.model.mesh.field.setString(zone_field_id, "F", expr)
            threshold_ids.append(zone_field_id)

    # Edge seeds (Abaqus-style local seeds). Each seed produces either a
    # transfinite curve assignment (by_number) or one or more Distance +
    # MathEval size fields appended to threshold_ids. Min() combination at
    # the end gives local-seed-wins precedence as long as the seed size is
    # finer than other contributors — the common FEA case.
    if edge_seeds:
        curve_map = _curve_endpoint_map(gmsh)
        match_tol = max(float(sizing.h_feature) * 0.5, 1e-6)
        for seed in edge_seeds:
            if not getattr(seed, "is_valid", lambda: False)():
                continue
            matches = _resolve_seed(seed, curve_map, match_tol)
            if not matches:
                continue

            if seed.method == "by_number":
                n = max(int(seed.seed_count), 1)
                bias = seed.bias
                ratio = max(float(seed.bias_ratio or 1.0), 1.0)
                for ctag, fine_pt_tag, _elen in matches:
                    if bias == "none":
                        gmsh.model.mesh.setTransfiniteCurve(ctag, n + 1, "Progression", 1.0)
                    elif bias == "single":
                        per_elem_ratio = ratio ** (1.0 / max(n - 1, 1))
                        # gmsh's Progression coefficient: r>1 means "fine at
                        # the curve's start, coarse at its end" in the curve's
                        # natural orientation. Use 1/r when the user's fine
                        # end is the curve's end so the bias still points
                        # toward the user-intended fine vertex.
                        p1_tag, _, p2_tag, _ = curve_map[ctag]
                        coeff = per_elem_ratio if fine_pt_tag == p1_tag else 1.0 / per_elem_ratio
                        gmsh.model.mesh.setTransfiniteCurve(ctag, n + 1, "Progression", coeff)
                    else:  # double
                        gmsh.model.mesh.setTransfiniteCurve(ctag, n + 1, "Bump", 1.0 / ratio)
                continue

            # method == "by_size"
            h_bulk = float(sizing.h_bulk)
            if seed.bias == "none":
                h_seed = float(seed.element_size)
                if h_seed <= 0:
                    continue
                ctags = [m[0] for m in matches]
                falloff = max(matches, key=lambda m: m[2])[2] / 3.0
                falloff = max(falloff, 1e-6)
                dist_id = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(dist_id, "CurvesList", ctags)
                gmsh.model.mesh.field.setNumber(dist_id, "Sampling", 200)
                math_id = gmsh.model.mesh.field.add("MathEval")
                delta = h_bulk - h_seed
                expr = f"{h_seed:.10g} + ({delta:.10g})*(1.0 - exp(-F{dist_id}/{falloff:.10g}))"
                gmsh.model.mesh.field.setString(math_id, "F", expr)
                threshold_ids.append(math_id)
            else:
                # bias == "single" or "double" — by_size with min/max ends.
                # NOTE: flip_bias is already encoded in `fine_pt_tag` (we pick
                # the opposite endpoint when flipped). Don't double-flip via
                # effective_min_max here; just normalize so h_fine ≤ h_coarse.
                h_fine = float(min(seed.min_size, seed.max_size))
                h_coarse = float(max(seed.min_size, seed.max_size))
                if h_fine <= 0 or h_coarse <= 0:
                    continue
                for ctag, fine_pt_tag, edge_len in matches:
                    delta = h_coarse - h_fine
                    if seed.bias == "single":
                        scale = max(edge_len / 3.0, 1e-6)
                        dist_id = gmsh.model.mesh.field.add("Distance")
                        gmsh.model.mesh.field.setNumbers(dist_id, "PointsList", [fine_pt_tag])
                        gmsh.model.mesh.field.setNumber(dist_id, "Sampling", 100)
                        math_id = gmsh.model.mesh.field.add("MathEval")
                        expr = f"{h_fine:.10g} + ({delta:.10g})*(1.0 - exp(-F{dist_id}/{scale:.10g}))"
                        gmsh.model.mesh.field.setString(math_id, "F", expr)
                        threshold_ids.append(math_id)
                    else:  # double
                        p1_tag, _, p2_tag, _ = curve_map[ctag]
                        scale = max(edge_len / 6.0, 1e-6)
                        d1 = gmsh.model.mesh.field.add("Distance")
                        gmsh.model.mesh.field.setNumbers(d1, "PointsList", [p1_tag])
                        gmsh.model.mesh.field.setNumber(d1, "Sampling", 100)
                        d2 = gmsh.model.mesh.field.add("Distance")
                        gmsh.model.mesh.field.setNumbers(d2, "PointsList", [p2_tag])
                        gmsh.model.mesh.field.setNumber(d2, "Sampling", 100)
                        e1 = gmsh.model.mesh.field.add("MathEval")
                        gmsh.model.mesh.field.setString(
                            e1, "F",
                            f"{h_fine:.10g} + ({delta:.10g})*(1.0 - exp(-F{d1}/{scale:.10g}))",
                        )
                        e2 = gmsh.model.mesh.field.add("MathEval")
                        gmsh.model.mesh.field.setString(
                            e2, "F",
                            f"{h_fine:.10g} + ({delta:.10g})*(1.0 - exp(-F{d2}/{scale:.10g}))",
                        )
                        m_id = gmsh.model.mesh.field.add("Min")
                        gmsh.model.mesh.field.setNumbers(m_id, "FieldsList", [e1, e2])
                        threshold_ids.append(m_id)

    # Vertex seeds (Abaqus "Seed at vertex"). Each seed anchors a Distance
    # field on the picked gmsh point and grows the size from `target_size`
    # at the vertex to h_bulk over `influence_radius`. The MathEval ramp
    # uses the same exponential form as edge seeds so transitions are smooth.
    if vertex_seeds:
        # Reuse the same curve_map if we have it; otherwise build it now.
        try:
            _ = curve_map  # noqa: F821  — bound earlier when edge_seeds present
        except NameError:
            curve_map = _curve_endpoint_map(gmsh)
        vmatch_tol = max(float(sizing.h_feature) * 0.5, 1e-6)
        for vseed in vertex_seeds:
            if not getattr(vseed, "is_valid", lambda: False)():
                continue
            pt_tag = _resolve_vertex_seed(vseed, curve_map, vmatch_tol)
            if pt_tag is None:
                continue
            h_target = float(vseed.target_size)
            h_bulk_v = float(sizing.h_bulk)
            radius = max(float(vseed.influence_radius), 1e-9)
            scale = max(radius / 3.0, 1e-9)
            delta = h_bulk_v - h_target
            dist_id = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(dist_id, "PointsList", [int(pt_tag)])
            gmsh.model.mesh.field.setNumber(dist_id, "Sampling", 50)
            math_id = gmsh.model.mesh.field.add("MathEval")
            expr = f"{h_target:.10g} + ({delta:.10g})*(1.0 - exp(-F{dist_id}/{scale:.10g}))"
            gmsh.model.mesh.field.setString(math_id, "F", expr)
            threshold_ids.append(math_id)

    # Boundary-layer seeds (Abaqus "Inflation"). gmsh has a dedicated
    # `BoundaryLayer` field that emits stacked thin elements parallel to a
    # set of curves, growing geometrically into the interior. It is set via
    # `setAsBoundaryLayer` (NOT setAsBackgroundMesh) and combines naturally
    # with the regular size field — interior elements still follow the
    # Min-combined size field; only the near-edge band gets layered.
    if boundary_layer_seeds:
        try:
            _ = curve_map  # noqa: F821 — reuse from edge/vertex blocks if set
        except NameError:
            curve_map = _curve_endpoint_map(gmsh)
        bl_match_tol = max(float(sizing.h_feature) * 0.5, 1e-6)
        for bls in boundary_layer_seeds:
            if not getattr(bls, "is_valid", lambda: False)():
                continue
            # Resolve each edge_ref to a curve tag (reuse edge-seed resolver).
            seed_proxy = type("_BL", (), {
                "edge_refs": bls.edge_refs,
                "flip_bias": False,
            })()
            matches = _resolve_seed(seed_proxy, curve_map, bl_match_tol)
            curve_tags = sorted({int(m[0]) for m in matches})
            if not curve_tags:
                continue
            bl_id = gmsh.model.mesh.field.add("BoundaryLayer")
            gmsh.model.mesh.field.setNumbers(bl_id, "CurvesList", curve_tags)
            gmsh.model.mesh.field.setNumber(bl_id, "Size", float(bls.first_layer_size))
            gmsh.model.mesh.field.setNumber(bl_id, "Ratio", float(bls.growth_ratio))
            # Total thickness: either user-capped or derived from layer geometry.
            total = float(bls.total_thickness())
            if total > 0:
                gmsh.model.mesh.field.setNumber(bl_id, "Thickness", total)
            try:
                gmsh.model.mesh.field.setNumber(bl_id, "Quads", 1 if bls.quads else 0)
            except Exception:
                pass
            try:
                gmsh.model.mesh.field.setAsBoundaryLayer(bl_id)
            except Exception:
                # Older gmsh versions may use a different API; degrade
                # gracefully by treating it as a normal size field
                # (Distance + Threshold) so the seed at least refines the
                # near-edge region.
                math_id = gmsh.model.mesh.field.add("MathEval")
                d_fallback = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(d_fallback, "CurvesList", curve_tags)
                gmsh.model.mesh.field.setString(
                    math_id, "F",
                    f"{float(bls.first_layer_size):.10g} + "
                    f"({float(sizing.h_bulk) - float(bls.first_layer_size):.10g})*"
                    f"(1.0 - exp(-F{d_fallback}/{max(total, 1e-9):.10g}))",
                )
                threshold_ids.append(math_id)

    # Curvature control (Abaqus "Curvature control" checkbox on by-size, no-bias
    # seeds). gmsh's Mesh.MeshSizeFromCurvature option is shared by the whole
    # model, so we collapse the per-seed requests into one. With N seeds asking
    # for curvature control, we honor the most aggressive (smallest) deviation
    # factor and the smallest min-size factor — that way every request is at
    # least satisfied.
    #
    # Abaqus uses the chord-to-edge-length ratio h/L. gmsh uses the count of
    # elements per full 2π turn. For small angles the two are related by:
    #     N ≈ π / (4·(h/L))
    # which gives N ≈ 8 elements per circle for the dialog's default h/L = 0.1.
    if edge_seeds:
        import math as _math
        curv_seeds = [
            s for s in edge_seeds
            if getattr(s, "curvature_control", False)
            and getattr(s, "method", "") == "by_size"
            and getattr(s, "bias", "") == "none"
        ]
        if curv_seeds:
            smallest_f = min(float(s.max_deviation_factor) for s in curv_seeds)
            smallest_f = max(smallest_f, 1e-4)
            n_per_2pi = _math.pi / (4.0 * smallest_f)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", n_per_2pi)
            smallest_min_factor = min(float(s.min_size_factor or 0.1) for s in curv_seeds)
            smallest_min_factor = max(smallest_min_factor, 0.001)
            h_min = float(sizing.h_bulk) * smallest_min_factor
            gmsh.option.setNumber("Mesh.MeshSizeMin", h_min)

    if not threshold_ids:
        # Single-part with no interfaces, no holes, no zones: enforce a uniform bulk size.
        const_id = gmsh.model.mesh.field.add("Constant")
        gmsh.model.mesh.field.setNumber(const_id, "VIn", float(sizing.h_bulk))
        gmsh.model.mesh.field.setNumber(const_id, "VOut", float(sizing.h_bulk))
        gmsh.model.mesh.field.setAsBackgroundMesh(const_id)
        return const_id

    if len(threshold_ids) == 1:
        gmsh.model.mesh.field.setAsBackgroundMesh(threshold_ids[0])
        return threshold_ids[0]

    min_id = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_id, "FieldsList", threshold_ids)
    gmsh.model.mesh.field.setAsBackgroundMesh(min_id)
    return min_id


def generate_surface_mesh(parts, sizing, *, dedup_tol=None, custom_zones=None, edge_seeds=None, part_mesh_overrides=None, vertex_seeds=None, boundary_layer_seeds=None, matched_edge_pairs=None):
    """Heterogeneous multi-part 2D mesh with adaptive sizing.

    Args:
        parts: iterable of GmshPartSpec
        sizing: MeshSizingPolicy (or None for uniform h = sizing.h_bulk only)
        dedup_tol: coordinate rounding tolerance for shared-edge deduplication.
            Defaults to h_feature * 0.05 so interfaces resolve, but is bounded
            below by 1e-9 to avoid pathological coincidence.
        custom_zones: optional iterable of CustomMeshZone for local refinement.

    Returns:
        dict with keys:
            nodes: (N, 2) ndarray of x,y
            triangles: (M, 3) ndarray of node indices
            triangle_part_ids: (M,) ndarray of int part ids
            surfaces: dict[part_id -> gmsh surface tag]
            interface_curves: list of curve tags shared by >= 2 surfaces
            hole_curves: list of inner-loop curve tags
    """
    if not verify_gmsh_available():
        raise GmshError(
            "Gmsh Python API not available. Open Dependency Check and install 'gmsh'."
        )
    try:
        import gmsh
    except Exception as exc:
        raise GmshError("Gmsh Python API not available.") from exc

    parts = list(parts or [])
    if not parts:
        raise GmshError("No parts to mesh.")
    if sizing is None:
        raise GmshError("MeshSizingPolicy is required for the heterogeneous 2D mesher.")

    if dedup_tol is None:
        dedup_tol = max(float(sizing.h_feature) * 0.05, 1e-9)
    else:
        dedup_tol = max(float(dedup_tol), 1e-12)

    try:
        if hasattr(gmsh, "isInitialized") and gmsh.isInitialized():
            try:
                with _SuppressSignalInWorker():
                    gmsh.finalize()
            except Exception:
                pass
    except Exception:
        pass
    with _SuppressSignalInWorker():
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cpd_simstudio_2d")
    try:
        surfaces, curve_to_surfaces, hole_curves = _build_gmsh_2d_geometry(
            gmsh, parts, dedup_tol
        )
        # _build_gmsh_2d_geometry already calls gmsh.model.occ.synchronize().

        # Register one physical group per part (dim=2). The group tag is the part id.
        part_to_phys = {}
        seen_pids = set()
        for part in parts:
            pid = int(part.id)
            if pid in seen_pids:
                continue
            seen_pids.add(pid)
            surf_tags = surfaces.get(pid, [])
            if not surf_tags:
                continue
            phys = gmsh.model.addPhysicalGroup(2, list(surf_tags), tag=pid)
            gmsh.model.setPhysicalName(2, phys, str(part.name or f"part_{pid}"))
            part_to_phys[pid] = phys

        interface_curves, hole_curve_list, boundary_curve_list = _classify_feature_curves(
            curve_to_surfaces, hole_curves
        )

        _install_size_field(
            gmsh,
            sizing,
            interface_curves,
            hole_curve_list,
            custom_zones=custom_zones,
            boundary_curves=boundary_curve_list,
            edge_seeds=edge_seeds,
            part_mesh_overrides=part_mesh_overrides,
            surfaces=surfaces,
            vertex_seeds=vertex_seeds,
            boundary_layer_seeds=boundary_layer_seeds,
        )

        # Generate
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.RecombineAll", 0)
        # Frontal-Delaunay (algorithm 6) honours the size field much more
        # smoothly than the default MeshAdapt — produces a gradual gradient
        # from h_feature near boundaries out to h_bulk in the interior,
        # instead of a sharp ring of fine elements followed by an abrupt
        # jump to bulk size.
        try:
            gmsh.option.setNumber("Mesh.Algorithm", 6)
        except Exception:
            pass

        # Edge-pair periodicity. Each matched pair forces the slave edge's
        # mesh nodes to mirror the master's, via gmsh.setPeriodic with an
        # affine translation transform. v1 supports translation-only pairs
        # (the most common case — parallel edges on opposite sides of a
        # rectangular domain or repeated geometry). Pairs that don't match
        # by translation are silently skipped.
        if matched_edge_pairs:
            try:
                _ = curve_map  # noqa: F821 reuse if already built
            except NameError:
                curve_map = _curve_endpoint_map(gmsh)
            mp_tol = max(float(sizing.h_feature) * 0.5, 1e-6)
            for pair in matched_edge_pairs:
                if not getattr(pair, "is_valid", lambda: False)():
                    continue
                # Resolve master and slave to curve tags.
                m_proxy = type("_P", (), {"edge_refs": [pair.master], "flip_bias": False})()
                s_proxy = type("_P", (), {"edge_refs": [pair.slave], "flip_bias": False})()
                m_match = _resolve_seed(m_proxy, curve_map, mp_tol)
                s_match = _resolve_seed(s_proxy, curve_map, mp_tol)
                if not m_match or not s_match:
                    continue
                m_tag = int(m_match[0][0])
                s_tag = int(s_match[0][0])
                # Translation that maps slave coordinates → master.
                # Use start-to-start; if lengths/orientations don't match
                # cleanly the translation is a best-effort approximation.
                try:
                    ms = pair.master["start"]; me = pair.master["end"]
                    ss = pair.slave["start"];  se = pair.slave["end"]
                    tx = ms[0] - ss[0]; ty = ms[1] - ss[1]
                    # 4x4 row-major affine.
                    affine = [
                        1.0, 0.0, 0.0, tx,
                        0.0, 1.0, 0.0, ty,
                        0.0, 0.0, 1.0, 0.0,
                        0.0, 0.0, 0.0, 1.0,
                    ]
                    gmsh.model.mesh.setPeriodic(1, [s_tag], [m_tag], affine)
                except Exception:
                    pass

        gmsh.model.mesh.generate(2)

        # Collect nodes
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        if len(node_tags) == 0:
            raise GmshError("Surface mesher produced no nodes.")
        coords = np.asarray(coords, dtype=float).reshape(-1, 3)[:, :2]
        tag_to_index = {int(t): i for i, t in enumerate(node_tags)}

        # Collect triangles per physical group so we know each triangle's part_id.
        all_tris = []
        all_part_ids = []
        for pid, phys_tag in part_to_phys.items():
            ent_tags = gmsh.model.getEntitiesForPhysicalGroup(2, phys_tag)
            for ent in ent_tags:
                etypes, etags, enodes = gmsh.model.mesh.getElements(2, int(ent))
                for etype, _tags, conn in zip(etypes, etags, enodes):
                    if etype != 2:  # 3-node triangle
                        continue
                    conn = np.asarray(conn, dtype=int).reshape(-1, 3)
                    for tri in conn:
                        all_tris.append([tag_to_index[int(n)] for n in tri])
                        all_part_ids.append(int(pid))

        if not all_tris:
            raise GmshError("Surface mesher produced no triangles.")

        return {
            "nodes": np.asarray(coords, dtype=float),
            "triangles": np.asarray(all_tris, dtype=int),
            "triangle_part_ids": np.asarray(all_part_ids, dtype=int),
            "surfaces": surfaces,
            "interface_curves": list(interface_curves),
            "hole_curves": list(hole_curve_list),
        }
    finally:
        try:
            with _SuppressSignalInWorker():
                gmsh.finalize()
        except Exception:
            pass


def generate_volume_mesh_from_cad(cad_paths, mesh_size, mesh_type="tetra", fallback_tetra=True):
    if not verify_gmsh_available():
        raise GmshError(
            "Gmsh Python API not available. Open Dependency Check and install 'gmsh'."
        )
    try:
        import gmsh
    except Exception as exc:
        raise GmshError(
            "Gmsh Python API not available. Install with 'sudo apt install python3-gmsh' "
            "or 'pip install gmsh'."
        ) from exc

    if not cad_paths:
        raise GmshError("No CAD files provided for meshing.")

    try:
        if hasattr(gmsh, "isInitialized") and gmsh.isInitialized():
            try:
                gmsh.finalize()
            except Exception:
                pass
    except Exception:
        pass
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cpd_simstudio_cad")
    try:
        for path in cad_paths:
            try:
                gmsh.model.occ.importShapes(path)
            except Exception as exc:
                raise GmshError(f"Failed to import CAD file into Gmsh: {path}") from exc
        gmsh.model.occ.synchronize()
        volumes = gmsh.model.occ.getEntities(3)
        if not volumes:
            raise GmshError(
                "No 3D volumes found in CAD import. "
                "Use STEP/IGES solids for volumetric meshing."
            )
        return _generate_mesh(gmsh, mesh_size, mesh_type, fallback_tetra)
    finally:
        gmsh.finalize()
