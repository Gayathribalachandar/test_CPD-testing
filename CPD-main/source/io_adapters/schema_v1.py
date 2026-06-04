import csv
import os
import re
from pathlib import Path
from collections import deque

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_DEFAULT = PROJECT_ROOT.parent / "workspace"


def _normalize_col_name(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _parse_int(value):
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def _parse_float(value):
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _read_dict_rows(path: Path):
    if not path.exists():
        return None, []

    lines = []
    with path.open("r", newline="") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            text = stripped.lstrip("\ufeff")
            if text.startswith("#"):
                continue
            lines.append(line)

    if not lines:
        return None, []

    lines[0] = lines[0].lstrip("\ufeff")
    reader = csv.DictReader(lines)
    if not reader.fieldnames:
        return None, []

    fieldnames = [(h.lstrip("\ufeff") if h else h) for h in reader.fieldnames]
    rows = []
    for row in reader:
        fixed = {}
        for key, value in row.items():
            fixed[(key.lstrip("\ufeff") if key else key)] = value
        rows.append(fixed)
    return fieldnames, rows


def _load_config():
    config_path = PROJECT_ROOT / "config.yml"
    if not config_path.exists():
        return {}
    with config_path.open("r") as f:
        data = yaml.safe_load(f)
    return data or {}


def _load_particles(input_dir: Path):
    path = input_dir / "particles.csv"
    fieldnames, rows = _read_dict_rows(path)
    if not fieldnames:
        raise ValueError(f"No particle rows found in {path}")

    colmap = {_normalize_col_name(h): h for h in fieldnames if h}
    id_col = colmap.get("particleid") or colmap.get("nodeid") or colmap.get("id")
    x_col = colmap.get("x") or colmap.get("posx") or colmap.get("coordx")
    y_col = colmap.get("y") or colmap.get("posy") or colmap.get("coordy")
    if x_col is None or y_col is None:
        raise ValueError(f"Missing x/y columns in {path}")

    pos_rows = []
    raw_ids = []
    for idx, row in enumerate(rows):
        x = _parse_float(row.get(x_col))
        y = _parse_float(row.get(y_col))
        if x is None or y is None:
            continue
        pid = _parse_int(row.get(id_col)) if id_col else idx
        if pid is None:
            continue
        raw_ids.append(pid)
        pos_rows.append([x, y])

    if not pos_rows:
        raise ValueError(f"No valid particle coordinates in {path}")

    pos = np.asarray(pos_rows, dtype=float)
    if np.max(np.abs(pos)) > 10.0:
        print("Warning: particle coordinates exceed expected range.")
        print("Check unit conversion (mm vs m).")
    raw_ids = np.asarray(raw_ids, dtype=np.int64)
    expected = np.arange(raw_ids.size, dtype=np.int64)

    remap_needed = bool(np.any(raw_ids != expected))
    remap_array = None
    id_to_index = {}

    if remap_needed:
        for idx, rid in enumerate(raw_ids.tolist()):
            if rid in id_to_index:
                raise ValueError(f"Duplicate particle_id {rid} in {path}")
            id_to_index[rid] = idx
        max_id = int(raw_ids.max()) if raw_ids.size > 0 else -1
        remap_array = np.full((max_id + 1,), -1, dtype=np.int64)
        for rid, idx in id_to_index.items():
            if rid >= 0:
                remap_array[rid] = idx
        print(f"Particle ids are non-sequential in {path.name}; applying remap to connectivity/targets.")
    else:
        id_to_index = {int(i): int(i) for i in expected}
        remap_array = expected.copy()

    print(f"Particles: {pos.shape[0]}")

    return pos, remap_needed, id_to_index, remap_array


def _remap_ids(arr: np.ndarray, id_to_index: dict[int, int], context: str):
    if arr.size == 0:
        return arr.astype(np.int64, copy=False)
    out = np.empty_like(arr, dtype=np.int64)
    flat = arr.reshape(-1)
    out_flat = out.reshape(-1)
    for i, old in enumerate(flat):
        key = int(old)
        if key not in id_to_index:
            raise ValueError(f"{context}: unknown particle_id {key} during remap")
        out_flat[i] = id_to_index[key]
    return out


def _load_triangles(input_dir: Path):
    path = input_dir / "connections.csv"
    fieldnames, rows = _read_dict_rows(path)
    if not fieldnames:
        raise ValueError(f"No triangle rows found in {path}")

    colmap = {_normalize_col_name(h): h for h in fieldnames if h}
    p1_col = colmap.get("p1")
    p2_col = colmap.get("p2")
    p3_col = colmap.get("p3")
    mat_col = colmap.get("materialid")
    if p1_col is None or p2_col is None or p3_col is None or mat_col is None:
        raise ValueError(f"connections.csv must include p1,p2,p3,material_id columns: {path}")

    tri_rows = []
    tri_material = []
    override_cols = {
        "E": colmap.get("e"),
        "Nu": colmap.get("nu"),
        "rho": colmap.get("rho"),
        "fail_SE": colmap.get("failse"),
        "c": colmap.get("c"),
    }
    tri_overrides = {key: [] for key in override_cols}
    for row in rows:
        p1 = _parse_int(row.get(p1_col))
        p2 = _parse_int(row.get(p2_col))
        p3 = _parse_int(row.get(p3_col))
        if p1 is None or p2 is None or p3 is None:
            continue
        tri_rows.append([p1, p2, p3])
        mid = _parse_int(row.get(mat_col))
        if mid is None:
            raise ValueError(f"connections.csv contains triangle with missing material_id: {path}")
        tri_material.append(mid)
        for key, col in override_cols.items():
            value = _parse_float(row.get(col)) if col is not None else None
            tri_overrides[key].append(np.nan if value is None else float(value))

    if not tri_rows:
        raise ValueError(f"No valid triangle connectivity found in {path}")

    tri_material_arr = np.asarray(tri_material, dtype=np.int64)
    tri_override_arrs = {
        key: np.asarray(values, dtype=float) for key, values in tri_overrides.items()
    }
    return np.asarray(tri_rows, dtype=np.int64), tri_material_arr, tri_override_arrs


def _load_materials(input_dir: Path):
    path = input_dir / "materials.csv"
    fieldnames, rows = _read_dict_rows(path)
    if not fieldnames:
        raise ValueError(f"No material rows found in {path}")

    materials = {}
    rigid_material_ids = set()
    colmap = {_normalize_col_name(h): h for h in fieldnames if h}
    required_cols = {
        "materialid": "material_id",
        "name": "name",
        "e": "E",
        "nu": "nu",
        "rho": "rho",
        "failse": "fail_SE",
        "c": "c",
    }
    missing = [label for norm, label in required_cols.items() if colmap.get(norm) is None]
    if missing:
        raise ValueError(f"materials.csv missing required columns: {', '.join(missing)}")

    behavior_col = colmap.get("behavior")
    symmetry_col = colmap.get("symmetry")
    damage_col = colmap.get("damage")
    reserved_cols = set(required_cols.keys()) | {"behavior", "symmetry", "damage"}
    extra_cols = {
        norm: original
        for norm, original in colmap.items()
        if norm not in reserved_cols
    }

    for row in rows:
        mid = _parse_int(row.get(colmap["materialid"]))
        if mid is None:
            continue
        name = str(row.get(colmap["name"], "")).strip().lower()
        vals = {
            "E": _parse_float(row.get(colmap["e"])),
            "Nu": _parse_float(row.get(colmap["nu"])),
            "rho": _parse_float(row.get(colmap["rho"])),
            "fail_SE": _parse_float(row.get(colmap["failse"])),
            "c": _parse_float(row.get(colmap["c"])),
        }
        if any(v is None for v in vals.values()):
            raise ValueError(f"materials.csv contains non-numeric material row for material_id {mid}")
        parameters = {}
        for norm_key, original_key in extra_cols.items():
            raw = row.get(original_key)
            if raw in (None, ""):
                continue
            parsed = _parse_float(raw)
            parameters[original_key] = float(parsed) if parsed is not None else raw
        materials[mid] = {
            **{key: float(value) for key, value in vals.items()},
            "behavior": str(row.get(behavior_col, "") or "").strip().lower() if behavior_col else "",
            "symmetry": str(row.get(symmetry_col, "") or "").strip().lower() if symmetry_col else "",
            "damage": str(row.get(damage_col, "") or "").strip().lower() if damage_col else "",
            "parameters": parameters,
        }
        if vals["E"] >= 1e14 or "rigid" in name or materials[mid]["behavior"] == "rigid":
            rigid_material_ids.add(mid)

    if not materials:
        raise ValueError(f"No valid materials found in {path}")

    return materials, rigid_material_ids


def _build_tri_material_arrays(tri_material, materials, tri_overrides=None):
    if tri_material is None or tri_material.size == 0:
        raise ValueError("connections.csv contains no triangle material ids")

    tri_material_arr = np.asarray(tri_material, dtype=np.int64).reshape(-1)
    missing_ids = [
        int(mid)
        for mid in np.unique(tri_material_arr)
        if int(mid) not in materials
    ]
    if missing_ids:
        shown = ", ".join(str(v) for v in missing_ids[:10])
        extra = ", ..." if len(missing_ids) > 10 else ""
        raise ValueError(f"Missing material ids in materials.csv: {shown}{extra}")

    n_tri = tri_material_arr.shape[0]
    E_tri = np.empty((n_tri,), dtype=float)
    Nu_tri = np.empty((n_tri,), dtype=float)
    rho_tri = np.empty((n_tri,), dtype=float)
    fail_SE_tri = np.empty((n_tri,), dtype=float)
    c_tri = np.empty((n_tri,), dtype=float)
    overrides = tri_overrides or {}
    E_override = np.asarray(overrides.get("E", np.full((n_tri,), np.nan)), dtype=float).reshape(-1)
    Nu_override = np.asarray(overrides.get("Nu", np.full((n_tri,), np.nan)), dtype=float).reshape(-1)
    rho_override = np.asarray(overrides.get("rho", np.full((n_tri,), np.nan)), dtype=float).reshape(-1)
    fail_override = np.asarray(overrides.get("fail_SE", np.full((n_tri,), np.nan)), dtype=float).reshape(-1)
    c_override = np.asarray(overrides.get("c", np.full((n_tri,), np.nan)), dtype=float).reshape(-1)

    for i, mid in enumerate(tri_material_arr):
        mat = materials[int(mid)]
        E_tri[i] = float(E_override[i]) if np.isfinite(E_override[i]) else float(mat["E"])
        Nu_tri[i] = float(Nu_override[i]) if np.isfinite(Nu_override[i]) else float(mat["Nu"])
        rho_tri[i] = float(rho_override[i]) if np.isfinite(rho_override[i]) else float(mat["rho"])
        fail_SE_tri[i] = float(fail_override[i]) if np.isfinite(fail_override[i]) else float(mat["fail_SE"])
        c_tri[i] = float(c_override[i]) if np.isfinite(c_override[i]) else float(mat["c"])

    return E_tri, Nu_tri, rho_tri, fail_SE_tri, c_tri


def _build_rigid_structures(tri, tri_material, rigid_material_ids):
    rigid_bodies = {}
    if (
        tri is None
        or tri_material is None
        or tri.size == 0
        or tri_material.size == 0
        or not rigid_material_ids
    ):
        return np.zeros((0,), dtype=bool), rigid_bodies

    tri_material_arr = np.asarray(tri_material, dtype=np.int64).reshape(-1)
    tri = np.asarray(tri, dtype=np.int64)
    rigid_material_ids = np.array(sorted(set(int(v) for v in rigid_material_ids)), dtype=np.int64)

    rigid_tri_mask = np.isin(tri_material_arr, rigid_material_ids)
    if rigid_tri_mask.size == 0:
        return rigid_tri_mask.astype(bool, copy=False), rigid_bodies

    rigid_triangle_nodes = np.unique(tri[rigid_tri_mask].ravel()) if np.any(rigid_tri_mask) else np.array([], dtype=np.int64)
    non_rigid_nodes = np.unique(tri[~rigid_tri_mask].ravel()) if np.any(~rigid_tri_mask) else np.array([], dtype=np.int64)
    shared_nodes = np.intersect1d(rigid_triangle_nodes, non_rigid_nodes) if rigid_triangle_nodes.size > 0 else np.array([], dtype=np.int64)
    if shared_nodes.size > 0:
        print(f"Rigid/enforced nodes shared with non-rigid triangles excluded: {shared_nodes.size}")

    for mid in rigid_material_ids.tolist():
        tri_idx = np.flatnonzero(tri_material_arr == mid)
        if tri_idx.size == 0:
            continue

        local_to_global = tri_idx
        n_local = tri_idx.size
        edge_to_tris = {}
        adj = [[] for _ in range(n_local)]

        for local_idx, tri_global_idx in enumerate(local_to_global):
            nodes = tri[tri_global_idx]
            edges = [(nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0])]
            for i_node, j_node in edges:
                n0, n1 = (int(i_node), int(j_node))
                if n0 > n1:
                    n0, n1 = n1, n0
                edge = (n0, n1)
                neighbours = edge_to_tris.get(edge)
                if neighbours is None:
                    edge_to_tris[edge] = [local_idx]
                else:
                    for other in neighbours:
                        adj[local_idx].append(other)
                        adj[other].append(local_idx)
                    neighbours.append(local_idx)

        comps = []
        seen = np.zeros((n_local,), dtype=bool)
        for start in range(n_local):
            if seen[start]:
                continue

            queue = deque([start])
            seen[start] = True
            comp = []
            while queue:
                cur = queue.popleft()
                comp.append(cur)
                for nb in adj[cur]:
                    if not seen[nb]:
                        seen[nb] = True
                        queue.append(nb)
            comps.append(comp)

        body_nodes = []
        for comp in comps:
            comp_nodes = np.unique(tri[local_to_global[comp]].ravel())
            if shared_nodes.size > 0:
                comp_nodes = np.setdiff1d(comp_nodes, shared_nodes, assume_unique=False)
            if comp_nodes.size > 0:
                body_nodes.append(comp_nodes.astype(np.int32, copy=False))

        if body_nodes:
            rigid_bodies[int(mid)] = body_nodes

    return rigid_tri_mask.astype(bool, copy=False), rigid_bodies


def _load_targets(path: Path, id_to_index: dict[int, int]):
    fieldnames, rows = _read_dict_rows(path)
    if not fieldnames:
        return {}

    colmap = {_normalize_col_name(h): h for h in fieldnames if h}
    pid_col = colmap.get("particleid") or colmap.get("nodeid") or colmap.get("id")
    if pid_col is None:
        return {}

    id_cols = [h for h in fieldnames if h != pid_col]
    targets = {}
    for row in rows:
        raw_pid = _parse_int(row.get(pid_col))
        if raw_pid is None:
            continue
        pid = id_to_index.get(int(raw_pid))
        if pid is None:
            continue
        for col in id_cols:
            tid = _parse_int(row.get(col))
            if tid is None:
                continue
            targets.setdefault(int(tid), []).append(int(pid))

    return {k: np.unique(np.asarray(v, dtype=np.int32)) for k, v in targets.items() if len(v) > 0}


def _load_time_profiles(path: Path, prefix: str, default_id: int = 1):
    profiles = {}
    fieldnames, rows = _read_dict_rows(path)
    if not fieldnames:
        return profiles

    colmap = {_normalize_col_name(h): h for h in fieldnames if h}
    time_col = colmap.get("time") or colmap.get("t")
    if time_col is None:
        return profiles

    x_cols = {}
    y_cols = {}
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)(x|y)$")

    for original in fieldnames:
        if not original:
            continue
        norm = _normalize_col_name(original)
        if norm in (f"{prefix}x", f"{prefix}y"):
            if norm.endswith("x"):
                x_cols[default_id] = original
            else:
                y_cols[default_id] = original
            continue
        m = pattern.match(norm)
        if not m:
            continue
        pid = int(m.group(1))
        axis = m.group(2)
        if axis == "x":
            x_cols[pid] = original
        else:
            y_cols[pid] = original

    profile_ids = sorted(set(x_cols.keys()) | set(y_cols.keys()))
    if not profile_ids:
        return profiles

    data = {pid: {"t": [], "x": [], "y": []} for pid in profile_ids}

    for row in rows:
        t = _parse_float(row.get(time_col))
        if t is None:
            continue
        for pid in profile_ids:
            x = _parse_float(row.get(x_cols.get(pid))) if pid in x_cols else 0.0
            y = _parse_float(row.get(y_cols.get(pid))) if pid in y_cols else 0.0
            if x is None:
                x = 0.0
            if y is None:
                y = 0.0
            data[pid]["t"].append(t)
            data[pid]["x"].append(x)
            data[pid]["y"].append(y)

    for pid, series in data.items():
        if not series["t"]:
            continue
        t = np.asarray(series["t"], dtype=float)
        x = np.asarray(series["x"], dtype=float)
        y = np.asarray(series["y"], dtype=float)
        order = np.argsort(t, kind="stable")
        profiles[pid] = (t[order], x[order], y[order])

    return profiles


def _load_fixed_bc(workspace_dir: Path, id_to_index: dict[int, int]):
    path = workspace_dir / "input" / "fixed.csv"
    fieldnames, rows = _read_dict_rows(path)
    if not fieldnames:
        return np.empty((0, 1), dtype=np.int32)

    colmap = {_normalize_col_name(h): h for h in fieldnames if h}
    id_col = colmap.get("particleid") or colmap.get("nodeid") or colmap.get("id")
    if id_col is None:
        return np.empty((0, 1), dtype=np.int32)

    out = []
    for row in rows:
        raw = _parse_int(row.get(id_col))
        if raw is None:
            continue
        remapped = id_to_index.get(int(raw))
        if remapped is None:
            continue
        out.append([remapped])

    if not out:
        return np.empty((0, 1), dtype=np.int32)
    return np.asarray(out, dtype=np.int32)


def _filter_near_zero_triangles(pos, tri, tri_material, area_eps, tri_overrides=None):
    if tri is None or tri.size == 0:
        return tri, tri_material, tri_overrides, 0

    p1 = pos[tri[:, 0]]
    p2 = pos[tri[:, 1]]
    p3 = pos[tri[:, 2]]
    signed = 0.5 * (
        p1[:, 0] * (p2[:, 1] - p3[:, 1])
        + p2[:, 0] * (p3[:, 1] - p1[:, 1])
        + p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    keep = np.abs(signed) >= area_eps
    near_zero_count = int(np.count_nonzero(~keep))
    if near_zero_count > 0:
        print(f"Near-zero triangles excluded: {near_zero_count} (area_eps={area_eps})")
    tri_out = tri[keep]
    tri_mat_out = tri_material[keep] if tri_material is not None else None
    overrides_out = None
    if isinstance(tri_overrides, dict):
        overrides_out = {}
        for key, values in tri_overrides.items():
            arr = np.asarray(values, dtype=float).reshape(-1)
            overrides_out[key] = arr[keep] if arr.size == keep.size else arr
    return tri_out, tri_mat_out, overrides_out, near_zero_count


def _compute_dx_min(pos: np.ndarray, tri: np.ndarray) -> float:
    if tri is None or tri.size == 0:
        raise ValueError("Cannot compute dx_min without triangle connectivity")
    tri = np.asarray(tri, dtype=np.int64)
    pos = np.asarray(pos, dtype=float)
    p1 = pos[tri[:, 0]]
    p2 = pos[tri[:, 1]]
    p3 = pos[tri[:, 2]]
    e12 = np.linalg.norm(p1 - p2, axis=1)
    e23 = np.linalg.norm(p2 - p3, axis=1)
    e31 = np.linalg.norm(p3 - p1, axis=1)
    edge_lengths = np.concatenate((e12, e23, e31))
    positive = edge_lengths[edge_lengths > 0.0]
    if positive.size == 0:
        raise ValueError("Cannot compute dx_min because all triangle edges are zero")
    return float(np.min(positive))


def _print_startup_diagnostics(
    pos: np.ndarray,
    tri: np.ndarray,
    materials: dict[int, dict[str, float]],
    E_tri: np.ndarray,
    rho_tri: np.ndarray,
    dt: float,
):
    dx_min = _compute_dx_min(pos, tri)
    wave_speeds = np.sqrt(np.maximum(E_tri, 0.0) / np.maximum(rho_tri, 1e-30))
    wave_speed_max = float(np.max(wave_speeds)) if wave_speeds.size > 0 else 0.0
    dt_limit = float("inf") if wave_speed_max <= 0.0 else float(dx_min / wave_speed_max)

    print(f"number_of_particles={int(pos.shape[0])}")
    print(f"number_of_triangles={int(tri.shape[0])}")
    print(f"number_of_materials={int(len(materials))}")
    print(f"dx_min={dx_min:.9g}")
    print(f"wave_speed_max={wave_speed_max:.9g}")
    print(f"dt_limit={dt_limit:.9g}")

    if np.isfinite(dt_limit) and float(dt) > dt_limit:
        print(
            "Warning: config.yml time_step exceeds estimated stability limit. "
            f"time_step={float(dt):.9g}, dt_limit={dt_limit:.9g}"
        )


def load_workspace_inputs_v1(workspace_dir: str | Path):
    workspace = Path(workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)
    input_dir = workspace / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (workspace / "output").mkdir(parents=True, exist_ok=True)

    config = _load_config()
    sim_cfg = config.get("simulation", {}) or {}

    dt = float(sim_cfg.get("time_step", 1e-4))
    total_steps = int(sim_cfg.get("total_steps", 1000))

    env_dt = os.environ.get("CPD_TIME_STEP")
    if env_dt:
        parsed = _parse_float(env_dt)
        if parsed is not None:
            dt = parsed

    env_steps = os.environ.get("CPD_TOTAL_STEPS")
    if env_steps:
        parsed = _parse_int(env_steps)
        if parsed is not None:
            total_steps = parsed

    write_steps = max(1, int(sim_cfg.get("write_every_steps", 1)))

    device = str(sim_cfg.get("device", "cpu"))
    g = float(sim_cfg.get("gravity", 0.0))

    force_post_ramp_mode = "hold"

    area_eps = float(sim_cfg.get("area_eps", 1e-14))

    pos, remap_needed, id_to_index, _ = _load_particles(input_dir)
    tri, tri_material, tri_overrides = _load_triangles(input_dir)
    if remap_needed:
        tri = _remap_ids(tri, id_to_index, "connections.csv")

    if tri.min() < 0 or tri.max() >= pos.shape[0]:
        raise ValueError("Triangle connectivity contains out-of-range node ids")

    tri, tri_material, tri_overrides, _ = _filter_near_zero_triangles(
        pos,
        tri,
        tri_material,
        area_eps,
        tri_overrides=tri_overrides,
    )
    print(f"Triangles: {tri.shape[0]}")
    unique_tri_mats = int(np.unique(tri_material).size)
    print(f"Unique triangle materials: {unique_tri_mats}")
    tri_material_ids = sorted(int(m) for m in np.unique(tri_material))

    materials, rigid_material_ids = _load_materials(input_dir)
    print(f"Materials loaded: {len(materials)}")
    print(f"Triangle material ids: {tri_material_ids}")

    E_tri, Nu_tri, rho_tri, fail_SE_tri, c_tri = _build_tri_material_arrays(
        tri_material,
        materials,
        tri_overrides=tri_overrides,
    )
    #check for youngs mod, creating a csv file
    debug_csv = workspace / "output" / "triangle_material_debug.csv"
    with debug_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "triangle_id", "p1", "p2", "p3",
            "material_id", "E", "Nu", "rho", "fail_SE", "c"
        ])
        for i in range(tri.shape[0]):
            writer.writerow([
                i,
                int(tri[i, 0]),
                int(tri[i, 1]),
                int(tri[i, 2]),
                int(tri_material[i]),
                float(E_tri[i]),
                float(Nu_tri[i]),
                float(rho_tri[i]),
                float(fail_SE_tri[i]),
                float(c_tri[i]),
            ])
    print(f"Wrote triangle material debug CSV: {debug_csv}")

    rigid_tri_mask, rigid_bodies = _build_rigid_structures(tri, tri_material, rigid_material_ids)
    _print_startup_diagnostics(pos, tri, materials, E_tri, rho_tri, dt)

    velocity_targets = _load_targets(input_dir / "velocity_targets.csv", id_to_index)
    velocity_profiles = _load_time_profiles(input_dir / "velocity_time.csv", "v")

    force_targets = _load_targets(input_dir / "force_targets.csv", id_to_index)
    force_profiles = _load_time_profiles(input_dir / "force_time.csv", "f")

    fixed_bc = _load_fixed_bc(workspace, id_to_index)

    print(f"Velocity profiles: {len(velocity_profiles)}")
    print(f"Force profiles: {len(force_profiles)}")

    return {
        "pos": pos,
        "tri": tri,
        "material_models": materials,
        "tri_material": tri_material,
        "E_tri": E_tri,
        "Nu_tri": Nu_tri,
        "rho_tri": rho_tri,
        "fail_SE_tri": fail_SE_tri,
        "c_tri": c_tri,
        "velocity_profiles": velocity_profiles,
        "velocity_targets": velocity_targets,
        "force_profiles": force_profiles,
        "force_targets": force_targets,
        "dt": dt,
        "total_steps": total_steps,
        "device": device,
        "g": g,
        "write_every": write_steps,
        "debug": True,
        "force_post_ramp_mode": force_post_ramp_mode,
        "fixed_bc": fixed_bc,
        "rigid_tri_mask": rigid_tri_mask,
        "rigid_bodies": rigid_bodies,
    }
