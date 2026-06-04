import time

import numba
import numpy as np

#imports for printing the csv files
import csv
from pathlib import Path

nthreads = 8
numba.set_num_threads(nthreads)


def _constitutive_dispatch_name(behavior):
    key = str(behavior or "elastic").strip().lower()
    if key in {"elastic", "plastic", "hyperelastic", "viscoelastic", "rigid"}:
        return key
    return "elastic"


def _resolve_material_model_dispatch(material_models, tri_material):
    if not isinstance(material_models, dict) or tri_material is None:
        return {}
    tri_material_arr = np.asarray(tri_material, dtype=np.int64).reshape(-1)
    if tri_material_arr.size == 0:
        return {}
    dispatch = {}
    for mid in np.unique(tri_material_arr):
        model = material_models.get(int(mid), {}) if isinstance(material_models, dict) else {}
        dispatch[int(mid)] = {
            "behavior": _constitutive_dispatch_name(model.get("behavior", "elastic")),
            "symmetry": str(model.get("symmetry", "isotropic") or "isotropic"),
            "damage": str(model.get("damage", "none") or "none"),
            "parameters": dict(model.get("parameters", {}) or {}),
        }
    return dispatch


def _as_tri_param_array(param, n_tri, name):
    arr = np.asarray(param, dtype=float)
    if arr.ndim == 0:
        return np.full((n_tri,), arr.item(), dtype=float)
    arr = arr.reshape(-1)
    if arr.size == 1:
        return np.full((n_tri,), float(arr[0]), dtype=float)
    if arr.size != n_tri:
        raise ValueError(f"{name} must have length {n_tri}, got {arr.size}")
    return arr.astype(float, copy=False)


def _normalize_targets_map(raw_targets, n_nodes):
    targets = {}
    if not isinstance(raw_targets, dict):
        return targets
    for key, value in raw_targets.items():
        try:
            tid = int(key)
        except Exception:
            continue
        arr = np.asarray(value, dtype=int).reshape(-1)
        if arr.size == 0:
            continue
        arr = arr[(arr >= 0) & (arr < n_nodes)]
        if arr.size == 0:
            continue
        targets[tid] = np.unique(arr)
    return targets


def _normalize_profiles_map(raw_profiles, x_key, y_key):
    profiles = {}
    if not isinstance(raw_profiles, dict):
        return profiles

    for key, value in raw_profiles.items():
        try:
            pid = int(key)
        except Exception:
            continue

        t_arr = x_arr = y_arr = None
        if isinstance(value, (tuple, list)) and len(value) >= 3:
            t_arr = np.asarray(value[0], dtype=float).reshape(-1)
            x_arr = np.asarray(value[1], dtype=float).reshape(-1)
            y_arr = np.asarray(value[2], dtype=float).reshape(-1)
        elif isinstance(value, dict):
            t_arr = np.asarray(value.get("time", value.get("t", [])), dtype=float).reshape(-1)
            x_arr = np.asarray(value.get(x_key, value.get("x", [])), dtype=float).reshape(-1)
            y_arr = np.asarray(value.get(y_key, value.get("y", [])), dtype=float).reshape(-1)

        if t_arr is None or x_arr is None or y_arr is None:
            continue
        n = min(t_arr.size, x_arr.size, y_arr.size)
        if n <= 0:
            continue

        t_arr = t_arr[:n]
        x_arr = x_arr[:n]
        y_arr = y_arr[:n]

        order = np.argsort(t_arr, kind="stable")
        t_arr = t_arr[order]
        x_arr = x_arr[order]
        y_arr = y_arr[order]

        profiles[pid] = (
            t_arr.astype(float, copy=False),
            x_arr.astype(float, copy=False),
            y_arr.astype(float, copy=False),
        )

    return profiles


def _interp_xy(t_value, time_arr, x_arr, y_arr):
    x_val = float(np.interp(t_value, time_arr, x_arr, left=x_arr[0], right=x_arr[-1]))
    y_val = float(np.interp(t_value, time_arr, y_arr, left=y_arr[0], right=y_arr[-1]))
    return x_val, y_val


def _eval_velocity_profile(t_value, profile):
    time_arr, vx_arr, vy_arr = profile
    if time_arr.size == 0:
        return 0.0, 0.0
    return _interp_xy(t_value, time_arr, vx_arr, vy_arr)


def _eval_force_profile(t_value, profile, post_mode):
    time_arr, fx_arr, fy_arr = profile
    if time_arr.size == 0:
        return 0.0, 0.0

    t_end = float(time_arr[-1])
    if t_value <= t_end:
        return _interp_xy(t_value, time_arr, fx_arr, fy_arr)

    mode = str(post_mode or "hold").strip().lower()
    if mode == "zero":
        return 0.0, 0.0
    if mode == "repeat" and t_end > 0.0:
        t_mod = t_value % t_end
        return _interp_xy(t_mod, time_arr, fx_arr, fy_arr)

    # default: hold
    return float(fx_arr[-1]), float(fy_arr[-1])


def _build_force_array(t_value, force_targets, force_profiles, n_nodes, post_mode):
    if not force_targets or not force_profiles:
        return np.zeros((n_nodes, 2), dtype=float)

    ext = np.zeros((n_nodes, 2), dtype=float)
    for fid, nodes in force_targets.items():
        profile = force_profiles.get(fid)
        if profile is None:
            continue
        fx, fy = _eval_force_profile(t_value, profile, post_mode)
        if fx == 0.0 and fy == 0.0:
            continue
        ext[nodes, 0] += fx
        ext[nodes, 1] += fy
    return ext


def _build_velocity_bc(t_value, velocity_targets, velocity_profiles):
    if not velocity_targets or not velocity_profiles:
        return np.empty((0, 3), dtype=float)

    vel_map = {}
    for vid, nodes in velocity_targets.items():
        profile = velocity_profiles.get(vid)
        if profile is None:
            continue
        vx, vy = _eval_velocity_profile(t_value, profile)
        for nid in nodes:
            if nid not in vel_map:
                vel_map[nid] = [0.0, 0.0]
            vel_map[nid][0] += vx
            vel_map[nid][1] += vy

    if not vel_map:
        return np.empty((0, 3), dtype=float)

    ids = np.array(sorted(vel_map.keys()), dtype=np.int32)
    out = np.empty((ids.size, 3), dtype=float)
    out[:, 0] = ids
    for i, nid in enumerate(ids):
        out[i, 1] = vel_map[int(nid)][0]
        out[i, 2] = vel_map[int(nid)][1]
    return out


def _calc_masses_cpu(n_nodes, tri, areas, rho_tri):
    contrib = (areas * rho_tri) / 3.0
    masses = np.bincount(tri.ravel(), weights=np.repeat(contrib, 3), minlength=n_nodes).astype(float)
    masses = masses.reshape(-1, 1)
    masses = np.maximum(masses, 1e-12)
    return masses


def _calc_damping_cpu(n_nodes, tri, areas, c_tri):
    contrib = areas / 3.0
    num = np.bincount(
        tri.ravel(),
        weights=np.repeat(contrib * c_tri, 3),
        minlength=n_nodes,
    ).astype(float)
    den = np.bincount(
        tri.ravel(),
        weights=np.repeat(contrib, 3),
        minlength=n_nodes,
    ).astype(float)
    c_nodes = np.zeros((n_nodes, 1), dtype=float)
    mask = den > 0.0
    c_nodes[mask, 0] = num[mask] / den[mask]
    return c_nodes


def _normalize_rigid_tri_mask(mask, n_tri):
    if mask is None:
        return np.zeros((n_tri,), dtype=bool)
    arr = np.asarray(mask, dtype=bool).reshape(-1)
    if arr.size == 0:
        return np.zeros((n_tri,), dtype=bool)
    if arr.size == 1:
        return np.full((n_tri,), bool(arr[0]), dtype=bool)
    if arr.size != n_tri:
        raise ValueError(f"rigid_tri_mask must have length {n_tri}, got {arr.size}")
    return arr


def _normalize_rigid_bodies(rigid_bodies, n_nodes):
    bodies = []
    if isinstance(rigid_bodies, dict):
        body_groups = rigid_bodies.values()
    elif isinstance(rigid_bodies, (list, tuple)):
        body_groups = rigid_bodies
    else:
        return bodies

    for value in body_groups:
        node_lists = value if isinstance(value, (list, tuple)) else (value,)
        for nodes in node_lists:
            arr = np.asarray(nodes, dtype=np.int64).reshape(-1)
            if arr.size == 0:
                continue
            arr = arr[(arr >= 0) & (arr < n_nodes)]
            arr = np.unique(arr)
            if arr.size > 0:
                bodies.append(arr.astype(np.int32, copy=False))
    return bodies


def _apply_rigid_bodies_cpu(pos, vel, masses, rigid_body_nodes, dt):
    for nodes in rigid_body_nodes:
        m = masses[nodes, 0]
        m_sum = float(np.sum(m))
        if m_sum <= 1e-20:
            continue

        x = pos[nodes]
        v = vel[nodes]
        x_cm = np.sum(m[:, None] * x, axis=0) / m_sum
        V = np.sum(m[:, None] * v, axis=0) / m_sum

        r = x - x_cm
        L = float(np.sum(r[:, 0] * (m * v[:, 1]) - r[:, 1] * (m * v[:, 0])))
        I = float(np.sum(m * (r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1])))
        omega = 0.0 if I <= 1e-20 else L / I

        theta = omega * dt
        ct = np.cos(theta)
        st = np.sin(theta)
        r_new = np.empty_like(r)
        r_new[:, 0] = ct * r[:, 0] - st * r[:, 1]
        r_new[:, 1] = st * r[:, 0] + ct * r[:, 1]

        pos[nodes] = x_cm + r_new
        vel[nodes, 0] = V[0] - omega * r_new[:, 1]
        vel[nodes, 1] = V[1] + omega * r_new[:, 0]


def _apply_rigid_bodies_gpu(cp, pos, vel, masses, rigid_body_nodes_gpu, dt):
    for nodes in rigid_body_nodes_gpu:
        m = masses[nodes, 0]
        m_sum = float(cp.sum(m).item())
        if m_sum <= 1e-20:
            continue

        x = pos[nodes]
        v = vel[nodes]
        x_cm = cp.sum(m[:, None] * x, axis=0) / m_sum
        V = cp.sum(m[:, None] * v, axis=0) / m_sum

        r = x - x_cm
        L = float(cp.sum(r[:, 0] * (m * v[:, 1]) - r[:, 1] * (m * v[:, 0])).item())
        I = float(cp.sum(m * (r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1])).item())
        omega = 0.0 if I <= 1e-20 else L / I

        theta = omega * dt
        ct = float(np.cos(theta))
        st = float(np.sin(theta))
        r_new = cp.empty_like(r)
        r_new[:, 0] = ct * r[:, 0] - st * r[:, 1]
        r_new[:, 1] = st * r[:, 0] + ct * r[:, 1]

        pos[nodes] = x_cm + r_new
        vel[nodes, 0] = V[0] - omega * r_new[:, 1]
        vel[nodes, 1] = V[1] + omega * r_new[:, 0]


def _precompute_triangle_kinematics(ref_pos, tri):
    tri = np.asarray(tri, dtype=np.int64)
    ref = np.asarray(ref_pos, dtype=float)
    p1 = ref[tri[:, 0]]
    p2 = ref[tri[:, 1]]
    p3 = ref[tri[:, 2]]

    two_a = (
        p1[:, 0] * (p2[:, 1] - p3[:, 1])
        + p2[:, 0] * (p3[:, 1] - p1[:, 1])
        + p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    safe_two_a = np.where(np.abs(two_a) < 1e-20, np.nan, two_a)

    b1 = p2[:, 1] - p3[:, 1]
    b2 = p3[:, 1] - p1[:, 1]
    b3 = p1[:, 1] - p2[:, 1]
    c1 = p3[:, 0] - p2[:, 0]
    c2 = p1[:, 0] - p3[:, 0]
    c3 = p2[:, 0] - p1[:, 0]

    return {
        "inv_two_a": 1.0 / safe_two_a,
        "b1": b1,
        "b2": b2,
        "b3": b3,
        "c1": c1,
        "c2": c2,
        "c3": c3,
    }


def _compute_postprocess_fields(pos, ref_pos, tri, tri_kin, E_arr, Nu_arr):
    pos = np.asarray(pos, dtype=float)
    ref_pos = np.asarray(ref_pos, dtype=float)
    tri = np.asarray(tri, dtype=np.int64)

    displacement = pos - ref_pos
    u1 = displacement[tri[:, 0]]
    u2 = displacement[tri[:, 1]]
    u3 = displacement[tri[:, 2]]

    inv_two_a = tri_kin["inv_two_a"]
    b1 = tri_kin["b1"]
    b2 = tri_kin["b2"]
    b3 = tri_kin["b3"]
    c1 = tri_kin["c1"]
    c2 = tri_kin["c2"]
    c3 = tri_kin["c3"]

    exx = inv_two_a * (b1 * u1[:, 0] + b2 * u2[:, 0] + b3 * u3[:, 0])
    eyy = inv_two_a * (c1 * u1[:, 1] + c2 * u2[:, 1] + c3 * u3[:, 1])
    gxy = inv_two_a * (
        c1 * u1[:, 0] + b1 * u1[:, 1]
        + c2 * u2[:, 0] + b2 * u2[:, 1]
        + c3 * u3[:, 0] + b3 * u3[:, 1]
    )

    strain = np.stack((exx, eyy, gxy), axis=1)

    denom = np.maximum(1.0 - Nu_arr * Nu_arr, 1e-20)
    scale = E_arr / denom
    sxx = scale * (exx + Nu_arr * eyy)
    syy = scale * (Nu_arr * exx + eyy)
    sxy = scale * (0.5 * (1.0 - Nu_arr) * gxy)
    vm = np.sqrt(np.maximum(sxx * sxx + syy * syy - sxx * syy + 3.0 * sxy * sxy, 0.0))

    stress = np.stack((sxx, syy, sxy, vm), axis=1)
    return displacement, strain, stress


def run_simulation(
    pos,
    tri,
    E_tri,
    Nu_tri,
    rho_tri,
    fail_SE_tri,
    c_tri,
    velocity_profiles,
    velocity_targets,
    force_profiles,
    force_targets,
    dt,
    total_steps,
    *,
    material_models=None,
    tri_material=None,
    device="cpu",
    g=0.0,
    write_every=None,
    debug=False,
    force_post_ramp_mode="hold",
    fixed_bc=None,
    rigid_tri_mask=None,
    rigid_bodies=None,
    return_fields=False,
):
    dt = float(dt)
    N = int(total_steps)
    if N <= 0:
        raise ValueError("total_steps must be > 0")

    pos_in = np.asarray(pos, dtype=float)
    if pos_in.ndim != 2 or pos_in.shape[1] < 2:
        raise ValueError("pos must be a 2D array with at least 2 columns")
    pos_in = pos_in[:, :2]
    n_nodes = pos_in.shape[0]

    tri_in = None
    if tri is not None:
        tri_in = np.asarray(tri, dtype=np.int64)
        if tri_in.ndim != 2 or tri_in.shape[1] != 3:
            raise ValueError("tri must have shape (n_tri, 3)")
        if tri_in.size > 0 and (tri_in.min() < 0 or tri_in.max() >= n_nodes):
            raise ValueError("tri contains invalid node indices")

    material_dispatch = _resolve_material_model_dispatch(material_models, tri_material)
    if material_dispatch and debug:
        active_behaviors = sorted({entry.get("behavior", "elastic") for entry in material_dispatch.values()})
        print(f"Material behaviors active: {', '.join(active_behaviors)}")

    velocity_profiles = _normalize_profiles_map(velocity_profiles, "vx", "vy")
    force_profiles = _normalize_profiles_map(force_profiles, "fx", "fy")
    velocity_targets = _normalize_targets_map(velocity_targets, n_nodes)
    force_targets = _normalize_targets_map(force_targets, n_nodes)

    fixed_in = np.asarray(fixed_bc) if fixed_bc is not None else np.empty((0, 1), dtype=np.int32)
    if fixed_in.size == 0:
        fixed_in = np.empty((0, 1), dtype=np.int32)
    else:
        fixed_in = fixed_in.astype(np.int32).reshape(-1, 1)

    if write_every is None:
        write_every = max(1, N // 50)
    write_every = max(1, int(write_every))

    n_snaps = N // write_every + 1
    pos_history = np.zeros((n_snaps, n_nodes, 2), dtype=float)

    mode = str(force_post_ramp_mode or "hold").strip().lower()
    if mode not in ("hold", "repeat", "zero"):
        mode = "hold"

    device = str(device).strip().lower()

    if device == "cpu":
        from cpu_utils import (
            delaunay_triangulate,
            calc_ele_areas_cpu,
            zero_tls,
            compute_tls,
            reduce_tls,
            update_cpu,
        )

        pos_cpu = np.asarray(pos_in.copy(), dtype=float)
        if tri_in is None:
            tri_cpu = delaunay_triangulate(pos_cpu)
        else:
            tri_cpu = tri_in

        n_tri = tri_cpu.shape[0]
        E_arr = _as_tri_param_array(E_tri, n_tri, "E_tri")
        Nu_arr = _as_tri_param_array(Nu_tri, n_tri, "Nu_tri")
        rho_arr = _as_tri_param_array(rho_tri, n_tri, "rho_tri")
        fail_arr = _as_tri_param_array(fail_SE_tri, n_tri, "fail_SE_tri")
        c_arr = _as_tri_param_array(c_tri, n_tri, "c_tri")

        rigid_mask_cpu = _normalize_rigid_tri_mask(rigid_tri_mask, n_tri)
        rigid_body_nodes_cpu = _normalize_rigid_bodies(rigid_bodies, n_nodes)
        displacement_history = np.zeros((n_snaps, n_nodes, 2), dtype=float)
        strain_history = np.zeros((n_snaps, n_tri, 3), dtype=float)
        stress_history = np.zeros((n_snaps, n_tri, 4), dtype=float)
        strain_frame = np.full((n_tri, 3), np.nan, dtype=float)
        stress_frame = np.full((n_tri, 4), np.nan, dtype=float)


        vel_cpu = np.zeros((n_nodes, 2), dtype=float)
        ref_pos = np.asarray(pos_cpu.copy(), dtype=float)
        tri_kin = _precompute_triangle_kinematics(ref_pos, tri_cpu)
        active_tri = np.ones((n_tri,), dtype=np.bool_)
        if rigid_mask_cpu.size > 0:
            active_tri[rigid_mask_cpu] = False
        areas = calc_ele_areas_cpu(pos_cpu, tri_cpu)
        masses = _calc_masses_cpu(n_nodes, tri_cpu, areas, rho_arr)
        c_nodes = _calc_damping_cpu(n_nodes, tri_cpu, areas, c_arr)
        print(f"Mass range: min={float(np.min(masses)):.6e}, max={float(np.max(masses)):.6e}")

        tls = np.empty((nthreads, n_nodes, 2), dtype=float)
        barrier_tls = np.zeros((nthreads,), dtype=np.int64)
        forces = np.zeros((n_nodes, 2), dtype=float)
        snap_idx = 0
        start = time.perf_counter()

        for i in range(N):
            t_value = i * dt
            zero_tls(tls)
            barrier_tls[:] = 0
            strain_frame.fill(np.nan)
            stress_frame.fill(np.nan)
            compute_tls(
                ref_pos,
                pos_cpu,
                tri_cpu,
                active_tri,
                areas,
                tls,
                barrier_tls,
                E_arr,
                Nu_arr,
                fail_arr,
                strain_frame,
                stress_frame,
            )
            reduce_tls(tls, forces)

            if force_targets and force_profiles:
                forces += _build_force_array(t_value, force_targets, force_profiles, n_nodes, mode)

            vel_bc = _build_velocity_bc(t_value, velocity_targets, velocity_profiles)
            pos_cpu, vel_cpu = update_cpu(pos_cpu, vel_cpu, masses, forces, fixed_in, vel_bc, dt, g, c_nodes)
            if rigid_body_nodes_cpu:
                _apply_rigid_bodies_cpu(pos_cpu, vel_cpu, masses, rigid_body_nodes_cpu, dt)

            barrier_count = int(np.sum(barrier_tls))
            if barrier_count > 0:
                print(f"Barrier activations: {barrier_count}")

            if i % write_every == 0:
                if debug:
                    end = time.perf_counter()
                    print("Time steps completed:", i)
                    print("Time elapsed (s):", end - start)
                pos_history[snap_idx] = pos_cpu
                strain_history[snap_idx] = strain_frame
                stress_history[snap_idx] = stress_frame
                snap_idx += 1

        displacement_history[:] = pos_history - ref_pos

    elif device == "gpu":
        import cupy as cp
        from gpu_utils import (
            delaunay_triangulate,
            calc_ele_areas_gpu,
            calc_force_s_vect_gpu,
            update_gpu,
        )

        pos_gpu = cp.asarray(pos_in)
        if tri_in is None:
            tri_host = delaunay_triangulate(pos_in)
        else:
            tri_host = tri_in

        n_tri = tri_host.shape[0]
        E_arr = _as_tri_param_array(E_tri, n_tri, "E_tri")
        Nu_arr = _as_tri_param_array(Nu_tri, n_tri, "Nu_tri")
        rho_arr = _as_tri_param_array(rho_tri, n_tri, "rho_tri")
        fail_arr = _as_tri_param_array(fail_SE_tri, n_tri, "fail_SE_tri")
        c_arr = _as_tri_param_array(c_tri, n_tri, "c_tri")
        rigid_mask_host = _normalize_rigid_tri_mask(rigid_tri_mask, n_tri)
        rigid_body_nodes = _normalize_rigid_bodies(rigid_bodies, n_nodes)
        rigid_body_nodes_gpu = [cp.asarray(nodes, dtype=cp.int32) for nodes in rigid_body_nodes]
        displacement_history = np.zeros((n_snaps, n_nodes, 2), dtype=float)
        strain_history = np.zeros((n_snaps, n_tri, 3), dtype=float)
        stress_history = np.zeros((n_snaps, n_tri, 4), dtype=float)

        tri_gpu = cp.asarray(tri_host)
        active_tri = cp.ones((n_tri,), dtype=bool)
        if rigid_mask_host.size > 0:
            active_tri[cp.asarray(rigid_mask_host)] = False
        vel_gpu = cp.zeros((n_nodes, 2), dtype=cp.float64)
        ref_pos = cp.asarray(pos_gpu.copy())
        ref_pos_host = np.asarray(pos_in.copy(), dtype=float)
        areas = calc_ele_areas_gpu(pos_gpu, tri_gpu)

        rho_gpu = cp.asarray(rho_arr)
        c_gpu_tri = cp.asarray(c_arr)
        contrib_mass = (areas * rho_gpu) / 3.0
        masses = cp.bincount(
            tri_gpu.ravel(),
            weights=cp.repeat(contrib_mass, 3),
            minlength=n_nodes,
        ).reshape(-1, 1)
        masses = cp.maximum(masses, 1e-12)
        print(
            "Mass range: min={:.6e}, max={:.6e}".format(
                float(cp.min(masses).item()),
                float(cp.max(masses).item()),
            )
        )

        contrib = areas / 3.0
        c_num = cp.bincount(
            tri_gpu.ravel(),
            weights=cp.repeat(contrib * c_gpu_tri, 3),
            minlength=n_nodes,
        )
        c_den = cp.bincount(
            tri_gpu.ravel(),
            weights=cp.repeat(contrib, 3),
            minlength=n_nodes,
        )
        c_nodes = cp.zeros((n_nodes, 1), dtype=cp.float64)
        mask = c_den > 0.0
        c_nodes[mask, 0] = c_num[mask] / c_den[mask]

        fixed_gpu = cp.asarray(fixed_in, dtype=cp.int32)
        force_targets_gpu = {fid: cp.asarray(nodes, dtype=cp.int32) for fid, nodes in force_targets.items()}

        E_gpu = cp.asarray(E_arr)
        Nu_gpu = cp.asarray(Nu_arr)
        fail_gpu = cp.asarray(fail_arr)

        snap_idx = 0
        start = time.perf_counter()
        strain_gpu = None
        stress_gpu = None
        for i in range(N):
            t_value = i * dt
            forces, barrier_count, strain_gpu, stress_gpu = calc_force_s_vect_gpu(
                ref_pos,
                pos_gpu,
                tri_gpu,
                active_tri,
                areas,
                E_gpu,
                Nu_gpu,
                fail_gpu,
                return_strain_stress=True,
            )

            if force_targets_gpu and force_profiles:
                ext = cp.zeros_like(forces)
                for fid, nodes in force_targets_gpu.items():
                    profile = force_profiles.get(fid)
                    if profile is None:
                        continue
                    fx, fy = _eval_force_profile(t_value, profile, mode)
                    if fx == 0.0 and fy == 0.0:
                        continue
                    ext[nodes, 0] += fx
                    ext[nodes, 1] += fy
                forces += ext

            vel_bc_np = _build_velocity_bc(t_value, velocity_targets, velocity_profiles)
            vel_bc_gpu = cp.asarray(vel_bc_np) if vel_bc_np.size > 0 else cp.empty((0, 3))

            pos_gpu, vel_gpu = update_gpu(
                pos_gpu,
                vel_gpu,
                masses,
                forces,
                fixed_gpu,
                vel_bc_gpu,
                dt,
                g,
                c_nodes,
            )
            if rigid_body_nodes_gpu:
                _apply_rigid_bodies_gpu(cp, pos_gpu, vel_gpu, masses, rigid_body_nodes_gpu, dt)

            if barrier_count > 0:
                print(f"Barrier activations: {barrier_count}")

            if i % write_every == 0:
                if debug:
                    end = time.perf_counter()
                    print("Time steps completed:", i)
                    print("Time elapsed (s):", end - start)
                pos_frame = pos_gpu.get()
                pos_history[snap_idx] = pos_frame
                if strain_gpu is not None:
                    strain_history[snap_idx] = strain_gpu.get()
                if stress_gpu is not None:
                    stress_history[snap_idx] = stress_gpu.get()
                snap_idx += 1

        displacement_history[:] = pos_history - ref_pos_host

    else:
        raise ValueError(f"Unsupported device: {device}")

    if return_fields:
        return {
            "pos_history": pos_history,
            "displacement_history": displacement_history,
            "strain_history": strain_history,
            "stress_history": stress_history,
        }

    return pos_history
