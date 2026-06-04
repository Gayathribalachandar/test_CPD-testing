"""Solver export adapter and the sole owner of solver CSV I/O.

Architecture rule:
- This is the only module allowed to write solver-facing CSV files.
- UI modules (panels/main_window/sketch_view) must call into this adapter.
"""

from __future__ import annotations

import ast
import copy
import csv
import logging
import math
import os
import shutil
from typing import Any

import numpy as np
import yaml
from PySide6.QtWidgets import QMessageBox

from app_config import DEFAULT_DX, get_workspace_dir, get_workspace_path
from mesh_utils import map_geometry_to_nodes
from models import (
    Interface,
    Material,
    normalize_heterogeneity_config,
    normalize_material_field_config,
)
from material_registry import (
    all_registry_parameter_keys,
    infer_behavior_from_mat_type,
    normalize_material_behavior,
    normalize_material_damage,
    normalize_material_properties,
    normalize_material_symmetry,
)


def _normalize_material_store(data: Any) -> dict[int, Any]:
    if not isinstance(data, dict):
        return {}
    out: dict[int, Any] = {}
    for key, value in data.items():
        try:
            out[int(key)] = value
        except Exception:
            continue
    return out


def _resolve_export_materials(project_state: Any, view: Any) -> dict[int, Any]:
    state_materials = _normalize_material_store(getattr(project_state, "materials", None))
    if state_materials:
        return state_materials
    return _normalize_material_store(getattr(view, "materials", None))


def _solver_material_row(mat) -> dict[str, float | int | str]:
    behavior = normalize_material_behavior(
        getattr(mat, "behavior", infer_behavior_from_mat_type(getattr(mat, "mat_type", "")))
    )
    symmetry = normalize_material_symmetry(getattr(mat, "symmetry", "isotropic"))
    damage = normalize_material_damage(getattr(mat, "damage", "none"))
    props = normalize_material_properties(getattr(mat, "properties", {}) or {}, behavior, symmetry, damage)
    mat_type = str(getattr(mat, "mat_type", "") or "").upper()
    material_id = int(getattr(mat, "serial"))
    name = str(getattr(mat, "name", f"material_{material_id}"))

    rho = float(props.get("density", 1.0))
    c = float(props.get("hardening_rate", props.get("damping", props.get("cohesive_strength", 0.0))))
    fail_se = float(props.get("failure_energy", props.get("yield_stress", 1e9)))

    def _safe_positive_values(*values):
        out = []
        for value in values:
            try:
                value = float(value)
            except Exception:
                continue
            if math.isfinite(value) and value > 0.0:
                out.append(value)
        return out

    def _geometric_mean(values, default):
        clean = _safe_positive_values(*values)
        if not clean:
            return float(default)
        if len(clean) == 1:
            return clean[0]
        log_sum = 0.0
        for value in clean:
            log_sum += math.log(value)
        return float(math.exp(log_sum / len(clean)))

    def _average(values, default):
        clean = []
        for value in values:
            try:
                value = float(value)
            except Exception:
                continue
            if math.isfinite(value):
                clean.append(value)
        if not clean:
            return float(default)
        return float(sum(clean) / len(clean))

    if behavior == "hyperelastic" or mat_type == "NEOHOOK":
        shear = float(props.get("shear_modulus", 0.0))
        bulk = float(props.get("bulk_modulus", 0.0))
        denom = 3.0 * bulk + shear
        if abs(denom) < 1e-20:
            e_val = float(props.get("youngs_modulus", 1.0))
            nu_val = float(props.get("poisson_ratio", 0.3))
        else:
            e_val = 9.0 * bulk * shear / denom
            nu_val = (3.0 * bulk - 2.0 * shear) / (2.0 * denom)
    elif behavior == "rigid" or mat_type == "RIGID":
        e_val = float(props.get("youngs_modulus", 1e15))
        nu_val = float(props.get("poisson_ratio", 0.499))
        fail_se = max(fail_se, 1e12)
    else:
        if symmetry == "orthotropic":
            e_val = _geometric_mean(
                [
                    props.get("youngs_modulus_x"),
                    props.get("youngs_modulus_y"),
                    props.get("youngs_modulus"),
                ],
                props.get("youngs_modulus", 1.0),
            )
            nu_val = _average(
                [
                    props.get("poisson_ratio_xy"),
                    props.get("poisson_ratio"),
                ],
                props.get("poisson_ratio", 0.3),
            )
        elif symmetry == "anisotropic":
            e_val = _geometric_mean(
                [
                    props.get("youngs_modulus_x"),
                    props.get("youngs_modulus_y"),
                    props.get("youngs_modulus_z"),
                    props.get("youngs_modulus"),
                ],
                props.get("youngs_modulus", 1.0),
            )
            nu_val = _average(
                [
                    props.get("poisson_ratio_xy"),
                    props.get("poisson_ratio_yz"),
                    props.get("poisson_ratio_xz"),
                    props.get("poisson_ratio"),
                ],
                props.get("poisson_ratio", 0.3),
            )
        else:
            e_val = float(props.get("youngs_modulus", 1.0))
            nu_val = float(props.get("poisson_ratio", 0.3))
    nu_val = max(-0.499, min(0.499, float(nu_val)))

    return {
        "material_id": material_id,
        "name": name,
        "behavior": behavior,
        "symmetry": symmetry,
        "damage": damage,
        "E": e_val,
        "nu": nu_val,
        "rho": rho,
        "fail_SE": fail_se,
        "c": c,
        "parameters": copy.deepcopy(props),
    }


def _export_part_store(project_state: Any, view: Any) -> dict[int, Any]:
    parts = list(getattr(project_state, "parts", None) or getattr(view, "parts", None) or [])
    out: dict[int, Any] = {}
    for part in parts:
        try:
            out[int(getattr(part, "id"))] = part
        except Exception:
            continue
    return out


def _append_row_meta(row: dict[str, Any], item: str) -> None:
    text = str(item or "").strip()
    if not text:
        return
    existing = str(row.get("meta", "") or "").strip()
    row["meta"] = f"{existing};{text}" if existing else text


def _safe_eval_material_expression(expr: str, variables: dict[str, float]) -> float:
    expr = str(expr or "").strip()
    if not expr:
        raise ValueError("empty expression")

    allowed_funcs = {
        "abs": abs,
        "min": min,
        "max": max,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "atan2": math.atan2,
        "exp": math.exp,
        "log": math.log,
        "log10": math.log10,
        "pow": pow,
    }
    allowed_names = set(allowed_funcs.keys()) | set(variables.keys())
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.UAdd,
        ast.Call,
        ast.Load,
        ast.Name,
        ast.Constant,
    )

    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"unsupported expression node: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(f"unsupported name: {node.id}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in allowed_funcs:
                raise ValueError("unsupported function call")

    scope = {key: float(value) for key, value in variables.items()}
    scope.update(allowed_funcs)
    value = eval(compile(tree, "<material_expr>", "eval"), {"__builtins__": {}}, scope)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("expression result is not finite")
    return value


def _clamp_solver_scalar(key: str, value: float) -> float:
    if key == "nu":
        return max(-0.499, min(0.499, float(value)))
    if key in {"E", "rho", "fail_SE"}:
        return max(1e-12, float(value))
    return float(value)


def _value_noise_2d(x: float, y: float, scale: float, seed: int) -> float:
    scale = max(float(scale), 1e-12)
    u = float(x) / scale
    v = float(y) / scale
    i0 = math.floor(u)
    j0 = math.floor(v)
    tx = u - i0
    ty = v - j0

    def _hash(ii: int, jj: int) -> float:
        key = (ii * 73856093) ^ (jj * 19349663) ^ int(seed)
        rng = np.random.default_rng(key & 0xFFFFFFFF)
        return float(rng.uniform(-1.0, 1.0))

    def _smooth(t: float) -> float:
        return t * t * (3.0 - 2.0 * t)

    sx = _smooth(tx)
    sy = _smooth(ty)
    v00 = _hash(i0, j0)
    v10 = _hash(i0 + 1, j0)
    v01 = _hash(i0, j0 + 1)
    v11 = _hash(i0 + 1, j0 + 1)
    vx0 = v00 * (1.0 - sx) + v10 * sx
    vx1 = v01 * (1.0 - sx) + v11 * sx
    return float(vx0 * (1.0 - sy) + vx1 * sy)


def _base_geometry_variables(part: Any) -> dict[str, float]:
    geom = getattr(part, "geometry", None)
    if geom is not None and not getattr(geom, "is_empty", True):
        try:
            xmin, ymin, xmax, ymax = [float(v) for v in geom.bounds]
            centroid = getattr(geom, "centroid", None)
            xc = float(getattr(centroid, "x", 0.0))
            yc = float(getattr(centroid, "y", 0.0))
        except Exception:
            xmin = ymin = xmax = ymax = 0.0
            xc = yc = 0.0
    else:
        xmin = ymin = xmax = ymax = 0.0
        xc = yc = 0.0
    width = max(xmax - xmin, 1e-12)
    height = max(ymax - ymin, 1e-12)
    return {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "width": width,
        "height": height,
        "L": max(width, height, 1e-12),
        "xc": xc,
        "yc": yc,
    }


def _apply_triangle_property_override(row: dict[str, Any], scalars: dict[str, float], field_tag: str) -> None:
    for key in ("E", "nu", "rho", "fail_SE", "c"):
        if key in scalars:
            row[key] = float(_clamp_solver_scalar(key, scalars[key]))
    _append_row_meta(row, field_tag)


def _evaluate_material_field_for_row(
    row: dict[str, Any],
    part: Any,
    base_scalars: dict[str, float],
    field_cfg: dict[str, Any],
) -> dict[str, float]:
    centroid = row.get("_centroid")
    if centroid is None:
        return dict(base_scalars)
    x = float(centroid[0])
    y = float(centroid[1])
    geom_vars = _base_geometry_variables(part)
    prop_key = str(field_cfg.get("property_key", "E") or "E")
    if prop_key not in {"E", "rho", "nu"}:
        prop_key = "E"
    field_type = str(field_cfg.get("field_type", "linear_gradient") or "linear_gradient")
    scalars = dict(base_scalars)
    base_value = float(base_scalars.get(prop_key, 0.0))

    if field_type == "linear_gradient":
        linear = dict(field_cfg.get("linear_gradient", {}) or {})
        direction = str(linear.get("direction", "x") or "x").lower()
        if direction == "y":
            ratio = (y - geom_vars["ymin"]) / max(geom_vars["height"], 1e-12)
        elif direction == "diag":
            rx = (x - geom_vars["xmin"]) / max(geom_vars["width"], 1e-12)
            ry = (y - geom_vars["ymin"]) / max(geom_vars["height"], 1e-12)
            ratio = 0.5 * (rx + ry)
        else:
            ratio = (x - geom_vars["xmin"]) / max(geom_vars["width"], 1e-12)
        ratio = max(0.0, min(1.0, float(ratio)))
        min_value = float(linear.get("min", base_value))
        max_value = float(linear.get("max", base_value))
        scalars[prop_key] = min_value + (max_value - min_value) * ratio
        return scalars

    if field_type == "radial_gradient":
        radial = dict(field_cfg.get("radial_gradient", {}) or {})
        cx = float(radial.get("center_x", geom_vars["xc"]))
        cy = float(radial.get("center_y", geom_vars["yc"]))
        radius = max(float(radial.get("radius", geom_vars["L"])), 1e-12)
        core = float(radial.get("core", base_value))
        shell = float(radial.get("shell", base_value))
        ratio = max(0.0, min(1.0, math.hypot(x - cx, y - cy) / radius))
        scalars[prop_key] = core + (shell - core) * ratio
        return scalars

    if field_type == "random_field":
        rnd = dict(field_cfg.get("random_field", {}) or {})
        mean = float(rnd.get("mean", base_value))
        std = float(rnd.get("std", 0.0))
        correlation_length = max(float(rnd.get("correlation_length", geom_vars["L"])), 1e-12)
        seed = rnd.get("seed")
        try:
            seed_value = int(seed) if seed not in (None, "") else int(getattr(part, "id", 0)) * 7919
        except Exception:
            seed_value = int(getattr(part, "id", 0)) * 7919
        scalars[prop_key] = mean + std * _value_noise_2d(x, y, correlation_length, seed_value)
        return scalars

    equation = dict(field_cfg.get("user_equation", {}) or {})
    expression = str(equation.get("expression", "") or "").strip()
    variables = {
        "x": x,
        "y": y,
        **geom_vars,
        "r": math.hypot(x - geom_vars["xc"], y - geom_vars["yc"]),
        "theta": math.atan2(y - geom_vars["yc"], x - geom_vars["xc"]),
        "base_value": base_value,
        "pi": math.pi,
    }
    scalars[prop_key] = _safe_eval_material_expression(expression, variables) if expression else base_value
    return scalars


def _apply_heterogeneous_material_assignments(
    connections_rows: list[dict[str, Any]],
    export_materials: dict[int, Any],
    project_state: Any,
    view: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows = list(connections_rows or [])
    if not rows:
        return rows, []

    part_store = _export_part_store(project_state, view)
    warnings: list[str] = []

    for part_id, part in part_store.items():
        assignment_mode = str(getattr(part, "material_assignment_mode", "homogeneous")).lower()
        if assignment_mode not in {"heterogeneous", "material_field"}:
            continue
        method = str(getattr(part, "heterogeneity_method", "region_based") or "region_based").lower()
        if assignment_mode == "material_field":
            method = "material_field"
        if method == "region_based":
            continue
        base_material_id = getattr(part, "material_id", None)
        try:
            base_material_id = int(base_material_id)
        except Exception:
            warnings.append(f"Part {part_id}: heterogeneous assignment skipped because no base material is assigned.")
            continue
        base_material = export_materials.get(base_material_id)
        if base_material is None:
            warnings.append(f"Part {part_id}: heterogeneous assignment references missing base material id {base_material_id}.")
            continue
        config = normalize_heterogeneity_config(copy.deepcopy(getattr(part, "heterogeneity_config", {})))
        part_rows = []
        for row in rows:
            if str(row.get("zone_kind", "part")).lower() == "interface":
                continue
            row_part_id = row.get("part_id")
            if row_part_id in ("", None):
                continue
            try:
                if int(row_part_id) != int(part_id):
                    continue
            except Exception:
                continue
            part_rows.append(row)
        if not part_rows:
            continue

        if method == "random_distribution":
            distribution = []
            for item in config.get("materials", []):
                try:
                    mat_id = int(item.get("material_id"))
                    frac = float(item.get("fraction", 0.0))
                except Exception:
                    continue
                if mat_id not in export_materials or frac <= 0.0:
                    continue
                distribution.append((mat_id, frac))
            if not distribution:
                distribution = [(base_material_id, 1.0)]
                warnings.append(f"Part {part_id}: random distribution had no valid material fractions; using base material only.")
            total = sum(frac for _mid, frac in distribution)
            probs = np.array([frac / total for _mid, frac in distribution], dtype=float)
            mat_ids = np.array([mid for mid, _frac in distribution], dtype=int)
            seed = config.get("random_seed")
            try:
                seed_value = int(seed) if seed not in (None, "") else int(part_id) * 9973
            except Exception:
                seed_value = int(part_id) * 9973
            rng = np.random.default_rng(seed_value)
            ordered_rows = sorted(part_rows, key=lambda row: int(row.get("triangle_id", 0)))
            chosen = rng.choice(mat_ids, size=len(ordered_rows), p=probs)
            for row, material_id in zip(ordered_rows, chosen.tolist()):
                row["material_id"] = int(material_id)
                _append_row_meta(row, "heterogeneity=random_distribution")
            continue

        if method in {"field_gradient_distribution", "material_field"}:
            base_scalars = _solver_material_row(base_material)
            if method == "field_gradient_distribution":
                expressions = dict(config.get("expressions", {}) or {})
                active_keys = [key for key in ("E", "nu", "rho", "fail_SE", "c") if str(expressions.get(key, "") or "").strip()]
                if not active_keys:
                    warnings.append(f"Part {part_id}: field distribution has no active expressions; using base material.")
                    continue
                field_cfg = {
                    "property_key": active_keys[0],
                    "field_type": "user_equation",
                    "linear_gradient": {},
                    "radial_gradient": {},
                    "random_field": {},
                    "user_equation": {"expression": str(expressions.get(active_keys[0], "") or "")},
                }
            else:
                field_cfg = normalize_material_field_config(
                    copy.deepcopy(getattr(part, "material_field_config", {}))
                )
            for row in part_rows:
                try:
                    scalars = _evaluate_material_field_for_row(row, part, base_scalars, field_cfg)
                except Exception as exc:
                    warnings.append(f"Part {part_id}: material field evaluation failed: {exc}")
                    break
                try:
                    _apply_triangle_property_override(
                        row,
                        scalars,
                        f"heterogeneity={method}",
                    )
                except Exception:
                    continue
            continue

        warnings.append(f"Part {part_id}: unknown heterogeneous method '{method}', using base material.")

    return rows, warnings


def _resolve_export_view(project_state: Any):
    if project_state is None:
        return None, {}
    settings = getattr(project_state, "solver_settings", None)
    if not isinstance(settings, dict):
        return None, {}
    view = settings.get("_sketch_view") or settings.get("sketch_view")
    options = settings.get("_export_options")
    if not isinstance(options, dict):
        options = {}
    return view, options


def export_project_to_workspace(project_state, workspace_path):
    """Export current project data into workspace CSVs used by solver backends."""
    if workspace_path is None:
        workspace_path = ""
    try:
        _ = os.path.abspath(str(workspace_path))
    except Exception:
        pass

    view, options = _resolve_export_view(project_state)
    if view is None:
        return False

    silent = bool(options.get("silent", False))
    export_mode = str(options.get("export_mode", "full"))
    async_mesh = bool(options.get("async_mesh", True))
    force_remesh = bool(options.get("force_remesh", False))

    return bool(
        export_csv_impl(
            view,
            project_state=project_state,
            silent=silent,
            export_mode=export_mode,
            async_mesh=async_mesh,
            force_remesh=force_remesh,
        )
    )

def write_time_series_csv(self, path, profiles, prefix, axes, value_scale=1.0):
    profiles = profiles or {}
    time_step, total_steps = self._get_sim_time_settings()
    if time_step <= 0 or total_steps < 0:
        time_step = 0.0
        total_steps = 0
    if not os.path.isabs(path):
        path = self._workspace_path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ids = sorted(profiles.keys())
    try:
        scale = float(value_scale)
    except Exception:
        scale = 1.0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["time"]
        for pid in ids:
            for axis in axes:
                header.append(f"{prefix}{pid}{axis}")
        w.writerow(header)
        if not ids:
            return
        for step in range(total_steps + 1):
            t = step * time_step
            row = [f"{t:.6f}"]
            for pid in ids:
                vals = self._eval_profile_at_time(
                    profiles.get(pid, []),
                    t,
                    [f"{prefix}{axis}" for axis in axes],
                )
                if scale != 1.0:
                    vals = [float(v) * scale for v in vals]
                row.extend([f"{v:.6f}" for v in vals])
            w.writerow(row)

def write_time_profiles_config(self, force_profiles, velocity_profiles):
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CPD-main", "config.yml")
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
    except Exception:
        config = {}
    time_profiles = config.get("time_profiles", {})
    forces_out = []
    for fid in sorted(force_profiles.keys()):
        segments = force_profiles.get(fid, [])
        forces_out.append({"id": int(fid), "segments": segments})
    velocities_out = []
    for vid in sorted(velocity_profiles.keys()):
        segments = velocity_profiles.get(vid, [])
        velocities_out.append({"id": int(vid), "segments": segments})
    time_profiles["forces"] = forces_out
    time_profiles["velocities"] = velocities_out
    try:
        time_profiles["unit_scale_to_m"] = float(self._unit_scale_to_meters())
    except Exception:
        time_profiles["unit_scale_to_m"] = 1.0
    time_profiles["unit"] = "m"
    config["time_profiles"] = time_profiles
    try:
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
    except Exception:
        pass

def save_velocity_csv(self, path=None, write_header=True):
    """Persist current velocity BCs (solver format by default)."""
    velocity_map = self._build_velocity_map()
    if path is None:
        path = self._workspace_input_path("velocity.csv")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["particle_id", "vx", "vy"])
            for nid in sorted(velocity_map.keys()):
                vx, vy = velocity_map[nid]
                w.writerow([nid, f"{vx:.6f}", f"{vy:.6f}"])
        return True
    except Exception:
        return False

# --- Material & Operation Management ---

def export_cpd_main_inputs(
    self,
    nodes,
    fixed_nodes,
    _force_acc,
    velocity_map=None,
    particle_material_map=None,
):
    """Write CPD-main solver inputs (particles/fixed/velocity) into workspace/input."""
    input_dir = self._workspace_input_path()
    os.makedirs(input_dir, exist_ok=True)
    legacy_setup_dir = self._workspace_path("setup")

    nodes_for_cpd = self._convert_nodes_to_meters(nodes)
    self._warn_if_cpd_bounds_exceeded(nodes_for_cpd)
    particles_path = os.path.join(input_dir, "particles.csv")
    with open(particles_path, "w", newline="") as f:
        w = csv.writer(f)
        # Solver input particles are geometry only; material assignment belongs to connections/elements.
        w.writerow(["particle_id", "x", "y"])
        for nid, (x, y) in enumerate(nodes_for_cpd):
            w.writerow([int(nid), f"{float(x):.6f}", f"{float(y):.6f}"])
    # Keep only one canonical particles file in input.
    legacy_nodes_path = os.path.join(input_dir, "nodes.csv")
    if os.path.exists(legacy_nodes_path):
        try:
            os.remove(legacy_nodes_path)
        except Exception:
            pass
    # Remove deprecated static force setup file; force inputs now flow through
    # workspace force target/time profile files.
    legacy_force_path = os.path.join(input_dir, "for.csv")
    if os.path.exists(legacy_force_path):
        try:
            os.remove(legacy_force_path)
        except Exception:
            pass

    fixed_path = os.path.join(input_dir, "fixed.csv")
    with open(fixed_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["particle_id"])
        for nid in sorted(fixed_nodes):
            w.writerow([int(nid)])

    velocity_map = velocity_map or {}
    velocity_path = os.path.join(input_dir, "velocity.csv")
    with open(velocity_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["particle_id", "vx", "vy"])
        for nid in sorted(velocity_map.keys()):
            vx, vy = velocity_map[nid]
            w.writerow([int(nid), f"{float(vx):.6f}", f"{float(vy):.6f}"])

    # Remove stale legacy setup copies to keep a single canonical input folder.
    for fname in (
        "particles.csv",
        "connections.csv",
        "materials.csv",
        "nodes.csv",
        "fixed.csv",
        "velocity.csv",
        "for.csv",
    ):
        stale = os.path.join(legacy_setup_dir, fname)
        if os.path.exists(stale):
            try:
                os.remove(stale)
            except Exception:
                pass

def export_csv_impl(self, project_state=None, silent=False, export_mode="full", async_mesh=True, force_remesh=False):
    logger = logging.getLogger(__name__)
    try:
        setattr(self, "_last_export_error", "")
    except Exception:
        pass
    needs_mesh = bool(force_remesh) or self.global_nodes.size == 0 or self.global_elements.size == 0
    if not needs_mesh:
        needs_part_mapping = any(
            item.get("part_id") is not None for item in (self.bcs or [])
        ) or any(
            item.get("part_id") is not None for item in (self.loads or [])
        )
        if needs_part_mapping and not self.element_part_map:
            needs_mesh = True

    if needs_mesh:
        if not silent:
            QMessageBox.warning(
                self,
                "Export",
                "Particles and connections must already exist. Generate particles and click Generate Connections before exporting.",
            )
        return False

    final_nodes_array = self.global_nodes
    export_full = (export_mode == "full")
    tol = max(1.0, DEFAULT_DX * 0.6)
    total_time = self._get_sim_total_time()
    length_scale = self._unit_scale_to_meters()
    velocity_scale = length_scale

    def _remove_workspace_file(fname):
        path = self._workspace_path(fname)
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

    def _dedupe_node_mappings(entries):
        merged = {}
        for item in entries or []:
            if not isinstance(item, dict):
                continue
            bc_obj = item.get("bc")
            if bc_obj is None:
                continue
            try:
                node_id = int(item.get("node_id"))
            except Exception:
                continue
            try:
                node_count = float(item.get("node_count", 1.0))
            except Exception:
                node_count = 1.0
            if node_count <= 0:
                node_count = 1.0
            key = (node_id, id(bc_obj))
            if key in merged:
                merged[key]["node_count"] = max(float(merged[key].get("node_count", 1.0)), node_count)
                continue
            merged[key] = {"node_id": node_id, "bc": bc_obj, "node_count": node_count}
        return list(merged.values())


    try:
        element_map = {item["element_idx"]: item for item in self.element_part_map}
        connections_rows, conn_export_summary = self._build_connections_export_rows()
        interface_warnings = self._validate_interface_definitions()
        export_materials = _resolve_export_materials(project_state, self)
        connections_rows, heterogeneity_warnings = _apply_heterogeneous_material_assignments(
            connections_rows,
            export_materials,
            project_state,
            self,
        )
        export_qa_messages = (
            list(conn_export_summary.get("warnings", []))
            + list(interface_warnings)
            + list(heterogeneity_warnings)
        )
        self._report_frontend_mesh_export_warnings(export_qa_messages, silent=silent)
        used_material_ids = set()
        invalid_material_ids = set()
        for row in connections_rows:
            material_value = row.get("material_id")
            if material_value in (None, ""):
                continue
            try:
                used_material_ids.add(int(material_value))
            except Exception:
                invalid_material_ids.add(str(material_value))
        if invalid_material_ids:
            msg = (
                "Export failed: triangles contain invalid material IDs. "
                f"Invalid values: {', '.join(sorted(invalid_material_ids))}"
            )
            try:
                setattr(self, "_last_export_error", msg)
            except Exception:
                pass
            if not silent:
                QMessageBox.critical(self, "Export Error", msg)
            return False
        used_material_ids = sorted(used_material_ids)
        particle_materials = {}
        for row in connections_rows:
            mat_id = row.get("material_id")
            if mat_id in (None, ""):
                continue
            try:
                mat_id = int(mat_id)
            except Exception:
                continue
            for key in ("p1", "p2", "p3"):
                try:
                    node_id = int(row.get(key))
                except Exception:
                    continue
                mats = particle_materials.setdefault(node_id, set())
                mats.add(mat_id)

        def _particle_material_value(node_id):
            mats = sorted(particle_materials.get(int(node_id), set()))
            if not mats:
                return ""
            if len(mats) == 1:
                return mats[0]
            return "|".join(str(mid) for mid in mats)

        if export_full:
            input_dir = self._workspace_input_path()
            os.makedirs(input_dir, exist_ok=True)

            # 1. Preview geometry (UI units only, not solver-facing)
            with open(self._workspace_path("preview_particles.csv"), "w", newline='') as f:
                f.write("# unit: m\n")
                w = csv.writer(f)
                w.writerow(["particle_id", "x", "y", "meta"])
                for i, (x, y) in enumerate(final_nodes_array):
                    w.writerow([i, f"{x:.6f}", f"{y:.6f}", ""])
            _remove_workspace_file("particles.csv")

            # 2. Solver connectivity (indices only)
            with open(os.path.join(input_dir, "connections.csv"), "w", newline='') as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "triangle_id",
                        "p1",
                        "p2",
                        "p3",
                        "part_id",
                        "material_id",
                        "E",
                        "nu",
                        "rho",
                        "fail_SE",
                        "c",
                    ]
                )
                for row in connections_rows:
                    w.writerow(
                        [
                            row.get("triangle_id", ""),
                            row.get("p1", ""),
                            row.get("p2", ""),
                            row.get("p3", ""),
                            row.get("part_id", ""),
                            row.get("material_id", ""),
                            row.get("E", ""),
                            row.get("nu", ""),
                            row.get("rho", ""),
                            row.get("fail_SE", ""),
                            row.get("c", ""),
                        ]
                    )
            _remove_workspace_file("connections.csv")

            # 3. Solver materials (SI units only)
            if used_material_ids:
                missing_ids = [mid for mid in used_material_ids if mid not in export_materials]
                if missing_ids:
                    msg = (
                        "Export failed: triangles reference material IDs that are missing from project materials. "
                        f"Missing IDs: {', '.join(str(mid) for mid in missing_ids)}"
                    )
                    try:
                        setattr(self, "_last_export_error", msg)
                    except Exception:
                        pass
                    if not silent:
                        QMessageBox.critical(self, "Export Error", msg)
                    return False

            if used_material_ids:
                material_rows = [_solver_material_row(export_materials[material_id]) for material_id in used_material_ids]
                preferred_keys = all_registry_parameter_keys()
                used_keys = []
                seen_keys = set()
                for key in preferred_keys:
                    for row in material_rows:
                        if key in (row.get("parameters", {}) or {}):
                            seen_keys.add(key)
                            used_keys.append(key)
                            break
                for row in material_rows:
                    for key in (row.get("parameters", {}) or {}).keys():
                        if key not in seen_keys:
                            seen_keys.add(key)
                            used_keys.append(key)
                with open(os.path.join(input_dir, "materials.csv"), "w", newline='') as f:
                    w = csv.writer(f)
                    w.writerow(
                        [
                            "material_id",
                            "name",
                            "behavior",
                            "symmetry",
                            "damage",
                            "E",
                            "nu",
                            "rho",
                            "fail_SE",
                            "c",
                            *used_keys,
                        ]
                    )
                    for row in material_rows:
                        params = row.get("parameters", {}) or {}
                        w.writerow(
                            [
                                row["material_id"],
                                row["name"],
                                row.get("behavior", ""),
                                row.get("symmetry", ""),
                                row.get("damage", ""),
                                f"{float(row['E']):.12g}",
                                f"{float(row['nu']):.12g}",
                                f"{float(row['rho']):.12g}",
                                f"{float(row['fail_SE']):.12g}",
                                f"{float(row['c']):.12g}",
                                *[
                                    ""
                                    if key not in params
                                    else f"{float(params[key]):.12g}" if isinstance(params[key], (int, float))
                                    else params[key]
                                    for key in used_keys
                                ],
                            ]
                        )
            else:
                stale = os.path.join(input_dir, "materials.csv")
                if os.path.exists(stale):
                    try:
                        os.remove(stale)
                    except Exception:
                        pass
                msg = "Export failed: no materials are referenced by the exported triangles."
                try:
                    setattr(self, "_last_export_error", msg)
                except Exception:
                    pass
                if not silent:
                    QMessageBox.critical(self, "Export Error", msg)
                return False
            _remove_workspace_file("materials.csv")

            # 4. Operations/History
            if self.operations:
                with open(self._workspace_path("operations.csv"), "w", newline='') as f:
                    w = csv.writer(f)
                    w.writerow(["op_id", "operation_type", "material_serial", "shape_type"])
                    for op in self.operations:
                        shape_type = op.shape_data.get('type', 'unknown')
                        w.writerow([op.id, op.op_type, op.material_id or "None", shape_type])
            else:
                _remove_workspace_file("operations.csv")

        # Handle Rigid Parts by creating fixed BCs for all their nodes
        rigid_parts = [p for p in self.parts if p.is_rigid]
        rigid_nodes = set()
        if rigid_parts:
            element_map = {item['element_idx']: item for item in self.element_part_map}
            for i, tri in enumerate(self.global_elements):
                part_info = element_map.get(i)
                if part_info and any(p.id == part_info['part_id'] for p in rigid_parts):
                    rigid_nodes.update(tri)

        # 5. BCs
        mapped_bcs = map_geometry_to_nodes(final_nodes_array, self.bcs, tol)
        for bc in self.bcs:
            if bc.get("ids"):
                ids = list(bc.get("ids", []))
                count = max(1, len(ids))
                for nid in ids:
                    mapped_bcs.append({"node_id": int(nid), "bc": bc, "node_count": count})
            part_id = bc.get("part_id")
            if part_id is not None:
                part_nodes = self._part_node_ids_from_mesh(
                    part_id,
                    nodes=final_nodes_array,
                    elements=self.global_elements,
                    element_part_map=self.element_part_map,
                )
                count = max(1, len(part_nodes))
                for nid in part_nodes:
                    mapped_bcs.append({"node_id": int(nid), "bc": bc, "node_count": count})

        # Add rigid node BCs, avoiding duplicates
        for nid in rigid_nodes:
            is_already_fixed = any(m['node_id'] == nid and m['bc']['type'] == 'fix_xy' for m in mapped_bcs)
            if not is_already_fixed:
                mapped_bcs.append({'node_id': nid, 'bc': {'type': 'fix_xy'}})
        mapped_bcs = _dedupe_node_mappings(mapped_bcs)
        debug_fixed_like_rows = []
        for m in mapped_bcs:
            nid = int(m["node_id"])
            btype = str(m["bc"].get("type", ""))
            if btype in ("fix_x", "fix_y", "fix_xy"):
                debug_fixed_like_rows.append((nid, btype, float(m["bc"].get("val", 0.0) or 0.0)))
        if debug_fixed_like_rows:
            logger.info("BC export fixed/partial-fixed candidates: %d", len(debug_fixed_like_rows))
            for nid, btype, val in debug_fixed_like_rows:
                logger.debug("BC export row candidate: node=%d type=%s value=%s", nid, btype, val)

        if export_full:
            with open(self._workspace_path("bc.csv"), "w", newline='') as f:
                f.write("# unit_system: SI\n")
                f.write("# length_unit: m\n")
                f.write("# velocity_unit: m/s\n")
                w = csv.writer(f)
                w.writerow(["bc_id", "target_id", "target_type", "bc_type", "components", "values", "comment"])
                bc_id_counter = 0
                processed_node_bcs = set() # Avoid duplicates for nodes on shared edges
                for m in mapped_bcs:
                    nid = int(m['node_id'])
                    btype = m['bc']['type']
                    if (nid, btype) in processed_node_bcs: continue

                    target_type = 'particle'
                    comment = ''
                    if btype == "fix_xy":
                        logger.debug("Exporting BC row: node=%d type=fix_xy -> ux=0, uy=0", nid)
                        w.writerow([bc_id_counter, nid, target_type, 'fixed', 'ux', 0, comment]); bc_id_counter += 1
                        w.writerow([bc_id_counter, nid, target_type, 'fixed', 'uy', 0, comment]); bc_id_counter += 1
                    elif btype == "fix_x":
                        logger.debug("Exporting BC row: node=%d type=fix_x -> ux=0", nid)
                        w.writerow([bc_id_counter, nid, target_type, 'roller', 'ux', 0, comment]); bc_id_counter += 1
                    elif btype == "fix_y":
                        logger.debug("Exporting BC row: node=%d type=fix_y -> uy=0", nid)
                        w.writerow([bc_id_counter, nid, target_type, 'roller', 'uy', 0, comment]); bc_id_counter += 1
                    elif btype == "velocity_x":
                        v_si = float(m['bc'].get("val", 0.0)) * velocity_scale
                        w.writerow([bc_id_counter, nid, target_type, 'velocity', 'vx', f"{v_si:.6f}", comment]); bc_id_counter += 1
                    elif btype == "velocity_y":
                        v_si = float(m['bc'].get("val", 0.0)) * velocity_scale
                        w.writerow([bc_id_counter, nid, target_type, 'velocity', 'vy', f"{v_si:.6f}", comment]); bc_id_counter += 1

                    processed_node_bcs.add((nid, btype))

        # 6. Loads (Forces and Moments)
        # Process Forces
        force_loads = [ld for ld in self.loads if ld.get("type") == "force"]
        mapped_forces = map_geometry_to_nodes(final_nodes_array, force_loads, tol)
        for ld in [ld for ld in self.loads if ld.get("type") == "force" and ld.get("ids")]:
            ids = list(ld.get("ids", []))
            count = max(1, len(ids))
            for nid in ids:
                mapped_forces.append({"node_id": int(nid), "bc": ld, "node_count": count})
        for ld in [ld for ld in self.loads if ld.get("type") == "force" and ld.get("part_id") is not None]:
            part_nodes = self._part_node_ids_from_mesh(
                ld.get("part_id"),
                nodes=final_nodes_array,
                elements=self.global_elements,
                element_part_map=self.element_part_map,
            )
            count = max(1, len(part_nodes))
            for nid in part_nodes:
                mapped_forces.append({"node_id": int(nid), "bc": ld, "node_count": count})
        mapped_forces = _dedupe_node_mappings(mapped_forces)
        if force_loads and not mapped_forces:
            msg = (
                "Force load(s) are defined but no mesh nodes were targeted.\n"
                "Re-apply force on valid edges/faces or regenerate particle connections."
            )
            if not silent:
                QMessageBox.warning(self, "Force Mapping", msg)
            return False
        force_acc = {}
        force_targets = {}
        force_profiles = {}
        for ld in force_loads:
            fid = self._ensure_force_id(ld)
            force_profiles[fid] = self._normalize_force_profile(ld, total_time)
        for m in mapped_forces:
            nid = int(m['node_id']); bc = m['bc']
            fid = self._ensure_force_id(bc)
            force_targets.setdefault(nid, set()).add(fid)
            share = 1.0 / m.get('node_count', 1.0)
            fx = bc.get('fx', 0.0) * share
            fy = bc.get('fy', 0.0) * share
            fz = bc.get('fz', 0.0) * share
            if nid not in force_acc:
                force_acc[nid] = [0.0, 0.0, 0.0]
            force_acc[nid][0] += fx
            force_acc[nid][1] += fy
            force_acc[nid][2] += fz

        # Process Moments
        moment_loads = [ld for ld in self.loads if ld.get("type") == "moment"]
        mapped_moments = map_geometry_to_nodes(final_nodes_array, moment_loads, tol)
        for ld in [ld for ld in self.loads if ld.get("type") == "moment" and ld.get("ids")]:
            ids = list(ld.get("ids", []))
            count = max(1, len(ids))
            for nid in ids:
                mapped_moments.append({"node_id": int(nid), "bc": ld, "node_count": count})
        for ld in [ld for ld in self.loads if ld.get("type") == "moment" and ld.get("part_id") is not None]:
            part_nodes = self._part_node_ids_from_mesh(
                ld.get("part_id"),
                nodes=final_nodes_array,
                elements=self.global_elements,
                element_part_map=self.element_part_map,
            )
            count = max(1, len(part_nodes))
            for nid in part_nodes:
                mapped_moments.append({"node_id": int(nid), "bc": ld, "node_count": count})
        mapped_moments = _dedupe_node_mappings(mapped_moments)
        mom_acc = {}
        for m in mapped_moments:
            nid = int(m['node_id'])
            val = m['bc'].get('m', 0.0)
            mom_acc[nid] = mom_acc.get(nid, 0.0) + val

        if export_full:
            with open(self._workspace_path("loads.csv"), "w", newline='') as f:
                f.write("# loads_unit: N\n")
                f.write("# moments_unit: N*m\n")
                f.write("# unit: m\n")
                w = csv.writer(f)
                w.writerow(["load_id", "target_id", "target_type", "load_type", "components", "magnitude", "direction", "comment"])
                load_id_counter = 0

                for nid in sorted(force_acc.keys()):
                    fx, fy, _fz = force_acc[nid]
                    if abs(fx) > 1e-9:
                        w.writerow([load_id_counter, nid, 'particle', 'force', 'fx', f"{fx:.6f}", '', ''])
                        load_id_counter += 1
                    if abs(fy) > 1e-9:
                        w.writerow([load_id_counter, nid, 'particle', 'force', 'fy', f"{fy:.6f}", '', ''])
                        load_id_counter += 1

                for nid in sorted(mom_acc.keys()):
                    moment = mom_acc[nid]
                    if abs(moment) > 1e-9:
                        w.writerow([load_id_counter, nid, 'particle', 'moment', 'm', f"{moment:.6f}", '', ''])
                        load_id_counter += 1

        # Force target mapping (always written for solver)
        os.makedirs(self._workspace_input_path(), exist_ok=True)
        if force_targets:
            max_overlap = max(len(ids) for ids in force_targets.values())
        else:
            max_overlap = 0
        with open(self._workspace_input_path("force_targets.csv"), "w", newline="") as f:
            w = csv.writer(f)
            header = ["particle_id"] + [f"force_id_{i+1}" for i in range(max_overlap)]
            w.writerow(header)
            for nid in sorted(force_targets.keys()):
                ids = sorted(force_targets[nid])
                row = [nid] + ids + [""] * (max_overlap - len(ids))
                w.writerow(row)
        # -------------------------------------------------
        # Export particle velocities (m/s): particle_id, vx, vy
        # -------------------------------------------------
        vel_scale = self._unit_scale_to_meters()
        velocity_map = {}
        for m in mapped_bcs:
            nid = int(m["node_id"])
            btype = m["bc"]["type"]
            if btype not in ("velocity_x", "velocity_y", "fix_x", "fix_y"):
                continue
            vel_id = self._ensure_velocity_id(m["bc"])
            if nid not in velocity_map:
                velocity_map[nid] = [0.0, 0.0]
            val = float(m["bc"].get("val", 0.0)) * vel_scale
            if btype in ("velocity_x", "fix_x"):
                velocity_map[nid][0] = val
            else:
                velocity_map[nid][1] = val

        velocity_targets = {}
        velocity_profiles = {}
        partial_fixed_rows = []
        velocity_bcs = [
            b
            for b in self.bcs
            if b.get("type") in ("velocity_x", "velocity_y", "velocity_z", "fix_x", "fix_y")
        ]
        for bc in velocity_bcs:
            vid = self._ensure_velocity_id(bc)
            velocity_profiles[vid] = self._normalize_velocity_profile(bc, total_time)
        for m in mapped_bcs:
            nid = int(m["node_id"])
            btype = m["bc"]["type"]
            if btype not in ("velocity_x", "velocity_y", "velocity_z", "fix_x", "fix_y"):
                continue
            vid = self._ensure_velocity_id(m["bc"])
            velocity_targets.setdefault(nid, set()).add(vid)
            if btype in ("fix_x", "fix_y"):
                partial_fixed_rows.append((nid, btype, vid))
        if velocity_bcs and not velocity_targets:
            msg = (
                "Velocity BC(s) are defined but no mesh nodes were targeted.\n"
                "Re-apply velocity BC on valid geometry or regenerate particle connections."
            )
            if not silent:
                QMessageBox.warning(self, "Velocity Mapping", msg)
            return False

        # Velocity target mapping (always written for solver)
        if velocity_targets:
            max_overlap = max(len(ids) for ids in velocity_targets.values())
        else:
            max_overlap = 0
        with open(self._workspace_input_path("velocity_targets.csv"), "w", newline="") as f:
            w = csv.writer(f)
            header = ["particle_id"] + [f"velocity_id_{i+1}" for i in range(max_overlap)]
            w.writerow(header)
            for nid in sorted(velocity_targets.keys()):
                ids = sorted(velocity_targets[nid])
                row = [nid] + ids + [""] * (max_overlap - len(ids))
                w.writerow(row)
        if partial_fixed_rows:
            logger.info("Partial-fixed BC export rows: %d", len(partial_fixed_rows))
            for nid, btype, vid in partial_fixed_rows:
                logger.debug(
                    "Partial-fixed BC exported via velocity targets: node=%d type=%s velocity_id=%d",
                    nid,
                    btype,
                    vid,
                )

        self._write_time_profiles_config(force_profiles, velocity_profiles)
        axes = ("x", "y", "z") if self.project_mode == "3d" else ("x", "y")
        self._write_time_series_csv(
            self._workspace_input_path("force_time.csv"),
            force_profiles,
            "f",
            axes,
            value_scale=1.0,
        )
        # Velocity profiles are solver-facing; always export in m/s.
        self._write_time_series_csv(
            self._workspace_input_path("velocity_time.csv"),
            velocity_profiles,
            "v",
            axes,
            value_scale=self._unit_scale_to_meters(),
        )
        # Keep target/time-series files only in workspace/input.
        for stale_dir in (self._workspace_path(), self._workspace_path("setup")):
            for fname in (
                "force_targets.csv",
                "velocity_targets.csv",
                "force_time.csv",
                "velocity_time.csv",
            ):
                stale = os.path.join(stale_dir, fname)
                if os.path.exists(stale):
                    try:
                        os.remove(stale)
                    except Exception:
                        pass
        # Remove redundant legacy debug derivatives.
        for stale in ("fixed_particles.csv", "particle_forces.csv", "particle_velocities.csv"):
            _remove_workspace_file(stale)
        # Keep workspace/input/velocity.csv in sync for the solver.
        self.save_velocity_csv()

        # 7. Interfaces
        if self.interfaces:
                with open(self._workspace_path("interfaces.csv"), "w", newline='') as f:
                    f.write("# unit: m\n")
                    w = csv.writer(f)
                    w.writerow(
                        [
                            "interface_id",
                            "part1_id",
                            "part2_id",
                            "interface_type",
                            "material_id",
                            "material_mode",
                            "thickness",
                            "target_dx",
                            "layer_mode",
                            "placement_mode",
                            "friction_coeff",
                            "status",
                            "notes",
                        ]
                    )
                    for iface in self.interfaces:
                        w.writerow([
                            iface.id,
                            iface.part1_id,
                            iface.part2_id,
                            iface.interface_type,
                            getattr(iface, "material_id", ""),
                            getattr(iface, "material_mode", "auto"),
                            getattr(iface, "thickness", ""),
                            getattr(iface, "target_dx", ""),
                            getattr(iface, "layer_mode", "single_layer_ring"),
                            getattr(iface, "placement_mode", getattr(Interface, "DEFAULT_PLACEMENT_MODE", "matrix_side")),
                            getattr(iface, "friction_coeff", 0.0),
                            getattr(iface, "status", ""),
                            getattr(iface, "notes", ""),
                        ])
        else:
            _remove_workspace_file("interfaces.csv")
        # Remove legacy filename during migration.
        _remove_workspace_file("interactions.csv")

        # 9. Create aggregated solver_particles.csv
        # First, create maps for quick lookup of BCs and loads per node
        node_bcs = {}  # {nid: {'ux': val, 'uy': val}}
        fixed_components = {}  # {nid: {'ux': 0, 'uy': 0}} from fixed-only BC types
        for m in mapped_bcs:
            nid = int(m['node_id'])
            if nid not in node_bcs:
                node_bcs[nid] = {}
            btype = m['bc']['type']
            if btype == 'fix_xy':
                node_bcs[nid]['ux'] = 0
                node_bcs[nid]['uy'] = 0
                fixed_components[nid] = {'ux': 0, 'uy': 0}
            elif btype == 'fix_x':
                node_bcs[nid]['ux'] = 0
                fixed_components.setdefault(nid, {})['ux'] = 0
            elif btype == 'fix_y':
                node_bcs[nid]['uy'] = 0
                fixed_components.setdefault(nid, {})['uy'] = 0
            elif btype == 'velocity_x':  # Prescribed velocity for solver particles export.
                node_bcs[nid]['ux'] = float(m['bc'].get("val", 0.0)) * velocity_scale
            elif btype == 'velocity_y':
                node_bcs[nid]['uy'] = float(m['bc'].get("val", 0.0)) * velocity_scale

        if export_full:
            with open(self._workspace_path("solver_particles.csv"), "w", newline='') as f:
                f.write("# unit_system: SI\n")
                f.write("# length_unit: m\n")
                f.write("# velocity_unit: m/s\n")
                f.write("# force_unit: N\n")
                f.write("# moment_unit: N*m\n")
                w = csv.writer(f)
                w.writerow(["particle_id", "x", "y", "ux", "uy", "fx", "fy", "m", "material_serial"])
                for i, (x, y) in enumerate(final_nodes_array):
                    nid = i
                    ux = node_bcs.get(nid, {}).get('ux', '')
                    uy = node_bcs.get(nid, {}).get('uy', '')
                    x_si = float(x) * length_scale
                    y_si = float(y) * length_scale
                    # Use .get for forces and moments as well, providing a default for nodes with no loads
                    fx_val, fy_val, _fz_val = force_acc.get(nid, ('', '', ''))
                    fx = f"{fx_val:.6f}" if isinstance(fx_val, float) else ''
                    fy = f"{fy_val:.6f}" if isinstance(fy_val, float) else ''
                    moment_val = mom_acc.get(nid, '')
                    moment = f"{moment_val:.6f}" if isinstance(moment_val, float) else ''
                    ux_out = f"{float(ux):.6f}" if isinstance(ux, (int, float)) else ux
                    uy_out = f"{float(uy):.6f}" if isinstance(uy, (int, float)) else uy
                    w.writerow([nid, f"{x_si:.6f}", f"{y_si:.6f}", ux_out, uy_out, fx, fy, moment, _particle_material_value(nid)])

        # Collect fully fixed particles from fixed BC components (ux=0 AND uy=0)
        # and pass to CPD-main input export.
        fully_fixed_nodes = []

        for nid, comps in fixed_components.items():
            if comps.get('ux', None) == 0 and comps.get('uy', None) == 0:
                fully_fixed_nodes.append(nid)
        if fully_fixed_nodes:
            logger.info("Fully fixed BC export rows: %d", len(fully_fixed_nodes))
            for nid in sorted(fully_fixed_nodes):
                logger.debug("Fully fixed BC exported to fixed.csv: node=%d", int(nid))

        particle_material_map = {
            int(i): _particle_material_value(i)
            for i in range(len(final_nodes_array))
        }
        self._export_cpd_main_inputs(
            final_nodes_array,
            fully_fixed_nodes,
            force_acc,
            velocity_map,
            particle_material_map=particle_material_map,
        )

        export_3d = (
            export_full
            and self.project_mode == "3d"
            and self.global_nodes_3d.size > 0
            and self.global_elements_3d.size > 0
        )
        if export_3d:
            n2d = len(final_nodes_array)
            layers = int(self.extrude_layers)

            with open(self._workspace_path("particles_3d.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["particle_id", "x", "y", "z"])
                for nid, (x, y, z) in enumerate(self.global_nodes_3d):
                    w.writerow([nid, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])

            part_map_3d = {m["element_idx"]: m for m in self.element_part_map_3d}
            with open(self._workspace_path("connections_3d.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["cid", "p1", "p2", "p3", "p4", "part_id", "material_id"])
                for eid, tet in enumerate(self.global_elements_3d):
                    part_info = part_map_3d.get(eid, {})
                    w.writerow(
                        [
                            eid,
                            int(tet[0]),
                            int(tet[1]),
                            int(tet[2]),
                            int(tet[3]),
                            part_info.get("part_id", ""),
                            part_info.get("material_id", ""),
                        ]
                    )

            if n2d > 0:
                bc_components = {}
                vel_components = {}
                for m in mapped_bcs:
                    nid = int(m["node_id"])
                    btype = m["bc"]["type"]
                    if btype in ("fix_xy", "fix_x", "fix_y", "fix_z"):
                        comps = bc_components.setdefault(nid, {})
                        if btype in ("fix_xy", "fix_x"):
                            comps["ux"] = 0
                        if btype in ("fix_xy", "fix_y"):
                            comps["uy"] = 0
                        if btype == "fix_z":
                            comps["uz"] = 0
                    if btype in ("velocity_x", "velocity_y", "velocity_z"):
                        comps = vel_components.setdefault(nid, {})
                        val = float(m["bc"].get("val", 0.0))
                        if btype == "velocity_x":
                            comps["vx"] = val
                        elif btype == "velocity_y":
                            comps["vy"] = val
                        else:
                            comps["vz"] = val

                fixed_nodes_3d = []
                for nid, comps in bc_components.items():
                    if comps.get("ux") == 0 and comps.get("uy") == 0 and comps.get("uz") == 0:
                        for li in range(layers + 1):
                            fixed_nodes_3d.append(li * n2d + nid)
                with open(self._workspace_path("fixed_particles_3d.csv"), "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["particle_id"])
                    for nid in sorted(set(fixed_nodes_3d)):
                        w.writerow([nid])

                with open(self._workspace_path("bc_3d.csv"), "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["particle_id", "ux", "uy", "uz"])
                    for nid, comps in sorted(bc_components.items()):
                        for li in range(layers + 1):
                            w.writerow(
                                [
                                    li * n2d + nid,
                                    comps.get("ux", ""),
                                    comps.get("uy", ""),
                                    comps.get("uz", ""),
                                ]
                            )

                with open(self._workspace_path("velocity_bc_3d.csv"), "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["particle_id", "vx", "vy", "vz"])
                    for nid, comps in sorted(vel_components.items()):
                        for li in range(layers + 1):
                            w.writerow(
                                [
                                    li * n2d + nid,
                                    comps.get("vx", 0.0),
                                    comps.get("vy", 0.0),
                                    comps.get("vz", 0.0),
                                ]
                            )

                with open(self._workspace_path("force_bc_3d.csv"), "w", newline="") as f:
                    w = csv.writer(f)
                    for nid in sorted(force_acc.keys()):
                        fx, fy, fz = force_acc[nid]
                        if abs(fx) < 1e-12 and abs(fy) < 1e-12 and abs(fz) < 1e-12:
                            continue
                        for li in range(layers + 1):
                            w.writerow(
                                [
                                    li * n2d + nid,
                                    f"{float(fx):.6f}",
                                    f"{float(fy):.6f}",
                                    f"{float(fz):.6f}",
                                ]
                            )

        else:
            for stale in (
                "particles_3d.csv",
                "connections_3d.csv",
                "fixed_particles_3d.csv",
                "bc_3d.csv",
                "velocity_bc_3d.csv",
                "force_bc_3d.csv",
            ):
                _remove_workspace_file(stale)

        # 10. Initial Velocities
        if export_full and self.initial_velocities:
            vel_scale = self._unit_scale_to_meters()
            with open(self._workspace_path("initial_velocities.csv"), "w", newline='') as f:
                w = csv.writer(f)
                w.writerow(["particle_id", "vx", "vy"])

                written_nodes = set()
                element_map = {item['element_idx']: item for item in self.element_part_map}

                for iv in self.initial_velocities:
                    part_id = iv['part_id']
                    vx = float(iv['vx']) * vel_scale
                    vy = float(iv['vy']) * vel_scale

                    for i, tri in enumerate(self.global_elements):
                        part_info = element_map.get(i)
                        if part_info and part_info['part_id'] == part_id:
                            for node_idx in tri:
                                if node_idx not in written_nodes:
                                    w.writerow([node_idx, vx, vy])
                                    written_nodes.add(node_idx)
        else:
            _remove_workspace_file("initial_velocities.csv")

        if not silent:
            QMessageBox.information(
                self,
                "Export",
                "Job Complete: Global connection CSV files exported to workspace/ successfully.",
            )
        return True

    except Exception as e:
        if not silent:
            QMessageBox.critical(self, "Export Error", str(e))
        return False

# --- Input Events ---
