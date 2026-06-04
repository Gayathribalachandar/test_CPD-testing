"""Schema/version helpers for CPD project files."""

from __future__ import annotations

import copy
from typing import Callable, Dict, Any, Mapping

from models import Interface, Material
from project_state import ProjectState

APP_VERSION = "CPD SimStudio v28"
CURRENT_SCHEMA_VERSION = 6

DEFAULT_CONNECTION_SETTINGS = {
    "min_spacing_factor": 1.0,
    "boundary_thickness": 0.0,
    "boundary_spacing_factor": 1.0,
}

DEFAULT_PREVIEW_SETTINGS = {
    "fast_preview_enabled": True,
    "fast_preview_connection_limit": 0,
    "gpu_point_preview_enabled": True,
    "gpu_point_preview_auto": True,
    "gpu_point_preview_threshold": 250_000,
}


_UNSERIALIZABLE = object()


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _migrate_to_1(project_data: Dict[str, Any]) -> None:
    meta = project_data.setdefault("project_meta", {})
    meta.setdefault("version", "1.0")
    meta.setdefault("mode", "2d")
    meta.setdefault("last_stage", "GEOMETRY")


def _migrate_to_2(project_data: Dict[str, Any]) -> None:
    if "mesh_settings" in project_data and "connection_settings" not in project_data:
        project_data["connection_settings"] = project_data.pop("mesh_settings")


def _migrate_to_3(project_data: Dict[str, Any]) -> None:
    preview = project_data.setdefault("preview_settings", {})
    for key, value in DEFAULT_PREVIEW_SETTINGS.items():
        preview.setdefault(key, value)


def _migrate_to_4(project_data: Dict[str, Any]) -> None:
    conn = project_data.setdefault("connection_settings", {})
    for key, value in DEFAULT_CONNECTION_SETTINGS.items():
        conn.setdefault(key, value)


def _migrate_to_5(project_data: Dict[str, Any]) -> None:
    connections = project_data.setdefault("connections", {})
    # Copy legacy mesh references (if any)
    legacy_mesh = project_data.pop("mesh", {})
    if legacy_mesh:
        legacy_particles = legacy_mesh.get("particles")
        if legacy_particles is None:
            legacy_particles = legacy_mesh.get("solver_particles")
        if legacy_particles is None:
            legacy_particles = legacy_mesh.get("nodes")
        legacy_connections = legacy_mesh.get("connections")
        if legacy_connections is None:
            legacy_connections = legacy_mesh.get("elements")
        if legacy_particles is not None and "particles" not in connections:
            connections["particles"] = legacy_particles
        if legacy_connections is not None and "connections" not in connections:
            connections["connections"] = legacy_connections
    # Normalize any nodes/elements authored in older schemas
    if "particles" not in connections and "nodes" in connections:
        connections["particles"] = connections.pop("nodes")
    if "connections" not in connections and "elements" in connections:
        connections["connections"] = connections.pop("elements")


def _migrate_to_6(project_data: Dict[str, Any]) -> None:
    # Canonicalize interface definitions (formerly "interactions").
    if "interfaces" not in project_data and "interactions" in project_data:
        project_data["interfaces"] = project_data.get("interactions", [])


MIGRATIONS: Dict[int, Callable[[Dict[str, Any]], None]] = {
    1: _migrate_to_1,
    2: _migrate_to_2,
    3: _migrate_to_3,
    4: _migrate_to_4,
    5: _migrate_to_5,
    6: _migrate_to_6,
}


def apply_schema_migrations(project_data: Dict[str, Any]) -> Dict[str, Any]:
    meta = project_data.setdefault("project_meta", {})
    current_version = _to_int(meta.get("schema_version"))
    next_version = max(1, current_version + 1) if current_version < CURRENT_SCHEMA_VERSION else CURRENT_SCHEMA_VERSION
    for version in range(current_version + 1, CURRENT_SCHEMA_VERSION + 1):
        migration = MIGRATIONS.get(version)
        if migration:
            migration(project_data)
    meta["schema_version"] = CURRENT_SCHEMA_VERSION
    meta.setdefault("version", "2.0")
    meta.setdefault("app_version", APP_VERSION)
    return project_data


def _json_sanitize(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            sanitized = _json_sanitize(item)
            if sanitized is _UNSERIALIZABLE:
                continue
            out[str(key)] = sanitized
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            sanitized = _json_sanitize(item)
            if sanitized is _UNSERIALIZABLE:
                continue
            out.append(sanitized)
        return out
    return _UNSERIALIZABLE


def _normalize_solver_settings(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    out: Dict[str, Any] = {}
    for key, item in value.items():
        key_str = str(key)
        # Runtime pointers/options are kept in memory only.
        if key_str.startswith("_"):
            continue
        sanitized = _json_sanitize(item)
        if sanitized is _UNSERIALIZABLE:
            continue
        out[key_str] = sanitized
    return out


def _serialize_materials(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    out: Dict[str, Any] = {}
    for key, item in value.items():
        serial = getattr(item, "serial", key)
        try:
            serial_key = str(int(serial))
        except Exception:
            serial_key = str(serial)
        if isinstance(item, Material):
            out[serial_key] = {
                "serial": getattr(item, "serial", key),
                "name": getattr(item, "name", f"Material {serial_key}"),
                "mat_type": getattr(item, "mat_type", ""),
                "behavior": getattr(item, "behavior", ""),
                "damage": getattr(item, "damage", "none"),
                "symmetry": getattr(item, "symmetry", "isotropic"),
                "properties": copy.deepcopy(getattr(item, "properties", {}) or {}),
            }
        elif isinstance(item, Mapping):
            out[serial_key] = {
                "serial": item.get("serial", serial),
                "name": item.get("name", f"Material {serial_key}"),
                "mat_type": item.get("mat_type", item.get("type", "")),
                "behavior": item.get("behavior", ""),
                "damage": item.get("damage", "none"),
                "symmetry": item.get("symmetry", "isotropic"),
                "properties": copy.deepcopy(item.get("properties", {}) or {}),
            }
    return out


def _serialize_interfaces(value: Any) -> list[Dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, Interface):
            out.append(
                {
                    "id": getattr(item, "id", None),
                    "part1_id": getattr(item, "part1_id", None),
                    "part2_id": getattr(item, "part2_id", None),
                    "interface_type": getattr(item, "interface_type", ""),
                    "friction_coeff": getattr(item, "friction_coeff", 0.0),
                    "material_id": getattr(item, "material_id", None),
                    "material_mode": getattr(item, "material_mode", "auto"),
                    "thickness": getattr(item, "thickness", None),
                    "target_dx": getattr(item, "target_dx", None),
                    "layer_mode": getattr(item, "layer_mode", None),
                    "placement_mode": getattr(item, "placement_mode", None),
                    "status": getattr(item, "status", ""),
                    "notes": getattr(item, "notes", ""),
                }
            )
        elif isinstance(item, Mapping):
            out.append(copy.deepcopy(dict(item)))
    return out


def project_state_from_project_data(project_data: Mapping[str, Any] | None) -> ProjectState:
    """Build ProjectState from a migrated project file payload."""
    if not isinstance(project_data, Mapping):
        return ProjectState()
    meta = project_data.get("project_meta", {}) if isinstance(project_data.get("project_meta"), Mapping) else {}
    geometry = project_data.get("geometry", {})
    geometry_parts = geometry.get("parts", []) if isinstance(geometry, Mapping) else []
    payload = {
        "analysis_type": project_data.get("analysis_type", meta.get("analysis_type", "static")),
        "dimension": project_data.get("dimension", meta.get("dimension", "2D")),
        "parts": copy.deepcopy(project_data.get("parts", geometry_parts)),
        "materials": copy.deepcopy(project_data.get("materials", {})),
        "interfaces": copy.deepcopy(
            project_data.get("interfaces", project_data.get("interactions", []))
        ),
        "boundary_conditions": copy.deepcopy(
            project_data.get("boundary_conditions", project_data.get("bcs", []))
        ),
        "loads": copy.deepcopy(project_data.get("loads", [])),
        "mesh_data": copy.deepcopy(project_data.get("mesh_data", {})),
        "solver_settings": _normalize_solver_settings(project_data.get("solver_settings", {})),
        "custom_mesh_zones": copy.deepcopy(project_data.get("custom_mesh_zones", [])),
        "edge_seeds": copy.deepcopy(project_data.get("edge_seeds", [])),
        "edge_seed_templates": copy.deepcopy(project_data.get("edge_seed_templates", [])),
        "vertex_seeds": copy.deepcopy(project_data.get("vertex_seeds", [])),
        "boundary_layer_seeds": copy.deepcopy(project_data.get("boundary_layer_seeds", [])),
        "matched_edge_pairs": copy.deepcopy(project_data.get("matched_edge_pairs", [])),
        "part_mesh_overrides": copy.deepcopy(project_data.get("part_mesh_overrides", {})),
    }
    return ProjectState.from_dict(payload)


def merge_project_state_into_project_data(
    project_data: Dict[str, Any],
    project_state: ProjectState | Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """Write canonical ProjectState payload into schema-compatible keys."""
    if not isinstance(project_data, dict):
        project_data = {}
    if isinstance(project_state, ProjectState):
        state_dict = project_state.to_dict()
    else:
        state_dict = ProjectState.from_dict(project_state).to_dict()

    project_data["analysis_type"] = str(state_dict.get("analysis_type", "static") or "static")
    project_data["dimension"] = str(state_dict.get("dimension", "2D") or "2D")
    project_data["materials"] = _serialize_materials(state_dict.get("materials", {}))
    project_data["interfaces"] = _serialize_interfaces(state_dict.get("interfaces", []))
    project_data["bcs"] = _json_sanitize(copy.deepcopy(state_dict.get("boundary_conditions", [])))
    project_data["loads"] = _json_sanitize(copy.deepcopy(state_dict.get("loads", [])))
    project_data["solver_settings"] = _normalize_solver_settings(
        state_dict.get("solver_settings", {})
    )
    project_data["custom_mesh_zones"] = _json_sanitize(
        copy.deepcopy(state_dict.get("custom_mesh_zones", []))
    )
    project_data["edge_seeds"] = _json_sanitize(
        copy.deepcopy(state_dict.get("edge_seeds", []))
    )
    project_data["edge_seed_templates"] = _json_sanitize(
        copy.deepcopy(state_dict.get("edge_seed_templates", []))
    )
    project_data["vertex_seeds"] = _json_sanitize(
        copy.deepcopy(state_dict.get("vertex_seeds", []))
    )
    project_data["boundary_layer_seeds"] = _json_sanitize(
        copy.deepcopy(state_dict.get("boundary_layer_seeds", []))
    )
    project_data["matched_edge_pairs"] = _json_sanitize(
        copy.deepcopy(state_dict.get("matched_edge_pairs", []))
    )
    project_data["part_mesh_overrides"] = _json_sanitize(
        copy.deepcopy(state_dict.get("part_mesh_overrides", {}))
    )
    return project_data
