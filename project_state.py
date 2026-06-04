"""In-memory project model for UI/solver decoupling."""

from __future__ import annotations

import copy
import math
import numbers
from dataclasses import dataclass, field
from typing import Any, Mapping

from shapely import wkt

from models import (
    BoundaryLayerSeed,
    CustomMeshZone,
    EdgeSeed,
    Interface,
    Material,
    MatchedEdgePair,
    Part,
    VertexSeed,
    normalize_heterogeneity_config,
    normalize_material_field_config,
)
from material_registry import (
    normalize_material_behavior,
    normalize_material_damage,
    normalize_material_properties,
    normalize_material_symmetry,
)


@dataclass
class ProjectState:
    """Container for project data with no UI, I/O, or solver coupling."""

    current_stage: str = "geometry"
    analysis_type: str = "static"
    dimension: str = "2D"
    parts: list[Any] = field(default_factory=list)
    materials: dict[Any, Any] = field(default_factory=dict)
    interfaces: list[Any] = field(default_factory=list)
    boundary_conditions: list[Any] = field(default_factory=list)
    loads: list[Any] = field(default_factory=list)
    mesh_data: dict[str, Any] = field(default_factory=dict)
    solver_settings: dict[str, Any] = field(default_factory=dict)
    custom_mesh_zones: list[Any] = field(default_factory=list)
    edge_seeds: list[Any] = field(default_factory=list)
    edge_seed_templates: list[Any] = field(default_factory=list)
    vertex_seeds: list[Any] = field(default_factory=list)
    boundary_layer_seeds: list[Any] = field(default_factory=list)
    matched_edge_pairs: list[Any] = field(default_factory=list)
    # Per-part sizing overrides: {part_id: {"h_bulk": float}}. Missing keys
    # inherit the global MeshSizingPolicy. Stored as a dict so future fields
    # (e.g. h_feature override, growth rate) can join without a schema change.
    part_mesh_overrides: dict[int, Any] = field(default_factory=dict)

    @staticmethod
    def _is_finite_number(value: Any) -> bool:
        try:
            return math.isfinite(float(value))
        except Exception:
            return False

    @classmethod
    def _is_valid_point(cls, value: Any) -> bool:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return False
        return cls._is_finite_number(value[0]) and cls._is_finite_number(value[1])

    @classmethod
    def _is_valid_coords(cls, coords: Any) -> bool:
        if isinstance(coords, (list, tuple)):
            if not coords:
                return False
            # Point: [x, y] or (x, y)
            if len(coords) >= 2 and all(not isinstance(v, (list, tuple)) for v in coords[:2]):
                return cls._is_valid_point(coords)
            # Polyline/segment: [[x1, y1], [x2, y2], ...]
            return all(cls._is_valid_point(pt) for pt in coords)
        return False

    @staticmethod
    def _part_field(item: Any, field_name: str, default=None):
        if isinstance(item, Mapping):
            return item.get(field_name, default)
        return getattr(item, field_name, default)

    @staticmethod
    def _normalize_materials(data: Any) -> dict[Any, Any]:
        if not isinstance(data, Mapping):
            return {}
        materials: dict[Any, Any] = {}
        max_serial = 0
        for key, value in data.items():
            try:
                serial = int(key)
            except Exception:
                serial = key
            if isinstance(value, Material):
                mat = copy.deepcopy(value)
                try:
                    mat.serial = int(getattr(mat, "serial", serial))
                except Exception:
                    mat.serial = serial
                mat.symmetry = normalize_material_symmetry(getattr(mat, "symmetry", "isotropic"))
                mat.behavior = normalize_material_behavior(
                    getattr(mat, "behavior", getattr(mat, "mat_type", "elastic"))
                )
                mat.damage = normalize_material_damage(getattr(mat, "damage", "none"))
                mat.properties = normalize_material_properties(
                    copy.deepcopy(getattr(mat, "properties", {}) or {}),
                    mat.behavior,
                    mat.symmetry,
                    mat.damage,
                )
            elif isinstance(value, Mapping):
                name = value.get("name", f"Material {serial}")
                mat_type = value.get("mat_type", value.get("type", "ELAS1"))
                properties = copy.deepcopy(value.get("properties", {}))
                mat = Material(
                    name,
                    mat_type,
                    properties,
                    symmetry=value.get("symmetry", "isotropic"),
                    behavior=value.get("behavior"),
                    damage=value.get("damage", "none"),
                )
                try:
                    mat.serial = int(value.get("serial", serial))
                except Exception:
                    mat.serial = serial
                mat.symmetry = normalize_material_symmetry(
                    value.get("symmetry", getattr(mat, "symmetry", "isotropic"))
                )
                mat.behavior = normalize_material_behavior(
                    value.get("behavior", getattr(mat, "behavior", getattr(mat, "mat_type", "elastic")))
                )
                mat.damage = normalize_material_damage(
                    value.get("damage", getattr(mat, "damage", "none"))
                )
                mat.properties = normalize_material_properties(
                    copy.deepcopy(value.get("properties", getattr(mat, "properties", {})) or {}),
                    mat.behavior,
                    mat.symmetry,
                    mat.damage,
                )
            else:
                continue
            materials[mat.serial] = mat
            try:
                max_serial = max(max_serial, int(mat.serial))
            except Exception:
                pass
        if max_serial > 0:
            Material._serial_counter = max(int(getattr(Material, "_serial_counter", 0)), max_serial)
        return materials

    @staticmethod
    def _normalize_parts(data: Any) -> list[Any]:
        if not isinstance(data, (list, tuple)):
            return []
        parts: list[Any] = []
        max_id = 0
        for item in data:
            if isinstance(item, Part):
                part = copy.deepcopy(item)
            elif isinstance(item, Mapping):
                geom = None
                geom_wkt = item.get("geometry_wkt")
                if geom_wkt:
                    try:
                        geom = wkt.loads(geom_wkt)
                    except Exception:
                        geom = None
                part = Part(
                    item.get("name", "Unnamed Part"),
                    geometry=geom,
                    is_void=bool(item.get("is_void", False)),
                )
                part.id = item.get("id", part.id)
                part.parent_id = item.get("parent_id")
                part.material_id = item.get("material_id")
                part.is_rigid = bool(item.get("is_rigid", False))
                part.is_direct_edit = bool(item.get("is_direct_edit", False))
                part.part_type = str(item.get("part_type", "void" if part.is_void else "solid") or "solid")
                part.material_assignment_mode = str(
                    item.get("material_assignment_mode", "homogeneous") or "homogeneous"
                )
                part.heterogeneity_method = str(item.get("heterogeneity_method", "region_based") or "region_based")
                part.heterogeneity_config = normalize_heterogeneity_config(
                    item.get("heterogeneity_config", {})
                )
                part.material_field_config = normalize_material_field_config(
                    item.get("material_field_config", {})
                )
                part.material_symmetry = normalize_material_symmetry(item.get("material_symmetry", "isotropic"))
                part.material_behavior = normalize_material_behavior(item.get("material_behavior", "elastic"))
                part.material_damage = normalize_material_damage(item.get("material_damage", "none"))
                part.particles = copy.deepcopy(item.get("particles", []))
                part.sketches = copy.deepcopy(item.get("sketches", []))
                part.sketch_meta = copy.deepcopy(item.get("sketch_meta", []))
                part.dimensions = copy.deepcopy(item.get("dimensions", []))
                part.constraints = copy.deepcopy(item.get("constraints", []))
                part.cad_source = copy.deepcopy(item.get("cad_source", None))
            else:
                continue
            part.material_assignment_mode = str(
                getattr(part, "material_assignment_mode", "homogeneous") or "homogeneous"
            )
            part.heterogeneity_method = str(
                getattr(part, "heterogeneity_method", "region_based") or "region_based"
            )
            part.heterogeneity_config = normalize_heterogeneity_config(
                copy.deepcopy(getattr(part, "heterogeneity_config", {}))
            )
            part.material_field_config = normalize_material_field_config(
                copy.deepcopy(getattr(part, "material_field_config", {}))
            )
            part.material_symmetry = normalize_material_symmetry(getattr(part, "material_symmetry", "isotropic"))
            part.material_behavior = normalize_material_behavior(getattr(part, "material_behavior", "elastic"))
            part.material_damage = normalize_material_damage(getattr(part, "material_damage", "none"))
            parts.append(part)
            try:
                max_id = max(max_id, int(getattr(part, "id", 0)))
            except Exception:
                pass
        if max_id > 0:
            Part._part_counter = max(int(getattr(Part, "_part_counter", 0)), max_id)
        return parts

    @staticmethod
    def _normalize_interfaces(data: Any) -> list[Any]:
        if not isinstance(data, (list, tuple)):
            return []
        interfaces: list[Any] = []
        max_id = 0
        for item in data:
            if isinstance(item, Interface):
                iface = copy.deepcopy(item)
            elif isinstance(item, Mapping):
                try:
                    iface = Interface(
                        item.get("part1_id"),
                        item.get("part2_id"),
                        item.get("interface_type", item.get("type", "GLUE")),
                    )
                    iface.id = int(item.get("id", iface.id))
                    iface.friction_coeff = float(item.get("friction_coeff", item.get("friction", getattr(iface, "friction_coeff", 0.0))))
                    iface.material_id = item.get("material_id")
                    iface.material_mode = item.get("material_mode", getattr(iface, "material_mode", "auto"))
                    iface.thickness = item.get("thickness")
                    iface.target_dx = item.get("target_dx")
                    iface.layer_mode = item.get("layer_mode", getattr(iface, "layer_mode", "single_layer_ring"))
                    iface.placement_mode = item.get(
                        "placement_mode",
                        getattr(iface, "placement_mode", getattr(Interface, "DEFAULT_PLACEMENT_MODE", "matrix_side")),
                    )
                    iface.status = item.get("status", "")
                    iface.notes = item.get("notes", "")
                except Exception:
                    continue
            else:
                continue
            interfaces.append(iface)
            try:
                max_id = max(max_id, int(getattr(iface, "id", 0)))
            except Exception:
                pass
        if max_id > 0:
            Interface._interface_counter = max(int(getattr(Interface, "_interface_counter", 0)), max_id)
            Interface._interaction_counter = max(int(getattr(Interface, "_interaction_counter", 0)), max_id)
        return interfaces

    def to_dict(self) -> dict[str, Any]:
        """Return a deep-copied dictionary snapshot."""
        return {
            "current_stage": copy.deepcopy(self.current_stage),
            "analysis_type": copy.deepcopy(self.analysis_type),
            "dimension": copy.deepcopy(self.dimension),
            "parts": copy.deepcopy(self.parts),
            "materials": copy.deepcopy(self.materials),
            "interfaces": copy.deepcopy(self.interfaces),
            "boundary_conditions": copy.deepcopy(self.boundary_conditions),
            "loads": copy.deepcopy(self.loads),
            "mesh_data": copy.deepcopy(self.mesh_data),
            "solver_settings": copy.deepcopy(self.solver_settings),
            "custom_mesh_zones": [
                z.to_dict() if hasattr(z, "to_dict") else copy.deepcopy(z)
                for z in (self.custom_mesh_zones or [])
            ],
            "edge_seeds": [
                s.to_dict() if hasattr(s, "to_dict") else copy.deepcopy(s)
                for s in (self.edge_seeds or [])
            ],
            "edge_seed_templates": [
                s.to_dict() if hasattr(s, "to_dict") else copy.deepcopy(s)
                for s in (self.edge_seed_templates or [])
            ],
            "vertex_seeds": [
                s.to_dict() if hasattr(s, "to_dict") else copy.deepcopy(s)
                for s in (self.vertex_seeds or [])
            ],
            "boundary_layer_seeds": [
                s.to_dict() if hasattr(s, "to_dict") else copy.deepcopy(s)
                for s in (self.boundary_layer_seeds or [])
            ],
            "matched_edge_pairs": [
                p.to_dict() if hasattr(p, "to_dict") else copy.deepcopy(p)
                for p in (self.matched_edge_pairs or [])
            ],
            "part_mesh_overrides": {
                int(pid): dict(v) for pid, v in (self.part_mesh_overrides or {}).items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ProjectState":
        """Build a ProjectState instance from a mapping."""
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise TypeError("ProjectState.from_dict expects a mapping or None.")
        return cls(
            current_stage=str(data.get("current_stage", "geometry")),
            analysis_type=str(data.get("analysis_type", "static") or "static").lower(),
            dimension=str(data.get("dimension", "2D") or "2D").upper(),
            parts=cls._normalize_parts(data.get("parts", [])),
            materials=cls._normalize_materials(data.get("materials", {})),
            interfaces=cls._normalize_interfaces(data.get("interfaces", [])),
            boundary_conditions=copy.deepcopy(
                data.get("boundary_conditions", data.get("bcs", []))
            ),
            loads=copy.deepcopy(data.get("loads", [])),
            mesh_data=copy.deepcopy(data.get("mesh_data", {})),
            solver_settings=copy.deepcopy(data.get("solver_settings", {})),
            custom_mesh_zones=cls._normalize_custom_mesh_zones(
                data.get("custom_mesh_zones", [])
            ),
            edge_seeds=cls._normalize_edge_seeds(
                data.get("edge_seeds", [])
            ),
            edge_seed_templates=cls._normalize_edge_seed_templates(
                data.get("edge_seed_templates", [])
            ),
            vertex_seeds=cls._normalize_vertex_seeds(
                data.get("vertex_seeds", [])
            ),
            boundary_layer_seeds=cls._normalize_boundary_layer_seeds(
                data.get("boundary_layer_seeds", [])
            ),
            matched_edge_pairs=cls._normalize_matched_pairs(
                data.get("matched_edge_pairs", [])
            ),
            part_mesh_overrides=cls._normalize_part_overrides(
                data.get("part_mesh_overrides", {})
            ),
        )

    @staticmethod
    def _normalize_matched_pairs(data: Any) -> list:
        if not isinstance(data, (list, tuple)):
            return []
        out: list = []
        for item in data:
            pair = (MatchedEdgePair.from_dict(item)
                    if not isinstance(item, MatchedEdgePair) else item)
            if pair is None or not pair.is_valid():
                continue
            out.append(pair)
        return out

    @staticmethod
    def _normalize_edge_seed_templates(data: Any) -> list:
        """Templates are EdgeSeed instances with no edge_refs — pure config."""
        if not isinstance(data, (list, tuple)):
            return []
        from models import EdgeSeed as _ES
        out: list = []
        for item in data:
            if isinstance(item, _ES):
                # accept directly; clear refs to mark as a template
                item.edge_refs = []
                out.append(item)
                continue
            if not isinstance(item, dict):
                continue
            # Strip edge_refs from the payload; templates are config-only.
            cfg = dict(item)
            cfg["edge_refs"] = []
            seed = _ES.from_dict(cfg)
            if seed is None:
                continue
            out.append(seed)
        return out

    @staticmethod
    def _normalize_boundary_layer_seeds(data: Any) -> list:
        if not isinstance(data, (list, tuple)):
            return []
        out: list = []
        for item in data:
            seed = (BoundaryLayerSeed.from_dict(item)
                    if not isinstance(item, BoundaryLayerSeed) else item)
            if seed is None or not seed.is_valid():
                continue
            out.append(seed)
        return out

    @staticmethod
    def _normalize_vertex_seeds(data: Any) -> list:
        if not isinstance(data, (list, tuple)):
            return []
        out: list = []
        for item in data:
            seed = VertexSeed.from_dict(item) if not isinstance(item, VertexSeed) else item
            if seed is None or not seed.is_valid():
                continue
            out.append(seed)
        return out

    @staticmethod
    def _normalize_part_overrides(data: Any) -> dict:
        if not isinstance(data, dict):
            return {}
        out: dict[int, dict] = {}
        for k, v in data.items():
            try:
                pid = int(k)
            except Exception:
                continue
            if not isinstance(v, dict):
                continue
            override = {}
            for key in ("h_bulk", "h_feature"):
                if key in v and v[key] not in (None, ""):
                    try:
                        val = float(v[key])
                        if val > 0:
                            override[key] = val
                    except Exception:
                        pass
            if override:
                out[pid] = override
        return out

    @staticmethod
    def _normalize_edge_seeds(data: Any) -> list:
        if not isinstance(data, (list, tuple)):
            return []
        out: list = []
        for item in data:
            seed = EdgeSeed.from_dict(item) if not isinstance(item, EdgeSeed) else item
            if seed is None or not getattr(seed, "edge_refs", None):
                continue
            out.append(seed)
        return out

    @staticmethod
    def _normalize_custom_mesh_zones(data: Any) -> list:
        if not isinstance(data, (list, tuple)):
            return []
        out: list = []
        for item in data:
            zone = CustomMeshZone.from_dict(item) if not isinstance(item, CustomMeshZone) else item
            if zone is None or not getattr(zone, "points", None):
                continue
            out.append(zone)
        return out

    def validate(self) -> tuple[bool, list[str]]:
        """Validate state and return (is_structurally_valid, warnings)."""
        warnings: list[str] = []
        is_valid = True

        if not isinstance(self.parts, list):
            warnings.append("parts must be a list.")
            is_valid = False
        if str(getattr(self, "analysis_type", "static")).lower() not in {"static", "dynamic", "fluid", "fsi"}:
            warnings.append("analysis_type must be 'static', 'dynamic', 'fluid', or 'fsi'.")
            is_valid = False
        if str(getattr(self, "dimension", "2D")).upper() not in {"2D", "3D"}:
            warnings.append("dimension must be '2D' or '3D'.")
            is_valid = False
        if not isinstance(self.materials, dict):
            warnings.append("materials must be a dict.")
            is_valid = False
        if not isinstance(self.interfaces, list):
            warnings.append("interfaces must be a list.")
            is_valid = False
        if not isinstance(self.boundary_conditions, list):
            warnings.append("boundary_conditions must be a list.")
            is_valid = False
        if not isinstance(self.loads, list):
            warnings.append("loads must be a list.")
            is_valid = False
        if not isinstance(self.mesh_data, dict):
            warnings.append("mesh_data must be a dict.")
            is_valid = False
        if not isinstance(self.solver_settings, dict):
            warnings.append("solver_settings must be a dict.")
            is_valid = False

        if not is_valid:
            return (False, warnings)

        material_ids = set(self.materials.keys())
        for key in list(material_ids):
            try:
                material_ids.add(int(key))
            except Exception:
                continue

        part_ids = set()
        for idx, part in enumerate(self.parts):
            part_id = self._part_field(part, "id")
            if part_id is not None:
                part_ids.add(part_id)
                try:
                    part_ids.add(int(part_id))
                except Exception:
                    pass
            is_void = bool(self._part_field(part, "is_void", False))
            if is_void:
                continue
            material_id = self._part_field(part, "material_id")
            part_name = self._part_field(part, "name", f"Part {idx + 1}")
            if material_id in (None, ""):
                warnings.append(f"Part '{part_name}' has no material assignment.")
                continue
            valid_material = material_id in material_ids
            if not valid_material:
                try:
                    valid_material = int(material_id) in material_ids
                except Exception:
                    valid_material = False
            if not valid_material:
                warnings.append(
                    f"Part '{part_name}' references missing material id '{material_id}'."
                )

        for idx, iface in enumerate(self.interfaces):
            p1 = self._part_field(iface, "part1_id")
            p2 = self._part_field(iface, "part2_id")
            if p1 is None or p2 is None:
                warnings.append(f"Interface #{idx + 1} is missing part references.")
                continue
            if part_ids:
                p1_ok = p1 in part_ids
                p2_ok = p2 in part_ids
                if not p1_ok:
                    try:
                        p1_ok = int(p1) in part_ids
                    except Exception:
                        p1_ok = False
                if not p2_ok:
                    try:
                        p2_ok = int(p2) in part_ids
                    except Exception:
                        p2_ok = False
                if not p1_ok or not p2_ok:
                    warnings.append(
                        f"Interface #{idx + 1} references missing part ids ({p1}, {p2})."
                    )

        def _validate_attr_entries(entries: list[Any], kind: str) -> None:
            for idx, entry in enumerate(entries):
                if not isinstance(entry, Mapping):
                    warnings.append(f"{kind} #{idx + 1} is not a valid mapping.")
                    continue
                part_id = entry.get("part_id")
                if part_id is not None and part_ids:
                    part_ok = part_id in part_ids
                    if not part_ok:
                        try:
                            part_ok = int(part_id) in part_ids
                        except Exception:
                            part_ok = False
                    if not part_ok:
                        warnings.append(
                            f"{kind} #{idx + 1} references missing part id '{part_id}'."
                        )

                ids = entry.get("ids")
                if ids is not None:
                    if not isinstance(ids, (list, tuple)) or not all(
                        isinstance(nid, numbers.Integral) and int(nid) >= 0 for nid in ids
                    ):
                        warnings.append(f"{kind} #{idx + 1} has invalid node ids.")

                coords = entry.get("coords")
                requires_coords = part_id is None and ids is None
                if requires_coords and not self._is_valid_coords(coords):
                    warnings.append(f"{kind} #{idx + 1} has invalid coordinates.")

        _validate_attr_entries(self.boundary_conditions, "BC")
        _validate_attr_entries(self.loads, "Load")

        return (True, warnings)
