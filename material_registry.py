from __future__ import annotations

import copy
from typing import Any


SYMMETRY_OPTIONS = [
    ("Isotropic", "isotropic"),
    ("Orthotropic", "orthotropic"),
    ("Anisotropic", "anisotropic"),
]


MATERIAL_PARAMETER_DEFINITIONS: dict[str, dict[str, Any]] = {
    "density": {"label": "Density", "default": 1.0, "minimum": 0.0, "maximum": 1e12, "step": 1.0},
    "youngs_modulus": {
        "label": "Young's Modulus",
        "default": 1.0,
        "minimum": 0.0,
        "maximum": 1e15,
        "step": 1e6,
    },
    "youngs_modulus_x": {
        "label": "Young's Modulus X",
        "default": 1.0,
        "minimum": 0.0,
        "maximum": 1e15,
        "step": 1e6,
    },
    "youngs_modulus_y": {
        "label": "Young's Modulus Y",
        "default": 1.0,
        "minimum": 0.0,
        "maximum": 1e15,
        "step": 1e6,
    },
    "youngs_modulus_z": {
        "label": "Young's Modulus Z",
        "default": 1.0,
        "minimum": 0.0,
        "maximum": 1e15,
        "step": 1e6,
    },
    "poisson_ratio": {
        "label": "Poisson Ratio",
        "default": 0.3,
        "minimum": -1.0,
        "maximum": 0.5,
        "step": 0.01,
    },
    "poisson_ratio_xy": {
        "label": "Poisson Ratio XY",
        "default": 0.3,
        "minimum": -1.0,
        "maximum": 0.5,
        "step": 0.01,
    },
    "poisson_ratio_yz": {
        "label": "Poisson Ratio YZ",
        "default": 0.3,
        "minimum": -1.0,
        "maximum": 0.5,
        "step": 0.01,
    },
    "poisson_ratio_xz": {
        "label": "Poisson Ratio XZ",
        "default": 0.3,
        "minimum": -1.0,
        "maximum": 0.5,
        "step": 0.01,
    },
    "shear_modulus": {"label": "Shear Modulus", "default": 1.0, "minimum": 0.0, "maximum": 1e15, "step": 1e6},
    "shear_modulus_xy": {
        "label": "Shear Modulus XY",
        "default": 1.0,
        "minimum": 0.0,
        "maximum": 1e15,
        "step": 1e6,
    },
    "shear_modulus_yz": {
        "label": "Shear Modulus YZ",
        "default": 1.0,
        "minimum": 0.0,
        "maximum": 1e15,
        "step": 1e6,
    },
    "shear_modulus_xz": {
        "label": "Shear Modulus XZ",
        "default": 1.0,
        "minimum": 0.0,
        "maximum": 1e15,
        "step": 1e6,
    },
    "bulk_modulus": {"label": "Bulk Modulus", "default": 1.0, "minimum": 0.0, "maximum": 1e15, "step": 1e6},
    "yield_stress": {"label": "Yield Stress", "default": 1.0, "minimum": 0.0, "maximum": 1e12, "step": 1e6},
    "hardening_rate": {
        "label": "Hardening Rate",
        "default": 1.0,
        "minimum": 0.0,
        "maximum": 1e12,
        "step": 1e4,
    },
    "damping": {"label": "Damping", "default": 0.0, "minimum": 0.0, "maximum": 1e6, "step": 0.01},
    "failure_energy": {
        "label": "Failure Energy",
        "default": 1.0,
        "minimum": 0.0,
        "maximum": 1e15,
        "step": 1e6,
    },
    "cohesive_strength": {
        "label": "Cohesive Strength",
        "default": 1.0,
        "minimum": 0.0,
        "maximum": 1e12,
        "step": 1e6,
    },
}


MATERIAL_MODELS: dict[str, dict[str, Any]] = {
    "elastic": {
        "label": "Elastic",
        "legacy_mat_type": "ELAS1",
        "parameters": {
            "isotropic": ["density", "youngs_modulus", "poisson_ratio"],
            "orthotropic": [
                "density",
                "youngs_modulus_x",
                "youngs_modulus_y",
                "poisson_ratio_xy",
                "shear_modulus_xy",
            ],
            "anisotropic": [
                "density",
                "youngs_modulus_x",
                "youngs_modulus_y",
                "youngs_modulus_z",
                "poisson_ratio_xy",
                "poisson_ratio_yz",
                "poisson_ratio_xz",
                "shear_modulus_xy",
                "shear_modulus_yz",
                "shear_modulus_xz",
            ],
        },
    },
    "plastic": {
        "label": "Plastic",
        "legacy_mat_type": "VISCOPLASTIC",
        "parameters": {
            "isotropic": [
                "density",
                "youngs_modulus",
                "poisson_ratio",
                "yield_stress",
                "hardening_rate",
            ],
            "orthotropic": [
                "density",
                "youngs_modulus_x",
                "youngs_modulus_y",
                "poisson_ratio_xy",
                "shear_modulus_xy",
                "yield_stress",
                "hardening_rate",
            ],
            "anisotropic": [
                "density",
                "youngs_modulus_x",
                "youngs_modulus_y",
                "youngs_modulus_z",
                "poisson_ratio_xy",
                "poisson_ratio_yz",
                "poisson_ratio_xz",
                "shear_modulus_xy",
                "shear_modulus_yz",
                "shear_modulus_xz",
                "yield_stress",
                "hardening_rate",
            ],
        },
    },
    "hyperelastic": {
        "label": "Hyperelastic",
        "legacy_mat_type": "NEOHOOK",
        "parameters": {
            "default": ["density", "shear_modulus", "bulk_modulus"],
        },
    },
    "viscoelastic": {
        "label": "Viscoelastic",
        "legacy_mat_type": "ELAS1",
        "parameters": {
            "isotropic": ["density", "youngs_modulus", "poisson_ratio", "damping"],
            "orthotropic": [
                "density",
                "youngs_modulus_x",
                "youngs_modulus_y",
                "poisson_ratio_xy",
                "shear_modulus_xy",
                "damping",
            ],
            "anisotropic": [
                "density",
                "youngs_modulus_x",
                "youngs_modulus_y",
                "youngs_modulus_z",
                "poisson_ratio_xy",
                "poisson_ratio_yz",
                "poisson_ratio_xz",
                "shear_modulus_xy",
                "shear_modulus_yz",
                "shear_modulus_xz",
                "damping",
            ],
        },
    },
    "rigid": {
        "label": "Rigid",
        "legacy_mat_type": "RIGID",
        "parameters": {
            "default": ["density"],
        },
    },
}


MATERIAL_DAMAGE_MODELS: dict[str, dict[str, Any]] = {
    "none": {"label": "None", "parameters": []},
    "fracture": {"label": "Fracture", "parameters": ["failure_energy"]},
    "cohesive": {"label": "Cohesive", "parameters": ["cohesive_strength", "failure_energy"]},
}


def normalize_material_symmetry(value: Any) -> str:
    symmetry = str(value or "isotropic").lower()
    if symmetry not in {"isotropic", "orthotropic", "anisotropic"}:
        return "isotropic"
    return symmetry


def normalize_material_behavior(value: Any) -> str:
    behavior = str(value or "elastic").lower()
    if behavior not in MATERIAL_MODELS:
        return "elastic"
    return behavior


def normalize_material_damage(value: Any) -> str:
    damage = str(value or "none").lower()
    if damage not in MATERIAL_DAMAGE_MODELS:
        return "none"
    return damage


def infer_behavior_from_mat_type(mat_type: Any) -> str:
    mat_type = str(mat_type or "").upper()
    for behavior, spec in MATERIAL_MODELS.items():
        if str(spec.get("legacy_mat_type", "")).upper() == mat_type:
            return behavior
    if mat_type == "ELAS2":
        return "elastic"
    return "elastic"


def legacy_mat_type_for_behavior(behavior: Any, current_mat_type: Any = None) -> str:
    behavior_key = normalize_material_behavior(behavior)
    current = str(current_mat_type or "").upper()
    if behavior_key == "elastic" and current in {"ELAS1", "ELAS2"}:
        return current
    return str(MATERIAL_MODELS.get(behavior_key, {}).get("legacy_mat_type", "ELAS1"))


def material_behavior_options() -> list[tuple[str, str]]:
    return [(spec.get("label", key.title()), key) for key, spec in MATERIAL_MODELS.items()]


def material_damage_options() -> list[tuple[str, str]]:
    return [(spec.get("label", key.title()), key) for key, spec in MATERIAL_DAMAGE_MODELS.items()]


def material_symmetry_options() -> list[tuple[str, str]]:
    return list(SYMMETRY_OPTIONS)


def behavior_label(value: Any) -> str:
    key = normalize_material_behavior(value)
    return str(MATERIAL_MODELS.get(key, {}).get("label", key.title()))


def damage_label(value: Any) -> str:
    key = normalize_material_damage(value)
    return str(MATERIAL_DAMAGE_MODELS.get(key, {}).get("label", key.title()))


def parameter_definition(key: str) -> dict[str, Any]:
    spec = copy.deepcopy(MATERIAL_PARAMETER_DEFINITIONS.get(str(key), {}))
    if not spec:
        spec = {"label": str(key).replace("_", " ").title(), "default": 0.0}
    return spec


def parameter_label(key: str) -> str:
    return str(parameter_definition(key).get("label", str(key).replace("_", " ").title()))


def default_parameter_value(key: str) -> float:
    return float(parameter_definition(key).get("default", 0.0))


def _unique_ordered(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def material_parameter_keys(
    behavior: Any,
    symmetry: Any = "isotropic",
    damage: Any = "none",
) -> list[str]:
    behavior_key = normalize_material_behavior(behavior)
    symmetry_key = normalize_material_symmetry(symmetry)
    damage_key = normalize_material_damage(damage)
    model_spec = MATERIAL_MODELS.get(behavior_key, {})
    param_map = model_spec.get("parameters", {}) or {}
    params = list(param_map.get(symmetry_key) or param_map.get("default") or param_map.get("isotropic") or [])
    damage_params = list((MATERIAL_DAMAGE_MODELS.get(damage_key, {}) or {}).get("parameters", []) or [])
    return _unique_ordered(params + damage_params)


def normalize_material_properties(
    properties: Any,
    behavior: Any,
    symmetry: Any = "isotropic",
    damage: Any = "none",
    *,
    preserve_unknown: bool = True,
) -> dict[str, Any]:
    src = dict(properties or {}) if isinstance(properties, dict) else {}
    keys = material_parameter_keys(behavior, symmetry, damage)
    normalized = {
        key: src.get(key, default_parameter_value(key))
        for key in keys
    }
    if preserve_unknown:
        for key, value in src.items():
            if key not in normalized:
                normalized[key] = value
    return normalized


def material_property_schema(
    behavior: Any,
    symmetry: Any = "isotropic",
    damage: Any = "none",
) -> list[dict[str, Any]]:
    schema: list[dict[str, Any]] = []
    for key in material_parameter_keys(behavior, symmetry, damage):
        spec = parameter_definition(key)
        schema.append(
            {
                "name": spec.get("label", key.replace("_", " ").title()),
                "key": key,
                "type": "float",
                "minimum": spec.get("minimum", -1e12),
                "maximum": spec.get("maximum", 1e12),
                "decimals": spec.get("decimals", 6),
                "step": spec.get("step", 1.0),
                "default": spec.get("default", 0.0),
                "tooltip": spec.get("tooltip", ""),
            }
        )
    return schema


def all_registry_parameter_keys() -> list[str]:
    keys: list[str] = []
    for behavior in MATERIAL_MODELS:
        for symmetry in ("isotropic", "orthotropic", "anisotropic"):
            keys.extend(material_parameter_keys(behavior, symmetry, "none"))
            keys.extend(material_parameter_keys(behavior, symmetry, "fracture"))
            keys.extend(material_parameter_keys(behavior, symmetry, "cohesive"))
    return _unique_ordered(keys)
