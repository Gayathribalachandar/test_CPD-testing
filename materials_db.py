from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from models import Material


DEFAULT_CSV_PATH = Path(__file__).resolve().parent / "materials_data.csv"


NUMERIC_FIELDS = {
    "ultimate_tensile_strength",
    "yield_strength",
    "elongation_pct",
    "brinell_hb",
    "vickers_hv",
    "youngs_modulus",
    "shear_modulus",
    "poisson_ratio",
    "density_kg_m3",
    "critical_strain_energy_density",
}


@dataclass
class MaterialRecord:
    data: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.data.get("material_name", "")).strip()

    @property
    def category(self) -> str:
        return str(self.data.get("Material_category", "")).strip()

    @property
    def behavior_tag(self) -> str:
        return str(self.data.get("behavior_tag", "")).strip()

    @property
    def condition(self) -> str:
        return str(self.data.get("condition", "")).strip()


_CACHE: list[MaterialRecord] | None = None


def _parse_value(key: str, value: str) -> Any:
    if key in NUMERIC_FIELDS:
        try:
            return float(value)
        except Exception:
            return None
    return value


def load_csv(path: Path | None = None) -> list[MaterialRecord]:
    csv_path = Path(path or DEFAULT_CSV_PATH)
    records: list[MaterialRecord] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for key, value in (row or {}).items():
                if key is None:
                    continue
                parsed[key] = _parse_value(key, value)
            records.append(MaterialRecord(parsed))
    return records


def get_records(path: Path | None = None, *, use_cache: bool = True) -> list[MaterialRecord]:
    global _CACHE
    if use_cache and _CACHE is not None:
        return _CACHE
    _CACHE = load_csv(path)
    return _CACHE


def clear_cache() -> None:
    global _CACHE
    _CACHE = None


def _norm(text: Any) -> str:
    return str(text or "").strip().lower()


def filter_records(
    records: Iterable[MaterialRecord],
    *,
    name: str | None = None,
    category: str | None = None,
    behavior_tag: str | None = None,
    condition: str | None = None,
) -> list[MaterialRecord]:
    name_q = _norm(name)
    category_q = _norm(category)
    behavior_q = _norm(behavior_tag)
    condition_q = _norm(condition)

    out: list[MaterialRecord] = []
    for rec in records:
        if name_q and name_q not in _norm(rec.name):
            continue
        if category_q and category_q != _norm(rec.category):
            continue
        if behavior_q and behavior_q != _norm(rec.behavior_tag):
            continue
        if condition_q and condition_q != _norm(rec.condition):
            continue
        out.append(rec)
    return out


def record_to_material(record: MaterialRecord) -> Material:
    data = record.data
    name = str(data.get("material_name") or "Material")

    properties = {}
    if data.get("density_kg_m3") is not None:
        properties["density"] = float(data["density_kg_m3"])
    if data.get("youngs_modulus") is not None:
        properties["youngs_modulus"] = float(data["youngs_modulus"])
    if data.get("shear_modulus") is not None:
        properties["shear_modulus"] = float(data["shear_modulus"])
    if data.get("poisson_ratio") is not None:
        properties["poisson_ratio"] = float(data["poisson_ratio"])
    if data.get("yield_strength") is not None:
        properties["yield_stress"] = float(data["yield_strength"])
    if data.get("ultimate_tensile_strength") is not None:
        properties["ultimate_strength"] = float(data["ultimate_tensile_strength"])
    if data.get("critical_strain_energy_density") is not None:
        properties["critical_strain_energy_density"] = float(data["critical_strain_energy_density"])

    mat = Material(
        name=name,
        mat_type="ELAS1",
        properties=properties,
        symmetry="isotropic",
        behavior="elastic",
        damage="none",
    )

    # Attach metadata for filtering/provenance (optional for now).
    metadata = {
        "behavior_tag": data.get("behavior_tag"),
        "standard": data.get("standard"),
        "Material_category": data.get("Material_category"),
        "condition": data.get("condition"),
        "description": data.get("description"),
    }
    setattr(mat, "metadata", metadata)

    return mat
