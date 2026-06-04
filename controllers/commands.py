from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .command_bus import BaseCommand


@dataclass(slots=True)
class AddBoundaryConditionCommand(BaseCommand):
    entries: list[dict[str, Any]] = field(default_factory=list)
    entry_kind: str = "bc"
    save_velocity: bool = True


@dataclass(slots=True)
class AddMaterialCommand(BaseCommand):
    name: str = ""
    mat_type: str = ""
    behavior: str = "elastic"
    damage: str = "none"
    symmetry: str = "isotropic"
    properties: dict[str, Any] = field(default_factory=dict)
    selected_part_id: int | None = None
    auto_assign_selected_part: bool = True
    announce_assignment: bool = False


@dataclass(slots=True)
class GenerateParticlesCommand(BaseCommand):
    mesh_config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunSolverCommand(BaseCommand):
    pass
