from __future__ import annotations

from PySide6.QtWidgets import QMessageBox

from models import Material
from material_registry import (
    legacy_mat_type_for_behavior,
    normalize_material_behavior,
    normalize_material_damage,
    normalize_material_properties,
    normalize_material_symmetry,
)
from project_stages import ProjectStage
from .commands import AddMaterialCommand


class MaterialController:
    workflow_key = "materials"
    label = "Materials"
    stage = ProjectStage.MATERIALS
    show_sketch = False
    show_geometry = False
    show_loads = False
    hint = "Left-click a part, right-click to assign a material."

    def __init__(self, window):
        self.window = window

    def register_command_handlers(self, command_bus):
        command_bus.register_handler(AddMaterialCommand, self.handle_add_material_command)

    def workflow_definition(self, tab):
        return {
            "label": self.label,
            "stage": self.stage,
            "tab": tab,
            "show_sketch": self.show_sketch,
            "show_geometry": self.show_geometry,
            "show_loads": self.show_loads,
        }

    def hint_for(self, _stage=None):
        return self.hint

    def materials_store(self):
        return self.window.project_state.materials

    def sync_materials_to_view(self):
        try:
            self.window.view.deserialize_materials(self.materials_store())
        except Exception:
            pass

    def upsert_material(self, name, mat_type, properties, symmetry="isotropic", behavior="elastic", damage="none"):
        view = self.window.view
        store = self.materials_store()
        view.push_undo_state()
        existing_mat = next(
            (m for m in store.values() if str(getattr(m, "name", "")).lower() == str(name).lower()),
            None,
        )
        created = existing_mat is None
        if existing_mat is not None:
            existing_mat.name = name
            existing_mat.behavior = normalize_material_behavior(behavior)
            existing_mat.damage = normalize_material_damage(damage)
            existing_mat.symmetry = normalize_material_symmetry(symmetry)
            existing_mat.mat_type = legacy_mat_type_for_behavior(existing_mat.behavior, mat_type)
            existing_mat.properties = normalize_material_properties(
                dict(properties),
                existing_mat.behavior,
                existing_mat.symmetry,
                existing_mat.damage,
                preserve_unknown=False,
            )
            mat = existing_mat
        else:
            mat = Material(
                name,
                mat_type,
                dict(properties),
                symmetry=symmetry,
                behavior=behavior,
                damage=damage,
            )
            store[mat.serial] = mat
        self.sync_materials_to_view()
        return mat, created

    def handle_add_material_command(self, command):
        if not isinstance(command, AddMaterialCommand):
            raise TypeError("Unsupported command for MaterialController")
        mat, created = self.upsert_material(
            command.name,
            command.mat_type,
            command.properties,
            symmetry=command.symmetry,
            behavior=command.behavior,
            damage=command.damage,
        )
        assigned = False
        part = None
        if command.auto_assign_selected_part and command.selected_part_id is not None:
            part = next(
                (entry for entry in self.window.project_state.parts if entry.id == command.selected_part_id),
                None,
            )
            if part is not None and not getattr(part, "is_void", False):
                assigned = bool(
                    self.assign_material_to_part(
                        part,
                        mat.serial,
                        announce=bool(command.announce_assignment),
                    )
                )
        self.window.event_bus.publish(
            "material.upserted",
            {
                "material": mat,
                "created": created,
                "assigned": assigned,
                "part": part,
            },
        )
        return {
            "material": mat,
            "created": created,
            "assigned": assigned,
            "part": part,
        }

    def assign_material_to_part(
        self,
        part,
        material_id,
        announce=False,
        assignment_mode=None,
        heterogeneity_method=None,
        heterogeneity_config=None,
        material_field_config=None,
        symmetry=None,
        behavior=None,
        damage=None,
    ):
        self.sync_materials_to_view()
        target_part = part
        try:
            part_id = int(getattr(part, "id"))
        except Exception:
            part_id = getattr(part, "id", None)
        if part_id is not None:
            target_part = next(
                (entry for entry in self.window.project_state.parts if getattr(entry, "id", None) == part_id),
                part,
            )
        return self.window.view.assign_material_to_part(
            target_part,
            material_id,
            announce=announce,
            assignment_mode=assignment_mode,
            heterogeneity_method=heterogeneity_method,
            heterogeneity_config=heterogeneity_config,
            material_field_config=material_field_config,
            symmetry=symmetry,
            behavior=behavior,
            damage=damage,
        )

    def validate_before_bc_stage(self):
        missing = [
            part.name for part in self.window.view.parts
            if not part.is_void and part.material_id is None
        ]
        if not missing:
            return True
        QMessageBox.warning(
            self.window,
            "Materials Missing",
            "Assign materials to all solid parts before continuing:\n\n" + "\n".join(missing),
        )
        return False
