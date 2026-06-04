from __future__ import annotations

import copy

from PySide6.QtWidgets import QMessageBox

from project_stages import ProjectStage
from .commands import AddBoundaryConditionCommand


class BCController:
    workflow_key = "bc"
    label = "Boundary Conditions"
    stage = ProjectStage.BCS
    show_sketch = False
    show_geometry = False
    show_loads = True

    def __init__(self, window):
        self.window = window

    def register_command_handlers(self, command_bus):
        command_bus.register_handler(AddBoundaryConditionCommand, self.handle_add_boundary_condition_command)

    def workflow_definition(self, tab, *, workflow_key=None, label=None, stage=None):
        return {
            "workflow_key": workflow_key or self.workflow_key,
            "label": label or self.label,
            "stage": stage or self.stage,
            "tab": tab,
            "show_sketch": self.show_sketch,
            "show_geometry": self.show_geometry,
            "show_loads": self.show_loads,
        }

    def hint_for(self, stage=None):
        if stage == ProjectStage.BCS:
            return (
                "Define boundary conditions. 2D: right-click vertex/edge or selected part to "
                "apply BCs. 3D: choose Face/Edge/Point and apply from BC panel."
            )
        return (
            "Define loads. 2D: right-click vertex/edge or selected part to apply loads. 3D: "
            "choose Face/Edge/Point and apply from Loads panel."
        )

    def _sync_attr_state(self, *, emit_bcs=False, emit_loads=False, save_velocity=False):
        view = self.window.view
        view.bcs = copy.deepcopy(self.window.project_state.boundary_conditions)
        view.loads = copy.deepcopy(self.window.project_state.loads)
        if hasattr(view, "_sanitize_bc_load_entries"):
            try:
                view._sanitize_bc_load_entries()
            except Exception:
                pass
        if save_velocity:
            view.save_velocity_csv()
        if emit_bcs:
            view.bcsChanged.emit()
        if emit_loads:
            view.loadsChanged.emit()

    def add_bc_entries(self, entries, *, save_velocity=True):
        if not entries:
            return []
        self.window.view.push_undo_state()
        self.window.project_state.boundary_conditions.extend(copy.deepcopy(list(entries)))
        self._sync_attr_state(emit_bcs=True, save_velocity=save_velocity)
        return self.window.project_state.boundary_conditions

    def add_load_entries(self, entries):
        if not entries:
            return []
        self.window.view.push_undo_state()
        self.window.project_state.loads.extend(copy.deepcopy(list(entries)))
        self._sync_attr_state(emit_loads=True)
        return self.window.project_state.loads

    def handle_add_boundary_condition_command(self, command):
        if not isinstance(command, AddBoundaryConditionCommand):
            raise TypeError("Unsupported command for BCController")
        if command.entry_kind == "load":
            entries = self.add_load_entries(command.entries)
        else:
            entries = self.add_bc_entries(command.entries, save_velocity=bool(command.save_velocity))
        self.window.event_bus.publish(
            "boundary_condition.added",
            {
                "entry_kind": command.entry_kind,
                "entries": list(command.entries),
                "result": entries,
            },
        )
        return entries

    def update_bc(self, index, updater=None, *, save_velocity=True):
        if not isinstance(index, int) or not (0 <= index < len(self.window.project_state.boundary_conditions)):
            return None
        self.window.view.push_undo_state()
        entry = self.window.project_state.boundary_conditions[index]
        if callable(updater):
            updater(entry)
        elif updater is not None:
            self.window.project_state.boundary_conditions[index] = copy.deepcopy(updater)
            entry = self.window.project_state.boundary_conditions[index]
        self._sync_attr_state(emit_bcs=True, save_velocity=save_velocity)
        return entry

    def update_load(self, index, updater=None):
        if not isinstance(index, int) or not (0 <= index < len(self.window.project_state.loads)):
            return None
        self.window.view.push_undo_state()
        entry = self.window.project_state.loads[index]
        if callable(updater):
            updater(entry)
        elif updater is not None:
            self.window.project_state.loads[index] = copy.deepcopy(updater)
            entry = self.window.project_state.loads[index]
        self._sync_attr_state(emit_loads=True)
        return entry

    def delete_bc(self, index, *, save_velocity=True):
        if not isinstance(index, int) or not (0 <= index < len(self.window.project_state.boundary_conditions)):
            return False
        self.window.view.push_undo_state()
        del self.window.project_state.boundary_conditions[index]
        self._sync_attr_state(emit_bcs=True, save_velocity=save_velocity)
        return True

    def delete_load(self, index):
        if not isinstance(index, int) or not (0 <= index < len(self.window.project_state.loads)):
            return False
        self.window.view.push_undo_state()
        del self.window.project_state.loads[index]
        self._sync_attr_state(emit_loads=True)
        return True

    def apply_bc_from_toolbar(self):
        panel = getattr(self.window, "properties_panel", None)
        if panel is None or not hasattr(panel, "bcs_tab"):
            return
        panel.tabs.setCurrentWidget(panel.bcs_tab)
        panel.bcs_tab._apply_bc_from_selection()

    def apply_load_from_toolbar(self):
        panel = getattr(self.window, "properties_panel", None)
        if panel is None or not hasattr(panel, "loads_tab"):
            return
        panel.tabs.setCurrentWidget(panel.loads_tab)
        panel.loads_tab._apply_bc_from_selection()

    def validate_before_solve(self):
        if self.window.view.bcs or self.window.view.loads:
            return True
        QMessageBox.warning(
            self.window,
            "No Boundary Conditions",
            "Define at least one boundary condition or load before solving.",
        )
        return False
