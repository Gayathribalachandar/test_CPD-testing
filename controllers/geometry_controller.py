from __future__ import annotations

from PySide6.QtWidgets import QMessageBox

from project_stages import ProjectStage


class GeometryController:
    workflow_key = "geometry"
    label = "Geometry"
    stage = ProjectStage.GEOMETRY
    show_sketch = True
    show_geometry = False
    show_loads = False
    hint = (
        "Use the Geometry stage panel to create parts, confirm solids, and cut holes. "
        "In 3D mode use 2.5D Features for add/cut/intersect."
    )

    def __init__(self, window):
        self.window = window

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

    def finish_stage(self):
        if not self.window.view.parts:
            QMessageBox.warning(
                self.window,
                "No Geometry",
                "Create at least one part before proceeding.",
            )
            return
        self.window.advance_to_next_stage()
