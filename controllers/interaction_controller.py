from __future__ import annotations

from project_stages import ProjectStage


class InteractionController:
    workflow_key = "interactions"
    label = "Interactions"
    stage = ProjectStage.INTERFACES
    show_sketch = False
    show_geometry = False
    show_loads = False
    hint = "Define interactions between neighboring materials and particle regions."

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
