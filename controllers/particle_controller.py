from __future__ import annotations

from PySide6.QtWidgets import QMessageBox

from project_stages import ProjectStage
from .commands import GenerateParticlesCommand


class ParticleController:
    workflow_key = "particles"
    label = "Particles"
    stage = ProjectStage.MESH
    show_sketch = False
    show_geometry = False
    show_loads = False
    hint = "Generate particles, then generate connections. Connections reuse the current particles so nodes stay consistent."

    def __init__(self, window):
        self.window = window

    def register_command_handlers(self, command_bus):
        command_bus.register_handler(GenerateParticlesCommand, self.handle_generate_particles_command)

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

    def generate_particles(self):
        return self.window._generate_gmsh_mesh_impl()

    def handle_generate_particles_command(self, command):
        if not isinstance(command, GenerateParticlesCommand):
            raise TypeError("Unsupported command for ParticleController")
        result = self.window._generate_particles_from_command_impl(command.mesh_config)
        self.window.event_bus.publish(
            "particles.generated",
            {
                "mesh_config": dict(command.mesh_config),
                "result": result,
            },
        )
        return result

    def validate_before_solve(self):
        global_nodes = getattr(self.window.view, "global_nodes", None)
        global_elements = getattr(self.window.view, "global_elements", None)
        connections_ready = (
            global_nodes is not None
            and global_elements is not None
            and len(global_nodes) > 0
            and len(global_elements) > 0
        )
        if connections_ready:
            return True
        QMessageBox.warning(
            self.window,
            "Connections Missing",
            "Generate particles and then click Generate Connections in the Particles stage before solving.",
        )
        return False
