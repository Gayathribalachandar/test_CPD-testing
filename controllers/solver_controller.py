from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QMessageBox

from job_manager import JobManager
from project_stages import ProjectStage
from .commands import RunSolverCommand


class SolverController:
    workflow_key = "solve"
    label = "Solve"
    stage = ProjectStage.JOB
    show_sketch = False
    show_geometry = False
    show_loads = False
    hint = "Run the solver and visualize results."

    def __init__(self, window):
        self.window = window
        self.job_manager = JobManager(window)
        self.job_manager.jobFinished.connect(self._on_job_finished)

    def register_command_handlers(self, command_bus):
        command_bus.register_handler(RunSolverCommand, self.handle_run_solver_command)

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

    def collect_solver_settings_for_state(self):
        return self.window._collect_solver_settings_for_state_impl()

    def sync_solver_settings_to_project_state(self):
        return self.window._sync_solver_settings_to_project_state_impl()

    def apply_solver_settings_to_ui(self):
        return self.window._apply_solver_settings_to_ui_impl()

    def export_job_artifacts(self, project_file, export_inputs=True, include_results=True):
        return self.window._export_job_artifacts_impl(
            project_file,
            export_inputs=export_inputs,
            include_results=include_results,
        )

    def run_solver_job(self, *, config_path, workspace_folder, total_steps=0):
        if self.job_manager.track_current_job() is not None:
            QMessageBox.information(
                self.window,
                "Simulation Running",
                "A simulation job is already running. Cancel it before starting another one.",
            )
            return None
        view = getattr(self.window, "view", None)
        if view is not None and hasattr(view, "release_results_file_handles"):
            try:
                view.release_results_file_handles()
            except Exception:
                pass
        run_script = Path(__file__).resolve().parents[1] / "CPD-main" / "source" / "run_cpd.py"
        if not run_script.exists():
            QMessageBox.warning(
                self.window,
                "Solver Missing",
                f"Solver entrypoint not found:\n{run_script}",
            )
            return None
        return self.job_manager.run_job(
            config_path=config_path,
            workspace_folder=workspace_folder,
            command=[sys.executable, str(run_script)],
            cwd=str(run_script.parent),
            env_updates={"CPD_WORKSPACE_DIR": str(workspace_folder)},
            total_steps=total_steps,
        )

    def handle_run_solver_command(self, command):
        if not isinstance(command, RunSolverCommand):
            raise TypeError("Unsupported command for SolverController")
        result = self.window._run_solver_command_impl()
        self.window.event_bus.publish("solver.run.dispatched", {"result": result})
        return result

    def cancel_job(self):
        self.job_manager.cancel_job()

    def current_job(self):
        return self.job_manager.track_current_job()

    def stop(self):
        self.job_manager.stop()

    def _on_job_finished(self, job):
        if getattr(job, "status", "") != "completed":
            return
        try:
            self.window.view.load_and_run_visualization()
        except Exception:
            pass
        try:
            if self.window.active_stage.value < ProjectStage.RESULTS.value:
                self.window.view.stageAdvanceRequested.emit(ProjectStage.RESULTS)
        except Exception:
            pass
