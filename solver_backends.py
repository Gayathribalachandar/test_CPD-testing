from __future__ import annotations

import os
import sys
from typing import Dict, List, Sequence, Tuple

from plugin_manager import register_solver_backend
from app_config import get_workspace_dir


class BaseSolverBackend:
    name = "Base Solver"
    description = """Abstract solver backend interface."""

    def is_available(self, project_dir: str) -> bool:
        return True

    def prepare_run(
        self, project_dir: str, settings: Dict[str, object]
    ) -> Tuple[Sequence[str], str, Dict[str, object]]:
        """Return (command, cwd, env_updates)."""
        raise NotImplementedError


@register_solver_backend
class CpdSolverBackend(BaseSolverBackend):
    name = "CPD"  # used for menu/toolbar labels
    description = "Integrated CPD-main Python solver."

    def is_available(self, project_dir: str) -> bool:
        cpd_main = os.path.join(project_dir, "CPD-main")
        return os.path.isdir(cpd_main)

    def prepare_run(
        self, project_dir: str, settings: Dict[str, object]
    ) -> Tuple[Sequence[str], str, Dict[str, object]]:
        project_dir = os.path.abspath(project_dir)
        cpd_main_dir = os.path.join(project_dir, "CPD-main")
        run_script = os.path.join(cpd_main_dir, "source", "run_cpd.py")
        command = [sys.executable, "-u", run_script]
        env = {"PYTHONUNBUFFERED": "1"}
        env["CPD_WORKSPACE_DIR"] = str(get_workspace_dir())
        if settings.get("time_step") is not None:
            env["CPD_TIME_STEP"] = str(settings["time_step"])
        if settings.get("total_steps") is not None:
            env["CPD_TOTAL_STEPS"] = str(settings["total_steps"])
        dt = settings.get("device")
        if dt is not None:
            env["CPD_DEVICE"] = str(dt)
        g_val = settings.get("g")
        if g_val is not None:
            env["CPD_G"] = str(g_val)
        return command, cpd_main_dir, env


def get_solver_backends() -> List[BaseSolverBackend]:
    from plugin_manager import get_solver_backend_classes

    return [cls() for cls in get_solver_backend_classes()]


def get_available_backends(project_dir: str) -> List[BaseSolverBackend]:
    return [b for b in get_solver_backends() if b.is_available(project_dir)]


def get_default_backend(project_dir: str) -> BaseSolverBackend | None:
    backends = get_available_backends(project_dir)
    if not backends:
        return None
    return backends[0]
