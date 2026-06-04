# This is appconfig
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

APP_NAME = "CPD-SimStudio"
__version__ = "0.1.0"
WORKSPACE_DIR_NAME = "workspace"
AUTOSAVE_DIR_NAME = "autosave"
RECENT_PROJECTS_FILE = "recent_projects.json"
UPDATE_REPO = os.environ.get("CPD_UPDATE_REPO", "").strip()
UPDATE_CHECK_ON_STARTUP = os.environ.get("CPD_UPDATE_CHECK_ON_STARTUP", "1").strip() not in {
    "0",
    "false",
    "False",
}
try:
    UPDATE_REQUEST_TIMEOUT_SEC = float(os.environ.get("CPD_UPDATE_TIMEOUT_SEC", "4.0"))
except Exception:
    UPDATE_REQUEST_TIMEOUT_SEC = 4.0

SCENE_W = 2000
SCENE_H = 2000
SCENE_EXTENT = 1.0e6
VIEW_W = 1200
VIEW_H = 800
GRID_MINOR = 25
GRID_MAJOR = 100
DEFAULT_DX = 5.0
MESH_MIN_SPACING_FACTOR = 1.0
MESH_NODE_SOFT_LIMIT = 250_000
MESH_NODE_HARD_LIMIT = 3_000_000
PREVIEW_CONNECTION_LIMIT = 10_000_000
FAST_PREVIEW_CONNECTION_LIMIT = 2_000
FAST_PREVIEW_ENABLED = False
GPU_POINT_PREVIEW_ENABLED = True
GPU_POINT_PREVIEW_POINT_SIZE = 1.5
GPU_POINT_PREVIEW_AUTO_ENABLED = True
GPU_POINT_PREVIEW_AUTO_THRESHOLD = 250_000
GPU_POINT_PREVIEW_MAX_POINTS = 500_000
RASTER_PREVIEW_ENABLED = True
RASTER_PREVIEW_THRESHOLD = 150_000
RASTER_PREVIEW_MAX_PIXELS = 900_000
ERASE_TOL = 12.0
SNAP_TOL = 10.0
START_MAXIMIZED = True


def is_frozen_build() -> bool:
    return bool(getattr(sys, "frozen", False))


def get_project_root() -> Path:
    return Path(__file__).resolve().parent


def get_runtime_root() -> Path:
    if is_frozen_build():
        if sys.platform.startswith("win"):
            appdata = os.environ.get("APPDATA")
            if appdata:
                return Path(appdata) / APP_NAME
            return Path.home() / "AppData" / "Roaming" / APP_NAME
        if sys.platform == "darwin":
            return Path.home() / "Library" / "Application Support" / APP_NAME
        return Path.home() / ".local" / "share" / APP_NAME
    return get_project_root()


def get_workspace_dir() -> Path:
    workspace = get_runtime_root() / WORKSPACE_DIR_NAME
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def get_workspace_path(*parts: str) -> Path:
    return get_workspace_dir().joinpath(*parts)


def get_autosave_dir() -> Path:
    autosave = get_workspace_path(AUTOSAVE_DIR_NAME)
    autosave.mkdir(parents=True, exist_ok=True)
    return autosave


def get_recent_projects_file() -> Path:
    recent_path = get_workspace_path(RECENT_PROJECTS_FILE)
    recent_path.parent.mkdir(parents=True, exist_ok=True)
    return recent_path


def configure_app_logging() -> Path:
    log_dir = get_workspace_path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    root_logger = logging.getLogger()
    log_file_path = str(log_file.resolve())
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            if Path(handler.baseFilename).resolve() == Path(log_file_path):
                return log_file

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root_logger.addHandler(file_handler)
    if root_logger.level == logging.NOTSET or root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)
    logging.getLogger(__name__).info("Logging initialized")
    return log_file
