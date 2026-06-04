import copy
import csv
import gc
import json
import re
import math
import hashlib
import time
import os
import shutil
import subprocess
import yaml
import importlib

import numpy as np
import pandas as pd

from shapely import wkt
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import unary_union, polygonize_full, polygonize, linemerge, transform as shp_transform
from shapely.prepared import prep
from shapely.affinity import translate as shp_translate, scale as shp_scale
from scipy.spatial import Delaunay, cKDTree

from PySide6.QtCore import Qt, QPointF, QRectF, QLineF, Signal, QThread, QObject, QTimer, QMimeData, Slot
from PySide6.QtGui import (
    QPen,
    QPainter,
    QPainterPath,
    QColor,
    QBrush,
    QPolygonF,
    QTransform,
    QImage,
    QPixmap,
    QLinearGradient,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsView,
    QGroupBox,
    QInputDialog,
    QMessageBox,
    QMenu,
    QGraphicsLineItem,
    QGraphicsTextItem,
    QProgressDialog,
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QDialogButtonBox,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QStackedWidget,
    QWidget,
    QLabel,
)

from app_config import (
    SCENE_W,
    SCENE_H,
    SCENE_EXTENT,
    GRID_MINOR,
    GRID_MAJOR,
    DEFAULT_DX,
    MESH_MIN_SPACING_FACTOR,
    MESH_NODE_SOFT_LIMIT,
    MESH_NODE_HARD_LIMIT,
    PREVIEW_CONNECTION_LIMIT,
    FAST_PREVIEW_CONNECTION_LIMIT,
    FAST_PREVIEW_ENABLED,
    GPU_POINT_PREVIEW_ENABLED,
    GPU_POINT_PREVIEW_AUTO_ENABLED,
    GPU_POINT_PREVIEW_AUTO_THRESHOLD,
    GPU_POINT_PREVIEW_MAX_POINTS,
    RASTER_PREVIEW_ENABLED,
    RASTER_PREVIEW_THRESHOLD,
    RASTER_PREVIEW_MAX_PIXELS,
    ERASE_TOL,
    SNAP_TOL,
    get_workspace_dir,
    get_workspace_path,
)
from geometry_utils import dist, point_line_dist, get_solid_features
from ui_numeric import ScientificDoubleSpinBox as QDoubleSpinBox
from ui_numeric import ScientificSpinBox as QSpinBox

def safe_arc(path, x, y, w, h, start, span):
    if path is None:
        return False
    values = (x, y, w, h, start, span)
    try:
        numeric = [float(v) for v in values]
    except Exception:
        return False
    if not all(math.isfinite(v) for v in numeric):
        return False
    if numeric[2] <= 0.0 or numeric[3] <= 0.0:
        return False
    path.arcTo(*numeric)
    return True

GRID_TARGET_MINOR_PX = 10.0

def _snap_125(value):
    if value <= 0 or not math.isfinite(value):
        return 1.0
    exponent = math.floor(math.log10(value))
    base = 10.0 ** exponent
    ratio = value / base
    if ratio < 1.5:
        mult = 1.0
    elif ratio < 3.5:
        mult = 2.0
    elif ratio < 7.5:
        mult = 5.0
    else:
        mult = 10.0
    return mult * base

from cad_kernel import CadKernel
from mesh_utils import (
    dedupe_min_distance,
    poisson_sample,
    square_lattice_sample,
    stabilize_particle_cloud,
    sample_ring,
    triangle_quality_metrics,
    triangle_quality_ok,
    generate_mesh,
    map_geometry_to_nodes,
)
from models import (
    Material,
    MeshSizingPolicy,
    Operation,
    Part,
    Interface,
    normalize_heterogeneity_config,
    normalize_material_field_config,
)
from material_registry import (
    material_behavior_options,
    material_damage_options,
    material_symmetry_options,
    normalize_material_behavior,
    normalize_material_damage,
    normalize_material_properties,
    normalize_material_symmetry,
)
from project_state import ProjectState
from project_stages import ProjectStage
from bisect import bisect_right


class DimensionTextItem(QGraphicsTextItem):
    """Dimension text item with enhanced editing (SolidWorks-style)."""
    
    def __init__(self, text, dim_id, view):
        super().__init__(text)
        self._dim_id = dim_id
        self._view = view
        self.setZValue(5)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsFocusable, True)
        self.setTextInteractionFlags(Qt.NoTextInteraction)
        self.setAcceptHoverEvents(True)
        self._drag_press_scene_pos = None
        self._drag_start_item_pos = None
        self._drag_moved = False
        self._editing = False
        self._setup_tooltip()

    def _setup_tooltip(self):
        """Setup helpful tooltip for dimension editing."""
        try:
            tooltip = (
                "Smart Dimension Editor\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "Double-click to edit inline\n"
                "Type new value and press Enter\n"
                "Press Escape to cancel\n"
                "Right-click for context menu"
            )
            self.setToolTip(tooltip)
        except Exception:
            pass

    def mousePressEvent(self, event):
        if self._editing:
            super().mousePressEvent(event)
            return
        if event.button() == Qt.LeftButton:
            if self._view is not None:
                try:
                    self._view.select_dimension(self._dim_id)
                except Exception:
                    pass
            self._drag_press_scene_pos = event.scenePos()
            self._drag_start_item_pos = self.pos()
            self._drag_moved = False
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_press_scene_pos is not None and self._drag_start_item_pos is not None:
            delta = event.scenePos() - self._drag_press_scene_pos
            self.setPos(self._drag_start_item_pos + delta)
            if abs(delta.x()) > 1.0 or abs(delta.y()) > 1.0:
                self._drag_moved = True
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._drag_press_scene_pos is not None:
            moved = bool(self._drag_moved)
            self._drag_press_scene_pos = None
            self._drag_start_item_pos = None
            self._drag_moved = False
            if moved and self._view:
                try:
                    center = self.sceneBoundingRect().center()
                    self._view._finish_dimension_label_drag(self._dim_id, (center.x(), center.y()))
                except Exception:
                    pass
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def start_inline_edit(self, focus_reason=Qt.OtherFocusReason):
        if self._view is not None:
            try:
                self._view.select_dimension(self._dim_id, suppress_redraw=True)
            except Exception:
                pass
        self._drag_press_scene_pos = None
        self._drag_start_item_pos = None
        self._drag_moved = False
        self._editing = True
        self.setTextInteractionFlags(Qt.TextEditorInteraction)
        self.setFocus(focus_reason)
        try:
            cursor = self.textCursor()
            cursor.select(QTextCursor.Document)
            self.setTextCursor(cursor)
        except Exception:
            pass
        
    def mouseDoubleClickEvent(self, event):
        self.start_inline_edit(Qt.MouseFocusReason)
        event.accept()

    def hoverEnterEvent(self, event):
        """Highlight dimension on hover."""
        if not self._editing and self._view is not None:
            try:
                self._view.select_dimension(self._dim_id)
            except Exception:
                pass
        super().hoverEnterEvent(event)

    def _finish_inline_edit(self, applied, redraw_on_failure=False):
        self._editing = False
        try:
            self.setTextInteractionFlags(Qt.NoTextInteraction)
        except Exception:
            pass
        try:
            self.clearFocus()
        except Exception:
            pass
        if redraw_on_failure and not applied and self._view is not None:
            try:
                self._view.redraw()
            except Exception:
                pass
        return bool(applied)

    def commit_inline_edit(self):
        applied = False
        if self._view is not None:
            try:
                applied = bool(self._view.update_dimension_from_text(self._dim_id, self.toPlainText()))
            except Exception:
                applied = False
        return self._finish_inline_edit(applied, redraw_on_failure=True)

    def cancel_inline_edit(self):
        self._finish_inline_edit(False, redraw_on_failure=False)
        if self._view is not None:
            try:
                self._view.redraw()
            except Exception:
                pass

    def keyPressEvent(self, event):
        if self._editing and event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.commit_inline_edit()
            event.accept()
            return
        if self._editing and event.key() == Qt.Key_Escape:
            self.cancel_inline_edit()
            event.accept()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event):
        """Apply dimension change when focus is lost."""
        if self._editing:
            self.commit_inline_edit()
        else:
            try:
                self.setTextInteractionFlags(Qt.NoTextInteraction)
            except Exception:
                return
        super().focusOutEvent(event)


class MeshWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)
    canceled = Signal()

    def __init__(self, mesh_fn, *args, **kwargs):
        super().__init__()
        self._mesh_fn = mesh_fn
        self._args = args
        self._kwargs = kwargs
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def is_canceled(self):
        return self._cancel_requested

    def _emit_progress(self, value, message):
        self.progress.emit(int(value), str(message))

    def run(self):
        try:
            result = self._mesh_fn(
                *self._args,
                progress_cb=self._emit_progress,
                cancel_cb=self.is_canceled,
            )
            if self._cancel_requested:
                self.canceled.emit()
                return
            self.finished.emit(result)
        except RuntimeError as exc:
            if self._cancel_requested or "canceled" in str(exc).lower():
                self.canceled.emit()
                return
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(str(exc))


class SketchView(QGraphicsView):
    stageAdvanceRequested = Signal(object)
    partsChanged = Signal()
    partSelectionChanged = Signal(object)
    mesh3dUpdated = Signal(object, object)
    animationFramesLoaded = Signal(int)
    animationFrameChanged = Signal(int)
    animationPlaybackStateChanged = Signal(bool)
    replayParticleSelected = Signal(object)
    replayScopeSelectionChanged = Signal(object)

    geometryChanged = Signal()
    materialsChanged = Signal()
    interfacesChanged = Signal()
    interactionsChanged = Signal()
    bcsChanged = Signal()
    loadsChanged = Signal()
    # Canvas status-bar feed (X/Y in scene units, current tool name, zoom).
    cursorScenePositionChanged = Signal(float, float)
    toolChanged = Signal(str)
    zoomChanged = Signal(float)

    _MESH_ARRAY_KEYS = (
        "global_nodes",
        "global_elements",
        "global_nodes_3d",
        "global_elements_3d",
    )
    _MESH_LIST_KEYS = (
        "element_part_map",
        "element_part_map_3d",
    )

    @classmethod
    def _mesh_default_values(cls):
        return {
            "part_meshes": {},
            "global_nodes": np.array([]),
            "global_elements": np.array([]),
            "element_part_map": [],
            "global_nodes_3d": np.array([]),
            "global_elements_3d": np.array([]),
            "element_part_map_3d": [],
            "mesh_validation": {},
        }

    def _ensure_mesh_data_containers(self, mesh_data):
        if not isinstance(mesh_data, dict):
            return
        defaults = self._mesh_default_values()
        for key, fallback in defaults.items():
            value = mesh_data.get(key)
            if key == "part_meshes":
                if not isinstance(value, dict):
                    mesh_data[key] = {}
            elif key in self._MESH_ARRAY_KEYS:
                if not isinstance(value, np.ndarray):
                    mesh_data[key] = np.array([])
            elif key in self._MESH_LIST_KEYS:
                if not isinstance(value, list):
                    mesh_data[key] = []
            elif key == "mesh_validation":
                if not isinstance(value, dict):
                    mesh_data[key] = {}
            elif key not in mesh_data:
                mesh_data[key] = self._clone_mesh_value(fallback)

    def _ensure_project_state_containers(self, state):
        if state is None:
            return
        if not isinstance(getattr(state, "parts", None), list):
            state.parts = []
        if not isinstance(getattr(state, "materials", None), dict):
            state.materials = {}
        if not isinstance(getattr(state, "interfaces", None), list):
            state.interfaces = []
        if not isinstance(getattr(state, "boundary_conditions", None), list):
            state.boundary_conditions = []
        if not isinstance(getattr(state, "loads", None), list):
            state.loads = []
        if not isinstance(getattr(state, "mesh_data", None), dict):
            state.mesh_data = {}
        self._ensure_mesh_data_containers(state.mesh_data)
        if not isinstance(getattr(state, "solver_settings", None), dict):
            state.solver_settings = {}

    def _mesh_store(self):
        state = self._require_project_state()
        return state.mesh_data

    def _store_mesh_validation(self, stats=None):
        store = self._mesh_store()
        store["mesh_validation"] = dict(stats or {})
        return store["mesh_validation"]

    def get_mesh_validation_stats(self):
        try:
            node_count = len(self.global_nodes) if self.global_nodes is not None else 0
        except Exception:
            node_count = 0
        try:
            tri_count = len(self.global_elements) if self.global_elements is not None else 0
        except Exception:
            tri_count = 0
        if node_count <= 0 and tri_count <= 0:
            return {}
        store = self._mesh_store()
        stats = store.get("mesh_validation", {})
        return dict(stats) if isinstance(stats, dict) else {}

    def get_mesh_validation_readout(self):
        stats = self.get_mesh_validation_stats()
        if not stats:
            return "Mesh validation: --"
        return (
            "Mesh validation: "
            f"particles={int(stats.get('particle_count', 0))} | "
            f"triangles={int(stats.get('triangle_count', 0))} | "
            f"orphans removed={int(stats.get('orphan_count', 0))} | "
            f"rejected={int(stats.get('rejected_triangle_count', 0))} | "
            f"near-zero area={int(stats.get('near_zero_area_rejected', 0))}"
        )

    def _require_project_state(self):
        state = getattr(self, "_project_state", None)
        if state is None:
            state = ProjectState()
            self._project_state = state
        self._ensure_project_state_containers(state)
        return state

    @staticmethod
    def _is_empty_store_value(value):
        if isinstance(value, np.ndarray):
            return value.size == 0
        if isinstance(value, (list, tuple, dict, set)):
            return len(value) == 0
        return value is None

    @staticmethod
    def _clone_mesh_value(value):
        if isinstance(value, np.ndarray):
            return np.array(value, copy=True)
        if isinstance(value, list):
            return list(value)
        if isinstance(value, dict):
            return dict(value)
        return value

    @property
    def project_state(self):
        return getattr(self, "_project_state", None)

    @project_state.setter
    def project_state(self, state):
        prev_parts = list(self.parts) if hasattr(self, "parts") else []
        prev_materials = dict(self.materials) if hasattr(self, "materials") else {}
        prev_interfaces = list(self.interfaces) if hasattr(self, "interfaces") else []
        prev_bcs = list(self.bcs) if hasattr(self, "bcs") else []
        prev_loads = list(self.loads) if hasattr(self, "loads") else []
        prev_mesh = {}
        prev_state = getattr(self, "_project_state", None)
        if prev_state is not None and isinstance(getattr(prev_state, "mesh_data", None), dict):
            self._ensure_project_state_containers(prev_state)
            for key, value in prev_state.mesh_data.items():
                prev_mesh[key] = self._clone_mesh_value(value)

        if state is None:
            state = ProjectState()
        self._ensure_project_state_containers(state)
        self._project_state = state

        if not state.parts and prev_parts:
            state.parts = prev_parts
        if not state.materials and prev_materials:
            state.materials = prev_materials
        if not state.interfaces and prev_interfaces:
            state.interfaces = prev_interfaces
        if not state.boundary_conditions and prev_bcs:
            state.boundary_conditions = prev_bcs
        if not state.loads and prev_loads:
            state.loads = prev_loads

        mesh_store = self._mesh_store()
        default_mesh = self._mesh_default_values()
        for key, fallback in default_mesh.items():
            current = mesh_store.get(key)
            if key in prev_mesh and self._is_empty_store_value(current):
                mesh_store[key] = self._clone_mesh_value(prev_mesh[key])
            elif key not in mesh_store:
                mesh_store[key] = self._clone_mesh_value(fallback)

    @property
    def parts(self):
        state = self._require_project_state()
        return state.parts

    @parts.setter
    def parts(self, value):
        payload = list(value) if isinstance(value, (list, tuple)) else []
        state = self._require_project_state()
        state.parts = payload

    @property
    def materials(self):
        state = self._require_project_state()
        return state.materials

    @materials.setter
    def materials(self, value):
        payload = dict(value) if isinstance(value, dict) else {}
        state = self._require_project_state()
        state.materials = payload

    @property
    def interfaces(self):
        state = self._require_project_state()
        return state.interfaces

    @interfaces.setter
    def interfaces(self, value):
        payload = list(value) if isinstance(value, (list, tuple)) else []
        state = self._require_project_state()
        state.interfaces = payload

    @property
    def bcs(self):
        state = self._require_project_state()
        return state.boundary_conditions

    @bcs.setter
    def bcs(self, value):
        payload = list(value) if isinstance(value, (list, tuple)) else []
        state = self._require_project_state()
        state.boundary_conditions = payload

    @property
    def loads(self):
        state = self._require_project_state()
        return state.loads

    @loads.setter
    def loads(self, value):
        payload = list(value) if isinstance(value, (list, tuple)) else []
        state = self._require_project_state()
        state.loads = payload

    @property
    def part_meshes(self):
        store = self._mesh_store()
        if not isinstance(store.get("part_meshes"), dict):
            store["part_meshes"] = {}
        return store["part_meshes"]

    @part_meshes.setter
    def part_meshes(self, value):
        store = self._mesh_store()
        store["part_meshes"] = dict(value) if isinstance(value, dict) else {}

    @property
    def global_nodes(self):
        store = self._mesh_store()
        value = store.get("global_nodes")
        if not isinstance(value, np.ndarray):
            value = np.array([])
            store["global_nodes"] = value
        return value

    @global_nodes.setter
    def global_nodes(self, value):
        store = self._mesh_store()
        try:
            store["global_nodes"] = np.asarray(value)
        except Exception:
            store["global_nodes"] = np.array([])

    @property
    def global_elements(self):
        store = self._mesh_store()
        value = store.get("global_elements")
        if not isinstance(value, np.ndarray):
            value = np.array([])
            store["global_elements"] = value
        return value

    @global_elements.setter
    def global_elements(self, value):
        store = self._mesh_store()
        try:
            store["global_elements"] = np.asarray(value)
        except Exception:
            store["global_elements"] = np.array([])

    @property
    def element_part_map(self):
        store = self._mesh_store()
        value = store.get("element_part_map")
        if not isinstance(value, list):
            value = []
            store["element_part_map"] = value
        return value

    @element_part_map.setter
    def element_part_map(self, value):
        store = self._mesh_store()
        store["element_part_map"] = list(value) if isinstance(value, (list, tuple)) else []

    @property
    def global_nodes_3d(self):
        store = self._mesh_store()
        value = store.get("global_nodes_3d")
        if not isinstance(value, np.ndarray):
            value = np.array([])
            store["global_nodes_3d"] = value
        return value

    @global_nodes_3d.setter
    def global_nodes_3d(self, value):
        store = self._mesh_store()
        try:
            store["global_nodes_3d"] = np.asarray(value)
        except Exception:
            store["global_nodes_3d"] = np.array([])

    @property
    def global_elements_3d(self):
        store = self._mesh_store()
        value = store.get("global_elements_3d")
        if not isinstance(value, np.ndarray):
            value = np.array([])
            store["global_elements_3d"] = value
        return value

    @global_elements_3d.setter
    def global_elements_3d(self, value):
        store = self._mesh_store()
        try:
            store["global_elements_3d"] = np.asarray(value)
        except Exception:
            store["global_elements_3d"] = np.array([])

    @property
    def element_part_map_3d(self):
        store = self._mesh_store()
        value = store.get("element_part_map_3d")
        if not isinstance(value, list):
            value = []
            store["element_part_map_3d"] = value
        return value

    @element_part_map_3d.setter
    def element_part_map_3d(self, value):
        store = self._mesh_store()
        store["element_part_map_3d"] = list(value) if isinstance(value, (list, tuple)) else []

    def _current_mesh_geometry_signature(self):
        payload = {
            "project_mode": str(getattr(self, "project_mode", "2d")),
            "extrude_height": float(getattr(self, "extrude_height", 0.0) or 0.0),
            "extrude_layers": int(getattr(self, "extrude_layers", 0) or 0),
            "parts": [],
        }
        for part in getattr(self, "parts", []) or []:
            geom = getattr(part, "geometry", None)
            try:
                geom_wkt = geom.wkt if geom is not None else ""
            except Exception:
                geom_wkt = ""
            payload["parts"].append(
                {
                    "id": int(getattr(part, "id", 0) or 0),
                    "name": str(getattr(part, "name", "")),
                    "material_id": getattr(part, "material_id", None),
                    "is_void": bool(getattr(part, "is_void", False)),
                    "is_rigid": bool(getattr(part, "is_rigid", False)),
                    "geometry_wkt": geom_wkt,
                }
            )
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _mesh_config_signature(self, mesh_config=None):
        mode = "dx"
        dx = self.last_mesh_dx
        target_nodes = self.last_mesh_target_nodes
        distribution = str(getattr(self, "mesh_distribution", "global_poisson")).lower()
        backend = str(getattr(self, "mesh_backend", "auto")).lower()
        if isinstance(mesh_config, dict) and mesh_config:
            mode = str(mesh_config.get("mode", mode)).lower()
            if mode in ("spacing",):
                mode = "dx"
            elif mode in ("count", "nodes", "total"):
                mode = "count"
            dx = mesh_config.get("dx", dx)
            target_nodes = mesh_config.get("target_nodes", target_nodes)
            distribution = str(mesh_config.get("distribution", distribution)).lower()
            backend = str(mesh_config.get("backend", backend)).lower()
        payload = {
            "mode": mode,
            "dx": None if dx is None else round(float(dx), 8),
            "target_nodes": None if target_nodes is None else int(target_nodes),
            "distribution": distribution,
            "backend": backend,
        }
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _mark_mesh_current(self, mesh_config=None):
        store = self._mesh_store()
        store["geometry_signature"] = self._current_mesh_geometry_signature()
        store["config_signature"] = self._mesh_config_signature(mesh_config)

    def has_current_mesh(self, mesh_config=None):
        if self.project_mode == "3d":
            nodes = self.global_nodes_3d
            elements = self.global_elements_3d
        else:
            nodes = self.global_nodes
            elements = self.global_elements
        if nodes is None or elements is None or len(nodes) == 0 or len(elements) == 0:
            return False
        store = self._mesh_store()
        return (
            store.get("geometry_signature") == self._current_mesh_geometry_signature()
            and store.get("config_signature") == self._mesh_config_signature(mesh_config)
        )

    def has_current_particle_set(self, mesh_config=None):
        nodes = self.global_nodes
        if nodes is None or len(nodes) == 0:
            return False
        store = self._mesh_store()
        return (
            store.get("geometry_signature") == self._current_mesh_geometry_signature()
            and store.get("config_signature") == self._mesh_config_signature(mesh_config)
        )

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self._project_state = None
        self.project_state = ProjectState()
        self.setRenderHints(
            QPainter.Antialiasing
            | QPainter.TextAntialiasing
            | QPainter.SmoothPixmapTransform
        )
        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, False)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setCacheMode(QGraphicsView.CacheNone)
        self.setBackgroundBrush(QColor(246, 249, 253))
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setAcceptDrops(True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

        # Scene Setup
        self.setSceneRect(-SCENE_EXTENT, -SCENE_EXTENT, 2.0 * SCENE_EXTENT, 2.0 * SCENE_EXTENT)
        self.centerOn(0, 0)
        self._y_axis_up = True
        if self._y_axis_up:
            self.scale(1, -1)

        # State Variables
        # Default to the hand/select tool so the canvas starts in "click-and-
        # drag to move" mode. Drawing tools activate only when the user
        # explicitly clicks a shape/template button. Right-click returns the
        # cursor to this default state.
        self.tool = "select"
        self.mode = "idle"
        self.active_module = "Part" # Default module
        
        # Data Storage
        self.current_unit = "mm"
        self.sketches = [] 
        self.sketch_meta = []
        self.solid_geometry = None # Main Shapely object
        self.current = [] 
        self.preview_items = []
        self.bcs = [] 
        self.loads = [] 
        self._panel_attr_focus_kind = None
        self._panel_attr_focus_entry_ref = None
        self._next_force_id = 1
        self._next_velocity_id = 1
        self.hover_item = None
        self.polygon_sides = 3
        self.material_paint_mode = False
        self.selected_part_id = None
        self.snap_grid = True
        self.snap_endpoints = True
        self.snap_midpoints = True
        self.snap_angle = True
        self.precision_sketch_mode_enabled = False
        self.parametric_enabled = False
        self.freeform_auto_convert_enabled = True
        self._snap_indicator = None
        self.command_last_point = (0.0, 0.0)
        self.display_mode = "geometry"
        self.navigation_mode_enabled = False
        self._nav_rotating = False
        self._nav_rotate_start = None
        self._nav_right_dragged = False
        self._last_mouse_context_menu_ts = 0.0
        self._slot_width = 10.0
        self._projection_angle = math.radians(35.0)
        self._warned_cpd_bounds = False
        self.dimensions = []
        self.constraints = []
        self._dimension_items = {}
        self._active_dimension_item_ids = set()
        self._mesh_cache = {"key": None, "parts_signature": None, "result": None}
        self.show_dimensions = True
        self._mesh_status_pct = None
        self._dimension_id_counter = 0
        self.selected_dimension_id = None
        self._pending_dimension = None
        self._pending_constraint = None
        self.show_dimensions = True
        self._sketch_edit_mode = False
        self._dim_offset = 14.0
        self._pending_meta = None
        self._displacement_vectors = []
        self._result_scalar_values = None
        self._result_field_label = ""
        self.grid_visible = True
        self._effective_grid_spacing = float(GRID_MINOR)
        self.paint_brush = {"type": "fix_xy", "fx": 0.0, "fy": 0.0, "val": 0.0}
        self._paint_active = False
        self._paint_points = []
        self._paint_preview_item = None
        self._pending_attr_edit = None
        self._arc_select_points = []
        self._arc_segment_polyline = None
        self._arc_segment_preview_item = None
        self._transform_tool = None
        self._transform_base = None
        self._transform_line = []
        self._transform_preview_item = None
        self._transform_drag_active = False
        self._transform_drag_moved = False
        self._transform_drag_start_view = None
        self._transform_drag_threshold_px = 6
        self._rect_draw_mode = "corner"
        self._rect_use_dimensions = False
        self._circle_draw_mode = "point"
        self._pending_rectangle_start = None
        self._rect_cursor_message = ""
        
        # Mesh Data
        self.part_meshes = {}
        self.global_nodes = np.array([])
        self.global_elements = np.array([])
        self.element_part_map = []
        self.element_part_map_3d = []
        self.mesh_distribution = "poisson"
        self.mesh_backend = "auto"
        self.last_mesh_dx = None
        self.last_mesh_target_nodes = None
        self._interface_preview_cache = None
        self._interface_preview_cache_sig = None
        self.interface_preview_color = QColor(70, 150, 255)
        self.mesh_preview_line_width = 1.0
        self.mesh_preview_particle_size = 3.0
        
        # NEW: Material & Operation Stack
        self.materials = {}  # {serial: Material object}
        self.operations = []  # List of Operation objects (history tree)
        self.current_material_id = None  # Selected material for next shape
        self.material_color_map = {}  # Map material_id to QColor
        
        # NEW: Parts & Interfaces
        self.parts = []  # List of Part objects
        self.interfaces = []  # List of Interface objects
        self.part_counter = 0  # For auto-naming parts
        self.initial_velocities = []
        self._porous_sketch_name = None
        self.porous_settings = {}
        self._pending_generated_feature_settings = None
        self._editing_part_shape_id = None

        # Animation State
        self.is_visualization_mode = False
        self.animation_frames = []
        self.current_frame_index = 0
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.advance_animation_frame)
        self._lazy_results_enabled = False
        self._animation_frame_count = 0
        self._animation_frame_loading = False
        self._pending_lazy_frame_index = None
        self._current_animation_positions = None
        self._current_animation_velocity = None
        self._current_frame_packet = None
        self._results_legend_state = None
        self._results_preview_auto_fit_pending = False
        self.results_preview_threshold = 20000
        self.results_preview_lod_limit = 50000
        self.show_anim_nodes = True
        self.show_anim_elements = True
        self.show_anim_element_alpha = 0.6
        self._replay_pick_mode = "node"
        self._replay_selected_nodes = set()
        self._replay_selected_triangles = set()
        self._replay_selected_mesh_edges = set()
        self._replay_selected_geometry_edges = []
        self._replay_selected_bc_targets = []
        self._replay_scope_item = None
        # Mesh-node dots are off by default — the triangle edges already convey
        # the mesh density. Users can re-enable via the "Particles" checkbox in
        # the Mesh panel.
        self.show_mesh_nodes = False
        self.show_mesh_elements = False
        self.show_anim_bc_markers = True
        self.show_anim_load_vectors = True
        self.show_anim_particle_ids = False
        self._replay_lod_active = False
        self._replay_visible_particle_indices = None
        self._replay_selected_particle_index = None
        self._replay_particle_ids = None
        self._replay_particle_materials = []
        self._replay_particle_parts = []
        self._replay_particle_id_to_index = {}
        self._replay_part_to_indices = {}
        self._replay_particle_bc_labels = {}
        self._replay_particle_load_vectors = {}
        self._replay_metadata_particle_count = 0
        self._results_debug_overlay_items = []
        self.fast_preview_enabled = FAST_PREVIEW_ENABLED
        self.fast_preview_connection_limit = FAST_PREVIEW_CONNECTION_LIMIT
        self.gpu_point_preview_enabled = GPU_POINT_PREVIEW_ENABLED
        self.gpu_point_preview_auto = GPU_POINT_PREVIEW_AUTO_ENABLED
        self.gpu_point_preview_threshold = GPU_POINT_PREVIEW_AUTO_THRESHOLD
        self.gpu_point_preview_max_points = GPU_POINT_PREVIEW_MAX_POINTS
        self.raster_preview_enabled = RASTER_PREVIEW_ENABLED
        self.raster_preview_threshold = RASTER_PREVIEW_THRESHOLD
        self.raster_preview_max_pixels = RASTER_PREVIEW_MAX_PIXELS
        # Geometry draw LOD: favor responsiveness for dense porous/pattern models.
        self.geometry_fast_draw_enabled = True
        self.geometry_fast_draw_max_holes = 120
        self.geometry_fast_draw_max_sketches = 500
        self.geometry_fast_draw_max_preview_polys = 300
        self.geometry_fast_draw_max_path_points = 800
        self.geometry_fast_draw_min_hole_pixels = 3.0
        self.geometry_fast_draw_simplify_pixels = 0.75
        self.mesh_min_spacing_factor = MESH_MIN_SPACING_FACTOR
        self.mesh_boundary_thickness = 0.0
        self.mesh_boundary_spacing_factor = 1.0
        self.project_mode = "2d"
        self.cad_kernel = CadKernel()
        self.use_cad_kernel = True
        self.extrude_height = 10.0
        self.extrude_layers = 4
        self.global_nodes_3d = np.array([])
        self.global_elements_3d = np.array([])

        self.frames = []
        self.current_frame = 0
        self.undo_stack = []
        self.redo_stack = []
        self._history_blocked = False
        self._panning = False
        self._pan_start = None
        self._space_pressed = False
        self._redraw_pending = False
        self._zoom_window_active = False
        self._zoom_window_start = None
        self._zoom_window_item = None
        self._zoom_window_temporary = False
        self._zoom = 1.0
        self._mesh_time_history = []
        self._mesh_thread = None
        self._mesh_worker = None
        self._mesh_progress_dialog = None
        self._mesh_preview_only = False
        self._mesh_on_done = None
        self._mesh_task_kind = "connections"
        # Custom-zone polygon-draw state (Results page Custom mode).
        self._zone_draw_active = False
        self._zone_draw_points = []
        self._zone_draw_hover = None
        self._zone_draw_callback = None
        self._zone_overlay_polygons = []
        self._zone_overlay_visible = False
        self._pending_custom_zones = []
        self._pending_edge_seeds = []
        self._pending_vertex_seeds = []
        self._pending_boundary_layer_seeds = []
        self._pending_matched_edge_pairs = []
        self._pending_part_mesh_overrides = {}
        # Edge-seed picking state (Mesh stage)
        self._edge_seed_pick_mode = None  # None | "single" | "multi"
        self._edge_seed_picked = []  # list of {"part_id", "start", "end"}
        self._edge_seed_hover = None  # current edge under cursor: same dict shape
        self._edge_seed_callback = None
        # Seed preview: live tick marks shown while the Local Seeds dialog is open
        self._seed_preview_ticks = []  # flat list of (x, y) world coords
        # Vertex-seed picking state (Mesh stage)
        self._vertex_seed_pick_active = False
        self._vertex_seed_hover = None  # (part_id, (x, y)) or None
        self._vertex_seed_callback = None
        self._vertex_seed_highlight = []  # list of (part_id, (x, y)) — persistent overlay
        # Partition picking state (Mesh stage)
        self._partition_pick_active = False
        self._partition_points = []  # 0, 1, or 2 (x, y) points so far
        self._partition_hover = None  # last cursor scene-pos for live preview line
        self._partition_callback = None
        # Measure-tool state (Workspace UX): pick two points, show distance +
        # angle. `_measure_first` holds (x, y) of the first picked point;
        # `_measure_second` is set on the second click and persists the
        # finished measurement until the next click or Esc.
        self._measure_first = None
        self._measure_second = None
        # Smart alignment guides shown while a part is being click-dragged.
        # Each entry is a dict {orient: 'v'|'h', value: float, y_min/y_max
        # (for vertical) or x_min/x_max (for horizontal)}. Cleared on
        # release.
        self._align_guides = []
        self._mesh_last_progress_ts = 0.0
        self._redraw_in_progress = False
        self._mesh_status_pending = None
        self._mesh_status_timer = QTimer(self)
        self._mesh_status_timer.setSingleShot(True)
        self._mesh_status_timer.setInterval(120)
        self._mesh_status_timer.timeout.connect(self._flush_mesh_status)
        self._mesh_qa_cache = None
        self._mesh_qa_cache_sig = None

    def set_tool(self, tool):
        tool = str(tool or "")
        self._ensure_tool_module(tool)
        if tool == "dimension" and not self._can_use_smart_dimensions():
            self._announce_status("Smart Dimension is only available while editing a sketch.")
            return
        self.tool = tool
        self.current_tool = tool
        if tool == "dimension":
            self.precision_sketch_mode_enabled = True
            self.show_dimensions = True
            window = self.window()
            if window is not None and hasattr(window, "_set_precision_sketch_mode"):
                try:
                    window._set_precision_sketch_mode(True, announce=False)
                except Exception:
                    pass
            else:
                self.set_dimensions_visible(True)
        self.mode = "idle"
        self.current.clear()
        self._pending_rectangle_start = None
        self._clear_preview()
        self.hover_item = None
        draw_tools = (
            "line",
            "rectangle",
            "circle",
            "ellipse",
            "polygon",
            "polyline",
            "arc",
            "freeform",
            "zoom_window",
        )
        if tool in draw_tools or tool in ("move", "copy", "mirror"):
            self.setCursor(Qt.CrossCursor)
        elif tool == "select":
            # Open-hand "grab" cursor signals click-and-drag move mode.
            self.setCursor(Qt.OpenHandCursor)
        elif tool == "measure":
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        if tool != "measure":
            self._measure_first = None
            self._measure_second = None
        if tool not in ("move", "copy", "mirror"):
            self._reset_transform_state()
        if tool != "dimension":
            self._pending_dimension = None
        if tool != "constraint":
            self._pending_constraint = None
        if tool != "paint_bc":
            self._paint_active = False
            self._paint_points = []
            self._clear_paint_preview()
        if tool != "arc_segment":
            self._clear_arc_segment_selection()
        if tool != "zoom_window":
            self._clear_zoom_window()
        if tool in ("move", "copy", "mirror"):
            self._transform_tool = tool
            self._transform_base = None
            self._transform_line = []
            self._clear_transform_preview()
            self._transform_drag_active = False
            self._transform_drag_moved = False
            self._transform_drag_start_view = None
            if tool == "move":
                self._announce_status("Move: click base point, then target point.")
            elif tool == "copy":
                self._announce_status("Copy: click base point, then target point. Esc to finish.")
            elif tool == "mirror":
                self._announce_status("Mirror: click first point of mirror line.")
        if self.navigation_mode_enabled:
            self.setDragMode(QGraphicsView.RubberBandDrag if tool == "select" else QGraphicsView.NoDrag)
        else:
            self.setDragMode(QGraphicsView.NoDrag)
        self._show_tool_hint(tool)
        try:
            self.toolChanged.emit(str(tool))
        except Exception:
            pass
        self.redraw()

    def _can_use_smart_dimensions(self):
        if self.in_sketch_edit_mode():
            return True
        if str(getattr(self, "tool", "")).lower() == "dimension":  # ADD THIS
            return True                                              # ADD THIS
        return (
            str(getattr(self, "active_module", "")).lower() == "part"
            and self.display_mode in ("geometry", "sketch_edit")
            and bool(getattr(self, "sketches", []))
        )

    def _ensure_tool_module(self, tool):
        """Route part-edit tools to the Geometry/Part module so they do not silently no-op."""
        part_only_tools = {
            "line",
            "rectangle",
            "circle",
            "ellipse",
            "polygon",
            "polyline",
            "arc",
            "freeform",
            "slot",
            "spline",
            "move",
            "copy",
            "mirror",
            "trim",
            "dimension",
            "constraint",
            "erase",
        }
        if tool not in part_only_tools or self.active_module == "Part":
            return

        switched = False
        window = self.window()
        tabs = getattr(getattr(window, "properties_panel", None), "tabs", None)
        if tabs and hasattr(tabs, "setCurrentIndex") and tabs.isTabEnabled(0):
            if tabs.currentIndex() != 0:
                tabs.setCurrentIndex(0)
                switched = True

        if self.active_module != "Part":
            self.set_module("Part")
            switched = True

        if switched:
            self._announce_status("Switched to Geometry module for sketch edit tools.")

    def start_attr_reassign(self, item, kind):
        if kind not in ("bc", "load"):
            return
        self._pending_attr_edit = {"kind": kind, "item": item}
        window = self.window()
        if window and hasattr(window, "statusBar"):
            window.statusBar().showMessage(
                "Click a vertex or edge to move the selected attribute.", 5000
            )

    def set_rectangle_draw_options(self, mode="corner", use_dimensions=False):
        mode_key = "center" if str(mode).strip().lower().startswith("center") else "corner"
        self._rect_draw_mode = mode_key
        self._rect_use_dimensions = bool(use_dimensions)

    def set_circle_draw_mode(self, mode="point"):
        mode_key = str(mode or "point").strip().lower()
        self._circle_draw_mode = "radius" if mode_key == "radius" else "point"

    def set_grid_visible(self, enabled):
        self.grid_visible = bool(enabled)
        self.redraw()

    def _show_tool_hint(self, tool):
        hint = ""
        if tool == "arc":
            hint = "Arc: click start, click end, then click a point on the arc."
        elif tool == "arc_segment":
            hint = "Arc segment: click start, end, then mid on a curve to select a segment."
        elif tool == "line":
            hint = "Line: click start point, then click end point."
        elif tool == "circle":
            if self._circle_draw_mode == "radius":
                hint = "Circle: click center, then enter radius."
            else:
                hint = "Circle: click center, then click radius point."
        elif tool == "polygon":
            hint = "Polygon: click center, then click to set size."
        elif tool == "polyline":
            hint = "Polyline: click to add points, right-click to finish. Click near start to close."
        elif tool == "freeform":
            hint = (
                "Freeform: drag to scribble and release to finish. "
                "Enable Auto-Convert to fit a clean shape."
            )
        elif tool == "slot":
            hint = "Slot: click start and end, then enter width."
        elif tool == "rectangle":
            hint = "Rectangle: 2-corner parametric rectangle. Click first corner, move, then click opposite corner."
        elif tool == "trim":
            hint = "Trim: click a sketch segment to remove it."
        elif tool == "dimension":
            hint = (
                "Smart Dimension: click a segment/vertex, choose type "
                "(length/radius/diameter/arc length/angle), then click to place."
            )
        elif tool == "constraint":
            hint = "Constraint: click a segment, choose constraint, then click another segment if needed."
        elif tool == "zoom_window":
            hint = "Zoom Window: drag a rectangle to zoom. Ctrl + left-drag also works temporarily."
        if hint:
            window = self.window()
            if window and hasattr(window, "statusBar"):
                window.statusBar().showMessage(hint, 5000)

    def _start_zoom_window(self, pt, temporary=False):
        self._zoom_window_active = True
        self._zoom_window_start = pt
        self._zoom_window_temporary = bool(temporary)
        self._remove_scene_item_safe(self._zoom_window_item)
        pen = QPen(QColor(0, 120, 255), 1, Qt.DashLine)
        self._zoom_window_item = self.scene().addRect(
            pt[0], pt[1], 1, 1, pen, QBrush(Qt.transparent)
        )

    def _finish_zoom_window(self):
        if self._zoom_window_item:
            try:
                rect = self._zoom_window_item.rect()
            except RuntimeError:
                rect = None
            if rect is not None and rect.width() > 1 and rect.height() > 1:
                self._apply_fit_rect(rect)
        restore_cursor = self._zoom_window_temporary
        self._clear_zoom_window()
        if restore_cursor:
            draw_tools = (
                "line",
                "rectangle",
                "circle",
                "ellipse",
                "polygon",
                "polyline",
                "arc",
                "freeform",
                "zoom_window",
            )
            if self.tool in draw_tools or self.tool in ("move", "copy", "mirror"):
                self.setCursor(Qt.CrossCursor)
            else:
                self.setCursor(Qt.ArrowCursor)

    def set_module(self, module_name):
        """Switches the active module state."""
        # Stop any running animation when switching modules
        if self.is_visualization_mode and module_name != "Results":
            self.animation_timer.stop()
            self._emit_animation_playback_state()
            self.is_visualization_mode = False
            self.set_display_mode("geometry")

        if module_name != "Property" and self.material_paint_mode:
            self.set_material_paint_mode(False)

        self.active_module = module_name
        if module_name == "Part" and not self.is_visualization_mode:
            self.display_mode = "geometry"
        default_tool = self._default_sketch_tool_name() if module_name == "Part" else "select"
        self.set_tool(default_tool) # Reset tool when switching modules
        self.redraw()

    def set_display_mode(self, mode):
        if mode not in ("geometry", "sketch_edit", "mesh", "mesh_3d", "results", "bc"):
            return
        self.display_mode = mode
        if mode != "results":
            # Hide the live results overlay but keep cached field data
            # (_displacement_vectors, _result_scalar_values, _result_field_label,
            # _results_legend_state) so it redraws when the user returns to
            # the Results stage without forcing a re-load.
            self._emit_animation_playback_state()
            window = self.window()
            if window is not None and hasattr(window, "_hide_results_point_preview"):
                try:
                    window._hide_results_point_preview(force_view=True)
                except Exception:
                    pass
        self.redraw()

    def set_navigation_mode(self, enabled):
        self.navigation_mode_enabled = bool(enabled)
        if not self.navigation_mode_enabled:
            self._nav_rotating = False
            self._nav_rotate_start = None
            self._nav_right_dragged = False
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.ArrowCursor)
        if self.navigation_mode_enabled:
            self.setDragMode(QGraphicsView.RubberBandDrag if self.tool == "select" else QGraphicsView.NoDrag)
        else:
            self.setDragMode(QGraphicsView.NoDrag)

    def set_dimensions_visible(self, enabled):
        enabled = bool(enabled) and (
            bool(getattr(self, "precision_sketch_mode_enabled", False))
            or self._can_use_smart_dimensions()
        )
        if enabled == self.show_dimensions:
            return
        self.show_dimensions = enabled
        self.geometryChanged.emit()
        self.redraw()

    def ensure_y_axis_up(self):
        if not self._y_axis_up:
            return
        if self.transform().m22() > 0:
            self.scale(1, -1)

    def _default_sketch_meta(self, points):
        return {"type": "polyline", "points": copy.deepcopy(points)}

    def _sync_sketch_meta_list(self, sketches, meta):
        source_meta = list(meta or [])
        synced = []
        for idx, points in enumerate(sketches):
            if idx < len(source_meta):
                item = copy.deepcopy(source_meta[idx])
                if not isinstance(item, dict) or not item:
                    item = self._default_sketch_meta(points)
            else:
                item = self._default_sketch_meta(points)
            if str(item.get("type", "polyline")).lower() == "rectangle":
                item = self._normalize_rectangle_meta(item, fallback_points=points)
            synced.append(item)
        return synced

    def _sync_all_sketch_meta(self):
        self.sketch_meta = self._sync_sketch_meta_list(self.sketches, self.sketch_meta)
        for part in self.parts:
            part.sketch_meta = self._sync_sketch_meta_list(
                getattr(part, "sketches", []), getattr(part, "sketch_meta", [])
            )

    def _ensure_dimensions(self):
        if self.sketches and not self.dimensions:
            for idx, meta in enumerate(self.sketch_meta):
                self._auto_create_dimensions("sketch", None, idx, meta)
        for part in self.parts:
            if getattr(part, "sketches", []) and not getattr(part, "dimensions", []):
                part.dimensions = []
                for idx, meta in enumerate(getattr(part, "sketch_meta", [])):
                    self._auto_create_dimensions("part", part, idx, meta)

    def _recalc_dimension_counter(self):
        max_id = 0
        for dim in self.dimensions:
            try:
                max_id = max(max_id, int(dim.get("id", 0)))
            except Exception:
                continue
        for part in self.parts:
            for dim in getattr(part, "dimensions", []):
                try:
                    max_id = max(max_id, int(dim.get("id", 0)))
                except Exception:
                    continue
        self._dimension_id_counter = max(self._dimension_id_counter, max_id)

    def _next_dimension_id(self):
        self._dimension_id_counter += 1
        return self._dimension_id_counter

    def _owner_collections(self, owner_type, owner_part=None):
        if owner_type == "part" and owner_part is not None:
            if not hasattr(owner_part, "sketches"):
                owner_part.sketches = []
            if not hasattr(owner_part, "sketch_meta"):
                owner_part.sketch_meta = []
            if not hasattr(owner_part, "dimensions"):
                owner_part.dimensions = []
            if not hasattr(owner_part, "constraints"):
                owner_part.constraints = []
            return (
                owner_part.sketches,
                owner_part.sketch_meta,
                owner_part.dimensions,
                owner_part.constraints,
            )
        return self.sketches, self.sketch_meta, self.dimensions, self.constraints

    def _active_owner(self):
        if self.sketches:
            return "sketch", None
        if self.selected_part_id is not None:
            part = next((p for p in self.parts if p.id == self.selected_part_id), None)
            if part and getattr(part, "sketches", []):
                return "part", part
        return None, None

    def _resolve_active_owner(self, pt=None):
        owner_type, owner_part = self._active_owner()
        if owner_type is not None:
            return owner_type, owner_part
        if pt is None:
            return None, None
        part = self._get_part_at_point(pt, include_void=False)
        if part is None or not getattr(part, "sketches", []):
            return None, None
        self.set_selected_part(getattr(part, "id", None), emit_signal=True)
        return "part", part

    def _append_sketch(self, points, meta=None, owner_type="sketch", owner_part=None):
        sketches, sketch_meta, dimensions, constraints = self._owner_collections(owner_type, owner_part)
        stored_points = copy.deepcopy(points)
        stored_meta = copy.deepcopy(meta) if isinstance(meta, dict) else meta
        if stored_meta is None:
            stored_meta = self._default_sketch_meta(stored_points)
        if str((stored_meta or {}).get("type", "polyline")).lower() == "rectangle":
            stored_meta = self._normalize_rectangle_meta(stored_meta, fallback_points=stored_points)
            stored_points = self._build_points_from_meta(stored_meta, fallback_points=stored_points)
        sketches.append(stored_points)
        sketch_meta.append(stored_meta)
        sketch_index = len(sketches) - 1
        self._auto_create_dimensions(owner_type, owner_part, sketch_index, stored_meta)
        self._auto_create_constraints(owner_type, owner_part, sketch_index, stored_meta)

    def _remove_owner_sketch(self, owner_type, owner_part, sketch_index):
        sketches, metas, dimensions, constraints = self._owner_collections(owner_type, owner_part)
        if sketch_index < 0 or sketch_index >= len(sketches):
            return False
        sketches.pop(sketch_index)
        if sketch_index < len(metas):
            metas.pop(sketch_index)

        dimensions[:] = [
            d for d in dimensions if int(d.get("sketch_index", -1)) != int(sketch_index)
        ]
        for dim in dimensions:
            if int(dim.get("sketch_index", -1)) > int(sketch_index):
                dim["sketch_index"] = int(dim.get("sketch_index", 0)) - 1

        constraints[:] = [
            c
            for c in constraints
            if int(c.get("sketch_index", -1)) != int(sketch_index)
            and int(c.get("other_sketch_index", -1)) != int(sketch_index)
        ]
        for con in constraints:
            if int(con.get("sketch_index", -1)) > int(sketch_index):
                con["sketch_index"] = int(con.get("sketch_index", 0)) - 1
            if int(con.get("other_sketch_index", -1)) > int(sketch_index):
                con["other_sketch_index"] = int(con.get("other_sketch_index", 0)) - 1
        return True

    def _remove_sketch(self, sketch_index):
        self._remove_owner_sketch("sketch", None, sketch_index)

    def _get_sketch_points(self, owner_type, owner_part, sketch_index):
        sketches, _, _, _ = self._owner_collections(owner_type, owner_part)
        if sketch_index < 0 or sketch_index >= len(sketches):
            return None
        return sketches[sketch_index]

    def _get_sketch_meta(self, owner_type, owner_part, sketch_index):
        _, metas, _, _ = self._owner_collections(owner_type, owner_part)
        if sketch_index < 0 or sketch_index >= len(metas):
            return {}
        return metas[sketch_index] or {}

    def _coerce_xy_tuple(self, value):
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return None
        try:
            return (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            return None

    def _rectangle_mode_key(self, mode=None):
        mode_key = str(mode or "two_corner").strip().lower()
        if mode_key == "center":
            return "center"
        return "two_corner"

    def _rectangle_meta_from_origin_size(self, origin, width, height, mode="two_corner", base_meta=None):
        origin_pt = self._coerce_xy_tuple(origin) or (0.0, 0.0)
        try:
            width_val = max(0.0, float(width))
        except (TypeError, ValueError):
            width_val = 0.0
        try:
            height_val = max(0.0, float(height))
        except (TypeError, ValueError):
            height_val = 0.0

        mode_key = self._rectangle_mode_key(mode)
        x0, y0 = origin_pt
        out = {
            "type": "rectangle",
            "mode": mode_key,
            "origin": (x0, y0),
            "width": width_val,
            "height": height_val,
        }
        if isinstance(base_meta, dict):
            for key, val in base_meta.items():
                if key not in {"type", "mode", "p1", "p2", "origin", "width", "height"}:
                    out[key] = copy.deepcopy(val)
        return out

    def _rectangle_meta_from_input_points(self, p1, p2, mode="two_corner", base_meta=None):
        pt1 = self._coerce_xy_tuple(p1)
        pt2 = self._coerce_xy_tuple(p2)
        if pt1 is None or pt2 is None:
            return None

        mode_key = self._rectangle_mode_key(mode)
        if mode_key == "center":
            cx, cy = pt1
            px, py = pt2
            hx = abs(px - cx)
            hy = abs(py - cy)
            origin = (cx - hx, cy - hy)
            width = hx * 2.0
            height = hy * 2.0
        else:
            x0, y0 = pt1
            x1, y1 = pt2
            origin = (min(x0, x1), min(y0, y1))
            width = abs(x1 - x0)
            height = abs(y1 - y0)

        return self._rectangle_meta_from_origin_size(
            origin,
            width,
            height,
            mode=mode_key,
            base_meta=base_meta,
        )

    def _normalize_rectangle_meta(self, meta, fallback_points=None):
        meta_copy = copy.deepcopy(meta or {})
        origin = self._coerce_xy_tuple(meta_copy.get("origin"))
        has_origin_size = (
            "origin" in meta_copy
            and "width" in meta_copy
            and "height" in meta_copy
        )
        try:
            width_val = max(0.0, float(meta_copy.get("width", 0.0)))
        except (TypeError, ValueError):
            width_val = None
        try:
            height_val = max(0.0, float(meta_copy.get("height", 0.0)))
        except (TypeError, ValueError):
            height_val = None

        if has_origin_size and origin is not None and width_val is not None and height_val is not None:
            return self._rectangle_meta_from_origin_size(
                origin,
                width_val,
                height_val,
                mode=meta_copy.get("mode"),
                base_meta=meta_copy,
            )

        p1 = self._coerce_xy_tuple(meta_copy.get("p1"))
        p2 = self._coerce_xy_tuple(meta_copy.get("p2"))
        if p1 is not None and p2 is not None:
            normalized = self._rectangle_meta_from_input_points(
                p1,
                p2,
                mode=meta_copy.get("mode"),
                base_meta=meta_copy,
            )
            if normalized is not None:
                return normalized

        if (origin is None or width_val is None or height_val is None) and fallback_points:
            xs = []
            ys = []
            for pt in list(fallback_points or []):
                xy = self._coerce_xy_tuple(pt)
                if xy is None:
                    continue
                xs.append(xy[0])
                ys.append(xy[1])
            if xs and ys:
                origin = (min(xs), min(ys))
                width_val = max(xs) - min(xs)
                height_val = max(ys) - min(ys)

        return self._rectangle_meta_from_origin_size(
            origin or (0.0, 0.0),
            width_val if width_val is not None else 0.0,
            height_val if height_val is not None else 0.0,
            mode=meta_copy.get("mode"),
            base_meta=meta_copy,
        )

    def _circle_meta_from_center_radius(self, center, radius, base_meta=None):
        center_pt = self._coerce_xy_tuple(center) or (0.0, 0.0)
        try:
            radius_val = max(0.0, float(radius))
        except (TypeError, ValueError):
            radius_val = 0.0
        out = {
            "type": "circle",
            "center": center_pt,
            "radius": radius_val,
        }
        if isinstance(base_meta, dict):
            for key, val in base_meta.items():
                if key not in {"type", "center", "radius"}:
                    out[key] = copy.deepcopy(val)
        return out

    def _normalize_circle_meta(self, meta, fallback_points=None):
        meta_copy = copy.deepcopy(meta or {})
        center = self._coerce_xy_tuple(meta_copy.get("center"))
        try:
            radius_val = max(0.0, float(meta_copy.get("radius", 0.0)))
        except (TypeError, ValueError):
            radius_val = None

        if (center is None or radius_val is None or radius_val <= 0.0) and fallback_points:
            try:
                fit = self._fit_circle_kasa(fallback_points)
            except Exception:
                fit = None
            if fit is not None:
                center = self._coerce_xy_tuple(fit.get("center"))
                try:
                    radius_val = max(0.0, float(fit.get("radius", 0.0)))
                except (TypeError, ValueError):
                    radius_val = 0.0

        return self._circle_meta_from_center_radius(
            center or (0.0, 0.0),
            radius_val if radius_val is not None else 0.0,
            base_meta=meta_copy,
        )

    def _line_meta_from_points(self, p1, p2, base_meta=None):
        p1_pt = self._coerce_xy_tuple(p1) or (0.0, 0.0)
        p2_pt = self._coerce_xy_tuple(p2) or p1_pt
        out = {
            "type": "line",
            "p1": p1_pt,
            "p2": p2_pt,
        }
        if isinstance(base_meta, dict):
            for key, val in base_meta.items():
                if key not in {"type", "p1", "p2"}:
                    out[key] = copy.deepcopy(val)
        return out

    def _normalize_line_meta(self, meta, fallback_points=None):
        meta_copy = copy.deepcopy(meta or {})
        p1 = self._coerce_xy_tuple(meta_copy.get("p1"))
        p2 = self._coerce_xy_tuple(meta_copy.get("p2"))
        if p1 is None or p2 is None:
            pts = list(fallback_points or [])
            if len(pts) >= 2:
                p1 = self._coerce_xy_tuple(pts[0])
                p2 = self._coerce_xy_tuple(pts[1])
        return self._line_meta_from_points(
            p1 or (0.0, 0.0),
            p2 or p1 or (0.0, 0.0),
            base_meta=meta_copy,
        )

    def _is_persistent_parametric_annotation(self, meta, dim_type):
        meta_type = str((meta or {}).get("type", "")).lower()
        dim_key = str(dim_type or "").lower()
        if meta_type == "line" and dim_key == "linear":
            return True
        if meta_type == "rectangle" and dim_key in {"rect_width", "rect_height"}:
            return True
        if meta_type == "circle" and dim_key in {"diameter", "radius"}:
            return True
        return False

    def _is_sketch_visible(self, owner_type, owner_part, sketch_index):
        try:
            meta = self._get_sketch_meta(owner_type, owner_part, sketch_index) or {}
        except Exception:
            meta = {}
        try:
            return bool(meta.get("visible", True))
        except Exception:
            return True

    def _find_dimension(self, dim_id):
        for dim in self.dimensions:
            if dim.get("id") == dim_id:
                return dim, "sketch", None
        for part in self.parts:
            for dim in getattr(part, "dimensions", []):
                if dim.get("id") == dim_id:
                    return dim, "part", part
        return None, None, None

    def _build_points_from_meta(self, meta, fallback_points=None):
        meta_type = str(meta.get("type", "polyline")).lower()
        if meta_type == "line":
            line_meta = self._normalize_line_meta(meta, fallback_points=fallback_points)
            p1 = line_meta.get("p1")
            p2 = line_meta.get("p2")
            if p1 and p2:
                return [tuple(p1), tuple(p2)]
            return fallback_points if fallback_points is not None else []
        if meta_type == "rectangle":
            rect_meta = self._normalize_rectangle_meta(meta, fallback_points=fallback_points)
            origin = rect_meta.get("origin", (0.0, 0.0))
            width = float(rect_meta.get("width", 0.0) or 0.0)
            height = float(rect_meta.get("height", 0.0) or 0.0)
            x0, y0 = origin
            return [
                (x0, y0),
                (x0 + width, y0),
                (x0 + width, y0 + height),
                (x0, y0 + height),
                (x0, y0),
            ]
        if meta_type == "circle":
            circle_meta = self._normalize_circle_meta(meta, fallback_points=fallback_points)
            center = circle_meta.get("center", (0.0, 0.0))
            radius = float(circle_meta.get("radius", 0.0) or 0.0)
            cx, cy = center
            if radius <= 0:
                return fallback_points if fallback_points is not None else []
            pts = [
                (cx + radius * math.cos(t), cy + radius * math.sin(t))
                for t in np.linspace(0, 2 * math.pi, 64, endpoint=False)
            ]
            pts.append(pts[0])
            return pts
        if meta_type == "arc":
            center = meta.get("center")
            radius = float(meta.get("radius", 0.0))
            a_start = meta.get("start_angle")
            a_end = meta.get("end_angle")
            if center is None or a_start is None or a_end is None or radius <= 0:
                return fallback_points if fallback_points is not None else []
            cx, cy = center
            if a_end <= a_start:
                a_end += 2 * math.pi
            steps = max(12, min(128, int(abs(a_end - a_start) * radius / 10)))
            pts = []
            for i in range(steps + 1):
                ang = a_start + (i / steps) * (a_end - a_start)
                pts.append((cx + radius * math.cos(ang), cy + radius * math.sin(ang)))
            return pts
        if meta_type == "slot":
            p1 = meta.get("p1")
            p2 = meta.get("p2")
            width = float(meta.get("width", 0.0))
            if p1 and p2 and width > 0:
                return self._build_slot_vertices(tuple(p1), tuple(p2), width)
            return fallback_points if fallback_points is not None else []
        if meta_type == "polygon":
            center = meta.get("center", (0.0, 0.0))
            radius = float(meta.get("radius", 0.0))
            sides = int(meta.get("sides", 3))
            angle = float(meta.get("angle", 0.0))
            if radius <= 0 or sides < 3:
                return fallback_points if fallback_points is not None else []
            cx, cy = center
            pts = []
            for i in range(sides):
                theta = angle + 2 * math.pi * i / sides
                pts.append((cx + radius * math.cos(theta), cy + radius * math.sin(theta)))
            pts.append(pts[0])
            return pts
        if meta_type == "polyline":
            return copy.deepcopy(meta.get("points", fallback_points if fallback_points is not None else []))
        return fallback_points if fallback_points is not None else []

    def _dedupe_stroke_points(self, points, tol=1e-6):
        out = []
        for p in (points if points is not None else []):
            try:
                pt = (float(p[0]), float(p[1]))
            except Exception:
                continue
            if not out:
                out.append(pt)
                continue
            try:
                if dist(out[-1], pt) <= float(tol):
                    continue
            except Exception:
                if out[-1] == pt:
                    continue
            out.append(pt)
        return out

    def _stroke_path_length(self, points):
        pts = points if points is not None else []
        if len(pts) < 2:
            return 0.0
        total = 0.0
        for i in range(len(pts) - 1):
            try:
                total += float(dist(pts[i], pts[i + 1]))
            except Exception:
                continue
        return float(total)

    def _prepare_freeform_stroke(self, points):
        pts = self._dedupe_stroke_points(points, tol=1e-6)
        if len(pts) < 2:
            return pts, False
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        path_len = self._stroke_path_length(pts)
        end_gap = float(dist(pts[0], pts[-1]))
        # Prefer loop-closure from traveled path ratio so rough freehand loops still close
        # even if first/last points are not perfectly snapped.
        close_tol_geom = max(float(SNAP_TOL), 0.03 * float(diag))
        close_tol_path = 0.12 * float(path_len)
        close_tol = max(close_tol_geom, close_tol_path)
        try:
            closed = bool(
                len(pts) >= 3
                and end_gap <= close_tol
                and path_len >= max(1.2 * end_gap, 0.50 * float(diag))
            )
        except Exception:
            closed = False
        if closed:
            if dist(pts[0], pts[-1]) <= 1e-9:
                pts[-1] = pts[0]
            else:
                pts.append(pts[0])
        return pts, closed

    def _fit_circle_kasa(self, points):
        pts = np.asarray(points if points is not None else [], dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
            return None
        x = pts[:, 0]
        y = pts[:, 1]
        A = np.column_stack((2.0 * x, 2.0 * y, np.ones(len(pts), dtype=float)))
        b = x * x + y * y
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            return None
        cx = float(sol[0])
        cy = float(sol[1])
        c0 = float(sol[2])
        rr = c0 + cx * cx + cy * cy
        if not math.isfinite(rr) or rr <= 1e-12:
            return None
        radius = math.sqrt(rr)
        if not math.isfinite(radius) or radius <= 1e-9:
            return None
        radii = np.hypot(x - cx, y - cy)
        residuals = np.abs(radii - radius)
        res_mean = float(np.mean(residuals)) if residuals.size else 0.0
        res_max = float(np.max(residuals)) if residuals.size else 0.0
        res_rel = res_mean / max(radius, 1e-9)
        return {
            "center": (cx, cy),
            "radius": float(radius),
            "res_mean": res_mean,
            "res_max": res_max,
            "res_rel": float(res_rel),
        }

    def _infer_arc_from_open_stroke(self, points):
        pts = self._dedupe_stroke_points(points, tol=1e-6)
        if len(pts) < 4:
            return None
        fit = self._fit_circle_kasa(pts)
        if not fit:
            return None
        cx, cy = fit["center"]
        r = float(fit["radius"])
        res_rel = float(fit.get("res_rel", 1.0))
        if not math.isfinite(r) or r <= 1e-9 or not math.isfinite(res_rel):
            return None
        if res_rel > 0.18:
            return None

        p_arr = np.asarray(pts, dtype=float)
        try:
            angles = np.unwrap(np.arctan2(p_arr[:, 1] - cy, p_arr[:, 0] - cx))
        except Exception:
            return None
        if angles.size < 2:
            return None
        delta = float(angles[-1] - angles[0])
        span = abs(delta)
        if span < math.radians(10.0):
            return None

        end_gap = float(dist(pts[0], pts[-1]))
        # Very long near-closed open stroke is better represented as a circle.
        if span >= math.radians(300.0) and end_gap <= 0.95 * r and res_rel <= 0.14:
            return self._circle_meta_from_center_radius(
                (cx, cy),
                r,
                base_meta={
                    "source_tool": "freeform",
                    "auto_converted": True,
                },
            )

        if span >= math.radians(355.0):
            return None

        if delta >= 0.0:
            a0 = float(angles[0])
            a1 = float(angles[-1])
        else:
            # Arc drawer is CCW; swap endpoints for CW strokes.
            a0 = float(angles[-1])
            a1 = float(angles[0])
        while a1 <= a0:
            a1 += 2.0 * math.pi
        if (a1 - a0) < math.radians(10.0) or (a1 - a0) >= math.radians(355.0):
            return None
        return {
            "type": "arc",
            "center": (cx, cy),
            "radius": r,
            "start_angle": a0,
            "end_angle": a1,
            "source_tool": "freeform",
            "auto_converted": True,
        }

    def _infer_freeform_shape_meta(self, points):
        pts, is_closed = self._prepare_freeform_stroke(points)
        if len(pts) < 2:
            return None, pts

        # Open freeform stroke: optionally collapse near-straight scribbles to a line.
        if not is_closed:
            p1 = pts[0]
            p2 = pts[-1]
            length = dist(p1, p2)
            if length > 1e-9 and len(pts) >= 3:
                devs = [point_line_dist(p, p1, p2) for p in pts]
                max_dev = max(devs) if devs else 0.0
                mean_dev = float(sum(devs) / len(devs)) if devs else 0.0
                score = (max_dev / length) + 0.5 * (mean_dev / length)
                if score <= 0.06:
                    meta = self._line_meta_from_points(
                        p1,
                        p2,
                        base_meta={
                            "source_tool": "freeform",
                            "auto_converted": True,
                        },
                    )
                    return meta, self._build_points_from_meta(meta, pts)
            arc_meta = self._infer_arc_from_open_stroke(pts)
            if arc_meta is not None:
                return arc_meta, self._build_points_from_meta(arc_meta, pts)
            return None, pts

        if len(pts) < 4:
            return None, pts

        try:
            poly = Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
        except Exception:
            poly = None
        if poly is None or getattr(poly, "is_empty", True):
            return None, pts
        if getattr(poly, "geom_type", "") == "MultiPolygon":
            try:
                geoms = list(getattr(poly, "geoms", []))
                poly = max(geoms, key=lambda g: float(getattr(g, "area", 0.0) or 0.0))
            except Exception:
                return None, pts
        if poly is None or getattr(poly, "is_empty", True) or getattr(poly, "geom_type", "") != "Polygon":
            return None, pts

        area = float(abs(getattr(poly, "area", 0.0) or 0.0))
        if area <= 1e-9:
            return None, pts
        minx, miny, maxx, maxy = poly.bounds
        width = float(maxx - minx)
        height = float(maxy - miny)
        diag = math.hypot(width, height)
        base_pts = pts[:-1] if len(pts) >= 2 else pts
        if not base_pts:
            return None, pts

        candidates = []

        def _add_candidate(score, meta):
            try:
                score = float(score)
            except Exception:
                return
            if not math.isfinite(score):
                return
            verts = self._build_points_from_meta(meta, base_pts)
            if len(verts) < 2:
                return
            candidates.append((score, meta, verts))

        try:
            centroid = poly.centroid
            cx = float(centroid.x)
            cy = float(centroid.y)
        except Exception:
            cx = float(sum(p[0] for p in base_pts) / len(base_pts))
            cy = float(sum(p[1] for p in base_pts) / len(base_pts))

        # Circle candidate
        try:
            radii = [dist((cx, cy), p) for p in base_pts]
            r_mean = float(sum(radii) / len(radii)) if radii else 0.0
            if r_mean > 1e-9:
                rel_std = float(np.std(radii) / max(r_mean, 1e-9))
                area_ref = math.pi * r_mean * r_mean
                area_err = abs(area - area_ref) / max(area_ref, 1e-9)
                score = rel_std + 0.65 * area_err
                if rel_std <= 0.28 and area_err <= 0.35:
                    _add_candidate(
                        score,
                        self._circle_meta_from_center_radius(
                            (cx, cy),
                            r_mean,
                            base_meta={
                                "source_tool": "freeform",
                                "auto_converted": True,
                            },
                        ),
                    )
        except Exception:
            pass

        # Axis-aligned rectangle candidate
        if width > 1e-9 and height > 1e-9:
            try:
                rect_area = width * height
                fill_ratio = area / max(rect_area, 1e-9)
                min_dim = max(min(width, height), 1e-9)
                edge_ds = [
                    min(abs(p[0] - minx), abs(p[0] - maxx), abs(p[1] - miny), abs(p[1] - maxy))
                    for p in base_pts
                ]
                mean_edge = float(sum(edge_ds) / len(edge_ds)) / min_dim if edge_ds else 1.0
                corner_tol = 0.22 * min_dim
                corners = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
                corner_hits = 0
                for c in corners:
                    if any(dist(c, p) <= corner_tol for p in base_pts):
                        corner_hits += 1
                score = (1.0 - min(1.0, fill_ratio)) + mean_edge
                if fill_ratio >= 0.70 and mean_edge <= 0.18 and corner_hits >= 3:
                    _add_candidate(
                        score,
                        self._rectangle_meta_from_origin_size(
                            (minx, miny),
                            width,
                            height,
                            mode="two_corner",
                            base_meta={
                                "source_tool": "freeform",
                                "auto_converted": True,
                            },
                        ),
                    )
            except Exception:
                pass

        # Slot/capsule candidate
        try:
            mrr = poly.minimum_rotated_rectangle
            mrr_pts = self._dedupe_stroke_points(list(getattr(mrr.exterior, "coords", [])))
            if len(mrr_pts) >= 4:
                if len(mrr_pts) >= 2 and dist(mrr_pts[0], mrr_pts[-1]) <= 1e-9:
                    mrr_pts = mrr_pts[:-1]
                edges = []
                for i in range(len(mrr_pts)):
                    a = mrr_pts[i]
                    b = mrr_pts[(i + 1) % len(mrr_pts)]
                    L = dist(a, b)
                    if L <= 1e-9:
                        continue
                    direction = ((b[0] - a[0]) / L, (b[1] - a[1]) / L)
                    edges.append((L, direction))
                if len(edges) >= 2:
                    major = max(e[0] for e in edges)
                    minor = min(e[0] for e in edges)
                    if minor > 1e-9 and major / minor >= 1.4:
                        width_guess = float(minor)
                        core_len = float(major - width_guess)
                        if core_len > 0.15 * width_guess:
                            long_len, long_dir = max(edges, key=lambda item: item[0])
                            c = mrr.centroid
                            cx_mrr = float(c.x)
                            cy_mrr = float(c.y)
                            half = 0.5 * core_len
                            p1 = (cx_mrr - long_dir[0] * half, cy_mrr - long_dir[1] * half)
                            p2 = (cx_mrr + long_dir[0] * half, cy_mrr + long_dir[1] * half)
                            slot_pts = self._build_slot_vertices(p1, p2, width_guess)
                            slot_poly = Polygon(slot_pts) if len(slot_pts) >= 4 else None
                            if slot_poly is not None and slot_poly.is_valid and not slot_poly.is_empty:
                                sym_ratio = float(poly.symmetric_difference(slot_poly).area) / max(area, 1e-9)
                                if sym_ratio <= 0.42:
                                    _add_candidate(
                                        sym_ratio + 0.04,
                                        {
                                            "type": "slot",
                                            "p1": p1,
                                            "p2": p2,
                                            "width": width_guess,
                                            "source_tool": "freeform",
                                            "auto_converted": True,
                                        },
                                    )
        except Exception:
            pass

        # Regular polygon candidate from simplified corners.
        try:
            ring = LineString(pts)
            simp_tol = max(0.02 * diag, 0.5 * float(SNAP_TOL))
            simple = ring.simplify(simp_tol, preserve_topology=False)
            simp_pts = self._dedupe_stroke_points(list(getattr(simple, "coords", [])))
            if len(simp_pts) >= 4 and dist(simp_pts[0], simp_pts[-1]) <= max(float(SNAP_TOL), 0.03 * diag):
                if dist(simp_pts[0], simp_pts[-1]) > 1e-9:
                    simp_pts.append(simp_pts[0])
                verts = simp_pts[:-1]
                n = len(verts)
                if 3 <= n <= 12:
                    rvals = [dist((cx, cy), p) for p in verts]
                    r_mean = float(sum(rvals) / len(rvals)) if rvals else 0.0
                    if r_mean > 1e-9:
                        rel_r = float(np.std(rvals) / max(r_mean, 1e-9))
                        e_vals = [dist(verts[i], verts[(i + 1) % n]) for i in range(n)]
                        e_mean = float(sum(e_vals) / len(e_vals)) if e_vals else 0.0
                        rel_e = float(np.std(e_vals) / max(e_mean, 1e-9)) if e_mean > 1e-9 else 1.0
                        area_ref = 0.5 * n * r_mean * r_mean * math.sin((2.0 * math.pi) / n)
                        area_err = abs(area - area_ref) / max(area_ref, 1e-9)
                        score = 0.45 * rel_r + 0.35 * rel_e + 0.20 * area_err + 0.02 * max(0, n - 3)
                        if rel_r <= 0.32 and rel_e <= 0.32 and area_err <= 0.45:
                            angle = math.atan2(verts[0][1] - cy, verts[0][0] - cx)
                            _add_candidate(
                                score,
                                {
                                    "type": "polygon",
                                    "center": (cx, cy),
                                    "radius": r_mean,
                                    "sides": int(n),
                                    "angle": float(angle),
                                    "source_tool": "freeform",
                                    "auto_converted": True,
                                },
                            )
        except Exception:
            pass

        if not candidates:
            return None, pts

        best_score, best_meta, best_pts = min(candidates, key=lambda item: item[0])
        if best_score > 0.75:
            return None, pts
        return copy.deepcopy(best_meta), best_pts

    def _update_sketch_from_meta(self, owner_type, owner_part, sketch_index, meta):
        sketches, metas, _, _ = self._owner_collections(owner_type, owner_part)
        if sketch_index < 0 or sketch_index >= len(sketches):
            return False
        raw_meta = copy.deepcopy(meta or {})
        points = self._build_points_from_meta(raw_meta, sketches[sketch_index])
        if str(raw_meta.get("type", "polyline")).lower() == "rectangle":
            stored_meta = self._normalize_rectangle_meta(raw_meta, fallback_points=points)
        else:
            stored_meta = copy.deepcopy(raw_meta)
        sketches[sketch_index] = points
        if sketch_index < len(metas):
            metas[sketch_index] = stored_meta
        return True

    def _segment_points(self, points, segment_index):
        if points is None or len(points) == 0 or segment_index < 0 or segment_index >= len(points) - 1:
            return None, None
        return points[segment_index], points[segment_index + 1]

    def _dimension_value(self, dim, owner_type, owner_part):
        dim_type = dim.get("dim_type")
        sketch_index = dim.get("sketch_index", -1)
        meta = self._get_sketch_meta(owner_type, owner_part, sketch_index)
        meta_type = str(meta.get("type", "polyline")).lower()
        points = self._get_sketch_points(owner_type, owner_part, sketch_index)
        if points is None:
            points = []
        seg_index = dim.get("segment_index", 0)
        if dim_type in ("rect_width", "rect_height"):
            if dim_type == "rect_width":
                return abs(float(meta.get("width", 0.0)))
            return abs(float(meta.get("height", 0.0)))
        if dim_type == "diameter":
            radius = float(meta.get("radius", 0.0))
            return 2.0 * radius
        if dim_type == "radius":
            return abs(float(meta.get("radius", 0.0)))
        if dim_type == "slot_length":
            p1 = meta.get("p1")
            p2 = meta.get("p2")
            if p1 and p2:
                return dist(p1, p2)
            return 0.0
        if dim_type == "slot_width":
            return abs(float(meta.get("width", 0.0)))
        if dim_type == "polygon_radius":
            return abs(float(meta.get("radius", 0.0)))
        if dim_type == "angle":
            other_idx = dim.get("other_segment_index")
            p1, p2 = self._segment_points(points, seg_index)
            q1, q2 = self._segment_points(points, other_idx)
            if not p1 or not p2 or not q1 or not q2:
                return 0.0
            v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=float)
            v2 = np.array([q2[0] - q1[0], q2[1] - q1[1]], dtype=float)
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 <= 1e-9 or n2 <= 1e-9:
                return 0.0
            dot_val = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            return math.degrees(math.acos(dot_val))
        if dim_type == "linear":
            if meta_type == "rectangle":
                p1, p2 = self._segment_points(points, seg_index)
                if not p1 or not p2:
                    return 0.0
                if abs(p2[0] - p1[0]) >= abs(p2[1] - p1[1]):
                    return abs(float(meta.get("width", 0.0)))
                return abs(float(meta.get("height", 0.0)))
            p1, p2 = self._segment_points(points, seg_index)
            if not p1 or not p2:
                return 0.0
            return dist(p1, p2)
        return 0.0

    def _dimension_label_pos(self, dim, owner_type, owner_part):
        dim_type = dim.get("dim_type")
        sketch_index = dim.get("sketch_index", -1)
        points = self._get_sketch_points(owner_type, owner_part, sketch_index)
        if points is None:
            points = []
        meta = self._get_sketch_meta(owner_type, owner_part, sketch_index)
        seg_index = dim.get("segment_index", 0)
        offset = float(dim.get("offset", self._dim_offset))
        offset_dir = dim.get("offset_dir")
        if dim_type in ("linear", "rect_width", "rect_height"):
            p1, p2 = self._segment_points(points, seg_index)
            if not p1 or not p2:
                return (0.0, 0.0)
            mid = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.hypot(dx, dy)
            if length <= 1e-9:
                return mid
            normal = (-dy / length, dx / length)
            if offset_dir:
                normal = offset_dir
            return (mid[0] + normal[0] * offset, mid[1] + normal[1] * offset)
        if dim_type in ("diameter", "radius"):
            center = meta.get("center", (0.0, 0.0))
            radius = abs(float(meta.get("radius", 0.0)))
            dir_vec = offset_dir or (1.0, 0.0)
            return (
                center[0] + dir_vec[0] * (radius + offset),
                center[1] + dir_vec[1] * (radius + offset),
            )
        if dim_type in ("slot_length", "slot_width"):
            p1 = meta.get("p1")
            p2 = meta.get("p2")
            if not p1 or not p2:
                return (0.0, 0.0)
            mid = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.hypot(dx, dy)
            if length <= 1e-9:
                return mid
            normal = (-dy / length, dx / length)
            if dim_type == "slot_length":
                if offset_dir:
                    normal = offset_dir
                return (mid[0] + normal[0] * offset, mid[1] + normal[1] * offset)
            if offset_dir:
                normal = offset_dir
            return (mid[0] + normal[0] * (float(meta.get("width", 0.0)) + offset), mid[1] + normal[1] * (float(meta.get("width", 0.0)) + offset))
        if dim_type == "polygon_radius":
            center = meta.get("center", (0.0, 0.0))
            radius = abs(float(meta.get("radius", 0.0)))
            angle = float(meta.get("angle", 0.0))
            return (
                center[0] + math.cos(angle) * (radius + offset),
                center[1] + math.sin(angle) * (radius + offset),
            )
        if dim_type == "angle":
            p1, p2 = self._segment_points(points, seg_index)
            if not p1 or not p2:
                return (0.0, 0.0)
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.hypot(dx, dy)
            if length <= 1e-9:
                return p1
            normal = (dx / length, dy / length)
            return (p1[0] + normal[0] * offset, p1[1] + normal[1] * offset)
        return (0.0, 0.0)

    def _auto_fit_after_dimension_change(self, owner_type=None, owner_part=None):
        """Auto-fit the viewport around the geometry that was just dimensioned."""
        try:
            bounds = []

            if owner_type == "part" and owner_part is not None:
                self._append_geometry_bounds(bounds, getattr(owner_part, "geometry", None))
                sketches = list(getattr(owner_part, "sketches", []) or [])
                if not bounds:
                    for idx, sketch in enumerate(sketches):
                        if self._is_sketch_visible("part", owner_part, idx):
                            self._append_points_bounds(bounds, sketch)
                if not bounds:
                    for sketch in sketches:
                        self._append_points_bounds(bounds, sketch)
            else:
                sketches = list(getattr(self, "sketches", []) or [])
                for idx, sketch in enumerate(sketches):
                    if self._is_sketch_visible("sketch", None, idx):
                        self._append_points_bounds(bounds, sketch)
                if not bounds:
                    for sketch in sketches:
                        self._append_points_bounds(bounds, sketch)

            self._append_points_bounds(bounds, getattr(self, "current", None))

            rect = self._fit_rect_from_bounds(bounds) if bounds else None
            if rect is None:
                rect = self._model_fit_rect()
            if rect is not None and self._apply_fit_rect(rect):
                return
        except Exception:
            pass

        try:
            self.fit_view()
        except Exception:
            pass

    def get_active_sketch_dimensions_summary(self):
        """Get a summary of all dimensions in the active sketch (SolidWorks-style display)."""
        dimensions_info = []
        owner_type, owner_part = self._active_owner()
        
        if owner_type is None:
            return dimensions_info
        
        _, _, dimensions, _ = self._owner_collections(owner_type, owner_part)
        
        for dim in dimensions:
            try:
                dim_id = dim.get("id")
                dim_type = dim.get("dim_type", "unknown")
                sketch_index = int(dim.get("sketch_index", -1))
                
                current_value = self._dimension_current_value(dim, owner_type, owner_part) if hasattr(self, "_dimension_current_value") else 0.0
                
                # Build descriptive name
                if dim_type == "linear":
                    name = "Length"
                elif dim_type == "rect_width":
                    name = "Width"
                elif dim_type == "rect_height":
                    name = "Height"
                elif dim_type == "diameter":
                    name = "Diameter"
                elif dim_type == "radius":
                    name = "Radius"
                elif dim_type == "angle":
                    name = "Angle"
                elif dim_type == "point_distance":
                    name = "Distance"
                elif dim_type == "slot_length":
                    name = "Slot Length"
                elif dim_type == "slot_width":
                    name = "Slot Width"
                elif dim_type == "polygon_radius":
                    name = "Polygon Radius"
                elif dim_type == "arc_length":
                    name = "Arc Length"
                else:
                    name = str(dim_type).replace("_", " ").title()
                
                # Add unit for non-angle dimensions
                unit = "°" if dim_type == "angle" else f" {self.current_unit}"
                
                dimensions_info.append({
                    "id": dim_id,
                    "type": dim_type,
                    "name": name,
                    "value": current_value,
                    "display": f"{name}: {current_value:.2f}{unit}",
                    "sketch_index": sketch_index,
                })
            except Exception:
                continue
        
        return dimensions_info

    def _set_segment_length(self, owner_type, owner_part, sketch_index, seg_index, new_len):
        sketches, metas, _, _ = self._owner_collections(owner_type, owner_part)
        if sketch_index < 0 or sketch_index >= len(sketches):
            return False
        points = sketches[sketch_index]
        p1, p2 = self._segment_points(points, seg_index)
        if not p1 or not p2:
            return False
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            dx, dy, length = 1.0, 0.0, 1.0
        ux, uy = dx / length, dy / length
        new_p2 = (p1[0] + ux * new_len, p1[1] + uy * new_len)
        points[seg_index + 1] = new_p2
        if points is not None and len(points) > 0 and dist(points[0], points[-1]) <= 1e-6:
            points[-1] = points[0]
        sketches[sketch_index] = points
        meta = metas[sketch_index] if sketch_index < len(metas) else {}
        if meta.get("type") == "line":
            meta["p1"] = points[0]
            meta["p2"] = points[1]
            metas[sketch_index] = meta
        elif meta.get("type") == "polyline":
            meta["points"] = copy.deepcopy(points)
            metas[sketch_index] = meta
        return True

    def _set_segment_angle(self, owner_type, owner_part, sketch_index, seg_index, other_index, angle_deg):
        points = self._get_sketch_points(owner_type, owner_part, sketch_index)
        if points is None:
            points = []
        p1, p2 = self._segment_points(points, seg_index)
        q1, q2 = self._segment_points(points, other_index)
        if not p1 or not p2 or not q1 or not q2:
            return False
        v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=float)
        v2 = np.array([q2[0] - q1[0], q2[1] - q1[1]], dtype=float)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 <= 1e-9 or n2 <= 1e-9:
            return False
        dir1 = v1 / n1
        target_angle = math.radians(angle_deg)
        rot1 = np.array(
            [dir1[0] * math.cos(target_angle) - dir1[1] * math.sin(target_angle),
             dir1[0] * math.sin(target_angle) + dir1[1] * math.cos(target_angle)]
        )
        rot2 = np.array(
            [dir1[0] * math.cos(-target_angle) - dir1[1] * math.sin(-target_angle),
             dir1[0] * math.sin(-target_angle) + dir1[1] * math.cos(-target_angle)]
        )
        if np.dot(rot1, v2 / n2) < np.dot(rot2, v2 / n2):
            rot1 = rot2
        new_q2 = (q1[0] + rot1[0] * n2, q1[1] + rot1[1] * n2)
        points[other_index + 1] = new_q2
        if points is not None and len(points) > 0 and dist(points[0], points[-1]) <= 1e-6:
            points[-1] = points[0]
        sketches, metas, _, _ = self._owner_collections(owner_type, owner_part)
        sketches[sketch_index] = points
        meta = metas[sketch_index] if sketch_index < len(metas) else {}
        if meta.get("type") == "polyline":
            meta["points"] = copy.deepcopy(points)
            metas[sketch_index] = meta
        return True

    def _apply_constraints(self, owner_type, owner_part, sketch_index):
        sketches, metas, _, constraints = self._owner_collections(owner_type, owner_part)
        if sketch_index < 0 or sketch_index >= len(sketches):
            return
        points = sketches[sketch_index]
        meta = metas[sketch_index] if sketch_index < len(metas) else {}
        meta_type = str(meta.get("type", "polyline")).lower()
        if meta_type not in ("line", "polyline"):
            return
        for con in constraints:
            if con.get("sketch_index") != sketch_index:
                continue
            seg_index = con.get("segment_index", 0)
            p1, p2 = self._segment_points(points, seg_index)
            if not p1 or not p2:
                continue
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.hypot(dx, dy)
            if length <= 1e-9:
                continue
            ctype = con.get("type")
            if ctype == "horizontal":
                points[seg_index + 1] = (p2[0], p1[1])
            elif ctype == "vertical":
                points[seg_index + 1] = (p1[0], p2[1])
            elif ctype in ("parallel", "perpendicular", "equal"):
                other_index = con.get("other_segment_index")
                q1, q2 = self._segment_points(points, other_index)
                if not q1 or not q2:
                    continue
                v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=float)
                v2 = np.array([q2[0] - q1[0], q2[1] - q1[1]], dtype=float)
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 <= 1e-9 or n2 <= 1e-9:
                    continue
                dir1 = v1 / n1
                if ctype == "parallel":
                    new_dir = dir1
                    new_len = n2
                elif ctype == "perpendicular":
                    new_dir = np.array([-dir1[1], dir1[0]])
                    new_len = n2
                else:  # equal
                    new_dir = v2 / n2
                    new_len = n1
                new_q2 = (q1[0] + new_dir[0] * new_len, q1[1] + new_dir[1] * new_len)
                points[other_index + 1] = new_q2
        if points is not None and len(points) > 0 and dist(points[0], points[-1]) <= 1e-6:
            points[-1] = points[0]
        sketches[sketch_index] = points
        if meta.get("type") == "line":
            meta["p1"] = points[0]
            meta["p2"] = points[1]
            metas[sketch_index] = meta
        elif meta.get("type") == "polyline":
            meta["points"] = copy.deepcopy(points)
            metas[sketch_index] = meta

    def _find_nearest_segment(self, pt, owner_type, owner_part):
        sketches, metas, _, _ = self._owner_collections(owner_type, owner_part)
        best = None
        best_dist = SNAP_TOL * 1.5
        for s_idx, points in enumerate(sketches):
            if len(points) < 2:
                continue
            for i in range(len(points) - 1):
                d = point_line_dist(pt, points[i], points[i + 1])
                if d < best_dist:
                    best_dist = d
                    best = (s_idx, i, points[i], points[i + 1])
        if best is None:
            return None
        meta = metas[best[0]] if best[0] < len(metas) else {}
        return best + (meta,)

    def _handle_dimension_click(self, pt):
        owner_type, owner_part = self._active_owner()
        if owner_type is None:
            return
        if self._pending_dimension:
            pending = self._pending_dimension
            dim = pending["dim"]
            p1, p2 = pending["p1"], pending["p2"]
            owner_type = pending.get("owner_type", owner_type)
            owner_part = pending.get("owner_part", owner_part)
            mid = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.hypot(dx, dy)
            if length > 1e-9:
                normal = (-dy / length, dx / length)
            else:
                normal = (0.0, 1.0)
            vec = (pt[0] - mid[0], pt[1] - mid[1])
            mag = math.hypot(vec[0], vec[1])
            if mag > 1e-6:
                dim["offset"] = mag
                dim["offset_dir"] = (vec[0] / mag, vec[1] / mag)
            else:
                dim["offset"] = self._dim_offset
                dim["offset_dir"] = normal
            self.push_undo_state()
            _, _, dimensions, _ = self._owner_collections(owner_type, owner_part)
            dimensions.append(dim)
            self._pending_dimension = None
            self.redraw()
            return

        result = self._find_nearest_segment(pt, owner_type, owner_part)
        if not result:
            return
        sketch_index, seg_index, p1, p2, meta = result
        meta_type = str(meta.get("type", "polyline")).lower()

        if meta_type in ("circle", "arc"):
            dim_type = "diameter" if meta_type == "circle" else "radius"
            dim = {
                "id": self._next_dimension_id(),
                "dim_type": dim_type,
                "sketch_index": sketch_index,
                "segment_index": seg_index,
            }
            self.push_undo_state()
            _, _, dimensions, _ = self._owner_collections(owner_type, owner_part)
            dimensions.append(dim)
            self.redraw()
            return

        dim_kind = "Linear"
        if meta_type not in ("rectangle", "circle", "arc"):
            choice, ok = QInputDialog.getItem(
                self,
                "Dimension Type",
                "Choose dimension type:",
                ["Linear", "Angle"],
                0,
                False,
            )
            if not ok:
                return
            dim_kind = choice

        if dim_kind == "Angle":
            self._pending_dimension = {
                "mode": "angle",
                "owner_type": owner_type,
                "owner_part": owner_part,
                "sketch_index": sketch_index,
                "segment_index": seg_index,
            }
            return

        if meta_type == "rectangle":
            if abs(p2[0] - p1[0]) >= abs(p2[1] - p1[1]):
                dim_type = "rect_width"
            else:
                dim_type = "rect_height"
        else:
            dim_type = "linear"

        dim = {
            "id": self._next_dimension_id(),
            "dim_type": dim_type,
            "sketch_index": sketch_index,
            "segment_index": seg_index,
        }
        self._pending_dimension = {
            "dim": dim,
            "p1": p1,
            "p2": p2,
            "owner_type": owner_type,
            "owner_part": owner_part,
        }

    def _handle_angle_dimension_click(self, pt):
        pending = self._pending_dimension
        if not pending:
            return False
        owner_type = pending.get("owner_type")
        owner_part = pending.get("owner_part")
        sketch_index = pending.get("sketch_index")
        seg_index = pending.get("segment_index")
        result = self._find_nearest_segment(pt, owner_type, owner_part)
        if not result:
            return False
        other_index = result[1]
        if other_index == seg_index:
            return False
        dim = {
            "id": self._next_dimension_id(),
            "dim_type": "angle",
            "sketch_index": sketch_index,
            "segment_index": seg_index,
            "other_segment_index": other_index,
        }
        self.push_undo_state()
        _, _, dimensions, _ = self._owner_collections(owner_type, owner_part)
        dimensions.append(dim)
        self._pending_dimension = None
        self.redraw()
        return True

    def _handle_constraint_click(self, pt):
        owner_type, owner_part = self._active_owner()
        if owner_type is None:
            return
        if self._pending_constraint:
            pending = self._pending_constraint
            result = self._find_nearest_segment(pt, owner_type, owner_part)
            if not result:
                return
            other_index = result[1]
            constraint = pending["constraint"]
            constraint["other_segment_index"] = other_index
            self.push_undo_state()
            _, _, _, constraints = self._owner_collections(owner_type, owner_part)
            constraints.append(constraint)
            self._apply_constraints(owner_type, owner_part, constraint["sketch_index"])
            if owner_type == "part" and owner_part is not None:
                self._update_part_geometry_from_sketches(owner_part)
                self.rebuild_display_geometry()
                self.partsChanged.emit()
            self._pending_constraint = None
            self.redraw()
            return

        result = self._find_nearest_segment(pt, owner_type, owner_part)
        if not result:
            return
        sketch_index, seg_index, p1, p2, _ = result
        options = ["Horizontal", "Vertical", "Parallel", "Perpendicular", "Equal Length"]
        choice, ok = QInputDialog.getItem(
            self,
            "Constraint",
            "Choose constraint:",
            options,
            0,
            False,
        )
        if not ok:
            return
        ctype_map = {
            "Horizontal": "horizontal",
            "Vertical": "vertical",
            "Parallel": "parallel",
            "Perpendicular": "perpendicular",
            "Equal Length": "equal",
        }
        ctype = ctype_map.get(choice)
        constraint = {
            "type": ctype,
            "sketch_index": sketch_index,
            "segment_index": seg_index,
        }
        if ctype in ("parallel", "perpendicular", "equal"):
            self._pending_constraint = {"constraint": constraint}
            return
        self.push_undo_state()
        _, _, _, constraints = self._owner_collections(owner_type, owner_part)
        constraints.append(constraint)
        self._apply_constraints(owner_type, owner_part, sketch_index)
        if owner_type == "part" and owner_part is not None:
            self._update_part_geometry_from_sketches(owner_part)
            self.rebuild_display_geometry()
            self.partsChanged.emit()
        self.redraw()

    def _find_nearest_segment(self, owner_type, owner_part, pt, tol=SNAP_TOL):
        sketches, _, _, _ = self._owner_collections(owner_type, owner_part)
        best = None
        best_dist = tol
        for si, sketch in enumerate(sketches):
            if not self._is_sketch_visible(owner_type, owner_part, si):
                continue
            if len(sketch) < 2:
                continue
            for seg_idx in range(len(sketch) - 1):
                d = point_line_dist(pt, sketch[seg_idx], sketch[seg_idx + 1])
                if d <= best_dist:
                    best_dist = d
                    best = (si, seg_idx)
        return best

    def _find_nearest_vertex(self, owner_type, owner_part, pt, tol=SNAP_TOL):
        sketches, _, _, _ = self._owner_collections(owner_type, owner_part)
        best = None
        best_dist = tol
        for si, sketch in enumerate(sketches):
            if not self._is_sketch_visible(owner_type, owner_part, si):
                continue
            for vi, p in enumerate(sketch):
                d = dist(pt, p)
                if d <= best_dist:
                    best_dist = d
                    best = (si, vi)
        return best

    def _reset_sketch_auto_annotations(self, owner_type, owner_part, sketch_index):
        """Rebuild auto dimensions/constraints for a single sketch after shape replacement."""
        sketches, metas, dimensions, constraints = self._owner_collections(owner_type, owner_part)
        if sketch_index < 0 or sketch_index >= len(sketches):
            return False

        dimensions[:] = [
            d for d in dimensions
            if int(d.get("sketch_index", -1)) != int(sketch_index)
        ]
        constraints[:] = [
            c for c in constraints
            if int(c.get("sketch_index", -1)) != int(sketch_index)
            and int(c.get("other_sketch_index", -1)) != int(sketch_index)
        ]

        meta = metas[sketch_index] if sketch_index < len(metas) else {}
        self._auto_create_dimensions(owner_type, owner_part, sketch_index, meta)
        self._auto_create_constraints(owner_type, owner_part, sketch_index, meta)
        return True

    def _sketch_shape_edit_defaults(self, meta, points):
        pts = [tuple(p) for p in (points if points is not None else [])]
        if len(pts) >= 2 and dist(pts[0], pts[-1]) <= 1e-9:
            base_pts = pts[:-1]
        else:
            base_pts = pts
        if not base_pts:
            base_pts = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

        xs = [float(p[0]) for p in base_pts]
        ys = [float(p[1]) for p in base_pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        bbox_w = max(1.0, max_x - min_x)
        bbox_h = max(1.0, max_y - min_y)
        cx = float(sum(xs) / len(xs))
        cy = float(sum(ys) / len(ys))
        radius_guess = max(
            1.0,
            max((dist((cx, cy), p) for p in base_pts), default=0.0),
            0.5 * min(bbox_w, bbox_h),
        )
        p1_guess = (min_x, cy)
        p2_guess = (max_x, cy if bbox_w > 1e-9 else cy + bbox_h)

        def _f2(value, default=0.0):
            try:
                return float(value)
            except Exception:
                return float(default)

        meta = meta or {}
        meta_type = str(meta.get("type", "polyline")).lower()
        defaults = {
            "line": {
                "p1x": _f2(p1_guess[0]),
                "p1y": _f2(p1_guess[1]),
                "p2x": _f2(p2_guess[0]),
                "p2y": _f2(p2_guess[1]),
            },
            "rectangle": {
                "cx": _f2((min_x + max_x) * 0.5),
                "cy": _f2((min_y + max_y) * 0.5),
                "width": _f2(bbox_w, 10.0),
                "height": _f2(bbox_h, 10.0),
            },
            "circle": {
                "cx": _f2(cx),
                "cy": _f2(cy),
                "radius": _f2(radius_guess, 10.0),
            },
            "slot": {
                "p1x": _f2(p1_guess[0]),
                "p1y": _f2(p1_guess[1]),
                "p2x": _f2(p2_guess[0]),
                "p2y": _f2(p2_guess[1]),
                "width": _f2(max(1.0, 0.3 * min(bbox_w, bbox_h)), 3.0),
            },
            "polygon": {
                "cx": _f2(cx),
                "cy": _f2(cy),
                "radius": _f2(radius_guess, 10.0),
                "sides": max(3, int(meta.get("sides", 6) or 6)),
                "angle_deg": math.degrees(_f2(meta.get("angle", 0.0), 0.0)),
            },
        }

        if meta_type == "line":
            line_meta = self._normalize_line_meta(meta, fallback_points=points)
            p1 = line_meta.get("p1")
            p2 = line_meta.get("p2")
            if p1 and p2:
                defaults["line"] = {
                    "p1x": _f2(p1[0]),
                    "p1y": _f2(p1[1]),
                    "p2x": _f2(p2[0]),
                    "p2y": _f2(p2[1]),
                }
        elif meta_type == "rectangle":
            rect_meta = self._normalize_rectangle_meta(meta, fallback_points=points)
            origin = rect_meta.get("origin", (min_x, min_y))
            w = abs(_f2(rect_meta.get("width", bbox_w), bbox_w))
            h = abs(_f2(rect_meta.get("height", bbox_h), bbox_h))
            ox = _f2(origin[0] if isinstance(origin, (list, tuple)) and len(origin) >= 2 else min_x, min_x)
            oy = _f2(origin[1] if isinstance(origin, (list, tuple)) and len(origin) >= 2 else min_y, min_y)
            defaults["rectangle"] = {
                "cx": ox + 0.5 * max(0.001, w),
                "cy": oy + 0.5 * max(0.001, h),
                "width": max(0.001, w),
                "height": max(0.001, h),
            }
        elif meta_type == "circle":
            circle_meta = self._normalize_circle_meta(meta, fallback_points=points)
            center = circle_meta.get("center", (cx, cy))
            defaults["circle"] = {
                "cx": _f2(center[0] if isinstance(center, (list, tuple)) and len(center) >= 2 else cx, cx),
                "cy": _f2(center[1] if isinstance(center, (list, tuple)) and len(center) >= 2 else cy, cy),
                "radius": max(0.001, abs(_f2(circle_meta.get("radius", radius_guess), radius_guess))),
            }
        elif meta_type == "slot":
            p1 = meta.get("p1")
            p2 = meta.get("p2")
            if p1 and p2:
                defaults["slot"].update(
                    {
                        "p1x": _f2(p1[0]),
                        "p1y": _f2(p1[1]),
                        "p2x": _f2(p2[0]),
                        "p2y": _f2(p2[1]),
                    }
                )
            defaults["slot"]["width"] = max(
                0.001,
                abs(_f2(meta.get("width", defaults["slot"]["width"]), defaults["slot"]["width"]))
            )
        elif meta_type == "polygon":
            center = meta.get("center", (cx, cy))
            defaults["polygon"] = {
                "cx": _f2(center[0] if isinstance(center, (list, tuple)) and len(center) >= 2 else cx, cx),
                "cy": _f2(center[1] if isinstance(center, (list, tuple)) and len(center) >= 2 else cy, cy),
                "radius": max(0.001, abs(_f2(meta.get("radius", radius_guess), radius_guess))),
                "sides": max(3, int(meta.get("sides", 6) or 6)),
                "angle_deg": math.degrees(_f2(meta.get("angle", 0.0), 0.0)),
            }

        supported = {"line", "rectangle", "circle", "slot", "polygon"}
        initial_type = meta_type if meta_type in supported else "polygon"
        return {
            "meta_type": meta_type,
            "initial_type": initial_type,
            "supported_current": meta_type in supported,
            "values": defaults,
        }

    def _prompt_sketch_shape_meta_edit(self, meta, points):
        info = self._sketch_shape_edit_defaults(meta, points)
        labels = {
            "line": "Line",
            "rectangle": "Rectangle",
            "circle": "Circle",
            "slot": "Slot",
            "polygon": "Polygon",
        }
        type_order = ["line", "rectangle", "circle", "slot", "polygon"]
        unit = str(getattr(self, "current_unit", "") or "").strip()
        unit_suffix = f" ({unit})" if unit else ""

        dlg = QDialog(self)
        dlg.setWindowTitle("Edit Sketch Shape")
        layout = QVBoxLayout(dlg)

        if not info.get("supported_current", False):
            note = QLabel(
                f"Current shape type '{info.get('meta_type', 'shape')}' is not directly parametric.\n"
                "Replace it with a supported shape below."
            )
        else:
            note = QLabel("Edit parameters or replace this sketch shape with another parametric shape.")
        note.setWordWrap(True)
        layout.addWidget(note)

        top_form = QFormLayout()
        type_combo = QComboBox()
        for key in type_order:
            type_combo.addItem(labels[key], key)
        try:
            type_combo.setCurrentIndex(type_order.index(info["initial_type"]))
        except Exception:
            type_combo.setCurrentIndex(0)
        top_form.addRow("Shape type", type_combo)
        layout.addLayout(top_form)

        def dspin(value, minimum=-1e9, maximum=1e9, decimals=3):
            w = QDoubleSpinBox()
            w.setDecimals(decimals)
            w.setRange(float(minimum), float(maximum))
            w.setValue(float(value))
            return w

        def ispin(value, minimum=3, maximum=360):
            w = QSpinBox()
            w.setRange(int(minimum), int(maximum))
            w.setValue(int(value))
            return w

        def build_page(specs):
            page = QWidget()
            form = QFormLayout(page)
            refs = {}
            for spec in specs:
                if spec.get("kind") == "int":
                    w = ispin(spec["value"], spec.get("min", 3), spec.get("max", 360))
                else:
                    w = dspin(
                        spec["value"],
                        spec.get("min", -1e9),
                        spec.get("max", 1e9),
                        spec.get("decimals", 3),
                    )
                refs[spec["name"]] = w
                form.addRow(spec["label"], w)
            return page, refs

        vals = info["values"]
        page_specs = {
            "line": [
                {"name": "p1x", "label": f"Start X{unit_suffix}", "value": vals["line"]["p1x"]},
                {"name": "p1y", "label": f"Start Y{unit_suffix}", "value": vals["line"]["p1y"]},
                {"name": "p2x", "label": f"End X{unit_suffix}", "value": vals["line"]["p2x"]},
                {"name": "p2y", "label": f"End Y{unit_suffix}", "value": vals["line"]["p2y"]},
            ],
            "rectangle": [
                {"name": "cx", "label": f"Center X{unit_suffix}", "value": vals["rectangle"]["cx"]},
                {"name": "cy", "label": f"Center Y{unit_suffix}", "value": vals["rectangle"]["cy"]},
                {"name": "width", "label": f"Width{unit_suffix}", "value": vals["rectangle"]["width"], "min": 0.001},
                {"name": "height", "label": f"Height{unit_suffix}", "value": vals["rectangle"]["height"], "min": 0.001},
            ],
            "circle": [
                {"name": "cx", "label": f"Center X{unit_suffix}", "value": vals["circle"]["cx"]},
                {"name": "cy", "label": f"Center Y{unit_suffix}", "value": vals["circle"]["cy"]},
                {"name": "radius", "label": f"Radius{unit_suffix}", "value": vals["circle"]["radius"], "min": 0.001},
            ],
            "slot": [
                {"name": "p1x", "label": f"Start X{unit_suffix}", "value": vals["slot"]["p1x"]},
                {"name": "p1y", "label": f"Start Y{unit_suffix}", "value": vals["slot"]["p1y"]},
                {"name": "p2x", "label": f"End X{unit_suffix}", "value": vals["slot"]["p2x"]},
                {"name": "p2y", "label": f"End Y{unit_suffix}", "value": vals["slot"]["p2y"]},
                {"name": "width", "label": f"Width{unit_suffix}", "value": vals["slot"]["width"], "min": 0.001},
            ],
            "polygon": [
                {"name": "cx", "label": f"Center X{unit_suffix}", "value": vals["polygon"]["cx"]},
                {"name": "cy", "label": f"Center Y{unit_suffix}", "value": vals["polygon"]["cy"]},
                {"name": "radius", "label": f"Radius{unit_suffix}", "value": vals["polygon"]["radius"], "min": 0.001},
                {"name": "sides", "label": "Sides", "value": vals["polygon"]["sides"], "kind": "int", "min": 3, "max": 360},
                {"name": "angle_deg", "label": "Rotation (deg)", "value": vals["polygon"]["angle_deg"], "min": -360.0, "max": 360.0},
            ],
        }

        stack = QStackedWidget()
        pages = {}
        refs_by_type = {}
        for key in type_order:
            page, refs = build_page(page_specs[key])
            pages[key] = page
            refs_by_type[key] = refs
            stack.addWidget(page)

        def sync_stack():
            key = str(type_combo.currentData() or type_order[0])
            stack.setCurrentWidget(pages.get(key, pages[type_order[0]]))

        type_combo.currentIndexChanged.connect(lambda _=None: sync_stack())
        sync_stack()
        layout.addWidget(stack)

        warn = QLabel("Changing shape type regenerates dimensions and constraints for this sketch.")
        warn.setWordWrap(True)
        layout.addWidget(warn)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.Accepted:
            return None

        selected_type = str(type_combo.currentData() or type_order[0]).lower()
        refs = refs_by_type[selected_type]

        geom_keys = {
            "type",
            "points",
            "p1",
            "p2",
            "center",
            "origin",
            "width",
            "height",
            "radius",
            "sides",
            "angle",
            "start_angle",
            "end_angle",
        }
        out_meta = {
            k: copy.deepcopy(v)
            for k, v in (meta or {}).items()
            if k not in geom_keys
        }
        out_meta["type"] = selected_type

        if selected_type == "line":
            out_meta = self._line_meta_from_points(
                (refs["p1x"].value(), refs["p1y"].value()),
                (refs["p2x"].value(), refs["p2y"].value()),
                base_meta=out_meta,
            )
        elif selected_type == "rectangle":
            cx = refs["cx"].value()
            cy = refs["cy"].value()
            width = max(0.001, refs["width"].value())
            height = max(0.001, refs["height"].value())
            out_meta = self._rectangle_meta_from_origin_size(
                (cx - width * 0.5, cy - height * 0.5),
                width,
                height,
                mode=(meta or {}).get("mode", "two_corner"),
                base_meta=out_meta,
            )
        elif selected_type == "circle":
            out_meta = self._circle_meta_from_center_radius(
                (refs["cx"].value(), refs["cy"].value()),
                max(0.001, refs["radius"].value()),
                base_meta=out_meta,
            )
        elif selected_type == "slot":
            out_meta["p1"] = (refs["p1x"].value(), refs["p1y"].value())
            out_meta["p2"] = (refs["p2x"].value(), refs["p2y"].value())
            out_meta["width"] = max(0.001, refs["width"].value())
        elif selected_type == "polygon":
            out_meta["center"] = (refs["cx"].value(), refs["cy"].value())
            out_meta["radius"] = max(0.001, refs["radius"].value())
            out_meta["sides"] = max(3, int(refs["sides"].value()))
            out_meta["angle"] = math.radians(refs["angle_deg"].value())
        else:
            return None

        return out_meta

    def _edit_sketch_shape_by_index(self, owner_type, owner_part, sketch_index):
        if owner_type is None:
            QMessageBox.information(self, "Edit Sketch Shape", "No sketch geometry is available to edit.")
            return False
        if owner_type == "part" and owner_part is not None and bool(getattr(owner_part, "is_direct_edit", False)):
            QMessageBox.information(
                self,
                "Edit Sketch Shape",
                "Direct-edit part sketches should be edited via 'Edit Shape' (active sketch) and Confirm Part.",
            )
            return False
        points = copy.deepcopy(self._get_sketch_points(owner_type, owner_part, sketch_index))
        if points is None:
            points = []
        if len(points) == 0:
            QMessageBox.information(self, "Edit Sketch Shape", "Selected sketch shape has no editable points.")
            return False
        current_meta = copy.deepcopy(self._get_sketch_meta(owner_type, owner_part, sketch_index) or {})
        new_meta = self._prompt_sketch_shape_meta_edit(current_meta, points)
        if not new_meta:
            return False

        self.push_undo_state()
        if not self._update_sketch_from_meta(owner_type, owner_part, sketch_index, new_meta):
            QMessageBox.warning(self, "Edit Sketch Shape", "Could not rebuild the selected sketch shape.")
            return False
        self._reset_sketch_auto_annotations(owner_type, owner_part, sketch_index)

        if owner_type == "part" and owner_part is not None:
            self._update_part_geometry_from_sketches(owner_part)
            self.rebuild_display_geometry()
            self.partsChanged.emit()
        else:
            self.geometryChanged.emit()
        self.redraw()
        try:
            self.mesh3dUpdated.emit(None, None)
        except Exception:
            pass

        if getattr(self, "_editing_part_shape_id", None) not in (None, ""):
            self._announce_status("Sketch shape updated. Use Confirm Part to apply the changes.")
        else:
            self._announce_status("Sketch shape updated.")
        return True

    def _edit_sketch_shape_at(self, pt, owner_type=None, owner_part=None):
        if owner_type is None:
            owner_type, inferred_part = self._active_owner()
            if owner_part is None:
                owner_part = inferred_part
        if owner_type is None:
            QMessageBox.information(self, "Edit Sketch Shape", "No sketch geometry is available to edit.")
            return False

        hit = self._find_nearest_segment(owner_type, owner_part, pt)
        if hit is None:
            hit_vertex = self._find_nearest_vertex(owner_type, owner_part, pt)
            if hit_vertex is None:
                QMessageBox.information(
                    self,
                    "Edit Sketch Shape",
                    "Right-click near a sketch edge/shape to edit its parameters.",
                )
                return False
            hit = (hit_vertex[0], 0)
        return self._edit_sketch_shape_by_index(owner_type, owner_part, int(hit[0]))

    def edit_sketch_entity_by_index(self, sketch_index, owner_type="sketch", owner_part=None):
        try:
            idx = int(sketch_index)
        except Exception:
            return False
        return self._edit_sketch_shape_by_index(owner_type, owner_part, idx)

    def _refresh_after_sketch_owner_change(self, owner_type, owner_part):
        if owner_type == "part" and owner_part is not None:
            self._update_part_geometry_from_sketches(owner_part)
            self.rebuild_display_geometry()
            self.partsChanged.emit()
        self.geometryChanged.emit()
        self.redraw()
        try:
            self.mesh3dUpdated.emit(None, None)
        except Exception:
            pass
        return True

    def _refresh_after_sketch_meta_change(self, owner_type, owner_part, emit_parts=False):
        if emit_parts and owner_type == "part" and owner_part is not None:
            self.partsChanged.emit()
        self.geometryChanged.emit()
        self.redraw()
        try:
            self.mesh3dUpdated.emit(None, None)
        except Exception:
            pass
        return True

    def delete_sketch_entity_by_index(self, sketch_index, owner_type="sketch", owner_part=None, confirm=True):
        try:
            idx = int(sketch_index)
        except Exception:
            return False
        sketches, _, _, _ = self._owner_collections(owner_type, owner_part)
        if idx < 0 or idx >= len(sketches):
            return False
        if owner_type == "part" and owner_part is not None and bool(getattr(owner_part, "is_direct_edit", False)):
            QMessageBox.information(
                self,
                "Delete Sketch",
                "Direct-edit part sketches should be edited via 'Edit Shape' (active sketch) and Confirm Part.",
            )
            return False
        if owner_type == "part" and owner_part is not None and len(sketches) <= 1:
            reply = QMessageBox.question(
                self,
                "Delete Sketch",
                f"'{owner_part.name}' has only one sketch left.\nDelete the whole part instead?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return False
            return bool(self.delete_part(owner_part, confirm=False))

        if confirm:
            label = "active sketch" if owner_type == "sketch" else "part sketch"
            reply = QMessageBox.question(
                self,
                "Delete Sketch",
                f"Delete {label} #{idx + 1}?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return False

        self.push_undo_state()
        if not self._remove_owner_sketch(owner_type, owner_part, idx):
            return False
        self._refresh_after_sketch_owner_change(owner_type, owner_part)
        return True

    def duplicate_sketch_entity_by_index(self, sketch_index, dx=10.0, dy=10.0, owner_type="sketch", owner_part=None):
        try:
            idx = int(sketch_index)
        except Exception:
            return False
        sketches, metas, _, _ = self._owner_collections(owner_type, owner_part)
        if idx < 0 or idx >= len(sketches):
            return False
        if owner_type == "part" and owner_part is not None and bool(getattr(owner_part, "is_direct_edit", False)):
            QMessageBox.information(
                self,
                "Duplicate Sketch",
                "Direct-edit part sketches should be edited via 'Edit Shape' (active sketch) and Confirm Part.",
            )
            return False

        src_points = copy.deepcopy(sketches[idx] or [])
        src_meta = copy.deepcopy(metas[idx] if idx < len(metas) else {})
        if not src_points:
            return False
        new_points = self._transform_points(src_points, dx=dx, dy=dy)
        new_meta = self._transform_meta(src_meta, src_points, dx=dx, dy=dy)

        self.push_undo_state()
        self._append_sketch(new_points, meta=new_meta, owner_type=owner_type, owner_part=owner_part)
        self._refresh_after_sketch_owner_change(owner_type, owner_part)
        return True

    def is_sketch_entity_visible_by_index(self, sketch_index, owner_type="sketch", owner_part=None):
        try:
            idx = int(sketch_index)
        except Exception:
            return True
        sketches, _, _, _ = self._owner_collections(owner_type, owner_part)
        if idx < 0 or idx >= len(sketches):
            return True
        return self._is_sketch_visible(owner_type, owner_part, idx)

    def set_sketch_entity_visible_by_index(self, sketch_index, visible=True, owner_type="sketch", owner_part=None):
        try:
            idx = int(sketch_index)
        except Exception:
            return False
        sketches, metas, _, _ = self._owner_collections(owner_type, owner_part)
        if idx < 0 or idx >= len(sketches):
            return False
        while len(metas) < len(sketches):
            metas.append(self._default_sketch_meta(sketches[len(metas)]))
        current_meta = copy.deepcopy(metas[idx] or {})
        new_visible = bool(visible)
        if bool(current_meta.get("visible", True)) == new_visible:
            return True
        self.push_undo_state()
        current_meta["visible"] = new_visible
        metas[idx] = current_meta
        return self._refresh_after_sketch_meta_change(
            owner_type,
            owner_part,
            emit_parts=(owner_type == "part" and owner_part is not None),
        )

    def toggle_sketch_entity_visibility_by_index(self, sketch_index, owner_type="sketch", owner_part=None):
        current = self.is_sketch_entity_visible_by_index(
            sketch_index,
            owner_type=owner_type,
            owner_part=owner_part,
        )
        return self.set_sketch_entity_visible_by_index(
            sketch_index,
            visible=not current,
            owner_type=owner_type,
            owner_part=owner_part,
        )

    def rename_sketch_entity_by_index(self, sketch_index, name, owner_type="sketch", owner_part=None):
        try:
            idx = int(sketch_index)
        except Exception:
            return False
        sketches, metas, _, _ = self._owner_collections(owner_type, owner_part)
        if idx < 0 or idx >= len(sketches):
            return False
        while len(metas) < len(sketches):
            metas.append(self._default_sketch_meta(sketches[len(metas)]))
        label = str(name or "").strip()
        current_meta = copy.deepcopy(metas[idx] or {})
        old_label = str(current_meta.get("name", "") or "").strip()
        if old_label == label:
            return True
        self.push_undo_state()
        if label:
            current_meta["name"] = label
        else:
            current_meta.pop("name", None)
        metas[idx] = current_meta
        return self._refresh_after_sketch_meta_change(
            owner_type,
            owner_part,
            emit_parts=(owner_type == "part" and owner_part is not None),
        )

    def reorder_sketch_entities(self, ordered_indices, owner_type="sketch", owner_part=None):
        try:
            order = [int(i) for i in (ordered_indices if ordered_indices is not None else [])]
        except Exception:
            return False
        sketches, metas, dimensions, constraints = self._owner_collections(owner_type, owner_part)
        count = len(sketches)
        if count <= 1:
            return True
        if len(order) != count or sorted(order) != list(range(count)):
            return False
        if order == list(range(count)):
            return True
        if owner_type == "part" and owner_part is not None and bool(getattr(owner_part, "is_direct_edit", False)):
            QMessageBox.information(
                self,
                "Reorder Sketches",
                "Direct-edit part sketches should be edited via 'Edit Shape' (active sketch) and Confirm Part.",
            )
            return False

        while len(metas) < len(sketches):
            metas.append(self._default_sketch_meta(sketches[len(metas)]))

        old_sketches = list(sketches)
        old_metas = list(metas)
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(order)}

        self.push_undo_state()
        sketches[:] = [old_sketches[old_idx] for old_idx in order]
        metas[:] = [old_metas[old_idx] for old_idx in order]

        for dim in dimensions:
            try:
                old_idx = int(dim.get("sketch_index", -1))
            except Exception:
                continue
            if old_idx in old_to_new:
                dim["sketch_index"] = int(old_to_new[old_idx])

        for con in constraints:
            for key in ("sketch_index", "other_sketch_index"):
                try:
                    old_idx = int(con.get(key, -1))
                except Exception:
                    continue
                if old_idx in old_to_new:
                    con[key] = int(old_to_new[old_idx])

        self._refresh_after_sketch_owner_change(owner_type, owner_part)
        return True

    def _auto_create_dimensions(self, owner_type, owner_part, sketch_index, meta):
        _, _, dimensions, _ = self._owner_collections(owner_type, owner_part)
        if bool(meta.get("skip_auto_dimension", False)):
            return
        if (
            str(meta.get("source_tool", "")).strip().lower() == "freeform"
            and not bool(meta.get("auto_converted", False))
        ):
            return
        meta_type = str(meta.get("type", "polyline")).lower()
        if meta_type == "line":
            dimensions.append(
                {
                    "id": self._next_dimension_id(),
                    "dim_type": "linear",
                    "sketch_index": sketch_index,
                    "segment_index": 0,
                }
            )
        elif meta_type == "rectangle":
            dimensions.extend(
                [
                    {
                        "id": self._next_dimension_id(),
                        "dim_type": "rect_width",
                        "sketch_index": sketch_index,
                        "segment_index": 0,
                    },
                    {
                        "id": self._next_dimension_id(),
                        "dim_type": "rect_height",
                        "sketch_index": sketch_index,
                        "segment_index": 1,
                    },
                ]
            )
        elif meta_type == "circle":
            dimensions.append(
                {
                    "id": self._next_dimension_id(),
                    "dim_type": "diameter",
                    "sketch_index": sketch_index,
                }
            )
        elif meta_type == "arc":
            dimensions.append(
                {
                    "id": self._next_dimension_id(),
                    "dim_type": "radius",
                    "sketch_index": sketch_index,
                }
            )
        elif meta_type == "slot":
            dimensions.extend(
                [
                    {
                        "id": self._next_dimension_id(),
                        "dim_type": "slot_length",
                        "sketch_index": sketch_index,
                    },
                    {
                        "id": self._next_dimension_id(),
                        "dim_type": "slot_width",
                        "sketch_index": sketch_index,
                    },
                ]
            )
        elif meta_type == "polygon":
            dimensions.append(
                {
                    "id": self._next_dimension_id(),
                    "dim_type": "polygon_radius",
                    "sketch_index": sketch_index,
                }
            )
        else:
            points = self._get_sketch_points(owner_type, owner_part, sketch_index)
            if points is None:
                points = []
            for seg_index in range(max(0, len(points) - 1)):
                dimensions.append(
                    {
                        "id": self._next_dimension_id(),
                        "dim_type": "linear",
                        "sketch_index": sketch_index,
                        "segment_index": seg_index,
                    }
                )

    def _auto_create_constraints(self, owner_type, owner_part, sketch_index, meta):
        _, _, _, constraints = self._owner_collections(owner_type, owner_part)
        meta_type = str(meta.get("type", "polyline")).lower()
        points = self._get_sketch_points(owner_type, owner_part, sketch_index)
        if points is None:
            points = []
        if meta_type == "line" and len(points) >= 2:
            p1, p2 = points[0], points[1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            if abs(dy) <= abs(dx) * 0.1:
                constraints.append(
                    {
                        "type": "horizontal",
                        "sketch_index": sketch_index,
                        "segment_index": 0,
                    }
                )
            elif abs(dx) <= abs(dy) * 0.1:
                constraints.append(
                    {
                        "type": "vertical",
                        "sketch_index": sketch_index,
                        "segment_index": 0,
                    }
                )
        if meta_type == "rectangle" and len(points) >= 4:
            constraints.extend(
                [
                    {
                        "type": "parallel",
                        "sketch_index": sketch_index,
                        "segment_index": 0,
                        "other_sketch_index": sketch_index,
                        "other_segment_index": 2,
                    },
                    {
                        "type": "parallel",
                        "sketch_index": sketch_index,
                        "segment_index": 1,
                        "other_sketch_index": sketch_index,
                        "other_segment_index": 3,
                    },
                    {
                        "type": "perpendicular",
                        "sketch_index": sketch_index,
                        "segment_index": 0,
                        "other_sketch_index": sketch_index,
                        "other_segment_index": 1,
                    },
                    {
                        "type": "equal",
                        "sketch_index": sketch_index,
                        "segment_index": 0,
                        "other_sketch_index": sketch_index,
                        "other_segment_index": 2,
                    },
                    {
                        "type": "equal",
                        "sketch_index": sketch_index,
                        "segment_index": 1,
                        "other_sketch_index": sketch_index,
                        "other_segment_index": 3,
                    },
                ]
            )

    def set_project_mode(self, mode):
        mode = "3d" if str(mode).lower().startswith("3") else "2d"
        self.project_mode = mode

    def update_3d_settings(self, height=None, layers=None, rebuild=False):
        changed = False
        if height is not None:
            height = float(height)
            if height != self.extrude_height:
                self.extrude_height = height
                changed = True
        if layers is not None:
            layers = int(layers)
            if layers != self.extrude_layers:
                self.extrude_layers = layers
                changed = True
        if changed:
            self.geometryChanged.emit()
        window = self.window()
        if window and hasattr(window, "_update_mode_indicator"):
            window._update_mode_indicator()
        if rebuild and self.project_mode == "3d":
            if self._cad_kernel_ready() or (
                self.global_nodes is not None
                and len(self.global_nodes) > 0
                and self.global_elements is not None
                and len(self.global_elements) > 0
            ):
                if self.build_3d_mesh() and self.display_mode == "mesh_3d":
                    self.redraw()

    def set_mesh_generation_settings(
        self,
        min_spacing_factor=None,
        boundary_thickness=None,
        boundary_spacing_factor=None,
    ):
        changed = False
        if min_spacing_factor is not None:
            min_spacing_factor = float(min_spacing_factor)
            if min_spacing_factor != self.mesh_min_spacing_factor:
                self.mesh_min_spacing_factor = min_spacing_factor
                changed = True
        if boundary_thickness is not None:
            boundary_thickness = float(boundary_thickness)
            if boundary_thickness != self.mesh_boundary_thickness:
                self.mesh_boundary_thickness = boundary_thickness
                changed = True
        if boundary_spacing_factor is not None:
            boundary_spacing_factor = float(boundary_spacing_factor)
            if boundary_spacing_factor != self.mesh_boundary_spacing_factor:
                self.mesh_boundary_spacing_factor = boundary_spacing_factor
                changed = True
        if changed:
            self.geometryChanged.emit()

    def stop_visualization(self):
        self.animation_timer.stop()
        self._emit_animation_playback_state()
        self._release_animation_result_handles()
        self.is_visualization_mode = False
        self.animation_frames.clear()
        self._lazy_results_enabled = False
        self._animation_frame_count = 0
        self._animation_frame_loading = False
        self._pending_lazy_frame_index = None
        self._current_animation_positions = None
        self._current_animation_velocity = None
        self._current_frame_packet = None
        self._replay_lod_active = False
        self._replay_visible_particle_indices = None
        self._replay_selected_particle_index = None
        self._replay_selected_nodes = set()
        self._replay_selected_triangles = set()
        self._replay_selected_mesh_edges = set()
        self._replay_selected_geometry_edges = []
        self._replay_selected_bc_targets = []
        self._replay_particle_ids = None
        self._replay_particle_materials = []
        self._replay_particle_parts = []
        self._replay_particle_id_to_index = {}
        self._replay_part_to_indices = {}
        self._replay_particle_bc_labels = {}
        self._replay_particle_load_vectors = {}
        self._replay_metadata_particle_count = 0
        self._clear_results_debug_overlays()
        self._clear_results_legend()
        self._results_preview_auto_fit_pending = False
        self.current_frame_index = 0
        self._displacement_vectors = []
        window = self.window()
        if window is not None and hasattr(window, "_hide_results_point_preview"):
            try:
                window._hide_results_point_preview(clear_points=True, force_view=True)
            except Exception:
                pass
        self.animationFramesLoaded.emit(0)
        self.animationFrameChanged.emit(0)
        self.replayParticleSelected.emit({})
        controller = self._get_results_controller()
        if controller is not None and hasattr(controller, "clear_results_source"):
            try:
                controller.clear_results_source()
            except Exception:
                pass

    def _release_numpy_result_handle(self, value):
        current = value
        seen = set()
        while current is not None:
            ident = id(current)
            if ident in seen:
                break
            seen.add(ident)
            mmap_obj = getattr(current, "_mmap", None)
            if mmap_obj is not None:
                try:
                    mmap_obj.close()
                except Exception:
                    pass
            close_fn = getattr(current, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
            base = getattr(current, "base", None)
            if base is current:
                break
            current = base

    def _release_animation_result_handles(self):
        for frame in list(getattr(self, "animation_frames", []) or []):
            self._release_numpy_result_handle(frame)
        packet = getattr(self, "_current_frame_packet", None)
        if isinstance(packet, dict):
            for value in packet.values():
                self._release_numpy_result_handle(value)
        self._release_numpy_result_handle(getattr(self, "_current_animation_positions", None))
        self._release_numpy_result_handle(getattr(self, "_current_animation_velocity", None))
        self._release_numpy_result_handle(getattr(self, "_result_scalar_values", None))

    def release_results_file_handles(self):
        self.stop_visualization()
        gc.collect()

    def _get_results_controller(self):
        window = self.window()
        if window is None:
            return None
        return getattr(window, "results_controller", None)

    def _animation_frame_total(self):
        if getattr(self, "_lazy_results_enabled", False):
            return int(getattr(self, "_animation_frame_count", 0))
        return len(getattr(self, "animation_frames", []) or [])

    def _emit_animation_playback_state(self):
        try:
            playing = bool(getattr(self, "animation_timer", None) and self.animation_timer.isActive())
        except Exception:
            playing = False
        try:
            self.animationPlaybackStateChanged.emit(bool(playing))
        except Exception:
            pass

    def _normalize_animation_positions(self, positions):
        if positions is None:
            return None
        try:
            arr = np.asarray(positions, dtype=float)
        except Exception:
            return None
        if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] < 2:
            return None
        return np.array(arr[:, :2], dtype=float, copy=True)

    def _validated_current_frame_packet(self):
        packet = getattr(self, "_current_frame_packet", None)
        if not isinstance(packet, dict):
            return None
        raw_positions = self._normalize_animation_positions(packet.get("raw_positions"))
        display_positions = self._normalize_animation_positions(packet.get("display_positions"))
        if raw_positions is None or display_positions is None:
            return None
        display_indices = packet.get("display_indices")
        try:
            display_indices = np.asarray(display_indices, dtype=int).reshape(-1)
        except Exception:
            display_indices = None
        if display_indices is None or display_indices.size != int(display_positions.shape[0]):
            if int(display_positions.shape[0]) == int(raw_positions.shape[0]):
                display_indices = np.arange(int(display_positions.shape[0]), dtype=int)
            else:
                return None
        validated = dict(packet)
        validated["raw_positions"] = raw_positions
        validated["display_positions"] = display_positions
        validated["display_indices"] = display_indices
        return validated

    def _has_valid_current_animation_frame(self):
        return self._validated_current_frame_packet() is not None

    def ensure_animation_frame_initialized(self, index=None):
        total = self._animation_frame_total()
        if total <= 0:
            return False
        packet = self._validated_current_frame_packet()
        if packet is not None:
            current_positions = self._normalize_animation_positions(
                getattr(self, "_current_animation_positions", None)
            )
            if current_positions is None or int(current_positions.shape[0]) != int(packet["raw_positions"].shape[0]):
                self._current_animation_positions = np.array(packet["raw_positions"], dtype=float, copy=True)
            return True
        try:
            frame_index = int(index if index is not None else getattr(self, "current_frame_index", 0))
        except Exception:
            frame_index = 0
        frame_index = max(0, min(frame_index, total - 1))
        if getattr(self, "_lazy_results_enabled", False):
            if not getattr(self, "_animation_frame_loading", False):
                self._request_lazy_animation_frame(frame_index)
            return False
        frames = list(getattr(self, "animation_frames", []) or [])
        if not frames:
            return False
        self.set_animation_frame(frame_index)
        return self._has_valid_current_animation_frame()

    def _scene_units_for_pixels(self, pixel_size=8.0):
        try:
            scale_x = abs(float(self.transform().m11()))
            if scale_x > 1e-9:
                return float(pixel_size) / scale_x
        except Exception:
            pass
        return float(pixel_size)

    def _results_render_metrics(self, particle_count=None, display_count=None):
        try:
            total_count = max(1, int(particle_count or 0))
        except Exception:
            total_count = 1
        try:
            visible_count = max(1, int(display_count or total_count))
        except Exception:
            visible_count = total_count

        density_count = max(total_count, visible_count)
        density_scale = min(1.0, math.sqrt(2500.0 / float(max(1, density_count))))

        node_px = max(1.2, min(4.0, 4.0 * density_scale))
        connection_px = max(0.18, min(1.0, 1.0 * (density_scale ** 0.9)))
        overlay_px = max(2.0, node_px * 1.5)
        overlay_pen_px = max(0.6, connection_px * 1.15)

        node_radius = max(self._scene_units_for_pixels(node_px) * 0.5, 0.3)
        overlay_radius = max(self._scene_units_for_pixels(overlay_px) * 0.5, node_radius * 1.2)
        pick_radius = max(self._scene_units_for_pixels(max(6.0, overlay_px * 1.75)), overlay_radius * 1.75)
        load_vector_scale = max(self._scene_units_for_pixels(16.0), overlay_radius * 3.0)

        return {
            "node_radius": node_radius,
            "node_diameter": node_radius * 2.0,
            "connection_pen_width": connection_px,
            "overlay_radius": overlay_radius,
            "overlay_pen_width": overlay_pen_px,
            "pick_radius": pick_radius,
            "load_vector_scale": load_vector_scale,
        }

    def _clear_results_debug_overlays(self):
        for item in list(getattr(self, "_results_debug_overlay_items", []) or []):
            try:
                scene = item.scene()
                if scene is not None:
                    scene.removeItem(item)
            except Exception:
                pass
        self._results_debug_overlay_items = []

    def _clear_results_legend(self):
        self._results_legend_state = None
        try:
            self.viewport().update()
        except Exception:
            pass

    def _format_results_legend_value(self, value):
        try:
            val = float(value)
        except Exception:
            return "--"
        if not math.isfinite(val):
            return "--"
        abs_val = abs(val)
        if abs_val >= 1.0e4 or (0.0 < abs_val < 1.0e-3):
            return f"{val:.3e}"
        return f"{val:.4g}"

    def _build_results_legend_state(self, field_packet):
        if not isinstance(field_packet, dict):
            return None
        field_key = str(field_packet.get("key") or "none").strip().lower()
        if field_key == "none":
            return None
        values = field_packet.get("values")
        if values is None:
            return None
        arr = np.asarray(values, dtype=float).reshape(-1)
        finite = np.isfinite(arr)
        if not np.any(finite):
            return None
        vals = arr[finite]
        controller = self._get_results_controller()
        if controller is not None and hasattr(controller, "field_legend_metadata"):
            meta = controller.field_legend_metadata(field_key, vals)
        else:
            label = str(field_packet.get("label") or "")
            meta = {
                "key": field_key,
                "label": label,
                "title": label.split(":", 1)[-1].strip() if ":" in label else label,
                "unit": "",
                "scale": 1.0,
                "domain": field_packet.get("domain"),
                "source": None,
            }
        scale = float(meta.get("scale") or 1.0)
        if scale not in (0.0, 1.0):
            vals = vals / scale
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        palette = "jet" if str(meta.get("domain") or field_packet.get("domain") or "").strip().lower() == "triangle" else "viridis"
        return {
            "key": field_key,
            "title": str(meta.get("title") or meta.get("label") or field_packet.get("label") or ""),
            "unit": str(meta.get("unit") or "").strip(),
            "vmin": vmin,
            "vmax": vmax,
            "palette": palette,
        }

    def _update_results_legend_from_field_packet(self, field_packet):
        self._results_legend_state = self._build_results_legend_state(field_packet)
        try:
            self.viewport().update()
        except Exception:
            pass

    def _draw_results_legend_overlay(self, painter):
        if self.display_mode != "results" or not self.is_visualization_mode:
            return
        legend = getattr(self, "_results_legend_state", None)
        if not legend:
            return
        viewport = self.viewport()
        if viewport is None:
            return
        viewport_rect = viewport.rect()
        if viewport_rect.width() < 120 or viewport_rect.height() < 120:
            return

        box_width = 154.0
        box_padding = 12.0
        bar_width = 18.0
        tick_count = 6
        title_height = 36.0
        unit_height = 18.0
        bar_height = float(max(120.0, min(240.0, viewport_rect.height() - 160.0)))
        box_height = title_height + unit_height + bar_height + 24.0
        box_x = 16.0
        box_y = max(16.0, (float(viewport_rect.height()) - box_height) * 0.5)
        bar_x = box_x + box_padding
        bar_y = box_y + title_height + unit_height
        label_x = bar_x + bar_width + 12.0
        label_width = box_width - (label_x - box_x) - box_padding

        painter.save()
        painter.resetTransform()
        painter.setRenderHint(QPainter.Antialiasing, True)

        background = QRectF(box_x, box_y, box_width, box_height)
        painter.setPen(QPen(QColor(120, 128, 144, 170), 1.0))
        painter.setBrush(QColor(250, 252, 255, 228))
        painter.drawRoundedRect(background, 10.0, 10.0)

        title_font = painter.font()
        title_font.setBold(True)
        title_font.setPointSizeF(max(8.0, title_font.pointSizeF()))
        painter.setFont(title_font)
        painter.setPen(QColor(24, 30, 42))
        painter.drawText(
            QRectF(box_x + box_padding, box_y + 8.0, box_width - 2.0 * box_padding, 18.0),
            Qt.AlignLeft | Qt.AlignVCenter,
            str(legend.get("title") or "Contour"),
        )

        unit_text = str(legend.get("unit") or "").strip()
        info_font = painter.font()
        info_font.setBold(False)
        info_font.setPointSizeF(max(7.5, info_font.pointSizeF() - 0.5))
        painter.setFont(info_font)
        painter.setPen(QColor(80, 88, 104))
        painter.drawText(
            QRectF(box_x + box_padding, box_y + 24.0, box_width - 2.0 * box_padding, 16.0),
            Qt.AlignLeft | Qt.AlignVCenter,
            f"Units: {unit_text or '--'}",
        )

        gradient = QLinearGradient(bar_x, bar_y + bar_height, bar_x, bar_y)
        colors = self._scalar_colors(np.linspace(0.0, 1.0, 8), palette=str(legend.get("palette") or "jet"), alpha=255)
        if colors:
            denom = max(1, len(colors) - 1)
            for idx, color in enumerate(colors):
                gradient.setColorAt(float(idx) / float(denom), color)
        else:
            gradient.setColorAt(0.0, QColor(0, 0, 180))
            gradient.setColorAt(0.5, QColor(0, 220, 120))
            gradient.setColorAt(1.0, QColor(220, 0, 0))
        painter.setPen(QPen(QColor(70, 78, 94, 190), 1.0))
        painter.setBrush(QBrush(gradient))
        painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_width, bar_height), 4.0, 4.0)

        vmin = float(legend.get("vmin", 0.0))
        vmax = float(legend.get("vmax", 0.0))
        if tick_count <= 1:
            tick_values = [vmax]
        else:
            tick_values = np.linspace(vmax, vmin, tick_count)
        for idx, tick_value in enumerate(tick_values):
            t = 0.0 if tick_count <= 1 else float(idx) / float(tick_count - 1)
            y_tick = bar_y + t * bar_height
            painter.setPen(QPen(QColor(80, 88, 104, 190), 1.0))
            painter.drawLine(
                QPointF(bar_x + bar_width, y_tick),
                QPointF(bar_x + bar_width + 6.0, y_tick),
            )
            painter.setPen(QColor(24, 30, 42))
            painter.drawText(
                QRectF(label_x, y_tick - 9.0, label_width, 18.0),
                Qt.AlignLeft | Qt.AlignVCenter,
                self._format_results_legend_value(tick_value),
            )

        painter.restore()

    def drawBackground(self, painter, rect):
        try:
            super().drawBackground(painter, rect)
        except Exception:
            pass
        if not getattr(self, "grid_visible", True):
            return
        try:
            t = self.transform()
            px_per_unit = max(abs(t.m11()), abs(t.m22()))
            if px_per_unit <= 0.0 or not math.isfinite(px_per_unit):
                return
            minor = _snap_125(GRID_TARGET_MINOR_PX / px_per_unit)
            if minor <= 0.0 or not math.isfinite(minor):
                return
            self._effective_grid_spacing = float(minor)

            left = float(rect.left())
            right = float(rect.right())
            top = float(rect.top())
            bottom = float(rect.bottom())
            if not (math.isfinite(left) and math.isfinite(right) and math.isfinite(top) and math.isfinite(bottom)):
                return

            i0 = int(math.floor(left / minor))
            i1 = int(math.ceil(right / minor))
            j0 = int(math.floor(top / minor))
            j1 = int(math.ceil(bottom / minor))

            MAX_LINES_PER_AXIS = 4000
            if (i1 - i0) > MAX_LINES_PER_AXIS or (j1 - j0) > MAX_LINES_PER_AXIS:
                return

            minor_lines = []
            major_lines = []
            for i in range(i0, i1 + 1):
                x = i * minor
                line = QLineF(x, top, x, bottom)
                if i % 5 == 0:
                    major_lines.append(line)
                else:
                    minor_lines.append(line)
            for j in range(j0, j1 + 1):
                y = j * minor
                line = QLineF(left, y, right, y)
                if j % 5 == 0:
                    major_lines.append(line)
                else:
                    minor_lines.append(line)

            pen_minor = QPen(QColor(230, 234, 239, 150))
            pen_minor.setCosmetic(True)
            pen_minor.setWidth(0)
            pen_major = QPen(QColor(214, 219, 226, 180))
            pen_major.setCosmetic(True)
            pen_major.setWidth(0)
            pen_axis = QPen(QColor(188, 192, 198, 200))
            pen_axis.setCosmetic(True)
            pen_axis.setWidth(0)

            painter.save()
            try:
                if minor_lines:
                    painter.setPen(pen_minor)
                    painter.drawLines(minor_lines)
                if major_lines:
                    painter.setPen(pen_major)
                    painter.drawLines(major_lines)
                painter.setPen(pen_axis)
                if top <= 0.0 <= bottom:
                    painter.drawLine(QLineF(left, 0.0, right, 0.0))
                if left <= 0.0 <= right:
                    painter.drawLine(QLineF(0.0, top, 0.0, bottom))
            finally:
                painter.restore()
        except Exception:
            pass

    def current_grid_spacing(self):
        return float(getattr(self, "_effective_grid_spacing", GRID_MINOR))

    def drawForeground(self, painter, rect):
        try:
            super().drawForeground(painter, rect)
        except Exception:
            pass
        self._draw_results_legend_overlay(painter)
        self._draw_custom_zone_overlay(painter)
        self._draw_edge_seed_overlay(painter)
        self._draw_vertex_seed_overlay(painter)
        self._draw_seed_preview_overlay(painter)
        self._draw_partition_overlay(painter)
        self._draw_marquee_overlay(painter)
        self._draw_drawing_crosshair(painter)
        self._draw_measure_overlay(painter)
        self._draw_alignment_guides(painter)

    def _draw_marquee_overlay(self, painter):
        if not getattr(self, "_marquee_active", False):
            return
        start = self._marquee_start_scene_pt
        end = self._marquee_current_scene_pt
        if start is None or end is None:
            return
        x = min(start[0], end[0])
        y = min(start[1], end[1])
        w = abs(end[0] - start[0])
        h = abs(end[1] - start[1])
        if w <= 0 or h <= 0:
            return
        painter.save()
        try:
            pen = QPen(QColor(37, 99, 235, 220))
            pen.setCosmetic(True)
            pen.setWidth(1)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QColor(37, 99, 235, 30))
            painter.drawRect(QRectF(x, y, w, h))
        finally:
            painter.restore()

    def _draw_drawing_crosshair(self, painter):
        """Full-canvas crosshair guides at the cursor while a draw tool is
        active. Helps align points across the canvas."""
        if self.tool == "select":
            return
        if self.tool not in (
            "line", "rectangle", "circle", "ellipse", "polygon",
            "polyline", "arc", "freeform",
        ):
            return
        cursor_pt = getattr(self, "_last_cursor_scene_pt", None)
        if cursor_pt is None:
            return
        rect = self.mapToScene(self.viewport().rect()).boundingRect()
        painter.save()
        try:
            pen = QPen(QColor(150, 160, 175, 90))
            pen.setCosmetic(True)
            pen.setWidth(1)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            cx, cy = float(cursor_pt[0]), float(cursor_pt[1])
            painter.drawLine(QPointF(rect.left(), cy), QPointF(rect.right(), cy))
            painter.drawLine(QPointF(cx, rect.top()), QPointF(cx, rect.bottom()))
        finally:
            painter.restore()

    def _draw_alignment_guides(self, painter):
        """Render Figma-style pink alignment guides while a part is being
        dragged and one of its bbox edges or center aligns with another
        part's. Cleared on drag release."""
        guides = getattr(self, "_align_guides", None)
        if not guides:
            return
        painter.save()
        try:
            pen = QPen(QColor(244, 63, 94, 230))  # rose-500
            pen.setCosmetic(True)
            pen.setWidthF(1.6)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            for g in guides:
                if g.get("orient") == "v":
                    x = float(g.get("value", 0.0))
                    y1 = float(g.get("y_min", 0.0))
                    y2 = float(g.get("y_max", 0.0))
                    # Extend the guide by 5% past both ends for visual punch.
                    pad = max(abs(y2 - y1) * 0.05, 0.0)
                    painter.drawLine(QPointF(x, y1 - pad), QPointF(x, y2 + pad))
                elif g.get("orient") == "h":
                    y = float(g.get("value", 0.0))
                    x1 = float(g.get("x_min", 0.0))
                    x2 = float(g.get("x_max", 0.0))
                    pad = max(abs(x2 - x1) * 0.05, 0.0)
                    painter.drawLine(QPointF(x1 - pad, y), QPointF(x2 + pad, y))
        finally:
            painter.restore()

    def _draw_measure_overlay(self, painter):
        """Render the measure-tool line, end markers, and distance/angle
        label. Triggered while the measure tool is active and at least the
        first point has been picked. Once the second point lands, the
        measurement persists until the user clicks again or switches tools."""
        if self.tool != "measure":
            return
        p1 = self._measure_first
        if p1 is None:
            return
        # Endpoint of the live measurement: either the committed second
        # click, or the cursor while the user is still picking it.
        p2 = self._measure_second
        if p2 is None:
            p2 = getattr(self, "_last_cursor_scene_pt", None)
        if p2 is None:
            return
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        dx = x2 - x1
        dy = y2 - y1
        dist_val = math.hypot(dx, dy)
        if dist_val <= 0.0:
            return
        painter.save()
        try:
            # Measurement line
            line_pen = QPen(QColor(220, 38, 38, 230))
            line_pen.setCosmetic(True)
            line_pen.setWidth(2)
            painter.setPen(line_pen)
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
            # End markers
            marker_pen = QPen(QColor(220, 38, 38, 230))
            marker_pen.setCosmetic(True)
            marker_pen.setWidth(2)
            painter.setPen(marker_pen)
            painter.setBrush(QColor(255, 255, 255, 230))
            r = 4.0
            transform = painter.worldTransform()
            sx = max(abs(transform.m11()), 1e-6)
            sy = max(abs(transform.m22()), 1e-6)
            painter.drawEllipse(QPointF(x1, y1), r / sx, r / sy)
            painter.drawEllipse(QPointF(x2, y2), r / sx, r / sy)
            # Label: distance + angle (degrees, CCW from +X)
            unit = getattr(self, "current_unit", "mm")
            ang_deg = math.degrees(math.atan2(dy, dx))
            label = f"{dist_val:.3f} {unit}  ∠{ang_deg:.1f}°"
            # Reset to device coordinates so the label has fixed pixel size,
            # then place near the midpoint with a small upward offset.
            painter.resetTransform()
            mid = self.mapFromScene(QPointF((x1 + x2) * 0.5, (y1 + y2) * 0.5))
            font = painter.font()
            font.setPointSize(9)
            font.setBold(True)
            painter.setFont(font)
            metrics = painter.fontMetrics()
            text_w = metrics.horizontalAdvance(label)
            text_h = metrics.height()
            pad_x, pad_y = 6, 3
            box_x = mid.x() + 10
            box_y = mid.y() - text_h - 10
            box_rect = QRectF(box_x, box_y, text_w + pad_x * 2, text_h + pad_y * 2)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(17, 24, 39, 235))
            painter.drawRoundedRect(box_rect, 4, 4)
            painter.setPen(QColor(255, 255, 255, 245))
            painter.drawText(
                QRectF(box_rect.x() + pad_x, box_rect.y() + pad_y, text_w, text_h),
                Qt.AlignLeft | Qt.AlignVCenter,
                label,
            )
        finally:
            painter.restore()

    # ----- Edge-seed picking API (Mesh stage) ---------------------------

    # ----- Live seed-preview ticks (used by LocalSeedsDialog) ---------

    def set_seed_preview_ticks(self, points):
        """Replace the live tick set. `points` is a flat list of (x, y) world
        coordinates. Triggers a viewport redraw."""
        cleaned = []
        for p in points or []:
            try:
                cleaned.append((float(p[0]), float(p[1])))
            except Exception:
                continue
        self._seed_preview_ticks = cleaned
        self.viewport().update()

    def clear_seed_preview(self):
        if not self._seed_preview_ticks:
            return
        self._seed_preview_ticks = []
        self.viewport().update()

    def _draw_seed_preview_overlay(self, painter):
        if not self._seed_preview_ticks:
            return
        try:
            from PySide6.QtGui import QPen, QBrush, QColor
            from PySide6.QtCore import QPointF
        except Exception:
            return
        try:
            painter.save()
        except Exception:
            return
        try:
            t = painter.worldTransform()
            scale = float(t.m11()) if t.m11() else 1.0
            r = 3.5 / max(abs(scale), 1e-9)
            pen = QPen(QColor(255, 0, 120, 230))
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(255, 0, 120, 230)))
            for x, y in self._seed_preview_ticks:
                painter.drawEllipse(QPointF(x, y), r, r)
        except Exception:
            pass
        finally:
            try:
                painter.restore()
            except Exception:
                pass

    # ----- Smart edge selection helpers (used by MeshPanel) -----------

    def edges_of_part(self, part_id):
        """All boundary edges of the named part as edge-ref dicts."""
        out = []
        for pid, s, e in self._iter_part_edges():
            if int(pid) == int(part_id):
                out.append({"part_id": int(pid), "start": s, "end": e})
        return out

    def edges_in_length_range(self, min_length=None, max_length=None):
        """Edge refs whose Euclidean length falls in the given range. Either
        bound can be None to indicate 'no limit'."""
        import math as _math
        lo = float(min_length) if min_length is not None else 0.0
        hi = float(max_length) if max_length is not None else float("inf")
        out = []
        for pid, s, e in self._iter_part_edges():
            ln = _math.hypot(e[0] - s[0], e[1] - s[1])
            if lo <= ln <= hi:
                out.append({"part_id": int(pid), "start": s, "end": e})
        return out

    def grow_chain_from_edge(self, seed_ref, sharp_angle_deg=45.0):
        """Starting from a single edge, walk along the part boundary and
        collect every neighboring edge that shares an endpoint AND continues
        at less than `sharp_angle_deg` of turn. Stops at sharp corners or
        junctions where 3+ edges meet at a vertex.

        Returns a list of edge-ref dicts including the seed.
        """
        import math as _math
        if not seed_ref:
            return []
        pid = int(seed_ref.get("part_id", 0))
        # Build adjacency map: vertex_key -> list of (start, end, ref) of edges of THIS part.
        part_edges = []
        vert_map = {}

        def _vkey(p):
            return (round(float(p[0]), 6), round(float(p[1]), 6))

        for p_id, s, e in self._iter_part_edges():
            if int(p_id) != pid:
                continue
            ref = {"part_id": int(p_id), "start": s, "end": e}
            part_edges.append(ref)
            for v in (s, e):
                vert_map.setdefault(_vkey(v), []).append(ref)

        if not part_edges:
            return [seed_ref]

        def _same_ref(a, b):
            return (a["part_id"] == b["part_id"] and a["start"] == b["start"] and a["end"] == b["end"])

        # Helper: angle between edge directions (in radians), 0 = colinear.
        def _angle_between(a_ref, b_ref, shared_vertex):
            # vector from shared_vertex along each ref
            def _outward(ref, shared):
                s, e = ref["start"], ref["end"]
                if _vkey(s) == _vkey(shared):
                    return (e[0] - s[0], e[1] - s[1])
                return (s[0] - e[0], s[1] - e[1])
            v1 = _outward(a_ref, shared_vertex)
            v2 = _outward(b_ref, shared_vertex)
            n1 = _math.hypot(*v1); n2 = _math.hypot(*v2)
            if n1 < 1e-12 or n2 < 1e-12:
                return _math.pi
            cos_t = max(-1.0, min(1.0, (v1[0]*v2[0] + v1[1]*v2[1]) / (n1 * n2)))
            return _math.acos(cos_t)

        sharp = _math.radians(float(sharp_angle_deg))
        # 'Continues' = angle between outward directions is close to pi (180°),
        # because outward vectors point AWAY from the shared vertex. A sharp
        # corner has angle close to 0 (vectors point into each other).
        smooth_threshold = _math.pi - sharp

        visited = set()
        def _ref_key(r):
            return (int(r["part_id"]), _vkey(r["start"]), _vkey(r["end"]))
        result = [seed_ref]
        visited.add(_ref_key(seed_ref))

        # Walk in both directions from the seed.
        for endpoint in (seed_ref["start"], seed_ref["end"]):
            cur = seed_ref
            cur_end = endpoint
            while True:
                neighbors = [r for r in vert_map.get(_vkey(cur_end), []) if not _same_ref(r, cur)]
                if len(neighbors) != 1:
                    break  # junction or dead end
                nxt = neighbors[0]
                if _ref_key(nxt) in visited:
                    break
                ang = _angle_between(cur, nxt, cur_end)
                if ang < smooth_threshold:
                    break  # too sharp a corner
                visited.add(_ref_key(nxt))
                result.append(nxt)
                # advance: cur_end becomes the OTHER endpoint of nxt
                if _vkey(nxt["start"]) == _vkey(cur_end):
                    cur_end = nxt["end"]
                else:
                    cur_end = nxt["start"]
                cur = nxt
        return result

    def _iter_part_edges(self):
        """Yield (part_id, (x1,y1), (x2,y2)) for every boundary edge of every
        non-void part. Iterates both exterior and interior rings."""
        for part in getattr(self, "parts", None) or []:
            if getattr(part, "is_void", False):
                continue
            geom = getattr(part, "geometry", None)
            if geom is None or getattr(geom, "is_empty", True):
                continue
            geoms = list(getattr(geom, "geoms", [geom]))  # handle MultiPolygon
            for g in geoms:
                rings = []
                exterior = getattr(g, "exterior", None)
                if exterior is not None:
                    rings.append(exterior)
                rings.extend(getattr(g, "interiors", []) or [])
                for ring in rings:
                    coords = list(getattr(ring, "coords", []))
                    if len(coords) < 2:
                        continue
                    for i in range(len(coords) - 1):
                        x1, y1 = float(coords[i][0]), float(coords[i][1])
                        x2, y2 = float(coords[i + 1][0]), float(coords[i + 1][1])
                        if abs(x1 - x2) < 1e-12 and abs(y1 - y2) < 1e-12:
                            continue
                        yield int(part.id), (x1, y1), (x2, y2)

    def _edge_at_scene_point(self, pt, tol):
        """Return the (part_id, start, end) edge closest to `pt` within `tol`."""
        from geometry_utils import point_line_dist as _pld
        best = None
        best_d = float("inf")
        for pid, s, e in self._iter_part_edges():
            try:
                d = float(_pld(pt, s, e))
            except Exception:
                continue
            if d < best_d:
                best_d = d
                best = (pid, s, e)
        if best is not None and best_d <= tol:
            return best
        return None

    def _edge_seed_pick_tol(self):
        """Tolerance in scene units for an edge to count as 'under the cursor'.
        Scales with viewport zoom so it's roughly constant in screen pixels."""
        try:
            t = self.transform()
            scale = max(abs(float(t.m11())), 1e-9)
            return 8.0 / scale  # ~8 pixels
        except Exception:
            return 8.0

    def begin_edge_seed_pick(self, mode="single", on_complete=None):
        """Start an edge-pick session.

        mode: "single" — first click ends the session and calls on_complete
              with a one-element list. "multi" — clicks accumulate; press
              Enter / Return / right-click to finish, Esc to cancel.
        on_complete(edge_refs): edge_refs is a list of dicts:
              [{"part_id": int, "start": (x,y), "end": (x,y)}, ...]
        """
        m = "multi" if str(mode) == "multi" else "single"
        self._edge_seed_pick_mode = m
        self._edge_seed_picked = []
        self._edge_seed_hover = None
        self._edge_seed_callback = on_complete
        self.setCursor(Qt.PointingHandCursor)
        self.viewport().update()
        win = self.window()
        if win is not None and hasattr(win, "statusBar"):
            try:
                if m == "single":
                    win.statusBar().showMessage(
                        "Click an edge to seed. Esc to cancel.", 6000
                    )
                else:
                    win.statusBar().showMessage(
                        "Click edges to add. Click again to remove. Enter / right-click to finish, Esc to cancel.",
                        8000,
                    )
            except Exception:
                pass

    def cancel_edge_seed_pick(self):
        if self._edge_seed_pick_mode is None:
            return
        self._edge_seed_pick_mode = None
        self._edge_seed_picked = []
        self._edge_seed_hover = None
        cb = self._edge_seed_callback
        self._edge_seed_callback = None
        self.unsetCursor()
        self.viewport().update()
        if callable(cb):
            try:
                cb([])  # empty list signals cancellation
            except Exception:
                pass

    def _finish_edge_seed_pick(self):
        if self._edge_seed_pick_mode is None:
            return
        picked = list(self._edge_seed_picked)
        cb = self._edge_seed_callback
        self._edge_seed_pick_mode = None
        self._edge_seed_picked = []
        self._edge_seed_hover = None
        self._edge_seed_callback = None
        self.unsetCursor()
        self.viewport().update()
        if callable(cb):
            try:
                cb(picked)
            except Exception:
                pass

    def _draw_edge_seed_overlay(self, painter):
        """Highlight the hovered edge and any picked edges during a pick session."""
        if self._edge_seed_pick_mode is None and not self._edge_seed_picked:
            return
        try:
            from PySide6.QtGui import QPen, QColor
            from PySide6.QtCore import QPointF
        except Exception:
            return
        try:
            painter.save()
        except Exception:
            return
        try:
            # Picked edges — solid blue, thick.
            if self._edge_seed_picked:
                pen = QPen(QColor(30, 120, 255, 220))
                pen.setWidth(3)
                pen.setCosmetic(True)
                painter.setPen(pen)
                for ref in self._edge_seed_picked:
                    s, e = ref["start"], ref["end"]
                    painter.drawLine(QPointF(s[0], s[1]), QPointF(e[0], e[1]))
            # Hovered edge — translucent yellow, thinner.
            if self._edge_seed_pick_mode is not None and self._edge_seed_hover is not None:
                hover_ref = self._edge_seed_hover
                pen = QPen(QColor(255, 200, 0, 220))
                pen.setWidth(3)
                pen.setCosmetic(True)
                painter.setPen(pen)
                s, e = hover_ref["start"], hover_ref["end"]
                painter.drawLine(QPointF(s[0], s[1]), QPointF(e[0], e[1]))
        except Exception:
            pass
        finally:
            try:
                painter.restore()
            except Exception:
                pass

    def _draw_partition_overlay(self, painter):
        """Draw the in-progress partition line: first picked point + rubber-
        band to cursor; or both picked points + the connecting line."""
        if not self._partition_pick_active:
            return
        try:
            from PySide6.QtGui import QPen, QBrush, QColor
            from PySide6.QtCore import QPointF
        except Exception:
            return
        try:
            painter.save()
        except Exception:
            return
        try:
            pen = QPen(QColor(180, 0, 200, 230))
            pen.setStyle(Qt.DashLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(180, 0, 200, 230)))
            t = painter.worldTransform()
            scale = float(t.m11()) if t.m11() else 1.0
            r = 4.0 / max(abs(scale), 1e-9)
            for x, y in self._partition_points:
                painter.drawEllipse(QPointF(x, y), r, r)
            if len(self._partition_points) == 1 and self._partition_hover is not None:
                p0 = self._partition_points[0]
                p1 = self._partition_hover
                painter.drawLine(QPointF(p0[0], p0[1]), QPointF(p1[0], p1[1]))
            elif len(self._partition_points) == 2:
                p0, p1 = self._partition_points
                painter.drawLine(QPointF(p0[0], p0[1]), QPointF(p1[0], p1[1]))
        except Exception:
            pass
        finally:
            try:
                painter.restore()
            except Exception:
                pass

    def _draw_vertex_seed_overlay(self, painter):
        """Highlight the hovered vertex and any persistent seeded vertices."""
        if not (self._vertex_seed_pick_active or self._vertex_seed_highlight):
            return
        try:
            from PySide6.QtGui import QPen, QBrush, QColor
            from PySide6.QtCore import QPointF
        except Exception:
            return
        try:
            painter.save()
        except Exception:
            return
        try:
            # Marker radius in scene units: ~6 viewport pixels.
            t = painter.worldTransform()
            scale = float(t.m11()) if t.m11() else 1.0
            r_pick = 5.0 / max(abs(scale), 1e-9)
            r_hover = 7.0 / max(abs(scale), 1e-9)
            # Persistent seeded vertices — filled blue diamond.
            if self._vertex_seed_highlight:
                pen = QPen(QColor(30, 120, 255, 230))
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.setBrush(QBrush(QColor(30, 120, 255, 140)))
                for _pid, (x, y) in self._vertex_seed_highlight:
                    painter.drawEllipse(QPointF(x, y), r_pick, r_pick)
            # Hovered vertex during pick mode — yellow ring.
            if self._vertex_seed_pick_active and self._vertex_seed_hover is not None:
                pen = QPen(QColor(255, 200, 0, 230))
                pen.setWidth(2)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                _pid, (x, y) = self._vertex_seed_hover
                painter.drawEllipse(QPointF(x, y), r_hover, r_hover)
        except Exception:
            pass
        finally:
            try:
                painter.restore()
            except Exception:
                pass

    # ----- Vertex-seed picking API (Mesh stage) ------------------------

    def _iter_part_vertices(self):
        """Yield (part_id, (x, y)) for every unique boundary vertex of every
        non-void part. Deduplicates within each ring."""
        for part in getattr(self, "parts", None) or []:
            if getattr(part, "is_void", False):
                continue
            geom = getattr(part, "geometry", None)
            if geom is None or getattr(geom, "is_empty", True):
                continue
            geoms = list(getattr(geom, "geoms", [geom]))
            seen = set()
            for g in geoms:
                rings = []
                exterior = getattr(g, "exterior", None)
                if exterior is not None:
                    rings.append(exterior)
                rings.extend(getattr(g, "interiors", []) or [])
                for ring in rings:
                    for x, y, *_ in (getattr(ring, "coords", []) or []):
                        key = (round(float(x), 6), round(float(y), 6))
                        if key in seen:
                            continue
                        seen.add(key)
                        yield int(part.id), (float(x), float(y))

    def _vertex_at_scene_point(self, pt, tol):
        """Return the (part_id, (x,y)) vertex closest to `pt` within `tol`."""
        import math as _math
        best = None
        best_d = float("inf")
        for pid, v in self._iter_part_vertices():
            d = _math.hypot(v[0] - pt[0], v[1] - pt[1])
            if d < best_d:
                best_d = d
                best = (pid, v)
        if best is not None and best_d <= tol:
            return best
        return None

    def _vertex_seed_pick_tol(self):
        try:
            t = self.transform()
            scale = max(abs(float(t.m11())), 1e-9)
            return 10.0 / scale
        except Exception:
            return 10.0

    def begin_vertex_seed_pick(self, on_complete=None):
        """Single-vertex pick. First left-click ends the pick and calls
        on_complete with (part_id, (x, y)) or None on cancel/miss."""
        self._vertex_seed_pick_active = True
        self._vertex_seed_hover = None
        self._vertex_seed_callback = on_complete
        self.setCursor(Qt.PointingHandCursor)
        self.viewport().update()
        win = self.window()
        if win is not None and hasattr(win, "statusBar"):
            try:
                win.statusBar().showMessage(
                    "Click a vertex to seed a point-anchored refinement. Esc to cancel.",
                    6000,
                )
            except Exception:
                pass

    def cancel_vertex_seed_pick(self):
        if not self._vertex_seed_pick_active:
            return
        self._vertex_seed_pick_active = False
        self._vertex_seed_hover = None
        cb = self._vertex_seed_callback
        self._vertex_seed_callback = None
        self.unsetCursor()
        self.viewport().update()
        if callable(cb):
            try:
                cb(None)
            except Exception:
                pass

    def _finish_vertex_seed_pick(self, pid_vertex):
        if not self._vertex_seed_pick_active:
            return
        cb = self._vertex_seed_callback
        self._vertex_seed_pick_active = False
        self._vertex_seed_hover = None
        self._vertex_seed_callback = None
        self.unsetCursor()
        self.viewport().update()
        if callable(cb):
            try:
                cb(pid_vertex)
            except Exception:
                pass

    def set_vertex_seed_highlight(self, vertex_refs):
        """Persistently highlight seeded vertices on the canvas. Pass [] to clear.
        vertex_refs: list of (part_id, (x, y)) tuples or dicts {part_id, point}."""
        cleaned = []
        for r in vertex_refs or []:
            if isinstance(r, dict):
                try:
                    pid = int(r.get("part_id", 0))
                    pt = r.get("point") or (r.get("x"), r.get("y"))
                    x, y = float(pt[0]), float(pt[1])
                except Exception:
                    continue
                cleaned.append((pid, (x, y)))
            elif isinstance(r, tuple) and len(r) == 2:
                try:
                    pid = int(r[0])
                    x, y = float(r[1][0]), float(r[1][1])
                except Exception:
                    continue
                cleaned.append((pid, (x, y)))
        self._vertex_seed_highlight = cleaned
        self.viewport().update()

    # ----- Face partition API (Mesh stage) ----------------------------

    def begin_partition_pick(self, on_complete=None):
        """Enter partition mode. The user clicks two points; after the second
        click, the line cuts whatever part it crosses. on_complete is called
        with the list of NEW part ids created (or [] on cancel / no split)."""
        self._partition_pick_active = True
        self._partition_points = []
        self._partition_hover = None
        self._partition_callback = on_complete
        self.setCursor(Qt.CrossCursor)
        self.viewport().update()
        win = self.window()
        if win is not None and hasattr(win, "statusBar"):
            try:
                win.statusBar().showMessage(
                    "Click two points on a part to draw a partition line. Esc to cancel.",
                    8000,
                )
            except Exception:
                pass

    def cancel_partition_pick(self):
        if not self._partition_pick_active:
            return
        self._partition_pick_active = False
        self._partition_points = []
        self._partition_hover = None
        cb = self._partition_callback
        self._partition_callback = None
        self.unsetCursor()
        self.viewport().update()
        if callable(cb):
            try:
                cb([])
            except Exception:
                pass

    def _commit_partition(self):
        """Once two points are picked, find the affected part and split it."""
        if len(self._partition_points) != 2:
            return
        p0, p1 = self._partition_points
        # Find which part the line crosses. A line "crosses" a part if it
        # intersects the part's interior in a line of positive length.
        try:
            from shapely.geometry import LineString, Point as _Pt
        except Exception:
            self.cancel_partition_pick()
            return
        cut = LineString([p0, p1])
        target = None
        for part in (self.parts or []):
            if getattr(part, "is_void", False):
                continue
            geom = getattr(part, "geometry", None)
            if geom is None or getattr(geom, "is_empty", True):
                continue
            try:
                inter = geom.intersection(cut)
            except Exception:
                continue
            if not inter.is_empty and getattr(inter, "length", 0) > 1e-9:
                target = part
                break
        new_ids = []
        if target is not None:
            new_ids = self._perform_partition(target, p0, p1)
        cb = self._partition_callback
        self._partition_pick_active = False
        self._partition_points = []
        self._partition_hover = None
        self._partition_callback = None
        self.unsetCursor()
        self.viewport().update()
        if callable(cb):
            try:
                cb(new_ids)
            except Exception:
                pass

    def _perform_partition(self, part, p0, p1):
        """Split `part` by the line p0→p1. Returns list of new part ids.

        Each new piece becomes a fresh Part inheriting the original part's
        material assignment and other metadata. The original part is removed
        from the project. BCs/loads tied to the original part_id may need to
        be re-pointed by the user — we do not attempt that here.
        """
        from models import Part as _PartCls
        from geometry_utils import split_polygon_by_line
        pieces = split_polygon_by_line(part.geometry, p0, p1)
        if len(pieces) < 2:
            win = self.window()
            if win is not None and hasattr(win, "statusBar"):
                try:
                    win.statusBar().showMessage(
                        "Partition line did not split the part — no change.", 5000
                    )
                except Exception:
                    pass
            return []
        new_parts = []
        for i, piece in enumerate(pieces, start=1):
            np_obj = _PartCls(
                name=f"{part.name or 'Part'} [P{i}]",
                geometry=piece,
                is_void=False,
            )
            # Copy material + other relevant metadata.
            for attr in (
                "material_id", "material_type", "material_props",
                "material_assignment_mode", "heterogeneity_method",
                "heterogeneity_config", "material_field_config",
                "material_symmetry", "material_behavior", "material_damage",
                "is_rigid",
            ):
                try:
                    setattr(np_obj, attr, getattr(part, attr))
                except Exception:
                    pass
            new_parts.append(np_obj)
        # Replace the original part in-place: find its index, splice in new parts.
        try:
            idx = self.parts.index(part)
        except ValueError:
            idx = len(self.parts)
        del self.parts[idx]
        for j, np_obj in enumerate(new_parts):
            self.parts.insert(idx + j, np_obj)
        # Emit signals so the rest of the app updates.
        try:
            self.rebuild_display_geometry()
        except Exception:
            pass
        try:
            self.partsChanged.emit()
        except Exception:
            pass
        try:
            self.geometryChanged.emit()
        except Exception:
            pass
        try:
            self.redraw()
        except Exception:
            pass
        return [p.id for p in new_parts]

    # ----- Edge-seed highlight (kept here for code locality) -----------

    def set_edge_seed_highlight(self, edge_refs):
        """Persistently highlight the given edges (e.g., to show seeds that
        already exist in the project). Pass [] to clear."""
        cleaned = []
        for r in edge_refs or []:
            if not isinstance(r, dict):
                continue
            try:
                pid = int(r.get("part_id"))
                s = (float(r["start"][0]), float(r["start"][1]))
                e = (float(r["end"][0]), float(r["end"][1]))
                cleaned.append({"part_id": pid, "start": s, "end": e})
            except Exception:
                continue
        # Reuse the picked-list for rendering when not in pick mode.
        if self._edge_seed_pick_mode is None:
            self._edge_seed_picked = cleaned
        self.viewport().update()

    # ----- Custom mesh zone drawing API ---------------------------------

    def begin_zone_draw(self, on_complete=None):
        """Enter polygon-draw mode. Caller receives the closed polygon points
        via on_complete(list_of_(x,y))."""
        self._zone_draw_active = True
        self._zone_draw_points = []
        self._zone_draw_hover = None
        self._zone_draw_callback = on_complete
        self._zone_overlay_visible = True
        self.setCursor(Qt.CrossCursor)
        self.viewport().update()
        win = self.window()
        if win is not None and hasattr(win, "statusBar"):
            try:
                win.statusBar().showMessage(
                    "Click to add polygon vertices. Double-click, right-click, or Enter to finish. Esc to cancel.",
                    6000,
                )
            except Exception:
                pass

    def cancel_zone_draw(self):
        if not self._zone_draw_active:
            return
        self._zone_draw_active = False
        self._zone_draw_points = []
        self._zone_draw_hover = None
        self._zone_draw_callback = None
        self.unsetCursor()
        self.viewport().update()

    def _finish_zone_draw(self):
        if not self._zone_draw_active:
            return
        pts = list(self._zone_draw_points)
        callback = self._zone_draw_callback
        self._zone_draw_active = False
        self._zone_draw_points = []
        self._zone_draw_hover = None
        self._zone_draw_callback = None
        self.unsetCursor()
        self.viewport().update()
        if len(pts) < 3:
            win = self.window()
            if win is not None and hasattr(win, "statusBar"):
                try:
                    win.statusBar().showMessage("Zone needs at least 3 vertices — discarded.", 4000)
                except Exception:
                    pass
            return
        if callable(callback):
            try:
                callback(pts)
            except Exception:
                pass

    def set_zone_overlay(self, polygons):
        """Replace the persistent zone overlay list. polygons is a list of
        point lists (each polygon = list of (x,y) tuples)."""
        cleaned = []
        for poly in polygons or []:
            pts = [(float(p[0]), float(p[1])) for p in poly or [] if len(p) >= 2]
            if len(pts) >= 3:
                cleaned.append(pts)
        self._zone_overlay_polygons = cleaned
        self.viewport().update()

    def set_zone_overlay_visible(self, visible):
        self._zone_overlay_visible = bool(visible)
        self.viewport().update()

    def _draw_custom_zone_overlay(self, painter):
        if not (self._zone_overlay_visible or self._zone_draw_active):
            return
        try:
            from PySide6.QtGui import QPainterPath, QPen, QBrush, QColor
            from PySide6.QtCore import QPointF
        except Exception:
            return
        try:
            painter.save()
        except Exception:
            return
        try:
            # Persistent zones — translucent orange fill, solid orange outline.
            if self._zone_overlay_visible and self._zone_overlay_polygons:
                fill = QBrush(QColor(255, 140, 0, 70))
                pen = QPen(QColor(255, 120, 0, 220))
                pen.setWidth(0)  # cosmetic — independent of view scale
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.setBrush(fill)
                for poly in self._zone_overlay_polygons:
                    path = QPainterPath()
                    path.moveTo(QPointF(poly[0][0], poly[0][1]))
                    for x, y in poly[1:]:
                        path.lineTo(QPointF(x, y))
                    path.closeSubpath()
                    painter.drawPath(path)
            # In-progress polyline (while drawing).
            if self._zone_draw_active and self._zone_draw_points:
                pen = QPen(QColor(255, 70, 0, 255))
                pen.setStyle(Qt.DashLine)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                pts = self._zone_draw_points
                for i in range(len(pts) - 1):
                    painter.drawLine(QPointF(pts[i][0], pts[i][1]), QPointF(pts[i + 1][0], pts[i + 1][1]))
                # Rubber-band segment to current cursor position.
                if self._zone_draw_hover is not None:
                    painter.drawLine(
                        QPointF(pts[-1][0], pts[-1][1]),
                        QPointF(self._zone_draw_hover[0], self._zone_draw_hover[1]),
                    )
                # Vertex markers.
                marker_pen = QPen(QColor(255, 70, 0, 255))
                marker_pen.setCosmetic(True)
                painter.setPen(marker_pen)
                painter.setBrush(QBrush(QColor(255, 70, 0, 255)))
                # Marker radius in scene units: ~3 viewport pixels.
                t = painter.worldTransform()
                scale = float(t.m11()) if t.m11() else 1.0
                r = 3.0 / max(abs(scale), 1e-9)
                for x, y in pts:
                    painter.drawEllipse(QPointF(x, y), r, r)
        except Exception:
            pass
        finally:
            try:
                painter.restore()
            except Exception:
                pass

    def _display_indices_for_results(self, particle_count):
        count = int(particle_count or 0)
        if count <= 0:
            self._replay_lod_active = False
            return np.empty((0,), dtype=int)
        if count <= int(self.results_preview_threshold):
            self._replay_lod_active = False
            return np.arange(count, dtype=int)
        self._replay_lod_active = True
        stride = max(
            1,
            int(np.ceil(count / float(max(1, int(self.results_preview_lod_limit))))),
        )
        return np.arange(0, count, stride, dtype=int)

    def _iter_replay_target_indices(self, entry):
        if entry is None:
            return []
        getter = entry.get if hasattr(entry, "get") else None
        if getter is None:
            try:
                entry = dict(entry)
                getter = entry.get
            except Exception:
                return []

        def _normalize_many(value):
            if value is None:
                return []
            if isinstance(value, np.ndarray):
                values = value.reshape(-1).tolist()
            elif isinstance(value, (list, tuple, set)):
                values = list(value)
            else:
                values = [value]
            out = []
            for item in values:
                if item in (None, ""):
                    continue
                out.append(item)
            return out

        indices = []
        for key in (
            "particle_indices",
            "target_indices",
            "indices",
            "node_indices",
            "selected_indices",
        ):
            for item in _normalize_many(getter(key)):
                try:
                    indices.append(int(item))
                except Exception:
                    pass
        if indices:
            return indices

        for key in (
            "particle_ids",
            "target_particle_ids",
            "node_ids",
            "particle_id",
            "node_id",
        ):
            for item in _normalize_many(getter(key)):
                try:
                    pid = int(float(item))
                except Exception:
                    continue
                idx = self._replay_particle_id_to_index.get(pid)
                if idx is not None:
                    indices.append(int(idx))
        if indices:
            return indices

        for key in ("part_id", "part", "part_index"):
            part_value = getter(key)
            if part_value in (None, ""):
                continue
            for candidate in (part_value, str(part_value)):
                mapped = self._replay_part_to_indices.get(candidate)
                if mapped:
                    return list(mapped)
            try:
                part_int = int(float(part_value))
            except Exception:
                continue
            mapped = self._replay_part_to_indices.get(part_int)
            if mapped:
                return list(mapped)
        return []

    def _extract_load_vector_xy(self, entry):
        getter = entry.get if hasattr(entry, "get") else None
        if getter is None:
            return np.zeros(2, dtype=float)

        def _first_numeric(keys):
            for key in keys:
                value = getter(key)
                if value in (None, ""):
                    continue
                try:
                    return float(value)
                except Exception:
                    continue
            return 0.0

        x_val = _first_numeric(("fx", "force_x", "x", "value_x", "vx", "ux"))
        y_val = _first_numeric(("fy", "force_y", "y", "value_y", "vy", "uy"))
        if abs(x_val) > 0.0 or abs(y_val) > 0.0:
            return np.asarray([x_val, y_val], dtype=float)

        magnitude = _first_numeric(("magnitude", "force", "value"))
        if abs(magnitude) <= 0.0:
            return np.zeros(2, dtype=float)
        dir_x = _first_numeric(("dir_x", "direction_x", "nx"))
        dir_y = _first_numeric(("dir_y", "direction_y", "ny"))
        norm = float(np.hypot(dir_x, dir_y))
        if norm <= 1e-12:
            axis = str(getter("axis") or "").strip().lower()
            if axis == "x":
                dir_x = 1.0
            elif axis == "y":
                dir_y = 1.0
            norm = float(np.hypot(dir_x, dir_y))
        if norm <= 1e-12:
            return np.zeros(2, dtype=float)
        return magnitude * np.asarray([dir_x / norm, dir_y / norm], dtype=float)

    def _ensure_replay_particle_metadata(self, particle_count):
        count = int(particle_count or 0)
        if count <= 0:
            self._replay_particle_ids = None
            self._replay_particle_materials = []
            self._replay_particle_parts = []
            self._replay_particle_id_to_index = {}
            self._replay_part_to_indices = {}
            self._replay_particle_bc_labels = {}
            self._replay_particle_load_vectors = {}
            self._replay_metadata_particle_count = 0
            return
        if self._replay_metadata_particle_count == count and self._replay_particle_ids is not None:
            return

        particle_ids = np.arange(count, dtype=int)
        particle_materials = [""] * count
        particle_parts = [None] * count
        csv_rows = []
        particles_path = os.path.join(
            os.path.dirname(__file__),
            "workspace",
            "input",
            "particles.csv",
        )
        if os.path.exists(particles_path):
            try:
                csv_mod = __import__("csv")
                with open(particles_path, newline="", encoding="utf-8") as handle:
                    csv_rows = list(csv_mod.DictReader(handle))
            except Exception:
                csv_rows = []

        def _row_value(row, keys):
            for key in keys:
                value = row.get(key)
                if value not in (None, ""):
                    return value
            return None

        for idx, row in enumerate(csv_rows[:count]):
            pid_value = _row_value(row, ("particle_id", "id", "node_id", "pid"))
            try:
                if pid_value not in (None, ""):
                    particle_ids[idx] = int(float(pid_value))
            except Exception:
                pass
            material_value = _row_value(row, ("material_id", "material", "mat_id"))
            if material_value not in (None, ""):
                particle_materials[idx] = str(material_value)
            part_value = _row_value(row, ("part_id", "part", "part_index"))
            if part_value not in (None, ""):
                try:
                    particle_parts[idx] = int(float(part_value))
                except Exception:
                    particle_parts[idx] = str(part_value)

        particle_id_to_index = {}
        part_to_indices = {}
        for idx, pid in enumerate(particle_ids.tolist()):
            particle_id_to_index[int(pid)] = int(idx)
        for idx, part_value in enumerate(particle_parts):
            if part_value in (None, ""):
                continue
            for candidate in (part_value, str(part_value)):
                part_to_indices.setdefault(candidate, []).append(int(idx))

        self._replay_particle_ids = particle_ids
        self._replay_particle_materials = particle_materials
        self._replay_particle_parts = particle_parts
        self._replay_particle_id_to_index = particle_id_to_index
        self._replay_part_to_indices = part_to_indices

        bc_labels = {}
        for entry in list(self.bcs or []):
            getter = entry.get if hasattr(entry, "get") else None
            kind = str(
                (getter("type") if getter else None)
                or (getter("bc_type") if getter else None)
                or (getter("kind") if getter else None)
                or "bc"
            ).replace("_", " ").strip().title()
            for idx in self._iter_replay_target_indices(entry):
                if 0 <= idx < count:
                    bc_labels.setdefault(int(idx), []).append(kind)

        load_vectors = {}
        for entry in list(self.loads or []):
            vec = self._extract_load_vector_xy(entry)
            if vec.shape[0] < 2:
                continue
            for idx in self._iter_replay_target_indices(entry):
                if not (0 <= idx < count):
                    continue
                existing = load_vectors.get(int(idx))
                if existing is None:
                    load_vectors[int(idx)] = np.asarray(vec[:2], dtype=float)
                else:
                    load_vectors[int(idx)] = np.asarray(existing, dtype=float) + np.asarray(
                        vec[:2], dtype=float
                    )

        self._replay_particle_bc_labels = {
            int(idx): ", ".join(labels) for idx, labels in bc_labels.items()
        }
        self._replay_particle_load_vectors = load_vectors
        self._replay_metadata_particle_count = count

    def _emit_replay_particle_info(self, particle_index=None):
        packet = self._current_frame_packet or {}
        raw_positions = packet.get("raw_positions")
        if raw_positions is None:
            self._replay_selected_particle_index = None
            self.replayParticleSelected.emit({})
            return

        if particle_index is None:
            self._replay_selected_particle_index = None
            self.replayParticleSelected.emit({})
            return

        try:
            idx = int(particle_index)
        except Exception:
            self._replay_selected_particle_index = None
            self.replayParticleSelected.emit({})
            return
        if idx < 0 or idx >= len(raw_positions):
            self._replay_selected_particle_index = None
            self.replayParticleSelected.emit({})
            return

        self._replay_selected_particle_index = idx
        position = np.asarray(raw_positions[idx, :2], dtype=float)
        velocity_field = self._current_animation_velocity
        if velocity_field is not None and idx < len(velocity_field):
            velocity = np.asarray(velocity_field[idx, :2], dtype=float)
        else:
            velocity = np.zeros(2, dtype=float)

        particle_id = idx
        if self._replay_particle_ids is not None and idx < len(self._replay_particle_ids):
            try:
                particle_id = int(self._replay_particle_ids[idx])
            except Exception:
                particle_id = idx
        material = "--"
        if idx < len(self._replay_particle_materials):
            material = str(self._replay_particle_materials[idx] or "--")
        bc_value = str(self._replay_particle_bc_labels.get(idx, "--") or "--")

        self.replayParticleSelected.emit(
            {
                "particle_index": int(idx),
                "particle_id": particle_id,
                "position": (float(position[0]), float(position[1])),
                "velocity": (float(velocity[0]), float(velocity[1])),
                "material": material,
                "bc": bc_value,
            }
        )

    def _update_results_debug_overlays(self):
        self._clear_results_debug_overlays()
        if self.display_mode != "results" or not self.is_visualization_mode:
            return
        packet = self._validated_current_frame_packet()
        if packet is None:
            return
        display_positions = packet["display_positions"]
        display_indices = packet["display_indices"]
        raw_positions = packet["raw_positions"]
        if int(display_positions.shape[0]) <= 0:
            return

        scene = self.scene()
        metrics = self._results_render_metrics(
            particle_count=int(raw_positions.shape[0]),
            display_count=int(display_positions.shape[0]),
        )
        marker_size = float(metrics["overlay_radius"])
        overlay_pen_w = float(metrics["overlay_pen_width"])
        text_step = max(1, int(np.ceil(int(display_positions.shape[0]) / 2000.0)))
        items = []

        for local_idx, raw_idx in enumerate(display_indices.tolist()):
            try:
                x_val, y_val = display_positions[local_idx, :2]
            except Exception:
                continue
            if self._replay_selected_particle_index == int(raw_idx):
                item = scene.addEllipse(
                    x_val - marker_size,
                    y_val - marker_size,
                    marker_size * 2.0,
                    marker_size * 2.0,
                    pen=QPen(QColor(255, 226, 64), max(0.8, overlay_pen_w)),
                )
                items.append(item)
            if self.show_anim_bc_markers and int(raw_idx) in self._replay_particle_bc_labels:
                pen = QPen(QColor(255, 208, 96), max(0.8, overlay_pen_w))
                items.append(
                    scene.addLine(
                        x_val - marker_size,
                        y_val - marker_size,
                        x_val + marker_size,
                        y_val + marker_size,
                        pen,
                    )
                )
                items.append(
                    scene.addLine(
                        x_val - marker_size,
                        y_val + marker_size,
                        x_val + marker_size,
                        y_val - marker_size,
                        pen,
                    )
                )
            if self.show_anim_load_vectors:
                load_vec = self._replay_particle_load_vectors.get(int(raw_idx))
                if load_vec is not None:
                    norm = float(np.hypot(load_vec[0], load_vec[1]))
                    if norm > 1e-12:
                        scale = float(metrics["load_vector_scale"])
                        dx_val = float(load_vec[0]) / norm * scale
                        dy_val = float(load_vec[1]) / norm * scale
                        items.append(
                            scene.addLine(
                                x_val,
                                y_val,
                                x_val + dx_val,
                                y_val + dy_val,
                                QPen(QColor(255, 90, 90), max(0.8, overlay_pen_w)),
                            )
                        )
            if self.show_anim_particle_ids and local_idx % text_step == 0:
                pid_value = int(raw_idx)
                if self._replay_particle_ids is not None and int(raw_idx) < len(self._replay_particle_ids):
                    try:
                        pid_value = int(self._replay_particle_ids[int(raw_idx)])
                    except Exception:
                        pid_value = int(raw_idx)
                text_item = scene.addText(str(pid_value))
                text_item.setDefaultTextColor(QColor(180, 230, 255))
                text_item.setPos(x_val + marker_size, y_val + marker_size)
                items.append(text_item)

        self._results_debug_overlay_items = items

    def set_replay_debug_overlays(
        self,
        show_particles=None,
        show_connections=None,
        show_bc_markers=None,
        show_load_vectors=None,
        show_particle_ids=None,
    ):
        if show_particles is not None or show_connections is not None:
            self.set_animation_visibility(
                show_nodes=show_particles if show_particles is not None else None,
                show_mesh=show_connections if show_connections is not None else None,
            )
        if show_bc_markers is not None:
            self.show_anim_bc_markers = bool(show_bc_markers)
        if show_load_vectors is not None:
            self.show_anim_load_vectors = bool(show_load_vectors)
        if show_particle_ids is not None:
            self.show_anim_particle_ids = bool(show_particle_ids)
        self._update_results_debug_overlays()

    def _normalized_replay_pick_mode(self, mode):
        key = str(mode or "node").strip().lower()
        aliases = {
            "point": "node",
            "node": "node",
            "face": "triangle",
            "triangle": "triangle",
            "edge": "geometry_edge",
            "geometry_edge": "geometry_edge",
            "bc_target": "bc_target",
        }
        return aliases.get(key, "node")

    def _clear_replay_scope_selection(self, *, clear_particle=True):
        self._replay_selected_nodes = set()
        self._replay_selected_triangles = set()
        self._replay_selected_mesh_edges = set()
        self._replay_selected_geometry_edges = []
        self._replay_selected_bc_targets = []
        if clear_particle:
            self._replay_selected_particle_index = None

    def set_replay_pick_mode(self, mode):
        self._replay_pick_mode = self._normalized_replay_pick_mode(mode)
        self._clear_replay_scope_selection(clear_particle=True)
        try:
            self._emit_replay_particle_info(None)
        except Exception:
            pass
        try:
            packet = self._current_frame_packet or {}
            self._update_replay_scope_highlight(packet.get("raw_positions"))
        except Exception:
            pass
        try:
            self.replayScopeSelectionChanged.emit(self.get_replay_scope_selection())
        except Exception:
            pass

    def get_replay_scope_selection(self):
        return {
            "mode": self._replay_pick_mode,
            "nodes": sorted(int(i) for i in self._replay_selected_nodes),
            "triangles": sorted(int(i) for i in self._replay_selected_triangles),
            "mesh_edges": [tuple(int(v) for v in e) for e in self._replay_selected_mesh_edges],
            "edges": [tuple(int(v) for v in e) for e in self._replay_selected_mesh_edges],
            "geometry_edges": [
                (
                    (float(edge[0][0]), float(edge[0][1])),
                    (float(edge[1][0]), float(edge[1][1])),
                )
                for edge in list(self._replay_selected_geometry_edges or [])
                if isinstance(edge, (list, tuple)) and len(edge) == 2
            ],
            "bc_targets": [int(idx) for idx in list(self._replay_selected_bc_targets or [])],
        }

    def _update_replay_scope_highlight(self, current_positions):
        if self._replay_scope_item is not None:
            try:
                if self._replay_scope_item.scene() is self.scene():
                    self.scene().removeItem(self._replay_scope_item)
            except Exception:
                pass
            self._replay_scope_item = None

        current_positions = self._normalize_animation_positions(current_positions)
        if (
            current_positions is None
            and not self._replay_selected_geometry_edges
            and not self._replay_selected_bc_targets
        ):
            return
        if (
            not self._replay_selected_triangles
            and not self._replay_selected_mesh_edges
            and not self._replay_selected_geometry_edges
            and not self._replay_selected_bc_targets
        ):
            return
        elements = getattr(self, "global_elements", [])
        try:
            elem_arr = np.asarray(elements, dtype=int)
        except Exception:
            elem_arr = np.array([])

        path = QPainterPath()
        if self._replay_selected_triangles:
            if current_positions is None or elem_arr.ndim != 2 or elem_arr.shape[1] < 3:
                return
            for idx in sorted(self._replay_selected_triangles):
                if idx < 0 or idx >= len(elem_arr):
                    continue
                tri = elem_arr[idx]
                try:
                    p1 = current_positions[int(tri[0])]
                    p2 = current_positions[int(tri[1])]
                    p3 = current_positions[int(tri[2])]
                except Exception:
                    continue
                path.moveTo(p1[0], p1[1])
                path.lineTo(p2[0], p2[1])
                path.lineTo(p3[0], p3[1])
                path.closeSubpath()
        elif self._replay_selected_mesh_edges:
            if current_positions is None:
                return
            for edge in self._replay_selected_mesh_edges:
                try:
                    i1, i2 = int(edge[0]), int(edge[1])
                    p1 = current_positions[i1]
                    p2 = current_positions[i2]
                except Exception:
                    continue
                path.moveTo(p1[0], p1[1])
                path.lineTo(p2[0], p2[1])
        elif self._replay_selected_geometry_edges:
            for edge in list(self._replay_selected_geometry_edges or []):
                try:
                    p1 = edge[0]
                    p2 = edge[1]
                except Exception:
                    continue
                path.moveTo(float(p1[0]), float(p1[1]))
                path.lineTo(float(p2[0]), float(p2[1]))
        elif self._replay_selected_bc_targets:
            bcs = list(self.bcs or [])
            for index in list(self._replay_selected_bc_targets or []):
                if not (0 <= int(index) < len(bcs)):
                    continue
                bc = bcs[int(index)]
                if bc.get("part_id") is not None and bc.get("coords") in (None, "", []):
                    try:
                        part_id = int(bc.get("part_id"))
                    except Exception:
                        part_id = None
                    if part_id is not None:
                        part = next((p for p in self.parts if getattr(p, "id", None) == part_id), None)
                        boundary_geom = getattr(getattr(part, "geometry", None), "boundary", None)
                        if boundary_geom is not None:
                            for line in self._iter_line_geometries(boundary_geom):
                                try:
                                    coords = list(line.coords)
                                except Exception:
                                    coords = []
                                if len(coords) < 2:
                                    continue
                                path.moveTo(float(coords[0][0]), float(coords[0][1]))
                                for pt in coords[1:]:
                                    path.lineTo(float(pt[0]), float(pt[1]))
                            continue
                shape, geom = self._resolve_attr_marker_geometry(bc)
                if geom is None:
                    continue
                if shape == "point":
                    cx, cy = float(geom[0]), float(geom[1])
                    path.addEllipse(QRectF(cx - 8.0, cy - 8.0, 16.0, 16.0))
                else:
                    p1, p2 = geom
                    path.moveTo(float(p1[0]), float(p1[1]))
                    path.lineTo(float(p2[0]), float(p2[1]))

        if path.isEmpty():
            return
        pen = QPen(QColor(255, 166, 64, 230), 2.0)
        brush = QBrush(QColor(255, 166, 64, 60))
        try:
            item = self.scene().addPath(path, pen, brush)
            self._replay_scope_item = item
        except Exception:
            self._replay_scope_item = None

    def _pick_replay_particle(self, scene_point):
        packet = self._current_frame_packet or {}
        display_positions = packet.get("display_positions")
        display_indices = packet.get("display_indices")
        if display_positions is None or display_indices is None:
            self._clear_replay_scope_selection(clear_particle=True)
            self._emit_replay_particle_info(None)
            try:
                self.replayScopeSelectionChanged.emit(self.get_replay_scope_selection())
            except Exception:
                pass
            self._update_results_debug_overlays()
            return True
        if len(display_positions) <= 0:
            self._clear_replay_scope_selection(clear_particle=True)
            self._emit_replay_particle_info(None)
            try:
                self.replayScopeSelectionChanged.emit(self.get_replay_scope_selection())
            except Exception:
                pass
            self._update_results_debug_overlays()
            return True

        click_xy = np.asarray([[float(scene_point.x()), float(scene_point.y())]], dtype=float)
        deltas = np.asarray(display_positions[:, :2], dtype=float) - click_xy
        distances = np.einsum("ij,ij->i", deltas, deltas)
        nearest_local = int(np.argmin(distances))
        raw_positions = packet.get("raw_positions")
        metrics = self._results_render_metrics(
            particle_count=len(raw_positions) if raw_positions is not None else len(display_positions),
            display_count=len(display_positions),
        )
        radius = float(metrics["pick_radius"])
        if float(distances[nearest_local]) <= float(radius * radius):
            raw_index = int(display_indices[nearest_local])
            self._clear_replay_scope_selection(clear_particle=False)
            self._replay_selected_nodes = {raw_index}
            self._emit_replay_particle_info(raw_index)
        else:
            self._clear_replay_scope_selection(clear_particle=True)
            self._emit_replay_particle_info(None)
        try:
            self.replayScopeSelectionChanged.emit(self.get_replay_scope_selection())
        except Exception:
            pass
        self._update_results_debug_overlays()
        return True

    def _replay_scope_pick_tolerance(self, pixels=6.0):
        try:
            tol = float(self._scene_units_for_pixels(float(pixels)))
        except Exception:
            tol = float(pixels)
        if not math.isfinite(tol) or tol <= 0.0:
            tol = 1.0
        return max(1.0e-6, tol)

    def _pick_replay_geometry_edge(self, scene_point):
        if not self.solid_geometry:
            self._clear_replay_scope_selection(clear_particle=True)
            try:
                self.replayScopeSelectionChanged.emit(self.get_replay_scope_selection())
            except Exception:
                pass
            self._update_replay_scope_highlight(None)
            return False
        _verts, edges = get_solid_features(self.solid_geometry)
        if not edges:
            return False
        pt = (float(scene_point.x()), float(scene_point.y()))
        # Results-scope picking should follow viewport scale, not the large sketch-edit snap radius.
        tol = self._replay_scope_pick_tolerance(6.0)
        best_edge = None
        best_dist = float("inf")
        for edge in edges:
            try:
                seg_dist = float(point_line_dist(pt, edge[0], edge[1]))
            except Exception:
                continue
            if seg_dist < best_dist:
                best_dist = seg_dist
                best_edge = edge
        self._clear_replay_scope_selection(clear_particle=True)
        if best_edge is not None and best_dist <= tol:
            self._replay_selected_geometry_edges = [
                (
                    (float(best_edge[0][0]), float(best_edge[0][1])),
                    (float(best_edge[1][0]), float(best_edge[1][1])),
                )
            ]
        try:
            self.replayScopeSelectionChanged.emit(self.get_replay_scope_selection())
        except Exception:
            pass
        self._update_replay_scope_highlight(None)
        return True

    def _pick_replay_bc_target(self, scene_point):
        point = Point(float(scene_point.x()), float(scene_point.y()))
        tol = self._replay_scope_pick_tolerance(8.0)
        best_index = None
        best_score = float("inf")
        for index, bc in enumerate(list(self.bcs or [])):
            if bc.get("part_id") is not None and bc.get("coords") in (None, "", []):
                try:
                    part_id = int(bc.get("part_id"))
                except Exception:
                    part_id = None
                if part_id is None:
                    continue
                part = next((p for p in self.parts if getattr(p, "id", None) == part_id), None)
                boundary_geom = getattr(getattr(part, "geometry", None), "boundary", None)
                if boundary_geom is None:
                    continue
                try:
                    distance = float(boundary_geom.distance(point))
                except Exception:
                    continue
                if distance <= tol and distance < best_score:
                    best_index = int(index)
                    best_score = distance
                continue
            shape, geom = self._resolve_attr_marker_geometry(bc)
            if geom is None:
                continue
            if shape == "point":
                distance = float(math.hypot(point.x - float(geom[0]), point.y - float(geom[1])))
            else:
                try:
                    distance = float(point_line_dist((point.x, point.y), geom[0], geom[1]))
                except Exception:
                    continue
            if distance <= tol and distance < best_score:
                best_index = int(index)
                best_score = distance
        self._clear_replay_scope_selection(clear_particle=True)
        if best_index is not None:
            self._replay_selected_bc_targets = [int(best_index)]
        try:
            self.replayScopeSelectionChanged.emit(self.get_replay_scope_selection())
        except Exception:
            pass
        self._update_replay_scope_highlight(None)
        return True

    def _pick_replay_scope(self, scene_point):
        mode = self._normalized_replay_pick_mode(getattr(self, "_replay_pick_mode", "node"))
        if mode == "node":
            return self._pick_replay_particle(scene_point)
        if mode == "geometry_edge":
            return self._pick_replay_geometry_edge(scene_point)
        if mode == "bc_target":
            return self._pick_replay_bc_target(scene_point)
        packet = self._current_frame_packet or {}
        current_positions = packet.get("raw_positions")
        if current_positions is None or len(current_positions) == 0:
            return False
        elements = getattr(self, "global_elements", [])
        try:
            elem_arr = np.asarray(elements, dtype=int)
        except Exception:
            elem_arr = np.array([])
        if elem_arr.ndim != 2 or elem_arr.shape[1] < 3:
            return False

        px = float(scene_point.x())
        py = float(scene_point.y())
        tol = float(self._scene_units_for_pixels(6.0))

        def _point_in_tri(p, a, b, c):
            v0x, v0y = c[0] - a[0], c[1] - a[1]
            v1x, v1y = b[0] - a[0], b[1] - a[1]
            v2x, v2y = p[0] - a[0], p[1] - a[1]
            den = v0x * v1y - v1x * v0y
            if abs(den) < 1e-12:
                return False
            u = (v2x * v1y - v1x * v2y) / den
            v = (v0x * v2y - v2x * v0y) / den
            return u >= -1e-6 and v >= -1e-6 and (u + v) <= 1.0 + 1e-6

        def _seg_dist(p, a, b):
            ax, ay = a
            bx, by = b
            px, py = p
            dx, dy = bx - ax, by - ay
            if abs(dx) + abs(dy) < 1e-12:
                return math.hypot(px - ax, py - ay)
            t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
            t = max(0.0, min(1.0, t))
            cx, cy = ax + t * dx, ay + t * dy
            return math.hypot(px - cx, py - cy)

        picked_face = None
        best_dist = float("inf")
        pxy = (px, py)
        for idx, tri in enumerate(elem_arr):
            try:
                i1, i2, i3 = int(tri[0]), int(tri[1]), int(tri[2])
                a = current_positions[i1]
                b = current_positions[i2]
                c = current_positions[i3]
            except Exception:
                continue
            if mode == "triangle":
                if _point_in_tri(pxy, a, b, c):
                    picked_face = int(idx)
                    break
            else:
                d1 = _seg_dist(pxy, a, b)
                d2 = _seg_dist(pxy, b, c)
                d3 = _seg_dist(pxy, c, a)
                dmin = min(d1, d2, d3)
                if dmin < best_dist:
                    best_dist = dmin
                    picked_face = int(idx)

        self._clear_replay_scope_selection(clear_particle=True)
        if mode == "triangle":
            self._replay_selected_triangles = {picked_face} if picked_face is not None else set()
        else:
            if picked_face is None or best_dist > tol:
                self._replay_selected_mesh_edges = set()
            else:
                tri = elem_arr[picked_face]
                i1, i2, i3 = int(tri[0]), int(tri[1]), int(tri[2])
                a = current_positions[i1]
                b = current_positions[i2]
                c = current_positions[i3]
                d1 = _seg_dist(pxy, a, b)
                d2 = _seg_dist(pxy, b, c)
                d3 = _seg_dist(pxy, c, a)
                if d1 <= d2 and d1 <= d3:
                    edge = (i1, i2)
                elif d2 <= d1 and d2 <= d3:
                    edge = (i2, i3)
                else:
                    edge = (i3, i1)
                self._replay_selected_mesh_edges = {tuple(sorted(edge))}

        try:
            self.replayScopeSelectionChanged.emit(self.get_replay_scope_selection())
        except Exception:
            pass
        try:
            packet = self._current_frame_packet or {}
            current_positions = packet.get("raw_positions")
            self._update_replay_scope_highlight(current_positions)
        except Exception:
            pass
        return True

    def first_animation_frame(self):
        self.set_animation_frame(0)

    def previous_animation_frame(self):
        if self._animation_frame_total() <= 0:
            return
        self.set_animation_frame(max(0, int(self.current_frame_index) - 1))

    def next_animation_frame(self):
        total = self._animation_frame_total()
        if total <= 0:
            return
        self.set_animation_frame(min(total - 1, int(self.current_frame_index) + 1))

    def last_animation_frame(self):
        total = self._animation_frame_total()
        if total <= 0:
            return
        self.set_animation_frame(total - 1)

    def _request_lazy_animation_frame(self, index):
        total = self._animation_frame_total()
        controller = self._get_results_controller()
        if controller is None or total <= 0:
            return False
        frame_index = int(max(0, min(index, total - 1)))
        self._pending_lazy_frame_index = frame_index
        self._animation_frame_loading = True
        controller.request_frame(frame_index)
        return True

    def apply_loaded_animation_frame(self, index, frame_data):
        total = self._animation_frame_total()
        if total <= 0:
            return

        frame_index = int(max(0, min(index, total - 1)))
        if isinstance(frame_data, dict):
            positions_data = frame_data.get("positions")
        else:
            positions_data = frame_data
        current_positions = self._normalize_animation_positions(positions_data)
        self._animation_frame_loading = False
        self._pending_lazy_frame_index = None
        if current_positions is None:
            return
        velocity_field = None
        controller = self._get_results_controller()
        previous_cached = controller.get_cached_frame(frame_index - 1) if controller and frame_index > 0 else None
        next_cached = controller.get_cached_frame(frame_index + 1) if controller and frame_index + 1 < total else None
        if previous_cached is not None:
            previous_cached = self._normalize_animation_positions(previous_cached)
            if previous_cached is not None and previous_cached.shape[0] == current_positions.shape[0]:
                velocity_field = current_positions - previous_cached
        elif next_cached is not None:
            next_cached = self._normalize_animation_positions(next_cached)
            if next_cached is not None and next_cached.shape[0] == current_positions.shape[0]:
                velocity_field = next_cached - current_positions
        elif (
            self._current_animation_positions is not None
            and np.asarray(self._current_animation_positions).shape == current_positions.shape
            and abs(int(self.current_frame_index) - frame_index) == 1
        ):
            previous_positions = np.asarray(self._current_animation_positions, dtype=float)
            if frame_index >= int(self.current_frame_index):
                velocity_field = current_positions - previous_positions
            else:
                velocity_field = previous_positions - current_positions

        self._current_animation_velocity = (
            np.asarray(velocity_field[:, :2], dtype=float)
            if velocity_field is not None
            else None
        )
        self.current_frame_index = frame_index
        self.is_visualization_mode = True
        self.display_mode = "results"
        self._ensure_replay_particle_metadata(len(current_positions))
        display_indices = self._display_indices_for_results(len(current_positions))
        if len(display_indices) > 0:
            display_positions = current_positions[display_indices]
        else:
            display_positions = current_positions
        field_packet = controller.get_field_frame(frame_index) if controller is not None else {}
        node_scalar_values = None
        triangle_scalar_values = None
        result_field_label = ""
        if isinstance(field_packet, dict):
            result_field_label = str(field_packet.get("label") or "")
            values = field_packet.get("values")
            domain = field_packet.get("domain")
            if values is not None:
                if domain == "node":
                    values = np.asarray(values, dtype=float).reshape(-1)
                    if values.size == len(current_positions):
                        node_scalar_values = values
                elif domain == "triangle":
                    triangle_scalar_values = np.asarray(values, dtype=float).reshape(-1)
                    if triangle_scalar_values.size == len(current_positions):
                        node_scalar_values = triangle_scalar_values
        if node_scalar_values is None and triangle_scalar_values is not None:
            try:
                elements = np.asarray(getattr(self, "global_elements", []), dtype=int)
                if elements.ndim == 2 and elements.shape[1] >= 3 and len(elements) == len(triangle_scalar_values):
                    sums = np.zeros(len(current_positions), dtype=float)
                    counts = np.zeros(len(current_positions), dtype=float)
                    tri_vals = np.asarray(triangle_scalar_values, dtype=float)
                    tri_nodes = elements[:, :3].astype(int, copy=False)
                    np.add.at(sums, tri_nodes[:, 0], tri_vals)
                    np.add.at(sums, tri_nodes[:, 1], tri_vals)
                    np.add.at(sums, tri_nodes[:, 2], tri_vals)
                    np.add.at(counts, tri_nodes[:, 0], 1.0)
                    np.add.at(counts, tri_nodes[:, 1], 1.0)
                    np.add.at(counts, tri_nodes[:, 2], 1.0)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        node_scalar_values = np.where(counts > 0, sums / counts, np.nan)
            except Exception:
                pass
        self._replay_visible_particle_indices = display_indices
        self._current_animation_positions = current_positions
        self._result_field_label = result_field_label if field_packet.get("key") != "none" else ""
        self._result_scalar_values = node_scalar_values
        self._update_results_legend_from_field_packet(field_packet)
        self._current_frame_packet = {
            "raw_positions": current_positions,
            "display_positions": display_positions,
            "display_indices": display_indices,
            "node_scalar_values": node_scalar_values,
            "display_scalar_values": (
                node_scalar_values[display_indices]
                if node_scalar_values is not None and len(display_indices) > 0
                else node_scalar_values
            ),
            "triangle_scalar_values": triangle_scalar_values,
            "result_field_label": self._result_field_label,
        }

        window = self.window()
        if window is not None and hasattr(window, "_hide_results_point_preview"):
            try:
                window._hide_results_point_preview(clear_points=True, force_view=True)
            except Exception:
                pass

        if not hasattr(self, "_anim_node_items"):
            self._anim_node_items = []
        if not hasattr(self, "_anim_element_items"):
            self._anim_element_items = []
        if not self._anim_node_items and not self._anim_element_items:
            try:
                self.scene().clear()
            except Exception:
                pass
            self._draw_grid()

        original_show_elements = self.show_anim_elements
        if self._replay_lod_active:
            self.show_anim_elements = False
        self._apply_scene_animation_frame(display_positions)
        self.show_anim_elements = original_show_elements
        self._update_results_debug_overlays()
        if self._replay_selected_particle_index is not None:
            self._emit_replay_particle_info(self._replay_selected_particle_index)
        self._results_preview_auto_fit_pending = False
        self.animationFrameChanged.emit(frame_index)

    def refresh_results_field(self):
        if self.display_mode != "results" or not self.is_visualization_mode:
            return
        packet = self._current_frame_packet or {}
        raw_positions = packet.get("raw_positions")
        display_positions = packet.get("display_positions")
        if raw_positions is None or display_positions is None:
            return
        controller = self._get_results_controller()
        if controller is None:
            return
        frame_index = max(0, int(getattr(self, "current_frame_index", 0)))
        field_packet = controller.get_field_frame(frame_index)
        node_scalar_values = None
        triangle_scalar_values = None
        label = ""
        if isinstance(field_packet, dict):
            label = str(field_packet.get("label") or "")
            values = field_packet.get("values")
            domain = field_packet.get("domain")
            if values is not None:
                if domain == "node":
                    arr = np.asarray(values, dtype=float).reshape(-1)
                    if arr.size == len(raw_positions):
                        node_scalar_values = arr
                elif domain == "triangle":
                    triangle_scalar_values = np.asarray(values, dtype=float).reshape(-1)
        display_indices = packet.get("display_indices")
        self._result_field_label = label if field_packet.get("key") != "none" else ""
        self._result_scalar_values = node_scalar_values
        self._update_results_legend_from_field_packet(field_packet)
        packet["node_scalar_values"] = node_scalar_values
        packet["display_scalar_values"] = (
            node_scalar_values[display_indices]
            if node_scalar_values is not None and display_indices is not None and len(display_indices) > 0
            else node_scalar_values
        )
        packet["triangle_scalar_values"] = triangle_scalar_values
        packet["result_field_label"] = self._result_field_label
        self._current_frame_packet = packet
        original_show_elements = self.show_anim_elements
        if self._replay_lod_active:
            self.show_anim_elements = False
        self._apply_scene_animation_frame(display_positions)
        self.show_anim_elements = original_show_elements
        self._update_results_debug_overlays()
        if self._replay_selected_particle_index is not None:
            self._emit_replay_particle_info(self._replay_selected_particle_index)

    def set_animation_playing(self, playing=True):
        if self._animation_frame_total() <= 0:
            self.animation_timer.stop()
            self._emit_animation_playback_state()
            return
        if playing:
            self.ensure_animation_frame_initialized(getattr(self, "current_frame_index", 0))
            self.animation_timer.start(50)
        else:
            self.animation_timer.stop()
        self._emit_animation_playback_state()

    def set_animation_frame(self, index):
        total = self._animation_frame_total()
        if total <= 0:
            return
        idx = int(max(0, min(index, total - 1)))
        self.animation_timer.stop()
        self._emit_animation_playback_state()
        self.is_visualization_mode = True
        if getattr(self, "_lazy_results_enabled", False):
            self.current_frame_index = idx
            self._request_lazy_animation_frame(idx)
            return
        self.current_frame_index = idx - 1
        self.advance_animation_frame()

    def set_animation_visibility(self, show_nodes=None, show_mesh=None):
        if show_nodes is not None:
            self.show_anim_nodes = bool(show_nodes)
        if show_mesh is not None:
            self.show_anim_elements = bool(show_mesh)
        def safe_set_visible(it, state):
            if not it:
                return
            try:
                it.setVisible(state)
            except RuntimeError:
                pass

        if hasattr(self, "_anim_node_items"):
            for item in self._anim_node_items:
                safe_set_visible(item, self.show_anim_nodes)
        if hasattr(self, "_anim_element_items"):
            for item in self._anim_element_items:
                safe_set_visible(item, self.show_anim_elements)
        if hasattr(self, "_smooth_contour_item") and self._smooth_contour_item is not None:
            safe_set_visible(self._smooth_contour_item, self.show_anim_elements)
        if self.display_mode == "results":
            packet = self._validated_current_frame_packet()
            window = self.window()
            if window is not None and hasattr(window, "_hide_results_point_preview"):
                try:
                    window._hide_results_point_preview(force_view=True)
                except Exception:
                    pass
            if packet is not None:
                current_positions = packet["raw_positions"]
                display_positions = packet["display_positions"]
                if self._normalize_animation_positions(getattr(self, "_current_animation_positions", None)) is None:
                    self._current_animation_positions = np.array(current_positions, dtype=float, copy=True)
                original_show_elements = self.show_anim_elements
                if self._replay_lod_active:
                    self.show_anim_elements = False
                self._apply_scene_animation_frame(display_positions)
                self.show_anim_elements = original_show_elements
                self._update_results_debug_overlays()
                if self._replay_selected_particle_index is not None:
                    self._emit_replay_particle_info(self._replay_selected_particle_index)
            else:
                self.ensure_animation_frame_initialized(getattr(self, "current_frame_index", 0))
                self.scene().update()

    def set_animation_element_alpha(self, alpha):
        try:
            alpha = float(alpha)
        except Exception:
            alpha = 0.6
        self.show_anim_element_alpha = max(0.05, min(1.0, alpha))
        if self.display_mode == "results":
            packet = self._validated_current_frame_packet()
            if packet is not None:
                self._apply_scene_animation_frame(packet["display_positions"])
                self._update_results_debug_overlays()
            else:
                self.ensure_animation_frame_initialized(getattr(self, "current_frame_index", 0))

    def set_mesh_view_visibility(self, show_nodes=None, show_mesh=None):
        if show_nodes is not None:
            self.show_mesh_nodes = bool(show_nodes)
        if show_mesh is not None:
            self.show_mesh_elements = bool(show_mesh)
        if self.display_mode in ("mesh", "mesh_3d"):
            self.redraw()

    def set_interface_preview_color(self, color):
        changed = False
        if isinstance(color, QColor):
            new_color = QColor(color)
            if not new_color.isValid():
                return
        elif isinstance(color, str):
            new_color = QColor(color)
            if not new_color.isValid():
                return
        elif isinstance(color, (tuple, list)) and len(color) >= 3:
            try:
                r = int(color[0]); g = int(color[1]); b = int(color[2])
                a = int(color[3]) if len(color) >= 4 else 255
            except Exception:
                return
            new_color = QColor(r, g, b, a)
        else:
            return
        cur = getattr(self, "interface_preview_color", None)
        if not isinstance(cur, QColor) or cur != new_color:
            self.interface_preview_color = new_color
            changed = True
        if changed and self.display_mode in ("mesh", "mesh_3d"):
            self.redraw()

    def set_mesh_preview_style(self, line_width=None, particle_size=None):
        changed = False
        if line_width is not None:
            try:
                lw = float(line_width)
            except Exception:
                lw = self.mesh_preview_line_width
            lw = max(0.1, min(20.0, lw))
            if abs(float(getattr(self, "mesh_preview_line_width", 1.0)) - lw) > 1e-9:
                self.mesh_preview_line_width = lw
                changed = True
        if particle_size is not None:
            try:
                ps = float(particle_size)
            except Exception:
                ps = self.mesh_preview_particle_size
            ps = max(0.5, min(50.0, ps))
            if abs(float(getattr(self, "mesh_preview_particle_size", 3.0)) - ps) > 1e-9:
                self.mesh_preview_particle_size = ps
                changed = True
        if changed and self.display_mode in ("mesh", "mesh_3d"):
            self.redraw()

    def set_gpu_point_preview(self, enabled=None):
        self.set_gpu_point_preview_settings(enabled=enabled)

    def set_gpu_point_preview_settings(self, enabled=None, auto=None, threshold=None):
        changed = False
        if enabled is not None:
            self.gpu_point_preview_enabled = bool(enabled)
            changed = True
        if auto is not None:
            self.gpu_point_preview_auto = bool(auto)
            changed = True
        if threshold is not None:
            try:
                threshold_val = int(threshold)
            except (TypeError, ValueError):
                threshold_val = GPU_POINT_PREVIEW_AUTO_THRESHOLD
            self.gpu_point_preview_threshold = max(0, threshold_val)
            changed = True
        if changed and self.display_mode == "mesh":
            self.redraw()

    def _effective_mesh_elements_visible(self):
        elements = self.global_elements
        if not self.show_mesh_elements or elements is None or len(elements) == 0:
            return False
        max_elements = int(PREVIEW_CONNECTION_LIMIT)
        if bool(getattr(self, "fast_preview_enabled", False)):
            fast_limit = int(getattr(self, "fast_preview_connection_limit", FAST_PREVIEW_CONNECTION_LIMIT))
            if fast_limit <= 0:
                return False
            max_elements = min(max_elements, fast_limit)
        return len(elements) <= max_elements

    def should_use_gpu_point_preview(self):
        if not getattr(self, "gpu_point_preview_enabled", False):
            return False
        max_points = int(getattr(self, "gpu_point_preview_max_points", GPU_POINT_PREVIEW_MAX_POINTS))
        if max_points > 0 and len(self.global_nodes) > max_points:
            return False
        if getattr(self, "gpu_point_preview_auto", False):
            threshold = int(getattr(self, "gpu_point_preview_threshold", 0))
            if threshold > 0 and len(self.global_nodes) < threshold:
                return False
        if self.display_mode != "mesh":
            return False
        if not self.show_mesh_nodes:
            return False
        if self.global_nodes is None or len(self.global_nodes) == 0:
            return False
        if self._effective_mesh_elements_visible():
            return False
        return True

    def should_use_gpu_results_preview(self, node_count=None):
        if not getattr(self, "gpu_point_preview_enabled", False):
            return False
        if self.display_mode != "results":
            return False
        try:
            count = int(node_count if node_count is not None else 0)
        except Exception:
            count = 0
        threshold = max(1, int(getattr(self, "results_preview_threshold", 20000)))
        return count >= threshold

    def should_use_raster_preview(self, node_count, show_elements, show_nodes):
        if not getattr(self, "raster_preview_enabled", False):
            return False
        if self.display_mode != "mesh":
            return False
        if not show_nodes:
            return False
        threshold = int(getattr(self, "raster_preview_threshold", 0))
        if threshold > 0 and node_count < threshold:
            return False
        if show_elements:
            return False
        return True

    def _draw_raster_particles_preview(self, nodes):
        if nodes is None or len(nodes) == 0:
            return
        try:
            vp = self.viewport().size()
            width = max(1, int(vp.width()))
            height = max(1, int(vp.height()))
            max_pixels = int(getattr(self, "raster_preview_max_pixels", RASTER_PREVIEW_MAX_PIXELS))
            if max_pixels > 0 and width * height > max_pixels:
                scale = math.sqrt(max_pixels / float(width * height))
                width = max(1, int(width * scale))
                height = max(1, int(height * scale))
            rect = self.scene().sceneRect()
            left, right = float(rect.left()), float(rect.right())
            top, bottom = float(rect.top()), float(rect.bottom())
            span_x = right - left
            span_y = bottom - top
            if span_x <= 0 or span_y <= 0:
                return

            pts = np.asarray(nodes, dtype=float)
            x = pts[:, 0]
            y = pts[:, 1]
            x_norm = (x - left) / span_x
            y_norm = (y - top) / span_y
            mask = (x_norm >= 0.0) & (x_norm <= 1.0) & (y_norm >= 0.0) & (y_norm <= 1.0)
            if not np.any(mask):
                return
            x_norm = x_norm[mask]
            y_norm = y_norm[mask]
            xi = np.clip((x_norm * (width - 1)).astype(np.int32), 0, width - 1)
            yi = np.clip((y_norm * (height - 1)).astype(np.int32), 0, height - 1)

            img = np.zeros((height, width, 4), dtype=np.uint8)
            img[yi, xi] = (35, 35, 35, 255)

            # Detach from the NumPy buffer to avoid use-after-free in Qt paint.
            qimg = QImage(img.data, width, height, QImage.Format_RGBA8888).copy()
            pix = QPixmap.fromImage(qimg)
            item = self.scene().addPixmap(pix)
            try:
                item.setTransformationMode(Qt.FastTransformation)
            except Exception:
                pass
            item.setTransform(QTransform(span_x / width, 0, 0, span_y / height, left, top))
            item.setZValue(-5)
        except Exception:
            return

    def set_fast_preview(self, enabled=None, limit=None):
        changed = False
        if enabled is not None:
            self.fast_preview_enabled = bool(enabled)
            changed = True
        if limit is not None:
            try:
                limit_val = int(limit)
            except (TypeError, ValueError):
                limit_val = FAST_PREVIEW_CONNECTION_LIMIT
            self.fast_preview_connection_limit = max(0, limit_val)
            changed = True
        if changed and self.display_mode in ("mesh", "mesh_3d"):
            self.redraw()

    def center_origin(self):
        self.centerOn(0, 0)

    def set_paint_brush(self, brush_type, fx=0.0, fy=0.0, val=0.0):
        self.paint_brush = {
            "type": brush_type,
            "fx": float(fx),
            "fy": float(fy),
            "val": float(val),
        }

    def _project_to_edge(self, pt, a, b):
        ax, ay = a
        bx, by = b
        px, py = pt
        dx = bx - ax
        dy = by - ay
        if dx == 0 and dy == 0:
            return a
        t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        return (ax + t * dx, ay + t * dy)

    def _segment_intersection(self, a, b, c, d):
        ax, ay = a
        bx, by = b
        cx, cy = c
        dx, dy = d
        den = (ax - bx) * (cy - dy) - (ay - by) * (cx - dx)
        if abs(den) < 1e-9:
            return None
        t = ((ax - cx) * (cy - dy) - (ay - cy) * (cx - dx)) / den
        u = ((ax - cx) * (ay - by) - (ay - cy) * (ax - bx)) / den
        if 0 <= t <= 1 and 0 <= u <= 1:
            return (ax + t * (bx - ax), ay + t * (by - ay))
        return None

    def _snap_paint_point(self, pt):
        if not self.solid_geometry:
            return pt
        _, edges = get_solid_features(self.solid_geometry)
        best = pt
        best_dist = SNAP_TOL
        for edge in edges:
            proj = self._project_to_edge(pt, edge[0], edge[1])
            d = dist(pt, proj)
            if d < best_dist:
                best_dist = d
                best = proj
        return best

    def _start_paint(self, pt):
        self._paint_active = True
        self._paint_points = []
        self._paint_preview_item = None
        snap_pt = self._snap_paint_point(pt)
        self._paint_points.append(snap_pt)
        self._update_paint_preview()

    def _update_paint(self, pt):
        if not self._paint_active:
            return
        snap_pt = self._snap_paint_point(pt)
        if not self._paint_points or dist(self._paint_points[-1], snap_pt) >= SNAP_TOL * 0.5:
            self._paint_points.append(snap_pt)
            self._update_paint_preview()

    def _finish_paint(self):
        if not self._paint_active:
            return
        self._paint_active = False
        if len(self._paint_points) < 2:
            self._clear_paint_preview()
            return

        segments = list(zip(self._paint_points[:-1], self._paint_points[1:]))
        brush_type = self.paint_brush.get("type")
        if not brush_type:
            self._clear_paint_preview()
            return

        self.push_undo_state()
        if brush_type in ("fix_xy", "fix_x", "fix_y", "fix_z", "velocity_x", "velocity_y", "velocity_z"):
            val = self.paint_brush.get("val", 0.0)
            for seg in segments:
                bc = {"type": brush_type, "coords": seg}
                if "velocity" in brush_type:
                    bc["val"] = val
                self.bcs.append(bc)
            self.bcsChanged.emit()
        elif brush_type == "force":
            fx = self.paint_brush.get("fx", 0.0)
            fy = self.paint_brush.get("fy", 0.0)
            for seg in segments:
                self.loads.append({"type": "force", "coords": seg, "fx": fx, "fy": fy})
            self.loadsChanged.emit()
        elif brush_type == "force_z":
            val = self.paint_brush.get("val", 0.0)
            for seg in segments:
                self.loads.append({"type": "force", "coords": seg, "fx": 0.0, "fy": 0.0, "fz": val, "axis": "z"})
            self.loadsChanged.emit()
        elif brush_type == "moment":
            val = self.paint_brush.get("val", 0.0)
            for seg in segments:
                self.loads.append({"type": "moment", "coords": seg, "m": val})
            self.loadsChanged.emit()

        self._clear_paint_preview()
        self.redraw()

    def _update_paint_preview(self):
        self._clear_paint_preview()
        if len(self._paint_points) < 2:
            return
        path = QPainterPath()
        path.moveTo(*self._paint_points[0])
        for p in self._paint_points[1:]:
            path.lineTo(*p)
        pen = QPen(QColor(0, 120, 255), 2, Qt.DashLine)
        self._paint_preview_item = self.scene().addPath(path, pen)

    def _clear_paint_preview(self):
        self._remove_scene_item_safe(self._paint_preview_item)
        self._paint_preview_item = None

    def set_snap_grid(self, enabled):
        self.snap_grid = bool(enabled)

    def set_snap_endpoints(self, enabled):
        self.snap_endpoints = bool(enabled)

    def set_endpoint_snap(self, enabled):
        self.set_snap_endpoints(enabled)

    def set_midpoint_snap(self, enabled):
        self.snap_midpoints = bool(enabled)

    def set_angle_snap(self, enabled):
        self.snap_angle = bool(enabled)

    def set_precision_sketch_mode(self, enabled):
        enabled = bool(enabled) and str(getattr(self, "tool", "")).lower() != "freeform"
        self.precision_sketch_mode_enabled = enabled
        self.set_dimensions_visible(enabled)
        window = self.window()
        if window and hasattr(window, "_update_interaction_hints"):
            try:
                window._update_interaction_hints()
            except Exception:
                pass

    def set_parametric_mode(self, enabled):
        self.parametric_enabled = bool(enabled)

    def set_freeform_auto_convert(self, enabled, announce=True):
        self.freeform_auto_convert_enabled = bool(enabled)
        if announce:
            state = "enabled" if self.freeform_auto_convert_enabled else "disabled"
            self._announce_status(f"Freeform auto-convert {state}.")

    def _default_sketch_tool_name(self):
        # The canvas defaults to the hand/select tool so the user can pan,
        # click, and drag parts. Drawing tools activate only when the user
        # explicitly clicks a shape button or template.
        return "select"

    def restore_default_sketch_tool(self):
        if str(getattr(self, "active_module", "")).lower() == "part":
            self.set_tool(self._default_sketch_tool_name())

    def restore_default_freeform_tool(self):
        self.restore_default_sketch_tool()

    def _default_scene_rect(self):
        return QRectF(-SCENE_EXTENT, -SCENE_EXTENT, 2.0 * SCENE_EXTENT, 2.0 * SCENE_EXTENT)

    def _scene_rect_for_model(self, rect=None):
        target_rect = rect if rect is not None else self._model_fit_rect()
        default_rect = self._default_scene_rect()
        if target_rect is None or target_rect.width() <= 0.0 or target_rect.height() <= 0.0:
            return default_rect
        # Pad the model bounds by 5x in each direction so the user has plenty of
        # room to pan, but keep the scene scaled to the model — otherwise the
        # huge default extent makes scrollbar thumb drags jump enormous
        # distances relative to the model.
        target = QRectF(target_rect)
        pad_x = max(target.width() * 5.0, 1.0)
        pad_y = max(target.height() * 5.0, 1.0)
        padded = target.adjusted(-pad_x, -pad_y, pad_x, pad_y)
        return padded

    def _sync_scene_rect_to_model(self, rect=None, force=False):
        target = self._scene_rect_for_model(rect)
        current = QRectF(self.sceneRect())
        if force or current.width() <= 0.0 or current.height() <= 0.0 or not current.contains(target):
            self.setSceneRect(target)
            return target
        return current

    def set_material_paint_mode(self, enabled):
        self.material_paint_mode = bool(enabled)
        self.setCursor(Qt.CrossCursor if self.material_paint_mode else Qt.ArrowCursor)

    def set_unit(self, unit):
        unit = self._normalize_length_unit(unit)
        if unit == self.current_unit:
            return
        self.current_unit = unit
        self.geometryChanged.emit()
        self.redraw()   

    def _normalize_length_unit(self, unit=None):
        unit_key = str(self.current_unit if unit is None else unit).strip().lower()
        return unit_key if unit_key in {"mm", "cm", "m"} else "mm"

    def _length_unit_scale_to_meters(self, unit):
        return {
            "mm": 0.001,
            "cm": 0.01,
            "m": 1.0,
        }.get(str(unit or "").strip().lower())

    def _part_uses_legacy_ui_storage(self, part):
        if part is None:
            return True
        marker = str(getattr(part, "storage_units", "") or "").strip().lower()
        return marker != "m"

    def _owner_uses_si_units(self, owner_type, owner_part=None):
        return False

    def _owner_value_to_ui(self, value, owner_type, owner_part=None):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        return numeric

    def _owner_value_from_ui(self, value, owner_type, owner_part=None):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        return numeric

    def _scale_point_tuple(self, point, scale):
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return point
        try:
            return (float(point[0]) * scale, float(point[1]) * scale)
        except (TypeError, ValueError):
            return point

    def _scale_points_collection(self, points, scale):
        return [self._scale_point_tuple(pt, scale) for pt in (points or [])]

    def _scale_sketch_meta_for_ui(self, meta, scale):
        meta_copy = copy.deepcopy(meta or {})
        for key in ("center", "origin", "p1", "p2"):
            if meta_copy.get(key) is not None:
                meta_copy[key] = self._scale_point_tuple(meta_copy.get(key), scale)
        if meta_copy.get("points") is not None:
            meta_copy["points"] = self._scale_points_collection(meta_copy.get("points"), scale)
        for key in ("width", "height", "radius"):
            if meta_copy.get(key) is not None:
                try:
                    meta_copy[key] = float(meta_copy.get(key)) * scale
                except (TypeError, ValueError):
                    pass
        return meta_copy

    def _scale_dimensions_for_ui(self, dimensions, scale):
        scaled = copy.deepcopy(dimensions or [])
        for dim in scaled:
            if str(dim.get("dim_type", "")).lower() != "angle" and dim.get("value") is not None:
                try:
                    dim["value"] = float(dim.get("value")) * scale
                except (TypeError, ValueError):
                    pass
            if dim.get("offset") is not None:
                dim["offset"] = self._scale_point_tuple(dim.get("offset"), scale)
        return scaled

    def _normalize_part_storage_units(self, part):
        if part is None:
            return
        marker = str(getattr(part, "storage_units", "") or "").strip().lower()
        if marker in {"", "ui", "display"}:
            part.storage_units = "ui"
            return
        unit_scale = self._length_unit_scale_to_meters(marker)
        current_scale = self._unit_scale_to_meters()
        if unit_scale is None or current_scale <= 0.0:
            part.storage_units = "ui"
            return
        scale_to_ui = float(unit_scale) / float(current_scale)
        if abs(scale_to_ui - 1.0) > 1e-12:
            geom = getattr(part, "geometry", None)
            if geom is not None and not getattr(geom, "is_empty", True):
                try:
                    part.geometry = shp_scale(geom, xfact=scale_to_ui, yfact=scale_to_ui, origin=(0.0, 0.0))
                except Exception:
                    pass
            part.sketches = [
                self._scale_points_collection(sketch, scale_to_ui)
                for sketch in (getattr(part, "sketches", []) or [])
            ]
            part.sketch_meta = [
                self._scale_sketch_meta_for_ui(meta, scale_to_ui)
                for meta in (getattr(part, "sketch_meta", []) or [])
            ]
            part.dimensions = self._scale_dimensions_for_ui(
                getattr(part, "dimensions", []) or [],
                scale_to_ui,
            )
        part.storage_units = "ui"

    def _practical_sketch_default(self, fallback, current=None):
        try:
            numeric = float(current)
            if math.isfinite(numeric) and 100.0 <= numeric <= 500.0:
                return numeric
        except (TypeError, ValueError):
            pass
        return float(fallback)

    def set_selected_part(self, part_id, emit_signal=False):
        if part_id == self.selected_part_id:
            return
        self.selected_part_id = part_id
        window = self.window()
        if window and hasattr(window, "_update_interaction_hints"):
            try:
                window._update_interaction_hints()
            except Exception:
                pass
        if self.active_module == "Property" and part_id is not None:
            part = next((p for p in self.parts if p.id == part_id), None)
            if part:
                self._sync_material_panel_from_part(part)
        if emit_signal:
            self.partSelectionChanged.emit(part_id)
        self.redraw()

    def get_selected_part(self):
        if self.selected_part_id is None:
            return None
        return next((p for p in self.parts if p.id == self.selected_part_id), None)

    def _clear_panel_attr_focus(self):
        self._panel_attr_focus_kind = None
        self._panel_attr_focus_entry_ref = None
        self._apply_panel_attr_focus_to_3d(None, None)
        self.redraw()

    def _is_panel_attr_focus(self, kind, entry):
        return (
            entry is not None
            and self._panel_attr_focus_kind == kind
            and self._panel_attr_focus_entry_ref is entry
        )

    def _panel_interaction_part_ids(self, entry):
        if entry is None:
            return ()
        values = []
        for key in ("part1_id", "part2_id"):
            try:
                raw = entry.get(key) if isinstance(entry, dict) else getattr(entry, key, None)
                if raw not in (None, "", -1):
                    values.append(int(raw))
            except Exception:
                continue
        deduped = []
        seen = set()
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        return tuple(deduped)

    def _panel_interaction_node_ids(self, entry):
        if entry is None or self.global_elements is None or len(self.global_elements) == 0:
            return []
        try:
            iface_id = int(entry.get("id") if isinstance(entry, dict) else getattr(entry, "id", -1))
        except Exception:
            return []
        try:
            rows = self._get_interface_preview_rows()
        except Exception:
            return []
        if not rows:
            return []
        node_ids = set()
        elem_count = len(self.global_elements)
        for row in rows:
            try:
                row_iface_id = int(row.get("interface_id"))
                elem_idx = int(row.get("_element_idx"))
            except Exception:
                continue
            if row_iface_id != iface_id or elem_idx < 0 or elem_idx >= elem_count:
                continue
            try:
                tri = self.global_elements[elem_idx]
            except Exception:
                continue
            for node_id in tri[:3]:
                try:
                    node_ids.add(int(node_id))
                except Exception:
                    continue
        return sorted(node_ids)

    def _apply_panel_attr_focus_to_3d(self, kind, entry):
        main = self.window()
        if not main:
            return
        view_3d = getattr(main, "view_3d", None)
        if view_3d is None or not bool(getattr(main, "_workspace_3d", False)):
            return
        if not hasattr(view_3d, "highlight_node_ids"):
            return
        if entry is None:
            try:
                view_3d.highlight_node_ids([], target="auto")
            except Exception:
                pass
            return
        if str(kind or "").lower() == "interaction":
            node_ids = self._panel_interaction_node_ids(entry)
            try:
                view_3d.highlight_node_ids(node_ids, target="auto")
            except Exception:
                pass
            if node_ids:
                try:
                    if hasattr(view_3d, "focus_node_ids"):
                        view_3d.focus_node_ids(node_ids)
                except Exception:
                    pass
            return
        ids = entry.get("ids")
        target = entry.get("target", "auto")
        if ids is not None:
            try:
                view_3d.highlight_node_ids(ids, target=target)
            except Exception:
                pass
            try:
                if hasattr(view_3d, "focus_node_ids"):
                    view_3d.focus_node_ids(ids)
            except Exception:
                pass
        else:
            try:
                view_3d.highlight_node_ids([], target="auto")
            except Exception:
                pass

    def _focus_panel_attr_entry_2d(self, entry):
        if entry is None:
            return
        part_id = entry.get("part_id")
        if part_id is not None:
            try:
                pid = int(part_id)
            except Exception:
                pid = None
            if pid is not None:
                part = next((p for p in self.parts if p.id == pid), None)
                if part and getattr(part, "geometry", None) is not None and not part.geometry.is_empty:
                    try:
                        minx, miny, maxx, maxy = part.geometry.bounds
                        rect = QRectF(minx, miny, maxx - minx, maxy - miny)
                        if rect.width() > 0 and rect.height() > 0:
                            pad = max(rect.width(), rect.height()) * 0.08
                            if pad <= 0:
                                pad = 10.0
                            self._apply_fit_rect(rect.adjusted(-pad, -pad, pad, pad))
                            return
                    except Exception:
                        pass

        coords = entry.get("coords")
        if isinstance(coords, np.ndarray):
            coords = coords.tolist()
        if isinstance(coords, (list, tuple)):
            # Point coordinate
            if (
                len(coords) >= 2
                and isinstance(coords[0], (int, float))
                and isinstance(coords[1], (int, float))
            ):
                x = float(coords[0])
                y = float(coords[1])
                pad = 25.0
                self._apply_fit_rect(QRectF(x - pad, y - pad, pad * 2.0, pad * 2.0))
                return
            # Segment/polyline first segment
            if (
                len(coords) >= 2
                and isinstance(coords[0], (list, tuple, np.ndarray))
                and isinstance(coords[1], (list, tuple, np.ndarray))
                and len(coords[0]) >= 2
                and len(coords[1]) >= 2
            ):
                try:
                    pts = np.asarray(coords, dtype=float)
                except Exception:
                    pts = None
                if pts is not None and pts.ndim == 2 and pts.shape[1] >= 2:
                    if self._fit_points(pts[:, :2]):
                        return

        ids = entry.get("ids")
        if ids is not None and len(ids) > 0 and self.global_nodes is not None:
            try:
                nodes = np.asarray(self.global_nodes, dtype=float)
            except Exception:
                nodes = None
            if nodes is not None and nodes.ndim == 2 and nodes.shape[1] >= 2 and len(nodes) > 0:
                valid = []
                for nid in ids:
                    try:
                        idx = int(nid)
                    except Exception:
                        continue
                    if 0 <= idx < len(nodes):
                        valid.append(idx)
                if valid:
                    self._fit_points(nodes[np.asarray(sorted(set(valid)), dtype=int), :2])

    def _focus_panel_attr_entry(self, kind, entry):
        kind = str(kind or "").lower()
        main = self.window()
        if main and bool(getattr(main, "_workspace_3d", False)):
            self._apply_panel_attr_focus_to_3d(kind, entry)
            return
        if kind == "interaction":
            part_ids = self._panel_interaction_part_ids(entry)
            points = []
            for part_id in part_ids:
                part = next((p for p in self.parts if getattr(p, "id", None) == part_id), None)
                geom = getattr(part, "geometry", None) if part is not None else None
                if geom is None or getattr(geom, "is_empty", True):
                    continue
                try:
                    minx, miny, maxx, maxy = geom.bounds
                except Exception:
                    continue
                points.extend(
                    [
                        (float(minx), float(miny)),
                        (float(maxx), float(maxy)),
                    ]
                )
            if points and self._fit_points(np.asarray(points, dtype=float)):
                return
        self._focus_panel_attr_entry_2d(entry)

    def set_panel_attr_focus(self, kind=None, entry=None):
        if not kind or entry is None:
            self._clear_panel_attr_focus()
            return
        kind = str(kind).lower()
        if kind not in {"bc", "load", "interaction"}:
            self._clear_panel_attr_focus()
            return
        self._panel_attr_focus_kind = kind
        self._panel_attr_focus_entry_ref = entry
        part_id = entry.get("part_id") if isinstance(entry, dict) else getattr(entry, "part_id", None)
        if kind in {"bc", "load"} and part_id is not None and not bool(getattr(self.window(), "_workspace_3d", False)):
            try:
                self.set_selected_part(int(part_id), emit_signal=False)
            except Exception:
                pass
        self._focus_panel_attr_entry(kind, entry)
        self.redraw()

    def _snapshot_state(self):
        return {
            "sketches": copy.deepcopy(self.sketches),
            "sketch_meta": copy.deepcopy(self.sketch_meta),
            "dimensions": copy.deepcopy(self.dimensions),
            "constraints": copy.deepcopy(self.constraints),
            "show_dimensions": self.show_dimensions,
            "dimension_id_counter": self._dimension_id_counter,
            "parts": self.serialize_geometry(),
            "materials": self.serialize_materials(),
            "bcs": copy.deepcopy(self.bcs),
            "loads": copy.deepcopy(self.loads),
            "interfaces": self.serialize_interfaces(),
            "initial_velocities": copy.deepcopy(self.initial_velocities),
            "current_unit": self._normalize_length_unit(),
            "operations": [
                {
                    "id": op.id,
                    "shape_data": copy.deepcopy(op.shape_data),
                    "op_type": op.op_type,
                    "material_id": op.material_id,
                }
                for op in self.operations
            ],
        }

    @property
    def interactions(self):
        """Backward-compatible alias; canonical name is `interfaces`."""
        return self.interfaces

    @interactions.setter
    def interactions(self, value):
        self.interfaces = list(value) if isinstance(value, (list, tuple)) else []

    def _emit_interfaces_changed(self):
        self._interface_preview_cache = None
        self._interface_preview_cache_sig = None
        self.interfacesChanged.emit()
        self.interactionsChanged.emit()
        self.redraw()

    def _clear_part_shape_edit_session(self):
        self._editing_part_shape_id = None
        self._sketch_edit_mode = False
        if self.display_mode == "sketch_edit":
            self.set_display_mode("geometry")

    def in_sketch_edit_mode(self):
        return bool(getattr(self, "_sketch_edit_mode", False))

    def enter_sketch_edit_mode(self, part_id):
        target_part = None
        try:
            target_id = int(part_id)
        except Exception:
            return False
        target_part = next((p for p in self.parts if int(getattr(p, "id", -1)) == target_id), None)
        if target_part is None:
            return False
        self._editing_part_shape_id = target_id
        self._sketch_edit_mode = True
        self.set_selected_part(target_id, emit_signal=True)
        self.set_navigation_mode(False)
        self.set_module("Part")
        self.set_display_mode("sketch_edit")
        self.set_tool("select")
        self.set_precision_sketch_mode(True)
        window = self.window()
        if window is not None and hasattr(window, "_on_sketch_edit_mode_changed"):
            try:
                window._on_sketch_edit_mode_changed(True, target_part)
            except Exception:
                pass
        self.geometryChanged.emit()
        self.redraw()
        return True

    def exit_sketch_edit_mode(self):
        target_part = self.get_part_shape_edit_target()
        self._sketch_edit_mode = False
        if self.display_mode == "sketch_edit":
            self.set_display_mode("geometry")
        self.set_navigation_mode(False)
        self.set_precision_sketch_mode(False)
        self.restore_default_sketch_tool()
        window = self.window()
        if window is not None and hasattr(window, "_on_sketch_edit_mode_changed"):
            try:
                window._on_sketch_edit_mode_changed(False, target_part)
            except Exception:
                pass

    def get_part_shape_edit_target(self):
        target_id = getattr(self, "_editing_part_shape_id", None)
        if target_id in (None, ""):
            return None
        try:
            tid = int(target_id)
        except Exception:
            return None
        return next((p for p in self.parts if int(getattr(p, "id", -1)) == tid), None)

    def begin_part_shape_edit(self, part=None):
        """
        Load a part's stored sketches into the active sketch buffer for editing.

        Confirm Part will update the same part in-place instead of creating a new part
        while this edit session is active.
        """
        if part is None:
            part = self.get_selected_part()
        if part is None:
            QMessageBox.warning(self, "Edit Shape", "Select a part first.")
            return False
        if str(getattr(part, "part_type", "")).lower() == "particle_set":
            QMessageBox.information(
                self,
                "Particle Set",
                "Particles are generated from geometry. Edit sketch to modify.",
            )
            return False
        if getattr(part, "is_void", False):
            QMessageBox.warning(self, "Edit Shape", "Editing hole/void shapes in sketch mode is not supported yet.")
            return False
        self._normalize_part_storage_units(part)
        sketches = copy.deepcopy(getattr(part, "sketches", []) or [])
        generated_from_geometry = False
        generated_payload = None
        if not sketches:
            generated_payload = self._geometry_to_editable_boundary_sketches(getattr(part, "geometry", None))
            if generated_payload is None:
                msg = (
                    "This part does not have editable sketch data, and its geometry cannot be safely converted "
                    "to sketch-edit mode.\n\n"
                    "Current limitation: parts with internal holes/void loops inside the same part geometry "
                    "cannot yet be edited as a generated sketch."
                )
                QMessageBox.information(self, "Edit Shape", msg)
                return False
            sketches = copy.deepcopy(generated_payload.get("sketches", []) or [])
            generated_from_geometry = True
        if self.sketches:
            reply = QMessageBox.question(
                self,
                "Replace Current Sketch",
                "Current sketch entities will be replaced with the selected part's sketch for editing.\nContinue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return False
        scale_to_ui = 1.0 if self._part_uses_legacy_ui_storage(part) else (1.0 / self._unit_scale_to_meters())
        self.sketches = self._scale_points_collection(sketches, scale_to_ui)
        if generated_from_geometry:
            payload = generated_payload or {}
            self.sketch_meta = [
                self._scale_sketch_meta_for_ui(meta, scale_to_ui)
                for meta in (payload.get("sketch_meta", []) or [])
            ]
            self.dimensions = []
            self.constraints = []
        else:
            self.sketch_meta = [
                self._scale_sketch_meta_for_ui(meta, scale_to_ui)
                for meta in (getattr(part, "sketch_meta", []) or [])
            ]
            self.dimensions = self._scale_dimensions_for_ui(
                getattr(part, "dimensions", []) or [],
                scale_to_ui,
            )
            self.constraints = copy.deepcopy(getattr(part, "constraints", []) or [])
        self._sync_all_sketch_meta()
        self._ensure_dimensions()
        self._porous_sketch_name = None
        self.enter_sketch_edit_mode(int(getattr(part, "id")))
        window = self.window()
        if window and hasattr(window, "statusBar"):
            if generated_from_geometry:
                window.statusBar().showMessage(
                    f"Editing shape of '{getattr(part, 'name', 'Part')}' from generated boundary sketch. "
                    "Right-click a sketch shape to edit parameters, then use Confirm Part to update it.",
                    9000,
                )
            else:
                window.statusBar().showMessage(
                    f"Editing shape of '{getattr(part, 'name', 'Part')}'. Right-click a sketch shape to edit parameters, then use Confirm Part.",
                    9000,
                )
        return True

    def _restore_state(self, state):
        self._history_blocked = True
        try:
            self._clear_part_shape_edit_session()
            self.current_unit = self._normalize_length_unit(state.get("current_unit", "mm"))
            self.sketches = copy.deepcopy(state.get("sketches", []))
            self.sketch_meta = copy.deepcopy(state.get("sketch_meta", []))
            self.dimensions = copy.deepcopy(state.get("dimensions", []))
            self.constraints = copy.deepcopy(state.get("constraints", []))
            self.show_dimensions = bool(state.get("show_dimensions", True))
            self._dimension_id_counter = int(state.get("dimension_id_counter", 0))
            self.deserialize_materials(state.get("materials", {}))
            self.deserialize_geometry(state.get("parts", {}))
            self._sync_all_sketch_meta()
            self._ensure_dimensions()
            self._recalc_dimension_counter()
            self.operations = []
            max_op_id = 0
            for op_data in state.get("operations", []):
                op = Operation(
                    copy.deepcopy(op_data.get("shape_data", {})),
                    op_data.get("op_type", "ADD"),
                    op_data.get("material_id"),
                )
                op.id = op_data.get("id", op.id)
                max_op_id = max(max_op_id, op.id)
                self.operations.append(op)
            Operation._op_counter = max_op_id
            self.bcs = copy.deepcopy(state.get("bcs", []))
            self.loads = copy.deepcopy(state.get("loads", []))
            self._sanitize_bc_load_entries()
            self.initial_velocities = copy.deepcopy(state.get("initial_velocities", []))
            max_force_id = 0
            for ld in self.loads:
                try:
                    max_force_id = max(max_force_id, int(ld.get("force_id", 0)))
                except Exception:
                    continue
            self._next_force_id = max_force_id + 1
            max_vel_id = 0
            for bc in self.bcs:
                try:
                    max_vel_id = max(max_vel_id, int(bc.get("vel_id", 0)))
                except Exception:
                    continue
            self._next_velocity_id = max_vel_id + 1

            interface_items = state.get("interfaces")
            if interface_items is None:
                interface_items = state.get("interactions", [])
            self.deserialize_interfaces(interface_items)

            self.part_meshes.clear()
            self.global_nodes = np.array([])
            self.global_elements = np.array([])
            self.element_part_map = []
            self.selected_part_id = None
            self.material_color_map = {}
            self.rebuild_display_geometry()
            self.redraw()
        finally:
            self._history_blocked = False

        self.partsChanged.emit()
        self.materialsChanged.emit()
        self._emit_interfaces_changed()
        self.geometryChanged.emit()
        self.bcsChanged.emit()
        self.loadsChanged.emit()

    def _normalize_attr_entry(self, entry):
        if not isinstance(entry, dict):
            return None
        out = copy.deepcopy(entry)

        coords = out.get("coords", [])
        if isinstance(coords, np.ndarray):
            coords = coords.tolist()
        if isinstance(coords, tuple):
            coords = list(coords)
        norm_coords = []
        if isinstance(coords, (list, tuple)):
            if (
                len(coords) >= 2
                and isinstance(coords[0], (int, float))
                and isinstance(coords[1], (int, float))
            ):
                norm_coords = [float(coords[0]), float(coords[1])]
            elif (
                len(coords) >= 2
                and isinstance(coords[0], (list, tuple, np.ndarray))
                and isinstance(coords[1], (list, tuple, np.ndarray))
                and len(coords[0]) >= 2
                and len(coords[1]) >= 2
            ):
                norm_coords = [
                    [float(coords[0][0]), float(coords[0][1])],
                    [float(coords[1][0]), float(coords[1][1])],
                ]
        out["coords"] = norm_coords

        ids = out.get("ids")
        norm_ids = []
        if ids is not None:
            if not isinstance(ids, (list, tuple, set, np.ndarray)):
                ids = [ids]
            for nid in ids:
                try:
                    norm_ids.append(int(nid))
                except Exception:
                    continue
            norm_ids = sorted(set(norm_ids))
            if norm_ids:
                out["ids"] = norm_ids
            else:
                out.pop("ids", None)

        part_id = out.get("part_id")
        if part_id is not None:
            try:
                out["part_id"] = int(part_id)
            except Exception:
                out.pop("part_id", None)

        if not out.get("coords") and not out.get("ids") and out.get("part_id") is None:
            return None
        return out

    def _sanitize_bc_load_entries(self):
        bcs_clean = []
        loads_clean = []

        for bc in self.bcs:
            norm = self._normalize_attr_entry(bc)
            if norm is not None:
                bcs_clean.append(norm)

        for ld in self.loads:
            norm = self._normalize_attr_entry(ld)
            if norm is not None:
                loads_clean.append(norm)

        self.bcs = bcs_clean
        self.loads = loads_clean

    def push_undo_state(self):
        if self._history_blocked:
            return
        self.undo_stack.append(self._snapshot_state())
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack:
            return
        current = self._snapshot_state()
        state = self.undo_stack.pop()
        self.redo_stack.append(current)
        self._restore_state(state)

    def redo(self):
        if not self.redo_stack:
            return
        current = self._snapshot_state()
        state = self.redo_stack.pop()
        self.undo_stack.append(current)
        self._restore_state(state)

    def _resolve_material_target(self, part):
        """If part is a void with a solid parent, return the parent so material
        assignment lands on the surrounding shape (the void area is naturally
        excluded). Returns the original part if it's not a void or if no solid
        parent can be found (orphan void).
        """
        if part is None or not getattr(part, "is_void", False):
            return part
        parent_id = getattr(part, "parent_id", None)
        if parent_id is None:
            return part
        for candidate in getattr(self, "parts", []) or []:
            if getattr(candidate, "id", None) == parent_id and not getattr(candidate, "is_void", False):
                return candidate
        return part

    def assign_material_to_part(
        self,
        part,
        material_id=None,
        announce=True,
        assignment_mode=None,
        heterogeneity_method=None,
        heterogeneity_config=None,
        material_field_config=None,
        symmetry=None,
        behavior=None,
        damage=None,
    ):
        if part is None:
            QMessageBox.warning(self, "Invalid", "No part selected for material assignment.")
            return False

        state = self._require_project_state()
        try:
            part_id = int(getattr(part, "id"))
        except Exception:
            part_id = getattr(part, "id", None)
        target_part = next(
            (entry for entry in getattr(state, "parts", []) if getattr(entry, "id", None) == part_id),
            None,
        )
        if target_part is None:
            QMessageBox.warning(self, "Invalid", "Selected part was not found in the project state.")
            return False

        if target_part.is_void:
            redirected = self._resolve_material_target(target_part)
            if redirected is None or redirected is target_part or getattr(redirected, "is_void", False):
                QMessageBox.warning(
                    self,
                    "Invalid",
                    "Cannot assign material to a void without a solid parent part.",
                )
                return False
            QMessageBox.information(
                self,
                "Material Assignment",
                f"'{target_part.name}' is a hole/void. Material was applied to its parent "
                f"part '{redirected.name}' (the hole is automatically excluded from the "
                f"material region).",
            )
            target_part = redirected

        if not self.can_assign_material():
            QMessageBox.warning(self, "Locked", "Switch to Materials stage first.")
            return False

        if material_id is None:
            material_id = self.current_material_id
        if material_id is None:
            QMessageBox.warning(self, "Select Material", "Select an active material first.")
            return False

        try:
            material_id = int(material_id)
        except (TypeError, ValueError):
            pass
        material_store = getattr(state, "materials", {})
        if material_id not in material_store:
            # Last-resort sync: if panel has the material, pull it in
            panel = getattr(self.window(), "properties_panel", None)
            if panel and hasattr(panel, "materials_tab"):
                mat = panel.materials_tab.materials.get(material_id)
                if mat:
                    material_store[material_id] = mat
            if material_id not in material_store:
                print(f"[assign_material_to_part] missing material_id={material_id}, available={list(material_store.keys())}")
                QMessageBox.warning(self, "Invalid", "Selected material not found.")
                return False

        self.push_undo_state()
        material = material_store[material_id]
        target_part.material_id = material_id
        if assignment_mode is not None:
            target_part.material_assignment_mode = str(assignment_mode or "homogeneous")
        if heterogeneity_method is not None:
            target_part.heterogeneity_method = str(heterogeneity_method or "region_based")
        if heterogeneity_config is not None:
            target_part.heterogeneity_config = normalize_heterogeneity_config(
                copy.deepcopy(heterogeneity_config)
            )
        if material_field_config is not None:
            target_part.material_field_config = normalize_material_field_config(
                copy.deepcopy(material_field_config)
            )
        target_part.material_symmetry = normalize_material_symmetry(
            symmetry if symmetry is not None else getattr(material, "symmetry", "isotropic")
        )
        target_part.material_behavior = normalize_material_behavior(
            behavior if behavior is not None else getattr(material, "behavior", "elastic")
        )
        target_part.material_damage = normalize_material_damage(
            damage if damage is not None else getattr(material, "damage", "none")
        )
        target_part.material_type = getattr(material, "mat_type", None)
        target_part.material_props = normalize_material_properties(
            copy.deepcopy(getattr(material, "properties", {}) or {}),
            getattr(target_part, "material_behavior", getattr(material, "behavior", "elastic")),
            getattr(target_part, "material_symmetry", getattr(material, "symmetry", "isotropic")),
            getattr(target_part, "material_damage", getattr(material, "damage", "none")),
        )
        if target_part is not part:
            try:
                part.material_id = material_id
                if assignment_mode is not None:
                    part.material_assignment_mode = target_part.material_assignment_mode
                if heterogeneity_method is not None:
                    part.heterogeneity_method = target_part.heterogeneity_method
                if heterogeneity_config is not None:
                    part.heterogeneity_config = copy.deepcopy(target_part.heterogeneity_config)
                if material_field_config is not None:
                    part.material_field_config = copy.deepcopy(target_part.material_field_config)
                part.material_symmetry = target_part.material_symmetry
                part.material_behavior = target_part.material_behavior
                part.material_damage = target_part.material_damage
                part.material_type = target_part.material_type
                part.material_props = copy.deepcopy(target_part.material_props)
            except Exception:
                pass
        self.materialsChanged.emit()
        self.partsChanged.emit()
        self.set_selected_part(target_part.id, emit_signal=True)
        self._sync_material_panel_from_part(target_part)
        self.rebuild_display_geometry()
        self.redraw()
        mat_name = getattr(material, "name", str(material_id))
        if announce:
            QMessageBox.information(
                self,
                "Material Assigned",
                f"{mat_name} assigned to {target_part.name}.",
            )
        window = self.window()
        if window and hasattr(window, "statusBar"):
            window.statusBar().showMessage(
                f"Material assigned: {target_part.name} -> {mat_name}",
                4000,
            )
        return True

    def _pick_material_from_part(self, part):
        if part.material_id is None:
            QMessageBox.information(self, "No Material", "This part has no material assigned.")
            return
        panel = getattr(self.window(), "properties_panel", None)
        if panel and hasattr(panel, "materials_tab"):
            panel.materials_tab.set_active_material(part.material_id)
        else:
            self.set_current_material(part.material_id)

    def _open_material_editor(self):
        panel = getattr(self.window(), "properties_panel", None)
        if panel and hasattr(panel, "materials_tab"):
            panel.tabs.setCurrentWidget(panel.materials_tab)
            panel.materials_tab.focus_name_input()

    def _material_assignment_menu_spec(self):
        return [
            ("Homogeneous", "homogeneous"),
            ("Heterogeneous", "heterogeneous"),
            ("Material Field", "material_field"),
        ], [
            ("Region Based", "region_based"),
            ("Random Distribution", "random_distribution"),
        ], [
            ("Linear Gradient", "linear_gradient"),
            ("Radial Gradient", "radial_gradient"),
            ("Random Field", "random_field"),
            ("User Equation", "user_equation"),
        ], [
            *material_symmetry_options(),
        ], [
            *material_behavior_options(),
        ], [
            *material_damage_options(),
        ]

    def _populate_material_behavior_menu(
        self,
        menu,
        part,
        assignment_mode,
        method_value,
        symmetry,
        behavior,
        damage,
    ):
        mats = sorted(self.materials.values(), key=lambda m: m.serial)
        if not mats:
            no_action = menu.addAction("No materials defined")
            no_action.setEnabled(False)
            return
        for mat in mats:
            action = menu.addAction(f"{mat.name} ({mat.mat_type})")
            material_field_config = None
            heterogeneity_method = method_value
            if assignment_mode == "material_field":
                material_field_config = normalize_material_field_config(
                    {
                        "property_key": "E",
                        "field_type": str(method_value or "linear_gradient"),
                    }
                )
                heterogeneity_method = "region_based"
            action.triggered.connect(
                lambda _,
                p=part,
                serial=mat.serial,
                a=assignment_mode,
                hm=heterogeneity_method,
                mfc=copy.deepcopy(material_field_config),
                s=symmetry,
                b=behavior: self.assign_material_to_part(
                    p,
                    serial,
                    assignment_mode=a,
                    heterogeneity_method=hm,
                    material_field_config=mfc,
                    symmetry=s,
                    behavior=b,
                    damage=damage,
                )
            )

    def build_material_menu(self, part):
        if part.is_void:
            redirected = self._resolve_material_target(part)
            if redirected is None or redirected is part or getattr(redirected, "is_void", False):
                QMessageBox.warning(
                    self,
                    "Invalid",
                    "Cannot assign material to a void without a solid parent part.",
                )
                return None
            QMessageBox.information(
                self,
                "Material Assignment",
                f"'{part.name}' is a hole/void. The material menu will apply to its "
                f"parent part '{redirected.name}' (the hole is automatically excluded "
                f"from the material region).",
            )
            part = redirected
        if not self.can_assign_material():
            QMessageBox.warning(self, "Locked", "Switch to Materials stage first.")
            return None
        menu = QMenu(self)
        header = menu.addAction(f"Assign Material -> {part.name}")
        header.setEnabled(False)
        menu.addSeparator()
        assignment_modes, heterogeneity_methods, material_field_types, symmetries, behaviors, damages = (
            self._material_assignment_menu_spec()
        )
        for assignment_text, assignment_value in assignment_modes:
            assignment_menu = menu.addMenu(assignment_text)
            if assignment_value == "heterogeneous":
                method_items = heterogeneity_methods
            elif assignment_value == "material_field":
                method_items = material_field_types
            else:
                method_items = [("Default", "region_based")]
            for method_text, method_value in method_items:
                method_menu = assignment_menu if assignment_value == "homogeneous" else assignment_menu.addMenu(method_text)
                for symmetry_text, symmetry_value in symmetries:
                    symmetry_menu = method_menu.addMenu(symmetry_text)
                    for behavior_text, behavior_value in behaviors:
                        behavior_menu = symmetry_menu.addMenu(behavior_text)
                        for damage_text, damage_value in damages:
                            damage_menu = behavior_menu.addMenu(damage_text)
                            self._populate_material_behavior_menu(
                                damage_menu,
                                part,
                                assignment_value,
                                method_value,
                                symmetry_value,
                                behavior_value,
                                damage_value,
                            )
        menu.addSeparator()
        other_action = menu.addAction("Other...")
        other_action.triggered.connect(self._open_material_editor)
        return menu

    def _populate_material_submenu(self, menu, part):
        if menu is None or part is None or getattr(part, "is_void", False):
            return
        assignment_modes, heterogeneity_methods, material_field_types, symmetries, behaviors, damages = (
            self._material_assignment_menu_spec()
        )
        for assignment_text, assignment_value in assignment_modes:
            assignment_menu = menu.addMenu(assignment_text)
            if assignment_value == "heterogeneous":
                method_items = heterogeneity_methods
            elif assignment_value == "material_field":
                method_items = material_field_types
            else:
                method_items = [("Default", "region_based")]
            for method_text, method_value in method_items:
                method_menu = assignment_menu if assignment_value == "homogeneous" else assignment_menu.addMenu(method_text)
                for symmetry_text, symmetry_value in symmetries:
                    symmetry_menu = method_menu.addMenu(symmetry_text)
                    for behavior_text, behavior_value in behaviors:
                        behavior_menu = symmetry_menu.addMenu(behavior_text)
                        for damage_text, damage_value in damages:
                            damage_menu = behavior_menu.addMenu(damage_text)
                            self._populate_material_behavior_menu(
                                damage_menu,
                                part,
                                assignment_value,
                                method_value,
                                symmetry_value,
                                behavior_value,
                                damage_value,
                            )
        menu.addSeparator()
        other_action = menu.addAction("Other...")
        other_action.triggered.connect(self._open_material_editor)
    def _sync_material_panel_from_part(self, part):
        panel = getattr(self.window(), "properties_panel", None)
        if panel and hasattr(panel, "materials_tab"):
            panel.materials_tab._update_selected_part_display(getattr(part, "id", None))
            if part.material_id is not None:
                panel.materials_tab.select_material(part.material_id, update_editor=True)

    def disable_all_tools(self):
        """
        Disable all sketch / load tools.
        Used by stage-based UI control.
        """
        self.tool = "select"
        self.mode = "idle"
        self.current.clear()
        self._clear_preview()

    def enable_geometry_tools(self):
        self.set_module("Part")

    def enable_material_tools(self):
        self.set_module("Property")

    def can_assign_material(self):
        """
        Central guard: material assignment allowed only in MATERIALS stage
        """
        stage_key = str(getattr(getattr(self, "project_state", None), "current_stage", "") or "").lower()
        return (
            stage_key == "materials"
            or str(getattr(self, "active_module", "")).lower() == "property"
        )

    def enable_bc_tools(self):
        self.set_module("Load")

    def enable_mesh_tools(self):
        self.set_module("Mesh")

    def enable_job_tools(self):
        self.set_module("Job")

    def enable_interaction_tools(self):
        self.set_module("Interface")

    def advance_animation_frame(self):
        total_frames = self._animation_frame_total()
        if total_frames <= 0:
            if getattr(self, "animation_timer", None):
                try: self.animation_timer.stop()
                except Exception: pass
            self._emit_animation_playback_state()
            self.is_visualization_mode = False
            return

        if getattr(self, "_lazy_results_enabled", False):
            if getattr(self, "_animation_frame_loading", False):
                return
            if not hasattr(self, "current_frame_index"):
                self.current_frame_index = -1
            next_index = (self.current_frame_index + 1) % total_frames
            self._request_lazy_animation_frame(next_index)
            return

        if not hasattr(self, "current_frame_index"):
            self.current_frame_index = -1
        self.current_frame_index = (self.current_frame_index + 1) % len(self.animation_frames)
        current_positions = self._normalize_animation_positions(self.animation_frames[self.current_frame_index])
        if current_positions is None:
            return
        previous_positions = None
        if isinstance(getattr(self, "_current_animation_positions", None), np.ndarray):
            previous_positions = np.asarray(self._current_animation_positions, dtype=float)
            if previous_positions.shape != current_positions.shape:
                previous_positions = None
        self._current_animation_velocity = (
            current_positions - previous_positions
            if previous_positions is not None
            else None
        )
        self._ensure_replay_particle_metadata(len(current_positions))
        display_indices = self._display_indices_for_results(len(current_positions))
        display_positions = (
            current_positions[display_indices]
            if len(display_indices) > 0
            else current_positions
        )
        self._replay_visible_particle_indices = display_indices
        self._current_animation_positions = current_positions
        self._current_frame_packet = {
            "raw_positions": current_positions,
            "display_positions": display_positions,
            "display_indices": display_indices,
        }
        original_show_elements = self.show_anim_elements
        if self._replay_lod_active:
            self.show_anim_elements = False
        self._apply_scene_animation_frame(display_positions)
        self.show_anim_elements = original_show_elements
        self._update_results_debug_overlays()
        try:
            self._update_replay_scope_highlight(current_positions)
        except Exception:
            pass
        if self._replay_selected_particle_index is not None:
            self._emit_replay_particle_info(self._replay_selected_particle_index)
        self.animationFrameChanged.emit(int(self.current_frame_index))

    def _apply_scene_animation_frame(self, current_positions):
        current_positions = self._normalize_animation_positions(current_positions)
        if current_positions is None:
            return False

        # Ensure storage for animated items
        if not hasattr(self, "_anim_node_items"):
            self._anim_node_items = []
        if not hasattr(self, "_anim_element_items"):
            self._anim_element_items = []
        if not hasattr(self, "_smooth_contour_item"):
            self._smooth_contour_item = None

        def _is_alive(item):
            return self._item_scene_safe(item) is not None

        def _hide_item_at(items, idx):
            item = items[idx]
            if item is None:
                return
            if not _is_alive(item):
                items[idx] = None
                return
            try:
                item.setVisible(False)
            except RuntimeError:
                items[idx] = None
            except Exception:
                pass

        show_mesh = getattr(self, "show_anim_elements", True)
        show_nodes = getattr(self, "show_anim_nodes", True)
        current_animation_positions = self._normalize_animation_positions(
            getattr(self, "_current_animation_positions", None)
        )
        packet = self._validated_current_frame_packet() or {}
        if current_animation_positions is None:
            raw_positions = packet.get("raw_positions")
            if raw_positions is not None:
                current_animation_positions = np.array(raw_positions, dtype=float, copy=True)
                self._current_animation_positions = np.array(raw_positions, dtype=float, copy=True)
        particle_count = (
            int(current_animation_positions.shape[0])
            if current_animation_positions is not None
            else int(current_positions.shape[0])
        )
        metrics = self._results_render_metrics(
            particle_count=particle_count,
            display_count=int(current_positions.shape[0]),
        )
        node_radius = float(metrics["node_radius"])
        node_diameter = float(metrics["node_diameter"])
        display_scalar_values = packet.get("display_scalar_values")
        triangle_scalar_values = packet.get("triangle_scalar_values")
        has_triangle_field = triangle_scalar_values is not None
        # Pen for triangle edges — cosmetic so width is in device pixels and
        # never thickens on zoom.  When a scalar field is active the pen is
        # more transparent so the per-triangle colours read cleanly.
        if has_triangle_field:
            element_pen = QPen(
                QColor(18, 18, 18, int(255 * max(0.05, min(1.0, self.show_anim_element_alpha * 0.6)))),
                0.8,
            )
        else:
            element_pen = QPen(
                QColor(32, 32, 32, int(255 * max(0.05, min(1.0, self.show_anim_element_alpha)))),
                1.0,
            )
        element_pen.setCosmetic(True)   # zoom-invariant: never thickens on zoom-in
        node_pen = QPen(Qt.transparent)
        # Mesh connecting dots (the triangle vertices) are ALWAYS solid black,
        # regardless of the contour colour of the surrounding triangles
        # (red / orange / yellow / green / blue).  We deliberately ignore the
        # per-node scalar palette so the nodes never inherit the mesh colours.
        node_brush = QBrush(QColor(0, 0, 0))
        node_colors = None
        global_elements = getattr(self, "global_elements", [])
        try:
            triangle_count = len(global_elements)
        except Exception:
            triangle_count = 0

        # Size the Results vertex dots relative to the local mesh spacing so they
        # read as small, evenly-sized points — not the large grid blobs produced
        # by the fixed scene-unit floor in _results_render_metrics, which balloon
        # when zoomed in or on small-scale models.  We only ever SHRINK the
        # metric-derived radius (never enlarge it), so this cannot regress.
        # Guarded against downsampled frames where element indices would exceed
        # the (reduced) current_positions array.
        try:
            _ga = np.asarray(global_elements, dtype=int)
            _pa = np.asarray(current_positions, dtype=float)
            if _ga.size and _pa.shape[0] > 2 and int(_ga.max()) < _pa.shape[0]:
                _e = np.linalg.norm(_pa[_ga[:, 1]] - _pa[_ga[:, 0]], axis=1)
                _e = _e[np.isfinite(_e) & (_e > 0)]
                if _e.size:
                    _r = 0.13 * float(np.median(_e))   # dot radius ≈ 13% of edge
                    if _r > 0:
                        node_radius = max(min(node_radius, _r), 0.04)
                        node_diameter = node_radius * 2.0
        except Exception:
            pass

        triangle_colors = self._scalar_colors(
            triangle_scalar_values,
            expected_len=triangle_count,
            palette="jet",
            alpha=228,
        )

        # raw_positions holds the FULL (un-downsampled) node array.
        # global_elements indices reference this full array, so the smooth
        # contour renderer MUST use raw_positions, not current_positions
        # (which is the downsampled display_positions).  Using the wrong
        # array caused the earlier "spider-web" crash.
        raw_positions = packet.get("raw_positions")

        def _remove_smooth_contour():
            if self._smooth_contour_item is not None:
                try:
                    sc = self._item_scene_safe(self._smooth_contour_item)
                    if sc is not None:
                        sc.removeItem(self._smooth_contour_item)
                except Exception:
                    pass
                self._smooth_contour_item = None

        # Create or update element items
        if show_mesh:
            if has_triangle_field and raw_positions is not None:
                # ── SMOOTH CONTOUR MODE ─────────────────────────────────
                # 1. Render Gouraud-shaded gradient image (z = -1)
                _remove_smooth_contour()
                self._smooth_contour_item = self._render_smooth_contour(
                    raw_positions, global_elements, triangle_scalar_values,
                    palette="jet", alpha=228,
                )
                # 2. Thin cosmetic mesh lines on top (z = 0, transparent fill)
                contour_line_pen = QPen(QColor(0, 0, 0, 35), 0.8)
                contour_line_pen.setCosmetic(True)
                for ei, element_indices in enumerate(getattr(self, "global_elements", [])):
                    if any(idx < 0 or idx >= len(raw_positions) for idx in element_indices):
                        continue
                    p1 = raw_positions[element_indices[0]]
                    p2 = raw_positions[element_indices[1]]
                    p3 = raw_positions[element_indices[2]]
                    path = QPainterPath()
                    path.moveTo(p1[0], p1[1]); path.lineTo(p2[0], p2[1]); path.lineTo(p3[0], p3[1]); path.closeSubpath()
                    reused = False
                    if ei < len(self._anim_element_items):
                        item = self._anim_element_items[ei]
                        if item is not None and _is_alive(item):
                            try:
                                item.setPath(path)
                                item.setPen(contour_line_pen)
                                item.setBrush(QBrush(Qt.transparent))
                                item.setVisible(True)
                                reused = True
                            except RuntimeError:
                                self._anim_element_items[ei] = None
                    if not reused:
                        item = self.scene().addPath(path, contour_line_pen)
                        item.setBrush(QBrush(Qt.transparent))
                        if ei < len(self._anim_element_items):
                            self._anim_element_items[ei] = item
                        else:
                            self._anim_element_items.append(item)
                element_count = len(getattr(self, "global_elements", []))
                for extra in range(element_count, len(self._anim_element_items)):
                    _hide_item_at(self._anim_element_items, extra)
            else:
                # ── FLAT SHADING MODE ───────────────────────────────────
                _remove_smooth_contour()
                for ei, element_indices in enumerate(getattr(self, "global_elements", [])):
                    if any(idx < 0 or idx >= len(current_positions) for idx in element_indices):
                        continue
                    p1 = current_positions[element_indices[0]]
                    p2 = current_positions[element_indices[1]]
                    p3 = current_positions[element_indices[2]]
                    path = QPainterPath()
                    path.moveTo(p1[0], p1[1]); path.lineTo(p2[0], p2[1]); path.lineTo(p3[0], p3[1]); path.closeSubpath()
                    reused = False
                    if ei < len(self._anim_element_items):
                        item = self._anim_element_items[ei]
                        if item is not None and _is_alive(item):
                            try:
                                item.setPath(path)
                                item.setPen(element_pen)
                                if triangle_colors is not None and ei < len(triangle_colors):
                                    item.setBrush(QBrush(triangle_colors[ei]))
                                else:
                                    item.setBrush(QBrush(Qt.transparent))
                                item.setVisible(True)
                                reused = True
                            except RuntimeError:
                                self._anim_element_items[ei] = None
                    if not reused:
                        item = self.scene().addPath(path, element_pen)
                        if triangle_colors is not None and ei < len(triangle_colors):
                            item.setBrush(QBrush(triangle_colors[ei]))
                        else:
                            item.setBrush(QBrush(Qt.transparent))
                        if ei < len(self._anim_element_items):
                            self._anim_element_items[ei] = item
                        else:
                            self._anim_element_items.append(item)
                element_count = len(getattr(self, "global_elements", []))
                for extra in range(element_count, len(self._anim_element_items)):
                    _hide_item_at(self._anim_element_items, extra)
        else:
            _remove_smooth_contour()
            for idx in range(len(self._anim_element_items)):
                _hide_item_at(self._anim_element_items, idx)

        # Create or update node items
        # Reuse existing ellipse items if available
        if show_nodes:
            for ni, (x, y) in enumerate(current_positions):
                reused = False
                if ni < len(self._anim_node_items):
                    item = self._anim_node_items[ni]
                    if item is not None and _is_alive(item):
                        try:
                            item.setRect(x - node_radius, y - node_radius, node_diameter, node_diameter)
                            item.setPen(node_pen)
                            if node_colors is not None and ni < len(node_colors):
                                item.setBrush(QBrush(node_colors[ni]))
                            else:
                                item.setBrush(node_brush)
                            item.setVisible(True)
                            reused = True
                        except RuntimeError:
                            self._anim_node_items[ni] = None
                if not reused:
                    brush = QBrush(node_colors[ni]) if node_colors is not None and ni < len(node_colors) else node_brush
                    item = self.scene().addEllipse(
                        x - node_radius, y - node_radius, node_diameter, node_diameter, node_pen, brush
                    )
                    if ni < len(self._anim_node_items):
                        self._anim_node_items[ni] = item
                    else:
                        self._anim_node_items.append(item)
        else:
            for idx in range(len(self._anim_node_items)):
                _hide_item_at(self._anim_node_items, idx)

        # Optionally hide any extra items if frame has fewer nodes/elements than before
        if show_nodes:
            for extra in range(len(current_positions), len(self._anim_node_items)):
                _hide_item_at(self._anim_node_items, extra)

        # Final update
        self.scene().update()
        return True

    def _render_smooth_contour(self, positions, elements, tri_scalars,
                               palette="jet", alpha=228):
        """
        Gouraud-shaded smooth contour rendered as a QPixmap scene item.

        IMPORTANT: `positions` must be the FULL raw node array (not the
        downsampled display_positions), because global_elements indices
        reference the full array.  Passing display_positions here caused
        the earlier spider-web crash.

        Steps
        -----
        1. Average element-centred scalars → node-averaged scalars.
        2. Map each node scalar to RGBA via the requested colourmap.
        3. CPU-rasterise every triangle with vectorised barycentric
           interpolation → smooth gradient, no visible triangle boundaries.
        4. Upload as QImage, transform to align with scene coords, z = -1.
        """
        try:
            from PySide6.QtGui import QImage, QPixmap, QTransform as QTr

            pos      = np.asarray(positions,   dtype=float)
            elems    = np.asarray(elements,    dtype=int)
            tri_vals = np.asarray(tri_scalars, dtype=float).ravel()

            n_nodes = len(pos)
            n_tris  = len(elems)
            if n_nodes < 3 or n_tris < 1:
                return None

            # 1. Element → node-averaged scalars
            node_sum = np.zeros(n_nodes, dtype=float)
            node_cnt = np.zeros(n_nodes, dtype=float)
            for ci in range(3):
                idx = elems[:, ci]
                np.add.at(node_sum, idx, tri_vals)
                np.add.at(node_cnt, idx, 1.0)
            node_vals = np.where(node_cnt > 0,
                                 node_sum / np.maximum(node_cnt, 1.0), 0.0)

            # 2. Normalise and map to per-node RGBA
            finite = np.isfinite(node_vals)
            if not np.any(finite):
                return None
            vmin = float(np.min(node_vals[finite]))
            vmax = float(np.max(node_vals[finite]))
            rng  = vmax - vmin
            norm_v = (np.zeros(n_nodes, dtype=float)
                      if abs(rng) < 1e-12
                      else np.clip((node_vals - vmin) / rng, 0.0, 1.0))
            try:
                from matplotlib import cm as mpl_cm
                cmap = getattr(mpl_cm, str(palette), mpl_cm.jet)
            except Exception:
                return None
            rgba_nodes = (cmap(norm_v) * 255.0).astype(np.uint8)  # (N, 4)

            # 3. Scene bounds → image resolution
            xs, ys = pos[:, 0], pos[:, 1]
            x_min, x_max = float(xs.min()), float(xs.max())
            y_min, y_max = float(ys.min()), float(ys.max())
            w_sc = x_max - x_min
            h_sc = y_max - y_min
            if w_sc < 1e-10 or h_sc < 1e-10:
                return None

            max_px = 512 if n_tris <= 1000 else (384 if n_tris <= 4000 else 256)
            if w_sc >= h_sc:
                img_w = max_px
                img_h = max(1, int(round(max_px * h_sc / w_sc)))
            else:
                img_h = max_px
                img_w = max(1, int(round(max_px * w_sc / h_sc)))

            # 4. Node positions in pixel space (row 0 = scene y_max)
            px_f = (pos[:, 0] - x_min) * (img_w / w_sc)
            py_f = (y_max - pos[:, 1]) * (img_h / h_sc)

            # 5. Barycentric rasterisation
            img_buf  = np.zeros((img_h, img_w, 4), dtype=np.float32)
            occupied = np.zeros((img_h, img_w),    dtype=bool)

            i0, i1, i2 = elems[:, 0], elems[:, 1], elems[:, 2]
            x0, x1, x2 = px_f[i0], px_f[i1], px_f[i2]
            y0, y1, y2 = py_f[i0], py_f[i1], py_f[i2]
            c0 = rgba_nodes[i0].astype(np.float32)
            c1 = rgba_nodes[i1].astype(np.float32)
            c2 = rgba_nodes[i2].astype(np.float32)

            for ti in range(n_tris):
                bx0 = max(0,         int(min(x0[ti], x1[ti], x2[ti])))
                bx1 = min(img_w - 1, int(max(x0[ti], x1[ti], x2[ti])) + 1)
                by0 = max(0,         int(min(y0[ti], y1[ti], y2[ti])))
                by1 = min(img_h - 1, int(max(y0[ti], y1[ti], y2[ti])) + 1)
                if bx0 > bx1 or by0 > by1:
                    continue
                denom = ((y1[ti] - y2[ti]) * (x0[ti] - x2[ti]) +
                         (x2[ti] - x1[ti]) * (y0[ti] - y2[ti]))
                if abs(denom) < 0.5:
                    continue
                inv_d = 1.0 / denom
                gx = np.arange(bx0, bx1 + 1, dtype=np.float32)
                gy = np.arange(by0, by1 + 1, dtype=np.float32)
                GX, GY = np.meshgrid(gx, gy)
                w0 = ((y1[ti] - y2[ti]) * (GX - x2[ti]) +
                      (x2[ti] - x1[ti]) * (GY - y2[ti])) * inv_d
                w1 = ((y2[ti] - y0[ti]) * (GX - x2[ti]) +
                      (x0[ti] - x2[ti]) * (GY - y2[ti])) * inv_d
                w2   = 1.0 - w0 - w1
                mask = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
                if not mask.any():
                    continue
                colour = (w0[mask, np.newaxis] * c0[ti] +
                          w1[mask, np.newaxis] * c1[ti] +
                          w2[mask, np.newaxis] * c2[ti])
                ri  = GY[mask].astype(int)
                ci_ = GX[mask].astype(int)
                img_buf[ri, ci_]  = colour
                occupied[ri, ci_] = True

            # 6. Build QImage
            out = np.zeros((img_h, img_w, 4), dtype=np.uint8)
            out[occupied, :3] = np.clip(img_buf[occupied, :3], 0, 255).astype(np.uint8)
            out[occupied,  3] = int(max(0, min(255, alpha)))
            raw  = bytes(out.tobytes())
            qimg = QImage(raw, img_w, img_h,
                          img_w * 4, QImage.Format_RGBA8888).copy()
            pixmap = QPixmap.fromImage(qimg)

            # 7. Place in scene with correct affine transform.
            #    setPos at (x_min, y_max); scale maps image x→right, image y→up.
            #    Combined with the view's scale(1,-1) the image appears unflipped.
            item = self.scene().addPixmap(pixmap)
            t = QTr(w_sc / img_w, 0.0,
                    0.0,           -(h_sc / img_h),
                    0.0,           0.0)
            item.setTransform(t)
            item.setPos(x_min, y_max)
            item.setZValue(-1)
            return item

        except Exception:
            return None

    def _scalar_colors(self, values, expected_len=None, palette="viridis", alpha=170):
        if values is None:
            return None
        arr = np.asarray(values, dtype=float).reshape(-1)
        if expected_len is not None and arr.size != int(expected_len):
            return None
        finite = np.isfinite(arr)
        if not np.any(finite):
            return None
        vmin = float(np.min(arr[finite]))
        vmax = float(np.max(arr[finite]))
        if abs(vmax - vmin) < 1e-12:
            norm = np.zeros_like(arr, dtype=float)
        else:
            norm = (arr - vmin) / (vmax - vmin)
        norm[~finite] = 0.0
        try:
            from matplotlib import cm as mpl_cm
            cmap = getattr(mpl_cm, str(palette), None)
            if cmap is None:
                cmap = mpl_cm.viridis
            rgba = np.asarray(cmap(norm), dtype=float)
            colors = []
            for idx, c in enumerate(rgba):
                if not finite[idx]:
                    colors.append(QColor(0, 0, 0, 0))
                    continue
                colors.append(
                    QColor(
                        int(max(0.0, min(1.0, c[0])) * 255.0),
                        int(max(0.0, min(1.0, c[1])) * 255.0),
                        int(max(0.0, min(1.0, c[2])) * 255.0),
                        int(max(0, min(255, int(alpha)))),
                    )
                )
            return colors
        except Exception:
            return None

    def _build_velocity_map(self):
        vel_scale = self._unit_scale_to_meters()
        velocity_map = {}
        source = []
        if self.global_nodes is not None and len(self.global_nodes) > 0:
            # Map geometry to nodes when mesh exists
            tol = max(1.0, DEFAULT_DX * 0.6)
            mapped_bcs = map_geometry_to_nodes(self.global_nodes, self.bcs, tol)
            source.extend(mapped_bcs)
            # Include any direct node id mappings (3D selection, etc.)
            for bc in self.bcs:
                if bc.get("type") not in ("velocity_x", "velocity_y", "velocity_z", "fix_x", "fix_y"):
                    continue
                for nid in bc.get("ids", []):
                    source.append({"node_id": nid, "bc": bc})
                part_id = bc.get("part_id")
                if part_id is not None:
                    for nid in self._part_node_ids_from_mesh(part_id):
                        source.append({"node_id": nid, "bc": bc})
        for m in source:
            nid = int(m["node_id"])
            btype = m["bc"]["type"]
            if btype not in ("velocity_x", "fix_x"):
                continue
            if nid not in velocity_map:
                velocity_map[nid] = [0.0, 0.0]
            val = float(m["bc"].get("val", 0.0)) * vel_scale
            if btype in ("velocity_x", "fix_x"):
                velocity_map[nid][0] = val
            else:
                velocity_map[nid][1] = val
        return velocity_map

    def _part_node_ids_from_mesh(self, part_id, nodes=None, elements=None, element_part_map=None):
        try:
            pid = int(part_id)
        except Exception:
            return []
        if nodes is None:
            nodes = self.global_nodes
        if elements is None:
            elements = self.global_elements
        if element_part_map is None:
            element_part_map = self.element_part_map
        if nodes is None or elements is None:
            return []
        try:
            elem_arr = np.asarray(elements, dtype=int)
        except Exception:
            return []
        if elem_arr.ndim != 2 or len(elem_arr) == 0:
            return []
        node_count = len(nodes)
        part_map = {}
        for item in element_part_map or []:
            if not isinstance(item, dict):
                continue
            try:
                eidx = int(item.get("element_idx"))
                pidx = int(item.get("part_id"))
            except Exception:
                continue
            part_map[eidx] = pidx
        if not part_map:
            return []
        node_ids = set()
        for eidx, conn in enumerate(elem_arr):
            if part_map.get(eidx) != pid:
                continue
            for nid in conn:
                try:
                    node_id = int(nid)
                except Exception:
                    continue
                if 0 <= node_id < node_count:
                    node_ids.add(node_id)
        return sorted(node_ids)

    def _compact_mesh_nodes_and_elements(self, nodes, elements, element_part_map=None):
        try:
            node_arr = np.asarray(nodes, dtype=float)
        except Exception as exc:
            raise ValueError(f"Invalid mesh particle array: {exc}") from exc
        try:
            elem_arr = np.asarray(elements, dtype=int)
        except Exception as exc:
            raise ValueError(f"Invalid mesh triangle array: {exc}") from exc

        if node_arr.ndim != 2 or node_arr.shape[1] < 2:
            raise ValueError("Mesh particles must be a Nx2 array.")
        if elem_arr.size == 0:
            raise ValueError("Connection generation did not generate any connections within the part boundaries.")
        if elem_arr.ndim != 2 or elem_arr.shape[1] != 3:
            raise ValueError("Mesh connections must be a Nx3 triangle array.")

        node_count = len(node_arr)
        if np.any(elem_arr < 0) or np.any(elem_arr >= node_count):
            raise ValueError("Mesh triangles reference particle indices outside the particle array.")

        used_indices = np.unique(elem_arr.reshape(-1))
        if len(used_indices) == 0:
            raise ValueError("Mesh compaction found no particles referenced by triangles.")

        remap = np.full(node_count, -1, dtype=int)
        remap[used_indices] = np.arange(len(used_indices), dtype=int)
        compact_nodes = node_arr[used_indices]
        compact_elements = remap[elem_arr]

        unique_used = np.unique(compact_elements.reshape(-1))
        assert len(unique_used) == len(compact_nodes)
        if len(unique_used) != len(compact_nodes):
            raise ValueError("Mesh validation failed: some particles are not referenced by any triangle.")

        compact_part_map = []
        for item in element_part_map or []:
            if not isinstance(item, dict):
                continue
            compact_part_map.append(dict(item))

        return compact_nodes, compact_elements, compact_part_map

    def _ensure_force_id(self, load):
        try:
            force_id = int(load.get("force_id"))
            if force_id > 0:
                return force_id
        except Exception:
            pass
        force_id = int(getattr(self, "_next_force_id", 1))
        load["force_id"] = force_id
        self._next_force_id = force_id + 1
        return force_id

    def _ensure_velocity_id(self, bc):
        try:
            vel_id = int(bc.get("vel_id"))
            if vel_id > 0:
                return vel_id
        except Exception:
            pass
        vel_id = int(getattr(self, "_next_velocity_id", 1))
        bc["vel_id"] = vel_id
        self._next_velocity_id = vel_id + 1
        return vel_id

    def _get_sim_total_time(self):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CPD-main", "config.yml")
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
        except Exception:
            return 0.0
        sim = config.get("simulation", {})
        try:
            time_step = float(sim.get("time_step", 0.0))
        except Exception:
            time_step = 0.0
        try:
            total_steps = float(sim.get("total_steps", 0.0))
        except Exception:
            total_steps = 0.0
        return max(0.0, time_step * total_steps)

    def _get_sim_time_settings(self):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CPD-main", "config.yml")
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
        except Exception:
            return 0.0, 0
        sim = config.get("simulation", {})
        try:
            time_step = float(sim.get("time_step", 0.0))
        except Exception:
            time_step = 0.0
        try:
            total_steps = int(sim.get("total_steps", 0))
        except Exception:
            total_steps = 0
        return max(0.0, time_step), max(0, total_steps)

    def _safe_eval_expr(self, expr, t_value=0.0):
        if expr is None:
            return 0.0
        text = str(expr).strip()
        if not text:
            return 0.0
        text = text.replace("^", "**")
        text = re.sub(r"(?<=\d)(?=t)", "*", text)
        text = re.sub(r"(?<=\))(?=t)", "*", text)
        safe = {
            "t": float(t_value),
            "pi": math.pi,
            "e": math.e,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "sinh": math.sinh,
            "cosh": math.cosh,
            "tanh": math.tanh,
            "exp": math.exp,
            "log": math.log,
            "sqrt": math.sqrt,
            "abs": abs,
            "min": min,
            "max": max,
        }
        try:
            return float(eval(text, {"__builtins__": {}}, safe))
        except Exception:
            return 0.0

    def _eval_profile_at_time(self, segments, t_value, keys):
        if not segments:
            return [0.0] * len(keys)
        seg = None
        for i, s in enumerate(segments):
            t0 = float(s.get("t0", 0.0))
            t1 = float(s.get("t1", t0))
            if t_value < t0:
                continue
            if t_value <= t1 or i == len(segments) - 1:
                seg = s
                break
        if seg is None:
            return [0.0] * len(keys)
        vals = []
        for key in keys:
            expr = seg.get(key, "0")
            vals.append(self._safe_eval_expr(expr, t_value))
        return vals

    def _workspace_dir(self):
        return str(get_workspace_dir())

    def _workspace_path(self, *parts):
        return str(get_workspace_path(*parts))

    def _workspace_input_path(self, *parts):
        return self._workspace_path("input", *parts)

    def _workspace_output_path(self, *parts):
        return self._workspace_path("output", *parts)

    def _write_time_series_csv(self, path, profiles, prefix, axes, value_scale=1.0):
        from solver_exporter import write_time_series_csv
        return write_time_series_csv(self, path, profiles, prefix, axes, value_scale=value_scale)

    def _normalize_force_profile(self, load, total_time):
        profile = load.get("time_profile") or []
        mode_is_percent = str(load.get("time_profile_mode", "absolute")).lower().startswith("percent")
        if not profile:
            fx = load.get("fx", 0.0)
            fy = load.get("fy", 0.0)
            fz = load.get("fz", 0.0)
            return [{"t0": 0.0, "t1": total_time, "fx": str(fx), "fy": str(fy), "fz": str(fz)}]
        normalized = []
        for seg in profile:
            t0 = float(seg.get("t0", 0.0))
            t1 = float(seg.get("t1", total_time))
            if mode_is_percent and total_time > 0:
                t0 = t0 * total_time / 100.0
                t1 = t1 * total_time / 100.0
            fx = seg.get("fx", seg.get("expr_fx", seg.get("expr", "0")))
            fy = seg.get("fy", seg.get("expr_fy", "0"))
            fz = seg.get("fz", seg.get("expr_fz", "0"))
            normalized.append({"t0": t0, "t1": t1, "fx": str(fx), "fy": str(fy), "fz": str(fz)})
        return normalized

    def _normalize_velocity_profile(self, bc, total_time):
        profile = bc.get("time_profile") or []
        btype = bc.get("type")
        mode_is_percent = str(bc.get("time_profile_mode", "absolute")).lower().startswith("percent")
        if not profile:
            val = bc.get("val", 0.0)
            if btype in ("velocity_x", "fix_x"):
                return [{"t0": 0.0, "t1": total_time, "vx": str(val), "vy": "0"}]
            if btype in ("velocity_y", "fix_y"):
                return [{"t0": 0.0, "t1": total_time, "vx": "0", "vy": str(val)}]
            return [{"t0": 0.0, "t1": total_time, "vx": "0", "vy": "0", "vz": str(val)}]
        normalized = []
        for seg in profile:
            t0 = float(seg.get("t0", 0.0))
            t1 = float(seg.get("t1", total_time))
            if mode_is_percent and total_time > 0:
                t0 = t0 * total_time / 100.0
                t1 = t1 * total_time / 100.0
            expr = seg.get("expr", seg.get("v"))
            vx = seg.get("vx")
            vy = seg.get("vy")
            vz = seg.get("vz")
            if expr is not None:
                if btype in ("velocity_x", "fix_x"):
                    vx = expr
                    vy = "0"
                    vz = "0"
                else:
                    if btype in ("velocity_y", "fix_y"):
                        vy = expr
                        vx = "0"
                        vz = "0"
                    else:
                        vz = expr
                        vx = "0"
                        vy = "0"
            if vx is None:
                vx = "0"
            if vy is None:
                vy = "0"
            if vz is None:
                vz = "0"
            normalized.append({"t0": t0, "t1": t1, "vx": str(vx), "vy": str(vy), "vz": str(vz)})
        return normalized

    def _write_time_profiles_config(self, force_profiles, velocity_profiles):
        from solver_exporter import write_time_profiles_config
        return write_time_profiles_config(self, force_profiles, velocity_profiles)

    def save_velocity_csv(self, path=None, write_header=True):
        from solver_exporter import save_velocity_csv
        return save_velocity_csv(self, path=path, write_header=write_header)

    def set_current_material(self, material_id):
        """Set the active material for next shape creation"""
        self.current_material_id = material_id if material_id != -1 else None
            
    def add_operation(self, shape_data, op_type="ADD", material_id=None):
        """Add a completed shape to the operation stack"""
        self.push_undo_state()
        operation = Operation(shape_data, op_type, material_id)
        self.operations.append(operation)
        self.update_geometry_from_operations()
        self.redraw()
        
    def update_geometry_from_operations(self):
        """Rebuild solid_geometry from operation stack in order"""
        self.solid_geometry = None
        for op in self.operations:
            geom = self._shape_to_geometry(op.shape_data)
            if geom is None or geom.is_empty:
                continue
            if self.solid_geometry is None:
                self.solid_geometry = geom
            else:
                if op.op_type == "ADD":
                    self.solid_geometry = self.solid_geometry.union(geom).buffer(0)
                elif op.op_type == "SUBTRACT":
                    self.solid_geometry = self.solid_geometry.difference(geom).buffer(0)
    
    def _shape_to_geometry(self, shape_data):
        """Convert shape_data dict to Shapely geometry"""
        if 'verts' not in shape_data:
            return None
        verts = shape_data['verts']
        if len(verts) < 3:
            return None
        try:
            return Polygon(verts)
        except:
            return None

    def _calculate_arc_points(self, p1, p2, p3):
        """Calculates points for an arc passing through p1, p2, and p3."""
        x1, y1 = p1; x2, y2 = p2; x3, y3 = p3

        D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if abs(D) < 1e-9: return [p1, p2, p3] # Collinear, return a line

        ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / D
        uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / D
        
        center = (ux, uy)
        radius = dist(p1, center)

        a1 = math.atan2(y1 - uy, x1 - ux)
        a2 = math.atan2(y2 - uy, x2 - ux)
        a3 = math.atan2(y3 - uy, x3 - ux)

        # Normalize angles to [0, 2*pi] to simplify logic
        a1 = (a1 + 2 * math.pi) % (2 * math.pi)
        a2 = (a2 + 2 * math.pi) % (2 * math.pi)
        a3 = (a3 + 2 * math.pi) % (2 * math.pi)

        # Determine sweep direction
        if a1 < a3:
            is_ccw = a1 < a2 < a3
        else: # Wraps around 2*pi
            is_ccw = not (a3 < a2 < a1)

        if not is_ccw:
            a1, a3 = a3, a1
        
        # Ensure sweep is always positive
        if a3 <= a1:
            a3 += 2 * math.pi

        num_segments = int(abs(a3 - a1) * radius / 10) # Approx. 10 pixels per segment
        num_segments = max(2, min(num_segments, 128))
        
        points = []
        for i in range(num_segments + 1):
            angle = a1 + (i / num_segments) * (a3 - a1)
            x = ux + radius * math.cos(angle)
            y = uy + radius * math.sin(angle)
            points.append((x, y))
        
        return points

    def _build_slot_vertices(self, p1, p2, width, segments=24):
        if width <= 0:
            return []
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            return []
        r = width / 2.0
        theta = math.atan2(dy, dx)
        angles2 = np.linspace(theta + math.pi / 2, theta - math.pi / 2, segments)
        arc2 = [(p2[0] + r * math.cos(a), p2[1] + r * math.sin(a)) for a in angles2]
        angles1 = np.linspace(theta - math.pi / 2, theta + math.pi / 2, segments)
        arc1 = [(p1[0] + r * math.cos(a), p1[1] + r * math.sin(a)) for a in angles1]
        p1a = arc1[-1]
        p2a = arc2[0]
        p1b = arc1[0]
        points = [p1a, p2a]
        if len(arc2) > 1:
            points.extend(arc2[1:])
        points.append(p1b)
        if len(arc1) > 1:
            points.extend(arc1[1:])
        points.append(points[0])
        return points

    def _arc_params_from_points(self, p_start, p_end, p_mid):
        x1, y1 = p_start
        x2, y2 = p_mid
        x3, y3 = p_end
        D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if abs(D) < 1e-9:
            return None, None, None, None
        ux = (
            (x1**2 + y1**2) * (y2 - y3)
            + (x2**2 + y2**2) * (y3 - y1)
            + (x3**2 + y3**2) * (y1 - y2)
        ) / D
        uy = (
            (x1**2 + y1**2) * (x3 - x2)
            + (x2**2 + y2**2) * (x1 - x3)
            + (x3**2 + y3**2) * (x2 - x1)
        ) / D
        center = (ux, uy)
        radius = dist(p_start, center)
        a_start = math.atan2(y1 - uy, x1 - ux) % (2 * math.pi)
        a_end = math.atan2(p_end[1] - uy, p_end[0] - ux) % (2 * math.pi)
        a_mid = math.atan2(p_mid[1] - uy, p_mid[0] - ux) % (2 * math.pi)

        if a_start < a_end:
            ccw = a_start < a_mid < a_end
        else:
            ccw = not (a_end < a_mid < a_start)

        if not ccw:
            a_start, a_end = a_end, a_start
        if a_end <= a_start:
            a_end += 2 * math.pi
        return center, radius, a_start, a_end
    
    def get_operation_list(self):
        """Return formatted list of operations for display"""
        op_list = []
        for op in self.operations:
            mat_name = "No Material"
            if op.material_id and op.material_id in self.materials:
                mat_name = self.materials[op.material_id].name
            op_list.append({
                'id': op.id,
                'op_type': op.op_type,
                'material': mat_name,
                'material_id': op.material_id
            })
        return op_list

    # --- Geometry Logic ---
    def _sketches_to_shapely(self):
        return self._sketches_to_shapely_list(self.sketches)

    def _sketches_to_shapely_list(self, sketches):
        segments = []
        for s in sketches:
            if len(s) < 2: continue
            for i in range(len(s)-1):
                a, b = s[i], s[i+1]
                if dist(a,b) > 1e-9:
                    segments.append(LineString([a, b]))
        if not segments: return None
        merged = unary_union(segments)
        try:
            polys, _, _, _ = polygonize_full(merged)
            polys = list(polys) if polys is not None else []
        except Exception:
            polys = list(polygonize(merged))
        if not polys: return None
        result = unary_union(polys)
        try: result = result.buffer(0)
        except: pass
        return result

    def confirm_solid(self):
        stored_sketches = copy.deepcopy(self.sketches or [])
        stored_sketch_meta = copy.deepcopy(self.sketch_meta or [])
        stored_dimensions = copy.deepcopy(self.dimensions or [])

        new_solid = self._sketches_to_shapely_list(stored_sketches)
        if new_solid is None or new_solid.is_empty:
            QMessageBox.information(self, "Info", "No closed regions found.")
            return

        if getattr(self, "_editing_part_shape_id", None) not in (None, "") and self.get_part_shape_edit_target() is None:
            self._clear_part_shape_edit_session()
        edit_target = self.get_part_shape_edit_target()
        is_editing_existing_part = edit_target is not None

        # Ask user for part name (or confirm rename while editing)
        if is_editing_existing_part:
            default_name = str(getattr(edit_target, "name", "") or f"Part {len(self.parts) + 1}")
            title = "Edit Part Shape"
            prompt = f"Update selected part shape.\nPart name:"
        else:
            default_name = self._porous_sketch_name or f"Part {len(self.parts) + 1}"
            title = "Part Name"
            prompt = f"Enter name for this part (default: {default_name}):"
        part_name, ok = QInputDialog.getText(
            None, title,
            prompt,
            text=default_name
        )

        if not ok or not part_name.strip():
            return

        part_name = part_name.strip()
        self.push_undo_state()

        if is_editing_existing_part:
            target_part = edit_target
            target_part.name = part_name
            target_part.geometry = new_solid
            target_part.sketches = copy.deepcopy(stored_sketches)
            target_part.sketch_meta = copy.deepcopy(stored_sketch_meta)
            target_part.dimensions = copy.deepcopy(stored_dimensions)
            target_part.constraints = copy.deepcopy(self.constraints)
            target_part.storage_units = "ui"
            target_part.is_direct_edit = False
            if self._pending_generated_feature_settings:
                feature_kind = "porous_holes" if getattr(target_part, "is_void", False) else "porous_particles"
                if feature_kind == "porous_particles":
                    geoms = [new_solid] if isinstance(new_solid, Polygon) else list(getattr(new_solid, "geoms", [new_solid]))
                    target_part.part_type = "particle_set"
                    target_part.particles = self._particle_records_from_polygons(geoms)
                else:
                    target_part.part_type = "void" if getattr(target_part, "is_void", False) else "solid"
                    target_part.particles = []
                self._set_generated_feature_metadata(
                    target_part,
                    feature_kind=feature_kind,
                    settings=self._pending_generated_feature_settings,
                )
            self._sync_cad_shape(target_part)

            # Best-effort parent (nesting) refresh for edited solid: choose smallest covering solid.
            if not getattr(target_part, "is_void", False):
                old_parent = getattr(target_part, "parent_id", None)
                parent_candidates = []
                for existing_part in self.parts:
                    if existing_part is target_part:
                        continue
                    if getattr(existing_part, "is_void", False):
                        continue
                    geom = getattr(existing_part, "geometry", None)
                    if geom is None or getattr(geom, "is_empty", True):
                        continue
                    try:
                        if geom.covers(new_solid):
                            area = float(getattr(geom, "area", 0.0) or 0.0)
                            parent_candidates.append((area, int(existing_part.id), existing_part))
                    except Exception:
                        continue
                if parent_candidates:
                    parent_candidates.sort(key=lambda item: (item[0], item[1]))
                    target_part.parent_id = parent_candidates[0][1]
                else:
                    target_part.parent_id = None
                if old_parent != getattr(target_part, "parent_id", None):
                    parent = self.get_parent_part(target_part)
                    if parent is not None:
                        print(
                            f"Part '{target_part.name}' (id={target_part.id}) is nested inside "
                            f"'{parent.name}' (id={parent.id})"
                        )
        else:
            # Create new Part with ORIGINAL geometry (NOT combined or modified)
            new_part = Part(part_name, geometry=new_solid)
            new_part.sketches = copy.deepcopy(stored_sketches)
            new_part.sketch_meta = copy.deepcopy(stored_sketch_meta)
            new_part.dimensions = copy.deepcopy(stored_dimensions)
            new_part.constraints = copy.deepcopy(self.constraints)
            new_part.storage_units = "ui"
            new_part.is_direct_edit = False
            if self._pending_generated_feature_settings:
                geoms = [new_solid] if isinstance(new_solid, Polygon) else list(getattr(new_solid, "geoms", [new_solid]))
                new_part.part_type = "particle_set"
                new_part.particles = self._particle_records_from_polygons(geoms)
                self._set_generated_feature_metadata(
                    new_part,
                    feature_kind="porous_particles",
                    settings=self._pending_generated_feature_settings,
                )
            self._sync_cad_shape(new_part)

            # Check if this part is inside an existing part (for nesting detection)
            for existing_part in self.parts:
                if not existing_part.is_void and existing_part.geometry and existing_part.geometry.covers(new_solid):
                    # Mark this as nested inside existing_part
                    new_part.parent_id = existing_part.id
                    print(f"Part '{part_name}' (id={new_part.id}) is nested inside '{existing_part.name}' (id={existing_part.id})")
                    break

            # Add to parts list WITHOUT modifying geometry
            self.parts.append(new_part)

        # Invalidate mesh/interface preview caches because geometry changed.
        self._mesh_cache = {"key": None, "parts_signature": None, "result": None}
        self._mesh_qa_cache = None
        self._mesh_qa_cache_sig = None

        # Update display geometry for visualization ONLY
        self.rebuild_display_geometry()

        self.sketches.clear()
        self.sketch_meta.clear()
        self.dimensions.clear()
        self.constraints.clear()
        self._porous_sketch_name = None
        self._pending_generated_feature_settings = None
        self.exit_sketch_edit_mode()
        self._clear_part_shape_edit_session()
        self._clear_preview()
        self.redraw()

        # Notify UI only after the transient sketch buffer has been cleared.
        self.partsChanged.emit()
        self._emit_interfaces_changed()
        self.geometryChanged.emit()
        QTimer.singleShot(50, self.fit_view)

        if is_editing_existing_part:
            window = self.window()
            if window and hasattr(window, "statusBar"):
                window.statusBar().showMessage(f"Updated part shape: {part_name}", 6000)
    
    def rebuild_display_geometry(self):
        """
        Rebuilds a single 'solid_geometry' object for non-visual tasks like
        meshing and hover detection. This object is a union of all solid parts,
        respecting the composite structure where nested parts displace parent material.
        Material assignment is NOT required for a part to be included.
        """
        if not self.parts:
            self.solid_geometry = None
            return

        # This will become the final unioned geometry for meshing/hover
        final_union = None
        
        # Process each part to find its visible, materialized geometry
        sorted_parts = sorted(self.parts, key=lambda p: p.id)
        for part in sorted_parts:
            # We only include solid parts in the final geometry. Voids are handled as children.
            if part.is_void:
                continue

            # Start with the part's original geometry
            geom_to_draw = part.geometry
            if geom_to_draw is None or geom_to_draw.is_empty:
                continue

            # Subtract all direct children from this part's geometry
            # This ensures children "replace" the parent's material
            children = self.get_child_parts(part)
            for child in children:
                if child.geometry and not child.geometry.is_empty:
                    try:
                        geom_to_draw = geom_to_draw.difference(child.geometry).buffer(0)
                    except Exception:
                        # Ignore occasional topology errors for robustness
                        pass

            # If there's a valid geometry left, union it with the final result
            if geom_to_draw and not geom_to_draw.is_empty:
                if final_union is None:
                    final_union = geom_to_draw
                else:
                    # Use a try-except block for robustness in union operations
                    try:
                        final_union = final_union.union(geom_to_draw).buffer(0)
                    except Exception:
                        pass

        self.solid_geometry = final_union

    def _update_part_geometry_from_sketches(self, part):
        if not part or not getattr(part, "sketches", None):
            return
        if getattr(part, "is_direct_edit", False):
            return
        geom = self._sketches_to_shapely_list(part.sketches)
        if geom is None or geom.is_empty:
            return
        part.geometry = geom
        part.storage_units = "ui"

    
    def get_parent_part(self, part):
        """Get the parent part if this part is nested"""
        if part.parent_id is None:
            return None
        return next((p for p in self.parts if p.id == part.parent_id), None)
    
    def get_child_parts(self, part):
        """Get all parts nested directly inside this part"""
        return [p for p in self.parts if p.parent_id == part.id]

    def cut_hole(self, hole_name_base=None, skip_undo=False, merge_voids=None):
        """
        Create a hole/cavity in the current solid.
        This operation cuts through ALL intersecting parts by creating
        a new nested 'void' part for each affected solid.
        """
        if not self.parts:
            QMessageBox.warning(self, "Error", "Create a main part first.")
            return False
        
        sketches_copy = copy.deepcopy(self.sketches or [])
        meta_copy = copy.deepcopy(self.sketch_meta or [])
        dims_copy = copy.deepcopy(self.dimensions or [])
        cons_copy = copy.deepcopy(self.constraints)

        holes_geom = self._sketches_to_shapely_list(sketches_copy)
        if holes_geom is None or holes_geom.is_empty:
            QMessageBox.warning(self, "Error", "No valid sketches to cut.")
            return False
        
        if hole_name_base is None:
            hole_name_base = self._porous_sketch_name
        if merge_voids is None:
            merge_voids = bool(self._porous_sketch_name)
        if not hole_name_base:
            hole_name_base, ok = QInputDialog.getText(
                None, "Cut/Hole Name", 
                f"Enter base name for this cut operation (e.g., 'Main Cut'):",
                text=f"Cut {len([p for p in self.parts if p.is_void]) + 1}"
            )
            if not ok or not hole_name_base.strip():
                return False
        hole_name_base = hole_name_base.strip()

        if not skip_undo:
            self.push_undo_state()
        
        # Find all solid parts that the new hole sketch intersects
        affected_parts = [
            p for p in self.parts 
            if not p.is_void and p.geometry and p.geometry.intersects(holes_geom)
        ]
        
        if not affected_parts:
            QMessageBox.warning(self, "Warning", "The hole does not intersect with any existing solid parts.")
            return False

        parts_added_count = 0
        for existing_part in affected_parts:
            # The geometry of this specific void is the intersection of the
            # hole sketch and the part it's cutting. This ensures the void
            # is confined to the boundary of the parent part.
            try:
                void_geom = existing_part.geometry.intersection(holes_geom)
            except Exception:
                void_geom = None

            if void_geom is None or void_geom.is_empty:
                continue

            if merge_voids:
                hole_part_name = (
                    hole_name_base
                    if len(affected_parts) == 1
                    else f"{hole_name_base} (in {existing_part.name})"
                )
                hole_part = Part(hole_part_name, geometry=void_geom, is_void=True)
                hole_part.parent_id = existing_part.id
                hole_part.sketches = copy.deepcopy(sketches_copy)
                hole_part.sketch_meta = copy.deepcopy(meta_copy)
                hole_part.dimensions = copy.deepcopy(dims_copy)
                hole_part.constraints = copy.deepcopy(cons_copy)
                hole_part.storage_units = "ui"
                self._sync_cad_shape(hole_part)
                self.parts.append(hole_part)
                parts_added_count += 1
                print(f"Created void part '{hole_part_name}' inside '{existing_part.name}'")
                continue

            geoms = [void_geom] if isinstance(void_geom, Polygon) else list(void_geom.geoms)
            for geom in geoms:
                if geom is None or geom.is_empty:
                    continue
                hole_part_name = f"{hole_name_base} (in {existing_part.name})"
                hole_part = Part(hole_part_name, geometry=geom, is_void=True)
                hole_part.parent_id = existing_part.id
                hole_part.sketches = copy.deepcopy(sketches_copy)
                hole_part.sketch_meta = copy.deepcopy(meta_copy)
                hole_part.dimensions = copy.deepcopy(dims_copy)
                hole_part.constraints = copy.deepcopy(cons_copy)
                hole_part.storage_units = "ui"
                self._sync_cad_shape(hole_part)
                
                self.parts.append(hole_part)
                parts_added_count += 1
                print(f"Created void part '{hole_part_name}' inside '{existing_part.name}'")

        if parts_added_count > 0:
            self.rebuild_display_geometry()
            self.partsChanged.emit()
            QMessageBox.information(self, "Success", f"Cut operation created {parts_added_count} void(s) in intersecting parts.")
        else:
            QMessageBox.warning(self, "Warning", "Cut operation did not result in any geometric changes.")

        self.sketches.clear()
        self.sketch_meta.clear()
        self.dimensions.clear()
        self.constraints.clear()
        self._porous_sketch_name = None
        self.restore_default_sketch_tool()
        self.redraw()
        return parts_added_count > 0

    def apply_sketch_feature_to_selected_part(self, op_type):
        if self.selected_part_id is None:
            QMessageBox.warning(self, "Feature", "Select a target part first.")
            return
        part = next((p for p in self.parts if p.id == self.selected_part_id), None)
        if part is None or part.geometry is None or part.geometry.is_empty:
            QMessageBox.warning(self, "Feature", "Selected part has no valid geometry.")
            return
        feature_sketches = copy.deepcopy(self.sketches or [])
        feature_geom = self._sketches_to_shapely_list(feature_sketches)
        if feature_geom is None or feature_geom.is_empty:
            QMessageBox.warning(self, "Feature", "No valid sketch to apply.")
            return
        op_type = str(op_type).lower()
        op_label = {"add": "Add", "cut": "Cut", "intersect": "Intersect"}.get(op_type, "Feature")

        self.push_undo_state()
        try:
            if op_type == "add":
                new_geom = part.geometry.union(feature_geom)
            elif op_type == "cut":
                new_geom = part.geometry.difference(feature_geom)
            elif op_type == "intersect":
                new_geom = part.geometry.intersection(feature_geom)
            else:
                QMessageBox.warning(self, "Feature", "Unknown feature operation.")
                return
            try:
                new_geom = new_geom.buffer(0)
            except Exception:
                pass
        except Exception as exc:
            QMessageBox.warning(self, "Feature", f"{op_label} failed: {exc}")
            return

        if new_geom is None or new_geom.is_empty:
            QMessageBox.warning(self, "Feature", f"{op_label} produced empty geometry.")
            return

        part.geometry = new_geom
        part.storage_units = "ui"
        part.is_direct_edit = True
        part.sketches = []
        part.sketch_meta = []
        part.dimensions = []
        part.constraints = []
        self._sync_cad_shape(part)

        self.sketches.clear()
        self.sketch_meta.clear()
        self.dimensions.clear()
        self.constraints.clear()
        self.restore_default_sketch_tool()
        self.rebuild_display_geometry()
        self.partsChanged.emit()
        self.geometryChanged.emit()
        self.redraw()
        window = self.window()
        if window and hasattr(window, "statusBar"):
            window.statusBar().showMessage(
                f"{op_label} feature applied to '{part.name}' (direct edit).",
                6000,
            )
    
    def serialize_geometry(self):
        return {
            "parts": [
                {
                    "id": p.id,
                    "name": p.name,
                    "geometry_wkt": wkt.dumps(p.geometry) if p.geometry else None,                    
                    "material_id": p.material_id,
                    "material_assignment_mode": getattr(p, "material_assignment_mode", "homogeneous"),
                    "heterogeneity_method": getattr(p, "heterogeneity_method", "region_based"),
                    "heterogeneity_config": copy.deepcopy(getattr(p, "heterogeneity_config", {})),
                    "material_field_config": copy.deepcopy(getattr(p, "material_field_config", {})),
                    "material_symmetry": getattr(p, "material_symmetry", "isotropic"),
                    "material_behavior": getattr(p, "material_behavior", "elastic"),
                    "material_damage": getattr(p, "material_damage", "none"),
                    "parent_id": p.parent_id,
                    "is_void": p.is_void,
                    "is_rigid": p.is_rigid,
                    "is_direct_edit": getattr(p, "is_direct_edit", False),
                    "part_type": getattr(p, "part_type", "void" if p.is_void else "solid"),
                    "particles": copy.deepcopy(getattr(p, "particles", [])),
                    "storage_units": getattr(p, "storage_units", None) or "ui",
                    "sketches": copy.deepcopy(getattr(p, "sketches", [])),
                    "sketch_meta": copy.deepcopy(getattr(p, "sketch_meta", [])),
                    "dimensions": copy.deepcopy(getattr(p, "dimensions", [])),
                    "constraints": copy.deepcopy(getattr(p, "constraints", [])),
                    "cad_source": copy.deepcopy(getattr(p, "cad_source", None)),
                    "generated_feature_kind": getattr(p, "generated_feature_kind", None),
                    "generated_feature_settings": copy.deepcopy(getattr(p, "generated_feature_settings", None)),
                }
                for p in self.parts
            ]
        }
    
    def serialize_materials(self):
        return {
            serial: {
                "name": mat.name,
                "type": mat.mat_type,
                "behavior": getattr(mat, "behavior", "elastic"),
                "damage": getattr(mat, "damage", "none"),
                "symmetry": getattr(mat, "symmetry", "isotropic"),
                "properties": mat.properties
            }
            for serial, mat in self.materials.items()
        }

    def serialize_interfaces(self):
        return [
            {
                "id": iface.get("id") if isinstance(iface, dict) else iface.id,
                "part1_id": iface.get("part1_id") if isinstance(iface, dict) else iface.part1_id,
                "part2_id": iface.get("part2_id") if isinstance(iface, dict) else iface.part2_id,
                "type": (
                    iface.get("interface_type", iface.get("type", "GLUE"))
                    if isinstance(iface, dict)
                    else getattr(iface, "interface_type", "GLUE")
                ),
                "friction": float(
                    iface.get("friction_coeff", iface.get("friction", 0.0))
                    if isinstance(iface, dict)
                    else getattr(iface, "friction_coeff", 0.0)
                ),
                "material_id": iface.get("material_id") if isinstance(iface, dict) else getattr(iface, "material_id", None),
                "material_mode": (
                    iface.get("material_mode", "auto")
                    if isinstance(iface, dict)
                    else getattr(iface, "material_mode", "auto")
                ),
                "thickness": iface.get("thickness") if isinstance(iface, dict) else getattr(iface, "thickness", None),
                "target_dx": iface.get("target_dx") if isinstance(iface, dict) else getattr(iface, "target_dx", None),
                "layer_mode": (
                    iface.get("layer_mode", "single_layer_ring")
                    if isinstance(iface, dict)
                    else getattr(iface, "layer_mode", "single_layer_ring")
                ),
                "placement_mode": (
                    iface.get("placement_mode", getattr(Interface, "DEFAULT_PLACEMENT_MODE", "matrix_side"))
                    if isinstance(iface, dict)
                    else getattr(iface, "placement_mode", getattr(Interface, "DEFAULT_PLACEMENT_MODE", "matrix_side"))
                ),
                "status": iface.get("status", "") if isinstance(iface, dict) else getattr(iface, "status", ""),
                "notes": iface.get("notes", "") if isinstance(iface, dict) else getattr(iface, "notes", ""),
            }
            for iface in self.interfaces
        ]

    def deserialize_interfaces(self, interface_items):
        self.interfaces = []
        max_id = 0
        for item in interface_items or []:
            try:
                iface = Interface(item["part1_id"], item["part2_id"], item.get("type", "GLUE"))
                iface.id = item.get("id", iface.id)
                iface.friction_coeff = float(item.get("friction", getattr(iface, "friction_coeff", 0.0)))
                iface.material_id = item.get("material_id")
                iface.material_mode = item.get("material_mode", getattr(iface, "material_mode", "auto"))
                iface.thickness = item.get("thickness")
                iface.target_dx = item.get("target_dx")
                iface.layer_mode = item.get("layer_mode", getattr(iface, "layer_mode", "single_layer_ring"))
                iface.placement_mode = item.get(
                    "placement_mode",
                    getattr(iface, "placement_mode", getattr(Interface, "DEFAULT_PLACEMENT_MODE", "matrix_side")),
                )
                iface.status = item.get("status", "")
                iface.notes = item.get("notes", "")
                self.interfaces.append(iface)
                max_id = max(max_id, iface.id)
            except Exception:
                continue
        Interface._interface_counter = max_id
        Interface._interaction_counter = max_id

    def _path_is_closed(self, pts, tol=1e-6):
        if not pts or len(pts) < 3:
            return False
        return dist(pts[0], pts[-1]) <= tol

    def _add_imported_part(self, pts, name):
        if not pts or len(pts) < 3:
            return False
        ring = list(pts)
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        try:
            geom = Polygon(ring)
        except Exception:
            return False
        if geom.is_empty:
            return False

        part = Part(name, geometry=geom)
        part.sketches = [ring]
        part.sketch_meta = [{"type": "polyline", "points": ring}]
        part.dimensions = []
        part.constraints = []
        part.storage_units = "ui"
        self._auto_create_dimensions("part", part, 0, part.sketch_meta[0])
        self._auto_create_constraints("part", part, 0, part.sketch_meta[0])
        for existing_part in self.parts:
            if not existing_part.is_void and existing_part.geometry and existing_part.geometry.covers(geom):
                part.parent_id = existing_part.id
                break
        self.parts.append(part)
        return True

    def add_imported_geometry(self, sketches, convert_closed=False, base_name="Imported Part"):
        added_sketches = 0
        added_parts = 0

        if sketches:
            self.push_undo_state()

        for pts in sketches:
            if not pts or len(pts) < 2:
                continue
            if convert_closed and self._path_is_closed(pts):
                part_name = f"{base_name} {len(self.parts) + 1}"
                if self._add_imported_part(pts, part_name):
                    added_parts += 1
                continue
            self._append_sketch(list(pts), meta={"type": "polyline", "points": list(pts)})
            added_sketches += 1

        if added_parts:
            self.rebuild_display_geometry()
            self.partsChanged.emit()
            self.geometryChanged.emit()

        if added_sketches:
            self.geometryChanged.emit()

        if added_parts or added_sketches:
            self.restore_default_sketch_tool()
            self.redraw()

        return added_sketches, added_parts

    def add_imported_sketches(self, sketches):
        added_sketches, _ = self.add_imported_geometry(sketches, convert_closed=False)
        return added_sketches

    def import_cad_shape(self, path):
        if not self._cad_kernel_ready():
            QMessageBox.warning(
                self,
                "CAD Import",
                "CAD kernel not available. Install pythonocc-core (or OCP) to enable 3D CAD import.",
            )
            return False
        shape = self.cad_kernel.import_shape(path)
        if shape is None:
            QMessageBox.warning(self, "CAD Import", "Unable to read CAD file.")
            return False
        name = os.path.splitext(os.path.basename(path))[0] or f"Imported Solid {len(self.parts) + 1}"
        part = Part(name, geometry=None)
        part.cad_shape = shape
        part.cad_source = {"type": "import", "path": path}
        self.parts.append(part)
        self.partsChanged.emit()
        self.geometryChanged.emit()
        if self.project_mode == "3d":
            self.build_3d_mesh()
        self.redraw()
        return True

    def _parse_point_tokens(self, tokens, idx, base_point=None):
        if base_point is None:
            base_point = self.command_last_point
        token = tokens[idx]
        consumed = 1
        if token.startswith("@"):
            rel = token[1:]
            if "<" in rel:
                dist_str, ang_str = rel.split("<", 1)
                dist_val = float(dist_str)
                ang_val = math.radians(float(ang_str))
                dx = dist_val * math.cos(ang_val)
                dy = dist_val * math.sin(ang_val)
            elif "," in rel:
                dx_str, dy_str = rel.split(",", 1)
                dx = float(dx_str)
                dy = float(dy_str)
            else:
                if idx + 1 >= len(tokens):
                    raise ValueError("Relative point needs dx dy")
                dx = float(rel)
                dy = float(tokens[idx + 1])
                consumed = 2
            return (base_point[0] + dx, base_point[1] + dy), consumed

        if "," in token:
            x_str, y_str = token.split(",", 1)
            return (float(x_str), float(y_str)), consumed

        if idx + 1 >= len(tokens):
            raise ValueError("Point needs x y")
        x_val = float(token)
        y_val = float(tokens[idx + 1])
        consumed = 2
        return (x_val, y_val), consumed

    def execute_command(self, text):
        raw = text.strip()
        if not raw:
            return False, "Empty command."
        tokens = raw.split()
        cmd = tokens[0].lower()

        if cmd in ("help", "?"):
            return (
                True,
                "Commands: line, rect, circle, slot, polygon, polyline, confirm, cut, undo, redo, snap.",
            )

        if cmd == "undo":
            self.undo()
            return True, "Undo."

        if cmd == "redo":
            self.redo()
            return True, "Redo."

        if cmd == "confirm":
            self.confirm_solid()
            return True, "Confirm part."

        if cmd == "cut":
            self.cut_hole()
            return True, "Cut hole."

        if cmd == "snap":
            if len(tokens) >= 3:
                target = tokens[1].lower()
                state = tokens[2].lower()
                enabled = state in ("on", "1", "true")
                if target == "grid":
                    self.set_snap_grid(enabled)
                    return True, f"Snap grid {'on' if enabled else 'off'}."
                if target in ("end", "endpoints"):
                    self.set_snap_endpoints(enabled)
                    return True, f"Snap endpoints {'on' if enabled else 'off'}."
            return False, "Usage: snap grid on|off OR snap endpoints on|off."

        if cmd == "line":
            if len(tokens) < 2:
                return False, "Usage: line x y x y OR line @dx @dy."
            idx = 1
            p1 = self.command_last_point
            p2, consumed = self._parse_point_tokens(tokens, idx, base_point=p1)
            idx += consumed
            if idx < len(tokens):
                p1 = p2
                p2, consumed = self._parse_point_tokens(tokens, idx, base_point=p1)
            self.push_undo_state()
            meta = self._line_meta_from_points(p1, p2)
            self._append_sketch(self._build_points_from_meta(meta, fallback_points=[]), meta=meta)
            self.command_last_point = p2
            self.geometryChanged.emit()
            self.redraw()
            return True, "Line added."

        if cmd == "rect":
            if len(tokens) == 3:
                origin = self.command_last_point
                width = float(tokens[1])
                height = float(tokens[2])
            else:
                if len(tokens) < 5:
                    return False, "Usage: rect x y w h OR rect w h."
                origin, consumed = self._parse_point_tokens(tokens, 1)
                width = float(tokens[1 + consumed])
                height = float(tokens[2 + consumed])
            x0, y0 = origin
            pts = [
                (x0, y0),
                (x0 + width, y0),
                (x0 + width, y0 + height),
                (x0, y0 + height),
                (x0, y0),
            ]
            self.push_undo_state()
            meta = self._rectangle_meta_from_origin_size((x0, y0), width, height, mode="two_corner")
            self._append_sketch(pts, meta=meta)
            self.command_last_point = (x0 + width, y0 + height)
            self.geometryChanged.emit()
            self.redraw()
            return True, "Rectangle added."

        if cmd == "circle":
            if len(tokens) == 2:
                center = self.command_last_point
                radius = float(tokens[1])
            else:
                if len(tokens) < 4:
                    return False, "Usage: circle x y r OR circle r."
                center, consumed = self._parse_point_tokens(tokens, 1)
                radius = float(tokens[1 + consumed])
            cx, cy = center
            self.push_undo_state()
            meta = self._circle_meta_from_center_radius((cx, cy), radius)
            pts = self._build_points_from_meta(meta, fallback_points=[])
            self._append_sketch(pts, meta=meta)
            self.command_last_point = (cx + radius, cy)
            self.geometryChanged.emit()
            self.redraw()
            return True, "Circle added."

        if cmd == "slot":
            if len(tokens) < 3:
                return False, "Usage: slot x y w OR slot x y x y w."
            idx = 1
            p1 = self.command_last_point
            p2, consumed = self._parse_point_tokens(tokens, idx, base_point=p1)
            idx += consumed
            remaining = len(tokens) - idx
            if remaining >= 2:
                p1 = p2
                p2, consumed = self._parse_point_tokens(tokens, idx, base_point=p1)
                idx += consumed
            if idx >= len(tokens):
                return False, "Usage: slot x y w OR slot x y x y w."
            width = float(tokens[idx])
            verts = self._build_slot_vertices(p1, p2, width)
            if not verts:
                return False, "Slot length too small."
            self.push_undo_state()
            meta = {"type": "slot", "p1": p1, "p2": p2, "width": width}
            self._append_sketch(verts, meta=meta)
            self.command_last_point = p2
            self.geometryChanged.emit()
            self.redraw()
            return True, "Slot added."

        if cmd in ("polygon", "poly"):
            if len(tokens) == 3:
                sides = int(tokens[1])
                center = self.command_last_point
                radius = float(tokens[2])
            else:
                if len(tokens) < 5:
                    return False, "Usage: polygon n x y r OR polygon n r."
                sides = int(tokens[1])
                center, consumed = self._parse_point_tokens(tokens, 2)
                radius = float(tokens[2 + consumed])
            if sides < 3:
                return False, "Polygon needs at least 3 sides."
            cx, cy = center
            pts = []
            for i in range(sides):
                theta = 2 * math.pi * i / sides
                pts.append((cx + radius * math.cos(theta), cy + radius * math.sin(theta)))
            pts.append(pts[0])
            self.push_undo_state()
            meta = {"type": "polygon", "center": (cx, cy), "radius": radius, "sides": sides}
            self._append_sketch(pts, meta=meta)
            self.command_last_point = pts[-2]
            self.geometryChanged.emit()
            self.redraw()
            return True, "Polygon added."

        if cmd in ("polyline", "pline"):
            if len(tokens) < 3:
                return False, "Usage: polyline x y x y [...]."
            pts = []
            idx = 1
            base = self.command_last_point
            while idx < len(tokens):
                point, consumed = self._parse_point_tokens(tokens, idx, base_point=base)
                pts.append(point)
                base = point
                idx += consumed
            if len(pts) < 2:
                return False, "Polyline needs at least 2 points."
            self.push_undo_state()
            meta = {"type": "polyline", "points": list(pts)}
            self._append_sketch(pts, meta=meta)
            self.command_last_point = pts[-1]
            self.geometryChanged.emit()
            self.redraw()
            return True, "Polyline added."

        return False, f"Unknown command: {cmd}"

    def deserialize_geometry(self, geometry_data):
        self.parts.clear()
        parts_data = geometry_data.get("parts", [])
        max_id = 0
        for p_data in parts_data:
            geom = wkt.loads(p_data["geometry_wkt"]) if p_data.get("geometry_wkt") else None
            part = Part(p_data.get("name", "Unnamed Part"), geom)
            part.id = p_data["id"]
            part.parent_id = p_data.get("parent_id")
            part.material_id = p_data.get("material_id")
            part.material_assignment_mode = str(
                p_data.get("material_assignment_mode", "homogeneous") or "homogeneous"
            )
            part.heterogeneity_method = str(p_data.get("heterogeneity_method", "region_based") or "region_based")
            part.heterogeneity_config = normalize_heterogeneity_config(
                p_data.get("heterogeneity_config", {})
            )
            part.material_field_config = normalize_material_field_config(
                p_data.get("material_field_config", {})
            )
            part.material_symmetry = normalize_material_symmetry(p_data.get("material_symmetry", "isotropic"))
            part.material_behavior = normalize_material_behavior(p_data.get("material_behavior", "elastic"))
            part.material_damage = normalize_material_damage(p_data.get("material_damage", "none"))
            part.is_void = p_data.get("is_void", False)
            part.is_rigid = p_data.get("is_rigid", False)
            part.is_direct_edit = p_data.get("is_direct_edit", False)
            part.part_type = p_data.get("part_type", "void" if part.is_void else "solid")
            part.particles = copy.deepcopy(p_data.get("particles", []))
            part.storage_units = p_data.get("storage_units") or "ui"
            part.sketches = copy.deepcopy(p_data.get("sketches", []))
            part.sketch_meta = copy.deepcopy(p_data.get("sketch_meta", []))
            part.dimensions = copy.deepcopy(p_data.get("dimensions", []))
            part.constraints = copy.deepcopy(p_data.get("constraints", []))
            part.cad_source = copy.deepcopy(p_data.get("cad_source", None))
            part.generated_feature_kind = p_data.get("generated_feature_kind")
            part.generated_feature_settings = copy.deepcopy(p_data.get("generated_feature_settings", None))
            self._normalize_part_storage_units(part)
            if part.cad_source and self.cad_kernel.available():
                try:
                    src_path = part.cad_source.get("path")
                    if src_path:
                        part.cad_shape = self.cad_kernel.import_shape(src_path)
                except Exception:
                    part.cad_shape = None
            self.parts.append(part)
            if part.id > max_id:
                max_id = part.id
        Part._part_counter = max_id

    def deserialize_materials(self, data):
        self.materials = {}

        def _coerce_material(entry, fallback_serial=None):
            if isinstance(entry, Material):
                mat = copy.deepcopy(entry)
                try:
                    mat.serial = int(getattr(mat, "serial", fallback_serial))
                except Exception:
                    if fallback_serial is not None:
                        mat.serial = fallback_serial
                mat.symmetry = normalize_material_symmetry(getattr(mat, "symmetry", "isotropic"))
                mat.behavior = normalize_material_behavior(getattr(mat, "behavior", getattr(mat, "mat_type", "elastic")))
                mat.damage = normalize_material_damage(getattr(mat, "damage", "none"))
                mat.properties = normalize_material_properties(
                    copy.deepcopy(getattr(mat, "properties", {}) or {}),
                    mat.behavior,
                    mat.symmetry,
                    mat.damage,
                )
                return mat
            if isinstance(entry, dict):
                mat = Material(
                    entry["name"],
                    entry.get("mat_type", entry.get("type")),
                    entry["properties"],
                    symmetry=entry.get("symmetry", "isotropic"),
                    behavior=entry.get("behavior"),
                    damage=entry.get("damage", "none"),
                )
                if "serial" in entry:
                    mat.serial = int(entry["serial"])
                elif fallback_serial is not None:
                    mat.serial = int(fallback_serial)
                mat.symmetry = normalize_material_symmetry(entry.get("symmetry", getattr(mat, "symmetry", "isotropic")))
                mat.behavior = normalize_material_behavior(entry.get("behavior", getattr(mat, "behavior", getattr(mat, "mat_type", "elastic"))))
                mat.damage = normalize_material_damage(entry.get("damage", getattr(mat, "damage", "none")))
                mat.properties = normalize_material_properties(
                    copy.deepcopy(entry.get("properties", getattr(mat, "properties", {})) or {}),
                    mat.behavior,
                    mat.symmetry,
                    mat.damage,
                )
                return mat
            return None

        # Case 1: Data is a LIST
        if isinstance(data, list):
            for d in data:
                mat = _coerce_material(d)
                if mat is not None:
                    self.materials[int(mat.serial)] = mat

        # Case 2: Data is a DICTIONARY keyed by serial
        elif isinstance(data, dict):
            for s, d in data.items():
                try:
                    serial = int(s)
                except Exception:
                    serial = s
                mat = _coerce_material(d, fallback_serial=serial)
                if mat is not None:
                    self.materials[int(mat.serial)] = mat

        # Reset the counter to avoid ID conflicts for new materials
        if self.materials:
            Material._serial_counter = max(self.materials.keys())
    
    def export_mesh_csv(self, silent=False):
        self.export_csv(silent=silent)

    # --- Mesh Logic ---
    def run_cpd(self, preview_only=False, mesh_config=None):
        if not self.parts:
            QMessageBox.warning(self, "Error", "No parts to connect.")
            return False
        mesh_start = time.perf_counter()
        dx = None
        target_nodes = None
        mesh_backend = getattr(self, "mesh_backend", "auto")
        if mesh_config:
            mode = str(mesh_config.get("mode", "dx")).lower()
            if mode in ("dx", "spacing"):
                try:
                    dx = float(mesh_config.get("dx", DEFAULT_DX))
                except (TypeError, ValueError):
                    QMessageBox.warning(self, "Error", "Invalid spacing (dx).")
                    return False
            elif mode in ("count", "nodes", "total"):
                try:
                    target_nodes = int(mesh_config.get("target_nodes", 1000))
                except (TypeError, ValueError):
                    QMessageBox.warning(self, "Error", "Invalid target particle count.")
                    return False
            else:
                QMessageBox.warning(self, "Error", "Invalid particle connection sizing mode.")
                return False
            mesh_distribution = str(mesh_config.get("distribution", "global_poisson")).lower()
            if mesh_distribution not in ("poisson", "square", "global_poisson", "global_square"):
                mesh_distribution = "global_poisson"
            self.mesh_distribution = mesh_distribution
            mesh_backend = str(mesh_config.get("backend", mesh_backend)).lower()
        else:
            mesh_modes = ["By spacing (dx)", "By total particle count"]
            mesh_mode, ok = QInputDialog.getItem(
                self,
                "Particle Connections",
                "Choose sizing method:",
                mesh_modes,
                0,
                False,
            )
            if not ok:
                return False
            if mesh_mode == "By spacing (dx)":
                dx, ok = QInputDialog.getDouble(
                    self,
                    "Particle Connections",
                    f"Spacing dx ({self.current_unit}):",
                    DEFAULT_DX,
                    0.1,
                    5000,
                    2,
                )
                if not ok:
                    return False
            else:
                target_nodes, ok = QInputDialog.getInt(
                    self,
                    "Particle Connections",
                    "Target total particles:",
                    1000,
                    10,
                    1000000,
                    10,
                )
                if not ok:
                    return False

            mesh_distributions = [
                "Global Poisson (uniform)",
                "Global Square lattice",
                "Per-part Poisson",
                "Per-part Square lattice",
            ]
            default_dist = 0
            if getattr(self, "mesh_distribution", "global_poisson") == "global_square":
                default_dist = 1
            elif getattr(self, "mesh_distribution", "global_poisson") == "poisson":
                default_dist = 2
            elif getattr(self, "mesh_distribution", "global_poisson") == "square":
                default_dist = 3
            mesh_distribution_label, ok = QInputDialog.getItem(
                self,
                "Particle Connections",
                "Choose particle distribution:",
                mesh_distributions,
                default_dist,
                False,
            )
            if not ok:
                return False
            mesh_distribution = "global_poisson"
            if "Global Square" in mesh_distribution_label:
                mesh_distribution = "global_square"
            elif "Per-part Poisson" in mesh_distribution_label:
                mesh_distribution = "poisson"
            elif "Per-part Square" in mesh_distribution_label:
                mesh_distribution = "square"
            self.mesh_distribution = mesh_distribution

            backend_labels = [
                "Auto (fastest available)",
                "Triangle (fastest 2D)",
                "Gmsh",
                "CGAL/pygalmesh",
                "SciPy Delaunay (legacy)",
            ]
            backend_choice, ok = QInputDialog.getItem(
                self,
                "Particle Connections",
                "Choose triangulation backend:",
                backend_labels,
                0,
                False,
            )
            if not ok:
                return False
            label = backend_choice.lower()
            if "triangle" in label:
                mesh_backend = "triangle"
            elif "gmsh" in label:
                mesh_backend = "gmsh"
            elif "cgal" in label or "pygalmesh" in label:
                mesh_backend = "pygalmesh"
            elif "scipy" in label:
                mesh_backend = "scipy"
            else:
                mesh_backend = "auto"

        mesh_backend = self._select_mesh_backend(mesh_backend, dx, target_nodes, self.mesh_distribution)
        self.mesh_backend = mesh_backend
        dx, target_nodes, proceed = self._guard_mesh_density(dx, target_nodes, self.mesh_distribution)
        if not proceed:
            return False

        dx, target_nodes, proceed = self._guard_mesh_density(dx, target_nodes, self.mesh_distribution)
        if not proceed:
            return False

        if self.project_mode == "3d":
            height = float(self.extrude_height)
            layers = int(self.extrude_layers)
            if height <= 0.0 or layers < 1:
                QMessageBox.warning(
                    self,
                    "Extrude",
                    "Extrude height and layers must be positive for 3D connection preview.",
                )
                return False
            self.extrude_height = height
            self.extrude_layers = layers

        self.part_meshes.clear()
        self.global_nodes = np.array([])
        self.global_elements = np.array([])
        self.element_part_map_3d = []
        self.element_part_map = []

        parts_to_mesh = [p for p in self.parts if p.material_id is not None and not p.is_void]
        if not parts_to_mesh:
            QMessageBox.warning(self, "Error", "No parts with assigned material to connect.")
            return False

        all_boundaries = []
        part_effective_geoms = {}
        rigid_part_ids = set()
        total_area = 0.0

        def _append_unique(points_list, points_set, pt):
            if pt in points_set:
                return
            points_set.add(pt)
            points_list.append(pt)

        def _add_spaced_points(coords, min_spacing, points_list, points_set):
            last = None
            for pt in coords:
                p = tuple(pt)
                if last is None or dist(last, p) >= min_spacing:
                    _append_unique(points_list, points_set, p)
                    last = p
            if coords:
                _append_unique(points_list, points_set, tuple(coords[-1]))

        def _add_ring_samples(ring, spacing, points_list, points_set, min_points=12):
            if ring is None:
                return []
            try:
                length = float(ring.length)
            except Exception:
                length = 0.0
            if length <= 0:
                return []
            target = max(1.0, float(spacing))
            if min_points and length > 0:
                target = min(target, length / max(min_points, 1))
            samples = None
            try:
                coords = [tuple(pt) for pt in list(ring.coords)]
            except Exception:
                coords = []
            # For low-vertex polygonal edges (e.g., rectangles), sample each segment separately so
            # edge spacing stays uniform and corner-anchored instead of drifting around full perimeter.
            unique_count = 0
            if coords:
                unique_count = len(coords)
                if len(coords) > 1 and dist(coords[0], coords[-1]) <= 1e-9:
                    unique_count -= 1
            if 2 <= unique_count <= 16 and len(coords) >= 2:
                samples = []
                for seg_idx in range(len(coords) - 1):
                    a = coords[seg_idx]
                    b = coords[seg_idx + 1]
                    seg_len = dist(a, b)
                    if seg_len <= 1e-12:
                        continue
                    # Equal subdivision per edge gives more uniform spacing on straight edges.
                    nseg = max(1, int(round(seg_len / target)))
                    for j in range(nseg + 1):
                        if seg_idx > 0 and j == 0:
                            continue
                        t = float(j) / float(nseg)
                        samples.append(
                            [
                                (1.0 - t) * float(a[0]) + t * float(b[0]),
                                (1.0 - t) * float(a[1]) + t * float(b[1]),
                            ]
                        )
            if samples is None:
                samples = sample_ring(ring, target)
            out = []
            for pt in samples:
                p = tuple(pt)
                _append_unique(points_list, points_set, p)
                out.append(p)
            return out

        def _add_ring_vertices(
            ring,
            points_list,
            points_set,
            protected_list=None,
            protected_set=None,
            corner_angle_deg=15.0,
        ):
            if ring is None:
                return
            try:
                coords = list(ring.coords)
            except Exception:
                return
            if not coords:
                return
            pts = [tuple(pt) for pt in coords]
            if len(pts) > 1 and dist(pts[0], pts[-1]) <= 1e-9:
                pts = pts[:-1]
            if not pts:
                return

            closed = len(pts) >= 3
            angle_thresh = math.radians(max(0.0, float(corner_angle_deg)))
            keep = []

            def _mark(pt_tuple):
                _append_unique(points_list, points_set, pt_tuple)
                if protected_list is not None and protected_set is not None and pt_tuple not in protected_set:
                    protected_set.add(pt_tuple)
                    protected_list.append(pt_tuple)

            if not closed:
                if pts:
                    _mark(pts[0])
                    if len(pts) > 1:
                        _mark(pts[-1])
                return

            n = len(pts)
            for i in range(n):
                prev_pt = pts[(i - 1) % n]
                pt = pts[i]
                next_pt = pts[(i + 1) % n]
                v1 = (pt[0] - prev_pt[0], pt[1] - prev_pt[1])
                v2 = (next_pt[0] - pt[0], next_pt[1] - pt[1])
                l1 = math.hypot(v1[0], v1[1])
                l2 = math.hypot(v2[0], v2[1])
                if l1 <= 1e-9 or l2 <= 1e-9:
                    keep.append(pt)
                    continue
                dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)
                dot = max(-1.0, min(1.0, dot))
                turn = math.acos(dot)
                if turn >= angle_thresh:
                    keep.append(pt)

            if not keep and pts:
                # Fallback for degenerate/over-simplified rings: keep one anchor.
                keep = [pts[0]]
            for pt in keep:
                _mark(pt)

        def _filter_points_min_spacing(boundary_pts, interior_pts, min_spacing, protected=None):
            if not boundary_pts and not interior_pts:
                return []
            cell = max(min_spacing, 1e-9)
            min_sq = min_spacing * min_spacing
            grid = {}
            kept = []
            protected = protected or set()

            def _grid_key(p):
                return (int(math.floor(p[0] / cell)), int(math.floor(p[1] / cell)))

            def _can_place(p):
                gx, gy = _grid_key(p)
                for i in range(gx - 1, gx + 2):
                    for j in range(gy - 1, gy + 2):
                        for q in grid.get((i, j), []):
                            dx = p[0] - q[0]
                            dy = p[1] - q[1]
                            if dx * dx + dy * dy < min_sq:
                                return False
                return True

            # Keep all boundary points (do not cull) to preserve edges/corners.
            for p in boundary_pts:
                if p in protected or _can_place(p):
                    kept.append(p)
                    grid.setdefault(_grid_key(p), []).append(p)
            # Cull only interior points against existing boundary + interior.
            for p in interior_pts:
                if _can_place(p):
                    kept.append(p)
                    grid.setdefault(_grid_key(p), []).append(p)
            return kept

        # 1. Collect all boundaries and effective geometries
        for part in parts_to_mesh:
            mat_type = (part.material_type or "").lower()
            is_rigid_part = part.is_rigid or mat_type in ("rigid", "rigid_body")
            if is_rigid_part:
                rigid_part_ids.add(part.id)

            effective_geom = part.geometry
            if effective_geom is None or effective_geom.is_empty:
                continue
            
            children = self.get_child_parts(part)
            for child in children:
                if child.geometry and not child.geometry.is_empty:
                    try:
                        effective_geom = effective_geom.difference(child.geometry).buffer(0)
                    except Exception:
                        pass # Ignore topology errors
            
            if not effective_geom or effective_geom.is_empty:
                continue
            
            part_effective_geoms[part.id] = effective_geom

            geoms = [effective_geom] if isinstance(effective_geom, Polygon) else list(effective_geom.geoms)
            for g in geoms:
                all_boundaries.append(g.exterior)
                for interior in g.interiors:
                    all_boundaries.append(interior)

        boundary_union = unary_union(all_boundaries) if all_boundaries else None
        geoms_to_sample = []
        if boundary_union is not None:
            if boundary_union.geom_type == "LineString":
                geoms_to_sample.append(boundary_union)
            elif boundary_union.geom_type in ("MultiLineString", "GeometryCollection"):
                geoms_to_sample.extend(list(boundary_union.geoms))

        lattice_origin = None
        if mesh_distribution == "square":
            bounds = [
                geom.bounds for geom in part_effective_geoms.values()
                if geom is not None and not geom.is_empty
            ]
            if bounds:
                minx = min(b[0] for b in bounds)
                miny = min(b[1] for b in bounds)
                lattice_origin = (minx, miny)
        elif mesh_distribution == "global_square":
            bounds = [
                geom.bounds for geom in part_effective_geoms.values()
                if geom is not None and not geom.is_empty
            ]
            if bounds:
                minx = min(b[0] for b in bounds)
                miny = min(b[1] for b in bounds)
                lattice_origin = (minx, miny)

        union_geom = None
        if mesh_distribution.startswith("global"):
            geoms = [g for g in part_effective_geoms.values() if g is not None and not g.is_empty]
            if geoms:
                try:
                    union_geom = unary_union(geoms)
                    if union_geom is not None and not union_geom.is_empty:
                        try:
                            union_geom = union_geom.buffer(0)
                        except Exception:
                            pass
                except Exception:
                    union_geom = None

        triangulation_constraints = {
            "segments": None,
            "holes": None,
            "regions": None,
            "region_attr_map": {},
            "spacing": None,
        }

        def _collect_mesh_points(spacing):
            triangulation_constraints["segments"] = None
            triangulation_constraints["holes"] = None
            triangulation_constraints["regions"] = None
            triangulation_constraints["region_attr_map"] = {}
            try:
                spacing = float(spacing)
                triangulation_constraints["spacing"] = float(spacing)
            except Exception:
                triangulation_constraints["spacing"] = None
                return []

            min_spacing_factor = max(0.1, float(getattr(self, "mesh_min_spacing_factor", MESH_MIN_SPACING_FACTOR)))
            min_spacing = max(1e-9, spacing * min_spacing_factor)
            boundary_points = []
            boundary_set = set()
            boundary_vertices = []
            boundary_vertices_set = set()
            interior_points = []
            interior_set = set()
            layer_boundaries = []
            constraint_chains = []
            boundary_thickness = max(0.0, float(getattr(self, "mesh_boundary_thickness", 0.0)))
            boundary_spacing_factor = max(
                0.1, min(1.0, float(getattr(self, "mesh_boundary_spacing_factor", 1.0)))
            )
            boundary_spacing = max(1e-9, spacing * boundary_spacing_factor)
            layer_spacing = max(1e-9, spacing * boundary_spacing_factor)
            uniform_boundary = boundary_spacing_factor >= 0.99
            boundary_min_points = 1 if uniform_boundary else 16
            interface_payload = self._build_interface_mesh_sampling_payload(part_effective_geoms, spacing)
            interface_sampling = interface_payload.get("interface_sampling", {})
            skip_boundary_specs = list(interface_payload.get("skip_boundary_specs", []) or [])
            circle_fill_specs = list(interface_payload.get("circle_fill_specs", []) or [])

            def _skip_generic_boundary_ring(ring):
                for spec in skip_boundary_specs:
                    try:
                        if self._ring_matches_skip_spec(ring, spec):
                            return True
                    except Exception:
                        continue
                return False

            def _append_constraint_chain(points_seq, closed=False):
                try:
                    seq = [tuple(map(float, p[:2])) for p in (points_seq if points_seq is not None else [])]
                except Exception:
                    seq = []
                if len(seq) < 2:
                    return
                compact = []
                for p in seq:
                    if compact and compact[-1] == p:
                        continue
                    compact.append(p)
                if len(compact) < 2:
                    return
                if closed and len(compact) > 2 and compact[0] == compact[-1]:
                    compact = compact[:-1]
                if len(compact) < 2:
                    return
                constraint_chains.append({"points": compact, "closed": bool(closed)})

            def _add_lattice_points(geom):
                if geom is None or getattr(geom, "is_empty", True):
                    return
                try:
                    geoms = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
                except Exception:
                    geoms = [geom]
                for g in geoms:
                    if g is None or getattr(g, "is_empty", True):
                        continue
                    if not isinstance(g, Polygon):
                        continue
                    try:
                        lattice_pts = square_lattice_sample(g, spacing, origin=lattice_origin)
                    except Exception:
                        lattice_pts = []
                    for pt in lattice_pts:
                        _append_unique(interior_points, interior_set, tuple(pt))

            if boundary_thickness > 1e-9:
                num_layers = max(1, int(math.ceil(boundary_thickness / layer_spacing)))
                step = boundary_thickness / num_layers if num_layers > 0 else boundary_thickness
                for part_id, effective_geom in part_effective_geoms.items():
                    if effective_geom is None or effective_geom.is_empty:
                        continue
                    for i in range(1, num_layers + 1):
                        offset = step * i
                        try:
                            inner = effective_geom.buffer(-offset)
                        except Exception:
                            continue
                        if inner.is_empty:
                            break
                        geoms = [inner] if isinstance(inner, Polygon) else list(inner.geoms)
                        for g in geoms:
                            layer_boundaries.append(g.exterior)
                            for interior in g.interiors:
                                layer_boundaries.append(interior)

            for ring in all_boundaries:
                if _skip_generic_boundary_ring(ring):
                    continue
                _add_ring_vertices(
                    ring,
                    boundary_points,
                    boundary_set,
                    boundary_vertices,
                    boundary_vertices_set,
                    corner_angle_deg=15.0,
                )
                ring_chain = _add_ring_samples(
                    ring,
                    boundary_spacing,
                    boundary_points,
                    boundary_set,
                    min_points=boundary_min_points,
                )
                _append_constraint_chain(ring_chain, closed=True)
            for ring in layer_boundaries:
                _add_ring_vertices(
                    ring,
                    boundary_points,
                    boundary_set,
                    boundary_vertices,
                    boundary_vertices_set,
                    corner_angle_deg=15.0,
                )
                ring_chain = _add_ring_samples(
                    ring,
                    layer_spacing,
                    boundary_points,
                    boundary_set,
                    min_points=boundary_min_points,
                )
                _append_constraint_chain(ring_chain, closed=True)
            # Do not resample boundary_union lines here. all_boundaries already contains exteriors/interiors,
            # and a second pass can add phase-shifted points that make edge spacing appear non-uniform.
            boundary_points, boundary_vertices, interior_points = self._merge_interface_sampling_payload_into_point_sets(
                interface_payload,
                boundary_points=boundary_points,
                boundary_set=boundary_set,
                boundary_vertices=boundary_vertices,
                boundary_vertices_set=boundary_vertices_set,
                interior_points=interior_points,
                interior_set=interior_set,
            )
            iface_constraint_chains = [c for c in list(interface_payload.get("constraint_chains", []) or []) if c]
            if iface_constraint_chains:
                # Prefer interface chains over generic boundary chains when de-duplicating overlaps.
                constraint_chains = iface_constraint_chains + constraint_chains

            if constraint_chains:
                constraint_chains = self._dedupe_constraint_chains(constraint_chains, spacing)
            if constraint_chains:
                boundary_points, boundary_vertices = self._protect_constraint_chain_points(
                    constraint_chains,
                    boundary_points=boundary_points,
                    boundary_set=boundary_set,
                    boundary_vertices=boundary_vertices,
                    boundary_vertices_set=boundary_vertices_set,
                )

            if mesh_distribution == "global_square":
                lattice_geom = union_geom
                if lattice_geom is not None and not getattr(lattice_geom, "is_empty", True):
                    try:
                        eroded_geom = lattice_geom.buffer(-spacing * 0.5)
                    except Exception:
                        eroded_geom = None
                    if eroded_geom is not None and not getattr(eroded_geom, "is_empty", True):
                        lattice_geom = eroded_geom
                _add_lattice_points(lattice_geom)
                for spec in circle_fill_specs:
                    for pt in self._hex_lattice_sample_circle(
                        spec["center"],
                        spec["fill_radius"],
                        spacing,
                        angle=spec.get("phase", 0.0),
                    ):
                        _append_unique(interior_points, interior_set, tuple(pt))
            elif mesh_distribution == "square":
                for part_id, effective_geom in part_effective_geoms.items():
                    if effective_geom is None or effective_geom.is_empty:
                        continue
                    lattice_geom = effective_geom
                    if part_id in rigid_part_ids:
                        try:
                            boundary_band = effective_geom.boundary.buffer(spacing * 0.45)
                        except Exception:
                            boundary_band = None
                        if boundary_band is not None and not getattr(boundary_band, "is_empty", True):
                            try:
                                boundary_band = boundary_band.intersection(effective_geom)
                            except Exception:
                                pass
                            if not getattr(boundary_band, "is_empty", True):
                                lattice_geom = boundary_band
                    else:
                        try:
                            eroded_geom = effective_geom.buffer(-spacing * 0.5)
                        except Exception:
                            eroded_geom = None
                        if eroded_geom is not None and not getattr(eroded_geom, "is_empty", True):
                            lattice_geom = eroded_geom
                    _add_lattice_points(lattice_geom)
            elif mesh_distribution == "global_poisson":
                if union_geom is not None and not union_geom.is_empty:
                    eroded_geom = union_geom.buffer(-spacing * 0.5)
                    if eroded_geom is not None and not eroded_geom.is_empty and circle_fill_specs:
                        for spec in circle_fill_specs:
                            try:
                                fill_disk = Point(spec["center"]).buffer(float(spec["fill_radius"]), resolution=64)
                                eroded_geom = eroded_geom.difference(fill_disk)
                            except Exception:
                                continue
                    if not eroded_geom.is_empty:
                        geoms = [eroded_geom] if isinstance(eroded_geom, Polygon) else list(eroded_geom.geoms)
                        for g in geoms:
                            interior_pts = poisson_sample(g, spacing)
                            for pt in interior_pts:
                                _append_unique(interior_points, interior_set, tuple(pt))
                for spec in circle_fill_specs:
                    for pt in self._hex_lattice_sample_circle(
                        spec["center"],
                        spec["fill_radius"],
                        spacing,
                        angle=spec.get("phase", 0.0),
                    ):
                        _append_unique(interior_points, interior_set, tuple(pt))
            else:
                for part_id, effective_geom in part_effective_geoms.items():
                    if part_id in rigid_part_ids:
                        continue
                    circle_spec = None
                    for spec in circle_fill_specs:
                        if int(spec.get("part_id", -1)) == int(part_id):
                            circle_spec = spec
                            break
                    if circle_spec is not None:
                        for pt in self._hex_lattice_sample_circle(
                            circle_spec["center"],
                            circle_spec["fill_radius"],
                            spacing,
                            angle=circle_spec.get("phase", 0.0),
                        ):
                            _append_unique(interior_points, interior_set, tuple(pt))
                        continue
                    eroded_geom = effective_geom.buffer(-spacing * 0.5)
                    if not eroded_geom.is_empty:
                        geoms = [eroded_geom] if isinstance(eroded_geom, Polygon) else list(eroded_geom.geoms)
                        for g in geoms:
                            interior_pts = poisson_sample(g, spacing)
                            for pt in interior_pts:
                                _append_unique(interior_points, interior_set, tuple(pt))

            if self.interfaces and constraint_chains:
                constraint_chains = self._dedupe_constraint_chains(constraint_chains, spacing)
            if self.interfaces and constraint_chains:
                boundary_points, boundary_vertices = self._protect_constraint_chain_points(
                    constraint_chains,
                    boundary_points=boundary_points,
                    boundary_set=boundary_set,
                    boundary_vertices=boundary_vertices,
                    boundary_vertices_set=boundary_vertices_set,
                )
            filtered_points = _filter_points_min_spacing(
                boundary_points,
                interior_points,
                min_spacing,
                protected=boundary_vertices_set,
            )
            stabilization_geom = union_geom
            if stabilization_geom is None or getattr(stabilization_geom, "is_empty", True):
                live_geoms = [g for g in part_effective_geoms.values() if g is not None and not g.is_empty]
                if live_geoms:
                    try:
                        stabilization_geom = unary_union(live_geoms)
                    except Exception:
                        stabilization_geom = None
            filtered_points = stabilize_particle_cloud(
                filtered_points,
                spacing,
                geometry=stabilization_geom,
                boundary=boundary_union,
            )
            filtered_points = dedupe_min_distance(filtered_points, min_spacing)
            if self.interfaces and constraint_chains:
                triangulation_constraints["segments"] = self._build_triangle_pslg_segments(
                    filtered_points,
                    constraint_chains,
                    spacing,
                )
            return filtered_points

        if target_nodes is not None:
            total_area = sum(
                geom.area for pid, geom in part_effective_geoms.items()
                if pid not in rigid_part_ids
            )
            if total_area <= 0:
                total_area = sum(geom.area for geom in part_effective_geoms.values())
            if total_area <= 0:
                QMessageBox.warning(self, "Error", "Could not estimate connection area.")
                return False
            dx = math.sqrt(total_area / max(target_nodes, 1))
            dx = max(0.1, min(5000, dx))
            max_iter = 4
            all_points = set()
            for i in range(max_iter):
                all_points = _collect_mesh_points(dx)
                count = len(all_points)
                if count < 3:
                    break
                err_ratio = abs(count - target_nodes) / max(target_nodes, 1)
                if err_ratio <= 0.15 or i == max_iter - 1:
                    break
                dx *= (count / max(target_nodes, 1)) ** 0.5
                dx = max(0.1, min(5000, dx))
            window = self.window()
            if window and hasattr(window, "statusBar"):
                window.statusBar().showMessage(
                    f"Target {target_nodes} particles → using dx≈{dx:.3f} ({self.current_unit}).",
                    5000,
                )
        else:
            all_points = _collect_mesh_points(dx)

        if len(all_points) < 3:
            QMessageBox.warning(self, "Error", "Not enough unique points to generate connections.")
            return False

        est = self._estimate_mesh_time(len(all_points))
        if est is not None:
            self._announce_status(f"Generating particles & connections (~{est:.1f}s)...", 5000)
        else:
            self._announce_status("Generating particles & connections...", 4000)

        self.global_nodes = np.array(list(all_points))
        raw_particle_count = len(self.global_nodes)
        
        # 4. Perform triangulation (backend selectable)
        try:
            tri_segments = triangulation_constraints.get("segments")
            global_elements_all = self._triangulate_points(self.global_nodes, mesh_backend, segments=tri_segments)
        except Exception as e:
            QMessageBox.critical(self, "Particle Generation Error", str(e))
            return False

        # 5. Assign elements to parts
        valid_elements_list = []
        element_part_map_list = []
        part_prepared = {}
        part_eroded_prepared = {}
        part_relaxed_prepared = {}
        part_bounds = {}
        edge_margin = 0.0
        if dx is not None:
            edge_margin = max(1e-6, float(dx) * 0.2)
        cover_tol = max(1e-8, float(dx) * 1e-3) if dx is not None else 1e-6
        min_area_tol = max(1e-10, float(dx * dx) * 0.01) if dx is not None else 1e-10
        for part in parts_to_mesh:
            effective_geom = part_effective_geoms.get(part.id)
            if not effective_geom:
                continue
            part_bounds[part.id] = effective_geom.bounds
            try:
                part_prepared[part.id] = prep(effective_geom)
            except Exception:
                part_prepared[part.id] = None
            try:
                relaxed_geom = effective_geom.buffer(cover_tol)
            except Exception:
                relaxed_geom = None
            if relaxed_geom is not None and not relaxed_geom.is_empty:
                try:
                    part_relaxed_prepared[part.id] = prep(relaxed_geom)
                except Exception:
                    pass
            if edge_margin > 0:
                try:
                    eroded = effective_geom.buffer(-edge_margin)
                except Exception:
                    eroded = None
                if eroded is not None and not eroded.is_empty:
                    try:
                        part_eroded_prepared[part.id] = prep(eroded)
                    except Exception:
                        pass

        def _bounds_contains(outer, inner):
            return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]
        
        total_elements = len(global_elements_all)
        quality_rejected_count = 0
        near_zero_area_rejected = 0
        for i, element in enumerate(global_elements_all):
            tri_pts = self.global_nodes[element]
            quality = triangle_quality_metrics(tri_pts)
            if quality["area"] < float(min_area_tol):
                near_zero_area_rejected += 1
                quality_rejected_count += 1
                continue
            if quality["min_angle_deg"] < 15.0 or quality["aspect_ratio"] > 5.0:
                quality_rejected_count += 1
                continue
            try:
                xs = tri_pts[:, 0]
                ys = tri_pts[:, 1]
                tri_bounds = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
            except Exception:
                tri_bounds = None
            tri_poly = None
            p_centroid = None
            
            for part in parts_to_mesh:
                effective_geom = part_effective_geoms.get(part.id)
                if not effective_geom:
                    continue
                inside = False
                bounds = part_bounds.get(part.id)
                if tri_bounds is not None and bounds is not None:
                    if not _bounds_contains(bounds, tri_bounds):
                        continue
                eroded_prep = part_eroded_prepared.get(part.id)
                if eroded_prep is not None:
                    try:
                        if (
                            eroded_prep.covers(Point(tri_pts[0]))
                            and eroded_prep.covers(Point(tri_pts[1]))
                            and eroded_prep.covers(Point(tri_pts[2]))
                        ):
                            inside = True
                    except Exception:
                        inside = False
                if not inside:
                    prep_geom = part_prepared.get(part.id)
                    if prep_geom is not None:
                        if tri_poly is None:
                            try:
                                tri_poly = Polygon(tri_pts)
                            except Exception:
                                tri_poly = None
                        if tri_poly is not None and prep_geom.covers(tri_poly):
                            inside = True
                    if not inside:
                        relaxed_prep = part_relaxed_prepared.get(part.id)
                        if relaxed_prep is not None and tri_poly is not None:
                            try:
                                if relaxed_prep.covers(tri_poly):
                                    inside = True
                            except Exception:
                                pass
                    if not inside:
                        if p_centroid is None:
                            centroid = np.mean(tri_pts, axis=0)
                            p_centroid = Point(centroid)
                        if tri_poly is None:
                            if effective_geom.covers(p_centroid):
                                inside = True
                        else:
                            # Numeric fallback for boundary-touching triangles that should belong to
                            # this part but fail exact polygon covers() due precision at shared interfaces.
                            centroid_ok = False
                            try:
                                centroid_ok = bool(effective_geom.covers(p_centroid))
                            except Exception:
                                centroid_ok = False
                            if centroid_ok:
                                try:
                                    overlap_area = float(getattr(effective_geom.intersection(tri_poly), "area", 0.0) or 0.0)
                                except Exception:
                                    overlap_area = 0.0
                                try:
                                    tri_area = float(getattr(tri_poly, "area", 0.0) or 0.0)
                                except Exception:
                                    tri_area = 0.0
                                if tri_area > 1e-12 and (overlap_area / tri_area) >= 0.90:
                                    inside = True
                if inside:
                    current_element_index = len(valid_elements_list)
                    valid_elements_list.append(element)
                    element_part_map_list.append({
                        'element_idx': current_element_index, 
                        'part_id': part.id,
                        'material_id': part.material_id,
                    })
                    break 
        
        accepted_triangle_count = len(valid_elements_list)
        try:
            compact_nodes, compact_elements, compact_part_map = self._compact_mesh_nodes_and_elements(
                self.global_nodes,
                valid_elements_list,
                element_part_map_list,
            )
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
            if not preview_only:
                self.stageAdvanceRequested.emit(ProjectStage.JOB)
            return False

        self.global_nodes = compact_nodes
        self.global_elements = compact_elements
        self.element_part_map = compact_part_map
        orphan_count = max(0, int(raw_particle_count) - int(len(compact_nodes)))
        outside_rejected_count = max(
            0,
            int(total_elements) - int(quality_rejected_count) - int(accepted_triangle_count),
        )
        self._store_mesh_validation(
            {
                "particle_count": int(len(compact_nodes)),
                "triangle_count": int(len(compact_elements)),
                "orphan_count": int(orphan_count),
                "rejected_triangle_count": int(quality_rejected_count + outside_rejected_count),
                "near_zero_area_rejected": int(near_zero_area_rejected),
                "quality_rejected_count": int(quality_rejected_count),
                "outside_rejected_count": int(outside_rejected_count),
                "raw_particle_count": int(raw_particle_count),
                "raw_triangle_count": int(total_elements),
            }
        )
        self.last_mesh_dx = dx
        self.last_mesh_target_nodes = target_nodes
        self._mark_mesh_current(
            {
                "mode": "dx" if dx is not None else "count",
                "dx": dx,
                "target_nodes": target_nodes,
                "distribution": self.mesh_distribution,
                "backend": mesh_backend,
            }
        )

        mesh_end = time.perf_counter()
        total_time = max(0.0, mesh_end - mesh_start)
        if len(self.global_nodes) > 0 and total_time > 0:
            self._mesh_time_history.append((len(self.global_nodes), total_time))
            if len(self._mesh_time_history) > 5:
                self._mesh_time_history = self._mesh_time_history[-5:]

        if len(self.global_elements) == 0:
            QMessageBox.warning(self, "Error", "Connection generation did not generate any connections within the part boundaries.")
            if not preview_only:
                self.stageAdvanceRequested.emit(ProjectStage.JOB)
            return False

        if self.project_mode == "3d":
            if not self.build_3d_mesh():
                QMessageBox.warning(
                    self,
                    "Extrude Error",
                    "Failed to build 3D connections from the 2D surface connections.",
                )
                return False

        if not preview_only:
            mesh_msg = (
                f"Particles generated from geometry.\nTotal Particles: {len(self.global_nodes)}\n"
                f"Internal Connections: {len(self.global_elements)}"
            )
            if target_nodes is not None:
                mesh_msg += f"\nTarget Particles: {target_nodes}"
            spacing_stats = self._mesh_spacing_stats(self.global_nodes)
            if spacing_stats:
                min_d, avg_d, max_d = spacing_stats
                mesh_msg += (
                    f"\nNearest spacing (approx): "
                    f"{min_d:.3f}/{avg_d:.3f}/{max_d:.3f} {self.current_unit}"
                )
            QMessageBox.information(
                self,
                "Particles Ready",
                mesh_msg,
            )
            self._show_preview_window()
            # Keep user in Mesh stage so the preview is immediately visible,
            # then auto-advance to Job after a short delay.
            self.stageAdvanceRequested.emit(ProjectStage.MESH)
            QTimer.singleShot(3000, lambda: self.stageAdvanceRequested.emit(ProjectStage.JOB))
        return True

    def _resolve_mesh_config(self, mesh_config=None):
        dx = None
        target_nodes = None
        mesh_distribution = "global_poisson"
        mesh_backend = getattr(self, "mesh_backend", "auto")
        sizing_policy = None
        if mesh_config:
            mode = str(mesh_config.get("mode", "dx")).lower()
            if mode in ("dx", "spacing"):
                try:
                    dx = float(mesh_config.get("dx", DEFAULT_DX))
                except (TypeError, ValueError):
                    raise ValueError("Invalid spacing (dx).")
            elif mode in ("count", "nodes", "total"):
                try:
                    target_nodes = int(mesh_config.get("target_nodes", 1000))
                except (TypeError, ValueError):
                    raise ValueError("Invalid target particle count.")
            else:
                raise ValueError("Invalid particle connection sizing mode.")
            mesh_distribution = "global_poisson"
            mesh_backend = str(mesh_config.get("backend", mesh_backend)).lower()
            sizing_payload = mesh_config.get("sizing")
            if sizing_payload is not None:
                sizing_policy = MeshSizingPolicy.from_dict(sizing_payload)
            if mesh_backend == "gmsh-2d-adaptive" and sizing_policy is None and dx is not None:
                # Auto-build a default sizing policy from dx so the new path is callable
                # from places that don't yet expose the three sizing inputs.
                try:
                    sizing_policy = MeshSizingPolicy(h_bulk=float(dx))
                except Exception:
                    sizing_policy = None
            if sizing_policy is not None:
                mesh_distribution = "gmsh_2d_adaptive"
            # Stash any Results-page custom zones for the gmsh worker to consume.
            zones_payload = mesh_config.get("custom_zones") or []
            zones_obj = []
            try:
                from models import CustomMeshZone as _CMZ
                for z in zones_payload:
                    if isinstance(z, _CMZ):
                        zones_obj.append(z)
                    else:
                        zone = _CMZ.from_dict(z)
                        if zone is not None and getattr(zone, "points", None):
                            zones_obj.append(zone)
            except Exception:
                zones_obj = []
            self._pending_custom_zones = zones_obj

            # Same plumbing for Mesh-stage edge seeds.
            seeds_payload = mesh_config.get("edge_seeds") or []
            seeds_obj = []
            try:
                from models import EdgeSeed as _ES
                for s in seeds_payload:
                    if isinstance(s, _ES):
                        seeds_obj.append(s)
                    else:
                        seed = _ES.from_dict(s)
                        if seed is not None and seed.is_valid():
                            seeds_obj.append(seed)
            except Exception:
                seeds_obj = []
            self._pending_edge_seeds = seeds_obj

            # Vertex seeds (Mesh-stage point-anchored refinement).
            vseeds_payload = mesh_config.get("vertex_seeds") or []
            vseeds_obj = []
            try:
                from models import VertexSeed as _VS
                for s in vseeds_payload:
                    if isinstance(s, _VS):
                        vseeds_obj.append(s)
                    else:
                        seed = _VS.from_dict(s)
                        if seed is not None and seed.is_valid():
                            vseeds_obj.append(seed)
            except Exception:
                vseeds_obj = []
            self._pending_vertex_seeds = vseeds_obj

            # Boundary-layer seeds (CFD-style inflation).
            blseeds_payload = mesh_config.get("boundary_layer_seeds") or []
            blseeds_obj = []
            try:
                from models import BoundaryLayerSeed as _BLS
                for s in blseeds_payload:
                    if isinstance(s, _BLS):
                        blseeds_obj.append(s)
                    else:
                        seed = _BLS.from_dict(s)
                        if seed is not None and seed.is_valid():
                            blseeds_obj.append(seed)
            except Exception:
                blseeds_obj = []
            self._pending_boundary_layer_seeds = blseeds_obj

            # Matched edge pairs (gmsh setPeriodic).
            mp_payload = mesh_config.get("matched_edge_pairs") or []
            mp_obj = []
            try:
                from models import MatchedEdgePair as _MEP
                for p in mp_payload:
                    if isinstance(p, _MEP):
                        mp_obj.append(p)
                    else:
                        pair = _MEP.from_dict(p)
                        if pair is not None and pair.is_valid():
                            mp_obj.append(pair)
            except Exception:
                mp_obj = []
            self._pending_matched_edge_pairs = mp_obj

            # Per-part h_bulk / h_feature overrides.
            overrides_payload = mesh_config.get("part_mesh_overrides") or {}
            overrides_obj = {}
            if isinstance(overrides_payload, dict):
                for k, v in overrides_payload.items():
                    if not isinstance(v, dict):
                        continue
                    try:
                        pid = int(k)
                    except Exception:
                        continue
                    sub = {}
                    for key in ("h_bulk", "h_feature"):
                        try:
                            val = float(v.get(key, 0.0) or 0.0)
                        except Exception:
                            val = 0.0
                        if val > 0:
                            sub[key] = val
                    if sub:
                        overrides_obj[pid] = sub
            self._pending_part_mesh_overrides = overrides_obj
        else:
            mesh_modes = ["By spacing (dx)", "By total particle count"]
            mesh_mode, ok = QInputDialog.getItem(
                self,
                "Particle Connections",
                "Choose sizing method:",
                mesh_modes,
                0,
                False,
            )
            if not ok:
                raise RuntimeError("Connection generation canceled")
            if mesh_mode == "By spacing (dx)":
                dx, ok = QInputDialog.getDouble(
                    self,
                    "Particle Connections",
                    f"Spacing dx ({self.current_unit}):",
                    DEFAULT_DX,
                    0.1,
                    5000,
                    2,
                )
                if not ok:
                    raise RuntimeError("Connection generation canceled")
            else:
                target_nodes, ok = QInputDialog.getInt(
                    self,
                    "Particle Connections",
                    "Target total particles:",
                    1000,
                    10,
                    1000000,
                    10,
                )
                if not ok:
                    raise RuntimeError("Connection generation canceled")
            backend_labels = [
                "Auto (fastest available)",
                "Triangle (fastest 2D)",
                "Gmsh",
                "CGAL/pygalmesh",
                "SciPy Delaunay (legacy)",
            ]
            backend_choice, ok = QInputDialog.getItem(
                self,
                "Particle Connections",
                "Choose triangulation backend:",
                backend_labels,
                0,
                False,
            )
            if not ok:
                raise RuntimeError("Connection generation canceled")
            label = backend_choice.lower()
            if "triangle" in label:
                mesh_backend = "triangle"
            elif "gmsh" in label:
                mesh_backend = "gmsh"
            elif "cgal" in label or "pygalmesh" in label:
                mesh_backend = "pygalmesh"
            elif "scipy" in label:
                mesh_backend = "scipy"
            else:
                mesh_backend = "auto"
        mesh_backend = self._select_mesh_backend(mesh_backend, dx, target_nodes, mesh_distribution)
        self.mesh_backend = mesh_backend
        self.mesh_sizing_policy = sizing_policy
        return dx, target_nodes, mesh_distribution, mesh_backend

    def _mesh_parts_signature(self):
        signatures = []
        for part in sorted(self.parts, key=lambda p: str(getattr(p, "id", ""))):
            geom = getattr(part, "geometry", None)
            if geom is None or geom.is_empty:
                geom_hash = "empty"
            else:
                try:
                    geom_hash = hashlib.sha1(geom.wkb).hexdigest()
                except Exception:
                    geom_hash = f"area:{float(getattr(geom, 'area', 0.0)):.6f}"
            signatures.append(
                (
                    str(getattr(part, "id", "")),
                    getattr(part, "material_id", None),
                    bool(getattr(part, "is_rigid", False)),
                    geom_hash,
                )
            )
        return tuple(signatures)

    def _build_mesh_cache_key(self, dx, target_nodes, mesh_distribution, mesh_backend, parts_signature, particle_only=False):
        extrude_height = float(getattr(self, "extrude_height", 0.0) or 0.0)
        extrude_layers = int(getattr(self, "extrude_layers", 1) or 1)
        sizing_fp = None
        sp = getattr(self, "mesh_sizing_policy", None)
        if sp is not None:
            sizing_fp = (
                round(float(sp.h_bulk), 8),
                round(float(sp.h_feature), 8),
                round(float(sp.transition_width), 8),
            )
        # Include custom zones in the cache key so re-meshing with different
        # zones doesn't hit the previous result.
        zones_fp = None
        zones = list(getattr(self, "_pending_custom_zones", None) or [])
        if zones:
            zones_fp = tuple(
                (
                    int(getattr(z, "approx_node_count", 0) or 0),
                    tuple((round(float(x), 6), round(float(y), 6)) for (x, y) in getattr(z, "points", []) or []),
                )
                for z in zones
            )
        # Per-part overrides also belong in the cache key.
        overrides_fp = None
        overrides = dict(getattr(self, "_pending_part_mesh_overrides", None) or {})
        if overrides:
            overrides_fp = tuple(
                (
                    int(pid),
                    round(float(v.get("h_bulk", 0.0)), 8),
                    round(float(v.get("h_feature", 0.0)), 8),
                )
                for pid, v in sorted(overrides.items())
            )
        # Matched edge pairs also belong in the cache key.
        mp_fp = None
        mp = list(getattr(self, "_pending_matched_edge_pairs", None) or [])
        if mp:
            mp_fp = tuple(
                (
                    round(float(p.master["start"][0]), 6),
                    round(float(p.master["start"][1]), 6),
                    round(float(p.master["end"][0]), 6),
                    round(float(p.master["end"][1]), 6),
                    round(float(p.slave["start"][0]), 6),
                    round(float(p.slave["start"][1]), 6),
                    round(float(p.slave["end"][0]), 6),
                    round(float(p.slave["end"][1]), 6),
                )
                for p in mp
            )
        # Boundary-layer seeds also belong in the cache key.
        blseeds_fp = None
        blseeds = list(getattr(self, "_pending_boundary_layer_seeds", None) or [])
        if blseeds:
            blseeds_fp = tuple(
                (
                    round(float(s.first_layer_size), 8),
                    round(float(s.growth_ratio), 6),
                    int(s.num_layers),
                    bool(s.quads),
                    round(float(s.max_thickness), 8),
                    tuple(
                        (
                            int(r.get("part_id", 0)),
                            round(float(r["start"][0]), 6),
                            round(float(r["start"][1]), 6),
                            round(float(r["end"][0]), 6),
                            round(float(r["end"][1]), 6),
                        )
                        for r in (s.edge_refs or [])
                    ),
                )
                for s in blseeds
            )
        # Vertex seeds also belong in the cache key.
        vseeds_fp = None
        vseeds = list(getattr(self, "_pending_vertex_seeds", None) or [])
        if vseeds:
            vseeds_fp = tuple(
                (
                    round(float(s.point[0]), 6),
                    round(float(s.point[1]), 6),
                    round(float(s.target_size), 8),
                    round(float(s.influence_radius), 8),
                    int(getattr(s, "part_id", 0) or 0),
                )
                for s in vseeds
            )
        # Same for edge seeds — different seeds must invalidate cache.
        seeds_fp = None
        seeds = list(getattr(self, "_pending_edge_seeds", None) or [])
        if seeds:
            seeds_fp = tuple(
                (
                    str(getattr(s, "method", "")),
                    str(getattr(s, "bias", "")),
                    bool(getattr(s, "flip_bias", False)),
                    round(float(getattr(s, "element_size", 0.0)), 8),
                    round(float(getattr(s, "min_size", 0.0)), 8),
                    round(float(getattr(s, "max_size", 0.0)), 8),
                    int(getattr(s, "seed_count", 0)),
                    round(float(getattr(s, "bias_ratio", 1.0)), 6),
                    tuple(
                        (
                            int(r.get("part_id", 0)),
                            round(float(r["start"][0]), 6),
                            round(float(r["start"][1]), 6),
                            round(float(r["end"][0]), 6),
                            round(float(r["end"][1]), 6),
                        )
                        for r in getattr(s, "edge_refs", []) or []
                    ),
                )
                for s in seeds
            )
        return (
            None if dx is None else round(float(dx), 8),
            target_nodes,
            mesh_distribution,
            mesh_backend,
            parts_signature,
            bool(particle_only),
            self.project_mode,
            extrude_height if self.project_mode == "3d" else None,
            extrude_layers if self.project_mode == "3d" else None,
            sizing_fp,
            zones_fp,
            seeds_fp,
            vseeds_fp,
            blseeds_fp,
            mp_fp,
            overrides_fp,
        )

    def _copy_cached_mesh_result(self, cached_result):
        if not cached_result:
            return None
        return copy.deepcopy(cached_result)

    def _extract_part_loops_for_gmsh(self, parts_to_mesh):
        """Pull (outer_loop, inner_loops) per part from Shapely effective geometry.

        A part with a MultiPolygon contributes one entry per polygon piece,
        all sharing the original part.id, so the gmsh mesher can group them
        into a single physical group downstream.
        """
        from gmsh_mesher import GmshPartSpec

        specs = []
        for part in parts_to_mesh:
            geom = getattr(part, "geometry", None)
            if geom is None or geom.is_empty:
                continue
            effective = geom
            for child in self.get_child_parts(part):
                cgeom = getattr(child, "geometry", None)
                if cgeom is None or cgeom.is_empty:
                    continue
                try:
                    effective = effective.difference(cgeom).buffer(0)
                except Exception:
                    pass
            if effective is None or getattr(effective, "is_empty", True):
                continue
            polys = [effective] if isinstance(effective, Polygon) else list(getattr(effective, "geoms", []))
            for poly in polys:
                if not isinstance(poly, Polygon) or poly.is_empty:
                    continue
                outer = [(float(x), float(y)) for x, y, *_ in poly.exterior.coords]
                inners = []
                for ring in poly.interiors:
                    inner = [(float(x), float(y)) for x, y, *_ in ring.coords]
                    if len(inner) >= 3:
                        inners.append(inner)
                if len(outer) < 3:
                    continue
                specs.append(
                    GmshPartSpec(
                        id=int(part.id),
                        name=str(getattr(part, "name", "") or f"part_{part.id}"),
                        material_id=getattr(part, "material_id", None),
                        outer_loop=outer,
                        inner_loops=inners,
                    )
                )
        return specs

    def _compute_mesh_via_gmsh_2d(
        self,
        parts_to_mesh,
        sizing,
        dx,
        target_nodes,
        mesh_distribution,
        mesh_backend,
        cache_key,
        parts_signature,
        progress_cb=None,
        cancel_cb=None,
    ):
        from gmsh_mesher import generate_surface_mesh
        from mesh_utils import part_tagged_triangles_to_element_map

        def _check_cancel():
            if cancel_cb and cancel_cb():
                raise RuntimeError("Connection generation canceled")

        if progress_cb:
            progress_cb(15, "Extracting part loops")
        specs = self._extract_part_loops_for_gmsh(parts_to_mesh)
        if not specs:
            raise RuntimeError("No valid part geometry available for the gmsh 2D mesher.")
        _check_cancel()

        if progress_cb:
            progress_cb(35, "Running gmsh 2D mesher")
        custom_zones = list(getattr(self, "_pending_custom_zones", None) or [])
        edge_seeds = list(getattr(self, "_pending_edge_seeds", None) or [])
        vertex_seeds = list(getattr(self, "_pending_vertex_seeds", None) or [])
        boundary_layer_seeds = list(getattr(self, "_pending_boundary_layer_seeds", None) or [])
        matched_pairs = list(getattr(self, "_pending_matched_edge_pairs", None) or [])
        part_overrides = dict(getattr(self, "_pending_part_mesh_overrides", None) or {})
        mesh = generate_surface_mesh(
            specs,
            sizing,
            custom_zones=custom_zones,
            edge_seeds=edge_seeds,
            part_mesh_overrides=part_overrides,
            vertex_seeds=vertex_seeds,
            boundary_layer_seeds=boundary_layer_seeds,
            matched_edge_pairs=matched_pairs,
        )
        _check_cancel()

        if progress_cb:
            progress_cb(80, "Mapping part_id to triangles")
        part_material_lookup = {
            int(p.id): getattr(p, "material_id", None)
            for p in parts_to_mesh
            if getattr(p, "id", None) is not None
        }
        element_part_map = part_tagged_triangles_to_element_map(
            mesh["triangle_part_ids"], part_material_lookup
        )

        nodes = np.asarray(mesh["nodes"], dtype=float)
        elements = np.asarray(mesh["triangles"], dtype=int)

        mesh_validation = {
            "particle_count": int(len(nodes)),
            "triangle_count": int(len(elements)),
            "orphan_count": 0,
            "rejected_triangle_count": 0,
            "near_zero_area_rejected": 0,
            "quality_rejected_count": 0,
            "outside_rejected_count": 0,
            "raw_particle_count": int(len(nodes)),
            "raw_triangle_count": int(len(elements)),
            "interface_curve_count": int(len(mesh.get("interface_curves", []))),
            "hole_curve_count": int(len(mesh.get("hole_curves", []))),
        }

        result = {
            "nodes": nodes,
            "elements": elements,
            "element_part_map": element_part_map,
            "mesh_distribution": mesh_distribution,
            "dx": dx,
            "target_nodes": target_nodes,
            "mesh_geometry_signature": self._current_mesh_geometry_signature(),
            "mesh_config_signature": self._mesh_config_signature(
                {
                    "mode": "dx" if dx is not None else "count",
                    "dx": dx,
                    "target_nodes": target_nodes,
                    "distribution": mesh_distribution,
                    "backend": mesh_backend,
                    "sizing": sizing.to_dict(),
                }
            ),
            "mesh_validation": mesh_validation,
        }
        self._mesh_cache = {
            "key": cache_key,
            "parts_signature": parts_signature,
            "result": copy.deepcopy(result),
        }
        if progress_cb:
            progress_cb(100, "Done")
        return result

    def _compute_mesh_data_worker(
        self,
        dx,
        target_nodes,
        mesh_distribution,
        mesh_backend,
        particle_only=False,
        existing_nodes=None,
        progress_cb=None,
        cancel_cb=None,
    ):
        task_noun = "Particle generation" if particle_only else "Connection generation"

        def _check_cancel():
            if cancel_cb and cancel_cb():
                raise RuntimeError(f"{task_noun} canceled")

        parts_signature = self._mesh_parts_signature()
        cache_key = self._build_mesh_cache_key(
            dx,
            target_nodes,
            mesh_distribution,
            mesh_backend,
            parts_signature,
            particle_only=particle_only,
        )
        cache = getattr(self, "_mesh_cache", None)
        if (
            cache
            and cache.get("key") == cache_key
            and cache.get("parts_signature") == parts_signature
            and cache.get("result") is not None
        ):
            if progress_cb:
                progress_cb(100, "Using cached particles" if particle_only else "Using cached connections")
            return self._copy_cached_mesh_result(cache["result"])

        reuse_nodes = None
        if not particle_only and existing_nodes is not None:
            try:
                candidate = np.asarray(existing_nodes, dtype=float)
            except Exception:
                candidate = None
            if candidate is not None and candidate.ndim == 2 and candidate.shape[1] >= 2 and len(candidate) >= 3:
                reuse_nodes = candidate

        _check_cancel()
        if not self.parts:
            raise ValueError("No parts available for particle generation.")

        parts_to_mesh = [
            p for p in self.parts
            if not p.is_void and (particle_only or p.material_id is not None)
        ]
        if not parts_to_mesh:
            if particle_only:
                raise ValueError("No solid parts available for particle generation.")
            raise ValueError("No parts with assigned material available for particle generation.")

        # Heterogeneous multi-part 2D path with adaptive sizing.
        sizing_policy = getattr(self, "mesh_sizing_policy", None)
        if (
            not particle_only
            and mesh_backend == "gmsh-2d-adaptive"
            and sizing_policy is not None
            and self.project_mode != "3d"
        ):
            return self._compute_mesh_via_gmsh_2d(
                parts_to_mesh,
                sizing_policy,
                dx,
                target_nodes,
                mesh_distribution,
                mesh_backend,
                cache_key,
                parts_signature,
                progress_cb=progress_cb,
                cancel_cb=cancel_cb,
            )

        if self.project_mode == "3d":
            height = float(self.extrude_height)
            layers = int(self.extrude_layers)
            if height <= 0.0 or layers < 1:
                raise ValueError("Extrude height and layers must be positive for 3D connection preview.")

        if progress_cb:
            progress_cb(5, "Collecting boundaries...")

        all_boundaries = []
        part_effective_geoms = {}
        rigid_part_ids = set()
        total_area = 0.0

        def _append_unique(points_list, points_set, pt):
            if pt in points_set:
                return
            points_set.add(pt)
            points_list.append(pt)

        def _add_ring_samples(ring, spacing, points_list, points_set, min_points=12):
            samples = self._sample_authoritative_boundary_ring_points(
                ring,
                spacing,
                min_points=min_points,
            )
            out = []
            for pt in samples:
                p = tuple(pt)
                _append_unique(points_list, points_set, p)
                out.append(p)
            return out

        def _add_ring_vertices(
            ring,
            points_list,
            points_set,
            protected_list=None,
            protected_set=None,
            corner_angle_deg=15.0,
        ):
            if ring is None:
                return
            try:
                coords = list(ring.coords)
            except Exception:
                return
            if not coords:
                return
            pts = [tuple(pt) for pt in coords]
            if len(pts) > 1 and dist(pts[0], pts[-1]) <= 1e-9:
                pts = pts[:-1]
            if not pts:
                return

            closed = len(pts) >= 3
            angle_thresh = math.radians(max(0.0, float(corner_angle_deg)))

            def _mark(pt_tuple):
                _append_unique(points_list, points_set, pt_tuple)
                if protected_list is not None and protected_set is not None and pt_tuple not in protected_set:
                    protected_set.add(pt_tuple)
                    protected_list.append(pt_tuple)

            if not closed:
                _mark(pts[0])
                if len(pts) > 1:
                    _mark(pts[-1])
                return

            keep = []
            n = len(pts)
            for i in range(n):
                prev_pt = pts[(i - 1) % n]
                pt = pts[i]
                next_pt = pts[(i + 1) % n]
                v1 = (pt[0] - prev_pt[0], pt[1] - prev_pt[1])
                v2 = (next_pt[0] - pt[0], next_pt[1] - pt[1])
                l1 = math.hypot(v1[0], v1[1])
                l2 = math.hypot(v2[0], v2[1])
                if l1 <= 1e-9 or l2 <= 1e-9:
                    keep.append(pt)
                    continue
                dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)
                dot = max(-1.0, min(1.0, dot))
                turn = math.acos(dot)
                if turn >= angle_thresh:
                    keep.append(pt)

            if not keep and pts:
                keep = [pts[0]]
            for pt in keep:
                _mark(pt)

        def _filter_points_min_spacing(boundary_pts, interior_pts, min_spacing, protected=None):
            if not boundary_pts and not interior_pts:
                return []
            cell = max(min_spacing, 1e-9)
            min_sq = min_spacing * min_spacing
            grid = {}
            kept = []
            protected = protected or set()

            def _grid_key(p):
                return (int(math.floor(p[0] / cell)), int(math.floor(p[1] / cell)))

            def _can_place(p):
                gx, gy = _grid_key(p)
                for i in range(gx - 1, gx + 2):
                    for j in range(gy - 1, gy + 2):
                        for q in grid.get((i, j), []):
                            dx = p[0] - q[0]
                            dy = p[1] - q[1]
                            if dx * dx + dy * dy < min_sq:
                                return False
                return True

            for p in boundary_pts:
                if p in protected or _can_place(p):
                    kept.append(p)
                    grid.setdefault(_grid_key(p), []).append(p)
            for p in interior_pts:
                if _can_place(p):
                    kept.append(p)
                    grid.setdefault(_grid_key(p), []).append(p)
            return kept

        for part in parts_to_mesh:
            mat_type = (part.material_type or "").lower()
            is_rigid_part = part.is_rigid or mat_type in ("rigid", "rigid_body")
            if is_rigid_part:
                rigid_part_ids.add(part.id)

            effective_geom = part.geometry
            if effective_geom is None or effective_geom.is_empty:
                continue

            children = self.get_child_parts(part)
            for child in children:
                if child.geometry and not child.geometry.is_empty:
                    try:
                        effective_geom = effective_geom.difference(child.geometry).buffer(0)
                    except Exception:
                        pass

            if not effective_geom or effective_geom.is_empty:
                continue

            part_effective_geoms[part.id] = effective_geom
            try:
                total_area += float(effective_geom.area)
            except Exception:
                pass

            geoms = [effective_geom] if isinstance(effective_geom, Polygon) else list(effective_geom.geoms)
            for g in geoms:
                all_boundaries.append(g.exterior)
                for interior in g.interiors:
                    all_boundaries.append(interior)

        _check_cancel()
        if target_nodes is not None:
            est_nodes = int(target_nodes)
        else:
            est_nodes = self._estimate_nodes_from_area(total_area, dx, mesh_distribution)
        if est_nodes is not None and est_nodes > MESH_NODE_HARD_LIMIT:
            raise ValueError(
                f"Estimated particle count {est_nodes:,} exceeds the hard limit of "
                f"{MESH_NODE_HARD_LIMIT:,}. Increase dx or lower the target particles."
            )
        boundary_union = unary_union(all_boundaries) if all_boundaries else None
        geoms_to_sample = []
        if boundary_union is not None:
            if boundary_union.geom_type == "LineString":
                geoms_to_sample.append(boundary_union)
            elif boundary_union.geom_type in ("MultiLineString", "GeometryCollection"):
                geoms_to_sample.extend(list(boundary_union.geoms))

        lattice_origin = None
        if mesh_distribution in ("square", "global_square"):
            bounds = [
                geom.bounds for geom in part_effective_geoms.values()
                if geom is not None and not geom.is_empty
            ]
            if bounds:
                minx = min(b[0] for b in bounds)
                miny = min(b[1] for b in bounds)
                lattice_origin = (minx, miny)

        union_geom = None
        if mesh_distribution.startswith("global"):
            geoms = [g for g in part_effective_geoms.values() if g is not None and not g.is_empty]
            if geoms:
                try:
                    union_geom = unary_union(geoms)
                    if union_geom is not None and not union_geom.is_empty:
                        try:
                            union_geom = union_geom.buffer(0)
                        except Exception:
                            pass
                except Exception:
                    union_geom = None

        triangulation_constraints = {
            "segments": None,
            "holes": None,
            "regions": None,
            "region_attr_map": {},
            "spacing": None,
        }

        def _collect_mesh_points(spacing):
            triangulation_constraints["segments"] = None
            triangulation_constraints["holes"] = None
            triangulation_constraints["regions"] = None
            triangulation_constraints["region_attr_map"] = {}
            try:
                spacing = float(spacing)
                triangulation_constraints["spacing"] = float(spacing)
            except Exception:
                triangulation_constraints["spacing"] = None
                return []

            min_spacing_factor = max(0.1, float(getattr(self, "mesh_min_spacing_factor", MESH_MIN_SPACING_FACTOR)))
            min_spacing = max(1e-9, spacing * min_spacing_factor)
            boundary_points = []
            boundary_set = set()
            boundary_vertices = []
            boundary_vertices_set = set()
            interior_points = []
            interior_set = set()
            layer_boundaries = []
            constraint_chains = []
            boundary_thickness = max(0.0, float(getattr(self, "mesh_boundary_thickness", 0.0)))
            boundary_spacing_factor = max(
                0.1, min(1.0, float(getattr(self, "mesh_boundary_spacing_factor", 1.0)))
            )
            boundary_spacing = max(1e-9, spacing * boundary_spacing_factor)
            layer_spacing = max(1e-9, spacing * boundary_spacing_factor)
            uniform_boundary = boundary_spacing_factor >= 0.99
            boundary_min_points = 1 if uniform_boundary else 16
            interface_payload = self._build_interface_mesh_sampling_payload(part_effective_geoms, spacing)
            skip_boundary_specs = list(interface_payload.get("skip_boundary_specs", []) or [])
            circle_fill_specs = list(interface_payload.get("circle_fill_specs", []) or [])

            def _skip_generic_boundary_ring(ring):
                for spec in skip_boundary_specs:
                    try:
                        if self._ring_matches_skip_spec(ring, spec):
                            return True
                    except Exception:
                        continue
                return False

            def _append_constraint_chain(points_seq, closed=False):
                try:
                    seq = [tuple(map(float, p[:2])) for p in (points_seq if points_seq is not None else [])]
                except Exception:
                    seq = []
                if len(seq) < 2:
                    return
                compact = []
                for p in seq:
                    if compact and compact[-1] == p:
                        continue
                    compact.append(p)
                if len(compact) < 2:
                    return
                if closed and len(compact) > 2 and compact[0] == compact[-1]:
                    compact = compact[:-1]
                if len(compact) < 2:
                    return
                constraint_chains.append({"points": compact, "closed": bool(closed)})

            def _add_lattice_points(geom):
                if geom is None or getattr(geom, "is_empty", True):
                    return
                try:
                    geoms = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
                except Exception:
                    geoms = [geom]
                for g in geoms:
                    if g is None or getattr(g, "is_empty", True):
                        continue
                    if not isinstance(g, Polygon):
                        continue
                    try:
                        lattice_pts = square_lattice_sample(g, spacing, origin=lattice_origin)
                    except Exception:
                        lattice_pts = []
                    for pt in lattice_pts:
                        _append_unique(interior_points, interior_set, tuple(pt))

            if boundary_thickness > 1e-9:
                num_layers = max(1, int(math.ceil(boundary_thickness / layer_spacing)))
                step = boundary_thickness / num_layers if num_layers > 0 else boundary_thickness
                for part_id, effective_geom in part_effective_geoms.items():
                    if effective_geom is None or effective_geom.is_empty:
                        continue
                    for i in range(1, num_layers + 1):
                        offset = step * i
                        try:
                            inner = effective_geom.buffer(-offset)
                        except Exception:
                            continue
                        if inner.is_empty:
                            break
                        geoms = [inner] if isinstance(inner, Polygon) else list(inner.geoms)
                        for g in geoms:
                            layer_boundaries.append(g.exterior)
                            for interior in g.interiors:
                                layer_boundaries.append(interior)

            for ring in all_boundaries:
                if _skip_generic_boundary_ring(ring):
                    continue
                _add_ring_vertices(
                    ring,
                    boundary_points,
                    boundary_set,
                    boundary_vertices,
                    boundary_vertices_set,
                    corner_angle_deg=15.0,
                )
                ring_chain = _add_ring_samples(
                    ring,
                    boundary_spacing,
                    boundary_points,
                    boundary_set,
                    min_points=boundary_min_points,
                )
                _append_constraint_chain(ring_chain, closed=True)
            for ring in layer_boundaries:
                _add_ring_vertices(
                    ring,
                    boundary_points,
                    boundary_set,
                    boundary_vertices,
                    boundary_vertices_set,
                    corner_angle_deg=15.0,
                )
                ring_chain = _add_ring_samples(
                    ring,
                    layer_spacing,
                    boundary_points,
                    boundary_set,
                    min_points=boundary_min_points,
                )
                _append_constraint_chain(ring_chain, closed=True)

            boundary_points, boundary_vertices, interior_points = self._merge_interface_sampling_payload_into_point_sets(
                interface_payload,
                boundary_points=boundary_points,
                boundary_set=boundary_set,
                boundary_vertices=boundary_vertices,
                boundary_vertices_set=boundary_vertices_set,
                interior_points=interior_points,
                interior_set=interior_set,
            )
            iface_constraint_chains = [c for c in list(interface_payload.get("constraint_chains", []) or []) if c]
            if iface_constraint_chains:
                constraint_chains = iface_constraint_chains + constraint_chains
            if constraint_chains:
                constraint_chains = self._dedupe_constraint_chains(constraint_chains, spacing)
            if constraint_chains:
                boundary_points, boundary_vertices = self._protect_constraint_chain_points(
                    constraint_chains,
                    boundary_points=boundary_points,
                    boundary_set=boundary_set,
                    boundary_vertices=boundary_vertices,
                    boundary_vertices_set=boundary_vertices_set,
                )

            if mesh_distribution == "global_square":
                lattice_geom = union_geom
                if lattice_geom is not None and not getattr(lattice_geom, "is_empty", True):
                    try:
                        eroded_geom = lattice_geom.buffer(-spacing * 0.5)
                    except Exception:
                        eroded_geom = None
                    if eroded_geom is not None and not getattr(eroded_geom, "is_empty", True):
                        lattice_geom = eroded_geom
                _add_lattice_points(lattice_geom)
                for spec in circle_fill_specs:
                    for pt in self._hex_lattice_sample_circle(
                        spec["center"],
                        spec["fill_radius"],
                        spacing,
                        angle=spec.get("phase", 0.0),
                    ):
                        _append_unique(interior_points, interior_set, tuple(pt))
            elif mesh_distribution == "square":
                for part_id, effective_geom in part_effective_geoms.items():
                    if effective_geom is None or effective_geom.is_empty:
                        continue
                    lattice_geom = effective_geom
                    if part_id in rigid_part_ids:
                        try:
                            boundary_band = effective_geom.boundary.buffer(spacing * 0.45)
                        except Exception:
                            boundary_band = None
                        if boundary_band is not None and not getattr(boundary_band, "is_empty", True):
                            try:
                                boundary_band = boundary_band.intersection(effective_geom)
                            except Exception:
                                pass
                            if not getattr(boundary_band, "is_empty", True):
                                lattice_geom = boundary_band
                    else:
                        try:
                            eroded_geom = effective_geom.buffer(-spacing * 0.5)
                        except Exception:
                            eroded_geom = None
                        if eroded_geom is not None and not getattr(eroded_geom, "is_empty", True):
                            lattice_geom = eroded_geom
                    _add_lattice_points(lattice_geom)
            elif mesh_distribution == "global_poisson":
                if union_geom is not None and not union_geom.is_empty:
                    eroded_geom = union_geom.buffer(-spacing * 0.5)
                    if eroded_geom is not None and not eroded_geom.is_empty and circle_fill_specs:
                        for spec in circle_fill_specs:
                            try:
                                fill_disk = Point(spec["center"]).buffer(float(spec["fill_radius"]), resolution=64)
                                eroded_geom = eroded_geom.difference(fill_disk)
                            except Exception:
                                continue
                    if not eroded_geom.is_empty:
                        geoms = [eroded_geom] if isinstance(eroded_geom, Polygon) else list(eroded_geom.geoms)
                        for g in geoms:
                            interior_pts = poisson_sample(g, spacing)
                            for pt in interior_pts:
                                _append_unique(interior_points, interior_set, tuple(pt))
                for spec in circle_fill_specs:
                    for pt in self._hex_lattice_sample_circle(
                        spec["center"],
                        spec["fill_radius"],
                        spacing,
                        angle=spec.get("phase", 0.0),
                    ):
                        _append_unique(interior_points, interior_set, tuple(pt))
            else:
                for part_id, effective_geom in part_effective_geoms.items():
                    if part_id in rigid_part_ids:
                        continue
                    circle_spec = None
                    for spec in circle_fill_specs:
                        if int(spec.get("part_id", -1)) == int(part_id):
                            circle_spec = spec
                            break
                    if circle_spec is not None:
                        for pt in self._hex_lattice_sample_circle(
                            circle_spec["center"],
                            circle_spec["fill_radius"],
                            spacing,
                            angle=circle_spec.get("phase", 0.0),
                        ):
                            _append_unique(interior_points, interior_set, tuple(pt))
                        continue
                    eroded_geom = effective_geom.buffer(-spacing * 0.5)
                    if not eroded_geom.is_empty:
                        geoms = [eroded_geom] if isinstance(eroded_geom, Polygon) else list(eroded_geom.geoms)
                        for g in geoms:
                            interior_pts = poisson_sample(g, spacing)
                            for pt in interior_pts:
                                _append_unique(interior_points, interior_set, tuple(pt))

            filtered_points = _filter_points_min_spacing(
                boundary_points,
                interior_points,
                min_spacing,
                protected=boundary_vertices_set,
            )
            if constraint_chains:
                triangulation_constraints["segments"] = self._build_triangle_pslg_segments(
                    filtered_points,
                    constraint_chains,
                    spacing,
                )
            region_payload = self._build_triangle_region_seed_payload(part_effective_geoms, parts_to_mesh)
            triangulation_constraints["holes"] = region_payload.get("holes")
            triangulation_constraints["regions"] = region_payload.get("regions")
            triangulation_constraints["region_attr_map"] = dict(region_payload.get("region_attr_map", {}) or {})
            return filtered_points

        if reuse_nodes is not None:
            if dx is None and target_nodes is not None:
                total_area = sum(
                    geom.area for pid, geom in part_effective_geoms.items()
                    if pid not in rigid_part_ids
                )
                if total_area <= 0:
                    total_area = sum(geom.area for geom in part_effective_geoms.values())
                if total_area > 0:
                    dx = math.sqrt(total_area / max(target_nodes, 1))
                    dx = max(0.1, min(5000, dx))
            if progress_cb:
                progress_cb(25, "Reusing existing particles")
            try:
                all_points = {
                    (float(p[0]), float(p[1])) for p in reuse_nodes
                    if len(p) >= 2 and math.isfinite(p[0]) and math.isfinite(p[1])
                }
            except Exception:
                all_points = set()
            all_points = list(all_points)
        elif target_nodes is not None:
            total_area = sum(
                geom.area for pid, geom in part_effective_geoms.items()
                if pid not in rigid_part_ids
            )
            if total_area <= 0:
                total_area = sum(geom.area for geom in part_effective_geoms.values())
            if total_area <= 0:
                raise ValueError("Could not estimate connection area.")
            dx = math.sqrt(total_area / max(target_nodes, 1))
            dx = max(0.1, min(5000, dx))
            max_iter = 4
            all_points = set()
            for i in range(max_iter):
                _check_cancel()
                all_points = _collect_mesh_points(dx)
                count = len(all_points)
                if count < 3:
                    break
                err_ratio = abs(count - target_nodes) / max(target_nodes, 1)
                if err_ratio <= 0.15 or i == max_iter - 1:
                    break
                dx *= (count / max(target_nodes, 1)) ** 0.5
                dx = max(0.1, min(5000, dx))
        else:
            all_points = _collect_mesh_points(dx)

        if len(all_points) < 3:
            raise ValueError("Not enough unique points to generate particles.")

        est = self._estimate_mesh_time(len(all_points))
        if progress_cb:
            if est is not None:
                progress_cb(30, f"Sampling points (~{est:.1f}s est)")
            else:
                progress_cb(30, "Sampling points")

        _check_cancel()
        global_nodes = np.array(list(all_points))
        raw_particle_count = len(global_nodes)

        if particle_only:
            if progress_cb:
                progress_cb(95, "Finalizing particles")
            mesh_validation = {
                "particle_count": int(raw_particle_count),
                "triangle_count": 0,
                "orphan_count": 0,
                "rejected_triangle_count": 0,
                "near_zero_area_rejected": 0,
                "quality_rejected_count": 0,
                "outside_rejected_count": 0,
                "raw_particle_count": int(raw_particle_count),
                "raw_triangle_count": 0,
            }
            result = {
                "nodes": np.asarray(global_nodes, dtype=float),
                "elements": np.empty((0, 3), dtype=int),
                "element_part_map": [],
                "mesh_distribution": mesh_distribution,
                "dx": dx,
                "target_nodes": target_nodes,
                "mesh_geometry_signature": self._current_mesh_geometry_signature(),
                "mesh_config_signature": self._mesh_config_signature(
                    {
                        "mode": "dx" if dx is not None else "count",
                        "dx": dx,
                        "target_nodes": target_nodes,
                        "distribution": mesh_distribution,
                        "backend": mesh_backend,
                    }
                ),
                "mesh_validation": mesh_validation,
            }
            self._mesh_cache = {
                "key": cache_key,
                "parts_signature": parts_signature,
                "result": copy.deepcopy(result),
            }
            return result

        if progress_cb:
            progress_cb(55, "Triangulating")
        _check_cancel()
        try:
            tri_segments = triangulation_constraints.get("segments")
            tri_holes = triangulation_constraints.get("holes")
            tri_regions = triangulation_constraints.get("regions")
            tri_payload = self._triangulate_points(
                global_nodes,
                mesh_backend,
                segments=tri_segments,
                holes=tri_holes,
                regions=tri_regions,
                return_metadata=True,
            )
            global_elements_all = np.asarray(tri_payload.get("triangles", np.empty((0, 3), dtype=int)), dtype=int)
            tri_region_attrs = tri_payload.get("triangle_attributes")
        except Exception as e:
            raise RuntimeError(str(e))

        if progress_cb:
            progress_cb(60, "Assigning connections")

        valid_elements_list = []
        element_part_map_list = []
        part_prepared = {}
        part_eroded_prepared = {}
        part_relaxed_prepared = {}
        part_bounds = {}
        edge_margin = 0.0
        if dx is not None:
            edge_margin = max(1e-6, float(dx) * 0.2)
        cover_tol = max(1e-8, float(dx) * 1e-3) if dx is not None else 1e-6
        min_area_tol = max(1e-10, float(dx * dx) * 0.01) if dx is not None else 1e-10
        for part in parts_to_mesh:
            effective_geom = part_effective_geoms.get(part.id)
            if not effective_geom:
                continue
            part_bounds[part.id] = effective_geom.bounds
            try:
                part_prepared[part.id] = prep(effective_geom)
            except Exception:
                part_prepared[part.id] = None
            try:
                relaxed_geom = effective_geom.buffer(cover_tol)
            except Exception:
                relaxed_geom = None
            if relaxed_geom is not None and not relaxed_geom.is_empty:
                try:
                    part_relaxed_prepared[part.id] = prep(relaxed_geom)
                except Exception:
                    pass
            if edge_margin > 0:
                try:
                    eroded = effective_geom.buffer(-edge_margin)
                except Exception:
                    eroded = None
                if eroded is not None and not eroded.is_empty:
                    try:
                        part_eroded_prepared[part.id] = prep(eroded)
                    except Exception:
                        pass

        def _bounds_contains(outer, inner):
            return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]

        total_elements = len(global_elements_all)
        quality_rejected_count = 0
        near_zero_area_rejected = 0
        step = max(500, total_elements // 50) if total_elements else 500
        region_attr_map = dict(triangulation_constraints.get("region_attr_map", {}) or {})
        authoritative_edge_set = set()
        if tri_segments is not None:
            try:
                for seg in np.asarray(tri_segments, dtype=int):
                    if len(seg) < 2:
                        continue
                    a, b = sorted((int(seg[0]), int(seg[1])))
                    authoritative_edge_set.add((a, b))
            except Exception:
                authoritative_edge_set = set()
        for i, element in enumerate(global_elements_all):
            if cancel_cb and cancel_cb():
                raise RuntimeError("Connection generation canceled")
            if progress_cb and total_elements and i % step == 0:
                pct = 60 + int(35 * (i / max(1, total_elements)))
                progress_cb(pct, f"Assigning connections {i}/{total_elements}")
            tri_pts = global_nodes[element]
            tri_edges = (
                tuple(sorted((int(element[0]), int(element[1])))),
                tuple(sorted((int(element[1]), int(element[2])))),
                tuple(sorted((int(element[2]), int(element[0])))),
            )
            touches_authoritative_boundary = any(edge in authoritative_edge_set for edge in tri_edges)
            quality = triangle_quality_metrics(tri_pts)
            if quality["area"] < float(min_area_tol):
                near_zero_area_rejected += 1
                quality_rejected_count += 1
                continue
            if (quality["min_angle_deg"] < 15.0 or quality["aspect_ratio"] > 5.0) and not touches_authoritative_boundary:
                quality_rejected_count += 1
                continue
            try:
                xs = tri_pts[:, 0]
                ys = tri_pts[:, 1]
                tri_bounds = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
            except Exception:
                tri_bounds = None
            tri_poly = None
            p_centroid = None
            region_part_info = None
            if tri_region_attrs is not None and i < len(tri_region_attrs):
                try:
                    region_key = int(round(float(tri_region_attrs[i])))
                except Exception:
                    region_key = None
                if region_key is not None:
                    region_part_info = dict(region_attr_map.get(region_key, {}) or {})
            if region_part_info:
                current_element_index = len(valid_elements_list)
                valid_elements_list.append(element)
                element_part_map_list.append(
                    {
                        "element_idx": current_element_index,
                        "part_id": region_part_info.get("part_id"),
                        "material_id": region_part_info.get("material_id"),
                    }
                )
                continue

            for part in parts_to_mesh:
                effective_geom = part_effective_geoms.get(part.id)
                if not effective_geom:
                    continue
                bounds = part_bounds.get(part.id)
                if tri_bounds is not None and bounds is not None:
                    if not _bounds_contains(bounds, tri_bounds):
                        continue
                inside = False
                eroded_prep = part_eroded_prepared.get(part.id)
                if eroded_prep is not None:
                    try:
                        if (
                            eroded_prep.covers(Point(tri_pts[0]))
                            and eroded_prep.covers(Point(tri_pts[1]))
                            and eroded_prep.covers(Point(tri_pts[2]))
                        ):
                            inside = True
                    except Exception:
                        inside = False
                if not inside:
                    prep_geom = part_prepared.get(part.id)
                    if prep_geom is not None:
                        if tri_poly is None:
                            try:
                                tri_poly = Polygon(tri_pts)
                            except Exception:
                                tri_poly = None
                        if tri_poly is not None and prep_geom.covers(tri_poly):
                            inside = True
                    if not inside:
                        relaxed_prep = part_relaxed_prepared.get(part.id)
                        if relaxed_prep is not None and tri_poly is not None:
                            try:
                                if relaxed_prep.covers(tri_poly):
                                    inside = True
                            except Exception:
                                pass
                    if not inside:
                        if p_centroid is None:
                            centroid = np.mean(tri_pts, axis=0)
                            p_centroid = Point(centroid)
                        if tri_poly is None:
                            if effective_geom.covers(p_centroid):
                                inside = True
                        else:
                            # Numeric fallback for boundary-touching triangles that should belong to
                            # this part but fail exact polygon covers() due precision at shared interfaces.
                            centroid_ok = False
                            try:
                                centroid_ok = bool(effective_geom.covers(p_centroid))
                            except Exception:
                                centroid_ok = False
                            if centroid_ok:
                                try:
                                    overlap_area = float(getattr(effective_geom.intersection(tri_poly), "area", 0.0) or 0.0)
                                except Exception:
                                    overlap_area = 0.0
                                try:
                                    tri_area = float(getattr(tri_poly, "area", 0.0) or 0.0)
                                except Exception:
                                    tri_area = 0.0
                                if tri_area > 1e-12 and (overlap_area / tri_area) >= 0.90:
                                    inside = True
                if inside:
                    current_element_index = len(valid_elements_list)
                    valid_elements_list.append(element)
                    element_part_map_list.append({
                        "element_idx": current_element_index,
                        "part_id": part.id,
                        "material_id": part.material_id,
                    })
                    break

        accepted_triangle_count = len(valid_elements_list)
        boundary_validation = self._validate_authoritative_segment_coverage(valid_elements_list, tri_segments)
        global_nodes, global_elements, element_part_map_list = self._compact_mesh_nodes_and_elements(
            global_nodes,
            valid_elements_list,
            element_part_map_list,
        )
        orphan_count = max(0, int(raw_particle_count) - int(len(global_nodes)))
        outside_rejected_count = max(
            0,
            int(total_elements) - int(quality_rejected_count) - int(accepted_triangle_count),
        )
        mesh_validation = {
            "particle_count": int(len(global_nodes)),
            "triangle_count": int(len(global_elements)),
            "orphan_count": int(orphan_count),
            "rejected_triangle_count": int(quality_rejected_count + outside_rejected_count),
            "near_zero_area_rejected": int(near_zero_area_rejected),
            "quality_rejected_count": int(quality_rejected_count),
            "outside_rejected_count": int(outside_rejected_count),
            "raw_particle_count": int(raw_particle_count),
            "raw_triangle_count": int(total_elements),
            **boundary_validation,
        }

        if progress_cb:
            progress_cb(95, "Finalizing")

        result = {
            "nodes": global_nodes,
            "elements": global_elements,
            "element_part_map": element_part_map_list,
            "mesh_distribution": mesh_distribution,
            "dx": dx,
            "target_nodes": target_nodes,
            "mesh_geometry_signature": self._current_mesh_geometry_signature(),
            "mesh_config_signature": self._mesh_config_signature(
                {
                    "mode": "dx" if dx is not None else "count",
                    "dx": dx,
                    "target_nodes": target_nodes,
                    "distribution": mesh_distribution,
                    "backend": mesh_backend,
                }
            ),
            "mesh_validation": mesh_validation,
        }
        self._mesh_cache = {
            "key": cache_key,
            "parts_signature": parts_signature,
            "result": copy.deepcopy(result),
        }
        return result

    def _apply_mesh_result(self, result):
        self.global_nodes = result.get("nodes", np.array([]))
        self.global_elements = result.get("elements", np.array([]))
        self.element_part_map = result.get("element_part_map", [])
        self.mesh_distribution = result.get("mesh_distribution", "global_poisson")
        self.last_mesh_dx = result.get("dx")
        self.last_mesh_target_nodes = result.get("target_nodes")
        store = self._mesh_store()
        store["geometry_signature"] = result.get("mesh_geometry_signature")
        store["config_signature"] = result.get("mesh_config_signature")
        self._store_mesh_validation(result.get("mesh_validation", {}))

    def run_cpd_async(self, preview_only=False, mesh_config=None, on_done=None):
        if self._mesh_thread and self._mesh_thread.isRunning():
            self._queue_mesh_status("Connection generation already running...")
            return False
        parts = [p for p in getattr(self, "parts", []) or [] if not getattr(p, "is_void", False)]
        if not parts:
            return False
        if not any(getattr(p, "geometry", None) is not None and not getattr(getattr(p, "geometry", None), "is_empty", True) for p in parts):
            return False
        if not any(getattr(p, "material_id", None) is not None for p in parts):
            return False
        try:
            dx, target_nodes, mesh_distribution, mesh_backend = self._resolve_mesh_config(mesh_config)
        except RuntimeError:
            return False
        except ValueError as exc:
            QMessageBox.warning(self, "Error", str(exc))
            return False
        dx, target_nodes, proceed = self._guard_mesh_density(dx, target_nodes, mesh_distribution)
        if not proceed:
            return False

        progress = QProgressDialog("Generating connections...", "Cancel", 0, 100, self.window() or self)
        progress.setWindowTitle("Generating Connections")
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setMinimumDuration(0)
        progress.show()
        self._mesh_progress_dialog = progress
        self._mesh_preview_only = bool(preview_only)
        self._mesh_on_done = on_done
        self._mesh_task_kind = "connections"
        self._mesh_last_progress_ts = 0.0
        self._mesh_status_pct = None

        existing_nodes = None
        if self.has_current_particle_set(mesh_config) and self.global_nodes is not None and len(self.global_nodes) > 0:
            existing_nodes = np.asarray(self.global_nodes, dtype=float)
        thread = QThread()
        worker = MeshWorker(
            self._compute_mesh_data_worker,
            dx,
            target_nodes,
            mesh_distribution,
            mesh_backend,
            False,
            existing_nodes,
        )
        worker.moveToThread(thread)

        progress.canceled.connect(worker.request_cancel)
        worker.progress.connect(self._handle_mesh_progress, Qt.QueuedConnection)
        worker.finished.connect(self._handle_mesh_finished, Qt.QueuedConnection)
        worker.failed.connect(self._handle_mesh_failed, Qt.QueuedConnection)
        worker.canceled.connect(self._handle_mesh_canceled, Qt.QueuedConnection)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.canceled.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        thread.started.connect(worker.run)

        self._mesh_thread = thread
        self._mesh_worker = worker
        self._mesh_start_ts = time.perf_counter()
        thread.start()
        return True

    def run_particle_generation_async(self, mesh_config=None, on_done=None):
        if self._mesh_thread and self._mesh_thread.isRunning():
            self._queue_mesh_status("Particle generation already running...")
            return False
        parts = [p for p in getattr(self, "parts", []) or [] if not getattr(p, "is_void", False)]
        if not parts:
            return False
        if not any(getattr(p, "geometry", None) is not None and not getattr(getattr(p, "geometry", None), "is_empty", True) for p in parts):
            return False
        try:
            dx, target_nodes, mesh_distribution, mesh_backend = self._resolve_mesh_config(mesh_config)
        except RuntimeError:
            return False
        except ValueError as exc:
            QMessageBox.warning(self, "Error", str(exc))
            return False
        dx, target_nodes, proceed = self._guard_mesh_density(dx, target_nodes, mesh_distribution)
        if not proceed:
            return False

        progress = QProgressDialog("Generating particles...", "Cancel", 0, 100, self.window() or self)
        progress.setWindowTitle("Generating Particles")
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setMinimumDuration(0)
        progress.show()
        self._mesh_progress_dialog = progress
        self._mesh_preview_only = True
        self._mesh_on_done = on_done
        self._mesh_task_kind = "particles"
        self._mesh_last_progress_ts = 0.0
        self._mesh_status_pct = None

        thread = QThread()
        worker = MeshWorker(self._compute_mesh_data_worker, dx, target_nodes, mesh_distribution, mesh_backend, True)
        worker.moveToThread(thread)

        progress.canceled.connect(worker.request_cancel)
        worker.progress.connect(self._handle_mesh_progress, Qt.QueuedConnection)
        worker.finished.connect(self._handle_mesh_finished, Qt.QueuedConnection)
        worker.failed.connect(self._handle_mesh_failed, Qt.QueuedConnection)
        worker.canceled.connect(self._handle_mesh_canceled, Qt.QueuedConnection)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.canceled.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        thread.started.connect(worker.run)

        self._mesh_thread = thread
        self._mesh_worker = worker
        self._mesh_start_ts = time.perf_counter()
        thread.start()
        return True

    def _cleanup_mesh_task(self):
        if self._mesh_progress_dialog:
            try:
                self._mesh_progress_dialog.reset()
            except Exception:
                pass
            self._mesh_progress_dialog = None
        worker = self._mesh_worker
        self._mesh_worker = None
        if worker:
            try:
                worker.request_cancel()
            except Exception:
                pass
            try:
                worker.deleteLater()
            except Exception:
                pass
        if self._mesh_thread:
            try:
                if QThread.currentThread() != self._mesh_thread:
                    self._mesh_thread.quit()
                    self._mesh_thread.wait()
            except Exception:
                pass
            self._mesh_thread = None
        self._mesh_on_done = None
        self._mesh_preview_only = False
        self._mesh_task_kind = "connections"
        self._mesh_status_pct = None
        self._mesh_last_progress_ts = 0.0

    def stop_background_tasks(self):
        self._cleanup_mesh_task()
        controller = self._get_results_controller()
        if controller is not None and hasattr(controller, "stop"):
            try:
                controller.stop()
            except Exception:
                pass

    @Slot(int, str)
    def _handle_mesh_progress(self, value, message):
        if not self._mesh_progress_dialog:
            return
        now = time.monotonic()
        if value not in (0, 100) and (now - self._mesh_last_progress_ts) < 0.05:
            return
        self._mesh_last_progress_ts = now
        label = message
        if self._mesh_start_ts is not None and value > 0:
            elapsed = time.perf_counter() - self._mesh_start_ts
            remaining = elapsed * (100 - value) / max(value, 1)
            label = f"{message} (ETA {self._format_duration(remaining)})"
        self._mesh_progress_dialog.setLabelText(label)
        self._mesh_progress_dialog.setValue(int(value))
        try:
            pct = int(value)
        except Exception:
            pct = None
        if pct is not None:
            last_pct = getattr(self, "_mesh_status_pct", None)
            if last_pct is None or abs(pct - last_pct) >= 5 or pct in (0, 100):
                self._mesh_status_pct = pct
                task_label = "Particles" if getattr(self, "_mesh_task_kind", "connections") == "particles" else "Connections"
                status_msg = f"{task_label} {pct}% - {message}"
                self._queue_mesh_status(status_msg, 1500)

    @Slot(object)
    def _handle_mesh_finished(self, result):
        self._apply_mesh_result(result)
        task_kind = getattr(self, "_mesh_task_kind", "connections")
        mesh_end = time.perf_counter()
        mesh_start = getattr(self, "_mesh_start_ts", None)
        if mesh_start is not None:
            total_time = max(0.0, mesh_end - mesh_start)
            if len(self.global_nodes) > 0 and total_time > 0:
                self._mesh_time_history.append((len(self.global_nodes), total_time))
                if len(self._mesh_time_history) > 5:
                    self._mesh_time_history = self._mesh_time_history[-5:]
        if task_kind != "particles" and self.project_mode == "3d":
            if not self.build_3d_mesh():
                QMessageBox.warning(
                    self,
                    "Extrude Error",
                    "Failed to build 3D connections from the 2D surface connections.",
                )
                on_done = self._mesh_on_done
                self._cleanup_mesh_task()
                if on_done:
                    on_done(False)
                return
        if task_kind == "particles":
            spacing_stats = self._mesh_spacing_stats(self.global_nodes)
            message = f"Particles generated from geometry.\nTotal Particles: {len(self.global_nodes)}"
            if result.get("target_nodes") is not None:
                message += f"\nTarget Particles: {result.get('target_nodes')}"
            if spacing_stats:
                min_d, avg_d, max_d = spacing_stats
                message += (
                    f"\nNearest spacing (approx): "
                    f"{min_d:.3f}/{avg_d:.3f}/{max_d:.3f} {self.current_unit}"
                )
            self.display_mode = "mesh"
            self.show_mesh_elements = False
            self.show_mesh_nodes = True
            self.redraw()
            self._queue_mesh_status("Particle generation complete.", 4000)
            on_done = self._mesh_on_done
            self._cleanup_mesh_task()
            if on_done:
                on_done(True)
            else:
                QMessageBox.information(self, "Particles Ready", message)
            return
        if not self._mesh_preview_only:
            mesh_msg = (
                f"Particles generated from geometry.\nTotal Particles: {len(self.global_nodes)}\n"
                f"Internal Connections: {len(self.global_elements)}"
            )
            if result.get("target_nodes") is not None:
                mesh_msg += f"\nTarget Particles: {result.get('target_nodes')}"
            spacing_stats = self._mesh_spacing_stats(self.global_nodes)
            if spacing_stats:
                min_d, avg_d, max_d = spacing_stats
                mesh_msg += (
                    f"\nNearest spacing (approx): "
                    f"{min_d:.3f}/{avg_d:.3f}/{max_d:.3f} {self.current_unit}"
                )
            QMessageBox.information(self, "Particles Ready", mesh_msg)
            self._show_preview_window()
            self.stageAdvanceRequested.emit(ProjectStage.MESH)
            QTimer.singleShot(3000, lambda: self.stageAdvanceRequested.emit(ProjectStage.JOB))
        self._queue_mesh_status("Particle generation complete.", 4000)
        on_done = self._mesh_on_done
        self._cleanup_mesh_task()
        if on_done:
            on_done(True)

    @Slot(str)
    def _handle_mesh_failed(self, message):
        title = "Particle Generation Error" if getattr(self, "_mesh_task_kind", "connections") == "particles" else "Connection Generation Error"
        QMessageBox.critical(self, title, message)
        on_done = self._mesh_on_done
        self._cleanup_mesh_task()
        if on_done:
            on_done(False)

    @Slot()
    def _handle_mesh_canceled(self):
        task_label = "Particle generation" if getattr(self, "_mesh_task_kind", "connections") == "particles" else "Connection generation"
        self._queue_mesh_status(f"{task_label} canceled.")
        on_done = self._mesh_on_done
        self._cleanup_mesh_task()
        if on_done:
            on_done(False)

    def _show_preview_window(self):
        if len(self.global_nodes) == 0:
            QMessageBox.warning(self, "Preview", "No connections to preview.")
            return
        self.display_mode = "mesh"
        self.redraw()

    def _unit_scale_to_meters(self):
        unit = self._normalize_length_unit()
        return self._length_unit_scale_to_meters(unit) or 0.001

    def _meters_to_display_scale(self):
        unit = self._normalize_length_unit()
        return {
            "mm": 1000.0,
            "cm": 100.0,
            "m": 1.0,
        }.get(unit, 1000.0)

    def _display_length_to_si(self, value):
        try:
            return float(value) * self._unit_scale_to_meters()
        except (TypeError, ValueError):
            return 0.0

    def _si_length_to_display(self, value):
        try:
            return float(value) * self._meters_to_display_scale()
        except (TypeError, ValueError):
            return 0.0

    def _convert_nodes_to_meters(self, nodes):
        if nodes is None:
            return np.empty((0, 2), dtype=float)
        return np.asarray(nodes, dtype=float) * self._unit_scale_to_meters()

    def _mesh_spacing_stats(self, nodes, sample_limit=5000):
        if nodes is None:
            return None
        nodes = np.asarray(nodes, dtype=float)
        if len(nodes) < 2:
            return None
        if len(nodes) > sample_limit:
            idx = np.random.choice(len(nodes), size=sample_limit, replace=False)
            sample = nodes[idx]
        else:
            sample = nodes
        try:
            tree = cKDTree(nodes)
            dists, _ = tree.query(sample, k=2)
            nearest = dists[:, 1]
            return float(np.min(nearest)), float(np.mean(nearest)), float(np.max(nearest))
        except Exception:
            return None

    def _stats_from_values(self, values):
        if not values:
            return None
        arr = np.asarray(list(values), dtype=float)
        if arr.size == 0:
            return None
        arr = arr[np.isfinite(arr)]
        arr = arr[arr > 1e-12]
        if arr.size == 0:
            return None
        return float(np.min(arr)), float(np.mean(arr)), float(np.max(arr))

    def _ring_cumulative_lengths(self, ring):
        if not ring or len(ring) < 2:
            return None, None, 0.0
        seg_lengths = []
        cum = [0.0]
        total = 0.0
        for i in range(len(ring) - 1):
            try:
                seg_len = float(dist(ring[i], ring[i + 1]))
            except Exception:
                seg_len = 0.0
            seg_lengths.append(seg_len)
            total += max(0.0, seg_len)
            cum.append(total)
        return np.asarray(cum, dtype=float), np.asarray(seg_lengths, dtype=float), float(total)

    def _boundary_edge_spacing_stats(self, nodes, tol=None):
        if nodes is None:
            return None
        nodes = np.asarray(nodes, dtype=float)
        if nodes.ndim != 2 or len(nodes) < 2 or nodes.shape[1] < 2:
            return None
        rings = self._get_boundary_rings()
        if not rings:
            return None
        try:
            dx = float(self.last_mesh_dx or 0.0)
        except Exception:
            dx = 0.0
        if tol is None:
            tol = max(1e-6, 0.15 * dx) if dx > 0 else 1e-3
        all_gaps = []
        for ring in rings:
            if not ring or len(ring) < 2:
                continue
            cum, seg_lengths, total = self._ring_cumulative_lengths(ring)
            if cum is None or seg_lengths is None or total <= 1e-9:
                continue
            s_values = []
            used_nodes = set()
            for nid in range(len(nodes)):
                pt = (float(nodes[nid, 0]), float(nodes[nid, 1]))
                proj = self._project_point_to_ring(pt, ring, cum, seg_lengths)
                if not proj:
                    continue
                d_to_ring, s_val, _proj_pt = proj
                if d_to_ring <= tol:
                    if nid in used_nodes:
                        continue
                    used_nodes.add(nid)
                    s_values.append(float(s_val))
            if len(s_values) < 2:
                continue
            s_values = sorted(s_values)
            ring_gaps = []
            for i in range(1, len(s_values)):
                gap = s_values[i] - s_values[i - 1]
                if gap > 1e-12:
                    ring_gaps.append(gap)
            wrap_gap = total - s_values[-1] + s_values[0]
            if wrap_gap > 1e-12:
                ring_gaps.append(wrap_gap)
            if ring_gaps:
                all_gaps.extend(ring_gaps)
        return self._stats_from_values(all_gaps)

    def _interface_spacing_stats(self, nodes):
        if nodes is None:
            return None
        nodes = np.asarray(nodes, dtype=float)
        if nodes.ndim != 2 or len(nodes) < 2 or nodes.shape[1] < 2:
            return None
        if self.global_elements is None or len(self.global_elements) == 0 or not self.interfaces:
            return None
        try:
            rows = self._get_interface_preview_rows()
        except Exception:
            return None

    def _interface_topology_diagnostics(self):
        """
        Live topology diagnostics from the current in-memory preview mesh.

        Uses current mesh arrays + interface preview classification rows (not workspace CSV files).
        """
        if self.project_mode == "3d":
            return None
        if self.global_nodes is None or self.global_elements is None:
            return None
        if len(self.global_nodes) == 0 or len(self.global_elements) == 0 or not self.interfaces:
            return None
        try:
            rows = self._get_interface_preview_rows()
        except Exception:
            return None
        if not rows:
            return None

        # Count interface triangles by interface id from current preview classification.
        iface_triangles = {}
        for row in rows:
            if str(row.get("zone_kind", "")).lower() != "interface":
                continue
            try:
                iid = int(row.get("interface_id"))
            except Exception:
                continue
            iface_triangles[iid] = iface_triangles.get(iid, 0) + 1

        # Build edge -> zone usage map from preview rows.
        edge_zone_usage = {}
        for row in rows:
            try:
                p1 = int(row.get("p1"))
                p2 = int(row.get("p2"))
                p3 = int(row.get("p3"))
            except Exception:
                continue
            zkind = str(row.get("zone_kind", "")).lower()
            if zkind == "interface":
                try:
                    zone_key = ("interface", int(row.get("interface_id")))
                except Exception:
                    continue
            else:
                try:
                    zone_key = ("part", int(row.get("part_id")))
                except Exception:
                    continue
            for edge in ((p1, p2), (p2, p3), (p3, p1)):
                ekey = tuple(sorted(edge))
                zones = edge_zone_usage.setdefault(ekey, set())
                zones.add(zone_key)

        iface_objs = {}
        for iface in (self.interfaces or []):
            try:
                iface_objs[int(getattr(iface, "id", -1))] = iface
            except Exception:
                continue
        if not iface_objs:
            return None

        try:
            part_effective_geoms = self._interface_effective_geom_map()
        except Exception:
            part_effective_geoms = {}

        per_interface = {}
        total_iface_triangles = 0
        total_direct_matrix_inclusion_edges = 0
        total_shared_matrix_interface_edges = 0
        total_shared_inclusion_interface_edges = 0
        active_iface_count = 0

        for iface_id, iface in sorted(iface_objs.items()):
            tri_count = int(iface_triangles.get(iface_id, 0))
            try:
                side_info = self._interface_matrix_inclusion_parts(iface, part_effective_geoms)
            except Exception:
                side_info = None
            matrix_id = side_info.get("matrix_id") if side_info else None
            inclusion_id = side_info.get("inclusion_id") if side_info else None
            if matrix_id in (None, "") or inclusion_id in (None, ""):
                diag = {
                    "interface_id": iface_id,
                    "triangles": tri_count,
                    "matrix_part_id": matrix_id,
                    "inclusion_part_id": inclusion_id,
                    "shared_matrix_interface_edges": 0,
                    "shared_inclusion_interface_edges": 0,
                    "direct_matrix_inclusion_edges": 0,
                }
                per_interface[iface_id] = diag
                if tri_count > 0:
                    total_iface_triangles += tri_count
                    active_iface_count += 1
                continue

            z_iface = ("interface", int(iface_id))
            z_mat = ("part", int(matrix_id))
            z_inc = ("part", int(inclusion_id))
            shared_m_if = 0
            shared_i_if = 0
            direct_m_i = 0
            for zones in edge_zone_usage.values():
                has_mat = z_mat in zones
                has_inc = z_inc in zones
                has_iface = z_iface in zones
                if has_mat and has_iface:
                    shared_m_if += 1
                if has_inc and has_iface:
                    shared_i_if += 1
                if has_mat and has_inc and not has_iface:
                    direct_m_i += 1

            diag = {
                "interface_id": int(iface_id),
                "triangles": tri_count,
                "matrix_part_id": int(matrix_id),
                "inclusion_part_id": int(inclusion_id),
                "shared_matrix_interface_edges": int(shared_m_if),
                "shared_inclusion_interface_edges": int(shared_i_if),
                "direct_matrix_inclusion_edges": int(direct_m_i),
            }
            per_interface[iface_id] = diag
            if tri_count > 0:
                active_iface_count += 1
            total_iface_triangles += tri_count
            total_direct_matrix_inclusion_edges += direct_m_i
            total_shared_matrix_interface_edges += shared_m_if
            total_shared_inclusion_interface_edges += shared_i_if

        return {
            "interfaces_total": int(len(iface_objs)),
            "interfaces_active": int(active_iface_count),
            "interface_triangles": int(total_iface_triangles),
            "direct_matrix_inclusion_edges": int(total_direct_matrix_inclusion_edges),
            "shared_matrix_interface_edges": int(total_shared_matrix_interface_edges),
            "shared_inclusion_interface_edges": int(total_shared_inclusion_interface_edges),
            "per_interface": per_interface,
        }
        if not rows:
            return None
        iface_node_ids = set()
        elem_count = len(self.global_elements)
        for row in rows:
            if str(row.get("zone_kind", "")).lower() != "interface":
                continue
            try:
                ei = int(row.get("_element_idx"))
            except Exception:
                continue
            if ei < 0 or ei >= elem_count:
                continue
            try:
                tri = self.global_elements[ei]
                for nid in tri[:3]:
                    iid = int(nid)
                    if 0 <= iid < len(nodes):
                        iface_node_ids.add(iid)
            except Exception:
                continue
        if len(iface_node_ids) < 2:
            return None
        ids = np.fromiter(sorted(iface_node_ids), dtype=int)
        pts = nodes[ids, :2]
        try:
            tree = cKDTree(pts)
            dists, _ = tree.query(pts, k=2)
            nearest = dists[:, 1]
            return self._stats_from_values(nearest)
        except Exception:
            return None

    def _mesh_qa_signature(self):
        try:
            node_len = len(self.global_nodes) if self.global_nodes is not None else 0
        except Exception:
            node_len = 0
        try:
            elem_len = len(self.global_elements) if self.global_elements is not None else 0
        except Exception:
            elem_len = 0
        try:
            dx = round(float(self.last_mesh_dx), 8) if self.last_mesh_dx is not None else None
        except Exception:
            dx = None
        try:
            iface_sig = self._interface_preview_signature()
        except Exception:
            iface_sig = None
        return (
            self.project_mode,
            id(self.global_nodes),
            node_len,
            id(self.global_elements),
            elem_len,
            id(self.solid_geometry),
            dx,
            iface_sig,
        )

    def get_mesh_qa_stats(self):
        if self.project_mode == "3d":
            return {"boundary": None, "interface": None, "topology": None, "unit": self.current_unit or ""}
        sig = self._mesh_qa_signature()
        if self._mesh_qa_cache is not None and self._mesh_qa_cache_sig == sig:
            return self._mesh_qa_cache
        nodes = self.global_nodes if self.global_nodes is not None and len(self.global_nodes) else None
        stats = {
            "boundary": self._boundary_edge_spacing_stats(nodes),
            "interface": self._interface_spacing_stats(nodes),
            "topology": self._interface_topology_diagnostics(),
            "unit": self.current_unit or "",
        }
        self._mesh_qa_cache = stats
        self._mesh_qa_cache_sig = sig
        return stats

    def get_interface_topology_diagnostics(self):
        stats = self.get_mesh_qa_stats() or {}
        return stats.get("topology")

    def get_mesh_qa_readout(self):
        stats = self.get_mesh_qa_stats() or {}
        unit = stats.get("unit", "") or ""
        if self.project_mode == "3d":
            return "Boundary edge spacing: --\nInteraction spacing: -- (2D particle QA only)"

        def _fmt(label, vals):
            if not vals:
                return f"{label}: --"
            mn, avg, mx = vals
            suffix = f" {unit}" if unit else ""
            return f"{label}: {mn:.3f}/{avg:.3f}/{mx:.3f}{suffix}"

        def _fmt_topology(diag):
            if not diag:
                return "Interaction topology: --"
            return (
                "Interaction topology: "
                f"tri={int(diag.get('interface_triangles', 0))} | "
                f"m-if={int(diag.get('shared_matrix_interface_edges', 0))} | "
                f"i-if={int(diag.get('shared_inclusion_interface_edges', 0))} | "
                f"direct m-i={int(diag.get('direct_matrix_inclusion_edges', 0))}"
            )

        return "\n".join(
            [
                _fmt("Boundary edge spacing", stats.get("boundary")),
                _fmt("Interaction spacing", stats.get("interface")),
                _fmt_topology(stats.get("topology")),
            ]
        )

    def _warn_if_cpd_bounds_exceeded(self, nodes_m):
        self._warned_cpd_bounds = False
        return

    def preview_3d_mesh(self):
        if self.project_mode != "3d":
            QMessageBox.warning(self, "3D Preview", "Switch to a 3D project to preview 3D connections.")
            return False
        if self._cad_kernel_ready():
            if not self.build_3d_mesh():
                QMessageBox.warning(self, "3D Preview", "Failed to build CAD preview connections.")
                return False
            self.display_mode = "mesh_3d"
            self.redraw()
            self.mesh3dUpdated.emit(self.global_nodes_3d, self.global_elements_3d)
            return True
        if (
            self.global_nodes is None
            or len(self.global_nodes) == 0
            or self.global_elements is None
            or len(self.global_elements) == 0
        ):
            if not self.run_cpd(preview_only=True):
                return False
        if self.global_nodes_3d is None or len(self.global_nodes_3d) == 0:
            if not self.build_3d_mesh():
                QMessageBox.warning(self, "3D Preview", "Failed to build 3D connections for preview.")
                return False
        self.display_mode = "mesh_3d"
        self.redraw()
        self.mesh3dUpdated.emit(self.global_nodes_3d, self.global_elements_3d)
        return True

    def _export_cpd_main_inputs(
        self,
        nodes,
        fixed_nodes,
        _force_acc,
        velocity_map=None,
        particle_material_map=None,
    ):
        from solver_exporter import export_cpd_main_inputs
        return export_cpd_main_inputs(
            self,
            nodes,
            fixed_nodes,
            _force_acc,
            velocity_map=velocity_map,
            particle_material_map=particle_material_map,
        )

    def _mirror_inputs_to_cpd_setup(self, input_files):
        setup_dir = self._workspace_input_path()
        os.makedirs(setup_dir, exist_ok=True)
        for fname in input_files:
            src = self._workspace_path(fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(setup_dir, fname))

    def _interface_effective_geom_map(self, part_ids=None):
        """Effective part geometries (subtract child void/embedded parts) for interface operations."""
        part_ids_filter = None
        if part_ids is not None:
            try:
                part_ids_filter = {int(pid) for pid in part_ids}
            except Exception:
                part_ids_filter = None
        out = {}
        for part in getattr(self, "parts", []) or []:
            try:
                pid = int(getattr(part, "id"))
            except Exception:
                continue
            if part_ids_filter is not None and pid not in part_ids_filter:
                continue
            geom = getattr(part, "geometry", None)
            if geom is None or getattr(geom, "is_empty", True):
                continue
            eff = geom
            try:
                for child in self.get_child_parts(part):
                    cgeom = getattr(child, "geometry", None)
                    if cgeom is None or getattr(cgeom, "is_empty", True):
                        continue
                    try:
                        eff = eff.difference(cgeom).buffer(0)
                    except Exception:
                        pass
            except Exception:
                pass
            if eff is None or getattr(eff, "is_empty", True):
                continue
            out[pid] = eff
        return out

    def _interface_centerline_geometry(self, iface, part_effective_geoms=None):
        """Best-effort shared boundary geometry for an interface pair."""
        if iface is None:
            return None
        try:
            p1_id = int(getattr(iface, "part1_id"))
            p2_id = int(getattr(iface, "part2_id"))
        except Exception:
            return None
        if part_effective_geoms is None:
            part_effective_geoms = self._interface_effective_geom_map([p1_id, p2_id])
        g1 = part_effective_geoms.get(p1_id)
        g2 = part_effective_geoms.get(p2_id)
        if g1 is None or g2 is None:
            return None
        def _has_linework(geom):
            if geom is None or getattr(geom, "is_empty", True):
                return False
            try:
                return bool(self._iter_line_geometries(geom))
            except Exception:
                return False
        centerline = None
        try:
            centerline = g1.boundary.intersection(g2.boundary)
        except Exception:
            centerline = None
        if not _has_linework(centerline):
            try:
                overlap = g1.intersection(g2)
            except Exception:
                overlap = None
            if overlap is not None and not getattr(overlap, "is_empty", True):
                try:
                    centerline = overlap.boundary
                except Exception:
                    centerline = overlap
        if not _has_linework(centerline):
            try:
                if g1.covers(g2):
                    centerline = g2.boundary
                elif g2.covers(g1):
                    centerline = g1.boundary
            except Exception:
                pass
        if not _has_linework(centerline):
            # Robust fallback for matrix-with-hole cases where boolean ops perturb boundary
            # coordinates and exact boundary intersection returns empty/points only.
            try:
                side_info = self._interface_matrix_inclusion_parts(
                    iface,
                    {
                        p1_id: g1,
                        p2_id: g2,
                    },
                )
            except Exception:
                side_info = None
            inc_geom = side_info.get("inclusion_geom") if side_info else None
            if inc_geom is not None and not getattr(inc_geom, "is_empty", True):
                try:
                    centerline = inc_geom.boundary
                except Exception:
                    centerline = inc_geom
        if not _has_linework(centerline):
            return None
        return centerline

    def _interface_placement_mode(self, iface):
        mode = str(
            getattr(
                iface,
                "placement_mode",
                getattr(Interface, "DEFAULT_PLACEMENT_MODE", "matrix_side"),
            )
            or getattr(Interface, "DEFAULT_PLACEMENT_MODE", "matrix_side")
        ).strip().lower()
        # Frontend implementation currently supports matrix-side coating mode.
        if mode not in {"matrix_side"}:
            return "matrix_side"
        return mode

    def _interface_layer_mode(self, iface):
        mode = str(
            getattr(
                iface,
                "layer_mode",
                getattr(Interface, "DEFAULT_LAYER_MODE", "single_layer_ring"),
            )
            or getattr(Interface, "DEFAULT_LAYER_MODE", "single_layer_ring")
        ).strip().lower()
        if mode not in {"single_layer_ring"}:
            return "single_layer_ring"
        return mode

    def _interface_effective_sampling_dx(self, iface, mesh_spacing):
        """
        Effective interface sampling spacing for the current mesh run.

        For `single_layer_ring`, interface spacing must follow the active mesh dx so the one-triangle
        ring updates when the user changes the Particle Connections dx in the UI.
        """
        try:
            mesh_spacing = float(mesh_spacing)
        except Exception:
            mesh_spacing = 0.0
        if mesh_spacing <= 0.0:
            mesh_spacing = 1.0
        layer_mode = self._interface_layer_mode(iface)
        if layer_mode == "single_layer_ring":
            return float(mesh_spacing)
        try:
            target_dx = float(getattr(iface, "target_dx", 0.0) or 0.0)
        except Exception:
            target_dx = 0.0
        return float(target_dx if target_dx > 0.0 else mesh_spacing)

    def _interface_effective_thickness(self, iface, mesh_spacing):
        """
        Effective interface band thickness for the current mesh run.

        `single_layer_ring` is a one-element-strip approximation, so its thickness should track the
        active mesh dx regardless of the stored interface thickness value.
        """
        try:
            mesh_spacing = float(mesh_spacing)
        except Exception:
            mesh_spacing = 0.0
        if mesh_spacing <= 0.0:
            mesh_spacing = 1.0
        layer_mode = self._interface_layer_mode(iface)
        if layer_mode == "single_layer_ring":
            return float(mesh_spacing)
        try:
            thickness = float(getattr(iface, "thickness", 0.0) or 0.0)
        except Exception:
            thickness = 0.0
        return float(thickness if thickness > 0.0 else mesh_spacing)

    def _interface_matrix_inclusion_parts(self, iface, part_effective_geoms=None):
        """Infer matrix/inclusion roles for interface placement from containment/area."""
        if iface is None:
            return None
        try:
            p1_id = int(getattr(iface, "part1_id"))
            p2_id = int(getattr(iface, "part2_id"))
        except Exception:
            return None
        if p1_id == p2_id:
            return None
        if part_effective_geoms is None:
            part_effective_geoms = self._interface_effective_geom_map([p1_id, p2_id])
        g1 = part_effective_geoms.get(p1_id)
        g2 = part_effective_geoms.get(p2_id)
        if g1 is None or g2 is None:
            return None

        matrix_id = None
        inclusion_id = None
        try:
            if g1.covers(g2):
                matrix_id, inclusion_id = p1_id, p2_id
            elif g2.covers(g1):
                matrix_id, inclusion_id = p2_id, p1_id
        except Exception:
            pass
        if matrix_id is None:
            try:
                a1 = float(getattr(g1, "area", 0.0) or 0.0)
            except Exception:
                a1 = 0.0
            try:
                a2 = float(getattr(g2, "area", 0.0) or 0.0)
            except Exception:
                a2 = 0.0
            if a1 >= a2:
                matrix_id, inclusion_id = p1_id, p2_id
            else:
                matrix_id, inclusion_id = p2_id, p1_id
        return {
            "matrix_id": matrix_id,
            "inclusion_id": inclusion_id,
            "matrix_geom": part_effective_geoms.get(matrix_id),
            "inclusion_geom": part_effective_geoms.get(inclusion_id),
        }

    def _iter_line_geometries(self, geom):
        """Yield line-like components (LineString/LinearRing) from any geometry."""
        if geom is None or getattr(geom, "is_empty", True):
            return []
        raw_lines = []
        stack = [geom]
        while stack:
            g = stack.pop()
            if g is None or getattr(g, "is_empty", True):
                continue
            gtype = getattr(g, "geom_type", "")
            if gtype in ("LineString", "LinearRing"):
                raw_lines.append(g)
                continue
            sub = getattr(g, "geoms", None)
            if sub is not None:
                try:
                    stack.extend(list(sub))
                except Exception:
                    pass
        if not raw_lines:
            return []
        if len(raw_lines) == 1:
            return raw_lines

        merge_inputs = []
        for g in raw_lines:
            try:
                coords = list(g.coords)
            except Exception:
                coords = []
            if len(coords) >= 2:
                try:
                    merge_inputs.append(LineString(coords))
                except Exception:
                    merge_inputs.append(g)
            else:
                merge_inputs.append(g)

        try:
            merged = linemerge(merge_inputs)
        except Exception:
            merged = None
        if merged is None or getattr(merged, "is_empty", True):
            return raw_lines

        out = []
        stack = [merged]
        while stack:
            g = stack.pop()
            if g is None or getattr(g, "is_empty", True):
                continue
            gtype = getattr(g, "geom_type", "")
            if gtype in ("LineString", "LinearRing"):
                out.append(g)
                continue
            sub = getattr(g, "geoms", None)
            if sub is not None:
                try:
                    stack.extend(list(sub))
                except Exception:
                    pass
        return out or raw_lines

    def _sample_line_geometry_points(self, line, spacing, min_points=12):
        """
        Sample a line/ring with spacing ~dx, anchoring to low-vertex polygon edges.

        This mirrors the boundary sampling behavior used in mesh-point collection so interface
        centerlines do not get a phase-shifted duplicate ring relative to polygon boundaries.
        """
        if line is None:
            return []
        try:
            length = float(getattr(line, "length", 0.0) or 0.0)
        except Exception:
            length = 0.0
        try:
            spacing = float(spacing)
        except Exception:
            spacing = 0.0
        if length <= 0.0 or spacing <= 1e-12:
            return []
        target = max(1.0, spacing)
        target_floor = max(1.0, 0.85 * spacing)
        try:
            min_points_i = int(min_points)
        except Exception:
            min_points_i = 12

        try:
            coords = [tuple(map(float, pt[:2])) for pt in list(line.coords)]
        except Exception:
            coords = []

        samples = None
        unique_count = 0
        if coords:
            unique_count = len(coords)
            try:
                if len(coords) > 1 and dist(coords[0], coords[-1]) <= 1e-9:
                    unique_count -= 1
            except Exception:
                pass
        # Do not aggressively densify highly tessellated curves (e.g., Shapely buffer arcs).
        # Keep spacing close to dx so curved interfaces are approximated by short straight segments,
        # rather than forcing every source polyline vertex to survive.
        if min_points_i > 0:
            try:
                min_target = float(length) / max(min_points_i, 1)
            except Exception:
                min_target = target
            if unique_count > 16:
                target = min(target, max(target_floor, min_target))
            else:
                target = min(target, min_target)
        target = max(target_floor, target)

        prefer_per_segment = bool(2 <= unique_count <= 16 and len(coords) >= 2)
        if prefer_per_segment and unique_count >= 4:
            try:
                is_closed = bool(len(coords) > 2 and dist(coords[0], coords[-1]) <= 1e-9)
            except Exception:
                is_closed = bool(len(coords) > 2 and coords[0] == coords[-1])

            angle_thresh = math.radians(25.0)

            def _turn_angle(prev_pt, pt, next_pt):
                v1 = (pt[0] - prev_pt[0], pt[1] - prev_pt[1])
                v2 = (next_pt[0] - pt[0], next_pt[1] - pt[1])
                l1 = math.hypot(v1[0], v1[1])
                l2 = math.hypot(v2[0], v2[1])
                if l1 <= 1e-9 or l2 <= 1e-9:
                    return math.pi
                dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)
                dot = max(-1.0, min(1.0, dot))
                return math.acos(dot)

            sharp_count = 0
            pts_eval = coords[:-1] if is_closed and len(coords) > 1 else coords
            if is_closed and len(pts_eval) >= 3:
                n = len(pts_eval)
                for i in range(n):
                    if _turn_angle(pts_eval[(i - 1) % n], pts_eval[i], pts_eval[(i + 1) % n]) >= angle_thresh:
                        sharp_count += 1
                # Smooth closed curves (few/no corners) should use spacing-based resampling.
                if sharp_count < 3:
                    prefer_per_segment = False
            elif len(coords) >= 3:
                for i in range(1, len(coords) - 1):
                    if _turn_angle(coords[i - 1], coords[i], coords[i + 1]) >= angle_thresh:
                        sharp_count += 1
                # Open smooth arcs should use spacing-based resampling.
                if sharp_count < 1:
                    prefer_per_segment = False

        if prefer_per_segment:
            samples = []
            for seg_idx in range(len(coords) - 1):
                a = coords[seg_idx]
                b = coords[seg_idx + 1]
                try:
                    seg_len = float(dist(a, b))
                except Exception:
                    seg_len = 0.0
                if seg_len <= 1e-12:
                    continue
                nseg = max(1, int(round(seg_len / target)))
                for j in range(nseg + 1):
                    if seg_idx > 0 and j == 0:
                        continue
                    t = float(j) / float(nseg)
                    samples.append(
                        (
                            (1.0 - t) * float(a[0]) + t * float(b[0]),
                            (1.0 - t) * float(a[1]) + t * float(b[1]),
                        )
                    )
        if samples is None:
            try:
                samples = [tuple(map(float, pt[:2])) for pt in sample_ring(line, target)]
            except Exception:
                samples = []
        return samples

    def _infer_circle_from_polygon(self, geom):
        """Best-effort circle fit for a polygonal shape (used for circle-aware meshing)."""
        if geom is None or getattr(geom, "is_empty", True):
            return None
        try:
            if getattr(geom, "geom_type", "") == "MultiPolygon":
                geoms = list(getattr(geom, "geoms", []))
                if len(geoms) != 1:
                    return None
                geom = geoms[0]
        except Exception:
            return None
        if getattr(geom, "geom_type", "") != "Polygon":
            return None
        try:
            if len(getattr(geom, "interiors", [])) > 0:
                return None
        except Exception:
            return None
        try:
            coords = list(geom.exterior.coords)
        except Exception:
            return None
        if not coords:
            return None
        pts = [tuple(map(float, pt[:2])) for pt in coords]
        if len(pts) > 1 and dist(pts[0], pts[-1]) <= 1e-9:
            pts = pts[:-1]
        if len(pts) < 12:
            return None
        try:
            c = geom.centroid
            cx = float(c.x)
            cy = float(c.y)
        except Exception:
            return None
        rr = []
        for x, y in pts:
            r = math.hypot(x - cx, y - cy)
            if not math.isfinite(r):
                continue
            rr.append(r)
        if len(rr) < 8:
            return None
        r_mean = float(sum(rr) / len(rr))
        if r_mean <= 1e-9:
            return None
        r_var = float(sum((r - r_mean) ** 2 for r in rr) / max(1, len(rr)))
        r_std = math.sqrt(max(0.0, r_var))
        rel_std = r_std / max(r_mean, 1e-9)
        try:
            area = float(getattr(geom, "area", 0.0) or 0.0)
        except Exception:
            area = 0.0
        area_ref = math.pi * r_mean * r_mean
        area_err = abs(area - area_ref) / max(area_ref, 1e-9)
        try:
            minx, miny, maxx, maxy = geom.bounds
            w = float(maxx - minx)
            h = float(maxy - miny)
        except Exception:
            w = h = 0.0
        aspect = min(w, h) / max(w, h, 1e-9) if max(w, h) > 1e-9 else 0.0
        if rel_std > 0.06:
            return None
        if area_err > 0.12:
            return None
        if aspect < 0.92:
            return None
        phase = 0.0
        try:
            x0, y0 = pts[0]
            phase = math.atan2(y0 - cy, x0 - cx)
        except Exception:
            phase = 0.0
        return {
            "center": (cx, cy),
            "radius": r_mean,
            "phase": phase,
            "rel_std": rel_std,
            "area_err": area_err,
            "aspect": aspect,
        }

    def _circle_spec_from_part_sketch_meta(self, part, effective_geom=None):
        """
        Exact circle parameters from original sketch metadata, validated against the part geometry.

        Returns None unless we can confidently identify a single circle defining the part.
        """
        if part is None:
            return None
        metas = list(getattr(part, "sketch_meta", []) or [])
        sketches = list(getattr(part, "sketches", []) or [])
        candidates = []
        for idx, meta in enumerate(metas):
            if not isinstance(meta, dict):
                continue
            if str(meta.get("type", "")).strip().lower() != "circle":
                continue
            center = meta.get("center")
            try:
                radius = float(meta.get("radius", 0.0) or 0.0)
            except Exception:
                radius = 0.0
            if not center or radius <= 0.0:
                continue
            try:
                cx = float(center[0])
                cy = float(center[1])
            except Exception:
                continue
            phase = 0.0
            if idx < len(sketches):
                pts = list(sketches[idx] if sketches[idx] is not None else [])
                if pts:
                    try:
                        x0, y0 = float(pts[0][0]), float(pts[0][1])
                        phase = math.atan2(y0 - cy, x0 - cx)
                    except Exception:
                        phase = 0.0
            candidates.append(
                {
                    "index": idx,
                    "center": (cx, cy),
                    "radius": radius,
                    "phase": phase,
                }
            )
        if not candidates:
            return None
        if effective_geom is None or getattr(effective_geom, "is_empty", True):
            return candidates[0] if len(candidates) == 1 else None

        # Validate candidates against effective geometry; require a clear best match.
        scored = []
        try:
            g_centroid = effective_geom.centroid
            gx = float(g_centroid.x)
            gy = float(g_centroid.y)
        except Exception:
            gx = gy = None
        try:
            g_area = float(getattr(effective_geom, "area", 0.0) or 0.0)
        except Exception:
            g_area = 0.0
        try:
            minx, miny, maxx, maxy = effective_geom.bounds
            gw = float(maxx - minx)
            gh = float(maxy - miny)
        except Exception:
            gw = gh = 0.0
        for cand in candidates:
            cx, cy = cand["center"]
            r = float(cand["radius"])
            if r <= 0.0:
                continue
            centroid_err = 0.0
            if gx is not None and gy is not None:
                centroid_err = math.hypot(cx - gx, cy - gy) / max(r, 1e-9)
            area_ref = math.pi * r * r
            area_err = abs(g_area - area_ref) / max(area_ref, 1e-9)
            size_err = 0.0
            if gw > 0.0 and gh > 0.0:
                size_err = (abs(gw - 2.0 * r) + abs(gh - 2.0 * r)) / max(2.0 * r, 1e-9)
            score = centroid_err + area_err + 0.5 * size_err
            scored.append((score, centroid_err, area_err, size_err, cand))
        if not scored:
            return None
        scored.sort(key=lambda x: x[0])
        best = scored[0]
        _score, centroid_err, area_err, size_err, cand = best
        # Strong guards: avoid activating circle-aware mode on uncertain matches.
        if centroid_err > 0.08:
            return None
        if area_err > 0.15:
            return None
        if size_err > 0.18:
            return None
        if len(scored) > 1 and abs(scored[1][0] - scored[0][0]) < 0.05:
            return None
        out = dict(cand)
        out["source"] = "part_sketch_meta"
        out["centroid_err"] = centroid_err
        out["area_err"] = area_err
        out["size_err"] = size_err
        return out

    def _sample_circle_ring_points(self, center, radius, spacing, phase=0.0, min_points=12):
        """Uniform angular sampling of a circle with target chord length ~ spacing."""
        try:
            cx = float(center[0])
            cy = float(center[1])
            radius = float(radius)
            spacing = float(spacing)
            phase = float(phase)
        except Exception:
            return []
        if radius <= 1e-9 or spacing <= 1e-9:
            return []
        circ = 2.0 * math.pi * radius
        n = max(int(min_points), int(round(circ / spacing)))
        n = max(n, 8)
        pts = []
        for k in range(n):
            ang = phase + (2.0 * math.pi * float(k) / float(n))
            pts.append((cx + radius * math.cos(ang), cy + radius * math.sin(ang)))
        return pts

    def _sample_circle_ring_points_count(self, center, radius, count, phase=0.0):
        """Uniform angular sampling with explicit point count (used for aligned concentric rings)."""
        try:
            cx = float(center[0])
            cy = float(center[1])
            radius = float(radius)
            count = int(count)
            phase = float(phase)
        except Exception:
            return []
        if radius <= 1e-9 or count < 3:
            return []
        pts = []
        for k in range(count):
            ang = phase + (2.0 * math.pi * float(k) / float(count))
            pts.append((cx + radius * math.cos(ang), cy + radius * math.sin(ang)))
        return pts

    def _ring_matches_circle_spec(self, ring, spec):
        """Check whether a sampled boundary ring matches a circle to be handled explicitly."""
        if ring is None or spec is None:
            return False
        try:
            coords = list(ring.coords)
        except Exception:
            return False
        if not coords:
            return False
        pts = [tuple(map(float, pt[:2])) for pt in coords]
        if len(pts) > 1 and dist(pts[0], pts[-1]) <= 1e-9:
            pts = pts[:-1]
        if len(pts) < 8:
            return False
        try:
            cx = float(spec.get("center", (0.0, 0.0))[0])
            cy = float(spec.get("center", (0.0, 0.0))[1])
            r0 = float(spec.get("radius", 0.0))
            dx = float(spec.get("dx", 0.0) or 0.0)
        except Exception:
            return False
        if r0 <= 1e-9:
            return False
        try:
            minx, miny, maxx, maxy = ring.bounds
            ring_cx = 0.5 * (float(minx) + float(maxx))
            ring_cy = 0.5 * (float(miny) + float(maxy))
            center_tol = max(1e-3, 0.75 * max(dx, 1.0))
            if abs(ring_cx - cx) > center_tol or abs(ring_cy - cy) > center_tol:
                return False
        except Exception:
            pass
        step = max(1, len(pts) // 64)
        sample = pts[::step]
        if not sample:
            sample = pts
        rr = [math.hypot(x - cx, y - cy) for x, y in sample]
        if not rr:
            return False
        mean_r = sum(rr) / len(rr)
        max_dev = max(abs(r - r0) for r in rr)
        rel_dev = max_dev / max(r0, 1e-9)
        # Boolean ops often polygonize circle boundaries coarsely; allow a looser ring match here
        # so we skip the generic sampling ring and keep the explicit shared interface ring only.
        tol_abs = max(1e-3, 0.35 * max(dx, 1.0))
        return abs(mean_r - r0) <= tol_abs and rel_dev <= 0.20

    def _ring_matches_linework_spec(self, ring, spec):
        """Check if a ring matches a non-circular shared interface boundary to skip duplicate sampling."""
        if ring is None or spec is None or not isinstance(spec, dict):
            return False
        linework = spec.get("linework")
        if linework is None or getattr(linework, "is_empty", True):
            return False
        try:
            dx = float(spec.get("dx", 0.0) or 0.0)
        except Exception:
            dx = 0.0
        tol = max(1e-6, 0.30 * max(dx, 1.0))
        try:
            ring_len = float(getattr(ring, "length", 0.0) or 0.0)
        except Exception:
            ring_len = 0.0
        try:
            line_len = float(getattr(linework, "length", 0.0) or 0.0)
        except Exception:
            line_len = 0.0
        if ring_len <= 1e-9 or line_len <= 1e-9:
            return False
        # Avoid skipping a whole polygon boundary for a partial contact segment.
        if line_len < 0.80 * ring_len:
            return False
        try:
            d1 = float(ring.hausdorff_distance(linework))
        except Exception:
            try:
                d1 = float(ring.distance(linework))
            except Exception:
                d1 = 1e9
        try:
            d2 = float(linework.hausdorff_distance(ring))
        except Exception:
            try:
                d2 = float(linework.distance(ring))
            except Exception:
                d2 = 1e9
        return d1 <= tol and d2 <= tol

    def _ring_matches_skip_spec(self, ring, spec):
        """Unified matcher for generic boundary rings that should be skipped during meshing."""
        if ring is None or spec is None:
            return False
        if isinstance(spec, dict) and spec.get("kind") == "linework":
            return self._ring_matches_linework_spec(ring, spec)
        return self._ring_matches_circle_spec(ring, spec)

    def _hex_lattice_sample_circle(self, center, radius, spacing, angle=0.0):
        """Center-anchored hexagonal lattice clipped to a circle (uniform dx spacing)."""
        try:
            cx = float(center[0])
            cy = float(center[1])
            radius = float(radius)
            spacing = float(spacing)
            angle = float(angle)
        except Exception:
            return []
        if radius <= 1e-9 or spacing <= 1e-9:
            return []
        h = 0.5 * math.sqrt(3.0) * spacing
        if h <= 1e-12:
            return []
        max_j = int(math.ceil(radius / h)) + 2
        max_i = int(math.ceil(radius / spacing)) + 2
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rr_sq = radius * radius + 1e-12
        pts = []
        seen = set()
        for j in range(-max_j, max_j + 1):
            y_local = float(j) * h
            x_shift = 0.5 * spacing if (j & 1) else 0.0
            for i in range(-max_i, max_i + 1):
                x_local = float(i) * spacing + x_shift
                if x_local * x_local + y_local * y_local > rr_sq:
                    continue
                xr = x_local * cos_a - y_local * sin_a
                yr = x_local * sin_a + y_local * cos_a
                p = (cx + xr, cy + yr)
                key = (round(p[0], 12), round(p[1], 12))
                if key in seen:
                    continue
                seen.add(key)
                pts.append(p)
        return pts

    def _suppress_duplicate_circle_ring_boundary_points(
        self,
        boundary_points,
        interface_anchor_points,
        merge_specs,
    ):
        """
        Remove generic boundary points that duplicate an explicitly injected circle ring.

        This keeps one shared contour (instead of two interleaved near-coincident rings) so
        interface and inclusion triangulations connect cleanly across the same nodes.
        """
        if not boundary_points or not interface_anchor_points or not merge_specs:
            return boundary_points
        try:
            anchors = [tuple(map(float, p[:2])) for p in interface_anchor_points]
        except Exception:
            anchors = [tuple(p) for p in (interface_anchor_points if interface_anchor_points is not None else [])]
        if not anchors:
            return boundary_points

        anchor_set = set(anchors)
        prepared = []
        for spec in merge_specs:
            if not isinstance(spec, dict):
                continue
            try:
                cx = float(spec.get("center", (0.0, 0.0))[0])
                cy = float(spec.get("center", (0.0, 0.0))[1])
                r0 = float(spec.get("radius", 0.0) or 0.0)
                dx = float(spec.get("dx", 0.0) or 0.0)
            except Exception:
                continue
            if r0 <= 1e-9:
                continue
            dx = max(dx, 1.0)
            tol_rad = max(1e-6, 0.60 * dx)
            tol_snap_sq = (0.95 * dx) * (0.95 * dx)
            ring_anchors = []
            for a in anchors:
                rr = math.hypot(a[0] - cx, a[1] - cy)
                if abs(rr - r0) <= tol_rad:
                    ring_anchors.append(a)
            if not ring_anchors:
                continue
            prepared.append((cx, cy, r0, tol_rad, tol_snap_sq, ring_anchors))
        if not prepared:
            return boundary_points

        filtered = []
        for p in boundary_points:
            try:
                pt = (float(p[0]), float(p[1]))
            except Exception:
                filtered.append(p)
                continue
            if pt in anchor_set:
                filtered.append(pt)
                continue
            drop = False
            for cx, cy, r0, tol_rad, tol_snap_sq, ring_anchors in prepared:
                rr = math.hypot(pt[0] - cx, pt[1] - cy)
                if abs(rr - r0) > tol_rad:
                    continue
                # For explicit interface inner rings, any extra boundary points on the same circle
                # create a second interleaved contour and break green-blue connectivity.
                # Keep only the injected anchor ring on this radius.
                drop = True
                for a in ring_anchors:
                    dxp = pt[0] - a[0]
                    dyp = pt[1] - a[1]
                    if dxp * dxp + dyp * dyp <= tol_snap_sq:
                        # If it nearly coincides with an anchor, it's the duplicate we want to remove.
                        drop = True
                        break
                # If it is on the same ring but not close to any anchor, still drop it to avoid
                # preserving a second phase-shifted contour.
                break
            if not drop:
                filtered.append(pt)
        return filtered

    def _interface_band_buffer_distance(self, thickness, sample_dx, placement_mode="matrix_side"):
        """
        Buffer distance from interface centerline used to build the interface-band region.

        For matrix-side coatings, interface thickness is interpreted as a one-sided outward
        coating width (so thickness=dx gives an outer ring at ~R+dx around a circular inclusion).
        """
        try:
            sample_dx = float(sample_dx)
        except Exception:
            sample_dx = 0.0
        if sample_dx <= 0.0:
            sample_dx = 1.0
        try:
            thickness = float(thickness)
        except Exception:
            thickness = 0.0
        mode = str(placement_mode or "matrix_side").strip().lower()
        if mode == "matrix_side":
            band_dist = thickness if thickness > 0.0 else sample_dx
            # Keep a visible and numerically stable one-sided coating band.
            band_dist = max(band_dist, 0.50 * sample_dx)
            band_dist = min(band_dist, 2.50 * sample_dx)
            return band_dist
        band_dist = 0.5 * thickness if thickness > 0.0 else 0.5 * sample_dx
        band_dist = max(band_dist, 0.35 * sample_dx)
        band_dist = min(band_dist, 1.25 * sample_dx)
        return band_dist

    def _append_unique_point(self, out_list, out_set, pt):
        try:
            p = tuple(pt)
        except Exception:
            return
        if p in out_set:
            return
        out_set.add(p)
        out_list.append(p)

    def _sample_authoritative_boundary_ring_points(self, ring, spacing, min_points=12):
        """
        Sample a closed boundary loop with corner-aware local refinement.

        Straight segments stay close to the requested spacing, while short segments and sharp
        corners receive denser support so the constrained boundary survives triangulation cleanly.
        """
        if ring is None:
            return []
        try:
            length = float(getattr(ring, "length", 0.0) or 0.0)
        except Exception:
            length = 0.0
        try:
            spacing = float(spacing)
        except Exception:
            spacing = 0.0
        if length <= 0.0 or spacing <= 1e-12:
            return []
        try:
            coords = [tuple(map(float, pt[:2])) for pt in list(ring.coords)]
        except Exception:
            coords = []
        if len(coords) < 2:
            return []

        closed = False
        try:
            if len(coords) > 1 and dist(coords[0], coords[-1]) <= 1e-9:
                closed = True
        except Exception:
            if len(coords) > 1 and coords[0] == coords[-1]:
                closed = True
        if closed:
            coords = coords[:-1]
        if len(coords) < 2:
            return []

        target = max(1.0, spacing)
        try:
            min_points_i = max(0, int(min_points))
        except Exception:
            min_points_i = 12
        if min_points_i > 0 and length > 0.0:
            target = min(target, length / max(min_points_i, 1))
        target = max(1e-9, target)

        # High-vertex smooth curves (for example buffer-generated circles) should keep near-uniform
        # spacing instead of preserving every polyline vertex.
        if len(coords) > 48:
            try:
                return [tuple(map(float, pt[:2])) for pt in sample_ring(ring, target)]
            except Exception:
                return []

        min_target = max(1e-9, 0.35 * target)
        sharp_turn = math.radians(100.0)
        medium_turn = math.radians(65.0)
        n = len(coords)

        def _edge_len(a, b):
            try:
                return float(dist(a, b))
            except Exception:
                return 0.0

        def _local_vertex_spacing(idx):
            if not closed and idx in (0, n - 1):
                return target
            prev_pt = coords[(idx - 1) % n]
            pt = coords[idx]
            next_pt = coords[(idx + 1) % n]
            l1 = _edge_len(prev_pt, pt)
            l2 = _edge_len(pt, next_pt)
            if l1 <= 1e-12 or l2 <= 1e-12:
                return min_target
            v1 = (pt[0] - prev_pt[0], pt[1] - prev_pt[1])
            v2 = (next_pt[0] - pt[0], next_pt[1] - pt[1])
            try:
                dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)
            except Exception:
                dot = 1.0
            dot = max(-1.0, min(1.0, dot))
            turn = math.acos(dot)
            local = target
            if turn >= sharp_turn:
                local *= 0.55
            elif turn >= medium_turn:
                local *= 0.75
            if turn >= medium_turn:
                local = min(local, max(min_target, 0.50 * min(l1, l2)))
            return max(min_target, min(target, local))

        samples = []
        seg_count = n if closed else (n - 1)
        for seg_idx in range(seg_count):
            a = coords[seg_idx]
            b = coords[(seg_idx + 1) % n] if closed else coords[seg_idx + 1]
            seg_len = _edge_len(a, b)
            if seg_len <= 1e-12:
                continue
            local_target = target
            # Only tighten the entire segment spacing for genuinely short/tight features. Long
            # straight edges should stay close to the requested dx, with exact corner anchors.
            if seg_len <= 2.0 * target:
                local_target = min(
                    target,
                    _local_vertex_spacing(seg_idx),
                    _local_vertex_spacing((seg_idx + 1) % n if closed else seg_idx + 1),
                )
            local_target = max(min_target, min(local_target, seg_len))
            nseg = max(1, int(math.ceil(seg_len / local_target)))
            for j in range(nseg + 1):
                if seg_idx > 0 and j == 0:
                    continue
                t = float(j) / float(nseg)
                samples.append(
                    (
                        (1.0 - t) * float(a[0]) + t * float(b[0]),
                        (1.0 - t) * float(a[1]) + t * float(b[1]),
                    )
                )
        return samples

    def _dedupe_constraint_chains(self, chains, snap_dx):
        """
        Remove duplicate/overlapping chains that trace the same boundary support.

        This is applied before re-protecting chain points so a duplicate shared-boundary contour
        cannot be reintroduced into the meshing point set.
        """
        if not chains:
            return []
        try:
            snap_dx = float(snap_dx)
        except Exception:
            snap_dx = 0.0
        dup_chain_tol = max(1e-6, 0.30 * snap_dx) if snap_dx > 0 else 1e-4
        kept = []
        kept_geoms = []
        for chain in chains:
            if isinstance(chain, dict):
                seq = list(chain.get("points", []) if chain.get("points", []) is not None else [])
                closed = bool(chain.get("closed", False))
            else:
                seq = list(chain if chain is not None else [])
                closed = False
                chain = {"points": seq, "closed": False}
            if len(seq) < 2:
                continue
            chain_geom = None
            try:
                coords = [(float(p[0]), float(p[1])) for p in seq]
                if closed and len(coords) > 2 and coords[0] != coords[-1]:
                    coords = list(coords) + [coords[0]]
                if len(coords) >= 2:
                    chain_geom = LineString(coords)
            except Exception:
                chain_geom = None
            skip_chain = False
            if chain_geom is not None and not getattr(chain_geom, "is_empty", True):
                try:
                    chain_len = float(getattr(chain_geom, "length", 0.0) or 0.0)
                except Exception:
                    chain_len = 0.0
                for prev_geom in kept_geoms:
                    if prev_geom is None or getattr(prev_geom, "is_empty", True):
                        continue
                    try:
                        if bool(prev_geom.buffer(dup_chain_tol).covers(chain_geom)):
                            skip_chain = True
                            break
                        if bool(chain_geom.buffer(dup_chain_tol).covers(prev_geom)):
                            skip_chain = True
                            break
                    except Exception:
                        pass
                    try:
                        prev_len = float(getattr(prev_geom, "length", 0.0) or 0.0)
                    except Exception:
                        prev_len = 0.0
                    if chain_len <= 1e-9 or prev_len <= 1e-9:
                        continue
                    if min(chain_len, prev_len) < 0.80 * max(chain_len, prev_len):
                        continue
                    try:
                        d1 = float(chain_geom.hausdorff_distance(prev_geom))
                    except Exception:
                        d1 = 1e9
                    try:
                        d2 = float(prev_geom.hausdorff_distance(chain_geom))
                    except Exception:
                        d2 = 1e9
                    if max(d1, d2) <= dup_chain_tol:
                        skip_chain = True
                        break
            if skip_chain:
                continue
            kept.append(chain)
            if chain_geom is not None and not getattr(chain_geom, "is_empty", True):
                kept_geoms.append(chain_geom)
        return kept

    def _build_triangle_pslg_segments(self, points, chains, snap_dx):
        """
        Build Triangle PSLG segments by snapping ordered boundary/interface chains to mesh nodes.

        `chains` items may be dicts like {"points": [...], "closed": bool} or raw point lists.
        """
        if points is None:
            return None
        try:
            pts_arr = np.asarray(points, dtype=float)
        except Exception:
            return None
        if pts_arr.ndim != 2 or pts_arr.shape[0] < 2 or pts_arr.shape[1] < 2:
            return None
        if not chains:
            return None
        chains = self._dedupe_constraint_chains(chains, snap_dx)
        if not chains:
            return None

        exact_index = {}
        for i, p in enumerate(pts_arr):
            exact_index[(round(float(p[0]), 12), round(float(p[1]), 12))] = int(i)
        try:
            tree = cKDTree(pts_arr[:, :2])
        except Exception:
            tree = None

        try:
            snap_dx = float(snap_dx)
        except Exception:
            snap_dx = 0.0
        # Constraint chains should generally snap exactly (their points are protected before filtering).
        # Keep nearest-neighbor snapping only as a small fallback for numeric jitter.
        snap_tol = max(1e-8, 0.20 * snap_dx) if snap_dx > 0 else 1e-6
        max_gap = max(1e-8, 2.25 * snap_dx) if snap_dx > 0 else None
        segments = set()

        def _snap_index(pt):
            try:
                x = float(pt[0]); y = float(pt[1])
            except Exception:
                return None
            key = (round(x, 12), round(y, 12))
            idx = exact_index.get(key)
            if idx is not None:
                return int(idx)
            if tree is None:
                return None
            try:
                d, idx = tree.query((x, y), k=1)
            except Exception:
                return None
            try:
                if not np.isfinite(d) or float(d) > snap_tol:
                    return None
            except Exception:
                return None
            return int(idx)

        for chain in chains or []:
            if isinstance(chain, dict):
                seq = list(chain.get("points", []) if chain.get("points", []) is not None else [])
                closed = bool(chain.get("closed", False))
            else:
                seq = list(chain if chain is not None else [])
                closed = False
            if len(seq) < 2:
                continue
            idxs = []
            for pt in seq:
                idx = _snap_index(pt)
                if idx is None:
                    continue
                if idxs and idxs[-1] == idx:
                    continue
                idxs.append(idx)
            if len(idxs) < 2:
                continue
            if idxs[0] == idxs[-1]:
                idxs = idxs[:-1]
            if len(idxs) < 2:
                continue

            def _append_segment(a, b):
                if a == b:
                    return
                if max_gap is not None:
                    try:
                        gap = float(np.linalg.norm(pts_arr[int(a), :2] - pts_arr[int(b), :2]))
                    except Exception:
                        gap = None
                    if gap is None or not np.isfinite(gap) or gap > max_gap:
                        return
                aa = int(a)
                bb = int(b)
                segments.add((aa, bb) if aa < bb else (bb, aa))

            # Keep partially preserved chains so a few filtered support points do not erase the
            # entire boundary loop, but split across any oversized gaps.
            for a, b in zip(idxs, idxs[1:]):
                _append_segment(a, b)
            if closed and len(idxs) >= 3 and idxs[0] != idxs[-1]:
                _append_segment(idxs[-1], idxs[0])

        if not segments:
            return None
        return np.asarray(sorted(segments), dtype=int)

    def _build_triangle_region_seed_payload(self, part_effective_geoms, parts_to_mesh):
        """
        Build Triangle hole markers and region seeds from effective part geometry.

        Hole markers are emitted only for real voids, not for nested solid inclusions that are
        represented as a parent part's interior ring.
        """
        if not part_effective_geoms or not parts_to_mesh:
            return {"holes": None, "regions": None, "region_attr_map": {}}

        occupied_regions = []
        for part_id, geom in (part_effective_geoms or {}).items():
            if geom is None or getattr(geom, "is_empty", True):
                continue
            try:
                occupied_regions.append((int(part_id), prep(geom), geom))
            except Exception:
                occupied_regions.append((int(part_id), None, geom))

        holes = []
        hole_keys = set()
        regions = []
        region_attr_map = {}

        for part in parts_to_mesh or []:
            try:
                part_id = int(getattr(part, "id"))
            except Exception:
                continue
            geom = part_effective_geoms.get(part_id)
            if geom is None or getattr(geom, "is_empty", True):
                continue
            region_attr_map[part_id] = {
                "part_id": part_id,
                "material_id": getattr(part, "material_id", None),
            }
            try:
                geoms = [geom] if isinstance(geom, Polygon) else list(getattr(geom, "geoms", []))
            except Exception:
                geoms = [geom]
            for g in geoms:
                if g is None or getattr(g, "is_empty", True):
                    continue
                if getattr(g, "geom_type", "") != "Polygon":
                    continue
                try:
                    region_pt = g.representative_point()
                    if region_pt is not None and not getattr(region_pt, "is_empty", True):
                        regions.append([float(region_pt.x), float(region_pt.y), float(part_id), 0.0])
                except Exception:
                    pass
                for ring in getattr(g, "interiors", []):
                    try:
                        hole_poly = Polygon(ring)
                    except Exception:
                        hole_poly = None
                    if hole_poly is None or getattr(hole_poly, "is_empty", True):
                        continue
                    try:
                        hole_pt = hole_poly.representative_point()
                    except Exception:
                        hole_pt = None
                    if hole_pt is None or getattr(hole_pt, "is_empty", True):
                        continue
                    occupied = False
                    for other_part_id, other_prep, other_geom in occupied_regions:
                        if int(other_part_id) == int(part_id):
                            continue
                        try:
                            if other_prep is not None:
                                if bool(other_prep.covers(hole_pt)):
                                    occupied = True
                                    break
                            elif bool(other_geom.covers(hole_pt)):
                                occupied = True
                                break
                        except Exception:
                            continue
                    if occupied:
                        continue
                    key = (round(float(hole_pt.x), 12), round(float(hole_pt.y), 12))
                    if key in hole_keys:
                        continue
                    hole_keys.add(key)
                    holes.append([float(hole_pt.x), float(hole_pt.y)])

        return {
            "holes": np.asarray(holes, dtype=float) if holes else None,
            "regions": np.asarray(regions, dtype=float) if regions else None,
            "region_attr_map": region_attr_map,
        }

    def _validate_authoritative_segment_coverage(self, elements, segments):
        if segments is None:
            return {
                "authoritative_boundary_segment_count": 0,
                "authoritative_boundary_segments_preserved": 0,
                "authoritative_boundary_segments_missing": 0,
                "authoritative_boundary_node_count": 0,
                "authoritative_boundary_nodes_preserved": 0,
            }
        try:
            elem_arr = np.asarray(elements, dtype=int)
        except Exception:
            elem_arr = np.empty((0, 3), dtype=int)
        try:
            seg_arr = np.asarray(segments, dtype=int)
        except Exception:
            seg_arr = np.empty((0, 2), dtype=int)
        if seg_arr.ndim != 2 or seg_arr.shape[1] != 2 or len(seg_arr) == 0:
            return {
                "authoritative_boundary_segment_count": 0,
                "authoritative_boundary_segments_preserved": 0,
                "authoritative_boundary_segments_missing": 0,
                "authoritative_boundary_node_count": 0,
                "authoritative_boundary_nodes_preserved": 0,
            }
        edge_set = set()
        used_nodes = set()
        if elem_arr.ndim == 2 and elem_arr.shape[1] == 3:
            for tri in elem_arr:
                try:
                    a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
                except Exception:
                    continue
                used_nodes.update((a, b, c))
                for edge in ((a, b), (b, c), (c, a)):
                    aa, bb = sorted((int(edge[0]), int(edge[1])))
                    edge_set.add((aa, bb))
        missing = 0
        segment_nodes = set()
        for seg in seg_arr:
            try:
                a, b = int(seg[0]), int(seg[1])
            except Exception:
                continue
            aa, bb = sorted((a, b))
            segment_nodes.update((aa, bb))
            if (aa, bb) not in edge_set:
                missing += 1
        preserved_nodes = len(segment_nodes & used_nodes)
        total_segments = int(len(seg_arr))
        return {
            "authoritative_boundary_segment_count": total_segments,
            "authoritative_boundary_segments_preserved": max(0, total_segments - int(missing)),
            "authoritative_boundary_segments_missing": int(missing),
            "authoritative_boundary_node_count": int(len(segment_nodes)),
            "authoritative_boundary_nodes_preserved": int(preserved_nodes),
        }

    def _build_interface_mesh_sampling_payload(self, part_effective_geoms, spacing):
        """
        Shared interface sampling payload used by both sync and async meshing paths.

        Centralizes parsing of interface sampling outputs so behavior does not drift between paths.
        """
        interface_sampling = self._collect_interface_layer_sampling_points(
            part_effective_geoms,
            spacing,
        )
        skip_boundary_specs = list(interface_sampling.get("skip_boundary_specs", []) or [])
        circle_fill_parts = dict(interface_sampling.get("circle_fill_parts", {}) or {})
        circle_inner_merge_specs = list(interface_sampling.get("circle_inner_merge_specs", []) or [])
        constraint_chains = list(interface_sampling.get("constraint_chains", []) or [])

        circle_fill_specs = []
        for _pid, spec in circle_fill_parts.items():
            try:
                cx, cy = spec.get("center", (0.0, 0.0))
                cx = float(cx)
                cy = float(cy)
                r0 = float(spec.get("radius", 0.0) or 0.0)
                phase = float(spec.get("phase", 0.0) or 0.0)
            except Exception:
                continue
            fill_r = r0 - 0.55 * float(spacing)
            if fill_r <= 0.30 * float(spacing):
                continue
            circle_fill_specs.append(
                {
                    "part_id": int(_pid),
                    "center": (cx, cy),
                    "radius": r0,
                    "fill_radius": fill_r,
                    "phase": phase,
                }
            )
        return {
            "interface_sampling": interface_sampling,
            "skip_boundary_specs": skip_boundary_specs,
            "circle_inner_merge_specs": circle_inner_merge_specs,
            "circle_fill_specs": circle_fill_specs,
            "constraint_chains": constraint_chains,
        }

    def _merge_interface_sampling_payload_into_point_sets(
        self,
        payload,
        *,
        boundary_points,
        boundary_set,
        boundary_vertices,
        boundary_vertices_set,
        interior_points,
        interior_set,
    ):
        """
        Merge interface sampling points into meshing point buckets (sync/async shared behavior).
        Returns (boundary_points, boundary_vertices, interior_points) lists (mutated in place).
        """
        payload = payload or {}
        interface_sampling = payload.get("interface_sampling") or {}
        for pt in interface_sampling.get("boundary_points", []) or []:
            self._append_unique_point(boundary_points, boundary_set, tuple(pt))
        for pt in interface_sampling.get("protected_points", []) or []:
            p = tuple(pt)
            if p not in boundary_vertices_set:
                boundary_vertices_set.add(p)
                boundary_vertices.append(p)
        for pt in interface_sampling.get("interior_points", []) or []:
            self._append_unique_point(interior_points, interior_set, tuple(pt))

        circle_inner_merge_specs = list(payload.get("circle_inner_merge_specs", []) or [])
        if circle_inner_merge_specs:
            boundary_points = self._suppress_duplicate_circle_ring_boundary_points(
                boundary_points,
                interface_sampling.get("protected_points", []) or [],
                circle_inner_merge_specs,
            )
        linework_specs = [
            spec for spec in (payload.get("skip_boundary_specs", []) or [])
            if isinstance(spec, dict) and str(spec.get("kind", "")).lower() == "linework"
        ]
        if linework_specs:
            boundary_points = self._suppress_duplicate_linework_boundary_points(
                boundary_points,
                interface_sampling.get("protected_points", []) or [],
                linework_specs,
            )
        return boundary_points, boundary_vertices, interior_points

    def _suppress_duplicate_linework_boundary_points(
        self,
        boundary_points,
        interface_anchor_points,
        linework_specs,
    ):
        """
        Remove generic boundary points that lie on a shared interface linework already sampled explicitly.

        Shape-general counterpart to the circle-ring cleanup.
        """
        if not boundary_points or not interface_anchor_points or not linework_specs:
            return boundary_points
        try:
            anchors = [tuple(map(float, p[:2])) for p in interface_anchor_points]
        except Exception:
            anchors = [tuple(p) for p in (interface_anchor_points if interface_anchor_points is not None else [])]
        if not anchors:
            return boundary_points
        anchor_set = set(anchors)
        prepared_specs = []
        for spec in linework_specs:
            if not isinstance(spec, dict):
                continue
            linework = spec.get("linework")
            if linework is None or getattr(linework, "is_empty", True):
                continue
            try:
                dx = float(spec.get("dx", 0.0) or 0.0)
            except Exception:
                dx = 0.0
            tol = max(1e-6, 0.35 * max(dx, 1.0))
            prepared_specs.append((linework, tol))
        if not prepared_specs:
            return boundary_points

        filtered = []
        for p in boundary_points:
            try:
                pt = (float(p[0]), float(p[1]))
            except Exception:
                filtered.append(p)
                continue
            if pt in anchor_set:
                filtered.append(pt)
                continue
            drop = False
            p_obj = None
            for linework, tol in prepared_specs:
                try:
                    if p_obj is None:
                        p_obj = Point(pt)
                    d = float(linework.distance(p_obj))
                except Exception:
                    continue
                if d <= tol:
                    drop = True
                    break
            if not drop:
                filtered.append(pt)
        return filtered

    def _protect_constraint_chain_points(
        self,
        chains,
        *,
        boundary_points,
        boundary_set,
        boundary_vertices,
        boundary_vertices_set,
    ):
        """
        Ensure constrained chain points survive min-spacing filtering.

        Triangle PSLG segments require their endpoints to exist in the final point set; otherwise
        later snapping can distort segments or create invalid crossings.
        """
        if not chains:
            return boundary_points, boundary_vertices
        for chain in chains:
            if isinstance(chain, dict):
                seq = list(chain.get("points", []) if chain.get("points", []) is not None else [])
            else:
                seq = list(chain if chain is not None else [])
            for pt in seq:
                try:
                    p = (float(pt[0]), float(pt[1]))
                except Exception:
                    continue
                self._append_unique_point(boundary_points, boundary_set, p)
                if p not in boundary_vertices_set:
                    boundary_vertices_set.add(p)
                    boundary_vertices.append(p)
        return boundary_points, boundary_vertices

    def _collect_interface_layer_sampling_points(self, part_effective_geoms, spacing):
        """
        Generate explicit interface-band sampling points for meshing.
        Returns dict with boundary/protected/interior point lists and info messages.
        """
        result = {
            "boundary_points": [],
            "protected_points": [],
            "interior_points": [],
            "messages": [],
            "skip_boundary_specs": [],
            "circle_fill_parts": {},
            "circle_inner_merge_specs": [],
            "constraint_chains": [],
        }
        if not self.interfaces:
            return result
        try:
            spacing = float(spacing)
        except Exception:
            spacing = 0.0
        if spacing <= 0.0:
            return result
        bset = set()
        pset = set()
        iset = set()
        boundary_points = []
        protected_points = []
        interior_points = []
        constraint_chains = []
        iface_count_with_points = 0
        for iface in self.interfaces:
            try:
                p1_id = int(getattr(iface, "part1_id"))
                p2_id = int(getattr(iface, "part2_id"))
            except Exception:
                continue
            if p1_id == p2_id:
                continue
            g1 = part_effective_geoms.get(p1_id)
            g2 = part_effective_geoms.get(p2_id)
            if g1 is None or g2 is None:
                continue
            placement_mode = self._interface_placement_mode(iface)
            side_info = self._interface_matrix_inclusion_parts(iface, part_effective_geoms)
            matrix_geom = side_info.get("matrix_geom") if side_info else None
            centerline = self._interface_centerline_geometry(iface, part_effective_geoms)
            if centerline is None or getattr(centerline, "is_empty", True):
                continue
            sample_dx = self._interface_effective_sampling_dx(iface, spacing)
            if sample_dx <= 0:
                sample_dx = spacing
            sample_dx = max(sample_dx, 0.85 * spacing)
            thickness = self._interface_effective_thickness(iface, sample_dx)
            band_dist = self._interface_band_buffer_distance(thickness, sample_dx, placement_mode)
            if band_dist <= 0:
                continue
            try:
                band = centerline.buffer(band_dist)
            except Exception:
                continue
            if band is None or getattr(band, "is_empty", True):
                continue
            try:
                pair_region = unary_union([g1, g2])
                if pair_region is not None and not getattr(pair_region, "is_empty", True):
                    try:
                        pair_region = pair_region.buffer(0)
                    except Exception:
                        pass
                    band = band.intersection(pair_region)
            except Exception:
                pass
            if band is None or getattr(band, "is_empty", True):
                continue
            band_region = band
            if placement_mode == "matrix_side" and matrix_geom is not None and not getattr(matrix_geom, "is_empty", True):
                try:
                    band_region = band.intersection(matrix_geom)
                except Exception:
                    band_region = band
            if band_region is None or getattr(band_region, "is_empty", True):
                continue

            def _append_unique(out_list, out_set, pt):
                try:
                    p = (float(pt[0]), float(pt[1]))
                except Exception:
                    return
                key = (round(p[0], 12), round(p[1], 12))
                if key in out_set:
                    return
                out_set.add(key)
                out_list.append(p)

            def _append_constraint_chain(points_seq, closed=False):
                try:
                    seq = [(float(p[0]), float(p[1])) for p in (points_seq if points_seq is not None else [])]
                except Exception:
                    seq = []
                if len(seq) < 2:
                    return
                compact = []
                for p in seq:
                    if compact:
                        try:
                            if dist(compact[-1], p) <= 1e-9:
                                continue
                        except Exception:
                            if compact[-1] == p:
                                continue
                    compact.append(p)
                if len(compact) < 2:
                    return
                if closed and len(compact) > 2:
                    try:
                        if dist(compact[0], compact[-1]) <= 1e-9:
                            compact = compact[:-1]
                    except Exception:
                        pass
                if len(compact) < 2:
                    return
                constraint_chains.append({"points": compact, "closed": bool(closed)})

            def _append_line_vertices(
                line,
                protect=False,
                corner_angle_deg=30.0,
                anchor_points=None,
                snap_dx=None,
            ):
                if line is None:
                    return
                try:
                    coords = [tuple(map(float, pt[:2])) for pt in list(line.coords)]
                except Exception:
                    return
                if not coords:
                    return
                is_closed = False
                if len(coords) > 1:
                    try:
                        is_closed = bool(dist(coords[0], coords[-1]) <= 1e-9)
                    except Exception:
                        is_closed = bool(coords[0] == coords[-1])
                    if is_closed:
                        coords = coords[:-1]
                if not coords:
                    return

                angle_thresh = math.radians(max(0.0, float(corner_angle_deg)))
                keep_pts = []
                try:
                    anchor_pts = [tuple(map(float, p[:2])) for p in (anchor_points if anchor_points is not None else [])]
                except Exception:
                    anchor_pts = [tuple(p) for p in (anchor_points if anchor_points is not None else [])]
                try:
                    snap_dx = float(snap_dx)
                except Exception:
                    snap_dx = 0.0
                near_anchor_tol = max(1e-6, 0.40 * snap_dx) if snap_dx > 0.0 else 1e-6

                def _push_keep(pt):
                    if not keep_pts:
                        keep_pts.append(pt)
                        return
                    try:
                        if dist(keep_pts[-1], pt) <= 1e-9:
                            return
                    except Exception:
                        if keep_pts[-1] == pt:
                            return
                    keep_pts.append(pt)

                def _near_anchor(pt):
                    if not anchor_pts:
                        return False
                    for q in anchor_pts:
                        try:
                            if dist(pt, q) <= near_anchor_tol:
                                return True
                        except Exception:
                            if pt == q:
                                return True
                    return False

                if is_closed and len(coords) >= 3:
                    n = len(coords)
                    for i in range(n):
                        prev_pt = coords[(i - 1) % n]
                        pt = coords[i]
                        next_pt = coords[(i + 1) % n]
                        v1 = (pt[0] - prev_pt[0], pt[1] - prev_pt[1])
                        v2 = (next_pt[0] - pt[0], next_pt[1] - pt[1])
                        l1 = math.hypot(v1[0], v1[1])
                        l2 = math.hypot(v2[0], v2[1])
                        if l1 <= 1e-9 or l2 <= 1e-9:
                            _push_keep(pt)
                            continue
                        dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)
                        dot = max(-1.0, min(1.0, dot))
                        turn = math.acos(dot)
                        if turn >= angle_thresh:
                            _push_keep(pt)
                    if not keep_pts:
                        _push_keep(coords[0])
                else:
                    _push_keep(coords[0])
                    for i in range(1, max(1, len(coords) - 1)):
                        if i >= len(coords) - 1:
                            break
                        prev_pt = coords[i - 1]
                        pt = coords[i]
                        next_pt = coords[i + 1]
                        v1 = (pt[0] - prev_pt[0], pt[1] - prev_pt[1])
                        v2 = (next_pt[0] - pt[0], next_pt[1] - pt[1])
                        l1 = math.hypot(v1[0], v1[1])
                        l2 = math.hypot(v2[0], v2[1])
                        if l1 <= 1e-9 or l2 <= 1e-9:
                            _push_keep(pt)
                            continue
                        dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)
                        dot = max(-1.0, min(1.0, dot))
                        turn = math.acos(dot)
                        if turn >= angle_thresh:
                            _push_keep(pt)
                    if len(coords) > 1:
                        _push_keep(coords[-1])

                for p in keep_pts:
                    if _near_anchor(p):
                        continue
                    _append_unique(boundary_points, bset, p)
                    if protect:
                        _append_unique(protected_points, pset, p)

            thin_single_layer = thickness > 0.0 and thickness <= 1.35 * sample_dx
            circle_info = None
            inclusion_id = None
            inclusion_geom = side_info.get("inclusion_geom") if side_info else None
            nested_matrix_inclusion = False
            try:
                if (
                    placement_mode == "matrix_side"
                    and matrix_geom is not None
                    and not getattr(matrix_geom, "is_empty", True)
                    and inclusion_geom is not None
                    and not getattr(inclusion_geom, "is_empty", True)
                ):
                    nested_matrix_inclusion = bool(matrix_geom.covers(inclusion_geom))
            except Exception:
                nested_matrix_inclusion = False
            # Exact contour-ring sampling for simple circle inclusions (uses part sketch metadata).
            # Falls back automatically if validation fails, so non-circular/ambiguous cases are safe.
            circle_contour_enabled = bool(getattr(self, "mesh_circle_contour_sampling_enabled", True))
            if circle_contour_enabled and placement_mode == "matrix_side" and side_info is not None:
                inclusion_id = side_info.get("inclusion_id")
                inclusion_part = None
                if inclusion_id is not None:
                    try:
                        iid = int(inclusion_id)
                        inclusion_part = next((p for p in self.parts if int(getattr(p, "id", -1)) == iid), None)
                    except Exception:
                        inclusion_part = None
                circle_info = self._circle_spec_from_part_sketch_meta(inclusion_part, inclusion_geom)
            added_before = len(boundary_points)
            handled_circle_band = False
            if thin_single_layer and placement_mode == "matrix_side" and circle_info is not None:
                cx, cy = circle_info.get("center", (0.0, 0.0))
                inner_r = float(circle_info.get("radius", 0.0) or 0.0)
                phase = float(circle_info.get("phase", 0.0) or 0.0)
                outer_r = inner_r + float(band_dist)
                support_r = outer_r + float(sample_dx)
                inner_support_r = inner_r - float(sample_dx)
                if inner_r > 1e-9 and outer_r > inner_r + 1e-9 and support_r > outer_r + 1e-9:
                    # Use one shared angular count so triangles between rings are well-shaped/aligned.
                    circ_inner = max(0.0, 2.0 * math.pi * inner_r)
                    n_ring = int(math.floor(circ_inner / max(float(sample_dx), 1e-9)))
                    n_ring = max(12, n_ring)
                    inner_pts = self._sample_circle_ring_points_count((cx, cy), inner_r, n_ring, phase=phase)
                    outer_pts = self._sample_circle_ring_points_count((cx, cy), outer_r, n_ring, phase=phase)
                    support_pts = self._sample_circle_ring_points_count((cx, cy), support_r, n_ring, phase=phase)
                    inner_support_pts = []
                    if inner_support_r > max(1e-9, 0.60 * float(sample_dx)):
                        circ_inner_support = max(0.0, 2.0 * math.pi * inner_support_r)
                        n_inner_support = int(
                            math.floor(circ_inner_support / max(float(sample_dx), 1e-9))
                        )
                        n_inner_support = max(12, n_inner_support)
                        inner_support_pts = self._sample_circle_ring_points_count(
                            (cx, cy),
                            inner_support_r,
                            n_inner_support,
                            phase=phase,
                        )
                    geom_tol = max(1e-6, 0.60 * float(sample_dx))
                    accepted_inner = []
                    accepted_outer = []
                    accepted_support = []
                    accepted_inner_support = []
                    band_relaxed = band_region
                    matrix_relaxed = matrix_geom
                    inclusion_relaxed = inclusion_geom
                    try:
                        band_relaxed = band_region.buffer(geom_tol)
                    except Exception:
                        band_relaxed = band_region
                    try:
                        if matrix_geom is not None and not getattr(matrix_geom, "is_empty", True):
                            matrix_relaxed = matrix_geom.buffer(geom_tol)
                    except Exception:
                        matrix_relaxed = matrix_geom
                    try:
                        if inclusion_geom is not None and not getattr(inclusion_geom, "is_empty", True):
                            inclusion_relaxed = inclusion_geom.buffer(geom_tol)
                    except Exception:
                        inclusion_relaxed = inclusion_geom
                    for pt in inner_pts:
                        try:
                            p_obj = Point(pt)
                        except Exception:
                            p_obj = None
                        ok = True
                        if p_obj is not None and inclusion_relaxed is not None and not getattr(inclusion_relaxed, "is_empty", True):
                            try:
                                ok = bool(inclusion_relaxed.covers(p_obj))
                            except Exception:
                                ok = True
                        if ok:
                            accepted_inner.append(pt)
                    for pt in outer_pts:
                        try:
                            p_obj = Point(pt)
                        except Exception:
                            p_obj = None
                        ok = True
                        if p_obj is not None:
                            try:
                                if band_relaxed is not None and not getattr(band_relaxed, "is_empty", True):
                                    ok = bool(band_relaxed.covers(p_obj))
                                if ok and matrix_relaxed is not None and not getattr(matrix_relaxed, "is_empty", True):
                                    ok = bool(matrix_relaxed.covers(p_obj))
                            except Exception:
                                ok = True
                        if ok:
                            accepted_outer.append(pt)
                    for pt in inner_support_pts:
                        try:
                            p_obj = Point(pt)
                        except Exception:
                            p_obj = None
                        ok = True
                        if p_obj is not None and inclusion_relaxed is not None and not getattr(inclusion_relaxed, "is_empty", True):
                            try:
                                ok = bool(inclusion_relaxed.covers(p_obj))
                            except Exception:
                                ok = True
                        if ok:
                            accepted_inner_support.append(pt)
                    for pt in support_pts:
                        try:
                            p_obj = Point(pt)
                        except Exception:
                            p_obj = None
                        ok = True
                        if p_obj is not None and matrix_relaxed is not None and not getattr(matrix_relaxed, "is_empty", True):
                            try:
                                ok = bool(matrix_relaxed.covers(p_obj))
                            except Exception:
                                ok = True
                        if ok and p_obj is not None and band_relaxed is not None and not getattr(band_relaxed, "is_empty", True):
                            try:
                                # Support ring should sit just outside the interface band.
                                ok = not bool(band_relaxed.covers(p_obj))
                            except Exception:
                                ok = ok
                        if ok:
                            accepted_support.append(pt)

                    inner_ok = len(accepted_inner) >= max(8, int(0.70 * max(1, len(inner_pts))))
                    outer_ok = len(accepted_outer) >= max(8, int(0.70 * max(1, len(outer_pts))))
                    support_ok = len(accepted_support) >= max(8, int(0.60 * max(1, len(support_pts))))
                    if inner_ok and outer_ok and support_ok:
                        for pt in accepted_inner:
                            _append_unique(boundary_points, bset, pt)
                            _append_unique(protected_points, pset, pt)
                        for pt in accepted_inner_support:
                            _append_unique(boundary_points, bset, pt)
                        for pt in accepted_outer:
                            _append_unique(boundary_points, bset, pt)
                            _append_unique(protected_points, pset, pt)
                        for pt in accepted_support:
                            _append_unique(boundary_points, bset, pt)
                        _append_constraint_chain(accepted_inner, closed=True)
                        _append_constraint_chain(accepted_outer, closed=True)
                        _append_constraint_chain(accepted_support, closed=True)
                        if accepted_inner_support:
                            _append_constraint_chain(accepted_inner_support, closed=True)
                        if inclusion_id not in (None, ""):
                            try:
                                inc_id = int(inclusion_id)
                                result["skip_boundary_specs"].append(
                                    {
                                        "part_id": inc_id,
                                        "center": (float(cx), float(cy)),
                                        "radius": float(inner_r),
                                        "dx": float(sample_dx),
                                        "iface_id": int(getattr(iface, "id", -1)),
                                    }
                                )
                                result["circle_inner_merge_specs"].append(
                                    {
                                        "center": (float(cx), float(cy)),
                                        "radius": float(inner_r),
                                        "dx": float(sample_dx),
                                        "iface_id": int(getattr(iface, "id", -1)),
                                    }
                                )
                                # Also skip generic sampling of the outer interface contour if it
                                # coincides with a polygonized circle in a future geometry path.
                                result["skip_boundary_specs"].append(
                                    {
                                        "part_id": inc_id,
                                        "center": (float(cx), float(cy)),
                                        "radius": float(outer_r),
                                        "dx": float(sample_dx),
                                        "iface_id": int(getattr(iface, "id", -1)),
                                    }
                                )
                            except Exception:
                                pass
                        handled_circle_band = True

            if (
                thin_single_layer
                and placement_mode == "matrix_side"
                and not handled_circle_band
                and nested_matrix_inclusion
            ):
                skip_linework = None
                try:
                    if inclusion_geom is not None and not getattr(inclusion_geom, "is_empty", True):
                        skip_linework = inclusion_geom.boundary
                except Exception:
                    skip_linework = None
                if skip_linework is None or getattr(skip_linework, "is_empty", True):
                    skip_linework = centerline
                if skip_linework is not None and not getattr(skip_linework, "is_empty", True):
                    result["skip_boundary_specs"].append(
                        {
                            "kind": "linework",
                            "linework": skip_linework,
                            "dx": float(sample_dx),
                            "iface_id": int(getattr(iface, "id", -1)),
                        }
                    )

            if not handled_circle_band:
                for line in self._iter_line_geometries(centerline):
                    try:
                        line_pts = self._sample_line_geometry_points(line, sample_dx, min_points=12)
                        _append_line_vertices(
                            line,
                            protect=True,
                            anchor_points=line_pts,
                            snap_dx=sample_dx,
                        )
                        for pt in line_pts:
                            _append_unique(boundary_points, bset, pt)
                            _append_unique(protected_points, pset, pt)
                        _append_constraint_chain(line_pts, closed=(getattr(line, "is_ring", False) or getattr(line, "geom_type", "") == "LinearRing"))
                    except Exception:
                        continue

            if thin_single_layer and placement_mode == "matrix_side" and not handled_circle_band:
                if nested_matrix_inclusion and inclusion_geom is not None and not getattr(inclusion_geom, "is_empty", True):
                    try:
                        inner_support_geom = inclusion_geom.buffer(-float(sample_dx))
                    except Exception:
                        inner_support_geom = None
                    if inner_support_geom is not None and not getattr(inner_support_geom, "is_empty", True):
                        for line in self._iter_line_geometries(getattr(inner_support_geom, "boundary", None)):
                            try:
                                d_line = float(centerline.distance(line))
                            except Exception:
                                d_line = float(sample_dx)
                            if d_line < max(0.25 * float(sample_dx), 1e-6):
                                continue
                            try:
                                line_pts = self._sample_line_geometry_points(line, sample_dx, min_points=12)
                                _append_line_vertices(
                                    line,
                                    protect=False,
                                    anchor_points=line_pts,
                                    snap_dx=sample_dx,
                                )
                                for pt in line_pts:
                                    _append_unique(boundary_points, bset, pt)
                                _append_constraint_chain(line_pts, closed=(getattr(line, "is_ring", False) or getattr(line, "geom_type", "") == "LinearRing"))
                            except Exception:
                                continue
                # Single coating layer: reuse shared boundary nodes + one outer curve in matrix.
                outer_lines = []
                fallback_lines = []
                for line in self._iter_line_geometries(getattr(band_region, "boundary", None)):
                    fallback_lines.append(line)
                    try:
                        # Avoid re-sampling the shared boundary (already added via centerline/all_boundaries).
                        # Keep only the offset outer curve(s) of the matrix-side coating band.
                        d_line = float(centerline.distance(line))
                    except Exception:
                        d_line = 0.0
                    if d_line >= max(0.35 * float(band_dist), 0.40 * float(sample_dx)):
                        outer_lines.append(line)
                lines_to_sample = outer_lines if outer_lines else fallback_lines
                for line in lines_to_sample:
                    try:
                        line_pts = self._sample_line_geometry_points(line, sample_dx, min_points=12)
                        _append_line_vertices(
                            line,
                            protect=True,
                            anchor_points=line_pts,
                            snap_dx=sample_dx,
                        )
                        for pt in line_pts:
                            _append_unique(boundary_points, bset, pt)
                            _append_unique(protected_points, pset, pt)
                        _append_constraint_chain(line_pts, closed=(getattr(line, "is_ring", False) or getattr(line, "geom_type", "") == "LinearRing"))
                    except Exception:
                        continue
            elif not thin_single_layer:
                # Offset curves are cullable (not protected) so they don't stack too densely.
                for line in self._iter_line_geometries(getattr(band_region, "boundary", None)):
                    try:
                        line_pts = self._sample_line_geometry_points(line, sample_dx, min_points=12)
                        for pt in line_pts:
                            _append_unique(boundary_points, bset, pt)
                        _append_constraint_chain(line_pts, closed=(getattr(line, "is_ring", False) or getattr(line, "geom_type", "") == "LinearRing"))
                    except Exception:
                        continue
                # Interior samples are useful mainly for thicker interface bands.
                try:
                    band_geoms = (
                        [band_region]
                        if isinstance(band_region, Polygon)
                        else list(getattr(band_region, "geoms", []))
                    )
                except Exception:
                    band_geoms = [band_region]
                for g in band_geoms:
                    if g is None or getattr(g, "is_empty", True):
                        continue
                    if getattr(g, "geom_type", "") != "Polygon":
                        continue
                    try:
                        pts = poisson_sample(g, sample_dx)
                    except Exception:
                        pts = []
                    for pt in pts:
                        _append_unique(interior_points, iset, pt)

            if len(boundary_points) > added_before:
                iface_count_with_points += 1

        if iface_count_with_points > 0:
            result["messages"].append(
                f"Interaction layer sampling added around {iface_count_with_points} interaction(s)."
            )
        result["boundary_points"] = boundary_points
        result["protected_points"] = protected_points
        result["interior_points"] = interior_points
        result["constraint_chains"] = constraint_chains
        return result

    def _interface_preview_signature(self):
        try:
            iface_sig = tuple(
                sorted(
                    (
                        int(getattr(iface, "id", -1)),
                        int(getattr(iface, "part1_id", -1)),
                        int(getattr(iface, "part2_id", -1)),
                        str(getattr(iface, "interface_type", "")),
                        str(getattr(iface, "material_id", "")),
                        str(getattr(iface, "thickness", "")),
                        str(getattr(iface, "target_dx", "")),
                        str(getattr(iface, "placement_mode", getattr(Interface, "DEFAULT_PLACEMENT_MODE", "matrix_side"))),
                    )
                    for iface in (self.interfaces or [])
                )
            )
        except Exception:
            iface_sig = ()
        try:
            elem_len = len(self.global_elements) if self.global_elements is not None else 0
        except Exception:
            elem_len = 0
        try:
            node_len = len(self.global_nodes) if self.global_nodes is not None else 0
        except Exception:
            node_len = 0
        return (
            id(self.global_nodes),
            node_len,
            id(self.global_elements),
            elem_len,
            id(self.element_part_map),
            len(self.element_part_map or []),
            iface_sig,
        )

    def _get_interface_preview_rows(self):
        """Cached interface-classified rows for 2D mesh preview coloring."""
        sig = self._interface_preview_signature()
        if self._interface_preview_cache is not None and self._interface_preview_cache_sig == sig:
            return self._interface_preview_cache
        rows, _summary = self._build_connections_export_rows()
        self._interface_preview_cache = rows
        self._interface_preview_cache_sig = sig
        return rows

    def _validate_interface_definitions(self):
        """Frontend-only QA checks for interface-layer definitions."""
        warnings = []
        pair_seen = {}
        for iface in getattr(self, "interfaces", []) or []:
            try:
                pair = tuple(sorted((int(iface.part1_id), int(iface.part2_id))))
            except Exception:
                pair = None
            if pair:
                pair_seen.setdefault(pair, []).append(int(getattr(iface, "id", -1)))
            mat_id = getattr(iface, "material_id", None)
            if mat_id in (None, "", -1):
                warnings.append(
                    f"Interaction {getattr(iface, 'id', '?')}: missing interaction material_id."
                )
            layer_mode = self._interface_layer_mode(iface)
            try:
                thickness = float(getattr(iface, "thickness", 0.0) or 0.0)
            except Exception:
                thickness = 0.0
            try:
                target_dx = float(getattr(iface, "target_dx", 0.0) or 0.0)
            except Exception:
                target_dx = 0.0
            if thickness <= 0.0 and layer_mode != "single_layer_ring":
                warnings.append(
                    f"Interaction {getattr(iface, 'id', '?')}: thickness should be > 0."
                )
            if target_dx <= 0.0 and layer_mode != "single_layer_ring":
                warnings.append(
                    f"Interaction {getattr(iface, 'id', '?')}: particle spacing should be > 0."
                )
            if layer_mode == "single_layer_ring":
                try:
                    iface.status = "OK" if mat_id not in (None, "", -1) else "WARN:NoMaterial"
                except Exception:
                    pass
            elif thickness > 0.0 and target_dx > 0.0:
                ratio = thickness / target_dx
                if ratio < 0.6 or ratio > 1.8:
                    warnings.append(
                        f"Interaction {getattr(iface, 'id', '?')}: thickness/spacing={ratio:.2f} "
                        "is outside single-layer target range (~1.0)."
                    )
                try:
                    iface.status = "OK" if 0.6 <= ratio <= 1.8 and mat_id not in (None, "", -1) else f"WARN:t/dx={ratio:.2f}"
                except Exception:
                    pass
        for pair, iface_ids in pair_seen.items():
            if len(iface_ids) > 1:
                warnings.append(
                    f"Duplicate interaction definitions for part pair {pair}: ids={sorted(iface_ids)}."
                )
        return warnings

    def _build_connections_export_rows(self):
        """Build canonical connections.csv rows and frontend QA summary."""
        element_map = {item["element_idx"]: item for item in (self.element_part_map or [])}
        live_parts = {}
        for part in (self.parts or []):
            part_id = getattr(part, "id", None)
            if part_id is None:
                continue
            live_parts[part_id] = part
            try:
                live_parts[int(part_id)] = part
            except Exception:
                pass
        rows = []
        seen_triangles = {}
        duplicate_count = 0
        degenerate_count = 0
        edge_usage = {}
        elements = getattr(self, "global_elements", [])
        if elements is None:
            elements = []
        nodes_array = np.asarray(getattr(self, "global_nodes", np.array([])))
        for elem_idx, tri in enumerate(elements):
            if tri is None or len(tri) < 3:
                continue
            try:
                p1, p2, p3 = int(tri[0]), int(tri[1]), int(tri[2])
            except Exception:
                continue
            tri_key = tuple(sorted((p1, p2, p3)))
            if len(set(tri_key)) < 3:
                degenerate_count += 1
                continue
            if tri_key in seen_triangles:
                duplicate_count += 1
                continue
            seen_triangles[tri_key] = elem_idx
            part_info = dict(element_map.get(elem_idx, {}))
            part_id = part_info.get("part_id", "")
            live_part = live_parts.get(part_id)
            zone_kind = str(part_info.get("zone_kind", "part") or "part").lower()
            material_id = part_info.get("material_id", "")
            if zone_kind != "interface" and live_part is not None:
                # Always stamp part-owned triangles from the current project-state part assignment.
                material_id = getattr(live_part, "material_id", material_id)
            row = {
                "triangle_id": len(rows),
                "p1": p1,
                "p2": p2,
                "p3": p3,
                "part_id": part_id,
                "material_id": material_id,
                "zone_kind": part_info.get("zone_kind", "part"),
                "interface_id": part_info.get("interface_id", ""),
                "meta": part_info.get("meta", ""),
                "_element_idx": int(elem_idx),
            }
            try:
                tri_pts = nodes_array[[p1, p2, p3]]
                centroid = np.mean(tri_pts, axis=0)
                row["_centroid"] = (float(centroid[0]), float(centroid[1]))
                e12 = float(np.linalg.norm(tri_pts[0] - tri_pts[1]))
                e23 = float(np.linalg.norm(tri_pts[1] - tri_pts[2]))
                e31 = float(np.linalg.norm(tri_pts[2] - tri_pts[0]))
                row["_avg_edge"] = (e12 + e23 + e31) / 3.0
            except Exception:
                row["_centroid"] = None
                row["_avg_edge"] = None
            rows.append(row)
            for edge in ((p1, p2), (p2, p3), (p3, p1)):
                ekey = tuple(sorted(edge))
                edge_usage.setdefault(ekey, []).append(row["triangle_id"])

        warnings = []
        interface_stats = {}
        if rows and getattr(self, "interfaces", None):
            part_map = {int(p.id): p for p in (self.parts or []) if getattr(p, "id", None) is not None}
            effective_geom_cache = {}

            def _effective_geom_for_part(part_id):
                try:
                    part_id = int(part_id)
                except Exception:
                    return None
                if part_id in effective_geom_cache:
                    return effective_geom_cache.get(part_id)
                part = part_map.get(part_id)
                geom = getattr(part, "geometry", None) if part is not None else None
                if geom is None or getattr(geom, "is_empty", True):
                    effective_geom_cache[part_id] = None
                    return None
                eff = geom
                try:
                    for child in self.get_child_parts(part):
                        cgeom = getattr(child, "geometry", None)
                        if cgeom is None or getattr(cgeom, "is_empty", True):
                            continue
                        try:
                            eff = eff.difference(cgeom).buffer(0)
                        except Exception:
                            pass
                except Exception:
                    pass
                if eff is None or getattr(eff, "is_empty", True):
                    effective_geom_cache[part_id] = None
                else:
                    effective_geom_cache[part_id] = eff
                return effective_geom_cache.get(part_id)

            centerline_cache = {}

            def _interface_centerline(iface):
                iface_id = int(getattr(iface, "id", -1))
                if iface_id in centerline_cache:
                    return centerline_cache.get(iface_id)
                g1 = _effective_geom_for_part(getattr(iface, "part1_id", None))
                g2 = _effective_geom_for_part(getattr(iface, "part2_id", None))
                centerline = None
                def _has_linework(geom):
                    if geom is None or getattr(geom, "is_empty", True):
                        return False
                    try:
                        return bool(self._iter_line_geometries(geom))
                    except Exception:
                        return False
                if g1 is not None and g2 is not None:
                    try:
                        centerline = g1.boundary.intersection(g2.boundary)
                    except Exception:
                        centerline = None
                    if not _has_linework(centerline):
                        try:
                            overlap = g1.intersection(g2)
                        except Exception:
                            overlap = None
                        if overlap is not None and not getattr(overlap, "is_empty", True):
                            try:
                                centerline = overlap.boundary
                            except Exception:
                                centerline = overlap
                    if not _has_linework(centerline):
                        try:
                            if g1.covers(g2):
                                centerline = g2.boundary
                            elif g2.covers(g1):
                                centerline = g1.boundary
                        except Exception:
                            pass
                    if not _has_linework(centerline):
                        try:
                            side_info = self._interface_matrix_inclusion_parts(
                                iface,
                                {
                                    int(getattr(iface, "part1_id")): g1,
                                    int(getattr(iface, "part2_id")): g2,
                                },
                            )
                        except Exception:
                            side_info = None
                        inc_geom = side_info.get("inclusion_geom") if side_info else None
                        if inc_geom is not None and not getattr(inc_geom, "is_empty", True):
                            try:
                                centerline = inc_geom.boundary
                            except Exception:
                                centerline = inc_geom
                    if not _has_linework(centerline):
                        centerline = None
                centerline_cache[iface_id] = centerline
                return centerline

            rows_by_part = {}
            for row in rows:
                try:
                    pid = int(row.get("part_id"))
                except Exception:
                    continue
                rows_by_part.setdefault(pid, []).append(row)
            rows_by_triangle_id = {}
            for row in rows:
                try:
                    rows_by_triangle_id[int(row.get("triangle_id"))] = row
                except Exception:
                    continue

            tri_geom_cache = {}

            def _triangle_geom_for_row(row):
                try:
                    tri_id = int(row.get("triangle_id"))
                except Exception:
                    tri_id = id(row)
                cached = tri_geom_cache.get(tri_id)
                if cached is not None:
                    return cached
                poly = None
                area = 0.0
                centroid_pt = None
                try:
                    p1 = int(row.get("p1"))
                    p2 = int(row.get("p2"))
                    p3 = int(row.get("p3"))
                    tri_pts = nodes_array[[p1, p2, p3]]
                    poly = Polygon(tri_pts)
                    if poly is not None and not getattr(poly, "is_empty", True):
                        area = float(getattr(poly, "area", 0.0) or 0.0)
                except Exception:
                    poly = None
                    area = 0.0
                centroid = row.get("_centroid")
                if centroid is not None:
                    try:
                        centroid_pt = Point(float(centroid[0]), float(centroid[1]))
                    except Exception:
                        centroid_pt = None
                out = (poly, area, centroid_pt)
                tri_geom_cache[tri_id] = out
                return out

            for iface in (self.interfaces or []):
                iface_id = int(getattr(iface, "id", -1))
                try:
                    p1_id = int(getattr(iface, "part1_id"))
                    p2_id = int(getattr(iface, "part2_id"))
                except Exception:
                    warnings.append(f"Interaction {iface_id}: invalid part ids.")
                    continue
                try:
                    iface_mat = int(getattr(iface, "material_id"))
                except Exception:
                    iface_mat = None
                if iface_mat is None:
                    warnings.append(f"Interaction {iface_id}: missing interaction material_id; skipping interaction-zone export.")
                    try:
                        iface.status = "WARN:NoMaterial"
                    except Exception:
                        pass
                    continue
                placement_mode = self._interface_placement_mode(iface)
                layer_mode = self._interface_layer_mode(iface)
                side_info = self._interface_matrix_inclusion_parts(
                    iface,
                    {
                        p1_id: _effective_geom_for_part(p1_id),
                        p2_id: _effective_geom_for_part(p2_id),
                    },
                )
                matrix_part_id = side_info.get("matrix_id") if side_info else None
                sample_dx = self._interface_effective_sampling_dx(
                    iface,
                    float(getattr(self, "last_mesh_dx", 0.0) or 0.0),
                )
                if sample_dx <= 0.0:
                    sample_dx = 1.0
                thickness = self._interface_effective_thickness(iface, sample_dx)
                target_dx = float(sample_dx)
                band_dist = self._interface_band_buffer_distance(thickness, sample_dx, placement_mode)
                thresholds = []
                for v in (
                    band_dist,
                    max(band_dist, 0.50 * target_dx) if target_dx > 0 else 0.0,
                    max(band_dist, 0.75 * target_dx) if target_dx > 0 else 0.0,
                    max(band_dist, 1.00 * target_dx) if target_dx > 0 else 0.0,
                    max(band_dist, 1.25 * target_dx) if target_dx > 0 else 0.0,
                ):
                    try:
                        v = float(v)
                    except Exception:
                        v = 0.0
                    if v <= 0:
                        continue
                    if not thresholds or abs(thresholds[-1] - v) > 1e-12:
                        thresholds.append(v)
                if not thresholds:
                    warnings.append(f"Interaction {iface_id}: effective thickness/particle spacing not set (>0); skipping interaction-zone export.")
                    try:
                        iface.status = "WARN:NoDX/Thickness"
                    except Exception:
                        pass
                    continue

                centerline = _interface_centerline(iface)
                if centerline is None or getattr(centerline, "is_empty", True):
                    warnings.append(
                        f"Interaction {iface_id}: unable to detect shared boundary/centerline for parts {p1_id}-{p2_id}."
                    )
                    try:
                        iface.status = "WARN:NoBoundary"
                    except Exception:
                        pass
                    continue

                g1_eff = _effective_geom_for_part(p1_id)
                g2_eff = _effective_geom_for_part(p2_id)
                if g1_eff is None or g2_eff is None:
                    warnings.append(f"Interaction {iface_id}: missing effective geometry for parts {p1_id}-{p2_id}.")
                    try:
                        iface.status = "WARN:NoGeom"
                    except Exception:
                        pass
                    continue
                try:
                    band_region = centerline.buffer(float(band_dist))
                except Exception:
                    band_region = None
                if band_region is None or getattr(band_region, "is_empty", True):
                    warnings.append(f"Interaction {iface_id}: failed to build interaction band geometry.")
                    try:
                        iface.status = "WARN:NoBand"
                    except Exception:
                        pass
                    continue
                try:
                    pair_region = unary_union([g1_eff, g2_eff])
                except Exception:
                    pair_region = None
                if pair_region is not None and not getattr(pair_region, "is_empty", True):
                    try:
                        pair_region = pair_region.buffer(0)
                    except Exception:
                        pass
                    try:
                        band_region = band_region.intersection(pair_region)
                    except Exception:
                        pass
                if placement_mode == "matrix_side":
                    matrix_geom = side_info.get("matrix_geom") if side_info else None
                    if matrix_geom is not None and not getattr(matrix_geom, "is_empty", True):
                        try:
                            band_region = band_region.intersection(matrix_geom)
                        except Exception:
                            pass
                if band_region is None or getattr(band_region, "is_empty", True):
                    warnings.append(f"Interaction {iface_id}: interaction band region is empty after clipping.")
                    try:
                        iface.status = "WARN:EmptyBand"
                    except Exception:
                        pass
                    continue
                band_tol = max(1e-8, 1e-3 * float(sample_dx))
                try:
                    band_region_relaxed = band_region.buffer(band_tol)
                except Exception:
                    band_region_relaxed = band_region
                try:
                    band_region_prep = prep(band_region_relaxed)
                except Exception:
                    band_region_prep = None
                centerline_core = None
                centerline_core_prep = None
                try:
                    core_tol = max(band_tol, 0.18 * float(sample_dx))
                except Exception:
                    core_tol = band_tol
                try:
                    if centerline is not None and not getattr(centerline, "is_empty", True):
                        centerline_core = centerline.buffer(core_tol)
                except Exception:
                    centerline_core = None
                if centerline_core is not None and not getattr(centerline_core, "is_empty", True):
                    try:
                        centerline_core_prep = prep(centerline_core)
                    except Exception:
                        centerline_core_prep = None

                if placement_mode == "matrix_side" and matrix_part_id in (p1_id, p2_id):
                    candidate_rows = list(rows_by_part.get(matrix_part_id, []))
                else:
                    candidate_rows = list(rows_by_part.get(p1_id, [])) + list(rows_by_part.get(p2_id, []))
                if not candidate_rows:
                    warnings.append(f"Interaction {iface_id}: no candidate helper triangles found for parts {p1_id}-{p2_id}.")
                    try:
                        iface.status = "WARN:NoCandidates"
                    except Exception:
                        pass
                    continue

                conflict_ids = set()
                selected_pairs = []
                side_counts = {p1_id: 0, p2_id: 0}
                overlap_ratio_sum = 0.0
                overlap_ratio_count = 0
                inclusion_part_id = side_info.get("inclusion_id") if side_info else None
                single_layer_matrix_side = bool(
                    placement_mode == "matrix_side"
                    and str(layer_mode).lower() == "single_layer_ring"
                    and matrix_part_id not in (None, "")
                    and inclusion_part_id not in (None, "")
                )
                for row in candidate_rows:
                    existing_zone = str(row.get("zone_kind", "")).lower()
                    existing_iface = row.get("interface_id")
                    if existing_zone == "interface" and existing_iface not in ("", None, iface_id):
                        try:
                            conflict_ids.add(int(row["triangle_id"]))
                        except Exception:
                            pass
                        continue
                    tri_poly, tri_area, centroid_pt = _triangle_geom_for_row(row)
                    if tri_poly is None or getattr(tri_poly, "is_empty", True) or tri_area <= 1e-12:
                        continue
                    shares_pair_boundary_edge = False
                    if (
                        placement_mode == "matrix_side"
                        and inclusion_part_id not in (None, "")
                        and int(row.get("part_id", -1)) == int(matrix_part_id)
                    ):
                        try:
                            row_edges = (
                                tuple(sorted((int(row.get("p1")), int(row.get("p2"))))),
                                tuple(sorted((int(row.get("p2")), int(row.get("p3"))))),
                                tuple(sorted((int(row.get("p3")), int(row.get("p1"))))),
                            )
                        except Exception:
                            row_edges = ()
                        for ekey in row_edges:
                            for nbr_tid in edge_usage.get(ekey, []):
                                try:
                                    nbr_tid = int(nbr_tid)
                                except Exception:
                                    continue
                                try:
                                    if nbr_tid == int(row.get("triangle_id")):
                                        continue
                                except Exception:
                                    pass
                                nbr_row = rows_by_triangle_id.get(nbr_tid)
                                if not nbr_row:
                                    continue
                                try:
                                    nbr_pid = int(nbr_row.get("part_id"))
                                except Exception:
                                    continue
                                if nbr_pid == int(inclusion_part_id):
                                    shares_pair_boundary_edge = True
                                    break
                            if shares_pair_boundary_edge:
                                break
                    try:
                        if (
                            band_region_prep is not None
                            and not band_region_prep.intersects(tri_poly)
                            and not shares_pair_boundary_edge
                        ):
                            continue
                    except Exception:
                        pass
                    try:
                        overlap_geom = tri_poly.intersection(band_region_relaxed)
                        overlap_area = float(getattr(overlap_geom, "area", 0.0) or 0.0)
                    except Exception:
                        overlap_area = 0.0
                    if overlap_area <= 1e-12 and not shares_pair_boundary_edge:
                        continue
                    overlap_ratio = max(0.0, min(1.0, overlap_area / max(tri_area, 1e-12)))
                    centroid_in_band = False
                    if centroid_pt is not None:
                        try:
                            if band_region_prep is not None:
                                centroid_in_band = bool(band_region_prep.covers(centroid_pt))
                            else:
                                centroid_in_band = bool(band_region_relaxed.covers(centroid_pt))
                        except Exception:
                            centroid_in_band = False
                    touches_centerline = False
                    if centerline_core is not None and not getattr(centerline_core, "is_empty", True):
                        try:
                            if centerline_core_prep is not None:
                                touches_centerline = bool(centerline_core_prep.intersects(tri_poly))
                            else:
                                touches_centerline = bool(centerline_core.intersects(tri_poly))
                        except Exception:
                            touches_centerline = False
                    # Exact annulus classification:
                    # Prefer triangles mostly inside the interface band, but keep some boundary-cut
                    # triangles when their centroid lies in-band so the annulus stays continuous.
                    keep_boundary_cut = (
                        placement_mode == "matrix_side"
                        and touches_centerline
                        and overlap_ratio >= 0.03
                    )
                    keep_single_layer_fallback = False
                    if (
                        single_layer_matrix_side
                        and touches_centerline
                        and int(row.get("part_id", -1)) == int(matrix_part_id)
                        and overlap_area > 1e-12
                    ):
                        try:
                            tri_avg_edge = float(row.get("_avg_edge") or 0.0)
                        except Exception:
                            tri_avg_edge = 0.0
                        # Generic non-circular inclusions can yield boundary-cut matrix triangles with
                        # very low overlap ratio in a one-layer band; keep a local, size-limited strip.
                        edge_size_ok = True
                        if target_dx > 0.0 and tri_avg_edge > 0.0:
                            edge_size_ok = tri_avg_edge <= max(2.75 * target_dx, 3.0 * float(band_dist))
                        near_if = False
                        if centroid_pt is not None:
                            try:
                                near_if = float(centerline.distance(centroid_pt)) <= max(
                                    1.15 * float(band_dist),
                                    0.95 * float(target_dx) if target_dx > 0.0 else float(band_dist),
                                )
                            except Exception:
                                near_if = False
                        keep_single_layer_fallback = bool(
                            edge_size_ok and (near_if or centroid_in_band or shares_pair_boundary_edge)
                        )
                    if (
                        overlap_ratio < 0.45
                        and not (centroid_in_band and overlap_ratio >= 0.12)
                        and not keep_boundary_cut
                        and not keep_single_layer_fallback
                        and not shares_pair_boundary_edge
                    ):
                        continue
                    if (
                        single_layer_matrix_side
                        and int(row.get("part_id", -1)) == int(matrix_part_id)
                    ):
                        max_vertex_dist = None
                        try:
                            row_node_ids = (
                                int(row.get("p1")),
                                int(row.get("p2")),
                                int(row.get("p3")),
                            )
                            max_vertex_dist = max(
                                float(
                                    centerline.distance(
                                        Point(
                                            float(nodes_array[node_id, 0]),
                                            float(nodes_array[node_id, 1]),
                                        )
                                    )
                                )
                                for node_id in row_node_ids
                            )
                        except Exception:
                            max_vertex_dist = None
                        vertex_band_tol = max(
                            band_tol,
                            0.20 * float(target_dx) if target_dx > 0.0 else band_tol,
                        )
                        if (
                            max_vertex_dist is not None
                            and max_vertex_dist > (float(band_dist) + float(vertex_band_tol))
                        ):
                            continue
                    dist_to_if = 0.0
                    if centroid_pt is not None:
                        try:
                            dist_to_if = float(centerline.distance(centroid_pt))
                        except Exception:
                            dist_to_if = 0.0
                    selected_pairs.append((row, dist_to_if, overlap_ratio))
                    overlap_ratio_sum += overlap_ratio
                    overlap_ratio_count += 1
                    try:
                        side_counts[int(row.get("part_id"))] += 1
                    except Exception:
                        pass

                # Topology-enforced backstop: if any edge is still shared directly between the
                # matrix and inclusion for this interface pair, promote the matrix-side triangle on
                # that edge into the interface zone even if overlap heuristics missed it.
                if placement_mode == "matrix_side" and matrix_part_id not in (None, "") and inclusion_part_id not in (None, ""):
                    try:
                        selected_ids = {
                            int(sel_row.get("triangle_id"))
                            for sel_row, _sel_dist, _sel_ov in selected_pairs
                        }
                    except Exception:
                        selected_ids = set()
                    for ekey, tri_ids_on_edge in (edge_usage or {}).items():
                        if not tri_ids_on_edge or len(tri_ids_on_edge) < 2:
                            continue
                        matrix_row = None
                        inc_row = None
                        for tid in tri_ids_on_edge:
                            try:
                                tid_i = int(tid)
                            except Exception:
                                continue
                            row_obj = rows_by_triangle_id.get(tid_i)
                            if not row_obj:
                                continue
                            try:
                                pid_i = int(row_obj.get("part_id"))
                            except Exception:
                                continue
                            if pid_i == int(matrix_part_id):
                                matrix_row = row_obj
                            elif pid_i == int(inclusion_part_id):
                                inc_row = row_obj
                        if matrix_row is None or inc_row is None:
                            continue
                        try:
                            tri_id = int(matrix_row.get("triangle_id"))
                        except Exception:
                            continue
                        if tri_id in selected_ids:
                            continue
                        existing_zone = str(matrix_row.get("zone_kind", "")).lower()
                        existing_iface = matrix_row.get("interface_id")
                        if existing_zone == "interface" and existing_iface not in ("", None, iface_id):
                            try:
                                conflict_ids.add(tri_id)
                            except Exception:
                                pass
                            continue
                        _tri_poly_force, _tri_area_force, centroid_force = _triangle_geom_for_row(matrix_row)
                        dist_to_if_force = 0.0
                        if centroid_force is not None:
                            try:
                                dist_to_if_force = float(centerline.distance(centroid_force))
                            except Exception:
                                dist_to_if_force = 0.0
                        selected_pairs.append((matrix_row, dist_to_if_force, 0.0))
                        selected_ids.add(tri_id)
                        overlap_ratio_count += 1
                        try:
                            side_counts[int(matrix_row.get("part_id"))] += 1
                        except Exception:
                            pass

                if not selected_pairs:
                    warnings.append(
                        f"Interaction {iface_id}: no helper triangles found in the interaction band region (parts {p1_id}-{p2_id})."
                    )
                    try:
                        iface.status = "WARN:NoZoneTriangles"
                    except Exception:
                        pass
                    continue

                used_thr = float(band_dist)
                applied = 0
                edge_lengths = []
                max_dist = 0.0
                tri_ids = []
                for row, dist_to_if, overlap_ratio in selected_pairs:
                    existing_zone = str(row.get("zone_kind", "")).lower()
                    existing_iface = row.get("interface_id")
                    if existing_zone == "interface" and existing_iface not in ("", None, iface_id):
                        continue
                    base_mat = row.get("material_id", "")
                    row["zone_kind"] = "interface"
                    row["interface_id"] = iface_id
                    row["material_id"] = iface_mat
                    meta_items = []
                    if row.get("meta"):
                        meta_items.append(str(row.get("meta")))
                    if base_mat not in ("", None):
                        meta_items.append(f"base_material_id={base_mat}")
                    meta_items.append(f"iface_place={placement_mode}")
                    meta_items.append(f"iface_dist={dist_to_if:.6g}")
                    meta_items.append(f"iface_overlap={overlap_ratio:.4f}")
                    row["meta"] = ";".join(meta_items)
                    applied += 1
                    tri_ids.append(int(row["triangle_id"]))
                    try:
                        max_dist = max(max_dist, float(dist_to_if))
                    except Exception:
                        pass
                    avg_edge = row.get("_avg_edge")
                    if avg_edge is not None:
                        edge_lengths.append(float(avg_edge))

                if not applied:
                    warnings.append(f"Interaction {iface_id}: candidate helper triangles conflicted with existing interaction zones.")
                    try:
                        iface.status = "WARN:Conflict"
                    except Exception:
                        pass
                    continue

                one_sided = side_counts.get(p1_id, 0) == 0 or side_counts.get(p2_id, 0) == 0
                expected_one_sided = bool(placement_mode == "matrix_side")
                if one_sided:
                    if not expected_one_sided:
                        warnings.append(
                            f"Interaction {iface_id}: interaction zone is one-sided "
                            f"(part {p1_id}: {side_counts.get(p1_id,0)}, part {p2_id}: {side_counts.get(p2_id,0)} triangles)."
                        )
                if conflict_ids:
                    warnings.append(
                        f"Interaction {iface_id}: skipped {len(conflict_ids)} helper triangles due to overlap with another interaction zone."
                    )
                if target_dx > 0 and edge_lengths:
                    avg_edge_len = sum(edge_lengths) / len(edge_lengths)
                    if avg_edge_len < 0.4 * target_dx or avg_edge_len > 2.0 * target_dx:
                        warnings.append(
                            f"Interaction {iface_id}: avg interaction helper-triangle edge ≈ {avg_edge_len:.4g} "
                            f"vs target spacing {target_dx:.4g}."
                        )
                else:
                    avg_edge_len = None

                try:
                    ratio_txt = ""
                    if target_dx > 0:
                        ratio_txt = f" t/dx={thickness/target_dx:.2f}"
                    side_txt = f"{side_counts.get(p1_id,0)}/{side_counts.get(p2_id,0)}"
                    iface.status = (
                        f"{'WARN' if (one_sided and not expected_one_sided) else 'OK'}:{applied}tri "
                        f"sides={side_txt} place={placement_mode} mode={layer_mode} band={used_thr:.3g}{ratio_txt}"
                    )
                except Exception:
                    pass

                interface_stats[iface_id] = {
                    "triangles": applied,
                    "threshold": used_thr,
                    "side_counts": side_counts,
                    "placement_mode": placement_mode,
                    "avg_edge": avg_edge_len,
                    "max_centroid_distance": max_dist,
                    "avg_overlap_ratio": (
                        (overlap_ratio_sum / overlap_ratio_count) if overlap_ratio_count > 0 else None
                    ),
                    "triangle_ids": tri_ids,
                }

        non_manifold_edges = [e for e, tri_ids in edge_usage.items() if len(tri_ids) > 2]
        interface_rows = [
            r for r in rows
            if str(r.get("zone_kind", "")).lower() == "interface" or r.get("interface_id") not in ("", None)
        ]
        if duplicate_count:
            warnings.append(f"Skipped {duplicate_count} duplicate triangles in connections.csv export.")
        if degenerate_count:
            warnings.append(f"Skipped {degenerate_count} degenerate triangles in connections.csv export.")
        if non_manifold_edges:
            warnings.append(
                f"Detected {len(non_manifold_edges)} non-manifold edges (used by >2 triangles)."
            )
        mesh_validation = self.get_mesh_validation_stats()
        try:
            missing_boundary_segments = int(mesh_validation.get("authoritative_boundary_segments_missing", 0) or 0)
        except Exception:
            missing_boundary_segments = 0
        if missing_boundary_segments > 0:
            warnings.append(
                f"Mesh validation: {missing_boundary_segments} authoritative boundary segments are missing from exported connectivity."
            )
        if self.interfaces and not interface_rows:
            warnings.append(
                "INFO: Interactions are defined, but no explicit interaction-layer helper triangles exist in the particle connection set yet "
                "(connections.csv currently contains part helper triangles only)."
            )
        return rows, {
            "duplicate_triangles": duplicate_count,
            "degenerate_triangles": degenerate_count,
            "non_manifold_edges": len(non_manifold_edges),
            "interface_triangle_rows": len(interface_rows),
            "interface_stats": interface_stats,
            "warnings": warnings,
        }

    def _report_frontend_mesh_export_warnings(self, messages, silent=False):
        msgs = [str(m).strip() for m in (messages or []) if str(m).strip()]
        if not msgs:
            return
        summary = " | ".join(msgs[:3])
        try:
            self._announce_status(f"Export QA: {summary}", 7000)
        except Exception:
            pass
        for msg in msgs:
            print(f"[export_csv][QA] {msg}")
        severe = any(not m.startswith("INFO:") for m in msgs)
        if not silent and severe:
            QMessageBox.warning(
                self,
                "Particles / Interactions QA",
                "\n".join(msgs[:12]),
            )

    def load_mesh_from_files(self, particles_path, connections_path):
        try:
            particles_df = pd.read_csv(particles_path, comment="#", skipinitialspace=True)
            particles_df.columns = particles_df.columns.str.strip()
            if "x" not in particles_df.columns or "y" not in particles_df.columns:
                return False
            particles = particles_df[["x", "y"]].to_numpy()

            connections_df = pd.read_csv(connections_path, comment="#", skipinitialspace=True)
            connections_df.columns = connections_df.columns.str.strip()
            if {"p1", "p2", "p3"}.issubset(connections_df.columns):
                connections = connections_df[["p1", "p2", "p3"]].astype(int).to_numpy()
            elif {"n1", "n2", "n3"}.issubset(connections_df.columns):
                connections = connections_df[["n1", "n2", "n3"]].astype(int).to_numpy()
            else:
                return False

            self.global_nodes = particles
            self.global_elements = connections
            self.part_meshes = {}

            self.element_part_map = []
            material_col = None
            if "material_id" in connections_df.columns:
                material_col = "material_id"
            elif "material_serial" in connections_df.columns:
                material_col = "material_serial"
            if "part_id" in connections_df.columns and material_col:
                for idx, row in connections_df.iterrows():
                    if pd.isna(row["part_id"]) or pd.isna(row[material_col]):
                        continue
                    item = {
                        "element_idx": int(idx),
                        "part_id": int(row["part_id"]),
                        "material_id": int(row[material_col]),
                    }
                    if "zone_kind" in connections_df.columns and not pd.isna(row.get("zone_kind")):
                        item["zone_kind"] = str(row.get("zone_kind"))
                    if "interface_id" in connections_df.columns and not pd.isna(row.get("interface_id")):
                        try:
                            item["interface_id"] = int(row.get("interface_id"))
                        except Exception:
                            item["interface_id"] = row.get("interface_id")
                    self.element_part_map.append(item)
            return True
        except Exception:
            return False

    def build_3d_mesh(self):
        if self._cad_kernel_ready():
            if self._build_3d_mesh_cad():
                return True
        if self.global_nodes is None or len(self.global_nodes) == 0:
            return False
        if self.global_elements is None or len(self.global_elements) == 0:
            return False
        if self.extrude_height <= 0 or self.extrude_layers < 1:
            return False

        nodes_2d = np.asarray(self.global_nodes)
        n2d = len(nodes_2d)
        layers = int(self.extrude_layers)
        z_levels = np.linspace(0.0, float(self.extrude_height), layers + 1)

        nodes_3d = np.zeros(((layers + 1) * n2d, 3), dtype=float)
        for li, z in enumerate(z_levels):
            start = li * n2d
            nodes_3d[start : start + n2d, 0:2] = nodes_2d
            nodes_3d[start : start + n2d, 2] = z

        part_map = {m["element_idx"]: m for m in self.element_part_map}
        tets = []
        tet_part_map = []

        for tri_idx, tri in enumerate(self.global_elements):
            if len(tri) < 3:
                continue
            n1, n2, n3 = int(tri[0]), int(tri[1]), int(tri[2])
            part_info = part_map.get(tri_idx)
            for li in range(layers):
                base = li * n2d
                top = (li + 1) * n2d
                b1, b2, b3 = base + n1, base + n2, base + n3
                t1, t2, t3 = top + n1, top + n2, top + n3
                tet_defs = [
                    (b1, b2, b3, t1),
                    (t1, t2, t3, b2),
                    (t1, t3, b2, b3),
                ]
                for tet in tet_defs:
                    tets.append(tet)
                    if part_info:
                        tet_part_map.append(
                            {
                                "element_idx": len(tets) - 1,
                                "part_id": part_info["part_id"],
                                "material_id": part_info["material_id"],
                            }
                        )

        if not tets:
            return False

        self.global_nodes_3d = nodes_3d
        self.global_elements_3d = np.array(tets, dtype=int)
        self.element_part_map_3d = tet_part_map
        self.mesh3dUpdated.emit(self.global_nodes_3d, self.global_elements_3d)
        return True

    def _cad_kernel_ready(self):
        return (
            self.project_mode == "3d"
            and self.use_cad_kernel
            and self.cad_kernel is not None
            and self.cad_kernel.available()
        )

    def _cad_shape_from_shapely(self, geom):
        if geom is None or geom.is_empty or not self._cad_kernel_ready():
            return None
        shapes = []
        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        else:
            return None
        for poly in polys:
            exterior = list(poly.exterior.coords)
            holes = [list(ring.coords) for ring in poly.interiors]
            face = self.cad_kernel.face_from_polygon(exterior, holes)
            solid = self.cad_kernel.extrude(face, float(self.extrude_height))
            if solid is not None:
                shapes.append(solid)
        if not shapes:
            return None
        shape = shapes[0]
        for other in shapes[1:]:
            fused = self.cad_kernel.boolean(shape, other, "union")
            if fused is not None:
                shape = fused
        return shape

    def _build_cad_assembly_shape(self):
        if not self._cad_kernel_ready():
            return None
        shape_map = {}
        for part in self.parts:
            shape = getattr(part, "cad_shape", None)
            if shape is None and part.geometry:
                shape = self._cad_shape_from_shapely(part.geometry)
            if shape is not None:
                shape_map[part.id] = shape
        for part in self.parts:
            if not part.is_void:
                continue
            parent_id = part.parent_id
            if parent_id is None:
                continue
            parent_shape = shape_map.get(parent_id)
            cut_shape = shape_map.get(part.id)
            if parent_shape is None or cut_shape is None:
                continue
            cut_result = self.cad_kernel.boolean(parent_shape, cut_shape, "cut")
            if cut_result is not None:
                shape_map[parent_id] = cut_result
        solids = []
        for part in self.parts:
            if part.is_void:
                continue
            shape = shape_map.get(part.id)
            if shape is not None:
                solids.append(shape)
        if not solids:
            return None
        shape = solids[0]
        for other in solids[1:]:
            fused = self.cad_kernel.boolean(shape, other, "union")
            if fused is not None:
                shape = fused
        return shape

    def _build_3d_mesh_cad(self):
        shape = self._build_cad_assembly_shape()
        if shape is None:
            return False
        nodes, faces = self.cad_kernel.tessellate(shape, linear_deflection=0.5, angular_deflection=0.5)
        if nodes is None or faces is None or len(nodes) == 0 or len(faces) == 0:
            return False
        self.global_nodes_3d = np.asarray(nodes, dtype=float)
        self.global_elements_3d = np.asarray(faces, dtype=int)
        self.element_part_map_3d = []
        self.mesh3dUpdated.emit(self.global_nodes_3d, self.global_elements_3d)
        return True

    def _sync_cad_shape(self, part):
        if part is None or not self._cad_kernel_ready():
            return
        if part.geometry is None or part.geometry.is_empty:
            return
        part.cad_shape = self._cad_shape_from_shapely(part.geometry)

    # --- Export Logic ---
    def export_csv(self, silent=False, export_mode="full", async_mesh=True, force_remesh=False):
        try:
            from solver_exporter import export_project_to_workspace
        except Exception:
            return self._export_csv_impl(
                silent=silent,
                export_mode=export_mode,
                async_mesh=async_mesh,
                force_remesh=force_remesh,
            )

        state = getattr(self, "project_state", None)
        if state is None or not isinstance(getattr(state, "solver_settings", None), dict):
            state = self._require_project_state()
        state.solver_settings["_sketch_view"] = self
        state.solver_settings["_export_options"] = {
            "silent": bool(silent),
            "export_mode": str(export_mode),
            "async_mesh": bool(async_mesh),
            "force_remesh": bool(force_remesh),
        }
        return bool(export_project_to_workspace(state, self._workspace_path()))

    def _export_csv_impl(self, silent=False, export_mode="full", async_mesh=True, force_remesh=False):
        from solver_exporter import export_csv_impl
        return export_csv_impl(
            self,
            silent=silent,
            export_mode=export_mode,
            async_mesh=async_mesh,
            force_remesh=force_remesh,
        )

    def mousePressEvent(self, e):
        p = self.mapToScene(e.position().toPoint())
        pt = (p.x(), p.y())
        raw_pt = pt
        if self._partition_pick_active:
            if e.button() == Qt.LeftButton:
                self._partition_points.append((float(pt[0]), float(pt[1])))
                if len(self._partition_points) >= 2:
                    self._commit_partition()
                else:
                    self.viewport().update()
                e.accept()
                return
            if e.button() == Qt.RightButton:
                self.cancel_partition_pick()
                e.accept()
                return
        if self._vertex_seed_pick_active:
            if e.button() == Qt.LeftButton:
                v = self._vertex_at_scene_point(pt, self._vertex_seed_pick_tol())
                self._finish_vertex_seed_pick(v)
                e.accept()
                return
            if e.button() == Qt.RightButton:
                self.cancel_vertex_seed_pick()
                e.accept()
                return
        if self._edge_seed_pick_mode is not None:
            if e.button() == Qt.LeftButton:
                edge = self._edge_at_scene_point(pt, self._edge_seed_pick_tol())
                if edge is not None:
                    pid, s, eend = edge
                    ref = {"part_id": pid, "start": s, "end": eend}
                    if self._edge_seed_pick_mode == "single":
                        self._edge_seed_picked = [ref]
                        self._finish_edge_seed_pick()
                        e.accept()
                        return
                    # multi: toggle in/out of the picked set
                    existing_idx = -1
                    for i, p_ref in enumerate(self._edge_seed_picked):
                        if (
                            p_ref["part_id"] == pid
                            and p_ref["start"] == s
                            and p_ref["end"] == eend
                        ):
                            existing_idx = i
                            break
                    if existing_idx >= 0:
                        del self._edge_seed_picked[existing_idx]
                    else:
                        self._edge_seed_picked.append(ref)
                    self.viewport().update()
                e.accept()
                return
            if e.button() == Qt.RightButton:
                # Right-click in multi-mode ends the pick (matches polygon UX).
                if self._edge_seed_pick_mode == "multi":
                    self._finish_edge_seed_pick()
                    e.accept()
                    return
        if self._zone_draw_active:
            if e.button() == Qt.LeftButton:
                self._zone_draw_points.append((float(pt[0]), float(pt[1])))
                self.viewport().update()
                e.accept()
                return
            if e.button() == Qt.RightButton:
                # Right-click ends the polygon (alternative to double-click).
                self._finish_zone_draw()
                e.accept()
                return
        if e.button() == Qt.LeftButton:
            dim_item = self._dimension_item_at_view_pos(e.position().toPoint())
            if dim_item is not None:
                super().mousePressEvent(e)
                return
        if (
            e.button() == Qt.LeftButton
            and self.display_mode == "results"
            and self.is_visualization_mode
            and self._current_frame_packet is not None
        ):
            self._pick_replay_scope(p)
            e.accept()
            return
        if self.navigation_mode_enabled and e.button() == Qt.LeftButton and self.tool == "select":
            self.setDragMode(QGraphicsView.RubberBandDrag)
            super().mousePressEvent(e)
            return

        if (
            e.button() == Qt.LeftButton
            and (e.modifiers() & Qt.ControlModifier)
            and not self._space_pressed
        ):
            # Temporary CAD-style window zoom from any stage/module/tool.
            self._start_zoom_window(pt, temporary=(self.tool != "zoom_window"))
            return

        if (
            e.button() == Qt.LeftButton
            and self._pending_attr_edit
            and self.active_module in ("Load", "Boundary")
        ):
            if not self.hover_item:
                window = self.window()
                if window and hasattr(window, "statusBar"):
                    window.statusBar().showMessage(
                        "Hover a vertex or edge, then click to move the attribute.", 5000
                    )
                return
            itype, coords = self.hover_item
            if itype not in ("vertex", "edge"):
                return
            item = self._pending_attr_edit["item"]
            current_coords = item.get("coords")
            if coords != current_coords:
                self.push_undo_state()
                self._del_attrs(coords, record_history=False)
                item["coords"] = coords
            kind = self._pending_attr_edit["kind"]
            if kind == "bc":
                self.bcsChanged.emit()
            else:
                self.loadsChanged.emit()
            self._pending_attr_edit = None
            self.redraw()
            return

        if e.button() == Qt.MiddleButton or (self._space_pressed and e.button() == Qt.LeftButton):
            self._panning = True
            self._pan_start = e.position()
            self.setCursor(Qt.ClosedHandCursor)
            return

        if self.tool == "zoom_window" and e.button() == Qt.LeftButton:
            self._start_zoom_window(pt, temporary=False)
            return

        # Measure tool — click point 1, click point 2, distance + angle
        # displayed. A third click starts a fresh measurement; right-click
        # cancels and returns to the select tool.
        if self.tool == "measure":
            if e.button() == Qt.LeftButton:
                snap_pt = self._input_point(pt) if self.active_module == "Part" else pt
                if self._measure_first is None or self._measure_second is not None:
                    # First click of a new measurement (or restart after one finished).
                    self._measure_first = (float(snap_pt[0]), float(snap_pt[1]))
                    self._measure_second = None
                    self._announce_status("Measure: click second point. Right-click to exit.")
                else:
                    self._measure_second = (float(snap_pt[0]), float(snap_pt[1]))
                    dx = self._measure_second[0] - self._measure_first[0]
                    dy = self._measure_second[1] - self._measure_first[1]
                    dist_val = math.hypot(dx, dy)
                    ang_deg = math.degrees(math.atan2(dy, dx))
                    unit = getattr(self, "current_unit", "mm")
                    self._announce_status(
                        f"Distance: {dist_val:.3f} {unit}   Angle: {ang_deg:.1f}°   (click again for a new measurement)"
                    )
                self.viewport().update()
                e.accept()
                return
            if e.button() == Qt.RightButton:
                self._measure_first = None
                self._measure_second = None
                self.set_tool("select")
                e.accept()
                return

        # Right Click Logic
        if e.button() == Qt.RightButton:
            # Cancel any active draw tool — return to the default hand/select
            # cursor. This matches Figma/AutoCAD convention: right-click pops
            # out of drawing mode. Mid-draw or specialised tools below handle
            # their own right-click semantics (commit/cancel/context menu).
            _cancellable_draw_tools = (
                "rectangle", "circle", "ellipse", "polygon",
                "line", "freeform", "polyline", "arc",
            )
            if (
                self.tool in _cancellable_draw_tools
                and self.mode == "idle"
            ):
                self.current.clear()
                self._clear_preview()
                self.set_tool("select")
                e.accept()
                return
            if self.tool == "arc_segment":
                if self.active_module in ("Load", "Boundary") and self._arc_segment_polyline:
                    if hasattr(e, "globalPosition"):
                        global_pos = e.globalPosition().toPoint()
                    elif hasattr(e, "globalPos"):
                        global_pos = e.globalPos()
                    else:
                        global_pos = self.mapToGlobal(self.mapFromScene(0, 0))
                    self._show_arc_segment_menu(global_pos)
                else:
                    self._clear_arc_segment_selection()
                return
            if self.tool == "polyline" and self.mode == "drawing":
                if len(self.current) >= 2:
                    self._finalize_current()
                return
            if self.tool == "arc" and self.mode in ("drawing_arc_1", "drawing_arc_2"):
                self.set_tool("select")
                return
            if self.tool in ("dimension", "constraint"):
                self._pending_dimension = None
                self._pending_constraint = None
                return
            if not self.navigation_mode_enabled and self._show_context_menu(e):
                self._last_mouse_context_menu_ts = time.monotonic()
                e.accept()
                return
            if self.tool == "select":
                return super().mousePressEvent(e)

        # Left Click Logic
        if e.button() == Qt.LeftButton:
            if self.tool == "arc_segment":
                if self.active_module in ("Part", "Load", "Boundary"):
                    pick_pt = self._input_point(pt) if self.active_module == "Part" else pt
                    self._handle_arc_segment_click(pick_pt)
                else:
                    self._announce_status("Arc segment selection is only available in Part, BCs, or Loads.")
                return
            if self.tool == "paint_bc" and self.active_module in ("Load", "Boundary"):
                self._start_paint(pt)
                return
            # Universal canvas pan — when the hand/select tool is active AND
            # the user clicks on empty canvas (no part, no sketch under the
            # cursor), drag the studio view around. Works in every stage
            # (Geometry, Materials, BCs, Mesh, etc.), not just Part mode.
            # Shift+drag empty → rubber-band marquee selection of parts.
            if self.tool == "select" and not self.navigation_mode_enabled:
                part_at_cursor = self._get_part_at_point(raw_pt)
                sketch_at_cursor = self._sketch_at_point(raw_pt) if self.active_module == "Part" else None
                if part_at_cursor is None and sketch_at_cursor is None:
                    shift_held = bool(e.modifiers() & Qt.ShiftModifier)
                    if shift_held and self.active_module == "Part":
                        # Marquee selection — capture start, rubber-band rect
                        # drawn in mouseMoveEvent, parts inside it selected on
                        # release.
                        self._marquee_start_scene_pt = raw_pt
                        self._marquee_current_scene_pt = raw_pt
                        self._marquee_active = True
                        self.setCursor(Qt.CrossCursor)
                        return
                    self._panning = True
                    self._pan_start = e.position()
                    self.setCursor(Qt.ClosedHandCursor)
                    # Clear part selection in modules where empty-click means
                    # "no target" (Property/Boundary/Load).
                    if self.active_module in ("Property", "Boundary", "Load"):
                        self.set_selected_part(None, emit_signal=True)
                    return
            if self.active_module == "Property":
                clicked_part = self._get_part_at_point(raw_pt)
                if clicked_part:
                    if e.modifiers() & Qt.ShiftModifier:
                        self.set_selected_part(clicked_part.id, emit_signal=True)
                        self._pick_material_from_part(clicked_part)
                    elif self.material_paint_mode:
                        self.assign_material_to_part(clicked_part)
                    else:
                        self.set_selected_part(clicked_part.id, emit_signal=True)
                return

            if self.active_module == "Part":
                if self.tool in ("move", "copy", "mirror"):
                    part = self.get_selected_part()
                    if not part:
                        part = self._get_part_at_point(raw_pt)
                        if part:
                            self.set_selected_part(part.id, emit_signal=True)
                            if self.tool == "mirror":
                                self._announce_status("Mirror: click first point of mirror line.")
                            else:
                                self._announce_status("Specify base point.")
                        else:
                            self._announce_status("Click a part to select it.")
                        return
                    snap_pt = self._input_point(pt)
                    if self.tool in ("move", "copy"):
                        if self._transform_base is None:
                            self._transform_base = snap_pt
                            self._transform_drag_active = True
                            self._transform_drag_moved = False
                            self._transform_drag_start_view = e.position()
                            self._announce_status("Drag to target, or click a target point.")
                            return
                        dx = snap_pt[0] - self._transform_base[0]
                        dy = snap_pt[1] - self._transform_base[1]
                        if self.tool == "move":
                            self.move_selected_part(dx, dy)
                            self.set_tool("select")
                        else:
                            self.copy_selected_part(dx, dy, count=1, name_suffix="Copy")
                            self._announce_status("Specify next point or press Esc to finish.")
                        return
                    if self.tool == "mirror":
                        if not self._transform_line:
                            self._transform_line = [snap_pt]
                            self._announce_status("Specify second point of mirror line.")
                            return
                        p1 = self._transform_line[0]
                        p2 = snap_pt
                        if dist(p1, p2) <= 1e-9:
                            self._announce_status("Mirror line is too short. Pick another point.")
                            return
                        resp = QMessageBox.question(
                            self,
                            "Mirror",
                            "Delete original geometry after mirror?",
                            QMessageBox.Yes | QMessageBox.No,
                            QMessageBox.No,
                        )
                        keep_original = resp == QMessageBox.No
                        self.mirror_selected_part_line(p1, p2, keep_original=keep_original)
                        self.set_tool("select")
                        return
                if self.tool == "dimension":
                    if self._pending_dimension and self._pending_dimension.get("mode") == "angle":
                        if self._handle_angle_dimension_click(pt):
                            return
                    self._handle_dimension_click(pt)
                    return
                if self.tool == "trim":
                    if self._trim_sketch_at(pt):
                        return
                    return
                if self.tool == "constraint":
                    self._handle_constraint_click(pt)
                    return
                if self.tool == "erase":
                    if self._erase_sketch_at(pt):
                        return
                    part = self._get_part_at_point(raw_pt, include_void=True)
                    if part:
                        self.delete_part(part, confirm=True)
                    return

                if self.tool == "select":
                    part = self._get_part_at_point(raw_pt)
                    ctrl_held = bool(e.modifiers() & Qt.ControlModifier)
                    if part:
                        # Ctrl+Click → toggle this part in the multi-selection
                        # so the user can combine many parts at once via Ctrl+P.
                        if ctrl_held:
                            ids = getattr(self, "multi_selected_part_ids", None)
                            if ids is None:
                                ids = set()
                                if self.selected_part_id is not None:
                                    ids.add(int(self.selected_part_id))
                                self.multi_selected_part_ids = ids
                            pid = int(part.id)
                            if pid in ids:
                                ids.discard(pid)
                            else:
                                ids.add(pid)
                            self.set_selected_part(part.id, emit_signal=True)
                            self.redraw()
                            return
                        # Regular click → single selection + start drag.
                        self.multi_selected_part_ids = {int(part.id)}
                        self.set_selected_part(part.id, emit_signal=True)
                        # Push a single undo snapshot at drag start; the move
                        # itself updates the geometry incrementally without
                        # creating an undo per pixel.
                        self.push_undo_state()
                        self._drag_part_id = int(part.id)
                        self._drag_part_start_scene_pt = raw_pt
                        self._drag_part_last_scene_pt = raw_pt
                        self._drag_part_moved = False
                        self.setCursor(Qt.ClosedHandCursor)
                        return
                    # No part under cursor — try an un-confirmed sketch next.
                    sketch_idx = self._sketch_at_point(raw_pt)
                    if sketch_idx is not None:
                        self.push_undo_state()
                        self._drag_sketch_index = int(sketch_idx)
                        self._drag_sketch_last_scene_pt = raw_pt
                        self._drag_sketch_moved = False
                        self.setCursor(Qt.ClosedHandCursor)
                        return
                    # Empty canvas — start a viewport pan so the user can drag
                    # the studio around (like Figma / CAD pan with left button).
                    self.multi_selected_part_ids = set()
                    self.set_selected_part(None, emit_signal=True)
                    self._panning = True
                    self._pan_start = e.position()
                    self.setCursor(Qt.ClosedHandCursor)
                    return

                pt = self._input_point(pt)

                if self.tool == "select":
                    part = self._get_part_at_point(pt)
                    if part:
                        self.set_selected_part(part.id, emit_signal=True)
                    return

                if self.tool in ("line", "rectangle", "circle", "ellipse", "freeform"):
                    if self.tool == "rectangle" and self.mode == "drawing":
                        return
                    if self.tool == "rectangle" and self.mode == "idle":
                        if self._rect_use_dimensions:
                            params = self._prompt_rect_params()
                            if params:
                                width, height = params
                                if self._rect_draw_mode == "center":
                                    end = (pt[0] + width / 2.0, pt[1] + height / 2.0)
                                else:
                                    end = (pt[0] + width, pt[1] + height)
                                self.current = [pt]
                                self._pending_rectangle_start = pt
                                self._finalize_shape(end)
                                return
                            return
                        self.current = [pt]
                        self.mode = "drawing"
                        self._pending_rectangle_start = pt
                        if self._rect_draw_mode == "center":
                            self._announce_status("Rectangle (center): click center, then corner.")
                        else:
                            self._announce_status("Rectangle (2-corner): click first corner, then opposite corner.")
                        return
                    if self.tool == "circle" and self.mode == "idle" and self._circle_draw_mode == "radius":
                        radius = self._prompt_circle_radius()
                        if radius is not None:
                            end = (pt[0] + radius, pt[1])
                            self.current = [pt]
                            self._finalize_shape(end)
                        return
                    if self.parametric_enabled and self.mode == "idle":
                        if self.tool == "line":
                            params = self._prompt_line_params()
                            if params:
                                start = params["start"]
                                end = params["end"]
                                self.current = [start]
                                self._finalize_shape(end)
                                return
                        elif self.tool == "slot":
                            params = self._prompt_slot_params()
                            if params:
                                length, angle, width = params
                                end = (
                                    pt[0] + length * math.cos(math.radians(angle)),
                                    pt[1] + length * math.sin(math.radians(angle)),
                                )
                                self._slot_width = width
                                self.current = [pt]
                                self._finalize_shape(end)
                                return
                    self.current = [pt]
                    self.mode = "drawing"
                    return

                if self.tool in ("polyline", "spline"):
                    if self.mode == "idle":
                        self.current = [pt]
                        self.mode = "drawing"
                    else:
                        self.current.append(pt)
                        if self.tool == "polyline":
                            self._commit_last_segment()
                    return

                if self.tool == "arc":
                    if self.mode == "idle":
                        self.current = [pt]
                        self.mode = "drawing_arc_1"
                    elif self.mode == "drawing_arc_1":
                        self.current.append(pt)
                        self.mode = "drawing_arc_2"
                    elif self.mode == "drawing_arc_2":
                        self.current.append(pt)
                        self._finalize_shape(pt)
                    return

                if self.tool == "polygon":
                    if self.mode == "idle":
                        if self.parametric_enabled:
                            params = self._prompt_polygon_params()
                            if params:
                                sides, radius = params
                                self.polygon_sides = sides
                                end = (pt[0] + radius, pt[1])
                                self.current = [pt]
                                self._finalize_shape(end)
                                return
                        n, ok = QInputDialog.getInt(
                            self, "Polygon", "Sides:", self.polygon_sides, 3, 360
                        )
                        if not ok:
                            return
                        self.polygon_sides = n
                        self.current = [pt]
                        self.mode = "sizing"
                    elif self.mode == "sizing":
                        self._finalize_shape(pt)
                    return

            if self.tool == "select":
                part = self._get_part_at_point(raw_pt)
                if part:
                    self.set_selected_part(part.id, emit_signal=True)
                return

            if self.tool == "initial_velocity":
                clicked_part = None
                for part in reversed(self.parts):
                    if not part.is_void and part.geometry and part.geometry.contains(Point(pt)):
                        clicked_part = part
                        break

                if clicked_part:
                    unit = self.current_unit or "m"
                    vx, ok1 = QInputDialog.getDouble(
                        self,
                        "Initial Velocity",
                        f"Enter Vx ({unit}/s):",
                        0.0,
                        -1e6,
                        1e6,
                        3,
                    )
                    if ok1:
                        vy, ok2 = QInputDialog.getDouble(
                            self,
                            "Initial Velocity",
                            f"Enter Vy ({unit}/s):",
                            0.0,
                            -1e6,
                            1e6,
                            3,
                        )
                        if ok2:
                            self.push_undo_state()
                            self.initial_velocities = [
                                iv for iv in self.initial_velocities if iv["part_id"] != clicked_part.id
                            ]
                            self.initial_velocities.append(
                                {"part_id": clicked_part.id, "vx": vx, "vy": vy}
                            )
                            QMessageBox.information(
                                self,
                                "Success",
                                f"Initial velocity ({vx}, {vy}) {unit}/s set for part '{clicked_part.name}'.",
                            )
                            self.set_tool("select")
                else:
                    QMessageBox.warning(self, "Info", "No part was clicked.")
                return

        super().mousePressEvent(e)

    def mouseDoubleClickEvent(self, e):
        if self._zone_draw_active and e.button() == Qt.LeftButton:
            # The press that triggered this double-click already added a point;
            # drop it so the user doesn't get a duplicate vertex.
            if self._zone_draw_points:
                self._zone_draw_points.pop()
            self._finish_zone_draw()
            e.accept()
            return
        super().mouseDoubleClickEvent(e)

    def mouseMoveEvent(self, e):
        p = self.mapToScene(e.position().toPoint()); pt = (p.x(), p.y())
        try:
            self.cursorScenePositionChanged.emit(float(pt[0]), float(pt[1]))
        except Exception:
            pass
        # Remember the last cursor scene point so drawForeground can render
        # crosshair guides while a draw tool is active.
        self._last_cursor_scene_pt = (float(pt[0]), float(pt[1]))
        if self.tool in (
            "line", "rectangle", "circle", "ellipse", "polygon",
            "polyline", "arc", "freeform", "measure",
        ):
            self.viewport().update()
        # Part-drag handler — moves the selected part with the cursor when the
        # user has pressed and held on it in select mode.
        if (
            getattr(self, "_drag_part_id", None) is not None
            and (e.buttons() & Qt.LeftButton)
        ):
            start = self._drag_part_last_scene_pt
            if start is not None:
                dx = pt[0] - start[0]
                dy = pt[1] - start[1]
                if abs(dx) > 1e-9 or abs(dy) > 1e-9:
                    dx_snap, dy_snap, guides = self._compute_alignment_snap(
                        self._drag_part_id, dx, dy
                    )
                    self._move_part_by_delta(self._drag_part_id, dx_snap, dy_snap)
                    # Advance "last" by the actual delta so the cursor stays
                    # in sync — the snap only adjusts position, not cursor.
                    self._drag_part_last_scene_pt = (
                        start[0] + dx_snap, start[1] + dy_snap,
                    )
                    self._align_guides = guides
                    self._drag_part_moved = True
                    self.viewport().update()
            return
        # Sketch-drag handler — translates an unconfirmed sketch with the
        # cursor (sketches that haven't been turned into a Part yet).
        if (
            getattr(self, "_drag_sketch_index", None) is not None
            and (e.buttons() & Qt.LeftButton)
        ):
            start = self._drag_sketch_last_scene_pt
            if start is not None:
                dx = pt[0] - start[0]
                dy = pt[1] - start[1]
                if abs(dx) > 1e-9 or abs(dy) > 1e-9:
                    self._move_sketch_by_delta(self._drag_sketch_index, dx, dy)
                    self._drag_sketch_last_scene_pt = pt
                    self._drag_sketch_moved = True
            return
        # Marquee selection update — paint the rubber-band rectangle while
        # the user drags. Selection is finalised on release.
        if (
            getattr(self, "_marquee_active", False)
            and (e.buttons() & Qt.LeftButton)
        ):
            self._marquee_current_scene_pt = pt
            self.viewport().update()
            return
        # Canvas pan — left-button-on-empty-space in select mode. Translates
        # the viewport's scrollbars so the user can drag the studio around
        # with the standard hand-grab pattern.
        if self._panning and (e.buttons() & Qt.LeftButton) and self.tool == "select":
            delta = e.position() - self._pan_start
            self._pan_start = e.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            return
        # Hover feedback in select mode — change cursor to PointingHand when
        # the user is hovering over a grabbable part; OpenHand otherwise.
        # Skipped while a drag is in progress (the drag handler owns the
        # cursor) and while drawing tools are active.
        if (
            self.tool == "select"
            and not self._panning
            and not (e.buttons() & Qt.LeftButton)
            and not self.navigation_mode_enabled
        ):
            hovered_part = self._get_part_at_point(pt)
            if hovered_part is not None:
                if self.cursor().shape() != Qt.PointingHandCursor:
                    self.setCursor(Qt.PointingHandCursor)
            else:
                if self.cursor().shape() != Qt.OpenHandCursor:
                    self.setCursor(Qt.OpenHandCursor)
        if self._partition_pick_active:
            self._partition_hover = (float(pt[0]), float(pt[1]))
            self.viewport().update()
            return
        if self._vertex_seed_pick_active:
            v = self._vertex_at_scene_point(pt, self._vertex_seed_pick_tol())
            if v != self._vertex_seed_hover:
                self._vertex_seed_hover = v
                self.viewport().update()
            return
        if self._edge_seed_pick_mode is not None:
            edge = self._edge_at_scene_point(pt, self._edge_seed_pick_tol())
            new_hover = None
            if edge is not None:
                pid, s, eend = edge
                new_hover = {"part_id": pid, "start": s, "end": eend}
            if new_hover != self._edge_seed_hover:
                self._edge_seed_hover = new_hover
                self.viewport().update()
            return
        if self._zone_draw_active:
            self._zone_draw_hover = (float(pt[0]), float(pt[1]))
            self.viewport().update()
            # Fall through so cursor tooltips etc. can still update, but skip
            # the heavy hover-pick logic below.
            return
        if self.navigation_mode_enabled and self._nav_rotating and self._nav_rotate_start is not None:
            if not (e.buttons() & Qt.RightButton):
                self._nav_rotating = False
                self._nav_rotate_start = None
                self._nav_right_dragged = False
            else:
                diff = e.position() - self._nav_rotate_start
                self._nav_rotate_start = e.position()
                if abs(float(diff.x())) > 1.0 or abs(float(diff.y())) > 1.0:
                    self._nav_right_dragged = True
                self.rotate(float(diff.x()) * 0.2)
                return
        if self._panning:
            if not ((e.buttons() & Qt.MiddleButton) or ((e.buttons() & Qt.LeftButton) and self._space_pressed)):
                self._panning = False
                self._pan_start = None
                self.setCursor(Qt.OpenHandCursor if self._space_pressed else Qt.ArrowCursor)
            else:
                delta = e.position() - self._pan_start
                self._pan_start = e.position()
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
                return

        if self.tool == "rectangle":
            try:
                snap_pt = self._input_point(pt)
            except Exception:
                snap_pt = pt
            self._update_rectangle_cursor_status(snap_pt)

        if self.tool == "arc_segment" and self._arc_select_points:
            preview = self._arc_select_points + [pt]
            self._update_arc_segment_preview(preview, provisional=True)
            return

        if self._paint_active:
            self._update_paint(pt)
            return

        if self._zoom_window_active and self._zoom_window_item:
            x0, y0 = self._zoom_window_start
            x1, y1 = pt
            rect = QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
            self._zoom_window_item.setRect(rect)
            return

        if self.tool in ("move", "copy") and self._transform_base:
            if self._transform_drag_active and self._transform_drag_start_view is not None:
                delta = e.position() - self._transform_drag_start_view
                if (
                    abs(float(delta.x())) >= self._transform_drag_threshold_px
                    or abs(float(delta.y())) >= self._transform_drag_threshold_px
                ):
                    self._transform_drag_moved = True
            try:
                snap_pt = self._input_point(pt)
            except Exception:
                snap_pt = pt
            self._update_transform_preview(self._transform_base, snap_pt)
            return
        if self.tool == "mirror" and self._transform_line:
            try:
                snap_pt = self._input_point(pt)
            except Exception:
                snap_pt = pt
            self._update_transform_preview(self._transform_line[0], snap_pt)
            return

        if self.tool == "select":
            self._handle_hover(pt)
        if self.mode in ("drawing", "sizing"):
            try:
                snap_pt = self._input_point(pt)
                self._update_draw_preview(snap_pt)
            except Exception:
                pass
        else:
            draw_tools = (
                "line",
                "rectangle",
                "circle",
                "ellipse",
                "polygon",
                "polyline",
                "arc",
            )
            if self.active_module == "Part" and self.tool in draw_tools:
                try:
                    self._input_point(pt)
                    self._clear_preview()
                    self._draw_snap_indicator()
                except Exception:
                    pass
            else:
                self._snap_indicator = None
                self._clear_preview()
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        p = self.mapToScene(e.position().toPoint()); pt = (p.x(), p.y())
        # Finish a part-drag (move) operation. Push an undo state only if the
        # part was actually moved (not a simple click).
        if getattr(self, "_drag_part_id", None) is not None and e.button() == Qt.LeftButton:
            moved = bool(getattr(self, "_drag_part_moved", False))
            self._drag_part_id = None
            self._drag_part_start_scene_pt = None
            self._drag_part_last_scene_pt = None
            self._drag_part_moved = False
            # Hide the alignment guides as soon as the drag ends.
            self._align_guides = []
            self.viewport().update()
            self.setCursor(Qt.OpenHandCursor)
            if moved:
                self.geometryChanged.emit()
                self.partsChanged.emit()
            e.accept()
            return
        # Finish a sketch-drag operation.
        if getattr(self, "_drag_sketch_index", None) is not None and e.button() == Qt.LeftButton:
            self._drag_sketch_index = None
            self._drag_sketch_last_scene_pt = None
            self._drag_sketch_moved = False
            self.setCursor(Qt.OpenHandCursor)
            e.accept()
            return
        # Finish marquee selection — collect all parts whose centroid falls
        # within the rubber-band rectangle and add them to the multi-selection.
        if getattr(self, "_marquee_active", False) and e.button() == Qt.LeftButton:
            self._marquee_active = False
            start = self._marquee_start_scene_pt
            end = self._marquee_current_scene_pt
            self._marquee_start_scene_pt = None
            self._marquee_current_scene_pt = None
            if start is not None and end is not None:
                min_x, max_x = sorted([start[0], end[0]])
                min_y, max_y = sorted([start[1], end[1]])
                hit_ids = set()
                for part in self.parts:
                    if getattr(part, "is_void", False):
                        continue
                    geom = getattr(part, "geometry", None)
                    if geom is None or geom.is_empty:
                        continue
                    try:
                        c = geom.centroid
                        if min_x <= c.x <= max_x and min_y <= c.y <= max_y:
                            hit_ids.add(int(part.id))
                    except Exception:
                        continue
                if hit_ids:
                    self.multi_selected_part_ids = hit_ids
                    first = next(iter(hit_ids))
                    self.set_selected_part(first, emit_signal=True)
                    self.viewport().update()
                    if self.window() and hasattr(self.window(), "show_toast"):
                        try:
                            self.window().show_toast(f"Selected {len(hit_ids)} parts", kind="info")
                        except Exception:
                            pass
            self.setCursor(Qt.OpenHandCursor)
            self.viewport().update()
            e.accept()
            return
        # Finish a select-mode canvas pan.
        if self._panning and e.button() == Qt.LeftButton and self.tool == "select":
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.OpenHandCursor)
            e.accept()
            return
        if self.navigation_mode_enabled and e.button() == Qt.RightButton and self._nav_rotating:
            dragged = bool(self._nav_right_dragged)
            self._nav_rotating = False
            self._nav_rotate_start = None
            self._nav_right_dragged = False
            if dragged:
                e.accept()
                return
            if self._show_context_menu(e):
                self._last_mouse_context_menu_ts = time.monotonic()
                e.accept()
                return
            e.accept()
            return
        if self.navigation_mode_enabled and e.button() == Qt.LeftButton and self.dragMode() == QGraphicsView.RubberBandDrag:
            super().mouseReleaseEvent(e)
            self.setDragMode(QGraphicsView.RubberBandDrag if self.navigation_mode_enabled and self.tool == "select" else QGraphicsView.NoDrag)
            return
        if self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            return

        if e.button() == Qt.LeftButton:
            if self._zoom_window_active:
                self._finish_zoom_window()
                return

            if self._paint_active:
                self._finish_paint()
                return

            if (
                self.tool in ("move", "copy")
                and self._transform_drag_active
                and self._transform_base is not None
            ):
                self._transform_drag_active = False
                self._transform_drag_start_view = None
                try:
                    snap_pt = self._input_point(pt)
                except Exception:
                    snap_pt = pt
                if self._transform_drag_moved:
                    dx = snap_pt[0] - self._transform_base[0]
                    dy = snap_pt[1] - self._transform_base[1]
                    self._transform_drag_moved = False
                    self._transform_base = None
                    self._clear_transform_preview()
                    if self.tool == "move":
                        self.move_selected_part(dx, dy)
                        self.set_tool("select")
                    else:
                        self.copy_selected_part(dx, dy, count=1, name_suffix="Copy")
                        self._announce_status("Copy created. Drag again or press Esc to finish.")
                    return
                self._transform_drag_moved = False
                self._announce_status("Specify target point.")
                return

            if self.tool in ("line", "rectangle", "circle", "ellipse") and self.mode == "drawing":
                snap_pt = self._input_point(pt)
                if self.tool == "rectangle" and self._pending_rectangle_start is not None:
                    self.current = [self._pending_rectangle_start]
                self._finalize_shape(snap_pt)
            elif self.tool == "freeform" and self.mode == "drawing":
                self.current.append(pt); self._finalize_current()
        super().mouseReleaseEvent(e)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    def _edit_part_from_context(self, part):
        if part is None:
            return
        self.set_selected_part(getattr(part, "id", None), emit_signal=True)
        try:
            self.set_module("Part")
        except Exception:
            pass
        try:
            self.begin_part_shape_edit(part)
        except Exception:
            pass

    def _suggest_interaction_part_pair(self, part):
        if part is None:
            return None
        try:
            clicked_id = int(getattr(part, "id", -1))
        except Exception:
            return None
        try:
            selected_id = int(self.selected_part_id) if self.selected_part_id is not None else None
        except Exception:
            selected_id = None
        if selected_id is not None and selected_id != clicked_id:
            return (selected_id, clicked_id)
        for other in self.parts:
            try:
                other_id = int(getattr(other, "id", -1))
            except Exception:
                continue
            if other_id != clicked_id and not getattr(other, "is_void", False):
                return (clicked_id, other_id)
        return (clicked_id, None)

    def _open_interaction_dialog_for_part(self, part):
        if part is None:
            return
        main = self.window()
        if not main or not hasattr(main, "properties_panel"):
            return
        panel = getattr(main, "properties_panel", None)
        if panel is None or not hasattr(panel, "interfaces_tab"):
            return
        try:
            panel.tabs.setCurrentWidget(panel.interfaces_tab)
        except Exception:
            pass
        pair = self._suggest_interaction_part_pair(part)
        try:
            panel.interfaces_tab.define_interface(preset_part_ids=pair)
        except TypeError:
            panel.interfaces_tab.define_interface()

    def _populate_part_context_actions(self, menu, part, *, include_bc_actions=True):
        if menu is None or part is None:
            return
        edit_label = "Edit Sketch" if getattr(part, "sketches", None) else "Edit Geometry"
        menu.addAction(
            f"Select Part: {part.name}",
            lambda p=part: self.set_selected_part(p.id, emit_signal=True),
        )
        menu.addAction(edit_label, lambda p=part: self._edit_part_from_context(p))
        menu.addAction("Delete", lambda p=part: self.delete_part(p, confirm=True))
    
    def _show_context_menu(self, event):
        if event is None:
            return False
        if hasattr(event, "position"):
            view_pos = event.position().toPoint()
        elif hasattr(event, "pos"):
            view_pos = event.pos()
        else:
            view_pos = None
        if hasattr(event, "globalPosition"):
            global_pos = event.globalPosition().toPoint()
        elif hasattr(event, "globalPos"):
            global_pos = event.globalPos()
        else:
            global_pos = None
        return self._show_context_menu_at(view_pos, global_pos)

    def _show_context_menu_at(self, view_pos, global_pos=None):
        if view_pos is None:
            if global_pos is not None:
                view_pos = self.viewport().mapFromGlobal(global_pos)
            else:
                view_pos = self.viewport().rect().center()
        if global_pos is None:
            global_pos = self.viewport().mapToGlobal(view_pos)
        scene_pos = self.mapToScene(view_pos)

        stage = getattr(self, "active_stage", None)
        stage_label = stage.name.title() if stage else None

        if self.active_module == "Property":
            part = self._get_part_at_point((scene_pos.x(), scene_pos.y()), include_void=False)
            if not part:
                return False
            self.set_selected_part(part.id, emit_signal=True)
            menu = QMenu(self)
            if stage_label:
                label = menu.addAction(f"Stage: {stage_label}")
                label.setEnabled(False)
                menu.addSeparator()
            self._populate_part_context_actions(menu, part)
            menu.exec(global_pos)
            menu.close()
            return True

        if self.active_module in ("Boundary", "Load"):
            is_bc_module = self.active_module == "Boundary"
            load_pos = scene_pos
            click_pt = (float(load_pos.x()), float(load_pos.y()))
            self._handle_hover(click_pt)
            unified_2d_static_menu = getattr(self, "project_mode", "2d") != "3d"
            if not self.hover_item:
                part = self._get_part_at_point((load_pos.x(), load_pos.y()), include_void=False)
                if not part:
                    return False
                self.set_selected_part(part.id, emit_signal=True)
                menu = QMenu(self)
                if stage_label:
                    label = menu.addAction(f"Stage: {stage_label}")
                    label.setEnabled(False)
                    menu.addSeparator()
                if unified_2d_static_menu:
                    menu.addAction("Hover an edge or corner to apply static BCs/loads here.").setEnabled(False)
                    menu.addSeparator()
                    advanced_menu = menu.addMenu(f"Whole Part (Advanced): {part.name}")
                    advanced_menu.addAction("Fix UX & UY (Part)", lambda p=part: self._apply_fixed_bc_for_part("fix_xy", p.id))
                    advanced_menu.addAction("Fix UX (Part)", lambda p=part: self._apply_fixed_bc_for_part("fix_x", p.id))
                    advanced_menu.addAction("Fix UY (Part)", lambda p=part: self._apply_fixed_bc_for_part("fix_y", p.id))
                    advanced_menu.addSeparator()
                    advanced_menu.addAction(
                        "Velocity X (Part)",
                        lambda p=part: self._apply_time_profile_for_part("velocity", "x", p.id),
                    )
                    advanced_menu.addAction(
                        "Velocity Y (Part)",
                        lambda p=part: self._apply_time_profile_for_part("velocity", "y", p.id),
                    )
                    advanced_menu.addSeparator()
                    advanced_menu.addAction(
                        "Force X (Part)",
                        lambda p=part: self._apply_time_profile_for_part("force", "x", p.id),
                    )
                    advanced_menu.addAction(
                        "Force Y (Part)",
                        lambda p=part: self._apply_time_profile_for_part("force", "y", p.id),
                    )
                else:
                    menu.addAction(f"Selected Part: {part.name}").setEnabled(False)
                    menu.addSeparator()
                    if is_bc_module:
                        menu.addAction("Fix UX & UY (Part)", lambda p=part: self._apply_fixed_bc_for_part("fix_xy", p.id))
                        menu.addAction("Fix UX (Part)", lambda p=part: self._apply_fixed_bc_for_part("fix_x", p.id))
                        menu.addAction("Fix UY (Part)", lambda p=part: self._apply_fixed_bc_for_part("fix_y", p.id))
                        if getattr(self, "project_mode", "2d") == "3d":
                            menu.addAction("Fix UZ (Part)", lambda p=part: self._apply_fixed_bc_for_part("fix_z", p.id))
                        menu.addSeparator()
                        menu.addAction(
                            "Velocity X (Part)",
                            lambda p=part: self._apply_time_profile_for_part("velocity", "x", p.id),
                        )
                        menu.addAction(
                            "Velocity Y (Part)",
                            lambda p=part: self._apply_time_profile_for_part("velocity", "y", p.id),
                        )
                        if getattr(self, "project_mode", "2d") == "3d":
                            menu.addAction(
                                "Velocity Z (Part)",
                                lambda p=part: self._apply_time_profile_for_part("velocity", "z", p.id),
                            )
                    else:
                        menu.addAction(
                            "Force X (Part)",
                            lambda p=part: self._apply_time_profile_for_part("force", "x", p.id),
                        )
                        menu.addAction(
                            "Force Y (Part)",
                            lambda p=part: self._apply_time_profile_for_part("force", "y", p.id),
                        )
                        if getattr(self, "project_mode", "2d") == "3d":
                            menu.addAction(
                                "Force Z (Part)",
                                lambda p=part: self._apply_time_profile_for_part("force", "z", p.id),
                            )
                menu.addSeparator()
                self._populate_part_context_actions(menu, part, include_bc_actions=False)
                menu.exec(global_pos)
                menu.close()
                return True
            menu = QMenu(self)
            if stage_label:
                label = menu.addAction(f"Stage: {stage_label}")
                label.setEnabled(False)
                menu.addSeparator()
            itype, coords = self.hover_item
            menu.addAction(f"Selected: {itype}").setEnabled(False)
            menu.addSeparator()
            if unified_2d_static_menu:
                if itype == "vertex":
                    menu.addAction("Fix UX & UY", lambda: self._apply_fixed_bc("fix_xy", coords))
                    menu.addAction("Fix UX", lambda: self._apply_fixed_bc("fix_x", coords))
                    menu.addAction("Fix UY", lambda: self._apply_fixed_bc("fix_y", coords))
                    menu.addSeparator()
                    menu.addAction("Displacement X", lambda: self._add_displacement("x", coords))
                    menu.addAction("Displacement Y", lambda: self._add_displacement("y", coords))
                    menu.addSeparator()
                    menu.addAction("Velocity X", lambda: self._apply_time_profile("velocity", "x", coords))
                    menu.addAction("Velocity Y", lambda: self._apply_time_profile("velocity", "y", coords))
                    menu.addSeparator()
                    menu.addAction("Force X", lambda: self._add_force_component("x", coords))
                    menu.addAction("Force Y", lambda: self._add_force_component("y", coords))
                    menu.addAction("Add Moment", lambda: self._add_load("moment", coords))
                else:
                    menu.addAction("Fix Edge", lambda: self._apply_fixed_bc("fix_xy", coords))
                    menu.addAction("Displacement X", lambda: self._add_displacement("x", coords))
                    menu.addAction("Displacement Y", lambda: self._add_displacement("y", coords))
                    menu.addSeparator()
                    menu.addAction("Velocity X", lambda: self._apply_time_profile("velocity", "x", coords))
                    menu.addAction("Velocity Y", lambda: self._apply_time_profile("velocity", "y", coords))
                    menu.addSeparator()
                    menu.addAction("Force X", lambda: self._add_force_component("x", coords))
                    menu.addAction("Force Y", lambda: self._add_force_component("y", coords))
            else:
                if itype == "vertex":
                    if is_bc_module:
                        menu.addAction("Fix UX & UY", lambda: self._apply_fixed_bc("fix_xy", coords))
                        menu.addAction("Fix UX", lambda: self._apply_fixed_bc("fix_x", coords))
                        menu.addAction("Fix UY", lambda: self._apply_fixed_bc("fix_y", coords))
                        if getattr(self, "project_mode", "2d") == "3d":
                            menu.addAction("Fix UZ", lambda: self._apply_fixed_bc("fix_z", coords))
                        menu.addSeparator()
                        menu.addAction("Velocity X", lambda: self._apply_time_profile("velocity", "x", coords))
                        menu.addAction("Velocity Y", lambda: self._apply_time_profile("velocity", "y", coords))
                        if getattr(self, "project_mode", "2d") == "3d":
                            menu.addAction("Velocity Z", lambda: self._apply_time_profile("velocity", "z", coords))
                    else:
                        menu.addAction("Force X", lambda: self._apply_time_profile("force", "x", coords))
                        menu.addAction("Force Y", lambda: self._apply_time_profile("force", "y", coords))
                        if getattr(self, "project_mode", "2d") == "3d":
                            menu.addAction("Force Z", lambda: self._apply_time_profile("force", "z", coords))
                        menu.addAction("Add Moment", lambda: self._add_load("moment", coords))
                else:
                    if is_bc_module:
                        menu.addAction("Fix Edge", lambda: self._apply_fixed_bc("fix_xy", coords))
                        menu.addAction("Velocity X", lambda: self._apply_time_profile("velocity", "x", coords))
                        menu.addAction("Velocity Y", lambda: self._apply_time_profile("velocity", "y", coords))
                        if getattr(self, "project_mode", "2d") == "3d":
                            menu.addAction("Velocity Z", lambda: self._apply_time_profile("velocity", "z", coords))
                    else:
                        menu.addAction("Force X", lambda: self._apply_time_profile("force", "x", coords))
                        menu.addAction("Force Y", lambda: self._apply_time_profile("force", "y", coords))
                        if getattr(self, "project_mode", "2d") == "3d":
                            menu.addAction("Force Z", lambda: self._apply_time_profile("force", "z", coords))
            menu.addSeparator()
            menu.addAction("Clear Attributes Here", lambda: self._del_attrs(coords))
            menu.exec(global_pos)
            menu.close()
            return True

        try:
            item = self.itemAt(view_pos)
        except Exception:
            item = None
        if isinstance(item, DimensionTextItem):
            menu = QMenu(self)
            if stage_label:
                label = menu.addAction(f"Stage: {stage_label}")
                label.setEnabled(False)
                menu.addSeparator()
            edit_action = menu.addAction("Edit Dimension")
            chosen = menu.exec(global_pos)
            menu.close()
            if chosen == edit_action:
                item.start_inline_edit(Qt.MouseFocusReason)
            return True

        try:
            geom = self._find_geometry_at_scene_pos(scene_pos)
        except Exception:
            geom = None

        # Geometry / other stages: show stage-aware view options + part actions if applicable.
        menu = QMenu(self)
        if stage_label:
            label = menu.addAction(f"Stage: {stage_label}")
            label.setEnabled(False)
            menu.addSeparator()
        if geom is not None:
            menu.addAction("Add / Edit Dimension", lambda g=geom: self._open_dimension_for_geometry(g))
            menu.addSeparator()
        part = self._get_part_at_point((scene_pos.x(), scene_pos.y()), include_void=True)
        if part:
            if self.active_module == "Part":
                owner_type, owner_part = self._active_owner()
                if owner_type is not None:
                    click_pt = (float(scene_pos.x()), float(scene_pos.y()))
                    seg = self._find_nearest_segment(owner_type, owner_part, click_pt)
                    if seg is not None:
                        meta = self._get_sketch_meta(owner_type, owner_part, int(seg[0])) or {}
                        meta_type = str(meta.get("type", "shape")).title()
                        menu.addAction(
                            f"Edit Sketch Shape ({meta_type})...",
                            lambda p=click_pt, ot=owner_type, op=owner_part: self._edit_sketch_shape_at(p, ot, op),
                        )
                        menu.addSeparator()
            self._populate_part_context_actions(menu, part)
            menu.addSeparator()
        menu.addAction("Fit Screen", self.fit_view)
        menu.addAction("Frame Selection", self.fit_selection)
        menu.addAction("Center Origin", self.center_origin)
        menu.exec(global_pos)
        menu.close()
        return True

    def contextMenuEvent(self, event):
        if (time.monotonic() - float(getattr(self, "_last_mouse_context_menu_ts", 0.0) or 0.0)) < 0.35:
            self._last_mouse_context_menu_ts = 0.0
            event.accept()
            return
        if self._show_context_menu(event):
            event.accept()
            return
        event.ignore()

    # --- BC Helpers ---   
    def _add_velocity(self, axis, coords):
        unit = self.current_unit or "m"
        val, ok = QInputDialog.getDouble(
            self,
            "Velocity",
            f"Velocity {axis.upper()} ({unit}/s):",
            0.0,
            -1000,
            1000,
            2,
        )
        if ok:
            self.push_undo_state()
            self._del_attrs(coords, record_history=False)
            self.bcs.append({'type': f"velocity_{axis}", 'coords': coords, 'val': val})
            self.bcsChanged.emit()
            self.redraw()

    def _add_velocity_vector(self, coords):
        unit = self.current_unit or "m"
        vx, ok = QInputDialog.getDouble(
            self,
            "Velocity",
            f"Velocity X ({unit}/s):",
            0.0,
            -1000,
            1000,
            2,
        )
        if not ok:
            return
        vy, ok = QInputDialog.getDouble(
            self,
            "Velocity",
            f"Velocity Y ({unit}/s):",
            0.0,
            -1000,
            1000,
            2,
        )
        if not ok:
            return
        self.push_undo_state()
        self._del_attrs(coords, record_history=False)
        self.bcs.append({'type': "velocity_x", 'coords': coords, 'val': vx})
        self.bcs.append({'type': "velocity_y", 'coords': coords, 'val': vy})
        self.bcsChanged.emit()
        self.redraw()

    def _add_displacement(self, axis, coords):
        unit = self.current_unit or "m"
        val, ok = QInputDialog.getDouble(
            self,
            "Displacement",
            f"Displacement {axis.upper()} ({unit}):",
            0.0,
            -1e9,
            1e9,
            6,
        )
        if not ok:
            return
        self.push_undo_state()
        self._del_attrs(coords, record_history=False)
        self.bcs.append(
            {
                "type": f"velocity_{axis}",
                "coords": coords,
                "val": val,
                "bc_mode": "displacement",
                "display_type": f"Displacement U{axis.upper()}",
            }
        )
        self.bcsChanged.emit()
        self.redraw()

    def _add_force_component(self, axis, coords):
        val, ok = QInputDialog.getDouble(
            self,
            f"Force {axis.upper()}",
            f"Force {axis.upper()} (N):",
            0.0,
            -1e9,
            1e9,
            3,
        )
        if not ok:
            return
        self.push_undo_state()
        load = {
            "type": "force",
            "coords": coords,
            "fx": 0.0,
            "fy": 0.0,
            "display_type": f"Force F{axis.upper()}",
            "axis": str(axis).lower(),
        }
        if str(axis).lower() == "x":
            load["fx"] = val
        elif str(axis).lower() == "y":
            load["fy"] = val
        else:
            load["fz"] = val
        self.loads.append(load)
        self.loadsChanged.emit()
        self.redraw()

    def _add_load(self, ltype, coords):
        if ltype == "moment":
            m, ok = QInputDialog.getDouble(self, "Moment", "Moment (N*m):", 0.0, -1e9, 1e9, 3)
            if ok:
                self.push_undo_state()
                self.loads.append({'type': 'moment', 'coords': coords, 'm': m})
                self.redraw()
            return
        fx, ok = QInputDialog.getDouble(self, "Load FX", "Force X (N):", 0.0, -1e9, 1e9, 3)
        if not ok: return
        fy, ok = QInputDialog.getDouble(self, "Load FY", "Force Y (N):", 0.0, -1e9, 1e9, 3)
        if ok:
            self.push_undo_state()
            self.loads.append({'type': 'force', 'coords': coords, 'fx': fx, 'fy': fy})
            self.loadsChanged.emit()
            self.redraw()

    def _add_bc(self, bctype, coords):
        self.push_undo_state()
        self._del_attrs(coords, record_history=False)
        self.bcs.append({'type': bctype, 'coords': coords})
        self.redraw()

        self.bcsChanged.emit()
        # Stay in BCs stage until user advances manually.

    def _focus_bcs_tab(self):
        main = self.window()
        if main and hasattr(main, "properties_panel"):
            panel = main.properties_panel
            if hasattr(panel, "bcs_tab"):
                panel.tabs.setCurrentWidget(panel.bcs_tab)
                return
            panel.tabs.setCurrentWidget(panel.loads_tab)

    def _focus_loads_tab(self):
        main = self.window()
        if main and hasattr(main, "properties_panel"):
            panel = main.properties_panel
            panel.tabs.setCurrentWidget(panel.loads_tab)

    def _apply_fixed_bc(self, bctype, coords):
        self._focus_bcs_tab()
        self._add_bc(bctype, coords)

    def _apply_time_profile(self, kind, axis, coords):
        if kind == "velocity":
            self._focus_bcs_tab()
        else:
            self._focus_loads_tab()
        main = self.window()
        if not main or not hasattr(main, "properties_panel"):
            return
        panel = main.properties_panel
        if kind == "velocity" and hasattr(panel, "bcs_tab"):
            panel.bcs_tab.apply_time_profile_from_context(kind, axis, coords)
            return
        if not hasattr(panel, "loads_tab"):
            return
        panel.loads_tab.apply_time_profile_from_context(kind, axis, coords)

    def _apply_time_profile_for_part(self, kind, axis, part_id):
        if kind == "velocity":
            self._focus_bcs_tab()
        else:
            self._focus_loads_tab()
        main = self.window()
        if not main or not hasattr(main, "properties_panel"):
            return
        panel = main.properties_panel
        if kind == "velocity" and hasattr(panel, "bcs_tab"):
            panel.bcs_tab.apply_time_profile_to_part(kind, axis, part_id)
            return
        if not hasattr(panel, "loads_tab"):
            return
        panel.loads_tab.apply_time_profile_to_part(kind, axis, part_id)

    def _apply_fixed_bc_for_part(self, bctype, part_id):
        self._focus_bcs_tab()
        main = self.window()
        if not main or not hasattr(main, "properties_panel"):
            return
        panel = main.properties_panel
        if hasattr(panel, "bcs_tab"):
            panel.bcs_tab.apply_fixed_bc_to_part(bctype, part_id)
            return
        if not hasattr(panel, "loads_tab"):
            return
        panel.loads_tab.apply_fixed_bc_to_part(bctype, part_id)

    def _del_attrs(self, coords, record_history=True):
        if record_history and (
            any(b['coords'] == coords for b in self.bcs)
            or any(l['coords'] == coords for l in self.loads)
        ):
            self.push_undo_state()
        self.bcs = [b for b in self.bcs if b['coords'] != coords]
        self.loads = [l for l in self.loads if l['coords'] != coords]
        self.redraw()

    # --- Drawing Helpers (Simplified for brevity, logic identical to original) ---
    def _finalize_shape(self, pt):
        if not self.current:
            return

        start = self.current[0]
        verts = []
        self._pending_meta = None
        line_meta = None
        circle_meta = None

        if self.tool == "line":
            line_meta = self._line_meta_from_points(start, pt)
            verts = self._build_points_from_meta(line_meta, fallback_points=[])
        elif self.tool == "rectangle":
            rect_mode = "center" if self._rect_draw_mode == "center" else "two_corner"
            self._pending_meta = self._rectangle_meta_from_input_points(start, pt, mode=rect_mode)
            if self._pending_meta is not None:
                verts = self._build_points_from_meta(self._pending_meta, fallback_points=[])
        elif self.tool == "circle":
            circle_meta = self._circle_meta_from_center_radius(start, dist(start, pt))
            if float(circle_meta.get("radius", 0.0) or 0.0) > 0.0:
                verts = self._build_points_from_meta(circle_meta, fallback_points=[])
        elif self.tool == "slot":
            width = float(self._slot_width)
            if not self.parametric_enabled:
                width_input, ok = QInputDialog.getDouble(
                    self,
                    "Slot Width",
                    f"Width ({self.current_unit}):",
                    float(self._slot_width),
                    0.1,
                    1e9,
                    3,
                )
                if not ok:
                    self.current = []
                    self.mode = "idle"
                    self._clear_preview()
                    return
                width = float(width_input)
                self._slot_width = width
            verts = self._build_slot_vertices(start, pt, width)
        elif self.tool == "ellipse":
            x0, y0 = start
            x1, y1 = pt
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            rx, ry = abs(x1 - x0) / 2, abs(y1 - y0) / 2
            if rx > 0 and ry > 0:
                pts = [
                    (cx + rx * math.cos(t), cy + ry * math.sin(t))
                    for t in np.linspace(0, 2 * math.pi, 64, endpoint=False)
                ]
                pts.append(pts[0])
                verts = pts
        elif self.tool == "polygon":
            cx, cy = start
            radius = dist(start, pt)
            angle = math.atan2(pt[1] - cy, pt[0] - cx)
            for i in range(self.polygon_sides):
                theta = angle + 2 * math.pi * i / self.polygon_sides
                verts.append((cx + radius * math.cos(theta), cy + radius * math.sin(theta)))
            verts.append(verts[0])
        elif self.tool == "arc":
            if len(self.current) == 3:
                p1, p2, p3 = self.current
                verts = self._calculate_arc_points(p1, p3, p2)

        if self.tool == "line":
            self._pending_meta = line_meta or self._line_meta_from_points(start, pt)
        elif self.tool == "circle":
            self._pending_meta = circle_meta or self._circle_meta_from_center_radius(start, dist(start, pt))
        elif self.tool == "slot":
            self._pending_meta = {"type": "slot", "p1": start, "p2": pt, "width": float(self._slot_width)}
        elif self.tool == "polygon":
            self._pending_meta = {
                "type": "polygon",
                "center": start,
                "radius": dist(start, pt),
                "sides": int(self.polygon_sides),
                "angle": math.atan2(pt[1] - start[1], pt[0] - start[0]),
            }
        elif self.tool == "arc" and len(self.current) == 3:
            p1, p2, p3 = self.current
            center, radius, start_angle, end_angle = self._arc_params_from_points(p1, p2, p3)
            if center is not None:
                self._pending_meta = {
                    "type": "arc",
                    "center": center,
                    "radius": radius,
                    "start_angle": start_angle,
                    "end_angle": end_angle,
                }

        self.current = verts
        self._finalize_current()

    def _finalize_current(self):
        if len(self.current) >= 2:
            self.push_undo_state()
            if self.tool == "polyline" and len(self.current) >= 3:
                if dist(self.current[0], self.current[-1]) <= SNAP_TOL:
                    if dist(self.current[0], self.current[-1]) > 1e-6:
                        self.current.append(self.current[0])
                    else:
                        self.current[-1] = self.current[0]
            if self.tool == "freeform" and len(self.current) >= 3:
                prepared_pts, _ = self._prepare_freeform_stroke(self.current)
                self.current = prepared_pts

            meta = self._pending_meta
            if (
                meta is None
                and self.tool == "freeform"
                and bool(getattr(self, "freeform_auto_convert_enabled", False))
            ):
                inferred_meta, inferred_points = self._infer_freeform_shape_meta(self.current)
                if inferred_meta is not None and len(inferred_points) >= 2:
                    self.current = inferred_points
                    meta = inferred_meta
                    shape_label = str(meta.get("type", "shape")).title()
                    self._announce_status(f"Freeform auto-converted to {shape_label}.")
            if meta is None:
                meta = self._default_sketch_meta(self.current.copy())
                if self.tool == "freeform":
                    meta["source_tool"] = "freeform"
                    meta["skip_auto_dimension"] = True
            elif meta.get("type") == "polyline":
                meta["points"] = self.current.copy()
            self._append_sketch(self.current.copy(), meta=meta)
            self.geometryChanged.emit()
            self._pending_meta = None
        self._pending_rectangle_start = None
        self.current.clear(); self.mode = "idle"; self._clear_preview(); self.redraw()

    def _commit_last_segment(self):
        if len(self.current) >= 2:
            a,b = self.current[-2], self.current[-1]
            try: self.scene().addLine(a[0], a[1], b[0], b[1], QPen(Qt.black, 2))
            except: pass

    def _draw_snap_indicator(self):
        if not self._snap_indicator:
            return
        sx, sy = self._snap_indicator
        try:
            item = self._safe_add_ellipse(
                sx - 3,
                sy - 3,
                6,
                6,
                QPen(QColor(0, 120, 255), 1),
                QBrush(Qt.white),
                context="snap_indicator",
            )
            if item:
                self.preview_items.append(item)
        except Exception:
            pass

    def _add_preview_text(self, text, pos, color=None):
        try:
            item = self.scene().addText(str(text))
        except Exception:
            return
        item.setDefaultTextColor(color or QColor(30, 70, 120))
        try:
            rect = item.boundingRect()
            px = float(pos[0]) - 0.5 * float(rect.width())
            py = float(pos[1]) - 0.5 * float(rect.height())
        except Exception:
            px, py = float(pos[0]), float(pos[1])
        item.setPos(px, py)
        if self._y_axis_up:
            item.setTransform(QTransform(1, 0, 0, -1, 0, 0))
        self.preview_items.append(item)

    def _update_draw_preview(self, pt):
        self._clear_preview()
        pen = QPen(Qt.black, 1, Qt.DashLine)
        if self.tool == "freeform":
            self.current.append(pt)
            path = QPainterPath()
            if self.current:
                path.moveTo(*self.current[0])
                for p in self.current[1:]: path.lineTo(*p)
            try: self.preview_items.append(self.scene().addPath(path, QPen(Qt.black, 1)))
            except: pass
            return
        if not self.current: return
        # Simple rubber banding for shapes
        if self.tool in ("line","polyline"):
            p0 = self.current[-1]
            try: self.preview_items.append(self.scene().addLine(p0[0], p0[1], pt[0], pt[1], pen))
            except: pass
            if self.tool == "line":
                length = dist(p0, pt)
                unit = str(self._normalize_length_unit())
                mid = ((p0[0] + pt[0]) * 0.5, (p0[1] + pt[1]) * 0.5 - 12.0)
                self._add_preview_text(f"L: {length:.3f} {unit}", mid)
        elif self.tool == "rectangle":
            x0, y0 = self.current[0]
            x1, y1 = pt
            if self._rect_draw_mode == "center":
                hx = abs(x1 - x0)
                hy = abs(y1 - y0)
                minx = x0 - hx
                maxx = x0 + hx
                miny = y0 - hy
                maxy = y0 + hy
            else:
                minx = min(x0, x1)
                maxx = max(x0, x1)
                miny = min(y0, y1)
                maxy = max(y0, y1)
            try:
                self.preview_items.append(
                    self.scene().addRect(minx, miny, maxx - minx, maxy - miny, pen)
                )
            except Exception:
                pass
            width = maxx - minx
            height = maxy - miny
            width_display = width
            height_display = height
            unit = str(self._normalize_length_unit())
            top_mid = ((minx + maxx) * 0.5, miny - 12.0)
            right_mid = (maxx + 20.0, (miny + maxy) * 0.5)
            self._add_preview_text(f"W: {width_display:.3f} {unit}", top_mid)
            self._add_preview_text(f"H: {height_display:.3f} {unit}", right_mid)
        elif self.tool == "circle":
            r = dist(self.current[0], pt)
            try:
                item = self._safe_add_ellipse(
                    self.current[0][0] - r,
                    self.current[0][1] - r,
                    2 * r,
                    2 * r,
                    pen,
                    context="circle_preview",
                )
                if item:
                    self.preview_items.append(item)
            except Exception:
                pass
            try:
                self.preview_items.append(
                    self.scene().addLine(
                        self.current[0][0],
                        self.current[0][1],
                        pt[0],
                        pt[1],
                        pen,
                    )
                )
            except Exception:
                pass
            unit = str(self._normalize_length_unit())
            mid = ((self.current[0][0] + pt[0]) * 0.5, (self.current[0][1] + pt[1]) * 0.5 - 12.0)
            self._add_preview_text(f"R: {r:.3f} {unit}", mid)
        elif self.tool == "slot":
            verts = self._build_slot_vertices(self.current[0], pt, float(self._slot_width))
            if verts:
                path = QPainterPath()
                path.moveTo(*verts[0])
                for p in verts[1:]:
                    path.lineTo(*p)
                try:
                    self.preview_items.append(self.scene().addPath(path, pen))
                except Exception:
                    pass
        elif self.tool == "arc":
            if self.mode == "drawing_arc_1": # Drawing line from p1 to p2
                p0 = self.current[0]
                self.preview_items.append(self.scene().addLine(p0[0], p0[1], pt[0], pt[1], pen))
            elif self.mode == "drawing_arc_2": # Drawing arc through p1, p2, pt
                if len(self.current) < 2: return
                p1, p2 = self.current[0], self.current[1]
                arc_pts = self._calculate_arc_points(p1, pt, p2) # p1-start, p2-end, pt-mid
                if arc_pts and len(arc_pts) > 1:
                    path = QPainterPath(); path.moveTo(*arc_pts[0])
                    for p_arc in arc_pts[1:]: path.lineTo(*p_arc)
                    self.preview_items.append(self.scene().addPath(path, pen))

        self._draw_snap_indicator()

    def _clear_preview(self):
        scene = self.scene()
        for item in self.preview_items:
            try:
                if self._item_scene_safe(item):
                    scene.removeItem(item)
            except: pass
        self.preview_items = []

    def _item_scene_safe(self, item):
        if item is None:
            return None
        try:
            return item.scene()
        except RuntimeError:
            return None
        except Exception:
            return None

    def _remove_scene_item_safe(self, item):
        scene = self._item_scene_safe(item)
        if scene is None:
            return
        try:
            scene.removeItem(item)
        except RuntimeError:
            pass
        except Exception:
            pass

    def _dimension_item_at_view_pos(self, view_pos):
        try:
            item = self.itemAt(view_pos)
        except Exception:
            item = None
        while item is not None:
            if isinstance(item, DimensionTextItem):
                return item
            try:
                item = item.parentItem()
            except Exception:
                item = None
        return None

    def _clear_zoom_window(self):
        self._remove_scene_item_safe(self._zoom_window_item)
        self._zoom_window_item = None
        self._zoom_window_active = False
        self._zoom_window_start = None
        self._zoom_window_temporary = False

    def _collect_snap_points(self):
        data = {
            "endpoints": [],
            "midpoints": [],
            "centroids": [],
            "intersections": [],
            "edges": [],
        }

        for s_idx, s in enumerate(self.sketches):
            if not self._is_sketch_visible("sketch", None, s_idx):
                continue
            if len(s) < 2:
                continue
            for i in range(len(s) - 1):
                a, b = s[i], s[i + 1]
                if dist(a, b) < 1e-9:
                    continue
                data["endpoints"].extend([a, b])
                data["edges"].append((a, b))
                data["midpoints"].append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))

            if len(s) > 3 and dist(s[0], s[-1]) < 1e-6:
                try:
                    poly = Polygon(s)
                    if poly.is_valid and not poly.is_empty:
                        c = poly.centroid
                        data["centroids"].append((c.x, c.y))
                except Exception:
                    pass

        if self.parts and self.solid_geometry:
            verts, edges = get_solid_features(self.solid_geometry)
            data["endpoints"].extend(verts)
            data["edges"].extend(edges)
            for edge in edges:
                data["midpoints"].append(
                    ((edge[0][0] + edge[1][0]) / 2, (edge[0][1] + edge[1][1]) / 2)
                )
            for part in self.parts:
                if part.geometry and not part.geometry.is_empty:
                    c = part.geometry.centroid
                    data["centroids"].append((c.x, c.y))

        # Intersections from sketch edges only (keep it lightweight).
        segs = []
        for s_idx, s in enumerate(self.sketches):
            if not self._is_sketch_visible("sketch", None, s_idx):
                continue
            if len(s) < 2:
                continue
            for i in range(len(s) - 1):
                a, b = s[i], s[i + 1]
                if dist(a, b) > 1e-9:
                    segs.append((a, b))
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                pt = self._segment_intersection(segs[i][0], segs[i][1], segs[j][0], segs[j][1])
                if pt:
                    data["intersections"].append(pt)

        return data

    def _apply_snapping(self, pt, use_grid=True, use_endpoints=True):
        snapped = False
        snapped_pt = pt

        if use_endpoints and self.snap_endpoints:
            snap_data = self._collect_snap_points()
            for key in ("intersections", "endpoints", "midpoints", "centroids"):
                candidates = snap_data.get(key, [])
                if candidates:
                    nearest = min(candidates, key=lambda p: dist(pt, p))
                    if dist(pt, nearest) <= SNAP_TOL:
                        snapped_pt = nearest
                        snapped = True
                        break

            if not snapped and snap_data.get("edges"):
                best = None
                best_dist = SNAP_TOL
                for edge in snap_data["edges"]:
                    proj = self._project_to_edge(pt, edge[0], edge[1])
                    d = dist(pt, proj)
                    if d < best_dist:
                        best_dist = d
                        best = proj
                if best is not None:
                    snapped_pt = best
                    snapped = True

        if not snapped and use_grid and self.snap_grid:
            snapped_pt = (
                round(pt[0] / GRID_MINOR) * GRID_MINOR,
                round(pt[1] / GRID_MINOR) * GRID_MINOR,
            )
            snapped = True

        self._snap_indicator = snapped_pt if snapped else None
        return snapped_pt

    def _input_point(self, pt):
        if self.active_module != "Part":
            return pt
        if self.tool in ("freeform",):
            return pt
        return self._apply_snapping(pt, use_grid=True, use_endpoints=True)

    def _announce_status(self, message, timeout=5000):
        window = self.window()
        if window and hasattr(window, "statusBar"):
            window.statusBar().showMessage(message, timeout)

    def _queue_mesh_status(self, message, timeout=1500):
        self._mesh_status_pending = (message, timeout)
        if not self._mesh_status_timer.isActive():
            self._mesh_status_timer.start()

    def _flush_mesh_status(self):
        if not self._mesh_status_pending:
            return
        message, timeout = self._mesh_status_pending
        self._mesh_status_pending = None
        self._announce_status(message, timeout)

    def _update_rectangle_cursor_status(self, pt):
        if self.tool != "rectangle":
            return
        stage = "select first point"
        if self.mode == "idle":
            stage = (
                "select center point"
                if self._rect_draw_mode == "center"
                else "select first corner"
            )
        else:
            stage = (
                "select corner"
                if self._rect_draw_mode == "center"
                else "select opposite corner"
            )
        message = (
            f"Rectangle ({'2-corner' if self._rect_draw_mode != 'center' else 'center'}) · {stage}. "
            f"x={pt[0]:.3f}, y={pt[1]:.3f}"
        )
        if message == self._rect_cursor_message:
            return
        self._rect_cursor_message = message
        self._announce_status(message, timeout=800)

    def _format_duration(self, seconds):
        if seconds is None:
            return "--"
        try:
            seconds = float(seconds)
        except Exception:
            return "--"
        if seconds < 0:
            return "--"
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"

    def _estimate_mesh_time(self, n_points):
        if not n_points or n_points < 50:
            return None
        history = getattr(self, "_mesh_time_history", [])
        if not history:
            return None
        n0, t0 = history[-1]
        if not n0 or n0 < 50 or t0 <= 0:
            return None
        # scale roughly with n log n
        try:
            scale = (n_points * math.log(max(n_points, 2))) / (n0 * math.log(max(n0, 2)))
        except Exception:
            scale = n_points / max(n0, 1)
        est = t0 * scale
        return max(0.1, est)

    def get_mesh_backend_status(self):
        def _check(mod):
            try:
                return importlib.util.find_spec(mod) is not None
            except Exception:
                return False

        gmsh_ok = _check("gmsh")
        return {
            "auto": True,
            "triangle": _check("triangle"),
            "gmsh": gmsh_ok,
            "gmsh-2d-adaptive": gmsh_ok,
            "pygalmesh": _check("pygalmesh"),
            "scipy": _check("scipy"),
        }

    def estimate_mesh_nodes(self, dx, target_nodes, mesh_distribution):
        est, _ = self._estimate_mesh_node_count(dx, target_nodes, mesh_distribution)
        return est

    def _select_mesh_backend(self, backend, dx, target_nodes, mesh_distribution, allow_warn=True):
        status = self.get_mesh_backend_status()
        backend = (backend or "auto").lower()
        if backend == "auto":
            for choice in ("triangle", "gmsh", "pygalmesh", "scipy"):
                if status.get(choice, False):
                    return choice
            return "scipy"
        if status.get(backend, False):
            return backend
        if allow_warn:
            QMessageBox.warning(
                self,
                "Backend Missing",
                f"Selected backend '{backend}' is not available. Falling back to the fastest available.",
            )
        for choice in ("triangle", "gmsh", "pygalmesh", "scipy"):
            if status.get(choice, False):
                return choice
        return "scipy"

    def _triangulate_points(self, points, backend, segments=None, holes=None, regions=None, return_metadata=False):
        backend = (backend or "scipy").lower()
        use_segments = segments is not None and len(segments) > 0
        use_holes = holes is not None and len(holes) > 0
        use_regions = regions is not None and len(regions) > 0
        constrained = bool(use_segments or use_holes or use_regions)
        if constrained and backend in ("auto", "scipy", "gmsh", "pygalmesh"):
            try:
                import triangle  # noqa: F401
                backend = "triangle"
            except Exception:
                if backend != "triangle":
                    raise RuntimeError(
                        "Constrained 2D meshing requires the Triangle backend. "
                        "SciPy Delaunay does not preserve boundary segments, holes, or region seeds."
                    )
        if backend == "triangle":
            try:
                import triangle as tr
            except Exception as exc:
                raise RuntimeError(f"Triangle backend not available: {exc}") from exc
            data = {"vertices": np.asarray(points, dtype=float)}
            opts = "Q"
            if constrained:
                opts = "pQ"
            if use_segments:
                data["segments"] = np.asarray(segments, dtype=int)
            if use_holes:
                data["holes"] = np.asarray(holes, dtype=float)
            if use_regions:
                data["regions"] = np.asarray(regions, dtype=float)
                opts += "A"
            try:
                result = tr.triangulate(data, opts)
            except Exception as exc:
                raise RuntimeError(f"Triangle failed: {exc}") from exc
            verts_out = result.get("vertices")
            if verts_out is not None and len(verts_out) > len(points):
                raise RuntimeError(
                    "Triangle added Steiner vertices in constrained triangulation; current pipeline "
                    "expects fixed input vertices only."
                )
            tris = result.get("triangles")
            if tris is None or len(tris) == 0:
                raise RuntimeError("Triangle returned no triangles.")
            tris_arr = np.asarray(tris, dtype=int)
            if not return_metadata:
                return tris_arr
            triangle_attributes = result.get("triangle_attributes")
            if triangle_attributes is not None:
                try:
                    triangle_attributes = np.asarray(triangle_attributes, dtype=float).reshape(-1)
                except Exception:
                    triangle_attributes = None
                if triangle_attributes is not None and len(triangle_attributes) != len(tris_arr):
                    triangle_attributes = None
            return {
                "triangles": tris_arr,
                "triangle_attributes": triangle_attributes,
            }
        if backend in ("gmsh", "pygalmesh"):
            status = self.get_mesh_backend_status()
            if status.get("triangle", False):
                return self._triangulate_points(
                    points,
                    "triangle",
                    segments=segments,
                    holes=holes,
                    regions=regions,
                    return_metadata=return_metadata,
                )
            if constrained:
                raise RuntimeError(
                    "Constrained 2D meshing requires the Triangle backend. "
                    "SciPy Delaunay cannot honor boundary segments, holes, or region seeds."
                )
            return self._triangulate_points(
                points,
                "scipy",
                segments=segments,
                holes=holes,
                regions=regions,
                return_metadata=return_metadata,
            )
        if constrained:
            raise RuntimeError(
                "Constrained 2D meshing requires the Triangle backend. "
                "SciPy Delaunay cannot honor boundary segments, holes, or region seeds."
            )
        try:
            tri = Delaunay(np.asarray(points, dtype=float))
            tris_arr = np.asarray(tri.simplices, dtype=int)
            if return_metadata:
                return {"triangles": tris_arr, "triangle_attributes": None}
            return tris_arr
        except Exception as exc:
            raise RuntimeError(f"SciPy Delaunay failed: {exc}") from exc

    def _mesh_density_factor(self, mesh_distribution):
        if mesh_distribution and "square" in mesh_distribution:
            return 1.1
        return 1.25

    def _estimate_nodes_from_area(self, area, dx, mesh_distribution):
        if area is None or area <= 0 or dx is None or dx <= 0:
            return None
        factor = self._mesh_density_factor(mesh_distribution)
        return int(max(1, (area * factor) / max(dx * dx, 1e-12)))

    def _estimate_mesh_node_count(self, dx, target_nodes, mesh_distribution, parts=None):
        if target_nodes is not None:
            try:
                return int(max(1, target_nodes)), None
            except Exception:
                return None, None
        if dx is None or dx <= 0:
            return None, None
        parts = parts or [p for p in self.parts if p.material_id is not None and not p.is_void]
        total_area = 0.0
        for part in parts:
            geom = getattr(part, "geometry", None)
            if geom is None or geom.is_empty:
                continue
            try:
                total_area += float(geom.area)
            except Exception:
                continue
        if total_area <= 0:
            return None, None
        est = self._estimate_nodes_from_area(total_area, dx, mesh_distribution)
        return est, total_area

    def _guard_mesh_density(self, dx, target_nodes, mesh_distribution):
        est, area = self._estimate_mesh_node_count(dx, target_nodes, mesh_distribution)
        if est is None:
            return dx, target_nodes, True
        soft_limit = int(getattr(self, "mesh_node_soft_limit", MESH_NODE_SOFT_LIMIT))
        hard_limit = int(getattr(self, "mesh_node_hard_limit", MESH_NODE_HARD_LIMIT))
        if est <= soft_limit:
            return dx, target_nodes, True

        unit = self.current_unit or ""
        header = "Connection Density Warning"
        text = f"Estimated particle count: ~{est:,}."
        if dx is not None:
            text += f" (dx={dx:.4g} {unit})"
        if target_nodes is not None:
            text += f" (target={int(target_nodes):,})"
        info = "Large particle sets can crash or freeze the UI."

        suggested_dx = None
        if area and area > 0 and dx is not None:
            factor = self._mesh_density_factor(mesh_distribution)
            suggested_dx = math.sqrt((area * factor) / max(soft_limit, 1))
            suggested_dx = max(0.1, suggested_dx)
            info += f"\nSuggested dx ≥ {suggested_dx:.3f} {unit}."
        if target_nodes is not None:
            info += f"\nSuggested target particles ≤ {soft_limit:,}."

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle(header)
        msg.setText(text)
        msg.setInformativeText(info)

        adjust_btn = None
        target_btn = None
        if suggested_dx is not None:
            adjust_btn = msg.addButton("Adjust dx", QMessageBox.AcceptRole)
        target_btn = msg.addButton("Use target particles", QMessageBox.ActionRole)
        continue_btn = None
        if est < hard_limit:
            continue_btn = msg.addButton("Continue anyway", QMessageBox.DestructiveRole)
        cancel_btn = msg.addButton(QMessageBox.Cancel)
        msg.setDefaultButton(cancel_btn)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked == cancel_btn:
            return dx, target_nodes, False
        if clicked == adjust_btn and suggested_dx is not None:
            return suggested_dx, None, True
        if clicked == target_btn:
            return None, soft_limit, True
        if clicked == continue_btn:
            return dx, target_nodes, True

        return dx, target_nodes, False

    def _reset_transform_state(self):
        self._transform_tool = None
        self._transform_base = None
        self._transform_line = []
        self._transform_drag_active = False
        self._transform_drag_moved = False
        self._transform_drag_start_view = None
        self._clear_transform_preview()

    def _clear_transform_preview(self):
        self._remove_scene_item_safe(self._transform_preview_item)
        self._transform_preview_item = None

    def _update_transform_preview(self, p1, p2):
        if not p1 or not p2:
            return
        self._remove_scene_item_safe(self._transform_preview_item)
        pen = QPen(QColor(0, 120, 255, 180), 1.5, Qt.DashLine)
        try:
            self._transform_preview_item = self.scene().addLine(p1[0], p1[1], p2[0], p2[1], pen)
        except RuntimeError:
            self._transform_preview_item = None

    def _clear_arc_segment_selection(self):
        self._arc_select_points = []
        self._arc_segment_polyline = None
        self._remove_scene_item_safe(self._arc_segment_preview_item)
        self._arc_segment_preview_item = None

    def _update_arc_segment_preview(self, points, provisional=False):
        self._remove_scene_item_safe(self._arc_segment_preview_item)
        if points is None or len(points) < 2:
            self._arc_segment_preview_item = None
            return
        path = QPainterPath()
        path.moveTo(*points[0])
        for p in points[1:]:
            path.lineTo(*p)
        if provisional:
            pen = QPen(QColor(0, 120, 255, 160), 1.5, Qt.DashLine)
        else:
            pen = QPen(QColor(0, 120, 255), 2.0, Qt.DashLine)
        try:
            self._arc_segment_preview_item = self.scene().addPath(path, pen)
        except Exception:
            self._arc_segment_preview_item = None

    def _get_boundary_rings(self):
        if self.solid_geometry is None or self.solid_geometry.is_empty:
            return []
        geoms = [self.solid_geometry] if isinstance(self.solid_geometry, Polygon) else list(self.solid_geometry.geoms)
        rings = []
        for g in geoms:
            if not isinstance(g, Polygon):
                continue
            exterior = list(g.exterior.coords)
            if exterior:
                rings.append(exterior)
            for interior in g.interiors:
                coords = list(interior.coords)
                if coords:
                    rings.append(coords)
        return rings

    def _project_point_to_ring(self, pt, ring, cum, seg_lengths):
        best = None
        for i in range(len(ring) - 1):
            ax, ay = ring[i]
            bx, by = ring[i + 1]
            dx = bx - ax
            dy = by - ay
            len_sq = dx * dx + dy * dy
            if len_sq <= 1e-12:
                continue
            t = ((pt[0] - ax) * dx + (pt[1] - ay) * dy) / len_sq
            t = max(0.0, min(1.0, t))
            proj = (ax + t * dx, ay + t * dy)
            d = dist(pt, proj)
            s = cum[i] + t * seg_lengths[i]
            if best is None or d < best[0]:
                best = (d, s, proj)
        return best

    def _point_at_ring_s(self, ring, cum, seg_lengths, s):
        if not ring or len(ring) < 2:
            return None
        total = cum[-1]
        if total <= 1e-12:
            return ring[0]
        if s <= 0.0:
            return ring[0]
        if s >= total:
            return ring[-1]
        idx = min(len(seg_lengths) - 1, max(0, bisect_right(cum, s) - 1))
        seg_len = seg_lengths[idx]
        if seg_len <= 1e-12:
            return ring[idx]
        t = (s - cum[idx]) / seg_len
        ax, ay = ring[idx]
        bx, by = ring[idx + 1]
        return (ax + t * (bx - ax), ay + t * (by - ay))

    def _collect_ring_segment_forward(self, ring, cum, seg_lengths, s_start, s_end):
        total = cum[-1]
        if total <= 1e-12:
            return []

        def collect_no_wrap(a, b):
            if b < a:
                return []
            start_pt = self._point_at_ring_s(ring, cum, seg_lengths, a)
            end_pt = self._point_at_ring_s(ring, cum, seg_lengths, b)
            if start_pt is None or end_pt is None:
                return []
            i_start = min(len(seg_lengths) - 1, max(0, bisect_right(cum, a) - 1))
            i_end = min(len(seg_lengths) - 1, max(0, bisect_right(cum, b) - 1))
            pts = [start_pt]
            for i in range(i_start + 1, i_end + 1):
                pts.append(ring[i])
            if not pts or dist(pts[-1], end_pt) > 1e-6:
                pts.append(end_pt)
            return pts

        if s_end >= s_start:
            return collect_no_wrap(s_start, s_end)
        first = collect_no_wrap(s_start, total)
        second = collect_no_wrap(0.0, s_end)
        if first and second and dist(first[-1], second[0]) <= 1e-6:
            second = second[1:]
        return first + second

    def _downsample_polyline(self, points, max_points=80):
        if points is None or len(points) == 0:
            return points
        if len(points) <= max_points:
            return points
        idxs = np.linspace(0, len(points) - 1, max_points, dtype=int)
        sampled = [points[i] for i in idxs]
        if sampled[-1] != points[-1]:
            sampled[-1] = points[-1]
        return sampled

    def _polyline_to_segments(self, points):
        segments = []
        for i in range(len(points) - 1):
            if dist(points[i], points[i + 1]) <= 1e-9:
                continue
            segments.append((points[i], points[i + 1]))
        return segments

    def _segment_key(self, segment):
        if not isinstance(segment, (list, tuple)) or len(segment) != 2:
            return None
        a, b = segment
        if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
            return None
        if len(a) != 2 or len(b) != 2:
            return None
        if a > b:
            a, b = b, a
        return (round(a[0], 6), round(a[1], 6), round(b[0], 6), round(b[1], 6))

    def _remove_attrs_for_segments(self, segments):
        if not segments:
            return
        seg_keys = {self._segment_key(seg) for seg in segments}
        self.bcs = [
            b for b in self.bcs
            if self._segment_key(b.get("coords")) not in seg_keys
        ]
        self.loads = [
            l for l in self.loads
            if self._segment_key(l.get("coords")) not in seg_keys
        ]

    def _select_arc_segment(self, p_start, p_end, p_mid):
        rings = self._get_boundary_rings()
        if not rings:
            return None

        tol = SNAP_TOL * 2.5
        best = None
        fallback = None
        for ring in rings:
            ring_closed = ring
            if ring[0] != ring[-1]:
                ring_closed = ring + [ring[0]]
            if len(ring_closed) < 2:
                continue
            seg_lengths = []
            cum = [0.0]
            for i in range(len(ring_closed) - 1):
                seg_len = dist(ring_closed[i], ring_closed[i + 1])
                seg_lengths.append(seg_len)
                cum.append(cum[-1] + seg_len)
            total = cum[-1]
            if total <= 1e-9:
                continue
            proj_start = self._project_point_to_ring(p_start, ring_closed, cum, seg_lengths)
            proj_end = self._project_point_to_ring(p_end, ring_closed, cum, seg_lengths)
            proj_mid = self._project_point_to_ring(p_mid, ring_closed, cum, seg_lengths)
            if not proj_start or not proj_end or not proj_mid:
                continue
            ds, s_start, p_start_proj = proj_start
            de, s_end, p_end_proj = proj_end
            dm, s_mid, p_mid_proj = proj_mid
            score = ds + de + dm
            if fallback is None or score < fallback["score"]:
                fallback = {
                    "ring": ring_closed,
                    "cum": cum,
                    "seg_lengths": seg_lengths,
                    "s_start": s_start,
                    "s_end": s_end,
                    "s_mid": s_mid,
                    "p_start": p_start_proj,
                    "p_end": p_end_proj,
                    "score": score,
                }
            if max(ds, de, dm) <= tol:
                if best is None or score < best["score"]:
                    best = {
                        "ring": ring_closed,
                        "cum": cum,
                        "seg_lengths": seg_lengths,
                        "s_start": s_start,
                        "s_end": s_end,
                        "s_mid": s_mid,
                        "p_start": p_start_proj,
                        "p_end": p_end_proj,
                        "score": score,
                    }

        selection = best or fallback
        if not selection:
            return None
        if selection["score"] > SNAP_TOL * 6:
            return None

        ring = selection["ring"]
        cum = selection["cum"]
        seg_lengths = selection["seg_lengths"]
        s_start = selection["s_start"]
        s_end = selection["s_end"]
        s_mid = selection["s_mid"]
        total = cum[-1]
        forward_len = (s_end - s_start) % total
        mid_forward = (s_mid - s_start) % total
        forward_ok = mid_forward <= forward_len
        backward_len = total - forward_len
        mid_backward = (s_mid - s_end) % total
        backward_ok = mid_backward <= backward_len

        if forward_ok or not backward_ok:
            points = self._collect_ring_segment_forward(ring, cum, seg_lengths, s_start, s_end)
        else:
            points = list(reversed(self._collect_ring_segment_forward(ring, cum, seg_lengths, s_end, s_start)))
        if points is None or len(points) < 2:
            return None
        points[0] = selection["p_start"]
        points[-1] = selection["p_end"]
        return points

    def _handle_arc_segment_click(self, pt):
        if self.solid_geometry is None or self.solid_geometry.is_empty:
            self._announce_status("No geometry available for arc segment selection.")
            return
        self._arc_select_points.append(pt)
        if len(self._arc_select_points) < 3:
            self._update_arc_segment_preview(self._arc_select_points, provisional=True)
            return
        p_start, p_end, p_mid = self._arc_select_points[:3]
        self._arc_select_points = []
        points = self._select_arc_segment(p_start, p_end, p_mid)
        if points is None or len(points) == 0:
            self._update_arc_segment_preview([], provisional=True)
            self._announce_status("Could not resolve arc segment. Try clicking closer to the boundary.")
            return
        points = self._downsample_polyline(points, max_points=80)
        self._arc_segment_polyline = points
        self._update_arc_segment_preview([], provisional=False)
        self.redraw()
        if self.active_module in ("Load", "Boundary"):
            if self.active_module == "Boundary":
                self._announce_status("Arc segment selected. Right-click to apply BC.")
            else:
                self._announce_status("Arc segment selected. Right-click to apply load.")
        else:
            self._announce_status("Arc segment selected.")

    def _add_bc_polyline(self, bctype, points):
        segments = self._polyline_to_segments(points)
        if not segments:
            return
        self.push_undo_state()
        self._remove_attrs_for_segments(segments)
        for seg in segments:
            self.bcs.append({"type": bctype, "coords": seg})
        self.bcsChanged.emit()
        self.redraw()
        self._clear_arc_segment_selection()

    def _add_velocity_polyline(self, axis, points):
        self._apply_time_profile_polyline("velocity", axis, points)

    def _add_velocity_vector_polyline(self, points):
        self._apply_time_profile_polyline("velocity", "x", points)
        self._apply_time_profile_polyline("velocity", "y", points)

    def _add_load_polyline(self, ltype, points):
        if ltype != "force":
            return
        self._apply_time_profile_polyline("force", "x", points)
        self._apply_time_profile_polyline("force", "y", points)

    def _add_displacement_polyline(self, axis, points):
        axis = str(axis or "x").strip().lower()
        val, ok = QInputDialog.getDouble(
            self,
            "Displacement",
            f"Displacement {axis.upper()} ({self.current_unit or 'm'}):",
            0.0,
            -1e9,
            1e9,
            6,
        )
        if not ok:
            return
        segments = self._polyline_to_segments(points)
        if not segments:
            return
        self.push_undo_state()
        self._remove_attrs_for_segments(segments)
        for seg in segments:
            self.bcs.append(
                {
                    "type": f"velocity_{axis}",
                    "coords": seg,
                    "val": val,
                    "bc_mode": "displacement",
                    "display_type": f"Displacement U{axis.upper()}",
                }
            )
        self.bcsChanged.emit()
        self.redraw()
        self._clear_arc_segment_selection()

    def _add_force_component_polyline(self, axis, points):
        axis = str(axis or "x").strip().lower()
        val, ok = QInputDialog.getDouble(
            self,
            f"Force {axis.upper()}",
            f"Force {axis.upper()} (N):",
            0.0,
            -1e9,
            1e9,
            3,
        )
        if not ok:
            return
        segments = self._polyline_to_segments(points)
        if not segments:
            return
        self.push_undo_state()
        self._remove_attrs_for_segments(segments)
        for seg in segments:
            record = {
                "type": "force",
                "coords": seg,
                "fx": 0.0,
                "fy": 0.0,
                "display_type": f"Force F{axis.upper()}",
                "axis": axis,
            }
            if axis == "x":
                record["fx"] = val
            elif axis == "y":
                record["fy"] = val
            else:
                record["fz"] = val
            self.loads.append(record)
        self.loadsChanged.emit()
        self.redraw()
        self._clear_arc_segment_selection()

    def _apply_time_profile_polyline(self, kind, axis, points):
        segments = self._polyline_to_segments(points)
        if not segments:
            return
        main = self.window()
        if not main or not hasattr(main, "properties_panel"):
            return
        panel = main.properties_panel
        if kind == "velocity" and hasattr(panel, "bcs_tab"):
            panel.bcs_tab._open_profile_editor(kind, axis, coords=segments)
            return
        if hasattr(panel, "loads_tab"):
            panel.loads_tab._open_profile_editor(kind, axis, coords=segments)

    def _show_arc_segment_menu(self, global_pos):
        if not self._arc_segment_polyline:
            return
        menu = QMenu(self)
        menu.addAction("Arc Segment").setEnabled(False)
        menu.addSeparator()
        menu.addAction("Fix Edge", lambda: self._add_bc_polyline("fix_xy", self._arc_segment_polyline))
        menu.addAction("Displacement X", lambda: self._add_displacement_polyline("x", self._arc_segment_polyline))
        menu.addAction("Displacement Y", lambda: self._add_displacement_polyline("y", self._arc_segment_polyline))
        menu.addAction("Velocity X", lambda: self._apply_time_profile_polyline("velocity", "x", self._arc_segment_polyline))
        menu.addAction("Velocity Y", lambda: self._apply_time_profile_polyline("velocity", "y", self._arc_segment_polyline))
        if getattr(self, "project_mode", "2d") == "3d":
            menu.addAction("Velocity Z", lambda: self._apply_time_profile_polyline("velocity", "z", self._arc_segment_polyline))
        menu.addAction("Force X", lambda: self._add_force_component_polyline("x", self._arc_segment_polyline))
        menu.addAction("Force Y", lambda: self._add_force_component_polyline("y", self._arc_segment_polyline))
        if getattr(self, "project_mode", "2d") == "3d":
            menu.addAction("Force Z", lambda: self._add_force_component_polyline("z", self._arc_segment_polyline))
        menu.addSeparator()
        menu.addAction("Clear Arc Selection", self._clear_arc_segment_selection)
        menu.exec(global_pos)
        menu.close()

    def _prompt_line_params(self):
        """Show a dialog asking for start (X1, Y1) and end (X2, Y2) coordinates
        to create a line.  Returns a dict ``{'start': (x1, y1), 'end': (x2, y2)}``
        on acceptance, or ``None`` if the user cancels."""

        dialog = QDialog(self)
        dialog.setWindowTitle("Create Line — Enter Coordinates")
        dialog.setMinimumWidth(340)
        layout = QVBoxLayout(dialog)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        unit = getattr(self, "current_unit", "mm")
        default_val = self._practical_sketch_default(200.0)

        # --- header label ---
        header = QLabel(f"Define line by start and end coordinates ({unit})")
        header.setStyleSheet("font-weight: 700; font-size: 13px; color: #1a1f29;")
        layout.addWidget(header)

        # --- Start-point group ---
        start_group = QGroupBox("Start Point")
        start_form = QFormLayout(start_group)
        start_form.setContentsMargins(10, 14, 10, 8)
        start_form.setSpacing(8)

        spin_x1 = QDoubleSpinBox()
        spin_x1.setRange(-1e9, 1e9)
        spin_x1.setDecimals(4)
        spin_x1.setValue(0.0)
        spin_x1.setMinimumHeight(30)
        spin_x1.setToolTip("X coordinate of the start point")
        start_form.addRow(f"X₁ ({unit}):", spin_x1)

        spin_y1 = QDoubleSpinBox()
        spin_y1.setRange(-1e9, 1e9)
        spin_y1.setDecimals(4)
        spin_y1.setValue(0.0)
        spin_y1.setMinimumHeight(30)
        spin_y1.setToolTip("Y coordinate of the start point")
        start_form.addRow(f"Y₁ ({unit}):", spin_y1)

        layout.addWidget(start_group)

        # --- End-point group ---
        end_group = QGroupBox("End Point")
        end_form = QFormLayout(end_group)
        end_form.setContentsMargins(10, 14, 10, 8)
        end_form.setSpacing(8)

        spin_x2 = QDoubleSpinBox()
        spin_x2.setRange(-1e9, 1e9)
        spin_x2.setDecimals(4)
        spin_x2.setValue(default_val)
        spin_x2.setMinimumHeight(30)
        spin_x2.setToolTip("X coordinate of the end point")
        end_form.addRow(f"X₂ ({unit}):", spin_x2)

        spin_y2 = QDoubleSpinBox()
        spin_y2.setRange(-1e9, 1e9)
        spin_y2.setDecimals(4)
        spin_y2.setValue(0.0)
        spin_y2.setMinimumHeight(30)
        spin_y2.setToolTip("Y coordinate of the end point")
        end_form.addRow(f"Y₂ ({unit}):", spin_y2)

        layout.addWidget(end_group)

        # --- Length / Angle preview ---
        preview_label = QLabel()
        preview_label.setStyleSheet(
            "color: #5f6b76; font-size: 11px; padding: 4px 0px;"
        )

        def _update_preview():
            dx = spin_x2.value() - spin_x1.value()
            dy = spin_y2.value() - spin_y1.value()
            length = math.sqrt(dx * dx + dy * dy)
            angle = math.degrees(math.atan2(dy, dx))
            preview_label.setText(
                f"Length: {length:.4f} {unit}   |   Angle: {angle:.2f}°"
            )

        spin_x1.valueChanged.connect(lambda _: _update_preview())
        spin_y1.valueChanged.connect(lambda _: _update_preview())
        spin_x2.valueChanged.connect(lambda _: _update_preview())
        spin_y2.valueChanged.connect(lambda _: _update_preview())
        _update_preview()
        layout.addWidget(preview_label)

        # --- buttons ---
        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)

        if dialog.exec() != QDialog.Accepted:
            return None

        start = (spin_x1.value(), spin_y1.value())
        end = (spin_x2.value(), spin_y2.value())

        # Guard against zero-length lines
        if math.isclose(start[0], end[0], abs_tol=1e-9) and math.isclose(start[1], end[1], abs_tol=1e-9):
            QMessageBox.warning(
                self, "Create Line",
                "Start and end points are identical — cannot create a zero-length line.",
            )
            return None

        return {"start": start, "end": end}

    def _prompt_rect_params(self):
        width, ok = QInputDialog.getDouble(
            self,
            "Rectangle Width",
            f"Width ({self.current_unit}):",
            self._practical_sketch_default(300.0),
            -1e9,
            1e9,
            3,
        )
        if not ok:
            return None
        height, ok = QInputDialog.getDouble(
            self,
            "Rectangle Height",
            f"Height ({self.current_unit}):",
            self._practical_sketch_default(200.0),
            -1e9,
            1e9,
            3,
        )
        if not ok:
            return None
        return width, height

    def _prompt_rectangle_mode(self):
        options = [
            "Corner (2-point)",
            "Center (2-point)",
            "Corner + dimensions",
            "Center + dimensions",
        ]
        default_map = {
            ("corner", False): 0,
            ("center", False): 1,
            ("corner", True): 2,
            ("center", True): 3,
        }
        default_idx = default_map.get((self._rect_draw_mode, self._rect_use_dimensions), 0)
        choice, ok = QInputDialog.getItem(
            self, "Rectangle", "Method:", options, default_idx, False
        )
        if not ok:
            return None
        if choice.startswith("Corner"):
            mode = "corner"
        else:
            mode = "center"
        use_dims = "dimensions" in choice
        return mode, use_dims

    def _prompt_circle_radius(self):
        radius, ok = QInputDialog.getDouble(
            self,
            "Circle Radius",
            f"Radius ({self.current_unit}):",
            self._practical_sketch_default(150.0),
            0.0,
            1e9,
            3,
        )
        if not ok:
            return None
        return radius

    def _prompt_slot_params(self):
        length, ok = QInputDialog.getDouble(
            self,
            "Slot Length",
            f"Length ({self.current_unit}):",
            self._practical_sketch_default(300.0),
            0.1,
            1e9,
            3,
        )
        if not ok:
            return None
        angle, ok = QInputDialog.getDouble(
            self,
            "Slot Angle",
            "Angle (deg):",
            0.0,
            -360.0,
            360.0,
            2,
        )
        if not ok:
            return None
        width, ok = QInputDialog.getDouble(
            self,
            "Slot Width",
            f"Width ({self.current_unit}):",
            self._practical_sketch_default(100.0, self._slot_width),
            0.1,
            1e9,
            3,
        )
        if not ok:
            return None
        return length, angle, width

    def _prompt_polygon_params(self):
        sides, ok = QInputDialog.getInt(
            self,
            "Polygon",
            "Sides:",
            self.polygon_sides,
            3,
            360,
        )
        if not ok:
            return None
        radius, ok = QInputDialog.getDouble(
            self,
            "Polygon Radius",
            f"Radius ({self.current_unit}):",
            self._practical_sketch_default(150.0),
            0.0,
            1e9,
            3,
        )
        if not ok:
            return None
        return sides, radius

    def _erase_sketch_at(self, pt):
        for si, sketch in enumerate(self.sketches):
            if not self._is_sketch_visible("sketch", None, si):
                continue
            if len(sketch) < 2:
                continue
            for i in range(len(sketch) - 1):
                if point_line_dist(pt, sketch[i], sketch[i + 1]) <= ERASE_TOL:
                    self.push_undo_state()
                    self._remove_sketch(si)
                    self.geometryChanged.emit()
                    self.redraw()
                    return True
        return False

    def _trim_sketch_at(self, pt):
        for si, sketch in enumerate(self.sketches):
            if not self._is_sketch_visible("sketch", None, si):
                continue
            if len(sketch) < 2:
                continue
            for i in range(len(sketch) - 1):
                if point_line_dist(pt, sketch[i], sketch[i + 1]) <= ERASE_TOL:
                    self.push_undo_state()
                    left = sketch[: i + 1]
                    right = sketch[i + 1 :]
                    new_sketches = []
                    if len(left) >= 2:
                        new_sketches.append(left)
                    if len(right) >= 2:
                        new_sketches.append(right)
                    self.sketches.pop(si)
                    for offset, s in enumerate(new_sketches):
                        self.sketches.insert(si + offset, s)
                    self.sketch_meta = []
                    self.dimensions = []
                    self.constraints = []
                    self._sync_all_sketch_meta()
                    self.geometryChanged.emit()
                    self.redraw()
                    return True
        return False

    def join_sketches(self):
        if len(self.sketches) < 2:
            QMessageBox.information(self, "Join Sketches", "Need at least two sketches to join.")
            return False
        self.push_undo_state()
        sketches = [list(s) for s in self.sketches if len(s) >= 2]
        tol = SNAP_TOL * 1.5
        changed_any = False
        merged = True
        while merged:
            merged = False
            for i in range(len(sketches)):
                if merged:
                    break
                for j in range(i + 1, len(sketches)):
                    s1 = sketches[i]
                    s2 = sketches[j]
                    if dist(s1[-1], s2[0]) <= tol:
                        sketches[i] = s1 + s2[1:]
                    elif dist(s1[-1], s2[-1]) <= tol:
                        sketches[i] = s1 + list(reversed(s2[:-1]))
                    elif dist(s1[0], s2[-1]) <= tol:
                        sketches[i] = s2 + s1[1:]
                    elif dist(s1[0], s2[0]) <= tol:
                        sketches[i] = list(reversed(s2)) + s1[1:]
                    else:
                        continue
                    sketches.pop(j)
                    merged = True
                    changed_any = True
                    break
        if not changed_any:
            QMessageBox.information(self, "Join Sketches", "No endpoints were close enough to join.")
            return False
        self.sketches = sketches
        self.sketch_meta = []
        self.dimensions = []
        self.constraints = []
        self._sync_all_sketch_meta()
        self.geometryChanged.emit()
        self.redraw()
        return True

    def _transform_point(self, pt, dx=0.0, dy=0.0, mirror_axis=None, mirror_offset=0.0, mirror_line=None):
        x, y = pt
        if mirror_line:
            (x1, y1), (x2, y2) = mirror_line
            dxl = x2 - x1
            dyl = y2 - y1
            denom = dxl * dxl + dyl * dyl
            if denom > 0:
                t = ((x - x1) * dxl + (y - y1) * dyl) / denom
                proj_x = x1 + t * dxl
                proj_y = y1 + t * dyl
                x = 2 * proj_x - x
                y = 2 * proj_y - y
        if mirror_axis == "vertical":
            x = 2 * mirror_offset - x
        elif mirror_axis == "horizontal":
            y = 2 * mirror_offset - y
        return (x + dx, y + dy)

    def _transform_points(self, points, dx=0.0, dy=0.0, mirror_axis=None, mirror_offset=0.0, mirror_line=None):
        return [
            self._transform_point(p, dx, dy, mirror_axis, mirror_offset, mirror_line)
            for p in points
        ]

    def _transform_meta(self, meta, points, dx=0.0, dy=0.0, mirror_axis=None, mirror_offset=0.0, mirror_line=None):
        if mirror_axis or mirror_line:
            transformed = self._transform_points(points, dx, dy, mirror_axis, mirror_offset, mirror_line)
            return {"type": "polyline", "points": transformed}
        meta = copy.deepcopy(meta or {})
        meta_type = str(meta.get("type", "polyline")).lower()
        for key in ("p1", "p2", "center", "origin"):
            if key in meta and meta[key] is not None:
                meta[key] = self._transform_point(meta[key], dx, dy)
        if "points" in meta:
            meta["points"] = self._transform_points(meta.get("points", []), dx, dy)
        if meta_type == "rectangle":
            return self._normalize_rectangle_meta(meta, fallback_points=points)
        return meta

    def _clone_part_with_geometry(self, part, geometry, sketches, metas, name):
        new_part = Part(name, geometry=geometry, is_void=part.is_void)
        new_part.material_id = part.material_id
        new_part.material_type = part.material_type
        new_part.material_props = copy.deepcopy(getattr(part, "material_props", {}))
        new_part.parent_id = part.parent_id
        new_part.is_rigid = part.is_rigid
        new_part.is_direct_edit = True
        new_part.cad_source = copy.deepcopy(getattr(part, "cad_source", None))
        new_part.sketches = copy.deepcopy(sketches)
        new_part.sketch_meta = copy.deepcopy(metas)
        new_part.dimensions = copy.deepcopy(getattr(part, "dimensions", []))
        new_part.constraints = copy.deepcopy(getattr(part, "constraints", []))
        new_part.storage_units = "ui"
        self._sync_cad_shape(new_part)
        return new_part

    def _sketch_at_point(self, pt, tol_px=6.0):
        """Return the index of an unconfirmed sketch (in self.sketches) whose
        polyline passes within `tol_px` of the given scene point, or None if
        no sketch is near enough. Used by the drag-to-move handler so sketches
        that haven't been turned into a Part yet are still movable."""
        if not pt or not getattr(self, "sketches", None):
            return None
        try:
            tol = float(self._scene_units_for_pixels(tol_px))
        except Exception:
            tol = float(tol_px)
        sample = (float(pt[0]), float(pt[1]))
        best_idx = None
        best_dist = tol
        for idx, sketch in enumerate(self.sketches or []):
            if not sketch or len(sketch) < 1:
                continue
            for i in range(len(sketch)):
                a = (float(sketch[i][0]), float(sketch[i][1]))
                if i + 1 < len(sketch):
                    b = (float(sketch[i + 1][0]), float(sketch[i + 1][1]))
                    d = point_line_dist(sample, a, b)
                else:
                    d = math.hypot(sample[0] - a[0], sample[1] - a[1])
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
        return best_idx

    def _move_sketch_by_delta(self, sketch_index, dx, dy):
        """Translate sketch points (and meta) by (dx, dy)."""
        if sketch_index is None or sketch_index < 0:
            return False
        if sketch_index >= len(self.sketches or []):
            return False
        sketch = self.sketches[sketch_index]
        if not sketch:
            return False
        self.sketches[sketch_index] = self._transform_points(sketch, dx, dy)
        if sketch_index < len(self.sketch_meta or []):
            meta = self.sketch_meta[sketch_index] or {}
            self.sketch_meta[sketch_index] = self._transform_meta(meta, sketch, dx, dy)
        self.geometryChanged.emit()
        self.redraw()
        return True

    def combine_selected_parts(self):
        """Union all multi-selected parts (Ctrl+Click selections) into a
        single part. Bound to Ctrl+P. Falls back to the currently selected
        part if no multi-selection exists."""
        ids = set(getattr(self, "multi_selected_part_ids", None) or set())
        if self.selected_part_id is not None:
            try:
                ids.add(int(self.selected_part_id))
            except Exception:
                pass
        if len(ids) < 2:
            QMessageBox.information(
                self,
                "Combine Parts",
                "Select at least 2 parts (Ctrl+Click) before pressing Ctrl+P.",
            )
            return False
        parts_to_merge = [
            p for p in self.parts
            if int(getattr(p, "id", -1)) in ids and p.geometry is not None
        ]
        if len(parts_to_merge) < 2:
            return False
        self.push_undo_state()
        try:
            merged_geom = unary_union([p.geometry for p in parts_to_merge])
        except Exception as exc:
            QMessageBox.warning(self, "Combine Parts", f"Could not merge geometries: {exc}")
            return False
        primary = parts_to_merge[0]
        primary.geometry = merged_geom
        # Sketches / dimensions / constraints become invalid after a boolean
        # union — drop them and treat the merged solid as direct-edit geometry.
        primary.sketches = []
        primary.sketch_meta = []
        primary.dimensions = []
        primary.constraints = []
        primary.is_direct_edit = True
        # Drop interfaces and child references that pointed at the removed
        # parts, then physically remove those parts from the model.
        removed_ids = {int(getattr(p, "id", -1)) for p in parts_to_merge[1:]}
        self.parts = [p for p in self.parts if int(getattr(p, "id", -1)) not in removed_ids]
        self.interfaces = [
            iface for iface in self.interfaces
            if iface.part1_id not in removed_ids and iface.part2_id not in removed_ids
        ]
        for child in self.parts:
            if getattr(child, "parent_id", None) in removed_ids:
                child.parent_id = int(primary.id)
        self.multi_selected_part_ids = {int(primary.id)}
        self.set_selected_part(primary.id, emit_signal=True)
        self.rebuild_display_geometry()
        self.partsChanged.emit()
        self.geometryChanged.emit()
        self.redraw()
        if self.window() and hasattr(self.window(), "statusBar"):
            self.window().statusBar().showMessage(
                f"Combined {len(parts_to_merge)} parts into '{primary.name}'.", 5000
            )
        # Slide-in toast for instant feedback (replaces a static status line).
        if self.window() and hasattr(self.window(), "show_toast"):
            try:
                self.window().show_toast(
                    f"Combined {len(parts_to_merge)} parts into '{primary.name}'.",
                    kind="success",
                )
            except Exception:
                pass
        return True

    def _compute_alignment_snap(self, part_id, dx, dy):
        """Compute alignment snap when dragging a part. Compares the dragged
        part's bbox edges and center against every other part's bbox edges
        and center. If a candidate is within `tol_scene` units, snap to it
        and record a guide line. Returns (dx_adj, dy_adj, guides_list)."""
        try:
            pid = int(part_id)
        except Exception:
            return dx, dy, []
        dragged = next((p for p in self.parts if int(getattr(p, "id", -1)) == pid), None)
        if dragged is None or getattr(dragged, "geometry", None) is None:
            return dx, dy, []
        try:
            minx0, miny0, maxx0, maxy0 = dragged.geometry.bounds
        except Exception:
            return dx, dy, []
        # Tentative new bbox of the dragged part (after the proposed delta).
        minx = minx0 + dx
        maxx = maxx0 + dx
        miny = miny0 + dy
        maxy = maxy0 + dy
        cx = (minx + maxx) * 0.5
        cy = (miny + maxy) * 0.5
        # Snap tolerance in scene units = ~7 screen pixels.
        zoom = max(getattr(self, "_zoom", 1.0) or 1.0, 1e-6)
        tol_scene = 7.0 / zoom
        best_v = None  # (orient='v', value, dx_adj, other_minx, other_miny, other_maxx, other_maxy, kind)
        best_h = None
        for other in self.parts:
            if int(getattr(other, "id", -1)) == pid:
                continue
            if getattr(other, "geometry", None) is None:
                continue
            try:
                ominx, ominy, omaxx, omaxy = other.geometry.bounds
            except Exception:
                continue
            ocx = (ominx + omaxx) * 0.5
            ocy = (ominy + omaxy) * 0.5
            # Vertical-guide candidates: pairs (dragged-x, other-x).
            v_pairs = [
                (minx, ominx, "L-L"), (minx, omaxx, "L-R"), (minx, ocx, "L-C"),
                (maxx, ominx, "R-L"), (maxx, omaxx, "R-R"), (maxx, ocx, "R-C"),
                (cx,   ominx, "C-L"), (cx,   omaxx, "C-R"), (cx,   ocx,   "C-C"),
            ]
            for d_val, o_val, _kind in v_pairs:
                delta = o_val - d_val
                if abs(delta) <= tol_scene:
                    if best_v is None or abs(delta) < abs(best_v["delta"]):
                        best_v = {
                            "delta": delta,
                            "value": o_val,
                            "other": (ominx, ominy, omaxx, omaxy),
                            "dragged_post": (minx + delta, miny, maxx + delta, maxy),
                        }
            # Horizontal-guide candidates.
            h_pairs = [
                (miny, ominy, "T-T"), (miny, omaxy, "T-B"), (miny, ocy, "T-C"),
                (maxy, ominy, "B-T"), (maxy, omaxy, "B-B"), (maxy, ocy, "B-C"),
                (cy,   ominy, "C-T"), (cy,   omaxy, "C-B"), (cy,   ocy, "C-C"),
            ]
            for d_val, o_val, _kind in h_pairs:
                delta = o_val - d_val
                if abs(delta) <= tol_scene:
                    if best_h is None or abs(delta) < abs(best_h["delta"]):
                        best_h = {
                            "delta": delta,
                            "value": o_val,
                            "other": (ominx, ominy, omaxx, omaxy),
                            "dragged_post": (minx, miny + delta, maxx, maxy + delta),
                        }
        guides = []
        dx_adj = dx
        dy_adj = dy
        if best_v is not None:
            dx_adj += best_v["delta"]
            ominx, ominy, omaxx, omaxy = best_v["other"]
            dminx, dminy, dmaxx, dmaxy = best_v["dragged_post"]
            guides.append({
                "orient": "v",
                "value": best_v["value"],
                "y_min": min(ominy, dminy),
                "y_max": max(omaxy, dmaxy),
            })
        if best_h is not None:
            dy_adj += best_h["delta"]
            ominx, ominy, omaxx, omaxy = best_h["other"]
            dminx, dminy, dmaxx, dmaxy = best_h["dragged_post"]
            guides.append({
                "orient": "h",
                "value": best_h["value"],
                "x_min": min(ominx, dminx),
                "x_max": max(omaxx, dmaxx),
            })
        return dx_adj, dy_adj, guides

    def _move_part_by_delta(self, part_id, dx, dy, push_undo=False):
        """Translate a part by (dx, dy) without going through the selected-
        part API. Used by the click-drag move handler so the drag stream can
        update the geometry incrementally without spamming undo states.
        Pass push_undo=True at the start of a drag to record a single undo
        snapshot for the whole operation."""
        try:
            pid = int(part_id)
        except Exception:
            return False
        part = next((p for p in self.parts if int(getattr(p, "id", -1)) == pid), None)
        if not part or part.geometry is None:
            return False
        if push_undo:
            self.push_undo_state()
        part.geometry = shp_translate(part.geometry, xoff=dx, yoff=dy)
        sketches = []
        metas = []
        for idx, sketch in enumerate(getattr(part, "sketches", []) or []):
            meta = {}
            if idx < len(getattr(part, "sketch_meta", [])):
                meta = part.sketch_meta[idx]
            sketches.append(self._transform_points(sketch, dx, dy))
            metas.append(self._transform_meta(meta, sketch, dx, dy))
        part.sketches = sketches
        part.sketch_meta = metas
        self._sync_cad_shape(part)
        self.rebuild_display_geometry()
        self.redraw()
        return True

    def move_selected_part(self, dx, dy):
        part = self.get_selected_part()
        if not part or part.geometry is None:
            return False
        self.push_undo_state()
        part.geometry = shp_translate(part.geometry, xoff=dx, yoff=dy)
        sketches = []
        metas = []
        for idx, sketch in enumerate(getattr(part, "sketches", []) or []):
            meta = {}
            if idx < len(getattr(part, "sketch_meta", [])):
                meta = part.sketch_meta[idx]
            sketches.append(self._transform_points(sketch, dx, dy))
            metas.append(self._transform_meta(meta, sketch, dx, dy))
        part.sketches = sketches
        part.sketch_meta = metas
        self._sync_cad_shape(part)
        self.rebuild_display_geometry()
        self.partsChanged.emit()
        self.geometryChanged.emit()
        self.redraw()
        return True

    def copy_selected_part(self, dx, dy, count=1, name_suffix="Copy"):
        part = self.get_selected_part()
        if not part or part.geometry is None:
            return False
        self.push_undo_state()
        for i in range(count):
            offset_x = dx * (i + 1)
            offset_y = dy * (i + 1)
            geom = shp_translate(part.geometry, xoff=offset_x, yoff=offset_y)
            sketches = []
            metas = []
            for idx, sketch in enumerate(getattr(part, "sketches", []) or []):
                meta = {}
                if idx < len(getattr(part, "sketch_meta", [])):
                    meta = part.sketch_meta[idx]
                sketches.append(self._transform_points(sketch, offset_x, offset_y))
                metas.append(self._transform_meta(meta, sketch, offset_x, offset_y))
            name = f"{part.name} {name_suffix} {i + 1}"
            new_part = self._clone_part_with_geometry(part, geom, sketches, metas, name)
            self.parts.append(new_part)
        self.rebuild_display_geometry()
        self.partsChanged.emit()
        self.geometryChanged.emit()
        self.redraw()
        return True

    def mirror_selected_part(self, axis="vertical", offset=0.0, keep_original=True):
        part = self.get_selected_part()
        if not part or part.geometry is None:
            return False
        self.push_undo_state()
        geom = part.geometry
        if axis == "vertical":
            geom = shp_translate(geom, xoff=-offset)
            geom = shp_scale(geom, xfact=-1, yfact=1, origin=(0, 0))
            geom = shp_translate(geom, xoff=offset)
        else:
            geom = shp_translate(geom, yoff=-offset)
            geom = shp_scale(geom, xfact=1, yfact=-1, origin=(0, 0))
            geom = shp_translate(geom, yoff=offset)
        sketches = []
        metas = []
        for idx, sketch in enumerate(getattr(part, "sketches", []) or []):
            meta = {}
            if idx < len(getattr(part, "sketch_meta", [])):
                meta = part.sketch_meta[idx]
            sketches.append(self._transform_points(sketch, mirror_axis=axis, mirror_offset=offset))
            metas.append(self._transform_meta(meta, sketch, mirror_axis=axis, mirror_offset=offset))
        if keep_original:
            name = f"{part.name} Mirror"
            new_part = self._clone_part_with_geometry(part, geom, sketches, metas, name)
            self.parts.append(new_part)
        else:
            part.geometry = geom
            part.sketches = sketches
            part.sketch_meta = metas
            self._sync_cad_shape(part)
        self.rebuild_display_geometry()
        self.partsChanged.emit()
        self.geometryChanged.emit()
        self.redraw()
        return True

    def mirror_selected_part_line(self, p1, p2, keep_original=True):
        part = self.get_selected_part()
        if not part or part.geometry is None:
            return False
        if dist(p1, p2) <= 1e-9:
            return False

        def _reflect_coords(x, y, z=None):
            rx, ry = self._transform_point((x, y), mirror_line=(p1, p2))
            return (rx, ry) if z is None else (rx, ry, z)

        self.push_undo_state()
        geom = shp_transform(_reflect_coords, part.geometry)
        sketches = []
        metas = []
        for idx, sketch in enumerate(getattr(part, "sketches", []) or []):
            meta = {}
            if idx < len(getattr(part, "sketch_meta", [])):
                meta = part.sketch_meta[idx]
            sketches.append(self._transform_points(sketch, mirror_line=(p1, p2)))
            metas.append(self._transform_meta(meta, sketch, mirror_line=(p1, p2)))
        if keep_original:
            name = f"{part.name} Mirror"
            new_part = self._clone_part_with_geometry(part, geom, sketches, metas, name)
            self.parts.append(new_part)
        else:
            part.geometry = geom
            part.sketches = sketches
            part.sketch_meta = metas
            self._sync_cad_shape(part)
        self.rebuild_display_geometry()
        self.partsChanged.emit()
        self.geometryChanged.emit()
        self.redraw()
        return True

    def delete_part(self, part, confirm=True):
        if not part:
            return False
        if confirm:
            label = "hole" if part.is_void else "part"
            reply = QMessageBox.question(
                self,
                "Delete",
                f"Delete {label} '{part.name}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return False

        self.push_undo_state()

        ids_to_remove = {part.id}
        ids_to_remove.update(p.id for p in self.parts if p.parent_id == part.id)
        self.parts = [p for p in self.parts if p.id not in ids_to_remove]
        self.interfaces = [
            iface
            for iface in self.interfaces
            if iface.part1_id not in ids_to_remove and iface.part2_id not in ids_to_remove
        ]

        if self.selected_part_id in ids_to_remove:
            self.selected_part_id = None
        if getattr(self, "_editing_part_shape_id", None) in ids_to_remove:
            # The deleted part's sketches were loaded into the active sketch
            # buffer for shape editing — drop them so a new sketch starts clean
            # instead of appending onto orphaned geometry.
            self.sketches.clear()
            self.sketch_meta.clear()
            self.dimensions.clear()
            self.constraints.clear()
            self.current.clear()
            self._clear_preview()
            self._clear_part_shape_edit_session()

        self.rebuild_display_geometry()
        self.partsChanged.emit()
        self._emit_interfaces_changed()
        self.geometryChanged.emit()
        self.bcsChanged.emit()
        self.loadsChanged.emit()
        self.redraw()
        if self.window() and hasattr(self.window(), "show_toast"):
            try:
                deleted_name = getattr(part, "name", None) or f"Part {getattr(part, 'id', '?')}"
                self.window().show_toast(f"Deleted '{deleted_name}'.", kind="warn")
            except Exception:
                pass
        return True

    def _handle_hover(self, pt):
        if not self.solid_geometry: 
            if self.hover_item: self.hover_item = None; self.redraw()
            return
        verts, edges = get_solid_features(self.solid_geometry)
        found = None
        if self.snap_endpoints:
            for v in verts:
                if dist(pt, v) < SNAP_TOL:
                    found = ('vertex', v)
                    break
        if not found:
            for edge in edges:
                if point_line_dist(pt, edge[0], edge[1]) < SNAP_TOL: found = ('edge', edge); break
        if found != self.hover_item:
            self.hover_item = found; self.redraw()

    def clear_all(self):
        self.stop_visualization()
        self._clear_part_shape_edit_session()
        self.sketches.clear()
        self.sketch_meta.clear()
        self.current.clear()
        self.solid_geometry = None
        self.parts.clear()
        self.interfaces.clear()
        self.bcs.clear()
        self.loads.clear()
        self.preview_items.clear()
        self.part_meshes.clear()
        self.operations.clear()
        self.global_nodes = np.array([])
        self.global_elements = np.array([])
        self.global_nodes_3d = np.array([])
        self.global_elements_3d = np.array([])
        self.element_part_map_3d = []
        self._pore_preview_polys = []
        self.element_part_map = []
        self._interface_preview_cache = None
        self._interface_preview_cache_sig = None
        self.initial_velocities = []
        self.selected_part_id = None
        self._pending_attr_edit = None
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.command_last_point = (0.0, 0.0)
        self.display_mode = "geometry"
        self._displacement_vectors = []
        self._result_scalar_values = None
        self._result_field_label = ""
        self._clear_results_legend()
        self.dimensions.clear()
        self.constraints.clear()
        self._clear_dimension_item_cache()
        self._dimension_id_counter = 0
        self._pending_dimension = None
        self._pending_constraint = None
        Part._part_counter = 0
        Interface._interface_counter = 0
        Interface._interaction_counter = 0
        Operation._op_counter = 0
        self.mesh_min_spacing_factor = MESH_MIN_SPACING_FACTOR
        self.mesh_boundary_thickness = 0.0
        self.mesh_boundary_spacing_factor = 1.0
        self.show_dimensions = True
        self.setSceneRect(self._default_scene_rect())
        self.rebuild_display_geometry()
        self.partsChanged.emit()
        self._emit_interfaces_changed()
        self.geometryChanged.emit()
        self.redraw()

    def _ensure_min_animation_frames(self, frames, min_frames=21):
        count = len(frames)
        if count >= min_frames or count == 0:
            return frames
        if count == 1:
            return [np.array(frames[0], copy=True) for _ in range(min_frames)]

        frames_arr = [np.asarray(frame, dtype=float) for frame in frames]
        idxs = np.linspace(0, count - 1, min_frames)
        new_frames = []
        for t in idxs:
            i0 = int(np.floor(t))
            i1 = int(np.ceil(t))
            if i0 == i1:
                new_frames.append(np.array(frames_arr[i0], copy=True))
                continue
            alpha = t - i0
            new_frames.append(frames_arr[i0] * (1.0 - alpha) + frames_arr[i1] * alpha)
        return new_frames

    def load_and_run_visualization(self, results_root=None):
        """
        Loads simulation results from CSV files in workspace/output/results
        and prepares them for replay in the SketchView.
        """
        self.animation_timer.stop()
        self._emit_animation_playback_state()
        self._release_animation_result_handles()
        self.is_visualization_mode = False
        self.animation_frames.clear()
        self._lazy_results_enabled = False
        self._animation_frame_count = 0
        self._animation_frame_loading = False
        self._pending_lazy_frame_index = None
        self._current_animation_positions = None
        self._current_animation_velocity = None
        self._current_frame_packet = None
        self._results_preview_auto_fit_pending = False
        self.current_frame_index = -1
        self._displacement_vectors = []
        self._result_scalar_values = None
        self._result_field_label = ""
        self._clear_results_legend()
        self._clear_results_debug_overlays()
        self._replay_selected_particle_index = None
        self._replay_selected_nodes = set()
        self._replay_selected_triangles = set()
        self._replay_selected_mesh_edges = set()
        self._replay_selected_geometry_edges = []
        self._replay_selected_bc_targets = []
        try:
            self.replayParticleSelected.emit({})
        except Exception:
            pass
        window = self.window()
        if window is not None and hasattr(window, "_hide_results_point_preview"):
            try:
                window._hide_results_point_preview(clear_points=True, force_view=True)
            except Exception:
                pass

        try:
            controller = self._get_results_controller()
            if controller is None:
                raise RuntimeError("Results controller is not available for lazy loading.")
            frame_count = int(
                controller.open_results(
                    results_root=results_root,
                    unit_scale=self._unit_scale_to_meters(),
                )
            )
            if frame_count <= 0:
                QMessageBox.warning(
                    self,
                    "Visualization Error",
                    "Could not load any animation frames from simulation outputs.",
                )
                return
            self.set_module("Results")
            self.is_visualization_mode = True
            self.display_mode = "results"
            # Drop Python references to scene items BEFORE clearing the scene.
            # scene().clear() deletes the underlying C++ QGraphicsItems; any
            # subsequent reassignment of these lists would trigger Python's
            # refcount-driven destructor on already-deleted C++ objects,
            # producing a native access violation. Reproduced reliably when
            # loading the result frames of a refined mesh.
            for attr in (
                "_anim_node_items",
                "_anim_element_items",
                "preview_items",
                "_results_debug_overlay_items",
            ):
                if hasattr(self, attr):
                    try:
                        setattr(self, attr, [])
                    except Exception:
                        pass
            if hasattr(self, "_dimension_items"):
                try:
                    self._dimension_items = {}
                except Exception:
                    pass
            self.scene().clear()
            self._draw_grid()
            self._lazy_results_enabled = True
            self._animation_frame_count = frame_count
            self._results_preview_auto_fit_pending = True
            self._request_lazy_animation_frame(0)
            self.animationFramesLoaded.emit(frame_count)
            self._emit_animation_playback_state()
            window = self.window()
            if window and hasattr(window, "properties_panel"):
                panel = window.properties_panel
                if hasattr(panel, "results_tab"):
                    panel.tabs.setCurrentWidget(panel.results_tab)
            QMessageBox.information(
                self,
                "Results Loaded",
                f"Opened {frame_count} frames with lazy loading. Replay is ready to play.",
            )

        except Exception as e:
            QMessageBox.critical(self, "Visualization Error", f"An error occurred while loading results: {e}")
            self.animation_frames.clear()
            self.is_visualization_mode = False

    # --- Render ---
    def redraw(self):
        if self.is_visualization_mode:
            return
        window = self.window()
        if window is not None and hasattr(window, "_ui_ready") and not window._ui_ready:
            return
        if getattr(self, "_redraw_pending", False) or self._redraw_in_progress:
            return
        self._redraw_pending = True
        QTimer.singleShot(0, self._do_redraw)

    def _world_units_per_pixel(self):
        try:
            p0 = self.mapToScene(0, 0)
            p1 = self.mapToScene(1, 0)
            units = math.hypot(p1.x() - p0.x(), p1.y() - p0.y())
            if np.isfinite(units) and units > 0:
                return float(units)
        except Exception:
            pass
        return 1.0

    def _decimate_path_points(self, coords, max_points):
        if not coords:
            return []
        if max_points <= 0 or len(coords) <= max_points:
            return coords
        if max_points <= 2:
            return [coords[0], coords[-1]]
        step = max(1, int(math.ceil((len(coords) - 1) / float(max_points - 1))))
        pts = list(coords[::step])
        if pts[-1] != coords[-1]:
            pts.append(coords[-1])
        return pts

    def _estimated_geom_area(self, geom):
        try:
            area = float(getattr(geom, "area", 0.0) or 0.0)
            if np.isfinite(area) and area > 0:
                return area
        except Exception:
            pass
        try:
            minx, miny, maxx, maxy = geom.bounds
            return max(0.0, maxx - minx) * max(0.0, maxy - miny)
        except Exception:
            return 0.0

    def _select_draw_child_geometries(self, children, units_per_pixel):
        candidates = []
        for child in children:
            geom = getattr(child, "geometry", None)
            if geom is None or geom.is_empty:
                continue
            candidates.append((self._estimated_geom_area(geom), geom))
        if not candidates:
            return []
        if not getattr(self, "geometry_fast_draw_enabled", False):
            return [geom for _, geom in candidates]

        min_hole_world = max(
            0.0,
            float(getattr(self, "geometry_fast_draw_min_hole_pixels", 0.0))
            * float(units_per_pixel),
        )
        min_hole_area = min_hole_world * min_hole_world
        if min_hole_area > 0.0:
            candidates = [item for item in candidates if item[0] >= min_hole_area]
            if not candidates:
                return []

        max_holes = int(getattr(self, "geometry_fast_draw_max_holes", 0) or 0)
        if max_holes > 0 and len(candidates) > max_holes:
            candidates.sort(key=lambda it: it[0], reverse=True)
            candidates = candidates[:max_holes]
        return [geom for _, geom in candidates]

    def _prepare_draw_geometry(self, geom, units_per_pixel):
        if geom is None or geom.is_empty:
            return geom
        if not getattr(self, "geometry_fast_draw_enabled", False):
            return geom
        simplify_px = float(getattr(self, "geometry_fast_draw_simplify_pixels", 0.0) or 0.0)
        tol = max(0.0, simplify_px * float(units_per_pixel))
        if tol <= 0.0:
            return geom
        try:
            simple = geom.simplify(tol, preserve_topology=True)
            if simple is not None and not simple.is_empty:
                return simple
        except Exception:
            pass
        return geom

    def _draw_part_geometry_layers(self, *, for_mesh=False):
        default_colors = [
            QColor(214, 226, 240), QColor(240, 214, 214), QColor(214, 240, 214),
            QColor(240, 232, 214), QColor(232, 214, 240), QColor(214, 240, 240)
        ]
        interaction_focus_ids = set()
        if str(getattr(self, "_panel_attr_focus_kind", "") or "").lower() == "interaction":
            interaction_focus_ids = set(self._panel_interaction_part_ids(getattr(self, "_panel_attr_focus_entry_ref", None)))

        if not self.material_color_map:
            next_color_idx = 0
        else:
            next_color_idx = len(self.material_color_map)
        for serial in sorted(self.materials.keys()):
            if serial not in self.material_color_map:
                self.material_color_map[serial] = default_colors[next_color_idx % len(default_colors)]
                next_color_idx += 1

        sorted_parts = sorted(self.parts, key=lambda p: p.id)
        units_per_pixel = self._world_units_per_pixel()

        for part in sorted_parts:
            if part.is_void:
                continue

            geom_to_draw = part.geometry
            if geom_to_draw is None or geom_to_draw.is_empty:
                continue

            children = self.get_child_parts(part)
            child_geoms = self._select_draw_child_geometries(children, units_per_pixel)
            if child_geoms:
                try:
                    holes_geom = child_geoms[0] if len(child_geoms) == 1 else unary_union(child_geoms)
                    if holes_geom is not None and not holes_geom.is_empty:
                        geom_to_draw = geom_to_draw.difference(holes_geom)
                except Exception:
                    for child_geom in child_geoms:
                        try:
                            geom_to_draw = geom_to_draw.difference(child_geom)
                        except Exception:
                            pass

            try:
                if hasattr(geom_to_draw, "is_valid") and not geom_to_draw.is_valid:
                    geom_to_draw = geom_to_draw.buffer(0)
            except Exception:
                pass

            if not geom_to_draw or geom_to_draw.is_empty:
                continue

            if part.material_id is not None:
                fill_color = self.material_color_map.get(
                    part.material_id, QColor(200, 200, 200)
                )
                pen = QPen(Qt.black, 1.5)
                pen.setCosmetic(True)
            else:
                fill_color = QColor(230, 230, 230)
                pen = QPen(Qt.red, 1.5, Qt.DashLine)
                pen.setCosmetic(True)

            if for_mesh:
                mesh_fill = QColor(fill_color)
                mesh_fill.setAlpha(36)
                mesh_pen_color = QColor(pen.color())
                mesh_pen_color.setAlpha(170)
                pen = QPen(mesh_pen_color, max(1.0, pen.widthF() * 0.8), pen.style())
                fill_brush = QBrush(mesh_fill)
            else:
                fill_brush = QBrush(fill_color)

            # Focus mode — when ANY part is selected, dim all non-selected
            # parts so the user's attention stays on the active selection.
            selected_ids = set(getattr(self, "multi_selected_part_ids", None) or set())
            if self.selected_part_id is not None:
                selected_ids.add(int(self.selected_part_id))
            is_selected_now = int(getattr(part, "id", -1)) in selected_ids
            if selected_ids and not is_selected_now and not for_mesh:
                dim_fill = QColor(fill_color)
                dim_fill.setAlpha(80)
                dim_pen = QColor(pen.color())
                dim_pen.setAlpha(110)
                fill_brush = QBrush(dim_fill)
                pen = QPen(dim_pen, pen.widthF(), pen.style())
                pen.setCosmetic(True)

            geom_for_path = self._prepare_draw_geometry(geom_to_draw, units_per_pixel)
            path = self._geometry_to_path(geom_for_path, units_per_pixel, include_holes=True)
            if path.isEmpty():
                continue

            self.scene().addPath(path, pen, fill_brush)
            if part.id == self.selected_part_id:
                self.scene().addPath(
                    path,
                    QPen(QColor(247, 206, 70), 3.2 if not for_mesh else 2.4),
                    QBrush(Qt.transparent),
                )
            # Highlight multi-selected (Ctrl+Clicked) parts in addition to
            # the primary selection so the user can see what Ctrl+P will
            # combine.
            elif int(getattr(part, "id", -1)) in selected_ids and not for_mesh:
                self.scene().addPath(
                    path,
                    QPen(QColor(37, 99, 235, 220), 2.4, Qt.DashLine),
                    QBrush(Qt.transparent),
                )
            if int(getattr(part, "id", -1)) in interaction_focus_ids:
                self.scene().addPath(
                    path,
                    QPen(QColor(70, 150, 255), 2.6 if not for_mesh else 2.0),
                    QBrush(QColor(70, 150, 255, 20) if not for_mesh else Qt.transparent),
                )

        if getattr(self, "_pore_preview_polys", []):
            pen = QPen(QColor(0, 120, 255), 1.5, Qt.DashLine)
            preview_polys = list(self._pore_preview_polys)
            max_preview = int(getattr(self, "geometry_fast_draw_max_preview_polys", 0) or 0)
            if (
                getattr(self, "geometry_fast_draw_enabled", False)
                and max_preview > 0
                and len(preview_polys) > max_preview
            ):
                step = max(1, int(math.ceil(len(preview_polys) / float(max_preview))))
                preview_polys = preview_polys[::step]
            for poly in preview_polys:
                if poly is None or poly.is_empty:
                    continue
                poly_for_path = self._prepare_draw_geometry(poly, units_per_pixel)
                path = self._geometry_to_path(poly_for_path, units_per_pixel, include_holes=False)
                if not path.isEmpty():
                    self.scene().addPath(path, pen)

    def _geometry_to_path(self, geom, units_per_pixel, include_holes=True):
        path = QPainterPath()
        if geom is None or geom.is_empty:
            return path

        geoms = [geom] if isinstance(geom, Polygon) else list(getattr(geom, "geoms", []))
        max_points = int(getattr(self, "geometry_fast_draw_max_path_points", 0) or 0)
        min_hole_world = 0.0
        if include_holes and getattr(self, "geometry_fast_draw_enabled", False):
            min_hole_world = max(
                0.0,
                float(getattr(self, "geometry_fast_draw_min_hole_pixels", 0.0))
                * float(units_per_pixel),
            )

        for poly in geoms:
            ext = self._decimate_path_points(list(poly.exterior.coords), max_points)
            if not ext:
                continue
            path.moveTo(*ext[0])
            for p in ext[1:]:
                path.lineTo(*p)

            if not include_holes:
                continue
            for interior in poly.interiors:
                if min_hole_world > 0.0:
                    try:
                        minx, miny, maxx, maxy = interior.bounds
                        if (maxx - minx) < min_hole_world and (maxy - miny) < min_hole_world:
                            continue
                    except Exception:
                        pass
                hole_coords = self._decimate_path_points(list(interior.coords), max_points)
                if not hole_coords:
                    continue
                path.moveTo(*hole_coords[0])
                for p in hole_coords[1:]:
                    path.lineTo(*p)
        return path

    def _part_boundary_paths_from_mesh(self, node_positions=None, elements=None, element_part_map=None):
        if node_positions is None:
            node_positions = self.global_nodes
        if elements is None:
            elements = self.global_elements
        if element_part_map is None:
            element_part_map = self.element_part_map
        try:
            pos_arr = np.asarray(node_positions, dtype=float)
            elem_arr = np.asarray(elements, dtype=int)
        except Exception:
            return {}
        if pos_arr.ndim != 2 or pos_arr.shape[1] < 2:
            return {}
        if elem_arr.ndim != 2 or elem_arr.shape[1] < 3 or len(elem_arr) == 0:
            return {}

        tri_to_part = {}
        for item in element_part_map or []:
            if not isinstance(item, dict):
                continue
            try:
                tri_to_part[int(item.get("element_idx"))] = int(item.get("part_id"))
            except Exception:
                continue
        if not tri_to_part:
            return {}

        edge_counts = {}
        node_count = len(pos_arr)
        for elem_idx, tri in enumerate(elem_arr):
            part_id = tri_to_part.get(int(elem_idx))
            if part_id is None:
                continue
            try:
                a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            except Exception:
                continue
            if min(a, b, c) < 0 or max(a, b, c) >= node_count:
                continue
            for i1, i2 in ((a, b), (b, c), (c, a)):
                if i1 == i2:
                    continue
                e0, e1 = (i1, i2) if i1 < i2 else (i2, i1)
                key = (int(part_id), int(e0), int(e1))
                edge_counts[key] = edge_counts.get(key, 0) + 1

        paths = {}
        for (part_id, i1, i2), count in edge_counts.items():
            if count != 1:
                continue
            try:
                p1 = pos_arr[int(i1)]
                p2 = pos_arr[int(i2)]
            except Exception:
                continue
            path = paths.get(int(part_id))
            if path is None:
                path = QPainterPath()
                paths[int(part_id)] = path
            path.moveTo(float(p1[0]), float(p1[1]))
            path.lineTo(float(p2[0]), float(p2[1]))
        return paths

    def _draw_part_boundary_overlay_from_mesh(self, node_positions=None, *, line_width=1.6, item_sink=None):
        paths = self._part_boundary_paths_from_mesh(node_positions=node_positions)
        if not paths:
            return [] if item_sink is None else item_sink

        overlay_items = [] if item_sink is None else item_sink
        base_width = max(0.6, float(line_width) * 0.7)
        halo_width = max(base_width + 0.4, base_width * 1.35)

        halo_pen = QPen(QColor(255, 255, 255, 80), halo_width)
        halo_pen.setCapStyle(Qt.RoundCap)
        halo_pen.setJoinStyle(Qt.RoundJoin)
        halo_pen.setCosmetic(True)   # zoom-invariant — boundary never thickens on zoom-in

        normal_pen = QPen(QColor(18, 22, 28, 110), base_width)
        normal_pen.setCapStyle(Qt.RoundCap)
        normal_pen.setJoinStyle(Qt.RoundJoin)
        normal_pen.setCosmetic(True)   # zoom-invariant

        selected_pen = QPen(QColor(247, 206, 70, 170), max(base_width + 0.4, base_width * 1.2))
        selected_pen.setCapStyle(Qt.RoundCap)
        selected_pen.setJoinStyle(Qt.RoundJoin)
        selected_pen.setCosmetic(True)   # zoom-invariant

        for part_id, path in paths.items():
            if path.isEmpty():
                continue
            try:
                halo_item = self.scene().addPath(path, halo_pen, QBrush(Qt.transparent))
                overlay_items.append(halo_item)
                edge_pen = selected_pen if int(part_id) == int(getattr(self, "selected_part_id", -1) or -1) else normal_pen
                edge_item = self.scene().addPath(path, edge_pen, QBrush(Qt.transparent))
                overlay_items.append(edge_item)
            except Exception:
                continue
        return overlay_items

    def _do_redraw(self):
        if self.is_visualization_mode:
            self._redraw_pending = False
            return
        window = self.window()
        if window is not None and hasattr(window, "_ui_ready") and not window._ui_ready:
            self._redraw_pending = False
            return
        if self.scene() is None:
            self._redraw_pending = False
            return
        if self._redraw_in_progress:
            self._redraw_pending = False
            return
        self._redraw_pending = False
        self._redraw_in_progress = True
        try:
            self.ensure_y_axis_up()
            self._sync_scene_rect_to_model()
            try:
                if self._has_active_dimension_editor():
                    self._remove_non_dimension_scene_items()
                else:
                    self._detach_dimension_items_from_scene()
                    self.scene().clear()
            except Exception:
                pass
            self._draw_grid()

            if self.display_mode == "mesh":
                self._draw_part_geometry_layers(for_mesh=True)
                self._draw_mesh_view()
            elif self.display_mode == "mesh_3d":
                self._draw_mesh_view_3d()
            elif self.display_mode == "results":
                self._draw_displacement_view()
            else:
                # --- Draw Composite Parts ---
                self._draw_part_geometry_layers(for_mesh=False)

                # Draw Sketches
                if self.active_module == "Part": # Only show raw sketches in Part module
                    pen = QPen(Qt.red, 2, Qt.DashLine)
                    sketches = [
                        (idx, s)
                        for idx, s in enumerate(self.sketches)
                        if self._is_sketch_visible("sketch", None, idx)
                    ]
                    max_sketches = int(getattr(self, "geometry_fast_draw_max_sketches", 0) or 0)
                    if (
                        getattr(self, "geometry_fast_draw_enabled", False)
                        and max_sketches > 0
                        and len(sketches) > max_sketches
                    ):
                        step = max(1, int(math.ceil(len(sketches) / float(max_sketches))))
                        sketches = sketches[::step]
                    for _, s in sketches:
                        points = self._decimate_path_points(
                            list(s),
                            int(getattr(self, "geometry_fast_draw_max_path_points", 0) or 0),
                        )
                        if len(points) > 1:
                            path = QPainterPath()
                            path.moveTo(*points[0])
                            for p in points[1:]:
                                path.lineTo(*p)
                            self.scene().addPath(path, pen)

                # Draw Hover Highlight
                if self.hover_item and self.active_module in ["Boundary", "Load", "Interface"]:
                    itype, coords = self.hover_item
                    hpen = QPen(QColor(0,150,255), 3)
                    if itype == 'vertex':
                        self._safe_add_ellipse(
                            coords[0]-4,
                            coords[1]-4,
                            8,
                            8,
                            hpen,
                            QBrush(Qt.white),
                            context="hover_vertex",
                        )
                    else:
                        self.scene().addLine(coords[0][0], coords[0][1], coords[1][0], coords[1][1], hpen)

                # Draw Attributes (Module sensitive visibility)
                if self.active_module in ["Boundary", "Load", "Job", "Mesh"]:
                    self._sanitize_bc_load_entries()
                    for bc in self.bcs: self._draw_bc_marker(bc)
                    for ld in self.loads: self._draw_load_marker(ld)

                self._draw_dimensions()
                self._draw_arc_segment_selection()
        finally:
            self._redraw_in_progress = False

    def _draw_arc_segment_selection(self):
        if not self._arc_segment_polyline or len(self._arc_segment_polyline) < 2:
            return
        path = QPainterPath()
        path.moveTo(*self._arc_segment_polyline[0])
        for p in self._arc_segment_polyline[1:]:
            path.lineTo(*p)
        pen = QPen(QColor(0, 120, 255), 2.0, Qt.DashLine)
        try:
            self.scene().addPath(path, pen)
        except Exception:
            pass

    def _draw_grid(self):
        # Grid is now painted by drawBackground based on the current viewport.
        # This hook is retained so existing callers after scene.clear() can
        # still request a repaint without special-casing.
        vp = self.viewport()
        if vp is not None:
            vp.update()

    def _draw_mesh_view(self):
        if self.global_nodes is None or len(self.global_nodes) == 0:
            self._add_info_text("No connections to display.")
            return
        if self.should_use_gpu_point_preview():
            return

        nodes = self.global_nodes
        elements = self.global_elements
        node_count = len(nodes)
        elem_count = len(elements) if elements is not None else 0
        max_nodes = 5000
        max_elements = int(PREVIEW_CONNECTION_LIMIT)
        fast_enabled = bool(getattr(self, "fast_preview_enabled", False))
        fast_limit = int(getattr(self, "fast_preview_connection_limit", FAST_PREVIEW_CONNECTION_LIMIT))
        if fast_enabled:
            if fast_limit <= 0:
                max_elements = 0
            else:
                max_elements = min(max_elements, fast_limit)
        show_nodes = bool(self.show_mesh_nodes)
        show_elements = bool(self.show_mesh_elements) and elements is not None and elem_count > 0

        warnings = []
        if show_elements and elem_count > max_elements:
            show_elements = False
            if fast_enabled:
                warnings.append(
                    f"Fast preview: connections hidden ({elem_count:,} exceeds limit {max_elements:,})."
                )
            else:
                warnings.append(
                    f"Connections hidden: {elem_count:,} exceeds preview limit ({max_elements:,})."
                )
        node_step = 1
        if show_nodes and node_count > max_nodes:
            node_step = max(1, int(math.ceil(node_count / max_nodes)))
            preview_count = (node_count + node_step - 1) // node_step
            warnings.append(
                f"Particles downsampled: showing ~{preview_count:,} of {node_count:,}."
            )
        use_raster = self.should_use_raster_preview(node_count, show_elements, show_nodes)
        if use_raster:
            warnings.append("Raster preview: rendering particle density for speed.")
        if warnings:
            self._add_info_text("Preview simplified:\n" + "\n".join(warnings))

        if (elements is None or elem_count == 0) and not show_nodes and not warnings:
            self._add_info_text("No connections to display.")
            return
        if not show_elements and not show_nodes:
            return
        if use_raster:
            self._draw_raster_particles_preview(nodes)
            return

        def _vivid(color):
            if not isinstance(color, QColor):
                color = QColor(60, 60, 60)
            h, s, v, a = color.getHsv()
            if h < 0:
                h = 0
            s = max(s, 160)
            v = max(v, 200)
            return QColor.fromHsv(h, s, v, 255)

        try:
            mesh_line_w = float(getattr(self, "mesh_preview_line_width", 1.0))
        except Exception:
            mesh_line_w = 1.0
        mesh_line_w = max(0.1, min(20.0, mesh_line_w))
        try:
            mesh_node_size = float(getattr(self, "mesh_preview_particle_size", 3.0))
        except Exception:
            mesh_node_size = 3.0
        mesh_node_size = max(0.5, min(50.0, mesh_node_size))
        mesh_node_r = 0.5 * mesh_node_size

        element_part = {}
        part_map = {p.id: p for p in self.parts}
        if self.element_part_map and show_elements:
            for m in self.element_part_map:
                try:
                    element_part[int(m.get("element_idx"))] = int(m.get("part_id"))
                except Exception:
                    continue

        interface_element_indices = set()
        interface_node_ids = set()
        base_iface_color = getattr(self, "interface_preview_color", QColor(70, 150, 255))
        if not isinstance(base_iface_color, QColor):
            base_iface_color = QColor(70, 150, 255)
        node_iface_color = QColor(base_iface_color)
        node_iface_color.setAlpha(255)
        if (elements is not None and elem_count > 0) and self.interfaces:
            try:
                for row in self._get_interface_preview_rows():
                    if str(row.get("zone_kind", "")).lower() != "interface":
                        continue
                    try:
                        ei = int(row.get("_element_idx"))
                    except Exception:
                        continue
                    if ei < 0 or ei >= elem_count:
                        continue
                    interface_element_indices.add(ei)
                    tri = elements[ei]
                    try:
                        interface_node_ids.update(int(nid) for nid in tri[:3])
                    except Exception:
                        pass
            except Exception:
                interface_element_indices.clear()
                interface_node_ids.clear()

        if show_elements:
            if element_part:
                elements_by_part = {}
                for idx, tri in enumerate(elements):
                    if idx in interface_element_indices:
                        continue
                    pid = element_part.get(idx)
                    if pid is None:
                        continue
                    elements_by_part.setdefault(pid, []).append(tri)
                for pid, tris in elements_by_part.items():
                    path = QPainterPath()
                    for tri in tris:
                        try:
                            p1 = nodes[tri[0]]
                            p2 = nodes[tri[1]]
                            p3 = nodes[tri[2]]
                        except Exception:
                            continue
                        path.moveTo(p1[0], p1[1])
                        path.lineTo(p2[0], p2[1])
                        path.lineTo(p3[0], p3[1])
                        path.closeSubpath()
                    part = part_map.get(pid)
                    color = QColor(60, 60, 60)
                    if part and part.material_id is not None:
                        color = self.material_color_map.get(part.material_id, color)
                    mesh_color = _vivid(color)
                    _edge_pen = QPen(mesh_color, mesh_line_w)
                    _edge_pen.setCosmetic(True)   # zoom-invariant: never thickens on zoom-in
                    self.scene().addPath(path, _edge_pen, QBrush(Qt.transparent))
                    if pid == self.selected_part_id:
                        _sel_pen = QPen(QColor(247, 206, 70), max(2.0, mesh_line_w + 1.8))
                        _sel_pen.setCosmetic(True)
                        self.scene().addPath(
                            path,
                            _sel_pen,
                            QBrush(QColor(247, 206, 70, 24)),
                        )
            else:
                path = QPainterPath()
                for idx, tri in enumerate(elements):
                    if idx in interface_element_indices:
                        continue
                    try:
                        p1 = nodes[tri[0]]
                        p2 = nodes[tri[1]]
                        p3 = nodes[tri[2]]
                    except Exception:
                        continue
                    path.moveTo(p1[0], p1[1])
                    path.lineTo(p2[0], p2[1])
                    path.lineTo(p3[0], p3[1])
                    path.closeSubpath()

                _fallback_pen = QPen(QColor(60, 60, 60), mesh_line_w)
                _fallback_pen.setCosmetic(True)   # zoom-invariant
                self.scene().addPath(path, _fallback_pen, QBrush(Qt.transparent))

        if show_elements and interface_element_indices:
            edge_iface_color = QColor(base_iface_color)
            edge_iface_color.setAlpha(235)
            fill_iface_color = QColor(base_iface_color)
            fill_iface_color.setAlpha(55)
            iface_fill_path = QPainterPath()
            iface_edge_path = QPainterPath()
            for ei in sorted(interface_element_indices):
                try:
                    tri = elements[ei]
                    p1 = nodes[tri[0]]
                    p2 = nodes[tri[1]]
                    p3 = nodes[tri[2]]
                except Exception:
                    continue
                iface_fill_path.moveTo(p1[0], p1[1])
                iface_fill_path.lineTo(p2[0], p2[1])
                iface_fill_path.lineTo(p3[0], p3[1])
                iface_fill_path.closeSubpath()
                iface_edge_path.moveTo(p1[0], p1[1])
                iface_edge_path.lineTo(p2[0], p2[1])
                iface_edge_path.lineTo(p3[0], p3[1])
                iface_edge_path.closeSubpath()
            self.scene().addPath(
                iface_fill_path,
                QPen(Qt.transparent),
                QBrush(fill_iface_color),
            )
            _iface_pen = QPen(edge_iface_color, mesh_line_w)
            _iface_pen.setCosmetic(True)   # zoom-invariant
            self.scene().addPath(
                iface_edge_path,
                _iface_pen,
                QBrush(Qt.transparent),
            )

        if show_nodes:
            node_part = {}
            node_part_sets = {}
            part_area = {}
            for _pid, _part in part_map.items():
                try:
                    part_area[int(_pid)] = float(getattr(getattr(_part, "geometry", None), "area", 0.0) or 0.0)
                except Exception:
                    part_area[int(_pid)] = 0.0
            if element_part and show_elements:
                for idx, tri in enumerate(elements):
                    pid = element_part.get(idx)
                    if pid is None:
                        continue
                    for nid in tri:
                        node_part_sets.setdefault(int(nid), set()).add(int(pid))
                        if nid not in node_part:
                            node_part[nid] = pid
                        elif node_part[nid] != pid:
                            node_part[nid] = None

            # Per-node local mesh size (mean of incident triangle edge lengths).
            # Used to scale the displayed dot size: nodes in fine-mesh regions
            # (boundaries) get smaller dots, bulk nodes get the user's full
            # configured size. This makes the dot density visually match the
            # mesh density instead of producing a uniform black blob at fine
            # boundaries.
            node_size_scale = None
            if elements is not None and elem_count > 0 and len(nodes) > 0:
                try:
                    elem_arr = np.asarray(elements[:elem_count], dtype=int)
                    e_uv = np.vstack([
                        elem_arr[:, [0, 1]],
                        elem_arr[:, [1, 2]],
                        elem_arr[:, [2, 0]],
                    ])
                    node_arr = np.asarray(nodes, dtype=float)
                    edge_vec = node_arr[e_uv[:, 1]] - node_arr[e_uv[:, 0]]
                    edge_lens = np.linalg.norm(edge_vec, axis=1)
                    sums = np.zeros(len(nodes), dtype=float)
                    cnts = np.zeros(len(nodes), dtype=float)
                    np.add.at(sums, e_uv[:, 0], edge_lens)
                    np.add.at(sums, e_uv[:, 1], edge_lens)
                    np.add.at(cnts, e_uv[:, 0], 1.0)
                    np.add.at(cnts, e_uv[:, 1], 1.0)
                    local_size = np.where(cnts > 0, sums / np.maximum(cnts, 1.0), 0.0)
                    positives = local_size[local_size > 0]
                    if positives.size > 0:
                        bulk_ref = float(np.percentile(positives, 95))
                        if bulk_ref > 0:
                            # Bulk -> 1.0, boundary scales down. Min 0.3 so dots stay visible.
                            node_size_scale = np.clip(local_size / bulk_ref, 0.3, 1.0)
                except Exception:
                    node_size_scale = None

            def _node_dot_size(idx):
                if node_size_scale is not None and 0 <= idx < node_size_scale.size:
                    return mesh_node_size * float(node_size_scale[idx])
                return mesh_node_size

            for i in range(0, len(nodes), node_step):
                if i in interface_node_ids:
                    continue
                x, y = nodes[i]
                # Node dots are always black — they must not inherit the mesh/
                # material colour even when the surrounding triangles are red,
                # orange, yellow, green, or blue.
                color = QColor(0, 0, 0)
                this_size = _node_dot_size(i)
                this_r = 0.5 * this_size
                self._safe_add_ellipse(
                    x - this_r,
                    y - this_r,
                    this_size,
                    this_size,
                    QPen(Qt.transparent),
                    QBrush(color),
                    context="mesh_nodes_2d",
                )
            if interface_node_ids:
                for nid in sorted(interface_node_ids):
                    if nid < 0 or nid >= len(nodes):
                        continue
                    try:
                        x, y = nodes[nid]
                    except Exception:
                        continue
                    iface_node_size = _node_dot_size(int(nid))
                    iface_node_r = 0.5 * iface_node_size
                    iface_core_size = max(0.8, 0.55 * iface_node_size)
                    iface_core_r = 0.5 * iface_core_size
                    self._safe_add_ellipse(
                        x - iface_node_r,
                        y - iface_node_r,
                        iface_node_size,
                        iface_node_size,
                        QPen(Qt.transparent),
                        QBrush(node_iface_color),
                        context="mesh_nodes_2d",
                    )
                    # Show a part-colored core inside interface nodes so shared boundaries do not
                    # look visually disconnected (e.g., blue inclusion tied to green interface).
                    part_ids = sorted(int(pid) for pid in node_part_sets.get(int(nid), set()))
                    if part_ids:
                        chosen_pid = part_ids[0]
                        if len(part_ids) > 1:
                            # Prefer the smaller-area part for shared nodes (typically the inclusion).
                            chosen_pid = min(part_ids, key=lambda pid: part_area.get(int(pid), float("inf")))
                        core_color = QColor(255, 255, 255)
                        part = part_map.get(int(chosen_pid))
                        if part and part.material_id is not None:
                            core_color = _vivid(self.material_color_map.get(part.material_id, core_color))
                        self._safe_add_ellipse(
                            x - iface_core_r,
                            y - iface_core_r,
                            iface_core_size,
                            iface_core_size,
                            QPen(Qt.transparent),
                            QBrush(core_color),
                            context="mesh_nodes_2d",
                        )

        if show_elements:
            self._draw_part_boundary_overlay_from_mesh(
                nodes,
                line_width=max(1.4, mesh_line_w + 0.55),
            )

    def _project_3d_points(self, nodes):
        angle = self._projection_angle
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        proj = np.zeros((len(nodes), 2), dtype=float)
        proj[:, 0] = nodes[:, 0] + nodes[:, 2] * cos_a
        proj[:, 1] = nodes[:, 1] + nodes[:, 2] * sin_a
        return proj

    def _draw_mesh_view_3d(self):
        if self.global_nodes_3d is None or len(self.global_nodes_3d) == 0:
            self._add_info_text("No 3D connections to display.")
            return

        nodes = np.asarray(self.global_nodes_3d)
        proj = self._project_3d_points(nodes)
        elements = self.global_elements_3d

        if self.show_mesh_elements and elements is not None and len(elements) > 0:
            edge_set = set()
            for tet in elements:
                try:
                    a, b, c, d = [int(i) for i in tet]
                except Exception:
                    continue
                pairs = ((a, b), (a, c), (a, d), (b, c), (b, d), (c, d))
                for i1, i2 in pairs:
                    if i1 == i2:
                        continue
                    edge = (i1, i2) if i1 < i2 else (i2, i1)
                    edge_set.add(edge)

            edges = list(edge_set)
            max_edges = 20000
            if len(edges) > max_edges:
                step = max(1, len(edges) // max_edges)
                edges = edges[::step]

            path = QPainterPath()
            for i1, i2 in edges:
                try:
                    p1 = proj[i1]
                    p2 = proj[i2]
                except Exception:
                    continue
                path.moveTo(p1[0], p1[1])
                path.lineTo(p2[0], p2[1])
            self.scene().addPath(path, QPen(QColor(70, 70, 70), 1), QBrush(Qt.transparent))

        if self.show_mesh_nodes:
            max_nodes = 8000
            step = 1 if len(nodes) <= max_nodes else max(1, len(nodes) // max_nodes)
            z_vals = nodes[:, 2]
            z_min = float(np.min(z_vals)) if len(z_vals) else 0.0
            z_max = float(np.max(z_vals)) if len(z_vals) else 1.0
            z_range = max(z_max - z_min, 1e-9)
            for idx in range(0, len(nodes), step):
                x, y = proj[idx]
                z_norm = (nodes[idx, 2] - z_min) / z_range
                shade = int(40 + 160 * (1.0 - z_norm))
                color = QColor(shade, shade, shade)
                self._safe_add_ellipse(
                    x - 1.2,
                    y - 1.2,
                    2.4,
                    2.4,
                    QPen(Qt.transparent),
                    QBrush(color),
                    context="mesh_nodes_3d",
                )

    def _safe_add_ellipse(self, x, y, w, h, pen=None, brush=None, context=None):
        if not hasattr(self, "_nan_ellipse_warned"):
            self._nan_ellipse_warned = False
        if not all(math.isfinite(v) for v in (x, y, w, h)):
            if not self._nan_ellipse_warned:
                ctx = f" context={context}" if context else ""
                print(f"[WARN] Skipping ellipse with non-finite params: x={x}, y={y}, w={w}, h={h}.{ctx}")
                self._nan_ellipse_warned = True
            return None
        if w <= 0 or h <= 0:
            return None
        try:
            return self.scene().addEllipse(x, y, w, h, pen or QPen(), brush or QBrush())
        except Exception:
            return None

    def _draw_displacement_view(self):
        if not self._displacement_vectors:
            self._add_info_text("No displacement data.")
            return

        if self.should_use_gpu_results_preview(len(self._displacement_vectors)):
            window = self.window()
            if window is not None and hasattr(window, "_show_results_point_preview"):
                positions = np.asarray(
                    [(x + dx, y + dy) for (x, y, dx, dy) in self._displacement_vectors],
                    dtype=float,
                )
                scalars = None
                if self._result_scalar_values is not None:
                    scalars = np.asarray(self._result_scalar_values, dtype=float).reshape(-1)
                try:
                    if window._show_results_point_preview(positions, scalars=scalars, auto_fit=True):
                        return
                except Exception:
                    pass

        window = self.window()
        if window is not None and hasattr(window, "_hide_results_point_preview"):
            try:
                window._hide_results_point_preview(force_view=True)
            except Exception:
                pass

        default_color = QColor(30, 120, 220)
        scalar_vals = self._result_scalar_values
        colors = None
        if scalar_vals is not None:
            arr = np.asarray(scalar_vals, dtype=float).reshape(-1)
            if arr.size == len(self._displacement_vectors):
                finite = np.isfinite(arr)
                if np.any(finite):
                    vmin = float(np.min(arr[finite]))
                    vmax = float(np.max(arr[finite]))
                    if abs(vmax - vmin) < 1e-12:
                        norm = np.zeros_like(arr, dtype=float)
                    else:
                        norm = (arr - vmin) / (vmax - vmin)
                    norm[~finite] = 0.0
                    try:
                        from matplotlib import cm as mpl_cm
                        rgba = np.asarray(mpl_cm.viridis(norm), dtype=float)
                        colors = [
                            QColor(
                                int(max(0.0, min(1.0, c[0])) * 255.0),
                                int(max(0.0, min(1.0, c[1])) * 255.0),
                                int(max(0.0, min(1.0, c[2])) * 255.0),
                            )
                            for c in rgba
                        ]
                    except Exception:
                        colors = None

        for idx, (x, y, dx, dy) in enumerate(self._displacement_vectors):
            line_color = colors[idx] if colors is not None and idx < len(colors) else default_color
            pen = QPen(line_color, 1.5)
            head_pen = QPen(line_color, 1.5)
            head_brush = QBrush(line_color)
            x2 = x + dx
            y2 = y + dy
            self.scene().addLine(x, y, x2, y2, pen)
            self._safe_add_ellipse(x - 1.6, y - 1.6, 3.2, 3.2, QPen(Qt.transparent), QBrush(line_color), context="result_nodes")
            # Simple arrow head
            ang = math.atan2(dy, dx)
            hlen = 5.0
            left = (x2 - hlen * math.cos(ang - math.pi / 6), y2 - hlen * math.sin(ang - math.pi / 6))
            right = (x2 - hlen * math.cos(ang + math.pi / 6), y2 - hlen * math.sin(ang + math.pi / 6))
            poly = QPolygonF([QPointF(x2, y2), QPointF(left[0], left[1]), QPointF(right[0], right[1])])
            self.scene().addPolygon(poly, head_pen, head_brush)
        try:
            deformed_positions = np.asarray(
                [(x + dx, y + dy) for (x, y, dx, dy) in self._displacement_vectors],
                dtype=float,
            )
        except Exception:
            deformed_positions = None
        if deformed_positions is not None and len(deformed_positions) > 0:
            pass
        if self._result_field_label:
            self._add_info_text(f"Color by: {self._result_field_label} (viridis)")

    def _clear_dimension_item_cache(self):
        scene = self.scene()
        for dim_id, item in list(getattr(self, "_dimension_items", {}).items()):
            if item is None:
                continue
            try:
                if scene is not None and item.scene() is scene:
                    scene.removeItem(item)
            except Exception:
                pass
        self._dimension_items = {}
        self._active_dimension_item_ids = set()

    def _detach_dimension_items_from_scene(self):
        scene = self.scene()
        if scene is None:
            return
        for item in list(getattr(self, "_dimension_items", {}).values()):
            if item is None:
                continue
            try:
                if item.scene() is scene:
                    scene.removeItem(item)
            except Exception:
                pass

    def _remove_non_dimension_scene_items(self):
        scene = self.scene()
        if scene is None:
            return
        for item in list(scene.items()):
            if isinstance(item, DimensionTextItem):
                continue
            try:
                scene.removeItem(item)
            except Exception:
                pass

    def _has_active_dimension_editor(self):
        for item in list(getattr(self, "_dimension_items", {}).values()):
            if item is None:
                continue
            try:
                if bool(getattr(item, "_editing", False)):
                    return True
            except Exception:
                continue
        return False

    def _active_dimension_editor_item(self):
        scene = self.scene()
        if scene is not None:
            try:
                focus_item = scene.focusItem()
            except Exception:
                focus_item = None
            if isinstance(focus_item, DimensionTextItem) and bool(getattr(focus_item, "_editing", False)):
                return focus_item
        for item in list(getattr(self, "_dimension_items", {}).values()):
            if item is None:
                continue
            try:
                if bool(getattr(item, "_editing", False)):
                    return item
            except Exception:
                continue
        return None

    def _sync_dimension_items(self):
        scene = self.scene()
        active_ids = set(getattr(self, "_active_dimension_item_ids", set()))
        for dim_id, item in list(getattr(self, "_dimension_items", {}).items()):
            if dim_id in active_ids:
                continue
            if item is not None:
                try:
                    if scene is not None and item.scene() is scene:
                        scene.removeItem(item)
                except Exception:
                    pass
            self._dimension_items.pop(dim_id, None)
        self._active_dimension_item_ids = set()

    def _add_dimension_text(self, text, pos, dim_id):
        try:
            dim_key = int(dim_id)
        except Exception:
            dim_key = dim_id
        if dim_key is None:
            return
        self._active_dimension_item_ids.add(dim_key)
        item = self._dimension_items.get(dim_key)
        if item is None:
            item = DimensionTextItem(text, dim_key, self)
            self._dimension_items[dim_key] = item
        else:
            try:
                if not bool(getattr(item, "_editing", False)) and item.toPlainText() != str(text):
                    item.setPlainText(str(text))
            except Exception:
                item = DimensionTextItem(text, dim_key, self)
                self._dimension_items[dim_key] = item
        try:
            item.setDefaultTextColor(QColor(40, 40, 40))
        except Exception:
            item = DimensionTextItem(text, dim_key, self)
            self._dimension_items[dim_key] = item
            item.setDefaultTextColor(QColor(40, 40, 40))
        try:
            rect = item.boundingRect()
            px = float(pos[0]) - 0.5 * float(rect.width())
            py = float(pos[1]) - 0.5 * float(rect.height())
        except Exception:
            px, py = pos[0], pos[1]
        if not bool(getattr(item, "_editing", False)):
            item.setPos(px, py)
        if self._y_axis_up:
            item.setTransform(QTransform(1, 0, 0, -1, 0, 0))
        if item.scene() is None:
            self.scene().addItem(item)

    def _draw_linear_dimension(self, p1, p2, value, dim_id, offset=None):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            return
        nx, ny = -dy / length, dx / length
        if offset is None:
            offset = (nx * self._dim_offset, ny * self._dim_offset)
        p1o = (p1[0] + offset[0], p1[1] + offset[1])
        p2o = (p2[0] + offset[0], p2[1] + offset[1])
        pen = QPen(QColor(30, 70, 120), 1)
        self.scene().addLine(p1[0], p1[1], p1o[0], p1o[1], pen)
        self.scene().addLine(p2[0], p2[1], p2o[0], p2o[1], pen)
        self.scene().addLine(p1o[0], p1o[1], p2o[0], p2o[1], pen)
        mid = ((p1o[0] + p2o[0]) / 2.0, (p1o[1] + p2o[1]) / 2.0)
        self._add_dimension_text(f"{value:.3f}", mid, dim_id)

    def _draw_angle_dimension(self, p_prev, p_vertex, p_next, angle_deg, dim_id, offset=None):
        v1 = (p_prev[0] - p_vertex[0], p_prev[1] - p_vertex[1])
        v2 = (p_next[0] - p_vertex[0], p_next[1] - p_vertex[1])
        len1 = math.hypot(v1[0], v1[1])
        len2 = math.hypot(v2[0], v2[1])
        if len1 <= 1e-9 or len2 <= 1e-9:
            return
        v1n = (v1[0] / len1, v1[1] / len1)
        v2n = (v2[0] / len2, v2[1] / len2)
        if offset is None:
            offset = (0.0, 0.0)
        arc_r = self._dim_offset * 0.9
        p1 = (p_vertex[0] + v1n[0] * arc_r, p_vertex[1] + v1n[1] * arc_r)
        p2 = (p_vertex[0] + v2n[0] * arc_r, p_vertex[1] + v2n[1] * arc_r)
        pen = QPen(QColor(120, 80, 30), 1)
        self.scene().addLine(p_vertex[0], p_vertex[1], p1[0], p1[1], pen)
        self.scene().addLine(p_vertex[0], p_vertex[1], p2[0], p2[1], pen)
        text_pos = (p_vertex[0] + offset[0], p_vertex[1] + offset[1])
        if offset == (0.0, 0.0):
            text_pos = (p_vertex[0] + arc_r, p_vertex[1] + arc_r)
        self._add_dimension_text(f"{angle_deg:.2f} deg", text_pos, dim_id)

    def _draw_radius_dimension(self, center, radius, dim_id, offset=None, label="R", display_value=None):
        if radius <= 0:
            return
        if offset is None:
            offset = (self._dim_offset, self._dim_offset)
        p_edge = (center[0] + radius, center[1])
        pen = QPen(QColor(30, 70, 120), 1)
        self.scene().addLine(center[0], center[1], p_edge[0], p_edge[1], pen)
        text_pos = (center[0] + offset[0], center[1] + offset[1])
        if display_value is None:
            display_value = radius
        self._add_dimension_text(f"{label}{display_value:.3f}", text_pos, dim_id)

    def _arc_sweep_angle(self, meta):
        try:
            start_angle = float(meta.get("start_angle", 0.0))
            end_angle = float(meta.get("end_angle", start_angle))
        except Exception:
            return 0.0
        delta = end_angle - start_angle
        if not math.isfinite(delta):
            return 0.0
        return float(delta)

    def _draw_arc_length_dimension(self, meta, dim_id, offset=None, display_value=None):
        center = meta.get("center")
        radius = float(meta.get("radius", 0.0) or 0.0)
        if center is None or radius <= 0.0:
            return
        sweep = self._arc_sweep_angle(meta)
        sweep_abs = abs(float(sweep))
        if sweep_abs <= 1e-9:
            return
        mid_angle = float(meta.get("start_angle", 0.0)) + 0.5 * sweep
        arc_mid = (
            center[0] + radius * math.cos(mid_angle),
            center[1] + radius * math.sin(mid_angle),
        )
        if offset is None:
            offset = (
                math.cos(mid_angle) * self._dim_offset,
                math.sin(mid_angle) * self._dim_offset,
            )
        text_pos = (arc_mid[0] + offset[0], arc_mid[1] + offset[1])
        pen = QPen(QColor(120, 80, 30), 1, Qt.DashLine)
        self.scene().addLine(arc_mid[0], arc_mid[1], text_pos[0], text_pos[1], pen)
        arc_len = radius * sweep_abs if display_value is None else display_value
        self._add_dimension_text(f"L{arc_len:.3f}", text_pos, dim_id)

    def _draw_dimensions_for_owner(self, owner_type, owner_part, dimensions):
        sketches, metas, _, _ = self._owner_collections(owner_type, owner_part)
        for dim in dimensions:
            dim_type = str(dim.get("dim_type", "")).lower()
            sketch_index = int(dim.get("sketch_index", -1))
            if sketch_index < 0 or sketch_index >= len(sketches):
                continue
            if not self._is_sketch_visible(owner_type, owner_part, sketch_index):
                continue
            points = sketches[sketch_index]
            meta = metas[sketch_index] if sketch_index < len(metas) else {}
            if bool(meta.get("skip_auto_dimension", False)):
                continue
            if dim_type == "linear" and str(meta.get("type", "")).lower() == "line":
                meta = self._normalize_line_meta(meta, fallback_points=points)
            elif dim_type in {"rect_width", "rect_height"}:
                meta = self._normalize_rectangle_meta(meta, fallback_points=points)
            elif dim_type in {"diameter", "radius"} and str(meta.get("type", "")).lower() == "circle":
                meta = self._normalize_circle_meta(meta, fallback_points=points)
            if not self.show_dimensions and not self._is_persistent_parametric_annotation(meta, dim_type):
                continue
            draw_offset = dim.get("offset")
            if draw_offset is None:
                draw_offset = self._auto_dimension_offset_for_draw(
                    owner_type, owner_part, dim, points, meta
                )
            if dim.get("id") == self.selected_dimension_id:
                self._highlight_dimension_geometry(owner_type, owner_part, dim)
            if dim_type == "linear":
                seg_index = int(dim.get("segment_index", 0))
                if seg_index < 0 or seg_index + 1 >= len(points):
                    continue
                p1 = points[seg_index]
                p2 = points[seg_index + 1]
                value = self._dimension_current_value_ui(dim, owner_type, owner_part)
                self._draw_linear_dimension(p1, p2, value, dim.get("id"), draw_offset)
            elif dim_type == "point_distance":
                first_idx = int(dim.get("first_vertex_index", -1))
                second_idx = int(dim.get("second_vertex_index", -1))
                if first_idx < 0 or second_idx < 0 or first_idx >= len(points) or second_idx >= len(points):
                    continue
                p1 = points[first_idx]
                p2 = points[second_idx]
                value = self._dimension_current_value_ui(dim, owner_type, owner_part)
                self._draw_linear_dimension(p1, p2, value, dim.get("id"), draw_offset)
            elif dim_type == "rect_width":
                if len(points) < 2:
                    continue
                p1, p2 = points[0], points[1]
                value = self._dimension_current_value_ui(dim, owner_type, owner_part)
                self._draw_linear_dimension(p1, p2, value, dim.get("id"), draw_offset)
            elif dim_type == "rect_height":
                if len(points) < 3:
                    continue
                p1, p2 = points[1], points[2]
                value = self._dimension_current_value_ui(dim, owner_type, owner_part)
                self._draw_linear_dimension(p1, p2, value, dim.get("id"), draw_offset)
            elif dim_type == "diameter":
                center = meta.get("center")
                radius = meta.get("radius")
                if center is None or radius is None:
                    continue
                value = self._dimension_current_value_ui(dim, owner_type, owner_part)
                self._draw_radius_dimension(
                    center, radius, dim.get("id"), draw_offset, label="D", display_value=value
                )
            elif dim_type == "radius":
                center = meta.get("center")
                radius = meta.get("radius")
                if center is None or radius is None:
                    continue
                value = self._dimension_current_value_ui(dim, owner_type, owner_part)
                self._draw_radius_dimension(
                    center, radius, dim.get("id"), draw_offset, label="R", display_value=value
                )
            elif dim_type == "arc_length":
                if str(meta.get("type", "")).lower() != "arc":
                    continue
                value = self._dimension_current_value_ui(dim, owner_type, owner_part)
                self._draw_arc_length_dimension(meta, dim.get("id"), draw_offset, display_value=value)
            elif dim_type == "slot_length":
                p1 = meta.get("p1")
                p2 = meta.get("p2")
                if p1 is None or p2 is None:
                    continue
                value = self._dimension_current_value_ui(dim, owner_type, owner_part)
                self._draw_linear_dimension(p1, p2, value, dim.get("id"), draw_offset)
            elif dim_type == "slot_width":
                width = meta.get("width")
                p1 = meta.get("p1")
                p2 = meta.get("p2")
                if width is None or p1 is None or p2 is None:
                    continue
                mid = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                length = math.hypot(dx, dy)
                if length <= 1e-9:
                    continue
                nx, ny = -dy / length, dx / length
                half = width / 2.0
                a = (mid[0] + nx * half, mid[1] + ny * half)
                b = (mid[0] - nx * half, mid[1] - ny * half)
                value = self._dimension_current_value_ui(dim, owner_type, owner_part)
                self._draw_linear_dimension(a, b, value, dim.get("id"), draw_offset)
            elif dim_type == "polygon_radius":
                center = meta.get("center")
                radius = meta.get("radius")
                if center is None or radius is None:
                    continue
                value = self._dimension_current_value_ui(dim, owner_type, owner_part)
                self._draw_radius_dimension(
                    center, radius, dim.get("id"), draw_offset, label="R", display_value=value
                )
            elif dim_type == "angle":
                v_idx = int(dim.get("vertex_index", 0))
                if v_idx <= 0 or v_idx + 1 >= len(points):
                    continue
                p_prev = points[v_idx - 1]
                p_vertex = points[v_idx]
                p_next = points[v_idx + 1]
                v1 = (p_prev[0] - p_vertex[0], p_prev[1] - p_vertex[1])
                v2 = (p_next[0] - p_vertex[0], p_next[1] - p_vertex[1])
                len1 = math.hypot(v1[0], v1[1])
                len2 = math.hypot(v2[0], v2[1])
                if len1 <= 1e-9 or len2 <= 1e-9:
                    continue
                dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (len1 * len2)
                dot = max(-1.0, min(1.0, dot))
                angle_deg = math.degrees(math.acos(dot))
                self._draw_angle_dimension(
                    p_prev, p_vertex, p_next, angle_deg, dim.get("id"), draw_offset
                )

    def _draw_dimensions(self):
        if self.display_mode not in ("geometry", "sketch_edit"):
            self._clear_dimension_item_cache()
            return
        self._active_dimension_item_ids = set()
        edit_part = self.get_part_shape_edit_target()
        edit_part_id = int(getattr(edit_part, "id", -1)) if edit_part is not None else None
        if self.dimensions:
            self._draw_dimensions_for_owner("sketch", None, self.dimensions)
        for part in self.parts:
            if (
                edit_part_id is not None
                and self.active_module == "Part"
                and int(getattr(part, "id", -2)) == edit_part_id
                and self.sketches
            ):
                continue
            dims = getattr(part, "dimensions", [])
            if dims:
                self._draw_dimensions_for_owner("part", part, dims)
        self._sync_dimension_items()

    def get_pore_domain_geometry(self):
        if self.selected_part_id is not None:
            part = next((p for p in self.parts if p.id == self.selected_part_id), None)
            if part and part.geometry and not part.geometry.is_empty:
                return part.geometry
        if self.solid_geometry is not None and not self.solid_geometry.is_empty:
            return self.solid_geometry
        try:
            geom = self._sketches_to_shapely()
            if geom is not None and not geom.is_empty:
                return geom
        except Exception:
            pass
        return None

    def show_pore_preview(self, polygons):
        self._pore_preview_polys = list(polygons or [])
        self.redraw()

    def clear_pore_preview(self):
        if self._pore_preview_polys:
            self._pore_preview_polys = []
            self.redraw()

    def _apply_pore_geometry(self, polygons, mode="holes", base_name="Feature"):
        if not polygons:
            return False
        self.push_undo_state()
        holes_geom = unary_union(polygons)
        if holes_geom is None or holes_geom.is_empty:
            return False
        parts_added = 0
        if str(mode).lower().startswith("hole"):
            solids = [p for p in self.parts if not p.is_void and p.geometry and not p.geometry.is_empty]
            for existing_part in solids:
                if not existing_part.geometry.intersects(holes_geom):
                    continue
                try:
                    void_geom = existing_part.geometry.intersection(holes_geom)
                except Exception:
                    continue
                if void_geom is None or void_geom.is_empty:
                    continue
                geoms = [void_geom] if isinstance(void_geom, Polygon) else list(void_geom.geoms)
                for geom in geoms:
                    if geom is None or geom.is_empty:
                        continue
                    hole_part_name = f"{base_name} {parts_added + 1} (in {existing_part.name})"
                    hole_part = Part(hole_part_name, geometry=geom, is_void=True)
                    hole_part.parent_id = existing_part.id
                    hole_part.storage_units = "ui"
                    self._set_generated_feature_metadata(
                        hole_part,
                        feature_kind="porous_holes",
                        settings=getattr(self, "porous_settings", None),
                    )
                    self._sync_cad_shape(hole_part)
                    self.parts.append(hole_part)
                    parts_added += 1
        else:
            geoms = [holes_geom] if isinstance(holes_geom, Polygon) else list(holes_geom.geoms)
            particle_geoms = [geom for geom in geoms if geom is not None and not geom.is_empty]
            if particle_geoms:
                grouped_geom = unary_union(particle_geoms)
                name = str(base_name or "ParticleCloud") or "ParticleCloud"
                new_part = Part(name, geometry=grouped_geom, is_void=False)
                new_part.part_type = "particle_set"
                new_part.particles = self._particle_records_from_polygons(particle_geoms)
                new_part.sketches, new_part.sketch_meta = self._polygons_to_sketches(particle_geoms)
                new_part.storage_units = "ui"
                for meta in list(getattr(new_part, "sketch_meta", []) or []):
                    if isinstance(meta, dict):
                        meta["generated_pattern"] = "porous"
                        meta["skip_auto_dimension"] = True
                self._set_generated_feature_metadata(
                    new_part,
                    feature_kind="porous_particles",
                    settings=getattr(self, "porous_settings", None),
                )
                for existing_part in self.parts:
                    if not existing_part.is_void and existing_part.geometry and existing_part.geometry.covers(grouped_geom):
                        new_part.parent_id = existing_part.id
                        break
                self._sync_cad_shape(new_part)
                self.parts.append(new_part)
                parts_added = 1
        if parts_added:
            self.partsChanged.emit()
            self.geometryChanged.emit()
            self.rebuild_display_geometry()
            self.redraw()
            return True
        return False

    def _polygons_to_sketches(self, polygons):
        sketches = []
        metas = []
        for poly in polygons:
            if poly is None or poly.is_empty:
                continue
            geoms = [poly] if isinstance(poly, Polygon) else list(poly.geoms)
            for g in geoms:
                ext = list(g.exterior.coords)
                if len(ext) < 3:
                    continue
                pts = [(float(x), float(y)) for x, y in ext]
                if pts and pts[0] != pts[-1]:
                    pts.append(pts[0])
                sketches.append(pts)
                metas.append(
                    {
                        "type": "polyline",
                        "points": copy.deepcopy(pts),
                        "generated_pattern": "porous",
                        "skip_auto_dimension": True,
                    }
                )
        return sketches, metas

    def _particle_records_from_polygons(self, polygons):
        records = []
        for poly in list(polygons or []):
            if poly is None or getattr(poly, "is_empty", True):
                continue
            try:
                center = poly.centroid
            except Exception:
                continue
            width = 0.0
            height = 0.0
            angle_deg = 0.0
            try:
                rect = poly.minimum_rotated_rectangle
                coords = list(rect.exterior.coords)
                if len(coords) >= 4:
                    edge_1 = dist(coords[0], coords[1])
                    edge_2 = dist(coords[1], coords[2])
                    width = float(max(edge_1, edge_2))
                    height = float(min(edge_1, edge_2))
                    dx = float(coords[1][0] - coords[0][0])
                    dy = float(coords[1][1] - coords[0][1])
                    if edge_2 > edge_1:
                        dx = float(coords[2][0] - coords[1][0])
                        dy = float(coords[2][1] - coords[1][1])
                    angle_deg = float(math.degrees(math.atan2(dy, dx)))
            except Exception:
                try:
                    minx, miny, maxx, maxy = poly.bounds
                    width = float(maxx - minx)
                    height = float(maxy - miny)
                except Exception:
                    pass
            records.append(
                {
                    "x": float(center.x),
                    "y": float(center.y),
                    "angle": float(angle_deg),
                    "width": float(width),
                    "height": float(height),
                }
            )
        return records

    def _geometry_to_editable_boundary_sketches(self, geom):
        """
        Convert a part geometry into editable boundary polyline sketches.

        Intended as a fallback for direct-edit/CAD parts that do not retain sketch history.
        Returns None when conversion is unsafe (e.g., holes in the same part geometry, which the
        current sketch->solid conversion cannot round-trip without filling them).
        """
        if geom is None or getattr(geom, "is_empty", True):
            return None
        try:
            safe_geom = geom.buffer(0)
        except Exception:
            safe_geom = geom
        if safe_geom is None or getattr(safe_geom, "is_empty", True):
            return None
        try:
            geoms = [safe_geom] if isinstance(safe_geom, Polygon) else list(getattr(safe_geom, "geoms", []))
        except Exception:
            geoms = [safe_geom]
        if not geoms:
            return None

        # Guard: interior rings (holes) cannot be round-tripped by current sketch polygonization path.
        for g in geoms:
            if g is None or getattr(g, "is_empty", True):
                continue
            if getattr(g, "geom_type", "") != "Polygon":
                return None
            try:
                if len(getattr(g, "interiors", [])) > 0:
                    return None
            except Exception:
                return None

        sketches = []
        metas = []
        for g in geoms:
            if g is None or getattr(g, "is_empty", True):
                continue
            try:
                ext = list(g.exterior.coords)
            except Exception:
                continue
            if len(ext) < 3:
                continue
            pts = [(float(x), float(y)) for x, y in ext]
            try:
                if pts and dist(pts[0], pts[-1]) > 1e-9:
                    pts.append(pts[0])
            except Exception:
                if pts and pts[0] != pts[-1]:
                    pts.append(pts[0])
            if len(pts) < 4:
                continue
            sketches.append(pts)
            metas.append(
                {
                    "type": "polyline",
                    "points": copy.deepcopy(pts),
                    "generated_from_geometry_boundary": True,
                    "skip_auto_dimension": True,
                }
            )
        if not sketches:
            return None
        return {"sketches": sketches, "sketch_meta": metas}

    def apply_pore_sketches(self, polygons, base_name="Feature", push_undo=True):
        sketches, metas = self._polygons_to_sketches(polygons)
        if not sketches:
            return False
        if push_undo:
            self.push_undo_state()
        self.sketches = sketches
        self.sketch_meta = metas
        self.dimensions = []
        self.constraints = []
        self._porous_sketch_name = base_name or "Feature"
        self._pending_generated_feature_settings = copy.deepcopy(getattr(self, "porous_settings", None))
        self.geometryChanged.emit()
        self.redraw()
        return True

    def _set_generated_feature_metadata(self, part, feature_kind=None, settings=None):
        if part is None:
            return
        try:
            part.generated_feature_kind = str(feature_kind or "") or None
        except Exception:
            part.generated_feature_kind = None
        try:
            part.generated_feature_settings = copy.deepcopy(settings) if settings else None
        except Exception:
            part.generated_feature_settings = None

    def is_generated_feature_part(self, part):
        if part is None:
            return False
        if getattr(part, "generated_feature_settings", None):
            return True
        for meta in list(getattr(part, "sketch_meta", []) or []):
            if str((meta or {}).get("generated_pattern", "")).lower() == "porous":
                return True
        return False

    def update_generated_feature_part(self, part, polygons, settings=None, base_name=None):
        if part is None or not polygons:
            return False
        union_geom = unary_union(polygons)
        if union_geom is None or union_geom.is_empty:
            return False
        sketches, metas = self._polygons_to_sketches(polygons)
        if not sketches:
            return False
        self.push_undo_state()
        part.geometry = union_geom
        part.name = str(base_name or getattr(part, "name", "Feature") or "Feature")
        part.sketches = sketches
        part.sketch_meta = metas
        part.dimensions = []
        part.constraints = []
        part.is_direct_edit = False
        feature_kind = "porous_holes" if getattr(part, "is_void", False) else "porous_particles"
        if feature_kind == "porous_particles":
            part.part_type = "particle_set"
            part.particles = self._particle_records_from_polygons(polygons)
        else:
            part.part_type = "void" if getattr(part, "is_void", False) else "solid"
            part.particles = []
        self._set_generated_feature_metadata(part, feature_kind=feature_kind, settings=settings)
        self._sync_cad_shape(part)
        self.rebuild_display_geometry()
        self.partsChanged.emit()
        self.geometryChanged.emit()
        self.redraw()
        return True

    def _find_dimension_by_id(self, dim_id):
        for idx, dim in enumerate(self.dimensions):
            if dim.get("id") == dim_id:
                return dim, "sketch", None, self.dimensions
        for part in self.parts:
            for dim in getattr(part, "dimensions", []):
                if dim.get("id") == dim_id:
                    return dim, "part", part, part.dimensions
        return None, None, None, None

    def select_dimension(self, dim_id, suppress_redraw=False):
        try:
            self.selected_dimension_id = int(dim_id) if dim_id is not None else None
        except Exception:
            self.selected_dimension_id = None

        if not suppress_redraw:
            self.redraw()

    def _dimension_entity_id(self, owner_type, owner_part, dim):
        owner_id = "sketch"
        if owner_type == "part" and owner_part is not None:
            owner_id = f"part:{getattr(owner_part, 'id', 'unknown')}"
        sketch_index = int(dim.get("sketch_index", -1))
        dim_type = str(dim.get("dim_type", ""))
        if dim_type == "point_distance":
            a = int(dim.get("first_vertex_index", -1))
            b = int(dim.get("second_vertex_index", -1))
            return f"{owner_id}:sketch:{sketch_index}:vertices:{a}:{b}"
        if dim_type == "angle":
            vertex_index = int(dim.get("vertex_index", -1))
            return f"{owner_id}:sketch:{sketch_index}:angle:{vertex_index}"
        segment_index = int(dim.get("segment_index", -1))
        return f"{owner_id}:sketch:{sketch_index}:segment:{segment_index}:{dim_type}"

    def _dimension_current_value(self, dim, owner_type, owner_part):
        dim_type = str(dim.get("dim_type", "")).lower()
        sketch_index = int(dim.get("sketch_index", -1))
        points = self._get_sketch_points(owner_type, owner_part, sketch_index) or []
        meta_list = self._owner_collections(owner_type, owner_part)[1]
        meta = meta_list[sketch_index] if 0 <= sketch_index < len(meta_list) else {}
        if dim_type == "linear":
            if str(meta.get("type", "")).lower() == "line":
                meta = self._normalize_line_meta(meta, fallback_points=points)
                p1 = meta.get("p1")
                p2 = meta.get("p2")
                if p1 and p2:
                    return dist(p1, p2)
            seg_index = int(dim.get("segment_index", 0))
            if 0 <= seg_index + 1 < len(points):
                return dist(points[seg_index], points[seg_index + 1])
            return 0.0
        if dim_type == "point_distance":
            a = int(dim.get("first_vertex_index", -1))
            b = int(dim.get("second_vertex_index", -1))
            if 0 <= a < len(points) and 0 <= b < len(points):
                return dist(points[a], points[b])
            return 0.0
        if dim_type == "rect_width":
            meta = self._normalize_rectangle_meta(meta, fallback_points=points)
            return abs(float(meta.get("width", 0.0)))
        if dim_type == "rect_height":
            meta = self._normalize_rectangle_meta(meta, fallback_points=points)
            return abs(float(meta.get("height", 0.0)))
        if dim_type == "diameter":
            if str(meta.get("type", "")).lower() == "circle":
                meta = self._normalize_circle_meta(meta, fallback_points=points)
            return float(meta.get("radius", 0.0)) * 2.0
        if dim_type == "radius":
            if str(meta.get("type", "")).lower() == "circle":
                meta = self._normalize_circle_meta(meta, fallback_points=points)
            return float(meta.get("radius", 0.0))
        if dim_type == "arc_length":
            radius = float(meta.get("radius", 0.0) or 0.0)
            sweep = abs(self._arc_sweep_angle(meta))
            return radius * sweep
        if dim_type == "slot_length":
            p1 = meta.get("p1")
            p2 = meta.get("p2")
            if p1 and p2:
                return dist(p1, p2)
            return 0.0
        if dim_type == "slot_width":
            return float(meta.get("width", 0.0))
        if dim_type == "polygon_radius":
            return float(meta.get("radius", 0.0))
        if dim_type == "angle":
            v_idx = int(dim.get("vertex_index", 0))
            if 0 < v_idx + 1 < len(points):
                p_prev = points[v_idx - 1]
                p_vertex = points[v_idx]
                p_next = points[v_idx + 1]
                v1 = (p_prev[0] - p_vertex[0], p_prev[1] - p_vertex[1])
                v2 = (p_next[0] - p_vertex[0], p_next[1] - p_vertex[1])
                len1 = math.hypot(v1[0], v1[1])
                len2 = math.hypot(v2[0], v2[1])
                if len1 > 1e-9 and len2 > 1e-9:
                    dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (len1 * len2)
                    dot = max(-1.0, min(1.0, dot))
                    return math.degrees(math.acos(dot))
            return 0.0
        return 0.0
    
    def _dimension_current_value_ui(self, dim, owner_type, owner_part):
        """Get dimension value in UI units for display."""
        raw_value = self._dimension_current_value(dim, owner_type, owner_part)
        dim_type = str(dim.get("dim_type", "")).lower()
        
        if dim_type == "angle":
            return raw_value
        return self._owner_value_to_ui(raw_value, owner_type, owner_part)

    def _sync_dimension_metadata(self, dim, owner_type, owner_part):
        if dim is None:
            return
        dim_type = str(dim.get("dim_type", ""))
        dim["dimension_type"] = dim_type
        dim["entity_id"] = self._dimension_entity_id(owner_type, owner_part, dim)
        dim["value"] = self._dimension_current_value(dim, owner_type, owner_part)

    def _build_dimension_record(self, owner_type, owner_part, dim_type, sketch_index, **extra):
        dim = {
            "id": self._next_dimension_id(),
            "dim_type": dim_type,
            "dimension_type": dim_type,
            "sketch_index": int(sketch_index),
        }
        dim.update(extra)
        self._sync_dimension_metadata(dim, owner_type, owner_part)
        return dim

    def _highlight_dimension_geometry(self, owner_type, owner_part, dim):
        dim_type = str(dim.get("dim_type", "")).lower()
        sketch_index = int(dim.get("sketch_index", -1))
        points = self._get_sketch_points(owner_type, owner_part, sketch_index) or []
        meta_list = self._owner_collections(owner_type, owner_part)[1]
        meta = meta_list[sketch_index] if 0 <= sketch_index < len(meta_list) else {}
        pen = QPen(QColor(255, 140, 30), 3.0)
        brush = QBrush(QColor(255, 210, 120, 35))
        if dim_type in ("linear", "rect_width", "rect_height"):
            seg_index = int(dim.get("segment_index", 0))
            if 0 <= seg_index + 1 < len(points):
                p1 = points[seg_index]
                p2 = points[seg_index + 1]
                self.scene().addLine(p1[0], p1[1], p2[0], p2[1], pen)
            return
        if dim_type == "point_distance":
            a = int(dim.get("first_vertex_index", -1))
            b = int(dim.get("second_vertex_index", -1))
            if 0 <= a < len(points) and 0 <= b < len(points):
                p1 = points[a]
                p2 = points[b]
                self.scene().addLine(p1[0], p1[1], p2[0], p2[1], pen)
                self.scene().addEllipse(p1[0] - 3.5, p1[1] - 3.5, 7, 7, pen, brush)
                self.scene().addEllipse(p2[0] - 3.5, p2[1] - 3.5, 7, 7, pen, brush)
            return
        if dim_type in ("diameter", "radius", "arc_length", "polygon_radius"):
            center = meta.get("center")
            radius = float(meta.get("radius", 0.0) or 0.0)
            if center is None or radius <= 0.0:
                return
            self.scene().addEllipse(center[0] - radius, center[1] - radius, radius * 2.0, radius * 2.0, pen)
            self.scene().addLine(center[0], center[1], center[0] + radius, center[1], pen)
            return
        if dim_type in ("slot_length", "slot_width"):
            p1 = meta.get("p1")
            p2 = meta.get("p2")
            if p1 is not None and p2 is not None:
                self.scene().addLine(p1[0], p1[1], p2[0], p2[1], pen)
            return
        if dim_type == "angle":
            v_idx = int(dim.get("vertex_index", 0))
            if 0 < v_idx + 1 < len(points):
                p_prev = points[v_idx - 1]
                p_vertex = points[v_idx]
                p_next = points[v_idx + 1]
                self.scene().addLine(p_prev[0], p_prev[1], p_vertex[0], p_vertex[1], pen)
                self.scene().addLine(p_vertex[0], p_vertex[1], p_next[0], p_next[1], pen)
                self.scene().addEllipse(p_vertex[0] - 3.5, p_vertex[1] - 3.5, 7, 7, pen, brush)

    def _finalize_dimension_change(self, dim, owner_type, owner_part):
        """Finalize a dimension edit without leaving sketch-edit mode or refitting the view."""
        self._sync_dimension_metadata(dim, owner_type, owner_part)
        if owner_type == "part" and owner_part is not None:
            # Keep part-owned sketch edits in preview mode. Rebuild the display geometry
            # so the user sees the updated shape, but do not emit part-commit signals.
            owner_part.geometry = self._sketches_to_shapely_list(owner_part.sketches)
            self.rebuild_display_geometry()
        self.select_dimension(dim.get("id"), suppress_redraw=True)
        try:
            self.redraw()
        except Exception:
            pass
        window = self.window()
        inspector = getattr(window, "property_inspector", None) if window is not None else None
        if inspector is not None and hasattr(inspector, "refresh_current_selection"):
            QTimer.singleShot(0, inspector.refresh_current_selection)

    def _dimension_owner_geometry(self, owner_type, owner_part, points, meta):
        try:
            if owner_type == "part" and owner_part is not None:
                geom = getattr(owner_part, "geometry", None)
                if geom is not None and not getattr(geom, "is_empty", True):
                    return geom
        except Exception:
            pass
        meta = meta or {}
        meta_type = str(meta.get("type", "polyline")).lower()
        try:
            if meta_type in ("rectangle", "polygon", "slot", "polyline"):
                if points is not None and len(points) >= 4 and dist(points[0], points[-1]) <= 1e-6:
                    poly = Polygon(points)
                    if poly is not None and not poly.is_empty:
                        return poly
            elif meta_type == "circle":
                center = meta.get("center")
                radius = float(meta.get("radius", 0.0) or 0.0)
                if center is not None and radius > 0:
                    return Point(center).buffer(radius)
        except Exception:
            pass
        return None

    def _point_on_or_inside_geometry(self, geom, pt):
        if geom is None:
            return False
        try:
            return bool(geom.buffer(1e-9).covers(Point(pt)))
        except Exception:
            try:
                return bool(geom.covers(Point(pt)))
            except Exception:
                return False

    def _choose_outside_offset(self, anchor, directions, offset_mag, owner_geom=None, min_extra=0.0):
        try:
            mag = float(offset_mag) + max(0.0, float(min_extra))
        except Exception:
            mag = float(self._dim_offset)
        mag = max(1.0, mag)
        best_vec = None
        best_score = -1e18
        for d in directions or []:
            try:
                dx, dy = float(d[0]), float(d[1])
            except Exception:
                continue
            n = math.hypot(dx, dy)
            if n <= 1e-9:
                continue
            ux, uy = dx / n, dy / n
            vec = (ux * mag, uy * mag)
            p = (anchor[0] + vec[0], anchor[1] + vec[1])
            inside = self._point_on_or_inside_geometry(owner_geom, p) if owner_geom is not None else False
            score = 1000.0 if not inside else 0.0
            if owner_geom is not None:
                try:
                    score += float(owner_geom.distance(Point(p)))
                except Exception:
                    pass
            if score > best_score:
                best_score = score
                best_vec = vec
        if best_vec is None:
            return (mag, 0.0)
        return best_vec

    def _auto_dimension_offset_for_draw(self, owner_type, owner_part, dim, points, meta):
        dim_type = str(dim.get("dim_type", "")).lower()
        owner_geom = self._dimension_owner_geometry(owner_type, owner_part, points, meta)
        base_offset = float(self._dim_offset)

        if dim_type in ("linear", "rect_width", "rect_height"):
            seg_index = int(dim.get("segment_index", 0))
            if seg_index < 0 or seg_index + 1 >= len(points):
                return None
            p1 = points[seg_index]
            p2 = points[seg_index + 1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.hypot(dx, dy)
            if length <= 1e-9:
                return None
            mid = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
            n1 = (-dy / length, dx / length)
            n2 = (dy / length, -dx / length)
            return self._choose_outside_offset(mid, [n1, n2], base_offset, owner_geom=owner_geom)

        if dim_type == "slot_length":
            p1 = meta.get("p1"); p2 = meta.get("p2")
            if not p1 or not p2:
                return None
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.hypot(dx, dy)
            if length <= 1e-9:
                return None
            mid = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
            n1 = (-dy / length, dx / length)
            n2 = (dy / length, -dx / length)
            return self._choose_outside_offset(mid, [n1, n2], base_offset, owner_geom=owner_geom)

        if dim_type == "slot_width":
            width = float(meta.get("width", 0.0) or 0.0)
            p1 = meta.get("p1"); p2 = meta.get("p2")
            if width <= 0 or not p1 or not p2:
                return None
            mid = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.hypot(dx, dy)
            if length <= 1e-9:
                return None
            d1 = (dx / length, dy / length)
            d2 = (-dx / length, -dy / length)
            return self._choose_outside_offset(
                mid, [d1, d2], base_offset, owner_geom=owner_geom, min_extra=width
            )

        if dim_type == "arc_length":
            center = meta.get("center")
            radius = abs(float(meta.get("radius", 0.0) or 0.0))
            if center is None or radius <= 0.0:
                return None
            sweep = self._arc_sweep_angle(meta)
            if abs(sweep) <= 1e-9:
                return None
            mid_angle = float(meta.get("start_angle", 0.0)) + 0.5 * sweep
            anchor = (
                center[0] + radius * math.cos(mid_angle),
                center[1] + radius * math.sin(mid_angle),
            )
            radial = (math.cos(mid_angle), math.sin(mid_angle))
            return self._choose_outside_offset(
                anchor,
                [radial, (-radial[0], -radial[1])],
                base_offset,
                owner_geom=owner_geom,
            )

        if dim_type in ("diameter", "radius", "polygon_radius"):
            center = meta.get("center")
            radius = abs(float(meta.get("radius", 0.0) or 0.0))
            if center is None:
                return None
            dirs = []
            if owner_geom is not None:
                try:
                    c = owner_geom.centroid
                    vx = float(center[0]) - float(c.x)
                    vy = float(center[1]) - float(c.y)
                    if math.hypot(vx, vy) > 1e-9:
                        dirs.extend([(vx, vy), (-vx, -vy)])
                except Exception:
                    pass
            dirs.extend([(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)])
            return self._choose_outside_offset(
                center, dirs, base_offset, owner_geom=owner_geom, min_extra=radius
            )

        if dim_type == "angle":
            v_idx = int(dim.get("vertex_index", 0))
            if v_idx <= 0 or v_idx + 1 >= len(points):
                return None
            p_prev = points[v_idx - 1]
            p_vertex = points[v_idx]
            p_next = points[v_idx + 1]
            v1 = np.array([p_prev[0] - p_vertex[0], p_prev[1] - p_vertex[1]], dtype=float)
            v2 = np.array([p_next[0] - p_vertex[0], p_next[1] - p_vertex[1]], dtype=float)
            n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
            if n1 <= 1e-9 or n2 <= 1e-9:
                return None
            v1 /= n1
            v2 /= n2
            bis = v1 + v2
            if np.linalg.norm(bis) <= 1e-9:
                bis = np.array([v1[1], -v1[0]], dtype=float)
            return self._choose_outside_offset(
                p_vertex,
                [(float(bis[0]), float(bis[1])), (float(-bis[0]), float(-bis[1]))],
                base_offset * 1.4,
                owner_geom=owner_geom,
            )

        return None

    def _dimension_drag_anchor(self, dim, owner_type, owner_part):
        dim_type = str(dim.get("dim_type", "")).lower()
        sketch_index = int(dim.get("sketch_index", -1))
        points = self._get_sketch_points(owner_type, owner_part, sketch_index) or []
        meta = self._get_sketch_meta(owner_type, owner_part, sketch_index)
        if dim_type in ("linear", "rect_width", "rect_height"):
            seg_index = int(dim.get("segment_index", 0))
            if seg_index < 0 or seg_index + 1 >= len(points):
                return None
            p1 = points[seg_index]
            p2 = points[seg_index + 1]
            return ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
        if dim_type == "point_distance":
            first_idx = int(dim.get("first_vertex_index", -1))
            second_idx = int(dim.get("second_vertex_index", -1))
            if first_idx < 0 or second_idx < 0 or first_idx >= len(points) or second_idx >= len(points):
                return None
            p1 = points[first_idx]
            p2 = points[second_idx]
            return ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
        if dim_type in ("diameter", "radius", "polygon_radius"):
            center = meta.get("center")
            return tuple(center) if center is not None else None
        if dim_type == "arc_length":
            center = meta.get("center")
            radius = abs(float(meta.get("radius", 0.0) or 0.0))
            if center is None or radius <= 0.0:
                return None
            sweep = self._arc_sweep_angle(meta)
            if abs(sweep) <= 1e-9:
                return None
            mid_angle = float(meta.get("start_angle", 0.0)) + 0.5 * sweep
            return (
                center[0] + radius * math.cos(mid_angle),
                center[1] + radius * math.sin(mid_angle),
            )
        if dim_type == "slot_length":
            p1 = meta.get("p1"); p2 = meta.get("p2")
            if p1 and p2:
                return ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
            return None
        if dim_type == "slot_width":
            p1 = meta.get("p1"); p2 = meta.get("p2")
            width = float(meta.get("width", 0.0) or 0.0)
            if not p1 or not p2:
                return None
            mid = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.hypot(dx, dy)
            if length <= 1e-9 or width <= 0:
                return mid
            nx, ny = -dy / length, dx / length
            half = width * 0.5
            return (mid[0] + nx * half, mid[1] + ny * half)
        if dim_type == "angle":
            v_idx = int(dim.get("vertex_index", 0))
            if 0 <= v_idx < len(points):
                return points[v_idx]
        return None

    def _finish_dimension_label_drag(self, dim_id, center_pos):
        dim, owner_type, owner_part, _ = self._find_dimension_by_id(dim_id)
        if dim is None:
            return False
        anchor = self._dimension_drag_anchor(dim, owner_type, owner_part)
        if anchor is None:
            return False
        try:
            cx = float(center_pos[0]); cy = float(center_pos[1])
        except Exception:
            return False
        offset = (cx - anchor[0], cy - anchor[1])
        if math.hypot(offset[0], offset[1]) <= 1e-6:
            return False
        self.push_undo_state()
        dim["offset"] = offset
        self.geometryChanged.emit()
        self.redraw()
        return True

    def _apply_point_distance_dimension(self, owner_type, owner_part, sketch_index, first_index, second_index, new_value):
        sketches, metas, _, _ = self._owner_collections(owner_type, owner_part)
        if sketch_index < 0 or sketch_index >= len(sketches):
            return
        points = sketches[sketch_index]
        if (
            first_index < 0
            or second_index < 0
            or first_index >= len(points)
            or second_index >= len(points)
            or first_index == second_index
        ):
            return
        p1 = points[first_index]
        p2 = points[second_index]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            dx, dy, length = 1.0, 0.0, 1.0
        ux, uy = dx / length, dy / length
        new_p2 = (p1[0] + ux * new_value, p1[1] + uy * new_value)
        self._set_point(points, second_index, new_p2)
        meta = metas[sketch_index] if sketch_index < len(metas) else {}
        if meta.get("type") == "polyline":
            meta["points"] = points
            metas[sketch_index] = meta

    def _is_closed(self, points):
        if len(points) < 3:
            return False
        return dist(points[0], points[-1]) <= 1e-6

    def _set_point(self, points, idx, new_pt):
        if idx < 0 or idx >= len(points):
            return
        points[idx] = new_pt
        if self._is_closed(points):
            if idx == 0:
                points[-1] = new_pt
            elif idx == len(points) - 1:
                points[0] = new_pt

    def _segment_direction(self, p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            return (1.0, 0.0), 0.0
        return (dx / length, dy / length), length

    def _apply_linear_dimension(self, owner_type, owner_part, sketch_index, segment_index, new_value, anchor="start"):
        sketches, metas, _, constraints = self._owner_collections(owner_type, owner_part)
        if sketch_index < 0 or sketch_index >= len(sketches):
            return
        points = sketches[sketch_index]
        if segment_index < 0 or segment_index + 1 >= len(points):
            return
        meta = metas[sketch_index] if sketch_index < len(metas) else {}
        if str(meta.get("type", "")).lower() == "rectangle":
            rect_dim_type = None
            if segment_index in (0, 2):
                rect_dim_type = "rect_width"
            elif segment_index in (1, 3):
                rect_dim_type = "rect_height"
            if rect_dim_type is not None:
                self._apply_rect_dimension(owner_type, owner_part, sketch_index, new_value, rect_dim_type)
                return
        if meta.get("type") == "line":
            meta = self._normalize_line_meta(meta, fallback_points=points)
            p1 = meta.get("p1", points[segment_index])
            p2 = meta.get("p2", points[segment_index + 1])
        else:
            p1 = points[segment_index]
            p2 = points[segment_index + 1]
        direction, _ = self._segment_direction(p1, p2)
        for con in constraints:
            if con.get("sketch_index") == sketch_index and con.get("segment_index") == segment_index:
                if con.get("type") == "horizontal":
                    direction = (1.0 if direction[0] >= 0 else -1.0, 0.0)
                elif con.get("type") == "vertical":
                    direction = (0.0, 1.0 if direction[1] >= 0 else -1.0)
        anchor = str(anchor).lower()
        move_end = anchor != "end"
        if move_end:
            new_p2 = (p1[0] + direction[0] * new_value, p1[1] + direction[1] * new_value)
            new_p1 = p1
        else:
            new_p1 = (p2[0] - direction[0] * new_value, p2[1] - direction[1] * new_value)
            new_p2 = p2
        if meta.get("type") == "line":
            rebuilt_meta = self._line_meta_from_points(new_p1, new_p2, base_meta=meta)
            self._update_sketch_from_meta(owner_type, owner_part, sketch_index, rebuilt_meta)
        else:
            if move_end:
                self._set_point(points, segment_index + 1, new_p2)
            else:
                self._set_point(points, segment_index, new_p1)
            if meta.get("type") == "polyline":
                meta["points"] = points
                metas[sketch_index] = meta
        self._apply_related_constraints(owner_type, owner_part, sketch_index, segment_index)

    def _apply_rect_dimension(self, owner_type, owner_part, sketch_index, value, dim_type):
        sketches, metas, _, _ = self._owner_collections(owner_type, owner_part)
        if sketch_index < 0 or sketch_index >= len(sketches):
            return

        meta = metas[sketch_index] if sketch_index < len(metas) else {}
        meta = self._normalize_rectangle_meta(meta, fallback_points=sketches[sketch_index])
        if meta.get("type") != "rectangle":
            return

        try:
            new_value = max(1e-9, float(value))
        except (TypeError, ValueError):
            return

        if dim_type == "rect_width":
            width = new_value
            height = float(meta.get("height", 0.0) or 0.0)
        elif dim_type == "rect_height":
            width = float(meta.get("width", 0.0) or 0.0)
            height = new_value
        else:
            return

        origin = self._coerce_xy_tuple(meta.get("origin")) or (0.0, 0.0)
        mode = self._rectangle_mode_key(meta.get("mode"))
        if mode == "center":
            center = (
                origin[0] + float(meta.get("width", 0.0) or 0.0) * 0.5,
                origin[1] + float(meta.get("height", 0.0) or 0.0) * 0.5,
            )
            origin = (center[0] - width * 0.5, center[1] - height * 0.5)

        rebuilt_meta = self._rectangle_meta_from_origin_size(
            origin,
            width,
            height,
            mode=mode,
            base_meta=meta,
        )
        self._update_sketch_from_meta(owner_type, owner_part, sketch_index, rebuilt_meta)

    def _apply_circle_dimension(self, owner_type, owner_part, sketch_index, new_value, dim_type):
        sketches, metas, _, _ = self._owner_collections(owner_type, owner_part)
        if sketch_index < 0 or sketch_index >= len(sketches):
            return
        meta = metas[sketch_index] if sketch_index < len(metas) else {}
        meta_type = str(meta.get("type", "")).lower()
        if meta_type not in ("circle", "arc"):
            return
        if meta_type == "circle":
            meta = self._normalize_circle_meta(meta, fallback_points=sketches[sketch_index])
        center = meta.get("center")
        if center is None:
            return
        radius = new_value
        if dim_type == "diameter":
            radius = new_value / 2.0
        elif dim_type == "arc_length":
            if meta.get("type") != "arc":
                return
            sweep = abs(self._arc_sweep_angle(meta))
            if sweep <= 1e-9:
                return
            radius = new_value / sweep
        radius = max(1e-9, float(radius))
        if meta_type == "circle":
            rebuilt_meta = self._circle_meta_from_center_radius(center, radius, base_meta=meta)
            self._update_sketch_from_meta(owner_type, owner_part, sketch_index, rebuilt_meta)
        else:
            meta["radius"] = radius
            start_angle = meta.get("start_angle", 0.0)
            end_angle = meta.get("end_angle", start_angle + math.pi)
            num_segments = max(2, min(128, int(abs(end_angle - start_angle) * radius / 10)))
            pts = []
            for i in range(num_segments + 1):
                angle = start_angle + (i / num_segments) * (end_angle - start_angle)
                x = center[0] + radius * math.cos(angle)
                y = center[1] + radius * math.sin(angle)
                pts.append((x, y))
            sketches[sketch_index] = pts
            metas[sketch_index] = meta

    def _apply_slot_dimension(self, owner_type, owner_part, sketch_index, new_value, dim_type):
        sketches, metas, _, _ = self._owner_collections(owner_type, owner_part)
        if sketch_index < 0 or sketch_index >= len(sketches):
            return
        meta = metas[sketch_index] if sketch_index < len(metas) else {}
        if meta.get("type") != "slot":
            return
        p1 = meta.get("p1")
        p2 = meta.get("p2")
        width = meta.get("width")
        if p1 is None or p2 is None or width is None:
            return
        if dim_type == "slot_width":
            width = new_value
        else:
            direction, _ = self._segment_direction(p1, p2)
            p2 = (p1[0] + direction[0] * new_value, p1[1] + direction[1] * new_value)
        meta["p2"] = p2
        meta["width"] = width
        sketches[sketch_index] = self._build_slot_vertices(p1, p2, width)
        metas[sketch_index] = meta

    def _apply_polygon_dimension(self, owner_type, owner_part, sketch_index, new_value):
        sketches, metas, _, _ = self._owner_collections(owner_type, owner_part)
        if sketch_index < 0 or sketch_index >= len(sketches):
            return
        meta = metas[sketch_index] if sketch_index < len(metas) else {}
        if meta.get("type") != "polygon":
            return
        center = meta.get("center")
        sides = int(meta.get("sides", 0))
        angle = float(meta.get("angle", 0.0))
        if center is None or sides < 3:
            return
        radius = new_value
        meta["radius"] = radius
        pts = []
        for i in range(sides):
            theta = angle + 2 * math.pi * i / sides
            pts.append((center[0] + radius * math.cos(theta), center[1] + radius * math.sin(theta)))
        pts.append(pts[0])
        sketches[sketch_index] = pts
        metas[sketch_index] = meta

    def _apply_angle_dimension(self, owner_type, owner_part, sketch_index, vertex_index, new_angle_deg):
        sketches, _, _, _ = self._owner_collections(owner_type, owner_part)
        if sketch_index < 0 or sketch_index >= len(sketches):
            return
        points = sketches[sketch_index]
        if vertex_index <= 0 or vertex_index + 1 >= len(points):
            return
        p_prev = points[vertex_index - 1]
        p_vertex = points[vertex_index]
        p_next = points[vertex_index + 1]
        v1 = (p_prev[0] - p_vertex[0], p_prev[1] - p_vertex[1])
        v2 = (p_next[0] - p_vertex[0], p_next[1] - p_vertex[1])
        len2 = math.hypot(v2[0], v2[1])
        if len2 <= 1e-9:
            return
        v1_dir, _ = self._segment_direction(p_vertex, p_prev)
        v2_dir, _ = self._segment_direction(p_vertex, p_next)
        sign = 1.0
        cross = v1_dir[0] * v2_dir[1] - v1_dir[1] * v2_dir[0]
        if cross < 0:
            sign = -1.0
        new_angle = math.radians(new_angle_deg)
        base_angle = math.atan2(v1_dir[1], v1_dir[0])
        target_angle = base_angle + sign * new_angle
        new_dir = (math.cos(target_angle), math.sin(target_angle))
        new_p_next = (p_vertex[0] + new_dir[0] * len2, p_vertex[1] + new_dir[1] * len2)
        self._set_point(points, vertex_index + 1, new_p_next)

    def _apply_related_constraints(self, owner_type, owner_part, sketch_index, segment_index):
        sketches, _, _, constraints = self._owner_collections(owner_type, owner_part)
        metas = self._owner_collections(owner_type, owner_part)[1]
        if sketch_index < 0 or sketch_index >= len(sketches):
            return
        points = sketches[sketch_index]
        if segment_index < 0 or segment_index + 1 >= len(points):
            return
        p1 = points[segment_index]
        p2 = points[segment_index + 1]
        base_dir, base_len = self._segment_direction(p1, p2)
        for con in constraints:
            if con.get("sketch_index") != sketch_index or con.get("segment_index") != segment_index:
                continue
            ctype = con.get("type")
            other_sketch = con.get("other_sketch_index")
            other_seg = con.get("other_segment_index")
            if ctype in ("parallel", "perpendicular", "equal") and other_sketch is not None:
                if other_sketch < 0 or other_sketch >= len(sketches):
                    continue
                other_points = sketches[other_sketch]
                if other_seg < 0 or other_seg + 1 >= len(other_points):
                    continue
                op1 = other_points[other_seg]
                op2 = other_points[other_seg + 1]
                other_dir, other_len = self._segment_direction(op1, op2)
                target_dir = other_dir
                if ctype == "parallel":
                    target_dir = base_dir
                    target_len = other_len
                elif ctype == "perpendicular":
                    target_dir = (-base_dir[1], base_dir[0])
                    target_len = other_len
                else:
                    target_dir = other_dir
                    target_len = base_len
                new_op2 = (op1[0] + target_dir[0] * target_len, op1[1] + target_dir[1] * target_len)
                if other_sketch == sketch_index:
                    self._set_point(other_points, other_seg + 1, new_op2)
                else:
                    other_points[other_seg + 1] = new_op2
                if other_sketch < len(metas) and metas[other_sketch].get("type") == "polyline":
                    metas[other_sketch]["points"] = other_points

    def _dimension_value_from_ui(self, value, dim_type, owner_type, owner_part=None):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        if str(dim_type or "").lower() == "angle":
            return numeric
        return self._owner_value_from_ui(numeric, owner_type, owner_part)

    def edit_dimension(self, dim_id):
        """Edit dimension with enhanced dialog, descriptive UI, and auto-fit (SolidWorks-style)."""
        dim, owner_type, owner_part, _ = self._find_dimension_by_id(dim_id)
        if dim is None:
            return
        self.select_dimension(dim_id, suppress_redraw=True)
        if owner_type == "part" and owner_part is not None and getattr(owner_part, "is_direct_edit", False):
            QMessageBox.information(
                self,
                "Direct Edit Part",
                "This part was modified by a direct 2.5D feature, so sketch dimensions are read-only.",
            )
            return
        if not self._can_use_smart_dimensions():
            QMessageBox.information(
            self,
            "Smart Dimension",
            "Dimensions can only be edited before confirming a part.",
            )
            return
        dim_type = dim.get("dim_type")
        sketch_index = int(dim.get("sketch_index", -1))
        points = self._get_sketch_points(owner_type, owner_part, sketch_index) or []
        meta_list = self._owner_collections(owner_type, owner_part)[1]
        meta = meta_list[sketch_index] if 0 <= sketch_index < len(meta_list) else {}
        current_value = self._dimension_current_value_ui(dim, owner_type, owner_part)
        # Build descriptive title and label based on dimension type
        title = "Edit Dimension"
        display_unit = self._normalize_length_unit()
        value_label = f"New value ({display_unit}):"
        decimals = 3
        if dim_type == "angle":
            title = "Edit Angle Dimension"
            value_label = "New value (degrees):"
            decimals = 2
        elif dim_type == "linear":
            title = "Edit Length Dimension"
            value_label = f"New length ({display_unit}):"
        elif dim_type == "rect_width":
            title = "Edit Width Dimension"
            value_label = f"New width ({display_unit}):"
        elif dim_type == "rect_height":
            title = "Edit Height Dimension"
            value_label = f"New height ({display_unit}):"
        elif dim_type == "diameter":
            title = "Edit Diameter Dimension"
            value_label = f"New diameter ({display_unit}):"
        elif dim_type == "radius":
            title = "Edit Radius Dimension"
            value_label = f"New radius ({display_unit}):"
        elif dim_type == "point_distance":
            title = "Edit Distance Dimension"
            value_label = f"New distance ({display_unit}):"
        elif dim_type in ("slot_length", "slot_width"):
            title = "Edit Slot Dimension"
            value_label = f"New value ({display_unit}):"
        elif dim_type == "polygon_radius":
            title = "Edit Polygon Radius"
            value_label = f"New radius ({display_unit}):"
        elif dim_type == "arc_length":
            title = "Edit Arc Length"
            value_label = f"New length ({display_unit}):"
        new_value, ok = QInputDialog.getDouble(
            self,
            title,
            value_label,
            float(current_value),
            0.0,
            1e9,
            decimals,
        )
        if not ok:
            return
        new_value_units = self._dimension_value_from_ui(new_value, dim_type, owner_type, owner_part)

        self.push_undo_state()
        if dim_type == "linear":
            anchor = dim.get("anchor", "start")
            self._apply_linear_dimension(
                owner_type,
                owner_part,
                sketch_index,
                int(dim.get("segment_index", 0)),
                new_value_units,
                anchor=anchor,
            )
        elif dim_type == "point_distance":
            self._apply_point_distance_dimension(
                owner_type,
                owner_part,
                sketch_index,
                int(dim.get("first_vertex_index", -1)),
                int(dim.get("second_vertex_index", -1)),
                new_value_units,
            )
        elif dim_type in ("rect_width", "rect_height"):
            self._apply_rect_dimension(owner_type, owner_part, sketch_index, new_value_units, dim_type)
        elif dim_type in ("diameter", "radius", "arc_length"):
            self._apply_circle_dimension(owner_type, owner_part, sketch_index, new_value_units, dim_type)
        elif dim_type in ("slot_length", "slot_width"):
            self._apply_slot_dimension(owner_type, owner_part, sketch_index, new_value_units, dim_type)
        elif dim_type == "polygon_radius":
            self._apply_polygon_dimension(owner_type, owner_part, sketch_index, new_value_units)
        elif dim_type == "angle":
            self._apply_angle_dimension(owner_type, owner_part, sketch_index, int(dim.get("vertex_index", 0)), new_value)
        self._finalize_dimension_change(dim, owner_type, owner_part)

    def begin_inline_dimension_edit(self, dim_id):
        try:
            dim_id = int(dim_id)
        except Exception:
            return False
        dim, owner_type, owner_part, _ = self._find_dimension_by_id(dim_id)
        if dim is None:
            return False
        if owner_type == "part" and owner_part is not None and getattr(owner_part, "is_direct_edit", False):
            return False
        window = self.window()
        if window is not None and hasattr(window, "_set_precision_sketch_mode"):
            try:
                window._set_precision_sketch_mode(True, announce=False)
            except Exception:
                pass
        else:
            self.set_dimensions_visible(True)
        self.select_dimension(dim_id)
        self.redraw()
        scene = self.scene()
        if scene is None:
            return False
        for item in scene.items():
            if isinstance(item, DimensionTextItem) and int(getattr(item, "_dim_id", -1)) == dim_id:
                try:
                    self.ensureVisible(item.sceneBoundingRect(), 24, 24)
                except Exception:
                    pass
                def _activate_item(target=item):
                    try:
                        self.setFocus(Qt.OtherFocusReason)
                    except Exception:
                        pass
                    try:
                        self.viewport().setFocus()
                    except Exception:
                        pass
                    try:
                        current_scene = self.scene()
                        if current_scene is not None:
                            current_scene.setFocusItem(target, Qt.OtherFocusReason)
                    except Exception:
                        pass
                    try:
                        target.start_inline_edit()
                    except Exception:
                        pass

                QTimer.singleShot(0, _activate_item)
                return True
        return False

    def update_dimension_from_text(self, dim_id, text):
        dim, owner_type, owner_part, _ = self._find_dimension_by_id(dim_id)
        if dim is None:
            return False
        if not self._can_use_smart_dimensions():
            return False

        self.select_dimension(dim_id, suppress_redraw=True)

        raw = str(text or "").strip()
        if not raw:
            return False

        cleaned = raw.lower()
        cleaned = re.sub(r"[a-zA-Z=]", " ", cleaned)

        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
        if not match:
            return False

        try:
            new_value_ui = float(match.group(0))
        except Exception:
            return False

        dim_type = dim.get("dim_type")
        new_value_units = self._dimension_value_from_ui(
            new_value_ui,
            dim_type,
            owner_type,
            owner_part,
        )

        if new_value_units <= 0:
            return False

        self.push_undo_state()
        sketch_index = int(dim.get("sketch_index", -1))

        if dim_type == "linear":
            self._apply_linear_dimension(
                owner_type, owner_part, sketch_index,
                int(dim.get("segment_index", 0)),
                new_value_units,
                anchor=dim.get("anchor", "start"),
            )

        elif dim_type == "point_distance":
            self._apply_point_distance_dimension(
                owner_type, owner_part, sketch_index,
                int(dim.get("first_vertex_index", -1)),
                int(dim.get("second_vertex_index", -1)),
                new_value_units,
            )

        elif dim_type in ("rect_width", "rect_height"):
            self._apply_rect_dimension(owner_type, owner_part, sketch_index, new_value_units, dim_type)

        elif dim_type in ("diameter", "radius", "arc_length"):
            self._apply_circle_dimension(owner_type, owner_part, sketch_index, new_value_units, dim_type)

        elif dim_type in ("slot_length", "slot_width"):
            self._apply_slot_dimension(owner_type, owner_part, sketch_index, new_value_units, dim_type)

        elif dim_type == "polygon_radius":
            self._apply_polygon_dimension(owner_type, owner_part, sketch_index, new_value_units)

        elif dim_type == "angle":
            self._apply_angle_dimension(
                owner_type, owner_part, sketch_index,
                int(dim.get("vertex_index", 0)),
                new_value_ui,
            )
        else:
            return False

        self._finalize_dimension_change(dim, owner_type, owner_part)
        return True

    def _apply_horizontal_vertical_constraint(self, owner_type, owner_part, sketch_index, segment_index, ctype):
        sketches, _, _, _ = self._owner_collections(owner_type, owner_part)
        if sketch_index < 0 or sketch_index >= len(sketches):
            return
        points = sketches[sketch_index]
        if segment_index < 0 or segment_index + 1 >= len(points):
            return
        p1 = points[segment_index]
        p2 = points[segment_index + 1]
        if ctype == "horizontal":
            new_p2 = (p2[0], p1[1])
        else:
            new_p2 = (p1[0], p2[1])
        self._set_point(points, segment_index + 1, new_p2)

    def _add_constraint(self, owner_type, owner_part, constraint):
        _, _, _, constraints = self._owner_collections(owner_type, owner_part)
        constraints.append(constraint)

    def _handle_constraint_click(self, pt):
        owner_type, owner_part = self._resolve_active_owner(pt)
        if owner_type is None:
            window = self.window()
            if window and hasattr(window, "statusBar"):
                window.statusBar().showMessage("Select or click a part with sketch geometry first.", 4000)
            return
        seg = self._find_nearest_segment(owner_type, owner_part, pt)
        if not seg:
            return
        sketch_index, segment_index = seg

        if self._pending_constraint:
            pending = self._pending_constraint
            if pending["owner_type"] != owner_type or pending.get("owner_id") != getattr(owner_part, "id", None):
                self._pending_constraint = None
                return
            self.push_undo_state()
            constraint = {
                "type": pending["type"],
                "sketch_index": pending["sketch_index"],
                "segment_index": pending["segment_index"],
                "other_sketch_index": sketch_index,
                "other_segment_index": segment_index,
            }
            self._add_constraint(owner_type, owner_part, constraint)
            self._apply_related_constraints(owner_type, owner_part, pending["sketch_index"], pending["segment_index"])
            self._pending_constraint = None
            self.geometryChanged.emit()
            self.redraw()
            return

        options = ["horizontal", "vertical", "parallel", "perpendicular", "equal"]
        choice, ok = QInputDialog.getItem(
            self,
            "Constraint",
            "Constraint type:",
            options,
            0,
            False,
        )
        if not ok:
            return
        choice = choice.lower()
        if choice in ("horizontal", "vertical"):
            self.push_undo_state()
            constraint = {
                "type": choice,
                "sketch_index": sketch_index,
                "segment_index": segment_index,
            }
            self._add_constraint(owner_type, owner_part, constraint)
            self._apply_horizontal_vertical_constraint(owner_type, owner_part, sketch_index, segment_index, choice)
            self.geometryChanged.emit()
            self.redraw()
            return
        self._pending_constraint = {
            "type": choice,
            "owner_type": owner_type,
            "owner_id": getattr(owner_part, "id", None),
            "sketch_index": sketch_index,
            "segment_index": segment_index,
        }
        window = self.window()
        if window and hasattr(window, "statusBar"):
            window.statusBar().showMessage("Select a second segment to complete the constraint.", 5000)

    def _handle_dimension_click(self, pt):
        owner_type, owner_part = self._resolve_active_owner(pt)
        if owner_type is None:
            window = self.window()
            if window and hasattr(window, "statusBar"):
                window.statusBar().showMessage("Select or click a part with sketch geometry first.", 4000)
            return
        sketches, metas, dimensions, _ = self._owner_collections(owner_type, owner_part)

        if self._pending_dimension:
            pending = dict(self._pending_dimension)
            if pending.get("mode") == "point_pair":
                first_vertex = pending.get("first_vertex")
                second_vertex = self._find_nearest_vertex(owner_type, owner_part, pt)
                if not first_vertex or not second_vertex:
                    return
                first_sketch, first_index = first_vertex
                second_sketch, second_index = second_vertex
                if int(first_sketch) != int(second_sketch) or int(first_index) == int(second_index):
                    window = self.window()
                    if window and hasattr(window, "statusBar"):
                        window.statusBar().showMessage(
                            "Pick two different points on the same sketch to create a dimension.",
                            4000,
                        )
                    return
                points = sketches[first_sketch]
                p1 = points[int(first_index)]
                p2 = points[int(second_index)]
                direction, length = self._segment_direction(p1, p2)
                if length <= 1e-9:
                    self._pending_dimension = None
                    return
                nx, ny = -direction[1], direction[0]
                offset_mag = self._dim_offset
                if abs(int(first_index) - int(second_index)) == 1:
                    segment_index = min(int(first_index), int(second_index))
                    dim = self._build_dimension_record(
                        owner_type,
                        owner_part,
                        "linear",
                        int(first_sketch),
                        segment_index=int(segment_index),
                        anchor="start" if int(first_index) <= int(second_index) else "end",
                        offset=(nx * offset_mag, ny * offset_mag),
                    )
                else:
                    dim = self._build_dimension_record(
                        owner_type,
                        owner_part,
                        "point_distance",
                        int(first_sketch),
                        first_vertex_index=int(first_index),
                        second_vertex_index=int(second_index),
                        offset=(nx * offset_mag, ny * offset_mag),
                    )
                self.push_undo_state()
                dimensions.append(dim)
                self._pending_dimension = None
                self._finalize_dimension_change(dim, owner_type, owner_part)
                return

            anchor = pending["anchor"]
            offset = (pt[0] - anchor[0], pt[1] - anchor[1])
            if abs(offset[0]) < 1e-6 and abs(offset[1]) < 1e-6:
                offset = None
            dim = pending["dim"]
            if offset is not None:
                dim["offset"] = offset
            self.push_undo_state()
            dimensions.append(dim)
            self._pending_dimension = None
            self._finalize_dimension_change(dim, owner_type, owner_part)
            return

        vertex = self._find_nearest_vertex(owner_type, owner_part, pt)
        if vertex:
            sketch_index, vertex_index = vertex
            self._pending_dimension = {
                "mode": "point_pair",
                "owner_type": owner_type,
                "owner_id": getattr(owner_part, "id", None),
                "first_vertex": (int(sketch_index), int(vertex_index)),
            }
            window = self.window()
            if window and hasattr(window, "statusBar"):
                window.statusBar().showMessage("Click the second point for the dimension.", 5000)
            return

        seg = self._find_nearest_segment(owner_type, owner_part, pt)
        if not seg:
            return
        sketch_index, segment_index = seg
        meta = metas[sketch_index] if sketch_index < len(metas) else {}
        meta_type = str(meta.get("type", "polyline")).lower()
        dim_type = "linear"
        if meta_type == "rectangle":
            if segment_index in (0, 2):
                dim_type = "rect_width"
            elif segment_index in (1, 3):
                dim_type = "rect_height"
        elif meta_type == "circle":
            options = ["diameter", "radius"]
            choice, ok = QInputDialog.getItem(
                self, "Dimension", "Circle dimension:", options, 0, False
            )
            if not ok:
                return
            dim_type = choice
        elif meta_type == "slot":
            options = ["Length", "Width"]
            choice, ok = QInputDialog.getItem(self, "Dimension", "Slot dimension:", options, 0, False)
            if not ok:
                return
            dim_type = "slot_length" if choice == "Length" else "slot_width"
        elif meta_type == "arc":
            options = ["Radius", "Arc Length"]
            choice, ok = QInputDialog.getItem(self, "Dimension", "Arc dimension:", options, 0, False)
            if not ok:
                return
            dim_type = "radius" if choice == "Radius" else "arc_length"
        dim = self._build_dimension_record(
            owner_type,
            owner_part,
            dim_type,
            sketch_index,
            segment_index=segment_index,
        )
        points = sketches[sketch_index]
        if dim_type == "linear" and segment_index + 1 < len(points):
            p1 = points[segment_index]
            p2 = points[segment_index + 1]
            dim["anchor"] = "start" if dist(pt, p1) <= dist(pt, p2) else "end"
        anchor = points[segment_index]
        if segment_index + 1 < len(points):
            mid = (
                (points[segment_index][0] + points[segment_index + 1][0]) / 2.0,
                (points[segment_index][1] + points[segment_index + 1][1]) / 2.0,
            )
            anchor = mid
        self._pending_dimension = {"dim": dim, "anchor": anchor}
        window = self.window()
        if window and hasattr(window, "statusBar"):
            window.statusBar().showMessage("Click to place the dimension.", 5000)

    def _draw_bc_marker(self, bc):
        shape, geom = self._resolve_attr_marker_geometry(bc)
        if geom is None:
            return
        is_focused = self._is_panel_attr_focus("bc", bc)
        if is_focused and bc.get("part_id") is not None and bc.get("coords") in (None, "", []):
            try:
                target_part_id = int(bc.get("part_id"))
            except Exception:
                target_part_id = None
            if target_part_id is not None:
                target_part = next((p for p in self.parts if getattr(p, "id", None) == target_part_id), None)
                boundary_geom = getattr(getattr(target_part, "geometry", None), "boundary", None)
                if boundary_geom is not None:
                    edge_pen = QPen(QColor(255, 140, 30), 3.6)
                    for line in self._iter_line_geometries(boundary_geom):
                        try:
                            coords = list(line.coords)
                        except Exception:
                            coords = []
                        if len(coords) < 2:
                            continue
                        path = QPainterPath()
                        path.moveTo(float(coords[0][0]), float(coords[0][1]))
                        for pt in coords[1:]:
                            path.lineTo(float(pt[0]), float(pt[1]))
                        self.scene().addPath(path, edge_pen, QBrush(Qt.NoBrush))
        if shape == "point":
            center = geom
        else:
            p1, p2 = geom
            center = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
        if is_focused:
            hpen = QPen(QColor(255, 140, 30), 4.0)
            if shape == "point":
                self.scene().addEllipse(
                    center[0] - 10,
                    center[1] - 10,
                    20,
                    20,
                    hpen,
                    QBrush(QColor(255, 210, 120, 60)),
                )
            else:
                p1, p2 = geom
                self.scene().addLine(p1[0], p1[1], p2[0], p2[1], hpen)
                self.scene().addEllipse(
                    center[0] - 8,
                    center[1] - 8,
                    16,
                    16,
                    hpen,
                    QBrush(QColor(255, 210, 120, 40)),
                )
        if 'velocity' in bc['type']:
            self.scene().addRect(center[0]-4, center[1]-4, 8, 8, QPen(Qt.black), QBrush(Qt.blue))
        elif 'xy' in bc['type']:
            poly = QPolygonF([QPointF(center[0], center[1]), QPointF(center[0]-6, center[1]+10), QPointF(center[0]+6, center[1]+10)])
            self.scene().addPolygon(poly, QPen(Qt.black), QBrush(Qt.red))
        else:
            self.scene().addRect(center[0]-5, center[1]-5, 10, 10, QPen(Qt.black), QBrush(QColor(255,100,100)))

    def _draw_load_marker(self, ld):
        is_focused = self._is_panel_attr_focus("load", ld)
        if ld['type'] == 'moment':
            shape, geom = self._resolve_attr_marker_geometry(ld)
            if geom is None:
                return
            if shape == "point":
                center = geom
            else:
                p1, p2 = geom
                center = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
            if is_focused:
                self.scene().addEllipse(
                    center[0]-11,
                    center[1]-11,
                    22,
                    22,
                    QPen(QColor(255, 165, 40), 2.5),
                    QBrush(QColor(255, 210, 120, 50)),
                )
            self.scene().addEllipse(center[0]-6, center[1]-6, 12, 12, QPen(Qt.darkMagenta), QBrush(QColor(200,160,255)))
            return
        try:
            fx = float(ld.get("fx", 0.0))
        except Exception:
            fx = 0.0
        try:
            fy = float(ld.get("fy", 0.0))
        except Exception:
            fy = 0.0
        try:
            fz = float(ld.get("fz", 0.0))
        except Exception:
            fz = 0.0
        axis = str(ld.get("axis", "")).strip().lower()

        eps = 1e-12
        vx, vy = fx, fy
        if abs(vx) <= eps and abs(vy) <= eps:
            if axis == "y":
                vy = -1.0 if fy < 0.0 else 1.0
            elif axis == "z":
                # In 2D canvas, represent out-of-plane force with a vertical arrow.
                vy = -1.0 if fz < 0.0 else 1.0
            else:
                vx = -1.0 if fx < 0.0 else 1.0

        direction_norm = math.hypot(vx, vy)
        if direction_norm <= eps:
            vx, vy = 1.0, 0.0
            direction_norm = 1.0
        ux, uy = vx / direction_norm, vy / direction_norm

        value_mag = max(abs(fx), abs(fy), abs(fz))
        arrow_len = 30.0 if value_mag > eps else 22.0
        marker_pen = QPen(Qt.darkGreen, 2)
        if value_mag <= eps:
            marker_pen.setStyle(Qt.DashLine)

        def draw_arrow_at(px, py):
            tip_x = px + arrow_len * ux
            tip_y = py + arrow_len * uy
            if is_focused:
                self.scene().addLine(px, py, tip_x, tip_y, QPen(QColor(255, 165, 40), 4))
            self.scene().addLine(px, py, tip_x, tip_y, marker_pen)

            # Draw arrowhead so loads are clearly visible as force symbols.
            head_len = max(6.0, arrow_len * 0.22)
            head_ang = math.radians(28.0)
            back_x, back_y = -ux, -uy
            left_x = tip_x + head_len * (back_x * math.cos(head_ang) - back_y * math.sin(head_ang))
            left_y = tip_y + head_len * (back_x * math.sin(head_ang) + back_y * math.cos(head_ang))
            right_x = tip_x + head_len * (back_x * math.cos(-head_ang) - back_y * math.sin(-head_ang))
            right_y = tip_y + head_len * (back_x * math.sin(-head_ang) + back_y * math.cos(-head_ang))
            self.scene().addLine(tip_x, tip_y, left_x, left_y, marker_pen)
            self.scene().addLine(tip_x, tip_y, right_x, right_y, marker_pen)

        shape, geom = self._resolve_attr_marker_geometry(ld)
        if geom is None:
            return
        if is_focused and shape == "point":
            self.scene().addEllipse(
                geom[0]-10,
                geom[1]-10,
                20,
                20,
                QPen(QColor(255, 165, 40), 2.5),
                QBrush(QColor(255, 210, 120, 50)),
            )
        elif is_focused and shape != "point":
            p1, p2 = geom
            self.scene().addLine(p1[0], p1[1], p2[0], p2[1], QPen(QColor(255, 165, 40), 3))
        if shape == "point":
            draw_arrow_at(geom[0], geom[1])
        else:
            p1, p2 = geom; length = dist(p1, p2); steps = max(3, int(length/20))
            dx = (p2[0]-p1[0])/steps; dy = (p2[1]-p1[1])/steps
            for i in range(steps+1): draw_arrow_at(p1[0]+i*dx, p1[1]+i*dy)

    def _resolve_attr_marker_geometry(self, entry):
        coords = entry.get("coords")
        if isinstance(coords, np.ndarray):
            coords = coords.tolist()
        if isinstance(coords, (list, tuple)):
            if (
                len(coords) >= 2
                and isinstance(coords[0], (int, float))
                and isinstance(coords[1], (int, float))
            ):
                return "point", (float(coords[0]), float(coords[1]))
            if (
                len(coords) >= 2
                and isinstance(coords[0], (list, tuple, np.ndarray))
                and isinstance(coords[1], (list, tuple, np.ndarray))
                and len(coords[0]) >= 2
                and len(coords[1]) >= 2
            ):
                p1 = (float(coords[0][0]), float(coords[0][1]))
                p2 = (float(coords[1][0]), float(coords[1][1]))
                return "segment", (p1, p2)

        part_id = entry.get("part_id")
        if part_id is not None:
            try:
                pid = int(part_id)
            except Exception:
                pid = None
            if pid is not None:
                part = next((p for p in self.parts if p.id == pid), None)
                if part and part.geometry is not None:
                    try:
                        center = part.geometry.centroid
                        return "point", (float(center.x), float(center.y))
                    except Exception:
                        pass

        ids = entry.get("ids")
        if ids is None or len(ids) == 0 or self.global_nodes is None:
            return None, None
        try:
            nodes = np.asarray(self.global_nodes, dtype=float)
        except Exception:
            return None, None
        if nodes.ndim != 2 or nodes.shape[1] < 2 or len(nodes) == 0:
            return None, None

        valid_ids = []
        for nid in ids:
            try:
                idx = int(nid)
            except Exception:
                continue
            if 0 <= idx < len(nodes):
                valid_ids.append(idx)
        if not valid_ids:
            return None, None
        if len(valid_ids) == 2:
            p1 = tuple(float(v) for v in nodes[valid_ids[0], :2])
            p2 = tuple(float(v) for v in nodes[valid_ids[1], :2])
            return "segment", (p1, p2)
        pts = nodes[np.asarray(valid_ids, dtype=int), :2]
        center = np.mean(pts, axis=0)
        return "point", (float(center[0]), float(center[1]))

    def show_displacement_vectors(self, initial_pos, displacement, scalar_values=None, field_label=""):
        if initial_pos is None or displacement is None:
            return
        self.animation_timer.stop()
        self.is_visualization_mode = False
        self._animation_frame_loading = False
        self.display_mode = "results"
        self._result_field_label = str(field_label or "")
        if scalar_values is None:
            self._result_scalar_values = None
            self._clear_results_legend()
        else:
            self._result_scalar_values = np.asarray(scalar_values, dtype=float).reshape(-1)
            self._update_results_legend_from_field_packet(
                {
                    "key": "disp_mag",
                    "label": str(field_label or "Displacement magnitude"),
                    "domain": "node",
                    "values": self._result_scalar_values,
                }
            )
        self._displacement_vectors = [
            (float(x), float(y), float(dx), float(dy))
            for (x, y), (dx, dy) in zip(initial_pos, displacement)
        ]
        self.redraw()

    def _add_info_text(self, text):
        rect = self.sceneRect()
        item = self.scene().addText(text)
        item.setDefaultTextColor(QColor(60, 60, 60))
        item.setPos(rect.left() + 20, rect.top() + 20)
        if self._y_axis_up:
            item.setTransform(QTransform(1, 0, 0, -1, 0, 0))

    def _get_part_at_point(self, pt, include_void=False):
        if not pt:
            return None
        if self.display_mode == "mesh":
            mesh_part = self._get_part_from_mesh_triangle_hit(pt, include_void=include_void)
            if mesh_part is not None:
                return mesh_part
        point = Point(pt)
        edge_tol = 0.0
        try:
            edge_tol = float(self._scene_units_for_pixels(6.0))
        except Exception:
            edge_tol = 0.0
        for part in reversed(self.parts):
            geom = getattr(part, "geometry", None)
            if not geom:
                continue
            try:
                hit = bool(geom.covers(point))
            except Exception:
                try:
                    hit = bool(geom.contains(point))
                except Exception:
                    hit = False
            if not hit and edge_tol > 0.0:
                try:
                    hit = float(geom.boundary.distance(point)) <= edge_tol
                except Exception:
                    hit = False
            if hit:
                if part.is_void and not include_void:
                    continue
                return part
        return None

    def _get_part_from_mesh_triangle_hit(self, pt, include_void=False):
        try:
            nodes = np.asarray(self.global_nodes, dtype=float)
            elements = np.asarray(self.global_elements, dtype=int)
        except Exception:
            return None
        if nodes.ndim != 2 or len(nodes) == 0 or elements.ndim != 2 or len(elements) == 0:
            return None
        element_to_part = {}
        for item in self.element_part_map or []:
            if not isinstance(item, dict):
                continue
            try:
                element_to_part[int(item.get("element_idx"))] = int(item.get("part_id"))
            except Exception:
                continue
        if not element_to_part:
            return None
        hit_point = Point(pt)
        for elem_idx in range(len(elements) - 1, -1, -1):
            part_id = element_to_part.get(elem_idx)
            if part_id is None:
                continue
            try:
                tri = elements[elem_idx]
                tri_poly = Polygon(nodes[tri[:3]])
            except Exception:
                continue
            try:
                hit = bool(tri_poly.covers(hit_point))
            except Exception:
                hit = False
            if not hit:
                continue
            part = next((p for p in self.parts if getattr(p, "id", None) == part_id), None)
            if part is None:
                continue
            if getattr(part, "is_void", False) and not include_void:
                continue
            return part
        return None
    
    def keyPressEvent(self, event):
        editor_item = self._active_dimension_editor_item()
        if editor_item is not None:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                editor_item.commit_inline_edit()
                event.accept()
                return
            if event.key() == Qt.Key_Escape:
                editor_item.cancel_inline_edit()
                event.accept()
                return

        if self._zone_draw_active:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self._finish_zone_draw()
                event.accept()
                return
            if event.key() == Qt.Key_Escape:
                self.cancel_zone_draw()
                event.accept()
                return

        if self._edge_seed_pick_mode is not None:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self._finish_edge_seed_pick()
                event.accept()
                return
            if event.key() == Qt.Key_Escape:
                self.cancel_edge_seed_pick()
                event.accept()
                return

        if self._vertex_seed_pick_active:
            if event.key() == Qt.Key_Escape:
                self.cancel_vertex_seed_pick()
                event.accept()
                return

        if self._partition_pick_active:
            if event.key() == Qt.Key_Escape:
                self.cancel_partition_pick()
                event.accept()
                return

        if event.key() == Qt.Key_Escape:
            if self._pending_attr_edit:
                self._pending_attr_edit = None
                window = self.window()
                if window and hasattr(window, "statusBar"):
                    window.statusBar().showMessage("Move canceled.", 3000)
            if self._pending_dimension:
                self._pending_dimension = None
            if self._pending_constraint:
                self._pending_constraint = None
            self.mode = "idle"
            self.current.clear()
            self._snap_indicator = None
            self._clear_preview()
            self._clear_zoom_window()
            if self._paint_active:
                self._paint_active = False
                self._paint_points = []
                self._clear_paint_preview()
            self.set_tool("select")
            return

        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.active_module == "Part" and self.sketches:
                self.confirm_solid()
                return

        if event.key() == Qt.Key_Space:
            self._space_pressed = True
            if not self._panning:
                self.setCursor(Qt.OpenHandCursor)
            return

        if event.key() == Qt.Key_Z:
            self.set_tool("zoom_window")
            return

        if event.key() == Qt.Key_Delete:
            if self.selected_part_id is not None:
                part = next((p for p in self.parts if p.id == self.selected_part_id), None)
                if part:
                    self.delete_part(part, confirm=True)
                    return

        if event.key() == Qt.Key_M and self.active_module == "Property":
            if self.hover_item:
                pt = self.hover_item[1] if self.hover_item[0] == 'vertex' else None
                part = self._get_part_at_point(pt)
                if part:
                    self.window().properties_panel.assembly_tab.assign_material_direct(part)
                    return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space:
            self._space_pressed = False
            if not self._panning:
                self.setCursor(Qt.ArrowCursor)
            return
        super().keyReleaseEvent(event)

    def wheelEvent(self, event):
        pixel_delta = event.pixelDelta() if hasattr(event, "pixelDelta") else None
        if (
            pixel_delta is not None
            and not pixel_delta.isNull()
            and not (event.modifiers() & Qt.ControlModifier)
        ):
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(pixel_delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(pixel_delta.y())
            )
            event.accept()
            return
        delta = event.angleDelta().y()
        if delta == 0 and pixel_delta is not None and not pixel_delta.isNull():
            delta = int(pixel_delta.y())
        if delta == 0:
            return
        factor = math.pow(1.15, float(delta) / 120.0)
        new_zoom = getattr(self, "_zoom", 1.0) * factor
        if 0.1 <= new_zoom <= 10.0:
            self._zoom = new_zoom
            # Force anchor-under-mouse at scale time so wheel-zoom always
            # zooms toward the cursor location, even if a previous operation
            # (fit-view, frame-selection) left a different anchor.
            old_anchor = self.transformationAnchor()
            self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            self.scale(factor, factor)
            self.setTransformationAnchor(old_anchor)
            try:
                self.zoomChanged.emit(float(new_zoom))
            except Exception:
                pass
        event.accept()

    def _apply_fit_rect(self, rect):
        if rect is None or rect.width() <= 0 or rect.height() <= 0:
            return False
        old_transform_anchor = self.transformationAnchor()
        old_resize_anchor = self.resizeAnchor()
        try:
            fit_rect = QRectF(rect)
            self._sync_scene_rect_to_model(fit_rect, force=True)
            self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
            self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
            self.resetTransform()
            self.fitInView(fit_rect, Qt.KeepAspectRatio)
            if self._y_axis_up:
                self.scale(1, -1)
            self.centerOn(fit_rect.center())
            self._zoom = 1.0
            return True
        finally:
            self.setTransformationAnchor(old_transform_anchor)
            self.setResizeAnchor(old_resize_anchor)

    def _append_geometry_bounds(self, bounds_list, geom):
        if geom is None or getattr(geom, "is_empty", True):
            return
        try:
            minx, miny, maxx, maxy = geom.bounds
        except Exception:
            return
        values = (float(minx), float(miny), float(maxx), float(maxy))
        if not all(math.isfinite(v) for v in values):
            return
        bounds_list.append(values)

    def _append_points_bounds(self, bounds_list, points):
        if points is None:
            return
        try:
            pts = np.asarray(points, dtype=float)
        except Exception:
            return
        if pts.size == 0 or pts.ndim != 2 or pts.shape[1] < 2:
            return
        xs = pts[:, 0]
        ys = pts[:, 1]
        mask = np.isfinite(xs) & np.isfinite(ys)
        if not np.any(mask):
            return
        bounds_list.append(
            (
                float(np.min(xs[mask])),
                float(np.min(ys[mask])),
                float(np.max(xs[mask])),
                float(np.max(ys[mask])),
            )
        )

    def _fit_rect_from_bounds(self, bounds_list):
        if not bounds_list:
            return None
        minx = min(b[0] for b in bounds_list)
        miny = min(b[1] for b in bounds_list)
        maxx = max(b[2] for b in bounds_list)
        maxy = max(b[3] for b in bounds_list)
        if not all(math.isfinite(v) for v in (minx, miny, maxx, maxy)):
            return None

        width = maxx - minx
        height = maxy - miny
        span = max(width, height)
        if span <= 0.0:
            span = 20.0
        if width <= 0.0:
            half = 0.5 * span
            minx -= half
            maxx += half
        if height <= 0.0:
            half = 0.5 * span
            miny -= half
            maxy += half
        pad = max(0.06 * span, 8.0)
        return QRectF(minx - pad, miny - pad, (maxx - minx) + 2.0 * pad, (maxy - miny) + 2.0 * pad)

    def _model_fit_rect(self):
        bounds = []

        if self.display_mode == "results":
            packet = self._current_frame_packet or {}
            self._append_points_bounds(bounds, packet.get("raw_positions"))
            self._append_points_bounds(bounds, packet.get("display_positions"))

        if self.display_mode == "mesh":
            self._append_points_bounds(bounds, self.global_nodes)
        elif self.display_mode == "mesh_3d":
            try:
                proj = self._project_3d_points(np.asarray(self.global_nodes_3d, dtype=float))
            except Exception:
                proj = None
            self._append_points_bounds(bounds, proj)

        for part in getattr(self, "parts", []) or []:
            self._append_geometry_bounds(bounds, getattr(part, "geometry", None))

        self._append_geometry_bounds(bounds, getattr(self, "solid_geometry", None))

        for idx, sketch in enumerate(getattr(self, "sketches", []) or []):
            if self._is_sketch_visible("sketch", None, idx):
                self._append_points_bounds(bounds, sketch)
        self._append_points_bounds(bounds, getattr(self, "current", None))

        return self._fit_rect_from_bounds(bounds)

    def fit_view(self):
        try:
            rect = self._model_fit_rect()
            if rect is None:
                rect = self._scene_rect_for_model()
            self._apply_fit_rect(rect)
        except Exception:
            pass

    def fit_selection(self):
        target_geom = None
        if self.selected_part_id is not None:
            part = next((p for p in self.parts if p.id == self.selected_part_id), None)
            if part and part.geometry and not part.geometry.is_empty:
                target_geom = part.geometry

        if target_geom is None:
            if self.display_mode == "mesh" and self.global_nodes is not None and len(self.global_nodes) > 0:
                if self._fit_points(self.global_nodes):
                    return
            if (
                self.display_mode == "mesh_3d"
                and self.global_nodes_3d is not None
                and len(self.global_nodes_3d) > 0
            ):
                try:
                    proj = self._project_3d_points(np.asarray(self.global_nodes_3d))
                except Exception:
                    proj = None
                if proj is not None and self._fit_points(proj):
                    return

        if target_geom is None and self.solid_geometry and not self.solid_geometry.is_empty:
            target_geom = self.solid_geometry

        if target_geom is None and self.sketches:
            points = []
            for sketch in self.sketches:
                points.extend(sketch)
            if self.current:
                points.extend(self.current)
            if points is not None and len(points) > 0 and self._fit_points(points):
                return

        if target_geom is None:
            self.fit_view()
            return

        minx, miny, maxx, maxy = target_geom.bounds
        rect = QRectF(minx, miny, maxx - minx, maxy - miny)
        if rect.width() <= 0 or rect.height() <= 0:
            self.fit_view()
            return

        pad = max(rect.width(), rect.height()) * 0.05
        if pad <= 0:
            pad = 10.0
        rect = rect.adjusted(-pad, -pad, pad, pad)
        return self._apply_fit_rect(rect)

    def _fit_points(self, points):
        try:
            pts = np.asarray(points, dtype=float)
        except Exception:
            return False
        if pts.size == 0 or pts.ndim != 2 or pts.shape[1] < 2:
            return False
        xs = pts[:, 0]
        ys = pts[:, 1]
        mask = np.isfinite(xs) & np.isfinite(ys)
        if not np.any(mask):
            return False
        minx = float(np.min(xs[mask]))
        maxx = float(np.max(xs[mask]))
        miny = float(np.min(ys[mask]))
        maxy = float(np.max(ys[mask]))
        rect = QRectF(minx, miny, maxx - minx, maxy - miny)
        if rect.width() <= 0 or rect.height() <= 0:
            return False
        pad = max(rect.width(), rect.height()) * 0.05
        if pad <= 0:
            pad = 10.0
        rect = rect.adjusted(-pad, -pad, pad, pad)
        return self._apply_fit_rect(rect)

    def fit_to_origin(self):
        rect = QRectF(-10, -10, 20, 20)
        self._apply_fit_rect(rect)

    def dropEvent(self, event):
        if not self.can_assign_material():
            QMessageBox.warning(self, "Locked", "Switch to Materials stage first.")
            return

        try:
            mat_serial = int(event.mimeData().text())
        except ValueError:
            return

        pos = self.mapToScene(event.position().toPoint())
        part = self._get_part_at_point((pos.x(), pos.y()))

        if not part:
            return

        self.assign_material_to_part(part, mat_serial)

    def _preferred_dimension_type_for_geometry(self, geom):
        if not isinstance(geom, dict):
            return "linear"
        meta = geom.get("meta") or {}
        points = geom.get("points") or []
        meta_type = str(meta.get("type", "polyline")).lower()
        segment_index = int(geom.get("segment_index", 0))

        if meta_type == "rectangle":
            p1, p2 = self._segment_points(points, segment_index)
            if p1 is not None and p2 is not None and abs(p2[0] - p1[0]) >= abs(p2[1] - p1[1]):
                return "rect_width"
            return "rect_height"
        if meta_type == "circle":
            return "diameter"
        if meta_type == "arc":
            return "radius"
        if meta_type == "slot":
            lengths = []
            for idx in range(max(0, len(points) - 1)):
                try:
                    lengths.append(float(dist(points[idx], points[idx + 1])))
                except Exception:
                    lengths.append(0.0)
            if lengths:
                current_len = lengths[min(segment_index, len(lengths) - 1)]
                min_len = min(lengths)
                max_len = max(lengths)
                if max_len > 0.0 and current_len >= ((min_len + max_len) * 0.5):
                    return "slot_length"
            return "slot_width"
        if meta_type == "polygon":
            return "linear"
        return "linear"

    def _dimension_types_for_geometry(self, geom):
        preferred = self._preferred_dimension_type_for_geometry(geom)
        meta = geom.get("meta") if isinstance(geom, dict) else {}
        meta_type = str((meta or {}).get("type", "polyline")).lower()
        if meta_type == "circle":
            alternate = "radius" if preferred == "diameter" else "diameter"
            return [preferred, alternate]
        if meta_type == "arc":
            alternate = "arc_length" if preferred == "radius" else "radius"
            return [preferred, alternate]
        if meta_type == "slot":
            alternate = "slot_width" if preferred == "slot_length" else "slot_length"
            return [preferred, alternate]
        if meta_type == "polygon":
            return [preferred, "polygon_radius"]
        return [preferred]

    def _find_geometry_at_scene_pos(self, scene_pos, tol=None):
        if scene_pos is None:
            return None
        owner_type, owner_part = self._active_owner()
        if owner_type is None:
            return None
        try:
            click_pt = (float(scene_pos.x()), float(scene_pos.y()))
        except Exception:
            return None
        if tol is None:
            try:
                tol = max(float(SNAP_TOL), float(self._scene_units_for_pixels(8.0)))
            except Exception:
                tol = float(SNAP_TOL)
        result = self._find_nearest_segment(owner_type, owner_part, click_pt, tol=tol)
        if result is None:
            return None
        sketch_index, segment_index = result
        sketches, metas, _, _ = self._owner_collections(owner_type, owner_part)
        if not (0 <= sketch_index < len(sketches)):
            return None
        points = sketches[sketch_index]
        meta = metas[sketch_index] if 0 <= sketch_index < len(metas) else {}
        return {
            "owner_type": owner_type,
            "owner_part": owner_part,
            "sketch_index": int(sketch_index),
            "segment_index": int(segment_index),
            "points": points,
            "meta": meta,
            "click_point": click_pt,
        }

    def _find_dimension_for_geometry(self, geom):
        if not isinstance(geom, dict):
            return None
        owner_type = geom.get("owner_type")
        owner_part = geom.get("owner_part")
        sketch_index = int(geom.get("sketch_index", -1))
        segment_index = int(geom.get("segment_index", -1))
        if owner_type is None or sketch_index < 0 or segment_index < 0:
            return None
        _, _, dimensions, _ = self._owner_collections(owner_type, owner_part)
        candidate_types = [str(dim_type).lower() for dim_type in self._dimension_types_for_geometry(geom)]

        for dim in dimensions:
            try:
                dim_sketch_index = int(dim.get("sketch_index", -1))
                dim_segment_index = int(dim.get("segment_index", -1))
            except Exception:
                continue
            dim_type = str(dim.get("dim_type", "")).lower()
            if dim_sketch_index != sketch_index or dim_segment_index != segment_index:
                continue
            if dim_type in candidate_types:
                return dim

        reusable_types = {
            "rect_width",
            "rect_height",
            "diameter",
            "radius",
            "slot_length",
            "slot_width",
            "polygon_radius",
            "arc_length",
        }
        for dim in dimensions:
            try:
                dim_sketch_index = int(dim.get("sketch_index", -1))
            except Exception:
                continue
            dim_type = str(dim.get("dim_type", "")).lower()
            if dim_sketch_index != sketch_index or dim_type not in reusable_types:
                continue
            if dim_type in candidate_types:
                return dim
        return None

    def _create_dimension_for_geometry(self, geom):
        if not isinstance(geom, dict):
            return None
        owner_type = geom.get("owner_type")
        owner_part = geom.get("owner_part")
        sketch_index = int(geom.get("sketch_index", -1))
        segment_index = int(geom.get("segment_index", -1))
        if owner_type is None or sketch_index < 0 or segment_index < 0:
            return None
        dim_type = str(self._dimension_types_for_geometry(geom)[0]).lower()
        dim = self._build_dimension_record(
            owner_type,
            owner_part,
            dim_type,
            sketch_index,
            segment_index=segment_index,
        )
        points = geom.get("points") or self._get_sketch_points(owner_type, owner_part, sketch_index) or []
        click_pt = geom.get("click_point")
        if dim_type == "linear" and click_pt is not None and segment_index + 1 < len(points):
            p1 = points[segment_index]
            p2 = points[segment_index + 1]
            dim["anchor"] = "start" if dist(click_pt, p1) <= dist(click_pt, p2) else "end"
        self.push_undo_state()
        _, _, dimensions, _ = self._owner_collections(owner_type, owner_part)
        dimensions.append(dim)
        self._finalize_dimension_change(dim, owner_type, owner_part)
        return dim.get("id")

    def _open_dimension_for_geometry(self, geom):
        dim = self._find_dimension_for_geometry(geom)
    
        if dim is None:
            dim_id = self._create_dimension_for_geometry(geom)
        else:
            dim_id = dim.get("id")
    
        self.edit_dimension(dim_id)
