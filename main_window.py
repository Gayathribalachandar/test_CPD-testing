import copy
import hashlib
import json
import logging
import math
import os
import random
import signal
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import Point, Polygon, box

def _maybe_reexec_with_venv():
    if os.environ.get("CPD_SKIP_VENV_REEXEC") == "1":
        return
    root = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(root, ".venv", "bin", "python"),
        os.path.join(root, ".venv", "Scripts", "python.exe"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        if os.path.abspath(path) == os.path.abspath(sys.executable):
            return
        env = os.environ.copy()
        env["CPD_SKIP_VENV_REEXEC"] = "1"
        os.execvpe(path, [path] + sys.argv, env)


_maybe_reexec_with_venv()

from PySide6.QtCore import Qt, QSize, QTimer, QSettings, QUrl, QPoint, QPointF, QRect, QRectF, QEvent, QThread, Signal, QEasingCurve, QPropertyAnimation
from PySide6.QtGui import QAction, QActionGroup, QDesktopServices, QPalette, QColor, QPixmap, QPainter, QIcon, QKeySequence, QBrush, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QCompleter,
    QDockWidget,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGraphicsScene,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QDoubleSpinBox,
    QProgressDialog,
    QToolBar,
    QVBoxLayout,
    QCheckBox,
    QPlainTextEdit,
    QToolButton,
    QGroupBox,
    QRadioButton,
    QButtonGroup,
    QSpinBox,
    QWidget,
    QSplitter,
    QStackedWidget,
    QScrollArea,
    QSizePolicy,
    QLayout,
    QLayoutItem,
    QSlider,
    QColorDialog,
    QGraphicsOpacityEffect,
)

from app_config import (
    __version__ as APP_BUILD_VERSION,
    SCENE_H,
    SCENE_W,
    VIEW_H,
    VIEW_W,
    START_MAXIMIZED,
    UPDATE_CHECK_ON_STARTUP,
    UPDATE_REPO,
    UPDATE_REQUEST_TIMEOUT_SEC,
    WORKSPACE_DIR_NAME,
    get_autosave_dir,
    get_recent_projects_file,
    configure_app_logging,
    get_project_root,
    get_workspace_dir,
    get_workspace_path,
)
from importers import ImporterError, ImporterUnavailable, get_import_filter, import_file
from panels import PropertiesPanel
from project_tree import ProjectTree
from project_state import ProjectState
from project_stages import ProjectStage
from stage_bar import StageBar
from ui_numeric import ScientificDoubleSpinBox as QDoubleSpinBox
from ui_numeric import ScientificSpinBox as QSpinBox
from ui_numeric import is_numeric_text, parse_numeric_text
from controllers import (
    CommandBus,
    EventBus,
    BCController,
    GeometryController,
    InteractionController,
    MaterialController,
    ParticleController,
    ResultsController,
    SolverController,
    GenerateParticlesCommand,
    RunSolverCommand,
)
from quick_start_dialog import QuickStartDialog
from startup_dialog import StartupDialog
from save_up_to_dialog import SaveUpToDialog
from dependency_dialog import DependencyCheckDialog, collect_dependency_report
from sketch_view import SketchView
from ui_icons import get_icon, get_stage_icon
from viewport_3d import Mesh3DView
from point_cloud_view import PointCloudView2D
from gmsh_mesher import GmshError, generate_volume_mesh
from models import (
    FIELD_DISTRIBUTION_PROPERTY_KEYS,
    Interface,
    Material,
    normalize_heterogeneity_config,
    normalize_material_field_config,
)
from material_registry import (
    all_registry_parameter_keys,
    behavior_label,
    damage_label,
    default_parameter_value as registry_default_parameter_value,
    infer_behavior_from_mat_type,
    legacy_mat_type_for_behavior,
    material_behavior_options,
    material_damage_options,
    material_parameter_keys,
    material_property_schema,
    material_symmetry_options,
    normalize_material_behavior,
    normalize_material_damage,
    normalize_material_properties,
    normalize_material_symmetry,
)
from project_schema import (
    APP_VERSION,
    CURRENT_SCHEMA_VERSION,
    apply_schema_migrations,
    merge_project_state_into_project_data,
    project_state_from_project_data,
)
from ui_theme import (
    UI_TOKENS,
    apply_professional_theme,
    configure_qt_runtime,
    primitive_button_size,
    primitive_icon_size,
    toolbar_icon_size,
)
from update_checker import (
    ReleaseInfo,
    get_available_update,
    select_preferred_asset,
)


def _project_root_dir():
    return str(get_project_root())


def _workspace_dir():
    return str(get_workspace_dir())


def _workspace_path(*parts):
    return str(get_workspace_path(*parts))


def _workspace_input_path(*parts):
    return _workspace_path("input", *parts)


def _workspace_output_path(*parts):
    return _workspace_path("output", *parts)


def _iface_get(iface, key, default=None):
    if isinstance(iface, dict):
        return iface.get(key, default)
    return getattr(iface, key, default)


def _iface_set(iface, key, value):
    if isinstance(iface, dict):
        iface[key] = value
        return
    setattr(iface, key, value)


class MiniMapWidget(QWidget):
    """Bird's-eye overview of the canvas. Draws all parts at a tiny scale,
    plus a viewport rectangle showing what's currently on screen. Click /
    drag to recenter the main view. Auto-redraws when parts change or the
    user pans/zooms."""

    def __init__(self, view, parent=None):
        super().__init__(parent)
        self.setObjectName("MiniMap")
        self._view = view
        self._padding = 6
        self.setFixedSize(180, 130)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Mini-map — click or drag to pan the canvas")
        self._dragging = False
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(60)
        self._refresh_timer.timeout.connect(self.update)
        # Hook view signals so the map repaints when the world changes.
        try:
            if hasattr(view, "partsChanged"):
                view.partsChanged.connect(self._refresh_timer.start)
            if hasattr(view, "zoomChanged"):
                view.zoomChanged.connect(lambda *_: self._refresh_timer.start())
            hbar = view.horizontalScrollBar()
            vbar = view.verticalScrollBar()
            if hbar is not None:
                hbar.valueChanged.connect(lambda *_: self._refresh_timer.start())
            if vbar is not None:
                vbar.valueChanged.connect(lambda *_: self._refresh_timer.start())
        except Exception:
            pass

    def _world_bounds(self):
        """Bounding rect of all parts (and the current viewport). Returns
        a QRectF in scene coordinates, or None if there's nothing yet."""
        parts = getattr(self._view, "parts", None) or []
        xs = []
        ys = []
        for part in parts:
            geom = getattr(part, "geometry", None) or getattr(part, "geom", None)
            if geom is None:
                continue
            try:
                minx, miny, maxx, maxy = geom.bounds
                xs.extend([minx, maxx])
                ys.extend([miny, maxy])
            except Exception:
                continue
        # Always include the current viewport so the indicator stays visible
        # even when the scene is empty.
        try:
            vp_rect = self._view.mapToScene(self._view.viewport().rect()).boundingRect()
            xs.extend([vp_rect.left(), vp_rect.right()])
            ys.extend([vp_rect.top(), vp_rect.bottom()])
        except Exception:
            pass
        if not xs or not ys:
            return None
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        # Pad 10% so content doesn't touch the frame
        w = max(maxx - minx, 1e-6)
        h = max(maxy - miny, 1e-6)
        pad_x = w * 0.1
        pad_y = h * 0.1
        return QRectF(minx - pad_x, miny - pad_y, w + 2 * pad_x, h + 2 * pad_y)

    def _scale_to_widget(self):
        """Return (scale, offset_x, offset_y, world_rect) mapping scene
        coords into the widget's drawable area. None if no content."""
        world = self._world_bounds()
        if world is None or world.width() <= 0 or world.height() <= 0:
            return None
        pad = self._padding
        avail_w = max(1.0, self.width() - 2 * pad)
        avail_h = max(1.0, self.height() - 2 * pad)
        scale = min(avail_w / world.width(), avail_h / world.height())
        # Center the drawing in the widget.
        used_w = world.width() * scale
        used_h = world.height() * scale
        ox = pad + (avail_w - used_w) * 0.5
        oy = pad + (avail_h - used_h) * 0.5
        return scale, ox, oy, world

    def _world_to_widget(self, x, y, m):
        scale, ox, oy, world = m
        # Y axis on the canvas is "up positive" — flip to widget Y.
        wx = ox + (x - world.left()) * scale
        wy = oy + (world.bottom() - y) * scale
        return wx, wy

    def _widget_to_world(self, wx, wy, m):
        scale, ox, oy, world = m
        x = world.left() + (wx - ox) / scale
        y = world.bottom() - (wy - oy) / scale
        return x, y

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            # Background frame
            painter.fillRect(self.rect(), QColor(20, 24, 32, 235))
            painter.setPen(QPen(QColor(80, 90, 110, 220), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
            m = self._scale_to_widget()
            if m is None:
                painter.setPen(QColor(160, 170, 185, 200))
                painter.drawText(self.rect(), Qt.AlignCenter, "No geometry yet")
                return
            scale, ox, oy, world = m
            # Draw each part as a filled polygon
            painter.setPen(QPen(QColor(120, 150, 220, 220), 1.0))
            painter.setBrush(QColor(120, 150, 220, 130))
            parts = getattr(self._view, "parts", None) or []
            for part in parts:
                geom = getattr(part, "geometry", None) or getattr(part, "geom", None)
                if geom is None:
                    continue
                polys = []
                try:
                    geom_type = getattr(geom, "geom_type", "")
                    if geom_type == "Polygon":
                        polys.append(list(geom.exterior.coords))
                    elif geom_type == "MultiPolygon":
                        for sub in geom.geoms:
                            polys.append(list(sub.exterior.coords))
                except Exception:
                    continue
                for ring in polys:
                    if len(ring) < 2:
                        continue
                    pts = [QPointF(*self._world_to_widget(x, y, m)) for x, y in ring]
                    from PySide6.QtGui import QPolygonF
                    painter.drawPolygon(QPolygonF(pts))
            # Draw viewport indicator (current visible region)
            try:
                vp_rect = self._view.mapToScene(self._view.viewport().rect()).boundingRect()
                tl = self._world_to_widget(vp_rect.left(), vp_rect.top(), m)
                br = self._world_to_widget(vp_rect.right(), vp_rect.bottom(), m)
                vp_w = abs(br[0] - tl[0])
                vp_h = abs(br[1] - tl[1])
                vp_x = min(tl[0], br[0])
                vp_y = min(tl[1], br[1])
                painter.setBrush(QColor(255, 220, 100, 50))
                painter.setPen(QPen(QColor(255, 200, 60, 230), 1.4))
                painter.drawRect(QRectF(vp_x, vp_y, vp_w, vp_h))
            except Exception:
                pass
        finally:
            painter.end()

    def _recenter_view_at(self, widget_pos):
        m = self._scale_to_widget()
        if m is None:
            return
        x, y = self._widget_to_world(widget_pos.x(), widget_pos.y(), m)
        try:
            self._view.centerOn(QPointF(x, y))
        except Exception:
            pass
        self._refresh_timer.start()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._recenter_view_at(event.position())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._recenter_view_at(event.position())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = False
            event.accept()
            return
        super().mouseReleaseEvent(event)


class SelectionMiniToolbar(QWidget):
    """Floating mini-toolbar that appears next to the selected part with
    quick actions (Delete, Duplicate, Combine, Edit). Hidden when no
    selection. Repositions when the view pans/zooms or the part moves."""

    def __init__(self, view, main_window, parent=None):
        super().__init__(parent or view.viewport())
        self.setObjectName("SelectionMiniToolbar")
        self._view = view
        self._main = main_window
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        def make_btn(text, tip, handler):
            b = QToolButton(self)
            b.setObjectName("MiniToolbarButton")
            b.setText(text)
            b.setToolTip(tip)
            b.setCursor(Qt.PointingHandCursor)
            b.setAutoRaise(True)
            b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(handler)
            return b

        self._btn_duplicate = make_btn("⧉", "Duplicate selection", self._on_duplicate)
        self._btn_combine = make_btn("∪", "Combine selected parts (Ctrl+P)", self._on_combine)
        self._btn_delete = make_btn("🗑", "Delete selected part", self._on_delete)
        layout.addWidget(self._btn_duplicate)
        layout.addWidget(self._btn_combine)
        layout.addWidget(self._btn_delete)
        self.hide()

        # Wire view signals to reposition / hide the toolbar.
        try:
            if hasattr(view, "partSelectionChanged"):
                view.partSelectionChanged.connect(lambda *_: self.refresh())
            if hasattr(view, "partsChanged"):
                view.partsChanged.connect(lambda *_: self.refresh())
            if hasattr(view, "zoomChanged"):
                view.zoomChanged.connect(lambda *_: self.refresh())
            hbar = view.horizontalScrollBar()
            vbar = view.verticalScrollBar()
            if hbar is not None:
                hbar.valueChanged.connect(lambda *_: self.refresh())
            if vbar is not None:
                vbar.valueChanged.connect(lambda *_: self.refresh())
        except Exception:
            pass

    def _selected_part(self):
        if getattr(self._view, "active_module", "") != "Part":
            return None
        pid = getattr(self._view, "selected_part_id", None)
        if pid is None:
            return None
        return next(
            (p for p in getattr(self._view, "parts", []) if int(getattr(p, "id", -1)) == int(pid)),
            None,
        )

    def refresh(self):
        part = self._selected_part()
        if part is None or getattr(part, "geometry", None) is None:
            self.hide()
            return
        try:
            minx, miny, maxx, maxy = part.geometry.bounds
        except Exception:
            self.hide()
            return
        # Map the part's top-center to viewport pixels. Canvas y-axis is
        # flipped (up positive) so the "top" of the part is at maxy.
        cx_scene = (minx + maxx) * 0.5
        top_scene = maxy
        view_pt = self._view.mapFromScene(QPointF(cx_scene, top_scene))
        # Multi-select count → show Combine only when 2+ parts selected.
        multi = getattr(self._view, "multi_selected_part_ids", None) or set()
        self._btn_combine.setVisible(len(multi) >= 2)
        self.adjustSize()
        # Place toolbar 14 px above the top of the part, horizontally
        # centered on it. Clamp to viewport.
        x = int(view_pt.x() - self.width() // 2)
        y = int(view_pt.y() - self.height() - 14)
        vp = self._view.viewport()
        vp_w = vp.width()
        vp_h = vp.height()
        x = max(4, min(vp_w - self.width() - 4, x))
        # If the toolbar would clip off the top, drop it below the part.
        if y < 4:
            bot_pt = self._view.mapFromScene(QPointF(cx_scene, miny))
            y = int(bot_pt.y() + 14)
        y = max(4, min(vp_h - self.height() - 4, y))
        self.move(x, y)
        self.show()
        self.raise_()

    def _on_delete(self):
        part = self._selected_part()
        if part is None:
            return
        try:
            self._view.delete_part(part, confirm=True)
        except Exception:
            pass

    def _on_duplicate(self):
        part = self._selected_part()
        if part is None:
            return
        try:
            self._view.copy_selected_part(20.0, -20.0, count=1, name_suffix="Copy")
        except Exception:
            pass

    def _on_combine(self):
        try:
            self._view.combine_selected_parts()
        except Exception:
            pass


class UpdateCheckWorker(QThread):
    completed = Signal(object, str)

    def __init__(self, repo: str, current_version: str, timeout_sec: float):
        super().__init__()
        self.repo = repo
        self.current_version = current_version
        self.timeout_sec = timeout_sec

    def stop(self):
        try:
            self.requestInterruption()
        except Exception:
            pass
        try:
            self.quit()
        except Exception:
            pass
        try:
            if QThread.currentThread() != self:
                self.wait()
        except Exception:
            pass

    def run(self):
        try:
            result = get_available_update(
                repo=self.repo,
                current_version=self.current_version,
                timeout=self.timeout_sec,
            )
            self.completed.emit(result, "")
        except Exception as exc:
            self.completed.emit(None, str(exc))


class PorousMaterialDialog(QDialog):
    def __init__(self, sketch_view, parent=None, initial_settings=None, target_part=None):
        super().__init__(parent)
        self.sketch_view = sketch_view
        self.target_part = target_part
        self._preview_polys = []
        self._last_generation_report = {}
        self._min_size = None
        self.setWindowTitle("Edit Feature" if target_part is not None else "Porous/Particle Generator")
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        form = QFormLayout()
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(["Circle", "Square", "Rectangle", "Triangle", "Polygon"])
        form.addRow("Feature shape", self.shape_combo)

        self.polygon_sides = QSpinBox()
        self.polygon_sides.setRange(3, 12)
        self.polygon_sides.setValue(6)
        form.addRow("Polygon sides", self.polygon_sides)

        self.aspect_spin = QDoubleSpinBox()
        self.aspect_spin.setRange(0.1, 10.0)
        self.aspect_spin.setSingleStep(0.1)
        self.aspect_spin.setValue(1.0)
        form.addRow("Rect aspect (W/H)", self.aspect_spin)

        self.dist_combo = QComboBox()
        self.dist_combo.addItems(["Uniform", "Normal"])
        form.addRow("Size distribution", self.dist_combo)

        self.size_min = QDoubleSpinBox()
        self.size_min.setRange(0.1, 1e6)
        self.size_min.setValue(2.0)
        self.size_min.setSingleStep(0.5)
        form.addRow("Size min", self.size_min)

        self.size_max = QDoubleSpinBox()
        self.size_max.setRange(0.1, 1e6)
        self.size_max.setValue(5.0)
        self.size_max.setSingleStep(0.5)
        form.addRow("Size max", self.size_max)

        self.size_mean = QDoubleSpinBox()
        self.size_mean.setRange(0.1, 1e6)
        self.size_mean.setValue(3.0)
        self.size_mean.setSingleStep(0.5)
        form.addRow("Size mean", self.size_mean)

        self.size_std = QDoubleSpinBox()
        self.size_std.setRange(0.0, 1e6)
        self.size_std.setValue(0.5)
        self.size_std.setSingleStep(0.1)
        form.addRow("Size std", self.size_std)

        self.layout_combo = QComboBox()
        self.layout_combo.addItems(["Random", "Lattice"])
        form.addRow("Distribution", self.layout_combo)

        self.lattice_combo = QComboBox()
        self.lattice_combo.addItems(["Square", "Rectangular", "Triangular"])
        form.addRow("Lattice type", self.lattice_combo)

        self.spacing_spin = QDoubleSpinBox()
        self.spacing_spin.setRange(0.1, 1e6)
        self.spacing_spin.setValue(8.0)
        self.spacing_spin.setSingleStep(0.5)
        form.addRow("Avg spacing", self.spacing_spin)

        self.spacing_y_spin = QDoubleSpinBox()
        self.spacing_y_spin.setRange(0.1, 1e6)
        self.spacing_y_spin.setValue(8.0)
        self.spacing_y_spin.setSingleStep(0.5)
        form.addRow("Spacing Y", self.spacing_y_spin)

        self.random_rot = QCheckBox("Random rotation")
        self.random_rot.setChecked(True)
        form.addRow("", self.random_rot)

        self.reject_overlaps = QCheckBox("Reject overlaps")
        self.reject_overlaps.setChecked(True)
        self.reject_overlaps.setToolTip(
            "When enabled, overlapping solid particles are rejected during generation."
        )
        form.addRow("", self.reject_overlaps)

        mode_group = QGroupBox("Result")
        mode_layout = QHBoxLayout(mode_group)
        self.mode_holes = QRadioButton("Create holes (porous)")
        self.mode_particles = QRadioButton("Create particles (composite)")
        self.mode_holes.setChecked(True)
        mode_layout.addWidget(self.mode_holes)
        mode_layout.addWidget(self.mode_particles)

        self.name_base = QLineEdit("Feature")
        form.addRow("Feature name", self.name_base)

        layout.addLayout(form)
        layout.addWidget(mode_group)

        self.status_label = QLabel("")
        self.status_label.setObjectName("MinorStatusLabel")
        layout.addWidget(self.status_label)

        btn_row = QHBoxLayout()
        self.preview_btn = QPushButton("Preview")
        self.apply_btn = QPushButton("Apply")
        create_label = "Update Feature" if target_part is not None else "Create Part"
        self.create_part_btn = QPushButton(create_label)
        self.cut_hole_btn = QPushButton("Cut Hole")
        self.cancel_btn = QPushButton("Close")
        btn_row.addWidget(self.preview_btn)
        btn_row.addWidget(self.apply_btn)
        btn_row.addWidget(self.create_part_btn)
        btn_row.addWidget(self.cut_hole_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

        self.preview_btn.clicked.connect(self._preview)
        self.apply_btn.clicked.connect(self._apply)
        self.create_part_btn.clicked.connect(self._apply_and_confirm_part)
        self.cut_hole_btn.clicked.connect(self._apply_and_cut_hole)
        self.cancel_btn.clicked.connect(self.reject)
        self.shape_combo.currentIndexChanged.connect(self._update_visibility)
        self.dist_combo.currentIndexChanged.connect(self._update_visibility)
        self.layout_combo.currentIndexChanged.connect(self._update_visibility)
        self.lattice_combo.currentIndexChanged.connect(self._update_visibility)
        self.mode_holes.toggled.connect(self._update_visibility)
        self.mode_particles.toggled.connect(self._update_visibility)

        self._update_visibility()
        self._load_settings(initial_settings=initial_settings)
        if self.target_part is not None:
            self.apply_btn.setVisible(False)
            self.cut_hole_btn.setVisible(False)

    def closeEvent(self, event):
        self._clear_preview()
        super().closeEvent(event)

    def _update_visibility(self):
        shape = self.shape_combo.currentText().lower()
        self.polygon_sides.setVisible(shape == "polygon")
        self.aspect_spin.setVisible(shape == "rectangle")
        dist = self.dist_combo.currentText().lower()
        uniform = dist.startswith("uniform")
        self.size_min.setVisible(uniform)
        self.size_max.setVisible(uniform)
        self.size_mean.setVisible(not uniform)
        self.size_std.setVisible(not uniform)
        layout_mode = self.layout_combo.currentText().lower()
        lattice = layout_mode.startswith("lattice")
        self.lattice_combo.setVisible(lattice)
        self.spacing_y_spin.setVisible(lattice and self.lattice_combo.currentText().lower().startswith("rect"))
        particles_mode = bool(self.mode_particles.isChecked())
        self.reject_overlaps.setEnabled(particles_mode)

    def _get_domain(self):
        geom = self.sketch_view.get_pore_domain_geometry()
        if geom is None or geom.is_empty:
            QMessageBox.warning(self, "Porous/Particle", "No valid geometry found for features.")
            return None
        return geom

    def _sample_size(self):
        if self.dist_combo.currentText().lower().startswith("uniform"):
            mn = float(self.size_min.value())
            mx = float(self.size_max.value())
            if mx < mn:
                mn, mx = mx, mn
            return random.uniform(mn, mx)
        mean = float(self.size_mean.value())
        std = float(self.size_std.value())
        for _ in range(5):
            val = random.gauss(mean, std)
            if val > 0:
                return val
        return max(0.1, mean)

    def _build_shape(self, size):
        shape = self.shape_combo.currentText().lower()
        size = max(0.1, float(size))
        if shape == "circle":
            return Point(0, 0).buffer(size / 2.0, resolution=32)
        if shape == "square":
            side = size / math.sqrt(2)
            half = side / 2.0
            return Polygon([(-half, -half), (half, -half), (half, half), (-half, half)])
        if shape == "rectangle":
            aspect = float(self.aspect_spin.value())
            h = size / math.sqrt(aspect * aspect + 1.0)
            w = aspect * h
            return Polygon([(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)])
        if shape == "triangle":
            r = size / 2.0
            pts = []
            for i in range(3):
                ang = math.radians(90 + i * 120)
                pts.append((r * math.cos(ang), r * math.sin(ang)))
            return Polygon(pts)
        if shape == "polygon":
            sides = int(self.polygon_sides.value())
            r = size / 2.0
            pts = []
            for i in range(sides):
                ang = 2 * math.pi * i / sides
                pts.append((r * math.cos(ang), r * math.sin(ang)))
            return Polygon(pts)
        return Point(0, 0).buffer(size / 2.0, resolution=16)

    def _generate_centers(self, domain):
        minx, miny, maxx, maxy = domain.bounds
        spacing = max(0.1, float(self.spacing_spin.value()))
        layout_mode = self.layout_combo.currentText().lower()
        centers = []
        if layout_mode.startswith("random"):
            target = max(1, int(domain.area / max(spacing * spacing, 1e-6)))
            attempts = 0
            limit = target * 25
            min_dist_sq = (spacing * 0.8) ** 2
            while len(centers) < target and attempts < limit:
                x = random.uniform(minx, maxx)
                y = random.uniform(miny, maxy)
                if not domain.contains(Point(x, y)):
                    attempts += 1
                    continue
                if all((x - cx) ** 2 + (y - cy) ** 2 >= min_dist_sq for cx, cy in centers):
                    centers.append((x, y))
                attempts += 1
            return centers

        lattice_type = self.lattice_combo.currentText().lower()
        sx = spacing
        sy = spacing
        if lattice_type.startswith("rect"):
            sy = max(0.1, float(self.spacing_y_spin.value()))
        if lattice_type.startswith("tri"):
            sy = spacing * math.sqrt(3) / 2.0
        y = miny + sy / 2.0
        row = 0
        while y <= maxy - sy / 2.0:
            if lattice_type.startswith("tri") and row % 2 == 1:
                x = minx + sx
            else:
                x = minx + sx / 2.0
            while x <= maxx - sx / 2.0:
                if domain.contains(Point(x, y)):
                    centers.append((x, y))
                x += sx
            y += sy
            row += 1
        return centers

    def _grid_keys_for_bounds(self, bounds, cell_size):
        minx, miny, maxx, maxy = bounds
        ix0 = int(math.floor(minx / cell_size))
        iy0 = int(math.floor(miny / cell_size))
        ix1 = int(math.floor(maxx / cell_size))
        iy1 = int(math.floor(maxy / cell_size))
        keys = []
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                keys.append((ix, iy))
        return keys

    def _has_feature_overlap(self, geom, placed_polys, cell_map, cell_size):
        checked = set()
        for key in self._grid_keys_for_bounds(geom.bounds, cell_size):
            for idx in cell_map.get(key, []):
                if idx in checked:
                    continue
                checked.add(idx)
                other = placed_polys[idx]
                try:
                    overlap_area = geom.intersection(other).area
                    if overlap_area > 1e-9:
                        return True
                except Exception:
                    try:
                        if geom.intersects(other):
                            return True
                    except Exception:
                        continue
        return False

    def _feature_overlap_warning_text(self):
        report = getattr(self, "_last_generation_report", {}) or {}
        if not bool(report.get("enforce_non_overlap", False)):
            return ""
        overlap = int(report.get("skipped_overlap", 0) or 0)
        if overlap <= 0:
            return ""
        placed = int(report.get("placed", 0) or 0)
        total = int(report.get("total_centers", 0) or 0)
        unit = getattr(self.sketch_view, "current_unit", "m")
        lines = [
            f"{overlap} generated solid particle(s) were rejected to prevent overlap.",
            f"Placed {placed} non-overlapping solid particle(s) out of {total} candidate location(s).",
        ]
        suggested_spacing = report.get("suggested_spacing")
        if suggested_spacing:
            lines.append(f"Suggested change: set Avg spacing to at least {float(suggested_spacing):.3f} {unit}.")
        suggested_spacing_y = report.get("suggested_spacing_y")
        if suggested_spacing_y:
            lines.append(f"Suggested change: set Spacing Y to at least {float(suggested_spacing_y):.3f} {unit}.")
        lines.append("You can also lower Size max / Size std to reduce clashes.")
        return "\n".join(lines)

    def _generate_polygons(self):
        domain = self._get_domain()
        if domain is None:
            self._last_generation_report = {}
            return []
        centers = self._generate_centers(domain)
        if not centers:
            self._last_generation_report = {
                "total_centers": 0,
                "placed": 0,
                "skipped_outside": 0,
                "skipped_overlap": 0,
            }
            return []
        polys = []
        min_size = None
        max_span = 0.0
        skipped_outside = 0
        skipped_overlap = 0
        enforce_non_overlap = bool(self.mode_particles.isChecked() and self.reject_overlaps.isChecked())
        spacing_x = max(0.1, float(self.spacing_spin.value()))
        spacing_y = spacing_x
        if self.layout_combo.currentText().lower().startswith("lattice"):
            lattice = self.lattice_combo.currentText().lower()
            if lattice.startswith("rect"):
                spacing_y = max(0.1, float(self.spacing_y_spin.value()))
            elif lattice.startswith("tri"):
                spacing_y = spacing_x * math.sqrt(3) / 2.0
        cell_size = max(0.1, min(spacing_x, spacing_y) * 0.75)
        cell_map = {}
        for cx, cy in centers:
            size = self._sample_size()
            base = self._build_shape(size)
            if self.random_rot.isChecked():
                base = rotate(base, random.uniform(0, 360), origin=(0, 0))
            geom = translate(base, cx, cy)
            if not domain.contains(geom):
                skipped_outside += 1
                continue
            bx0, by0, bx1, by1 = geom.bounds
            max_span = max(max_span, bx1 - bx0, by1 - by0)
            if enforce_non_overlap and self._has_feature_overlap(geom, polys, cell_map, cell_size):
                skipped_overlap += 1
                continue
            idx = len(polys)
            polys.append(geom)
            if enforce_non_overlap:
                for key in self._grid_keys_for_bounds(geom.bounds, cell_size):
                    cell_map.setdefault(key, []).append(idx)
            if min_size is None or size < min_size:
                min_size = size
        self._min_size = min_size
        suggested = max(spacing_x, max_span * 1.05) if max_span > 0 else spacing_x
        suggested_y = None
        if self.layout_combo.currentText().lower().startswith("lattice"):
            suggested_y = max(spacing_y, max_span * 1.05) if max_span > 0 else spacing_y
        self._last_generation_report = {
            "total_centers": len(centers),
            "placed": len(polys),
            "skipped_outside": skipped_outside,
            "skipped_overlap": skipped_overlap,
            "enforce_non_overlap": enforce_non_overlap,
            "suggested_spacing": suggested,
            "suggested_spacing_y": suggested_y,
        }
        return polys

    def _clear_preview(self):
        self._preview_polys = []
        self.sketch_view.clear_pore_preview()

    def _preview(self):
        self._save_settings()
        self._preview_polys = self._generate_polygons()
        if not self._preview_polys:
            overlap_text = self._feature_overlap_warning_text()
            if overlap_text:
                QMessageBox.warning(self, "Porous/Particle", overlap_text)
                return
            QMessageBox.information(self, "Porous/Particle", "No features generated with current settings.")
            return
        overlap_text = self._feature_overlap_warning_text()
        if overlap_text:
            QMessageBox.warning(self, "Porous/Particle", overlap_text)
        self.sketch_view.show_pore_preview(self._preview_polys)
        report = getattr(self, "_last_generation_report", {}) or {}
        rejected = int(report.get("skipped_overlap", 0) or 0)
        enforce_non_overlap = bool(report.get("enforce_non_overlap", False))
        if enforce_non_overlap and rejected > 0:
            self.status_label.setText(
                f"Preview: {len(self._preview_polys)} non-overlapping solid particles ({rejected} rejected)"
            )
        elif enforce_non_overlap:
            self.status_label.setText(f"Preview: {len(self._preview_polys)} non-overlapping solid particles")
        else:
            if self.mode_particles.isChecked():
                self.status_label.setText(
                    f"Preview: {len(self._preview_polys)} solid particles (overlap allowed by setting)"
                )
            else:
                self.status_label.setText(
                    f"Preview: {len(self._preview_polys)} pores/features (overlap allowed in hole mode)"
                )

    def _current_mesh_dx(self):
        main = self.parent()
        if main and hasattr(main, "properties_panel"):
            mesh_tab = getattr(main.properties_panel, "mesh_tab", None)
            if mesh_tab and mesh_tab.sizing_combo.currentIndex() == 0:
                return float(mesh_tab.dx_spin.value())
        return None

    def _apply(self):
        self._save_settings()
        ok = self._ensure_pore_sketches(push_undo=True)
        if ok:
            QMessageBox.information(
                self,
                "Porous/Particle",
                "Feature sketches created.\nUse Confirm Part to create particles, or Cut Hole to subtract from the main part.",
            )
        else:
            QMessageBox.information(self, "Porous/Particle", "No features were applied.")

    def _ensure_pore_sketches(self, push_undo=True):
        if not self._preview_polys:
            self._preview()
        if not self._preview_polys:
            return False
        name = self.name_base.text().strip() or "Feature"
        ok = self.sketch_view.apply_pore_sketches(
            self._preview_polys,
            base_name=name,
            push_undo=push_undo,
        )
        if ok:
            self._clear_preview()
            return True
        return False

    def _apply_and_confirm_part(self):
        name = self.name_base.text().strip() or "Feature"
        if self.target_part is not None:
            if not self._preview_polys:
                self._preview()
            if not self._preview_polys:
                return
            self._save_settings()
            ok = self.sketch_view.update_generated_feature_part(
                self.target_part,
                self._preview_polys,
                settings=self._collect_settings(),
                base_name=name,
            )
            if ok:
                self._clear_preview()
                self.accept()
            return
        if not self._ensure_pore_sketches(push_undo=False):
            return
        self.sketch_view.confirm_solid()
        self.accept()

    def _apply_and_cut_hole(self):
        if not self._ensure_pore_sketches(push_undo=False):
            return
        name = self.name_base.text().strip() or "Feature"
        self.sketch_view.cut_hole(hole_name_base=name, skip_undo=False, merge_voids=True)
        self.accept()

    def _collect_settings(self):
        return {
            "shape": self.shape_combo.currentText(),
            "polygon_sides": int(self.polygon_sides.value()),
            "aspect": float(self.aspect_spin.value()),
            "distribution": self.dist_combo.currentText(),
            "size_min": float(self.size_min.value()),
            "size_max": float(self.size_max.value()),
            "size_mean": float(self.size_mean.value()),
            "size_std": float(self.size_std.value()),
            "layout": self.layout_combo.currentText(),
            "lattice": self.lattice_combo.currentText(),
            "spacing": float(self.spacing_spin.value()),
            "spacing_y": float(self.spacing_y_spin.value()),
            "random_rot": bool(self.random_rot.isChecked()),
            "reject_overlaps": bool(self.reject_overlaps.isChecked()),
            "mode": "holes" if self.mode_holes.isChecked() else "particles",
            "name_base": self.name_base.text(),
        }

    def _apply_settings(self, settings):
        if not settings:
            return
        shape = settings.get("shape")
        if shape:
            idx = self.shape_combo.findText(shape)
            if idx >= 0:
                self.shape_combo.setCurrentIndex(idx)
        self.polygon_sides.setValue(int(settings.get("polygon_sides", self.polygon_sides.value())))
        self.aspect_spin.setValue(float(settings.get("aspect", self.aspect_spin.value())))
        dist = settings.get("distribution")
        if dist:
            idx = self.dist_combo.findText(dist)
            if idx >= 0:
                self.dist_combo.setCurrentIndex(idx)
        self.size_min.setValue(float(settings.get("size_min", self.size_min.value())))
        self.size_max.setValue(float(settings.get("size_max", self.size_max.value())))
        self.size_mean.setValue(float(settings.get("size_mean", self.size_mean.value())))
        self.size_std.setValue(float(settings.get("size_std", self.size_std.value())))
        layout = settings.get("layout")
        if layout:
            idx = self.layout_combo.findText(layout)
            if idx >= 0:
                self.layout_combo.setCurrentIndex(idx)
        lattice = settings.get("lattice")
        if lattice:
            idx = self.lattice_combo.findText(lattice)
            if idx >= 0:
                self.lattice_combo.setCurrentIndex(idx)
        self.spacing_spin.setValue(float(settings.get("spacing", self.spacing_spin.value())))
        self.spacing_y_spin.setValue(float(settings.get("spacing_y", self.spacing_y_spin.value())))
        self.random_rot.setChecked(bool(settings.get("random_rot", self.random_rot.isChecked())))
        self.reject_overlaps.setChecked(bool(settings.get("reject_overlaps", self.reject_overlaps.isChecked())))
        if settings.get("mode") == "particles":
            self.mode_particles.setChecked(True)
        else:
            self.mode_holes.setChecked(True)
        name_base = settings.get("name_base")
        if name_base:
            self.name_base.setText(name_base)
        self._update_visibility()

    def _save_settings(self):
        self.sketch_view.porous_settings = self._collect_settings()

    def _load_settings(self, initial_settings=None):
        settings = initial_settings if initial_settings is not None else getattr(self.sketch_view, "porous_settings", None)
        self._apply_settings(settings)

class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=0, h_spacing=6, v_spacing=6):
        super().__init__(parent)
        self._items = []
        self._h_spacing = h_spacing
        self._v_spacing = v_spacing
        self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        left, top, right, bottom = self.getContentsMargins()
        size += QSize(left + right, top + bottom)
        return size

    def _do_layout(self, rect, test_only):
        left, top, right, bottom = self.getContentsMargins()
        effective = rect.adjusted(left, top, -right, -bottom)
        x = effective.x()
        y = effective.y()
        line_height = 0

        for item in self._items:
            widget = item.widget()
            if widget is not None and not widget.isVisible():
                continue
            item_size = item.sizeHint()
            next_x = x + item_size.width() + self._h_spacing
            if next_x - self._h_spacing > effective.right() and line_height > 0:
                x = effective.x()
                y = y + line_height + self._v_spacing
                next_x = x + item_size.width() + self._h_spacing
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item_size))
            x = next_x
            line_height = max(line_height, item_size.height())

        return y + line_height - rect.y() + bottom


class _InspectorFormPage(QWidget):
    def __init__(self, inspector, parent=None):
        super().__init__(parent)
        self.inspector = inspector
        self.summary_text = "No selection"
        self._finalized = False
        self._form_layouts = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._host = QWidget()
        self._content_layout = QVBoxLayout(self._host)
        self._content_layout.setContentsMargins(12, 12, 12, 12)
        self._content_layout.setSpacing(8)
        self._content_layout.setAlignment(Qt.AlignTop)
        self._active_form_layout = None
        self._scroll.setWidget(self._host)
        layout.addWidget(self._scroll, 1)

    def clear_form(self):
        self._active_form_layout = None
        self._finalized = False
        self._form_layouts = []
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                while child_layout.count():
                    sub_item = child_layout.takeAt(0)
                    sub_widget = sub_item.widget()
                    if sub_widget is not None:
                        sub_widget.deleteLater()

    def set_summary(self, text):
        self.summary_text = str(text or "No selection")

    def _ensure_section(self, title="Basic Properties"):
        if self._active_form_layout is None:
            self.add_section(title)
        return self._active_form_layout

    def add_section_container(self, title):
        group = QGroupBox(str(title))
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)
        self._content_layout.addWidget(group)
        self._active_form_layout = None
        return group, layout

    def add_section(self, title):
        group = QGroupBox(str(title))
        form_layout = QFormLayout(group)
        form_layout.setContentsMargins(12, 10, 12, 10)
        form_layout.setHorizontalSpacing(10)
        form_layout.setVerticalSpacing(8)
        form_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form_layout.setFormAlignment(Qt.AlignTop | Qt.AlignLeft)
        form_layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._content_layout.addWidget(group)
        self._active_form_layout = form_layout
        self._form_layouts.append(form_layout)
        self._sync_form_wrap_policy()
        return form_layout

    def finalize_form(self):
        if self._finalized:
            return
        self._content_layout.addStretch(1)
        self._finalized = True

    def add_readonly_row(self, label, value, form_layout=None):
        form = form_layout or self._ensure_section()
        field = QLabel(str(value))
        field.setObjectName("SummaryLabel")
        field.setWordWrap(True)
        form.addRow(f"{label}:", field)

    def add_text_row(self, label, value, callback, form_layout=None):
        form = form_layout or self._ensure_section()
        field = QLineEdit(str("" if value is None else value))
        field.editingFinished.connect(lambda w=field, cb=callback: cb(w.text()))
        form.addRow(f"{label}:", field)

    def add_float_row(self, label, value, callback, form_layout=None):
        form = form_layout or self._ensure_section()
        field = QDoubleSpinBox()
        field.setDecimals(6)
        field.setRange(-1e12, 1e12)
        field.setValue(float(value or 0.0))
        field.valueChanged.connect(lambda v, cb=callback: cb(float(v)))
        form.addRow(f"{label}:", field)

    def add_int_row(self, label, value, callback, form_layout=None):
        form = form_layout or self._ensure_section()
        field = QSpinBox()
        field.setRange(-10**9, 10**9)
        parsed = parse_numeric_text(value)
        field.setValue(int(round(parsed if parsed is not None else 0)))
        field.valueChanged.connect(lambda v, cb=callback: cb(int(v)))
        form.addRow(f"{label}:", field)

    def add_bool_row(self, label, value, callback, form_layout=None):
        form = form_layout or self._ensure_section()
        field = QCheckBox()
        field.setChecked(bool(value))
        field.toggled.connect(lambda v, cb=callback: cb(bool(v)))
        form.addRow(f"{label}:", field)

    def add_combo_row(self, label, items, current_value, callback, form_layout=None):
        form = form_layout or self._ensure_section()
        field = QComboBox()
        field.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for text, data in items:
            field.addItem(str(text), data)
        idx = field.findData(current_value)
        if idx < 0:
            idx = field.findText(str(current_value))
        if idx >= 0:
            field.setCurrentIndex(idx)
        field.currentIndexChanged.connect(lambda _i, w=field, cb=callback: cb(w.currentData()))
        form.addRow(f"{label}:", field)

    def add_property_value_row(self, key, value, callback, form_layout=None):
        if isinstance(value, bool):
            self.add_bool_row(key, value, callback, form_layout=form_layout)
        elif isinstance(value, int):
            self.add_int_row(key, value, callback, form_layout=form_layout)
        elif isinstance(value, float):
            self.add_float_row(key, value, callback, form_layout=form_layout)
        elif isinstance(value, str) and is_numeric_text(value):
            self.add_float_row(key, parse_numeric_text(value), callback, form_layout=form_layout)
        else:
            self.add_text_row(key, value, callback, form_layout=form_layout)

    def populate(self, payload):
        self.clear_form()
        self.set_summary("No selection")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_form_wrap_policy()

    def _sync_form_wrap_policy(self):
        width = int(self._scroll.viewport().width() or self.width() or 0)
        row_wrap = QFormLayout.WrapLongRows if width < 360 else QFormLayout.DontWrapRows
        for layout in self._form_layouts:
            try:
                layout.setRowWrapPolicy(row_wrap)
            except Exception:
                pass


class DefaultEmptyPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        host = QWidget()
        host_layout = QVBoxLayout(host)
        host_layout.setContentsMargins(12, 12, 12, 12)
        host_layout.setSpacing(8)
        self.label = QLabel("Select an object from the model tree or viewport.")
        self.label.setObjectName("MinorStatusLabel")
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        host_layout.addWidget(self.label)
        host_layout.addStretch(1)
        self._scroll.setWidget(host)
        layout.addWidget(self._scroll, 1)

    def set_message(self, text):
        self.label.setText(str(text))


class PropertyEditor(QWidget):
    valueChanged = Signal(str, object)

    def __init__(self, schema=None, data=None, parent=None, on_change=None):
        super().__init__(parent)
        self._schema = []
        self._widgets = {}
        self._on_change = on_change
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setLabelAlignment(Qt.AlignLeft)
        layout.setFormAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(8)
        self._layout = layout
        self.set_schema(schema or [], data or {})

    @property
    def widgets(self):
        return self._widgets

    def clear(self):
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._widgets = {}

    def set_schema(self, schema, data=None):
        self.clear()
        self._schema = list(schema or [])
        values = data or {}
        for field in self._schema:
            key = str(field.get("key", "")).strip()
            if not key:
                continue
            label = str(field.get("name", key))
            widget = self._build_widget(field, values.get(key, field.get("default")))
            if widget is None:
                continue
            self._widgets[key] = widget
            self._layout.addRow(label, widget)
        self._sync_wrap_policy()

    def value(self, key):
        widget = self._widgets.get(str(key))
        if widget is None:
            return None
        if isinstance(widget, QLineEdit):
            return widget.text()
        if isinstance(widget, QDoubleSpinBox):
            return float(widget.value())
        if isinstance(widget, QSpinBox):
            return int(widget.value())
        if isinstance(widget, QComboBox):
            data = widget.currentData()
            return widget.currentText() if data is None else data
        if isinstance(widget, QCheckBox):
            return bool(widget.isChecked())
        return None

    def values(self):
        return {key: self.value(key) for key in self._widgets}

    def _emit_change(self, key):
        value = self.value(key)
        if callable(self._on_change):
            self._on_change(str(key), value)
        self.valueChanged.emit(str(key), value)

    def _build_widget(self, field, value):
        key = str(field.get("key", "")).strip()
        field_type = str(field.get("type", "string")).strip().lower()
        tooltip = str(field.get("tooltip", "") or "")
        if field_type == "float":
            widget = QDoubleSpinBox()
            widget.setRange(float(field.get("minimum", -1e12)), float(field.get("maximum", 1e12)))
            widget.setDecimals(int(field.get("decimals", 6)))
            widget.setMinimumHeight(26)
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            widget.setValue(float(0.0 if value in (None, "") else value))
            widget.valueChanged.connect(lambda _v, k=key: self._emit_change(k))
        elif field_type == "int":
            widget = QSpinBox()
            widget.setRange(int(field.get("minimum", -(10**9))), int(field.get("maximum", 10**9)))
            widget.setMinimumHeight(26)
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            parsed = parse_numeric_text(value)
            widget.setValue(int(round(parsed if parsed is not None else 0)))
            widget.valueChanged.connect(lambda _v, k=key: self._emit_change(k))
        elif field_type == "enum":
            widget = QComboBox()
            widget.setMinimumHeight(26)
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            options = list(field.get("options", []) or [])
            for option in options:
                if isinstance(option, (tuple, list)) and len(option) >= 2:
                    text, data = option[0], option[1]
                else:
                    text = data = option
                widget.addItem(str(text), data)
            idx = widget.findData(value)
            if idx < 0:
                idx = widget.findText(str(value))
            if idx >= 0:
                widget.setCurrentIndex(idx)
            widget.currentIndexChanged.connect(lambda _i, k=key: self._emit_change(k))
        elif field_type == "bool":
            widget = QCheckBox()
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            widget.setChecked(bool(value))
            widget.toggled.connect(lambda _v, k=key: self._emit_change(k))
        else:
            widget = QLineEdit("" if value is None else str(value))
            widget.setMinimumHeight(26)
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            widget.editingFinished.connect(lambda k=key: self._emit_change(k))
        if tooltip:
            widget.setToolTip(tooltip)
        return widget

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_wrap_policy()

    def _sync_wrap_policy(self):
        width = int(self.width() or self.sizeHint().width() or 0)
        row_wrap = QFormLayout.WrapLongRows if width < 340 else QFormLayout.DontWrapRows
        try:
            self._layout.setRowWrapPolicy(row_wrap)
        except Exception:
            pass


def _material_property_keys_for_behavior_symmetry(material_type, symmetry, behavior=None, damage="none"):
    behavior_key = normalize_material_behavior(behavior or infer_behavior_from_mat_type(material_type))
    symmetry_key = normalize_material_symmetry(symmetry)
    damage_key = normalize_material_damage(damage)
    mat_type = legacy_mat_type_for_behavior(behavior_key, material_type)
    return mat_type, material_parameter_keys(behavior_key, symmetry_key, damage_key)


def _default_material_property_value(key):
    return registry_default_parameter_value(key)


def _material_property_field_type(value):
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)) or (isinstance(value, str) and is_numeric_text(value)):
        return "float"
    return "string"


class PartPropertiesPanel(_InspectorFormPage):
    def __init__(self, inspector, parent=None):
        super().__init__(inspector, parent)
        self._dimension_list = None
        self._edit_dimension_button = None

    def _distribution_text(self, config):
        cfg = normalize_heterogeneity_config(config)
        parts = []
        for item in cfg.get("materials", []):
            try:
                mat_id = int(item.get("material_id"))
            except Exception:
                continue
            try:
                frac = float(item.get("fraction", 0.0))
            except Exception:
                frac = 0.0
            if frac <= 0.0:
                continue
            parts.append(f"{mat_id}:{frac}")
        return ", ".join(parts)

    def _set_part_random_distribution_text(self, part, text):
        cfg = normalize_heterogeneity_config(copy.deepcopy(getattr(part, "heterogeneity_config", {})))
        materials = []
        for chunk in str(text or "").split(","):
            item = chunk.strip()
            if not item or ":" not in item:
                continue
            left, right = item.split(":", 1)
            try:
                mat_id = int(left.strip())
                fraction = float(right.strip())
            except Exception:
                continue
            if fraction > 1.0:
                fraction /= 100.0
            if fraction <= 0.0:
                continue
            materials.append({"material_id": mat_id, "fraction": fraction})
        cfg["materials"] = materials
        setattr(part, "heterogeneity_config", cfg)
        self.inspector._notify_change("part")

    def _set_part_field_expression(self, part, key, value):
        cfg = normalize_heterogeneity_config(copy.deepcopy(getattr(part, "heterogeneity_config", {})))
        expressions = dict(cfg.get("expressions", {}) or {})
        expressions[str(key)] = str(value or "")
        cfg["expressions"] = expressions
        setattr(part, "heterogeneity_config", cfg)
        self.inspector._notify_change("part")

    def _set_part_random_seed(self, part, value):
        cfg = normalize_heterogeneity_config(copy.deepcopy(getattr(part, "heterogeneity_config", {})))
        text = str(value or "").strip()
        try:
            cfg["random_seed"] = int(text) if text else None
        except Exception:
            cfg["random_seed"] = None
        setattr(part, "heterogeneity_config", cfg)
        self.inspector._notify_change("part")

    def _set_part_material_field_value(self, part, group, key, value):
        cfg = normalize_material_field_config(copy.deepcopy(getattr(part, "material_field_config", {})))
        group_data = dict(cfg.get(group, {}) or {})
        if key == "expression":
            group_data[key] = str(value or "")
        elif key == "seed":
            text = str(value or "").strip()
            try:
                group_data[key] = int(text) if text else None
            except Exception:
                group_data[key] = None
        else:
            try:
                group_data[key] = float(value)
            except Exception:
                group_data[key] = 0.0
        cfg[group] = group_data
        setattr(part, "material_field_config", cfg)
        self.inspector._notify_change("part")

    def _part_material_base_props(self, part):
        merged = {}
        material = self.inspector._resolve_material(getattr(part, "material_id", None))
        if material is not None:
            merged.update(copy.deepcopy(getattr(material, "properties", {}) or {}))
        merged.update(copy.deepcopy(getattr(part, "material_props", {}) or {}))
        return material, merged

    def _reconfigure_part_material_schema(self, part, symmetry=None, behavior=None, damage=None):
        if symmetry is not None:
            part.material_symmetry = normalize_material_symmetry(symmetry)
        if behavior is not None:
            part.material_behavior = normalize_material_behavior(behavior)
        if damage is not None:
            part.material_damage = normalize_material_damage(damage)
        material, merged = self._part_material_base_props(part)
        source_type = getattr(material, "mat_type", None) if material is not None else getattr(part, "material_type", None)
        effective_type, prop_keys = _material_property_keys_for_behavior_symmetry(
            source_type,
            getattr(part, "material_symmetry", "isotropic"),
            getattr(part, "material_behavior", "elastic"),
            getattr(part, "material_damage", "none"),
        )
        part.material_type = effective_type
        part.material_props = normalize_material_properties(
            merged,
            getattr(part, "material_behavior", "elastic"),
            getattr(part, "material_symmetry", "isotropic"),
            getattr(part, "material_damage", "none"),
            preserve_unknown=False,
        )
        self.inspector._notify_change("part")

    def _set_part_material_reference(self, part, material_id):
        if material_id in ("", None):
            part.material_id = None
            part.material_type = None
            part.material_props = {}
            self.inspector._notify_change("part")
            return
        part.material_id = material_id
        material = self.inspector._resolve_material(material_id)
        if material is not None:
            part.material_symmetry = getattr(material, "symmetry", "isotropic")
            part.material_behavior = getattr(
                material,
                "behavior",
                infer_behavior_from_mat_type(getattr(material, "mat_type", "")),
            )
            part.material_damage = getattr(material, "damage", "none")
        self._reconfigure_part_material_schema(part)

    def _set_part_material_prop(self, part, key, value):
        props = copy.deepcopy(getattr(part, "material_props", {}) or {})
        props[str(key)] = value
        part.material_props = props
        self.inspector._notify_change("part")

    def populate(self, payload):
        super().populate(payload)
        self._dimension_list = None
        self._edit_dimension_button = None
        kind = str((payload or {}).get("kind", "") or "")
        selection_ctx = self.inspector._selection_dimension_context(payload)
        owner_type = selection_ctx.get("owner_type")
        owner_part = selection_ctx.get("owner_part")
        dimensions = selection_ctx.get("dimensions", [])
        selected_sketch_index = selection_ctx.get("sketch_index")
        if kind == "active_sketch":
            label = f"Sketch {int(selected_sketch_index) + 1}" if selected_sketch_index is not None and int(selected_sketch_index) >= 0 else "Active Sketch"
            self.set_summary(label)
            sketch_form = self.add_section("Sketch")
            self.add_readonly_row("Type", "Active Sketch", form_layout=sketch_form)
            if selected_sketch_index is not None and int(selected_sketch_index) >= 0:
                self.add_readonly_row("Index", int(selected_sketch_index) + 1, form_layout=sketch_form)
            self._populate_dimensions_section(owner_type, owner_part, dimensions)
            self.finalize_form()
            return
        if kind == "part_sketch":
            if owner_part is None:
                self.set_summary("Selected sketch is no longer available.")
                self.finalize_form()
                return
            sketch_label = f"Sketch {int(selected_sketch_index) + 1}" if selected_sketch_index is not None and int(selected_sketch_index) >= 0 else "Sketch"
            self.set_summary(f"{getattr(owner_part, 'name', 'Part')}: {sketch_label}")
            sketch_form = self.add_section("Sketch")
            self.add_readonly_row("Part", getattr(owner_part, "name", ""), form_layout=sketch_form)
            if selected_sketch_index is not None and int(selected_sketch_index) >= 0:
                self.add_readonly_row("Index", int(selected_sketch_index) + 1, form_layout=sketch_form)
            self._populate_dimensions_section(owner_type, owner_part, dimensions)
            self.finalize_form()
            return
        part = self.inspector._resolve_part(payload.get("part_id"))
        if part is None:
            self.set_summary("Selected part is no longer available.")
            self.finalize_form()
            return
        name = getattr(part, "name", f"Part {getattr(part, 'id', '?')}")
        self.set_summary(f"Part: {name}")
        basic_form = self.add_section("Basic Properties")
        self.add_readonly_row("ID", getattr(part, "id", ""), form_layout=basic_form)
        self.add_readonly_row("Type", getattr(part, "part_type", "solid"), form_layout=basic_form)
        self.add_text_row(
            "Name",
            getattr(part, "name", ""),
            lambda v: self.inspector._set_obj_attr(part, "name", v),
            form_layout=basic_form,
        )
        materials = [("Unassigned", None)]
        for serial, mat in sorted(
            self.inspector._material_store().items(),
            key=lambda item: str(getattr(item[1], "name", item[0])),
        ):
            materials.append((f"{getattr(mat, 'name', serial)} ({serial})", serial))
        self.add_combo_row(
            "Material",
            materials,
            getattr(part, "material_id", None),
            lambda v, obj=part: self._set_part_material_reference(obj, v),
            form_layout=basic_form,
        )
        classification_form = self.add_section("Material Classification")
        self.add_combo_row(
            "Assignment",
            [("Homogeneous", "homogeneous"), ("Heterogeneous", "heterogeneous"), ("Material Field", "material_field")],
            getattr(part, "material_assignment_mode", "homogeneous"),
            lambda v: self.inspector._set_obj_attr(part, "material_assignment_mode", v),
            form_layout=classification_form,
        )
        field_cfg = normalize_material_field_config(copy.deepcopy(getattr(part, "material_field_config", {})))
        self.add_combo_row(
            "Symmetry",
            material_symmetry_options(),
            getattr(part, "material_symmetry", "isotropic"),
            lambda v, obj=part: self._reconfigure_part_material_schema(obj, symmetry=v),
            form_layout=classification_form,
        )
        self.add_combo_row(
            "Behavior",
            material_behavior_options(),
            getattr(part, "material_behavior", "elastic"),
            lambda v, obj=part: self._reconfigure_part_material_schema(obj, behavior=v),
            form_layout=classification_form,
        )
        self.add_combo_row(
            "Damage",
            material_damage_options(),
            getattr(part, "material_damage", "none"),
            lambda v, obj=part: self._reconfigure_part_material_schema(obj, damage=v),
            form_layout=classification_form,
        )
        assignment_mode = str(getattr(part, "material_assignment_mode", "homogeneous"))
        effective_mat_type, effective_prop_keys = _material_property_keys_for_behavior_symmetry(
            getattr(part, "material_type", None),
            getattr(part, "material_symmetry", "isotropic"),
            getattr(part, "material_behavior", "elastic"),
            getattr(part, "material_damage", "none"),
        )
        _, merged_material_props = self._part_material_base_props(part)
        if assignment_mode == "heterogeneous":
            self.add_combo_row(
                "Method",
                [
                    ("Region Based", "region_based"),
                    ("Random Distribution", "random_distribution"),
                ],
                getattr(part, "heterogeneity_method", "region_based"),
                lambda v: self.inspector._set_obj_attr(part, "heterogeneity_method", v),
                form_layout=classification_form,
            )
            hetero_form = self.add_section("Heterogeneity")
            method = str(getattr(part, "heterogeneity_method", "region_based") or "region_based")
            cfg = normalize_heterogeneity_config(copy.deepcopy(getattr(part, "heterogeneity_config", {})))
            if method == "random_distribution":
                self.add_text_row(
                    "Fractions",
                    self._distribution_text(cfg),
                    lambda v, obj=part: self._set_part_random_distribution_text(obj, v),
                    form_layout=hetero_form,
                )
                self.add_text_row(
                    "Random Seed",
                    "" if cfg.get("random_seed") in (None, "") else str(cfg.get("random_seed")),
                    lambda v, obj=part: self._set_part_random_seed(obj, v),
                    form_layout=hetero_form,
                )
            elif method == "field_gradient_distribution":
                for key in FIELD_DISTRIBUTION_PROPERTY_KEYS:
                    self.add_text_row(
                        f"{key}(x,y)",
                        (cfg.get("expressions", {}) or {}).get(key, ""),
                        lambda v, obj=part, field_key=key: self._set_part_field_expression(obj, field_key, v),
                        form_layout=hetero_form,
                    )
            else:
                self.add_readonly_row(
                    "Info",
                    "Each region/part keeps its own assigned material during export.",
                    form_layout=hetero_form,
                )
        elif assignment_mode == "material_field":
            self.add_combo_row(
                "Field Property",
                [("Young's Modulus", "E"), ("Density", "rho"), ("Poisson Ratio", "nu")],
                field_cfg.get("property_key", "E"),
                lambda v: self.inspector._set_obj_attr(part, "material_field_config", {**field_cfg, "property_key": v}),
                form_layout=classification_form,
            )
            self.add_combo_row(
                "Field Type",
                [
                    ("Linear Gradient", "linear_gradient"),
                    ("Radial Gradient", "radial_gradient"),
                    ("Random Field", "random_field"),
                    ("User Equation", "user_equation"),
                ],
                field_cfg.get("field_type", "linear_gradient"),
                lambda v: self.inspector._set_obj_attr(part, "material_field_config", {**field_cfg, "field_type": v}),
                form_layout=classification_form,
            )
            field_form = self.add_section("Material Field")
            property_key = str(field_cfg.get("property_key", "E"))
            field_type = str(field_cfg.get("field_type", "linear_gradient"))
            if field_type == "linear_gradient":
                linear = dict((field_cfg.get("linear_gradient", {}) or {}))
                self.add_float_row(
                    f"{property_key}_min",
                    linear.get("min", 0.0),
                    lambda v, obj=part: self._set_part_material_field_value(obj, "linear_gradient", "min", v),
                    form_layout=field_form,
                )
                self.add_float_row(
                    f"{property_key}_max",
                    linear.get("max", 0.0),
                    lambda v, obj=part: self._set_part_material_field_value(obj, "linear_gradient", "max", v),
                    form_layout=field_form,
                )
                self.add_combo_row(
                    "Direction",
                    [("X", "x"), ("Y", "y"), ("Diagonal", "diag")],
                    linear.get("direction", "x"),
                    lambda v, obj=part: self._set_part_material_field_value(obj, "linear_gradient", "direction", v),
                    form_layout=field_form,
                )
            elif field_type == "radial_gradient":
                radial = dict((field_cfg.get("radial_gradient", {}) or {}))
                for label, key in (
                    ("Center X", "center_x"),
                    ("Center Y", "center_y"),
                    ("Radius", "radius"),
                    (f"{property_key}_core", "core"),
                    (f"{property_key}_shell", "shell"),
                ):
                    self.add_float_row(
                        label,
                        radial.get(key, 0.0),
                        lambda v, obj=part, field_key=key: self._set_part_material_field_value(
                            obj, "radial_gradient", field_key, v
                        ),
                        form_layout=field_form,
                    )
            elif field_type == "random_field":
                rnd = dict((field_cfg.get("random_field", {}) or {}))
                for label, key in (
                    (f"{property_key}_mean", "mean"),
                    (f"{property_key}_std", "std"),
                    ("Correlation Length", "correlation_length"),
                ):
                    self.add_float_row(
                        label,
                        rnd.get(key, 0.0),
                        lambda v, obj=part, field_key=key: self._set_part_material_field_value(
                            obj, "random_field", field_key, v
                        ),
                        form_layout=field_form,
                    )
                self.add_text_row(
                    "Seed",
                    "" if rnd.get("seed") in (None, "") else str(rnd.get("seed")),
                    lambda v, obj=part: self._set_part_material_field_value(obj, "random_field", "seed", v),
                    form_layout=field_form,
                )
            else:
                self.add_text_row(
                    f"{property_key}(x,y)",
                    str((field_cfg.get("user_equation", {}) or {}).get("expression", "") or ""),
                    lambda v, obj=part: self._set_part_material_field_value(obj, "user_equation", "expression", v),
                    form_layout=field_form,
                )
        if getattr(part, "material_id", None) not in (None, ""):
            effective_form = self.add_section("Effective Material Properties")
            self.add_readonly_row(
                "Schema",
                f"{behavior_label(getattr(part, 'material_behavior', 'elastic'))} / "
                f"{str(getattr(part, 'material_symmetry', 'isotropic')).title()} / "
                f"{damage_label(getattr(part, 'material_damage', 'none'))} / {effective_mat_type}",
                form_layout=effective_form,
            )
            self.add_readonly_row(
                "Base Material",
                getattr(self.inspector._resolve_material(getattr(part, "material_id", None)), "name", "Assigned"),
                form_layout=effective_form,
            )
            for key in effective_prop_keys:
                self.add_property_value_row(
                    key.replace("_", " ").title(),
                    merged_material_props.get(key, _default_material_property_value(key)),
                    lambda v, obj=part, prop_key=key: self._set_part_material_prop(obj, prop_key, v),
                    form_layout=effective_form,
                )
        state_form = None
        for attr in ("is_void", "is_rigid"):
            if hasattr(part, attr):
                if state_form is None:
                    state_form = self.add_section("State")
                self.add_bool_row(
                    attr.replace("_", " ").title(),
                    getattr(part, attr),
                    lambda v, a=attr: self.inspector._set_obj_attr(part, a, v),
                    form_layout=state_form,
                )
        self._populate_dimensions_section(owner_type, owner_part, dimensions)
        if self.inspector._is_generated_feature_part(part):
            feature_form = self.add_section("Generated Feature")
            feature_settings = getattr(part, "generated_feature_settings", None) or {}
            particle_count = len(getattr(part, "particles", []) or [])
            if particle_count:
                self.add_readonly_row("Particle Count", particle_count, form_layout=feature_form)
                self.add_readonly_row(
                    "Hint",
                    "Particles are generated from geometry. Edit sketch to modify.",
                    form_layout=feature_form,
                )
            self.add_readonly_row(
                "Type",
                str(getattr(part, "generated_feature_kind", "") or "porous_particles").replace("_", " ").title(),
                form_layout=feature_form,
            )
            if feature_settings:
                for key in ("shape", "distribution", "layout", "lattice", "size_min", "size_max", "size_mean", "size_std", "spacing", "spacing_y"):
                    if key in feature_settings:
                        self.add_readonly_row(
                            key.replace("_", " ").title(),
                            feature_settings.get(key),
                            form_layout=feature_form,
                        )
            action_group, action_layout = self.add_section_container("Feature Actions")
            edit_button = QPushButton("Edit Feature")
            edit_button.clicked.connect(lambda _=False, p=part: self.inspector._edit_generated_feature(p))
            action_layout.addWidget(edit_button)
        self.finalize_form()

    def eventFilter(self, watched, event):
        viewport = self._dimension_list.viewport() if self._dimension_list is not None else None
        if watched is viewport and event is not None and event.type() == QEvent.Leave:
            self._highlight_dimension_item(self._dimension_list.currentItem() if self._dimension_list is not None else None)
        return super().eventFilter(watched, event)

    def _populate_dimensions_section(self, owner_type, owner_part, dimensions):
        if not dimensions:
            self.inspector._set_dimension_highlight(None)
            return

        _, section_layout = self.add_section_container("Dimensions")
        hint = QLabel("Hover to highlight. Click to select. Double-click to edit on sketch.")
        hint.setObjectName("MinorStatusLabel")
        hint.setWordWrap(True)
        section_layout.addWidget(hint)

        self._dimension_list = QListWidget(self)
        self._dimension_list.setMouseTracking(True)
        self._dimension_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._dimension_list.itemEntered.connect(self._highlight_dimension_item)
        self._dimension_list.currentItemChanged.connect(lambda current, _previous: self._highlight_dimension_item(current))
        self._dimension_list.itemDoubleClicked.connect(self._edit_dimension_item)
        self._dimension_list.viewport().installEventFilter(self)

        current_item = None
        selected_dim_id = getattr(self.inspector.sketch_view, "selected_dimension_id", None)
        for dim in dimensions:
            dim_id = dim.get("id")
            label = self.inspector._dimension_display_text(dim, owner_type, owner_part)
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, dim_id)
            item.setToolTip(label)
            self._dimension_list.addItem(item)
            if dim_id == selected_dim_id:
                current_item = item
        section_layout.addWidget(self._dimension_list, 1)

        self._edit_dimension_button = QPushButton("Edit on Sketch")
        self._edit_dimension_button.clicked.connect(self._edit_current_dimension)
        section_layout.addWidget(self._edit_dimension_button)

        if current_item is not None:
            self._dimension_list.setCurrentItem(current_item)
        elif self._dimension_list.count() > 0:
            self._dimension_list.setCurrentRow(0)
        self._update_dimension_button_state()

    def _dimension_id_from_item(self, item):
        if item is None:
            return None
        try:
            dim_id = item.data(Qt.UserRole)
            return int(dim_id) if dim_id is not None else None
        except Exception:
            return None

    def _highlight_dimension_item(self, item):
        self.inspector._set_dimension_highlight(self._dimension_id_from_item(item))
        self._update_dimension_button_state()

    def _update_dimension_button_state(self):
        if self._edit_dimension_button is None:
            return
        self._edit_dimension_button.setEnabled(self._dimension_id_from_item(self._dimension_list.currentItem() if self._dimension_list is not None else None) is not None)

    def _edit_current_dimension(self):
        item = self._dimension_list.currentItem() if self._dimension_list is not None else None
        if item is not None:
            self._edit_dimension_item(item)

    def _edit_dimension_item(self, item):
        dim_id = self._dimension_id_from_item(item)
        if dim_id is None:
            return
        try:
            if self.inspector.sketch_view.begin_inline_dimension_edit(dim_id):
                return
        except Exception:
            pass
        try:
            self.inspector.sketch_view.edit_dimension(dim_id)
        except Exception:
            return
        self.inspector.refresh_current_selection()


class MaterialPropertiesPanel(_InspectorFormPage):
    def _reconfigure_material_schema(self, material, *, behavior=None, symmetry=None, damage=None):
        if behavior is not None:
            material.behavior = normalize_material_behavior(behavior)
        else:
            material.behavior = normalize_material_behavior(
                getattr(material, "behavior", infer_behavior_from_mat_type(getattr(material, "mat_type", "")))
            )
        if symmetry is not None:
            material.symmetry = normalize_material_symmetry(symmetry)
        else:
            material.symmetry = normalize_material_symmetry(getattr(material, "symmetry", "isotropic"))
        if damage is not None:
            material.damage = normalize_material_damage(damage)
        else:
            material.damage = normalize_material_damage(getattr(material, "damage", "none"))
        material.mat_type = legacy_mat_type_for_behavior(material.behavior, getattr(material, "mat_type", None))
        material.properties = normalize_material_properties(
            copy.deepcopy(getattr(material, "properties", {}) or {}),
            material.behavior,
            material.symmetry,
            material.damage,
            preserve_unknown=False,
        )
        self.inspector._notify_change("material")

    def populate(self, payload):
        super().populate(payload)
        material = self.inspector._resolve_material(payload.get("serial"))
        if material is None:
            self.set_summary("Selected material is no longer available.")
            self.finalize_form()
            return
        self.set_summary(f"Material: {getattr(material, 'name', 'Material')}")
        behavior = normalize_material_behavior(
            getattr(material, "behavior", infer_behavior_from_mat_type(getattr(material, "mat_type", "")))
        )
        damage = normalize_material_damage(getattr(material, "damage", "none"))
        meta_form = self.add_section("Material")
        self.add_text_row(
            "Name",
            getattr(material, "name", ""),
            lambda value, obj=material: self.inspector._set_obj_attr(obj, "name", value),
            form_layout=meta_form,
        )
        self.add_combo_row(
            "Behavior",
            material_behavior_options(),
            behavior,
            lambda value, obj=material: self._reconfigure_material_schema(obj, behavior=value),
            form_layout=meta_form,
        )
        self.add_combo_row(
            "Symmetry",
            material_symmetry_options(),
            getattr(material, "symmetry", "isotropic"),
            lambda value, obj=material: self._reconfigure_material_schema(obj, symmetry=value),
            form_layout=meta_form,
        )
        if damage != "none":
            self.add_combo_row(
                "Damage",
                material_damage_options(),
                damage,
                lambda value, obj=material: self._reconfigure_material_schema(obj, damage=value),
                form_layout=meta_form,
            )
        props = getattr(material, "properties", {}) or {}
        _mat_type, _prop_keys = _material_property_keys_for_behavior_symmetry(
            getattr(material, "mat_type", ""),
            getattr(material, "symmetry", "isotropic"),
            behavior,
            damage,
        )
        prop_schema = []
        prop_data = {}
        for field in material_property_schema(behavior, getattr(material, "symmetry", "isotropic"), damage):
            key = field["key"]
            value = props.get(key, field.get("default", 0.0))
            field = dict(field)
            field["type"] = _material_property_field_type(value)
            prop_schema.append(field)
            prop_data[key] = value
        values_group, values_layout = self.add_section_container("Parameters")
        values_layout.addWidget(
            PropertyEditor(
                prop_schema,
                prop_data,
                parent=self,
                on_change=lambda key, value, obj=material: self.inspector._set_material_prop(obj, key, value),
            )
        )
        self.finalize_form()


class _EntryPropertiesPanel(_InspectorFormPage):
    def __init__(self, inspector, entry_kind, parent=None):
        super().__init__(inspector, parent)
        self.entry_kind = str(entry_kind)

    def populate(self, payload):
        super().populate(payload)
        entry = self.inspector._resolve_entry(self.entry_kind, payload.get("index"))
        if entry is None:
            self.set_summary("Selected entry is no longer available.")
            self.finalize_form()
            return
        label = "Boundary Condition" if self.entry_kind == "bc" else "Load"
        entry_label = self._entry_label(entry)
        self.set_summary(f"{label}: {entry_label}")
        basic_form = self.add_section(label)
        self.add_text_row(
            "Name",
            str(entry.get("name", "") or entry_label),
            lambda value, obj=entry: self.inspector._set_entry_value(obj, "name", value),
            form_layout=basic_form,
        )
        self.add_readonly_row("Type", entry.get("display_type") or entry.get("type", ""), form_layout=basic_form)
        self.add_readonly_row("Magnitude", self._entry_magnitude_text(entry), form_layout=basic_form)
        target_label = "Assigned Edges" if self.entry_kind == "bc" else "Assigned Target"
        self.add_readonly_row(target_label, self._entry_target_text(entry), form_layout=basic_form)
        if entry.get("ids"):
            self.add_readonly_row("Particle IDs", ", ".join(str(v) for v in entry.get("ids") or []), form_layout=basic_form)
        self.finalize_form()

    def _entry_label(self, entry):
        if not isinstance(entry, dict):
            return ""
        return str(
            entry.get("name")
            or entry.get("display_type")
            or entry.get("type")
            or ("Boundary Condition" if self.entry_kind == "bc" else "Load")
        )

    def _entry_target_text(self, entry):
        if not isinstance(entry, dict):
            return ""
        coords = entry.get("coords")
        if isinstance(coords, np.ndarray):
            coords = coords.tolist()
        if isinstance(coords, (list, tuple)):
            if len(coords) >= 2 and all(isinstance(v, (int, float)) for v in coords[:2]):
                return f"Point ({float(coords[0]):.4g}, {float(coords[1]):.4g})"
            if (
                len(coords) >= 2
                and isinstance(coords[0], (list, tuple, np.ndarray))
                and isinstance(coords[1], (list, tuple, np.ndarray))
            ):
                points = []
                for pt in coords:
                    try:
                        points.append(f"({float(pt[0]):.4g}, {float(pt[1]):.4g})")
                    except Exception:
                        continue
                if len(points) == 2:
                    return f"Edge {points[0]} -> {points[1]}"
                if points:
                    return "Polyline " + " -> ".join(points)
        if entry.get("part_id") is not None:
            return f"Part {entry.get('part_id')} boundary"
        ids = entry.get("ids")
        if ids:
            return f"{entry.get('target', 'selection')}: {len(ids)} particle(s)"
        return "Unassigned"

    def _entry_magnitude_text(self, entry):
        parts = []
        for key in ("val", "value", "m", "pressure", "gravity"):
            if entry.get(key) not in (None, ""):
                parts.append(f"{key}={entry.get(key)}")
        for key in ("fx", "fy", "fz", "vx", "vy", "vz"):
            if entry.get(key) not in (None, ""):
                try:
                    if abs(float(entry.get(key))) <= 0.0:
                        continue
                except Exception:
                    pass
                parts.append(f"{key.upper()}={entry.get(key)}")
        return ", ".join(parts) if parts else "0"


class BCPropertiesPanel(_EntryPropertiesPanel):
    def __init__(self, inspector, parent=None):
        super().__init__(inspector, "bc", parent)


class LoadPropertiesPanel(_EntryPropertiesPanel):
    def __init__(self, inspector, parent=None):
        super().__init__(inspector, "load", parent)


class InteractionPropertiesPanel(_InspectorFormPage):
    @staticmethod
    def _interface_status_text(thickness, target_dx, material_id):
        warnings = []
        try:
            t = float(thickness)
        except Exception:
            t = 0.0
        try:
            dx = float(target_dx)
        except Exception:
            dx = 0.0
        if material_id in (None, "", -1):
            warnings.append("NoMaterial")
        if t <= 0:
            warnings.append("NoThickness")
        if dx <= 0:
            warnings.append("NoDX")
        if t > 0.0 and dx > 0.0:
            ratio = t / dx
            if ratio < 0.6 or ratio > 1.8:
                warnings.append(f"t/dx={ratio:.2f}")
        return "OK" if not warnings else "WARN:" + ",".join(warnings)

    def _apply_interface_settings(self, iface, editor):
        values = editor.values()
        _iface_set(iface, "name", str(values.get("name", "") or "").strip())
        _iface_set(iface, "friction_coeff", float(values.get("friction_coeff", 0.0) or 0.0))
        _iface_set(iface, "thickness", float(values.get("thickness", 0.0) or 0.0))
        _iface_set(iface, "target_dx", float(values.get("target_dx", 0.0) or 0.0))
        _iface_set(
            iface,
            "placement_mode",
            str(values.get("placement_mode") or getattr(Interface, "DEFAULT_PLACEMENT_MODE", "matrix_side")),
        )
        _iface_set(iface, "notes", str(values.get("notes", "") or "").strip())
        _iface_set(
            iface,
            "status",
            self._interface_status_text(
                _iface_get(iface, "thickness"),
                _iface_get(iface, "target_dx"),
                _iface_get(iface, "material_id"),
            ),
        )
        self.inspector._notify_change("interaction")

    def _apply_interface_material_parameters(self, material, editor):
        self.inspector._set_material_props(material, editor.values())

    def populate(self, payload):
        super().populate(payload)
        iface = self.inspector._resolve_interface(payload.get("index"))
        if iface is None:
            self.set_summary("Selected interaction is no longer available.")
            self.finalize_form()
            return
        part1 = self.inspector._resolve_part(getattr(iface, "part1_id", None))
        part2 = self.inspector._resolve_part(getattr(iface, "part2_id", None))
        title = getattr(iface, "name", "") or f"Interaction {getattr(iface, 'id', '?')}"
        self.set_summary(title)
        form = self.add_section("Interaction")
        self.add_readonly_row("Name", getattr(iface, "name", "") or "", form_layout=form)
        self.add_readonly_row("Type", getattr(iface, "interface_type", ""), form_layout=form)
        self.add_readonly_row(
            "Parts",
            f"{getattr(part1, 'name', getattr(iface, 'part1_id', '?'))} <-> "
            f"{getattr(part2, 'name', getattr(iface, 'part2_id', '?'))}",
            form_layout=form,
        )
        self.add_readonly_row("Material", _iface_get(iface, "material_id", ""), form_layout=form)
        self.add_readonly_row("Status", _iface_get(iface, "status", "") or "", form_layout=form)

        _, settings_layout = self.add_section_container("Interaction Settings")
        settings_hint = QLabel("Change values here, then click Apply Interaction Settings to commit them.")
        settings_hint.setObjectName("MinorStatusLabel")
        settings_hint.setWordWrap(True)
        settings_layout.addWidget(settings_hint)
        settings_schema = [
            {"name": "Name", "key": "name", "type": "string"},
            {"name": "Friction Coefficient", "key": "friction_coeff", "type": "float", "minimum": 0.0, "maximum": 2.0},
            {"name": "Thickness", "key": "thickness", "type": "float", "minimum": 0.0, "maximum": 1e12},
            {"name": "Target Spacing", "key": "target_dx", "type": "float", "minimum": 0.0, "maximum": 1e12},
            {
                "name": "Placement",
                "key": "placement_mode",
                "type": "enum",
                "options": [
                    (label, key)
                    for key, label in Interface.PLACEMENT_MODES.items()
                    if str(key) == "matrix_side"
                ],
            },
            {"name": "Notes", "key": "notes", "type": "string"},
        ]
        settings_data = {
            "name": _iface_get(iface, "name", "") or "",
            "friction_coeff": _iface_get(iface, "friction_coeff", 0.0),
            "thickness": _iface_get(iface, "thickness", 0.0) or 0.0,
            "target_dx": _iface_get(iface, "target_dx", 0.0) or 0.0,
            "placement_mode": _iface_get(
                iface,
                "placement_mode",
                getattr(Interface, "DEFAULT_PLACEMENT_MODE", "matrix_side"),
            ),
            "notes": _iface_get(iface, "notes", "") or "",
        }
        settings_editor = PropertyEditor(settings_schema, settings_data, parent=self)
        settings_layout.addWidget(settings_editor)
        apply_settings_btn = QPushButton("Apply Interaction Settings")
        apply_settings_btn.clicked.connect(
            lambda _checked=False, obj=iface, editor=settings_editor: self._apply_interface_settings(obj, editor)
        )
        settings_layout.addWidget(apply_settings_btn)

        material = self.inspector._resolve_material(_iface_get(iface, "material_id", None))
        material_info_form = self.add_section("Interface Material")
        self.add_readonly_row(
            "Mode",
            str(_iface_get(iface, "material_mode", "auto") or "auto").title(),
            form_layout=material_info_form,
        )
        self.add_readonly_row(
            "Linked Material",
            getattr(material, "name", _iface_get(iface, "material_id", "Unassigned")),
            form_layout=material_info_form,
        )
        if material is not None:
            props = getattr(material, "properties", {}) or {}
            behavior = normalize_material_behavior(
                getattr(material, "behavior", infer_behavior_from_mat_type(getattr(material, "mat_type", "")))
            )
            damage = normalize_material_damage(getattr(material, "damage", "none"))
            symmetry = getattr(material, "symmetry", "isotropic")
            _mat_type, _ = _material_property_keys_for_behavior_symmetry(
                getattr(material, "mat_type", ""),
                symmetry,
                behavior,
                damage,
            )
            self.add_readonly_row(
                "Schema",
                f"{behavior_label(behavior)} / {str(symmetry).title()} / {damage_label(damage)} / {_mat_type}",
                form_layout=material_info_form,
            )

            _, material_layout = self.add_section_container("Interface Material Parameters")
            material_hint = QLabel("Adjust interface material parameters here, then click Apply Material Parameters.")
            material_hint.setObjectName("MinorStatusLabel")
            material_hint.setWordWrap(True)
            material_layout.addWidget(material_hint)
            prop_schema = []
            prop_data = {}
            for field in material_property_schema(behavior, symmetry, damage):
                key = field["key"]
                value = props.get(key, field.get("default", 0.0))
                field = dict(field)
                field["type"] = _material_property_field_type(value)
                prop_schema.append(field)
                prop_data[key] = value
            prop_editor = PropertyEditor(prop_schema, prop_data, parent=self)
            material_layout.addWidget(prop_editor)
            apply_material_btn = QPushButton("Apply Material Parameters")
            apply_material_btn.clicked.connect(
                lambda _checked=False, obj=material, editor=prop_editor: self._apply_interface_material_parameters(
                    obj, editor
                )
            )
            material_layout.addWidget(apply_material_btn)
        else:
            self.add_readonly_row(
                "Parameters",
                "No linked material is assigned to this interaction yet.",
                form_layout=material_info_form,
            )
        self.finalize_form()


class ParticlePropertiesPanel(_InspectorFormPage):
    def populate(self, payload):
        super().populate(payload)
        self.set_summary("Particle Settings")
        _, basic_layout = self.add_section_container("Basic Properties")
        basic_layout.addWidget(
            PropertyEditor(
                [
                    {
                        "name": "Distribution",
                        "key": "mesh_distribution",
                        "type": "enum",
                        "options": [("Poisson Disk", "global_poisson")],
                    },
                    {
                        "name": "Backend",
                        "key": "mesh_backend",
                        "type": "enum",
                        "options": [("Auto", "auto"), ("CPU", "cpu"), ("GPU", "gpu")],
                    },
                ],
                {
                    "mesh_distribution": getattr(self.inspector.sketch_view, "mesh_distribution", "poisson"),
                    "mesh_backend": getattr(self.inspector.sketch_view, "mesh_backend", "auto"),
                },
                parent=self,
                on_change=lambda key, value: self.inspector._set_mesh_setting(key, value),
            )
        )
        global_nodes = getattr(self.inspector.sketch_view, "global_nodes", None)
        global_elements = getattr(self.inspector.sketch_view, "global_elements", None)
        summary_form = self.add_section("Target")
        self.add_readonly_row("Particles", len(global_nodes) if global_nodes is not None else 0, form_layout=summary_form)
        self.add_readonly_row("Connections", len(global_elements) if global_elements is not None else 0, form_layout=summary_form)
        self.finalize_form()


class PropertyInspectorPanel(QWidget):
    def __init__(self, sketch_view, parent=None, project_state=None):
        super().__init__(parent)
        self.sketch_view = sketch_view
        self.project_state = project_state
        self._selection_payload = None
        self._selection_kind = None
        self.setMinimumHeight(72)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(4)
        title = QToolButton()
        title.setProperty("dockIconButton", True)
        title.setIcon(get_icon("property_inspector", size=18))
        title.setToolButtonStyle(Qt.ToolButtonIconOnly)
        title.setToolTip("Property Inspector\nEdit properties of the selected object.")
        header_row.addWidget(title, 0, Qt.AlignLeft)
        header_row.addStretch(1)
        layout.addLayout(header_row)
        self.summary_label = QLabel("No selection")
        self.summary_label.setObjectName("SummaryLabel")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.stack = QStackedWidget(self)
        self.empty_panel = DefaultEmptyPanel(self)
        self.part_panel = PartPropertiesPanel(self, self)
        self.material_panel = MaterialPropertiesPanel(self, self)
        self.interaction_panel = InteractionPropertiesPanel(self, self)
        self.bc_panel = BCPropertiesPanel(self, self)
        self.load_panel = LoadPropertiesPanel(self, self)
        self.particle_panel = ParticlePropertiesPanel(self, self)
        for page in (
            self.empty_panel,
            self.part_panel,
            self.material_panel,
            self.interaction_panel,
            self.bc_panel,
            self.load_panel,
            self.particle_panel,
        ):
            self.stack.addWidget(page)
        layout.addWidget(self.stack, 1)

    def set_project_state(self, project_state):
        self.project_state = project_state
        if self._selection_payload is not None:
            self.set_selection_payload(self._selection_payload)
        else:
            self.clear_selection()

    def refresh_current_selection(self):
        if self._selection_payload is not None:
            self.set_selection_payload(self._selection_payload)
        else:
            self.clear_selection()

    def clear_selection(self):
        self.set_selection_payload(None)

    def set_selection_payload(self, payload):
        self._selection_payload = payload if isinstance(payload, dict) else None
        self._selection_kind = None
        page = self.empty_panel
        summary = "No selection"
        kind = None
        if not isinstance(payload, dict) or str(payload.get("kind", "")) not in {"part", "active_sketch", "part_sketch"}:
            self._set_dimension_highlight(None)
        if isinstance(payload, dict):
            kind = str(payload.get("kind", ""))
            self._selection_kind = kind
            if kind in {"", "category"}:
                self._selection_kind = None
                self.empty_panel.set_message("Select an object from the model tree or viewport.")
                summary = "No selection"
            elif kind in {"part", "active_sketch", "part_sketch"}:
                page = self.part_panel
            elif kind == "material":
                page = self.material_panel
            elif kind == "interaction":
                page = self.interaction_panel
            elif kind == "bc":
                page = self.bc_panel
            elif kind == "load":
                page = self.load_panel
            elif kind in {"mesh", "mesh_nodes", "mesh_elements", "particles"}:
                page = self.particle_panel
            else:
                self.empty_panel.set_message("No editable properties for this selection.")
            if page is not self.empty_panel:
                page.populate(payload)
                summary = getattr(page, "summary_text", summary)
            elif kind not in {"", "category"}:
                summary = "No editable properties for this selection."
        else:
            self.empty_panel.set_message("Select an object from the model tree or viewport.")
        if kind in {"bc", "load"}:
            entry = self._resolve_entry(kind, payload.get("index")) if isinstance(payload, dict) else None
            try:
                self.sketch_view.set_panel_attr_focus(kind, entry)
            except Exception:
                pass
        elif kind == "interaction":
            entry = self._resolve_interface(payload.get("index")) if isinstance(payload, dict) else None
            try:
                self.sketch_view.set_panel_attr_focus(kind, entry)
            except Exception:
                pass
        else:
            try:
                self.sketch_view.set_panel_attr_focus(None, None)
            except Exception:
                pass
        self.stack.setCurrentWidget(page)
        self.summary_label.setText(summary)
        window = self.window()
        if window is not None and hasattr(window, "_adjust_right_splitter_for_stage"):
            try:
                current_stage = getattr(getattr(window, "project_state", None), "current_stage", None)
                if current_stage is not None:
                    window._adjust_right_splitter_for_stage(current_stage)
            except Exception:
                pass
        if hasattr(window, "_update_interaction_hints"):
            window._update_interaction_hints()

    def _resolve_part(self, part_id):
        if part_id in (None, "", -1):
            return None
        parts = list(getattr(self.project_state, "parts", []) or getattr(self.sketch_view, "parts", []) or [])
        for part in parts:
            if int(getattr(part, "id", -1)) == int(part_id):
                return part
        return None

    def _set_dimension_highlight(self, dim_id):
        try:
            self.sketch_view.select_dimension(dim_id)
        except Exception:
            pass

    def _selection_dimension_context(self, payload):
        if not isinstance(payload, dict):
            return {"owner_type": None, "owner_part": None, "dimensions": [], "sketch_index": None}
        kind = str(payload.get("kind", "") or "")
        if kind == "active_sketch":
            try:
                sketch_index = int(payload.get("sketch_index", -1))
            except Exception:
                sketch_index = -1
            dimensions = list(getattr(self.sketch_view, "dimensions", []) or [])
            if sketch_index >= 0:
                dimensions = [dim for dim in dimensions if int(dim.get("sketch_index", -1)) == sketch_index]
            return {
                "owner_type": "sketch",
                "owner_part": None,
                "dimensions": dimensions,
                "sketch_index": sketch_index,
            }
        if kind == "part_sketch":
            part = self._resolve_part(payload.get("part_id"))
            try:
                sketch_index = int(payload.get("sketch_index", -1))
            except Exception:
                sketch_index = -1
            dimensions = list(getattr(part, "dimensions", []) or []) if part is not None else []
            if sketch_index >= 0:
                dimensions = [dim for dim in dimensions if int(dim.get("sketch_index", -1)) == sketch_index]
            return {
                "owner_type": "part",
                "owner_part": part,
                "dimensions": dimensions,
                "sketch_index": sketch_index,
            }
        part = self._resolve_part(payload.get("part_id"))
        owner_type, owner_part, dimensions = self._part_dimension_context(part)
        return {
            "owner_type": owner_type,
            "owner_part": owner_part,
            "dimensions": dimensions,
            "sketch_index": None,
        }

    def _part_dimension_context(self, part):
        if part is None:
            return "part", None, []
        try:
            edit_target = self.sketch_view.get_part_shape_edit_target()
        except Exception:
            edit_target = None
        try:
            part_id = int(getattr(part, "id", -1))
            edit_part_id = int(getattr(edit_target, "id", -2)) if edit_target is not None else None
        except Exception:
            part_id = getattr(part, "id", None)
            edit_part_id = getattr(edit_target, "id", None) if edit_target is not None else None
        if edit_part_id is not None and part_id == edit_part_id:
            return "sketch", None, list(getattr(self.sketch_view, "dimensions", []) or [])
        return "part", part, list(getattr(part, "dimensions", []) or [])

    def _format_dimension_value(self, dim_type, value):
        if value is None:
            return "?"
        if str(dim_type).lower() == "angle":
            return f"{float(value):.2f} deg"
        unit = str(getattr(self.sketch_view, "current_unit", "") or "").strip()
        suffix = f" {unit}" if unit else ""
        return f"{float(value):.3f}{suffix}"

    def _dimension_display_text(self, dim, owner_type, owner_part):
        dim_type = str((dim or {}).get("dim_type", "") or "").lower()
        dim_id = (dim or {}).get("id")
        label_map = {
            "linear": "Length",
            "point_distance": "Distance",
            "rect_width": "Width",
            "rect_height": "Height",
            "diameter": "Diameter",
            "radius": "Radius",
            "arc_length": "Arc Length",
            "slot_length": "Slot Length",
            "slot_width": "Slot Width",
            "polygon_radius": "Polygon Radius",
            "angle": "Angle",
        }
        try:
            current_value = self.sketch_view._dimension_current_value_ui(dim, owner_type, owner_part)
        except Exception:
            current_value = dim.get("value")
        value_text = self._format_dimension_value(dim_type, current_value)
        prefix = label_map.get(dim_type, dim_type.replace("_", " ").title() if dim_type else "Dimension")
        try:
            sketch_index = int(dim.get("sketch_index", -1))
        except Exception:
            sketch_index = -1
        sketch_text = f"Sketch {sketch_index + 1} | " if sketch_index >= 0 else ""
        id_text = f"#{dim_id} " if dim_id is not None else ""
        return f"{sketch_text}{id_text}{prefix} = {value_text}"

    def _is_generated_feature_part(self, part):
        try:
            return bool(self.sketch_view.is_generated_feature_part(part))
        except Exception:
            return False

    def _material_store(self):
        materials = getattr(self.project_state, "materials", {})
        if isinstance(materials, dict):
            return materials
        return {}

    def _resolve_material(self, serial):
        return self._material_store().get(serial)

    def _resolve_entry(self, kind, index):
        entries = []
        if kind == "bc":
            entries = list(getattr(self.project_state, "boundary_conditions", []) or [])
        elif kind == "load":
            entries = list(getattr(self.project_state, "loads", []) or [])
        if index is None:
            return None
        if 0 <= int(index) < len(entries):
            return entries[int(index)]
        return None

    def _resolve_interface(self, index):
        entries = list(getattr(self.project_state, "interfaces", []) or [])
        if index is None:
            return None
        if 0 <= int(index) < len(entries):
            return entries[int(index)]
        return None

    def _set_obj_attr(self, obj, attr, value):
        try:
            setattr(obj, attr, value)
        except Exception:
            return
        if self._selection_kind == "part":
            mirror = self._resolve_part(getattr(obj, "id", None))
            if mirror is not None and mirror is not obj:
                try:
                    setattr(mirror, attr, value)
                except Exception:
                    pass
        elif self._selection_kind == "material":
            mirror = self._resolve_material(getattr(obj, "serial", None))
            if mirror is not None and mirror is not obj:
                try:
                    setattr(mirror, attr, value)
                except Exception:
                    pass
        self._notify_change(self._selection_kind)

    def _edit_generated_feature(self, part):
        if part is None:
            return
        window = self.window()
        if window is None or not hasattr(window, "_open_porous_dialog"):
            return
        settings = copy.deepcopy(getattr(part, "generated_feature_settings", None) or {})
        if not settings:
            settings = copy.deepcopy(getattr(self.sketch_view, "porous_settings", None) or {})
        if not settings:
            settings = {"mode": "particles", "name_base": getattr(part, "name", "Feature")}
        settings["name_base"] = str(getattr(part, "name", settings.get("name_base", "Feature")))
        window._open_porous_dialog(initial_settings=settings, target_part=part)

    def _set_material_prop(self, material, key, value):
        try:
            props = getattr(material, "properties", {}) or {}
            props[key] = value
            material.properties = props
        except Exception:
            return
        mirror = self._resolve_material(getattr(material, "serial", None))
        if mirror is not None and mirror is not material:
            try:
                mirror_props = getattr(mirror, "properties", {}) or {}
                mirror_props[key] = value
                mirror.properties = mirror_props
            except Exception:
                pass
        self._notify_change("material")
        if self._selection_kind == "interaction":
            try:
                self.refresh_current_selection()
            except Exception:
                pass

    def _set_material_props(self, material, values):
        try:
            props = copy.deepcopy(getattr(material, "properties", {}) or {})
            for key, value in dict(values or {}).items():
                props[str(key)] = value
            material.properties = props
        except Exception:
            return
        mirror = self._resolve_material(getattr(material, "serial", None))
        if mirror is not None and mirror is not material:
            try:
                mirror.properties = copy.deepcopy(props)
            except Exception:
                pass
        self._notify_change("material")
        if self._selection_kind == "interaction":
            try:
                self.refresh_current_selection()
            except Exception:
                pass

    def _set_entry_value(self, entry, key, value):
        if key == "part_id" and value in ("", None):
            entry[key] = None
        else:
            entry[key] = value
        self._notify_change(self._selection_kind)

    def _set_mesh_setting(self, key, value):
        setattr(self.sketch_view, key, value)
        settings = getattr(self.project_state, "solver_settings", None)
        if isinstance(settings, dict):
            settings[key] = value
        self._notify_change("mesh")

    def _notify_change(self, kind):
        panel = getattr(self.window(), "properties_panel", None)
        if kind == "part":
            try:
                self.sketch_view.partsChanged.emit()
            except Exception:
                pass
            if panel is not None and hasattr(panel, "materials_tab"):
                try:
                    panel.materials_tab._update_selected_part_display(
                        getattr(self.sketch_view, "selected_part_id", None)
                    )
                except Exception:
                    pass
            if panel is not None and hasattr(panel, "assembly_tab"):
                try:
                    panel.assembly_tab.refresh()
                except Exception:
                    pass
            try:
                self.refresh_current_selection()
            except Exception:
                pass
        elif kind == "material":
            try:
                self.sketch_view.materialsChanged.emit()
                self.sketch_view.partsChanged.emit()
            except Exception:
                pass
            if panel is not None and hasattr(panel, "materials_tab") and hasattr(panel.materials_tab, "refresh_material_list"):
                try:
                    panel.materials_tab.refresh_material_list()
                except Exception:
                    pass
            if panel is not None and hasattr(panel, "assembly_tab"):
                try:
                    panel.assembly_tab.refresh()
                except Exception:
                    pass
            try:
                self.refresh_current_selection()
            except Exception:
                pass
        elif kind == "interaction":
            try:
                self.sketch_view.interfacesChanged.emit()
            except Exception:
                pass
            if panel is not None and hasattr(panel, "interfaces_tab"):
                try:
                    panel.interfaces_tab.refresh_list()
                except Exception:
                    pass
            try:
                self.refresh_current_selection()
            except Exception:
                pass
        elif kind == "bc":
            self.sketch_view.bcs = copy.deepcopy(getattr(self.project_state, "boundary_conditions", []))
            try:
                self.sketch_view.bcsChanged.emit()
            except Exception:
                pass
            if panel is not None and hasattr(panel, "bcs_tab"):
                try:
                    panel.bcs_tab.refresh_lists()
                except Exception:
                    pass
        elif kind == "load":
            self.sketch_view.loads = copy.deepcopy(getattr(self.project_state, "loads", []))
            try:
                self.sketch_view.loadsChanged.emit()
            except Exception:
                pass
            if panel is not None and hasattr(panel, "loads_tab"):
                try:
                    panel.loads_tab.refresh_lists()
                except Exception:
                    pass
        elif kind == "mesh":
            if panel is not None and hasattr(panel, "mesh_tab") and hasattr(panel.mesh_tab, "refresh"):
                try:
                    panel.mesh_tab.refresh()
                except Exception:
                    pass
        try:
            self.sketch_view.redraw()
        except Exception:
            pass
        window = self.window()
        if window is not None and hasattr(window, "project_tree"):
            try:
                window.project_tree.refresh_from_model()
            except Exception:
                pass
        if self._selection_payload is not None:
            try:
                self.set_selection_payload(self._selection_payload)
            except Exception:
                pass
        if window is not None and hasattr(window, "_update_interaction_hints"):
            try:
                window._update_interaction_hints()
            except Exception:
                pass


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.Window)
        self.setWindowTitle("CPD SimStudio v25")
        self.setMinimumSize(960, 640)
        self._status_timeout_ms = 4000
        try:
            app = QApplication.instance()
            if app is not None:
                app.installEventFilter(self)
        except Exception:
            pass

        # --- Project tracking ---
        self.current_project_file = None
        self.recent_projects = []
        self.recent_file_store = str(get_recent_projects_file())
        self.project_dirty = False
        self.project_mode = "2d"
        self._update_worker = None
        self._update_check_manual = False
        self.model3d = self._new_model3d()
        self._workspace_3d = False
        self._gizmo_mode = "translate"
        self._undo_stack_3d = []
        self._redo_stack_3d = []
        self._max_undo_3d = 50
        self._in_3d_drag = False
        self._mesh_regen_timer = QTimer(self)
        self._mesh_regen_timer.setSingleShot(True)
        self._mesh_regen_timer.timeout.connect(self._generate_gmsh_mesh)
        self._gmsh_progress = None
        self._gmsh_pending = False
        self._gmsh_in_progress = False
        self._ui_ready = False
        self._view_3d_ready = False
        self._orientation_widget_enabled_3d = True
        self._view_navigation_enabled = False
        self._primitive_highlight_style = "solid"
        self._primitive_highlight_color = "amber"

        # --- Session Projects ---
        self.session_projects = {}   # {project_name: project_data_snapshot}
        self.active_project_name = None        

        # --- Core Layout ---
        self.scene = QGraphicsScene(-SCENE_W/2, -SCENE_H/2, SCENE_W, SCENE_H)
        self.view = SketchView(self.scene)
        self.view.set_project_mode(self.project_mode)
        self.project_state = ProjectState()
        self.project_state.analysis_type = "static"
        self.project_state.dimension = "2D"
        self.view.project_state = self.project_state
        self.event_bus = EventBus()
        self.command_bus = CommandBus(self.event_bus)
        self.command_bus.add_middleware(self._command_execution_middleware)
        self.event_bus.subscribe("command.failed", self._on_command_failed)
        self.event_bus.subscribe("command.completed", self._on_command_completed)
        self.view_3d = None
        self.view_stack = QStackedWidget()
        self.view_stack.addWidget(self.view)
        self.view_stack.setCurrentIndex(0)
        self.point_view = None
        self._results_point_preview_active = False

        # --- Right-side Properties Panel ---
        self.properties_panel = PropertiesPanel(self.view, project_state=self.project_state)
        self.properties_panel.bcs_tab.set_viewport(None)
        self.properties_panel.bcs_tab.set_workspace_mode(self._workspace_3d)
        self.properties_panel.loads_tab.set_viewport(None)
        self.properties_panel.loads_tab.set_workspace_mode(self._workspace_3d)
        self.property_inspector = PropertyInspectorPanel(self.view, project_state=self.project_state)
        self.property_inspector.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._init_controllers()
        
        central_widget = QWidget()
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        canvas_widget = QWidget()
        self._canvas_widget = canvas_widget
        canvas_layout = QVBoxLayout(canvas_widget)
        canvas_layout.addWidget(self.view_stack, 1)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(0)
        # Mini-map: floating bird's-eye widget anchored top-right of the
        # canvas. Click/drag to recenter the main view. Hidden by default;
        # toggle from View menu (Ctrl+M).
        self._mini_map = MiniMapWidget(self.view, canvas_widget)
        self._mini_map.hide()
        canvas_widget.installEventFilter(self)
        # Selection mini-toolbar — quick actions next to the selected part.
        self._selection_toolbar = SelectionMiniToolbar(self.view, self)
        # Canvas status bar — slim strip at the bottom of the viewport showing
        # cursor position, current tool, zoom level, and selection count.
        # Updated live from sketch_view via signals.
        self._canvas_status_bar = QWidget()
        self._canvas_status_bar.setObjectName("CanvasStatusBar")
        self._canvas_status_bar.setFixedHeight(22)
        _csb_layout = QHBoxLayout(self._canvas_status_bar)
        _csb_layout.setContentsMargins(8, 0, 8, 0)
        _csb_layout.setSpacing(16)
        self._status_xy_label = QLabel("X: 0.0   Y: 0.0")
        self._status_xy_label.setObjectName("CanvasStatusItem")
        self._status_tool_label = QLabel("Tool: Select")
        self._status_tool_label.setObjectName("CanvasStatusItem")
        self._status_zoom_label = QLabel("Zoom: 100%")
        self._status_zoom_label.setObjectName("CanvasStatusItem")
        self._status_sel_label = QLabel("0 selected")
        self._status_sel_label.setObjectName("CanvasStatusItem")
        _csb_layout.addWidget(self._status_xy_label)
        _csb_layout.addWidget(self._status_tool_label)
        _csb_layout.addStretch(1)
        _csb_layout.addWidget(self._status_sel_label)
        _csb_layout.addWidget(self._status_zoom_label)
        canvas_layout.addWidget(self._canvas_status_bar, 0)
        # Wire the sketch view's signals to update the status bar.
        if hasattr(self.view, "cursorScenePositionChanged"):
            self.view.cursorScenePositionChanged.connect(self._on_canvas_cursor_moved)
        if hasattr(self.view, "toolChanged"):
            self.view.toolChanged.connect(self._on_canvas_tool_changed)
        if hasattr(self.view, "zoomChanged"):
            self.view.zoomChanged.connect(self._on_canvas_zoom_changed)
        if hasattr(self.view, "partsChanged"):
            self.view.partsChanged.connect(self._update_status_selection)
        if hasattr(self.view, "partSelectionChanged"):
            self.view.partSelectionChanged.connect(self._update_status_selection)

        model_panel = QWidget()
        model_panel.setProperty("card", True)
        model_layout = QVBoxLayout(model_panel)
        model_layout.setContentsMargins(2, 2, 2, 2)
        model_layout.setSpacing(2)
        # Header row: title on the left, × close button on the right.
        nav_header = QWidget()
        nav_header_layout = QHBoxLayout(nav_header)
        nav_header_layout.setContentsMargins(2, 0, 2, 0)
        nav_header_layout.setSpacing(2)
        model_title = QLabel("Project Navigator")
        model_title.setObjectName("SectionTitleLabel")
        nav_header_layout.addWidget(model_title, 1)
        self._nav_close_button = QToolButton()
        self._nav_close_button.setObjectName("PanelCloseButton")
        self._nav_close_button.setText("✕")  # ✕
        self._nav_close_button.setToolTip("Close Project Navigator (Ctrl+B)")
        self._nav_close_button.setMinimumSize(22, 22)
        self._nav_close_button.setMaximumSize(24, 24)
        self._nav_close_button.setCursor(Qt.PointingHandCursor)
        self._nav_close_button.clicked.connect(lambda: self._toggle_left_panel(False))
        nav_header_layout.addWidget(self._nav_close_button, 0)
        model_layout.addWidget(nav_header)
        self.project_tree = ProjectTree(model_panel, sketch_view=self.view, project_state=self.project_state)
        self.project_tree.objectSelected.connect(self.property_inspector.set_selection_payload)
        self.project_tree.setMinimumWidth(0)
        self.project_tree.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        model_layout.addWidget(self.project_tree, 1)
        model_panel.setMinimumWidth(80)
        model_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.view.partsChanged.connect(self.project_tree.refresh_from_model)
        self.view.materialsChanged.connect(self.project_tree.refresh_from_model)
        self.view.interfacesChanged.connect(self.project_tree.refresh_from_model)
        self.view.bcsChanged.connect(self.project_tree.refresh_from_model)
        self.view.loadsChanged.connect(self.project_tree.refresh_from_model)
        # Re-fit navigator width whenever content changes.
        self.view.partsChanged.connect(self._sync_nav_panel_width)
        self.view.materialsChanged.connect(self._sync_nav_panel_width)
        self.view.interfacesChanged.connect(self._sync_nav_panel_width)
        self.view.bcsChanged.connect(self._sync_nav_panel_width)
        self.view.loadsChanged.connect(self._sync_nav_panel_width)
        if hasattr(self.view, "partSelectionChanged"):
            self.view.partSelectionChanged.connect(self._on_view_part_selected)
        if hasattr(self.view, "animationFramesLoaded"):
            self.view.animationFramesLoaded.connect(self.project_tree.refresh_from_model)
            self.view.animationFramesLoaded.connect(lambda *_: self._update_interaction_hints())

        properties_dock = QWidget()
        properties_dock.setProperty("card", True)
        properties_dock.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        properties_dock.setMaximumWidth(290)
        properties_dock_layout = QVBoxLayout(properties_dock)
        properties_dock_layout.setContentsMargins(2, 2, 2, 2)
        properties_dock_layout.setSpacing(2)
        # The property inspector is now embedded inside PropertiesPanel as a
        # "Metadata" section instead of being a separate panel below.
        self.properties_panel.set_metadata_panel(self.property_inspector)
        properties_dock_layout.addWidget(self.properties_panel, 1)

        # Edge-rail buttons — appear when a side panel is hidden so the user
        # can re-show it with one click. Thin (16 px wide), full panel height.
        self._left_rail = QToolButton()
        self._left_rail.setObjectName("PanelEdgeRail")
        self._left_rail.setProperty("rail", "left")
        self._left_rail.setText("▶")
        self._left_rail.setToolTip("Show Project Navigator (Ctrl+B)")
        self._left_rail.setCursor(Qt.PointingHandCursor)
        self._left_rail.setFixedWidth(16)
        self._left_rail.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self._left_rail.clicked.connect(lambda: self._toggle_left_panel(True))
        self._left_rail.hide()

        self._right_rail = QToolButton()
        self._right_rail.setObjectName("PanelEdgeRail")
        self._right_rail.setProperty("rail", "right")
        self._right_rail.setText("◀")
        self._right_rail.setToolTip("Show Properties Panel (Ctrl+J)")
        self._right_rail.setCursor(Qt.PointingHandCursor)
        self._right_rail.setFixedWidth(16)
        self._right_rail.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self._right_rail.clicked.connect(lambda: self._toggle_right_panel(True))
        self._right_rail.hide()

        # Wire the right-panel close button from PropertiesPanel.
        self.properties_panel.closeRequested.connect(lambda: self._toggle_right_panel(False))

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._left_rail)
        splitter.addWidget(model_panel)
        splitter.addWidget(canvas_widget)
        splitter.addWidget(properties_dock)
        splitter.addWidget(self._right_rail)
        # Side panels (and their rails) do NOT stretch; only the centre canvas
        # grows with the window. Rails are at index 0 and 4 (outer edges).
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 5)
        splitter.setStretchFactor(3, 0)
        splitter.setStretchFactor(4, 0)
        splitter.setHandleWidth(6)
        splitter.setChildrenCollapsible(False)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(4, False)
        splitter.setObjectName("MainSplitter")
        self.properties_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.property_inspector.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        default_panel_w = 290
        # Compute the initial navigator width from actual content so no label
        # is clipped on first show.  The scrollbar width (~15px) is included so
        # the scrollbar never obscures the rightmost characters.
        _vsb_w = self.project_tree.verticalScrollBar().sizeHint().width()
        default_tree_w = max(
            self.project_tree._compute_preferred_width() + _vsb_w + 4, 80
        )
        default_canvas_w = max(200, int(VIEW_W) - default_panel_w - default_tree_w - 32)
        # Splitter children: [left_rail, model_panel, canvas, properties_dock, right_rail]
        splitter.setSizes([0, default_tree_w, default_canvas_w, default_panel_w, 0])
        root_layout.addWidget(splitter, 1)
        self._main_splitter = splitter
        self._properties_dock = properties_dock
        self._model_panel = model_panel
        # Bottom-splitter inspector replaced by embedded "Metadata" section.
        self._right_splitter = None

        command_bar = QWidget()
        command_bar.setObjectName("CommandBar")
        command_layout = QVBoxLayout(command_bar)
        command_layout.setContentsMargins(
            UI_TOKENS.spacing_md,
            UI_TOKENS.spacing_xs,
            UI_TOKENS.spacing_md,
            UI_TOKENS.spacing_xs,
        )
        command_layout.setSpacing(UI_TOKENS.spacing_xs)
        command_row = QHBoxLayout()
        command_row.setContentsMargins(0, 0, 0, 0)
        command_label = QLabel("Command:")
        command_label.setObjectName("CommandLabel")
        self.command_input = QLineEdit()
        self.command_input.setObjectName("CommandInput")
        self.command_input.setPlaceholderText(
            "Type a command (ex: line 0 0 100 0). Use help for a list."
        )
        self.command_input.setMinimumHeight(26)
        self.command_input.returnPressed.connect(self._handle_command)
        self.command_input.textChanged.connect(self._update_command_hint)
        self._command_hints = {
            "line": "line x y x y OR line @dx @dy",
            "rect": "rect x y w h OR rect w h",
            "circle": "circle x y r OR circle r",
            "slot": "slot x y x y w OR slot x y w",
            "polygon": "polygon n x y r OR polygon n r",
            "polyline": "polyline x y x y [...]",
            "confirm": "confirm",
            "cut": "cut",
            "undo": "undo",
            "redo": "redo",
            "snap": "snap grid on|off OR snap endpoints on|off",
        }
        command_templates = list(self._command_hints.keys()) + ["help", "?"]
        completer = QCompleter(command_templates, self)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        completer.setCompletionMode(QCompleter.InlineCompletion)
        self.command_input.setCompleter(completer)
        self.command_status = QLabel("")
        self.command_status.setObjectName("CommandStatus")
        self.command_status.setWordWrap(True)
        command_row.addWidget(command_label)
        command_row.addWidget(self.command_input, 1)
        command_layout.addLayout(command_row)
        command_layout.addWidget(self.command_status)
        root_layout.addWidget(command_bar)
        self.command_bar = command_bar
        self._command_bar_anim = None
        self._command_fade_anim = None
        self._command_opacity_effect = QGraphicsOpacityEffect(self.command_bar)
        self.command_bar.setGraphicsEffect(self._command_opacity_effect)
        self._command_opacity_effect.setOpacity(1.0)
        self._update_command_hint("")
        self.command_bar.setVisible(False)

        self.setCentralWidget(central_widget)
        self.statusBar().showMessage("Ready", self._status_timeout_ms)
        self._settings = QSettings("CPD-Modeller", "CPD-SimStudio")
        self._stage_hints_enabled = self._settings.value(
            "ui/stage_hints_enabled",
            True,
            type=bool,
        )
        self._orientation_widget_enabled_3d = self._settings.value(
            "3d/orientation_widget_enabled", True, type=bool
        )
        self._view_navigation_enabled = self._settings.value(
            "3d/navigation_enabled", False, type=bool
        )
        if hasattr(self, "orientation_widget_checkbox"):
            try:
                self.orientation_widget_checkbox.blockSignals(True)
                self.orientation_widget_checkbox.setChecked(bool(self._orientation_widget_enabled_3d))
            finally:
                try:
                    self.orientation_widget_checkbox.blockSignals(False)
                except Exception:
                    pass
        highlight_style = self._settings.value("3d/primitive_highlight_style", "solid", type=str)
        highlight_color = self._settings.value("3d/primitive_highlight_color", "amber", type=str)
        if str(highlight_style).lower() in {"solid", "strong_edge", "wireframe"}:
            self._primitive_highlight_style = str(highlight_style).lower()
        if str(highlight_color).lower() in {"amber", "cyan", "lime", "magenta", "red"}:
            self._primitive_highlight_color = str(highlight_color).lower()
        startup_prompt_version = self._settings.value("startup_prompt_version", 0, type=int)
        if startup_prompt_version < 1:
            self._settings.setValue("show_startup", True)
            self._settings.setValue("startup_prompt_version", 1)
        self._mode_tip_shown = False
        self.stage_hints_toggle_btn = QToolButton(self)
        self.stage_hints_toggle_btn.setCheckable(True)
        self.stage_hints_toggle_btn.setText("")
        self.stage_hints_toggle_btn.setIcon(get_icon("help", size=16))
        self.stage_hints_toggle_btn.setIconSize(QSize(16, 16))
        self.stage_hints_toggle_btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.stage_hints_toggle_btn.setFixedSize(24, 24)
        self.stage_hints_toggle_btn.setToolTip("Stage Hints\nToggle stage helper hints.")
        self.stage_hints_toggle_btn.toggled.connect(
            lambda enabled: self._set_stage_hints_enabled(enabled, announce=True)
        )
        self.stage_hints_toggle_btn.blockSignals(True)
        self.stage_hints_toggle_btn.setChecked(bool(self._stage_hints_enabled))
        self.stage_hints_toggle_btn.blockSignals(False)
        self.statusBar().addPermanentWidget(self.stage_hints_toggle_btn)
        self.mode_indicator = QLabel("")
        self.mode_indicator.setObjectName("ModeIndicator")
        self.statusBar().addPermanentWidget(self.mode_indicator)
        self.workflow_stage_indicator = QLabel("")
        self.workflow_stage_indicator.setObjectName("ModeIndicator")
        self.statusBar().addPermanentWidget(self.workflow_stage_indicator)
        self.workflow_stage_indicator.hide()
        self.interaction_mode_indicator = QLabel("")
        self.interaction_mode_indicator.setObjectName("ModeIndicator")
        self.statusBar().addPermanentWidget(self.interaction_mode_indicator)
        self.nav_indicator = QLabel("")
        self.nav_indicator.setObjectName("MinorStatusLabel")
        self.statusBar().addPermanentWidget(self.nav_indicator)
        self.selection_indicator = QLabel("")
        self.selection_indicator.setObjectName("MinorStatusLabel")
        self.statusBar().addPermanentWidget(self.selection_indicator)
        self.node_count_indicator = QLabel("")
        self.node_count_indicator.setObjectName("MinorStatusLabel")
        self.statusBar().addPermanentWidget(self.node_count_indicator)
        self.element_count_indicator = QLabel("")
        self.element_count_indicator.setObjectName("MinorStatusLabel")
        self.statusBar().addPermanentWidget(self.element_count_indicator)
        self.units_indicator = QLabel("")
        self.units_indicator.setObjectName("MinorStatusLabel")
        self.statusBar().addPermanentWidget(self.units_indicator)
        self.mesh_stats_indicator = QLabel("")
        self.mesh_stats_indicator.setObjectName("MinorStatusLabel")
        self.statusBar().addPermanentWidget(self.mesh_stats_indicator)
        self.mesh_stats_indicator.hide()
        self.mouse_hint_label = QLabel("")
        self.mouse_hint_label.setObjectName("MinorStatusLabel")
        self.statusBar().addPermanentWidget(self.mouse_hint_label, 1)
        self._app = QApplication.instance()
        self._shutdown_hooks_connected = False
        self._current_theme = "light"
        self._toolbar_style = ""
        self._apply_theme("light")
        self.RECENT_FILES_PATH = os.path.join(os.path.dirname(__file__), "recent_files.json")
        self.recent_files = []
        self._create_menu_bar()
        self._create_primitive_dock()
        self._create_sketch_toolbar()
        self._create_workflow_ribbon()
        self._create_main_toolbar()
        self._create_loads_toolbar()
        self._sync_mode_ui()
        self._sync_project_dimension_state()
        self._set_view_navigation_enabled(bool(getattr(self, "_view_navigation_enabled", False)), persist=False)
        self._set_precision_sketch_mode(False, announce=False)
        self.properties_panel.tabs.currentChanged.connect(self._sync_workflow_ribbon_from_tab)
        self._update_workflow_ribbon_state(ProjectStage.GEOMETRY, preferred_key="geometry")
        self._refresh_workflow_architecture()

        # 3D view is created lazily when needed.

        self.view.partsChanged.connect(self.properties_panel.assembly_tab.refresh)
        self.view.materialsChanged.connect(self.properties_panel.assembly_tab.refresh)
        self.view.interfacesChanged.connect(self.properties_panel.assembly_tab.refresh)
        self.view.bcsChanged.connect(self.properties_panel.assembly_tab.refresh)
        self.view.loadsChanged.connect(self.properties_panel.assembly_tab.refresh)
        self.view.partsChanged.connect(self._mark_dirty)
        self.view.partSelectionChanged.connect(self.properties_panel.assembly_tab.select_part)
        self.view.partsChanged.connect(self._refresh_workflow_architecture)
        self.view.materialsChanged.connect(self._refresh_workflow_architecture)
        self.view.interfacesChanged.connect(self._refresh_workflow_architecture)
        self.view.partsChanged.connect(self._update_interaction_hints)
        self.view.partsChanged.connect(self.properties_panel.interfaces_tab.refresh_list)
        self.view.interfacesChanged.connect(self.properties_panel.interfaces_tab.refresh_list)
        self.properties_panel.nextStageRequested.connect(self.advance_to_next_stage)
        self.properties_panel.prevStageRequested.connect(self.retreat_to_prev_stage)
        self.view.stageAdvanceRequested.connect(self.advance_stage)
        self.view.geometryChanged.connect(self._mark_dirty)
        self.view.materialsChanged.connect(self._mark_dirty)
        self.view.interfacesChanged.connect(self._mark_dirty)
        self.view.bcsChanged.connect(self._mark_dirty)
        self.view.loadsChanged.connect(self._mark_dirty)
        self.view.materialsChanged.connect(self._update_interaction_hints)
        self.view.interfacesChanged.connect(self._update_interaction_hints)
        self.view.bcsChanged.connect(self._update_interaction_hints)
        self.view.loadsChanged.connect(self._update_interaction_hints)
        self.view.bcsChanged.connect(self.properties_panel.bcs_tab.refresh_lists)
        self.view.loadsChanged.connect(self.properties_panel.loads_tab.refresh_lists)
        self.properties_panel.tabs.currentChanged.connect(lambda *_: self._update_interaction_hints())

        self.change_module("Part")

        # Auto-load last project
        # self._load_last_project()

        self.apply_stage_ui(ProjectStage.GEOMETRY)
        self._start_maximized = True
        self._startup_state_request_id = 0
        if self._app is not None and not self._shutdown_hooks_connected:
            try:
                self._app.aboutToQuit.connect(self._stop_background_threads)
                self._shutdown_hooks_connected = True
            except Exception:
                pass
        QTimer.singleShot(50, self._run_startup_prompts)
        self._init_autosave()
        self._sync_workspace_ui()
        QTimer.singleShot(0, self._mark_ui_ready)

        # Hide the main status bar — its info is already shown in the
        # canvas status bar and toolbar, so the bottom strip is redundant.
        self.statusBar().hide()

    def _init_controllers(self):
        self.geometry_controller = GeometryController(self)
        self.particle_controller = ParticleController(self)
        self.material_controller = MaterialController(self)
        self.interaction_controller = InteractionController(self)
        self.bc_controller = BCController(self)
        self.solver_controller = SolverController(self)
        self.results_controller = ResultsController(self)
        for controller in (
            self.material_controller,
            self.bc_controller,
            self.particle_controller,
            self.solver_controller,
        ):
            register = getattr(controller, "register_command_handlers", None)
            if callable(register):
                register(self.command_bus)
        if hasattr(self, "properties_panel") and hasattr(self.properties_panel, "materials_tab"):
            try:
                self.properties_panel.materials_tab.set_material_controller(self.material_controller)
            except Exception:
                pass
        if hasattr(self, "properties_panel") and hasattr(self.properties_panel, "bcs_tab"):
            try:
                self.properties_panel.bcs_tab.set_bc_controller(self.bc_controller)
            except Exception:
                pass
        if hasattr(self, "properties_panel") and hasattr(self.properties_panel, "loads_tab"):
            try:
                self.properties_panel.loads_tab.set_bc_controller(self.bc_controller)
            except Exception:
                pass
        if hasattr(self, "properties_panel") and hasattr(self.properties_panel, "job_tab"):
            try:
                self.properties_panel.job_tab.set_solver_controller(self.solver_controller)
            except Exception:
                pass
        if hasattr(self, "properties_panel") and hasattr(self.properties_panel, "results_tab"):
            try:
                self.properties_panel.results_tab.set_results_controller(self.results_controller)
            except Exception:
                pass
        try:
            self.results_controller.frameReady.connect(self.view.apply_loaded_animation_frame)
            self.results_controller.frameLoadStarted.connect(self._on_results_frame_load_started)
            self.results_controller.frameLoadFailed.connect(self._on_results_frame_load_failed)
            if hasattr(self.view, "replayParticleSelected"):
                self.view.replayParticleSelected.connect(self._on_replay_particle_selected)
        except Exception:
            pass
        self.stage_controllers = {
            controller.workflow_key: controller
            for controller in (
                self.geometry_controller,
                self.particle_controller,
                self.material_controller,
                self.interaction_controller,
                self.bc_controller,
                self.solver_controller,
                self.results_controller,
            )
        }
        self._stage_controller_by_stage = {
            ProjectStage.GEOMETRY: self.geometry_controller,
            ProjectStage.MESH: self.particle_controller,
            ProjectStage.MATERIALS: self.material_controller,
            ProjectStage.INTERFACES: self.interaction_controller,
            ProjectStage.BCS: self.bc_controller,
            ProjectStage.LOADS: self.bc_controller,
            ProjectStage.JOB: self.solver_controller,
            ProjectStage.RESULTS: self.results_controller,
        }

    def _mark_ui_ready(self):
        self._ui_ready = True
        try:
            self.view.redraw()
        except Exception:
            pass
        self._sync_point_preview(refresh_data=True)

    def _ensure_view_3d(self):
        if self.view_3d is not None and self._view_3d_ready:
            return self.view_3d
        self.view_3d = Mesh3DView()
        self.view_3d.gizmoMoved.connect(self._move_selected_primitive)
        self.view_3d.gizmoRotated.connect(self._rotate_selected_primitive)
        self.view_3d.gizmoScaled.connect(self._scale_selected_primitive)
        self.view_3d.gizmoDragStarted.connect(self._on_gizmo_drag_started)
        self.view_3d.gizmoDragFinished.connect(self._on_gizmo_drag_finished)
        self.view_3d.set_context_menu_hook(self._extend_3d_context_menu)
        self.view_3d.installEventFilter(self)
        self.view_3d.set_material_style("metal")
        self.view_3d.set_visibility(show_nodes=True, show_mesh=True)
        if hasattr(self.view_3d, "set_orientation_overlay_enabled"):
            self.view_3d.set_orientation_overlay_enabled(
                bool(getattr(self, "_orientation_widget_enabled_3d", True))
            )
        if hasattr(self.view_3d, "set_view_navigation_enabled"):
            self.view_3d.set_view_navigation_enabled(
                bool(getattr(self, "_view_navigation_enabled", False))
            )
        if hasattr(self, "view_stack"):
            self.view_stack.addWidget(self.view_3d)
        try:
            self.view.mesh3dUpdated.connect(self.view_3d.set_mesh)
        except Exception:
            pass
        try:
            self.view.mesh3dUpdated.connect(lambda *_: self._update_interaction_hints())
        except Exception:
            pass
        if hasattr(self.view_3d, "selectionChanged"):
            try:
                self.view_3d.selectionChanged.connect(self._update_interaction_hints)
            except Exception:
                pass
            try:
                self.view_3d.selectionChanged.connect(self._on_view3d_selection_changed)
            except Exception:
                pass
        if hasattr(self, "properties_panel"):
            viewport = self.view_3d if self._workspace_3d else None
            self.properties_panel.bcs_tab.set_viewport(viewport)
            self.properties_panel.loads_tab.set_viewport(viewport)
        self._view_3d_ready = True
        return self.view_3d

    def _call_view_3d(self, method_name, *args, **kwargs):
        view_3d = self._ensure_view_3d()
        if view_3d is None:
            return
        method = getattr(view_3d, method_name, None)
        if method:
            method(*args, **kwargs)

    def _prepare_modal_dialog(self, dialog):
        if dialog is None:
            return None
        try:
            dialog.setParent(self)
        except Exception:
            pass
        try:
            dialog.setWindowModality(Qt.ApplicationModal)
        except Exception:
            pass
        try:
            dialog.raise_()
            dialog.activateWindow()
        except Exception:
            pass
        return dialog

    def _init_autosave(self):
        self._autosave_enabled = True
        self._autosave_interval_ms = 30 * 1000
        self._autosave_session_id = str(int(time.time()))
        self._last_autosave_path = None
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(self._autosave_interval_ms)
        self._autosave_timer.timeout.connect(self._autosave)
        self._autosave_timer.start()

    def _bind_action_tip(self, action, tip):
        title = ""
        try:
            title = str(action.text() or "").replace("&", "").strip()
        except Exception:
            title = ""
        tooltip = f"{title}\n{tip}" if title and tip and title not in str(tip) else str(tip)
        action.setToolTip(tooltip)
        action.setStatusTip(str(tip))
        action.hovered.connect(lambda t=tip: self.statusBar().showMessage(t, self._status_timeout_ms))

    def _create_primitive_dock(self):
        dock = QDockWidget("3D Shapes", self)
        dock.setObjectName("PrimitiveDock")
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        dock.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetClosable
            | QDockWidget.DockWidgetFloatable
        )
        dock.setMinimumWidth(64)
        dock.setMaximumWidth(130)
        panel = QWidget()
        panel.setMinimumWidth(64)
        panel.setMaximumWidth(130)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(
            UI_TOKENS.spacing_xs,
            UI_TOKENS.spacing_xs,
            UI_TOKENS.spacing_xs,
            UI_TOKENS.spacing_xs,
        )
        layout.setSpacing(UI_TOKENS.spacing_sm)

        primitive_icon_px = primitive_icon_size()
        primitive_button_px = primitive_button_size()
        toolbar_icon_px = toolbar_icon_size()
        def icon_button(icon_name, tooltip):
            btn = QPushButton()
            btn.setObjectName("PrimitiveDockIconButton")
            btn.setIcon(get_icon(icon_name))
            btn.setIconSize(QSize(primitive_icon_px, primitive_icon_px))
            btn.setFixedSize(primitive_button_px, primitive_button_px)
            btn.setToolTip(tooltip)
            btn.setAccessibleName(tooltip)
            btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            return btn
        def icon_label(icon_name, tooltip, size=14):
            lbl = QLabel()
            lbl.setPixmap(get_icon(icon_name).pixmap(size, size))
            lbl.setToolTip(tooltip)
            return lbl
        def iconize_combo(combo, items, keep_text=False):
            combo.setIconSize(QSize(toolbar_icon_px, toolbar_icon_px))
            for idx, icon_name, tooltip in items:
                if idx < 0 or idx >= combo.count():
                    continue
                combo.setItemIcon(idx, get_icon(icon_name))
                if not keep_text:
                    combo.setItemText(idx, "")
                combo.setItemData(idx, tooltip, Qt.ToolTipRole)
        def make_section(title, content, expanded=True, icon_name=None, icon_only=False):
            header = QToolButton()
            header.setCheckable(True)
            header.setChecked(expanded)
            icon_box_w = UI_TOKENS.section_header_icon_box_w
            icon_box_h = UI_TOKENS.section_header_icon_box_h
            def build_header_icon(expanded_state):
                if not icon_name:
                    return QIcon()
                icon_px = get_icon(icon_name).pixmap(
                    UI_TOKENS.section_header_icon_size,
                    UI_TOKENS.section_header_icon_size,
                )
                px = QPixmap(icon_box_w, icon_box_h)
                px.fill(Qt.transparent)
                painter = QPainter(px)
                painter.setRenderHint(QPainter.Antialiasing)
                painter.setPen(QColor("#475569"))
                arrow = "▾" if expanded_state else "▸"
                painter.drawText(QRect(0, 0, 12, icon_box_h), Qt.AlignCenter, arrow)
                painter.drawPixmap(14, 1, icon_px)
                painter.end()
                return QIcon(px)
            if icon_name and icon_only:
                header.setIcon(build_header_icon(expanded))
                header.setIconSize(QSize(icon_box_w, icon_box_h))
            elif icon_name:
                header.setIcon(get_icon(icon_name))
                header.setIconSize(QSize(UI_TOKENS.section_header_icon_size, UI_TOKENS.section_header_icon_size))
            header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            header.setMinimumHeight(24)
            header.setMaximumHeight(24)
            header.setAutoRaise(True)
            header.setArrowType(Qt.NoArrow)
            header.setObjectName("SectionHeaderButton")
            if icon_only:
                header.setText("")
                header.setToolTip(title)
                header.setAccessibleName(title)
                header.setToolButtonStyle(Qt.ToolButtonIconOnly)
            else:
                header.setText(f"{'▾' if expanded else '▸'} {title}")
                header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            def toggle(checked):
                content.setVisible(checked)
                if icon_only:
                    if icon_name:
                        header.setIcon(build_header_icon(checked))
                else:
                    header.setText(f"{'▾' if checked else '▸'} {title}")
            header.toggled.connect(toggle)
            container = QWidget()
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(4)
            container_layout.addWidget(header)
            container_layout.addWidget(content)
            content.setVisible(expanded)
            return container

        shapes_container = QWidget()
        shapes_layout = FlowLayout(shapes_container, margin=0, h_spacing=6, v_spacing=6)
        shapes_container.setLayout(shapes_layout)
        shapes_scroll = QScrollArea()
        shapes_scroll.setWidgetResizable(True)
        shapes_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        shapes_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        shapes_scroll.setFrameShape(QScrollArea.NoFrame)
        shapes_scroll.setWidget(shapes_container)
        self.shape_scroll_area = shapes_scroll
        shapes_content = QWidget()
        shapes_content_layout = QVBoxLayout(shapes_content)
        shapes_content_layout.setContentsMargins(0, 0, 0, 0)
        shapes_content_layout.addWidget(shapes_scroll)
        self._shape_buttons = []
        def build_primitive_action(ptype, label, icon_name):
            action = QAction(get_icon(icon_name), label, self)
            action.triggered.connect(lambda _, n=ptype: self._add_primitive(n))
            return action

        def add_category(label, actions, icon_name=None):
            button = QToolButton(self)
            menu = QMenu(button)
            for act in actions:
                menu.addAction(act)
            button.setMenu(menu)
            button.setText("")
            button.setToolTip(label)
            button.setAccessibleName(label)
            if icon_name:
                button.setIcon(get_icon(icon_name))
            button.setObjectName("PrimitiveDockIconButton")
            button.setIconSize(QSize(primitive_icon_px, primitive_icon_px))
            button.setFixedSize(primitive_button_px, primitive_button_px)
            button.setToolButtonStyle(Qt.ToolButtonIconOnly)
            button.setPopupMode(QToolButton.InstantPopup)
            shapes_layout.addWidget(button)
            self._shape_buttons.append((button, label))
            return button

        primitive_actions = [
            build_primitive_action("box", "Box", "prim_box"),
            build_primitive_action("cylinder", "Cylinder", "prim_cylinder"),
            build_primitive_action("sphere", "Sphere", "prim_sphere"),
            build_primitive_action("cone", "Cone", "prim_cone"),
            build_primitive_action("ring", "Ring", "prim_ring"),
            build_primitive_action("extrude", "Extrude", "prim_box"),
            build_primitive_action("revolve", "Revolve", "prim_cylinder"),
        ]
        add_category("Primitives", primitive_actions, icon_name="prim_box")
        self._update_shape_button_labels(dock.width())
        layout.addWidget(make_section("Shapes", shapes_content, expanded=False, icon_name="prim_box", icon_only=True))

        objects_content = QWidget()
        objects_layout = QVBoxLayout(objects_content)
        objects_layout.setContentsMargins(0, 0, 0, 0)
        self.primitive_list = QListWidget()
        self.primitive_list.setObjectName("PrimitiveList")
        self.primitive_list.setMinimumWidth(0)
        self.primitive_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.primitive_list.itemSelectionChanged.connect(self._on_primitive_selection_changed)
        self.primitive_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.primitive_list.customContextMenuRequested.connect(self._show_primitive_material_menu)
        self.primitive_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        objects_layout.addWidget(self.primitive_list, 1)
        list_btn_row = QHBoxLayout()
        undo_btn = icon_button("undo", "Undo")
        redo_btn = icon_button("redo", "Redo")
        edit_btn = icon_button("edit", "Edit")
        delete_btn = icon_button("delete", "Delete")
        list_btn_row.addWidget(undo_btn)
        list_btn_row.addWidget(redo_btn)
        list_btn_row.addWidget(edit_btn)
        list_btn_row.addWidget(delete_btn)
        objects_layout.addLayout(list_btn_row)
        undo_btn.clicked.connect(self._undo_3d)
        redo_btn.clicked.connect(self._redo_3d)
        edit_btn.clicked.connect(self._edit_selected_primitive)
        delete_btn.clicked.connect(self._delete_selected_shapes)
        layout.addWidget(make_section("Objects", objects_content, expanded=False, icon_name="edit", icon_only=True))

        combine_content = QWidget()
        combine_layout = QVBoxLayout(combine_content)
        combine_layout.setContentsMargins(0, 0, 0, 0)
        bool_row = QVBoxLayout()
        union_btn = icon_button("boolean_union", "Union")
        cut_btn = icon_button("boolean_subtract", "Subtract")
        inter_btn = icon_button("boolean_intersect", "Intersect")
        bool_row.addWidget(union_btn)
        bool_row.addWidget(cut_btn)
        bool_row.addWidget(inter_btn)
        combine_layout.addLayout(bool_row)
        union_btn.clicked.connect(lambda: self._apply_boolean_op("union"))
        cut_btn.clicked.connect(lambda: self._apply_boolean_op("cut"))
        inter_btn.clicked.connect(lambda: self._apply_boolean_op("intersect"))

        self.show_operands_checkbox = QCheckBox("Show operands")
        self.show_operands_checkbox.setChecked(False)
        self.show_operands_checkbox.setText("")
        self.show_operands_checkbox.setIcon(get_icon("mesh_view"))
        self.show_operands_checkbox.setToolTip("Show operands")
        self.show_operands_checkbox.toggled.connect(lambda _: self._refresh_3d_view())
        combine_layout.addWidget(self.show_operands_checkbox)
        layout.addWidget(make_section("Combine", combine_content, expanded=False, icon_name="boolean_union", icon_only=True))

        grid_content = QWidget()
        grid_layout = QVBoxLayout(grid_content)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        snap_row = QVBoxLayout()
        self.snap_checkbox = QCheckBox("Snap to grid")
        self.snap_checkbox.setChecked(True)
        self.snap_checkbox.setText("")
        self.snap_checkbox.setIcon(get_icon("snap_grid"))
        self.snap_checkbox.setToolTip("Snap to grid")
        self.snap_size_spin = QDoubleSpinBox()
        self.snap_size_spin.setDecimals(2)
        self.snap_size_spin.setRange(0.1, 1e6)
        self.snap_size_spin.setValue(20.0)
        self.snap_size_spin.setSingleStep(1.0)
        self.snap_size_spin.setToolTip("Grid spacing")
        self.snap_size_spin.setMaximumWidth(90)
        snap_row.addWidget(self.snap_checkbox)
        grid_label = QLabel("")
        grid_label.setToolTip("Grid spacing")
        snap_row.addWidget(grid_label)
        snap_row.addWidget(self.snap_size_spin)
        grid_layout.addLayout(snap_row)
        self.snap_checkbox.toggled.connect(self._update_grid_settings)
        self.snap_size_spin.valueChanged.connect(self._update_grid_settings)
        layout.addWidget(make_section("Grid", grid_content, expanded=False, icon_name="snap_grid", icon_only=True))

        gizmo_content = QWidget()
        gizmo_layout = QVBoxLayout(gizmo_content)
        gizmo_layout.setContentsMargins(0, 0, 0, 0)
        gizmo_row = QVBoxLayout()
        self.gizmo_move_btn = QToolButton()
        self.gizmo_move_btn.setIcon(get_icon("gizmo_move"))
        self.gizmo_move_btn.setToolTip("Move (X/Y/Z handles)")
        self.gizmo_move_btn.setCheckable(True)
        self.gizmo_rotate_btn = QToolButton()
        self.gizmo_rotate_btn.setIcon(get_icon("gizmo_rotate"))
        self.gizmo_rotate_btn.setToolTip("Rotate (axis rings)")
        self.gizmo_rotate_btn.setCheckable(True)
        self.gizmo_scale_btn = QToolButton()
        self.gizmo_scale_btn.setIcon(get_icon("gizmo_scale"))
        self.gizmo_scale_btn.setToolTip("Scale (axis handles)")
        self.gizmo_scale_btn.setCheckable(True)
        for btn in (self.gizmo_move_btn, self.gizmo_rotate_btn, self.gizmo_scale_btn):
            btn.setIconSize(QSize(20, 20))
            btn.setAutoRaise(True)
        gizmo_row.addWidget(self.gizmo_move_btn)
        gizmo_row.addWidget(self.gizmo_rotate_btn)
        gizmo_row.addWidget(self.gizmo_scale_btn)
        gizmo_layout.addLayout(gizmo_row)
        self.gizmo_move_btn.clicked.connect(lambda: self._set_gizmo_mode("translate"))
        self.gizmo_rotate_btn.clicked.connect(lambda: self._set_gizmo_mode("rotate"))
        self.gizmo_scale_btn.clicked.connect(lambda: self._set_gizmo_mode("scale"))
        self._set_gizmo_mode("translate")
        move_hint = QLabel("Tip: Drag axis handles to move. Use Shift + drag for free move.")
        move_hint.setVisible(False)
        layout.addWidget(
            make_section("Transform", gizmo_content, expanded=False, icon_name="gizmo_move", icon_only=True)
        )

        material_content = QWidget()
        material_layout = QVBoxLayout(material_content)
        material_layout.setContentsMargins(0, 0, 0, 0)
        material_row = QHBoxLayout()
        material_row.addWidget(QLabel("Material"))
        self.material_combo = QComboBox()
        self.material_combo.addItems(["Silver", "Plastic", "Metal", "Clay", "Wireframe"])
        self.material_combo.setCurrentText("Metal")
        self.material_combo.setToolTip("Material style")
        iconize_combo(
            self.material_combo,
            [
                (0, "stage_materials", "Silver"),
                (1, "stage_materials", "Plastic"),
                (2, "stage_materials", "Metal"),
                (3, "stage_materials", "Clay"),
                (4, "stage_materials", "Wireframe"),
            ],
            keep_text=True,
        )
        self.material_combo.currentTextChanged.connect(self._on_material_style_changed)
        material_row.addWidget(self.material_combo)
        material_layout.addLayout(material_row)
        material_section = make_section(
            "Material", material_content, expanded=False, icon_name="stage_materials", icon_only=False
        )

        appearance_content = QWidget()
        appearance_layout = QVBoxLayout(appearance_content)
        appearance_layout.setContentsMargins(0, 0, 0, 0)
        bg_row = QHBoxLayout()
        bg_row.addWidget(QLabel("Background"))
        self.bg_preset_combo = QComboBox()
        self.bg_preset_combo.addItems(["Light"])
        self.bg_preset_combo.setCurrentText("Light")
        self.bg_preset_combo.setToolTip("Background preset")
        iconize_combo(self.bg_preset_combo, [(0, "frame", "Light")], keep_text=True)
        self.bg_preset_combo.currentTextChanged.connect(self._apply_3d_background_preset)
        bg_row.addWidget(self.bg_preset_combo, 1)
        appearance_layout.addLayout(bg_row)
        bg_btn_row = QHBoxLayout()
        self.bg_custom_btn = QPushButton("Custom...")
        self.bg_custom_btn.clicked.connect(self._pick_3d_background_color)
        self.bg_custom_btn.setVisible(False)
        self.bg_custom_btn.setEnabled(False)
        self.bg_reset_btn = QPushButton("Reset")
        self.bg_reset_btn.clicked.connect(lambda: self._apply_3d_background_preset("Light"))
        self.bg_reset_btn.setIcon(get_icon("undo"))
        self.bg_reset_btn.setToolTip("Reset background")
        bg_btn_row.addWidget(self.bg_custom_btn)
        bg_btn_row.addWidget(self.bg_reset_btn)
        appearance_layout.addLayout(bg_btn_row)
        obj_row = QHBoxLayout()
        obj_row.addWidget(QLabel("Object color"))
        self.obj_color_btn = QPushButton("Color")
        self.obj_color_btn.clicked.connect(self._pick_3d_object_color)
        self.obj_color_btn.setIcon(get_icon("edit"))
        self.obj_color_btn.setToolTip("Object color")
        self.obj_color_reset_btn = QPushButton("Reset")
        self.obj_color_reset_btn.clicked.connect(self._reset_3d_object_color)
        self.obj_color_reset_btn.setIcon(get_icon("undo"))
        self.obj_color_reset_btn.setToolTip("Reset object color")
        obj_row.addWidget(self.obj_color_btn)
        obj_row.addWidget(self.obj_color_reset_btn)
        appearance_layout.addLayout(obj_row)
        nav_row = QHBoxLayout()
        nav_row.addWidget(QLabel("Navigation"))
        self.orientation_widget_checkbox = QCheckBox("View Cube")
        self.orientation_widget_checkbox.setChecked(True)
        self.orientation_widget_checkbox.setToolTip("Show on-screen 3D orientation widget")
        self.orientation_widget_checkbox.toggled.connect(self._set_3d_orientation_widget_enabled)
        nav_row.addWidget(self.orientation_widget_checkbox)
        self.view_nav_checkbox = QCheckBox("View Navigation")
        self.view_nav_checkbox.setChecked(bool(getattr(self, "_view_navigation_enabled", False)))
        self.view_nav_checkbox.setToolTip(
            "Enable CAD-style navigation (middle-pan, right-rotate, wheel-zoom)."
        )
        self.view_nav_checkbox.toggled.connect(self._set_view_navigation_enabled)
        nav_row.addWidget(self.view_nav_checkbox)
        nav_row.addStretch(1)
        appearance_layout.addLayout(nav_row)
        appearance_section = make_section(
            "Appearance", appearance_content, expanded=False, icon_name="frame", icon_only=False
        )

        mesh_content = QWidget()
        mesh_layout = QVBoxLayout(mesh_content)
        mesh_layout.setContentsMargins(0, 0, 0, 0)
        mesh_row = QHBoxLayout()
        self.mesh_type_combo = QComboBox()
        self.mesh_type_combo.addItems(["Tetra", "Hex-dominant"])
        self.mesh_type_combo.setCurrentText("Tetra")
        self.mesh_type_combo.setToolTip("Generation type")
        iconize_combo(
            self.mesh_type_combo,
            [
                (0, "mesh_3d", "Tetra"),
                (1, "mesh_elements", "Hex-dominant"),
            ],
        )
        self.mesh_size_spin = QDoubleSpinBox()
        self.mesh_size_spin.setDecimals(3)
        self.mesh_size_spin.setRange(0.05, 1e6)
        self.mesh_size_spin.setValue(2.0)
        self.mesh_size_spin.setSingleStep(0.5)
        self.mesh_size_spin.setToolTip("Connection size")
        mesh_row.addWidget(icon_label("mesh_3d", "Generation type"))
        mesh_row.addWidget(self.mesh_type_combo, 1)
        mesh_row.addWidget(icon_label("dimension", "Connection size"))
        mesh_row.addWidget(self.mesh_size_spin)
        mesh_layout.addLayout(mesh_row)
        self.mesh_type_combo.currentTextChanged.connect(self._schedule_gmsh_regen)
        self.mesh_size_spin.valueChanged.connect(self._schedule_gmsh_regen)
        mesh_btn_row = QHBoxLayout()
        self.mesh_generate_btn = QPushButton("Generate Connections")
        self.mesh_generate_btn.setIcon(get_icon("mesh_preview"))
        self.mesh_generate_btn.setText("")
        self.mesh_generate_btn.setToolTip("Generate connections")
        self.mesh_generate_btn.clicked.connect(self._generate_gmsh_mesh)
        mesh_btn_row.addWidget(self.mesh_generate_btn)
        mesh_layout.addLayout(mesh_btn_row)
        mesh_vis_row = QHBoxLayout()
        mesh_vis_row.addWidget(icon_label("mesh_view", "Display"))
        self.mesh_display_combo = QComboBox()
        self.mesh_display_combo.addItems(["Particles + Connections", "Particles only", "Connections only"])
        self.mesh_display_combo.setToolTip("Display mode")
        iconize_combo(
            self.mesh_display_combo,
            [
                (0, "mesh_view", "Particles + Connections"),
                (1, "mesh_nodes", "Particles only"),
                (2, "mesh_elements", "Connections only"),
            ],
        )
        self.mesh_display_combo.currentTextChanged.connect(self._update_3d_mesh_visibility)
        mesh_vis_row.addWidget(self.mesh_display_combo, 1)
        mesh_layout.addLayout(mesh_vis_row)
        mesh_opts_row = QHBoxLayout()
        self.mesh_dim_checkbox = QCheckBox("Dim connection edges")
        self.mesh_dim_checkbox.setChecked(True)
        self.mesh_dim_checkbox.setText("")
        self.mesh_dim_checkbox.setIcon(get_icon("dimension"))
        self.mesh_dim_checkbox.setToolTip("Dim connection edges")
        self.mesh_dim_checkbox.toggled.connect(
            lambda checked: self._call_view_3d("set_mesh_dim", checked)
        )
        mesh_opts_row.addWidget(self.mesh_dim_checkbox)
        self.mesh_wireframe_checkbox = QCheckBox("Show wireframe edges")
        self.mesh_wireframe_checkbox.setChecked(False)
        self.mesh_wireframe_checkbox.setText("")
        self.mesh_wireframe_checkbox.setIcon(get_icon("mesh_elements"))
        self.mesh_wireframe_checkbox.setToolTip("Show wireframe edges")
        self.mesh_wireframe_checkbox.toggled.connect(
            lambda checked: self._call_view_3d("set_wireframe_visible", checked)
        )
        mesh_opts_row.addWidget(self.mesh_wireframe_checkbox)
        self.mesh_xray_checkbox = QCheckBox("X-ray connections")
        self.mesh_xray_checkbox.setChecked(False)
        self.mesh_xray_checkbox.setText("")
        self.mesh_xray_checkbox.setIcon(get_icon("mesh_xray"))
        self.mesh_xray_checkbox.setToolTip("X-ray connections")
        self.mesh_xray_checkbox.toggled.connect(
            lambda checked: self._call_view_3d("set_mesh_xray", checked)
        )
        mesh_opts_row.addWidget(self.mesh_xray_checkbox)
        mesh_opts_row.addStretch(1)
        mesh_layout.addLayout(mesh_opts_row)
        self.mesh_status_label = QLabel("")
        self.mesh_status_label.setObjectName("MinorStatusLabel")
        self.mesh_status_label.setWordWrap(True)
        mesh_layout.addWidget(self.mesh_status_label)
        self.mesh_quality_btn = QPushButton("Skewness")
        self.mesh_quality_btn.clicked.connect(self._show_skewness_distribution)
        self.mesh_quality_btn.setText("")
        self.mesh_quality_btn.setIcon(get_icon("mesh_quality"))
        self.mesh_quality_btn.setToolTip("Skewness")
        mesh_layout.addWidget(self.mesh_quality_btn)
        mesh_section = make_section(
            "Particle Generation (Gmsh)", mesh_content, expanded=False, icon_name="mesh_3d", icon_only=True
        )

        nodes_content = QWidget()
        nodes_layout = QVBoxLayout(nodes_content)
        nodes_layout.setContentsMargins(0, 0, 0, 0)
        nodes_opts_row = QHBoxLayout()
        self.show_nodes_checkbox = QCheckBox("Show particles")
        self.show_nodes_checkbox.setChecked(False)
        self.show_nodes_checkbox.setText("")
        self.show_nodes_checkbox.setIcon(get_icon("mesh_nodes"))
        self.show_nodes_checkbox.setToolTip("Show particles")
        self.show_nodes_checkbox.toggled.connect(
            lambda checked: self._call_view_3d("set_nodes_visible", show_nodes=checked)
        )
        nodes_opts_row.addWidget(self.show_nodes_checkbox)
        self.show_surface_nodes_checkbox = QCheckBox("Show surface particles")
        self.show_surface_nodes_checkbox.setChecked(True)
        self.show_surface_nodes_checkbox.setText("")
        self.show_surface_nodes_checkbox.setIcon(get_icon("mesh_nodes_surface"))
        self.show_surface_nodes_checkbox.setToolTip("Show surface particles")
        self.show_surface_nodes_checkbox.toggled.connect(
            lambda checked: self._call_view_3d("set_nodes_visible", show_surface=checked)
        )
        nodes_opts_row.addWidget(self.show_surface_nodes_checkbox)
        self.show_interior_nodes_checkbox = QCheckBox("Show interior particles")
        self.show_interior_nodes_checkbox.setChecked(True)
        self.show_interior_nodes_checkbox.setText("")
        self.show_interior_nodes_checkbox.setIcon(get_icon("mesh_nodes_interior"))
        self.show_interior_nodes_checkbox.setToolTip("Show interior particles")
        self.show_interior_nodes_checkbox.toggled.connect(
            lambda checked: self._call_view_3d("set_nodes_visible", show_interior=checked)
        )
        nodes_opts_row.addWidget(self.show_interior_nodes_checkbox)
        nodes_layout.addLayout(nodes_opts_row)
        node_size_row = QHBoxLayout()
        node_size_row.addWidget(icon_label("mesh_nodes", "Particle size"))
        self.node_size_slider = QSlider(Qt.Horizontal)
        self.node_size_slider.setRange(1, 10)
        self.node_size_slider.setValue(5)
        self.node_size_slider.setToolTip("Particle size")
        self.node_size_slider.valueChanged.connect(
            lambda value: self._call_view_3d("set_node_display", size=float(value))
        )
        node_size_row.addWidget(self.node_size_slider, 1)
        nodes_layout.addLayout(node_size_row)
        nodes_section = make_section(
            "Particles", nodes_content, expanded=False, icon_name="mesh_nodes", icon_only=True
        )

        layout.addStretch(1)
        dock.setWidget(panel)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        dock.setVisible(False)
        dock.setWindowTitle("")
        dock.setTitleBarWidget(QWidget())
        self.primitive_dock = dock
        dock.installEventFilter(self)
        self._refresh_primitive_list()
        self._material_section = material_section
        self._appearance_section = appearance_section
        self._mesh_section_3d = mesh_section
        self._nodes_section_3d = nodes_section
        self._attach_3d_sections()

        delete_action = QAction(self)
        delete_action.setShortcuts(["Delete", "Backspace"])
        delete_action.triggered.connect(self._handle_delete_action)
        self.addAction(delete_action)

    def _update_shape_button_labels(self, width):
        if not hasattr(self, "_shape_buttons"):
            return
        for btn, label in self._shape_buttons:
            btn.setText("")

    def _attach_3d_sections(self):
        if not hasattr(self, "properties_panel"):
            return
        if hasattr(self.properties_panel, "assembly_tab"):
            if hasattr(self.properties_panel.assembly_tab, "add_external_section"):
                if getattr(self, "_material_section", None) is not None:
                    self.properties_panel.assembly_tab.add_external_section(self._material_section)
                if getattr(self, "_appearance_section", None) is not None:
                    self.properties_panel.assembly_tab.add_external_section(self._appearance_section)
        if hasattr(self.properties_panel, "mesh_tab"):
            if hasattr(self.properties_panel.mesh_tab, "add_external_section"):
                if getattr(self, "_mesh_section_3d", None) is not None:
                    self.properties_panel.mesh_tab.add_external_section(self._mesh_section_3d)
                if getattr(self, "_nodes_section_3d", None) is not None:
                    self.properties_panel.mesh_tab.add_external_section(self._nodes_section_3d)
        self._set_3d_sections_visible(self._workspace_3d)

    def _set_3d_sections_visible(self, visible):
        for widget in (
            getattr(self, "_material_section", None),
            getattr(self, "_appearance_section", None),
            getattr(self, "_mesh_section_3d", None),
            getattr(self, "_nodes_section_3d", None),
        ):
            if widget is not None:
                widget.setVisible(bool(visible))
        try:
            if hasattr(self, "properties_panel") and hasattr(self.properties_panel, "assembly_tab"):
                self.properties_panel.assembly_tab.refresh_external_section_visibility()
            if hasattr(self, "properties_panel") and hasattr(self.properties_panel, "mesh_tab"):
                self.properties_panel.mesh_tab.refresh_external_section_visibility()
        except Exception:
            pass

    def eventFilter(self, obj, event):
        if event is not None and event.type() == QEvent.Wheel:
            if isinstance(obj, (QComboBox, QSpinBox, QDoubleSpinBox)):
                return True
        if obj is getattr(self, "primitive_dock", None) and event.type() == QEvent.Resize:
            self._update_shape_button_labels(obj.width())
        if obj is getattr(self, "_canvas_widget", None) and event.type() == QEvent.Resize:
            self._reposition_mini_map()
        if (
            obj is getattr(self, "view_3d", None)
            and event is not None
            and event.type() == QEvent.MouseButtonPress
        ):
            try:
                if event.button() == Qt.LeftButton and bool(getattr(self, "_workspace_3d", False)):
                    module_name = self._current_context_module()
                    if module_name in {"Part", "Property"}:
                        self._sync_3d_primitive_selection_from_context_event(event)
            except Exception:
                pass
        return super().eventFilter(obj, event)

    def _add_primitive(self, primitive_type):
        if not getattr(self.view, "cad_kernel", None) or not self.view.cad_kernel.available():
            self._show_cad_kernel_install_dialog()
            return
        params = self._prompt_primitive_params(primitive_type)
        if not params:
            return
        mouse_place = bool(params.pop("mouse_place", False))
        self._push_3d_undo()
        prim_id = self._next_model3d_id()
        entry = {
            "id": prim_id,
            "type": primitive_type,
            "params": params["params"],
            "transform": params["transform"],
        }
        self.model3d.setdefault("primitives", []).append(entry)
        self.project_dirty = True
        self._refresh_primitive_list()
        self._refresh_3d_view()
        if mouse_place:
            self._begin_mouse_place_primitive(prim_id, entry["transform"].get("tz", 0.0))

    def _prompt_primitive_params(self, primitive_type, existing=None):
        unit = getattr(self.view, "current_unit", "")
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Create {primitive_type.title()}")
        layout = QVBoxLayout(dlg)
        form = QFormLayout()

        def add_spin(label, value, minimum=-1e9, maximum=1e9):
            spin = QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setRange(minimum, maximum)
            spin.setValue(float(value))
            form.addRow(label, spin)
            return spin

        existing = existing or {}
        transform = existing.get("transform", {})
        params_in = existing.get("params", {})

        cx = add_spin(f"Center X ({unit})", transform.get("tx", 0.0))
        cy = add_spin(f"Center Y ({unit})", transform.get("ty", 0.0))
        cz = add_spin(f"Center Z ({unit})", transform.get("tz", 0.0))
        rx = add_spin("Rotate X (deg)", transform.get("rx", 0.0), -360.0, 360.0)
        ry = add_spin("Rotate Y (deg)", transform.get("ry", 0.0), -360.0, 360.0)
        rz = add_spin("Rotate Z (deg)", transform.get("rz", 0.0), -360.0, 360.0)

        params = {}
        if primitive_type == "box":
            w = add_spin(f"Width ({unit})", params_in.get("width", 20.0), 0.001, 1e9)
            d = add_spin(f"Depth ({unit})", params_in.get("depth", 20.0), 0.001, 1e9)
            h = add_spin(f"Height ({unit})", params_in.get("height", 20.0), 0.001, 1e9)
            params = {"width": w, "depth": d, "height": h}
        elif primitive_type == "cylinder":
            r = add_spin(f"Radius ({unit})", params_in.get("radius", 10.0), 0.001, 1e9)
            h = add_spin(f"Height ({unit})", params_in.get("height", 20.0), 0.001, 1e9)
            params = {"radius": r, "height": h}
        elif primitive_type == "sphere":
            r = add_spin(f"Radius ({unit})", params_in.get("radius", 10.0), 0.001, 1e9)
            params = {"radius": r}
        elif primitive_type == "cone":
            r1 = add_spin(f"Base radius ({unit})", params_in.get("radius_base", 10.0), 0.001, 1e9)
            r2 = add_spin(f"Top radius ({unit})", params_in.get("radius_top", 0.0), 0.0, 1e9)
            h = add_spin(f"Height ({unit})", params_in.get("height", 20.0), 0.001, 1e9)
            params = {"radius_base": r1, "radius_top": r2, "height": h}
        elif primitive_type == "ring":
            mr = add_spin(f"Major radius ({unit})", params_in.get("major_radius", 15.0), 0.001, 1e9)
            tr = add_spin(f"Tube radius ({unit})", params_in.get("tube_radius", 5.0), 0.001, 1e9)
            params = {"major_radius": mr, "tube_radius": tr}
        elif primitive_type == "extrude":
            w = add_spin(f"Profile width ({unit})", params_in.get("profile_width", 20.0), 0.001, 1e9)
            d = add_spin(f"Profile depth ({unit})", params_in.get("profile_depth", 20.0), 0.001, 1e9)
            h = add_spin(f"Extrude height ({unit})", params_in.get("height", 20.0), 0.001, 1e9)
            params = {"profile_width": w, "profile_depth": d, "height": h}
        elif primitive_type == "revolve":
            r = add_spin(f"Profile radius ({unit})", params_in.get("profile_radius", 10.0), 0.001, 1e9)
            h = add_spin(f"Profile height ({unit})", params_in.get("height", 20.0), 0.001, 1e9)
            params = {"profile_radius": r, "height": h}
        else:
            QMessageBox.warning(self, "3D Primitive", "Unsupported primitive type.")
            return None

        self._prepare_modal_dialog(dlg)
        layout.addLayout(form)
        place_label = "Place with mouse after creating"
        if existing is not None:
            place_label = "Place with mouse after closing"
        mouse_place = QCheckBox(place_label)
        mouse_place.setChecked(existing is None)
        layout.addWidget(mouse_place)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.Accepted:
            return None

        transform = {
            "tx": cx.value(),
            "ty": cy.value(),
            "tz": cz.value(),
            "rx": rx.value(),
            "ry": ry.value(),
            "rz": rz.value(),
        }
        out_params = {k: v.value() for k, v in params.items()}
        return {
            "transform": transform,
            "params": out_params,
            "mouse_place": mouse_place.isChecked(),
        }

    def _refresh_primitive_list(self):
        if not hasattr(self, "primitive_list"):
            return
        self._sync_model3d_state()
        self.primitive_list.clear()
        for prim in self.model3d.get("primitives", []):
            name = f"{prim.get('type', 'shape').title()} #{prim.get('id')}"
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, {"id": prim.get("id"), "kind": "primitive"})
            self.primitive_list.addItem(item)
        for op in self.model3d.get("operations", []):
            name = f"Result #{op.get('id')} ({op.get('op')})"
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, {"id": op.get("id"), "kind": "operation"})
            self.primitive_list.addItem(item)
        if hasattr(self, "properties_panel") and hasattr(self.properties_panel, "assembly_tab"):
            self.properties_panel.assembly_tab.refresh()

    def _assign_material_to_primitive(self, prim, material):
        if not prim or material is None:
            return
        self._push_3d_undo()
        prim["material_id"] = int(material.serial)
        prim["material_type"] = material.mat_type
        self.project_dirty = True
        self._refresh_primitive_list()
        if hasattr(self, "properties_panel") and hasattr(self.properties_panel, "assembly_tab"):
            self.properties_panel.assembly_tab.refresh()

    def _set_3d_view_visibility(self, show_nodes=None, show_mesh=None):
        view_3d = getattr(self, "view_3d", None)
        if view_3d is None:
            return
        if show_nodes is None:
            show_nodes = bool(getattr(view_3d, "show_nodes", True))
        if show_mesh is None:
            show_mesh = bool(getattr(view_3d, "show_mesh", True))
        view_3d.set_visibility(show_nodes=show_nodes, show_mesh=show_mesh)

    def _active_attr_panel(self):
        panel = getattr(self, "properties_panel", None)
        if panel is None or not hasattr(panel, "tabs"):
            return None
        idx = panel.tabs.currentIndex()
        if idx == self._stage_to_tab_index(ProjectStage.BCS) and hasattr(panel, "bcs_tab"):
            return panel.bcs_tab
        if idx == self._stage_to_tab_index(ProjectStage.LOADS) and hasattr(panel, "bcs_tab"):
            return panel.bcs_tab
        module = getattr(self.view, "active_module", "")
        if module == "Boundary" and hasattr(panel, "bcs_tab"):
            return panel.bcs_tab
        if module == "Load" and hasattr(panel, "bcs_tab"):
            return panel.bcs_tab
        if hasattr(panel, "bcs_tab"):
            return panel.bcs_tab
        if hasattr(panel, "loads_tab"):
            return panel.loads_tab
        return None

    def _set_3d_selection_target(self, label):
        attr_panel = self._active_attr_panel()
        if attr_panel is not None:
            combo = attr_panel.selection_target_combo
            idx = combo.findText(label, Qt.MatchFixedString)
            if idx >= 0:
                combo.setCurrentIndex(idx)
                self._update_interaction_hints()
                return
        if self.view_3d is not None:
            self.view_3d.set_selection_mode(str(label).lower())
        self._update_interaction_hints()

    def _on_view_part_selected(self, part_id):
        if part_id is None:
            if hasattr(self, "property_inspector"):
                self.property_inspector.clear_selection()
            if hasattr(self, "project_tree"):
                try:
                    self.project_tree.set_active_stage(getattr(self, "active_stage", ProjectStage.GEOMETRY))
                except Exception:
                    pass
            self._update_interaction_hints()
            return
        payload = {"kind": "part", "part_id": int(part_id), "stage": ProjectStage.GEOMETRY}
        if hasattr(self, "property_inspector"):
            self.property_inspector.set_selection_payload(payload)
        if hasattr(self, "project_tree") and hasattr(self.project_tree, "select_part"):
            try:
                # Selection came from the canvas — sync the tree but don't
                # let it re-fit/zoom the view to the part.
                self.project_tree.select_part(part_id, fit_view=False)
            except Exception:
                pass

    def _on_view3d_selection_changed(self):
        view_3d = getattr(self, "view_3d", None)
        inspector = getattr(self, "property_inspector", None)
        if view_3d is None or inspector is None:
            return
        try:
            faces = len(view_3d.get_selected_faces())
            edges = len(view_3d.get_selected_edges())
            nodes = len(view_3d.get_selected_nodes())
        except Exception:
            return
        if nodes > 0:
            inspector.set_selection_payload({"kind": "mesh_nodes", "stage": ProjectStage.MESH})
            return
        if faces > 0 or edges > 0:
            inspector.set_selection_payload({"kind": "mesh_elements", "stage": ProjectStage.MESH})
            return
        inspector.clear_selection()

    def _apply_3d_selection_with_type(self, label):
        attr_panel = self._active_attr_panel()
        if attr_panel is None:
            return
        combo = getattr(attr_panel, "paint_type_combo", None)
        if combo is not None:
            idx = combo.findText(str(label), Qt.MatchFixedString)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        attr_panel._apply_bc_from_selection()

    def _clear_3d_selection(self):
        view_3d = getattr(self, "view_3d", None)
        if view_3d is None:
            return
        if hasattr(view_3d, "clear_selection"):
            view_3d.clear_selection()
        if hasattr(view_3d, "clear_node_selection"):
            view_3d.clear_node_selection()
        if hasattr(view_3d, "clear_edge_selection"):
            view_3d.clear_edge_selection()

    def _apply_bc_from_toolbar(self):
        return self.bc_controller.apply_bc_from_toolbar()

    def _apply_load_from_toolbar(self):
        return self.bc_controller.apply_load_from_toolbar()

    def _on_unit_changed(self, unit):
        """Handle unit selection change from toolbar dropdown."""
        if hasattr(self, 'view') and hasattr(self.view, 'set_unit'):
            self.view.set_unit(unit)

    def _primitive_highlight_style_options(self):
        return [
            ("solid", "Solid Overlay"),
            ("strong_edge", "Strong Edge"),
            ("wireframe", "Wireframe Only"),
        ]

    def _primitive_highlight_color_options(self):
        return [
            ("amber", "Amber"),
            ("cyan", "Cyan"),
            ("lime", "Lime"),
            ("magenta", "Magenta"),
            ("red", "Red"),
        ]

    def _primitive_highlight_overlay_args(self):
        base_colors = {
            "amber": ((1.0, 0.92, 0.16), (0.98, 0.76, 0.05)),
            "cyan": ((0.10, 0.78, 1.0), (0.00, 0.62, 0.95)),
            "lime": ((0.50, 1.0, 0.18), (0.28, 0.92, 0.06)),
            "magenta": ((1.0, 0.35, 0.85), (0.95, 0.10, 0.72)),
            "red": ((1.0, 0.36, 0.26), (0.95, 0.14, 0.08)),
        }
        fill_rgb, edge_rgb = base_colors.get(
            getattr(self, "_primitive_highlight_color", "amber"),
            base_colors["amber"],
        )
        style = str(getattr(self, "_primitive_highlight_style", "solid")).lower()
        if style == "wireframe":
            return {
                "color": (fill_rgb[0], fill_rgb[1], fill_rgb[2], 0.0),
                "edge_color": (edge_rgb[0], edge_rgb[1], edge_rgb[2], 1.0),
                "draw_faces": False,
                "draw_edges": True,
            }
        if style == "strong_edge":
            return {
                "color": (fill_rgb[0], fill_rgb[1], fill_rgb[2], 0.08),
                "edge_color": (edge_rgb[0], edge_rgb[1], edge_rgb[2], 1.0),
                "draw_faces": True,
                "draw_edges": True,
            }
        return {
            "color": (fill_rgb[0], fill_rgb[1], fill_rgb[2], 0.14),
            "edge_color": (edge_rgb[0], edge_rgb[1], edge_rgb[2], 0.95),
            "draw_faces": True,
            "draw_edges": True,
        }

    def _set_primitive_highlight_style(self, style_key):
        style_key = str(style_key).lower()
        valid = {key for key, _label in self._primitive_highlight_style_options()}
        if style_key not in valid:
            return
        self._primitive_highlight_style = style_key
        if hasattr(self, "_settings"):
            self._settings.setValue("3d/primitive_highlight_style", style_key)
        self._update_3d_primitive_highlight()

    def _set_primitive_highlight_color(self, color_key):
        color_key = str(color_key).lower()
        valid = {key for key, _label in self._primitive_highlight_color_options()}
        if color_key not in valid:
            return
        self._primitive_highlight_color = color_key
        if hasattr(self, "_settings"):
            self._settings.setValue("3d/primitive_highlight_color", color_key)
        self._update_3d_primitive_highlight()

    def _add_3d_highlight_menu(self, menu):
        if menu is None:
            return
        submenu = menu.addMenu("Primitive Highlight")
        style_menu = submenu.addMenu("Style")
        current_style = str(getattr(self, "_primitive_highlight_style", "solid")).lower()
        for key, label in self._primitive_highlight_style_options():
            action = style_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(key == current_style)
            action.triggered.connect(lambda _, k=key: self._set_primitive_highlight_style(k))
        color_menu = submenu.addMenu("Color")
        current_color = str(getattr(self, "_primitive_highlight_color", "amber")).lower()
        for key, label in self._primitive_highlight_color_options():
            action = color_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(key == current_color)
            action.triggered.connect(lambda _, k=key: self._set_primitive_highlight_color(k))

    def _3d_face_view_presets(self):
        return [
            ("Front", 0.0, 0.0),
            ("Back", 180.0, 0.0),
            ("Right", 90.0, 0.0),
            ("Left", -90.0, 0.0),
            ("Top", 0.0, 90.0),
            ("Bottom", 0.0, -90.0),
        ]

    def _3d_corner_view_presets(self):
        iso = 35.264
        return [
            ("Top Front Right", 45.0, iso),
            ("Top Front Left", -45.0, iso),
            ("Top Back Right", 135.0, iso),
            ("Top Back Left", -135.0, iso),
            ("Bottom Front Right", 45.0, -iso),
            ("Bottom Front Left", -45.0, -iso),
            ("Bottom Back Right", 135.0, -iso),
            ("Bottom Back Left", -135.0, -iso),
        ]

    def _3d_edge_view_presets(self):
        return [
            ("Top Front", 0.0, 45.0),
            ("Top Right", 90.0, 45.0),
            ("Top Back", 180.0, 45.0),
            ("Top Left", -90.0, 45.0),
            ("Bottom Front", 0.0, -45.0),
            ("Bottom Right", 90.0, -45.0),
            ("Bottom Back", 180.0, -45.0),
            ("Bottom Left", -90.0, -45.0),
            ("Front Right", 45.0, 0.0),
            ("Front Left", -45.0, 0.0),
            ("Back Right", 135.0, 0.0),
            ("Back Left", -135.0, 0.0),
        ]

    def _apply_3d_camera_view(self, azimuth, elevation):
        view_3d = getattr(self, "view_3d", None)
        if view_3d is None:
            return
        try:
            view_3d.set_view(float(azimuth), float(elevation))
        except Exception:
            pass

    def _set_3d_orientation_widget_enabled(self, enabled, persist=True):
        enabled = bool(enabled)
        self._orientation_widget_enabled_3d = enabled
        ctrl = getattr(self, "orientation_widget_checkbox", None)
        if ctrl is not None:
            try:
                ctrl.blockSignals(True)
                ctrl.setChecked(enabled)
            finally:
                try:
                    ctrl.blockSignals(False)
                except Exception:
                    pass
        view_3d = getattr(self, "view_3d", None)
        if view_3d is not None and hasattr(view_3d, "set_orientation_overlay_enabled"):
            try:
                view_3d.set_orientation_overlay_enabled(enabled)
            except Exception:
                pass
        if persist and hasattr(self, "_settings"):
            try:
                self._settings.setValue("3d/orientation_widget_enabled", enabled)
            except Exception:
                pass

    def _set_view_navigation_enabled(self, enabled, persist=True):
        enabled = bool(enabled)
        self._view_navigation_enabled = enabled
        view_3d = getattr(self, "view_3d", None)
        if view_3d is not None and hasattr(view_3d, "set_view_navigation_enabled"):
            try:
                view_3d.set_view_navigation_enabled(enabled)
            except Exception:
                pass
        if hasattr(self, "view") and hasattr(self.view, "set_navigation_mode"):
            try:
                self.view.set_navigation_mode(enabled)
            except Exception:
                pass
        ctrl = getattr(self, "view_navigation_action", None)
        if ctrl is not None and ctrl.isChecked() != enabled:
            try:
                ctrl.blockSignals(True)
                ctrl.setChecked(enabled)
            finally:
                try:
                    ctrl.blockSignals(False)
                except Exception:
                    pass
        chk = getattr(self, "view_nav_checkbox", None)
        if chk is not None and chk.isChecked() != enabled:
            try:
                chk.blockSignals(True)
                chk.setChecked(enabled)
            finally:
                try:
                    chk.blockSignals(False)
                except Exception:
                    pass
        if persist and hasattr(self, "_settings"):
            try:
                self._settings.setValue("3d/navigation_enabled", enabled)
            except Exception:
                pass
        self._update_interaction_hints()

    def _add_3d_navigation_menu(self, menu):
        if menu is None:
            return
        view_3d = getattr(self, "view_3d", None)
        if view_3d is None:
            return
        nav_menu = menu.addMenu("Navigation / View")
        overlay_action = nav_menu.addAction("View Cube / Orientation Widget")
        overlay_action.setCheckable(True)
        overlay_action.setChecked(bool(getattr(self, "_orientation_widget_enabled_3d", True)))
        overlay_action.triggered.connect(self._set_3d_orientation_widget_enabled)
        nav_menu.addSeparator()
        std_menu = nav_menu.addMenu("Standard Views")

        faces_menu = std_menu.addMenu("Faces")
        for label, az, el in self._3d_face_view_presets():
            faces_menu.addAction(label, lambda a=az, e=el: self._apply_3d_camera_view(a, e))

        corners_menu = std_menu.addMenu("Corners")
        for label, az, el in self._3d_corner_view_presets():
            corners_menu.addAction(label, lambda a=az, e=el: self._apply_3d_camera_view(a, e))

        edges_menu = std_menu.addMenu("Edges")
        for label, az, el in self._3d_edge_view_presets():
            edges_menu.addAction(label, lambda a=az, e=el: self._apply_3d_camera_view(a, e))

        std_menu.addSeparator()
        std_menu.addAction("Isometric", lambda: self._apply_3d_camera_view(45.0, 35.264))

        nav_menu.addSeparator()
        if hasattr(view_3d, "fit_view"):
            nav_menu.addAction("Fit View", view_3d.fit_view)
        if hasattr(view_3d, "center_origin"):
            nav_menu.addAction("Center Origin", view_3d.center_origin)
        if hasattr(view_3d, "set_random_view"):
            nav_menu.addAction("Random View", view_3d.set_random_view)

    def _set_3d_bool_control(self, attr_name, value, apply_fn=None):
        ctrl = getattr(self, attr_name, None)
        if ctrl is not None and hasattr(ctrl, "setChecked"):
            try:
                ctrl.blockSignals(True)
                ctrl.setChecked(bool(value))
            finally:
                try:
                    ctrl.blockSignals(False)
                except Exception:
                    pass
        if callable(apply_fn):
            try:
                apply_fn(bool(value))
            except Exception:
                pass

    def _apply_3d_display_style_preset(self, preset_key):
        view_3d = self._ensure_view_3d()
        if view_3d is None:
            return
        preset = str(preset_key or "").strip().lower()
        presets = {
            "realistic": {"material": "Metal", "wireframe": False, "xray": False, "dim": False},
            "shaded": {"material": "Plastic", "wireframe": False, "xray": False, "dim": False},
            "shaded_edges": {"material": "Clay", "wireframe": False, "xray": False, "dim": False},
            "wireframe": {"material": "Wireframe", "wireframe": False, "xray": False, "dim": False},
            "xray_shaded": {"material": "Silver", "wireframe": False, "xray": True, "dim": True},
        }
        config = presets.get(preset)
        if not config:
            return

        material_label = config["material"]
        if hasattr(self, "material_combo"):
            try:
                self.material_combo.blockSignals(True)
                self.material_combo.setCurrentText(material_label)
            finally:
                try:
                    self.material_combo.blockSignals(False)
                except Exception:
                    pass
        self._on_material_style_changed(material_label)
        self._set_3d_bool_control(
            "mesh_wireframe_checkbox",
            config["wireframe"],
            lambda checked: self._call_view_3d("set_wireframe_visible", checked),
        )
        self._set_3d_bool_control(
            "mesh_xray_checkbox",
            config["xray"],
            lambda checked: self._call_view_3d("set_mesh_xray", checked),
        )
        self._set_3d_bool_control(
            "mesh_dim_checkbox",
            config["dim"],
            lambda checked: self._call_view_3d("set_mesh_dim", checked),
        )

    def _current_3d_display_style_preset(self):
        view_3d = getattr(self, "view_3d", None)
        if view_3d is None:
            return None
        material = str(getattr(view_3d, "_material_style", "")).lower()
        wire = bool(getattr(view_3d, "_wireframe_enabled", False))
        xray = bool(getattr(view_3d, "_mesh_xray_enabled", False))
        dim = bool(getattr(view_3d, "_mesh_dim_enabled", False))
        if material == "wireframe" and not xray:
            return "wireframe"
        if material == "clay" and not wire and not xray:
            return "shaded_edges"
        if material == "plastic" and not wire and not xray:
            return "shaded"
        if material == "metal" and not wire and not xray:
            return "realistic"
        if material == "silver" and xray:
            return "xray_shaded"
        return None

    def _add_3d_display_style_menu(self, menu):
        if menu is None:
            return
        view_3d = getattr(self, "view_3d", None)
        if view_3d is None:
            return
        style_menu = menu.addMenu("Display Style")
        current = self._current_3d_display_style_preset()
        options = [
            ("realistic", "Realistic"),
            ("shaded", "Shaded"),
            ("shaded_edges", "Shaded With Edges"),
            ("wireframe", "Wireframe"),
            ("xray_shaded", "X-Ray Shaded"),
        ]
        for key, label in options:
            action = style_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(key == current)
            action.triggered.connect(lambda _, k=key: self._apply_3d_display_style_preset(k))
        style_menu.addSeparator()
        material_menu = style_menu.addMenu("Material Style (Advanced)")
        for label in ("Silver", "Plastic", "Metal", "Clay", "Wireframe"):
            action = material_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(
                str(getattr(view_3d, "_material_style", "")).lower() == label.lower()
            )
            action.triggered.connect(lambda _, t=label: self._on_material_style_changed(t))

    def _populate_material_menu(self, menu, label, assign_fn):
        header = menu.addAction(f"Assign Material -> {label}")
        header.setEnabled(False)
        menu.addSeparator()
        mats = sorted(getattr(self.project_state, "materials", {}).values(), key=lambda m: m.serial)
        if not mats:
            no_action = menu.addAction("No materials defined")
            no_action.setEnabled(False)
        else:
            for mat in mats:
                action = menu.addAction(f"{mat.name} ({mat.mat_type})")
                action.triggered.connect(lambda _, m=mat: assign_fn(m))
        menu.addSeparator()
        other_action = menu.addAction("Other...")
        other_action.triggered.connect(self.view._open_material_editor)

    def _show_primitive_material_menu_for_id(self, prim_id, global_pos=None, label=None):
        if not self.view.can_assign_material():
            QMessageBox.warning(self, "Locked", "Switch to Materials stage first.")
            return
        prim = next((p for p in self.model3d.get("primitives", []) if p.get("id") == prim_id), None)
        if not prim:
            return
        menu = QMenu(self)
        self._populate_material_menu(
            menu,
            label or f"Primitive #{prim_id}",
            lambda mat, prim=prim: self._assign_material_to_primitive(prim, mat),
        )
        if global_pos is None:
            global_pos = self.mapToGlobal(self.rect().center())
        menu.exec(global_pos)

    def _material_target_from_selection(self):
        part = self.view.get_selected_part()
        if part and not part.is_void:
            return ("part", part, part.name)
        if not hasattr(self, "primitive_list"):
            return None
        item = self.primitive_list.currentItem()
        if not item:
            return None
        data = item.data(Qt.UserRole) or {}
        if data.get("kind") != "primitive":
            return None
        prim_id = data.get("id")
        prim = next((p for p in self.model3d.get("primitives", []) if p.get("id") == prim_id), None)
        if not prim:
            return None
        return ("primitive", prim, item.text())

    def _pick_face_in_viewport_mesh(self, view_3d, nodes, faces, x, y):
        if view_3d is None:
            return None
        try:
            nodes = np.asarray(nodes, dtype=float)
            faces = np.asarray(faces, dtype=int)
        except Exception:
            return None
        if nodes.ndim != 2 or nodes.shape[0] == 0 or nodes.shape[1] < 3:
            return None
        if faces.ndim != 2 or faces.shape[0] == 0 or faces.shape[1] != 3:
            return None
        try:
            screen, depth, visible = view_3d._project_points_to_screen(nodes[:, :3])
        except Exception:
            return None
        if screen is None or depth is None or visible is None:
            return None
        try:
            face_visible = visible[faces].all(axis=1)
        except Exception:
            return None
        if not np.any(face_visible):
            return None

        tri = faces[face_visible]
        p0 = screen[tri[:, 0]]
        p1 = screen[tri[:, 1]]
        p2 = screen[tri[:, 2]]
        e0 = p1 - p0
        e1 = p2 - p0
        rel = np.array([float(x), float(y)], dtype=float) - p0
        den = e0[:, 0] * e1[:, 1] - e1[:, 0] * e0[:, 1]
        nondeg = np.abs(den) > 1e-12
        a = np.zeros(len(tri), dtype=float)
        b = np.zeros(len(tri), dtype=float)
        a[nondeg] = (rel[nondeg, 0] * e1[nondeg, 1] - e1[nondeg, 0] * rel[nondeg, 1]) / den[nondeg]
        b[nondeg] = (e0[nondeg, 0] * rel[nondeg, 1] - rel[nondeg, 0] * e0[nondeg, 1]) / den[nondeg]
        inside = nondeg & (a >= -0.02) & (b >= -0.02) & ((a + b) <= 1.02)
        original_ids = np.nonzero(face_visible)[0]
        z0 = depth[tri[:, 0]]
        z1 = depth[tri[:, 1]]
        z2 = depth[tri[:, 2]]
        z_interp = z0 + a * (z1 - z0) + b * (z2 - z0)

        if np.any(inside):
            inside_ids = original_ids[inside]
            inside_depth = z_interp[inside]
            best_local = int(np.argmin(inside_depth))
            return {
                "face_index": int(inside_ids[best_local]),
                "rank": (0, float(inside_depth[best_local])),
            }

        try:
            d01, _ = view_3d._point_segment_distance_2d(float(x), float(y), p0, p1)
            d12, _ = view_3d._point_segment_distance_2d(float(x), float(y), p1, p2)
            d20, _ = view_3d._point_segment_distance_2d(float(x), float(y), p2, p0)
        except Exception:
            return None
        dist = np.minimum(d01, np.minimum(d12, d20))
        try:
            threshold = max(10.0, float(view_3d.get_default_node_size()) * 2.5)
        except Exception:
            threshold = 10.0
        near = nondeg & np.isfinite(dist) & (dist <= threshold)
        if not np.any(near):
            return None
        near_ids = original_ids[near]
        near_dist = dist[near]
        near_depth = z_interp[near]
        order = np.lexsort((near_depth, near_dist))
        best_local = int(order[0])
        return {
            "face_index": int(near_ids[best_local]),
            "rank": (1, float(near_dist[best_local]), float(near_depth[best_local])),
        }

    def _pick_3d_primitive_id_from_viewport_pos(self, x, y):
        view_3d = getattr(self, "view_3d", None)
        if view_3d is None:
            return None
        cad_kernel = getattr(self.view, "cad_kernel", None)
        if cad_kernel is None or not cad_kernel.available():
            return None

        best_prim_id = None
        best_rank = None
        for prim in self.model3d.get("primitives", []):
            prim_id = prim.get("id")
            shape = self._build_primitive_shape(prim)
            if shape is None:
                continue
            try:
                nodes, faces = cad_kernel.tessellate(shape, 0.5, 0.5)
            except Exception:
                continue
            pick = self._pick_face_in_viewport_mesh(view_3d, nodes, faces, x, y)
            if not pick:
                continue
            rank = pick.get("rank")
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_prim_id = int(prim_id)
        return best_prim_id

    def _sync_3d_primitive_selection_from_context_event(self, event):
        if not hasattr(self, "primitive_list"):
            return
        if event is None:
            return
        pos = None
        try:
            pos = event.position().toPoint()
        except Exception:
            try:
                posf = event.position()
                pos = posf.toPoint() if hasattr(posf, "toPoint") else posf
            except Exception:
                pos = None
        if pos is None:
            return
        prim_id = self._pick_3d_primitive_id_from_viewport_pos(pos.x(), pos.y())
        if prim_id is None:
            return
        self._select_primitive_by_id(prim_id)

    def _current_context_module(self):
        panel = getattr(self, "properties_panel", None)
        if panel and hasattr(panel, "tabs"):
            idx = panel.tabs.currentIndex()
            module_map = {
                0: "Part",
                1: "Property",
                2: "Interaction",
                3: "Boundary",
                4: "Load",
                5: "Particles",
                6: "Job",
                7: "Results",
            }
            return module_map.get(idx, "Part")
        module = getattr(self.view, "active_module", None)
        return module or "Part"

    def _extend_3d_context_menu(self, menu, _event):
        menu.clear()
        menu.addAction("Apply Load", self._apply_load_from_toolbar)
        menu.addAction("Apply Boundary Condition", self._apply_bc_from_toolbar)
        menu.addAction("Delete", self._delete_context_target)
        menu.addAction("Properties", self._open_properties_from_context)

    def _delete_context_target(self):
        if self._workspace_3d:
            self._delete_selected_shapes()
            return
        part = self.view.get_selected_part()
        if part is not None:
            self.view.delete_part(part, confirm=True)

    def _open_properties_from_context(self):
        panel = getattr(self, "properties_panel", None)
        if panel is None or not hasattr(panel, "tabs"):
            return
        if hasattr(panel, "materials_tab"):
            panel.tabs.setCurrentWidget(panel.materials_tab)

    def _show_primitive_material_menu(self, pos):
        if not hasattr(self, "primitive_list"):
            return
        item = self.primitive_list.itemAt(pos)
        if not item:
            return
        data = item.data(Qt.UserRole) or {}
        if data.get("kind") != "primitive":
            return
        if not self.view.can_assign_material():
            QMessageBox.warning(self, "Locked", "Switch to Materials stage first.")
            return
        prim_id = data.get("id")
        prim = next((p for p in self.model3d.get("primitives", []) if p.get("id") == prim_id), None)
        if not prim:
            return
        menu = QMenu(self)
        self._populate_material_menu(
            menu,
            item.text(),
            lambda mat, prim=prim: self._assign_material_to_primitive(prim, mat),
        )
        menu.exec(self.primitive_list.mapToGlobal(pos))

    def enable_validation_mode(self):
        if hasattr(self, "validation_manager"):
            return
        try:
            from ui.validation.manager import ValidationManager
        except ImportError:
            QMessageBox.warning(self, "Validation Mode", "Validation module could not be loaded.")
            return
        self.validation_manager = ValidationManager(self)

    def _selected_shape_ids(self):
        if not hasattr(self, "primitive_list"):
            return []
        ids = []
        for item in self.primitive_list.selectedItems():
            data = item.data(Qt.UserRole) or {}
            if "id" in data:
                ids.append(int(data["id"]))
        return ids

    def _select_primitive_by_id(self, prim_id):
        if not hasattr(self, "primitive_list"):
            return
        self.primitive_list.clearSelection()
        for i in range(self.primitive_list.count()):
            item = self.primitive_list.item(i)
            data = item.data(Qt.UserRole) or {}
            if data.get("kind") == "primitive" and data.get("id") == prim_id:
                self.primitive_list.setCurrentItem(item)
                item.setSelected(True)
                break

    def _update_3d_primitive_highlight(self):
        view_3d = getattr(self, "view_3d", None)
        if view_3d is None or not hasattr(view_3d, "set_selection_overlay"):
            return
        if not bool(getattr(self, "_workspace_3d", False)):
            if hasattr(view_3d, "clear_selection_overlay"):
                view_3d.clear_selection_overlay()
            return
        if not hasattr(self, "primitive_list"):
            if hasattr(view_3d, "clear_selection_overlay"):
                view_3d.clear_selection_overlay()
            return

        item = self.primitive_list.currentItem()
        data = item.data(Qt.UserRole) if item is not None else None
        data = data or {}
        if data.get("kind") != "primitive":
            if hasattr(view_3d, "clear_selection_overlay"):
                view_3d.clear_selection_overlay()
            return

        prim_id = data.get("id")
        prim = next((p for p in self.model3d.get("primitives", []) if p.get("id") == prim_id), None)
        if not prim:
            if hasattr(view_3d, "clear_selection_overlay"):
                view_3d.clear_selection_overlay()
            return

        cad_kernel = getattr(self.view, "cad_kernel", None)
        if cad_kernel is None or not cad_kernel.available():
            if hasattr(view_3d, "clear_selection_overlay"):
                view_3d.clear_selection_overlay()
            return

        shape = self._build_primitive_shape(prim)
        if shape is None:
            if hasattr(view_3d, "clear_selection_overlay"):
                view_3d.clear_selection_overlay()
            return
        try:
            nodes, faces = cad_kernel.tessellate(shape, 0.5, 0.5)
        except Exception:
            nodes, faces = None, None
        overlay_args = self._primitive_highlight_overlay_args()
        view_3d.set_selection_overlay(nodes, faces, **overlay_args)

    def _begin_mouse_place_primitive(self, prim_id, z_plane=0.0):
        if not self._workspace_3d:
            return
        view_3d = self._ensure_view_3d()
        if view_3d is None or not hasattr(view_3d, "request_point_pick"):
            return
        self._select_primitive_by_id(prim_id)
        view_3d.request_point_pick(
            lambda x, y, z: self._apply_mouse_placement(prim_id, x, y, z),
            z_plane=z_plane,
        )
        self.statusBar().showMessage(
            "Click in the 3D view to place the shape. Drag the scale gizmo to resize.",
            self._status_timeout_ms,
        )

    def _apply_mouse_placement(self, prim_id, x, y, z):
        prim = next((p for p in self.model3d.get("primitives", []) if p.get("id") == prim_id), None)
        if not prim:
            return
        transform = prim.get("transform", {})
        transform.update({"tx": x, "ty": y, "tz": z})
        prim["transform"] = transform
        self.project_dirty = True
        self._refresh_3d_view()
        self._update_gizmo_position()
        self._set_gizmo_mode("scale")

    def _edit_selected_primitive(self):
        if not hasattr(self, "primitive_list"):
            return
        selected = self.primitive_list.selectedItems()
        if not selected:
            QMessageBox.information(self, "Edit Primitive", "Select a primitive to edit.")
            return
        data = selected[0].data(Qt.UserRole) or {}
        if data.get("kind") != "primitive":
            QMessageBox.information(self, "Edit Primitive", "Select a primitive (not a boolean result).")
            return
        prim_id = data.get("id")
        prim = next((p for p in self.model3d.get("primitives", []) if p.get("id") == prim_id), None)
        if not prim:
            return
        params = self._prompt_primitive_params(prim.get("type"), existing=prim)
        if not params:
            return
        mouse_place = bool(params.pop("mouse_place", False))
        self._push_3d_undo()
        prim["params"] = params["params"]
        prim["transform"] = params["transform"]
        self.project_dirty = True
        self._refresh_primitive_list()
        self._refresh_3d_view()
        if mouse_place:
            self._begin_mouse_place_primitive(prim_id, prim["transform"].get("tz", 0.0))

    def _delete_selected_shapes(self):
        ids = set(self._selected_shape_ids())
        if not ids:
            return
        self._push_3d_undo()
        self.model3d["primitives"] = [
            p for p in self.model3d.get("primitives", []) if p.get("id") not in ids
        ]
        self.model3d["operations"] = [
            o for o in self.model3d.get("operations", [])
            if o.get("id") not in ids and o.get("a") not in ids and o.get("b") not in ids
        ]
        self.project_dirty = True
        self._refresh_primitive_list()
        self._refresh_3d_view()

    def _apply_boolean_op(self, op):
        ids = self._selected_shape_ids()
        if len(ids) < 2:
            QMessageBox.information(self, "Boolean", "Select two shapes to apply a boolean.")
            return
        self._push_3d_undo()
        a_id, b_id = ids[0], ids[1]
        op_id = self._next_model3d_id()
        self.model3d.setdefault("operations", []).append(
            {"id": op_id, "op": op, "a": a_id, "b": b_id}
        )
        self.project_dirty = True
        self._refresh_primitive_list()
        self._refresh_3d_view()

    def _refresh_3d_view(self):
        if self.view_3d is None:
            self._ensure_view_3d()
        if self.view_3d is None:
            return
        if not getattr(self.view, "cad_kernel", None) or not self.view.cad_kernel.available():
            self.view_3d.clear_mesh()
            return
        shapes = {}
        for prim in self.model3d.get("primitives", []):
            shape = self._build_primitive_shape(prim)
            if shape is not None:
                shapes[prim.get("id")] = shape
        for op in self.model3d.get("operations", []):
            a = shapes.get(op.get("a"))
            b = shapes.get(op.get("b"))
            if a is None or b is None:
                continue
            result = self.view.cad_kernel.boolean(a, b, op.get("op"))
            if result is not None:
                shapes[op.get("id")] = result

        if not shapes:
            self.view_3d.clear_mesh()
            return

        show_operands = False
        if hasattr(self, "show_operands_checkbox"):
            show_operands = self.show_operands_checkbox.isChecked()
        force_combined_for_picking = False
        try:
            force_combined_for_picking = (
                getattr(self, "active_stage", None) is not None
                and self.active_stage.value >= ProjectStage.BCS.value
            )
        except Exception:
            force_combined_for_picking = False
        if force_combined_for_picking:
            show_operands = False
        operations = self.model3d.get("operations", [])
        if operations and not show_operands:
            last_id = operations[-1].get("id")
            combined = shapes.get(last_id)
        else:
            if show_operands and shapes and len(shapes) > 1:
                meshes = []
                palette = [
                    (0.75, 0.78, 0.85, 0.9),
                    (0.7, 0.85, 0.75, 0.85),
                    (0.85, 0.75, 0.75, 0.85),
                    (0.8, 0.8, 0.65, 0.85),
                    (0.7, 0.75, 0.9, 0.85),
                ]
                for idx, shape in enumerate(shapes.values()):
                    nodes, faces = self.view.cad_kernel.tessellate(shape, 0.5, 0.5)
                    if nodes is None or faces is None:
                        continue
                    meshes.append(
                        {
                            "nodes": nodes,
                            "faces": faces,
                            "color": palette[idx % len(palette)],
                            "edge_color": (0.1, 0.1, 0.1, 0.15),
                        }
                    )
                self.view_3d.set_meshes(meshes)
                self._update_gizmo_position()
                self._update_3d_primitive_highlight()
                return
            combined = None
            for shape in shapes.values():
                if combined is None:
                    combined = shape
                else:
                    combined = self.view.cad_kernel.boolean(combined, shape, "add")

        if combined is None:
            self.view_3d.clear_mesh()
            return

        nodes, faces, face_ids = self.view.cad_kernel.tessellate(
            combined, 0.5, 0.5, with_face_ids=True
        )
        if nodes is None or faces is None:
            self.view_3d.clear_mesh()
            return
        self.view_3d.set_mesh(nodes, faces, face_group_ids=face_ids)
        topology = self.view.cad_kernel.extract_topology(combined, edge_samples=64)
        if hasattr(self.view_3d, "set_cad_topology"):
            self.view_3d.set_cad_topology(topology)
        self._update_gizmo_position()
        self._update_3d_primitive_highlight()

    def _update_grid_settings(self):
        if self.view_3d is None:
            return
        enabled = getattr(self, "snap_checkbox", None)
        spacing = getattr(self, "snap_size_spin", None)
        self.view_3d.set_grid_snap(
            enabled.isChecked() if enabled else True,
            spacing.value() if spacing else 20.0,
        )

    def _on_primitive_selection_changed(self):
        if hasattr(self, "show_operands_checkbox"):
            self.show_operands_checkbox.setChecked(True)
        self._update_gizmo_position()
        self._refresh_3d_view()

    def _update_gizmo_position(self):
        if self.view_3d is None:
            return
        module_name = self._current_context_module()
        if module_name not in {"Part", "Property"}:
            self.view_3d.set_gizmo(enabled=False)
            return
        selected = []
        if hasattr(self, "primitive_list"):
            selected = self.primitive_list.selectedItems()
        if not selected:
            self.view_3d.set_gizmo(enabled=False)
            return
        data = selected[0].data(Qt.UserRole) or {}
        if data.get("kind") != "primitive":
            self.view_3d.set_gizmo(enabled=False)
            return
        prim_id = data.get("id")
        prim = next((p for p in self.model3d.get("primitives", []) if p.get("id") == prim_id), None)
        if not prim:
            self.view_3d.set_gizmo(enabled=False)
            return
        transform = prim.get("transform", {})
        self.view_3d.set_gizmo(
            (transform.get("tx", 0.0), transform.get("ty", 0.0), transform.get("tz", 0.0)),
            enabled=True,
        )
        if hasattr(self, "_gizmo_mode"):
            self.view_3d.set_gizmo_mode(self._gizmo_mode)

    def _move_selected_primitive(self, x, y, z):
        if not hasattr(self, "primitive_list"):
            return
        selected = self.primitive_list.selectedItems()
        if not selected:
            return
        data = selected[0].data(Qt.UserRole) or {}
        if data.get("kind") != "primitive":
            return
        prim_id = data.get("id")
        prim = next((p for p in self.model3d.get("primitives", []) if p.get("id") == prim_id), None)
        if not prim:
            return
        if not self._in_3d_drag:
            self._push_3d_undo()
        transform = prim.get("transform", {})
        transform.update({"tx": x, "ty": y, "tz": z})
        prim["transform"] = transform
        self.project_dirty = True
        self._refresh_3d_view()

    def _rotate_selected_primitive(self, rx, ry, rz):
        if not hasattr(self, "primitive_list"):
            return
        selected = self.primitive_list.selectedItems()
        if not selected:
            return
        data = selected[0].data(Qt.UserRole) or {}
        if data.get("kind") != "primitive":
            return
        prim_id = data.get("id")
        prim = next((p for p in self.model3d.get("primitives", []) if p.get("id") == prim_id), None)
        if not prim:
            return
        if not self._in_3d_drag:
            self._push_3d_undo()
        transform = prim.get("transform", {})
        transform["rx"] = float(transform.get("rx", 0.0)) + float(rx)
        transform["ry"] = float(transform.get("ry", 0.0)) + float(ry)
        transform["rz"] = float(transform.get("rz", 0.0)) + float(rz)
        prim["transform"] = transform
        self.project_dirty = True
        self._refresh_3d_view()

    def _scale_selected_primitive(self, sx, sy, sz):
        if not hasattr(self, "primitive_list"):
            return
        selected = self.primitive_list.selectedItems()
        if not selected:
            return
        data = selected[0].data(Qt.UserRole) or {}
        if data.get("kind") != "primitive":
            return
        prim_id = data.get("id")
        prim = next((p for p in self.model3d.get("primitives", []) if p.get("id") == prim_id), None)
        if not prim:
            return
        if not self._in_3d_drag:
            self._push_3d_undo()
        params = prim.get("params", {})
        ptype = prim.get("type")

        def clamp(val, minimum=0.001):
            return max(minimum, float(val))

        sx = float(sx)
        sy = float(sy)
        sz = float(sz)
        radial = (sx + sy) * 0.5
        overall = (sx + sy + sz) / 3.0
        if ptype == "box":
            params["width"] = clamp(params.get("width", 1.0) * sx)
            params["depth"] = clamp(params.get("depth", 1.0) * sy)
            params["height"] = clamp(params.get("height", 1.0) * sz)
        elif ptype == "cylinder":
            params["radius"] = clamp(params.get("radius", 1.0) * radial)
            params["height"] = clamp(params.get("height", 1.0) * sz)
        elif ptype == "sphere":
            params["radius"] = clamp(params.get("radius", 1.0) * overall)
        elif ptype == "cone":
            params["radius_base"] = clamp(params.get("radius_base", 1.0) * radial)
            params["radius_top"] = clamp(params.get("radius_top", 0.0) * radial)
            params["height"] = clamp(params.get("height", 1.0) * sz)
        elif ptype == "ring":
            params["major_radius"] = clamp(params.get("major_radius", 1.0) * radial)
            params["tube_radius"] = clamp(params.get("tube_radius", 0.2) * overall)
        elif ptype == "extrude":
            params["profile_width"] = clamp(params.get("profile_width", 1.0) * sx)
            params["profile_depth"] = clamp(params.get("profile_depth", 1.0) * sy)
            params["height"] = clamp(params.get("height", 1.0) * sz)
        elif ptype == "revolve":
            params["profile_radius"] = clamp(params.get("profile_radius", 1.0) * radial)
            params["height"] = clamp(params.get("height", 1.0) * sz)
        prim["params"] = params
        self.project_dirty = True
        self._refresh_3d_view()

    def _build_primitive_shape(self, prim):
        if not prim:
            return None
        ptype = prim.get("type")
        params = prim.get("params", {})
        transform = prim.get("transform", {})
        center = (
            float(transform.get("tx", 0.0)),
            float(transform.get("ty", 0.0)),
            float(transform.get("tz", 0.0)),
        )
        rotation = (
            float(transform.get("rx", 0.0)),
            float(transform.get("ry", 0.0)),
            float(transform.get("rz", 0.0)),
        )
        ck = self.view.cad_kernel
        if ptype == "box":
            shape = ck.make_box(
                center,
                (
                    float(params.get("width", 1.0)),
                    float(params.get("depth", 1.0)),
                    float(params.get("height", 1.0)),
                ),
            )
            return ck.rotate(shape, center, rotation)
        if ptype == "cylinder":
            shape = ck.make_cylinder(
                center,
                float(params.get("radius", 1.0)),
                float(params.get("height", 1.0)),
            )
            return ck.rotate(shape, center, rotation)
        if ptype == "sphere":
            shape = ck.make_sphere(center, float(params.get("radius", 1.0)))
            return ck.rotate(shape, center, rotation)
        if ptype == "cone":
            shape = ck.make_cone(
                center,
                float(params.get("radius_base", 1.0)),
                float(params.get("radius_top", 0.0)),
                float(params.get("height", 1.0)),
            )
            return ck.rotate(shape, center, rotation)
        if ptype == "ring":
            shape = ck.make_torus(
                center,
                float(params.get("major_radius", 1.0)),
                float(params.get("tube_radius", 0.2)),
            )
            return ck.rotate(shape, center, rotation)
        if ptype == "extrude":
            pw = float(params.get("profile_width", 1.0))
            pd = float(params.get("profile_depth", 1.0))
            h = float(params.get("height", 1.0))
            face = ck.face_from_polygon(
                [
                    (-0.5 * pw, -0.5 * pd),
                    (0.5 * pw, -0.5 * pd),
                    (0.5 * pw, 0.5 * pd),
                    (-0.5 * pw, 0.5 * pd),
                ]
            )
            shape = ck.extrude(face, h)
            shape = ck.translate(shape, (center[0], center[1], center[2] - 0.5 * h))
            return ck.rotate(shape, center, rotation)
        if ptype == "revolve":
            radius = float(params.get("profile_radius", 1.0))
            height = float(params.get("height", 1.0))
            face = ck.face_from_polygon(
                [
                    (max(1e-6, radius * 0.4), -0.5 * height),
                    (radius, -0.5 * height),
                    (radius, 0.5 * height),
                    (max(1e-6, radius * 0.4), 0.5 * height),
                ]
            )
            if face is None:
                return None
            shape = ck.revolve(face, axis_origin=(0.0, 0.0, 0.0), axis_dir=(0.0, 1.0, 0.0), angle_deg=360.0)
            shape = ck.translate(shape, (center[0], center[1], center[2]))
            return ck.rotate(shape, center, rotation)
        return None

    def _generate_gmsh_mesh(self):
        mesh_tab = getattr(getattr(self, "properties_panel", None), "mesh_tab", None)
        mesh_config = mesh_tab._build_mesh_config() if mesh_tab is not None else {}
        return self.execute_app_command(GenerateParticlesCommand(mesh_config=mesh_config))

    def _generate_gmsh_mesh_impl(self):
        try:
            import gmsh  # noqa: F401
        except Exception:
            self._show_dependency_check_dialog()
            QMessageBox.warning(
                self,
                "Gmsh",
                "Gmsh Python API is not available. Install dependencies and retry.",
            )
            return
        use_cad = False
        cad_paths = []
        if not self.model3d.get("primitives"):
            cad_paths = self._collect_cad_import_paths()
            if cad_paths:
                use_cad = True
            else:
                if self._try_preview_cad_connections():
                    return
                QMessageBox.information(
                    self,
                    "Gmsh",
                    "Create at least one 3D shape or import CAD with solid volumes before generating connections.",
                )
                return
        if hasattr(self, "_mesh_regen_timer") and self._mesh_regen_timer.isActive():
            self._mesh_regen_timer.stop()
        if self._gmsh_in_progress:
            self._gmsh_pending = True
            return
        mesh_size = float(self.mesh_size_spin.value()) if hasattr(self, "mesh_size_spin") else 2.0
        mesh_type = "tetra"
        if hasattr(self, "mesh_type_combo"):
            mesh_type_label = self.mesh_type_combo.currentText().strip().lower()
            mesh_type = "hex-dominant" if "hex" in mesh_type_label else "tetra"
        progress = QProgressDialog("Generating 3D connections (Gmsh)...", None, 0, 0, self)
        progress.setWindowTitle("Generating Connections")
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.raise_()
        progress.activateWindow()
        progress.show()
        QApplication.processEvents()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._gmsh_progress = progress
        self._gmsh_in_progress = True

        if hasattr(self, "mesh_generate_btn"):
            self.mesh_generate_btn.setEnabled(False)
            self.mesh_generate_btn.setText("Generating...")
        if hasattr(self, "mesh_type_combo"):
            self.mesh_type_combo.setEnabled(False)
        if hasattr(self, "mesh_size_spin"):
            self.mesh_size_spin.setEnabled(False)

        start_time = time.time()
        try:
            if use_cad:
                from gmsh_mesher import generate_volume_mesh_from_cad
                nodes, elements, used_type = generate_volume_mesh_from_cad(
                    cad_paths, mesh_size, mesh_type=mesh_type, fallback_tetra=True
                )
            else:
                nodes, elements, used_type = generate_volume_mesh(
                    self.model3d, mesh_size, mesh_type=mesh_type, fallback_tetra=True
                )
        except GmshError as exc:
            self._finish_gmsh_progress()
            if use_cad and self._try_preview_cad_connections():
                return
            print(f"[Gmsh] Particle generation error: {exc}")
            QMessageBox.warning(self, "Gmsh", str(exc))
            return
        except Exception as exc:
            self._finish_gmsh_progress()
            print(f"[Gmsh] Unexpected particle generation error: {exc}")
            traceback.print_exc()
            QMessageBox.critical(self, "Gmsh", f"Connection generation failed: {exc}")
            return

        elapsed = time.time() - start_time
        if self._gmsh_progress:
            self._gmsh_progress.setLabelText(
                f"Generating 3D connections (Gmsh)...\nElapsed: {elapsed:.1f}s"
            )
        self._on_gmsh_mesh_finished(nodes, elements, used_type, elapsed=elapsed)

    def _try_preview_cad_connections(self):
        if not hasattr(self, "view") or not hasattr(self.view, "preview_3d_mesh"):
            return False
        if not self.view.preview_3d_mesh():
            return False
        view_3d = self._ensure_view_3d()
        if view_3d is not None:
            view_3d.set_material_style("mesh")
            view_3d.set_node_display(size=5.0)
            view_3d.set_node_colors_auto()
            view_3d.set_show_all_nodes(True)
            if hasattr(self.view, "global_nodes_3d") and self.view.global_nodes_3d is not None:
                view_3d.set_hover_nodes(self.view.global_nodes_3d)
            self._update_3d_mesh_visibility()
        if hasattr(self, "mesh_status_label"):
            nodes = (
                len(self.view.global_nodes_3d)
                if getattr(self.view, "global_nodes_3d", None) is not None
                else 0
            )
            elems = (
                len(self.view.global_elements_3d)
                if getattr(self.view, "global_elements_3d", None) is not None
                else 0
            )
            self.mesh_status_label.setText(
                f"Particles: {nodes} · Connections: {elems} · Type: CAD surface"
            )
        QMessageBox.information(
            self,
            "Connections",
            "Generated surface connections from CAD import.\n"
            "Note: volumetric connections via Gmsh require 3D primitives or solid CAD files (STEP/IGES).",
        )
        return True

    def _collect_cad_import_paths(self):
        paths = []
        if not hasattr(self, "view") or not hasattr(self.view, "parts"):
            return paths
        for part in self.view.parts:
            src = getattr(part, "cad_source", None)
            if not isinstance(src, dict):
                continue
            if src.get("type") != "import":
                continue
            path = src.get("path")
            if path and os.path.exists(path):
                paths.append(path)
        # de-dupe while preserving order
        seen = set()
        uniq = []
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            uniq.append(path)
        return uniq

    def _schedule_gmsh_regen(self):
        if not self.model3d.get("primitives", []):
            return
        if hasattr(self, "mesh_status_label"):
            self.mesh_status_label.setText("Connection settings changed — regenerating...")
        self._mesh_regen_timer.start(600)

    def _finish_gmsh_progress(self):
        QApplication.restoreOverrideCursor()
        if self._gmsh_progress:
            self._gmsh_progress.close()
            self._gmsh_progress.deleteLater()
            self._gmsh_progress = None
        self._gmsh_in_progress = False
        if hasattr(self, "mesh_generate_btn"):
            self.mesh_generate_btn.setEnabled(True)
            self.mesh_generate_btn.setText("Generate Connections")
        if hasattr(self, "mesh_type_combo"):
            self.mesh_type_combo.setEnabled(True)
        if hasattr(self, "mesh_size_spin"):
            self.mesh_size_spin.setEnabled(True)

    def _on_gmsh_mesh_finished(self, nodes, elements, used_type, elapsed=None):
        self._finish_gmsh_progress()
        if nodes is None or elements is None or len(nodes) == 0 or len(elements) == 0:
            QMessageBox.warning(self, "Gmsh", "No connections generated.")
            return

        self.view.global_nodes_3d = np.asarray(nodes, dtype=float)
        self.view.global_elements_3d = np.asarray(elements, dtype=int)
        self.view.mesh3dUpdated.emit(self.view.global_nodes_3d, self.view.global_elements_3d)
        view_3d = self._ensure_view_3d()
        if view_3d is not None:
            view_3d.set_material_style("mesh")
            view_3d.set_node_display(size=5.0)
            view_3d.set_node_colors_auto()
            view_3d.set_show_all_nodes(True)
            view_3d.set_hover_nodes(self.view.global_nodes_3d)
            self._update_3d_mesh_visibility()

        if hasattr(self, "mesh_status_label"):
            self.mesh_status_label.setText(
                f"Particles: {len(nodes)} · Connections: {len(elements)} · Type: {used_type}"
            )
        if hasattr(self, "mesh_type_combo"):
            mesh_type_label = self.mesh_type_combo.currentText().strip().lower()
            if "hex" in mesh_type_label:
                if used_type.startswith("tetra"):
                    QMessageBox.information(
                        self,
                        "Gmsh Particles",
                        "Hex-dominant connection generation was not available for this model.\n"
                        "Falling back to tetrahedral connections.",
                    )
                elif "displayed as tetra" in used_type:
                    QMessageBox.information(
                        self,
                        "Gmsh Connections",
                        "Hex-dominant connections generated.\n"
                        "Note: the viewer displays hex connections as tetrahedra.",
                    )
        QMessageBox.information(
            self,
            "Particles Ready",
            (
                f"Particles generated successfully.\nParticles: {len(nodes)}\nConnections: {len(elements)}"
                + (f"\nTime: {elapsed:.1f}s" if elapsed is not None else "")
            ),
        )
        if self._gmsh_pending:
            self._gmsh_pending = False
            self._generate_gmsh_mesh()

    def execute_app_command(self, command):
        return self.command_bus.execute(command)

    def _command_execution_middleware(self, command, next_step):
        started_at = time.perf_counter()
        result = next_step()
        self.event_bus.publish(
            "command.timed",
            {
                "command": command,
                "command_name": getattr(command, "command_name", type(command).__name__),
                "elapsed_sec": time.perf_counter() - started_at,
            },
        )
        return result

    def _on_command_failed(self, payload):
        error = payload.get("error")
        command_name = payload.get("command_name", "Command")
        try:
            self.statusBar().showMessage(f"{command_name} failed: {error}", 5000)
        except Exception:
            pass

    def _on_command_completed(self, payload):
        command_name = payload.get("command_name", "Command")
        try:
            self.statusBar().showMessage(f"{command_name} completed.", 2000)
        except Exception:
            pass

    def _generate_particles_from_command_impl(self, mesh_config):
        mesh_tab = getattr(getattr(self, "properties_panel", None), "mesh_tab", None)
        if mesh_tab is None:
            return {"accepted": False}
        if self.view.project_mode == "3d":
            if bool(getattr(self, "_workspace_3d", False)):
                self._generate_gmsh_mesh_impl()
                return {"accepted": True, "mode": "3d"}
            if not self.view.preview_3d_mesh():
                return {"accepted": False, "mode": "3d"}
        else:
            def _finish(success):
                if not success:
                    return
                self.view.set_display_mode("mesh")
                mesh_tab.show_nodes.blockSignals(True)
                mesh_tab.show_nodes.setChecked(True)
                mesh_tab.show_nodes.blockSignals(False)
                mesh_tab.show_mesh.blockSignals(True)
                mesh_tab.show_mesh.setChecked(False)
                mesh_tab.show_mesh.blockSignals(False)
                mesh_tab.mesh_view_toggle.blockSignals(True)
                mesh_tab.mesh_view_toggle.setChecked(True)
                mesh_tab.mesh_view_toggle.blockSignals(False)
                mesh_tab._update_stats()
                mesh_tab._update_display()
                mesh_tab._sync_point_preview(refresh=True)

            if not self.view.run_particle_generation_async(mesh_config=mesh_config, on_done=_finish):
                return {"accepted": False, "mode": "2d"}
        mesh_tab._update_stats()
        mesh_tab._update_display()
        return {"accepted": True, "mode": self.view.project_mode}

    def _run_solver_command_impl(self):
        job_tab = getattr(getattr(self, "properties_panel", None), "job_tab", None)
        if job_tab is None:
            QMessageBox.warning(self, "Solver", "Job panel is not available.")
            return {"job": None}

        missing_stage, missing_msg = job_tab._first_missing_run_stage()
        if missing_stage is not None:
            QMessageBox.warning(job_tab, "Run Prerequisite Missing", missing_msg)
            job_tab.log_output(missing_msg)
            job_tab._focus_stage(missing_stage)
            return {"job": None, "blocked": True}

        job_tab.output_log.clear()
        job_tab.progress_bar.setValue(0)
        job_tab.eta_label.setText("ETA: --")
        job_tab._last_progress_step = 0
        job_tab._solver_status_pct = None
        if hasattr(self.view, "release_results_file_handles"):
            try:
                self.view.release_results_file_handles()
            except Exception:
                pass
        QApplication.processEvents()
        job_tab.log_output(f"Exporting latest model data to {WORKSPACE_DIR_NAME}/ CSV files...")
        try:
            from solver_exporter import export_project_to_workspace
        except Exception:
            export_project_to_workspace = None
        assert callable(export_project_to_workspace), (
            "Direct solver file writes from panels are forbidden. "
            "Use solver_exporter.export_project_to_workspace only."
        )

        job_tab._sync_solver_settings_state()
        project_state = job_tab.project_state or getattr(self, "project_state", None)
        job_tab.project_state = project_state
        if project_state is not None and not isinstance(getattr(project_state, "solver_settings", None), dict):
            try:
                project_state.solver_settings = {}
            except Exception:
                project_state = None
        if project_state is None:
            class _ScratchState:
                solver_settings = {}

            project_state = _ScratchState()

        # Keep runtime export wiring out of persisted project_state.solver_settings.
        export_settings = dict(getattr(project_state, "solver_settings", {}) or {})
        export_settings["_export_options"] = {
            "silent": True,
            "export_mode": "full",
            "async_mesh": False,
            "force_remesh": False,
        }
        export_settings["_sketch_view"] = self.view

        class _ExportState:
            def __init__(self, solver_settings):
                self.solver_settings = solver_settings

        export_state = _ExportState(export_settings)
        export_ok = (
            export_project_to_workspace(export_state, _workspace_dir())
            if callable(export_project_to_workspace)
            else False
        )
        if not export_ok:
            export_error = str(getattr(self.view, "_last_export_error", "") or "").strip()
            if not export_error:
                export_error = "Export failed: No connections generated or error occurred."
            job_tab.log_output(export_error)
            job_tab._announce_status(export_error, 6000)
            return {"job": None, "export_ok": False}
        if getattr(self.view, "project_mode", "2d") == "3d":
            reply = QMessageBox.question(
                job_tab,
                "3D Mode",
                "3D solver is not available yet. Run the 2D solver on the base connections?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
        else:
            reply = QMessageBox.Yes
        if reply != QMessageBox.Yes:
            job_tab.log_output("Run canceled for 3D project.")
            return {"job": None, "canceled": True}
        job_tab.log_output(f"Export complete ({WORKSPACE_DIR_NAME}/).")
        job_tab.log_output("-" * 40)
        if job_tab.auto_dt_checkbox.isChecked():
            job_tab._recompute_dt(silent=True)
        job_tab._apply_simulation_settings()
        job_tab.log_output("Starting simulation engine...")
        job_tab._announce_status("Simulation starting...", 3000)
        controller = self.solver_controller
        if controller is None or not hasattr(controller, "run_solver_job"):
            QMessageBox.warning(job_tab, "Solver Controller", "Solver controller is not available.")
            return {"job": None}
        job_tab.run_button.setEnabled(False)
        job_tab._set_pause_state(False)
        job_tab.stop_button.setEnabled(True)
        job_tab._auto_visualized = False
        job_tab._job_start_time = time.perf_counter()
        job_tab._pause_started = None
        job_tab._paused_total = 0.0
        job_tab.elapsed_label.setText("Elapsed: 0.0 s")
        job_tab._elapsed_timer.start(200)
        job = controller.run_solver_job(
            config_path=job_tab._config_path(),
            workspace_folder=_workspace_dir(),
            total_steps=int(job_tab.total_steps_spin.value()),
        )
        if job is None:
            job_tab._elapsed_timer.stop()
            job_tab.run_button.setEnabled(True)
            job_tab.stop_button.setEnabled(False)
            return {"job": None}
        job_tab._current_job_id = job.job_id
        return {"job": job}


    def _set_gizmo_mode(self, mode):
        self._gizmo_mode = str(mode).lower()
        if hasattr(self, "gizmo_move_btn"):
            self.gizmo_move_btn.setChecked(self._gizmo_mode == "translate")
            self.gizmo_rotate_btn.setChecked(self._gizmo_mode == "rotate")
            self.gizmo_scale_btn.setChecked(self._gizmo_mode == "scale")
        if self.view_3d is not None:
            self.view_3d.set_gizmo_mode(self._gizmo_mode)

    def _on_material_style_changed(self, text):
        view_3d = self._ensure_view_3d()
        if view_3d is not None:
            style = str(text).lower()
            view_3d.set_material_style(text)
            if "wireframe" in style:
                view_3d.set_node_display(size=6.0)
                view_3d.set_node_colors_auto()
            else:
                view_3d.set_node_display(size=5.0)
                view_3d.set_node_colors_auto()
            self._update_3d_mesh_visibility()
            if hasattr(self.view, "global_nodes_3d") and self.view.global_nodes_3d is not None:
                view_3d.set_hover_nodes(self.view.global_nodes_3d)

    def _apply_3d_background_preset(self, name):
        view_3d = self._ensure_view_3d()
        if view_3d is None:
            return
        if not name:
            return
        label = str(name).strip().lower()
        if "custom" in label:
            self._pick_3d_background_color()
            return
        presets = {
            "light": (245, 247, 250),
        }
        rgb = presets.get(label, presets["light"])
        view_3d.set_background_color(rgb)

    def _pick_3d_background_color(self):
        view_3d = self._ensure_view_3d()
        if view_3d is None:
            return
        current = getattr(view_3d, "_bg_color", (245, 247, 250))
        color = QColorDialog.getColor(QColor(*current), self, "Pick Background Color")
        if not color.isValid():
            return
        view_3d.set_background_color(color)

    def _edge_color_from_fill(self, fill):
        r, g, b, _a = fill
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        if lum < 0.5:
            edge = (min(1.0, r + 0.35), min(1.0, g + 0.35), min(1.0, b + 0.35), 0.7)
        else:
            edge = (r * 0.35, g * 0.35, b * 0.35, 0.7)
        return edge

    def _pick_3d_object_color(self):
        view_3d = self._ensure_view_3d()
        if view_3d is None:
            return
        color = QColorDialog.getColor(QColor(240, 245, 255), self, "Pick Object Color")
        if not color.isValid():
            return
        fill = (color.redF(), color.greenF(), color.blueF(), 1.0)
        edge = self._edge_color_from_fill(fill)
        view_3d.set_mesh_color_override(fill, edge)

    def _reset_3d_object_color(self):
        if self.view_3d is not None:
            self.view_3d.clear_mesh_color_override()

    def _show_skewness_distribution(self):
        nodes = getattr(self.view, "global_nodes_3d", None)
        elements = getattr(self.view, "global_elements_3d", None)
        if nodes is None or elements is None or len(nodes) == 0 or len(elements) == 0:
            QMessageBox.information(self, "Skewness", "No 3D connections available.")
            return
        elems = np.asarray(elements, dtype=int)
        if elems.ndim != 2 or elems.shape[1] != 4:
            QMessageBox.information(self, "Skewness", "Skewness is available only for tetra connections.")
            return
        nodes = np.asarray(nodes, dtype=float)
        a = nodes[elems[:, 0]]
        b = nodes[elems[:, 1]]
        c = nodes[elems[:, 2]]
        d = nodes[elems[:, 3]]
        v = np.abs(np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)) / 6.0
        edges = [
            b - a,
            c - a,
            d - a,
            c - b,
            d - b,
            d - c,
        ]
        sum_l2 = np.zeros(len(v), dtype=float)
        for e in edges:
            sum_l2 += np.einsum("ij,ij->i", e, e)
        with np.errstate(divide="ignore", invalid="ignore"):
            q = 12.0 * np.power(3.0 * v, 2.0 / 3.0) / sum_l2
        q = np.clip(q, 0.0, 1.0)
        skew = 1.0 - q
        skew = skew[~np.isnan(skew)]
        if skew.size == 0:
            QMessageBox.information(self, "Skewness", "Could not compute skewness.")
            return

        bins = np.linspace(0.0, 1.0, 11)
        counts, _ = np.histogram(skew, bins=bins)
        total = int(np.sum(counts))
        lines = [
            f"Skewness (0=best, 1=worst) | Total connections: {total}",
            f"Min: {float(np.min(skew)):.3f}  Mean: {float(np.mean(skew)):.3f}  Max: {float(np.max(skew)):.3f}",
            "",
        ]
        for i in range(len(counts)):
            lo = bins[i]
            hi = bins[i + 1]
            pct = 100.0 * counts[i] / total if total else 0.0
            bar = "#" * int(pct / 2)
            lines.append(f"{lo:.1f}-{hi:.1f}: {counts[i]:6d}  {pct:5.1f}%  {bar}")

        dlg = QDialog(self)
        dlg.setWindowTitle("Connection Skewness Distribution")
        self._prepare_modal_dialog(dlg)
        layout = QVBoxLayout(dlg)
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText("\n".join(lines))
        layout.addWidget(text)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dlg.reject)
        buttons.accepted.connect(dlg.accept)
        layout.addWidget(buttons)
        dlg.resize(520, 360)
        dlg.exec()

    def _update_3d_mesh_visibility(self):
        view_3d = self._ensure_view_3d()
        if view_3d is None:
            return
        show_nodes = True
        show_mesh = True
        if hasattr(self, "mesh_display_combo"):
            mode = self.mesh_display_combo.currentText().strip().lower()
            if "particles only" in mode:
                show_nodes = True
                show_mesh = False
            elif "connections only" in mode:
                show_nodes = False
                show_mesh = True
            else:
                show_nodes = True
                show_mesh = True
        view_3d.set_visibility(show_nodes=show_nodes, show_mesh=show_mesh)

    def _on_gizmo_drag_started(self, mode):
        if not self._workspace_3d:
            return
        if not self._in_3d_drag:
            self._push_3d_undo()
        self._in_3d_drag = True

    def _on_gizmo_drag_finished(self, mode):
        self._in_3d_drag = False

    def _push_3d_undo(self):
        snapshot = copy.deepcopy(self.model3d)
        self._undo_stack_3d.append(snapshot)
        if len(self._undo_stack_3d) > self._max_undo_3d:
            self._undo_stack_3d.pop(0)
        self._redo_stack_3d.clear()

    def _undo_3d(self):
        if not self._undo_stack_3d:
            return
        self._redo_stack_3d.append(copy.deepcopy(self.model3d))
        self.model3d = copy.deepcopy(self._undo_stack_3d.pop())
        self._sync_model3d_state()
        self.project_dirty = True
        self._refresh_primitive_list()
        self._refresh_3d_view()

    def _redo_3d(self):
        if not self._redo_stack_3d:
            return
        self._undo_stack_3d.append(copy.deepcopy(self.model3d))
        self.model3d = copy.deepcopy(self._redo_stack_3d.pop())
        self._sync_model3d_state()
        self.project_dirty = True
        self._refresh_primitive_list()
        self._refresh_3d_view()

    def _handle_delete_action(self):
        if self._workspace_3d:
            self._delete_selected_shapes()

    def _handle_undo_action(self):
        if self._workspace_3d:
            self._undo_3d()
        else:
            self.view.undo()

    def _handle_redo_action(self):
        if self._workspace_3d:
            self._redo_3d()
        else:
            self.view.redo()

    def _show_cad_kernel_install_dialog(self):
        py_ver = sys.version_info
        py_label = f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}"
        warning = None
        if py_ver >= (3, 12):
            warning = (
                "Your Python version is 3.12+ and the automatic install may fail. "
                "If that happens, we recommend using Python 3.11."
            )
        venv_hint = self._detect_venv_ocp()
        kernel_hint = self._detect_cad_kernel_hint()

        dlg = QDialog(self)
        dlg.setWindowTitle("Enable 3D Shapes")
        self._prepare_modal_dialog(dlg)
        layout = QVBoxLayout(dlg)

        title = QLabel("<b>Enable 3D Shapes</b>")
        layout.addWidget(title)

        intro = QLabel(
            "To create 3D shapes, this app needs a one-time component called the CAD Kernel."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        env = QLabel(f"Current Python: {py_label}\nExecutable: {sys.executable}")
        env.setWordWrap(True)
        layout.addWidget(env)

        if venv_hint:
            hint = QLabel(
                "OCP was found in your .venv, but the app is not running from it.\n"
                "Restart using the .venv to enable 3D shapes."
            )
            hint.setWordWrap(True)
            hint.setObjectName("InfoHintLabel")
            layout.addWidget(hint)

            cmd = "source .venv/bin/activate\npython main_window.py"
            cmd_box = QPlainTextEdit()
            cmd_box.setReadOnly(True)
            cmd_box.setPlainText(cmd)
            layout.addWidget(cmd_box)

        if warning:
            warn_label = QLabel(warning)
            warn_label.setWordWrap(True)
            warn_label.setObjectName("WarnHintLabel")
            layout.addWidget(warn_label)

        if kernel_hint:
            hint = QLabel(kernel_hint)
            hint.setWordWrap(True)
            hint.setObjectName("NeutralHintLabel")
            layout.addWidget(hint)

        btn_row = QHBoxLayout()
        install_btn = QPushButton("Install Automatically")
        steps_btn = QPushButton("Show Step-by-Step")
        help_btn = QPushButton("Open Help Page")
        copy_btn = QPushButton("Copy Launch Command")
        close_btn = QPushButton("Close")
        btn_row.addWidget(install_btn)
        btn_row.addWidget(steps_btn)
        btn_row.addWidget(help_btn)
        btn_row.addWidget(copy_btn)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        install_btn.clicked.connect(lambda: self._install_cad_kernel())
        steps_btn.clicked.connect(lambda: self._show_cad_kernel_steps_dialog())
        help_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://pypi.org/project/OCP/")))
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText("source .venv/bin/activate\npython main_window.py"))
        close_btn.clicked.connect(dlg.reject)

        dlg.exec()

    def _install_cad_kernel(self):
        py_ver = sys.version_info
        if py_ver >= (3, 12):
            reply = QMessageBox.question(
                self,
                "Python Version Warning",
                "Your Python 3.12+ environment may fail to install OCP.\n"
                "Continue anyway?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if reply != QMessageBox.Yes:
                return
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "cadquery-ocp"],
                capture_output=True,
                text=True,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Install Failed", str(exc))
            return
        if result.returncode == 0:
            QMessageBox.information(
                self,
                "Install Complete",
                "CAD kernel installed. Please restart the app.",
            )
            return
        details = (result.stderr or result.stdout or "").strip()
        if len(details) > 4000:
            details = details[-4000:]
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Install Failed")
        box.setText("Automatic install failed.")
        box.setInformativeText(
            "If you have Conda, we recommend:\n"
            "conda install -c conda-forge pythonocc-core"
        )
        if details:
            box.setDetailedText(details)
        box.exec()

    def _show_cad_kernel_steps_dialog(self):
        py_ver = sys.version_info
        use_py311 = py_ver >= (3, 12)
        title = "Step-by-Step Install"
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        self._prepare_modal_dialog(dlg)
        layout = QVBoxLayout(dlg)

        intro = QLabel(
            "Copy and paste these commands into a terminal. This installs the CAD Kernel."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        commands = []
        if use_py311:
            commands.extend(
                [
                    "# Recommended (Python 3.11)",
                    "python3.11 -m venv .venv",
                    "source .venv/bin/activate",
                    "pip install cadquery-ocp",
                    "",
                    "# If you already use Python 3.11, just run:",
                    "pip install cadquery-ocp",
                ]
            )
        else:
            commands.extend(
                [
                    "source .venv/bin/activate",
                    "pip install cadquery-ocp",
                ]
            )
        commands.append("")
        commands.append("# Conda alternative:")
        commands.append("conda install -c conda-forge pythonocc-core")

        edit = QPlainTextEdit()
        edit.setReadOnly(True)
        edit.setPlainText("\n".join(commands))
        layout.addWidget(edit)

        btn_row = QHBoxLayout()
        copy_btn = QPushButton("Copy Commands")
        close_btn = QPushButton("Close")
        btn_row.addWidget(copy_btn)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        def copy_commands():
            QApplication.clipboard().setText(edit.toPlainText())

        copy_btn.clicked.connect(copy_commands)
        close_btn.clicked.connect(dlg.reject)

        dlg.exec()

    def _detect_venv_ocp(self):
        root = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(root, ".venv", "bin", "python"),
            os.path.join(root, ".venv", "Scripts", "python.exe"),
        ]
        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                result = subprocess.run(
                    [path, "-c", "import OCP"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    if os.path.abspath(path) != os.path.abspath(sys.executable):
                        return path
            except Exception:
                continue
        return None

    def _detect_cad_kernel_hint(self):
        try:
            import importlib.util
            ocp_spec = importlib.util.find_spec("ocp")
            ocp_cap_spec = importlib.util.find_spec("OCP")
            occ_spec = importlib.util.find_spec("OCC")
        except Exception:
            return None
        if ocp_spec and not ocp_cap_spec and not occ_spec:
            return (
                "It looks like the PyPI package named 'ocp' is installed, "
                "but it is not the CAD kernel. Use cadquery-ocp or conda "
                "pythonocc-core instead."
            )
        return None

    def _new_model3d(self):
        return {
            "version": 1,
            "primitives": [],
            "features": [],
            "operations": [],
            "metadata": {},
            "next_id": 1,
        }

    def _sync_model3d_state(self):
        if not isinstance(self.model3d, dict):
            self.model3d = self._new_model3d()
            return
        self.model3d.setdefault("version", 1)
        self.model3d.setdefault("primitives", [])
        self.model3d.setdefault("features", [])
        self.model3d.setdefault("operations", [])
        self.model3d.setdefault("metadata", {})
        if "next_id" not in self.model3d:
            next_id = 1
            for prim in self.model3d.get("primitives", []):
                try:
                    next_id = max(next_id, int(prim.get("id", 0)) + 1)
                except Exception:
                    continue
            self.model3d["next_id"] = next_id

    def _next_model3d_id(self):
        self._sync_model3d_state()
        next_id = int(self.model3d.get("next_id", 1))
        self.model3d["next_id"] = next_id + 1
        return next_id

    def _apply_theme(self, theme):
        theme = "light"
        self._current_theme = theme
        if self._app:
            self._toolbar_style = apply_professional_theme(self._app, theme)
        self._settings.setValue("theme", theme)

    def _preview_mesh(self):
        mesh_config = None
        if hasattr(self, "properties_panel"):
            mesh_tab = getattr(self.properties_panel, "mesh_tab", None)
            if mesh_tab:
                try:
                    mesh_config = mesh_tab._build_mesh_config()
                except Exception:
                    mesh_config = None
        if not self.view.has_current_mesh(mesh_config):
            QMessageBox.information(
                self,
                "Connection View",
                "No generated connections are available. Click Generate Connections in the Particles stage first.",
            )
            return
        if hasattr(self, "mesh_view_action"):
            self.mesh_view_action.setChecked(True)
        self.view.ensure_y_axis_up()
        self._sync_point_preview(refresh_data=True)

    def show_shortcuts_overlay(self):
        """Floating cheat-sheet card with every keyboard shortcut, grouped
        by category. Triggered by `?` key. Press Esc or click outside to
        dismiss."""
        dlg = getattr(self, "_shortcuts_dialog", None)
        if dlg is None:
            from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout
            dlg = QDialog(self)
            dlg.setWindowTitle("Keyboard Shortcuts")
            dlg.setObjectName("ShortcutsOverlay")
            dlg.setModal(False)
            dlg.resize(560, 480)
            outer = QVBoxLayout(dlg)
            outer.setContentsMargins(20, 20, 20, 20)
            outer.setSpacing(12)
            title = QLabel("Keyboard Shortcuts")
            title.setObjectName("ShortcutsTitle")
            outer.addWidget(title)
            groups = [
                ("View / Navigation", [
                    ("Ctrl+B", "Toggle left panel (Project Navigator)"),
                    ("Ctrl+J", "Toggle right panel (Properties)"),
                    ("Ctrl+0 / Home", "Fit all geometry to viewport"),
                    ("F", "Frame selected part"),
                    ("M", "Measure distance / angle"),
                    ("Ctrl+M", "Toggle mini-map overview"),
                    ("F11", "Full screen"),
                    ("Mouse wheel", "Zoom toward cursor"),
                    ("Middle-drag / Space+drag", "Pan canvas"),
                    ("Left-drag empty", "Pan canvas (select mode)"),
                ]),
                ("Selection / Edit", [
                    ("Click", "Select part"),
                    ("Ctrl+Click", "Toggle in multi-selection"),
                    ("Click+Drag part", "Move part"),
                    ("Right-click", "Cancel current draw tool"),
                    ("Ctrl+P", "Combine selected parts"),
                    ("Ctrl+Z / Ctrl+Y", "Undo / Redo"),
                    ("Delete", "Delete selected part"),
                ]),
                ("File / Project", [
                    ("Ctrl+N", "New project"),
                    ("Ctrl+O", "Open project"),
                    ("Ctrl+S / Ctrl+Shift+S", "Save / Save As"),
                    ("Ctrl+I", "Import geometry"),
                    ("Ctrl+Q", "Exit"),
                ]),
                ("Workspace", [
                    ("Ctrl+1 / Ctrl+2", "Switch 2D / 3D workspace"),
                    ("Ctrl+Home", "Center origin"),
                    ("?", "Show this shortcut list"),
                ]),
            ]
            for group_name, items in groups:
                gl = QLabel(group_name)
                gl.setObjectName("ShortcutsGroup")
                outer.addWidget(gl)
                for key, desc in items:
                    row = QHBoxLayout()
                    row.setContentsMargins(8, 0, 0, 0)
                    row.setSpacing(12)
                    kl = QLabel(key)
                    kl.setObjectName("ShortcutsKey")
                    kl.setMinimumWidth(180)
                    dl = QLabel(desc)
                    dl.setObjectName("ShortcutsDesc")
                    dl.setWordWrap(True)
                    row.addWidget(kl, 0)
                    row.addWidget(dl, 1)
                    outer.addLayout(row)
            outer.addStretch(1)
            hint = QLabel("Press Esc or click outside to close.")
            hint.setObjectName("ShortcutsHint")
            hint.setAlignment(Qt.AlignCenter)
            outer.addWidget(hint)
            self._shortcuts_dialog = dlg
        # Center the dialog over the parent window.
        parent_geom = self.geometry()
        dlg.move(
            parent_geom.center().x() - dlg.width() // 2,
            parent_geom.center().y() - dlg.height() // 2,
        )
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _reposition_mini_map(self):
        mini = getattr(self, "_mini_map", None)
        parent = getattr(self, "_canvas_widget", None)
        if mini is None or parent is None:
            return
        margin = 12
        bar_h = self._canvas_status_bar.height() if hasattr(self, "_canvas_status_bar") else 0
        x = parent.width() - mini.width() - margin
        y = parent.height() - mini.height() - bar_h - margin
        mini.move(max(0, x), max(0, y))

    def _toggle_mini_map(self, checked):
        mini = getattr(self, "_mini_map", None)
        if mini is None:
            return
        if checked:
            self._reposition_mini_map()
            mini.show()
            mini.raise_()
            mini.update()
        else:
            mini.hide()

    def show_toast(self, message, kind="info", duration_ms=2500):
        """Slide-in toast notification at the bottom-right of the canvas.
        Auto-dismisses after `duration_ms`. kind ∈ {"info","success","warn"}.
        Calling again replaces the current toast."""
        if not message:
            return
        canvas_host = getattr(self, "_canvas_status_bar", None)
        parent = canvas_host.parentWidget() if canvas_host is not None else self
        # Re-use the existing toast widget if still alive, otherwise create one.
        toast = getattr(self, "_toast_label", None)
        if toast is None:
            toast = QLabel(parent)
            toast.setObjectName("ToastNotification")
            toast.setAttribute(Qt.WA_TransparentForMouseEvents)
            toast.setAlignment(Qt.AlignCenter)
            toast.setMargin(8)
            self._toast_label = toast
            self._toast_timer = QTimer(self)
            self._toast_timer.setSingleShot(True)
            self._toast_timer.timeout.connect(self._fade_toast_out)
            self._toast_fade_anim = None
        # Style by kind via dynamic property (theme picks up colors via QSS).
        toast.setProperty("toastKind", str(kind))
        toast.setText(str(message))
        toast.adjustSize()
        # Position bottom-right of the canvas parent, above the status bar.
        margin = 16
        bar_h = self._canvas_status_bar.height() if hasattr(self, "_canvas_status_bar") else 0
        x = parent.width() - toast.width() - margin
        y = parent.height() - toast.height() - bar_h - margin
        toast.move(max(0, x), max(0, y))
        toast.setWindowOpacity(1.0)
        toast.show()
        toast.raise_()
        # Force a fresh style polish in case the kind property changed.
        toast.style().unpolish(toast)
        toast.style().polish(toast)
        self._toast_timer.start(int(duration_ms))

    def _fade_toast_out(self):
        toast = getattr(self, "_toast_label", None)
        if toast is None or not toast.isVisible():
            return
        anim = QPropertyAnimation(toast, b"windowOpacity", self)
        anim.setDuration(400)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.finished.connect(toast.hide)
        anim.start(QPropertyAnimation.DeleteWhenStopped)
        self._toast_fade_anim = anim

    def _on_canvas_cursor_moved(self, x, y):
        if hasattr(self, "_status_xy_label"):
            self._status_xy_label.setText(f"X: {x:.2f}   Y: {y:.2f}")

    def _on_canvas_tool_changed(self, tool_name):
        label_map = {
            "select": "Select / Hand",
            "rectangle": "Rectangle",
            "circle": "Circle",
            "ellipse": "Ellipse",
            "polygon": "Polygon",
            "line": "Line",
            "polyline": "Polyline",
            "arc": "Arc",
            "freeform": "Free Curve",
            "dimension": "Dimension",
            "constraint": "Constraint",
            "zoom_window": "Zoom Window",
            "move": "Move",
            "copy": "Copy",
            "mirror": "Mirror",
        }
        if hasattr(self, "_status_tool_label"):
            label = label_map.get(str(tool_name), str(tool_name).title() or "Select / Hand")
            self._status_tool_label.setText(f"Tool: {label}")

    def _on_canvas_zoom_changed(self, zoom):
        if hasattr(self, "_status_zoom_label"):
            self._status_zoom_label.setText(f"Zoom: {int(zoom * 100)}%")

    def _update_status_selection(self, *_):
        if not hasattr(self, "_status_sel_label"):
            return
        count = 0
        if getattr(self.view, "selected_part_id", None) is not None:
            count = max(count, 1)
        multi = getattr(self.view, "multi_selected_part_ids", None) or set()
        count = max(count, len(multi))
        if count == 0:
            self._status_sel_label.setText("0 selected")
        elif count == 1:
            self._status_sel_label.setText("1 selected")
        else:
            self._status_sel_label.setText(f"{count} selected")

    def _sync_nav_panel_width(self, *_):
        """Resize the Project Navigator panel to exactly fit its longest label.

        Called once at startup (via QTimer) and again whenever the model
        changes (parts / materials / interfaces / BCs added or removed).
        Includes the vertical-scrollbar width so the scrollbar never hides
        the rightmost characters of a label.
        """
        tree     = getattr(self, "project_tree", None)
        panel    = getattr(self, "_model_panel", None)
        splitter = getattr(self, "_main_splitter", None)
        if tree is None or panel is None:
            return
        content_w = tree._compute_preferred_width()
        vsb_w     = tree.verticalScrollBar().sizeHint().width()
        target_w  = max(content_w + vsb_w + 4, 80)
        panel.setMinimumWidth(target_w)
        panel.setMaximumWidth(target_w)
        if splitter is not None:
            sizes = list(splitter.sizes())
            if len(sizes) < 3 or sum(sizes) == 0:
                return
            if sizes[1] != target_w:
                delta    = sizes[1] - target_w
                sizes[1] = target_w
                sizes[2] = max(200, sizes[2] + delta)
                splitter.setSizes(sizes)

    def _toggle_left_panel(self, show):
        """Show/hide the Project Navigator (left side panel). When hidden,
        a thin edge-rail button takes its place so the user can click to
        re-show it. Also wired to Ctrl+B and the View menu."""
        if not hasattr(self, "_model_panel") or self._model_panel is None:
            return
        show = bool(show)
        self._model_panel.setVisible(show)
        if hasattr(self, "_left_rail") and self._left_rail is not None:
            self._left_rail.setVisible(not show)
        action = getattr(self, "_view_left_panel_action", None)
        if action is not None and action.isChecked() != show:
            action.blockSignals(True)
            action.setChecked(show)
            action.blockSignals(False)

    def _toggle_right_panel(self, show):
        """Show/hide the Properties panel (right side). Re-opened via the
        edge-rail, Ctrl+J shortcut, or View menu."""
        if not hasattr(self, "_properties_dock") or self._properties_dock is None:
            return
        show = bool(show)
        self._properties_dock.setVisible(show)
        if hasattr(self, "_right_rail") and self._right_rail is not None:
            self._right_rail.setVisible(not show)
        action = getattr(self, "_view_right_panel_action", None)
        if action is not None and action.isChecked() != show:
            action.blockSignals(True)
            action.setChecked(show)
            action.blockSignals(False)

    def _toggle_mesh_view(self, enabled):
        if self._workspace_3d:
            if enabled and hasattr(self, "mesh_view_action"):
                self.mesh_view_action.blockSignals(True)
                self.mesh_view_action.setChecked(False)
                self.mesh_view_action.blockSignals(False)
            return
        self.view.stop_visualization()
        self.view.ensure_y_axis_up()
        if enabled and hasattr(self, "mesh_3d_action"):
            self.mesh_3d_action.blockSignals(True)
            self.mesh_3d_action.setChecked(False)
            self.mesh_3d_action.blockSignals(False)
        if enabled:
            self.view.set_display_mode("mesh")
            self._sync_point_preview(refresh_data=True)
        else:
            self.view.set_display_mode("geometry")
            if hasattr(self, "view_stack"):
                self.view_stack.setCurrentWidget(self.view)

    def _set_mesh_nodes_visibility(self, enabled):
        self.view.set_mesh_view_visibility(show_nodes=enabled)
        if self.view_3d is not None:
            self.view_3d.set_visibility(show_nodes=enabled)
        self._sync_point_preview(refresh_data=False)

    def _set_mesh_elements_visibility(self, enabled):
        self.view.set_mesh_view_visibility(show_mesh=enabled)
        if self.view_3d is not None:
            self.view_3d.set_visibility(show_mesh=enabled)
        self._sync_point_preview(refresh_data=False)

    def _on_results_frame_load_started(self, frame_index):
        if getattr(self.view, "display_mode", "") != "results":
            return
        total = 0
        try:
            total = int(self.results_controller.frame_count())
        except Exception:
            total = 0
        message = f"Loading frame {int(frame_index) + 1}"
        if total > 0:
            message = f"{message}/{total}"
        self.statusBar().showMessage(message, 1500)

    def _on_results_frame_load_failed(self, frame_index, error_message):
        self.statusBar().showMessage(
            f"Failed to load frame {int(frame_index) + 1}: {error_message}",
            4000,
        )

    def _on_replay_particle_selected(self, info):
        if not info:
            self.statusBar().showMessage("Replay selection cleared", 1200)
            return
        particle_id = info.get("particle_id", "--")
        position = info.get("position")
        velocity = info.get("velocity")
        try:
            pos_text = f"({float(position[0]):.4g}, {float(position[1]):.4g})"
        except Exception:
            pos_text = "--"
        try:
            vel_text = f"({float(velocity[0]):.4g}, {float(velocity[1]):.4g})"
        except Exception:
            vel_text = "--"
        self.statusBar().showMessage(
            f"Particle {particle_id}  Pos {pos_text}  Vel {vel_text}",
            2500,
        )

    def _show_results_point_preview(self, points, scalars=None, auto_fit=False):
        if not hasattr(self, "view_stack"):
            return False
        if self.point_view is None:
            self.point_view = PointCloudView2D()
            self.view_stack.addWidget(self.point_view)
        try:
            if scalars is not None:
                self.point_view.set_points_with_scalars(points, scalars, auto_fit=auto_fit)
            else:
                self.point_view.set_points(points, auto_fit=auto_fit)
        except Exception:
            try:
                self.point_view.clear_points()
            except Exception:
                pass
            return False
        self._results_point_preview_active = True
        self.view_stack.setCurrentWidget(self.point_view)
        return True

    def _hide_results_point_preview(self, *, clear_points=False, force_view=False):
        was_active = bool(getattr(self, "_results_point_preview_active", False))
        self._results_point_preview_active = False
        if clear_points and self.point_view is not None:
            try:
                self.point_view.clear_points()
            except Exception:
                pass
        if (force_view or was_active) and hasattr(self, "view_stack"):
            if self.point_view is not None and self.view_stack.currentWidget() == self.point_view:
                self.view_stack.setCurrentWidget(self.view)

    def _sync_point_preview(self, refresh_data=False):
        if not hasattr(self, "view_stack"):
            return
        if not getattr(self, "_ui_ready", True):
            return
        if self.view.display_mode == "results":
            return
        if getattr(self, "_results_point_preview_active", False):
            self._hide_results_point_preview(force_view=True)
        if self._workspace_3d or self.view.display_mode != "mesh":
            if self.point_view is not None and self.view_stack.currentWidget() == self.point_view:
                self.view_stack.setCurrentWidget(self.view)
            return
        use_gpu = False
        if hasattr(self.view, "should_use_gpu_point_preview"):
            use_gpu = self.view.should_use_gpu_point_preview()
        if use_gpu:
            if self.point_view is None:
                self.point_view = PointCloudView2D()
                self.view_stack.addWidget(self.point_view)
            auto_fit = refresh_data or self.view_stack.currentWidget() != self.point_view
            try:
                self.point_view.set_points(self.view.global_nodes, auto_fit=auto_fit)
            except Exception:
                self.point_view.clear_points()
            self.view_stack.setCurrentWidget(self.point_view)
        else:
            if self.view_stack.currentWidget() != self.view:
                self.view_stack.setCurrentWidget(self.view)

    def _reset_point_preview_view(self):
        if not hasattr(self, "point_view") or self.point_view is None:
            return
        if not hasattr(self.view, "should_use_gpu_point_preview"):
            return
        if not self.view.should_use_gpu_point_preview():
            return
        if self.view.global_nodes is None or len(self.view.global_nodes) == 0:
            return
        try:
            self.point_view.set_points(self.view.global_nodes, auto_fit=True)
        except Exception:
            pass

    def _frame_selection(self):
        if hasattr(self, "view_stack"):
            current = self.view_stack.currentWidget()
            if current is getattr(self, "point_view", None):
                self._reset_point_preview_view()
                return
            if current is getattr(self, "view_3d", None):
                if hasattr(self.view_3d, "fit_view"):
                    self.view_3d.fit_view()
                    return
        self.view.fit_selection()

    def _fit_screen(self):
        if hasattr(self, "view_stack"):
            current = self.view_stack.currentWidget()
            if current is getattr(self, "point_view", None):
                points = None
                packet = getattr(self.view, "_current_frame_packet", {}) or {}
                for key in ("raw_positions", "display_positions"):
                    candidate = packet.get(key)
                    if candidate is not None and len(candidate) > 0:
                        points = candidate
                        break
                if points is None:
                    nodes = getattr(self.view, "global_nodes", None)
                    if nodes is not None and len(nodes) > 0:
                        points = nodes
                if points is not None:
                    try:
                        scalars = getattr(self.view, "_result_scalar_values", None)
                        scalar_arr = np.asarray(scalars, dtype=float).reshape(-1) if scalars is not None else None
                        if scalar_arr is not None and len(scalar_arr) == len(points):
                            self.point_view.set_points_with_scalars(points, scalar_arr, auto_fit=True)
                        else:
                            self.point_view.set_points(points, auto_fit=True)
                    except Exception:
                        try:
                            self.point_view.set_points(points, auto_fit=True)
                        except Exception:
                            pass
                    return
            if current is getattr(self, "view_3d", None):
                if hasattr(self.view_3d, "fit_view"):
                    self.view_3d.fit_view()
                    return
        self.view.fit_view()

    def _center_origin(self):
        if hasattr(self, "view_stack"):
            current = self.view_stack.currentWidget()
            if current is getattr(self, "view_3d", None):
                if hasattr(self.view_3d, "center_origin"):
                    self.view_3d.center_origin()
                    return
        self.view.center_origin()

    def _toggle_mesh_view_3d(self, enabled):
        if self._workspace_3d:
            if enabled and hasattr(self, "mesh_3d_action"):
                self.mesh_3d_action.blockSignals(True)
                self.mesh_3d_action.setChecked(False)
                self.mesh_3d_action.blockSignals(False)
            return
        self.view.stop_visualization()
        self.view.ensure_y_axis_up()
        if enabled and hasattr(self, "mesh_view_action"):
            self.mesh_view_action.blockSignals(True)
            self.mesh_view_action.setChecked(False)
            self.mesh_view_action.blockSignals(False)
        if enabled:
            view_3d = self._ensure_view_3d()
            if view_3d is None:
                return
            if not self.view.preview_3d_mesh():
                self.mesh_3d_action.blockSignals(True)
                self.mesh_3d_action.setChecked(False)
                self.mesh_3d_action.blockSignals(False)
                if hasattr(self, "view_stack"):
                    self.view_stack.setCurrentIndex(0)
                return
            if hasattr(self, "view_stack"):
                self.view_stack.setCurrentWidget(view_3d)
            show_nodes = True
            show_mesh = True
            if hasattr(self, "mesh_nodes_action"):
                show_nodes = self.mesh_nodes_action.isChecked()
            if hasattr(self, "mesh_elements_action"):
                show_mesh = self.mesh_elements_action.isChecked()
            view_3d.set_visibility(show_nodes=show_nodes, show_mesh=show_mesh)
            self.statusBar().showMessage(
                "3D view: drag to rotate, right-drag to pan, wheel to zoom. Double-click for random view, right-click for standard views.",
                self._status_timeout_ms,
            )
        else:
            self.view.set_display_mode("geometry")
            if hasattr(self, "view_stack"):
                self.view_stack.setCurrentWidget(self.view)

    def _toggle_workspace_3d(self, enabled):
        if enabled:
            if self.project_mode != "3d":
                self.project_mode = "3d"
                self.view.set_project_mode(self.project_mode)
                self._sync_mode_ui()
            view_3d = self._ensure_view_3d()
            self._workspace_3d = True
            self._sync_model3d_state()
            if hasattr(self, "mesh_view_action"):
                self.mesh_view_action.blockSignals(True)
                self.mesh_view_action.setChecked(False)
                self.mesh_view_action.setEnabled(False)
                self.mesh_view_action.blockSignals(False)
            if hasattr(self, "mesh_3d_action"):
                self.mesh_3d_action.blockSignals(True)
                self.mesh_3d_action.setChecked(False)
                self.mesh_3d_action.setEnabled(False)
                self.mesh_3d_action.blockSignals(False)
            if hasattr(self, "view_stack") and view_3d is not None:
                self.view_stack.setCurrentWidget(view_3d)
            if hasattr(self, "primitive_dock"):
                try:
                    if self.primitive_dock.isFloating():
                        self.addDockWidget(Qt.LeftDockWidgetArea, self.primitive_dock)
                        self.primitive_dock.setFloating(False)
                except Exception:
                    pass
                self.primitive_dock.setVisible(True)
            self._update_grid_settings()
            self._refresh_3d_view()
            self._refresh_primitive_list()
            if hasattr(self, "sketch_toolbar"):
                self.sketch_toolbar.setVisible(False)
            if hasattr(self, "loads_toolbar"):
                self.loads_toolbar.setVisible(False)
            if hasattr(self, "command_bar"):
                self.toggle_command_bar(False)
            if hasattr(self, "command_action"):
                self.command_action.setEnabled(False)
            self.statusBar().showMessage(
                "3D workspace: primitives/booleans coming next. 3D navigation: drag to rotate, right-drag to pan.",
                self._status_timeout_ms,
            )
        else:
            self._workspace_3d = False
            if hasattr(self, "mesh_view_action"):
                self.mesh_view_action.setEnabled(True)
            if hasattr(self, "mesh_3d_action"):
                self.mesh_3d_action.setEnabled(self.project_mode == "3d")
            if hasattr(self, "view_stack"):
                self.view_stack.setCurrentWidget(self.view)
            if hasattr(self, "primitive_dock"):
                self.primitive_dock.setVisible(False)
            if hasattr(self, "sketch_toolbar"):
                self.sketch_toolbar.setVisible(True)
            if hasattr(self, "loads_toolbar"):
                self.loads_toolbar.setVisible(True)
            if hasattr(self, "command_action"):
                self.command_action.setEnabled(True)
                if self.command_action.isChecked() and hasattr(self, "command_bar"):
                    self.toggle_command_bar(True)
        if hasattr(self, "properties_panel"):
            self.properties_panel.bcs_tab.set_viewport(self.view_3d if self._workspace_3d else None)
            self.properties_panel.bcs_tab.set_workspace_mode(self._workspace_3d)
            self.properties_panel.loads_tab.set_viewport(self.view_3d if self._workspace_3d else None)
            self.properties_panel.loads_tab.set_workspace_mode(self._workspace_3d)
        self._set_3d_sections_visible(self._workspace_3d)
        self._sync_workspace_ui()
        self._sync_project_dimension_state()
        if hasattr(self, "properties_panel") and hasattr(self.properties_panel, "assembly_tab"):
            try:
                self.properties_panel.assembly_tab.refresh_project_setup()
            except Exception:
                pass
        self._set_stage_toolbar_visibility(getattr(self, "_workflow_stage_key", "geometry"))
        self._refresh_workflow_architecture()
        if hasattr(self, "_settings"):
            self._settings.setValue("workspace/last_3d", bool(self._workspace_3d))

    def _sync_workspace_ui(self):
        if hasattr(self, "workspace_3d_action"):
            self.workspace_3d_action.blockSignals(True)
            self.workspace_3d_action.setChecked(self._workspace_3d)
            self.workspace_3d_action.blockSignals(False)
        if hasattr(self, "workspace_2d_action"):
            self.workspace_2d_action.blockSignals(True)
            self.workspace_2d_action.setChecked(not self._workspace_3d)
            self.workspace_2d_action.blockSignals(False)
    
    def _show_quick_start_dialog(self):
        show_on_startup = self._settings.value("show_quick_start", True, type=bool)
        dialog = QuickStartDialog(show_on_startup, self)
        dialog.exec()
        self._settings.setValue("show_quick_start", dialog.show_on_startup())

    def _show_quick_start_if_needed(self):
        if self._settings.value("show_quick_start", True, type=bool):
            self._show_quick_start_dialog()

    def _show_dependency_check_dialog(self):
        dialog = DependencyCheckDialog(self)
        dialog.exec()

    def _show_startup_dialog(self):
        if self._prompt_restore_autosave():
            return
        show_on_startup = self._settings.value("show_startup", True, type=bool)
        if not show_on_startup:
            return
        rows, required_missing, _ = collect_dependency_report()
        if required_missing > 0:
            dialog = DependencyCheckDialog(self)
            dialog.exec()
        self._update_recent_menu()
        dialog = StartupDialog(self.recent_projects, show_on_startup, self)
        if dialog.exec():
            if dialog.action == "new_2d":
                self.new_project(mode="2d")
            elif dialog.action == "new_3d":
                self.new_project(mode="3d")
            elif dialog.action == "open_recent" and dialog.selected_path:
                self._load_project_from_path(dialog.selected_path)
        self._settings.setValue("show_startup", dialog.show_on_startup())

    def _run_startup_prompts(self):
        if self._prompt_restore_autosave():
            return
        rows, required_missing, _ = collect_dependency_report()
        if required_missing > 0:
            dialog = DependencyCheckDialog(self)
            dialog.exec()

    def _open_test_guide(self):
        guide_path = os.path.join(os.path.dirname(__file__), "TEST_RUN_GUIDE.md")
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(guide_path)))

    def schedule_startup_update_check(self):
        if not UPDATE_CHECK_ON_STARTUP:
            return
        QTimer.singleShot(2500, lambda: self._start_update_check(manual=False))

    def _check_for_updates_action(self):
        self._start_update_check(manual=True)

    def _start_update_check(self, manual=False):
        if self._update_worker is not None and self._update_worker.isRunning():
            if manual:
                QMessageBox.information(
                    self,
                    "Check for Updates",
                    "Update check is already running.",
                )
            return

        repo = (UPDATE_REPO or "").strip()
        if not repo:
            if manual:
                QMessageBox.information(
                    self,
                    "Check for Updates",
                    "Update repository is not configured.\nSet CPD_UPDATE_REPO to owner/repo.",
                )
            return

        self._update_check_manual = bool(manual)
        self._update_worker = UpdateCheckWorker(
            repo=repo,
            current_version=APP_BUILD_VERSION,
            timeout_sec=UPDATE_REQUEST_TIMEOUT_SEC,
        )
        self._update_worker.completed.connect(self._on_update_check_finished)
        self._update_worker.finished.connect(self._on_update_worker_finished)
        self._update_worker.start()

        if manual:
            self.statusBar().showMessage("Checking for updates...", self._status_timeout_ms)

    def _on_update_worker_finished(self):
        worker = self._update_worker
        self._update_worker = None
        if worker is not None:
            worker.deleteLater()

    def _stop_background_threads(self):
        if getattr(self, "_stopping_background_threads", False):
            return
        self._stopping_background_threads = True
        controller = getattr(self, "results_controller", None)
        if controller is not None and hasattr(controller, "stop"):
            try:
                controller.stop()
            except Exception:
                pass
        solver_controller = getattr(self, "solver_controller", None)
        if solver_controller is not None and hasattr(solver_controller, "stop"):
            try:
                solver_controller.stop()
            except Exception:
                pass
        view = getattr(self, "view", None)
        if view is not None and hasattr(view, "stop_background_tasks"):
            try:
                view.stop_background_tasks()
            except Exception:
                pass
        worker = getattr(self, "_update_worker", None)
        self._update_worker = None
        if worker is not None:
            try:
                worker.stop()
            except Exception:
                pass
            try:
                worker.deleteLater()
            except Exception:
                pass
        self._stopping_background_threads = False

    def _on_update_check_finished(self, release_info_obj, error_text):
        manual = self._update_check_manual
        self._update_check_manual = False

        if error_text:
            logging.getLogger(__name__).info("Update check error: %s", error_text)
            if manual:
                QMessageBox.warning(
                    self,
                    "Check for Updates",
                    f"Update check failed:\n{error_text}",
                )
            return

        release_info = release_info_obj if isinstance(release_info_obj, ReleaseInfo) else None
        if not release_info:
            if manual:
                QMessageBox.information(
                    self,
                    "Check for Updates",
                    f"You are on the latest version ({APP_BUILD_VERSION}).",
                )
            return

        preferred_asset = select_preferred_asset(release_info.assets)
        target_url = preferred_asset.url if preferred_asset else (release_info.html_url or "")
        display_target = preferred_asset.name if preferred_asset else "release page"

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle("Update Available")
        box.setText(
            f"A newer version is available: {release_info.version}\nCurrent version: {APP_BUILD_VERSION}"
        )
        box.setInformativeText(
            f"Published: {release_info.published_at or 'unknown'}\n"
            f"Suggested download: {display_target}\n\nOpen it now?"
        )
        open_btn = box.addButton("Open Download", QMessageBox.AcceptRole)
        box.addButton("Later", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() == open_btn and target_url:
            QDesktopServices.openUrl(QUrl(target_url))

    # =========================
    # MENU BAR
    # =========================
    def _create_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        file_menu.addAction(QAction("New", self, shortcut="Ctrl+N", triggered=self.new_project))
        file_menu.addAction(QAction("Open...", self, shortcut="Ctrl+O", triggered=self.load_project))
        recover_action = QAction("Recover Autosave...", self)
        recover_action.triggered.connect(self._recover_autosave)
        file_menu.addAction(recover_action)
        file_menu.addAction(QAction("Import...", self, shortcut="Ctrl+I", triggered=self.import_geometry))
        file_menu.addAction(QAction("Import 3D CAD...", self, triggered=self.import_cad_geometry))

        self.recent_menu = QMenu("Open Recent", self)
        file_menu.addMenu(self.recent_menu)
        self._update_recent_menu()
        file_menu.addSeparator()

        file_menu.addAction(QAction("Save", self, shortcut="Ctrl+S", triggered=self.save_project))
        file_menu.addAction(QAction("Save As...", self, shortcut="Ctrl+Shift+S", triggered=self.save_project_as))     

        save_up_to_action = QAction("Save Up To...", self)
        save_up_to_action.triggered.connect(self.save_project_up_to)
        file_menu.addAction(save_up_to_action)        

        file_menu.addSeparator()
        file_menu.addAction("Export", self.view.export_csv)

        file_menu.addSeparator()
        file_menu.addAction(QAction("Exit", self, shortcut="Ctrl+Q", triggered=self.close))

        view_menu = menubar.addMenu("View")

        # --- Actions defined for toolbar / shortcuts (NOT added to View menu) ---
        workspace_group = QActionGroup(self)
        workspace_group.setExclusive(True)
        self.workspace_2d_action = QAction(get_icon("workspace_2d"), "2D Workspace", self)
        self.workspace_2d_action.setCheckable(True)
        self.workspace_2d_action.setChecked(True)
        self.workspace_2d_action.setShortcut("Ctrl+1")
        self.workspace_2d_action.triggered.connect(lambda: self._toggle_workspace_3d(False))
        self._bind_action_tip(self.workspace_2d_action, "Switch to the 2D sketching workspace.")
        workspace_group.addAction(self.workspace_2d_action)
        self.workspace_3d_action = QAction(get_icon("workspace_3d"), "3D Workspace", self)
        self.workspace_3d_action.setCheckable(True)
        self.workspace_3d_action.setShortcut("Ctrl+2")
        self.workspace_3d_action.triggered.connect(lambda: self._toggle_workspace_3d(True))
        self._bind_action_tip(self.workspace_3d_action, "Switch to the 3D modeling workspace.")
        workspace_group.addAction(self.workspace_3d_action)

        fit_action = QAction(get_icon("fit"), "Fit Screen", self)
        fit_action.setShortcuts(["Ctrl+0", "Home"])
        fit_action.triggered.connect(self._fit_screen)
        self._bind_action_tip(fit_action, "Fit the full model to the window (Ctrl+0 / Home).")
        self.fit_action = fit_action

        frame_action = QAction(get_icon("frame"), "Frame Selection", self)
        frame_action.setShortcut("F")
        frame_action.triggered.connect(self._frame_selection)
        self._bind_action_tip(frame_action, "Frame the selected part or geometry.")
        self.frame_action = frame_action

        zoom_action = QAction(get_icon("zoom_window"), "Zoom Window", self)
        zoom_action.triggered.connect(lambda: self.view.set_tool("zoom_window"))
        self._bind_action_tip(zoom_action, "Drag a rectangle to zoom. Tip: Ctrl + left-drag for temporary window zoom.")
        self.zoom_action = zoom_action

        measure_action = QAction(get_icon("measure"), "Measure", self)
        measure_action.setShortcut("M")
        measure_action.setShortcutContext(Qt.ApplicationShortcut)
        measure_action.triggered.connect(lambda: self.view.set_tool("measure"))
        self._bind_action_tip(measure_action, "Measure distance and angle between two points (M). Right-click to exit.")
        self.addAction(measure_action)
        self.measure_action = measure_action

        self._mini_map_action = QAction(get_icon("minimap"), "Mini-Map", self)
        self._mini_map_action.setCheckable(True)
        self._mini_map_action.setChecked(False)
        self._mini_map_action.setShortcut("Ctrl+M")
        self._mini_map_action.setShortcutContext(Qt.ApplicationShortcut)
        self._mini_map_action.toggled.connect(self._toggle_mini_map)
        self._bind_action_tip(self._mini_map_action, "Show or hide the mini-map overview (Ctrl+M).")
        self.addAction(self._mini_map_action)
        self.mini_map_action = self._mini_map_action

        full_action = QAction(get_icon("full_screen"), "Full Screen", self)
        full_action.setShortcut("F11")
        full_action.triggered.connect(self.toggle_full_screen)
        self._bind_action_tip(full_action, "Toggle full screen mode.")
        self.full_action = full_action
        self._refresh_full_screen_action()

        command_action = QAction(get_icon("command_bar"), "Command Bar", self)
        command_action.setCheckable(True)
        command_action.setChecked(False)
        command_action.toggled.connect(self.toggle_command_bar)
        self._bind_action_tip(command_action, "Show or hide the command input bar.")
        self.command_action = command_action

        origin_action = QAction(get_icon("origin"), "Center Origin", self)
        origin_action.setShortcut("Ctrl+Home")
        origin_action.triggered.connect(self._center_origin)
        self._bind_action_tip(origin_action, "Center the view on the origin.")
        self.origin_action = origin_action

        grid_action = QAction(get_icon("snap_grid"), "Grid", self)
        grid_action.setCheckable(True)
        grid_action.setChecked(True)
        grid_action.toggled.connect(self.view.set_grid_visible)
        self._bind_action_tip(grid_action, "Show or hide the sketch grid.")
        self.grid_action = grid_action

        dim_action = QAction(get_icon("dimension"), "Dimensions", self)
        dim_action.setCheckable(True)
        dim_action.setChecked(True)
        dim_action.toggled.connect(self.view.set_dimensions_visible)
        self._bind_action_tip(dim_action, "Show or hide sketch dimensions.")
        self.dim_action = dim_action

        hints_action = QAction("Stage Hints", self)
        hints_action.setCheckable(True)
        hints_action.setChecked(bool(getattr(self, "_stage_hints_enabled", True)))
        hints_action.toggled.connect(
            lambda enabled: self._set_stage_hints_enabled(enabled, announce=True)
        )
        self._bind_action_tip(hints_action, "Show short helper instructions when switching stages.")
        self.stage_hints_action = hints_action

        # --- View menu items (only items NOT already on the top toolbar) ---
        self._view_left_panel_action = QAction("Project Navigator", self)
        self._view_left_panel_action.setCheckable(True)
        self._view_left_panel_action.setChecked(True)
        self._view_left_panel_action.setShortcut("Ctrl+B")
        self._view_left_panel_action.toggled.connect(self._toggle_left_panel)
        self._bind_action_tip(self._view_left_panel_action, "Show or hide the Project Navigator (left panel).")
        view_menu.addAction(self._view_left_panel_action)

        self._view_right_panel_action = QAction("Properties Panel", self)
        self._view_right_panel_action.setCheckable(True)
        self._view_right_panel_action.setChecked(True)
        self._view_right_panel_action.setShortcut("Ctrl+J")
        self._view_right_panel_action.toggled.connect(self._toggle_right_panel)
        self._bind_action_tip(self._view_right_panel_action, "Show or hide the Properties panel (right side).")
        view_menu.addAction(self._view_right_panel_action)

        view_menu.addSeparator()

        self._combine_parts_action = QAction("Combine Parts", self)
        self._combine_parts_action.setShortcut("Ctrl+P")
        self._combine_parts_action.setShortcutContext(Qt.ApplicationShortcut)
        self._combine_parts_action.triggered.connect(
            lambda: self.view.combine_selected_parts()
        )
        self._bind_action_tip(self._combine_parts_action, "Combine selected parts into one (Ctrl+Click multiple, then Ctrl+P).")
        self.addAction(self._combine_parts_action)
        view_menu.addAction(self._combine_parts_action)

        self._shortcuts_action = QAction("Keyboard Shortcuts", self)
        self._shortcuts_action.setShortcuts([QKeySequence("?"), QKeySequence("Shift+/"), QKeySequence("F1")])
        self._shortcuts_action.setShortcutContext(Qt.ApplicationShortcut)
        self._shortcuts_action.triggered.connect(self.show_shortcuts_overlay)
        self._bind_action_tip(self._shortcuts_action, "Show keyboard shortcuts (?, Shift+/, F1).")
        self.addAction(self._shortcuts_action)
        view_menu.addAction(self._shortcuts_action)

        help_menu = menubar.addMenu("Help")
        quick_action = QAction("Quick Start", self)
        quick_action.triggered.connect(self._show_quick_start_dialog)
        self._bind_action_tip(quick_action, "Open the quick start guide.")
        help_menu.addAction(quick_action)
        dep_action = QAction("Dependency Check", self)
        dep_action.triggered.connect(self._show_dependency_check_dialog)
        self._bind_action_tip(dep_action, "Show dependency status and versions.")
        help_menu.addAction(dep_action)
        guide_action = QAction("Open Test Guide", self)
        guide_action.triggered.connect(self._open_test_guide)
        self._bind_action_tip(guide_action, "Open the test run guide file.")
        help_menu.addAction(guide_action)
        update_action = QAction("Check for Updates", self)
        update_action.triggered.connect(self._check_for_updates_action)
        self._bind_action_tip(update_action, "Check GitHub releases for a newer app version.")
        help_menu.addAction(update_action)

    def save_project_up_to(self):
        dialog = SaveUpToDialog(self)
        if dialog.exec():
            stage = dialog.get_selected_stage()
            self._save_project_with_stage(stage)

    def _save_project_with_stage(self, stage):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project",
            "",
            "CPD Project (*.cpd)"
        )

        if not file_path:
            return
        if not file_path.lower().endswith(".cpd"):
            file_path = f"{file_path}.cpd"

        project_data = {
            "project_meta": {
                "version": "2.0",
                "schema_version": CURRENT_SCHEMA_VERSION,
                "app_version": APP_VERSION,
                "last_stage": stage.name,
                "analysis_type": self._normalized_analysis_type(),
                "dimension": self._normalized_dimension(),
                "mode": self.project_mode,
                "workspace": "3d" if getattr(self, "_workspace_3d", False) else "2d",
            }
        }
        project_data["model3d"] = self.model3d

        # ---------------- GEOMETRY ----------------
        project_data["units"] = "m"
        project_data["sketches"] = self.view.sketches
        project_data["sketch_meta"] = self.view.sketch_meta
        project_data["dimensions"] = self.view.dimensions
        project_data["constraints"] = self.view.constraints
        project_data["show_dimensions"] = self.view.show_dimensions
        project_data["geometry"] = self.view.serialize_geometry()
        project_data["modeling"] = {
            "extrude_height": float(self.view.extrude_height),
            "extrude_layers": int(self.view.extrude_layers),
        }
        project_data["connection_settings"] = {
            "min_spacing_factor": float(getattr(self.view, "mesh_min_spacing_factor", 1.0)),
            "boundary_thickness": float(getattr(self.view, "mesh_boundary_thickness", 0.0)),
            "boundary_spacing_factor": float(getattr(self.view, "mesh_boundary_spacing_factor", 1.0)),
        }
        project_data["preview_settings"] = {
            "fast_preview_enabled": bool(getattr(self.view, "fast_preview_enabled", True)),
            "fast_preview_connection_limit": int(
                getattr(self.view, "fast_preview_connection_limit", 0)
            ),
            "gpu_point_preview_enabled": bool(
                getattr(self.view, "gpu_point_preview_enabled", True)
            ),
            "gpu_point_preview_auto": bool(
                getattr(self.view, "gpu_point_preview_auto", True)
            ),
            "gpu_point_preview_threshold": int(
                getattr(self.view, "gpu_point_preview_threshold", 0)
            ),
            "freeform_auto_convert_enabled": bool(
                getattr(self.view, "freeform_auto_convert_enabled", False)
            ),
        }
        self._warn_project_state_validation("save")
        schema_state = self._build_schema_project_state()
        merge_project_state_into_project_data(project_data, schema_state)

        stage_rank = self._workflow_stage_rank(stage)
        if stage_rank < self._workflow_stage_rank(ProjectStage.MATERIALS):
            project_data.pop("materials", None)
        if stage_rank < self._workflow_stage_rank(ProjectStage.INTERFACES):
            project_data.pop("interfaces", None)
        if stage_rank < self._workflow_stage_rank(ProjectStage.LOADS):
            project_data.pop("bcs", None)
            project_data.pop("loads", None)
            project_data.pop("initial_velocities", None)
        else:
            project_data["initial_velocities"] = self.view.initial_velocities

        if stage_rank >= self._workflow_stage_rank(ProjectStage.MESH):
            self.view.export_mesh_csv(silent=True)
            project_data["connections"] = {
                "particles": f"{WORKSPACE_DIR_NAME}/input/particles.csv",
                "connections": f"{WORKSPACE_DIR_NAME}/input/connections.csv",
            }

        if stage_rank >= self._workflow_stage_rank(ProjectStage.JOB):
            project_data["job"] = {
                "results_dir": f"{WORKSPACE_DIR_NAME}/output/results/",
                "completed": stage_rank >= self._workflow_stage_rank(ProjectStage.RESULTS),
            }

        project_data = apply_schema_migrations(project_data)

        with open(file_path, "w") as f:
            json.dump(project_data, f, indent=4)

        export_inputs = stage_rank >= self._workflow_stage_rank(ProjectStage.LOADS)
        include_results = stage_rank >= self._workflow_stage_rank(ProjectStage.RESULTS)
        self._export_job_artifacts(
            file_path,
            export_inputs=export_inputs,
            include_results=include_results,
        )

        self._add_recent_project(file_path)
        self.project_dirty = False

    def _load_project_from_path(self, file_path, *, show_message=True):
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "Error", "Project file not found.")
            return False
    
        try:
            logging.getLogger(__name__).info("Project restore started: %s", file_path)
            with open(file_path, "r") as f:
                project_data = json.load(f)
            project_data = apply_schema_migrations(project_data)
            loaded_state = project_state_from_project_data(project_data)
    
            # FULL RESET
            self.view.clear_all()
            self._reset_stage_panels(clear_bc_lists=True)
            self.project_state = ProjectState.from_dict(loaded_state.to_dict())
            self.view.project_state = self.project_state
            if hasattr(self, "properties_panel") and hasattr(self.properties_panel, "set_project_state"):
                self.properties_panel.set_project_state(self.project_state)
            if hasattr(self, "project_tree") and hasattr(self.project_tree, "set_sources"):
                self.project_tree.set_sources(self.view, self.project_state)
            if hasattr(self, "property_inspector") and hasattr(self.property_inspector, "set_project_state"):
                self.property_inspector.set_project_state(self.project_state)
    
            # ---- META ----
            meta = project_data.get("project_meta", {})
            stage_name = meta.get("last_stage", "GEOMETRY")
            mode = meta.get("mode", "2d")
            if file_path.lower().endswith(".cpdproj"):
                mode = "2d"
            self.project_mode = "3d" if str(mode).lower().startswith("3") else "2d"
            self.view.set_project_mode(self.project_mode)
            self._sync_mode_ui()

            workspace_pref = meta.get("workspace")
            if workspace_pref is None:
                workspace_pref = meta.get("last_workspace")
            workspace_3d = None
            if isinstance(workspace_pref, str):
                if workspace_pref.lower().startswith("3"):
                    workspace_3d = True
                elif workspace_pref.lower().startswith("2"):
                    workspace_3d = False
            elif isinstance(workspace_pref, bool):
                workspace_3d = workspace_pref
            if workspace_3d is None:
                if hasattr(self, "_settings") and self._settings.contains("workspace/last_3d"):
                    workspace_3d = self._settings.value("workspace/last_3d", False, type=bool)
                else:
                    workspace_3d = self.project_mode == "3d"

            # ---- 3D MODEL ----
            model3d = project_data.get("model3d")
            if isinstance(model3d, dict):
                self.model3d = model3d
            else:
                self.model3d = self._new_model3d()
            self._sync_model3d_state()
            self._undo_stack_3d.clear()
            self._redo_stack_3d.clear()
            self._toggle_workspace_3d(workspace_3d)
            self.project_state.analysis_type = self._normalized_analysis_type(
                meta.get("analysis_type", getattr(self.project_state, "analysis_type", "static"))
            )
            self.project_state.dimension = self._normalized_dimension(
                meta.get("dimension", getattr(self.project_state, "dimension", "2D"))
            )
            if hasattr(self, "properties_panel") and hasattr(self.properties_panel, "assembly_tab"):
                try:
                    self.properties_panel.assembly_tab.refresh_project_setup()
                except Exception:
                    pass

            modeling = project_data.get("modeling", {})
            if "extrude_height" in modeling:
                self.view.extrude_height = float(modeling.get("extrude_height", self.view.extrude_height))
            if "extrude_layers" in modeling:
                self.view.extrude_layers = int(modeling.get("extrude_layers", self.view.extrude_layers))
            connection_settings = project_data.get("connection_settings")
            if connection_settings is None:
                connection_settings = project_data.get("mesh_settings", {})
            if connection_settings:
                self.view.set_mesh_generation_settings(
                    min_spacing_factor=connection_settings.get("min_spacing_factor"),
                    boundary_thickness=connection_settings.get("boundary_thickness"),
                    boundary_spacing_factor=connection_settings.get("boundary_spacing_factor"),
                )
            preview_settings = project_data.get("preview_settings", {})
            if preview_settings:
                self.view.set_fast_preview(
                    enabled=preview_settings.get("fast_preview_enabled"),
                    limit=preview_settings.get("fast_preview_connection_limit"),
                )
                if hasattr(self.view, "set_gpu_point_preview_settings"):
                    self.view.set_gpu_point_preview_settings(
                        enabled=preview_settings.get("gpu_point_preview_enabled"),
                        auto=preview_settings.get("gpu_point_preview_auto"),
                        threshold=preview_settings.get("gpu_point_preview_threshold"),
                    )
                elif "gpu_point_preview_enabled" in preview_settings:
                    self.view.set_gpu_point_preview(preview_settings.get("gpu_point_preview_enabled"))
                if "freeform_auto_convert_enabled" in preview_settings:
                    auto_convert_enabled = bool(
                        preview_settings.get("freeform_auto_convert_enabled")
                    )
                    if hasattr(self.view, "set_freeform_auto_convert"):
                        self.view.set_freeform_auto_convert(auto_convert_enabled, announce=False)
                    else:
                        self.view.freeform_auto_convert_enabled = auto_convert_enabled
                    action = getattr(self, "freeform_autoconvert_action", None)
                    if action is not None:
                        action.blockSignals(True)
                        action.setChecked(auto_convert_enabled)
                        action.blockSignals(False)
    
            # ---- UNITS ----
            self.view.set_unit("m")
    
            # ---- GEOMETRY ----
            self.view.sketches = project_data.get("sketches", [])
            self.view.sketch_meta = project_data.get("sketch_meta", [])
            self.view.dimensions = project_data.get("dimensions", [])
            self.view.constraints = project_data.get("constraints", [])
            self.view.show_dimensions = bool(project_data.get("show_dimensions", True))
            if hasattr(self, "dim_action"):
                self.dim_action.setChecked(self.view.show_dimensions)
            self.view.deserialize_geometry(project_data.get("geometry", {}))
            self.project_state.parts = copy.deepcopy(getattr(self.view, "parts", []))
            self.view._sync_all_sketch_meta()
            self.view._ensure_dimensions()
            self.view._recalc_dimension_counter()
    
            # ---- MATERIALS ----
            self.view.deserialize_materials(self.project_state.materials)
            if hasattr(self, "properties_panel") and hasattr(self.properties_panel, "materials_tab"):
                self.properties_panel.materials_tab.set_project_state(self.project_state)
            if hasattr(self, "project_tree"):
                self.project_tree.refresh_from_model()
            if hasattr(self, "property_inspector"):
                self.property_inspector.clear_selection()

            # ---- INTERACTIONS ----
            self.view.deserialize_interfaces(self.project_state.interfaces)
            self.view._emit_interfaces_changed()
    
            # ---- BC & LOADS ----
            self.view.bcs = copy.deepcopy(self.project_state.boundary_conditions)
            self.view.loads = copy.deepcopy(self.project_state.loads)
            if hasattr(self.view, "_sanitize_bc_load_entries"):
                self.view._sanitize_bc_load_entries()
            self.view.initial_velocities = project_data.get("initial_velocities", [])
            self.view.bcsChanged.emit()
            self.view.loadsChanged.emit()
            self._apply_solver_settings_to_ui()
            self._warn_project_state_validation("load")
            self._refresh_workflow_architecture()
    
            # ---- REBUILD VISUAL GEOMETRY ----
            self.view.rebuild_display_geometry()
            self.view.redraw()
            self.properties_panel.assembly_tab.refresh()
    
            # ---- STAGE ----
            try:
                desired_stage = ProjectStage[stage_name]
            except Exception:
                desired_stage = ProjectStage.GEOMETRY

            has_materials = bool(getattr(self.project_state, "materials", {}))
            has_material_assignments = any(
                getattr(part, "material_id", None) not in (None, "")
                for part in getattr(self.project_state, "parts", [])
                if not getattr(part, "is_void", False)
            )
            if has_materials or has_material_assignments:
                desired_stage = max(desired_stage, ProjectStage.MATERIALS, key=self._workflow_stage_rank)
            if self._workflow_has_fluid_stage():
                desired_stage = max(desired_stage, ProjectStage.FLUID, key=self._workflow_stage_rank)
            if getattr(self.project_state, "interfaces", None):
                desired_stage = max(desired_stage, ProjectStage.INTERFACES, key=self._workflow_stage_rank)
            if self.view.loads:
                desired_stage = max(desired_stage, ProjectStage.LOADS, key=self._workflow_stage_rank)
            elif self.view.bcs:
                desired_stage = max(desired_stage, ProjectStage.BCS, key=self._workflow_stage_rank)
            if self._workflow_has_fracture_stage():
                desired_stage = max(desired_stage, ProjectStage.FRACTURE, key=self._workflow_stage_rank)

            project_dir = os.path.dirname(file_path)
            project_name = os.path.splitext(os.path.basename(file_path))[0]
            artifacts_dir = os.path.join(project_dir, f"{project_name}_artifacts")
            inputs_dir = os.path.join(artifacts_dir, "inputs")
            results_dir = os.path.join(artifacts_dir, "results")
            self._restore_workspace_results_from_artifacts(file_path)

            mesh_ready = any(
                os.path.exists(os.path.join(inputs_dir, fname))
                for fname in (
                    "solver_particles.csv",
                    "preview_particles.csv",
                    "solver_nodal.csv",
                    "geometry.csv",
                    "elements.csv",
                )
            ) or any(
                os.path.exists(os.path.join(inputs_dir, "cpd_main_input", fname))
                for fname in ("particles.csv", "connections.csv")
            )
            results_ready = False
            if os.path.isdir(results_dir):
                results_ready = any(
                    fname.endswith(".csv") or fname == "pos_history.npy"
                    for fname in os.listdir(results_dir)
                )
            if not results_ready:
                workspace_results_dir = _workspace_output_path("results")
                workspace_pos_history = _workspace_output_path("pos_history.npy")
                legacy_workspace_results_dir = _workspace_path("results")
                legacy_workspace_pos_history = _workspace_path("pos_history.npy")
                legacy_results_dir = os.path.join(os.path.dirname(__file__), "results")
                legacy_pos_history = os.path.join(
                    os.path.dirname(__file__), "CPD-main", "source", "pos_history.npy"
                )
                if os.path.isdir(workspace_results_dir):
                    results_ready = any(
                        fname.endswith(".csv") for fname in os.listdir(workspace_results_dir)
                    )
                if not results_ready and os.path.exists(workspace_pos_history):
                    results_ready = True
                if not results_ready and os.path.isdir(legacy_workspace_results_dir):
                    results_ready = any(
                        fname.endswith(".csv") for fname in os.listdir(legacy_workspace_results_dir)
                    )
                if not results_ready and os.path.exists(legacy_workspace_pos_history):
                    results_ready = True
                if not results_ready and os.path.isdir(legacy_results_dir):
                    results_ready = any(
                        fname.endswith(".csv") for fname in os.listdir(legacy_results_dir)
                    )
                if not results_ready and os.path.exists(legacy_pos_history):
                    results_ready = True

            if mesh_ready:
                desired_stage = max(desired_stage, ProjectStage.MESH, key=self._workflow_stage_rank)
            if results_ready:
                desired_stage = max(desired_stage, ProjectStage.RESULTS, key=self._workflow_stage_rank)

            self.apply_stage_ui(desired_stage)
            if (
                desired_stage == ProjectStage.MATERIALS
                and self.properties_panel.tabs.isTabEnabled(1)
            ):
                self.properties_panel.tabs.setCurrentWidget(self.properties_panel.materials_tab)
            elif self.view.loads and self.properties_panel.tabs.isTabEnabled(3):
                self.properties_panel.tabs.setCurrentWidget(self.properties_panel.bcs_tab)
            elif self.view.bcs and self.properties_panel.tabs.isTabEnabled(3):
                self.properties_panel.tabs.setCurrentWidget(self.properties_panel.bcs_tab)

            self._load_mesh_from_project_files(file_path, project_data)
            if hasattr(self, "properties_panel") and hasattr(self.properties_panel, "mesh_tab"):
                self.properties_panel.mesh_tab.sync_preview_settings()
            self._sync_point_preview(refresh_data=True)
    
            self.current_project_file = file_path
            self.setWindowTitle(f"CPD SimStudio v25 - {os.path.basename(file_path)}")
            self.project_dirty = False
            self._schedule_startup_window_state(reason="project-restore")
            self._log_window_state(f"Project restore finished [{os.path.basename(file_path)}]")
    
            if show_message:
                QMessageBox.information(self, "Project Loaded", "Project loaded successfully.")
            return True
    
        except Exception as e:
            logging.getLogger(__name__).exception("Project restore failed: %s", file_path)
            QMessageBox.critical(self, "Load Error", str(e))
            return False

    def _load_mesh_from_project_files(self, file_path, project_data):
        project_dir = os.path.dirname(file_path)
        project_name = os.path.splitext(os.path.basename(file_path))[0]
        artifacts_dir = os.path.join(project_dir, f"{project_name}_artifacts")
        inputs_dir = os.path.join(artifacts_dir, "inputs")
        workspace_dir = _workspace_dir()

        # This loader reads 2D particle/triangle CSVs. In 3D projects, restoring these
        # from global workspace files can show stale meshes from another project/session.
        if str(getattr(self, "project_mode", "2d")).lower().startswith("3"):
            return

        candidates = []
        conn_meta = project_data.get("connections") or {}
        particles_file = conn_meta.get("particles") or conn_meta.get("nodes")
        connections_file = conn_meta.get("connections") or conn_meta.get("elements")
        if particles_file and connections_file:
            candidates.append(
                (
                    os.path.join(project_dir, particles_file),
                    os.path.join(project_dir, connections_file),
                )
            )

        candidates.extend(
            [
                (
                    os.path.join(inputs_dir, "cpd_main_input", "particles.csv"),
                    os.path.join(inputs_dir, "cpd_main_input", "connections.csv"),
                ),
                (
                    os.path.join(inputs_dir, "particles.csv"),
                    os.path.join(inputs_dir, "connections.csv"),
                ),
                (
                    os.path.join(inputs_dir, "solver_particles.csv"),
                    os.path.join(inputs_dir, "connections.csv"),
                ),
                (
                    os.path.join(inputs_dir, "solver_nodal.csv"),
                    os.path.join(inputs_dir, "elements.csv"),
                ),
                (
                    os.path.join(inputs_dir, "geometry.csv"),
                    os.path.join(inputs_dir, "elements.csv"),
                ),
                (
                    os.path.join(workspace_dir, "input", "particles.csv"),
                    os.path.join(workspace_dir, "input", "connections.csv"),
                ),
                (
                    os.path.join(workspace_dir, "solver_particles.csv"),
                    os.path.join(workspace_dir, "input", "connections.csv"),
                ),
                (
                    os.path.join(workspace_dir, "solver_nodal.csv"),
                    os.path.join(workspace_dir, "elements.csv"),
                ),
                (
                    os.path.join(workspace_dir, "geometry.csv"),
                    os.path.join(workspace_dir, "elements.csv"),
                ),
                (
                    os.path.join(project_dir, "preview_particles.csv"),
                    os.path.join(project_dir, "connections.csv"),
                ),
                (
                    os.path.join(project_dir, "solver_particles.csv"),
                    os.path.join(project_dir, "connections.csv"),
                ),
                (
                    os.path.join(project_dir, "solver_nodal.csv"),
                    os.path.join(project_dir, "elements.csv"),
                ),
                (
                    os.path.join(project_dir, "geometry.csv"),
                    os.path.join(project_dir, "elements.csv"),
                ),
            ]
        )

        for particles_path, connections_path in candidates:
            if not (os.path.exists(particles_path) and os.path.exists(connections_path)):
                continue
            if self.view.load_mesh_from_files(particles_path, connections_path):
                break

    def _handle_command(self):
        text = self.command_input.text().strip()
        if not text:
            return
        ok, message = self.view.execute_command(text)
        if message:
            self.command_status.setText(message)
        else:
            self.command_status.setText("")
        if ok:
            self.command_input.clear()

    def _update_command_hint(self, text):
        text = text.strip()
        if not text:
            self.command_status.setText("Commands: line, rect, circle, slot, polygon, polyline, confirm, cut, undo, redo, snap.")
            return
        cmd = text.split()[0].lower()
        hint = self._command_hints.get(cmd)
        if not hint:
            for key, value in self._command_hints.items():
                if key.startswith(cmd):
                    hint = value
                    break
        if not hint:
            hint = f"Unknown command '{cmd}'. Type help for the list."
        self.command_status.setText(hint or "")

    def toggle_full_screen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
        QTimer.singleShot(0, self._refresh_full_screen_action)

    def _refresh_full_screen_action(self):
        action = getattr(self, "full_action", None)
        if action is None:
            return
        if self.isFullScreen():
            action.setIcon(get_icon("exit_full_screen"))
            action.setText("Exit Full Screen")
            self._bind_action_tip(action, "Exit full screen mode.")
        else:
            action.setIcon(get_icon("full_screen"))
            action.setText("Full Screen")
            self._bind_action_tip(action, "Enter full screen mode.")

    def changeEvent(self, event):
        if event is not None and event.type() == QEvent.WindowStateChange:
            QTimer.singleShot(0, self._refresh_full_screen_action)
        super().changeEvent(event)

    def _animate_command_bar_visibility(self, visible):
        if not hasattr(self, "command_bar"):
            return
        bar = self.command_bar
        effect = getattr(self, "_command_opacity_effect", None)
        if effect is None:
            effect = QGraphicsOpacityEffect(bar)
            bar.setGraphicsEffect(effect)
            self._command_opacity_effect = effect

        if self._command_bar_anim is not None:
            self._command_bar_anim.stop()
        if self._command_fade_anim is not None:
            self._command_fade_anim.stop()

        start_h = bar.height() if bar.isVisible() else 0
        target_h = max(bar.sizeHint().height(), 1)
        if visible:
            bar.setVisible(True)
            start_h = max(start_h, 1)
            end_h = target_h
            effect.setOpacity(max(effect.opacity(), 0.01))
        else:
            end_h = 0

        h_anim = QPropertyAnimation(bar, b"maximumHeight", self)
        h_anim.setDuration(170)
        h_anim.setStartValue(start_h)
        h_anim.setEndValue(end_h)
        h_anim.setEasingCurve(QEasingCurve.OutCubic if visible else QEasingCurve.InCubic)

        fade_anim = QPropertyAnimation(effect, b"opacity", self)
        fade_anim.setDuration(150)
        fade_anim.setStartValue(effect.opacity() if bar.isVisible() else 0.0)
        fade_anim.setEndValue(1.0 if visible else 0.0)
        fade_anim.setEasingCurve(QEasingCurve.OutCubic if visible else QEasingCurve.InCubic)

        def _finish():
            if visible:
                bar.setMaximumHeight(16777215)
                effect.setOpacity(1.0)
            else:
                bar.setVisible(False)
                bar.setMaximumHeight(16777215)
                effect.setOpacity(1.0)

        h_anim.finished.connect(_finish)
        self._command_bar_anim = h_anim
        self._command_fade_anim = fade_anim
        h_anim.start()
        fade_anim.start()

    def toggle_command_bar(self, visible):
        self._animate_command_bar_visibility(bool(visible))

    def _log_window_state(self, message):
        state = self.windowState()
        logging.getLogger(__name__).info(
            "%s (visible=%s maximized=%s minimized=%s fullscreen=%s state=%s)",
            message,
            self.isVisible(),
            self.isMaximized(),
            self.isMinimized(),
            self.isFullScreen(),
            getattr(state, "value", state),
        )

    def _apply_startup_window_state(self, request_id=None, reason="startup"):
        if request_id is not None and request_id != getattr(self, "_startup_state_request_id", 0):
            return
        self._log_window_state(f"Startup maximize apply begin [{reason}]")
        if not self._start_maximized or self.isFullScreen() or not self.isVisible():
            return
        target_state = self.windowState() | Qt.WindowMaximized
        if self.windowState() != target_state:
            self.setWindowState(target_state)
        self._log_window_state(f"Startup maximize applied [{reason}]")
        # Re-fit the navigator panel after the window reaches its final size.
        # 250 ms gives the maximize animation time to settle before we measure.
        QTimer.singleShot(250, self._sync_nav_panel_width)

    def _schedule_startup_window_state(self, reason="startup", delay_ms=150):
        if not self._start_maximized:
            return
        self._startup_state_request_id = int(getattr(self, "_startup_state_request_id", 0)) + 1
        request_id = self._startup_state_request_id
        self._log_window_state(f"Startup maximize requested [{reason}]")
        QTimer.singleShot(
            max(0, int(delay_ms)),
            lambda rid=request_id, why=str(reason): self._apply_startup_window_state(rid, why),
        )

    def _normalized_analysis_type(self, value=None):
        raw = getattr(self.project_state, "analysis_type", "static") if value is None else value
        normalized = str(raw or "static").strip().lower().replace("_", "-")
        if normalized in {"fsi", "fluid-structure-interaction", "fluid-structure interaction"}:
            return "fsi"
        if normalized.startswith("fluid"):
            return "fluid"
        if normalized.startswith("dyn") or "explicit" in normalized:
            return "dynamic"
        return "static"

    def _normalized_dimension(self, value=None):
        raw = getattr(self.project_state, "dimension", "2D") if value is None else value
        return "3D" if str(raw or "2D").strip().upper().startswith("3") else "2D"

    def _reference_stage_order(self):
        return [
            ProjectStage.GEOMETRY,
            ProjectStage.MATERIALS,
            ProjectStage.FLUID,
            ProjectStage.INTERFACES,
            ProjectStage.BCS,
            ProjectStage.FRACTURE,
            ProjectStage.MESH,
            ProjectStage.JOB,
            ProjectStage.RESULTS,
        ]

    def _workflow_solid_parts(self):
        parts = list(getattr(self.project_state, "parts", []) or getattr(self.view, "parts", []) or [])
        return [
            part for part in parts
            if not bool(getattr(part, "is_void", False))
        ]

    def _workflow_has_multiple_material_regions(self):
        solid_parts = self._workflow_solid_parts()
        direct_material_ids = {
            int(getattr(part, "material_id", -1))
            for part in solid_parts
            if getattr(part, "material_id", None) not in (None, "", -1)
        }
        if len(direct_material_ids) > 1:
            return True
        for part in solid_parts:
            mode = str(getattr(part, "material_assignment_mode", "homogeneous") or "homogeneous").lower()
            if mode == "heterogeneous":
                config = getattr(part, "heterogeneity_config", {}) or {}
                if len(config.get("materials", []) or []) > 1:
                    return True
        return False

    def _workflow_has_interactions_stage(self):
        if self._normalized_analysis_type() == "fsi":
            return True
        if bool(getattr(self.project_state, "interfaces", []) or []):
            return True
        if len(self._workflow_solid_parts()) > 1:
            return True
        return self._workflow_has_multiple_material_regions()

    def _workflow_has_fracture_stage(self):
        materials = list((getattr(self.project_state, "materials", {}) or {}).values())
        for material in materials:
            if str(getattr(material, "damage", "none") or "none").lower() != "none":
                return True
        for part in self._workflow_solid_parts():
            if str(getattr(part, "material_damage", "none") or "none").lower() != "none":
                return True
        return False

    def _workflow_has_fluid_stage(self):
        analysis_type = self._normalized_analysis_type()
        if analysis_type in {"fluid", "fsi"}:
            return True
        materials = list((getattr(self.project_state, "materials", {}) or {}).values())
        for material in materials:
            behavior = str(getattr(material, "behavior", "") or "").strip().lower()
            mat_type = str(getattr(material, "mat_type", "") or "").strip().lower()
            if behavior == "fluid" or mat_type == "fluid":
                return True
        for part in self._workflow_solid_parts():
            if str(getattr(part, "material_behavior", "") or "").strip().lower() == "fluid":
                return True
        return False

    def _workflow_stage_sequence(self):
        stages = [ProjectStage.GEOMETRY, ProjectStage.MATERIALS]
        if self._workflow_has_fluid_stage():
            stages.append(ProjectStage.FLUID)
        if self._workflow_has_interactions_stage():
            stages.append(ProjectStage.INTERFACES)
        stages.append(ProjectStage.BCS)
        if self._workflow_has_fracture_stage():
            stages.append(ProjectStage.FRACTURE)
        stages.extend([ProjectStage.MESH, ProjectStage.JOB, ProjectStage.RESULTS])
        return stages

    def _workflow_stage_group(self, stage):
        if stage == ProjectStage.LOADS:
            return ProjectStage.BCS
        return stage

    def _coerce_stage_to_workflow(self, stage):
        normalized = self._workflow_stage_group(stage)
        sequence = self._workflow_stage_sequence()
        if normalized in sequence:
            return normalized
        if not sequence:
            return ProjectStage.GEOMETRY
        ref_order = self._reference_stage_order()
        try:
            target_rank = ref_order.index(normalized)
        except ValueError:
            target_rank = 0
        fallback = sequence[0]
        for candidate in sequence:
            try:
                candidate_rank = ref_order.index(candidate)
            except ValueError:
                continue
            if candidate_rank <= target_rank:
                fallback = candidate
            elif candidate_rank > target_rank:
                break
        return fallback

    def _selection_payload_stage(self, payload):
        if not isinstance(payload, dict):
            return None
        stage = payload.get("stage")
        if stage == ProjectStage.LOADS:
            stage = ProjectStage.BCS
        return stage if isinstance(stage, ProjectStage) else None

    def _selection_payload_is_visible(self, payload, sequence=None):
        stage = self._selection_payload_stage(payload)
        visible_stages = set(sequence or self._workflow_stage_sequence())
        visible_stages.add(ProjectStage.GEOMETRY)
        return stage is None or stage in visible_stages

    def _clear_inactive_stage_selection(self, sequence=None):
        inspector = getattr(self, "property_inspector", None)
        payload = getattr(inspector, "_selection_payload", None) if inspector is not None else None
        if self._selection_payload_is_visible(payload, sequence):
            return False
        if inspector is not None:
            inspector.clear_selection()
        tree = getattr(self, "project_tree", None)
        if tree is not None:
            try:
                tree.clearSelection()
            except Exception:
                pass
        return True

    def _stage_to_tab_index(self, stage):
        tab_map = {
            ProjectStage.GEOMETRY: 0,
            ProjectStage.MATERIALS: 1,
            ProjectStage.FLUID: 2,
            ProjectStage.INTERFACES: 3,
            ProjectStage.BCS: 4,
            ProjectStage.LOADS: 4,
            ProjectStage.FRACTURE: 5,
            ProjectStage.MESH: 6,
            ProjectStage.JOB: 7,
            ProjectStage.RESULTS: 8,
        }
        return tab_map.get(stage, 0)

    def _tab_index_to_stage(self, index):
        stage_map = {
            0: ProjectStage.GEOMETRY,
            1: ProjectStage.MATERIALS,
            2: ProjectStage.FLUID,
            3: ProjectStage.INTERFACES,
            4: ProjectStage.BCS,
            5: ProjectStage.FRACTURE,
            6: ProjectStage.MESH,
            7: ProjectStage.JOB,
            8: ProjectStage.RESULTS,
        }
        return stage_map.get(int(index), getattr(self, "active_stage", ProjectStage.GEOMETRY))

    def _current_tab_stage(self):
        if not hasattr(self, "properties_panel") or not hasattr(self.properties_panel, "tabs"):
            return getattr(self, "active_stage", ProjectStage.GEOMETRY)
        return self._tab_index_to_stage(self.properties_panel.tabs.currentIndex())

    def _on_sketch_edit_mode_changed(self, active, part=None):
        active = bool(active)
        action = getattr(self, "confirm_action", None)
        if action is not None:
            if active:
                action.setText("Finish Sketch")
                self._bind_action_tip(action, "Finish sketch editing and update the selected part.")
            else:
                action.setText("Confirm Part")
                self._bind_action_tip(action, "Convert closed sketches into a new part.")
        if active:
            self._set_precision_sketch_mode(True, announce=False)
        else:
            self._set_precision_sketch_mode(False, announce=False)
        self._update_stage_nav_buttons()

    def enter_sketch_edit_mode(self, part_id):
        self.apply_stage_ui(ProjectStage.GEOMETRY)
        target_part = next((p for p in self.project_state.parts if int(getattr(p, "id", -1)) == int(part_id)), None)
        if target_part is None:
            return False
        ok = self.view.begin_part_shape_edit(target_part)
        if ok:
            self._on_sketch_edit_mode_changed(True, target_part)
        return ok

    def _update_stage_nav_buttons(self):
        panel = getattr(self, "properties_panel", None)
        if panel is None:
            return
        current_stage = self._coerce_stage_to_workflow(self._current_tab_stage())
        prev_stage = self._adjacent_workflow_stage(current_stage, direction=-1)
        next_stage = self._adjacent_workflow_stage(current_stage, direction=1)
        if hasattr(panel, "set_navigation_state"):
            panel.set_navigation_state(prev_stage, next_stage)
            return
        if hasattr(panel, "_update_stage_nav_buttons"):
            panel._update_stage_nav_buttons()

    def _workflow_stage_rank(self, stage):
        normalized = self._coerce_stage_to_workflow(stage)
        sequence = self._workflow_stage_sequence()
        try:
            return sequence.index(normalized)
        except ValueError:
            return -1

    def _adjacent_workflow_stage(self, current_stage, *, direction):
        stages = self._workflow_stage_sequence()
        current_rank = self._workflow_stage_rank(self._coerce_stage_to_workflow(current_stage))
        if current_rank < 0:
            return None
        step = 1 if int(direction) >= 0 else -1
        index = current_rank + step
        return stages[index] if 0 <= index < len(stages) else None

    def advance_to_next_stage(self):
        current = self._coerce_stage_to_workflow(self._current_tab_stage())
        next_stage = self._adjacent_workflow_stage(current, direction=1)
        if next_stage is None:
            self.statusBar().showMessage("Already at final stage.", self._status_timeout_ms)
            return
        if self._workflow_stage_rank(next_stage) <= self._workflow_stage_rank(
            getattr(self, "active_stage", ProjectStage.GEOMETRY)
        ):
            target_idx = self._stage_to_tab_index(next_stage)
            if self.properties_panel.tabs.currentIndex() != target_idx:
                self.properties_panel.tabs.setCurrentIndex(target_idx)
            return
        self.advance_stage(next_stage)

    def retreat_to_prev_stage(self):
        current = self._coerce_stage_to_workflow(self._current_tab_stage())
        prev_stage = self._adjacent_workflow_stage(current, direction=-1)
        if prev_stage is None:
            self.statusBar().showMessage("Already at first stage.", self._status_timeout_ms)
            return
        target_idx = self._stage_to_tab_index(prev_stage)
        if self.properties_panel.tabs.currentIndex() != target_idx:
            self.properties_panel.tabs.setCurrentIndex(target_idx)

    def _sync_project_dimension_state(self):
        dimension = "3D" if bool(getattr(self, "_workspace_3d", False)) else "2D"
        if getattr(self, "project_state", None) is not None:
            self.project_state.dimension = dimension
        return dimension

    def set_project_analysis_type(self, analysis_type, announce=True):
        normalized = self._normalized_analysis_type(analysis_type)
        if getattr(self, "project_state", None) is None:
            return
        changed = normalized != self._normalized_analysis_type()
        self.project_state.analysis_type = normalized
        if hasattr(self, "properties_panel") and hasattr(self.properties_panel, "assembly_tab"):
            try:
                self.properties_panel.assembly_tab.refresh_project_setup()
            except Exception:
                pass
        self._refresh_workflow_architecture()
        if changed and announce:
            label_map = {
                "static": "Static Mechanical",
                "dynamic": "Dynamic Explicit",
                "fluid": "Fluid",
                "fsi": "Fluid-Structure Interaction",
            }
            label = label_map.get(normalized, "Static Mechanical")
            self.statusBar().showMessage(f"Analysis type set to {label}.", self._status_timeout_ms)

    def set_project_dimension(self, dimension, announce=True):
        normalized = self._normalized_dimension(dimension)
        target_workspace_3d = normalized == "3D"
        if bool(getattr(self, "_workspace_3d", False)) != target_workspace_3d:
            self._toggle_workspace_3d(target_workspace_3d)
        else:
            self._sync_project_dimension_state()
            self._refresh_workflow_architecture()
        if hasattr(self, "properties_panel") and hasattr(self.properties_panel, "assembly_tab"):
            try:
                self.properties_panel.assembly_tab.refresh_project_setup()
            except Exception:
                pass
        if announce:
            self.statusBar().showMessage(f"Project dimension set to {normalized}.", self._status_timeout_ms)

    def _refresh_workflow_architecture(self):
        sequence = self._workflow_stage_sequence()
        if hasattr(self, "stage_bar"):
            try:
                self.stage_bar.set_stage_order(sequence)
            except Exception:
                pass
        if hasattr(self, "project_tree"):
            try:
                self.project_tree.set_visible_stages(sequence)
                self.project_tree.set_active_stage(self._coerce_stage_to_workflow(getattr(self, "active_stage", ProjectStage.GEOMETRY)))
            except Exception:
                pass
        current_stage = self._coerce_stage_to_workflow(getattr(self, "active_stage", ProjectStage.GEOMETRY))
        if current_stage not in sequence:
            current_stage = sequence[0] if sequence else ProjectStage.GEOMETRY
        self._clear_inactive_stage_selection(sequence)
        if getattr(self, "_workflow_stage_updating", False):
            return
        if getattr(self, "active_stage", None) != current_stage:
            self.apply_stage_ui(current_stage)
            return
        if getattr(self, "project_state", None) is not None:
            self.project_state.current_stage = self._workflow_key_for_stage(current_stage)
        self._sync_stage_tab_enabled_state(current_stage)
        if hasattr(self, "stage_bar"):
            self.stage_bar.set_active_stage(current_stage)
        if hasattr(self, "project_tree"):
            self.project_tree.set_active_stage(current_stage)
        if not getattr(self, "_workflow_stage_updating", False):
            self._update_workflow_ribbon_state(current_stage)
        self._update_stage_nav_buttons()

    def _sync_stage_tab_enabled_state(self, stage):
        if not hasattr(self, "properties_panel") or not hasattr(self.properties_panel, "tabs"):
            return
        workflow_sequence = self._workflow_stage_sequence()
        stage_rank = self._workflow_stage_rank(stage)
        enabled = {
            0: ProjectStage.GEOMETRY in workflow_sequence and stage_rank >= self._workflow_stage_rank(ProjectStage.GEOMETRY),
            1: ProjectStage.MATERIALS in workflow_sequence and stage_rank >= self._workflow_stage_rank(ProjectStage.MATERIALS),
            2: ProjectStage.FLUID in workflow_sequence and stage_rank >= self._workflow_stage_rank(ProjectStage.FLUID),
            3: ProjectStage.INTERFACES in workflow_sequence and stage_rank >= self._workflow_stage_rank(ProjectStage.INTERFACES),
            4: ProjectStage.BCS in workflow_sequence and stage_rank >= self._workflow_stage_rank(ProjectStage.BCS),
            5: ProjectStage.FRACTURE in workflow_sequence and stage_rank >= self._workflow_stage_rank(ProjectStage.FRACTURE),
            6: ProjectStage.MESH in workflow_sequence and stage_rank >= self._workflow_stage_rank(ProjectStage.MESH),
            7: ProjectStage.JOB in workflow_sequence and stage_rank >= self._workflow_stage_rank(ProjectStage.JOB),
            8: ProjectStage.RESULTS in workflow_sequence and stage_rank >= self._workflow_stage_rank(ProjectStage.RESULTS),
        }
        for index, state in enabled.items():
            self.properties_panel.tabs.setTabEnabled(index, bool(state))

    def apply_stage_ui(self, stage: ProjectStage):
        """
        Central UI stage controller (Abaqus-style)
        """
        stage = self._coerce_stage_to_workflow(stage)
        stage_key = self._workflow_key_for_stage(stage)
        self.current_stage = stage_key
        if getattr(self, "project_state", None) is not None:
            self.project_state.current_stage = stage_key
        self.active_stage = stage
        self._active_project_stage = stage
        self.view.active_stage = stage
        workflow_sequence = self._workflow_stage_sequence()

        self.view.disable_all_tools()
        self._sync_stage_tab_enabled_state(stage)

        stage_rank = self._workflow_stage_rank(stage)
        if ProjectStage.GEOMETRY in workflow_sequence and stage_rank >= self._workflow_stage_rank(ProjectStage.GEOMETRY):
            self.properties_panel.tabs.setTabEnabled(0, True)
            self.view.enable_geometry_tools()

        if ProjectStage.MATERIALS in workflow_sequence and stage_rank >= self._workflow_stage_rank(ProjectStage.MATERIALS):
            self.properties_panel.tabs.setTabEnabled(1, True)
            self.view.enable_material_tools()

        if ProjectStage.FLUID in workflow_sequence and stage_rank >= self._workflow_stage_rank(ProjectStage.FLUID):
            self.properties_panel.tabs.setTabEnabled(2, True)

        if ProjectStage.INTERFACES in workflow_sequence and stage_rank >= self._workflow_stage_rank(ProjectStage.INTERFACES):
            self.properties_panel.tabs.setTabEnabled(3, True)
            self.view.enable_interaction_tools()

        if ProjectStage.BCS in workflow_sequence and stage_rank >= self._workflow_stage_rank(ProjectStage.BCS):
            self.properties_panel.tabs.setTabEnabled(4, True)

        if ProjectStage.FRACTURE in workflow_sequence and stage_rank >= self._workflow_stage_rank(ProjectStage.FRACTURE):
            self.properties_panel.tabs.setTabEnabled(5, True)

        if ProjectStage.MESH in workflow_sequence and stage_rank >= self._workflow_stage_rank(ProjectStage.MESH):
            self.properties_panel.tabs.setTabEnabled(6, True)
            self.view.enable_mesh_tools()

        if stage in {ProjectStage.BCS, ProjectStage.LOADS}:
            self.view.enable_bc_tools()
            self.view.set_module("Boundary" if stage == ProjectStage.BCS else "Load")

        if ProjectStage.JOB in workflow_sequence and stage_rank >= self._workflow_stage_rank(ProjectStage.JOB):
            self.properties_panel.tabs.setTabEnabled(7, True)
            self.view.enable_job_tools()
        if ProjectStage.RESULTS in workflow_sequence and stage_rank >= self._workflow_stage_rank(ProjectStage.RESULTS):
            self.properties_panel.tabs.setTabEnabled(8, True)
        
        if hasattr(self, "stage_bar"):
            self.stage_bar.set_active_stage(stage)
        self.properties_panel.set_stage(stage)
        target_idx = self._stage_to_tab_index(stage)
        self.properties_panel.show_stage(stage)
        if hasattr(self, "mesh_view_action"):
            if stage == ProjectStage.MESH:
                self.mesh_view_action.setChecked(True)
            elif stage != ProjectStage.MESH and self.mesh_view_action.isChecked():
                self.mesh_view_action.setChecked(False)
        if stage in {ProjectStage.BCS, ProjectStage.LOADS}:
            self.view.set_display_mode("bc")
        elif stage == ProjectStage.MESH:
            self.view.set_display_mode("mesh")
        elif stage == ProjectStage.RESULTS:
            self.view.set_display_mode("results")
        else:
            self.view.set_display_mode("geometry")
        hint = self._stage_hint_text(stage)
        if hint and bool(getattr(self, "_stage_hints_enabled", True)):
            self.statusBar().showMessage(hint, self._status_timeout_ms)
        if not getattr(self, "_workflow_stage_updating", False):
            self._update_workflow_ribbon_state(stage)
        self._update_stage_nav_buttons()
        self._update_interaction_hints()
        self._adjust_right_splitter_for_stage(stage)

    def _adjust_right_splitter_for_stage(self, stage):
        splitter = getattr(self, "_right_splitter", None)
        if splitter is None:
            return
        sizes = splitter.sizes()
        total = int(sum(sizes)) if sizes else int(splitter.height())
        if total <= 0:
            return
        inspector = getattr(self, "property_inspector", None)
        has_selection = bool(getattr(inspector, "_selection_payload", None)) if inspector is not None else False
        if not has_selection:
            if stage == ProjectStage.RESULTS:
                top_ratio = 0.86
            elif stage == ProjectStage.GEOMETRY:
                top_ratio = 0.90
            else:
                top_ratio = 0.88
        elif stage == ProjectStage.RESULTS:
            top_ratio = 0.74
        elif stage == ProjectStage.GEOMETRY:
            top_ratio = 0.74
        else:
            top_ratio = 0.66
        top_size = max(280, int(total * top_ratio))
        bottom_size = max(72 if not has_selection else 160, total - top_size)
        splitter.setSizes([top_size, bottom_size])

    def _stage_hint_text(self, stage):
        controller = getattr(self, "_stage_controller_by_stage", {}).get(stage)
        if controller is not None and hasattr(controller, "hint_for"):
            return str(controller.hint_for(stage) or "")
        if stage == ProjectStage.FLUID:
            return "Review fluid regions and fluid-participating materials for the current model."
        if stage == ProjectStage.FRACTURE:
            return "Review damage-enabled materials and fracture-related part overrides before solving."
        return ""

    def _set_stage_hints_enabled(self, enabled, announce=True):
        enabled = bool(enabled)
        if getattr(self, "_stage_hints_enabled", True) == enabled and not announce:
            return
        self._stage_hints_enabled = enabled
        if hasattr(self, "_settings"):
            self._settings.setValue("ui/stage_hints_enabled", enabled)
        action = getattr(self, "stage_hints_action", None)
        if action is not None and action.isChecked() != enabled:
            action.blockSignals(True)
            action.setChecked(enabled)
            action.blockSignals(False)
        btn = getattr(self, "stage_hints_toggle_btn", None)
        if btn is not None and btn.isChecked() != enabled:
            btn.blockSignals(True)
            btn.setChecked(enabled)
            btn.blockSignals(False)
        if announce and hasattr(self, "statusBar"):
            self.statusBar().showMessage(
                "Stage hints enabled." if enabled else "Stage hints disabled.",
                self._status_timeout_ms,
            )

    def _reset_stage_panels(self, clear_bc_lists=False):
        panel = getattr(self, "properties_panel", None)
        if panel is None:
            return
        bcs_tab = getattr(panel, "bcs_tab", None)
        if bcs_tab is not None and hasattr(bcs_tab, "reset_stage_state"):
            try:
                bcs_tab.reset_stage_state(clear_lists=bool(clear_bc_lists))
            except Exception:
                pass
        loads_tab = getattr(panel, "loads_tab", None)
        if loads_tab is not None and hasattr(loads_tab, "reset_stage_state"):
            try:
                loads_tab.reset_stage_state(clear_lists=bool(clear_bc_lists))
            except Exception:
                pass
        results_tab = getattr(panel, "results_tab", None)
        if results_tab is not None and hasattr(results_tab, "reset_stage_state"):
            try:
                results_tab.reset_stage_state()
            except Exception:
                pass

    # =========================
    # TOOLBARS (UNCHANGED)
    # =========================
    def _create_sketch_toolbar(self):
        sketch_toolbar = QToolBar("Sketch")
        sketch_toolbar.setObjectName("SketchToolbar")
        sketch_toolbar.setMovable(False)
        sketch_toolbar.setFloatable(False)
        self.addToolBar(Qt.LeftToolBarArea, sketch_toolbar)
        toolbar_icon_px = toolbar_icon_size()
        sketch_toolbar.setIconSize(QSize(toolbar_icon_px, toolbar_icon_px))
        sketch_toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        if self._toolbar_style:
            sketch_toolbar.setStyleSheet(self._toolbar_style)
        sketch_toolbar.setContentsMargins(
            UI_TOKENS.spacing_xs,
            UI_TOKENS.spacing_xs,
            UI_TOKENS.spacing_xs,
            UI_TOKENS.spacing_xs,
        )

        action_group = QActionGroup(self)
        action_group.setExclusive(True)
        sketch_mode_tools = {
            "select",
            "arc_segment",
            "move",
            "copy",
            "mirror",
            "trim",
            "dimension",
            "constraint",
        }

        def build_tool_action(name, tool_name, icon_name=None, tip=None, default=False):
            action = QAction(name, self)
            action.setCheckable(True)
            if icon_name:
                action.setIcon(get_icon(icon_name))
            if tip:
                self._bind_action_tip(action, tip)
            if tool_name in sketch_mode_tools:
                action.triggered.connect(lambda checked=False, name=tool_name: self._set_sketch_geometry_tool(name))
            else:
                action.triggered.connect(lambda checked=False, name=tool_name: self.view.set_tool(name))
            if default:
                action.setChecked(True)
            action_group.addAction(action)
            return action

        select_action = build_tool_action("Select", "select", "select", "Select parts or geometry.")
        arc_segment_action = build_tool_action(
            "Arc Segment",
            "arc_segment",
            "arc_select",
            "Select a curve segment: click start, end, then mid on the boundary.",
        )
        sketch_toolbar.addAction(select_action)
        sketch_toolbar.addAction(arc_segment_action)

        def add_category(label, actions, icon_name=None):
            button = QToolButton(self)
            menu = QMenu(button)
            for act in actions:
                menu.addAction(act)
            button.setMenu(menu)
            button.setText("")
            button.setToolTip(label)
            button.setAccessibleName(label)
            if icon_name:
                button.setIcon(get_icon(icon_name))
            button.setIconSize(QSize(toolbar_icon_px, toolbar_icon_px))
            button.setToolButtonStyle(Qt.ToolButtonIconOnly)
            button.setPopupMode(QToolButton.InstantPopup)
            sketch_toolbar.addWidget(button)

        edit_actions = [
            build_tool_action("Move", "move", "gizmo_move", "Move: click base point, then target point."),
            build_tool_action("Copy", "copy", "copy", "Copy: click base point, then target point. Esc to finish."),
            build_tool_action("Mirror", "mirror", "mirror", "Mirror: click two points for mirror line."),
            build_tool_action("Trim", "trim", "trim", "Trim: click a sketch segment to remove."),
        ]
        dimension_actions = [
            build_tool_action(
                "Smart Dim",
                "dimension",
                "dimension",
                "Smart dimensions: length, radius, diameter, arc length, and angle.",
            ),
        ]

        sketch_toolbar.addSeparator()
        add_category("Edit", edit_actions, icon_name="gizmo_move")
        add_category("Dimension", dimension_actions, icon_name="dimension")

        sketch_toolbar.addSeparator()
        confirm_action = QAction(get_icon("confirm"), "Confirm Part", self)
        confirm_action.triggered.connect(self.view.confirm_solid)
        self._bind_action_tip(confirm_action, "Convert closed sketches into a new part.")
        sketch_toolbar.addAction(confirm_action)
        self.confirm_action = confirm_action
        cut_action = QAction(get_icon("cut"), "Cut Hole", self)
        cut_action.triggered.connect(self.view.cut_hole)
        self._bind_action_tip(cut_action, "Cut a hole using the current sketches.")
        sketch_toolbar.addAction(cut_action)

        pattern_action = QAction(get_icon("pattern"), "Linear Pattern", self)
        pattern_action.triggered.connect(self._linear_pattern_selected_part)
        self._bind_action_tip(pattern_action, "Create a linear pattern from the selected part.")
        sketch_toolbar.addAction(pattern_action)
        join_action = QAction(get_icon("join"), "Join", self)
        join_action.triggered.connect(self._join_sketches)
        self._bind_action_tip(join_action, "Join sketch endpoints that are close together.")
        sketch_toolbar.addAction(join_action)
        porous_action = QAction(get_icon("porous"), "Porous/Particle", self)
        porous_action.triggered.connect(self._open_porous_dialog)
        self._bind_action_tip(
            porous_action,
            "Generate porous/particle features: holes or particles with preview.",
        )
        sketch_toolbar.addAction(porous_action)

        sketch_toolbar.addSeparator()
        undo_action = QAction(get_icon("undo"), "Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self._handle_undo_action)
        self._bind_action_tip(undo_action, "Undo last change.")
        sketch_toolbar.addAction(undo_action)
        redo_action = QAction(get_icon("redo"), "Redo", self)
        redo_action.setShortcuts(["Ctrl+Y", "Ctrl+Shift+Z"])
        redo_action.triggered.connect(self._handle_redo_action)
        self._bind_action_tip(redo_action, "Redo last undone change.")
        sketch_toolbar.addAction(redo_action)

        sketch_toolbar.addSeparator()
        param_action = QAction(get_icon("parametric"), "Parametric", self)
        param_action.setCheckable(True)
        param_action.toggled.connect(self.view.set_parametric_mode)
        self._bind_action_tip(param_action, "Use numeric inputs for drawing.")
        sketch_toolbar.addAction(param_action)
        auto_convert_action = QAction(get_icon("auto_convert"), "Auto Convert Scribble", self)
        auto_convert_action.setCheckable(True)
        auto_convert_action.setChecked(
            bool(getattr(self.view, "freeform_auto_convert_enabled", False))
        )
        auto_convert_action.toggled.connect(
            lambda enabled: self.view.set_freeform_auto_convert(enabled)
        )
        self._bind_action_tip(
            auto_convert_action,
            "When enabled, freeform scribbles auto-fit to line/circle/rectangle/slot/polygon.",
        )
        sketch_toolbar.addAction(auto_convert_action)
        self.freeform_autoconvert_action = auto_convert_action

        # Units cycling button (mm -> cm -> m -> mm)
        sketch_toolbar.addSeparator()
        self._unit_cycle = ["mm", "cm", "m"]
        self._unit_index = 0
        units_button = QToolButton(self)
        units_button.setText(self._unit_cycle[self._unit_index])
        units_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        units_button.setAutoRaise(False)
        units_button.setMinimumSize(34, 24)
        units_button.setMaximumWidth(44)
        units_button.setToolTip("Click to cycle drawing unit: mm → cm → m")
        units_button.setStyleSheet(
            "QToolButton { font-weight: 600; padding: 2px 6px; }"
        )

        def _cycle_unit():
            self._unit_index = (self._unit_index + 1) % len(self._unit_cycle)
            new_unit = self._unit_cycle[self._unit_index]
            units_button.setText(new_unit)
            self._on_unit_changed(new_unit)

        units_button.clicked.connect(_cycle_unit)
        sketch_toolbar.addWidget(units_button)
        self.units_button = units_button

        self.sketch_toolbar = sketch_toolbar

    def _workflow_ribbon_definitions(self):
        return {
            self.geometry_controller.workflow_key: self.geometry_controller.workflow_definition(
                getattr(self.properties_panel, "assembly_tab", None)
            ),
            self.material_controller.workflow_key: self.material_controller.workflow_definition(
                getattr(self.properties_panel, "materials_tab", None)
            ),
            "fluid": {
                "label": "Fluid",
                "stage": ProjectStage.FLUID,
                "tab": getattr(self.properties_panel, "fluid_tab", None),
                "show_sketch": False,
                "show_geometry": False,
                "show_loads": False,
            },
            self.interaction_controller.workflow_key: self.interaction_controller.workflow_definition(
                getattr(self.properties_panel, "interfaces_tab", None)
            ),
            "bc": self.bc_controller.workflow_definition(
                getattr(self.properties_panel, "bcs_tab", None),
                workflow_key="bc",
                label="Boundary Conditions",
                stage=ProjectStage.BCS,
            ),
            "fracture": {
                "label": "Fracture",
                "stage": ProjectStage.FRACTURE,
                "tab": getattr(self.properties_panel, "fracture_tab", None),
                "show_sketch": False,
                "show_geometry": False,
                "show_loads": False,
            },
            self.particle_controller.workflow_key: self.particle_controller.workflow_definition(
                getattr(self.properties_panel, "mesh_tab", None)
            ),
            self.solver_controller.workflow_key: self.solver_controller.workflow_definition(
                getattr(self.properties_panel, "job_tab", None)
            ),
            self.results_controller.workflow_key: self.results_controller.workflow_definition(
                getattr(self.properties_panel, "results_tab", None)
            ),
        }

    def _workflow_key_for_stage(self, stage):
        if stage == ProjectStage.GEOMETRY:
            return "geometry"
        if stage == ProjectStage.MATERIALS:
            return "materials"
        if stage == ProjectStage.FLUID:
            return "fluid"
        if stage == ProjectStage.INTERFACES:
            return "interactions"
        if stage == ProjectStage.BCS:
            return "bc"
        if stage == ProjectStage.LOADS:
            return "bc"
        if stage == ProjectStage.FRACTURE:
            return "fracture"
        if stage == ProjectStage.MESH:
            return "particles"
        if stage == ProjectStage.JOB:
            return "solve"
        if stage == ProjectStage.RESULTS:
            return "results"
        return "geometry"

    def _create_workflow_ribbon(self):
        # Build the StageBar widget only — no separate QToolBar.
        # The widget is inserted into MainToolbar by _create_main_toolbar()
        # so that stage labels and icon buttons share a single full-width bar.
        # This eliminates the two-toolbar width-allocation conflict that caused
        # stage labels to be clipped or pushed off-screen.
        self._workflow_stage_key = "geometry"
        self._workflow_stage_defs = self._workflow_ribbon_definitions()
        self.stage_bar = StageBar(self, icon_only=False)
        self.stage_bar.set_icons(
            {
                ProjectStage.GEOMETRY: get_stage_icon("geometry", size=20, state="current"),
                ProjectStage.MATERIALS: get_stage_icon("materials", size=20, state="future"),
                ProjectStage.FLUID: get_stage_icon("stage_fluid", size=20, state="future"),
                ProjectStage.INTERFACES: get_stage_icon("interactions", size=20, state="future"),
                ProjectStage.BCS: get_stage_icon("bc", size=20, state="future"),
                ProjectStage.FRACTURE: get_stage_icon("stage_fracture", size=20, state="future"),
                ProjectStage.MESH: get_stage_icon("particles", size=20, state="future"),
                ProjectStage.JOB: get_stage_icon("solve", size=20, state="future"),
                ProjectStage.RESULTS: get_stage_icon("results", size=20, state="future"),
            }
        )
        self.stage_bar.stageRequested.connect(
            lambda stage: self._activate_workflow_stage(self._workflow_key_for_stage(stage))
        )
        self.workflow_stage_buttons = {}
        self.workflow_ribbon = None  # toolbar created in _create_main_toolbar

    def _set_stage_toolbar_visibility(self, key):
        cfg = self._workflow_stage_defs.get(str(key), {})
        if hasattr(self, "sketch_toolbar"):
            self.sketch_toolbar.setVisible(bool(cfg.get("show_sketch", False)))
        if hasattr(self, "loads_toolbar"):
            self.loads_toolbar.setVisible(bool(cfg.get("show_loads", False)))
        if hasattr(self, "primitive_dock"):
            self.primitive_dock.setVisible(
                bool(cfg.get("show_sketch", False)) and bool(getattr(self, "_workspace_3d", False))
            )

    def _update_workflow_ribbon_state(self, stage, preferred_key=None):
        key = str(preferred_key or self._workflow_key_for_stage(stage))
        if key not in self._workflow_stage_defs:
            key = "geometry"
        self._workflow_stage_key = key
        cfg = self._workflow_stage_defs.get(key, {})
        label = cfg.get("label", str(key).title())
        if hasattr(self, "stage_bar"):
            self.stage_bar.set_active_stage(stage)
        if hasattr(self, "workflow_stage_indicator"):
            self.workflow_stage_indicator.hide()
        self._set_stage_toolbar_visibility(key)

    def _activate_workflow_stage(self, key):
        if str(key) == "loads":
            key = "bc"
        cfg = self._workflow_stage_defs.get(str(key))
        if not cfg:
            return
        self._workflow_stage_key = str(key)
        self._workflow_stage_updating = True
        try:
            self.apply_stage_ui(cfg["stage"])
            tab = cfg.get("tab")
            if tab is not None and hasattr(self.properties_panel, "tabs"):
                self.properties_panel.tabs.setCurrentWidget(tab)
        finally:
            self._workflow_stage_updating = False
        self._update_workflow_ribbon_state(cfg["stage"], preferred_key=key)

    def _sync_workflow_ribbon_from_tab(self, index):
        _ = index
        if getattr(self, "_workflow_stage_updating", False):
            return
        idx = -1
        if hasattr(self.properties_panel, "tabs"):
            idx = self.properties_panel.tabs.currentIndex()
        key_map = {
            0: "geometry",
            1: "materials",
            2: "fluid",
            3: "interactions",
            4: "bc",
            5: "fracture",
            6: "particles",
            7: "solve",
            8: "results",
        }
        key = key_map.get(idx)
        stage = (
            self._coerce_stage_to_workflow(self._tab_index_to_stage(idx))
            if idx >= 0
            else getattr(self, "_active_project_stage", ProjectStage.GEOMETRY)
        )
        self.active_stage = stage
        self._active_project_stage = stage
        self.view.active_stage = stage
        if hasattr(self, "stage_bar"):
            self.stage_bar.set_active_stage(stage)
        if hasattr(self, "project_tree"):
            self.project_tree.set_active_stage(stage)
        if hasattr(self, "properties_panel"):
            self.properties_panel.set_stage(stage)
        self._update_workflow_ribbon_state(stage, preferred_key=key)
        self._update_interaction_hints()

    def _set_sketch_geometry_tool(self, tool_name):
        view = getattr(self, "view", None)
        if view is None or not hasattr(view, "set_tool"):
            return
        try:
            if getattr(self, "active_stage", None) != ProjectStage.GEOMETRY:
                self.apply_stage_ui(ProjectStage.GEOMETRY)
        except Exception:
            pass
        if hasattr(view, "set_module"):
            try:
                view.set_module("Part")
            except Exception:
                pass
        if str(tool_name).lower() == "freeform":
            self._set_precision_sketch_mode(False, announce=False)
        view.set_tool(str(tool_name))
        self._update_interaction_hints()

    def _activate_geometry_extrude(self):
        if getattr(self, "_workspace_3d", False) and hasattr(self, "_add_primitive"):
            self._add_primitive("extrude")
            return
        self.statusBar().showMessage(
            "Extrude is available in the 3D geometry workspace.",
            self._status_timeout_ms,
        )

    def _import_geometry_from_toolbar(self):
        for target in (
            getattr(self, "import_geometry", None),
            getattr(self, "import_dxf", None),
            getattr(getattr(self, "view", None), "import_geometry", None),
            getattr(getattr(self, "view", None), "import_dxf", None),
        ):
            if callable(target):
                target()
                return
        self.statusBar().showMessage("Import Geometry is not available in the current workspace.", self._status_timeout_ms)

    def _set_precision_sketch_mode(self, enabled, announce=True):
        enabled = bool(enabled)
        if hasattr(self, "view") and hasattr(self.view, "set_precision_sketch_mode"):
            self.view.set_precision_sketch_mode(enabled)
            enabled = bool(getattr(self.view, "precision_sketch_mode_enabled", enabled))
        action = getattr(self, "precision_sketch_action", None)
        if action is not None and action.isChecked() != enabled:
            try:
                action.blockSignals(True)
                action.setChecked(enabled)
            finally:
                try:
                    action.blockSignals(False)
                except Exception:
                    pass
        dim_action = getattr(self, "dim_action", None)
        if dim_action is not None and dim_action.isChecked() != enabled:
            try:
                dim_action.blockSignals(True)
                dim_action.setChecked(enabled)
            finally:
                try:
                    dim_action.blockSignals(False)
                except Exception:
                    pass
        if announce:
            label = "Precision sketch mode enabled." if enabled else "Precision sketch mode disabled."
            self.statusBar().showMessage(label, self._status_timeout_ms)
        self._update_interaction_hints()

    def _create_main_toolbar(self):
        main_toolbar = QToolBar("Main")
        main_toolbar.setObjectName("MainToolbar")
        main_toolbar.setMovable(False)
        main_toolbar.setFloatable(False)
        self.addToolBar(Qt.TopToolBarArea, main_toolbar)
        toolbar_icon_px = toolbar_icon_size()
        main_toolbar.setIconSize(QSize(toolbar_icon_px, toolbar_icon_px))
        main_toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        if self._toolbar_style:
            main_toolbar.setStyleSheet(self._toolbar_style)
        # Slim top/bottom margins (2 px) so the 30 px workflow-stage buttons get
        # their full height inside the toolbar and never have their text
        # descenders clipped.  Left/right keep the normal 4 px breathing room.
        main_toolbar.setContentsMargins(
            UI_TOKENS.spacing_xs,
            2,
            UI_TOKENS.spacing_xs,
            2,
        )

        # Stage bar lives in this toolbar so it shares the full window width
        # with the icon buttons — no separate toolbar means no width-allocation
        # conflict and no stage labels ever being clipped or hidden.
        if hasattr(self, "stage_bar"):
            main_toolbar.addWidget(self.stage_bar)
            main_toolbar.addSeparator()
        self.workflow_ribbon = main_toolbar

        if hasattr(self, "workspace_2d_action") and hasattr(self, "workspace_3d_action"):
            main_toolbar.addSeparator()
            main_toolbar.addAction(self.workspace_2d_action)
            main_toolbar.addAction(self.workspace_3d_action)

        main_toolbar.addSeparator()
        export_action = QAction(get_icon("export"), "Export CSV", self)
        export_action.triggered.connect(self.view.export_csv)
        self._bind_action_tip(export_action, f"Export the model to {WORKSPACE_DIR_NAME}/ CSV files.")
        main_toolbar.addAction(export_action)
        main_toolbar.addSeparator()
        if hasattr(self, "zoom_action"):
            main_toolbar.addAction(self.zoom_action)
        if hasattr(self, "fit_action"):
            main_toolbar.addAction(self.fit_action)
        if hasattr(self, "frame_action"):
            main_toolbar.addAction(self.frame_action)
        if hasattr(self, "measure_action"):
            main_toolbar.addAction(self.measure_action)
        if hasattr(self, "mini_map_action"):
            main_toolbar.addAction(self.mini_map_action)
        nav_action = QAction(get_icon("gizmo_move"), "View Navigation", self)
        nav_action.setCheckable(True)
        nav_action.setChecked(bool(getattr(self, "_view_navigation_enabled", False)))
        nav_action.toggled.connect(self._set_view_navigation_enabled)
        self._bind_action_tip(
            nav_action,
            "Toggle navigation mode: left-drag selection, middle-pan, right-rotate, wheel-zoom.",
        )
        main_toolbar.addAction(nav_action)
        self.view_navigation_action = nav_action
        if hasattr(self, "origin_action"):
            main_toolbar.addAction(self.origin_action)
        if hasattr(self, "grid_action"):
            main_toolbar.addAction(self.grid_action)
        if hasattr(self, "command_action"):
            main_toolbar.addAction(self.command_action)
        if hasattr(self, "full_action"):
            main_toolbar.addAction(self.full_action)
        self.main_toolbar = main_toolbar

    def _sync_mode_ui(self):
        is_3d = self.project_mode == "3d"
        if hasattr(self, "mesh_view_action"):
            if is_3d:
                self.mesh_view_action.setText("2D Connections")
                self.mesh_view_action.setToolTip("Preview the base 2D connections.")
                self.mesh_view_action.setStatusTip("Preview the base 2D connections.")
            else:
                self.mesh_view_action.setText("Connection View")
                self.mesh_view_action.setToolTip("Toggle connection view in the main canvas.")
                self.mesh_view_action.setStatusTip("Toggle connection view in the main canvas.")
        if hasattr(self, "mesh_3d_action"):
            self.mesh_3d_action.setEnabled(is_3d)
            if not is_3d and self.mesh_3d_action.isChecked():
                self.mesh_3d_action.setChecked(False)
        if hasattr(self, "properties_panel") and hasattr(self.properties_panel, "assembly_tab"):
            self.properties_panel.assembly_tab.refresh()
        self._update_mode_indicator()
        if is_3d and not self._mode_tip_shown:
            self.statusBar().showMessage(
                "3D mode: set extrude height/layers in Geometry, use 3D Preview to view tetra connections. Solver is still 2D.",
                9000,
            )
            self._mode_tip_shown = True
        if not is_3d:
            self._mode_tip_shown = False

    def _update_mode_indicator(self):
        if not hasattr(self, "mode_indicator"):
            return
        if self.project_mode == "3d":
            height = float(getattr(self.view, "extrude_height", 0.0))
            layers = int(getattr(self.view, "extrude_layers", 0))
            unit = getattr(self.view, "current_unit", "")
            text = f"3D · H {height:.3f} {unit} · L {layers}"
            tip = "3D mode: extruded tetra connections for preview/export. Solver is still 2D."
        else:
            text = "2D"
            tip = "2D mode: planar connections and 2D solver."
        self.mode_indicator.setText(text)
        self.mode_indicator.setToolTip(tip)
        self._update_interaction_hints()

    def _status_selected_object_text(self):
        inspector = getattr(self, "property_inspector", None)
        payload = getattr(inspector, "_selection_payload", None)
        if isinstance(payload, dict):
            kind = str(payload.get("kind", "")).lower()
            if kind == "part":
                try:
                    part = inspector._resolve_part(payload.get("part_id"))
                except Exception:
                    part = None
                if part is not None:
                    return str(getattr(part, "name", "Part"))
                return "Part"
            if kind == "material":
                try:
                    material = inspector._resolve_material(payload.get("serial"))
                except Exception:
                    material = None
                if material is not None:
                    return str(getattr(material, "name", "Material"))
                return "Material"
            if kind == "bc":
                name = payload.get("name")
                if name:
                    return str(name)
                try:
                    return f"BC {int(payload.get('index', 0)) + 1}"
                except Exception:
                    return "Boundary Condition"
            if kind == "load":
                name = payload.get("name")
                if name:
                    return str(name)
                try:
                    return f"Load {int(payload.get('index', 0)) + 1}"
                except Exception:
                    return "Load"
            if kind in {"mesh", "mesh_nodes", "mesh_elements"}:
                return "Particles"
            if kind:
                return kind.replace("_", " ").title()

        selected_part = getattr(self.view, "get_selected_part", lambda: None)()
        if selected_part is not None:
            return str(getattr(selected_part, "name", "Part"))
        return "None"

    def _update_interaction_hints(self):
        if getattr(self, "_updating_interaction_hints", False):
            self._pending_interaction_hints_update = True
            return
        self._updating_interaction_hints = True
        workflow_stage_indicator = getattr(self, "workflow_stage_indicator", None)
        interaction_mode_indicator = getattr(self, "interaction_mode_indicator", None)
        nav_indicator = getattr(self, "nav_indicator", None)
        selection_indicator = getattr(self, "selection_indicator", None)
        node_count_indicator = getattr(self, "node_count_indicator", None)
        element_count_indicator = getattr(self, "element_count_indicator", None)
        units_indicator = getattr(self, "units_indicator", None)
        mesh_stats_indicator = getattr(self, "mesh_stats_indicator", None)
        mouse_hint_label = getattr(self, "mouse_hint_label", None)
        if (
            workflow_stage_indicator is None
            or interaction_mode_indicator is None
            or nav_indicator is None
            or selection_indicator is None
            or node_count_indicator is None
            or element_count_indicator is None
            or units_indicator is None
            or mesh_stats_indicator is None
            or mouse_hint_label is None
        ):
            self._updating_interaction_hints = False
            return

        try:
            is_3d = self.project_mode == "3d"
            selection_map = {
                "none": "Object",
                "auto": "Auto",
                "face": "Face",
                "edge": "Edge",
                "point": "Point",
            }
            if is_3d:
                raw_selection = getattr(getattr(self, "view_3d", None), "_selection_mode", "auto")
                selection_mode_text = selection_map.get(str(raw_selection).lower(), str(raw_selection).title())
                interaction_mode = "View Mode" if self._view_navigation_enabled else "Selection Mode"
                view_3d = getattr(self, "view_3d", None)
                selected_text = self._status_selected_object_text()
                if view_3d is not None and hasattr(view_3d, "selection_counts"):
                    try:
                        counts = view_3d.selection_counts()
                        if selected_text == "None":
                            selected_text = (
                                f"{counts.get('faces', 0)}F {counts.get('edges', 0)}E {counts.get('points', 0)}P"
                                if any(counts.values()) else selection_mode_text
                            )
                    except Exception:
                        selected_text = selection_mode_text if selected_text == "None" else selected_text
                elif selected_text == "None":
                    selected_text = selection_mode_text
                if self._view_navigation_enabled:
                    mouse_text = "Mouse: L drag select | M pan | R rotate | Wheel zoom"
                else:
                    mouse_text = "Mouse: L select | Enable View Navigation for pan and rotate"
                node_count = 0
                element_count = 0
                if view_3d is not None and getattr(view_3d, "_last_mesh", None) is not None:
                    try:
                        nodes, elements = view_3d._last_mesh
                        node_count = len(nodes)
                        element_count = len(elements)
                    except Exception:
                        pass
                if node_count == 0:
                    connection_nodes = getattr(self.view, "connection_nodes", None)
                    node_count = len(connection_nodes) if connection_nodes is not None else 0
                if element_count == 0:
                    connections = getattr(self.view, "connections", None)
                    element_count = len(connections) if connections is not None else 0
            else:
                current_tool = str(
                    getattr(self.view, "tool", getattr(self.view, "current_tool", "")) or ""
                ).lower()
                interaction_mode = "View Mode" if self._view_navigation_enabled else (
                    "Selection Mode" if current_tool in {"select", "selection", "part", "move"} else "Sketch Mode"
                )
                selected_text = self._status_selected_object_text()
                mouse_text = "Mouse: L draw/select | Wheel zoom | Enable View Navigation for pan"
                connection_nodes = getattr(self.view, "connection_nodes", None)
                connections = getattr(self.view, "connections", None)
                node_count = len(connection_nodes) if connection_nodes is not None else 0
                element_count = len(connections) if connections is not None else 0

            interaction_mode_indicator.setText(interaction_mode)
            nav_indicator.setText("NAV" if self._view_navigation_enabled else "")
            selection_indicator.setText(f"⌖ {selected_text}")
            node_count_indicator.setText(f"● {node_count}")
            element_count_indicator.setText(f"▲ {element_count}")
            units_indicator.setText(str(getattr(self.view, "current_unit", "m") or "m"))
            mesh_stats_indicator.setText("")
            mouse_hint_label.setText(
                mouse_text
                .replace("Mouse: ", "")
                .replace("Wheel", "W")
                .replace("left-drag", "L-drag")
                .replace("rotate", "orbit")
            )
            workflow_stage_indicator.setToolTip("Current workflow stage ribbon selection.")
            interaction_mode_indicator.setToolTip("Current interaction mode for the active workspace.")
            nav_indicator.setToolTip("Shows whether viewport navigation shortcuts are active.")
            selection_indicator.setToolTip("Currently selected object or active sub-selection summary.")
            node_count_indicator.setToolTip("Current number of particles in the active workspace.")
            element_count_indicator.setToolTip("Current number of internal connections in the active workspace.")
            units_indicator.setToolTip("Active project units.")
            mesh_stats_indicator.setToolTip("Reserved for additional particle statistics.")
            mouse_hint_label.setToolTip("Current mouse control summary for the active workspace.")
        finally:
            self._updating_interaction_hints = False
            if getattr(self, "_pending_interaction_hints_update", False):
                self._pending_interaction_hints_update = False
                QTimer.singleShot(0, self._update_interaction_hints)

    def _create_loads_toolbar(self):
        loads_toolbar = QToolBar("Loads & BCs")
        loads_toolbar.setObjectName("LoadsToolbar")
        loads_toolbar.setMovable(False)
        loads_toolbar.setFloatable(False)
        self.addToolBar(Qt.TopToolBarArea, loads_toolbar)
        toolbar_icon_px = toolbar_icon_size()
        loads_toolbar.setIconSize(QSize(toolbar_icon_px, toolbar_icon_px))
        loads_toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        if self._toolbar_style:
            loads_toolbar.setStyleSheet(self._toolbar_style)
        loads_toolbar.setContentsMargins(
            UI_TOKENS.spacing_xs,
            UI_TOKENS.spacing_xs,
            UI_TOKENS.spacing_xs,
            UI_TOKENS.spacing_xs,
        )

        iv_action = QAction(get_icon("velocity"), "Initial Velocity", self)
        iv_action.setCheckable(True)
        iv_action.triggered.connect(lambda: self.view.set_tool("initial_velocity"))
        iv_action.setEnabled(False)
        self._bind_action_tip(
            iv_action,
            "Legacy: not used by current CPD-main run path. Use the Boundary Conditions stage for velocity BCs/profiles.",
        )
        loads_toolbar.addAction(iv_action)
        apply_load_action = QAction(get_icon("load"), "Apply Load", self)
        apply_load_action.triggered.connect(self._apply_load_from_toolbar)
        self._bind_action_tip(
            apply_load_action,
            "Apply load to current selection (same action as right-click apply load).",
        )
        loads_toolbar.addAction(apply_load_action)
        apply_bc_action = QAction(get_icon("constraint"), "Apply Boundary Condition", self)
        apply_bc_action.triggered.connect(self._apply_bc_from_toolbar)
        self._bind_action_tip(
            apply_bc_action,
            "Apply boundary condition to current selection (same action as right-click apply BC).",
        )
        loads_toolbar.addAction(apply_bc_action)
        self.loads_toolbar = loads_toolbar

    def finish_geometry_stage(self):
        return self.geometry_controller.finish_stage()

    def _open_porous_dialog(self, initial_settings=None, target_part=None):
        dialog = PorousMaterialDialog(
            self.view,
            self,
            initial_settings=initial_settings,
            target_part=target_part,
        )
        dialog.exec()

    def _prompt_vector(self, title, dx=0.0, dy=0.0, label_dx="dx", label_dy="dy"):
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        self._prepare_modal_dialog(dialog)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        dx_spin = QDoubleSpinBox()
        dx_spin.setRange(-1e9, 1e9)
        dx_spin.setDecimals(4)
        dx_spin.setValue(dx)
        dy_spin = QDoubleSpinBox()
        dy_spin.setRange(-1e9, 1e9)
        dy_spin.setDecimals(4)
        dy_spin.setValue(dy)
        form.addRow(label_dx, dx_spin)
        form.addRow(label_dy, dy_spin)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        ok = dialog.exec() == QDialog.Accepted
        return dx_spin.value(), dy_spin.value(), ok

    def _prompt_linear_pattern(self, dx=0.0, dy=0.0, count=3):
        dialog = QDialog(self)
        dialog.setWindowTitle("Linear Pattern")
        self._prepare_modal_dialog(dialog)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        count_spin = QSpinBox()
        count_spin.setRange(1, 1000)
        count_spin.setValue(count)
        dx_spin = QDoubleSpinBox()
        dx_spin.setRange(-1e9, 1e9)
        dx_spin.setDecimals(4)
        dx_spin.setValue(dx)
        dy_spin = QDoubleSpinBox()
        dy_spin.setRange(-1e9, 1e9)
        dy_spin.setDecimals(4)
        dy_spin.setValue(dy)
        form.addRow("Copies", count_spin)
        form.addRow("dx", dx_spin)
        form.addRow("dy", dy_spin)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        ok = dialog.exec() == QDialog.Accepted
        return count_spin.value(), dx_spin.value(), dy_spin.value(), ok

    def _prompt_mirror(self, default_x=0.0, default_y=0.0):
        dialog = QDialog(self)
        dialog.setWindowTitle("Mirror")
        self._prepare_modal_dialog(dialog)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        axis_combo = QComboBox()
        axis_combo.addItems(["Vertical (x = ...)", "Horizontal (y = ...)"])
        offset_spin = QDoubleSpinBox()
        offset_spin.setRange(-1e9, 1e9)
        offset_spin.setDecimals(4)
        offset_spin.setValue(default_x)
        keep_box = QCheckBox("Keep original (create mirrored copy)")
        keep_box.setChecked(True)
        form.addRow("Axis", axis_combo)
        form.addRow("Offset", offset_spin)
        layout.addLayout(form)
        layout.addWidget(keep_box)

        def _sync_offset():
            if axis_combo.currentIndex() == 0:
                offset_spin.setValue(default_x)
            else:
                offset_spin.setValue(default_y)

        axis_combo.currentIndexChanged.connect(_sync_offset)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        ok = dialog.exec() == QDialog.Accepted
        axis = "vertical" if axis_combo.currentIndex() == 0 else "horizontal"
        return axis, offset_spin.value(), keep_box.isChecked(), ok

    def _selected_part_or_warn(self):
        if self.view.active_module != "Part":
            QMessageBox.information(self, "Geometry Tool", "These tools are available in the Geometry module.")
            return None
        part = self.view.get_selected_part()
        if not part:
            QMessageBox.information(self, "Select Part", "Select a part first.")
            return None
        return part

    def _move_selected_part(self):
        if self.view.active_module != "Part":
            QMessageBox.information(self, "Geometry Tool", "Move is available in the Geometry module.")
            return
        self.view.set_tool("move")

    def _copy_selected_part(self):
        if self.view.active_module != "Part":
            QMessageBox.information(self, "Geometry Tool", "Copy is available in the Geometry module.")
            return
        self.view.set_tool("copy")

    def _linear_pattern_selected_part(self):
        part = self._selected_part_or_warn()
        if not part:
            return
        count, dx, dy, ok = self._prompt_linear_pattern()
        if ok:
            self.view.copy_selected_part(dx, dy, count=count, name_suffix="Pattern")

    def _mirror_selected_part(self):
        if self.view.active_module != "Part":
            QMessageBox.information(self, "Geometry Tool", "Mirror is available in the Geometry module.")
            return
        self.view.set_tool("mirror")

    def _join_sketches(self):
        if self.view.active_module != "Part":
            QMessageBox.information(self, "Join Sketches", "Join is available in the Geometry module.")
            return
        self.view.join_sketches()

    # =========================
    # PROJECT MANAGEMENT
    # =========================
    def new_project(self, mode=None):
        # QAction.triggered passes a checked bool; treat it as no explicit mode.
        if isinstance(mode, bool):
            mode = None
        if self.project_dirty:
            # Capture the latest edits before asking whether to discard.
            self._autosave()
            reply = self._prompt_unsaved_changes(
                "The current project has unsaved changes.\nSave before creating a new project?"
            )

            if reply == "save":
                self.save_project()
                if self.project_dirty:  # Save cancelled
                    return
            elif reply == "cancel":
                return

        if mode is None:
            mode = self._prompt_project_mode()
        if not mode:
            return

        # FULL RESET
        self.view.clear_all()
        self._reset_stage_panels(clear_bc_lists=True)
        self.project_state = ProjectState()
        self.view.project_state = self.project_state
        if hasattr(self, "properties_panel") and hasattr(self.properties_panel, "set_project_state"):
            self.properties_panel.set_project_state(self.project_state)
        if hasattr(self, "project_tree") and hasattr(self.project_tree, "set_sources"):
            self.project_tree.set_sources(self.view, self.project_state)
        if hasattr(self, "property_inspector") and hasattr(self.property_inspector, "set_project_state"):
            self.property_inspector.set_project_state(self.project_state)
        self.current_project_file = None
        self.project_dirty = False
        self.project_mode = mode
        self.view.set_project_mode(mode)
        self.model3d = self._new_model3d()
        self._undo_stack_3d.clear()
        self._redo_stack_3d.clear()
        self._toggle_workspace_3d(mode == "3d")
        self._sync_mode_ui()
        if hasattr(self, "dim_action"):
            self.dim_action.setChecked(self.view.show_dimensions)

        self.setWindowTitle("CPD SimStudio v28 - New Project")
        self.apply_stage_ui(ProjectStage.GEOMETRY)
        self.view.fit_view()


    def save_project(self):
        if not self.current_project_file:
            self.save_project_as()
            return
        self._write_project_file(self.current_project_file)
        name = os.path.basename(self.current_project_file)
        if hasattr(self, "statusBar"):
            self.statusBar().showMessage(
                f"Saved (overwritten): {name}",
                self._status_timeout_ms,
            )
        QMessageBox.information(
            self,
            "Saved",
            f"Saved by overwriting the existing file:\n{name}",
        )

    def save_project_as(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project",
            "",
            "CPD Project (*.cpd)"
        )
        if not file_path:
            return
        if not file_path.lower().endswith(".cpd"):
            file_path = f"{file_path}.cpd"

        self._write_project_file(file_path)    
    
    def load_project(self):
        if self.project_dirty:
            # Capture the latest edits before asking whether to discard.
            self._autosave()
            reply = self._prompt_unsaved_changes(
                "The current project has unsaved changes.\nSave before opening another project?"
            )

            if reply == "save":
                self.save_project()
                if self.project_dirty:
                    return
            elif reply == "cancel":
                return

        filePath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            "",
            "CPD Project (*.cpd *.cpdproj)"
        )

        if not filePath:
            return

        self._load_project_from_path(filePath)

    def _prompt_project_mode(self):
        options = ["2D", "3D"]
        default_idx = 1 if self.project_mode == "3d" else 0
        selection, ok = QInputDialog.getItem(
            self,
            "New Project",
            "Project mode:",
            options,
            default_idx,
            False,
        )
        if not ok:
            return None
        return "3d" if selection.lower().startswith("3") else "2d"

    def import_geometry(self):
        file_filter = get_import_filter()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Geometry",
            "",
            file_filter,
        )

        if not file_path:
            return

        # Auto-route 3D CAD files (STEP/IGES/STL) to the 3D import flow so the
        # user does not have to know they need a different menu entry.
        from importers import CAD_3D_EXTENSIONS
        ext = os.path.splitext(file_path)[1].lower()
        if ext in CAD_3D_EXTENSIONS:
            reply = QMessageBox.question(
                self,
                "3D CAD File",
                f"'{os.path.basename(file_path)}' is a 3D CAD file ({ext}). "
                "Import it as 3D geometry (project will switch to 3D mode)?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply != QMessageBox.Yes:
                return
            if self.project_mode != "3d":
                self.project_mode = "3d"
                self.view.set_project_mode(self.project_mode)
                self.apply_stage_ui(ProjectStage.GEOMETRY)
            if self.view.import_cad_shape(file_path):
                self.project_dirty = True
            return

        try:
            sketches = import_file(file_path)
        except ImporterUnavailable as exc:
            QMessageBox.warning(self, "Importer Missing", str(exc))
            return
        except ImporterError as exc:
            QMessageBox.warning(self, "Import Error", str(exc))
            return
        except Exception as exc:
            QMessageBox.critical(self, "Import Error", str(exc))
            return

        if not sketches:
            QMessageBox.warning(self, "Import", "No drawable geometry found in the file.")
            return

        if self.active_stage.value > ProjectStage.GEOMETRY.value:
            reply = QMessageBox.question(
                self,
                "Import Geometry",
                "Importing will reset the project to Geometry and clear existing data. Continue?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if reply != QMessageBox.Yes:
                return
            self.view.clear_all()
            self._reset_stage_panels(clear_bc_lists=True)
            self.apply_stage_ui(ProjectStage.GEOMETRY)
        elif self.view.parts or self.view.sketches:
            reply = QMessageBox.question(
                self,
                "Import Geometry",
                "Replace existing geometry? Yes = Replace, No = Append, Cancel = Abort.",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if reply == QMessageBox.Cancel:
                return
            if reply == QMessageBox.Yes:
                self.view.clear_all()
                self._reset_stage_panels(clear_bc_lists=True)
                self.apply_stage_ui(ProjectStage.GEOMETRY)

        def is_closed_path(pts, tol=1e-6):
            if not pts or len(pts) < 3:
                return False
            dx = pts[0][0] - pts[-1][0]
            dy = pts[0][1] - pts[-1][1]
            return (dx * dx + dy * dy) <= tol * tol

        closed_count = sum(1 for pts in sketches if is_closed_path(pts))
        convert_closed = False
        base_name = "Imported Part"

        if closed_count:
            reply = QMessageBox.question(
                self,
                "Import Geometry",
                f"Detected {closed_count} closed path(s). Convert them to parts?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Cancel:
                return
            convert_closed = reply == QMessageBox.Yes
            if convert_closed:
                name, ok = QInputDialog.getText(
                    self,
                    "Part Naming",
                    "Base name for imported parts:",
                    text=base_name,
                )
                if ok and name.strip():
                    base_name = name.strip()

        added_sketches, added_parts = self.view.add_imported_geometry(
            sketches,
            convert_closed=convert_closed,
            base_name=base_name,
        )
        if added_sketches or added_parts:
            self.project_dirty = True
            QMessageBox.information(
                self,
                "Import Complete",
                f"Imported {added_parts} part(s) and {added_sketches} sketch path(s).",
            )

    def import_cad_geometry(self):
        filters = (
            "CAD Files (*.step *.stp *.iges *.igs *.stl "
            "*.STEP *.STP *.IGES *.IGS *.STL);;All Files (*)"
        )
        file_path, _ = QFileDialog.getOpenFileName(self, "Import 3D CAD", "", filters)
        if not file_path:
            return
        if self.project_mode != "3d":
            reply = QMessageBox.question(
                self,
                "Switch to 3D",
                "Importing CAD will switch the project to 3D mode. Continue?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Yes,
            )
            if reply != QMessageBox.Yes:
                return
            self.project_mode = "3d"
            self.view.set_project_mode(self.project_mode)
            self.apply_stage_ui(ProjectStage.GEOMETRY)
        ok = self.view.import_cad_shape(file_path)
        if ok:
            self.project_dirty = True

    def _collect_solver_settings_for_state(self):
        return self.solver_controller.collect_solver_settings_for_state()

    def _collect_solver_settings_for_state_impl(self):
        settings = {}
        state = getattr(self, "project_state", None)
        if state is not None and isinstance(getattr(state, "solver_settings", None), dict):
            for key, value in state.solver_settings.items():
                key_str = str(key)
                if key_str.startswith("_"):
                    continue
                settings[key_str] = copy.deepcopy(value)

        job_tab = getattr(getattr(self, "properties_panel", None), "job_tab", None)
        if job_tab is not None and hasattr(job_tab, "export_solver_settings"):
            try:
                exported = job_tab.export_solver_settings()
            except Exception:
                exported = {}
            if isinstance(exported, dict):
                settings.update(copy.deepcopy(exported))
        return settings

    def _sync_solver_settings_to_project_state(self):
        return self.solver_controller.sync_solver_settings_to_project_state()

    def _sync_solver_settings_to_project_state_impl(self):
        state = getattr(self, "project_state", None)
        if state is None:
            state = ProjectState()
            self.project_state = state
            self.view.project_state = state
            if hasattr(self, "properties_panel") and hasattr(self.properties_panel, "set_project_state"):
                self.properties_panel.set_project_state(state)
            if hasattr(self, "project_tree") and hasattr(self.project_tree, "set_sources"):
                self.project_tree.set_sources(self.view, self.project_state)
            if hasattr(self, "property_inspector") and hasattr(self.property_inspector, "set_project_state"):
                self.property_inspector.set_project_state(self.project_state)
        if not isinstance(getattr(state, "solver_settings", None), dict):
            state.solver_settings = {}
        private_entries = {}
        for key, value in state.solver_settings.items():
            key_str = str(key)
            if not key_str.startswith("_"):
                continue
            if key_str == "_sketch_view":
                continue
            try:
                copy.deepcopy(value)
            except Exception:
                continue
            private_entries[key_str] = value
        public_entries = self._collect_solver_settings_for_state()
        merged = {}
        merged.update(private_entries)
        merged.update(public_entries)
        state.solver_settings = merged

    def _build_schema_project_state(self):
        self._sync_solver_settings_to_project_state()
        state = getattr(self, "project_state", None)
        parts = []
        boundary_conditions = []
        loads = []
        solver_settings = {}
        if state is not None:
            parts = copy.deepcopy(getattr(self.view, "parts", getattr(state, "parts", [])))
            boundary_conditions = copy.deepcopy(getattr(state, "boundary_conditions", []))
            loads = copy.deepcopy(getattr(state, "loads", []))
            solver_settings = copy.deepcopy(getattr(state, "solver_settings", {}))
        payload = {
            "analysis_type": self._normalized_analysis_type(),
            "dimension": self._normalized_dimension(),
            "parts": parts,
            "materials": copy.deepcopy(getattr(state, "materials", {})),
            "interfaces": self.view.serialize_interfaces(),
            "boundary_conditions": boundary_conditions,
            "loads": loads,
            "solver_settings": solver_settings,
        }
        return ProjectState.from_dict(payload)

    def _apply_solver_settings_to_ui(self):
        return self.solver_controller.apply_solver_settings_to_ui()

    def _apply_solver_settings_to_ui_impl(self):
        state = getattr(self, "project_state", None)
        if state is None:
            return
        settings = getattr(state, "solver_settings", {})
        if not isinstance(settings, dict):
            return
        job_tab = getattr(getattr(self, "properties_panel", None), "job_tab", None)
        if job_tab is not None and hasattr(job_tab, "apply_solver_settings"):
            try:
                job_tab.apply_solver_settings(copy.deepcopy(settings))
            except Exception:
                pass

    def _warn_project_state_validation(self, context):
        state = getattr(self, "project_state", None)
        if state is None or not hasattr(state, "validate"):
            return
        try:
            _, warnings = state.validate()
        except Exception:
            return
        if not warnings:
            return
        logger = logging.getLogger(__name__)
        for warning in warnings:
            logger.warning("ProjectState warning during %s: %s", context, warning)
        if hasattr(self, "statusBar") and callable(getattr(self, "statusBar")):
            self.statusBar().showMessage(
                f"{context}: {warnings[0]}",
                self._status_timeout_ms,
            )
     
    def _write_project_file(self, file_path):
        project_data = self._build_project_data()

        with open(file_path, "w") as f:
            json.dump(project_data, f, indent=4)

        self._export_job_artifacts(
            file_path,
            export_inputs=True,
            include_results=True,
        )

        self.current_project_file = file_path
        self._add_recent_project(file_path)
        self.setWindowTitle(f"CPD SimStudio v28 - {os.path.basename(file_path)}")
        self.project_dirty = False    
        self._clear_autosave()

    def _build_project_data(self):
        self._warn_project_state_validation("save")
        active_stage = getattr(self, "active_stage", ProjectStage.GEOMETRY)
        project_meta = {
            "version": "2.0",
            "schema_version": CURRENT_SCHEMA_VERSION,
            "app_version": APP_VERSION,
            "last_stage": active_stage.name,
            "analysis_type": self._normalized_analysis_type(),
            "dimension": self._normalized_dimension(),
            "mode": self.project_mode,
            "workspace": "3d" if getattr(self, "_workspace_3d", False) else "2d",
        }
        project_data = {
            "model3d": self.model3d,
            "units": "m",
            "sketches": self.view.sketches,
            "sketch_meta": self.view.sketch_meta,
            "dimensions": self.view.dimensions,
            "constraints": self.view.constraints,
            "show_dimensions": self.view.show_dimensions,
            "geometry": self.view.serialize_geometry(),
            "modeling": {
                "extrude_height": float(self.view.extrude_height),
                "extrude_layers": int(self.view.extrude_layers),
            },
            "connection_settings": {
                "min_spacing_factor": float(getattr(self.view, "mesh_min_spacing_factor", 1.0)),
                "boundary_thickness": float(getattr(self.view, "mesh_boundary_thickness", 0.0)),
                "boundary_spacing_factor": float(getattr(self.view, "mesh_boundary_spacing_factor", 1.0)),
            },
            "preview_settings": {
                "fast_preview_enabled": bool(getattr(self.view, "fast_preview_enabled", True)),
                "fast_preview_connection_limit": int(
                    getattr(self.view, "fast_preview_connection_limit", 0)
                ),
                "gpu_point_preview_enabled": bool(
                    getattr(self.view, "gpu_point_preview_enabled", True)
                ),
                "gpu_point_preview_auto": bool(
                    getattr(self.view, "gpu_point_preview_auto", True)
                ),
                "gpu_point_preview_threshold": int(
                    getattr(self.view, "gpu_point_preview_threshold", 0)
                ),
                "freeform_auto_convert_enabled": bool(
                    getattr(self.view, "freeform_auto_convert_enabled", False)
                ),
            },
            "initial_velocities": self.view.initial_velocities,
        }
        schema_state = self._build_schema_project_state()
        merge_project_state_into_project_data(project_data, schema_state)
        project_data["project_meta"] = project_meta
        return apply_schema_migrations(project_data)

    def _autosave_dir(self):
        return str(get_autosave_dir())

    def _autosave_path(self):
        base_dir = self._autosave_dir()
        if self.current_project_file:
            base = os.path.splitext(os.path.basename(self.current_project_file))[0]
            digest = hashlib.md5(self.current_project_file.encode("utf-8")).hexdigest()[:8]
            filename = f"{base}.{digest}.autosave.cpd"
        else:
            filename = f"unsaved_{self._autosave_session_id}.autosave.cpd"
        return os.path.join(base_dir, filename)

    def _autosave(self):
        if not self._autosave_enabled or not self.project_dirty:
            return
        try:
            autosave_path = self._autosave_path()
            project_data = self._build_project_data()
            with open(autosave_path, "w") as f:
                json.dump(project_data, f, indent=4)
            self._last_autosave_path = autosave_path
            self.statusBar().showMessage(
                f"Autosaved to {os.path.basename(autosave_path)}",
                self._status_timeout_ms,
            )
        except Exception as exc:
            self.statusBar().showMessage(
                f"Autosave failed: {exc}",
                self._status_timeout_ms,
            )

    def _clear_autosave(self):
        if not self._last_autosave_path:
            return
        try:
            if os.path.exists(self._last_autosave_path):
                os.remove(self._last_autosave_path)
        except Exception:
            pass
        self._last_autosave_path = None

    def _list_autosave_files(self):
        files = []
        seen = set()
        for base_dir in self._autosave_search_dirs():
            try:
                names = os.listdir(base_dir)
            except Exception:
                continue
            for name in names:
                if not name.endswith(".autosave.cpd"):
                    continue
                path = os.path.abspath(os.path.join(base_dir, name))
                if not os.path.isfile(path) or path in seen:
                    continue
                files.append(path)
                seen.add(path)
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return files

    def _autosave_search_dirs(self):
        dirs = []
        seen = set()

        def _add(path):
            if not path:
                return
            abs_path = os.path.abspath(path)
            if abs_path in seen:
                return
            if os.path.isdir(abs_path):
                dirs.append(abs_path)
                seen.add(abs_path)

        _add(self._autosave_dir())

        project_root = _project_root_dir()
        _add(os.path.join(project_root, "autosave"))
        _add(os.path.join(project_root, WORKSPACE_DIR_NAME, "autosave"))
        _add(os.path.join(os.path.dirname(project_root), WORKSPACE_DIR_NAME, "autosave"))

        if self.current_project_file:
            project_dir = os.path.dirname(os.path.abspath(self.current_project_file))
            _add(os.path.join(project_dir, "autosave"))
            _add(os.path.join(project_dir, WORKSPACE_DIR_NAME, "autosave"))

        return dirs

    def _prompt_restore_autosave(self):
        files = self._list_autosave_files()
        if not files:
            return False
        latest = files[0]
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(latest)))
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Autosave Found")
        box.setText("Autosave file(s) were found.")
        box.setInformativeText(
            f"Latest autosave: {os.path.basename(latest)}\nLast modified: {timestamp}\n\nRestore it now?"
        )
        restore_btn = box.addButton("Restore", QMessageBox.AcceptRole)
        ignore_btn = box.addButton("Ignore", QMessageBox.RejectRole)
        discard_btn = box.addButton("Discard All", QMessageBox.DestructiveRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked == restore_btn:
            self._load_autosave(latest)
            return True
        if clicked == discard_btn:
            for path in files:
                try:
                    os.remove(path)
                except Exception:
                    pass
        return False

    def _recover_autosave(self):
        files = self._list_autosave_files()
        autosave_dir = os.path.dirname(files[0]) if files else self._autosave_dir()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Recover Autosave",
            autosave_dir,
            "Autosave (*.autosave.cpd);;All Files (*)"
        )
        if not file_path:
            if not files:
                QMessageBox.information(
                    self,
                    "Recover Autosave",
                    "No autosave files were found in known locations.\n"
                    f"Current autosave folder: {self._autosave_dir()}",
                )
            return
        self._load_autosave(file_path)

    def _load_autosave(self, file_path):
        logging.getLogger(__name__).info("Autosave recovery started: %s", file_path)
        if not self._load_project_from_path(file_path, show_message=False):
            logging.getLogger(__name__).warning("Autosave recovery failed: %s", file_path)
            return
        self.current_project_file = None
        self.project_dirty = True
        self._last_autosave_path = file_path
        self.setWindowTitle("CPD SimStudio v28 - Recovered Autosave")
        self._schedule_startup_window_state(reason="autosave-recovery")
        self._log_window_state(f"Autosave recovery finished [{os.path.basename(file_path)}]")
        QMessageBox.information(
            self,
            "Autosave Restored",
            "Autosave has been restored. Save the project to keep changes.",
        )
        reply = QMessageBox.question(
            self,
            "Save Recovered Project",
            "Recovered autosave is loaded.\nDo you want to save it now as a .cpd file?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply == QMessageBox.Yes:
            self.save_project_as()

    def _export_job_artifacts(self, project_file, export_inputs=True, include_results=True):
        return self.solver_controller.export_job_artifacts(
            project_file,
            export_inputs=export_inputs,
            include_results=include_results,
        )

    def _export_job_artifacts_impl(self, project_file, export_inputs=True, include_results=True):
        base_dir = _project_root_dir()
        workspace_dir = _workspace_dir()
        project_dir = os.path.dirname(project_file)
        project_name = os.path.splitext(os.path.basename(project_file))[0]
        artifacts_dir = os.path.join(project_dir, f"{project_name}_artifacts")
        inputs_dir = os.path.join(artifacts_dir, "inputs")
        results_dir = os.path.join(artifacts_dir, "results")
        os.makedirs(inputs_dir, exist_ok=True)

        export_ok = False
        if export_inputs:
            try:
                export_ok = self.view.export_csv(silent=True)
            except Exception:
                export_ok = False

        if export_ok:
            input_files = [
                "bc.csv",
                "loads.csv",
                "solver_particles.csv",
                "preview_particles.csv",
                "initial_velocities.csv",
                "interfaces.csv",
                "operations.csv",
                "particles_3d.csv",
                "connections_3d.csv",
                "fixed_particles_3d.csv",
                "force_bc_3d.csv",
            ]
            for fname in input_files:
                src = os.path.join(workspace_dir, fname)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(inputs_dir, fname))
            # Legacy artifact filename fallback (pre-interfaces.csv).
            legacy_interfaces_src = os.path.join(workspace_dir, "interactions.csv")
            interfaces_dst = os.path.join(inputs_dir, "interfaces.csv")
            if (not os.path.exists(interfaces_dst)) and os.path.exists(legacy_interfaces_src):
                shutil.copy2(legacy_interfaces_src, interfaces_dst)

            cpd_input_src = _workspace_input_path()
            if not os.path.isdir(cpd_input_src):
                cpd_input_src = os.path.join(workspace_dir, "setup")
            cpd_input_dest = os.path.join(inputs_dir, "cpd_main_input")
            os.makedirs(cpd_input_dest, exist_ok=True)
            for fname in (
                "particles.csv",
                "connections.csv",
                "materials.csv",
                "fixed.csv",
                "velocity.csv",
                "force_targets.csv",
                "velocity_targets.csv",
                "force_time.csv",
                "velocity_time.csv",
            ):
                src = os.path.join(cpd_input_src, fname)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(cpd_input_dest, fname))

            config_src = os.path.join(base_dir, "CPD-main", "config.yml")
            if os.path.exists(config_src):
                shutil.copy2(config_src, os.path.join(inputs_dir, "config.yml"))

        if include_results:
            os.makedirs(results_dir, exist_ok=True)
            results_src = _workspace_output_path("results")
            if not os.path.isdir(results_src):
                results_src = os.path.join(workspace_dir, "results")
            if os.path.isdir(results_src):
                for fname in os.listdir(results_src):
                    src = os.path.join(results_src, fname)
                    if os.path.isfile(src):
                        shutil.copy2(src, os.path.join(results_dir, fname))

            for fname in ("initial_pos.csv", "final_pos.csv"):
                src = _workspace_output_path(fname)
                if not os.path.exists(src):
                    src = os.path.join(workspace_dir, fname)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(results_dir, fname))

            for fname in (
                "pos_history.npy",
                "displacement_history.npy",
                "strain_history.npy",
                "stress_history.npy",
            ):
                src = _workspace_output_path(fname)
                if not os.path.exists(src):
                    src = os.path.join(workspace_dir, fname)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(results_dir, fname))

    def _restore_workspace_results_from_artifacts(self, project_file):
        return self.results_controller.restore_workspace_results_from_artifacts(project_file)

    def _restore_workspace_results_from_artifacts_impl(self, project_file):
        workspace_dir = _workspace_dir()
        workspace_output_dir = _workspace_output_path()
        workspace_results_dir = _workspace_output_path("results")
        workspace_pos_history = _workspace_output_path("pos_history.npy")
        workspace_displacement_history = _workspace_output_path("displacement_history.npy")
        workspace_strain_history = _workspace_output_path("strain_history.npy")
        workspace_stress_history = _workspace_output_path("stress_history.npy")
        legacy_workspace_results_dir = os.path.join(workspace_dir, "results")
        legacy_workspace_pos_history = os.path.join(workspace_dir, "pos_history.npy")

        project_dir = os.path.dirname(project_file)
        project_name = os.path.splitext(os.path.basename(project_file))[0]
        artifacts_results = os.path.join(project_dir, f"{project_name}_artifacts", "results")
        if not os.path.isdir(artifacts_results):
            return False

        # Clear stale workspace results from previously opened projects.
        if os.path.isdir(workspace_results_dir):
            for fname in os.listdir(workspace_results_dir):
                if fname.startswith("step_") and fname.endswith(".csv"):
                    try:
                        os.remove(os.path.join(workspace_results_dir, fname))
                    except Exception:
                        pass
        if os.path.isdir(legacy_workspace_results_dir):
            for fname in os.listdir(legacy_workspace_results_dir):
                if fname.startswith("step_") and fname.endswith(".csv"):
                    try:
                        os.remove(os.path.join(legacy_workspace_results_dir, fname))
                    except Exception:
                        pass
        for stale_path in (
            _workspace_output_path("initial_pos.csv"),
            _workspace_output_path("final_pos.csv"),
            workspace_pos_history,
            workspace_displacement_history,
            workspace_strain_history,
            workspace_stress_history,
            os.path.join(workspace_dir, "initial_pos.csv"),
            os.path.join(workspace_dir, "final_pos.csv"),
            legacy_workspace_pos_history,
        ):
            if os.path.exists(stale_path):
                try:
                    os.remove(stale_path)
                except Exception:
                    pass

        copied_any = False
        os.makedirs(workspace_output_dir, exist_ok=True)
        os.makedirs(workspace_results_dir, exist_ok=True)
        for fname in os.listdir(artifacts_results):
            src = os.path.join(artifacts_results, fname)
            if not os.path.isfile(src):
                continue
            if fname.startswith("step_") and fname.endswith(".csv"):
                shutil.copy2(src, os.path.join(workspace_results_dir, fname))
                copied_any = True
            elif fname == "pos_history.npy":
                shutil.copy2(src, workspace_pos_history)
                copied_any = True
            elif fname == "displacement_history.npy":
                shutil.copy2(src, workspace_displacement_history)
                copied_any = True
            elif fname == "strain_history.npy":
                shutil.copy2(src, workspace_strain_history)
                copied_any = True
            elif fname == "stress_history.npy":
                shutil.copy2(src, workspace_stress_history)
                copied_any = True
            elif fname in ("initial_pos.csv", "final_pos.csv"):
                shutil.copy2(src, _workspace_output_path(fname))
                copied_any = True
        return copied_any

    def _mark_dirty(self):
        self.project_dirty = True

    def _prompt_unsaved_changes(self, text, title="Unsaved Changes", default="save"):
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle(title)
        lines = str(text or "").split("\n", 1)
        box.setText(lines[0] if lines else "")
        if len(lines) > 1 and lines[1]:
            box.setInformativeText(lines[1])
        save_btn = box.addButton("Save", QMessageBox.AcceptRole)
        dont_save_btn = box.addButton("Don't Save", QMessageBox.DestructiveRole)
        cancel_btn = box.addButton("Cancel", QMessageBox.RejectRole)
        default_map = {
            "save": save_btn,
            "discard": dont_save_btn,
            "cancel": cancel_btn,
        }
        box.setDefaultButton(default_map.get(default, save_btn))
        box.exec()

        clicked = box.clickedButton()
        if clicked == save_btn:
            return "save"
        if clicked == dont_save_btn:
            return "discard"
        return "cancel"

    def _confirm_discard_changes(self):
        if not self.project_dirty:
            return True

        reply = self._prompt_unsaved_changes(
            "The project has unsaved changes.\nSave before continuing?"
        )

        if reply == "save":
            self.save_project()
            return not self.project_dirty
        elif reply == "discard":
            return True
        else:
            return False 
   
    # =========================
    # RECENT FILES
    # =========================
    def _add_recent_project(self, path):
        if path in self.recent_projects:
            self.recent_projects.remove(path)
        self.recent_projects.insert(0, path)
        self.recent_projects = self.recent_projects[:5]

        with open(self.recent_file_store, "w") as f:
            json.dump(self.recent_projects, f)

        self._update_recent_menu()

    def _update_recent_menu(self):
        self.recent_menu.clear()
        if os.path.exists(self.recent_file_store):
            with open(self.recent_file_store, "r") as f:
                self.recent_projects = json.load(f)

        for path in self.recent_projects:
            action = QAction(path, self)
            action.triggered.connect(lambda _, p=path: self._open_project_safe(p))
            self.recent_menu.addAction(action)

    def _load_last_project(self):
        if os.path.exists(self.recent_file_store):
            with open(self.recent_file_store, "r") as f:
                files = json.load(f)
                if files:
                    self._load_project_from_path(files[0])
        
    def change_module(self, module_name):
        """Updates view state based on selected module. UI is now persistent."""
        self.view.set_module(module_name)
        # The properties panel is now persistent. 
        # This method can be used to switch tabs in the properties panel later.

    def closeEvent(self, event):
        if self.project_dirty:
            # Capture the latest edits before asking whether to discard.
            self._autosave()
            reply = self._prompt_unsaved_changes(
                "The project has unsaved changes.\nSave before exiting?"
            )

            if reply == "save":
                self.save_project()
                if self.project_dirty:  # user cancelled save
                    event.ignore()
                    return
            elif reply == "cancel":
                event.ignore()
                return
        self._stop_background_threads()
        event.accept()

    def _snapshot_current_project(self):
        # Ensure transient/non-serializable solver settings are stripped before snapshotting.
        self._sync_solver_settings_to_project_state()
        state_snapshot = ProjectState.from_dict(self.project_state.to_dict()).to_dict()
        schema_state = self._build_schema_project_state()
        snapshot = {
            "units": "m",
            "sketches": copy.deepcopy(self.view.sketches),
            "sketch_meta": copy.deepcopy(self.view.sketch_meta),
            "dimensions": copy.deepcopy(self.view.dimensions),
            "constraints": copy.deepcopy(self.view.constraints),
            "show_dimensions": self.view.show_dimensions,
            "geometry": self.view.serialize_geometry(),
            "initial_velocities": copy.deepcopy(self.view.initial_velocities),
            "stage": self.active_stage.name,
            "mode": self.project_mode,
            "freeform_auto_convert_enabled": bool(
                getattr(self.view, "freeform_auto_convert_enabled", False)
            ),
            "modeling": {
                "extrude_height": float(self.view.extrude_height),
                "extrude_layers": int(self.view.extrude_layers),
            },
            "connection_settings": {
                "min_spacing_factor": float(getattr(self.view, "mesh_min_spacing_factor", 1.0)),
                "boundary_thickness": float(getattr(self.view, "mesh_boundary_thickness", 0.0)),
                "boundary_spacing_factor": float(getattr(self.view, "mesh_boundary_spacing_factor", 1.0)),
            },
            "project_state": state_snapshot,
        }
        merge_project_state_into_project_data(snapshot, schema_state)
        return snapshot


    def _restore_project_snapshot(self, data):
        self.view.clear_all()
        self._reset_stage_panels(clear_bc_lists=True)
        if isinstance(data, dict) and "project_state" in data:
            self.project_state = ProjectState.from_dict(data.get("project_state", {}))
        else:
            self.project_state = project_state_from_project_data(data)
        self.project_state.analysis_type = self._normalized_analysis_type(
            getattr(self.project_state, "analysis_type", data.get("analysis_type", "static"))
        )
        self.project_state.dimension = self._normalized_dimension(
            getattr(self.project_state, "dimension", data.get("dimension", "2D"))
        )
        self.view.project_state = self.project_state
        if hasattr(self, "properties_panel") and hasattr(self.properties_panel, "set_project_state"):
            self.properties_panel.set_project_state(self.project_state)
        if hasattr(self, "project_tree") and hasattr(self.project_tree, "set_sources"):
            self.project_tree.set_sources(self.view, self.project_state)
        if hasattr(self, "property_inspector") and hasattr(self.property_inspector, "set_project_state"):
            self.property_inspector.set_project_state(self.project_state)
        mode = data.get("mode", "2d")
        self.project_mode = "3d" if str(mode).lower().startswith("3") else "2d"
        self.view.set_project_mode(self.project_mode)
        self._sync_mode_ui()
        modeling = data.get("modeling", {})
        if "extrude_height" in modeling:
            self.view.extrude_height = float(modeling.get("extrude_height", self.view.extrude_height))
        if "extrude_layers" in modeling:
            self.view.extrude_layers = int(modeling.get("extrude_layers", self.view.extrude_layers))
        connection_settings = data.get("connection_settings")
        if connection_settings is None:
            connection_settings = data.get("mesh_settings", {})
        if connection_settings:
            self.view.set_mesh_generation_settings(
                min_spacing_factor=connection_settings.get("min_spacing_factor"),
                boundary_thickness=connection_settings.get("boundary_thickness"),
                boundary_spacing_factor=connection_settings.get("boundary_spacing_factor"),
            )
        self.view.set_unit("m")
        self.view.sketches = copy.deepcopy(data.get("sketches", []))
        self.view.sketch_meta = copy.deepcopy(data.get("sketch_meta", []))
        self.view.dimensions = copy.deepcopy(data.get("dimensions", []))
        self.view.constraints = copy.deepcopy(data.get("constraints", []))
        self.view.show_dimensions = bool(data.get("show_dimensions", True))
        if hasattr(self, "dim_action"):
            self.dim_action.setChecked(self.view.show_dimensions)
        self._set_precision_sketch_mode(bool(self.view.show_dimensions), announce=False)
        self.view.deserialize_geometry(data.get("geometry", {}))
        self.project_state.parts = copy.deepcopy(getattr(self.view, "parts", []))
        self.view._sync_all_sketch_meta()
        self.view._ensure_dimensions()
        self.view._recalc_dimension_counter()
        self.view.deserialize_materials(self.project_state.materials)
        if hasattr(self, "properties_panel") and hasattr(self.properties_panel, "materials_tab"):
            self.properties_panel.materials_tab.set_project_state(self.project_state)
        if hasattr(self, "project_tree"):
            self.project_tree.refresh_from_model()
        if hasattr(self, "property_inspector"):
            self.property_inspector.clear_selection()
        self.view.deserialize_interfaces(self.project_state.interfaces)
        self.view._emit_interfaces_changed()
        auto_convert_enabled = bool(data.get("freeform_auto_convert_enabled", True))
        if hasattr(self.view, "set_freeform_auto_convert"):
            self.view.set_freeform_auto_convert(auto_convert_enabled, announce=False)
        else:
            self.view.freeform_auto_convert_enabled = auto_convert_enabled
        action = getattr(self, "freeform_autoconvert_action", None)
        if action is not None:
            action.blockSignals(True)
            action.setChecked(auto_convert_enabled)
            action.blockSignals(False)
        self.view.bcs = copy.deepcopy(self.project_state.boundary_conditions)
        self.view.loads = copy.deepcopy(self.project_state.loads)
        if hasattr(self.view, "_sanitize_bc_load_entries"):
            self.view._sanitize_bc_load_entries()
        self.view.initial_velocities = data.get("initial_velocities", [])
        self.view.bcsChanged.emit()
        self.view.loadsChanged.emit()
        self._apply_solver_settings_to_ui()
        self._warn_project_state_validation("restore")
        self._refresh_workflow_architecture()
        self.view.rebuild_display_geometry()
        self.view.redraw()
        desired_stage = ProjectStage[data.get("stage", "GEOMETRY")]
        has_materials = bool(getattr(self.project_state, "materials", {}))
        has_material_assignments = any(
            getattr(part, "material_id", None) not in (None, "")
            for part in getattr(self.project_state, "parts", [])
            if not getattr(part, "is_void", False)
        )
        if has_materials or has_material_assignments:
            desired_stage = max(desired_stage, ProjectStage.MATERIALS, key=self._workflow_stage_rank)
        if self._workflow_has_fluid_stage():
            desired_stage = max(desired_stage, ProjectStage.FLUID, key=self._workflow_stage_rank)
        if getattr(self.project_state, "interfaces", None):
            desired_stage = max(desired_stage, ProjectStage.INTERFACES, key=self._workflow_stage_rank)
        if self.view.loads:
            desired_stage = max(desired_stage, ProjectStage.LOADS, key=self._workflow_stage_rank)
        elif self.view.bcs:
            desired_stage = max(desired_stage, ProjectStage.BCS, key=self._workflow_stage_rank)
        if self._workflow_has_fracture_stage():
            desired_stage = max(desired_stage, ProjectStage.FRACTURE, key=self._workflow_stage_rank)
        self.apply_stage_ui(desired_stage)
        if desired_stage == ProjectStage.MATERIALS and self.properties_panel.tabs.isTabEnabled(1):
            self.properties_panel.tabs.setCurrentWidget(self.properties_panel.materials_tab)
        elif self.view.loads and self.properties_panel.tabs.isTabEnabled(3):
            self.properties_panel.tabs.setCurrentWidget(self.properties_panel.bcs_tab)
        elif self.view.bcs and self.properties_panel.tabs.isTabEnabled(3):
            self.properties_panel.tabs.setCurrentWidget(self.properties_panel.bcs_tab)
        project_name = f"Session_{len(self.session_projects) + 1}"
        self.project_tree.add_session_project(project_name)


    def _open_project_safe(self, path):
            reply = self._prompt_unsaved_changes(
                "Current project has unsaved changes.\nOpen another project without saving?",
                title="Unsaved Project",
            )

            if reply == "save":
                self.save_project()
                if self.project_dirty:
                    return
            elif reply == "discard":
                name = os.path.basename(self.current_project_file) if self.current_project_file else f"Unsaved_{len(self.session_projects)+1}"
                self.session_projects[name] = self._snapshot_current_project()
            else:
                return
            self._load_project_from_path(path)

    def advance_stage(self, stage):
        if stage == ProjectStage.BCS:
            if not self.material_controller.validate_before_bc_stage():
                return

        if stage == ProjectStage.JOB:
            if not self.particle_controller.validate_before_solve():
                return
            if not self.bc_controller.validate_before_solve():
                return

        if not isinstance(stage, ProjectStage):
            return
        if self._workflow_stage_rank(stage) <= self._workflow_stage_rank(self.active_stage):
            return

        self.apply_stage_ui(stage)


if __name__ == "__main__":
    import argparse
    import sys

    from PySide6.QtWidgets import QApplication

    configure_qt_runtime()

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--validate-ui", action="store_true")
    args, remaining = parser.parse_known_args()
    validate_ui_mode = args.validate_ui or os.environ.get("CPD_UI_VALIDATE") == "1"
    sys.argv = [sys.argv[0]] + remaining

    os.chdir(_project_root_dir())
    log_path = configure_app_logging()
    # Enable Python's native crash handler so segfaults from C extensions
    # (gmsh, Qt) leave a stack trace in workspace/logs/fault.log instead of
    # vanishing silently. The file is opened in append mode so multiple
    # crashes accumulate.
    try:
        import faulthandler
        fault_log_path = log_path.parent / "fault.log"
        _fault_log_file = open(str(fault_log_path), "a", encoding="utf-8")
        _fault_log_file.write(f"\n=== Session started at {time.ctime()} ===\n")
        _fault_log_file.flush()
        faulthandler.enable(file=_fault_log_file, all_threads=True)
    except Exception as _fault_exc:
        print(f"faulthandler unavailable: {_fault_exc}")
    logger = logging.getLogger(__name__)
    logger.info("Starting CPD SimStudio")
    logger.info("Workspace directory: %s", _workspace_dir())
    logger.info("App log file: %s", log_path)
    app = QApplication(sys.argv)
    # Let Ctrl+C from a terminal request a normal Qt shutdown.
    signal.signal(signal.SIGINT, lambda *_args: app.quit())
    _sigint_pump = QTimer()
    _sigint_pump.setInterval(200)
    _sigint_pump.timeout.connect(lambda: None)
    _sigint_pump.start()
    win = Main()
    if validate_ui_mode:
        win.enable_validation_mode()
    else:
        win.schedule_startup_update_check()
    win.show()
    logger.info(
        "Startup window shown (qt_platform=%s xdg_session=%s wayland_display=%s display=%s)",
        os.environ.get("QT_QPA_PLATFORM", ""),
        os.environ.get("XDG_SESSION_TYPE", ""),
        os.environ.get("WAYLAND_DISPLAY", ""),
        os.environ.get("DISPLAY", ""),
    )
    win._schedule_startup_window_state(reason="startup-main", delay_ms=150)
    exit_code = app.exec()
    _sigint_pump.stop()
    sys.exit(exit_code)

    
