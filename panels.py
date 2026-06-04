"""UI panels for CPD SimStudio."""

import copy
import os
import re
import signal
import subprocess
import sys
import time
import threading
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except Exception:
    try:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    except Exception:
        FigureCanvas = None

try:
    from matplotlib.figure import Figure
except Exception:
    Figure = None

from PySide6.QtCore import Qt, QObject, QThread, QTimer, Signal, QMimeData, Slot, QSize, QEasingCurve, QPropertyAnimation, QSettings, QPoint
from PySide6.QtGui import QDrag, QFont, QColor, QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QFrame,
    QFileDialog,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenu,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QRadioButton,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QSplitter,
    QStackedWidget,
    QSpinBox,
    QSlider,
    QStyle,
    QToolButton,
    QButtonGroup,
    QTableWidget,
    QTableWidgetItem,
    QAbstractSpinBox,
    QGraphicsOpacityEffect,
    QScrollArea,
    QSizePolicy,
)

from shapely.geometry import Point

import materials_db
from mesh_utils import map_geometry_to_nodes

from models import (
    BoundaryLayerSeed,
    EdgeSeed,
    FIELD_DISTRIBUTION_PROPERTY_KEYS,
    Interface,
    Material,
    VertexSeed,
    normalize_heterogeneity_config,
    normalize_material_field_config,
)
from material_registry import (
    behavior_label,
    damage_label,
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
from project_state import ProjectState
from project_stages import ProjectStage
from ui_icons import ICON_SIZE, get_icon
from solver_backends import get_default_backend
from app_config import (
    DEFAULT_DX,
    WORKSPACE_DIR_NAME,
    get_project_root,
    get_workspace_dir,
    get_workspace_path,
)
from app_config import (
    FAST_PREVIEW_CONNECTION_LIMIT,
    FAST_PREVIEW_ENABLED,
    PREVIEW_CONNECTION_LIMIT,
    GPU_POINT_PREVIEW_ENABLED,
    GPU_POINT_PREVIEW_AUTO_ENABLED,
    GPU_POINT_PREVIEW_AUTO_THRESHOLD,
)
from app_config import SNAP_TOL
from controllers.commands import (
    AddBoundaryConditionCommand,
    AddMaterialCommand,
    GenerateParticlesCommand,
    RunSolverCommand,
)
from ui_numeric import ScientificDoubleSpinBox as QDoubleSpinBox
from ui_numeric import ScientificSpinBox as QSpinBox


def _project_root_dir():
    return str(get_project_root())


def _workspace_dir():
    return str(get_workspace_dir())


def _workspace_path(*parts):
    return str(get_workspace_path(*parts))


def _workspace_output_path(*parts):
    return _workspace_path("output", *parts)


def _style_icon(widget, style_name, fallback_name=None, theme_names=None):
    """Prefer native platform icons, fall back to app icons when unavailable."""
    if theme_names:
        names = theme_names if isinstance(theme_names, (list, tuple)) else [theme_names]
        for name in names:
            try:
                themed = QIcon.fromTheme(str(name))
            except Exception:
                themed = QIcon()
            if not themed.isNull():
                return themed
    enum_val = getattr(QStyle, str(style_name), None)
    if enum_val is not None:
        try:
            icon = widget.style().standardIcon(enum_val)
            if not icon.isNull():
                return icon
        except Exception:
            pass
    if fallback_name:
        return get_icon(fallback_name)
    return QIcon()


def _resolve_project_state(sketch_view, project_state=None):
    state = project_state
    if state is None and sketch_view is not None:
        state = getattr(sketch_view, "project_state", None)
    if state is None:
        state = ProjectState()
        if sketch_view is not None:
            try:
                sketch_view.project_state = state
            except Exception:
                pass
    return state


def _iface_get(iface, key, default=None):
    if hasattr(iface, key):
        return getattr(iface, key, default)
    if isinstance(iface, dict):
        if key in iface:
            return iface.get(key, default)
        alias_map = {
            "interface_type": "type",
            "friction_coeff": "friction",
        }
        alias = alias_map.get(key)
        if alias is not None:
            return iface.get(alias, default)
    return default


def _iface_set(iface, key, value):
    if isinstance(iface, dict):
        alias_map = {
            "interface_type": "type",
            "friction_coeff": "friction",
        }
        iface[alias_map.get(key, key)] = value
        return
    setattr(iface, key, value)


DOCK_MARGIN = 2
DOCK_SECTION_SPACING = 2
DOCK_ROW_SPACING = 2
DOCK_ICON_BTN_MIN = 24


def _apply_layout_metrics(layout, margins=(DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN), spacing=DOCK_SECTION_SPACING):
    if layout is None:
        return
    try:
        layout.setContentsMargins(*margins)
    except Exception:
        pass
    try:
        layout.setSpacing(spacing)
    except Exception:
        pass


def _make_dock_separator():
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Plain)
    line.setObjectName("DockSectionSeparator")
    return line


def _make_cross_tab_notice_label(parent=None):
    label = QLabel("", parent)
    label.setObjectName("CrossTabNotice")
    label.setWordWrap(True)
    label.setVisible(False)
    label.setStyleSheet(
        "QLabel#CrossTabNotice {"
        " background-color: #fef3c7;"
        " color: #92400e;"
        " border: 1px solid #fde68a;"
        " border-radius: 4px;"
        " padding: 6px 8px;"
        " font-size: 11px;"
        "}"
    )
    return label


def _show_panel_notice(label, message, *, duration_ms=4500):
    if label is None:
        return
    label.setText(str(message))
    label.setVisible(True)
    timer = getattr(label, "_notice_clear_timer", None)
    if timer is not None:
        try:
            timer.stop()
        except Exception:
            pass
    timer = QTimer(label)
    timer.setSingleShot(True)

    def _clear():
        label.setText("")
        label.setVisible(False)

    timer.timeout.connect(_clear)
    timer.start(int(max(500, duration_ms)))
    label._notice_clear_timer = timer


_COLLAPSIBLE_SETTINGS_KEY = "ui/collapsibles"


def _persist_collapsible(group_box, settings_key, body_widget=None, *, default_expanded=False):
    """Wire a checkable QGroupBox so its expand/collapse state persists across sessions.

    Reads the saved state from QSettings and applies it. Then connects toggled to
    keep saving. If body_widget is given, also keeps its visibility in sync.
    """
    if group_box is None:
        return
    try:
        settings = QSettings("CPD-Modeller", "CPD-SimStudio")
        raw = settings.value(f"{_COLLAPSIBLE_SETTINGS_KEY}/{settings_key}", default_expanded, type=bool)
        expanded = bool(raw)
    except Exception:
        expanded = bool(default_expanded)
    group_box.blockSignals(True)
    group_box.setChecked(expanded)
    group_box.blockSignals(False)
    if body_widget is not None:
        body_widget.setVisible(expanded)

    def _on_toggled(checked):
        if body_widget is not None:
            body_widget.setVisible(bool(checked))
        try:
            QSettings("CPD-Modeller", "CPD-SimStudio").setValue(
                f"{_COLLAPSIBLE_SETTINGS_KEY}/{settings_key}", bool(checked)
            )
        except Exception:
            pass

    group_box.toggled.connect(_on_toggled)


def _make_empty_state_label(message, parent=None):
    label = QLabel(str(message), parent)
    label.setObjectName("EmptyStateLabel")
    label.setWordWrap(True)
    label.setAlignment(Qt.AlignCenter)
    # setWordWrap(True) alone does not shrink the label's minimumSizeHint().
    # Combine with Ignored horizontal size policy + zero min width so the
    # layout can compress it to the panel's actual width and let wordWrap
    # take over for the height.
    label.setMinimumWidth(0)
    label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    label.setStyleSheet(
        "QLabel#EmptyStateLabel {"
        " color: #8a93a3;"
        " font-style: italic;"
        " padding: 10px 8px;"
        "}"
    )
    return label


def _bind_tree_empty_state(tree_widget, empty_label):
    """Show empty_label whenever tree_widget has no top-level items."""
    if tree_widget is None or empty_label is None:
        return

    def _refresh():
        try:
            count = tree_widget.topLevelItemCount()
        except Exception:
            count = 0
        empty_label.setVisible(count == 0)

    model = tree_widget.model()
    if model is not None:
        model.rowsInserted.connect(lambda *_: _refresh())
        model.rowsRemoved.connect(lambda *_: _refresh())
        model.modelReset.connect(_refresh)
    _refresh()


def _layout_container(layout):
    widget = QWidget()
    widget.setLayout(layout)
    return widget


def _reflow_button_grid(
    layout,
    widgets,
    width,
    *,
    min_button_width=136,
    max_columns=None,
    stretch_columns=True,
):
    if layout is None:
        return
    while layout.count():
        item = layout.takeAt(0)
        child_layout = item.layout()
        if child_layout is not None:
            child_layout.setParent(None)
    items = [widget for widget in widgets if widget is not None]
    if not items:
        return
    width = max(1, int(width or 0))
    computed_columns = max(1, width // max(1, int(min_button_width)))
    if max_columns is not None:
        computed_columns = min(int(max_columns), computed_columns)
    columns = max(1, min(len(items), computed_columns))
    for column in range(max(len(items) + 1, int(max_columns or 0) + 1, 8)):
        layout.setColumnStretch(column, 0)
    for row in range(max(len(items) + 1, 8)):
        layout.setRowStretch(row, 0)
    for index, widget in enumerate(items):
        row = index // columns
        column = index % columns
        layout.addWidget(widget, row, column)
    for column in range(columns):
        layout.setColumnStretch(column, 1 if stretch_columns else 0)
    if not stretch_columns:
        layout.setColumnStretch(columns, 1)
    layout.setRowStretch((len(items) + columns - 1) // columns, 1)


def _configure_dock_button(button, *, expanding=True):
    if button is None:
        return
    target_h = 28
    current_min_h = int(button.minimumHeight() or 0)
    if current_min_h <= 0 or current_min_h > target_h:
        button.setMinimumHeight(target_h)
    if isinstance(button, QToolButton):
        current_min_w = int(button.minimumWidth() or 0)
        if current_min_w <= 0 or current_min_w > 28:
            button.setMinimumWidth(28)
    else:
        button.setMinimumWidth(0)
    button.setMaximumHeight(target_h + 2)
    button.setSizePolicy(
        QSizePolicy.Expanding if expanding else QSizePolicy.Fixed,
        QSizePolicy.Fixed,
    )


def _set_responsive_button_text(button, *, full, compact=None, icon_only=False):
    if button is None:
        return
    has_icon = False
    try:
        has_icon = not button.icon().isNull()
    except Exception:
        has_icon = False
    if icon_only and has_icon:
        button.setText("")
        return
    if compact is not None:
        button.setText(str(compact))
        return
    button.setText(str(full))


def _configure_dock_label(label, *, wrap=False):
    if label is None:
        return
    label.setWordWrap(bool(wrap))
    label.setMinimumHeight(max(20, label.minimumHeight()))
    label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)


def _configure_dock_table(table):
    if table is None:
        return
    header = table.horizontalHeader()
    header.setCascadingSectionResizes(True)
    header.setMinimumSectionSize(24)
    table.verticalHeader().setDefaultSectionSize(26)
    table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    # Keep the floor low so other widgets in a tight dock can remain visible.
    table.setMinimumHeight(max(70, table.minimumHeight()))
    table.setMinimumWidth(0)
    # No horizontal scrollbar — content gets clipped instead, keeping the
    # right dock panel free of horizontal scroll regardless of column widths.
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)


def _configure_dock_tree(tree):
    if tree is None:
        return
    header = tree.header()
    if header is not None:
        header.setCascadingSectionResizes(True)
        header.setMinimumSectionSize(24)
    tree.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    # Keep the floor low so other widgets below the tree (Assign section,
    # action buttons, etc.) stay visible when the dock is short.
    tree.setMinimumHeight(max(70, tree.minimumHeight()))
    tree.setMinimumWidth(0)
    # No horizontal scrollbar — content gets clipped instead, keeping the
    # right dock panel free of horizontal scroll regardless of column widths.
    tree.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)


def _wrap_in_dock_scroll(widget, parent=None):
    page = QWidget(parent)
    page_layout = QVBoxLayout(page)
    page_layout.setContentsMargins(0, 0, 0, 0)
    page_layout.setSpacing(0)
    scroll = QScrollArea(page)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    # Never show horizontal scrollbars at the dock-panel level — if content
    # is slightly wider than the viewport it's clipped instead of forcing a
    # scrollbar that competes with the dedicated tree/list scrollbars inside.
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    # In addition to hiding the H-scrollbar, force its range/value to zero so
    # any wheel-event or programmatic pan can't shift the content sideways.
    hbar = scroll.horizontalScrollBar()
    hbar.setRange(0, 0)
    hbar.valueChanged.connect(lambda _v, b=hbar: b.setValue(0))
    widget.setParent(scroll)
    widget.setMinimumWidth(0)
    widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
    scroll.setWidget(widget)
    page_layout.addWidget(scroll)
    return page


# ---------------------------------------------------------------------------
# Shape Template generators (used by Geometry > Tools tab "Quick Templates")
# Each function returns a closed list of (x, y) points centered/oriented in a
# sensible default position; the user picks the dimensions via a small dialog.
# ---------------------------------------------------------------------------

def _gen_template_rectangle(width, height):
    w, h = width / 2.0, height / 2.0
    return [(-w, -h), (w, -h), (w, h), (-w, h), (-w, -h)]


def _gen_template_i_beam(width, height, web_t, flange_t):
    hw, hh, hwt = width / 2.0, height / 2.0, web_t / 2.0
    ft = flange_t
    return [
        (-hw,  hh), ( hw,  hh), ( hw,  hh - ft), ( hwt,  hh - ft),
        ( hwt, -hh + ft), ( hw, -hh + ft), ( hw, -hh), (-hw, -hh),
        (-hw, -hh + ft), (-hwt, -hh + ft), (-hwt,  hh - ft), (-hw,  hh - ft),
        (-hw,  hh),
    ]


def _gen_template_t_section(width, height, web_t, flange_t):
    hw, hh, hwt = width / 2.0, height / 2.0, web_t / 2.0
    ft = flange_t
    return [
        (-hw,  hh), ( hw,  hh), ( hw,  hh - ft), ( hwt,  hh - ft),
        ( hwt, -hh), (-hwt, -hh), (-hwt,  hh - ft), (-hw,  hh - ft),
        (-hw,  hh),
    ]


def _gen_template_l_bracket(width, height, thickness):
    t = thickness
    return [(0, 0), (width, 0), (width, t), (t, t), (t, height), (0, height), (0, 0)]


def _gen_template_c_channel(width, height, web_t, flange_t):
    hh = height / 2.0
    return [
        (0,  hh), (width,  hh), (width,  hh - flange_t), (web_t,  hh - flange_t),
        (web_t, -hh + flange_t), (width, -hh + flange_t), (width, -hh), (0, -hh),
        (0,  hh),
    ]


def _gen_template_disc(diameter, segments=64):
    r = diameter / 2.0
    return [
        (r * math.cos(2.0 * math.pi * i / segments),
         r * math.sin(2.0 * math.pi * i / segments))
        for i in range(segments + 1)
    ]


def _gen_template_hexagon(side):
    return [
        (side * math.cos(math.pi * i / 3.0),
         side * math.sin(math.pi * i / 3.0))
        for i in range(7)
    ]


def _gen_template_triangle(base, height):
    return [(-base / 2.0, 0.0), (base / 2.0, 0.0), (0.0, height), (-base / 2.0, 0.0)]


def _gen_template_z_section(width, height, web_t, flange_t):
    """Z-section: top flange goes right of web, bottom flange goes left."""
    hh = height / 2.0
    hwt = web_t / 2.0
    return [
        (-hwt,        hh),
        ( width - hwt, hh),
        ( width - hwt, hh - flange_t),
        ( hwt,        hh - flange_t),
        ( hwt,       -hh + flange_t),
        ( hwt - width, -hh + flange_t),
        ( hwt - width, -hh),
        (-hwt,       -hh),
        (-hwt,        hh),
    ]


def _gen_template_cruciform(width, height, arm_t):
    """Plus/cruciform: vertical + horizontal arms cross at the centre."""
    hw, hh = width / 2.0, height / 2.0
    ht = arm_t / 2.0
    return [
        (-ht,  hh), ( ht,  hh), ( ht,  ht), ( hw,  ht),
        ( hw, -ht), ( ht, -ht), ( ht, -hh), (-ht, -hh),
        (-ht, -ht), (-hw, -ht), (-hw,  ht), (-ht,  ht),
        (-ht,  hh),
    ]


def _gen_template_trapezoid(bottom_width, top_width, height):
    """Symmetric trapezoid sitting on its bottom edge."""
    bw, tw = bottom_width / 2.0, top_width / 2.0
    return [(-bw, 0.0), (bw, 0.0), (tw, height), (-tw, height), (-bw, 0.0)]


def _gen_template_ellipse(width, height, segments=64):
    a, b = width / 2.0, height / 2.0
    return [
        (a * math.cos(2.0 * math.pi * i / segments),
         b * math.sin(2.0 * math.pi * i / segments))
        for i in range(segments + 1)
    ]


# Hollow templates return (outer_points, hole_points). The caller adds the
# outer as a solid Part and the inner as a void child Part.

def _gen_template_pipe(outer_d, inner_d, segments=64):
    if inner_d >= outer_d:
        inner_d = outer_d * 0.5
    return (_gen_template_disc(outer_d, segments),
            _gen_template_disc(inner_d, segments))


def _gen_template_hollow_rectangle(outer_w, outer_h, wall_t):
    inner_w = max(outer_w - 2.0 * wall_t, 1e-6)
    inner_h = max(outer_h - 2.0 * wall_t, 1e-6)
    return (_gen_template_rectangle(outer_w, outer_h),
            _gen_template_rectangle(inner_w, inner_h))


def _gen_template_plate_with_hole(width, height, hole_d, segments=64):
    if hole_d >= min(width, height):
        hole_d = min(width, height) * 0.4
    return (_gen_template_rectangle(width, height),
            _gen_template_disc(hole_d, segments))


def _gen_template_dogbone(total_length, gauge_length, grip_width, gauge_width, segments=24):
    """ASTM E8-style dog-bone tensile test specimen. Symmetric about both axes
    with cosine-smoothed transitions between the wide grip ends and the narrow
    central gauge section."""
    L = total_length / 2.0
    Lg = gauge_length / 2.0
    Wg = grip_width / 2.0
    Wn = gauge_width / 2.0
    if L <= Lg or Wg <= Wn:
        # Degenerate — fall back to a plain rectangle.
        return _gen_template_rectangle(total_length, grip_width)
    transition_dx = L - Lg
    pts = []
    # Trace CCW from the top-left grip corner.
    pts.append((-L, Wg))
    # Top-left transition: (-L, Wg) → (-Lg, Wn) narrowing
    for i in range(1, segments + 1):
        t = i / segments
        s = (1.0 - math.cos(math.pi * t)) / 2.0
        x = -L + t * transition_dx
        y = Wg + s * (Wn - Wg)
        pts.append((x, y))
    # Gauge top edge (straight)
    pts.append((Lg, Wn))
    # Top-right transition: (Lg, Wn) → (L, Wg) widening
    for i in range(1, segments + 1):
        t = i / segments
        s = (1.0 - math.cos(math.pi * t)) / 2.0
        x = Lg + t * transition_dx
        y = Wn + s * (Wg - Wn)
        pts.append((x, y))
    # Right grip edge (straight down)
    pts.append((L, -Wg))
    # Bottom-right transition: (L, -Wg) → (Lg, -Wn) narrowing
    for i in range(1, segments + 1):
        t = i / segments
        s = (1.0 - math.cos(math.pi * t)) / 2.0
        x = L - t * transition_dx
        y = -Wg + s * (Wg - Wn)
        pts.append((x, y))
    # Gauge bottom edge (straight)
    pts.append((-Lg, -Wn))
    # Bottom-left transition: (-Lg, -Wn) → (-L, -Wg) widening
    for i in range(1, segments + 1):
        t = i / segments
        s = (1.0 - math.cos(math.pi * t)) / 2.0
        x = -Lg - t * transition_dx
        y = -Wn - s * (Wg - Wn)
        pts.append((x, y))
    # Close
    pts.append((-L, Wg))
    return pts


# Spec for each template: name, symbol shown on button, list of (label,
# default_value, min, max) for parameter dialog, and generator function.
_SHAPE_TEMPLATE_SPECS = {
    "rectangle": {
        "name": "Rectangle",
        "symbol": "▭",
        "params": [("Width", 50.0, 0.01, 1e5), ("Height", 25.0, 0.01, 1e5)],
        "generator": _gen_template_rectangle,
    },
    "i_beam": {
        "name": "I-Beam",
        "symbol": "I",
        "params": [
            ("Flange width", 100.0, 0.01, 1e5),
            ("Height", 200.0, 0.01, 1e5),
            ("Web thickness", 6.0, 0.01, 1e4),
            ("Flange thickness", 10.0, 0.01, 1e4),
        ],
        "generator": _gen_template_i_beam,
    },
    "t_section": {
        "name": "T-Section",
        "symbol": "T",
        "params": [
            ("Flange width", 100.0, 0.01, 1e5),
            ("Height", 100.0, 0.01, 1e5),
            ("Web thickness", 8.0, 0.01, 1e4),
            ("Flange thickness", 12.0, 0.01, 1e4),
        ],
        "generator": _gen_template_t_section,
    },
    "l_bracket": {
        "name": "L-Bracket",
        "symbol": "L",
        "params": [
            ("Width", 60.0, 0.01, 1e5),
            ("Height", 60.0, 0.01, 1e5),
            ("Thickness", 6.0, 0.01, 1e4),
        ],
        "generator": _gen_template_l_bracket,
    },
    "c_channel": {
        "name": "C-Channel",
        "symbol": "C",
        "params": [
            ("Width", 50.0, 0.01, 1e5),
            ("Height", 100.0, 0.01, 1e5),
            ("Web thickness", 6.0, 0.01, 1e4),
            ("Flange thickness", 10.0, 0.01, 1e4),
        ],
        "generator": _gen_template_c_channel,
    },
    "disc": {
        "name": "Disc",
        "symbol": "●",
        "params": [("Diameter", 50.0, 0.01, 1e5)],
        "generator": _gen_template_disc,
    },
    "hexagon": {
        "name": "Hexagon",
        "symbol": "⬢",
        "params": [("Side length", 25.0, 0.01, 1e5)],
        "generator": _gen_template_hexagon,
    },
    "triangle": {
        "name": "Triangle",
        "symbol": "▲",
        "params": [("Base", 50.0, 0.01, 1e5), ("Height", 40.0, 0.01, 1e5)],
        "generator": _gen_template_triangle,
    },
    "z_section": {
        "name": "Z-Section",
        "symbol": "Z",
        "params": [
            ("Flange width", 60.0, 0.01, 1e5),
            ("Height", 200.0, 0.01, 1e5),
            ("Web thickness", 6.0, 0.01, 1e4),
            ("Flange thickness", 10.0, 0.01, 1e4),
        ],
        "generator": _gen_template_z_section,
    },
    "cruciform": {
        "name": "Cruciform",
        "symbol": "✚",
        "params": [
            ("Width", 100.0, 0.01, 1e5),
            ("Height", 100.0, 0.01, 1e5),
            ("Arm thickness", 20.0, 0.01, 1e4),
        ],
        "generator": _gen_template_cruciform,
    },
    "trapezoid": {
        "name": "Trapezoid",
        "symbol": "⏢",
        "params": [
            ("Bottom width", 80.0, 0.01, 1e5),
            ("Top width", 40.0, 0.01, 1e5),
            ("Height", 50.0, 0.01, 1e5),
        ],
        "generator": _gen_template_trapezoid,
    },
    "ellipse": {
        "name": "Ellipse",
        "symbol": "⬭",
        "params": [
            ("Width", 80.0, 0.01, 1e5),
            ("Height", 40.0, 0.01, 1e5),
        ],
        "generator": _gen_template_ellipse,
    },
    "pipe": {
        "name": "Pipe",
        "symbol": "◎",
        "params": [
            ("Outer diameter", 60.0, 0.01, 1e5),
            ("Inner diameter", 40.0, 0.01, 1e5),
        ],
        "generator": _gen_template_pipe,
        "hollow": True,
    },
    "hollow_rect": {
        "name": "Hollow Rectangle",
        "symbol": "⊟",
        "params": [
            ("Outer width", 80.0, 0.01, 1e5),
            ("Outer height", 50.0, 0.01, 1e5),
            ("Wall thickness", 5.0, 0.01, 1e4),
        ],
        "generator": _gen_template_hollow_rectangle,
        "hollow": True,
    },
    "plate_hole": {
        "name": "Plate with Hole",
        "symbol": "⊡",
        "params": [
            ("Width", 100.0, 0.01, 1e5),
            ("Height", 60.0, 0.01, 1e5),
            ("Hole diameter", 20.0, 0.01, 1e5),
        ],
        "generator": _gen_template_plate_with_hole,
        "hollow": True,
    },
    "dogbone": {
        "name": "Dog-bone Specimen",
        "symbol": "D",
        "params": [
            ("Total length", 200.0, 0.01, 1e5),
            ("Gauge length", 80.0, 0.01, 1e5),
            ("Grip width", 50.0, 0.01, 1e5),
            ("Gauge width", 20.0, 0.01, 1e5),
        ],
        "generator": _gen_template_dogbone,
    },
}


def _finalize_dock_panel(widget):
    if widget is None:
        return
    for label in widget.findChildren(QLabel):
        # Labels marked with these object names need wordWrap=True because
        # their text can be longer than the narrow right-dock panel and we
        # want them to wrap to a new line instead of forcing the panel wider
        # or clipping horizontally.
        allow_wrap = label.objectName() in ("MinorStatusLabel", "EmptyStateLabel", "CrossTabNotice")
        _configure_dock_label(label, wrap=allow_wrap)
        if allow_wrap:
            label.setMinimumWidth(0)
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    for button in widget.findChildren(QPushButton):
        _configure_dock_button(button)
    for button in widget.findChildren(QToolButton):
        _configure_dock_button(button)
    for table in widget.findChildren(QTableWidget):
        _configure_dock_table(table)
    for tree in widget.findChildren(QTreeWidget):
        _configure_dock_tree(tree)
    # QListWidget instances — kill their horizontal scrollbars too, so the
    # right dock panel never gets a horizontal scrollbar from a list overflowing.
    for list_widget in widget.findChildren(QListWidget):
        list_widget.setMinimumWidth(0)
        list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    # Inner QScrollArea instances inside any panel — disable horizontal scroll
    # AND keep the horizontal scrollbar's range/value at 0 even when Qt
    # recomputes it on resize. valueChanged + rangeChanged keep it pinned.
    def _pin_hbar_to_zero(bar):
        bar.setRange(0, 0)
        bar.setValue(0)
    for sa in widget.findChildren(QScrollArea):
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        hbar = sa.horizontalScrollBar()
        _pin_hbar_to_zero(hbar)
        try:
            hbar.valueChanged.connect(lambda _v, b=hbar: _pin_hbar_to_zero(b))
            hbar.rangeChanged.connect(lambda _lo, _hi, b=hbar: _pin_hbar_to_zero(b))
        except Exception:
            pass
    field_types = (QComboBox, QLineEdit, QDoubleSpinBox, QSpinBox)
    seen_fields = set()
    for field_type in field_types:
        for field in widget.findChildren(field_type):
            field_id = id(field)
            if field_id in seen_fields:
                continue
            seen_fields.add(field_id)
            field.setMinimumHeight(26)
            field.setMaximumHeight(30)
            field.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    # Combos can have very long item text — cap their minimum so they don't
    # force the panel wider than the dock when in narrow mode.
    for combo in widget.findChildren(QComboBox):
        if combo.sizeAdjustPolicy() == QComboBox.AdjustToContents:
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(min(combo.minimumContentsLength() or 8, 8))
        combo.setMinimumWidth(60)
    for text_edit in widget.findChildren(QTextEdit):
        text_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        text_edit.setMinimumHeight(max(140, text_edit.minimumHeight()))


def _configure_expression_combo(combo, value):
    if combo is None:
        return combo
    combo.setEditable(True)
    combo.setInsertPolicy(QComboBox.NoInsert)
    combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(16)
    combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    combo.setEditText(str(value))
    combo.setToolTip(
        "Expression in time t. Examples: 0, 10*t, t**2, sin(2*pi*t), 0.05*sin(20*t)."
    )
    editor = combo.lineEdit()
    if editor is not None:
        editor.setClearButtonEnabled(True)
        editor.setPlaceholderText("Enter expression, e.g. 0.05*sin(20*t)")
        editor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    return combo


def _seed_tick_params(seed_dict, edge_length):
    """Return a list of parametric positions t in [0, 1] for tick marks along
    a single edge, based on the edge-seed config. Each t maps to a point at
    edge_start + t * (edge_end - edge_start).

    Handles every Method × Bias combination that LocalSeedsDialog can produce.
    For Double bias the spacings are mirrored: fine at both ends, coarse in the
    middle. For Single bias the spacings grow geometrically; flip_bias swaps
    which end is fine.
    """
    method = str(seed_dict.get("method", "by_size"))
    bias = str(seed_dict.get("bias", "none"))
    flip = bool(seed_dict.get("flip_bias", False))
    if edge_length <= 0:
        return [0.0, 1.0]
    if method == "by_number":
        N = max(1, int(seed_dict.get("seed_count", 0) or 0))
        if N <= 1:
            return [0.0, 1.0]
        if bias == "none":
            return [i / N for i in range(N + 1)]
        r = max(1.0, float(seed_dict.get("bias_ratio", 1.0) or 1.0))
        per_r = r ** (1.0 / max(N - 1, 1))
        if bias == "single":
            spacings = [per_r ** i for i in range(N)]
        else:  # double
            half = N // 2
            left = [per_r ** i for i in range(half)]
            right = list(reversed(left))
            spacings = left + right
            while len(spacings) < N:
                spacings.append(left[-1] if left else 1.0)
        if flip and bias == "single":
            spacings.reverse()
        total = sum(spacings) or 1.0
        ts = [0.0]
        acc = 0.0
        for s in spacings:
            acc += s / total
            ts.append(min(1.0, acc))
        return ts
    # method == "by_size"
    if bias == "none":
        h = float(seed_dict.get("element_size", 0.0) or 0.0)
        if h <= 0:
            return [0.0, 1.0]
        N = max(1, int(round(edge_length / h)))
        if N > 200:  # cap for visualization sanity
            N = 200
        return [i / N for i in range(N + 1)]
    # bias == single or double, by_size — solve for N s.t. geometric series
    # h_min, h_min*r, ..., h_min*r^(N-1) sums to ~edge_length, with r derived
    # from h_max/h_min.
    h_min = float(seed_dict.get("min_size", 0.0) or 0.0)
    h_max = float(seed_dict.get("max_size", 0.0) or 0.0)
    if h_min <= 0 or h_max <= 0:
        return [0.0, 1.0]
    if h_min > h_max:
        h_min, h_max = h_max, h_min
    avg = 0.5 * (h_min + h_max)
    N = max(2, int(round(edge_length / avg)))
    if N > 200:
        N = 200
    r_total = h_max / h_min
    per_r = r_total ** (1.0 / max(N - 1, 1)) if N > 1 else 1.0
    if bias == "single":
        spacings = [h_min * (per_r ** i) for i in range(N)]
        if flip:
            spacings.reverse()
    else:  # double
        half = N // 2
        left = [h_min * (per_r ** i) for i in range(half)]
        right = list(reversed(left))
        spacings = left + right
        while len(spacings) < N:
            spacings.append(left[-1] if left else h_min)
    total = sum(spacings) or 1.0
    ts = [0.0]
    acc = 0.0
    for s in spacings:
        acc += s / total
        ts.append(min(1.0, acc))
    return ts


class BoundaryLayerDialog(QDialog):
    """Inflation / boundary-layer dialog. Configures stacked thin elements
    parallel to the picked edges, growing geometrically into the interior.
    """

    applied = Signal(object)

    def __init__(self, parent=None, edge_refs=None, initial_seed=None):
        super().__init__(parent)
        self.setWindowTitle("Boundary Layer")
        self.setModal(True)
        self.setMinimumWidth(360)
        self._edge_refs = list(edge_refs or [])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        hint = QLabel(
            "Stack thin layers parallel to the selected edges, growing "
            "geometrically into the interior. Common for CFD boundary "
            "layers, thermal gradients, and contact-zone refinement."
        )
        hint.setObjectName("MinorStatusLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        scope = QLabel(f"Applies to {len(self._edge_refs)} edge(s)")
        scope.setObjectName("MinorStatusLabel")
        layout.addWidget(scope)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        self.first_layer_spin = QDoubleSpinBox()
        self.first_layer_spin.setDecimals(5)
        self.first_layer_spin.setRange(1e-6, 1e6)
        self.first_layer_spin.setValue(0.05)
        self.first_layer_spin.setSingleStep(0.01)
        self.first_layer_spin.setToolTip("Thickness of the first (thinnest) layer at the wall.")
        form.addRow("First layer thickness:", self.first_layer_spin)

        self.ratio_spin = QDoubleSpinBox()
        self.ratio_spin.setDecimals(3)
        self.ratio_spin.setRange(1.0, 5.0)
        self.ratio_spin.setSingleStep(0.05)
        self.ratio_spin.setValue(1.2)
        self.ratio_spin.setToolTip("Geometric growth ratio between consecutive layers (>= 1.0).")
        form.addRow("Growth ratio:", self.ratio_spin)

        self.num_layers_spin = QSpinBox()
        self.num_layers_spin.setRange(1, 200)
        self.num_layers_spin.setValue(5)
        self.num_layers_spin.setToolTip("Number of layers to extrude from the edge.")
        form.addRow("Number of layers:", self.num_layers_spin)

        self.max_thickness_spin = QDoubleSpinBox()
        self.max_thickness_spin.setDecimals(4)
        self.max_thickness_spin.setRange(0.0, 1e6)
        self.max_thickness_spin.setValue(0.0)
        self.max_thickness_spin.setSpecialValueText("auto")
        self.max_thickness_spin.setToolTip(
            "Optional cap on total inflation thickness. 0 = no cap "
            "(use the natural total = first × (1 − r^N) / (1 − r))."
        )
        form.addRow("Max total thickness:", self.max_thickness_spin)

        self.quads_check = QCheckBox("Generate quad elements in the boundary-layer band")
        self.quads_check.setToolTip(
            "When checked, the boundary-layer band is meshed with quadrilaterals "
            "instead of the default triangles. Useful for structured-looking layers."
        )
        layout.addLayout(form)
        layout.addWidget(self.quads_check)

        # Computed-summary line — live updates as the user types.
        self.summary_label = QLabel("")
        self.summary_label.setObjectName("MinorStatusLabel")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        for sp in (self.first_layer_spin, self.ratio_spin, self.num_layers_spin, self.max_thickness_spin):
            sp.valueChanged.connect(self._refresh_summary)

        # Set creation
        set_row = QHBoxLayout()
        self.set_check = QCheckBox("Create set with name:")
        self.set_edit = QLineEdit("Boundary Layer-1")
        self.set_edit.setEnabled(False)
        self.set_check.toggled.connect(self.set_edit.setEnabled)
        set_row.addWidget(self.set_check)
        set_row.addWidget(self.set_edit, 1)
        layout.addLayout(set_row)

        btn_row = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.apply_btn = QPushButton("Apply")
        self.defaults_btn = QPushButton("Defaults")
        self.cancel_btn = QPushButton("Cancel")
        for b in (self.ok_btn, self.apply_btn, self.defaults_btn, self.cancel_btn):
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        self.ok_btn.clicked.connect(self._on_ok)
        self.apply_btn.clicked.connect(self._on_apply)
        self.defaults_btn.clicked.connect(self._on_defaults)
        self.cancel_btn.clicked.connect(self.reject)

        if initial_seed is not None:
            self._load_from_seed(initial_seed)
        self._refresh_summary()

    def _refresh_summary(self):
        h0 = float(self.first_layer_spin.value())
        r = float(self.ratio_spin.value())
        N = int(self.num_layers_spin.value())
        cap = float(self.max_thickness_spin.value())
        if abs(r - 1.0) < 1e-9:
            total = h0 * N
        else:
            total = h0 * (1.0 - r ** N) / (1.0 - r)
        if cap > 0:
            total = min(total, cap)
        # Outermost layer thickness
        h_last = h0 * (r ** (N - 1)) if N > 0 else 0.0
        self.summary_label.setText(
            f"Total inflation thickness: {total:.4g}  ·  "
            f"Outermost layer: {h_last:.4g}  ·  Layers: {N}"
        )

    def _on_defaults(self):
        self.first_layer_spin.setValue(0.05)
        self.ratio_spin.setValue(1.2)
        self.num_layers_spin.setValue(5)
        self.max_thickness_spin.setValue(0.0)
        self.quads_check.setChecked(False)
        self.set_check.setChecked(False)
        self.set_edit.setText("Boundary Layer-1")
        self._refresh_summary()

    def _load_from_seed(self, seed):
        try:
            self.first_layer_spin.setValue(float(seed.first_layer_size))
        except Exception:
            pass
        try:
            self.ratio_spin.setValue(float(seed.growth_ratio))
        except Exception:
            pass
        try:
            self.num_layers_spin.setValue(int(seed.num_layers))
        except Exception:
            pass
        try:
            self.max_thickness_spin.setValue(float(seed.max_thickness))
        except Exception:
            pass
        self.quads_check.setChecked(bool(seed.quads))
        if seed.set_name:
            self.set_check.setChecked(True)
            self.set_edit.setText(seed.set_name)

    def build_seed(self):
        seed = BoundaryLayerSeed(
            edge_refs=list(self._edge_refs),
            first_layer_size=float(self.first_layer_spin.value()),
            growth_ratio=float(self.ratio_spin.value()),
            num_layers=int(self.num_layers_spin.value()),
            quads=bool(self.quads_check.isChecked()),
            max_thickness=float(self.max_thickness_spin.value()),
            set_name=(self.set_edit.text().strip() if self.set_check.isChecked() else ""),
        )
        return seed if seed.is_valid() else None

    def _on_apply(self):
        seed = self.build_seed()
        if seed is None:
            QMessageBox.warning(self, "Invalid", "First-layer thickness must be > 0 and at least one edge must be picked.")
            return
        self.applied.emit(seed)

    def _on_ok(self):
        seed = self.build_seed()
        if seed is None:
            QMessageBox.warning(self, "Invalid", "First-layer thickness must be > 0 and at least one edge must be picked.")
            return
        self.applied.emit(seed)
        self.accept()


class VertexSeedDialog(QDialog):
    """Compact dialog for a vertex-anchored refinement seed.

    Two numbers: target element size AT the vertex, and influence radius over
    which the mesh grows back to the global bulk size.
    """

    applied = Signal(object)

    def __init__(self, parent=None, part_id=0, point=(0.0, 0.0), initial_seed=None):
        super().__init__(parent)
        self.setWindowTitle("Vertex Seed")
        self.setModal(True)
        self.setMinimumWidth(340)
        self._part_id = int(part_id)
        self._point = (float(point[0]), float(point[1]))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        loc_lbl = QLabel(
            f"Vertex: part {self._part_id}, ({self._point[0]:g}, {self._point[1]:g})"
        )
        loc_lbl.setObjectName("MinorStatusLabel")
        layout.addWidget(loc_lbl)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        self.target_size_spin = QDoubleSpinBox()
        self.target_size_spin.setDecimals(4)
        self.target_size_spin.setRange(1e-6, 1e6)
        self.target_size_spin.setValue(0.1)
        self.target_size_spin.setSingleStep(0.05)
        self.target_size_spin.setToolTip(
            "Element edge length AT the vertex. The mesh refines to this "
            "size right at the point and grows back to the global bulk over "
            "the influence radius."
        )
        form.addRow("Target size at vertex:", self.target_size_spin)

        self.influence_radius_spin = QDoubleSpinBox()
        self.influence_radius_spin.setDecimals(4)
        self.influence_radius_spin.setRange(1e-6, 1e6)
        self.influence_radius_spin.setValue(2.0)
        self.influence_radius_spin.setSingleStep(0.1)
        self.influence_radius_spin.setToolTip(
            "Distance over which size grows from target_size at the vertex "
            "back to the global bulk size."
        )
        form.addRow("Influence radius:", self.influence_radius_spin)
        layout.addLayout(form)

        set_row = QHBoxLayout()
        self.set_check = QCheckBox("Create set with name:")
        self.set_edit = QLineEdit("Vertex Seeds-1")
        self.set_edit.setEnabled(False)
        self.set_check.toggled.connect(self.set_edit.setEnabled)
        set_row.addWidget(self.set_check)
        set_row.addWidget(self.set_edit, 1)
        layout.addLayout(set_row)

        btn_row = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.apply_btn = QPushButton("Apply")
        self.defaults_btn = QPushButton("Defaults")
        self.cancel_btn = QPushButton("Cancel")
        for b in (self.ok_btn, self.apply_btn, self.defaults_btn, self.cancel_btn):
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        self.ok_btn.clicked.connect(self._on_ok)
        self.apply_btn.clicked.connect(self._on_apply)
        self.defaults_btn.clicked.connect(self._on_defaults)
        self.cancel_btn.clicked.connect(self.reject)

        if initial_seed is not None:
            self._load_from_seed(initial_seed)

    def _on_defaults(self):
        self.target_size_spin.setValue(0.1)
        self.influence_radius_spin.setValue(2.0)
        self.set_check.setChecked(False)
        self.set_edit.setText("Vertex Seeds-1")

    def _load_from_seed(self, seed):
        try:
            self.target_size_spin.setValue(float(seed.target_size))
        except Exception:
            pass
        try:
            self.influence_radius_spin.setValue(float(seed.influence_radius))
        except Exception:
            pass
        if seed.set_name:
            self.set_check.setChecked(True)
            self.set_edit.setText(seed.set_name)

    def build_seed(self):
        seed = VertexSeed(
            point=self._point,
            target_size=float(self.target_size_spin.value()),
            influence_radius=float(self.influence_radius_spin.value()),
            part_id=self._part_id,
            set_name=(self.set_edit.text().strip() if self.set_check.isChecked() else ""),
        )
        return seed if seed.is_valid() else None

    def _on_apply(self):
        seed = self.build_seed()
        if seed is None:
            QMessageBox.warning(self, "Invalid seed",
                                "Target size and influence radius must both be > 0.")
            return
        self.applied.emit(seed)

    def _on_ok(self):
        seed = self.build_seed()
        if seed is None:
            QMessageBox.warning(self, "Invalid seed",
                                "Target size and influence radius must both be > 0.")
            return
        self.applied.emit(seed)
        self.accept()


class LocalSeedsDialog(QDialog):
    """Abaqus-style Local Seeds dialog (Basic tab).

    Lets the user configure an EdgeSeed for one or more pre-selected edges.
    Mirrors the controls in the Abaqus dialog:
      - Method: By size / By number
      - Bias:   None / Single / Double
      - Sizing Controls (conditional on Method+Bias):
          by_size + none           → element size + curvature control
          by_size + single|double  → min size, max size, Flip
          by_number + none         → number of elements
          by_number + single|double→ number of elements, bias ratio, Flip
      - Set creation: optional friendly name
      - Buttons: OK / Apply / Defaults / Cancel
    """

    applied = Signal(object)  # emits the validated EdgeSeed when Apply/OK is clicked

    def __init__(self, parent=None, edge_refs=None, initial_seed=None, sketch_view=None):
        super().__init__(parent)
        self.setWindowTitle("Local Seeds")
        self.setModal(True)
        self.setMinimumWidth(380)
        self._edge_refs = list(edge_refs or [])
        # sketch_view used for the live tick-preview while the dialog is open.
        # We intentionally accept None — if the caller doesn't pass one, the
        # preview just no-ops, the rest of the dialog still works.
        self._sketch_view = sketch_view

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Tab container — Basic / Constraints (matches the Abaqus dialog).
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # ===== Basic tab =====
        basic_tab = QWidget()
        basic_layout = QVBoxLayout(basic_tab)
        basic_layout.setContentsMargins(8, 8, 8, 8)
        basic_layout.setSpacing(10)
        self.tabs.addTab(basic_tab, "Basic")

        # --- Method + Bias side-by-side ---
        top_row = QHBoxLayout()
        method_group = QGroupBox("Method")
        m_l = QVBoxLayout(method_group)
        self.method_by_size = QRadioButton("By size")
        self.method_by_number = QRadioButton("By number")
        self.method_by_size.setChecked(True)
        m_l.addWidget(self.method_by_size)
        m_l.addWidget(self.method_by_number)
        top_row.addWidget(method_group, 1)

        bias_group = QGroupBox("Bias")
        b_l = QVBoxLayout(bias_group)
        self.bias_none = QRadioButton("None")
        self.bias_single = QRadioButton("Single")
        self.bias_double = QRadioButton("Double")
        self.bias_none.setChecked(True)
        b_l.addWidget(self.bias_none)
        b_l.addWidget(self.bias_single)
        b_l.addWidget(self.bias_double)
        top_row.addWidget(bias_group, 1)
        basic_layout.addLayout(top_row)

        # --- Sizing Controls ---
        self.sizing_group = QGroupBox("Sizing Controls")
        sizing_layout = QVBoxLayout(self.sizing_group)

        # by_size + none: Approximate element size + curvature group
        self._row_elem_size = QHBoxLayout()
        self._row_elem_size.addWidget(QLabel("Approximate element size:"))
        self.element_size_spin = QDoubleSpinBox()
        self.element_size_spin.setDecimals(4)
        self.element_size_spin.setRange(1e-6, 1e6)
        self.element_size_spin.setValue(1.0)
        self._row_elem_size.addWidget(self.element_size_spin, 1)
        sizing_layout.addLayout(self._row_elem_size)

        self.curvature_check = QCheckBox("Curvature control")
        self.curvature_check.setToolTip(
            "Refine the mesh where the boundary curves (chord-height ratio). "
            "Works only when the underlying geometry uses curved primitives "
            "(arcs / splines). Polyline-approximated curves see zero curvature "
            "and are unaffected."
        )
        sizing_layout.addWidget(self.curvature_check)
        self.curvature_sub = QWidget()
        cs_layout = QVBoxLayout(self.curvature_sub)
        cs_layout.setContentsMargins(20, 0, 0, 0)
        cs_layout.setSpacing(4)
        dev_row = QHBoxLayout()
        dev_row.addWidget(QLabel("Maximum deviation factor (0.0 < h/L < 1.0):"))
        self.curvature_dev_factor = QDoubleSpinBox()
        self.curvature_dev_factor.setDecimals(4)
        self.curvature_dev_factor.setRange(0.001, 0.999)
        self.curvature_dev_factor.setValue(0.1)
        dev_row.addWidget(self.curvature_dev_factor)
        cs_layout.addLayout(dev_row)
        self.curvature_min_default = QRadioButton("Use default (0.1)")
        self.curvature_min_default.setChecked(True)
        self.curvature_min_specify = QRadioButton("Specify (0.0 < min < 1.0)")
        min_row = QHBoxLayout()
        min_row.addWidget(self.curvature_min_specify)
        self.curvature_min_value = QDoubleSpinBox()
        self.curvature_min_value.setDecimals(4)
        self.curvature_min_value.setRange(0.001, 0.999)
        self.curvature_min_value.setValue(0.1)
        self.curvature_min_value.setEnabled(False)
        min_row.addWidget(self.curvature_min_value)
        cs_layout.addWidget(self.curvature_min_default)
        cs_layout.addLayout(min_row)
        sizing_layout.addWidget(self.curvature_sub)

        # by_size + single|double: per-vertex sizes (replaces the old
        # Min/Max + Flip workflow — users now name the fine end directly).
        self._row_min = QHBoxLayout()
        self._row_min.addWidget(QLabel("Size at start vertex:"))
        self.start_size_spin = QDoubleSpinBox()
        self.start_size_spin.setDecimals(4)
        self.start_size_spin.setRange(1e-6, 1e6)
        self.start_size_spin.setValue(0.1)
        self.start_size_spin.setToolTip(
            "Element size at the START endpoint of each picked edge. "
            "Set this smaller than 'Size at end' to bias fine toward the start."
        )
        self._row_min.addWidget(self.start_size_spin, 1)
        sizing_layout.addLayout(self._row_min)

        self._row_max = QHBoxLayout()
        self._row_max.addWidget(QLabel("Size at end vertex:"))
        self.end_size_spin = QDoubleSpinBox()
        self.end_size_spin.setDecimals(4)
        self.end_size_spin.setRange(1e-6, 1e6)
        self.end_size_spin.setValue(1.0)
        self.end_size_spin.setToolTip(
            "Element size at the END endpoint of each picked edge."
        )
        self._row_max.addWidget(self.end_size_spin, 1)
        sizing_layout.addLayout(self._row_max)

        # Keep the old attribute names as aliases for any code that still
        # references them (mostly nothing — build_seed handles the
        # translation). This avoids touching unrelated callers.
        self.min_size_spin = self.start_size_spin
        self.max_size_spin = self.end_size_spin

        # by_number: count + (bias_ratio + flip when biased)
        self._row_count = QHBoxLayout()
        self._row_count.addWidget(QLabel("Number of elements:"))
        self.seed_count_spin = QSpinBox()
        self.seed_count_spin.setRange(1, 100000)
        self.seed_count_spin.setValue(10)
        self._row_count.addWidget(self.seed_count_spin, 1)
        sizing_layout.addLayout(self._row_count)

        self._row_ratio = QHBoxLayout()
        self._row_ratio.addWidget(QLabel("Bias ratio:"))
        self.bias_ratio_spin = QDoubleSpinBox()
        self.bias_ratio_spin.setDecimals(2)
        self.bias_ratio_spin.setRange(1.0, 1000.0)
        self.bias_ratio_spin.setValue(2.0)
        self._row_ratio.addWidget(self.bias_ratio_spin, 1)
        sizing_layout.addLayout(self._row_ratio)

        # Flip button (only shown when bias != none)
        self._row_flip = QHBoxLayout()
        self._row_flip.addWidget(QLabel("Flip bias:"))
        self.flip_button = QPushButton("Flip")
        self.flip_button.setCheckable(True)
        self._row_flip.addWidget(self.flip_button)
        self._row_flip.addStretch(1)
        sizing_layout.addLayout(self._row_flip)

        basic_layout.addWidget(self.sizing_group)

        # --- Set creation ---
        set_group = QGroupBox("Set Creation")
        s_l = QHBoxLayout(set_group)
        self.set_check = QCheckBox("Create set with name:")
        self.set_edit = QLineEdit("Edge Seeds-1")
        self.set_edit.setEnabled(False)
        self.set_check.toggled.connect(self.set_edit.setEnabled)
        s_l.addWidget(self.set_check)
        s_l.addWidget(self.set_edit, 1)
        basic_layout.addWidget(set_group)
        basic_layout.addStretch(1)

        # ===== Constraints tab =====
        constraints_tab = QWidget()
        constraints_layout = QVBoxLayout(constraints_tab)
        constraints_layout.setContentsMargins(8, 8, 8, 8)
        constraints_layout.setSpacing(10)
        self.tabs.addTab(constraints_tab, "Constraints")

        constraints_hint = QLabel(
            "Constraints control how this seed interacts with neighboring "
            "edges and the global mesher."
        )
        constraints_hint.setObjectName("MinorStatusLabel")
        constraints_hint.setWordWrap(True)
        constraints_layout.addWidget(constraints_hint)

        self.propagate_check = QCheckBox(
            "Propagate to neighboring edges (sharing endpoints)"
        )
        self.propagate_check.setToolTip(
            "Apply the same size field to any edge that shares an endpoint "
            "with one of the seeded edges. Useful when a fillet or corner "
            "boundary should be matched on both sides of the joint."
        )
        constraints_layout.addWidget(self.propagate_check)
        constraints_layout.addStretch(1)

        # --- Buttons ---
        btn_row = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.apply_btn = QPushButton("Apply")
        self.save_template_btn = QPushButton("Save as template...")
        self.save_template_btn.setToolTip(
            "Save this configuration as a named template (excluding the "
            "current edge selection). Apply later to any other edges."
        )
        self.defaults_btn = QPushButton("Defaults")
        self.cancel_btn = QPushButton("Cancel")
        for b in (self.ok_btn, self.apply_btn, self.save_template_btn, self.defaults_btn, self.cancel_btn):
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        # --- Signals ---
        self.method_by_size.toggled.connect(self._refresh_sizing_visibility)
        self.method_by_number.toggled.connect(self._refresh_sizing_visibility)
        self.bias_none.toggled.connect(self._refresh_sizing_visibility)
        self.bias_single.toggled.connect(self._refresh_sizing_visibility)
        self.bias_double.toggled.connect(self._refresh_sizing_visibility)
        self.curvature_check.toggled.connect(self._refresh_curvature_visibility)
        self.curvature_min_specify.toggled.connect(self.curvature_min_value.setEnabled)
        self.ok_btn.clicked.connect(self._on_ok)
        self.apply_btn.clicked.connect(self._on_apply)
        self.save_template_btn.clicked.connect(self._on_save_as_template)
        self.defaults_btn.clicked.connect(self._on_defaults)
        self.cancel_btn.clicked.connect(self.reject)

        if initial_seed is not None:
            self._load_from_seed(initial_seed)
        self._refresh_sizing_visibility()
        self._refresh_curvature_visibility()

        # Wire every meaningful widget to the live tick preview so users see
        # spacing change as they type. The preview is purely visual on the
        # canvas; project state is only touched on Apply / OK.
        for sp in (
            self.element_size_spin, self.min_size_spin, self.max_size_spin,
            self.seed_count_spin, self.bias_ratio_spin,
        ):
            sp.valueChanged.connect(self._refresh_seed_preview)
        for rb in (
            self.method_by_size, self.method_by_number,
            self.bias_none, self.bias_single, self.bias_double,
        ):
            rb.toggled.connect(self._refresh_seed_preview)
        self.flip_button.toggled.connect(self._refresh_seed_preview)
        # Push the initial preview now.
        self._refresh_seed_preview()

    def _refresh_seed_preview(self, *_args):
        sv = self._sketch_view
        if sv is None or not hasattr(sv, "set_seed_preview_ticks"):
            return
        seed = self.build_seed()
        if seed is None:
            try:
                sv.clear_seed_preview()
            except Exception:
                pass
            return
        seed_dict = seed.to_dict()
        ticks = []
        for ref in self._edge_refs:
            sx, sy = ref["start"]
            ex, ey = ref["end"]
            import math as _math
            edge_len = _math.hypot(ex - sx, ey - sy)
            params = _seed_tick_params(seed_dict, edge_len)
            for t in params:
                ticks.append((sx + t * (ex - sx), sy + t * (ey - sy)))
        try:
            sv.set_seed_preview_ticks(ticks)
        except Exception:
            pass

    def _clear_preview(self):
        sv = self._sketch_view
        if sv is not None and hasattr(sv, "clear_seed_preview"):
            try:
                sv.clear_seed_preview()
            except Exception:
                pass

    def closeEvent(self, ev):
        self._clear_preview()
        super().closeEvent(ev)

    def accept(self):
        self._clear_preview()
        super().accept()

    def reject(self):
        self._clear_preview()
        super().reject()

    # ----- helpers -----

    def _set_row_visible(self, layout, visible):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            w = item.widget() if item is not None else None
            if w is not None:
                w.setVisible(visible)

    def _refresh_sizing_visibility(self, _checked=None):
        by_size = self.method_by_size.isChecked()
        bias = (
            "none" if self.bias_none.isChecked()
            else "single" if self.bias_single.isChecked()
            else "double"
        )
        # Show/hide rows.
        self._set_row_visible(self._row_elem_size, by_size and bias == "none")
        self.curvature_check.setVisible(by_size and bias == "none")
        self.curvature_sub.setVisible(
            by_size and bias == "none" and self.curvature_check.isChecked()
        )
        self._set_row_visible(self._row_min, by_size and bias != "none")
        self._set_row_visible(self._row_max, by_size and bias != "none")
        self._set_row_visible(self._row_count, not by_size)
        self._set_row_visible(self._row_ratio, (not by_size) and bias != "none")
        # Flip is only relevant for by_number bias (where the user enters a
        # ratio, not direct endpoint sizes). For by_size + bias, the user
        # already picks which end is fine by typing different start/end sizes,
        # so the Flip toggle would be redundant.
        self._set_row_visible(self._row_flip, (not by_size) and bias != "none")

    def _refresh_curvature_visibility(self, _checked=None):
        show = (
            self.method_by_size.isChecked()
            and self.bias_none.isChecked()
            and self.curvature_check.isChecked()
        )
        self.curvature_sub.setVisible(show)

    def _on_defaults(self):
        self.method_by_size.setChecked(True)
        self.bias_none.setChecked(True)
        self.element_size_spin.setValue(1.0)
        self.start_size_spin.setValue(0.1)
        self.end_size_spin.setValue(1.0)
        self.seed_count_spin.setValue(10)
        self.bias_ratio_spin.setValue(2.0)
        self.flip_button.setChecked(False)
        self.curvature_check.setChecked(False)
        self.curvature_dev_factor.setValue(0.1)
        self.curvature_min_default.setChecked(True)
        self.curvature_min_value.setValue(0.1)
        self.set_check.setChecked(False)
        self.set_edit.setText("Edge Seeds-1")
        self.propagate_check.setChecked(False)
        self._refresh_sizing_visibility()

    def _load_from_seed(self, seed):
        if seed.method == "by_number":
            self.method_by_number.setChecked(True)
        else:
            self.method_by_size.setChecked(True)
        if seed.bias == "single":
            self.bias_single.setChecked(True)
        elif seed.bias == "double":
            self.bias_double.setChecked(True)
        else:
            self.bias_none.setChecked(True)
        if seed.element_size > 0:
            self.element_size_spin.setValue(seed.element_size)
        # Convert stored (min, max, flip) back into start/end sizes.
        if seed.min_size > 0 and seed.max_size > 0:
            lo, hi = min(seed.min_size, seed.max_size), max(seed.min_size, seed.max_size)
            if seed.flip_bias:
                # flip means the FINE end is the edge's END vertex
                self.start_size_spin.setValue(hi)
                self.end_size_spin.setValue(lo)
            else:
                self.start_size_spin.setValue(lo)
                self.end_size_spin.setValue(hi)
        if seed.seed_count > 0:
            self.seed_count_spin.setValue(seed.seed_count)
        if seed.bias_ratio > 0:
            self.bias_ratio_spin.setValue(seed.bias_ratio)
        self.flip_button.setChecked(bool(seed.flip_bias))
        self.curvature_check.setChecked(bool(seed.curvature_control))
        self.curvature_dev_factor.setValue(seed.max_deviation_factor)
        self.set_check.setChecked(bool(seed.set_name))
        if seed.set_name:
            self.set_edit.setText(seed.set_name)
        self.propagate_check.setChecked(bool(getattr(seed, "propagate_to_neighbors", False)))

    def build_seed(self):
        """Construct an EdgeSeed from the current dialog state, or return None
        if the inputs are invalid."""
        bias = (
            "none" if self.bias_none.isChecked()
            else "single" if self.bias_single.isChecked()
            else "double"
        )
        method = "by_size" if self.method_by_size.isChecked() else "by_number"

        # Convert per-vertex sizes into (min, max, flip) for the existing
        # storage / backend model. flip = True means the "fine" end is the
        # edge's END vertex, which is exactly when start_size > end_size.
        s_size = float(self.start_size_spin.value())
        e_size = float(self.end_size_spin.value())
        flip_from_endpoints = s_size > e_size
        ui_min = min(s_size, e_size) if (s_size > 0 and e_size > 0) else 0.0
        ui_max = max(s_size, e_size) if (s_size > 0 and e_size > 0) else 0.0

        # For by_size + bias: use the per-vertex translation. For by_number +
        # bias, the user still uses the Flip toggle (no direct endpoint sizes).
        if method == "by_size" and bias != "none":
            effective_flip = flip_from_endpoints
        else:
            effective_flip = self.flip_button.isChecked()

        seed = EdgeSeed(
            edge_refs=list(self._edge_refs),
            method=method,
            bias=bias,
            flip_bias=effective_flip,
            element_size=float(self.element_size_spin.value()),
            min_size=ui_min,
            max_size=ui_max,
            seed_count=int(self.seed_count_spin.value()),
            bias_ratio=float(self.bias_ratio_spin.value()),
            curvature_control=bool(self.curvature_check.isChecked()),
            max_deviation_factor=float(self.curvature_dev_factor.value()),
            min_size_factor=(
                float(self.curvature_min_value.value())
                if self.curvature_min_specify.isChecked()
                else 0.1
            ),
            set_name=(self.set_edit.text().strip() if self.set_check.isChecked() else ""),
            propagate_to_neighbors=self.propagate_check.isChecked(),
        )
        return seed if seed.is_valid() else None

    def _on_apply(self):
        seed = self.build_seed()
        if seed is None:
            QMessageBox.warning(
                self,
                "Invalid seed",
                "The seed values are incomplete. Set sizes/counts > 0 and pick edges before applying.",
            )
            return
        self.applied.emit(seed)

    def _on_ok(self):
        seed = self.build_seed()
        if seed is None:
            QMessageBox.warning(
                self,
                "Invalid seed",
                "The seed values are incomplete. Set sizes/counts > 0 and pick edges before applying.",
            )
            return
        self.applied.emit(seed)
        self.accept()

    # Templates: emit an EdgeSeed with the current configuration but no
    # edge_refs. The caller decides where to store it (typically
    # project_state.edge_seed_templates). The dialog itself does not touch
    # the project state — the parent does, via the signal below.
    saveTemplateRequested = Signal(object, str)  # (template_seed, name)

    def _on_save_as_template(self):
        # Validate the configuration ignoring edge_refs (templates don't
        # need edges to be picked).
        seed = self.build_seed()
        if seed is None:
            # Try building with placeholder edges so is_valid passes — we
            # only care that the configuration is internally consistent.
            try:
                bias = ("none" if self.bias_none.isChecked()
                        else "single" if self.bias_single.isChecked()
                        else "double")
                method = "by_size" if self.method_by_size.isChecked() else "by_number"
                s_size, e_size = float(self.start_size_spin.value()), float(self.end_size_spin.value())
                ui_min = min(s_size, e_size) if (s_size > 0 and e_size > 0) else 0.0
                ui_max = max(s_size, e_size) if (s_size > 0 and e_size > 0) else 0.0
                seed = EdgeSeed(
                    edge_refs=[{"part_id": 0, "start": (0, 0), "end": (1, 0)}],  # placeholder, stripped before save
                    method=method, bias=bias,
                    flip_bias=(method == "by_size" and bias != "none" and s_size > e_size) or self.flip_button.isChecked(),
                    element_size=float(self.element_size_spin.value()),
                    min_size=ui_min, max_size=ui_max,
                    seed_count=int(self.seed_count_spin.value()),
                    bias_ratio=float(self.bias_ratio_spin.value()),
                    curvature_control=bool(self.curvature_check.isChecked()),
                    max_deviation_factor=float(self.curvature_dev_factor.value()),
                    min_size_factor=(float(self.curvature_min_value.value())
                                     if self.curvature_min_specify.isChecked() else 0.1),
                    propagate_to_neighbors=self.propagate_check.isChecked(),
                )
                if not seed.is_valid():
                    QMessageBox.warning(self, "Invalid template",
                                        "Set sizes / counts > 0 before saving as a template.")
                    return
            except Exception as exc:
                QMessageBox.warning(self, "Invalid template", str(exc))
                return
        # Ask for a template name.
        name, ok = QInputDialog.getText(self, "Save Template", "Template name:",
                                         text=self.set_edit.text().strip() or "Edge Seed Template")
        if not ok or not name.strip():
            return
        # Strip edge_refs and emit. The parent stores it in project state.
        seed.edge_refs = []
        seed.set_name = name.strip()
        self.saveTemplateRequested.emit(seed, name.strip())


class AssemblyTreeWidget(QTreeWidget):
    sketchReorderDropped = Signal(object)
    renameShortcutRequested = Signal()

    def _row_owner_key(self, item):
        if item is None:
            return None
        data = item.data(0, Qt.UserRole)
        if not isinstance(data, dict):
            return None
        kind = data.get("kind")
        if kind == "active_sketch":
            return ("sketch", None)
        if kind == "part_sketch":
            return ("part", data.get("part_id"))
        return None

    def _container_owner_key(self, item):
        if item is None:
            return None
        data = item.data(0, Qt.UserRole)
        if not isinstance(data, dict):
            return None
        kind = data.get("kind")
        if kind == "active_sketch_header":
            return ("sketch", None)
        if kind == "part":
            return ("part", data.get("id"))
        return None

    def dropEvent(self, event):
        dragged_item = self.currentItem()
        row_owner = self._row_owner_key(dragged_item)
        dragged_index = None
        if dragged_item is not None:
            dragged_data = dragged_item.data(0, Qt.UserRole)
            if isinstance(dragged_data, dict):
                try:
                    dragged_index = int(dragged_data.get("sketch_index"))
                except Exception:
                    dragged_index = None
        if row_owner is None:
            event.ignore()
            return

        super().dropEvent(event)

        moved_item = self.currentItem() or dragged_item
        payload = {
            "valid": False,
            "owner_type": row_owner[0],
            "part_id": row_owner[1],
            "order": [],
            "dragged_index": dragged_index,
        }
        if moved_item is None:
            self.sketchReorderDropped.emit(payload)
            return

        moved_row_owner = self._row_owner_key(moved_item)
        parent_item = moved_item.parent()
        container_owner = self._container_owner_key(parent_item)
        if moved_row_owner != row_owner or container_owner != row_owner:
            self.sketchReorderDropped.emit(payload)
            return

        order = []
        for i in range(parent_item.childCount()):
            child = parent_item.child(i)
            if self._row_owner_key(child) != row_owner:
                continue
            data = child.data(0, Qt.UserRole)
            if not isinstance(data, dict):
                continue
            try:
                order.append(int(data.get("sketch_index")))
            except Exception:
                self.sketchReorderDropped.emit(payload)
                return

        if sorted(order) != list(range(len(order))):
            self.sketchReorderDropped.emit(payload)
            return

        payload["valid"] = True
        payload["order"] = order
        self.sketchReorderDropped.emit(payload)

    def keyPressEvent(self, event):
        try:
            if int(event.key()) == int(Qt.Key_F2):
                self.renameShortcutRequested.emit()
                event.accept()
                return
        except Exception:
            pass
        super().keyPressEvent(event)


class OperationHistoryTree(QWidget):
    """Widget displaying the part assembly and allowing modifications."""

    def __init__(self, sketch_view, parent=None, project_state=None):
        super().__init__(parent)
        self.sketch_view = sketch_view
        self.project_state = _resolve_project_state(sketch_view, project_state)
        self._suppress_selection = False
        self._suppress_item_changed = False
        self._inline_rename_ctx = None
        self._col_part_id = 0
        self._col_part = 1
        self._col_geom_type = 2
        self._col_material = -1
        self._col_bcs = -1
        self._col_loads = -1
        self._col_interfaces = -1
        self._col_status = 2
        self._col_rigid = 2
        self._col_vis = 2
        layout = QVBoxLayout(self)
        _apply_layout_metrics(layout, margins=(6, 6, 6, 6), spacing=4)

        # Compact inline Project Setup row — Analysis + Dim on a single line.
        setup_row = QHBoxLayout()
        setup_row.setContentsMargins(0, 0, 0, 0)
        setup_row.setSpacing(6)
        analysis_label = QLabel("Analysis")
        analysis_label.setObjectName("SummaryLabel")
        setup_row.addWidget(analysis_label)
        self.analysis_type_combo = QComboBox()
        self.analysis_type_combo.addItem("Static Mechanical", "static")
        self.analysis_type_combo.addItem("Dynamic Explicit", "dynamic")
        self.analysis_type_combo.addItem("Fluid", "fluid")
        self.analysis_type_combo.addItem("Fluid-Structure Interaction", "fsi")
        self.analysis_type_combo.currentIndexChanged.connect(self._on_analysis_type_changed)
        self.analysis_type_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        # Cap minimum width so the longest item ("Fluid-Structure Interaction")
        # doesn't force the whole panel wider than the right dock can hold.
        self.analysis_type_combo.setMinimumContentsLength(8)
        self.analysis_type_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.analysis_type_combo.setMinimumWidth(80)
        self.analysis_type_combo.setMaximumWidth(180)
        setup_row.addWidget(self.analysis_type_combo, 1)
        dim_label = QLabel("Dim")
        dim_label.setObjectName("SummaryLabel")
        setup_row.addWidget(dim_label)
        self.dimension_combo = QComboBox()
        self.dimension_combo.addItem("2D", "2D")
        self.dimension_combo.addItem("3D", "3D")
        self.dimension_combo.currentIndexChanged.connect(self._on_dimension_changed)
        self.dimension_combo.setMaximumWidth(80)
        setup_row.addWidget(self.dimension_combo, 0)
        layout.addLayout(setup_row)

        self.create_layout = QGridLayout()
        self.create_layout.setContentsMargins(0, 0, 0, 0)
        self.create_layout.setSpacing(2)

        def build_tool_button(icon_name, label, tooltip, callback, *, compact_label=None, checkable=False, checked=False):
            button = QToolButton(self)
            button.setProperty("dockIconButton", True)
            button.setIcon(get_icon(icon_name, size=14))
            button.setIconSize(QSize(14, 14))
            button.setText(str(label or ""))
            button.setToolTip(str(tooltip or ""))
            button.setAutoRaise(False)
            button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            button.setCheckable(bool(checkable))
            button.setChecked(bool(checked))
            button.setMinimumSize(DOCK_ICON_BTN_MIN, 28)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.setProperty("responsiveFullText", str(label or ""))
            compact_text = str(compact_label or label or "").strip()
            if not compact_text and label:
                compact_text = str(label).split()[0]
            button.setProperty("responsiveCompactText", compact_text)
            if checkable:
                button.toggled.connect(callback)
            else:
                button.clicked.connect(callback)
            return button

        def build_menu_tool_button(icon_name, label, tooltip, callback, menu_entries, *, compact_label=None):
            button = build_tool_button(
                icon_name,
                label,
                tooltip,
                callback,
                compact_label=compact_label,
            )
            menu = QMenu(button)
            for label, handler in list(menu_entries or []):
                menu.addAction(str(label), handler)
            button.setMenu(menu)
            button.setPopupMode(QToolButton.MenuButtonPopup)
            return button

        create_buttons = [
            build_menu_tool_button(
                "rectangle",
                "Rectangle",
                "Rectangle\nClick: use the current rectangle method. Menu: corner, center, and dimensioned rectangle modes.",
                lambda _checked=False: self._activate_rectangle_tool(),
                [
                    ("Corner Rectangle", lambda: self._activate_rectangle_tool(mode="corner", use_dimensions=False)),
                    ("Center Rectangle", lambda: self._activate_rectangle_tool(mode="center", use_dimensions=False)),
                    ("Corner Rectangle + Dimensions", lambda: self._activate_rectangle_tool(mode="corner", use_dimensions=True)),
                    ("Center Rectangle + Dimensions", lambda: self._activate_rectangle_tool(mode="center", use_dimensions=True)),
                ],
                compact_label="Rect",
            ),
            build_menu_tool_button(
                "circle",
                "Circle",
                "Circle\nClick: use the current circle method. Menu: center-point or center-radius.",
                lambda _checked=False: self._activate_circle_tool(),
                [
                    ("Center + Point", lambda: self._activate_circle_tool(mode="point")),
                    ("Center + Radius", lambda: self._activate_circle_tool(mode="radius")),
                ],
            ),
            build_tool_button(
                "polygon",
                "Polygon",
                "Polygon\nCreate a polygon sketch.",
                lambda _checked=False: self._activate_panel_geometry_tool("polygon"),
            ),
            build_menu_tool_button(
                "line",
                "Line",
                "Line\nClick: draw a line interactively. Menu: draw or enter coordinates.",
                lambda _checked=False: self._activate_panel_geometry_tool("line"),
                [
                    ("Draw Line", lambda: self._activate_panel_geometry_tool("line")),
                    ("Line by Coordinates", lambda: self._activate_line_by_coordinates()),
                ],
            ),
            build_tool_button(
                "freecurve",
                "Free Curve",
                "Free Curve\nCreate a freeform sketch curve.",
                lambda _checked=False: self._activate_panel_geometry_tool("freeform"),
                compact_label="Curve",
            ),
            build_tool_button(
                "import",
                "Import",
                "Import Geometry\nImport external geometry into the current model.",
                lambda _checked=False: self._trigger_panel_geometry_import(),
            ),
            build_tool_button(
                "extrude",
                "Extrude",
                "Extrude\nCreate an extruded solid from the current geometry.",
                lambda _checked=False: self._trigger_panel_geometry_extrude(),
            ),
        ]
        snap_buttons = [
            build_tool_button(
                "dimension",
                "Dimension",
                "Precision Sketch\nToggle dimension-driven sketch mode.",
                self._toggle_panel_precision,
                checkable=True,
                checked=bool(getattr(self.sketch_view, "precision_sketch_mode_enabled", False)),
                compact_label="Dim",
            ),
            build_tool_button(
                "snap_grid",
                "Grid Snap",
                "Grid Snap\nSnap sketch points to the grid.",
                lambda enabled: self._toggle_panel_snap(enabled, "set_snap_grid"),
                checkable=True,
                checked=True,
                compact_label="Grid",
            ),
            build_tool_button(
                "snap_endpoint",
                "Endpoint Snap",
                "Endpoint Snap\nSnap to existing endpoints while sketching.",
                lambda enabled: self._toggle_panel_snap(enabled, "set_endpoint_snap", "set_snap_endpoints"),
                checkable=True,
                checked=True,
                compact_label="End",
            ),
            build_tool_button(
                "snap_midpoint",
                "Midpoint Snap",
                "Midpoint Snap\nSnap to segment midpoints while sketching.",
                lambda enabled: self._toggle_panel_snap(enabled, "set_midpoint_snap"),
                checkable=True,
                checked=True,
                compact_label="Mid",
            ),
            build_tool_button(
                "snap_angle",
                "Angle Snap",
                "Angle Snap\nConstrain sketch angles to clean increments.",
                lambda enabled: self._toggle_panel_snap(enabled, "set_angle_snap"),
                checkable=True,
                checked=True,
                compact_label="Angle",
            ),
        ]
        self._create_buttons = list(create_buttons)
        self._snap_buttons = list(snap_buttons)
        for index, button in enumerate(self._create_buttons):
            self.create_layout.addWidget(button, index // 4, index % 4)

        self.snaps_layout = QGridLayout()
        self.snaps_layout.setContentsMargins(0, 0, 0, 0)
        self.snaps_layout.setSpacing(2)
        for index, button in enumerate(self._snap_buttons):
            self.snaps_layout.addWidget(button, index // 3, index % 3)

        self.tree = AssemblyTreeWidget()
        self.tree.setHeaderLabels(
            ["ID", "Part Name", "Geometry Type"]
        )
        self.tree.setColumnCount(3)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        self.tree.itemClicked.connect(self._on_item_clicked)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.tree.itemChanged.connect(self._on_tree_item_changed)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_tree_context_menu)
        self.tree.sketchReorderDropped.connect(self._handle_tree_sketch_reorder_drop)
        self.tree.renameShortcutRequested.connect(self._handle_tree_rename_shortcut)
        self.tree.setDragEnabled(True)
        self.tree.setAcceptDrops(True)
        self.tree.setDropIndicatorShown(True)
        self.tree.setDragDropMode(QAbstractItemView.InternalMove)
        self.tree.setDefaultDropAction(Qt.MoveAction)
        self.tree.setEditTriggers(QAbstractItemView.NoEditTriggers)
        header = self.tree.header()
        header.setMinimumSectionSize(28)
        header.setStretchLastSection(False)
        header.setCascadingSectionResizes(True)
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Interactive)
        # Tighten column widths so the tree fits in the narrow right panel
        # without forcing a horizontal scrollbar at the dock level.
        header.setDefaultSectionSize(40)
        header.setMinimumSectionSize(20)
        header.resizeSection(0, 30)
        header.resizeSection(2, 80)
        self.tree.setMinimumWidth(0)
        self.tree.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.part_actions_layout = QGridLayout()
        _apply_layout_metrics(self.part_actions_layout, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        self.btn_edit_shape = QPushButton("Edit Shape")
        self.btn_edit_shape.clicked.connect(self.edit_part_shape)
        self.btn_edit_shape.setToolTip(
            "Edit Shape\nLoad the selected part sketch for editing and update it with Confirm Part."
        )
        self.btn_edit_shape.setIcon(get_icon("edit", size=18))
        self.btn_copy = QPushButton("Copy")
        self.btn_copy.clicked.connect(self.copy_selected_entity)
        self.btn_copy.setToolTip("Copy\nCopy the selected part or sketch entity.")
        self.btn_copy.setIcon(get_icon("copy", size=18))
        self.btn_toggle_rigid = QPushButton("Toggle Rigid")
        self.btn_toggle_rigid.clicked.connect(self.toggle_rigid)
        self.btn_toggle_rigid.setToolTip("Toggle Rigid\nToggle the rigid flag for the selected part.")
        self.btn_toggle_rigid.setIcon(get_icon("rigid", size=18))
        self.btn_delete = QPushButton("Delete Part")
        self.btn_delete.clicked.connect(self.delete_part)
        self.btn_delete.setToolTip("Delete\nDelete the selected part.")
        self.btn_delete.setIcon(get_icon("delete", size=18))

        self._part_action_buttons = [
            self.btn_edit_shape,
            self.btn_copy,
            self.btn_delete,
        ]
        for index, button in enumerate(self._part_action_buttons):
            self.part_actions_layout.addWidget(button, 0, index)

        self.sketch_actions_layout = QGridLayout()
        _apply_layout_metrics(self.sketch_actions_layout, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        self.btn_edit_sketch = QPushButton("Edit Sketch")
        self.btn_edit_sketch.clicked.connect(self._edit_selected_sketch_entity)
        self.btn_edit_sketch.setToolTip("Edit Sketch\nEdit the selected sketch entity.")
        self.btn_edit_sketch.setIcon(get_icon("edit", size=18))
        self.btn_copy_sketch = QPushButton("Copy Sketch")
        self.btn_copy_sketch.clicked.connect(self._duplicate_selected_sketch_entity)
        self.btn_copy_sketch.setToolTip("Copy Sketch\nDuplicate the selected sketch entity.")
        self.btn_copy_sketch.setIcon(get_icon("copy", size=18))
        self.btn_delete_sketch = QPushButton("Delete Sketch")
        self.btn_delete_sketch.clicked.connect(self._delete_selected_sketch_entity)
        self.btn_delete_sketch.setToolTip("Delete Sketch\nDelete the selected sketch entity.")
        self.btn_delete_sketch.setIcon(get_icon("delete", size=18))
        self.btn_rename_sketch = QPushButton("Rename Sketch")
        self.btn_rename_sketch.clicked.connect(self._rename_selected_sketch_entity)
        self.btn_rename_sketch.setToolTip("Rename Sketch\nRename the selected sketch row.")
        self.btn_rename_sketch.setIcon(get_icon("edit", size=18))
        self.btn_move_up = QPushButton("Move Up")
        self.btn_move_up.clicked.connect(self._move_selected_sketch_entity_up)
        self.btn_move_up.setToolTip("Move Up\nMove the selected sketch up in its sketch list.")
        self.btn_move_up.setIcon(get_icon("move_up", size=18))
        self.btn_move_down = QPushButton("Move Down")
        self.btn_move_down.clicked.connect(self._move_selected_sketch_entity_down)
        self.btn_move_down.setToolTip("Move Down\nMove the selected sketch down in its sketch list.")
        self.btn_move_down.setIcon(get_icon("move_down", size=18))
        self._sketch_action_buttons = [
            self.btn_edit_sketch,
            self.btn_copy_sketch,
            self.btn_delete_sketch,
            self.btn_rename_sketch,
            self.btn_move_up,
            self.btn_move_down,
        ]
        for index, button in enumerate(self._sketch_action_buttons):
            self.sketch_actions_layout.addWidget(button, 0, index)
        for button in (
            self.btn_edit_shape,
            self.btn_copy,
            self.btn_delete,
            self.btn_edit_sketch,
            self.btn_copy_sketch,
            self.btn_delete_sketch,
            self.btn_rename_sketch,
            self.btn_move_up,
            self.btn_move_down,
        ):
            button.setProperty("dockIconButton", True)
            button.setMinimumSize(DOCK_ICON_BTN_MIN, 28)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_toggle_rigid.hide()

        # ---- Sub-tab assembly (Tools / Parts) ----
        # Tools tab combines Create + Snaps; Parts tab combines tree + Actions.
        self.geom_tabs = QTabWidget()
        self.geom_tabs.setDocumentMode(True)
        self.geom_tabs.tabBar().setExpanding(True)

        # Tools tab — compact rows: Create + Snaps + Quick Templates.
        tools_page = QWidget()
        tools_page_layout = QVBoxLayout(tools_page)
        tools_page_layout.setContentsMargins(4, 4, 4, 4)
        tools_page_layout.setSpacing(2)
        create_section_label = QLabel("Create")
        create_section_label.setObjectName("MinorStatusLabel")
        create_section_label.setContentsMargins(2, 0, 0, 0)
        tools_page_layout.addWidget(create_section_label)
        tools_page_layout.addLayout(self.create_layout)
        snaps_section_label = QLabel("Snaps & Precision")
        snaps_section_label.setObjectName("MinorStatusLabel")
        snaps_section_label.setContentsMargins(2, 4, 0, 0)
        tools_page_layout.addWidget(snaps_section_label)
        tools_page_layout.addLayout(self.snaps_layout)

        # Quick Templates — pre-built parametric shapes (I-beam, T-section, etc.)
        templates_section_label = QLabel("Quick Templates")
        templates_section_label.setObjectName("MinorStatusLabel")
        templates_section_label.setContentsMargins(2, 6, 0, 0)
        tools_page_layout.addWidget(templates_section_label)
        self.templates_layout = QGridLayout()
        self.templates_layout.setContentsMargins(0, 0, 0, 0)
        self.templates_layout.setSpacing(2)
        self._template_buttons = []
        for index, (kind, spec) in enumerate(_SHAPE_TEMPLATE_SPECS.items()):
            btn = QToolButton(self)
            btn.setText(spec["symbol"])
            btn.setToolTip(
                f"{spec['name']}\nInsert a parametric {spec['name'].lower()} "
                "as a new part."
            )
            btn.setStyleSheet(
                "QToolButton { font-size: 13px; font-weight: bold; padding: 1px; }"
            )
            # Fixed width so 4 columns of templates always fit in the narrow
            # right-panel without forcing a horizontal scrollbar.
            btn.setFixedWidth(46)
            btn.setMinimumHeight(26)
            btn.setMaximumHeight(30)
            btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            btn.clicked.connect(lambda _checked=False, k=kind: self._insert_template(k))
            self._template_buttons.append(btn)
            self.templates_layout.addWidget(btn, index // 4, index % 4)
        tools_page_layout.addLayout(self.templates_layout)
        tools_page_layout.addStretch(1)
        self.geom_tabs.addTab(tools_page, "Tools")

        # Parts tab — tree (flex) + context-sensitive action row at the bottom.
        parts_page = QWidget()
        parts_page_layout = QVBoxLayout(parts_page)
        parts_page_layout.setContentsMargins(0, 0, 0, 0)
        parts_page_layout.setSpacing(2)
        parts_page_layout.addWidget(self.tree, 1)

        # Action row stack — switches between empty hint / Part / Sketch actions.
        self.actions_stack = QStackedWidget()
        self.actions_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        empty_actions_page = QWidget()
        empty_actions_layout = QVBoxLayout(empty_actions_page)
        empty_actions_layout.setContentsMargins(8, 6, 8, 6)
        empty_actions_hint = QLabel("Select a part or sketch.")
        empty_actions_hint.setAlignment(Qt.AlignCenter)
        empty_actions_hint.setObjectName("SummaryLabel")
        empty_actions_hint.setWordWrap(True)
        # Don't let the hint's natural width inflate the stack's minimumSizeHint
        # — keep the empty page narrow.
        empty_actions_hint.setMinimumWidth(0)
        empty_actions_page.setMinimumWidth(0)
        empty_actions_layout.addWidget(empty_actions_hint)
        self.actions_stack.addWidget(empty_actions_page)

        part_actions_page = QWidget()
        part_actions_page_layout = QVBoxLayout(part_actions_page)
        part_actions_page_layout.setContentsMargins(6, 4, 6, 6)
        part_actions_page_layout.setSpacing(2)
        part_actions_page_layout.addLayout(self.part_actions_layout)
        self.actions_stack.addWidget(part_actions_page)

        sketch_actions_page = QWidget()
        sketch_actions_page_layout = QVBoxLayout(sketch_actions_page)
        sketch_actions_page_layout.setContentsMargins(6, 4, 6, 6)
        sketch_actions_page_layout.setSpacing(2)
        sketch_actions_page_layout.addLayout(self.sketch_actions_layout)
        self.actions_stack.addWidget(sketch_actions_page)

        parts_page_layout.addWidget(self.actions_stack, 0)
        self.geom_tabs.addTab(parts_page, "Parts")

        # Parts is the most common landing tab — open there by default.
        self.geom_tabs.setCurrentIndex(1)

        # Update Actions row whenever tree selection changes.
        self.tree.itemSelectionChanged.connect(self._update_actions_tab_context)

        layout.addWidget(self.geom_tabs, 1)
        # Legacy 2D/2.5D sections removed.
        self._external_container = QWidget()
        self._external_separator = _make_dock_separator()
        layout.addWidget(self._external_separator)
        self._external_layout = QVBoxLayout(self._external_container)
        _apply_layout_metrics(self._external_layout, margins=(0, 0, 0, 0), spacing=DOCK_SECTION_SPACING)
        # External container auto-sizes to its content; tabs above take flex space.
        layout.addWidget(self._external_container, 0)
        try:
            self.sketch_view.geometryChanged.connect(self.refresh)
        except Exception:
            pass
        self._update_action_button_density()
        self.refresh_external_section_visibility()
        self.refresh_project_setup()
        _finalize_dock_panel(self)
        self._update_action_button_density()

    def add_external_section(self, widget):
        if widget is None:
            return
        self._external_layout.addWidget(widget)
        self.refresh_external_section_visibility()

    def refresh_external_section_visibility(self):
        has_visible_content = False
        for index in range(self._external_layout.count()):
            item = self._external_layout.itemAt(index)
            widget = item.widget() if item is not None else None
            if widget is not None and not widget.isHidden():
                has_visible_content = True
                break
        self._external_container.setVisible(has_visible_content)
        self._external_separator.setVisible(has_visible_content)

    def set_project_state(self, project_state):
        self.project_state = _resolve_project_state(self.sketch_view, project_state)
        self.refresh_project_setup()
        self.refresh()

    def _update_action_button_density(self):
        width = int(self.contentsRect().width() or self.width() or 0)
        if width <= 0:
            return
        compact = width < 470
        # Tools tab: force compact icon-only mode at typical panel widths so the
        # Create row (7 buttons) and Snaps row (5 buttons) each fit on a SINGLE
        # row. Text labels only return at very wide panel widths.
        icon_only = width < 640
        action_compact = width < 960
        action_icon_only = width < 760
        # 7 Create buttons must fit in ~280px usable width. min_button_width
        # is tightened to 26 in icon-only mode (7 × 26 = 182, leaves room for
        # padding/spacing) so the Tools row never overflows horizontally.
        _reflow_button_grid(
            self.create_layout,
            getattr(self, "_create_buttons", []),
            max(1, width - 8),
            min_button_width=70 if not icon_only else 26,
            max_columns=7,
        )
        if hasattr(self, "snaps_layout"):
            _reflow_button_grid(
                self.snaps_layout,
                getattr(self, "_snap_buttons", []),
                max(1, width - 8),
                min_button_width=70 if not icon_only else 26,
                max_columns=5,
            )
        for button in list(getattr(self, "_create_buttons", [])) + list(getattr(self, "_snap_buttons", [])):
            full = str(button.property("responsiveFullText") or button.toolTip().split("\n", 1)[0] or "")
            compact_text = str(button.property("responsiveCompactText") or full or "")
            _set_responsive_button_text(
                button,
                full=full,
                compact=compact_text if compact else None,
                icon_only=icon_only,
            )
            button.setToolButtonStyle(Qt.ToolButtonIconOnly if icon_only else Qt.ToolButtonTextBesideIcon)
            # Compact button size for icon-only mode so they all fit on one row
            # within the narrow (~290 px) right panel without horizontal scroll.
            # setFixedWidth overrides Qt's internal sizeHint, which is what
            # the layout system uses to compute minimum column widths.
            if icon_only:
                button.setFixedWidth(36)
                button.setMaximumHeight(26)
            else:
                button.setMinimumWidth(0)
                button.setMaximumWidth(16777215)
                button.setMaximumHeight(30)
        _reflow_button_grid(
            self.part_actions_layout,
            getattr(self, "_part_action_buttons", []),
            max(1, width - 24),
            min_button_width=44 if action_icon_only else 124,
            max_columns=3,
            stretch_columns=not action_icon_only,
        )
        # Sketch actions: always fit all 6 buttons on a single row, even if it
        # means falling back to icon-only mode early. min_button_width is
        # tightened so 6 columns fit in the typical panel width.
        _reflow_button_grid(
            self.sketch_actions_layout,
            getattr(self, "_sketch_action_buttons", []),
            max(1, width - 24),
            min_button_width=40 if action_icon_only else 70,
            max_columns=6,
            stretch_columns=True,
        )
        _set_responsive_button_text(
            self.btn_edit_shape,
            full="Edit Shape",
            compact="Edit" if action_compact else None,
            icon_only=action_icon_only,
        )
        _set_responsive_button_text(
            self.btn_copy,
            full="Copy",
            compact="Copy",
            icon_only=action_icon_only,
        )
        _set_responsive_button_text(
            self.btn_delete,
            full="Delete Part",
            compact="Delete" if action_compact else None,
            icon_only=action_icon_only,
        )
        _set_responsive_button_text(
            self.btn_edit_sketch,
            full="Edit Sketch",
            compact="Edit" if action_compact else None,
            icon_only=action_icon_only,
        )
        _set_responsive_button_text(
            self.btn_copy_sketch,
            full="Copy Sketch",
            compact="Copy" if action_compact else None,
            icon_only=action_icon_only,
        )
        _set_responsive_button_text(
            self.btn_delete_sketch,
            full="Delete Sketch",
            compact="Delete" if action_compact else None,
            icon_only=action_icon_only,
        )
        _set_responsive_button_text(
            self.btn_rename_sketch,
            full="Rename Sketch",
            compact="Rename" if action_compact else None,
            icon_only=action_icon_only,
        )
        _set_responsive_button_text(
            self.btn_move_up,
            full="Move Up",
            compact="Up" if action_compact else None,
            icon_only=action_icon_only,
        )
        _set_responsive_button_text(
            self.btn_move_down,
            full="Move Down",
            compact="Down" if action_compact else None,
            icon_only=action_icon_only,
        )
        for button in getattr(self, "_part_action_buttons", []) + getattr(self, "_sketch_action_buttons", []):
            if action_icon_only:
                # setFixedWidth locks Qt's intrinsic sizeHint so the layout
                # cannot expand columns to fit each button's natural width.
                button.setFixedWidth(36)
                button.setMaximumHeight(26)
                button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            else:
                button.setMinimumWidth(0)
                button.setMaximumWidth(16777215)
                button.setMaximumHeight(30)
                button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def _activate_panel_geometry_tool(self, tool_name):
        main = self.window()
        handler = getattr(main, "_set_sketch_geometry_tool", None) if main is not None else None
        if callable(handler):
            handler(str(tool_name))
            return
        setter = getattr(self.sketch_view, "set_tool", None)
        if callable(setter):
            setter(str(tool_name))

    def _activate_rectangle_tool(self, mode=None, use_dimensions=None):
        if mode is not None or use_dimensions is not None:
            setter = getattr(self.sketch_view, "set_rectangle_draw_options", None)
            if callable(setter):
                setter(
                    mode if mode is not None else getattr(self.sketch_view, "_rect_draw_mode", "corner"),
                    getattr(self.sketch_view, "_rect_use_dimensions", False)
                    if use_dimensions is None
                    else bool(use_dimensions),
                )
            else:
                if mode is not None:
                    self.sketch_view._rect_draw_mode = "center" if str(mode).lower().startswith("center") else "corner"
                if use_dimensions is not None:
                    self.sketch_view._rect_use_dimensions = bool(use_dimensions)
        self._activate_panel_geometry_tool("rectangle")

    def _activate_circle_tool(self, mode=None):
        if mode is not None:
            setter = getattr(self.sketch_view, "set_circle_draw_mode", None)
            if callable(setter):
                setter(mode)
            else:
                self.sketch_view._circle_draw_mode = "radius" if str(mode).lower() == "radius" else "point"
        self._activate_panel_geometry_tool("circle")

    def _activate_line_by_coordinates(self):
        """Open the coordinate dialog and create a line directly from the
        entered start/end coordinates — no interactive mouse clicks needed."""
        view = self.sketch_view
        params = view._prompt_line_params()
        if params is None:
            return
        start = params["start"]
        end = params["end"]
        # Ensure we're in the Geometry stage with the line tool active.
        main = self.window()
        handler = getattr(main, "_set_sketch_geometry_tool", None) if main is not None else None
        if callable(handler):
            handler("line")
        else:
            setter = getattr(view, "set_tool", None)
            if callable(setter):
                setter("line")
        # Push undo, build the line meta, and add the sketch.
        view.push_undo_state()
        meta = view._line_meta_from_points(start, end)
        pts = view._build_points_from_meta(meta, fallback_points=[])
        view._append_sketch(pts, meta=meta)
        view.command_last_point = end
        view.geometryChanged.emit()
        view.redraw()
        if main is not None and hasattr(main, "statusBar"):
            main.statusBar().showMessage(
                f"Line created: ({start[0]:.2f}, {start[1]:.2f}) → ({end[0]:.2f}, {end[1]:.2f})",
                6000,
            )

    def _trigger_panel_geometry_import(self):
        main = self.window()
        handler = getattr(main, "_import_geometry_from_toolbar", None) if main is not None else None
        if callable(handler):
            handler()
            return
        for name in ("import_geometry", "import_dxf"):
            target = getattr(self.sketch_view, name, None)
            if callable(target):
                target()
                return

    def _trigger_panel_geometry_extrude(self):
        main = self.window()
        handler = getattr(main, "_activate_geometry_extrude", None) if main is not None else None
        if callable(handler):
            handler()

    def _toggle_panel_precision(self, enabled):
        main = self.window()
        enabled = bool(enabled)
        handler = getattr(main, "_set_precision_sketch_mode", None) if main is not None else None
        tool_handler = getattr(main, "_set_sketch_geometry_tool", None) if main is not None else None
        if callable(handler):
            handler(enabled, False)
        else:
            target = getattr(self.sketch_view, "set_precision_sketch_mode", None)
            if callable(target):
                target(enabled)
        if enabled:
            if callable(tool_handler):
                tool_handler("dimension")
            else:
                setter = getattr(self.sketch_view, "set_tool", None)
                if callable(setter):
                    setter("dimension")
        else:
            current_tool = str(getattr(self.sketch_view, "tool", getattr(self.sketch_view, "current_tool", "")) or "")
            if current_tool == "dimension":
                if callable(tool_handler):
                    tool_handler("select")
                else:
                    setter = getattr(self.sketch_view, "set_tool", None)
                    if callable(setter):
                        setter("select")

    def _toggle_panel_snap(self, enabled, *method_names):
        for name in method_names:
            target = getattr(self.sketch_view, str(name), None)
            if callable(target):
                target(bool(enabled))
                return

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_action_button_density()

    def refresh_project_setup(self):
        analysis = str(getattr(self.project_state, "analysis_type", "static") or "static").lower()
        dimension = str(getattr(self.project_state, "dimension", "2D") or "2D").upper()
        analysis_index = self.analysis_type_combo.findData(analysis)
        self.analysis_type_combo.blockSignals(True)
        self.analysis_type_combo.setCurrentIndex(analysis_index if analysis_index >= 0 else 0)
        self.analysis_type_combo.blockSignals(False)
        dimension_index = self.dimension_combo.findData(dimension)
        self.dimension_combo.blockSignals(True)
        self.dimension_combo.setCurrentIndex(dimension_index if dimension_index >= 0 else 0)
        self.dimension_combo.blockSignals(False)

    def _on_analysis_type_changed(self):
        main = self.window()
        if main is not None and hasattr(main, "set_project_analysis_type"):
            main.set_project_analysis_type(self.analysis_type_combo.currentData())

    def _on_dimension_changed(self):
        main = self.window()
        if main is not None and hasattr(main, "set_project_dimension"):
            main.set_project_dimension(self.dimension_combo.currentData())

    def _parts_source(self):
        parts = getattr(self.sketch_view, "parts", None)
        if isinstance(parts, list):
            return parts
        return list(getattr(self.project_state, "parts", []) or [])

    def _find_part_by_id(self, part_id):
        return next((p for p in self._parts_source() if p.id == part_id), None)

    def _tree_key_from_item(self, item):
        if item is None:
            return None
        data = item.data(0, Qt.UserRole)
        if isinstance(data, dict):
            kind = data.get("kind")
            if kind == "part":
                return ("part", data.get("id"))
            if kind == "primitive":
                return ("primitive", data.get("id"))
            if kind == "part_sketch":
                return ("part_sketch", data.get("part_id"), data.get("sketch_index"))
            if kind == "active_sketch":
                return ("active_sketch", data.get("sketch_index"))
            return None
        return ("part", data)

    def _iter_tree_items(self):
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            if top is None:
                continue
            yield top
            stack = [top]
            while stack:
                node = stack.pop()
                for j in range(node.childCount()):
                    child = node.child(j)
                    if child is None:
                        continue
                    yield child
                    stack.append(child)

    def _restore_tree_selection(self, key):
        if not key:
            return
        self._suppress_selection = True
        try:
            for item in self._iter_tree_items():
                if self._tree_key_from_item(item) == key:
                    self.tree.setCurrentItem(item)
                    item.setSelected(True)
                    parent = item.parent()
                    while parent is not None:
                        parent.setExpanded(True)
                        parent = parent.parent()
                    return
        finally:
            self._suppress_selection = False

    def _sketch_row_name(self, sketch_index, meta, points):
        meta_type = str((meta or {}).get("type", "polyline")).lower()
        label_map = {
            "line": "Line",
            "rectangle": "Rectangle",
            "circle": "Circle",
            "slot": "Slot",
            "polygon": "Polygon",
            "polyline": "Polyline",
            "arc": "Arc",
        }
        shape_name = label_map.get(meta_type, meta_type.title() if meta_type else "Sketch")
        extras = []
        if meta_type == "polygon":
            try:
                extras.append(f"{int((meta or {}).get('sides', 0))} sides")
            except Exception:
                pass
        if points:
            try:
                extras.append(f"{max(0, len(points) - 1)} seg")
            except Exception:
                pass
        suffix = f" ({', '.join(extras)})" if extras else ""
        custom_name = str((meta or {}).get("name", "") or "").strip()
        if custom_name:
            return f"Sketch {int(sketch_index) + 1}: {custom_name} [{shape_name}{suffix}]"
        return f"Sketch {int(sketch_index) + 1}: {shape_name}{suffix}"

    def _sketch_row_visible(self, meta):
        try:
            return bool((meta or {}).get("visible", True))
        except Exception:
            return True

    def _apply_sketch_visibility_cell(self, item, visible):
        if not visible:
            muted = QColor(150, 150, 150)
            item.setForeground(self._col_part, muted)
            item.setForeground(self._col_geom_type, muted)

    def _sketch_selection_key(self, info, sketch_index=None):
        if not info:
            return None
        idx = info.get("sketch_index") if sketch_index is None else sketch_index
        if info.get("owner_type") == "part":
            part = info.get("owner_part")
            return ("part_sketch", getattr(part, "id", None), int(idx))
        return ("active_sketch", int(idx))

    def _resolve_selected_sketch_entity(self, item=None):
        item = item or self.tree.currentItem()
        if item is None:
            return None
        data = item.data(0, Qt.UserRole)
        if not isinstance(data, dict):
            return None
        kind = data.get("kind")
        if kind == "active_sketch":
            return {
                "kind": kind,
                "owner_type": "sketch",
                "owner_part": None,
                "sketch_index": int(data.get("sketch_index", -1)),
                "data": data,
            }
        if kind == "part_sketch":
            part = self._find_part_by_id(data.get("part_id"))
            if part is None:
                return None
            return {
                "kind": kind,
                "owner_type": "part",
                "owner_part": part,
                "sketch_index": int(data.get("sketch_index", -1)),
                "data": data,
            }
        return None

    def _current_custom_sketch_name(self, info):
        if not info:
            return ""
        try:
            sketches, metas, _, _ = self.sketch_view._owner_collections(info["owner_type"], info["owner_part"])
        except Exception:
            return ""
        idx = int(info.get("sketch_index", -1))
        if idx < 0 or idx >= len(sketches):
            return ""
        meta = metas[idx] if idx < len(metas) else {}
        return str((meta or {}).get("name", "") or "").strip()

    def _begin_inline_rename_selected_sketch(self):
        item = self.tree.currentItem()
        info = self._resolve_selected_sketch_entity(item)
        if not item or not info:
            return False
        if int(item.columnCount()) <= self._col_part:
            return False
        self._inline_rename_ctx = {
            "key": self._tree_key_from_item(item),
            "old_display": str(item.text(self._col_part) or ""),
            "old_custom": self._current_custom_sketch_name(info),
        }
        self.tree.editItem(item, self._col_part)
        return True

    def _normalize_inline_sketch_name(self, raw_text, info, ctx):
        raw = str(raw_text or "").strip()
        if ctx and raw == str(ctx.get("old_display", "")):
            return str(ctx.get("old_custom", "") or "").strip()
        if not raw:
            return ""
        try:
            prefix = f"Sketch {int(info.get('sketch_index', -1)) + 1}:"
        except Exception:
            prefix = ""
        if prefix and raw.startswith(prefix):
            tail = raw[len(prefix):].strip()
            if not tail:
                return ""
            if tail.endswith("]") and " [" in tail:
                try:
                    base, _ = tail.rsplit(" [", 1)
                    base = str(base or "").strip()
                    if base:
                        return base
                except Exception:
                    pass
            return tail
        return raw

    def _prompt_copy_offset(self, title="Copy"):
        unit = getattr(self.sketch_view, "current_unit", "") or "m"
        dx, ok = QInputDialog.getDouble(self, title, f"Offset X ({unit}):", 10.0, -1e9, 1e9, 3)
        if not ok:
            return None
        dy, ok = QInputDialog.getDouble(self, title, f"Offset Y ({unit}):", 10.0, -1e9, 1e9, 3)
        if not ok:
            return None
        return float(dx), float(dy)

    def _edit_selected_sketch_entity(self):
        info = self._resolve_selected_sketch_entity()
        if not info:
            return False
        main = self.window()
        if info.get("owner_type") == "part" and info.get("owner_part") is not None:
            if main is not None and hasattr(main, "enter_sketch_edit_mode"):
                return bool(main.enter_sketch_edit_mode(getattr(info["owner_part"], "id", None)))
        if info.get("owner_type") == "sketch":
            try:
                self.sketch_view.set_module("Part")
            except Exception:
                pass
            if main is not None and hasattr(main, "_set_precision_sketch_mode"):
                try:
                    main._set_precision_sketch_mode(True, announce=False)
                except Exception:
                    pass
            else:
                try:
                    self.sketch_view.set_dimensions_visible(True)
                except Exception:
                    pass
            try:
                self.sketch_view.set_tool("select")
            except Exception:
                pass
            try:
                self.sketch_view.redraw()
            except Exception:
                pass
            if main is not None and hasattr(main, "statusBar"):
                main.statusBar().showMessage(
                    "Edit directly on the sketch. Double-click a dimension to edit, or use Smart Dimension to add one.",
                    7000,
                )
            self.refresh()
            return True
        try:
            self.sketch_view.set_module("Part")
        except Exception:
            pass
        ok = False
        if hasattr(self.sketch_view, "edit_sketch_entity_by_index"):
            ok = bool(
                self.sketch_view.edit_sketch_entity_by_index(
                    info["sketch_index"],
                    owner_type=info["owner_type"],
                    owner_part=info["owner_part"],
                )
            )
        if ok:
            self.refresh()
        return ok

    def _delete_selected_sketch_entity(self):
        info = self._resolve_selected_sketch_entity()
        if not info:
            return False
        if hasattr(self.sketch_view, "delete_sketch_entity_by_index"):
            ok = bool(
                self.sketch_view.delete_sketch_entity_by_index(
                    info["sketch_index"],
                    owner_type=info["owner_type"],
                    owner_part=info["owner_part"],
                    confirm=True,
                )
            )
            if ok:
                self.refresh()
            return ok
        return False

    def _duplicate_selected_sketch_entity(self):
        info = self._resolve_selected_sketch_entity()
        if not info:
            return False
        offset = self._prompt_copy_offset("Duplicate Sketch")
        if not offset:
            return False
        dx, dy = offset
        if hasattr(self.sketch_view, "duplicate_sketch_entity_by_index"):
            ok = bool(
                self.sketch_view.duplicate_sketch_entity_by_index(
                    info["sketch_index"],
                    dx=dx,
                    dy=dy,
                    owner_type=info["owner_type"],
                    owner_part=info["owner_part"],
                )
            )
            if ok:
                self.refresh()
            return ok
        return False

    def _rename_selected_sketch_entity(self):
        if self._begin_inline_rename_selected_sketch():
            return True
        info = self._resolve_selected_sketch_entity()
        if not info:
            return False
        sketches, metas, _, _ = self.sketch_view._owner_collections(info["owner_type"], info["owner_part"])
        idx = int(info["sketch_index"])
        if idx < 0 or idx >= len(sketches):
            return False
        meta = metas[idx] if idx < len(metas) else {}
        current_name = str((meta or {}).get("name", "") or "").strip()
        name, ok = QInputDialog.getText(
            self,
            "Rename Sketch",
            "Sketch name (leave blank to reset to automatic label):",
            text=current_name,
        )
        if not ok:
            return False
        if hasattr(self.sketch_view, "rename_sketch_entity_by_index"):
            ok = bool(
                self.sketch_view.rename_sketch_entity_by_index(
                    idx,
                    name,
                    owner_type=info["owner_type"],
                    owner_part=info["owner_part"],
                )
            )
            if ok:
                self.refresh()
            return ok
        return False

    def _handle_tree_rename_shortcut(self):
        self._begin_inline_rename_selected_sketch()

    def _on_tree_item_changed(self, item, column):
        if self._suppress_item_changed:
            return
        if item is None or column != self._col_part:
            return
        ctx = self._inline_rename_ctx
        if not isinstance(ctx, dict):
            return
        if self._tree_key_from_item(item) != ctx.get("key"):
            return

        self._inline_rename_ctx = None
        info = self._resolve_selected_sketch_entity(item)
        if not info:
            self.refresh()
            return

        desired_name = self._normalize_inline_sketch_name(item.text(self._col_part), info, ctx)
        old_name = str(ctx.get("old_custom", "") or "").strip()
        if desired_name == old_name:
            self.refresh()
            self._restore_tree_selection(self._sketch_selection_key(info))
            return

        if hasattr(self.sketch_view, "rename_sketch_entity_by_index"):
            ok = bool(
                self.sketch_view.rename_sketch_entity_by_index(
                    info["sketch_index"],
                    desired_name,
                    owner_type=info["owner_type"],
                    owner_part=info["owner_part"],
                )
            )
            self.refresh()
            if ok:
                self._restore_tree_selection(self._sketch_selection_key(info))
            return
        self.refresh()

    def _move_selected_sketch_entity(self, delta):
        try:
            delta = int(delta)
        except Exception:
            return False
        if delta == 0:
            return False
        info = self._resolve_selected_sketch_entity()
        if not info:
            QMessageBox.information(self, "Move Sketch", "Select a sketch row first.")
            return False
        sketches, _, _, _ = self.sketch_view._owner_collections(info["owner_type"], info["owner_part"])
        count = len(sketches)
        if count <= 1:
            return False
        idx = int(info["sketch_index"])
        new_idx = idx + delta
        if new_idx < 0 or new_idx >= count:
            return False
        order = list(range(count))
        order[idx], order[new_idx] = order[new_idx], order[idx]
        if not hasattr(self.sketch_view, "reorder_sketch_entities"):
            return False
        ok = bool(
            self.sketch_view.reorder_sketch_entities(
                order,
                owner_type=info["owner_type"],
                owner_part=info["owner_part"],
            )
        )
        self.refresh()
        if ok:
            self._restore_tree_selection(self._sketch_selection_key(info, sketch_index=new_idx))
        return ok

    def _move_selected_sketch_entity_up(self):
        return self._move_selected_sketch_entity(-1)

    def _move_selected_sketch_entity_down(self):
        return self._move_selected_sketch_entity(1)

    def _selected_sketch_entity_visible(self):
        info = self._resolve_selected_sketch_entity()
        if not info:
            return True
        if hasattr(self.sketch_view, "is_sketch_entity_visible_by_index"):
            try:
                return bool(
                    self.sketch_view.is_sketch_entity_visible_by_index(
                        info["sketch_index"],
                        owner_type=info["owner_type"],
                        owner_part=info["owner_part"],
                    )
                )
            except Exception:
                return True
        return True

    def _set_selected_sketch_entity_visibility(self, visible):
        info = self._resolve_selected_sketch_entity()
        if not info:
            return False
        if hasattr(self.sketch_view, "set_sketch_entity_visible_by_index"):
            ok = bool(
                self.sketch_view.set_sketch_entity_visible_by_index(
                    info["sketch_index"],
                    visible=visible,
                    owner_type=info["owner_type"],
                    owner_part=info["owner_part"],
                )
            )
            if ok:
                self.refresh()
            return ok
        return False

    def _toggle_selected_sketch_entity_visibility(self):
        return self._set_selected_sketch_entity_visibility(not self._selected_sketch_entity_visible())

    def _on_item_clicked(self, item, column):
        data = item.data(0, Qt.UserRole)
        if not isinstance(data, dict):
            return

    def _handle_tree_sketch_reorder_drop(self, payload):
        if not isinstance(payload, dict):
            return
        if not payload.get("valid"):
            self.refresh()
            QMessageBox.information(
                self,
                "Reorder Sketches",
                "Sketches can only be reordered within the same group (Active Sketches or the same Part).",
            )
            return

        owner_type = str(payload.get("owner_type") or "sketch")
        part = None
        if owner_type == "part":
            part = self._find_part_by_id(payload.get("part_id"))
            if part is None:
                self.refresh()
                return
        order = list(payload.get("order") or [])
        if not hasattr(self.sketch_view, "reorder_sketch_entities"):
            self.refresh()
            return
        ok = bool(
            self.sketch_view.reorder_sketch_entities(
                order,
                owner_type=owner_type,
                owner_part=part,
            )
        )
        self.refresh()
        if not ok:
            return

        dragged_index = payload.get("dragged_index")
        try:
            if dragged_index is not None:
                new_idx = int(order.index(int(dragged_index)))
                if owner_type == "part" and part is not None:
                    self._restore_tree_selection(("part_sketch", part.id, new_idx))
                else:
                    self._restore_tree_selection(("active_sketch", new_idx))
        except Exception:
            pass

    def _show_tree_context_menu(self, pos):
        item = self.tree.itemAt(pos)
        if item is None:
            return
        self.tree.setCurrentItem(item)
        data = item.data(0, Qt.UserRole)
        if not isinstance(data, dict):
            return
        kind = data.get("kind")
        menu = QMenu(self)

        if kind in ("active_sketch", "part_sketch"):
            menu.addAction("Edit Sketch", self._edit_selected_sketch_entity)
            menu.addAction("Rename Sketch", self._rename_selected_sketch_entity)
            menu.addAction("Move Sketch Up", self._move_selected_sketch_entity_up)
            menu.addAction("Move Sketch Down", self._move_selected_sketch_entity_down)
            if self._selected_sketch_entity_visible():
                menu.addAction("Hide Sketch", lambda: self._set_selected_sketch_entity_visibility(False))
            else:
                menu.addAction("Show Sketch", lambda: self._set_selected_sketch_entity_visibility(True))
            menu.addSeparator()
            menu.addAction("Copy Sketch", self._duplicate_selected_sketch_entity)
            menu.addAction("Delete Sketch", self._delete_selected_sketch_entity)
        elif kind == "part":
            part = self._find_part_by_id(data.get("id"))
            if part is None:
                return
            if hasattr(self.sketch_view, "_populate_part_context_actions"):
                self.sketch_view._populate_part_context_actions(menu, part)
            else:
                menu.addAction("Edit Shape", self.edit_part_shape)
                menu.addAction("Copy Part", self.copy_selected_entity)
                menu.addAction("Delete Part", self.delete_part)
        elif kind == "primitive":
            main = self.window()
            if main and hasattr(main, "_edit_selected_primitive"):
                menu.addAction("Edit Primitive", main._edit_selected_primitive)
            if main and hasattr(main, "_delete_selected_shapes"):
                menu.addAction("Delete Primitive", main._delete_selected_shapes)
        else:
            return

        if not menu.isEmpty():
            menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _insert_template(self, kind):
        """Prompt the user for template dimensions, generate the geometry, and
        add it as a new Part. Hollow templates (Pipe, RHS, Plate-with-Hole)
        produce an outer solid plus a void child part."""
        spec = _SHAPE_TEMPLATE_SPECS.get(kind)
        if not spec:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Insert {spec['name']}")
        dialog.setMinimumWidth(320)
        main_layout = QVBoxLayout(dialog)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(14, 14, 14, 14)

        unit = getattr(self.sketch_view, "current_unit", "") or ""

        # --- Start Position group ---
        from PySide6.QtWidgets import QGroupBox
        pos_group = QGroupBox("Start Position")
        pos_form = QFormLayout(pos_group)
        pos_form.setContentsMargins(10, 14, 10, 8)
        pos_form.setSpacing(6)

        start_x_spin = QDoubleSpinBox()
        start_x_spin.setRange(-1e9, 1e9)
        start_x_spin.setDecimals(4)
        start_x_spin.setValue(0.0)
        start_x_spin.setMinimumHeight(28)
        start_x_spin.setToolTip("X coordinate of the start position")
        pos_form.addRow(f"Start X ({unit}):" if unit else "Start X:", start_x_spin)

        start_y_spin = QDoubleSpinBox()
        start_y_spin.setRange(-1e9, 1e9)
        start_y_spin.setDecimals(4)
        start_y_spin.setValue(0.0)
        start_y_spin.setMinimumHeight(28)
        start_y_spin.setToolTip("Y coordinate of the start position")
        pos_form.addRow(f"Start Y ({unit}):" if unit else "Start Y:", start_y_spin)

        main_layout.addWidget(pos_group)

        # --- Dimensions group ---
        dim_group = QGroupBox("Dimensions")
        dim_form = QFormLayout(dim_group)
        dim_form.setContentsMargins(10, 14, 10, 8)
        dim_form.setSpacing(6)

        spinboxes = []
        for label, default, min_v, max_v in spec["params"]:
            sb = QDoubleSpinBox()
            sb.setRange(float(min_v), float(max_v))
            sb.setDecimals(3)
            sb.setValue(float(default))
            sb.setMinimumHeight(28)
            if unit:
                sb.setSuffix(f" {unit}")
            dim_form.addRow(label, sb)
            spinboxes.append(sb)

        main_layout.addWidget(dim_group)

        # --- Buttons ---
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        main_layout.addWidget(buttons)

        if dialog.exec() != QDialog.Accepted:
            return

        start_x = start_x_spin.value()
        start_y = start_y_spin.value()

        try:
            values = [sb.value() for sb in spinboxes]
            result = spec["generator"](*values)
        except Exception as exc:
            QMessageBox.warning(self, "Template", f"Could not build template: {exc}")
            return

        def _offset_points(pts, dx, dy):
            """Shift all points by (dx, dy)."""
            if not pts:
                return pts
            return [(x + dx, y + dy) for x, y in pts]

        if spec.get("hollow"):
            if not (isinstance(result, tuple) and len(result) == 2):
                QMessageBox.warning(
                    self, "Template",
                    "Hollow template generator must return (outer, hole) tuple.",
                )
                return
            outer_pts, hole_pts = result
            if (not outer_pts or not hole_pts
                    or len(outer_pts) < 3 or len(hole_pts) < 3):
                QMessageBox.warning(self, "Template", "Template produced invalid geometry.")
                return
            # Apply start-position offset
            outer_pts = _offset_points(outer_pts, start_x, start_y)
            hole_pts = _offset_points(hole_pts, start_x, start_y)
            try:
                # 1. Add the outer solid part.
                self.sketch_view.add_imported_geometry(
                    [outer_pts], convert_closed=True, base_name=spec["name"]
                )
                outer_part = self.sketch_view.parts[-1] if self.sketch_view.parts else None
                # 2. Add the hole as a child part and flip it to a void so
                #    rebuild_display_geometry subtracts it from the outer.
                self.sketch_view.add_imported_geometry(
                    [hole_pts], convert_closed=True, base_name=f"{spec['name']} Hole"
                )
                if (outer_part is not None
                        and self.sketch_view.parts
                        and self.sketch_view.parts[-1] is not outer_part):
                    inner_part = self.sketch_view.parts[-1]
                    inner_part.is_void = True
                    inner_part.parent_id = outer_part.id
                    self.sketch_view.rebuild_display_geometry()
                    self.sketch_view.partsChanged.emit()
                    self.sketch_view.geometryChanged.emit()
            except Exception as exc:
                QMessageBox.warning(self, "Template", f"Could not insert template: {exc}")
            return

        # Solid (non-hollow) template
        points = result
        if not points or len(points) < 3:
            QMessageBox.warning(self, "Template", "Template produced no geometry.")
            return
        # Apply start-position offset
        points = _offset_points(points, start_x, start_y)
        try:
            self.sketch_view.add_imported_geometry(
                [points], convert_closed=True, base_name=spec["name"]
            )
        except Exception as exc:
            QMessageBox.warning(self, "Template", f"Could not insert template: {exc}")

    def _update_actions_tab_context(self):
        """Switch the Actions sub-tab page based on what's selected in the tree."""
        stack = getattr(self, "actions_stack", None)
        if stack is None:
            return
        item = self.tree.currentItem() if hasattr(self, "tree") else None
        if not item:
            stack.setCurrentIndex(0)
            return
        data = item.data(0, Qt.UserRole)
        if isinstance(data, dict):
            kind = data.get("kind")
            if kind in ("part", "primitive"):
                stack.setCurrentIndex(1)
                return
            if kind in ("part_sketch", "active_sketch"):
                stack.setCurrentIndex(2)
                return
        stack.setCurrentIndex(0)

    def _on_selection_changed(self):
        if self._suppress_selection:
            return
        item = self.tree.currentItem()
        if not item:
            return
        data = item.data(0, Qt.UserRole)
        main = self.window()
        inspector = getattr(main, "property_inspector", None) if main is not None else None
        if isinstance(data, dict):
            kind = data.get("kind")
            if kind == "primitive":
                self.sketch_view.set_selected_part(None)
                if inspector is not None:
                    inspector.set_selection_payload(None)
                main = self.window()
                if main and hasattr(main, "_select_primitive_by_id"):
                    main._select_primitive_by_id(data.get("id"))
                return
            if kind == "part_sketch":
                self.sketch_view.set_selected_part(data.get("part_id"))
                if inspector is not None:
                    inspector.set_selection_payload(
                        {
                            "kind": "part_sketch",
                            "part_id": data.get("part_id"),
                            "sketch_index": data.get("sketch_index"),
                            "stage": ProjectStage.GEOMETRY,
                        }
                    )
                return
            if kind == "active_sketch":
                try:
                    edit_target = self.sketch_view.get_part_shape_edit_target()
                except Exception:
                    edit_target = None
                if edit_target is not None:
                    self.sketch_view.set_selected_part(getattr(edit_target, "id", None))
                if inspector is not None:
                    inspector.set_selection_payload(
                        {
                            "kind": "active_sketch",
                            "sketch_index": data.get("sketch_index"),
                            "stage": ProjectStage.GEOMETRY,
                        }
                    )
                return
            if kind not in ("part",):
                if inspector is not None:
                    inspector.set_selection_payload(None)
                return
        if isinstance(data, dict):
            part_id = data.get("id")
        else:
            part_id = data
        self.sketch_view.set_selected_part(part_id)
        if inspector is not None and part_id not in (None, "", -1):
            inspector.set_selection_payload(
                {
                    "kind": "part",
                    "part_id": int(part_id),
                    "stage": ProjectStage.GEOMETRY,
                }
            )

    def _sync_material_colors(self):
        color_map = self.sketch_view.material_color_map
        default_colors = [
            QColor(214, 226, 240),
            QColor(240, 214, 214),
            QColor(214, 240, 214),
            QColor(240, 232, 214),
            QColor(232, 214, 240),
            QColor(214, 240, 240),
        ]
        next_idx = len(color_map)
        for serial in sorted(self.project_state.materials.keys()):
            if serial not in color_map:
                color_map[serial] = default_colors[next_idx % len(default_colors)]
                next_idx += 1

    def _material_color_for(self, material_id):
        if material_id is None:
            return None
        self._sync_material_colors()
        color_map = self.sketch_view.material_color_map
        if material_id not in color_map:
            palette = [
                QColor(214, 226, 240),
                QColor(240, 214, 214),
                QColor(214, 240, 214),
                QColor(240, 232, 214),
                QColor(232, 214, 240),
                QColor(214, 240, 240),
            ]
            color_map[material_id] = palette[len(color_map) % len(palette)]
        return color_map.get(material_id)

    def _apply_material_color(self, item, material_id):
        color = self._material_color_for(material_id)
        if color is None:
            return
        swatch = QColor(color)
        swatch = swatch.lighter(135)
        swatch.setAlpha(200)
        lum = 0.299 * swatch.red() + 0.587 * swatch.green() + 0.114 * swatch.blue()
        text_color = QColor(20, 20, 20) if lum > 160 else QColor(250, 250, 250)
        for col in range(self.tree.columnCount()):
            item.setBackground(col, swatch)
            item.setForeground(col, text_color)

    def _on_item_double_clicked(self, item, column):
        data = item.data(0, Qt.UserRole)
        if isinstance(data, dict):
            kind = data.get("kind")
            if kind in ("active_sketch", "part_sketch"):
                if column == self._col_part:
                    self._begin_inline_rename_selected_sketch()
                    return
                if column == self._col_geom_type:
                    self._edit_selected_sketch_entity()
                return
            if kind == "primitive":
                return
            if kind != "part":
                return
        part_id = data.get("id") if isinstance(data, dict) else data
        part = self._find_part_by_id(part_id)
        main = self.window()
        if not part or not main or not hasattr(main, "properties_panel"):
            return
        if column == self._col_part:
            self.edit_part_shape()

    def select_part(self, part_id):
        self._suppress_selection = True
        try:
            if part_id is None:
                self.tree.clearSelection()
                return
            for idx in range(self.tree.topLevelItemCount()):
                item = self.tree.topLevelItem(idx)
                data = item.data(0, Qt.UserRole)
                if isinstance(data, dict):
                    if data.get("kind") == "part" and data.get("id") == part_id:
                        self.tree.setCurrentItem(item)
                        break
                elif data == part_id:
                    self.tree.setCurrentItem(item)
                    break
        finally:
            self._suppress_selection = False

    def toggle_rigid(self):
        selected = self.tree.currentItem()
        if not selected:
            QMessageBox.warning(self, "Warning", "No part selected.")
            return
        data = selected.data(0, Qt.UserRole)
        if isinstance(data, dict):
            kind = data.get("kind")
            if kind == "part_sketch":
                part_id = data.get("part_id")
            elif kind == "part":
                part_id = data.get("id")
            else:
                QMessageBox.warning(self, "Warning", "Select a part row (or one of its sketch rows).")
                return
        else:
            part_id = data
        part = self._find_part_by_id(part_id)
        if part:
            self.sketch_view.push_undo_state()
            part.is_rigid = not part.is_rigid
            print(f"Part '{part.name}' is_rigid set to {part.is_rigid}")
            self.refresh()
            self.sketch_view.redraw()
        else:
            QMessageBox.warning(self, "Warning", "Selected part no longer exists. Refresh and reselect.")

    def refresh(self):
        """Update tree with current parts."""
        selected_key = None
        selected = self.tree.currentItem()
        if selected is not None:
            selected_key = self._tree_key_from_item(selected)
        self._inline_rename_ctx = None
        self._suppress_item_changed = True
        try:
            self.tree.clear()

            active_sketches = list(getattr(self.sketch_view, "sketches", []) or [])
            active_metas = list(getattr(self.sketch_view, "sketch_meta", []) or [])
            if active_sketches:
                active_header = QTreeWidgetItem(["", "Active Sketches", "Sketch Group"])
                active_header.setFlags(Qt.ItemIsEnabled | Qt.ItemIsDropEnabled)
                active_header.setData(0, Qt.UserRole, {"kind": "active_sketch_header"})
                self.tree.addTopLevelItem(active_header)
                for idx, sketch in enumerate(active_sketches):
                    meta = active_metas[idx] if idx < len(active_metas) else {}
                    name = self._sketch_row_name(idx, meta, sketch)
                    visible = self._sketch_row_visible(meta)
                    child = QTreeWidgetItem(["", name, "Active Sketch"])
                    child.setData(0, Qt.UserRole, {"kind": "active_sketch", "sketch_index": idx})
                    child.setFlags(
                        Qt.ItemIsEnabled
                        | Qt.ItemIsSelectable
                        | Qt.ItemIsDragEnabled
                        | Qt.ItemIsDropEnabled
                        | Qt.ItemIsEditable
                    )
                    for col in range(self.tree.columnCount()):
                        child.setBackground(col, QColor(247, 247, 247))
                    self._apply_sketch_visibility_cell(child, visible)
                    active_header.addChild(child)
                active_header.setExpanded(True)

            for part in self._parts_source():
                geom_type = "Part"
                if part.is_void:
                    geom_type = "Void"
                elif getattr(self.sketch_view, "is_generated_feature_part", None) and self.sketch_view.is_generated_feature_part(part):
                    geom_type = "Feature"
                elif part.parent_id is not None:
                    geom_type = "Nested"
                elif getattr(part, "is_direct_edit", False):
                    geom_type = "Direct"
                item = QTreeWidgetItem([str(part.id), part.name, geom_type])
                item.setData(0, Qt.UserRole, {"kind": "part", "id": part.id})
                item.setToolTip(
                    0,
                    f"Part ID: {part.id}\nPart Name: {part.name}\nGeometry Type: {geom_type}",
                )
                item.setFlags((item.flags() | Qt.ItemIsDropEnabled) & ~Qt.ItemIsDragEnabled)
                if geom_type != "Part":
                    for col in range(self.tree.columnCount()):
                        item.setBackground(col, QColor(230, 230, 230))

                self.tree.addTopLevelItem(item)

                part_sketches = list(getattr(part, "sketches", []) or [])
                part_metas = list(getattr(part, "sketch_meta", []) or [])
                for idx, sketch in enumerate(part_sketches):
                    meta = part_metas[idx] if idx < len(part_metas) else {}
                    visible = self._sketch_row_visible(meta)
                    sketch_item = QTreeWidgetItem(["", self._sketch_row_name(idx, meta, sketch), "Sketch"])
                    sketch_item.setData(
                        0,
                        Qt.UserRole,
                        {"kind": "part_sketch", "part_id": part.id, "sketch_index": idx},
                    )
                    sketch_item.setFlags(
                        Qt.ItemIsEnabled
                        | Qt.ItemIsSelectable
                        | Qt.ItemIsDragEnabled
                        | Qt.ItemIsDropEnabled
                        | Qt.ItemIsEditable
                    )
                    for col in range(self.tree.columnCount()):
                        sketch_item.setBackground(col, QColor(245, 245, 245))
                    self._apply_sketch_visibility_cell(sketch_item, visible)
                    item.addChild(sketch_item)
                if part_sketches:
                    item.setExpanded(True)

            main = self.window()
            if (
                getattr(self.sketch_view, "project_mode", "2d") == "3d"
                and main
                and hasattr(main, "model3d")
            ):
                primitives = list(main.model3d.get("primitives", []))
                if primitives:
                    header = QTreeWidgetItem(["", "3D Primitives", "Primitive Group"])
                    header.setFlags(Qt.ItemIsEnabled)
                    self.tree.addTopLevelItem(header)
                for prim in primitives:
                    prim_id = prim.get("id")
                    name = f"{prim.get('type', 'shape').title()} #{prim_id}"
                    item = QTreeWidgetItem([str(prim_id), name, "3D Primitive"])
                    item.setData(0, Qt.UserRole, {"kind": "primitive", "id": prim_id})
                    for col in range(self.tree.columnCount()):
                        item.setBackground(col, QColor(240, 240, 240))
                    self.tree.addTopLevelItem(item)

            if selected_key is not None:
                self._restore_tree_selection(selected_key)
        finally:
            self._suppress_item_changed = False
        # Legacy sections removed.

    def _count_attrs_for_part(self, part, attrs):
        if not part or not part.geometry or part.geometry.is_empty:
            return 0
        tol = float(SNAP_TOL) * 1.5
        geom = part.geometry.buffer(tol)
        count = 0
        for attr in attrs:
            coords = attr.get("coords")
            if self._coords_belong_to_part(coords, geom):
                count += 1
        return count

    def _coords_belong_to_part(self, coords, geom):
        if coords is None:
            return False
        if self._is_point(coords):
            return geom.contains(Point(coords))
        if self._is_edge(coords):
            mid = ((coords[0][0] + coords[1][0]) / 2.0, (coords[0][1] + coords[1][1]) / 2.0)
            return geom.contains(Point(mid))
        return False

    def _is_point(self, value):
        if isinstance(value, (tuple, list, np.ndarray)) and len(value) == 2:
            try:
                float(value[0])
                float(value[1])
                return True
            except (TypeError, ValueError):
                return False
        return False

    def _is_edge(self, value):
        return (
            isinstance(value, (tuple, list, np.ndarray))
            and len(value) == 2
            and self._is_point(value[0])
            and self._is_point(value[1])
        )

    def delete_part(self):
        """Delete selected part from assembly."""
        selected = self.tree.currentItem()
        if not selected:
            QMessageBox.warning(self, "Warning", "No part selected.")
            return
        data = selected.data(0, Qt.UserRole)
        if isinstance(data, dict) and data.get("kind") in ("active_sketch", "part_sketch"):
            self._delete_selected_sketch_entity()
            return
        if isinstance(data, dict) and data.get("kind") != "part":
            QMessageBox.warning(self, "Warning", "Select a part row (or sketch row) to delete.")
            return
        part_id = data.get("id") if isinstance(data, dict) else data
        part = self._find_part_by_id(part_id)
        if part and self.sketch_view.delete_part(part, confirm=True):
            self.refresh()
        elif not part:
            QMessageBox.warning(self, "Warning", "Selected part no longer exists. Refresh and reselect.")

    def edit_part_shape(self):
        """Load selected part sketches into the Geometry/Part sketch editor for in-place shape update."""
        selected = self.tree.currentItem()
        if not selected:
            QMessageBox.warning(self, "Edit Shape", "No part selected.")
            return
        data = selected.data(0, Qt.UserRole)
        if isinstance(data, dict) and data.get("kind") in ("active_sketch", "part_sketch"):
            self._edit_selected_sketch_entity()
            return
        if isinstance(data, dict) and data.get("kind") != "part":
            QMessageBox.warning(self, "Edit Shape", "Select a part row or sketch row.")
            return
        part_id = data.get("id") if isinstance(data, dict) else data
        part = self._find_part_by_id(part_id)
        if not part:
            QMessageBox.warning(self, "Edit Shape", "Selected part no longer exists. Refresh and reselect.")
            self.refresh()
            return
        if str(getattr(part, "part_type", "")).lower() == "particle_set":
            QMessageBox.information(
                self,
                "Particle Set",
                "Particles are generated from geometry. Edit sketch to modify.",
            )
            return

        main = self.window()
        panel = getattr(main, "properties_panel", None) if main else None
        tabs = getattr(panel, "tabs", None) if panel else None
        if tabs is not None:
            try:
                if tabs.isTabEnabled(0):
                    tabs.setCurrentIndex(0)
            except Exception:
                pass
        try:
            self.sketch_view.set_module("Part")
        except Exception:
            pass
        if main is not None and hasattr(main, "enter_sketch_edit_mode"):
            main.enter_sketch_edit_mode(part.id)
        elif hasattr(self.sketch_view, "begin_part_shape_edit"):
            self.sketch_view.begin_part_shape_edit(part)
        else:
            QMessageBox.information(self, "Edit Shape", "Part shape edit is not available in this build.")

    def copy_selected_entity(self):
        selected = self.tree.currentItem()
        if not selected:
            QMessageBox.warning(self, "Copy", "No entity selected.")
            return
        data = selected.data(0, Qt.UserRole)
        if isinstance(data, dict) and data.get("kind") in ("active_sketch", "part_sketch"):
            self._duplicate_selected_sketch_entity()
            return
        if isinstance(data, dict) and data.get("kind") == "part":
            offset = self._prompt_copy_offset("Copy Part")
            if not offset:
                return
            dx, dy = offset
            self.sketch_view.set_selected_part(data.get("id"))
            self.sketch_view.copy_selected_part(dx, dy, count=1, name_suffix="Copy")
            self.refresh()
            return
        QMessageBox.information(self, "Copy", "Select a part or sketch row to copy.")

    def assign_material_to_part(self):
        """Assign material to selected part, preventing assignment to voids."""
        selected = self.tree.currentItem()
        if not selected:
            QMessageBox.warning(self, "Warning", "No part selected.")
            return

        data = selected.data(0, Qt.UserRole)
        if isinstance(data, dict) and data.get("kind") == "part_sketch":
            data = {"kind": "part", "id": data.get("part_id")}
        if isinstance(data, dict) and data.get("kind") == "active_sketch":
            QMessageBox.information(self, "Material", "Active sketches are not parts. Confirm Part first or select a part row.")
            return
        if isinstance(data, dict) and data.get("kind") == "primitive":
            main = self.window()
            if main and hasattr(main, "_show_primitive_material_menu_for_id"):
                main._show_primitive_material_menu_for_id(
                    data.get("id"),
                    global_pos=self.mapToGlobal(self.rect().center()),
                    label=selected.text(0),
                )
            return

        part_id = data.get("id") if isinstance(data, dict) else data
        part_to_modify = self._find_part_by_id(part_id)

        if not part_to_modify:
            QMessageBox.critical(self, "Error", "Selected part not found in the data model.")
            return

        # Voids are handled by sketch_view.build_material_menu, which redirects
        # the assignment to the void's solid parent part.

        main_window = self.window()
        if not hasattr(main_window, "active_stage"):
            return

        if not self.sketch_view.can_assign_material():
            QMessageBox.warning(
                self,
                "Locked",
                "Material assignment is only allowed in the Materials stage.",
            )
            return

        menu = self.sketch_view.build_material_menu(part_to_modify)
        if not menu:
            return
        sender = self.sender()
        if isinstance(sender, QWidget):
            menu.exec(sender.mapToGlobal(sender.rect().bottomLeft()))
        else:
            menu.exec(self.mapToGlobal(self.rect().center()))

    def assign_material_direct(self, part):
        if not self.sketch_view.can_assign_material():
            QMessageBox.warning(self, "Locked", "Switch to Materials stage first.")
            return

        menu = self.sketch_view.build_material_menu(part)
        if not menu:
            return
        menu.exec(self.mapToGlobal(self.rect().center()))


class StageIconTabs(QWidget):
    """Right-rail stage selector with active label + icon, icon-only for other stages."""

    currentChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stack = QStackedWidget(self)
        self._pages = []
        self._buttons = []
        self._labels = []
        self._tooltips = []
        self._enabled = []
        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)

        self._rail = QWidget(self)
        self._rail_layout = QVBoxLayout(self._rail)
        self._rail_layout.setContentsMargins(0, 0, 0, 0)
        self._rail_layout.setSpacing(6)
        self._rail_layout.addStretch(1)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self._stack, 1)
        layout.addWidget(self._rail, 0, Qt.AlignRight)
        self._tab_fade_anim = None

    def addTab(self, widget, icon, text=""):
        page = _wrap_in_dock_scroll(widget, self._stack)
        idx = self._stack.addWidget(page)
        self._pages.append(widget)
        self._labels.append(text or "")
        self._tooltips.append(text or "")
        self._enabled.append(True)

        btn = QToolButton(self._rail)
        btn.setProperty("stageBtn", True)
        btn.setProperty("dockSectionTab", True)
        btn.setCheckable(True)
        btn.setIcon(icon)
        btn.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
        btn.setAutoRaise(False)
        btn.clicked.connect(lambda checked=False, i=idx: self.setCurrentIndex(i))
        self._button_group.addButton(btn, idx)
        self._buttons.append(btn)
        self._rail_layout.insertWidget(self._rail_layout.count() - 1, btn, 0, Qt.AlignRight)

        if idx == 0:
            self.setCurrentIndex(0)
        else:
            self._refresh_button_presentation(self.currentIndex())
        return idx

    def count(self):
        return self._stack.count()

    def currentIndex(self):
        return self._stack.currentIndex()

    def currentWidget(self):
        idx = self.currentIndex()
        if 0 <= idx < len(self._pages):
            return self._pages[idx]
        return None

    def currentScrollArea(self):
        page = self._stack.currentWidget()
        if page is None:
            return None
        return page.findChild(QScrollArea)

    def widget(self, index):
        index = int(index)
        if 0 <= index < len(self._pages):
            return self._pages[index]
        return None

    def scrollCurrentToTop(self):
        scroll = self.currentScrollArea()
        if scroll is None:
            return

        def _apply():
            bar = scroll.verticalScrollBar()
            if bar is not None:
                bar.setValue(bar.minimum())

        QTimer.singleShot(0, _apply)

    def ensureWidgetVisible(self, widget, top_margin=14):
        scroll = self.currentScrollArea()
        if scroll is None or widget is None:
            return

        def _apply():
            content = scroll.widget()
            if content is None:
                return
            try:
                scroll.ensureWidgetVisible(widget, 0, int(top_margin))
                pos = widget.mapTo(content, QPoint(0, 0))
                bar = scroll.verticalScrollBar()
                if bar is not None:
                    bar.setValue(max(bar.minimum(), int(pos.y()) - int(top_margin)))
            except Exception:
                pass

        QTimer.singleShot(0, _apply)

    def setCurrentIndex(self, index):
        index = int(index)
        if index < 0 or index >= self.count():
            return
        if not self.isTabEnabled(index):
            return
        prev = self._stack.currentIndex()
        prev_widget = self._stack.currentWidget() if prev >= 0 else None
        self._stack.setCurrentIndex(index)
        if 0 <= index < len(self._buttons):
            self._buttons[index].setChecked(True)
        self._refresh_button_presentation(index)
        if prev != index:
            self._animate_current_tab(prev_widget)
            self.currentChanged.emit(index)

    def setCurrentWidget(self, widget):
        idx = self.indexOf(widget)
        if idx >= 0:
            self.setCurrentIndex(idx)

    def indexOf(self, widget):
        try:
            return self._pages.index(widget)
        except ValueError:
            return -1

    def setTabToolTip(self, index, text):
        index = int(index)
        if 0 <= index < len(self._tooltips):
            self._tooltips[index] = text or ""
        if 0 <= index < len(self._buttons):
            self._buttons[index].setToolTip(text or "")

    def tabToolTip(self, index):
        index = int(index)
        if 0 <= index < len(self._tooltips):
            return self._tooltips[index]
        return ""

    def tabText(self, index):
        index = int(index)
        if 0 <= index < len(self._labels):
            return self._labels[index]
        return ""

    def setTabEnabled(self, index, enabled):
        index = int(index)
        if index < 0 or index >= len(self._enabled):
            return
        state = bool(enabled)
        self._enabled[index] = state
        btn = self._buttons[index]
        btn.setEnabled(state)
        if not state and self.currentIndex() == index:
            for i, ok in enumerate(self._enabled):
                if ok:
                    self.setCurrentIndex(i)
                    break
        else:
            self._refresh_button_presentation(self.currentIndex())

    def setRailVisible(self, visible):
        self._rail.setVisible(bool(visible))

    def setRailInteractive(self, interactive):
        state = bool(interactive)
        for btn in self._buttons:
            btn.setEnabled(state)

    def isTabEnabled(self, index):
        index = int(index)
        if 0 <= index < len(self._enabled):
            return bool(self._enabled[index])
        return False

    def _refresh_button_presentation(self, current_index):
        for i, btn in enumerate(self._buttons):
            btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
            btn.setText("")
            btn.setMinimumSize(28, 28)
            btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def _animate_current_tab(self, previous_widget):
        widget = self._stack.currentWidget()
        if widget is None or widget is previous_widget:
            return
        effect = widget.graphicsEffect()
        if not isinstance(effect, QGraphicsOpacityEffect):
            effect = QGraphicsOpacityEffect(widget)
            widget.setGraphicsEffect(effect)
        effect.setOpacity(0.0)
        anim = QPropertyAnimation(effect, b"opacity", self)
        anim.setDuration(160)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        self._tab_fade_anim = anim
        anim.start()


class MaterialsPanel(QWidget):
    ASSIGNMENT_MODE_OPTIONS = [
        ("Homogeneous", "homogeneous"),
        ("Heterogeneous", "heterogeneous"),
        ("Material Field", "material_field"),
    ]
    HETEROGENEITY_METHOD_OPTIONS = [
        ("Region Based", "region_based"),
        ("Random Distribution", "random_distribution"),
        ("Field / Gradient Distribution", "field_gradient_distribution"),
    ]
    MATERIAL_FIELD_PROPERTY_OPTIONS = [
        ("Young's Modulus", "E"),
        ("Density", "rho"),
        ("Poisson Ratio", "nu"),
    ]
    MATERIAL_FIELD_TYPE_OPTIONS = [
        ("Linear Gradient", "linear_gradient"),
        ("Radial Gradient", "radial_gradient"),
        ("Random Field", "random_field"),
        ("User Equation", "user_equation"),
    ]
    SYMMETRY_OPTIONS = material_symmetry_options()
    BEHAVIOR_OPTIONS = material_behavior_options()
    DAMAGE_OPTIONS = material_damage_options()

    def __init__(self, sketch_view, parent=None, project_state=None):
        super().__init__(parent)
        self.sketch_view = sketch_view
        self.project_state = _resolve_project_state(sketch_view, project_state)
        self.material_controller = None
        self._create_label_full = "Create / Update"
        self._new_label_full = "New"
        self._delete_label_full = "Delete"
        self._assign_label_full = "Assign"
        self._assignment_heterogeneity_config = normalize_heterogeneity_config({})
        self._assignment_material_field_config = normalize_material_field_config({})
        self._registry_editor_enabled = False
        self._suppress_auto_assign = False
        self._material_pick_source = "project"
        self._csv_records = materials_db.get_records()
        self._custom_records: list[materials_db.MaterialRecord] = []
        self._load_custom_records()

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # Library view = main Properties content (shown in stage tab).
        # Project view = Metadata content (shown via top toggle on this stage).
        # Both views are built in this __init__; the library_view is mounted to
        # outer_layout, while project_view is exposed via get_project_view()
        # for the PropertiesPanel to register as Materials-stage metadata.
        self.library_view = QWidget(self)
        library_layout = QVBoxLayout(self.library_view)
        _apply_layout_metrics(library_layout, margins=(6, 6, 6, 6), spacing=6)
        outer_layout.addWidget(self.library_view, 1)

        self.project_view = QWidget()
        project_layout = QVBoxLayout(self.project_view)
        _apply_layout_metrics(project_layout, margins=(6, 6, 6, 6), spacing=6)

        # Legacy "layout" name kept pointing at library_layout so existing code
        # paths (which call `layout.addWidget(...)`) keep working without
        # widespread renames.
        layout = library_layout
        self._materials_outer_layout = outer_layout
        self._materials_scroll_area = None
        self._materials_library_layout = library_layout
        self._materials_project_layout = project_layout

        hint = QLabel("Pick a part, choose a material, click Assign.")
        hint.setToolTip(
            "Assign flow: left-click a part, right-click to choose a material. Use 'Other...' to edit or create materials."
        )
        hint.setWordWrap(True)
        hint.setMinimumWidth(0)
        hint.setMaximumWidth(16777215)
        # sizeHint = 100 px so the label doesn't force the panel wider; word
        # wrap handles the rest at render time.
        hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout.addWidget(hint)
        self._cross_tab_notice = _make_cross_tab_notice_label(self)
        layout.addWidget(self._cross_tab_notice)
        # Part selector lives WITH the Library (assignment workflow stays in
        # the Properties view next to the material list).
        self.part_select_combo = QComboBox()
        self.part_select_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.part_select_combo.currentIndexChanged.connect(self._on_part_select_combo_changed)
        part_select_row = QHBoxLayout()
        part_select_row.setContentsMargins(0, 0, 0, 0)
        part_select_row.setSpacing(6)
        part_select_label = QLabel("Part")
        part_select_label.setObjectName("MinorStatusLabel")
        part_select_row.addWidget(part_select_label, 0)
        part_select_row.addWidget(self.part_select_combo, 1)
        layout.addLayout(part_select_row)

        csv_group = QGroupBox("Material Library")
        csv_layout = QVBoxLayout(csv_group)
        _apply_layout_metrics(csv_layout, margins=(12, 12, 12, 12), spacing=8)
        filters_toolbar = QHBoxLayout()
        _apply_layout_metrics(filters_toolbar, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        # Save a reference so we can append the Assign button to this row
        # later (after btn_assign is constructed further down in __init__).
        self._csv_filters_toolbar = filters_toolbar
        self.csv_search_toggle = QToolButton()
        self.csv_search_toggle.setIcon(get_icon("search", size=16))
        self.csv_search_toggle.setIconSize(QSize(16, 16))
        self.csv_search_toggle.setCheckable(True)
        self.csv_search_toggle.setToolTip("Search materials by name")
        self.csv_filter_toggle = QToolButton()
        self.csv_filter_toggle.setIcon(get_icon("filter", size=16))
        self.csv_filter_toggle.setIconSize(QSize(16, 16))
        self.csv_filter_toggle.setCheckable(True)
        self.csv_filter_toggle.setToolTip("Filter materials by category and behavior")
        filters_toolbar.addWidget(self.csv_search_toggle)
        filters_toolbar.addWidget(self.csv_filter_toggle)
        filters_toolbar.addStretch(1)
        csv_layout.addLayout(filters_toolbar)

        self._csv_search_container = QWidget()
        csv_search_layout = QHBoxLayout(self._csv_search_container)
        _apply_layout_metrics(csv_search_layout, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        self.csv_search_input = QLineEdit()
        self.csv_search_input.setPlaceholderText("Search material name")
        self.csv_search_input.textChanged.connect(self.refresh_csv_list)
        csv_search_layout.addWidget(self.csv_search_input)
        self._csv_search_container.setVisible(False)
        csv_layout.addWidget(self._csv_search_container)

        self._csv_filter_container = QWidget()
        csv_filter_form = QFormLayout(self._csv_filter_container)
        csv_filter_form.setLabelAlignment(Qt.AlignLeft)
        csv_filter_form.setFormAlignment(Qt.AlignTop)
        csv_filter_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        _apply_layout_metrics(csv_filter_form, margins=(0, 0, 0, 0), spacing=6)
        self.csv_category_combo = QComboBox()
        self.csv_category_combo.currentIndexChanged.connect(self.refresh_csv_list)
        csv_filter_form.addRow("Category", self.csv_category_combo)
        self.csv_behavior_combo = QComboBox()
        self.csv_behavior_combo.currentIndexChanged.connect(self.refresh_csv_list)
        csv_filter_form.addRow("Behavior (Ductile/Brittle)", self.csv_behavior_combo)
        self._csv_filter_container.setVisible(False)
        csv_layout.addWidget(self._csv_filter_container)

        def _on_search_toggled(checked):
            self._csv_search_container.setVisible(bool(checked))
            if checked:
                self.csv_search_input.setFocus()

        self.csv_search_toggle.toggled.connect(_on_search_toggled)
        self.csv_filter_toggle.toggled.connect(self._csv_filter_container.setVisible)

        self.csv_list = QTreeWidget()
        self.csv_list.setSelectionMode(QTreeWidget.SingleSelection)
        self.csv_list.setHeaderLabels(["Name", "Category", "Behavior", "Condition"])
        self.csv_list.setColumnCount(4)
        self.csv_list.setUniformRowHeights(True)
        self.csv_list.setMinimumHeight(80)
        self.csv_list.setMinimumWidth(0)
        self.csv_list.setContextMenuPolicy(Qt.CustomContextMenu)
        csv_header = self.csv_list.header()
        csv_header.setStretchLastSection(False)
        csv_header.setCascadingSectionResizes(True)
        for idx in (0, 1, 2, 3):
            csv_header.setSectionResizeMode(idx, QHeaderView.Interactive)
        csv_header.resizeSection(0, 160)
        csv_header.resizeSection(1, 120)
        csv_header.resizeSection(2, 120)
        csv_header.resizeSection(3, 120)
        self.csv_list.itemSelectionChanged.connect(self._on_csv_selection_changed)
        self.csv_list.customContextMenuRequested.connect(self._show_csv_context_menu)
        csv_layout.addWidget(self.csv_list, 1)

        csv_details_group = QGroupBox("Selected Material Details")
        csv_details_group.setCheckable(True)
        csv_details_outer = QVBoxLayout(csv_details_group)
        _apply_layout_metrics(csv_details_outer, margins=(8, 8, 8, 8), spacing=4)
        self._csv_details_body = QWidget(csv_details_group)
        csv_details_layout = QFormLayout(self._csv_details_body)
        _apply_layout_metrics(csv_details_layout, margins=(0, 0, 0, 0), spacing=6)
        self.csv_detail_labels = {}
        detail_specs = [
            ("ultimate_tensile_strength", "Ultimate Tensile Strength (Pa)"),
            ("yield_strength", "Yield Strength (Pa)"),
            ("elongation_pct", "Elongation (%)"),
            ("brinell_hb", "Brinell Hardness (HB)"),
            ("vickers_hv", "Vickers Hardness (HV)"),
            ("youngs_modulus", "Young's Modulus (Pa)"),
            ("shear_modulus", "Shear Modulus (Pa)"),
            ("poisson_ratio", "Poisson Ratio (-)"),
            ("density_kg_m3", "Density (kg/m^3)"),
            ("critical_strain_energy_density", "Critical Strain Energy Density (J/m^3)"),
        ]
        for key, label in detail_specs:
            value_label = QLabel("—")
            value_label.setObjectName("MinorStatusLabel")
            self.csv_detail_labels[key] = value_label
            csv_details_layout.addRow(label, value_label)
        csv_details_outer.addWidget(self._csv_details_body)
        _persist_collapsible(csv_details_group, "materials/selected_material_details", self._csv_details_body, default_expanded=False)
        csv_layout.addWidget(csv_details_group)

        # Damage toggle + Create New Material on a single row to save vertical
        # space. The full meaning ("use critical strain energy") goes into the
        # tooltip since the long label would force a second row.
        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(8)
        damage_label = QLabel("Damage")
        damage_label.setObjectName("MinorStatusLabel")
        damage_label.setToolTip("Damage: use critical strain energy density.")
        action_row.addWidget(damage_label, 0)
        self.csv_damage_combo = QComboBox()
        self.csv_damage_combo.addItem("Off", False)
        self.csv_damage_combo.addItem("On", True)
        self.csv_damage_combo.setToolTip("Toggle damage modeling using critical strain energy.")
        self.csv_damage_combo.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.csv_damage_combo.setMinimumWidth(80)
        action_row.addWidget(self.csv_damage_combo, 0)
        self.btn_create_custom = QPushButton("Create New Material")
        self.btn_create_custom.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_create_custom.clicked.connect(self._open_create_material_dialog)
        action_row.addWidget(self.btn_create_custom, 1)
        csv_layout.addLayout(action_row)

        layout.addWidget(csv_group, 3)
        layout.addWidget(_make_dock_separator())

        library_group = QGroupBox("Project Materials")
        library_group.setCheckable(True)
        library_group.setChecked(False)
        library_layout = QVBoxLayout(library_group)
        _apply_layout_metrics(library_layout, margins=(12, 12, 12, 12), spacing=8)
        self._library_body = QWidget(library_group)
        library_body_layout = QVBoxLayout(self._library_body)
        _apply_layout_metrics(library_body_layout, margins=(0, 0, 0, 0), spacing=8)
        self.mat_list = QTreeWidget()
        self.mat_list.setDragEnabled(True)
        self.mat_list.setSelectionMode(QTreeWidget.SingleSelection)
        self.mat_list.setHeaderLabels(["Name", "Behavior", "Serial"])
        self.mat_list.setColumnCount(3)
        self.mat_list.setUniformRowHeights(True)
        self.mat_list.setMinimumHeight(80)
        self.mat_list.setContextMenuPolicy(Qt.CustomContextMenu)
        header = self.mat_list.header()
        header.setStretchLastSection(False)
        header.setCascadingSectionResizes(True)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Interactive)
        header.setSectionResizeMode(2, QHeaderView.Interactive)
        header.resizeSection(1, 96)
        header.resizeSection(2, 72)
        self.mat_list.itemClicked.connect(self.on_material_select_from_list)
        self.mat_list.customContextMenuRequested.connect(self._show_material_context_menu)
        library_body_layout.addWidget(self.mat_list, 1)
        self._mat_list_empty_label = _make_empty_state_label(
            "No project materials yet — pick from the Material Library above and click Assign.",
            self._library_body,
        )
        library_body_layout.addWidget(self._mat_list_empty_label)
        _bind_tree_empty_state(self.mat_list, self._mat_list_empty_label)
        library_layout.addWidget(self._library_body)
        _persist_collapsible(library_group, "materials/project_materials", self._library_body, default_expanded=False)

        editor_group = QGroupBox("Material Editor (Legacy)")
        editor_layout = QVBoxLayout(editor_group)
        _apply_layout_metrics(editor_layout, margins=(12, 12, 12, 12), spacing=8)

        basic_form = QFormLayout()
        basic_form.setLabelAlignment(Qt.AlignLeft)
        basic_form.setFormAlignment(Qt.AlignTop)
        basic_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        basic_form.setHorizontalSpacing(10)
        basic_form.setVerticalSpacing(8)
        self.name_input = QLineEdit()
        self.name_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.name_input.setMinimumHeight(26)
        basic_form.addRow("Name", self.name_input)

        self.type_combo = QComboBox()
        for text, value in self.BEHAVIOR_OPTIONS:
            self.type_combo.addItem(text, value)
        self.type_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.type_combo.setMinimumHeight(26)
        self.type_combo.currentIndexChanged.connect(self.update_property_fields)
        basic_form.addRow("Behavior", self.type_combo)
        self.editor_symmetry_combo = QComboBox()
        for text, value in self.SYMMETRY_OPTIONS:
            self.editor_symmetry_combo.addItem(text, value)
        self.editor_symmetry_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.editor_symmetry_combo.setMinimumHeight(26)
        self.editor_symmetry_combo.currentIndexChanged.connect(self.update_property_fields)
        basic_form.addRow("Symmetry", self.editor_symmetry_combo)
        self.editor_damage_combo = QComboBox()
        for text, value in self.DAMAGE_OPTIONS:
            self.editor_damage_combo.addItem(text, value)
        self.editor_damage_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.editor_damage_combo.setMinimumHeight(26)
        self.editor_damage_combo.currentIndexChanged.connect(self.update_property_fields)
        basic_form.addRow("Damage", self.editor_damage_combo)
        editor_layout.addLayout(basic_form)

        self.prop_widgets = {}
        self.properties_form = QFormLayout()
        self.properties_form.setLabelAlignment(Qt.AlignLeft)
        self.properties_form.setFormAlignment(Qt.AlignTop)
        self.properties_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.properties_form.setHorizontalSpacing(10)
        self.properties_form.setVerticalSpacing(8)
        editor_layout.addLayout(self.properties_form)

        self.btn_create = QPushButton("Create / Update")
        self.btn_create.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_create.clicked.connect(self.create_material)
        self.btn_create.setIcon(_style_icon(self, "SP_DialogApplyButton", "confirm", ("dialog-ok-apply",)))
        self.btn_new = QPushButton("New")
        self.btn_new.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_new.clicked.connect(self.prepare_new_material)
        self.btn_new.setIcon(_style_icon(self, "SP_FileDialogNewFolder", "add", ("list-add", "document-new")))
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_delete.clicked.connect(self.delete_selected_material)
        self.btn_delete.setIcon(_style_icon(self, "SP_TrashIcon", "delete", ("edit-delete",)))
        self._material_button_grid = QGridLayout()
        _apply_layout_metrics(self._material_button_grid, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        self._material_action_buttons = [self.btn_create, self.btn_new, self.btn_delete]
        editor_layout.addLayout(self._material_button_grid)
        if not self._registry_editor_enabled:
            editor_group.setVisible(False)

        if self._registry_editor_enabled:
            self.update_property_fields()
        assignment_group = QGroupBox("Assign")
        self.assignment_group = assignment_group
        assignment_layout = QFormLayout(assignment_group)
        self.assignment_form = assignment_layout
        assignment_layout.setLabelAlignment(Qt.AlignLeft)
        assignment_layout.setFormAlignment(Qt.AlignTop)
        assignment_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        _apply_layout_metrics(assignment_layout, margins=(12, 12, 12, 12), spacing=DOCK_ROW_SPACING)
        self.selected_part_display = QLabel("No part selected")
        self.selected_part_display.setObjectName("MinorStatusLabel")
        assignment_layout.addRow("Selected Part", self.selected_part_display)
        self.assignment_mode_combo = QComboBox()
        for text, value in self.ASSIGNMENT_MODE_OPTIONS:
            self.assignment_mode_combo.addItem(text, value)
        self.assignment_mode_combo.currentIndexChanged.connect(self._update_assignment_method_ui)
        assignment_layout.addRow("Assignment", self.assignment_mode_combo)
        self.active_material_combo = QComboBox()
        self.active_material_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.active_material_combo.currentIndexChanged.connect(self.on_active_material_selected)
        # Material picker on its own row so its full text fits in the narrow
        # right-dock panel (combining it with Assignment forced both combos to
        # truncate "Homogeneous" → "Hom..." and material names to "[No M...").
        assignment_layout.addRow("Material", self.active_material_combo)
        self.heterogeneity_method_combo = QComboBox()
        for text, value in self.HETEROGENEITY_METHOD_OPTIONS:
            self.heterogeneity_method_combo.addItem(text, value)
        self.heterogeneity_method_combo.currentIndexChanged.connect(self._update_assignment_method_ui)
        assignment_layout.addRow("Method", self.heterogeneity_method_combo)
        self.material_field_property_combo = QComboBox()
        for text, value in self.MATERIAL_FIELD_PROPERTY_OPTIONS:
            self.material_field_property_combo.addItem(text, value)
        self.material_field_property_combo.currentIndexChanged.connect(self._update_assignment_method_ui)
        assignment_layout.addRow("Field Property", self.material_field_property_combo)
        self.material_field_type_combo = QComboBox()
        for text, value in self.MATERIAL_FIELD_TYPE_OPTIONS:
            self.material_field_type_combo.addItem(text, value)
        self.material_field_type_combo.currentIndexChanged.connect(self._update_assignment_method_ui)
        assignment_layout.addRow("Field Type", self.material_field_type_combo)
        self.symmetry_combo = QComboBox()
        for text, value in self.SYMMETRY_OPTIONS:
            self.symmetry_combo.addItem(text, value)
        assignment_layout.addRow("Symmetry", self.symmetry_combo)
        self.behavior_combo = QComboBox()
        for text, value in self.BEHAVIOR_OPTIONS:
            self.behavior_combo.addItem(text, value)
        assignment_layout.addRow("Behavior", self.behavior_combo)
        self.damage_combo = QComboBox()
        for text, value in self.DAMAGE_OPTIONS:
            self.damage_combo.addItem(text, value)
        assignment_layout.addRow("Damage", self.damage_combo)
        # active_material_combo was created earlier and combined with the
        # assignment_mode_combo into a single row. Just keep the row-visibility
        # housekeeping for the previously-hidden combos here.
        self._set_form_row_visible(self.symmetry_combo, False)
        self._set_form_row_visible(self.behavior_combo, False)
        self._set_form_row_visible(self.damage_combo, False)
        self.heterogeneity_summary = QLabel("Per-region material assignment")
        self.heterogeneity_summary.setWordWrap(True)
        self.heterogeneity_summary.setObjectName("MinorStatusLabel")
        assignment_layout.addRow("Distribution", self.heterogeneity_summary)
        self.btn_configure_distribution = QPushButton("Configure Distribution")
        self.btn_configure_distribution.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_configure_distribution.clicked.connect(self._configure_assignment_details)
        assignment_layout.addRow("", self.btn_configure_distribution)
        self.btn_assign = QPushButton("Assign")
        self.btn_assign.setProperty("primary", True)
        self.btn_assign.setMinimumHeight(34)
        self.btn_assign.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_assign.clicked.connect(self.assign_to_selected_part)
        self.btn_assign.setIcon(_style_icon(self, "SP_DialogApplyButton", "confirm", ("dialog-ok-apply",)))

        # Assignment + legacy editor stay alongside the Library (Properties
        # view) so the user can pick a material AND assign in one place.
        # Only the Project Materials list (library_group) moves to the
        # Metadata view as a read-only summary of what's been added.
        layout.addWidget(assignment_group, 0)
        layout.addWidget(editor_group, 0)
        project_layout.addWidget(library_group, 1)

        # Place the Assign button INLINE inside the Material Library toolbar
        # — IMMEDIATELY next to the filter icon (no gap). Shrink it so it
        # visually pairs with the toggle icons.
        self.btn_assign.setMinimumHeight(28)
        self.btn_assign.setMaximumHeight(30)
        self.btn_assign.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn_assign.setMinimumWidth(96)
        self.btn_assign.setMaximumWidth(120)
        # Insert at index 2 — right after [search, filter] and BEFORE the
        # existing stretch — so Assign sits flush against the filter icon and
        # the stretch pushes any extra space to the right of it.
        self._csv_filters_toolbar.insertWidget(2, self.btn_assign)
        # The sticky-footer attribute is kept (set to None) for any legacy
        # code paths that referenced it.
        self._materials_sticky_footer = None

        if hasattr(self.sketch_view, "partSelectionChanged"):
            try:
                self.sketch_view.partSelectionChanged.connect(self._update_selected_part_display)
            except Exception:
                pass
        self.refresh_csv_filters()
        self.refresh_csv_list()
        self.refresh_material_list()
        self.refresh_part_list()
        self._update_selected_part_display(getattr(self.sketch_view, "selected_part_id", None))
        self._update_assignment_method_ui()
        self._update_responsive_layout()
        _finalize_dock_panel(self)
        # project_view is not a Qt child of MaterialsPanel (it's a top-level
        # widget that gets reparented later by register_stage_metadata),
        # so the previous call missed widgets inside it. Finalize it directly.
        _finalize_dock_panel(self.project_view)
        try:
            self.sketch_view.partsChanged.connect(self.refresh_part_list)
        except Exception:
            pass
        try:
            self.sketch_view.materialsChanged.connect(self.refresh_part_list)
        except Exception:
            pass

    @property
    def materials(self):
        return self._materials_store()

    def set_project_state(self, project_state):
        self.project_state = _resolve_project_state(self.sketch_view, project_state)
        self.sketch_view.project_state = self.project_state
        current_serial = getattr(self.sketch_view, "current_material_id", None)
        self.refresh_csv_filters()
        self.refresh_csv_list()
        self.refresh_material_list()
        self.refresh_part_list()
        if current_serial in self._materials_store():
            self.select_material(current_serial)
        elif self._materials_store():
            first_serial = next(iter(sorted(self._materials_store().keys())))
            self.select_material(first_serial)
        else:
            if self._registry_editor_enabled:
                self.prepare_new_material()
        self._update_selected_part_display(getattr(self.sketch_view, "selected_part_id", None))

    def set_material_controller(self, controller):
        self.material_controller = controller

    def _all_csv_records(self) -> list[materials_db.MaterialRecord]:
        return list(self._csv_records) + list(self._custom_records)

    def _custom_cache_path(self):
        return Path(get_workspace_path("materials_custom_cache.json"))

    def _load_custom_records(self):
        path = self._custom_cache_path()
        if not path.exists():
            return
        try:
            import json
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict):
                        self._custom_records.append(materials_db.MaterialRecord(item))
        except Exception:
            pass

    def _save_custom_records(self):
        path = self._custom_cache_path()
        try:
            import json
            payload = [rec.data for rec in self._custom_records]
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    def _reset_csv_combo(self, combo, items):
        current = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("All", "")
        for item in items:
            combo.addItem(item, item)
        if current:
            idx = combo.findData(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def get_project_view(self):
        """Return the widget that holds Project Materials + Assignment for
        registration as Materials-stage Metadata in PropertiesPanel."""
        return self.project_view

    def refresh_csv_filters(self):
        records = self._all_csv_records()
        categories = sorted({rec.category for rec in records if rec.category})
        behaviors = sorted({rec.behavior_tag for rec in records if rec.behavior_tag})
        self._reset_csv_combo(self.csv_category_combo, categories)
        self._reset_csv_combo(self.csv_behavior_combo, behaviors)

    def refresh_csv_list(self):
        records = materials_db.filter_records(
            self._all_csv_records(),
            name=self.csv_search_input.text().strip(),
            category=str(self.csv_category_combo.currentData() or "").strip() or None,
            behavior_tag=str(self.csv_behavior_combo.currentData() or "").strip() or None,
        )
        self.csv_list.clear()
        for record in records:
            name = record.name or "Unnamed"
            item = QTreeWidgetItem(
                [
                    name,
                    record.category,
                    record.behavior_tag,
                    record.condition,
                ]
            )
            item.setData(0, Qt.UserRole, record)
            self.csv_list.addTopLevelItem(item)
        if records:
            self.csv_list.setCurrentItem(self.csv_list.topLevelItem(0))
        else:
            self._update_csv_details(None)

    def _current_csv_record(self):
        item = self.csv_list.currentItem()
        if item is None:
            return None
        return item.data(0, Qt.UserRole)

    def _on_csv_selection_changed(self):
        record = self._current_csv_record()
        self._update_csv_details(record)
        if record is not None:
            self._material_pick_source = "csv"

    def _format_detail_value(self, value):
        if value in (None, ""):
            return "—"
        if isinstance(value, (int, float)):
            return f"{value:.6g}"
        return str(value)

    def _update_csv_details(self, record):
        data = record.data if record is not None else {}
        for key, label in self.csv_detail_labels.items():
            label.setText(self._format_detail_value(data.get(key)))

    def _parse_optional_float(self, text, field_label):
        value = str(text or "").strip()
        if not value:
            return None
        try:
            return float(value)
        except Exception:
            raise ValueError(f"Invalid number for {field_label}")

    def _add_material_from_record(self, record, damage_on=False):
        existing = self._find_material_by_record(record)
        if existing is not None:
            return existing
        mat = materials_db.record_to_material(record)
        properties = copy.deepcopy(getattr(mat, "properties", {}) or {})
        if damage_on:
            critical = properties.get("critical_strain_energy_density")
            if critical is not None:
                properties["failure_energy"] = critical
            else:
                QMessageBox.warning(
                    self,
                    "Damage Setting",
                    "Damage is ON, but Critical Strain Energy Density is missing.\n"
                    "Failure energy will not be set for this material.",
                )
        else:
            properties.pop("failure_energy", None)

        main = self.window()
        execute_command = getattr(main, "execute_app_command", None) if main is not None else None
        if self.material_controller is None or not callable(execute_command):
            QMessageBox.warning(self, "Material Controller", "Material command bus is not available.")
            return None
        result = execute_command(
            AddMaterialCommand(
                name=str(getattr(mat, "name", "Material")),
                mat_type=str(getattr(mat, "mat_type", "ELAS1")),
                behavior=str(getattr(mat, "behavior", "elastic")),
                damage=str(getattr(mat, "damage", "none")),
                symmetry=str(getattr(mat, "symmetry", "isotropic")),
                properties=properties,
                selected_part_id=None,
                auto_assign_selected_part=False,
                announce_assignment=False,
            )
        )
        created_mat = result.get("material")
        if created_mat is not None:
            setattr(created_mat, "metadata", dict(record.data))
            self.refresh_material_list()
            self.set_active_material(getattr(created_mat, "serial", None))
        return created_mat

    def _find_material_by_record(self, record):
        if record is None:
            return None
        target = {
            "material_name": str(record.data.get("material_name", "")).strip(),
            "Material_category": str(record.data.get("Material_category", "")).strip(),
            "behavior_tag": str(record.data.get("behavior_tag", "")).strip(),
            "condition": str(record.data.get("condition", "")).strip(),
        }
        for mat in self._materials_store().values():
            meta = getattr(mat, "metadata", {}) or {}
            if not isinstance(meta, dict):
                continue
            if all(str(meta.get(k, "")).strip() == v for k, v in target.items()):
                return mat
        return None

    def insert_selected_csv_material(self):
        record = self._current_csv_record()
        if record is None:
            QMessageBox.warning(self, "Insert Material", "Select a CSV material first.")
            return
        damage_on = bool(self.csv_damage_combo.currentData())
        created = self._add_material_from_record(record, damage_on=damage_on)
        if created is not None:
            assigned = False
            part = self._selected_part()
            if part is not None:
                assigned = self._assign_material_id_to_part(part, created.serial)
            if assigned:
                QMessageBox.information(
                    self,
                    "Insert Material",
                    f"Inserted '{created.name}' and assigned it to {part.name}.",
                )
            else:
                QMessageBox.information(self, "Insert Material", f"Inserted '{created.name}'.")

    def _open_create_material_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Create New Material")
        layout = QVBoxLayout(dialog)
        _apply_layout_metrics(layout, margins=(12, 12, 12, 12), spacing=8)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignTop)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(6)

        name_input = QLineEdit()
        form.addRow("Material Name", name_input)

        behavior_combo = QComboBox()
        behavior_combo.addItem("Ductile", "Ductile")
        behavior_combo.addItem("Brittle", "Brittle")
        form.addRow("Behavior Tag", behavior_combo)

        category_input = QLineEdit()
        form.addRow("Category", category_input)
        standard_input = QLineEdit()
        form.addRow("Standard", standard_input)
        condition_input = QLineEdit()
        form.addRow("Condition", condition_input)

        uts_input = QLineEdit()
        form.addRow("Ultimate Tensile Strength (Pa)", uts_input)
        ys_input = QLineEdit()
        form.addRow("Yield Strength (Pa)", ys_input)
        elong_input = QLineEdit()
        form.addRow("Elongation (%)", elong_input)
        brinell_input = QLineEdit()
        form.addRow("Brinell Hardness (HB)", brinell_input)
        vickers_input = QLineEdit()
        form.addRow("Vickers Hardness (HV)", vickers_input)
        youngs_input = QLineEdit()
        form.addRow("Young's Modulus (Pa)", youngs_input)
        shear_input = QLineEdit()
        form.addRow("Shear Modulus (Pa)", shear_input)
        poisson_input = QLineEdit()
        form.addRow("Poisson Ratio (-)", poisson_input)
        density_input = QLineEdit()
        form.addRow("Density (kg/m^3)", density_input)
        crit_input = QLineEdit("1.0e8")
        form.addRow("Critical Strain Energy Density (J/m^3)", crit_input)

        desc_input = QTextEdit()
        desc_input.setFixedHeight(60)
        form.addRow("Description", desc_input)

        damage_combo = QComboBox()
        damage_combo.addItem("Off", False)
        damage_combo.addItem("On", True)
        form.addRow("Damage (use critical strain energy)", damage_combo)

        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.Accepted:
            return

        name = name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Create Material", "Material name cannot be empty.")
            return

        try:
            data = {
                "behavior_tag": str(behavior_combo.currentData() or "Ductile"),
                "standard": standard_input.text().strip(),
                "Material_category": category_input.text().strip(),
                "material_name": name,
                "condition": condition_input.text().strip(),
                "ultimate_tensile_strength": self._parse_optional_float(uts_input.text(), "Ultimate Tensile Strength"),
                "yield_strength": self._parse_optional_float(ys_input.text(), "Yield Strength"),
                "elongation_pct": self._parse_optional_float(elong_input.text(), "Elongation"),
                "brinell_hb": self._parse_optional_float(brinell_input.text(), "Brinell Hardness"),
                "vickers_hv": self._parse_optional_float(vickers_input.text(), "Vickers Hardness"),
                "youngs_modulus": self._parse_optional_float(youngs_input.text(), "Young's Modulus"),
                "shear_modulus": self._parse_optional_float(shear_input.text(), "Shear Modulus"),
                "poisson_ratio": self._parse_optional_float(poisson_input.text(), "Poisson Ratio"),
                "density_kg_m3": self._parse_optional_float(density_input.text(), "Density"),
                "critical_strain_energy_density": self._parse_optional_float(
                    crit_input.text(),
                    "Critical Strain Energy Density",
                ),
                "description": desc_input.toPlainText().strip(),
            }
        except ValueError as exc:
            QMessageBox.warning(self, "Create Material", str(exc))
            return

        damage_on = bool(damage_combo.currentData())
        if damage_on and data.get("critical_strain_energy_density") is None:
            QMessageBox.warning(
                self,
                "Create Material",
                "Damage is ON, but Critical Strain Energy Density is missing.\n"
                "Please provide a value or switch Damage to Off.",
            )
            return
        record = materials_db.MaterialRecord(data)
        self._custom_records.append(record)
        self._save_custom_records()
        self.refresh_csv_filters()
        self.refresh_csv_list()
        created = self._add_material_from_record(record, damage_on=damage_on)
        if created is not None:
            QMessageBox.information(self, "Create Material", f"Created '{created.name}'.")

    def on_active_material_selected(self):
        mat_id = self.active_material_combo.currentData()
        try:
            mat_id = int(mat_id)
        except (TypeError, ValueError):
            pass
        if mat_id not in (None, -1):
            self._material_pick_source = "project"
        self.sketch_view.set_current_material(mat_id)
        if mat_id is None or mat_id == -1:
            return
        mat = self._materials_store().get(mat_id)
        if mat and self._registry_editor_enabled:
            self._load_material_into_editor(mat)
        self._update_selected_part_display(getattr(self.sketch_view, "selected_part_id", None))

    def set_active_material(self, serial):
        index = self.active_material_combo.findData(serial)
        if index != -1:
            self.active_material_combo.setCurrentIndex(index)

    def focus_name_input(self):
        if self._registry_editor_enabled:
            self.name_input.setFocus()

    def select_material(self, serial, update_editor=True):
        if serial is None:
            return
        self.set_active_material(serial)
        target_item = None
        for idx in range(self.mat_list.topLevelItemCount()):
            item = self.mat_list.topLevelItem(idx)
            try:
                if int(item.text(2)) == int(serial):
                    target_item = item
                    break
            except ValueError:
                continue
        if target_item:
            self.mat_list.setCurrentItem(target_item)
        if update_editor:
            mat = self._materials_store().get(int(serial))
            if mat is not None and self._registry_editor_enabled:
                self._load_material_into_editor(mat)
        self._sync_material_selection_to_inspector()

    def _material_payload(self, serial):
        if serial is None:
            return None
        return {
            "kind": "material",
            "serial": serial,
            "stage": ProjectStage.MATERIALS,
        }

    def _sync_material_selection_to_inspector(self):
        item = self.mat_list.currentItem()
        main = self.window()
        inspector = getattr(main, "property_inspector", None) if main is not None else None
        if inspector is None:
            return
        if item is None:
            inspector.set_selection_payload(None)
            return
        try:
            serial = int(item.text(2))
        except Exception:
            inspector.set_selection_payload(None)
            return
        inspector.set_selection_payload(self._material_payload(serial))

    def _show_material_context_menu(self, pos):
        item = self.mat_list.itemAt(pos)
        if item is None:
            return
        self.mat_list.setCurrentItem(item)
        menu = QMenu(self.mat_list)
        menu.addAction("Duplicate", self._duplicate_selected_material)
        menu.addAction("Delete", self.delete_selected_material)
        menu.exec(self.mat_list.viewport().mapToGlobal(pos))

    def _show_csv_context_menu(self, pos):
        item = self.csv_list.itemAt(pos)
        if item is None:
            return
        self.csv_list.setCurrentItem(item)
        menu = QMenu(self.csv_list)
        menu.addAction("Create New Material", self._open_create_material_dialog)
        menu.exec(self.csv_list.viewport().mapToGlobal(pos))

    def _duplicate_selected_material(self):
        item = self.mat_list.currentItem()
        if item is None:
            QMessageBox.warning(self, "Duplicate Material", "Select a material first.")
            return
        try:
            serial = int(item.text(2))
        except Exception:
            QMessageBox.warning(self, "Duplicate Material", "Selected material is invalid.")
            return
        material = self._materials_store().get(serial)
        if material is None:
            return
        self.sketch_view.push_undo_state()
        clone = Material(
            f"{getattr(material, 'name', 'Material')} Copy",
            getattr(material, "mat_type", ""),
            copy.deepcopy(getattr(material, "properties", {}) or {}),
            symmetry=getattr(material, "symmetry", "isotropic"),
            behavior=getattr(material, "behavior", "elastic"),
            damage=getattr(material, "damage", "none"),
        )
        self._materials_store()[clone.serial] = clone
        self.refresh_material_list()
        self.select_material(clone.serial)
        self._sync_material_selection_to_inspector()

    def prepare_new_material(self):
        if not self._registry_editor_enabled:
            return
        self.name_input.blockSignals(True)
        self.name_input.clear()
        self.name_input.blockSignals(False)
        self.editor_symmetry_combo.setCurrentIndex(0)
        self.editor_damage_combo.setCurrentIndex(0)
        self.type_combo.setCurrentIndex(0)
        self.update_property_fields()
        self.name_input.setFocus()

    def _selected_part(self):
        part_id = getattr(self.sketch_view, "selected_part_id", None)
        if part_id is None:
            return None
        return next((p for p in self.project_state.parts if p.id == part_id), None)

    def _update_selected_part_display(self, part_id=None):
        if not hasattr(self, "selected_part_display"):
            return
        if part_id is None:
            part_id = getattr(self.sketch_view, "selected_part_id", None)
        part = next((p for p in self.project_state.parts if p.id == part_id), None) if part_id is not None else None
        if part is None:
            self.selected_part_display.setText("No part selected")
            if hasattr(self, "assignment_group"):
                self.assignment_group.setEnabled(False)
            self._sync_assignment_controls_from_part(None)
            self._sync_part_list_selection(None)
            return
        if hasattr(self, "assignment_group"):
            self.assignment_group.setEnabled(True)
        self.selected_part_display.setText(f"#{part.id} {part.name}")
        self._sync_assignment_controls_from_part(part)
        self._sync_part_list_selection(part.id)

    def _set_combo_data(self, combo, value, fallback_index=0):
        if combo is None:
            return
        index = combo.findData(value)
        combo.blockSignals(True)
        combo.setCurrentIndex(index if index >= 0 else fallback_index)
        combo.blockSignals(False)

    def _sync_assignment_controls_from_part(self, part):
        mode = getattr(part, "material_assignment_mode", "homogeneous") if part is not None else "homogeneous"
        heterogeneity_method = (
            getattr(part, "heterogeneity_method", "region_based") if part is not None else "region_based"
        )
        field_cfg = (
            normalize_material_field_config(copy.deepcopy(getattr(part, "material_field_config", {})))
            if part is not None
            else normalize_material_field_config({})
        )
        symmetry = getattr(part, "material_symmetry", "isotropic") if part is not None else "isotropic"
        behavior = getattr(part, "material_behavior", "elastic") if part is not None else "elastic"
        damage = getattr(part, "material_damage", "none") if part is not None else "none"
        config = (
            normalize_heterogeneity_config(copy.deepcopy(getattr(part, "heterogeneity_config", {})))
            if part is not None
            else normalize_heterogeneity_config({})
        )
        self._set_combo_data(self.assignment_mode_combo, mode)
        self._set_combo_data(self.heterogeneity_method_combo, heterogeneity_method)
        self._set_combo_data(self.material_field_property_combo, field_cfg.get("property_key", "E"))
        self._set_combo_data(self.material_field_type_combo, field_cfg.get("field_type", "linear_gradient"))
        self._set_combo_data(self.symmetry_combo, symmetry)
        self._set_combo_data(self.behavior_combo, behavior)
        self._set_combo_data(self.damage_combo, damage)
        if part is not None and getattr(part, "material_id", None) not in (None, "", -1):
            self._set_combo_data(self.active_material_combo, getattr(part, "material_id", None))
        self._assignment_heterogeneity_config = config
        self._assignment_material_field_config = field_cfg
        self._update_assignment_method_ui()

    def _set_form_row_visible(self, widget, visible):
        if widget is None:
            return
        try:
            widget.setVisible(bool(visible))
        except Exception:
            pass
        label = None
        try:
            label = self.assignment_form.labelForField(widget)
        except Exception:
            label = None
        if label is not None:
            label.setVisible(bool(visible))

    def _editor_material_symmetry(self):
        return str(self.editor_symmetry_combo.currentData() or "isotropic")

    def _editor_material_behavior(self):
        return str(self.type_combo.currentData() or "elastic")

    def _editor_material_damage(self):
        return str(self.editor_damage_combo.currentData() or "none")

    def _distribution_summary(self, method=None, config=None):
        method = str(method or self.heterogeneity_method_combo.currentData() or "region_based")
        cfg = normalize_heterogeneity_config(config if config is not None else self._assignment_heterogeneity_config)
        if method == "random_distribution":
            rows = []
            for item in cfg.get("materials", []):
                material = self._materials_store().get(item.get("material_id"))
                label = getattr(material, "name", str(item.get("material_id")))
                try:
                    pct = float(item.get("fraction", 0.0)) * 100.0
                except Exception:
                    pct = 0.0
                rows.append(f"{label} {pct:.1f}%")
            if not rows:
                return "Random material fractions not configured"
            seed = cfg.get("random_seed")
            seed_txt = f", seed={seed}" if seed not in (None, "") else ""
            return "; ".join(rows) + seed_txt
        if method == "field_gradient_distribution":
            labels = [key for key, expr in (cfg.get("expressions", {}) or {}).items() if str(expr or "").strip()]
            return "Spatial fields: " + (", ".join(labels) if labels else "use base material values")
        return "Per-region material assignment"

    def _material_field_summary(self, config=None):
        cfg = normalize_material_field_config(
            config if config is not None else self._assignment_material_field_config
        )
        prop = str(cfg.get("property_key", "E"))
        field_type = str(cfg.get("field_type", "linear_gradient"))
        if field_type == "linear_gradient":
            linear = cfg.get("linear_gradient", {})
            return (
                f"{prop}: {linear.get('min', 0.0):.4g} to {linear.get('max', 0.0):.4g} "
                f"along {linear.get('direction', 'x')}"
            )
        if field_type == "radial_gradient":
            radial = cfg.get("radial_gradient", {})
            return (
                f"{prop}: core {radial.get('core', 0.0):.4g}, shell {radial.get('shell', 0.0):.4g}, "
                f"r={radial.get('radius', 0.0):.4g}"
            )
        if field_type == "random_field":
            rnd = cfg.get("random_field", {})
            seed = rnd.get("seed")
            seed_txt = f", seed={seed}" if seed not in (None, "") else ""
            return (
                f"{prop}: mean {rnd.get('mean', 0.0):.4g}, std {rnd.get('std', 0.0):.4g}, "
                f"Lc {rnd.get('correlation_length', 0.0):.4g}{seed_txt}"
            )
        equation = (cfg.get("user_equation", {}) or {}).get("expression", "")
        return f"{prop}(x,y) = {equation or 'base value'}"

    def _update_assignment_method_ui(self):
        mode = str(self.assignment_mode_combo.currentData() or "homogeneous")
        method = str(self.heterogeneity_method_combo.currentData() or "region_based")
        field_type = str(self.material_field_type_combo.currentData() or "linear_gradient")
        is_heterogeneous = mode == "heterogeneous"
        is_material_field = mode == "material_field"
        self._set_form_row_visible(self.heterogeneity_method_combo, is_heterogeneous)
        self._set_form_row_visible(self.material_field_property_combo, is_material_field)
        self._set_form_row_visible(self.material_field_type_combo, is_material_field)
        self._set_form_row_visible(self.heterogeneity_summary, is_heterogeneous or is_material_field)
        self._set_form_row_visible(self.btn_configure_distribution, is_heterogeneous or is_material_field)
        if hasattr(self, "heterogeneity_summary"):
            if is_material_field:
                self.heterogeneity_summary.setText(
                    self._material_field_summary(config=self._assignment_material_field_config)
                )
            else:
                self.heterogeneity_summary.setText(
                    self._distribution_summary(method=method, config=self._assignment_heterogeneity_config)
                )
        if hasattr(self, "btn_configure_distribution"):
            self.btn_configure_distribution.setText(
                "Configure Field" if is_material_field else "Configure Distribution"
            )

    def _configure_assignment_details(self):
        mode = str(self.assignment_mode_combo.currentData() or "homogeneous")
        if mode == "material_field":
            self._configure_material_field()
            return
        self._configure_heterogeneity()

    def _configure_heterogeneity(self):
        method = str(self.heterogeneity_method_combo.currentData() or "region_based")
        config = normalize_heterogeneity_config(copy.deepcopy(self._assignment_heterogeneity_config))
        if method == "random_distribution":
            updated = self._open_random_distribution_dialog(config)
        elif method == "field_gradient_distribution":
            updated = self._open_field_distribution_dialog(config)
        else:
            QMessageBox.information(
                self,
                "Region Based",
                "Region-based heterogeneity uses each part/region's assigned material during export.",
            )
            updated = config
        if updated is None:
            return
        self._assignment_heterogeneity_config = normalize_heterogeneity_config(updated)
        self._update_assignment_method_ui()

    def _configure_material_field(self):
        field_type = str(self.material_field_type_combo.currentData() or "linear_gradient")
        config = normalize_material_field_config(copy.deepcopy(self._assignment_material_field_config))
        config["property_key"] = str(self.material_field_property_combo.currentData() or "E")
        config["field_type"] = field_type
        if field_type == "linear_gradient":
            updated = self._open_linear_material_field_dialog(config)
        elif field_type == "radial_gradient":
            updated = self._open_radial_material_field_dialog(config)
        elif field_type == "random_field":
            updated = self._open_random_material_field_dialog(config)
        else:
            updated = self._open_user_equation_field_dialog(config)
        if updated is None:
            return
        self._assignment_material_field_config = normalize_material_field_config(updated)
        self._update_assignment_method_ui()

    def _open_random_distribution_dialog(self, config):
        dialog = QDialog(self)
        dialog.setWindowTitle("Random Material Distribution")
        layout = QVBoxLayout(dialog)
        _apply_layout_metrics(layout, margins=(12, 12, 12, 12), spacing=8)
        help_label = QLabel("Set material fractions for the selected part. Fractions are normalized during export.")
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        seed_form = QFormLayout()
        seed_input = QLineEdit("" if config.get("random_seed") in (None, "") else str(config.get("random_seed")))
        seed_form.addRow("Random Seed", seed_input)
        layout.addLayout(seed_form)
        table = QTableWidget(0, 2, dialog)
        table.setHorizontalHeaderLabels(["Material", "Fraction (%)"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        layout.addWidget(table, 1)

        def _add_distribution_row(material_id=None, fraction=0.0):
            row = table.rowCount()
            table.insertRow(row)
            combo = QComboBox(table)
            for serial, mat in sorted(self._materials_store().items()):
                combo.addItem(f"{mat.name} ({mat.mat_type})", int(serial))
            if material_id is not None:
                idx = combo.findData(int(material_id))
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            spin = QDoubleSpinBox(table)
            spin.setDecimals(6)
            spin.setRange(0.0, 100.0)
            spin.setSingleStep(5.0)
            spin.setSuffix(" %")
            spin.setValue(float(fraction or 0.0) * 100.0)
            table.setCellWidget(row, 0, combo)
            table.setCellWidget(row, 1, spin)

        rows = list(config.get("materials", []))
        if not rows:
            current_mat = self.active_material_combo.currentData()
            try:
                current_mat = int(current_mat)
            except Exception:
                current_mat = None
            if current_mat not in self._materials_store():
                current_mat = next(iter(sorted(self._materials_store().keys())), None)
            if current_mat is not None:
                rows = [{"material_id": current_mat, "fraction": 1.0}]
        for item in rows:
            _add_distribution_row(item.get("material_id"), item.get("fraction", 0.0))

        button_row = QHBoxLayout()
        add_btn = QPushButton("Add Row")
        del_btn = QPushButton("Delete Row")
        button_row.addWidget(add_btn)
        button_row.addWidget(del_btn)
        layout.addLayout(button_row)
        add_btn.clicked.connect(lambda: _add_distribution_row())
        del_btn.clicked.connect(lambda: table.removeRow(table.currentRow()) if table.currentRow() >= 0 else None)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.Accepted:
            return None

        materials = []
        for row in range(table.rowCount()):
            combo = table.cellWidget(row, 0)
            spin = table.cellWidget(row, 1)
            if combo is None or spin is None:
                continue
            try:
                material_id = int(combo.currentData())
                fraction = float(spin.value()) / 100.0
            except Exception:
                continue
            if fraction <= 0.0:
                continue
            materials.append({"material_id": material_id, "fraction": fraction})
        updated = normalize_heterogeneity_config(config)
        updated["materials"] = materials
        seed_text = seed_input.text().strip()
        try:
            updated["random_seed"] = int(seed_text) if seed_text else None
        except Exception:
            updated["random_seed"] = None
        return updated

    def _open_field_distribution_dialog(self, config):
        dialog = QDialog(self)
        dialog.setWindowTitle("Field / Gradient Distribution")
        layout = QVBoxLayout(dialog)
        _apply_layout_metrics(layout, margins=(12, 12, 12, 12), spacing=8)
        help_label = QLabel(
            "Expressions are evaluated at each triangle centroid during export.\n"
            "Available variables: x, y, xmin, xmax, ymin, ymax, width, height, L, xc, yc, r, theta,\n"
            "base_E, base_nu, base_rho, base_fail_SE, base_c, pi."
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        form = QFormLayout()
        edits = {}
        expressions = (config.get("expressions", {}) or {})
        for key in FIELD_DISTRIBUTION_PROPERTY_KEYS:
            edit = QLineEdit(str(expressions.get(key, "") or ""))
            edits[key] = edit
            form.addRow(f"{key}(x,y)", edit)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.Accepted:
            return None
        updated = normalize_heterogeneity_config(config)
        updated["expressions"] = {key: edits[key].text().strip() for key in FIELD_DISTRIBUTION_PROPERTY_KEYS}
        return updated

    def _open_linear_material_field_dialog(self, config):
        dialog = QDialog(self)
        dialog.setWindowTitle("Linear Gradient Field")
        layout = QFormLayout(dialog)
        _apply_layout_metrics(layout, margins=(12, 12, 12, 12), spacing=8)
        linear = dict((config.get("linear_gradient", {}) or {}))
        prop_key = str(config.get("property_key", "E"))
        min_edit = QDoubleSpinBox(dialog)
        min_edit.setRange(-1e15, 1e15)
        min_edit.setDecimals(6)
        min_edit.setValue(float(linear.get("min", 0.0) or 0.0))
        max_edit = QDoubleSpinBox(dialog)
        max_edit.setRange(-1e15, 1e15)
        max_edit.setDecimals(6)
        max_edit.setValue(float(linear.get("max", 0.0) or 0.0))
        direction_combo = QComboBox(dialog)
        for text, value in (("X", "x"), ("Y", "y"), ("Diagonal", "diag")):
            direction_combo.addItem(text, value)
        idx = direction_combo.findData(str(linear.get("direction", "x") or "x"))
        direction_combo.setCurrentIndex(idx if idx >= 0 else 0)
        layout.addRow(f"{prop_key}_min", min_edit)
        layout.addRow(f"{prop_key}_max", max_edit)
        layout.addRow("Direction", direction_combo)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec() != QDialog.Accepted:
            return None
        updated = normalize_material_field_config(config)
        updated["linear_gradient"] = {
            "min": float(min_edit.value()),
            "max": float(max_edit.value()),
            "direction": str(direction_combo.currentData() or "x"),
        }
        return updated

    def _open_radial_material_field_dialog(self, config):
        dialog = QDialog(self)
        dialog.setWindowTitle("Radial Gradient Field")
        layout = QFormLayout(dialog)
        _apply_layout_metrics(layout, margins=(12, 12, 12, 12), spacing=8)
        radial = dict((config.get("radial_gradient", {}) or {}))
        prop_key = str(config.get("property_key", "E"))
        widgets = {}
        for label, key in (
            ("Center X", "center_x"),
            ("Center Y", "center_y"),
            ("Radius", "radius"),
            (f"{prop_key}_core", "core"),
            (f"{prop_key}_shell", "shell"),
        ):
            widget = QDoubleSpinBox(dialog)
            widget.setRange(-1e15, 1e15)
            widget.setDecimals(6)
            widget.setValue(float(radial.get(key, 0.0) or 0.0))
            widgets[key] = widget
            layout.addRow(label, widget)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec() != QDialog.Accepted:
            return None
        updated = normalize_material_field_config(config)
        updated["radial_gradient"] = {key: float(widget.value()) for key, widget in widgets.items()}
        return updated

    def _open_random_material_field_dialog(self, config):
        dialog = QDialog(self)
        dialog.setWindowTitle("Random Field")
        layout = QFormLayout(dialog)
        _apply_layout_metrics(layout, margins=(12, 12, 12, 12), spacing=8)
        rnd = dict((config.get("random_field", {}) or {}))
        prop_key = str(config.get("property_key", "E"))
        mean_edit = QDoubleSpinBox(dialog)
        mean_edit.setRange(-1e15, 1e15)
        mean_edit.setDecimals(6)
        mean_edit.setValue(float(rnd.get("mean", 0.0) or 0.0))
        std_edit = QDoubleSpinBox(dialog)
        std_edit.setRange(0.0, 1e15)
        std_edit.setDecimals(6)
        std_edit.setValue(float(rnd.get("std", 0.0) or 0.0))
        corr_edit = QDoubleSpinBox(dialog)
        corr_edit.setRange(1e-12, 1e15)
        corr_edit.setDecimals(6)
        corr_edit.setValue(max(float(rnd.get("correlation_length", 1.0) or 1.0), 1e-12))
        seed_edit = QLineEdit("" if rnd.get("seed") in (None, "") else str(rnd.get("seed")))
        layout.addRow(f"{prop_key}_mean", mean_edit)
        layout.addRow(f"{prop_key}_std", std_edit)
        layout.addRow("Correlation Length", corr_edit)
        layout.addRow("Seed", seed_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec() != QDialog.Accepted:
            return None
        updated = normalize_material_field_config(config)
        try:
            seed_value = int(seed_edit.text().strip()) if seed_edit.text().strip() else None
        except Exception:
            seed_value = None
        updated["random_field"] = {
            "mean": float(mean_edit.value()),
            "std": float(std_edit.value()),
            "correlation_length": float(corr_edit.value()),
            "seed": seed_value,
        }
        return updated

    def _open_user_equation_field_dialog(self, config):
        dialog = QDialog(self)
        dialog.setWindowTitle("User Equation Field")
        layout = QVBoxLayout(dialog)
        _apply_layout_metrics(layout, margins=(12, 12, 12, 12), spacing=8)
        prop_key = str(config.get("property_key", "E"))
        help_label = QLabel(
            f"Define {prop_key}(x,y). Available variables: x, y, xmin, xmax, ymin, ymax, width, height, "
            "L, xc, yc, r, theta, base_value, pi."
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        edit = QLineEdit(str(((config.get('user_equation', {}) or {}).get('expression', '')) or ''))
        form = QFormLayout()
        form.addRow(f"{prop_key}(x,y)", edit)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.Accepted:
            return None
        updated = normalize_material_field_config(config)
        updated["user_equation"] = {"expression": edit.text().strip()}
        return updated

    def refresh_active_material_combo(self):
        self.active_material_combo.blockSignals(True)
        current_selection = self.active_material_combo.currentData()
        self.active_material_combo.clear()
        self.active_material_combo.addItem("[No Material]", -1)

        for serial, mat in sorted(self._materials_store().items()):
            mat_display = f"{mat.name} ({behavior_label(getattr(mat, 'behavior', infer_behavior_from_mat_type(mat.mat_type)))})"
            self.active_material_combo.addItem(mat_display, serial)

        index = self.active_material_combo.findData(current_selection)
        self.active_material_combo.setCurrentIndex(index if index != -1 else 0)
        self.active_material_combo.blockSignals(False)
        self.on_active_material_selected()

    def on_material_select_from_list(self, item, column):
        try:
            serial = int(item.text(2))
            mat = self._materials_store().get(serial)
            if not mat:
                return
            self._material_pick_source = "project"
            self.set_active_material(serial)
            if self._registry_editor_enabled:
                self._load_material_into_editor(mat)
            self._sync_material_selection_to_inspector()
        except (ValueError, IndexError):
            pass

    def _load_material_into_editor(self, mat):
        if not self._registry_editor_enabled:
            return
        self.name_input.blockSignals(True)
        self.name_input.setText(mat.name)
        self.name_input.blockSignals(False)
        symmetry_index = self.editor_symmetry_combo.findData(getattr(mat, "symmetry", "isotropic"))
        self.editor_symmetry_combo.blockSignals(True)
        self.editor_symmetry_combo.setCurrentIndex(symmetry_index if symmetry_index >= 0 else 0)
        self.editor_symmetry_combo.blockSignals(False)
        damage_index = self.editor_damage_combo.findData(getattr(mat, "damage", "none"))
        self.editor_damage_combo.blockSignals(True)
        self.editor_damage_combo.setCurrentIndex(damage_index if damage_index >= 0 else 0)
        self.editor_damage_combo.blockSignals(False)
        behavior_index = self.type_combo.findData(getattr(mat, "behavior", infer_behavior_from_mat_type(mat.mat_type)))
        self.type_combo.blockSignals(True)
        self.type_combo.setCurrentIndex(behavior_index if behavior_index >= 0 else 0)
        self.type_combo.blockSignals(False)
        self.update_property_fields()
        QTimer.singleShot(0, lambda: self.populate_properties(mat.properties))

    def populate_properties(self, properties):
        for prop_key, widget in self.prop_widgets.items():
            if prop_key in properties:
                widget.setValue(properties[prop_key])

    def clear_property_fields(self):
        while self.properties_form.count():
            child = self.properties_form.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.prop_widgets.clear()

    def update_property_fields(self):
        if not self._registry_editor_enabled:
            return
        self.clear_property_fields()
        behavior = self._editor_material_behavior()
        symmetry = self._editor_material_symmetry()
        damage = self._editor_material_damage()
        prop_schema = material_property_schema(behavior, symmetry, damage)
        for field_spec in prop_schema:
            prop = field_spec["key"]
            label = field_spec.get("name", prop.replace("_", " ").title())
            widget = QDoubleSpinBox()
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            widget.setMinimumHeight(26)
            widget.setDecimals(int(field_spec.get("decimals", 6)))
            widget.setSingleStep(float(field_spec.get("step", 1.0)))
            widget.setRange(float(field_spec.get("minimum", -1e12)), float(field_spec.get("maximum", 1e12)))
            widget.setValue(float(field_spec.get("default", 0.0)))
            tooltip = str(field_spec.get("tooltip", "") or "")
            if tooltip:
                widget.setToolTip(tooltip)

            self.prop_widgets[prop] = widget
            self.properties_form.addRow(label, widget)

    def create_material(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Material name cannot be empty.")
            return

        behavior = self._editor_material_behavior()
        damage = self._editor_material_damage()
        mat_type = legacy_mat_type_for_behavior(behavior)
        properties = {key: widget.value() for key, widget in self.prop_widgets.items()}
        main = self.window()
        execute_command = getattr(main, "execute_app_command", None) if main is not None else None
        if self.material_controller is None or not callable(execute_command):
            QMessageBox.warning(self, "Material Controller", "Material command bus is not available.")
            return
        result = execute_command(
            AddMaterialCommand(
                name=name,
                mat_type=mat_type,
                behavior=behavior,
                damage=damage,
                symmetry=str(self.editor_symmetry_combo.currentData() or "isotropic"),
                properties=properties,
                selected_part_id=None,
                auto_assign_selected_part=False,
                announce_assignment=False,
            )
        )
        mat = result["material"]
        created = bool(result.get("created"))

        if created:
            QMessageBox.information(self, "Success", f"Material '{name}' (s={mat.serial}) created.")
        else:
            QMessageBox.information(self, "Success", f"Material '{name}' updated.")

        self.sketch_view.set_current_material(mat.serial)
        self.set_active_material(mat.serial)

        self.refresh_material_list()
        self.set_active_material(mat.serial)
        self._load_material_into_editor(mat)

    def refresh_material_list(self):
        self.mat_list.clear()
        for serial, mat in sorted(self._materials_store().items()):
            behavior_tag = ""
            metadata = getattr(mat, "metadata", {}) or {}
            if isinstance(metadata, dict):
                behavior_tag = str(metadata.get("behavior_tag", "") or "")
            behavior_display = behavior_tag or behavior_label(
                getattr(mat, "behavior", infer_behavior_from_mat_type(mat.mat_type))
            )
            item = QTreeWidgetItem([mat.name, behavior_display, str(mat.serial)])
            color = self.sketch_view.material_color_map.get(serial)
            if color:
                for col in range(self.mat_list.columnCount()):
                    item.setBackground(col, color)
            self.mat_list.addTopLevelItem(item)
        self.refresh_active_material_combo()
        self.sketch_view.redraw()
        self.sketch_view.materialsChanged.emit()
        self.sketch_view.partsChanged.emit()
        self.refresh_part_list()
        self._update_selected_part_display(getattr(self.sketch_view, "selected_part_id", None))

    def delete_selected_material(self):
        item = self.mat_list.currentItem()
        if item is None:
            QMessageBox.warning(self, "Delete Material", "Select a material first.")
            return
        try:
            serial = int(item.text(2))
        except Exception:
            QMessageBox.warning(self, "Delete Material", "Selected material is invalid.")
            return
        material = self._materials_store().get(serial)
        if material is None:
            return
        reply = QMessageBox.question(
            self,
            "Delete Material",
            f"Delete material '{material.name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.sketch_view.push_undo_state()
        self._materials_store().pop(serial, None)
        affected_parts = [
            part
            for part in self.project_state.parts
            if getattr(part, "material_id", None) == serial
        ]
        for part in affected_parts:
            part.material_id = None
        if getattr(self.sketch_view, "current_material_id", None) == serial:
            self.sketch_view.set_current_material(-1)
        self.refresh_material_list()
        if affected_parts:
            count = len(affected_parts)
            _show_panel_notice(
                getattr(self, "_cross_tab_notice", None),
                f"Deleted material '{material.name}'. Cleared assignment from "
                f"{count} part{'s' if count != 1 else ''} (Geometry tab).",
            )
        else:
            _show_panel_notice(
                getattr(self, "_cross_tab_notice", None),
                f"Deleted material '{material.name}'.",
            )

    def _update_responsive_layout(self):
        width = int(self.contentsRect().width() or self.width() or 0)
        _reflow_button_grid(
            self._material_button_grid,
            self._material_action_buttons,
            width - 24,
            min_button_width=120,
            max_columns=3,
        )
        compact = width < 430
        icon_only = width < 350
        _set_responsive_button_text(
            self.btn_create,
            full=self._create_label_full,
            compact="Update" if compact else None,
            icon_only=icon_only,
        )
        _set_responsive_button_text(
            self.btn_new,
            full=self._new_label_full,
            compact="New",
            icon_only=icon_only,
        )
        _set_responsive_button_text(
            self.btn_delete,
            full=self._delete_label_full,
            compact="Delete",
            icon_only=icon_only,
        )
        _set_responsive_button_text(
            self.btn_assign,
            full=self._assign_label_full,
            compact="Assign",
            icon_only=icon_only,
        )
        row_wrap = QFormLayout.WrapLongRows if width < 360 else QFormLayout.DontWrapRows
        self.properties_form.setRowWrapPolicy(row_wrap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_responsive_layout()

    def assign_to_selected_part(self):
        part = self._selected_part()
        if not part:
            QMessageBox.warning(
                self,
                "Select Part",
                "Select a part first in the viewport or Geometry panel.",
            )
            return

        mat_id = None
        if self._material_pick_source == "csv":
            record = self._current_csv_record()
            if record is None:
                QMessageBox.warning(self, "Select Material", "Select a material first.")
                return
            created = self._add_material_from_record(record, damage_on=bool(self.csv_damage_combo.currentData()))
            if created is None:
                return
            mat_id = created.serial
        else:
            mat_id = self.active_material_combo.currentData()
            try:
                mat_id = int(mat_id)
            except (TypeError, ValueError):
                mat_id = None
            if mat_id is None or mat_id == -1:
                record = self._current_csv_record()
                if record is None:
                    QMessageBox.warning(self, "Select Material", "Select a material first.")
                    return
                created = self._add_material_from_record(record, damage_on=bool(self.csv_damage_combo.currentData()))
                if created is None:
                    return
                mat_id = created.serial

        assigned = self._assign_material_id_to_part(part, mat_id)
        if assigned:
            self._update_selected_part_display(part.id)

    def _materials_store(self):
        return self.project_state.materials

    def refresh_part_list(self):
        if not hasattr(self, "part_select_combo"):
            return
        self.part_select_combo.blockSignals(True)
        self.part_select_combo.clear()
        self.part_select_combo.addItem("Select Part", None)
        materials = self._materials_store()
        for part in self.project_state.parts:
            mat_name = "Unassigned"
            mat_id = getattr(part, "material_id", None)
            if mat_id in materials:
                mat_name = getattr(materials[mat_id], "name", "Assigned")
            label = f"#{part.id} {part.name}  ·  {mat_name}"
            self.part_select_combo.addItem(label, int(part.id))
        self._sync_part_list_selection(getattr(self.sketch_view, "selected_part_id", None))
        self.part_select_combo.blockSignals(False)

    def _sync_part_list_selection(self, part_id):
        if not hasattr(self, "part_select_combo"):
            return
        self.part_select_combo.blockSignals(True)
        try:
            if part_id is None:
                self.part_select_combo.setCurrentIndex(0)
                return
            idx = self.part_select_combo.findData(int(part_id))
            self.part_select_combo.setCurrentIndex(idx if idx >= 0 else 0)
        finally:
            self.part_select_combo.blockSignals(False)

    def _on_part_select_combo_changed(self):
        if not hasattr(self, "part_select_combo"):
            return
        part_id = self.part_select_combo.currentData()
        if part_id is None:
            try:
                self.sketch_view.set_selected_part(None, emit_signal=True)
            except Exception:
                pass
            return
        try:
            self.sketch_view.set_selected_part(int(part_id), emit_signal=True)
        except Exception:
            pass

    def _assign_material_id_to_part(self, part, mat_id):
        if part is None:
            return False
        if self.material_controller is None:
            QMessageBox.warning(self, "Material Controller", "Material controller is not available.")
            return False
        assigned = self.material_controller.assign_material_to_part(
            part,
            mat_id,
            announce=False,
            assignment_mode=str(self.assignment_mode_combo.currentData() or "homogeneous"),
            heterogeneity_method=str(self.heterogeneity_method_combo.currentData() or "region_based"),
            heterogeneity_config=copy.deepcopy(self._assignment_heterogeneity_config),
            material_field_config=copy.deepcopy(self._assignment_material_field_config),
            symmetry=str(self.symmetry_combo.currentData() or "isotropic"),
            behavior=str(self.behavior_combo.currentData() or "elastic"),
            damage=str(self.damage_combo.currentData() or "none"),
        )
        if assigned:
            self.refresh_part_list()
        return assigned

    def _assign_csv_record_to_part(self, record, part):
        if record is None or part is None:
            return False
        damage_on = bool(self.csv_damage_combo.currentData())
        created = self._add_material_from_record(record, damage_on=damage_on)
        if created is None:
            return False
        return self._assign_material_id_to_part(part, created.serial)

    def startDrag(self, supportedActions):
        item = self.mat_list.currentItem()
        if not item:
            return

        serial = item.text(2)
        mime = QMimeData()
        mime.setText(serial)

        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.CopyAction)


class TimeProfileDialog(QDialog):
    def __init__(self, title, total_time, mode, profile=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._mode = mode
        self._total_time = max(0.0, float(total_time))
        self._profile = profile or []
        self._updating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        help_text = (
            "Define piecewise values as a function of time t.\n"
            "Start times are fixed and update automatically when you change end times."
        )
        help_label = QLabel(help_text)
        help_label.setWordWrap(True)
        help_label.setObjectName("MinorStatusLabel")
        layout.addWidget(help_label)

        self.table = QTableWidget()
        if self._mode == "force":
            headers = ["Start t", "End t", "Fx(t)", "Fy(t)"]
        else:
            headers = ["Start t", "End t", "V(t)"]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table, 1)

        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("Add Row")
        self.remove_btn = QPushButton("Remove Row")
        self.add_btn.clicked.connect(self._add_row)
        self.remove_btn.clicked.connect(self._remove_row)
        btn_row.addWidget(self.add_btn)
        btn_row.addWidget(self.remove_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        action_row = QHBoxLayout()
        action_row.addStretch(1)
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        action_row.addWidget(ok_btn)
        action_row.addWidget(cancel_btn)
        layout.addLayout(action_row)

        if not self._profile:
            self._profile = [{"t0": 0.0, "t1": self._total_time, "expr": "0"}]
        self._load_profile(self._profile)

    def _new_time_spin(self, value, read_only=False, maximum=None):
        spin = QDoubleSpinBox()
        spin.setDecimals(6)
        spin.setRange(0.0, max(self._total_time, 0.0))
        if maximum is not None:
            spin.setMaximum(maximum)
        spin.setSingleStep(1.0)
        spin.setValue(float(value))
        if read_only:
            spin.setReadOnly(True)
            spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        return spin

    def _load_profile(self, profile):
        self._updating = True
        self.table.setRowCount(0)
        for entry in profile:
            t0 = float(entry.get("t0", 0.0))
            t1 = float(entry.get("t1", self._total_time))
            if self._mode == "force":
                fx = entry.get("fx", entry.get("expr_fx", entry.get("expr", "0")))
                fy = entry.get("fy", entry.get("expr_fy", "0"))
                self._append_row(t0, t1, fx, fy)
            else:
                expr = entry.get("expr", entry.get("v", "0"))
                if "vx" in entry or "vy" in entry:
                    expr = entry.get("vx") if entry.get("vx") not in (None, "") else entry.get("vy", "0")
                self._append_row(t0, t1, expr)
        self._normalize_rows()
        self._updating = False

    def _new_expr_combo(self, value):
        combo = QComboBox()
        presets = [
            "0",
            "t",
            "10*t",
            "t**2",
            "t**3",
            "pi",
            "e",
            "sin(t)",
            "cos(t)",
            "tan(t)",
            "sin(2*pi*t)",
            "exp(t)",
            "log(t)",
            "sqrt(t)",
            "abs(t)",
            "min(t, 1)",
            "max(t, 1)",
            "sinh(t)",
            "cosh(t)",
            "tanh(t)",
        ]
        combo.addItems(presets)
        return _configure_expression_combo(combo, value)

    def _append_row(self, t0, t1, expr, expr_y=None):
        row = self.table.rowCount()
        self.table.insertRow(row)
        start_spin = self._new_time_spin(t0, read_only=True)
        end_spin = self._new_time_spin(t1)
        end_spin.valueChanged.connect(lambda _v, r=row: self._on_end_changed(r))
        self.table.setCellWidget(row, 0, start_spin)
        self.table.setCellWidget(row, 1, end_spin)
        self.table.setCellWidget(row, 2, self._new_expr_combo(expr))
        if self._mode == "force":
            self.table.setCellWidget(row, 3, self._new_expr_combo(expr_y if expr_y is not None else "0"))

    def _on_end_changed(self, row):
        if self._updating:
            return
        self._normalize_rows()

    def _normalize_rows(self):
        self._updating = True
        row_count = self.table.rowCount()
        if row_count == 0:
            self._updating = False
            return
        for row in range(row_count):
            start_spin = self.table.cellWidget(row, 0)
            end_spin = self.table.cellWidget(row, 1)
            if row == 0:
                start_spin.setValue(0.0)
            else:
                prev_end = self.table.cellWidget(row - 1, 1).value()
                start_spin.setValue(prev_end + 1.0)
            end_spin.setReadOnly(False)
            end_spin.setButtonSymbols(QAbstractSpinBox.UpDownArrows)
            end_spin.setMaximum(self._total_time)
            if row == row_count - 1 and end_spin.value() <= 0.0:
                end_spin.setValue(self._total_time)
            if end_spin.value() > self._total_time:
                end_spin.setValue(self._total_time)
        self._updating = False

    def _add_row(self):
        row_count = self.table.rowCount()
        if row_count == 0:
            self._append_row(0.0, self._total_time, "0", "0")
            self._normalize_rows()
            return
        last_end = self.table.cellWidget(row_count - 1, 1).value()
        if last_end >= self._total_time:
            QMessageBox.information(
                self,
                "Add Row",
                "Reduce the previous end time to add another row.",
            )
            return
        start = last_end + 1.0
        end = min(self._total_time, start + 1.0)
        if self._mode == "force":
            self._append_row(start, end, "0", "0")
        else:
            self._append_row(start, end, "0")
        self._normalize_rows()

    def _remove_row(self):
        row = self.table.currentRow()
        if row < 0:
            return
        if self.table.rowCount() <= 1:
            QMessageBox.information(self, "Remove Row", "At least one row is required.")
            return
        self.table.removeRow(row)
        self._normalize_rows()

    def get_profile(self):
        profile = []
        row_count = self.table.rowCount()
        for row in range(row_count):
            start_spin = self.table.cellWidget(row, 0)
            end_spin = self.table.cellWidget(row, 1)
            t0 = float(start_spin.value())
            t1 = float(end_spin.value())
            if self._mode == "force":
                fx_widget = self.table.cellWidget(row, 2)
                fy_widget = self.table.cellWidget(row, 3)
                fx = fx_widget.currentText().strip() if isinstance(fx_widget, QComboBox) else "0"
                fy = fy_widget.currentText().strip() if isinstance(fy_widget, QComboBox) else "0"
                profile.append({"t0": t0, "t1": t1, "fx": fx or "0", "fy": fy or "0"})
            else:
                expr_widget = self.table.cellWidget(row, 2)
                expr = expr_widget.currentText().strip() if isinstance(expr_widget, QComboBox) else "0"
                profile.append({"t0": t0, "t1": t1, "expr": expr or "0"})
        return profile


class ScalarTimeProfileDialog(QDialog):
    def __init__(self, title, total_time, default_expr="0", profile=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._total_time = max(0.0, float(total_time))
        self._profile = profile or []
        self._default_expr = default_expr
        self._updating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        help_text = (
            "Define piecewise values as a function of time t.\n"
            "Start times are fixed and update automatically when you change end times."
        )
        help_label = QLabel(help_text)
        help_label.setWordWrap(True)
        help_label.setObjectName("MinorStatusLabel")
        layout.addWidget(help_label)

        self.table = QTableWidget()
        headers = ["Start t", "End t", "Expr(t)"]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table, 1)

        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("Add Row")
        self.remove_btn = QPushButton("Remove Row")
        self.add_btn.clicked.connect(self._add_row)
        self.remove_btn.clicked.connect(self._remove_row)
        btn_row.addWidget(self.add_btn)
        btn_row.addWidget(self.remove_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        action_row = QHBoxLayout()
        action_row.addStretch(1)
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        action_row.addWidget(ok_btn)
        action_row.addWidget(cancel_btn)
        layout.addLayout(action_row)

        if not self._profile:
            self._profile = [{"t0": 0.0, "t1": self._total_time, "expr": self._default_expr}]
        self._load_profile(self._profile)

    def _new_time_spin(self, value, read_only=False):
        spin = QDoubleSpinBox()
        spin.setDecimals(6)
        spin.setRange(0.0, max(self._total_time, 0.0))
        spin.setSingleStep(1.0)
        spin.setValue(float(value))
        if read_only:
            spin.setReadOnly(True)
            spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        return spin

    def _new_expr_combo(self, value):
        combo = QComboBox()
        presets = [
            "0",
            "t",
            "10*t",
            "t**2",
            "t**3",
            "pi",
            "e",
            "sin(t)",
            "cos(t)",
            "tan(t)",
            "sin(2*pi*t)",
            "exp(t)",
            "log(t)",
            "sqrt(t)",
            "abs(t)",
            "min(t, 1)",
            "max(t, 1)",
            "sinh(t)",
            "cosh(t)",
            "tanh(t)",
        ]
        combo.addItems(presets)
        return _configure_expression_combo(combo, value)

    def _load_profile(self, profile):
        self._updating = True
        self.table.setRowCount(0)
        for entry in profile:
            t0 = float(entry.get("t0", 0.0))
            t1 = float(entry.get("t1", self._total_time))
            expr = entry.get("expr", self._default_expr)
            self._append_row(t0, t1, expr)
        self._normalize_rows()
        self._updating = False

    def _append_row(self, t0, t1, expr):
        row = self.table.rowCount()
        self.table.insertRow(row)
        start_spin = self._new_time_spin(t0, read_only=True)
        end_spin = self._new_time_spin(t1)
        end_spin.valueChanged.connect(lambda _v, r=row: self._on_end_changed(r))
        self.table.setCellWidget(row, 0, start_spin)
        self.table.setCellWidget(row, 1, end_spin)
        self.table.setCellWidget(row, 2, self._new_expr_combo(expr))

    def _on_end_changed(self, row):
        if self._updating:
            return
        self._normalize_rows()

    def _normalize_rows(self):
        self._updating = True
        row_count = self.table.rowCount()
        if row_count == 0:
            self._updating = False
            return
        for row in range(row_count):
            start_spin = self.table.cellWidget(row, 0)
            end_spin = self.table.cellWidget(row, 1)
            if row == 0:
                start_spin.setValue(0.0)
            else:
                prev_end = self.table.cellWidget(row - 1, 1).value()
                start_spin.setValue(prev_end + 1.0)
            end_spin.setReadOnly(False)
            end_spin.setButtonSymbols(QAbstractSpinBox.UpDownArrows)
            end_spin.setMaximum(self._total_time)
            if row == row_count - 1 and end_spin.value() <= 0.0:
                end_spin.setValue(self._total_time)
            if end_spin.value() > self._total_time:
                end_spin.setValue(self._total_time)
        self._updating = False

    def _add_row(self):
        row_count = self.table.rowCount()
        if row_count == 0:
            self._append_row(0.0, self._total_time, self._default_expr)
            self._normalize_rows()
            return
        last_end = self.table.cellWidget(row_count - 1, 1).value()
        if last_end >= self._total_time:
            QMessageBox.information(
                self,
                "Add Row",
                "Reduce the previous end time to add another row.",
            )
            return
        start = last_end + 1.0
        end = min(self._total_time, start + 1.0)
        self._append_row(start, end, self._default_expr)
        self._normalize_rows()

    def _remove_row(self):
        row = self.table.currentRow()
        if row < 0:
            return
        if self.table.rowCount() <= 1:
            QMessageBox.information(self, "Remove Row", "At least one row is required.")
            return
        self.table.removeRow(row)
        self._normalize_rows()

    def get_profile(self):
        profile = []
        row_count = self.table.rowCount()
        for row in range(row_count):
            start_spin = self.table.cellWidget(row, 0)
            end_spin = self.table.cellWidget(row, 1)
            expr_widget = self.table.cellWidget(row, 2)
            t0 = float(start_spin.value())
            t1 = float(end_spin.value())
            expr = expr_widget.currentText().strip() if isinstance(expr_widget, QComboBox) else self._default_expr
            profile.append({"t0": t0, "t1": t1, "expr": expr or self._default_expr})
        return profile


class ScalarTimeProfileEditor(QWidget):
    applied = Signal(list)
    canceled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._total_time = 0.0
        self._default_expr = "0"
        self._time_mode = "percent"
        self._source_mode = "percent"
        self._updating = False

        layout = QVBoxLayout(self)
        _apply_layout_metrics(layout, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.title_label = QLabel("Time Profile")
        self.title_label.setObjectName("SectionTitleLabel")
        self.time_mode_label = QLabel("Time format")
        self.time_mode_combo = QComboBox()
        self.time_mode_combo.addItems(["Percent"])
        self.time_mode_combo.setEnabled(False)
        layout.addWidget(self.title_label)
        self.time_mode_row = QHBoxLayout()
        _apply_layout_metrics(self.time_mode_row, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        self.time_mode_row.addWidget(self.time_mode_label)
        self.time_mode_row.addWidget(self.time_mode_combo, 1)
        layout.addLayout(self.time_mode_row)

        self.table_label = QLabel("Piecewise segments")
        self.table_label.setObjectName("MinorStatusLabel")
        layout.addWidget(self.table_label)
        self.table = QTableWidget()
        headers = ["Start t", "End t", "Expr(t)"]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        header = self.table.horizontalHeader()
        header.setMinimumSectionSize(48)
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table.setColumnWidth(0, 84)
        self.table.setColumnWidth(1, 84)
        self.table.setColumnWidth(2, 240)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.setMinimumHeight(132)
        layout.addWidget(self.table, 1)

        self.btn_row = QHBoxLayout()
        _apply_layout_metrics(self.btn_row, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        self.add_btn = QPushButton("Add Row")
        self.remove_btn = QPushButton("Remove Row")
        self.add_btn.clicked.connect(self._add_row)
        self.remove_btn.clicked.connect(self._remove_row)
        self.btn_row.addWidget(self.add_btn, 1)
        self.btn_row.addWidget(self.remove_btn, 1)
        layout.addLayout(self.btn_row)

        self.action_row = QHBoxLayout()
        _apply_layout_metrics(self.action_row, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        self.apply_btn = QPushButton("Apply")
        self.cancel_btn = QPushButton("Cancel")
        self.apply_btn.clicked.connect(self._emit_apply)
        self.cancel_btn.clicked.connect(self._emit_cancel)
        self.action_row.addWidget(self.apply_btn, 1)
        self.action_row.addWidget(self.cancel_btn, 1)
        layout.addLayout(self.action_row)

    def set_profile(self, title, total_time, profile=None, default_expr="0", mode="absolute"):
        self.title_label.setText(title or "Time Profile")
        self._total_time = max(0.0, float(total_time))
        self._default_expr = default_expr or "0"
        self._source_mode = "percent" if str(mode).lower().startswith("percent") else "absolute"
        self._time_mode = "percent"
        self.time_mode_combo.blockSignals(True)
        self.time_mode_combo.setCurrentIndex(0)
        self.time_mode_combo.blockSignals(False)
        if not profile:
            profile = [{"t0": 0.0, "t1": self._display_max(), "expr": self._default_expr}]
        self._load_profile(profile)

    def get_time_mode(self):
        return self._time_mode

    def _update_responsive_layout(self):
        width = int(self.contentsRect().width() or self.width() or 0)
        if width <= 0:
            return
        narrow = width < 360
        self.time_mode_row.setDirection(QBoxLayout.TopToBottom if narrow else QBoxLayout.LeftToRight)
        self.table.setColumnWidth(0, 68 if narrow else 84)
        self.table.setColumnWidth(1, 68 if narrow else 84)
        expr_width = max(180, width - (160 if narrow else 190))
        self.table.setColumnWidth(2, expr_width)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_responsive_layout()

    def _display_max(self):
        return 100.0 if self._time_mode == "percent" else max(self._total_time, 0.0)

    def _to_display_time(self, value):
        if self._time_mode != "percent":
            return float(value)
        if self._total_time <= 0:
            return 0.0
        return float(value) / self._total_time * 100.0

    def _to_absolute_time(self, value):
        if self._time_mode != "percent":
            return float(value)
        return float(value) * self._total_time / 100.0 if self._total_time > 0 else 0.0

    def _new_time_spin(self, value, read_only=False):
        spin = QDoubleSpinBox()
        spin.setDecimals(6)
        spin.setRange(0.0, self._display_max())
        spin.setSingleStep(1.0)
        spin.setValue(float(value))
        spin.setMinimumWidth(72)
        spin.setMaximumWidth(96)
        if read_only:
            spin.setReadOnly(True)
            spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        return spin

    def _new_expr_combo(self, value):
        combo = QComboBox()
        presets = [
            "0",
            "t",
            "10*t",
            "t**2",
            "t**3",
            "pi",
            "e",
            "sin(t)",
            "cos(t)",
            "tan(t)",
            "sin(2*pi*t)",
            "exp(t)",
            "log(t)",
            "sqrt(t)",
            "abs(t)",
            "min(t, 1)",
            "max(t, 1)",
            "sinh(t)",
            "cosh(t)",
            "tanh(t)",
        ]
        combo.addItems(presets)
        return _configure_expression_combo(combo, value)

    def _load_profile(self, profile):
        self._updating = True
        self.table.setRowCount(0)
        source_is_percent = str(getattr(self, "_source_mode", "absolute")).lower().startswith("percent")
        for entry in profile:
            raw_t0 = float(entry.get("t0", 0.0))
            raw_t1 = float(entry.get("t1", self._total_time))
            if self._time_mode == "percent" and source_is_percent:
                t0 = raw_t0
                t1 = raw_t1
            else:
                t0 = self._to_display_time(raw_t0)
                t1 = self._to_display_time(raw_t1)
            expr = entry.get("expr", self._default_expr)
            self._append_row(t0, t1, expr)
        self._normalize_rows()
        self._updating = False

    def _append_row(self, t0, t1, expr):
        row = self.table.rowCount()
        self.table.insertRow(row)
        start_spin = self._new_time_spin(t0, read_only=True)
        end_spin = self._new_time_spin(t1)
        end_spin.valueChanged.connect(lambda _v, r=row: self._on_end_changed(r))
        self.table.setCellWidget(row, 0, start_spin)
        self.table.setCellWidget(row, 1, end_spin)
        self.table.setCellWidget(row, 2, self._new_expr_combo(expr))

    def _on_end_changed(self, row):
        if self._updating:
            return
        self._normalize_rows()

    def _normalize_rows(self):
        self._updating = True
        row_count = self.table.rowCount()
        if row_count == 0:
            self._updating = False
            return
        for row in range(row_count):
            start_spin = self.table.cellWidget(row, 0)
            end_spin = self.table.cellWidget(row, 1)
            if row == 0:
                start_spin.setValue(0.0)
            else:
                prev_end = self.table.cellWidget(row - 1, 1).value()
                start_spin.setValue(prev_end + 1.0)
            end_spin.setReadOnly(False)
            end_spin.setButtonSymbols(QAbstractSpinBox.UpDownArrows)
            end_spin.setMaximum(self._display_max())
            if row == row_count - 1 and end_spin.value() <= 0.0:
                end_spin.setValue(self._display_max())
            if end_spin.value() > self._display_max():
                end_spin.setValue(self._display_max())
        self._updating = False

    def _add_row(self):
        row_count = self.table.rowCount()
        if row_count == 0:
            self._append_row(0.0, self._display_max(), self._default_expr)
            self._normalize_rows()
            return
        last_end = self.table.cellWidget(row_count - 1, 1).value()
        if last_end >= self._display_max():
            QMessageBox.information(
                self,
                "Add Row",
                "Reduce the previous end time to add another row.",
            )
            return
        start = last_end + 1.0
        end = min(self._display_max(), start + 1.0)
        self._append_row(start, end, self._default_expr)
        self._normalize_rows()

    def _remove_row(self):
        row = self.table.currentRow()
        if row < 0:
            return
        if self.table.rowCount() <= 1:
            QMessageBox.information(self, "Remove Row", "At least one row is required.")
            return
        self.table.removeRow(row)
        self._normalize_rows()

    def get_profile(self):
        profile = []
        row_count = self.table.rowCount()
        for row in range(row_count):
            start_spin = self.table.cellWidget(row, 0)
            end_spin = self.table.cellWidget(row, 1)
            expr_widget = self.table.cellWidget(row, 2)
            if self._time_mode == "percent":
                t0 = float(start_spin.value())
                t1 = float(end_spin.value())
            else:
                t0 = self._to_absolute_time(start_spin.value())
                t1 = self._to_absolute_time(end_spin.value())
            expr = expr_widget.currentText().strip() if isinstance(expr_widget, QComboBox) else self._default_expr
            profile.append({"t0": t0, "t1": t1, "expr": expr or self._default_expr})
        return profile

    def _on_time_mode_changed(self, _idx):
        return

    def _emit_apply(self):
        self.applied.emit(self.get_profile())

    def _emit_cancel(self):
        self.canceled.emit()


class TimeValueTableDialog(QDialog):
    def __init__(self, title, total_time, value_label="Velocity", points=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title or "Piecewise Values")
        self._total_time = max(0.0, float(total_time))
        self._value_label = str(value_label or "Value")
        self._points = list(points or [])

        layout = QVBoxLayout(self)
        _apply_layout_metrics(layout, margins=(10, 10, 10, 10), spacing=DOCK_ROW_SPACING)

        help_label = QLabel(
            f"Define piecewise {self._value_label.lower()} values using time-value rows."
        )
        help_label.setWordWrap(True)
        help_label.setObjectName("MinorStatusLabel")
        layout.addWidget(help_label)

        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Time (s)", self._value_label])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setMinimumHeight(220)
        layout.addWidget(self.table, 1)

        btn_row = QHBoxLayout()
        _apply_layout_metrics(btn_row, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        add_btn = QPushButton("Add Row")
        del_btn = QPushButton("Delete Row")
        add_btn.clicked.connect(self._add_row)
        del_btn.clicked.connect(self._delete_row)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(del_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        action_row = QHBoxLayout()
        _apply_layout_metrics(action_row, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        action_row.addStretch(1)
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        action_row.addWidget(ok_btn)
        action_row.addWidget(cancel_btn)
        layout.addLayout(action_row)

        if not self._points:
            self._points = [(0.0, 0.0)]
        self._load_points(self._points)

    def _new_time_spin(self, value):
        spin = QDoubleSpinBox()
        spin.setDecimals(6)
        spin.setRange(0.0, max(self._total_time, 0.0))
        spin.setValue(float(value))
        return spin

    def _new_value_spin(self, value):
        spin = QDoubleSpinBox()
        spin.setDecimals(6)
        spin.setRange(-1e12, 1e12)
        spin.setValue(float(value))
        return spin

    def _load_points(self, points):
        self.table.setRowCount(0)
        for time_value, scalar_value in points:
            self._append_row(time_value, scalar_value)

    def _append_row(self, time_value, scalar_value):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setCellWidget(row, 0, self._new_time_spin(time_value))
        self.table.setCellWidget(row, 1, self._new_value_spin(scalar_value))

    def _add_row(self):
        last_time = 0.0
        last_value = 0.0
        if self.table.rowCount() > 0:
            last_time = float(self.table.cellWidget(self.table.rowCount() - 1, 0).value())
            last_value = float(self.table.cellWidget(self.table.rowCount() - 1, 1).value())
        self._append_row(min(self._total_time, last_time), last_value)

    def _delete_row(self):
        row = self.table.currentRow()
        if row < 0:
            return
        if self.table.rowCount() <= 1:
            QMessageBox.information(self, "Delete Row", "At least one row is required.")
            return
        self.table.removeRow(row)

    def get_points(self):
        points = []
        for row in range(self.table.rowCount()):
            time_spin = self.table.cellWidget(row, 0)
            value_spin = self.table.cellWidget(row, 1)
            points.append((float(time_spin.value()), float(value_spin.value())))
        points.sort(key=lambda pair: pair[0])
        if not points:
            points = [(0.0, 0.0)]
        if points[0][0] != 0.0:
            points[0] = (0.0, points[0][1])
        return points


class BCLoadsPanel(QWidget):
    def __init__(self, sketch_view, parent=None, panel_mode="combined", project_state=None):
        super().__init__(parent)
        self.sketch_view = sketch_view
        self.project_state = _resolve_project_state(sketch_view, project_state)
        self.bc_controller = None
        mode = str(panel_mode or "combined").strip().lower()
        if mode not in ("combined", "bc", "load"):
            mode = "combined"
        self.panel_mode = mode
        self._is_bc_mode = mode == "bc"
        self._is_load_mode = mode == "load"
        self._viewport = None
        self._index_role = Qt.UserRole + 1
        self._entry_kind_role = Qt.UserRole + 2
        self._selection_enabled = False
        self._workspace_is_3d = False
        self._selection_timer = QTimer(self)
        self._selection_timer.setInterval(200)
        self._selection_timer.timeout.connect(self._update_selection_label)
        self._profile_pending = None
        self._attr_list_syncing = False
        self._paint_types_master = self._build_paint_types()
        self._paint_types = dict(self._paint_types_master)
        self._responsive_rows = []
        self._velocity_piecewise_profiles = {"x": [], "y": [], "z": []}

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_content = QWidget()
        scroll_content.setMinimumWidth(0)
        # Prevent the inner widget from ever exceeding the viewport width —
        # otherwise long combobox items / wide rows can shift the whole
        # content left even when the scrollbar is hidden.
        scroll_content.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        scroll_area.setWidget(scroll_content)
        # Lock horizontal scrolling to position 0 no matter what — content
        # that's wider than the viewport gets clipped on the right instead of
        # being scrolled away on the left.
        hbar = scroll_area.horizontalScrollBar()
        hbar.setRange(0, 0)
        hbar.valueChanged.connect(lambda _v: hbar.setValue(0))
        outer_layout.addWidget(scroll_area, 1)
        layout = QVBoxLayout(scroll_content)
        _apply_layout_metrics(layout)
        self._bc_outer_layout = outer_layout
        self._bc_scroll_area = scroll_area

        bc_title = "Boundary Conditions" if mode == "combined" else ("BCs" if not self._is_load_mode else "BCs (read-only)")
        self._bc_label = QLabel(bc_title)
        self._bc_label.setObjectName("SectionTitleLabel")
        self._bc_label.setToolTip("Boundary Conditions")
        layout.addWidget(self._bc_label)
        self.bc_list = QTreeWidget()
        self.bc_list.setHeaderLabels(["Name", "Type"])
        self.bc_list.setAlternatingRowColors(True)
        self.bc_list.setUniformRowHeights(True)
        self.bc_list.setRootIsDecorated(False)
        # Name is short ("BC 1", "Force FX"), Type is longer ("Fixed Support").
        # Make Name fit to content with a small floor, and let Type take the
        # remaining width so labels like "Fixed Support" aren't truncated.
        self.bc_list.header().setStretchLastSection(True)
        self.bc_list.header().setSectionResizeMode(0, QHeaderView.Interactive)
        self.bc_list.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self.bc_list.header().resizeSection(0, 70)
        self.bc_list.itemSelectionChanged.connect(lambda: self._on_attr_list_selection_changed("bc"))
        self.bc_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.bc_list.customContextMenuRequested.connect(self._show_bc_context_menu)
        _configure_dock_tree(self.bc_list)
        self.bc_list.setMinimumHeight(92)
        self.bc_scroll = self.bc_list
        layout.addWidget(self.bc_list, 1)
        self._bc_empty_label = _make_empty_state_label(
            "No boundary conditions yet — click Add Boundary Condition below to create one.",
            self,
        )
        layout.addWidget(self._bc_empty_label)
        _bind_tree_empty_state(self.bc_list, self._bc_empty_label)

        self.add_bc_btn = QPushButton("Add Boundary Condition")
        self.add_bc_btn.setProperty("primary", True)
        self.add_bc_btn.setMinimumHeight(32)
        self.add_bc_btn.setIcon(_style_icon(self, "SP_FileDialogNewFolder", "new", ("list-add",)))
        self._add_bc_label_full = "Add Boundary Condition"
        self.add_bc_btn.clicked.connect(self._open_add_boundary_condition_dialog)
        layout.addWidget(self.add_bc_btn)

        load_title = "Loads" if not self._is_bc_mode else "Loads (read-only)"
        self._load_label = QLabel(load_title)
        self._load_label.setObjectName("SectionTitleLabel")
        layout.addWidget(self._load_label)
        self.load_list = QTreeWidget()
        self.load_list.setHeaderLabels(["Type", "Value", "Target"])
        self.load_list.setAlternatingRowColors(True)
        self.load_list.setUniformRowHeights(True)
        self.load_list.setRootIsDecorated(False)
        self.load_list.header().setStretchLastSection(False)
        self.load_list.header().setSectionResizeMode(0, QHeaderView.Interactive)
        self.load_list.header().setSectionResizeMode(1, QHeaderView.Interactive)
        self.load_list.header().setSectionResizeMode(2, QHeaderView.Stretch)
        self.load_list.header().resizeSection(0, 96)
        self.load_list.header().resizeSection(1, 120)
        self.load_list.itemSelectionChanged.connect(lambda: self._on_attr_list_selection_changed("load"))
        _configure_dock_tree(self.load_list)
        self.load_list.setMinimumHeight(92)
        self.load_scroll = self.load_list
        layout.addWidget(self.load_list, 1)
        self._load_empty_label = _make_empty_state_label(
            "No loads yet — configure type/direction below and click Apply Load.",
            self,
        )
        layout.addWidget(self._load_empty_label)
        _bind_tree_empty_state(self.load_list, self._load_empty_label)

        self.selection_container = QGroupBox("Apply To")
        self.selection_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        selection_layout = QVBoxLayout(self.selection_container)
        _apply_layout_metrics(selection_layout, margins=(DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN), spacing=DOCK_ROW_SPACING)
        self.selection_form = QFormLayout()
        self.selection_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.selection_form.setRowWrapPolicy(QFormLayout.DontWrapRows)
        _apply_layout_metrics(self.selection_form, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        self.selection_target_label = QLabel("Apply to:")
        self.selection_target_combo = QComboBox()
        if self._is_bc_mode:
            self.selection_target_combo.addItems(["Edge"])
        else:
            self.selection_target_combo.addItems(["Auto", "Face", "Edge", "Point"])
        self.selection_target_combo.currentIndexChanged.connect(self._on_selection_target_changed)
        self.selection_form.addRow(self.selection_target_label, self.selection_target_combo)
        selection_layout.addLayout(self.selection_form)
        self.selection_status_label = QLabel("Selected: none")
        self.selection_status_label.setObjectName("MinorStatusLabel")
        selection_layout.addWidget(self.selection_status_label)
        layout.addWidget(self.selection_container)

        settings_title = "Type"
        self.settings_group = QGroupBox(settings_title)
        self.settings_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.settings_layout = QFormLayout(self.settings_group)
        self.settings_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        _apply_layout_metrics(self.settings_layout, margins=(DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN), spacing=DOCK_ROW_SPACING)
        if self._is_load_mode:
            self.load_type_combo = QComboBox()
            self.load_type_combo.addItem(get_icon("force", size=16), "Force / Load", "force")
            self.settings_layout.addRow("Load:", self.load_type_combo)
            self.load_entry_mode_combo = QComboBox()
            self.load_entry_mode_combo.addItem(get_icon("force", size=16), "Components", "components")
            self.load_entry_mode_combo.addItem(get_icon("magnitude", size=16), "Magnitude", "magnitude")
            self.load_entry_mode_combo.currentIndexChanged.connect(self._update_constraint_form_ui)
            self.settings_layout.addRow("Mode:", self.load_entry_mode_combo)
            self.load_direction_label = QLabel("Direction:")
            self.load_direction_combo = QComboBox()
            self.load_direction_combo.addItem(get_icon("axis_x", size=16), "X", "x")
            self.load_direction_combo.addItem(get_icon("axis_y", size=16), "Y", "y")
            self.load_direction_combo.addItem(get_icon("axis_z", size=16), "Z", "z")
            self.load_direction_combo.currentIndexChanged.connect(self._update_constraint_form_ui)
            self.settings_layout.addRow("Direction:", self.load_direction_combo)
            self.load_magnitude_label = QLabel("Magnitude:")
            self.load_magnitude_spin = QDoubleSpinBox()
            self.load_magnitude_spin.setRange(-1e9, 1e9)
            self.load_magnitude_spin.setValue(0.0)
            self.load_magnitude_spin.valueChanged.connect(self._update_constraint_form_ui)
            self.settings_layout.addRow("Magnitude:", self.load_magnitude_spin)
        else:
            self.bc_type_combo = QComboBox()
            self.bc_type_combo.addItem(get_icon("fix", size=16), "Fixed Support", "fixed")
            self.bc_type_combo.addItem(get_icon("displacement", size=16), "Prescribed Displacement", "displacement")
            self.bc_type_combo.addItem(get_icon("velocity", size=16), "Prescribed Velocity", "velocity")
            if self.panel_mode == "combined":
                self.bc_type_combo.addItem(get_icon("force", size=16), "Force / Load", "force")
            self.bc_type_combo.currentIndexChanged.connect(self._update_constraint_form_ui)
            self.settings_layout.addRow("BC Type:", self.bc_type_combo)
            self.velocity_mode_label = QLabel("Input:")
            self.velocity_mode_combo = QComboBox()
            self.velocity_mode_combo.addItem("Expression", "piecewise")
            self.velocity_mode_combo.currentIndexChanged.connect(self._update_constraint_form_ui)
            self.settings_layout.addRow("Input:", self.velocity_mode_combo)
        layout.addWidget(self.settings_group)

        self.dof_group = QGroupBox("Direction")
        self.dof_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.dof_layout = QHBoxLayout(self.dof_group)
        _apply_layout_metrics(self.dof_layout, margins=(DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN), spacing=DOCK_ROW_SPACING)
        self.axis_checks = {}
        self._dof_widgets = []
        axis_colors = {"x": "#d64545", "y": "#2f9d5d", "z": "#2f6fd6"}
        for axis in ("x", "y", "z"):
            button = QPushButton(axis.upper())
            button.setCheckable(True)
            button.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
            button.setMinimumHeight(30)
            button.setMinimumWidth(DOCK_ICON_BTN_MIN)
            button.setToolTip(f"{axis.upper()} direction")
            button.setStyleSheet(
                f"""
                QPushButton {{
                    color: {axis_colors[axis]};
                    font-weight: 700;
                    border: 1px solid {axis_colors[axis]};
                    border-radius: 15px;
                    padding: 4px 14px;
                    background: rgba(255, 255, 255, 0.92);
                }}
                QPushButton:checked {{
                    color: white;
                    background: {axis_colors[axis]};
                }}
                """
            )
            button.toggled.connect(self._update_constraint_form_ui)
            self.axis_checks[axis] = button
            self._dof_widgets.append(button)
            self.dof_layout.addWidget(button)
        layout.addWidget(self.dof_group)
        # Direction picker is hidden from the UI per user request; the X axis
        # is selected by default so BC application still has a target axis.
        # The button widgets remain in memory because other code (apply,
        # serialization) reads axis_checks[*].isChecked().
        # blockSignals while toggling because the X toggle is wired to
        # _update_constraint_form_ui which references widgets (values_group,
        # etc.) that don't exist yet at this point in __init__.
        self.axis_checks["x"].blockSignals(True)
        self.axis_checks["x"].setChecked(True)
        self.axis_checks["x"].blockSignals(False)
        self.dof_group.setVisible(False)

        self.values_group = QGroupBox("Condition")
        self.values_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.values_layout = QFormLayout(self.values_group)
        self.values_layout.setLabelAlignment(Qt.AlignLeft)
        self.values_layout.setFormAlignment(Qt.AlignTop)
        self.values_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        _apply_layout_metrics(self.values_layout, margins=(DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN), spacing=DOCK_ROW_SPACING)
        self.constraint_value_row = QWidget()
        self.constraint_value_row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        constraint_value_layout = QHBoxLayout(self.constraint_value_row)
        constraint_value_layout.setContentsMargins(0, 0, 0, 0)
        constraint_value_layout.setSpacing(DOCK_ROW_SPACING)
        self.constraint_symbol_label = QLabel("u =")
        self.constraint_symbol_label.setStyleSheet("font-weight: 700;")
        self.constraint_symbol_label.setMinimumWidth(32)
        self.constraint_scalar_spin = QDoubleSpinBox()
        self.constraint_scalar_spin.setRange(-1e9, 1e9)
        self.constraint_scalar_spin.setDecimals(6)
        self.constraint_scalar_spin.setValue(0.0)
        self.constraint_scalar_spin.setMinimumHeight(28)
        self.constraint_scalar_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.constraint_scalar_spin.valueChanged.connect(self._update_constraint_form_ui)
        self.constraint_static_label = QLabel("0")
        self.constraint_static_label.setStyleSheet("font-weight: 700;")
        self.constraint_static_label.setVisible(False)
        self.constraint_unit_label = QLabel("mm")
        self.constraint_unit_label.setObjectName("MinorStatusLabel")
        constraint_value_layout.addWidget(self.constraint_symbol_label)
        constraint_value_layout.addWidget(self.constraint_scalar_spin, 1)
        constraint_value_layout.addWidget(self.constraint_static_label)
        constraint_value_layout.addWidget(self.constraint_unit_label)
        constraint_value_layout.addStretch(1)
        self.values_layout.addRow("Equation", self.constraint_value_row)
        self.constraint_hint_label = QLabel("")
        self.constraint_hint_label.setObjectName("MinorStatusLabel")
        self.constraint_hint_label.setWordWrap(True)
        self.values_layout.addRow("", self.constraint_hint_label)
        self.axis_value_rows = {}
        self.axis_value_labels = {}
        self.axis_value_spins = {}
        self.axis_locked_labels = {}
        self.axis_profile_buttons = {}
        self.axis_profile_labels = {}
        for axis in ("x", "y", "z"):
            axis_label = QLabel(f"V{axis}:")
            axis_label.setStyleSheet(f"color: {axis_colors[axis]}; font-weight: 600;")
            row = QWidget()
            row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(DOCK_ROW_SPACING)
            spin = QDoubleSpinBox()
            spin.setRange(-1e9, 1e9)
            spin.setDecimals(6)
            spin.setValue(0.0)
            spin.setMinimumHeight(26)
            spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            spin.valueChanged.connect(self._update_constraint_form_ui)
            locked = QLabel("Locked")
            locked.setObjectName("MinorStatusLabel")
            locked.setVisible(False)
            profile_btn = QPushButton("Edit Table")
            profile_btn.setVisible(False)
            profile_btn.clicked.connect(lambda _checked=False, a=axis: self._edit_velocity_piecewise_axis(a))
            profile_summary = QLabel("1 row")
            profile_summary.setObjectName("MinorStatusLabel")
            profile_summary.setVisible(False)
            row_layout.addWidget(spin, 1)
            row_layout.addWidget(profile_btn)
            row_layout.addWidget(profile_summary)
            row_layout.addWidget(locked)
            self.values_layout.addRow(axis_label, row)
            self.axis_value_rows[axis] = row
            self.axis_value_labels[axis] = axis_label
            self.axis_value_spins[axis] = spin
            self.axis_locked_labels[axis] = locked
            self.axis_profile_buttons[axis] = profile_btn
            self.axis_profile_labels[axis] = profile_summary
        self.velocity_expression_row = QWidget()
        self.velocity_expression_row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        velocity_expression_layout = QHBoxLayout(self.velocity_expression_row)
        velocity_expression_layout.setContentsMargins(0, 0, 0, 0)
        velocity_expression_layout.setSpacing(DOCK_ROW_SPACING)
        self.velocity_expression_button = QPushButton("Edit Velocity Expression")
        self.velocity_expression_button.clicked.connect(self._edit_selected_velocity_expression)
        self.velocity_expression_hint = QLabel("Choose X/Y in DOF Selection, then edit the velocity expression.")
        self.velocity_expression_hint.setObjectName("MinorStatusLabel")
        self.velocity_expression_hint.setWordWrap(True)
        velocity_expression_layout.addWidget(self.velocity_expression_button)
        velocity_expression_layout.addWidget(self.velocity_expression_hint, 1)
        self.values_layout.addRow("Expression", self.velocity_expression_row)
        self.velocity_expression_row.setVisible(False)
        layout.addWidget(self.values_group)

        self.apply_group = QGroupBox("Apply")
        self.apply_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        apply_layout = QHBoxLayout(self.apply_group)
        _apply_layout_metrics(apply_layout, margins=(DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN), spacing=DOCK_ROW_SPACING)
        if self._is_bc_mode:
            apply_label = "Apply BC"
        elif self._is_load_mode:
            apply_label = "Apply Load"
        else:
            apply_label = "Apply BC/Load"
        self._apply_label_full = apply_label
        self.apply_bc_btn = QPushButton(apply_label)
        self.apply_bc_btn.setProperty("primary", True)
        self.apply_bc_btn.setMinimumHeight(32)
        self.apply_bc_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.apply_bc_btn.clicked.connect(self._apply_bc_from_selection)
        self.apply_bc_btn.setIcon(_style_icon(self, "SP_DialogApplyButton", "confirm", ("dialog-ok-apply",)))
        self.apply_bc_btn.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.apply_bc_btn.setToolTip("Apply BC")
        self.clear_selection_btn = QPushButton("Clear selection")
        self.clear_selection_btn.setMinimumHeight(32)
        self.clear_selection_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.clear_selection_btn.clicked.connect(self._clear_selection)
        self.clear_selection_btn.setIcon(_style_icon(self, "SP_DialogResetButton", "erase", ("edit-clear",)))
        self.clear_selection_btn.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.clear_selection_btn.setToolTip("Clear selection")
        self._selection_btn_widgets = [self.apply_bc_btn, self.clear_selection_btn]
        apply_layout.addWidget(self.apply_bc_btn, 1)
        apply_layout.addWidget(self.clear_selection_btn, 1)
        if self._is_bc_mode:
            apply_part_label = "Apply BC to Selected Part"
        elif self._is_load_mode:
            apply_part_label = "Apply Load to Selected Part"
        else:
            apply_part_label = "Apply to Selected Part"
        self._apply_part_label_full = apply_part_label
        self.apply_part_btn = QPushButton(apply_part_label)
        self.apply_part_btn.setProperty("primary", True)
        self.apply_part_btn.setMinimumHeight(32)
        self.apply_part_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.apply_part_btn.clicked.connect(self._apply_to_selected_part)
        self.apply_part_btn.setIcon(_style_icon(self, "SP_DialogApplyButton", "confirm", ("dialog-ok-apply",)))
        self.apply_part_btn.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.apply_part_btn.setToolTip("Apply to selected part")
        apply_layout.addWidget(self.apply_part_btn, 1)
        self._sticky_footer = QFrame(self)
        self._sticky_footer.setObjectName("BCStickyFooter")
        self._sticky_footer.setFrameShape(QFrame.StyledPanel)
        sticky_layout = QVBoxLayout(self._sticky_footer)
        _apply_layout_metrics(sticky_layout, margins=(DOCK_MARGIN, 2, DOCK_MARGIN, 2), spacing=0)
        sticky_layout.addWidget(self.apply_group)
        outer_layout.addWidget(self._sticky_footer, 0)

        self.advanced_group = QGroupBox("Advanced")
        self.advanced_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.advanced_group.setCheckable(True)
        self.advanced_group.setChecked(False)
        advanced_layout = QVBoxLayout(self.advanced_group)
        _apply_layout_metrics(
            advanced_layout,
            margins=(DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN),
            spacing=DOCK_SECTION_SPACING,
        )
        self.advanced_body = QWidget(self.advanced_group)
        self.advanced_body_layout = QVBoxLayout(self.advanced_body)
        _apply_layout_metrics(
            self.advanced_body_layout,
            margins=(0, 0, 0, 0),
            spacing=DOCK_SECTION_SPACING,
        )
        advanced_layout.addWidget(self.advanced_body)
        self.advanced_group.toggled.connect(self._set_advanced_section_expanded)
        layout.addWidget(self.advanced_group)

        self.profile_editor = ScalarTimeProfileEditor(self)
        self.profile_editor.setVisible(False)
        self.profile_editor.applied.connect(self._apply_profile_editor)
        self.profile_editor.canceled.connect(self._cancel_profile_editor)
        self._profile_editor_restore_hidden = False
        self.profile_section = QGroupBox("Velocity Time Profile")
        self.profile_section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.profile_section.setCheckable(True)
        self.profile_section.setChecked(False)
        profile_section_layout = QVBoxLayout(self.profile_section)
        _apply_layout_metrics(
            profile_section_layout,
            margins=(DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN),
            spacing=DOCK_ROW_SPACING,
        )
        profile_section_layout.addWidget(self.profile_editor)
        self.profile_section.toggled.connect(self._set_profile_section_expanded)
        self.advanced_body_layout.addWidget(self.profile_section)

        self.btn_layout = QGridLayout()
        _apply_layout_metrics(self.btn_layout, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        self.refresh_btn = QPushButton("Refresh List")
        self.refresh_btn.setMinimumHeight(30)
        self.refresh_btn.clicked.connect(self.refresh_lists)
        self.refresh_btn.setIcon(_style_icon(self, "SP_BrowserReload", "redo", ("view-refresh",)))
        self.refresh_btn.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.refresh_btn.setToolTip("Refresh list")
        self.edit_selected_btn = QPushButton("Edit Selected")
        self.edit_selected_btn.setMinimumHeight(30)
        self.edit_selected_btn.clicked.connect(self.edit_selected)
        self.edit_selected_btn.setIcon(_style_icon(self, "SP_FileDialogDetailedView", "edit", ("document-edit",)))
        self.edit_selected_btn.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.edit_selected_btn.setToolTip("Edit selected")
        self.time_profile_btn = QPushButton("Time Profile")
        self.time_profile_btn.setMinimumHeight(30)
        self.time_profile_btn.clicked.connect(self.edit_time_profile)
        self.time_profile_btn.setIcon(_style_icon(self, "SP_FileDialogListView", "frame", ("view-calendar",)))
        self.time_profile_btn.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.time_profile_btn.setToolTip("Edit time profile")
        self.change_type_btn = QPushButton("Change Type")
        self.change_type_btn.setMinimumHeight(30)
        self.change_type_btn.clicked.connect(self.change_selected_type)
        self.change_type_btn.setIcon(_style_icon(self, "SP_BrowserReload", "parametric", ("view-refresh",)))
        self.change_type_btn.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.change_type_btn.setToolTip("Change type")
        self.move_selected_btn = QPushButton("Move Selected")
        self.move_selected_btn.setMinimumHeight(30)
        self.move_selected_btn.clicked.connect(self.move_selected)
        self.move_selected_btn.setIcon(_style_icon(self, "SP_ArrowRight", "gizmo_move", ("transform-move",)))
        self.move_selected_btn.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.move_selected_btn.setToolTip("Move selected")
        self.delete_selected_btn = QPushButton("Delete Selected")
        self.delete_selected_btn.setMinimumHeight(30)
        self.delete_selected_btn.clicked.connect(self.clear_selected)
        self.delete_selected_btn.setIcon(_style_icon(self, "SP_TrashIcon", "delete", ("edit-delete",)))
        self.delete_selected_btn.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.delete_selected_btn.setToolTip("Delete selected")
        self._edit_label_full = "Edit Selected"
        self._delete_label_full = "Delete Selected"
        self._change_type_label_full = "Change Type"
        self._move_label_full = "Move Selected"
        self._advanced_btn_widgets = [self.refresh_btn, self.edit_selected_btn, self.time_profile_btn]
        self._advanced_row_1_buttons = [self.refresh_btn, self.edit_selected_btn, self.time_profile_btn]
        self.advanced_body_layout.addLayout(self.btn_layout)

        self.btn_layout2 = QGridLayout()
        _apply_layout_metrics(self.btn_layout2, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        self._advanced_btn_widgets2 = [self.change_type_btn, self.move_selected_btn, self.delete_selected_btn]
        self._advanced_row_2_buttons = [self.change_type_btn, self.move_selected_btn, self.delete_selected_btn]
        self.advanced_body_layout.addLayout(self.btn_layout2)

        self.split_ops_row = QWidget()
        self.split_ops_layout = QGridLayout(self.split_ops_row)
        _apply_layout_metrics(self.split_ops_layout, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        self.edit_bc_btn = QPushButton("Edit BC")
        self.edit_bc_btn.setMinimumHeight(30)
        self.edit_bc_btn.setIcon(_style_icon(self, "SP_FileDialogDetailedView", "edit", ("document-edit",)))
        self.edit_bc_btn.clicked.connect(self._edit_selected_bc_only)
        self.delete_bc_btn = QPushButton("Delete BC")
        self.delete_bc_btn.setMinimumHeight(30)
        self.delete_bc_btn.setIcon(_style_icon(self, "SP_TrashIcon", "delete", ("edit-delete",)))
        self.delete_bc_btn.clicked.connect(self._delete_selected_bc_only)
        self.edit_load_btn = QPushButton("Edit Load")
        self.edit_load_btn.setMinimumHeight(30)
        self.edit_load_btn.setIcon(_style_icon(self, "SP_FileDialogDetailedView", "edit", ("document-edit",)))
        self.edit_load_btn.clicked.connect(self._edit_selected_load_only)
        self.delete_load_btn = QPushButton("Delete Load")
        self.delete_load_btn.setMinimumHeight(30)
        self.delete_load_btn.setIcon(_style_icon(self, "SP_TrashIcon", "delete", ("edit-delete",)))
        self.delete_load_btn.clicked.connect(self._delete_selected_load_only)
        self._split_ops_widgets = [self.edit_bc_btn, self.delete_bc_btn, self.edit_load_btn, self.delete_load_btn]
        self._split_ops_buttons = [self.edit_bc_btn, self.delete_bc_btn, self.edit_load_btn, self.delete_load_btn]
        self.advanced_body_layout.addWidget(self.split_ops_row)

        self.paint_group = QGroupBox("Brush")
        self.paint_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.paint_group.setCheckable(True)
        self.paint_group.setChecked(False)
        paint_group_layout = QVBoxLayout(self.paint_group)
        _apply_layout_metrics(
            paint_group_layout,
            margins=(DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN),
            spacing=DOCK_ROW_SPACING,
        )
        self.paint_body = QWidget(self.paint_group)
        paint_body_layout = QVBoxLayout(self.paint_body)
        _apply_layout_metrics(
            paint_body_layout,
            margins=(0, 0, 0, 0),
            spacing=DOCK_ROW_SPACING,
        )
        paint_group_layout.addWidget(self.paint_body)
        self.paint_group.toggled.connect(self._set_paint_section_expanded)
        self.paint_form = QFormLayout()
        _apply_layout_metrics(self.paint_form, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        self.paint_form.setLabelAlignment(Qt.AlignLeft)
        self.paint_form.setFormAlignment(Qt.AlignTop)
        self.paint_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.paint_type_combo = QComboBox()
        self.paint_type_combo.addItems(self._paint_types.keys())
        self.paint_type_combo.currentIndexChanged.connect(self._update_paint_brush)
        self.force_axis_combo = QComboBox()
        self.force_axis_combo.addItem("Force X", "x")
        self.force_axis_combo.addItem("Force Y", "y")
        self.force_axis_combo.addItem("Force Z", "z")
        self.force_axis_combo.currentIndexChanged.connect(self._update_paint_brush)
        self.force_axis_combo.setVisible(False)
        self.paint_button = QPushButton("Paint: Off")
        self.paint_button.setCheckable(True)
        self.paint_button.toggled.connect(self.on_paint_toggled)
        self._paint_row_widgets = [self.paint_type_combo, self.force_axis_combo, self.paint_button]
        paint_toggle_row = QHBoxLayout()
        _apply_layout_metrics(paint_toggle_row, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        paint_toggle_row.addWidget(self.force_axis_combo)
        paint_toggle_row.addWidget(self.paint_button)
        paint_toggle_row.addStretch(1)
        self.paint_form.addRow("Brush type", self.paint_type_combo)
        self.paint_form.addRow("Brush mode", _layout_container(paint_toggle_row))
        paint_body_layout.addLayout(self.paint_form)

        self.paint_value_form = QFormLayout()
        _apply_layout_metrics(self.paint_value_form, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        self.paint_value_form.setLabelAlignment(Qt.AlignLeft)
        self.paint_value_form.setFormAlignment(Qt.AlignTop)
        self.paint_value_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.paint_fx_label = QLabel("Fx")
        self.paint_fx = QDoubleSpinBox()
        self.paint_fx.setRange(-1e9, 1e9)
        self.paint_fx.setValue(0.0)
        self.paint_fx.valueChanged.connect(self._update_paint_brush)
        self.paint_fy_label = QLabel("Fy")
        self.paint_fy = QDoubleSpinBox()
        self.paint_fy.setRange(-1e9, 1e9)
        self.paint_fy.setValue(0.0)
        self.paint_fy.valueChanged.connect(self._update_paint_brush)
        self.paint_val_label = QLabel("Val")
        self.paint_val = QDoubleSpinBox()
        self.paint_val.setRange(-1e9, 1e9)
        self.paint_val.setValue(0.0)
        self.paint_val.valueChanged.connect(self._update_paint_brush)
        self._paint_value_widgets = [
            self.paint_fx_label,
            self.paint_fx,
            self.paint_fy_label,
            self.paint_fy,
            self.paint_val_label,
            self.paint_val,
        ]
        self.paint_value_form.addRow("Fx", self.paint_fx)
        self.paint_value_form.addRow("Fy", self.paint_fy)
        self.paint_value_form.addRow("Value", self.paint_val)
        paint_body_layout.addLayout(self.paint_value_form)
        self.advanced_body_layout.addWidget(self.paint_group)
        self.paint_group.setVisible(False)

        self.refresh_lists()
        self._selection_timer.start()
        self._set_selection_enabled(False)
        self._apply_mode_visibility()
        self.axis_checks["x"].setChecked(True)
        self.axis_checks["y"].setChecked(False)
        self.axis_checks["z"].setChecked(False)
        if self._is_load_mode:
            self.axis_checks["x"].setChecked(True)
        self._refresh_constraint_axis_visibility(bool(getattr(self.sketch_view, "project_mode", "2d") == "3d"))
        self._update_constraint_form_ui()
        self._set_advanced_section_expanded(False)
        self._set_profile_section_expanded(False)
        self._set_paint_section_expanded(False)
        self._update_responsive_layout()
        _finalize_dock_panel(self)

    def _build_paint_types(self):
        if self._is_bc_mode:
            return {
                "Fix XY": "fix_xy",
                "Fix X": "fix_x",
                "Fix Y": "fix_y",
                "Fix Z": "fix_z",
                "Velocity X": "velocity_x",
                "Velocity Y": "velocity_y",
                "Velocity Z": "velocity_z",
            }
        if self._is_load_mode:
            return {
                "Force (Fx,Fy)": "force",
                "Force Z": "force_z",
                "Moment": "moment",
            }
        return {
            "Fix XY": "fix_xy",
            "Fix X": "fix_x",
            "Fix Y": "fix_y",
            "Fix Z": "fix_z",
            "Velocity X": "velocity_x",
            "Velocity Y": "velocity_y",
            "Velocity Z": "velocity_z",
            "Force (Fx,Fy)": "force",
            "Force Z": "force_z",
            "Moment": "moment",
        }

    def _apply_mode_visibility(self):
        if self.panel_mode == "combined":
            self._load_label.setVisible(False)
            self.load_scroll.setVisible(False)
            self.selection_container.setVisible(bool(getattr(self, "_workspace_is_3d", False)))
            self.settings_group.setVisible(True)
            self.dof_group.setVisible(False)
            self.values_group.setVisible(True)
            self.apply_group.setVisible(True)
            self.advanced_group.setVisible(True)
            self.split_ops_row.setVisible(False)
            self.paint_group.setVisible(False)
            self.add_bc_btn.setVisible(False)
            self._edit_label_full = "Edit"
            self._delete_label_full = "Delete"
            self._change_type_label_full = "Change Type"
            self._move_label_full = "Move"
            self._update_action_button_density()
            return
        if self._is_bc_mode:
            self._load_label.setVisible(False)
            self.load_scroll.setVisible(False)
            self.edit_load_btn.setVisible(False)
            self.delete_load_btn.setVisible(False)
            self.split_ops_row.setVisible(False)
            self.advanced_group.setVisible(False)
            self._edit_label_full = "Edit BC"
            self._delete_label_full = "Delete BC"
            self._change_type_label_full = "Change BC Type"
            self._move_label_full = "Move BC"
        elif self._is_load_mode:
            self._bc_label.setVisible(False)
            self.bc_scroll.setVisible(False)
            self.edit_bc_btn.setVisible(False)
            self.delete_bc_btn.setVisible(False)
            self.split_ops_row.setVisible(False)
            self._edit_label_full = "Edit Load"
            self._delete_label_full = "Delete Load"
            self._change_type_label_full = "Change Load Type"
            self._move_label_full = "Move Load"
        else:
            self._edit_label_full = "Edit Selected"
            self._delete_label_full = "Delete Selected"
            self._change_type_label_full = "Change Type"
            self._move_label_full = "Move Selected"
        self._update_action_button_density()

    def _set_button_caption(self, button, *, full, compact, icon_only):
        if button is None:
            return
        if icon_only:
            button.setText("")
        elif compact is not None:
            button.setText(compact)
        else:
            button.setText(full)

    def _set_advanced_section_expanded(self, expanded):
        expanded = bool(expanded)
        if hasattr(self, "advanced_group"):
            self.advanced_group.blockSignals(True)
            self.advanced_group.setChecked(expanded)
            self.advanced_group.blockSignals(False)
        if hasattr(self, "advanced_body"):
            self.advanced_body.setVisible(expanded)
        if expanded:
            self._reveal_panel_widget(getattr(self, "advanced_group", None), top_margin=12)

    def _set_paint_section_expanded(self, expanded):
        expanded = bool(expanded)
        if hasattr(self, "paint_group"):
            self.paint_group.blockSignals(True)
            self.paint_group.setChecked(expanded)
            self.paint_group.blockSignals(False)
        if hasattr(self, "paint_body"):
            self.paint_body.setVisible(expanded)

    def _update_action_button_density(self):
        width = int(self.width())
        compact = width < 430
        icon_only = width < 350
        self._set_button_caption(
            self.add_bc_btn,
            full=self._add_bc_label_full,
            compact="Add BC" if compact else None,
            icon_only=icon_only,
        )
        self._set_button_caption(
            self.apply_bc_btn,
            full=self._apply_label_full,
            compact="Apply",
            icon_only=icon_only,
        )
        self._set_button_caption(
            self.clear_selection_btn,
            full="Clear selection",
            compact="Clear" if compact else None,
            icon_only=icon_only,
        )
        self._set_button_caption(
            self.apply_part_btn,
            full=self._apply_part_label_full,
            compact="Apply Part",
            icon_only=icon_only,
        )
        self._set_button_caption(
            self.refresh_btn,
            full="Refresh List",
            compact="Refresh" if compact else None,
            icon_only=icon_only,
        )
        self._set_button_caption(
            self.time_profile_btn,
            full="Time Profile",
            compact="Profile" if compact else None,
            icon_only=icon_only,
        )
        self._set_button_caption(
            self.change_type_btn,
            full=self._change_type_label_full,
            compact="Type",
            icon_only=icon_only,
        )
        self._set_button_caption(
            self.move_selected_btn,
            full=self._move_label_full,
            compact="Move",
            icon_only=icon_only,
        )
        self._set_button_caption(
            self.edit_selected_btn,
            full=self._edit_label_full,
            compact="Edit",
            icon_only=icon_only,
        )
        self._set_button_caption(
            self.delete_selected_btn,
            full=self._delete_label_full,
            compact="Delete",
            icon_only=icon_only,
        )
        self._set_button_caption(
            self.paint_button,
            full="Paint: On" if self.paint_button.isChecked() else "Paint: Off",
            compact="Paint On" if self.paint_button.isChecked() else "Paint Off",
            icon_only=icon_only,
        )
        self._update_responsive_layout()

    def _reflow_grid_row(self, layout, widgets, stacked):
        if isinstance(layout, QHBoxLayout):
            return
        while layout.count():
            layout.takeAt(0)
        for index in range(len(widgets) + 1):
            layout.setColumnStretch(index, 0)
            layout.setRowStretch(index, 0)
        if stacked:
            for row, widget in enumerate(widgets):
                layout.addWidget(widget, row, 0)
            layout.setColumnStretch(0, 1)
            layout.setRowStretch(len(widgets), 1)
        else:
            for column, widget in enumerate(widgets):
                layout.addWidget(widget, 0, column)
            layout.setColumnStretch(len(widgets), 1)

    def _update_responsive_layout(self):
        width = int(self.contentsRect().width() or self.width() or 0)
        if width <= 0:
            return
        row_width = max(1, width - 24)
        _reflow_button_grid(self.btn_layout, getattr(self, "_advanced_row_1_buttons", []), row_width, min_button_width=120, max_columns=3)
        _reflow_button_grid(self.btn_layout2, getattr(self, "_advanced_row_2_buttons", []), row_width, min_button_width=120, max_columns=3)
        _reflow_button_grid(self.split_ops_layout, getattr(self, "_split_ops_buttons", []), row_width, min_button_width=120, max_columns=2)
        row_wrap = QFormLayout.WrapLongRows if width < 360 else QFormLayout.DontWrapRows
        self.settings_layout.setRowWrapPolicy(row_wrap)
        self.values_layout.setRowWrapPolicy(row_wrap)
        self.selection_form.setRowWrapPolicy(row_wrap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_action_button_density()

    def set_bc_controller(self, controller):
        self.bc_controller = controller

    def set_viewport(self, viewport):
        previous = getattr(self, "_viewport", None)
        if previous is not None and hasattr(previous, "selectionChanged"):
            try:
                previous.selectionChanged.disconnect(self._update_selection_label)
            except Exception:
                pass
        self._viewport = viewport
        if self._viewport is not None and hasattr(self._viewport, "selectionChanged"):
            try:
                self._viewport.selectionChanged.connect(self._update_selection_label)
            except Exception:
                pass
        self._on_selection_target_changed()
        self._update_selection_label()
        self._set_selection_enabled(self._viewport is not None)

    def set_workspace_mode(self, is_3d):
        self._workspace_is_3d = bool(is_3d)
        self._set_selection_enabled(bool(is_3d) and self._viewport is not None)
        self._refresh_paint_type_options(bool(is_3d))
        self._refresh_constraint_axis_visibility(bool(is_3d))
        self._update_constraint_form_ui()
        if bool(is_3d) and self._viewport is not None:
            self._on_selection_target_changed()

    def _refresh_paint_type_options(self, is_3d):
        current = self.paint_type_combo.currentText()
        items = list(self._paint_types_master.keys())
        if not is_3d:
            items = [item for item in items if item not in ("Fix Z", "Velocity Z", "Force Z")]
        self._paint_types = {label: self._paint_types_master[label] for label in items}
        self.paint_type_combo.blockSignals(True)
        self.paint_type_combo.clear()
        if items:
            self.paint_type_combo.addItems(items)
        if current in items:
            self.paint_type_combo.setCurrentText(current)
        elif items:
            self.paint_type_combo.setCurrentIndex(0)
        self.paint_type_combo.blockSignals(False)
        self.force_axis_combo.blockSignals(True)
        self.force_axis_combo.clear()
        self.force_axis_combo.addItem("Force X", "x")
        self.force_axis_combo.addItem("Force Y", "y")
        if is_3d:
            self.force_axis_combo.addItem("Force Z", "z")
        self.force_axis_combo.blockSignals(False)
        self._update_paint_brush()

    def reset_stage_state(self, clear_lists=False):
        self._cancel_profile_editor()
        self._profile_pending = None
        self.bc_list.clearSelection()
        self.load_list.clearSelection()
        if clear_lists:
            self.bc_list.clear()
            self.load_list.clear()
        else:
            self.refresh_lists()
        if self.paint_button.isChecked():
            self.paint_button.setChecked(False)
        self._update_selection_label()
        self._update_constraint_form_ui()
        if hasattr(self.sketch_view, "set_panel_attr_focus"):
            self.sketch_view.set_panel_attr_focus(None, None)

    def _main_window(self):
        try:
            sketch_window = self.sketch_view.window()
        except Exception:
            sketch_window = None
        if (
            sketch_window is not None
            and hasattr(sketch_window, "execute_app_command")
            and hasattr(sketch_window, "project_state")
        ):
            return sketch_window
        widget = self.parentWidget()
        while widget is not None:
            if hasattr(widget, "execute_app_command") and hasattr(widget, "project_state"):
                return widget
            widget = widget.parentWidget()
        window = self.window()
        if window is not None and hasattr(window, "execute_app_command") and hasattr(window, "project_state"):
            return window
        return None

    def _announce_status(self, message):
        window = self._main_window()
        if window is not None and hasattr(window, "statusBar"):
            window.statusBar().showMessage(str(message), 4000)

    def _annotate_3d_attr_marker(self, marker_type, ids=None, axis=None):
        viewport = self._viewport
        if viewport is None or not hasattr(viewport, "add_bc_load_marker"):
            return
        try:
            viewport.add_bc_load_marker(marker_type, node_ids=ids, axis=axis)
        except Exception:
            pass

    def _selected_force_axis(self):
        axis = str(self.force_axis_combo.currentData() or "x").strip().lower()
        if axis not in ("x", "y", "z"):
            axis = "x"
        if axis == "z" and self.force_axis_combo.findData("z") < 0:
            axis = "x"
        return axis

    def _constraint_axes_visible(self):
        return ("x", "y", "z") if getattr(self.sketch_view, "project_mode", "2d") == "3d" else ("x", "y")

    def _combined_force_mode_active(self):
        if self.panel_mode != "combined" or not hasattr(self, "bc_type_combo"):
            return False
        return str(self.bc_type_combo.currentData() or "").strip().lower() == "force"

    def _uses_load_workflow(self):
        return bool(self._is_load_mode or self._combined_force_mode_active())

    def _sync_panel_module_context(self):
        target_module = "Load" if self._uses_load_workflow() else "Boundary"
        try:
            if getattr(self.sketch_view, "active_module", None) != target_module:
                self.sketch_view.set_module(target_module)
                if (
                    self.panel_mode == "combined"
                    and not bool(getattr(self, "_workspace_is_3d", False))
                    and target_module == "Load"
                ):
                    self._announce_status("Force mode active. Right-click an edge or point in the sketch view to apply force.")
        except Exception:
            pass

    def _dock_scroll_area(self):
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, QScrollArea):
                return parent
            parent = parent.parent()
        return None

    def _scroll_panel_to_top(self):
        window = self._main_window()
        properties_panel = getattr(window, "properties_panel", None) if window is not None else None
        if properties_panel is not None and hasattr(properties_panel, "scroll_current_panel_to_top"):
            properties_panel.scroll_current_panel_to_top()
            return
        scroll = self._dock_scroll_area()
        if scroll is None:
            return

        def _apply():
            bar = scroll.verticalScrollBar()
            if bar is not None:
                bar.setValue(bar.minimum())

        QTimer.singleShot(0, _apply)

    def _reveal_panel_widget(self, widget, top_margin=14):
        if widget is None:
            return
        window = self._main_window()
        properties_panel = getattr(window, "properties_panel", None) if window is not None else None
        if properties_panel is not None and hasattr(properties_panel, "reveal_panel_widget"):
            properties_panel.reveal_panel_widget(widget, top_margin=top_margin)
            return
        scroll = self._dock_scroll_area()
        if scroll is None:
            return

        def _apply():
            content = scroll.widget()
            if content is None:
                return
            try:
                scroll.ensureWidgetVisible(widget, 0, int(top_margin))
                pos = widget.mapTo(content, QPoint(0, 0))
                bar = scroll.verticalScrollBar()
                if bar is not None:
                    bar.setValue(max(bar.minimum(), int(pos.y()) - int(top_margin)))
            except Exception:
                pass

        QTimer.singleShot(0, _apply)

    def _constraint_axis_label(self, axis):
        prefix = "F" if self._uses_load_workflow() else "u"
        if not self._uses_load_workflow() and hasattr(self, "bc_type_combo"):
            kind = str(self.bc_type_combo.currentData() or "fixed").strip().lower()
            if kind == "velocity":
                prefix = "v"
        return f"{prefix}{str(axis).upper()}"

    def _current_constraint_kind(self):
        if self._is_load_mode:
            return "force"
        if hasattr(self, "bc_type_combo"):
            return str(self.bc_type_combo.currentData() or "fixed").strip().lower()
        return "fixed"

    def _current_length_unit(self):
        unit = str(getattr(self.sketch_view, "current_unit", "mm") or "mm").strip()
        return unit or "mm"

    def _constraint_symbol(self, kind=None):
        kind = str(kind or self._current_constraint_kind()).strip().lower()
        if kind == "force":
            return "F ="
        if kind == "velocity":
            return "v ="
        return "u ="

    def _constraint_unit_text(self, kind=None):
        kind = str(kind or self._current_constraint_kind()).strip().lower()
        unit = self._current_length_unit()
        if kind == "force":
            return "N"
        if kind == "velocity":
            return f"{unit}/s"
        if kind == "fixed":
            return ""
        return unit

    def _constraint_hint_text(self, kind=None):
        kind = str(kind or self._current_constraint_kind()).strip().lower()
        if kind == "force":
            return "Applies external load"
        if kind == "velocity":
            return "Prescribes rate of motion"
        if kind == "fixed":
            return "Constrains displacement to zero"
        return "Prescribes position"

    def _bc_kind_label(self, kind):
        kind = str(kind or "").strip().lower()
        if kind == "fixed":
            return "Fixed Support"
        if kind == "displacement":
            return "Prescribed Displacement"
        if kind == "velocity":
            return "Prescribed Velocity"
        if kind == "force":
            return "Force / Load"
        return "Boundary Condition"

    def _bc_kind_for_entry(self, entry):
        bc_type = str((entry or {}).get("type") or "").strip().lower()
        bc_mode = str((entry or {}).get("bc_mode") or "").strip().lower()
        if bc_type.startswith("fix_"):
            return "fixed"
        if bc_mode == "displacement":
            return "displacement"
        if bc_mode == "velocity" or bc_type.startswith("velocity_"):
            return "velocity"
        return ""

    def _bc_axes_for_entry(self, entry):
        bc_type = str((entry or {}).get("type") or "").strip().lower()
        if bc_type == "fix_xy":
            return {"x", "y"}
        if bc_type.startswith("fix_") or bc_type.startswith("velocity_"):
            axis = bc_type.rsplit("_", 1)[-1]
            if axis in {"x", "y", "z"}:
                return {axis}
        return set()

    def _normalize_bc_coords(self, coords):
        normalized = []
        for point in list(coords or []):
            if not isinstance(point, (tuple, list)) or len(point) < 2:
                continue
            try:
                normalized.append((round(float(point[0]), 9), round(float(point[1]), 9)))
            except Exception:
                continue
        return tuple(normalized)

    def _bc_targets_overlap(self, existing, candidate):
        try:
            existing_part = existing.get("part_id")
            candidate_part = candidate.get("part_id")
            if existing_part is not None or candidate_part is not None:
                return (
                    existing_part is not None
                    and candidate_part is not None
                    and int(existing_part) == int(candidate_part)
                )
        except Exception:
            pass
        existing_ids = {int(nid) for nid in list(existing.get("ids") or []) if nid is not None}
        candidate_ids = {int(nid) for nid in list(candidate.get("ids") or []) if nid is not None}
        if existing_ids and candidate_ids:
            return bool(existing_ids & candidate_ids)
        existing_coords = self._normalize_bc_coords(existing.get("coords") or [])
        candidate_coords = self._normalize_bc_coords(candidate.get("coords") or [])
        if existing_coords and candidate_coords:
            return existing_coords == candidate_coords
        return False

    def _bc_target_text(self, entry):
        if entry.get("part_id") is not None:
            return f"part {int(entry.get('part_id'))}"
        ids = [int(nid) for nid in list(entry.get("ids") or []) if nid is not None]
        if ids:
            return f"selection ({len(set(ids))} node(s))"
        coords = list(entry.get("coords") or [])
        if coords:
            return "selected geometry"
        return "the current target"

    def _find_bc_conflict(self, candidates, *, ignore_index=None):
        entries = list(getattr(self.project_state, "boundary_conditions", []) or [])
        for candidate in list(candidates or []):
            candidate_kind = self._bc_kind_for_entry(candidate)
            if candidate_kind not in {"fixed", "displacement", "velocity"}:
                continue
            candidate_axes = self._bc_axes_for_entry(candidate)
            if not candidate_axes:
                continue
            for index, existing in enumerate(entries):
                if ignore_index is not None and int(index) == int(ignore_index):
                    continue
                existing_kind = self._bc_kind_for_entry(existing)
                if existing_kind not in {"fixed", "displacement", "velocity"}:
                    continue
                if not self._bc_targets_overlap(existing, candidate):
                    continue
                overlap_axes = candidate_axes & self._bc_axes_for_entry(existing)
                if overlap_axes:
                    return {
                        "existing": existing,
                        "existing_index": int(index),
                        "existing_kind": existing_kind,
                        "candidate": candidate,
                        "candidate_kind": candidate_kind,
                        "axes": overlap_axes,
                    }
        return None

    def _validate_bc_candidates(self, candidates, *, ignore_index=None):
        conflict = self._find_bc_conflict(candidates, ignore_index=ignore_index)
        if conflict is None:
            return True
        axis_text = ", ".join(axis.upper() for axis in sorted(conflict["axes"]))
        existing_label = self._bc_kind_label(conflict["existing_kind"])
        candidate_label = self._bc_kind_label(conflict["candidate_kind"])
        target_text = self._bc_target_text(conflict["candidate"])
        if conflict["existing_kind"] == conflict["candidate_kind"]:
            message = (
                f"{existing_label} is already defined on {axis_text} for {target_text}.\n"
                "Edit the existing boundary condition instead of stacking another one on the same DOF."
            )
        else:
            message = (
                f"{candidate_label} cannot be applied on {axis_text} for {target_text} because "
                f"{existing_label} already exists on that DOF."
            )
        QMessageBox.warning(self, "Conflicting Boundary Condition", message)
        return False

    def _velocity_mode(self):
        if self._is_load_mode or not hasattr(self, "velocity_mode_combo"):
            return "piecewise"
        return str(self.velocity_mode_combo.currentData() or "piecewise").strip().lower()

    def _project_analysis_type(self):
        return str(getattr(self.project_state, "analysis_type", "static") or "static").strip().lower()

    def _project_dimension(self):
        return str(getattr(self.project_state, "dimension", "2D") or "2D").strip().upper()

    def _hide_apply_part_in_current_context(self):
        return self._project_analysis_type() == "static" and self._project_dimension() == "2D"

    def _velocity_points_to_profile(self, points):
        total_time = max(0.0, float(self._get_total_time()))
        clean_points = sorted(
            [(float(t), float(v)) for t, v in (points or [])],
            key=lambda pair: pair[0],
        )
        if not clean_points:
            clean_points = [(0.0, 0.0)]
        if clean_points[0][0] != 0.0:
            clean_points[0] = (0.0, clean_points[0][1])
        profile = []
        for index, (time_value, scalar_value) in enumerate(clean_points):
            next_time = total_time if index + 1 >= len(clean_points) else float(clean_points[index + 1][0])
            next_time = max(float(time_value), float(next_time))
            profile.append(
                {
                    "t0": float(time_value),
                    "t1": float(next_time),
                    "expr": f"{float(scalar_value):.12g}",
                }
            )
        return profile

    def _profile_to_velocity_points(self, entry):
        points = []
        stored = entry.get("time_value_pairs")
        if isinstance(stored, (list, tuple)):
            for pair in stored:
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    try:
                        points.append((float(pair[0]), float(pair[1])))
                    except Exception:
                        continue
        if points:
            return sorted(points, key=lambda pair: pair[0])
        profile = entry.get("time_profile") or []
        for seg in profile:
            try:
                time_value = float(seg.get("t0", 0.0))
            except Exception:
                time_value = 0.0
            expr = seg.get("expr", seg.get("v", entry.get("val", 0.0)))
            scalar_value = self._safe_eval_expr(expr, time_value)
            if scalar_value is None:
                try:
                    scalar_value = float(entry.get("val", 0.0))
                except Exception:
                    scalar_value = 0.0
            points.append((time_value, float(scalar_value)))
        if not points:
            points = [(0.0, float(entry.get("val", 0.0) or 0.0))]
        return sorted(points, key=lambda pair: pair[0])

    def _velocity_points_for_axis(self, axis):
        axis = str(axis or "x").strip().lower()
        points = list(self._velocity_piecewise_profiles.get(axis) or [])
        if points:
            return points
        return [(0.0, 0.0)]

    def _update_velocity_profile_summary(self, axis):
        label = self.axis_profile_labels.get(axis)
        if label is None:
            return
        points = self._velocity_points_for_axis(axis)
        label.setText(f"{len(points)} row" if len(points) == 1 else f"{len(points)} rows")

    def _edit_selected_velocity_expression(self):
        axes = [axis for axis in self._selected_constraint_axes() if axis in {"x", "y", "z"}]
        if not axes:
            visible_axes = list(self._constraint_axes_visible())
            axes = [visible_axes[0]] if visible_axes else ["x"]
        self._edit_velocity_piecewise_axis(axes[0])

    def _edit_velocity_piecewise_axis(self, axis, entry=None, target_index=None):
        axis = str(axis or "x").strip().lower()
        points = self._profile_to_velocity_points(entry) if entry is not None else self._velocity_points_for_axis(axis)
        unit = self.sketch_view.current_unit or "m"
        dialog = TimeValueTableDialog(
            f"{self._constraint_axis_label(axis)} Piecewise Velocity ({unit}/s)",
            self._get_total_time(),
            value_label=self._constraint_axis_label(axis),
            points=points,
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted:
            return False
        points = dialog.get_points()
        if entry is not None:
            profile = self._velocity_points_to_profile(points)

            def _update_velocity(edit_entry):
                edit_entry["time_value_pairs"] = list(points)
                edit_entry["time_profile"] = profile
                edit_entry["time_profile_mode"] = "absolute"
                edit_entry["val"] = float(points[0][1]) if points else 0.0

            if self.bc_controller is not None and target_index is not None:
                self.bc_controller.update_bc(target_index, _update_velocity, save_velocity=True)
            else:
                self.sketch_view.push_undo_state()
                _update_velocity(entry)
                self._sync_attr_fallback_state(emit_bcs=True, save_velocity=True)
            self.refresh_lists()
            self.sketch_view.redraw()
            return True
        self._velocity_piecewise_profiles[axis] = list(points)
        self._update_velocity_profile_summary(axis)
        return True

    def _selected_constraint_axes(self):
        visible_axes = set(self._constraint_axes_visible())
        axes = []
        for axis in ("x", "y", "z"):
            checkbox = self.axis_checks.get(axis)
            if checkbox is None or axis not in visible_axes:
                continue
            if checkbox.isChecked():
                axes.append(axis)
        return axes

    def _refresh_constraint_axis_visibility(self, is_3d):
        visible_axes = {"x", "y", "z"} if bool(is_3d) else {"x", "y"}
        for axis in ("x", "y", "z"):
            visible = axis in visible_axes
            checkbox = self.axis_checks.get(axis)
            if checkbox is not None:
                checkbox.setVisible(visible)
                if not visible:
                    checkbox.blockSignals(True)
                    checkbox.setChecked(False)
                    checkbox.blockSignals(False)
            row = self.axis_value_rows.get(axis)
            if row is not None:
                row.setVisible(visible)
        if not any(
            self.axis_checks[axis].isChecked()
            for axis in visible_axes
            if axis in self.axis_checks
        ):
            default_axis = "x" if "x" in visible_axes else None
            if default_axis is not None and default_axis in self.axis_checks:
                self.axis_checks[default_axis].blockSignals(True)
                self.axis_checks[default_axis].setChecked(True)
                self.axis_checks[default_axis].blockSignals(False)

    def _update_constraint_form_ui(self):
        visible_axes = set(self._constraint_axes_visible())
        if self._is_load_mode:
            self.constraint_value_row.setVisible(False)
            self.constraint_hint_label.setVisible(False)
            entry_mode = str(self.load_entry_mode_combo.currentData() or "components").strip().lower()
            components_mode = entry_mode != "magnitude"
            self.load_direction_label.setVisible(not components_mode)
            self.load_direction_combo.setVisible(not components_mode)
            self.load_magnitude_label.setVisible(not components_mode)
            self.load_magnitude_spin.setVisible(not components_mode)
            self.dof_group.setVisible(False)
            self.values_group.setVisible(components_mode)
            for axis in ("x", "y", "z"):
                label = self.axis_value_labels.get(axis)
                row = self.axis_value_rows.get(axis)
                spin = self.axis_value_spins.get(axis)
                locked = self.axis_locked_labels.get(axis)
                button = self.axis_checks.get(axis)
                if label is not None:
                    label.setText(axis.upper())
                if button is not None:
                    button.setToolTip(f"{axis.upper()} Force\nInclude the {axis.upper()} force component.")
                row_visible = components_mode and axis in visible_axes and self.axis_checks[axis].isChecked()
                if label is not None:
                    label.setVisible(row_visible)
                if row is not None:
                    row.setVisible(row_visible)
                if spin is not None:
                    spin.setVisible(row_visible)
                    spin.setEnabled(row_visible)
                if locked is not None:
                    locked.setVisible(False)
            self._apply_label_full = "Apply Load"
            self._apply_part_label_full = "Apply Load to Selected Part"
            self.apply_bc_btn.setToolTip("Apply force")
            self.apply_part_btn.setToolTip("Apply force to selected part")
            self._update_action_button_density()
            self._sync_panel_module_context()
            return

        bc_kind = self._current_constraint_kind()
        fixed_mode = bc_kind == "fixed"
        force_mode = bc_kind == "force"
        self.dof_group.setVisible(False)
        self.values_group.setVisible(True)
        self.constraint_value_row.setVisible(True)
        self.constraint_hint_label.setVisible(True)
        self.constraint_symbol_label.setText(self._constraint_symbol(bc_kind))
        self.constraint_unit_label.setText(self._constraint_unit_text(bc_kind))
        self.constraint_hint_label.setText(self._constraint_hint_text(bc_kind))
        self.constraint_scalar_spin.setVisible(not fixed_mode)
        self.constraint_scalar_spin.setEnabled(not fixed_mode)
        self.constraint_static_label.setVisible(fixed_mode)
        self.constraint_unit_label.setVisible(bool(self._constraint_unit_text(bc_kind)))
        if hasattr(self, "velocity_mode_label"):
            self.velocity_mode_label.setVisible(False)
        if hasattr(self, "velocity_mode_combo"):
            self.velocity_mode_combo.setVisible(False)
        if hasattr(self, "velocity_expression_row"):
            self.velocity_expression_row.setVisible(False)
        self.apply_part_btn.setVisible(not self._hide_apply_part_in_current_context())
        for axis in ("x", "y", "z"):
            label = self.axis_value_labels.get(axis)
            row = self.axis_value_rows.get(axis)
            spin = self.axis_value_spins.get(axis)
            locked = self.axis_locked_labels.get(axis)
            button = self.axis_checks.get(axis)
            profile_button = self.axis_profile_buttons.get(axis)
            profile_label = self.axis_profile_labels.get(axis)
            if button is not None:
                button.setVisible(axis in visible_axes)
                if force_mode:
                    tip = f"Apply load along {axis.upper()}."
                elif bc_kind == "velocity":
                    tip = f"Prescribe velocity along {axis.upper()}."
                elif fixed_mode:
                    tip = f"Constrain {axis.upper()} displacement to zero."
                else:
                    tip = f"Prescribe displacement along {axis.upper()}."
                button.setToolTip(tip)
            if label is not None:
                label.setVisible(False)
            if row is not None:
                row.setVisible(False)
            if spin is not None:
                spin.setVisible(False)
                spin.setEnabled(False)
            if profile_button is not None:
                profile_button.setVisible(False)
            if profile_label is not None:
                profile_label.setVisible(False)
            if locked is not None:
                locked.setText("")
                locked.setVisible(False)
        if force_mode:
            self._apply_label_full = "Apply Load"
            self._apply_part_label_full = "Apply Load to Selected Part"
            self.apply_bc_btn.setToolTip("Apply load")
            self.apply_part_btn.setToolTip("Apply load to selected part")
        else:
            self._apply_label_full = "Apply BC"
            self._apply_part_label_full = "Apply BC to Selected Part"
            self.apply_bc_btn.setToolTip("Apply boundary condition")
            self.apply_part_btn.setToolTip("Apply boundary condition to selected part")
        self._update_action_button_density()
        self._sync_panel_module_context()

    def _build_bc_records(self, *, ids=None, part_id=None, target_kind=""):
        bc_kind = self._current_constraint_kind()
        axes = self._selected_constraint_axes()
        if not axes:
            QMessageBox.information(self, "Apply BC", "Enable at least one DOF before applying a boundary condition.")
            return None
        records = []
        scalar_value = float(self.constraint_scalar_spin.value())
        for axis in axes:
            record = {"coords": []}
            if ids is not None:
                record["ids"] = list(ids)
                record["target"] = target_kind
            if part_id is not None:
                record["part_id"] = int(part_id)
                record["target"] = "part"
            if bc_kind == "fixed":
                record["type"] = f"fix_{axis}"
                record["display_type"] = f"Fixed Support {axis.upper()}"
            else:
                record["type"] = f"velocity_{axis}"
                record["val"] = scalar_value
                record["bc_mode"] = bc_kind
                if bc_kind == "velocity":
                    points = [(0.0, scalar_value)]
                    record["time_value_pairs"] = list(points)
                    record["time_profile"] = self._velocity_points_to_profile(points)
                    record["time_profile_mode"] = "absolute"
                    record["display_type"] = f"Prescribed Velocity {axis.upper()}"
                else:
                    record["display_type"] = f"Prescribed Displacement {axis.upper()}"
            records.append((record, axis))
        if not self._validate_bc_candidates([record for record, _axis in records]):
            return None
        return bc_kind, records

    def _build_load_record(self, *, ids=None, part_id=None, target_kind=""):
        entry_mode = "components"
        if self._is_load_mode and hasattr(self, "load_entry_mode_combo"):
            entry_mode = str(self.load_entry_mode_combo.currentData() or "components").strip().lower()
        components = {"x": 0.0, "y": 0.0, "z": 0.0}
        if entry_mode == "magnitude" and hasattr(self, "load_direction_combo") and hasattr(self, "load_magnitude_spin"):
            axis = str(self.load_direction_combo.currentData() or "x").strip().lower()
            if axis not in self._constraint_axes_visible():
                axis = "x"
            components[axis] = float(self.load_magnitude_spin.value())
            active_axes = [axis]
        elif self.panel_mode == "combined" and self._combined_force_mode_active():
            active_axes = self._selected_constraint_axes()
            if not active_axes:
                QMessageBox.information(self, "Apply Load", "Select at least one direction before applying a load.")
                return None
            scalar_value = float(self.constraint_scalar_spin.value())
            for axis in active_axes:
                components[axis] = scalar_value
        else:
            active_axes = self._selected_constraint_axes()
            if not active_axes:
                QMessageBox.information(self, "Apply Load", "Enable at least one load direction before applying a load.")
                return None
            for axis in active_axes:
                components[axis] = float(self.axis_value_spins[axis].value())
        load_type = "force"
        display_type = "Force / Load"
        if self._is_load_mode and hasattr(self, "load_type_combo"):
            load_type = str(self.load_type_combo.currentData() or "force").strip().lower()
            display_type = self.load_type_combo.currentText().strip() or "Force"
        record = {
            "type": load_type,
            "display_type": display_type,
            "coords": [],
            "fx": components["x"],
            "fy": components["y"],
            "fz": components["z"],
            "entry_mode": entry_mode,
        }
        if len(active_axes) == 1:
            record["axis"] = active_axes[0]
        if ids is not None:
            record["ids"] = list(ids)
            record["target"] = target_kind
        if part_id is not None:
            record["part_id"] = int(part_id)
            record["target"] = "part"
        return record, active_axes

    def _on_selection_target_changed(self):
        if self._viewport is None:
            return
        mode = self.selection_target_combo.currentText().strip().lower()
        self._viewport.set_selection_mode(mode)

    def _update_selection_label(self):
        if self._viewport is None:
            if self._selection_enabled:
                self.selection_status_label.setText("Selected: none")
            else:
                self.selection_status_label.setText("3D workspace only")
            return
        if hasattr(self._viewport, "get_selection_counts"):
            counts = self._viewport.get_selection_counts() or {}
            faces = int(counts.get("faces", 0))
            edges = int(counts.get("edges", 0))
            points = int(counts.get("points", 0))
        else:
            faces = len(self._viewport.get_selected_faces())
            edges = len(self._viewport.get_selected_edges())
            points = len(self._viewport.get_selected_nodes())
        mode = str(self.selection_target_combo.currentText() or "auto").strip().lower()
        labels = {
            "face": ("face", faces),
            "edge": ("edge", edges),
            "point": ("point", points),
        }
        if mode in labels:
            label, count = labels[mode]
            suffix = "" if int(count) == 1 else "s"
            self.selection_status_label.setText(f"Selected: {int(count)} {label}{suffix}")
            return
        parts = []
        for label, count in (("face", faces), ("edge", edges), ("point", points)):
            if int(count) <= 0:
                continue
            suffix = "" if int(count) == 1 else "s"
            parts.append(f"{int(count)} {label}{suffix}")
        self.selection_status_label.setText("Selected: " + (" · ".join(parts) if parts else "none"))

    def _collect_selection_node_ids(self, mode):
        if self._viewport is None:
            return []
        if hasattr(self._viewport, "get_selected_node_ids_for_mode"):
            try:
                ids = self._viewport.get_selected_node_ids_for_mode(mode)
                return sorted(int(nid) for nid in ids if nid is not None)
            except Exception:
                pass
        ids = set()
        if mode == "face":
            if hasattr(self._viewport, "get_selected_face_nodes"):
                ids = set(self._viewport.get_selected_face_nodes())
            else:
                ids = set(self._viewport.get_selected_faces())
        elif mode == "edge":
            edges = self._viewport.get_selected_edges()
            for edge in edges:
                if isinstance(edge, (tuple, list)) and len(edge) == 2:
                    ids.update(edge)
        else:
            ids = set(self._viewport.get_selected_nodes())
        return sorted(int(nid) for nid in ids if nid is not None)

    def _set_selection_enabled(self, enabled):
        self._selection_enabled = bool(enabled)
        show_selection_target = bool(getattr(self, "_workspace_is_3d", False))
        self.selection_container.setVisible(show_selection_target)
        self.selection_target_combo.setEnabled(self._selection_enabled)
        self.apply_bc_btn.setEnabled(self._selection_enabled)
        self.clear_selection_btn.setEnabled(self._selection_enabled)
        if self._selection_enabled and self._viewport is not None:
            self._on_selection_target_changed()

    def _clear_selection(self):
        if self._viewport is None:
            return
        self._viewport.clear_selection()
        if hasattr(self._viewport, "clear_node_selection"):
            self._viewport.clear_node_selection()
        if hasattr(self._viewport, "clear_edge_selection"):
            self._viewport.clear_edge_selection()
        self._update_selection_label()

    def _sync_attr_fallback_state(self, *, emit_bcs=False, emit_loads=False, save_velocity=False):
        try:
            self.sketch_view.bcs = copy.deepcopy(self.project_state.boundary_conditions)
            self.sketch_view.loads = copy.deepcopy(self.project_state.loads)
            if hasattr(self.sketch_view, "_sanitize_bc_load_entries"):
                self.sketch_view._sanitize_bc_load_entries()
        except Exception:
            pass
        if save_velocity:
            self.sketch_view.save_velocity_csv()
        if emit_bcs:
            self.sketch_view.bcsChanged.emit()
        if emit_loads:
            self.sketch_view.loadsChanged.emit()

    def _apply_bc_from_selection(self):
        if self._viewport is None:
            return
        mode = self.selection_target_combo.currentText().strip().lower()
        ids = self._collect_selection_node_ids(mode)
        if ids is None or len(ids) == 0:
            QMessageBox.information(self, "Apply Load" if self._uses_load_workflow() else "Apply BC", "No selection to apply.")
            return
        if self._uses_load_workflow():
            built = self._build_load_record(ids=ids, target_kind=mode)
            if built is None:
                return
            load_record, active_axes = built
            main = self._main_window()
            execute_command = getattr(main, "execute_app_command", None) if main is not None else None
            if self.bc_controller is None or not callable(execute_command):
                QMessageBox.warning(self, "Boundary Conditions", "Boundary-condition command bus is not available.")
                return
            execute_command(AddBoundaryConditionCommand(entries=[load_record], entry_kind="load"))
            self.refresh_lists()
            self.sketch_view.redraw()
            for axis in active_axes:
                self._annotate_3d_attr_marker("force", ids=ids, axis=axis)
            self._announce_status(f"Applied force load to {len(ids)} selected node(s).")
            return

        built = self._build_bc_records(ids=ids, target_kind=mode)
        if built is None:
            return
        bc_kind, records = built
        bc_entries = [bc_record for bc_record, _axis in records]
        main = self._main_window()
        execute_command = getattr(main, "execute_app_command", None) if main is not None else None
        if self.bc_controller is None or not callable(execute_command):
            QMessageBox.warning(self, "Boundary Conditions", "Boundary-condition command bus is not available.")
            return
        execute_command(
            AddBoundaryConditionCommand(
                entries=bc_entries,
                entry_kind="bc",
                save_velocity=True,
            )
        )
        self.refresh_lists()
        self.sketch_view.redraw()
        if bc_kind == "fixed":
            self._annotate_3d_attr_marker("fixed", ids=ids)
        else:
            for _bc_record, axis in records:
                self._annotate_3d_attr_marker("velocity", ids=ids, axis=axis)
        self._announce_status(f"Applied {self.bc_type_combo.currentText().strip()} BC to {len(ids)} selected node(s).")

    def _apply_to_selected_part(self):
        part = self.sketch_view.get_selected_part()
        if part is None:
            QMessageBox.information(
                self,
                "Apply to Part",
                "Select a part first in the viewport, then apply BC/load.",
            )
            return
        self._apply_bc_to_part(part.id)

    def _apply_bc_to_part(self, part_id):
        if self._uses_load_workflow():
            built = self._build_load_record(part_id=int(part_id), target_kind="part")
            if built is None:
                return
            load_record, active_axes = built
            main = self._main_window()
            execute_command = getattr(main, "execute_app_command", None) if main is not None else None
            if self.bc_controller is None or not callable(execute_command):
                QMessageBox.warning(self, "Boundary Conditions", "Boundary-condition command bus is not available.")
                return
            execute_command(AddBoundaryConditionCommand(entries=[load_record], entry_kind="load"))
            self.refresh_lists()
            self.sketch_view.redraw()
            for axis in active_axes:
                self._annotate_3d_attr_marker("force", ids=None, axis=axis)
            self._announce_status(f"Applied force load to part {int(part_id)}.")
            return

        built = self._build_bc_records(part_id=int(part_id), target_kind="part")
        if built is None:
            return
        bc_kind, records = built
        bc_entries = [bc_record for bc_record, _axis in records]
        main = self._main_window()
        execute_command = getattr(main, "execute_app_command", None) if main is not None else None
        if self.bc_controller is None or not callable(execute_command):
            QMessageBox.warning(self, "Boundary Conditions", "Boundary-condition command bus is not available.")
            return
        execute_command(
            AddBoundaryConditionCommand(
                entries=bc_entries,
                entry_kind="bc",
                save_velocity=True,
            )
        )
        self.refresh_lists()
        self.sketch_view.redraw()
        if bc_kind == "fixed":
            self._annotate_3d_attr_marker("fixed", ids=None)
        else:
            for _bc_record, axis in records:
                self._annotate_3d_attr_marker("velocity", ids=None, axis=axis)
        self._announce_status(f"Applied {self.bc_type_combo.currentText().strip()} BC to part {int(part_id)}.")

    def refresh_lists(self):
        selected_bc_index = None
        selected_load_index = None
        selected_bc_kind = None
        current_bc = self.bc_list.currentItem()
        current_load = self.load_list.currentItem()
        if current_bc is not None:
            selected_bc_index = current_bc.data(0, self._index_role)
            selected_bc_kind = current_bc.data(0, self._entry_kind_role)
        if current_load is not None:
            selected_load_index = current_load.data(0, self._index_role)
        self.bc_list.clear()
        if self.panel_mode == "combined":
            for idx, bc in enumerate(self.project_state.boundary_conditions):
                item = QTreeWidgetItem([self._bc_display_name(bc, idx), self._bc_type_label(bc)])
                item.setData(0, Qt.UserRole, bc)
                item.setData(0, self._index_role, idx)
                item.setData(0, self._entry_kind_role, "bc")
                tooltip = self._bc_tooltip(bc, idx)
                item.setToolTip(0, tooltip)
                item.setToolTip(1, tooltip)
                self.bc_list.addTopLevelItem(item)
                if selected_bc_kind == "bc" and selected_bc_index == idx:
                    self.bc_list.setCurrentItem(item)
                    item.setSelected(True)
            for idx, load in enumerate(self.project_state.loads):
                item = QTreeWidgetItem([self._load_display_name(load, idx), self._load_type_label(load)])
                item.setData(0, Qt.UserRole, load)
                item.setData(0, self._index_role, idx)
                item.setData(0, self._entry_kind_role, "load")
                tooltip = self._load_tooltip(load, idx)
                item.setToolTip(0, tooltip)
                item.setToolTip(1, tooltip)
                self.bc_list.addTopLevelItem(item)
                if selected_bc_kind == "load" and selected_bc_index == idx:
                    self.bc_list.setCurrentItem(item)
                    item.setSelected(True)
            self.load_list.clear()
            self._sync_attr_entry_preview_from_lists()
            return
        for idx, bc in enumerate(self.project_state.boundary_conditions):
            item = QTreeWidgetItem([self._bc_display_name(bc, idx), self._bc_type_label(bc)])
            item.setData(0, Qt.UserRole, bc)
            item.setData(0, self._index_role, idx)
            item.setData(0, self._entry_kind_role, "bc")
            item.setToolTip(0, self._bc_tooltip(bc, idx))
            item.setToolTip(1, self._bc_tooltip(bc, idx))
            self.bc_list.addTopLevelItem(item)
            if selected_bc_index == idx:
                self.bc_list.setCurrentItem(item)
                item.setSelected(True)

        self.load_list.clear()
        for idx, load in enumerate(self.project_state.loads):
            if load.get("part_id") is not None:
                coords_str = f"part {load.get('part_id')}"
            elif load.get("ids") is not None:
                coords_str = f"{load.get('target', 'sel')} {load.get('ids')}"
            else:
                coords_str = str(load["coords"])
            if load["type"] == "force":
                fx = load.get("fx", 0.0)
                fy = load.get("fy", 0.0)
                fz = load.get("fz")
                if fz is not None:
                    val_str = f"fx={fx}, fy={fy}, fz={fz}"
                else:
                    val_str = f"fx={fx}, fy={fy}"
            elif load["type"] == "moment":
                val_str = f"m={load['m']}"
            else:
                val_str = "N/A"
            item = QTreeWidgetItem([load.get("display_type", load["type"]), val_str, coords_str])
            item.setData(0, Qt.UserRole, load)
            item.setData(0, self._index_role, idx)
            self.load_list.addTopLevelItem(item)
            if selected_load_index == idx:
                self.load_list.setCurrentItem(item)
                item.setSelected(True)
        self._sync_attr_entry_preview_from_lists()

    def select_bc_index(self, bc_index):
        if not isinstance(bc_index, int):
            return
        self.bc_list.clearSelection()
        for row in range(self.bc_list.topLevelItemCount()):
            item = self.bc_list.topLevelItem(row)
            if item is None:
                continue
            if item.data(0, self._entry_kind_role) == "bc" and item.data(0, self._index_role) == bc_index:
                self.bc_list.setCurrentItem(item)
                item.setSelected(True)
                self.bc_list.scrollToItem(item)
                self._on_attr_list_selection_changed("bc")
                return

    def focus_velocity_input(self, axis="x"):
        axis = str(axis or "x").strip().lower()
        if axis not in self.axis_value_spins:
            axis = "x"
        self.bc_type_combo.setCurrentIndex(max(0, self.bc_type_combo.findData("velocity")))
        self._update_constraint_form_ui()
        spin = getattr(self, "constraint_scalar_spin", None)
        self._reveal_panel_widget(spin or self.values_group, top_margin=18)
        if spin is not None and spin.isVisible():
            spin.setFocus(Qt.TabFocusReason)
            spin.selectAll()

    def _set_only_attr_list_active(self, source):
        if self._attr_list_syncing:
            return
        self._attr_list_syncing = True
        try:
            if source == "bc":
                self.load_list.clearSelection()
            elif source == "load":
                self.bc_list.clearSelection()
        finally:
            self._attr_list_syncing = False

    def _sync_attr_entry_preview_from_lists(self):
        if self._attr_list_syncing:
            return
        window = self._main_window()
        inspector = getattr(window, "property_inspector", None) if window is not None else None
        bc_item = self.bc_list.currentItem()
        load_item = self.load_list.currentItem()
        if bc_item is not None and bc_item.isSelected():
            bc = bc_item.data(0, Qt.UserRole)
            kind = str(bc_item.data(0, self._entry_kind_role) or "bc")
            if hasattr(self.sketch_view, "set_panel_attr_focus"):
                self.sketch_view.set_panel_attr_focus(kind, bc)
            if inspector is not None:
                stage = ProjectStage.BCS if self.panel_mode == "combined" else (
                    ProjectStage.BCS if kind == "bc" else ProjectStage.LOADS
                )
                inspector.set_selection_payload(
                    {
                        "kind": kind,
                        "index": int(bc_item.data(0, self._index_role) or 0),
                        "stage": stage,
                    }
                )
            return
        if load_item is not None and load_item.isSelected():
            load = load_item.data(0, Qt.UserRole)
            if hasattr(self.sketch_view, "set_panel_attr_focus"):
                self.sketch_view.set_panel_attr_focus("load", load)
            if inspector is not None:
                inspector.set_selection_payload(
                    {
                        "kind": "load",
                        "index": int(load_item.data(0, self._index_role) or 0),
                        "stage": ProjectStage.BCS if self.panel_mode == "combined" else ProjectStage.LOADS,
                    }
                )
            return
        if hasattr(self.sketch_view, "set_panel_attr_focus"):
            self.sketch_view.set_panel_attr_focus(None, None)
        if inspector is not None:
            inspector.set_selection_payload(None)

    def _on_attr_list_selection_changed(self, source):
        if self._attr_list_syncing:
            return
        if source == "bc" and self.bc_list.selectedItems():
            self._set_only_attr_list_active("bc")
        elif source == "load" and self.load_list.selectedItems():
            self._set_only_attr_list_active("load")
        self._sync_attr_entry_preview_from_lists()

    def _get_selected_bc(self):
        if self._is_load_mode:
            return None
        item = self.bc_list.currentItem()
        if not item:
            return None
        if self.panel_mode == "combined" and item.data(0, self._entry_kind_role) != "bc":
            return None
        idx = item.data(0, self._index_role)
        if isinstance(idx, int) and 0 <= idx < len(self.project_state.boundary_conditions):
            return idx, self.project_state.boundary_conditions[idx]
        bc = item.data(0, Qt.UserRole)
        if bc is None:
            return None
        for i, entry in enumerate(self.project_state.boundary_conditions):
            if entry == bc:
                return i, entry
        return None

    def _bc_display_name(self, bc, idx):
        name = str(bc.get("name") or "").strip()
        if name:
            return name
        display = str(bc.get("display_type") or "").strip()
        if display:
            return display
        return f"BC {int(idx) + 1}"

    def _bc_type_label(self, bc):
        bc_kind = self._bc_kind_for_entry(bc)
        if bc_kind:
            return self._bc_kind_label(bc_kind)
        bc_type = str(bc.get("type") or "").strip().lower()
        return str(bc.get("display_type") or bc_type or "BC").strip() or "BC"

    def _bc_tooltip(self, bc, idx):
        lines = [
            f"Name: {self._bc_display_name(bc, idx)}",
            f"Type: {self._bc_type_label(bc)}",
        ]
        display = str(bc.get("display_type") or "").strip()
        if display and display != self._bc_display_name(bc, idx):
            lines.append(f"Detail: {display}")
        return "\n".join(lines)

    def _load_display_name(self, load, idx):
        name = str(load.get("name") or "").strip()
        if name:
            return name
        display = str(load.get("display_type") or "").strip()
        if display:
            return display
        return f"Load {int(idx) + 1}"

    def _load_type_label(self, load):
        load_type = str(load.get("type") or "").strip().lower()
        if load_type == "force":
            return "Force / Load"
        if load_type == "moment":
            return "Moment"
        return str(load.get("display_type") or load_type or "Load").strip() or "Load"

    def _load_tooltip(self, load, idx):
        lines = [
            f"Name: {self._load_display_name(load, idx)}",
            f"Type: {self._load_type_label(load)}",
        ]
        display = str(load.get("display_type") or "").strip()
        if display and display != self._load_display_name(load, idx):
            lines.append(f"Detail: {display}")
        return "\n".join(lines)

    def _get_selected_load(self):
        if self._is_bc_mode:
            return None
        item = self.bc_list.currentItem() if self.panel_mode == "combined" else self.load_list.currentItem()
        if not item:
            return None
        if self.panel_mode == "combined" and item.data(0, self._entry_kind_role) != "load":
            return None
        idx = item.data(0, self._index_role)
        if isinstance(idx, int) and 0 <= idx < len(self.project_state.loads):
            return idx, self.project_state.loads[idx]
        load = item.data(0, Qt.UserRole)
        if load is None:
            return None
        for i, entry in enumerate(self.project_state.loads):
            if entry == load:
                return i, entry
        return None

    def _edit_selected_bc_only(self):
        selected = self._get_selected_bc()
        if not selected:
            QMessageBox.warning(self, "Edit BC", "Select a BC entry first.")
            return
        self._set_only_attr_list_active("bc")
        self._open_bc_edit_dialog()

    def _delete_selected_bc_only(self):
        selected = self._get_selected_bc()
        if not selected:
            QMessageBox.warning(self, "Delete BC", "Select a BC entry first.")
            return
        self.clear_selected()

    def _edit_selected_load_only(self):
        selected = self._get_selected_load()
        if not selected:
            QMessageBox.warning(self, "Edit Load", "Select a load entry first.")
            return
        self._set_only_attr_list_active("load")
        self.edit_selected()

    def _delete_selected_load_only(self):
        selected = self._get_selected_load()
        if not selected:
            QMessageBox.warning(self, "Delete Load", "Select a load entry first.")
            return
        self.clear_selected()

    def clear_selected(self):
        selected_bc = self._get_selected_bc()
        if selected_bc:
            bc_idx, _ = selected_bc
            if self.bc_controller is not None:
                deleted = self.bc_controller.delete_bc(bc_idx, save_velocity=True)
            else:
                self.sketch_view.push_undo_state()
                deleted = False
                if 0 <= bc_idx < len(self.project_state.boundary_conditions):
                    del self.project_state.boundary_conditions[bc_idx]
                    deleted = True
                if deleted:
                    self._sync_attr_fallback_state(emit_bcs=True, save_velocity=True)
            if not deleted:
                return
            self.refresh_lists()
            self.sketch_view.redraw()
            self._announce_status("Deleted selected BC.")
            return

        selected_load = self._get_selected_load()
        if selected_load:
            load_idx, _ = selected_load
            if self.bc_controller is not None:
                deleted = self.bc_controller.delete_load(load_idx)
            else:
                self.sketch_view.push_undo_state()
                deleted = False
                if 0 <= load_idx < len(self.project_state.loads):
                    del self.project_state.loads[load_idx]
                    deleted = True
                if deleted:
                    self._sync_attr_fallback_state(emit_loads=True)
            if not deleted:
                return
            self.refresh_lists()
            self.sketch_view.redraw()
            self._announce_status("Deleted selected load.")
            return
        if self._is_bc_mode:
            QMessageBox.warning(self, "Delete BC", "Select a BC to delete.")
        elif self._is_load_mode:
            QMessageBox.warning(self, "Delete Load", "Select a load to delete.")
        else:
            QMessageBox.warning(self, "Delete Selected", "Select a BC or load to delete.")

    def edit_selected(self):
        selected_bc = self._get_selected_bc()
        if selected_bc:
            self._open_bc_edit_dialog()
            return

        selected_load = self._get_selected_load()
        if selected_load:
            _, load_to_edit = selected_load
            ltype = load_to_edit.get("type")
            if ltype == "force":
                axis = load_to_edit.get("axis")
                if axis is None:
                    fx = float(load_to_edit.get("fx", 0.0))
                    fy = float(load_to_edit.get("fy", 0.0))
                    fz = float(load_to_edit.get("fz", 0.0))
                    if abs(fz) > max(abs(fx), abs(fy)):
                        axis = "z"
                    elif abs(fx) >= abs(fy):
                        axis = "x"
                    else:
                        axis = "y"
                self._open_profile_editor("force", axis, target_kind="load", target_obj=load_to_edit)
            elif ltype == "moment":
                current = float(load_to_edit.get("m", 0.0))
                val, ok = QInputDialog.getDouble(
                    self,
                    "Edit Load",
                    "Moment (N*m):",
                    current,
                    -1e9,
                    1e9,
                    3,
                )
                if ok:
                    if self.bc_controller is not None:
                        self.bc_controller.update_load(
                            selected_load[0],
                            lambda entry: entry.__setitem__("m", val),
                        )
                    else:
                        self.sketch_view.push_undo_state()
                        load_to_edit["m"] = val
                        self._sync_attr_fallback_state(emit_loads=True)
                    self.refresh_lists()
                    self.sketch_view.redraw()
            else:
                QMessageBox.information(
                    self,
                    "No Editable Value",
                    "Selected load has no editable value.",
                )
            return

        if self._is_bc_mode:
            QMessageBox.warning(self, "Edit BC", "Select a BC to edit.")
        elif self._is_load_mode:
            QMessageBox.warning(self, "Edit Load", "Select a load to edit.")
        else:
            QMessageBox.warning(self, "Edit Selected", "Select a BC or load to edit.")

    def _get_total_time(self):
        try:
            window = self._main_window()
            panel = getattr(window, "properties_panel", None) if window is not None else None
            job_tab = getattr(panel, "job_tab", None) if panel is not None else None
            if job_tab is not None:
                if bool(job_tab.auto_dt_checkbox.isChecked()) and float(job_tab.time_step_spin.value()) <= 0.0:
                    job_tab._recompute_dt(silent=True)
                time_step = float(job_tab.time_step_spin.value())
                total_steps = float(job_tab.total_steps_spin.value())
                if total_steps <= 0:
                    total_steps = 10000.0
                total_time = time_step * total_steps
                if total_time > 0:
                    return total_time
        except Exception:
            pass

        project_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(project_dir, "CPD-main", "config.yml")
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
            total_steps = float(sim.get("total_steps", 10000.0))
        except Exception:
            total_steps = 10000.0
        if total_steps <= 0:
            total_steps = 10000.0
        return max(0.0, time_step * total_steps)

    def _safe_eval_expr(self, expr, t_value=0.0):
        if expr is None:
            return None
        text = str(expr).strip()
        if not text:
            return None
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
            return None

    def _extract_scalar_profile(self, entry, axis, kind, mode="absolute"):
        axis = (axis or "x").lower()
        mode_is_percent = str(mode).lower().startswith("percent")
        total_time = self._get_total_time()
        profile = entry.get("time_profile") or []
        if not profile:
            t1_default = 100.0 if mode_is_percent else total_time
            if kind == "velocity":
                default_val = entry.get("val", 0.0)
                return [{"t0": 0.0, "t1": t1_default, "expr": str(default_val)}]
            default_val = entry.get(f"f{axis}", entry.get("fx", 0.0) if axis == "x" else entry.get("fy", 0.0))
            return [{"t0": 0.0, "t1": t1_default, "expr": str(default_val)}]
        scalar = []
        for seg in profile:
            t0 = float(seg.get("t0", 0.0))
            t1 = float(seg.get("t1", total_time))
            if mode_is_percent and total_time > 0.0 and (t0 > 100.0 or t1 > 100.0):
                # Backward compatibility for older profiles saved as absolute time despite percent mode.
                t0 = t0 / total_time * 100.0
                t1 = t1 / total_time * 100.0
            if kind == "velocity":
                expr = seg.get("expr")
                if expr is None:
                    expr = seg.get("vx") if axis == "x" else seg.get("vy") if axis == "y" else seg.get("vz")
            else:
                expr = seg.get("fx") if axis == "x" else seg.get("fy") if axis == "y" else seg.get("fz")
            scalar.append({"t0": t0, "t1": t1, "expr": str(expr) if expr is not None else "0"})
        return scalar

    def _build_force_time_profile(self, profile, axis):
        axis = (axis or "x").lower()
        out = []
        for seg in profile:
            expr = seg.get("expr", "0")
            out.append(
                {
                    "t0": seg.get("t0", 0.0),
                    "t1": seg.get("t1", 0.0),
                    "fx": expr if axis == "x" else "0",
                    "fy": expr if axis == "y" else "0",
                    "fz": expr if axis == "z" else "0",
                }
            )
        return out

    def _open_profile_editor(self, kind, axis, coords=None, ids=None, part_id=None, target_kind=None, target_obj=None):
        total_time = self._get_total_time()
        if total_time <= 0:
            QMessageBox.warning(
                self,
                "Time Profile",
                "Set a valid time step and total steps in the Job tab first.",
            )
            return

        axis = (axis or "x").lower()
        unit = self.sketch_view.current_unit or "m"
        if kind == "velocity":
            title = f"Velocity {axis.upper()} Time Profile ({unit}/s)"
        else:
            title = f"Force {axis.upper()} Time Profile (N)"

        profile = None
        if target_obj is not None:
            mode = target_obj.get("time_profile_mode", "percent")
            profile = self._extract_scalar_profile(target_obj, axis, kind, mode=mode)
        else:
            mode = "percent"
        self._profile_pending = {
            "kind": kind,
            "axis": axis,
            "coords": coords,
            "ids": ids,
            "part_id": part_id,
            "target": target_kind,
            "target_obj": target_obj,
            "target_index": None,
        }
        if target_obj is not None:
            if target_kind == "bc":
                selected = self._get_selected_bc()
                if selected:
                    self._profile_pending["target_index"] = selected[0]
            elif target_kind == "load":
                selected = self._get_selected_load()
                if selected:
                    self._profile_pending["target_index"] = selected[0]
        self._profile_editor_restore_hidden = bool(
            hasattr(self, "advanced_group") and not self.advanced_group.isVisible()
        )
        if self._profile_editor_restore_hidden:
            self.advanced_group.setVisible(True)
        self.profile_editor.set_profile(title, total_time, profile, default_expr="0", mode=mode)
        self._set_advanced_section_expanded(True)
        self._set_profile_section_expanded(True)
        self._update_responsive_layout()
        self._reveal_panel_widget(self.profile_section, top_margin=12)
        if kind == "velocity" and not self._is_load_mode and self._velocity_mode() != "piecewise":
            self.focus_velocity_input(axis)

    def _set_profile_section_expanded(self, expanded):
        expanded = bool(expanded)
        if hasattr(self, "profile_section"):
            self.profile_section.blockSignals(True)
            self.profile_section.setChecked(expanded)
            self.profile_section.blockSignals(False)
        self.profile_editor.setVisible(expanded)
        if expanded:
            self._reveal_panel_widget(getattr(self, "profile_section", None), top_margin=12)

    def _apply_profile_editor(self, profile):
        pending = self._profile_pending or {}
        kind = pending.get("kind")
        axis = pending.get("axis")
        coords = pending.get("coords")
        ids = pending.get("ids")
        part_id = pending.get("part_id")
        target_obj = pending.get("target_obj")
        target_index = pending.get("target_index")
        if not kind or not axis:
            return
        eval_expr = profile[0].get("expr", "0") if profile else "0"
        eval_val = self._safe_eval_expr(eval_expr, 0.0)
        mode = self.profile_editor.get_time_mode()

        if target_obj is not None:
            if kind == "velocity":
                def _update_velocity(entry):
                    entry["time_profile"] = profile
                    entry["val"] = eval_val if eval_val is not None else 0.0
                    entry["time_profile_mode"] = mode
            else:
                time_profile = self._build_force_time_profile(profile, axis)

                def _update_force(entry):
                    entry["time_profile"] = time_profile
                    entry["axis"] = axis
                    entry["fx"] = eval_val if axis == "x" and eval_val is not None else 0.0
                    entry["fy"] = eval_val if axis == "y" and eval_val is not None else 0.0
                    entry["fz"] = eval_val if axis == "z" and eval_val is not None else 0.0
                    entry["time_profile_mode"] = mode
            if kind == "velocity":
                if self.bc_controller is not None and target_index is not None:
                    self.bc_controller.update_bc(target_index, _update_velocity, save_velocity=True)
                else:
                    self.sketch_view.push_undo_state()
                    _update_velocity(target_obj)
                    self.sketch_view.bcsChanged.emit()
                    self.sketch_view.save_velocity_csv()
            else:
                if self.bc_controller is not None and target_index is not None:
                    self.bc_controller.update_load(target_index, _update_force)
                else:
                    self.sketch_view.push_undo_state()
                    _update_force(target_obj)
                    self.sketch_view.loadsChanged.emit()
            self.refresh_lists()
            self.sketch_view.redraw()
            if kind == "velocity":
                self._annotate_3d_attr_marker("velocity", ids=ids, axis=axis)
            elif kind == "force":
                self._annotate_3d_attr_marker("force", ids=ids, axis=axis)
            self._announce_status(
                "Updated time profile for selected velocity BC."
                if kind == "velocity"
                else "Updated time profile for selected force load."
            )
            self._cancel_profile_editor()
            return

        coord_list = []
        if coords is not None:
            if isinstance(coords, list):
                coord_list = coords
            else:
                coord_list = [coords]
        if not coord_list and ids is not None:
            coord_list = [None]
        if not coord_list and part_id is not None:
            coord_list = [None]

        if kind == "velocity":
            bc_entries = []
            for coord in coord_list:
                bc_record = {
                    "type": f"velocity_{axis}",
                    "coords": coord,
                    "val": eval_val if eval_val is not None else 0.0,
                    "time_profile": profile,
                    "time_profile_mode": mode,
                }
                if ids is not None:
                    bc_record["ids"] = list(ids)
                    bc_record["coords"] = []
                    bc_record["target"] = pending.get("target", "")
                if part_id is not None:
                    bc_record["part_id"] = int(part_id)
                    bc_record["coords"] = []
                    bc_record["target"] = pending.get("target", "part")
                if coord is not None:
                    self.sketch_view._del_attrs(coord, record_history=False)
                bc_entries.append(bc_record)
            main = self._main_window()
            execute_command = getattr(main, "execute_app_command", None) if main is not None else None
            if self.bc_controller is None or not callable(execute_command):
                QMessageBox.warning(self, "Boundary Conditions", "Boundary-condition command bus is not available.")
                return
            execute_command(
                AddBoundaryConditionCommand(
                    entries=bc_entries,
                    entry_kind="bc",
                    save_velocity=True,
                )
            )
            new_bc_index = max(0, len(self.project_state.boundary_conditions) - len(bc_entries))
        elif kind == "force":
            time_profile = self._build_force_time_profile(profile, axis)
            load_entries = []
            for coord in coord_list:
                load_record = {
                    "type": "force",
                    "coords": coord,
                    "fx": eval_val if axis == "x" and eval_val is not None else 0.0,
                    "fy": eval_val if axis == "y" and eval_val is not None else 0.0,
                    "fz": eval_val if axis == "z" and eval_val is not None else 0.0,
                    "axis": axis,
                    "time_profile": time_profile,
                    "time_profile_mode": mode,
                }
                if ids is not None:
                    load_record["ids"] = list(ids)
                    load_record["coords"] = []
                    load_record["target"] = pending.get("target", "")
                if part_id is not None:
                    load_record["part_id"] = int(part_id)
                    load_record["coords"] = []
                    load_record["target"] = pending.get("target", "part")
                if coord is not None:
                    self.sketch_view._del_attrs(coord, record_history=False)
                load_entries.append(load_record)
            main = self._main_window()
            execute_command = getattr(main, "execute_app_command", None) if main is not None else None
            if self.bc_controller is None or not callable(execute_command):
                QMessageBox.warning(self, "Boundary Conditions", "Boundary-condition command bus is not available.")
                return
            execute_command(AddBoundaryConditionCommand(entries=load_entries, entry_kind="load"))
            new_bc_index = None
        self.refresh_lists()
        if kind == "velocity":
            self.select_bc_index(new_bc_index)
            self.focus_velocity_input(axis)
        self.sketch_view.redraw()
        if kind == "velocity":
            self._annotate_3d_attr_marker("velocity", ids=ids, axis=axis)
        elif kind == "force":
            self._annotate_3d_attr_marker("force", ids=ids, axis=axis)
        if kind == "velocity":
            if part_id is not None:
                self._announce_status(f"Applied velocity {axis.upper()} profile to part {int(part_id)}.")
            elif ids is not None:
                self._announce_status(f"Applied velocity {axis.upper()} profile to {len(ids)} selected node(s).")
        elif kind == "force":
            if part_id is not None:
                self._announce_status(f"Applied force {axis.upper()} profile to part {int(part_id)}.")
            elif ids is not None:
                self._announce_status(f"Applied force {axis.upper()} profile to {len(ids)} selected node(s).")
        if hasattr(self.sketch_view, "_clear_arc_segment_selection"):
            self.sketch_view._clear_arc_segment_selection()
        self._cancel_profile_editor()

    def _cancel_profile_editor(self):
        self._profile_pending = None
        self._set_profile_section_expanded(False)
        self._set_advanced_section_expanded(False)
        if getattr(self, "_profile_editor_restore_hidden", False):
            self.advanced_group.setVisible(False)
            self._profile_editor_restore_hidden = False
            self._update_responsive_layout()

    def edit_time_profile(self):
        selected_bc = self._get_selected_bc()
        if selected_bc:
            bc_index, bc_to_edit = selected_bc
            btype = bc_to_edit.get("type")
            if btype in ("velocity_x", "velocity_y", "velocity_z"):
                axis = btype.split("_")[-1]
                self._edit_velocity_piecewise_axis(axis, entry=bc_to_edit, target_index=bc_index)
                return
            QMessageBox.information(
                self,
                "Time Profile",
                "Time profiles are available only for velocity BCs and force loads.",
            )
            return

        selected_load = self._get_selected_load()
        if selected_load:
            _, load_to_edit = selected_load
            ltype = load_to_edit.get("type")
            if ltype == "force":
                axis = load_to_edit.get("axis")
                if axis is None:
                    fx = float(load_to_edit.get("fx", 0.0))
                    fy = float(load_to_edit.get("fy", 0.0))
                    fz = float(load_to_edit.get("fz", 0.0))
                    if abs(fz) > max(abs(fx), abs(fy)):
                        axis = "z"
                    elif abs(fx) >= abs(fy):
                        axis = "x"
                    else:
                        axis = "y"
                self._open_profile_editor("force", axis, target_kind="load", target_obj=load_to_edit)
                return
            QMessageBox.information(
                self,
                "Time Profile",
                "Time profiles are available only for velocity BCs and force loads.",
            )
            return

        if self._is_bc_mode:
            QMessageBox.warning(self, "Time Profile", "Select a velocity BC first.")
        elif self._is_load_mode:
            QMessageBox.warning(self, "Time Profile", "Select a force load first.")
        else:
            QMessageBox.warning(self, "Time Profile", "Select a velocity BC or force load first.")

    def apply_time_profile_from_context(self, kind, axis, coords):
        self._open_profile_editor(kind, axis, coords=coords)

    def apply_time_profile_to_part(self, kind, axis, part_id):
        self._open_profile_editor(kind, axis, part_id=int(part_id), target_kind="part")

    def apply_fixed_bc_to_part(self, bc_type, part_id):
        bc_record = {
            "type": bc_type,
            "part_id": int(part_id),
            "target": "part",
            "coords": [],
        }
        main = self._main_window()
        execute_command = getattr(main, "execute_app_command", None) if main is not None else None
        if self.bc_controller is None or not callable(execute_command):
            QMessageBox.warning(self, "Boundary Conditions", "Boundary-condition command bus is not available.")
            return
        execute_command(
            AddBoundaryConditionCommand(
                entries=[bc_record],
                entry_kind="bc",
                save_velocity=True,
            )
        )
        self.refresh_lists()
        self.sketch_view.redraw()
        self._announce_status(f"Applied {bc_type} to part {int(part_id)}.")

    def prompt_scalar_profile(self, kind, axis):
        self._open_profile_editor(kind, axis)
        return None

    def change_selected_type(self):
        selected_bc = self._get_selected_bc()
        if selected_bc:
            bc_index, bc_to_edit = selected_bc
            bc_types = self._bc_type_choices()
            labels = [label for label, _payload in bc_types]
            current_idx = self._bc_choice_index_for_entry(bc_to_edit, bc_types)
            label, ok = QInputDialog.getItem(
                self,
                "Change BC Type",
                "BC Type:",
                labels,
                current_idx,
                False,
            )
            if not ok:
                return
            payload = dict(bc_types)[label]
            new_type = str(payload.get("type") or "fix_x").strip().lower()
            new_mode = str(payload.get("bc_mode") or "").strip().lower()
            display_type = str(payload.get("display_type") or label).strip()
            candidate = {
                "type": new_type,
                "display_type": display_type,
                "coords": copy.deepcopy(bc_to_edit.get("coords", [])),
            }
            if bc_to_edit.get("ids") is not None:
                candidate["ids"] = copy.deepcopy(list(bc_to_edit.get("ids") or []))
                candidate["target"] = bc_to_edit.get("target")
            if bc_to_edit.get("part_id") is not None:
                candidate["part_id"] = bc_to_edit.get("part_id")
                candidate["target"] = bc_to_edit.get("target", "part")
            if new_mode:
                candidate["bc_mode"] = new_mode
                candidate["val"] = float(bc_to_edit.get("val", 0.0) or 0.0)
            if not self._validate_bc_candidates([candidate], ignore_index=bc_index):
                return

            def _update_bc(entry):
                entry["type"] = new_type
                entry["display_type"] = display_type
                if new_mode:
                    entry["bc_mode"] = new_mode
                else:
                    entry.pop("bc_mode", None)
                if new_type.startswith("fix_"):
                    entry.pop("val", None)
                    entry.pop("time_profile", None)
                    entry.pop("time_profile_mode", None)
                    entry.pop("time_value_pairs", None)
                elif new_mode == "velocity":
                    points = [(0.0, float(entry.get("val", 0.0) or 0.0))]
                    entry["val"] = float(points[0][1]) if points else 0.0
                    entry["time_value_pairs"] = list(points)
                    entry["time_profile"] = self._velocity_points_to_profile(points)
                    entry["time_profile_mode"] = "absolute"
                else:
                    entry["val"] = float(entry.get("val", 0.0) or 0.0)
                    entry.pop("time_profile", None)
                    entry.pop("time_profile_mode", None)
                    entry.pop("time_value_pairs", None)
            if self.bc_controller is not None:
                self.bc_controller.update_bc(bc_index, _update_bc, save_velocity=True)
            else:
                self.sketch_view.push_undo_state()
                _update_bc(bc_to_edit)
                self._sync_attr_fallback_state(emit_bcs=True, save_velocity=True)
            self.refresh_lists()
            self.sketch_view.redraw()
            return

        selected_load = self._get_selected_load()
        if selected_load:
            _, load_to_edit = selected_load
            load_types = [
                ("Force (Fx,Fy)", "force"),
                ("Moment", "moment"),
            ]
            labels = [label for label, _ in load_types]
            current_type = load_to_edit.get("type")
            current_idx = next(
                (i for i, (_, t) in enumerate(load_types) if t == current_type), 0
            )
            label, ok = QInputDialog.getItem(
                self,
                "Change Load Type",
                "Load Type:",
                labels,
                current_idx,
                False,
            )
            if not ok:
                return
            new_type = dict(load_types)[label]
            if new_type == "force":
                fx, ok = QInputDialog.getDouble(
                    self,
                    "Force",
                    "Force X (N):",
                    float(load_to_edit.get("fx", 0.0)),
                    -1e9,
                    1e9,
                    3,
                )
                if not ok:
                    return
                fy, ok = QInputDialog.getDouble(
                    self,
                    "Force",
                    "Force Y (N):",
                    float(load_to_edit.get("fy", 0.0)),
                    -1e9,
                    1e9,
                    3,
                )
                if not ok:
                    return
                new_vals = ("force", fx, fy)
            else:
                val, ok = QInputDialog.getDouble(
                    self,
                    "Moment",
                    "Moment (N*m):",
                    float(load_to_edit.get("m", 0.0)),
                    -1e9,
                    1e9,
                    3,
                )
                if not ok:
                    return
                new_vals = ("moment", val, None)
            def _update_load(entry):
                entry["type"] = new_type
                if new_vals[0] == "force":
                    entry["fx"] = new_vals[1]
                    entry["fy"] = new_vals[2]
                    entry.pop("m", None)
                else:
                    entry["m"] = new_vals[1]
                    entry.pop("fx", None)
                    entry.pop("fy", None)
                    entry.pop("fz", None)
                    entry.pop("axis", None)
                    entry.pop("time_profile", None)
                    entry.pop("time_profile_mode", None)
            if self.bc_controller is not None:
                self.bc_controller.update_load(selected_load[0], _update_load)
            else:
                self.sketch_view.push_undo_state()
                _update_load(load_to_edit)
                self._sync_attr_fallback_state(emit_loads=True)
            self.refresh_lists()
            self.sketch_view.redraw()
            return

        if self._is_bc_mode:
            QMessageBox.warning(self, "Change BC Type", "Select a BC to change.")
        elif self._is_load_mode:
            QMessageBox.warning(self, "Change Load Type", "Select a load to change.")
        else:
            QMessageBox.warning(self, "Change Type", "Select a BC or load to change.")

    def move_selected(self):
        selected_bc = self._get_selected_bc()
        if selected_bc:
            _, bc_to_move = selected_bc
            self._begin_move(bc_to_move, "bc")
            return

        selected_load = self._get_selected_load()
        if selected_load:
            _, load_to_move = selected_load
            self._begin_move(load_to_move, "load")
            return

        if self._is_bc_mode:
            QMessageBox.warning(self, "Move BC", "Select a BC to move.")
        elif self._is_load_mode:
            QMessageBox.warning(self, "Move Load", "Select a load to move.")
        else:
            QMessageBox.warning(self, "Move Selected", "Select a BC or load to move.")

    def _show_bc_context_menu(self, pos):
        item = self.bc_list.itemAt(pos)
        if item is None:
            return
        self.bc_list.setCurrentItem(item)
        item.setSelected(True)
        self._on_attr_list_selection_changed("bc")
        entry_kind = str(item.data(0, self._entry_kind_role) or "bc")
        menu = QMenu(self.bc_list)
        if entry_kind == "load":
            menu.addAction("Edit", self.edit_selected)
            menu.addAction("Rename", self._rename_selected_entry)
            menu.addAction("Highlight", self._highlight_selected_entry)
            menu.addSeparator()
            menu.addAction("Delete", self.clear_selected)
        else:
            menu.addAction("Edit", self._open_bc_edit_dialog)
            menu.addAction("Rename", self._rename_selected_entry)
            menu.addAction("Highlight", self._highlight_selected_entry)
            menu.addSeparator()
            menu.addAction("Delete", self._delete_selected_bc_only)
        menu.exec(self.bc_list.viewport().mapToGlobal(pos))

    def _rename_selected_entry(self):
        selected = self._get_selected_bc()
        entry_kind = "bc"
        if selected:
            entry_index, entry = selected
            current_name = self._bc_display_name(entry, entry_index)
            title = "Rename BC"
        else:
            selected = self._get_selected_load()
            if not selected:
                QMessageBox.warning(self, "Rename Entry", "Select an entry first.")
                return
            entry_kind = "load"
            entry_index, entry = selected
            current_name = self._load_display_name(entry, entry_index)
            title = "Rename Load"
        name, ok = QInputDialog.getText(self, title, "Name:", text=current_name)
        if not ok:
            return
        name = str(name or "").strip()
        if not name:
            return

        def _update(entry):
            entry["name"] = name

        if self.bc_controller is not None and entry_kind == "bc":
            self.bc_controller.update_bc(entry_index, _update, save_velocity=True)
        elif self.bc_controller is not None and entry_kind == "load":
            self.bc_controller.update_load(entry_index, _update)
        else:
            self.sketch_view.push_undo_state()
            _update(entry)
            self._sync_attr_fallback_state(
                emit_bcs=entry_kind == "bc",
                emit_loads=entry_kind == "load",
                save_velocity=entry_kind == "bc",
            )
        self.refresh_lists()
        if entry_kind == "bc":
            self.select_bc_index(entry_index)
        self.sketch_view.redraw()

    def _highlight_selected_entry(self):
        selected = self._get_selected_bc() or self._get_selected_load()
        if not selected:
            QMessageBox.warning(self, "Highlight", "Select an entry first.")
            return
        self._sync_attr_entry_preview_from_lists()

    def _bc_type_choices(self):
        axis_labels = {"x": "X", "y": "Y", "z": "Z"}
        choices = [("Fixed Support XY", {"type": "fix_xy", "bc_mode": None, "display_type": "Fixed Support XY"})]
        for axis in self._constraint_axes_visible():
            axis_label = axis_labels.get(axis, axis.upper())
            choices.append((f"Fixed Support {axis_label}", {"type": f"fix_{axis}", "bc_mode": None, "display_type": f"Fixed Support {axis_label}"}))
            choices.append((f"Prescribed Velocity {axis_label}", {"type": f"velocity_{axis}", "bc_mode": "velocity", "display_type": f"Prescribed Velocity {axis_label}"}))
            choices.append((f"Prescribed Displacement {axis_label}", {"type": f"velocity_{axis}", "bc_mode": "displacement", "display_type": f"Prescribed Displacement {axis_label}"}))
        return choices

    def _bc_choice_index_for_entry(self, bc_entry, choices):
        entry_type = str(bc_entry.get("type") or "").strip().lower()
        entry_mode = str(bc_entry.get("bc_mode") or "").strip().lower()
        for idx, (_label, payload) in enumerate(choices):
            if entry_type == str(payload.get("type") or "").strip().lower() and entry_mode == str(payload.get("bc_mode") or "").strip().lower():
                return idx
        if entry_type == "fix_xy":
            return 0
        return 0

    def _open_bc_edit_dialog(self):
        selected = self._get_selected_bc()
        if not selected:
            QMessageBox.warning(self, "Edit BC", "Select a BC entry first.")
            return
        bc_index, bc_entry = selected
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Boundary Condition")
        dialog.setModal(True)
        dialog.setMinimumWidth(360)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        layout.addLayout(form)

        name_edit = QLineEdit(self._bc_display_name(bc_entry, bc_index))
        form.addRow("Name:", name_edit)

        choices = self._bc_type_choices()
        type_combo = QComboBox()
        for label, payload in choices:
            type_combo.addItem(label, payload)
        type_combo.setCurrentIndex(self._bc_choice_index_for_entry(bc_entry, choices))
        form.addRow("Type:", type_combo)

        value_spin = QDoubleSpinBox()
        value_spin.setRange(-1e9, 1e9)
        value_spin.setDecimals(6)
        value_spin.setValue(float(bc_entry.get("val", 0.0) or 0.0))
        value_label = QLabel("u:")
        form.addRow(value_label, value_spin)

        profile_btn = QPushButton("Edit Time Profile")
        profile_summary = QLabel()
        profile_summary.setObjectName("MinorStatusLabel")
        profile_row = QWidget()
        profile_row_layout = QHBoxLayout(profile_row)
        profile_row_layout.setContentsMargins(0, 0, 0, 0)
        profile_row_layout.setSpacing(6)
        profile_row_layout.addWidget(profile_btn)
        profile_row_layout.addWidget(profile_summary)
        profile_row_layout.addStretch(1)
        form.addRow("Time Profile:", profile_row)

        def _refresh_profile_summary():
            points = self._profile_to_velocity_points(bc_entry)
            profile_summary.setText(f"{len(points)} row" if len(points) == 1 else f"{len(points)} rows")

        def _edit_profile():
            updated = self._edit_velocity_piecewise_axis(
                str((type_combo.currentData() or {}).get("type", "velocity_x")).split("_")[-1],
                entry=bc_entry,
                target_index=bc_index,
            )
            if updated:
                _refresh_profile_summary()

        profile_btn.clicked.connect(_edit_profile)
        _refresh_profile_summary()

        def _sync_dialog():
            payload = type_combo.currentData() or {}
            mode = str(payload.get("bc_mode") or "").strip().lower()
            is_fixed = str(payload.get("type") or "").strip().lower().startswith("fix_")
            is_velocity = mode == "velocity"
            axis = str(payload.get("type") or "velocity_x").split("_")[-1].upper()
            if is_velocity:
                value_label.setText(f"v{axis} ({self._current_length_unit()}/s):")
            elif is_fixed:
                value_label.setText("u = 0")
            else:
                value_label.setText(f"u{axis} ({self._current_length_unit()}):")
            value_label.setVisible(not is_fixed and not is_velocity)
            value_spin.setVisible(not is_fixed and not is_velocity)
            profile_row.setVisible(is_velocity)
            profile_btn.setVisible(mode == "velocity")

        type_combo.currentIndexChanged.connect(_sync_dialog)
        _sync_dialog()

        buttons_row = QHBoxLayout()
        layout.addLayout(buttons_row)
        buttons_row.addStretch(1)
        cancel_btn = QPushButton("Cancel")
        save_btn = QPushButton("Save")
        save_btn.setProperty("primary", True)
        buttons_row.addWidget(cancel_btn)
        buttons_row.addWidget(save_btn)
        cancel_btn.clicked.connect(dialog.reject)
        save_btn.clicked.connect(dialog.accept)

        if dialog.exec() != QDialog.Accepted:
            return

        payload = type_combo.currentData() or {}
        name = str(name_edit.text() or "").strip()
        new_type = str(payload.get("type") or "fix_x").strip().lower()
        new_mode = str(payload.get("bc_mode") or "").strip().lower()
        display_type = str(payload.get("display_type") or type_combo.currentText()).strip()
        new_value = float(value_spin.value())
        candidate = {
            "type": new_type,
            "display_type": display_type,
            "coords": copy.deepcopy(bc_entry.get("coords", [])),
        }
        if bc_entry.get("ids") is not None:
            candidate["ids"] = copy.deepcopy(list(bc_entry.get("ids") or []))
            candidate["target"] = bc_entry.get("target")
        if bc_entry.get("part_id") is not None:
            candidate["part_id"] = bc_entry.get("part_id")
            candidate["target"] = bc_entry.get("target", "part")
        if new_mode:
            candidate["bc_mode"] = new_mode
        if new_type.startswith("fix_"):
            pass
        elif new_mode == "velocity":
            candidate["val"] = float(bc_entry.get("val", 0.0) or 0.0)
        else:
            candidate["val"] = new_value
        if not self._validate_bc_candidates([candidate], ignore_index=bc_index):
            return

        def _update(entry):
            entry["name"] = name or self._bc_display_name(entry, bc_index)
            entry["type"] = new_type
            entry["display_type"] = display_type
            if new_mode:
                entry["bc_mode"] = new_mode
            else:
                entry.pop("bc_mode", None)
            if new_type.startswith("fix_"):
                entry.pop("val", None)
                entry.pop("time_profile", None)
                entry.pop("time_profile_mode", None)
                entry.pop("time_value_pairs", None)
            elif new_mode == "velocity":
                points = self._profile_to_velocity_points(entry)
                entry["time_value_pairs"] = list(points)
                entry["time_profile"] = self._velocity_points_to_profile(points)
                entry["time_profile_mode"] = "absolute"
                entry["val"] = float(points[0][1]) if points else 0.0
            else:
                entry["val"] = new_value
                entry.pop("time_profile", None)
                entry.pop("time_profile_mode", None)
                entry.pop("time_value_pairs", None)

        if self.bc_controller is not None:
            self.bc_controller.update_bc(bc_index, _update, save_velocity=True)
        else:
            self.sketch_view.push_undo_state()
            _update(bc_entry)
            self._sync_attr_fallback_state(emit_bcs=True, save_velocity=True)
        self.refresh_lists()
        self.select_bc_index(bc_index)
        self.sketch_view.redraw()

    def _open_add_boundary_condition_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Boundary Condition")
        dialog.setModal(True)
        dialog.setMinimumWidth(420)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        layout.addLayout(form)

        name_edit = QLineEdit()
        form.addRow("Name:", name_edit)

        type_combo = QComboBox()
        type_combo.addItem("Fixed Support", "fixed")
        type_combo.addItem("Prescribed Displacement", "displacement")
        type_combo.addItem("Prescribed Velocity", "velocity")
        type_combo.addItem("Force / Load", "force")
        form.addRow("BC Type:", type_combo)

        target_label = QLabel("Use the current edge selection in the viewport before saving.")
        target_label.setWordWrap(True)
        form.addRow("Target:", target_label)

        dof_box = QGroupBox("Direction")
        dof_layout = QHBoxLayout(dof_box)
        axis_buttons = {}
        axis_colors = {"x": "#d64545", "y": "#2f9d5d", "z": "#2f6fd6"}
        for axis in self._constraint_axes_visible():
            button = QPushButton(axis.upper())
            button.setCheckable(True)
            button.setChecked(axis == "x")
            button.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
            button.setMinimumHeight(30)
            button.setMinimumWidth(DOCK_ICON_BTN_MIN)
            button.setStyleSheet(
                f"""
                QPushButton {{
                    color: {axis_colors[axis]};
                    font-weight: 700;
                    border: 1px solid {axis_colors[axis]};
                    border-radius: 15px;
                    padding: 4px 14px;
                    background: rgba(255, 255, 255, 0.92);
                }}
                QPushButton:checked {{
                    color: white;
                    background: {axis_colors[axis]};
                }}
                """
            )
            axis_buttons[axis] = button
            dof_layout.addWidget(button)
        dof_layout.addStretch(1)
        layout.addWidget(dof_box)

        values_group = QGroupBox("Condition")
        values_form = QFormLayout(values_group)
        equation_row = QWidget()
        equation_layout = QHBoxLayout(equation_row)
        equation_layout.setContentsMargins(0, 0, 0, 0)
        equation_layout.setSpacing(8)
        equation_symbol = QLabel("u =")
        equation_symbol.setStyleSheet("font-weight: 700;")
        equation_symbol.setMinimumWidth(32)
        scalar_spin = QDoubleSpinBox()
        scalar_spin.setRange(-1e9, 1e9)
        scalar_spin.setDecimals(6)
        scalar_spin.setValue(0.0)
        scalar_spin.setMinimumHeight(28)
        scalar_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        static_label = QLabel("0")
        static_label.setStyleSheet("font-weight: 700;")
        unit_label = QLabel(self._current_length_unit())
        unit_label.setObjectName("MinorStatusLabel")
        equation_layout.addWidget(equation_symbol)
        equation_layout.addWidget(scalar_spin, 1)
        equation_layout.addWidget(static_label)
        equation_layout.addWidget(unit_label)
        equation_layout.addStretch(1)
        values_form.addRow("Equation", equation_row)
        hint_label = QLabel("")
        hint_label.setObjectName("MinorStatusLabel")
        hint_label.setWordWrap(True)
        values_form.addRow("", hint_label)
        profile_note = QLabel("Velocity time profiles can be edited after creation from the Advanced section.")
        profile_note.setObjectName("MinorStatusLabel")
        profile_note.setWordWrap(True)
        values_form.addRow("", profile_note)
        layout.addWidget(values_group)

        def _sync_dialog():
            kind = str(type_combo.currentData() or "fixed")
            equation_symbol.setText(self._constraint_symbol(kind))
            unit_label.setText(self._constraint_unit_text(kind))
            hint_label.setText(self._constraint_hint_text(kind))
            scalar_spin.setVisible(kind != "fixed")
            scalar_spin.setEnabled(kind != "fixed")
            static_label.setVisible(kind == "fixed")
            unit_label.setVisible(bool(self._constraint_unit_text(kind)))
            profile_note.setVisible(kind == "velocity")

        type_combo.currentIndexChanged.connect(_sync_dialog)
        _sync_dialog()

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel_btn = QPushButton("Cancel")
        save_btn = QPushButton("Save")
        save_btn.setProperty("primary", True)
        buttons.addWidget(cancel_btn)
        buttons.addWidget(save_btn)
        layout.addLayout(buttons)
        cancel_btn.clicked.connect(dialog.reject)
        save_btn.clicked.connect(dialog.accept)

        if dialog.exec() != QDialog.Accepted:
            return

        edge_ids = self._collect_selection_node_ids("edge")
        target_fields = {"coords": []}
        if edge_ids:
            target_fields["ids"] = list(edge_ids)
            target_fields["target"] = "edge"
        else:
            polyline = list(getattr(self.sketch_view, "_arc_segment_polyline", []) or [])
            if len(polyline) < 2:
                QMessageBox.information(self, "Add Boundary Condition", "Select one or more target edges first.")
                return
            target_fields["coords"] = polyline

        name = str(name_edit.text() or "").strip()
        kind = str(type_combo.currentData() or "fixed")
        main = self._main_window()
        execute_command = getattr(main, "execute_app_command", None) if main is not None else None
        if self.bc_controller is None or not callable(execute_command):
            QMessageBox.warning(self, "Boundary Conditions", "Boundary-condition command bus is not available.")
            return

        axes = [axis for axis, button in axis_buttons.items() if button.isChecked()]
        if not axes:
            QMessageBox.information(self, "Add Boundary Condition", "Select at least one direction.")
            return
        scalar_value = float(scalar_spin.value())
        if kind == "force":
            components = {"x": 0.0, "y": 0.0, "z": 0.0}
            for axis in axes:
                components[axis] = scalar_value
            record = {
                "type": "force",
                "display_type": "Force / Load",
                **target_fields,
                "fx": components["x"],
                "fy": components["y"],
                "fz": components["z"],
                "axis": axes[0] if len(axes) == 1 else None,
                "name": name or "Force / Load",
            }
            execute_command(AddBoundaryConditionCommand(entries=[record], entry_kind="load"))
        else:
            records = []
            for axis in axes:
                record = {
                    **target_fields,
                    "coords": copy.deepcopy(target_fields.get("coords", [])),
                    "name": name or (
                        f"{self._bc_kind_label(kind)} {axis.upper()}"
                        if kind != "fixed"
                        else f"Fixed Support {axis.upper()}"
                    ),
                }
                if kind == "fixed":
                    record["type"] = f"fix_{axis}"
                    record["display_type"] = f"Fixed Support {axis.upper()}"
                else:
                    record["type"] = f"velocity_{axis}"
                    record["bc_mode"] = kind
                    record["val"] = scalar_value
                    if kind == "velocity":
                        points = [(0.0, scalar_value)]
                        record["time_value_pairs"] = list(points)
                        record["time_profile"] = self._velocity_points_to_profile(points)
                        record["time_profile_mode"] = "absolute"
                        record["display_type"] = f"Prescribed Velocity {axis.upper()}"
                    else:
                        record["display_type"] = f"Prescribed Displacement {axis.upper()}"
                records.append(record)
            if not self._validate_bc_candidates(records):
                return
            execute_command(AddBoundaryConditionCommand(entries=records, entry_kind="bc", save_velocity=True))

        self.refresh_lists()
        self.sketch_view.redraw()
        self._announce_status("Boundary condition created.")

    def _begin_move(self, item, kind):
        if item.get("part_id") is not None and not item.get("ids") and not item.get("coords"):
            QMessageBox.information(
                self,
                "Move Selected",
                "Part-level BC/load applies to the entire part and cannot be moved.\n"
                "Select and edit the entry instead.",
            )
            return
        self.sketch_view.set_module("Boundary" if kind == "bc" else "Load")
        self.sketch_view.set_tool("select")
        main = self._main_window()
        if main and hasattr(main, "properties_panel"):
            if kind == "bc" and hasattr(main.properties_panel, "bcs_tab"):
                main.properties_panel.tabs.setCurrentWidget(main.properties_panel.bcs_tab)
            elif hasattr(main.properties_panel, "loads_tab"):
                main.properties_panel.tabs.setCurrentWidget(main.properties_panel.loads_tab)
        self.sketch_view.start_attr_reassign(item, kind)

    def _update_paint_brush(self):
        brush_type = self._paint_types.get(self.paint_type_combo.currentText(), "fix_xy")
        unit = self.sketch_view.current_unit or "m"
        self.force_axis_combo.setVisible(brush_type == "force")
        if brush_type in ("velocity_x", "velocity_y", "velocity_z"):
            self.paint_val_label.setText(f"Val ({unit}/s)")
        elif brush_type == "force_z":
            self.paint_val_label.setText("Val (N)")
        elif brush_type == "force":
            axis = self._selected_force_axis().upper()
            self.paint_val_label.setText(f"Val F{axis} (N)")
        else:
            self.paint_val_label.setText("Val")
        self.sketch_view.set_paint_brush(
            brush_type,
            fx=self.paint_fx.value(),
            fy=self.paint_fy.value(),
            val=self.paint_val.value(),
        )

    def on_paint_toggled(self, enabled):
        if enabled and self.sketch_view.active_module not in ("Load", "Boundary"):
            QMessageBox.warning(self, "Paint Brush", "Switch to the Boundary Conditions stage to paint.")
            self.paint_button.setChecked(False)
            return
        self._update_paint_brush()
        if enabled:
            self.sketch_view.set_tool("paint_bc")
        else:
            self.sketch_view.set_tool("select")
        self._update_action_button_density()


class JobPanel(QWidget):
    """A widget for managing and monitoring simulation jobs."""

    def __init__(self, sketch_view, parent=None, project_state=None):
        super().__init__(parent)
        self.sketch_view = sketch_view
        self.project_state = _resolve_project_state(sketch_view, project_state)
        self.solver_controller = None
        self._connected_job_manager = None
        self._current_job_id = None
        self._auto_visualized = False
        self._total_steps = None
        self._refresh_percent = 2.0
        self._progress_total_steps = None
        self._progress_pattern = re.compile(r"Time steps completed\s*[:=]?\s*(\d+)", re.IGNORECASE)
        self._pause_active = False
        self._pause_supported = (
            hasattr(signal, "SIGSTOP")
            and hasattr(signal, "SIGCONT")
            and os.name != "nt"
        )
        self._solver_status_pct = None
        self._recompute_label_full = "Recompute"
        self._run_label_full = "Run Simulation"
        self._pause_label_full = "Pause"
        self._stop_label_full = "Cancel"

        layout = QVBoxLayout(self)
        _apply_layout_metrics(layout)

        layout.addWidget(QLabel("<b>Job Manager</b>"))

        settings_container = QWidget()
        settings_container_layout = QVBoxLayout(settings_container)
        _apply_layout_metrics(
            settings_container_layout,
            margins=(0, 0, 0, 0),
            spacing=DOCK_SECTION_SPACING,
        )

        self.settings_form = QFormLayout()
        self.settings_form.setLabelAlignment(Qt.AlignLeft)
        self.settings_form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.settings_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.settings_form.setHorizontalSpacing(8)
        self.settings_form.setVerticalSpacing(6)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "gpu"])
        self.settings_form.addRow("Device:", self.device_combo)

        self.g_spin = QDoubleSpinBox()
        self.g_spin.setRange(-1e4, 1e4)
        self.g_spin.setDecimals(3)
        self.g_spin.setValue(0.0)
        self.settings_form.addRow("g:", self.g_spin)

        self.time_step_spin = QDoubleSpinBox()
        self.time_step_spin.setDecimals(8)
        self.time_step_spin.setRange(1e-9, 1e6)
        self.time_step_spin.setSingleStep(0.0001)
        self.settings_form.addRow("Time step (s):", self.time_step_spin)

        self.auto_dt_checkbox = QCheckBox("Auto dt from connections + material")
        self.auto_dt_checkbox.setChecked(True)
        self.auto_dt_checkbox.toggled.connect(self._on_auto_dt_toggled)
        self.recompute_dt_btn = QPushButton("Recompute")
        self.recompute_dt_btn.clicked.connect(self._recompute_dt)
        self.recompute_dt_btn.setIcon(_style_icon(self, "SP_BrowserReload", "redo", ("view-refresh",)))
        auto_dt_row = QHBoxLayout()
        auto_dt_row.addWidget(self.auto_dt_checkbox)
        auto_dt_row.addWidget(self.recompute_dt_btn)
        auto_dt_row.addStretch(1)
        self.settings_form.addRow("Auto dt:", auto_dt_row)
        self.dt_info_label = QLabel("dt: --")
        self.dt_info_label.setObjectName("MinorStatusLabel")
        self.settings_form.addRow("", self.dt_info_label)

        self.total_steps_spin = QSpinBox()
        max_step = min(2**31 - 1, 100_000_000)
        self.total_steps_spin.setRange(1, max_step)
        self.total_steps_spin.setSingleStep(1000)
        self.settings_form.addRow("Total steps:", self.total_steps_spin)

        self.refresh_spin = QDoubleSpinBox()
        self.refresh_spin.setRange(0.01, 100.0)
        self.refresh_spin.setDecimals(2)
        self.refresh_spin.setSingleStep(0.5)
        self.refresh_spin.setValue(self._refresh_percent)
        self.settings_form.addRow("Write every %:", self.refresh_spin)

        self.auto_save_results = QCheckBox("Auto-save results")
        self.auto_save_results.setChecked(True)
        self.settings_form.addRow("Results:", self.auto_save_results)

        settings_container_layout.addLayout(self.settings_form)

        info_row = QHBoxLayout()
        self.total_time_label = QLabel("Total time: --")
        info_row.addWidget(self.total_time_label)
        self.refresh_label = QLabel("Refresh: --")
        info_row.addWidget(self.refresh_label)
        info_row.addStretch(1)
        settings_container_layout.addLayout(info_row)

        layout.addWidget(settings_container)
        layout.addWidget(_make_dock_separator())

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setObjectName("JobProgressBar")
        layout.addWidget(self.progress_bar)

        control_row = QHBoxLayout()
        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.run_job)
        self.run_button.setIcon(get_icon("play"))
        self.run_button.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.run_button.setToolTip("Run Simulation")
        self.run_button.setText("")
        control_row.addWidget(self.run_button)
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.setEnabled(False)
        self.pause_button.setIcon(get_icon("pause"))
        self.pause_button.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.pause_button.setToolTip("Pause")
        self.pause_button.setText("")
        self.pause_button.setVisible(False)
        self.stop_button = QPushButton("Cancel")
        self.stop_button.clicked.connect(self.stop_job)
        self.stop_button.setEnabled(False)
        self.stop_button.setIcon(get_icon("stop"))
        self.stop_button.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.stop_button.setToolTip("Cancel")
        self.stop_button.setText("")
        control_row.addWidget(self.stop_button)
        control_row.addStretch(1)
        layout.addLayout(control_row)

        self.status_label = QLabel("Status: Idle")
        layout.addWidget(self.status_label)
        self.elapsed_label = QLabel("Elapsed: 0.0 s")
        layout.addWidget(self.elapsed_label)
        self.eta_label = QLabel("ETA: --")
        layout.addWidget(self.eta_label)

        self.output_log = QTextEdit()
        self.output_log.setReadOnly(True)
        self.output_log.setFont(QFont("Courier", 9))
        layout.addWidget(QLabel("Output Log:"))
        layout.addWidget(self.output_log, 1)

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._update_elapsed)
        self._job_start_time = None
        self._pause_started = None
        self._paused_total = 0.0
        self._last_progress_step = 0
        self._load_simulation_defaults()
        self.time_step_spin.valueChanged.connect(self._update_sim_labels)
        self.refresh_spin.valueChanged.connect(self._on_refresh_changed)
        self.total_steps_spin.valueChanged.connect(self._on_total_steps_changed)
        self.device_combo.currentTextChanged.connect(self._sync_solver_settings_state)
        self.g_spin.valueChanged.connect(self._sync_solver_settings_state)
        self.time_step_spin.valueChanged.connect(self._sync_solver_settings_state)
        self.total_steps_spin.valueChanged.connect(self._sync_solver_settings_state)
        self.refresh_spin.valueChanged.connect(self._sync_solver_settings_state)
        self.auto_dt_checkbox.toggled.connect(self._sync_solver_settings_state)
        self.auto_save_results.toggled.connect(self._sync_solver_settings_state)
        self._update_sim_labels()
        self._on_auto_dt_toggled(self.auto_dt_checkbox.isChecked())
        self._sync_solver_settings_state()
        self._update_action_button_density()
        _finalize_dock_panel(self)

    def _update_action_button_density(self):
        width = int(self.contentsRect().width() or self.width() or 0)
        compact = width < 430
        icon_only = width < 350
        _set_responsive_button_text(
            self.recompute_dt_btn,
            full=self._recompute_label_full,
            compact="Recompute" if compact else None,
            icon_only=icon_only,
        )
        _set_responsive_button_text(
            self.run_button,
            full=self._run_label_full,
            compact="Run",
            icon_only=icon_only,
        )
        _set_responsive_button_text(
            self.pause_button,
            full=self._pause_label_full,
            compact="Pause",
            icon_only=icon_only,
        )
        _set_responsive_button_text(
            self.stop_button,
            full=self._stop_label_full,
            compact="Cancel",
            icon_only=icon_only,
        )
        row_wrap = QFormLayout.WrapLongRows if width < 360 else QFormLayout.DontWrapRows
        self.settings_form.setRowWrapPolicy(row_wrap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_action_button_density()

    def export_solver_settings(self):
        return {
            "device": self.device_combo.currentText().strip().lower(),
            "g": float(self.g_spin.value()),
            "time_step": float(self.time_step_spin.value()),
            "total_steps": int(self.total_steps_spin.value()),
            "write_every_percent_steps": float(self.refresh_spin.value()),
            "auto_dt_enabled": bool(self.auto_dt_checkbox.isChecked()),
            "auto_save_results": bool(self.auto_save_results.isChecked()),
        }

    def apply_solver_settings(self, settings):
        if not isinstance(settings, dict):
            return
        device = str(settings.get("device", "")).strip().lower()
        if device in ("cpu", "gpu"):
            self.device_combo.setCurrentText(device)
        if "g" in settings:
            try:
                self.g_spin.setValue(float(settings.get("g", 0.0)))
            except Exception:
                pass
        if "time_step" in settings:
            try:
                self.time_step_spin.setValue(float(settings.get("time_step", self.time_step_spin.value())))
            except Exception:
                pass
        if "total_steps" in settings:
            try:
                self.total_steps_spin.setValue(int(settings.get("total_steps", self.total_steps_spin.value())))
            except Exception:
                pass
        if "write_every_percent_steps" in settings:
            try:
                self.refresh_spin.setValue(
                    float(settings.get("write_every_percent_steps", self.refresh_spin.value()))
                )
            except Exception:
                pass
        if "auto_dt_enabled" in settings:
            self.auto_dt_checkbox.setChecked(bool(settings.get("auto_dt_enabled")))
        if "auto_save_results" in settings:
            self.auto_save_results.setChecked(bool(settings.get("auto_save_results")))
        self._sync_solver_settings_state()

    def set_solver_controller(self, controller):
        self._disconnect_job_manager(self._connected_job_manager)
        self._connected_job_manager = None
        self.solver_controller = controller
        manager = getattr(controller, "job_manager", None) if controller is not None else None
        if manager is None:
            return
        try:
            manager.jobOutput.connect(self._on_job_output)
            manager.jobProgress.connect(self._on_job_progress)
            manager.jobStatus.connect(self._on_job_status)
            manager.jobStarted.connect(self._on_job_started)
            manager.jobFinished.connect(self._on_job_finished)
            self._connected_job_manager = manager
        except Exception:
            pass

    def _disconnect_job_manager(self, manager):
        if manager is None:
            return
        for signal_name, slot in (
            ("jobOutput", self._on_job_output),
            ("jobProgress", self._on_job_progress),
            ("jobStatus", self._on_job_status),
            ("jobStarted", self._on_job_started),
            ("jobFinished", self._on_job_finished),
        ):
            signal_obj = getattr(manager, signal_name, None)
            if signal_obj is None:
                continue
            try:
                signal_obj.disconnect(slot)
            except Exception:
                pass

    def _sync_solver_settings_state(self, *_args):
        state = self.project_state
        if state is None:
            state = getattr(self.window(), "project_state", None)
            self.project_state = state
        if state is None:
            return
        if not isinstance(getattr(state, "solver_settings", None), dict):
            try:
                state.solver_settings = {}
            except Exception:
                return
        state.solver_settings.update(self.export_solver_settings())

    def run_job(self):
        main = self.window()
        execute_command = getattr(main, "execute_app_command", None) if main is not None else None
        if self.solver_controller is None or not callable(execute_command):
            QMessageBox.warning(self, "Solver Controller", "Solver command bus is not available.")
            return
        result = execute_command(RunSolverCommand()) or {}
        job = result.get("job")
        if job is None:
            return
        self.status_label.setText(f"Status: {str(job.status).title()}")
        self.log_output(f"Started {job.job_id}")

    def _focus_stage(self, stage):
        if not isinstance(stage, ProjectStage):
            return
        main = self.window()
        if (
            main is not None
            and hasattr(main, "active_stage")
            and stage.value > main.active_stage.value
        ):
            self.sketch_view.stageAdvanceRequested.emit(stage)
            return
        if (
            main is not None
            and hasattr(main, "properties_panel")
            and hasattr(main.properties_panel, "tabs")
        ):
            if hasattr(main, "_stage_to_tab_index"):
                try:
                    tab_index = int(main._stage_to_tab_index(stage))
                except Exception:
                    tab_index = 0
            else:
                tab_index_map = {
                    ProjectStage.GEOMETRY: 0,
                    ProjectStage.MATERIALS: 1,
                    ProjectStage.INTERFACES: 2,
                    ProjectStage.BCS: 3,
                    ProjectStage.LOADS: 3,
                    ProjectStage.MESH: 4,
                    ProjectStage.JOB: 5,
                    ProjectStage.RESULTS: 6,
                }
                tab_index = tab_index_map.get(stage, 0)
            main.properties_panel.tabs.setCurrentIndex(tab_index)

    def _first_missing_run_stage(self):
        parts = [
            p for p in getattr(self.sketch_view, "parts", [])
            if not getattr(p, "is_void", False)
        ]
        if not parts:
            return (
                ProjectStage.GEOMETRY,
                "Create and confirm geometry first, then generate particles in the Particles stage.",
            )

        bcs = getattr(self.sketch_view, "bcs", []) or []
        loads = getattr(self.sketch_view, "loads", []) or []
        global_nodes = getattr(self.sketch_view, "global_nodes", np.array([]))
        global_elements = getattr(self.sketch_view, "global_elements", np.array([]))
        particles_ready = global_nodes is not None and len(global_nodes) > 0
        connections_ready = particles_ready and global_elements is not None and len(global_elements) > 0
        if not particles_ready:
            return (
                ProjectStage.MESH,
                "Particles are not generated. Go to the Particles stage and generate particles before Run.",
            )
        if not connections_ready:
            return (
                ProjectStage.MESH,
                "Connections are not generated. Click Generate Connections in the Particles stage before Run.",
            )

        missing_material = [
            p.name for p in parts
            if getattr(p, "material_id", None) is None
        ]
        if missing_material:
            return (
                ProjectStage.MATERIALS,
                "Assign materials to all solid parts before running simulation.",
            )

        if not bcs and not loads:
            return (
                ProjectStage.BCS,
                "Define at least one boundary condition or load before running simulation.",
            )

        needs_part_mapping = any(item.get("part_id") is not None for item in bcs) or any(
            item.get("part_id") is not None for item in loads
        )
        element_part_map = getattr(self.sketch_view, "element_part_map", None)
        if needs_part_mapping and (element_part_map is None or len(element_part_map) == 0):
            return (
                ProjectStage.MESH,
                "Part-to-particle mapping is missing. Rebuild particles in the Particles stage before Run.",
            )

        return None, ""

    def log_output(self, text):
        self.output_log.append(text)
        self.output_log.verticalScrollBar().setValue(
            self.output_log.verticalScrollBar().maximum()
        )

    def update_status(self, status):
        normalized = str(status or "").strip()
        self.status_label.setText(f"Status: {normalized.title() if normalized else 'Idle'}")
        lower = normalized.lower()
        if lower == "paused":
            self._set_pause_state(True)
        elif lower == "running":
            self._set_pause_state(False)
        elif lower.startswith("completed"):
            self.progress_bar.setValue(100)
        elif lower.startswith("failed"):
            self.progress_bar.setValue(0)
        if lower in ("completed", "failed", "stopped"):
            self._elapsed_timer.stop()
            self._set_pause_state(False)
            self._pause_started = None
            self.eta_label.setText("ETA: --")
        if lower == "completed":
            self._announce_status("Simulation completed successfully.", 4000)
        elif lower == "stopped":
            self._announce_status("Simulation stopped.", 4000)
        elif lower.startswith("failed"):
            self._announce_status("Simulation failed.", 4000)

    def _job_matches(self, job_id):
        return not self._current_job_id or str(job_id) == str(self._current_job_id)

    def _on_job_started(self, job):
        if job is None:
            return
        self._current_job_id = getattr(job, "job_id", None)
        self.update_status(getattr(job, "status", "queued"))

    def _on_job_output(self, job_id, text):
        if not self._job_matches(job_id):
            return
        self.log_output(text)

    def _on_job_progress(self, job_id, pct):
        if not self._job_matches(job_id):
            return
        try:
            pct = max(0, min(100, int(pct)))
        except Exception:
            return
        self.progress_bar.setValue(pct)
        if self._progress_total_steps:
            self._last_progress_step = int((pct / 100.0) * max(1, self._progress_total_steps))
        self._update_eta()
        self._update_status_bar_message(pct)

    def _on_job_status(self, job_id, status):
        if not self._job_matches(job_id):
            return
        self.update_status(status)

    def _on_job_finished(self, job):
        if job is None:
            return
        if not self._job_matches(getattr(job, "job_id", None)):
            return
        self.run_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self._current_job_id = None
        log_path = str(getattr(job, "log_path", "") or "")
        if log_path:
            self.log_output(f"Solver log: {log_path}")
        if getattr(job, "status", "") == "completed":
            if self.auto_save_results.isChecked():
                saved_dir = self._auto_save_results()
                if saved_dir:
                    self.log_output(f"Auto-saved results to: {saved_dir}")
            self._persist_results_to_project_artifacts()

    def _current_elapsed(self):
        if self._job_start_time is None:
            return 0.0
        now = time.perf_counter()
        paused = self._paused_total
        if self._pause_started is not None:
            paused += now - self._pause_started
        return max(0.0, now - self._job_start_time - paused)

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

    def _announce_status(self, message, timeout=3000):
        window = self.window()
        if window and hasattr(window, "statusBar"):
            window.statusBar().showMessage(message, timeout)

    def _update_eta(self):
        if not self._progress_total_steps or self._last_progress_step <= 0:
            self.eta_label.setText("ETA: --")
            return
        step = min(self._last_progress_step, self._progress_total_steps)
        if step >= self._progress_total_steps:
            self.eta_label.setText("ETA: 0s")
            return
        elapsed = self._current_elapsed()
        remaining = elapsed * (self._progress_total_steps - step) / max(step, 1)
        self.eta_label.setText(f"ETA: {self._format_duration(remaining)}")

    def _update_status_bar_message(self, pct):
        if pct is None:
            return
        try:
            pct = max(0, min(100, int(pct)))
        except Exception:
            return
        last_pct = self._solver_status_pct
        if last_pct is not None and abs(pct - last_pct) < 5 and pct not in (0, 100):
            return
        self._solver_status_pct = pct
        eta_text = self.eta_label.text()
        detail = ""
        if eta_text and eta_text.startswith("ETA:"):
            eta_value = eta_text[4:].strip()
        else:
            eta_value = eta_text
        if eta_value and eta_value not in ("--", ""):
            detail = f" (ETA {eta_value})"
        self._announce_status(f"Simulation {pct}%{detail}", 2500)

    def _persist_results_to_project_artifacts(self):
        window = self.window()
        if not window:
            return
        project_file = getattr(window, "current_project_file", None)
        if not project_file:
            self.log_output("Results persistence skipped: save the project to store reusable artifacts.")
            return
        export_fn = getattr(window, "_export_job_artifacts", None)
        if not callable(export_fn):
            return
        try:
            export_fn(project_file, export_inputs=False, include_results=True)
            self.log_output("Results persisted to project artifacts for reuse.")
        except Exception as exc:
            self.log_output(f"Warning: could not persist results artifacts ({exc}).")

    def _update_elapsed(self):
        if self._job_start_time is None:
            return
        elapsed = self._current_elapsed()
        self.elapsed_label.setText(f"Elapsed: {elapsed:.2f} s")
        self._update_eta()

    def toggle_pause(self):
        self.log_output("Pause/Resume is not available in the current job manager.")

    def stop_job(self):
        controller = self.solver_controller or getattr(self.window(), "solver_controller", None)
        if controller is None or not hasattr(controller, "cancel_job"):
            return
        controller.cancel_job()
        self.log_output("--- CANCEL REQUESTED ---")
        self.stop_button.setEnabled(False)
        self.update_status("stopping")

    def _set_pause_state(self, paused):
        self._pause_active = bool(paused)
        if self._pause_active:
            self.pause_button.setIcon(get_icon("play"))
            self.pause_button.setToolTip("Resume")
        else:
            self.pause_button.setIcon(get_icon("pause"))
            self.pause_button.setToolTip("Pause")

    def _config_path(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, "CPD-main", "config.yml")

    def _auto_save_results(self):
        workspace_dir = _workspace_dir()
        results_src = _workspace_output_path("results")
        if not os.path.isdir(results_src):
            results_src = os.path.join(workspace_dir, "results")
        pos_history_src = _workspace_output_path("pos_history.npy")
        if not os.path.exists(pos_history_src):
            pos_history_src = os.path.join(workspace_dir, "pos_history.npy")
        if not os.path.isdir(results_src) and not os.path.exists(pos_history_src):
            return None

        saved_root = os.path.join(workspace_dir, "saved_results")
        os.makedirs(saved_root, exist_ok=True)
        stamp = time.strftime("auto_%Y%m%d_%H%M%S")
        dest_dir = os.path.join(saved_root, stamp)
        os.makedirs(dest_dir, exist_ok=True)

        if os.path.isdir(results_src):
            shutil.copytree(results_src, os.path.join(dest_dir, "results"))
        if os.path.exists(pos_history_src):
            shutil.copy2(pos_history_src, os.path.join(dest_dir, "pos_history.npy"))
        for fname in ("displacement_history.npy", "strain_history.npy", "stress_history.npy"):
            src = _workspace_output_path(fname)
            if not os.path.exists(src):
                src = os.path.join(workspace_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dest_dir, fname))
        initial_pos = _workspace_output_path("initial_pos.csv")
        final_pos = _workspace_output_path("final_pos.csv")
        if not os.path.exists(initial_pos):
            initial_pos = os.path.join(workspace_dir, "initial_pos.csv")
        if not os.path.exists(final_pos):
            final_pos = os.path.join(workspace_dir, "final_pos.csv")
        if os.path.exists(initial_pos):
            shutil.copy2(initial_pos, os.path.join(dest_dir, "initial_pos.csv"))
        if os.path.exists(final_pos):
            shutil.copy2(final_pos, os.path.join(dest_dir, "final_pos.csv"))

        return dest_dir

    def _read_sim_config(self):
        config_path = self._config_path()
        if not os.path.exists(config_path):
            return {}
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    def _write_sim_config(self, config):
        config_path = self._config_path()
        try:
            with open(config_path, "w") as f:
                yaml.safe_dump(config, f, sort_keys=False)
            return True
        except Exception:
            return False

    def _load_simulation_defaults(self):
        config = self._read_sim_config()
        sim = config.get("simulation", {})
        time_step = sim.get("time_step", 0.0)
        total_steps = sim.get("total_steps", 10000)
        g_val = sim.get("gravity", 0.0)
        device_val = sim.get("device", "cpu")
        self.device_combo.setCurrentText(str(device_val).lower())
        try:
            self.g_spin.setValue(float(g_val))
        except Exception:
            self.g_spin.setValue(0.0)
        try:
            self._total_steps = int(total_steps)
        except Exception:
            self._total_steps = 10000
        if self._total_steps <= 0:
            self._total_steps = 10000
        self._progress_total_steps = self._total_steps
        try:
            self.time_step_spin.setValue(float(time_step))
        except Exception:
            self.time_step_spin.setValue(0.0)
        self.total_steps_spin.setValue(self._total_steps)
        try:
            write_every_steps = int(sim.get("write_every_steps", 0))
            if self._total_steps > 0 and write_every_steps > 0:
                self._refresh_percent = max(0.0, min(100.0, 100.0 * write_every_steps / self._total_steps))
        except Exception:
            pass
        self.refresh_spin.setValue(self._refresh_percent)

    def _on_auto_dt_toggled(self, enabled):
        enabled = bool(enabled)
        self.time_step_spin.setReadOnly(enabled)
        if enabled:
            self.time_step_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
            self._recompute_dt(silent=True)
        else:
            self.time_step_spin.setButtonSymbols(QAbstractSpinBox.UpDownArrows)
            self.dt_info_label.setText("dt: manual")

    def _compute_min_dx(self):
        nodes = getattr(self.sketch_view, "global_nodes", None)
        elements = getattr(self.sketch_view, "global_elements", None)
        if nodes is None or len(nodes) == 0:
            return None
        nodes = np.asarray(nodes, dtype=float)
        min_dx = None
        if elements is not None and len(elements) > 0:
            elements = np.asarray(elements, dtype=int)
            if elements.ndim == 2 and elements.shape[1] >= 3:
                for tri in elements:
                    idxs = [int(i) for i in tri[:3]]
                    p0, p1, p2 = nodes[idxs[0]], nodes[idxs[1]], nodes[idxs[2]]
                    for a, b in ((p0, p1), (p1, p2), (p2, p0)):
                        d = float(np.linalg.norm(a - b))
                        if d <= 0:
                            continue
                        if min_dx is None or d < min_dx:
                            min_dx = d
        if min_dx is None:
            if len(nodes) > 2000:
                return None
            diff = nodes[:, None, :] - nodes[None, :, :]
            dist = np.linalg.norm(diff, axis=-1)
            dist[dist == 0] = np.inf
            min_dx = float(np.min(dist))
        scale = 1.0
        if hasattr(self.sketch_view, "_unit_scale_to_meters"):
            try:
                scale = float(self.sketch_view._unit_scale_to_meters())
            except Exception:
                scale = 1.0
        return min_dx * scale if min_dx is not None else None

    def _compute_dt_from_model(self):
        mat = self._select_material_for_sim()
        if mat is None:
            return None, "dt: no material"
        props = self._map_material_to_cpd(mat)
        E = float(props.get("E", 0.0))
        rho = float(props.get("rho", 0.0))
        if E <= 0 or rho <= 0:
            return None, "dt: invalid E or density"
        min_dx = self._compute_min_dx()
        if min_dx is None or min_dx <= 0:
            unit_scale = 1.0
            try:
                unit_scale = float(self.sketch_view._unit_scale_to_meters())
            except Exception:
                unit_scale = 1.0
            min_dx = max(1e-9, float(DEFAULT_DX) * unit_scale)
            dt = 0.9 * min_dx / math.sqrt(E / rho)
            unit = getattr(self.sketch_view, "current_unit", "m") or "m"
            return dt, f"dt: {dt:.6g} s (default dx={DEFAULT_DX:.3f} {unit})"
        dt = 0.9 * min_dx / math.sqrt(E / rho)
        return dt, f"dt: {dt:.6g} s (dx={min_dx:.3e} m)"

    def _recompute_dt(self, silent=False):
        dt, info = self._compute_dt_from_model()
        if dt is None:
            self.dt_info_label.setText(info or "dt: --")
            if not silent:
                self.log_output("Auto dt skipped: " + (info or "missing data"))
            return False
        self.time_step_spin.blockSignals(True)
        self.time_step_spin.setValue(float(dt))
        self.time_step_spin.blockSignals(False)
        self.dt_info_label.setText(info)
        self._update_sim_labels()
        return True

    def _on_total_steps_changed(self, value):
        try:
            self._total_steps = int(value)
        except Exception:
            self._total_steps = 10000
        if self._total_steps <= 0:
            self._total_steps = 10000
        self._progress_total_steps = self._total_steps
        self._update_sim_labels()

    def _on_refresh_changed(self, value):
        try:
            self._refresh_percent = float(value)
        except Exception:
            self._refresh_percent = 2.0
        self._update_sim_labels()

    def _update_sim_labels(self):
        if self._total_steps:
            total_time = float(self._total_steps) * float(self.time_step_spin.value())
            self.total_time_label.setText(f"Total time: {total_time:.6g} s")
            write_steps = max(1, int(self._refresh_percent * self._total_steps / 100))
            frame_count = self._total_steps // write_steps + 1
            self.refresh_label.setText(f"Refresh: {self._refresh_percent:.0f}% ({frame_count} frames)")
        else:
            self.total_time_label.setText("Total time: --")
            self.refresh_label.setText(f"Refresh: {self._refresh_percent:.0f}%")

    def _update_progress_from_output(self, text):
        if not text or not self._progress_total_steps:
            return
        match = self._progress_pattern.search(text)
        if not match:
            return
        try:
            step = int(match.group(1))
        except Exception:
            return
        self._last_progress_step = step
        pct = int(min(100, (step / max(1, self._progress_total_steps)) * 100))
        if pct != self.progress_bar.value():
            self.progress_bar.setValue(pct)
        self._update_eta()
        self._update_status_bar_message(pct)

    def _select_material_for_sim(self):
        for part in self.project_state.parts:
            if part.is_void:
                continue
            if part.material_id and part.material_id in self.project_state.materials:
                return self.project_state.materials[part.material_id]
        mat_id = getattr(self.sketch_view, "current_material_id", None)
        if mat_id and mat_id in self.project_state.materials:
            return self.project_state.materials[mat_id]
        if self.project_state.materials:
            return next(iter(self.project_state.materials.values()))
        return None

    def _map_material_to_cpd(self, mat):
        props = mat.properties or {}
        mat_type = (mat.mat_type or "").upper()
        rho = float(props.get("density", 1.0))
        c = float(props.get("hardening_rate", 0.1))
        fail_se = float(props.get("failure_energy", props.get("yield_stress", 10000.0)))

        if mat_type == "NEOHOOK" and "shear_modulus" in props and "bulk_modulus" in props:
            shear = float(props.get("shear_modulus", 0.0))
            bulk = float(props.get("bulk_modulus", 0.0))
            denom = (3.0 * bulk + shear)
            if denom:
                youngs = 9.0 * bulk * shear / denom
                nu = (3.0 * bulk - 2.0 * shear) / (2.0 * denom)
            else:
                youngs = float(props.get("youngs_modulus", 100.0))
                nu = float(props.get("poisson_ratio", 0.3))
        elif mat_type == "RIGID":
            youngs = float(props.get("youngs_modulus", 1e9))
            nu = float(props.get("poisson_ratio", 0.3))
            fail_se = max(fail_se, 1e9)
        else:
            youngs = float(props.get("youngs_modulus", 100.0))
            nu = float(props.get("poisson_ratio", 0.3))

        return {
            "E": youngs,
            "Nu": nu,
            "rho": rho,
            "fail_SE": fail_se,
            "c": c,
        }

    def _apply_simulation_settings(self):
        config = self._read_sim_config()
        sim = config.get("simulation", {})
        total_steps = int(self.total_steps_spin.value())

        if self.auto_dt_checkbox.isChecked():
            self._recompute_dt(silent=True)
        time_step = float(self.time_step_spin.value())
        write_every_steps = max(1, int(float(self._refresh_percent) * total_steps / 100.0))
        sim["time_step"] = time_step
        sim["total_steps"] = total_steps
        sim["write_every_steps"] = write_every_steps
        sim["gravity"] = float(self.g_spin.value())
        sim["device"] = self.device_combo.currentText().strip().lower()
        config["simulation"] = sim
        config.pop("material", None)

        self._total_steps = total_steps
        self._progress_total_steps = total_steps
        self._update_sim_labels()

        if self._write_sim_config(config):
            self.log_output(
                f"Simulation settings applied: dt={time_step:.6g}, total_steps={total_steps}, "
                f"write_every_steps={write_every_steps}"
            )
            self._sync_solver_settings_state()
        else:
            self.log_output("Warning: could not update CPD-main/config.yml.")


class ResultsPanel(QWidget):
    def __init__(self, sketch_view, parent=None, project_state=None):
        super().__init__(parent)
        self.sketch_view = sketch_view
        self.project_state = _resolve_project_state(sketch_view, project_state)
        self._frame_count = 0
        self._playing = False
        self._results_label = ""
        self._results_controller = None
        self._load_label_full = "Load"
        self._load_saved_label_full = "Load Saved"
        self._save_label_full = "Save"
        self._clear_label_full = "Clear"
        self._compare_label_full = "Apply"
        self._history_selection_required = True
        self._history_selection_has_data = False
        self._history_selection_label = ""
        self._history_selection_scope = "node"

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_content = QWidget()
        scroll_content.setMinimumWidth(0)
        scroll_content.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        scroll_area.setWidget(scroll_content)
        hbar = scroll_area.horizontalScrollBar()
        hbar.setRange(0, 0)
        hbar.valueChanged.connect(lambda _v: hbar.setValue(0))
        outer_layout.addWidget(scroll_area, 1)
        layout = QVBoxLayout(scroll_content)
        _apply_layout_metrics(layout)
        self._results_outer_layout = outer_layout
        self._results_scroll_area = scroll_area

        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        title_label = QLabel("Results")
        title_label.setObjectName("SectionTitleLabel")
        title_row.addWidget(title_label)
        title_row.addStretch(1)
        self.results_status_label = QLabel("No results")
        self.results_status_label.setObjectName("SummaryLabel")
        title_row.addWidget(self.results_status_label)
        layout.addLayout(title_row)

        mode_row = QHBoxLayout()
        _apply_layout_metrics(mode_row, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        mode_label = QLabel("Mode")
        mode_label.setObjectName("SummaryLabel")
        mode_row.addWidget(mode_label)
        self.mode_default_btn = QPushButton("Default")
        self.mode_default_btn.setCheckable(True)
        self.mode_default_btn.setChecked(True)
        self.mode_default_btn.clicked.connect(lambda: self._set_mode("default"))
        mode_row.addWidget(self.mode_default_btn)
        self.mode_custom_btn = QPushButton("Custom")
        self.mode_custom_btn.setCheckable(True)
        self.mode_custom_btn.setToolTip("Refine the mesh inside user-drawn zones, then regenerate")
        self.mode_custom_btn.clicked.connect(lambda: self._set_mode("custom"))
        mode_row.addWidget(self.mode_custom_btn)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)
        self._mode = "default"

        content_row = QVBoxLayout()
        _apply_layout_metrics(content_row, margins=(0, 0, 0, 0), spacing=DOCK_SECTION_SPACING)
        layout.addLayout(content_row)

        controls_card = QFrame()
        controls_card.setFrameShape(QFrame.StyledPanel)
        controls_card.setProperty("card", True)
        controls_layout = QVBoxLayout(controls_card)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(8)
        controls_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        content_row.addWidget(controls_card, 0)

        action_row_1 = QHBoxLayout()
        _apply_layout_metrics(action_row_1, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        self.load_button = QPushButton("Load Results")
        self.load_button.setProperty("primary", True)
        self.load_button.clicked.connect(self._load_results)
        self.load_button.setIcon(_style_icon(self, "SP_BrowserReload", "frame", ("view-refresh", "document-open-recent")))
        self.load_button.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.load_button.setToolTip(f"Load latest results from {WORKSPACE_DIR_NAME}/ and start animation")
        self.load_button.setText("Load")
        action_row_1.addWidget(self.load_button)
        self.load_saved_button = QPushButton("Load Saved")
        self.load_saved_button.clicked.connect(self._load_saved_results)
        self.load_saved_button.setIcon(_style_icon(self, "SP_DialogOpenButton", "open", ("document-open", "folder-open")))
        self.load_saved_button.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.load_saved_button.setToolTip(
            f"Load saved results from {WORKSPACE_DIR_NAME}/saved_results or another folder"
        )
        self.load_saved_button.setText("Load Saved")
        action_row_1.addWidget(self.load_saved_button)
        action_row_1.addStretch(1)
        controls_layout.addLayout(action_row_1)

        action_row_2 = QHBoxLayout()
        _apply_layout_metrics(action_row_2, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        self.save_button = QPushButton("Save Results As")
        self.save_button.clicked.connect(self._save_results_as)
        self.save_button.setIcon(_style_icon(self, "SP_DialogSaveButton", "save", ("document-save", "document-save-as")))
        self.save_button.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.save_button.setToolTip(
            f"Save current results snapshot from {WORKSPACE_DIR_NAME}/saved_results"
        )
        self.save_button.setText("Save")
        action_row_2.addWidget(self.save_button)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self._clear_results)
        self.clear_button.setIcon(_style_icon(self, "SP_DialogCloseButton", "stop", ("edit-clear",)))
        self.clear_button.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.clear_button.setToolTip("Stop animation and clear frames")
        self.clear_button.setText("Clear")
        action_row_2.addWidget(self.clear_button)
        action_row_2.addStretch(1)
        controls_layout.addLayout(action_row_2)

        playback_row = QHBoxLayout()
        _apply_layout_metrics(playback_row, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        self.first_frame_button = QPushButton()
        self.first_frame_button.clicked.connect(lambda: self._step_replay("first_animation_frame"))
        self.first_frame_button.setEnabled(False)
        self.first_frame_button.setIcon(_style_icon(self, "SP_MediaSkipBackward", "rewind", ("media-skip-backward",)))
        self.first_frame_button.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.first_frame_button.setToolTip("First frame")
        playback_row.addWidget(self.first_frame_button)
        self.prev_frame_button = QPushButton()
        self.prev_frame_button.clicked.connect(lambda: self._step_replay("previous_animation_frame"))
        self.prev_frame_button.setEnabled(False)
        self.prev_frame_button.setIcon(_style_icon(self, "SP_MediaSeekBackward", "back", ("media-seek-backward",)))
        self.prev_frame_button.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.prev_frame_button.setToolTip("Previous frame")
        playback_row.addWidget(self.prev_frame_button)
        self._play_icon = _style_icon(self, "SP_MediaPlay", "play", ("media-playback-start",))
        self._pause_icon = _style_icon(self, "SP_MediaPause", "pause", ("media-playback-pause",))
        self.play_pause_button = QPushButton("Play")
        self.play_pause_button.setProperty("primary", True)
        self.play_pause_button.clicked.connect(self.toggle_playback)
        self.play_pause_button.setEnabled(False)
        self.play_pause_button.setIcon(self._play_icon)
        self.play_pause_button.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.play_pause_button.setToolTip("Play Animation")
        playback_row.addWidget(self.play_pause_button)
        self.play_button = self.play_pause_button
        self.next_frame_button = QPushButton()
        self.next_frame_button.clicked.connect(lambda: self._step_replay("next_animation_frame"))
        self.next_frame_button.setEnabled(False)
        self.next_frame_button.setIcon(_style_icon(self, "SP_MediaSeekForward", "forward", ("media-seek-forward",)))
        self.next_frame_button.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.next_frame_button.setToolTip("Next frame")
        playback_row.addWidget(self.next_frame_button)
        self.last_frame_button = QPushButton()
        self.last_frame_button.clicked.connect(lambda: self._step_replay("last_animation_frame"))
        self.last_frame_button.setEnabled(False)
        self.last_frame_button.setIcon(_style_icon(self, "SP_MediaSkipForward", "skip", ("media-skip-forward",)))
        self.last_frame_button.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.last_frame_button.setToolTip("Last frame")
        playback_row.addWidget(self.last_frame_button)
        playback_row.addStretch(1)
        self.frame_label = QLabel("Frame: 0/0")
        self.frame_label.setObjectName("SummaryLabel")
        playback_row.addWidget(self.frame_label)
        controls_layout.addLayout(playback_row)

        compare_form = QFormLayout()
        _apply_layout_metrics(compare_form, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        color_by_label = QLabel("Result")
        color_by_label.setObjectName("SummaryLabel")
        self.compare_mode_combo = QComboBox()
        self.compare_mode_combo.addItem("None", "none")
        self.compare_mode_combo.setEnabled(False)
        self.compare_mode_combo.setMinimumWidth(60)
        self.compare_mode_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.compare_mode_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.compare_mode_combo.setMinimumContentsLength(8)
        self.compare_button = QPushButton("Apply")
        self.compare_button.setProperty("primary", True)
        self.compare_button.clicked.connect(self._apply_selected_compare_field)
        self.compare_button.setEnabled(False)
        self.compare_button.setIcon(_style_icon(self, "SP_FileDialogContentsView", "frame", ("view-statistics",)))
        self.compare_button.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.compare_button.setToolTip("Apply selected animated result field")
        self.compare_button.setText("Apply")
        self.compare_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.compare_button.setMinimumWidth(DOCK_ICON_BTN_MIN)
        self.compare_button.setMinimumHeight(34)
        compare_form.addRow(color_by_label, self.compare_mode_combo)
        controls_layout.addLayout(compare_form)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self._on_frame_slider)
        controls_layout.addWidget(self.frame_slider)

        # Default-mode mirror of the Custom-mode "Show vertex dots" toggle —
        # same control, same behavior, kept in sync so the two modes never
        # disagree about whether dots are visible.
        self.default_show_dots = QCheckBox("Show vertex dots")
        self.default_show_dots.setChecked(False)
        self.default_show_dots.setToolTip(
            "Toggle particle/vertex markers on the mesh. "
            "Off = clean triangle edges only."
        )
        self.default_show_dots.toggled.connect(self._on_default_dots_toggled)
        controls_layout.addWidget(self.default_show_dots)

        info_card = QFrame()
        info_card.setFrameShape(QFrame.StyledPanel)
        info_card.setProperty("card", True)
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(10, 10, 10, 10)
        info_layout.setSpacing(8)
        info_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content_row.addWidget(info_card, 1)

        custom_card = QFrame()
        custom_card.setFrameShape(QFrame.StyledPanel)
        custom_card.setProperty("card", True)
        custom_layout = QVBoxLayout(custom_card)
        custom_layout.setContentsMargins(10, 10, 10, 10)
        custom_layout.setSpacing(8)
        custom_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        custom_card.setVisible(False)
        content_row.addWidget(custom_card, 1)
        self._custom_mode_card = custom_card

        custom_title = QLabel("Mesh Customization")
        custom_title.setObjectName("SectionTitleLabel")
        custom_layout.addWidget(custom_title)

        custom_hint = QLabel(
            "Draw polygon zones on the canvas to locally refine the mesh. "
            "Click points to add vertices; double-click or press Enter to close the polygon."
        )
        custom_hint.setObjectName("MinorStatusLabel")
        custom_hint.setWordWrap(True)
        custom_layout.addWidget(custom_hint)

        custom_btn_row = QHBoxLayout()
        _apply_layout_metrics(custom_btn_row, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        self.zone_select_btn = QPushButton("Select Area")
        self.zone_select_btn.setProperty("primary", True)
        self.zone_select_btn.setToolTip("Start drawing a polygon zone on the canvas")
        self.zone_select_btn.clicked.connect(self._on_select_zone_clicked)
        custom_btn_row.addWidget(self.zone_select_btn)
        self.zone_clear_all_btn = QPushButton("Clear All")
        self.zone_clear_all_btn.setToolTip("Remove all custom refinement zones")
        self.zone_clear_all_btn.clicked.connect(self._on_clear_all_zones_clicked)
        custom_btn_row.addWidget(self.zone_clear_all_btn)
        custom_btn_row.addStretch(1)
        custom_layout.addLayout(custom_btn_row)

        # Vertex dots toggle — default off so the regenerated mesh shows only
        # triangle edges (matches the gmsh sample aesthetic).
        self.zone_show_dots = QCheckBox("Show vertex dots")
        self.zone_show_dots.setChecked(False)
        self.zone_show_dots.setToolTip(
            "Toggle particle/vertex markers on the regenerated mesh. "
            "Off = clean triangle edges only."
        )
        self.zone_show_dots.toggled.connect(self._on_zone_dots_toggled)
        custom_layout.addWidget(self.zone_show_dots)

        self.zone_list_container = QWidget()
        self.zone_list_layout = QVBoxLayout(self.zone_list_container)
        _apply_layout_metrics(self.zone_list_layout, margins=(0, 0, 0, 0), spacing=4)
        custom_layout.addWidget(self.zone_list_container)

        self.zone_empty_label = QLabel("No zones defined. Click 'Select Area' to draw one.")
        self.zone_empty_label.setObjectName("MinorStatusLabel")
        self.zone_empty_label.setWordWrap(True)
        custom_layout.addWidget(self.zone_empty_label)

        custom_layout.addStretch(1)

        self.zone_regenerate_btn = QPushButton("OK — Regenerate Mesh")
        self.zone_regenerate_btn.setProperty("primary", True)
        self.zone_regenerate_btn.setToolTip("Apply zones and rebuild the mesh")
        self.zone_regenerate_btn.clicked.connect(self._on_regenerate_with_zones_clicked)
        custom_layout.addWidget(self.zone_regenerate_btn)

        self.zone_run_sim_btn = QPushButton("Run Simulation with New Mesh")
        self.zone_run_sim_btn.setToolTip("Run the solver against the regenerated mesh")
        self.zone_run_sim_btn.clicked.connect(self._on_run_sim_after_regen_clicked)
        self.zone_run_sim_btn.setVisible(False)
        custom_layout.addWidget(self.zone_run_sim_btn)

        self.compare_summary_label = QLabel("Field: --")
        self.compare_summary_label.setObjectName("SummaryLabel")
        self.compare_summary_label.setWordWrap(True)
        info_layout.addWidget(self.compare_summary_label)

        activity_row = QVBoxLayout()
        activity_row.setSpacing(4)
        self.disp_activity_label = QLabel("Disp: --")
        self.disp_activity_label.setObjectName("MinorStatusLabel")
        self.stress_activity_label = QLabel("Stress: --")
        self.stress_activity_label.setObjectName("MinorStatusLabel")
        self.strain_activity_label = QLabel("Strain: --")
        self.strain_activity_label.setObjectName("MinorStatusLabel")
        activity_row.addWidget(self.disp_activity_label)
        activity_row.addWidget(self.stress_activity_label)
        activity_row.addWidget(self.strain_activity_label)
        info_layout.addLayout(activity_row)

        curve_group = QGroupBox("Response Plot")
        curve_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        curve_layout = QVBoxLayout(curve_group)
        _apply_layout_metrics(curve_layout, margins=(8, 8, 8, 8), spacing=DOCK_ROW_SPACING)
        quantity_row = QHBoxLayout()
        _apply_layout_metrics(quantity_row, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        quantity_label = QLabel("Quantity")
        quantity_label.setObjectName("SummaryLabel")
        quantity_row.addWidget(quantity_label)
        self.curve_quantity_combo = QComboBox()
        self.curve_quantity_combo.currentIndexChanged.connect(self._on_response_quantity_changed)
        quantity_row.addWidget(self.curve_quantity_combo, 1)
        curve_layout.addLayout(quantity_row)
        curve_controls = QHBoxLayout()
        _apply_layout_metrics(curve_controls, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        curve_label = QLabel("Component")
        curve_label.setObjectName("SummaryLabel")
        curve_controls.addWidget(curve_label)
        self.curve_mode_combo = QComboBox()
        self.curve_mode_combo.currentIndexChanged.connect(self._on_response_subtype_changed)
        self.curve_mode_combo.setEnabled(False)
        curve_controls.addWidget(self.curve_mode_combo, 1)
        curve_layout.addLayout(curve_controls)
        scope_row = QHBoxLayout()
        _apply_layout_metrics(scope_row, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        scope_label = QLabel("Scope")
        scope_label.setObjectName("SummaryLabel")
        scope_row.addWidget(scope_label)
        self.curve_scope_combo = QComboBox()
        self.curve_scope_combo.currentIndexChanged.connect(self._on_history_scope_changed)
        scope_row.addWidget(self.curve_scope_combo, 1)
        curve_layout.addLayout(scope_row)
        self.curve_scope_hint_label = QLabel("Load results to configure a response plot.")
        self.curve_scope_hint_label.setObjectName("MinorStatusLabel")
        self.curve_scope_hint_label.setWordWrap(True)
        curve_layout.addWidget(self.curve_scope_hint_label)
        self.curve_value_label = QLabel("Load results to show a response history.")
        self.curve_value_label.setObjectName("MinorStatusLabel")
        self.curve_value_label.setWordWrap(True)
        curve_layout.addWidget(self.curve_value_label)
        self._stress_strain_placeholder = QLabel(
            "A time-history response plot will appear here after loading results."
        )
        self._stress_strain_placeholder.setObjectName("MinorStatusLabel")
        self._stress_strain_placeholder.setAlignment(Qt.AlignCenter)
        self._stress_strain_placeholder.setWordWrap(True)
        self._stress_strain_placeholder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._stress_strain_placeholder.setMinimumHeight(110)
        self._stress_strain_placeholder.setMaximumHeight(140)
        curve_layout.addWidget(self._stress_strain_placeholder, 1)
        self._stress_strain_canvas = None
        self._stress_strain_figure = None
        self._stress_strain_axes = None
        if FigureCanvas is not None and Figure is not None:
            self._stress_strain_figure = Figure(figsize=(4.2, 1.75), constrained_layout=True)
            self._stress_strain_canvas = FigureCanvas(self._stress_strain_figure)
            self._stress_strain_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self._stress_strain_canvas.setMinimumHeight(155)
            self._stress_strain_canvas.setMaximumHeight(195)
            self._stress_strain_canvas.setVisible(False)
            self._stress_strain_axes = self._stress_strain_figure.add_subplot(111)
            curve_layout.addWidget(self._stress_strain_canvas, 0)
        curve_group.setMaximumHeight(330)
        info_layout.addWidget(curve_group, 0)

        display_layout = QHBoxLayout()
        display_layout.setSpacing(8)
        display_label = QLabel("Show")
        display_label.setObjectName("SummaryLabel")
        display_layout.addWidget(display_label)
        self.show_nodes_checkbox = QCheckBox("Particles")
        self.show_nodes_checkbox.setChecked(True)
        self.show_nodes_checkbox.toggled.connect(
            lambda checked: self.sketch_view.set_replay_debug_overlays(show_particles=checked)
        )
        display_layout.addWidget(self.show_nodes_checkbox)
        self.show_mesh_checkbox = QCheckBox("Connections")
        self.show_mesh_checkbox.setChecked(True)
        self.show_mesh_checkbox.toggled.connect(
            lambda checked: self.sketch_view.set_replay_debug_overlays(show_connections=checked)
        )
        display_layout.addWidget(self.show_mesh_checkbox)
        self.element_alpha_slider = QSlider(Qt.Horizontal)
        self.element_alpha_slider.setRange(10, 100)
        self.element_alpha_slider.setValue(60)
        self.element_alpha_slider.setToolTip("Element transparency")
        self.element_alpha_slider.valueChanged.connect(self._on_element_alpha_changed)
        display_layout.addWidget(QLabel("Element Alpha"))
        display_layout.addWidget(self.element_alpha_slider)
        self.show_bc_checkbox = QCheckBox("BC")
        self.show_bc_checkbox.setChecked(True)
        self.show_bc_checkbox.toggled.connect(
            lambda checked: self.sketch_view.set_replay_debug_overlays(show_bc_markers=checked)
        )
        display_layout.addWidget(self.show_bc_checkbox)
        self.show_load_checkbox = QCheckBox("Loads")
        self.show_load_checkbox.setChecked(True)
        self.show_load_checkbox.toggled.connect(
            lambda checked: self.sketch_view.set_replay_debug_overlays(show_load_vectors=checked)
        )
        display_layout.addWidget(self.show_load_checkbox)
        self.show_ids_checkbox = QCheckBox("IDs")
        self.show_ids_checkbox.setChecked(False)
        self.show_ids_checkbox.toggled.connect(
            lambda checked: self.sketch_view.set_replay_debug_overlays(show_particle_ids=checked)
        )
        display_layout.addWidget(self.show_ids_checkbox)
        display_layout.addStretch(1)
        info_layout.addLayout(display_layout)

        inspector_group = QGroupBox("Frame Inspector")
        inspector_layout = QFormLayout(inspector_group)
        _apply_layout_metrics(inspector_layout, margins=(8, 8, 8, 8), spacing=DOCK_ROW_SPACING)
        self.inspect_particle_id = QLabel("--")
        self.inspect_position = QLabel("--")
        self.inspect_velocity = QLabel("--")
        self.inspect_material = QLabel("--")
        self.inspect_bc = QLabel("--")
        for label_widget in (
            self.inspect_particle_id,
            self.inspect_position,
            self.inspect_velocity,
            self.inspect_material,
            self.inspect_bc,
        ):
            label_widget.setObjectName("MinorStatusLabel")
            label_widget.setWordWrap(True)
        inspector_layout.addRow("Particle", self.inspect_particle_id)
        inspector_layout.addRow("Position", self.inspect_position)
        inspector_layout.addRow("Velocity", self.inspect_velocity)
        inspector_layout.addRow("Material", self.inspect_material)
        inspector_layout.addRow("BC", self.inspect_bc)
        info_layout.addWidget(inspector_group)

        self.sketch_view.animationFramesLoaded.connect(self._on_frames_loaded)
        self.sketch_view.animationFrameChanged.connect(self._on_frame_changed)
        if hasattr(self.sketch_view, "animationPlaybackStateChanged"):
            try:
                self.sketch_view.animationPlaybackStateChanged.connect(self._on_animation_playback_state_changed)
            except Exception:
                pass
        self.sketch_view.replayParticleSelected.connect(self._on_replay_particle_selected)
        if hasattr(self.sketch_view, "replayScopeSelectionChanged"):
            try:
                self.sketch_view.replayScopeSelectionChanged.connect(self._on_results_selection_changed)
            except Exception:
                pass
        view_3d = self._get_view_3d()
        if view_3d is not None and hasattr(view_3d, "selectionChanged"):
            try:
                view_3d.selectionChanged.connect(self._on_results_selection_changed)
            except Exception:
                pass
        window_controller = getattr(self.window(), "results_controller", None)
        if window_controller is not None:
            self.set_results_controller(window_controller)

        self._results_sticky_footer = QFrame(self)
        self._results_sticky_footer.setObjectName("ResultsStickyFooter")
        self._results_sticky_footer.setFrameShape(QFrame.StyledPanel)
        sticky_layout = QVBoxLayout(self._results_sticky_footer)
        _apply_layout_metrics(sticky_layout, margins=(DOCK_MARGIN, 2, DOCK_MARGIN, 2), spacing=0)
        sticky_layout.addWidget(self.compare_button)
        outer_layout.addWidget(self._results_sticky_footer, 0)

        self._default_mode_widgets = [controls_card, info_card, self._results_sticky_footer]
        self._zone_row_widgets = []
        self._rebuild_zone_list()

        self._update_action_button_density()
        self._reset_particle_inspector()
        _finalize_dock_panel(self)

    def set_results_controller(self, controller):
        if controller is self._results_controller:
            self._refresh_result_field_options()
            self._refresh_history_curve_options()
            self._update_history_selection_from_view()
            self._refresh_stress_strain_plot(self.frame_slider.value() if self._frame_count > 0 else 0)
            return
        self._results_controller = controller
        if controller is None:
            self._refresh_history_curve_options()
            self._refresh_stress_strain_plot(0)
            return
        try:
            controller.frameLoadStarted.connect(self._on_frame_load_started)
            controller.frameLoadFailed.connect(self._on_frame_load_failed)
        except Exception:
            pass
        self._refresh_result_field_options()
        self._refresh_history_curve_options()
        self._on_history_scope_changed()
        self._update_history_selection_from_view()
        self._refresh_stress_strain_plot(0)

    def _response_quantity_options(self):
        controller = self._results_controller
        if controller is not None and hasattr(controller, "response_plot_quantities"):
            return list(controller.response_plot_quantities() or [])
        return []

    def _response_subtype_options(self, quantity):
        controller = self._results_controller
        if controller is not None and hasattr(controller, "response_plot_subtypes"):
            return list(controller.response_plot_subtypes(quantity) or [])
        return []

    def _response_scope_options(self, quantity):
        controller = self._results_controller
        if controller is not None and hasattr(controller, "response_plot_scopes"):
            return list(controller.response_plot_scopes(quantity) or [])
        return []

    def _current_response_quantity(self):
        controller = self._results_controller
        current = str(self.curve_quantity_combo.currentData() or "").strip().lower()
        options = self._response_quantity_options()
        if any(str(key).strip().lower() == current for key, _label in options):
            return current
        if controller is not None and hasattr(controller, "default_response_plot_quantity"):
            return str(controller.default_response_plot_quantity() or "").strip().lower()
        return str(options[0][0]).strip().lower() if options else "displacement"

    def _current_response_subtype(self):
        controller = self._results_controller
        quantity = self._current_response_quantity()
        current = str(self.curve_mode_combo.currentData() or "").strip().lower()
        options = self._response_subtype_options(quantity)
        if any(str(key).strip().lower() == current for key, _label in options):
            return current
        if controller is not None and hasattr(controller, "default_response_plot_subtype"):
            return str(controller.default_response_plot_subtype(quantity) or "").strip().lower()
        return str(options[0][0]).strip().lower() if options else ""

    def _current_response_scope(self):
        controller = self._results_controller
        quantity = self._current_response_quantity()
        current = str(self.curve_scope_combo.currentData() or "").strip().lower()
        options = self._response_scope_options(quantity)
        if any(str(key).strip().lower() == current for key, _label in options):
            return current
        if controller is not None and hasattr(controller, "default_response_plot_scope"):
            return str(controller.default_response_plot_scope(quantity) or "").strip().lower()
        return str(options[0][0]).strip().lower() if options else "node"

    def _response_scope_label(self, scope):
        controller = self._results_controller
        if controller is not None and hasattr(controller, "RESPONSE_SCOPE_LABELS"):
            label = controller.RESPONSE_SCOPE_LABELS.get(str(scope or "").strip().lower())
            if label:
                return str(label)
        text = str(scope or "").strip().replace("_", " ")
        return text.title() if text else "Target"

    def _response_quantity_label(self, quantity):
        for key, label in self._response_quantity_options():
            if str(key).strip().lower() == str(quantity or "").strip().lower():
                return str(label)
        return str(quantity or "Response").replace("_", " ").title()

    def _response_pick_mode_for_scope(self, scope):
        scope_key = str(scope or "").strip().lower()
        mapping = {
            "node": "node",
            "geometry_edge": "geometry_edge",
            "triangle": "triangle",
            "bc_target": "bc_target",
        }
        return mapping.get(scope_key, "node")

    def _response_3d_target_for_scope(self, scope):
        scope_key = str(scope or "").strip().lower()
        mapping = {
            "node": "point",
            "geometry_edge": "edge",
            "triangle": "face",
            "bc_target": "none",
        }
        return mapping.get(scope_key, "none")

    def _base_results_status_text(self):
        if self._frame_count <= 0:
            return "No results"
        label = f"{self._frame_count} frames"
        if self._results_label:
            label = f"{label} ({self._results_label})"
        return label

    def _set_button_caption(self, button, *, full, compact, icon_only):
        if button is None:
            return
        if icon_only:
            button.setText("")
        elif compact is not None:
            button.setText(compact)
        else:
            button.setText(full)

    def _actual_playback_state(self):
        try:
            return bool(self.sketch_view.animation_timer.isActive())
        except Exception:
            return bool(self._playing)

    def _set_playback_button_state(self, playing=None):
        self._playing = self._actual_playback_state() if playing is None else bool(playing)
        buttons_enabled = self._frame_count > 0
        self.play_pause_button.setEnabled(buttons_enabled)
        self.play_pause_button.setIcon(self._pause_icon if self._playing else self._play_icon)
        self.play_pause_button.setToolTip("Pause Animation" if self._playing else "Play Animation")
        self._update_action_button_density()

    def _update_action_button_density(self):
        width = int(self.width())
        compact = width < 430
        icon_only = width < 350
        self._set_button_caption(
            self.load_button,
            full=self._load_label_full,
            compact="Latest",
            icon_only=icon_only,
        )
        self._set_button_caption(
            self.load_saved_button,
            full=self._load_saved_label_full,
            compact="Saved",
            icon_only=icon_only,
        )
        self._set_button_caption(
            self.save_button,
            full=self._save_label_full,
            compact="Save",
            icon_only=icon_only,
        )
        self._set_button_caption(
            self.clear_button,
            full=self._clear_label_full,
            compact="Clear",
            icon_only=icon_only,
        )
        self._set_button_caption(
            self.play_pause_button,
            full="Pause" if self._playing else "Play",
            compact="Pause" if self._playing else "Play",
            icon_only=icon_only,
        )
        self._set_button_caption(
            self.compare_button,
            full=self._compare_label_full,
            compact="Apply",
            icon_only=icon_only,
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_action_button_density()

    def _on_animation_playback_state_changed(self, playing):
        self._set_playback_button_state(bool(playing))

    def _load_results(self):
        self._results_label = "latest"
        self.sketch_view.load_and_run_visualization()
        self._announce_status("Loading latest results...")

    def _load_saved_results(self):
        saved_root = _workspace_path("saved_results")
        os.makedirs(saved_root, exist_ok=True)
        selected = QFileDialog.getExistingDirectory(
            self,
            "Load Saved Results",
            saved_root,
        )
        if not selected:
            return
        label = os.path.basename(selected) or "saved"
        self._results_label = f"saved: {label}"
        self.sketch_view.load_and_run_visualization(results_root=selected)
        self._announce_status(f"Loading saved results: {label}")

    def _save_results_as(self):
        workspace_dir = _workspace_dir()
        results_src = _workspace_output_path("results")
        if not os.path.isdir(results_src):
            results_src = os.path.join(workspace_dir, "results")
        pos_history_src = _workspace_output_path("pos_history.npy")
        if not os.path.exists(pos_history_src):
            pos_history_src = os.path.join(workspace_dir, "pos_history.npy")
        if not os.path.isdir(results_src) and not os.path.exists(pos_history_src):
            QMessageBox.information(
                self,
                "No Results",
                f"No results found in {WORKSPACE_DIR_NAME}/ yet. Run a simulation first.",
            )
            return

        name, ok = QInputDialog.getText(
            self,
            "Save Results As",
            "Results name:",
        )
        if not ok:
            return
        name = (name or "").strip()
        if not name:
            name = time.strftime("results_%Y%m%d_%H%M%S")

        saved_root = os.path.join(workspace_dir, "saved_results")
        os.makedirs(saved_root, exist_ok=True)
        dest_dir = os.path.join(saved_root, name)
        if os.path.exists(dest_dir):
            QMessageBox.warning(self, "Save Results", "That results name already exists.")
            return
        os.makedirs(dest_dir, exist_ok=True)

        if os.path.isdir(results_src):
            shutil.copytree(results_src, os.path.join(dest_dir, "results"))
        if os.path.exists(pos_history_src):
            shutil.copy2(pos_history_src, os.path.join(dest_dir, "pos_history.npy"))
        for fname in ("displacement_history.npy", "strain_history.npy", "stress_history.npy"):
            src = _workspace_output_path(fname)
            if not os.path.exists(src):
                src = os.path.join(workspace_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dest_dir, fname))
        initial_pos = _workspace_output_path("initial_pos.csv")
        final_pos = _workspace_output_path("final_pos.csv")
        if not os.path.exists(initial_pos):
            initial_pos = os.path.join(workspace_dir, "initial_pos.csv")
        if not os.path.exists(final_pos):
            final_pos = os.path.join(workspace_dir, "final_pos.csv")
        if os.path.exists(initial_pos):
            shutil.copy2(initial_pos, os.path.join(dest_dir, "initial_pos.csv"))
        if os.path.exists(final_pos):
            shutil.copy2(final_pos, os.path.join(dest_dir, "final_pos.csv"))

        QMessageBox.information(self, "Save Results", f"Results saved to: {dest_dir}")
        self._announce_status(f"Saved results snapshot: {name}")

    def _clear_results(self):
        self.sketch_view.stop_visualization()
        self.sketch_view.redraw()
        window = self.window()
        view_3d = getattr(window, "view_3d", None) if window is not None else None
        if view_3d is not None and hasattr(view_3d, "set_scalar_node_colors"):
            try:
                view_3d.set_scalar_node_colors(None)
            except Exception:
                pass
        self._results_label = ""
        self._set_results_state(0)
        self._announce_status("Cleared loaded results.")

    def _set_results_state(self, count):
        self._frame_count = int(count)
        if self._frame_count <= 0:
            self.frame_slider.setEnabled(False)
            self.play_pause_button.setEnabled(False)
            self.first_frame_button.setEnabled(False)
            self.prev_frame_button.setEnabled(False)
            self.next_frame_button.setEnabled(False)
            self.last_frame_button.setEnabled(False)
            self.compare_button.setEnabled(False)
            self.compare_mode_combo.setEnabled(False)
            self.curve_mode_combo.setEnabled(False)
            self.save_button.setEnabled(False)
            self._set_playback_button_state(False)
            self.frame_label.setText("Frame: 0/0")
            self.results_status_label.setText("No results")
            self.compare_summary_label.setText("Field: --")
            self.disp_activity_label.setText("Disp: --")
            self.stress_activity_label.setText("Stress: --")
            self.strain_activity_label.setText("Strain: --")
            window = self.window()
            view_3d = getattr(window, "view_3d", None) if window is not None else None
            if view_3d is not None and hasattr(view_3d, "set_scalar_node_colors"):
                try:
                    view_3d.set_scalar_node_colors(None)
                except Exception:
                    pass
            self._reset_particle_inspector()
            self._refresh_history_curve_options()
            self._refresh_stress_strain_plot(0)
            return
        self.frame_slider.setEnabled(True)
        self.first_frame_button.setEnabled(True)
        self.prev_frame_button.setEnabled(True)
        self.next_frame_button.setEnabled(True)
        self.last_frame_button.setEnabled(True)
        self.compare_button.setEnabled(True)
        self.compare_mode_combo.setEnabled(True)
        self.save_button.setEnabled(True)
        self._set_playback_button_state()
        self.results_status_label.setText(self._base_results_status_text())
        self._refresh_result_field_options()
        self._refresh_history_curve_options()
        self._refresh_stress_strain_plot(self.frame_slider.value())

    def toggle_playback(self):
        if self._frame_count <= 0:
            return
        if self._actual_playback_state():
            self._pause_replay()
        else:
            self._play_replay()

    def _play_replay(self):
        if self._frame_count <= 0:
            return
        self.sketch_view.set_animation_playing(True)
        self._set_playback_button_state()

    def _pause_replay(self):
        if self._frame_count <= 0:
            return
        self.sketch_view.set_animation_playing(False)
        self._set_playback_button_state()

    def _step_replay(self, method_name):
        if self._frame_count <= 0:
            return
        self._pause_replay()
        method = getattr(self.sketch_view, str(method_name), None)
        if callable(method):
            method()

    def _on_frames_loaded(self, count):
        self._set_results_state(count)
        if self._frame_count <= 0:
            return
        target_frame = 0
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, self._frame_count - 1)
        self.frame_slider.setValue(target_frame)
        self.frame_slider.setSingleStep(1)
        self.frame_slider.setPageStep(max(1, self._frame_count // 12))
        self.frame_slider.setTickInterval(max(1, self._frame_count // 12))
        self.frame_slider.blockSignals(False)
        self._set_playback_button_state(False)
        self._refresh_result_field_options()
        self._refresh_history_curve_options()
        preferred_field = "none"
        controller = self._results_controller
        if controller is not None:
            preferred_field = str(controller.active_result_field() or "").strip().lower()
            if not preferred_field or preferred_field == "none":
                preferred_field = str(controller.default_result_field() or "none").strip().lower()
        idx = self.compare_mode_combo.findData(preferred_field)
        if idx >= 0:
            self.compare_mode_combo.setCurrentIndex(idx)
        self.compare_summary_label.setText("Field: --")
        self.frame_label.setText(f"Frame: 1/{self._frame_count}")
        self._update_sample_activity()
        if hasattr(self, "show_mesh_checkbox"):
            self.show_mesh_checkbox.blockSignals(True)
            self.show_mesh_checkbox.setChecked(True)
            self.show_mesh_checkbox.blockSignals(False)
        if hasattr(self, "show_nodes_checkbox"):
            self.show_nodes_checkbox.blockSignals(True)
            self.show_nodes_checkbox.setChecked(True)
            self.show_nodes_checkbox.blockSignals(False)
        ensure_frame = getattr(self.sketch_view, "ensure_animation_frame_initialized", None)
        if callable(ensure_frame):
            try:
                ensure_frame(target_frame)
            except Exception:
                pass
        self.sketch_view.set_replay_debug_overlays(show_connections=True, show_particles=True)
        self._on_element_alpha_changed(self.element_alpha_slider.value() if hasattr(self, "element_alpha_slider") else 60)
        self._update_history_selection_from_view()
        self._refresh_stress_strain_plot(0)
        try:
            self._apply_selected_compare_field()
        except Exception:
            pass

    def _on_frame_changed(self, idx):
        if self._frame_count <= 0:
            return
        index = int(idx)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(index)
        self.frame_slider.blockSignals(False)
        self.frame_label.setText(f"Frame: {index + 1}/{self._frame_count}")
        self.results_status_label.setText(self._base_results_status_text())
        controller = self._results_controller
        if controller is not None:
            self.compare_summary_label.setText(controller.summarize_field(index))
        self._refresh_stress_strain_plot(index)

    def _on_frame_slider(self, value):
        if self._frame_count <= 0:
            return
        self.sketch_view.set_animation_playing(False)
        self._set_playback_button_state(False)
        self.sketch_view.set_animation_frame(int(value))

    def _format_vector_text(self, value):
        if value in (None, "", "--"):
            return "--"
        try:
            x_val, y_val = value[:2]
            return f"({float(x_val):.4g}, {float(y_val):.4g})"
        except Exception:
            return str(value)

    def _reset_particle_inspector(self):
        self.inspect_particle_id.setText("--")
        self.inspect_position.setText("--")
        self.inspect_velocity.setText("--")
        self.inspect_material.setText("--")
        self.inspect_bc.setText("--")

    def _on_replay_particle_selected(self, info):
        if not info:
            self._reset_particle_inspector()
            return
        self.inspect_particle_id.setText(str(info.get("particle_id", "--")))
        self.inspect_position.setText(self._format_vector_text(info.get("position")))
        self.inspect_velocity.setText(self._format_vector_text(info.get("velocity")))
        self.inspect_material.setText(str(info.get("material", "--") or "--"))
        self.inspect_bc.setText(str(info.get("bc", "--") or "--"))

    def _on_frame_load_started(self, index):
        if self._frame_count <= 0:
            return
        self.results_status_label.setText(f"Loading frame {int(index) + 1}/{self._frame_count}...")

    def _on_frame_load_failed(self, index, error_message):
        self.results_status_label.setText(self._base_results_status_text())
        self._announce_status(f"Frame {int(index) + 1} load failed: {error_message}")

    def _announce_status(self, message):
        window = self.window()
        if window is not None and hasattr(window, "statusBar"):
            window.statusBar().showMessage(str(message), 4000)

    def _on_response_quantity_changed(self):
        self._refresh_history_curve_options()
        self._on_history_scope_changed()
        frame_index = int(self.frame_slider.value()) if self._frame_count > 0 else 0
        self._refresh_stress_strain_plot(frame_index)

    def _on_response_subtype_changed(self):
        frame_index = int(self.frame_slider.value()) if self._frame_count > 0 else 0
        self._refresh_stress_strain_plot(frame_index)

    def _refresh_history_curve_options(self):
        controller = self._results_controller
        quantity_current = str(self.curve_quantity_combo.currentData() or "").strip().lower()
        subtype_current = str(self.curve_mode_combo.currentData() or "").strip().lower()
        scope_current = str(self.curve_scope_combo.currentData() or "").strip().lower()

        quantity_options = self._response_quantity_options()
        self.curve_quantity_combo.blockSignals(True)
        self.curve_quantity_combo.clear()
        for key, label in quantity_options:
            self.curve_quantity_combo.addItem(label, key)
        if controller is not None and quantity_options:
            preferred_quantity = (
                quantity_current
                if any(str(key).strip().lower() == quantity_current for key, _label in quantity_options)
                else self._current_response_quantity()
            )
            idx = self.curve_quantity_combo.findData(preferred_quantity)
            if idx >= 0:
                self.curve_quantity_combo.setCurrentIndex(idx)
        self.curve_quantity_combo.blockSignals(False)
        self.curve_quantity_combo.setEnabled(bool(quantity_options))

        quantity = self._current_response_quantity()
        subtype_options = self._response_subtype_options(quantity)
        self.curve_mode_combo.blockSignals(True)
        self.curve_mode_combo.clear()
        for key, label in subtype_options:
            self.curve_mode_combo.addItem(label, key)
        if subtype_options:
            preferred_subtype = (
                subtype_current
                if any(str(key).strip().lower() == subtype_current for key, _label in subtype_options)
                else self._current_response_subtype()
            )
            idx = self.curve_mode_combo.findData(preferred_subtype)
            if idx >= 0:
                self.curve_mode_combo.setCurrentIndex(idx)
        self.curve_mode_combo.blockSignals(False)
        self.curve_mode_combo.setEnabled(bool(subtype_options and self._frame_count > 0))

        scope_options = self._response_scope_options(quantity)
        self.curve_scope_combo.blockSignals(True)
        self.curve_scope_combo.clear()
        for key, label in scope_options:
            self.curve_scope_combo.addItem(label, key)
        if scope_options:
            preferred_scope = (
                scope_current
                if any(str(key).strip().lower() == scope_current for key, _label in scope_options)
                else self._current_response_scope()
            )
            idx = self.curve_scope_combo.findData(preferred_scope)
            if idx >= 0:
                self.curve_scope_combo.setCurrentIndex(idx)
        self.curve_scope_combo.blockSignals(False)
        self.curve_scope_combo.setEnabled(bool(scope_options))

        hint_text = (
            controller.response_plot_hint(quantity)
            if controller is not None and hasattr(controller, "response_plot_hint")
            else "Load results to configure a response plot."
        )
        self.curve_scope_hint_label.setText(str(hint_text or ""))

    def _refresh_stress_strain_plot(self, frame_index=0):
        controller = self._results_controller
        quantity = self._current_response_quantity()
        quantity_label = self._response_quantity_label(quantity)
        scope_label = self._response_scope_label(self._history_selection_scope).lower()
        self.curve_mode_combo.setEnabled(bool(self.curve_mode_combo.count() > 0 and self._frame_count > 0))
        if controller is None:
            self.curve_value_label.setText("Load results to show a response history.")
            self._stress_strain_placeholder.setText(
                "A time-history response plot will appear here after loading results."
            )
            self._stress_strain_placeholder.setVisible(True)
            if self._stress_strain_canvas is not None:
                self._stress_strain_canvas.setVisible(False)
            return
        if self._history_selection_required and not self._history_selection_has_data:
            self.curve_value_label.setText(f"Select a {scope_label} to view {quantity_label.lower()} history.")
            self._stress_strain_placeholder.setText("The response plot updates from the current scope selection.")
            self._stress_strain_placeholder.setVisible(True)
            if self._stress_strain_canvas is not None:
                self._stress_strain_canvas.setVisible(False)
            return

        subtype = self._current_response_subtype()
        curve = controller.response_curve(quantity, subtype)
        available = bool(curve.get("available"))
        if not available:
            message = str(curve.get("message") or "").strip()
            self.curve_value_label.setText(message or "No compatible response history is available.")
            self._stress_strain_placeholder.setText(message or "This results set does not include compatible history outputs yet.")
            self._stress_strain_placeholder.setVisible(True)
            if self._stress_strain_canvas is not None:
                self._stress_strain_canvas.setVisible(False)
            return

        x_vals = np.asarray(curve.get("x", []), dtype=float).reshape(-1)
        y_vals = np.asarray(curve.get("y", []), dtype=float).reshape(-1)
        finite = np.isfinite(x_vals) & np.isfinite(y_vals)
        if x_vals.size == 0 or y_vals.size == 0 or not np.any(finite):
            self.curve_value_label.setText("Response data is present but could not be plotted.")
            self._stress_strain_placeholder.setText("No finite response samples are available to draw.")
            self._stress_strain_placeholder.setVisible(True)
            if self._stress_strain_canvas is not None:
                self._stress_strain_canvas.setVisible(False)
            return

        current_index = max(0, min(int(frame_index), len(x_vals) - 1))
        if bool(finite[current_index]):
            y_label = str(curve.get("y_label", "Value")).strip()
            self.curve_value_label.setText(
                f"Current frame: time={float(x_vals[current_index]):.4g} s, "
                f"{y_label}={float(y_vals[current_index]):.4g}"
            )
        else:
            self.curve_value_label.setText("Current frame does not have a finite response sample.")

        if self._stress_strain_canvas is None or self._stress_strain_axes is None:
            self._stress_strain_placeholder.setText("Matplotlib is not available. Response plot is disabled.")
            self._stress_strain_placeholder.setVisible(True)
            return

        self._stress_strain_placeholder.setVisible(False)
        self._stress_strain_canvas.setVisible(True)
        ax = self._stress_strain_axes
        ax.clear()
        ax.plot(x_vals[finite], y_vals[finite], color="#185a9d", linewidth=2.0)
        if bool(finite[current_index]):
            marker_x = float(x_vals[current_index])
            marker_y = float(y_vals[current_index])
            ax.scatter(
                [marker_x],
                [marker_y],
                s=42,
                color="#f08a24",
                edgecolors="white",
                linewidths=0.6,
                zorder=4,
            )
            ax.axvline(marker_x, color="#f08a24", alpha=0.18, linewidth=1.0)
            ax.axhline(marker_y, color="#f08a24", alpha=0.18, linewidth=1.0)
        ax.set_title(str(curve.get("title", "Response vs Time")), fontsize=9.5)
        ax.set_xlabel(str(curve.get("x_label", "Time (s)")), fontsize=8.5)
        ax.set_ylabel(str(curve.get("y_label", "Value")), fontsize=8.5)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.28)
        for spine in ax.spines.values():
            spine.set_color("#9aa4b2")
        ax.tick_params(axis="both", labelsize=8)
        self._stress_strain_canvas.draw_idle()

    def _get_view_3d(self):
        window = self.window()
        return getattr(window, "view_3d", None) if window is not None else None

    def _on_results_selection_changed(self):
        self._update_history_selection_from_view()
        frame_index = int(self.frame_slider.value()) if self._frame_count > 0 else 0
        self._refresh_stress_strain_plot(frame_index)

    def _on_history_scope_changed(self):
        scope = self._current_response_scope()
        self._history_selection_scope = scope
        try:
            self._pause_replay()
        except Exception:
            pass
        try:
            self.sketch_view.set_replay_pick_mode(self._response_pick_mode_for_scope(scope))
        except Exception:
            pass
        window = self.window()
        if window is not None and hasattr(window, "_set_3d_selection_target"):
            try:
                window._set_3d_selection_target(self._response_3d_target_for_scope(scope))
            except Exception:
                pass
        view_3d = self._get_view_3d()
        if view_3d is not None and hasattr(view_3d, "clear_selection"):
            try:
                view_3d.clear_selection()
                if hasattr(view_3d, "clear_node_selection"):
                    view_3d.clear_node_selection()
                if hasattr(view_3d, "clear_edge_selection"):
                    view_3d.clear_edge_selection()
            except Exception:
                pass
        self._update_history_selection_from_view()

    def _selection_mapping_tolerance(self):
        return max(1.0, float(DEFAULT_DX) * 0.6)

    def _normalized_segment_key(self, segment):
        helper = getattr(self.sketch_view, "_segment_key", None)
        if callable(helper):
            try:
                return helper(segment)
            except Exception:
                pass
        if not isinstance(segment, (list, tuple)) or len(segment) != 2:
            return None
        try:
            a = (float(segment[0][0]), float(segment[0][1]))
            b = (float(segment[1][0]), float(segment[1][1]))
        except Exception:
            return None
        if a > b:
            a, b = b, a
        return (round(a[0], 6), round(a[1], 6), round(b[0], 6), round(b[1], 6))

    def _map_geometry_edge_to_node_ids(self, edge):
        try:
            nodes = np.asarray(getattr(self.sketch_view, "global_nodes", []), dtype=float)
        except Exception:
            return []
        if nodes.ndim != 2 or nodes.shape[0] == 0 or nodes.shape[1] < 2:
            return []
        try:
            mapped = map_geometry_to_nodes(
                np.asarray(nodes[:, :2], dtype=float),
                [{"coords": edge}],
                self._selection_mapping_tolerance(),
            )
        except Exception:
            return []
        return sorted({int(item.get("node_id")) for item in mapped if item.get("node_id") is not None})

    def _resolve_entry_node_ids(self, entry):
        if entry is None:
            return []
        ids = entry.get("ids")
        if ids:
            return sorted({int(nid) for nid in ids if nid is not None})
        part_id = entry.get("part_id")
        if part_id is not None and hasattr(self.sketch_view, "_iter_replay_target_indices"):
            try:
                return sorted({int(nid) for nid in self.sketch_view._iter_replay_target_indices(entry)})
            except Exception:
                return []
        coords = entry.get("coords")
        if coords:
            try:
                nodes = np.asarray(getattr(self.sketch_view, "global_nodes", []), dtype=float)
            except Exception:
                nodes = np.array([])
            if nodes.ndim == 2 and nodes.shape[0] > 0 and nodes.shape[1] >= 2:
                try:
                    mapped = map_geometry_to_nodes(
                        np.asarray(nodes[:, :2], dtype=float),
                        [entry],
                        self._selection_mapping_tolerance(),
                    )
                except Exception:
                    mapped = []
                return sorted({int(item.get("node_id")) for item in mapped if item.get("node_id") is not None})
        return []

    def _build_force_load_matches_for_node_selection(self, node_ids):
        selected = {int(nid) for nid in list(node_ids or []) if nid is not None}
        if not selected:
            return []
        matches = []
        for index, entry in enumerate(list(getattr(self.project_state, "loads", []) or [])):
            if str(entry.get("type") or "").strip().lower() != "force":
                continue
            target_node_ids = self._resolve_entry_node_ids(entry)
            if not target_node_ids:
                continue
            target_set = {int(nid) for nid in target_node_ids}
            overlap = selected & target_set
            if not overlap:
                continue
            matches.append(
                {
                    "index": int(index),
                    "scale": float(len(overlap)) / float(max(1, len(target_set))),
                }
            )
        return matches

    def _build_force_load_matches_for_geometry_edge(self, edge, node_ids):
        edge_key = self._normalized_segment_key(edge)
        edge_node_ids = {int(nid) for nid in list(node_ids or []) if nid is not None}
        matches = []
        for index, entry in enumerate(list(getattr(self.project_state, "loads", []) or [])):
            if str(entry.get("type") or "").strip().lower() != "force":
                continue
            entry_key = self._normalized_segment_key(entry.get("coords"))
            if edge_key is not None and entry_key is not None and entry_key == edge_key:
                matches.append({"index": int(index), "scale": 1.0})
                continue
            if entry.get("part_id") is not None:
                continue
            target_node_ids = {int(nid) for nid in self._resolve_entry_node_ids(entry)}
            if edge_node_ids and target_node_ids and target_node_ids == edge_node_ids:
                matches.append({"index": int(index), "scale": 1.0})
        seen = set()
        deduped = []
        for match in matches:
            idx = int(match.get("index", -1))
            if idx in seen:
                continue
            seen.add(idx)
            deduped.append(match)
        return deduped

    def _is_results_2d_active(self):
        return (
            getattr(self.sketch_view, "display_mode", "") == "results"
            and getattr(self.sketch_view, "is_visualization_mode", False)
        )

    def _update_history_selection_from_view(self):
        scope = self._current_response_scope()
        view_3d = self._get_view_3d()
        triangles = []
        nodes = []
        label = ""
        selection_payload = {"scope": scope}
        if self._is_results_2d_active():
            try:
                selection = self.sketch_view.get_replay_scope_selection()
            except Exception:
                selection = None
            if isinstance(selection, dict):
                if scope == "triangle":
                    triangles = sorted({int(idx) for idx in list(selection.get("triangles") or [])})
                    if triangles:
                        label = f"{len(triangles)} triangle(s)"
                elif scope == "node":
                    nodes = sorted({int(idx) for idx in list(selection.get("nodes") or [])})
                    if nodes:
                        label = f"{len(nodes)} node(s)"
                        selection_payload["load_matches"] = self._build_force_load_matches_for_node_selection(nodes)
                elif scope == "geometry_edge":
                    geometry_edges = list(selection.get("geometry_edges") or [])
                    if geometry_edges:
                        edge = geometry_edges[0]
                        nodes = self._map_geometry_edge_to_node_ids(edge)
                        label = "1 geometry edge"
                        selection_payload["geometry_edge"] = edge
                        selection_payload["load_matches"] = self._build_force_load_matches_for_geometry_edge(edge, nodes)
                elif scope == "bc_target":
                    bc_indices = sorted({int(idx) for idx in list(selection.get("bc_targets") or [])})
                    if bc_indices:
                        label = f"{len(bc_indices)} BC target(s)"
                        selection_payload["bc_indices"] = bc_indices
        elif view_3d is not None:
            if scope == "triangle":
                try:
                    triangles = sorted({int(idx) for idx in list(view_3d.get_selected_faces() or [])})
                except Exception:
                    triangles = []
                if triangles:
                    label = f"{len(triangles)} triangle(s)"
                    try:
                        nodes = sorted({int(nid) for nid in view_3d.get_selected_face_nodes()})
                    except Exception:
                        nodes = []
            elif scope == "node":
                if hasattr(view_3d, "get_selected_node_ids_for_mode"):
                    try:
                        nodes = sorted({int(nid) for nid in view_3d.get_selected_node_ids_for_mode("point")})
                    except Exception:
                        nodes = []
                else:
                    try:
                        nodes = sorted({int(nid) for nid in list(view_3d.get_selected_nodes() or [])})
                    except Exception:
                        nodes = []
                if nodes:
                    label = f"{len(nodes)} node(s)"
                    selection_payload["load_matches"] = self._build_force_load_matches_for_node_selection(nodes)
            elif scope == "geometry_edge":
                if hasattr(view_3d, "get_selected_node_ids_for_mode"):
                    try:
                        nodes = sorted({int(nid) for nid in view_3d.get_selected_node_ids_for_mode("edge")})
                    except Exception:
                        nodes = []
                try:
                    edge_count = int((view_3d.get_selection_counts() or {}).get("edges", 0))
                except Exception:
                    edge_count = 0
                if nodes or edge_count > 0:
                    label = f"{max(1, edge_count)} geometry edge(s)"
                    selection_payload["load_matches"] = self._build_force_load_matches_for_geometry_edge(None, nodes)
        selection_payload["node_ids"] = list(nodes)
        selection_payload["triangle_ids"] = list(triangles)
        selection_payload["label"] = label
        self._history_selection_has_data = bool(triangles or nodes or selection_payload.get("bc_indices"))
        self._history_selection_label = label
        controller = self._results_controller
        if controller is not None:
            controller.set_history_selection(
                active=self._history_selection_required,
                node_ids=nodes,
                triangle_ids=triangles,
                label=label,
                scope=scope,
                selection=selection_payload,
            )

    def _triangles_for_edges_2d(self, edges):
        elements = getattr(self.sketch_view, "global_elements", [])
        try:
            face_arr = np.asarray(elements, dtype=int)
        except Exception:
            return []
        if face_arr.ndim != 2 or face_arr.shape[1] < 3:
            return []
        edge_set = set()
        for edge in edges:
            try:
                a, b = int(edge[0]), int(edge[1])
            except Exception:
                continue
            edge_set.add((min(a, b), max(a, b)))
        if not edge_set:
            return []
        tri_ids = []
        for idx, tri in enumerate(face_arr):
            try:
                a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            except Exception:
                continue
            e1 = (min(a, b), max(a, b))
            e2 = (min(b, c), max(b, c))
            e3 = (min(c, a), max(c, a))
            if e1 in edge_set or e2 in edge_set or e3 in edge_set:
                tri_ids.append(int(idx))
        return tri_ids

    def _on_element_alpha_changed(self, value):
        try:
            alpha = max(0.1, min(1.0, float(value) / 100.0))
        except Exception:
            alpha = 0.6
        try:
            self.sketch_view.set_animation_element_alpha(alpha)
        except Exception:
            pass

    def _triangles_for_edges(self, view_3d, edges):
        faces = getattr(view_3d, "_last_faces", None)
        if faces is None:
            return []
        try:
            face_arr = np.asarray(faces, dtype=int)
        except Exception:
            return []
        if face_arr.ndim != 2 or face_arr.shape[1] < 3:
            return []
        edge_set = set()
        for edge in edges:
            try:
                a, b = int(edge[0]), int(edge[1])
            except Exception:
                continue
            edge_set.add((min(a, b), max(a, b)))
        if not edge_set:
            return []
        tri_ids = []
        for idx, tri in enumerate(face_arr):
            try:
                a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            except Exception:
                continue
            e1 = (min(a, b), max(a, b))
            e2 = (min(b, c), max(b, c))
            e3 = (min(c, a), max(c, a))
            if e1 in edge_set or e2 in edge_set or e3 in edge_set:
                tri_ids.append(int(idx))
        return tri_ids

    def reset_stage_state(self):
        self._set_playback_button_state(False)
        self._results_label = ""
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self._refresh_result_field_options()
        self._refresh_history_curve_options()
        idx = self.compare_mode_combo.findData("none")
        if idx >= 0:
            self.compare_mode_combo.setCurrentIndex(idx)
        self._set_results_state(0)
        self._reset_particle_inspector()
        self._refresh_stress_strain_plot(0)

    def sync_stage_state(self):
        frame_total = 0
        total_fn = getattr(self.sketch_view, "_animation_frame_total", None)
        if callable(total_fn):
            try:
                frame_total = int(total_fn())
            except Exception:
                frame_total = 0
        if frame_total <= 0:
            frames = getattr(self.sketch_view, "animation_frames", []) or []
            frame_total = len(frames)
        self._set_results_state(frame_total)
        self._refresh_stress_strain_plot(self.frame_slider.value() if frame_total > 0 else 0)

    def _load_position_pair(self):
        workspace_dir = _workspace_dir()
        initial_pos_path = _workspace_output_path("initial_pos.csv")
        final_pos_path = _workspace_output_path("final_pos.csv")
        if not os.path.exists(initial_pos_path):
            initial_pos_path = os.path.join(workspace_dir, "initial_pos.csv")
        if not os.path.exists(final_pos_path):
            final_pos_path = os.path.join(workspace_dir, "final_pos.csv")
        results_dir = _workspace_output_path("results")
        if not os.path.isdir(results_dir):
            results_dir = os.path.join(workspace_dir, "results")
        pos_history_path = _workspace_output_path("pos_history.npy")
        if not os.path.exists(pos_history_path):
            pos_history_path = os.path.join(workspace_dir, "pos_history.npy")

        initial_pos = None
        final_pos = None
        frames = getattr(self.sketch_view, "animation_frames", [])
        if frames:
            initial_pos = np.asarray(frames[0], dtype=float)
            final_pos = np.asarray(frames[-1], dtype=float)
        else:
            candidates = []
            result_files = []
            if os.path.isdir(results_dir):
                result_files = sorted(
                    f for f in os.listdir(results_dir)
                    if f.startswith("step_") and f.endswith(".csv")
                )
                if result_files:
                    candidates.append(("results", os.path.getmtime(os.path.join(results_dir, result_files[-1]))))
            if os.path.exists(pos_history_path):
                candidates.append(("pos_history", os.path.getmtime(pos_history_path)))
            if os.path.exists(initial_pos_path) and os.path.exists(final_pos_path):
                candidates.append(
                    ("initial_final", max(os.path.getmtime(initial_pos_path), os.path.getmtime(final_pos_path)))
                )
            if not candidates:
                raise RuntimeError(f"No simulation results found in {WORKSPACE_DIR_NAME}/.")

            source = max(candidates, key=lambda item: item[1])[0]
            if source == "pos_history":
                pos_history = np.load(pos_history_path)
                if pos_history.ndim != 3 or pos_history.shape[2] != 2:
                    raise RuntimeError("pos_history.npy has an unexpected shape.")
                unit_scale = self.sketch_view._unit_scale_to_meters()
                if unit_scale:
                    pos_history = pos_history / float(unit_scale)
                initial_pos = np.asarray(pos_history[0], dtype=float)
                final_pos = np.asarray(pos_history[-1], dtype=float)
            elif source == "results":
                if len(result_files) < 2:
                    raise RuntimeError("Not enough result frames to compare.")
                initial_df = pd.read_csv(os.path.join(results_dir, result_files[0]))
                final_df = pd.read_csv(os.path.join(results_dir, result_files[-1]))
                initial_pos = initial_df[["x", "y"]].to_numpy(dtype=float)
                final_pos = final_df[["x", "y"]].to_numpy(dtype=float)
            else:
                initial_df = pd.read_csv(initial_pos_path)
                final_df = pd.read_csv(final_pos_path)
                initial_pos = initial_df[["x", "y"]].to_numpy(dtype=float)
                final_pos = final_df[["x", "y"]].to_numpy(dtype=float)

        if initial_pos.shape != final_pos.shape:
            raise RuntimeError("Position files have different shapes.")
        return initial_pos, final_pos

    def _estimate_reference_modulus(self):
        mats = getattr(self.project_state, "materials", {}) or {}
        for mat in mats.values():
            props = getattr(mat, "properties", {}) or {}
            try:
                val = float(props.get("youngs_modulus", 0.0))
            except Exception:
                val = 0.0
            if val > 0.0:
                return val
        return 1.0

    def _sample_fields_from_positions(self, initial_pos, final_pos):
        displacement = final_pos - initial_pos
        u1 = displacement[:, 0]
        u2 = displacement[:, 1]
        umag = np.linalg.norm(displacement, axis=1)
        span = float(np.ptp(initial_pos[:, 0]) + np.ptp(initial_pos[:, 1]))
        char_len = max(span, 1e-9)
        e11 = u1 / char_len
        e22 = u2 / char_len
        e12 = 0.5 * (u1 + u2) / char_len
        emag = np.sqrt(np.maximum(0.0, e11 ** 2 + e22 ** 2 + 2.0 * e12 ** 2))
        e_ref = self._estimate_reference_modulus()
        s11 = e_ref * e11
        s22 = e_ref * e22
        s12 = 0.5 * e_ref * e12
        svm = np.sqrt(np.maximum(0.0, s11 ** 2 - s11 * s22 + s22 ** 2 + 3.0 * s12 ** 2))
        sed = 0.5 * np.maximum(0.0, s11 * e11 + s22 * e22 + 2.0 * s12 * e12)
        return {
            "disp": displacement,
            "u1": u1,
            "u2": u2,
            "umag": umag,
            "e11": e11,
            "e22": e22,
            "e12": e12,
            "emag": emag,
            "s11": s11,
            "s22": s22,
            "s12": s12,
            "svm": svm,
            "sed": sed,
        }

    def _format_summary(self, values):
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return "min --, max --, mean --"
        return (
            f"min {float(np.min(arr)):.4g}, "
            f"max {float(np.max(arr)):.4g}, "
            f"mean {float(np.mean(arr)):.4g}"
        )

    def _show_scalar_preview(self, initial_pos, displacement, scalar_vals, mode_label="Displacement"):
        scalar = np.asarray(scalar_vals, dtype=float).reshape(-1)
        self.sketch_view.show_displacement_vectors(
            initial_pos,
            displacement,
            scalar_values=scalar,
            field_label=str(mode_label),
        )
        window = self.window()
        view_3d = getattr(window, "view_3d", None) if window is not None else None
        if view_3d is not None and hasattr(view_3d, "set_scalar_node_colors"):
            try:
                view_3d.set_scalar_node_colors(scalar, cmap_name="viridis")
            except Exception:
                pass

    def _apply_selected_compare_field(self):
        controller = self._results_controller
        if controller is None:
            return
        mode = str(self.compare_mode_combo.currentData() or "none")
        applied = controller.set_active_result_field(mode)
        idx = self.compare_mode_combo.findData(applied)
        if idx >= 0 and idx != self.compare_mode_combo.currentIndex():
            self.compare_mode_combo.blockSignals(True)
            self.compare_mode_combo.setCurrentIndex(idx)
            self.compare_mode_combo.blockSignals(False)
        self._sync_result_display_toggles(applied)
        self.sketch_view.refresh_results_field()
        frame_index = max(0, int(self.frame_slider.value()))
        self.compare_summary_label.setText(controller.summarize_field(frame_index))
        self._update_sample_activity()
        label = controller.field_label(applied)
        self._announce_status(f"Applied {label} coloring.")

    def _sync_result_display_toggles(self, field_key):
        controller = self._results_controller
        if controller is None:
            return
        spec = controller.RESULT_FIELD_SPECS.get(str(field_key or "none").strip().lower(), {})
        domain = str(spec.get("domain") or "").strip().lower()
        source = str(spec.get("source") or "").strip().lower()
        if domain == "triangle" and source in {"stress", "strain"}:
            if hasattr(self, "show_nodes_checkbox"):
                self.show_nodes_checkbox.blockSignals(True)
                self.show_nodes_checkbox.setChecked(True)
                self.show_nodes_checkbox.blockSignals(False)
            self.sketch_view.set_replay_debug_overlays(show_particles=True)
            return
        if domain == "node":
            if hasattr(self, "show_nodes_checkbox"):
                self.show_nodes_checkbox.blockSignals(True)
                self.show_nodes_checkbox.setChecked(True)
                self.show_nodes_checkbox.blockSignals(False)
            self.sketch_view.set_replay_debug_overlays(show_particles=True)

    def _update_sample_activity(self):
        controller = self._results_controller
        if controller is None:
            self.disp_activity_label.setText("Disp: --")
            self.stress_activity_label.setText("Stress: --")
            self.strain_activity_label.setText("Strain: --")
            return
        try:
            summary = controller.activity_summary()
            self.disp_activity_label.setText(summary.get("disp", "Disp: --"))
            self.stress_activity_label.setText(summary.get("stress", "Stress: --"))
            self.strain_activity_label.setText(summary.get("strain", "Strain: --"))
        except Exception:
            self.disp_activity_label.setText("Disp: --")
            self.stress_activity_label.setText("Stress: --")
            self.strain_activity_label.setText("Strain: --")

    def compare_positions(self):
        idx = self.compare_mode_combo.findData("disp_mag")
        if idx >= 0:
            self.compare_mode_combo.setCurrentIndex(idx)
        self._apply_selected_compare_field()

    def compare_results(self):
        self._apply_selected_compare_field()

    def _refresh_result_field_options(self):
        controller = self._results_controller
        if controller is None:
            return
        combo_value = str(self.compare_mode_combo.currentData() or "").strip().lower()
        active_value = str(controller.active_result_field() or "none").strip().lower()
        available_keys = {str(key).strip().lower() for key, _label in controller.result_field_options()}
        if combo_value and combo_value != "none" and combo_value in available_keys:
            current = combo_value
        elif active_value and active_value != "none" and active_value in available_keys:
            current = active_value
        else:
            current = str(controller.default_result_field() or "none").strip().lower()
        options = controller.result_field_options()
        self.compare_mode_combo.blockSignals(True)
        self.compare_mode_combo.clear()
        for key, label in options:
            self.compare_mode_combo.addItem(label, key)
        idx = self.compare_mode_combo.findData(current)
        if idx < 0:
            idx = self.compare_mode_combo.findData(controller.active_result_field())
        if idx < 0:
            idx = self.compare_mode_combo.findData("none")
        if idx >= 0:
            self.compare_mode_combo.setCurrentIndex(idx)
        self.compare_mode_combo.blockSignals(False)

    # ----- Custom mesh zones (Results page Custom mode) -------------------

    def _set_mode(self, mode):
        mode = "custom" if str(mode) == "custom" else "default"
        if mode == self._mode:
            self.mode_default_btn.setChecked(mode == "default")
            self.mode_custom_btn.setChecked(mode == "custom")
            return
        self._mode = mode
        self.mode_default_btn.setChecked(mode == "default")
        self.mode_custom_btn.setChecked(mode == "custom")
        is_custom = mode == "custom"
        for w in self._default_mode_widgets:
            try:
                w.setVisible(not is_custom)
            except Exception:
                pass
        self._custom_mode_card.setVisible(is_custom)
        # When leaving custom mode, cancel any in-progress polygon draw and
        # hide overlays so the user is not left in a weird state.
        if not is_custom and hasattr(self.sketch_view, "cancel_zone_draw"):
            try:
                self.sketch_view.cancel_zone_draw()
            except Exception:
                pass
        if is_custom and hasattr(self.sketch_view, "set_zone_overlay_visible"):
            try:
                self.sketch_view.set_zone_overlay_visible(True)
            except Exception:
                pass
            # Re-sync overlay polygons from project state — covers the case
            # where zones were loaded from a saved project.
            self._rebuild_zone_list()
            self._sync_zone_overlay()
            # Apply the dot-visibility toggle so entering Custom mode immediately
            # shows the clean (no-dots) aesthetic by default.
            self._apply_vertex_dot_visibility()
        elif hasattr(self.sketch_view, "set_zone_overlay_visible"):
            try:
                self.sketch_view.set_zone_overlay_visible(False)
            except Exception:
                pass

    def _apply_vertex_dot_visibility(self):
        show = bool(getattr(self, "zone_show_dots", None) and self.zone_show_dots.isChecked())
        # The canvas has two independent dot-rendering paths depending on
        # display_mode: "mesh"/"mesh_3d" → show_mesh_nodes;
        # "results" → show_anim_nodes (via set_animation_visibility).
        if hasattr(self.sketch_view, "set_mesh_view_visibility"):
            try:
                self.sketch_view.set_mesh_view_visibility(show_nodes=show)
            except Exception:
                pass
        if hasattr(self.sketch_view, "set_animation_visibility"):
            try:
                self.sketch_view.set_animation_visibility(show_nodes=show)
            except Exception:
                pass
        # Also sync the existing default-mode "Particles" checkbox so both
        # toggles agree (single source of truth across the panel). Block the
        # signal so we don't re-enter through its slot.
        if hasattr(self, "show_nodes_checkbox"):
            try:
                self.show_nodes_checkbox.blockSignals(True)
                self.show_nodes_checkbox.setChecked(show)
                self.show_nodes_checkbox.blockSignals(False)
            except Exception:
                pass
        # set_animation_visibility flips per-item visibility, but the canvas
        # may need a manual repaint to pick it up if no frame is being applied.
        try:
            self.sketch_view.viewport().update()
        except Exception:
            pass

    def _on_zone_dots_toggled(self, checked):
        # Mirror the state into the default-mode checkbox so both stay in sync.
        if hasattr(self, "default_show_dots"):
            try:
                self.default_show_dots.blockSignals(True)
                self.default_show_dots.setChecked(bool(checked))
                self.default_show_dots.blockSignals(False)
            except Exception:
                pass
        self._apply_vertex_dot_visibility()

    def _on_default_dots_toggled(self, checked):
        # Mirror into the Custom-mode checkbox.
        if hasattr(self, "zone_show_dots"):
            try:
                self.zone_show_dots.blockSignals(True)
                self.zone_show_dots.setChecked(bool(checked))
                self.zone_show_dots.blockSignals(False)
            except Exception:
                pass
        self._apply_vertex_dot_visibility()

    def _on_select_zone_clicked(self):
        if not hasattr(self.sketch_view, "begin_zone_draw"):
            QMessageBox.information(
                self,
                "Not available",
                "Polygon zone drawing is not available in the current view.",
            )
            return
        try:
            self.sketch_view.begin_zone_draw(on_complete=self._on_zone_drawn)
        except Exception as exc:
            QMessageBox.warning(self, "Zone draw failed", str(exc))

    def _on_zone_drawn(self, points):
        """Called by sketch_view when the user finishes drawing a polygon."""
        try:
            from models import CustomMeshZone
            zone = CustomMeshZone(points=list(points or []), approx_node_count=500)
        except Exception as exc:
            QMessageBox.warning(self, "Zone error", str(exc))
            return
        if not zone.points or len(zone.points) < 3:
            return
        zones = getattr(self.project_state, "custom_mesh_zones", None)
        if zones is None:
            zones = []
            self.project_state.custom_mesh_zones = zones
        zones.append(zone)
        self._rebuild_zone_list()
        self._sync_zone_overlay()

    def _on_clear_all_zones_clicked(self):
        if not getattr(self.project_state, "custom_mesh_zones", None):
            return
        self.project_state.custom_mesh_zones = []
        self._rebuild_zone_list()
        self._sync_zone_overlay()

    def _remove_zone_at(self, index):
        zones = getattr(self.project_state, "custom_mesh_zones", None) or []
        if 0 <= index < len(zones):
            del zones[index]
            self.project_state.custom_mesh_zones = zones
            self._rebuild_zone_list()
            self._sync_zone_overlay()

    def _on_zone_node_count_changed(self, index, value):
        zones = getattr(self.project_state, "custom_mesh_zones", None) or []
        if 0 <= index < len(zones):
            try:
                zones[index].approx_node_count = int(value)
            except Exception:
                return
            # Refresh derived-size label inline.
            row = self._zone_row_widgets[index]
            size_label = row.get("size_label")
            if size_label is not None:
                size = zones[index].derived_mesh_size()
                size_label.setText(f"~ size {size:.4g}" if size > 0 else "~ size --")

    def _rebuild_zone_list(self):
        # Clear existing rows
        while self.zone_list_layout.count():
            item = self.zone_list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._zone_row_widgets = []
        zones = getattr(self.project_state, "custom_mesh_zones", None) or []
        self.zone_empty_label.setVisible(len(zones) == 0)
        for i, zone in enumerate(zones):
            row_widget = QWidget()
            row = QHBoxLayout(row_widget)
            _apply_layout_metrics(row, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
            row.addWidget(QLabel(f"Zone {i + 1}"))
            spin = QSpinBox()
            spin.setRange(1, 1_000_000)
            spin.setValue(int(getattr(zone, "approx_node_count", 500) or 500))
            spin.setSuffix(" nodes")
            spin.setToolTip("Approximate target node count inside this zone")
            spin.valueChanged.connect(lambda v, idx=i: self._on_zone_node_count_changed(idx, v))
            row.addWidget(spin)
            size = zone.derived_mesh_size() if hasattr(zone, "derived_mesh_size") else 0.0
            size_label = QLabel(f"~ size {size:.4g}" if size > 0 else "~ size --")
            size_label.setObjectName("MinorStatusLabel")
            row.addWidget(size_label)
            row.addStretch(1)
            remove_btn = QPushButton("Remove")
            remove_btn.clicked.connect(lambda _checked=False, idx=i: self._remove_zone_at(idx))
            row.addWidget(remove_btn)
            self.zone_list_layout.addWidget(row_widget)
            self._zone_row_widgets.append({"widget": row_widget, "spin": spin, "size_label": size_label})

    def _sync_zone_overlay(self):
        if hasattr(self.sketch_view, "set_zone_overlay"):
            zones = getattr(self.project_state, "custom_mesh_zones", None) or []
            try:
                self.sketch_view.set_zone_overlay(
                    [getattr(z, "points", []) for z in zones]
                )
            except Exception:
                pass

    def _on_regenerate_with_zones_clicked(self):
        zones = getattr(self.project_state, "custom_mesh_zones", None) or []
        if not zones:
            QMessageBox.information(
                self,
                "No zones",
                "Add at least one refinement zone before regenerating the mesh.",
            )
            return
        # Re-entrancy guard: running gmsh twice in parallel from this button
        # causes a native crash (two gmsh.initialize() calls in racing threads).
        # We use a local flag because the two-stage flow (particles → connections)
        # has a brief gap where the worker thread is finished but the next one
        # hasn't started; checking only the worker thread would miss that window.
        if getattr(self, "_zone_regen_in_progress", False):
            QMessageBox.information(
                self,
                "Already regenerating",
                "Mesh regeneration is already in progress. Wait for it to finish before clicking OK again.",
            )
            return
        mesh_thread = getattr(self.sketch_view, "_mesh_thread", None)
        if mesh_thread is not None and mesh_thread.isRunning():
            QMessageBox.information(
                self,
                "Already regenerating",
                "Mesh regeneration is already in progress. Wait for it to finish before clicking OK again.",
            )
            return
        self._zone_regen_in_progress = True
        self._zone_regen_settled_ticks = 0
        # Reuse the MeshPanel's full particle+connections regen path so zones
        # flow into the same gmsh-adaptive route that built the current mesh.
        window = self.window()
        mesh_panel = None
        if window is not None:
            mesh_panel = getattr(window, "mesh_tab", None)
            if mesh_panel is None:
                props = getattr(window, "properties_panel", None)
                if props is not None:
                    mesh_panel = getattr(props, "mesh_tab", None)
        if mesh_panel is None or not hasattr(mesh_panel, "generate_particles_and_connections"):
            QMessageBox.warning(
                self,
                "Mesh panel unavailable",
                "Could not locate the Mesh panel to trigger regeneration.",
            )
            return
        self.zone_regenerate_btn.setEnabled(False)
        self.zone_regenerate_btn.setText("Regenerating...")
        try:
            mesh_panel.generate_particles_and_connections()
        except Exception as exc:
            self.zone_regenerate_btn.setEnabled(True)
            self.zone_regenerate_btn.setText("OK — Regenerate Mesh")
            self._zone_regen_in_progress = False
            QMessageBox.warning(self, "Regeneration failed", str(exc))
            return
        # Poll the worker thread to re-enable the OK button once both stages
        # (particles → connections) have finished. Done with a small QTimer
        # rather than wiring into existing signals to keep this change local.
        from PySide6.QtCore import QTimer
        self._zone_regen_watchdog = QTimer(self)
        self._zone_regen_watchdog.setInterval(300)
        self._zone_regen_watchdog.timeout.connect(self._zone_regen_check_done)
        self._zone_regen_watchdog.start()
        # Reveal the "Run Simulation" hint button now; it's only a hint, so
        # showing it during regen is fine.
        self._on_mesh_regen_finished()

    def _zone_regen_check_done(self):
        # The two-stage flow may briefly show no running thread between the
        # particle stage finishing and the connections stage starting. Require
        # several consecutive "no thread running" ticks before declaring done.
        thread = getattr(self.sketch_view, "_mesh_thread", None)
        if thread is not None and thread.isRunning():
            self._zone_regen_settled_ticks = 0
            return
        self._zone_regen_settled_ticks = int(getattr(self, "_zone_regen_settled_ticks", 0)) + 1
        if self._zone_regen_settled_ticks < 4:  # ~1.2s of quiet before unlock
            return
        try:
            self._zone_regen_watchdog.stop()
        except Exception:
            pass
        self._zone_regen_in_progress = False
        try:
            self.zone_regenerate_btn.setEnabled(True)
            self.zone_regenerate_btn.setText("OK — Regenerate Mesh")
        except Exception:
            pass
        # Re-apply the dot-visibility preference. The particle generation
        # stage of the regen flow force-sets show_mesh_nodes=True at the
        # sketch_view level; we override it here so the user's Custom-mode
        # preference (default off) is honored after regen.
        self._apply_vertex_dot_visibility()

    def _on_mesh_regen_finished(self, *_args, **_kwargs):
        # Show the "Run Simulation" hint button only after a successful regen.
        try:
            self.zone_run_sim_btn.setVisible(True)
        except Exception:
            pass

    def _on_run_sim_after_regen_clicked(self):
        window = self.window()
        if window is None:
            return
        execute_command = getattr(window, "execute_app_command", None)
        if callable(execute_command):
            try:
                result = execute_command(RunSolverCommand()) or {}
            except Exception as exc:
                QMessageBox.warning(self, "Run failed", str(exc))
                return
            if result.get("blocked"):
                # The solver impl already raised a "missing stage" dialog and
                # focused the relevant stage — nothing more to do here.
                return
            if result.get("job") is None:
                QMessageBox.information(
                    self,
                    "Run simulation",
                    "Solver did not start. Check the Job panel for prerequisites.",
                )
            return
        QMessageBox.information(
            self,
            "Run simulation",
            "Open the Job panel and start the solver to run with the new mesh.",
        )


class MeshPanel(QWidget):
    def __init__(self, sketch_view, parent=None, project_state=None):
        super().__init__(parent)
        self.sketch_view = sketch_view
        self.project_state = _resolve_project_state(sketch_view, project_state)
        self._settings = QSettings("CPD-Modeller", "CPD-SimStudio")
        self._generate_label_full = "Generate"
        self._connections_label_full = "Generate Connections"
        self._back_label_full = "Geometry"
        self._reset_preview_label_full = "Reset Preview View"

        layout = QVBoxLayout(self)
        _apply_layout_metrics(layout)

        header_label = QLabel("Particles")
        header_label.setObjectName("SummaryLabel")
        layout.addWidget(header_label)

        # ---- Sub-tab assembly: Seeding / Boundary+Partition / Generate ----
        self.mesh_tabs = QTabWidget()
        self.mesh_tabs.setDocumentMode(True)
        self.mesh_tabs.tabBar().setExpanding(True)
        layout.addWidget(self.mesh_tabs, 1)

        # Seeding tab — Local Seeds, Vertex Seeds, Part Seeds
        seeding_page = QWidget()
        seeding_page_layout = QVBoxLayout(seeding_page)
        _apply_layout_metrics(seeding_page_layout, margins=(4, 4, 4, 4), spacing=4)
        self.mesh_tabs.addTab(seeding_page, "Seeding")

        # Boundary & Partition tab — Boundary Layers, Partitions
        boundary_page = QWidget()
        boundary_page_layout = QVBoxLayout(boundary_page)
        _apply_layout_metrics(boundary_page_layout, margins=(4, 4, 4, 4), spacing=4)
        self.mesh_tabs.addTab(boundary_page, "Boundary && Partition")

        # Generate tab — Generation controls, Display options, Summary
        generate_page = QWidget()
        generate_page_layout = QVBoxLayout(generate_page)
        _apply_layout_metrics(generate_page_layout, margins=(4, 4, 4, 4), spacing=4)
        self.mesh_tabs.addTab(generate_page, "Generate")

        generation_group = QGroupBox("Generation")
        generation_layout = QVBoxLayout(generation_group)
        _apply_layout_metrics(
            generation_layout,
            margins=(DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN),
            spacing=DOCK_ROW_SPACING,
        )
        generate_page_layout.addWidget(generation_group)

        # Local Seeds (Abaqus-style edge seeding) — sits between Generation
        # and Display in the Mesh panel.
        seeds_group = QGroupBox("Local Seeds")
        seeds_layout = QVBoxLayout(seeds_group)
        _apply_layout_metrics(
            seeds_layout,
            margins=(DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN),
            spacing=DOCK_ROW_SPACING,
        )
        seeds_hint = QLabel(
            "Bias mesh density along one or more edges. Local seeds override "
            "the global element size where they apply."
        )
        seeds_hint.setObjectName("MinorStatusLabel")
        seeds_hint.setWordWrap(True)
        seeds_layout.addWidget(seeds_hint)

        # 3 buttons per row. Short labels keep the panel narrow; the full
        # action lives in the tooltip.
        seeds_btn_row = QHBoxLayout()
        _apply_layout_metrics(seeds_btn_row, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        self.add_single_seed_btn = QPushButton("Single")
        self.add_single_seed_btn.setToolTip(
            "Seed Single Edge\nPick one edge on the canvas and open the Local Seeds dialog."
        )
        self.add_single_seed_btn.clicked.connect(lambda: self._begin_edge_seed_pick("single"))
        self.add_multi_seed_btn = QPushButton("Multiple")
        self.add_multi_seed_btn.setToolTip(
            "Seed Multiple Edges\nPick several edges on the canvas (click to add/remove, Enter or "
            "right-click to finish), then configure them as one seed."
        )
        self.add_multi_seed_btn.clicked.connect(lambda: self._begin_edge_seed_pick("multi"))
        self.clear_seeds_btn = QPushButton("Clear")
        self.clear_seeds_btn.setToolTip("Clear all seeds — remove every edge seed from the project.")
        self.clear_seeds_btn.clicked.connect(self._clear_all_edge_seeds)
        for btn in (self.add_single_seed_btn, self.add_multi_seed_btn, self.clear_seeds_btn):
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setMinimumWidth(0)
            btn.setMaximumWidth(120)
            seeds_btn_row.addWidget(btn, 1)
        seeds_layout.addLayout(seeds_btn_row)

        # Smart-selection helpers — quick ways to pick many edges at once.
        smart_btn_row = QHBoxLayout()
        _apply_layout_metrics(smart_btn_row, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        smart_label = QLabel("Smart:")
        smart_label.setObjectName("MinorStatusLabel")
        smart_btn_row.addWidget(smart_label)
        self.seed_by_chain_btn = QPushButton("Chain")
        self.seed_by_chain_btn.setToolTip(
            "By Chain\nPick one edge. The selection automatically grows along the "
            "connected boundary until it hits a sharp corner or junction."
        )
        self.seed_by_chain_btn.clicked.connect(self._begin_edge_seed_by_chain)
        smart_btn_row.addWidget(self.seed_by_chain_btn, 1)
        self.seed_by_part_btn = QPushButton("Part")
        self.seed_by_part_btn.setToolTip(
            "By Part\nPick any edge of a part — all boundary edges of that part are "
            "selected and passed to the Local Seeds dialog."
        )
        self.seed_by_part_btn.clicked.connect(self._begin_edge_seed_by_part)
        smart_btn_row.addWidget(self.seed_by_part_btn, 1)
        self.seed_by_length_btn = QPushButton("Length")
        self.seed_by_length_btn.setToolTip(
            "By Length\nSelect all edges whose length falls within a given range, then "
            "configure them as one seed."
        )
        self.seed_by_length_btn.clicked.connect(self._begin_edge_seed_by_length)
        smart_btn_row.addWidget(self.seed_by_length_btn, 1)
        for btn in (self.seed_by_chain_btn, self.seed_by_part_btn, self.seed_by_length_btn):
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setMinimumWidth(0)
            btn.setMaximumWidth(100)
        seeds_layout.addLayout(smart_btn_row)

        # Seed templates — save a configuration once, apply it to many edges later.
        templates_label = QLabel("Templates")
        templates_label.setObjectName("MinorStatusLabel")
        seeds_layout.addWidget(templates_label)
        self.templates_list_container = QWidget()
        self.templates_list_layout = QVBoxLayout(self.templates_list_container)
        self.templates_list_layout.setContentsMargins(0, 0, 0, 0)
        self.templates_list_layout.setSpacing(2)
        seeds_layout.addWidget(self.templates_list_container)
        self.templates_empty_label = QLabel(
            "No templates saved yet. Click 'Save as template...' inside the Local Seeds dialog."
        )
        self.templates_empty_label.setObjectName("MinorStatusLabel")
        self.templates_empty_label.setWordWrap(True)
        seeds_layout.addWidget(self.templates_empty_label)
        self._template_row_widgets = []

        # Matched edge pairs (periodic / symmetric meshing).
        pair_label = QLabel("Matched edge pairs")
        pair_label.setObjectName("MinorStatusLabel")
        seeds_layout.addWidget(pair_label)
        pair_btn_row = QHBoxLayout()
        _apply_layout_metrics(pair_btn_row, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        self.add_pair_btn = QPushButton("Match Edge Pair...")
        self.add_pair_btn.setToolTip(
            "Pick two opposing edges (master then slave). Mesh nodes on the "
            "slave will be forced to match the master's spacing — required "
            "for periodic boundary conditions and structured/symmetric meshes."
        )
        self.add_pair_btn.clicked.connect(self._begin_match_pair_pick)
        self.clear_pairs_btn = QPushButton("Clear pairs")
        self.clear_pairs_btn.clicked.connect(self._clear_all_pairs)
        pair_btn_row.addWidget(self.add_pair_btn)
        pair_btn_row.addWidget(self.clear_pairs_btn)
        pair_btn_row.addStretch(1)
        seeds_layout.addLayout(pair_btn_row)
        self.pairs_list_container = QWidget()
        self.pairs_list_layout = QVBoxLayout(self.pairs_list_container)
        self.pairs_list_layout.setContentsMargins(0, 0, 0, 0)
        self.pairs_list_layout.setSpacing(2)
        seeds_layout.addWidget(self.pairs_list_container)
        self.pairs_empty_label = QLabel("No matched pairs.")
        self.pairs_empty_label.setObjectName("MinorStatusLabel")
        seeds_layout.addWidget(self.pairs_empty_label)
        self._pair_row_widgets = []

        # Per-seed list. Each row carries its own Edit and Remove buttons so
        # users can change one seed without clearing all of them.
        self.seeds_list_container = QWidget()
        self.seeds_list_layout = QVBoxLayout(self.seeds_list_container)
        self.seeds_list_layout.setContentsMargins(0, 0, 0, 0)
        self.seeds_list_layout.setSpacing(2)
        seeds_layout.addWidget(self.seeds_list_container)
        self.seeds_empty_label = QLabel("No seeds defined.")
        self.seeds_empty_label.setObjectName("MinorStatusLabel")
        seeds_layout.addWidget(self.seeds_empty_label)
        self._seed_row_widgets = []
        seeding_page_layout.addWidget(seeds_group)

        # Vertex Seeds (point-anchored refinement at corners / holes / notches).
        vseeds_group = QGroupBox("Vertex Seeds")
        vseeds_layout = QVBoxLayout(vseeds_group)
        _apply_layout_metrics(
            vseeds_layout,
            margins=(DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN),
            spacing=DOCK_ROW_SPACING,
        )
        vseeds_hint = QLabel(
            "Refine the mesh around a picked corner / hole vertex / notch. "
            "Useful for resolving stress concentrations."
        )
        vseeds_hint.setObjectName("MinorStatusLabel")
        vseeds_hint.setWordWrap(True)
        vseeds_layout.addWidget(vseeds_hint)

        vseeds_btn_row = QHBoxLayout()
        _apply_layout_metrics(vseeds_btn_row, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        self.add_vertex_seed_btn = QPushButton("Seed at Vertex")
        self.add_vertex_seed_btn.setToolTip(
            "Pick a vertex on the canvas, then set target size and influence radius."
        )
        self.add_vertex_seed_btn.clicked.connect(self._begin_vertex_seed_pick)
        self.clear_vertex_seeds_btn = QPushButton("Clear all vertex seeds")
        self.clear_vertex_seeds_btn.clicked.connect(self._clear_all_vertex_seeds)
        vseeds_btn_row.addWidget(self.add_vertex_seed_btn)
        vseeds_btn_row.addWidget(self.clear_vertex_seeds_btn)
        vseeds_btn_row.addStretch(1)
        vseeds_layout.addLayout(vseeds_btn_row)

        self.vseeds_list_container = QWidget()
        self.vseeds_list_layout = QVBoxLayout(self.vseeds_list_container)
        self.vseeds_list_layout.setContentsMargins(0, 0, 0, 0)
        self.vseeds_list_layout.setSpacing(2)
        vseeds_layout.addWidget(self.vseeds_list_container)
        self.vseeds_empty_label = QLabel("No vertex seeds defined.")
        self.vseeds_empty_label.setObjectName("MinorStatusLabel")
        vseeds_layout.addWidget(self.vseeds_empty_label)
        self._vseed_row_widgets = []
        seeding_page_layout.addWidget(vseeds_group)

        # Boundary Layers (CFD-style inflation).
        bl_group = QGroupBox("Boundary Layers")
        bl_layout = QVBoxLayout(bl_group)
        _apply_layout_metrics(
            bl_layout,
            margins=(DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN),
            spacing=DOCK_ROW_SPACING,
        )
        bl_hint = QLabel(
            "Stack thin elements parallel to picked edges, growing into the "
            "interior. Used for CFD walls, thermal layers, contact zones."
        )
        bl_hint.setObjectName("MinorStatusLabel")
        bl_hint.setWordWrap(True)
        bl_layout.addWidget(bl_hint)

        bl_btn_row = QHBoxLayout()
        _apply_layout_metrics(bl_btn_row, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        self.add_bl_btn = QPushButton("Add Boundary Layer...")
        self.add_bl_btn.setToolTip("Pick one or more edges, then configure the inflation parameters.")
        self.add_bl_btn.clicked.connect(self._begin_bl_pick)
        self.clear_bl_btn = QPushButton("Clear all")
        self.clear_bl_btn.clicked.connect(self._clear_all_bl_seeds)
        bl_btn_row.addWidget(self.add_bl_btn)
        bl_btn_row.addWidget(self.clear_bl_btn)
        bl_btn_row.addStretch(1)
        bl_layout.addLayout(bl_btn_row)

        self.bl_list_container = QWidget()
        self.bl_list_layout = QVBoxLayout(self.bl_list_container)
        self.bl_list_layout.setContentsMargins(0, 0, 0, 0)
        self.bl_list_layout.setSpacing(2)
        bl_layout.addWidget(self.bl_list_container)
        self.bl_empty_label = QLabel("No boundary layers defined.")
        self.bl_empty_label.setObjectName("MinorStatusLabel")
        bl_layout.addWidget(self.bl_empty_label)
        self._bl_row_widgets = []
        boundary_page_layout.addWidget(bl_group)

        # Face partitioning (split a part by a drawn line into sub-parts).
        part_group = QGroupBox("Partitions")
        part_layout = QVBoxLayout(part_group)
        _apply_layout_metrics(
            part_layout,
            margins=(DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN),
            spacing=DOCK_ROW_SPACING,
        )
        part_hint = QLabel(
            "Split a part into multiple sub-parts by drawing a straight cut "
            "line. Each sub-part inherits the original's material and becomes "
            "addressable for per-part mesh sizing and BCs."
        )
        part_hint.setObjectName("MinorStatusLabel")
        part_hint.setWordWrap(True)
        part_layout.addWidget(part_hint)

        part_btn_row = QHBoxLayout()
        _apply_layout_metrics(part_btn_row, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        self.partition_btn = QPushButton("Partition Face...")
        self.partition_btn.setToolTip(
            "Click two points on a part to draw a cut line. The part will be "
            "split into the resulting sub-pieces."
        )
        self.partition_btn.clicked.connect(self._begin_partition_pick)
        part_btn_row.addWidget(self.partition_btn)
        part_btn_row.addStretch(1)
        part_layout.addLayout(part_btn_row)

        self.partition_status_label = QLabel("")
        self.partition_status_label.setObjectName("MinorStatusLabel")
        self.partition_status_label.setWordWrap(True)
        part_layout.addWidget(self.partition_status_label)
        boundary_page_layout.addWidget(part_group)

        # Per-part element size overrides. Mirrors Abaqus's "seed by part"
        # workflow — each row corresponds to one part with an editable size.
        # Empty value = inherit the global bulk size.
        part_seeds_group = QGroupBox("Part Seeds")
        part_seeds_layout = QVBoxLayout(part_seeds_group)
        _apply_layout_metrics(
            part_seeds_layout,
            margins=(DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN),
            spacing=DOCK_ROW_SPACING,
        )
        ps_hint = QLabel(
            "Override the global element size on a per-part basis. Smaller "
            "values refine; the override is honored when finer than the "
            "global bulk. Empty = inherit global."
        )
        ps_hint.setObjectName("MinorStatusLabel")
        ps_hint.setWordWrap(True)
        part_seeds_layout.addWidget(ps_hint)
        self.part_seeds_list_container = QWidget()
        self.part_seeds_list_layout = QVBoxLayout(self.part_seeds_list_container)
        self.part_seeds_list_layout.setContentsMargins(0, 0, 0, 0)
        self.part_seeds_list_layout.setSpacing(2)
        part_seeds_layout.addWidget(self.part_seeds_list_container)
        self.part_seeds_empty_label = QLabel("No parts available.")
        self.part_seeds_empty_label.setObjectName("MinorStatusLabel")
        part_seeds_layout.addWidget(self.part_seeds_empty_label)
        # Live estimate — updates as the user types in any spinbox.
        self.part_seeds_estimate_label = QLabel("")
        self.part_seeds_estimate_label.setObjectName("MinorStatusLabel")
        self.part_seeds_estimate_label.setWordWrap(True)
        part_seeds_layout.addWidget(self.part_seeds_estimate_label)
        self._part_seed_row_widgets = []
        seeding_page_layout.addWidget(part_seeds_group)
        seeding_page_layout.addStretch(1)
        boundary_page_layout.addStretch(1)

        display_group = QGroupBox("Display")
        display_group_layout = QVBoxLayout(display_group)
        _apply_layout_metrics(
            display_group_layout,
            margins=(DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN),
            spacing=DOCK_ROW_SPACING,
        )
        generate_page_layout.addWidget(display_group)
        summary_group = QGroupBox("Summary")
        summary_layout = QVBoxLayout(summary_group)
        _apply_layout_metrics(
            summary_layout,
            margins=(DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN),
            spacing=DOCK_ROW_SPACING,
        )
        generate_page_layout.addWidget(summary_group)
        generate_page_layout.addStretch(1)

        self._generation_button_grid = QGridLayout()
        _apply_layout_metrics(self._generation_button_grid, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.clicked.connect(self.generate_particles_and_connections)
        self.generate_btn.setIcon(get_icon("mesh_preview"))
        self.generate_btn.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.generate_btn.setToolTip("Generate particles and connections from the current geometry.")

        self.connections_btn = QPushButton("Generate Connections")
        self.connections_btn.clicked.connect(self.generate_connections)
        self.connections_btn.setIcon(get_icon("connections"))
        self.connections_btn.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.connections_btn.setToolTip(
            "Generate Delaunay connections from the current particle set (reuses existing particles if present)."
        )
        self.connections_btn.setVisible(False)

        self.back_btn = QPushButton("Geometry")
        self.back_btn.clicked.connect(self.back_to_geometry)
        self.back_btn.setIcon(get_icon("select"))
        self.back_btn.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.back_btn.setToolTip("Return to geometry view")
        self._generation_action_buttons = [self.generate_btn, self.back_btn]
        generation_layout.addLayout(self._generation_button_grid)

        self.generation_form = QFormLayout()
        self.generation_form.setLabelAlignment(Qt.AlignLeft)
        self.generation_form.setFormAlignment(Qt.AlignTop)
        self.generation_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        _apply_layout_metrics(self.generation_form, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        self.sizing_combo = QComboBox()
        self.sizing_combo.addItems(["By spacing (dx)", "By total particle count"])
        self.sizing_combo.currentIndexChanged.connect(self._update_sizing_mode)
        self.sizing_combo.currentIndexChanged.connect(self._update_backend_hint)
        self.sizing_combo.setToolTip("Choose whether generation is driven by spacing or particle count.")
        self.generation_form.addRow("Sizing", self.sizing_combo)

        self.dx_spin = QDoubleSpinBox()
        self.dx_spin.setRange(0.1, 5000.0)
        self.dx_spin.setDecimals(3)
        self.dx_spin.setValue(DEFAULT_DX)
        self.dx_spin.setSingleStep(0.5)
        self.dx_spin.valueChanged.connect(self._update_backend_hint)

        self.nodes_spin = QSpinBox()
        self.nodes_spin.setRange(10, 1_000_000)
        self.nodes_spin.setSingleStep(100)
        self.nodes_spin.setValue(1000)
        self.nodes_spin.valueChanged.connect(self._update_backend_hint)

        self.dist_combo = QComboBox()
        self.dist_combo.addItem("Poisson Disk Sampling", "global_poisson")
        self.dist_combo.setEnabled(False)
        self.dist_combo.currentIndexChanged.connect(self._update_backend_hint)
        self.backend_combo = QComboBox()
        self.backend_combo.currentIndexChanged.connect(self._update_backend_hint)
        self.generation_form.addRow("Spacing", self.dx_spin)
        self.generation_form.addRow("Target", self.nodes_spin)
        self.generation_form.addRow("Pattern", self.dist_combo)
        self.generation_form.addRow("Backend", self.backend_combo)

        # Adaptive sizing controls (Gmsh adaptive multi-part backend only).
        self.h_bulk_spin = QDoubleSpinBox()
        self.h_bulk_spin.setRange(0.001, 5000.0)
        self.h_bulk_spin.setDecimals(4)
        self.h_bulk_spin.setSingleStep(0.1)
        self.h_bulk_spin.setValue(DEFAULT_DX)
        self.h_bulk_spin.setToolTip("Target element size in the part interior (bulk).")

        self.h_feature_spin = QDoubleSpinBox()
        self.h_feature_spin.setRange(0.0, 5000.0)
        self.h_feature_spin.setDecimals(4)
        self.h_feature_spin.setSingleStep(0.05)
        self.h_feature_spin.setValue(0.0)
        self.h_feature_spin.setToolTip(
            "Element size at holes and multi-part interfaces only. The "
            "outermost edges of the geometry use h_bulk (use Edge Seeds for "
            "per-edge refinement on outer edges). 0 = auto (h_bulk * 0.7). "
            "Try h_bulk / 4 to h_bulk / 8 for tight stress-concentration "
            "refinement at holes."
        )

        self.transition_width_spin = QDoubleSpinBox()
        self.transition_width_spin.setRange(0.0, 50000.0)
        self.transition_width_spin.setDecimals(4)
        self.transition_width_spin.setSingleStep(0.5)
        self.transition_width_spin.setValue(0.0)
        self.transition_width_spin.setToolTip(
            "Distance over which size grows from h_feature toward h_bulk "
            "(exponential approach). 0 = auto (max(15 * h_feature, 10 * h_bulk))."
        )

        self.generation_form.addRow("h_bulk", self.h_bulk_spin)
        self.generation_form.addRow("h_feature", self.h_feature_spin)
        self.generation_form.addRow("Transition", self.transition_width_spin)
        generation_layout.addLayout(self.generation_form)

        display_row = QHBoxLayout()
        display_row.addWidget(QLabel("Show"))
        self.mesh_view_toggle = QCheckBox("Mesh view")
        self.mesh_view_toggle.setChecked(False)
        self.mesh_view_toggle.setToolTip(
            "Toggle between generated particles/connections and the underlying sketch geometry."
        )
        self.mesh_view_toggle.toggled.connect(self._set_mesh_view_mode)
        display_row.addWidget(self.mesh_view_toggle)
        self.show_nodes = QCheckBox("Particles")
        self.show_nodes.setChecked(False)
        self.show_nodes.toggled.connect(self._update_display)
        display_row.addWidget(self.show_nodes)
        self.show_mesh = QCheckBox("Connections")
        self.show_mesh.setChecked(False)
        self.show_mesh.toggled.connect(self._update_display)
        display_row.addWidget(self.show_mesh)
        display_row.addStretch(1)
        display_group_layout.addLayout(display_row)

        self.display_form = QFormLayout()
        self.display_form.setLabelAlignment(Qt.AlignLeft)
        self.display_form.setFormAlignment(Qt.AlignTop)
        self.display_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        _apply_layout_metrics(self.display_form, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        self.interface_color_btn = QPushButton("Color")
        self.interface_color_btn.clicked.connect(self._pick_interface_preview_color)
        self.interface_color_btn.setToolTip("Color used for interaction preview lines.")
        self.display_form.addRow("Preview color", self.interface_color_btn)
        self._load_interface_preview_color_setting()

        preview_style_widget = QWidget()
        preview_style_row = QHBoxLayout(preview_style_widget)
        _apply_layout_metrics(preview_style_row, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        self.conn_line_spin = QDoubleSpinBox()
        self.conn_line_spin.setRange(0.1, 20.0)
        self.conn_line_spin.setDecimals(2)
        self.conn_line_spin.setSingleStep(0.1)
        self.conn_line_spin.valueChanged.connect(self._update_mesh_preview_style)
        preview_style_row.addWidget(self.conn_line_spin)
        preview_style_row.addWidget(QLabel("Particle size"))
        self.particle_size_spin = QDoubleSpinBox()
        self.particle_size_spin.setRange(0.5, 50.0)
        self.particle_size_spin.setDecimals(2)
        self.particle_size_spin.setSingleStep(0.2)
        self.particle_size_spin.valueChanged.connect(self._update_mesh_preview_style)
        preview_style_row.addWidget(self.particle_size_spin)
        preview_style_row.addStretch(1)
        self.display_form.addRow("Preview style", preview_style_widget)
        display_group_layout.addLayout(self.display_form)
        self._load_mesh_preview_style_settings()

        fast_row = QHBoxLayout()
        self.fast_preview_check = QCheckBox("Fast preview")
        self.fast_preview_check.setChecked(
            bool(getattr(self.sketch_view, "fast_preview_enabled", FAST_PREVIEW_ENABLED))
        )
        self.fast_preview_check.toggled.connect(self._update_fast_preview)
        self.fast_preview_check.setToolTip(
            "Skip drawing connections above the limit for a faster UI preview."
        )
        fast_row.addWidget(self.fast_preview_check)
        conn_limit_label = QLabel("Conn. limit")
        conn_limit_label.setToolTip("Maximum previewed connections when fast preview is enabled.")
        fast_row.addWidget(conn_limit_label)
        self.fast_preview_limit = QSpinBox()
        self.fast_preview_limit.setRange(0, int(PREVIEW_CONNECTION_LIMIT))
        self.fast_preview_limit.setSingleStep(1000)
        self.fast_preview_limit.setValue(
            int(getattr(self.sketch_view, "fast_preview_connection_limit", FAST_PREVIEW_CONNECTION_LIMIT))
        )
        self.fast_preview_limit.valueChanged.connect(self._update_fast_preview)
        self.fast_preview_limit.setToolTip("Max connections to draw when fast preview is enabled.")
        fast_row.addWidget(self.fast_preview_limit)
        fast_row.addStretch(1)
        display_group_layout.addLayout(fast_row)

        gpu_row = QHBoxLayout()
        self.gpu_preview_check = QCheckBox("GPU point preview")
        self.gpu_preview_check.setChecked(
            bool(getattr(self.sketch_view, "gpu_point_preview_enabled", GPU_POINT_PREVIEW_ENABLED))
        )
        self.gpu_preview_check.toggled.connect(self._update_gpu_preview)
        self.gpu_preview_check.setToolTip(
            "Use GPU point-cloud rendering when only particles are shown."
        )
        gpu_row.addWidget(self.gpu_preview_check)
        self.gpu_auto_check = QCheckBox("Auto threshold")
        self.gpu_auto_check.setChecked(
            bool(getattr(self.sketch_view, "gpu_point_preview_auto", GPU_POINT_PREVIEW_AUTO_ENABLED))
        )
        self.gpu_auto_check.toggled.connect(self._update_gpu_preview)
        gpu_row.addWidget(self.gpu_auto_check)
        gpu_row.addWidget(QLabel("Threshold"))
        self.gpu_threshold_spin = QSpinBox()
        self.gpu_threshold_spin.setRange(0, 10_000_000)
        self.gpu_threshold_spin.setSingleStep(50_000)
        self.gpu_threshold_spin.setValue(
            int(
                getattr(
                    self.sketch_view,
                    "gpu_point_preview_threshold",
                    GPU_POINT_PREVIEW_AUTO_THRESHOLD,
                )
            )
        )
        self.gpu_threshold_spin.valueChanged.connect(self._update_gpu_preview)
        self.gpu_threshold_spin.setToolTip("Minimum particles required to auto-enable GPU preview.")
        gpu_row.addWidget(self.gpu_threshold_spin)
        gpu_row.addStretch(1)
        display_group_layout.addLayout(gpu_row)

        reset_row = QHBoxLayout()
        self.reset_preview_btn = QPushButton("Reset Preview View")
        self.reset_preview_btn.clicked.connect(self._reset_point_preview)
        reset_row.addWidget(self.reset_preview_btn)
        reset_row.addStretch(1)
        display_group_layout.addLayout(reset_row)

        self.stats_label = QLabel("Particles: -- | Connections: --")
        self.stats_label.setObjectName("MinorStatusLabel")
        summary_layout.addWidget(self.stats_label)

        self.mesh_qa_label = QLabel(
            "Boundary spacing: --\nInteraction spacing: --"
        )
        self.mesh_qa_label.setWordWrap(True)
        self.mesh_qa_label.setObjectName("MinorStatusLabel")
        summary_layout.addWidget(self.mesh_qa_label)

        self.backend_hint = QLabel("")
        self.backend_hint.setWordWrap(True)
        self.backend_hint.setObjectName("MinorStatusLabel")
        summary_layout.addWidget(self.backend_hint)

        hint = QLabel(
            "Generate will create particles with Poisson disk sampling and then triangulate connections. "
            "Existing particles are reused so node IDs stay consistent."
        )
        hint.setWordWrap(True)
        hint.setObjectName("MinorStatusLabel")
        summary_layout.addWidget(hint)

        self._external_container = QWidget()
        self._external_layout = QVBoxLayout(self._external_container)
        _apply_layout_metrics(
            self._external_layout,
            margins=(0, DOCK_SECTION_SPACING, 0, 0),
            spacing=DOCK_SECTION_SPACING,
        )
        layout.addWidget(self._external_container)

        self._init_backend_combo()
        self._update_sizing_mode()
        self._update_fast_preview()
        self._update_gpu_preview()
        self._update_backend_hint()
        self._update_responsive_layout()
        self.refresh_external_section_visibility()
        _finalize_dock_panel(self)

    def add_external_section(self, widget):
        if widget is None:
            return
        self._external_layout.addWidget(widget)
        self.refresh_external_section_visibility()

    def refresh_external_section_visibility(self):
        has_visible_content = False
        for index in range(self._external_layout.count()):
            item = self._external_layout.itemAt(index)
            widget = item.widget() if item is not None else None
            if widget is not None and not widget.isHidden():
                has_visible_content = True
                break
        self._external_container.setVisible(has_visible_content)

    def generate_particles(self):
        mesh_config = self._build_mesh_config()
        main = self.window()
        execute_command = getattr(main, "execute_app_command", None) if main is not None else None
        if execute_command is None:
            QMessageBox.warning(self, "Particles", "Particle command bus is not available.")
            return
        execute_command(GenerateParticlesCommand(mesh_config=mesh_config))

    def generate_particles_and_connections(self):
        mesh_config = self._build_mesh_config()
        if self.sketch_view.project_mode == "3d":
            self.generate_connections()
            return

        def _finish(success):
            if not success:
                return
            self.sketch_view.set_display_mode("mesh")
            self.show_nodes.blockSignals(True)
            self.show_nodes.setChecked(True)
            self.show_nodes.blockSignals(False)
            self.show_mesh.blockSignals(True)
            self.show_mesh.setChecked(False)
            self.show_mesh.blockSignals(False)
            if hasattr(self, "mesh_view_toggle"):
                self.mesh_view_toggle.blockSignals(True)
                self.mesh_view_toggle.setChecked(True)
                self.mesh_view_toggle.blockSignals(False)
            self._update_stats()
            self._update_display()
            self._sync_point_preview(refresh=True)
            self.generate_connections()

        if not self.sketch_view.run_particle_generation_async(mesh_config=mesh_config, on_done=_finish):
            QMessageBox.warning(self, "Particles", "Particle generation failed.")

    def generate_connections(self):
        mesh_config = self._build_mesh_config()
        main = self.window()
        if main is not None and bool(getattr(main, "_workspace_3d", False)):
            generate_impl = getattr(main, "_generate_gmsh_mesh_impl", None)
            if callable(generate_impl):
                generate_impl()
            return
        def _finish(success):
            if not success:
                return
            if hasattr(self, "mesh_view_toggle"):
                self.mesh_view_toggle.blockSignals(True)
                self.mesh_view_toggle.setChecked(True)
                self.mesh_view_toggle.blockSignals(False)
            self.show_nodes.blockSignals(True)
            self.show_nodes.setChecked(True)
            self.show_nodes.blockSignals(False)
            self.show_mesh.blockSignals(True)
            self.show_mesh.setChecked(True)
            self.show_mesh.blockSignals(False)
            self.sketch_view.set_display_mode("mesh")
            self._update_stats()
            self._update_display()
            self._sync_point_preview(refresh=True)
        if not self.sketch_view.run_cpd_async(preview_only=True, mesh_config=mesh_config, on_done=_finish):
            QMessageBox.warning(self, "Connections", "Connection generation failed.")

    def rebuild_particles(self):
        main = self.window()
        if main is not None and hasattr(main, "statusBar"):
            try:
                main.statusBar().showMessage("Regenerating particles from current geometry...", 3000)
            except Exception:
                pass
        self.generate_particles()

    def preview_mesh(self):
        self.generate_particles()

    def back_to_geometry(self):
        if hasattr(self, "mesh_view_toggle"):
            self.mesh_view_toggle.blockSignals(True)
            self.mesh_view_toggle.setChecked(False)
            self.mesh_view_toggle.blockSignals(False)
        self.sketch_view.set_display_mode("geometry")
        self._sync_point_preview(refresh=False)

    def refresh(self):
        self._update_stats()
        self._update_display()

    def on_mesh_panel_open(self):
        mesh_config = self._build_mesh_config()
        self._update_stats()
        if hasattr(self.sketch_view, "has_current_mesh") and self.sketch_view.has_current_mesh(mesh_config):
            if hasattr(self, "mesh_view_toggle"):
                self.mesh_view_toggle.blockSignals(True)
                self.mesh_view_toggle.setChecked(True)
                self.mesh_view_toggle.blockSignals(False)
            self.show_nodes.blockSignals(True)
            self.show_nodes.setChecked(True)
            self.show_nodes.blockSignals(False)
            self.show_mesh.blockSignals(True)
            self.show_mesh.setChecked(False)
            self.show_mesh.blockSignals(False)
            self.sketch_view.set_display_mode("mesh")
            self._update_display()
            self._sync_point_preview(refresh=True)
            return
        if hasattr(self.sketch_view, "has_current_particle_set") and self.sketch_view.has_current_particle_set(mesh_config):
            if hasattr(self, "mesh_view_toggle"):
                self.mesh_view_toggle.blockSignals(True)
                self.mesh_view_toggle.setChecked(True)
                self.mesh_view_toggle.blockSignals(False)
            self.show_nodes.blockSignals(True)
            self.show_nodes.setChecked(True)
            self.show_nodes.blockSignals(False)
            self.show_mesh.blockSignals(True)
            self.show_mesh.setChecked(False)
            self.show_mesh.blockSignals(False)
            self.sketch_view.set_display_mode("mesh")
            self._update_display()
            self._sync_point_preview(refresh=True)
            return
        if hasattr(self, "mesh_view_toggle"):
            self.mesh_view_toggle.blockSignals(True)
            self.mesh_view_toggle.setChecked(False)
            self.mesh_view_toggle.blockSignals(False)
        self.sketch_view.set_display_mode("geometry")
        self._update_display()
        self._sync_point_preview(refresh=False)

    def _set_mesh_view_mode(self, enabled):
        enabled = bool(enabled)
        has_particles = False
        has_mesh = False
        if hasattr(self.sketch_view, "global_nodes") and hasattr(self.sketch_view, "global_elements"):
            nodes = getattr(self.sketch_view, "global_nodes", None)
            elems = getattr(self.sketch_view, "global_elements", None)
            has_particles = nodes is not None and len(nodes) > 0
            has_mesh = has_particles and elems is not None and len(elems) > 0
        if enabled and not has_particles:
            self.mesh_view_toggle.blockSignals(True)
            self.mesh_view_toggle.setChecked(False)
            self.mesh_view_toggle.blockSignals(False)
            return
        if enabled:
            self.show_nodes.blockSignals(True)
            self.show_nodes.setChecked(True)
            self.show_nodes.blockSignals(False)
            if not has_mesh:
                self.show_mesh.blockSignals(True)
                self.show_mesh.setChecked(False)
                self.show_mesh.blockSignals(False)
        self.sketch_view.set_display_mode("mesh" if enabled else "geometry")
        self._update_display()
        self._sync_point_preview(refresh=enabled)

    def _update_display(self):
        if self.sketch_view.project_mode == "3d":
            main = self.window()
            if main and hasattr(main, "_call_view_3d"):
                main._call_view_3d(
                    "set_visibility",
                    show_nodes=self.show_nodes.isChecked(),
                    show_mesh=self.show_mesh.isChecked(),
                )
        else:
            self.sketch_view.set_mesh_view_visibility(
                show_nodes=self.show_nodes.isChecked(),
                show_mesh=self.show_mesh.isChecked(),
            )
        self._sync_point_preview(refresh=False)

    def _update_fast_preview(self):
        enabled = self.fast_preview_check.isChecked()
        if hasattr(self, "fast_preview_limit"):
            self.fast_preview_limit.setEnabled(enabled)
            limit = int(self.fast_preview_limit.value())
        else:
            limit = int(FAST_PREVIEW_CONNECTION_LIMIT)
        if hasattr(self.sketch_view, "set_fast_preview"):
            self.sketch_view.set_fast_preview(enabled=enabled, limit=limit)
        self._sync_point_preview(refresh=False)

    def _update_gpu_preview(self):
        enabled = self.gpu_preview_check.isChecked()
        auto = self.gpu_auto_check.isChecked()
        threshold = int(self.gpu_threshold_spin.value())
        self.gpu_auto_check.setEnabled(enabled)
        self.gpu_threshold_spin.setEnabled(enabled and auto)
        if hasattr(self.sketch_view, "set_gpu_point_preview_settings"):
            self.sketch_view.set_gpu_point_preview_settings(
                enabled=enabled,
                auto=auto,
                threshold=threshold,
            )
        elif hasattr(self.sketch_view, "set_gpu_point_preview"):
            self.sketch_view.set_gpu_point_preview(enabled=enabled)
        self._sync_point_preview(refresh=False)

    def _sync_point_preview(self, refresh=False):
        main = self.window()
        if not main or main is self:
            return
        if hasattr(main, "_sync_point_preview"):
            try:
                main._sync_point_preview(refresh_data=refresh)
            except TypeError:
                main._sync_point_preview(refresh)

    def _reset_point_preview(self):
        main = self.window()
        if not main or main is self:
            return
        if hasattr(main, "_reset_point_preview_view"):
            main._reset_point_preview_view()

    def _interface_preview_qcolor(self):
        color = getattr(self.sketch_view, "interface_preview_color", QColor(70, 150, 255))
        if not isinstance(color, QColor):
            color = QColor(70, 150, 255)
        if not color.isValid():
            color = QColor(70, 150, 255)
        return QColor(color)

    def _update_interface_color_button(self):
        if not hasattr(self, "interface_color_btn"):
            return
        color = self._interface_preview_qcolor()
        text_color = "#000000" if color.lightness() > 140 else "#ffffff"
        self.interface_color_btn.setStyleSheet(
            "QPushButton {"
            f"background: {color.name()}; color: {text_color}; border: 1px solid #666; padding: 2px 8px;"
            "}"
        )
        self.interface_color_btn.setToolTip(f"Current interaction preview color: {color.name().upper()}")

    def _update_responsive_layout(self):
        width = int(self.contentsRect().width() or self.width() or 0)
        _reflow_button_grid(
            self._generation_button_grid,
            self._generation_action_buttons,
            width - 24,
            min_button_width=124,
            max_columns=3,
        )
        compact = width < 430
        icon_only = width < 350
        _set_responsive_button_text(
            self.generate_btn,
            full=self._generate_label_full,
            compact="Generate",
            icon_only=icon_only,
        )
        _set_responsive_button_text(
            self.connections_btn,
            full=self._connections_label_full,
            compact="Connect",
            icon_only=icon_only,
        )
        _set_responsive_button_text(
            self.back_btn,
            full=self._back_label_full,
            compact="Geom",
            icon_only=icon_only,
        )
        _set_responsive_button_text(
            self.reset_preview_btn,
            full=self._reset_preview_label_full,
            compact="Reset",
            icon_only=icon_only,
        )
        row_wrap = QFormLayout.WrapLongRows if width < 360 else QFormLayout.DontWrapRows
        self.generation_form.setRowWrapPolicy(row_wrap)
        self.display_form.setRowWrapPolicy(row_wrap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_responsive_layout()

    def _apply_interface_preview_color(self, color, persist=True):
        if not isinstance(color, QColor) or not color.isValid():
            return
        if hasattr(self.sketch_view, "set_interface_preview_color"):
            self.sketch_view.set_interface_preview_color(color)
        else:
            self.sketch_view.interface_preview_color = QColor(color)
            self.sketch_view.redraw()
        self._update_interface_color_button()
        if persist and hasattr(self, "_settings"):
            self._settings.setValue("mesh/interface_preview_color", color.name())

    def _load_interface_preview_color_setting(self):
        raw = None
        if hasattr(self, "_settings"):
            raw = self._settings.value("mesh/interface_preview_color", "#4696ff", type=str)
        color = QColor(raw or "#4696ff")
        if not color.isValid():
            color = QColor(70, 150, 255)
        self._apply_interface_preview_color(color, persist=False)

    def _pick_interface_preview_color(self):
        current = self._interface_preview_qcolor()
        color = QColorDialog.getColor(current, self, "Pick Interaction Preview Color")
        if not color.isValid():
            return
        self._apply_interface_preview_color(color, persist=True)

    def _apply_mesh_preview_style(self, *, line_width=None, particle_size=None, persist=True):
        if hasattr(self.sketch_view, "set_mesh_preview_style"):
            self.sketch_view.set_mesh_preview_style(line_width=line_width, particle_size=particle_size)
        else:
            if line_width is not None:
                self.sketch_view.mesh_preview_line_width = float(line_width)
            if particle_size is not None:
                self.sketch_view.mesh_preview_particle_size = float(particle_size)
            self.sketch_view.redraw()
        if persist and hasattr(self, "_settings"):
            if line_width is not None:
                self._settings.setValue("mesh/preview_line_thickness", float(line_width))
            if particle_size is not None:
                self._settings.setValue("mesh/preview_particle_size", float(particle_size))

    def _load_mesh_preview_style_settings(self):
        line_width = self._settings.value(
            "mesh/preview_line_thickness",
            float(getattr(self.sketch_view, "mesh_preview_line_width", 1.0)),
            type=float,
        )
        particle_size = self._settings.value(
            "mesh/preview_particle_size",
            float(getattr(self.sketch_view, "mesh_preview_particle_size", 3.0)),
            type=float,
        )
        if hasattr(self, "conn_line_spin"):
            self.conn_line_spin.blockSignals(True)
            self.conn_line_spin.setValue(float(line_width))
            self.conn_line_spin.blockSignals(False)
        if hasattr(self, "particle_size_spin"):
            self.particle_size_spin.blockSignals(True)
            self.particle_size_spin.setValue(float(particle_size))
            self.particle_size_spin.blockSignals(False)
        self._apply_mesh_preview_style(
            line_width=float(line_width),
            particle_size=float(particle_size),
            persist=False,
        )

    def _update_mesh_preview_style(self):
        self._apply_mesh_preview_style(
            line_width=float(self.conn_line_spin.value()),
            particle_size=float(self.particle_size_spin.value()),
            persist=True,
        )

    def sync_preview_settings(self):
        if hasattr(self, "fast_preview_check"):
            self.fast_preview_check.blockSignals(True)
            self.fast_preview_check.setChecked(
                bool(getattr(self.sketch_view, "fast_preview_enabled", FAST_PREVIEW_ENABLED))
            )
            self.fast_preview_check.blockSignals(False)
        if hasattr(self, "fast_preview_limit"):
            self.fast_preview_limit.blockSignals(True)
            self.fast_preview_limit.setValue(
                int(
                    getattr(
                        self.sketch_view,
                        "fast_preview_connection_limit",
                        FAST_PREVIEW_CONNECTION_LIMIT,
                    )
                )
            )
            self.fast_preview_limit.blockSignals(False)
            self.fast_preview_limit.setEnabled(self.fast_preview_check.isChecked())
        if hasattr(self, "gpu_preview_check"):
            self.gpu_preview_check.blockSignals(True)
            self.gpu_preview_check.setChecked(
                bool(getattr(self.sketch_view, "gpu_point_preview_enabled", GPU_POINT_PREVIEW_ENABLED))
            )
            self.gpu_preview_check.blockSignals(False)
        if hasattr(self, "gpu_auto_check"):
            self.gpu_auto_check.blockSignals(True)
            self.gpu_auto_check.setChecked(
                bool(getattr(self.sketch_view, "gpu_point_preview_auto", GPU_POINT_PREVIEW_AUTO_ENABLED))
            )
            self.gpu_auto_check.blockSignals(False)
        if hasattr(self, "gpu_threshold_spin"):
            self.gpu_threshold_spin.blockSignals(True)
            self.gpu_threshold_spin.setValue(
                int(
                    getattr(
                        self.sketch_view,
                        "gpu_point_preview_threshold",
                        GPU_POINT_PREVIEW_AUTO_THRESHOLD,
                    )
                )
            )
            self.gpu_threshold_spin.blockSignals(False)
            self.gpu_threshold_spin.setEnabled(
                self.gpu_preview_check.isChecked() and self.gpu_auto_check.isChecked()
            )
        if hasattr(self, "interface_color_btn"):
            self._update_interface_color_button()
        if hasattr(self, "conn_line_spin") and hasattr(self, "particle_size_spin"):
            self._load_mesh_preview_style_settings()

    def _update_sizing_mode(self):
        by_spacing = self.sizing_combo.currentIndex() == 0
        self.dx_spin.setVisible(by_spacing)
        self.nodes_spin.setVisible(not by_spacing)

    def _init_backend_combo(self):
        self.backend_combo.clear()
        status = {}
        if hasattr(self.sketch_view, "get_mesh_backend_status"):
            status = self.sketch_view.get_mesh_backend_status()
        entries = [
            ("auto", "Auto (fastest available)"),
            ("triangle", "Triangle (fastest 2D)"),
            ("gmsh", "Gmsh"),
            ("gmsh-2d-adaptive", "Gmsh (adaptive multi-part)"),
            ("pygalmesh", "CGAL/pygalmesh"),
            ("scipy", "SciPy Delaunay (legacy)"),
        ]
        for key, label in entries:
            available = status.get(key, key in ("auto", "scipy"))
            text = label if available else f"{label} (missing)"
            self.backend_combo.addItem(text, key)
        current = getattr(self.sketch_view, "mesh_backend", "auto")
        idx = self.backend_combo.findData(current)
        self.backend_combo.setCurrentIndex(idx if idx >= 0 else 0)
        try:
            self.backend_combo.currentIndexChanged.connect(self._update_adaptive_sizing_visibility)
        except Exception:
            pass
        self._update_adaptive_sizing_visibility()
        # Sync the per-seed list and canvas highlight with any seeds restored
        # from the project file (or pre-existing in project_state).
        self._rebuild_seed_list()
        self._sync_seed_highlight()
        self._rebuild_template_list()
        self._rebuild_pair_list()
        self._rebuild_vertex_seed_list()
        self._sync_vertex_seed_highlight()
        self._rebuild_bl_list()
        # Part Seeds: build once now, then rebuild when parts change.
        self._rebuild_part_seeds_list()
        try:
            self.sketch_view.partsChanged.connect(self._rebuild_part_seeds_list)
        except Exception:
            pass

    # ----- Edge seeding (Abaqus-style local seeds) -----

    def _begin_edge_seed_pick(self, mode):
        if not hasattr(self.sketch_view, "begin_edge_seed_pick"):
            QMessageBox.information(
                self, "Not available",
                "Edge picking is not supported in the current canvas mode.",
            )
            return
        # Make sure the canvas is showing geometry (not the mesh triangles)
        # so the user sees clean edges to pick. The display_mode flip is
        # purely visual; project state is unchanged.
        try:
            if getattr(self.sketch_view, "display_mode", "") not in ("geometry", "sketch_edit"):
                self.sketch_view.display_mode = "geometry"
                self.sketch_view.redraw()
        except Exception:
            pass

        def _on_pick_done(edge_refs):
            if not edge_refs:
                return  # user canceled or picked nothing
            dlg = LocalSeedsDialog(parent=self.window(), edge_refs=edge_refs, sketch_view=self.sketch_view)
            dlg.applied.connect(self._on_seed_applied)
            dlg.saveTemplateRequested.connect(self._on_template_save_from_dialog)
            dlg.exec()

        try:
            self.sketch_view.begin_edge_seed_pick(mode=mode, on_complete=_on_pick_done)
        except Exception as exc:
            QMessageBox.warning(self, "Edge pick failed", str(exc))

    def _ensure_geometry_view(self):
        try:
            if getattr(self.sketch_view, "display_mode", "") not in ("geometry", "sketch_edit"):
                self.sketch_view.display_mode = "geometry"
                self.sketch_view.redraw()
        except Exception:
            pass

    def _open_local_seeds_for(self, edge_refs):
        if not edge_refs:
            QMessageBox.information(self, "Empty selection", "No edges matched.")
            return
        dlg = LocalSeedsDialog(parent=self.window(), edge_refs=edge_refs)
        dlg.applied.connect(self._on_seed_applied)
        dlg.exec()

    def _begin_edge_seed_by_chain(self):
        """Pick one edge; auto-grow along the connected smooth chain."""
        if not hasattr(self.sketch_view, "begin_edge_seed_pick"):
            return
        self._ensure_geometry_view()

        def _on_done(edge_refs):
            if not edge_refs:
                return
            seed_ref = edge_refs[0]
            try:
                chain = self.sketch_view.grow_chain_from_edge(seed_ref)
            except Exception as exc:
                QMessageBox.warning(self, "Chain selection failed", str(exc))
                return
            self._open_local_seeds_for(chain)

        try:
            self.sketch_view.begin_edge_seed_pick(mode="single", on_complete=_on_done)
        except Exception as exc:
            QMessageBox.warning(self, "Edge pick failed", str(exc))

    def _begin_edge_seed_by_part(self):
        """Pick one edge; select every boundary edge of the same part."""
        if not hasattr(self.sketch_view, "begin_edge_seed_pick"):
            return
        self._ensure_geometry_view()

        def _on_done(edge_refs):
            if not edge_refs:
                return
            pid = int(edge_refs[0].get("part_id", 0))
            try:
                refs = self.sketch_view.edges_of_part(pid)
            except Exception as exc:
                QMessageBox.warning(self, "Part selection failed", str(exc))
                return
            self._open_local_seeds_for(refs)

        try:
            self.sketch_view.begin_edge_seed_pick(mode="single", on_complete=_on_done)
        except Exception as exc:
            QMessageBox.warning(self, "Edge pick failed", str(exc))

    def _begin_edge_seed_by_length(self):
        """No canvas pick — just filter by an edge-length range."""
        # Tiny range dialog (built inline to avoid another QDialog subclass).
        dlg = QDialog(self.window())
        dlg.setWindowTitle("Select Edges by Length")
        dlg.setModal(True)
        v = QVBoxLayout(dlg)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)
        v.addWidget(QLabel("Pick all edges whose length lies in the range:"))
        form = QFormLayout()
        min_spin = QDoubleSpinBox()
        min_spin.setRange(0.0, 1e9); min_spin.setDecimals(4); min_spin.setValue(0.0)
        min_spin.setSpecialValueText("no lower bound")
        max_spin = QDoubleSpinBox()
        max_spin.setRange(0.0, 1e9); max_spin.setDecimals(4); max_spin.setValue(0.0)
        max_spin.setSpecialValueText("no upper bound")
        form.addRow("Min length:", min_spin)
        form.addRow("Max length:", max_spin)
        v.addLayout(form)
        count_lbl = QLabel("Matched: ?")
        count_lbl.setObjectName("MinorStatusLabel")
        v.addWidget(count_lbl)

        def _refresh_count():
            mn = min_spin.value() if min_spin.value() > 0 else None
            mx = max_spin.value() if max_spin.value() > 0 else None
            try:
                refs = self.sketch_view.edges_in_length_range(mn, mx)
            except Exception:
                refs = []
            count_lbl.setText(f"Matched: {len(refs)} edge(s)")
            dlg._matched_refs = refs

        min_spin.valueChanged.connect(_refresh_count)
        max_spin.valueChanged.connect(_refresh_count)
        _refresh_count()

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("Use selection")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)
        btn_row.addStretch(1)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        v.addLayout(btn_row)
        if dlg.exec() != QDialog.Accepted:
            return
        refs = getattr(dlg, "_matched_refs", [])
        self._open_local_seeds_for(refs)

    # ----- Matched edge pairs (gmsh setPeriodic) -----

    def _begin_match_pair_pick(self):
        if not hasattr(self.sketch_view, "begin_edge_seed_pick"):
            return
        self._ensure_geometry_view()

        def _on_done(edge_refs):
            if len(edge_refs) != 2:
                QMessageBox.information(
                    self, "Wrong selection",
                    f"Pick exactly two edges (got {len(edge_refs)}). The first is the master, "
                    "the second is the slave whose mesh will mirror the master's.",
                )
                return
            from models import MatchedEdgePair
            pair = MatchedEdgePair(master=edge_refs[0], slave=edge_refs[1])
            if not pair.is_valid():
                QMessageBox.warning(self, "Invalid pair", "Could not form a valid pair from the selection.")
                return
            pairs = getattr(self.project_state, "matched_edge_pairs", None)
            if pairs is None:
                pairs = []
                self.project_state.matched_edge_pairs = pairs
            pairs.append(pair)
            self._rebuild_pair_list()

        try:
            self.sketch_view.begin_edge_seed_pick(mode="multi", on_complete=_on_done)
        except Exception as exc:
            QMessageBox.warning(self, "Edge pick failed", str(exc))

    def _remove_pair_at(self, index):
        pairs = getattr(self.project_state, "matched_edge_pairs", None) or []
        if 0 <= index < len(pairs):
            del pairs[index]
            self._rebuild_pair_list()

    def _clear_all_pairs(self):
        if not getattr(self.project_state, "matched_edge_pairs", None):
            return
        self.project_state.matched_edge_pairs = []
        self._rebuild_pair_list()

    def _pair_row_label(self, index, pair):
        m, s = pair.master, pair.slave
        return (
            f"Pair {index + 1}: master ({m['start'][0]:g}, {m['start'][1]:g})→({m['end'][0]:g}, {m['end'][1]:g}) | "
            f"slave ({s['start'][0]:g}, {s['start'][1]:g})→({s['end'][0]:g}, {s['end'][1]:g})"
        )

    def _rebuild_pair_list(self):
        if not hasattr(self, "pairs_list_layout"):
            return
        while self.pairs_list_layout.count():
            item = self.pairs_list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._pair_row_widgets = []
        pairs = getattr(self.project_state, "matched_edge_pairs", None) or []
        self.pairs_empty_label.setVisible(len(pairs) == 0)
        for i, pair in enumerate(pairs):
            row = QWidget()
            l = QHBoxLayout(row)
            l.setContentsMargins(0, 0, 0, 0)
            l.setSpacing(DOCK_ROW_SPACING)
            label = QLabel(self._pair_row_label(i, pair))
            label.setWordWrap(True)
            l.addWidget(label, 1)
            rem_btn = QPushButton("Remove")
            rem_btn.clicked.connect(lambda _c=False, idx=i: self._remove_pair_at(idx))
            l.addWidget(rem_btn)
            self.pairs_list_layout.addWidget(row)
            self._pair_row_widgets.append(row)

    # ----- Edge seed templates -----

    def _on_template_save_from_dialog(self, template_seed, name):
        """Receive a saved template from LocalSeedsDialog and store it."""
        tlist = getattr(self.project_state, "edge_seed_templates", None)
        if tlist is None:
            tlist = []
            self.project_state.edge_seed_templates = tlist
        # Replace existing template with same name (idempotent saves).
        replaced = False
        for i, t in enumerate(tlist):
            if str(getattr(t, "set_name", "")) == name:
                tlist[i] = template_seed
                replaced = True
                break
        if not replaced:
            tlist.append(template_seed)
        self._rebuild_template_list()

    def _apply_template(self, template_index):
        tlist = getattr(self.project_state, "edge_seed_templates", None) or []
        if not (0 <= template_index < len(tlist)):
            return
        template = tlist[template_index]
        if not hasattr(self.sketch_view, "begin_edge_seed_pick"):
            return
        self._ensure_geometry_view()

        def _on_done(edge_refs):
            if not edge_refs:
                return
            # Build a fresh EdgeSeed from the template's config + picked edges.
            new_seed = EdgeSeed(
                edge_refs=edge_refs,
                method=template.method,
                bias=template.bias,
                flip_bias=template.flip_bias,
                element_size=template.element_size,
                min_size=template.min_size,
                max_size=template.max_size,
                seed_count=template.seed_count,
                bias_ratio=template.bias_ratio,
                curvature_control=template.curvature_control,
                max_deviation_factor=template.max_deviation_factor,
                min_size_factor=template.min_size_factor,
                set_name=str(getattr(template, "set_name", "")),
                propagate_to_neighbors=template.propagate_to_neighbors,
            )
            if not new_seed.is_valid():
                QMessageBox.warning(self, "Template apply failed",
                                    "Template is not valid for the picked edges.")
                return
            self._on_seed_applied(new_seed)

        try:
            self.sketch_view.begin_edge_seed_pick(mode="multi", on_complete=_on_done)
        except Exception as exc:
            QMessageBox.warning(self, "Edge pick failed", str(exc))

    def _remove_template_at(self, index):
        tlist = getattr(self.project_state, "edge_seed_templates", None) or []
        if 0 <= index < len(tlist):
            del tlist[index]
            self._rebuild_template_list()

    def _template_summary(self, t):
        """One-line summary of a template's settings."""
        if t.method == "by_size":
            if t.bias == "none":
                detail = f"by size = {t.element_size:g}"
            else:
                detail = f"by size {t.min_size:g}–{t.max_size:g} ({t.bias})"
        else:
            if t.bias == "none":
                detail = f"by count = {t.seed_count}"
            else:
                detail = f"by count = {t.seed_count}, ratio {t.bias_ratio:g} ({t.bias})"
        if t.propagate_to_neighbors:
            detail += ", propagated"
        return detail

    def _rebuild_template_list(self):
        if not hasattr(self, "templates_list_layout"):
            return
        while self.templates_list_layout.count():
            item = self.templates_list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._template_row_widgets = []
        templates = getattr(self.project_state, "edge_seed_templates", None) or []
        self.templates_empty_label.setVisible(len(templates) == 0)
        for i, t in enumerate(templates):
            row = QWidget()
            row_l = QHBoxLayout(row)
            row_l.setContentsMargins(0, 0, 0, 0)
            row_l.setSpacing(DOCK_ROW_SPACING)
            name = str(getattr(t, "set_name", "") or f"Template {i + 1}")
            label = QLabel(f"{name} — {self._template_summary(t)}")
            label.setWordWrap(True)
            row_l.addWidget(label, 1)
            apply_btn = QPushButton("Apply to edges...")
            apply_btn.setToolTip("Pick one or more edges and create a seed from this template.")
            apply_btn.clicked.connect(lambda _c=False, idx=i: self._apply_template(idx))
            row_l.addWidget(apply_btn)
            rem_btn = QPushButton("Remove")
            rem_btn.clicked.connect(lambda _c=False, idx=i: self._remove_template_at(idx))
            row_l.addWidget(rem_btn)
            self.templates_list_layout.addWidget(row)
            self._template_row_widgets.append(row)

    def _on_seed_applied(self, seed):
        seeds = getattr(self.project_state, "edge_seeds", None)
        if seeds is None:
            seeds = []
            self.project_state.edge_seeds = seeds
        seeds.append(seed)
        self._rebuild_seed_list()
        self._sync_seed_highlight()

    def _on_seed_updated(self, index, updated_seed):
        """Replace the seed at `index` with the dialog's new version."""
        seeds = getattr(self.project_state, "edge_seeds", None) or []
        if 0 <= index < len(seeds):
            seeds[index] = updated_seed
            self._rebuild_seed_list()
            self._sync_seed_highlight()

    def _edit_seed_at(self, index):
        seeds = getattr(self.project_state, "edge_seeds", None) or []
        if not (0 <= index < len(seeds)):
            return
        seed = seeds[index]
        dlg = LocalSeedsDialog(
            parent=self.window(),
            edge_refs=list(seed.edge_refs or []),
            initial_seed=seed,
            sketch_view=self.sketch_view,
        )
        dlg.applied.connect(lambda updated, idx=index: self._on_seed_updated(idx, updated))
        dlg.saveTemplateRequested.connect(self._on_template_save_from_dialog)
        dlg.exec()

    def _remove_seed_at(self, index):
        seeds = getattr(self.project_state, "edge_seeds", None) or []
        if 0 <= index < len(seeds):
            del seeds[index]
            self._rebuild_seed_list()
            self._sync_seed_highlight()

    def _clear_all_edge_seeds(self):
        if not getattr(self.project_state, "edge_seeds", None):
            return
        self.project_state.edge_seeds = []
        self._rebuild_seed_list()
        self._sync_seed_highlight()

    def _sync_seed_highlight(self):
        if not hasattr(self.sketch_view, "set_edge_seed_highlight"):
            return
        seeds = getattr(self.project_state, "edge_seeds", None) or []
        refs = []
        for s in seeds:
            refs.extend(getattr(s, "edge_refs", []) or [])
        try:
            self.sketch_view.set_edge_seed_highlight(refs)
        except Exception:
            pass

    def _seed_row_label(self, index, seed):
        name = (getattr(seed, "set_name", "") or f"Seed {index + 1}").strip()
        n_edges = len(getattr(seed, "edge_refs", []) or [])
        method = "size" if seed.method == "by_size" else "count"
        if method == "size":
            if seed.bias == "none":
                detail = f"by size = {seed.element_size:g}"
            else:
                detail = f"by size {seed.min_size:g}…{seed.max_size:g} ({seed.bias})"
                if seed.flip_bias:
                    detail += ", flipped"
        else:
            if seed.bias == "none":
                detail = f"by count = {seed.seed_count}"
            else:
                detail = f"by count = {seed.seed_count}, ratio = {seed.bias_ratio:g} ({seed.bias})"
                if seed.flip_bias:
                    detail += ", flipped"
        edge_word = "edge" if n_edges == 1 else "edges"
        return f"{name} — {detail} — {n_edges} {edge_word}"

    # ----- Face partition (split a part with a line) -----

    def _begin_partition_pick(self):
        if not hasattr(self.sketch_view, "begin_partition_pick"):
            QMessageBox.information(
                self, "Not available",
                "Partitioning is not supported in the current canvas mode.",
            )
            return
        # The partition mutates geometry; force the canvas into geometry mode
        # so the user can see what they're cutting.
        try:
            if getattr(self.sketch_view, "display_mode", "") not in ("geometry", "sketch_edit"):
                self.sketch_view.display_mode = "geometry"
                self.sketch_view.redraw()
        except Exception:
            pass

        def _on_done(new_part_ids):
            if not new_part_ids:
                self.partition_status_label.setText("Last partition: no change.")
                return
            count = len(new_part_ids)
            self.partition_status_label.setText(
                f"Last partition: split into {count} sub-parts (ids: {new_part_ids})."
            )
            # Partition created new parts → Part Seeds list needs to refresh.
            try:
                self._rebuild_part_seeds_list()
            except Exception:
                pass

        try:
            self.sketch_view.begin_partition_pick(on_complete=_on_done)
        except Exception as exc:
            QMessageBox.warning(self, "Partition failed", str(exc))

    # ----- Boundary layer seeds (CFD inflation) -----

    def _begin_bl_pick(self):
        if not hasattr(self.sketch_view, "begin_edge_seed_pick"):
            QMessageBox.information(
                self, "Not available",
                "Edge picking is not supported in the current canvas mode.",
            )
            return
        try:
            if getattr(self.sketch_view, "display_mode", "") not in ("geometry", "sketch_edit"):
                self.sketch_view.display_mode = "geometry"
                self.sketch_view.redraw()
        except Exception:
            pass

        def _on_done(edge_refs):
            if not edge_refs:
                return
            dlg = BoundaryLayerDialog(parent=self.window(), edge_refs=edge_refs)
            dlg.applied.connect(self._on_bl_applied)
            dlg.exec()

        try:
            self.sketch_view.begin_edge_seed_pick(mode="multi", on_complete=_on_done)
        except Exception as exc:
            QMessageBox.warning(self, "Edge pick failed", str(exc))

    def _on_bl_applied(self, seed):
        seeds = getattr(self.project_state, "boundary_layer_seeds", None)
        if seeds is None:
            seeds = []
            self.project_state.boundary_layer_seeds = seeds
        seeds.append(seed)
        self._rebuild_bl_list()

    def _on_bl_updated(self, index, updated):
        seeds = getattr(self.project_state, "boundary_layer_seeds", None) or []
        if 0 <= index < len(seeds):
            seeds[index] = updated
            self._rebuild_bl_list()

    def _edit_bl_at(self, index):
        seeds = getattr(self.project_state, "boundary_layer_seeds", None) or []
        if not (0 <= index < len(seeds)):
            return
        seed = seeds[index]
        dlg = BoundaryLayerDialog(parent=self.window(), edge_refs=list(seed.edge_refs or []), initial_seed=seed)
        dlg.applied.connect(lambda updated, idx=index: self._on_bl_updated(idx, updated))
        dlg.exec()

    def _remove_bl_at(self, index):
        seeds = getattr(self.project_state, "boundary_layer_seeds", None) or []
        if 0 <= index < len(seeds):
            del seeds[index]
            self._rebuild_bl_list()

    def _clear_all_bl_seeds(self):
        if not getattr(self.project_state, "boundary_layer_seeds", None):
            return
        self.project_state.boundary_layer_seeds = []
        self._rebuild_bl_list()

    def _bl_row_label(self, index, seed):
        name = (getattr(seed, "set_name", "") or f"Boundary Layer {index + 1}").strip()
        n_edges = len(seed.edge_refs or [])
        total = seed.total_thickness()
        return (
            f"{name} — {seed.num_layers} layers × {seed.first_layer_size:g}, "
            f"ratio {seed.growth_ratio:g}, total ~{total:.3g}  ({n_edges} edge{'s' if n_edges != 1 else ''})"
        )

    def _rebuild_bl_list(self):
        if not hasattr(self, "bl_list_layout"):
            return
        while self.bl_list_layout.count():
            item = self.bl_list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._bl_row_widgets = []
        seeds = getattr(self.project_state, "boundary_layer_seeds", None) or []
        self.bl_empty_label.setVisible(len(seeds) == 0)
        for i, seed in enumerate(seeds):
            row = QWidget()
            l = QHBoxLayout(row)
            l.setContentsMargins(0, 0, 0, 0)
            l.setSpacing(DOCK_ROW_SPACING)
            label = QLabel(self._bl_row_label(i, seed))
            label.setWordWrap(True)
            l.addWidget(label, 1)
            edit_btn = QPushButton("Edit")
            edit_btn.clicked.connect(lambda _c=False, idx=i: self._edit_bl_at(idx))
            l.addWidget(edit_btn)
            rem_btn = QPushButton("Remove")
            rem_btn.clicked.connect(lambda _c=False, idx=i: self._remove_bl_at(idx))
            l.addWidget(rem_btn)
            self.bl_list_layout.addWidget(row)
            self._bl_row_widgets.append(row)

    # ----- Vertex seeds (point-anchored refinement) -----

    def _begin_vertex_seed_pick(self):
        if not hasattr(self.sketch_view, "begin_vertex_seed_pick"):
            QMessageBox.information(
                self, "Not available",
                "Vertex picking is not supported in the current canvas mode.",
            )
            return
        try:
            if getattr(self.sketch_view, "display_mode", "") not in ("geometry", "sketch_edit"):
                self.sketch_view.display_mode = "geometry"
                self.sketch_view.redraw()
        except Exception:
            pass

        def _on_pick_done(pid_vertex):
            if pid_vertex is None:
                return
            pid, vpt = pid_vertex
            dlg = VertexSeedDialog(parent=self.window(), part_id=pid, point=vpt)
            dlg.applied.connect(self._on_vertex_seed_applied)
            dlg.exec()

        try:
            self.sketch_view.begin_vertex_seed_pick(on_complete=_on_pick_done)
        except Exception as exc:
            QMessageBox.warning(self, "Vertex pick failed", str(exc))

    def _on_vertex_seed_applied(self, seed):
        seeds = getattr(self.project_state, "vertex_seeds", None)
        if seeds is None:
            seeds = []
            self.project_state.vertex_seeds = seeds
        seeds.append(seed)
        self._rebuild_vertex_seed_list()
        self._sync_vertex_seed_highlight()

    def _on_vertex_seed_updated(self, index, updated):
        seeds = getattr(self.project_state, "vertex_seeds", None) or []
        if 0 <= index < len(seeds):
            seeds[index] = updated
            self._rebuild_vertex_seed_list()
            self._sync_vertex_seed_highlight()

    def _edit_vertex_seed_at(self, index):
        seeds = getattr(self.project_state, "vertex_seeds", None) or []
        if not (0 <= index < len(seeds)):
            return
        seed = seeds[index]
        dlg = VertexSeedDialog(
            parent=self.window(),
            part_id=seed.part_id,
            point=seed.point,
            initial_seed=seed,
        )
        dlg.applied.connect(lambda updated, idx=index: self._on_vertex_seed_updated(idx, updated))
        dlg.exec()

    def _remove_vertex_seed_at(self, index):
        seeds = getattr(self.project_state, "vertex_seeds", None) or []
        if 0 <= index < len(seeds):
            del seeds[index]
            self._rebuild_vertex_seed_list()
            self._sync_vertex_seed_highlight()

    def _clear_all_vertex_seeds(self):
        if not getattr(self.project_state, "vertex_seeds", None):
            return
        self.project_state.vertex_seeds = []
        self._rebuild_vertex_seed_list()
        self._sync_vertex_seed_highlight()

    def _sync_vertex_seed_highlight(self):
        if not hasattr(self.sketch_view, "set_vertex_seed_highlight"):
            return
        seeds = getattr(self.project_state, "vertex_seeds", None) or []
        refs = [(int(s.part_id), s.point) for s in seeds]
        try:
            self.sketch_view.set_vertex_seed_highlight(refs)
        except Exception:
            pass

    def _vertex_seed_row_label(self, index, seed):
        name = (getattr(seed, "set_name", "") or f"Vertex {index + 1}").strip()
        x, y = seed.point
        return (
            f"{name} @ ({x:g}, {y:g}) — h={seed.target_size:g}, "
            f"radius={seed.influence_radius:g}"
        )

    def _rebuild_vertex_seed_list(self):
        if not hasattr(self, "vseeds_list_layout"):
            return
        while self.vseeds_list_layout.count():
            item = self.vseeds_list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._vseed_row_widgets = []
        seeds = getattr(self.project_state, "vertex_seeds", None) or []
        self.vseeds_empty_label.setVisible(len(seeds) == 0)
        for i, seed in enumerate(seeds):
            row = QWidget()
            l = QHBoxLayout(row)
            l.setContentsMargins(0, 0, 0, 0)
            l.setSpacing(DOCK_ROW_SPACING)
            label = QLabel(self._vertex_seed_row_label(i, seed))
            label.setWordWrap(True)
            l.addWidget(label, 1)
            edit_btn = QPushButton("Edit")
            edit_btn.clicked.connect(lambda _c=False, idx=i: self._edit_vertex_seed_at(idx))
            l.addWidget(edit_btn)
            rem_btn = QPushButton("Remove")
            rem_btn.clicked.connect(lambda _c=False, idx=i: self._remove_vertex_seed_at(idx))
            l.addWidget(rem_btn)
            self.vseeds_list_layout.addWidget(row)
            self._vseed_row_widgets.append(row)

    # ----- Per-part h_bulk overrides -----

    def _rebuild_part_seeds_list(self):
        if not hasattr(self, "part_seeds_list_layout"):
            return
        # Clear existing rows.
        while self.part_seeds_list_layout.count():
            item = self.part_seeds_list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._part_seed_row_widgets = []
        parts = [p for p in (getattr(self.project_state, "parts", None) or [])
                 if not getattr(p, "is_void", False)]
        overrides = getattr(self.project_state, "part_mesh_overrides", None) or {}
        self.part_seeds_empty_label.setVisible(len(parts) == 0)
        # Header row labels — keeps the per-part rows aligned & easy to read.
        header = QWidget()
        hl = QHBoxLayout(header)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(DOCK_ROW_SPACING)
        # Tighter columns so the row fits in the narrow right dock without
        # being clipped on the right (Reset button was disappearing).
        for txt, w in (("Part", 60), ("Bulk", 50), ("Feat", 50), ("", 0)):
            lab = QLabel(txt)
            lab.setObjectName("MinorStatusLabel")
            if w:
                lab.setMinimumWidth(w)
            hl.addWidget(lab, 1 if txt == "Part" else 0)
        self.part_seeds_list_layout.addWidget(header)

        for part in parts:
            pid = int(part.id)
            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0)
            row_l.setSpacing(DOCK_ROW_SPACING)
            label = QLabel(getattr(part, "name", f"Part {pid}") or f"Part {pid}")
            label.setMinimumWidth(60)
            label.setMaximumWidth(100)
            label.setToolTip(getattr(part, "name", f"Part {pid}") or f"Part {pid}")
            row_l.addWidget(label, 1)
            override_now = overrides.get(pid) or {}

            bulk_spin = QDoubleSpinBox()
            bulk_spin.setDecimals(4)
            bulk_spin.setRange(0.0, 1e6)
            bulk_spin.setSingleStep(0.1)
            bulk_spin.setSpecialValueText("inherit")
            bulk_spin.setMinimumWidth(50)
            bulk_spin.setMaximumWidth(80)
            try:
                bulk_spin.setValue(max(0.0, float(override_now.get("h_bulk", 0.0))))
            except Exception:
                bulk_spin.setValue(0.0)
            bulk_spin.setToolTip(
                "Bulk element size for this part. 0 = inherit global bulk size. "
                "Can be set finer OR coarser than the global value — both are "
                "honored exactly inside this part."
            )
            bulk_spin.valueChanged.connect(
                lambda v, p_id=pid: self._on_part_size_changed(p_id, "h_bulk", v)
            )
            row_l.addWidget(bulk_spin)

            feat_spin = QDoubleSpinBox()
            feat_spin.setDecimals(4)
            feat_spin.setRange(0.0, 1e6)
            feat_spin.setSingleStep(0.05)
            feat_spin.setSpecialValueText("inherit")
            feat_spin.setMinimumWidth(50)
            feat_spin.setMaximumWidth(80)
            try:
                feat_spin.setValue(max(0.0, float(override_now.get("h_feature", 0.0))))
            except Exception:
                feat_spin.setValue(0.0)
            feat_spin.setToolTip(
                "Feature (boundary-adjacent) element size for this part. "
                "0 = inherit global h_feature. Smaller than h_bulk gives a "
                "gradient mesh refined toward the part's edges."
            )
            feat_spin.valueChanged.connect(
                lambda v, p_id=pid: self._on_part_size_changed(p_id, "h_feature", v)
            )
            row_l.addWidget(feat_spin)

            reset_btn = QPushButton("↺")
            reset_btn.setToolTip("Reset — clear all overrides for this part (inherit globals).")
            reset_btn.setMaximumWidth(28)
            reset_btn.setMinimumWidth(24)
            reset_btn.clicked.connect(
                lambda _c=False, p_id=pid, b=bulk_spin, f=feat_spin:
                    (b.setValue(0.0), f.setValue(0.0))
            )
            row_l.addWidget(reset_btn, 0)
            self.part_seeds_list_layout.addWidget(row_w)
            self._part_seed_row_widgets.append({
                "part_id": pid,
                "row": row_w,
                "bulk_spin": bulk_spin,
                "feat_spin": feat_spin,
            })
        self._refresh_part_seeds_estimate()

    def _on_part_size_changed(self, part_id: int, key: str, value: float):
        if key not in ("h_bulk", "h_feature"):
            return
        overrides = getattr(self.project_state, "part_mesh_overrides", None)
        if overrides is None:
            overrides = {}
            self.project_state.part_mesh_overrides = overrides
        pid = int(part_id)
        current = dict(overrides.get(pid) or {})
        if value <= 0:
            current.pop(key, None)
        else:
            current[key] = float(value)
        if current:
            overrides[pid] = current
        else:
            overrides.pop(pid, None)
        # Live node-count estimate: cheap math, runs as the user types.
        self._refresh_part_seeds_estimate()

    def _global_h_bulk(self):
        """Return the current global bulk size from the Generation form so
        the live estimate uses what the user actually has dialed in."""
        try:
            if self.sizing_combo.currentIndex() == 0:
                return float(self.dx_spin.value())
            # By total particle count: estimate dx from total area and target count.
            total_area = self._total_parts_area()
            tn = int(self.nodes_spin.value())
            if total_area > 0 and tn > 0:
                import math as _m
                return _m.sqrt(total_area / float(tn))
        except Exception:
            pass
        return float(globals().get("DEFAULT_DX", 1.0) or 1.0)

    def _total_parts_area(self):
        parts = [p for p in (getattr(self.project_state, "parts", None) or [])
                 if not getattr(p, "is_void", False)]
        total = 0.0
        for p in parts:
            geom = getattr(p, "geometry", None)
            if geom is None or getattr(geom, "is_empty", True):
                continue
            try:
                total += float(geom.area)
            except Exception:
                continue
        return total

    def _refresh_part_seeds_estimate(self):
        if not hasattr(self, "part_seeds_estimate_label"):
            return
        parts = [p for p in (getattr(self.project_state, "parts", None) or [])
                 if not getattr(p, "is_void", False)]
        if not parts:
            self.part_seeds_estimate_label.setText("")
            return
        overrides = getattr(self.project_state, "part_mesh_overrides", None) or {}
        global_h = self._global_h_bulk()
        # Crude estimate: each element in a triangle mesh roughly occupies
        # h² * √3 / 4 ≈ 0.433·h² of area, so node count per part ≈ area·2/h²
        # for an equilateral-triangle assumption. We use the simpler
        # area/h² which slightly underestimates but stays consistent across
        # parts so the trend is what matters.
        total = 0.0
        for p in parts:
            geom = getattr(p, "geometry", None)
            if geom is None or getattr(geom, "is_empty", True):
                continue
            try:
                area = float(geom.area)
            except Exception:
                continue
            if area <= 0:
                continue
            override = overrides.get(int(p.id), {}) or {}
            h = float(override.get("h_bulk") or global_h) or global_h
            if h <= 0:
                continue
            total += area / (h * h)
        if total <= 0:
            self.part_seeds_estimate_label.setText("")
            return
        self.part_seeds_estimate_label.setText(
            f"Estimated total: ~{int(round(total)):,} nodes  "
            f"(rough area/h² heuristic — actual count depends on refinement)"
        )

    def _rebuild_seed_list(self):
        if not hasattr(self, "seeds_list_layout"):
            return
        # Clear all existing rows.
        while self.seeds_list_layout.count():
            item = self.seeds_list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._seed_row_widgets = []
        seeds = getattr(self.project_state, "edge_seeds", None) or []
        self.seeds_empty_label.setVisible(len(seeds) == 0)
        for i, seed in enumerate(seeds):
            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0)
            row_l.setSpacing(DOCK_ROW_SPACING)
            label = QLabel(self._seed_row_label(i, seed))
            label.setWordWrap(True)
            row_l.addWidget(label, 1)
            edit_btn = QPushButton("Edit")
            edit_btn.setToolTip("Open the Local Seeds dialog with this seed's values")
            edit_btn.clicked.connect(lambda _c=False, idx=i: self._edit_seed_at(idx))
            row_l.addWidget(edit_btn)
            rem_btn = QPushButton("Remove")
            rem_btn.setToolTip("Delete this seed only (leaves others intact)")
            rem_btn.clicked.connect(lambda _c=False, idx=i: self._remove_seed_at(idx))
            row_l.addWidget(rem_btn)
            self.seeds_list_layout.addWidget(row_w)
            self._seed_row_widgets.append(row_w)

    def _build_mesh_config(self):
        if self.sizing_combo.currentIndex() == 0:
            mode = "dx"
            dx = float(self.dx_spin.value())
            target_nodes = None
        else:
            mode = "count"
            dx = None
            target_nodes = int(self.nodes_spin.value())
        distribution = "global_poisson"
        backend = self.backend_combo.currentData()
        if hasattr(self.sketch_view, "mesh_backend") and backend:
            self.sketch_view.mesh_backend = backend
        config = {
            "mode": mode,
            "dx": dx,
            "target_nodes": target_nodes,
            "distribution": distribution,
            "backend": backend,
        }
        if backend == "gmsh-2d-adaptive":
            try:
                h_bulk = float(self.h_bulk_spin.value())
            except Exception:
                h_bulk = float(dx) if dx else DEFAULT_DX
            if h_bulk <= 0.0:
                h_bulk = float(dx) if dx else DEFAULT_DX
            config["sizing"] = {
                "h_bulk": h_bulk,
                "h_feature": float(self.h_feature_spin.value()),
                "transition_width": float(self.transition_width_spin.value()),
            }
        # Forward any Results-page custom refinement zones the user has drawn.
        zones = getattr(self.project_state, "custom_mesh_zones", None) or []
        if zones:
            config["custom_zones"] = [
                z.to_dict() if hasattr(z, "to_dict") else z for z in zones
            ]
        # Forward Mesh-stage edge seeds (Abaqus-style local seeds).
        seeds = getattr(self.project_state, "edge_seeds", None) or []
        if seeds:
            config["edge_seeds"] = [
                s.to_dict() if hasattr(s, "to_dict") else s for s in seeds
            ]
        # Forward Mesh-stage vertex seeds (point-anchored refinement).
        vseeds = getattr(self.project_state, "vertex_seeds", None) or []
        if vseeds:
            config["vertex_seeds"] = [
                s.to_dict() if hasattr(s, "to_dict") else s for s in vseeds
            ]
        # Forward Mesh-stage boundary-layer seeds (CFD inflation).
        blseeds = getattr(self.project_state, "boundary_layer_seeds", None) or []
        if blseeds:
            config["boundary_layer_seeds"] = [
                s.to_dict() if hasattr(s, "to_dict") else s for s in blseeds
            ]
        # Forward matched edge pairs (gmsh setPeriodic).
        mpairs = getattr(self.project_state, "matched_edge_pairs", None) or []
        if mpairs:
            config["matched_edge_pairs"] = [
                p.to_dict() if hasattr(p, "to_dict") else p for p in mpairs
            ]
        # Forward per-part h_bulk overrides.
        overrides = getattr(self.project_state, "part_mesh_overrides", None) or {}
        if overrides:
            config["part_mesh_overrides"] = {
                int(pid): dict(v) for pid, v in overrides.items()
            }
        return config

    def _update_adaptive_sizing_visibility(self, *args):
        """Show the three sizing rows only when the adaptive gmsh backend is selected."""
        if not hasattr(self, "h_bulk_spin"):
            return
        is_adaptive = self.backend_combo.currentData() == "gmsh-2d-adaptive"
        for spin in (self.h_bulk_spin, self.h_feature_spin, self.transition_width_spin):
            try:
                row_label = self.generation_form.labelForField(spin)
            except Exception:
                row_label = None
            spin.setVisible(is_adaptive)
            if row_label is not None:
                row_label.setVisible(is_adaptive)

    def _update_stats(self):
        mesh_data = self.project_state.mesh_data
        nodes_3d = mesh_data.get("global_nodes_3d")
        elems_3d = mesh_data.get("global_elements_3d")
        nodes_2d = mesh_data.get("global_nodes")
        elems_2d = mesh_data.get("global_elements")
        if self.sketch_view.project_mode == "3d" and isinstance(nodes_3d, np.ndarray) and nodes_3d.size > 0:
            nodes = len(nodes_3d)
            elems = len(elems_3d) if isinstance(elems_3d, np.ndarray) else 0
        else:
            nodes = len(nodes_2d) if isinstance(nodes_2d, np.ndarray) else 0
            elems = len(elems_2d) if isinstance(elems_2d, np.ndarray) else 0
        self.stats_label.setText(f"Particles: {nodes} | Internal connections: {elems}")
        if hasattr(self.sketch_view, "get_mesh_qa_readout"):
            try:
                qa_text = self.sketch_view.get_mesh_qa_readout()
                validation_text = (
                    self.sketch_view.get_mesh_validation_readout()
                    if hasattr(self.sketch_view, "get_mesh_validation_readout")
                    else ""
                )
                details = [text for text in (qa_text, validation_text) if str(text or "").strip()]
                self.mesh_qa_label.setText("\n".join(details))
            except Exception:
                self.mesh_qa_label.setText("Boundary edge spacing: --\nInteraction spacing: --")

    def _update_backend_hint(self):
        dx = None
        target_nodes = None
        if self.sizing_combo.currentIndex() == 0:
            dx = float(self.dx_spin.value())
        else:
            target_nodes = int(self.nodes_spin.value())
        distribution = "global_poisson"
        est_nodes = None
        if hasattr(self.sketch_view, "estimate_mesh_nodes"):
            est_nodes = self.sketch_view.estimate_mesh_nodes(dx, target_nodes, distribution)
        backend = self.backend_combo.currentData()
        rec = None
        if est_nodes is not None:
            if est_nodes >= 200_000:
                rec = "Triangle"
            else:
                rec = "SciPy"
        if backend == "auto":
            if rec:
                self.backend_hint.setText(f"Recommended backend: {rec} (auto picks fastest available).")
            else:
                self.backend_hint.setText("Auto backend picks the fastest available.")
        else:
            if rec:
                self.backend_hint.setText(f"Recommended backend: {rec} for ~{est_nodes:,} particles.")
            else:
                self.backend_hint.setText("")


class InterfacesPanel(QWidget):
    def __init__(self, sketch_view, parent=None, project_state=None):
        super().__init__(parent)
        self.sketch_view = sketch_view
        self.project_state = _resolve_project_state(sketch_view, project_state)
        self._define_label_full = "Add Interaction"
        self._edit_label_full = "Edit"
        self._delete_label_full = "Delete"

        layout = QVBoxLayout(self)
        _apply_layout_metrics(layout)

        layout.addWidget(QLabel("<b>Interactions</b>"))
        self._cross_tab_notice = _make_cross_tab_notice_label(self)
        layout.addWidget(self._cross_tab_notice)

        self._action_button_grid = QGridLayout()
        _apply_layout_metrics(self._action_button_grid, margins=(0, 0, 0, 0), spacing=DOCK_ROW_SPACING)
        self.define_btn = QPushButton("Add Interaction")
        self.define_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.define_btn.clicked.connect(lambda _checked=False: self.define_interface())
        self.define_btn.setIcon(_style_icon(self, "SP_FileDialogNewFolder", "add", ("list-add",)))
        self.edit_btn = QPushButton("Edit")
        self.edit_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.edit_btn.clicked.connect(self.edit_selected_interface)
        self.edit_btn.setIcon(_style_icon(self, "SP_FileDialogDetailedView", "edit", ("document-edit",)))
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.delete_btn.clicked.connect(self.delete_selected_interface)
        self.delete_btn.setIcon(_style_icon(self, "SP_TrashIcon", "delete", ("edit-delete",)))
        self._action_buttons = [self.define_btn, self.edit_btn, self.delete_btn]
        layout.addLayout(self._action_button_grid)
        layout.addWidget(_make_dock_separator())

        self.interface_list = QTreeWidget()
        self.interface_list.setHeaderLabels(
            ["Part 1", "Part 2", "Type", "Placement", "Material", "t", "spacing", "mu", "Status", "ID"]
        )
        _configure_dock_tree(self.interface_list)
        layout.addWidget(self.interface_list)
        self._interface_empty_label = _make_empty_state_label(
            "No interactions yet — click Add Interaction above to define one between two parts.",
            self,
        )
        layout.addWidget(self._interface_empty_label)
        _bind_tree_empty_state(self.interface_list, self._interface_empty_label)
        self.interface_list.itemDoubleClicked.connect(self._on_interface_item_double_clicked)
        self.interface_list.itemSelectionChanged.connect(self._sync_selected_interface_to_inspector)
        self.interface_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.interface_list.customContextMenuRequested.connect(self._show_interface_context_menu)
        if hasattr(self.sketch_view, "interfacesChanged"):
            try:
                self.sketch_view.interfacesChanged.connect(self.refresh_list)
            except Exception:
                pass

        self.refresh_list()
        self._update_responsive_layout()
        _finalize_dock_panel(self)

    def _update_responsive_layout(self):
        width = int(self.contentsRect().width() or self.width() or 0)
        _reflow_button_grid(self._action_button_grid, self._action_buttons, width - 24, min_button_width=124, max_columns=3)
        compact = width < 430
        icon_only = width < 350
        _set_responsive_button_text(
            self.define_btn,
            full=self._define_label_full,
            compact="Add",
            icon_only=icon_only,
        )
        _set_responsive_button_text(
            self.edit_btn,
            full=self._edit_label_full,
            compact="Edit",
            icon_only=icon_only,
        )
        _set_responsive_button_text(
            self.delete_btn,
            full=self._delete_label_full,
            compact="Delete",
            icon_only=icon_only,
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_responsive_layout()

    def _main_window(self):
        return self.window()

    def _materials_tab(self):
        main = self._main_window()
        if not main or not hasattr(main, "properties_panel"):
            return None
        return getattr(main.properties_panel, "materials_tab", None)

    def _populate_material_combo(self, combo):
        combo.clear()
        combo.addItem("[Select Material]", -1)
        for serial, mat in sorted(self.project_state.materials.items()):
            combo.addItem(f"{mat.name} ({mat.mat_type})", int(serial))

    def _current_mesh_dx_hint(self):
        dx_hint = getattr(self.sketch_view, "last_mesh_dx", None)
        if dx_hint and dx_hint > 0:
            return float(dx_hint)
        main = self._main_window()
        try:
            mesh_tab = main.properties_panel.mesh_tab
            if hasattr(mesh_tab, "dx_spin"):
                val = float(mesh_tab.dx_spin.value())
                if val > 0:
                    return val
        except Exception:
            pass
        return None

    def _interface_status_text(self, thickness, target_dx, material_id):
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
        if t > 0 and dx > 0:
            ratio = t / dx
            if ratio < 0.6 or ratio > 1.8:
                warnings.append(f"t/dx={ratio:.2f}")
        return "OK" if not warnings else "WARN:" + ",".join(warnings)

    def _auto_interface_material_name(self, iface_id, type_key):
        return f"IFACE_{int(iface_id)}_{str(type_key).upper()}"

    def _ensure_interface_material_for_type(self, iface_id, type_key):
        spec = Interface.preset_material_spec(type_key)
        if not spec:
            return None
        name = self._auto_interface_material_name(iface_id, type_key)
        mtab = self._materials_tab()
        material_controller = getattr(mtab, "material_controller", None) if mtab is not None else None
        if material_controller is not None:
            mat, _created = material_controller.upsert_material(
                name,
                spec.get("mat_type", "ELAS1"),
                dict(spec.get("properties", {})),
            )
        else:
            existing = next(
                (m for m in self.project_state.materials.values() if str(m.name).lower() == name.lower()),
                None,
            )
            if existing is not None:
                existing.mat_type = spec.get("mat_type", existing.mat_type)
                existing.properties = dict(spec.get("properties", existing.properties))
                mat = existing
            else:
                mat = Material(name, spec.get("mat_type", "ELAS1"), dict(spec.get("properties", {})))
                self.project_state.materials[mat.serial] = mat
            self.sketch_view.deserialize_materials(list(self.project_state.materials.values()))
        # Keep Materials panel in sync if available.
        if mtab is not None:
            try:
                mtab.refresh_material_list()
                mtab.select_material(mat.serial, update_editor=False)
            except Exception:
                pass
        self.sketch_view.materialsChanged.emit()
        _show_panel_notice(
            getattr(self, "_cross_tab_notice", None),
            f"Auto-created material '{mat.name}' (visible in the Materials tab).",
        )
        return mat

    def _set_material_mode_ui(self, mode_combo, type_combo, material_combo, helper_label):
        mode = str(mode_combo.currentData() or "auto")
        type_key = str(type_combo.currentData() or "")
        force_custom = type_key == Interface.CUSTOM_TYPE_KEY
        if force_custom and mode != "existing":
            idx = mode_combo.findData("existing")
            if idx >= 0:
                mode_combo.blockSignals(True)
                mode_combo.setCurrentIndex(idx)
                mode_combo.blockSignals(False)
                mode = "existing"
        enable_custom = force_custom or mode == "existing"
        material_combo.setEnabled(enable_custom)
        helper_text = "Preset interaction type will auto-create a material."
        if force_custom:
            helper_text = "Others (Custom) requires choosing an existing material."
        elif enable_custom:
            helper_text = "Using selected custom material for interaction-layer helper triangles."
        helper_label.setText(helper_text)

    def _interface_from_item(self, item):
        if item is None:
            return None
        try:
            iface_id = int(item.data(0, Qt.UserRole))
        except Exception:
            try:
                iface_id = int(item.text(max(0, item.columnCount() - 1)))
            except Exception:
                return None
        for iface in self.project_state.interfaces:
            try:
                if int(_iface_get(iface, "id", -1)) == iface_id:
                    return iface
            except Exception:
                continue
        return None

    def _selected_interface(self):
        return self._interface_from_item(self.interface_list.currentItem())

    def _selected_interface_payload(self):
        iface = self._selected_interface()
        if iface is None:
            return None
        for index, candidate in enumerate(self.project_state.interfaces):
            if candidate is iface:
                return {
                    "kind": "interaction",
                    "index": int(index),
                    "stage": ProjectStage.INTERFACES,
                }
            try:
                if int(_iface_get(candidate, "id", -1)) == int(_iface_get(iface, "id", -2)):
                    return {
                        "kind": "interaction",
                        "index": int(index),
                        "stage": ProjectStage.INTERFACES,
                    }
            except Exception:
                continue
        return None

    def _sync_selected_interface_to_inspector(self):
        main = self._main_window()
        inspector = getattr(main, "property_inspector", None) if main is not None else None
        payload = self._selected_interface_payload()
        if inspector is None:
            return
        inspector.set_selection_payload(payload)

    def _highlight_selected_interface(self):
        payload = self._selected_interface_payload()
        if payload is None:
            QMessageBox.information(self, "Highlight Interaction", "Select an interaction row first.")
            return
        main = self._main_window()
        inspector = getattr(main, "property_inspector", None) if main is not None else None
        if inspector is not None:
            inspector.set_selection_payload(payload)

    def _show_interface_context_menu(self, pos):
        item = self.interface_list.itemAt(pos)
        if item is None:
            return
        self.interface_list.setCurrentItem(item)
        menu = QMenu(self.interface_list)
        menu.addAction("Edit", self.edit_selected_interface)
        menu.addAction("Highlight", self._highlight_selected_interface)
        menu.addSeparator()
        menu.addAction("Delete", self.delete_selected_interface)
        menu.exec(self.interface_list.viewport().mapToGlobal(pos))

    def _on_interface_item_double_clicked(self, item, _column):
        iface = self._interface_from_item(item)
        if iface is not None:
            self.edit_selected_interface()

    def refresh_list(self):
        selected_iface = self._selected_interface()
        selected_id = int(_iface_get(selected_iface, "id", -1)) if selected_iface is not None else None
        self.interface_list.clear()
        part_map = {p.id: p.name for p in self.project_state.parts}
        for iface in self.project_state.interfaces:
            p1_id = _iface_get(iface, "part1_id")
            p2_id = _iface_get(iface, "part2_id")
            p1_name = part_map.get(p1_id, f"ID:{p1_id}")
            p2_name = part_map.get(p2_id, f"ID:{p2_id}")
            mat_id = _iface_get(iface, "material_id", None)
            try:
                mat_id_txt = str(int(mat_id))
            except Exception:
                mat_id_txt = ""
            thickness = _iface_get(iface, "thickness", None)
            target_dx = _iface_get(iface, "target_dx", None)
            status = _iface_get(iface, "status", "") or self._interface_status_text(
                thickness, target_dx, mat_id
            )
            iface_type = _iface_get(iface, "interface_type", "GLUE")
            placement_mode = _iface_get(
                iface,
                "placement_mode",
                getattr(Interface, "DEFAULT_PLACEMENT_MODE", "matrix_side"),
            )
            friction_coeff = _iface_get(iface, "friction_coeff", 0.0)
            iface_id = _iface_get(iface, "id", -1)
            item = QTreeWidgetItem(
                [
                    p1_name,
                    p2_name,
                    str(iface_type),
                    str(
                        Interface.PLACEMENT_MODES.get(
                            placement_mode,
                            placement_mode,
                        )
                    ),
                    mat_id_txt,
                    "" if thickness in (None, "") else f"{float(thickness):.4g}",
                    "" if target_dx in (None, "") else f"{float(target_dx):.4g}",
                    f"{float(friction_coeff):.4g}",
                    str(status),
                    str(iface_id),
                ]
            )
            item.setData(0, Qt.UserRole, int(iface_id))
            self.interface_list.addTopLevelItem(item)
            if selected_id is not None and int(iface_id) == selected_id:
                self.interface_list.setCurrentItem(item)

    def define_interface(self, iface_to_edit=None, preset_part_ids=None):
        # QPushButton.clicked emits a `checked: bool` argument. When this slot is connected
        # directly, that bool can arrive here and be mistaken for an interface object.
        if isinstance(iface_to_edit, bool):
            iface_to_edit = None
        elif iface_to_edit is not None and _iface_get(iface_to_edit, "part1_id", None) is None:
            QMessageBox.warning(self, "Interaction", "Invalid interaction selection for editing.")
            return
        if len(self.project_state.parts) < 2:
            QMessageBox.warning(self, "Error", "Need at least 2 parts to define an interaction.")
            return
        is_edit_mode = iface_to_edit is not None

        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Material Interaction" if is_edit_mode else "Define Material Interaction")
        dialog.setWindowModality(Qt.ApplicationModal)
        dialog.raise_()
        dialog.activateWindow()
        layout = QVBoxLayout(dialog)

        layout.addWidget(QLabel("Select two parts for interaction definition:"))

        solid_parts = [p for p in self.project_state.parts if not p.is_void]

        if len(solid_parts) < 2:
            QMessageBox.warning(
                self,
                "Interaction Error",
                "At least two solid parts are required to define an interaction.",
            )
            return

        part1_combo = QComboBox()
        part2_combo = QComboBox()

        for part in solid_parts:
            part1_combo.addItem(part.name, part.id)
            part2_combo.addItem(part.name, part.id)

        if len(solid_parts) == 2:
            part1_combo.setCurrentIndex(0)
            part2_combo.setCurrentIndex(1)
        if preset_part_ids and not is_edit_mode:
            first_id = preset_part_ids[0] if len(preset_part_ids) > 0 else None
            second_id = preset_part_ids[1] if len(preset_part_ids) > 1 else None
            idx1 = part1_combo.findData(first_id)
            if idx1 >= 0:
                part1_combo.setCurrentIndex(idx1)
            idx2 = part2_combo.findData(second_id)
            if idx2 >= 0:
                part2_combo.setCurrentIndex(idx2)
            elif second_id is None and first_id is not None:
                for row in range(part2_combo.count()):
                    if part2_combo.itemData(row) != first_id:
                        part2_combo.setCurrentIndex(row)
                        break
        if is_edit_mode:
            idx1 = part1_combo.findData(_iface_get(iface_to_edit, "part1_id", None))
            if idx1 >= 0:
                part1_combo.setCurrentIndex(idx1)
            idx2 = part2_combo.findData(_iface_get(iface_to_edit, "part2_id", None))
            if idx2 >= 0:
                part2_combo.setCurrentIndex(idx2)

        layout.addWidget(QLabel("Part 1:"))
        layout.addWidget(part1_combo)
        layout.addWidget(QLabel("Part 2:"))
        layout.addWidget(part2_combo)

        type_combo = QComboBox()
        preferred_types = [
            "BONDED",
            "FRICTIONAL",
            "SOFT",
            "DAMAGEABLE",
            Interface.CUSTOM_TYPE_KEY,
        ]
        added = set()
        for type_key in preferred_types:
            type_name = Interface.TYPES.get(type_key)
            if type_name:
                type_combo.addItem(type_name, type_key)
                added.add(type_key)
        for type_key, type_name in Interface.TYPES.items():
            if type_key in added:
                continue
            type_combo.addItem(type_name, type_key)
        layout.addWidget(QLabel("Interaction Type:"))
        layout.addWidget(type_combo)
        if is_edit_mode:
            type_idx = type_combo.findData(_iface_get(iface_to_edit, "interface_type", None))
            if type_idx >= 0:
                type_combo.setCurrentIndex(type_idx)

        friction_spin = QDoubleSpinBox()
        friction_spin.setRange(0, 2.0)
        friction_spin.setValue(0.3)
        layout.addWidget(QLabel("Friction Coefficient (if applicable):"))
        layout.addWidget(friction_spin)
        if is_edit_mode:
            try:
                friction_spin.setValue(float(_iface_get(iface_to_edit, "friction_coeff", 0.0)))
            except Exception:
                pass

        mat_mode_combo = QComboBox()
        mat_mode_combo.addItem("Auto from Interaction Type", "auto")
        mat_mode_combo.addItem("Use Existing Material (Custom)", "existing")
        layout.addWidget(QLabel("Interaction Material Source:"))
        layout.addWidget(mat_mode_combo)

        material_combo = QComboBox()
        self._populate_material_combo(material_combo)
        layout.addWidget(QLabel("Custom Interaction Material (for Others / Custom mode):"))
        layout.addWidget(material_combo)
        if is_edit_mode:
            mode_idx = mat_mode_combo.findData(_iface_get(iface_to_edit, "material_mode", "auto"))
            if mode_idx >= 0:
                mat_mode_combo.setCurrentIndex(mode_idx)
            try:
                mat_idx = material_combo.findData(int(_iface_get(iface_to_edit, "material_id", -1)))
            except Exception:
                mat_idx = -1
            if mat_idx >= 0:
                material_combo.setCurrentIndex(mat_idx)

        mat_hint_label = QLabel("")
        mat_hint_label.setWordWrap(True)
        layout.addWidget(mat_hint_label)

        thickness_spin = QDoubleSpinBox()
        thickness_spin.setDecimals(6)
        thickness_spin.setRange(0.0, 1e9)
        thickness_spin.setSingleStep(0.1)
        dx_hint = self._current_mesh_dx_hint()
        if dx_hint is not None and dx_hint > 0:
            thickness_spin.setValue(float(dx_hint))
        else:
            thickness_spin.setValue(1.0)
        layout.addWidget(QLabel(f"Interaction Thickness ({self.sketch_view.current_unit}):"))
        layout.addWidget(thickness_spin)
        if is_edit_mode:
            try:
                if _iface_get(iface_to_edit, "thickness", None) not in (None, ""):
                    thickness_spin.setValue(float(_iface_get(iface_to_edit, "thickness")))
            except Exception:
                pass

        target_dx_spin = QDoubleSpinBox()
        target_dx_spin.setDecimals(6)
        target_dx_spin.setRange(0.0, 1e9)
        target_dx_spin.setSingleStep(0.1)
        if dx_hint is not None and dx_hint > 0:
            target_dx_spin.setValue(float(dx_hint))
        else:
            target_dx_spin.setValue(1.0)
        layout.addWidget(QLabel(f"Target particle spacing ({self.sketch_view.current_unit}):"))
        layout.addWidget(target_dx_spin)
        if is_edit_mode:
            try:
                if _iface_get(iface_to_edit, "target_dx", None) not in (None, ""):
                    target_dx_spin.setValue(float(_iface_get(iface_to_edit, "target_dx")))
            except Exception:
                pass

        placement_combo = QComboBox()
        placement_combo.addItem(
            Interface.PLACEMENT_MODES.get("matrix_side", "Matrix-side (Coating) (Recommended)"),
            "matrix_side",
        )
        # Placeholder future modes are shown only after implementation to avoid user confusion.
        layout.addWidget(QLabel("Interaction Placement:"))
        layout.addWidget(placement_combo)
        placement_hint = QLabel(
            "Matrix-side coating keeps the inclusion geometry unchanged and creates the interaction layer outside the inclusion boundary."
        )
        placement_hint.setWordWrap(True)
        placement_hint.setObjectName("MinorStatusLabel")
        layout.addWidget(placement_hint)
        if is_edit_mode:
            mode_idx = placement_combo.findData(
                _iface_get(
                    iface_to_edit,
                    "placement_mode",
                    getattr(Interface, "DEFAULT_PLACEMENT_MODE", "matrix_side"),
                )
            )
            if mode_idx >= 0:
                placement_combo.setCurrentIndex(mode_idx)

        notes_input = QLineEdit()
        notes_input.setPlaceholderText("Optional notes (e.g., coating / interphase assumptions)")
        layout.addWidget(QLabel("Notes (optional):"))
        layout.addWidget(notes_input)
        if is_edit_mode:
            notes_input.setText(str(_iface_get(iface_to_edit, "notes", "") or ""))

        status_label = QLabel("")
        status_label.setWordWrap(True)
        layout.addWidget(status_label)

        ui_state = {
            "last_type": str(type_combo.currentData() or "") if is_edit_mode else None,
        }

        def _refresh_interface_ui(*_args):
            type_key = str(type_combo.currentData() or "")
            default_mu = Interface.TYPE_DEFAULT_FRICTION.get(type_key)
            if default_mu is not None and ui_state.get("last_type") != type_key:
                friction_spin.setValue(float(default_mu))
            ui_state["last_type"] = type_key
            friction_enabled = type_key in {"FRICTIONAL", "ROUGH", "GLIDING", "CONTACT"}
            friction_spin.setEnabled(friction_enabled)
            self._set_material_mode_ui(
                mat_mode_combo, type_combo, material_combo, mat_hint_label
            )
            status_txt = self._interface_status_text(
                thickness_spin.value(),
                target_dx_spin.value(),
                material_combo.currentData()
                if (type_key == Interface.CUSTOM_TYPE_KEY or str(mat_mode_combo.currentData()) == "existing")
                else 0,
            )
            ratio = ""
            if target_dx_spin.value() > 0:
                ratio = f" (t/dx={thickness_spin.value()/target_dx_spin.value():.2f})"
            status_label.setText(f"Interaction layer check: {status_txt}{ratio}")

        type_combo.currentIndexChanged.connect(_refresh_interface_ui)
        mat_mode_combo.currentIndexChanged.connect(_refresh_interface_ui)
        material_combo.currentIndexChanged.connect(_refresh_interface_ui)
        thickness_spin.valueChanged.connect(_refresh_interface_ui)
        target_dx_spin.valueChanged.connect(_refresh_interface_ui)
        _refresh_interface_ui()

        btn_ok = QPushButton("Update Interaction" if is_edit_mode else "Create Interaction")
        btn_ok.clicked.connect(dialog.accept)
        layout.addWidget(btn_ok)

        if dialog.exec() == QDialog.Accepted:
            part1_id, part2_id = part1_combo.currentData(), part2_combo.currentData()
            if part1_id == part2_id:
                QMessageBox.warning(
                    self, "Error", "Cannot define an interaction between a part and itself."
                )
                return

            self.sketch_view.push_undo_state()
            if is_edit_mode:
                interface_obj = iface_to_edit
                _iface_set(interface_obj, "part1_id", part1_id)
                _iface_set(interface_obj, "part2_id", part2_id)
                _iface_set(interface_obj, "interface_type", type_combo.currentData())
            else:
                interface_obj = Interface(part1_id, part2_id, type_combo.currentData())
            _iface_set(interface_obj, "friction_coeff", friction_spin.value())
            _iface_set(interface_obj, "layer_mode", getattr(Interface, "DEFAULT_LAYER_MODE", "single_layer_ring"))
            _iface_set(interface_obj, "placement_mode", str(
                placement_combo.currentData() or getattr(Interface, "DEFAULT_PLACEMENT_MODE", "matrix_side")
            ))
            _iface_set(interface_obj, "thickness", float(thickness_spin.value()))
            _iface_set(interface_obj, "target_dx", float(target_dx_spin.value()))
            _iface_set(interface_obj, "notes", notes_input.text().strip())
            _iface_set(interface_obj, "material_mode", str(mat_mode_combo.currentData() or "auto"))

            if (
                str(_iface_get(interface_obj, "interface_type")) == Interface.CUSTOM_TYPE_KEY
                and _iface_get(interface_obj, "material_mode") != "existing"
            ):
                _iface_set(interface_obj, "material_mode", "existing")

            if _iface_get(interface_obj, "material_mode") == "existing":
                try:
                    selected_material_id = int(material_combo.currentData())
                except Exception:
                    selected_material_id = -1
                if selected_material_id <= 0 or selected_material_id not in self.project_state.materials:
                    QMessageBox.warning(
                        self,
                        "Interaction Material",
                        "Choose an existing material for 'Others (Custom)' / custom interaction mode.",
                    )
                    return
                interface_obj.material_id = selected_material_id
            else:
                auto_mat = self._ensure_interface_material_for_type(
                    interface_obj.id, interface_obj.interface_type
                )
                if auto_mat is None:
                    QMessageBox.warning(
                        self,
                        "Interaction Material",
                        "No preset material is available for this interaction type. "
                        "Use 'Others (Custom)' and choose a material.",
                    )
                    return
                _iface_set(interface_obj, "material_id", int(auto_mat.serial))

            _iface_set(
                interface_obj,
                "status",
                self._interface_status_text(
                    _iface_get(interface_obj, "thickness"),
                    _iface_get(interface_obj, "target_dx"),
                    _iface_get(interface_obj, "material_id"),
                ),
            )
            if not is_edit_mode:
                self.project_state.interfaces.append(interface_obj)
            self.refresh_list()
            if hasattr(self.sketch_view, "_emit_interfaces_changed"):
                self.sketch_view._emit_interfaces_changed()
            else:
                self.sketch_view.interfacesChanged.emit()

    def edit_selected_interface(self):
        iface = self._selected_interface()
        if iface is None:
            QMessageBox.information(self, "Edit Interaction", "Select an interaction row first.")
            return
        self.define_interface(iface_to_edit=iface)

    def delete_selected_interface(self):
        iface = self._selected_interface()
        if iface is None:
            QMessageBox.information(self, "Delete Interaction", "Select an interaction row first.")
            return
        part_map = {p.id: p.name for p in self.project_state.parts}
        p1_id = _iface_get(iface, "part1_id", None)
        p2_id = _iface_get(iface, "part2_id", None)
        p1 = part_map.get(p1_id, str(p1_id or ""))
        p2 = part_map.get(p2_id, str(p2_id or ""))
        iface_id = _iface_get(iface, "id", "?")
        reply = QMessageBox.question(
            self,
            "Delete Interaction",
            f"Delete interaction #{iface_id} ({p1} <-> {p2})?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.sketch_view.push_undo_state()
        self.project_state.interfaces = [
            item for item in self.project_state.interfaces
            if item is not iface and int(_iface_get(item, "id", -1)) != int(_iface_get(iface, "id", -1))
        ]
        self.refresh_list()
        if hasattr(self.sketch_view, "_emit_interfaces_changed"):
            self.sketch_view._emit_interfaces_changed()
        else:
            self.sketch_view.interfacesChanged.emit()

    # Backward-compatible method alias during migration.
    def define_interaction(self):
        self.define_interface()


# Backward-compatible class alias during migration.
InteractionsPanel = InterfacesPanel


class FluidStagePanel(QWidget):
    def __init__(self, sketch_view, parent=None, project_state=None):
        super().__init__(parent)
        self.sketch_view = sketch_view
        self.project_state = _resolve_project_state(sketch_view, project_state)

        layout = QVBoxLayout(self)
        _apply_layout_metrics(layout)

        title = QLabel("Fluid")
        title.setObjectName("SummaryLabel")
        layout.addWidget(title)

        info = QLabel(
            "Summary only. This stage appears only when fluid participation is required. "
            "Use it to review fluid materials and regions before solving."
        )
        info.setWordWrap(True)
        info.setObjectName("MinorStatusLabel")
        layout.addWidget(info)

        self.summary_label = QLabel("No fluid materials detected.")
        self.summary_label.setWordWrap(True)
        self.summary_label.setObjectName("MinorStatusLabel")
        layout.addWidget(self.summary_label)

        self.table = QTreeWidget()
        self.table.setHeaderLabels(["Item", "Classification"])
        self.table.setRootIsDecorated(False)
        self.table.setUniformRowHeights(True)
        self.table.setAlternatingRowColors(True)
        self.table.header().setStretchLastSection(True)
        self.table.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        layout.addWidget(self.table, 1)

        self.refresh_summary()
        _finalize_dock_panel(self)

    def set_project_state(self, project_state):
        self.project_state = _resolve_project_state(self.sketch_view, project_state)
        self.refresh_summary()

    @staticmethod
    def _material_is_fluid(material):
        behavior = str(getattr(material, "behavior", "") or "").strip().lower()
        mat_type = str(getattr(material, "mat_type", "") or "").strip().lower()
        return behavior == "fluid" or mat_type == "fluid"

    @staticmethod
    def _part_is_fluid(part):
        return str(getattr(part, "material_behavior", "") or "").strip().lower() == "fluid"

    def refresh_summary(self):
        self.table.clear()
        materials_store = getattr(self.project_state, "materials", {}) or {}
        parts = list(getattr(self.project_state, "parts", []) or [])
        fluid_materials = [
            material for material in materials_store.values()
            if self._material_is_fluid(material)
        ]
        fluid_part_ids = {
            int(getattr(part, "id", -1))
            for part in parts
            if self._part_is_fluid(part)
        }

        for material in fluid_materials:
            serial = getattr(material, "serial", None)
            linked_count = sum(
                1 for part in parts
                if getattr(part, "material_id", None) == serial or self._part_is_fluid(part)
            )
            detail = "Fluid material"
            if linked_count > 0:
                detail = f"Fluid material [{linked_count} part(s)]"
            item = QTreeWidgetItem([getattr(material, "name", f"Material {serial}"), detail])
            item.setIcon(0, get_icon("fluid", size=16))
            self.table.addTopLevelItem(item)

        for part in parts:
            if int(getattr(part, "id", -1)) not in fluid_part_ids:
                continue
            item = QTreeWidgetItem([
                getattr(part, "name", f"Part {getattr(part, 'id', '?')}"),
                "Fluid region",
            ])
            item.setIcon(0, get_icon("geometry", size=16))
            self.table.addTopLevelItem(item)

        analysis = str(getattr(self.project_state, "analysis_type", "static") or "static").lower()
        count = self.table.topLevelItemCount()
        if count > 0:
            self.summary_label.setText(f"{count} fluid-stage item(s) active.")
        elif analysis in {"fluid", "fsi"}:
            self.summary_label.setText(
                "This project uses a fluid-capable analysis type. Assign a fluid material to activate fluid regions."
            )
        else:
            self.summary_label.setText("No fluid materials detected.")


class FractureStagePanel(QWidget):
    def __init__(self, sketch_view, parent=None, project_state=None):
        super().__init__(parent)
        self.sketch_view = sketch_view
        self.project_state = _resolve_project_state(sketch_view, project_state)

        layout = QVBoxLayout(self)
        _apply_layout_metrics(layout)

        title = QLabel("Fracture")
        title.setObjectName("SummaryLabel")
        layout.addWidget(title)

        info = QLabel(
            "Summary only. This stage appears only when a damage model is enabled. "
            "Review damage-enabled materials and part overrides here."
        )
        info.setWordWrap(True)
        info.setObjectName("MinorStatusLabel")
        layout.addWidget(info)

        self.summary_label = QLabel("No damage-enabled materials detected.")
        self.summary_label.setWordWrap(True)
        self.summary_label.setObjectName("MinorStatusLabel")
        layout.addWidget(self.summary_label)

        self.table = QTreeWidget()
        self.table.setHeaderLabels(["Item", "Damage Model"])
        self.table.setRootIsDecorated(False)
        self.table.setUniformRowHeights(True)
        self.table.setAlternatingRowColors(True)
        self.table.header().setStretchLastSection(True)
        self.table.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        layout.addWidget(self.table, 1)

        self.refresh_summary()
        _finalize_dock_panel(self)

    def set_project_state(self, project_state):
        self.project_state = _resolve_project_state(self.sketch_view, project_state)
        self.refresh_summary()

    def refresh_summary(self):
        self.table.clear()
        materials_store = getattr(self.project_state, "materials", {}) or {}
        parts = list(getattr(self.project_state, "parts", []) or [])

        for material in materials_store.values():
            damage = str(getattr(material, "damage", "none") or "none").lower()
            if damage == "none":
                continue
            item = QTreeWidgetItem([
                getattr(material, "name", f"Material {getattr(material, 'serial', '?')}"),
                damage_label(damage),
            ])
            item.setIcon(0, get_icon("fracture", size=16))
            self.table.addTopLevelItem(item)

        for part in parts:
            damage = str(getattr(part, "material_damage", "none") or "none").lower()
            if damage == "none":
                continue
            item = QTreeWidgetItem([
                getattr(part, "name", f"Part {getattr(part, 'id', '?')}"),
                f"Part override: {damage_label(damage)}",
            ])
            item.setIcon(0, get_icon("geometry", size=16))
            self.table.addTopLevelItem(item)

        count = self.table.topLevelItemCount()
        self.summary_label.setText(
            f"{count} fracture-stage item(s) detected." if count > 0 else "No damage-enabled materials detected."
        )


class PropertiesPanel(QWidget):
    nextStageRequested = Signal()
    prevStageRequested = Signal()
    closeRequested = Signal()

    def __init__(self, sketch_view, parent=None, project_state=None):
        super().__init__(parent)
        self.sketch_view = sketch_view
        self.project_state = _resolve_project_state(sketch_view, project_state)
        self.sketch_view.project_state = self.project_state

        self.setObjectName("PropertiesPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN)
        layout.setSpacing(DOCK_SECTION_SPACING)

        header = QWidget()
        header.setProperty("card", True)
        header_layout = QHBoxLayout(header)
        _apply_layout_metrics(
            header_layout,
            margins=(DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN),
            spacing=DOCK_ROW_SPACING,
        )
        self.stage_icon_label = QLabel()
        self.stage_icon_label.setMinimumSize(22, 22)
        self.stage_icon_label.setMaximumSize(22, 22)
        header_layout.addWidget(self.stage_icon_label, 0, Qt.AlignVCenter)
        stage_text_widget = QWidget()
        stage_text_layout = QVBoxLayout(stage_text_widget)
        stage_text_layout.setContentsMargins(0, 0, 0, 0)
        stage_text_layout.setSpacing(2)
        self.stage_label = QLabel("Geometry")
        self.stage_label.setObjectName("SummaryLabel")
        self.stage_panel_label = QLabel("Geometry tools")
        self.stage_panel_label.setObjectName("MinorStatusLabel")
        stage_text_layout.addWidget(self.stage_label)
        stage_text_layout.addWidget(self.stage_panel_label)
        header_layout.addWidget(stage_text_widget, 0, Qt.AlignVCenter)
        header_layout.addStretch(1)
        self.prev_stage_button = QToolButton()
        self.prev_stage_button.setProperty("secondary", True)
        self.prev_stage_button.setProperty("dockIconButton", True)
        self.prev_stage_button.setIcon(get_icon("undo", size=18))
        self.prev_stage_button.setToolTip("Previous Stage\nMove to the previous workflow stage.")
        self.prev_stage_button.clicked.connect(self.prevStageRequested.emit)
        header_layout.addWidget(self.prev_stage_button)
        self.next_stage_button = QToolButton()
        self.next_stage_button.setProperty("primary", True)
        self.next_stage_button.setProperty("dockIconButton", True)
        self.next_stage_button.setIcon(get_icon("next_stage", size=18))
        self.next_stage_button.setToolTip("Next Stage\nAdvance to the next workflow stage.")
        self.next_stage_button.clicked.connect(self.nextStageRequested.emit)
        header_layout.addWidget(self.next_stage_button)
        self.prev_stage_button.hide()
        self.next_stage_button.hide()

        # Top-right toggle: switch between Properties view (Tools/Parts/etc.)
        # and Metadata view (selected item's properties). Icon-only to keep
        # the header narrow.
        self.view_toggle_button = QToolButton()
        self.view_toggle_button.setProperty("dockIconButton", True)
        self.view_toggle_button.setCheckable(True)
        self.view_toggle_button.setChecked(False)
        self.view_toggle_button.setIcon(get_icon("property_inspector", size=16))
        self.view_toggle_button.setIconSize(QSize(16, 16))
        self.view_toggle_button.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.view_toggle_button.setToolTip(
            "Toggle view\nSwitch between Properties (tools, lists) and Metadata "
            "(selected item's properties)."
        )
        self.view_toggle_button.setMinimumSize(24, 24)
        self.view_toggle_button.setMaximumSize(28, 28)
        self.view_toggle_button.toggled.connect(self._on_view_toggle)
        self.view_toggle_button.hide()  # shown only after set_metadata_panel
        header_layout.addWidget(self.view_toggle_button)

        # Close button (×) — hides the entire Properties panel. Re-opened
        # via the canvas edge-rail, Ctrl+J shortcut, or View menu.
        self.close_button = QToolButton()
        self.close_button.setObjectName("PanelCloseButton")
        self.close_button.setText("✕")  # ✕
        self.close_button.setToolTip("Close Properties panel (Ctrl+J)")
        self.close_button.setMinimumSize(24, 24)
        self.close_button.setMaximumSize(26, 26)
        self.close_button.setCursor(Qt.PointingHandCursor)
        self.close_button.clicked.connect(self.closeRequested.emit)
        header_layout.addWidget(self.close_button)

        header.setVisible(True)
        layout.addWidget(header)
        self.tabs = StageIconTabs()
        self.tabs.setRailVisible(False)
        self.tabs.setRailInteractive(False)
        self.tabs.currentChanged.connect(lambda _idx: self._update_header_label())

        # Stacked content: page 0 = stage tabs (Properties), page 1 = Metadata.
        # The toggle button in the header flips between them.
        self._content_stack = QStackedWidget()
        self._content_stack.addWidget(self.tabs)
        # Page 1 is itself a stack so the Metadata view can swap content based
        # on the active stage (e.g., Materials stage shows Project Materials +
        # Assignment instead of the global Property Inspector).
        self._metadata_stack = QStackedWidget()
        self._content_stack.addWidget(self._metadata_stack)
        layout.addWidget(self._content_stack, 1)
        self._metadata_container = None
        self._stage_metadata_index = {}  # stage -> index in _metadata_stack
        # Index of the default metadata page (filled by set_metadata_panel).
        # -1 until the default is registered.
        self._default_metadata_index = -1

        self.assembly_tab = OperationHistoryTree(self.sketch_view, project_state=self.project_state)
        self.tabs.addTab(self.assembly_tab, get_icon("parts", size=18), "Geometry")
        self.tabs.setTabToolTip(0, "Geometry")

        self.materials_tab = MaterialsPanel(self.sketch_view, project_state=self.project_state)
        self.tabs.addTab(self.materials_tab, get_icon("materials", size=18), "Materials")
        self.tabs.setTabToolTip(1, "Materials")
        # Materials-stage Metadata view = Project Materials + Assignment.
        self.register_stage_metadata(
            ProjectStage.MATERIALS,
            self.materials_tab.get_project_view(),
        )

        self.fluid_tab = FluidStagePanel(self.sketch_view, project_state=self.project_state)
        self.tabs.addTab(self.fluid_tab, get_icon("fluid", size=18), "Fluid")
        self.tabs.setTabToolTip(2, "Fluid")

        self.interfaces_tab = InterfacesPanel(self.sketch_view, project_state=self.project_state)
        self.interactions_tab = self.interfaces_tab  # Backward-compatible attribute alias
        self.tabs.addTab(self.interfaces_tab, get_icon("interactions", size=18), "Interactions")
        self.tabs.setTabToolTip(3, "Interactions")

        self.bcs_tab = BCLoadsPanel(self.sketch_view, panel_mode="combined", project_state=self.project_state)
        self.tabs.addTab(self.bcs_tab, get_icon("bc", size=18), "Boundary Conditions")
        self.tabs.setTabToolTip(4, "Boundary Conditions")

        self.loads_tab = self.bcs_tab

        self.fracture_tab = FractureStagePanel(self.sketch_view, project_state=self.project_state)
        self.tabs.addTab(self.fracture_tab, get_icon("fracture", size=18), "Fracture")
        self.tabs.setTabToolTip(5, "Fracture")

        self.mesh_tab = MeshPanel(self.sketch_view, project_state=self.project_state)
        self.tabs.addTab(self.mesh_tab, get_icon("particles", size=18), "Particles")
        self.tabs.setTabToolTip(6, "Particles")

        self.job_tab = JobPanel(self.sketch_view, project_state=self.project_state)
        self.tabs.addTab(self.job_tab, get_icon("solve", size=18), "Solve")
        self.tabs.setTabToolTip(7, "Solve")

        self.results_tab = ResultsPanel(self.sketch_view, project_state=self.project_state)
        self.tabs.addTab(self.results_tab, get_icon("results", size=18), "Results")
        self.tabs.setTabToolTip(8, "Results")

        nav_footer = QWidget()
        nav_footer_layout = QHBoxLayout(nav_footer)
        _apply_layout_metrics(
            nav_footer_layout,
            margins=(0, DOCK_ROW_SPACING, 0, 0),
            spacing=DOCK_ROW_SPACING,
        )
        self.bottom_prev_stage_button = QPushButton("Previous")
        self.bottom_prev_stage_button.setToolTip("Previous Stage")
        self.bottom_prev_stage_button.setIcon(get_icon("undo", size=14))
        self.bottom_prev_stage_button.setMinimumWidth(60)
        self.bottom_prev_stage_button.clicked.connect(self.prevStageRequested.emit)
        self.bottom_next_stage_button = QPushButton("Next")
        self.bottom_next_stage_button.setToolTip("Next Stage")
        self.bottom_next_stage_button.setProperty("primary", True)
        self.bottom_next_stage_button.setIcon(get_icon("next_stage", size=14))
        self.bottom_next_stage_button.setMinimumWidth(60)
        self.bottom_next_stage_button.clicked.connect(self.nextStageRequested.emit)
        nav_footer_layout.addWidget(self.bottom_prev_stage_button, 1)
        nav_footer_layout.addWidget(self.bottom_next_stage_button, 1)
        layout.addWidget(nav_footer)

        self._current_stage_name = "Geometry"
        self._update_header_label()

    def set_metadata_panel(self, inspector_widget):
        """Register a property-inspector widget as the DEFAULT Metadata view.
        This is shown for any stage that hasn't registered a stage-specific
        metadata widget via register_stage_metadata()."""
        if inspector_widget is None or self._metadata_container is not None:
            return
        container = QWidget()
        container.setObjectName("MetadataContainer")
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        inspector_widget.setParent(container)
        container_layout.addWidget(inspector_widget, 1)
        idx = self._metadata_stack.addWidget(container)
        self._metadata_container = container
        self._default_metadata_index = idx
        # Apply the dock-panel finalization to the inspector so its inner
        # QScrollAreas get the same horizontal-lock treatment as the stage
        # panels. (The inspector class lives in main_window.py and otherwise
        # would not be swept.)
        _finalize_dock_panel(inspector_widget)
        # If no stage was active when this was called, switch the metadata
        # view to the default page so the toggle has something to show.
        if not self._stage_metadata_index:
            self._metadata_stack.setCurrentIndex(idx)
        self.view_toggle_button.show()

    def register_stage_metadata(self, stage, widget):
        """Register a stage-specific widget to show in the Metadata view when
        this stage is active. Falls back to set_metadata_panel's default for
        any stage that doesn't have a registration."""
        if widget is None:
            return
        container = QWidget()
        container.setObjectName(f"MetadataContainer_{stage.name}")
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        widget.setParent(container)
        container_layout.addWidget(widget, 1)
        idx = self._metadata_stack.addWidget(container)
        self._stage_metadata_index[stage] = idx
        # If the toggle button wasn't shown because the default inspector
        # hadn't been set, show it anyway since this stage has its own metadata.
        self.view_toggle_button.show()

    def _on_view_toggle(self, checked):
        """Header toggle handler — swap between Properties (page 0) and
        Metadata (page 1) in the content stack."""
        if self._metadata_container is None:
            return
        if checked:
            self._content_stack.setCurrentIndex(1)
            self.view_toggle_button.setIcon(get_icon("parts", size=16))
            self.view_toggle_button.setToolTip("Switch to Properties view")
            self.stage_panel_label.setText("Metadata for current selection")
        else:
            self._content_stack.setCurrentIndex(0)
            self.view_toggle_button.setIcon(get_icon("property_inspector", size=16))
            self.view_toggle_button.setToolTip("Switch to Metadata view")
            self._update_header_label()

    def set_project_state(self, project_state):
        self.project_state = _resolve_project_state(self.sketch_view, project_state)
        self.sketch_view.project_state = self.project_state
        for panel_name in (
            "assembly_tab",
            "materials_tab",
            "fluid_tab",
            "interfaces_tab",
            "bcs_tab",
            "fracture_tab",
            "loads_tab",
            "mesh_tab",
            "job_tab",
            "results_tab",
        ):
            panel = getattr(self, panel_name, None)
            if panel is not None:
                if hasattr(panel, "set_project_state"):
                    panel.set_project_state(self.project_state)
                else:
                    panel.project_state = self.project_state
        if hasattr(self, "materials_tab"):
            self.materials_tab.refresh_material_list()
        if hasattr(self, "assembly_tab"):
            self.assembly_tab.refresh()

    def set_stage(self, stage):
        stage_name = stage.name.title()
        if stage == ProjectStage.MESH:
            stage_name = "Particles"
        elif stage == ProjectStage.FLUID:
            stage_name = "Fluid"
        elif stage == ProjectStage.BCS:
            stage_name = "Boundary Conditions"
        elif stage == ProjectStage.INTERFACES:
            stage_name = "Interactions"
        elif stage == ProjectStage.FRACTURE:
            stage_name = "Fracture"
        elif stage == ProjectStage.LOADS:
            stage_name = "Boundary Conditions"
        elif stage == ProjectStage.JOB:
            stage_name = "Solve"
        elif stage == ProjectStage.RESULTS:
            stage_name = "Results"
        self._current_stage_name = stage_name
        self.show_stage(stage)

    def show_stage(self, stage):
        stage_to_index = {
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
        index = stage_to_index.get(stage, 0)
        module_map = {
            0: "Part",
            1: "Property",
            2: "Property",
            3: "Interface",
            4: "Boundary",
            5: "Property",
            6: "Mesh",
            7: "Job",
            8: "Results",
        }
        module_name = module_map.get(index, "Part")
        if self.tabs.currentIndex() != index:
            self.tabs.setCurrentIndex(index)
        self._refresh_current_panel_layout()
        self.scroll_current_panel_to_top()
        self._update_header_label()
        self._update_stage_nav_buttons()
        self.sketch_view.set_module(module_name)
        # Swap the Metadata view content to the stage-specific override (if
        # registered) so the toggle shows the right context for this stage.
        if hasattr(self, "_metadata_stack"):
            fallback = self._default_metadata_index
            if fallback < 0:
                fallback = 0
            meta_idx = self._stage_metadata_index.get(stage, fallback)
            if 0 <= meta_idx < self._metadata_stack.count():
                self._metadata_stack.setCurrentIndex(meta_idx)
        if module_name == "Boundary" and hasattr(self, "bcs_tab"):
            self.bcs_tab.reset_stage_state(clear_lists=False)
        elif module_name == "Property" and stage == ProjectStage.FLUID and hasattr(self, "fluid_tab"):
            self.fluid_tab.refresh_summary()
        elif module_name == "Property" and stage == ProjectStage.FRACTURE and hasattr(self, "fracture_tab"):
            self.fracture_tab.refresh_summary()
        elif module_name == "Load" and hasattr(self, "loads_tab"):
            self.loads_tab.reset_stage_state(clear_lists=False)
        elif module_name == "Mesh" and hasattr(self, "mesh_tab"):
            self.mesh_tab.on_mesh_panel_open()
        elif module_name == "Results" and hasattr(self, "results_tab"):
            self.results_tab.sync_stage_state()
        main = self.window()
        if main is not None:
            try:
                if hasattr(main, "view_3d") and main.view_3d is not None and hasattr(main, "_workspace_3d"):
                    if bool(getattr(main, "_workspace_3d", False)):
                        if module_name in ("Boundary", "Load"):
                            target_tab = self.bcs_tab
                            combo = getattr(target_tab, "selection_target_combo", None)
                            if combo is not None and combo.currentText() == "Auto":
                                combo.setCurrentText("Face")
                            target_tab._on_selection_target_changed()
                        if hasattr(main, "_update_gizmo_position"):
                            main._update_gizmo_position()
            except Exception:
                pass

    def scroll_current_panel_to_top(self):
        if hasattr(self, "tabs"):
            self.tabs.scrollCurrentToTop()

    def reveal_panel_widget(self, widget, top_margin=14):
        if hasattr(self, "tabs"):
            self.tabs.ensureWidgetVisible(widget, top_margin=top_margin)

    def _update_stage_nav_buttons(self):
        if not hasattr(self, "tabs"):
            return
        idx = int(self.tabs.currentIndex())
        last = int(self.tabs.count()) - 1
        self.prev_stage_button.setEnabled(idx > 0)
        self.next_stage_button.setEnabled(idx < last)
        if hasattr(self, "bottom_prev_stage_button"):
            self.bottom_prev_stage_button.setEnabled(idx > 0)
        if hasattr(self, "bottom_next_stage_button"):
            self.bottom_next_stage_button.setEnabled(idx < last)

    def set_navigation_state(self, prev_stage, next_stage):
        has_prev = prev_stage is not None
        has_next = next_stage is not None
        self.prev_stage_button.setEnabled(has_prev)
        self.next_stage_button.setEnabled(has_next)
        if hasattr(self, "bottom_prev_stage_button"):
            self.bottom_prev_stage_button.setEnabled(has_prev)
        if hasattr(self, "bottom_next_stage_button"):
            self.bottom_next_stage_button.setEnabled(has_next)

    def _refresh_current_panel_layout(self):
        if not hasattr(self, "tabs"):
            return
        panel = self.tabs.currentWidget()
        if panel is None:
            return
        for method_name in ("_update_responsive_layout", "_update_action_button_density"):
            method = getattr(panel, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass

    def _current_panel_name(self):
        if not hasattr(self, "tabs"):
            return "Geometry"
        index = self.tabs.currentIndex()
        return self.tabs.tabText(index) or "Panel"

    def _stage_name_for_index(self, index=None):
        if index is None and hasattr(self, "tabs"):
            index = self.tabs.currentIndex()
        stage_map = {
            0: "Geometry",
            1: "Materials",
            2: "Fluid",
            3: "Interactions",
            4: "Boundary Conditions",
            5: "Fracture",
            6: "Particles",
            7: "Solve",
            8: "Results",
        }
        return stage_map.get(index, str(getattr(self, "_current_stage_name", "") or "Geometry"))

    def _current_stage_icon_name(self, stage_name=None):
        label = str(stage_name or self._stage_name_for_index()).strip().lower()
        icon_map = {
            "geometry": "parts",
            "materials": "materials",
            "fluid": "fluid",
            "interactions": "interactions",
            "boundary conditions": "bc",
            "fracture": "fracture",
            "particles": "particles",
            "solve": "solve",
            "results": "results",
        }
        return icon_map.get(label, "parts")

    def _update_header_label(self):
        panel = self._current_panel_name()
        stage_name = self._stage_name_for_index()
        self._current_stage_name = stage_name
        stage_hint_map = {
            "Geometry": "Sketch and part tools",
            "Materials": "Library and part assignment",
            "Fluid": "Summary only",
            "Interactions": "Interaction tools",
            "Boundary Conditions": "BC and load tools",
            "Fracture": "Summary only",
            "Particles": "Generation and mesh QA",
            "Solve": "Export and run",
            "Results": "Result display controls",
        }
        self.stage_label.setText(stage_name)
        self.stage_panel_label.setText(stage_hint_map.get(stage_name, panel))
        try:
            self.stage_icon_label.setPixmap(get_icon(self._current_stage_icon_name(stage_name), size=18).pixmap(QSize(18, 18)))
        except Exception:
            self.stage_icon_label.clear()
