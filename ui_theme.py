from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPalette, QSurfaceFormat


@dataclass(frozen=True)
class UiTokens:
    font_family: str = "Segoe UI"
    font_size: int = 10
    small_font_size: int = 10
    spacing_xs: int = 4
    spacing_sm: int = 6
    spacing_md: int = 10
    radius_sm: int = 0
    radius_md: int = 0
    radius_lg: int = 0
    toolbar_icon_size: int = 24
    primitive_icon_size: int = 18
    primitive_button_size: int = 38
    section_header_icon_size: int = 18
    section_header_icon_box_w: int = 40
    section_header_icon_box_h: int = 24


UI_TOKENS = UiTokens()


def configure_qt_runtime() -> None:
    """Apply global Qt quality flags before QApplication is created."""
    try:
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:
        pass

    for attr_name in ("AA_EnableHighDpiScaling", "AA_UseHighDpiPixmaps"):
        attr = getattr(Qt.ApplicationAttribute, attr_name, None)
        if attr is None:
            continue
        try:
            QCoreApplication.setAttribute(attr, True)
        except Exception:
            pass

    try:
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setStencilBufferSize(8)
        fmt.setSwapBehavior(QSurfaceFormat.DoubleBuffer)
        fmt.setSamples(8)
        QSurfaceFormat.setDefaultFormat(fmt)
    except Exception:
        pass


def _build_palette() -> QPalette:
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor("#f3f5f7"))
    pal.setColor(QPalette.WindowText, QColor("#18212b"))
    pal.setColor(QPalette.Base, QColor("#ffffff"))
    pal.setColor(QPalette.AlternateBase, QColor("#f8fafb"))
    pal.setColor(QPalette.Text, QColor("#18212b"))
    pal.setColor(QPalette.Button, QColor("#eef1f4"))
    pal.setColor(QPalette.ButtonText, QColor("#18212b"))
    pal.setColor(QPalette.ToolTipBase, QColor("#ffffff"))
    pal.setColor(QPalette.ToolTipText, QColor("#18212b"))
    pal.setColor(QPalette.Highlight, QColor("#516678"))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.BrightText, QColor("#ffffff"))
    return pal


def _build_stylesheet() -> str:
    t = UI_TOKENS
    style = """
QMainWindow, QWidget {
    color: #18212b;
    background: #f3f5f7;
}

QSplitter#MainSplitter::handle {
    background: #d7dde3;
    border-radius: __RADIUS_SM__px;
}

QWidget#CommandBar {
    background: #f8fafb;
    border-top: 1px solid #d9dfe5;
}
QLabel#CommandLabel {
    color: #2d3a46;
    font-weight: 700;
}
QLabel#CommandStatus, QLabel#MinorStatusLabel {
    color: #5f6b76;
    font-size: __SMALL_FONT_SIZE__px;
}
QLabel#ModeIndicator {
    padding: 4px 10px;
    border: 1px solid #d5dce3;
    border-radius: __RADIUS_MD__px;
    background: #ffffff;
    color: #344250;
    font-weight: 700;
}
QLabel#SectionTitleLabel {
    color: #263341;
    font-weight: 700;
}
QLabel#InfoHintLabel {
    color: #2f6f57;
}
QLabel#WarnHintLabel {
    color: #b45309;
}
QLabel#NeutralHintLabel {
    color: #18212b;
}
QLabel#SummaryLabel {
    color: #5f6b76;
    font-size: __SMALL_FONT_SIZE__px;
}
QFrame[card="true"], QWidget[card="true"] {
    background: #ffffff;
    border: 1px solid #e6e9ee;
    border-radius: __RADIUS_MD__px;
}

/* Canvas status bar — slim strip at the bottom of the central viewport. */
QWidget#CanvasStatusBar {
    background: #f8fafb;
    border-top: 1px solid #e6e9ee;
}
QLabel#CanvasStatusItem {
    color: #6b7280;
    font-size: 11px;
    padding: 0px 4px;
}

/* Shortcuts overlay dialog (? key). */
QDialog#ShortcutsOverlay {
    background: #ffffff;
    border: 1px solid #e6e9ee;
    border-radius: 8px;
}
QLabel#ShortcutsTitle {
    color: #1a1f29;
    font-size: 16px;
    font-weight: 700;
    padding-bottom: 4px;
    border-bottom: 1px solid #e6e9ee;
}
QLabel#ShortcutsGroup {
    color: #2563eb;
    font-size: 11px;
    font-weight: 700;
    padding-top: 8px;
    margin-bottom: 2px;
}
QLabel#ShortcutsKey {
    color: #1a1f29;
    font-family: "Consolas", "Menlo", monospace;
    font-size: 11px;
    font-weight: 600;
    background: #f3f4f6;
    border: 1px solid #e6e9ee;
    border-radius: 3px;
    padding: 2px 6px;
}
QLabel#ShortcutsDesc {
    color: #4b5563;
    font-size: 12px;
}
QLabel#ShortcutsHint {
    color: #9ca3af;
    font-size: 11px;
    padding-top: 8px;
    border-top: 1px solid #e6e9ee;
}

/* Toast notification floating at the bottom-right of the canvas. */
QLabel#ToastNotification {
    background: #1f2937;
    color: #f3f4f6;
    border-radius: 6px;
    padding: 8px 14px;
    font-size: 12px;
    font-weight: 500;
}
QLabel#ToastNotification[toastKind="success"] {
    background: #166534;
    color: #ecfdf5;
}
QLabel#ToastNotification[toastKind="warn"] {
    background: #b45309;
    color: #fffbeb;
}
QLabel#ToastNotification[toastKind="error"] {
    background: #b91c1c;
    color: #fef2f2;
}

/* Mini-map: bird's-eye overview anchored in the corner of the canvas. */
QWidget#MiniMap {
    background: rgba(20, 24, 32, 235);
    border: 1px solid rgba(80, 90, 110, 220);
    border-radius: 4px;
}

/* Selection mini-toolbar — appears above the selected part. */
QWidget#SelectionMiniToolbar {
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 6px;
}
QToolButton#MiniToolbarButton {
    background: transparent;
    color: #f3f4f6;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 13px;
}
QToolButton#MiniToolbarButton:hover {
    background: rgba(255, 255, 255, 0.12);
    border-color: rgba(255, 255, 255, 0.15);
}
QToolButton#MiniToolbarButton:pressed {
    background: rgba(255, 255, 255, 0.20);
}

/* Modern × close button used in panel headers (left + right docks). */
QToolButton#PanelCloseButton {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    color: #6b7280;
    font-size: 13px;
    font-weight: 600;
    padding: 0px;
}
QToolButton#PanelCloseButton:hover {
    background: #fee2e2;
    color: #b91c1c;
    border-color: #fecaca;
}
QToolButton#PanelCloseButton:pressed {
    background: #fecaca;
    color: #991b1b;
}

/* Thin click-to-show edge rail used when a side panel is hidden. */
QToolButton#PanelEdgeRail {
    background: #eef1f5;
    border: 1px solid #dde3e8;
    border-radius: 0px;
    color: #6b7280;
    font-size: 11px;
    padding: 0px;
}
QToolButton#PanelEdgeRail:hover {
    background: #dbeafe;
    color: #2563eb;
    border-color: #bfdbfe;
}
QToolButton#PanelEdgeRail:pressed {
    background: #bfdbfe;
    color: #1d4ed8;
}

QMenuBar {
    background: #eef1f4;
    border-bottom: 1px solid #d7dde3;
    padding: 0px;
    spacing: 0px;
}
QMenuBar::item {
    padding: 3px 10px;
    margin: 0px;
    border-radius: 0px;
}
QMenuBar::item:selected {
    background: #e8edf1;
}
QMenu {
    background: #fdfefe;
    border: 1px solid #d7dde3;
    padding: 6px;
}
QMenu::item {
    padding: 6px 24px;
    border-radius: 0px;
}
QMenu::item:selected {
    background: #edf1f5;
    color: #263341;
}

QDockWidget#PrimitiveDock {
    titlebar-close-icon: url(none);
    titlebar-normal-icon: url(none);
}
QDockWidget#PrimitiveDock::title {
    background: #eef1f4;
    color: #2d3a46;
    text-align: left;
    padding-left: 8px;
    border-bottom: 1px solid #d7dde3;
}
QDockWidget QWidget {
    padding: 4px;
}
QWidget#PropertiesPanel, QWidget#PropertyInspectorPanel {
    padding: 2px;
}
QWidget#PropertiesPanel QPushButton,
QWidget#PropertiesPanel QToolButton {
    padding: 2px 8px;
    min-height: 28px;
}
QWidget#PropertiesPanel QComboBox,
QWidget#PropertiesPanel QLineEdit,
QWidget#PropertiesPanel QAbstractSpinBox {
    min-height: 26px;
    padding: 2px 6px;
}

QStatusBar {
    background: #eef1f4;
    border-top: 1px solid #d7dde3;
}
QDialog {
    background: #f6f7f9;
}
QDialog QPushButton {
    min-width: 96px;
}

QToolTip {
    background: #ffffff;
    color: #18212b;
    border: 1px solid #d7dde3;
    padding: 4px 6px;
}

QProgressBar#JobProgressBar {
    border: 1px solid #c8d0d8;
    border-radius: 0px;
    text-align: center;
    background: #f1f4f6;
}
QProgressBar#JobProgressBar::chunk {
    background: #2f9e66;
    border-radius: 0px;
}
QPlainTextEdit#HintBox {
    font-size: __HINT_FONT_SIZE__px;
    background-color: #fafbfc;
    border: 1px solid #d7dde3;
    border-radius: __RADIUS_SM__px;
}

QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #ffffff;
    border: 1px solid #d2d9df;
    border-radius: __RADIUS_SM__px;
    min-height: 30px;
    padding: 4px 8px;
}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #657d93;
}
QComboBox::drop-down {
    border: 0px;
    width: 22px;
}

QPushButton {
    background: #ffffff;
    border: 1px solid #d2d9df;
    border-radius: __RADIUS_MD__px;
    min-height: 24px;
    padding: 4px 8px;
}
QPushButton:hover {
    background: #f3f6f8;
    border-color: #b9c4ce;
}
QPushButton:pressed {
    background: #e9edf1;
}
QPushButton[primary="true"] {
    background: #44586c;
    border-color: #44586c;
    color: #ffffff;
    font-weight: 700;
}
QPushButton[primary="true"]:hover {
    background: #394b5d;
    border-color: #394b5d;
}
QPushButton[dockIconButton="true"], QToolButton[dockIconButton="true"] {
    min-width: 32px;
    min-height: 32px;
    max-width: 36px;
    max-height: 36px;
    padding: 4px;
    border: 1px solid #d2d9df;
    border-radius: __RADIUS_SM__px;
    background: #f7f8fa;
}
QPushButton[dockIconButton="true"]:hover, QToolButton[dockIconButton="true"]:hover {
    background: #edf1f4;
    border-color: #bcc6cf;
}
QPushButton[dockIconButton="true"]:pressed,
QPushButton[dockIconButton="true"]:checked,
QToolButton[dockIconButton="true"]:pressed,
QToolButton[dockIconButton="true"]:checked {
    background: #e2e8ee;
    border-color: #aab6c1;
}
QPushButton[primary="true"]:pressed {
    background: #2f3f4e;
}
QPushButton[secondary="true"] {
    background: #fafbfc;
}

QGroupBox {
    border: 1px solid #e6e9ee;
    border-radius: __RADIUS_MD__px;
    background: #fafbfc;
    margin-top: 14px;
    padding-top: 6px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    top: 0px;
    left: 10px;
    padding: 0 4px;
    color: #1a1f29;
    font-weight: 600;
}

/* Thin section divider used inside dock panels — softer than default frame
   shadow so it reads as a hairline rather than a chunky bar. */
QFrame#DockSectionSeparator {
    background: #e6e9ee;
    border: none;
    max-height: 1px;
    min-height: 1px;
    margin: 2px 0px;
}

QTabWidget::pane {
    border: 1px solid #e6e9ee;
    border-radius: 0px;
    background: #fcfdfd;
}
QTabBar::tab {
    background: #f0f3f5;
    color: #56626d;
    border: 1px solid transparent;
    border-bottom: 2px solid transparent;
    border-top-left-radius: 0px;
    border-top-right-radius: 0px;
    padding: 4px 8px;
    margin-right: 2px;
}
/* Modern underline indicator for active tab — replaces the heavy
   background-color swap with a thin accent bar at the bottom. */
QTabBar::tab:selected {
    background: transparent;
    color: #1a1f29;
    font-weight: 700;
    border-bottom: 2px solid #2563eb;
}
QTabBar::tab:hover:!selected {
    background: #f1f5f9;
    border-bottom: 2px solid #cbd5e1;
}

QTreeWidget, QListWidget, QTreeView, QListView {
    background: #ffffff;
    alternate-background-color: #ffffff;
    border: 1px solid #dde3e8;
    border-radius: 0px;
    padding: 2px;
}
QTreeWidget::item, QListWidget::item {
    min-height: 24px;
    padding: 3px 2px;
    border-radius: 0px;
    background: transparent;
}
QTreeWidget::item:hover, QListWidget::item:hover {
    background: transparent;
}
QTreeWidget::item:selected, QListWidget::item:selected {
    background: transparent;
    color: #1f2a35;
}
QHeaderView::section {
    background: #f5f7f9;
    color: #56626d;
    padding: 6px 8px;
    border: 0px;
    border-bottom: 1px solid #dde3e8;
    font-weight: 700;
}
QScrollArea {
    border: 0px;
    background: transparent;
}

QScrollBar:vertical, QScrollBar:horizontal {
    background: #eef1f4;
    border: none;
    margin: 0;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #c2cbd4;
    border-radius: 0px;
    min-height: 24px;
    min-width: 24px;
}
QScrollBar::handle:hover:vertical, QScrollBar::handle:hover:horizontal {
    background: #a8b4bf;
}

QToolBar#SketchToolbar,
QToolBar#GeometryToolbar,
QToolBar#MainToolbar,
QToolBar#LoadsToolbar,
QToolBar#WorkflowRibbon,
QToolBar#StageToolbar {
    background: #f8fafb;
    border-bottom: 1px solid #d9dfe5;
    spacing: 0px;
    padding: 0px 4px;
    margin: 0px;
}
QToolBar#MainToolbar {
    /* Tall enough that the 30 px workflow-stage buttons fit fully (including
       letter descenders) after the toolbar's own content margins are taken
       out — 34 px min keeps ~30 px usable; 38 px max stays compact. */
    min-height: 34px;
    max-height: 38px;
}
QToolBar#WorkflowRibbon,
QToolBar#StageToolbar {
    min-height: 36px;
    max-height: 40px;
}
QToolBar#SketchToolbar QToolButton,
QToolBar#GeometryToolbar QToolButton,
QToolBar#MainToolbar QToolButton,
QToolBar#LoadsToolbar QToolButton {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 0px;
    min-height: 26px;
    min-width: 26px;
    padding: 1px;
    margin: 0px;
}
QToolBar#SketchToolbar QToolButton:hover,
QToolBar#GeometryToolbar QToolButton:hover,
QToolBar#MainToolbar QToolButton:hover,
QToolBar#LoadsToolbar QToolButton:hover {
    background: #edf1f4;
    border: 1px solid #c0c9d2;
    color: #2d3a46;
}
QToolBar#SketchToolbar QToolButton:checked,
QToolBar#GeometryToolbar QToolButton:checked,
QToolBar#MainToolbar QToolButton:checked,
QToolBar#LoadsToolbar QToolButton:checked {
    background: #e4eaef;
    border: 1px solid #aeb9c3;
    color: #263341;
}
QToolBar::separator:horizontal {
    background: #d7dde3;
    width: 1px;
    margin: 2px 3px;
}
QToolBar::separator:vertical {
    background: #d7dde3;
    height: 1px;
    margin: 6px 2px;
}
QToolButton[workflowStage="true"] {
    min-height: 28px;
    max-height: 28px;
    min-width: 28px;
    max-width: 28px;
    padding: 2px;
    border: 1px solid transparent;
    border-radius: 0px;
    background: transparent;
    color: #56626d;
    margin: 0px;
}
QToolButton[workflowStage="true"]:hover {
    background: #f0f3f5;
    border-color: #d8dee4;
}
QToolButton[workflowStage="true"][workflowState="completed"] {
    background: #fafbfc;
    border-color: #e1e6eb;
    color: #62707c;
}
QToolButton[workflowStage="true"][workflowState="future"] {
    color: #98a2ab;
}
QToolButton[workflowStage="true"]:checked,
QToolButton[workflowStage="true"][workflowState="current"] {
    background: #e7edf1;
    border-color: #b9c4ce;
    color: #263341;
}
QToolButton[workflowStage="true"][workflowState="future"]:hover {
    color: #6b7782;
}
QToolButton[workflowStage="true"]::menu-indicator {
    image: none;
}
/* ------------------------------------------------------------------ *
 * Workflow stage bar (the Geometry / Materials / ... tab row).
 *
 * This selector has higher specificity (two IDs: #MainToolbar and
 * #WorkflowStageBar) than the generic "QToolBar#MainToolbar QToolButton"
 * rule above, so it WINS and fully governs the stage buttons.  Without it
 * the generic rule's min-height/padding/border applied instead and clipped
 * the text descenders.
 *
 * Height bookkeeping (must equal the 30 px fixed height set in stage_bar.py
 * so the painted box equals the laid-out box — otherwise descenders clip):
 *     min-height 22 + padding (3+3) + border-bottom 2 = 30 px.
 * ------------------------------------------------------------------ */
QToolBar#MainToolbar QWidget#WorkflowStageBar QToolButton {
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    border-radius: 0px;
    padding: 3px 10px;
    margin: 0px;
    min-height: 22px;
    min-width: 0px;
    max-width: 16777215px;
    color: #374151;
    font-size: 12px;
    font-weight: 600;
}
QToolBar#MainToolbar QWidget#WorkflowStageBar QToolButton:hover {
    background: #eef1f4;
    color: #111827;
    border-bottom: 2px solid #b0bbc5;
}
QToolBar#MainToolbar QWidget#WorkflowStageBar QToolButton:checked,
QToolBar#MainToolbar QWidget#WorkflowStageBar QToolButton[active="true"] {
    background: transparent;
    color: #1d4ed8;
    border-bottom: 2px solid #2563eb;
}
QToolBar#MainToolbar QWidget#WorkflowStageBar QToolButton:disabled {
    background: transparent;
    color: #5b6675;
    border-bottom: 2px solid transparent;
}
QToolBar#MainToolbar QWidget#WorkflowStageBar QToolButton:disabled:hover {
    background: transparent;
    color: #5b6675;
    border-bottom: 2px solid transparent;
}
QLabel#WorkflowStageLabel {
    color: #2d3a46;
    font-weight: 700;
    padding-left: 10px;
}

QToolButton#SectionHeaderButton {
    font-weight: 700;
    color: #2d3a46;
    border: 1px solid #d7dde3;
    border-radius: 0px;
    background: #f5f7f9;
    padding: 4px 6px;
}
QToolButton#SectionHeaderButton:checked {
    background: #eceff2;
}
QToolButton#SectionHeaderButton:hover {
    background: #edf1f4;
}

QToolButton[stageBtn="true"] {
    min-height: 28px;
    min-width: 28px;
    max-height: 32px;
    max-width: 32px;
    padding: 3px;
    border: 1px solid #d2d9df;
    border-radius: 0px;
    background: #f5f7f9;
    color: #2d3a46;
}
QToolButton[stageBtn="true"]:hover {
    background: #edf1f4;
}
QToolButton[stageBtn="true"]:checked {
    background: #e5eaee;
    border: 1px solid #aeb9c3;
    border-left: 3px solid #516678;
    color: #18212b;
}
QToolButton[dockSectionTab="true"] {
    min-height: 32px;
    min-width: 32px;
    max-height: 32px;
    max-width: 32px;
    padding: 4px;
}

QPushButton#PrimitiveDockIconButton,
QToolButton#PrimitiveDockIconButton {
    background: #fafbfc;
    border: 1px solid #d2d9df;
    border-radius: __RADIUS_MD__px;
    padding: 0px;
}
QPushButton#PrimitiveDockIconButton:hover,
QToolButton#PrimitiveDockIconButton:hover {
    background: #f0f3f5;
    border-color: #bcc6cf;
}
QPushButton#PrimitiveDockIconButton:pressed,
QToolButton#PrimitiveDockIconButton:pressed {
    background: #e7edf1;
}

QListWidget#PrimitiveList {
    background: #ffffff;
    border: 1px solid #d7dde3;
    border-radius: 0px;
}
QListWidget#PrimitiveList::item:selected {
    background: #e7edf1;
    color: #263341;
}
"""
    replacements = {
        "__RADIUS_SM__": str(t.radius_sm),
        "__RADIUS_MD__": str(t.radius_md),
        "__RADIUS_LG__": str(t.radius_lg),
        "__TAB_RADIUS__": str(t.radius_sm + 1),
        "__SMALL_FONT_SIZE__": str(t.small_font_size),
        "__HINT_FONT_SIZE__": str(max(9, t.font_size - 1)),
        "__SPACING_SM__": str(t.spacing_sm),
    }
    for key, value in replacements.items():
        style = style.replace(key, value)
    return style


def build_toolbar_style() -> str:
    t = UI_TOKENS
    return (
        "QToolBar { spacing: 0px; padding: 1px 4px; }"
        "QToolButton {"
        "  min-height: 26px;"
        "  min-width: 26px;"
        "  padding: 2px 4px;"
        "  margin: 0px 1px;"
        "  border: 1px solid transparent;"
        f"  border-radius: {t.radius_sm}px;"
        "  background: transparent;"
        "  color: #2d3a46;"
        "}"
        "QToolButton:hover {"
        "  background: #edf1f4;"
        "  border-color: #c6ced6;"
        "}"
        "QToolButton:checked, QToolButton:pressed {"
        "  background: #e5eaee;"
        "  border-color: #aeb9c3;"
        "  color: #18212b;"
        "}"
        "QToolBar::separator:horizontal {"
        "  width: 1px;"
        "  margin: 2px 3px;"
        "  background: #d7dde3;"
        "}"
        "QToolBar::separator:vertical {"
        "  height: 1px;"
        "  margin: 3px 2px;"
        "  background: #d7dde3;"
        "}"
    )


def build_stage_bar_style() -> str:
    """Stage-bar tab ribbon.  Every button uses the same font-size and
    font-weight so none looks larger than another.  Active stage is shown
    only by the blue bottom border and a darker text colour — never by a
    different weight or size.  Disabled/future stages use a readable grey
    instead of the near-invisible #cbd5e1."""
    return (
        "QToolButton {"
        "  background: transparent;"
        "  border: none;"
        "  border-bottom: 2px solid transparent;"
        "  border-radius: 0px;"
        # No QSS min-height: it would add to padding+border and exceed the
        # button's fixed height, making Qt lay out text for a taller box than
        # it paints — which clips letter descenders (the 'y' in Geometry /
        # Boundary).  The fixed height in stage_bar.py governs the size instead.
        "  padding: 4px 10px;"
        "  margin: 0px;"
        "  color: #374151;"
        "  font-size: 12px;"
        "  font-weight: 600;"
        "}"
        "QToolButton:hover {"
        "  background: #eef1f4;"
        "  color: #111827;"
        "  border-bottom: 2px solid #b0bbc5;"
        "}"
        "QToolButton:checked, QToolButton[active=\"true\"] {"
        "  background: transparent;"
        "  color: #1d4ed8;"
        "  font-size: 12px;"
        "  font-weight: 600;"
        "  border-bottom: 2px solid #2563eb;"
        "}"
        "QToolButton:disabled {"
        "  background: transparent;"
        "  color: #5b6675;"
        "  font-size: 12px;"
        "  font-weight: 600;"
        "  border-bottom: 2px solid transparent;"
        "}"
        "QToolButton:disabled:hover {"
        "  background: transparent;"
        "  color: #5b6675;"
        "  border-bottom: 2px solid transparent;"
        "}"
    )


def toolbar_icon_size() -> int:
    return int(UI_TOKENS.toolbar_icon_size)


def primitive_icon_size() -> int:
    return int(UI_TOKENS.primitive_icon_size)


def primitive_button_size() -> int:
    return int(UI_TOKENS.primitive_button_size)


def apply_professional_theme(app, theme: str = "light") -> str:
    # Placeholder for future dark/custom themes; currently a polished light theme.
    _ = theme
    app.setStyle("Fusion")
    app.setPalette(_build_palette())
    app.setStyleSheet(_build_stylesheet())
    app.setFont(QFont(UI_TOKENS.font_family, UI_TOKENS.font_size))
    return build_toolbar_style()
