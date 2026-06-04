from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import QHBoxLayout, QSizePolicy, QToolButton, QWidget

from project_stages import ProjectStage
from ui_theme import UI_TOKENS


class StageBar(QWidget):
    stageRequested = Signal(ProjectStage)

    def __init__(self, parent=None, icon_only=True):
        super().__init__(parent)
        # Object name lets the GLOBAL stylesheet target these buttons with a
        # high-specificity selector (QToolBar#MainToolbar QWidget#WorkflowStageBar
        # QToolButton) that decisively overrides the generic
        # "QToolBar#MainToolbar QToolButton" rule — otherwise the generic rule
        # wins on specificity and our compact, no-clip styling is ignored.
        self.setObjectName("WorkflowStageBar")
        self._buttons = {}
        self._icon_only = icon_only
        self._all_stage_order = [
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
        self._stage_order = list(self._all_stage_order)
        layout = QHBoxLayout(self)
        self._layout = layout
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        # Preferred horizontally: uses natural button width, never forces the
        # parent toolbar to over-allocate.  Fixed vertically: stays one row.
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        label_map = {
            ProjectStage.GEOMETRY: "Geometry",
            ProjectStage.FLUID: "Fluid",
            ProjectStage.INTERFACES: "Interactions",
            ProjectStage.BCS: "Boundary Conditions",
            ProjectStage.FRACTURE: "Fracture",
            ProjectStage.MESH: "Particles",
            ProjectStage.JOB: "Solve",
        }
        # Uniform 16-px icon for all stages — symmetric visual rhythm.
        icon_size = 16 if not self._icon_only else UI_TOKENS.toolbar_icon_size
        for stage in self._all_stage_order:
            btn = QToolButton()
            label = label_map.get(stage, stage.name.title())
            btn.setText(label)
            btn.setCheckable(True)
            btn.setToolButtonStyle(
                Qt.ToolButtonIconOnly if self._icon_only else Qt.ToolButtonTextBesideIcon
            )
            btn.setIconSize(QSize(icon_size, icon_size))
            btn.setToolTip(label)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setAutoRaise(True)
            btn.setFocusPolicy(Qt.NoFocus)
            # Fixed 30 px height on every button so labels line up exactly.
            # The matching global QSS rule sets min-height:22 + padding:3px +
            # border-bottom:2px = 30 px, so the painted height equals the layout
            # height — text descenders (the tail of 'y' in Geometry / Boundary)
            # are never clipped.  See build _build_stylesheet() in ui_theme.py.
            btn.setMinimumHeight(30)
            btn.setMaximumHeight(30)
            btn.clicked.connect(lambda checked=False, s=stage: self._on_clicked(s))
            layout.addWidget(btn)
            self._buttons[stage] = btn

        # Trailing stretch absorbs any surplus width so buttons never expand.
        layout.addStretch(1)

        # No per-widget stylesheet: styling now comes solely from the global
        # high-specificity rule in ui_theme.py so there is a single source of
        # truth and no competing-stylesheet ambiguity.
        self.set_stage_order(self._stage_order)

    def set_icons(self, icon_map):
        for stage, icon in icon_map.items():
            btn = self._buttons.get(stage)
            if btn and icon:
                btn.setIcon(icon)

    def _on_clicked(self, stage):
        if self._buttons.get(stage) and self._buttons[stage].isEnabled():
            self.stageRequested.emit(stage)

    def set_stage_order(self, stages):
        ordered = []
        seen = set()
        for stage in list(stages or []):
            if stage in self._buttons and stage not in seen:
                ordered.append(stage)
                seen.add(stage)
        if not ordered:
            ordered = [ProjectStage.GEOMETRY]
            seen = {ProjectStage.GEOMETRY}
        self._stage_order = ordered
        # Remove all widgets (buttons + stretch) from the layout cleanly.
        for stage in self._all_stage_order:
            btn = self._buttons.get(stage)
            if btn:
                self._layout.removeWidget(btn)
                btn.setVisible(False)
        while self._layout.count():
            self._layout.takeAt(0)
        # Re-add only the visible stages in order, then the trailing stretch.
        for stage in self._stage_order:
            self._layout.addWidget(self._buttons[stage])
            self._buttons[stage].setVisible(True)
        self._layout.addStretch(1)

    def set_active_stage(self, active_stage):
        active_visible = active_stage if active_stage in self._stage_order else (self._stage_order[0] if self._stage_order else active_stage)
        active_index = self._stage_order.index(active_visible) if active_visible in self._stage_order else -1
        for stage, btn in self._buttons.items():
            if btn is None:
                continue
            if stage not in self._stage_order:
                btn.setVisible(False)
                continue
            btn.setVisible(True)
            stage_index = self._stage_order.index(stage)
            btn.setEnabled(stage_index <= active_index)
            btn.setProperty("active", stage == active_visible)
            btn.setChecked(stage == active_visible)
            btn.style().unpolish(btn)
            btn.style().polish(btn)
