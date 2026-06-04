import os
import subprocess
import time
import traceback

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ValidationPanel(QDockWidget):
    def __init__(self, manager):
        super().__init__("Validation Mode", manager.main_window)
        self.manager = manager
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetClosable)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        title = QLabel("<b>CPU SimStudio UI Validation</b>")
        layout.addWidget(title)

        self._form_layout = QFormLayout()
        self._status_labels = {}
        layout.addLayout(self._form_layout)

        self._run_btn = QPushButton("Run UI Self-Checks")
        self._demo_btn = QPushButton("Open 3D Demo Scene")
        self._report_btn = QPushButton("Print Validation Report")
        btn_row = QHBoxLayout()
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._demo_btn)
        btn_row.addWidget(self._report_btn)
        layout.addLayout(btn_row)

        layout.addWidget(QLabel("<b>Action Log</b>"))
        self.log_area = QPlainTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumBlockCount(2000)
        layout.addWidget(self.log_area, 1)

        self.setWidget(container)

        self._run_btn.clicked.connect(manager.run_ui_checks)
        self._demo_btn.clicked.connect(manager.open_demo_scene)
        self._report_btn.clicked.connect(manager.print_validation_report)

    def add_check_row(self, label_text):
        label = QLabel(label_text)
        status_label = QLabel("PENDING")
        status_label.setStyleSheet("color: #1e293b;")
        self._form_layout.addRow(label, status_label)
        self._status_labels[label_text] = status_label
        return status_label

    def update_status_label(self, label_text, status, note=""):
        widget = self._status_labels.get(label_text)
        if widget is None:
            return
        colors = {"PASS": "#16a34a", "FAIL": "#dc2626", "PENDING": "#1e293b"}
        widget.setText(status)
        widget.setStyleSheet(f"color: {colors.get(status, '#1e293b')};")
        widget.setToolTip(note)

    def append_log(self, text):
        self.log_area.appendPlainText(text)


class ValidationScene:
    def __init__(self):
        self.nodes = np.array(
            [
                [-10.0, -10.0, -10.0],
                [10.0, -10.0, -10.0],
                [-10.0, 10.0, -10.0],
                [10.0, 10.0, -10.0],
                [-10.0, -10.0, 10.0],
                [10.0, -10.0, 10.0],
                [-10.0, 10.0, 10.0],
                [10.0, 10.0, 10.0],
            ],
            dtype=float,
        )
        # 12 triangles (two per face)
        self.faces = np.array(
            [
                [0, 1, 2],
                [1, 3, 2],
                [4, 6, 5],
                [5, 6, 7],
                [0, 4, 1],
                [1, 4, 5],
                [2, 3, 6],
                [3, 7, 6],
                [0, 2, 4],
                [2, 6, 4],
                [1, 5, 3],
                [3, 5, 7],
            ],
            dtype=int,
        )

    def load_into(self, view):
        view.clear_mesh()
        view.set_mesh(self.nodes, self.faces)
        view.set_visibility(show_nodes=True, show_mesh=True)
        view.set_nodes_visible(show_nodes=True, show_surface=True, show_interior=True)
        view.set_mesh_dim(True)
        view.set_mesh_xray(False)
        view.set_wireframe_visible(True)
        view.set_material_style("silver")
        view.opts["distance"] = 120
        try:
            view.setCameraPosition(pos=view._center, distance=120, azimuth=30, elevation=25)
        except Exception:
            view.set_view(30, 25)
        view.set_show_all_nodes(True)
        view.set_hover_nodes(self.nodes)
        # sync global nodes (if used elsewhere)
        view.global_nodes_3d = self.nodes
        view.global_elements_3d = self.faces
        view.mesh3dUpdated.emit(view.global_nodes_3d, view.global_elements_3d)


class ValidationManager:
    CHECKS = [
        ("Dock Resizing", "dock_resize"),
        ("Selection Modes", "selection"),
        ("BC/Loads Wiring", "bc_loads"),
        ("Mesh Visibility Toggles", "mesh_toggles"),
    ]

    def __init__(self, main_window):
        self.main_window = main_window
        self.scene = ValidationScene()
        self.panel = ValidationPanel(self)
        self.main_window.addDockWidget(Qt.RightDockWidgetArea, self.panel)
        self._status_states = {key: "PENDING" for _, key in self.CHECKS}
        self._status_notes = {}
        self._log_records = []
        for label_text, _key in self.CHECKS:
            self.panel.add_check_row(label_text)
        self.log("Validation mode enabled.")
        self.open_demo_scene()

    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self._log_records.append(entry)
        self.panel.append_log(entry)

    def _set_status(self, key, status, note=""):
        self._status_states[key] = status
        self._status_notes[key] = note
        label_text = next(label for label, name in self.CHECKS if name == key)
        self.panel.update_status_label(label_text, status, note)

    def _set_all_pending(self):
        for _, key in self.CHECKS:
            self._set_status(key, "PENDING", "")

    def run_ui_checks(self):
        self._set_all_pending()
        self.log("Starting UI self-checks...")
        self.open_demo_scene()
        for label, key in self.CHECKS:
            method = getattr(self, f"_check_{key}", None)
            if method is None:
                self._set_status(key, "FAIL", "Check not implemented.")
                continue
            try:
                method()
                self._set_status(key, "PASS", "All validations succeeded.")
                self.log(f"{label} check passed.")
            except Exception as exc:
                note = str(exc)
                tb = traceback.format_exc()
                self._set_status(key, "FAIL", note)
                self.log(f"{label} check failed: {note}")
                self.log(tb)
        self.log("UI self-checks completed.")

    def open_demo_scene(self):
        self.log("Loading deterministic demo scene...")
        self.main_window._toggle_workspace_3d(True)
        self.scene.load_into(self.main_window.view_3d)
        self.log("Demo scene ready.")

    def print_validation_report(self):
        report_lines = [
            f"Validation Report - {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Git commit: {self._git_hash() or 'unknown'}",
            "",
        ]
        for label, key in self.CHECKS:
            status = self._status_states.get(key, "PENDING")
            note = self._status_notes.get(key, "")
            report_lines.append(f"{label}: {status}{' - ' + note if note else ''}")
        report_lines.append("")
        report_lines.append("Recent log entries:")
        report_lines.extend(self._log_records[-20:])

        report = "\n".join(report_lines)
        print(report)
        report_path = os.path.join(os.getcwd(), "ui_validation_report.txt")
        try:
            with open(report_path, "w", encoding="utf-8") as fh:
                fh.write(report)
            self.log(f"Validation report written to {report_path}")
        except Exception as exc:
            self.log(f"Failed to write validation report: {exc}")

    def _git_hash(self):
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except Exception:
            return None

    def _check_dock_resize(self):
        dock = self.main_window.primitive_dock
        scroll_area = getattr(self.main_window, "shape_scroll_area", None)
        buttons = getattr(self.main_window, "_shape_buttons", [])
        if dock is None or scroll_area is None:
            raise RuntimeError("Primitive dock not available for validation checks.")
        widths = [140, 180, 240, 320]
        original_min = dock.minimumWidth()
        original_max = dock.maximumWidth()
        original_width = dock.width()
        try:
            for width in widths:
                dock.setFixedWidth(width)
                QApplication.processEvents()
                scroll_area.resize(scroll_area.width(), scroll_area.height())
                if scroll_area.horizontalScrollBar().isVisible():
                    raise AssertionError(f"Horizontal scrollbar visible at width {width}.")
                container = scroll_area.widget()
                viewport = scroll_area.viewport()
                if container and viewport and container.width() > viewport.width() + 2:
                    raise AssertionError(
                        f"Flow layout overflow at width {width}: container {container.width()} vs viewport {viewport.width()}".
                    )
                self.main_window._update_shape_button_labels(width)
                thresholded = width >= 260
                for btn, label in buttons:
                    text = btn.text()
                    if thresholded:
                        if text != label:
                            raise AssertionError(
                                f"Label not restored ({text!r}) at width {width}."
                            )
                    else:
                        if text:
                            raise AssertionError(
                                f"Label still visible ({text!r}) at narrow width {width}."
                            )
                self.log(f"Dock resize at {width}px verified.")
        finally:
            dock.setFixedWidth(original_width)
            dock.setMinimumWidth(original_min)
            dock.setMaximumWidth(original_max)
            QApplication.processEvents()

    def _check_selection(self):
        view = self.main_window.view_3d
        if view is None:
            raise RuntimeError("3D view not initialized.")
        self.open_demo_scene()
        center_x = view.width() * 0.5 or 200
        center_y = view.height() * 0.5 or 200
        modes = ["face", "edge", "point"]
        for mode in modes:
            view.set_selection_mode(mode)
            idx = self._simulate_pick(view, center_x, center_y, mode)
            if mode == "face":
                if idx is None:
                    raise AssertionError("No face picked.")
                view.selected_faces = {idx}
                view._update_face_highlight()
                if view._selection_highlight is None:
                    raise AssertionError("Face highlight missing.")
                view.selected_faces.remove(idx)
                view._update_face_highlight()
                if view.selected_faces:
                    raise AssertionError("Face selection did not clear after toggle.")
            elif mode == "point":
                if idx is None:
                    raise AssertionError("No point picked.")
                view.selected_nodes = {idx}
                view._update_selected_nodes_marker()
                if view._selected_nodes_item is None:
                    raise AssertionError("Point highlight missing.")
                view.selected_nodes.clear()
                view._update_selected_nodes_marker()
                if view.selected_nodes:
                    raise AssertionError("Point selection did not clear.")
            else:  # edge
                if idx is None:
                    raise AssertionError("No face available to derive edge.")
                origin, direction = self._ray_from_view(view, center_x, center_y)
                edge = view._closest_edge_to_ray(idx, origin, direction)
                if edge is None:
                    raise AssertionError("Failed to compute edge from ray.")
                edge = (min(edge[0], edge[1]), max(edge[0], edge[1]))
                view.selected_edges = {edge}
                view._update_edge_highlight()
                if view._selected_edges_item is None:
                    raise AssertionError("Edge highlight missing.")
                view.selected_edges.clear()
                view._update_edge_highlight()
                if view.selected_edges:
                    raise AssertionError("Edge selection did not clear.")
            self.log(f"Selection mode {mode} exercised.")
        view.clear_selection()
        view.clear_node_selection()
        view.clear_edge_selection()
        if view.get_selected_faces() or view.get_selected_edges() or view.get_selected_nodes():
            raise AssertionError("Selection clearing did not reset all sets.")

    def _simulate_pick(self, view, x, y, mode):
        if mode == "face":
            return view._pick_face(x, y)
        if mode == "point":
            return view._pick_point(x, y)
        if mode == "edge":
            return view._pick_face(x, y)
        return None

    def _ray_from_view(self, view, x, y):
        p1 = view._unproject(x, y, 0.0)
        p2 = view._unproject(x, y, 1.0)
        dir_vec = np.array([p2.x() - p1.x(), p2.y() - p1.y(), p2.z() - p1.z()], dtype=float)
        length = np.linalg.norm(dir_vec)
        if length < 1e-9:
            length = 1.0
        direction = dir_vec / length
        origin = np.array([p1.x(), p1.y(), p1.z()], dtype=float)
        return origin, direction

    def _check_bc_loads(self):
        loads_tab = self.main_window.properties_panel.loads_tab
        view = self.main_window.view_3d
        if loads_tab is None or view is None:
            raise RuntimeError("Loads panel or 3D view unavailable.")
        self._simulate_pick(view, view.width() * 0.5, view.height() * 0.5, "face")
        view.selected_faces = {0}
        view._update_face_highlight()
        self.main_window.properties_panel.tabs.setCurrentWidget(loads_tab)
        for idx, label in enumerate(["face", "edge", "point"]):
            loads_tab.selection_target_combo.setCurrentIndex(idx)
            QApplication.processEvents()
            if view._selection_mode != label:
                raise AssertionError(f"Viewport selection mode not updated to {label}.")
            text = loads_tab.selection_status_label.text()
            if label == "face" and "Faces 1" not in text:
                raise AssertionError("Selection count label missing face count.")
        loads_tab._apply_bc_from_selection()
        bcs = self.main_window.view.bcs
        if not bcs:
            raise AssertionError("BC record not created.")
        record = bcs[-1]
        if record.get("target") != "face":
            raise AssertionError("BC target mismatched.")
        if not record.get("ids"):
            raise AssertionError("BC record contains no ids.")
        self.log("BC/Loads wiring verified.")

    def _check_mesh_toggles(self):
        view = self.main_window.view_3d
        if view is None:
            raise RuntimeError("3D viewport unavailable.")
        combos = [
            {"xray": True, "dim": True, "wireframe": True},
            {"xray": False, "dim": True, "wireframe": True},
            {"xray": True, "dim": False, "wireframe": False},
        ]
        checkboxes = {
            "xray": self.main_window.mesh_xray_checkbox,
            "dim": self.main_window.mesh_dim_checkbox,
            "wireframe": self.main_window.mesh_wireframe_checkbox,
        }
        for combo in combos:
            for key, checkbox in checkboxes.items():
                checkbox.setChecked(combo[key])
            QApplication.processEvents()
            if view._mesh_xray_enabled != combo["xray"]:
                raise AssertionError("Mesh x-ray state mismatch.")
            if view._mesh_dim_enabled != combo["dim"]:
                raise AssertionError("Mesh dim state mismatch.")
            if view._wireframe_enabled != combo["wireframe"]:
                raise AssertionError("Mesh wireframe state mismatch.")
            mesh_visible = combo["wireframe"] and view._wireframe_item is not None
            if combo["wireframe"] and not mesh_visible:
                raise AssertionError("Wireframe overlay missing when requested.")
            if not combo["wireframe"] and view._wireframe_item is not None:
                raise AssertionError("Wireframe overlay persisted after toggle off.")
            if view.mesh_item is None:
                raise AssertionError("Mesh item missing after toggle.")
            alpha = view.mesh_item.opts.get("color", (0, 0, 0, 0))[3]
            if combo["xray"] and alpha > 0.2:
                raise AssertionError("X-ray alpha not applied.")
            if not combo["xray"] and alpha < 0.4:
                raise AssertionError("Opaque alpha not restored.")
            self.log(f"Mesh toggle combo {combo} verified.")
        # restore defaults
        checkboxes["xray"].setChecked(False)
        checkboxes["dim"].setChecked(True)
        checkboxes["wireframe"].setChecked(False)
        QApplication.processEvents()

    # binding names to method names expected by run_ui_checks
    _check_selection = _check_selection
    _check_bc_loads = _check_bc_loads
    _check_mesh_toggles = _check_mesh_toggles
    _check_dock_resize = _check_dock_resize
