from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractScrollArea,
    QHeaderView,
    QMenu,
    QMessageBox,
    QTreeWidget,
    QTreeWidgetItem,
)

from project_stages import ProjectStage
from ui_icons import get_icon


class ProjectTree(QTreeWidget):
    objectSelected = Signal(object)

    def __init__(self, parent=None, sketch_view=None, project_state=None):
        super().__init__(parent)
        self._sketch_view = sketch_view
        self._project_state = project_state
        self._updating = False
        self._suppress_selection_activation = False
        self.setObjectName("ProjectTree")
        self.setHeaderHidden(True)
        self.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.setIndentation(10)
        self.setUniformRowHeights(True)
        self.setAnimated(False)
        self.setAlternatingRowColors(False)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setRootIsDecorated(True)
        self.setExpandsOnDoubleClick(False)
        self.itemClicked.connect(self._on_item_clicked)
        self.itemSelectionChanged.connect(self._on_selection_changed)

        self.root = QTreeWidgetItem(["Project"])
        self.addTopLevelItem(self.root)
        self.root.setExpanded(True)
        self.root.setFlags(self.root.flags() & ~Qt.ItemIsSelectable)
        root_font = self.root.font(0)
        root_font.setBold(True)
        self.root.setFont(0, root_font)

        self.model_item = QTreeWidgetItem(["Model"])
        self.model_item.setData(0, Qt.UserRole, {"kind": "group", "key": "model"})
        self.model_item.setFlags(self.model_item.flags() & ~Qt.ItemIsSelectable)
        model_font = self.model_item.font(0)
        model_font.setBold(True)
        self.model_item.setFont(0, model_font)
        self.root.addChild(self.model_item)

        self.category_items = {}
        self._category_unlock_stage = {}
        self._category_active_stages = {}
        categories = [
            (
                "geometry",
                self.model_item,
                "Geometry",
                "geometry",
                ProjectStage.GEOMETRY,
                ProjectStage.GEOMETRY,
                {ProjectStage.GEOMETRY},
            ),
            (
                "materials",
                self.model_item,
                "Materials",
                "materials",
                ProjectStage.MATERIALS,
                ProjectStage.MATERIALS,
                {ProjectStage.MATERIALS},
            ),
            (
                "interactions",
                self.model_item,
                "Interactions",
                "interactions",
                ProjectStage.INTERFACES,
                ProjectStage.INTERFACES,
                {ProjectStage.INTERFACES},
            ),
            (
                "mesh",
                self.model_item,
                "Mesh",
                "particles",
                ProjectStage.MESH,
                ProjectStage.MESH,
                {ProjectStage.MESH},
            ),
            (
                "analysis",
                self.model_item,
                "Analysis",
                "solve",
                ProjectStage.BCS,
                ProjectStage.BCS,
                {ProjectStage.FLUID, ProjectStage.INTERFACES, ProjectStage.FRACTURE, ProjectStage.BCS, ProjectStage.JOB, ProjectStage.RESULTS},
            ),
            (
                "analysis_settings",
                None,
                "Analysis Settings",
                "particles",
                ProjectStage.FLUID,
                ProjectStage.FLUID,
                {ProjectStage.FLUID, ProjectStage.FRACTURE},
            ),
            (
                "bcs",
                None,
                "Boundary Condition",
                "bc",
                ProjectStage.BCS,
                ProjectStage.BCS,
                {ProjectStage.BCS, ProjectStage.LOADS},
            ),
            (
                "solution",
                None,
                "Solution",
                "solve",
                ProjectStage.JOB,
                ProjectStage.JOB,
                {ProjectStage.JOB, ProjectStage.RESULTS},
            ),
            (
                "plots",
                None,
                "Plots",
                "results",
                ProjectStage.RESULTS,
                ProjectStage.RESULTS,
                {ProjectStage.RESULTS},
            ),
        ]
        for key, parent, label, icon_name, stage, unlock_stage, active_stages in categories:
            item = QTreeWidgetItem([label])
            item.setData(0, Qt.UserRole, {"kind": "category", "key": key, "stage": stage})
            item.setToolTip(0, label)
            item.setData(0, Qt.UserRole + 1, icon_name)
            self._set_item_icon(item, icon_name)
            if parent is None:
                if key == "plots":
                    parent = self.category_items["solution"]
                else:
                    parent = self.category_items["analysis"]
            parent.addChild(item)
            self.category_items[key] = item
            self._category_unlock_stage[key] = unlock_stage
            self._category_active_stages[key] = set(active_stages)
        self.stage_items = {
            ProjectStage.GEOMETRY: self.category_items["geometry"],
            ProjectStage.MATERIALS: self.category_items["materials"],
            ProjectStage.FLUID: self.category_items["analysis_settings"],
            ProjectStage.INTERFACES: self.category_items["interactions"],
            ProjectStage.BCS: self.category_items["bcs"],
            ProjectStage.FRACTURE: self.category_items["analysis_settings"],
            ProjectStage.MESH: self.category_items["mesh"],
            ProjectStage.JOB: self.category_items["solution"],
            ProjectStage.RESULTS: self.category_items["plots"],
        }
        self._visible_stages = set(self.stage_items.keys())

        self.session_root = QTreeWidgetItem(["Session Projects"])
        self.session_root.setHidden(True)
        self.session_root.setExpanded(True)
        self.session_root.setFlags(self.session_root.flags() & ~Qt.ItemIsSelectable)
        self.addTopLevelItem(self.session_root)

        self.refresh_from_model()

    def set_visible_stages(self, stages):
        normalized = set(stages or [])
        normalized.update(self.stage_items.keys())
        normalized.add(ProjectStage.GEOMETRY)
        self._visible_stages = normalized
        for item in self.category_items.values():
            item.setHidden(False)
        self.refresh_from_model()

    def set_sources(self, sketch_view=None, project_state=None):
        if sketch_view is not None:
            self._sketch_view = sketch_view
        if project_state is not None:
            self._project_state = project_state
        self.refresh_from_model()

    # ------------------------------------------------------------------
    # Dynamic width sizing
    # ------------------------------------------------------------------

    def _compute_preferred_width(self) -> int:
        """Return the minimum viewport pixel width needed to show every tree
        item without clipping its text.

        With setRootIsDecorated(True) and setIndentation(10) Qt renders item
        text starting at:
            visual_left = (depth + 1) * indentation()
        (confirmed via visualRect — depth-0 items land at left=10, depth-3
        items land at left=40).  The formula below matches that exactly.
        """
        base_fm = self.fontMetrics()
        indent  = self.indentation()       # 10 px per level
        icon_w  = 20                       # 16-px icon + 4-px gap
        r_pad   = 8                        # right breathing room
        max_w   = 80                       # minimum floor

        def _scan(item, depth: int) -> None:
            nonlocal max_w
            if item is None or item.isHidden():
                return
            text  = item.text(0)
            f     = item.font(0)
            fm    = QFontMetrics(f) if f.bold() else base_fm
            row_w = (depth + 1) * indent + icon_w + fm.horizontalAdvance(text) + r_pad
            if row_w > max_w:
                max_w = row_w
            for i in range(item.childCount()):
                _scan(item.child(i), depth + 1)

        for i in range(self.topLevelItemCount()):
            _scan(self.topLevelItem(i), 0)
        return max_w

    def sizeHint(self) -> QSize:
        w = self._compute_preferred_width()
        return QSize(w, super().sizeHint().height())

    def minimumSizeHint(self) -> QSize:
        w = self._compute_preferred_width()
        return QSize(w, 80)

    def _window(self):
        return self.window()

    def _view(self):
        win = self._window()
        return self._sketch_view or getattr(win, "view", None)

    def _state(self):
        win = self._window()
        return self._project_state or getattr(win, "project_state", None)

    def _panel(self):
        win = self._window()
        return getattr(win, "properties_panel", None) if win is not None else None

    def _category_label(self, label, count):
        _ = count
        return label

    def refresh_from_model(self):
        if self._updating:
            return
        selected_payload = None
        current_item = self.currentItem()
        if current_item is not None:
            payload = current_item.data(0, Qt.UserRole)
            if isinstance(payload, dict):
                selected_payload = payload
        self._updating = True
        try:
            view = self._view()
            state = self._state()
            parts = list(getattr(view, "parts", []) or getattr(state, "parts", []) or [])
            materials_store = getattr(state, "materials", {})
            if isinstance(materials_store, dict):
                materials = list(materials_store.values())
            else:
                materials = list(materials_store or [])
            interfaces = list(getattr(state, "interfaces", getattr(view, "interfaces", [])) or [])
            bcs = list(getattr(state, "boundary_conditions", getattr(view, "bcs", [])) or [])
            loads = list(getattr(state, "loads", getattr(view, "loads", [])) or [])

            for item in self.category_items.values():
                item.takeChildren()

            self._populate_geometry(parts)
            self._populate_materials(materials, parts)
            self._populate_fluid(materials, parts)
            self._populate_interactions(interfaces, materials_store)
            self._populate_bcs(bcs, loads)
            self._populate_fracture(materials, parts)
            self._populate_particles(view)
            self._populate_solve(view)
            self._populate_results(view)
            self.root.setExpanded(True)
            self.model_item.setExpanded(True)
            self.category_items["geometry"].setExpanded(True)
            self.category_items["analysis"].setExpanded(True)
            self.category_items["solution"].setExpanded(True)
            self._refresh_completion_states()
            if selected_payload is not None:
                self._restore_payload_selection(selected_payload)
        finally:
            self._updating = False
        # Notify the parent splitter that our preferred width may have changed.
        self.updateGeometry()

    @staticmethod
    def _iface_value(iface, key, default=None):
        if isinstance(iface, dict):
            return iface.get(key, default)
        return getattr(iface, key, default)

    def _populate_geometry(self, parts):
        category = self.category_items["geometry"]
        for part in parts:
            name = getattr(part, "name", None) or f"Part {getattr(part, 'id', '?')}"
            item = QTreeWidgetItem([name])
            item.setIcon(0, get_icon("geometry", size=16))
            item.setData(
                0,
                Qt.UserRole,
                {
                    "kind": "part",
                    "part_id": int(getattr(part, "id", -1)),
                    "stage": ProjectStage.GEOMETRY,
                },
            )
            item.setToolTip(0, f"Geometry part {name}")
            category.addChild(item)
        category.setText(0, "Geometry")
        category.setExpanded(True)

    def _populate_particles(self, view):
        category = self.category_items["mesh"]
        global_nodes = getattr(view, "global_nodes", None)
        global_elements = getattr(view, "global_elements", None)
        particle_count = len(global_nodes) if global_nodes is not None else 0
        connection_count = len(global_elements) if global_elements is not None else 0
        for label, value, kind in (
            ("Particles", particle_count, "mesh_nodes"),
            ("Connections", connection_count, "mesh_elements"),
        ):
            if value:
                item = QTreeWidgetItem([label])
                item.setIcon(0, get_icon("particles" if kind == "mesh_nodes" else "connections", size=16))
                item.setData(0, Qt.UserRole, {"kind": kind, "stage": ProjectStage.MESH})
                item.setToolTip(0, f"{label}: {value}")
                category.addChild(item)
        category.setExpanded(True)

    def _populate_materials(self, materials, parts):
        category = self.category_items["materials"]
        used_material_ids = set()
        for part in parts:
            mat_id = getattr(part, "material_id", None)
            if mat_id not in (None, "", -1):
                used_material_ids.add(mat_id)
            if str(getattr(part, "material_assignment_mode", "homogeneous")).lower() != "heterogeneous":
                continue
            config = getattr(part, "heterogeneity_config", {}) or {}
            for item in config.get("materials", []) or []:
                try:
                    extra_id = int(item.get("material_id"))
                except Exception:
                    continue
                used_material_ids.add(extra_id)
        used_materials = [
            mat for mat in materials
            if getattr(mat, "serial", None) in used_material_ids
            and not str(getattr(mat, "name", "")).startswith("IFACE_")
        ]
        category.setText(0, "Materials")
        for mat in used_materials:
            serial = getattr(mat, "serial", None)
            linked_parts = []
            for part in parts:
                direct_match = getattr(part, "material_id", None) == serial
                hetero_match = False
                if not direct_match:
                    for item in ((getattr(part, "heterogeneity_config", {}) or {}).get("materials", []) or []):
                        try:
                            hetero_match = int(item.get("material_id", -1)) == int(serial)
                        except Exception:
                            hetero_match = False
                        if hetero_match:
                            break
                if direct_match or hetero_match:
                    linked_parts.append(int(getattr(part, "id")))
            name = getattr(mat, "name", None) or f"Material {serial}"
            item = QTreeWidgetItem([name])
            item.setIcon(0, get_icon("materials", size=16))
            item.setData(
                0,
                Qt.UserRole,
                {
                    "kind": "material",
                    "serial": serial,
                    "part_ids": linked_parts,
                    "stage": ProjectStage.MATERIALS,
                },
            )
            item.setToolTip(0, name)
            category.addChild(item)
        category.setExpanded(True)

    def _populate_fluid(self, materials, parts):
        category = self.category_items["analysis_settings"]
        for mat in materials:
            behavior = str(getattr(mat, "behavior", "") or "").strip().lower()
            mat_type = str(getattr(mat, "mat_type", "") or "").strip().lower()
            if behavior != "fluid" and mat_type != "fluid":
                continue
            linked_parts = [
                int(getattr(part, "id"))
                for part in parts
                if getattr(part, "material_id", None) == getattr(mat, "serial", None)
                or str(getattr(part, "material_behavior", "") or "").strip().lower() == "fluid"
            ]
            label = getattr(mat, "name", f"Material {getattr(mat, 'serial', '?')}")
            item = QTreeWidgetItem([label])
            item.setIcon(0, get_icon("fluid", size=16))
            item.setData(
                0,
                Qt.UserRole,
                {
                    "kind": "material",
                    "serial": getattr(mat, "serial", None),
                    "part_ids": linked_parts,
                    "stage": ProjectStage.FLUID,
                },
            )
            item.setToolTip(0, label)
            category.addChild(item)
        category.setExpanded(True)

    def _populate_interactions(self, interfaces, materials_store=None):
        category = self.category_items["interactions"]
        materials_store = materials_store if isinstance(materials_store, dict) else {}
        interaction_material_ids = []
        for iface in interfaces:
            mat_id = self._iface_value(iface, "material_id", None)
            if mat_id in (None, "", -1):
                continue
            try:
                interaction_material_ids.append(int(mat_id))
            except Exception:
                continue
        interaction_materials = []
        seen_materials = set()
        for mat_id in interaction_material_ids:
            if mat_id in seen_materials:
                continue
            mat = materials_store.get(mat_id)
            if mat is None:
                continue
            seen_materials.add(mat_id)
            interaction_materials.append(mat)
        for idx, iface in enumerate(interfaces):
            name = self._iface_value(iface, "name")
            part1 = self._iface_value(iface, "part1_id")
            part2 = self._iface_value(iface, "part2_id")
            if name:
                label = str(name)
            elif part1 is not None or part2 is not None:
                label = f"Part {part1} <-> Part {part2}"
            else:
                label = f"Interaction {idx + 1}"
            item = QTreeWidgetItem([label])
            item.setIcon(0, get_icon("interactions", size=16))
            item.setData(
                0,
                Qt.UserRole,
                {
                    "kind": "interaction",
                    "index": idx,
                    "stage": ProjectStage.INTERFACES,
                },
            )
            item.setToolTip(0, label)
            category.addChild(item)
        for mat in interaction_materials:
            serial = getattr(mat, "serial", None)
            name = getattr(mat, "name", None) or f"Material {serial}"
            item = QTreeWidgetItem([name])
            item.setIcon(0, get_icon("materials", size=16))
            item.setData(
                0,
                Qt.UserRole,
                {
                    "kind": "material",
                    "serial": serial,
                    "stage": ProjectStage.INTERFACES,
                    "interaction_material": True,
                },
            )
            item.setToolTip(0, f"Interaction material {name}")
            category.addChild(item)
        category.setExpanded(True)

    def _populate_bcs(self, bcs, loads):
        category = self.category_items["bcs"]
        category.setText(0, "Boundary Condition")
        for idx, bc in enumerate(bcs):
            target = self._entry_target_text(bc)
            label = f"{bc.get('type', 'BC')} - {target}"
            item = QTreeWidgetItem([label])
            item.setIcon(0, get_icon("bc", size=16))
            item.setData(
                0,
                Qt.UserRole,
                {
                    "kind": "bc",
                    "index": idx,
                    "part_id": bc.get("part_id"),
                    "stage": ProjectStage.BCS,
                },
            )
            item.setToolTip(0, label)
            category.addChild(item)
        for idx, load in enumerate(loads):
            target = self._entry_target_text(load)
            label = f"{load.get('type', 'Load')} - {target}"
            item = QTreeWidgetItem([label])
            item.setIcon(0, get_icon("loads", size=16))
            item.setData(
                0,
                Qt.UserRole,
                {
                    "kind": "load",
                    "index": idx,
                    "part_id": load.get("part_id"),
                    "stage": ProjectStage.BCS,
                },
            )
            item.setToolTip(0, label)
            category.addChild(item)
        category.setExpanded(True)

    def _populate_fracture(self, materials, parts):
        category = self.category_items["analysis_settings"]
        for mat in materials:
            damage = str(getattr(mat, "damage", "none") or "none").lower()
            if damage == "none":
                continue
            linked_parts = [
                int(getattr(part, "id"))
                for part in parts
                if getattr(part, "material_id", None) == getattr(mat, "serial", None)
                or str(getattr(part, "material_damage", "none") or "none").lower() != "none"
            ]
            mat_name = getattr(mat, "name", None) or f"Material {getattr(mat, 'serial', '?')}"
            label = mat_name
            item = QTreeWidgetItem([label])
            item.setIcon(0, get_icon("fracture", size=16))
            item.setData(
                0,
                Qt.UserRole,
                {
                    "kind": "material",
                    "serial": getattr(mat, "serial", None),
                    "part_ids": linked_parts,
                    "stage": ProjectStage.FRACTURE,
                },
            )
            item.setToolTip(0, label)
            category.addChild(item)
        for part in parts:
            damage = str(getattr(part, "material_damage", "none") or "none").lower()
            if damage == "none":
                continue
            part_name = getattr(part, "name", None) or f"Part {getattr(part, 'id', '?')}"
            label = part_name
            item = QTreeWidgetItem([label])
            item.setIcon(0, get_icon("geometry", size=16))
            item.setData(
                0,
                Qt.UserRole,
                {
                    "kind": "part",
                    "part_id": int(getattr(part, "id", -1)),
                    "stage": ProjectStage.FRACTURE,
                },
            )
            item.setToolTip(0, label)
            category.addChild(item)
        category.setExpanded(True)

    def _populate_results(self, view):
        category = self.category_items["plots"]
        items = []
        field_label = str(getattr(view, "_result_field_label", "") or "")
        if str(getattr(view, "display_mode", "")) == "results":
            summary = "Results"
            if field_label:
                summary = field_label
            items.append({"label": summary, "kind": "results_current"})
        for entry in items:
            item = QTreeWidgetItem([entry["label"]])
            item.setIcon(0, get_icon("results", size=16))
            item.setData(0, Qt.UserRole, {"kind": entry["kind"], "stage": ProjectStage.RESULTS})
            item.setToolTip(0, entry["label"])
            category.addChild(item)
        category.setExpanded(True)

    def _populate_solve(self, view):
        category = self.category_items["solution"]
        items = []
        if getattr(view, "animation_frames", None):
            items.append({"label": "Output", "kind": "solve_output"})
        elif getattr(view, "global_nodes", None) is not None:
            items.append({"label": "Ready", "kind": "solve_ready"})
        for entry in items:
            item = QTreeWidgetItem([entry["label"]])
            item.setIcon(0, get_icon("solve", size=16))
            item.setData(0, Qt.UserRole, {"kind": entry["kind"], "stage": ProjectStage.JOB})
            item.setToolTip(0, entry["label"])
            category.addChild(item)
        category.setExpanded(True)

    def _set_item_icon(self, item, icon_name, completed=False):
        if item is None:
            return
        item.setIcon(0, get_icon(icon_name, size=16, state="completed" if completed else None))

    def _iter_tree_children(self, parent):
        if parent is None:
            return
        for row in range(parent.childCount()):
            item = parent.child(row)
            if item is None:
                continue
            yield item
            yield from self._iter_tree_children(item)

    def _refresh_completion_states(self):
        completed_node_count = 0
        for key, item in self.category_items.items():
            base_icon = item.data(0, Qt.UserRole + 1) or "geometry"
            has_real_children = any(
                isinstance(child.data(0, Qt.UserRole), dict)
                and child.data(0, Qt.UserRole).get("kind") != "category"
                for child in self._iter_tree_children(item)
            )
            if has_real_children:
                completed_node_count += 1
            self._set_item_icon(item, base_icon, completed=has_real_children)
        self._set_item_icon(self.model_item, "model", completed=completed_node_count > 0)
        self._set_item_icon(self.root, "project", completed=completed_node_count > 0)
        self.category_items["geometry"].setText(0, "Geometry")
        self.category_items["materials"].setText(0, "Materials")
        self.category_items["interactions"].setText(0, "Interactions")
        self.category_items["mesh"].setText(0, "Mesh")
        self.category_items["analysis"].setText(0, "Analysis")
        self.category_items["analysis_settings"].setText(0, "Analysis Settings")
        self.category_items["bcs"].setText(0, "Boundary Condition")
        self.category_items["solution"].setText(0, "Solution")
        self.category_items["plots"].setText(0, "Plots")

    @staticmethod
    def _entry_target_text(entry):
        if entry.get("part_id") is not None:
            return f"part {entry.get('part_id')}"
        if entry.get("ids") is not None:
            return f"{entry.get('target', 'sel')} {entry.get('ids')}"
        coords = entry.get("coords", [])
        return str(coords)

    def _on_item_clicked(self, item, column):
        _ = item
        _ = column

    def _on_selection_changed(self):
        if self._updating or self._suppress_selection_activation:
            return
        item = self.currentItem()
        if item is None:
            return
        payload = item.data(0, Qt.UserRole)
        if not isinstance(payload, dict):
            return
        self._apply_selection_payload(payload)

    def _payload_stage(self, payload):
        if not isinstance(payload, dict):
            return None
        stage = payload.get("stage")
        if stage == ProjectStage.LOADS:
            stage = ProjectStage.BCS
        return stage if isinstance(stage, ProjectStage) else None

    def _payload_is_visible(self, payload):
        stage = self._payload_stage(payload)
        return stage is None or stage in self._visible_stages

    def _payload_identity(self, payload):
        if not isinstance(payload, dict):
            return None
        kind = str(payload.get("kind", "")).lower()
        if kind == "part":
            return ("part", int(payload.get("part_id", -1)))
        if kind == "material":
            return ("material", payload.get("serial"))
        if kind == "interaction":
            return ("interaction", int(payload.get("index", -1)))
        if kind in {"bc", "load"}:
            return (kind, int(payload.get("index", -1)))
        if kind in {"mesh", "mesh_nodes", "mesh_elements"}:
            return (kind,)
        if kind.startswith("solve"):
            return ("solve", kind)
        if kind.startswith("results"):
            return ("results", kind)
        return None

    def _restore_payload_selection(self, payload):
        target = self._payload_identity(payload)
        if target is None:
            return
        for category in self.category_items.values():
            if category.isHidden():
                continue
            for item in self._iter_tree_children(category):
                if item.isHidden():
                    continue
                item_payload = item.data(0, Qt.UserRole)
                if not self._payload_is_visible(item_payload):
                    continue
                if self._payload_identity(item_payload) == target:
                    self.setCurrentItem(item)
                    return

    def _apply_selection_payload(self, payload):
        self.objectSelected.emit(payload)
        kind = payload.get("kind")
        if kind == "category":
            self._focus_category(payload.get("key"), payload.get("stage"))
        elif kind == "part":
            self._select_part(payload.get("part_id"), focus_panel=False)
        elif kind == "material":
            self._select_material(payload.get("serial"), payload.get("part_ids") or [])
        elif kind == "interaction":
            self._focus_interaction(payload.get("index"))
        elif kind in {"bc", "load"}:
            self._focus_bc_load(kind, payload.get("index"), payload.get("part_id"))
        elif kind in {"mesh_nodes", "mesh_elements"}:
            self._focus_mesh()
        elif kind.startswith("solve"):
            self._focus_solve()
        elif kind.startswith("results"):
            self._focus_results()

    def _focus_category(self, key, stage):
        key = str(key or "").lower()
        panel = self._panel()
        if panel is not None and hasattr(panel, "set_stage") and isinstance(stage, ProjectStage):
            try:
                if key == "analysis":
                    panel.set_stage(ProjectStage.INTERFACES)
                else:
                    panel.set_stage(stage)
            except Exception:
                pass
        if key == "geometry":
            view = self._view()
            if view is not None and hasattr(view, "set_module"):
                try:
                    view.set_module("Part")
                    view.set_display_mode("geometry")
                except Exception:
                    pass
        elif key == "materials":
            if panel is not None and hasattr(panel, "tabs") and hasattr(panel, "materials_tab"):
                try:
                    panel.tabs.setCurrentWidget(panel.materials_tab)
                except Exception:
                    pass
        elif key == "mesh":
            self._focus_mesh()
        elif key in {"analysis", "interactions"}:
            self._focus_interactions()
        elif key == "analysis_settings":
            panel = self._panel()
            if panel is not None and hasattr(panel, "set_stage"):
                try:
                    panel.set_stage(ProjectStage.FLUID)
                except Exception:
                    pass
        elif key == "bcs":
            self._focus_bc_load("bc", None, None)
        elif key == "solution":
            self._focus_solve()
        elif key == "plots":
            self._focus_results()

    def _select_part(self, part_id, emit_signal=True, focus_panel=True):
        view = self._view()
        if view is None:
            return
        try:
            view.set_selected_part(part_id, emit_signal=bool(emit_signal))
        except Exception:
            return
        try:
            if hasattr(view, "fit_selection"):
                view.fit_selection()
        except Exception:
            pass
        _ = focus_panel

    def _select_material(self, serial, part_ids):
        panel = self._panel()
        if panel is not None and hasattr(panel, "materials_tab"):
            try:
                panel.materials_tab.select_material(serial)
            except Exception:
                pass
        if part_ids:
            self._select_part(part_ids[0], emit_signal=False, focus_panel=False)

    def _focus_bc_load(self, kind, index, part_id):
        panel = self._panel()
        if panel is None or not hasattr(panel, "tabs"):
            return
        target_panel = getattr(panel, "bcs_tab" if kind == "bc" else "loads_tab", None)
        target_list = getattr(target_panel, "bc_list" if kind == "bc" else "load_list", None)
        if target_panel is not None:
            try:
                panel.tabs.setCurrentWidget(target_panel)
            except Exception:
                pass
        if index is not None and target_list is not None and hasattr(target_panel, "_index_role"):
            for row in range(target_list.topLevelItemCount()):
                item = target_list.topLevelItem(row)
                if item.data(0, target_panel._index_role) == index:
                    target_list.setCurrentItem(item)
                    entry = item.data(0, Qt.UserRole)
                    if hasattr(view := self._view(), "set_panel_attr_focus"):
                        try:
                            view.set_panel_attr_focus(kind, entry)
                        except Exception:
                            pass
                    break
        if part_id is not None:
            self._select_part(part_id, emit_signal=False, focus_panel=False)

    def _focus_mesh(self):
        panel = self._panel()
        if panel is not None and hasattr(panel, "tabs") and hasattr(panel, "mesh_tab"):
            try:
                panel.tabs.setCurrentWidget(panel.mesh_tab)
            except Exception:
                pass
            mesh_tab = getattr(panel, "mesh_tab", None)
            if mesh_tab is not None and hasattr(mesh_tab, "mesh_view_toggle"):
                try:
                    mesh_tab.mesh_view_toggle.blockSignals(True)
                    mesh_tab.mesh_view_toggle.setChecked(True)
                    mesh_tab.mesh_view_toggle.blockSignals(False)
                    if hasattr(mesh_tab, "show_nodes"):
                        mesh_tab.show_nodes.blockSignals(True)
                        mesh_tab.show_nodes.setChecked(True)
                        mesh_tab.show_nodes.blockSignals(False)
                    if hasattr(mesh_tab, "show_mesh"):
                        mesh_tab.show_mesh.blockSignals(True)
                        mesh_tab.show_mesh.setChecked(False)
                        mesh_tab.show_mesh.blockSignals(False)
                    if hasattr(mesh_tab, "_update_display"):
                        mesh_tab._update_display()
                except Exception:
                    pass

    def _focus_results(self):
        panel = self._panel()
        if panel is not None and hasattr(panel, "tabs") and hasattr(panel, "results_tab"):
            try:
                panel.tabs.setCurrentWidget(panel.results_tab)
            except Exception:
                pass

    def _focus_solve(self):
        panel = self._panel()
        if panel is not None and hasattr(panel, "tabs") and hasattr(panel, "job_tab"):
            try:
                panel.tabs.setCurrentWidget(panel.job_tab)
            except Exception:
                pass

    def _focus_interactions(self):
        panel = self._panel()
        if panel is not None and hasattr(panel, "tabs") and hasattr(panel, "interfaces_tab"):
            try:
                panel.tabs.setCurrentWidget(panel.interfaces_tab)
            except Exception:
                pass

    def _focus_interaction(self, index):
        self._focus_interactions()
        panel = self._panel()
        interfaces_tab = getattr(panel, "interfaces_tab", None) if panel is not None else None
        if interfaces_tab is None or not hasattr(interfaces_tab, "interface_list"):
            return
        iface = self._resolve_interface(index)
        if iface is None:
            return
        iface_id = self._iface_value(iface, "id", None)
        tree = interfaces_tab.interface_list
        for row in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(row)
            if item is None:
                continue
            try:
                if int(item.data(0, Qt.UserRole)) == int(iface_id):
                    tree.setCurrentItem(item)
                    break
            except Exception:
                continue
        try:
            part_id = int(self._iface_value(iface, "part1_id", -1))
        except Exception:
            part_id = None
        if part_id is not None and part_id >= 0:
            self._select_part(part_id, emit_signal=False, focus_panel=False)

    def _resolve_part(self, part_id):
        view = self._view()
        if view is None or part_id is None:
            return None
        for part in getattr(view, "parts", []) or []:
            try:
                if int(getattr(part, "id", -1)) == int(part_id):
                    return part
            except Exception:
                continue
        return None

    def _resolve_material(self, serial):
        state = self._state()
        materials = getattr(state, "materials", {}) if state is not None else {}
        if not isinstance(materials, dict):
            return None
        return materials.get(serial)

    def _resolve_interface(self, index):
        state = self._state()
        interfaces = list(getattr(state, "interfaces", []) or [])
        if index is None:
            return None
        try:
            index = int(index)
        except Exception:
            return None
        if 0 <= index < len(interfaces):
            return interfaces[index]
        return None

    def _show_context_menu(self, pos, *, from_global=False):
        viewport_pos = pos if not from_global else self.viewport().mapFromGlobal(pos)
        item = self.itemAt(viewport_pos)
        if item is None:
            return
        self._suppress_selection_activation = True
        self._updating = True
        try:
            self.setCurrentItem(item)
        finally:
            self._updating = False
        payload = item.data(0, Qt.UserRole)
        if not isinstance(payload, dict):
            self._suppress_selection_activation = False
            return
        menu = QMenu(self)
        kind = payload.get("kind")
        if kind == "part":
            self._populate_part_context_menu(menu, payload)
        elif kind == "material":
            self._populate_material_context_menu(menu, payload)
        elif kind == "interaction":
            self._populate_interaction_context_menu(menu, payload)
        elif kind in {"bc", "load"}:
            self._populate_attr_context_menu(menu, payload)
        elif kind == "category":
            self._populate_category_context_menu(menu, payload, item)
        else:
            menu.addAction("Properties", lambda p=payload: self._apply_selection_payload(p))
        global_pos = self.viewport().mapToGlobal(viewport_pos)
        chosen = menu.exec(global_pos)
        _ = chosen
        menu.close()
        self._suppress_selection_activation = False

    def contextMenuEvent(self, event):
        self._show_context_menu(event.globalPos(), from_global=True)
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self._suppress_selection_activation = True
            item = self.itemAt(event.pos())
            if item is not None:
                self._updating = True
                try:
                    self.setCurrentItem(item)
                finally:
                    self._updating = False
                self._show_context_menu(event.pos(), from_global=False)
                self._suppress_selection_activation = False
            event.accept()
            return
        self._suppress_selection_activation = False
        super().mousePressEvent(event)

    def _populate_category_context_menu(self, menu, payload, item):
        key = str(payload.get("key", "")).lower()
        menu.addAction("Open", lambda p=payload: self._apply_selection_payload(p))
        if item.childCount() > 0:
            if item.isExpanded():
                menu.addAction("Collapse", lambda i=item: i.setExpanded(False))
            else:
                menu.addAction("Expand", lambda i=item: i.setExpanded(True))
        if key == "geometry":
            first_part = next(
                (
                    item.child(row).data(0, Qt.UserRole)
                    for row in range(item.childCount())
                    if isinstance(item.child(row).data(0, Qt.UserRole), dict)
                    and item.child(row).data(0, Qt.UserRole).get("kind") == "part"
                ),
                None,
            )
            if first_part is not None:
                menu.addAction("Edit First Part", lambda p=first_part: self._edit_payload(p))
        elif key == "materials":
            menu.addAction(
                "New Material",
                lambda: self._focus_category("materials", ProjectStage.MATERIALS),
            )

    def _populate_part_context_menu(self, menu, payload):
        part = self._resolve_part(payload.get("part_id"))
        view = self._view()
        if part is None or view is None:
            menu.addAction("Properties", lambda p=payload: self._apply_selection_payload(p))
            return
        edit_label = "Edit Sketch" if getattr(part, "sketches", None) else "Edit Geometry"
        menu.addAction("Select", lambda p=payload: self._apply_selection_payload(p))
        if not getattr(part, "is_void", False) and hasattr(view, "_populate_material_submenu") and hasattr(view, "can_assign_material"):
            assign_menu = menu.addMenu("Assign Material")
            try:
                view._populate_material_submenu(assign_menu, part)
            except Exception:
                pass
        bc_menu = menu.addMenu("Apply Boundary Condition")
        bc_menu.addAction("Fix UX & UY", lambda p=part: view._apply_fixed_bc_for_part("fix_xy", p.id))
        bc_menu.addAction("Fix UX", lambda p=part: view._apply_fixed_bc_for_part("fix_x", p.id))
        bc_menu.addAction("Fix UY", lambda p=part: view._apply_fixed_bc_for_part("fix_y", p.id))
        bc_menu.addSeparator()
        bc_menu.addAction("Velocity X", lambda p=part: view._apply_time_profile_for_part("velocity", "x", p.id))
        bc_menu.addAction("Velocity Y", lambda p=part: view._apply_time_profile_for_part("velocity", "y", p.id))
        bc_menu.addSeparator()
        bc_menu.addAction("Force X", lambda p=part: view._apply_time_profile_for_part("force", "x", p.id))
        bc_menu.addAction("Force Y", lambda p=part: view._apply_time_profile_for_part("force", "y", p.id))
        if getattr(view, "project_mode", "2d") == "3d":
            bc_menu.addAction("Fix UZ", lambda p=part: view._apply_fixed_bc_for_part("fix_z", p.id))
            bc_menu.addAction("Velocity Z", lambda p=part: view._apply_time_profile_for_part("velocity", "z", p.id))
            bc_menu.addAction("Force Z", lambda p=part: view._apply_time_profile_for_part("force", "z", p.id))
        menu.addAction("Create Interaction", lambda p=part: self._create_interaction_for_part(p))
        menu.addAction(edit_label, lambda p=payload: self._edit_payload(p))
        menu.addAction("Delete", lambda p=payload: self._delete_payload(p))

    def _populate_material_context_menu(self, menu, payload):
        menu.addAction("Edit", lambda p=payload: self._edit_payload(p))
        menu.addAction("Duplicate", lambda p=payload: self._duplicate_material(p.get("serial")))
        menu.addAction("Delete", lambda p=payload: self._delete_payload(p))

    def _populate_interaction_context_menu(self, menu, payload):
        menu.addAction("Edit", lambda p=payload: self._edit_payload(p))
        menu.addAction("Highlight", lambda p=payload: self._highlight_payload(p))
        menu.addAction("Delete", lambda p=payload: self._delete_payload(p))

    def _populate_attr_context_menu(self, menu, payload):
        kind = payload.get("kind")
        menu.addAction("Edit", lambda p=payload: self._edit_payload(p))
        menu.addAction("Rename", lambda p=payload: self._rename_payload(p))
        menu.addAction("Highlight", lambda p=payload: self._highlight_payload(p))
        menu.addAction("Delete", lambda p=payload: self._delete_payload(p))
        if kind == "load":
            menu.addAction("Properties", lambda p=payload: self._apply_selection_payload(p))

    def _rename_payload(self, payload):
        kind = payload.get("kind")
        panel = self._panel()
        if kind == "bc" and panel is not None and hasattr(panel, "bcs_tab"):
            self._focus_bc_load("bc", payload.get("index"), payload.get("part_id"))
            if hasattr(panel.bcs_tab, "_rename_selected_entry"):
                panel.bcs_tab._rename_selected_entry()
                self.refresh_from_model()
            return
        if kind == "load" and panel is not None and hasattr(panel, "loads_tab"):
            self._focus_bc_load("load", payload.get("index"), payload.get("part_id"))
            if hasattr(panel.loads_tab, "_rename_selected_entry"):
                panel.loads_tab._rename_selected_entry()
                self.refresh_from_model()
            return

    def _edit_payload(self, payload):
        kind = payload.get("kind")
        if kind == "part":
            view = self._view()
            part_id = payload.get("part_id")
            part = next((p for p in getattr(view, "parts", []) if int(getattr(p, "id", -1)) == int(part_id)), None)
            if part is None:
                return
            self._select_part(part_id)
            try:
                view.set_module("Part")
            except Exception:
                pass
            if hasattr(view, "begin_part_shape_edit"):
                try:
                    view.begin_part_shape_edit(part)
                except Exception:
                    pass
            return
        if kind == "material":
            self._select_material(payload.get("serial"), payload.get("part_ids") or [])
            panel = self._panel()
            if panel is not None and hasattr(panel, "materials_tab") and hasattr(panel.materials_tab, "focus_name_input"):
                try:
                    panel.materials_tab.focus_name_input()
                except Exception:
                    pass
            return
        panel = self._panel()
        if kind == "bc" and panel is not None and hasattr(panel, "bcs_tab"):
            self._focus_bc_load("bc", payload.get("index"), payload.get("part_id"))
            if hasattr(panel.bcs_tab, "_edit_selected_bc_only"):
                panel.bcs_tab._edit_selected_bc_only()
            return
        if kind == "load" and panel is not None and hasattr(panel, "loads_tab"):
            self._focus_bc_load("load", payload.get("index"), payload.get("part_id"))
            if hasattr(panel.loads_tab, "_edit_selected_load_only"):
                panel.loads_tab._edit_selected_load_only()
            return
        if kind == "interaction":
            panel = self._panel()
            iface = self._resolve_interface(payload.get("index"))
            if iface is None or panel is None or not hasattr(panel, "interfaces_tab"):
                return
            self._focus_interaction(payload.get("index"))
            try:
                panel.interfaces_tab.define_interface(iface_to_edit=iface)
            except Exception:
                pass

    def _delete_payload(self, payload):
        kind = payload.get("kind")
        if kind == "part":
            view = self._view()
            part_id = payload.get("part_id")
            part = next((p for p in getattr(view, "parts", []) if int(getattr(p, "id", -1)) == int(part_id)), None)
            if part is None:
                return
            if hasattr(view, "delete_part") and view.delete_part(part, confirm=True):
                self.refresh_from_model()
            return
        if kind == "material":
            self._delete_material(payload.get("serial"))
            return
        panel = self._panel()
        if kind == "bc" and panel is not None and hasattr(panel, "bcs_tab"):
            self._focus_bc_load("bc", payload.get("index"), payload.get("part_id"))
            if hasattr(panel.bcs_tab, "_delete_selected_bc_only"):
                panel.bcs_tab._delete_selected_bc_only()
                self.refresh_from_model()
            return
        if kind == "load" and panel is not None and hasattr(panel, "loads_tab"):
            self._focus_bc_load("load", payload.get("index"), payload.get("part_id"))
            if hasattr(panel.loads_tab, "_delete_selected_load_only"):
                panel.loads_tab._delete_selected_load_only()
                self.refresh_from_model()
            return
        if kind == "interaction":
            panel = self._panel()
            iface = self._resolve_interface(payload.get("index"))
            if iface is None or panel is None or not hasattr(panel, "interfaces_tab"):
                return
            self._focus_interaction(payload.get("index"))
            try:
                panel.interfaces_tab.delete_selected_interface()
                self.refresh_from_model()
            except Exception:
                pass

    def _delete_material(self, serial):
        material = self._resolve_material(serial)
        state = self._state()
        view = self._view()
        if material is None or state is None or view is None:
            return
        reply = QMessageBox.question(
            self,
            "Delete Material",
            f"Delete material '{getattr(material, 'name', serial)}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        try:
            view.push_undo_state()
        except Exception:
            pass
        state.materials.pop(serial, None)
        for part in getattr(state, "parts", []) or []:
            if getattr(part, "material_id", None) == serial:
                part.material_id = None
        try:
            view.materialsChanged.emit()
            view.partsChanged.emit()
            view.redraw()
        except Exception:
            pass
        self.refresh_from_model()

    def _duplicate_material(self, serial):
        material = self._resolve_material(serial)
        state = self._state()
        view = self._view()
        if material is None or state is None or view is None:
            return
        try:
            from copy import deepcopy
            from models import Material
        except Exception:
            return
        try:
            view.push_undo_state()
        except Exception:
            pass
        clone = Material(
            f"{getattr(material, 'name', 'Material')} Copy",
            getattr(material, "mat_type", ""),
            deepcopy(getattr(material, "properties", {}) or {}),
            symmetry=getattr(material, "symmetry", "isotropic"),
            behavior=getattr(material, "behavior", "elastic"),
            damage=getattr(material, "damage", "none"),
        )
        state.materials[clone.serial] = clone
        try:
            view.materialsChanged.emit()
            view.redraw()
        except Exception:
            pass
        self.refresh_from_model()
        self._apply_selection_payload({"kind": "material", "serial": clone.serial, "stage": ProjectStage.MATERIALS})

    def _highlight_payload(self, payload):
        kind = payload.get("kind")
        if kind in {"bc", "load"}:
            self._focus_bc_load(kind, payload.get("index"), payload.get("part_id"))
            return
        self._apply_selection_payload(payload)

    def _suggest_interaction_pair(self, clicked_part_id):
        view = self._view()
        solid_parts = [
            p for p in getattr(view, "parts", []) or []
            if not getattr(p, "is_void", False)
        ] if view is not None else []
        try:
            clicked_part_id = int(clicked_part_id)
        except Exception:
            return None
        selected_part_id = getattr(view, "selected_part_id", None) if view is not None else None
        try:
            if selected_part_id is not None and int(selected_part_id) != clicked_part_id:
                return (int(selected_part_id), clicked_part_id)
        except Exception:
            pass
        for part in solid_parts:
            try:
                pid = int(getattr(part, "id", -1))
            except Exception:
                continue
            if pid != clicked_part_id:
                return (clicked_part_id, pid)
        return (clicked_part_id, None)

    def _create_interaction_for_part(self, part):
        panel = self._panel()
        if panel is None or not hasattr(panel, "interfaces_tab") or part is None:
            return
        pair = self._suggest_interaction_pair(getattr(part, "id", None))
        self._focus_interactions()
        try:
            panel.interfaces_tab.define_interface(preset_part_ids=pair)
        except TypeError:
            panel.interfaces_tab.define_interface()

    def set_active_stage(self, active_stage):
        current_item = self.currentItem()
        current_payload = current_item.data(0, Qt.UserRole) if current_item is not None else None
        current_is_object = (
            isinstance(current_payload, dict)
            and current_payload.get("kind") not in {None, "category"}
            and self._payload_is_visible(current_payload)
        )
        for key, item in self.category_items.items():
            font = item.font(0)
            font.setBold(active_stage in self._category_active_stages.get(key, set()))
            item.setFont(0, font)
            item.setHidden(False)
            item.setDisabled(False)
        active_item = self.stage_items.get(active_stage)
        if (
            active_item is not None
            and not active_item.isHidden()
            and not current_is_object
        ):
            self._updating = True
            try:
                self.setCurrentItem(active_item)
            finally:
                self._updating = False

    def add_session_project(self, name):
        if not name:
            return None
        self.session_root.setHidden(False)
        item = QTreeWidgetItem([str(name)])
        item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
        item.setToolTip(0, str(name))
        self.session_root.addChild(item)
        self.session_root.setExpanded(True)
        return item

    def select_part(self, part_id, fit_view=True):
        if part_id is None:
            return
        parent = self.category_items.get("geometry")
        if parent is None:
            return
        for item in self._iter_tree_children(parent):
            payload = item.data(0, Qt.UserRole)
            if isinstance(payload, dict) and int(payload.get("part_id", -1)) == int(part_id):
                self._updating = True
                try:
                    self.setCurrentItem(item)
                finally:
                    self._updating = False
                # When the canvas drives the selection we just sync the tree
                # highlight — re-running the payload would re-trigger
                # fit_selection and zoom the view to the clicked part, which
                # the user doesn't want when they're just selecting it.
                if fit_view:
                    self._apply_selection_payload(payload)
                else:
                    try:
                        self.objectSelected.emit(payload)
                    except Exception:
                        pass
                return
