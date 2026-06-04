import math
import random

import numpy as np

from PySide6.QtCore import Qt, Signal, QRect, QRectF, QTimer, QPoint, QPointF
from PySide6.QtGui import QVector3D, QVector4D, QFont, QColor, QPainter, QPen, QBrush, QPolygonF
from PySide6.QtWidgets import QMenu, QToolTip, QRubberBand
from pyqtgraph.opengl import (
    GLGridItem,
    GLMeshItem,
    GLScatterPlotItem,
    GLTextItem,
    GLViewWidget,
    GLAxisItem,
    GLLinePlotItem,
)


def _tets_to_triangles(tets):
    faces = []
    counts = {}
    for tet in tets:
        a, b, c, d = [int(i) for i in tet]
        for tri in ((a, b, c), (a, b, d), (a, c, d), (b, c, d)):
            key = tuple(sorted(tri))
            counts[key] = counts.get(key, 0) + 1
            faces.append(tri)
    surface = [tri for tri in faces if counts[tuple(sorted(tri))] == 1]
    return np.array(surface, dtype=int) if surface else np.zeros((0, 3), dtype=int)


def _sanitize_triangle_faces(nodes, faces, face_group_ids=None):
    """Drop invalid/degenerate triangles before constructing GLMeshItem."""
    try:
        nodes_arr = np.asarray(nodes, dtype=float)
    except Exception:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int), None
    if nodes_arr.ndim != 2 or nodes_arr.shape[0] == 0 or nodes_arr.shape[1] < 3:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int), None
    nodes_xyz = np.asarray(nodes_arr[:, :3], dtype=float)

    try:
        faces_arr = np.asarray(faces, dtype=int)
    except Exception:
        return nodes_xyz, np.zeros((0, 3), dtype=int), None
    if faces_arr.ndim != 2 or faces_arr.shape[0] == 0 or faces_arr.shape[1] != 3:
        return nodes_xyz, np.zeros((0, 3), dtype=int), None

    groups_arr = None
    if face_group_ids is not None:
        try:
            groups = np.asarray(face_group_ids, dtype=int).reshape(-1)
            if len(groups) == len(faces_arr):
                groups_arr = groups
        except Exception:
            groups_arr = None

    keep = np.ones(len(faces_arr), dtype=bool)
    n_nodes = int(len(nodes_xyz))
    keep &= np.all(faces_arr >= 0, axis=1)
    keep &= np.all(faces_arr < n_nodes, axis=1)
    keep &= faces_arr[:, 0] != faces_arr[:, 1]
    keep &= faces_arr[:, 1] != faces_arr[:, 2]
    keep &= faces_arr[:, 0] != faces_arr[:, 2]

    finite_nodes = np.isfinite(nodes_xyz).all(axis=1)
    rows = np.where(keep)[0]
    if rows.size > 0:
        tris = faces_arr[rows]
        keep[rows] &= finite_nodes[tris].all(axis=1)

    rows = np.where(keep)[0]
    if rows.size > 0:
        tris = faces_arr[rows]
        a = nodes_xyz[tris[:, 0]]
        b = nodes_xyz[tris[:, 1]]
        c = nodes_xyz[tris[:, 2]]
        finite_pts = nodes_xyz[finite_nodes]
        if finite_pts.size == 0:
            scale = 1.0
        else:
            span = np.ptp(finite_pts, axis=0)
            scale = max(float(np.linalg.norm(span)), 1.0)
        min_cross_norm = (scale * scale) * 1e-12
        cross_norm = np.linalg.norm(np.cross(b - a, c - a), axis=1)
        keep[rows] &= np.isfinite(cross_norm) & (cross_norm > min_cross_norm)

    faces_out = faces_arr[keep]
    groups_out = groups_arr[keep] if groups_arr is not None else None
    return nodes_xyz, faces_out, groups_out


def _infer_face_groups_from_quads(nodes_xyz, faces_arr):
    try:
        nodes = np.asarray(nodes_xyz, dtype=float)
        faces = np.asarray(faces_arr, dtype=int)
    except Exception:
        return None
    if nodes.ndim != 2 or faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        return None
    groups = np.arange(len(faces), dtype=int)
    edge_map = {}
    for fi, tri in enumerate(faces):
        a, b, c = [int(v) for v in tri]
        for edge in ((a, b), (b, c), (c, a)):
            key = (min(edge[0], edge[1]), max(edge[0], edge[1]))
            edge_map.setdefault(key, []).append(fi)

    def _face_normal(idx):
        tri = faces[idx]
        p0 = nodes[int(tri[0])]
        p1 = nodes[int(tri[1])]
        p2 = nodes[int(tri[2])]
        n = np.cross(p1 - p0, p2 - p0)
        ln = float(np.linalg.norm(n))
        if ln <= 1e-12 or not np.isfinite(ln):
            return None
        return n / ln

    parent = list(range(len(faces)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for shared_edge, face_ids in edge_map.items():
        if len(face_ids) != 2:
            continue
        f0, f1 = int(face_ids[0]), int(face_ids[1])
        tri0 = set(int(v) for v in faces[f0])
        tri1 = set(int(v) for v in faces[f1])
        merged = tri0.union(tri1)
        if len(merged) != 4:
            continue
        n0 = _face_normal(f0)
        n1 = _face_normal(f1)
        if n0 is None or n1 is None:
            continue
        if float(np.dot(n0, n1)) < 0.999:
            continue
        idxs = sorted(int(v) for v in merged)
        pts = nodes[np.asarray(idxs, dtype=int)]
        centroid = np.mean(pts, axis=0)
        # Two triangles from one quad are near-coplanar.
        plane_err = np.abs(np.dot(pts - centroid, n0))
        if float(np.max(plane_err)) > 1e-6:
            continue
        union(f0, f1)

    root_to_gid = {}
    next_gid = 0
    out = np.zeros(len(faces), dtype=int)
    for i in range(len(faces)):
        root = find(i)
        if root not in root_to_gid:
            root_to_gid[root] = next_gid
            next_gid += 1
        out[i] = root_to_gid[root]
    return out


class Mesh3DView(GLViewWidget):
    gizmoMoved = Signal(float, float, float)
    gizmoRotated = Signal(float, float, float)
    gizmoScaled = Signal(float, float, float)
    gizmoDragStarted = Signal(str)
    gizmoDragFinished = Signal(str)
    selectionChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setToolTip(
            "3D controls: left-drag select, middle pan, right rotate, wheel zoom.\n"
            "Trackpad: two-finger scroll pan, Ctrl+scroll zoom, Alt+drag rotate."
        )
        self._bg_color = (242, 246, 251)
        self.setBackgroundColor(*self._bg_color)
        self.opts["distance"] = 200
        self.opts["elevation"] = 25
        self.opts["azimuth"] = 35
        self._base_distance = 200
        self._center = QVector3D(0, 0, 0)
        self._axis_length = 30.0
        self.grid_item = GLGridItem()
        self.grid_item.setSize(400, 400)
        self.grid_item.setSpacing(20, 20)
        self.grid_item.setColor((0.48, 0.54, 0.62, 0.42))
        self.grid_item.setDepthValue(-10)
        self.addItem(self.grid_item)
        self.axis_item = GLAxisItem()
        self.axis_item.setSize(self._axis_length, self._axis_length, self._axis_length)
        self.axis_item.setDepthValue(2)
        self.addItem(self.axis_item)
        self._grid_extent = 400.0
        self.plane_axes_item = GLLinePlotItem(mode="lines", antialias=True)
        self.plane_axes_item.setDepthValue(1)
        self.addItem(self.plane_axes_item)
        self.axis_labels = {
            "x": GLTextItem(text="X", color=(224, 62, 48, 255)),
            "y": GLTextItem(text="Y", color=(36, 148, 66, 255)),
            "z": GLTextItem(text="Z", color=(34, 94, 235, 255)),
        }
        for label in self.axis_labels.values():
            label.setData(font=QFont("Sans Serif", 10))
            label.setDepthValue(3)
            self.addItem(label)
        self.mesh_item = None
        self.mesh_items = []
        self._selection_overlay_item = None
        self.node_item = None
        self.node_item_surface = None
        self.node_item_interior = None
        self.node_item_selected = None
        self.show_mesh = True
        # Mesh-node dots default off — toggle from the Mesh panel if needed.
        self.show_nodes = False
        self.show_surface_nodes = True
        self.show_interior_nodes = True
        self._mesh_dim_enabled = False
        self._mesh_xray_enabled = False
        self._wireframe_enabled = False
        self._wireframe_item = None
        self._last_node_size = None
        self._surface_node_ids = np.array([], dtype=int)
        self._interior_node_ids = np.array([], dtype=int)
        self._grid_spacing = 20.0
        self._snap_enabled = True
        self._gizmo_enabled = False
        self._gizmo_pos = QVector3D(0, 0, 0)
        self._gizmo_items = []
        self._dragging_gizmo = False
        self._drag_plane_z = 0.0
        self._placement_active = False
        self._placement_callback = None
        self._placement_z = 0.0
        self._gizmo_mode = "translate"
        self._gizmo_length = 20.0
        self._active_axis = None
        self._drag_start_axis_param = None
        self._drag_start_pos = None
        self._drag_last_vec = None
        self._drag_mode = None
        self._material_style = "metal"
        self._mesh_color_override = None
        self._mesh_edge_color_override = None
        self._last_mesh = None
        self._last_meshes = None
        self._last_faces = None
        self._last_edges = None
        self._face_group_ids = None
        self._node_size_override = None
        self._node_color_override = None
        self._node_color_surface_override = None
        self._node_color_interior_override = None
        self._node_color_selected_override = None
        self._node_colors_auto = True
        self._contrast_palette = {}
        self._contrast_nodes = {}
        self._show_all_nodes = False
        self._hover_nodes = None
        self._hover_ids = None
        self._hover_enabled = True
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.timeout.connect(self._handle_hover_timer)
        self._view_anim_timer = QTimer(self)
        self._view_anim_timer.setInterval(16)
        self._view_anim_timer.timeout.connect(self._step_view_animation)
        self._view_anim = None
        self._context_menu_hook = None
        self._hover_pos = None
        self._selected_node_id = None
        self._selected_node_pos = None
        self._selection_mode = "none"
        self._orientation_overlay_enabled = True
        self._orientation_overlay_hover_key = None
        self._orientation_overlay_hitboxes = {}
        self._view_navigation_enabled = False
        self._suppress_next_context_menu = False
        self._nav_panning = False
        self._nav_rotating = False
        self._nav_last_pos = None
        self._nav_right_dragged = False
        self._selection_box = QRubberBand(QRubberBand.Rectangle, self)
        self._selection_box.setStyleSheet(
            "QRubberBand {"
            "border: 1px solid rgba(29, 78, 216, 0.95);"
            "background: rgba(96, 165, 250, 0.16);"
            "}"
        )
        self._selection_box.hide()
        self._selection_box_start = None
        self._selection_box_active = False
        self._selection_box_dragged = False
        self.selected_faces = set()
        self._selection_highlight = None
        self.selected_nodes = set()
        self.selected_topology_vertices = set()
        self._selected_nodes_item = None
        self._point_pick_threshold = None
        self.selected_edges = set()
        self.selected_topology_edges = set()
        self._selected_edges_item = None
        self._hover_face_indices = set()
        self._hover_face_item = None
        self._hover_edge_pick = None
        self._hover_edge_item = None
        self._hover_point_pick = None
        self._hover_point_item = None
        self._cad_topology = {"vertices": [], "edges": []}
        self._cad_topology_vertices = np.zeros((0, 3), dtype=float)
        self._cad_topology_vertex_ids = np.zeros((0,), dtype=int)
        self._cad_topology_vertex_lookup = {}
        self._cad_topology_edges = []
        self._cad_topology_edge_lookup = {}
        self._cad_topology_vertex_node_ids = {}
        self._cad_topology_edge_node_ids = {}
        self._bc_load_marker_items = []
        self._apply_contrast_theme()
        self._update_axis_labels()

    def clear_mesh(self):
        if self.mesh_item is not None:
            self.removeItem(self.mesh_item)
            self.mesh_item = None
        if self.mesh_items:
            for item in self.mesh_items:
                self.removeItem(item)
            self.mesh_items = []
        self.clear_selection_overlay()
        if self.node_item is not None:
            self.removeItem(self.node_item)
            self.node_item = None
        if self.node_item_surface is not None:
            self.removeItem(self.node_item_surface)
            self.node_item_surface = None
        if self.node_item_interior is not None:
            self.removeItem(self.node_item_interior)
            self.node_item_interior = None
        if self.node_item_selected is not None:
            self.removeItem(self.node_item_selected)
            self.node_item_selected = None
        if self._wireframe_item is not None:
            self.removeItem(self._wireframe_item)
            self._wireframe_item = None
        self._surface_node_ids = np.array([], dtype=int)
        self._interior_node_ids = np.array([], dtype=int)
        self._selected_node_id = None
        self._selected_node_pos = None
        self._last_mesh = None
        self._last_meshes = None
        self._last_faces = None
        self._last_edges = None
        self._face_group_ids = None
        self._cad_topology = {"vertices": [], "edges": []}
        self._cad_topology_vertices = np.zeros((0, 3), dtype=float)
        self._cad_topology_vertex_ids = np.zeros((0,), dtype=int)
        self._cad_topology_vertex_lookup = {}
        self._cad_topology_edges = []
        self._cad_topology_edge_lookup = {}
        self._cad_topology_vertex_node_ids = {}
        self._cad_topology_edge_node_ids = {}
        self.clear_bc_load_markers()
        self._selection_box.hide()
        self._selection_box_active = False
        self._selection_box_start = None
        self._selection_box_dragged = False
        self._clear_pick_hover()
        self.clear_selection()
        self.clear_node_selection()
        self.clear_edge_selection()

    def set_visibility(self, show_nodes=None, show_mesh=None):
        if show_nodes is not None:
            self.set_nodes_visible(show_nodes=show_nodes)
        if show_mesh is not None:
            self.show_mesh = bool(show_mesh)
            if self.mesh_item is not None:
                self.mesh_item.setVisible(self.show_mesh)
            for item in self.mesh_items:
                item.setVisible(self.show_mesh)
            if self._selection_overlay_item is not None:
                self._selection_overlay_item.setVisible(self.show_mesh)
            if self._wireframe_item is not None:
                self._wireframe_item.setVisible(self.show_mesh and self._wireframe_enabled)

    def clear_selection_overlay(self):
        if self._selection_overlay_item is not None:
            try:
                self.removeItem(self._selection_overlay_item)
            except Exception:
                pass
            self._selection_overlay_item = None

    def set_selection_overlay(
        self,
        nodes,
        faces,
        color=None,
        edge_color=None,
        draw_faces=True,
        draw_edges=True,
    ):
        self.clear_selection_overlay()
        if nodes is None or faces is None:
            return
        try:
            nodes_arr = np.asarray(nodes, dtype=float)
            faces_arr = np.asarray(faces, dtype=int)
        except Exception:
            return
        if nodes_arr.ndim != 2 or nodes_arr.shape[0] == 0 or nodes_arr.shape[1] < 3:
            return
        if faces_arr.ndim != 2 or faces_arr.shape[0] == 0:
            return
        if faces_arr.shape[1] == 4:
            faces_arr = _tets_to_triangles(faces_arr)
        nodes_arr, faces_arr, _ = _sanitize_triangle_faces(nodes_arr, faces_arr)
        if faces_arr.ndim != 2 or faces_arr.shape[0] == 0 or faces_arr.shape[1] != 3:
            return
        overlay_color = color or (1.0, 0.65, 0.1, 0.14)
        overlay_edge = edge_color or (1.0, 0.45, 0.05, 0.95)
        item = GLMeshItem(
            vertexes=nodes_arr,
            faces=faces_arr,
            smooth=False,
            drawEdges=bool(draw_edges),
            drawFaces=bool(draw_faces),
            shader="shaded",
            edgeColor=overlay_edge,
            color=overlay_color,
        )
        try:
            item.setGLOptions("translucent")
        except Exception:
            pass
        item.setVisible(self.show_mesh)
        self._selection_overlay_item = item
        self.addItem(item)

    def set_nodes_visible(self, show_nodes=None, show_surface=None, show_interior=None):
        if show_nodes is not None:
            self.show_nodes = bool(show_nodes)
        if show_surface is not None:
            self.show_surface_nodes = bool(show_surface)
        if show_interior is not None:
            self.show_interior_nodes = bool(show_interior)
        if self.node_item is not None:
            self.node_item.setVisible(self.show_nodes)
        if self.node_item_surface is not None:
            self.node_item_surface.setVisible(self.show_nodes and self.show_surface_nodes)
        if self.node_item_interior is not None:
            self.node_item_interior.setVisible(self.show_nodes and self.show_interior_nodes)
        if self.node_item_selected is not None:
            self.node_item_selected.setVisible(self.show_nodes)
        if self._selected_nodes_item is not None:
            self._selected_nodes_item.setVisible(self.show_nodes)
        if self._selected_edges_item is not None:
            self._selected_edges_item.setVisible(self.show_mesh)
        for item in self._bc_load_marker_items:
            try:
                item.setVisible(self.show_mesh)
            except Exception:
                pass

    def set_mesh_xray(self, enabled):
        self._mesh_xray_enabled = bool(enabled)
        self._refresh_mesh_style()

    def set_mesh_dim(self, enabled):
        self._mesh_dim_enabled = bool(enabled)
        self._refresh_mesh_style()
        self._update_wireframe_style()

    def set_wireframe_visible(self, enabled):
        self._wireframe_enabled = bool(enabled)
        self._update_wireframe_overlay()

    def set_node_display(self, size=None, color=None):
        if size is not None:
            self._node_size_override = float(size)
        if color is not None:
            self._node_color_override = color
            self._node_color_surface_override = color
            self._node_color_interior_override = color
        for item in (self.node_item, self.node_item_surface, self.node_item_interior, self.node_item_selected):
            if item is None:
                continue
            if self._node_size_override is not None:
                item.setData(size=self._node_size_override if item is not self.node_item_selected else self._node_size_override * 1.6)
            if self._node_color_override is not None and item is self.node_item:
                item.setData(color=self._node_color_override)
        self._update_node_colors()

    def set_node_colors(self, surface=None, interior=None, selected=None):
        self._node_colors_auto = False
        if surface is not None:
            self._node_color_surface_override = surface
        if interior is not None:
            self._node_color_interior_override = interior
        if selected is not None:
            self._node_color_selected_override = selected
        self._update_node_colors()

    def set_scalar_node_colors(self, scalars, cmap_name="viridis"):
        if self._last_mesh is None:
            return
        nodes, _ = self._last_mesh
        nodes = np.asarray(nodes, dtype=float)
        if nodes.ndim != 2 or nodes.shape[0] == 0:
            return
        if scalars is None:
            self.set_node_colors_auto()
            if self._last_mesh is not None:
                mesh_nodes, mesh_faces = self._last_mesh
                groups = self._face_group_ids.copy() if isinstance(self._face_group_ids, np.ndarray) else self._face_group_ids
                self.set_mesh(mesh_nodes, mesh_faces, face_group_ids=groups)
            return
        arr = np.asarray(scalars, dtype=float).reshape(-1)
        if arr.size == 0:
            return
        if arr.size != len(nodes):
            m = min(int(arr.size), int(len(nodes)))
            tmp = np.zeros(len(nodes), dtype=float)
            tmp[:m] = arr[:m]
            arr = tmp
        finite = np.isfinite(arr)
        if not np.any(finite):
            return
        vmin = float(np.min(arr[finite]))
        vmax = float(np.max(arr[finite]))
        if abs(vmax - vmin) < 1e-12:
            norm = np.zeros_like(arr, dtype=float)
        else:
            norm = (arr - vmin) / (vmax - vmin)
        norm[~finite] = 0.0
        try:
            from matplotlib import cm as mpl_cm
            cmap = getattr(mpl_cm, str(cmap_name), mpl_cm.viridis)
            rgba = np.asarray(cmap(norm), dtype=float)
        except Exception:
            rgba = np.column_stack(
                (
                    0.1 + 0.9 * norm,
                    0.2 + 0.7 * (1.0 - np.abs(norm - 0.5) * 2.0),
                    1.0 - norm * 0.8,
                    np.ones_like(norm),
                )
            )
        alpha = np.full((len(rgba), 1), 0.98, dtype=float)
        rgba = np.concatenate((rgba[:, :3], alpha), axis=1)
        if self._show_all_nodes:
            show_ids = np.arange(len(nodes), dtype=int)
        elif self._surface_node_ids.size > 0:
            show_ids = np.asarray(self._surface_node_ids, dtype=int)
        else:
            show_ids = np.arange(len(nodes), dtype=int)
        show_ids = show_ids[(show_ids >= 0) & (show_ids < len(nodes))]
        if show_ids.size == 0:
            show_ids = np.arange(len(nodes), dtype=int)
        node_size = self._node_size_override or self._last_node_size or self._node_marker_size(nodes)
        if self.node_item is None:
            self.node_item = GLScatterPlotItem(pos=nodes[show_ids], size=node_size, color=rgba[show_ids])
            self.addItem(self.node_item)
        else:
            self.node_item.setData(pos=nodes[show_ids], size=node_size, color=rgba[show_ids])
        if self.node_item_surface is not None and self._surface_node_ids.size > 0:
            ids = np.asarray(self._surface_node_ids, dtype=int)
            ids = ids[(ids >= 0) & (ids < len(nodes))]
            if ids.size > 0:
                self.node_item_surface.setData(pos=nodes[ids], size=node_size, color=rgba[ids])
        if self.node_item_interior is not None and self._interior_node_ids.size > 0:
            ids = np.asarray(self._interior_node_ids, dtype=int)
            ids = ids[(ids >= 0) & (ids < len(nodes))]
            if ids.size > 0:
                self.node_item_interior.setData(pos=nodes[ids], size=node_size, color=rgba[ids])
        self._node_colors_auto = False

    def set_show_all_nodes(self, enabled):
        self._show_all_nodes = bool(enabled)
        if self._last_mesh is not None:
            nodes, tets = self._last_mesh
            groups = self._face_group_ids.copy() if isinstance(self._face_group_ids, np.ndarray) else self._face_group_ids
            self.set_mesh(nodes, tets, face_group_ids=groups)

    def set_hover_nodes(self, nodes, ids=None, enabled=True):
        if nodes is None:
            self._hover_nodes = None
            self._hover_ids = None
            self._selected_node_id = None
            self._selected_node_pos = None
            self._update_selected_node_marker()
            return
        nodes = np.asarray(nodes, dtype=float)
        self._hover_nodes = nodes
        if ids is None:
            self._hover_ids = np.arange(len(nodes), dtype=int)
        else:
            self._hover_ids = np.asarray(ids, dtype=int)
        self._hover_enabled = bool(enabled)

    def _fit_camera_to_mesh(self, nodes):
        if nodes is None or len(nodes) == 0:
            return
        min_vals = nodes.min(axis=0)
        max_vals = nodes.max(axis=0)
        center = (min_vals + max_vals) / 2.0
        extent = max(max_vals - min_vals) if len(center) == 3 else max(max_vals - min_vals)
        extent = float(extent) if extent > 0 else 1.0
        distance = max(50.0, extent * 2.5)
        self._base_distance = distance
        self._center = QVector3D(float(center[0]), float(center[1]), float(center[2]))
        self.setCameraPosition(pos=self._center, distance=distance)
        self._grid_extent = extent * 4.0
        self.grid_item.setSize(self._grid_extent, self._grid_extent)
        if self.axis_item is not None:
            axis_len = getattr(self, "_axis_length", 40.0)
            self.axis_item.setSize(axis_len, axis_len, axis_len)
            self._update_axis_labels()
        self._update_plane_axes()
        spacing = max(1.0, extent / 10.0)
        self.grid_item.setSpacing(spacing, spacing)
        self._grid_spacing = spacing

    def _node_marker_size(self, nodes):
        if nodes is None or len(nodes) == 0:
            return 2.0
        min_vals = nodes.min(axis=0)
        max_vals = nodes.max(axis=0)
        extent = float(max(max_vals - min_vals))
        size = max(2.0, min(10.0, extent * 0.02))
        return size

    def get_default_node_size(self):
        if self._node_size_override is not None:
            return float(self._node_size_override)
        if self._last_node_size is not None:
            return float(self._last_node_size)
        return 4.0

    def _stop_view_animation(self, finish=False):
        if self._view_anim_timer.isActive():
            self._view_anim_timer.stop()
        anim = self._view_anim
        self._view_anim = None
        if finish and anim:
            self.setCameraPosition(
                pos=self._center,
                distance=float(anim.get("target_distance", self.opts.get("distance", self._base_distance))),
                azimuth=float(anim.get("target_azimuth", self.opts.get("azimuth", 0.0))),
                elevation=float(anim.get("target_elevation", self.opts.get("elevation", 0.0))),
            )

    def _step_view_animation(self):
        anim = self._view_anim
        if not anim:
            self._stop_view_animation(finish=False)
            return
        frame = int(anim.get("frame", 0)) + 1
        frames = max(1, int(anim.get("frames", 1)))
        t = min(1.0, float(frame) / float(frames))
        smooth = t * t * (3.0 - 2.0 * t)
        start_az = float(anim.get("start_azimuth", 0.0))
        target_az = float(anim.get("target_azimuth", 0.0))
        start_el = float(anim.get("start_elevation", 0.0))
        target_el = float(anim.get("target_elevation", 0.0))
        start_dist = float(anim.get("start_distance", self.opts.get("distance", self._base_distance)))
        target_dist = float(anim.get("target_distance", start_dist))
        az = start_az + self._norm_angle_deg(target_az - start_az) * smooth
        el = start_el + (target_el - start_el) * smooth
        dist = start_dist + (target_dist - start_dist) * smooth
        self.setCameraPosition(
            pos=self._center,
            distance=dist,
            azimuth=az,
            elevation=el,
        )
        anim["frame"] = frame
        self.update()
        if t >= 1.0:
            self._stop_view_animation(finish=True)

    def set_view(self, azimuth, elevation, animated=True):
        try:
            current_az = float(self.opts.get("azimuth", 0.0))
            current_el = float(self.opts.get("elevation", 0.0))
            current_dist = float(self.opts.get("distance", self._base_distance or 200.0))
        except Exception:
            current_az = 0.0
            current_el = 0.0
            current_dist = float(self._base_distance or 200.0)
        target_az = float(azimuth)
        target_el = float(elevation)
        if not animated:
            self._stop_view_animation(finish=False)
            self.setCameraPosition(
                pos=self._center,
                distance=current_dist,
                azimuth=target_az,
                elevation=target_el,
            )
            self.update()
            return
        da = abs(self._norm_angle_deg(target_az - current_az))
        de = abs(target_el - current_el)
        if da < 0.25 and de < 0.25:
            self._stop_view_animation(finish=False)
            self.setCameraPosition(
                pos=self._center,
                distance=current_dist,
                azimuth=target_az,
                elevation=target_el,
            )
            self.update()
            return
        self._view_anim = {
            "frame": 0,
            "frames": 14,
            "start_azimuth": current_az,
            "target_azimuth": target_az,
            "start_elevation": current_el,
            "target_elevation": target_el,
            "start_distance": current_dist,
            "target_distance": current_dist,
        }
        if not self._view_anim_timer.isActive():
            self._view_anim_timer.start()

    def set_random_view(self):
        azimuth = random.uniform(0, 360)
        elevation = random.uniform(-75, 75)
        self.set_view(azimuth, elevation)

    def fit_view(self):
        if self._last_mesh is not None:
            nodes, _ = self._last_mesh
            self._fit_camera_to_mesh(nodes)
            return
        if self._last_meshes is not None:
            min_vals = None
            max_vals = None
            for mesh in self._last_meshes:
                nodes = np.asarray(mesh.get("nodes", []), dtype=float)
                if nodes.size == 0:
                    continue
                cur_min = nodes.min(axis=0)
                cur_max = nodes.max(axis=0)
                if min_vals is None:
                    min_vals = cur_min
                    max_vals = cur_max
                else:
                    min_vals = np.minimum(min_vals, cur_min)
                    max_vals = np.maximum(max_vals, cur_max)
            if min_vals is not None:
                bounds = np.vstack((min_vals, max_vals))
                self._fit_camera_to_mesh(bounds)

    def center_origin(self):
        self._center = QVector3D(0.0, 0.0, 0.0)
        distance = self._base_distance if self._base_distance else self.opts.get("distance", 200)
        try:
            azimuth = self.opts.get("azimuth", None)
            elevation = self.opts.get("elevation", None)
        except Exception:
            azimuth = None
            elevation = None
        kwargs = {"pos": self._center, "distance": distance}
        if azimuth is not None:
            kwargs["azimuth"] = azimuth
        if elevation is not None:
            kwargs["elevation"] = elevation
        self.setCameraPosition(**kwargs)

    def mouseDoubleClickEvent(self, event):
        self.set_random_view()
        event.accept()

    def set_mesh(self, nodes, tets, face_group_ids=None):
        if nodes is None or tets is None:
            self.clear_mesh()
            return
        self._last_mesh = (np.asarray(nodes, dtype=float), np.asarray(tets, dtype=int))
        self._last_meshes = None
        nodes = np.asarray(nodes, dtype=float)
        tets = np.asarray(tets, dtype=int)
        if nodes.size == 0 or tets.size == 0:
            self.clear_mesh()
            return
        for item in self.mesh_items:
            self.removeItem(item)
        self.mesh_items = []
        faces = None
        if tets.ndim == 2 and tets.shape[1] == 3:
            faces = tets
        else:
            faces = _tets_to_triangles(tets)
        nodes, faces, groups = _sanitize_triangle_faces(nodes, faces, face_group_ids=face_group_ids)
        if faces.size > 0:
            self._surface_node_ids = np.unique(faces.reshape(-1))
        else:
            self._surface_node_ids = np.array([], dtype=int)
        self._last_faces = faces
        self._face_group_ids = None
        if groups is not None and isinstance(faces, np.ndarray) and faces.ndim == 2 and faces.shape[1] == 3:
            self._face_group_ids = groups
        elif isinstance(faces, np.ndarray) and faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0:
            inferred = _infer_face_groups_from_quads(nodes, faces)
            if inferred is not None and len(inferred) == len(faces):
                self._face_group_ids = inferred
        if faces.size > 0:
            edges = set()
            for tri in faces:
                a, b, c = [int(i) for i in tri]
                edges.add((min(a, b), max(a, b)))
                edges.add((min(b, c), max(b, c)))
                edges.add((min(c, a), max(c, a)))
            self._last_edges = sorted(edges)
        else:
            self._last_edges = []
        if self._surface_node_ids.size > 0:
            all_ids = np.arange(len(nodes), dtype=int)
            self._interior_node_ids = np.setdiff1d(all_ids, self._surface_node_ids, assume_unique=False)
        else:
            self._interior_node_ids = np.arange(len(nodes), dtype=int)
        if self.mesh_item is not None:
            self.removeItem(self.mesh_item)
        if self.node_item is not None:
            self.removeItem(self.node_item)
            self.node_item = None
        if self.node_item_surface is not None:
            self.removeItem(self.node_item_surface)
            self.node_item_surface = None
        if self.node_item_interior is not None:
            self.removeItem(self.node_item_interior)
            self.node_item_interior = None
        material = self._material_config()
        self.mesh_item = None
        if faces.size > 0:
            self.mesh_item = GLMeshItem(
                vertexes=nodes,
                faces=faces,
                smooth=True,
                drawEdges=False,
                drawFaces=material.get("draw_faces", True),
                shader=material["shader"],
                edgeColor=material["edge_color"],
                color=material["color"],
            )
        max_nodes = 40000
        node_size = self._node_size_override or self._node_marker_size(nodes)
        self._last_node_size = node_size
        contrast = self._contrast_nodes or {}
        surface_color = self._node_color_surface_override or contrast.get(
            "surface", (0.28, 0.29, 0.31, 0.85)
        )
        interior_color = self._node_color_interior_override or contrast.get(
            "interior", (0.32, 0.33, 0.35, 0.45)
        )
        all_color = self._node_color_override or contrast.get("all", (0.35, 0.36, 0.38, 0.18))
        surface_ids = self._surface_node_ids
        interior_ids = self._interior_node_ids
        if surface_ids.size > 0 and interior_ids.size > 0:
            max_each = max(1, max_nodes // 2)
        else:
            max_each = max_nodes
        surface_sample = self._sample_indices(surface_ids, max_each)
        interior_sample = self._sample_indices(interior_ids, max_each)
        if surface_sample.size > 0:
            self.node_item_surface = GLScatterPlotItem(
                pos=nodes[surface_sample], size=node_size, color=surface_color
            )
            self.addItem(self.node_item_surface)
        if interior_sample.size > 0:
            self.node_item_interior = GLScatterPlotItem(
                pos=nodes[interior_sample], size=node_size, color=interior_color
            )
            self.addItem(self.node_item_interior)
        if self._show_all_nodes:
            node_points = nodes
        elif surface_sample.size > 0:
            node_points = nodes[surface_sample]
        else:
            node_points = nodes
        self.node_item = GLScatterPlotItem(pos=node_points, size=node_size, color=all_color)
        self.addItem(self.node_item)
        if self.mesh_item is not None:
            self.addItem(self.mesh_item)
            self.mesh_item.setVisible(self.show_mesh)
        self._update_wireframe_overlay()
        self.set_nodes_visible(
            show_nodes=self.show_nodes,
            show_surface=self.show_surface_nodes,
            show_interior=self.show_interior_nodes,
        )
        self._rebuild_cad_topology_node_map()
        self._update_face_highlight()
        self._update_edge_highlight()
        self._update_selected_node_marker()
        self._fit_camera_to_mesh(nodes)

    def set_cad_topology(self, topology):
        self._cad_topology = {"vertices": [], "edges": []}
        self._cad_topology_vertices = np.zeros((0, 3), dtype=float)
        self._cad_topology_vertex_ids = np.zeros((0,), dtype=int)
        self._cad_topology_vertex_lookup = {}
        self._cad_topology_edges = []
        self._cad_topology_edge_lookup = {}
        if not topology:
            self.selected_topology_vertices = set()
            self.selected_topology_edges = set()
            self._cad_topology_vertex_node_ids = {}
            self._cad_topology_edge_node_ids = {}
            self._update_selected_nodes_marker()
            self._update_edge_highlight()
            return
        vertices_in = topology.get("vertices", []) if isinstance(topology, dict) else []
        edges_in = topology.get("edges", []) if isinstance(topology, dict) else []
        vertices = []
        vertex_ids = []
        vertex_lookup = {}
        for item in vertices_in:
            if not isinstance(item, dict):
                continue
            try:
                vid = int(item.get("id"))
                p = np.asarray(item.get("point", []), dtype=float).reshape(-1)
                if p.size < 3:
                    continue
                point = np.array([float(p[0]), float(p[1]), float(p[2])], dtype=float)
            except Exception:
                continue
            vertices.append(point)
            vertex_ids.append(vid)
            vertex_lookup[vid] = point
        if vertices:
            self._cad_topology_vertices = np.vstack(vertices)
            self._cad_topology_vertex_ids = np.asarray(vertex_ids, dtype=int)
            self._cad_topology_vertex_lookup = vertex_lookup
        edges = []
        edge_lookup = {}
        for item in edges_in:
            if not isinstance(item, dict):
                continue
            try:
                eid = int(item.get("id"))
            except Exception:
                continue
            pts = np.asarray(item.get("points", []), dtype=float)
            if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 3:
                continue
            pts = pts[:, :3]
            vertex_ids_pair = item.get("vertex_ids", (-1, -1))
            try:
                va = int(vertex_ids_pair[0])
            except Exception:
                va = -1
            try:
                vb = int(vertex_ids_pair[1])
            except Exception:
                vb = -1
            edge_data = {
                "id": eid,
                "points": pts,
                "is_closed": bool(item.get("is_closed", False)),
                "curve_type": str(item.get("curve_type", "")),
                "vertex_ids": (va, vb),
            }
            edges.append(edge_data)
            edge_lookup[eid] = edge_data
        self._cad_topology_edges = edges
        self._cad_topology_edge_lookup = edge_lookup
        self._cad_topology = {
            "vertices": list(vertices_in) if isinstance(vertices_in, list) else [],
            "edges": list(edges_in) if isinstance(edges_in, list) else [],
        }
        valid_vids = set(self._cad_topology_vertex_lookup.keys())
        valid_eids = set(self._cad_topology_edge_lookup.keys())
        self.selected_topology_vertices = {
            int(v) for v in self.selected_topology_vertices if int(v) in valid_vids
        }
        self.selected_topology_edges = {
            int(e) for e in self.selected_topology_edges if int(e) in valid_eids
        }
        self._rebuild_cad_topology_node_map()
        self._update_selected_nodes_marker()
        self._update_edge_highlight()

    def _estimate_node_spacing(self, nodes):
        pts = np.asarray(nodes, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2:
            return 1e-6
        n = int(pts.shape[0])
        sample_n = min(n, 180)
        if sample_n < n:
            ids = np.linspace(0, n - 1, sample_n, dtype=int)
            sample = pts[ids]
        else:
            sample = pts
        diff = sample[:, None, :] - sample[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        np.fill_diagonal(d2, np.inf)
        min_d = np.sqrt(np.min(d2, axis=1))
        finite = min_d[np.isfinite(min_d)]
        if finite.size == 0:
            return 1e-6
        spacing = float(np.median(finite))
        if spacing <= 0:
            spacing = float(np.mean(finite))
        if spacing <= 0:
            spacing = 1e-6
        return spacing

    def _nearest_node_ids_for_point(self, nodes, point, base_tol):
        pts = np.asarray(nodes, dtype=float)
        p = np.asarray(point, dtype=float).reshape(-1)
        if pts.ndim != 2 or pts.shape[0] == 0 or p.size < 3:
            return set()
        p = p[:3]
        d2 = np.einsum("ij,ij->i", pts - p, pts - p)
        if d2.size == 0:
            return set()
        min_idx = int(np.argmin(d2))
        min_d = float(math.sqrt(max(0.0, d2[min_idx])))
        tol = max(float(base_tol), min_d * 1.05 + 1e-9)
        ids = np.where(d2 <= tol * tol)[0]
        if ids.size == 0:
            return {min_idx}
        return {int(i) for i in ids.tolist()}

    def _nearest_node_id_for_point(self, nodes, point):
        pts = np.asarray(nodes, dtype=float)
        p = np.asarray(point, dtype=float).reshape(-1)
        if pts.ndim != 2 or pts.shape[0] == 0 or p.size < 3:
            return None
        p = p[:3]
        d2 = np.einsum("ij,ij->i", pts - p, pts - p)
        if d2.size == 0:
            return None
        return int(np.argmin(d2))

    def _rebuild_cad_topology_node_map(self):
        self._cad_topology_vertex_node_ids = {}
        self._cad_topology_edge_node_ids = {}
        if self._last_mesh is None:
            return
        nodes, _ = self._last_mesh
        nodes = np.asarray(nodes, dtype=float)
        if nodes.ndim != 2 or nodes.shape[0] == 0:
            return
        if self._cad_topology_vertices is None or self._cad_topology_vertices.size == 0:
            return
        min_vals = nodes.min(axis=0)
        max_vals = nodes.max(axis=0)
        diag = float(np.linalg.norm(max_vals - min_vals))
        spacing = self._estimate_node_spacing(nodes)
        base_tol = max(diag * 1e-6, spacing * 0.3, 1e-8)
        for vid, point in self._cad_topology_vertex_lookup.items():
            nid = self._nearest_node_id_for_point(nodes, point)
            if nid is not None:
                self._cad_topology_vertex_node_ids[int(vid)] = {int(nid)}
        for edge in self._cad_topology_edges:
            eid = int(edge.get("id", -1))
            pts = np.asarray(edge.get("points", []), dtype=float)
            if eid < 0 or pts.ndim != 2 or pts.shape[0] < 2:
                continue
            node_ids = set()
            for p in pts:
                nid = self._nearest_node_id_for_point(nodes, p)
                if nid is not None:
                    node_ids.add(int(nid))
            va, vb = edge.get("vertex_ids", (-1, -1))
            if int(va) in self._cad_topology_vertex_node_ids:
                node_ids.update(self._cad_topology_vertex_node_ids[int(va)])
            if int(vb) in self._cad_topology_vertex_node_ids:
                node_ids.update(self._cad_topology_vertex_node_ids[int(vb)])
            if node_ids:
                self._cad_topology_edge_node_ids[eid] = node_ids

    def set_meshes(self, meshes):
        self.clear_mesh()
        self.mesh_items = []
        if not meshes:
            return
        self._last_mesh = None
        self._last_meshes = meshes
        material = self._material_config()
        all_nodes = []
        for mesh in meshes:
            nodes = np.asarray(mesh.get("nodes", []), dtype=float)
            faces = np.asarray(mesh.get("faces", []), dtype=int)
            if nodes.size == 0 or faces.size == 0:
                continue
            if faces.ndim == 2 and faces.shape[1] == 4:
                faces = _tets_to_triangles(faces)
            nodes, faces, _ = _sanitize_triangle_faces(nodes, faces)
            if nodes.size == 0 or faces.size == 0:
                continue
            all_nodes.append(nodes)
            color = material["color"]
            edge_color = material["edge_color"]
            if self._mesh_color_override is None:
                color = mesh.get("color", color)
            if self._mesh_edge_color_override is None:
                edge_color = mesh.get("edge_color", edge_color)
            item = GLMeshItem(
                vertexes=nodes,
                faces=faces,
                smooth=True,
                drawEdges=material["draw_edges"],
                drawFaces=material.get("draw_faces", True),
                shader=material["shader"],
                edgeColor=edge_color,
                color=color,
            )
            item.setVisible(self.show_mesh)
            self.mesh_items.append(item)
            self.addItem(item)
        if all_nodes:
            merged = np.vstack(all_nodes)
            self._fit_camera_to_mesh(merged)

    def set_grid_snap(self, enabled, spacing):
        self._snap_enabled = bool(enabled)
        if spacing:
            self._grid_spacing = float(spacing)
            self.grid_item.setSpacing(self._grid_spacing, self._grid_spacing)
        self._update_plane_axes()

    def _bg_luminance(self):
        r, g, b = [max(0.0, min(1.0, v / 255.0)) for v in self._bg_color]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def _set_axis_colors(self, x_color, y_color, z_color, width=4.5):
        if self.axis_item is None or not hasattr(self.axis_item, "lineplot"):
            return
        try:
            x, y, z = self.axis_item.size()
        except Exception:
            size = self.axis_item.size()
            x, y, z = size[0], size[1], size[2]
        pos = np.array(
            [
                [0, 0, 0, x, 0, 0],
                [0, 0, 0, 0, y, 0],
                [0, 0, 0, 0, 0, z],
            ],
            dtype=np.float32,
        ).reshape((-1, 3))
        colors = np.array([x_color, y_color, z_color], dtype=np.float32)
        colors = np.hstack((colors, colors)).reshape((-1, 4))
        self.axis_item.lineplot.setData(pos=pos, color=colors, width=width)
        self.axis_item.update()

    def _update_axis_labels(self):
        if not self.axis_labels:
            return
        try:
            size = self.axis_item.size()
            axis_len = float(size[0])
        except Exception:
            axis_len = 40.0
        offset = axis_len * 0.02
        self.axis_labels["x"].setData(pos=(axis_len + offset, 0.0, 0.0))
        self.axis_labels["y"].setData(pos=(0.0, axis_len + offset, 0.0))
        self.axis_labels["z"].setData(pos=(0.0, 0.0, axis_len + offset))

    def _update_plane_axes(self):
        if self.plane_axes_item is None:
            return
        axis_len = getattr(self, "_axis_length", 40.0)
        half = float(axis_len)
        pos = np.array(
            [
                [-half, 0.0, 0.0],
                [half, 0.0, 0.0],
                [0.0, -half, 0.0],
                [0.0, half, 0.0],
            ],
            dtype=np.float32,
        )
        x_color = (1.0, 0.0, 0.0, 0.95)
        y_color = (0.0, 1.0, 0.0, 0.95)
        colors = np.array([x_color, x_color, y_color, y_color], dtype=np.float32)
        self.plane_axes_item.setData(pos=pos, color=colors, width=4.0)

    def _apply_contrast_theme(self):
        light_bg = self._bg_luminance() >= 0.6
        if light_bg:
            self._contrast_palette = {
                "grid": (0.44, 0.5, 0.58, 0.46),
                "axis": (0.18, 0.19, 0.2, 0.9),
                "wireframe": (0.12, 0.15, 0.19, 0.58),
            }
            self._contrast_nodes = {
                "all": (0.36, 0.39, 0.43, 0.2),
                "surface": (0.31, 0.34, 0.38, 0.88),
                "interior": (0.38, 0.42, 0.46, 0.48),
                "selected": (0.0, 0.86, 0.98, 0.98),
            }
        else:
            self._contrast_palette = {
                "grid": (0.7, 0.72, 0.76, 0.56),
                "axis": (0.92, 0.94, 0.98, 0.9),
                "wireframe": (0.84, 0.87, 0.92, 0.58),
            }
            self._contrast_nodes = {
                "all": (0.78, 0.81, 0.85, 0.22),
                "surface": (0.94, 0.96, 0.98, 0.98),
                "interior": (0.72, 0.76, 0.81, 0.48),
                "selected": (0.35, 0.95, 1.0, 0.98),
            }
        if self.grid_item is not None:
            self.grid_item.setColor(self._contrast_palette["grid"])
        self._set_axis_colors(
            (1.0, 0.0, 0.0, 0.95),
            (0.0, 1.0, 0.0, 0.95),
            (0.0, 0.0, 1.0, 0.95),
        )
        self._update_plane_axes()
        self._update_axis_labels()
        self._update_node_colors()
        self._update_wireframe_style()

    def set_background_color(self, color):
        if color is None:
            return
        if hasattr(color, "getRgb"):
            r, g, b, _a = color.getRgb()
            rgb = (int(r), int(g), int(b))
        else:
            rgb = tuple(int(v) for v in color[:3])
        self._bg_color = rgb
        self.setBackgroundColor(*rgb)
        self._apply_contrast_theme()

    def set_node_colors_auto(self):
        self._node_colors_auto = True
        self._node_color_override = None
        self._node_color_surface_override = None
        self._node_color_interior_override = None
        self._node_color_selected_override = None
        self._update_node_colors()

    def set_mesh_color_override(self, color=None, edge_color=None):
        self._mesh_color_override = color
        self._mesh_edge_color_override = edge_color
        self._refresh_mesh_style()

    def clear_mesh_color_override(self):
        self.set_mesh_color_override(None, None)

    def set_material_style(self, style):
        if style:
            self._material_style = str(style).lower()
        if self._last_mesh is not None:
            nodes, faces = self._last_mesh
            groups = self._face_group_ids.copy() if isinstance(self._face_group_ids, np.ndarray) else self._face_group_ids
            self.set_mesh(nodes, faces, face_group_ids=groups)
            return
        if self._last_meshes is not None:
            self.set_meshes(self._last_meshes)

    def _material_config(self):
        presets = {
            "dark": {
                "color": (0.45, 0.46, 0.5, 0.98),
                "edge_color": (0.18, 0.18, 0.2, 0.7),
                "shader": "shaded",
                "draw_edges": True,
                "draw_faces": True,
            },
            "plastic": {
                "color": (0.62, 0.64, 0.68, 0.97),
                "edge_color": (0.17, 0.19, 0.22, 0.3),
                "shader": "shaded",
                "draw_edges": True,
                "draw_faces": True,
            },
            "mesh": {
                "color": (0.58, 0.6, 0.64, 0.42),
                "edge_color": (0.16, 0.18, 0.22, 0.54),
                "shader": "shaded",
                "draw_edges": True,
                "draw_faces": True,
            },
            "metal": {
                "color": (0.56, 0.58, 0.62, 0.98),
                "edge_color": (0.14, 0.16, 0.19, 0.2),
                "shader": "shaded",
                "draw_edges": False,
                "draw_faces": True,
            },
            "silver": {
                "color": (0.72, 0.75, 0.79, 0.98),
                "edge_color": (0.18, 0.2, 0.24, 0.2),
                "shader": "shaded",
                "draw_edges": False,
                "draw_faces": True,
            },
            "clay": {
                "color": (0.64, 0.66, 0.7, 0.97),
                "edge_color": (0.18, 0.2, 0.24, 0.26),
                "shader": "shaded",
                "draw_edges": True,
                "draw_faces": True,
            },
            "wireframe": {
                "color": (0.0, 0.0, 0.0, 0.0),
                "edge_color": (0.2, 0.2, 0.22, 0.85),
                "shader": "shaded",
                "draw_edges": True,
                "draw_faces": False,
            },
        }
        material = dict(presets.get(self._material_style, presets["plastic"]))
        if self._mesh_color_override is not None:
            material["color"] = self._mesh_color_override
        if self._mesh_edge_color_override is not None:
            material["edge_color"] = self._mesh_edge_color_override
        if self._mesh_dim_enabled and self._material_style != "dark":
            r, g, b, _a = material["edge_color"]
            material["edge_color"] = (r, g, b, 0.12)
        if self._mesh_xray_enabled:
            r, g, b, _a = material["color"]
            material["color"] = (r, g, b, 0.12)
            material["draw_faces"] = True
        return material

    def _update_node_colors(self):
        contrast = self._contrast_nodes or {}
        all_color = self._node_color_override or contrast.get("all", (0.35, 0.36, 0.38, 0.18))
        surface_color = self._node_color_surface_override or contrast.get(
            "surface", (0.28, 0.29, 0.31, 0.85)
        )
        interior_color = self._node_color_interior_override or contrast.get(
            "interior", (0.32, 0.33, 0.35, 0.45)
        )
        selected_color = self._node_color_selected_override or contrast.get(
            "selected", (0.55, 0.57, 0.6, 0.95)
        )
        if self.node_item is not None:
            self.node_item.setData(color=all_color)
        if self.node_item_surface is not None:
            self.node_item_surface.setData(color=surface_color)
        if self.node_item_interior is not None:
            self.node_item_interior.setData(color=interior_color)
        if self.node_item_selected is not None:
            self.node_item_selected.setData(color=selected_color)

    def _refresh_mesh_style(self):
        if self._last_mesh is not None:
            nodes, faces = self._last_mesh
            groups = self._face_group_ids.copy() if isinstance(self._face_group_ids, np.ndarray) else self._face_group_ids
            self.set_mesh(nodes, faces, face_group_ids=groups)
        elif self._last_meshes is not None:
            self.set_meshes(self._last_meshes)
        self._update_wireframe_style()

    def _update_wireframe_style(self):
        if self._wireframe_item is None:
            return
        base = (self._contrast_palette or {}).get("wireframe", (0.1, 0.1, 0.12, 0.55))
        r, g, b, a = base
        if self._mesh_dim_enabled:
            a = 0.18
        if self._mesh_xray_enabled:
            a = min(a, 0.15)
        self._wireframe_item.setData(color=(r, g, b, a))

    def _update_wireframe_overlay(self):
        if self._wireframe_item is not None:
            self.removeItem(self._wireframe_item)
            self._wireframe_item = None
        if not self._wireframe_enabled or self._last_faces is None or self._last_mesh is None:
            return
        nodes, _ = self._last_mesh
        faces = self._last_faces
        if nodes.size == 0 or faces.size == 0:
            return
        edges = set()
        for tri in faces:
            a, b, c = [int(i) for i in tri]
            edges.add((min(a, b), max(a, b)))
            edges.add((min(b, c), max(b, c)))
            edges.add((min(c, a), max(c, a)))
        segments = []
        for a, b in edges:
            segments.append(nodes[a])
            segments.append(nodes[b])
        if not segments:
            return
        pos = np.asarray(segments, dtype=float)
        r, g, b, a = (0.1, 0.1, 0.12, 0.55)
        if self._mesh_dim_enabled:
            a = 0.18
        if self._mesh_xray_enabled:
            a = min(a, 0.15)
        self._wireframe_item = GLLinePlotItem(
            pos=pos, color=(r, g, b, a), width=1.0, mode="lines"
        )
        self.addItem(self._wireframe_item)
        self._wireframe_item.setVisible(self.show_mesh and self._wireframe_enabled)

    def set_selection_mode(self, mode):
        mode = str(mode).lower()
        if mode not in {"none", "auto", "face", "point", "edge"}:
            mode = "none"
        if mode != self._selection_mode:
            self._selection_mode = mode
            self._clear_pick_hover()
            try:
                window = self.window()
                if window and hasattr(window, "_update_interaction_hints"):
                    window._update_interaction_hints()
            except Exception:
                pass

    def set_view_navigation_enabled(self, enabled):
        self._view_navigation_enabled = bool(enabled)
        if not self._view_navigation_enabled:
            self._nav_panning = False
            self._nav_rotating = False
            self._nav_last_pos = None
            self._nav_right_dragged = False
        try:
            window = self.window()
            if window and hasattr(window, "_update_interaction_hints"):
                window._update_interaction_hints()
        except Exception:
            pass

    def _begin_view_pan(self, pos):
        self._nav_panning = True
        self._nav_last_pos = pos

    def _begin_view_rotate(self, pos):
        self._nav_rotating = True
        self._nav_last_pos = pos
        self._nav_right_dragged = False

    def _is_synthesized_pointer_event(self, event):
        if not hasattr(event, "source"):
            return False
        try:
            source = event.source()
        except Exception:
            return False
        if source is None:
            return False
        source_name = getattr(source, "name", str(source))
        return "Synthesized" in source_name and "NotSynthesized" not in source_name

    def _apply_zoom_delta(self, delta):
        delta = float(delta)
        if abs(delta) <= 1e-6:
            return
        distance = float(self.opts.get("distance", 10.0))
        factor = math.pow(0.999, delta)
        self.opts["distance"] = max(0.25, min(1.0e6, distance * factor))
        self.update()

    def view_navigation_enabled(self):
        return bool(self._view_navigation_enabled)

    def _emit_selection_changed(self):
        try:
            self.selectionChanged.emit()
        except Exception:
            pass

    def clear_selection(self):
        self.selected_faces = set()
        self._update_face_highlight()
        self._emit_selection_changed()

    def get_selected_faces(self):
        return set(self.selected_faces)

    def get_selected_face_nodes(self):
        if not self.selected_faces or self._last_faces is None:
            return set()
        nodes = set()
        for idx in self.selected_faces:
            if 0 <= idx < len(self._last_faces):
                tri = self._last_faces[idx]
                for nid in tri:
                    nodes.add(int(nid))
        return nodes

    def clear_node_selection(self):
        self.selected_nodes = set()
        self.selected_topology_vertices = set()
        self._update_selected_nodes_marker()
        self._emit_selection_changed()

    def get_selected_nodes(self):
        return set(self.selected_nodes)

    def clear_edge_selection(self):
        self.selected_edges = set()
        self.selected_topology_edges = set()
        self._update_edge_highlight()
        self._emit_selection_changed()

    def get_selected_edges(self):
        return set(self.selected_edges)

    def get_selection_counts(self):
        return {
            "faces": len(self.selected_faces),
            "edges": len(self.selected_edges) + len(self.selected_topology_edges),
            "points": len(self.selected_nodes) + len(self.selected_topology_vertices),
        }

    def clear_bc_load_markers(self):
        for item in list(getattr(self, "_bc_load_marker_items", [])):
            try:
                self.removeItem(item)
            except Exception:
                pass
        self._bc_load_marker_items = []

    def _marker_anchor_from_nodes(self, node_ids=None):
        if self._last_mesh is None:
            return None
        nodes, _ = self._last_mesh
        nodes = np.asarray(nodes, dtype=float)
        if nodes.ndim != 2 or nodes.shape[0] == 0:
            return None
        ids = []
        if node_ids is not None:
            for nid in node_ids:
                try:
                    idx = int(nid)
                except Exception:
                    continue
                if 0 <= idx < len(nodes):
                    ids.append(idx)
        if not ids and self.selected_faces and self._last_faces is not None:
            face_ids = [idx for idx in self.selected_faces if 0 <= idx < len(self._last_faces)]
            if face_ids:
                tri = np.asarray(self._last_faces[face_ids], dtype=int).reshape(-1)
                ids = [int(v) for v in np.unique(tri) if 0 <= int(v) < len(nodes)]
        if not ids and self.selected_nodes:
            ids = [int(v) for v in self.selected_nodes if 0 <= int(v) < len(nodes)]
        if not ids:
            return None
        pts = nodes[np.asarray(ids, dtype=int), :3]
        if pts.size == 0:
            return None
        return np.mean(pts, axis=0)

    def add_bc_load_marker(self, marker_type, node_ids=None, axis=None):
        anchor = self._marker_anchor_from_nodes(node_ids=node_ids)
        if anchor is None:
            return
        marker = str(marker_type or "").strip().lower()
        axis = str(axis or "z").strip().lower()
        axis_vec = np.array([0.0, 0.0, 1.0], dtype=float)
        if axis == "x":
            axis_vec = np.array([1.0, 0.0, 0.0], dtype=float)
        elif axis == "y":
            axis_vec = np.array([0.0, 1.0, 0.0], dtype=float)
        span = 10.0
        if self._last_mesh is not None:
            try:
                pts = np.asarray(self._last_mesh[0], dtype=float)
                bb = np.ptp(pts, axis=0)
                span = max(float(np.linalg.norm(bb)) * 0.06, 5.0)
            except Exception:
                span = 10.0
        axis_colors = {
            "x": (0.95, 0.25, 0.2, 0.95),
            "y": (0.20, 0.72, 0.30, 0.95),
            "z": (0.25, 0.48, 0.95, 0.95),
        }
        color = axis_colors.get(axis, (0.95, 0.25, 0.2, 0.95))
        if marker == "velocity":
            color = axis_colors.get(axis, (0.25, 0.78, 0.95, 0.95))
        elif marker == "force":
            color = axis_colors.get(axis, (0.95, 0.25, 0.2, 0.95))
        if marker == "fixed":
            color = (1.0, 0.62, 0.05, 0.95)

        def _add_line(points, width=2.5, rgba=color):
            item = GLLinePlotItem(
                pos=np.asarray(points, dtype=float),
                color=rgba,
                width=float(width),
                mode="lines",
            )
            self.addItem(item)
            self._bc_load_marker_items.append(item)

        if marker in {"force", "velocity"}:
            tip = anchor + axis_vec * span
            _add_line([anchor, tip], width=2.8, rgba=color)
            perp = np.cross(axis_vec, np.array([0.0, 1.0, 0.0], dtype=float))
            if float(np.linalg.norm(perp)) < 1e-6:
                perp = np.cross(axis_vec, np.array([1.0, 0.0, 0.0], dtype=float))
            perp = perp / max(float(np.linalg.norm(perp)), 1e-9)
            head_len = span * 0.25
            h1 = tip - axis_vec * head_len + perp * head_len * 0.5
            h2 = tip - axis_vec * head_len - perp * head_len * 0.5
            _add_line([tip, h1], width=2.2, rgba=color)
            _add_line([tip, h2], width=2.2, rgba=color)
            return

        if marker == "fixed":
            d = span * 0.32
            a = anchor + np.array([-d, -d, 0.0], dtype=float)
            b = anchor + np.array([d, d, 0.0], dtype=float)
            c = anchor + np.array([-d, d, 0.0], dtype=float)
            e = anchor + np.array([d, -d, 0.0], dtype=float)
            _add_line([a, b], width=3.0, rgba=color)
            _add_line([c, e], width=3.0, rgba=color)

    def highlight_node_ids(self, node_ids, target="auto"):
        target = str(target or "auto").lower()
        if target not in {"auto", "face", "edge", "point"}:
            target = "auto"
        self.selected_faces = set()
        self.selected_edges = set()
        self.selected_topology_edges = set()
        self.selected_nodes = set()
        self.selected_topology_vertices = set()

        if node_ids is None:
            self._update_face_highlight()
            self._update_edge_highlight()
            self._update_selected_nodes_marker()
            self._emit_selection_changed()
            return

        ids = set()
        for nid in node_ids:
            try:
                ids.add(int(nid))
            except Exception:
                continue
        if not ids:
            self._update_face_highlight()
            self._update_edge_highlight()
            self._update_selected_nodes_marker()
            self._emit_selection_changed()
            return

        valid_node_ids = set()
        if self._last_mesh is not None:
            nodes, _ = self._last_mesh
            try:
                node_count = int(len(nodes))
            except Exception:
                node_count = 0
            valid_node_ids = {nid for nid in ids if 0 <= nid < node_count}
        self.selected_nodes = set(valid_node_ids)

        faces = self._last_faces
        if faces is not None and target in {"auto", "face"}:
            try:
                faces_arr = np.asarray(faces, dtype=int)
            except Exception:
                faces_arr = None
            if faces_arr is not None and faces_arr.ndim == 2 and faces_arr.shape[1] == 3:
                face_matches = set()
                for idx, tri in enumerate(faces_arr):
                    tri_ids = {int(tri[0]), int(tri[1]), int(tri[2])}
                    if tri_ids.issubset(valid_node_ids):
                        face_matches.add(int(idx))
                if not face_matches and target == "face":
                    for idx, tri in enumerate(faces_arr):
                        if any(int(v) in valid_node_ids for v in tri):
                            face_matches.add(int(idx))
                self.selected_faces = face_matches

        if target in {"auto", "edge"} and self._last_edges:
            edge_matches = set()
            for edge in self._last_edges:
                try:
                    a, b = int(edge[0]), int(edge[1])
                except Exception:
                    continue
                if a in valid_node_ids and b in valid_node_ids:
                    edge_matches.add((min(a, b), max(a, b)))
            self.selected_edges = edge_matches

        self._update_face_highlight()
        self._update_edge_highlight()
        self._update_selected_nodes_marker()
        self._emit_selection_changed()

    def focus_node_ids(self, node_ids):
        if self._last_mesh is None or node_ids is None:
            return
        try:
            nodes, _ = self._last_mesh
            nodes_arr = np.asarray(nodes, dtype=float)
        except Exception:
            return
        if nodes_arr.ndim != 2 or nodes_arr.shape[0] == 0 or nodes_arr.shape[1] < 3:
            return
        valid_ids = []
        for nid in node_ids:
            try:
                idx = int(nid)
            except Exception:
                continue
            if 0 <= idx < len(nodes_arr):
                valid_ids.append(idx)
        if not valid_ids:
            return
        pts = nodes_arr[np.asarray(sorted(set(valid_ids)), dtype=int), :3]
        if pts.size == 0:
            return
        self._fit_camera_to_mesh(pts)

    def get_selected_node_ids_for_mode(self, mode):
        mode = str(mode).lower()
        if mode == "face":
            return sorted(self.get_selected_face_nodes())
        if mode == "point":
            ids = set(int(n) for n in self.selected_nodes)
            for vid in self.selected_topology_vertices:
                ids.update(self._cad_topology_vertex_node_ids.get(int(vid), set()))
            return sorted(int(nid) for nid in ids)
        if mode == "edge":
            ids = set()
            for edge in self.selected_edges:
                if isinstance(edge, (tuple, list)) and len(edge) == 2:
                    ids.add(int(edge[0]))
                    ids.add(int(edge[1]))
            for eid in self.selected_topology_edges:
                ids.update(self._cad_topology_edge_node_ids.get(int(eid), set()))
            return sorted(int(nid) for nid in ids)
        if mode == "auto":
            ids = set()
            ids.update(self.get_selected_face_nodes())
            ids.update(self.get_selected_node_ids_for_mode("edge"))
            ids.update(self.get_selected_node_ids_for_mode("point"))
            return sorted(int(nid) for nid in ids)
        return []

    def _begin_selection_box(self, event):
        if not hasattr(event, "position"):
            return
        p = event.position().toPoint()
        self._selection_box_start = QPoint(int(p.x()), int(p.y()))
        self._selection_box_active = True
        self._selection_box_dragged = False
        self._selection_box.setGeometry(QRect(self._selection_box_start, self._selection_box_start))
        self._selection_box.show()

    def _update_selection_box(self, event):
        if not self._selection_box_active or self._selection_box_start is None:
            return
        if not hasattr(event, "position"):
            return
        p = event.position().toPoint()
        rect = QRect(self._selection_box_start, QPoint(int(p.x()), int(p.y()))).normalized()
        self._selection_box.setGeometry(rect)
        if rect.width() > 4 or rect.height() > 4:
            self._selection_box_dragged = True

    def _finish_selection_box(self, event):
        if not self._selection_box_active:
            return False
        self._selection_box.hide()
        dragged = bool(self._selection_box_dragged)
        rect = QRect()
        if self._selection_box_start is not None and hasattr(event, "position"):
            p = event.position().toPoint()
            rect = QRect(self._selection_box_start, QPoint(int(p.x()), int(p.y()))).normalized()
        self._selection_box_active = False
        self._selection_box_start = None
        self._selection_box_dragged = False
        if dragged and rect.width() > 2 and rect.height() > 2:
            self._apply_selection_box(rect, append=bool(event.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier)))
            return True
        return False

    def _apply_selection_box(self, rect, append=False):
        if self._last_mesh is None:
            return
        nodes, _ = self._last_mesh
        nodes = np.asarray(nodes, dtype=float)
        if nodes.ndim != 2 or nodes.shape[0] == 0:
            return
        screen, _depth, visible = self._project_points_to_screen(nodes)
        if screen is None or visible is None:
            return
        x0 = float(rect.left())
        x1 = float(rect.right())
        y0 = float(rect.top())
        y1 = float(rect.bottom())
        in_rect = (
            visible
            & (screen[:, 0] >= x0)
            & (screen[:, 0] <= x1)
            & (screen[:, 1] >= y0)
            & (screen[:, 1] <= y1)
        )
        if not append:
            self.selected_faces = set()
            self.selected_edges = set()
            self.selected_topology_edges = set()
            self.selected_nodes = set()
            self.selected_topology_vertices = set()
        mode = self._selection_mode
        face_hits = set()
        edge_hits = set()
        point_hits = {int(i) for i in np.where(in_rect)[0]}
        if self._last_faces is not None:
            faces = np.asarray(self._last_faces, dtype=int)
            for fi, tri in enumerate(faces):
                tri = np.asarray(tri, dtype=int)
                if np.all(in_rect[tri]):
                    face_hits.add(int(fi))
            groups = getattr(self, "_face_group_ids", None)
            if groups is not None and len(face_hits) > 0:
                expanded = set()
                for fi in face_hits:
                    if 0 <= fi < len(groups):
                        gid = int(groups[fi])
                        grp = np.where(np.asarray(groups, dtype=int) == gid)[0]
                        expanded.update(int(i) for i in grp.tolist())
                if expanded:
                    face_hits = expanded
        if self._last_edges:
            for edge in self._last_edges:
                i, j = int(edge[0]), int(edge[1])
                if i < len(in_rect) and j < len(in_rect) and bool(in_rect[i]) and bool(in_rect[j]):
                    edge_hits.add((min(i, j), max(i, j)))

        if mode == "face":
            self.selected_faces.update(face_hits)
        elif mode == "edge":
            self.selected_edges.update(edge_hits)
        elif mode == "point":
            self.selected_nodes.update(point_hits)
        else:
            if face_hits:
                self.selected_faces.update(face_hits)
            elif edge_hits:
                self.selected_edges.update(edge_hits)
            else:
                self.selected_nodes.update(point_hits)
        self._update_face_highlight()
        self._update_edge_highlight()
        self._update_selected_nodes_marker()
        self._emit_selection_changed()

    def _clear_pick_hover(self):
        self._hover_face_indices = set()
        self._hover_edge_pick = None
        self._hover_point_pick = None
        if self._hover_face_item is not None:
            self.removeItem(self._hover_face_item)
            self._hover_face_item = None
        if self._hover_edge_item is not None:
            self.removeItem(self._hover_edge_item)
            self._hover_edge_item = None
        if self._hover_point_item is not None:
            self.removeItem(self._hover_point_item)
            self._hover_point_item = None

    def _set_hover_faces(self, faces):
        face_set = {int(idx) for idx in faces}
        if face_set == self._hover_face_indices:
            return
        self._hover_face_indices = face_set
        if self._hover_face_item is not None:
            self.removeItem(self._hover_face_item)
            self._hover_face_item = None
        if not self._hover_face_indices or self._last_mesh is None or self._last_faces is None:
            return
        nodes, _ = self._last_mesh
        faces_arr = np.asarray(self._last_faces, dtype=int)
        indices = [idx for idx in sorted(self._hover_face_indices) if 0 <= idx < len(faces_arr)]
        if not indices:
            return
        highlight_faces = faces_arr[indices]
        self._hover_face_item = GLMeshItem(
            vertexes=nodes,
            faces=highlight_faces,
            smooth=False,
            drawEdges=True,
            drawFaces=True,
            shader="shaded",
            edgeColor=(1.0, 0.66, 0.12, 0.98),
            color=(1.0, 0.82, 0.18, 0.3),
        )
        try:
            self._hover_face_item.setGLOptions("additive")
        except Exception:
            pass
        try:
            self._hover_face_item.setDepthValue(18)
        except Exception:
            pass
        self._hover_face_item.setVisible(self.show_mesh)
        self.addItem(self._hover_face_item)

    def _set_hover_edge(self, pick):
        if pick == self._hover_edge_pick:
            return
        self._hover_edge_pick = pick
        if self._hover_edge_item is not None:
            self.removeItem(self._hover_edge_item)
            self._hover_edge_item = None
        if pick is None:
            return
        kind, value = pick
        segments = []
        if kind == "mesh" and self._last_mesh is not None:
            nodes, _ = self._last_mesh
            i, j = value
            if 0 <= i < len(nodes) and 0 <= j < len(nodes):
                segments = [nodes[i], nodes[j]]
        elif kind == "topology":
            edge = self._cad_topology_edge_lookup.get(int(value))
            if edge:
                pts = np.asarray(edge.get("points", []), dtype=float)
                if pts.ndim == 2 and pts.shape[0] >= 2:
                    for i in range(len(pts) - 1):
                        segments.append(pts[i])
                        segments.append(pts[i + 1])
                    if bool(edge.get("is_closed", False)) and len(pts) > 2:
                        segments.append(pts[-1])
                        segments.append(pts[0])
        if not segments:
            return
        self._hover_edge_item = GLLinePlotItem(
            pos=np.asarray(segments, dtype=float),
            color=(1.0, 0.85, 0.2, 0.9),
            width=2.0,
            mode="lines",
        )
        self.addItem(self._hover_edge_item)

    def _set_hover_point(self, pick):
        if pick == self._hover_point_pick:
            return
        self._hover_point_pick = pick
        if self._hover_point_item is not None:
            self.removeItem(self._hover_point_item)
            self._hover_point_item = None
        if pick is None:
            return
        kind, value = pick
        pos = None
        if kind == "mesh" and self._last_mesh is not None:
            nodes, _ = self._last_mesh
            idx = int(value)
            if 0 <= idx < len(nodes):
                pos = np.asarray(nodes[idx], dtype=float)
        elif kind == "topology":
            point = self._cad_topology_vertex_lookup.get(int(value))
            if point is not None:
                pos = np.asarray(point, dtype=float)
        if pos is None or pos.size < 3:
            return
        size = (self._node_size_override or self._last_node_size or 4.0) * 1.5
        self._hover_point_item = GLScatterPlotItem(
            pos=np.asarray([pos[:3]], dtype=float),
            size=size,
            color=(1.0, 0.85, 0.2, 0.95),
        )
        self.addItem(self._hover_point_item)

    def _update_pick_hover(self, x, y):
        mode = self._selection_mode
        if mode == "auto":
            pick = self._pick_auto(x, y)
            if pick is None:
                self._set_hover_faces(set())
                self._set_hover_edge(None)
                self._set_hover_point(None)
                return
            kind, value = pick
            if kind == "face":
                face_idx = int(value)
                picked_faces = {face_idx}
                groups = getattr(self, "_face_group_ids", None)
                if groups is not None and isinstance(groups, np.ndarray) and 0 <= face_idx < len(groups):
                    group_id = int(groups[face_idx])
                    group_faces = np.where(groups == group_id)[0]
                    if len(group_faces) > 0:
                        picked_faces = {int(idx) for idx in group_faces}
                self._set_hover_faces(picked_faces)
                self._set_hover_edge(None)
                self._set_hover_point(None)
                return
            if kind == "edge":
                self._set_hover_faces(set())
                self._set_hover_point(None)
                self._set_hover_edge(value)
                return
            if kind == "point":
                self._set_hover_faces(set())
                self._set_hover_edge(None)
                self._set_hover_point(value)
                return
        if mode == "face":
            face_idx = self._pick_face(x, y)
            if face_idx is None:
                self._set_hover_faces(set())
                self._set_hover_edge(None)
                self._set_hover_point(None)
                return
            picked_faces = {int(face_idx)}
            groups = getattr(self, "_face_group_ids", None)
            if groups is not None and isinstance(groups, np.ndarray) and 0 <= face_idx < len(groups):
                group_id = int(groups[face_idx])
                group_faces = np.where(groups == group_id)[0]
                if len(group_faces) > 0:
                    picked_faces = {int(idx) for idx in group_faces}
            self._set_hover_faces(picked_faces)
            self._set_hover_edge(None)
            self._set_hover_point(None)
            return
        if mode == "edge":
            pick = self._pick_edge(x, y)
            self._set_hover_faces(set())
            self._set_hover_point(None)
            self._set_hover_edge(pick)
            return
        if mode == "point":
            pick = self._pick_point(x, y)
            self._set_hover_faces(set())
            self._set_hover_edge(None)
            self._set_hover_point(pick)
            return
        self._clear_pick_hover()

    def _update_face_highlight(self):
        if self._selection_highlight is not None:
            self.removeItem(self._selection_highlight)
            self._selection_highlight = None
        if not self.selected_faces or self._last_mesh is None or self._last_faces is None:
            return
        nodes, _ = self._last_mesh
        faces = self._last_faces
        indices = [idx for idx in self.selected_faces if 0 <= idx < len(faces)]
        if not indices:
            return
        highlight_faces = faces[indices]
        self._selection_highlight = GLMeshItem(
            vertexes=nodes,
            faces=highlight_faces,
            smooth=False,
            drawEdges=True,
            drawFaces=True,
            shader="shaded",
            edgeColor=(1.0, 0.5, 0.06, 1.0),
            color=(1.0, 0.68, 0.16, 0.32),
        )
        try:
            self._selection_highlight.setGLOptions("additive")
        except Exception:
            pass
        try:
            self._selection_highlight.setDepthValue(20)
        except Exception:
            pass
        self._selection_highlight.setVisible(self.show_mesh)
        self.addItem(self._selection_highlight)

    def _update_selected_nodes_marker(self):
        if self._selected_nodes_item is not None:
            self.removeItem(self._selected_nodes_item)
            self._selected_nodes_item = None
        mesh_points = []
        if self.selected_nodes and self._last_mesh is not None:
            nodes, _ = self._last_mesh
            indices = [idx for idx in self.selected_nodes if 0 <= idx < len(nodes)]
            if indices:
                mesh_points.append(np.asarray(nodes[indices], dtype=float))
        topo_points = []
        for vid in self.selected_topology_vertices:
            point = self._cad_topology_vertex_lookup.get(int(vid))
            if point is not None:
                topo_points.append(np.asarray(point, dtype=float))
        if not mesh_points and not topo_points:
            return
        pos_blocks = list(mesh_points)
        if topo_points:
            pos_blocks.append(np.asarray(topo_points, dtype=float))
        pos = np.vstack(pos_blocks)
        base_size = self._node_size_override or self._last_node_size or 4.0
        size = base_size * 1.8
        color = self._node_color_selected_override or (self._contrast_nodes or {}).get(
            "selected", (0.55, 0.57, 0.6, 0.95)
        )
        self._selected_nodes_item = GLScatterPlotItem(pos=pos, size=size, color=color)
        self.addItem(self._selected_nodes_item)
        self._selected_nodes_item.setVisible(self.show_nodes)

    def _update_edge_highlight(self):
        if self._selected_edges_item is not None:
            self.removeItem(self._selected_edges_item)
            self._selected_edges_item = None
        segments = []
        if self.selected_edges and self._last_mesh is not None:
            nodes, _ = self._last_mesh
            for i, j in self.selected_edges:
                if 0 <= i < len(nodes) and 0 <= j < len(nodes):
                    segments.append(nodes[i])
                    segments.append(nodes[j])
        for eid in self.selected_topology_edges:
            edge = self._cad_topology_edge_lookup.get(int(eid))
            if not edge:
                continue
            pts = np.asarray(edge.get("points", []), dtype=float)
            if pts.ndim != 2 or pts.shape[0] < 2:
                continue
            for i in range(len(pts) - 1):
                segments.append(pts[i])
                segments.append(pts[i + 1])
            if bool(edge.get("is_closed", False)) and len(pts) > 2:
                segments.append(pts[-1])
                segments.append(pts[0])
        if not segments:
            return
        pos = np.asarray(segments, dtype=float)
        self._selected_edges_item = GLLinePlotItem(
            pos=pos, color=(0.1, 0.9, 0.95, 0.98), width=3.0, mode="lines"
        )
        self.addItem(self._selected_edges_item)
        self._selected_edges_item.setVisible(self.show_mesh)

    def _pick_face(self, x, y):
        if self._last_mesh is None or self._last_faces is None:
            return None
        nodes, _ = self._last_mesh
        faces = self._last_faces
        if nodes.size == 0 or faces.size == 0:
            return None
        screen, depth, visible = self._project_points_to_screen(nodes)
        if screen is None or depth is None or visible is None:
            return None
        faces = np.asarray(faces, dtype=int)
        if faces.ndim != 2 or faces.shape[1] != 3:
            return None
        face_visible = visible[faces].all(axis=1)
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
            return int(inside_ids[int(np.argmin(inside_depth))])
        d01, _ = self._point_segment_distance_2d(float(x), float(y), p0, p1)
        d12, _ = self._point_segment_distance_2d(float(x), float(y), p1, p2)
        d20, _ = self._point_segment_distance_2d(float(x), float(y), p2, p0)
        dist = np.minimum(d01, np.minimum(d12, d20))
        threshold = max(10.0, float(self.get_default_node_size()) * 2.5)
        near = nondeg & np.isfinite(dist) & (dist <= threshold)
        if not np.any(near):
            return None
        near_ids = original_ids[near]
        near_dist = dist[near]
        near_depth = z_interp[near]
        order = np.lexsort((near_depth, near_dist))
        return int(near_ids[int(order[0])])

    def _closest_edge_to_ray(self, face_idx, origin, direction):
        if self._last_faces is None or self._last_mesh is None:
            return None
        faces = self._last_faces
        nodes, _ = self._last_mesh
        if face_idx < 0 or face_idx >= len(faces):
            return None
        tri = faces[face_idx]
        edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        best_edge = None
        best_dist = None
        for a, b in edges:
            p0 = nodes[a]
            p1 = nodes[b]
            dist = self._ray_segment_distance(origin, direction, p0, p1)
            if dist is None:
                continue
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_edge = (int(a), int(b))
        return best_edge

    @staticmethod
    def _ray_segment_distance(origin, direction, p0, p1):
        # Returns shortest distance between ray and segment.
        u = direction
        v = p1 - p0
        w0 = origin - p0
        a = float(np.dot(u, u))
        b = float(np.dot(u, v))
        c = float(np.dot(v, v))
        d = float(np.dot(u, w0))
        e = float(np.dot(v, w0))
        denom = a * c - b * b
        if abs(denom) < 1e-12:
            s = 0.0
        else:
            s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom if abs(denom) >= 1e-12 else e / c if c > 1e-12 else 0.0
        s = max(s, 0.0)
        t = max(0.0, min(1.0, t))
        closest_ray = origin + s * u
        closest_seg = p0 + t * v
        return float(np.linalg.norm(closest_ray - closest_seg))

    def _pick_topology_vertex(self, x, y):
        if self._cad_topology_vertices is None or self._cad_topology_vertices.size == 0:
            return None
        screen, _depth, visible = self._project_points_to_screen(self._cad_topology_vertices)
        if screen is None or visible is None or not np.any(visible):
            return None
        dx = screen[:, 0] - float(x)
        dy = screen[:, 1] - float(y)
        dist2 = dx * dx + dy * dy
        dist2[~visible] = np.inf
        idx = int(np.argmin(dist2))
        threshold = self._point_pick_threshold
        if threshold is None:
            threshold = max(10.0, float(self.get_default_node_size()) * 2.5)
        if dist2[idx] > threshold * threshold:
            return None
        return int(self._cad_topology_vertex_ids[idx])

    def _pick_topology_edge(self, x, y):
        if not self._cad_topology_edges:
            return None
        threshold = max(10.0, float(self.get_default_node_size()) * 2.5)
        best = None
        best_dist = None
        best_depth = None
        px = float(x)
        py = float(y)
        for edge in self._cad_topology_edges:
            pts = np.asarray(edge.get("points", []), dtype=float)
            if pts.ndim != 2 or pts.shape[0] < 2:
                continue
            screen, depth, visible = self._project_points_to_screen(pts)
            if screen is None or depth is None or visible is None:
                continue
            i0 = np.arange(len(pts) - 1, dtype=int)
            i1 = i0 + 1
            if bool(edge.get("is_closed", False)) and len(pts) > 2:
                i0 = np.concatenate((i0, np.array([len(pts) - 1], dtype=int)))
                i1 = np.concatenate((i1, np.array([0], dtype=int)))
            if i0.size == 0:
                continue
            a = screen[i0]
            b = screen[i1]
            seg_visible = visible[i0] | visible[i1]
            dist, t = self._point_segment_distance_2d(px, py, a, b)
            dist[~seg_visible] = np.inf
            j = int(np.argmin(dist))
            if not np.isfinite(dist[j]) or dist[j] > threshold:
                continue
            z0 = depth[i0[j]]
            z1 = depth[i1[j]]
            z = float(z0 + (z1 - z0) * t[j])
            d = float(dist[j])
            if (
                best is None
                or d < best_dist - 1e-9
                or (abs(d - best_dist) <= 1e-9 and z < best_depth)
            ):
                best = int(edge.get("id", -1))
                best_dist = d
                best_depth = z
        if best is None or best < 0:
            return None
        return best

    def _pick_point(self, x, y):
        has_topology_vertices = (
            self._cad_topology_vertices is not None and self._cad_topology_vertices.size > 0
        )
        topo_vid = self._pick_topology_vertex(x, y)
        if topo_vid is not None:
            return ("topology", int(topo_vid))
        if has_topology_vertices:
            return None
        if self._last_mesh is None:
            return None
        nodes, _ = self._last_mesh
        if nodes.size == 0:
            return None
        screen, _depth, visible = self._project_points_to_screen(nodes)
        if screen is None or visible is None:
            return None
        if not np.any(visible):
            return None
        dx = screen[:, 0] - float(x)
        dy = screen[:, 1] - float(y)
        dist2 = dx * dx + dy * dy
        dist2[~visible] = np.inf
        idx = int(np.argmin(dist2))
        threshold = self._point_pick_threshold
        if threshold is None:
            threshold = max(10.0, float(self.get_default_node_size()) * 2.5)
        if dist2[idx] > threshold * threshold:
            return None
        return ("mesh", idx)

    def _pick_edge(self, x, y):
        has_topology_edges = bool(self._cad_topology_edges)
        topo_edge = self._pick_topology_edge(x, y)
        if topo_edge is not None:
            return ("topology", int(topo_edge))
        if has_topology_edges:
            return None
        if self._last_mesh is None:
            return None
        nodes, _ = self._last_mesh
        if nodes.size == 0:
            return None
        if self._last_edges:
            edges = np.asarray(self._last_edges, dtype=int)
        elif self._last_faces is not None and len(self._last_faces) > 0:
            edge_set = set()
            for tri in np.asarray(self._last_faces, dtype=int):
                a, b, c = [int(v) for v in tri]
                edge_set.add((min(a, b), max(a, b)))
                edge_set.add((min(b, c), max(b, c)))
                edge_set.add((min(c, a), max(c, a)))
            if not edge_set:
                return None
            edges = np.asarray(sorted(edge_set), dtype=int)
        else:
            return None
        if edges.ndim != 2 or edges.shape[1] != 2:
            return None
        screen, depth, visible = self._project_points_to_screen(nodes)
        if screen is None or depth is None or visible is None:
            return None
        edge_visible = visible[edges].all(axis=1)
        if not np.any(edge_visible):
            return None
        a = screen[edges[:, 0]]
        b = screen[edges[:, 1]]
        dist, _ = self._point_segment_distance_2d(float(x), float(y), a, b)
        dist[~edge_visible] = np.inf
        threshold = max(10.0, float(self.get_default_node_size()) * 2.5)
        candidate = np.where(dist <= threshold)[0]
        if candidate.size == 0:
            return None
        edge_depth = (depth[edges[:, 0]] + depth[edges[:, 1]]) * 0.5
        order = np.lexsort((edge_depth[candidate], dist[candidate]))
        best = edges[int(candidate[int(order[0])])]
        return ("mesh", (int(min(best[0], best[1])), int(max(best[0], best[1]))))

    def _pick_auto(self, x, y):
        # Priority mirrors common CAD behavior: corner > edge > face.
        point_pick = self._pick_point(x, y)
        if point_pick is not None:
            return ("point", point_pick)
        edge_pick = self._pick_edge(x, y)
        if edge_pick is not None:
            return ("edge", edge_pick)
        face_idx = self._pick_face(x, y)
        if face_idx is not None:
            return ("face", int(face_idx))
        return None

    def _project_points_to_screen(self, nodes):
        try:
            pts = np.asarray(nodes, dtype=float)
        except Exception:
            return None, None, None
        if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 3:
            return None, None, None
        pts = pts[:, :3]
        w = max(1, self.width())
        h = max(1, self.height())
        viewport = self.getViewport() if hasattr(self, "getViewport") else QRect(0, 0, w, h)
        viewport_tuple = self._rect_to_tuple(viewport)
        region_tuple = viewport_tuple
        try:
            pm = self.projectionMatrix(region_tuple, viewport_tuple)
        except Exception:
            pm = self.projectionMatrix(region_tuple, viewport_tuple)
        m = pm * self.viewMatrix()
        # QMatrix4x4.copyDataTo() yields row-major data; C-order reshape matches QVector mapping.
        mat = np.asarray(m.copyDataTo(), dtype=float).reshape((4, 4), order="C")
        pts4 = np.concatenate((pts, np.ones((pts.shape[0], 1), dtype=float)), axis=1)
        clip = pts4 @ mat.T
        cw = clip[:, 3]
        valid = np.isfinite(cw) & (np.abs(cw) > 1e-12)
        ndc = np.full((pts.shape[0], 3), np.nan, dtype=float)
        ndc[valid] = clip[valid, :3] / cw[valid, None]
        x0, y0, vw, vh = viewport_tuple
        screen_x = x0 + (ndc[:, 0] + 1.0) * 0.5 * vw
        screen_y = y0 + (1.0 - ndc[:, 1]) * 0.5 * vh
        depth = ndc[:, 2]
        visible = (
            valid
            & np.isfinite(screen_x)
            & np.isfinite(screen_y)
            & np.isfinite(depth)
            & (depth >= -1.1)
            & (depth <= 1.1)
        )
        screen = np.column_stack((screen_x, screen_y))
        return screen, depth, visible

    @staticmethod
    def _point_segment_distance_2d(px, py, a, b):
        ab = b - a
        ap = np.array([px, py], dtype=float) - a
        denom = np.einsum("ij,ij->i", ab, ab)
        t = np.zeros(len(denom), dtype=float)
        mask = denom > 1e-12
        if np.any(mask):
            t[mask] = np.einsum("ij,ij->i", ap[mask], ab[mask]) / denom[mask]
        t = np.clip(t, 0.0, 1.0)
        closest = a + ab * t[:, None]
        diff = closest - np.array([px, py], dtype=float)
        dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        return dist, t

    def _update_selected_node_marker(self):
        if self._selected_node_pos is None:
            if self.node_item_selected is not None:
                self.removeItem(self.node_item_selected)
                self.node_item_selected = None
            return
        pos = np.array([self._selected_node_pos], dtype=float)
        size = (self._node_size_override or 4.0) * 1.6
        color = self._node_color_selected_override or (self._contrast_nodes or {}).get(
            "selected", (0.55, 0.57, 0.6, 0.95)
        )
        if self.node_item_selected is None:
            self.node_item_selected = GLScatterPlotItem(pos=pos, size=size, color=color)
            self.addItem(self.node_item_selected)
        else:
            self.node_item_selected.setData(pos=pos, size=size, color=color)
        self.node_item_selected.setVisible(self.show_nodes)

    @staticmethod
    def _sample_indices(ids, max_count):
        ids = np.asarray(ids, dtype=int)
        if ids.size == 0:
            return ids
        if len(ids) <= max_count:
            return ids
        return np.random.choice(ids, size=max_count, replace=False)

    def set_gizmo_mode(self, mode):
        if mode:
            self._gizmo_mode = str(mode).lower()
        self._rebuild_gizmo()

    def set_gizmo(self, position=None, enabled=True):
        self._gizmo_enabled = bool(enabled)
        if position is not None:
            self._gizmo_pos = QVector3D(float(position[0]), float(position[1]), float(position[2]))
            self._drag_plane_z = self._gizmo_pos.z()
        self._rebuild_gizmo()

    def _clear_gizmo(self):
        for item in self._gizmo_items:
            self.removeItem(item)
        self._gizmo_items = []

    def _rebuild_gizmo(self):
        self._clear_gizmo()
        if not self._gizmo_enabled:
            return
        self._gizmo_length = max(12.0, self._grid_spacing * 0.8)
        if self._gizmo_mode == "rotate":
            self._build_rotate_gizmo()
        elif self._gizmo_mode == "scale":
            self._build_scale_gizmo()
        else:
            self._build_translate_gizmo()

    def _build_translate_gizmo(self):
        x = self._gizmo_pos.x()
        y = self._gizmo_pos.y()
        z = self._gizmo_pos.z()
        length = self._gizmo_length
        self._gizmo_items.append(
            GLLinePlotItem(pos=np.array([[x, y, z], [x + length, y, z]]), color=(1, 0, 0, 0.95), width=2)
        )
        self._gizmo_items.append(
            GLLinePlotItem(pos=np.array([[x, y, z], [x, y + length, z]]), color=(0, 1, 0, 0.95), width=2)
        )
        self._gizmo_items.append(
            GLLinePlotItem(pos=np.array([[x, y, z], [x, y, z + length]]), color=(0, 0, 1, 0.95), width=2)
        )
        handle = np.array([[x + length, y, z], [x, y + length, z], [x, y, z + length]])
        colors = np.array([[1, 0, 0, 0.9], [0, 1, 0, 0.9], [0, 0, 1, 0.9]])
        self._gizmo_items.append(GLScatterPlotItem(pos=handle, size=6, color=colors))
        for item in self._gizmo_items:
            self.addItem(item)

    def _build_scale_gizmo(self):
        x = self._gizmo_pos.x()
        y = self._gizmo_pos.y()
        z = self._gizmo_pos.z()
        length = self._gizmo_length
        self._gizmo_items.append(
            GLLinePlotItem(pos=np.array([[x, y, z], [x + length, y, z]]), color=(1, 0, 0, 0.8), width=2)
        )
        self._gizmo_items.append(
            GLLinePlotItem(pos=np.array([[x, y, z], [x, y + length, z]]), color=(0, 1, 0, 0.8), width=2)
        )
        self._gizmo_items.append(
            GLLinePlotItem(pos=np.array([[x, y, z], [x, y, z + length]]), color=(0, 0, 1, 0.8), width=2)
        )
        handle = np.array([[x + length, y, z], [x, y + length, z], [x, y, z + length]])
        colors = np.array([[1, 0.4, 0.4, 0.9], [0.4, 1, 0.4, 0.9], [0.4, 0.4, 1, 0.9]])
        self._gizmo_items.append(GLScatterPlotItem(pos=handle, size=8, color=colors))
        for item in self._gizmo_items:
            self.addItem(item)

    def _build_rotate_gizmo(self):
        center = np.array([self._gizmo_pos.x(), self._gizmo_pos.y(), self._gizmo_pos.z()])
        radius = self._gizmo_length * 0.8
        steps = 64
        for axis, color in (
            ("x", (1, 0.0, 0.0, 0.9)),
            ("y", (0.0, 1.0, 0.0, 0.9)),
            ("z", (0.0, 0.0, 1.0, 0.9)),
        ):
            pts = []
            for i in range(steps + 1):
                theta = (i / steps) * math.tau
                if axis == "x":
                    pts.append(center + np.array([0, radius * math.cos(theta), radius * math.sin(theta)]))
                elif axis == "y":
                    pts.append(center + np.array([radius * math.cos(theta), 0, radius * math.sin(theta)]))
                else:
                    pts.append(center + np.array([radius * math.cos(theta), radius * math.sin(theta), 0]))
            self._gizmo_items.append(GLLinePlotItem(pos=np.array(pts), color=color, width=2))
        for item in self._gizmo_items:
            self.addItem(item)

    def _axis_hit_test(self, x, y):
        p1 = self._unproject(x, y, 0.0)
        p2 = self._unproject(x, y, 1.0)
        ray_dir = p2 - p1
        if ray_dir.lengthSquared() < 1e-9:
            return None
        ray_dir = ray_dir.normalized()
        origin = self._gizmo_pos
        length = self._gizmo_length
        threshold = max(2.0, self._grid_spacing * 0.15)
        axes = {
            "x": QVector3D(1, 0, 0),
            "y": QVector3D(0, 1, 0),
            "z": QVector3D(0, 0, 1),
        }
        best = None
        best_dist = 1e9
        best_s = None
        for name, axis in axes.items():
            s, t, dist = self._closest_line_params(origin, axis, p1, ray_dir)
            if dist is None or t is None or t < 0:
                continue
            if s < -length * 0.1 or s > length * 1.2:
                continue
            if dist < threshold and dist < best_dist:
                best = name
                best_dist = dist
                best_s = s
        if best is None:
            return None
        return best, best_s

    def _closest_line_params(self, p1, d1, p2, d2):
        u = np.array([d1.x(), d1.y(), d1.z()], dtype=float)
        v = np.array([d2.x(), d2.y(), d2.z()], dtype=float)
        w0 = np.array([p1.x() - p2.x(), p1.y() - p2.y(), p1.z() - p2.z()], dtype=float)
        a = float(np.dot(u, u))
        b = float(np.dot(u, v))
        c = float(np.dot(v, v))
        d = float(np.dot(u, w0))
        e = float(np.dot(v, w0))
        denom = a * c - b * b
        if abs(denom) < 1e-9:
            return None, None, None
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom
        c1 = w0 + s * u - t * v
        dist = float(np.linalg.norm(c1))
        return s, t, dist

    def _ray_plane_intersect_general(self, p1, p2, plane_point, plane_normal):
        ray_dir = p2 - p1
        denom = QVector3D.dotProduct(ray_dir, plane_normal)
        if abs(denom) < 1e-9:
            return None
        t = QVector3D.dotProduct(plane_point - p1, plane_normal) / denom
        if t < 0:
            return None
        return p1 + ray_dir * t

    def _rotation_hit_test(self, x, y):
        p1 = self._unproject(x, y, 0.0)
        p2 = self._unproject(x, y, 1.0)
        center = self._gizmo_pos
        radius = self._gizmo_length * 0.8
        tol = max(2.0, self._grid_spacing * 0.15)
        axes = {
            "x": QVector3D(1, 0, 0),
            "y": QVector3D(0, 1, 0),
            "z": QVector3D(0, 0, 1),
        }
        best = None
        best_vec = None
        for name, axis in axes.items():
            hit = self._ray_plane_intersect_general(p1, p2, center, axis)
            if hit is None:
                continue
            vec = hit - center
            dist = vec.length()
            if abs(dist - radius) <= tol:
                best = name
                if dist > 1e-6:
                    best_vec = vec / dist
                break
        if best is None or best_vec is None:
            return None
        return best, best_vec

    def _unproject(self, x, y, z):
        w = max(1, self.width())
        h = max(1, self.height())
        ndc_x = (2.0 * x) / w - 1.0
        ndc_y = 1.0 - (2.0 * y) / h
        ndc_z = 2.0 * z - 1.0
        # projectionMatrix() expects region/viewport in widget pixel space.
        # Using world-space bounds (viewRect) causes ray picking to miss.
        viewport = self.getViewport() if hasattr(self, "getViewport") else QRect(0, 0, w, h)
        viewport_tuple = self._rect_to_tuple(viewport)
        region_tuple = viewport_tuple
        try:
            pm = self.projectionMatrix(region_tuple, viewport_tuple)
        except Exception:
            pm = self.projectionMatrix(region_tuple, viewport_tuple)
        m = pm * self.viewMatrix()
        inv, ok = m.inverted()
        if not ok:
            return QVector3D()
        v4 = inv.map(QVector4D(ndc_x, ndc_y, ndc_z, 1.0))
        if abs(v4.w()) > 1e-9:
            return QVector3D(v4.x() / v4.w(), v4.y() / v4.w(), v4.z() / v4.w())
        return QVector3D(v4.x(), v4.y(), v4.z())

    @staticmethod
    def _rect_to_tuple(rect):
        if rect is None:
            return (0.0, 0.0, 0.0, 0.0)
        if isinstance(rect, (tuple, list)) and len(rect) == 4:
            return tuple(float(v) for v in rect)
        try:
            return (float(rect.x()), float(rect.y()), float(rect.width()), float(rect.height()))
        except Exception:
            return (0.0, 0.0, 0.0, 0.0)

    def _ray_plane_intersect(self, p1, p2, z_plane):
        dz = p2.z() - p1.z()
        if abs(dz) < 1e-9:
            return None
        t = (z_plane - p1.z()) / dz
        if t < 0:
            return None
        x = p1.x() + (p2.x() - p1.x()) * t
        y = p1.y() + (p2.y() - p1.y()) * t
        return QVector3D(x, y, z_plane)

    def request_point_pick(self, callback, z_plane=0.0):
        self._placement_callback = callback
        self._placement_active = True
        self._placement_z = float(z_plane)

    def cancel_point_pick(self):
        self._placement_active = False
        self._placement_callback = None

    def _notify_status(self, message, timeout_ms=2500):
        try:
            window = self.window()
            if window and hasattr(window, "statusBar"):
                window.statusBar().showMessage(message, timeout_ms)
            if window and hasattr(window, "_update_interaction_hints"):
                window._update_interaction_hints()
        except Exception:
            pass

    @staticmethod
    def _norm_angle_deg(angle):
        a = float(angle) % 360.0
        if a > 180.0:
            a -= 360.0
        return a

    def _orientation_overlay_view_presets(self):
        return {
            "front": ("Front", 0.0, 0.0),
            "back": ("Back", 180.0, 0.0),
            "left": ("Left", -90.0, 0.0),
            "right": ("Right", 90.0, 0.0),
            "top": ("Top", 0.0, 90.0),
            "bottom": ("Bottom", 0.0, -90.0),
            "iso": ("Iso", 45.0, 35.264),
        }

    def set_orientation_overlay_enabled(self, enabled):
        enabled = bool(enabled)
        if enabled == self._orientation_overlay_enabled:
            return
        self._orientation_overlay_enabled = enabled
        if not enabled:
            self._orientation_overlay_hover_key = None
            self._orientation_overlay_hitboxes = {}
        self.update()

    def _orientation_overlay_layout(self):
        if not self._orientation_overlay_enabled:
            self._orientation_overlay_hitboxes = {}
            return None, {}
        margin = 10.0
        pad = 10.0
        gap = 6.0
        pill_w = 52.0
        pill_h = 22.0
        title_h = 18.0
        cube_w = 86.0
        cube_h = 72.0
        panel_w = pad * 2 + cube_w + pill_w + gap
        panel_h = pad * 2 + title_h + cube_h + pill_h + gap
        x = max(6.0, float(self.width()) - panel_w - margin)
        y = margin
        panel_rect = QRectF(x, y, panel_w, panel_h)
        gx = x + pad + 8.0
        gy = y + pad + title_h + 12.0
        a = 20.0
        b = 12.0
        h = 28.0
        top_poly = QPolygonF(
            [
                QPointF(gx + a, gy),
                QPointF(gx + 2 * a, gy - b),
                QPointF(gx + 3 * a, gy),
                QPointF(gx + 2 * a, gy + b),
            ]
        )
        front_poly = QPolygonF(
            [
                QPointF(gx + a, gy),
                QPointF(gx + 2 * a, gy + b),
                QPointF(gx + 2 * a, gy + b + h),
                QPointF(gx + a, gy + h),
            ]
        )
        right_poly = QPolygonF(
            [
                QPointF(gx + 3 * a, gy),
                QPointF(gx + 3 * a, gy + h),
                QPointF(gx + 2 * a, gy + b + h),
                QPointF(gx + 2 * a, gy + b),
            ]
        )
        items = {
            "top": {"shape": top_poly, "kind": "poly", "label": "Top"},
            "front": {"shape": front_poly, "kind": "poly", "label": "Front"},
            "right": {"shape": right_poly, "kind": "poly", "label": "Right"},
            "back": {
                "shape": QRectF(gx + a + 4.0, y + pad + title_h, pill_w, pill_h),
                "kind": "rect",
                "label": "Back",
            },
            "left": {
                "shape": QRectF(x + pad, gy + h * 0.45, pill_w, pill_h),
                "kind": "rect",
                "label": "Left",
            },
            "bottom": {
                "shape": QRectF(gx + a + 4.0, gy + h + b + gap + 2.0, pill_w, pill_h),
                "kind": "rect",
                "label": "Bottom",
            },
            "iso": {
                "shape": QRectF(x + panel_w - pad - pill_w, gy + h * 0.45, pill_w, pill_h),
                "kind": "rect",
                "label": "ISO",
            },
        }
        self._orientation_overlay_hitboxes = items
        return panel_rect, items

    def _orientation_overlay_hit_key(self, x, y, items):
        if not items:
            return None
        pt = QPointF(float(x), float(y))
        for key, item in items.items():
            shape = item.get("shape")
            kind = item.get("kind")
            if kind == "poly" and hasattr(shape, "containsPoint") and shape.containsPoint(pt, Qt.OddEvenFill):
                return key
            if kind == "rect" and hasattr(shape, "contains") and shape.contains(float(x), float(y)):
                return key
        return None

    def _orientation_overlay_active_key(self):
        try:
            az = self._norm_angle_deg(self.opts.get("azimuth", 0.0))
            el = float(self.opts.get("elevation", 0.0))
        except Exception:
            return None
        best_key = None
        best_score = None
        for key, (_label, paz, pel) in self._orientation_overlay_view_presets().items():
            da = abs(self._norm_angle_deg(az - paz))
            de = abs(float(el) - float(pel))
            score = da + de * 1.4
            if best_score is None or score < best_score:
                best_score = score
                best_key = key
        if best_score is not None and best_score <= 18.0:
            return best_key
        return None

    def _show_orientation_overlay_menu(self, global_pos=None):
        menu = QMenu(self)
        menu.addAction("Fit View", self.fit_view)
        menu.addAction("Center Origin", self.center_origin)
        menu.addSeparator()
        corners = menu.addMenu("Corners")
        iso = 35.264
        for label, az, el in (
            ("Top Front Right", 45.0, iso),
            ("Top Front Left", -45.0, iso),
            ("Top Back Right", 135.0, iso),
            ("Top Back Left", -135.0, iso),
            ("Bottom Front Right", 45.0, -iso),
            ("Bottom Front Left", -45.0, -iso),
            ("Bottom Back Right", 135.0, -iso),
            ("Bottom Back Left", -135.0, -iso),
        ):
            corners.addAction(label, lambda a=az, e=el: self.set_view(a, e))
        edges = menu.addMenu("Edges")
        for label, az, el in (
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
        ):
            edges.addAction(label, lambda a=az, e=el: self.set_view(a, e))
        menu.addSeparator()
        menu.addAction("Random View", self.set_random_view)
        if global_pos is None:
            global_pos = self.mapToGlobal(self.rect().topRight())
        menu.exec(global_pos)

    def _handle_orientation_overlay_click(self, event):
        if not self._orientation_overlay_enabled or event.button() != Qt.LeftButton:
            return False
        _panel_rect, items = self._orientation_overlay_layout()
        if not items:
            return False
        try:
            posf = event.position()
            x = float(posf.x())
            y = float(posf.y())
        except Exception:
            return False
        clicked_key = self._orientation_overlay_hit_key(x, y, items)
        if clicked_key is None:
            return False
        presets = self._orientation_overlay_view_presets()
        if clicked_key in presets:
            label, az, el = presets[clicked_key]
            self.set_view(az, el)
            self._notify_status(f"ViewCube: {label}")
            self.update()
            event.accept()
            return True
        return False

    def _update_orientation_overlay_hover(self, event):
        if not self._orientation_overlay_enabled:
            if self._orientation_overlay_hover_key is not None:
                self._orientation_overlay_hover_key = None
                self.update()
            return
        _panel_rect, items = self._orientation_overlay_layout()
        hover_key = None
        if items and event is not None:
            try:
                posf = event.position()
                x = float(posf.x())
                y = float(posf.y())
                hover_key = self._orientation_overlay_hit_key(x, y, items)
            except Exception:
                hover_key = None
        if hover_key != self._orientation_overlay_hover_key:
            self._orientation_overlay_hover_key = hover_key
            self.update()

    def _paint_orientation_overlay(self):
        if not self._orientation_overlay_enabled:
            return
        panel_rect, items = self._orientation_overlay_layout()
        if panel_rect is None or not items:
            return
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setRenderHint(QPainter.TextAntialiasing, True)
            painter.setPen(QPen(QColor(24, 30, 36, 125), 1.0))
            painter.setBrush(QBrush(QColor(248, 251, 255, 226)))
            painter.drawRoundedRect(panel_rect, 10, 10)
            title_rect = QRectF(panel_rect.x() + 10, panel_rect.y() + 4, panel_rect.width() - 20, 15)
            painter.setPen(QColor(55, 65, 75))
            f = painter.font()
            f.setPointSize(max(7, f.pointSize() - 1))
            f.setBold(True)
            painter.setFont(f)
            painter.drawText(title_rect, Qt.AlignLeft | Qt.AlignVCenter, "VIEWCUBE")

            active_key = self._orientation_overlay_active_key()
            for key, item in items.items():
                is_active = key == active_key
                is_hover = key == self._orientation_overlay_hover_key
                bg = QColor(255, 255, 255, 220)
                edge = QColor(120, 130, 145, 210)
                text = QColor(45, 52, 60)
                if is_active:
                    bg = QColor(255, 196, 112, 235)
                    edge = QColor(224, 126, 18, 240)
                    text = QColor(80, 40, 5)
                elif is_hover:
                    bg = QColor(227, 238, 252, 235)
                    edge = QColor(92, 122, 164, 220)
                painter.setPen(QPen(edge, 1.2))
                painter.setBrush(QBrush(bg))
                shape = item.get("shape")
                if item.get("kind") == "poly":
                    painter.drawPolygon(shape)
                    bounds = shape.boundingRect()
                else:
                    painter.drawRoundedRect(shape, 5, 5)
                    bounds = shape
                painter.setPen(text)
                tf = painter.font()
                tf.setBold(is_active or key in {"front", "top", "right", "iso"})
                tf.setPointSize(8)
                painter.setFont(tf)
                painter.drawText(bounds, Qt.AlignCenter, item.get("label", key))
        finally:
            painter.end()

    def paintGL(self, *args, **kwargs):
        super().paintGL(*args, **kwargs)
        try:
            self._paint_orientation_overlay()
        except Exception:
            pass

    def mousePressEvent(self, event):
        if self._handle_orientation_overlay_click(event):
            return
        if self._placement_active:
            if event.button() == Qt.RightButton:
                self.cancel_point_pick()
                event.accept()
                return
            if event.button() == Qt.LeftButton:
                x = event.position().x()
                y = event.position().y()
                p1 = self._unproject(x, y, 0.0)
                p2 = self._unproject(x, y, 1.0)
                hit = self._ray_plane_intersect(p1, p2, self._placement_z)
                if hit is not None:
                    xw, yw, zw = hit.x(), hit.y(), hit.z()
                    if self._snap_enabled and self._grid_spacing > 0:
                        xw = round(xw / self._grid_spacing) * self._grid_spacing
                        yw = round(yw / self._grid_spacing) * self._grid_spacing
                    callback = self._placement_callback
                    self.cancel_point_pick()
                    if callback:
                        try:
                            callback(xw, yw, zw)
                        except Exception:
                            pass
                    event.accept()
                    return
        if self._view_navigation_enabled and event.button() == Qt.MiddleButton:
            self._begin_view_pan(event.position())
            event.accept()
            return
        if self._view_navigation_enabled and event.button() == Qt.LeftButton:
            if event.modifiers() & Qt.AltModifier:
                self._begin_view_rotate(event.position())
                event.accept()
                return
        if self._view_navigation_enabled and event.button() == Qt.RightButton:
            if self._is_synthesized_pointer_event(event):
                event.ignore()
                return
            self._begin_view_rotate(event.position())
            event.accept()
            return
        if event.button() == Qt.LeftButton and self._selection_mode in {"auto", "face", "edge", "point"}:
            self._begin_selection_box(event)
        if event.button() == Qt.LeftButton and self._selection_mode == "auto":
            pick = self._pick_auto(event.position().x(), event.position().y())
            if pick is None:
                self._notify_status("No entity under cursor.")
                event.accept()
                return
            pick_type, pick_value = pick
            multi = bool(event.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier))
            if pick_type == "face":
                face_idx = int(pick_value)
                picked_faces = {face_idx}
                groups = getattr(self, "_face_group_ids", None)
                if groups is not None and isinstance(groups, np.ndarray) and 0 <= face_idx < len(groups):
                    group_id = int(groups[face_idx])
                    group_faces = np.where(groups == group_id)[0]
                    if len(group_faces) > 0:
                        picked_faces = {int(idx) for idx in group_faces}
                if multi:
                    if picked_faces.issubset(self.selected_faces):
                        self.selected_faces.difference_update(picked_faces)
                    else:
                        self.selected_faces.update(picked_faces)
                else:
                    self.selected_nodes = set()
                    self.selected_topology_vertices = set()
                    self.selected_edges = set()
                    self.selected_topology_edges = set()
                    self.selected_faces = set(picked_faces)
                self._update_face_highlight()
                self._update_selected_nodes_marker()
                self._update_edge_highlight()
                self._emit_selection_changed()
                self._notify_status(
                    f"Face selected: {len(picked_faces)} tri(s) ({'add' if multi else 'set'})"
                )
                event.accept()
                return
            if pick_type == "edge":
                pick_kind, pick_edge = pick_value
                if pick_kind == "topology":
                    eid = int(pick_edge)
                    if multi:
                        if eid in self.selected_topology_edges:
                            self.selected_topology_edges.remove(eid)
                        else:
                            self.selected_topology_edges.add(eid)
                    else:
                        self.selected_faces = set()
                        self.selected_nodes = set()
                        self.selected_topology_vertices = set()
                        self.selected_edges = set()
                        self.selected_topology_edges = {eid}
                    label = f"Edge selected: CAD#{eid}"
                else:
                    edge = pick_edge
                    if multi:
                        if edge in self.selected_edges:
                            self.selected_edges.remove(edge)
                        else:
                            self.selected_edges.add(edge)
                    else:
                        self.selected_faces = set()
                        self.selected_nodes = set()
                        self.selected_topology_vertices = set()
                        self.selected_topology_edges = set()
                        self.selected_edges = {edge}
                    label = f"Edge selected: {edge[0]}-{edge[1]}"
                self._update_edge_highlight()
                self._update_face_highlight()
                self._update_selected_nodes_marker()
                self._emit_selection_changed()
                self._notify_status(f"{label} ({'add' if multi else 'set'})")
                event.accept()
                return
            if pick_type == "point":
                pick_kind, pick_node = pick_value
                if pick_kind == "topology":
                    vid = int(pick_node)
                    if multi:
                        if vid in self.selected_topology_vertices:
                            self.selected_topology_vertices.remove(vid)
                        else:
                            self.selected_topology_vertices.add(vid)
                    else:
                        self.selected_faces = set()
                        self.selected_edges = set()
                        self.selected_topology_edges = set()
                        self.selected_nodes = set()
                        self.selected_topology_vertices = {vid}
                    label = f"Vertex selected: {vid}"
                else:
                    node_idx = int(pick_node)
                    if multi:
                        if node_idx in self.selected_nodes:
                            self.selected_nodes.remove(node_idx)
                        else:
                            self.selected_nodes.add(node_idx)
                    else:
                        self.selected_faces = set()
                        self.selected_edges = set()
                        self.selected_topology_edges = set()
                        self.selected_topology_vertices = set()
                        self.selected_nodes = {node_idx}
                    label = f"Point selected: {node_idx}"
                self._update_selected_nodes_marker()
                self._update_face_highlight()
                self._update_edge_highlight()
                self._emit_selection_changed()
                self._notify_status(f"{label} ({'add' if multi else 'set'})")
                event.accept()
                return
            self._notify_status("No entity under cursor.")
            event.accept()
            return
        if event.button() == Qt.LeftButton and self._selection_mode == "face":
            face_idx = self._pick_face(event.position().x(), event.position().y())
            if face_idx is not None:
                picked_faces = {int(face_idx)}
                groups = getattr(self, "_face_group_ids", None)
                if (
                    groups is not None
                    and isinstance(groups, np.ndarray)
                    and 0 <= face_idx < len(groups)
                ):
                    group_id = int(groups[face_idx])
                    group_faces = np.where(groups == group_id)[0]
                    if len(group_faces) > 0:
                        picked_faces = {int(idx) for idx in group_faces}
                multi = bool(event.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier))
                if multi:
                    if picked_faces.issubset(self.selected_faces):
                        self.selected_faces.difference_update(picked_faces)
                    else:
                        self.selected_faces.update(picked_faces)
                else:
                    self.selected_faces = set(picked_faces)
                self._update_face_highlight()
                self._emit_selection_changed()
                self._notify_status(
                    f"Face selected: {len(picked_faces)} tri(s) ({'add' if multi else 'set'})"
                )
                event.accept()
                return
            self._notify_status("No face under cursor.")
        if event.button() == Qt.LeftButton and self._selection_mode == "point":
            pick = self._pick_point(event.position().x(), event.position().y())
            if pick is not None:
                pick_kind, pick_value = pick
                multi = bool(event.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier))
                if pick_kind == "topology":
                    vid = int(pick_value)
                    if multi:
                        if vid in self.selected_topology_vertices:
                            self.selected_topology_vertices.remove(vid)
                        else:
                            self.selected_topology_vertices.add(vid)
                    else:
                        self.selected_nodes = set()
                        self.selected_topology_vertices = {vid}
                    label = f"Vertex selected: {vid}"
                else:
                    node_idx = int(pick_value)
                    if multi:
                        if node_idx in self.selected_nodes:
                            self.selected_nodes.remove(node_idx)
                        else:
                            self.selected_nodes.add(node_idx)
                    else:
                        self.selected_topology_vertices = set()
                        self.selected_nodes = {node_idx}
                    label = f"Point selected: {node_idx}"
                self._update_selected_nodes_marker()
                self._emit_selection_changed()
                self._notify_status(f"{label} ({'add' if multi else 'set'})")
                event.accept()
                return
            self._notify_status("No point under cursor.")
        if event.button() == Qt.LeftButton and self._selection_mode == "edge":
            pick = self._pick_edge(event.position().x(), event.position().y())
            if pick is not None:
                pick_kind, pick_value = pick
                multi = bool(event.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier))
                if pick_kind == "topology":
                    eid = int(pick_value)
                    if multi:
                        if eid in self.selected_topology_edges:
                            self.selected_topology_edges.remove(eid)
                        else:
                            self.selected_topology_edges.add(eid)
                    else:
                        self.selected_edges = set()
                        self.selected_topology_edges = {eid}
                    label = f"Edge selected: CAD#{eid}"
                else:
                    edge = pick_value
                    if multi:
                        if edge in self.selected_edges:
                            self.selected_edges.remove(edge)
                        else:
                            self.selected_edges.add(edge)
                    else:
                        self.selected_topology_edges = set()
                        self.selected_edges = {edge}
                    label = f"Edge selected: {edge[0]}-{edge[1]}"
                self._update_edge_highlight()
                self._emit_selection_changed()
                self._notify_status(f"{label} ({'add' if multi else 'set'})")
                event.accept()
                return
            self._notify_status("No edge under cursor.")
        if self._gizmo_enabled and event.button() == Qt.LeftButton and self._selection_mode == "none":
            x = event.position().x()
            y = event.position().y()
            if self._gizmo_mode == "rotate":
                hit = self._rotation_hit_test(x, y)
                if hit is not None:
                    axis, vec = hit
                    self._dragging_gizmo = True
                    self._drag_mode = "rotate"
                    self._active_axis = axis
                    self._drag_last_vec = vec
                    self.gizmoDragStarted.emit(self._drag_mode)
                    event.accept()
                    return
            if self._gizmo_mode in ("translate", "scale"):
                axis_hit = self._axis_hit_test(x, y)
                if axis_hit is not None:
                    axis, s = axis_hit
                    self._dragging_gizmo = True
                    self._active_axis = axis
                    self._drag_start_axis_param = s
                    self._drag_start_pos = QVector3D(
                        self._gizmo_pos.x(), self._gizmo_pos.y(), self._gizmo_pos.z()
                    )
                    self._drag_mode = "scale" if self._gizmo_mode == "scale" else "translate"
                    self.gizmoDragStarted.emit(self._drag_mode)
                    event.accept()
                    return
            if self._gizmo_mode == "translate" and (event.modifiers() & Qt.ShiftModifier):
                p1 = self._unproject(x, y, 0.0)
                p2 = self._unproject(x, y, 1.0)
                hit = self._ray_plane_intersect(p1, p2, self._drag_plane_z)
                if hit is not None:
                    self._dragging_gizmo = True
                    self._drag_mode = "translate_plane"
                    self.gizmoDragStarted.emit(self._drag_mode)
                    event.accept()
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self._update_orientation_overlay_hover(event)
        if self._nav_panning and self._view_navigation_enabled:
            if not (event.buttons() & Qt.MiddleButton):
                self._nav_panning = False
                self._nav_last_pos = None
            else:
                if self._nav_last_pos is None:
                    self._nav_last_pos = event.position()
                diff = event.position() - self._nav_last_pos
                self._nav_last_pos = event.position()
                self.pan(float(diff.x()), float(diff.y()), 0.0, relative="view")
                event.accept()
                return
        if self._nav_rotating and self._view_navigation_enabled:
            rotate_buttons = event.buttons() & Qt.RightButton
            alt_left_rotate = bool(event.buttons() & Qt.LeftButton) and bool(event.modifiers() & Qt.AltModifier)
            if not rotate_buttons and not alt_left_rotate:
                self._nav_rotating = False
                self._nav_last_pos = None
                self._nav_right_dragged = False
            else:
                if self._nav_last_pos is None:
                    self._nav_last_pos = event.position()
                diff = event.position() - self._nav_last_pos
                self._nav_last_pos = event.position()
                if abs(float(diff.x())) > 1.0 or abs(float(diff.y())) > 1.0:
                    self._nav_right_dragged = True
                self.orbit(float(-diff.x()), float(diff.y()))
                event.accept()
                return
        if self._selection_box_active and (event.buttons() & Qt.LeftButton):
            self._update_selection_box(event)
            event.accept()
            return
        if (
            self._selection_mode in {"auto", "face", "edge", "point"}
            and not self._dragging_gizmo
        ):
            self._update_pick_hover(event.position().x(), event.position().y())
        elif self._selection_mode not in {"auto", "face", "edge", "point"}:
            self._clear_pick_hover()
        if self._hover_enabled and self.show_nodes and self._hover_nodes is not None:
            self._hover_pos = event.position()
            if not self._hover_timer.isActive():
                self._hover_timer.start(80)
        if self._dragging_gizmo:
            x = event.position().x()
            y = event.position().y()
            if self._drag_mode == "translate_plane":
                p1 = self._unproject(x, y, 0.0)
                p2 = self._unproject(x, y, 1.0)
                hit = self._ray_plane_intersect(p1, p2, self._drag_plane_z)
                if hit is not None:
                    xw, yw, zw = hit.x(), hit.y(), hit.z()
                    if self._snap_enabled and self._grid_spacing > 0:
                        xw = round(xw / self._grid_spacing) * self._grid_spacing
                        yw = round(yw / self._grid_spacing) * self._grid_spacing
                    self.set_gizmo((xw, yw, zw), enabled=True)
                    self.gizmoMoved.emit(xw, yw, zw)
                event.accept()
                return
            if self._drag_mode == "translate" and self._active_axis is not None:
                p1 = self._unproject(x, y, 0.0)
                p2 = self._unproject(x, y, 1.0)
                ray_dir = p2 - p1
                if ray_dir.lengthSquared() > 1e-9:
                    ray_dir = ray_dir.normalized()
                    axis_dir = {
                        "x": QVector3D(1, 0, 0),
                        "y": QVector3D(0, 1, 0),
                        "z": QVector3D(0, 0, 1),
                    }[self._active_axis]
                    s, t, _ = self._closest_line_params(self._drag_start_pos, axis_dir, p1, ray_dir)
                    if s is not None:
                        delta = s - (self._drag_start_axis_param or 0.0)
                        pos = self._drag_start_pos + axis_dir * delta
                        if self._snap_enabled and self._grid_spacing > 0:
                            if self._active_axis == "x":
                                pos.setX(round(pos.x() / self._grid_spacing) * self._grid_spacing)
                            elif self._active_axis == "y":
                                pos.setY(round(pos.y() / self._grid_spacing) * self._grid_spacing)
                            else:
                                pos.setZ(round(pos.z() / self._grid_spacing) * self._grid_spacing)
                        self.set_gizmo((pos.x(), pos.y(), pos.z()), enabled=True)
                        self.gizmoMoved.emit(pos.x(), pos.y(), pos.z())
                event.accept()
                return
            if self._drag_mode == "rotate" and self._active_axis is not None:
                axis_vec = {
                    "x": QVector3D(1, 0, 0),
                    "y": QVector3D(0, 1, 0),
                    "z": QVector3D(0, 0, 1),
                }[self._active_axis]
                p1 = self._unproject(x, y, 0.0)
                p2 = self._unproject(x, y, 1.0)
                hit = self._ray_plane_intersect_general(p1, p2, self._gizmo_pos, axis_vec)
                if hit is not None:
                    vec = hit - self._gizmo_pos
                    if vec.lengthSquared() > 1e-9 and self._drag_last_vec is not None:
                        vec = vec.normalized()
                        last = self._drag_last_vec
                        cross = QVector3D.crossProduct(last, vec)
                        sign = 1.0 if QVector3D.dotProduct(cross, axis_vec) >= 0 else -1.0
                        dot = max(-1.0, min(1.0, QVector3D.dotProduct(last, vec)))
                        angle = math.degrees(math.acos(dot)) * sign
                        self._drag_last_vec = vec
                        if abs(angle) > 1e-6:
                            dx = dy = dz = 0.0
                            if self._active_axis == "x":
                                dx = angle
                            elif self._active_axis == "y":
                                dy = angle
                            else:
                                dz = angle
                            self.gizmoRotated.emit(dx, dy, dz)
                event.accept()
                return
            if self._drag_mode == "scale" and self._active_axis is not None:
                p1 = self._unproject(x, y, 0.0)
                p2 = self._unproject(x, y, 1.0)
                ray_dir = p2 - p1
                if ray_dir.lengthSquared() > 1e-9:
                    ray_dir = ray_dir.normalized()
                    axis_dir = {
                        "x": QVector3D(1, 0, 0),
                        "y": QVector3D(0, 1, 0),
                        "z": QVector3D(0, 0, 1),
                    }[self._active_axis]
                    s, t, _ = self._closest_line_params(self._gizmo_pos, axis_dir, p1, ray_dir)
                    if s is not None:
                        if self._drag_start_axis_param is None:
                            self._drag_start_axis_param = s
                        delta = s - self._drag_start_axis_param
                        scale_ref = max(5.0, self._gizmo_length)
                        factor = 1.0 + (delta / scale_ref)
                        factor = max(0.1, min(10.0, factor))
                        self._drag_start_axis_param = s
                        sx = sy = sz = 1.0
                        if self._active_axis == "x":
                            sx = factor
                        elif self._active_axis == "y":
                            sy = factor
                        else:
                            sz = factor
                        self.gizmoScaled.emit(sx, sy, sz)
                event.accept()
                return
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton and self._nav_panning:
            self._nav_panning = False
            self._nav_last_pos = None
            event.accept()
            return
        if event.button() == Qt.LeftButton and self._nav_rotating:
            self._nav_rotating = False
            self._nav_last_pos = None
            self._nav_right_dragged = False
            event.accept()
            return
        if event.button() == Qt.RightButton and self._nav_rotating:
            self._nav_rotating = False
            self._nav_last_pos = None
            if self._nav_right_dragged:
                self._suppress_next_context_menu = True
            self._nav_right_dragged = False
            event.accept()
            return
        if event.button() == Qt.LeftButton and self._finish_selection_box(event):
            event.accept()
            return
        if self._dragging_gizmo and event.button() == Qt.LeftButton:
            mode = self._drag_mode or ""
            self._dragging_gizmo = False
            self._active_axis = None
            self._drag_start_axis_param = None
            self._drag_start_pos = None
            self._drag_last_vec = None
            self._drag_mode = None
            self.gizmoDragFinished.emit(mode)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        pixel_delta = event.pixelDelta() if hasattr(event, "pixelDelta") else None
        if pixel_delta is not None and not pixel_delta.isNull():
            if self._view_navigation_enabled and not (event.modifiers() & Qt.ControlModifier):
                self.pan(
                    float(pixel_delta.x()),
                    float(pixel_delta.y()),
                    0.0,
                    relative="view",
                )
            else:
                self._apply_zoom_delta(pixel_delta.y())
            event.accept()
            return
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        self._apply_zoom_delta(delta)
        event.accept()

    def _handle_hover_timer(self):
        if self._hover_nodes is None or not self._hover_enabled:
            return
        nodes = self._hover_nodes
        if not isinstance(nodes, np.ndarray) or nodes.size == 0:
            return
        if nodes.ndim != 2 or nodes.shape[1] != 3:
            return
        if self._hover_pos is None:
            return
        x = self._hover_pos.x()
        y = self._hover_pos.y()
        p1 = self._unproject(x, y, 0.0)
        p2 = self._unproject(x, y, 1.0)
        ray_dir = p2 - p1
        if ray_dir.lengthSquared() < 1e-9:
            return
        ray_dir = ray_dir.normalized()
        d = np.array([ray_dir.x(), ray_dir.y(), ray_dir.z()], dtype=float)
        origin = np.array([p1.x(), p1.y(), p1.z()], dtype=float)
        v = nodes - origin
        t = np.dot(v, d)
        t_mask = t >= 0
        if not np.any(t_mask):
            return
        proj = np.outer(t, d)
        diff = v - proj
        dist2 = np.einsum("ij,ij->i", diff, diff)
        dist2[~t_mask] = np.inf
        idx = int(np.argmin(dist2))
        threshold = max(0.5, self._grid_spacing * 0.2)
        if dist2[idx] > threshold * threshold:
            return
        node_id = int(self._hover_ids[idx]) if self._hover_ids is not None else idx
        nx, ny, nz = nodes[idx]
        self._selected_node_id = node_id
        self._selected_node_pos = (nx, ny, nz)
        self._update_selected_node_marker()
        QToolTip.showText(
            self.mapToGlobal(self._hover_pos.toPoint()),
            f"Particle {node_id}\n({nx:.3f}, {ny:.3f}, {nz:.3f})",
            self,
        )

    def contextMenuEvent(self, event):
        if self._suppress_next_context_menu:
            self._suppress_next_context_menu = False
            event.accept()
            return
        if hasattr(event, "globalPosition"):
            global_pos = event.globalPosition().toPoint()
        elif hasattr(event, "globalPos"):
            global_pos = event.globalPos()
        else:
            global_pos = self.mapToGlobal(self.rect().center())
        menu = QMenu(self)
        if self._context_menu_hook:
            try:
                self._context_menu_hook(menu, event)
            except Exception:
                pass
        menu.exec(global_pos)
        menu.close()
        event.accept()

    def set_context_menu_hook(self, hook):
        self._context_menu_hook = hook
