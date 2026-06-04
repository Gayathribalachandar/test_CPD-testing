import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QVector3D
from pyqtgraph.opengl import GLViewWidget, GLScatterPlotItem

from app_config import GPU_POINT_PREVIEW_POINT_SIZE, GPU_POINT_PREVIEW_MAX_POINTS


class PointCloudView2D(GLViewWidget):
    """GPU point-cloud preview for large 2D particle sets."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundColor(236, 241, 248)
        self.opts["distance"] = 200
        self.opts["elevation"] = 90
        self.opts["azimuth"] = 0
        self.opts["fov"] = 60
        self.opts["center"] = QVector3D(0, 0, 0)
        self._scatter = None
        self._point_size = float(GPU_POINT_PREVIEW_POINT_SIZE)
        self._point_color = (0.06, 0.13, 0.23, 0.9)
        self._last_bounds = None

    def clear_points(self):
        if self._scatter is not None:
            self.removeItem(self._scatter)
            self._scatter = None
        self._last_bounds = None
        self.update()

    def set_points(self, points, *, size=None, color=None, auto_fit=True):
        pts = np.asarray(points, dtype=np.float32)
        if pts.size == 0:
            self.clear_points()
            return
        if pts.ndim != 2 or pts.shape[1] < 2:
            return
        if pts.shape[1] == 2:
            z = np.zeros((len(pts), 1), dtype=np.float32)
            pts3 = np.hstack((pts, z))
        else:
            pts3 = pts[:, :3]

        if size is not None:
            self._point_size = float(size)
        point_color = self._point_color
        if color is not None:
            color_arr = np.asarray(color, dtype=np.float32)
            if color_arr.ndim == 2 and color_arr.shape[0] == len(pts3):
                point_color = color_arr
            else:
                self._point_color = color
                point_color = color

        fit_pts = pts3
        max_points = int(GPU_POINT_PREVIEW_MAX_POINTS)
        if max_points > 0 and len(pts3) > max_points:
            step = max(1, int(np.ceil(len(pts3) / max_points)))
            pts3 = pts3[::step]
            if isinstance(point_color, np.ndarray) and point_color.ndim == 2 and len(point_color) == len(fit_pts):
                point_color = point_color[::step]

        if self._scatter is None:
            self._scatter = GLScatterPlotItem(
                pos=pts3,
                size=self._point_size,
                color=point_color,
            )
            self.addItem(self._scatter)
        else:
            self._scatter.setData(
                pos=pts3,
                size=self._point_size,
                color=point_color,
            )

        if auto_fit:
            self._fit_to_points(fit_pts)
        self.update()

    def set_points_with_scalars(self, points, scalars, *, size=None, auto_fit=True):
        pts = np.asarray(points, dtype=np.float32)
        vals = np.asarray(scalars, dtype=np.float32).reshape(-1)
        if pts.ndim != 2 or pts.shape[0] == 0:
            self.clear_points()
            return
        if vals.size != pts.shape[0]:
            vals = np.resize(vals, pts.shape[0])
        finite = np.isfinite(vals)
        if np.any(finite):
            vmin = float(np.min(vals[finite]))
            vmax = float(np.max(vals[finite]))
            if abs(vmax - vmin) < 1e-12:
                norm = np.zeros_like(vals, dtype=np.float32)
            else:
                norm = (vals - vmin) / (vmax - vmin)
            norm[~finite] = 0.0
        else:
            norm = np.zeros_like(vals, dtype=np.float32)
        try:
            from matplotlib import cm as mpl_cm
            rgba = np.asarray(mpl_cm.viridis(norm), dtype=np.float32)
        except Exception:
            rgba = np.column_stack(
                (
                    0.1 + 0.9 * norm,
                    0.2 + 0.7 * (1.0 - np.abs(norm - 0.5) * 2.0),
                    1.0 - norm * 0.8,
                    np.ones_like(norm, dtype=np.float32),
                )
            ).astype(np.float32)
        self.set_points(pts, size=size, color=rgba, auto_fit=auto_fit)

    def _fit_to_points(self, pts):
        if pts.size == 0:
            return
        min_x = float(np.min(pts[:, 0]))
        max_x = float(np.max(pts[:, 0]))
        min_y = float(np.min(pts[:, 1]))
        max_y = float(np.max(pts[:, 1]))
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        span = max(max_x - min_x, max_y - min_y, 1.0)
        self.opts["center"] = QVector3D(cx, cy, 0.0)
        self.setCameraPosition(distance=span * 1.25, elevation=90, azimuth=0)

    def mousePressEvent(self, ev):
        lpos = ev.position() if hasattr(ev, "position") else ev.localPos()
        self.mousePos = lpos

    def mouseMoveEvent(self, ev):
        lpos = ev.position() if hasattr(ev, "position") else ev.localPos()
        if not hasattr(self, "mousePos"):
            self.mousePos = lpos
        diff = lpos - self.mousePos
        self.mousePos = lpos

        if ev.buttons() & Qt.MouseButton.LeftButton:
            self.pan(diff.x(), diff.y(), 0, relative="view")
        elif ev.buttons() & Qt.MouseButton.MiddleButton:
            self.pan(diff.x(), diff.y(), 0, relative="view")
        elif ev.buttons() & Qt.MouseButton.RightButton:
            self.pan(diff.x(), diff.y(), 0, relative="view")
