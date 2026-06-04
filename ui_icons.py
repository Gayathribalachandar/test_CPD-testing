import math

from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import (
    QBrush,
    QColor,
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)


ICON_SIZE = 16

_ICON_CACHE = {}


def _make_icon(draw_fn, color, size=ICON_SIZE):
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing)
    draw_fn(painter, size, QColor(color))
    painter.end()
    return QIcon(pix)


def _draw_arrow(painter, size, color, direction=1):
    pen = QPen(color, 1.8)
    painter.setPen(pen)
    painter.setBrush(QBrush(color))
    y = size * 0.5
    x0 = 3
    x1 = size - 3
    if direction < 0:
        painter.drawLine(x1, y, x0 + 5, y)
        arrow = QPolygonF(
            [
                QPointF(x0, y),
                QPointF(x0 + 5, y - 4),
                QPointF(x0 + 5, y + 4),
            ]
        )
    else:
        painter.drawLine(x0, y, x1 - 5, y)
        arrow = QPolygonF(
            [
                QPointF(x1, y),
                QPointF(x1 - 5, y - 4),
                QPointF(x1 - 5, y + 4),
            ]
        )
    painter.drawPolygon(arrow)


def _draw_select(painter, size, color):
    painter.setPen(QPen(color, 1.5))
    painter.setBrush(QBrush(color))
    path = QPainterPath()
    path.moveTo(4, 3)
    path.lineTo(size - 6, size * 0.55)
    path.lineTo(9, size - 4)
    path.lineTo(9, size - 9)
    path.closeSubpath()
    painter.drawPath(path)


def _draw_erase(painter, size, color):
    pen = QPen(color, 1.8)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    painter.drawRect(3, 3, size - 6, size - 6)
    painter.drawLine(4, 4, size - 4, size - 4)


def _draw_zoom_window(painter, size, color):
    painter.setBrush(Qt.NoBrush)
    sel_pen = QPen(color, 1.3)
    sel_pen.setStyle(Qt.DashLine)
    painter.setPen(sel_pen)
    painter.drawRect(QRectF(3.5, 3.5, size * 0.48, size * 0.48))

    mag_pen = QPen(color, 1.6)
    painter.setPen(mag_pen)
    lens_c = QPointF(size * 0.62, size * 0.62)
    lens_r = max(2.6, size * 0.18)
    painter.drawEllipse(lens_c, lens_r, lens_r)
    painter.drawLine(
        QPointF(lens_c.x() + lens_r * 0.65, lens_c.y() + lens_r * 0.65),
        QPointF(size - 3.0, size - 3.0),
    )


def _draw_line(painter, size, color):
    painter.setPen(QPen(color, 1.8))
    painter.drawLine(3, size - 4, size - 3, 4)


def _draw_polyline(painter, size, color):
    painter.setPen(QPen(color, 1.8))
    points = QPolygonF(
        [
            QPointF(3, size - 4),
            QPointF(size * 0.55, size * 0.55),
            QPointF(size - 4, 4),
        ]
    )
    painter.drawPolyline(points)


def _draw_rect(painter, size, color):
    painter.setPen(QPen(color, 1.8))
    painter.setBrush(Qt.NoBrush)
    painter.drawRect(3, 3, size - 6, size - 6)


def _draw_circle(painter, size, color):
    painter.setPen(QPen(color, 1.8))
    painter.setBrush(Qt.NoBrush)
    painter.drawEllipse(3, 3, size - 6, size - 6)


def _draw_workspace_2d(painter, size, color):
    pen = QPen(color, 1.6)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    painter.drawRect(3, 3, size - 6, size - 6)
    # XY axes in a flat plane
    ox = 5
    oy = size - 5
    painter.drawLine(ox, oy, size - 4, oy)
    painter.drawLine(ox, oy, ox, 4)
    painter.drawLine(size - 4, oy, size - 6.5, oy - 1.5)
    painter.drawLine(size - 4, oy, size - 6.5, oy + 1.5)
    painter.drawLine(ox, 4, ox - 1.5, 6.5)
    painter.drawLine(ox, 4, ox + 1.5, 6.5)


def _draw_workspace_3d(painter, size, color):
    pen = QPen(color, 1.6)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    front = QRectF(4, 6, size - 10, size - 10)
    painter.drawRect(front)
    offset = 4
    back = QRectF(front.x() + offset, front.y() - offset, front.width(), front.height())
    painter.drawRect(back)
    painter.drawLine(front.topLeft(), back.topLeft())
    painter.drawLine(front.topRight(), back.topRight())
    painter.drawLine(front.bottomRight(), back.bottomRight())
    painter.drawLine(front.bottomLeft(), back.bottomLeft())
    # Small axis hint for 3D readability
    o = QPointF(5, size - 5)
    painter.drawLine(o, QPointF(9, size - 5))
    painter.drawLine(o, QPointF(5, size - 9))
    painter.drawLine(o, QPointF(8, size - 8))


def _draw_polygon(painter, size, color):
    painter.setPen(QPen(color, 1.8))
    painter.setBrush(Qt.NoBrush)
    cx = cy = size / 2.0
    r = size / 2.0 - 3.5
    points = []
    for i in range(6):
        theta = (math.pi * 2 * i) / 6.0 - math.pi / 6
        points.append(QPointF(cx + r * math.cos(theta), cy + r * math.sin(theta)))
    painter.drawPolygon(QPolygonF(points))


def _draw_arc(painter, size, color):
    painter.setPen(QPen(color, 1.8))
    painter.setBrush(Qt.NoBrush)
    painter.drawArc(3, 3, size - 6, size - 6, 30 * 16, 220 * 16)


def _draw_freeform(painter, size, color):
    painter.setPen(QPen(color, 1.6))
    path = QPainterPath()
    path.moveTo(3, size * 0.7)
    path.cubicTo(size * 0.35, size * 0.2, size * 0.6, size * 0.9, size - 3, 4)
    painter.drawPath(path)


def _draw_auto_convert(painter, size, color):
    painter.setPen(QPen(color, 1.5))
    painter.setBrush(Qt.NoBrush)
    path = QPainterPath()
    path.moveTo(3, size * 0.72)
    path.cubicTo(size * 0.35, size * 0.22, size * 0.58, size * 0.92, size * 0.78, size * 0.32)
    painter.drawPath(path)
    painter.setBrush(QBrush(color))
    tri = QPolygonF(
        [
            QPointF(size * 0.80, size * 0.26),
            QPointF(size - 3, size * 0.50),
            QPointF(size * 0.72, size * 0.58),
        ]
    )
    painter.drawPolygon(tri)
    painter.drawEllipse(QRectF(size * 0.08, size * 0.08, size * 0.16, size * 0.16))


def _draw_slot(painter, size, color):
    painter.setPen(QPen(color, 1.8))
    painter.setBrush(Qt.NoBrush)
    height = size * 0.35
    rect = QRectF(3, (size - height) / 2.0, size - 6, height)
    radius = height / 2.0
    painter.drawRoundedRect(rect, radius, radius)


def _draw_dimension(painter, size, color):
    painter.setPen(QPen(color, 1.6))
    y = size * 0.6
    painter.drawLine(3, y, size - 3, y)
    painter.drawLine(3, y - 4, 3, y + 4)
    painter.drawLine(size - 3, y - 4, size - 3, y + 4)
    painter.setPen(QPen(color, 1.2))
    painter.drawText(QRectF(4, 2, size - 8, size * 0.4), Qt.AlignCenter, "D")


def _draw_constraint(painter, size, color):
    painter.setPen(QPen(color, 1.6))
    painter.setBrush(Qt.NoBrush)
    painter.drawEllipse(3, 4, 6, 6)
    painter.drawEllipse(size - 9, size - 10, 6, 6)
    painter.drawLine(8, 7, size - 8, size - 7)


def _draw_confirm(painter, size, color):
    painter.setPen(QPen(color, 1.8))
    painter.setBrush(Qt.NoBrush)
    painter.drawRect(3, 3, size - 6, size - 6)
    painter.drawLine(5, size * 0.55, size * 0.45, size - 5)
    painter.drawLine(size * 0.45, size - 5, size - 4, 5)


def _draw_cut(painter, size, color):
    painter.setPen(QPen(color, 1.8))
    painter.setBrush(Qt.NoBrush)
    painter.drawEllipse(3, 3, size - 6, size - 6)
    painter.drawEllipse(size * 0.35, size * 0.35, size * 0.3, size * 0.3)


def _draw_snap_grid(painter, size, color):
    painter.setPen(QPen(color, 1.2))
    for x in (6, 11, 16):
        painter.drawLine(x, 4, x, size - 4)
    for y in (6, 11, 16):
        painter.drawLine(4, y, size - 4, y)


def _draw_snap_endpoints(painter, size, color):
    painter.setPen(QPen(color, 1.6))
    painter.setBrush(Qt.NoBrush)
    painter.drawLine(4, size - 5, size - 4, 5)
    painter.setBrush(QBrush(color))
    painter.drawEllipse(2, size - 7, 4, 4)
    painter.drawEllipse(size - 6, 3, 4, 4)


def _draw_parametric(painter, size, color):
    painter.setPen(QPen(color, 1.6))
    painter.setBrush(QBrush(color))
    y = size * 0.6
    painter.drawLine(4, y, size - 4, y)
    painter.drawPolygon(
        QPolygonF(
            [
                QPointF(4, y),
                QPointF(7, y - 3),
                QPointF(7, y + 3),
            ]
        )
    )
    painter.drawPolygon(
        QPolygonF(
            [
                QPointF(size - 4, y),
                QPointF(size - 7, y - 3),
                QPointF(size - 7, y + 3),
            ]
        )
    )
    painter.setPen(QPen(color, 1.2))
    painter.drawLine(size * 0.5, 4, size * 0.5, size - 4)


def _draw_finish(painter, size, color):
    painter.setPen(QPen(color, 1.6))
    painter.setBrush(Qt.NoBrush)
    painter.drawLine(5, 4, 5, size - 4)
    flag = QPolygonF(
        [
            QPointF(5, 5),
            QPointF(size - 4, size * 0.25),
            QPointF(5, size * 0.45),
        ]
    )
    painter.setBrush(QBrush(color))
    painter.drawPolygon(flag)


def _draw_mesh(painter, size, color):
    painter.setPen(QPen(color, 1.6))
    points = QPolygonF(
        [
            QPointF(4, size - 4),
            QPointF(size / 2, 4),
            QPointF(size - 4, size - 4),
        ]
    )
    painter.drawPolygon(points)
    painter.setBrush(QBrush(color))
    for pt in points:
        painter.drawEllipse(pt, 1.5, 1.5)


def _draw_mesh_nodes(painter, size, color):
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(color))
    pts = [
        QPointF(4.0, 4.0),
        QPointF(size / 2.0, 4.0),
        QPointF(size - 4.0, 4.0),
        QPointF(4.0, size / 2.0),
        QPointF(size / 2.0, size / 2.0),
        QPointF(size - 4.0, size / 2.0),
        QPointF(size / 2.0, size - 4.0),
    ]
    for pt in pts:
        painter.drawEllipse(pt, 1.5, 1.5)


def _draw_mesh_nodes_surface(painter, size, color):
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(color))
    # Ring of nodes (surface/boundary-only).
    pts = [
        QPointF(size / 2.0, 3.5),
        QPointF(size - 4.0, size / 2.0),
        QPointF(size / 2.0, size - 4.0),
        QPointF(4.0, size / 2.0),
        QPointF(size * 0.28, size * 0.28),
        QPointF(size * 0.72, size * 0.28),
        QPointF(size * 0.72, size * 0.72),
        QPointF(size * 0.28, size * 0.72),
    ]
    for pt in pts:
        painter.drawEllipse(pt, 1.35, 1.35)
    painter.setPen(QPen(color, 1.0))
    painter.setBrush(Qt.NoBrush)
    painter.drawEllipse(QRectF(3.2, 3.2, size - 6.4, size - 6.4))


def _draw_mesh_nodes_interior(painter, size, color):
    painter.setPen(QPen(color, 1.0))
    painter.setBrush(Qt.NoBrush)
    painter.drawRect(QRectF(3.2, 3.2, size - 6.4, size - 6.4))
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(color))
    # Interior cluster of nodes.
    pts = [
        QPointF(size * 0.36, size * 0.36),
        QPointF(size * 0.50, size * 0.36),
        QPointF(size * 0.64, size * 0.36),
        QPointF(size * 0.36, size * 0.50),
        QPointF(size * 0.50, size * 0.50),
        QPointF(size * 0.64, size * 0.50),
        QPointF(size * 0.36, size * 0.64),
        QPointF(size * 0.50, size * 0.64),
        QPointF(size * 0.64, size * 0.64),
    ]
    for pt in pts:
        painter.drawEllipse(pt, 1.2, 1.2)


def _draw_mesh_elements(painter, size, color):
    painter.setPen(QPen(color, 1.6))
    painter.setBrush(Qt.NoBrush)
    p1 = QPointF(3.5, size - 4.0)
    p2 = QPointF(size * 0.45, 4.0)
    p3 = QPointF(size * 0.78, size * 0.45)
    p4 = QPointF(size - 3.5, size - 4.0)
    painter.drawPolygon(QPolygonF([p1, p2, p3]))
    painter.drawPolygon(QPolygonF([p1, p3, p4]))
    painter.drawLine(p2, p4)


def _draw_mesh_preview(painter, size, color):
    _draw_mesh(painter, size, color)
    painter.setBrush(QBrush(color))
    painter.setPen(QPen(color, 1.2))
    play = QPolygonF(
        [
            QPointF(size - 7, 4),
            QPointF(size - 3, size * 0.25),
            QPointF(size - 7, size * 0.46),
        ]
    )
    painter.drawPolygon(play)


def _draw_mesh_view(painter, size, color):
    painter.setPen(QPen(color, 1.2))
    painter.setBrush(Qt.NoBrush)
    eye_rect = QRectF(2.8, 2.5, size - 5.6, size * 0.34)
    painter.drawEllipse(eye_rect)
    painter.setBrush(QBrush(color))
    painter.drawEllipse(QPointF(size / 2, eye_rect.center().y()), 1.4, 1.4)
    painter.setBrush(Qt.NoBrush)
    painter.setPen(QPen(color, 1.3))
    base_y = size * 0.78
    pts = [QPointF(3.5, base_y), QPointF(size * 0.42, size * 0.48), QPointF(size - 3.5, base_y)]
    painter.drawPolygon(QPolygonF(pts))
    painter.drawLine(pts[0], QPointF(size * 0.68, size * 0.56))


def _draw_mesh_xray(painter, size, color):
    # Eye + translucent/wireframe cube motif.
    painter.setPen(QPen(color, 1.1))
    painter.setBrush(Qt.NoBrush)
    eye_rect = QRectF(2.8, 2.5, size - 5.6, size * 0.30)
    painter.drawEllipse(eye_rect)
    painter.setBrush(QBrush(color))
    painter.drawEllipse(QPointF(size / 2, eye_rect.center().y()), 1.25, 1.25)
    painter.setBrush(Qt.NoBrush)

    cube_pen = QPen(color, 1.25)
    cube_pen.setStyle(Qt.DashLine)
    painter.setPen(cube_pen)
    front = QRectF(4.0, size * 0.47, size * 0.40, size * 0.34)
    back = QRectF(front.x() + 3.0, front.y() - 2.6, front.width(), front.height())
    painter.drawRect(front)
    painter.drawRect(back)
    painter.drawLine(front.topLeft(), back.topLeft())
    painter.drawLine(front.topRight(), back.topRight())
    painter.drawLine(front.bottomLeft(), back.bottomLeft())
    painter.drawLine(front.bottomRight(), back.bottomRight())


def _draw_mesh_quality(painter, size, color):
    painter.setPen(QPen(color, 1.2))
    painter.setBrush(Qt.NoBrush)
    # Small histogram bars (quality distribution)
    bars = [
        QRectF(3.5, size - 5.0, 2.4, 2.0),
        QRectF(7.0, size - 7.0, 2.4, 4.0),
        QRectF(10.5, size - 10.0, 2.4, 7.0),
        QRectF(14.0, size - 6.5, 1.5, 3.5),
    ]
    for r in bars:
        painter.drawRect(r)
    # Overlay a skewed triangle to indicate skewness/element quality
    painter.setPen(QPen(color, 1.3))
    tri = QPolygonF(
        [
            QPointF(size * 0.56, size * 0.20),
            QPointF(size * 0.90, size * 0.34),
            QPointF(size * 0.63, size * 0.56),
        ]
    )
    painter.drawPolygon(tri)
    painter.drawLine(tri[0], tri[2])


def _draw_play(painter, size, color):
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(color))
    tri = QPolygonF(
        [
            QPointF(4, 3),
            QPointF(size - 4, size / 2),
            QPointF(4, size - 3),
        ]
    )
    painter.drawPolygon(tri)


def _draw_pause(painter, size, color):
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(color))
    bar_w = max(2.0, size * 0.18)
    gap = max(2.0, size * 0.12)
    x0 = size * 0.3 - bar_w / 2
    y0 = 3
    h = size - 6
    painter.drawRect(QRectF(x0, y0, bar_w, h))
    painter.drawRect(QRectF(x0 + bar_w + gap, y0, bar_w, h))


def _draw_stop(painter, size, color):
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(color))
    side = size - 6
    painter.drawRect(QRectF(3, 3, side, side))


def _draw_mesh_3d(painter, size, color):
    painter.setPen(QPen(color, 1.6))
    painter.setBrush(Qt.NoBrush)
    top = QPolygonF(
        [
            QPointF(size / 2, 3),
            QPointF(size - 4, size * 0.35),
            QPointF(size / 2, size * 0.6),
            QPointF(4, size * 0.35),
        ]
    )
    offset = QPointF(0, size * 0.28)
    bottom = QPolygonF([p + offset for p in top])
    painter.drawPolygon(top)
    painter.drawPolygon(bottom)
    for i in range(4):
        painter.drawLine(top[i], bottom[i])
    painter.drawLine(top[0], top[2])
    painter.drawLine(bottom[1], bottom[3])


def _draw_porous(painter, size, color):
    pen = QPen(color, 1.6)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    painter.drawRect(3, 3, size - 6, size - 6)
    r = size * 0.16
    centers = [
        (size * 0.35, size * 0.35),
        (size * 0.65, size * 0.35),
        (size * 0.35, size * 0.65),
        (size * 0.65, size * 0.65),
    ]
    for cx, cy in centers:
        painter.drawEllipse(QPointF(cx, cy), r, r)


def _draw_copy(painter, size, color):
    pen = QPen(color, 1.6)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    painter.drawRect(5, 5, size - 8, size - 8)
    painter.drawRect(3, 3, size - 8, size - 8)


def _draw_mirror(painter, size, color):
    pen = QPen(color, 1.6)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    painter.drawLine(size / 2, 3, size / 2, size - 3)
    left = QPolygonF(
        [
            QPointF(size * 0.2, size * 0.2),
            QPointF(size * 0.45, size * 0.5),
            QPointF(size * 0.2, size * 0.8),
        ]
    )
    right = QPolygonF(
        [
            QPointF(size * 0.8, size * 0.2),
            QPointF(size * 0.55, size * 0.5),
            QPointF(size * 0.8, size * 0.8),
        ]
    )
    painter.drawPolygon(left)
    painter.drawPolygon(right)


def _draw_trim(painter, size, color):
    pen = QPen(color, 1.6)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    painter.drawLine(3, size * 0.65, size - 3, size * 0.35)
    painter.drawLine(size * 0.45, size * 0.2, size * 0.65, size * 0.4)
    painter.drawLine(size * 0.45, size * 0.4, size * 0.65, size * 0.2)


def _draw_join(painter, size, color):
    pen = QPen(color, 1.6)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    painter.drawEllipse(QPointF(size * 0.4, size * 0.5), size * 0.22, size * 0.22)
    painter.drawEllipse(QPointF(size * 0.6, size * 0.5), size * 0.22, size * 0.22)
    painter.drawLine(size * 0.48, size * 0.35, size * 0.52, size * 0.35)
    painter.drawLine(size * 0.48, size * 0.65, size * 0.52, size * 0.65)


def _draw_pattern(painter, size, color):
    pen = QPen(color, 1.4)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    s = size * 0.22
    offsets = [(0.2, 0.2), (0.6, 0.2), (0.2, 0.6), (0.6, 0.6)]
    for ox, oy in offsets:
        painter.drawRect(QRectF(size * ox, size * oy, s, s))


def _draw_box(painter, size, color):
    pen = QPen(color, 1.6)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    front = QRectF(4, 6, size - 10, size - 10)
    painter.drawRect(front)
    offset = 3
    back = QRectF(front.x() + offset, front.y() - offset, front.width(), front.height())
    painter.drawRect(back)
    painter.drawLine(front.topLeft(), back.topLeft())
    painter.drawLine(front.topRight(), back.topRight())
    painter.drawLine(front.bottomRight(), back.bottomRight())
    painter.drawLine(front.bottomLeft(), back.bottomLeft())


def _draw_cylinder(painter, size, color):
    pen = QPen(color, 1.6)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    top_h = size * 0.28
    rect_top = QRectF(4, 4, size - 8, top_h)
    rect_bottom = QRectF(4, size - 4 - top_h, size - 8, top_h)
    painter.drawEllipse(rect_top)
    painter.drawEllipse(rect_bottom)
    painter.drawLine(4, 4 + top_h / 2, 4, size - 4 - top_h / 2)
    painter.drawLine(size - 4, 4 + top_h / 2, size - 4, size - 4 - top_h / 2)


def _draw_sphere(painter, size, color):
    pen = QPen(color, 1.6)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    painter.drawEllipse(3, 3, size - 6, size - 6)
    equator = QRectF(3, size * 0.35, size - 6, size * 0.3)
    painter.drawEllipse(equator)


def _draw_cone(painter, size, color):
    pen = QPen(color, 1.6)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    apex = QPointF(size / 2, 3)
    left = QPointF(4, size * 0.72)
    right = QPointF(size - 4, size * 0.72)
    painter.drawLine(apex, left)
    painter.drawLine(apex, right)
    base = QRectF(4, size * 0.62, size - 8, size * 0.26)
    painter.drawEllipse(base)


def _draw_ring(painter, size, color):
    pen = QPen(color, 1.6)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    painter.drawEllipse(3, 3, size - 6, size - 6)
    inner = QRectF(size * 0.35, size * 0.35, size * 0.3, size * 0.3)
    painter.drawEllipse(inner)


def _draw_edit(painter, size, color):
    pen = QPen(color, 1.6)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    painter.drawLine(4, size - 5, size - 6, 5)
    painter.drawLine(6, size - 4, 4, size - 6)
    painter.drawLine(size - 6, 5, size - 4, 7)


def _draw_delete(painter, size, color):
    pen = QPen(color, 1.6)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    painter.drawRect(5, 6, size - 10, size - 10)
    painter.drawLine(4, 6, size - 4, 6)
    painter.drawLine(size * 0.4, 4, size * 0.6, 4)
    painter.drawLine(size * 0.42, 8, size * 0.42, size - 6)
    painter.drawLine(size * 0.58, 8, size * 0.58, size - 6)


def _draw_boolean_union(painter, size, color):
    pen = QPen(color, 1.4)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    r = size * 0.55
    rect1 = QRectF(2, 4, r, r)
    rect2 = QRectF(size - 2 - r, 4, r, r)
    painter.drawEllipse(rect1)
    painter.drawEllipse(rect2)
    painter.setPen(QPen(color, 1.6))
    cx = size / 2
    cy = size * 0.78
    painter.drawLine(cx - 3, cy, cx + 3, cy)
    painter.drawLine(cx, cy - 3, cx, cy + 3)


def _draw_boolean_subtract(painter, size, color):
    pen = QPen(color, 1.4)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    r = size * 0.55
    rect1 = QRectF(2, 4, r, r)
    rect2 = QRectF(size - 2 - r, 4, r, r)
    painter.drawEllipse(rect1)
    painter.drawEllipse(rect2)
    painter.setPen(QPen(color, 1.6))
    cx = size / 2
    cy = size * 0.78
    painter.drawLine(cx - 3, cy, cx + 3, cy)


def _draw_boolean_intersect(painter, size, color):
    pen = QPen(color, 1.4)
    painter.setPen(pen)
    r = size * 0.55
    rect1 = QRectF(2, 4, r, r)
    rect2 = QRectF(size - 2 - r, 4, r, r)
    path1 = QPainterPath()
    path1.addEllipse(rect1)
    path2 = QPainterPath()
    path2.addEllipse(rect2)
    inter = path1.intersected(path2)
    painter.setBrush(Qt.NoBrush)
    painter.drawEllipse(rect1)
    painter.drawEllipse(rect2)
    painter.setBrush(QBrush(color))
    painter.drawPath(inter)


def _draw_next_stage(painter, size, color):
    painter.setPen(QPen(color, 1.8))
    painter.setBrush(QBrush(color))
    y = size * 0.5
    painter.drawLine(3, y, size - 7, y)
    arrow = QPolygonF(
        [
            QPointF(size - 7, y - 4),
            QPointF(size - 2, y),
            QPointF(size - 7, y + 4),
        ]
    )
    painter.drawPolygon(arrow)
    painter.drawLine(3, y - 5, 3, y + 5)


def _draw_export(painter, size, color):
    painter.setPen(QPen(color, 1.6))
    painter.setBrush(Qt.NoBrush)
    painter.drawRect(3, size * 0.55, size - 6, size * 0.35)
    painter.drawLine(size / 2, 3, size / 2, size * 0.55)
    painter.setBrush(QBrush(color))
    painter.drawPolygon(
        QPolygonF(
            [
                QPointF(size / 2, size * 0.6),
                QPointF(size / 2 - 4, size * 0.48),
                QPointF(size / 2 + 4, size * 0.48),
            ]
        )
    )


def _draw_gizmo_move(painter, size, color):
    pen = QPen(color, 1.6)
    painter.setPen(pen)
    painter.setBrush(QBrush(color))
    center = QPointF(size / 2, size / 2)
    painter.drawLine(3, center.y(), size - 3, center.y())
    painter.drawLine(center.x(), 3, center.x(), size - 3)
    painter.drawPolygon(
        QPolygonF(
            [
                QPointF(size - 3, center.y()),
                QPointF(size - 6, center.y() - 3),
                QPointF(size - 6, center.y() + 3),
            ]
        )
    )
    painter.drawPolygon(
        QPolygonF(
            [
                QPointF(3, center.y()),
                QPointF(6, center.y() - 3),
                QPointF(6, center.y() + 3),
            ]
        )
    )
    painter.drawPolygon(
        QPolygonF(
            [
                QPointF(center.x(), 3),
                QPointF(center.x() - 3, 6),
                QPointF(center.x() + 3, 6),
            ]
        )
    )
    painter.drawPolygon(
        QPolygonF(
            [
                QPointF(center.x(), size - 3),
                QPointF(center.x() - 3, size - 6),
                QPointF(center.x() + 3, size - 6),
            ]
        )
    )


def _draw_gizmo_rotate(painter, size, color):
    pen = QPen(color, 1.6)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    painter.drawArc(3, 3, size - 6, size - 6, 30 * 16, 300 * 16)
    painter.setBrush(QBrush(color))
    painter.drawPolygon(
        QPolygonF(
            [
                QPointF(size - 4, size * 0.4),
                QPointF(size - 7, size * 0.35),
                QPointF(size - 6, size * 0.48),
            ]
        )
    )


def _draw_gizmo_scale(painter, size, color):
    pen = QPen(color, 1.6)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    painter.drawRect(5, 5, size - 10, size - 10)
    painter.setBrush(QBrush(color))
    painter.drawPolygon(
        QPolygonF(
            [
                QPointF(size - 3, size - 6),
                QPointF(size - 8, size - 8),
                QPointF(size - 6, size - 3),
            ]
        )
    )


def _draw_fit(painter, size, color):
    painter.setPen(QPen(color, 1.4))
    painter.setBrush(Qt.NoBrush)
    painter.drawRect(3, 3, size - 6, size - 6)
    c = QPointF(size / 2.0, size / 2.0)
    painter.drawLine(c.x(), 4.0, c.x(), 7.5)
    painter.drawLine(c.x(), size - 4.0, c.x(), size - 7.5)
    painter.drawLine(4.0, c.y(), 7.5, c.y())
    painter.drawLine(size - 4.0, c.y(), size - 7.5, c.y())
    # Arrow heads pointing inward (fit to view)
    painter.drawLine(c.x(), 7.5, c.x() - 1.6, 9.1)
    painter.drawLine(c.x(), 7.5, c.x() + 1.6, 9.1)
    painter.drawLine(c.x(), size - 7.5, c.x() - 1.6, size - 9.1)
    painter.drawLine(c.x(), size - 7.5, c.x() + 1.6, size - 9.1)
    painter.drawLine(7.5, c.y(), 9.1, c.y() - 1.6)
    painter.drawLine(7.5, c.y(), 9.1, c.y() + 1.6)
    painter.drawLine(size - 7.5, c.y(), size - 9.1, c.y() - 1.6)
    painter.drawLine(size - 7.5, c.y(), size - 9.1, c.y() + 1.6)


def _draw_frame(painter, size, color):
    painter.setPen(QPen(color, 1.6))
    painter.setBrush(Qt.NoBrush)
    pad = 3.5
    arm = 4.0
    # Corner brackets (camera/frame style)
    painter.drawLine(pad, pad + arm, pad, pad)
    painter.drawLine(pad, pad, pad + arm, pad)
    painter.drawLine(size - pad - arm, pad, size - pad, pad)
    painter.drawLine(size - pad, pad, size - pad, pad + arm)
    painter.drawLine(pad, size - pad - arm, pad, size - pad)
    painter.drawLine(pad, size - pad, pad + arm, size - pad)
    painter.drawLine(size - pad - arm, size - pad, size - pad, size - pad)
    painter.drawLine(size - pad, size - pad - arm, size - pad, size - pad)
    # Selected-object rectangle inside the frame
    painter.drawRect(QRectF(size * 0.33, size * 0.33, size * 0.34, size * 0.24))


def _draw_full_screen(painter, size, color):
    painter.setPen(QPen(color, 1.6))
    painter.setBrush(Qt.NoBrush)
    pad = 3.2
    arm = 5.6
    # Standard "enter fullscreen" corners (expand outwards).
    painter.drawLine(pad, pad + arm, pad, pad)
    painter.drawLine(pad, pad, pad + arm, pad)
    painter.drawLine(size - pad - arm, pad, size - pad, pad)
    painter.drawLine(size - pad, pad, size - pad, pad + arm)
    painter.drawLine(pad, size - pad - arm, pad, size - pad)
    painter.drawLine(pad, size - pad, pad + arm, size - pad)
    painter.drawLine(size - pad - arm, size - pad, size - pad, size - pad)
    painter.drawLine(size - pad, size - pad - arm, size - pad, size - pad)


def _draw_exit_full_screen(painter, size, color):
    painter.setPen(QPen(color, 1.6))
    painter.setBrush(Qt.NoBrush)
    pad = 3.2
    inner = 6.2
    arm = 4.0
    # Standard "exit fullscreen" corners (contract inwards).
    painter.drawLine(inner, pad, inner, pad + arm)
    painter.drawLine(inner, pad, inner + arm, pad)
    painter.drawLine(size - inner, pad, size - inner, pad + arm)
    painter.drawLine(size - inner - arm, pad, size - inner, pad)
    painter.drawLine(inner, size - pad, inner, size - pad - arm)
    painter.drawLine(inner, size - pad, inner + arm, size - pad)
    painter.drawLine(size - inner, size - pad, size - inner, size - pad - arm)
    painter.drawLine(size - inner - arm, size - pad, size - inner, size - pad)


def _draw_command_bar(painter, size, color):
    painter.setPen(QPen(color, 1.6))
    painter.setBrush(Qt.NoBrush)
    rect = QRectF(3, size * 0.35, size - 6, size * 0.3)
    painter.drawRoundedRect(rect, 2, 2)
    x = 6
    y = size * 0.5
    painter.drawLine(x, y - 2, x + 3, y)
    painter.drawLine(x, y + 2, x + 3, y)
    painter.drawLine(x + 6, y + 3, x + 10, y + 3)

def _draw_origin(painter, size, color):
    painter.setPen(QPen(color, 1.6))
    cx = size / 2
    cy = size / 2
    painter.drawLine(cx, cy, size - 3, cy)
    painter.drawLine(cx, cy, cx, 3)
    painter.drawLine(cx, cy, 4, size - 4)
    # Arrowheads on +X and +Y
    painter.drawLine(size - 3, cy, size - 6, cy - 1.8)
    painter.drawLine(size - 3, cy, size - 6, cy + 1.8)
    painter.drawLine(cx, 3, cx - 1.8, 6)
    painter.drawLine(cx, 3, cx + 1.8, 6)
    painter.setBrush(Qt.NoBrush)
    painter.drawEllipse(cx - 2.5, cy - 2.5, 5, 5)


def _draw_velocity(painter, size, color):
    painter.setPen(QPen(color, 1.8))
    painter.setBrush(QBrush(color))
    painter.drawLine(3, size * 0.6, size - 5, size * 0.6)
    painter.drawPolygon(
        QPolygonF(
            [
                QPointF(size - 3, size * 0.6),
                QPointF(size - 7, size * 0.6 - 3),
                QPointF(size - 7, size * 0.6 + 3),
            ]
        )
    )
    painter.setPen(QPen(color, 1.2))
    painter.drawLine(3, size * 0.35, size * 0.45, size * 0.35)


def _draw_stage_geometry(painter, size, color):
    painter.setPen(QPen(color, 1.6))
    painter.setBrush(Qt.NoBrush)
    painter.drawRect(3, 3, size - 6, size - 6)
    painter.drawLine(4, size - 4, size - 4, 4)


def _draw_stage_materials(painter, size, color):
    painter.setPen(QPen(color, 1.6))
    painter.setBrush(Qt.NoBrush)
    painter.drawRect(4, 4, size - 8, size - 8)
    painter.drawLine(4, size * 0.65, size - 4, size * 0.65)
    painter.setBrush(QBrush(color))
    painter.drawRect(5, size * 0.65, size - 10, size * 0.2)


def _draw_stage_interfaces(painter, size, color):
    painter.setPen(QPen(color, 1.6))
    painter.setBrush(Qt.NoBrush)
    painter.drawEllipse(3, size / 2 - 4, 8, 8)
    painter.drawEllipse(size - 11, size / 2 - 4, 8, 8)
    painter.drawLine(11, size / 2, size - 11, size / 2)


# Backward-compatible alias (use _draw_stage_interfaces).
_draw_stage_interactions = _draw_stage_interfaces


def _draw_stage_bcs(painter, size, color):
    painter.setPen(QPen(color, 1.6))
    painter.setBrush(QBrush(color))
    base_y = size - 4
    poly = QPolygonF(
        [
            QPointF(size / 2, base_y - 8),
            QPointF(size / 2 - 6, base_y),
            QPointF(size / 2 + 6, base_y),
        ]
    )
    painter.drawPolygon(poly)
    painter.setBrush(Qt.NoBrush)
    painter.drawLine(4, base_y, size - 4, base_y)


def _draw_stage_job(painter, size, color):
    painter.setPen(QPen(color, 1.6))
    painter.setBrush(QBrush(color))
    poly = QPolygonF(
        [
            QPointF(5, 4),
            QPointF(size - 4, size / 2),
            QPointF(5, size - 4),
        ]
    )
    painter.drawPolygon(poly)


def _draw_stage_fluid(painter, size, color):
    painter.setPen(QPen(color, 1.6))
    painter.setBrush(QBrush(color.lighter(130)))
    path = QPainterPath()
    path.moveTo(size * 0.5, 3.0)
    path.cubicTo(size * 0.76, size * 0.28, size * 0.82, size * 0.55, size * 0.5, size - 3.0)
    path.cubicTo(size * 0.18, size * 0.55, size * 0.24, size * 0.28, size * 0.5, 3.0)
    painter.drawPath(path)


def _draw_stage_fracture(painter, size, color):
    painter.setPen(QPen(color, 1.8))
    painter.setBrush(Qt.NoBrush)
    painter.drawRect(3.5, 3.5, size - 7.0, size - 7.0)
    painter.drawLine(size * 0.48, 4.5, size * 0.42, size * 0.36)
    painter.drawLine(size * 0.42, size * 0.36, size * 0.56, size * 0.52)
    painter.drawLine(size * 0.56, size * 0.52, size * 0.44, size * 0.68)
    painter.drawLine(size * 0.44, size * 0.68, size * 0.58, size - 4.5)


def _draw_stage_mesh(painter, size, color):
    painter.setPen(QPen(color, 1.6))
    painter.setBrush(Qt.NoBrush)
    painter.drawRect(3, 3, size - 6, size - 6)
    painter.setPen(QPen(color, 1.1))
    mid = size / 2.0
    painter.drawLine(3, mid, size - 3, mid)
    painter.drawLine(mid, 3, mid, size - 3)
    painter.drawLine(3, 3, size - 3, size - 3)


def _draw_measure(painter, size, color):
    # Ruler-like icon: tilted bar with tick marks + a small distance arrow.
    painter.setPen(QPen(color, 1.4))
    painter.setBrush(Qt.NoBrush)
    # Diagonal ruler body
    rect_path = QPainterPath()
    rect_path.moveTo(size * 0.18, size * 0.68)
    rect_path.lineTo(size * 0.68, size * 0.18)
    rect_path.lineTo(size * 0.82, size * 0.32)
    rect_path.lineTo(size * 0.32, size * 0.82)
    rect_path.closeSubpath()
    painter.drawPath(rect_path)
    # Tick marks along the ruler (perpendicular to its long axis)
    painter.setPen(QPen(color, 1.0))
    for t in (0.25, 0.40, 0.55, 0.70):
        # Point on the upper edge of the ruler
        ux = size * (0.18 + (0.68 - 0.18) * t)
        uy = size * (0.68 + (0.18 - 0.68) * t)
        # Inward direction (perpendicular to 45° axis = (1, 1) normalised)
        dx, dy = size * 0.07, size * 0.07
        painter.drawLine(ux, uy, ux + dx, uy + dy)


def _draw_minimap(painter, size, color):
    # Mini-map icon: an outer frame with a smaller "viewport" rectangle inside.
    painter.setPen(QPen(color, 1.4))
    painter.setBrush(Qt.NoBrush)
    pad = 3.0
    painter.drawRect(pad, pad, size - 2 * pad, size - 2 * pad)
    # Inner viewport rectangle, offset slightly toward bottom-left.
    inner_x = pad + (size - 2 * pad) * 0.18
    inner_y = pad + (size - 2 * pad) * 0.30
    inner_w = (size - 2 * pad) * 0.42
    inner_h = (size - 2 * pad) * 0.42
    painter.setBrush(QColor(color))
    painter.setPen(QPen(color, 1.0))
    painter.drawRect(inner_x, inner_y, inner_w, inner_h)


def _draw_fluid(painter, size, color):
    painter.setPen(QPen(color, 1.6))
    painter.setBrush(Qt.NoBrush)
    for y_frac in (0.30, 0.55, 0.80):
        y = size * y_frac
        path = QPainterPath()
        path.moveTo(3.0, y)
        path.cubicTo(size * 0.33, y - 3.5, size * 0.66, y + 3.5, size - 3.0, y)
        painter.drawPath(path)


def _draw_fracture(painter, size, color):
    painter.setPen(QPen(color, 1.9))
    painter.setBrush(Qt.NoBrush)
    painter.drawLine(size * 0.55, 3.5, size * 0.40, size * 0.30)
    painter.drawLine(size * 0.40, size * 0.30, size * 0.58, size * 0.50)
    painter.drawLine(size * 0.58, size * 0.50, size * 0.38, size * 0.70)
    painter.drawLine(size * 0.38, size * 0.70, size * 0.55, size - 3.5)
    painter.setPen(QPen(color, 1.1))
    painter.drawLine(size * 0.58, size * 0.50, size * 0.78, size * 0.40)
    painter.drawLine(size * 0.40, size * 0.30, size * 0.22, size * 0.20)


_ICON_BUILDERS = {
    "select": (_draw_select, "#1f6feb"),
    "erase": (_draw_erase, "#d73a49"),
    "zoom_window": (_draw_zoom_window, "#0f766e"),
    "line": (_draw_line, "#374151"),
    "polyline": (_draw_polyline, "#374151"),
    "rect": (_draw_rect, "#374151"),
    "circle": (_draw_circle, "#374151"),
    "slot": (_draw_slot, "#374151"),
    "polygon": (_draw_polygon, "#374151"),
    "arc": (_draw_arc, "#374151"),
    "arc_select": (_draw_arc, "#0ea5e9"),
    "freeform": (_draw_freeform, "#374151"),
    "auto_convert": (_draw_auto_convert, "#0f766e"),
    "dimension": (_draw_dimension, "#0f766e"),
    "constraint": (_draw_constraint, "#0f766e"),
    "confirm": (_draw_confirm, "#2ea043"),
    "cut": (_draw_cut, "#b45309"),
    "undo": (lambda p, s, c: _draw_arrow(p, s, c, direction=-1), "#0ea5e9"),
    "redo": (lambda p, s, c: _draw_arrow(p, s, c, direction=1), "#0ea5e9"),
    "snap_grid": (_draw_snap_grid, "#f59e0b"),
    "snap_endpoints": (_draw_snap_endpoints, "#f59e0b"),
    "parametric": (_draw_parametric, "#0f766e"),
    "finish": (_draw_finish, "#2ea043"),
    "mesh": (_draw_mesh, "#2563eb"),
    "mesh_preview": (_draw_mesh_preview, "#2563eb"),
    "porous": (_draw_porous, "#0f766e"),
    "copy": (_draw_copy, "#0ea5e9"),
    "mirror": (_draw_mirror, "#0ea5e9"),
    "trim": (_draw_trim, "#ef4444"),
    "join": (_draw_join, "#10b981"),
    "pattern": (_draw_pattern, "#6366f1"),
    "play": (_draw_play, "#10b981"),
    "pause": (_draw_pause, "#f59e0b"),
    "stop": (_draw_stop, "#ef4444"),
    "mesh_view": (_draw_mesh_view, "#1d4ed8"),
    "mesh_xray": (_draw_mesh_xray, "#1d4ed8"),
    "mesh_quality": (_draw_mesh_quality, "#1d4ed8"),
    "mesh_3d": (_draw_mesh_3d, "#1d4ed8"),
    "mesh_nodes": (_draw_mesh_nodes, "#1d4ed8"),
    "mesh_nodes_surface": (_draw_mesh_nodes_surface, "#1d4ed8"),
    "mesh_nodes_interior": (_draw_mesh_nodes_interior, "#1d4ed8"),
    "mesh_elements": (_draw_mesh_elements, "#1d4ed8"),
    "prim_box": (_draw_box, "#0f766e"),
    "prim_cylinder": (_draw_cylinder, "#0f766e"),
    "prim_sphere": (_draw_sphere, "#0f766e"),
    "prim_cone": (_draw_cone, "#0f766e"),
    "prim_ring": (_draw_ring, "#0f766e"),
    "edit": (_draw_edit, "#0ea5e9"),
    "delete": (_draw_delete, "#ef4444"),
    "boolean_union": (_draw_boolean_union, "#10b981"),
    "boolean_subtract": (_draw_boolean_subtract, "#f97316"),
    "boolean_intersect": (_draw_boolean_intersect, "#6366f1"),
    "gizmo_move": (_draw_gizmo_move, "#0f766e"),
    "gizmo_rotate": (_draw_gizmo_rotate, "#f59e0b"),
    "gizmo_scale": (_draw_gizmo_scale, "#ef4444"),
    "next_stage": (_draw_next_stage, "#0ea5e9"),
    "export": (_draw_export, "#059669"),
    "fit": (_draw_fit, "#0f766e"),
    "frame": (_draw_frame, "#0891b2"),
    "measure": (_draw_measure, "#dc2626"),
    "minimap": (_draw_minimap, "#0f766e"),
    "full_screen": (_draw_full_screen, "#0f766e"),
    "exit_full_screen": (_draw_exit_full_screen, "#0f766e"),
    "command_bar": (_draw_command_bar, "#0f766e"),
    "origin": (_draw_origin, "#0f766e"),
    "velocity": (_draw_velocity, "#ef4444"),
    "workspace_2d": (_draw_workspace_2d, "#0f766e"),
    "workspace_3d": (_draw_workspace_3d, "#0284c7"),
    "stage_geometry": (_draw_stage_geometry, "#0f172a"),
    "stage_materials": (_draw_stage_materials, "#14b8a6"),
    "stage_interfaces": (_draw_stage_interfaces, "#0ea5e9"),
    "stage_interactions": (_draw_stage_interfaces, "#0ea5e9"),  # Backward-compatible alias
    "stage_bcs": (_draw_stage_bcs, "#f97316"),
    "stage_fluid": (_draw_stage_fluid, "#0284c7"),
    "stage_fracture": (_draw_stage_fracture, "#dc2626"),
    "fluid": (_draw_fluid, "#0284c7"),
    "fracture": (_draw_fracture, "#dc2626"),
    "stage_mesh": (_draw_stage_mesh, "#2563eb"),
    "stage_job": (_draw_stage_job, "#10b981"),
}


_STAGE_ICON_CACHE = {}
_SVG_ICON_CACHE = {}
_SVG_ICON_ALIASES = {
    "parts": "geometry",
    "rect": "rectangle",
    "freeform": "freecurve",
    "snap_endpoints": "snap_endpoint",
    "snap_end": "snap_endpoint",
    "snap_mid": "snap_midpoint",
    "snap_midpoints": "snap_midpoint",
    "zoom_window": "zoom",
    "search": "zoom",
    "load": "loads",
    "constraint": "bc",
    "graph": "results",
    "chain": "interactions",
    "cube": "materials",
    "particle": "particles",
    "connections": "connections",
    "selection": "select",
    "bc_loads": "bc_loads",
    "property_inspector": "inspector",
    "rigid": "fix",
}


def _render_stage_svg_icon(svg_markup, size):
    from PySide6.QtCore import QByteArray
    from PySide6.QtSvg import QSvgRenderer

    render_size = max(16, int(size or 24))
    pix = QPixmap(render_size, render_size)
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setRenderHint(QPainter.SmoothPixmapTransform)
    renderer = QSvgRenderer(QByteArray(svg_markup.encode("utf-8")))
    renderer.render(painter)
    painter.end()
    return QIcon(pix)

def _svg_common(stroke):
    return (
        f'stroke="{stroke}" stroke-width="1.8" stroke-linecap="round" '
        f'stroke-linejoin="round" fill="none"'
    )


def _svg_icon_markup(name, state="default"):
    icon_name = _SVG_ICON_ALIASES.get(str(name or "").lower(), str(name or "").lower())
    state_name = str(state or "default").lower()
    palette = {
        "project": "#5b6470",
        "model": "#5b6470",
        "geometry": "#5b6470",
        "particles": "#5b6470",
        "materials": "#5b6470",
        "interactions": "#5b6470",
        "bc": "#5b6470",
        "loads": "#5b6470",
        "bc_loads": "#5b6470",
        "solve": "#5b6470",
        "results": "#5b6470",
        "rectangle": "#5b6470",
        "circle": "#5b6470",
        "polygon": "#5b6470",
        "line": "#5b6470",
        "freecurve": "#5b6470",
        "extrude": "#5b6470",
        "import": "#5b6470",
        "snap_grid": "#5b6470",
        "snap_endpoint": "#5b6470",
        "filter": "#5b6470",
        "snap_midpoint": "#5b6470",
        "snap_angle": "#5b6470",
        "select": "#5b6470",
        "pan": "#5b6470",
        "rotate": "#5b6470",
        "zoom": "#5b6470",
        "box_select": "#5b6470",
        "fix": "#5b6470",
        "velocity": "#5b6470",
        "displacement": "#5b6470",
        "force": "#5b6470",
        "pressure": "#5b6470",
        "gravity": "#5b6470",
        "displacement_result": "#5b6470",
        "stress_result": "#5b6470",
        "strain_result": "#5b6470",
        "units": "#5b6470",
        "help": "#5b6470",
        "inspector": "#5b6470",
        "connections": "#5b6470",
        "axis_x": "#d64545",
        "axis_y": "#2f9d5d",
        "axis_z": "#2f6fd6",
        "magnitude": "#5b6470",
        "move_up": "#5b6470",
        "move_down": "#5b6470",
    }
    stage_palette = {
        "geometry": "#0f766e",
        "particles": "#2563eb",
        "materials": "#14b8a6",
        "interactions": "#0ea5e9",
        "bc_loads": "#f97316",
        "bc": "#f97316",
        "loads": "#f97316",
        "solve": "#10b981",
        "results": "#7c3aed",
    }
    stroke = stage_palette.get(icon_name, palette.get(icon_name, "#5b6470"))
    if state_name == "future":
        stroke = "#9aa8ba"
    elif state_name == "completed":
        stroke = stage_palette.get(icon_name, "#16a34a")
    common = _svg_common(stroke)
    body = ""
    if icon_name == "project":
        body = (
            f'<path d="M4 8V18H20V8H12L10.5 6.5H4Z" {common}/>'
            f'<path d="M7 12H17" {common}/>'
            f'<path d="M7 14.5H14" {common}/>'
        )
    elif icon_name == "model":
        body = (
            f'<rect x="5" y="7" width="12" height="12" rx="0.6" {common}/>'
            f'<path d="M5 7L8 4H20V16L17 19" {common}/>'
            f'<path d="M17 7L20 4" {common}/>'
        )
    elif icon_name == "geometry":
        body = (
            f'<rect x="5" y="6" width="7" height="6" rx="1.2" {common}/>'
            f'<circle cx="17.5" cy="9" r="3" {common}/>'
            f'<path d="M6 18L10 14L13 18L18 13L20 18" {common}/>'
        )
    elif icon_name == "particles":
        body = (
            f'<circle cx="7" cy="8" r="2.2" {common}/>'
            f'<circle cx="16" cy="7" r="2.2" {common}/>'
            f'<circle cx="12" cy="14" r="2.2" {common}/>'
            f'<circle cx="18" cy="17" r="2.2" {common}/>'
            f'<circle cx="7" cy="18" r="2.2" {common}/>'
            f'<path d="M8.8 9.2L10.3 12.2L14 8.8L16.3 14.9L9 17.3" {common}/>'
        )
    elif icon_name == "materials":
        body = (
            f'<path d="M12 4.8L18 8V16L12 19.2L6 16V8Z" {common}/>'
            f'<path d="M12 4.8V12.2" {common}/>'
            f'<path d="M6 8L12 12.2L18 8" {common}/>'
        )
    elif icon_name == "interactions":
        body = (
            f'<path d="M10 9.2C8.1 7.3 5.1 7.3 3.2 9.2C1.3 11.1 1.3 14.1 3.2 16C5.1 17.9 8.1 17.9 10 16L12 14" {common}/>'
            f'<path d="M14 10L16 8C17.9 6.1 20.9 6.1 22.8 8C24.7 9.9 24.7 12.9 22.8 14.8C20.9 16.7 17.9 16.7 16 14.8" {common}/>'
        )
    elif icon_name == "bc_loads":
        body = (
            f'<path d="M6 6V18" {common}/>'
            f'<path d="M9 8V16" {common}/>'
            f'<path d="M11.5 12H20" {common}/>'
            f'<path d="M16.5 8L20.5 12L16.5 16" {common}/>'
        )
    elif icon_name == "bc":
        body = (
            f'<path d="M6 5V19" {common}/>'
            f'<path d="M9 7V17" {common}/>'
            f'<path d="M11.5 7L7.5 11" {common}/>'
            f'<path d="M11.5 11L7.5 15" {common}/>'
            f'<path d="M11.5 15L7.5 19" {common}/>'
        )
    elif icon_name == "loads":
        body = (
            f'<path d="M5 12H19" {common}/>'
            f'<path d="M14.5 7.5L19 12L14.5 16.5" {common}/>'
        )
    elif icon_name == "solve":
        body = (
            f'<path d="M10 6L18 12L10 18Z" {common}/>'
            f'<circle cx="12" cy="12" r="8" {common}/>'
        )
    elif icon_name == "results":
        body = (
            f'<path d="M5 18V7" {common}/>'
            f'<path d="M5 18H19" {common}/>'
            f'<path d="M8 15L11 11L14 13L19 7" {common}/>'
        )
    elif icon_name == "rectangle":
        body = f'<rect x="5" y="6" width="14" height="12" rx="1.5" {common}/>'
    elif icon_name == "circle":
        body = f'<circle cx="12" cy="12" r="6.8" {common}/>'
    elif icon_name == "polygon":
        body = f'<path d="M8 6.5L16 6.5L20 12L16 17.5L8 17.5L4 12Z" {common}/>'
    elif icon_name == "line":
        body = (
            f'<path d="M6 18L18 6" {common}/>'
            f'<circle cx="6" cy="18" r="1.6" {common}/>'
            f'<circle cx="18" cy="6" r="1.6" {common}/>'
        )
    elif icon_name == "freecurve":
        body = f'<path d="M4.5 16.5C7 8 9.5 7 12 11.5C14.5 16 17 17 19.5 7.5" {common}/>'
    elif icon_name == "extrude":
        body = (
            f'<rect x="5.5" y="10" width="8" height="8" rx="1.2" {common}/>'
            f'<rect x="10.5" y="5" width="8" height="8" rx="1.2" {common}/>'
            f'<path d="M9.5 14L9.5 6.5" {common}/>'
            f'<path d="M7.2 8.8L9.5 6.5L11.8 8.8" {common}/>'
        )
    elif icon_name == "import":
        body = (
            f'<path d="M12 4.5V14.5" {common}/>'
            f'<path d="M8.5 8L12 4.5L15.5 8" {common}/>'
            f'<path d="M5 17.5H19" {common}/>'
            f'<path d="M7 14.5V17.5H17V14.5" {common}/>'
        )
    elif icon_name == "snap_grid":
        body = (
            f'<path d="M6 6H18M6 12H18M6 18H18M6 6V18M12 6V18M18 6V18" {common}/>'
            f'<circle cx="12" cy="12" r="2.1" {common}/>'
        )
    elif icon_name == "snap_endpoint":
        body = (
            f'<path d="M6 16L18 8" {common}/>'
            f'<circle cx="6" cy="16" r="2.4" {common}/>'
            f'<circle cx="18" cy="8" r="1.5" {common}/>'
        )
    elif icon_name == "snap_midpoint":
        body = (
            f'<path d="M5 15H19" {common}/>'
            f'<circle cx="12" cy="15" r="2.4" {common}/>'
            f'<path d="M12 6V12" {common}/>'
        )
    elif icon_name == "snap_angle":
        body = (
            f'<path d="M6 18V8H16" {common}/>'
            f'<path d="M10 18A4 4 0 0 0 6 14" {common}/>'
            f'<path d="M14.5 9L16 8L15 6.5" {common}/>'
        )
    elif icon_name == "select":
        body = (
            f'<path d="M6 6H10M14 6H18M6 18H10M14 18H18M6 6V10M6 14V18M18 6V10M18 14V18" {common}/>'
            f'<circle cx="12" cy="12" r="2.2" {common}/>'
        )
    elif icon_name == "pan":
        body = (
            f'<path d="M12 5V19M5 12H19" {common}/>'
            f'<path d="M9 8L12 5L15 8M9 16L12 19L15 16M8 9L5 12L8 15M16 9L19 12L16 15" {common}/>'
        )
    elif icon_name == "rotate":
        body = (
            f'<path d="M8 8A6 6 0 1 1 7.5 16.5" {common}/>'
            f'<path d="M8.5 5.5L8 8.8L4.8 8.3" {common}/>'
        )
    elif icon_name == "zoom":
        body = (
            f'<circle cx="10.5" cy="10.5" r="4.5" {common}/>'
            f'<path d="M14 14L19 19" {common}/>'
        )
    elif icon_name == "filter":
        body = (
            f'<path d="M5 5H19L14 12V18L10 19V12L5 5Z" {common}/>'
        )
    elif icon_name == "box_select":
        body = (
            f'<path d="M6 7H18V17H6Z" {common} stroke-dasharray="2.4 2.4"/>'
            f'<path d="M18 17L21 20" {common}/>'
        )
    elif icon_name == "fix":
        body = (
            f'<path d="M6 5V19" {common}/>'
            f'<path d="M9 7V17" {common}/>'
            f'<path d="M11.5 7L7.5 11M11.5 11L7.5 15M11.5 15L7.5 19" {common}/>'
            f'<path d="M13.5 12H18.5" {common}/>'
        )
    elif icon_name == "velocity":
        body = (
            f'<path d="M5 12H18" {common}/>'
            f'<path d="M13.5 7.5L18 12L13.5 16.5" {common}/>'
            f'<path d="M5 8H10M5 16H10" {common}/>'
        )
    elif icon_name == "displacement":
        body = (
            f'<rect x="5.5" y="8" width="6" height="8" rx="1.2" {common}/>'
            f'<path d="M12.5 12H19" {common}/>'
            f'<path d="M15.5 9L19 12L15.5 15" {common}/>'
        )
    elif icon_name == "force":
        body = (
            f'<path d="M5 12H19" {common}/>'
            f'<path d="M14 7L19 12L14 17" {common}/>'
        )
    elif icon_name == "pressure":
        body = (
            f'<path d="M7 7V17" {common}/>'
            f'<path d="M19 8H10M19 12H10M19 16H10" {common}/>'
            f'<path d="M15 6L19 8L15 10M15 10L19 12L15 14M15 14L19 16L15 18" {common}/>'
        )
    elif icon_name == "gravity":
        body = (
            f'<path d="M12 5V18" {common}/>'
            f'<path d="M8.5 14.5L12 18L15.5 14.5" {common}/>'
            f'<path d="M6 5H18" {common}/>'
        )
    elif icon_name == "displacement_result":
        body = (
            f'<path d="M5 18V8" {common}/>'
            f'<path d="M5 18H19" {common}/>'
            f'<path d="M8 14L11 11L14 12L18 8" {common}/>'
            f'<path d="M8 6H15" {common}/>'
        )
    elif icon_name == "stress_result":
        body = (
            f'<circle cx="12" cy="12" r="6.8" {common}/>'
            f'<circle cx="12" cy="12" r="3.4" {common}/>'
            f'<path d="M12 5.2V18.8M5.2 12H18.8" {common}/>'
        )
    elif icon_name == "strain_result":
        body = (
            f'<path d="M7 7L17 7L17 17L7 17Z" {common}/>'
            f'<path d="M9 9L6.5 6.5M15 9L17.5 6.5M9 15L6.5 17.5M15 15L17.5 17.5" {common}/>'
        )
    elif icon_name == "units":
        body = (
            f'<path d="M5 16L19 8" {common}/>'
            f'<path d="M7 15L8 17M10 13.5L11 15.5M13 12L14 14M16 10.5L17 12.5" {common}/>'
        )
    elif icon_name == "inspector":
        body = (
            f'<rect x="5.5" y="5.5" width="13" height="13" rx="1.8" {common}/>'
            f'<path d="M9 9H15M9 12H15M9 15H13" {common}/>'
            f'<circle cx="16.8" cy="15.8" r="2.2" {common}/>'
        )
    elif icon_name == "help":
        body = (
            f'<circle cx="12" cy="12" r="7" {common}/>'
            f'<path d="M9.6 9.2A2.7 2.7 0 0 1 14.2 11C14.2 12.8 12 13 12 14.8" {common}/>'
            f'<path d="M12 17.6H12.1" {common}/>'
        )
    elif icon_name == "connections":
        body = (
            f'<path d="M6 17L12 7L18 17Z" {common}/>'
            f'<path d="M9 12H15" {common}/>'
        )
    elif icon_name == "move_up":
        body = (
            f'<path d="M12 18V6" {common}/>'
            f'<path d="M7.5 10.5L12 6L16.5 10.5" {common}/>'
        )
    elif icon_name == "move_down":
        body = (
            f'<path d="M12 6V18" {common}/>'
            f'<path d="M7.5 13.5L12 18L16.5 13.5" {common}/>'
        )
    elif icon_name == "axis_x":
        body = (
            f'<path d="M5 12H19" {common}/>'
            f'<path d="M14 7L19 12L14 17" {common}/>'
        )
    elif icon_name == "axis_y":
        body = (
            f'<path d="M12 19V5" {common}/>'
            f'<path d="M7 10L12 5L17 10" {common}/>'
        )
    elif icon_name == "axis_z":
        body = (
            f'<path d="M7 17L17 7" {common}/>'
            f'<path d="M13 7H17V11" {common}/>'
        )
    elif icon_name == "magnitude":
        body = (
            f'<path d="M5 12H19" {common}/>'
            f'<path d="M14 7L19 12L14 17" {common}/>'
            f'<path d="M5 7V17M8 7V17" {common}/>'
        )
    else:
        return None

    badge = ""
    if state_name == "completed":
        badge = (
            '<circle cx="18" cy="6" r="4.1" fill="#16a34a" stroke="#ffffff" stroke-width="1.2"/>'
            '<path d="M16.1 6L17.4 7.4L20 4.8" stroke="#ffffff" stroke-width="1.6" '
            'stroke-linecap="round" stroke-linejoin="round" fill="none"/>'
        )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">'
        f"{body}{badge}</svg>"
    )


def get_icon(name, size=24, state=None):
    icon_name = _SVG_ICON_ALIASES.get(str(name or "").lower(), str(name or "").lower())
    cache_key = (icon_name, int(size or 24), str(state or "default").lower())
    if cache_key in _SVG_ICON_CACHE:
        return _SVG_ICON_CACHE[cache_key]
    svg_markup = _svg_icon_markup(icon_name, state=cache_key[2])
    if svg_markup:
        icon = _render_stage_svg_icon(svg_markup, cache_key[1])
        _SVG_ICON_CACHE[cache_key] = icon
        return icon
    if icon_name in _ICON_CACHE:
        return _ICON_CACHE[icon_name]
    builder = _ICON_BUILDERS.get(icon_name)
    if not builder:
        return QIcon()
    draw_fn, color = builder
    icon = _make_icon(draw_fn, color)
    _ICON_CACHE[icon_name] = icon
    return icon


def get_stage_icon(name, size=20, active=False, state=None):
    resolved_state = str(state or ("current" if active else "future")).lower()
    key = (str(name or "geometry").lower(), int(size), resolved_state)
    if key in _STAGE_ICON_CACHE:
        return _STAGE_ICON_CACHE[key]
    icon = get_icon(key[0], size=key[1], state=key[2])
    _STAGE_ICON_CACHE[key] = icon
    return icon
