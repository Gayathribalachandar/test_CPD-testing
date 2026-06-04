import math
import os

from plugin_manager import register_importer


class ImporterError(Exception):
    pass


class ImporterUnavailable(ImporterError):
    pass


class BaseImporter:
    name = "Base"
    extensions = []

    def load(self, path):
        raise NotImplementedError

    def file_filter(self):
        patterns = " ".join(f"*{ext}" for ext in self.extensions)
        return f"{self.name} ({patterns})"


def _arc_points(center, radius, start_deg, end_deg, segments=64):
    if radius <= 0:
        return []
    start = math.radians(start_deg)
    end = math.radians(end_deg)
    return _arc_points_rad(center, radius, start, end, ccw=True, segments=segments)


def _arc_points_rad(center, radius, start_rad, end_rad, ccw=True, segments=64):
    if radius <= 0:
        return []
    start = start_rad
    end = end_rad
    if ccw:
        if end <= start:
            end += 2 * math.pi
        delta = end - start
    else:
        if end >= start:
            end -= 2 * math.pi
        delta = end - start
    count = max(8, int(abs(delta) / (2 * math.pi) * segments))
    pts = []
    for i in range(count + 1):
        t = start + (delta * i / count)
        pts.append((center[0] + radius * math.cos(t), center[1] + radius * math.sin(t)))
    return pts


def _bulge_to_points(p1, p2, bulge):
    if abs(bulge) < 1e-9:
        return [p1, p2]
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    chord = math.hypot(dx, dy)
    if chord == 0:
        return [p1]
    theta = 4.0 * math.atan(bulge)
    sin_half = math.sin(theta / 2.0)
    if abs(sin_half) < 1e-9:
        return [p1, p2]
    radius = chord / (2.0 * sin_half)
    mid_x = (x1 + x2) / 2.0
    mid_y = (y1 + y2) / 2.0
    offset = math.sqrt(max(radius * radius - (chord * 0.5) ** 2, 0.0))
    ux = -dy / chord
    uy = dx / chord
    if bulge < 0:
        ux, uy = -ux, -uy
    cx = mid_x + ux * offset
    cy = mid_y + uy * offset
    start_ang = math.atan2(y1 - cy, x1 - cx)
    end_ang = math.atan2(y2 - cy, x2 - cx)
    return _arc_points_rad((cx, cy), abs(radius), start_ang, end_ang, ccw=bulge > 0)


@register_importer
class DxfImporter(BaseImporter):
    name = "DXF"
    extensions = [".dxf"]

    SUPPORTED_ENTITIES = {
        "LINE",
        "CIRCLE",
        "ARC",
        "ELLIPSE",
        "SPLINE",
        "LWPOLYLINE",
        "POLYLINE",
        "INSERT",
    }

    def load(self, path):
        try:
            import ezdxf
        except ImportError as exc:
            raise ImporterUnavailable(
                "DXF import requires 'ezdxf'. Install it and retry."
            ) from exc

        doc = ezdxf.readfile(path)
        msp = doc.modelspace()
        sketches = []
        skipped_counts = {}

        def add_points(points, closed=False):
            if not points or len(points) < 2:
                return
            pts = [(float(x), float(y)) for x, y in points]
            if closed and pts[0] != pts[-1]:
                pts.append(pts[0])
            sketches.append(pts)

        def add_polyline_points(points, closed=False):
            pts = []
            count = len(points)
            if count < 2:
                return
            for idx in range(count - 1):
                x1, y1, bulge = points[idx]
                x2, y2, _ = points[idx + 1]
                seg_pts = _bulge_to_points((x1, y1), (x2, y2), bulge)
                if pts:
                    seg_pts = seg_pts[1:]
                pts.extend(seg_pts)
            if closed:
                x1, y1, bulge = points[-1]
                x2, y2, _ = points[0]
                seg_pts = _bulge_to_points((x1, y1), (x2, y2), bulge)
                if pts:
                    seg_pts = seg_pts[1:]
                pts.extend(seg_pts)
            add_points(pts, closed=closed)

        def handle_entity(entity):
            etype = entity.dxftype()
            if etype == "LINE":
                start = entity.dxf.start
                end = entity.dxf.end
                add_points([(start.x, start.y), (end.x, end.y)])
            elif etype == "CIRCLE":
                center = entity.dxf.center
                radius = entity.dxf.radius
                pts = _arc_points((center.x, center.y), radius, 0.0, 360.0, segments=128)
                add_points(pts, closed=True)
            elif etype == "ARC":
                center = entity.dxf.center
                radius = entity.dxf.radius
                pts = _arc_points(
                    (center.x, center.y),
                    radius,
                    entity.dxf.start_angle,
                    entity.dxf.end_angle,
                    segments=96,
                )
                add_points(pts)
            elif etype == "ELLIPSE":
                pts = []
                if hasattr(entity, "flattening"):
                    try:
                        pts = [(p.x, p.y) for p in entity.flattening(distance=0.01)]
                    except Exception:
                        pts = []
                if pts:
                    is_full = abs((entity.dxf.end_param - entity.dxf.start_param) - 2 * math.pi) < 1e-6
                    add_points(pts, closed=is_full)
                else:
                    skipped_counts["ELLIPSE"] = skipped_counts.get("ELLIPSE", 0) + 1
            elif etype == "SPLINE":
                pts = []
                if hasattr(entity, "approximate"):
                    try:
                        pts = [(p.x, p.y) for p in entity.approximate(segments=128)]
                    except Exception:
                        pts = []
                if not pts and hasattr(entity, "fit_points"):
                    pts = [(p.x, p.y) for p in entity.fit_points]
                add_points(pts)
            else:
                skipped_counts[etype] = skipped_counts.get(etype, 0) + 1

        def expand_insert(insert_entity, depth=0):
            if depth > 4:
                return
            try:
                exploded = list(insert_entity.virtual_entities())
            except Exception:
                skipped_counts["INSERT"] = skipped_counts.get("INSERT", 0) + 1
                return
            for sub in exploded:
                sub_type = sub.dxftype()
                if sub_type == "LWPOLYLINE":
                    points = [(p[0], p[1], p[2]) for p in sub.get_points("xyb")]
                    add_polyline_points(points, closed=sub.closed)
                elif sub_type == "POLYLINE":
                    points = [
                        (v.dxf.location.x, v.dxf.location.y, getattr(v.dxf, "bulge", 0.0))
                        for v in sub.vertices()
                    ]
                    add_polyline_points(points, closed=sub.is_closed)
                elif sub_type == "INSERT":
                    expand_insert(sub, depth + 1)
                else:
                    handle_entity(sub)

        for entity in msp:
            etype = entity.dxftype()
            if etype == "LWPOLYLINE":
                points = [(p[0], p[1], p[2]) for p in entity.get_points("xyb")]
                add_polyline_points(points, closed=entity.closed)
                continue
            if etype == "POLYLINE":
                points = [
                    (v.dxf.location.x, v.dxf.location.y, getattr(v.dxf, "bulge", 0.0))
                    for v in entity.vertices()
                ]
                add_polyline_points(points, closed=entity.is_closed)
                continue
            if etype == "INSERT":
                expand_insert(entity)
                continue

            handle_entity(entity)

        if not sketches and skipped_counts:
            summary = ", ".join(f"{n}x {t}" for t, n in sorted(skipped_counts.items()))
            raise ImporterError(
                f"DXF parsed but no drawable 2D geometry was extracted. "
                f"Unsupported entities skipped: {summary}. "
                f"Tip: in your CAD tool, EXPLODE blocks/hatches and re-export the DXF."
            )

        return sketches


@register_importer
class SvgImporter(BaseImporter):
    name = "SVG"
    extensions = [".svg"]

    def load(self, path):
        try:
            from svgpathtools import svg2paths2
        except ImportError as exc:
            raise ImporterUnavailable(
                "SVG import requires 'svgpathtools'. Install it and retry."
            ) from exc

        paths, _, _ = svg2paths2(path)
        sketches = []

        for path_obj in paths:
            try:
                length = path_obj.length()
            except Exception:
                length = 0.0
            samples = max(24, int(length / 5.0)) if length else 64
            pts = []
            for i in range(samples + 1):
                t = i / samples
                point = path_obj.point(t)
                pts.append((float(point.real), float(point.imag)))
            if path_obj.isclosed() and pts and pts[0] != pts[-1]:
                pts.append(pts[0])
            if len(pts) >= 2:
                sketches.append(pts)

        return sketches


CAD_3D_EXTENSIONS = (".step", ".stp", ".iges", ".igs", ".stl")
SOLIDWORKS_EXTENSIONS = (".sldprt", ".sldasm", ".slddrw")


def get_importers():
    from plugin_manager import get_importer_classes

    return [cls() for cls in get_importer_classes()]


def get_import_filter():
    importers = get_importers()
    supported_2d = [ext for importer in importers for ext in importer.extensions]
    all_supported = supported_2d + list(CAD_3D_EXTENSIONS)
    supported = " ".join(f"*{ext}" for ext in all_supported)
    filters = [f"All Supported ({supported})"]
    filters.extend(importer.file_filter() for importer in importers)
    cad_pattern = " ".join(f"*{ext}" for ext in CAD_3D_EXTENSIONS)
    filters.append(f"3D CAD ({cad_pattern})")
    return ";;".join(filters)


def import_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in SOLIDWORKS_EXTENSIONS:
        raise ImporterError(
            "SolidWorks files (.sldprt/.sldasm) cannot be read directly. "
            "Open the file in SolidWorks and export it as STEP (.step) or "
            "IGES (.iges), then import that file instead."
        )
    if ext in CAD_3D_EXTENSIONS:
        raise ImporterError(
            f"'{ext}' is a 3D CAD format. Use 'File > Import 3D CAD' (requires "
            "the project to be in 3D mode and the CAD kernel to be installed)."
        )
    for importer in get_importers():
        if ext in importer.extensions:
            return importer.load(path)
    raise ImporterError(f"Unsupported import format: {ext or 'unknown'}")
