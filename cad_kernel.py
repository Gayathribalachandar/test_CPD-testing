import importlib
import math
import os


class CadKernel:
    def __init__(self):
        self.backend = None
        self._prefix = None
        self._mods = {}
        self._load_backend()

    def _import_module(self, prefix, name):
        return importlib.import_module(f"{prefix}.{name}")

    def _ensure_module(self, name):
        if name in self._mods:
            return self._mods[name]
        if not self._prefix:
            return None
        try:
            mod = self._import_module(self._prefix, name)
        except Exception:
            return None
        self._mods[name] = mod
        return mod

    def _cast_topods(self, shape, kind):
        if shape is None:
            return None
        topods_mod = self._mods.get("TopoDS")
        if topods_mod is None:
            return shape
        cast_name = str(kind).strip()
        candidates = []
        topods_ns = getattr(topods_mod, "topods", None)
        if topods_ns is not None:
            candidates.append(getattr(topods_ns, cast_name, None))
        candidates.append(getattr(topods_mod, f"topods_{cast_name}", None))
        candidates.append(getattr(topods_mod, f"{cast_name}_s", None))
        topo_cls = getattr(topods_mod, "TopoDS", None)
        if topo_cls is not None:
            candidates.append(getattr(topo_cls, f"{cast_name}_s", None))
            candidates.append(getattr(topo_cls, cast_name, None))
        candidates.append(getattr(topods_mod, cast_name, None))
        for caster in candidates:
            if not callable(caster):
                continue
            try:
                out = caster(shape)
                if out is not None:
                    return out
            except Exception:
                continue
        return shape

    @staticmethod
    def _shape_is_null(shape):
        if shape is None:
            return True
        try:
            return bool(shape.IsNull())
        except Exception:
            return False

    @staticmethod
    def _curve_type_name(curve_type):
        text = str(curve_type)
        if "." in text:
            text = text.split(".")[-1]
        if text.startswith("GeomAbs_"):
            text = text[len("GeomAbs_") :]
        return text.lower()

    def _load_backend(self):
        for prefix in ("OCC.Core", "OCP"):
            try:
                gp = self._import_module(prefix, "gp")
                brep_builder = self._import_module(prefix, "BRepBuilderAPI")
                brep_prim = self._import_module(prefix, "BRepPrimAPI")
                brep_algo = self._import_module(prefix, "BRepAlgoAPI")
                brep_mesh = self._import_module(prefix, "BRepMesh")
                top_exp = self._import_module(prefix, "TopExp")
                top_abs = self._import_module(prefix, "TopAbs")
                brep = self._import_module(prefix, "BRep")
                top_loc = self._import_module(prefix, "TopLoc")
                topo_ds = self._import_module(prefix, "TopoDS")
                step = self._import_module(prefix, "STEPControl")
                iges = self._import_module(prefix, "IGESControl")
                stl = self._import_module(prefix, "StlAPI")
            except Exception:
                continue
            self.backend = "OCC" if prefix == "OCC.Core" else "OCP"
            self._prefix = prefix
            self._mods = {
                "gp": gp,
                "BRepBuilderAPI": brep_builder,
                "BRepPrimAPI": brep_prim,
                "BRepAlgoAPI": brep_algo,
                "BRepMesh": brep_mesh,
                "TopExp": top_exp,
                "TopAbs": top_abs,
                "BRep": brep,
                "TopLoc": top_loc,
                "TopoDS": topo_ds,
                "STEPControl": step,
                "IGESControl": iges,
                "StlAPI": stl,
            }
            return

    def available(self):
        return self.backend is not None

    def _make_wire(self, points):
        if not points or len(points) < 3:
            return None
        gp = self._mods["gp"]
        brep_builder = self._mods["BRepBuilderAPI"]
        make_poly = brep_builder.BRepBuilderAPI_MakePolygon()
        for x, y in points:
            make_poly.Add(gp.gp_Pnt(float(x), float(y), 0.0))
        if points[0] != points[-1]:
            make_poly.Add(gp.gp_Pnt(float(points[0][0]), float(points[0][1]), 0.0))
        make_poly.Close()
        return make_poly.Wire()

    def face_from_polygon(self, exterior, holes=None):
        if not self.available():
            return None
        if not exterior or len(exterior) < 3:
            return None
        brep_builder = self._mods["BRepBuilderAPI"]
        outer_wire = self._make_wire(exterior)
        if outer_wire is None:
            return None
        face_maker = brep_builder.BRepBuilderAPI_MakeFace(outer_wire, True)
        if holes:
            for hole in holes:
                hole_wire = self._make_wire(hole)
                if hole_wire is not None:
                    face_maker.Add(hole_wire)
        return face_maker.Face()

    def extrude(self, face, height):
        if not self.available() or face is None:
            return None
        gp = self._mods["gp"]
        brep_prim = self._mods["BRepPrimAPI"]
        vec = gp.gp_Vec(0.0, 0.0, float(height))
        return brep_prim.BRepPrimAPI_MakePrism(face, vec).Shape()

    def revolve(self, face, axis_origin=(0.0, 0.0, 0.0), axis_dir=(0.0, 1.0, 0.0), angle_deg=360.0):
        if not self.available() or face is None:
            return None
        gp = self._mods["gp"]
        brep_prim = self._mods["BRepPrimAPI"]
        ox, oy, oz = [float(v) for v in axis_origin]
        dx, dy, dz = [float(v) for v in axis_dir]
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length <= 1e-12:
            dx, dy, dz = 0.0, 1.0, 0.0
        else:
            dx, dy, dz = dx / length, dy / length, dz / length
        axis = gp.gp_Ax1(gp.gp_Pnt(ox, oy, oz), gp.gp_Dir(dx, dy, dz))
        angle_rad = math.radians(float(angle_deg))
        maker = None
        try:
            maker = brep_prim.BRepPrimAPI_MakeRevol(face, axis, angle_rad, True)
        except TypeError:
            try:
                maker = brep_prim.BRepPrimAPI_MakeRevol(face, axis, angle_rad)
            except Exception:
                maker = brep_prim.BRepPrimAPI_MakeRevol(face, axis)
        except Exception:
            try:
                maker = brep_prim.BRepPrimAPI_MakeRevol(face, axis)
            except Exception:
                maker = None
        if maker is None:
            return None
        try:
            return maker.Shape()
        except Exception:
            return None

    def boolean(self, shape_a, shape_b, op):
        if not self.available() or shape_a is None or shape_b is None:
            return None
        brep_algo = self._mods["BRepAlgoAPI"]
        op = str(op).lower()
        if op in ("add", "fuse", "union"):
            return brep_algo.BRepAlgoAPI_Fuse(shape_a, shape_b).Shape()
        if op in ("cut", "subtract", "difference"):
            return brep_algo.BRepAlgoAPI_Cut(shape_a, shape_b).Shape()
        if op in ("intersect", "common"):
            return brep_algo.BRepAlgoAPI_Common(shape_a, shape_b).Shape()
        return None

    def import_shape(self, path):
        if not self.available():
            return None
        ext = os.path.splitext(path)[1].lower()
        if ext in (".step", ".stp"):
            reader = self._mods["STEPControl"].STEPControl_Reader()
            status = reader.ReadFile(path)
            if status != 1:
                return None
            reader.TransferRoots()
            return reader.OneShape()
        if ext in (".iges", ".igs"):
            reader = self._mods["IGESControl"].IGESControl_Reader()
            status = reader.ReadFile(path)
            if status != 1:
                return None
            reader.TransferRoots()
            return reader.OneShape()
        if ext in (".stl",):
            shape = self._mods["TopoDS"].TopoDS_Shape()
            reader = self._mods["StlAPI"].StlAPI_Reader()
            reader.Read(shape, path)
            return shape
        return None

    def tessellate(self, shape, linear_deflection=0.5, angular_deflection=0.5, with_face_ids=False):
        if not self.available() or shape is None:
            if with_face_ids:
                return None, None, None
            return None, None
        brep_mesh = self._mods["BRepMesh"]
        top_exp = self._mods["TopExp"]
        top_abs = self._mods["TopAbs"]
        brep = self._mods["BRep"]
        mesh = brep_mesh.BRepMesh_IncrementalMesh(
            shape, float(linear_deflection), False, float(angular_deflection), True
        )
        mesh.Perform()
        explorer = top_exp.TopExp_Explorer(shape, top_abs.TopAbs_FACE)
        nodes = []
        faces = []
        face_ids = []
        cad_face_idx = 0
        while explorer.More():
            face = explorer.Current()
            topods_mod = self._mods.get("TopoDS")
            if topods_mod is not None:
                try:
                    if hasattr(topods_mod, "topods") and hasattr(topods_mod.topods, "Face"):
                        face = topods_mod.topods.Face(face)
                    elif hasattr(topods_mod, "topods_Face"):
                        face = topods_mod.topods_Face(face)
                    elif hasattr(topods_mod, "Face_s"):
                        face = topods_mod.Face_s(face)
                    elif hasattr(topods_mod, "TopoDS") and hasattr(topods_mod.TopoDS, "Face_s"):
                        face = topods_mod.TopoDS.Face_s(face)
                    elif hasattr(topods_mod, "TopoDS") and hasattr(topods_mod.TopoDS, "Face"):
                        face = topods_mod.TopoDS.Face(face)
                except Exception:
                    explorer.Next()
                    continue
            if "Face" not in type(face).__name__:
                explorer.Next()
                continue
            loc = self._mods["TopLoc"].TopLoc_Location()
            triangulation = None
            tool = brep.BRep_Tool
            if hasattr(tool, "Triangulation_s"):
                try:
                    triangulation = tool.Triangulation_s(face, loc, 0)
                except TypeError:
                    triangulation = tool.Triangulation_s(face, loc)
            elif hasattr(tool, "Triangulation"):
                triangulation = tool.Triangulation(face, loc)
            if triangulation is None:
                explorer.Next()
                continue
            trsf = loc.Transformation()
            offset = len(nodes)
            if hasattr(triangulation, "Nodes") and hasattr(triangulation, "Triangles"):
                tri_nodes = triangulation.Nodes()
                tri_tris = triangulation.Triangles()
                for i in range(1, tri_nodes.Size() + 1):
                    pnt = tri_nodes.Value(i).Transformed(trsf)
                    nodes.append((pnt.X(), pnt.Y(), pnt.Z()))
                for i in range(1, tri_tris.Size() + 1):
                    tri = tri_tris.Value(i)
                    n1, n2, n3 = tri.Get()
                    faces.append((offset + n1 - 1, offset + n2 - 1, offset + n3 - 1))
                    face_ids.append(cad_face_idx)
            else:
                nb_nodes = triangulation.NbNodes()
                for i in range(1, nb_nodes + 1):
                    pnt = triangulation.Node(i)
                    if trsf is not None:
                        pnt = pnt.Transformed(trsf)
                    nodes.append((pnt.X(), pnt.Y(), pnt.Z()))
                nb_tris = triangulation.NbTriangles()
                for i in range(1, nb_tris + 1):
                    tri = triangulation.Triangle(i)
                    n1, n2, n3 = tri.Get()
                    faces.append((offset + n1 - 1, offset + n2 - 1, offset + n3 - 1))
                    face_ids.append(cad_face_idx)
            cad_face_idx += 1
            explorer.Next()
        if with_face_ids:
            return nodes, faces, face_ids
        return nodes, faces

    def extract_topology(self, shape, edge_samples=64):
        if not self.available() or shape is None:
            return {"vertices": [], "edges": []}
        top_exp = self._mods.get("TopExp")
        top_abs = self._mods.get("TopAbs")
        brep = self._mods.get("BRep")
        top_tools = self._ensure_module("TopTools")
        brep_adaptor = self._ensure_module("BRepAdaptor")
        if (
            top_exp is None
            or top_abs is None
            or brep is None
            or top_tools is None
            or brep_adaptor is None
        ):
            return {"vertices": [], "edges": []}
        try:
            vertex_map = top_tools.TopTools_IndexedMapOfShape()
            edge_map = top_tools.TopTools_IndexedMapOfShape()
        except Exception:
            return {"vertices": [], "edges": []}
        mapper = getattr(top_exp.TopExp, "MapShapes_s", None)
        if mapper is None:
            mapper = getattr(top_exp.TopExp, "MapShapes", None)
        if mapper is None:
            return {"vertices": [], "edges": []}
        try:
            mapper(shape, top_abs.TopAbs_VERTEX, vertex_map)
            mapper(shape, top_abs.TopAbs_EDGE, edge_map)
        except Exception:
            return {"vertices": [], "edges": []}

        tool = brep.BRep_Tool
        vertices = []
        for idx in range(1, int(vertex_map.Extent()) + 1):
            try:
                vertex = self._cast_topods(vertex_map.FindKey(idx), "Vertex")
                if self._shape_is_null(vertex):
                    continue
                if hasattr(tool, "Pnt_s"):
                    pnt = tool.Pnt_s(vertex)
                else:
                    pnt = tool.Pnt(vertex)
                vertices.append(
                    {
                        "id": int(idx - 1),
                        "point": (float(pnt.X()), float(pnt.Y()), float(pnt.Z())),
                    }
                )
            except Exception:
                continue

        first_vertex_fn = getattr(top_exp.TopExp, "FirstVertex_s", None)
        if first_vertex_fn is None:
            first_vertex_fn = getattr(top_exp.TopExp, "FirstVertex", None)
        last_vertex_fn = getattr(top_exp.TopExp, "LastVertex_s", None)
        if last_vertex_fn is None:
            last_vertex_fn = getattr(top_exp.TopExp, "LastVertex", None)

        edges = []
        edge_samples = max(8, int(edge_samples))
        for idx in range(1, int(edge_map.Extent()) + 1):
            try:
                edge = self._cast_topods(edge_map.FindKey(idx), "Edge")
                if self._shape_is_null(edge):
                    continue
                curve = brep_adaptor.BRepAdaptor_Curve(edge)
                u0 = float(curve.FirstParameter())
                u1 = float(curve.LastParameter())
                if not math.isfinite(u0) or not math.isfinite(u1):
                    continue
                curve_type = self._curve_type_name(curve.GetType())
                is_closed = bool(curve.IsClosed())
                if curve_type == "line":
                    sample_count = 2
                elif is_closed:
                    sample_count = max(24, edge_samples)
                else:
                    sample_count = edge_samples
                points = []
                if is_closed and sample_count > 2:
                    denom = float(sample_count)
                else:
                    denom = float(max(1, sample_count - 1))
                for i in range(sample_count):
                    t = float(i) / denom
                    u = u0 + (u1 - u0) * t
                    pnt = curve.Value(u)
                    points.append((float(pnt.X()), float(pnt.Y()), float(pnt.Z())))
                first_vid = -1
                last_vid = -1
                if callable(first_vertex_fn):
                    try:
                        try:
                            first_vertex = first_vertex_fn(edge, True)
                        except TypeError:
                            first_vertex = first_vertex_fn(edge)
                        first_vid = int(vertex_map.FindIndex(first_vertex)) - 1
                    except Exception:
                        first_vid = -1
                if callable(last_vertex_fn):
                    try:
                        try:
                            last_vertex = last_vertex_fn(edge, True)
                        except TypeError:
                            last_vertex = last_vertex_fn(edge)
                        last_vid = int(vertex_map.FindIndex(last_vertex)) - 1
                    except Exception:
                        last_vid = -1
                edges.append(
                    {
                        "id": int(idx - 1),
                        "curve_type": curve_type,
                        "is_closed": is_closed,
                        "vertex_ids": (int(first_vid), int(last_vid)),
                        "points": points,
                    }
                )
            except Exception:
                continue
        return {"vertices": vertices, "edges": edges}

    def make_box(self, center, size):
        if not self.available():
            return None
        gp = self._mods["gp"]
        brep_prim = self._mods["BRepPrimAPI"]
        cx, cy, cz = [float(v) for v in center]
        sx, sy, sz = [float(v) for v in size]
        corner = gp.gp_Pnt(cx - sx * 0.5, cy - sy * 0.5, cz - sz * 0.5)
        return brep_prim.BRepPrimAPI_MakeBox(corner, sx, sy, sz).Shape()

    def make_cylinder(self, center, radius, height):
        if not self.available():
            return None
        gp = self._mods["gp"]
        brep_prim = self._mods["BRepPrimAPI"]
        cx, cy, cz = [float(v) for v in center]
        axis = gp.gp_Ax2(
            gp.gp_Pnt(cx, cy, cz - float(height) * 0.5),
            gp.gp_Dir(0.0, 0.0, 1.0),
        )
        return brep_prim.BRepPrimAPI_MakeCylinder(
            axis, float(radius), float(height)
        ).Shape()

    def make_sphere(self, center, radius):
        if not self.available():
            return None
        gp = self._mods["gp"]
        brep_prim = self._mods["BRepPrimAPI"]
        cx, cy, cz = [float(v) for v in center]
        pnt = gp.gp_Pnt(cx, cy, cz)
        return brep_prim.BRepPrimAPI_MakeSphere(pnt, float(radius)).Shape()

    def make_cone(self, center, radius_base, radius_top, height):
        if not self.available():
            return None
        gp = self._mods["gp"]
        brep_prim = self._mods["BRepPrimAPI"]
        cx, cy, cz = [float(v) for v in center]
        axis = gp.gp_Ax2(
            gp.gp_Pnt(cx, cy, cz - float(height) * 0.5),
            gp.gp_Dir(0.0, 0.0, 1.0),
        )
        return brep_prim.BRepPrimAPI_MakeCone(
            axis,
            float(radius_base),
            float(radius_top),
            float(height),
        ).Shape()

    def make_torus(self, center, major_radius, tube_radius):
        if not self.available():
            return None
        gp = self._mods["gp"]
        brep_prim = self._mods["BRepPrimAPI"]
        cx, cy, cz = [float(v) for v in center]
        axis = gp.gp_Ax2(gp.gp_Pnt(cx, cy, cz), gp.gp_Dir(0, 0, 1))
        return brep_prim.BRepPrimAPI_MakeTorus(
            axis, float(major_radius), float(tube_radius)
        ).Shape()

    def rotate(self, shape, center, rotation):
        if not self.available() or shape is None:
            return shape
        rx, ry, rz = rotation or (0.0, 0.0, 0.0)
        if abs(rx) < 1e-9 and abs(ry) < 1e-9 and abs(rz) < 1e-9:
            return shape
        gp = self._mods["gp"]
        brep_builder = self._mods["BRepBuilderAPI"]
        cx, cy, cz = [float(v) for v in center]
        origin = gp.gp_Pnt(cx, cy, cz)
        axes = [
            (rx, gp.gp_Dir(1, 0, 0)),
            (ry, gp.gp_Dir(0, 1, 0)),
            (rz, gp.gp_Dir(0, 0, 1)),
        ]
        out = shape
        for angle, direction in axes:
            if abs(angle) < 1e-9:
                continue
            trsf = gp.gp_Trsf()
            trsf.SetRotation(gp.gp_Ax1(origin, direction), math.radians(float(angle)))
            try:
                out = brep_builder.BRepBuilderAPI_Transform(out, trsf, True).Shape()
            except TypeError:
                out = brep_builder.BRepBuilderAPI_Transform(out, trsf).Shape()
        return out

    def translate(self, shape, offset):
        if not self.available() or shape is None:
            return shape
        gp = self._mods["gp"]
        brep_builder = self._mods["BRepBuilderAPI"]
        dx, dy, dz = [float(v) for v in offset]
        if abs(dx) < 1e-9 and abs(dy) < 1e-9 and abs(dz) < 1e-9:
            return shape
        trsf = gp.gp_Trsf()
        trsf.SetTranslation(gp.gp_Vec(dx, dy, dz))
        try:
            return brep_builder.BRepBuilderAPI_Transform(shape, trsf, True).Shape()
        except TypeError:
            return brep_builder.BRepBuilderAPI_Transform(shape, trsf).Shape()
