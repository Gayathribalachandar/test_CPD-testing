from dataclasses import dataclass, field

from material_registry import (
    infer_behavior_from_mat_type,
    legacy_mat_type_for_behavior,
    normalize_material_behavior,
    normalize_material_damage,
    normalize_material_properties,
    normalize_material_symmetry,
)


FIELD_DISTRIBUTION_PROPERTY_KEYS = ("E", "nu", "rho", "fail_SE", "c")


@dataclass
class MeshSizingPolicy:
    """Adaptive sizing controls for the gmsh 2D surface mesher.

    Auto-defaults when not supplied:
      h_feature        = h_bulk * 0.7                          (subtle refinement, ~30% smaller than bulk)
      transition_width = max(15 * h_feature, 10 * h_bulk)      (very wide gradient)

    The auto-defaults prioritise an *imperceptible* boundary layer: elements
    near boundaries are only ~30% smaller than bulk, and the gradient is
    spread over a long distance so there's no visible "ring" of refinement.

    Combined with the MathEval exponential size field in gmsh_mesher, the
    resulting mesh shows continuous asymptotic growth rather than a clamped
    transition. For aggressive refinement (e.g. resolving stress
    concentrations), set h_feature explicitly via the Mesh panel — values
    around h_bulk / 4 to h_bulk / 8 are typical.
    """

    h_bulk: float
    h_feature: float = 0.0
    transition_width: float = 0.0

    def __post_init__(self):
        try:
            self.h_bulk = float(self.h_bulk)
        except Exception:
            raise ValueError("h_bulk must be a positive number")
        if self.h_bulk <= 0.0:
            raise ValueError("h_bulk must be > 0")
        if self.h_feature is None or float(self.h_feature) <= 0.0:
            self.h_feature = self.h_bulk * 0.7
        else:
            self.h_feature = float(self.h_feature)
        if self.transition_width is None or float(self.transition_width) <= 0.0:
            self.transition_width = max(15.0 * self.h_feature, 10.0 * self.h_bulk)
        else:
            self.transition_width = float(self.transition_width)

    @classmethod
    def from_dict(cls, data):
        if data is None:
            return None
        if isinstance(data, cls):
            return data
        try:
            return cls(
                h_bulk=float(data.get("h_bulk")),
                h_feature=float(data.get("h_feature", 0.0) or 0.0),
                transition_width=float(data.get("transition_width", 0.0) or 0.0),
            )
        except Exception:
            return None

    def to_dict(self):
        return {
            "h_bulk": float(self.h_bulk),
            "h_feature": float(self.h_feature),
            "transition_width": float(self.transition_width),
        }


@dataclass
class CustomMeshZone:
    """User-drawn refinement zone on the Results-page canvas.

    points: list of (x, y) polygon vertices in world coordinates (sketch units).
    approx_node_count: target node count gmsh should aim for inside the zone.
    """

    points: list = field(default_factory=list)
    approx_node_count: int = 0

    def __post_init__(self):
        cleaned = []
        for p in self.points or []:
            try:
                x, y = float(p[0]), float(p[1])
            except Exception:
                continue
            cleaned.append((x, y))
        self.points = cleaned
        try:
            self.approx_node_count = int(self.approx_node_count)
        except Exception:
            self.approx_node_count = 0
        if self.approx_node_count < 0:
            self.approx_node_count = 0

    def bbox(self):
        if not self.points:
            return None
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (min(xs), min(ys), max(xs), max(ys))

    def bbox_area(self):
        bb = self.bbox()
        if bb is None:
            return 0.0
        return max(bb[2] - bb[0], 0.0) * max(bb[3] - bb[1], 0.0)

    def polygon_area(self):
        """Signed-area shoelace, returned as absolute value. Uses the actual
        polygon outline (not the bbox), so derived sizing honors the drawn shape."""
        pts = self.points
        if len(pts) < 3:
            return 0.0
        s = 0.0
        n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            s += x1 * y2 - x2 * y1
        return abs(s) * 0.5

    def derived_mesh_size(self):
        # Convert approx_node_count to a target element edge length via
        # the simple area-per-node heuristic: each node occupies ~ area/N,
        # so the characteristic edge length is sqrt(area/N).
        # Use the polygon's actual area (not bbox), so node-count targets
        # are honored against the shape the user actually drew.
        area = self.polygon_area()
        if area <= 0.0:
            area = self.bbox_area()
        if area <= 0.0 or self.approx_node_count <= 0:
            return 0.0
        import math as _math
        return float(_math.sqrt(area / float(self.approx_node_count)))

    @classmethod
    def from_dict(cls, data):
        if data is None:
            return None
        if isinstance(data, cls):
            return data
        try:
            pts = data.get("points", []) or []
            n = data.get("approx_node_count", 0)
            return cls(points=list(pts), approx_node_count=int(n or 0))
        except Exception:
            return None

    def to_dict(self):
        return {
            "points": [(float(x), float(y)) for x, y in self.points],
            "approx_node_count": int(self.approx_node_count),
        }


@dataclass
class EdgeSeed:
    """Local edge seed (Abaqus-style Local Seeds dialog).

    Each seed targets one or more edges of the model boundary, identified by
    their endpoint coordinates so the seed survives mesh regeneration where
    gmsh curve tags change.

    edge_refs: list of {part_id: int, start: (x, y), end: (x, y)} dicts —
        each entry identifies one boundary segment. Multi-edge seeds carry
        more than one entry; single-edge seeds carry exactly one.
    method: "by_size" or "by_number".
    bias: "none", "single", or "double".
    flip_bias: swap which end of the edge is the fine end.
    element_size: target element size when method=by_size and bias=none.
    min_size, max_size: end sizes when bias is single or double.
    seed_count: number of edge subdivisions when method=by_number.
    bias_ratio: max_size / min_size ratio when method=by_number and bias != none.
    curvature_control, max_deviation_factor, min_size_factor: curvature
        refinement options (only used with method=by_size, bias=none).
    set_name: optional friendly label for the seed (the "Create set with name"
        field in Abaqus).
    """

    edge_refs: list = field(default_factory=list)
    method: str = "by_size"  # "by_size" | "by_number"
    bias: str = "none"  # "none" | "single" | "double"
    flip_bias: bool = False
    element_size: float = 0.0
    min_size: float = 0.0
    max_size: float = 0.0
    seed_count: int = 0
    bias_ratio: float = 1.0
    curvature_control: bool = False
    max_deviation_factor: float = 0.1
    min_size_factor: float = 0.1  # 0 < min < 1; 0.1 is the gmsh default ratio
    set_name: str = ""
    # Constraints tab fields.
    propagate_to_neighbors: bool = False  # apply this size to edges sharing endpoints

    def __post_init__(self):
        # Normalize edge_refs into a clean list of dicts with float coords.
        cleaned = []
        for ref in self.edge_refs or []:
            if not isinstance(ref, dict):
                continue
            try:
                pid = int(ref.get("part_id"))
                sx, sy = float(ref["start"][0]), float(ref["start"][1])
                ex, ey = float(ref["end"][0]), float(ref["end"][1])
            except Exception:
                continue
            cleaned.append({"part_id": pid, "start": (sx, sy), "end": (ex, ey)})
        self.edge_refs = cleaned

        method = str(self.method or "by_size").lower()
        self.method = method if method in ("by_size", "by_number") else "by_size"
        bias = str(self.bias or "none").lower()
        self.bias = bias if bias in ("none", "single", "double") else "none"

        try:
            self.element_size = float(self.element_size or 0.0)
        except Exception:
            self.element_size = 0.0
        try:
            self.min_size = float(self.min_size or 0.0)
        except Exception:
            self.min_size = 0.0
        try:
            self.max_size = float(self.max_size or 0.0)
        except Exception:
            self.max_size = 0.0
        try:
            self.seed_count = int(self.seed_count or 0)
        except Exception:
            self.seed_count = 0
        try:
            self.bias_ratio = float(self.bias_ratio or 1.0)
        except Exception:
            self.bias_ratio = 1.0
        self.curvature_control = bool(self.curvature_control)
        try:
            self.max_deviation_factor = float(self.max_deviation_factor or 0.1)
        except Exception:
            self.max_deviation_factor = 0.1
        try:
            self.min_size_factor = float(self.min_size_factor or 0.1)
        except Exception:
            self.min_size_factor = 0.1
        self.set_name = str(self.set_name or "")
        self.propagate_to_neighbors = bool(self.propagate_to_neighbors)

    def is_valid(self) -> bool:
        if not self.edge_refs:
            return False
        if self.method == "by_size":
            if self.bias == "none":
                return self.element_size > 0.0
            return self.min_size > 0.0 and self.max_size > 0.0 and self.min_size != self.max_size
        # by_number
        if self.seed_count <= 0:
            return False
        if self.bias != "none" and self.bias_ratio <= 0.0:
            return False
        return True

    def effective_min_max(self):
        """Return (h_fine, h_coarse) honoring flip_bias. Only meaningful when
        the seed has a bias and a min/max size."""
        if self.method == "by_size" and self.bias != "none":
            lo, hi = self.min_size, self.max_size
        else:
            return None
        if lo > hi:
            lo, hi = hi, lo
        if self.flip_bias:
            return hi, lo
        return lo, hi

    @classmethod
    def from_dict(cls, data):
        if data is None:
            return None
        if isinstance(data, cls):
            return data
        try:
            refs_in = data.get("edge_refs", []) or []
            refs = []
            for r in refs_in:
                if not isinstance(r, dict):
                    continue
                refs.append(
                    {
                        "part_id": int(r.get("part_id")),
                        "start": (float(r["start"][0]), float(r["start"][1])),
                        "end": (float(r["end"][0]), float(r["end"][1])),
                    }
                )
            return cls(
                edge_refs=refs,
                method=str(data.get("method", "by_size")),
                bias=str(data.get("bias", "none")),
                flip_bias=bool(data.get("flip_bias", False)),
                element_size=float(data.get("element_size", 0.0) or 0.0),
                min_size=float(data.get("min_size", 0.0) or 0.0),
                max_size=float(data.get("max_size", 0.0) or 0.0),
                seed_count=int(data.get("seed_count", 0) or 0),
                bias_ratio=float(data.get("bias_ratio", 1.0) or 1.0),
                curvature_control=bool(data.get("curvature_control", False)),
                max_deviation_factor=float(data.get("max_deviation_factor", 0.1) or 0.1),
                min_size_factor=float(data.get("min_size_factor", 0.1) or 0.1),
                set_name=str(data.get("set_name", "") or ""),
                propagate_to_neighbors=bool(data.get("propagate_to_neighbors", False)),
            )
        except Exception:
            return None

    def to_dict(self):
        return {
            "edge_refs": [
                {
                    "part_id": int(r["part_id"]),
                    "start": (float(r["start"][0]), float(r["start"][1])),
                    "end": (float(r["end"][0]), float(r["end"][1])),
                }
                for r in self.edge_refs
            ],
            "method": self.method,
            "bias": self.bias,
            "flip_bias": bool(self.flip_bias),
            "element_size": float(self.element_size),
            "min_size": float(self.min_size),
            "max_size": float(self.max_size),
            "seed_count": int(self.seed_count),
            "bias_ratio": float(self.bias_ratio),
            "curvature_control": bool(self.curvature_control),
            "max_deviation_factor": float(self.max_deviation_factor),
            "min_size_factor": float(self.min_size_factor),
            "set_name": str(self.set_name),
            "propagate_to_neighbors": bool(self.propagate_to_neighbors),
        }


@dataclass
class VertexSeed:
    """Point-anchored refinement (Abaqus 'seed at vertex' / stress-concentration).

    The mesh shrinks to `target_size` at the picked point and grows back to
    the global bulk size over `influence_radius`. Useful for refining around
    corners, hole edges, notches, and other stress-concentration features.

    point: (x, y) world coordinates of the vertex.
    target_size: element edge length AT the vertex.
    influence_radius: distance over which the size returns to h_bulk.
    part_id: which part the vertex belongs to (informational; the gmsh
        backend matches by coordinate).
    set_name: optional friendly label (Abaqus-style "Create set with name").
    """

    point: tuple = (0.0, 0.0)
    target_size: float = 0.0
    influence_radius: float = 0.0
    part_id: int = 0
    set_name: str = ""

    def __post_init__(self):
        try:
            self.point = (float(self.point[0]), float(self.point[1]))
        except Exception:
            self.point = (0.0, 0.0)
        try:
            self.target_size = float(self.target_size or 0.0)
        except Exception:
            self.target_size = 0.0
        try:
            self.influence_radius = float(self.influence_radius or 0.0)
        except Exception:
            self.influence_radius = 0.0
        try:
            self.part_id = int(self.part_id or 0)
        except Exception:
            self.part_id = 0
        self.set_name = str(self.set_name or "")

    def is_valid(self) -> bool:
        return self.target_size > 0.0 and self.influence_radius > 0.0

    @classmethod
    def from_dict(cls, data):
        if data is None:
            return None
        if isinstance(data, cls):
            return data
        try:
            pt = data.get("point", (0, 0))
            return cls(
                point=(float(pt[0]), float(pt[1])),
                target_size=float(data.get("target_size", 0.0) or 0.0),
                influence_radius=float(data.get("influence_radius", 0.0) or 0.0),
                part_id=int(data.get("part_id", 0) or 0),
                set_name=str(data.get("set_name", "") or ""),
            )
        except Exception:
            return None

    def to_dict(self):
        return {
            "point": (float(self.point[0]), float(self.point[1])),
            "target_size": float(self.target_size),
            "influence_radius": float(self.influence_radius),
            "part_id": int(self.part_id),
            "set_name": str(self.set_name),
        }


@dataclass
class BoundaryLayerSeed:
    """Inflation / boundary-layer seed (Abaqus 'Boundary Layer Seeding').

    Generates stacked thin elements PARALLEL to one or more edges, growing
    geometrically into the interior. Standard CFD / contact / heat-transfer
    technique to resolve gradients normal to a wall.

    edge_refs:        list of {part_id, start, end} — same shape as EdgeSeed.
    first_layer_size: thickness of the first (thinnest) layer adjacent to the edge.
    growth_ratio:     geometric growth factor between consecutive layers (>= 1).
    num_layers:       how many layers to extrude.
    quads:            emit quad elements in the boundary-layer region (else triangles).
    max_thickness:    optional cap on total inflation distance; 0 = no cap.
    set_name:         optional friendly label.

    Total inflation thickness = first_layer_size * (1 - r^N) / (1 - r) for r != 1
    where r = growth_ratio, N = num_layers.
    """

    edge_refs: list = field(default_factory=list)
    first_layer_size: float = 0.0
    growth_ratio: float = 1.2
    num_layers: int = 5
    quads: bool = False
    max_thickness: float = 0.0
    set_name: str = ""

    def __post_init__(self):
        cleaned = []
        for ref in self.edge_refs or []:
            if not isinstance(ref, dict):
                continue
            try:
                pid = int(ref.get("part_id"))
                sx, sy = float(ref["start"][0]), float(ref["start"][1])
                ex, ey = float(ref["end"][0]), float(ref["end"][1])
            except Exception:
                continue
            cleaned.append({"part_id": pid, "start": (sx, sy), "end": (ex, ey)})
        self.edge_refs = cleaned
        try:
            self.first_layer_size = float(self.first_layer_size or 0.0)
        except Exception:
            self.first_layer_size = 0.0
        try:
            self.growth_ratio = max(1.0, float(self.growth_ratio or 1.0))
        except Exception:
            self.growth_ratio = 1.0
        try:
            self.num_layers = max(1, int(self.num_layers or 1))
        except Exception:
            self.num_layers = 1
        self.quads = bool(self.quads)
        try:
            self.max_thickness = max(0.0, float(self.max_thickness or 0.0))
        except Exception:
            self.max_thickness = 0.0
        self.set_name = str(self.set_name or "")

    def is_valid(self) -> bool:
        return bool(self.edge_refs) and self.first_layer_size > 0.0 and self.num_layers >= 1

    def total_thickness(self) -> float:
        if self.first_layer_size <= 0 or self.num_layers <= 0:
            return 0.0
        r = self.growth_ratio
        N = self.num_layers
        if abs(r - 1.0) < 1e-9:
            t = self.first_layer_size * N
        else:
            t = self.first_layer_size * (1.0 - r ** N) / (1.0 - r)
        if self.max_thickness > 0:
            t = min(t, self.max_thickness)
        return t

    @classmethod
    def from_dict(cls, data):
        if data is None:
            return None
        if isinstance(data, cls):
            return data
        try:
            refs_in = data.get("edge_refs", []) or []
            refs = []
            for r in refs_in:
                if not isinstance(r, dict):
                    continue
                refs.append({
                    "part_id": int(r.get("part_id")),
                    "start": (float(r["start"][0]), float(r["start"][1])),
                    "end": (float(r["end"][0]), float(r["end"][1])),
                })
            return cls(
                edge_refs=refs,
                first_layer_size=float(data.get("first_layer_size", 0.0) or 0.0),
                growth_ratio=float(data.get("growth_ratio", 1.2) or 1.2),
                num_layers=int(data.get("num_layers", 5) or 5),
                quads=bool(data.get("quads", False)),
                max_thickness=float(data.get("max_thickness", 0.0) or 0.0),
                set_name=str(data.get("set_name", "") or ""),
            )
        except Exception:
            return None

    def to_dict(self):
        return {
            "edge_refs": [
                {"part_id": int(r["part_id"]),
                 "start": (float(r["start"][0]), float(r["start"][1])),
                 "end":   (float(r["end"][0]), float(r["end"][1]))}
                for r in self.edge_refs
            ],
            "first_layer_size": float(self.first_layer_size),
            "growth_ratio": float(self.growth_ratio),
            "num_layers": int(self.num_layers),
            "quads": bool(self.quads),
            "max_thickness": float(self.max_thickness),
            "set_name": str(self.set_name),
        }


@dataclass
class MatchedEdgePair:
    """Two opposing edges constrained to have matching node positions.

    Used for periodic boundary conditions or structured/symmetric meshes.
    Internally driven by gmsh's setPeriodic: the slave's mesh is rebuilt from
    the master's, applying an affine transform between them. v1 supports the
    common translation case (two parallel edges of equal length, same
    orientation); rotated/reflected pairs would need a more general affine.

    master:  {part_id, start, end}
    slave:   {part_id, start, end}
    """

    master: dict = field(default_factory=dict)
    slave: dict = field(default_factory=dict)
    set_name: str = ""

    def __post_init__(self):
        def _clean(ref):
            if not isinstance(ref, dict):
                return None
            try:
                return {
                    "part_id": int(ref.get("part_id")),
                    "start": (float(ref["start"][0]), float(ref["start"][1])),
                    "end":   (float(ref["end"][0]), float(ref["end"][1])),
                }
            except Exception:
                return None
        self.master = _clean(self.master) or {}
        self.slave = _clean(self.slave) or {}
        self.set_name = str(self.set_name or "")

    def is_valid(self) -> bool:
        return bool(self.master) and bool(self.slave) and self.master != self.slave

    @classmethod
    def from_dict(cls, data):
        if data is None:
            return None
        if isinstance(data, cls):
            return data
        try:
            return cls(
                master=dict(data.get("master") or {}),
                slave=dict(data.get("slave") or {}),
                set_name=str(data.get("set_name", "") or ""),
            )
        except Exception:
            return None

    def to_dict(self):
        return {
            "master": dict(self.master or {}),
            "slave": dict(self.slave or {}),
            "set_name": str(self.set_name),
        }


def default_heterogeneity_config():
    return {
        "materials": [],
        "random_seed": None,
        "expressions": {key: "" for key in FIELD_DISTRIBUTION_PROPERTY_KEYS},
    }


def normalize_heterogeneity_config(value):
    config = default_heterogeneity_config()
    if not isinstance(value, dict):
        return config

    materials = []
    for item in value.get("materials", []) or []:
        if not isinstance(item, dict):
            continue
        try:
            material_id = int(item.get("material_id"))
        except Exception:
            continue
        try:
            fraction = float(item.get("fraction", 0.0))
        except Exception:
            fraction = 0.0
        if fraction <= 0.0:
            continue
        materials.append({"material_id": material_id, "fraction": fraction})
    config["materials"] = materials

    seed = value.get("random_seed")
    if seed not in (None, ""):
        try:
            config["random_seed"] = int(seed)
        except Exception:
            config["random_seed"] = None

    expressions = value.get("expressions", {})
    if isinstance(expressions, dict):
        for key in FIELD_DISTRIBUTION_PROPERTY_KEYS:
            config["expressions"][key] = str(expressions.get(key, "") or "")

    return config


def default_material_field_config():
    return {
        "property_key": "E",
        "field_type": "linear_gradient",
        "linear_gradient": {
            "min": 0.0,
            "max": 0.0,
            "direction": "x",
        },
        "radial_gradient": {
            "center_x": 0.0,
            "center_y": 0.0,
            "radius": 1.0,
            "core": 0.0,
            "shell": 0.0,
        },
        "random_field": {
            "mean": 0.0,
            "std": 0.0,
            "correlation_length": 1.0,
            "seed": None,
        },
        "user_equation": {
            "expression": "",
        },
    }


def normalize_material_field_config(value):
    config = default_material_field_config()
    if not isinstance(value, dict):
        return config

    property_key = str(value.get("property_key", "E") or "E")
    if property_key not in {"E", "rho", "nu"}:
        property_key = "E"
    config["property_key"] = property_key

    field_type = str(value.get("field_type", "linear_gradient") or "linear_gradient")
    if field_type not in {"linear_gradient", "radial_gradient", "random_field", "user_equation"}:
        field_type = "linear_gradient"
    config["field_type"] = field_type

    linear = value.get("linear_gradient", {})
    if isinstance(linear, dict):
        for key in ("min", "max"):
            try:
                config["linear_gradient"][key] = float(linear.get(key, config["linear_gradient"][key]))
            except Exception:
                pass
        direction = str(linear.get("direction", config["linear_gradient"]["direction"]) or "x").lower()
        if direction not in {"x", "y", "diag", "radial_x", "radial_y"}:
            direction = "x"
        config["linear_gradient"]["direction"] = direction

    radial = value.get("radial_gradient", {})
    if isinstance(radial, dict):
        for key in ("center_x", "center_y", "radius", "core", "shell"):
            try:
                config["radial_gradient"][key] = float(radial.get(key, config["radial_gradient"][key]))
            except Exception:
                pass

    random_field = value.get("random_field", {})
    if isinstance(random_field, dict):
        for key in ("mean", "std", "correlation_length"):
            try:
                config["random_field"][key] = float(
                    random_field.get(key, config["random_field"][key])
                )
            except Exception:
                pass
        seed = random_field.get("seed", config["random_field"]["seed"])
        if seed in (None, ""):
            config["random_field"]["seed"] = None
        else:
            try:
                config["random_field"]["seed"] = int(seed)
            except Exception:
                config["random_field"]["seed"] = None

    equation = value.get("user_equation", {})
    if isinstance(equation, dict):
        config["user_equation"]["expression"] = str(equation.get("expression", "") or "")

    return config


class Material:
    """Represents a material with type and properties."""

    _serial_counter = 0

    TYPES = {
        "RIGID": {"name": "Rigid", "props": ["density"]},
        "ELAS1": {
            "name": "Elastic (Plane Stress)",
            "props": ["density", "youngs_modulus", "poisson_ratio", "failure_energy"],
        },
        "ELAS2": {
            "name": "Elastic (Plane Strain)",
            "props": ["density", "youngs_modulus", "poisson_ratio", "failure_energy"],
        },
        "NEOHOOK": {"name": "Neohookean", "props": ["density", "shear_modulus", "bulk_modulus"]},
        "VISCOPLASTIC": {
            "name": "Viscoplastic",
            "props": [
                "density",
                "youngs_modulus",
                "poisson_ratio",
                "yield_stress",
                "hardening_rate",
            ],
        },
    }

    def __init__(self, name, mat_type, properties, symmetry="isotropic", behavior=None, damage="none"):
        Material._serial_counter += 1
        self.serial = Material._serial_counter
        self.name = name
        self.behavior = normalize_material_behavior(behavior or infer_behavior_from_mat_type(mat_type))
        self.damage = normalize_material_damage(damage)
        self.symmetry = normalize_material_symmetry(symmetry)
        self.mat_type = legacy_mat_type_for_behavior(self.behavior, mat_type)  # Legacy solver-facing type
        self.properties = normalize_material_properties(properties, self.behavior, self.symmetry, self.damage)

    def __repr__(self):
        return (
            f"Material(s={self.serial}, type={self.mat_type}, behavior={self.behavior}, "
            f"damage={self.damage}, name='{self.name}')"
        )

    def get_cpd_lines(self):
        """Generate CPD format material definition lines."""
        lines = [f"MATERIAL TYPE {self.serial}, {self.mat_type}"]
        prop_vals = []
        for prop_key in self.TYPES[self.mat_type]["props"]:
            if prop_key in self.properties:
                prop_vals.append(str(self.properties[prop_key]))
        lines.append(f"MATERIAL PROPERTIES {self.serial}, {', '.join(prop_vals)}")
        return lines


class Operation:
    """Represents a shape operation in the history tree."""

    _op_counter = 0

    def __init__(self, shape_data, op_type="ADD", material_id=None):
        Operation._op_counter += 1
        self.id = Operation._op_counter
        self.shape_data = shape_data  # {'type': 'rect'|'circle'|etc, 'verts': [...]}
        self.op_type = op_type  # "ADD" or "SUBTRACT"
        self.material_id = material_id  # Reference to Material.serial
        self.geometry = None  # Cached Shapely geometry

    def __repr__(self):
        return f"Op(id={self.id}, type={self.op_type}, mat={self.material_id})"


class Part:
    """Represents a named body/part in the assembly."""

    _part_counter = 0

    def __init__(self, name, geometry=None, is_void=False):
        Part._part_counter += 1
        self.id = Part._part_counter
        self.name = name  # e.g., "Part 1", "Inclusion", etc.
        self.part_type = "void" if is_void else "solid"
        self.geometry = geometry  # Shapely geometry (ORIGINAL, never modified)
        self.cad_shape = None  # OpenCASCADE/OCP shape for true 3D
        self.cad_source = None  # Dict describing CAD import source
        self.material_id = None  # Material assigned to this part
        self.material_type = None  # Material type name
        self.material_props = {}  # Material properties
        self.parent_id = None  # If nested in another part, points to parent Part.id
        self.is_void = is_void  # True if this part represents a hole/cavity
        self.is_rigid = False  # True if this part should be treated as a rigid body
        self.is_direct_edit = False  # True if part was modified by direct boolean features
        self.material_assignment_mode = "homogeneous"
        self.heterogeneity_method = "region_based"
        self.heterogeneity_config = default_heterogeneity_config()
        self.material_field_config = default_material_field_config()
        self.material_symmetry = "isotropic"
        self.material_behavior = "elastic"
        self.material_damage = "none"
        self.generated_feature_kind = None  # e.g. "porous_particles" or "porous_holes"
        self.generated_feature_settings = None  # Dialog settings used to generate this feature part
        self.particles = []  # Particle metadata for grouped generated particle sets

    def __repr__(self):
        return (
            f"Part(id={self.id}, name='{self.name}', type='{self.part_type}', parent={self.parent_id}, "
            f"mat={self.material_id}, void={self.is_void}, rigid={self.is_rigid})"
        )


class Interface:
    """Represents an interface/contact relation between two parts/materials."""

    _interface_counter = 0
    _interaction_counter = 0
    CUSTOM_TYPE_KEY = "OTHERS"
    DEFAULT_LAYER_MODE = "single_layer_ring"
    DEFAULT_PLACEMENT_MODE = "matrix_side"
    PLACEMENT_MODES = {
        "matrix_side": "Matrix-side (Coating) (Recommended)",
        "centered": "Centered on Interface (Planned)",
        "inclusion_side": "Inclusion-side (Planned)",
    }

    TYPES = {
        # Preferred interphase/interface-layer presets (frontend workflow)
        "BONDED": "Bonded Interface (stiff interphase)",
        "FRICTIONAL": "Frictional Interface",
        "SOFT": "Soft/Compliant Interface",
        "DAMAGEABLE": "Damageable Interface",
        "OTHERS": "Others (Custom Material)",
        # Legacy contact relation labels kept for backward compatibility
        "FIXED": "Completely Fixed (No motion)",
        "GLUE": "Glued/Bonded (Tied contact)",
        "FRICTIONLESS": "Frictionless Sliding",
        "ROUGH": "Rough/Frictional Contact",
        "GLIDING": "Gliding with Friction",
        "CONTACT": "General Contact",
    }

    TYPE_DEFAULT_FRICTION = {
        "FRICTIONAL": 0.30,
        "ROUGH": 0.60,
        "GLIDING": 0.20,
        "CONTACT": 0.20,
        "FRICTIONLESS": 0.0,
    }

    TYPE_MATERIAL_PRESETS = {
        "BONDED": {
            "mat_type": "ELAS1",
            "name": "Interface-Bonded",
            "properties": {
                "density": 1500.0,
                "youngs_modulus": 5.0e9,
                "poisson_ratio": 0.30,
                "failure_energy": 5.0e5,
            },
        },
        "FRICTIONAL": {
            "mat_type": "ELAS1",
            "name": "Interface-Frictional",
            "properties": {
                "density": 1500.0,
                "youngs_modulus": 2.0e9,
                "poisson_ratio": 0.30,
                "failure_energy": 2.5e5,
            },
        },
        "SOFT": {
            "mat_type": "NEOHOOK",
            "name": "Interface-Soft",
            "properties": {
                "density": 1200.0,
                "shear_modulus": 5.0e5,
                "bulk_modulus": 2.0e6,
            },
        },
        "DAMAGEABLE": {
            "mat_type": "ELAS1",
            "name": "Interface-Damageable",
            "properties": {
                "density": 1400.0,
                "youngs_modulus": 1.0e9,
                "poisson_ratio": 0.28,
                "failure_energy": 5.0e4,
            },
        },
        # Legacy keys mapped to reasonable presets for auto-material creation.
        "FIXED": {
            "mat_type": "RIGID",
            "name": "Interface-Fixed",
            "properties": {
                "density": 1.0,
            },
        },
        "GLUE": {
            "mat_type": "ELAS1",
            "name": "Interface-Glue",
            "properties": {
                "density": 1500.0,
                "youngs_modulus": 5.0e9,
                "poisson_ratio": 0.30,
                "failure_energy": 5.0e5,
            },
        },
        "FRICTIONLESS": {
            "mat_type": "ELAS1",
            "name": "Interface-Frictionless",
            "properties": {
                "density": 1400.0,
                "youngs_modulus": 5.0e8,
                "poisson_ratio": 0.30,
                "failure_energy": 5.0e4,
            },
        },
        "ROUGH": {
            "mat_type": "ELAS1",
            "name": "Interface-Rough",
            "properties": {
                "density": 1500.0,
                "youngs_modulus": 2.0e9,
                "poisson_ratio": 0.30,
                "failure_energy": 2.0e5,
            },
        },
        "GLIDING": {
            "mat_type": "ELAS1",
            "name": "Interface-Gliding",
            "properties": {
                "density": 1450.0,
                "youngs_modulus": 1.5e9,
                "poisson_ratio": 0.30,
                "failure_energy": 1.5e5,
            },
        },
        "CONTACT": {
            "mat_type": "ELAS1",
            "name": "Interface-Contact",
            "properties": {
                "density": 1500.0,
                "youngs_modulus": 1.0e9,
                "poisson_ratio": 0.30,
                "failure_energy": 1.0e5,
            },
        },
    }

    def __init__(self, part1_id, part2_id, interface_type="GLUE"):
        Interface._interface_counter = max(
            int(getattr(Interface, "_interface_counter", 0)),
            int(getattr(Interface, "_interaction_counter", 0)),
        ) + 1
        Interface._interaction_counter = Interface._interface_counter
        self.id = Interface._interface_counter
        self.part1_id = part1_id  # Reference to Part.id
        self.part2_id = part2_id  # Reference to Part.id
        self.interface_type = interface_type  # Key from TYPES dict
        self.friction_coeff = float(self.TYPE_DEFAULT_FRICTION.get(interface_type, 0.0))
        # Frontend-only interphase metadata (solver/backend may ignore for now).
        self.material_id = None
        self.material_mode = "auto"
        self.thickness = None
        self.target_dx = None
        self.layer_mode = self.DEFAULT_LAYER_MODE
        self.placement_mode = self.DEFAULT_PLACEMENT_MODE
        self.status = ""
        self.notes = ""

    @property
    def interaction_type(self):
        return self.interface_type

    @interaction_type.setter
    def interaction_type(self, value):
        self.interface_type = value

    def __repr__(self):
        return (
            f"Interface(id={self.id}, p1={self.part1_id}, p2={self.part2_id}, "
            f"type={self.interface_type})"
        )

    @classmethod
    def preset_material_spec(cls, interface_type):
        """Return a copy of the frontend material preset for the interface type."""
        spec = cls.TYPE_MATERIAL_PRESETS.get(interface_type)
        if not isinstance(spec, dict):
            return None
        return {
            "mat_type": spec.get("mat_type"),
            "name": spec.get("name"),
            "properties": dict(spec.get("properties", {})),
        }


# Backward-compatible alias for older imports and project loaders.
Interaction = Interface
