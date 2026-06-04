from __future__ import annotations

import os
import math
from collections import OrderedDict

import numpy as np
import pandas as pd
import yaml
from PySide6.QtCore import QObject, QThread, Signal

from app_config import get_project_root, get_workspace_dir
from project_stages import ProjectStage


def _close_numpy_handle(value):
    current = value
    seen = set()
    while current is not None:
        ident = id(current)
        if ident in seen:
            break
        seen.add(ident)
        mmap_obj = getattr(current, "_mmap", None)
        if mmap_obj is not None:
            try:
                mmap_obj.close()
            except Exception:
                pass
        close_fn = getattr(current, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass
        base = getattr(current, "base", None)
        if base is current:
            break
        current = base


def _history_ref(path, shape):
    try:
        shape_tuple = tuple(int(v) for v in tuple(shape))
    except Exception:
        shape_tuple = ()
    return {"path": str(path), "shape": shape_tuple}


class FrameLoader(QThread):
    frameLoaded = Signal(int, int, object)
    loadFailed = Signal(int, int, str)

    def __init__(self, source_kind, source_ref, frame_index, request_token, unit_scale=1.0):
        super().__init__()
        self.source_kind = str(source_kind or "")
        self.source_ref = source_ref
        self.frame_index = int(frame_index)
        self.request_token = int(request_token)
        try:
            self.unit_scale = float(unit_scale)
        except Exception:
            self.unit_scale = 1.0

    def stop(self):
        try:
            self.requestInterruption()
        except Exception:
            pass
        try:
            self.quit()
        except Exception:
            pass
        try:
            if QThread.currentThread() != self:
                self.wait()
        except Exception:
            pass

    def run(self):
        pos_history = None
        try:
            if self.source_kind == "npy":
                pos_history = np.load(self.source_ref, mmap_mode="r")
                if pos_history.ndim != 3 or pos_history.shape[2] < 2:
                    raise ValueError(
                        "pos_history.npy has an unexpected shape. Expected (n_frames, n_particles, 2)."
                    )
                frame = np.array(pos_history[self.frame_index, :, :2], dtype=float, copy=True)
                if self.unit_scale not in (0.0, 1.0):
                    frame = np.array(frame / self.unit_scale, dtype=float, copy=True)
            elif self.source_kind == "csv":
                df = pd.read_csv(self.source_ref)
                if "particle_id" in df.columns:
                    df = df.set_index("particle_id")
                elif "node_id" in df.columns:
                    df = df.set_index("node_id")
                if "x" not in df.columns or "y" not in df.columns:
                    raise ValueError("Result CSV is missing x/y columns.")
                frame = df[["x", "y"]].to_numpy(dtype=float, copy=True)
            else:
                raise ValueError(f"Unsupported results source: {self.source_kind}")

            if self.isInterruptionRequested():
                return
            self.frameLoaded.emit(self.frame_index, self.request_token, frame)
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.loadFailed.emit(self.frame_index, self.request_token, str(exc))
        finally:
            _close_numpy_handle(pos_history)


class ResultsController(QObject):
    framesDiscovered = Signal(int)
    frameLoadStarted = Signal(int)
    frameReady = Signal(int, object)
    frameLoadFailed = Signal(int, str)

    workflow_key = "results"
    label = "Results"
    stage = ProjectStage.RESULTS
    show_sketch = False
    show_geometry = False
    show_loads = False
    hint = "Review saved results, play animations, and compare outputs."

    RESULT_FIELD_SPECS = OrderedDict(
        (
            ("none", {"label": "None", "domain": None, "source": None}),
            ("disp_mag", {"label": "Displacement magnitude", "domain": "node", "source": "displacement"}),
            ("exx", {"label": "Strain contour: EXX", "domain": "triangle", "source": "strain", "component": 0}),
            ("eyy", {"label": "Strain contour: EYY", "domain": "triangle", "source": "strain", "component": 1}),
            ("exy", {"label": "Strain contour: EXY", "domain": "triangle", "source": "strain", "component": 2}),
            ("zxx", {"label": "Stress contour: SXX", "domain": "triangle", "source": "stress", "component": 0}),
            ("zyy", {"label": "Stress contour: SYY", "domain": "triangle", "source": "stress", "component": 1}),
            ("zxy", {"label": "Stress contour: SXY", "domain": "triangle", "source": "stress", "component": 2}),
            ("vm", {"label": "Stress contour: Von Mises", "domain": "triangle", "source": "stress", "component": 3}),
        )
    )

    HISTORY_CURVE_SPECS = OrderedDict(
        (
            ("max_vm_time", {"label": "Max Von Mises vs Time", "source": "stress", "kind": "vm", "reducer": "max"}),
            ("mean_vm_time", {"label": "Mean Von Mises vs Time", "source": "stress", "kind": "vm", "reducer": "mean"}),
            (
                "max_eqv_strain_time",
                {"label": "Max Equivalent Strain vs Time", "source": "strain", "kind": "eqv", "reducer": "max"},
            ),
            (
                "mean_eqv_strain_time",
                {"label": "Mean Equivalent Strain vs Time", "source": "strain", "kind": "eqv", "reducer": "mean"},
            ),
            (
                "max_disp_mag_time",
                {"label": "Max Displacement vs Time", "source": "displacement", "kind": "mag", "reducer": "max"},
            ),
            (
                "mean_disp_mag_time",
                {"label": "Mean Displacement vs Time", "source": "displacement", "kind": "mag", "reducer": "mean"},
            ),
        )
    )

    RESPONSE_SCOPE_LABELS = OrderedDict(
        (
            ("node", "Node"),
            ("geometry_edge", "Geometry Edge"),
            ("bc_target", "BC Target"),
            ("triangle", "Triangle"),
        )
    )

    RESPONSE_QUANTITY_SPECS = OrderedDict(
        (
            (
                "displacement",
                {
                    "label": "Displacement",
                    "hint": "Displacement = nodal or edge displacement history.",
                    "scopes": ("node", "geometry_edge"),
                    "subtypes": OrderedDict(
                        (
                            ("mag", {"label": "|u|", "title": "Displacement Magnitude"}),
                            ("ux", {"label": "u_x", "title": "Displacement Ux"}),
                            ("uy", {"label": "u_y", "title": "Displacement Uy"}),
                        )
                    ),
                },
            ),
            (
                "force",
                {
                    "label": "Force",
                    "hint": "Force = externally applied load history on the selected target.",
                    "scopes": ("node", "geometry_edge"),
                    "subtypes": OrderedDict(
                        (
                            ("fx", {"label": "F_x", "title": "Applied Force Fx"}),
                            ("fy", {"label": "F_y", "title": "Applied Force Fy"}),
                            ("mag", {"label": "|F|", "title": "Applied Force Magnitude"}),
                        )
                    ),
                },
            ),
            (
                "reaction_force",
                {
                    "label": "Reaction Force",
                    "hint": "Reaction Force = support or boundary reaction at a BC target.",
                    "scopes": ("bc_target",),
                    "subtypes": OrderedDict(
                        (
                            ("rx", {"label": "R_x", "title": "Reaction Force Rx"}),
                            ("ry", {"label": "R_y", "title": "Reaction Force Ry"}),
                            ("mag", {"label": "|R|", "title": "Reaction Force Magnitude"}),
                        )
                    ),
                },
            ),
            (
                "stress",
                {
                    "label": "Stress",
                    "hint": "Stress = triangle or element-level result history.",
                    "scopes": ("triangle",),
                    "subtypes": OrderedDict(
                        (
                            ("vm", {"label": "σ_vm", "title": "Von Mises Stress"}),
                            ("s1", {"label": "σ_1", "title": "Max Principal Stress"}),
                            ("s2", {"label": "σ_2", "title": "Min Principal Stress"}),
                        )
                    ),
                },
            ),
            (
                "strain",
                {
                    "label": "Strain",
                    "hint": "Strain = triangle or element-level result history.",
                    "scopes": ("triangle",),
                    "subtypes": OrderedDict(
                        (
                            ("eq", {"label": "ε_eq", "title": "Equivalent Strain"}),
                            ("e1", {"label": "ε_1", "title": "Max Principal Strain"}),
                            ("e2", {"label": "ε_2", "title": "Min Principal Strain"}),
                        )
                    ),
                },
            ),
        )
    )

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self._source_kind = None
        self._source_ref = None
        self._csv_frame_paths = []
        self._frame_count = 0
        self._particle_count = 0
        self._unit_scale = 1.0
        self._cache_size = 5
        self._frame_cache = OrderedDict()
        self._loader = None
        self._request_token = 0
        self._pending_request = None
        self._displacement_history = None
        self._strain_history = None
        self._stress_history = None
        self._active_result_field = "none"
        self._stress_strain_curve_cache = {}
        self._history_curve_cache = {}
        self._history_selection_active = False
        self._history_selection_nodes = None
        self._history_selection_triangles = None
        self._history_selection_label = ""
        self._history_selection_scope_key = "node"
        self._history_selection_payload = {}

    def workflow_definition(self, tab):
        return {
            "label": self.label,
            "stage": self.stage,
            "tab": tab,
            "show_sketch": self.show_sketch,
            "show_geometry": self.show_geometry,
            "show_loads": self.show_loads,
        }

    def hint_for(self, _stage=None):
        return self.hint

    def restore_workspace_results_from_artifacts(self, project_file):
        return self.window._restore_workspace_results_from_artifacts_impl(project_file)

    def frame_count(self):
        return int(self._frame_count)

    def particle_count(self):
        return int(self._particle_count)

    def get_cached_frame(self, frame_index):
        return self._frame_cache.get(int(frame_index))

    def clear_results_source(self):
        self._request_token += 1
        self._pending_request = None
        self._stop_loader()
        self._loader = None
        for frame in list(self._frame_cache.values()):
            _close_numpy_handle(frame)
        _close_numpy_handle(self._displacement_history)
        _close_numpy_handle(self._strain_history)
        _close_numpy_handle(self._stress_history)
        self._source_kind = None
        self._source_ref = None
        self._csv_frame_paths = []
        self._frame_count = 0
        self._particle_count = 0
        self._unit_scale = 1.0
        self._frame_cache.clear()
        self._displacement_history = None
        self._strain_history = None
        self._stress_history = None
        self._active_result_field = "none"
        self._stress_strain_curve_cache.clear()
        self._history_curve_cache.clear()
        self._history_selection_active = False
        self._history_selection_nodes = None
        self._history_selection_triangles = None
        self._history_selection_label = ""
        self._history_selection_scope_key = "node"
        self._history_selection_payload = {}

    def set_history_selection(
        self,
        *,
        active,
        node_ids=None,
        triangle_ids=None,
        label=None,
        scope=None,
        selection=None,
    ):
        self._history_selection_active = bool(active)
        if node_ids is None:
            self._history_selection_nodes = None
        else:
            self._history_selection_nodes = sorted({int(i) for i in node_ids if i is not None})
        if triangle_ids is None:
            self._history_selection_triangles = None
        else:
            self._history_selection_triangles = sorted({int(i) for i in triangle_ids if i is not None})
        self._history_selection_label = str(label or "")
        scope_key = str(scope or self._history_selection_scope_key or "node").strip().lower()
        if scope_key not in self.RESPONSE_SCOPE_LABELS:
            scope_key = "node"
        self._history_selection_scope_key = scope_key
        self._history_selection_payload = dict(selection or {})
        self._stress_strain_curve_cache.clear()
        self._history_curve_cache.clear()

    def open_results(self, results_root=None, unit_scale=1.0):
        self.clear_results_source()
        try:
            self._unit_scale = float(unit_scale)
        except Exception:
            self._unit_scale = 1.0

        results_dir, pos_history_path = self._resolve_results_paths(results_root)
        output_dir = os.path.dirname(pos_history_path) if pos_history_path else None
        result_files = self._discover_csv_frames(results_dir)
        pos_history_exists = bool(pos_history_path and os.path.exists(pos_history_path))

        result_mtime = None
        if result_files:
            result_mtime = os.path.getmtime(result_files[-1])
        pos_mtime = os.path.getmtime(pos_history_path) if pos_history_exists else None
        use_pos_history = pos_mtime is not None and (result_mtime is None or pos_mtime >= result_mtime)

        if use_pos_history:
            pos_history = None
            try:
                pos_history = np.load(pos_history_path, mmap_mode="r")
                if pos_history.ndim != 3 or pos_history.shape[2] < 2:
                    raise ValueError(
                        "pos_history.npy has an unexpected shape. Expected (n_frames, n_particles, 2)."
                    )
                self._source_kind = "npy"
                self._source_ref = pos_history_path
                self._frame_count = int(pos_history.shape[0])
                self._particle_count = int(pos_history.shape[1])
                self._load_optional_histories(output_dir, self._frame_count)
            finally:
                _close_numpy_handle(pos_history)
        elif result_files:
            self._source_kind = "csv"
            self._csv_frame_paths = result_files
            self._frame_count = len(result_files)
            try:
                self._particle_count = int(len(pd.read_csv(result_files[0])))
            except Exception:
                self._particle_count = 0
            self._displacement_history = None
            self._strain_history = None
            self._stress_history = None
        else:
            raise FileNotFoundError(
                "No simulation results found in workspace/output/. Run a simulation first."
            )

        active_field = str(self._active_result_field or "none").strip().lower()
        if active_field == "none" or active_field not in self.available_result_fields():
            self._active_result_field = self.default_result_field()

        self.framesDiscovered.emit(int(self._frame_count))
        return int(self._frame_count)

    def default_result_field(self):
        available = self.available_result_fields()
        if "vm" in available:
            return "vm"
        if "disp_mag" in available:
            return "disp_mag"
        return "none"

    def result_field_options(self):
        options = []
        available = self.available_result_fields()
        for key, spec in self.RESULT_FIELD_SPECS.items():
            if key == "none" or key in available:
                options.append((key, spec["label"]))
        return options

    def available_result_fields(self):
        fields = {"none"}
        if self._displacement_history is not None:
            fields.add("disp_mag")
        if self._strain_history is not None:
            fields.update(("exx", "eyy", "exy"))
        if self._stress_history is not None:
            fields.update(("zxx", "zyy", "zxy", "vm"))
        return fields

    def active_result_field(self):
        return str(self._active_result_field or "none")

    def set_active_result_field(self, field_key):
        key = str(field_key or "none").strip().lower()
        if key not in self.RESULT_FIELD_SPECS or key not in self.available_result_fields():
            key = self.default_result_field()
        self._active_result_field = key
        return key

    def field_label(self, field_key=None):
        key = str(field_key or self._active_result_field or "none").strip().lower()
        return str(self.RESULT_FIELD_SPECS.get(key, self.RESULT_FIELD_SPECS["none"])["label"])

    def field_legend_metadata(self, field_key=None, values=None):
        key = str(field_key or self._active_result_field or "none").strip().lower()
        spec = self.RESULT_FIELD_SPECS.get(key, self.RESULT_FIELD_SPECS["none"])
        label = str(spec.get("label") or "None")
        source = str(spec.get("source") or "").strip().lower()
        unit = ""
        scale = 1.0
        if key == "disp_mag":
            try:
                unit = str(getattr(self.window.view, "current_unit", "") or "m").strip()
            except Exception:
                unit = "m"
        elif source == "strain":
            unit = "mm/mm"
        elif source == "stress":
            arr = np.asarray(values, dtype=float).reshape(-1) if values is not None else np.asarray([], dtype=float)
            finite = np.isfinite(arr)
            if np.any(finite):
                max_abs = float(np.max(np.abs(arr[finite])))
            else:
                max_abs = 0.0
            if max_abs >= 1.0e9:
                unit = "GPa"
                scale = 1.0e9
            elif max_abs >= 1.0e6:
                unit = "MPa"
                scale = 1.0e6
            elif max_abs >= 1.0e3:
                unit = "kPa"
                scale = 1.0e3
            else:
                unit = "Pa"
        title = label.split(":", 1)[-1].strip() if ":" in label else label
        return {
            "key": key,
            "label": label,
            "title": title,
            "unit": unit,
            "scale": float(scale or 1.0),
            "domain": spec.get("domain"),
            "source": spec.get("source"),
        }

    def _response_quantity_spec(self, quantity):
        key = str(quantity or "").strip().lower()
        if key in self.RESPONSE_QUANTITY_SPECS:
            return key, self.RESPONSE_QUANTITY_SPECS[key]
        first_key = next(iter(self.RESPONSE_QUANTITY_SPECS.keys()))
        return first_key, self.RESPONSE_QUANTITY_SPECS[first_key]

    def response_plot_quantities(self):
        return [
            (key, str(spec.get("label") or key.replace("_", " ").title()))
            for key, spec in self.RESPONSE_QUANTITY_SPECS.items()
        ]

    def default_response_plot_quantity(self):
        return next(iter(self.RESPONSE_QUANTITY_SPECS.keys()))

    def response_plot_subtypes(self, quantity):
        _key, spec = self._response_quantity_spec(quantity)
        return [
            (sub_key, str(sub_spec.get("label") or sub_key))
            for sub_key, sub_spec in (spec.get("subtypes") or OrderedDict()).items()
        ]

    def default_response_plot_subtype(self, quantity):
        options = self.response_plot_subtypes(quantity)
        if not options:
            return ""
        return str(options[0][0])

    def response_plot_scopes(self, quantity):
        _key, spec = self._response_quantity_spec(quantity)
        return [
            (scope_key, self.RESPONSE_SCOPE_LABELS.get(scope_key, scope_key.replace("_", " ").title()))
            for scope_key in tuple(spec.get("scopes") or ())
        ]

    def default_response_plot_scope(self, quantity):
        options = self.response_plot_scopes(quantity)
        if not options:
            return "node"
        return str(options[0][0])

    def response_plot_hint(self, quantity):
        _key, spec = self._response_quantity_spec(quantity)
        return str(spec.get("hint") or "")

    def _history_selection_data(self):
        payload = dict(getattr(self, "_history_selection_payload", {}) or {})
        payload.setdefault("scope", str(self._history_selection_scope_key or "node"))
        payload.setdefault("label", str(self._history_selection_label or ""))
        payload.setdefault("node_ids", list(self._history_selection_nodes or []))
        payload.setdefault("triangle_ids", list(self._history_selection_triangles or []))
        payload.setdefault("load_matches", list(payload.get("load_matches") or []))
        payload.setdefault("bc_indices", list(payload.get("bc_indices") or []))
        return payload

    def _history_ref_path(self, history):
        if isinstance(history, dict):
            path = str(history.get("path") or "").strip()
            if path:
                return path
        return None

    def _history_shape(self, history):
        if history is None:
            return ()
        if isinstance(history, dict):
            shape = history.get("shape")
            try:
                return tuple(int(v) for v in tuple(shape))
            except Exception:
                return ()
        try:
            return tuple(int(v) for v in np.shape(history))
        except Exception:
            return ()

    def _open_history_array(self, history):
        if history is None:
            return None, False
        path = self._history_ref_path(history)
        if path:
            arr = np.load(path, mmap_mode="r")
            return arr, True
        return history, False

    def _read_history_frame(self, history, frame_index):
        if history is None:
            return None
        shape = self._history_shape(history)
        if len(shape) < 2 or shape[0] <= 0:
            return None
        idx = int(max(0, min(int(frame_index), int(shape[0]) - 1)))
        arr = None
        owns_handle = False
        try:
            arr, owns_handle = self._open_history_array(history)
            if arr is None:
                return None
            return np.array(arr[idx], dtype=float, copy=True)
        except Exception:
            return None
        finally:
            if owns_handle:
                _close_numpy_handle(arr)

    def get_field_frame(self, frame_index, field_key=None):
        key = str(field_key or self._active_result_field or "none").strip().lower()
        spec = self.RESULT_FIELD_SPECS.get(key)
        if spec is None or spec.get("source") is None:
            return {"key": "none", "label": self.field_label("none"), "domain": None, "values": None}

        try:
            idx = int(max(0, min(int(frame_index), self._frame_count - 1)))
        except Exception:
            idx = 0

        source = spec["source"]
        history = None
        if source == "displacement":
            history = self._displacement_history
        elif source == "strain":
            history = self._strain_history
        elif source == "stress":
            history = self._stress_history
        if history is None:
            return {"key": key, "label": self.field_label(key), "domain": spec["domain"], "values": None}

        frame = self._read_history_frame(history, idx)
        if frame is None:
            return {"key": key, "label": self.field_label(key), "domain": spec["domain"], "values": None}
        if source == "displacement":
            if frame.ndim != 2 or frame.shape[1] < 2:
                values = None
            else:
                values = np.linalg.norm(frame[:, :2], axis=1)
        else:
            component = int(spec.get("component", 0))
            if frame.ndim != 2 or frame.shape[1] <= component:
                values = None
            else:
                values = np.asarray(frame[:, component], dtype=float)
        return {"key": key, "label": self.field_label(key), "domain": spec["domain"], "values": values}

    def summarize_field(self, frame_index, field_key=None):
        payload = self.get_field_frame(frame_index, field_key=field_key)
        values = payload.get("values")
        if values is None:
            return f"{payload['label']}: --"
        arr = np.asarray(values, dtype=float).reshape(-1)
        finite = np.isfinite(arr)
        if not np.any(finite):
            return f"{payload['label']}: --"
        vals = arr[finite]
        meta = self.field_legend_metadata(payload.get("key"), vals)
        scale = float(meta.get("scale") or 1.0)
        if scale not in (0.0, 1.0):
            vals = vals / scale
        unit = str(meta.get("unit") or "").strip()
        unit_suffix = f" [{unit}]" if unit else ""
        return (
            f"{payload['label']}{unit_suffix}: min {float(np.min(vals)):.4g}, "
            f"max {float(np.max(vals)):.4g}, mean {float(np.mean(vals)):.4g}"
        )

    def activity_summary(self):
        return {
            "disp": self._history_peak_summary(self._displacement_history, mode="vector_mag", label="Disp"),
            "stress": self._history_peak_summary(self._stress_history, component=3, label="Stress"),
            "strain": self._history_peak_summary(self._strain_history, component=0, label="Strain"),
        }

    def stress_strain_curve(self, mode="mean_eqv_vm"):
        key = str(mode or "mean_eqv_vm").strip().lower()
        if key not in {"mean_eqv_vm", "peak_eqv_vm"}:
            key = "mean_eqv_vm"

        cached = self._stress_strain_curve_cache.get(key)
        if cached is not None:
            return dict(cached)

        payload = {
            "available": False,
            "mode": key,
            "title": "Stress-Strain Response",
            "x_label": "Equivalent strain",
            "y_label": "Von Mises stress",
            "x": np.array([], dtype=float),
            "y": np.array([], dtype=float),
            "frame_count": 0,
            "selection_active": bool(self._history_selection_active),
            "selection_label": str(self._history_selection_label or ""),
        }

        if self._strain_history is None or self._stress_history is None:
            return payload

        strain = None
        stress = None
        close_strain = False
        close_stress = False
        try:
            strain, close_strain = self._open_history_array(self._strain_history)
            stress, close_stress = self._open_history_array(self._stress_history)
        except Exception:
            return payload
        try:
            strain = np.asarray(strain, dtype=float)
            stress = np.asarray(stress, dtype=float)

            if strain.ndim != 3 or strain.shape[2] < 3:
                return payload
            if stress.ndim != 3 or stress.shape[2] < 4:
                return payload

            frame_count = min(int(strain.shape[0]), int(stress.shape[0]))
            if self._frame_count > 0:
                frame_count = min(frame_count, int(self._frame_count))
            if frame_count <= 0:
                return payload

            strain = strain[:frame_count]
            stress = stress[:frame_count]

            exx = np.asarray(strain[:, :, 0], dtype=float)
            eyy = np.asarray(strain[:, :, 1], dtype=float)
            gxy = np.asarray(strain[:, :, 2], dtype=float)
            vm = np.asarray(stress[:, :, 3], dtype=float)

            # 2D equivalent strain companion to the already exported Von Mises stress.
            eqv_strain = np.sqrt(
                np.maximum(
                    exx * exx + eyy * eyy - exx * eyy + 0.75 * gxy * gxy,
                    0.0,
                )
            )
            vm = np.abs(vm)
            if self._history_selection_active:
                ids = self._history_selection_triangles
                if not ids:
                    return payload
                valid = [i for i in ids if 0 <= int(i) < eqv_strain.shape[1]]
                if not valid:
                    return payload
                eqv_strain = eqv_strain[:, valid]
                vm = vm[:, valid]

            def _reduce_over_triangles(values, reducer):
                arr = np.asarray(values, dtype=float)
                finite = np.isfinite(arr)
                out = np.full((arr.shape[0],), np.nan, dtype=float)
                if arr.ndim != 2:
                    return out
                valid_rows = np.any(finite, axis=1)
                if not np.any(valid_rows):
                    return out
                trimmed = arr[valid_rows]
                if reducer == "max":
                    out[valid_rows] = np.nanmax(trimmed, axis=1)
                else:
                    out[valid_rows] = np.nanmean(trimmed, axis=1)
                return out

            reducer = "max" if key == "peak_eqv_vm" else "mean"
            x_vals = _reduce_over_triangles(eqv_strain, reducer)
            y_vals = _reduce_over_triangles(vm, reducer)
            finite_curve = np.isfinite(x_vals) & np.isfinite(y_vals)
            if not np.any(finite_curve):
                return payload

            payload.update(
                {
                    "available": True,
                    "title": (
                        "Peak equivalent strain vs Von Mises stress"
                        if key == "peak_eqv_vm"
                        else "Mean equivalent strain vs Von Mises stress"
                    ),
                    "x": x_vals,
                    "y": y_vals,
                    "frame_count": frame_count,
                }
            )
            self._stress_strain_curve_cache[key] = dict(payload)
            return dict(payload)
        finally:
            if close_strain:
                _close_numpy_handle(strain)
            if close_stress:
                _close_numpy_handle(stress)

    def history_curve_options(self):
        options = []
        for key, spec in self.HISTORY_CURVE_SPECS.items():
            source = str(spec.get("source") or "").strip().lower()
            if source == "stress" and self._stress_history is None:
                continue
            if source == "strain" and self._strain_history is None:
                continue
            if source == "displacement" and self._displacement_history is None:
                continue
            options.append((key, str(spec.get("label") or key)))
        return options

    def default_history_curve_mode(self):
        options = self.history_curve_options()
        if not options:
            return "max_vm_time"
        for preferred in ("max_vm_time", "mean_vm_time", "max_eqv_strain_time", "max_disp_mag_time"):
            if any(key == preferred for key, _label in options):
                return preferred
        return str(options[0][0])

    def history_curve(self, mode="max_vm_time"):
        key = str(mode or self.default_history_curve_mode()).strip().lower()
        if key not in self.HISTORY_CURVE_SPECS:
            key = self.default_history_curve_mode()

        cached = self._history_curve_cache.get(key)
        if cached is not None:
            return dict(cached)

        payload = {
            "available": False,
            "mode": key,
            "title": "Response vs Time",
            "x_label": "Time (s)",
            "y_label": "Value",
            "x": np.array([], dtype=float),
            "y": np.array([], dtype=float),
            "frame_count": 0,
            "selection_active": bool(self._history_selection_active),
            "selection_label": str(self._history_selection_label or ""),
        }

        spec = self.HISTORY_CURVE_SPECS.get(key, {})
        source = str(spec.get("source") or "").strip().lower()
        reducer = str(spec.get("reducer") or "max").strip().lower()

        history = None
        if source == "stress":
            history = self._stress_history
        elif source == "strain":
            history = self._strain_history
        elif source == "displacement":
            history = self._displacement_history
        if history is None:
            return payload

        arr = None
        owns_handle = False
        try:
            arr, owns_handle = self._open_history_array(history)
            arr = np.asarray(arr, dtype=float)
        except Exception:
            return payload
        try:
            if arr.ndim != 3:
                return payload

            frame_count = int(arr.shape[0])
            if self._frame_count > 0:
                frame_count = min(frame_count, int(self._frame_count))
            if frame_count <= 0:
                return payload
            arr = arr[:frame_count]

            values = self._history_curve_values(arr, source=source, kind=str(spec.get("kind") or ""))
            if values is None:
                return payload

            if self._history_selection_active:
                if source == "displacement":
                    ids = self._history_selection_nodes
                else:
                    ids = self._history_selection_triangles
                if not ids:
                    return payload
                valid = [i for i in ids if 0 <= int(i) < values.shape[1]]
                if not valid:
                    return payload
                values = values[:, valid]

            y_vals = self._reduce_history_series(values, reducer=reducer)
            x_vals = self._result_time_values(frame_count)
            finite = np.isfinite(x_vals) & np.isfinite(y_vals)
            if not np.any(finite):
                return payload

            unit, scale = self._history_curve_unit_scale(source, str(spec.get("kind") or ""), y_vals)
            y_plot = np.asarray(y_vals, dtype=float)
            if scale not in (0.0, 1.0):
                y_plot = y_plot / float(scale)
            unit_suffix = f" [{unit}]" if unit else ""
            title = str(spec.get("label") or "Response vs Time")
            y_label = title.split(" vs ", 1)[0] + unit_suffix if " vs " in title else f"Value{unit_suffix}"

            payload.update(
                {
                    "available": True,
                    "title": title,
                    "x_label": "Time (s)",
                    "y_label": y_label,
                    "x": np.asarray(x_vals, dtype=float),
                    "y": np.asarray(y_plot, dtype=float),
                    "frame_count": frame_count,
                }
            )
            self._history_curve_cache[key] = dict(payload)
            return dict(payload)
        finally:
            if owns_handle:
                _close_numpy_handle(arr)

    def _response_curve_payload(self, quantity, subtype):
        quantity_key, quantity_spec = self._response_quantity_spec(quantity)
        subtype_key = str(subtype or "").strip().lower()
        subtype_spec = (quantity_spec.get("subtypes") or OrderedDict()).get(subtype_key)
        if subtype_spec is None:
            subtype_key = self.default_response_plot_subtype(quantity_key)
            subtype_spec = (quantity_spec.get("subtypes") or OrderedDict()).get(subtype_key, {})
        return {
            "available": False,
            "quantity": quantity_key,
            "subtype": subtype_key,
            "title": str(subtype_spec.get("title") or quantity_spec.get("label") or "Response"),
            "x_label": "Time (s)",
            "y_label": str(subtype_spec.get("label") or "Value"),
            "x": np.array([], dtype=float),
            "y": np.array([], dtype=float),
            "frame_count": 0,
            "selection_active": bool(self._history_selection_active),
            "selection_label": str(self._history_selection_label or ""),
            "message": "",
        }

    def _result_total_time(self):
        config_path = os.path.join(str(get_project_root()), "CPD-main", "config.yml")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as handle:
                    config = yaml.safe_load(handle) or {}
                sim = config.get("simulation", {}) or {}
                time_step = float(sim.get("time_step", 0.0))
                total_steps = int(sim.get("total_steps", 0))
                if time_step > 0.0 and total_steps > 0:
                    return float(time_step) * float(total_steps)
            except Exception:
                pass
        x_vals = self._result_time_values(self._frame_count)
        if x_vals.size > 0:
            return float(x_vals[-1])
        return 0.0

    def _safe_eval_expr(self, expr, t_value):
        view = getattr(self.window, "view", None)
        if view is not None and hasattr(view, "_safe_eval_expr"):
            try:
                return float(view._safe_eval_expr(expr, t_value))
            except Exception:
                pass
        safe = {
            "t": float(t_value),
            "time": float(t_value),
            "pi": math.pi,
            "e": math.e,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "sqrt": math.sqrt,
            "exp": math.exp,
            "log": math.log,
            "abs": abs,
            "min": min,
            "max": max,
        }
        try:
            return float(eval(str(expr or "0"), {"__builtins__": {}}, safe))
        except Exception:
            return 0.0

    def _normalize_force_profile(self, load, total_time):
        view = getattr(self.window, "view", None)
        if view is not None and hasattr(view, "_normalize_force_profile"):
            try:
                return list(view._normalize_force_profile(load, total_time) or [])
            except Exception:
                pass
        profile = load.get("time_profile") or []
        mode_is_percent = str(load.get("time_profile_mode", "absolute")).lower().startswith("percent")
        if not profile:
            return [
                {
                    "t0": 0.0,
                    "t1": float(total_time),
                    "fx": str(load.get("fx", 0.0)),
                    "fy": str(load.get("fy", 0.0)),
                    "fz": str(load.get("fz", 0.0)),
                }
            ]
        normalized = []
        for seg in profile:
            try:
                t0 = float(seg.get("t0", 0.0))
            except Exception:
                t0 = 0.0
            try:
                t1 = float(seg.get("t1", total_time))
            except Exception:
                t1 = float(total_time)
            if mode_is_percent and total_time > 0.0:
                t0 = t0 * float(total_time) / 100.0
                t1 = t1 * float(total_time) / 100.0
            normalized.append(
                {
                    "t0": float(t0),
                    "t1": float(t1),
                    "fx": str(seg.get("fx", seg.get("expr_fx", seg.get("expr", "0")))),
                    "fy": str(seg.get("fy", seg.get("expr_fy", "0"))),
                    "fz": str(seg.get("fz", seg.get("expr_fz", "0"))),
                }
            )
        return normalized

    def _eval_vector_profile_at_time(self, profile, t_value, keys):
        if not profile:
            return [0.0] * len(tuple(keys))
        segment = None
        for index, item in enumerate(profile):
            try:
                t0 = float(item.get("t0", 0.0))
            except Exception:
                t0 = 0.0
            try:
                t1 = float(item.get("t1", t0))
            except Exception:
                t1 = t0
            if float(t_value) < t0:
                continue
            if float(t_value) <= t1 or index == len(profile) - 1:
                segment = item
                break
        if segment is None:
            return [0.0] * len(tuple(keys))
        values = []
        for key in tuple(keys):
            values.append(self._safe_eval_expr(segment.get(key, "0"), t_value))
        return values

    def _force_entry_series(self, entry, x_vals):
        times = np.asarray(x_vals, dtype=float).reshape(-1)
        out = np.zeros((len(times), 3), dtype=float)
        if times.size <= 0:
            return out
        profile = self._normalize_force_profile(entry, self._result_total_time())
        for idx, time_value in enumerate(times.tolist()):
            fx, fy, fz = self._eval_vector_profile_at_time(profile, float(time_value), ("fx", "fy", "fz"))
            out[idx, 0] = float(fx)
            out[idx, 1] = float(fy)
            out[idx, 2] = float(fz)
        return out

    def _response_series_unit_scale(self, quantity, subtype, values):
        arr = np.asarray(values, dtype=float).reshape(-1)
        finite = arr[np.isfinite(arr)]
        max_abs = float(np.max(np.abs(finite))) if finite.size > 0 else 0.0
        quantity = str(quantity or "").strip().lower()
        subtype = str(subtype or "").strip().lower()
        if quantity == "stress":
            if max_abs >= 1.0e9:
                return "GPa", 1.0e9
            if max_abs >= 1.0e6:
                return "MPa", 1.0e6
            if max_abs >= 1.0e3:
                return "kPa", 1.0e3
            return "Pa", 1.0
        if quantity == "strain":
            return "mm/mm", 1.0
        if quantity == "displacement":
            try:
                return str(getattr(self.window.view, "current_unit", "") or "m").strip(), 1.0
            except Exception:
                return "m", 1.0
        if quantity in {"force", "reaction_force"}:
            return "N", 1.0
        return "", 1.0

    def _finalize_response_curve(self, payload, y_vals, *, quantity, subtype, title=None):
        x_vals = self._result_time_values(len(y_vals))
        y_series = np.asarray(y_vals, dtype=float).reshape(-1)
        if x_vals.size != y_series.size:
            x_vals = np.arange(y_series.size, dtype=float)
        finite = np.isfinite(x_vals) & np.isfinite(y_series)
        if y_series.size <= 0 or not np.any(finite):
            return payload
        unit, scale = self._response_series_unit_scale(quantity, subtype, y_series)
        y_plot = np.asarray(y_series, dtype=float)
        if scale not in (0.0, 1.0):
            y_plot = y_plot / float(scale)
        subtype_label = payload.get("y_label", "Value")
        if unit:
            subtype_label = f"{subtype_label} [{unit}]"
        payload.update(
            {
                "available": True,
                "title": str(title or payload.get("title") or "Response vs Time"),
                "x_label": "Time (s)",
                "y_label": str(subtype_label),
                "x": np.asarray(x_vals, dtype=float),
                "y": np.asarray(y_plot, dtype=float),
                "frame_count": int(len(y_plot)),
                "message": "",
            }
        )
        return payload

    def _reduce_selected_series(self, values, selected_ids):
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 2:
            return None
        if not selected_ids:
            return None
        valid = [int(idx) for idx in selected_ids if 0 <= int(idx) < arr.shape[1]]
        if not valid:
            return None
        picked = np.asarray(arr[:, valid], dtype=float)
        if picked.ndim != 2:
            return None
        if picked.shape[1] == 1:
            return np.asarray(picked[:, 0], dtype=float)
        return np.nanmean(picked, axis=1)

    def _displacement_response_curve(self, subtype):
        payload = self._response_curve_payload("displacement", subtype)
        if self._displacement_history is None:
            payload["message"] = "This results set does not include displacement history."
            return payload
        selection = self._history_selection_data()
        node_ids = list(selection.get("node_ids") or self._history_selection_nodes or [])
        if self._history_selection_active and not node_ids:
            payload["message"] = "Select a node or geometry edge to plot displacement."
            return payload
        arr = None
        owns_handle = False
        try:
            arr, owns_handle = self._open_history_array(self._displacement_history)
            arr = np.asarray(arr, dtype=float)
        except Exception:
            payload["message"] = "Displacement history could not be read."
            return payload
        try:
            if arr.ndim != 3 or arr.shape[2] < 2:
                payload["message"] = "Displacement history has an unexpected shape."
                return payload
            frame_count = int(min(arr.shape[0], max(0, int(self._frame_count) or int(arr.shape[0]))))
            if frame_count <= 0:
                payload["message"] = "No displacement frames are available."
                return payload
            arr = arr[:frame_count]
            if str(subtype).lower() == "ux":
                values = np.asarray(arr[:, :, 0], dtype=float)
            elif str(subtype).lower() == "uy":
                values = np.asarray(arr[:, :, 1], dtype=float)
            else:
                values = np.linalg.norm(arr[:, :, :2], axis=2)
            series = self._reduce_selected_series(values, node_ids)
            if series is None:
                payload["message"] = "The selected target does not map to any result nodes."
                return payload
            return self._finalize_response_curve(payload, series, quantity="displacement", subtype=subtype)
        finally:
            if owns_handle:
                _close_numpy_handle(arr)

    def _stress_response_values(self, arr, subtype):
        data = np.asarray(arr, dtype=float)
        if data.ndim != 3 or data.shape[2] < 4:
            return None
        sxx = np.asarray(data[:, :, 0], dtype=float)
        syy = np.asarray(data[:, :, 1], dtype=float)
        sxy = np.asarray(data[:, :, 2], dtype=float)
        if str(subtype).lower() == "vm":
            return np.abs(np.asarray(data[:, :, 3], dtype=float))
        avg = 0.5 * (sxx + syy)
        radius = np.sqrt(np.maximum((0.5 * (sxx - syy)) ** 2 + sxy * sxy, 0.0))
        if str(subtype).lower() == "s2":
            return avg - radius
        return avg + radius

    def _strain_response_values(self, arr, subtype):
        data = np.asarray(arr, dtype=float)
        if data.ndim != 3 or data.shape[2] < 3:
            return None
        exx = np.asarray(data[:, :, 0], dtype=float)
        eyy = np.asarray(data[:, :, 1], dtype=float)
        gxy = np.asarray(data[:, :, 2], dtype=float)
        key = str(subtype).lower()
        if key == "eq":
            return np.sqrt(np.maximum(exx * exx + eyy * eyy - exx * eyy + 0.75 * gxy * gxy, 0.0))
        avg = 0.5 * (exx + eyy)
        radius = np.sqrt(np.maximum((0.5 * (exx - eyy)) ** 2 + (0.5 * gxy) ** 2, 0.0))
        if key == "e2":
            return avg - radius
        return avg + radius

    def _triangle_response_curve(self, quantity, subtype):
        payload = self._response_curve_payload(quantity, subtype)
        history = self._stress_history if quantity == "stress" else self._strain_history
        if history is None:
            payload["message"] = f"This results set does not include {quantity.replace('_', ' ')} history."
            return payload
        selection = self._history_selection_data()
        triangle_ids = list(selection.get("triangle_ids") or self._history_selection_triangles or [])
        if self._history_selection_active and not triangle_ids:
            payload["message"] = f"Select a triangle to plot {quantity.replace('_', ' ')}."
            return payload
        arr = None
        owns_handle = False
        try:
            arr, owns_handle = self._open_history_array(history)
            arr = np.asarray(arr, dtype=float)
        except Exception:
            payload["message"] = f"{quantity.replace('_', ' ').title()} history could not be read."
            return payload
        try:
            if arr.ndim != 3:
                payload["message"] = f"{quantity.replace('_', ' ').title()} history has an unexpected shape."
                return payload
            frame_count = int(min(arr.shape[0], max(0, int(self._frame_count) or int(arr.shape[0]))))
            if frame_count <= 0:
                payload["message"] = f"No {quantity.replace('_', ' ')} frames are available."
                return payload
            arr = arr[:frame_count]
            values = (
                self._stress_response_values(arr, subtype)
                if quantity == "stress"
                else self._strain_response_values(arr, subtype)
            )
            if values is None:
                payload["message"] = f"{quantity.replace('_', ' ').title()} data is not available for this component."
                return payload
            series = self._reduce_selected_series(values, triangle_ids)
            if series is None:
                payload["message"] = "The selected target does not map to any result triangles."
                return payload
            return self._finalize_response_curve(payload, series, quantity=quantity, subtype=subtype)
        finally:
            if owns_handle:
                _close_numpy_handle(arr)

    def _available_load_entries(self):
        state = getattr(self.window, "project_state", None)
        loads = getattr(state, "loads", None)
        if loads is None:
            loads = getattr(getattr(self.window, "view", None), "loads", None)
        return list(loads or [])

    def _force_response_curve(self, subtype):
        payload = self._response_curve_payload("force", subtype)
        selection = self._history_selection_data()
        load_matches = list(selection.get("load_matches") or [])
        if self._history_selection_active and not load_matches:
            payload["message"] = "Select a node or geometry edge with an applied load to plot force."
            return payload
        loads = self._available_load_entries()
        if not loads:
            payload["message"] = "This project does not define any applied force loads."
            return payload
        frame_count = int(max(0, self._frame_count))
        if frame_count <= 0:
            payload["message"] = "Load results before plotting a force history."
            return payload
        x_vals = self._result_time_values(frame_count)
        if x_vals.size <= 0:
            x_vals = np.arange(frame_count, dtype=float)
        y_vals = np.zeros(frame_count, dtype=float)
        any_match = False
        for match in load_matches:
            try:
                load_index = int(match.get("index"))
            except Exception:
                continue
            if not (0 <= load_index < len(loads)):
                continue
            scale = float(match.get("scale", 1.0) or 1.0)
            if abs(scale) <= 1e-12:
                continue
            series = self._force_entry_series(loads[load_index], x_vals)
            any_match = True
            key = str(subtype).lower()
            if key == "fx":
                y_vals += np.asarray(series[:, 0], dtype=float) * scale
            elif key == "fy":
                y_vals += np.asarray(series[:, 1], dtype=float) * scale
            else:
                y_vals += np.linalg.norm(series[:, :2], axis=1) * abs(scale)
        if not any_match:
            payload["message"] = "No applied force history is attached to the selected target."
            return payload
        return self._finalize_response_curve(payload, y_vals, quantity="force", subtype=subtype)

    def _reaction_force_response_curve(self, subtype):
        payload = self._response_curve_payload("reaction_force", subtype)
        selection = self._history_selection_data()
        bc_indices = list(selection.get("bc_indices") or [])
        if self._history_selection_active and not bc_indices:
            payload["message"] = "Select a BC target to plot reaction force."
            return payload
        payload["message"] = "Reaction force history is not exported in the current results format."
        return payload

    def response_curve(self, quantity=None, subtype=None):
        quantity_key, _spec = self._response_quantity_spec(quantity)
        subtype_key = str(subtype or self.default_response_plot_subtype(quantity_key)).strip().lower()
        if quantity_key == "displacement":
            return self._displacement_response_curve(subtype_key)
        if quantity_key == "force":
            return self._force_response_curve(subtype_key)
        if quantity_key == "reaction_force":
            return self._reaction_force_response_curve(subtype_key)
        if quantity_key in {"stress", "strain"}:
            return self._triangle_response_curve(quantity_key, subtype_key)
        payload = self._response_curve_payload(quantity_key, subtype_key)
        payload["message"] = "Unsupported response quantity."
        return payload

    def request_frame(self, index):
        if self._frame_count <= 0:
            return False

        frame_index = int(max(0, min(index, self._frame_count - 1)))
        cached = self._frame_cache.pop(frame_index, None)
        self._request_token += 1
        token = self._request_token

        if cached is not None:
            self._frame_cache[frame_index] = cached
            self.frameReady.emit(frame_index, cached)
            return True

        if self._loader is not None and self._loader.isRunning():
            self._pending_request = (frame_index, token)
            self._stop_loader(wait=False)
            return True

        self._pending_request = None
        self._start_loader(frame_index, token)
        return True

    def _resolve_results_paths(self, results_root=None):
        if results_root:
            base_dir = os.path.abspath(results_root)
            output_dir = os.path.join(base_dir, "output")
            results_dir = os.path.join(output_dir, "results")
            legacy_results_dir = os.path.join(base_dir, "results")
            pos_history_path = os.path.join(output_dir, "pos_history.npy")
            legacy_pos_history = os.path.join(base_dir, "pos_history.npy")
            if not os.path.isdir(results_dir) and os.path.isdir(legacy_results_dir):
                results_dir = legacy_results_dir
            if not os.path.isdir(results_dir) and os.path.isdir(base_dir):
                direct_results = [
                    f for f in os.listdir(base_dir)
                    if f.startswith("step_") and f.endswith(".csv")
                ]
                if direct_results:
                    results_dir = base_dir
            if not os.path.exists(pos_history_path):
                if os.path.exists(legacy_pos_history):
                    pos_history_path = legacy_pos_history
                else:
                    fallback = os.path.join(base_dir, "CPD-main", "source", "pos_history.npy")
                    if os.path.exists(fallback):
                        pos_history_path = fallback
        else:
            project_root = str(get_project_root())
            workspace_dir = str(get_workspace_dir())
            output_dir = os.path.join(workspace_dir, "output")
            results_dir = os.path.join(output_dir, "results")
            pos_history_path = os.path.join(output_dir, "pos_history.npy")
            legacy_results_dir = os.path.join(workspace_dir, "results")
            legacy_pos = os.path.join(workspace_dir, "pos_history.npy")
            if not os.path.exists(pos_history_path):
                if os.path.exists(legacy_pos):
                    pos_history_path = legacy_pos
                else:
                    cpd_legacy_pos = os.path.join(project_root, "CPD-main", "source", "pos_history.npy")
                    if os.path.exists(cpd_legacy_pos):
                        pos_history_path = cpd_legacy_pos
            if not os.path.isdir(results_dir):
                if os.path.isdir(legacy_results_dir):
                    results_dir = legacy_results_dir
                else:
                    repo_results = os.path.join(project_root, "results")
                    if os.path.isdir(repo_results):
                        results_dir = repo_results
        return results_dir, pos_history_path

    def _discover_csv_frames(self, results_dir):
        if not results_dir or not os.path.isdir(results_dir):
            return []
        result_files = sorted(
            f for f in os.listdir(results_dir)
            if f.startswith("step_") and f.endswith(".csv")
        )
        return [os.path.join(results_dir, fname) for fname in result_files]

    def _load_optional_histories(self, output_dir, expected_frames):
        self._displacement_history = self._load_history_file(output_dir, "displacement_history.npy", expected_frames)
        self._strain_history = self._load_history_file(output_dir, "strain_history.npy", expected_frames)
        self._stress_history = self._load_history_file(output_dir, "stress_history.npy", expected_frames)

    def _load_history_file(self, output_dir, filename, expected_frames):
        if not output_dir:
            return None
        path = os.path.join(output_dir, filename)
        if not os.path.exists(path):
            return None
        arr = None
        try:
            arr = np.load(path, mmap_mode="r")
        except Exception:
            return None
        if arr.ndim < 2 or int(arr.shape[0]) != int(expected_frames):
            _close_numpy_handle(arr)
            return None
        shape = tuple(int(v) for v in arr.shape)
        _close_numpy_handle(arr)
        return _history_ref(path, shape)

    def _history_peak_summary(self, history, *, mode=None, component=None, label="Field"):
        if history is None:
            return f"{label}: --"
        arr = None
        owns_handle = False
        try:
            arr, owns_handle = self._open_history_array(history)
            arr = np.asarray(arr, dtype=float)
            if mode == "vector_mag":
                vals = np.linalg.norm(arr[..., :2], axis=-1)
                peak = float(np.nanmax(vals))
                return f"{label}: max |U|={peak:.4g}"
            if component is not None:
                vals = np.asarray(arr[..., int(component)], dtype=float)
                finite = vals[np.isfinite(vals)]
                if finite.size <= 0:
                    return f"{label}: --"
                return f"{label}: max={float(np.max(np.abs(finite))):.4g}"
        except Exception:
            return f"{label}: --"
        finally:
            if owns_handle:
                _close_numpy_handle(arr)
        return f"{label}: --"

    def _history_curve_values(self, arr, *, source, kind):
        if source == "displacement":
            if arr.shape[2] < 2:
                return None
            return np.linalg.norm(arr[:, :, :2], axis=2)
        if source == "stress":
            if kind == "vm":
                if arr.shape[2] < 4:
                    return None
                return np.abs(np.asarray(arr[:, :, 3], dtype=float))
            return None
        if source == "strain":
            if arr.shape[2] < 3:
                return None
            exx = np.asarray(arr[:, :, 0], dtype=float)
            eyy = np.asarray(arr[:, :, 1], dtype=float)
            gxy = np.asarray(arr[:, :, 2], dtype=float)
            if kind == "eqv":
                return np.sqrt(np.maximum(exx * exx + eyy * eyy - exx * eyy + 0.75 * gxy * gxy, 0.0))
            return None
        return None

    def _reduce_history_series(self, values, *, reducer="max"):
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 2:
            return np.array([], dtype=float)
        finite = np.isfinite(arr)
        out = np.full((arr.shape[0],), np.nan, dtype=float)
        valid_rows = np.any(finite, axis=1)
        if not np.any(valid_rows):
            return out
        trimmed = arr[valid_rows]
        if str(reducer).lower() == "mean":
            out[valid_rows] = np.nanmean(trimmed, axis=1)
        else:
            out[valid_rows] = np.nanmax(trimmed, axis=1)
        return out

    def _history_curve_unit_scale(self, source, kind, values):
        arr = np.asarray(values, dtype=float).reshape(-1)
        finite = arr[np.isfinite(arr)]
        max_abs = float(np.max(np.abs(finite))) if finite.size > 0 else 0.0
        if source == "stress" and kind == "vm":
            if max_abs >= 1.0e9:
                return "GPa", 1.0e9
            if max_abs >= 1.0e6:
                return "MPa", 1.0e6
            if max_abs >= 1.0e3:
                return "kPa", 1.0e3
            return "Pa", 1.0
        if source == "strain":
            return "mm/mm", 1.0
        if source == "displacement":
            try:
                return str(getattr(self.window.view, "current_unit", "") or "m").strip(), 1.0
            except Exception:
                return "m", 1.0
        return "", 1.0

    def _result_time_values(self, frame_count):
        count = max(0, int(frame_count))
        if count <= 0:
            return np.array([], dtype=float)
        time_step = None
        write_every = None
        config_path = os.path.join(str(get_project_root()), "CPD-main", "config.yml")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as handle:
                    config = yaml.safe_load(handle) or {}
                sim = config.get("simulation", {}) or {}
                time_step = float(sim.get("time_step", 0.0))
                write_every = int(sim.get("write_every_steps", 0))
            except Exception:
                time_step = None
                write_every = None
        if time_step is None or time_step <= 0.0:
            return np.arange(count, dtype=float)
        if write_every is None or write_every <= 0:
            write_every = 1
        return np.arange(count, dtype=float) * float(time_step) * float(write_every)

    def _start_loader(self, frame_index, token):
        if self._source_kind == "csv":
            source_ref = self._csv_frame_paths[frame_index]
        else:
            source_ref = self._source_ref

        loader = FrameLoader(
            self._source_kind,
            source_ref,
            frame_index,
            token,
            unit_scale=self._unit_scale,
        )
        try:
            loader.setParent(self)
        except Exception:
            pass
        loader.frameLoaded.connect(self._on_frame_loaded)
        loader.loadFailed.connect(self._on_frame_failed)
        loader.finished.connect(self._on_loader_finished)
        loader.finished.connect(loader.deleteLater)
        self._loader = loader
        self.frameLoadStarted.emit(int(frame_index))
        loader.start()

    def _cache_frame(self, frame_index, frame_data):
        self._frame_cache.pop(frame_index, None)
        self._frame_cache[frame_index] = frame_data
        while len(self._frame_cache) > int(self._cache_size):
            self._frame_cache.popitem(last=False)

    def _clear_loader(self, loader):
        if loader is None:
            return
        if self._loader is loader:
            self._loader = None

    def _stop_loader(self, wait=True):
        loader = self._loader
        if loader is None:
            return
        try:
            if wait and hasattr(loader, "stop"):
                loader.stop()
            else:
                loader.requestInterruption()
            if wait and loader.isRunning():
                loader.wait(2000)
            if wait and loader.isRunning():
                loader.terminate()
                loader.wait(2000)
        except Exception:
            pass

    def stop(self):
        self._request_token += 1
        self._pending_request = None
        self._stop_loader(wait=True)
        self._loader = None

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass

    def _consume_pending_request(self):
        pending = self._pending_request
        self._pending_request = None
        if pending is None:
            return
        frame_index, token = pending
        if token != self._request_token:
            return
        cached = self._frame_cache.pop(frame_index, None)
        if cached is not None:
            self._frame_cache[frame_index] = cached
            self.frameReady.emit(frame_index, cached)
            return
        self._start_loader(frame_index, token)

    def _on_frame_loaded(self, frame_index, token, frame_data):
        loader = self.sender()
        self._clear_loader(loader)
        self._cache_frame(frame_index, frame_data)
        if token == self._request_token:
            self.frameReady.emit(frame_index, frame_data)
        self._consume_pending_request()

    def _on_frame_failed(self, frame_index, token, error_message):
        loader = self.sender()
        self._clear_loader(loader)
        if token == self._request_token:
            self.frameLoadFailed.emit(int(frame_index), str(error_message))
        self._consume_pending_request()

    def _on_loader_finished(self):
        loader = self.sender()
        if loader is None:
            return
        if self._loader is loader:
            self._loader = None
            self._consume_pending_request()
