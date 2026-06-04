import sys
import importlib
from importlib import util as import_util

try:
    from importlib import metadata as importlib_metadata
except Exception:
    import importlib_metadata  # type: ignore

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QPlainTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


def _dependency_list():
    return [
        {
            "name": "Python",
            "required": True,
            "module": None,
            "package": None,
            "notes": "Runtime",
            "install_hint": "Use system Python 3.11+",
        },
        {"name": "PySide6", "required": True, "module": "PySide6", "package": "PySide6", "notes": "UI framework", "install_hint": "pip install PySide6"},
        {"name": "NumPy", "required": True, "module": "numpy", "package": "numpy", "notes": "Core math", "install_hint": "pip install numpy"},
        {"name": "SciPy", "required": True, "module": "scipy", "package": "scipy", "notes": "Delaunay/connection generation", "install_hint": "pip install scipy"},
        {"name": "Pandas", "required": True, "module": "pandas", "package": "pandas", "notes": "CSV tables", "install_hint": "pip install pandas"},
        {"name": "Shapely", "required": True, "module": "shapely", "package": "shapely", "notes": "Geometry", "install_hint": "pip install shapely"},
        {"name": "PyYAML", "required": True, "module": "yaml", "package": "PyYAML", "notes": "Config I/O", "install_hint": "pip install PyYAML"},
        {"name": "pyqtgraph", "required": False, "module": "pyqtgraph", "package": "pyqtgraph", "notes": "3D view", "install_hint": "pip install pyqtgraph"},
        {"name": "gmsh", "required": False, "module": "gmsh", "package": "gmsh", "notes": "Connection backend", "install_hint": "pip install gmsh"},
        {"name": "Triangle", "required": False, "module": "triangle", "package": "triangle", "notes": "Fast 2D connections", "install_hint": "pip install triangle"},
        {"name": "pygalmesh", "required": False, "module": "pygalmesh", "package": "pygalmesh", "notes": "CGAL connections", "install_hint": "pip install pygalmesh"},
        {"name": "OCP", "required": False, "module": "OCP", "package": "OCP", "notes": "CAD kernel", "install_hint": "pip install OCP"},
        {"name": "CuPy", "required": False, "module": "cupy", "package": "cupy", "notes": "GPU solver", "install_hint": "pip install cupy-cuda12x <match your CUDA version>"},
        {"name": "Numba", "required": False, "module": "numba", "package": "numba", "notes": "CPU acceleration", "install_hint": "pip install numba"},
    ]


def _check_dependency(dep):
    name = dep["name"]
    required = dep.get("required", False)
    module = dep.get("module")
    package = dep.get("package") or module
    notes = dep.get("notes", "")

    if name == "Python":
        version = sys.version.split()[0]
        return {
            "name": name,
            "status": "OK",
            "version": version,
            "notes": f"Required • {notes}",
        }

    available = False
    if module:
        try:
            available = import_util.find_spec(module) is not None
        except Exception:
            available = False
    status = "OK" if available else "Missing"
    version = "-"
    if available and package:
        try:
            version = importlib_metadata.version(package)
        except Exception:
            version = "unknown"

    return {
        "name": name,
        "status": status,
        "version": version,
        "notes": f"{'Required' if required else 'Optional'} • {notes}",
    }


def collect_dependency_report():
    deps = _dependency_list()
    rows = [_check_dependency(dep) for dep in deps]
    required_missing = 0
    optional_missing = 0
    for row in rows:
        if row["status"] != "OK":
            if "Required" in row["notes"]:
                required_missing += 1
            else:
                optional_missing += 1
    return rows, required_missing, optional_missing


class DependencyCheckDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dependency Check")
        self.setMinimumSize(720, 420)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.summary_label = QLabel("")
        self.summary_label.setObjectName("SummaryLabel")
        layout.addWidget(self.summary_label)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Dependency", "Status", "Version", "Notes"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)

        btn_row = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh")
        self.copy_btn = QPushButton("Copy Report")
        btn_row.addWidget(self.refresh_btn)
        btn_row.addWidget(self.copy_btn)
        btn_row.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        btn_row.addWidget(buttons)
        layout.addLayout(btn_row)

        self.install_hint_box = QPlainTextEdit()
        self.install_hint_box.setReadOnly(True)
        self.install_hint_box.setMaximumHeight(120)
        self.install_hint_box.setObjectName("HintBox")
        hint_label = QLabel("Install hints for missing dependencies:")
        hint_label.setObjectName("SectionTitleLabel")
        layout.addWidget(hint_label)
        layout.addWidget(self.install_hint_box)

        buttons.rejected.connect(self.reject)
        self.refresh_btn.clicked.connect(self.refresh)
        self.copy_btn.clicked.connect(self.copy_report)

        self._last_report = ""
        self.refresh()

    def refresh(self):
        deps = _dependency_list()
        rows, required_missing, optional_missing = collect_dependency_report()
        self.table.setRowCount(len(rows))

        for row_idx, row in enumerate(rows):
            for col_idx, key in enumerate(("name", "status", "version", "notes")):
                item = QTableWidgetItem(str(row[key]))
                if key == "status":
                    if row[key] == "OK":
                        item.setForeground(QBrush(QColor(22, 163, 74)))
                    else:
                        item.setForeground(QBrush(QColor(220, 38, 38)))
                self.table.setItem(row_idx, col_idx, item)

        self.table.resizeColumnsToContents()
        self.summary_label.setText(
            f"Required missing: {required_missing} | Optional missing: {optional_missing}"
        )

        lines = ["CPD Dependency Report:"]
        for row in rows:
            lines.append(
                f"- {row['name']}: {row['status']} (version: {row['version']}) [{row['notes']}]"
            )
        hint_lines = []
        for dep, row in zip(deps, rows):
            if row["status"] != "OK":
                hint = dep.get("install_hint")
                if hint:
                    hint_lines.append(f"{row['name']}: {hint}")
        if hint_lines:
            lines.append("")
            lines.append("Install hints:")
            lines.extend(f"- {hint}" for hint in hint_lines)
            self.install_hint_box.setPlainText("\n".join(hint_lines))
        else:
            self.install_hint_box.setPlainText("All detected dependencies are available.")
        self._last_report = "\n".join(lines)

    def copy_report(self):
        if not self._last_report:
            return
        QApplication.clipboard().setText(self._last_report)
