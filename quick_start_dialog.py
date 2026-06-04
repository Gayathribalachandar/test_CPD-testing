import os

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)


QUICK_START_TEXT = """\
CPD SimStudio Quick Start (Example: Single Rectangle)

1) Launch
   - Run: ./launch_cpd.sh
   - Alternative: python main_window.py (inside active .venv)

2) Geometry
   - Select Rect tool and drag a rectangle.
   - Click Confirm Part and name it.

3) Materials
   - Open Materials stage.
   - Right-click the part in the canvas to assign a material.

4) BC/Loads
   - Open BC/Loads stage.
   - Right-click a vertex or edge and add a fix or load.

5) Connections
   - Click Preview Connections in the top toolbar.
   - Use the Connection View toggle to show/hide connections in the canvas.

6) Job
   - Open Job stage and click Run Simulation.
   - Visualization should auto-start in the main canvas.

Need the full checklist?
- Open TEST_RUN_GUIDE.md in the project folder.
"""


class QuickStartDialog(QDialog):
    def __init__(self, show_on_startup=True, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Quick Start")
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        title = QLabel("<b>Quick Start</b>")
        layout.addWidget(title)

        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(QUICK_START_TEXT)
        text.setMinimumHeight(240)
        text.setLineWrapMode(QTextEdit.WidgetWidth)
        layout.addWidget(text)

        open_guide = QPushButton("Open Test Guide")
        open_guide.clicked.connect(self._open_test_guide)
        layout.addWidget(open_guide, alignment=Qt.AlignLeft)

        self._show_checkbox = QCheckBox("Show this on startup")
        self._show_checkbox.setChecked(show_on_startup)
        layout.addWidget(self._show_checkbox)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    def show_on_startup(self):
        return self._show_checkbox.isChecked()

    def _open_test_guide(self):
        guide_path = os.path.join(os.path.dirname(__file__), "TEST_RUN_GUIDE.md")
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(guide_path)))
