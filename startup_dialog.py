from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class StartupDialog(QDialog):
    def __init__(self, recent_paths=None, show_on_startup=True, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Start")
        self.setMinimumWidth(420)
        self.action = None
        self.selected_path = None

        recent_paths = recent_paths or []

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Start a Project</b>"))

        button_row = QWidget()
        button_layout = QHBoxLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        self.new_2d_btn = QPushButton("New 2D")
        self.new_2d_btn.clicked.connect(lambda: self._choose_action("new_2d"))
        self.new_3d_btn = QPushButton("New 3D")
        self.new_3d_btn.clicked.connect(lambda: self._choose_action("new_3d"))
        button_layout.addWidget(self.new_2d_btn)
        button_layout.addWidget(self.new_3d_btn)
        layout.addWidget(button_row)

        layout.addWidget(QLabel("Open Recent:"))
        self.recent_list = QListWidget()
        for path in recent_paths:
            item = QListWidgetItem(path)
            self.recent_list.addItem(item)
        self.recent_list.itemDoubleClicked.connect(self._open_selected)
        layout.addWidget(self.recent_list)

        open_btn = QPushButton("Open Selected")
        open_btn.clicked.connect(self._open_selected)
        open_btn.setEnabled(self.recent_list.count() > 0)
        layout.addWidget(open_btn, alignment=Qt.AlignLeft)

        self._show_checkbox = QCheckBox("Show this on startup")
        self._show_checkbox.setChecked(show_on_startup)
        layout.addWidget(self._show_checkbox)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _choose_action(self, action):
        self.action = action
        self.accept()

    def _open_selected(self):
        item = self.recent_list.currentItem()
        if not item:
            return
        self.selected_path = item.text()
        self.action = "open_recent"
        self.accept()

    def show_on_startup(self):
        return self._show_checkbox.isChecked()
