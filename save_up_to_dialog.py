from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QRadioButton,
    QPushButton, QButtonGroup, QHBoxLayout
)
from project_stages import ProjectStage


class SaveUpToDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save Project Up To")
        self.setFixedSize(320, 290)

        layout = QVBoxLayout(self)

        label = QLabel("Save project up to the following stage:")
        layout.addWidget(label)

        self.stage_group = QButtonGroup(self)

        self.rb_geometry = QRadioButton("Geometry")
        self.rb_materials = QRadioButton("Materials")
        self.rb_bcs = QRadioButton("BCs & Loads")
        self.rb_mesh = QRadioButton("Connections")
        self.rb_job = QRadioButton("Job")
        self.rb_results = QRadioButton("Results")

        self.stage_group.addButton(self.rb_geometry, ProjectStage.GEOMETRY.value)
        self.stage_group.addButton(self.rb_materials, ProjectStage.MATERIALS.value)
        self.stage_group.addButton(self.rb_bcs, ProjectStage.BCS.value)
        self.stage_group.addButton(self.rb_mesh, ProjectStage.MESH.value)
        self.stage_group.addButton(self.rb_job, ProjectStage.JOB.value)
        self.stage_group.addButton(self.rb_results, ProjectStage.RESULTS.value)

        self.rb_mesh.setChecked(True)  # Default selection

        layout.addWidget(self.rb_geometry)
        layout.addWidget(self.rb_materials)
        layout.addWidget(self.rb_bcs)
        layout.addWidget(self.rb_mesh)
        layout.addWidget(self.rb_job)
        layout.addWidget(self.rb_results)

        layout.addStretch()

        # Buttons
        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("Save")
        btn_cancel = QPushButton("Cancel")

        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

        btn_layout.addStretch()
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)

        layout.addLayout(btn_layout)

    def get_selected_stage(self):
        checked_id = self.stage_group.checkedId()
        return ProjectStage(checked_id)
