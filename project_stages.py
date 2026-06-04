# project_stages.py
from enum import Enum

class ProjectStage(Enum):
    GEOMETRY = 1
    MATERIALS = 2
    FLUID = 3
    INTERFACES = 4
    INTERACTIONS = 4  # Backward-compatible alias (use INTERFACES)
    BCS = 5
    FRACTURE = 6
    LOADS = 7
    MESH = 8
    JOB = 9
    RESULTS = 10
