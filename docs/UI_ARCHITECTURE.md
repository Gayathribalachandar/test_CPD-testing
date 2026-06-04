# UI Architecture

## Overview

CPD SimStudio uses a stage-oriented desktop UI for building particle-continuum
models, exporting solver inputs, and reviewing results. The UI is organized so
that workflow navigation, stage actions, viewport behavior, and solver export
remain aligned with the user pipeline:

`Geometry -> Particles -> Materials -> Interactions -> BC/Loads -> Solver -> Results`

## Main Components

### MainWindow

`main_window.py` is the primary UI shell. It is responsible for:
- application startup
- main layout composition
- workflow ribbon/stage switching
- wiring controllers, panels, viewports, and project state

### Controllers

Stage controllers live under `controllers/` and provide the workflow-facing
action layer for the major stages.

### Panels

`panels.py` contains the right-side stage panels and editor widgets for:
- geometry/part controls
- particles
- materials
- interactions
- BC/load definition
- solve controls
- results controls

### Viewport

The viewport layer is split across:
- `sketch_view.py` for 2D geometry, particle preview, export coordination, and
  results playback integration
- `viewport_3d.py` for 3D workspace navigation, selection, highlighting, and
  marker display

### Solver Exporter

Solver-facing file generation is handled through the UI/export layer in
`CPD-SIM-STUDIO`, with artifacts written to the runtime `workspace/` directory
for the bundled `CPD-main` backend.

## Controller Architecture

### geometry_controller

Handles geometry-stage progression and geometry-stage validation hooks before the
workflow advances to particle generation.

### particle_controller

Handles particle-generation entry points, particle rebuild requests, and
particle-readiness validation before solve.

### material_controller

Handles material-stage validation, especially ensuring required materials are
assigned before BC/load definition continues.

### interaction_controller

Owns workflow metadata and stage intent for interaction-definition workflows.

### bc_controller

Handles BC/load stage actions, including toolbar-driven apply actions and
solve-time validation that at least one BC or load exists.

### solver_controller

Coordinates solver settings synchronization, export preparation, and artifact
packaging for the solver handoff path.

### results_controller

Coordinates results restoration/loading paths and results-stage workflow
integration after solver artifacts are available.

## Data Flow

The primary UI data flow is:

`Geometry -> Particles -> Materials -> Interactions -> BC/Loads -> Solver -> Results`

At a high level:

1. Geometry is created or imported.
2. Particle generation converts geometry into particles and internal
   connections.
3. Materials and interactions are assigned.
4. BC/load definitions are stored in `ProjectState`.
5. Solver-export files are generated in `workspace/`.
6. Solver outputs are loaded back into the Results stage.

## Solver Export

The solver export contract includes the following primary files:

- `particles.csv`
- `connections.csv`
- `materials.csv`
- `velocity_targets.csv`
- `velocity_time.csv`
- `force_targets.csv`
- `force_time.csv`

These files are written in the runtime `workspace/` directory and are used by
the CPD solver launch path.

## Solver Output

The primary trajectory output consumed by the UI is:

- `pos_history.npy`

Additional frame/result files are written under `workspace/output/results/` and
loaded by the Results stage for playback and field visualization.
