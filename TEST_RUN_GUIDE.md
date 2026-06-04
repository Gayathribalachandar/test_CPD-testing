# CPD SimStudio Test Run Guide

Use this checklist before sharing builds or onboarding a new team member.

## 1) Environment Setup

Linux/macOS:

```bash
cd CPD-SIM-STUDIO
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
cd CPD-SIM-STUDIO
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional feature dependencies:

```bash
pip install gmsh OCP
```

## 2) Launch Smoke Test

Linux/macOS:

```bash
source .venv/bin/activate
python main_window.py
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python .\main_window.py
```

Check:
1. App starts without import errors.
2. Startup dialog appears (New 2D/New 3D/Open Recent).
3. Help -> Dependency Check opens and reports required deps as available.

Runtime workspace locations:
- Source run: `CPD-SIM-STUDIO/workspace/`
- Packaged Windows build: `%APPDATA%\CPD-SimStudio\workspace\`
- Packaged Linux build: `~/.local/share/CPD-SimStudio/workspace/`
- Packaged macOS build (if built manually): `~/Library/Application Support/CPD-SimStudio/workspace/`

## 3) 2D Workflow Smoke Test

1. Create `New 2D`.
2. Draw one rectangle in Geometry.
3. Click `Confirm Part`.
4. Move to Materials and assign one material.
5. Move to BC/Loads and add at least one fix or load.
6. In Mesh stage click `Preview Connections`.
7. In Job stage click `Run Simulation`.
8. In Results stage verify frame slider/playback works.

Expected artifacts in the runtime `workspace/`:
- `solver_particles.csv`
- `connections.csv`
- `input/particles.csv`, `input/fixed.csv`, `input/velocity.csv`
- `input/force_targets.csv`, `input/velocity_targets.csv`
- `output/results/step_*.csv` or `output/pos_history.npy`

## UI Functional Testing

Test the stage workflow in order and verify both visible behavior and generated
workspace artifacts.

### 1. Geometry

- create rectangle
- create circle
- extrude geometry
- import geometry

### 2. Particle Generation

- generate particles
- verify particle preview
- verify connections preview

### 3. Materials

- create material
- assign material to part

### 4. Interactions

- create interaction between parts
- verify interface visualization

### 5. Boundary Conditions

- apply displacement BC
- apply velocity BC
- apply force load
- verify BC markers appear in viewport

### 6. Solver

- run simulation
- confirm solver launches
- verify `pos_history.npy` is produced

### 7. Results

- load animation
- verify displacement visualization
- verify frame slider works

## 4) Project Save/Reload Check

1. Save project as `sample.cpd`.
2. Confirm `sample_artifacts/inputs` exists.
3. Close and reopen `sample.cpd`.
4. Verify geometry/materials/BCs are restored and stage progression is valid.

## 5) Optional 3D Checks

1. Create `New 3D`.
2. Add a 3D primitive or import CAD.
3. Generate 3D preview connections.
4. Export and verify:
- `workspace/particles_3d.csv`
- `workspace/connections_3d.csv`

Note:
- Current solver execution path is still 2D CPD backend.

## 6) Automated Unit Tests

From repository root (`CPD-SIM-UI`) on Linux/macOS:

```bash
source CPD-SIM-STUDIO/.venv/bin/activate
python -m unittest discover -s tests
```

From repository root (`CPD-SIM-UI`) on Windows PowerShell:

```powershell
.\CPD-SIM-STUDIO\.venv\Scripts\Activate.ps1
python -m unittest discover -s tests
```

If you only have this repository checked out, run the command above only when a
local `tests/` directory is present.

Passing tests currently cover:
- Geometry helpers
- Connection/node mapping helpers
- Solver input loading path (`CpdEngine`)
