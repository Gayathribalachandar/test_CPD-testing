# CPD SimStudio

CPD SimStudio is a desktop GUI for building CPD models, generating particle
connections, exporting solver inputs, running the bundled CPD solver, and
reviewing results.

## Version

Current version: `0.1.0`

## Latest Updates

- Latest features/fixes: see `CHANGELOG.md` (`[Unreleased]` first).
- Release history: see version sections in `CHANGELOG.md`.
- Project rule: update `CHANGELOG.md` whenever code/docs behavior changes.

The current project supports:
- 2D geometry-to-solver workflow
- 3D workspace for primitives/CAD and 3D connection preview/export
- CPD project save/load with schema migration (`.cpd`)
- mesh/connection preview modes for large particle counts
- solver handoff to `CPD-main`

## Repository Layout

- `main_window.py`: main GUI entry point
- `sketch_view.py`: geometry, meshing/export, visualization bridge
- `panels.py`: stage-specific UI panels (assembly/materials/interfaces/BCs/loads/mesh/job/results)
- `CPD-main/`: standalone CPD solver core (`config.yml`, `source/run_cpd.py`)
- `requirements.txt`: Python dependencies for SimStudio
- `TEST_RUN_GUIDE.md`: smoke-test checklist linked from the app Help menu
- `TESTER_README.md`: packaged install/run guide for Linux and Windows analysts/testers
- `../tests/` (in wrapper repo, if present): lightweight regression tests for
  geometry/mesh/solver I/O helpers

## Prerequisites

- Python 3.11 or newer (for source run)
- Linux/macOS/Windows with OpenGL-capable graphics driver
- Write access to the runtime workspace location
- For packaged binaries, use the Linux/Windows release zip assets

## Install as Software (No Python Required)

For analysts/testers who want a packaged app install/run flow, use:

- `TESTER_README.md`

That guide covers:
- Downloading release assets (`CPD-SimStudio-win.zip`, `CPD-SimStudio-linux.zip`)
- First run on Windows/Linux
- Runtime log/data locations
- Professional smoke-test and bug-report checklist

## Runtime Workspace Paths

- Source run (`python main_window.py`): `CPD-SIM-STUDIO/workspace/`
- Packaged Windows build (PyInstaller): `%APPDATA%\CPD-SimStudio\workspace\`
- Packaged Linux build (PyInstaller): `~/.local/share/CPD-SimStudio/workspace/`
- Packaged macOS build (if built manually): `~/Library/Application Support/CPD-SimStudio/workspace/`

## CI/CD Automation

- Push/PR (all branches): `.github/workflows/ci-build.yml`
  - Runs dependency install and unit tests (if `tests/` exists)
  - Builds Windows and Linux PyInstaller one-dir artifacts
  - Uploads folder artifacts and zipped packages for Windows/Linux to the workflow run
- Daily scheduled build (UTC): `.github/workflows/ci-build.yml`
  - Runs once per day to keep shareable binaries fresh even without new pushes
- Rolling nightly pre-release: `.github/workflows/nightly-pre-release.yml`
  - Runs on push to `main`/`master`, daily schedule, and manual dispatch
  - Publishes/updates a prerelease under tag `nightly` with stable download links:
    - `https://github.com/<owner>/<repo>/releases/download/nightly/CPD-SimStudio-linux-nightly.zip`
    - `https://github.com/<owner>/<repo>/releases/download/nightly/CPD-SimStudio-win-nightly.zip`
- Tag push (`v*`): `.github/workflows/release.yml`
  - Builds Linux and Windows release artifacts
  - Packages zip files
  - Publishes GitHub Release assets automatically

Tag example:

```bash
git tag v0.1.1
git push origin v0.1.1
```

## Update Check (In App)

SimStudio includes a release-check skeleton:
- Startup check (optional)
- Help menu action: `Help -> Check for Updates`

Configure with environment variables:
- `CPD_UPDATE_REPO=owner/repo`
- `CPD_UPDATE_CHECK_ON_STARTUP=1` (default) or `0`
- `CPD_UPDATE_TIMEOUT_SEC=4.0`

The app compares `app_config.__version__` with the latest GitHub Release tag
and opens the release/asset URL when an update is available.

## Setup (Source Run)

Run from the `CPD-SIM-STUDIO` directory.

Linux/macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Launch

Linux/macOS:

```bash
source .venv/bin/activate
python main_window.py
```

or use launcher script:

```bash
./launch_cpd.sh
```

`main_window.py` auto-reexecs with `.venv/bin/python` when that environment is
present, so the app and dependencies stay consistent.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python .\main_window.py
```

## Optional Dependencies

Install only if you need the corresponding capability:

- `pip install gmsh`: volumetric 3D connection generation through Gmsh
- `pip install OCP`: CAD kernel support (STEP/IGES/STL + CAD boolean workflow)
- `pip install cupy-cuda12x`: GPU solver path in `CPD-main` (match your CUDA)
- `pip install triangle`: optional fast 2D connection backend
- `pip install pygalmesh`: optional CGAL-based connection backend

## UI Workflow

The primary workflow is now:

`Geometry -> Particles -> Materials -> Interactions -> BC/Loads -> Solve -> Results`

What happens in each stage:

1. `Geometry`
   Create or import clean parts, sketch primitives, confirm solids, and prepare
   the model for particle generation.
2. `Particles`
   Set particle spacing, generate particles from the confirmed geometry, and
   rebuild the particle set when geometry changes.
3. `Materials`
   Define materials and assign them to parts.
4. `Interactions`
   Define contact/interface-style interaction relationships between neighboring
   parts or particle regions.
5. `BC/Loads`
   Apply displacement/velocity boundary conditions and force-style loads to the
   selected model region.
6. `Solve`
   Export solver inputs and launch the bundled CPD solver workflow.
7. `Results`
   Load animation data, play frames, and inspect displacement/stress/strain-style
   result fields.

The UI is now icon-first:
- Toolbars use icons instead of large text labels.
- Detailed descriptions are exposed through hover tooltips.
- Workflow switching follows the compact stage stepper at the top of the window.

## Solver Export and Output Files

SimStudio exports solver-facing files to the runtime `workspace/` directory.

Common exported files:
- `workspace/particles.csv`
- `workspace/connections.csv`
- `workspace/materials.csv`
- `workspace/velocity_targets.csv`
- `workspace/velocity_time.csv`
- `workspace/force_targets.csv`
- `workspace/force_time.csv`

Solver outputs are written to:
- `workspace/output/pos_history.npy`
- `workspace/output/results/`

The application also writes supporting runtime files under `workspace/input/`
for the bundled `CPD-main` launch path.

## Architecture Overview

The UI now uses a controller-oriented structure. Stage-specific workflow logic is
organized under:

- `controllers/geometry_controller.py`
- `controllers/particle_controller.py`
- `controllers/material_controller.py`
- `controllers/interaction_controller.py`
- `controllers/bc_controller.py`
- `controllers/solver_controller.py`
- `controllers/results_controller.py`

`MainWindow` is now primarily a UI shell responsible for:
- window/layout creation
- workflow ribbon and stage switching
- controller initialization and wiring

Stage controllers coordinate UI actions, `ProjectState` updates, viewport
updates, and solver/export handoff.

Important limitation:
- 3D mode currently supports 3D connection preview/export, but solver execution
  remains the 2D CPD backend.

## Files Generated During Export/Run

Typical solver input files are written under the runtime `workspace/`:
- `workspace/solver_particles.csv`
- `workspace/connections.csv`
- `workspace/bc.csv`
- `workspace/loads.csv`
- `workspace/input/particles.csv`
- `workspace/input/fixed.csv`
- `workspace/input/velocity.csv`
- `workspace/input/force_targets.csv`
- `workspace/input/velocity_targets.csv`
- `workspace/input/force_time.csv`
- `workspace/input/velocity_time.csv`

CSV schema now keeps explicit IDs and headers for robustness:
- `workspace/particles.csv`: `particle_id,x,y,meta`
- `workspace/connections.csv`: `triangle_id,p1,p2,p3,part_id,material_id,zone_kind,interface_id,meta`
- `workspace/input/particles.csv`: `particle_id,x,y` (solver input)
- `workspace/input/fixed.csv`: `particle_id`
- `workspace/input/velocity.csv`: `particle_id,vx,vy`
- `workspace/input/force_targets.csv`: `particle_id,force_id_1,...`
- `workspace/input/velocity_targets.csv`: `particle_id,velocity_id_1,...`

Solver outputs are written under the runtime `workspace/output/`:
- `workspace/output/results/step_*.csv`
- `workspace/output/pos_history.npy`
- `workspace/output/initial_pos.csv`
- `workspace/output/final_pos.csv`
- `workspace/saved_results/` (saved snapshots from the Results panel)

When you save a project, SimStudio can also package data to:
- `<project_name>_artifacts/inputs/`
- `<project_name>_artifacts/results/`

## Tests

If you are in the wrapper repository (`CPD-SIM-UI`) that contains `tests/`:

```bash
source CPD-SIM-STUDIO/.venv/bin/activate
python -m unittest discover -s tests
```

Windows PowerShell equivalent:

```powershell
.\CPD-SIM-STUDIO\.venv\Scripts\Activate.ps1
python -m unittest discover -s tests
```

If you are inside this repository and a local `tests/` folder exists:

```bash
source .venv/bin/activate
python -m unittest discover -s tests
```

Windows PowerShell equivalent:

```powershell
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s tests
```

## Troubleshooting

- Missing module at launch:
  Install dependencies with `pip install -r requirements.txt`.
- 3D CAD import unavailable:
  Install `OCP` and relaunch from the same `.venv`.
- Gmsh meshing errors:
  Install `gmsh` Python API and ensure the imported CAD contains closed solids.
- Empty/noisy preview for very large models:
  adjust fast preview / GPU point preview settings in the Mesh panel.

## Testing the Current Build

This version includes a large UI architecture refactor.

Testers should focus on validating the full modeling workflow:

1. Geometry creation
2. Particle generation
3. Material assignment
4. Interaction definition
5. Boundary conditions and loads
6. Solver execution
7. Results visualization

Detailed testing instructions are available in:

TEST_RUN_GUIDE.md
TESTER_README.md

Bug reports should include:

• operating system  
• Python version  
• steps to reproduce  
• screenshot or video  
• exported workspace files when solver related