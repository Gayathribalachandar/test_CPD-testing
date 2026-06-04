CPD-main Solver Notes
=====================

This folder contains the CPD solver backend used by the SimStudio Job stage.
It can also be run directly for standalone debugging.

Main files:
- config.yml
- source/run_cpd.py
- source/animate_cpd.py
- run_cpd

Runtime input/output locations:
- Solver inputs are typically read from `<project_root>/workspace/input/`
  (`particles.csv`, `fixed.csv`, `velocity.csv`).
- Input CSVs are expected with headers:
  - `particles.csv`: `particle_id,x,y`
  - `fixed.csv`: `particle_id`
  - `velocity.csv`: `particle_id,vx,vy`
- Time-target files are read from `<project_root>/workspace/input/`
  (`force_targets.csv`, `velocity_targets.csv`).
- Primary trajectory output is written to
  `<project_root>/workspace/output/pos_history.npy`.

Quick standalone run:
1) Open a terminal in this `CPD-main` folder.
2) Ensure exported input files exist under `../workspace/input/`.
3) Edit `config.yml` as needed.
4) Run:
   python3 source/run_cpd.py
5) Optional animation:
   python3 source/animate_cpd.py

Shortcut script:
- `./run_cpd` runs both simulation and animation.

Path resolution notes:
- Geometry file paths come from `config.yml -> geometry`.
- Relative geometry paths are resolved in this order:
  1) `<project_root>/workspace/<path_from_config>`
  2) `<project_root>/<path_from_config>`
  3) current working directory `<path_from_config>`
- Time profiles are read from `config.yml -> time_profiles`.
- `source/run_cpd.py` also keeps backward compatibility with legacy headerless
  setup/input CSVs.

Environment overrides supported by `source/run_cpd.py`:
- `CPD_WORKSPACE_DIR`
- `CPD_TIME_STEP`
- `CPD_TOTAL_STEPS`

Dependencies:
- numpy
- scipy
- numba
- pyyaml
- cupy (only when `simulation.device: gpu`)
