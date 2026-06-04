# Changelog

All notable changes to CPD SimStudio are documented in this file.

When making any code or documentation change, add a short entry to
`[Unreleased]` in the same commit.

## [Unreleased]

### UI Architecture Refactor

- Introduced stage controller architecture.
- Split UI logic into controllers.
- Implemented icon-first interface with tooltip labels.
- Replaced ribbon with compact workflow stepper.
- Added new workflow stages:
  `Geometry -> Particles -> Materials -> Interactions -> BC/Loads -> Solve -> Results`
- Improved BC and load definition UI.
- Improved solver CSV export reliability.
- Added explicit particle and connection export schemas.
- Improved result loading and visualization.

### Added

- Added this changelog to track latest features and fixes in Git.
- Added cross-platform quick-start/test instructions in project docs.
- Added a packaged install/run guide for Linux and Windows testers
  (`TESTER_README.md`).
- Added a persistent stage-hints toggle (`Hints` button + `View -> Stage Hints`)
  so helper instructions can be enabled/disabled per user preference.
- Added explicit BC/Load operation buttons in the BC/Loads panel
  (`Edit/Delete BC`, `Edit/Delete Load`) for clearer direct operations.
- Added inline results field controls (displacement, stress-sample, strain-sample)
  and in-panel summary/activity labels so compare workflows no longer depend on
  a popup tree dialog.
- Added a dedicated `Loads` stage after `BCs` so BC and load workflows are now
  separated into distinct tabs/stages (Abaqus-style progression).
- Added sketch `Smart Dim` support for arc-length dimensions (`L`) in addition
  to existing line/angle/radius/diameter dimensions, including in-canvas
  annotation placement and editable values.
- Added daily CI scheduling in `.github/workflows/ci-build.yml` so Linux/Windows
  binaries are rebuilt automatically once per day.
- Added Linux zip packaging to CI build workflow (non-release runs) so both
  Linux and Windows zipped artifacts are available on each push run.
- Added rolling nightly prerelease automation in
  `.github/workflows/nightly-pre-release.yml` (push to `main`/`master` + daily
  schedule + manual dispatch) with stable `nightly` release download links.

### Changed

- Standardized CPD-main IO folders to `workspace/input/` (solver geometry CSVs)
  and `workspace/output/` (trajectory/results files), and updated UI export/load
  paths accordingly with legacy fallbacks for older `workspace/setup/` and root
  workspace result files.
- Updated Run behavior to validate missing prerequisite stages (Geometry,
  Materials, BC/Loads, Mesh) and redirect users to the required stage instead
  of forcing mesh rebuild on every run.
- Consolidated solver-facing target/time CSVs under `workspace/input/`
  (`force_targets.csv`, `velocity_targets.csv`, `force_time.csv`,
  `velocity_time.csv`) and removed redundant legacy derivative exports
  (`fixed_particles.csv`, `particle_forces.csv`, `particle_velocities.csv`).
- Removed deprecated `workspace/input/for.csv` from the active solver flow;
  force input now uses `force_targets.csv` + `time_profiles.forces`.
- Updated `Run Simulation` export flow to refresh a full, synchronous set of
  workspace CSVs for each run and clean stale files that are not relevant to
  the current mode (for example stale 3D CSVs in 2D runs).
- Fixed mesh reload/export mapping so part metadata from `connections.csv`
  is read correctly and `Run Simulation` now forces remesh before export,
  preventing stale/empty BC target mapping in generated workspace CSVs.
- Hardened results animation item updates to handle deleted Qt graphics objects
  safely, preventing repeated `RuntimeError` crashes in
  `advance_animation_frame`.
- Tightened `fixed.csv` export so it includes only particles with both
  `ux=0` and `uy=0` from fixed constraints (velocity-prescribed nodes are
  excluded).
- Standardized solver/setup CSV exports to explicit headered ID schemas
  (`particle_id`, `triangle_id`, `material_serial` where relevant), and updated
  `run_cpd.py` to read both new headered files and legacy headerless files.
- Updated solver notes in `CPD-main/readme.txt` to match current
  `workspace/`-based input and output behavior.
- Updated `CPD-main/source/animate_cpd.py` to load
  `workspace/output/pos_history.npy` first, with legacy fallback paths.
- Updated READMEs to link packaged software install instructions for
  analysts/testers.
- Standardized solver-facing velocity time-series CSV export to SI (`m/s`)
  in `velocity_time.csv`.
- Removed legacy duplicate `workspace/input/nodes.csv` generation; solver setup
  now keeps a single canonical `workspace/input/particles.csv`.
- Deduplicated mapped BC/load node targets during CSV export to avoid duplicate
  rows and duplicate force/moment accumulation for the same target source.
- Exported `bc.csv` and `solver_particles.csv` in SI (MKS) units for solver-
  facing consistency.
- Updated mesh metadata/load preference to use `particles.csv` first for UI
  reloads, while keeping SI solver data in `solver_particles.csv`.
- Stopped mirroring `force_targets.csv`, `velocity_targets.csv`, `force_time.csv`,
  and `velocity_time.csv` into `workspace/input/`; these now remain only in
  `workspace/` (single source of truth).
- Changed BC force application flow to use an inline axis selector instead of
  repeated force-axis popup dialogs.
- Changed project reset/load/import flows to explicitly reset BC/Results panel
  stage state, preventing stale UI details from a previous project/session.
- Changed material assignment flow to emit explicit success feedback on assign.
- Split the former `BC/Loads` panel into separate `BCs` and `Loads` panels with
  mode-specific apply/edit/delete controls and stage-aware 2D/3D context menu
  actions.
- Rearranged the Results stage layout into a mixed horizontal/vertical card
  layout so controls, field compare, frame playback, display toggles, and
  activity summaries are clearer and less cluttered.
- Updated sketch toolbar wording from `Dimension` to `Smart Dim`, with clearer
  hints for CAD-style dimensioning workflows.
- Updated README automation docs to reflect push/PR CI scope, daily scheduled
  builds, Linux/Windows zip artifacts, and separate `BCs` + `Loads` stages.
- Updated tester docs with nightly prerelease asset names for faster sharing of
  in-progress builds.

### Fixed

- Fixed paint-mode toggling in BC/Loads so it no longer deletes the currently
  selected load row.
- Fixed solver export validation for force/velocity mappings: export now aborts
  with a clear warning when loads/BCs exist but map to zero mesh nodes.
- Fixed visualization reset handling by clearing animation frame state on
  `clear_all`, preventing stale results status in the Results stage.

## [0.1.0] - 2026-02-20

### Added

- Added Linux and Windows CI build matrix workflow.
- Added tagged-release workflow that builds and publishes Linux/Windows zip assets.
- Added Linux and Windows PyInstaller build scripts and spec files.
- Added in-app update-check scaffolding (`Help -> Check for Updates`).
- Added tester and smoke-test guides for onboarding and verification.

### Changed

- Updated runtime workspace handling for packaged builds on Windows/Linux.
- Updated setup/build documentation for release packaging.
