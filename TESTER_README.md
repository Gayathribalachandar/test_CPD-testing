# CPD SimStudio Packaged Install and Run Guide (Linux + Windows)

This guide is for analysts/testers who want to run CPD SimStudio as packaged
software (no Python setup).

Important:
- Current release assets are provided for Windows and Linux.
- Assets are published from tagged releases (`v*`) in GitHub Releases.
- A rolling nightly prerelease is also available under tag `nightly` for
  latest in-progress builds.

## 1) Download the Correct Asset

From GitHub Releases:

- Windows: `CPD-SimStudio-win.zip`
- Linux: `CPD-SimStudio-linux.zip`

Use only assets from the same release tag when comparing test results.

Nightly prerelease assets (stable links):

- Windows: `CPD-SimStudio-win-nightly.zip`
- Linux: `CPD-SimStudio-linux-nightly.zip`

## 2) Windows Install and First Run

1. Extract `CPD-SimStudio-win.zip` to a stable location (for example
   `C:\Tools\CPD-SimStudio\`).
2. Open the extracted app folder (typically `CPD-SimStudio/`).
3. Run `CPD-SimStudio.exe`.
4. If Windows SmartScreen appears, select `More info` -> `Run anyway`.
5. Optional: create a desktop shortcut to `CPD-SimStudio.exe`.

Runtime data path:
- `%APPDATA%\CPD-SimStudio\workspace\`

Key runtime files:
- Log: `%APPDATA%\CPD-SimStudio\workspace\logs\app.log`
- Export inputs: `%APPDATA%\CPD-SimStudio\workspace\*.csv`
- CPD-main inputs: `%APPDATA%\CPD-SimStudio\workspace\input\*.csv`
- Results frames: `%APPDATA%\CPD-SimStudio\workspace\output\results\step_*.csv`
- Trajectory: `%APPDATA%\CPD-SimStudio\workspace\output\pos_history.npy`
- Autosave: `%APPDATA%\CPD-SimStudio\workspace\autosave\`

## 3) Linux Install and First Run

1. Extract `CPD-SimStudio-linux.zip` (for example under `~/opt/`).
2. Open `CPD-SimStudio-linux/CPD-SimStudio/`.
3. Ensure the binary is executable:

```bash
chmod +x CPD-SimStudio
```

4. Launch:

```bash
./CPD-SimStudio
```

Runtime data path:
- `~/.local/share/CPD-SimStudio/workspace/`

Key runtime files:
- Log: `~/.local/share/CPD-SimStudio/workspace/logs/app.log`
- Export inputs: `~/.local/share/CPD-SimStudio/workspace/*.csv`
- CPD-main inputs: `~/.local/share/CPD-SimStudio/workspace/input/*.csv`
- Results frames: `~/.local/share/CPD-SimStudio/workspace/output/results/step_*.csv`
- Trajectory: `~/.local/share/CPD-SimStudio/workspace/output/pos_history.npy`
- Autosave: `~/.local/share/CPD-SimStudio/workspace/autosave/`

## 4) Professional Smoke Test Checklist

1. Launch the app.
2. Create `New 2D`.
3. Draw one rectangle and confirm part.
4. Assign one material.
5. Add at least one BC/load.
6. Generate connection preview.
7. Run simulation.
8. Confirm results playback works.
9. Confirm runtime files/log are created at the OS-specific workspace path.

## Testing the New UI

The UI has recently undergone a large refactor. Testers should expect both
visible workflow changes and internal architecture cleanup.

Major visible changes:
- icon-based workflow navigation
- compact workflow stepper
- particle-based modeling workflow
- controller-driven architecture

When reporting a bug include:

1. operating system
2. Python version (if running from source)
3. steps to reproduce
4. screenshot if possible
5. exported workspace files if solver related

## 5) Bug Report Checklist (Windows/Linux)

Include:

1. OS + version/build.
2. App release tag and asset name used (for example `v0.1.1`, `CPD-SimStudio-win.zip`).
3. Exact reproduction steps.
4. Expected vs actual behavior.
5. Screenshot/screen recording for UI issues.
6. `app.log` from runtime workspace.
7. If simulation-related: attach exported inputs and result files from workspace.

## 6) Upgrade to a New Release

1. Close running CPD SimStudio instance.
2. Download new release zip for your OS.
3. Extract to a new folder or replace old app folder.
4. Launch the new binary.

Your runtime workspace data remains in the OS-specific workspace path unless you
manually remove it.

## 7) Uninstall

1. Delete the extracted application folder.
2. Optional cleanup of runtime data:
- Windows: delete `%APPDATA%\CPD-SimStudio\`
- Linux: delete `~/.local/share/CPD-SimStudio/`
