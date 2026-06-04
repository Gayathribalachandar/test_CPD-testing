import os
import time
from pathlib import Path

import numpy as np

from cpd_solver import run_simulation
from io_adapters.schema_v1 import load_workspace_inputs_v1


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_DIR = Path(os.environ.get("CPD_WORKSPACE_DIR", str(PROJECT_ROOT / "workspace")))
WORKSPACE_INPUT_DIR = WORKSPACE_DIR / "input"
WORKSPACE_OUTPUT_DIR = WORKSPACE_DIR / "output"
POS_HISTORY_OUTPUT = WORKSPACE_OUTPUT_DIR / "pos_history.npy"
DISPLACEMENT_HISTORY_OUTPUT = WORKSPACE_OUTPUT_DIR / "displacement_history.npy"
STRAIN_HISTORY_OUTPUT = WORKSPACE_OUTPUT_DIR / "strain_history.npy"
STRESS_HISTORY_OUTPUT = WORKSPACE_OUTPUT_DIR / "stress_history.npy"
LEGACY_POS_HISTORY_OUTPUT = WORKSPACE_DIR / "pos_history.npy"


def _unlink_stale_output(stale_path, attempts=20, delay_seconds=0.25):
    path = Path(stale_path)
    for attempt in range(max(1, int(attempts))):
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False
        except PermissionError:
            if attempt + 1 >= max(1, int(attempts)):
                raise
            time.sleep(max(0.0, float(delay_seconds)))
    return False


def _read_schema_version(workspace_dir: Path) -> str:
    version_path = workspace_dir / "schema_version.txt"
    if not version_path.exists():
        return "v1"
    try:
        version = version_path.read_text(encoding="utf-8").strip().lower()
    except Exception:
        return "v1"
    return version or "v1"


def main():
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    WORKSPACE_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    WORKSPACE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Solver input directory: {WORKSPACE_INPUT_DIR}")

    for stale_path in (
        POS_HISTORY_OUTPUT,
        DISPLACEMENT_HISTORY_OUTPUT,
        STRAIN_HISTORY_OUTPUT,
        STRESS_HISTORY_OUTPUT,
        LEGACY_POS_HISTORY_OUTPUT,
    ):
        if stale_path.exists():
            _unlink_stale_output(stale_path)
            print(f"Deleted {stale_path}")

    schema_version = _read_schema_version(WORKSPACE_DIR)
    if schema_version != "v1":
        print(f"Warning: unsupported schema version '{schema_version}', falling back to v1 loader.")

    inputs = load_workspace_inputs_v1(WORKSPACE_DIR)
    outputs = run_simulation(**inputs, return_fields=True)
    np.save(str(POS_HISTORY_OUTPUT), outputs["pos_history"])
    np.save(str(DISPLACEMENT_HISTORY_OUTPUT), outputs["displacement_history"])
    np.save(str(STRAIN_HISTORY_OUTPUT), outputs["strain_history"])
    np.save(str(STRESS_HISTORY_OUTPUT), outputs["stress_history"])


if __name__ == "__main__":
    main()
