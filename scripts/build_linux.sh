#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/6] Cleaning previous build outputs..."
rm -rf .venv-build build/pyinstaller dist/CPD-SimStudio-linux

PYTHON_BIN="python3.11"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    PYTHON_BIN="python"
  fi
fi

echo "[2/6] Creating clean build virtualenv..."
"$PYTHON_BIN" -m venv .venv-build
source .venv-build/bin/activate

echo "[3/6] Installing project requirements..."
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "[4/6] Installing PyInstaller..."
pip install pyinstaller

echo "[5/6] Building CPD SimStudio (Linux onedir)..."
pyinstaller \
  --clean \
  --noconfirm \
  scripts/cpd_simstudio_linux.spec \
  --distpath dist/CPD-SimStudio-linux \
  --workpath build/pyinstaller

echo "[6/6] Build complete."
echo "Artifact directory: $ROOT_DIR/dist/CPD-SimStudio-linux/CPD-SimStudio"
