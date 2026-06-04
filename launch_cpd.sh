#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$DIR/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "CPD SimStudio: .venv not found."
  echo "Please run setup once:"
  echo "  python3.11 -m venv .venv"
  echo "  source .venv/bin/activate"
  echo "  pip install -r requirements.txt"
  echo "Optional: pip install OCP gmsh"
  exit 1
fi

cd "$DIR"

if [[ "$(uname -s)" == "Linux" && -z "${QT_QPA_PLATFORM:-}" ]]; then
  if [[ -n "${DISPLAY:-}" ]]; then
    export QT_QPA_PLATFORM="xcb"
  elif [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
    export QT_QPA_PLATFORM="wayland"
  fi
fi

exec "$PY" "$DIR/main_window.py"
