# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


if "__file__" in globals():
    ROOT = Path(__file__).resolve().parents[1]
else:
    ROOT = Path.cwd()
ENTRYPOINT = ROOT / "main_window.py"

EXCLUDED_DIRS = {
    ".git",
    ".github",
    ".venv",
    ".venv-build",
    ".venv-build-win",
    "build",
    "dist",
    "releases",
    "__pycache__",
    "workspace",
    "autosave",
    "saved_results",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".npy", ".log"}
ROOT_ASSET_SUFFIXES = {".ico", ".png", ".svg", ".desktop"}


def _should_skip(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in EXCLUDED_DIRS for part in rel.parts):
        return True
    if rel.name.startswith("recovered_"):
        return True
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    return False


def _scan_runtime_assets():
    datas = []

    # CPD-main is executed as an external subprocess, so keep its files on disk.
    cpd_root = ROOT / "CPD-main"
    if cpd_root.exists():
        for file_path in sorted(cpd_root.rglob("*")):
            if not file_path.is_file() or _should_skip(file_path):
                continue
            rel = file_path.relative_to(ROOT)
            datas.append((str(file_path), str(rel.parent)))

    # Scan UI package for non-Python data assets, if present.
    ui_root = ROOT / "ui"
    if ui_root.exists():
        for file_path in sorted(ui_root.rglob("*")):
            if not file_path.is_file() or _should_skip(file_path):
                continue
            if file_path.suffix.lower() == ".py":
                continue
            rel = file_path.relative_to(ROOT)
            datas.append((str(file_path), str(rel.parent)))

    # Scan root-level runtime assets (icons/desktop launcher metadata).
    for file_path in sorted(ROOT.rglob("*")):
        if not file_path.is_file() or _should_skip(file_path):
            continue
        rel = file_path.relative_to(ROOT)
        if not rel.parts:
            continue
        if rel.parts[0] in {"CPD-main", "ui"}:
            continue
        if file_path.suffix.lower() not in ROOT_ASSET_SUFFIXES:
            continue
        datas.append((str(file_path), str(rel.parent)))

    # De-duplicate while preserving deterministic ordering.
    unique = {}
    for src, dest in datas:
        unique[(src, dest)] = None
    return list(unique.keys())


def _first_icon():
    for icon_path in sorted(ROOT.rglob("*.ico")):
        if icon_path.is_file() and not _should_skip(icon_path):
            return str(icon_path)
    return None


datas = _scan_runtime_assets()
hiddenimports = collect_submodules("PySide6")
icon_path = _first_icon()

a = Analysis(
    [str(ENTRYPOINT)],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe_kwargs = dict(
    exclude_binaries=True,
    name="CPD-SimStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)
if icon_path:
    exe_kwargs["icon"] = icon_path

exe = EXE(
    pyz,
    a.scripts,
    [],
    **exe_kwargs,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name="CPD-SimStudio",
)
