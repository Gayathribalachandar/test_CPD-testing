Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

Write-Host "[1/8] Cleaning previous Windows build outputs..."
$pathsToRemove = @(
    ".venv-build-win",
    "build\pyinstaller-win",
    "dist\CPD-SimStudio-win",
    "releases\CPD-SimStudio-win.zip"
)
foreach ($path in $pathsToRemove) {
    if (Test-Path $path) {
        Remove-Item -Recurse -Force $path
    }
}

Write-Host "[2/8] Creating clean build virtualenv..."
if (Get-Command py -ErrorAction SilentlyContinue) {
    py -3.11 -m venv .venv-build-win
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    python -m venv .venv-build-win
} else {
    throw "Python not found. Install Python 3.11 and retry."
}

$VenvPython = Join-Path $RootDir ".venv-build-win\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Virtualenv python not found at $VenvPython"
}

Write-Host "[3/8] Upgrading pip..."
& $VenvPython -m pip install --upgrade pip

Write-Host "[4/8] Installing project requirements..."
& $VenvPython -m pip install -r requirements.txt

Write-Host "[5/8] Installing PyInstaller..."
& $VenvPython -m pip install pyinstaller

Write-Host "[6/8] Building CPD SimStudio (Windows one-dir)..."
& $VenvPython -m PyInstaller `
    --clean `
    --noconfirm `
    scripts\cpd_simstudio_windows.spec `
    --distpath dist\CPD-SimStudio-win `
    --workpath build\pyinstaller-win

Write-Host "[7/8] Creating release zip..."
New-Item -ItemType Directory -Force -Path releases | Out-Null
$DistFolder = Join-Path $RootDir "dist\CPD-SimStudio-win\CPD-SimStudio"
$ZipPath = Join-Path $RootDir "releases\CPD-SimStudio-win.zip"
if (-not (Test-Path $DistFolder)) {
    throw "Expected dist folder missing: $DistFolder"
}
if (Test-Path $ZipPath) {
    Remove-Item -Force $ZipPath
}
Compress-Archive -Path $DistFolder -DestinationPath $ZipPath

Write-Host "[8/8] Build complete."
Write-Host "Folder artifact: $DistFolder"
Write-Host "Zip artifact: $ZipPath"
