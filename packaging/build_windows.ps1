param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$project = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $project ".venv-windows-build"
$python = Join-Path $venv "Scripts\python.exe"
$dist = Join-Path $project "dist"

Push-Location $project
try {
    if (-not (Test-Path $python)) {
        py -3.11 -m venv $venv
        if ($LASTEXITCODE -ne 0) { throw "Python 3.11 virtual environment creation failed" }
    }

    & $python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
    & $python -m pip install -e . pyinstaller pytest
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }

    if (-not $SkipTests) {
        & $python -m pytest -q
        if ($LASTEXITCODE -ne 0) { throw "Tests failed; refusing to package" }
        & $python -m laparoskan --smoke-test
        if ($LASTEXITCODE -ne 0) { throw "Application smoke test failed; refusing to package" }
    }

    & $python -m PyInstaller --noconfirm --clean --onedir --windowed `
        --name Laparoskan `
        --collect-all SimpleITK `
        --collect-all vtkmodules `
        --hidden-import vtkmodules.qt.QVTKRenderWindowInteractor `
        packaging/launcher.py
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed" }

    $exe = Join-Path $dist "Laparoskan\Laparoskan.exe"
    if (-not (Test-Path $exe)) { throw "Built executable missing: $exe" }
    Write-Host "Portable application: $exe"

    $bundle = Join-Path $dist "Laparoskan"
    & $python -m pip freeze | Out-File -FilePath (Join-Path $bundle "DEPENDENCIES.txt") -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw "Dependency inventory failed" }
    & $python packaging/collect_licenses.py $bundle
    if ($LASTEXITCODE -ne 0) { throw "License inventory failed" }
    foreach ($name in @("README.md", "PROJECT_STATUS.md", "THIRD_PARTY.md", "BUILD_WINDOWS.md")) {
        $source = Join-Path $project $name
        if (Test-Path $source) { Copy-Item $source -Destination $bundle }
    }

    $zip = Join-Path $dist "Laparoskan-Windows-portable.zip"
    if (Test-Path $zip) { Remove-Item $zip -Force }
    Compress-Archive -Path (Join-Path $dist "Laparoskan") -DestinationPath $zip
    Write-Host "Portable archive: $zip"
}
finally {
    Pop-Location
}
