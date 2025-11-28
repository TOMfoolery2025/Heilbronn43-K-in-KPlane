# Manual Build Script (Run with venv already activated)
# Usage: Make sure (heilbron-43) is in your prompt, then run: .\manual_build.ps1

$ErrorActionPreference = "Stop"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Manual CUDA Build" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Verify Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found! Activate virtual environment first:" -ForegroundColor Red
    Write-Host "  D:/D_backup/2025/tum/25W/hackthon/Hackathon-Nov-25-Heilbronn43/heilbron-43/Scripts/Activate.ps1" -ForegroundColor Yellow
    exit 1
}

# Check pybind11
if (-not (Test-Path "pybind11")) {
    Write-Host "Downloading pybind11..." -ForegroundColor Yellow
    git clone --quiet --depth 1 https://github.com/pybind/pybind11.git
}
Write-Host "✓ pybind11 ready" -ForegroundColor Green

# Install Python packages
Write-Host "`nInstalling Python packages..." -ForegroundColor Yellow
pip install -q pybind11 pytest numpy
Write-Host "✓ Packages installed" -ForegroundColor Green

# Clean old build
Write-Host "`nCleaning old build..." -ForegroundColor Yellow
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "planar_cuda.pyd") { Remove-Item -Force "planar_cuda.pyd" }
Write-Host "✓ Cleaned" -ForegroundColor Green

# Create build directory
Write-Host "`nConfiguring with CMake..." -ForegroundColor Yellow
mkdir build | Out-Null
cd build

# Get Python executable
$pythonExe = (Get-Command python).Path
Write-Host "  Python: $pythonExe" -ForegroundColor Gray

# Configure
cmake .. -G "Visual Studio 17 2022" -A x64 -DPYTHON_EXECUTABLE="$pythonExe"

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n✗ CMake configuration failed!" -ForegroundColor Red
    cd ..
    exit 1
}

Write-Host "✓ Configuration successful" -ForegroundColor Green

# Build
Write-Host "`nBuilding..." -ForegroundColor Yellow
cmake --build . --config Release

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n✗ Build failed!" -ForegroundColor Red
    cd ..
    exit 1
}

Write-Host "✓ Build successful" -ForegroundColor Green

# Copy module
cd ..
Write-Host "`nInstalling module..." -ForegroundColor Yellow
if (Test-Path "build\Release\planar_cuda.pyd") {
    Copy-Item "build\Release\planar_cuda.pyd" "." -Force
    Write-Host "✓ Module installed: planar_cuda.pyd" -ForegroundColor Green
} else {
    Write-Host "✗ Module not found!" -ForegroundColor Red
    exit 1
}

# Test import
Write-Host "`nTesting import..." -ForegroundColor Yellow
$importTest = python -c "import planar_cuda; print(f'Version: {planar_cuda.__version__}')" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ $importTest" -ForegroundColor Green
} else {
    Write-Host "✗ Import failed!" -ForegroundColor Red
    Write-Host "  Add CUDA DLL path and try again:" -ForegroundColor Yellow
    Write-Host "  python -c `"import os; os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin'); import planar_cuda`"" -ForegroundColor Gray
    exit 1
}

# Run tests
Write-Host "`nRunning tests..." -ForegroundColor Yellow
pytest tests/cuda_tests/test_phase1_pipeline.py -v --tb=short

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host "  ✅ BUILD AND TESTS SUCCESSFUL!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
} else {
    Write-Host "`n========================================" -ForegroundColor Red
    Write-Host "  ✗ Tests failed" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    exit 1
}
