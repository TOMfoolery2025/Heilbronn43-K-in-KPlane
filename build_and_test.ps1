# Quick Build and Test Script for Phase 1
# Run: .\build_and_test.ps1

$ErrorActionPreference = "Stop"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Phase 1: CUDA Pipeline Build & Test" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Activate virtual environment if it exists
$venvPath = "D:/D_backup/2025/tum/25W/hackthon/Hackathon-Nov-25-Heilbronn43/heilbron-43/Scripts/Activate.ps1"
if (Test-Path $venvPath) {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & $venvPath
    Write-Host "  ✓ Virtual environment activated" -ForegroundColor Green
}

# Check prerequisites
Write-Host "`n🔍 Checking prerequisites..." -ForegroundColor Yellow

# Check CUDA
try {
    $nvccVersion = nvcc --version 2>&1 | Select-String "release"
    Write-Host "  ✓ CUDA: $nvccVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ CUDA not found! Install CUDA Toolkit 12.6" -ForegroundColor Red
    exit 1
}

# Check CMake
try {
    $cmakeVersion = cmake --version 2>&1 | Select-Object -First 1
    Write-Host "  ✓ CMake: $cmakeVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ CMake not found! Install from cmake.org" -ForegroundColor Red
    exit 1
}

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  ✓ Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Python not found!" -ForegroundColor Red
    exit 1
}

# Install Python dependencies
Write-Host "`n📦 Installing Python dependencies..." -ForegroundColor Yellow
pip install -q pybind11 pytest numpy
Write-Host "  ✓ Dependencies installed" -ForegroundColor Green

# Check for pybind11
Write-Host "`n📥 Checking pybind11..." -ForegroundColor Yellow
if (-not (Test-Path "pybind11")) {
    Write-Host "  Downloading pybind11..." -ForegroundColor Yellow
    git clone --quiet --depth 1 https://github.com/pybind/pybind11.git
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ✗ Failed to clone pybind11. Download manually from:" -ForegroundColor Red
        Write-Host "    https://github.com/pybind/pybind11/archive/refs/heads/master.zip" -ForegroundColor Yellow
        exit 1
    }
}
Write-Host "  ✓ pybind11 ready" -ForegroundColor Green

# Build
Write-Host "`n🔧 Building CUDA module..." -ForegroundColor Yellow

# Create build directory
if (-not (Test-Path "build")) {
    mkdir build | Out-Null
}

cd build

# Detect Visual Studio version
$vsVersions = @(
    @{Name="Visual Studio 17 2022"; Year=2022},
    @{Name="Visual Studio 16 2019"; Year=2019}
)

$generator = $null
foreach ($vs in $vsVersions) {
    $vsPath = "C:\Program Files\Microsoft Visual Studio\$($vs.Year)"
    if (Test-Path $vsPath) {
        $generator = $vs.Name
        Write-Host "  Found: $generator" -ForegroundColor Green
        break
    }
}

if (-not $generator) {
    Write-Host "  ✗ Visual Studio not found! Install VS 2019 or 2022 with C++ tools" -ForegroundColor Red
    cd ..
    exit 1
}

# Get Python executable path
$pythonExe = (Get-Command python).Path

# Configure
Write-Host "  Configuring with CMake..." -ForegroundColor Yellow
cmake .. -G $generator -A x64 -DPYTHON_EXECUTABLE="$pythonExe"

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ✗ CMake configuration failed!" -ForegroundColor Red
    cd ..
    exit 1
}

# Build
Write-Host "  Building..." -ForegroundColor Yellow
cmake --build . --config Release

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ✗ Build failed!" -ForegroundColor Red
    cd ..
    exit 1
}

cd ..

# Copy module
Write-Host "`n📋 Installing module..." -ForegroundColor Yellow
if (Test-Path "build\Release\planar_cuda.pyd") {
    Copy-Item "build\Release\planar_cuda.pyd" "." -Force
    Write-Host "  ✓ Module copied to project root" -ForegroundColor Green
} else {
    Write-Host "  ✗ planar_cuda.pyd not found in build\Release\" -ForegroundColor Red
    exit 1
}

# Test import
Write-Host "`n🧪 Testing module import..." -ForegroundColor Yellow
$importTest = python -c "import planar_cuda; print(f'Version: {planar_cuda.__version__}')" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ $importTest" -ForegroundColor Green
} else {
    Write-Host "  ✗ Import failed: $importTest" -ForegroundColor Red
    exit 1
}

# Run tests
Write-Host "`n🧪 Running test suite..." -ForegroundColor Yellow
pytest tests/cuda_tests/test_phase1_pipeline.py -v --tb=short

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host "  ✅ PHASE 1 COMPLETE!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "`nNext steps:" -ForegroundColor Cyan
    Write-Host "  • Review CUDA_HYBRID_ROADMAP.md" -ForegroundColor White
    Write-Host "  • Start Phase 2: Geometry kernels" -ForegroundColor White
    Write-Host "  • Read BUILD_INSTRUCTIONS.md for details`n" -ForegroundColor White
} else {
    Write-Host "`n========================================" -ForegroundColor Red
    Write-Host "  ✗ Tests failed" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "`nCheck BUILD_INSTRUCTIONS.md for troubleshooting`n" -ForegroundColor Yellow
    exit 1
}
