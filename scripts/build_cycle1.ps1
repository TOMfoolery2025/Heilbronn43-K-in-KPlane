# Cycle 1: Final Working Build Script
# Successfully compiles planar_cuda.cu into a Python module
#
# This script:
# 1. Creates a short junction to Visual Studio (C:\VS)
# 2. Initializes MSVC environment
# 3. Compiles the Cycle 1 CUDA module
# 4. Runs a quick test
#
# Usage: .\scripts\build_cycle1.ps1

$ErrorActionPreference = "Stop"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Cycle 1: Geometry Verification Build" -ForegroundColor Cyan  
Write-Host "========================================`n" -ForegroundColor Cyan

# Step 1: Create short junction to VS (avoids path length issues)
if (-not (Test-Path "C:\VS")) {
    Write-Host "Creating junction C:\VS -> Visual Studio..." -ForegroundColor Yellow
    New-Item -ItemType Junction -Path "C:\VS" -Target "C:\Program Files\Microsoft Visual Studio\2022\Community" -Force | Out-Null
}

# Step 2: Initialize MSVC environment
Write-Host "Initializing MSVC environment..." -ForegroundColor Yellow
$vsPath = "C:\VS\VC\Auxiliary\Build\vcvars64.bat"
$tempFile = [System.IO.Path]::GetTempFileName() + ".cmd"
"@echo off`ncall `"$vsPath`"`nset" | Out-File $tempFile -Encoding ASCII
$envVars = cmd /c $tempFile 2>&1
Remove-Item $tempFile

foreach($line in $envVars) {
    if($line -match '^([^=]+)=(.*)$') {
        $name=$matches[1]
        $value=$matches[2]
        if($name -ne 'PROMPT'){
            [Environment]::SetEnvironmentVariable($name,$value,'Process')
        }
    }
}

# Verify cl.exe is available
$clPath = Get-Command cl.exe -ErrorAction SilentlyContinue
if ($clPath) {
    Write-Host "[OK] MSVC initialized (cl.exe found at $($clPath.Source))" -ForegroundColor Green
} else {
    Write-Host "[ERROR] MSVC initialization failed - cl.exe not found" -ForegroundColor Red
    Write-Host "The environment variables may not have been set correctly" -ForegroundColor Yellow
    Write-Host "Trying to continue anyway..." -ForegroundColor Yellow
}

# Step 3: Compile
Write-Host "Compiling planar_cuda.cu..." -ForegroundColor Yellow

if (-not (Test-Path "build_artifacts")) {
    mkdir build_artifacts | Out-Null
}

$pb = (& heilbron-43\Scripts\python.exe -c "import pybind11; print(pybind11.get_include())")
$py = "C:\Users\aloha\AppData\Local\Programs\Python\Python311"

& nvcc --shared src/cuda_utils/planar_cuda.cu `
    -o build_artifacts/planar_cuda.pyd `
    -arch=sm_89 `
    --compiler-options "/EHsc /MD" `
    -Isrc `
    -I"$py\Include" `
    -I"$pb" `
    -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64" `
    -L"$py\libs" `
    -lcudart `
    -lpython311 `
    -Xlinker /NODEFAULTLIB:MSVCRT `
    -Xlinker legacy_stdio_definitions.lib `
    -Xlinker ucrt.lib `
    -Xlinker vcruntime.lib `
    -Xlinker msvcrt.lib

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Compilation failed" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "build_artifacts/planar_cuda.pyd")) {
    Write-Host "[ERROR] Module not created" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Compilation successful" -ForegroundColor Green

# Step 4: Quick test
Write-Host "`nTesting module..." -ForegroundColor Yellow

$test = @'
import os, sys
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
sys.path.insert(0, 'build_artifacts')
import planar_cuda
print(f'Version: {planar_cuda.__version__}')
s = planar_cuda.PlanarSolver([0,10,5], [0,0,10], [(0,1),(1,2),(2,0)])
c = s.calculate_total_crossings()
print(f'Triangle: {c} crossings')
assert c == 0, 'Test failed!'
print('[PASS] Quick test successful')
'@

& heilbron-43\Scripts\python.exe -c $test

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Test failed" -ForegroundColor Red
    exit 1
}

# Success!
Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  BUILD SUCCESSFUL!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Green

Write-Host "Cycle 1 complete! Next steps:" -ForegroundColor Cyan
Write-Host "  1. Run full tests: " -NoNewline -ForegroundColor White
Write-Host "heilbron-43\Scripts\python.exe -m pytest tests/cuda_tests/test_gpu_geometry.py -v" -ForegroundColor Yellow
Write-Host "  2. Read guide: " -NoNewline -ForegroundColor White
Write-Host "docs/CYCLE1_GUIDE.md" -ForegroundColor Yellow
Write-Host "  3. Next phase: " -NoNewline -ForegroundColor White
Write-Host "Cycle 2 - State Management" -ForegroundColor Yellow
Write-Host ""
