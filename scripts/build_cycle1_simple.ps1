# Cycle 1 Build Script - Simplified Approach
# Based on working build_final.ps1

$ErrorActionPreference = "Stop"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Cycle 1: Geometry Verification Build" -ForegroundColor Cyan  
Write-Host "  Test-Driven CUDA Development" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# ============================================================================
# Create Build Batch Script
# ============================================================================

$buildScript = @'
@echo off
echo [BUILD] Initializing MSVC environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

echo [BUILD] Compiling planar_cuda module...
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" --shared src/cuda_utils/planar_cuda.cu -o build_artifacts/planar_cuda.pyd -arch=sm_89 --compiler-options "/EHsc /MD" -Isrc -IC:\Users\aloha\AppData\Local\Programs\Python\Python311\Include -ID:\D_backup\2025\tum\25W\hackthon\Hackathon-Nov-25-Heilbronn43\heilbron-43\Lib\site-packages\pybind11\include -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64" -LC:\Users\aloha\AppData\Local\Programs\Python\Python311\libs -lcudart -lpython311 -Xlinker "/NODEFAULTLIB:MSVCRT" -Xlinker "legacy_stdio_definitions.lib" -Xlinker "ucrt.lib" -Xlinker "vcruntime.lib"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Compilation failed!
    exit /b 1
)

echo [BUILD] Build successful!
'@

# Create build directory
if (-not (Test-Path "build_artifacts")) {
    mkdir build_artifacts | Out-Null
}

# Write and execute batch script
$buildScript | Out-File -FilePath "build_cycle1_temp.bat" -Encoding ASCII
cmd /c build_cycle1_temp.bat

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Build failed!" -ForegroundColor Red
    Remove-Item "build_cycle1_temp.bat" -ErrorAction SilentlyContinue
    exit 1
}

Remove-Item "build_cycle1_temp.bat" -ErrorAction SilentlyContinue

# ============================================================================
# Test Module
# ============================================================================

Write-Host "`n[TEST] Testing module import..." -ForegroundColor Yellow

$testScript = @'
import os
import sys

# Add CUDA DLL directory
cuda_path = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin'
if os.path.exists(cuda_path):
    os.add_dll_directory(cuda_path)

# Add build_artifacts to path
sys.path.insert(0, 'build_artifacts')

# Import and test
import planar_cuda

print(f'Version: {planar_cuda.__version__}')
print(f'CUDA enabled: {planar_cuda.cuda_enabled}')

# Triangle test (should have 0 crossings)
solver = planar_cuda.PlanarSolver([0, 10, 5], [0, 0, 10], [(0, 1), (1, 2), (2, 0)])
crossings = solver.calculate_total_crossings()
print(f'Triangle test: {crossings} crossings')

if crossings == 0:
    print('[PASS] Module test successful!')
else:
    print('[FAIL] Expected 0 crossings!')
    sys.exit(1)
'@

& heilbron-43\Scripts\python.exe -c $testScript 2>&1 | ForEach-Object {
    if ($_ -match "\[PASS\]" -or $_ -match "Version:" -or $_ -match "CUDA enabled") {
        Write-Host $_ -ForegroundColor Green
    } elseif ($_ -match "\[FAIL\]") {
        Write-Host $_ -ForegroundColor Red
    } else {
        Write-Host $_ -ForegroundColor White
    }
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[ERROR] Module test failed!" -ForegroundColor Red
    exit 1
}

# ============================================================================
# Success!
# ============================================================================

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  BUILD SUCCESSFUL!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Green

Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Run full test suite: " -ForegroundColor White -NoNewline
Write-Host "pytest tests/cuda_tests/test_gpu_geometry.py -v -s" -ForegroundColor Yellow
Write-Host "  2. Read documentation: " -ForegroundColor White -NoNewline
Write-Host "docs/CYCLE1_GUIDE.md" -ForegroundColor Yellow
Write-Host ""
