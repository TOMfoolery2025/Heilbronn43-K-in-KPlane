# Final Working Build - Uses MSVC Environment
# This initializes the Visual Studio environment before building

$ErrorActionPreference = "Stop"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Cycle 1: Geometry Verification Build" -ForegroundColor Cyan  
Write-Host "========================================`n" -ForegroundColor Cyan

# Initialize MSVC environment
$vcvarsPath = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvarsPath)) {
    Write-Host "Error: Visual Studio 2022 Community not found!" -ForegroundColor Red
    Write-Host "Please install Visual Studio 2022 Community with C++ workload" -ForegroundColor Yellow
    exit 1
}

# Create build script that runs in MSVC environment
$buildScript = @'
@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

echo Compiling Cycle 1 CUDA module...
set PYBIND11_INCLUDE=D:\D_backup\2025\tum\25W\hackthon\Hackathon-Nov-25-Heilbronn43\heilbron-43\Lib\site-packages\pybind11\include
set PYTHON_BASE=C:\Users\aloha\AppData\Local\Programs\Python\Python311
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" --shared src/cuda_utils/planar_cuda.cu -o build_artifacts/planar_cuda.pyd -arch=sm_89 --compiler-options "/EHsc /MD" -Isrc -I"%PYTHON_BASE%\Include" -I"%PYBIND11_INCLUDE%" -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64" -L"%PYTHON_BASE%\libs" -lcudart -lpython311 -Xlinker "/NODEFAULTLIB:MSVCRT" -Xlinker "legacy_stdio_definitions.lib" -Xlinker "ucrt.lib" -Xlinker "vcruntime.lib" -Xlinker "msvcrt.lib"
if %ERRORLEVEL% NEQ 0 exit /b 1

echo Build successful!
'@

# Write batch script
$buildScript | Out-File -FilePath "build_cycle1.bat" -Encoding ASCII

# Create build directory
if (-not (Test-Path "build_artifacts")) {
    mkdir build_artifacts | Out-Null
}

# Run build
Write-Host "Running build with MSVC environment..." -ForegroundColor Yellow
cmd /c build_cycle1.bat

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nBuild failed!" -ForegroundColor Red
    exit 1
}

# Test
Write-Host "`nTesting module..." -ForegroundColor Yellow
$testScript = @'
import os
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
import sys
sys.path.insert(0, 'build_artifacts')
import planar_cuda
print(f'Version: {planar_cuda.__version__}')
print(f'CUDA enabled: {planar_cuda.cuda_enabled}')
solver = planar_cuda.PlanarSolver([0,10,5], [0,0,10], [(0,1),(1,2),(2,0)])
result = solver.calculate_total_crossings()
print(f'Triangle test: {result} crossings (expected: 0)')
assert result == 0, 'Test failed!'
print('Module test PASSED!')
'@

$testResult = python -c $testScript 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host $testResult -ForegroundColor Green
    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host "  BUILD SUCCESSFUL!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "`nNext: pytest tests/cuda_tests/test_gpu_geometry.py -v" -ForegroundColor Cyan
} else {
    Write-Host $testResult -ForegroundColor Red
    Write-Host "`nImport test failed!" -ForegroundColor Red
    exit 1
}
