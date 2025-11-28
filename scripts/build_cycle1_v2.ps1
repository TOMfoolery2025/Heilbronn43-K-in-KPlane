# Cycle 1 Build - Working Version
# Compiles planar_cuda.cu into a Python module

$ErrorActionPreference = "Stop"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Cycle 1: Geometry Verification" -ForegroundColor Cyan  
Write-Host "========================================`n" -ForegroundColor Cyan

# Create build directory
if (-not (Test-Path "build_artifacts")) {
    mkdir build_artifacts | Out-Null
}

# Create batch file with shortened paths using environment variables
$buildScript = @'
@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul

set CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin
set CUDA_LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64
set PY_INC=C:\Users\aloha\AppData\Local\Programs\Python\Python311\Include
set PY_LIB=C:\Users\aloha\AppData\Local\Programs\Python\Python311\libs

for /f "tokens=*" %%i in ('heilbron-43\Scripts\python.exe -c "import pybind11; print(pybind11.get_include())"') do set PB_INC=%%i

echo Compiling Cycle 1 module...
"%CUDA_BIN%\nvcc.exe" --shared ^
  src/cuda_utils/planar_cuda.cu ^
  -o build_artifacts/planar_cuda.pyd ^
  -arch=sm_89 ^
  --compiler-options "/EHsc /MD" ^
  -Isrc ^
  -I"%PY_INC%" ^
  -I"%PB_INC%" ^
  -L"%CUDA_LIB%" ^
  -L"%PY_LIB%" ^
  -lcudart ^
  -lpython311 ^
  -Xlinker "/NODEFAULTLIB:MSVCRT" ^
  -Xlinker "legacy_stdio_definitions.lib" ^
  -Xlinker "ucrt.lib" ^
  -Xlinker "vcruntime.lib"

if %ERRORLEVEL% NEQ 0 exit /b 1
echo Build successful!
'@

$buildScript | Out-File -FilePath "build_temp.bat" -Encoding ASCII
cmd /c build_temp.bat

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nBuild failed!" -ForegroundColor Red
    Remove-Item "build_temp.bat" -ErrorAction SilentlyContinue
    exit 1
}

Remove-Item "build_temp.bat" -ErrorAction SilentlyContinue

# Test import
Write-Host "`nTesting module..." -ForegroundColor Yellow

$test = @'
import os, sys
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
sys.path.insert(0, 'build_artifacts')
import planar_cuda
print(f'Version: {planar_cuda.__version__}')
solver = planar_cuda.PlanarSolver([0,10,5], [0,0,10], [(0,1),(1,2),(2,0)])
c = solver.calculate_total_crossings()
print(f'Triangle: {c} crossings')
assert c == 0, 'Test failed!'
print('SUCCESS!')
'@

& heilbron-43\Scripts\python.exe -c $test

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host "  BUILD SUCCESSFUL!" -ForegroundColor Green
    Write-Host "========================================`n" -ForegroundColor Green
    Write-Host "Run tests: pytest tests/cuda_tests/test_gpu_geometry.py -v" -ForegroundColor Cyan
} else {
    Write-Host "`nTest failed!" -ForegroundColor Red
    exit 1
}
