# Final Working Build - Uses MSVC Environment
# This initializes the Visual Studio environment before building

$ErrorActionPreference = "Stop"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  CUDA Build with MSVC Environment" -ForegroundColor Cyan  
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

echo Compiling CUDA kernel...
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" -c src/cuda_utils/vector_ops.cu -o build_direct/vector_ops.obj -arch=sm_89 --compiler-options "/EHsc /MD" -Isrc
if %ERRORLEVEL% NEQ 0 exit /b 1

echo Compiling C++ binding...
for /f "tokens=*" %%i in ('python -c "import pybind11; print(pybind11.get_include())"') do set PYBIND11_INCLUDE=%%i
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" -c src/cpp_binding/binding.cpp -o build_direct/binding.obj --compiler-options "/EHsc /MD" -Isrc -I"C:\Users\aloha\AppData\Local\Programs\Python\Python311\Include" -I"%PYBIND11_INCLUDE%"
if %ERRORLEVEL% NEQ 0 exit /b 1

echo Linking module...
for /f "tokens=*" %%i in ('python -c "import sys; print(sys.base_prefix)"') do set PYTHON_BASE=%%i
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" --shared build_direct/vector_ops.obj build_direct/binding.obj -o planar_cuda.pyd -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64" -L"%PYTHON_BASE%\libs" -lcudart -lpython311 -Xlinker "/NODEFAULTLIB:MSVCRT" -Xlinker "legacy_stdio_definitions.lib" -Xlinker "ucrt.lib" -Xlinker "vcruntime.lib" -Xlinker "msvcrt.lib"
if %ERRORLEVEL% NEQ 0 exit /b 1

echo Build successful!
'@

# Write batch script
$buildScript | Out-File -FilePath "build_msvc.bat" -Encoding ASCII

# Create build directory
if (Test-Path "build_direct") { Remove-Item -Recurse -Force "build_direct" }
mkdir build_direct | Out-Null

# Run build
Write-Host "Running build with MSVC environment..." -ForegroundColor Yellow
cmd /c build_msvc.bat

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nBuild failed!" -ForegroundColor Red
    exit 1
}

# Test
Write-Host "`nTesting module..." -ForegroundColor Yellow
$testScript = @'
import os
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
import planar_cuda
print(f'Version: {planar_cuda.__version__}')
print(f'CUDA enabled: {planar_cuda.cuda_enabled}')
result = planar_cuda.add_vectors([1,2,3], [4,5,6])
print(f'Test: [1,2,3] + [4,5,6] = {result}')
assert result == [5,7,9], 'Test failed!'
print('SUCCESS!')
'@

$testResult = python -c $testScript 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host $testResult -ForegroundColor Green
    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host "  BUILD SUCCESSFUL!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "`nNext: pytest tests/cuda_tests/test_phase1_pipeline.py -v" -ForegroundColor Cyan
} else {
    Write-Host $testResult -ForegroundColor Red
    Write-Host "`nImport test failed!" -ForegroundColor Red
    exit 1
}
