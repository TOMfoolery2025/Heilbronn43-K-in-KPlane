# Simplified Build Using Direct Compilation
# This bypasses CMake entirely

$ErrorActionPreference = "Stop"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Direct CUDA Compilation (No CMake)" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Paths
$nvcc = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe"
$cudaLib = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64"
$cudaInclude = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include"
$pythonExe = "D:\D_backup\2025\tum\25W\hackthon\Hackathon-Nov-25-Heilbronn43\heilbron-43\Scripts\python.exe"

# Get Python include and lib directories
Write-Host "Getting Python paths..." -ForegroundColor Yellow
$pythonInclude = & $pythonExe -c "import sysconfig; print(sysconfig.get_path('include'))"
$pythonPrefix = & $pythonExe -c "import sys; print(sys.prefix)"
$pythonLib = "$pythonPrefix\libs"
$pythonVersion = & $pythonExe -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')"

# Get pybind11 include
$pybind11Include = & $pythonExe -c "import pybind11; print(pybind11.get_include())"

Write-Host "  Python include: $pythonInclude" -ForegroundColor Gray
Write-Host "  Python lib: $pythonLib" -ForegroundColor Gray
Write-Host "  Python version: $pythonVersion" -ForegroundColor Gray
Write-Host "  pybind11: $pybind11Include" -ForegroundColor Gray

# Create build directory
if (Test-Path "build_direct") { Remove-Item -Recurse -Force "build_direct" }
mkdir build_direct | Out-Null

# Step 1: Compile CUDA code to object file
Write-Host "`nStep 1: Compiling CUDA kernel..." -ForegroundColor Yellow
& $nvcc `
    -c src/cuda_utils/vector_ops.cu `
    -o build_direct/vector_ops.obj `
    -arch=sm_89 `
    --compiler-options "/EHsc /MD" `
    -Xcompiler "/wd4819" `
    -Isrc `
    -I"$cudaInclude"

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ CUDA compilation failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ CUDA kernel compiled" -ForegroundColor Green

# Step 2: Compile C++ binding
Write-Host "`nStep 2: Compiling C++ binding..." -ForegroundColor Yellow
& $nvcc `
    -c src/cpp_binding/binding.cpp `
    -o build_direct/binding.obj `
    --compiler-options "/EHsc /MD /DVERSION_INFO=\`"0.1.0-phase1\`"" `
    -Xcompiler "/wd4819" `
    -Isrc `
    -I"$cudaInclude" `
    -I"$pythonInclude" `
    -I"$pybind11Include"

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ C++ compilation failed!" -ForegroundColor Red
    exit 1
}
# Step 3: Link into Python module
Write-Host "`nStep 3: Linking Python module..." -ForegroundColor Yellow
& $nvcc `
    --shared `
    build_direct/vector_ops.obj `
    build_direct/binding.obj `
    -o planar_cuda.pyd `
    -L"$cudaLib" `
    -L"$pythonLib" `
    -lcudart `
    "-lpython$pythonVersion"ib" `
    -lcudart `
    -lpython311

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Linking failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Module linked: planar_cuda.pyd" -ForegroundColor Green

# Test import
Write-Host "`nStep 4: Testing import..." -ForegroundColor Yellow
$testScript = @'
import os
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
import planar_cuda
print(f'Version: {planar_cuda.__version__}')
print(f'CUDA enabled: {planar_cuda.cuda_enabled}')
result = planar_cuda.add_vectors([1,2,3], [4,5,6])
print(f'Test: [1,2,3] + [4,5,6] = {result}')
assert result == [5,7,9], 'Test failed!'
print('All checks passed!')
'@

$testResult = & $pythonExe -c $testScript 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host $testResult -ForegroundColor Green
    
    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host "  BUILD SUCCESSFUL!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "`nRun tests with:" -ForegroundColor Cyan
    Write-Host "  pytest tests/cuda_tests/test_phase1_pipeline.py -v" -ForegroundColor White
} else {
    Write-Host $testResult -ForegroundColor Red
    Write-Host "`nImport test failed!" -ForegroundColor Red
    exit 1
}
