# Cycle 1 Build - Response File Method
# Uses @file.rsp to avoid command-line length limits

Write-Host "`n=== Cycle 1 Build ===" -ForegroundColor Cyan

if (-not (Test-Path "build_artifacts")) { mkdir build_artifacts | Out-Null }

# Create response file for nvcc
@"
--shared
src/cuda_utils/planar_cuda.cu
-o build_artifacts/planar_cuda.pyd
-arch=sm_89
--compiler-options /EHsc /MD
-Isrc
-IC:\Users\aloha\AppData\Local\Programs\Python\Python311\Include
-ID:\D_backup\2025\tum\25W\hackthon\Hackathon-Nov-25-Heilbronn43\heilbron-43\Lib\site-packages\pybind11\include
-L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64"
-LC:\Users\aloha\AppData\Local\Programs\Python\Python311\libs
-lcudart
-lpython311
-Xlinker /NODEFAULTLIB:MSVCRT
-Xlinker legacy_stdio_definitions.lib
-Xlinker ucrt.lib
-Xlinker vcruntime.lib
"@ | Out-File -FilePath "nvcc.rsp" -Encoding ASCII

# Build batch that calls nvcc with response file
@'
@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
nvcc @nvcc.rsp
'@ | Out-File -FilePath "b.bat" -Encoding ASCII

cmd /c b.bat
$r = $LASTEXITCODE
Remove-Item "b.bat","nvcc.rsp" -ErrorAction SilentlyContinue

if ($r -ne 0) { Write-Host "Build failed" -ForegroundColor Red; exit 1 }

Write-Host "[OK] Compiled successfully" -ForegroundColor Green

# Test
& heilbron-43\Scripts\python.exe -c "import os,sys; os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin'); sys.path.insert(0,'build_artifacts'); import planar_cuda; print(planar_cuda.__version__); s=planar_cuda.PlanarSolver([0,10,5],[0,0,10],[(0,1),(1,2),(2,0)]); print(f'Crossings: {s.calculate_total_crossings()}')"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[SUCCESS] Next: pytest tests/cuda_tests/test_gpu_geometry.py -v" -ForegroundColor Cyan
}
