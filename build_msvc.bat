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
