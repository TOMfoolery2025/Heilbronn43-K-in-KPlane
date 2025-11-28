"""
Setup script for planar_cuda Python package
Alternative to CMake for simpler builds
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
from pathlib import Path

class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        import subprocess
        
        # Create build directory
        build_dir = Path(self.build_temp)
        build_dir.mkdir(parents=True, exist_ok=True)
        
        # CMake configuration
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={Path(self.build_lib).absolute()}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DCMAKE_BUILD_TYPE=Release'
        ]
        
        # Build arguments
        build_args = ['--config', 'Release']
        
        # Run CMake
        subprocess.check_call(['cmake', str(Path().absolute())] + cmake_args, cwd=build_dir)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_dir)

setup(
    name='planar_cuda',
    version='0.1.0',
    author='LCN Team',
    description='CUDA-accelerated LCN solver',
    ext_modules=[CMakeExtension('planar_cuda')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20',
        'pytest>=6.0',
    ],
)
