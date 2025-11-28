import os
paths = os.environ['PATH'].split(';')
cuda_paths = [p for p in paths if 'CUDA' in p.upper()]
print('CUDA paths in PATH:')
for p in cuda_paths:
    print(f'  {p}')

print('\nChecking nvrtc64_120_0.dll:')
import os.path
dll_path = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvrtc64_120_0.dll'
print(f'  Exists: {os.path.exists(dll_path)}')
