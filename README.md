# Notes
- All code included
- Live Demo @ Heilbronn43LiveDemo
- Presenation @ Heilbron 43
- Group members: Swaraj Shrestha, Felix Nack, Tsai Ming-Han, Ziheng Li

# K-Planar Graph Minimizer

This repository contains a small GUI and optimizer to reduce edge crossings (k-planarity) of a graph. The GUI is built with CustomTkinter + Matplotlib; the optimizer uses a simulated annealing approach.

## Environment Requirements

### System Requirements
- **Operating System**: Windows 10/11
- **Python**: 3.9+ (3.10 or newer recommended, tested on 3.11.4)
- **Shell**: Windows PowerShell

### Python Dependencies
```
customtkinter      # Modern GUI framework
matplotlib         # Plotting and visualization
numpy             # Numerical operations
packaging         # Version handling
networkx          # Graph algorithms
```

### Optional: GPU Acceleration (CUDA)
For high-performance CUDA acceleration:

- **CUDA Toolkit**: 12.6.20 or compatible
- **GPU**: NVIDIA GPU with compute capability 8.0+ (tested on RTX 4060)
- **Compiler**: Visual Studio 2022 with C++ build tools
- **Additional Python Packages**:
  - `pybind11` - Python-C++ binding
  - `pytest` - Testing framework
  - `cupy-cuda12x` - CUDA array library (optional)

**CUDA Installation Path** (Windows default):
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
```

## Prerequisites
- Python 3.9+ (3.10 or newer recommended)
- Windows PowerShell (these commands are written for PowerShell)

## Quick Start (Windows PowerShell)

Create an isolated virtual environment named `heilbron-43`, activate it, and install dependencies:

```powershell
# From the repo root
# If 'python' is available on PATH
python -m venv .\heilbron-43
.\heilbron-43\Scripts\Activate.ps1
# If 'python' is not found, create with the launcher and then activate
# py -3 -m venv .\heilbron-43
# .\heilbron-43\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the unit tests to verify everything is working:

```powershell
python -m unittest -v
```

Launch the GUI app:

```powershell
python .\src\app.py
```

Close the app window to stop the optimizer thread cleanly.

## Data Files
- Example instances are under `live-2025-example-instances/`.
- A simple input example is provided in `sample.json`.
- The expected JSON format:
  - `nodes`: list of `{ id, x, y }` where `id` is a zero-based integer index.
  - `edges`: list of `{ source, target }` referencing node ids.
  - Optional: `width`, `height` to set canvas bounds (defaults are large).

## Common Issues
- Execution policy blocking activation:
  - If `Activate.ps1` is blocked, temporarily relax the policy for the current session:
    ```powershell
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\heilbron-43\Scripts\Activate.ps1
    ```
- Tkinter availability:
  - The official Python installer for Windows includes Tkinter by default. If `tkinter` errors appear, reinstall Python using the official installer from python.org.

## Deactivate / Remove the Environment
```powershell
deactivate
# To delete the venv folder entirely
Remove-Item -Recurse -Force .\heilbron-43
```

## Project Structure
- `src/app.py`: GUI application (load JSON, visualize, drag nodes, run optimizer)
- `src/solver.py`: Simulated annealing solver (positions + energy updates)
- `src/scorer.py`: Vectorized crossing counting utilities (k and total crossings)
- `tests/`: Unittest suite for scorer and solver
- `requirements.txt`: Python dependencies
