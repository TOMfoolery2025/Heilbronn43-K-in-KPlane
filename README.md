# K-Planar Graph Minimizer

This repository contains a small GUI and optimizer to reduce edge crossings (k-planarity) of a graph. The GUI is built with CustomTkinter + Matplotlib; the optimizer uses a simulated annealing approach.

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
