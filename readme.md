# Time-Series Clustering and Segment Analysis on PulseDB (Divide-and-Conquer)

This project implements **unsupervised time-series clustering** using **divide-and-conquer** strategies (no ML libraries), **closest-pair analysis inside clusters**, and **maximum subarray (Kadane)** analysis for per-segment activity. It is designed to analyze **PulseDB** 10-second ABP/PPG/ECG segments.

## Installation

```bash
python -m venv .venv
# PowerShell (Windows):
.\.venv\Scripts\Activate.ps1
# or temporarily bypass policy for this session:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
pip install -r requirements.txt

# Time-Series Clustering on PulseDB (small student project)

This repo contains a small divide-and-conquer time-series clustering project I put together for a class assignment. It clusters short segments (for example ABP/PPG/ECG slices) from PulseDB without heavy ML frameworks — just DTW or correlation, a closest-pair check to validate clusters, and Kadane's algorithm to find active parts of each signal.

Installation
------------

I used Python 3.8+. On Windows (PowerShell) the setup is:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Quick usage
-----------

Run a small toy verification (fast):

```powershell
python -m tests.run_toy
```

Run the main pipeline (it will use the builtin toy dataset if you don't pass --data):

```powershell
python -m src.main
```

For faster iteration, use the correlation metric instead of DTW:

```powershell
python -m src.main --metric corr --subset 200
```

Notes
-----
- Use `--subset N` to limit the number of segments processed while you experiment.
- `--viz` shows plots; don't use it on headless servers unless you save the figures instead.
- After each run the script writes a small JSON summary to `results/timing.json` with timing and distance-call stats.
