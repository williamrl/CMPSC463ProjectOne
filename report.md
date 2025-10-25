# Project Report (student-style)

This is a friendly report template for the time-series clustering project I completed. Fill each section with your outputs and short explanations. I tried to keep the writing casual and clear so graders can immediately see what was done.

1) Short project description
--------------------------------
I wrote a simple divide-and-conquer clustering tool for short time-series segments (PulseDB). The main pieces are:
- a distance function (DTW or correlation)
- a recursive split based on farthest seeds (DnC)
- a closest-pair check inside clusters to pick representatives
- Kadane's algorithm on each segment to find its most active interval

2) How to run (installation)
--------------------------------
Copy the instructions from `readme.md`. Quick reminder:

- Create a virtualenv and install requirements: `pip install -r requirements.txt`.
- Use `--subset` to limit data for faster runs during testing.

3) Code structure (short)
--------------------------
- `src/loader.py` — load time-series CSVs (expects rows=segments, cols=time).
- `src/similarity.py` — DTW and correlation distance wrappers (with stats/caching).
- `src/dnc_cluster.py` — divide-and-conquer clustering implementation.
- `src/closest_pair.py` — brute-force closest pair helper.
- `src/kadane.py` — Kadane max-subarray utilities.
- `src/visualize.py` — plotting helpers for quick inspection.
- `src/main.py` — CLI and pipeline orchestration.

4) Algorithms (brief, student-friendly)
------------------------------------
- DTW: Dynamic Time Warping aligns two series; it's flexible but slow.
- Correlation: cheap alternative that measures similarity of shape.
- DnC split: pick a random seed, find a farthest seed, and partition by closer seed.
- Closest pair: find the two series in a cluster with minimum distance (brute force here).
- Kadane: find the contiguous interval with maximum sum (used here on an activity signal derived from each segment).

5) Quick verification
-----------------------
I included `tests/run_toy.py` that runs the pipeline on a small synthetic dataset and prints the number of leaf clusters and distance-call stats. Use that for quick checks before running the full experiment.

6) Results with 1000 segments
------------------------------
Run the pipeline on your 1000-segment CSV (or a generated sample). The repo writes `results/timing.json` with runtime and stats; put pretty screenshots in this section, along with a short interpretation (2-4 bullets).

7) Discussion and limitations
------------------------------
Talk about where the algorithm worked, where DTW was slow, and what could be improved (caching, pruning, parallelism). Also mention any dataset caveats.

8) Conclusions and next steps
--------------------------------
Summarize the main takeaways and list small follow-up improvements (e.g., LB_Keogh pruning, parallel DTW, better seeding).

References and dataset
-----------------------
Add links to PulseDB and any references used.
