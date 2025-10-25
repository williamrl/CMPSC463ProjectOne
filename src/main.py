from __future__ import annotations
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
from typing import Optional
from typing import Optional
from src.loader import TimeSeriesLoader
from src.similarity import dtw_distance, corr_distance
from src.similarity import DistStats
from src.dnc_cluster import DnCClusterer
from src.report import summarize_clusters, kadane_table, print_summary
from src.visualize import plot_cluster_examples, plot_pair, plot_kadane_interval
from src.closest_pair import closest_pair_bruteforce
import time, json

def build_argparser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Divide-and-Conquer Time-Series Clustering on PulseDB segments")
    p.add_argument("--data", type=str, default=None,
                   help="Path or URL to CSV with rows=segments, cols=time. If omitted, a toy dataset is used.")
    p.add_argument("--subset", type=int, default=None, help="Optional limit on number of segments to process.")
    p.add_argument("--metric", type=str, choices=["dtw","corr"], default="dtw", help="Similarity metric.")
    p.add_argument("--min_size", type=int, default=25, help="Min cluster size to stop splitting.")
    p.add_argument("--diam_thresh", type=float, default=None, help="Optional diameter threshold to stop splitting.")
    p.add_argument("--max_depth", type=int, default=12, help="Max recursion depth.")
    p.add_argument("--seed_sample", type=int, default=200, help="Sample size used to pick farthest seeds.")
    p.add_argument("--viz", action="store_true", help="Show matplotlib visualizations.")
    p.add_argument("--kadane_mode", type=str, choices=["diff_abs","raw"], default="diff_abs",
                   help="Activity signal for Kadane.")
    return p.parse_args()

def load_data(args: argparse.Namespace) -> np.ndarray:
    if args.data is None:
        # Toy verification set (3 latent patterns with noise)
        rng = np.random.default_rng(0)
        T = 256
        nA, nB, nC = 40, 40, 20
        t = np.linspace(0, 2*np.pi, T)
        A = (np.sin(2.0*t)[None, :] + 0.10*rng.normal(size=(nA, T)))
        B = (np.sin(3.4*t + 0.7)[None, :] + 0.10*rng.normal(size=(nB, T)))
        C = (0.6*np.sign(np.sin(1.1*t))[None, :] + 0.12*rng.normal(size=(nC, T)))
        X = np.vstack([A, B, C])
        return X
    # Real data path/URL
    loader = TimeSeriesLoader(args.data, wide_format=True)
    X = loader.load()
    return X

def preprocess(X: np.ndarray, subset: Optional[int]) -> np.ndarray:
    X = TimeSeriesLoader.ensure_1d_segments(X)
    X = TimeSeriesLoader.normalize_zscore(X)
    X = TimeSeriesLoader.take_subset(X, subset)
    return X

def choose_metric(name: str):
    return dtw_distance if name == "dtw" else corr_distance

def main():
    args = build_argparser()
    X = load_data(args)
    X = preprocess(X, args.subset)
    dist_fn = choose_metric(args.metric)
    # Wrap distance function to collect call counts and enable optional caching
    dist_stats = DistStats(dist_fn, enable_cache=True)
    dist_fn = dist_stats.wrap()

    # Divide-and-Conquer clustering
    clusterer = DnCClusterer(
        dist_fn=dist_fn,
        min_size=args.min_size,
        diam_thresh=args.diam_thresh,
        max_depth=args.max_depth,
        seed_sample=args.seed_sample
    )
    print("⏳ Clustering...")
    t0 = time.perf_counter()
    try:
        tree = clusterer.fit(X)
        t1 = time.perf_counter()
    except KeyboardInterrupt:
        t1 = time.perf_counter()
        elapsed = t1 - t0
        print("\n⛔ Keyboard interrupt received. Clustering stopped early.")
        # Save partial stats (distance call count) so user can inspect progress
        try:
            calls = dist_stats.count
            cache_sz = dist_stats.cache_size()
            os.makedirs("results", exist_ok=True)
            with open(os.path.join("results", "timing.json"), "w", encoding="utf-8") as f:
                json.dump({"elapsed_s": elapsed, "dist_calls": calls, "cache_size": cache_sz, "interrupted": True}, f, indent=2)
            print(f"Saved partial timing to results/timing.json (elapsed {elapsed:.2f}s, dist_calls={calls})")
        except Exception:
            pass
        return
    leaves = DnCClusterer.collect_leaves(tree)
    print("✅ Clustering complete.")

    # Performance summary
    elapsed = t1 - t0
    print(f"Clustering time: {elapsed:.2f}s")
    try:
        calls = dist_stats.count
        cache_sz = dist_stats.cache_size()
        print(f"Distance function calls: {calls} (cache size={cache_sz})")
        # save to results/
        os.makedirs("results", exist_ok=True)
        with open(os.path.join("results", "timing.json"), "w", encoding="utf-8") as f:
            json.dump({"elapsed_s": elapsed, "dist_calls": calls, "cache_size": cache_sz}, f, indent=2)
    except Exception:
        pass

    # Closest pair per cluster
    rows = summarize_clusters(leaves, dist_fn)

    # Kadane analysis on a few segments (project requirement)
    kadane_rows = kadane_table(X, mode=args.kadane_mode, limit=10)

    # Optional visualizations
    if args.viz:
        plot_cluster_examples(leaves, n_per_cluster=3, suptitle="Sampled segments per leaf cluster")
        # Show a representative closest pair from the largest cluster
        if len(leaves) > 0 and leaves[0].shape[0] >= 2:
            (i, j), d = closest_pair_bruteforce(leaves[0], dist_fn)
            plot_pair(leaves[0][i], leaves[0][j], title=f"Closest pair (dist={d:.3f})")
        # Plot Kadane interval on one example
        score, s, e = kadane_rows[0]["score"], kadane_rows[0]["start"], kadane_rows[0]["end"]
        plot_kadane_interval(X[0], s, e, title=f"Kadane interval (score={score:.2f})")

    # Text summary
    print_summary(tree, leaves, rows, kadane_rows)

if __name__ == "__main__":
    print("Running main module...")
    main()
