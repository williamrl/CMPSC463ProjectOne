"""Save example figures for the report."""
import os
import numpy as np
from src.main import load_data, preprocess, choose_metric
from src.dnc_cluster import DnCClusterer
from src.closest_pair import closest_pair_bruteforce
from src.similarity import DistStats
from src.visualize import plot_cluster_examples, plot_pair, plot_kadane_interval
from src.kadane import find_max_subarray

def main():
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Load and preprocess toy data
    X = load_data(None)  # use built-in toy data
    X = preprocess(X, subset=100)
    
    # Set up clustering with stats
    base_fn = choose_metric("corr")  # use correlation for speed
    stats = DistStats(base_fn, enable_cache=True)
    dist_fn = stats.wrap()
    
    # Cluster
    print("⏳ Clustering...")
    clusterer = DnCClusterer(
        dist_fn=dist_fn,
        min_size=10,
        max_depth=8,
        seed_sample=20
    )
    tree = clusterer.fit(X)
    leaves = DnCClusterer.collect_leaves(tree)
    print(f"✅ Found {len(leaves)} clusters")
    
    # Save cluster examples
    plot_cluster_examples(
        leaves, n_per_cluster=3,
        suptitle="Example segments from each cluster",
        save_path="results/clusters.png"
    )
    
    # Save closest pair from largest cluster
    if len(leaves) > 0 and leaves[0].shape[0] >= 2:
        (i, j), d = closest_pair_bruteforce(leaves[0], dist_fn)
        plot_pair(
            leaves[0][i], leaves[0][j],
            title=f"Closest pair in largest cluster (dist={d:.3f})",
            save_path="results/closest_pair.png"
        )
    
    # Save Kadane interval example
    x = X[0]  # first segment
    start, end = find_max_subarray(x)
    plot_kadane_interval(
        x, start, end,
        title=f"Kadane interval for first segment",
        save_path="results/kadane_interval.png"
    )
    
    print("✅ Saved figures to results/")
    print(f"Distance calls: {stats.count}, cache hits: {stats.cache_size()}")

if __name__ == "__main__":
    main()