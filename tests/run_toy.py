from src.main import load_data, preprocess, choose_metric
from src.dnc_cluster import DnCClusterer
from src.similarity import DistStats
import numpy as np

if __name__ == "__main__":
    # small toy dataset
    rng = np.random.default_rng(1)
    T = 128
    n = 30
    t = np.linspace(0, 2*np.pi, T)
    X = np.vstack([
        (np.sin(2.0*t) + 0.05*rng.normal(size=T)) for _ in range(n//3)
    ] + [
        (np.sin(3.4*t + 0.7) + 0.05*rng.normal(size=T)) for _ in range(n//3)
    ] + [
        (0.6*np.sign(np.sin(1.1*t)) + 0.06*rng.normal(size=T)) for _ in range(n - 2*(n//3))
    ])

    X = preprocess(X, None)
    base = choose_metric("dtw")
    stats = DistStats(base, enable_cache=True)
    dist = stats.wrap()
    clusterer = DnCClusterer(dist_fn=dist, min_size=5, max_depth=6, seed_sample=10)

    print("Running toy clustering...")
    tree = clusterer.fit(X)
    leaves = DnCClusterer.collect_leaves(tree)
    print(f"Leaves: {len(leaves)}")
    print(f"Distance calls: {stats.count}, cache size: {stats.cache_size()}")
    print("Done.")
