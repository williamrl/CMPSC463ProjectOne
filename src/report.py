from __future__ import annotations
import numpy as np
from typing import Dict, Any, List, Callable, Tuple
from src.closest_pair import closest_pair_bruteforce
from src.kadane import activity_signal, kadane_max_subarray

def summarize_clusters(leaves: List[np.ndarray], dist_fn: Callable[[np.ndarray, np.ndarray], float]) -> List[Dict[str, Any]]:
    rows = []
    for ci, Xc in enumerate(leaves):
        entry: Dict[str, Any] = {"cluster_id": ci, "size": int(Xc.shape[0])}
        (i, j), d = closest_pair_bruteforce(Xc, dist_fn)
        entry["closest_pair"] = (int(i), int(j))
        entry["closest_distance"] = float(d)
        rows.append(entry)
    return rows

def kadane_table(X: np.ndarray, mode: str = "diff_abs", limit: int | None = 10) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    n = X.shape[0]
    k = n if limit is None else min(limit, n)
    for idx in range(k):
        sig = activity_signal(X[idx], mode=mode)
        score, s, e = kadane_max_subarray(sig)
        out.append({"segment": idx, "score": float(score), "start": int(s), "end": int(e)})
    return out

def print_summary(cluster_tree: Dict[str, Any], leaves: List[np.ndarray], cluster_rows: List[Dict[str, Any]], kadane_rows: List[Dict[str, Any]]):
    print("\n==================== SUMMARY ====================")
    print(f"Total leaves: {len(leaves)}")
    print("Cluster sizes:", [leaf.shape[0] for leaf in leaves])
    print("\nClosest pairs per cluster:")
    for r in cluster_rows:
        print(f"  Cluster {r['cluster_id']}: size={r['size']}, closest={r['closest_pair']}, dist={r['closest_distance']:.4f}")
    print("\nKadane (first few segments):")
    for r in kadane_rows:
        print(f"  seg {r['segment']}: score={r['score']:.3f}, interval=({r['start']},{r['end']})")
