import numpy as np
from typing import Callable, Tuple


def closest_pair_bruteforce(
    X: np.ndarray, dist_fn: Callable[[np.ndarray, np.ndarray], float]
) -> Tuple[Tuple[int, int], float]:
    """
    Find the closest pair of time-series segments in X using brute-force.
    Returns ((i, j), distance)
    """
    n = len(X)
    if n < 2:
        return ((0, 0), float("inf"))

    best_pair = (0, 1)
    best_dist = float("inf")

    for i in range(n):
        for j in range(i + 1, n):
            d = dist_fn(X[i], X[j])
            if d < best_dist:
                best_dist = d
                best_pair = (i, j)

    return best_pair, best_dist
