import numpy as np
from typing import Tuple


def activity_signal(x: np.ndarray, mode: str = "diff_abs") -> np.ndarray:
    """
    Compute an 'activity signal' from a time series.
    mode = 'diff_abs' → absolute difference between consecutive points
    mode = 'raw' → the raw signal itself
    """
    if mode == "diff_abs":
        return np.abs(np.diff(x))
    elif mode == "raw":
        return x
    else:
        raise ValueError("Invalid mode: choose 'diff_abs' or 'raw'")


def kadane_max_subarray(arr: np.ndarray) -> Tuple[float, int, int]:
    """
    Standard Kadane's algorithm to find the contiguous subarray
    with the maximum sum. Returns (max_sum, start_index, end_index)
    """
    max_sum = float("-inf")
    current_sum = 0
    start = 0
    best_start = 0
    best_end = 0

    for i, val in enumerate(arr):
        current_sum += val
        if current_sum > max_sum:
            max_sum = current_sum
            best_start = start
            best_end = i
        if current_sum < 0:
            current_sum = 0
            start = i + 1

    return max_sum, best_start, best_end
