import numpy as np
import pandas as pd
from typing import Optional


class TimeSeriesLoader:
    """
    Handles loading and preprocessing of time-series data
    (either from CSV or synthetic data).
    """

    def __init__(self, path: str, wide_format: bool = True):
        self.path = path
        self.wide_format = wide_format

    def load(self) -> np.ndarray:
        """
        Loads the CSV file and returns a NumPy array.
        Each row represents one time-series segment.
        """
        try:
            df = pd.read_csv(self.path)
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {self.path}: {e}")

        if self.wide_format:
            return df.to_numpy(dtype=float)
        else:
            # If data is in long format, pivot into wide format
            df_wide = df.pivot(index="segment_id", columns="time", values="value")
            return df_wide.to_numpy(dtype=float)

    # --- Static preprocessing helpers ---
    @staticmethod
    def normalize_zscore(X: np.ndarray) -> np.ndarray:
        """Normalize each segment (row) using z-score."""
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True)
        return (X - mean) / (std + 1e-8)

    @staticmethod
    def ensure_1d_segments(X: np.ndarray) -> np.ndarray:
        """Ensure all inputs are 2D (n_segments x n_timepoints)."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        elif X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        return X

    @staticmethod
    def take_subset(X: np.ndarray, subset: Optional[int]) -> np.ndarray:
        """Optionally select a subset of samples."""
        if subset is not None and subset < X.shape[0]:
            return X[:subset]
        return X
