from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


"""Simple plotting helpers used by the project.

These are intentionally lightweight: they show plots by default but accept an
optional save path so you can run them on headless machines and keep PNGs.
"""


def plot_cluster_examples(leaves: List[np.ndarray], n_per_cluster: int = 3, suptitle: str = "", save_path: Optional[str] = None):
    """Plot a few example segments from each leaf cluster.

    Args:
        leaves: list of numpy arrays (each array is a cluster of segments).
        n_per_cluster: how many examples to plot per cluster.
        suptitle: overall title for the figure.
        save_path: if provided, save the figure to this path instead of showing it.
    """
    rows = len(leaves)
    cols = n_per_cluster
    if rows == 0:
        return
    plt.figure(figsize=(3 * cols, 2 * rows))
    for r, Xc in enumerate(leaves):
        pick = min(cols, Xc.shape[0])
        # choose evenly spaced examples
        idx = np.linspace(0, Xc.shape[0]-1, pick, dtype=int)
        for c, i in enumerate(idx):
            ax = plt.subplot(rows, cols, r * cols + c + 1)
            ax.plot(Xc[i], lw=1)
            ax.set_title(f"cluster {r} seg {i}")
            ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle(suptitle)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_pair(x: np.ndarray, y: np.ndarray, title: str = "Closest Pair", save_path: Optional[str] = None):
    """Plot two signals on the same axes (used for representative pairs).

    Accepts a save_path to allow headless runs.
    """
    plt.figure(figsize=(6,3))
    plt.plot(x, label="A"); plt.plot(y, label="B")
    plt.legend(); plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_kadane_interval(x: np.ndarray, start: int, end: int, title: str = "Kadane interval", save_path: Optional[str] = None):
    """Show the signal and highlight the maximum subarray interval.

    start and end are inclusive indices. If start > end nothing special is drawn.
    """
    plt.figure(figsize=(6,3))
    plt.plot(x, alpha=0.6, label="signal")
    if start <= end:
        xs = range(start, end+1)
        plt.plot(xs, x[start:end+1], lw=2, label="max-interval")
    plt.legend(); plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
