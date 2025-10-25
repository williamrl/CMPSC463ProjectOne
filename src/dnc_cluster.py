import numpy as np


class DnCClusterer:
    """
    Divide-and-Conquer Clustering for Time-Series Segments.
    Splits recursively until clusters are small or have low diameter.
    """

    def __init__(self, dist_fn, min_size=25, diam_thresh=None, max_depth=12, seed_sample=200):
        self.dist_fn = dist_fn
        self.min_size = min_size
        self.diam_thresh = diam_thresh
        self.max_depth = max_depth
        self.seed_sample = seed_sample

    # ------------------------------------------------------------------

    def _cluster_diameter(self, X):
        """Compute the maximum pairwise distance (diameter) of the cluster."""
        n = len(X)
        max_d = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                xi = np.ravel(X[i])
                yj = np.ravel(X[j])
                d = self.dist_fn(xi, yj)
                if d > max_d:
                    max_d = d
        return max_d

    # ------------------------------------------------------------------

    def _split_cluster(self, X):
        """Divide cluster X into two subclusters using farthest-point heuristic."""
        n = len(X)
        if n < 2:
            return X, np.empty((0, X.shape[1]))

        X = np.array([np.ravel(x) for x in X])
        rng = np.random.default_rng()

        # seed-sampling: pick a small sample to choose the farthest seed
        sample_k = min(self.seed_sample, n)
        sample_idx = rng.choice(n, sample_k, replace=False)

        # initial seed index (pick random from full range)
        i0 = int(rng.integers(0, n))
        seed1 = X[i0]

        # find farthest seed among the sampled indices to reduce DTW calls
        dists_sample = [self.dist_fn(seed1, X[idx]) for idx in sample_idx]
        i1 = int(sample_idx[int(np.argmax(dists_sample))])
        seed2 = X[i1]

        cluster1, cluster2 = [], []
        for x in X:
            d1 = self.dist_fn(x, seed1)
            d2 = self.dist_fn(x, seed2)
            if d1 < d2:
                cluster1.append(x)
            else:
                cluster2.append(x)

        return np.array(cluster1), np.array(cluster2)

    # ------------------------------------------------------------------

    def _fit_recursive(self, X, depth):
        """Recursive DnC clustering core."""
        n = len(X)
        if n <= self.min_size or depth >= self.max_depth:
            return X

        diam = self._cluster_diameter(X)
        if self.diam_thresh is not None and diam <= self.diam_thresh:
            return X

        left, right = self._split_cluster(X)

        if len(left) == 0 or len(right) == 0:
            return X

        return {
            "depth": depth,
            "diam": diam,
            "left": self._fit_recursive(left, depth + 1),
            "right": self._fit_recursive(right, depth + 1),
        }

    # ------------------------------------------------------------------

    def fit(self, X):
        """Run the DnC clustering."""
        return self._fit_recursive(X, 0)

    # ------------------------------------------------------------------

    @staticmethod
    def collect_leaves(tree):
        """Gather all leaf clusters into a list."""
        if isinstance(tree, dict):
            return DnCClusterer.collect_leaves(tree["left"]) + DnCClusterer.collect_leaves(tree["right"])
        else:
            return [np.array(tree)]
