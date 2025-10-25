import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def ensure_flat(arr):
    """Force any numpy array or list into a clean 1D float array."""
    if isinstance(arr, list):
        arr = np.array(arr, dtype=float)
    arr = np.asarray(arr, dtype=float)
    return arr.reshape(-1)  # flatten safely


def dtw_distance(x, y):
    """DTW distance with absolute flattening protection."""
    x = ensure_flat(x)
    y = ensure_flat(y)
    # fastdtw will call dist(a, b) where a and b are elements of x and y.
    # For 1-D numeric time series those elements are scalars, so passing
    # scipy.spatial.distance.euclidean (which expects array-like inputs)
    # causes "Input vector should be 1-D." when it receives plain floats.
    # Use simple absolute difference for scalar distance instead.
    d, _ = fastdtw(x, y, dist=lambda a, b: float(abs(a - b)))
    return float(d)


class DistStats:
    """Wrap a distance function to collect basic stats and optional caching.

    Usage:
        ds = DistStats(dtw_distance)
        dist_fn = ds.wrap()
        d = dist_fn(a, b)
        print(ds.count, ds.cache_size())
    """

    def __init__(self, base_fn, enable_cache=True):
        self.base_fn = base_fn
        self.enable_cache = enable_cache
        self._count = 0
        self._cache = {}

    def wrap(self):
        def wrapped(a, b):
            # flatten inputs to stable types for caching
            aa = ensure_flat(a).tobytes()
            bb = ensure_flat(b).tobytes()
            # canonical key (order matters for asymmetric distances)
            key = (aa, bb)
            if self.enable_cache and key in self._cache:
                self._count += 1
                return self._cache[key]

            val = float(self.base_fn(a, b))
            self._count += 1
            if self.enable_cache:
                self._cache[key] = val
            return val

        return wrapped

    @property
    def count(self):
        return self._count

    def cache_size(self):
        return len(self._cache)

    def reset(self):
        self._count = 0
        self._cache.clear()


def corr_distance(x, y):
    """(1 - correlation)/2 distance in [0, 1]."""
    x = ensure_flat(x)
    y = ensure_flat(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return 1.0
    corr = np.corrcoef(x, y)[0, 1]
    return float((1 - corr) / 2)
