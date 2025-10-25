# Time-Series Clustering Results

## Example Run Output
Here are the results from running the clustering pipeline on a 100-segment subset using correlation distance:

```
Running main module...
⏳ Clustering...
✅ Clustering complete.
Clustering time: 0.44s
Distance function calls: 10,787 (cache size=5,445)

Total leaves: 15
Cluster sizes: [5, 9, 6, 10, 8, 4, 9, 2, 7, 10, 10, 7, 6, 2, 5]
```

### Cluster Analysis
The algorithm found 15 natural clusters with sizes ranging from 2 to 10 segments each. Key metrics:
- Average cluster size: 6.67 segments
- Largest cluster: 10 segments
- Smallest cluster: 2 segments
- Typical within-cluster distance: ~0.008-0.016 (correlation metric)

### Closest Pairs
Each cluster's closest pair gives insight into cluster quality. Examples:
```
Cluster 6: size=9, closest=(1, 7), dist=0.0073  # Very tight cluster
Cluster 2: size=6, closest=(3, 5), dist=0.0166  # More diverse cluster
```

### Kadane Intervals
The maximum subarray analysis found consistent intervals across segments:
```
seg 0: score=41.439, interval=(0,254)
seg 1: score=43.342, interval=(0,254)
...
```
This suggests the entire signal length contains significant activity.

### Performance Metrics
- Runtime: 0.44 seconds for 100 segments
- Distance calculations: 10,787 total calls
- Cache effectiveness: 5,445 cached values (~50% hit rate)

## Verification Results
The implementation was verified on both toy and real data:

1. DTW/Correlation distances preserve expected properties:
   - d(x,x) ≈ 0 (identity)
   - d(x,y) = d(y,x) (symmetry)
   - phase-shifted signals cluster together

2. Clustering correctly separates distinct patterns:
   - Synthetic sin/cos waves → separate clusters
   - Similar waveforms → same cluster
   - Small diameter within clusters

3. Kadane's algorithm finds meaningful intervals:
   - Detects obvious peaks in test data
   - Handles edge cases (all-negative, constant)