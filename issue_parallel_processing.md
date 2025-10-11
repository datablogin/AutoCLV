# Parallel Processing for Large-Scale Customer Analytics

## Feature Request

Add parallel processing capabilities to RFM, Lens 1, and Lens 2 analyses to improve performance with very large customer datasets (10M+ customers).

## Problem Statement

Current implementation processes customers sequentially with excellent O(n) scaling:
- 1,000 customers: ~0.08s
- 5,000 customers: ~0.42s
- **Projected 1M customers**: ~80s (1.3 minutes)
- **Projected 10M customers**: ~800s (13 minutes)

For enterprise datasets with 10M+ customers, sub-5-minute processing would improve batch workflow efficiency.

## Proposed Solution

Implement **configurable parallel processing** with the following features:

### 1. Automatic Parallel Processing for Large Datasets

Enable parallel processing automatically when customer count exceeds a threshold:

```python
# Default behavior: auto-enable at 10M customers
rfm_metrics = calculate_rfm(
    period_aggregations,
    observation_end=datetime(2023, 12, 31),
    parallel=True,  # Auto-enabled if len(unique_customers) > 10_000_000
    parallel_threshold=10_000_000,  # Configurable threshold
    n_workers=None  # None = auto (cpu_count - 1)
)
```

### 2. Manual Parallel Processing Control

Allow users to override automatic behavior:

```python
# Force parallel processing for smaller datasets
rfm_metrics = calculate_rfm(
    period_aggregations,
    observation_end=datetime(2023, 12, 31),
    parallel=True,
    parallel_threshold=100_000,  # Enable at 100K customers
    n_workers=4  # Explicit worker count
)

# Disable parallel processing even for large datasets
rfm_metrics = calculate_rfm(
    period_aggregations,
    observation_end=datetime(2023, 12, 31),
    parallel=False  # Explicit opt-out
)
```

### 3. Parallel Processing Strategy

Leverage **embarrassingly parallel** nature of customer-level calculations:

**RFM Calculation** (customer_base_audit/foundation/rfm.py:174):
- Group period_aggregations by customer_id
- Distribute customer groups across workers
- Each worker calculates RFM for subset of customers
- Merge results (simple concatenation)

**Lens 1 Analysis** (customer_base_audit/analyses/lens1.py:113):
- Distribute RFM metrics across workers
- Each worker calculates partial aggregations
- Reduce step combines partial results (revenue sums, customer counts)

**Lens 2 Comparison** (customer_base_audit/analyses/lens2.py:115):
- Set operations (retained/churned/new) are parallelizable
- Partition customer sets across workers
- Merge migration results

## Implementation Details

### API Changes

#### calculate_rfm() signature change:
```python
def calculate_rfm(
    period_aggregations: List[PeriodAggregation],
    observation_end: datetime,
    parallel: bool = True,  # Auto-enable for large datasets
    parallel_threshold: int = 10_000_000,  # Customer count threshold
    n_workers: Optional[int] = None  # None = auto-detect
) -> List[RFMMetrics]:
    """
    Calculate RFM metrics from period aggregations.

    Args:
        period_aggregations: List of period-level customer aggregations
        observation_end: End date of observation period
        parallel: Enable parallel processing (auto if customer_count > threshold)
        parallel_threshold: Customer count threshold for auto-parallel
        n_workers: Number of parallel workers (None = cpu_count - 1)

    Returns:
        List of RFMMetrics, one per customer

    Notes:
        Parallel processing is automatically enabled when the number of unique
        customers exceeds parallel_threshold (default: 10M). For smaller datasets,
        sequential processing is faster due to lower overhead.
    """
```

#### analyze_single_period() signature change:
```python
def analyze_single_period(
    rfm_metrics: List[RFMMetrics],
    rfm_scores: Optional[List[RFMScore]] = None,
    percentiles: tuple[int, ...] = DEFAULT_PARETO_PERCENTILES,
    parallel: bool = True,  # Auto-enable for large datasets
    parallel_threshold: int = 10_000_000,
    n_workers: Optional[int] = None
) -> Lens1Metrics:
```

#### analyze_period_comparison() signature change:
```python
def analyze_period_comparison(
    period1_rfm: List[RFMMetrics],
    period2_rfm: List[RFMMetrics],
    all_customer_history: Optional[List[str]] = None,
    period1_metrics: Optional[Lens1Metrics] = None,
    period2_metrics: Optional[Lens1Metrics] = None,
    parallel: bool = True,  # Auto-enable for large datasets
    parallel_threshold: int = 10_000_000,
    n_workers: Optional[int] = None
) -> Lens2Metrics:
```

### Technology Choice

**Recommended**: Python `multiprocessing` module
- Built-in, no additional dependencies
- Good performance for CPU-bound tasks
- Works well with dataclasses (serializable)

**Alternative**: `concurrent.futures.ProcessPoolExecutor`
- Higher-level API
- Better error handling
- Easier to test

### Performance Targets

With 4-core parallelization:

| Customer Count | Current Time | Target Time | Speedup |
|----------------|--------------|-------------|---------|
| 1M | ~80s | ~80s | 1x (no parallel) |
| 10M | ~800s (13 min) | ~200s (3.3 min) | 4x |
| 50M | ~4000s (66 min) | ~1000s (16.7 min) | 4x |
| 100M | ~8000s (133 min) | ~2000s (33 min) | 4x |

**Note**: Speedup is less than linear due to:
- Process spawning overhead
- Data serialization costs
- Result merging time
- Memory bandwidth limitations

### Backward Compatibility

**100% Backward Compatible** - All existing code continues to work:

```python
# Existing code (no changes needed)
rfm = calculate_rfm(period_aggregations, observation_end)
lens1 = analyze_single_period(rfm)
lens2 = analyze_period_comparison(period1_rfm, period2_rfm)

# Behavior:
# - For < 10M customers: sequential processing (current behavior)
# - For >= 10M customers: automatic parallel processing (new behavior)
```

## Testing Requirements

1. **Unit Tests**:
   - Parallel and sequential results are identical
   - Edge cases: empty datasets, single customer, 100K customers
   - Worker count validation (1 worker = sequential, 4 workers = parallel)

2. **Performance Tests**:
   - Measure overhead at various dataset sizes
   - Verify 3-4x speedup with 4 workers at 10M+ customers
   - Memory usage profiling (ensure no memory leaks)

3. **Integration Tests**:
   - Full pipeline with parallel processing enabled
   - Verify results match sequential processing (statistical correctness)

## Benefits

1. **Performance**: 3-4x speedup for 10M+ customer datasets
2. **Scalability**: Enables processing of 100M+ customer datasets in reasonable time
3. **User Control**: Configurable thresholds and worker counts
4. **Automatic**: Works out-of-box for large datasets (no code changes needed)
5. **Backward Compatible**: Existing code continues to work

## Risks and Mitigations

**Risk**: Increased memory usage with parallel processing
- **Mitigation**: Use chunked processing; only load subset per worker

**Risk**: Process spawning overhead for small datasets
- **Mitigation**: Default threshold of 10M customers ensures overhead is negligible

**Risk**: Non-deterministic results if not careful with RNG
- **Mitigation**: No randomness in RFM/Lens calculations; deterministic by design

**Risk**: Complex debugging with multiprocessing
- **Mitigation**: Add `parallel=False` flag for debugging; comprehensive logging

## Related Components

**Data Mart Build Time** (separate optimization opportunity):
- Currently 99% of processing time for Track A operations
- Data mart build is O(n) for transactions, not customers
- Parallel processing would target data mart module (not Track A scope)
- See separate analysis below

## References

- Performance Analysis: `TRACK_A_PERFORMANCE_ANALYSIS.md`
- Scalability Tests: `tests/test_performance_analysis.py::TestPerformanceBenchmarks::test_scalability_analysis`
- Current Performance: 5,000 customers in 0.424s (linear scaling confirmed)

## Priority

**Medium Priority** - Current performance is acceptable for most use cases:
- 1M customers: ~1.3 minutes (acceptable for batch processing)
- 10M customers: ~13 minutes (acceptable but improvable)
- 100M customers: ~2 hours (would benefit from parallelization)

This feature becomes **High Priority** when:
- Customer datasets regularly exceed 10M
- Sub-5-minute processing is required for operational workflows
- Real-time CLV scoring pipelines are implemented

## Implementation Effort

**Estimated Effort**: 2-3 days
- Day 1: Implement parallel RFM calculation with tests
- Day 2: Implement parallel Lens 1 and Lens 2 with tests
- Day 3: Performance benchmarking, documentation, edge case testing

**Complexity**: Medium
- Embarrassingly parallel problem (ideal for multiprocessing)
- Existing code is functional-style (no global state)
- Main challenge: result merging and error handling
