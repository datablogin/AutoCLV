---
date: 2025-10-11T03:07:08Z
researcher: Claude (Sonnet 4.5)
git_commit: b322e747a82a9cc896db9f95e61ad36901ab47bf
branch: feature/issue-31-validation-framework
repository: AutoCLV
topic: "Track B Architecture for Remaining Lens Analysis and Functions"
tags: [research, codebase, track-b, lens-analysis, architecture, models, validation]
status: complete
last_updated: 2025-10-10
last_updated_by: Claude (Sonnet 4.5)
---

# Research: Track B Architecture for Remaining Lens Analysis and Functions

**Date**: 2025-10-11T03:07:08Z
**Researcher**: Claude (Sonnet 4.5)
**Git Commit**: b322e747a82a9cc896db9f95e61ad36901ab47bf
**Branch**: feature/issue-31-validation-framework
**Repository**: AutoCLV

## Research Question

Research Track B architecture to document the existing patterns, implementations, and established design for lens analyses and model functions. Identify what remains to be implemented and document how the existing architectural patterns should be applied to the remaining work.

## Summary

Track B has successfully completed Phase 1-4 work (Cohorts, Lens 3, all models, validation framework) with a well-established architecture. Lenses 1-3 demonstrate a consistent pattern: frozen dataclasses, comprehensive validation, NumPy-style documentation, and Decimal precision for financial calculations. The model layer wraps PyMC-Marketing with enhanced validation. **Remaining work**: Lens 5 (Overall Customer Base Health, Issue #35) and CLI enhancements. The established patterns provide a clear template for implementing Lens 5 with the same rigor and consistency as existing lenses.

## Detailed Findings

### Track B Current Status

#### Completed (Phase 1-4)

**Phase 1: Foundation**
- ✅ Cohorts infrastructure (`customer_base_audit/foundation/cohorts.py`) - 22KB
- ✅ Lens 3: Cohort Evolution (`customer_base_audit/analyses/lens3.py`) - 422 lines

**Phase 3: Models**
- ✅ Model preparation (`customer_base_audit/models/model_prep.py`)
- ✅ BG/NBD model wrapper (`customer_base_audit/models/bg_nbd.py`) - PR #69
- ✅ Gamma-Gamma model wrapper (`customer_base_audit/models/gamma_gamma.py`) - PR #65
- ✅ CLV calculator (`customer_base_audit/models/clv_calculator.py`) - PR #71

**Phase 4: Validation**
- ✅ Model diagnostics (`customer_base_audit/validation/diagnostics.py`) - PR #74
- ✅ Validation framework (`customer_base_audit/validation/validation.py`) - PR #77 (recently fixed)

#### Remaining (Phase 5)

**Lens 5: Overall Customer Base Health**
- Location: `customer_base_audit/analyses/lens5.py` - currently 565-byte placeholder
- Status: Not implemented
- Issue: #35
- Dependencies: Lens 4 (multi-cohort comparison) from Track A

**CLI Enhancements**
- Batch scoring capability
- Report generation
- Status: Not implemented
- Dependencies: All lenses complete

### Lens Implementation Inventory

| Lens | Status | Lines | Tests | Description |
|------|--------|-------|-------|-------------|
| Lens 1 | ✅ Implemented | 290 | 480 lines (test_lens1.py) | Single Period Analysis |
| Lens 2 | ✅ Implemented | 370 | 737 lines (test_lens2.py) | Period-to-Period Comparison |
| Lens 3 | ✅ Implemented | 422 | 594 lines (test_lens3.py) | Single Cohort Evolution |
| Lens 4 | ⚠️ Placeholder | 15 | None | Multi-Cohort Comparison (Track A) |
| Lens 5 | ⚠️ Placeholder | 16 | None | Overall Customer Base Health (Track B) |

**Total Implementation**: 3 of 5 lenses complete (1,082 lines)
**Total Tests**: 1,811 lines of unit tests + 557 lines of integration tests

### Established Architectural Patterns

#### Pattern 1: Immutable Dataclasses with Validation

**Evidence from Lens 3** (`customer_base_audit/analyses/lens3.py:36-101`):

```python
@dataclass(frozen=True)
class CohortPeriodMetrics:
    """Metrics for a cohort in a specific period after acquisition."""
    period_number: int
    active_customers: int
    cumulative_activation_rate: float
    avg_orders_per_customer: float
    avg_revenue_per_customer: float
    avg_orders_per_cohort_member: float
    avg_revenue_per_cohort_member: float
    total_revenue: float

    def __post_init__(self) -> None:
        """Validate metrics are in reasonable ranges."""
        if self.period_number < 0:
            raise ValueError(f"period_number must be >= 0, got {self.period_number}")
        if self.active_customers < 0:
            raise ValueError(f"active_customers must be non-negative, got {self.active_customers}")
        if not (0 <= self.cumulative_activation_rate <= 1):
            raise ValueError(
                f"cumulative_activation_rate must be in [0, 1], got {self.cumulative_activation_rate}"
            )
        # ... more validation
```

**Key Characteristics**:
- `frozen=True` enforces immutability
- `__post_init__()` validates all constraints
- Descriptive error messages with actual values
- Type hints for all fields
- Comprehensive docstrings with Attributes section

#### Pattern 2: Main Analysis Functions

**Evidence from Lens 1** (`customer_base_audit/analyses/lens1.py:106-153`):

```python
def analyze_single_period(
    rfm_metrics: Sequence[RFMMetrics],
    rfm_scores: Sequence[RFMScore] | None = None,
) -> Lens1Metrics:
    """Perform Lens 1 analysis on a single period.

    Parameters
    ----------
    rfm_metrics:
        RFM metrics for all customers in the period
    rfm_scores:
        Optional RFM scores (1-5 quintiles) for distribution analysis

    Returns
    -------
    Lens1Metrics
        Complete single period analysis results

    Examples
    --------
    >>> from customer_base_audit.foundation.rfm import calculate_rfm
    >>> rfm = calculate_rfm(period_aggregations, observation_end)
    >>> metrics = analyze_single_period(rfm)
    >>> metrics.total_customers
    1000
    """
    # Early return for empty input
    if not rfm_metrics:
        return Lens1Metrics(/* zero-valued fields */)

    # Main analysis logic
    # ...

    return Lens1Metrics(/* calculated fields */)
```

**Key Characteristics**:
- Single entry point function
- Type-annotated parameters with defaults
- NumPy-style docstrings
- Early returns for edge cases
- Returns immutable dataclass

#### Pattern 3: Decimal Precision for Financial Calculations

**Evidence from Lens 1** (`customer_base_audit/analyses/lens1.py:21-37, 158-189`):

```python
# Module-level constants
PERCENTAGE_PRECISION = Decimal("0.01")  # 2 decimal places for percentages
DEFAULT_PARETO_PERCENTILES = (10, 20)

# In analysis function:
one_time_buyer_pct = (
    Decimal(one_time_buyers) / Decimal(total_customers) * 100
).quantize(PERCENTAGE_PRECISION, rounding=ROUND_HALF_UP)

total_revenue = sum(m.total_spend for m in rfm_metrics).quantize(
    Decimal("0.01"), rounding=ROUND_HALF_UP
)
```

**Key Characteristics**:
- `Decimal` type for all financial values
- Explicit conversion: `Decimal(value)`
- `.quantize()` with precision constants
- `ROUND_HALF_UP` rounding mode
- Module-level precision constants

#### Pattern 4: Model Wrapper Architecture

**Evidence from BG/NBD** (`customer_base_audit/models/bg_nbd.py:28-449`):

```python
@dataclass
class BGNBDConfig:
    """Configuration for BG/NBD model training."""
    method: str = "map"
    chains: int = 4
    draws: int = 2000
    tune: int = 1000
    random_seed: int = 42

class BGNBDModelWrapper:
    """Wrapper for BG/NBD purchase frequency model."""

    def __init__(self, config: BGNBDConfig = BGNBDConfig()) -> None:
        self.config = config
        self.model: Optional[BetaGeoModel] = None

    def _validate_bg_nbd_data(
        self, data: pd.DataFrame, operation: str, allow_empty: bool = False
    ) -> None:
        """Validate input data for BG/NBD operations."""
        # Comprehensive validation logic
        # ...

    def fit(self, data: pd.DataFrame) -> None:
        """Fit BG/NBD model to customer transaction data."""
        self._validate_bg_nbd_data(data, operation="fit", allow_empty=False)
        # Create and fit PyMC-Marketing model
        # ...

    def predict_purchases(
        self, data: pd.DataFrame, time_periods: float
    ) -> pd.DataFrame:
        """Predict expected number of purchases in next time_periods."""
        # State validation, then prediction
        # ...
```

**Key Characteristics**:
- Config dataclass with defaults
- Wrapper class with `Optional[Model]` attribute
- Private validation methods (`_validate_*`)
- State checking before predictions
- Consistent return types (DataFrames with customer_id)

#### Pattern 5: Comprehensive Validation

**Evidence from Validation Framework** (`customer_base_audit/validation/validation.py:225-243`):

```python
# Validate inputs
if len(actual) != len(predicted):
    raise ValueError(
        f"actual and predicted must have same length: "
        f"{len(actual)} != {len(predicted)}"
    )

if len(actual) == 0:
    raise ValueError("actual and predicted cannot be empty")

# Convert to numpy arrays for calculations
actual_values = actual.values.astype(float)
predicted_values = predicted.values.astype(float)

# Check for NaN or inf values
if np.any(np.isnan(actual_values)) or np.any(np.isinf(actual_values)):
    raise ValueError("actual contains NaN or inf values")
if np.any(np.isnan(predicted_values)) or np.any(np.isinf(predicted_values)):
    raise ValueError("predicted contains NaN or inf values")
```

**Key Characteristics**:
- Validate early, fail fast
- Descriptive error messages
- Check data types, NaN/Inf, duplicates
- Context in error messages
- Separate validation methods

### Model Implementation Architecture

#### BG/NBD Model Wrapper

**Location**: `customer_base_audit/models/bg_nbd.py` (449 lines)

**Structure**:
- `BGNBDConfig` dataclass for configuration
- `BGNBDModelWrapper` class wrapping PyMC-Marketing's `BetaGeoModel`
- Methods: `fit()`, `predict_purchases()`, `calculate_probability_alive()`
- Private: `_validate_bg_nbd_data()`, `_check_mcmc_convergence()`

**Key Integration**: Wraps PyMC-Marketing BetaGeoModel with enhanced validation and error handling.

#### Gamma-Gamma Model Wrapper

**Location**: `customer_base_audit/models/gamma_gamma.py` (303 lines)

**Structure**:
- `GammaGammaConfig` dataclass (similar to BGNBDConfig)
- `GammaGammaModelWrapper` class wrapping `GammaGammaModel`
- Methods: `fit()`, `predict_spend()`
- Private: `_check_mcmc_convergence()`

**Key Constraint**: Requires frequency >= 2 (excludes one-time buyers)

#### CLV Calculator

**Location**: `customer_base_audit/models/clv_calculator.py` (461 lines)

**Structure**:
- `CLVScore` dataclass with validation
- `CLVCalculator` class combining BG/NBD + Gamma-Gamma
- Methods: `calculate_clv()`
- Business parameters: profit_margin, discount_rate, time_horizon_months

**Key Formula**: `CLV = predicted_purchases × predicted_avg_value × profit_margin × discount_factor`

#### Model Preparation

**Location**: `customer_base_audit/models/model_prep.py` (315 lines)

**Functions**:
- `prepare_bg_nbd_inputs()` - Converts `PeriodAggregation` to BG/NBD format
- `prepare_gamma_gamma_inputs()` - Converts to Gamma-Gamma format
- Input dataclasses: `BGNBDInput`, `GammaGammaInput`

**Key Difference**:
- BG/NBD: `frequency = total_orders - 1` (repeat purchases)
- Gamma-Gamma: `frequency = total_orders` (all transactions)

### Validation Framework Architecture

#### Core Components

**Location**: `customer_base_audit/validation/validation.py` (474 lines)

**Exports**:
1. `ValidationMetrics` dataclass (lines 27-82)
2. `temporal_train_test_split()` function (lines 84-174)
3. `calculate_clv_metrics()` function (lines 177-297)
4. `cross_validate_clv()` function (lines 300-474)

#### ValidationMetrics Dataclass

```python
@dataclass(frozen=True)
class ValidationMetrics:
    mae: Decimal           # Mean Absolute Error
    mape: Decimal          # Mean Absolute Percentage Error
    rmse: Decimal          # Root Mean Squared Error
    arpe: Decimal          # Aggregate Revenue Percent Error
    r_squared: Decimal     # Can be negative!
    sample_size: int
```

**Target Performance**:
- MAPE < 20%: Individual predictions accurate
- ARPE < 10%: Aggregate revenue accurate
- R² > 0.5: Model explains >50% variance

#### Temporal Splitting

**Function**: `temporal_train_test_split(transactions, train_end_date, observation_end_date)`

**Returns**:
1. `train_transactions`: Before train_end_date (for model fitting)
2. `observation_transactions`: Up to observation_end_date (for RFM calculation)
3. `test_transactions`: Between dates (for ground truth)

**Key Design**: Respects temporal ordering, no data leakage.

#### Cross-Validation

**Function**: `cross_validate_clv(transactions, model_pipeline, n_folds=5, ...)`

**Strategy**: Forward-chaining with expanding windows:
- Fold 1: Train on months 0-12, test on 13-15
- Fold 2: Train on months 0-15, test on 16-18
- Fold 3: Train on months 0-18, test on 19-21

**Key Design**: Training window expands (never shrinks), simulates production retraining.

#### Recent Bug Fixes

**Context**: PR #77 recently fixed critical bugs:
1. R² can be negative (removed non-negativity validation)
2. Cross-validation bug: now correctly uses `obs_txns` not `train_txns`
3. Epsilon handling: uses `_EPSILON = 1e-10` for floating-point comparisons
4. Actual CLV calculation: requires explicit 'amount' column

### Implementation Patterns Summary

| Pattern | Description | Found In |
|---------|-------------|----------|
| Frozen Dataclasses | `frozen=True` with `__post_init__` validation | All analyses, validation |
| Decimal Arithmetic | `Decimal` type with `.quantize()` | Lenses 1-3, model prep |
| Type Hints | Full annotations on all functions | All modules |
| NumPy Docstrings | Extended docstrings with Examples | All public functions |
| Module Constants | ALL_CAPS constants at module level | Lenses 1-2, validation |
| Wrapper Pattern | Wrapping external libraries with validation | Models (PyMC-Marketing) |
| Early Returns | Handle empty/edge cases first | All analysis functions |
| Explicit Exports | `__all__` lists in `__init__.py` | All packages |
| Class-Based Tests | `TestClassName` with descriptive methods | All test files |

## Code References

### Lens Implementations
- `customer_base_audit/analyses/lens1.py:106-291` - Lens 1 analysis and revenue concentration
- `customer_base_audit/analyses/lens2.py:155-370` - Lens 2 period comparison
- `customer_base_audit/analyses/lens3.py:144-422` - Lens 3 cohort evolution
- `customer_base_audit/analyses/lens4.py:1-15` - Lens 4 placeholder (Track A)
- `customer_base_audit/analyses/lens5.py:1-16` - Lens 5 placeholder (Track B, Issue #35)

### Model Implementations
- `customer_base_audit/models/model_prep.py:105-315` - Input preparation functions
- `customer_base_audit/models/bg_nbd.py:54-449` - BG/NBD wrapper
- `customer_base_audit/models/gamma_gamma.py:49-303` - Gamma-Gamma wrapper
- `customer_base_audit/models/clv_calculator.py:138-461` - CLV calculator

### Validation Framework
- `customer_base_audit/validation/validation.py:27-82` - ValidationMetrics dataclass
- `customer_base_audit/validation/validation.py:84-174` - Temporal splitting
- `customer_base_audit/validation/validation.py:177-297` - Metrics calculation
- `customer_base_audit/validation/validation.py:300-474` - Cross-validation

### Test Coverage
- `tests/test_lens1.py` - 480 lines testing Lens 1
- `tests/test_lens2.py` - 737 lines testing Lens 2
- `tests/test_lens3.py` - 594 lines testing Lens 3
- `tests/test_validation.py` - 460 lines testing validation framework
- `tests/test_integration_five_lenses.py` - 411 lines integration tests

## Architecture Documentation

### Current Patterns for Lens Implementations

Based on Lenses 1-3, the established pattern for implementing a new lens:

1. **Module Structure**:
   - Module-level constants (precision, thresholds)
   - One or more frozen dataclasses for results
   - Main `analyze_*` function as entry point
   - Helper functions for specific calculations
   - Full test coverage in corresponding test file

2. **Dataclass Structure**:
   ```python
   @dataclass(frozen=True)
   class LensNMetrics:
       """Lens N: Description."""
       field1: Type
       field2: Type
       # ...

       def __post_init__(self) -> None:
           """Validate metrics."""
           # Comprehensive validation
   ```

3. **Main Analysis Function**:
   ```python
   def analyze_lens_n(
       input_data: Sequence[SomeType],
       optional_param: Type | None = None,
   ) -> LensNMetrics:
       """Perform Lens N analysis.

       Parameters
       ----------
       input_data: Description
       optional_param: Description

       Returns
       -------
       LensNMetrics
           Complete analysis results

       Examples
       --------
       >>> # Runnable example
       """
       # Early return for empty input
       if not input_data:
           return LensNMetrics(/* zero values */)

       # Analysis logic
       # ...

       return LensNMetrics(/* calculated values */)
   ```

4. **Helper Functions** (if needed):
   - Extract specific calculations
   - Provide reusable functionality
   - Follow same documentation standards

### Model Wrapper Patterns

Based on BG/NBD and Gamma-Gamma implementations:

1. **Configuration Dataclass**:
   - Fitting method ('map' or 'mcmc')
   - MCMC parameters (chains, draws, tune, seed)
   - Default values provided

2. **Wrapper Class Structure**:
   - `__init__` accepts config, initializes `model: Optional[ExternalModel]`
   - Private validation method (`_validate_*_data`)
   - `fit()` method for training
   - Prediction methods returning DataFrames
   - State checking before predictions

3. **Validation Approach**:
   - Required columns check
   - Data type validation
   - NaN/Inf checks
   - Domain-specific constraints
   - Descriptive error messages

### Validation Framework Patterns

Based on current validation.py implementation:

1. **Frozen Dataclasses for Results**:
   - All metrics as `Decimal` for precision
   - Validation in `__post_init__()`
   - Immutable after creation

2. **Temporal Awareness**:
   - Date-based splitting (not random)
   - Respects temporal ordering
   - Prevents data leakage

3. **Cross-Validation Strategy**:
   - Expanding window (not sliding)
   - Configurable fold sizes
   - Handles insufficient data gracefully

## Remaining Track B Work

### Lens 5: Overall Customer Base Health (Issue #35)

**Current State**: 565-byte placeholder at `customer_base_audit/analyses/lens5.py`

**Dependencies**:
- Lens 4 (Multi-Cohort Comparison) must be complete - this is Track A work
- All previous lenses (1-3) complete ✅

**Expected Implementation** (based on established patterns):

```python
@dataclass(frozen=True)
class Lens5Metrics:
    """Lens 5: Overall customer base health."""
    current_period_metrics: Lens1Metrics
    cohort_comparison: Lens4Metrics  # Requires Track A completion
    customer_base_composition: dict[str, int]  # Cohort -> count
    revenue_by_cohort: dict[str, Decimal]
    weighted_retention_rate: Decimal
    projected_annual_revenue: Decimal
    health_score: Decimal  # Composite 0-100 score

    def __post_init__(self) -> None:
        """Validate health metrics."""
        if not (0 <= self.health_score <= 100):
            raise ValueError(f"health_score must be in [0, 100], got {self.health_score}")
        # ... more validation

def analyze_overall_health(
    current_lens1: Lens1Metrics,
    lens4_cohorts: Lens4Metrics,
    clv_scores: Sequence[CLVScore],
) -> Lens5Metrics:
    """Integrative view synthesizing Lenses 1-4 for holistic health.

    Key analyses:
    - Customer base composition by cohort
    - Revenue contribution by cohort and tenure
    - Overall retention trends
    - Projected future revenue based on CLV
    """
    # Implementation following established patterns
    # ...
```

**Test File**: Create `tests/test_lens5.py` following the pattern of test_lens1-3.py

**Estimated Effort**: Similar to Lens 3 (~400-500 lines implementation, ~600 lines tests)

### CLI Enhancements

**Current State**: Not implemented

**Required Functionality**:

1. **Batch CLV Scoring**:
   ```bash
   clv score --input transactions.json --output clv_scores.csv --model-method map
   ```
   - Load transactions from JSON
   - Build data mart
   - Prepare model inputs
   - Train BG/NBD and Gamma-Gamma
   - Calculate CLV scores
   - Export to CSV

2. **Five Lenses Report Generation**:
   ```bash
   clv report --input transactions.json --output-dir reports/ --period-granularity MONTH
   ```
   - Generate all lens metrics
   - Export as JSON files
   - Create markdown summary

**Implementation Location**: Extend `customer_base_audit/cli.py`

**Integration Tests**: Create `tests/integration/test_batch_cli.py`

## Historical Context (from thoughts/)

**Enterprise CLV Implementation Plan**: `thoughts/shared/plans/2025-10-08-enterprise-clv-implementation.md`
- Comprehensive 10-week implementation plan
- Track B assigned: Models (BG/NBD, Gamma-Gamma), Lens 3, Lens 5, CLI enhancements
- Phase 5 (Weeks 7-8): Track B works on Lens 5 + CLI while Track A completes Lens 4
- Parallel development strategy using git worktrees

**AGENTS.md**: Root-level project guide
- Recently updated (lines 72-82) to reflect Phase 1-4 completion
- Track B status: "✅ PHASE 1-4 COMPLETE | 🔄 PHASE 5 IN PROGRESS"
- Lists remaining work: Lens 5 and CLI enhancements

## Related Research

None found in `thoughts/shared/research/` directory related to Track B architecture.

## Applying Patterns to Remaining Work

### For Lens 5 Implementation

Following the established architecture from Lenses 1-3:

1. **Define Dataclass** (like CohortPeriodMetrics, Lens3Metrics):
   - `Lens5Metrics` as frozen dataclass
   - Fields for composition, revenue, retention, projections, health score
   - Comprehensive `__post_init__` validation
   - Decimal types for financial values

2. **Implement Main Function** (like analyze_cohort_evolution):
   - `analyze_overall_health(current_lens1, lens4_cohorts, clv_scores) -> Lens5Metrics`
   - Early return for empty inputs
   - Calculate weighted metrics across cohorts
   - Compute composite health score (0-100)
   - NumPy-style docstring with examples

3. **Add Helper Functions** (if needed):
   - `calculate_health_score()` - Composite scoring algorithm
   - `project_annual_revenue()` - Forward-looking projection
   - Follow same documentation standards

4. **Create Test File**:
   - `tests/test_lens5.py` with class-based organization
   - Test dataclass validation
   - Test analysis function with known inputs
   - Test edge cases (empty inputs, extreme values)
   - Test health score boundaries (0-100)

5. **Update Package Exports**:
   - Add to `customer_base_audit/analyses/__init__.py`
   - Export `Lens5Metrics` and `analyze_overall_health`

### For CLI Enhancements

Following model wrapper patterns and validation framework:

1. **Batch Scoring Command**:
   - Validate input file exists
   - Parse transactions from JSON
   - Use existing model preparation functions
   - Train models with progress indicators
   - Calculate CLV using CLVCalculator
   - Export to CSV with proper schema
   - Handle errors gracefully with context

2. **Report Generation Command**:
   - Orchestrate all five lens analyses
   - Generate JSON exports for each lens
   - Create markdown summary with key metrics
   - Include visualizations (if time permits)
   - Validate all inputs before processing

3. **Integration Tests**:
   - Test batch scoring on Texas CLV synthetic data
   - Verify output CSV schema
   - Test report generation produces all expected files
   - Test error handling (missing files, invalid data)

### Resilient Design Principles

Based on existing implementations:

1. **Immutability**: Use frozen dataclasses to prevent accidental state mutations
2. **Validation**: Validate all inputs early with descriptive error messages
3. **Type Safety**: Full type annotations enable static analysis
4. **Precision**: Decimal arithmetic for financial calculations
5. **Edge Cases**: Explicit handling of empty inputs, zero values, NaN/Inf
6. **Documentation**: NumPy-style docstrings with runnable examples
7. **Testing**: Comprehensive unit and integration tests
8. **Modularity**: Separate concerns (prep, train, predict, validate)

## Open Questions

1. **Lens 4 Completion Timeline**: Lens 5 depends on Lens 4 (Track A). When will Lens 4 be available?
   - **Resolution**: Lens 5 can define interface expectations for Lens 4 metrics and proceed with implementation

2. **Health Score Algorithm**: What formula should be used for the composite health score (0-100)?
   - **Resolution**: Document the weighting: 40% retention, 30% cohort quality trend, 15% revenue concentration, 15% growth momentum

3. **CLI Output Formats**: Should CLI support additional formats beyond CSV/JSON (e.g., Parquet, Excel)?
   - **Resolution**: Start with CSV/JSON, add formats in future iterations if needed

4. **Performance Targets**: What are the expected performance requirements for batch CLV scoring?
   - **Resolution**: Existing target from plan: Process 100K transactions in <30 seconds

## Conclusion

Track B has established robust architectural patterns across analyses, models, and validation. The remaining work (Lens 5 and CLI enhancements) should follow these established patterns for consistency, maintainability, and quality. The frozen dataclass pattern, comprehensive validation, Decimal precision, and NumPy-style documentation are the cornerstones of the resilient architecture already in place. Lens 5 implementation can proceed by following the exact same structure as Lenses 1-3, ensuring the Five Lenses framework is completed with the same rigor and reliability.
