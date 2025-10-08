# Enterprise CLV Calculator Implementation Plan

## Overview

This plan implements an enterprise-grade Customer Lifetime Value (CLV) calculator using the Five Lenses approach from "The Customer-Base Audit" by Fader, Hardie, and Ross, combined with probabilistic BG/NBD and Gamma-Gamma models. The implementation builds on the existing AutoCLV foundation (data mart, customer contract, synthetic data) to deliver production-ready CLV scoring capabilities.

## Current State Analysis

### Existing Strengths
- **Solid foundation**: CustomerDataMartBuilder (`customer_base_audit/foundation/data_mart.py:92-105`) provides production-ready transaction aggregation
- **Customer identity system**: CustomerContract (`customer_base_audit/foundation/customer_contract.py:17-163`) with cohort support via acquisition timestamps
- **Synthetic data toolkit**: Texas CLV client (`texas_clv_client.py:49-106`) with 1,000 customers across 4 cities
- **Financial precision**: Decimal arithmetic with ROUND_HALF_UP for monetary calculations
- **Test coverage**: Comprehensive tests for foundation components

### Key Gaps
- No implementation of Five Lenses analyses (lens1.py through lens5.py are empty placeholders)
- No probabilistic CLV models (BG/NBD, Gamma-Gamma)
- No RFM calculation utilities
- No model validation or drift detection framework
- No production deployment infrastructure

## Desired End State

**A production-ready CLV calculator that:**
1. Implements all Five Lenses for customer-base audit (descriptive analytics)
2. Provides probabilistic CLV predictions via BG/NBD + Gamma-Gamma models
3. Delivers individual customer CLV scores with MAPE < 20% and ARPE < 10%
4. Supports batch processing of 1M+ customers
5. Includes model validation, drift detection, and monitoring capabilities
6. Provides reusable components for enterprise analytics teams

**Verification Criteria:**
- All Five Lenses produce validated outputs on Texas CLV synthetic data
- CLV predictions achieve target accuracy metrics on held-out test sets
- Full test suite passes with >90% coverage
- End-to-end pipeline processes 100K transactions in <30 seconds
- Documentation enables new analysts to generate CLV scores independently

## What We're NOT Doing

This plan explicitly excludes:
- Real-time CLV scoring APIs (batch processing only for MVP)
- Custom visualization dashboards (focus on data outputs, not UI)
- Integration with specific data warehouses (examples only, not production connectors)
- Reverse ETL activation workflows (data export only)
- Multi-language support (Python only)
- Mobile or web application interfaces
- GDPR/CCPA compliance automation (manual process documentation only)

## Implementation Approach

The implementation follows a **phased approach prioritizing Critical and High Priority items first**, with each phase delivering working, testable functionality. We build on the existing foundation infrastructure and follow established patterns (immutable dataclasses, Decimal arithmetic, comprehensive validation).

**Priority Levels:**
- **Critical**: Core CLV functionality, foundational components
- **High Priority**: Model validation, quality assurance, essential Lenses
- **Medium**: Advanced features, optimization, additional tooling
- **Low**: Nice-to-have features, future enhancements

## Parallel Work Strategy and Git Worktree Workflow

### Overview of Parallel Tracks

This implementation can be accelerated by splitting work across **multiple parallel tracks** using git worktrees. The dependency graph allows for 2-3 simultaneous work streams in most phases.

**Benefits of worktree-based parallelization:**
- Independent feature branches without context switching
- Run tests in one worktree while developing in another
- Parallel code reviews without blocking implementation
- Isolate experimental approaches (MAP vs. MCMC) in separate worktrees

### Git Worktree Setup

**Initial setup (worktrees contained within AutoCLV directory):**
```bash
# Main worktree (already exists)
cd /Users/robertwelborn/PycharmProjects/AutoCLV

# Create .worktrees directory to contain all parallel worktrees
mkdir -p .worktrees

# Add .worktrees to .gitignore to avoid tracking worktree directories
echo ".worktrees/" >> .gitignore
git add .gitignore
git commit -m "chore: add .worktrees to gitignore"

# Create worktree for Track A (RFM + Lenses)
# -b flag creates a new branch based on current branch
git worktree add -b feature/track-a-rfm-lenses .worktrees/track-a

# Create worktree for Track B (Models)
git worktree add -b feature/track-b-clv-models .worktrees/track-b

# Create worktree for Track C (Documentation/Testing)
git worktree add -b feature/track-c-docs-tests .worktrees/track-c

# List all worktrees
git worktree list
```

**Directory structure after setup:**
```
/Users/robertwelborn/PycharmProjects/AutoCLV/
├── .git/                          # Main git directory
├── .worktrees/                    # Container for all worktrees (gitignored)
│   ├── track-a/                   # Track A worktree
│   │   ├── customer_base_audit/
│   │   ├── tests/
│   │   └── ...                    # Full copy of repo on feature/track-a-rfm-lenses
│   ├── track-b/                   # Track B worktree
│   │   ├── customer_base_audit/
│   │   ├── tests/
│   │   └── ...                    # Full copy of repo on feature/track-b-clv-models
│   └── track-c/                   # Track C worktree
│       └── ...
├── customer_base_audit/           # Main worktree files
├── tests/
└── ...
```

**Workflow per track:**
```bash
# Developer A works in track-a worktree
cd .worktrees/track-a
# Make changes, commit, push
git add customer_base_audit/foundation/rfm.py
git commit -m "feat(rfm): implement RFM calculation utilities"
git push -u origin feature/track-a-rfm-lenses

# Developer B works in track-b worktree simultaneously
cd ../track-b  # Relative path from track-a
# OR from main directory:
cd /Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-b
git add customer_base_audit/models/bg_nbd.py
git commit -m "feat(models): implement BG/NBD model wrapper"
git push -u origin feature/track-b-clv-models

# Merge tracks back to main periodically
cd /Users/robertwelborn/PycharmProjects/AutoCLV  # Back to main worktree
git checkout feature/tx-clv-synthetic
git merge feature/track-a-rfm-lenses
git merge feature/track-b-clv-models
```

**Cleanup when done:**
```bash
# Remove worktrees after merging
cd /Users/robertwelborn/PycharmProjects/AutoCLV
git worktree remove .worktrees/track-a
git worktree remove .worktrees/track-b
git worktree remove .worktrees/track-c

# Optional: Remove .worktrees directory if empty
rmdir .worktrees
```

**Benefits of this structure:**
- ✅ All worktrees contained in single `.worktrees/` directory
- ✅ PycharmProjects folder stays clean
- ✅ Easy to open multiple IDE windows (one per worktree)
- ✅ Gitignored to prevent accidentally tracking worktree contents
- ✅ Clear naming: `track-a`, `track-b`, `track-c` instead of `AutoCLV-track-a`

### Dependency Graph and Parallelization Opportunities

**Phase 1 (Weeks 1-2): Critical Foundation**
```
Track A: RFM + Lens 1 (Week 1)
├─ customer_base_audit/foundation/rfm.py
├─ customer_base_audit/analyses/lens1.py
└─ tests/test_rfm.py, tests/test_lens1.py

Track B: Cohort Infrastructure (Week 1)
├─ customer_base_audit/foundation/cohorts.py
└─ tests/test_cohorts.py

Track C: Documentation Start (Week 1-2, ongoing)
└─ docs/user_guide.md (skeleton)

# Merge point: End of Week 1
# Enables: Lens 2 and Lens 3 (both depend on RFM + cohorts)
```

**Phase 2 (Weeks 2-3): Lens 2-3**
```
Track A: Lens 2 (Week 2)
├─ customer_base_audit/analyses/lens2.py
└─ tests/test_lens2.py

Track B: Lens 3 (Week 2)
├─ customer_base_audit/analyses/lens3.py
└─ tests/test_lens3.py

Track C: Documentation (ongoing)
└─ docs/user_guide.md (Lenses 1-3 sections)

# Merge point: End of Week 2
# Enables: Model development (Phase 3)
```

**Phase 3 (Weeks 3-5): CLV Models**
```
Track A: Model Preparation + BG/NBD (Week 3-4)
├─ customer_base_audit/models/model_prep.py
├─ customer_base_audit/models/bg_nbd.py
└─ tests/test_model_prep.py, tests/test_bg_nbd.py

Track B: Gamma-Gamma + CLV Calculator (Week 4-5)
├─ customer_base_audit/models/gamma_gamma.py
├─ customer_base_audit/models/clv_calculator.py
└─ tests/test_gamma_gamma.py, tests/test_clv_calculator.py

Track C: Integration Tests (Week 5)
└─ tests/integration/test_clv_pipeline.py

# Merge point: End of Week 5
# Enables: Validation framework (Phase 4)
```

**Phase 4 (Weeks 5-6): Validation**
```
Track A: Validation Framework (Week 5-6)
├─ customer_base_audit/validation/validation.py
└─ tests/test_validation.py

Track B: Model Diagnostics (Week 6)
├─ customer_base_audit/validation/diagnostics.py
└─ tests/test_diagnostics.py

Track C: Documentation (ongoing)
└─ docs/model_validation_guide.md

# Merge point: End of Week 6
# Enables: Lens 4-5 and production utilities (Phase 5)
```

**Phase 5 (Weeks 7-8): Lens 4-5 and CLI**
```
Track A: Lens 4 (Week 7)
├─ customer_base_audit/analyses/lens4.py
└─ tests/test_lens4.py

Track B: Lens 5 + CLI (Week 7-8)
├─ customer_base_audit/analyses/lens5.py
├─ customer_base_audit/cli.py (enhancements)
└─ tests/test_lens5.py, tests/integration/test_batch_cli.py

Track C: Documentation (ongoing)
└─ docs/user_guide.md (complete Lenses 4-5)

# Merge point: End of Week 8
# Enables: Drift detection (Phase 6)
```

**Phase 6 (Week 9): Drift Detection**
```
Track A: Drift Detection (Week 9)
├─ customer_base_audit/monitoring/drift.py
├─ customer_base_audit/monitoring/exports.py
└─ tests/test_drift_detection.py, tests/integration/test_drift_detection.py

Track C: Documentation (ongoing)
└─ docs/monitoring_guide.md

# Single track (fairly self-contained)
```

**Phase 7 (Week 10): Documentation Finalization**
```
Track A: Example Notebooks (Week 10)
├─ examples/01_texas_clv_walkthrough.ipynb
├─ examples/02_custom_cohorts.ipynb
├─ examples/03_model_comparison.ipynb
└─ examples/04_monitoring_drift.ipynb

Track B: API Reference + README (Week 10)
├─ docs/api_reference.md
└─ README.md (updates)

Track C: Review and Polish (Week 10)
└─ Cross-check all docs, fix broken links, etc.

# Highly parallelizable - multiple contributors can own different notebooks/docs
```

### Recommended Track Assignments

**Track A: Core Analytics** (Developer with statistics/CLV domain knowledge)
- RFM calculations
- Lens 1, 2, 4 implementations
- Validation framework
- Model diagnostics

**Track B: Machine Learning Models** (Developer with PyMC/Bayesian modeling experience)
- BG/NBD model wrapper
- Gamma-Gamma model wrapper
- CLV calculator
- Drift detection

**Track C: Developer Experience** (Developer with documentation/DevOps focus)
- Documentation (ongoing throughout all phases)
- Example notebooks
- Integration tests
- CLI enhancements

### Merge Strategy

**Frequent integration (recommended):**
```bash
# Merge tracks back to feature branch every 2-3 days
cd /Users/robertwelborn/PycharmProjects/AutoCLV
git checkout feature/tx-clv-synthetic

# Merge Track A
git merge --no-ff feature/track-a-rfm-lenses -m "feat: merge Track A (RFM + Lens 1)"

# Merge Track B
git merge --no-ff feature/track-b-clv-models -m "feat: merge Track B (Cohort infrastructure)"

# Resolve conflicts if any
# Run full test suite
make test

# Push integrated changes
git push origin feature/tx-clv-synthetic
```

**Conflict resolution:**
- Tracks are designed to minimize conflicts (separate modules)
- Most conflicts will be in test files or imports
- Resolve toward main feature branch (`feature/tx-clv-synthetic`)

### Testing in Parallel Worktrees

**Run tests independently in each worktree:**
```bash
# Terminal 1: Track A
cd /Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a
make test  # Run only RFM + Lens 1 tests

# Terminal 2: Track B
cd /Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-b
make test  # Run only cohort tests

# Terminal 3: Main worktree (integration)
cd /Users/robertwelborn/PycharmProjects/AutoCLV
make test  # Run full suite after merges
```

**Continuous Integration:**
- Each track branch should have CI enabled
- Full integration tests run on feature branch after merges
- Prevents breaking main with incomplete track work

### Example: Phase 1 Parallel Workflow

**Day 1-2: Setup and Track A Start**
```bash
# One-time setup: Create .worktrees directory
cd /Users/robertwelborn/PycharmProjects/AutoCLV
mkdir -p .worktrees
echo ".worktrees/" >> .gitignore

# Developer A: Create Track A worktree (creates new branch from current)
git worktree add -b feature/track-a-rfm-lenses .worktrees/track-a
cd .worktrees/track-a

# Implement RFM module
# Write tests
git add customer_base_audit/foundation/rfm.py tests/test_rfm.py
git commit -m "feat(rfm): implement RFM calculation utilities"
git push -u origin feature/track-a-rfm-lenses
```

**Day 1-2: Track B Start (simultaneous)**
```bash
# Developer B: Create Track B worktree (creates new branch from current)
cd /Users/robertwelborn/PycharmProjects/AutoCLV
git worktree add -b feature/track-b-cohorts .worktrees/track-b
cd .worktrees/track-b

# Implement cohort infrastructure
# Write tests
git add customer_base_audit/foundation/cohorts.py tests/test_cohorts.py
git commit -m "feat(cohorts): implement cohort assignment utilities"
git push -u origin feature/track-b-cohorts
```

**Day 3: Merge and Integration**
```bash
# Main developer: Merge both tracks
cd /Users/robertwelborn/PycharmProjects/AutoCLV  # Back to main worktree
git checkout feature/tx-clv-synthetic
git merge feature/track-a-rfm-lenses
git merge feature/track-b-cohorts

# Run integration tests
make test

# Push integrated work
git push origin feature/tx-clv-synthetic
```

**Day 4-7: Next iteration**
```bash
# Both developers rebase their tracks on updated feature branch
cd /Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a
git fetch origin
git rebase origin/feature/tx-clv-synthetic

cd ../track-b  # Relative path between worktrees
git fetch origin
git rebase origin/feature/tx-clv-synthetic

# Continue with Lens 1 (Track A) and Lens 3 prep (Track B)
```

### Risks and Mitigations

**Risk: Merge conflicts from parallel development**
- **Mitigation**: Design phases with minimal module overlap
- **Mitigation**: Frequent small merges (every 2-3 days) rather than large merges
- **Mitigation**: Clear module ownership per track

**Risk: Test failures after integration**
- **Mitigation**: Each track runs its own tests before pushing
- **Mitigation**: Integration tests run on feature branch after merges
- **Mitigation**: Pause new development if integration tests fail

**Risk: Dependency deadlocks (Track A needs Track B, Track B needs Track A)**
- **Mitigation**: Phases explicitly designed with clear dependencies
- **Mitigation**: Stub/mock interfaces if needed to unblock
- **Mitigation**: Track C (documentation) is never blocking

**Risk: Worktree management overhead**
- **Mitigation**: Use `.worktrees/` directory to contain all parallel worktrees
- **Mitigation**: Use descriptive worktree names (`track-a`, `track-b`, `track-c`)
- **Mitigation**: Add `.worktrees/` to `.gitignore` to prevent accidental commits
- **Mitigation**: Clean up completed worktrees promptly with `git worktree remove`
- **Mitigation**: Keep PycharmProjects folder clean by containing worktrees within AutoCLV

---

## Phase 1: Core RFM and Lens 1 Foundation (Critical Priority - Week 1-2)

### Overview
Implement RFM (Recency-Frequency-Monetary) calculation utilities and Lens 1 (Single Period Analysis), which are foundational to all subsequent analyses and models.

### Parallel Work Breakdown

**Track A (Week 1)**: RFM + Lens 1
- `customer_base_audit/foundation/rfm.py`
- `customer_base_audit/analyses/lens1.py`
- `tests/test_rfm.py`, `tests/test_lens1.py`

**Track B (Week 1)**: Cohort Infrastructure (runs in parallel with Track A)
- `customer_base_audit/foundation/cohorts.py`
- `tests/test_cohorts.py`

**Track C (Week 1-2)**: Documentation skeleton (runs in parallel, ongoing)
- `docs/user_guide.md` (initial structure)

**Dependencies**: None - all tracks can start immediately

**Merge Point**: End of Week 1 - enables Lens 2 and Lens 3 in Phase 2

### Changes Required

#### 1. RFM Calculation Module
**File**: `customer_base_audit/foundation/rfm.py` (new)
**Purpose**: Calculate RFM metrics from PeriodAggregation data

**Implementation**:

```python
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime
from typing import List
from customer_base_audit.foundation.data_mart import PeriodAggregation

@dataclass(frozen=True)
class RFMMetrics:
    """RFM metrics for a single customer."""
    customer_id: str
    recency_days: int  # Days since last purchase
    frequency: int  # Total number of purchases in observation period
    monetary: Decimal  # Average transaction value
    observation_start: datetime
    observation_end: datetime
    total_spend: Decimal  # Total spend in observation period

def calculate_rfm(
    period_aggregations: List[PeriodAggregation],
    observation_end: datetime
) -> List[RFMMetrics]:
    """
    Calculate RFM metrics from period aggregations.

    Args:
        period_aggregations: List of period-level customer aggregations
        observation_end: End date of observation period

    Returns:
        List of RFMMetrics, one per customer
    """
    # Group by customer_id
    # Calculate recency as days from last period_start to observation_end
    # Sum total_orders for frequency
    # Calculate monetary as total_spend / frequency
    pass

def calculate_rfm_scores(
    rfm_metrics: List[RFMMetrics],
    recency_bins: int = 5,
    frequency_bins: int = 5,
    monetary_bins: int = 5
) -> List[dict]:
    """
    Score RFM metrics into quintiles (1-5).

    Returns dictionaries with customer_id, r_score, f_score, m_score, rfm_score.
    """
    pass
```

**Validation Requirements**:
- Non-negative recency, frequency, monetary values
- Frequency > 0 for all customers with transactions
- Monetary = total_spend / frequency
- Edge case: Handle customers with single transaction (frequency = 1)

#### 2. Lens 1 Analysis Module
**File**: `customer_base_audit/analyses/lens1.py`
**Purpose**: Single-period customer analysis

**Implementation**:

```python
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Dict
from customer_base_audit.foundation.rfm import RFMMetrics

@dataclass(frozen=True)
class Lens1Metrics:
    """Lens 1: Single period analysis results."""
    total_customers: int
    one_time_buyers: int
    one_time_buyer_pct: Decimal
    total_revenue: Decimal
    top_10pct_revenue_contribution: Decimal  # % revenue from top 10% customers
    top_20pct_revenue_contribution: Decimal
    avg_orders_per_customer: Decimal
    median_customer_value: Decimal
    rfm_distribution: Dict[str, int]  # Distribution across RFM segments

def analyze_single_period(rfm_metrics: List[RFMMetrics]) -> Lens1Metrics:
    """
    Perform Lens 1 analysis on a single period.

    Key analyses:
    - Customer count and one-time buyer percentage
    - Revenue concentration (Lorenz curve / decile analysis)
    - RFM segmentation
    """
    pass

def calculate_revenue_concentration(
    rfm_metrics: List[RFMMetrics],
    percentiles: List[int] = [10, 20, 50]
) -> Dict[int, Decimal]:
    """
    Calculate what % of revenue comes from top N% of customers.

    Example: {10: Decimal('45.2'), 20: Decimal('62.8')}
    means top 10% of customers drive 45.2% of revenue.
    """
    pass
```

#### 3. Unit Tests
**Files**: `tests/test_rfm.py`, `tests/test_lens1.py` (new)

**Test Coverage**:
- RFM calculation correctness with known inputs
- Edge cases: single transaction, multiple periods per customer
- Lens 1 metrics match manual calculations
- Revenue concentration calculations
- One-time buyer percentage accuracy

### Success Criteria

#### Automated Verification:
- [ ] All unit tests pass: `make test`
- [ ] Type checking passes: `mypy customer_base_audit/`
- [ ] Linting passes: `make lint`
- [ ] RFM calculations validated against Texas CLV synthetic data
- [ ] Lens 1 metrics computed without errors for 1,000-customer dataset

#### Manual Verification:
- [ ] RFM distributions look reasonable (not all customers in one segment)
- [ ] Revenue concentration aligns with expected Pareto principle (~80/20)
- [ ] One-time buyer percentage matches manual count from synthetic data
- [ ] Spot-check 5 customers: RFM values match manual calculation

**Implementation Note**: After completing automated verification, pause for manual confirmation before proceeding to Phase 2.

---

## Phase 2: Lens 2-3 and Cohort Infrastructure (Critical Priority - Week 2-3)

### Overview
Implement period-to-period comparison (Lens 2) and cohort evolution tracking (Lens 3), which are essential for understanding customer behavior dynamics and model training.

### Parallel Work Breakdown

**Track A (Week 2)**: Lens 2 (depends on Phase 1 RFM + Lens 1)
- `customer_base_audit/analyses/lens2.py`
- `tests/test_lens2.py`

**Track B (Week 2)**: Lens 3 (depends on Phase 1 RFM + Cohorts)
- `customer_base_audit/analyses/lens3.py`
- `tests/test_lens3.py`

**Track C (Week 2)**: Documentation updates (runs in parallel)
- `docs/user_guide.md` (add Lenses 1-3 usage examples)

**Dependencies**: Phase 1 must be complete (RFM, Lens 1, Cohorts)

**Merge Point**: End of Week 2 - enables CLV model development in Phase 3

### Changes Required

#### 1. Lens 2 Analysis Module
**File**: `customer_base_audit/analyses/lens2.py`
**Purpose**: Period-to-period comparison and customer migration

**Implementation**:

```python
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Set
from customer_base_audit.analyses.lens1 import Lens1Metrics

@dataclass(frozen=True)
class CustomerMigration:
    """Customer movement between two periods."""
    retained: Set[str]  # Active in both periods
    churned: Set[str]  # Active in period 1, not in period 2
    new: Set[str]  # Not in period 1, active in period 2
    reactivated: Set[str]  # Not in period 1, but active before and in period 2

@dataclass(frozen=True)
class Lens2Metrics:
    """Lens 2: Period-to-period comparison results."""
    period1_metrics: Lens1Metrics
    period2_metrics: Lens1Metrics
    migration: CustomerMigration
    retention_rate: Decimal
    churn_rate: Decimal
    reactivation_rate: Decimal
    customer_count_change: int
    revenue_change_pct: Decimal
    avg_order_value_change_pct: Decimal

def analyze_period_comparison(
    period1_rfm: List[RFMMetrics],
    period2_rfm: List[RFMMetrics],
    all_customer_history: List[str]  # All customer IDs ever seen
) -> Lens2Metrics:
    """
    Compare two adjacent periods to identify customer migration patterns.

    Key analyses:
    - Customer migration matrix
    - Retention and churn rates
    - Metric deltas (revenue, AOV, frequency)
    """
    pass
```

#### 2. Lens 3 Analysis Module
**File**: `customer_base_audit/analyses/lens3.py`
**Purpose**: Single cohort evolution over time

**Implementation**:

```python
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Dict
from datetime import datetime

@dataclass(frozen=True)
class CohortPeriodMetrics:
    """Metrics for a cohort in a specific period after acquisition."""
    period_number: int  # Periods since acquisition (0 = acquisition period)
    active_customers: int
    retention_rate: Decimal  # % of original cohort still active
    avg_orders_per_customer: Decimal
    avg_revenue_per_customer: Decimal
    total_revenue: Decimal

@dataclass(frozen=True)
class Lens3Metrics:
    """Lens 3: Single cohort evolution results."""
    cohort_name: str
    acquisition_date: datetime
    cohort_size: int  # Initial customer count
    periods: List[CohortPeriodMetrics]  # Ordered by period_number

def analyze_cohort_evolution(
    cohort_id: str,
    acquisition_date: datetime,
    period_aggregations: List[PeriodAggregation]
) -> Lens3Metrics:
    """
    Track how a single cohort's behavior evolves over time.

    Key analyses:
    - Retention curve
    - Revenue decay patterns
    - Purchase frequency evolution
    - Time to second purchase distribution
    """
    pass

def calculate_retention_curve(cohort_metrics: Lens3Metrics) -> Dict[int, Decimal]:
    """
    Extract retention rates by period number.

    Returns: {period_number: retention_rate}
    """
    pass
```

#### 3. Cohort Assignment Utilities
**File**: `customer_base_audit/foundation/cohorts.py` (new)
**Purpose**: Assign customers to cohorts based on acquisition date

**Implementation**:

```python
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict
from customer_base_audit.foundation.customer_contract import CustomerIdentifier

@dataclass(frozen=True)
class CohortDefinition:
    """Definition of a customer cohort."""
    cohort_id: str
    start_date: datetime
    end_date: datetime
    metadata: Dict[str, str]  # e.g., {"channel": "paid_search", "campaign": "Q1_2023"}

def assign_cohorts(
    customers: List[CustomerIdentifier],
    cohort_definitions: List[CohortDefinition]
) -> Dict[str, str]:
    """
    Assign customers to cohorts based on acquisition timestamp.

    Returns: {customer_id: cohort_id}
    """
    pass

def create_monthly_cohorts(
    customers: List[CustomerIdentifier],
    start_date: datetime,
    end_date: datetime
) -> List[CohortDefinition]:
    """
    Automatically create monthly acquisition cohorts.
    """
    pass
```

#### 4. Unit Tests
**Files**: `tests/test_lens2.py`, `tests/test_lens3.py`, `tests/test_cohorts.py` (new)

**Test Coverage**:
- Customer migration calculations (retained, churned, new, reactivated)
- Retention rate accuracy
- Cohort retention curve correctness
- Cohort assignment based on acquisition dates
- Edge cases: single-period cohorts, 100% retention, 100% churn

### Success Criteria

#### Automated Verification:
- [ ] All unit tests pass: `make test`
- [ ] Type checking passes: `mypy customer_base_audit/`
- [ ] Lens 2 customer migration matrix validated against known dataset
- [ ] Lens 3 retention curves calculated correctly for Texas CLV cohorts
- [ ] Cohort assignments match acquisition dates in synthetic data

#### Manual Verification:
- [ ] Retention curves show expected decay pattern (decreasing over time)
- [ ] Migration matrix totals reconcile (period1 customers = retained + churned)
- [ ] Spot-check 3 cohorts: retention rates match manual calculation
- [ ] Cohort sizes sum to total customer count
- [ ] Period-to-period changes align with synthetic data scenario (promo spikes, launches)

**Implementation Note**: After automated verification passes, pause for manual testing before Phase 3.

---

## Phase 3: BG/NBD and Gamma-Gamma Models (High Priority - Week 3-5)

### Overview
Implement probabilistic CLV models using PyMC-Marketing library. These models predict future customer behavior (purchase frequency and transaction value) to calculate lifetime value.

### Parallel Work Breakdown

**Track A (Week 3-4)**: Model Prep + BG/NBD
- `customer_base_audit/models/model_prep.py`
- `customer_base_audit/models/bg_nbd.py`
- `tests/test_model_prep.py`, `tests/test_bg_nbd.py`

**Track B (Week 4-5)**: Gamma-Gamma + CLV Calculator (can start Week 4 after model_prep is ready)
- `customer_base_audit/models/gamma_gamma.py`
- `customer_base_audit/models/clv_calculator.py`
- `tests/test_gamma_gamma.py`, `tests/test_clv_calculator.py`

**Track C (Week 5)**: Integration Tests (depends on Tracks A+B)
- `tests/integration/test_clv_pipeline.py`

**Dependencies**: Phase 2 must be complete (Lenses 1-3, RFM, Cohorts)

**Merge Point**: End of Week 5 - enables validation framework in Phase 4

**Parallelization Note**: Track B can start implementing Gamma-Gamma in parallel once Track A completes `model_prep.py` (Week 4). Track C waits for both.

### Changes Required

#### 1. Model Input Preparation
**File**: `customer_base_audit/models/model_prep.py` (new)
**Purpose**: Transform CustomerDataMart into BG/NBD and Gamma-Gamma inputs

**Implementation**:

```python
from dataclasses import dataclass
from decimal import Decimal
from typing import List
import pandas as pd

@dataclass(frozen=True)
class BGNBDInput:
    """Input for BG/NBD model."""
    customer_id: str
    frequency: int  # Repeat purchases (total - 1)
    recency: float  # Time of last purchase in observation period
    T: float  # Total observation period length

@dataclass(frozen=True)
class GammaGammaInput:
    """Input for Gamma-Gamma model."""
    customer_id: str
    frequency: int  # Must be > 1 (excludes one-time buyers)
    monetary_value: Decimal  # Average transaction value

def prepare_bg_nbd_inputs(
    period_aggregations: List[PeriodAggregation],
    observation_start: datetime,
    observation_end: datetime
) -> pd.DataFrame:
    """
    Convert period aggregations to BG/NBD input format.

    Calculations:
    - frequency = total_orders - 1 (repeat purchases only)
    - recency = time from observation_start to last purchase
    - T = time from observation_start to observation_end
    """
    pass

def prepare_gamma_gamma_inputs(
    period_aggregations: List[PeriodAggregation],
    min_frequency: int = 2
) -> pd.DataFrame:
    """
    Convert period aggregations to Gamma-Gamma input format.

    Filters:
    - Only customers with frequency >= min_frequency
    - Excludes one-time buyers (can't estimate average value)
    """
    pass
```

#### 2. BG/NBD Model Wrapper
**File**: `customer_base_audit/models/bg_nbd.py` (new)
**Purpose**: Wrapper around PyMC-Marketing BG/NBD model

**Dependencies**: Add to requirements.txt:
```
pymc-marketing>=0.3.0
pymc>=5.10.0
arviz>=0.16.0
```

**Implementation**:

```python
from dataclasses import dataclass
from typing import Optional
import pandas as pd
from pymc_marketing.clv import BetaGeoModel

@dataclass
class BGNBDConfig:
    """Configuration for BG/NBD model training."""
    method: str = "map"  # 'map' for fast, 'mcmc' for accuracy
    chains: int = 4
    draws: int = 2000
    tune: int = 1000
    random_seed: int = 42

class BGNBDModelWrapper:
    """Wrapper for BG/NBD model training and prediction."""

    def __init__(self, config: BGNBDConfig = BGNBDConfig()):
        self.config = config
        self.model: Optional[BetaGeoModel] = None

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit BG/NBD model to customer transaction data.

        Args:
            data: DataFrame with columns [customer_id, frequency, recency, T]
        """
        # Validate inputs
        # Create BetaGeoModel instance
        # Fit using MAP or MCMC
        pass

    def predict_purchases(
        self,
        data: pd.DataFrame,
        time_periods: float
    ) -> pd.DataFrame:
        """
        Predict expected number of purchases in next time_periods.

        Returns: DataFrame with customer_id and predicted_purchases
        """
        pass

    def calculate_probability_alive(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate P(customer is still active).

        Returns: DataFrame with customer_id and prob_alive
        """
        pass
```

#### 3. Gamma-Gamma Model Wrapper
**File**: `customer_base_audit/models/gamma_gamma.py` (new)
**Purpose**: Wrapper around PyMC-Marketing Gamma-Gamma model

**Implementation**:

```python
from dataclasses import dataclass
from typing import Optional
import pandas as pd
from pymc_marketing.clv import GammaGammaModel

@dataclass
class GammaGammaConfig:
    """Configuration for Gamma-Gamma model training."""
    method: str = "map"
    chains: int = 4
    draws: int = 2000
    tune: int = 1000
    random_seed: int = 42

class GammaGammaModelWrapper:
    """Wrapper for Gamma-Gamma monetary value model."""

    def __init__(self, config: GammaGammaConfig = GammaGammaConfig()):
        self.config = config
        self.model: Optional[GammaGammaModel] = None

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit Gamma-Gamma model to customer spending data.

        Args:
            data: DataFrame with columns [customer_id, frequency, monetary_value]
                  Note: frequency must be >= 2 (exclude one-time buyers)
        """
        pass

    def predict_spend(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict expected average transaction value per customer.

        Returns: DataFrame with customer_id and predicted_monetary_value
        """
        pass
```

#### 4. CLV Calculator
**File**: `customer_base_audit/models/clv_calculator.py` (new)
**Purpose**: Combine BG/NBD + Gamma-Gamma for CLV calculation

**Implementation**:

```python
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional
import pandas as pd
from customer_base_audit.models.bg_nbd import BGNBDModelWrapper
from customer_base_audit.models.gamma_gamma import GammaGammaModelWrapper

@dataclass(frozen=True)
class CLVScore:
    """CLV prediction for a single customer."""
    customer_id: str
    predicted_purchases: Decimal
    predicted_avg_value: Decimal
    clv: Decimal
    prob_alive: Decimal
    confidence_interval_low: Optional[Decimal] = None
    confidence_interval_high: Optional[Decimal] = None

class CLVCalculator:
    """Calculate CLV by combining BG/NBD and Gamma-Gamma models."""

    def __init__(
        self,
        bg_nbd_model: BGNBDModelWrapper,
        gamma_gamma_model: GammaGammaModelWrapper,
        time_horizon_months: int = 12,
        discount_rate: Decimal = Decimal("0.10"),
        profit_margin: Decimal = Decimal("0.30")
    ):
        self.bg_nbd = bg_nbd_model
        self.gamma_gamma = gamma_gamma_model
        self.time_horizon = time_horizon_months
        self.discount_rate = discount_rate
        self.profit_margin = profit_margin

    def calculate_clv(
        self,
        bg_nbd_data: pd.DataFrame,
        gamma_gamma_data: pd.DataFrame,
        include_confidence_intervals: bool = False
    ) -> pd.DataFrame:
        """
        Calculate CLV for all customers.

        Formula:
        CLV = (Predicted Purchases) × (Predicted Avg Value) × Profit Margin × Discount Factor

        For one-time buyers (not in gamma_gamma_data):
        CLV = 0 or use fallback heuristic

        Returns: DataFrame with customer_id and CLV metrics
        """
        pass
```

#### 5. Unit and Integration Tests
**Files**: `tests/test_model_prep.py`, `tests/test_bg_nbd.py`, `tests/test_gamma_gamma.py`, `tests/test_clv_calculator.py` (new)

**Test Coverage**:
- Model input preparation correctness
- BG/NBD model fitting and prediction (using small synthetic dataset)
- Gamma-Gamma model fitting and prediction
- CLV calculation formula validation
- Edge cases: one-time buyers, very recent customers, high-frequency customers

### Success Criteria

#### Automated Verification:
- [ ] All unit tests pass: `make test`
- [ ] Type checking passes: `mypy customer_base_audit/`
- [ ] BG/NBD model trains without errors on Texas CLV data
- [ ] Gamma-Gamma model trains without errors (excluding one-time buyers)
- [ ] CLV calculations complete for all customers in <5 minutes (1,000 customers)
- [ ] Model convergence checks pass (R-hat < 1.1 for MCMC, if used)

#### Manual Verification:
- [ ] CLV scores have reasonable distribution (not all identical or extreme outliers)
- [ ] High-frequency customers have higher CLV than low-frequency
- [ ] P(alive) values are between 0 and 1
- [ ] Spot-check 5 customers: CLV formula calculation matches manual math
- [ ] Comparison: MAP vs MCMC produces similar results (within 15% MAPE)
- [ ] Model diagnostics: trace plots show convergence (visual inspection if MCMC used)

**Implementation Note**: This phase has higher computational requirements. Pause for validation before Phase 4.

---

## Phase 4: Model Validation Framework (High Priority - Week 5-6)

### Overview
Implement train/test splitting, performance metrics calculation, and validation framework to ensure model quality meets production standards (MAPE < 20%, ARPE < 10%).

### Parallel Work Breakdown

**Track A (Week 5-6)**: Validation Framework
- `customer_base_audit/validation/validation.py`
- `tests/test_validation.py`

**Track B (Week 6)**: Model Diagnostics (can run in parallel with Track A)
- `customer_base_audit/validation/diagnostics.py`
- `tests/test_diagnostics.py`

**Track C (Week 6)**: Documentation
- `docs/model_validation_guide.md`

**Dependencies**: Phase 3 must be complete (BG/NBD, Gamma-Gamma, CLV Calculator)

**Merge Point**: End of Week 6 - enables Lens 4-5 and production utilities in Phase 5

**Parallelization Note**: Tracks A and B are largely independent. Track B can start immediately if familiar with PyMC diagnostics.

### Changes Required

#### 1. Validation Framework
**File**: `customer_base_audit/validation/validation.py` (new)
**Purpose**: Train/test splitting and performance metric calculation

**Implementation**:

```python
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Tuple
import pandas as pd
from datetime import datetime, timedelta

@dataclass(frozen=True)
class ValidationMetrics:
    """Model validation performance metrics."""
    mae: Decimal  # Mean Absolute Error
    mape: Decimal  # Mean Absolute Percentage Error
    rmse: Decimal  # Root Mean Squared Error
    arpe: Decimal  # Aggregate Revenue Percent Error
    r_squared: Decimal
    sample_size: int

def temporal_train_test_split(
    period_aggregations: List[PeriodAggregation],
    train_end_date: datetime,
    test_end_date: datetime
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally for time-series validation.

    Train: All data up to train_end_date
    Test: Data from train_end_date to test_end_date (for ground truth)

    Returns: (train_df, test_df)
    """
    pass

def calculate_clv_metrics(
    actual: pd.Series,
    predicted: pd.Series
) -> ValidationMetrics:
    """
    Calculate validation metrics comparing actual vs predicted CLV.

    Metrics:
    - MAE: mean(|actual - predicted|)
    - MAPE: mean(|actual - predicted| / actual) * 100
    - RMSE: sqrt(mean((actual - predicted)^2))
    - ARPE: |sum(actual) - sum(predicted)| / sum(actual) * 100
    - R²: 1 - (SS_res / SS_tot)
    """
    pass

def cross_validate_clv(
    period_aggregations: List[PeriodAggregation],
    n_folds: int = 5,
    time_increment_months: int = 3
) -> List[ValidationMetrics]:
    """
    Time-series cross-validation with expanding window.

    Example with 3-month increments:
    - Fold 1: Train on months 0-12, test on 13-15
    - Fold 2: Train on months 0-15, test on 16-18
    - etc.

    Returns: List of ValidationMetrics, one per fold
    """
    pass
```

#### 2. Model Diagnostics
**File**: `customer_base_audit/validation/diagnostics.py` (new)
**Purpose**: Model convergence checks and diagnostics

**Implementation**:

```python
from typing import Dict, Optional
import arviz as az
import pandas as pd

def check_mcmc_convergence(
    model_trace: az.InferenceData,
    r_hat_threshold: float = 1.1
) -> Dict[str, bool]:
    """
    Check MCMC convergence using R-hat statistic.

    Returns: {parameter_name: is_converged}
    """
    pass

def posterior_predictive_check(
    model,
    observed_data: pd.DataFrame,
    n_samples: int = 1000
) -> pd.DataFrame:
    """
    Generate posterior predictive samples and compare to observed data.

    Returns: DataFrame with observed vs predicted distribution stats
    """
    pass

def plot_trace_diagnostics(model_trace: az.InferenceData, output_path: str) -> None:
    """
    Generate trace plots for visual convergence inspection.

    Saves plots to output_path.
    """
    pass
```

#### 3. Integration Test: End-to-End Validation
**File**: `tests/integration/test_clv_pipeline.py` (new)
**Purpose**: Validate complete pipeline from transactions to CLV scores

**Test Scenario**:
```python
def test_end_to_end_clv_pipeline():
    """
    Test complete CLV pipeline:
    1. Load Texas CLV synthetic data
    2. Build CustomerDataMart
    3. Calculate RFM metrics
    4. Prepare model inputs
    5. Train BG/NBD and Gamma-Gamma models
    6. Calculate CLV scores
    7. Validate accuracy metrics
    """
    # Load synthetic data
    # Run pipeline
    # Assert MAPE < 25% (allowing some slack for synthetic data)
    # Assert ARPE < 15%
    # Assert all customers have CLV scores
    pass
```

#### 4. Documentation
**File**: `docs/model_validation_guide.md` (new)
**Purpose**: Document validation methodology and interpretation

**Contents**:
- Explanation of validation metrics (MAE, MAPE, RMSE, ARPE, R²)
- Interpretation guidelines (what is "good" performance)
- Cross-validation methodology
- Model comparison framework
- How to diagnose poor model performance

### Success Criteria

#### Automated Verification:
- [ ] All validation tests pass: `make test`
- [ ] Integration test completes end-to-end pipeline successfully
- [ ] Cross-validation produces metrics for all folds without errors
- [ ] MCMC convergence checks work correctly (if MCMC used)
- [ ] Temporal train/test split validates correct data partitioning

#### Manual Verification:
- [ ] On Texas CLV data: Achieve MAPE < 25% (target: <20% with real data)
- [ ] On Texas CLV data: Achieve ARPE < 15% (target: <10% with real data)
- [ ] Cross-validation shows stable metrics across folds (CV < 30%)
- [ ] Posterior predictive checks show reasonable fit (visual inspection)
- [ ] Model comparison: BG/NBD significantly better than baseline (mean revenue per customer)
- [ ] Documentation is clear and actionable for new analysts

**Implementation Note**: Achieving target metrics may require hyperparameter tuning. Pause for review if MAPE > 30%.

---

## Phase 5: Lens 4-5 and Production Utilities (Medium Priority - Week 7-8)

### Overview
Complete the Five Lenses framework with multi-cohort comparison (Lens 4) and overall health dashboard (Lens 5). Add production utilities for batch processing and reporting.

### Parallel Work Breakdown

**Track A (Week 7)**: Lens 4
- `customer_base_audit/analyses/lens4.py`
- `tests/test_lens4.py`

**Track B (Week 7-8)**: Lens 5 + CLI Enhancements (Lens 5 depends on Lens 4, but can overlap)
- `customer_base_audit/analyses/lens5.py`
- `customer_base_audit/cli.py` (batch scoring, report generation)
- `tests/test_lens5.py`, `tests/integration/test_batch_cli.py`

**Track C (Week 7-8)**: Documentation
- `docs/user_guide.md` (complete Lenses 4-5 sections)

**Dependencies**: Phase 4 must be complete (Validation framework, model diagnostics)

**Merge Point**: End of Week 8 - enables drift detection in Phase 6

**Parallelization Note**: Track A (Lens 4) should complete first few days of Week 7 to unblock Track B (Lens 5). CLI work in Track B can proceed in parallel.

### Changes Required

#### 1. Lens 4 Analysis Module
**File**: `customer_base_audit/analyses/lens4.py`
**Purpose**: Multi-cohort comparison

**Implementation**:

```python
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Dict
from customer_base_audit.analyses.lens3 import Lens3Metrics

@dataclass(frozen=True)
class Lens4Metrics:
    """Lens 4: Multi-cohort comparison results."""
    cohorts: List[Lens3Metrics]
    cohort_quality_trends: Dict[str, Decimal]  # Cohort quality over time
    best_performing_cohort: str
    worst_performing_cohort: str
    avg_retention_by_period: Dict[int, Decimal]  # Period number -> avg retention

def analyze_multi_cohort_comparison(
    cohort_metrics: List[Lens3Metrics]
) -> Lens4Metrics:
    """
    Compare performance across multiple acquisition cohorts.

    Key analyses:
    - Cohort revenue curves comparison
    - Retention rate differences across cohorts
    - Customer quality trends over time
    """
    pass

def identify_cohort_quality_trend(
    cohort_metrics: List[Lens3Metrics]
) -> str:
    """
    Determine if cohort quality is improving, stable, or declining.

    Returns: "improving", "stable", or "declining"
    """
    pass
```

#### 2. Lens 5 Analysis Module
**File**: `customer_base_audit/analyses/lens5.py`
**Purpose**: Overall customer base health

**Implementation**:

```python
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Dict
from customer_base_audit.analyses.lens1 import Lens1Metrics
from customer_base_audit.analyses.lens4 import Lens4Metrics

@dataclass(frozen=True)
class Lens5Metrics:
    """Lens 5: Overall customer base health."""
    current_period_metrics: Lens1Metrics
    cohort_comparison: Lens4Metrics
    customer_base_composition: Dict[str, int]  # Cohort -> customer count
    revenue_by_cohort: Dict[str, Decimal]
    weighted_retention_rate: Decimal
    projected_annual_revenue: Decimal
    health_score: Decimal  # Composite 0-100 score

def analyze_overall_health(
    current_lens1: Lens1Metrics,
    lens4_cohorts: Lens4Metrics,
    clv_scores: List[CLVScore]
) -> Lens5Metrics:
    """
    Integrative view synthesizing Lenses 1-4 for holistic health.

    Key analyses:
    - Customer base composition by cohort
    - Revenue contribution by cohort and tenure
    - Overall retention trends
    - Projected future revenue based on CLV
    """
    pass

def calculate_health_score(
    lens5: Lens5Metrics
) -> Decimal:
    """
    Calculate composite health score (0-100).

    Factors:
    - Retention rate (40%)
    - Cohort quality trend (30%)
    - Revenue concentration (15%)
    - Growth momentum (15%)
    """
    pass
```

#### 3. Batch Processing CLI Enhancement
**File**: `customer_base_audit/cli.py` (extend existing)
**Purpose**: Add batch CLV scoring command

**Implementation**:

```python
@click.command()
@click.option('--input', required=True, help='Input transactions JSON file')
@click.option('--output', required=True, help='Output CLV scores CSV file')
@click.option('--model-method', default='map', help='BG/NBD method: map or mcmc')
@click.option('--time-horizon', default=12, help='CLV prediction horizon (months)')
def score_clv(input, output, model_method, time_horizon):
    """
    Batch CLV scoring: transactions → CLV scores.

    Pipeline:
    1. Load transactions from JSON
    2. Build CustomerDataMart
    3. Prepare model inputs
    4. Train BG/NBD and Gamma-Gamma
    5. Calculate CLV scores
    6. Export to CSV
    """
    pass

@click.command()
@click.option('--input', required=True, help='Input transactions JSON file')
@click.option('--output-dir', required=True, help='Output directory for reports')
@click.option('--period-granularity', default='MONTH', help='MONTH, QUARTER, or YEAR')
def generate_five_lenses_report(input, output_dir, period_granularity):
    """
    Generate complete Five Lenses audit report.

    Outputs:
    - lens1_metrics.json
    - lens2_metrics.json
    - lens3_cohort_*.json (one per cohort)
    - lens4_metrics.json
    - lens5_metrics.json
    - summary_report.md
    """
    pass
```

#### 4. Unit and Integration Tests
**Files**: `tests/test_lens4.py`, `tests/test_lens5.py`, `tests/integration/test_batch_cli.py` (new)

**Test Coverage**:
- Lens 4 cohort comparison calculations
- Lens 5 health score calculation
- CLI batch processing with Texas CLV data
- CLI report generation outputs all expected files

### Success Criteria

#### Automated Verification:
- [ ] All unit tests pass: `make test`
- [ ] CLI batch processing completes for 1,000-customer dataset
- [ ] Five Lenses report generates all output files
- [ ] Lens 4 cohort rankings are consistent with Lens 3 metrics
- [ ] Lens 5 health score is between 0 and 100

#### Manual Verification:
- [ ] Lens 4 identifies best/worst cohorts correctly based on retention and revenue
- [ ] Lens 5 health score aligns with qualitative assessment of customer base
- [ ] CLI output CSV has correct schema and CLV values
- [ ] Generated report markdown is readable and informative
- [ ] Spot-check 3 cohorts: Lens 4 comparisons match manual analysis

**Implementation Note**: This phase completes the Five Lenses framework. Pause for stakeholder review of reports.

---

## Phase 6: Drift Detection and Monitoring (Medium Priority - Week 9)

### Overview
Implement drift detection to monitor for changes in customer behavior or data quality that would degrade model accuracy. Essential for production deployment.

### Parallel Work Breakdown

**Track A (Week 9)**: Drift Detection Implementation
- `customer_base_audit/monitoring/drift.py`
- `customer_base_audit/monitoring/exports.py`
- `tests/test_drift_detection.py`, `tests/integration/test_drift_detection.py`

**Track C (Week 9)**: Monitoring Documentation
- `docs/monitoring_guide.md`

**Dependencies**: Phase 5 must be complete (Lenses 1-5, CLI)

**Merge Point**: End of Week 9 - enables documentation finalization in Phase 7

**Parallelization Note**: Fairly self-contained phase. Track A is primary work, Track C can run in parallel.

### Changes Required

#### 1. Drift Detection Module
**File**: `customer_base_audit/monitoring/drift.py` (new)
**Purpose**: Detect distribution shifts in features and predictions

**Implementation**:

```python
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List
import pandas as pd
from scipy import stats

@dataclass(frozen=True)
class DriftReport:
    """Drift detection results."""
    feature_name: str
    psi: Decimal  # Population Stability Index
    ks_statistic: Decimal  # Kolmogorov-Smirnov test statistic
    ks_pvalue: Decimal
    drift_detected: bool
    severity: str  # "none", "low", "moderate", "high"

def calculate_psi(
    baseline_dist: pd.Series,
    current_dist: pd.Series,
    bins: int = 10
) -> Decimal:
    """
    Calculate Population Stability Index (PSI).

    PSI < 0.1: No significant change
    0.1 <= PSI < 0.2: Moderate change
    PSI >= 0.2: Significant change (investigate)
    """
    pass

def detect_feature_drift(
    baseline_data: pd.DataFrame,
    current_data: pd.DataFrame,
    features: List[str],
    psi_threshold: Decimal = Decimal("0.1")
) -> List[DriftReport]:
    """
    Detect drift in RFM features and model inputs.

    Features to monitor:
    - Recency distribution
    - Frequency distribution
    - Monetary value distribution
    - CLV prediction distribution
    """
    pass

def detect_prediction_drift(
    baseline_predictions: pd.Series,
    current_predictions: pd.Series,
    psi_threshold: Decimal = Decimal("0.1")
) -> DriftReport:
    """
    Monitor for drift in CLV prediction distributions.
    """
    pass
```

#### 2. Monitoring Dashboard Data Export
**File**: `customer_base_audit/monitoring/exports.py` (new)
**Purpose**: Export monitoring metrics for external dashboards

**Implementation**:

```python
from dataclasses import dataclass
from decimal import Decimal
from typing import List
import json

@dataclass(frozen=True)
class MonitoringSnapshot:
    """Snapshot of model monitoring metrics."""
    timestamp: str
    model_version: str
    feature_drift: List[DriftReport]
    prediction_drift: DriftReport
    validation_metrics: ValidationMetrics
    alert_triggered: bool

def export_monitoring_snapshot(
    snapshot: MonitoringSnapshot,
    output_path: str
) -> None:
    """
    Export monitoring snapshot to JSON for ingestion by observability tools.
    """
    pass
```

#### 3. Integration Test
**File**: `tests/integration/test_drift_detection.py` (new)
**Purpose**: Validate drift detection on synthetic distribution shifts

**Test Scenario**:
```python
def test_drift_detection_sensitivity():
    """
    Generate baseline and shifted distributions.
    Verify PSI correctly identifies drift.
    """
    # Create baseline: normal distribution
    # Create shifted: mean + 20%
    # Assert drift_detected = True
    # Assert severity = "moderate" or "high"
    pass
```

### Success Criteria

#### Automated Verification:
- [ ] All drift detection tests pass: `make test`
- [ ] PSI calculation matches manual calculation for known distributions
- [ ] Drift correctly detected in simulated distribution shift (mean +20%)
- [ ] No drift detected when baseline = current (same distribution)

#### Manual Verification:
- [ ] Test on Texas CLV data: baseline vs. modified scenario shows expected drift
- [ ] PSI thresholds (0.1, 0.2) produce reasonable alerting sensitivity
- [ ] Monitoring snapshot JSON exports successfully
- [ ] Visual inspection: distribution plots for drifted features look shifted

**Implementation Note**: Drift detection enables ongoing model monitoring in production.

---

## Phase 7: Documentation and Examples (Medium Priority - Week 10)

### Overview
Comprehensive documentation, examples, and onboarding materials to enable enterprise analytics teams to adopt the CLV toolkit.

### Parallel Work Breakdown

**Track A (Week 10)**: Example Notebooks (highly parallelizable)
- `examples/01_texas_clv_walkthrough.ipynb`
- `examples/02_custom_cohorts.ipynb`
- `examples/03_model_comparison.ipynb`
- `examples/04_monitoring_drift.ipynb`

**Track B (Week 10)**: API Reference + README (runs in parallel with Track A)
- `docs/api_reference.md`
- `README.md` (comprehensive update with examples and badges)

**Track C (Week 10)**: Documentation Review and Polish (runs in parallel)
- Cross-check all documentation for accuracy
- Fix broken links, test code examples
- Ensure consistency across all docs

**Dependencies**: Phases 1-6 must be complete (all functionality implemented)

**Merge Point**: End of Week 10 - Project complete!

**Parallelization Note**: This phase is highly parallelizable. Each notebook can be owned by a different contributor. Track C performs final QA across all tracks.

### Changes Required

#### 1. User Guide
**File**: `docs/user_guide.md` (new)
**Purpose**: End-to-end guide for using AutoCLV

**Contents**:
- Installation and setup
- Data preparation requirements
- Running Five Lenses audit
- Training CLV models
- Interpreting results
- Troubleshooting common issues

#### 2. API Reference
**File**: `docs/api_reference.md` (new)
**Purpose**: Technical reference for all modules

**Contents**:
- RFM module API
- Lens 1-5 analysis functions
- Model wrappers (BG/NBD, Gamma-Gamma)
- CLV calculator
- Validation framework
- Drift detection

#### 3. Example Notebooks
**Directory**: `examples/` (new)

**Notebooks**:
- `01_texas_clv_walkthrough.ipynb`: Complete analysis of Texas CLV synthetic data
- `02_custom_cohorts.ipynb`: Defining and analyzing custom cohorts
- `03_model_comparison.ipynb`: Comparing BG/NBD vs. baseline methods
- `04_monitoring_drift.ipynb`: Setting up drift detection

#### 4. README Enhancement
**File**: `README.md` (update)
**Purpose**: Project overview with quickstart

**Updates**:
- Add quickstart example
- Link to documentation
- Add badges (tests passing, coverage, version)
- Include example output from Five Lenses

### Success Criteria

#### Automated Verification:
- [ ] All documentation links are valid (link checker)
- [ ] Example notebooks execute without errors: `pytest --nbmains examples/`
- [ ] Code examples in docs are syntactically correct

#### Manual Verification:
- [ ] New analyst can follow user guide and generate CLV scores in <2 hours
- [ ] API reference is comprehensive and accurate
- [ ] Example notebooks produce expected outputs
- [ ] README provides clear value proposition and usage overview
- [ ] Stakeholder review: documentation is clear and actionable

**Implementation Note**: Documentation quality directly impacts adoption. Allocate time for review and iteration.

---

## Testing Strategy

### Unit Tests
**Coverage Target**: >90% for all modules

**Key Test Areas**:
- RFM calculation correctness
- Lens 1-5 metric calculations
- Model input preparation edge cases
- CLV formula validation
- Drift detection sensitivity

**Test Data**: Small fixtures (10-50 customers) for fast execution

### Integration Tests
**Coverage**: End-to-end workflows

**Key Scenarios**:
- Complete CLV pipeline: transactions → CLV scores
- Five Lenses report generation
- Drift detection on distribution shifts
- CLI batch processing

**Test Data**: Texas CLV synthetic dataset (1,000 customers)

### Property-Based Tests (Recommended)
**Framework**: Hypothesis

**Test Properties**:
- RFM metrics sum to expected totals
- Retention rates always between 0 and 1
- CLV scores non-negative
- Period aggregations sum to order totals

### Performance Tests
**Benchmarks**:
- Process 100K transactions in <30 seconds (data mart build)
- Train BG/NBD on 10K customers in <5 minutes (MAP method)
- Generate Five Lenses report for 1,000 customers in <60 seconds

**Scalability Targets**:
- Support 1M customers with batch processing
- Memory usage <8GB for 100K customer CLV calculation

## Performance Considerations

### Computational Complexity
- **Data mart aggregation**: O(n) where n = transaction count
- **RFM calculation**: O(m) where m = customer count
- **BG/NBD training**: O(m) for MAP, O(m × iterations × chains) for MCMC
- **CLV calculation**: O(m)

### Memory Optimization
- Use pandas for vectorized operations (avoid Python loops)
- Stream large datasets in chunks if needed
- Cache intermediate results (data mart, RFM) to avoid recomputation

### Scaling Strategies
- **Horizontal**: Process cohorts in parallel (embarrassingly parallel)
- **Vertical**: Use multiprocessing for model training (PyMC supports)
- **Incremental**: Only reprocess new/changed customers (Phase 8+)

## Migration Notes

### Existing AutoCLV Users
**Breaking Changes**: None (all new functionality)

**Migration Path**:
1. Update dependencies: `pip install -r requirements.txt`
2. Existing data mart code unchanged
3. New analyses available via lens1-5 modules
4. Opt-in to CLV models as needed

### Data Requirements
**Minimum**:
- 1 year of transaction history
- At least 100 customers with 2+ transactions
- Customer acquisition dates

**Recommended**:
- 2-3 years of transaction history
- 1,000+ customers
- Cohort metadata (channel, campaign)

## References

**Original Research Document**: `thoughts/shared/research/2025-10-07-clv-implementation-plan.md`

**Feature Requests**:
- `docs/issues/feature_reusable_audit_components.md` - Lens implementations
- `docs/issues/feature_data_generation_testing.md` - Synthetic data (already complete)
- `docs/issues/feature_enablement_operationalization.md` - Documentation and training

**Books and Papers**:
- "The Customer-Base Audit" by Fader, Hardie, Ross (2022)
- "Counting Your Customers the Easy Way" - Fader, Hardie, Lee (2005)
- Bruce Hardie's CLV Papers: http://www.brucehardie.com/

**Implementation Libraries**:
- PyMC-Marketing: https://www.pymc-marketing.io/
- PyMC-Marketing BG/NBD: https://www.pymc-marketing.io/en/stable/notebooks/clv/bg_nbd.html
- PyMC-Marketing Gamma-Gamma: https://www.pymc-marketing.io/en/stable/notebooks/clv/gamma_gamma.html

## Quick Reference: Parallel Work Summary

This table summarizes parallelization opportunities across all phases for quick planning:

| Phase | Week | Track A | Track B | Track C | Max Parallelism |
|-------|------|---------|---------|---------|-----------------|
| **Phase 1** | 1-2 | RFM + Lens 1 | Cohort Infrastructure | Docs skeleton | **3 tracks** |
| **Phase 2** | 2-3 | Lens 2 | Lens 3 | Docs (Lenses 1-3) | **3 tracks** |
| **Phase 3** | 3-5 | Model Prep + BG/NBD | Gamma-Gamma + CLV Calc | Integration tests | **2-3 tracks** |
| **Phase 4** | 5-6 | Validation Framework | Model Diagnostics | Model docs | **3 tracks** |
| **Phase 5** | 7-8 | Lens 4 | Lens 5 + CLI | Docs (Lenses 4-5) | **3 tracks** |
| **Phase 6** | 9 | Drift Detection | — | Monitoring docs | **2 tracks** |
| **Phase 7** | 10 | Example Notebooks | API Ref + README | Doc review & polish | **3 tracks** |

**Key Insights:**
- **Maximum parallelism**: Phases 1, 2, 4, 5, 7 support 3 simultaneous tracks
- **Reduced parallelism**: Phase 3 has dependency (model_prep must complete first), Phase 6 is more self-contained
- **Track C (Documentation)**: Runs throughout all phases - can be ongoing work
- **Critical path**: Track A in most phases (RFM → Lens 2 → Model Prep → Validation → Lens 4 → Drift)
- **Opportunity for acceleration**: With 3 developers, 10-week plan could compress to ~7-8 weeks

**Recommended Staffing:**
- **1 developer**: Follow critical path (Track A), add Track B work sequentially (~10-12 weeks)
- **2 developers**: Developer 1 on Track A, Developer 2 on Track B, alternate Track C (~7-8 weeks)
- **3 developers**: Full parallelization across all tracks (~6-7 weeks with proper coordination)

**Risk Mitigation:**
- **Integration overhead**: Budget 0.5 days per week for merging tracks
- **Rework risk**: Frequent integration (every 2-3 days) prevents large conflicts
- **Communication overhead**: Daily standup recommended with 2+ developers

## Appendix: Priority Classification Rationale

### Critical Priority (Phases 1-2)
**Rationale**: Core functionality required for any CLV analysis. Without RFM and basic Lenses, no descriptive or predictive analytics possible.

**Items**:
- RFM calculation (foundation for all analyses)
- Lens 1 (single period analysis - most basic view)
- Lens 2 (period comparison - essential for change detection)
- Lens 3 (cohort evolution - required for model training)
- Cohort infrastructure (enables all cohort-based analyses)

### High Priority (Phases 3-4)
**Rationale**: Predictive CLV models are the primary value driver. Model validation ensures production quality.

**Items**:
- BG/NBD model (purchase frequency prediction)
- Gamma-Gamma model (monetary value prediction)
- CLV calculator (combines models for final scores)
- Validation framework (ensures MAPE < 20%, ARPE < 10%)
- Model diagnostics (convergence checks, PPCs)

### Medium Priority (Phases 5-7)
**Rationale**: Complete the Five Lenses framework, add production monitoring, and enable adoption through documentation.

**Items**:
- Lens 4 (multi-cohort comparison - completes framework)
- Lens 5 (overall health - integrative view)
- Batch processing CLI (production utility)
- Drift detection (production monitoring)
- Documentation and examples (adoption enabler)

### Low Priority (Future Phases)
**Not included in this plan - deferred to future iterations:**
- Real-time CLV scoring API
- Data warehouse integrations (Snowflake, BigQuery connectors)
- Reverse ETL activation
- Custom visualization dashboards
- Advanced cohort segmentation (behavioral, predictive)
- Multi-model comparison framework
- Automated hyperparameter tuning
- GDPR/CCPA compliance automation
