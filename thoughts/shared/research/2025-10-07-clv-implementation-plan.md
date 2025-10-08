---
date: 2025-10-07T16:37:59-05:00
researcher: Claude
git_commit: dc5d9bf4e80b834d00451ba4d841c473c6ba1a53
branch: feature/tx-clv-synthetic
repository: AutoCLV
topic: "Enterprise CLV Calculator Implementation Plan - Five Lenses Approach"
tags: [research, codebase, clv, customer-lifetime-value, five-lenses, customer-base-audit, bg-nbd, gamma-gamma, cohort-analysis]
status: complete
last_updated: 2025-10-07
last_updated_by: Claude
---

# Research: Enterprise CLV Calculator Implementation Plan - Five Lenses Approach

**Date**: 2025-10-07T16:37:59-05:00
**Researcher**: Claude
**Git Commit**: dc5d9bf4e80b834d00451ba4d841c473c6ba1a53
**Branch**: feature/tx-clv-synthetic
**Repository**: AutoCLV

## Research Question

How should we build a customer lifetime value calculator for enterprise clients using the Five Lenses approach from "The Customer-Base Audit" by Fader, Hardie, and Ross? What implementation plan will deliver high-quality, repeatable CLV scores?

## Summary

The AutoCLV codebase currently provides foundational infrastructure for customer analytics but does not yet implement CLV models. The project has:
- **Strong foundation**: Transaction aggregation, customer identity management, synthetic data generation
- **Clear architecture**: Placeholder files for the Five Lenses analyses (lens1.py through lens5.py)
- **Production-ready patterns**: Decimal arithmetic for financial calculations, comprehensive validation, cohort tracking

To build an enterprise-grade CLV calculator, the implementation should:
1. Implement the Five Lenses framework for customer-base audit (Chapters 3-7 from document.txt)
2. Add probabilistic CLV models (BG/NBD for frequency prediction, Gamma-Gamma for monetary value)
3. Build on existing data mart infrastructure for period and cohort aggregations
4. Ensure data quality with comprehensive validation and model monitoring

The Five Lenses approach provides a systematic framework for understanding customer behavior, while the BG/NBD and Gamma-Gamma models provide industry-standard predictive CLV calculations.

## Detailed Findings

### Current Codebase State

#### Foundation Infrastructure (Production-Ready)

**Customer Data Mart Builder** (`customer_base_audit/foundation/data_mart.py:92-105`)

The codebase has a mature data aggregation system that transforms raw transaction data into analytics-ready formats:

- **Order Aggregation** (`data_mart.py:107-217`): Groups line-level transactions into order summaries with total spend, margin, quantity, and distinct product counts
- **Period Aggregation** (`data_mart.py:219-268`): Rolls up orders into customer-period metrics at MONTH, QUARTER, or YEAR granularity
- **Period Normalization** (`data_mart.py:271-300`): Converts timestamps to standard period boundaries for consistent aggregation
- **Financial Precision**: Uses Decimal arithmetic with ROUND_HALF_UP quantization to 2 decimal places

This foundation directly supports the Five Lenses analyses which require customer × time aggregations.

**Customer Contract** (`customer_base_audit/foundation/customer_contract.py:17-163`)

The customer identity system provides:

- **Canonical customer records** with `customer_id`, `acquisition_ts`, `source_system`, and extensible `metadata`
- **Cross-system merging**: Deduplicates customer records, preserving earliest acquisition date and merging metadata
- **Validation**: Required field checks, timestamp range validation (1970 to present), type enforcement
- **Cohort support**: Acquisition timestamp enables cohort grouping, though cohort analysis not yet implemented

This supports cohort-based analysis required for Lenses 3-4.

**Synthetic Data Generation** (`customer_base_audit/synthetic/generator.py:70-269`)

The synthetic data toolkit generates realistic test datasets with:

- **Scenario configuration**: Promo spikes, launch ramps, churn rates, base purchase frequency
- **Cohort tracking**: Progressively adds customers based on acquisition dates, applies churn only to acquired customers
- **Transaction generation**: Respects acquisition dates, applies promotional and launch multipliers
- **Validation suite**: Non-negative amounts, order density, promo spike signal detection

The Texas CLV client (`texas_clv_client.py:49-106`) demonstrates a complete scenario with 1,000 customers across 4 cities with city-specific launch dates and promotions.

#### Placeholder Files for Five Lenses Analyses

The codebase has empty placeholder files ready for implementation:
- `customer_base_audit/analyses/lens1.py` - Single-period analysis
- `customer_base_audit/analyses/lens2.py` - Period-to-period comparison
- `customer_base_audit/analyses/lens3.py` - Single-cohort evolution
- `customer_base_audit/analyses/lens4.py` - Multi-cohort comparison
- `customer_base_audit/analyses/lens5.py` - Overall customer base health

These align with the Five Lenses framework structure from the book.

#### Tests and Quality Assurance

Comprehensive test coverage exists for:
- Customer contract validation and merging (`tests/test_customer_foundation.py:12-79`)
- Data mart order and period aggregation (`tests/test_customer_foundation.py:120-218`)
- Synthetic data generation with edge cases (`tests/test_synthetic_data.py`)

### The Five Lenses Framework (from document.txt)

The book "The Customer-Base Audit" by Peter Fader, Bruce Hardie, and Michael Ross presents a systematic approach to customer analytics through five distinct analytical lenses applied to the customer × time face of the data cube.

#### Lens 1: Vertical Slice (Single Period Analysis)

**Focus**: All customers active in a single time period (typically a quarter or year)

**Key Analyses** (Chapter 3):
- Customer value distribution: How different are your customers?
- Concentration metrics: How many one-time buyers? What percentage of customers drive 50% of revenue?
- Recency-Frequency-Monetary (RFM) segmentation

**Business Questions Answered**:
- How many customers does the firm really have?
- How do customers differ in their value?
- How many one-time buyers did we have last year?

**Example from Book**: American Airlines discovered that 87% of unique customers in a year flew only once, yet they represented over 50% of revenue.

#### Lens 2: Period vs Period (Change Analysis)

**Focus**: Two adjacent vertical slices to identify changes in buyer behavior

**Key Analyses** (Chapter 4):
- Period-on-period customer migration: retained, reactivated, new, lost
- Changes in average order value, purchase frequency
- Cohort composition shifts

**Business Questions Answered**:
- What changed since last period?
- Which customers returned vs. stayed vs. churned?
- What underlies period-on-period fluctuations in performance?

#### Lens 3: Horizontal Slice (Cohort Evolution)

**Focus**: How a single cohort's behavior evolves over time from first purchase

**Key Analyses** (Chapter 5):
- Retention curves: percentage of cohort active in each period after acquisition
- Revenue decay patterns
- Purchase frequency evolution
- Time to second purchase

**Business Questions Answered**:
- How does customer behavior evolve after acquisition?
- What is the typical lifecycle of a customer?
- When do most customers make their second purchase?

#### Lens 4: Multi-Cohort Comparison

**Focus**: Comparing performance across different acquisition cohorts

**Key Analyses** (Chapter 6):
- Cohort revenue curves comparison
- Retention rate differences across cohorts
- Customer quality trends over time

**Business Questions Answered**:
- Are recent cohorts better or worse than historical ones?
- Has customer quality improved over time?
- Which acquisition periods produced the best customers?

#### Lens 5: Overall Customer Base Health

**Focus**: Integrative view synthesizing Lenses 1-4 for holistic health assessment

**Key Analyses** (Chapter 7):
- Customer base composition by cohort
- Revenue contribution by cohort and tenure
- Overall retention trends
- Projected future revenue streams based on cohort performance

**Business Questions Answered**:
- How healthy is our customer base overall?
- Are we gaining or losing momentum?
- How realistic are our growth objectives given customer base composition?

#### Data Structure Requirements

The book specifies transaction data should include (`document.txt`, Table 2.1):
- `Order_ID`: Unique order identifier
- `Customer_ID`: Unique customer identifier
- `Date`: Transaction timestamp
- `SKU_ID`: Product identifier
- `Quantity`: Item quantity
- `Price`: Unit price

The AutoCLV `Transaction` dataclass (`generator.py:16-24`) and data mart aggregation (`data_mart.py:108-217`) already support this structure.

### Probabilistic CLV Models (Industry Standard)

#### BG/NBD Model - Purchase Frequency Prediction

**Source**: "Counting Your Customers the Easy Way" (Fader, Hardie, Lee, 2005)

The Beta-Geometric/Negative Binomial Distribution model is the industry standard for predicting future purchase frequency in non-contractual settings.

**Model Assumptions**:
1. While active, customers make purchases following a Poisson process with rate λ
2. Transaction rates vary across customers following a gamma distribution
3. After each transaction, customers have probability p of becoming inactive
4. Dropout probability p varies across customers following a beta distribution

**Required Inputs**:
- `customer_id`: Unique identifier
- `frequency`: Number of repeat purchases in observation period
- `recency`: Time of most recent purchase (in same units as T)
- `T`: Total observation period length

**Key Outputs**:
- Conditional expected transactions in next period
- Probability customer is still active
- Expected number of transactions over customer lifetime

**Implementation Libraries**:
- **Python**: PyMC-Marketing (https://www.pymc-marketing.io/en/stable/notebooks/clv/bg_nbd.html)
- **Python (legacy)**: lifetimes library (archived, use PyMC-Marketing)
- **R**: CLVTools (https://www.clvtools.com/)

**Validation**: BG/NBD produces "very similar results" to the more complex Pareto/NBD model while being "vastly easier to implement" (Fader et al., 2005)

#### Gamma-Gamma Model - Monetary Value Prediction

**Source**: "The Gamma-Gamma Model of Monetary Value" (Fader & Hardie)

Companion model to BG/NBD that predicts average transaction value per customer.

**Model Assumptions**:
1. Monetary value of customer transactions varies randomly around their average
2. Average transaction values vary across customers but remain stable over time for individuals
3. Distribution of average transaction values is independent of the transaction process

**Required Inputs**:
- `customer_id`: Unique identifier
- `frequency`: Number of purchases (must be > 1; exclude one-time buyers)
- `monetary_value`: Average transaction value for customer

**Key Outputs**:
- Expected average transaction value for future purchases
- Customer-level spending predictions

**Combined CLV Calculation**:
```
CLV = (Expected Transactions from BG/NBD) × (Expected Transaction Value from Gamma-Gamma) × Profit Margin
```

With discounting:
```
CLV = Σ [P(active in period t) × Expected transactions × Expected value × Discount factor^t]
```

#### Pareto/NBD Model - Alternative to BG/NBD

**Source**: PyMC Labs - Customer Lifetime Value in the Non-contractual Continuous Case

The original Buy-Till-You-Die model (Schmittlein, Morrison, Colombo, 1987).

**Key Differences from BG/NBD**:
- More mathematically complex
- Dropout follows exponential distribution rather than geometric
- Similar predictive accuracy to BG/NBD in most cases
- Preferred when continuous-time modeling is theoretically important

**When to Use**: BG/NBD is generally recommended unless specific business context requires continuous-time dropout modeling.

### Data Quality and Validation Requirements

#### Historical Data Requirements

**Source**: Microsoft Dynamics 365 CLV Prediction Guide

**Minimum Requirements**:
- **Time span**: 1 year of transaction history (minimum), 2-3 years recommended
- **Rule of thumb**: To predict CLV for N months, have at least 1.5-2× N months of historical data
- **Customer activity**: At least 2-3 transactions per customer across multiple dates
- **Recency**: Data should be current (stale data degrades predictions)

**Transaction Data Components**:
- Transaction ID (unique)
- Transaction date (timestamp)
- Transaction amount (monetary value)
- Return indicator (Boolean for refunds/returns)
- Customer ID

**RFM Summary Table Validation**:
- Check for missing values
- Validate range of transaction dates
- Verify unique customer and order counts
- Ensure reasonable distributions (no extreme outliers without investigation)

#### Data Quality Checks (Current Implementation)

The AutoCLV codebase implements comprehensive validation:

**Transaction-level validation** (`data_mart.py:108-217`):
- Missing required fields (order_id, customer_id, timestamp)
- Timestamp type enforcement
- Non-negative quantities and prices
- Non-negative line totals and costs
- Structured error messages with transaction index

**Customer-level validation** (`customer_contract.py:64-129`):
- Required fields (customer_id, acquisition_ts, source_system)
- Timestamp type and range validation (1970 to present)
- Metadata type checking
- Visibility filtering

**Synthetic data validation** (`validation.py:17-84`):
- Non-negative amounts
- Reasonable order density (average lines per order)
- Promo spike signal detection (ratio-based validation)

These validation patterns should be extended to validate RFM calculations and model inputs.

#### Model Performance Metrics

**Source**: LinkedIn/Microsoft - Testing CLV Models

**Individual Customer Level**:
- **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual CLV
- **Mean Absolute Percentage Error (MAPE)**: MAE expressed as percentage of actual value
- Lower is better; MAPE < 20% considered good for CLV

**Aggregate Level**:
- **Aggregate Revenue Percent Error (ARPE)**: Total predicted revenue vs. total actual revenue
- Important for financial forecasting
- Should be < 10% for production use

**Model Fit**:
- **R-squared**: Explained variance (higher is better)
- **Root Mean Squared Error (RMSE)**: Penalizes large errors more than MAE
- **Log-likelihood**: For probabilistic models

**Best Practice**: "Most business use cases require individual CLV metrics" - validate at both individual and aggregate levels, but prioritize individual-level accuracy.

### Cohort Analysis Best Practices

#### Why Cohort Analysis for CLV

**Source**: Nudge Now, Peel Insights - Cohort Analysis for CLV

Key benefits:
- **Investment evaluation**: "Most VCs are only interested in the range of LTV metrics produced by cohort analysis"
- **Customer quality tracking**: Identify whether newer cohorts are more or less valuable than historical ones
- **Purchase patterns**: Find drop-off points to improve lifetime value
- **Marketing effectiveness**: Accurately assess ad campaign performance by tracking cohort-level outcomes

#### Types of Cohort Analysis

1. **Acquisition Cohorts** (Primary for CLV):
   - Group customers by signup date or first purchase date
   - Track behavior over time from acquisition
   - Aligns with Lens 3 and Lens 4 analyses

2. **Behavioral Cohorts**:
   - Segment by past behaviors (high-value purchasers, frequent buyers)
   - Used for targeted interventions

3. **Predictive Cohorts**:
   - Define key behaviors and track subsequent actions
   - Used for churn prediction and upsell targeting

#### Cohort Analysis Implementation Guidelines

**Time Frame Selection**:
- **Daily cohorts**: High-frequency businesses (food delivery, gaming)
- **Weekly cohorts**: E-commerce, SaaS
- **Monthly cohorts**: Most common for CLV (recommended)
- **Quarterly/Yearly cohorts**: B2B, high-consideration purchases

**Segmentation Variables**:
- Marketing channel (organic, paid, referral)
- First product purchased
- Geographic region
- Promo code or campaign
- Customer tier or segment

**Best Practice from CLVTools**: "Probabilistic models work best when analyzing specific customer cohorts who joined the company at the same time"

**Current Codebase Support**:
- `CustomerIdentifier.acquisition_ts` (`customer_contract.py:40`) enables cohort grouping
- `CustomerIdentifier.metadata` (`customer_contract.py:43`) supports segmentation attributes
- Period aggregation (`data_mart.py:219-268`) provides cohort × time metrics

### Enterprise System Architecture

#### Data Pipeline Architecture

**Source**: Theta CLV Data Stack, Databricks

**Recommended Architecture**:

1. **Data Ingestion Layer**:
   - Source systems: CRM, e-commerce platform, point-of-sale, billing
   - ETL/ELT tools: Fivetran, Airbyte, Stitch, custom connectors
   - Streaming: Kafka, Kinesis for real-time data

2. **Data Warehouse Layer**:
   - Centralized storage: Snowflake, Google BigQuery, Redshift, Databricks
   - Unified view across transaction sources
   - Historical data retention (2-3 years recommended)

3. **Feature Engineering Layer**:
   - Centralized feature store (Feast, Tecton)
   - Precomputed RFM features
   - Cohort assignments
   - Customer aggregations

4. **Model Layer**:
   - Model training: PyMC-Marketing, CLVTools, custom implementations
   - Model versioning: MLflow, Weights & Biases
   - Hyperparameter tuning

5. **Serving Layer**:
   - Batch scoring: Daily/weekly updates
   - Real-time inference: API endpoints for on-demand scoring
   - Results storage: Database or data warehouse

6. **Activation Layer**:
   - Reverse ETL: Hightouch, Census, custom sync
   - Target systems: Marketing platforms, CRM, business intelligence tools

**Current AutoCLV Architecture Fit**:
- **CLI interface** (`cli.py:31-78`) supports batch processing from JSON files
- **CustomerDataMartBuilder** (`data_mart.py:92-105`) provides feature engineering foundation
- **Synthetic data** (`texas_clv_client.py`) enables testing without production data
- Missing: Model training, serving layer, activation components

#### Incremental Loading Strategies

**Source**: Hevo Data, Lightpoint Global

**Timestamp-based Loading** (Recommended for CLV):
```sql
SELECT * FROM transactions
WHERE updated_at > :last_sync_timestamp
```

**Benefits**:
- Simple implementation
- Low computational overhead
- Works with existing updated_at columns

**Change Data Capture (CDC)**:
- Captures only changed records at database level
- Requires database support (MySQL binlog, PostgreSQL WAL)
- Provides near-real-time updates

**Slowly Changing Dimensions (SCD)**:
- Type 2 SCD: Keep historical versions of customer records
- Important for tracking customer attribute changes over time
- Supports cohort membership evolution

**Recommendation for AutoCLV**: Start with timestamp-based incremental loading, upgrade to CDC if real-time requirements emerge.

#### Batch vs. Real-Time Processing

**Source**: Lightpoint Global, Switchboard Software

**Batch Processing** (Recommended for Initial Implementation):
- **Schedule**: Daily or weekly CLV recalculation
- **Pros**: Simpler implementation, lower infrastructure cost, sufficient for most use cases
- **Cons**: Data lag, not suitable for real-time personalization

**Real-Time Processing**:
- **Use cases**: Real-time personalization, fraud detection, dynamic pricing
- **Infrastructure**: Streaming pipelines, model serving APIs, caching layer
- **Complexity**: Higher development and operational overhead

**Hybrid Approach** (Recommended for Production):
- Batch model training (weekly/monthly retraining)
- Real-time inference for scoring (API-based)
- Cached precomputed scores for high-frequency access

**Evolution Path**: Organizations should start with batch processing and upgrade to hybrid/real-time as business needs and data freshness requirements grow.

### MLOps and Model Monitoring

#### Model Drift Detection

**Source**: Towards Data Science, KDnuggets, Evidently AI

**Types of Drift**:

1. **Data Drift** (Input Distribution Changes):
   - Changes in customer acquisition patterns
   - Shifts in purchase behavior
   - Seasonality effects
   - New marketing channels

2. **Concept Drift** (Relationship Changes):
   - Economic conditions affecting purchasing
   - Competitive landscape changes
   - Product mix evolution

3. **Upstream Data Changes**:
   - Schema changes in source systems
   - Data quality degradation
   - Integration pipeline failures

**Detection Methods**:

**Statistical Tests**:
- Kolmogorov-Smirnov test: Compare distributions
- Chi-squared test: Categorical feature shifts
- Population Stability Index (PSI): Industry standard for feature drift

**Distance Metrics**:
- KL Divergence: Measure distribution difference
- Wasserstein Distance: Earth mover's distance
- Jensen-Shannon Divergence: Symmetric KL divergence

**Monitoring Strategies**:

1. **Feature Monitoring**:
   - Track distributions of frequency, recency, monetary value
   - Alert on significant shifts (PSI > 0.1 indicates moderate shift)

2. **Prediction Monitoring**:
   - Monitor distribution of predicted CLV scores
   - Track percentage of customers in different value segments

3. **Performance Monitoring** (requires ground truth):
   - Track actual vs. predicted for completed customer lifetimes
   - Monitor MAE, MAPE, RMSE over time

**Mitigation Strategy**:
- Automated alerts for drift detection
- Regular retraining schedule (quarterly recommended for CLV)
- A/B testing of new model versions before full deployment
- Rollback capability for model versions

#### Model Validation Framework

**Source**: LinkedIn Advice - Testing CLV Models

**Validation Approach**:

1. **Train/Test Split**:
   - Temporal split (train on 2020-2022, test on 2023)
   - Ensures model handles time-based patterns

2. **Cross-Validation**:
   - Time-series cross-validation with expanding window
   - Example: Train on months 1-12, test on 13-15; train on 1-15, test on 16-18

3. **Holdout Cohort**:
   - Hold out recent cohorts completely
   - Validates model generalizes to new customer segments

**Model Comparison**:
- Compare BG/NBD vs. Pareto/NBD vs. simple heuristics
- Benchmark against historical average
- Validate assumptions (retention rates, churn rates)

**Best Practice**: "Finding the right balance between precision and practicality" - prioritize models that are accurate, interpretable, and maintainable.

### RFM Analysis Foundation

**Source**: Bruce Hardie - RFM and CLV, TechTarget, Omniconvert

RFM (Recency-Frequency-Monetary) analysis is foundational to both the Five Lenses framework and probabilistic CLV models.

#### RFM Components

**Recency (R)**:
- Time since last purchase/activity
- Recent customers more likely to purchase again
- Input to BG/NBD and Pareto/NBD models

**Frequency (F)**:
- Number of purchases in observation period
- Identifies loyal vs. one-time customers
- Primary input to CLV models

**Monetary (M)**:
- Total or average spending
- Identifies high-value customers
- Input to Gamma-Gamma model

#### RFM in Five Lenses Context

**Lens 1** (Single Period): RFM distribution shows customer heterogeneity within a period

**Lens 2** (Period Comparison): RFM changes reveal customer behavior shifts

**Lens 3** (Cohort Evolution): Average R, F, M evolution over cohort lifetime

**Lens 4** (Cohort Comparison): RFM differences across cohorts

**Lens 5** (Overall Health): Weighted RFM across entire customer base

#### Implementation in AutoCLV

The `PeriodAggregation` dataclass (`data_mart.py:36-45`) captures frequency and monetary metrics:
- `total_orders`: Frequency within period
- `total_spend`: Monetary value for period
- `period_start`: Enables recency calculation (time since period_start)

**Gap**: Explicit RFM calculation functions not yet implemented. Should add to lens1.py or create dedicated rfm.py module.

### Production Deployment Considerations

#### Data Privacy and Compliance

**Source**: RFM analysis resources

**GDPR/CCPA Requirements**:
- Right to be forgotten: Ability to delete customer records
- Data minimization: Only collect necessary transaction data
- Consent management: Track customer opt-in/opt-out
- Anonymization: Support for pseudonymized customer IDs

**Implementation in AutoCLV**:
- `CustomerIdentifier.is_visible` (`customer_contract.py:42`) supports visibility filtering
- `CustomerContract.enforce_visibility` (`customer_contract.py:61`) enables compliance enforcement

**Gaps**:
- No anonymization utilities
- No consent tracking in metadata
- No data retention/deletion workflows

#### Scalability Considerations

**Transaction Volume Handling**:
- Current CLI has 25 MiB file size limit (`cli.py:19-23`)
- Recommendation: Process in chunks for large datasets
- Consider streaming aggregation for very large scale

**Model Training Time**:
- BG/NBD with MCMC sampling can take hours for millions of customers
- Recommendation: Use MAP estimation for large scale, MCMC for accuracy-critical cases
- Consider distributed computing (Dask, Spark) for massive scale

**Storage Requirements**:
- Historical transaction data: Plan for 2-3 years retention
- Model artifacts: Version control for multiple model versions
- Feature store: Precomputed RFM features for all customers

#### Testing Strategy

**Unit Tests** (Already Implemented):
- Customer contract validation (`test_customer_foundation.py:12-79`)
- Data mart aggregation (`test_customer_foundation.py:120-218`)
- Synthetic data generation (`test_synthetic_data.py`)

**Integration Tests** (Needed):
- End-to-end: Raw transactions → data mart → RFM → CLV prediction
- Multi-cohort analysis workflows
- Data quality checks in pipeline

**Model Tests** (Needed):
- Model convergence checks (trace plots, R-hat statistics)
- Posterior predictive checks
- Cross-validation accuracy
- Drift detection simulation

**Property-Based Tests** (Recommended):
- Hypothesis testing for aggregation correctness
- Invariant checking (e.g., period aggregations sum to order totals)

## Code References

### Foundation Infrastructure
- `customer_base_audit/foundation/data_mart.py:92-105` - CustomerDataMartBuilder class
- `customer_base_audit/foundation/data_mart.py:107-217` - Transaction to order aggregation
- `customer_base_audit/foundation/data_mart.py:219-268` - Order to period aggregation
- `customer_base_audit/foundation/customer_contract.py:17-52` - CustomerIdentifier dataclass
- `customer_base_audit/foundation/customer_contract.py:55-163` - CustomerContract validation and merging

### Synthetic Data Generation
- `customer_base_audit/synthetic/generator.py:26-68` - ScenarioConfig for configurable scenarios
- `customer_base_audit/synthetic/generator.py:70-269` - Transaction generation with cohort tracking
- `customer_base_audit/synthetic/texas_clv_client.py:49-106` - Texas 4-city CLV scenario
- `customer_base_audit/synthetic/validation.py:17-84` - Validation utilities

### Analysis Placeholders (Ready for Implementation)
- `customer_base_audit/analyses/lens1.py` - Single-period analysis (empty)
- `customer_base_audit/analyses/lens2.py` - Period comparison (empty)
- `customer_base_audit/analyses/lens3.py` - Cohort evolution (empty)
- `customer_base_audit/analyses/lens4.py` - Multi-cohort comparison (empty)
- `customer_base_audit/analyses/lens5.py` - Overall health (empty)

### CLI and Utilities
- `customer_base_audit/cli.py:31-78` - CLI for building data marts from JSON

### Tests
- `tests/test_customer_foundation.py:12-79` - Customer contract tests
- `tests/test_customer_foundation.py:120-218` - Data mart builder tests
- `tests/test_synthetic_data.py` - Synthetic data generation tests
- `tests/test_lens1.py` - Lens 1 tests (placeholder)
- `tests/test_lens2.py` - Lens 2 tests (placeholder)

### Documentation
- `README.md` - Project overview and usage
- `docs/foundational_data_platform.md` - Customer contract and data mart documentation
- `docs/issues/feature_foundational_customer_platform.md` - Foundation platform feature spec
- `docs/issues/feature_reusable_audit_components.md` - Lens analyses feature spec

## Architecture Documentation

### Current Implementation Patterns

**Immutable Value Objects**:
- Extensive use of frozen dataclasses for immutable data: `Customer`, `Transaction`, `OrderAggregation`, `PeriodAggregation`, `CustomerIdentifier`, `ScenarioConfig`
- Benefits: Thread-safety, cacheable, predictable

**Builder Pattern**:
- `CustomerDataMartBuilder` constructs complex aggregations with configurable parameters
- Separates configuration (period granularities) from execution (build method)

**Validation Pattern**:
- `CustomerContract` separates validation from merging operations
- Structured error messages with context dictionaries
- Fail-fast with detailed error information

**Decimal Arithmetic for Finance**:
- All monetary calculations use Decimal internally
- Quantized to 2 decimal places with ROUND_HALF_UP
- Converted to float only for final output

**Factory Pattern**:
- `generate_customers_and_transactions()` creates synthetic datasets based on scenarios
- `ScenarioConfig` encapsulates generation parameters

### Data Flow

**Current Pipeline**:
```
Raw Transactions (JSON/dict)
  → CustomerDataMartBuilder.build()
  → _aggregate_orders() [Transaction → Order]
  → _aggregate_periods() [Order → Period]
  → CustomerDataMart (orders + periods by granularity)
  → Optional: as_dict() for JSON serialization
```

**Recommended CLV Pipeline**:
```
Raw Transactions
  → CustomerDataMart (existing)
  → RFM Calculation (new)
  → Cohort Assignment (new)
  → Model Input Preparation (new)
  → BG/NBD Model (new) → Expected frequency
  → Gamma-Gamma Model (new) → Expected monetary value
  → CLV Calculation (new) → Final CLV scores
  → Validation & Monitoring (new)
```

### Missing Components for Enterprise CLV

**Lens Implementations** (High Priority):
1. **Lens 1 - Single Period Analysis**:
   - Customer count and segmentation
   - Revenue concentration (top N% customers drive X% revenue)
   - One-time buyer percentage
   - RFM distribution visualization

2. **Lens 2 - Period Comparison**:
   - Customer migration matrix (retained, churned, new, reactivated)
   - Period-on-period metric changes
   - Cohort composition shifts

3. **Lens 3 - Cohort Evolution**:
   - Retention curves by cohort
   - Revenue decay patterns
   - Time to second purchase distribution

4. **Lens 4 - Multi-Cohort Comparison**:
   - Cohort performance benchmarking
   - Customer quality trends
   - Cohort revenue contribution over time

5. **Lens 5 - Overall Health**:
   - Customer base composition
   - Weighted retention metrics
   - Projected revenue based on cohort performance

**Probabilistic Models** (High Priority):
- BG/NBD model implementation
- Gamma-Gamma model implementation
- Model training pipelines
- Hyperparameter optimization
- Model versioning and persistence

**RFM Module** (Medium Priority):
- Calculate R, F, M metrics from period aggregations
- Support configurable RFM binning/scoring
- Integration with Five Lenses analyses

**Model Validation** (Medium Priority):
- Train/test splitting utilities
- Cross-validation framework
- Performance metric calculation
- Drift detection monitoring

**Production Infrastructure** (Lower Priority for MVP):
- Incremental data loading
- Model serving API
- Reverse ETL for activation
- Dashboard/visualization layer

## Historical Context (from thoughts/)

No prior research documents found in thoughts/ directory. This is the foundational research document for the CLV implementation.

## Related Research

No related research documents found. Future research topics to consider:
- Model comparison study: BG/NBD vs. Pareto/NBD vs. alternative models
- Scalability testing: Performance benchmarks at different data scales
- Industry-specific adaptations: B2B, subscription, marketplace contexts

## Open Questions

### Methodology Questions

1. **Model Selection Criteria**: Under what specific business conditions should we prefer Pareto/NBD over BG/NBD? Are there transaction patterns in the synthetic data that would favor one over the other?

2. **Cohort Definition**: Should cohorts be strictly acquisition-date based, or should we support behavioral cohorts (e.g., first product purchased, channel acquired through)?

3. **Time Granularity**: The book uses quarters and years; should we also support monthly granularity for higher-frequency businesses?

4. **Promotional Adjustments**: How should we handle promotional periods in CLV calculations? Should promo purchases be weighted differently?

### Implementation Questions

5. **Library Choice**: PyMC-Marketing (Bayesian MCMC) vs. lifetimes (MAP estimation) vs. CLVTools (R)? What are the accuracy-performance tradeoffs?

6. **Compute Resources**: What are realistic computation time expectations for BG/NBD MCMC sampling with 1,000 customers? 10,000? 1 million?

7. **Model Retraining Frequency**: How often should models be retrained? Monthly? Quarterly? Drift-triggered?

8. **One-Time Buyer Handling**: Gamma-Gamma excludes one-time buyers. What CLV methodology should we use for this segment?

### Data Quality Questions

9. **Historical Data Sufficiency**: Texas CLV client has data from city launch dates. Is 6-12 months sufficient for early-launched cities, or should we require more history?

10. **Returns/Refunds**: Current transaction schema doesn't have return indicators. Should we add support for negative transactions or a separate returns dimension?

11. **Acquisition Date Precision**: How do we handle customers with unknown acquisition dates in legacy data?

### Production Questions

12. **Serving Latency**: What are acceptable latency requirements for CLV score retrieval? Batch daily updates vs. real-time API?

13. **Version Management**: How should we version CLV models when methodologies change? Support multiple concurrent model versions?

14. **Explainability**: How do we explain CLV scores to business stakeholders? What visualizations or decompositions are most effective?

15. **Confidence Intervals**: Should we provide uncertainty estimates (Bayesian credible intervals) with CLV scores, or just point estimates?

## Recommended Implementation Plan

### Phase 1: Five Lenses Foundation (Weeks 1-3)

**Objective**: Implement descriptive analytics framework from the book

**Deliverables**:
1. **Lens 1 Module** (`lens1.py`):
   - Customer count and segmentation
   - Revenue concentration metrics
   - RFM distribution analysis
   - One-time buyer percentage

2. **Lens 2 Module** (`lens2.py`):
   - Customer migration matrix
   - Period-on-period comparison
   - Metric delta calculation

3. **Lens 3 Module** (`lens3.py`):
   - Cohort retention curves
   - Revenue decay by cohort
   - Time to second purchase

4. **RFM Utilities** (new `rfm.py`):
   - Calculate recency, frequency, monetary from PeriodAggregation
   - Scoring and binning utilities
   - Visualization helpers

**Validation**:
- Use Texas CLV client synthetic data
- Generate sample reports for each lens
- Compare outputs to book examples (Madrigal dataset)

### Phase 2: Probabilistic CLV Models (Weeks 4-6)

**Objective**: Implement BG/NBD and Gamma-Gamma for predictive CLV

**Deliverables**:
1. **Model Input Preparation** (new `model_prep.py`):
   - Convert CustomerDataMart to model input format
   - Calculate frequency, recency, T from period aggregations
   - Handle edge cases (one-time buyers, very recent customers)

2. **BG/NBD Implementation** (new `models/bg_nbd.py`):
   - Wrapper around PyMC-Marketing BG/NBD
   - Training pipeline with configurable priors
   - Prediction methods (expected transactions, P(alive))

3. **Gamma-Gamma Implementation** (new `models/gamma_gamma.py`):
   - Wrapper around PyMC-Marketing Gamma-Gamma
   - Filter one-time buyers
   - Spending predictions

4. **CLV Calculator** (new `clv_calculator.py`):
   - Combine BG/NBD + Gamma-Gamma
   - Support discounting and profit margins
   - Output CLV scores with optional confidence intervals

**Validation**:
- Cross-validation on synthetic data
- Compare MAP vs. MCMC accuracy
- Benchmark against simple heuristics (average revenue per customer)

### Phase 3: Model Validation and Quality Assurance (Weeks 7-8)

**Objective**: Ensure model quality and reliability

**Deliverables**:
1. **Validation Framework** (new `model_validation.py`):
   - Train/test splitting utilities
   - MAE, MAPE, RMSE, ARPE calculation
   - Cross-validation runner

2. **Model Diagnostics** (extend models):
   - Convergence checks (R-hat, trace plots)
   - Posterior predictive checks
   - Model comparison utilities

3. **Drift Detection** (new `monitoring/drift.py`):
   - PSI calculation for features
   - Distribution comparison tests
   - Alerting framework

4. **Documentation**:
   - Model cards documenting assumptions and limitations
   - Validation reports with metrics
   - User guide for interpreting CLV scores

### Phase 4: Production Infrastructure (Weeks 9-12)

**Objective**: Production-ready deployment capabilities

**Deliverables**:
1. **Incremental Loading** (extend `cli.py`):
   - Timestamp-based incremental processing
   - Support for CDC pipelines
   - Checkpoint management

2. **Model Serving**:
   - Batch scoring CLI
   - Optional: REST API for real-time scoring
   - Model versioning and persistence

3. **Lens 4 & 5 Modules**:
   - Multi-cohort comparison (lens4.py)
   - Overall health dashboard (lens5.py)

4. **Integration Examples**:
   - Snowflake integration example
   - BigQuery integration example
   - Reverse ETL activation example

**Validation**:
- End-to-end pipeline test with realistic data volumes
- Performance benchmarking (transactions/second, model training time)
- Scalability testing (1K, 10K, 100K, 1M customers)

### Success Metrics

**Accuracy**:
- MAPE < 20% at individual customer level
- ARPE < 10% at aggregate level
- Cross-validation RMSE competitive with industry benchmarks

**Reliability**:
- 95%+ of validation checks pass on synthetic data
- Drift detection catches simulated distribution shifts
- Model convergence in < 5 minutes for 10K customers (MAP)

**Usability**:
- CLI can process 100K transactions in < 30 seconds
- Documentation sufficient for new user to generate CLV scores
- Example reports for all five lenses

**Scalability**:
- Support 1M+ customers with batch processing
- Incremental loading reduces processing time by 80%+ for daily updates
- Model retraining completes in < 2 hours

## External Resources

### Books and Academic Papers
- [The Customer-Base Audit](https://www.pennpress.org/9781613631607/the-customer-base-audit/) - Fader, Hardie, Ross (2022) - Primary methodological source
- [Counting Your Customers the Easy Way](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=578087) - Fader, Hardie, Lee (2005) - BG/NBD model
- [Bruce Hardie's Website](http://www.brucehardie.com/) - Comprehensive CLV papers and notes
- [The Gamma-Gamma Model](https://www.brucehardie.com/notes/025/gamma_gamma.pdf) - Fader & Hardie - Monetary value model

### Implementation Libraries
- [PyMC-Marketing](https://www.pymc-marketing.io/) - Modern Python Bayesian CLV library
- [PyMC-Marketing BG/NBD](https://www.pymc-marketing.io/en/stable/notebooks/clv/bg_nbd.html) - BG/NBD implementation guide
- [PyMC-Marketing Gamma-Gamma](https://www.pymc-marketing.io/en/stable/notebooks/clv/gamma_gamma.html) - Gamma-Gamma guide
- [CLVTools](https://www.clvtools.com/) - Enterprise R package
- [Lifetimes Documentation](https://lifetimes.readthedocs.io/) - Legacy Python library (archived)

### Best Practices and Guides
- [Microsoft Dynamics - CLV Prediction](https://learn.microsoft.com/en-us/dynamics365/customer-insights/data/sample-guide-predict-clv) - Enterprise data quality standards
- [Nudge Now - Cohort Analysis](https://nudgenow.com/blogs/cohort-analysis-customer-lifetime-value-steps) - Cohort best practices
- [Peel Insights - Cohort 101](https://www.peelinsights.com/post/cohort-analysis-101-an-introduction) - Cohort analysis introduction
- [TechTarget - RFM Analysis](https://www.techtarget.com/searchdatamanagement/definition/RFM-analysis) - RFM foundations

### Model Validation and MLOps
- [LinkedIn - Testing CLV Models](https://www.linkedin.com/advice/0/how-do-you-test-experiment-different-clv-models) - QA framework
- [Towards Data Science - Model Drift](https://towardsdatascience.com/how-to-detect-model-drift-in-mlops-monitoring-7a039c22eaf9/) - Drift detection
- [KDnuggets - Managing Drift](https://www.kdnuggets.com/2023/05/managing-model-drift-production-mlops.html) - Production drift management
- [Evidently AI - Data Drift](https://www.evidentlyai.com/ml-in-production/data-drift) - Drift detection methods

### Enterprise Architecture
- [Theta - CLV Data Stack](https://thetaclv.com/resource/clv-data-stack/) - Modern data architecture
- [Databricks - CLV Notebooks](https://www.databricks.com/notebooks/CLV_Part_1_Customer_Lifetimes.html) - Production examples
- [Switchboard - CLV Engine](https://switchboard-software.com/post/the-customer-lifetime-value-optimization-engine-from-theory-to-automated-reality/) - System design patterns

### Research and Case Studies
- [McKinsey - Author Talks](https://www.mckinsey.com/featured-insights/mckinsey-on-books/author-talks-peter-fader-and-michael-ross-share-their-playbook-for-customer-centricity) - Fader & Ross interview
- [PyMC Labs - Pareto/NBD](https://www.pymc-labs.com/blog-posts/pareto-nbd) - Modern Bayesian implementation
- [Medium - BG/NBD & Gamma-Gamma](https://medium.com/@yassirafif/projecting-customer-lifetime-value-using-the-bg-nbd-and-the-gamma-gamma-models-9a937c60fe7f) - Practical tutorial
