---
name: Feature request
about: Suggest an idea for this project
title: '[FEATURE] Synthetic data generation and validation suite'
labels: enhancement
assignees: ''

---

**Is your feature request related to a problem? Please describe.**
We lack reliable synthetic datasets to exercise the audit pipeline, making it hard to test edge cases (promo spikes, cohort degradation, product launches) and risky to iterate when production data is unavailable. Current fixtures are ad hoc CSVs with limited realism.

**Describe the solution you'd like**
Develop a configurable synthetic-data toolkit tailored to marketing, sales, and loyalty scenarios:
- Faker/Mimesis + domain-specific generators for transactions, marketing touches, loyalty accrual/redemption, and channel attribution.
- Scenario packs (e.g., high-churn cohort, heavy promotion month, product recall) to stress-test Lens diagnostics.
- Data validation suite that asserts statistical properties (distribution skew, cohort decay curves) stay within expected ranges.

**Describe alternatives you've considered**
- Use production snapshots (privacy/compliance hurdles, difficult to share with vendors).
- Hand-crafted spreadsheets for demos (not scalable, misses behavioral nuance).

**Additional context**
- **Expected outcomes:**
  - Every CI run can validate audit logic without needing production data access.
  - Analysts can prototype “what-if” CLV scenarios safely.
  - Synthetic datasets become onboarding material for new engineers/data scientists.
- **Acceptability tests:**
  - Generator can produce at least three predefined scenarios with parameter overrides.
  - Statistical QA tests (e.g., Kolmogorov-Smirnov checks on spend distributions) pass across generated datasets.
  - Data passes through the foundational customer mart and reusable Lens library without manual adjustments.
- **KPIs:**
  - % of pipeline tests using synthetic data (target: 100% of CI suites).
  - Mean time to reproduce a reported bug with synthetic data (<30 minutes).
  - Number of compliance exceptions raised for using synthetic data (target: zero).
- **Integration tests:**
  - CI job that generates synthetic datasets and executes full Lens 1–5 pipeline to ensure compatibility.
  - Regression tests confirming synthetic outputs remain stable when generator dependencies upgrade.
