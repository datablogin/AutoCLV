---
name: Feature request
about: Suggest an idea for this project
title: '[FEATURE] Foundational customer data platform elements'
labels: enhancement
assignees: ''

---

**Is your feature request related to a problem? Please describe.**
We currently have inconsistent answers to core questions such as “How many customers do we have?” and “When was this customer acquired?”. Without a shared schema and reproducible customer×time aggregation we cannot scale the audit or CLV work, nor onboard new teams quickly. Data engineering spends large amounts of time recreating basic tables for every engagement.

**Describe the solution you'd like**
Deliver the foundational platform artifacts that codify our customer definition and produce reusable customer-period datasets:
- Customer definition contract (schema + documentation) including acquisition timestamp, visibility status, and merge rules across source systems.
- Reusable ETL/data-mart build that rolls raw transactions into order-level and period-level tables (quarterly + annual) with optional product enrichment.
- Automation hooks (dbt/Airflow or equivalent) so the mart can refresh on demand and be version-controlled.

**Describe alternatives you've considered**
- Continue with ad-hoc SQL/notebook jobs per project (high duplication, inconsistent business rules).
- Rely on downstream analytics teams to reshape data (pushes complexity to each squad, increases error risk).

**Additional context**
- **Expected outcomes:**
  - Single auditable customer count across functions.
  - One-click regeneration of customer×time tables for any business unit.
  - Reduced data onboarding time for new analytics projects (<1 day).
- **Acceptability tests:**
  - Customer contract validated against at least two disparate source systems without manual overrides.
  - Data-mart build runs successfully end-to-end on production-sized sample and passes schema/unit tests (e.g., no duplicate customer-period rows, mandatory fields non-null).
- **KPIs:**
  - % of analytics workstreams using the shared mart (target: 100% after rollout).
  - Time to answer “customer count” audit question (target: <5 minutes via SQL/BI).
  - Number of data defects detected in weekly QA (target: zero P1 issues after first month).
- **Integration tests:**
  - dbt or pytest-based tests confirming joins to downstream models (e.g., Lens computation library) still succeed after mart refresh.
  - Contract tests ensuring analytics services can deserialize the exported schema without changes.
