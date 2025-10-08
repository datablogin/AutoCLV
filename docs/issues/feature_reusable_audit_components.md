---
name: Feature request
about: Suggest an idea for this project
title: '[FEATURE] Reusable customer-base audit components'
labels: enhancement
assignees: ''

---

**Is your feature request related to a problem? Please describe.**
Our audit outputs are built manually per engagement, resulting in inconsistent Lens calculations, plotting styles, and decomposition logic. Analysts duplicate code to compute Lens 1–5 metrics, creating maintenance drag and raising the risk of divergent interpretations.

**Describe the solution you'd like**
Package shared audit capabilities into a reusable library:
- Metric decomposition utilities (transactions × AOV × margin, up/down tables, Lorenz/decile helpers).
- Lens computation module that produces standardized data frames for Lenses 1–5 directly from the customer-period mart.
- Visualization templates (histograms, cohort heatmaps, health dashboards) with styling aligned to the audit narrative.
- Cohort metadata tagging utilities so marketing/channel context is automatically attached to cohorts when analyses run.

**Describe alternatives you've considered**
- Maintain independent notebooks per team (high code drift, limited QA).
- Rely on BI tooling visuals only (difficult to enforce analytic rigor and decomposition logic).

**Additional context**
- **Expected outcomes:**
  - Analytics squads can run end-to-end audit packs via a single API call or CLI.
  - Stakeholder decks use consistent charts and language across business units.
  - Faster onboarding of new analysts through documented, tested package.
- **Acceptability tests:**
  - Unit tests covering each Lens output against fixture data.
  - Visual snapshot tests (e.g., matplotlib/seaborn or Plotly baseline images) to detect style regressions.
  - Documentation examples rendering without errors (doctest or nbconvert runs).
- **KPIs:**
  - Reduction in duplicated audit code across repos (>70% reduction measured via code search).
  - Time to produce a new audit report drops by 50%.
  - Audit review defect rate (incorrect metric definitions) reduced to zero within two cycles.
- **Integration tests:**
  - CI job executing the library against synthetic and sample production datasets, confirming Lens outputs feed into reporting notebooks without modification.
  - Compatibility check with current visualization pipeline (e.g., ensures charts embed cleanly in existing reporting stack).
