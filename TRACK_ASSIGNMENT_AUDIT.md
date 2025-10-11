# Track Assignment Audit - Open Issues

**Date**: 2025-10-10
**Auditor**: Claude (Track A Agent)
**Context**: Review triggered by Issue #31 mislabeling (validation framework incorrectly assigned to Track A instead of Track B)

## Summary

Found **6 issues with INCORRECT track assignments** and **8 issues MISSING track assignments**.

---

## ❌ Issues with INCORRECT Track Assignments

### Issue #40: [DOCS] Create API Reference and UPDATE README
- **Current Assignment**: Track B
- **Correct Assignment**: Track C
- **Reason**: Documentation files (`docs/api_reference.md`, `README.md`) belong to Track C
- **Files Affected**: `docs/api_reference.md`, `README.md`
- **Ownership Rule**: Track C owns `docs/**`

### Issue #39: [DOCS] Create Example Notebooks
- **Current Assignment**: Track A
- **Correct Assignment**: Track C
- **Reason**: Example notebooks (`examples/**`) belong to Track C
- **Files Affected**: `examples/01_texas_clv_walkthrough.ipynb`, `examples/02_custom_cohorts.ipynb`, etc.
- **Ownership Rule**: Track C owns `examples/**`

### Issue #37: [FEATURE] Implement Drift Detection Module
- **Current Assignment**: Track A
- **Correct Assignment**: Track B
- **Reason**: Drift detection is model diagnostics/monitoring (Track B scope)
- **Files Affected**: `customer_base_audit/monitoring/drift.py`, `customer_base_audit/monitoring/exports.py`
- **Ownership Rule**: Track B owns model validation and diagnostics

### Issue #36: [FEATURE] CLI Batch Processing Enhancements
- **Current Assignment**: Track B
- **Correct Assignment**: Track C
- **Reason**: CLI and operational tooling belong to Track C (integration/operational work)
- **Files Affected**: `customer_base_audit/cli.py`
- **Ownership Rule**: Track C owns operational tools and integration utilities

### Issue #34: [FEATURE] Implement Lens 4 Multi-Cohort Comparison
- **Current Assignment**: Track A
- **Correct Assignment**: Track B
- **Reason**: Lens 4 depends on cohorts and Lens 3, both owned by Track B
- **Files Affected**: `customer_base_audit/analyses/lens4.py`
- **Ownership Rule**: Track B owns `lens3.py` and `cohorts.py`, Lens 4 is cohort-centric
- **Note**: Track A owns Lens 1-2 (basic RFM analyses), Track B owns cohort-based lenses (3-5)

### Issue #31: [FEATURE] Implement Validation Framework
- **Current Assignment**: Track A (in issue body)
- **Correct Assignment**: Track B
- **Status**: ✅ Already fixed in AGENTS.md files
- **Reason**: Validation framework validates CLV models owned by Track B
- **Files Affected**: `customer_base_audit/validation/validation.py`, `customer_base_audit/validation/diagnostics.py`

---

## ⚠️ Issues MISSING Track Assignments

### Issue #78: [FEATURE] Pandas Integration Adapter Layer for Track A Components
- **Current Assignment**: None
- **Correct Assignment**: Track A
- **Reason**: Pandas adapters for RFM, Lens 1, Lens 2 (all Track A components)
- **Files Affected**: `customer_base_audit/pandas/rfm.py`, `customer_base_audit/pandas/lens1.py`, `customer_base_audit/pandas/lens2.py`

### Issue #75: [FEATURE] Parallel Processing for Large-Scale Customer Analytics (10M+ customers)
- **Current Assignment**: None
- **Correct Assignment**: Track A
- **Reason**: Parallel processing for RFM, Lens 1, Lens 2 calculations
- **Files Affected**: `customer_base_audit/foundation/rfm.py`, `customer_base_audit/analyses/lens1.py`, `customer_base_audit/analyses/lens2.py`

### Issue #64: Test Coverage & Documentation: model_prep missing tests and enhanced docs
- **Current Assignment**: None
- **Correct Assignment**: Track B
- **Reason**: Tests and docs for `model_prep.py` (Track B owns `models/**`)
- **Files Affected**: `customer_base_audit/models/model_prep.py`, `tests/test_model_prep.py`

### Issue #63: Code Quality: model_prep inconsistent DataFrame sorting and potential performance issues
- **Current Assignment**: None
- **Correct Assignment**: Track B
- **Reason**: Code quality fixes for `model_prep.py`
- **Files Affected**: `customer_base_audit/models/model_prep.py`

### Issue #62: High Priority: model_prep needs timezone handling, time calculation constants, and bias quantification
- **Current Assignment**: None
- **Correct Assignment**: Track B
- **Reason**: Enhancements to `model_prep.py`
- **Files Affected**: `customer_base_audit/models/model_prep.py`

### Issue #61: Critical: model_prep missing observation_start validation and has type inconsistency
- **Current Assignment**: None
- **Correct Assignment**: Track B
- **Reason**: Critical fixes for `model_prep.py`
- **Files Affected**: `customer_base_audit/models/model_prep.py`

### Issue #58: Data Safety: Cohort assignment silently drops customers outside cohort ranges
- **Current Assignment**: None
- **Correct Assignment**: Track B
- **Reason**: Bug fix for `cohorts.py` (Track B owns cohorts)
- **Files Affected**: `customer_base_audit/foundation/cohorts.py`

### Issue #54: Critical: Lens3 retention_rate has confusing semantics (cumulative vs. period-specific)
- **Current Assignment**: None
- **Correct Assignment**: Track B
- **Reason**: Semantic/naming issue in Lens 3 (Track B owns `lens3.py`)
- **Files Affected**: `customer_base_audit/analyses/lens3.py`

---

## Track Ownership Reference (Corrected)

### Track A: Core Analytics (RFM + Lenses 1-2)
- `customer_base_audit/foundation/rfm.py`
- `customer_base_audit/analyses/lens1.py`
- `customer_base_audit/analyses/lens2.py`
- Tests: `tests/test_rfm.py`, `tests/test_lens1.py`, `tests/test_lens2.py`

### Track B: Machine Learning Models + Cohort Analytics (Lenses 3-5)
- `customer_base_audit/models/**` (model_prep, bg_nbd, gamma_gamma, clv_calculator)
- `customer_base_audit/validation/**` (validation, diagnostics)
- `customer_base_audit/monitoring/**` (drift detection, model monitoring)
- `customer_base_audit/analyses/lens3.py` (cohort evolution)
- `customer_base_audit/analyses/lens4.py` (multi-cohort comparison)
- `customer_base_audit/analyses/lens5.py` (overall health)
- `customer_base_audit/foundation/cohorts.py`
- Tests: `tests/test_model_prep.py`, `tests/test_bg_nbd.py`, `tests/test_gamma_gamma.py`, `tests/test_clv_calculator.py`, `tests/test_validation.py`, `tests/test_lens3.py`, `tests/test_cohorts.py`

### Track C: Documentation and Examples
- `docs/**`
- `examples/**`
- `README.md`
- `tests/integration/**`
- `.github/workflows/**`
- CLI and operational tools (`customer_base_audit/cli.py`)

---

## Recommended Actions

### Immediate (High Priority)
1. **Issue #40** → Update track assignment from B to C
2. **Issue #39** → Update track assignment from A to C
3. **Issue #34** → Update track assignment from A to B
4. **Issue #31** → Update issue body to reflect Track B (AGENTS.md already corrected)

### Medium Priority
5. **Issue #37** → Update track assignment from A to B
6. **Issue #36** → Update track assignment from B to C

### Add Track Assignments
7. **Issues #78, #75** → Add Track A assignment
8. **Issues #61, #62, #63, #64, #54, #58** → Add Track B assignment

---

## Root Cause Analysis

**Why did these mislabelings occur?**

1. **Issue #31 (Validation)**: Original implementation plan assigned validation to Track A because it was seen as "validation of RFM/Lens results." However, validation framework primarily validates CLV models (BG/NBD, Gamma-Gamma), which are Track B scope.

2. **Issue #34 (Lens 4)**: Mistakenly grouped with Lenses 1-2 (Track A). However, Lens 4 is cohort-centric (depends on Track B's `cohorts.py` and `lens3.py`).

3. **Issues #39, #40**: Documentation and examples incorrectly distributed across Track A and B. All documentation should consolidate in Track C.

4. **Issues #36, #37**: Operational tools (CLI, monitoring) incorrectly assigned to Track A/B. These belong to Track C (operational/integration work).

**Prevention**: Update track assignment checklist to ask:
- Does this issue modify files in `models/**` or `validation/**`? → Track B
- Does this issue modify files in `docs/**` or `examples/**`? → Track C
- Does this issue modify `rfm.py`, `lens1.py`, or `lens2.py`? → Track A
- Does this issue modify `cohorts.py`, `lens3.py`, `lens4.py`, or `lens5.py`? → Track B

---

## Update Commands

```bash
# Fix Issue #40 (API Reference - Track C)
gh issue edit 40 --body "$(gh issue view 40 --json body -q '.body' | sed 's/Track: B (#17)/Track: C (#18)/' | sed 's/.worktrees\/track-b/.worktrees\/track-c/')"

# Fix Issue #39 (Example Notebooks - Track C)
gh issue edit 39 --body "$(gh issue view 39 --json body -q '.body' | sed 's/Track: A (#16)/Track: C (#18)/' | sed 's/.worktrees\/track-a/.worktrees\/track-c/')"

# Fix Issue #37 (Drift Detection - Track B)
gh issue edit 37 --body "$(gh issue view 37 --json body -q '.body' | sed 's/Track: A (#16)/Track: B (#17)/' | sed 's/.worktrees\/track-a/.worktrees\/track-b/')"

# Fix Issue #36 (CLI Batch Processing - Track C)
gh issue edit 36 --body "$(gh issue view 36 --json body -q '.body' | sed 's/Track: B (#17)/Track: C (#18)/' | sed 's/.worktrees\/track-b/.worktrees\/track-c/')"

# Fix Issue #34 (Lens 4 - Track B)
gh issue edit 34 --body "$(gh issue view 34 --json body -q '.body' | sed 's/Track: A (#16)/Track: B (#17)/' | sed 's/.worktrees\/track-a/.worktrees\/track-b/')"

# Fix Issue #31 (Validation Framework - Track B) - already fixed in AGENTS.md
gh issue edit 31 --body "$(gh issue view 31 --json body -q '.body' | sed 's/Track: A (#16)/Track: B (#17)/' | sed 's/.worktrees\/track-a/.worktrees\/track-b/')"
```

---

**Next Steps**:
1. Review and approve this audit
2. Execute update commands to correct track assignments
3. Update AGENTS.md files if needed (Track B already updated for validation and monitoring)
4. Communicate changes to team
