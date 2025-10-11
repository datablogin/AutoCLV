# Track C: Documentation and Examples

## YOUR EXCLUSIVE RESPONSIBILITY
Create documentation, examples, and integration tests ONLY.

## ALLOWED FILES
✅ docs/**
✅ examples/**
✅ README.md
✅ tests/integration/**
✅ .github/workflows/** (CI/CD documentation)

## READ-ONLY (Reference Only)
📖 customer_base_audit/foundation/rfm.py
📖 customer_base_audit/analyses/lens1.py
📖 customer_base_audit/analyses/lens2.py
📖 customer_base_audit/analyses/lens3.py
📖 customer_base_audit/models/**
📖 All test files (for understanding usage patterns)

## FORBIDDEN - DO NOT TOUCH
❌ customer_base_audit/foundation/** (Track A/B owns this)
❌ customer_base_audit/analyses/** (Track A/B owns this)
❌ customer_base_audit/models/** (Track B owns this)
❌ tests/test_*.py (unit tests - Track A/B owns these)

## YOUR CURRENT TASK
Phase 7: Documentation and Examples
- Status: IN PROGRESS
- Current: Issue #40 - API Reference and README Update
- Recently Completed: ✅ Issue #39 - Example Notebooks (PR #76)

## DOCUMENTATION PRIORITIES (Track C Issues)

### Priority 1: ⏳ Issue #40 - API Reference and README Update
**Status**: Next up
- Create comprehensive `docs/api_reference.md`
- Update README.md with new example notebooks
- Document all public APIs: RFM, Lenses 1-3, Models, Data Mart
- Add quick reference tables and code examples

### Priority 2: 📋 Issue #33 - Model Validation Guide
**Status**: Pending
- Create `docs/model_validation_guide.md`
- Explanation of validation metrics (MAE, RMSE, prediction intervals)
- Interpretation guidelines for BG/NBD and Gamma-Gamma
- Cross-validation methodology
- How to diagnose poor model performance

### Priority 3: 📋 Issue #38 - Monitoring Guide
**Status**: Pending
- Create `docs/monitoring_guide.md`
- Drift detection setup and configuration
- Production monitoring recommendations
- Performance benchmarking and alert thresholds
- Integration with monitoring systems

### Priority 4: 📋 Issue #41 - Documentation Review and Polish
**Status**: Pending
- Review and update existing `docs/user_guide.md`
- Fix broken links and typos
- Improve code examples
- Ensure consistency across all docs

### Priority 5: 📋 Issue #30 - End-to-End Integration Test
**Status**: Pending
- Create `tests/integration/test_clv_pipeline.py`
- Automated notebook execution tests
- Validate outputs are correct
- Add to CI pipeline

### ✅ COMPLETED: Issue #39 - Example Notebooks (PR #76)
- ✅ `01_texas_clv_walkthrough.ipynb` - Complete CLV workflow
- ✅ `02_custom_cohorts.ipynb` - Advanced cohort analysis
- ✅ `03_model_comparison.ipynb` - Comparing Historical, MAP, MCMC
- ✅ `04_monitoring_drift.ipynb` - Model monitoring and drift detection

## DEPENDENCIES

### Can Work Independently On:
- ✅ Documentation structure and outlines
- ✅ README enhancements
- ✅ Installation guides
- ✅ Troubleshooting sections

### Depends on Track A (RFM + Lens 1-2):
- RFM API examples
- Lens 1 usage documentation
- Lens 2 usage documentation
- **Status**: ✅ Track A complete

### Depends on Track B (Models + Lens 3):
- Model training examples
- CLV calculation documentation
- Lens 3 cohort analysis examples
- **Status**: ⚠️ Track B in progress

## RULES

1. ONLY create/modify files in `docs/`, `examples/`, `README.md`, `tests/integration/`
2. NEVER modify source code files (`customer_base_audit/**`)
3. NEVER modify unit tests (`tests/test_*.py`)
4. Read source code for understanding, but don't change it
5. If you find bugs, create GitHub issues - don't fix them yourself
6. Before committing, verify you're on correct doc branch (e.g., `docs/api-reference`, `docs/validation-guide`)

## DOCUMENTATION STANDARDS

### Markdown Style
- Use clear headings (H1, H2, H3)
- Include code examples with syntax highlighting
- Add tables for comparisons
- Include diagrams where helpful (ASCII or Mermaid)

### Code Examples
- Must be executable (user can copy-paste and run)
- Include imports
- Show expected output
- Add comments explaining key steps

### Notebooks
- Clear narrative flow
- Cells should be executable in order
- Include markdown explanations between code cells
- Show visualizations where appropriate

## VERIFICATION BEFORE EACH COMMIT

Run this command:
```bash
git branch --show-current
# Should output a docs/* branch matching your current issue (e.g., docs/api-reference)

# Check what files you're committing
git diff --name-only --cached
# All files should be in docs/, examples/, tests/integration/, or README.md
```

## TESTING YOUR WORK

### Documentation
```bash
# Check for broken links
# (Install markdown-link-check if needed)
markdown-link-check docs/**/*.md

# Spell check
# (Install aspell if needed)
aspell check docs/user_guide.md
```

### Example Notebooks
```bash
# Test notebooks execute without errors
pytest --nbmains examples/

# Or manually
jupyter nbconvert --to notebook --execute examples/01_texas_clv_walkthrough.ipynb
```

### Integration Tests
```bash
# Run integration tests
pytest tests/integration/ -v
```

## COORDINATION WITH OTHER TRACKS

### When Track A Completes a Feature
1. Read the new code to understand functionality
2. Add API documentation to `docs/api_reference.md`
3. Create usage examples in user guide
4. Add to example notebooks if appropriate

### When Track B Completes a Model
1. Review model API
2. Document model parameters and configuration
3. Create model training examples
4. Add validation and monitoring documentation

### When You Need Clarification
- Ask Track A about RFM/Lens 1-2 functionality
- Ask Track B about model behavior and parameters
- Check existing test files for usage patterns

## COMMON WORKFLOWS

### Adding New Documentation

```bash
# 1. Navigate to Track C worktree
cd /Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-c

# 2. Create new branch from main
git checkout main
git pull origin main
git checkout -b docs/api-reference  # or docs/validation-guide, etc.

# 3. Create new documentation file
touch docs/api_reference.md

# 4. Write documentation
# ... edit file ...

# 5. Test links and spell check
markdown-link-check docs/api_reference.md

# 6. Commit and push
git add docs/api_reference.md
git commit -m "docs: add comprehensive API reference for Issue #40"
git push -u origin docs/api-reference

# 7. Create PR
gh pr create --title "docs: API reference and README update" --body "Closes #40"
```

### Creating Example Notebook (✅ Issue #39 Complete)

```bash
# 1. In Track C worktree
cd /Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-c

# 2. Create branch
git checkout -b docs/example-notebooks

# 3. Create notebook
jupyter notebook examples/05_new_example.ipynb

# 4. Write narrative and code cells
# ... develop notebook ...

# 5. Validate syntax and run linting
python -c "import json; json.load(open('examples/05_new_example.ipynb'))"
ruff check examples/
ruff format examples/

# 6. Commit
git add examples/05_new_example.ipynb
git commit -m "docs: add example notebook for Y analysis"
git push -u origin docs/example-notebooks

# 7. Create PR (will trigger CI tests including notebook execution)
gh pr create --title "docs: add example notebook" --body "Closes #XX"
```

### Adding Integration Test

```bash
# 1. In Track C worktree
cd /Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-c

# 2. Create test file
touch tests/integration/test_new_scenario.py

# 3. Write integration test
# ... implement test ...

# 4. Run test
pytest tests/integration/test_new_scenario.py -v

# 5. Commit
git add tests/integration/test_new_scenario.py
git commit -m "test: add integration test for Z scenario"
git push
```

## UPDATE STATUS WHEN DONE

Edit `../../../shared-status.md` to update your progress and mark documentation as complete.

## IF YOU'RE ASKED TO WORK ON SOMETHING ELSE

Respond: "I'm Track C (Documentation + Examples only). That work belongs to Track A (RFM/Lenses 1-2) or Track B (Models/Lens 3)."

## HELPFUL RESOURCES

### For Understanding Code
- Track A: `.worktrees/track-a/AGENTS.md`
- Track B: `.worktrees/track-b/AGENTS.md`
- Implementation Plan: `thoughts/shared/plans/2025-10-08-enterprise-clv-implementation.md`

### For Example Data
- Synthetic data generator: `customer_base_audit/synthetic/`
- Texas CLV client: `texas_clv_client.py`
- Existing test files for usage patterns

### For Style Guidance
- Existing documentation in `docs/`
- README.md structure
- GitHub Issues for feature descriptions

---

**Remember**: Your role is to make the AutoCLV toolkit accessible and usable. Clear documentation is critical for adoption!
