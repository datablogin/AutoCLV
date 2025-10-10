# Track A: RFM and Lenses 1-2

## YOUR EXCLUSIVE RESPONSIBILITY
Implement RFM calculations and Lens 1-2 analyses ONLY.

## ALLOWED FILES
✅ customer_base_audit/foundation/rfm.py
✅ customer_base_audit/analyses/lens1.py
✅ customer_base_audit/analyses/lens2.py
✅ tests/test_rfm.py
✅ tests/test_lens1.py
✅ tests/test_lens2.py

## READ-ONLY (Reference Only)
📖 customer_base_audit/foundation/data_mart.py
📖 customer_base_audit/foundation/customer_contract.py

## FORBIDDEN - DO NOT TOUCH
❌ customer_base_audit/analyses/lens3.py (Track B owns this)
❌ customer_base_audit/models/** (Track B owns this)
❌ docs/** (Track C owns this)
❌ examples/** (Track C owns this)

## YOUR CURRENT TASK
Phase 2: Lens 2 (Period-to-Period Comparison)
- Status: Appears COMPLETE based on commits
- Next: Wait for merge before starting new work

## RULES
1. ONLY work in files listed under "ALLOWED FILES"
2. NEVER create or modify files in customer_base_audit/models/
3. NEVER modify documentation files
4. Run tests frequently: `make test`
5. Before committing, verify you're on branch: feature/track-a-rfm-lenses

## VERIFICATION BEFORE EACH COMMIT
Run this command:
```bash
git branch --show-current
# Should output: feature/track-a-rfm-lenses

# Check what files you're committing
git diff --name-only --cached
# All files should be in your ALLOWED FILES list above
```

## UPDATE STATUS WHEN DONE
Edit ../../../shared-status.md to update your progress and mark work as complete.

## IF YOU'RE ASKED TO WORK ON SOMETHING ELSE
Respond: "I'm Track A (RFM + Lenses 1-2 only). That work belongs to Track B (models) or Track C (docs)."
