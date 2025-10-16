# Claude Desktop MCP Setup Guide

## Testing Four Lenses MCP Tools with Claude Desktop

This guide explains how to configure Claude Desktop to connect to your local Four Lenses MCP server and test the Phase 2 lens implementations.

### Prerequisites

1. **Claude Desktop app** installed on your Mac
2. **Python virtual environment** activated (`source .venv/bin/activate`)
3. **Synthetic data** generated (already done: `synthetic_transactions.json`)

---

## Step 1: Configure Claude Desktop

Claude Desktop reads its MCP server configuration from:
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

### Create or Update Configuration

1. Open Terminal and edit the configuration file:

```bash
code ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

2. Add the Four Lenses MCP server configuration:

```json
{
  "mcpServers": {
    "four-lenses-analytics": {
      "command": "/Users/robertwelborn/PycharmProjects/AutoCLV/.venv/bin/python",
      "args": [
        "-m",
        "analytics.services.mcp_server.main"
      ],
      "cwd": "/Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a",
      "env": {
        "PYTHONPATH": "/Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a"
      }
    }
  }
}
```

**Important**: Adjust the paths if your installation is in a different location!

---

## Step 2: Restart Claude Desktop

After saving the configuration:

1. **Quit Claude Desktop** completely (Cmd+Q)
2. **Reopen Claude Desktop**
3. The MCP server should automatically connect

---

## Step 3: Test the MCP Tools

### Phase 1 Foundation Tools

Try these commands in Claude Desktop:

1. **Health Check**:
   ```
   Use the health_check tool to verify the MCP server is running
   ```

2. **Build Data Mart**:
   ```
   Use the build_customer_data_mart tool with this data:
   - transaction_data_path: /Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a/synthetic_transactions.json
   - period_granularities: ["quarter", "year"]
   ```

3. **Calculate RFM**:
   ```
   Use calculate_rfm_metrics with:
   - observation_end: 2024-06-30
   - calculate_scores: true
   ```

4. **Create Cohorts**:
   ```
   Use create_customer_cohorts with:
   - cohort_type: "quarterly"
   ```

### Phase 2 Lens Tools

Now test the four lens analyses:

#### **Lens 1: Single Period Snapshot**
```
Use analyze_single_period_snapshot with:
- period_name: "Q2 2024"
```

Expected output:
- Total customers
- One-time buyer percentage
- Revenue concentration (Pareto analysis)
- Customer health score (0-100)
- Concentration risk assessment
- Actionable recommendations

#### **Lens 2: Period-to-Period Comparison**

First, you'll need RFM for two periods. Ask Claude:
```
I want to compare Q1 2024 vs Q2 2024. Can you:
1. Calculate RFM for Q1 2024 (observation_end: 2024-03-31) and store as "rfm_metrics"
2. Calculate RFM for Q2 2024 (observation_end: 2024-06-30) and store as "rfm_metrics_period2"
3. Then use analyze_period_to_period_comparison with period1_name="Q1 2024" and period2_name="Q2 2024"
```

Expected output:
- Retention, churn, and reactivation rates
- Customer migration metrics
- Revenue growth analysis
- Growth momentum assessment

#### **Lens 3: Single Cohort Evolution**
```
Use analyze_cohort_lifecycle with:
- cohort_id: <one of the cohort IDs from create_customer_cohorts>
```

Expected output:
- Cohort size and periods analyzed
- Activation, revenue, and retention curves
- Cohort maturity assessment (early/growth/mature/declining)
- LTV trajectory (strong/moderate/weak)
- Cohort-specific recommendations

#### **Lens 4: Multi-Cohort Comparison**
```
Use compare_multiple_cohorts with:
- alignment_type: "left-aligned"
- include_margin: false
```

Expected output:
- Summary for each cohort
- Best and worst performing cohorts
- Key performance differences
- Time-to-second-purchase insights
- Comparative recommendations

---

## Step 4: Full Workflow Test

Try running the complete analysis pipeline:

```
Please run a complete Four Lenses analysis on my synthetic data:

1. Build the data mart from synthetic_transactions.json with quarterly and yearly granularities
2. Calculate RFM metrics for the most recent period (2024-06-30)
3. Create quarterly cohorts
4. Run all four lens analyses:
   - Lens 1: Current period snapshot
   - Lens 3: Evolution of the earliest cohort
   - Lens 4: Comparison of all cohorts
5. Summarize the key insights and recommendations

Data file path: /Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a/synthetic_transactions.json
```

---

## Recent Bug Fix: Context State Persistence

**Issue**: The MCP server was not maintaining state between tool calls. Each tool invocation received a fresh Context, causing "data mart not found" errors when trying to use RFM or lens tools.

**Fix**: Implemented a global `SharedState` singleton that persists data across tool calls:
- All foundation and lens tools now use `get_shared_state()` instead of `ctx.get_state()`
- Data (data mart, RFM metrics, cohorts, lens results) persists across tool invocations
- No configuration changes needed - just restart Claude Desktop

## Troubleshooting

### MCP Server Not Connecting

1. **Check logs**: Claude Desktop logs are in `~/Library/Logs/Claude/`
2. **Verify paths**: Ensure all paths in `claude_desktop_config.json` are absolute and correct
3. **Test manually**: Run the server standalone:
   ```bash
   cd /Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a
   source .venv/bin/activate
   python -m analytics.services.mcp_server.main
   ```

### Tool Not Found

- Make sure all Phase 2 lens tools are imported in `analytics/services/mcp_server/main.py`
- Check that `logger.info` shows "tools_registered=7" (3 foundation + 4 lens tools)

### Context State Errors

- MCP tools maintain state in context (data mart, RFM, cohorts)
- If you get "not found" errors, run the prerequisite tools first
- Example: Lens 1 requires RFM metrics → run `calculate_rfm_metrics` first

### Permission Errors

- Ensure the Python virtual environment has permissions to read the synthetic data file
- Try using absolute paths instead of relative paths

---

## Next Steps

Once you've verified Phase 2 works with Claude Desktop:

1. **Test different scenarios**: Use the various scenario configs from `customer_base_audit.synthetic`
2. **Compare cohorts**: Analyze how different acquisition cohorts perform
3. **Track changes over time**: Run Lens 2 to see period-to-period dynamics

---

## Synthetic Data Details

**Current dataset** (`synthetic_transactions.json`):
- 100 customers
- 1,410 orders
- 18 months of transaction history (Jan 2023 - Jun 2024)
- ~$128K total revenue
- Baseline scenario (no special events)

To generate new data with different characteristics, edit `generate_synthetic_test_data.py` and change:
- Number of customers (`n=100`)
- Date ranges
- Scenario (try `HIGH_CHURN_SCENARIO`, `HEAVY_PROMOTION_SCENARIO`, etc.)
