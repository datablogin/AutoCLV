# Phase 4A Features Test Prompt for Claude Desktop

This document provides a comprehensive test prompt to verify all Phase 4A observability and resilience features in Claude for Desktop.

## Prerequisites

1. Ensure Claude Desktop is configured with the MCP server (should be automatic)
2. The MCP server should show version 1.2.0 with Phase 4A features
3. You'll need a transactions CSV file for testing (or use the provided sample data)

## Comprehensive Test Prompt

Copy and paste the following prompt into Claude for Desktop:

---

**Test Prompt:**

```
I'd like to test all the Phase 4A observability features of the Four Lenses Analytics MCP server. Please help me run through the following comprehensive test:

## Part 1: Health Check & Server Status

1. Run the `health_check` tool to verify the MCP server is healthy and operational
   - Check what foundation data is currently available
   - Review system resource usage if available
   - Verify all component health statuses

## Part 2: Load Sample Data & Build Foundation

2. Load sample transaction data:
   - Use `load_transactions` with this sample CSV path: `/Users/robertwelborn/PycharmProjects/AutoCLV/tests/fixtures/synthetic_transactions_2023_2024.csv`
   - Verify the data loaded successfully

3. After loading, run `health_check` again to see:
   - How foundation data status has changed
   - What data is now available (transactions, data mart, RFM, cohorts)

## Part 3: Execute Orchestrated Analysis

4. Run an orchestrated analysis to test the metrics collection:
   - Use `run_orchestrated_analysis` with query: "Give me the overall customer base health"
   - This should trigger Lens 5 execution and collect metrics

5. Run another analysis:
   - Query: "Customer health snapshot"
   - This should trigger Lens 1 execution

6. Run a multi-lens analysis:
   - Query: "customer health and overall base health"
   - This should execute both Lens 1 and Lens 5 in parallel

## Part 4: Check Execution Metrics

7. Use `get_execution_metrics` to retrieve performance statistics:
   - Review total number of analyses run
   - Check per-lens execution statistics (count, success rate, avg duration)
   - Examine any error types that occurred
   - Review overall success rate and average duration

8. Analyze the metrics:
   - Which lens has the longest average execution time?
   - What's the overall success rate?
   - Are there any failed executions?

## Part 5: Test Error Scenarios (Optional)

9. Try running an analysis without data:
   - First, use `reset_execution_metrics` to clear current metrics
   - Run `run_orchestrated_analysis` with query: "customer snapshot"
   - Check `get_execution_metrics` to see the failure recorded

## Part 6: Verify Retry Logic

10. The retry logic is working behind the scenes:
    - Retries happen automatically on TimeoutError, ConnectionError, RuntimeError
    - With exponential backoff (2s-10s)
    - Up to 3 attempts per lens

## Part 7: Final Health Check

11. Run final `health_check` to see:
    - Server uptime
    - Complete system status
    - All component health

## Expected Behavior

After running all these tests, you should see:

✅ **Health Check**:
- Status: healthy or degraded (depending on data availability)
- Foundation data status showing what's loaded
- Resource usage metrics (if psutil available)

✅ **Execution Metrics**:
- Multiple analyses recorded
- Per-lens statistics showing execution counts and durations
- Success rates (should be high if data is loaded)
- Zero or minimal failed executions

✅ **Orchestrated Analysis**:
- Successful lens executions with insights
- Parallel execution for independent lenses
- Error tracking with specific error messages if failures occur

Please run through this comprehensive test and report:
1. Any tools that don't work as expected
2. The execution metrics summary
3. Overall system health status
4. Any interesting insights from the analyses
```

---

## Quick Test Prompt (Abbreviated Version)

If you want a shorter test, use this:

```
Test the Phase 4A observability features:

1. Run `health_check` - verify server is healthy
2. Load data: `load_transactions` with path `/Users/robertwelborn/PycharmProjects/AutoCLV/tests/fixtures/synthetic_transactions_2023_2024.csv`
3. Run analysis: `run_orchestrated_analysis` with query "overall customer base health"
4. Check metrics: `get_execution_metrics` - see performance stats
5. Run `health_check` again - verify updated status

Report the health status, metrics summary, and any issues.
```

---

## Expected Tools Available

The MCP server should expose these tools (14 total):

### Foundation Tools
- `load_transactions` - Load transaction data from CSV
- `build_customer_data_mart` - Build customer data mart
- `calculate_rfm_metrics` - Calculate RFM metrics
- `create_customer_cohorts` - Create customer cohorts

### Lens Tools
- `analyze_single_period_snapshot` - Lens 1: Current period snapshot
- `analyze_period_to_period_comparison` - Lens 2: Period comparison (placeholder)
- `analyze_cohort_lifecycle` - Lens 3: Cohort lifecycle (placeholder)
- `compare_multiple_cohorts` - Lens 4: Multi-cohort comparison
- `assess_overall_customer_base_health` - Lens 5: Overall health assessment

### Orchestration Tools
- `run_orchestrated_analysis` - Natural language orchestrated analysis

### Phase 4A Observability Tools (NEW!)
- `health_check` ⭐ - Comprehensive system health check
- `get_execution_metrics` ⭐ - Retrieve performance metrics
- `reset_execution_metrics` ⭐ - Reset metrics (useful for testing)

## Troubleshooting

If tools don't appear or aren't working:

1. **Restart Claude Desktop** - This ensures it picks up the updated MCP configuration
2. **Check MCP Server Logs** - Look in stderr output for any initialization errors
3. **Verify Python Environment** - Ensure all dependencies are installed (`tenacity`, `structlog`, etc.)
4. **Check File Paths** - Ensure the sample data file exists at the specified path

## Sample Data Location

The test prompt uses this file:
```
/Users/robertwelborn/PycharmProjects/AutoCLV/tests/fixtures/synthetic_transactions_2023_2024.csv
```

If this doesn't exist, you can use any CSV with these columns:
- `customer_id`: Customer identifier
- `order_date`: Date of transaction (YYYY-MM-DD format)
- `order_value`: Transaction amount

## Success Criteria

✅ All 14 tools are visible in Claude Desktop
✅ `health_check` returns healthy status with detailed component checks
✅ `get_execution_metrics` shows collected statistics after analyses
✅ `run_orchestrated_analysis` executes lenses and records metrics
✅ Metrics show accurate execution counts, durations, and success rates
✅ Health check shows updated foundation data status after loading

## Version Information

- **MCP Server Version**: 1.2.0
- **Phase**: 4A - Essential Observability & Resilience
- **Tools Count**: 14 (11 existing + 3 new observability tools)
- **New Features**:
  - Retry logic with tenacity (automatic, behind the scenes)
  - Comprehensive health checks with resource monitoring
  - In-memory execution metrics collection
  - OpenTelemetry tracing integration (traces visible in spans)

---

## Next Steps After Testing

Once you've verified all features work:

1. **Use in Real Workflows**: Incorporate health checks and metrics into regular analysis workflows
2. **Monitor Performance**: Use `get_execution_metrics` to track lens performance over time
3. **Debug Issues**: Use health checks to diagnose problems with foundation data
4. **Track Trends**: Monitor success rates and execution times to identify patterns

Enjoy testing the new Phase 4A observability features! 🎉
