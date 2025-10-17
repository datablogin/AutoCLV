# Phase 4A Tools Quick Reference

## New Tools (Phase 4A)

### 🏥 `health_check`

**Purpose**: Comprehensive system health monitoring

**Usage**:
```
Check the health of the MCP server and all dependencies
```

**Returns**:
- Overall health status (healthy/degraded/unhealthy)
- Individual component checks (MCP server, shared state, foundation data, resources)
- Foundation data availability (transactions, data mart, RFM, cohorts)
- System resource usage (CPU, memory if psutil available)
- Server uptime in seconds

**Example Output**:
```json
{
  "status": "healthy",
  "timestamp": "2025-10-16T23:30:00",
  "checks": {
    "mcp_server": "healthy",
    "shared_state": "healthy",
    "foundation_data": "available (7 items)",
    "system_resources": "healthy"
  },
  "foundation_data_status": {
    "transactions": true,
    "data_mart": true,
    "rfm_metrics": true,
    "rfm_scores": true,
    "period_aggregations": true,
    "cohort_definitions": true,
    "cohort_assignments": true
  },
  "resource_usage": {
    "memory_rss_mb": 125.5,
    "memory_percent": 1.2,
    "cpu_percent": 0.5
  },
  "uptime_seconds": 3600.5
}
```

**When to Use**:
- Before starting analysis workflow
- To diagnose issues (why is my analysis failing?)
- To verify data is loaded
- To monitor server resource usage

---

### 📊 `get_execution_metrics`

**Purpose**: Retrieve performance statistics for all analyses and lenses

**Usage**:
```
Get execution metrics to see how the analyses are performing
```

**Returns**:
- Total number of orchestrated analyses run
- Per-lens execution statistics:
  - Total executions
  - Successful executions
  - Failed executions
  - Average/min/max duration (milliseconds)
  - Success rate percentage
  - Error types and counts
- Overall metrics:
  - Average duration across all analyses
  - Overall success rate
  - Collection uptime

**Example Output**:
```json
{
  "timestamp": "2025-10-16T23:30:00",
  "total_analyses": 5,
  "overall_avg_duration_ms": 250.5,
  "overall_success_rate_pct": 95.0,
  "uptime_seconds": 3600.5,
  "lens_metrics": {
    "lens1": {
      "total_executions": 3,
      "successful_executions": 3,
      "failed_executions": 0,
      "avg_duration_ms": 150.0,
      "min_duration_ms": 120.0,
      "max_duration_ms": 180.0,
      "success_rate_pct": 100.0,
      "error_types": {}
    },
    "lens5": {
      "total_executions": 2,
      "successful_executions": 1,
      "failed_executions": 1,
      "avg_duration_ms": 350.0,
      "min_duration_ms": 300.0,
      "max_duration_ms": 400.0,
      "success_rate_pct": 50.0,
      "error_types": {
        "ValueError": 1
      }
    }
  }
}
```

**When to Use**:
- After running analyses to check performance
- To identify slow lenses
- To track error patterns
- To monitor success rates over time
- For performance optimization
- For debugging recurring failures

**Insights You Can Get**:
- "Which lens is slowest?" → Check `avg_duration_ms` for each lens
- "Are my analyses succeeding?" → Check `overall_success_rate_pct`
- "What errors are occurring?" → Check `error_types` for each lens
- "How long has the server been running?" → Check `uptime_seconds`

---

### 🔄 `reset_execution_metrics`

**Purpose**: Clear all collected metrics and restart tracking from zero

**Usage**:
```
Reset the execution metrics to start fresh tracking
```

**Returns**:
```json
{
  "status": "metrics reset successfully",
  "timestamp": "2025-10-16T23:30:00",
  "message": "All execution metrics have been cleared and restarted from zero"
}
```

**When to Use**:
- Before starting a new test run
- After making performance improvements (to measure impact)
- To clear metrics from a previous session
- When metrics have accumulated over long period

**Note**: This only resets metrics, not the actual data or health status

---

## Behind-the-Scenes Features

### 🔁 Automatic Retry Logic

**What it does**: Automatically retries failed lens executions

**Configuration**:
- **Max Attempts**: 3 retries
- **Backoff Strategy**: Exponential (2s min, 10s max)
- **Retry On**: TimeoutError, ConnectionError, RuntimeError
- **No Retry On**: ValueError, KeyError (permanent failures)

**Example Behavior**:
```
Attempt 1: Fails with TimeoutError → Wait 2 seconds
Attempt 2: Fails with TimeoutError → Wait 4 seconds
Attempt 3: Succeeds → Return result
```

**You don't need to do anything** - retry logic works automatically when you run analyses!

---

### 📈 OpenTelemetry Tracing

**What it does**: Creates distributed traces for all lens executions

**Captured Information**:
- Span for overall orchestrated analysis
- Nested spans for each lens execution
- Span attributes:
  - Query text
  - Lenses executed/failed
  - Execution time
  - Customer counts
  - Health scores
  - Error types

**Viewing Traces**: Currently in-memory only. Phase 4B will add export to Jaeger/Zipkin for visualization.

---

## Common Workflows

### Workflow 1: Pre-Analysis Health Check

```
1. health_check
2. If status is healthy and data is loaded → proceed
3. If status is degraded → load_transactions first
4. If status is unhealthy → investigate component failures
```

### Workflow 2: Performance Monitoring

```
1. reset_execution_metrics (start fresh)
2. Run your analyses (run_orchestrated_analysis multiple times)
3. get_execution_metrics (review performance)
4. Identify slow lenses or high failure rates
5. Optimize or investigate issues
```

### Workflow 3: Debugging Failures

```
1. Run analysis that's failing
2. get_execution_metrics (check which lens failed and error type)
3. health_check (verify foundation data is available)
4. Review error_types in metrics for specific error messages
5. Fix the issue (load data, check input parameters, etc.)
```

### Workflow 4: Regular Monitoring

```
Every hour/day:
1. health_check → Verify system is healthy
2. get_execution_metrics → Check success rates and performance
3. If success rate drops below 90% → Investigate
4. If avg duration increases significantly → Investigate performance
```

---

## Tips & Best Practices

### 🎯 When to Use Each Tool

| Situation | Tool to Use |
|-----------|-------------|
| Starting a new analysis session | `health_check` |
| Analysis failed unexpectedly | `health_check` + `get_execution_metrics` |
| Want to track performance | `get_execution_metrics` before/after |
| Starting performance testing | `reset_execution_metrics` first |
| Debugging why lenses are slow | `get_execution_metrics` (check durations) |
| Investigating errors | `get_execution_metrics` (check error_types) |

### 🚀 Performance Expectations

**Typical Durations** (with loaded data):
- Lens 1 (Snapshot): 100-200ms
- Lens 5 (Overall Health): 300-500ms
- Overall Analysis: 250-750ms (depending on lenses)

**Success Rates**:
- With data loaded: >95% success rate
- Without data loaded: Lenses will fail with ValueError

**Resource Usage**:
- Memory: ~100-200 MB for typical datasets
- CPU: <5% during idle, 20-50% during analysis

### ⚠️ Common Issues

**Issue**: `health_check` shows "no data loaded"
- **Solution**: Run `load_transactions` first

**Issue**: All lenses failing with ValueError
- **Solution**: Foundation data not available - load transactions

**Issue**: High error rate in `get_execution_metrics`
- **Solution**: Check `error_types` to identify root cause

**Issue**: Slow lens execution times
- **Solution**: Check dataset size - larger datasets take longer

---

## Integration with Existing Tools

Phase 4A tools work seamlessly with existing tools:

```
Traditional Workflow:
1. load_transactions
2. run_orchestrated_analysis
3. Done

Enhanced Workflow (with Phase 4A):
1. health_check (verify ready)
2. load_transactions (if needed)
3. run_orchestrated_analysis
4. get_execution_metrics (check performance)
5. health_check (verify still healthy)
```

---

## Version Info

- **MCP Server**: v1.2.0
- **Phase**: 4A Complete
- **Total Tools**: 14 (11 existing + 3 new)
- **Dependencies Added**: tenacity>=9.0.0

---

## Next Phase (4B) - Deferred

Phase 4B will add:
- OTLP export for traces (Jaeger/Zipkin)
- Circuit breakers for resilience
- Prometheus metrics export
- Enhanced health checks (liveness/readiness probes)

See [Issue #118](https://github.com/datablogin/AutoCLV/issues/118) for details.
