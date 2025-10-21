# Phased Implementation Plan: Agentic Five Lenses Architecture

**Date**: 2025-10-14
**Last Updated**: 2025-10-16 Evening
**Status**: Phase 4A Complete - Essential Observability & Resilience Implemented
**Related Research**: [Agentic Five Lenses Architecture Research](../research/2025-10-14-agentic-five-lenses-architecture.md)

---

## Executive Summary

This plan outlines the phased migration from AutoCLV's current analytics architecture to an agentic orchestration layer while preserving the proven core analytics engine (384 passing tests). The approach uses **LangGraph** for orchestration, **MCP protocol** for standardized communication, and wraps existing **Five Lenses** functions as stateless agent services.

**UPDATE (2025-10-16)**: All 5 lenses are now fully implemented! Lens 5 (Overall Customer Base Health) was completed with 905 lines of production code including comprehensive health scoring, cohort analysis, and actionable recommendations.

**Key Principles**:
- **Wrap, don't rewrite**: Keep all existing lens calculations as pure Python
- **LLMs orchestrate, don't calculate**: Use AI for query interpretation and result synthesis
- **Incremental migration**: Each phase delivers value independently
- **Preserve quality**: Maintain 100% test coverage throughout

**Timeline**: 5 weeks (adjustable based on priorities)
**Risk Level**: Low (hybrid architecture preserves existing functionality)

---

## Architecture Overview

### Current Architecture (Track A - 100% Complete) ✅

```
Raw Transactions
       ↓
CustomerDataMart (orders + period aggregations)
       ↓
    ┌──┴──────────────────┐
    │                     │
RFMMetrics          PeriodAggregation + Cohorts
    │                     │
    ├─→ Lens 1           ├─→ Lens 3
    ├─→ Lens 2           ├─→ Lens 4
    │                    └─→ Lens 5 ✅
    │
```

**Characteristics**:
- Pure Python functions
- Immutable dataclass results
- No side effects (except logging)
- Fully tested (384+ tests)
- Parallel processing support
- **All 5 lenses implemented** ✅

### Target Architecture (Agentic Layer + Core)

```
┌─────────────────────────────────────────────┐
│         Agentic Orchestration Layer         │
│                                             │
│  ┌──────────────────────────────────────┐  │
│  │    Query Interpreter (LLM)           │  │
│  │    - Natural language → intent       │  │
│  │    - Parameter extraction            │  │
│  └──────────────┬───────────────────────┘  │
│                 ↓                           │
│  ┌──────────────────────────────────────┐  │
│  │  LangGraph Coordinator StateGraph    │  │
│  │  - Dynamic workflow composition      │  │
│  │  - Parallel lens execution           │  │
│  │  - Error handling & retry            │  │
│  └──────────────┬───────────────────────┘  │
│                 ↓                           │
│  ┌──────────────────────────────────────┐  │
│  │    Result Synthesizer (LLM)          │  │
│  │    - Multi-lens aggregation          │  │
│  │    - Insight generation              │  │
│  └──────────────────────────────────────┘  │
└─────────────────┬───────────────────────────┘
                  │ MCP Protocol
                  ↓
┌─────────────────────────────────────────────┐
│         MCP Service Layer ✅                │
│                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Foundation│  │ Lens     │  │ Lens 5   │  │
│  │ Services  │  │ 1-4      │  │ Service  │  │
│  │ (Data     │  │ Services │  │ ✅       │  │
│  │  Mart ✅  │  │ ✅       │  │          │  │
│  │  RFM ✅   │  │          │  │          │  │
│  │  Cohorts✅│  │          │  │          │  │
│  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────┬───────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────┐
│    Existing Analytical Core (No Changes)   │
│                                             │
│  - RFM calculations ✅                      │
│  - Lenses 1-5 (proven implementations) ✅  │
│  - Data mart builders ✅                    │
│  - Cohort assignment ✅                     │
│  - 384+ passing tests ✅                    │
└─────────────────────────────────────────────┘
```

**Key Components**:
1. **Agentic Layer** (TODO - Phase 3-5): Query interpretation, orchestration, synthesis
2. **MCP Services** (✅ COMPLETE - Phase 0-2): Stateless services exposing existing functions
3. **Core Analytics** (✅ COMPLETE): Proven calculation engine with all 5 lenses

---

## Implementation Status (2025-10-16)

### ✅ Phases 0-2: COMPLETE (MCP Infrastructure + All Lens Services)

**Phase 0: Foundation Setup** ✅
- MCP server infrastructure operational
- FastMCP framework integrated
- Basic observability with structlog configured
- Shared state management implemented (`state.py`)

**Phase 1: Foundation Services** ✅
- `tools/data_mart.py`: CustomerDataMart builder wrapped as MCP tool
- `tools/rfm.py`: RFM calculation service with parallel processing support
- `tools/cohorts.py`: Cohort creation and assignment service
- All foundation services tested and operational

**Phase 2: Lens Services** ✅ **INCLUDING LENS 5!**
- `tools/lens1.py`: Single-period snapshot analysis (health scoring, concentration risk)
- `tools/lens2.py`: Period-to-period comparison (retention, churn, growth)
- `tools/lens3.py`: Single cohort evolution analysis
- `tools/lens4.py`: Multi-cohort comparison analysis
- `tools/lens5.py`: **Overall customer base health** ✅ (905 lines)
  - Comprehensive health scoring (0-100 + A-F grading)
  - Cohort revenue contribution analysis (C3 data)
  - Repeat behavior metrics by cohort
  - Cohort quality trends (improving/stable/declining)
  - Revenue predictability and acquisition dependence metrics
  - Actionable insights and recommendations generation

**Key Achievement**: All 5 lenses are now wrapped as MCP tools with rich response models, insight generation, and comprehensive error handling.

### ✅ Phase 3: LangGraph Coordinator - **COMPLETE**

**Status**: Implemented and tested successfully (2025-10-16)
**Implementation Time**: 3-4 hours (faster than planned 3-4 days)

**What Was Delivered**:
1. ✅ Created `orchestration/coordinator.py` with LangGraph StateGraph
2. ✅ Defined `AnalysisState` TypedDict for workflow state management
3. ✅ Implemented rule-based intent parsing (MVP approach)
4. ✅ Built scatter-gather pattern for parallel lens execution (Lens 1, 3, 4, 5)
5. ✅ Added result synthesis and aggregation logic
6. ✅ Created `tools/orchestrated_analysis.py` MCP tool
7. ✅ Added comprehensive test suite (8 tests, all passing)

**Key Features**:
- Parallel execution of independent lenses (Lens 1, 3, 4, 5)
- Sequential execution of Lens 2 (depends on Lens 1)
- Graceful error handling with partial results
- Foundation data readiness checks
- Execution time tracking and reporting

### ✅ Phase 4A: Essential Observability & Resilience - **COMPLETE**

**Implementation Date**: 2025-10-16 Evening
**Status**: Phase 4A (Essential features) implemented and tested

**Completed Features**:
- ✅ Basic structlog logging in all MCP tools
- ✅ Progress reporting via `ctx.report_progress()`
- ✅ Shared state management for context passing
- ✅ Automatic retry logic with exponential backoff (tenacity)
  - 3 retries with exponential backoff (2s min, 10s max)
  - Retries on TimeoutError, ConnectionError, RuntimeError
  - Integrated into coordinator lens execution
- ✅ Health check MCP tool (`health_check`)
  - System health monitoring (MCP server, shared state, foundation data)
  - Optional resource usage tracking (CPU, memory if psutil available)
  - Returns detailed health status with component checks
- ✅ Execution metrics MCP tool (`get_execution_metrics`, `reset_execution_metrics`)
  - Per-lens execution statistics (count, success rate, duration)
  - Overall analysis metrics (total analyses, success rate, avg duration)
  - Error type tracking for failed executions
  - In-memory metrics with thread-safe collection
- ✅ OpenTelemetry tracing integration
  - Distributed tracing in coordinator's analyze() method
  - Span attributes for lens execution (customer count, health scores, etc.)
  - Nested spans for calculation and insight generation steps
- ✅ Comprehensive test suite (14 new tests, all passing)
  - Metrics collector unit tests
  - Retry logic verification
  - Tracing integration tests
  - 435 total tests passing across entire project

**Deferred to Phase 4B (Advanced Observability)**:
- Production OpenTelemetry with OTLP export (Jaeger/Zipkin)
- Circuit breaker pattern for external dependencies (pybreaker)
- Advanced distributed tracing instrumentation
- Metrics export to Prometheus/other collectors

**Priority**: Phase 4A COMPLETE ✅ - Phase 4B can be added when scaling/production deployment is needed

---

### ⏸️ Phase 4B: Advanced Production Observability & Resilience - **DEFERRED**

**GitHub Issue**: [#118](https://github.com/datablogin/AutoCLV/issues/118)
**Status**: Not started - deferred until production scaling needs arise
**Estimated Effort**: 1 week
**Priority**: MEDIUM

#### Scope

Phase 4B adds advanced production-grade observability and resilience features that are valuable for scaled deployments but not required for the functional MVP.

#### 4B.1. Production OpenTelemetry with OTLP Export

**Goal**: Export traces to production observability backends (Jaeger, Zipkin, cloud providers)

```python
# analytics/services/mcp_server/observability.py
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider

def configure_production_telemetry(
    service_name: str = "mcp-five-lenses",
    otlp_endpoint: str = "localhost:4317",
    environment: str = "production"
):
    """Configure production OpenTelemetry with OTLP export."""

    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0",
        "deployment.environment": environment,
    })

    trace_provider = TracerProvider(resource=resource)
    otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
    trace_provider.add_span_processor(
        BatchSpanProcessor(otlp_exporter)
    )

    trace.set_tracer_provider(trace_provider)
```

**Tasks**:
- [ ] Add `opentelemetry-exporter-otlp-proto-grpc>=1.27.0` dependency
- [ ] Create production observability configuration
- [ ] Add environment-based configuration (dev vs prod)
- [ ] Implement sampling strategies for high-volume scenarios
- [ ] Configure batch span processing with appropriate timeouts
- [ ] Document Jaeger/Zipkin setup

**Deliverables**:
- Production-ready OTLP exporter configuration
- Docker Compose setup for local Jaeger/Zipkin testing
- Configuration guide for cloud providers (AWS X-Ray, GCP Cloud Trace, etc.)

#### 4B.2. Circuit Breaker Pattern

**Goal**: Prevent cascade failures when external dependencies fail

```python
# analytics/services/mcp_server/resilience/circuit_breakers.py
from pybreaker import CircuitBreaker, CircuitBreakerError

# Circuit breaker for file system operations
file_operations_breaker = CircuitBreaker(
    fail_max=5,  # Open after 5 failures
    timeout_duration=60,  # Stay open for 60 seconds
    name="file_operations"
)

@file_operations_breaker
async def load_large_dataset_with_breaker(path: str):
    """Load dataset with circuit breaker protection."""
    return await load_large_dataset(path)
```

**Tasks**:
- [ ] Add `pybreaker>=1.2.0` dependency
- [ ] Identify critical external dependencies requiring protection
- [ ] Implement circuit breakers for:
  - Large file operations
  - Database connections (if applicable)
  - External API calls (if any)
- [ ] Add circuit breaker state monitoring
- [ ] Configure failure thresholds and recovery strategies
- [ ] Add circuit breaker status to health check

**Deliverables**:
- Circuit breaker implementation for critical paths
- Monitoring dashboard for circuit breaker states
- Configuration guide

#### 4B.3. Advanced Distributed Tracing

**Goal**: Add comprehensive tracing across all components

**Tasks**:
- [ ] Add tracing to Lens 2, 3, 4 execution (currently only Lens 1 & 5)
- [ ] Instrument foundation data builders:
  - `build_customer_data_mart()`
  - `calculate_rfm()`
  - `create_cohorts()`
- [ ] Add span events for key milestones
- [ ] Implement trace context propagation across async operations
- [ ] Add custom span attributes for business metrics
- [ ] Document trace visualization best practices

**Files to modify**:
- `analytics/services/mcp_server/tools/lens2.py`
- `analytics/services/mcp_server/tools/lens3.py`
- `analytics/services/mcp_server/tools/lens4.py`
- `analytics/services/mcp_server/tools/data_mart.py`
- `analytics/services/mcp_server/tools/rfm.py`
- `analytics/services/mcp_server/tools/cohorts.py`

**Deliverables**:
- Complete tracing coverage for all lenses
- Foundation data builder instrumentation
- Trace visualization examples and documentation

#### 4B.4. Metrics Export to Prometheus

**Goal**: Export metrics to Prometheus for monitoring and alerting

```python
# analytics/services/mcp_server/metrics/prometheus_exporter.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Metrics
lens_execution_duration = Histogram(
    'lens_execution_duration_seconds',
    'Lens execution duration',
    ['lens_name']
)

lens_execution_total = Counter(
    'lens_execution_total',
    'Total lens executions',
    ['lens_name', 'status']
)

active_analyses = Gauge(
    'active_analyses',
    'Number of active analyses'
)
```

**Tasks**:
- [ ] Add `prometheus-client>=0.21.0` dependency
- [ ] Create Prometheus metrics exporter
- [ ] Expose `/metrics` endpoint
- [ ] Export key metrics:
  - Lens execution duration (histogram)
  - Request rate by lens (counter)
  - Error rate by error type (counter)
  - Active request count (gauge)
  - System resource usage (gauge)
- [ ] Integrate with existing metrics collector
- [ ] Create example Prometheus configuration
- [ ] Create example Grafana dashboards

**Deliverables**:
- Prometheus metrics endpoint
- Example Prometheus + Grafana setup
- Pre-built Grafana dashboards
- Alert rule examples

#### 4B.5. Enhanced Health Check

**Goal**: Add liveness/readiness probes and deep health checks

**Tasks**:
- [ ] Add liveness probe (server is running)
- [ ] Add readiness probe (server can handle requests)
- [ ] Implement deep health checks:
  - OpenTelemetry tracer status
  - Metrics collector working status
  - Circuit breaker states
  - Critical dependencies validation
- [ ] Add configurable timeout for health checks
- [ ] Return different HTTP status codes based on health state

**Files to modify**:
- `analytics/services/mcp_server/tools/health_check.py`

**Deliverables**:
- Kubernetes-ready liveness/readiness probes
- Comprehensive health check validation
- Health check documentation

#### 4B.6. Comprehensive Testing

**Goal**: Test all advanced observability features

**Tasks**:
- [ ] Test OTLP trace export (mock Jaeger/Zipkin)
- [ ] Test circuit breaker state transitions (closed → open → half-open)
- [ ] Test Prometheus metrics export
- [ ] Load testing with observability enabled
- [ ] Test trace context propagation
- [ ] Performance testing (ensure <5% overhead)

**Files to create**:
- `tests/services/mcp_server/test_phase4b_advanced_observability.py`
- `tests/services/mcp_server/test_circuit_breakers.py`
- `tests/services/mcp_server/test_prometheus_metrics.py`

**Deliverables**:
- Comprehensive test suite (target: 20+ tests)
- Performance benchmarks
- Load testing results

#### Success Criteria

- [ ] OTLP traces exported successfully to Jaeger/Zipkin
- [ ] Circuit breakers prevent cascade failures in failure scenarios
- [ ] Prometheus metrics endpoint returns valid metrics
- [ ] All lens executions have distributed tracing
- [ ] Health checks validate all observability components
- [ ] Test coverage >90% for new features
- [ ] Performance impact <5% with all features enabled
- [ ] Documentation includes setup guides for all backends

#### Dependencies

```toml
# Phase 4B: Advanced Observability & Resilience
"opentelemetry-exporter-otlp-proto-grpc>=1.27.0",
"pybreaker>=1.2.0",
"prometheus-client>=0.21.0",
```

#### Documentation Deliverables

- Setup guide for Jaeger/Zipkin
- Circuit breaker configuration guide
- Prometheus metrics reference
- Grafana dashboard guide
- Observability best practices
- Troubleshooting guide
- Production deployment checklist

#### Optional Enhancements

- Distributed tracing across multiple services
- Custom metrics dashboards (Grafana)
- Alert rules for critical metrics (AlertManager)
- Log aggregation integration (ELK/Loki)
- APM integration (Datadog, New Relic, etc.)

---

### ❌ Phase 5: Natural Language Interface - **NOT STARTED**

**Status**: Blocked by Phase 3 completion
**Dependencies**: Requires LangGraph coordinator
**Next Steps**: Cannot start until orchestration layer is functional

**What's Needed**:
1. Claude-powered query interpreter (`QueryInterpreter` class)
2. Result synthesizer with narrative generation (`ResultSynthesizer`)
3. Conversational analysis tool with context maintenance
4. Cost and latency optimization

**Priority**: LOW - Enhancement after core orchestration works

---

## Phase Breakdown

### Phase 0: Foundation Setup (Week 0 - 3 days) ✅ COMPLETE

**Goal**: Establish MCP infrastructure and development environment

#### Tasks

**0.1. MCP Development Environment**
- [ ] Install FastMCP: `pip install fastmcp`
- [ ] Install LangGraph: `pip install langgraph`
- [ ] Install OpenTelemetry: `pip install opentelemetry-sdk opentelemetry-instrumentation`
- [ ] Set up Claude Desktop config for local MCP testing
- [ ] Create `analytics/services/mcp_server/` directory structure

**0.2. Base MCP Server Scaffold**

Create `/Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a/analytics/services/mcp_server/main.py`:

```python
from fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from dataclasses import dataclass
import asyncpg

@dataclass
class AppContext:
    """Application-wide resources for MCP server."""
    db_pool: asyncpg.Pool
    config: dict

@asynccontextmanager
async def app_lifespan():
    """Initialize and cleanup MCP server resources."""
    # Startup
    db_pool = await asyncpg.create_pool(
        host='localhost',
        database='analytics',
        user='analytics_user',
        min_size=5,
        max_size=20
    )

    config = {
        "max_lookback_days": 730,
        "default_discount_rate": 0.1,
    }

    yield AppContext(db_pool=db_pool, config=config)

    # Shutdown
    await db_pool.close()

# Initialize MCP server
mcp = FastMCP(
    name="Four Lenses Analytics",
    version="0.1.0",
    description="AutoCLV Four Lenses customer analytics via MCP",
    lifespan=app_lifespan
)

if __name__ == "__main__":
    mcp.run()
```

**0.3. Observability Integration**

Create `/Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a/analytics/services/mcp_server/observability.py`:

```python
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
import structlog

def configure_observability(service_name: str = "mcp-four-lenses"):
    """Configure OpenTelemetry tracing and metrics."""

    # Configure tracing
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(
        BatchSpanProcessor(ConsoleSpanExporter())
    )
    trace.set_tracer_provider(trace_provider)

    # Configure metrics
    metric_reader = PeriodicExportingMetricReader(
        ConsoleMetricExporter(), export_interval_millis=5000
    )
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    # Configure structured logging
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.WriteLoggerFactory(),
    )

    return trace.get_tracer(service_name), metrics.get_meter(service_name)
```

**0.4. Testing Infrastructure**

Create `/Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a/tests/services/mcp_server/test_mcp_basic.py`:

```python
import pytest
from analytics.services.mcp_server.main import mcp

@pytest.mark.asyncio
async def test_mcp_server_initialization():
    """Test MCP server initializes correctly."""
    assert mcp.name == "Four Lenses Analytics"
    assert mcp.version == "0.1.0"

@pytest.mark.asyncio
async def test_mcp_health_check():
    """Test basic MCP server health."""
    # Will add actual health check tool in Phase 1
    pass
```

#### Success Criteria
- [ ] MCP server starts without errors
- [ ] Claude Desktop can connect to local MCP server
- [ ] OpenTelemetry traces appear in console
- [ ] Basic tests pass

#### Deliverables
- Working MCP server scaffold
- Observability configured
- Development environment documented

---

### Phase 1: Foundation Services (Week 1)

**Goal**: Expose foundation modules (Data Mart, RFM, Cohorts) as MCP tools

#### 1.1. Data Mart Service

Create `/Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a/analytics/services/mcp_server/tools/data_mart.py`:

```python
from fastmcp import Context
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal
from customer_base_audit.foundation.data_mart import (
    CustomerDataMartBuilder,
    PeriodGranularity,
)
from analytics.services.mcp_server.main import mcp
import structlog

logger = structlog.get_logger(__name__)

class BuildDataMartRequest(BaseModel):
    """Request to build customer data mart."""
    transaction_data_path: str = Field(
        description="Path to transaction data (CSV or JSON)"
    )
    period_granularities: list[Literal["month", "quarter", "year"]] = Field(
        default=["quarter", "year"],
        description="Period granularities to compute"
    )

class DataMartResponse(BaseModel):
    """Data mart build response."""
    order_count: int
    period_count: int
    customer_count: int
    granularities: list[str]
    date_range: tuple[str, str]

@mcp.tool()
async def build_customer_data_mart(
    request: BuildDataMartRequest,
    ctx: Context
) -> DataMartResponse:
    """
    Build customer data mart from raw transaction data.

    This tool aggregates raw transactions into order-level and period-level
    summaries, which are the foundation for all Four Lenses analyses.

    Args:
        request: Configuration for data mart build

    Returns:
        Summary statistics about the built data mart
    """
    await ctx.info(f"Building data mart from {request.transaction_data_path}")

    # Parse granularities
    granularities = tuple(
        PeriodGranularity(g.upper())
        for g in request.period_granularities
    )

    # Build data mart
    builder = CustomerDataMartBuilder(period_granularities=granularities)

    # Load transactions (existing helper from cli.py)
    from customer_base_audit.cli import _load_transactions
    transactions = _load_transactions(request.transaction_data_path)

    await ctx.report_progress(0.3, "Aggregating orders...")
    mart = builder.build(transactions)

    await ctx.report_progress(0.9, "Finalizing...")

    # Extract summary
    all_periods = []
    for granularity, periods in mart.period_aggregations.items():
        all_periods.extend(periods)

    dates = [p.period_start for p in all_periods]
    date_range = (
        min(dates).isoformat() if dates else "",
        max(dates).isoformat() if dates else ""
    )

    # Store in context for reuse
    ctx.set_state("data_mart", mart)

    await ctx.info("Data mart built successfully")

    return DataMartResponse(
        order_count=len(mart.order_aggregations),
        period_count=len(all_periods),
        customer_count=len(set(p.customer_id for p in all_periods)),
        granularities=[g.value for g in granularities],
        date_range=date_range
    )
```

#### 1.2. RFM Service

Create `/Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a/analytics/services/mcp_server/tools/rfm.py`:

```python
from fastmcp import Context
from pydantic import BaseModel, Field
from datetime import datetime
from customer_base_audit.foundation.rfm import calculate_rfm, calculate_rfm_scores
from analytics.services.mcp_server.main import mcp
import structlog

logger = structlog.get_logger(__name__)

class CalculateRFMRequest(BaseModel):
    """Request to calculate RFM metrics."""
    observation_end: datetime = Field(
        description="End date for RFM observation period"
    )
    enable_parallel: bool = Field(
        default=True,
        description="Enable parallel processing for large datasets"
    )
    calculate_scores: bool = Field(
        default=True,
        description="Also calculate RFM scores (1-5 binning)"
    )

class RFMResponse(BaseModel):
    """RFM calculation response."""
    metrics_count: int
    score_count: int
    date_range: tuple[str, str]
    parallel_enabled: bool

@mcp.tool()
async def calculate_rfm_metrics(
    request: CalculateRFMRequest,
    ctx: Context
) -> RFMResponse:
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics.

    This tool transforms period aggregations into RFM metrics for each
    customer, which are the foundation for Lens 1 and Lens 2 analyses.

    Args:
        request: Configuration for RFM calculation

    Returns:
        Summary of calculated RFM metrics
    """
    await ctx.info("Starting RFM calculation")

    # Get data mart from context
    mart = ctx.get_state("data_mart")
    if mart is None:
        raise ValueError(
            "Data mart not found. Run build_customer_data_mart first."
        )

    # Get period aggregations (use first granularity)
    first_granularity = list(mart.period_aggregations.keys())[0]
    period_aggregations = mart.period_aggregations[first_granularity]

    await ctx.report_progress(0.2, "Calculating RFM metrics...")

    # Calculate RFM
    rfm_metrics = calculate_rfm(
        period_aggregations=period_aggregations,
        observation_end=request.observation_end,
        parallel=request.enable_parallel
    )

    await ctx.report_progress(0.7, "Calculating RFM scores...")

    # Calculate scores if requested
    rfm_scores = []
    if request.calculate_scores:
        rfm_scores = calculate_rfm_scores(rfm_metrics)

    # Store in context
    ctx.set_state("rfm_metrics", rfm_metrics)
    ctx.set_state("rfm_scores", rfm_scores)

    # Extract date range
    dates = [m.observation_start for m in rfm_metrics]
    date_range = (
        min(dates).isoformat() if dates else "",
        max(dates).isoformat() if dates else ""
    )

    await ctx.info(f"RFM calculation complete: {len(rfm_metrics)} customers")

    return RFMResponse(
        metrics_count=len(rfm_metrics),
        score_count=len(rfm_scores),
        date_range=date_range,
        parallel_enabled=request.enable_parallel
    )
```

#### 1.3. Cohort Service

Create `/Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a/analytics/services/mcp_server/tools/cohorts.py`:

```python
from fastmcp import Context
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal
from customer_base_audit.foundation.cohorts import (
    create_monthly_cohorts,
    create_quarterly_cohorts,
    create_yearly_cohorts,
    assign_cohorts,
)
from analytics.services.mcp_server.main import mcp
import structlog

logger = structlog.get_logger(__name__)

class CreateCohortsRequest(BaseModel):
    """Request to create cohort definitions."""
    cohort_type: Literal["monthly", "quarterly", "yearly"] = Field(
        description="Type of cohorts to create"
    )
    start_date: datetime | None = Field(
        default=None,
        description="Start date (defaults to earliest customer)"
    )
    end_date: datetime | None = Field(
        default=None,
        description="End date (defaults to latest customer)"
    )

class CohortResponse(BaseModel):
    """Cohort creation response."""
    cohort_count: int
    customer_count: int
    date_range: tuple[str, str]
    cohort_type: str
    assignment_summary: dict[str, int]

@mcp.tool()
async def create_customer_cohorts(
    request: CreateCohortsRequest,
    ctx: Context
) -> CohortResponse:
    """
    Create cohort definitions and assign customers.

    This tool creates time-based cohorts (monthly, quarterly, yearly) and
    assigns customers based on acquisition dates. Required for Lens 3 and 4.

    Args:
        request: Configuration for cohort creation

    Returns:
        Summary of created cohorts and assignments
    """
    await ctx.info(f"Creating {request.cohort_type} cohorts")

    # Get data mart from context
    mart = ctx.get_state("data_mart")
    if mart is None:
        raise ValueError(
            "Data mart not found. Run build_customer_data_mart first."
        )

    # Extract customer identifiers (need acquisition dates)
    # For now, use first transaction date as acquisition proxy
    from customer_base_audit.foundation.customer_contract import CustomerIdentifier

    customer_first_dates = {}
    for granularity, periods in mart.period_aggregations.items():
        for period in periods:
            if period.customer_id not in customer_first_dates:
                customer_first_dates[period.customer_id] = period.period_start
            else:
                customer_first_dates[period.customer_id] = min(
                    customer_first_dates[period.customer_id],
                    period.period_start
                )

    customers = [
        CustomerIdentifier(customer_id=cid, acquisition_ts=acq_date)
        for cid, acq_date in customer_first_dates.items()
    ]

    await ctx.report_progress(0.3, "Creating cohort definitions...")

    # Create cohorts
    if request.cohort_type == "monthly":
        cohort_defs = create_monthly_cohorts(
            customers, request.start_date, request.end_date
        )
    elif request.cohort_type == "quarterly":
        cohort_defs = create_quarterly_cohorts(
            customers, request.start_date, request.end_date
        )
    else:
        cohort_defs = create_yearly_cohorts(
            customers, request.start_date, request.end_date
        )

    await ctx.report_progress(0.7, "Assigning customers to cohorts...")

    # Assign customers
    cohort_assignments = assign_cohorts(customers, cohort_defs)

    # Store in context
    ctx.set_state("cohort_definitions", cohort_defs)
    ctx.set_state("cohort_assignments", cohort_assignments)

    # Calculate summary
    assignment_summary = {}
    for cohort_id in set(cohort_assignments.values()):
        assignment_summary[cohort_id] = sum(
            1 for cid in cohort_assignments.values() if cid == cohort_id
        )

    date_range = (
        min(c.start_date for c in cohort_defs).isoformat(),
        max(c.end_date for c in cohort_defs).isoformat()
    )

    await ctx.info(f"Created {len(cohort_defs)} cohorts")

    return CohortResponse(
        cohort_count=len(cohort_defs),
        customer_count=len(customers),
        date_range=date_range,
        cohort_type=request.cohort_type,
        assignment_summary=assignment_summary
    )
```

#### 1.4. Integration Tests

Create `/Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a/tests/services/mcp_server/test_foundation_tools.py`:

```python
import pytest
from analytics.services.mcp_server.main import mcp
from analytics.services.mcp_server.tools import data_mart, rfm, cohorts

@pytest.mark.asyncio
async def test_data_mart_build_workflow():
    """Test complete data mart build workflow."""
    # This will use test fixtures from existing tests
    pass

@pytest.mark.asyncio
async def test_rfm_calculation_workflow():
    """Test RFM calculation with data mart context."""
    pass

@pytest.mark.asyncio
async def test_cohort_creation_workflow():
    """Test cohort creation and assignment."""
    pass
```

#### Success Criteria
- [ ] Data mart tool builds successfully from transactions
- [ ] RFM tool calculates metrics using data mart context
- [ ] Cohort tool creates definitions and assignments
- [ ] All tools report progress correctly
- [ ] Context state management works (data passes between tools)
- [ ] Integration tests pass

#### Deliverables
- 3 foundation MCP tools (data mart, RFM, cohorts)
- Context-based state management
- Integration test suite

---

### Phase 2: Lens Services (Week 2)

**Goal**: Wrap Lenses 1-4 as stateless MCP tools

#### 2.1. Lens 1 Service

Create `/Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a/analytics/services/mcp_server/tools/lens1.py`:

```python
from fastmcp import Context
from pydantic import BaseModel, Field
from decimal import Decimal
from customer_base_audit.analyses.lens1 import analyze_single_period, Lens1Metrics
from analytics.services.mcp_server.main import mcp
import structlog

logger = structlog.get_logger(__name__)

class Lens1Request(BaseModel):
    """Request for Lens 1 analysis."""
    period_name: str = Field(
        default="Current Period",
        description="Name for this analysis period"
    )

class Lens1Response(BaseModel):
    """Lens 1 analysis response."""
    period_name: str
    total_customers: int
    one_time_buyers: int
    one_time_buyer_pct: float
    total_revenue: float
    top_10pct_revenue_contribution: float
    top_20pct_revenue_contribution: float
    avg_orders_per_customer: float
    median_customer_value: float
    rfm_distribution: dict[str, int]

    # Insights
    customer_health_score: float  # 0-100
    concentration_risk: str  # "low", "medium", "high"
    recommendations: list[str]

@mcp.tool()
async def analyze_single_period_snapshot(
    request: Lens1Request,
    ctx: Context
) -> Lens1Response:
    """
    Lens 1: Single-period snapshot analysis.

    Analyzes customer base health for a single observation period, including:
    - Customer counts and one-time buyer percentage
    - Revenue distribution and concentration (Pareto analysis)
    - Average order frequency and median customer value
    - RFM segment distribution

    This is the foundation lens for understanding current customer base state.

    Args:
        request: Configuration for Lens 1 analysis

    Returns:
        Comprehensive single-period metrics with actionable insights
    """
    await ctx.info("Starting Lens 1 analysis")

    # Get RFM metrics from context
    rfm_metrics = ctx.get_state("rfm_metrics")
    rfm_scores = ctx.get_state("rfm_scores")

    if rfm_metrics is None:
        raise ValueError(
            "RFM metrics not found. Run calculate_rfm_metrics first."
        )

    await ctx.report_progress(0.3, "Analyzing customer distribution...")

    # Run Lens 1 analysis
    lens1_result = analyze_single_period(
        rfm_metrics=rfm_metrics,
        rfm_scores=rfm_scores if rfm_scores else None
    )

    await ctx.report_progress(0.7, "Generating insights...")

    # Calculate insights
    health_score = _calculate_customer_health_score(lens1_result)
    concentration_risk = _assess_concentration_risk(lens1_result)
    recommendations = _generate_lens1_recommendations(lens1_result)

    # Store in context
    ctx.set_state("lens1_result", lens1_result)

    await ctx.info("Lens 1 analysis complete")

    return Lens1Response(
        period_name=request.period_name,
        total_customers=lens1_result.total_customers,
        one_time_buyers=lens1_result.one_time_buyers,
        one_time_buyer_pct=float(lens1_result.one_time_buyer_pct),
        total_revenue=float(lens1_result.total_revenue),
        top_10pct_revenue_contribution=float(lens1_result.top_10pct_revenue_contribution),
        top_20pct_revenue_contribution=float(lens1_result.top_20pct_revenue_contribution),
        avg_orders_per_customer=float(lens1_result.avg_orders_per_customer),
        median_customer_value=float(lens1_result.median_customer_value),
        rfm_distribution=lens1_result.rfm_distribution,
        customer_health_score=health_score,
        concentration_risk=concentration_risk,
        recommendations=recommendations
    )

def _calculate_customer_health_score(metrics: Lens1Metrics) -> float:
    """Calculate 0-100 health score based on key indicators."""
    score = 100.0

    # Penalize high one-time buyer percentage
    if metrics.one_time_buyer_pct > Decimal("70"):
        score -= 30
    elif metrics.one_time_buyer_pct > Decimal("50"):
        score -= 15

    # Penalize extreme revenue concentration
    if metrics.top_10pct_revenue_contribution > Decimal("80"):
        score -= 20
    elif metrics.top_10pct_revenue_contribution > Decimal("60"):
        score -= 10

    # Reward healthy repeat purchase behavior
    if metrics.avg_orders_per_customer > Decimal("3"):
        score += 10

    return max(0.0, min(100.0, score))

def _assess_concentration_risk(metrics: Lens1Metrics) -> str:
    """Assess revenue concentration risk."""
    top10 = float(metrics.top_10pct_revenue_contribution)

    if top10 > 70:
        return "high"
    elif top10 > 50:
        return "medium"
    else:
        return "low"

def _generate_lens1_recommendations(metrics: Lens1Metrics) -> list[str]:
    """Generate actionable recommendations."""
    recs = []

    if metrics.one_time_buyer_pct > Decimal("60"):
        recs.append(
            f"HIGH: {metrics.one_time_buyer_pct}% one-time buyers. "
            f"Implement retention campaigns targeting first-time purchasers."
        )

    if metrics.top_10pct_revenue_contribution > Decimal("70"):
        recs.append(
            f"MEDIUM: Top 10% contribute {metrics.top_10pct_revenue_contribution}% of revenue. "
            f"Diversify customer base to reduce concentration risk."
        )

    if metrics.avg_orders_per_customer < Decimal("2"):
        recs.append(
            "MEDIUM: Low average orders per customer. "
            "Focus on repeat purchase incentives and loyalty programs."
        )

    if not recs:
        recs.append("Customer base health is strong. Maintain current strategies.")

    return recs
```

#### 2.2. Lens 2 Service

Create `/Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a/analytics/services/mcp_server/tools/lens2.py`:

```python
from fastmcp import Context
from pydantic import BaseModel, Field
from customer_base_audit.analyses.lens2 import analyze_period_comparison, Lens2Metrics
from analytics.services.mcp_server.main import mcp
import structlog

logger = structlog.get_logger(__name__)

class Lens2Request(BaseModel):
    """Request for Lens 2 analysis."""
    period1_name: str = Field(default="Period 1")
    period2_name: str = Field(default="Period 2")

class Lens2Response(BaseModel):
    """Lens 2 analysis response."""
    # Period summaries
    period1_customers: int
    period2_customers: int

    # Migration metrics
    retained_customers: int
    churned_customers: int
    new_customers: int
    reactivated_customers: int

    # Rates
    retention_rate: float
    churn_rate: float
    reactivation_rate: float

    # Growth metrics
    customer_count_change: int
    revenue_change_pct: float
    avg_order_value_change_pct: float

    # Insights
    growth_momentum: str  # "strong", "moderate", "declining", "negative"
    key_drivers: list[str]
    recommendations: list[str]

@mcp.tool()
async def analyze_period_to_period_comparison(
    request: Lens2Request,
    ctx: Context
) -> Lens2Response:
    """
    Lens 2: Period-to-period comparison analysis.

    Compares two time periods to track customer migration patterns:
    - Retention, churn, and reactivation rates
    - New customer acquisition
    - Revenue and AOV trends

    This lens reveals customer lifecycle dynamics and business momentum.

    Args:
        request: Configuration for Lens 2 analysis

    Returns:
        Period comparison metrics with growth insights
    """
    await ctx.info("Starting Lens 2 analysis")

    # This requires two separate RFM calculations
    # For MVP, we'll use a simplified approach with stored RFM
    # In production, would accept two RFM metric sets as parameters

    # Implementation similar to Lens 1
    # ... (detailed implementation omitted for brevity)

    pass
```

#### 2.3. Lens 3 Service

Create `/Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a/analytics/services/mcp_server/tools/lens3.py`:

```python
from fastmcp import Context
from pydantic import BaseModel, Field
from datetime import datetime
from customer_base_audit.analyses.lens3 import analyze_cohort_evolution, Lens3Metrics
from analytics.services.mcp_server.main import mcp

class Lens3Request(BaseModel):
    """Request for Lens 3 analysis."""
    cohort_id: str = Field(description="Cohort identifier to analyze")

class Lens3Response(BaseModel):
    """Lens 3 analysis response."""
    cohort_id: str
    cohort_size: int
    periods_analyzed: int

    # Key metrics by period
    activation_curve: dict[int, float]  # period -> cumulative activation rate
    revenue_curve: dict[int, float]  # period -> avg revenue per cohort member
    retention_curve: dict[int, float]  # period -> active customer rate

    # Insights
    cohort_maturity: str  # "early", "growth", "mature", "declining"
    ltv_trajectory: str  # "strong", "moderate", "weak"
    recommendations: list[str]

@mcp.tool()
async def analyze_cohort_lifecycle(
    request: Lens3Request,
    ctx: Context
) -> Lens3Response:
    """
    Lens 3: Single cohort evolution analysis.

    Tracks a single acquisition cohort through their lifecycle:
    - Cumulative activation rates over time
    - Revenue per cohort member by period
    - Retention patterns and churn dynamics

    This lens reveals cohort maturity and lifetime value patterns.

    Args:
        request: Configuration for Lens 3 analysis

    Returns:
        Cohort lifecycle metrics with maturity assessment
    """
    # Implementation similar to Lens 1 & 2
    pass
```

#### 2.4. Lens 4 Service

Create `/Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a/analytics/services/mcp_server/tools/lens4.py`:

```python
from fastmcp import Context
from pydantic import BaseModel, Field
from customer_base_audit.analyses.lens4 import compare_cohorts, Lens4Metrics
from analytics.services.mcp_server.main import mcp

class Lens4Request(BaseModel):
    """Request for Lens 4 analysis."""
    alignment_type: str = Field(
        default="left-aligned",
        description="Alignment mode: left-aligned or time-aligned"
    )

class Lens4Response(BaseModel):
    """Lens 4 analysis response."""
    cohort_count: int
    alignment_type: str

    # Decomposition summary
    cohort_summaries: list[dict]  # Per-cohort key metrics

    # Comparative insights
    best_performing_cohort: str
    worst_performing_cohort: str
    key_differences: list[str]
    recommendations: list[str]

@mcp.tool()
async def compare_multiple_cohorts(
    request: Lens4Request,
    ctx: Context
) -> Lens4Response:
    """
    Lens 4: Multi-cohort comparison analysis.

    Compares multiple acquisition cohorts to identify:
    - Cohort-level performance differences
    - AOF, AOV, and margin decomposition
    - Time-to-second-purchase patterns

    This lens reveals which acquisition periods yield the best customers.

    Args:
        request: Configuration for Lens 4 analysis

    Returns:
        Multi-cohort comparison with performance rankings
    """
    # Implementation similar to other lenses
    pass
```

#### 2.5. Lens Integration Tests

Create comprehensive test suite validating each lens service.

#### Success Criteria
- [ ] All 4 lens tools execute successfully
- [ ] Each lens returns structured Pydantic models
- [ ] Insights and recommendations generated correctly
- [ ] Context state management works across lenses
- [ ] Tests achieve >90% coverage

#### Deliverables
- 4 lens MCP tools with rich output models
- Insight generation logic for each lens
- Comprehensive test suite

---

### Phase 3: LangGraph Coordinator (Week 3)

**Goal**: Implement orchestration layer for dynamic lens execution

#### 3.1. StateGraph Definition

Create `/Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a/analytics/services/mcp_server/orchestration/coordinator.py`:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from pydantic import BaseModel
import operator

class AnalysisState(TypedDict):
    """State for Four Lenses analysis workflow."""

    # Input
    query: str  # Natural language query from user
    intent: dict  # Parsed intent (which lenses, parameters)

    # Foundation data
    data_mart_ready: bool
    rfm_ready: bool
    cohorts_ready: bool

    # Lens results
    lens1_result: dict | None
    lens2_result: dict | None
    lens3_result: dict | None
    lens4_result: dict | None

    # Aggregated output
    insights: Annotated[list[str], operator.add]  # Collect insights from all lenses
    recommendations: Annotated[list[str], operator.add]

    # Metadata
    lenses_executed: Annotated[list[str], operator.add]
    execution_time_ms: float
    error: str | None

class FourLensesCoordinator:
    """Orchestrates Four Lenses analysis workflow."""

    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph."""

        workflow = StateGraph(AnalysisState)

        # Add nodes
        workflow.add_node("parse_intent", self._parse_intent)
        workflow.add_node("prepare_foundation", self._prepare_foundation)
        workflow.add_node("execute_lens1", self._execute_lens1)
        workflow.add_node("execute_lens2", self._execute_lens2)
        workflow.add_node("execute_lens3", self._execute_lens3)
        workflow.add_node("execute_lens4", self._execute_lens4)
        workflow.add_node("synthesize_results", self._synthesize_results)

        # Define edges
        workflow.set_entry_point("parse_intent")
        workflow.add_edge("parse_intent", "prepare_foundation")
        workflow.add_conditional_edges(
            "prepare_foundation",
            self._route_to_lenses,
            {
                "lens1": "execute_lens1",
                "lens2": "execute_lens2",
                "lens3": "execute_lens3",
                "lens4": "execute_lens4",
                "synthesize": "synthesize_results"
            }
        )

        # All lenses route to synthesis
        workflow.add_edge("execute_lens1", "synthesize_results")
        workflow.add_edge("execute_lens2", "synthesize_results")
        workflow.add_edge("execute_lens3", "synthesize_results")
        workflow.add_edge("execute_lens4", "synthesize_results")

        workflow.add_edge("synthesize_results", END)

        return workflow.compile()

    async def _parse_intent(self, state: AnalysisState) -> AnalysisState:
        """Parse user query into structured intent."""
        # Use LLM to extract:
        # - Which lenses to run
        # - Date ranges
        # - Cohort filters
        # - Special parameters

        # For MVP, use simple keyword matching
        query = state["query"].lower()

        intent = {
            "lenses": [],
            "date_range": None,
            "cohort_filter": None
        }

        if "snapshot" in query or "current state" in query:
            intent["lenses"].append("lens1")

        if "compare" in query or "trend" in query:
            intent["lenses"].append("lens2")

        if "cohort" in query:
            intent["lenses"].extend(["lens3", "lens4"])

        # If no specific lens detected, run Lens 1 as default
        if not intent["lenses"]:
            intent["lenses"].append("lens1")

        state["intent"] = intent
        return state

    async def _prepare_foundation(self, state: AnalysisState) -> AnalysisState:
        """Ensure foundation data is prepared."""
        # Check if data mart, RFM, cohorts are ready
        # Call foundation MCP tools if needed

        state["data_mart_ready"] = True
        state["rfm_ready"] = True
        state["cohorts_ready"] = True

        return state

    def _route_to_lenses(self, state: AnalysisState) -> str:
        """Route to appropriate lens execution."""
        lenses_to_run = state["intent"]["lenses"]
        executed = state.get("lenses_executed", [])

        for lens in lenses_to_run:
            if lens not in executed:
                return lens

        return "synthesize"

    async def _execute_lens1(self, state: AnalysisState) -> AnalysisState:
        """Execute Lens 1 analysis."""
        # Call Lens 1 MCP tool
        # Store result in state

        state["lenses_executed"].append("lens1")
        state["lens1_result"] = {"placeholder": "Lens 1 result"}

        return state

    async def _synthesize_results(self, state: AnalysisState) -> AnalysisState:
        """Synthesize results from all executed lenses."""
        # Use LLM to generate coherent narrative
        # Aggregate insights and recommendations

        all_insights = []
        all_recommendations = []

        if state.get("lens1_result"):
            all_insights.append("Lens 1: Customer base health assessment complete")

        state["insights"] = all_insights
        state["recommendations"] = all_recommendations

        return state

    async def analyze(self, query: str) -> dict:
        """Run complete Four Lenses analysis from natural language query."""
        initial_state = AnalysisState(
            query=query,
            intent={},
            data_mart_ready=False,
            rfm_ready=False,
            cohorts_ready=False,
            lens1_result=None,
            lens2_result=None,
            lens3_result=None,
            lens4_result=None,
            insights=[],
            recommendations=[],
            lenses_executed=[],
            execution_time_ms=0.0,
            error=None
        )

        result = await self.graph.ainvoke(initial_state)
        return result
```

#### 3.2. MCP Tool for Orchestrated Analysis

Create `/Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a/analytics/services/mcp_server/tools/orchestrated_analysis.py`:

```python
from fastmcp import Context
from pydantic import BaseModel, Field
from analytics.services.mcp_server.orchestration.coordinator import FourLensesCoordinator
from analytics.services.mcp_server.main import mcp

class OrchestratedAnalysisRequest(BaseModel):
    """Request for orchestrated Four Lenses analysis."""
    query: str = Field(
        description="Natural language query describing desired analysis"
    )

class OrchestratedAnalysisResponse(BaseModel):
    """Orchestrated analysis response."""
    query: str
    lenses_executed: list[str]
    insights: list[str]
    recommendations: list[str]
    execution_time_ms: float

@mcp.tool()
async def run_orchestrated_analysis(
    request: OrchestratedAnalysisRequest,
    ctx: Context
) -> OrchestratedAnalysisResponse:
    """
    Run orchestrated Four Lenses analysis from natural language query.

    This tool uses LangGraph to:
    1. Parse the user's intent
    2. Determine which lenses to execute
    3. Run analyses in optimal order
    4. Synthesize results into coherent insights

    Example queries:
    - "Give me a snapshot of customer base health"
    - "Compare Q1 2024 to Q4 2023"
    - "Analyze the January 2024 acquisition cohort"
    - "Which cohorts have the best lifetime value?"

    Args:
        request: Natural language analysis query

    Returns:
        Synthesized insights from relevant lenses
    """
    await ctx.info(f"Running orchestrated analysis: {request.query}")

    coordinator = FourLensesCoordinator()
    result = await coordinator.analyze(request.query)

    return OrchestratedAnalysisResponse(
        query=request.query,
        lenses_executed=result["lenses_executed"],
        insights=result["insights"],
        recommendations=result["recommendations"],
        execution_time_ms=result["execution_time_ms"]
    )
```

#### 3.3. Parallel Execution

Enhance coordinator to run independent lenses in parallel:

```python
import asyncio

async def _execute_lenses_parallel(self, state: AnalysisState) -> AnalysisState:
    """Execute multiple lenses in parallel where possible."""
    lenses_to_run = state["intent"]["lenses"]

    # Lens 1 and 2 depend on RFM (can run in parallel)
    # Lens 3 and 4 depend on cohorts (can run in parallel)

    tasks = []

    if "lens1" in lenses_to_run:
        tasks.append(self._execute_lens1(state))

    if "lens2" in lenses_to_run:
        tasks.append(self._execute_lens2(state))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Merge results back into state
    for result in results:
        if isinstance(result, Exception):
            state["error"] = str(result)
        else:
            state.update(result)

    return state
```

#### Success Criteria
- [x] LangGraph state machine executes correctly
- [x] Intent parsing identifies correct lenses
- [x] Foundation preparation runs automatically
- [x] Lens execution follows dependency graph
- [x] Parallel execution works for independent lenses
- [x] Result synthesis aggregates insights coherently

#### Deliverables
- [x] LangGraph coordinator with StateGraph
- [x] Orchestrated analysis MCP tool
- [x] Parallel execution capability
- [x] Intent parsing logic

---

### Phase 4: Observability & Resilience (Week 4)

**Goal**: Add production-grade monitoring, error handling, and resilience

#### 4.1. OpenTelemetry Instrumentation

Enhance observability setup:

```python
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentation
from opentelemetry.sdk.trace import TracerProvider, Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentation

def configure_production_observability(
    service_name: str,
    otlp_endpoint: str = "localhost:4317"
):
    """Configure production OpenTelemetry with OTLP export."""

    # Create resource with service metadata
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0",
        "deployment.environment": "production"
    })

    # Configure tracer provider
    trace_provider = TracerProvider(resource=resource)

    # Add OTLP exporter (for Jaeger, Zipkin, or cloud providers)
    otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
    trace_provider.add_span_processor(
        BatchSpanProcessor(otlp_exporter)
    )

    trace.set_tracer_provider(trace_provider)

    # Auto-instrument libraries
    AsyncPGInstrumentation().instrument()

    return trace.get_tracer(service_name)
```

Add tracing to all MCP tools:

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

@mcp.tool()
async def analyze_single_period_snapshot(
    request: Lens1Request,
    ctx: Context
) -> Lens1Response:
    """Lens 1 with tracing."""

    with tracer.start_as_current_span("lens1.analyze") as span:
        span.set_attribute("period_name", request.period_name)

        # Get RFM metrics
        with tracer.start_as_current_span("lens1.get_rfm_context"):
            rfm_metrics = ctx.get_state("rfm_metrics")

        # Run analysis
        with tracer.start_as_current_span("lens1.calculate"):
            lens1_result = analyze_single_period(rfm_metrics)

        # Generate insights
        with tracer.start_as_current_span("lens1.generate_insights"):
            insights = _generate_lens1_recommendations(lens1_result)

        span.set_attribute("customer_count", lens1_result.total_customers)
        span.set_attribute("health_score", health_score)

        return Lens1Response(...)
```

#### 4.2. Error Handling & Retry Logic

Add resilience to coordinator:

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class FourLensesCoordinator:
    """Coordinator with resilience."""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        reraise=True
    )
    async def _execute_lens_with_retry(
        self,
        lens_func,
        state: AnalysisState
    ) -> AnalysisState:
        """Execute lens with automatic retry on transient failures."""
        try:
            return await lens_func(state)
        except Exception as e:
            logger.error(
                "lens_execution_failed",
                lens=lens_func.__name__,
                error=str(e),
                attempt=retry_state.attempt_number
            )
            raise

    async def _execute_lens1(self, state: AnalysisState) -> AnalysisState:
        """Execute Lens 1 with error handling."""
        try:
            # Call MCP tool
            result = await self._call_lens1_tool(state)

            # Validate result
            if not result or "error" in result:
                raise ValueError(f"Lens 1 returned invalid result: {result}")

            state["lens1_result"] = result
            state["lenses_executed"].append("lens1")

        except Exception as e:
            logger.error("lens1_execution_failed", error=str(e))
            state["error"] = f"Lens 1 failed: {str(e)}"

            # Continue workflow even if one lens fails
            state["lenses_executed"].append("lens1_failed")

        return state
```

#### 4.3. Health Check and Monitoring Tools

Add MCP tools for system health:

```python
@mcp.tool()
async def health_check(ctx: Context) -> dict:
    """
    Check health of MCP server and all dependencies.

    Returns:
        Health status for database, cache, and all services
    """
    health = {
        "status": "healthy",
        "checks": {},
        "timestamp": datetime.now().isoformat()
    }

    # Check database
    try:
        db_pool = ctx.request_context.lifespan_context.db_pool
        async with db_pool.acquire() as conn:
            await conn.execute("SELECT 1")
        health["checks"]["database"] = "healthy"
    except Exception as e:
        health["checks"]["database"] = f"unhealthy: {str(e)}"
        health["status"] = "degraded"

    # Check MCP tools
    try:
        # Verify all tools are registered
        health["checks"]["mcp_tools"] = len(mcp._tools)
    except Exception as e:
        health["checks"]["mcp_tools"] = f"error: {str(e)}"

    return health

@mcp.tool()
async def get_execution_metrics(ctx: Context) -> dict:
    """
    Get execution metrics for all lenses.

    Returns:
        Aggregated performance metrics
    """
    # Query metrics from OpenTelemetry collector
    # Or from in-memory metrics store

    return {
        "lens1_avg_duration_ms": 150,
        "lens2_avg_duration_ms": 200,
        "lens3_avg_duration_ms": 180,
        "lens4_avg_duration_ms": 250,
        "total_analyses_today": 42,
        "error_rate_pct": 0.5
    }
```

#### 4.4. Circuit Breaker Pattern

Add circuit breaker for external dependencies:

```python
from pybreaker import CircuitBreaker, CircuitBreakerError

# Create circuit breaker for database calls
db_breaker = CircuitBreaker(
    fail_max=5,  # Open after 5 failures
    timeout_duration=60,  # Stay open for 60 seconds
    name="database"
)

@db_breaker
async def fetch_data_with_breaker(query: str):
    """Fetch data with circuit breaker protection."""
    async with db_pool.acquire() as conn:
        return await conn.fetch(query)
```

#### Success Criteria
- [ ] All MCP tools have OpenTelemetry tracing
- [ ] Traces visible in Jaeger/Zipkin
- [ ] Automatic retry works for transient failures
- [ ] Circuit breaker prevents cascade failures
- [ ] Health check tool reports accurate status
- [ ] Metrics tool shows performance data
- [ ] Error handling allows graceful degradation

#### Deliverables
- Production OpenTelemetry configuration
- Retry logic in coordinator
- Circuit breaker for critical paths
- Health check and metrics MCP tools
- Error handling documentation

---

### Phase 5: Natural Language Interface (Week 5)

**Goal**: Add LLM-powered query interpretation and result synthesis

#### 5.1. Query Interpreter with Claude

Enhance intent parsing with Claude:

```python
from anthropic import AsyncAnthropic
import json

class QueryInterpreter:
    """Interprets natural language queries using Claude."""

    def __init__(self, api_key: str):
        self.client = AsyncAnthropic(api_key=api_key)

    async def parse_query(self, query: str) -> dict:
        """
        Parse natural language query into structured intent.

        Returns:
            {
                "lenses": ["lens1", "lens3"],
                "date_range": {"start": "2024-01-01", "end": "2024-03-31"},
                "filters": {"cohort_id": "2024-Q1"},
                "parameters": {"include_predictions": true}
            }
        """

        prompt = f"""
You are an expert in customer analytics using the Four Lenses methodology.

Parse this user query into a structured analysis plan:

Query: "{query}"

Available lenses:
- Lens 1: Single-period snapshot (current customer base health)
- Lens 2: Period-to-period comparison (retention, churn, growth)
- Lens 3: Single cohort evolution (lifecycle of one acquisition cohort)
- Lens 4: Multi-cohort comparison (which cohorts perform best)

Return JSON with this structure:
{{
  "lenses": ["lens1"],  // Which lenses to run
  "date_range": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}},  // null if not specified
  "filters": {{"cohort_id": "2024-Q1"}},  // Any cohort/segment filters
  "parameters": {{}},  // Additional parameters
  "reasoning": "Why these lenses were selected"
}}
"""

        response = await self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract JSON from response
        content = response.content[0].text

        # Parse JSON (handle markdown code blocks)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        intent = json.loads(content.strip())
        return intent
```

#### 5.2. Result Synthesizer with Claude

Add synthesis step to coordinator:

```python
class ResultSynthesizer:
    """Synthesizes multi-lens results using Claude."""

    def __init__(self, api_key: str):
        self.client = AsyncAnthropic(api_key=api_key)

    async def synthesize(
        self,
        query: str,
        lens_results: dict[str, dict]
    ) -> dict:
        """
        Synthesize results from multiple lenses into coherent narrative.

        Args:
            query: Original user query
            lens_results: Dict mapping lens names to their results

        Returns:
            {
                "summary": "Executive summary",
                "insights": ["Insight 1", "Insight 2", ...],
                "recommendations": ["Action 1", "Action 2", ...],
                "narrative": "Full narrative explanation"
            }
        """

        # Format lens results for Claude
        results_text = self._format_results(lens_results)

        prompt = f"""
You are an expert customer analytics consultant synthesizing analysis results.

Original query: "{query}"

Analysis results:
{results_text}

Provide a comprehensive synthesis with:
1. Executive summary (2-3 sentences)
2. Key insights (3-5 bullet points)
3. Actionable recommendations (3-5 bullet points)
4. Detailed narrative explanation

Return JSON with this structure:
{{
  "summary": "...",
  "insights": ["...", "..."],
  "recommendations": ["...", "..."],
  "narrative": "..."
}}
"""

        response = await self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.content[0].text

        # Parse JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]

        synthesis = json.loads(content.strip())
        return synthesis

    def _format_results(self, lens_results: dict[str, dict]) -> str:
        """Format lens results as readable text."""
        formatted = []

        for lens_name, result in lens_results.items():
            formatted.append(f"\n{lens_name.upper()} RESULTS:")
            formatted.append(json.dumps(result, indent=2))

        return "\n".join(formatted)
```

#### 5.3. Enhanced Orchestrated Analysis

Update coordinator to use LLM components:

```python
class FourLensesCoordinator:
    """Coordinator with LLM-powered interpretation and synthesis."""

    def __init__(self, anthropic_api_key: str):
        self.graph = self._build_graph()
        self.interpreter = QueryInterpreter(anthropic_api_key)
        self.synthesizer = ResultSynthesizer(anthropic_api_key)

    async def _parse_intent(self, state: AnalysisState) -> AnalysisState:
        """Parse intent using Claude."""
        intent = await self.interpreter.parse_query(state["query"])
        state["intent"] = intent

        logger.info(
            "intent_parsed",
            query=state["query"],
            lenses=intent["lenses"],
            reasoning=intent.get("reasoning")
        )

        return state

    async def _synthesize_results(self, state: AnalysisState) -> AnalysisState:
        """Synthesize results using Claude."""
        lens_results = {
            "lens1": state.get("lens1_result"),
            "lens2": state.get("lens2_result"),
            "lens3": state.get("lens3_result"),
            "lens4": state.get("lens4_result"),
        }

        # Filter out None results
        lens_results = {k: v for k, v in lens_results.items() if v is not None}

        synthesis = await self.synthesizer.synthesize(
            state["query"],
            lens_results
        )

        state["insights"] = synthesis["insights"]
        state["recommendations"] = synthesis["recommendations"]
        state["summary"] = synthesis["summary"]
        state["narrative"] = synthesis["narrative"]

        return state
```

#### 5.4. Conversational Analysis Tool

Add interactive analysis capability:

```python
@mcp.tool()
async def conversational_analysis(
    query: str,
    conversation_history: list[dict] | None = None,
    ctx: Context = None
) -> dict:
    """
    Run conversational Four Lenses analysis.

    This tool maintains conversation context to support follow-up questions:
    - "Now compare that to last quarter"
    - "What about the January cohort specifically?"
    - "Show me retention rates"

    Args:
        query: Natural language query
        conversation_history: Previous queries and results

    Returns:
        Analysis results with conversational context
    """
    # Maintain conversation state
    if conversation_history is None:
        conversation_history = []

    # Use Claude to determine if this is a follow-up
    # If so, resolve references ("that", "last quarter", etc.)

    # Run orchestrated analysis
    coordinator = FourLensesCoordinator(api_key=os.getenv("ANTHROPIC_API_KEY"))
    result = await coordinator.analyze(query)

    # Add to conversation history
    conversation_history.append({
        "query": query,
        "result": result,
        "timestamp": datetime.now().isoformat()
    })

    return {
        "result": result,
        "conversation_history": conversation_history
    }
```

#### Success Criteria
- [ ] Query interpreter correctly identifies lenses from natural language
- [ ] Result synthesizer generates coherent narratives
- [ ] Conversational analysis maintains context
- [ ] LLM costs are reasonable (<$0.50 per analysis)
- [ ] Response latency acceptable (<5 seconds end-to-end)

#### Deliverables
- Claude-powered query interpreter
- Result synthesizer with narrative generation
- Conversational analysis tool
- Cost and latency optimization

---

## Migration Strategy

### Data Migration
**Not applicable** - This is a new layer on top of existing code. No data migration needed.

### Code Migration
**Incremental wrapping approach**:
1. Phase 1-2: MCP services wrap existing functions (no changes to core)
2. Phase 3-5: New orchestration layer added above MCP services

### Testing Strategy

**Test Levels**:
1. **Unit Tests**: Each MCP tool tested independently
2. **Integration Tests**: Multi-tool workflows (data mart → RFM → Lens 1)
3. **End-to-End Tests**: Full orchestrated analysis from query to synthesis
4. **Regression Tests**: Existing 384 tests must continue passing

**Test Data**:
- Reuse existing test fixtures from Track A tests
- Add synthetic test cases for edge cases
- Performance test with 1M+ customer dataset

### Rollback Plan

**If issues arise**:
- Phase 1-2: Simply don't use MCP tools, continue using core directly
- Phase 3: Disable orchestrated analysis tool, use individual lens tools
- Phase 4-5: Disable LLM features, use manual orchestration

**Risk**: Very low - hybrid architecture means core functionality always available.

---

## Success Metrics

### Technical Metrics
- [ ] All 384 existing tests pass
- [ ] MCP server uptime >99.5%
- [ ] P95 latency for single lens <500ms
- [ ] P95 latency for orchestrated analysis <5s
- [ ] Error rate <1%
- [ ] OpenTelemetry traces captured for 100% of requests

### Business Metrics
- [ ] 3× faster analysis iteration (vs manual lens execution)
- [ ] 50% reduction in analyst time for routine analyses
- [ ] Natural language queries successfully interpreted >90% of time
- [ ] LLM costs <$0.50 per orchestrated analysis

### Quality Metrics
- [ ] Test coverage maintained at >90%
- [ ] Zero regressions in existing functionality
- [ ] Documentation coverage 100% for MCP tools
- [ ] Code review approval for all changes

---

## Risk Assessment

### High Risks
**None identified** - Hybrid architecture minimizes risk

### Medium Risks
1. **LLM Cost Overruns**
   - Mitigation: Token usage monitoring, caching, prompt optimization

2. **Latency Issues**
   - Mitigation: Parallel execution, caching, timeout configuration

3. **Query Interpretation Accuracy**
   - Mitigation: Extensive prompt engineering, user feedback loop, fallback to manual

### Low Risks
1. **MCP Protocol Changes**
   - Mitigation: Pin to stable version, monitor changelog

2. **OpenTelemetry Overhead**
   - Mitigation: Sampling configuration, async export

---

## Dependencies

### External Dependencies
- **FastMCP**: ^1.0.0 (MCP server framework)
- **LangGraph**: ^0.2.0 (Orchestration)
- **Anthropic SDK**: ^0.39.0 (Claude API)
- **OpenTelemetry SDK**: ^1.26.0 (Observability)
- **asyncpg**: ^0.29.0 (Async PostgreSQL)
- **structlog**: ^24.4.0 (Structured logging)

### Internal Dependencies
- Existing Track A modules (no changes required)
- Test fixtures from Track A test suite

---

## Documentation Plan

### Technical Documentation
- [ ] MCP server setup guide
- [ ] API reference for all MCP tools
- [ ] LangGraph state machine diagram
- [ ] OpenTelemetry trace visualization guide
- [ ] Troubleshooting guide

### User Documentation
- [ ] Natural language query examples
- [ ] Lens selection guide
- [ ] Interpretation of analysis results
- [ ] FAQ for common queries

### Developer Documentation
- [ ] Architecture decision records (ADRs)
- [ ] Contributing guide for new lenses
- [ ] Testing guide
- [ ] Deployment guide

---

## Timeline Summary

| Phase | Duration | Key Deliverables | Status |
|-------|----------|------------------|--------|
| Phase 0 | 3 days | MCP infrastructure, observability setup | ✅ COMPLETE |
| Phase 1 | 1 week | Foundation services (data mart, RFM, cohorts) | ✅ COMPLETE |
| Phase 2 | 1 week | Lens services (Lenses 1-5 including health assessment) | ✅ COMPLETE |
| Phase 3 | 4 hours | LangGraph coordinator, orchestration | ✅ COMPLETE |
| Phase 4A | 3 hours | Essential observability & resilience | ✅ COMPLETE |
| Phase 4B | 1 week | Advanced production observability | ⏸️ DEFERRED |
| Phase 5 | 1 week | Natural language interface, synthesis | ⏸️ DEFERRED |
| **Total** | **~1 week** | **Functional agentic architecture with observability** | **95% Complete** |

**Progress Update (2025-10-16 Late Evening)**:
- ✅ **Phases 0-3 Complete**: All MCP infrastructure, lens services, AND orchestration operational!
- ✅ **Phase 4A Complete**: Essential observability & resilience implemented (retry logic, health checks, metrics, tracing)
- ⏸️ **Phase 4B Deferred**: Advanced production features can be added when needed (OTLP export, circuit breakers)
- ⏸️ **Phase 5 Deferred**: Natural language interface is optional enhancement

**Key Achievement**: Core agentic orchestration system is fully functional with essential resilience features!

**Revised Timeline**:
- ✅ Phase 3 (LangGraph coordinator) - **COMPLETE** (same day)
- Week 4: Complete Phase 4 (Advanced observability & resilience) - **OPTIONAL**
- Week 5: Implement Phase 5 (Natural language interface) - **OPTIONAL**

**Note**: Timeline is flexible and can be adjusted based on priorities. Each phase delivers independent value.

---

## Phase 3, 4, 5 Evaluation Based on Current Implementation

### Phase 3: LangGraph Coordinator - Critical Assessment

**Feasibility**: HIGH ✅
- All prerequisites are in place (foundation + all 5 lens services)
- MCP tools have clean interfaces ready for orchestration
- Shared state management already handles context passing

**Scope Refinement Recommendations**:

**3.1. Simplified Intent Parsing (MVP)**
Instead of full LLM-based parsing initially, start with:
- **Rule-based keyword matching** for lens selection
- **Explicit lens parameter passing** (not inferred)
- Example: "Run lens1 and lens3 for Q4 2024"

```python
# Simpler than plan - no LLM needed for MVP
def parse_simple_intent(query: str) -> dict:
    lenses = []
    if "lens1" in query.lower() or "snapshot" in query.lower():
        lenses.append("lens1")
    if "lens2" in query.lower() or "compare" in query.lower():
        lenses.append("lens2")
    # ... etc
    return {"lenses": lenses, "params": extract_dates(query)}
```

**3.2. StateGraph Implementation - Keep Original Plan**
The StateGraph design is sound:
- Nodes for each lens + preparation + synthesis
- Conditional routing based on dependencies
- Parallel execution for independent lenses (1, 3, 4, 5)
- Error handling at node level

**Critical Path**:
1. Define `AnalysisState` TypedDict ✅ (from plan)
2. Create `FourLensesCoordinator` class ✅ (from plan)
3. Implement parallel execution for Lenses 1, 3, 4, 5 ✅ (from plan)
4. Handle Lens 2 dependency on Lens 1 (sequential if both requested)
5. Add basic result aggregation (collect all lens outputs)

**Estimated Effort**: 3-4 days (not full week)
- Day 1: StateGraph structure and node definitions
- Day 2: Parallel execution logic with asyncio.gather
- Day 3: Error handling and result aggregation
- Day 4: Testing and integration

**Risk**: LOW - Well-defined problem with clear inputs/outputs

### Phase 4: Observability & Resilience - Prioritization

**Current Gap Analysis**:

**What's Missing** (from plan):
1. ❌ OpenTelemetry OTLP export (Jaeger/Zipkin)
2. ❌ Automatic retry with tenacity
3. ❌ Circuit breaker with pybreaker
4. ❌ Health check tool
5. ❌ Execution metrics tool

**What's Adequate** (current implementation):
1. ✅ Basic structlog logging (sufficient for development)
2. ✅ Progress reporting (good user experience)
3. ✅ Error propagation (proper exception handling)

**Recommendation**: **DEFER ADVANCED FEATURES**

**Phase 4A: Essential Observability (1-2 days)** - Do This
- Add retry logic to lens execution (tenacity - 2 hours)
- Implement health check MCP tool (1 hour)
- Add basic execution metrics (in-memory counters - 2 hours)
- Document error handling patterns (1 hour)

**Phase 4B: Production Observability (Week)** - Defer Until Need
- Full OpenTelemetry with OTLP (requires infrastructure setup)
- Jaeger/Zipkin deployment and configuration
- Circuit breakers (only needed with external dependencies)
- Advanced tracing across distributed services

**Rationale**:
- Current system is not distributed (all MCP tools in same process)
- No external dependencies that require circuit breaking
- Retry logic + health checks cover 80% of resilience needs
- Full observability stack is overkill until multi-node deployment

**Revised Phase 4 Scope**:
- Essential observability: 1-2 days (do after Phase 3)
- Advanced features: Defer to post-MVP (when scaling becomes necessary)

### Phase 5: Natural Language Interface - Strategic Assessment

**Current Plan Evaluation**:

**Query Interpreter** (from plan):
- Uses Claude API to parse natural language → structured intent
- Returns JSON with lenses, date_range, filters, parameters
- Cost: ~$0.10-0.20 per query (with Claude 3.5 Sonnet)

**Result Synthesizer** (from plan):
- Aggregates multi-lens results
- Generates narrative explanations
- Cost: ~$0.30-0.40 per synthesis (longer context)

**Total LLM Cost**: ~$0.50 per orchestrated analysis ✅ (within plan target)

**Feasibility Assessment**: MEDIUM-HIGH ⚠️

**Challenges Not in Original Plan**:
1. **Prompt Engineering Complexity**
   - Need robust prompts that handle all 5 lenses
   - Date parsing can be ambiguous ("last quarter" vs "Q4 2024")
   - Cohort filtering requires understanding business logic

2. **Cost Control**
   - Dev/test iterations will burn through API credits quickly
   - Need caching layer for repeated queries
   - Should implement token counting and limits

3. **Latency Concerns**
   - Query parsing: ~1-2s (Claude API)
   - Lens execution: ~2-5s (deterministic)
   - Result synthesis: ~2-3s (Claude API)
   - **Total**: 5-10s (target was <5s) ⚠️

**Revised Recommendations**:

**Phase 5A: Hybrid Approach (Recommended)** - 3 days
- Rule-based intent parsing (no LLM) - instant, free
- Deterministic lens execution (existing) - fast
- Optional LLM synthesis for narrative generation only
- Total latency: 2-5s, Cost: $0.10-0.20 per query

**Phase 5B: Full NL Interface (Original Plan)** - 1 week
- LLM-based query interpretation
- LLM-based result synthesis
- Conversational context maintenance
- Requires: prompt engineering, caching, token optimization

**Decision Criteria**:
- If users need "tell me about customer health" → Do Phase 5B (full NL)
- If users are technical and can specify lens parameters → Do Phase 5A (hybrid)
- **Recommendation**: Start with 5A, add 5B based on user feedback

**Cost-Benefit**:
- Phase 5A: Low cost, fast, good for technical users ✅
- Phase 5B: Higher cost/latency, best UX for non-technical users

### Overall Recommendations

**Immediate Priorities** (Next 2 Weeks):
1. ✅ **Phase 3 (LangGraph Coordinator)**: 3-4 days - CRITICAL PATH
   - Use simplified intent parsing (rule-based)
   - Implement scatter-gather for parallel execution
   - Focus on correct orchestration, not fancy NL parsing

2. ⚠️ **Phase 4A (Essential Observability)**: 1-2 days - IMPORTANT
   - Add retry logic
   - Health check tool
   - Basic metrics
   - Skip advanced features until needed

3. ⏸️ **Phase 5A (Hybrid NL)**: 3 days - OPTIONAL ENHANCEMENT
   - Start with rule-based parsing
   - Add optional LLM synthesis for insights
   - Defer full conversational interface

**Defer to Later** (Post-MVP):
- Phase 4B: Advanced observability (OTLP, Jaeger, circuit breakers)
- Phase 5B: Full LLM-based query interpretation and conversation

**Estimated Time to Functional Orchestration**:
- **1 week** (Phase 3 + Phase 4A)
- **2 weeks** if adding Phase 5A

**Key Insight**: The original plan was comprehensive but **over-engineered for MVP**. With all 5 lenses already implemented as MCP tools, the focus should be on:
1. Getting orchestration working correctly (Phase 3)
2. Basic resilience and monitoring (Phase 4A)
3. Iterating based on real usage patterns before investing in full NL interface

---

## Next Steps

**Updated Action Plan** (Based on Evaluation):

1. ✅ **Phases 0-2 Complete**: All lens services operational
2. 🎯 **Begin Phase 3**: LangGraph coordinator (3-4 days)
   - Use simplified intent parsing (rule-based)
   - Implement StateGraph with parallel execution
   - Test orchestrated lens execution
3. 🎯 **Add Phase 4A**: Essential observability (1-2 days)
   - Retry logic for transient failures
   - Health check and basic metrics
4. 🤔 **Evaluate Phase 5**: Based on user needs
   - If technical users: Skip full NL, use rule-based
   - If non-technical users: Invest in LLM-based parsing
5. 📊 **Gather Usage Data**: Monitor Phase 3 performance before over-engineering

---

## References

- [Agentic Five Lenses Architecture Research](../research/2025-10-14-agentic-five-lenses-architecture.md) - Comprehensive research document
- [Track A Completion Status](../research/2025-10-11-track-a-completion-status.md) - Current state analysis
- [Enterprise CLV Implementation Plan](2025-10-08-enterprise-clv-implementation.md) - Original Track A plan
- [FastMCP Documentation](https://gofastmcp.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [MCP Specification](https://modelcontextprotocol.io/)
- [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/)
