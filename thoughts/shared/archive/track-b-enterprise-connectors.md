# Track-B Enterprise Infrastructure Archive

**Archive Date**: 2025-10-22
**Git Tag**: `archive/track-b-enterprise-exploration`
**Commit**: `769bd0f2d226a4960823d2d1fdcb45cb67dd79fb`
**Decision**: Archived in favor of Track-A

## Executive Summary

Track-B explored enterprise-grade infrastructure for data warehouse integrations, streaming analytics, and ML operations. While this infrastructure represents significant engineering effort, it was archived because:

1. **Core Lens implementations incomplete**: Lens 4 and Lens 5 were TODO stubs
2. **Bugs in foundational code**: Lens 1 precision bug (1 decimal vs 2 decimals)
3. **Missing data quality safeguards**: Lens 2 lacked extreme change warnings
4. **Architectural drift**: Different patterns from Track-A's cleaner implementation

Track-A provides superior implementations of the Five Lenses framework with better architectural patterns, while Track-B's enterprise infrastructure can be re-implemented if needed based on this documentation.

---

## Architecture Overview

### Directory Structure

```
analytics/
├── libs/
│   ├── data_warehouse/         # Data warehouse connectors
│   │   ├── connectors/
│   │   │   ├── base.py         # Abstract connector interface
│   │   │   ├── bigquery.py     # Google BigQuery connector
│   │   │   ├── redshift.py     # AWS Redshift connector
│   │   │   ├── snowflake.py    # Snowflake connector
│   │   │   ├── factory.py      # Connector factory pattern
│   │   │   ├── config.py       # Connection configuration
│   │   │   └── pool_proxy.py   # Connection pooling proxy
│   │   ├── olap/
│   │   │   ├── engine.py       # OLAP query engine
│   │   │   ├── cube.py         # OLAP cube operations
│   │   │   └── operations.py   # Slice/dice/drill operations
│   │   ├── query/
│   │   │   ├── cache.py        # Query result caching
│   │   │   └── federation.py  # Federated queries
│   │   ├── pool.py             # Connection pool management
│   │   └── dependencies.py     # Dependency injection
│   │
│   ├── streaming_analytics/    # Real-time streaming
│   │   ├── kafka_manager.py    # Apache Kafka integration
│   │   ├── processor.py        # Stream processing
│   │   ├── realtime_ml.py      # Real-time ML inference
│   │   ├── websocket_server.py # WebSocket API
│   │   ├── event_store.py      # Event schema/storage
│   │   ├── monitoring.py       # Stream metrics
│   │   ├── circuit_breaker.py  # Fault tolerance
│   │   └── connection_pool.py  # Connection management
│   │
│   ├── ml_models/              # ML infrastructure
│   │   ├── registry.py         # Model registry
│   │   ├── versioning.py       # Model versioning
│   │   ├── serving.py          # Model serving
│   │   ├── monitoring.py       # Model drift detection
│   │   ├── retraining.py       # Auto-retraining
│   │   ├── feature_store.py    # Feature management
│   │   ├── experiments.py      # Experiment tracking
│   │   └── ab_testing.py       # A/B testing framework
│   │
│   ├── observability/          # Monitoring & tracing
│   ├── workflow_orchestration/ # Airflow/Dagster
│   ├── analytics_core/         # Core analytics
│   ├── api_common/             # Shared API utilities
│   ├── config/                 # Configuration management
│   └── data_processing/        # ETL pipelines
│
├── services/
│   └── analytics_api/
│       └── routes/
│           └── data_warehouse.py  # REST API endpoints
│
└── docker/
    ├── streaming-analytics/    # Kafka/Flink deployment
    ├── observability/          # Prometheus/Grafana/OTEL
    └── Dockerfile.ml-inference # ML serving container
```

---

## 1. Data Warehouse Connectors

### Overview

Unified interface for connecting to enterprise data warehouses with:
- Async/await support for scalable operations
- Connection pooling with intelligent resource management
- Schema discovery and metadata operations
- Federated queries across multiple sources
- Query result caching and optimization
- Security features (credential sanitization, SQL injection prevention)

### Supported Warehouses

| Warehouse | File | Lines | Key Features |
|-----------|------|-------|--------------|
| **Google BigQuery** | `bigquery.py` | 389 | Streaming inserts, partition management, cost optimization |
| **AWS Redshift** | `redshift.py` | 358 | Cluster management, WLM queues, spectrum queries |
| **Snowflake** | `snowflake.py` | 425 | Virtual warehouse scaling, multi-cluster warehouses |
| **Azure Synapse** | Planned | - | Not implemented |
| **Databricks** | Planned | - | Not implemented |

### Base Connector Interface

**File**: `analytics/libs/data_warehouse/connectors/base.py` (289 lines)

**Key Classes**:

```python
class DataWarehouseConnector(ABC):
    """Abstract base class for all warehouse connectors."""

    @abstractmethod
    async def connect(self) -> ConnectionStatus
    @abstractmethod
    async def disconnect(self) -> None
    @abstractmethod
    async def execute_query(self, query: str) -> QueryResult
    @abstractmethod
    async def get_schema_info(self, table_name: str) -> SchemaInfo
    @abstractmethod
    async def health_check(self) -> HealthCheckResult
```

**Security Features**:
- `sanitize_error_message()`: Removes passwords, tokens, hosts from error logs
- `validate_sql_identifier()`: Prevents SQL injection in table/column names
- Regex patterns for sensitive data masking

**Data Structures**:
- `QueryResult`: Rows, metadata, execution stats
- `SchemaInfo`: Tables, columns, types, constraints
- `ConnectionStatus`: CONNECTED, DISCONNECTED, ERROR, CONNECTING
- `WarehouseType`: Enum of supported warehouses

### Connection Factory Pattern

**File**: `analytics/libs/data_warehouse/connectors/factory.py` (358 lines)

**Features**:
- Registry-based connector instantiation
- Configuration validation
- Credential injection from environment/secrets manager
- Connection pool integration
- Health check orchestration

**Usage Pattern**:
```python
factory = ConnectorFactory()
connector = await factory.create_connector(
    warehouse_type=WarehouseType.BIGQUERY,
    config=BigQueryConfig(project_id="...", dataset="...")
)
await connector.connect()
result = await connector.execute_query("SELECT * FROM table")
```

### Connection Pooling

**File**: `analytics/libs/data_warehouse/pool.py` (358 lines)

**Features**:
- Min/max pool size configuration
- Connection lifecycle management (acquire, release, evict)
- Health checks on idle connections
- Metrics: pool size, active connections, wait time
- Graceful shutdown with connection draining

### OLAP Engine

**Files**:
- `analytics/libs/data_warehouse/olap/engine.py` (358 lines)
- `analytics/libs/data_warehouse/olap/cube.py` (325 lines)
- `analytics/libs/data_warehouse/olap/operations.py` (321 lines)

**Operations**:
- **Slice**: Filter on single dimension (e.g., "show Q1 2024 only")
- **Dice**: Filter on multiple dimensions (e.g., "Q1 2024, USA, Premium customers")
- **Drill-down**: Increase granularity (e.g., year → quarter → month)
- **Roll-up**: Decrease granularity (e.g., month → quarter → year)
- **Pivot**: Rotate cube to view different perspectives

**Use Case**: Power multi-dimensional cohort analysis for Five Lenses

---

## 2. Streaming Analytics Infrastructure

### Overview

Real-time event processing infrastructure using Apache Kafka with:
- Stream processing with windowing and aggregations
- Real-time ML inference on event streams
- WebSocket API for live dashboard updates
- Event schema registry and validation
- Exactly-once processing guarantees
- Auto-scaling based on stream volume

### Kafka Manager

**File**: `analytics/libs/streaming_analytics/kafka_manager.py` (1,021 lines)

**Features**:
- Producer with batching, compression, idempotency
- Consumer with offset management, rebalancing
- Topic management (creation, partition scaling, retention)
- Schema registry integration (Avro/Protobuf)
- Dead letter queue for failed messages
- Metrics: throughput, lag, error rates

**Key Capabilities**:
```python
# Producer
await kafka.produce(
    topic="customer_events",
    key=customer_id,
    value=event_data,
    headers={"schema_version": "v2"}
)

# Consumer with exactly-once semantics
async for message in kafka.consume(
    topic="customer_events",
    group_id="clv_processor",
    enable_idempotence=True
):
    await process_event(message)
    await kafka.commit_offset(message)
```

### Stream Processor

**File**: `analytics/libs/streaming_analytics/processor.py` (684 lines)

**Window Types**:
- **Tumbling**: Fixed, non-overlapping (e.g., 1-hour windows)
- **Sliding**: Fixed size, overlapping (e.g., 5-min window every 1 min)
- **Session**: Dynamic based on inactivity gap

**Aggregation Types**:
- Count, Sum, Average, Min/Max
- Percentiles (P50, P95, P99)
- Distinct count (HyperLogLog)
- Custom aggregations

**Use Case**: Real-time CLV calculation as customers transact

### Real-time ML Pipeline

**File**: `analytics/libs/streaming_analytics/realtime_ml.py` (1,265 lines)

**Features**:
- Model loading from registry
- Feature extraction from events
- Batched inference for efficiency
- Model version routing (A/B testing)
- Prediction caching
- Drift detection

**Pipeline Flow**:
```
Event Stream → Feature Extraction → Model Inference → Prediction Stream
                     ↓
               Feature Store
```

**Use Case**: Real-time churn prediction, next purchase prediction

### WebSocket Server

**File**: `analytics/libs/streaming_analytics/websocket_server.py` (1,015 lines)

**Features**:
- Connection management (1000s of concurrent connections)
- Topic subscriptions (client subscribes to specific metrics)
- Heartbeat/keepalive
- Rate limiting per client
- Authentication/authorization
- Message compression

**Use Case**: Live CLV dashboards updating in real-time

### Event Store

**File**: `analytics/libs/streaming_analytics/event_store.py` (298 lines)

**Features**:
- Event schema validation (Pydantic models)
- Event sourcing pattern support
- Snapshot management
- Event replay capabilities
- Temporal queries (events at time T)

**Event Types**:
- `UserActionEvent`: Purchases, page views, clicks
- `SystemEvent`: Errors, deployments, scaling events
- `MetricEvent`: Calculated metrics (CLV, health scores)

---

## 3. ML Infrastructure

### Model Registry

**File**: `analytics/libs/ml_models/registry.py` (726 lines)

**Features**:
- Model versioning with semantic versioning
- Model metadata (metrics, hyperparameters, training data)
- Model lineage tracking
- Model promotion workflow (dev → staging → production)
- Model archival and rollback
- Integration with MLflow, Weights & Biases

**Model Lifecycle**:
```
Train → Register → Validate → Promote → Serve → Monitor → Retrain
```

### Model Serving

**File**: `analytics/libs/ml_models/serving.py` (979 lines)

**Features**:
- Model loading with caching
- Batched inference
- Multi-model serving (host multiple models)
- A/B testing support
- Shadow mode (test new models without affecting production)
- Dynamic batching for GPU utilization
- ONNX runtime support for fast inference

**API Pattern**:
```python
# Serve prediction
prediction = await model_server.predict(
    model_name="clv_predictor",
    version="v3.2.1",
    features=customer_features
)
```

### Feature Store

**File**: `analytics/libs/ml_models/feature_store.py` (766 lines)

**Features**:
- Feature definition and versioning
- Feature computation (batch and streaming)
- Feature materialization (pre-compute and cache)
- Point-in-time correct features (avoid data leakage)
- Feature monitoring (drift, quality)
- Integration with Feast

**Feature Groups**:
- Customer features (total_spend, order_frequency, recency)
- Cohort features (cohort_id, acquisition_date, cohort_quality)
- Temporal features (day_of_week, seasonality, trend)

### Model Monitoring

**File**: `analytics/libs/ml_models/monitoring.py` (1,025 lines)

**Metrics Tracked**:
- **Data Drift**: Input feature distribution changes (KS test, PSI)
- **Model Drift**: Prediction distribution changes
- **Performance Degradation**: Accuracy, precision, recall drop
- **Latency**: P50, P95, P99 inference time
- **Throughput**: Predictions per second

**Alerting**:
- Drift threshold exceeded
- Performance below baseline
- Error rate spike
- Resource exhaustion

### Auto-Retraining

**File**: `analytics/libs/ml_models/retraining.py` (850 lines)

**Triggers**:
- Scheduled (daily, weekly)
- Drift-based (data or model drift detected)
- Performance-based (accuracy drops below threshold)
- Manual (operator initiated)

**Pipeline**:
```
Trigger → Data Prep → Train → Validate → Compare → Promote/Reject
```

**Safety Checks**:
- New model must outperform old model by >5% on holdout set
- A/B test on 10% traffic before full rollout
- Canary deployment with automatic rollback

---

## 4. Key Design Patterns

### 1. Connector Factory Pattern

**Problem**: Need flexible way to instantiate different warehouse connectors
**Solution**: Registry-based factory with configuration injection

```python
# Registration
factory.register_connector(WarehouseType.BIGQUERY, BigQueryConnector)

# Usage
connector = factory.create_connector(
    warehouse_type=WarehouseType.BIGQUERY,
    config=config
)
```

### 2. Connection Pooling

**Problem**: Creating new connections is expensive
**Solution**: Pool of reusable connections with lifecycle management

**Benefits**:
- Reduced connection overhead (10-100x faster)
- Resource limits (prevent connection exhaustion)
- Health monitoring (evict stale connections)

### 3. Circuit Breaker

**File**: `analytics/libs/streaming_analytics/circuit_breaker.py` (278 lines)

**Problem**: Cascading failures when downstream service is unhealthy
**Solution**: Trip circuit after N consecutive failures, auto-recover

**States**:
- **CLOSED**: Normal operation, requests pass through
- **OPEN**: Service unhealthy, fail fast without calling
- **HALF_OPEN**: Testing recovery, allow 1 request through

### 4. Event Sourcing

**Problem**: Need audit trail and ability to replay history
**Solution**: Store all state changes as immutable events

**Benefits**:
- Complete audit trail
- Temporal queries (state at time T)
- Replay for debugging or data migration
- Event-driven architecture

---

## 5. Notable Implementation Details

### Security Features

**Credential Sanitization** (`base.py:17-45`):
```python
def sanitize_error_message(error_message: str) -> str:
    """Remove passwords, tokens, hosts from error messages."""
    sensitive_patterns = [
        (r'password[=:]\s*[\'"][^\'";]+[\'"]', "password=***"),
        (r'token[=:]\s*[\'"][^\'";]+[\'"]', "token=***"),
        # ... more patterns
    ]
    # Prevents credential leaks in logs/Sentry
```

**SQL Injection Prevention** (`base.py:48-84`):
```python
def validate_sql_identifier(identifier: str) -> str:
    """Validate table/column names to prevent injection."""
    # Check for malicious patterns: ;, --, DROP, etc.
    # Allow only alphanumeric + underscore + dot
    # Limit length to 128 chars
```

### Performance Optimizations

**Batched Inference** (`realtime_ml.py`):
- Accumulate requests for 50ms or 32 samples
- Single GPU call instead of 32 individual calls
- 10-20x throughput improvement

**Query Result Caching** (`query/cache.py`):
- LRU cache with TTL
- Cache key includes query + parameters
- Invalidation on data updates

**Connection Pooling** (`pool.py`):
- Warm pool (pre-connect on startup)
- Connection reuse (avoid handshake overhead)
- Idle timeout (evict unused connections)

### Fault Tolerance

**Exactly-Once Processing** (`kafka_manager.py`):
- Idempotent producer (deduplicate on broker)
- Transactional consumer (commit offset atomically)
- Use case: Ensure CLV calculations count each transaction exactly once

**Dead Letter Queue** (`kafka_manager.py`):
- Failed messages sent to DLQ topic
- Includes failure reason and retry count
- Manual review and reprocessing

**Circuit Breaker** (`circuit_breaker.py`):
- Fail fast when service is down
- Auto-recovery with exponential backoff
- Prevents cascade failures

---

## 6. Comparison to Track-A

### What Track-B Had

| Component | Status | Notes |
|-----------|--------|-------|
| Data Warehouse Connectors | ✅ Fully implemented | BigQuery, Redshift, Snowflake |
| OLAP Engine | ✅ Fully implemented | Slice, dice, drill operations |
| Streaming Analytics | ✅ Fully implemented | Kafka, WebSocket, real-time ML |
| ML Infrastructure | ✅ Fully implemented | Registry, serving, monitoring, auto-retrain |
| Observability | ✅ Fully implemented | Prometheus, Grafana, OTEL, Jaeger |
| Lens 1-3 | ⚠️ Implemented with bugs | Precision bug (Lens 1), missing warnings (Lens 2) |
| Lens 4-5 | ❌ TODO stubs | Not implemented |

### What Track-A Has

| Component | Status | Notes |
|-----------|--------|-------|
| Data Warehouse Connectors | ❌ Not implemented | Can re-implement from Track-B docs |
| OLAP Engine | ❌ Not implemented | Can re-implement if needed |
| Streaming Analytics | ❌ Not implemented | Can re-implement if needed |
| ML Infrastructure | ❌ Not implemented | Can re-implement if needed |
| Observability | ⚠️ Basic implementation | OpenTelemetry tracing in MCP server |
| Lens 1-3 | ✅ Implemented correctly | No bugs, better patterns |
| Lens 4-5 | ✅ Fully implemented | 822 + 905 lines respectively |

### Why Track-A Won

1. **Core mission complete**: All Five Lenses fully implemented
2. **Higher code quality**: Better constants, validation, logging
3. **No bugs**: Fixed Lens 1 precision, added Lens 2 warnings
4. **Better architecture**: Cleaner patterns, better documentation
5. **MCP server ready**: Track-A has working MCP server, Track-B does not

---

## 7. Migration Path (If Needed)

If enterprise infrastructure is needed in the future:

### Phase 1: Data Warehouse Connectors (2-3 weeks)

1. Copy `analytics/libs/data_warehouse/` to Track-A
2. Update imports for Track-A's package structure
3. Add tests using Track-A's pytest fixtures
4. Integrate with MCP server tools

**Value**: Query customer data from BigQuery/Redshift/Snowflake

### Phase 2: Streaming Analytics (3-4 weeks)

1. Copy `analytics/libs/streaming_analytics/` to Track-A
2. Set up Kafka in docker-compose
3. Create stream processor for real-time CLV updates
4. Add WebSocket endpoints to MCP server

**Value**: Real-time CLV dashboards

### Phase 3: ML Infrastructure (4-6 weeks)

1. Copy `analytics/libs/ml_models/` to Track-A
2. Set up model registry (MLflow)
3. Create feature store for CLV features
4. Deploy model serving API

**Value**: Predictive CLV, churn prediction, next purchase

### Phase 4: Full Observability (1-2 weeks)

1. Copy observability configs from `analytics/docker/observability/`
2. Set up Prometheus + Grafana dashboards
3. Configure alerting rules
4. Add distributed tracing

**Value**: Production-grade monitoring

---

## 8. Key Learnings from Track-B

### What Worked Well

1. **Unified connector interface**: Made switching warehouses trivial
2. **Security-first design**: Credential sanitization, SQL validation built-in
3. **Circuit breaker pattern**: Prevented cascade failures
4. **OLAP engine**: Powerful abstraction for multi-dimensional analysis
5. **Feature store**: Point-in-time correctness critical for ML

### What Didn't Work

1. **Too much infrastructure, too soon**: Built enterprise stack before core product
2. **Lens implementations neglected**: Lens 4-5 never finished
3. **Bugs in foundation**: Lens 1 precision bug went unnoticed
4. **Architectural drift**: Different patterns from main codebase
5. **No integration**: MCP server and enterprise stack never connected

### Best Practices to Adopt

1. **Core first, infrastructure later**: Finish Five Lenses before building connectors
2. **Security by default**: Adopt credential sanitization and SQL validation
3. **Observability from day 1**: Add metrics/tracing early, not later
4. **Test edge cases**: Track-A caught precision bugs Track-B missed
5. **Stay focused**: Don't build features you don't need yet

---

## 9. Reference Implementation Files

For future re-implementation, these files are exemplars:

### Data Warehouse
- **Best connector**: `analytics/libs/data_warehouse/connectors/bigquery.py` (clean API)
- **Best abstraction**: `analytics/libs/data_warehouse/connectors/base.py` (security features)
- **Best pattern**: `analytics/libs/data_warehouse/connectors/factory.py` (registry pattern)

### Streaming
- **Best Kafka wrapper**: `analytics/libs/streaming_analytics/kafka_manager.py` (comprehensive)
- **Best processor**: `analytics/libs/streaming_analytics/processor.py` (windowing)
- **Best fault tolerance**: `analytics/libs/streaming_analytics/circuit_breaker.py` (simple, effective)

### ML
- **Best registry**: `analytics/libs/ml_models/registry.py` (versioning, lineage)
- **Best serving**: `analytics/libs/ml_models/serving.py` (batching, A/B testing)
- **Best monitoring**: `analytics/libs/ml_models/monitoring.py` (drift detection)

---

## 10. Conclusion

Track-B represents a significant exploration of enterprise data infrastructure. While it won't be merged into Track-A, this documentation preserves the key learnings and implementation patterns for future use.

**Key Takeaway**: Build the product first (Five Lenses), then add enterprise infrastructure when customers demand it. Track-A's focused approach on core functionality proved superior to Track-B's premature optimization for enterprise scale.

**Archive Status**: Safely preserved at `archive/track-b-enterprise-exploration` (commit `769bd0f`)

---

## Appendix: File Statistics

### Track-B Analytics Library

| Library | Files | Total LOC | Purpose |
|---------|-------|-----------|---------|
| `data_warehouse` | 17 | ~12,500 | Warehouse connectors, OLAP, federation |
| `streaming_analytics` | 13 | ~18,000 | Kafka, stream processing, real-time ML |
| `ml_models` | 14 | ~21,000 | Model registry, serving, monitoring |
| `observability` | 9 | ~6,500 | Prometheus, Grafana, OTEL, Jaeger |
| `workflow_orchestration` | 8 | ~4,800 | Airflow DAGs, task scheduling |
| `analytics_core` | 10 | ~8,200 | Core analytics utilities |
| `api_common` | 7 | ~3,100 | Shared API middleware |
| `data_processing` | 7 | ~5,900 | ETL pipelines, transformations |
| **Total** | **85** | **~80,000** | Enterprise analytics platform |

### Track-A Core (for comparison)

| Component | Files | Total LOC | Purpose |
|-----------|-------|-----------|---------|
| `customer_base_audit` | 47 | ~15,000 | Five Lenses framework |
| `mcp` (server + tools) | 34 | ~8,500 | MCP server, LangGraph orchestration |
| `tests` | 52 | ~12,000 | Comprehensive test suite |
| **Total** | **133** | **~35,500** | CLV analysis MCP server |

**Observation**: Track-B has 2.2x more code than Track-A, but Track-A has complete Lens implementations while Track-B does not. More code ≠ more value.
