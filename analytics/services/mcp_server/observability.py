"""
Observability configuration for MCP Server

This module configures OpenTelemetry tracing, metrics, and structured logging
for the Four Lenses Analytics MCP server.

Phase 4A: Basic console exporting for development
Phase 4B: Production OTLP export to Jaeger/Zipkin/Cloud providers
"""

import os
import sys
from typing import Literal

import structlog
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

logger = structlog.get_logger(__name__)


def configure_observability(
    service_name: str = "mcp-four-lenses",
    environment: str = "development",
    otlp_endpoint: str | None = None,
    sampling_rate: float = 1.0,
):
    """
    Configure OpenTelemetry tracing and metrics.

    Phase 4A: Development mode (console export)
    Phase 4B: Production mode (OTLP export to Jaeger/Zipkin)

    Args:
        service_name: Name of the service for telemetry identification
        environment: Deployment environment (development, staging, production)
        otlp_endpoint: OTLP gRPC endpoint (e.g., 'localhost:4317' for Jaeger)
                      If None, uses environment variable OTLP_ENDPOINT or defaults to console
        sampling_rate: Trace sampling rate (0.0-1.0). Default 1.0 = sample all traces.
                      Use lower values (e.g., 0.1) in high-volume production.

    Returns:
        Tuple of (tracer, meter) for creating spans and metrics
    """

    # Get OTLP endpoint from parameter or environment
    otlp_endpoint = otlp_endpoint or os.getenv("OTLP_ENDPOINT")
    use_otlp = otlp_endpoint is not None

    # Create resource with service metadata
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "1.0.0",
            "deployment.environment": environment,
        }
    )

    # Configure tracing
    trace_provider = TracerProvider(resource=resource, sampler=_create_sampler(sampling_rate))

    if use_otlp:
        # Phase 4B: Production OTLP export
        logger.info(
            "configuring_otlp_tracing",
            endpoint=otlp_endpoint,
            environment=environment,
            sampling_rate=sampling_rate,
        )
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            # Use insecure connection for localhost (no TLS)
            otlp_exporter = OTLPSpanExporter(
                endpoint=otlp_endpoint,
                insecure=True  # Required for localhost without TLS
            )
            trace_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info("otlp_tracing_configured", endpoint=otlp_endpoint)
        except ImportError as e:
            logger.warning(
                "otlp_exporter_not_available_falling_back_to_console",
                error=str(e),
                message="Install opentelemetry-exporter-otlp-proto-grpc for production OTLP export",
            )
            trace_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    else:
        # Phase 4A: Development console export
        logger.info("configuring_console_tracing", environment=environment)
        trace_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(trace_provider)

    # Configure metrics
    if use_otlp:
        # Phase 4B: Production OTLP metrics export
        logger.info("configuring_otlp_metrics", endpoint=otlp_endpoint)
        try:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )

            # Use insecure connection for localhost (no TLS)
            otlp_metric_exporter = OTLPMetricExporter(
                endpoint=otlp_endpoint,
                insecure=True  # Required for localhost without TLS
            )
            metric_reader = PeriodicExportingMetricReader(
                otlp_metric_exporter, export_interval_millis=60000  # 60s intervals
            )
            meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
            logger.info("otlp_metrics_configured", endpoint=otlp_endpoint)
        except ImportError:
            logger.warning(
                "otlp_metric_exporter_not_available_falling_back_to_console",
                message="Install opentelemetry-exporter-otlp-proto-grpc for production metrics export",
            )
            metric_reader = PeriodicExportingMetricReader(
                ConsoleMetricExporter(), export_interval_millis=5000
            )
            meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    else:
        # Phase 4A: Development console export
        logger.info("configuring_console_metrics")
        metric_reader = PeriodicExportingMetricReader(
            ConsoleMetricExporter(), export_interval_millis=5000
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])

    metrics.set_meter_provider(meter_provider)

    # Configure structured logging - write to stderr to avoid interfering with MCP JSON on stdout
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.WriteLoggerFactory(file=sys.stderr),
    )

    tracer = trace.get_tracer(service_name)
    meter = metrics.get_meter(service_name)

    logger.info(
        "observability_configured",
        service_name=service_name,
        environment=environment,
        otlp_enabled=use_otlp,
        otlp_endpoint=otlp_endpoint if use_otlp else None,
        sampling_rate=sampling_rate,
    )

    return tracer, meter


def _create_sampler(sampling_rate: float):
    """Create a trace sampler based on sampling rate.

    Args:
        sampling_rate: Sampling rate between 0.0 and 1.0

    Returns:
        OpenTelemetry Sampler instance
    """
    from opentelemetry.sdk.trace.sampling import (
        ParentBasedTraceIdRatio,
        TraceIdRatioBased,
    )

    if sampling_rate >= 1.0:
        # Sample all traces (default for development)
        return ParentBasedTraceIdRatio(1.0)
    elif sampling_rate <= 0.0:
        # Sample no traces (not recommended, but supported)
        return TraceIdRatioBased(0.0)
    else:
        # Sample based on trace ID ratio (for production)
        return ParentBasedTraceIdRatio(sampling_rate)


def configure_production_telemetry(
    service_name: str = "mcp-five-lenses",
    otlp_endpoint: str = "localhost:4317",
    environment: Literal["staging", "production"] = "production",
    sampling_rate: float = 1.0,
):
    """Configure production OpenTelemetry with OTLP export.

    This is a convenience function for production deployments. It configures:
    - OTLP gRPC export to Jaeger, Zipkin, or cloud providers
    - Service metadata and resource attributes
    - Sampling strategies for high-volume scenarios
    - Batch span processing with appropriate timeouts

    Args:
        service_name: Name of the service for telemetry identification
        otlp_endpoint: OTLP gRPC endpoint (e.g., 'localhost:4317' for Jaeger)
        environment: Deployment environment (staging or production)
        sampling_rate: Trace sampling rate (0.0-1.0). Default 1.0 = sample all traces.
                      Use lower values (e.g., 0.1) for high-volume production.

    Returns:
        Tuple of (tracer, meter) for creating spans and metrics

    Example:
        >>> tracer, meter = configure_production_telemetry(
        ...     service_name="mcp-five-lenses",
        ...     otlp_endpoint="jaeger:4317",
        ...     environment="production",
        ...     sampling_rate=0.1  # Sample 10% of traces
        ... )
    """
    return configure_observability(
        service_name=service_name,
        environment=environment,
        otlp_endpoint=otlp_endpoint,
        sampling_rate=sampling_rate,
    )
