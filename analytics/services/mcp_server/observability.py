"""
Observability configuration for MCP Server

This module configures OpenTelemetry tracing, metrics, and structured logging
for the Four Lenses Analytics MCP server.
"""

import sys

import structlog
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter


def configure_observability(service_name: str = "mcp-four-lenses"):
    """
    Configure OpenTelemetry tracing and metrics.

    Args:
        service_name: Name of the service for telemetry identification

    Returns:
        Tuple of (tracer, meter) for creating spans and metrics
    """

    # Configure tracing
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(trace_provider)

    # Configure metrics
    metric_reader = PeriodicExportingMetricReader(
        ConsoleMetricExporter(), export_interval_millis=5000
    )
    meter_provider = MeterProvider(metric_readers=[metric_reader])
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

    return tracer, meter
