#!/usr/bin/env python3
"""
Test script to verify OTLP tracing to Jaeger works.

This creates a simple trace and sends it to Jaeger to verify connectivity.
"""

import os
import time

# Set environment variables
os.environ["OTLP_ENDPOINT"] = "localhost:4317"
os.environ["ENVIRONMENT"] = "development"

from analytics.services.mcp_server.observability import configure_observability

# Configure observability
tracer, meter = configure_observability(
    service_name="mcp-four-lenses",
    environment="development",
    otlp_endpoint="localhost:4317",
    sampling_rate=1.0,
)

print("✓ Observability configured")
print("✓ Creating test trace...")

# Create a test span
with tracer.start_as_current_span("test_trace") as span:
    span.set_attribute("test.type", "connectivity_check")
    span.set_attribute("test.timestamp", time.time())

    # Simulate some work
    time.sleep(0.1)

    # Create nested span
    with tracer.start_as_current_span("test_nested_operation") as nested_span:
        nested_span.set_attribute("operation", "nested_test")
        time.sleep(0.05)

print("✓ Test trace created")
print("\nWait 2-3 seconds for batch export, then check Jaeger UI:")
print("  http://localhost:16686")
print("  Service: mcp-four-lenses")
print("  Operation: test_trace")

# Give time for batch export
time.sleep(3)
print("\n✓ Test complete! Check Jaeger UI now.")
