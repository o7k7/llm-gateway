"""OpenTelemetry provider and exporter setup.

Default export target: Langfuse OTLP endpoint (Langfuse Cloud or self-hosted).
Authenticated via HTTP Basic auth using LANGFUSE_PUBLIC_KEY : LANGFUSE_SECRET_KEY.
"""
from __future__ import annotations

import base64
import logging

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)

from app.config import Config

logger = logging.getLogger(__name__)


def configure_observability(config: Config) -> None:
    """Initialize the global OTel TracerProvider.
    """
    if trace.get_tracer_provider().__class__.__name__ == "TracerProvider":
        logger.info("TracerProvider already configured; skipping")
        return

    resource = Resource.create(
        {
            "service.name": config.otel_service_name,
            "service.version": config.service_version,
            "deployment.environment": config.env,
        }
    )
    provider = TracerProvider(resource=resource)

    if config.langfuse_pub_key and config.langfuse_secret_key:
        exporter = _build_langfuse_exporter(config)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        logger.info(
            "OTel configured: exporting to Langfuse at %s",
            config.langfuse_host,
        )
    elif config.env == "dev":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        logger.info("OTel configured: console exporter (dev mode)")
    else:
        logger.warning(
            "OTel: no exporter configured. Spans will be created but not exported."
        )

    trace.set_tracer_provider(provider)


def shutdown_observability() -> None:
    """Flush any pending spans before the process exits."""
    provider = trace.get_tracer_provider()

    if hasattr(provider, "shutdown"):
        provider.shutdown()


def _build_langfuse_exporter(config: Config) -> OTLPSpanExporter:
    """Build an OTLP HTTP exporter targeting Langfuse.
    """
    pk = config.langfuse_pub_key.get_secret_value()
    sk = config.langfuse_secret_key.get_secret_value()
    auth = base64.b64encode(f"{pk}:{sk}".encode()).decode()

    endpoint = f"{config.langfuse_host.rstrip('/')}/api/public/otel/v1/traces"
    return OTLPSpanExporter(
        endpoint=endpoint,
        headers={"Authorization": f"Basic {auth}"},
    )
