"""Structured JSON logging with defense-in-depth secret redaction."""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Mapping
from typing import Any, cast

import orjson
import structlog
from structlog.typing import EventDict, Processor

SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "api_secret",
        "secret",
        "signature",
        "authorization",
        "x-mbx-apikey",
    }
)
SENSITIVE_QUERY_PATTERN = re.compile(r"(?i)(api[_-]?key|secret|signature|authorization)=([^&\s]+)")


def _mask_text(value: str) -> str:
    return SENSITIVE_QUERY_PATTERN.sub(lambda match: f"{match.group(1)}=[REDACTED]", value)


def redact_sensitive(value: Any, *, key: str | None = None) -> Any:
    """Return a copy with secrets masked, including nested mappings and sequences."""

    if key is not None and key.lower() in SENSITIVE_KEYS:
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {
            str(item_key): redact_sensitive(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive(item) for item in value)
    if isinstance(value, str):
        return _mask_text(value)
    return value


def _redaction_processor(
    _logger: logging.Logger, _method_name: str, event_dict: EventDict
) -> EventDict:
    redacted = redact_sensitive(event_dict)
    if not isinstance(redacted, dict):
        return event_dict
    return redacted


def _json_serializer(value: Any, **_: Any) -> str:
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS).decode("utf-8")


def configure_logging(*, level: str, service: str, environment: str) -> None:
    """Configure structlog once for newline-delimited JSON on stdout."""

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level, force=True)
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
        _redaction_processor,
        structlog.processors.JSONRenderer(serializer=_json_serializer),
    ]
    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(service=service, environment=environment)


def get_logger(**initial_values: Any) -> structlog.stdlib.BoundLogger:
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger().bind(**initial_values))
