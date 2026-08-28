"""Canonical JSON rules shared by experiment and result identities."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import is_dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path, PurePath
from typing import Any

import orjson
from pydantic import BaseModel

from binance_algo.common.errors import ResearchError

_WINDOWS_ABSOLUTE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")


def _decimal_string(value: Decimal) -> str:
    if not value.is_finite():
        raise ResearchError("canonical identity rejects non-finite decimals")
    normalized = format(value, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return "0" if normalized in {"", "-0"} else normalized


def _is_absolute_path_string(value: str) -> bool:
    return value.startswith("/") or bool(_WINDOWS_ABSOLUTE.match(value))


def canonicalize(value: Any, *, field_path: str = "root") -> Any:
    """Convert supported values to a deterministic, path-free JSON representation."""

    if isinstance(value, BaseModel):
        return canonicalize(value.model_dump(mode="python"), field_path=field_path)
    if is_dataclass(value) and not isinstance(value, type):
        raise ResearchError(
            f"canonical identity requires an explicit model, not a dataclass at {field_path}"
        )
    if isinstance(value, Enum):
        return canonicalize(value.value, field_path=field_path)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if not -(2**63) <= value <= 2**63 - 1:
            raise ResearchError(f"canonical identity integer exceeds 64-bit range at {field_path}")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ResearchError(f"canonical identity rejects NaN or infinity at {field_path}")
        return value
    if isinstance(value, Decimal):
        return _decimal_string(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ResearchError(f"canonical identity requires timezone-aware UTC at {field_path}")
        utc_value = value.astimezone(UTC)
        return utc_value.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, PurePath):
        if value.is_absolute():
            raise ResearchError(f"absolute path is forbidden in identity at {field_path}")
        return value.as_posix()
    if isinstance(value, str):
        if _is_absolute_path_string(value):
            raise ResearchError(f"absolute path string is forbidden in identity at {field_path}")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ResearchError(f"canonical identity requires string keys at {field_path}")
        output: dict[str, Any] = {}
        for key in sorted(value):
            if _is_absolute_path_string(key):
                raise ResearchError(f"absolute path key is forbidden in identity at {field_path}")
            output[key] = canonicalize(value[key], field_path=f"{field_path}.{key}")
        return output
    if isinstance(value, (list, tuple)):
        return [
            canonicalize(item, field_path=f"{field_path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ResearchError(
        f"canonical identity cannot serialize {type(value).__name__} at {field_path}"
    )


def canonical_json(value: Any) -> bytes:
    return orjson.dumps(canonicalize(value), option=orjson.OPT_SORT_KEYS)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def canonical_json_text(value: Any) -> str:
    return canonical_json(value).decode("utf-8")


def relative_path(value: Path) -> str:
    """Explicit helper for callers that intentionally persist non-identity relative paths."""

    if value.is_absolute():
        raise ResearchError("expected a relative path")
    return value.as_posix()


__all__ = ["canonical_json", "canonical_json_text", "canonical_sha256", "canonicalize"]
