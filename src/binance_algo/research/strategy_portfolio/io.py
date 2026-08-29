"""Small atomic-output helpers for non-scientific portfolio reports and config."""

from __future__ import annotations

import uuid
from pathlib import Path

from binance_algo.common.errors import ResearchError


def replace_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise ResearchError(f"cannot write {path}: {exc}") from exc


__all__ = ["replace_bytes"]
