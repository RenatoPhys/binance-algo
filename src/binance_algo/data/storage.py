"""Local atomic storage behind a small future-compatible protocol."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Literal, Protocol, cast

import orjson
import polars as pl

from binance_algo.common.errors import StorageError


class Storage(Protocol):
    root: Path

    def path(self, *parts: str) -> Path: ...

    def write_bytes_atomic(self, target: Path, payload: bytes) -> Path: ...

    def write_parquet_atomic(
        self, target: Path, frame: pl.DataFrame, *, compression: str
    ) -> Path: ...


class LocalFilesystemStorage:
    """Writes only below its configured root and promotes validated temp files atomically."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def path(self, *parts: str) -> Path:
        candidate = self.root.joinpath(*parts).resolve()
        if not candidate.is_relative_to(self.root):
            raise StorageError(f"storage path escapes configured root: {candidate}")
        return candidate

    def write_json_atomic(self, target: Path, value: object) -> Path:
        payload = orjson.dumps(value, option=orjson.OPT_SORT_KEYS) + b"\n"
        return self.write_bytes_atomic(target, payload)

    def write_bytes_atomic(self, target: Path, payload: bytes) -> Path:
        self._assert_target(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() == payload:
                return target
            raise StorageError(f"immutable target already exists with different content: {target}")

        temporary = target.with_name(f".{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except OSError as exc:
            if temporary.exists():
                temporary.unlink()
            raise StorageError(f"atomic write failed for {target}: {exc}") from exc
        return target

    def write_parquet_atomic(self, target: Path, frame: pl.DataFrame, *, compression: str) -> Path:
        self._assert_target(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            existing = pl.read_parquet(target)
            if existing.equals(frame, null_equal=True):
                return target
            raise StorageError(
                f"immutable parquet target already exists with different data: {target}"
            )

        temporary = target.with_name(f".{uuid.uuid4().hex}.tmp")
        try:
            parquet_compression = cast(
                Literal["lz4", "uncompressed", "snappy", "gzip", "brotli", "zstd"],
                compression,
            )
            frame.write_parquet(temporary, compression=parquet_compression)
            with temporary.open("r+b") as stream:
                os.fsync(stream.fileno())
            validated = pl.read_parquet(temporary)
            if validated.shape != frame.shape or validated.columns != frame.columns:
                raise StorageError(f"parquet validation failed before promotion: {target}")
            os.replace(temporary, target)
        except (OSError, pl.exceptions.PolarsError) as exc:
            if temporary.exists():
                temporary.unlink()
            raise StorageError(f"atomic parquet write failed for {target}: {exc}") from exc
        return target

    def _assert_target(self, target: Path) -> None:
        resolved = target.resolve()
        if not resolved.is_relative_to(self.root):
            raise StorageError(f"write target escapes configured root: {resolved}")
