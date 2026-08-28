"""Immutable, reusable array panels and a bounded per-process dataset cache."""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Any

import numpy as np
import polars as pl

from binance_algo.common.errors import ResearchError
from binance_algo.research.contracts import FEATURE_KEY_COLUMNS

UNIVERSE_METADATA_COLUMNS = (
    "is_available",
    "exclusion_reason",
    "listing_time_ms",
    "delisting_time_ms",
    "lookback_complete",
    "quality_passed",
    "liquidity_eligible",
)


def _unique_names(values: Iterable[str], *, role: str) -> tuple[str, ...]:
    names = tuple(values)
    if any(not name for name in names):
        raise ResearchError(f"{role} names must be non-empty")
    if len(names) != len(set(names)):
        raise ResearchError(f"{role} names must be unique")
    return names


def _frozen_array(
    value: np.ndarray[Any, Any],
    *,
    shape: tuple[int, ...],
    role: str,
) -> np.ndarray[Any, Any]:
    array = np.array(value, copy=True)
    if array.shape != shape:
        raise ResearchError(f"{role} has shape {array.shape}; expected {shape}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class PanelData:
    """Dense point-in-time panel with separate immutable analytical namespaces."""

    times: np.ndarray[Any, np.dtype[np.int64]]
    symbols: tuple[str, ...]
    features: Mapping[str, np.ndarray[Any, Any]]
    outcomes: Mapping[str, np.ndarray[Any, Any]]
    metadata: Mapping[str, np.ndarray[Any, Any]]
    availability: np.ndarray[Any, np.dtype[np.bool_]]

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=np.int64)
        symbols = tuple(self.symbols)
        if not len(times) or not symbols:
            raise ResearchError("PanelData requires at least one time and one symbol")
        if len(np.unique(times)) != len(times) or np.any(np.diff(times) <= 0):
            raise ResearchError("PanelData times must be unique and strictly increasing")
        if len(set(symbols)) != len(symbols) or tuple(sorted(symbols)) != symbols:
            raise ResearchError("PanelData symbols must be unique and sorted")
        shape = (len(times), len(symbols))
        namespaces = {
            "feature": dict(self.features),
            "outcome": dict(self.outcomes),
            "metadata": dict(self.metadata),
        }
        names = [name for values in namespaces.values() for name in values]
        if len(names) != len(set(names)):
            raise ResearchError("PanelData names cannot overlap across namespaces")
        frozen_namespaces: dict[str, Mapping[str, np.ndarray[Any, Any]]] = {}
        for namespace, values in namespaces.items():
            frozen_namespaces[namespace] = MappingProxyType(
                {
                    name: _frozen_array(
                        values[name],
                        shape=shape,
                        role=f"PanelData {namespace} {name}",
                    )
                    for name in sorted(values)
                }
            )
        availability = _frozen_array(
            np.asarray(self.availability, dtype=np.bool_),
            shape=shape,
            role="PanelData availability",
        )
        for namespace in ("feature", "outcome"):
            for name, analytical_array in frozen_namespaces[namespace].items():
                try:
                    valid = np.isfinite(analytical_array[availability])
                except TypeError as exc:
                    raise ResearchError(f"PanelData {namespace} {name} must be numeric") from exc
                if np.any(~valid):
                    raise ResearchError(
                        f"PanelData {namespace} {name} is non-finite when available"
                    )
        times = _frozen_array(times, shape=(len(times),), role="PanelData times")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "features", frozen_namespaces["feature"])
        object.__setattr__(self, "outcomes", frozen_namespaces["outcome"])
        object.__setattr__(self, "metadata", frozen_namespaces["metadata"])
        object.__setattr__(self, "availability", availability)

    @classmethod
    def from_frame(
        cls,
        frame: pl.DataFrame,
        *,
        feature_columns: Iterable[str] = (),
        outcome_columns: Iterable[str] = (),
        metadata_columns: Iterable[str] = (),
    ) -> PanelData:
        """Build one panel using vectorized key-to-index mappings, including partial panels."""

        feature_names = _unique_names(feature_columns, role="feature")
        outcome_names = _unique_names(outcome_columns, role="outcome")
        requested_metadata = _unique_names(metadata_columns, role="metadata")
        metadata_names = tuple(dict.fromkeys((*requested_metadata, *UNIVERSE_METADATA_COLUMNS)))
        all_names = (*feature_names, *outcome_names, *metadata_names)
        if len(all_names) != len(set(all_names)):
            raise ResearchError("PanelData columns cannot overlap across namespaces")
        missing_keys = sorted(set(FEATURE_KEY_COLUMNS).difference(frame.columns))
        if missing_keys:
            raise ResearchError(f"PanelData source is missing key columns: {missing_keys}")
        if any(frame[column].null_count() for column in FEATURE_KEY_COLUMNS):
            raise ResearchError("PanelData source contains null research keys")
        if frame.select(FEATURE_KEY_COLUMNS).is_duplicated().any():
            raise ResearchError("PanelData source contains duplicate research keys")
        required = (*feature_names, *outcome_names, *requested_metadata)
        missing = sorted(set(required).difference(frame.columns))
        if missing:
            raise ResearchError(f"PanelData source is missing required columns: {missing}")
        try:
            time_values = np.asarray(
                frame["decision_time_ms"].cast(pl.Int64, strict=True).to_numpy(),
                dtype=np.int64,
            )
            symbol_values = np.asarray(
                frame["symbol"].cast(pl.String, strict=True).to_numpy(),
                dtype=np.str_,
            )
        except (TypeError, ValueError, pl.exceptions.PolarsError) as exc:
            raise ResearchError("PanelData source has invalid research keys") from exc
        times = np.unique(time_values)
        symbols = tuple(sorted(set(symbol_values.tolist())))
        if not len(times) or not symbols:
            raise ResearchError("PanelData source cannot be empty")
        shape = (len(times), len(symbols))
        time_indices = np.searchsorted(times, time_values)
        symbol_axis = np.asarray(symbols, dtype=np.str_)
        symbol_indices = np.searchsorted(symbol_axis, symbol_values)
        row_present = np.zeros(shape, dtype=np.bool_)
        row_present[time_indices, symbol_indices] = True

        def numeric_matrix(column: str) -> np.ndarray[Any, np.dtype[np.float64]]:
            values = np.asarray(
                frame[column].cast(pl.Float64, strict=True).to_numpy(),
                dtype=np.float64,
            )
            matrix = np.full(shape, np.nan, dtype=np.float64)
            matrix[time_indices, symbol_indices] = values
            return matrix

        def typed_matrix(column: str, *, default: object) -> np.ndarray[Any, Any]:
            dtype = object if isinstance(default, str) else np.asarray(default).dtype
            if column not in frame.columns:
                return np.full(shape, default, dtype=dtype)
            values = frame[column].to_numpy()
            matrix = np.full(shape, default, dtype=dtype)
            matrix[time_indices, symbol_indices] = values
            return matrix

        try:
            features = {name: numeric_matrix(name) for name in feature_names}
            outcomes = {name: numeric_matrix(name) for name in outcome_names}
        except (TypeError, ValueError, pl.exceptions.PolarsError) as exc:
            raise ResearchError("PanelData analytical columns must be numeric") from exc
        finite_features = np.ones(shape, dtype=np.bool_)
        for values in features.values():
            finite_features &= np.isfinite(values)
        is_available = typed_matrix("is_available", default=False).astype(np.bool_)
        if "is_available" not in frame.columns:
            is_available = row_present.copy()
        lookback_complete = typed_matrix("lookback_complete", default=False).astype(np.bool_)
        if "lookback_complete" not in frame.columns:
            lookback_complete = row_present & finite_features
        quality_passed = typed_matrix("quality_passed", default=False).astype(np.bool_)
        if "quality_passed" not in frame.columns:
            quality_passed = row_present.copy()
        liquidity_eligible = typed_matrix("liquidity_eligible", default=False).astype(np.bool_)
        if "liquidity_eligible" not in frame.columns:
            liquidity_eligible = row_present.copy()
        availability = (
            row_present & is_available & lookback_complete & quality_passed & liquidity_eligible
        )
        metadata: dict[str, np.ndarray[Any, Any]] = {}
        for name in requested_metadata:
            if name in UNIVERSE_METADATA_COLUMNS:
                continue
            try:
                metadata[name] = numeric_matrix(name)
            except (TypeError, ValueError, pl.exceptions.PolarsError):
                metadata[name] = typed_matrix(name, default="")
        exclusion_reason = typed_matrix("exclusion_reason", default="")
        exclusion_reason[~row_present] = "missing_panel_row"
        metadata.update(
            {
                "is_available": is_available,
                "exclusion_reason": exclusion_reason,
                "listing_time_ms": typed_matrix("listing_time_ms", default=-1).astype(np.int64),
                "delisting_time_ms": typed_matrix("delisting_time_ms", default=-1).astype(np.int64),
                "lookback_complete": lookback_complete,
                "quality_passed": quality_passed,
                "liquidity_eligible": liquidity_eligible,
            }
        )
        required_value_mask = availability
        if "lookback_complete" not in frame.columns:
            required_value_mask = row_present & is_available & quality_passed & liquidity_eligible
        for name, values in (*features.items(), *outcomes.items()):
            if np.any(~np.isfinite(values[required_value_mask])):
                raise ResearchError(f"PanelData required field {name} is non-finite when available")
        return cls(
            times=times,
            symbols=symbols,
            features=features,
            outcomes=outcomes,
            metadata=metadata,
            availability=availability,
        )

    @property
    def symbol_to_index(self) -> Mapping[str, int]:
        return MappingProxyType({symbol: index for index, symbol in enumerate(self.symbols)})

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.times), len(self.symbols))

    @property
    def estimated_nbytes(self) -> int:
        arrays = (
            self.times,
            self.availability,
            *self.features.values(),
            *self.outcomes.values(),
            *self.metadata.values(),
        )
        return sum(array.nbytes for array in arrays)

    def require_complete(self, *, role: str) -> None:
        if not self.availability.all():
            missing = int(self.availability.size - np.count_nonzero(self.availability))
            raise ResearchError(f"{role} panel has {missing} unavailable symbol-time cells")

    def require_complete_range(self, start_ms: int, end_ms: int, *, role: str) -> None:
        availability = self.availability[self.time_slice(start_ms, end_ms)]
        if not availability.all():
            missing = int(availability.size - np.count_nonzero(availability))
            raise ResearchError(f"{role} panel has {missing} unavailable symbol-time cells")

    def time_slice(self, start_ms: int, end_ms: int) -> slice:
        if start_ms > end_ms:
            raise ResearchError("PanelData time range is reversed")
        left = int(np.searchsorted(self.times, start_ms, side="left"))
        right = int(np.searchsorted(self.times, end_ms, side="right"))
        if left == right:
            raise ResearchError("PanelData time range is empty")
        return slice(left, right)

    def matrix(
        self,
        name: str,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> np.ndarray[Any, Any]:
        namespaces = (self.features, self.outcomes, self.metadata)
        matches = [values[name] for values in namespaces if name in values]
        if len(matches) != 1:
            raise ResearchError(f"unknown or ambiguous PanelData field: {name}")
        values = matches[0]
        if start_ms is None and end_ms is None:
            return values
        if start_ms is None or end_ms is None:
            raise ResearchError("PanelData matrix slicing requires both time bounds")
        view = values[self.time_slice(start_ms, end_ms)]
        view.setflags(write=False)
        return view


@dataclass(frozen=True, slots=True)
class LoadedPanelDataset:
    frame: pl.DataFrame
    panel: PanelData
    projected_columns: tuple[str, ...]
    load_seconds: float


@dataclass(frozen=True, slots=True)
class DatasetCacheInfo:
    entries: int
    hits: int
    misses: int


class WorkerDatasetCache:
    """Small process-local LRU cache used by sequential trials assigned to a worker."""

    def __init__(self, *, max_entries: int = 2) -> None:
        if max_entries < 1:
            raise ResearchError("dataset cache must keep at least one entry")
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple[object, ...], LoadedPanelDataset] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._lock = threading.RLock()

    def load(
        self,
        path: Path,
        *,
        feature_columns: Iterable[str],
        outcome_columns: Iterable[str],
        metadata_columns: Iterable[str],
    ) -> LoadedPanelDataset:
        resolved = path.resolve()
        feature_names = _unique_names(feature_columns, role="feature")
        outcome_names = _unique_names(outcome_columns, role="outcome")
        metadata_names = _unique_names(metadata_columns, role="metadata")
        lazy = pl.scan_parquet(resolved)
        available_columns = set(lazy.collect_schema().names())
        optional_universe = tuple(
            name for name in UNIVERSE_METADATA_COLUMNS if name in available_columns
        )
        projected = tuple(
            dict.fromkeys(
                (
                    *FEATURE_KEY_COLUMNS,
                    *feature_names,
                    *outcome_names,
                    *metadata_names,
                    *optional_universe,
                )
            )
        )
        missing = sorted(set(projected).difference(available_columns))
        if missing:
            raise ResearchError(f"research dataset is missing projected columns: {missing}")
        stat = resolved.stat()
        key = (str(resolved), stat.st_size, stat.st_mtime_ns, projected)
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                self._hits += 1
                return cached
            started = perf_counter()
            frame = lazy.select(projected).collect()
            panel = PanelData.from_frame(
                frame,
                feature_columns=feature_names,
                outcome_columns=outcome_names,
                metadata_columns=(*metadata_names, *optional_universe),
            )
            loaded = LoadedPanelDataset(
                frame=frame,
                panel=panel,
                projected_columns=projected,
                load_seconds=perf_counter() - started,
            )
            self._entries[key] = loaded
            self._misses += 1
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
            return loaded

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._hits = 0
            self._misses = 0

    def info(self) -> DatasetCacheInfo:
        with self._lock:
            return DatasetCacheInfo(
                entries=len(self._entries),
                hits=self._hits,
                misses=self._misses,
            )


WORKER_DATASET_CACHE = WorkerDatasetCache()


def matrix_to_long_frame(
    *,
    times: np.ndarray[Any, np.dtype[np.int64]],
    symbols: tuple[str, ...],
    value_name: str,
    values: np.ndarray[Any, Any],
) -> pl.DataFrame:
    shape = (len(times), len(symbols))
    if values.shape != shape:
        raise ResearchError(f"long-form matrix has shape {values.shape}; expected {shape}")
    return pl.DataFrame(
        {
            "decision_time_ms": np.repeat(times, len(symbols)),
            "symbol": np.tile(np.asarray(symbols), len(times)),
            value_name: values.reshape(-1),
        }
    )


def finite_float(value: float, *, role: str) -> float:
    """Validate benchmark/diagnostic values without admitting silent NaNs."""

    if not math.isfinite(value):
        raise ResearchError(f"{role} must be finite")
    return value


__all__ = [
    "UNIVERSE_METADATA_COLUMNS",
    "WORKER_DATASET_CACHE",
    "DatasetCacheInfo",
    "LoadedPanelDataset",
    "PanelData",
    "WorkerDatasetCache",
    "finite_float",
    "matrix_to_long_frame",
]
