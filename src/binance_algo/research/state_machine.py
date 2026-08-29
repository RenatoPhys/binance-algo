"""Reusable causal state machine for sparse, fold-local research signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from binance_algo.common.errors import ResearchError

FloatMatrix = np.ndarray[Any, np.dtype[np.float64]]
BoolMatrix = np.ndarray[Any, np.dtype[np.bool_]]


@dataclass(frozen=True, slots=True)
class SparseStateResult:
    values: FloatMatrix
    entries: BoolMatrix
    exits: BoolMatrix
    forced_time_exits: BoolMatrix


def causal_sparse_state(
    entry_values: FloatMatrix,
    *,
    hold_hours: int,
    explicit_exit: BoolMatrix | None = None,
) -> SparseStateResult:
    """Enter only while flat, ignore retriggers, and reset state at matrix boundaries."""

    entries_source = np.asarray(entry_values, dtype=np.float64)
    if entries_source.ndim != 2 or not np.all(np.isfinite(entries_source)):
        raise ResearchError("sparse-state entry values must be a finite two-dimensional matrix")
    if hold_hours < 1:
        raise ResearchError("sparse-state holding period must be positive")
    exits_source = (
        np.zeros(entries_source.shape, dtype=np.bool_)
        if explicit_exit is None
        else np.asarray(explicit_exit, dtype=np.bool_)
    )
    if exits_source.shape != entries_source.shape:
        raise ResearchError("sparse-state explicit exits must match entry values")
    values = np.zeros_like(entries_source)
    entries = np.zeros(entries_source.shape, dtype=np.bool_)
    exits = np.zeros(entries_source.shape, dtype=np.bool_)
    forced = np.zeros(entries_source.shape, dtype=np.bool_)
    state = np.zeros(entries_source.shape[1], dtype=np.float64)
    remaining = np.zeros(entries_source.shape[1], dtype=np.int64)
    for row in range(len(entries_source)):
        exited_now = np.zeros(entries_source.shape[1], dtype=np.bool_)
        for column in range(entries_source.shape[1]):
            if state[column] != 0 and (exits_source[row, column] or remaining[column] <= 0):
                forced[row, column] = remaining[column] <= 0
                exits[row, column] = True
                exited_now[column] = True
                state[column] = 0.0
                remaining[column] = 0
            if state[column] == 0 and not exited_now[column] and entries_source[row, column] != 0:
                state[column] = entries_source[row, column]
                remaining[column] = hold_hours
                entries[row, column] = True
            values[row, column] = state[column]
            if state[column] != 0:
                remaining[column] -= 1
    values.setflags(write=False)
    entries.setflags(write=False)
    exits.setflags(write=False)
    forced.setflags(write=False)
    return SparseStateResult(
        values=values,
        entries=entries,
        exits=exits,
        forced_time_exits=forced,
    )


__all__ = ["SparseStateResult", "causal_sparse_state"]
