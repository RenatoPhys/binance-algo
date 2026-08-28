"""Structural contracts for fixed and trainable research strategies."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl

from binance_algo.research.contracts import FoldContext, StrategyScores, TrainingDataset


@runtime_checkable
class FittedStrategy(Protocol):
    """A strategy frozen using only the fold's training interval."""

    def score(
        self,
        features: pl.DataFrame,
        *,
        context: FoldContext,
    ) -> StrategyScores:
        """Score a projected feature view without labels or outcomes."""
        ...


@runtime_checkable
class Strategy(Protocol):
    """A stable strategy definition that can be fitted for each outer fold."""

    @property
    def strategy_id(self) -> str:
        """Return the stable strategy identifier."""
        ...

    @property
    def strategy_version(self) -> str:
        """Return the semantic strategy version."""
        ...

    def required_features(self) -> tuple[str, ...]:
        """Return the complete, explicit feature dependency list."""
        ...

    def target_column(self) -> str | None:
        """Return the separately supplied training target, if supervised."""
        ...

    def fit(
        self,
        train: TrainingDataset,
        *,
        context: FoldContext,
    ) -> FittedStrategy:
        """Fit exclusively on the training interval described by ``context``."""
        ...


__all__ = ["FittedStrategy", "Strategy"]
