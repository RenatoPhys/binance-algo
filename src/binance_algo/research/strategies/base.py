"""Structural contracts for fixed and trainable research strategies."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl

from binance_algo.research.contracts import FoldContext, StrategyScores, TrainingDataset
from binance_algo.research.panel import PanelData


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


@runtime_checkable
class PanelFittedStrategy(Protocol):
    """Optional fast path for scoring a shared immutable panel."""

    def score_panel(self, features: PanelData, *, context: FoldContext) -> StrategyScores: ...


@runtime_checkable
class PanelStrategy(Protocol):
    """Optional fast path for fitting from a shared immutable panel."""

    def fit_panel(
        self,
        train: PanelData,
        *,
        target: pl.DataFrame | None,
        context: FoldContext,
    ) -> FittedStrategy: ...


__all__ = ["FittedStrategy", "PanelFittedStrategy", "PanelStrategy", "Strategy"]
