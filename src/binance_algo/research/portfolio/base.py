"""Structural contract for converting scores into target portfolio weights."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl

from binance_algo.research.contracts import FoldContext


@runtime_checkable
class PortfolioPolicy(Protocol):
    """A versioned policy that maps strategy scores and market state to weights."""

    @property
    def policy_id(self) -> str:
        """Return the stable portfolio-policy identifier."""
        ...

    @property
    def policy_version(self) -> str:
        """Return the semantic portfolio-policy version."""
        ...

    def required_features(self) -> tuple[str, ...]:
        """Return market-state features required independently of the strategy."""
        ...

    def target_weights(
        self,
        scores: pl.DataFrame,
        market_state: pl.DataFrame,
        *,
        context: FoldContext,
    ) -> pl.DataFrame:
        """Produce long-form target weights for the supplied fold."""
        ...


__all__ = ["PortfolioPolicy"]
