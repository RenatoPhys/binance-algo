"""Strict immutable contracts for declared portfolios of strategy sleeves."""

from __future__ import annotations

import re
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from binance_algo.common.errors import ResearchError

PORTFOLIO_SCHEMA_VERSION = 1
MAXIMUM_COMPONENTS = 12
_PORTFOLIO_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")


class _ImmutableModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AccountingMode(StrEnum):
    SLEEVE = "sleeve"
    NETTED = "netted"


class WeightingMode(StrEnum):
    FIXED = "fixed"
    EQUAL_WEIGHT = "equal_weight"


class AlignmentPolicy(StrEnum):
    STRICT = "strict"
    INTERSECTION = "intersection"


class StrategyPortfolioComponent(_ImmutableModel):
    experiment_id: str = Field(min_length=1)
    run_id: str | None = Field(default=None, min_length=1)
    label: str = Field(min_length=1, max_length=200)
    capital_weight: Decimal | None = Field(default=None, ge=0, le=1)


class AlignmentSpec(_ImmutableModel):
    policy: AlignmentPolicy = AlignmentPolicy.STRICT
    require_same_dataset: bool = True
    require_same_label: bool = True
    require_same_execution_model: bool = True
    require_same_cost_model: bool = True
    require_same_split_plan: bool = True


class PortfolioAnalyticsSpec(_ImmutableModel):
    correlation_frequency: Literal["daily"] = "daily"
    rolling_windows_hours: tuple[Annotated[int, Field(ge=24, le=24 * 365)], ...] = (
        720,
        2160,
    )
    trade_epsilon: float = Field(default=1.0e-10, gt=0, le=1.0e-4)

    @model_validator(mode="after")
    def validate_windows(self) -> Self:
        if not self.rolling_windows_hours:
            raise ValueError("at least one rolling window is required")
        if len(self.rolling_windows_hours) > 6:
            raise ValueError("at most six rolling windows are supported")
        if tuple(sorted(set(self.rolling_windows_hours))) != self.rolling_windows_hours:
            raise ValueError("rolling windows must be unique and strictly increasing")
        return self


class StrategyPortfolioSpec(_ImmutableModel):
    portfolio_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=240)
    description: str = Field(min_length=1, max_length=4_000)
    accounting_mode: AccountingMode = AccountingMode.NETTED
    weighting: WeightingMode = WeightingMode.FIXED
    components: tuple[StrategyPortfolioComponent, ...] = Field(
        min_length=1,
        max_length=MAXIMUM_COMPONENTS,
    )
    alignment: AlignmentSpec = Field(default_factory=AlignmentSpec)
    analytics: PortfolioAnalyticsSpec = Field(default_factory=PortfolioAnalyticsSpec)

    @model_validator(mode="after")
    def validate_portfolio(self) -> Self:
        if not _PORTFOLIO_ID.fullmatch(self.portfolio_id):
            raise ValueError(
                "portfolio_id must use lowercase letters, digits, '.', '_' or '-' and start "
                "with a letter or digit"
            )
        experiment_ids = tuple(item.experiment_id for item in self.components)
        if len(set(experiment_ids)) != len(experiment_ids):
            raise ValueError("portfolio components must have unique experiment_id values")
        weights = tuple(item.capital_weight for item in self.components)
        if self.weighting is WeightingMode.FIXED:
            if any(weight is None for weight in weights):
                raise ValueError("fixed portfolios require capital_weight on every component")
            if sum((weight for weight in weights if weight is not None), Decimal(0)) != Decimal(1):
                raise ValueError("fixed portfolio capital weights must sum exactly to 1")
        elif any(weight is not None for weight in weights):
            raise ValueError(
                "equal_weight portfolios must not declare manual capital_weight values"
            )
        return self

    def resolved_weights(self) -> tuple[Decimal, ...]:
        if self.weighting is WeightingMode.EQUAL_WEIGHT:
            equal = Decimal(1) / Decimal(len(self.components))
            return tuple(equal for _ in self.components)
        return tuple(
            component.capital_weight
            for component in self.components
            if component.capital_weight is not None
        )


class PortfolioFile(_ImmutableModel):
    schema_version: Literal[1]
    portfolios: tuple[StrategyPortfolioSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> Self:
        identifiers = tuple(item.portfolio_id for item in self.portfolios)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("portfolio_id values must be unique")
        return self


def load_portfolio_file(path: Path) -> PortfolioFile:
    """Load a strict schema-v1 portfolio file with the safe YAML parser."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ResearchError("portfolio file root must be a mapping")
        return PortfolioFile.model_validate(payload)
    except ResearchError:
        raise
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise ResearchError(f"cannot load strategy portfolio file {path}: {exc}") from exc


__all__ = [
    "MAXIMUM_COMPONENTS",
    "PORTFOLIO_SCHEMA_VERSION",
    "AccountingMode",
    "AlignmentPolicy",
    "AlignmentSpec",
    "PortfolioAnalyticsSpec",
    "PortfolioFile",
    "StrategyPortfolioComponent",
    "StrategyPortfolioSpec",
    "WeightingMode",
    "load_portfolio_file",
]
