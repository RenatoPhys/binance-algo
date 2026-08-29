"""Pure explicit-cost rates shared by the backtest and strategy portfolios."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from binance_algo.common.errors import ResearchError
from binance_algo.config import FeeScheduleConfig, ResearchConfig


class CostModelLike(Protocol):
    @property
    def component_id(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def parameters(self) -> Mapping[str, Any]: ...


class CostParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    spread_bps: Decimal = Field(ge=0, le=100)
    slippage_bps: Decimal = Field(ge=0, le=100)
    initial_capital_usdt: Decimal = Field(gt=0)
    fee_schedule: FeeScheduleConfig


@dataclass(frozen=True, slots=True)
class RegisteredExplicitCostModel:
    component_id: str
    version: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ExplicitCostRates:
    fee_rate: float
    half_spread_rate: float
    slippage_rate: float


def fee_schedule_covers(schedule: FeeScheduleConfig, execution_time_ms: int) -> bool:
    event_date = datetime.fromtimestamp(execution_time_ms / 1_000, tz=UTC).date()
    return event_date >= schedule.effective_from and (
        schedule.effective_to is None or event_date <= schedule.effective_to
    )


def configured_cost_model(config: ResearchConfig) -> RegisteredExplicitCostModel:
    return RegisteredExplicitCostModel(
        component_id="configured_taker",
        version="1",
        parameters={
            "spread_bps": config.spread_bps,
            "slippage_bps": config.slippage_bps,
            "initial_capital_usdt": config.initial_capital_usdt,
            "fee_schedule": config.fee_schedule.model_dump(mode="json"),
        },
    )


def explicit_cost_rates(
    cost_model: CostModelLike,
    execution_time_ms: int,
    *,
    multiplier: float = 1.0,
) -> ExplicitCostRates:
    """Resolve taker fee, half-spread, and slippage rates at one execution time."""

    if (cost_model.component_id, cost_model.version) not in {
        ("configured_taker", "1"),
        ("configured_taker", "v1"),
    }:
        raise ResearchError(
            f"unsupported explicit cost model: {cost_model.component_id}:{cost_model.version}"
        )
    if not math.isfinite(multiplier) or multiplier < 0:
        raise ResearchError("cost multiplier must be finite and non-negative")
    try:
        parameters = CostParameters.model_validate(cost_model.parameters)
    except ValidationError as exc:
        raise ResearchError(f"invalid explicit cost model parameters: {exc}") from exc
    if not fee_schedule_covers(parameters.fee_schedule, execution_time_ms):
        raise ResearchError(f"fee schedule does not cover execution time {execution_time_ms}")
    return ExplicitCostRates(
        fee_rate=float(parameters.fee_schedule.taker_fee_rate) * multiplier,
        half_spread_rate=float(parameters.spread_bps) / 20_000 * multiplier,
        slippage_rate=float(parameters.slippage_bps) / 10_000 * multiplier,
    )


__all__ = [
    "CostModelLike",
    "CostParameters",
    "ExplicitCostRates",
    "RegisteredExplicitCostModel",
    "configured_cost_model",
    "explicit_cost_rates",
    "fee_schedule_covers",
]
