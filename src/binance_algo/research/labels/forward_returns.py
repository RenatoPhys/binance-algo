"""Gross and benchmark-residual one-hour forward-return labels."""

from binance_algo.research.labels.base import LabelDefinition, LabelRegistry

GROSS_FORWARD_RETURN_1H = LabelDefinition(
    label_id="gross_forward_return_1h:v1",
    name="gross_forward_return_1h",
    version="v1",
    horizon_minutes=60,
    execution_lag_bars=1,
    semantics=(
        "Gross return from the open immediately after the decision to the open 60 minutes later."
    ),
    target_column="future_return_1h",
)

RESIDUAL_FORWARD_RETURN_1H = LabelDefinition(
    label_id="residual_forward_return_1h:v1",
    name="residual_forward_return_1h",
    version="v1",
    horizon_minutes=60,
    execution_lag_bars=1,
    semantics="Gross forward return minus its causal-beta benchmark forward return.",
    target_column="future_residual_return_1h",
)

PHASE3_LABEL_REGISTRY = LabelRegistry((GROSS_FORWARD_RETURN_1H, RESIDUAL_FORWARD_RETURN_1H))

__all__ = [
    "GROSS_FORWARD_RETURN_1H",
    "PHASE3_LABEL_REGISTRY",
    "RESIDUAL_FORWARD_RETURN_1H",
]
