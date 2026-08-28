"""Versioned research labels and their registry."""

from binance_algo.research.labels.base import LabelDefinition, LabelRegistry
from binance_algo.research.labels.forward_returns import (
    GROSS_FORWARD_RETURN_1H,
    PHASE3_LABEL_REGISTRY,
    RESIDUAL_FORWARD_RETURN_1H,
)

__all__ = [
    "GROSS_FORWARD_RETURN_1H",
    "PHASE3_LABEL_REGISTRY",
    "RESIDUAL_FORWARD_RETURN_1H",
    "LabelDefinition",
    "LabelRegistry",
]
