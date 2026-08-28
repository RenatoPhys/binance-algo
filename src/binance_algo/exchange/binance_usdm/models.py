"""Parsing from Binance payloads into stable canonical metadata."""

from __future__ import annotations

import time
from typing import Any

import orjson
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from binance_algo.common.errors import DataContractError

SCHEMA_VERSION = 1


class ExchangeSymbolPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    symbol: str
    pair: str
    contract_type: str = Field(alias="contractType")
    status: str
    base_asset: str = Field(alias="baseAsset")
    quote_asset: str = Field(alias="quoteAsset")
    margin_asset: str = Field(alias="marginAsset")
    onboard_date_ms: int = Field(alias="onboardDate")
    delivery_date_ms: int = Field(alias="deliveryDate")
    price_precision: int = Field(alias="pricePrecision")
    quantity_precision: int = Field(alias="quantityPrecision")
    filters: list[dict[str, Any]]


class ExchangeInfoPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    server_time: int = Field(alias="serverTime")
    symbols: list[ExchangeSymbolPayload]


class FundingRatePayload(BaseModel):
    """Public funding event as returned by USD-M Futures."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    symbol: str
    funding_rate: str = Field(alias="fundingRate")
    funding_time_ms: int = Field(alias="fundingTime")
    mark_price: str | None = Field(default=None, alias="markPrice")
    rate_type: str = Field(default="Regular", alias="rateType")


class InstrumentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    exchange: str = "binance"
    market: str = "usdm_futures"
    symbol: str
    pair: str
    contract_type: str
    status: str
    base_asset: str
    quote_asset: str
    margin_asset: str
    onboard_date_ms: int
    delivery_date_ms: int
    price_precision: int
    quantity_precision: int
    tick_size: str
    step_size: str
    min_qty: str
    max_qty: str
    min_notional: str
    raw_filters_json: str
    valid_from_ms: int
    ingested_at_ns: int
    schema_version: int = SCHEMA_VERSION


def _filter_by_type(filters: list[dict[str, Any]], filter_type: str) -> dict[str, Any]:
    for item in filters:
        if item.get("filterType") == filter_type:
            return item
    raise DataContractError(f"required Binance filter is missing: {filter_type}")


def parse_instruments(
    payload: dict[str, Any],
    *,
    valid_from_ms: int | None = None,
    ingested_at_ns: int | None = None,
) -> list[InstrumentMetadata]:
    """Validate exchangeInfo and preserve exact numeric filter strings."""

    try:
        exchange_info = ExchangeInfoPayload.model_validate(payload)
    except ValidationError as exc:
        raise DataContractError(f"incompatible exchangeInfo payload: {exc}") from exc

    ingestion_time = ingested_at_ns if ingested_at_ns is not None else time.time_ns()
    snapshot_time = valid_from_ms if valid_from_ms is not None else exchange_info.server_time
    instruments: list[InstrumentMetadata] = []
    for symbol in exchange_info.symbols:
        price_filter = _filter_by_type(symbol.filters, "PRICE_FILTER")
        lot_size = _filter_by_type(symbol.filters, "LOT_SIZE")
        try:
            try:
                min_notional_filter = _filter_by_type(symbol.filters, "MIN_NOTIONAL")
                min_notional = str(min_notional_filter["notional"])
            except DataContractError:
                min_notional_filter = _filter_by_type(symbol.filters, "NOTIONAL")
                min_notional = str(min_notional_filter["minNotional"])
            instruments.append(
                InstrumentMetadata(
                    symbol=symbol.symbol,
                    pair=symbol.pair,
                    contract_type=symbol.contract_type,
                    status=symbol.status,
                    base_asset=symbol.base_asset,
                    quote_asset=symbol.quote_asset,
                    margin_asset=symbol.margin_asset,
                    onboard_date_ms=symbol.onboard_date_ms,
                    delivery_date_ms=symbol.delivery_date_ms,
                    price_precision=symbol.price_precision,
                    quantity_precision=symbol.quantity_precision,
                    tick_size=str(price_filter["tickSize"]),
                    step_size=str(lot_size["stepSize"]),
                    min_qty=str(lot_size["minQty"]),
                    max_qty=str(lot_size["maxQty"]),
                    min_notional=min_notional,
                    raw_filters_json=orjson.dumps(
                        symbol.filters, option=orjson.OPT_SORT_KEYS
                    ).decode("utf-8"),
                    valid_from_ms=snapshot_time,
                    ingested_at_ns=ingestion_time,
                )
            )
        except KeyError as exc:
            raise DataContractError(
                f"filter field {exc} missing for Binance symbol {symbol.symbol}"
            ) from exc
    return instruments
