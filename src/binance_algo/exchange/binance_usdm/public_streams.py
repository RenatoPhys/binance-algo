"""Canonical contracts and strict parsers for Binance USD-M public streams."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any, Literal, cast

import orjson
import polars as pl

from binance_algo.common.errors import DataContractError

StreamEventType = Literal["book_ticker", "aggregate_trades", "mark_price", "kline_1m"]
StreamRoute = Literal["public", "market"]
STREAM_SCHEMA_VERSION = 1
STREAM_EVENT_TYPES: tuple[StreamEventType, ...] = (
    "book_ticker",
    "aggregate_trades",
    "mark_price",
    "kline_1m",
)

COMMON_SCHEMA: dict[str, Any] = {
    "event_id": pl.String,
    "symbol": pl.String,
}
TRAILING_SCHEMA: dict[str, Any] = {
    "received_time_ns": pl.Int64,
    "processed_time_ns": pl.Int64,
    "connection_id": pl.String,
    "ingestion_run_id": pl.String,
    "source": pl.String,
    "schema_version": pl.Int32,
    "payload_json": pl.String,
}
STREAM_SCHEMAS: dict[StreamEventType, dict[str, Any]] = {
    "book_ticker": {
        **COMMON_SCHEMA,
        "update_id": pl.Int64,
        "bid_price_str": pl.String,
        "bid_qty_str": pl.String,
        "ask_price_str": pl.String,
        "ask_qty_str": pl.String,
        "bid_price": pl.Float64,
        "bid_qty": pl.Float64,
        "ask_price": pl.Float64,
        "ask_qty": pl.Float64,
        "event_time_ms": pl.Int64,
        "transaction_time_ms": pl.Int64,
        **TRAILING_SCHEMA,
    },
    "aggregate_trades": {
        **COMMON_SCHEMA,
        "aggregate_trade_id": pl.Int64,
        "price_str": pl.String,
        "quantity_str": pl.String,
        "price": pl.Float64,
        "quantity": pl.Float64,
        "first_trade_id": pl.Int64,
        "last_trade_id": pl.Int64,
        "trade_time_ms": pl.Int64,
        "buyer_is_maker": pl.Boolean,
        "event_time_ms": pl.Int64,
        **TRAILING_SCHEMA,
    },
    "mark_price": {
        **COMMON_SCHEMA,
        "mark_price_str": pl.String,
        "index_price_str": pl.String,
        "estimated_settle_price_str": pl.String,
        "funding_rate_str": pl.String,
        "mark_price": pl.Float64,
        "index_price": pl.Float64,
        "estimated_settle_price": pl.Float64,
        "funding_rate": pl.Float64,
        "next_funding_time_ms": pl.Int64,
        "event_time_ms": pl.Int64,
        **TRAILING_SCHEMA,
    },
    "kline_1m": {
        **COMMON_SCHEMA,
        "interval": pl.String,
        "open_time_ms": pl.Int64,
        "close_time_ms": pl.Int64,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "base_volume": pl.Float64,
        "quote_volume": pl.Float64,
        "trade_count": pl.Int64,
        "taker_buy_base_volume": pl.Float64,
        "taker_buy_quote_volume": pl.Float64,
        "is_closed": pl.Boolean,
        "event_time_ms": pl.Int64,
        "ingested_at_ns": pl.Int64,
        **TRAILING_SCHEMA,
    },
}


@dataclass(frozen=True, slots=True)
class StreamSubscription:
    route: StreamRoute
    names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CanonicalStreamEvent:
    event_type: StreamEventType
    stream_name: str
    symbol: str
    event_time_ms: int
    received_time_ns: int
    processed_time_ns: int
    connection_id: str
    ingestion_run_id: str
    event_id: str
    row: dict[str, object]


def build_stream_subscriptions(
    symbols: tuple[str, ...],
    *,
    book_ticker: bool,
    aggregate_trades: bool,
    mark_price: bool,
    kline_1m: bool,
) -> tuple[StreamSubscription, ...]:
    normalized = tuple(symbol.strip().lower() for symbol in symbols)
    if not normalized or any(not symbol for symbol in normalized):
        raise DataContractError("at least one non-empty stream symbol is required")
    if len(normalized) != len(set(normalized)):
        raise DataContractError("stream symbols must not contain duplicates")
    public = tuple(f"{symbol}@bookTicker" for symbol in normalized) if book_ticker else ()
    market: list[str] = []
    for symbol in normalized:
        if aggregate_trades:
            market.append(f"{symbol}@aggTrade")
        if mark_price:
            market.append(f"{symbol}@markPrice@1s")
        if kline_1m:
            market.append(f"{symbol}@kline_1m")
    subscriptions: list[StreamSubscription] = []
    if public:
        subscriptions.append(StreamSubscription(route="public", names=public))
    if market:
        subscriptions.append(StreamSubscription(route="market", names=tuple(market)))
    if not subscriptions:
        raise DataContractError("at least one public stream subscription is required")
    return tuple(subscriptions)


def combined_stream_url(base_url: str, subscription: StreamSubscription) -> str:
    root = base_url.rstrip("/")
    for suffix in ("/public", "/market", "/private"):
        if root.endswith(suffix):
            root = root.removesuffix(suffix)
            break
    streams = "/".join(subscription.names)
    return f"{root}/{subscription.route}/stream?streams={streams}"


def subscription_command(
    method: Literal["SUBSCRIBE", "UNSUBSCRIBE"], names: tuple[str, ...], request_id: int
) -> bytes:
    if request_id < 0:
        raise DataContractError("WebSocket request id must be an unsigned integer")
    if not names:
        raise DataContractError(f"{method} requires at least one stream")
    return orjson.dumps({"method": method, "params": names, "id": request_id})


def stream_frame(event_type: StreamEventType, rows: list[dict[str, object]]) -> pl.DataFrame:
    try:
        return pl.DataFrame(rows, schema=STREAM_SCHEMAS[event_type], strict=True)
    except (TypeError, ValueError, pl.exceptions.PolarsError) as exc:
        raise DataContractError(f"cannot materialize {event_type} micro-batch: {exc}") from exc


def stream_schema_contract_json(event_type: StreamEventType) -> str:
    contract = {
        "dataset": event_type,
        "key": ["event_id"],
        "schema_version": STREAM_SCHEMA_VERSION,
        "columns": {name: str(dtype) for name, dtype in STREAM_SCHEMAS[event_type].items()},
    }
    return orjson.dumps(contract, option=orjson.OPT_SORT_KEYS).decode("utf-8")


def parse_combined_message(
    message: bytes | str,
    *,
    received_time_ns: int,
    connection_id: str,
    ingestion_run_id: str,
    processed_time_ns: int | None = None,
) -> CanonicalStreamEvent | None:
    try:
        document = orjson.loads(message)
    except orjson.JSONDecodeError as exc:
        raise DataContractError(f"invalid WebSocket JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise DataContractError("combined WebSocket message must be an object")
    if "result" in document and "id" in document:
        return None
    stream_name = document.get("stream")
    payload = document.get("data")
    if not isinstance(stream_name, str) or not isinstance(payload, dict):
        raise DataContractError("combined WebSocket message requires stream and data")
    if payload.get("e") == "serverShutdown":
        raise DataContractError("Binance announced a WebSocket server shutdown")
    stream_symbol = stream_name.partition("@")[0].upper()
    if not stream_symbol or payload.get("s") != stream_symbol:
        raise DataContractError("stream name and payload symbol do not match")
    canonical_payload = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS).decode("utf-8")
    event_id = hashlib.sha256(f"{stream_name}\x1f{canonical_payload}".encode()).hexdigest()
    processed_ns = processed_time_ns if processed_time_ns is not None else time.time_ns()
    common = {
        "event_id": event_id,
        "received_time_ns": received_time_ns,
        "processed_time_ns": processed_ns,
        "connection_id": connection_id,
        "ingestion_run_id": ingestion_run_id,
        "source": "binance_market_stream",
        "schema_version": STREAM_SCHEMA_VERSION,
        "payload_json": canonical_payload,
    }
    if stream_name.endswith("@bookTicker"):
        return _book_ticker(stream_name, payload, common)
    if stream_name.endswith("@aggTrade"):
        return _aggregate_trade(stream_name, payload, common)
    if stream_name.endswith(("@markPrice", "@markPrice@1s")):
        return _mark_price(stream_name, payload, common)
    if stream_name.endswith("@kline_1m"):
        return _kline(stream_name, payload, common)
    raise DataContractError(f"unsupported Binance stream: {stream_name}")


def _required(payload: dict[str, Any], key: str, expected: type[object]) -> object:
    value = payload.get(key)
    if not isinstance(value, expected):
        raise DataContractError(f"stream payload field {key!r} has invalid type")
    return value


def _integer(payload: dict[str, Any], key: str) -> int:
    value = _required(payload, key, int)
    if isinstance(value, bool):
        raise DataContractError(f"stream payload field {key!r} must be an integer")
    return cast(int, value)


def _string(payload: dict[str, Any], key: str) -> str:
    return cast(str, _required(payload, key, str))


def _decimal(payload: dict[str, Any], key: str) -> tuple[str, float]:
    raw = _string(payload, key)
    try:
        value = float(raw)
    except ValueError as exc:
        raise DataContractError(f"stream payload field {key!r} is not numeric") from exc
    if not math.isfinite(value):
        raise DataContractError(f"stream payload field {key!r} must be finite")
    return raw, value


def _event(
    event_type: StreamEventType,
    stream_name: str,
    row: dict[str, object],
) -> CanonicalStreamEvent:
    return CanonicalStreamEvent(
        event_type=event_type,
        stream_name=stream_name,
        symbol=cast(str, row["symbol"]),
        event_time_ms=cast(int, row["event_time_ms"]),
        received_time_ns=cast(int, row["received_time_ns"]),
        processed_time_ns=cast(int, row["processed_time_ns"]),
        connection_id=cast(str, row["connection_id"]),
        ingestion_run_id=cast(str, row["ingestion_run_id"]),
        event_id=cast(str, row["event_id"]),
        row=row,
    )


def _book_ticker(
    stream_name: str, payload: dict[str, Any], common: dict[str, object]
) -> CanonicalStreamEvent:
    if payload.get("e") != "bookTicker":
        raise DataContractError("bookTicker stream has an unexpected event type")
    event_time = _integer(payload, "E")
    bid_price_str, bid_price = _decimal(payload, "b")
    bid_qty_str, bid_qty = _decimal(payload, "B")
    ask_price_str, ask_price = _decimal(payload, "a")
    ask_qty_str, ask_qty = _decimal(payload, "A")
    row = {
        "event_id": common["event_id"],
        "symbol": _string(payload, "s"),
        "update_id": _integer(payload, "u"),
        "bid_price_str": bid_price_str,
        "bid_qty_str": bid_qty_str,
        "ask_price_str": ask_price_str,
        "ask_qty_str": ask_qty_str,
        "bid_price": bid_price,
        "bid_qty": bid_qty,
        "ask_price": ask_price,
        "ask_qty": ask_qty,
        "event_time_ms": event_time,
        "transaction_time_ms": _integer(payload, "T"),
        **{key: value for key, value in common.items() if key != "event_id"},
    }
    return _event("book_ticker", stream_name, row)


def _aggregate_trade(
    stream_name: str, payload: dict[str, Any], common: dict[str, object]
) -> CanonicalStreamEvent:
    if payload.get("e") != "aggTrade":
        raise DataContractError("aggTrade stream has an unexpected event type")
    event_time = _integer(payload, "E")
    buyer_is_maker = _required(payload, "m", bool)
    price_str, price = _decimal(payload, "p")
    quantity_str, quantity = _decimal(payload, "q")
    row = {
        "event_id": common["event_id"],
        "symbol": _string(payload, "s"),
        "aggregate_trade_id": _integer(payload, "a"),
        "price_str": price_str,
        "quantity_str": quantity_str,
        "price": price,
        "quantity": quantity,
        "first_trade_id": _integer(payload, "f"),
        "last_trade_id": _integer(payload, "l"),
        "trade_time_ms": _integer(payload, "T"),
        "buyer_is_maker": buyer_is_maker,
        "event_time_ms": event_time,
        **{key: value for key, value in common.items() if key != "event_id"},
    }
    return _event("aggregate_trades", stream_name, row)


def _mark_price(
    stream_name: str, payload: dict[str, Any], common: dict[str, object]
) -> CanonicalStreamEvent:
    if payload.get("e") != "markPriceUpdate":
        raise DataContractError("markPrice stream has an unexpected event type")
    event_time = _integer(payload, "E")
    mark_price_str, mark_price = _decimal(payload, "p")
    index_price_str, index_price = _decimal(payload, "i")
    settle_price_str, settle_price = _decimal(payload, "P")
    funding_rate_str, funding_rate = _decimal(payload, "r")
    row = {
        "event_id": common["event_id"],
        "symbol": _string(payload, "s"),
        "mark_price_str": mark_price_str,
        "index_price_str": index_price_str,
        "estimated_settle_price_str": settle_price_str,
        "funding_rate_str": funding_rate_str,
        "mark_price": mark_price,
        "index_price": index_price,
        "estimated_settle_price": settle_price,
        "funding_rate": funding_rate,
        "next_funding_time_ms": _integer(payload, "T"),
        "event_time_ms": event_time,
        **{key: value for key, value in common.items() if key != "event_id"},
    }
    return _event("mark_price", stream_name, row)


def _kline(
    stream_name: str, payload: dict[str, Any], common: dict[str, object]
) -> CanonicalStreamEvent:
    if payload.get("e") != "kline":
        raise DataContractError("kline stream has an unexpected event type")
    event_time = _integer(payload, "E")
    raw_kline = _required(payload, "k", dict)
    kline = cast(dict[str, Any], raw_kline)
    if kline.get("i") != "1m":
        raise DataContractError("kline_1m stream returned a different interval")
    closed = _required(kline, "x", bool)
    _, open_price = _decimal(kline, "o")
    _, high_price = _decimal(kline, "h")
    _, low_price = _decimal(kline, "l")
    _, close_price = _decimal(kline, "c")
    _, base_volume = _decimal(kline, "v")
    _, quote_volume = _decimal(kline, "q")
    _, taker_buy_base_volume = _decimal(kline, "V")
    _, taker_buy_quote_volume = _decimal(kline, "Q")
    row = {
        "event_id": common["event_id"],
        "symbol": _string(payload, "s"),
        "interval": _string(kline, "i"),
        "open_time_ms": _integer(kline, "t"),
        "close_time_ms": _integer(kline, "T"),
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "base_volume": base_volume,
        "quote_volume": quote_volume,
        "trade_count": _integer(kline, "n"),
        "taker_buy_base_volume": taker_buy_base_volume,
        "taker_buy_quote_volume": taker_buy_quote_volume,
        "is_closed": closed,
        "event_time_ms": event_time,
        "ingested_at_ns": common["received_time_ns"],
        **{key: value for key, value in common.items() if key != "event_id"},
    }
    return _event("kline_1m", stream_name, row)
