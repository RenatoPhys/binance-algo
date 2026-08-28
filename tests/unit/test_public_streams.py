from __future__ import annotations

import orjson
import polars as pl
import pytest

from binance_algo.common.errors import DataContractError
from binance_algo.exchange.binance_usdm.public_streams import (
    STREAM_SCHEMAS,
    StreamSubscription,
    build_stream_subscriptions,
    combined_stream_url,
    parse_combined_message,
    stream_frame,
    stream_schema_contract_json,
    subscription_command,
)


def _combined(stream: str, payload: dict[str, object]) -> bytes:
    return orjson.dumps({"stream": stream, "data": payload})


@pytest.mark.parametrize(
    ("stream", "payload", "expected_type"),
    [
        (
            "btcusdt@bookTicker",
            {
                "e": "bookTicker",
                "E": 1_000,
                "T": 999,
                "s": "BTCUSDT",
                "u": 42,
                "b": "100.1",
                "B": "2.0",
                "a": "100.2",
                "A": "3.0",
            },
            "book_ticker",
        ),
        (
            "btcusdt@aggTrade",
            {
                "e": "aggTrade",
                "E": 1_001,
                "T": 1_000,
                "s": "BTCUSDT",
                "a": 10,
                "p": "100.15",
                "q": "0.5",
                "f": 20,
                "l": 21,
                "m": True,
            },
            "aggregate_trades",
        ),
        (
            "btcusdt@markPrice@1s",
            {
                "e": "markPriceUpdate",
                "E": 1_002,
                "s": "BTCUSDT",
                "p": "100.11",
                "i": "100.10",
                "P": "100.12",
                "r": "0.0001",
                "T": 8_000,
            },
            "mark_price",
        ),
        (
            "btcusdt@kline_1m",
            {
                "e": "kline",
                "E": 1_003,
                "s": "BTCUSDT",
                "k": {
                    "t": 0,
                    "T": 59_999,
                    "i": "1m",
                    "o": "100",
                    "h": "102",
                    "l": "99",
                    "c": "101",
                    "v": "5",
                    "q": "505",
                    "n": 7,
                    "V": "2",
                    "Q": "202",
                    "x": False,
                },
            },
            "kline_1m",
        ),
    ],
)
def test_parses_each_canonical_stream_contract(
    stream: str, payload: dict[str, object], expected_type: str
) -> None:
    message = _combined(stream, payload)
    first = parse_combined_message(
        message,
        received_time_ns=2_000_000_000,
        processed_time_ns=2_000_000_100,
        connection_id="connection-1",
        ingestion_run_id="run-1",
    )
    second = parse_combined_message(
        message,
        received_time_ns=3_000_000_000,
        processed_time_ns=3_000_000_100,
        connection_id="connection-2",
        ingestion_run_id="run-2",
    )

    assert first is not None and second is not None
    assert first.event_type == expected_type
    assert first.event_id == second.event_id
    assert first.row["payload_json"] == orjson.dumps(payload, option=orjson.OPT_SORT_KEYS).decode()
    frame = stream_frame(first.event_type, [first.row])
    assert dict(frame.schema) == STREAM_SCHEMAS[first.event_type]
    assert frame["schema_version"].dtype == pl.Int32
    contract = orjson.loads(stream_schema_contract_json(first.event_type))
    assert contract["dataset"] == expected_type
    assert contract["schema_version"] == 1
    assert contract["columns"] == {
        name: str(dtype) for name, dtype in STREAM_SCHEMAS[first.event_type].items()
    }


def test_builds_separate_routed_combined_subscriptions() -> None:
    subscriptions = build_stream_subscriptions(
        ("BTCUSDT", "ETHUSDT"),
        book_ticker=True,
        aggregate_trades=True,
        mark_price=True,
        kline_1m=True,
    )

    assert subscriptions[0] == StreamSubscription(
        route="public", names=("btcusdt@bookTicker", "ethusdt@bookTicker")
    )
    assert subscriptions[1].route == "market"
    assert combined_stream_url("wss://demo-fstream.binance.com", subscriptions[0]) == (
        "wss://demo-fstream.binance.com/public/stream?streams=btcusdt@bookTicker/ethusdt@bookTicker"
    )
    market_url = combined_stream_url("wss://demo-fstream.binance.com/market", subscriptions[1])
    assert market_url.startswith("wss://demo-fstream.binance.com/market/stream?streams=")
    assert "btcusdt@aggTrade" in market_url
    assert "ethusdt@kline_1m" in market_url

    with pytest.raises(DataContractError, match="duplicates"):
        build_stream_subscriptions(
            ("BTCUSDT", "btcusdt"),
            book_ticker=True,
            aggregate_trades=False,
            mark_price=False,
            kline_1m=False,
        )


def test_subscription_commands_and_control_ack() -> None:
    assert orjson.loads(subscription_command("SUBSCRIBE", ("btcusdt@aggTrade",), 7)) == {
        "method": "SUBSCRIBE",
        "params": ["btcusdt@aggTrade"],
        "id": 7,
    }
    assert (
        parse_combined_message(
            '{"result":null,"id":7}',
            received_time_ns=1,
            connection_id="c",
            ingestion_run_id="r",
        )
        is None
    )
    with pytest.raises(DataContractError, match="unsigned"):
        subscription_command("UNSUBSCRIBE", ("btcusdt@aggTrade",), -1)


def test_rejects_schema_drift_and_unknown_stream() -> None:
    with pytest.raises(DataContractError, match="invalid type"):
        parse_combined_message(
            _combined("btcusdt@bookTicker", {"e": "bookTicker", "E": "wrong", "s": "BTCUSDT"}),
            received_time_ns=1,
            connection_id="c",
            ingestion_run_id="r",
        )
    with pytest.raises(DataContractError, match="unsupported"):
        parse_combined_message(
            _combined("btcusdt@ticker", {"e": "ticker", "s": "BTCUSDT"}),
            received_time_ns=1,
            connection_id="c",
            ingestion_run_id="r",
        )
    with pytest.raises(DataContractError, match="not numeric"):
        parse_combined_message(
            _combined(
                "btcusdt@aggTrade",
                {
                    "e": "aggTrade",
                    "E": 1,
                    "T": 1,
                    "s": "BTCUSDT",
                    "a": 1,
                    "p": "not-a-price",
                    "q": "1",
                    "f": 1,
                    "l": 1,
                    "m": False,
                },
            ),
            received_time_ns=1,
            connection_id="c",
            ingestion_run_id="r",
        )
    with pytest.raises(DataContractError, match="symbol do not match"):
        parse_combined_message(
            _combined("btcusdt@bookTicker", {"e": "bookTicker", "s": "ETHUSDT"}),
            received_time_ns=1,
            connection_id="c",
            ingestion_run_id="r",
        )
