from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from binance_algo.common.errors import DataContractError
from binance_algo.exchange.binance_usdm.models import parse_instruments

FIXTURE = Path(__file__).parents[1] / "fixtures" / "exchange_info.json"


def load_fixture() -> dict[str, Any]:
    loaded = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_parses_real_anonymization_free_public_fixture() -> None:
    instruments = parse_instruments(load_fixture(), ingested_at_ns=123)

    assert [instrument.symbol for instrument in instruments] == [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
    ]
    bitcoin = instruments[0]
    assert bitcoin.tick_size == "0.10"
    assert bitcoin.step_size == "0.0001"
    assert bitcoin.min_notional == "50"
    assert bitcoin.valid_from_ms == 1_787_873_855_926
    assert bitcoin.ingested_at_ns == 123
    assert '"filterType":"PRICE_FILTER"' in bitcoin.raw_filters_json


def test_missing_required_filter_fails_explicitly() -> None:
    payload = copy.deepcopy(load_fixture())
    payload["symbols"][0]["filters"] = [
        item for item in payload["symbols"][0]["filters"] if item["filterType"] != "PRICE_FILTER"
    ]

    with pytest.raises(DataContractError, match="PRICE_FILTER"):
        parse_instruments(payload)


def test_dedicated_server_time_can_override_stale_exchange_info_time() -> None:
    instruments = parse_instruments(
        load_fixture(), valid_from_ms=1_787_873_999_999, ingested_at_ns=123
    )

    assert {instrument.valid_from_ms for instrument in instruments} == {1_787_873_999_999}
