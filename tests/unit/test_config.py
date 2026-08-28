from __future__ import annotations

from pathlib import Path

import pytest

from binance_algo.common.errors import ConfigurationError
from binance_algo.config import load_settings

PROJECT_ROOT = Path(__file__).parents[2]
BASE_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"


def test_default_configuration_is_public_data_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LIVE_TRADING", raising=False)
    monkeypatch.delenv("ALLOW_ORDER_SUBMISSION", raising=False)

    settings = load_settings(BASE_CONFIG)

    assert settings.binance.authenticated_environment == "demo"
    assert settings.binance.rest_base_url == "https://demo-fapi.binance.com"
    assert settings.binance.market_ws_base_url == "wss://demo-fstream.binance.com"
    assert settings.recorder.queue_capacity == 100_000
    assert settings.streams.depth is False
    assert settings.safety.live_trading is False
    assert settings.safety.allow_order_submission is False
    assert settings.safety.max_order_notional_usdt == 0
    assert settings.credentials.configured is False
    assert settings.data_root == PROJECT_ROOT / "var" / "data"


@pytest.mark.parametrize("name", ["LIVE_TRADING", "ALLOW_ORDER_SUBMISSION"])
def test_environment_cannot_enable_orders(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setenv(name, "true")

    with pytest.raises(ConfigurationError, match="prohibited"):
        load_settings(BASE_CONFIG)


def test_unknown_configuration_key_fails(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "base.yaml").write_text(
        BASE_CONFIG.read_text(encoding="utf-8") + "\nunknown_section: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="unknown_section"):
        load_settings(config_dir / "base.yaml")


def test_universe_symbols_cannot_repeat(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    base = BASE_CONFIG.read_text(encoding="utf-8").replace("    - SOLUSDT", "    - BTCUSDT")
    (config_dir / "base.yaml").write_text(base, encoding="utf-8")

    with pytest.raises(ConfigurationError, match="must not contain duplicates"):
        load_settings(config_dir / "base.yaml")


def test_websocket_endpoint_requires_tls(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    base = BASE_CONFIG.read_text(encoding="utf-8").replace(
        "wss://demo-fstream.binance.com", "ws://demo-fstream.binance.com"
    )
    (config_dir / "base.yaml").write_text(base, encoding="utf-8")

    with pytest.raises(ConfigurationError, match="market_ws_base_url must use WSS"):
        load_settings(config_dir / "base.yaml")


def test_depth_cannot_be_enabled_in_basic_recorder(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    base = BASE_CONFIG.read_text(encoding="utf-8").replace("  depth: false", "  depth: true")
    (config_dir / "base.yaml").write_text(base, encoding="utf-8")

    with pytest.raises(ConfigurationError, match="depth remains disabled"):
        load_settings(config_dir / "base.yaml")
