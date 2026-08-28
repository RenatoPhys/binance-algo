"""Strongly typed YAML configuration with narrow environment overrides."""

from __future__ import annotations

import os
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr, model_validator
from pydantic_settings import BaseSettings

from binance_algo.common.errors import ConfigurationError


class StrictModel(BaseModel):
    """Base model that rejects misspelled and obsolete configuration keys."""

    model_config = ConfigDict(extra="forbid")


class AppConfig(StrictModel):
    environment: str = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    timezone: Literal["UTC"] = "UTC"


class BinanceConfig(StrictModel):
    market: Literal["usdm_futures"] = "usdm_futures"
    authenticated_environment: Literal["demo"] = "demo"
    rest_base_url: str = "https://demo-fapi.binance.com"
    market_data_rest_base_url: str = "https://fapi.binance.com"
    market_ws_base_url: str = "wss://demo-fstream.binance.com"
    websocket_api_url: str = "wss://testnet.binancefuture.com/ws-fapi/v1"
    recv_window_ms: int = Field(default=5_000, ge=1_000, le=60_000)
    request_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    retry_max_attempts: int = Field(default=3, ge=1, le=8)
    retry_base_seconds: float = Field(default=0.25, ge=0, le=10)
    reconnect_max_seconds: float = Field(default=60.0, gt=0, le=600)
    clock_max_offset_ms: int = Field(default=1_000, ge=0, le=60_000)

    @model_validator(mode="after")
    def validate_endpoint(self) -> Self:
        self.rest_base_url = self.rest_base_url.rstrip("/")
        if not self.rest_base_url.startswith("https://"):
            raise ValueError("binance.rest_base_url must use HTTPS")
        self.market_data_rest_base_url = self.market_data_rest_base_url.rstrip("/")
        if not self.market_data_rest_base_url.startswith("https://"):
            raise ValueError("binance.market_data_rest_base_url must use HTTPS")
        self.market_ws_base_url = self.market_ws_base_url.rstrip("/")
        if not self.market_ws_base_url.startswith("wss://"):
            raise ValueError("binance.market_ws_base_url must use WSS")
        self.websocket_api_url = self.websocket_api_url.rstrip("/")
        if not self.websocket_api_url.startswith("wss://"):
            raise ValueError("binance.websocket_api_url must use WSS")
        return self


class StorageConfig(StrictModel):
    root: Path = Path("var/data")
    state_db: Path = Path("var/state/ingestion.sqlite3")
    research_db: Path = Path("var/state/research.sqlite3")
    duckdb: Path = Path("var/state/market_data.duckdb")
    reports_root: Path = Path("var/reports")
    parquet_compression: Literal["zstd", "snappy"] = "zstd"
    micro_batch_max_rows: int = Field(default=25_000, gt=0)
    micro_batch_max_seconds: int = Field(default=30, gt=0)


class ArchiveConfig(StrictModel):
    base_url: str = "https://data.binance.vision/data"
    request_timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    max_concurrency: int = Field(default=4, ge=1, le=32)
    max_attempts: int = Field(default=3, ge=1, le=8)
    retry_base_seconds: float = Field(default=0.5, ge=0, le=10)
    publication_lag_days: int = Field(default=1, ge=1, le=7)
    max_archive_bytes: int = Field(default=100_000_000, gt=0)
    max_uncompressed_bytes: int = Field(default=500_000_000, gt=0)
    chunk_bytes: int = Field(default=1_048_576, ge=65_536, le=8_388_608)

    @model_validator(mode="after")
    def validate_archive_endpoint(self) -> Self:
        self.base_url = self.base_url.rstrip("/")
        if not self.base_url.startswith("https://"):
            raise ValueError("archives.base_url must use HTTPS")
        return self


class UniverseConfig(StrictModel):
    seed_symbols: list[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    quote_asset: str = "USDT"
    contract_type: str = "PERPETUAL"
    require_status: str = "TRADING"
    minimum_listing_days: int = Field(default=90, ge=0)
    maximum_symbols: int = Field(default=10, ge=1, le=1_000)

    @model_validator(mode="after")
    def normalize_symbols(self) -> Self:
        normalized = [symbol.strip().upper() for symbol in self.seed_symbols]
        if not normalized or any(not symbol for symbol in normalized):
            raise ValueError("universe.seed_symbols must contain at least one non-empty symbol")
        if len(normalized) != len(set(normalized)):
            raise ValueError("universe.seed_symbols must not contain duplicates")
        if len(normalized) > self.maximum_symbols:
            raise ValueError("universe.seed_symbols exceeds universe.maximum_symbols")
        self.seed_symbols = normalized
        return self


class StreamsConfig(StrictModel):
    book_ticker: bool = True
    aggregate_trades: bool = True
    mark_price: bool = True
    kline_1m: bool = True
    depth: bool = False

    @model_validator(mode="after")
    def enforce_basic_recorder_scope(self) -> Self:
        if self.depth:
            raise ValueError("streams.depth remains disabled until the basic recorder is stable")
        if not any((self.book_ticker, self.aggregate_trades, self.mark_price, self.kline_1m)):
            raise ValueError("at least one public market stream must be enabled")
        return self


class RecorderConfig(StrictModel):
    queue_capacity: int = Field(default=100_000, ge=1_000, le=2_000_000)
    queue_put_timeout_seconds: float = Field(default=1.0, gt=0, le=30)
    stale_after_seconds: float = Field(default=15.0, gt=1, le=300)
    reconnect_base_seconds: float = Field(default=0.5, ge=0, le=30)
    reconnect_stable_after_seconds: float = Field(default=30.0, gt=0, le=600)
    connection_max_seconds: int = Field(default=82_800, ge=300, le=86_399)
    shutdown_timeout_seconds: float = Field(default=30.0, gt=1, le=300)
    metrics_host: str = "127.0.0.1"
    metrics_port: int = Field(default=9108, ge=0, le=65_535)


class FeeScheduleConfig(StrictModel):
    """Versioned research assumption; never inferred from the current account."""

    maker_fee_rate: Decimal = Field(ge=0, le=Decimal("0.01"))
    taker_fee_rate: Decimal = Field(ge=0, le=Decimal("0.01"))
    effective_from: date
    effective_to: date | None = None
    source: str = Field(min_length=1)
    account_tier: str = Field(min_length=1)
    bnb_discount: bool = False

    @model_validator(mode="after")
    def validate_effective_range(self) -> Self:
        if self.effective_to is not None and self.effective_to < self.effective_from:
            raise ValueError("fee_schedule.effective_to must not precede effective_from")
        return self


class ResearchConfig(StrictModel):
    """Causal baseline and validation protocol for Phase 3."""

    decision_interval_minutes: Literal[60] = 60
    forward_horizon_minutes: Literal[60] = 60
    beta_window_hours: int = Field(default=168, ge=24, le=720)
    gross_exposure: Decimal = Field(default=Decimal("0.50"), gt=0, le=1)
    annual_volatility_target: Decimal = Field(default=Decimal("0.15"), gt=0, le=1)
    max_symbol_weight: Decimal = Field(default=Decimal("0.25"), gt=0, le=1)
    no_trade_score_band: Decimal = Field(default=Decimal("0.10"), ge=0, le=5)
    momentum_weight_1h: Decimal = Field(default=Decimal("0.20"), ge=0, le=1)
    momentum_weight_4h: Decimal = Field(default=Decimal("0.30"), ge=0, le=1)
    momentum_weight_24h: Decimal = Field(default=Decimal("0.50"), ge=0, le=1)
    spread_bps: Decimal = Field(default=Decimal("1.0"), ge=0, le=100)
    slippage_bps: Decimal = Field(default=Decimal("1.0"), ge=0, le=100)
    initial_capital_usdt: Decimal = Field(default=Decimal("10000"), gt=0)
    walk_forward_train_days: int = Field(default=30, ge=7, le=3650)
    walk_forward_test_days: int = Field(default=14, ge=1, le=365)
    embargo_bars: int = Field(default=1, ge=1, le=24)
    block_bootstrap_samples: int = Field(default=500, ge=100, le=10_000)
    block_bootstrap_hours: int = Field(default=24, ge=2, le=720)
    random_seed: int = 42
    funding_required: bool = True
    fee_schedule: FeeScheduleConfig

    @model_validator(mode="after")
    def validate_research_protocol(self) -> Self:
        weight_sum = self.momentum_weight_1h + self.momentum_weight_4h + self.momentum_weight_24h
        if weight_sum != Decimal("1"):
            raise ValueError("research momentum weights must sum exactly to 1")
        if self.max_symbol_weight * 2 < self.gross_exposure / Decimal("3"):
            raise ValueError("research.max_symbol_weight is too small for the target exposure")
        return self


class ResearchPlatformConfig(StrictModel):
    max_campaign_trials: int = Field(default=1_000, ge=1, le=100_000)
    default_max_workers: int = Field(default=4, ge=1, le=32)
    default_artifact_policy: Literal["summary", "full"] = "summary"
    allow_dirty_code: bool = True
    promotion_requires_clean_git: bool = True
    stale_run_after_minutes: int = Field(default=60, ge=1, le=10_080)
    heartbeat_seconds: int = Field(default=30, ge=1, le=3_600)
    chart_generation: Literal["opt_in"] = "opt_in"
    min_profitable_folds: int = Field(default=2, ge=1)
    max_month_concentration: float = Field(default=0.75, gt=0, le=1)
    max_symbol_concentration: float = Field(default=0.75, gt=0, le=1)
    min_neighbor_positive_fraction: float = Field(default=0.50, ge=0, le=1)
    min_dsr_probability: float = Field(default=0.95, gt=0.5, lt=1)
    lockbox_manifest: str | None = None


class SafetyConfig(StrictModel):
    live_trading: bool = False
    allow_order_submission: bool = False
    max_order_notional_usdt: Decimal = Field(default=Decimal(0), ge=0)

    @model_validator(mode="after")
    def enforce_first_milestone(self) -> Self:
        if self.live_trading:
            raise ValueError("LIVE_TRADING=true is prohibited in the first milestone")
        if self.allow_order_submission:
            raise ValueError("ALLOW_ORDER_SUBMISSION=true is prohibited in the first milestone")
        if self.max_order_notional_usdt != 0:
            raise ValueError("max_order_notional_usdt must remain zero in the first milestone")
        return self


class Credentials(StrictModel):
    api_key: SecretStr | None = None
    api_secret: SecretStr | None = None

    @property
    def configured(self) -> bool:
        return bool(
            self.api_key
            and self.api_key.get_secret_value()
            and self.api_secret
            and self.api_secret.get_secret_value()
        )


class Settings(StrictModel):
    app: AppConfig
    binance: BinanceConfig
    storage: StorageConfig
    archives: ArchiveConfig
    universe: UniverseConfig
    streams: StreamsConfig
    recorder: RecorderConfig
    research: ResearchConfig
    research_platform: ResearchPlatformConfig = Field(default_factory=ResearchPlatformConfig)
    safety: SafetyConfig
    credentials: Credentials = Field(default_factory=Credentials)
    _project_root: Path = PrivateAttr(default_factory=Path.cwd)

    @property
    def project_root(self) -> Path:
        return self._project_root

    @property
    def data_root(self) -> Path:
        path = self.storage.root
        return path if path.is_absolute() else self.project_root / path

    @property
    def state_db_path(self) -> Path:
        path = self.storage.state_db
        return path if path.is_absolute() else self.project_root / path

    @property
    def research_db_path(self) -> Path:
        path = self.storage.research_db
        return path if path.is_absolute() else self.project_root / path

    @property
    def duckdb_path(self) -> Path:
        path = self.storage.duckdb
        return path if path.is_absolute() else self.project_root / path

    @property
    def reports_root(self) -> Path:
        path = self.storage.reports_root
        return path if path.is_absolute() else self.project_root / path


class EnvironmentOverrides(BaseSettings):
    """Allowlist of environment variables; arbitrary config is intentionally unsupported."""

    binance_api_key: SecretStr | None = None
    binance_api_secret: SecretStr | None = None
    binance_rest_base_url: str | None = None
    live_trading: bool | None = None
    allow_order_submission: bool | None = None


ENVIRONMENT_FIELDS = {
    "BINANCE_API_KEY": "binance_api_key",
    "BINANCE_API_SECRET": "binance_api_secret",
    "BINANCE_REST_BASE_URL": "binance_rest_base_url",
    "LIVE_TRADING": "live_trading",
    "ALLOW_ORDER_SUBMISSION": "allow_order_submission",
}


def _environment_overrides(env_file: Path) -> EnvironmentOverrides:
    raw_values = dotenv_values(env_file) if env_file.exists() else {}
    payload: dict[str, Any] = {}
    for environment_name, field_name in ENVIRONMENT_FIELDS.items():
        value = os.environ.get(environment_name, raw_values.get(environment_name))
        if value not in (None, ""):
            payload[field_name] = value
    return EnvironmentOverrides.model_validate(payload)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot load configuration {path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigurationError(f"configuration root must be a mapping: {path}")
    return {str(key): value for key, value in loaded.items()}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def load_settings(config_path: Path | str = Path("configs/base.yaml")) -> Settings:
    """Load base YAML, an optional overlay, then the small environment allowlist."""

    path = Path(config_path).resolve()
    base_path = path.parent / "base.yaml"
    if path.name != "base.yaml" and base_path.exists():
        payload = _deep_merge(_load_yaml(base_path), _load_yaml(path))
    else:
        payload = _load_yaml(path)

    project_root = path.parent.parent
    env_file = project_root / ".env"
    overrides = _environment_overrides(env_file)
    env_values = overrides.model_dump(exclude_none=True)

    binance_override: dict[str, Any] = {}
    if "binance_rest_base_url" in env_values:
        binance_override["rest_base_url"] = env_values["binance_rest_base_url"]
    if binance_override:
        payload = _deep_merge(payload, {"binance": binance_override})

    safety_override = {
        key: env_values[key]
        for key in ("live_trading", "allow_order_submission")
        if key in env_values
    }
    if safety_override:
        payload = _deep_merge(payload, {"safety": safety_override})

    payload["credentials"] = {
        "api_key": env_values.get("binance_api_key"),
        "api_secret": env_values.get("binance_api_secret"),
    }
    try:
        settings = Settings.model_validate(payload)
    except ValueError as exc:
        raise ConfigurationError(f"invalid configuration: {exc}") from exc
    settings._project_root = project_root
    return settings
