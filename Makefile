.PHONY: sync format quality test doctor snapshot universe

sync:
	uv sync

format:
	uv run ruff format .
	uv run ruff check --fix .

quality:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy src
	uv run pytest -m "not network"

test:
	uv run pytest

doctor:
	uv run binance-algo doctor

snapshot:
	uv run binance-algo exchange-info snapshot

universe:
	uv run binance-algo universe build
