FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LIVE_TRADING=false \
    ALLOW_ORDER_SUBMISSION=false

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs ./configs

RUN uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "--frozen", "binance-algo"]
CMD ["doctor"]
