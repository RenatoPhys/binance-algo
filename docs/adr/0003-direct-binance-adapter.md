# ADR 0003 — Adapter Binance direto e fino

Status: aceito em 2026-08-27.

O core não depende de CCXT nem de um SDK autogerado. O adapter local explicita endpoints, timeout,
retry, erros, rate limits e lifecycle.

Consequência: há mais código de integração sob nossa responsabilidade, compensado por contract
tests e menor acoplamento dos modelos canônicos ao venue.
