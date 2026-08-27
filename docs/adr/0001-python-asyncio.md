# ADR 0001 — Python 3.12+ e asyncio

Status: aceito em 2026-08-27.

Rede e serviços long-running usarão `asyncio`. O primeiro adapter usa `aiohttp` com lifecycle
explícito. Notebooks não serão runtime operacional.

Consequência: comandos CLI curtos criam e encerram seu event loop; serviços futuros compartilharão
as mesmas interfaces assíncronas.
