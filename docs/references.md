# Referências oficiais

Consultadas em 2026-08-27.

- [Informações gerais de USDⓈ-M Futures](https://developers.binance.com/docs/derivatives/usds-margined-futures/general-info)
- [Test Connectivity — GET /fapi/v1/ping](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Test-Connectivity)
- [Check Server Time — GET /fapi/v1/time](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Check-Server-Time)
- [Exchange Information — GET /fapi/v1/exchangeInfo](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Exchange-Information)
- [Códigos de erro USDⓈ-M](https://developers.binance.com/docs/derivatives/usds-margined-futures/error-code)
- [Changelog de Derivatives](https://developers.binance.com/docs/derivatives/change-log)
- [Conexão aos market streams USDⓈ-M](https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/websocket-market-streams/Connect)
- [Mudança para rotas public/market/private](https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/websocket-market-streams/Important-WebSocket-Change-Notice)
- [Subscribe/unsubscribe em streams](https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/websocket-market-streams/Live-Subscribing-Unsubscribing-to-streams)
- [Binance Public Data](https://github.com/binance/binance-public-data)
- [Binance Data Collection](https://data.binance.vision/)

Os três endpoints usados neste marco são públicos e têm request weight 1 conforme as páginas
específicas. A página de Exchange Information marca seu `serverTime` como ignorável e direciona
ao endpoint dedicado de hora; por isso o lineage usa `/fapi/v1/time`. URLs são configuráveis
porque paths, hosts e limites podem mudar.

Os documentos de WebSocket foram revisados em 2026-08-27. A conexão combinada usa
`/public/stream?streams=...` para `bookTicker` e `/market/stream?streams=...` para `aggTrade`,
`markPrice` e `kline`. Símbolos nos nomes de stream são minúsculos. O lifecycle considera o
limite de 24 horas, ping do servidor, limite de mensagens e até 1.024 streams por conexão. O host
continua configurável para acomodar novas mudanças oficiais sem misturar rotas públicas e
privadas.

O repositório `binance/binance-public-data` foi revisado novamente em 2026-08-27. Ele documenta
arquivos daily/monthly, publicação diária no dia seguinte, schema de USD-M klines e `.CHECKSUM`
SHA-256. URLs daily e monthly de BTCUSDT 1m foram verificadas no serviço oficial; o contract test
usa o arquivo daily de 2026-08-25.
