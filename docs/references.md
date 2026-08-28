# Referências oficiais

Consultadas em 2026-08-27.

- [Informações gerais de USDⓈ-M Futures](https://developers.binance.com/docs/derivatives/usds-margined-futures/general-info)
- [Test Connectivity — GET /fapi/v1/ping](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Test-Connectivity)
- [Check Server Time — GET /fapi/v1/time](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Check-Server-Time)
- [Exchange Information — GET /fapi/v1/exchangeInfo](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Exchange-Information)
- [Códigos de erro USDⓈ-M](https://developers.binance.com/docs/derivatives/usds-margined-futures/error-code)
- [Changelog de Derivatives](https://developers.binance.com/docs/derivatives/change-log)
- [Binance Public Data](https://github.com/binance/binance-public-data)
- [Binance Data Collection](https://data.binance.vision/)

Os três endpoints usados neste marco são públicos e têm request weight 1 conforme as páginas
específicas. A página de Exchange Information marca seu `serverTime` como ignorável e direciona
ao endpoint dedicado de hora; por isso o lineage usa `/fapi/v1/time`. URLs são configuráveis
porque paths, hosts e limites podem mudar.

O repositório `binance/binance-public-data` foi revisado novamente em 2026-08-27. Ele documenta
arquivos daily/monthly, publicação diária no dia seguinte, schema de USD-M klines e `.CHECKSUM`
SHA-256. URLs daily e monthly de BTCUSDT 1m foram verificadas no serviço oficial; o contract test
usa o arquivo daily de 2026-08-25.
