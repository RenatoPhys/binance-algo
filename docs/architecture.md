# Arquitetura

O data plane implementado até o state store e archive downloader é:

```text
YAML + env allowlist
        |
        v
REST público Binance -> raw JSON imutável -> parser canônico -> bronze Parquet
                                                        |
                                                        v
                                             universo point-in-time + manifesto

Binance Public Data -> .CHECKSUM -> ZIP .part/resume -> SHA-256 -> ZIP/CSV validation
                              |                                  |
                              v                                  v
                     SQLite WAL manifest <-------------- raw archive + CSV
```

O adapter Binance é fino e explícito: lifecycle da sessão, timeout, classificação de erro,
retry de GET seguro e captura de headers de rate limit permanecem no projeto. Modelos canônicos
não vazam DTOs do venue para consumidores futuros.

`LocalFilesystemStorage` é a primeira implementação de uma interface pequena. A promoção de
arquivos acontece somente após validação, e todos os destinos são confinados à raiz de storage.

`StateStore` abre transações `BEGIN IMMEDIATE`, ativa foreign keys, `busy_timeout` e WAL. O banco
armazena migrations, arquivos, jobs, checkpoints, resultados de qualidade e versões de schema.
Transições inválidas falham; timeout ou checksum incorreto nunca aparece como arquivo validado.

O archive downloader limita concorrência, tamanho comprimido/descomprimido e retries. Arquivos
parciais permanecem em `.part`; arquivos divergentes são movidos para quarantine, sem
sobrescrever silenciosamente o raw observado.

Research plane e trading plane não existem neste PR. A separação física será ampliada quando
houver entregáveis reais; não foram criadas árvores de arquivos vazios.
