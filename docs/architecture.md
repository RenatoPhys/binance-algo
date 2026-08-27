# Arquitetura

O primeiro marco implementa apenas o início do data plane:

```text
YAML + env allowlist
        |
        v
REST público Binance -> raw JSON imutável -> parser canônico -> bronze Parquet
                                                        |
                                                        v
                                             universo point-in-time + manifesto
```

O adapter Binance é fino e explícito: lifecycle da sessão, timeout, classificação de erro,
retry de GET seguro e captura de headers de rate limit permanecem no projeto. Modelos canônicos
não vazam DTOs do venue para consumidores futuros.

`LocalFilesystemStorage` é a primeira implementação de uma interface pequena. A promoção de
arquivos acontece somente após validação, e todos os destinos são confinados à raiz de storage.

Research plane e trading plane não existem neste PR. A separação física será ampliada quando
houver entregáveis reais; não foram criadas árvores de arquivos vazios.
