# Arquitetura

O data plane histórico implementado é:

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
                              |                                  |
                              v                                  v
                     lineage + schema v1 <---- canonical Parquet (bronze)
                              |                                  |
                              v                                  v
                    quality_results <--- audit + DuckDB view `klines`
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

O normalizador deriva `ingested_at_ns` do manifesto raw para produzir conteúdo determinístico,
remove duplicatas pela chave canônica conservando a primeira linha e ordena antes da promoção.
O nome do Parquet inclui o prefixo do checksum raw e a escrita imutável impede substituição
silenciosa. A política para detectar e versionar correções upstream continua explicitamente
pendente.

A auditoria valida schema, checksum, nulidade, ordem, unicidade, alinhamento temporal,
continuidade e invariantes de mercado por arquivo e no range agregado. A view DuckDB é recriada a
partir da lista explícita de arquivos `NORMALIZED`, não de uma varredura cega do filesystem.

O research plane existe desde a Fase 3; o trading plane não existe. A separação física do
research plane é ampliada somente por incrementos com contratos e testes reais, sem criar árvores
vazias ou antecipar um simulador de execução.

O data plane em tempo real é independente do histórico:

```text
Binance /public (bookTicker) ----+
                                 +-> parser estrito -> asyncio.Queue limitada
Binance /market (3 streams) -----+                         |
                                                           v
                                             buffers por stream/hora/símbolo
                                                           |
                                  SQLite checkpoint <- Parquet atômico + manifesto
                                                           |
                                                           v
                           DuckDB views explícitas -> quality report -> replay offline
```

Cada rota tem `connection_id`, detecção de staleness, rotação preventiva, reconnect com backoff
exponencial e jitter e resubscribe determinístico. Uma fila cheia é uma falha observável, nunca
uma permissão para perder dados. O shutdown primeiro encerra os produtores, depois drena a fila e
só então fecha writer e health server.

O `event_id` é SHA-256 do nome do stream e do payload JSON canônico; timestamps locais, conexão e
run não entram no hash. Isso torna duplicatas de transporte detectáveis sem apagar a proveniência.
O replay faz `UNION ALL BY NAME` apenas dos paths `VALIDATED` selecionados no SQLite e ordena por
`event_time_ms`, `received_time_ns`, `event_id` e tipo. O clock é uma interface injetável e o
motor não importa nem chama nenhum cliente de rede.

O research plane da Fase 3 é separado do recorder:

```text
klines fechadas ----+                         +-> labels no próximo open
                    +-> painel causal horário +-> score residual cross-sectional
funding REST --------+                         +-> walk-forward vetorizado
                                                          |
                                                          v
                                  preço + funding - fee - spread - slippage
                                                          |
                                                          v
                              estresses + bootstrap + relatório versionado
```

O universo é o seed fixo da especificação, não a lista atual ranqueada por liquidez. O builder
exige grid comum, não preenche o passado e elimina o timestamp inteiro quando qualquer símbolo
não possui lookback causal. O backtest não importa adapters Binance, não acessa rede e não produz
ordens. O modelo next-open serve para triagem; fills parciais, fila, latência e estado de conta
serão responsabilidades do motor orientado a eventos.

A Fase 3.5 começou pelos contratos, mantendo o motor financeiro intacto:

```text
FoldContext + TrainingDataset -> Strategy.fit -> FittedStrategy.score
                                      |                    |
                         features em allowlist      StrategyScores
                                                           |
                                                           v
                                                   PortfolioPolicy
```

`select_feature_view` entrega somente chaves e features declaradas. Colunas `future_*`,
`outcome_*` e `label_*` não podem ser features; targets de treino ficam separados e alinhados pela
chave. O PR 2 conectou esses contratos ao motor com equivalência integral ao golden baseline:

```text
baseline.py (compat config) -> ResidualMomentumStrategy
                            -> NeutralLongShortPolicy
                                      |
                                      v
backtest.py: train slice -> fit -> test score -> target weights -> accounting
```

O engine conhece somente os contratos, as colunas contábeis padronizadas e os folds. Nomes/pesos
de momentum e regras de seleção/neutralização não estão no engine. A CLI antiga constrói os
componentes pelo adaptador; não há rota paralela de cálculo.

O PR 4 acrescenta o plano de controle durável da pesquisa:

```text
DatasetReference + CodeFingerprint + componentes
                       |
                       v
                ExperimentSpec -- JSON canônico/SHA-256 --> experiment_id
                       |                                       |
                       v                                       v
        ResearchStore (definitions)              runs + attempts + states
                                                               |
                                                               v
                                             metrics + artifact checksums
                                                               |
                                                               v
                                                        result_digest
```

O banco `research.sqlite3` é separado do `ingestion.sqlite3`: o primeiro registra decisões e
resultados de pesquisa, enquanto o segundo continua sendo a autoridade sobre materialização de
dados. Paths são localizadores operacionais e não participam da identidade do experimento. O
PR 5 conecta a execução sem criar outro motor. O worker escreve o bundle em `tmp/research`,
valida Parquet/JSON, checksums e row counts, e promove o diretório inteiro para o layout por
experiment/run. Uma única transação final registra métricas e artifacts e grava o
`result_digest`; até esse ponto o run permanece `RUNNING`. Falhas são preservadas em quarantine.
Scores e posições long-form são materializados somente pela policy `full`; o JSON legado da curva
permanece apenas para a regressão do baseline. Factories explícitas rejeitam componentes e
parâmetros não allowlisted.

O PR 6 adiciona o coordenador local de campanhas:

```text
YAML estrito -> canonical campaign plan -> constraints -> immutable ExperimentSpecs
                                                       |
                                                       v
                                      cache verify -> local process workers
                                                       |
                                                       v
                         COMPLETED/PARTIAL/FAILED -> comparison de todos os trials
```

O coordenador registra definições e associações antes de executar. Cada worker abre sua conexão
SQLite; não há conexão compartilhada ou infraestrutura distribuída. Resume reaproveita apenas
sucessos íntegros e cria attempts novos para falhas. O report agregado é uma visão atômica e
mutável sobre runs imutáveis, não uma fonte paralela de verdade.
