# Protocolo de pesquisa — Fase 3.5

## Semântica temporal

| Campo | Semântica |
|---|---|
| `decision_time_ms` | fechamento confirmado da kline 1m usada na decisão |
| `feature_source_max_ms` | maior timestamp permitido nas features; nunca supera a decisão |
| `execution_time_ms` | open da próxima kline 1m, estritamente após a decisão |
| `label_end_time_ms` | open 60 minutos após a execução |
| `funding_rate_current` | último evento observado em ou antes da decisão |
| `outcome_funding_rate_1h` | soma de eventos em `(execution, label_end]` |

Um evento de funding exatamente no instante de rebalanceamento pertence à posição anterior. Isso
impede que o backtest escolha a posição depois de conhecer o débito/crédito daquele instante.
Funding e metadata não recebem backward-fill. Linhas com lookback incompleto são removidas para
todos os símbolos no timestamp, preservando um painel cross-sectional balanceado.

## Features e baseline

O conjunto versionado contém retornos log 5m/15m/1h/4h/24h, volatilidade realizada 24h, range
4h, quote volume e z-score, taker-buy imbalance, beta rolling de 7 dias, momentum residual
1h/4h/24h, regime de volatilidade e funding atual/variação. BTC usa ETH como benchmark, ETH usa
BTC e os demais usam a média BTC/ETH, evitando que cada anchor use exclusivamente a si próprio.

O score é a combinação cross-sectional 20%/30%/50% dos momenta residuais padronizados. A seleção
usa no-trade band, long no topo e short na cauda. Pesos são projetados para neutralidade líquida e
beta quando a geometria de três símbolos permite, limitados por símbolo e escalados por
volatilidade. `gross_exposure=0.50` é o teto inicial; alavancagem econômica é rejeitada.

Na Fase 3.5, essa semântica está separada em `ResidualMomentumStrategy:v1` e
`NeutralLongShortPolicy:v1`. Para cada outer fold, o engine projeta apenas as features declaradas
no intervalo de treino, chama `fit`, congela o resultado e pontua somente o teste. A estratégia
fixa valida o treino, mas não calibra nenhum parâmetro. Labels/outcomes não entram no score; a
política recebe apenas scores, beta e volatilidade realizada.

## Custos e validação

Cada rebalanceamento contabiliza turnover de uma via e usa taker fee, meio spread e slippage do
arquivo de configuração. Funding positivo é pago por long e recebido por short. A identidade
verificada por período é:

```text
net_return = price_pnl + funding_pnl - trading_fees - spread_cost - slippage_cost
```

O walk-forward é expanding, com 30 dias iniciais, 14 dias por teste e embargo de uma barra. O
baseline não calibra parâmetros no teste. O relatório inclui gross versus net, turnover,
capacidade por quote volume observado, rank IC, drawdown, regimes, custo 1,5×/2×, atraso de uma
barra, duas perturbações de pesos e bootstrap determinístico em blocos de 24 horas.

O resultado real de 90 dias é deliberadamente mantido mesmo sendo negativo. Ele valida que custos
podem rejeitar a hipótese e não deve ser interpretado como previsão, recomendação ou edge.

## Pré-registro na Fase 3.5

Antes de executar uma nova configuração, registre uma hipótese com mecanismo e critérios de
sucesso definidos ex ante. Depois componha um `ExperimentSpec` imutável que fixe dataset,
features, label, strategy, política de portfólio, custos, splits, validação, seed e code
fingerprint. Alterações nesses elementos criam outro `experiment_id`; um rerun sem alteração cria
somente outra tentativa.

O registry, experiment runner e campaign runner tornam esse contrato persistente e auditável, mas
não autorizam sweeps ad hoc. Toda campanha registra o número integral de trials e preserva falhas
e resultados negativos.

Ablações devem ser pré-declaradas como pares baseline/candidate dentro da campanha. Os deltas são
sempre interpretados como `com feature - sem feature`, mesmo em remove-one, e incluem retorno,
Sharpe, drawdown, rank IC, turnover, custos explícitos, capacidade e concentração mensal. A regra
automática e qualquer override ficam registrados com motivo. Uma rejeição é contextual: não
desativa a feature globalmente e não pode ser apagada do histórico.

Antes de `CANDIDATE`, gere o relatório robusto da campanha inteira. O melhor ponto precisa ser
avaliado contra folds, regimes, meses, símbolos, custos 1,5×, atraso e parâmetros vizinhos. DSR
usa todos os Sharpes comparáveis e as características dos retornos do selecionado. PBO só é
interpretado quando o desenho CSCV mínimo está presente; `NOT_APPLICABLE` não equivale a aprovação
nem reprovação.

Promoção exige proveniência Git limpa tanto no experimento quanto no código que toma a decisão.
O histórico atual é development OOS, não lockbox. Sem manifest/período independente, o gate
registra `NOT_AVAILABLE` e `PHASE4_CANDIDATE` permanece impossível.

## Execução em escala local

O dataset point-in-time é materializado antes do sweep. Cada worker usa lazy scan com projeção de
colunas e conserva um `PanelData` read-only; trials que mudam somente parâmetros reutilizam as
mesmas features/outcomes. Baseline, custos e atraso compartilham o painel e fatiam views temporais.
Runtime de cada attempt fica no registry; o benchmark separado mede também memória aproximada e
tamanho dos artifacts sem estabelecer SLA de CI.

O `availability` prepara exclusão localizada, mas não resolve survivorship bias por si só. Até
existirem snapshots históricos de listing, delisting, qualidade e liquidez, campanhas oficiais
devem continuar no universo fixado ex ante. É proibido usar o estado atual da exchange para
reconstruir composição histórica.

## Próximas hipóteses permitidas

Com a infraestrutura concluída, podem começar screenings pequenos e pré-registrados de momentum
mais lento (4h–168h), funding carry e mean reversion residual. Cada família deve ser uma strategy
versionada distinta, passar pelo registry/campaign/ledger e permanecer development OOS. Nenhuma
está apta à Fase 4 sem estabilidade líquida, correção por múltiplos testes e lockbox independente.
