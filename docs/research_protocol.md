# Protocolo de pesquisa — Fase 3

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
