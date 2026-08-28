# ADR 0005 — Dataset causal e baseline cross-sectional

- Status: accepted
- Date: 2026-08-28

## Context

O primeiro backtest precisa validar o research plane sem transformar um resultado histórico em
alegação de edge. O histórico disponível contém três contratos seed e 90 dias de klines 1m. Há
um snapshot atual de metadata, mas não uma série histórica de status/liquidez; aplicar esse
snapshot ao passado criaria survivorship leakage. Funding histórico público está disponível como
evento separado.

## Decision

Usar o universo fixo BTCUSDT, ETHUSDT e SOLUSDT definido ex ante na especificação. A expansão
para universo dinâmico fica proibida até existirem snapshots históricos. Cada decisão ocorre no
`close_time_ms` de uma kline fechada; a execução ocorre no próximo `open_time_ms`. O label de uma
hora vai desse próximo open ao open 60 minutos depois. Nenhum campo `outcome_*` entra no sinal.

As features são pequenas e interpretáveis: retornos 5m/15m/1h/4h/24h, volatilidade realizada,
range, volume z-score, taker imbalance, beta rolling, momentum residual e funding conhecido via
as-of join. Funding nunca recebe backward-fill. O score linear ranqueia momentum residual; o
portfólio compra o topo, vende a cauda, tenta neutralizar net/beta, limita peso, escala
volatilidade e nunca excede exposição bruta 1.

O motor usa folds expanding walk-forward com embargo, execução marketable no próximo open e
contabilidade separada de preço, funding, taker fee, meio spread e slippage. A taxa é uma
`FeeSchedule` versionada em YAML e declarada como hipótese, não como fee atual da conta. Cada
fold começa e termina flat. Estresses cobrem custo 1,5×/2×, atraso de uma barra, pesos de
momentum perturbados, regimes e bootstrap em blocos.

## Consequences

- O dataset e o relatório são reproduzíveis e falham quando timestamps ou funding são
  incompatíveis.
- O resultado pode — e neste marco deve poder — rejeitar o baseline sem ajuste oportunista.
- O modelo bar/next-open não estima fill intrabar, fila ou adverse selection; isso pertence ao
  simulador orientado a eventos da Fase 4.
- O universo fixo evita ranking de liquidez futuro, mas não resolve histórico de delistagens para
  um universo ampliado.

