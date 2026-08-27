# Controles de risco

Este marco não contém execução. As barreiras preventivas são:

1. endpoints Demo como default;
2. `LIVE_TRADING=false` obrigatório;
3. `ALLOW_ORDER_SUBMISSION=false` obrigatório;
4. notional máximo obrigatoriamente zero;
5. configuração falha se uma barreira for relaxada;
6. CI fixa as flags em `false`;
7. não há módulos, endpoints ou interfaces de criação/cancelamento de ordem;
8. não há funções de saque ou transferência.

O risk engine, kill switch, reconciliação e as demais travas de execução só podem ser
implementados depois do simulador orientado a eventos, antes da primeira ordem Demo.
