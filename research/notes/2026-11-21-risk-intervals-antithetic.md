# ICs de VaR/CVaR/P(perda) e variáveis antitéticas (rascunho para o vault AI Quant)

Status: rascunho. Vault `AI Quant` inacessível na sessão em nuvem; copiar para
o Obsidian com links internos (projeto GBM, decisões, experimentos).

## Objetivo e hipótese
Quantificar o erro Monte Carlo de VaR, CVaR e P(perda) e oferecer redução de
variância por antitéticas. Hipótese das antitéticas: razão de variâncias do
estimador da média = 1 - exp(-sigma^2 T).

## Implementação (branch ccr-0258fbf7-dn1zd5)
- `src/risk/intervals.py`: VaR por estatísticas de ordem (binomial exata,
  distribution-free); CVaR por bootstrap percentil (semente = seed da
  simulação, 1000 reamostras); P(perda) por Wilson (perda estrita `<`).
- `src/simulation/gbm.py`, `variance_reduction.py`: `antithetic=False` por
  padrão; colunas 0..m-1 usam Z e m..2m-1 usam -Z; n_sims ímpar e modo t são
  rejeitados; EP da média calculado sobre os m pares.
- `src/simulation/report.py`, `main.py`: `results.risk_intervals` no JSON e
  linhas no resumo; replay aceita relatórios antigos sem o bloco.
- Testes: `tests/test_risk_intervals.py`, `test_antithetic.py`,
  `test_report_intervals.py`.

## Metodologia e resultados medidos
- Cobertura vs referência lognormal (s0=100, mu=0,08, sigma=0,25, T=1):
  VaR 0,970 (R=500, n=1000); CVaR 0,950 (R=200, n=2000, B=400);
  P(perda) 0,951 (R=1000, n=500).
- Antitéticas (mu=0,08, sigma=0,2, T=1; 300 réplicas, n=2000): razão de
  variâncias 0,0389 observada vs 0,0392 teórica (~25x); cobertura do IC por
  pares 0,957.
- Revisão independente: índices do VaR conferidos contra scipy.stats.binom
  (n de 1 a 1e6) sem divergência. Achado P2 confirmado: bootstrap do CVaR com
  cauda pequena cobre bem menos (n=100, confiança 0,99: ~0,34; n=200, 0,95:
  ~0,86); corrigido com `tail_points` e `warning` (< 50 pontos).
- Suíte completa com scipy real: 362 testes, 51 subtestes aprovados.

## Limitações
- ICs cobrem só erro Monte Carlo com mu/sigma fixos; sem incerteza de
  calibração nem erro de modelo.
- Supõem amostras i.i.d.; antitéticas não expostas na CLI. Convergência por
  prefixos aninhados é inválida sobre a disposição em metades.
- CLI com rede (yfinance) e n_sims = 1e6 não executados.

## Decisões e próximos passos
- Antitéticas só na API até haver ICs próprios para pares.
- Próximos: variável de controle (útil só para payoffs, não para E[S(T)]),
  opções europeias vs Black-Scholes (declarar medida risco-neutra e
  dividendos), multiativo com Cholesky, arquivos de configuração, IC de
  calibração de mu/sigma.
