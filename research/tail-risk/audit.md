# Fase 1 — Auditoria do simulador (tail risk)

Estado: auditoria concluída por leitura de código + execuções listadas abaixo. Nada aqui é resultado de mercado.

## Mapa de módulos
| Função | Local |
|---|---|
| GBM / Student-t / trajetórias | `src/simulation/gbm.py` (`GbmSimulator`) |
| Calibração (252 pregões) | `src/calibration/estimation.py`, `GbmSimulator.calibrate` |
| VaR/CVaR (perdas `s0 - S_T`, unidades de preço) | `src/risk/statistics.py` |
| Gráfico original | **Não existe no repositório** (nenhum script usa df=4 nem calcula P(queda ≥ 8,6%)) |

## Verificação matemática
- **Normal:** incrementos lognormais exatos, `log(S_t/S_{t-1}) = (mu - σ²/2)dt + σ√dt·Z`. Correto; `mu` é drift do preço.
- **Student-t:** choques t independentes **a cada passo da grade** (diários), padronizados por `sqrt((df-2)/df)` → variância unitária (requer df>2; verificado em teste: Var[log R] = σ²T). `mu` entra como localização do log-retorno `(mu-σ²/2)dt`; **não** é drift esperado do preço, pois E[S_T] é infinito com df finito.
- **Semente:** `numpy.default_rng(seed)` por instância; mesma semente reproduz no mesmo NumPy.
- **Tempo:** anos; 5 pregões = 5/252 ano.
- **Retorno:** `risk_summary` usa retorno *simples* (perda `s0 - S_T`). A convenção "−8,6%" é ambígua; abaixo tratamos simples (`S_T/S_0-1 ≤ -8,6%`, log-limiar −8,99%) e log.
- **Lacuna corrigida:** o modo t só funcionava após `calibrate()`. Adicionado argumento opcional `df` (>2) ao construtor, compatível com o uso anterior (+1 teste). Não altera nenhum resultado existente.

## Student-t semanal vs. choques diários agregados (σ=20% a.a., 5 pregões, σ_sem=2,817%)
Probabilidade analítica de queda simples ≤ −8,6% (limiar log −8,99%), localização 0:
- Normal: 0,0706%.
- **t(4) aplicada ao retorno semanal (unit-var):** 0,535% — muito acima de 0,28%.
- **t(4) nos choques diários, somados em 5 dias (MC, 1M, semente 20260101):** 0,307%.
A soma de 5 choques tem cauda mais leve que uma t(4) semanal (CLT parcial). São modelos distintos.

## Reprodução de 0,06% / 0,28%
Com o simulador (`df=4`, 1M trajetórias, 5 passos, σ=20%, semente 20260101, limiar retorno simples ≤ −8,6%):

| mu | Normal | t(4) diário |
|---|---|---|
| 0,05 | 0,069% | 0,297% |
| 0,08 | 0,064% | 0,288% |
Analítico Normal: 0,0706% (mu=0), 0,0656% (0,05), 0,0610% (0,08), 0,0580% (0,10).

**Conclusão:** os números originais são reproduzíveis na ordem de grandeza (≈0,06% e ≈0,28%) com choques **diários** t(4), retorno **simples** e drift positivo (mu ≈ 5–10% a.a.). O `mu` usado originalmente é desconhecido; essa atribuição é inferida, não confirmada. Diferença residual em 1M trajetórias: erro MC ≈ ±0,005 p.p. (Normal ≈ ±0,003 p.p.). Nenhum código foi ajustado para casar os números. A figura original **não** deve ser descrita como "t semanal": isso daria ≈0,54%.

## Limitações
- Memória: `8·(steps+1)·n_sims` bytes, teto 256 MiB; 1M×5 cabe (≈48 MB). Para probabilidades de evento, preferir lotes (será feito na Fase 2).
- t(4) tem curtose infinita; preços exponenciados não têm média finita; nada de erro-padrão baseado em momentos (Fase 2 usa proporções).
- A Student-t i.i.d. não tem agrupamento de volatilidade; isso é testado via GARCH na Fase 3.

## Testes executados
- Linha de base: `pytest -q` → 400 passed. Após a mudança: 401 passed (`.venv`, NumPy 2.5.3, SciPy 1.18.1).
- Script: `research/tail-risk/audit_reproduce.py` → `audit_reproduce.json`.
