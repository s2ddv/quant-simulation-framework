# Fase 3 — Backtest histórico do S&P 500 (^GSPC, 1990–2026): GBM vs Student-t diária vs GARCH(1,1)-t

Substitui, para a pergunta histórica, o recorte SPY 2010–2025 de [`report.md`](report.md). Todos os números vêm de `backtest_gspc/{forecasts.csv,meta.json,metrics.json,metrics_output.txt}`; figuras em `figures_gspc/`.

## 1. Dados e desenho
- **Dados:** `^GSPC`, Yahoo Finance via yfinance, fechamento diário do índice de **preço** (sem dividendos; não é retorno total), 1990-01-02 a 2026-11-20, 9291 sessões, sem lacunas > 7 dias, sem preenchimento. Snapshot congelado com sha256 `7e07e6dd…` em `research/experiments/sp500-gspc-1990/` ([`PROVENANCE.md`](../experiments/sp500-gspc-1990/PROVENANCE.md)). O rótulo `adjusted_close` do snapshot é fixo da biblioteca e não indica ajuste de dividendos. Não é um banco *point-in-time*; o Yahoo pode revisar a história. A última sessão (2026-11-20) é o dado mais recente e mais sujeito a revisão.
- **Sem look-ahead:** janela *expanding*, mínimo de 1260 retornos (≈5 anos). **1990–1994 é só treino**; a primeira previsão é de 1994-12-23 e a última origem é 2026-11-13 (a janela dessa origem termina em 2026-11-20). Todas as 8026 previsões são fora da amostra. Parâmetros reestimados a cada 21 pregões com dados até a origem; o filtro do GARCH é atualizado diariamente. Há teste automatizado de ausência de look-ahead (`tests/test_backtest.py`).
- **Modelos (iguais aos da Fase 3 anterior):** (1) Gaussian GBM (lognormal exata em 5 dias); (2) Student-t i.i.d. diária, 5 dias = soma de 5 choques; (3) GARCH(1,1)-t por máxima verossimilhança em scipy. **383 reestimações do GARCH: 383 convergiram, 0 recorreram a parâmetros anteriores**.
- **Evento:** perda de 5 pregões `−(exp(Σ log-retornos) − 1) ≥ x`. **Janelas sobrepostas** (todas as origens; n = 8026) para frequências/Brier/O÷E; **não sobrepostas** (n ≈ 1606 por deslocamento, 5 deslocamentos) para Kupiec/Christoffersen/ES. ICs de janelas sobrepostas: Wilson com n/5 (aproximação). Razão observado/esperado (O÷E = Σ eventos / Σ probabilidades previstas) e diferenças de Brier têm IC por *moving-block bootstrap* (bloco 20, 2000 réplicas).
- **Regimes, definidos antes dos resultados e sem ajuste posterior:** (a) *ex-ante*: volatilidade realizada dos últimos 21 pregões na origem, anualizada, ≥ 20% (alta) vs < 20% (baixa); (b) *ex-post* (rótulos de calendário, só para leitura): baixa dot-com 2000-03-24→2002-10-09, crise 2007-10-09→2009-03-09, COVID 2020-02-19→2020-04-30, baixa de 2022 2022-01-03→2022-10-12, e o restante.

## 2. Resultados

### Pergunta 1 — Frequência histórica de perdas de 5 pregões ≥ 8,6%
(Sem modelo; sobre todas as janelas de 5 pregões de 1990-01 a 2026-11.)

| Medida | Resultado |
|---|---|
| Janelas sobrepostas | **43 de 9286 = 0,463%** (IC 95% Wilson n/5: 0,255%–0,919%) |
| Janelas não sobrepostas (5 deslocamentos) | 4 a 10 de 1858 (deslocamento 0: 10 = 0,54%, Clopper–Pearson 0,258%–0,988%) |
| Episódios distintos (eventos a ≤10 sessões = 1 episódio) | **14** |
| Fim da janela do 1º evento de cada episódio | 1998-08-31, 2000-04-14, 2001-09-18, 2002-07-22, 2008-10-03, 2008-10-27, 2008-11-11, 2009-02-17, 2011-08-08, 2015-08-24, 2020-02-27, 2022-06-13, 2025-04-04, 2026-11-20 |

Em palavras: cerca de 1 em cada 200 janelas semanais, em 14 episódios em ~36,9 anos (4 deles em 2008–09). Para comparação (≥5%: 183/9286 = 1,97%, 58 episódios; ≥10%: 28/9286 = 0,30%, 13 episódios). O número de episódios independentes (14), não o de janelas (43), limita a precisão. O último episódio (2026-11-20) usa o dado mais recente da base.

### Pergunta 2 — Calibração das probabilidades de eventos extremos (fora da amostra, janelas sobrepostas)
Frequência observada, probabilidade média prevista e razão O÷E (IC 95% bootstrap; **O÷E = 1 é calibração perfeita**).

| Perda ≥ | Observada (IC n/5) | Gaussian prev. · O÷E [IC] | Student-t prev. · O÷E [IC] | GARCH-t prev. · O÷E [IC] |
|---|---|---|---|---|
| 3% | 7,79% (6,58–9,20) | 8,62% · 0,90 [0,76; 1,05] | 7,20% · 1,08 [0,91; 1,26] | 6,85% · 1,14 [0,98; 1,29] |
| 5% | 2,21% (1,57–3,02) | 1,40% · **1,58 [1,13; 2,05]** | 1,86% · 1,18 [0,85; 1,54] | 2,26% · 0,98 [0,74; 1,22] |
| 8% | 0,62% (0,34–1,14) | 0,029% · **21 [10,6; 34]** | 0,39% · 1,60 [0,79; 2,57] | 0,61% · 1,03 [0,56; 1,59] |
| **8,6%** | **0,54%** (0,30–1,06) | 0,011% · **48 [22; 80]** | 0,31% · 1,76 [0,84; 2,90] | 0,49% · 1,10 [0,57; 1,76] |
| 10% | 0,35% (0,17–0,81) | 0,001% · **370 [149; 682]** | 0,18% · 1,90 [0,77; 3,47] | 0,30% · 1,16 [0,54; 1,99] |

Brier score (menor é melhor) e diferença com IC 95% (bloco-bootstrap):

| Perda ≥ | Gaussian | Student-t | GARCH-t | GARCH−Gaussian [IC] | GARCH−Student-t [IC] |
|---|---|---|---|---|---|
| 3% | 0,07271 | 0,07227 | 0,06852 | **−0,00419 [−0,0062; −0,0023]** | **−0,00375 [−0,0059; −0,0018]** |
| 5% | 0,02168 | 0,02164 | 0,02080 | −0,00088 [−0,0022; +0,0002] | −0,00084 [−0,0021; +0,0002] |
| 8,6% | 0,005357 | 0,005336 | 0,005245 | −0,00011 [−0,0005; +0,0002] | −0,00009 [−0,0005; +0,0002] |
| 10% | 0,003489 | 0,003481 | 0,003392 | −0,00010 [−0,0004; +0,0001] | −0,00009 [−0,0004; +0,0001] |

**Resposta:** em estimativa pontual, o **GARCH-t é o mais bem calibrado** (O÷E entre 0,98 e 1,16, todos os ICs contêm 1; menor Brier em todos os limiares). O Gaussian é o pior: em perdas ≥ 5% o IC de O÷E exclui 1 e, em perdas ≥ 8,6%, a frequência real é ≈ 48 vezes a prevista. A **superioridade do GARCH-t sobre a Student-t não está estatisticamente demonstrada nos eventos extremos**: o IC do Brier contém 0 em ≥ 5%, 8,6% e 10%, e os ICs de O÷E de ambos contêm 1 em ≥ 8%. Só em perdas ≥ 3% (evento comum, ~7,8%) a vantagem do GARCH-t é significativa. O Brier é insensível a eventos raros (todos os valores são dominados pela taxa-base), por isso o O÷E é mais informativo na cauda.

### Pergunta 3 — Os modelos de cauda pesada superestimam ou subestimam o risco?
- **Ao longo de toda a amostra:** Student-t **subestima** levemente a cauda (O÷E 1,6–1,9 em ≥ 8%, ICs contêm 1: não rejeita calibração) e o GARCH-t fica próximo de 1. Nenhum deles **superestima** de forma significativa a frequência de perdas ≥ 5%. Em perdas de 3% (evento comum) nenhum IC de O÷E exclui 1: o Gaussian superestima levemente (8,6% previsto contra 7,8% observado; O÷E 0,90), a Student-t e o GARCH-t subestimam levemente (1,08 e 1,14).
- **VaR/ES de 5 pregões** (janelas não sobrepostas, deslocamento 0; valores de p de Kupiec nos 5 deslocamentos entre parênteses):

| Nível | Modelo | Violações sobrepostas (IC n/5) | Não sobrepostas (n=1606) | Kupiec p | Christoffersen p | ES Z2 [IC bootstrap] |
|---|---|---|---|---|---|---|
| 95% | Gaussian | 5,68% (4,64–6,91) | 88 (esperado 80) | 0,385 (0,08–0,39) | **0,001** | 0,36 [0,08; 0,69] |
| 95% | Student-t | 6,08% (5,04–7,39) | 91 | 0,230 (**0,012**–0,23) | **0,005** | 0,26 [−0,00; 0,56] |
| 95% | GARCH-t | 6,32% (5,21–7,59) | 95 | 0,101 (**0,007**–0,10) | 0,846 | 0,30 [0,05; 0,59] |
| 99% | Gaussian | 2,27% (1,62–3,09) | 37 (esperado 16) | **<0,001** (todos <0,001) | 0,056 | **1,99 [1,04; 3,13]** |
| 99% | Student-t | 1,59% (1,11–2,36) | 28 | **0,007** (0,004–0,10) | 0,494 | **0,91 [0,23; 1,72]** |
| 99% | GARCH-t | 1,62% (1,11–2,36) | 27 | **0,012** (0,012–0,063) | 0,346 | **0,86 [0,16; 1,65]** |

**Resposta:** os modelos de cauda pesada **não superestimam** o risco extremo de forma sistemática; **ainda subestimam** o VaR e o ES a 99% de 5 pregões (violações de ~1,6% contra 1% esperado, Z2 > 0 com IC acima de 0), embora menos que o Gaussian (2,3%; ES muito mais subestimado). A rejeição de Kupiec a 99% é robusta para o Gaussian, mas **frágil para Student-t e GARCH-t** (varia com o deslocamento: p de 0,004 a 0,10 e de 0,012 a 0,063). A 95%, o Gaussian cobre bem, enquanto Student-t e GARCH-t têm ligeiro excesso de violações (6,1–6,3% contra 5%; IC n/5 exclui 5%; rejeição de Kupiec em 3 e 4 de 5 deslocamentos). Isso é coerente com a t padronizada ter quantil a 95% menor que o da normal para a mesma variância (o Gaussian parece "melhor" ali por acaso estrutural, não por modelar melhor a cauda). Christoffersen rejeita a independência das violações a 95% em alguns deslocamentos para Gaussian (p 0,000–0,049 em 3 de 5) e Student-t (0,000–0,005 em 2 de 5) e não para o GARCH-t (p ≥ 0,1 nos 5 deslocamentos), o que é consistente com a volatilidade dinâmica capturando o agrupamento. Os p-valores não têm correção para múltiplos testes.

### Pergunta 4 — Crise vs estabilidade
Perda ≥ 5% (eventos comuns o bastante) e VaR 99%. O÷E > 1 indica subestimação, < 1 superestimação. Entre parênteses, eventos observados.

| Regime | Janelas | Perda ≥5% obs. | Gaussian O÷E | Student-t O÷E | GARCH-t O÷E | Viol. VaR99 (G / t / GARCH-t) |
|---|---|---|---|---|---|---|
| Vol. baixa na origem (<20%) | 6250 | 1,02% (64) | 0,73 | 0,55 | 1,43 | 1,25 / 0,67 / 1,79% |
| Vol. alta na origem (≥20%) | 1776 | 6,36% (113) | 4,55 | 3,43 | 0,83 | 5,86 / 4,84 / 1,01% |
| Fora das janelas de crise (ex-post) | 6785 | 1,11% (75) | 0,76 | 0,57 | 0,75 | 1,27 / 0,74 / 1,50% |
| Dentro das janelas de crise (ex-post) | 1241 | 8,22% (102) | 7,66 | 5,40 | 1,24 | 7,74 / 6,29 / 2,26% |
| · baixa 2000–02 | 638 | 4,70% (30) | 6,84 | 4,04 | 1,09 | 6,11 / 4,70 / 2,04% |
| · 2007–09 | 356 | 11,24% (40) | 9,51 | 6,94 | 1,17 | 10,11 / 8,99 / 0,84% |
| · COVID 2020 | 51 | 23,53% (12) | 13,45 | 10,56 | 1,10 | 23,53 / 21,57 / 7,84% |
| · 2022 | 196 | 10,20% (20) | 5,21 | 4,37 | 2,14 | 4,59 / 2,55 / 4,08% |

Para perdas ≥ 8,6% (poucos eventos, leia com cautela): em crises observam-se 32 eventos (2,58%) contra Gaussian 0,006% e Student-t 0,20% (O÷E 459 e 13) e GARCH-t 1,99% (O÷E 1,29); fora das janelas, 11 eventos (0,16%) contra Gaussian 0,012% (O÷E 13), Student-t 0,32% (0,50) e GARCH-t 0,21% (0,77).

**Resposta:** os resultados mudam muito entre regimes (Fig. 5, 6):
- **Crise/volatilidade alta:** Gaussian e Student-t estática **subestimam fortemente** (O÷E 3–13 em perdas ≥ 5%; violações de VaR 99% de 2,5% a 24%), pois usam variância histórica média. O GARCH-t acompanha a volatilidade e mantém O÷E perto de 1 (violações de VaR 99% de 1,0% em vol. alta e de 0,8% em 2007–09), com exceções: COVID 2020 (7,8% de violações) e 2022 (O÷E 2,1). Uma explicação possível, não testada, é a velocidade da queda de 2020 frente à atualização do filtro.
- **Estabilidade/vol. baixa:** Gaussian e Student-t **superestimam** (O÷E 0,55–0,76 em ≥ 5%; a Student-t cobre o VaR 99% com 0,67–0,74% de violações, isto é, é conservadora). Aqui o GARCH-t tem o problema oposto em termos de razão: **subestima** com O÷E 1,43 em ≥ 5% e violações de VaR 99% de 1,8% (9 eventos ≥ 8,6% contra 2,9 esperados, IC largo). A interpretação provável, não testada, é que após períodos calmos o filtro prevê cauda fina e é surpreendido. Fora das janelas ex-post de crise o GARCH-t superestima levemente (O÷E 0,75), o que mostra que os dois recortes de "estabilidade" (ex-ante vs ex-post) não coincidem.
- Em suma, a vantagem da volatilidade dinâmica aparece sobretudo **nos regimes de crise**; os modelos estáticos erram em direções opostas conforme o regime.

## 3. Hipóteses e conclusão
- **H1 (o Gaussian subestima a frequência e a severidade de perdas extremas): suportada**, com forte evidência neste conjunto (1995–2026): O÷E de 21–370 em ≥ 8%; rejeição robusta de Kupiec a 99%; ES a 99% subestimado (Z2 IC acima de 0).
- **Modelos de cauda pesada com volatilidade dinâmica:** o GARCH-t é o mais bem calibrado em estimativa pontual e é decisivamente melhor que os modelos estáticos em crises. **Não se mostrou superioridade estatística sobre a Student-t nos eventos extremos incondicionais** (ICs de Brier e O÷E não separam; só em ≥ 3%).
- **A Student-t estática** corrige a ordem de grandeza da cauda incondicional, mas erra por regime (subestima em crise, superestima em calma) e não elimina a subestimação do VaR/ES a 99%.
- Pela evidência, não se pode afirmar que a vantagem venha das *caudas pesadas* ou da *volatilidade condicional*: falta um GARCH com inovações normais para separar os dois efeitos (não executado).

## 4. Limitações estatísticas
1. **Poucos episódios:** 14 episódios de ≥ 8,6% (4 em 2008–09) em ~37 anos; as janelas sobrepostas (43) não são independentes. ICs n/5 e bloco-bootstrap são aproximações.
2. **Múltiplos testes** sem correção (~30 testes); p entre 0,01 e 0,05, e a dependência do deslocamento, indicam fragilidade.
3. **Z1/Z2 do ES** são diagnósticos com IC bootstrap i.i.d., não testes exatos; com poucas violações são instáveis.
4. **Regimes** pequenos (COVID: 51 janelas, poucos episódios independentes). As janelas de crise ex-post usam datas conhecidas em retrospecto; o corte de 20% foi fixado a priori e não otimizado, mas é arbitrário.
5. **Índice de preço** sem dividendos; dados revisáveis; o último ponto (2026-11-20) cria um episódio na borda da amostra.
6. **Especificação:** t e GARCH-t simétricos (sem assimetria/alavancagem); μ por MLE sob t simétrica tende a exceder a média; sem propagação de incerteza de parâmetros; resolução Monte Carlo ≈ 1e-4 nas previsões do GARCH (40 mil trajetórias por origem), desprezível diante do erro estatístico.
7. 1990–94 só treina; crises de 1987 ficam fora.

## 5. Reprodução
```sh
.venv/bin/python research/tail-risk/fetch_sp500.py            # único passo com rede (Yahoo); não sobrescreve snapshot existente
.venv/bin/python research/tail-risk/run_backtest.py research/experiments/sp500-gspc-1990/GSPC.json research/tail-risk/backtest_gspc   # ~3 min, offline
.venv/bin/python research/tail-risk/analyze_backtest.py research/tail-risk/backtest_gspc research/experiments/sp500-gspc-1990/GSPC.json
.venv/bin/python research/tail-risk/make_figures.py research/tail-risk/backtest_gspc research/tail-risk/figures_gspc
MPLBACKEND=Agg .venv/bin/python -m pytest -q                  # 421 testes, offline
```
Semente 20260101; Python 3.13, NumPy 2.5.3, SciPy 1.18.1 (`backtest_gspc/meta.json`). Resultados exatos pressupõem as mesmas versões.
