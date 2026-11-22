# Ablação: GARCH(1,1) com inovações normais (^GSPC, 1995–2026, fora da amostra)

Pergunta: a vantagem do GARCH-t vem da **volatilidade condicional** ou das **caudas pesadas** das inovações? O GARCH-normal usa exatamente a mesma recursão de variância, os mesmos dados, origens (8026), agenda de reestimação (383 reestimações, todas convergiram), semente e 40 mil trajetórias por origem do GARCH-t; só as inovações mudam (normal em vez de t). Código: `run_garch_normal.py`, `analyze_garch_normal.py`, `src/backtest/garch.py` (`fit_garch_normal`), testes em `tests/test_backtest.py`. Saídas: `backtest_gspc/{forecasts_garch_n.csv,meta_garch_n.json,ablation_garch_normal.json,ablation_output.txt}`.

Quatro modelos formam um desenho 2×2:

| | Cauda fina (normal) | Cauda pesada (t) |
|---|---|---|
| **Variância constante** | Gaussian GBM | Student-t diária |
| **Variância condicional (GARCH)** | **GARCH-normal** | GARCH-t |

## Resultados

**Calibração de eventos (O÷E: observado/esperado; 1 = calibrado; IC 95% bloco-bootstrap)**

| Perda ≥ | Obs. | Gaussian | Student-t | **GARCH-normal** | GARCH-t |
|---|---|---|---|---|---|
| 3% | 625 | 0,90 [0,76; 1,05] | 1,08 [0,91; 1,26] | 1,10 [0,95; 1,24] | 1,14 [0,98; 1,29] |
| 5% | 177 | 1,58 [1,13; 2,05] | 1,18 [0,85; 1,54] | 1,01 [0,77; 1,26] | 0,98 [0,74; 1,22] |
| 8% | 50 | 21,3 [10,6; 34,4] | 1,60 [0,79; 2,57] | 1,16 [0,65; 1,83] | 1,03 [0,56; 1,59] |
| 8,6% | 43 | 47,6 [22,2; 79,8] | 1,76 [0,84; 2,90] | 1,26 [0,66; 2,06] | 1,10 [0,57; 1,76] |
| 10% | 28 | 370 [149; 682] | 1,90 [0,77; 3,47] | 1,35 [0,65; 2,35] | 1,16 [0,54; 1,99] |

**Diferença de Brier (primeiro − segundo; negativo favorece o primeiro; IC 95%)**

| Perda ≥ | GARCH-normal − Gaussian | GARCH-t − GARCH-normal |
|---|---|---|
| 3% | **−0,00421 [−0,00625; −0,00228]** | +0,00002 [−0,00011; +0,00015] |
| 5% | −0,00088 [−0,00222; +0,00025] | +0,00000 [−0,00006; +0,00007] |
| 8,6% | −0,00013 [−0,00056; +0,00019] | +0,00002 [−0,00001; +0,00005] |
| 10% | −0,00011 [−0,00040; +0,00011] | +0,00002 [+0,00000; +0,00004] |

(GARCH-normal − Student-t em 3%: −0,00376 [−0,00600; −0,00170].)

**VaR/ES de 5 pregões (violações sobrepostas, IC n/5; p de Kupiec nos 5 deslocamentos não sobrepostos)**

| Nível | Modelo | Violações | Kupiec p (min–max) | Christoffersen p (min–max) | ES Z2 [IC] |
|---|---|---|---|---|---|
| 99% | Student-t | 1,60% (1,11–2,36) | 0,004–0,102 | 0,00–0,49 | 0,91 [0,23; 1,72] |
| 99% | **GARCH-normal** | 1,78% (1,26–2,58) | 0,001–0,038 (**5 de 5 rejeitam**) | 0,13–0,63 | 1,11 [0,36; 1,95] |
| 99% | GARCH-t | 1,62% (1,11–2,36) | 0,012–0,063 (4 de 5 rejeitam) | 0,08–0,37 | 0,86 [0,16; 1,65] |
| 95% | **GARCH-normal** | 5,99% (4,92–7,25) | 0,022–0,385 (3 de 5 rejeitam) | 0,33–0,91 | 0,25 [−0,00; 0,54] |
| 95% | GARCH-t | 6,32% (5,21–7,59) | 0,007–0,101 (4 de 5 rejeitam) | 0,11–0,85 | 0,30 [0,05; 0,59] |

**Regimes (O÷E em perdas ≥ 8,6%; violações do VaR 99%)**

| Regime | Eventos ≥8,6% | GARCH-normal | GARCH-t | Viol. VaR99: GARCH-n / GARCH-t |
|---|---|---|---|---|
| Vol. alta na origem | 34 | 1,03 | 0,94 | 1,30% / 1,01% |
| Vol. baixa na origem | 9 | **8,75** | 3,14 | 1,92% / 1,79% |
| Dentro das crises (ex-post) | 32 | 1,38 | 1,29 | 2,66% / 2,26% |
| Fora das crises (ex-post) | 11 | 1,00 | 0,77 | 1,62% / 1,50% |

## Interpretação

1. **O ganho principal vem da volatilidade condicional, não das caudas pesadas.** O GARCH-normal já iguala o GARCH-t em calibração e Brier: a diferença de Brier entre os dois é ≈ 0,00002 em todos os limiares (ICs de GARCH-t − GARCH-normal contêm zero, exceto ≥10%, onde o GARCH-normal fica *marginalmente* melhor, por 0,6%). Contra o Gaussian estático, o GARCH-normal tem Brier significativamente melhor em 3% (e O÷E próximo de 1 em ≥5%), o mesmo que o GARCH-t.
2. **Caudas pesadas nas inovações dão um refinamento pequeno e estatisticamente inconclusivo.** Em perdas ≥ 8,6% o GARCH-t fica um pouco mais perto de O÷E = 1 (1,10 vs 1,26; ICs sobrepostos), com um pouco menos de violações do VaR 99% (1,62% vs 1,78%) e ES Z2 menor (0,86 vs 1,11, ICs sobrepostos). Nenhuma dessas diferenças é distinguível do ruído.
3. **Onde as caudas pesadas parecem importar: "surpresas" saindo de calmaria.** Com volatilidade baixa na origem, o GARCH-normal subestima bem mais os eventos ≥ 8,6% (O÷E 8,75 vs 3,14; apenas 9 eventos, IC não calculado para esse subgrupo). Esta é a única diferença visível, e é frágil.
4. **Por que a diferença é pequena (explicação provável, não testada aqui):** o horizonte é de 5 dias, e a soma de 5 choques t(ν≈5) já fica próxima da normal; a cauda do retorno semanal vem sobretudo da mistura de volatilidades (agrupamento), que o GARCH-normal também captura.
5. **Ainda falha no VaR 99%:** o GARCH-normal é rejeitado por Kupiec nos 5 deslocamentos; nenhum dos modelos cobre 99% perfeitamente.

## Consequência para as conclusões
A frase do relatório "não se pode dizer se a vantagem vem das caudas pesadas ou da volatilidade condicional" fica resolvida **nesta amostra**: o grosso vem da volatilidade condicional. Para o post/tese: o "Extremistão" de 5 pregões no S&P 500 aparece, em grande parte, como **agrupamento de volatilidade**, e a cauda pesada das inovações diárias acrescenta pouco depois disso. Isso não refuta a leitura de Taleb (o modelo gaussiano com variância constante falha, e o GARCH também é um modelo, falível); apenas refina o mecanismo.

## Limitações
- Mesmas do relatório principal: 14 episódios independentes de ≥ 8,6%, janelas sobrepostas (ICs aproximados), múltiplos testes sem correção, índice de preço, dado mais recente sujeito a revisão.
- Subgrupos pequenos (9 eventos em vol. baixa).
- Um único horizonte (5 dias) e uma única especificação simétrica; GARCH assimétrico (GJR/EGARCH) não foi testado.
- Resolução Monte Carlo ≈ 1e-4 nas probabilidades de cauda (40 mil trajetórias por origem).

## Reprodução
```sh
.venv/bin/python research/tail-risk/run_garch_normal.py        # ~40 s, offline; requer backtest_gspc/forecasts.csv
.venv/bin/python research/tail-risk/analyze_garch_normal.py
```
