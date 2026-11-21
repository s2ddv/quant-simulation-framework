# Risco de cauda no S&P 500: modelos gaussianos vs. caudas pesadas e volatilidade dinâmica

Relatório de pesquisa reprodutível. Todos os números abaixo vêm de arquivos neste diretório (`audit_reproduce.json`, `validation_results.json`, `backtest/*.json|csv`, `figures/`). Detalhes: [`audit.md`](audit.md), [`validation.md`](validation.md).

## 1. Research question
Modelos gaussianos (GBM) subestimam a frequência e a severidade de perdas extremas de 5 pregões em comparação com Student-t (choques diários) e GARCH(1,1) com inovações Student-t?

## 2. Hypothesis
**H1:** a frequência observada de perdas extremas excede a probabilidade prevista pelo GBM, e o ES/VaR de 99% do GBM subestima as perdas, mais do que nos modelos de cauda pesada. Hipótese a testar, não conclusão prévia. **H2 (secundária):** a volatilidade dinâmica (GARCH-t) melhora a previsão sobre a Student-t estática.

## 3. Methodology
- **Modelos** (retorno de 5 pregões, `R = exp(Σ r_d) − 1`, perda `= −R`):
  1. *Gaussian GBM:* log-retorno diário N(m, s²) i.i.d.; `m`, `s` pela média/desvio amostral expanding; 5 dias → lognormal exata (fórmulas fechadas para P, VaR, ES).
  2. *Student-t diária:* `t.fit` (ν, loc, scale) nos log-retornos diários; 5 dias = soma de 5 choques i.i.d. (Monte Carlo, 400 mil amostras por reestimação). Não é t aplicada ao retorno semanal (ver `audit.md`: t semanal dá ≈0,54% no evento −8,6% contra ≈0,31% para a soma diária).
  3. *GARCH(1,1)-t:* `r_t = μ + √h_t·z_t`, `h_t = ω + α e²_{t−1} + β h_{t−1}`, `z` t padronizada (variância 1). Estimado por máxima verossimilhança com `scipy` (SLSQP, `src/backtest/garch.py`); 40 mil trajetórias por origem a partir do estado filtrado.
- **Sem look-ahead:** janela *expanding* (≥1260 retornos ≈ 5 anos), reestimação a cada 21 pregões, previsão em cada origem com dados até a origem; a variância do GARCH é filtrada diariamente com parâmetros congelados. Teste automatizado verifica que alterar dados futuros não muda previsões (`tests/test_backtest.py`). Expanding foi escolhido porque a cauda (ν) precisa de amostras longas com crises; rolling descartaria crises antigas.
- **Janelas:** sobrepostas (todas as origens, n=2759, descritivas) e não sobrepostas (1 a cada 5, n=552, 5 deslocamentos). Testes formais (Kupiec, Christoffersen) usam só as **não sobrepostas**. ICs de frequência com janelas sobrepostas usam n/5 como tamanho efetivo (aproximação).
- **Métricas:** frequência de perda >3/5/8/10% vs. probabilidade prevista; VaR/ES 95% e 99%; Kupiec (cobertura), Christoffersen (independência); Brier; diferença de Brier por *moving-block bootstrap* (bloco 20). **ES:** estatísticas tipo Acerbi–Székely Z1/Z2 com IC bootstrap i.i.d. Limitação: é um diagnóstico, não um teste exato; com poucas violações é instável.

## 4. Data
SPY (ETF do S&P 500), preço de fechamento ajustado (dividendos reinvestidos), diário, 2010-01-04 a 2025-12-31, 4024 pregões, fonte yfinance, snapshot congelado `research/experiments/frozen-network-v1/SPY.json` (sha256 `9eb06637…`, obtido em 2026-11-01), verificado na leitura. Sem preenchimento nem exclusão. Escolha do usuário: usar apenas o SPY. **Não** é o índice ^GSPC, **não** cobre 1990–2009 (nem 1987, 2000–02 ou 2008) e é uma base atual de preços ajustados, não *point-in-time*. Primeira previsão: 2015-01-06; última origem: 2025-12-23 (132 reestimações).

## 5. Mathematical assumptions
Tempo em dias úteis (252/ano na calibração do GBM do simulador); retornos diários i.i.d. (Gaussian, Student-t) ou GARCH; ν>2 (variância finita); distribuições simétricas (sem assimetria); simulação usa `mu` como localização do log-retorno, não o drift esperado do preço (com t de ν finito E[S_T] é infinita). Medida física (histórica), não risco-neutra. Sem custos nem dividendos separados.

## 6. Experimental design
Sementes: `SeedSequence(20260101, spawn_key=(modelo, origem))`. Configuração completa em `backtest/meta.json` (inclui parâmetros por reestimação, convergência e ambiente: Python 3.13, NumPy 2.5.3). Convergência do GARCH: 132 de 132 reestimações convergiram, 0 recorreram a parâmetros anteriores; α+β entre 0,968 e 0,998; ν do GARCH ≈ 4,9–5,3; df da t i.i.d. ≈ 2,7–2,9 (valor muito baixo: reflete a mistura de volatilidades que o modelo i.i.d. não separa).

## 7. Results (SPY, previsões fora da amostra 2015–2025)
### 7.1 Simulação (não histórica)
Com variância igual, Gaussian, Student-t e GARCH-t divergem na cauda (Fig. 1): para perdas ≳ 10% a probabilidade gaussiana cai ordens de grandeza abaixo; nas perdas moderadas (≲ 7%) a Gaussian é maior que a Student-t. Mostra o mecanismo, não evidência empírica. Validação do simulador: `validation.md`.

### 7.2 Frequência observada vs. prevista (média das previsões) — Fig. 2
| Perda 5d | Observada (sobreposta) | IC 95% (n/5) | Gaussian | Student-t | GARCH-t |
|---|---|---|---|---|---|
| > 3% | 6,60% (182) | [4,75; 8,90] | 7,61% | 6,09% | 5,17% |
| > 5% | 2,21% (61) | [1,25; 3,76] | 1,01% | 1,73% | 1,81% |
| > 8% | 0,73% (20) | [0,28; 1,85] | 0,013% | 0,44% | 0,56% |
| > 10% | 0,47% (13) | [0,19; 1,59] | ≈0,000% | 0,23% | 0,31% |

Janelas não sobrepostas com perda > 8%: 2 a 6 de 552 (conforme o deslocamento); > 10%: 1 a 4. Os eventos de >8% concentram-se em **5–6 episódios** (ago/2015, fev/2018, fev–mar/2020, mai–jun/2022, mar–abr/2025). O número efetivo de episódios independentes é pequeno.

### 7.3 Brier score (menor é melhor; sobreposto)
| Evento | Gaussian | Student-t | GARCH-t |
|---|---|---|---|
| > 3% | 0,06186 | 0,06163 | 0,05948 |
| > 5% | 0,02183 | 0,02166 | 0,02123 |
| > 8% | 0,00725 | 0,00720 | 0,00691 |
| > 10% | 0,00471 | 0,00470 | 0,00447 |

Diferença de Brier (modelo − Gaussian) com IC 95% por bloco-bootstrap, p.ex.: GARCH-t >8%: −0,00034 [−0,00152; +0,00035]; GARCH-t >3%: −0,00238 [−0,00614; +0,00019]; Student-t >5%: −0,00017 [−0,00038; +0,000005] (limítrofe); Student-t >8%: −0,000046 [−0,000117; +0,000009]. **Todos os intervalos contêm zero** (o mais próximo de excluí-lo é o da Student-t >5%). O GARCH-t tem o menor Brier em todos os eventos, mas **sem significância estatística**.

### 7.4 VaR/ES e testes (janelas não sobrepostas, deslocamento 0, n=552)
| Nível / modelo | Violações (esperado) | Kupiec p | Christoffersen p | ES Z2 [IC 95%] |
|---|---|---|---|---|
| 95% Gaussian | 24 (27,6) | 0,473 | 0,018 | 0,24 [−0,26; 0,82] |
| 95% Student-t | 28 | 0,938 | 0,056 | 0,14 [−0,30; 0,63] |
| 95% GARCH-t | 36 | 0,116 | 0,293 | 0,49 [−0,01; 1,07] |
| 99% Gaussian | 13 (5,5) | **0,006** | 0,304 | **2,34 [0,63; 4,35]** |
| 99% Student-t | 6 | 0,840 | 0,048 | 0,34 [−0,58; 1,49] |
| 99% GARCH-t | 11 | **0,039** | 0,503 | 1,20 [−0,05; 2,81] |

Taxa de violação sobreposta (descritiva): 95%: 4,8% / 5,3% / 7,4%; 99%: 2,2% / 1,3% / 1,8% (Gaussian / t / GARCH-t). Fig. 3 compara VaR/ES médios previstos com as perdas realizadas. Z2>0 indica ES subestimado.

## 8. Statistical limitations
- **Poucos eventos e poucos episódios:** 5–6 episódios por trás de >8%. Os ICs são largos; as probabilidades de 8% e 10% mal discriminam t e GARCH-t.
- **Dependência:** janelas sobrepostas não são independentes; n/5 é aproximação. Kupiec/Christoffersen só nas não sobrepostas, e os p-valores variam com o deslocamento (`metrics.json`, `kupiec_p_by_offset`).
- **Múltiplos testes:** ~24 testes sem correção; resultados com p entre 0,01 e 0,05 (GARCH-t 99%, Christoffersen da t) são frágeis. Z1/Z2 são diagnósticos com IC bootstrap (veja §3).
- **Erro Monte Carlo** nas previsões t e GARCH-t (≈ 1e-4 de resolução em 40 mil trajetórias por origem para GARCH, 4e5 para t); desprezível frente ao erro estatístico dos eventos raros, mas existe.
- **Dados:** um ativo, 2010–2025, SPY ajustado (não o índice), sem crises de 1987/2000/2008 nem janela de avaliação pré-2015. A amostra de treino inicial inclui o regime pós-2010.
- **Especificação:** distribuições simétricas e localização estimada por MLE (nos modelos t o μ tende a exceder a média por assimetria negativa), o que reduz as probabilidades de queda; o df de ≈2,7 da t estática é artefato da mistura de volatilidades.
- **Incerteza de parâmetros** não é propagada nas previsões; os ICs de Monte Carlo (Fig. 4) cobrem apenas amostragem.

## 9. Discussion
1. Em perdas ≥ 8% e no VaR/ES 99%, o Gaussian subestimou a cauda: previu ≈0,01% para eventos observados em ≈0,7% das janelas (IC não contém a previsão), rejeitou a cobertura 99% (13 violações vs 5,5 esperadas, p=0,006) e o ES ficou subestimado (Z2 IC acima de 0). Na parte central a conclusão **inverte**: no limiar de 3% o Gaussian superestimou (7,6% vs 6,6%), efeito esperado de uma Student-t de mesma variância, mas também reflexo da janela expanding, que carrega volatilidade antiga alta.
2. Student-t e GARCH-t tiveram probabilidades de cauda compatíveis com a frequência observada (todas dentro dos ICs a 8% e 10%) e ES sem rejeição. O GARCH-t teve menor Brier em todos os eventos, mas sem significância. No VaR 99% a t estática (6 violações) ficou mais perto do esperado que o GARCH-t (11, p=0,039): **a volatilidade dinâmica não mostrou vantagem sobre a t estática** nesta amostra. Uma causa possível, não testada, é a localização e a variância condicional baixas em períodos calmos.
3. Ligação com Taleb: o resultado é coerente com a ideia de que a calibração gaussiana de cauda é frágil (Extremistão), porém aqui isso é medido em um único ativo e com 5–6 episódios; não demonstra a tese geral.

## 10. Conclusion
- **H1 (Gaussian subestima a cauda): suportada** para este conjunto (SPY, 2015–2025) nos limiares ≥ 8%, no VaR 99% e no ES 99%, com ressalvas: poucos episódios, um ativo, sem as crises de 1987/2000/2008. Na faixa central (3%) a evidência vai na direção oposta (o Gaussian superestima).
- **H2 (vol dinâmica melhora sobre a t estática): inconclusiva.** Brier favorável ao GARCH-t sem significância; cobertura do VaR 99% pior que a da t estática.
- Não se declara superioridade estatística entre Student-t e GARCH-t.

## 11. Reproduction
```sh
python -m venv .venv && .venv/bin/pip install -e '.[dev]'   # numpy, pandas, scipy, matplotlib, yfinance
MPLBACKEND=Agg .venv/bin/python -m pytest -q                 # 420 passed (offline)
.venv/bin/python research/tail-risk/audit_reproduce.py       # fase 1
.venv/bin/python research/tail-risk/validation_experiments.py   # fase 2 (~7 s)
.venv/bin/python research/tail-risk/run_backtest.py          # fase 3 (~50 s, offline, usa o snapshot congelado)
.venv/bin/python research/tail-risk/analyze_backtest.py
.venv/bin/python research/tail-risk/make_figures.py          # fase 4
```
Código: `src/validation/tail_probability.py`, `src/backtest/{garch,engine,metrics}.py`. A reprodução de números exatos pressupõe a mesma versão do NumPy.
