---
registro_id: quant-simulation-framework-e2-reducao-variancia-2026-11-23
projeto: quant-simulation-framework
data: 2026-11-23
tipo: nota-de-epico
epico: E2
branch: claude/serene-clarke-qqq11r
---

# E2 — Redução de variância

Especificação: [plano de expansão, §E2](../../docs/plano-expansao.md).
Pré-requisito: [E1 — intervalos de risco](2026-11-22-e1-intervalos-risco.md).

A equipe de `AGENTS.md` executou o épico em **delegação por instruções**, sem
perfis nativos, com no máximo dois especialistas ativos. O fluxo de cada tarefa:

1. `pesquisador_quant` deu um parecer prévio, somente leitura.
2. `backend_quant` implementou.
3. `testador_quant` e `revisor_quant` atuaram em paralelo sobre um commit estável.
4. O `corretor` tratou apenas falhas confirmadas, cada uma com nova revisão.

`src/simulation/gbm.py` teve um único escritor por vez, de E2.1 até E2.3.

## Objetivo e hipótese

**Objetivo:** reduzir o erro Monte Carlo de estimadores de **valor esperado**
(média terminal e esperanças de payoffs sob a medida física P) sem mudar a lei
simulada. Cada técnica deve reportar o SE correto para o estimador que usa e o
fator de redução de variância (VRF).

**Hipóteses verificadas:**
- **Antitéticas:** VRF teórico da média terminal ao mesmo custo
  `1/(1 − e^{−σ²T})`, com `Cov(S+,S−) = S0²e^{2(μ−σ²/2)T} − (E S_T)²`.
- **Variável de controle:** com `X = S_T`, o VRF é `≈ 1/(1−ρ²)`.
- **QMC randomizado:** ganho grande para integrandos suaves de baixa dimensão
  efetiva.

## Implementação

| Tarefa | Arquivos | Commits |
| --- | --- | --- |
| E2.1 antitéticas | `src/simulation/gbm.py`, `src/simulation/variance_reduction.py` | `cb73857`; testes `5459f33` |
| E2.2 variável de controle | `src/simulation/variance_reduction.py` | `30e1a12`; testes `06fd821`; correção `3f02cae` |
| E2.3 QMC (Sobol + ponte browniana) | `gbm.py`, `variance_reduction.py` | `4458ea6`; testes `e8b1e36`; correções `427cd2b`, `e1c592a` |
| E2.4 convergência e comparação | `src/validation/statistics.py` (só adições), docstrings | `065bee1`; testes `5db7297`; correções `4717e1f` |

**Contratos preservados:**
- O padrão continua `variance_reduction="none"`, bit a bit idêntico ao amostrador
  anterior. Verificado por revisão independente em 39 arrays (paths normais e t,
  terminal com vários `batch_size`, wrappers e ramos de log-espaço).
- A matriz de trajetórias continua `(steps + 1, n_sims)`, com a linha 0 igual a `s0`.
- `risk_summary`, `convergence_analysis` e os testes existentes não mudaram.

### Decisões e desvios registrados

- **Tipo `VarianceReduction`:** `Literal["none","antithetic","qmc"]`. A técnica
  `"control"` é um estimador (`control_variate_mean`), não um modo de amostragem.
- **Antitéticas:**
  - `n_sims` ímpar → `ValueError`;
  - colunas na ordem `[originais..., espelhos...]`;
  - `t_student` com antitéticas → `ValueError`;
  - SE calculado por pares (`antithetic_mean`).
- **Controle:**
  - β da mesma amostra (viés O(1/N), SE com ddof=2), β de um prefixo piloto
    independente (sem viés) ou β fixo;
  - `vrf` é a razão obtida `Var(Y)/Var(resíduo)`;
  - valores indefinidos viram `None`;
  - combinação com antitéticas não é suportada.
- **QMC:**
  - `n_sims = R·N`, com N potência de 2 (senão `ValueError`), e colunas em blocos
    por scramble;
  - `Sobol(d, scramble=True, bits=30, rng=Generator)`. O plano dizia `seed=`; o
    SciPy 1.18 usa `rng=`;
  - `u` recortado para `[2^-53, 1−2^-53]`;
  - terminal com d = 1; trajetórias com d = `steps` e ponte browniana em ordem BFS;
  - `steps` maior que `Sobol.MAXDIM` (21201 no SciPy instalado) → `ValueError`.
- **Fluxos RNG:**
  - QMC usa `SeedSequence(seed, spawn_key=(crc32("qmc_scramble"), call, r))`, em
    que `call` conta as chamadas QMC da instância. O QMC não consome `self.rng`.
  - As antitéticas usam o mesmo `self.rng`.
  - O NumPy não aceita `spawn_key` com string, por isso foram usadas constantes
    inteiras documentadas.
- **E2.4:**
  - `convergence_with_estimator(units, estimator, ...)` com grades `geometric` e
    `power2` e `n_evaluations` para comparar técnicas ao mesmo custo;
  - parâmetro `min_units` acrescentado, porque o controle com β da mesma amostra
    exige N ≥ 3;
  - adaptadores `antithetic_units` (prefixos de pares) e `qmc_units` (linhas
    ponto × scramble).

## Metodologia

Toda verificação usou sementes fixas e referências analíticas independentes:
momentos lognormais, Black–Scholes com drift μ sob P e quadratura. As
tolerâncias seguem a §2.5: 4·SE para médias, sem exigir queda monotônica do
erro.

- **Antitéticas:** o VRF foi comparado com o teórico a ±20%, com σ ≤ 0,5 e
  m ≥ 2^14. O parecer mostrou que σ = 1 com m = 4096 falha em 18% das sementes.
- **Controle:** o SE foi conferido contra uma regressão OLS com intercepto.
- **QMC:**
  - erro RMS em 50 sementes contra o MC simples no mesmo custo `N·R`;
  - ponte verificada de forma determinística (`A·Aᵀ = min(t_i,t_j)` com `Z = I`);
  - estratificação de rede por bloco;
  - mutantes detectados pela suíte.

## Resultados medidos

Os números abaixo vêm dos testes e scripts executados. A semente e o N estão em
cada linha. São medições de uma configuração, não garantias.

**Antitéticas** (média terminal, S0 = 100, μ = 0,05, T = 1, m = 2^14):

| σ | semente | VRF medido | VRF teórico |
| --- | --- | --- | --- |
| 0,2 | 404 | 25,76 | 25,50 |
| 0,5 | 405 | 4,408 | 4,521 |

- **Straddle `|S_T − K|`** com K na mediana, σ = 0,2, semente 606: VRF 0,540
  (quadratura 0,540; limite teórico 0,5). Aqui a técnica **piora**.

**Variável de controle** (call K = 100 sob P, S0 = 100, μ = 0,05, σ = 0,2, T = 1,
N = 2·10^5, semente 2202; valor analítico 10,98640):

| modo | estimativa | SE | VRF |
| --- | --- | --- | --- |
| mesma amostra | 10,99654 | 0,01319 | 6,88 (teórico 6,88) |
| piloto f = 0,1 | 10,99830 | 0,01391 | 6,85 |
| MC simples | — | 0,03461 | 1 |

- **Viés com β da mesma amostra:** N·viés fica entre −6 e −9, ou seja, O(1/N).
  Com o piloto, o viés não é detectável.
- **Casos sem ganho:** call K = 160 com VRF 1,11; put K = 60 com VRF 1,01.

**QMC** (R = 16, N = 2^12, 50 sementes 2400–2449, S0 = 100, μ = 0,07, σ = 0,3,
T = 1; ganho = razão dos MSE contra o MC simples com 65 536 avaliações):

| integrando | ganho de variância |
| --- | --- |
| média terminal | ≈ 9 100 (outra configuração/estimador: ≈ 5 900; incerteza ≈ ±25%) |
| `paths[-1]`, steps = 12 | ≈ 7 460 |
| média aritmética, steps = 12 | ≈ 13 300 |
| média aritmética, steps = 252, com ponte | ≈ 14 000 |
| média aritmética, steps = 252, sem ponte | ≈ 580 |
| barreira `1{min S < 80}`, steps = 252 | ≈ 13 |

- **Calibração do SE entre scrambles:** SE/RMS ≈ 0,92–1,03.

**Comparação ao mesmo custo** (`convergence_with_estimator`, 2^16 avaliações,
S0 = 100, μ = 0,05, σ = 0,2, T = 1, semente 20241; alvo E[S_T] = 105,12711):

| técnica | estimativa | SE | erro abs |
| --- | --- | --- | --- |
| none | 105,12536 | 8,30e-2 | 1,8e-3 |
| antitética (pares) | 105,13700 | 1,65e-2 | 9,9e-3 |
| QMC (R = 16) | 105,12896 | 1,28e-3 | 1,9e-3 |

## Testes realmente executados

- **Arquivos novos:**
  - `tests/test_antithetic.py` (73);
  - `tests/test_control_variate.py` (103) e `tests/test_control_variate_regression.py`;
  - `tests/test_qmc.py` (74), `tests/test_qmc_memory_regression.py` e
    `tests/test_brownian_bridge_regression.py`;
  - `tests/test_convergence_estimator.py` (85 casos) e
    `tests/test_convergence_estimator_regression.py`.
  - Os testes com muitas réplicas têm versão rápida e versão `@pytest.mark.slow`.
- **Suíte completa final:** `MPLBACKEND=Agg .venv/bin/python -m pytest -q` →
  2432 passed, 51 subtests passed. A base no início do E2 tinha 1759; o restante
  inclui trabalho de E3/E4 de outras sessões na mesma branch.
- **Lint e tipos:** `python -m ruff check` → All checks passed;
  `python -m mypy` → no issues in 78 source files.
- **Rede:** nenhum teste com rede. Os testes offline não comprovam a
  disponibilidade do provedor de dados.

## Falhas encontradas e corrigidas (com revisão posterior)

- **E2.2, P2:** com β fixo enorme, o resíduo estourava para ±inf e o resultado
  saía com SE = 0. Agora levanta `ValueError` (`3f02cae`).
- **E2.3, P2:** o orçamento de memória do QMC em trajetórias ignorava as matrizes
  LMS do Sobol. Com `steps = 21201` o pico chegava a 320 MiB, acima do limite de
  256 MiB. Agora conta `qmc_sobol_bytes` e rejeita antes de alocar (`427cd2b`).
  Uma revisão externa (`docs/revisoes/2026-11-22-revisao-e1-e3.md`) encontrou o
  mesmo problema de forma independente.
- **E2.3, P3:** o desvio condicional da ponte sofria underflow com horizonte
  ≤ 1e-160 (`e1c592a`).
- **E2.4, P3:**
  - `mean_estimator` devolvia `inf` silenciosamente;
  - `counts` aceitava `bool`;
  - a docstring do controle superestimava o efeito do piloto.
  - Corrigidos em `4717e1f`.

## Quando cada técnica não ajuda

- **Antitéticas:** payoff par ou não monótono em Z. O straddle chega a VRF ≈ 0,54
  (limite 0,5). Com σ√T grande, o VRF tende a 1.
- **Controle:**
  - pouca correlação com `S_T` (put de cauda profunda, VRF 1,01);
  - serve só para esperanças, não para quantis, VaR ou CVaR;
  - com N pequeno, o IC normal sub-cobre (0,77 com N = 10), mesmo com piloto ou β
    fixo.
- **QMC:**
  - integrandos descontínuos ou de alta dimensão efetiva (barreira: ganho ≈ 13);
  - grade sem ponte;
  - prefixos que não são potência de 2;
  - IC t_{R−1} com R = 16 sub-cobre (≈ 0,89–0,93 a 95%);
  - custo alto em d ≈ 21201.
- **Todas as técnicas:** os ICs i.i.d. do E1 (Wilson, VaR por estatísticas de
  ordem, bootstrap do CVaR) **não valem** para amostras antitéticas ou QMC. Por
  isso `results.risk_intervals` dos relatórios continua calculado só em
  `"none"`.

## Limitações

- Os ganhos são medições numa configuração GBM sob P. Não são previsões nem
  garantias para outros integrandos ou parâmetros.
- O SE capta apenas o erro Monte Carlo com μ e σ fixos. Não inclui incerteza de
  calibração nem erro de modelo.
- A igualdade bit a bit entre versões de NumPy e SciPy não é prometida.
- As constantes de memória do Sobol dependem do SciPy 1.18.
- A CLI e os relatórios ainda não expõem `variance_reduction`. O gráfico
  comparativo de convergência fica para o E9.4.

## Pendências e próximos passos

- **E9 (relatórios):**
  - serializar o VRF teórico `inf` (σ√T = 0) como `null` com `vrf_defined`
    (P3 do E2.1, ainda aberto);
  - suprimir ou adaptar `risk_intervals` se a CLI expuser antitéticas ou QMC.
- **Opcionais:**
  - P3 de cancelamento numérico no controle com deslocamentos ~1e12;
  - VRF enorme em relação afim exata;
  - custo de import de `scipy.stats.qmc` em `validation.statistics`;
  - texto da taxa de falso alarme em `tests/test_qmc.py`.
- **Limpeza:** há dois arquivos de teste do E1.4 com sobreposição
  (`tests/test_report_intervals.py` e `tests/test_report_risk_intervals.py`).
- **E3.3+:** o E2.1 e o E2.2 estão concluídos e revisados, então a precificação
  europeia por Monte Carlo pode usar o controle `e^{-rT}S_T`, com média
  `S0e^{-qT}` sob Q.
- **Obsidian:** o vault `AI Quant` não está acessível nesta sessão em nuvem. Esta
  nota é o rascunho a copiar.
