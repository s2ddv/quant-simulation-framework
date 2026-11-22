# Plano de expansão do simulador — especificação para a equipe de agentes

> **Natureza deste documento.** É uma especificação de trabalho, **não** uma
> autorização de implementação nem um relatório de resultados. Cada tarefa só
> começa quando o usuário (ou o agente principal, com autorização do usuário)
> a liberar. Nenhum número aqui é resultado experimental; tolerâncias e
> parâmetros citados são valores de projeto a confirmar nos testes.
>
> Base de referência: branch `claude/serene-clarke-qqq11r`, commit `ad2d68c`
> (2026-11-21). Antes de iniciar qualquer tarefa, confirme com `git log` e
> leitura do código que os pontos de partida descritos em §1 ainda valem.

---

## Sumário

0. [Como usar este documento](#0-como-usar-este-documento)
1. [Estado atual (ponto de partida verificado)](#1-estado-atual-ponto-de-partida-verificado)
2. [Contratos globais e regras invariantes](#2-contratos-globais-e-regras-invariantes)
3. [Mapa de épicos, dependências e ordem](#3-mapa-de-épicos-dependências-e-ordem)
4. [E1 — Incerteza Monte Carlo das métricas de risco](#e1--incerteza-monte-carlo-das-métricas-de-risco)
5. [E2 — Redução de variância](#e2--redução-de-variância)
6. [E3 — Precificação de derivativos](#e3--precificação-de-derivativos)
7. [E4 — Modelos com saltos (Merton, Kou)](#e4--modelos-com-saltos-merton-kou)
8. [E5 — Volatilidade estocástica (Heston) e GARCH como simulador](#e5--volatilidade-estocástica-heston-e-garch-como-simulador)
9. [E6 — Multiativos correlacionados e carteiras](#e6--multiativos-correlacionados-e-carteiras)
10. [E7 — Calibração, diagnóstico e incerteza de parâmetros](#e7--calibração-diagnóstico-e-incerteza-de-parâmetros)
11. [E8 — Desempenho e memória](#e8--desempenho-e-memória)
12. [E9 — CLI, relatórios e visualização](#e9--cli-relatórios-e-visualização)
13. [E10 — Engenharia, CI e documentação](#e10--engenharia-ci-e-documentação)
14. [Matriz de propriedade de arquivos](#14-matriz-de-propriedade-de-arquivos)
15. [Modelo de tarefa delegada (copiar e preencher)](#15-modelo-de-tarefa-delegada-copiar-e-preencher)
16. [Referências primárias](#16-referências-primárias)

---

## 0. Como usar este documento

- Cada tarefa tem um ID `Ex.y`, um **responsável** (papel de `.codex/agents/*.toml`),
  **dependências**, **arquivos permitidos** e **arquivos proibidos**, uma
  **especificação** (equações, API e casos degenerados), **testes exigidos** e
  **critérios de aceitação**. Uma tarefa só está concluída quando *todos* os
  critérios forem atendidos e verificados por quem não a implementou.
- O fluxo padrão de cada tarefa de código segue `AGENTS.md`:
  `pesquisador_quant` (parecer, se marcado) → `backend_quant`/`frontend`
  (implementação) → `testador_quant` (testes) ∥ `revisor_quant` (auditoria)
  → `corretor` (somente falhas confirmadas) → revalidação.
- **No máximo dois especialistas ativos ao mesmo tempo.** Tarefas marcadas
  como "paralelizável com" só podem andar em paralelo se os conjuntos de
  arquivos forem disjuntos (§14).
- Arquivos compartilhados (`main.py`, `src/simulation/report.py`, `README.md`,
  `pyproject.toml`, `.github/workflows/ci.yml`, `AGENTS.md`) pertencem ao
  **principal**, salvo transferência explícita registrada na tarefa.
- Se uma especificação daqui conflitar com o código atual, **pare e reporte**.
  Não ajuste o contrato silenciosamente.
- Commit e push só acontecem por decisão do principal com autorização do usuário.

Comandos de validação (a partir da raiz):

```sh
MPLBACKEND=Agg .venv/bin/python -m pytest -q
.venv/bin/python -m ruff check
.venv/bin/python -m mypy
```

Testes focados: `MPLBACKEND=Agg .venv/bin/python -m pytest -q tests/test_<arquivo>.py`.
Se não houver `.venv`, use o Python do ambiente com `pip install -e ".[dev]"`
somente se o principal autorizar a instalação.

---

## 1. Estado atual (ponto de partida verificado)

| Área | O que já existe | Arquivo |
| --- | --- | --- |
| Interface base | `BaseStochasticSimulator(s0, horizon, steps, n_sims, seed)`; `dt = horizon/steps`; RNG local `default_rng(seed)`; orçamento `MAX_WORKING_BYTES = 256 MiB`; `_check_memory(temporary_arrays=)`; métodos abstratos `calibrate`, `generate_paths` | `src/simulation/base.py` |
| GBM | `GbmSimulator`: modo `normal` exato; modo `t_student` com choques t padronizados por passo (não é GBM exato); `generate_terminal(batch_size, max_working_bytes)` exato e em lotes; `simulate_gbm`, `simulate_terminal_batched` | `src/simulation/gbm.py` |
| Calibração | `log_returns`, `estimate_gbm_params` (252 pregões, ddof=1, `mu = 252*mean + sigma²/2`); validação de preços; fundamentos; download | `src/calibration/*.py` |
| Risco | `risk_summary(paths, confidence)` → `expected_price`, `prob_loss`, `var`, `cvar` (perda `s0 - S_T`; CVaR inclui empates; sem clipping) | `src/risk/statistics.py` |
| Validação | `terminal_theory` (média, variância, quantis 5/50/95% em log-espaço, com proteção contra overflow/underflow); `compare_terminal`; `convergence_analysis` (prefixos aninhados); Wilson/Clopper-Pearson; probabilidade de perda analítica; sensibilidade de calibração; fora da amostra | `src/validation/*.py` |
| Backtest | `run_backtest` (expanding, reestimação, sementes por `(modelo, origem)`); GARCH(1,1)-t por MLE (`fit_garch_t`, `next_variance`, `simulate_cumulative`); Kupiec, Christoffersen, Brier, confiabilidade, Z1/Z2 de ES, Wilson efetivo | `src/backtest/*.py` |
| Previsão | Volatilidade causal (histórica/EWMA/NNLS), estudo congelado com replay e bootstrap por blocos | `src/forecasting/*.py` |
| Relatórios | `run_simulation` → JSON `schema_version: 1`, `model: "single_asset_gbm"`, IC da média terminal, convergência, ambiente; `write_report`/`read_report`/`read_replay` | `src/simulation/report.py` |
| CLI | `main.py` com flags planas (`--ticker`, `--horizon`, `--steps`, `--n-sims`, `--seed`, `--confidence`, `--replay`, `--out-of-sample`, sensibilidade, `--next-session`) | `main.py` |
| CI | ruff (`E,F,B,UP`), mypy não estrito, pytest em 3.10 e 3.12 | `.github/workflows/ci.yml`, `pyproject.toml` |

Itens **pendentes** do roadmap do README: IC para VaR/CVaR/probabilidade de
perda simulados; precificação europeia vs. Black–Scholes; redução de variância
(antitética e variável de controle); simulação multiativo correlacionada.

Pesquisa de cauda (`research/tail-risk/report.md`): no SPY entre 2015 e 2025, o
VaR 99% gaussiano foi rejeitado pelo teste de Kupiec (p = 0,006, 13 violações para
5,5 esperadas). O GARCH-t teve o menor Brier, mas os ICs contêm zero. Isso
**motiva** E4/E5, mas não comprova superioridade.

---

## 2. Contratos globais e regras invariantes

Valem para **todas** as tarefas; violá-los exige escopo explícito do usuário.

### 2.1 Unidades e convenções
- Preço na moeda do ativo; tempo em **anos**; `mu` em ano⁻¹; `sigma` em ano^-1/2;
  taxas `r`, `q` contínuas em ano⁻¹; intensidade de saltos `lambda` em ano⁻¹.
- Calibração diária com `TRADING_DAYS = 252` (importar de
  `calibration.estimation`, nunca redefinir).
- `mu` é o drift do **preço** (`E[S_T] = S0·e^{mu·T}`), não a média dos log-retornos.
- **Medidas**: simulações de risco usam a medida física P (parâmetros históricos).
  A precificação (E3) usa a medida risco-neutra Q, com drift `r - q` e um parâmetro
  de volatilidade explícito. É **proibido** precificar com `mu` histórico. Todo
  resultado de precificação deve registrar `measure: "risk_neutral"`, `r`, `q` e
  a origem de `sigma` (`"historical_proxy"` ou `"user_supplied"`).

### 2.2 Formatos de amostra
- Trajetórias de ativo único: `ndarray float64` com forma `(steps + 1, n_sims)` e
  linha 0 igual a `s0` em todas as colunas. **Inalterado.**
- Terminais: `float64 (n_sims,)`.
- Multiativo (E6, **novo contrato**): `(steps + 1, n_sims, n_assets)`. Ele **não**
  substitui o anterior; o valor da carteira volta ao formato `(steps + 1, n_sims)`.
- Perdas: `s0 - terminal`. VaR/CVaR em unidades de preço, podendo ser negativos.
  CVaR é a média das perdas `>= VaR`, com empates incluídos. **Inalterado.**

### 2.3 RNG e reprodutibilidade
- RNG local (`np.random.Generator`), nunca o global. Sementes inteiras `>= 0` ou `None`.
- Fluxos independentes: `np.random.SeedSequence(seed).spawn(k)` ou
  `SeedSequence(seed, spawn_key=(...))`, como já faz o backtest. Documente o
  mapeamento fluxo → uso (choques difusivos, saltos, variância, scrambles QMC).
- A mesma semente **não** promete igualdade entre versões do NumPy, ordem de
  sorteio, tamanho de lote ou formato de amostra. Testes comparam **leis**
  (momentos e quantis com tolerância), não bits, exceto em testes de
  determinismo da mesma chamada.
- Relatórios registram a convenção do RNG (`"NumPy default_rng/PCG64"`, ou
  `"scipy.stats.qmc.Sobol scrambled"` etc.) e as versões das bibliotecas.

### 2.4 Robustez numérica
- Valide entradas (tipo real, finitude, sinais, `bool` rejeitado), como em `base.py`.
- Trabalhe em log-espaço quando houver risco de overflow; nunca devolva
  `inf`/`nan` silenciosamente: levante `ValueError` com mensagem específica.
- Estime a memória antes de alocar: `8 bytes × elementos` por matriz `float64`
  e temporários. Respeite o `MAX_WORKING_BYTES` ou um parâmetro equivalente.
- Casos degenerados obrigatórios em toda função nova: `sigma = 0`, `n_sims = 1`,
  `steps = 1`, horizonte muito pequeno e parâmetros extremos porém finitos.

### 2.5 Tolerâncias estatísticas em testes
- Sementes fixas. Use tolerâncias **justificadas**: para uma média Monte Carlo,
  `|estimativa - teoria| <= k·SE` com `k = 4` (falso alarme ≈ 6e-5 por asserção
  sob normalidade assintótica), a menos que a tarefa defina outro valor.
- Nunca exija queda monotônica do erro entre prefixos aninhados.
- Para quantis, use o SE assintótico `sqrt(p(1-p)/N)/f(q_p)` com a densidade
  analítica, ou o intervalo de ordem binomial (E1.2).
- Para comparar distribuições inteiras, use KS ou Anderson–Darling com p-valor
  mínimo de 1e-3 e uma semente fixa. Documente a escolha.

### 2.6 Relatórios JSON
- Mudanças de esquema incrementam `schema_version` (atual: 1 → 2 em E9.2).
  `read_report` deve continuar lendo a v1. Os campos novos são opcionais na leitura.
- `json.dumps(report, allow_nan=False)` continua obrigatório.
- Não sobrescreva relatórios e artefatos existentes; escolha destinos novos.

### 2.7 Escopo e proibições
- Não alterar `research/experiments/**`, `research/tail-risk/**` (saídas
  publicadas e congeladas), `notebooks/` nem `figures/` sem tarefa específica.
- Não instalar dependências novas sem aprovação do principal. Use primeiro
  NumPy/SciPy/pandas/Matplotlib. Dependências opcionais (p. ex. `numba`) entram
  como *extras* em `pyproject.toml` e com *fallback* puro.
- Testes padrão são **offline**. Qualquer teste com rede fica marcado e fora da suíte padrão.

---

## 3. Mapa de épicos, dependências e ordem

```text
E10.1 (pytest markers / perf) ─┐
E1 (IC de risco) ──────────────┼──► E2 (redução de variância) ──► E3 (precificação)
                               │                                    │
E7.1 (diagnóstico GBM) ────────┤                                    ▼
                               ├──► E4 (saltos) ─────────────► E3.6 (Merton/Heston pricing)
                               ├──► E5 (Heston, GARCH sim) ────┘
                               └──► E6 (multiativo) ──► E6.4 (risco de carteira)
E3.9 (comparador de modelos) depende de E2, E3.3 e E5.3.
E8 (desempenho) depende de E2/E4/E5 estáveis.
E9 (CLI/relatórios) integra cada épico ao final; E9.1 (subcomandos) pode começar cedo.
```

| Ordem | Épico | Valor | Risco de contrato | Pré-requisitos |
| --- | --- | --- | --- | --- |
| 1 | E1 | Fecha item do roadmap; base para comparar E2 | Baixo (campos novos) | — |
| 2 | E2 | Menor erro MC com o mesmo custo; reutilizado por E3 | Baixo | E1 |
| 3 | E3 | Novo domínio (precificação) com referência analítica forte | Médio (nova medida) | E2.1/E2.2 |
| 4 | E7 | Justifica E4/E5 com evidência | Baixo | — |
| 5 | E4 | Caudas e saltos; motivado por tail-risk | Médio (novo simulador) | E1 |
| 6 | E5 | Volatilidade estocástica | Médio | E1, E3.1 para validar Heston |
| 7 | E6 | Multiativo | **Alto** (novo formato 3D) | E1 |
| 8 | E8 | Escala | Baixo se a lei for preservada | E2, E4, E5 |
| contínuo | E9, E10 | Integração e qualidade | Médio (schema v2) | por épico |

---

## E1 — Incerteza Monte Carlo das métricas de risco

**Objetivo.** Hoje `risk_summary` devolve estimativas pontuais. Este épico
adiciona intervalos de confiança do **estimador** (erro Monte Carlo) para
`prob_loss`, `var` e `cvar`, distinguidos explicitamente dos percentis da
distribuição e da incerteza de calibração.

### E1.1 IC para a probabilidade de perda
- **Responsável:** `backend_quant`. **Arquivos permitidos:** `src/risk/intervals.py` (novo),
  `tests/test_risk_intervals.py` (novo, via `testador_quant`).
  **Proibidos:** `src/risk/statistics.py` (o contrato de `risk_summary` não muda).
- **Especificação:** `prob_loss_interval(terminal, s0, level=0.95, method="wilson")`
  usa `k = #{S_T < s0}`, `n = N`. Reaproveite `validation.tail_probability.wilson_interval`
  e `clopper_pearson_interval`; não duplique fórmulas. Retorna
  `{"estimate", "lower", "upper", "method", "level", "n", "k"}`.
- **Testes:** `k = 0` e `k = n` (limites dentro de [0, 1], sem nan); cobertura
  empírica do IC em 2.000 réplicas com N = 500 e p = 0,3, dentro de
  [0,93; 0,97] para Wilson 95% (justifique pela variabilidade binomial das réplicas).
- **Aceitação:** a função é pura, com validação de entradas, sem alterar `risk_summary`.

### E1.2 IC para o VaR (quantil) — livre de distribuição
- **Especificação:** para o quantil de nível `p` das perdas ordenadas
  `L_(1) <= … <= L_(N)`, use o intervalo por estatísticas de ordem:
  `j = Binom.ppf(a/2, N, p)` e `k = Binom.ppf(1 - a/2, N, p) + 1`, com
  `IC = [L_(j), L_(k)]` e índices recortados para [1, N]. Cobertura `>= 1 - a`
  (conservador). Documente que o ponto continua sendo `_linear_percentile`
  (interpolação linear do NumPy), mas o IC usa estatísticas de ordem. A
  diferença é O(1/N) e deve ser declarada.
- **Alternativa opcional:** SE assintótico `sqrt(p(1-p)/N)/f(q_p)`, apenas quando a
  densidade analítica for conhecida (GBM). Nesse caso, compare com o IC por estatísticas de ordem.
- **Testes:** com GBM e `terminal_theory`, a perda de nível `p` é
  `s0 - Q_{1-p}(S_T)`. Para isso, estenda `terminal_theory` com quantis arbitrários
  em uma função nova, `terminal_quantile(s0, mu, sigma, T, p)`, em
  `src/validation/statistics.py`. Essa extensão é responsabilidade do
  `backend_quant` desta tarefa. O IC deve cobrir o valor teórico em ≥ 93% de
  1.000 réplicas (N = 2.000, p = 0,95). Teste também N pequeno (N = 10), em que
  os índices saturam.

### E1.3 IC para o CVaR
- **Especificação:** dois métodos.
  1. **Assintótico** (Brazauskas et al. 2008; Manistre e Hancock 2005):
     `Var(CVaR_hat) ≈ [Var(L | L >= VaR) + p·(CVaR - VaR)²] / (N·(1-p))`.
     Use a forma implementada **com referência** e verifique-a por simulação.
  2. **Bootstrap percentil** i.i.d. com `B` réplicas (padrão 1.000), RNG próprio
     (`SeedSequence(seed, spawn_key=("cvar_bootstrap",))`) e memória
     `O(N)` por réplica. **Não** materialize `B × N` de uma vez; processe em lotes.
- A definição do CVaR (empates incluídos) deve ser **idêntica** à de `risk_summary`.
  Reutilize a função interna ou extraia um helper compartilhado. O helper fica no
  arquivo novo, e `statistics.py` só muda se o principal transferir a propriedade.
- **Testes:** valor analítico do CVaR do GBM como referência, por integração
  numérica (`scipy.integrate.quad` sobre a densidade lognormal) ou fórmula fechada:
  `E[S_T | S_T <= x] = S0·e^{muT}·Φ(d - sigma√T)/Φ(d)`, com
  `d = (ln(x/S0) - (mu - sigma²/2)T)/(sigma√T)`. Daí `CVaR = s0 - E[S_T | S_T <= Q_{1-p}]`.
  Cobertura empírica em ≥ 90% de 500 réplicas, com tolerância justificada pelo
  viés do bootstrap em caudas.

### E1.4 Integração no relatório
- **Responsável:** principal (dono de `report.py`), ou transferência ao `backend_quant`.
- Adicione `results.risk_intervals = {"prob_loss": …, "var": …, "cvar": …}`
  **somente** no schema v2 (ver E9.2) ou como campo opcional da v1 sem quebrar `read_report`.
- Rótulos no JSON: `"interval_kind": "monte_carlo_estimator"`, para distinguir de
  `calibration_sensitivity` (incerteza de parâmetros).
- README: marcar o item do roadmap só depois da revisão.

**Critérios de aceitação do épico:** três funções testadas; cobertura empírica
documentada no teste; `risk_summary` intacto (os testes atuais passam sem
modificação); revisão do `revisor_quant` sobre as fórmulas assintóticas.

---

## E2 — Redução de variância

**Objetivo.** Reduzir o erro Monte Carlo de estimadores de **valor esperado**
(média terminal e, em E3, preços de opções) mantendo a lei correta. Cada
técnica reporta o fator de redução de variância (VRF) observado e o SE correto
para o estimador usado.

### Regras comuns
- O método antigo continua sendo o padrão. O novo é ativado por parâmetro
  `variance_reduction: Literal["none", "antithetic", "control", "qmc"]`.
- **Não** aplique técnicas que distorçam quantis sem documentar: amostras
  antitéticas continuam valendo como amostra da lei, mas os pares são
  dependentes. Os ICs de E1 que assumem i.i.d. **não** valem diretamente; use pares como unidade.

### E2.1 Variáveis antitéticas
- **Arquivos:** `src/simulation/variance_reduction.py` (novo), mais uma alteração mínima em
  `src/simulation/gbm.py`, que exige transferência do principal se `gbm.py`
  estiver com outro escritor.
- **Especificação:** para `n_sims = 2m`, sorteie `Z` com forma `(steps, m)` e use
  `[Z, -Z]`. Se `n_sims` for ímpar, **rejeite** com `ValueError`, para não
  introduzir um caminho solitário de peso diferente. Os terminais ficam em
  `(n_sims,)`, com a ordem `[originais..., espelhados...]` documentada.
- **Estimador e SE:** `Y_i = (f(S^+_i) + f(S^-_i))/2`, `i = 1..m`;
  `SE = std(Y, ddof=1)/sqrt(m)`. Exporte `antithetic_mean(values) -> {"estimate", "standard_error", "pairs"}`.
- **Testes:** (a) média terminal dentro de 4·SE de `S0·e^{muT}`; (b) o VRF da
  média terminal deve ser `> 1` para `sigma√T` moderado, já que `exp` é monótona e
  a covariância do par é negativa. Calcule também o VRF teórico
  `Var_plain/(2·Var_pair)`, com
  `Cov(S+, S-) = S0² e^{2(mu-σ²/2)T} - (E S_T)²`, e compare a ±20%;
  (c) a matriz tem a forma e a linha 0 corretas; (d) `n_sims` ímpar é rejeitado.

### E2.2 Variável de controle
- **Especificação:** para um alvo `Y = g(S)`, use o controle `X = S_T`, com
  `E[X] = S0·e^{muT}` exato. O estimador é `Ȳ - β̂(X̄ - E X)`, com
  `β̂ = Cov(X, Y)/Var(X)` estimado na **mesma** amostra (viés O(1/N), declarado)
  ou em uma amostra piloto independente (sem viés; opção `pilot_fraction`).
- O SE usa os resíduos `Y - β̂(X - EX)`. Reporte `beta`, `rho²` e `VRF ≈ 1/(1-rho²)`.
- **Uso em E3:** preço de opções com o controle `e^{-rT}S_T` (média `S0 e^{-qT}`
  sob Q). Em asiáticas aritméticas, o controle é a asiática geométrica fechada (E3.4).
- **Testes:** quando `Y = X`, o estimador recupera `E X` com SE ≈ 0, o que testa
  a degeneração `rho = 1` sem dividir por zero. `Var(X) = 0` (`sigma = 0`)
  deve cair em `β = 0` com aviso no resultado, sem `nan`.

### E2.3 Quasi-Monte Carlo randomizado (Sobol + scrambling)
- **Especificação:** `scipy.stats.qmc.Sobol(d, scramble=True, seed=Generator)`.
  - Terminal exato: `d = 1`, `Z = ndtri(u)`.
  - Trajetórias: `d = steps`, com **construção por ponte browniana** para concentrar
    a variância nas primeiras dimensões (Glasserman, §5.5). Para `steps > 21201`
    (limite de dimensão do Sobol do SciPy, a confirmar na versão instalada),
    rejeite com mensagem clara.
  - `N` deve ser potência de 2. Caso contrário, use `ValueError` ou um aviso, a decidir
    pelo principal e registrar.
  - Recorte `u` para `[ε, 1-ε]` com `ε = 2^-53`, evitando `ndtri(0) = -inf`.
  - **Erro:** `R` scrambles independentes (padrão `R = 16`) com
    `SE = std(médias_r, ddof=1)/sqrt(R)`. Um único scramble **não** tem SE
    estimável e o relatório deve dizer isso.
- **Testes:** média terminal dentro de 4·SE com `R = 16` e `N = 2^12`; erro
  RMS (sobre 50 sementes) menor que o do MC simples com o mesmo `N·R`,
  documentando o ganho medido (sem fixar um número mágico); ponte browniana
  reproduz `Var(W_t) = t` em cada ponto da grade (tolerância 4·SE).

### E2.4 Integração e comparação
- `validation.convergence_analysis` passa a aceitar estimador e SE
  externos (função nova `convergence_with_estimator`). A antiga não muda.
- Gráfico de convergência comparando `none`, `antithetic`, `control` e `qmc` (E9.4).
- **Aceitação do épico:** cada técnica tem SE correto, teste de lei e teste de
  VRF; documentação de quando cada técnica **não** ajuda (p.ex. antitéticas em
  payoff par/não monótono).

---

## E3 — Precificação de derivativos

**Objetivo.** Criar `src/pricing/` com precificação sob a medida risco-neutra,
validada contra fórmulas fechadas. É o primeiro módulo a usar Q, então as
regras de §2.1 são obrigatórias.

### E3.0 Parecer prévio (`pesquisador_quant`, somente leitura)
Entregar uma nota curta com: a dinâmica sob Q (`dS = (r - q)S dt + σS dW^Q`);
a escolha de `r` (input do usuário, sem download automático); dividendos
contínuos `q` vs. discretos (só contínuos neste épico); os limites do uso de
volatilidade histórica como proxy da implícita; e a lista de fórmulas de
referência com fonte (Hull; Haug, *Complete Guide to Option Pricing Formulas*;
Glasserman).

### E3.1 Black–Scholes–Merton analítico
- **Arquivos:** `src/pricing/__init__.py`, `src/pricing/black_scholes.py`,
  `tests/test_black_scholes.py`.
- **API:** `bs_price(S, K, T, r, q, sigma, kind: Literal["call","put"]) -> float`;
  `bs_greeks(...) -> dict` com `delta, gamma, vega, theta, rho` (unidades
  declaradas: vega por 1,00 de sigma, não por 1%; theta por ano).
- **Fórmulas:** `d1 = [ln(S/K) + (r - q + σ²/2)T]/(σ√T)`, `d2 = d1 - σ√T`;
  `C = S e^{-qT}Φ(d1) - K e^{-rT}Φ(d2)`; `P = K e^{-rT}Φ(-d2) - S e^{-qT}Φ(-d1)`.
  Use `scipy.special.ndtr` e calcule `log` antes de dividir.
- **Degenerações:** `T = 0` ou `σ = 0` → payoff descontado do forward
  determinístico (`max(S e^{(r-q)T} - K, 0)·e^{-rT}`), sem divisão por zero;
  opções muito fora do dinheiro devem dar preço ≥ 0, sem `-0.0` negativo relevante.
- **Testes:** paridade put-call `C - P = S e^{-qT} - K e^{-rT}` em 1e-10
  relativo; valores de tabela de Hull (citar exemplo e página); limites
  (`C ≤ S e^{-qT}`, `C ≥ max(S e^{-qT} - K e^{-rT}, 0)`); gregas contra
  diferenças finitas centrais com `h` justificado; vetorização opcional em `K`.

### E3.2 Volatilidade implícita
- **API:** `implied_volatility(price, S, K, T, r, q, kind, *, bracket=(1e-6, 5.0))`.
  Use Brent (`scipy.optimize.brentq`) sobre `bs_price - price`. Fora dos limites
  de não arbitragem, levante `ValueError`.
- **Testes:** ida e volta `σ → preço → σ` em uma grade de moneyness e T, com erro < 1e-8.

### E3.3 Monte Carlo europeu
- **Arquivos:** `src/pricing/monte_carlo.py`, `tests/test_pricing_mc.py`.
- **API:** `mc_european(S, K, T, r, q, sigma, kind, n_sims, seed, *, variance_reduction="none", batch_size=...)`
  → `{"price", "standard_error", "ci_95", "n_sims", "method", "measure": "risk_neutral", "rng"}`.
- Use a lei terminal exata, reaproveitando a lógica de `generate_terminal` com
  `mu = r - q`. **Atenção:** em `GbmSimulator`, o drift é `mu`. Sob Q, passe
  `mu = r - q` explicitamente e documente que isso é uma reparametrização, não a
  calibração histórica.
- **Testes:** `|MC - BS| ≤ 4·SE` para call e put ATM/ITM/OTM; cobertura do IC
  95% em 500 réplicas dentro de [0,93; 0,97]; com `antithetic` e `control`, SE
  menor que no simples com o mesmo N.

### E3.4 Opções asiáticas (dependentes de trajetória)
- **Especificação:** média **aritmética** discreta nas datas da grade (excluindo
  ou incluindo `t0`, opção explícita e registrada). O controle é a asiática
  **geométrica** discreta com fórmula fechada (Kemna e Vorst 1990, versão
  discreta): `ln G ~ N(m_G, v_G)`, com
  `m_G = ln S + (r - q - σ²/2)·(1/n)Σt_i` e
  `v_G = σ²/n² · Σ_i Σ_j min(t_i, t_j)`.
- O motor de trajetória com GBM exato nos pontos da grade dá **lei exata das
  médias discretas**. Não há erro de discretização para a média discreta, mas
  há para uma média contínua, que não é o objetivo.
- **Testes:** a fórmula geométrica fechada bate com o MC a 4·SE; o controle
  geométrico reduz o SE da aritmética (VRF > 1, documentado); o limite
  `n = 1` reduz-se a europeia.

### E3.5 Barreiras e lookback
- **Barreira** (up/down, in/out). Com monitoramento **discreto** nas datas da
  grade, o MC é exato para o contrato discreto. Para monitoramento **contínuo**, aplique a
  correção por **ponte browniana**: a probabilidade de cruzamento entre `S_i` e
  `S_{i+1}` (ambos acima de `B`, barreira inferior) é
  `exp(-2·ln(S_i/B)·ln(S_{i+1}/B)/(σ²Δt))`.
- Referência analítica contínua: Merton/Reiner–Rubinstein (Haug, cap. 4).
  Correção de Broadie–Glasserman–Kou para o discreto: o preço com monitoramento
  discreto ≈ fórmula contínua com a barreira **afastada de S0**,
  `B → B·e^{+β·σ√Δt}` para barreira superior e `B → B·e^{-β·σ√Δt}` para inferior,
  com `β = -ζ(1/2)/√(2π) ≈ 0,5826` (BGK 1997, Teorema 1; conferir na fonte).
- **Testes:** paridade in + out = vanilla (exata por construção, por caminho);
  barreira contínua com ponte vs. fórmula fechada a 4·SE; o discreto sem
  ponte converge para o contínuo quando `steps` cresce (testar a tendência com
  folga, sem monotonicidade estrita).
- **Lookback** de strike flutuante com fórmula de Goldman–Sosin–Gatto para
  validação contínua e a mesma ressalva sobre discreto vs. contínuo.

### E3.6 Opções americanas — Longstaff–Schwartz
- **Arquivo:** `src/pricing/american.py`.
- Regressão em base polinomial (Laguerre ponderado ou monômios de `S/K`,
  grau ≤ 3) **somente nos caminhos dentro do dinheiro**; amostras separadas
  para regressão e avaliação (estimador inferior sem viés de *foresight*).
  Reporte o limite inferior e, opcionalmente, o dual de Andersen–Broadie como superior.
- Memória: exige trajetórias completas; estime e respeite `_check_memory`.
- **Testes:** put americana de Longstaff–Schwartz (2001, Tabela 1: S = 36…44,
  K = 40, r = 0,06, σ = 0,2/0,4, T = 1 e 2 anos; corrigido de "T = 1/2", conferir na fonte) dentro de ±2·SE + 0,02 da referência
  por diferenças finitas citada no artigo; call americana sem dividendos ≈
  europeia (a 3·SE).

### E3.7 Gregas por Monte Carlo
- **Pathwise** para delta (`e^{-rT}·1{S_T>K}·S_T/S`) e vega; **likelihood ratio**
  para gamma ou payoffs descontínuos (digitais); **diferenças finitas com números
  aleatórios comuns** como base. Compare com `bs_greeks` a 4·SE.

### E3.8 Integração
- CLI: subcomando `price` (E9.1) com `--kind`, `--strike`, `--rate`,
  `--dividend-yield`, `--vol` (obrigatório ou `--vol-from-history`, com aviso).
- Relatório: `model: "black_scholes_mc"`, campos de §2.1. README: nova seção
  "Pricing" com fórmulas e limitações (volatilidade histórica ≠ implícita).

### E3.9 Comparador de modelos de precificação (GBM, Student-t, GARCH-t)

**Origem.** Pedido do usuário em 2026-11-22, a partir da figura
`research/tail-risk/figures/fig2_observed_vs_predicted.png`, que compara a
probabilidade histórica de perdas de 5 dias (^GSPC) com as previsões fora da
amostra do GBM gaussiano, do GBM com choques t e do GARCH(1,1)-t. A ideia é
precificar as mesmas opções europeias sob os três modelos e usar a evidência
de cauda para escolher o modelo adequado à situação atual.

**Decisões do usuário (2026-11-22).**
1. **Dependências:** esperar o E2. Nenhuma subtarefa começa antes de E2.1/E2.2
   (e, portanto, de E3.3) estarem concluídos e revisados.
2. **Risco-neutralização:** por correção de drift, aplicada **por passo diário**
   (ver E3.9.1 para a justificativa e o problema em aberto).
3. **Entradas:** ticker, `r`, `q`, strikes e vencimento são informados pelo
   usuário. Nada de taxas é baixado; o histórico de preços usa o fluxo de
   dados existente, com snapshot para replay.
4. **Saída:** **interativa** (pedido explícito; exceção à regra "dashboard só
   se solicitado" de `AGENTS.md`). Usar Plotly como extra opcional
   `interactive`, já presente na `main` (`--interactive`, `paths.html`). Esta
   branch ainda não contém esse commit: o principal integra a `main` antes
   de E3.9.4.

**Separação de medidas (obrigatória).** A figura de origem está na medida
**P** (probabilidades reais de perda) e é a **evidência** para escolher o
modelo. Os preços estão na medida **Q**. Para o GBM, Q é única (Black–Scholes).
Para t e GARCH-t, o mercado é incompleto e Q **não é única**: a regra de
correção de drift é uma escolha declarada em todo resultado
(`"risk_neutralization": "daily_drift_correction"`), não um fato do modelo.
Sem cotações de opções, nenhum painel afirma qual preço é o "correto".

#### E3.9.0 Parecer (`pesquisador_quant`, somente leitura)
- Correção de drift diária: condição de martingale
  `E^Q[S_{t+Δ}/S_t | F_t] = e^{(r-q)Δ}` em cada passo, com Δ = 1/252.
- **Problema em aberto a resolver no parecer:** com inovações Student-t nos
  log-retornos, `E[e^{σε}] = ∞` para todo σ > 0 (a função geradora de momentos
  da t não existe). Então `E[S_T] = ∞` e a correção analítica no log-retorno não
  está definida; o preço de uma call é infinito em teoria, e a média Monte
  Carlo é instável. Avaliar e recomendar uma das saídas, com fonte primária:
  (a) t truncada em ±k desvios, com a FGM truncada calculada numericamente;
  (b) correção empírica de martingale por passo (Duan e Simonato 1998,
  *Empirical martingale simulation*, Management Science); (c) correção no
  retorno simples (`S_{t+Δ} = S_t·e^{(r-q)Δ}·(1+R)/E[1+R]`, com R limitado
  inferiormente). Declarar o viés de cada opção e o que muda para puts
  (limitadas) e calls (não limitadas).
- GARCH-t: a mesma questão, com a variância condicional; comparar com Duan
  (1995, LRNVR), que pressupõe inovações normais.

#### E3.9.1 Simuladores sob Q (`backend_quant`)
- **Por que diário, não a cada 5 dias:** a correção por passo garante o
  martingale em **todas** as datas da grade (lei das esperanças iteradas), o
  que vale para qualquer vencimento, para opções dependentes de trajetória e
  para gregas. Corrigir só no horizonte de 5 dias acertaria a média terminal
  daquele vencimento e deixaria as datas intermediárias fora da condição de
  martingale. O horizonte de 5 dias é a janela de **avaliação P** da figura,
  não a frequência da correção Q.
- GBM: `mu = r - q` (E3.3). Student-t: `GbmSimulator(dist_type="t_student")`
  com a correção do E3.9.0. GARCH-t: `GarchTSimulator` (E5.3), que importa
  `backtest.garch` sem alterá-lo, mais a correção do E3.9.0.
- **Teste de aceitação:** o preço descontado é martingale em todas as datas,
  `|mean(e^{-r t_i} S_{t_i}) - S0 e^{-q t_i}| ≤ 4·SE` para cada `t_i` da grade.

#### E3.9.2 Comparador (`backend_quant`)
- **Arquivo:** `src/pricing/model_comparison.py`.
- Para uma grade de strikes × {call, put}: preço por modelo com SE e IC 95%,
  e a volatilidade implícita de BS correspondente (`implied_volatility`, E3.2),
  que gera o smile/skew de cada modelo.
- O GBM por MC deve coincidir com o BS a 4·SE (sanidade do motor).
- **Risco de modelo:** amplitude entre os preços dos modelos por strike.

#### E3.9.3 Evidência para escolher o modelo (`backend_quant`)
- Probabilidades de cauda de 5 dias na medida P **na data atual** (GARCH a
  partir da variância condicional de hoje) contra a frequência histórica com
  Wilson efetivo (`backtest.metrics.effective_wilson`, somente importação).
- Tabela de evidência por modelo: volatilidade (histórica, GARCH condicional,
  GARCH incondicional), Kupiec, Christoffersen, Brier e testes de ES de
  `backtest.metrics`, a partir de um backtest novo, gravado em diretório novo
  (nunca em `research/tail-risk/**`).
- **Recomendação:** o modelo mais bem calibrado fora da amostra na cauda
  relevante para a opção (puts fora do dinheiro ↔ cauda esquerda), com a
  evidência exibida e sem ocultar a amplitude entre modelos. A escolha é
  apoio à decisão, não previsão garantida.

#### E3.9.4 Visualização interativa (`frontend`)
- Página HTML Plotly (extra `interactive`, com *fallback* Matplotlib quando o
  extra estiver ausente) com quatro painéis: preços por modelo; smile por
  modelo; cauda P de hoje (versão atual da figura de origem); tabela de
  evidência e recomendação.
- Controles: seleção de modelos, strikes, call/put e vencimento, sobre
  resultados **pré-calculados** no JSON (sem simular no navegador).
- Testes estruturais sem comparar pixels, como `tests/test_plotting_design.py`.

#### E3.9.5 Integração (principal)
- Relatório JSON com `model: "pricing_model_comparison"`, os campos de §2.1
  por modelo, `risk_neutralization`, as sementes e os fluxos de RNG por modelo.
- Ponto de entrada no CLI depois do E9.1 (subcomando `price --compare`), ou
  pendência documentada.

**Revisão independente obrigatória** do `revisor_quant` para E3.9.0/E3.9.1
(risco-neutralização) e E3.9.3 (critério de escolha).

---

## E4 — Modelos com saltos (Merton, Kou)

**Motivação.** O GBM não reproduz caudas pesadas nem gaps (README, "Limitations";
`research/tail-risk`). Saltos são a extensão mínima que preserva simulação
exata nos pontos da grade.

### E4.1 Merton jump-diffusion (simulação exata)
- **Arquivos:** `src/simulation/jump_diffusion.py`, `tests/test_jump_diffusion.py`.
- **Dinâmica:** `dS/S_- = (mu - λk)dt + σ dW + (J - 1)dN`, com
  `ln J ~ N(m_J, δ²)`, `N` Poisson de intensidade `λ` (ano⁻¹) e
  `k = E[J - 1] = e^{m_J + δ²/2} - 1`. A compensação `-λk` garante
  `E[S_T] = S0·e^{muT}`, o que preserva a interpretação de `mu` (§2.1).
- **Incremento exato no passo Δt:**
  `Δln S = (mu - σ²/2 - λk)Δt + σ√Δt·Z + N_Δ·m_J + δ·√N_Δ·Z'`,
  com `N_Δ ~ Poisson(λΔt)` e `Z, Z'` normais independentes.
- **API:** `MertonJumpSimulator(BaseStochasticSimulator)` com
  `mu, sigma, jump_intensity, jump_mean, jump_std`. `generate_paths()` retorna
  `(steps+1, n_sims)`; `generate_terminal()` usa a soma direta com
  `N_T ~ Poisson(λT)`. Fluxos de RNG separados para difusão, contagens e tamanhos (§2.3).
- **Memória:** três matrizes `(steps, n_sims)` temporárias no pior caso; use
  `_check_memory(temporary_arrays=...)` com contagem real e reutilize buffers.
- **Teoria (nova função em `src/validation/jump_theory.py`):**
  - `E[S_T] = S0 e^{muT}`.
  - `E[S_T²] = S0² · exp((2mu + σ²)T + λT·(e^{2m_J+2δ²} - 1 - 2k))`.
  - Densidade/CDF de `ln S_T` como mistura Poisson de normais, truncada em
    `n_max` com cauda `P(N > n_max) < 1e-14`, e quantis via `brentq` na CDF.
- **Testes:** `λ = 0` recupera o GBM (mesma lei: KS p > 1e-3 e momentos a 4·SE);
  média e variância terminais a 4·SE; quantis 1%/5%/95% dentro do IC de E1.2; `λΔt`
  grande (p.ex. 50) sem overflow; `δ = 0` (saltos determinísticos).

### E4.2 Kou (saltos duplo-exponenciais)
- `ln J` assimétrico: com probabilidade `p`, `Exp(η1)` positivo (`η1 > 1`
  para `E[J]` finito); caso contrário `-Exp(η2)`.
  `k = p·η1/(η1-1) + (1-p)·η2/(η2+1) - 1`. A simulação exata soma `N_Δ`
  exponenciais por caminho e pode ser vetorizada via gama:
  soma de `n` `Exp(η)` ~ `Gamma(n, 1/η)`, separando a contagem de ups ~ `Binom(N, p)`.
- **Testes:** momentos analíticos via função geradora; `p ∈ {0, 1}`; `η1 ≤ 1` rejeitado.

### E4.3 Calibração de saltos
- **Arquivo:** `src/calibration/jumps.py`.
- MLE dos retornos diários pela mistura Poisson-normal truncada
  (Δt = 1/252), com multi-start e limites (`λ ∈ [0, 252]`, `δ > 1e-4`).
  Reporte `converged`, `nll`, número de inícios e **avisos de identificabilidade**
  (λ alto com δ pequeno ≈ difusão; comparar AIC/BIC com o GBM).
- **Testes:** recuperação de parâmetros em dados sintéticos (N = 5.000 dias)
  com tolerância baseada no erro padrão da informação de Fisher numérica
  (Hessiana por diferenças finitas), não em percentuais fixos.
- **Proibido:** baixar dados nos testes. Use séries sintéticas com semente.

### E4.4 Precificação sob saltos (opcional, depende de E3)
- Fórmula de Merton (1976) como série de BS ponderada por Poisson com
  `λ' = λ(1+k)`, `σ_n² = σ² + nδ²/T` e `r_n = r - λk + n·ln(1+k)/T`. É o MC sob Q
  com `mu = r - q`. Aceitação: MC a 4·SE da série.

---

## E5 — Volatilidade estocástica (Heston) e GARCH como simulador

### E5.1 Heston — simulação
- **Arquivos:** `src/simulation/heston.py`, `tests/test_heston.py`.
- **Dinâmica:** `dS = mu S dt + √v S dW1`, `dv = κ(θ - v)dt + ξ√v dW2`,
  `d⟨W1, W2⟩ = ρ dt`. A condição de Feller `2κθ ≥ ξ²` é **reportada**, não
  exigida (o QE lida com os dois casos).
- **Esquema:** QE de Andersen (2008) para `v` com `ψ_c = 1,5`, mais a discretização
  do log-preço de Andersen (central, γ1 = γ2 = 0,5) com correção de martingale opcional.
  *Fallback*: Euler com *full truncation* (Lord et al. 2010) como esquema de
  referência alternativo. O parâmetro `scheme` é explícito.
- **Contrato:** ao contrário do GBM, **há erro de discretização**. Isso deve estar na docstring
  e no relatório (`"discretization": "andersen_qe"`, `"exact_on_grid": false`).
  Retorne opcionalmente a matriz de variância (`return_variance=True`) com a
  mesma forma, contando-a na memória.
- **Teoria (`src/validation/heston_theory.py`):**
  - `E[v_t] = θ + (v0 - θ)e^{-κt}`;
    `Var[v_t] = v0 ξ² e^{-κt}(1 - e^{-κt})/κ + θξ²(1 - e^{-κt})²/(2κ)`.
  - `E[S_T] = S0 e^{muT}` (com correção de martingale).
  - Preço europeu semi-analítico via função característica na forma "little
    Heston trap" de Albrecher et al. (2007), para evitar descontinuidade do log
    complexo, com integração `scipy.integrate.quad` ou Gauss–Laguerre e tolerância documentada.
- **Testes:** momentos de `v` a 4·SE; com `ξ → 0` e `v0 = θ`, a lei se aproxima
  do GBM com `σ = √θ`; preço MC sob Q vs. semi-analítico em **casos de
  benchmark publicados** (Andersen 2008, casos I–III) dentro de `4·SE + viés
  de discretização tolerado`. O viés é declarado por caso e medido em ao menos
  dois `Δt`, verificando redução com folga e sem exigir monotonicidade estrita.

### E5.2 Heston — calibração
- **Opção A (preços de opções):** fora de escopo sem fonte de dados de opções. Só
  com tarefa do usuário.
- **Opção B (série histórica):** apenas um esboço/parecer do `pesquisador_quant`
  sobre os limites (variância latente; filtro de partículas ou método de momentos
  com volatilidade realizada). **Não implementar** sem decisão do usuário.

### E5.3 GARCH(1,1)-t como simulador de trajetórias
- Hoje `backtest.garch.simulate_cumulative` retorna somente log-retornos
  cumulativos, em unidade diária. Crie `GarchTSimulator(BaseStochasticSimulator)` em
  `src/simulation/garch.py`, que **importa** `backtest.garch` (sem duplicar o MLE)
  e devolve preços `(steps+1, n_sims)` com `steps` = dias (`horizon = steps/252`, validado).
- **Unidades:** o GARCH é diário. Converta e documente explicitamente. `mu` aqui é
  a **localização do log-retorno diário** (como em `research/tail-risk/report.md` §5),
  **não** o drift de preço. Registre isso no relatório para não confundir com §2.1.
- **Teste:** `simulate_cumulative` e o novo simulador, com a mesma semente e
  ordem de sorteio, produzem a mesma soma (determinismo interno); variância
  incondicional `ω/(1-α-β)` reproduzida em longo horizonte a 4·SE.
- **Proibido:** alterar `src/backtest/garch.py`, cujos resultados publicados dependem
  dele. Se for necessário refatorar, faça uma tarefa separada, com regressão
  bit a bit dos números de `research/tail-risk/backtest/metrics_output.txt`.

---

## E6 — Multiativos correlacionados e carteiras

**Aviso de contrato.** Este épico cria o formato 3D `(steps+1, n_sims, n_assets)`.
O usuário precisa aprovar o contrato antes da implementação (§2.2).

### E6.1 Simulador GBM multiativo
- **Arquivos:** `src/simulation/multi_asset.py`, `tests/test_multi_asset.py`.
- **Dinâmica:** `dS_i = mu_i S_i dt + σ_i S_i dW_i`, com `d⟨W_i, W_j⟩ = ρ_ij dt`.
  Incremento exato:
  `Δln S = (mu - σ²/2)Δt + √Δt · diag(σ) · L · Z`, com `L L^T = R` (Cholesky).
- **Validação de `R`:** simétrica (tolerância 1e-12), diagonal unitária,
  autovalores ≥ -1e-12. Para semidefinida (posto incompleto), use LDLᵀ ou
  autodecomposição com recorte e documente; **não** "conserte" matrizes
  inválidas silenciosamente (oferecer `nearest_correlation` de Higham 2002 apenas
  como função explícita separada).
- **Memória:** `8·(steps+1)·n_sims·n_assets` mais temporários; use `_check_memory`
  generalizado (novo parâmetro `n_assets` na verificação, sem mudar a
  assinatura existente).
- **Testes:** correlação amostral dos log-retornos terminais vs. `ρ_ij`, com IC de
  Fisher-z; `n_assets = 1` reproduz `GbmSimulator` em lei; `ρ = ±1` (posto 1);
  matriz não PSD rejeitada; covariância terminal analítica
  `Cov(S_i, S_j) = S_i0 S_j0 e^{(mu_i+mu_j)T}(e^{ρ_ij σ_i σ_j T} - 1)` a 4·SE.

### E6.2 Calibração multiativo
- **Arquivo:** `src/calibration/multi_asset.py`.
- Alinhamento por datas comuns (*inner join*), relatando quantas datas foram
  descartadas por ativo. Sem forward-fill, salvo opção explícita.
- `mu_i = 252·mean_i + σ_i²/2`, `σ_i = √252·std_i` e `R` = correlação amostral.
  Opcionalmente use *shrinkage* de Ledoit–Wolf, como função separada e documentada.
- Download multi-ticker em `calibration/data.py` (dono: principal ou transferência):
  snapshot com hash por ativo, no formato do snapshot existente.

### E6.3 Valor de carteira
- `portfolio_paths(asset_paths, weights, kind="quantities"|"initial_weights")`
  → `(steps+1, n_sims)`, com linha 0 = valor inicial comum. Isso permite
  reaproveitar `risk_summary` **sem alteração**.
- Pesos podem ser negativos (vendido); a soma zero torna `s0 = 0` e é inválida para
  `risk_summary` (perdas relativas). Rejeite com mensagem clara ou ofereça uma métrica absoluta.
- Sem rebalanceamento (*buy-and-hold*) por padrão; rebalanceamento periódico é
  opção futura e explícita.

### E6.4 Decomposição de risco
- VaR/CVaR por componente: Euler/contribuição marginal via
  `E[L_i | L ≥ VaR]` (estimador de cauda, com SE por bootstrap de E1.3).
  A soma das contribuições ao CVaR é igual ao CVaR da carteira, o que vale como teste de identidade.
- Teste: dois ativos independentes e idênticos com pesos iguais têm contribuições
  iguais dentro do SE.

---

## E7 — Calibração, diagnóstico e incerteza de parâmetros

### E7.1 Diagnóstico de adequação do GBM
- **Arquivo:** `src/validation/diagnostics.py`.
- Para os log-retornos diários: Jarque–Bera e Anderson–Darling (normalidade);
  Ljung–Box em `r` e `r²` (autocorrelação e clusters de volatilidade, lags 5/10/20);
  ARCH-LM de Engle; assimetria e curtose em excesso com SE assintótico.
- Saída como dicionário com estatística, p-valor, hipótese nula em texto e a
  **advertência** de que p-valores múltiplos não estão corrigidos (ofereça Holm opcional).
- **Testes:** dados sintéticos normais → p-valores não sistematicamente baixos (taxa
  de rejeição em 500 réplicas ≈ α ± 3 SE binomial); dados GARCH sintéticos →
  ARCH-LM rejeita na maioria (> 80%, justificável pelo poder sob os parâmetros escolhidos).

### E7.2 Incerteza de `mu` e `sigma`
- `SE(μ̂_log) = σ_d·√252/√n` em ano⁻¹ (observação: com 3 anos de dados, o SE do
  drift anual é ~σ/√3, enorme); `IC(σ)` via qui-quadrado sob normalidade e
  via bootstrap por blocos (robusto à dependência), reusando a lógica de
  `forecasting.uncertainty` se aplicável.
- **Propagação ao risco:** Monte Carlo "aninhado" ou *parameter bootstrap*. Sorteie
  `(μ, σ)` da distribuição amostral, calcule as métricas analíticas de
  `calibration_sensitivity` e reporte a faixa. O relatório distingue claramente:
  erro MC (E1) × incerteza de calibração (E7.2) × erro de modelo (E7.1).

### E7.3 Cache e reprodutibilidade de dados
- Cache local de snapshots por `(ticker, period, retrieved_at)` com sha256 (o
  formato já existe em `historical_snapshot`). O comando `--offline` falha se não houver
  cache, **sem** cair silenciosamente para a rede.
- Provedor abstrato (`PriceProvider` com `fetch(ticker, period) -> pd.Series`)
  para permitir CSV local como fonte (`--prices-csv path`), com validação por
  `validate_prices`. yfinance continua como padrão.

---

## E8 — Desempenho e memória

- **E8.1 Benchmarks reproduzíveis:** `benchmarks/` (novo) com scripts que medem
  tempo e pico de memória (`tracemalloc`) para `generate_paths`,
  `generate_terminal`, Merton e Heston em tamanhos fixos. Saída em JSON com
  ambiente. **Um benchmark por vez** (AGENTS.md). Não entra no CI padrão.
- **E8.2 Streaming de estatísticas:** para métricas apenas terminais em modelos
  dependentes de caminho (Heston, saltos), adicione `generate_terminal` em lotes
  que mantém apenas o estado `(n_batch,)` por passo. A memória fica
  O(batch_size), não O(steps·n_sims).
- **E8.3 Numba opcional:** extra `perf = ["numba"]`; kernels com *fallback*
  NumPy e teste de equivalência de **lei** (não bits) entre os dois caminhos.
  Só começa depois de E8.1 mostrar onde está o gargalo.
- **E8.4 float32 opcional:** somente se E8.1 mostrar ganho, com análise de erro
  (log-preço acumulado em float32 perde ~7 dígitos) e testes de finitude.
- **Aceitação:** nenhuma otimização entra sem benchmark antes/depois e sem os
  testes de lei correspondentes passando.

---

## E9 — CLI, relatórios e visualização

### E9.1 Subcomandos com compatibilidade
- **Dono:** principal (`main.py`), com possível transferência ao `frontend`.
- Estrutura: `main.py simulate|price|risk|calibrate|diagnose|replay`.
  **Sem subcomando**, o comportamento atual (flags planas, incluindo `--replay`
  e `--next-session`) continua idêntico. Todos os testes de `tests/test_main.py`
  passam sem modificação.
- Novas flags de `simulate`: `--model {gbm,t,merton,kou,heston,garch-t}`,
  `--variance-reduction {none,antithetic,control,qmc}` e `--assets T1,T2,...`
  com `--correlation-from-history` (E6).
- Arquivo de configuração opcional `--config run.toml` (`tomllib` em 3.11+;
  em 3.10, `tomli` só como dependência condicional; decisão do principal).
  Precedência: CLI > arquivo > padrão, registrada no relatório.

### E9.2 Schema v2 dos relatórios
- `schema_version: 2`; `model` ∈ {`single_asset_gbm`, `gbm_t_shocks`,
  `merton_jump`, `kou_jump`, `heston`, `garch_t`, `multi_asset_gbm`,
  `black_scholes_mc`}; `model_parameters` aninhado por modelo;
  `sampling` = {`variance_reduction`, `rng`, `qmc_scrambles`, `discretization`,
  `exact_on_grid`}; `results.risk_intervals` (E1); `measure` (`physical` |
  `risk_neutral`).
- `read_report` aceita v1 e v2, e o replay de v1 continua produzindo números idênticos
  (teste de regressão com um relatório v1 de fixture).
- Publique um JSON Schema em `docs/report-schema-v2.json`, validado nos testes sem
  dependência nova (validação manual de chaves e tipos) ou com `jsonschema` como extra de dev.

### E9.3 Gráficos
- **Dono:** `frontend`, em `src/simulation/plotting.py` e arquivos novos de plot.
  Siga `research/notes/2026-11-18-dark-chart-design.md` e
  `tests/test_plotting_design.py`.
- Novos: fan chart (percentis 5/25/50/75/95 por data); histograma de perdas com
  VaR/CVaR e seus ICs (E1) como faixas; convergência comparando técnicas de
  redução de variância (E2) com bandas ±2·SE; superfície de preço/vol implícita
  (E3); mapa de calor de correlação (E6); QQ-plot dos log-retornos vs. normal
  e t (E7.1).
- Os testes verificam estrutura (eixos, rótulos, legendas, cores do design) com
  `MPLBACKEND=Agg`, sem comparação de pixels.

### E9.4 Dashboard
- **Somente se o usuário pedir** (AGENTS.md). Fica fora deste plano.

---

## E10 — Engenharia, CI e documentação

- **E10.1 Marcadores de teste:** `slow` (cobertura empírica com muitas réplicas)
  e `network` em `pyproject.toml`. CI padrão: `-m "not slow and not network"`;
  um job semanal opcional (`schedule`) roda `slow`. Os testes de cobertura de E1/E2/E3
  ganham uma versão rápida (poucas réplicas, tolerância maior) e outra `slow`.
- **E10.2 Lint progressivo:** habilitar `I` (isort) em commit dedicado,
  conforme a nota do `pyproject.toml`; `check_untyped_defs` primeiro em
  `src/pricing` e `src/simulation` novos (`[[tool.mypy.overrides]]`).
- **E10.3 Documentação:** cada épico atualiza o README (seção própria e
  roadmap) e cria uma nota em `research/notes/AAAA-MM-DD-<épico>.md` com objetivo,
  metodologia, testes **realmente executados** e limitações. Registro no vault Obsidian
  conforme AGENTS.md; se o vault estiver inacessível, deixe um rascunho em `research/notes/`.
- **E10.4 Licença:** o README diz que não há licença. Isso é uma decisão do usuário e
  não deve ser implementado por agente.

---

## 14. Matriz de propriedade de arquivos

| Arquivo / diretório | Escritor padrão | Tarefas que o tocam | Observação |
| --- | --- | --- | --- |
| `main.py` | principal | E3.8, E9.1 | Transferência explícita ao `frontend` se delegado |
| `src/simulation/report.py` | principal | E1.4, E9.2 | Serializar tarefas que alteram schema |
| `src/simulation/base.py` | `backend_quant` (1 por vez) | E6.1 (`_check_memory`) | Mudança mínima, retrocompatível |
| `src/simulation/gbm.py` | `backend_quant` | E2.1, E2.3, E3.3 | **Serializar** E2.1 → E2.3 |
| `src/simulation/variance_reduction.py` (novo) | `backend_quant` | E2.* | — |
| `src/simulation/jump_diffusion.py` (novo) | `backend_quant` | E4.1, E4.2 | — |
| `src/simulation/heston.py`, `garch.py` (novos) | `backend_quant` | E5.1, E5.3 | — |
| `src/simulation/multi_asset.py` (novo) | `backend_quant` | E6.1, E6.3 | — |
| `src/simulation/plotting.py` | `frontend` | E9.3 | — |
| `src/risk/statistics.py` | **congelado** | — | Mudança só com escopo do usuário |
| `src/risk/intervals.py`, `decomposition.py` (novos) | `backend_quant` | E1, E6.4 | — |
| `src/pricing/**` (novo) | `backend_quant` | E3.* | — |
| `src/pricing/model_comparison.py` (novo) | `backend_quant` | E3.9.2, E3.9.3 | Importa `backtest.*` sem alterar |
| Plot interativo do comparador (novo) | `frontend` | E3.9.4 | Plotly como extra opcional |
| `src/validation/statistics.py` | `backend_quant` | E1.2 (`terminal_quantile`), E2.4 | Só adições |
| `src/validation/{jump,heston}_theory.py`, `diagnostics.py` (novos) | `backend_quant` | E4, E5, E7.1 | — |
| `src/calibration/{jumps,multi_asset}.py` (novos) | `backend_quant` | E4.3, E6.2 | — |
| `src/calibration/data.py` | principal | E6.2, E7.3 | Rede: nunca em testes padrão |
| `src/backtest/**` | **congelado** | — | Resultados publicados dependem dele |
| `tests/test_<novo>.py` | `testador_quant` | todas | O implementador pode criar teste mínimo; o testador amplia |
| `research/experiments/**`, `research/tail-risk/**` | **congelado** | — | Artefatos publicados |
| `README.md`, `pyproject.toml`, `ci.yml`, `AGENTS.md` | principal | E9, E10 | — |
| `docs/plano-expansao.md` (este) | principal | — | Atualizar status ao fechar tarefas |

---

## 15. Modelo de tarefa delegada (copiar e preencher)

```markdown
### Tarefa <ID> — <título>
- Papel: <pesquisador_quant | backend_quant | frontend | testador_quant | revisor_quant | corretor>
  (delegação nativa | delegação por instruções: incluir developer_instructions de .codex/agents/<papel>.toml)
- Somente leitura: <sim/não>
- Objetivo: <uma frase>
- Evidência de entrada: commit <hash>; seções deste plano: <§>; arquivos lidos: <...>
- Arquivos permitidos: <lista exata>
- Arquivos proibidos: <lista; no mínimo os congelados de §14>
- Dependências: <IDs concluídos e revisados>
- Especificação: <copiar a subseção do plano; registrar desvios propostos, sem aplicá-los>
- Testes exigidos: <lista>
- Critérios de conclusão: <lista verificável>
- Orçamento: memória máx. <X MiB>; uma suíte completa por vez; sem rede
- Retorno obrigatório: Papel e objetivo · Resultado e evidência · Alterações ·
  Validação (comandos e saídas) · Pendências
```

**Status das tarefas** (o principal atualiza; não marcar sem revisão independente):

| ID | Status | Commit | Revisado por | Notas |
| --- | --- | --- | --- | --- |
| E1.1 | concluído | `764c1e7` + `3ca0d5d` | `revisor_quant` (P3 corrigidos) | Wilson/Clopper-Pearson; cobertura exata 0,9547, empírica 0,9485 |
| E1.2 | concluído | `3c49052` + `1c6fa80`, `64eeb51` | `revisor_quant` (P3 corrigidos) | Estatísticas de ordem com `coverage` exata e `saturated`; `terminal_quantile` só por adição |
| E1.3 | concluído | `6312d1c` + `64eeb51` | `revisor_quant` (P3 corrigidos) | Assintótico e bootstrap em lotes; desvio: `spawn_key=(crc32(b"cvar_bootstrap"),)` (NumPy rejeita string) |
| E1.4 | concluído | `459a4d6` (outra sessão) + `5530b79`, `f522b73` | `revisor_quant` (P2 da CLI corrigido) | Campo opcional da v1; P3 pendentes: `unit`/`parameter_scope` (E9.2) e cobertura com `.1%` na CLI |
| E2.1 | pendente | — | — | — |
| E2.2 | pendente | — | — | — |
| E2.3 | pendente | — | — | — |
| E2.4 | pendente | — | — | — |
| E3.0 | concluído | `e8ac17b` (correções do plano) | principal | Parecer do `pesquisador_quant` (somente leitura); fontes primárias não lidas no original |
| E3.1 | concluído | `1223d8a` + testes | `revisor_quant` (sem achados de fórmula; lacunas P2 de teste fechadas e mutantes detectados) | `src/pricing/black_scholes.py`; P3 de robustez em extremos pendentes |
| E3.2 | concluído | `1223d8a` + testes | `revisor_quant` | Brent; ida e volta < 1e-8 onde bem condicionado |
| E3.3–E3.8 | **bloqueado** | — | — | Aguardam E2.1/E2.2 concluídos e revisados |
| E3.9.0–E3.9.5 | **bloqueado** | — | — | Decisão do usuário: esperar E2; depende também de E3.3 e E5.3 |
| E4.1–E4.4 | pendente | — | — | — |
| E5.1, E5.3 | pendente | — | — | E5.2 só parecer |
| E6.1–E6.4 | **bloqueado** | — | — | Requer aprovação do contrato 3D |
| E7.1–E7.3 | pendente | — | — | — |
| E8.1–E8.4 | pendente | — | — | E8.3/E8.4 condicionados a E8.1 |
| E9.1–E9.3 | pendente | — | — | E9.4 fora de escopo |
| E10.1–E10.3 | pendente | — | — | E10.4 decisão do usuário |

---

## 16. Referências primárias

- Glasserman, P. *Monte Carlo Methods in Financial Engineering*. Springer, 2003
  (antitéticas, controle, QMC e ponte browniana, gregas, americanas).
- Hull, J. *Options, Futures, and Other Derivatives* (BSM, gregas, exemplos numéricos).
- Haug, E. G. *The Complete Guide to Option Pricing Formulas*, 2ª ed. (barreiras,
  lookback, asiáticas geométricas).
- Kemna, A.; Vorst, A. "A pricing method for options based on average asset
  values". *J. Banking & Finance*, 1990.
- Broadie, M.; Glasserman, P.; Kou, S. "A continuity correction for discrete
  barrier options". *Math. Finance*, 1997.
- Longstaff, F.; Schwartz, E. "Valuing American options by simulation".
  *Rev. Financial Studies*, 2001.
- Merton, R. C. "Theory of rational option pricing". *Bell J. Econ. Manage. Sci.*
  4(1), 1973 (BSM com dividendo contínuo; base de E3.1).
- Merton, R. "Option pricing when underlying stock returns are discontinuous".
  *J. Financial Economics*, 1976.
- Kou, S. "A jump-diffusion model for option pricing". *Management Science*, 2002.
- Heston, S. "A closed-form solution for options with stochastic volatility".
  *Rev. Financial Studies*, 1993.
- Albrecher, H. et al. "The little Heston trap". *Wilmott Magazine*, 2007.
- Andersen, L. "Simple and efficient simulation of the Heston stochastic
  volatility model". *J. Computational Finance*, 2008.
- Lord, R.; Koekkoek, R.; van Dijk, D. "A comparison of biased simulation
  schemes for stochastic volatility models". *Quantitative Finance*, 2010.
- Higham, N. "Computing the nearest correlation matrix". *IMA J. Numer. Anal.*, 2002.
- Brazauskas, V.; Jones, B.; Puri, M.; Zitikis, R. "Estimating conditional
  tail expectation with actuarial applications in view". *J. Stat. Planning
  and Inference*, 2008.
- Kupiec, P. (1995); Christoffersen, P. (1998); Acerbi, C.; Székely, B. (2014) —
  já usados em `src/backtest/metrics.py`.

> Antes de usar uma fórmula desta lista em código, o `pesquisador_quant` ou o
> implementador deve conferi-la na fonte primária e citar seção ou equação na docstring.
> Este plano resume as fórmulas e pode conter erros de transcrição. A verificação
> por teste contra referência independente é obrigatória.
