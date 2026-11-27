# E6 — Multiativos correlacionados e carteiras (2026-11-27)

Links: [[docs/plano-expansao.md#E6]] · [[research/notes/2026-11-22-e1-intervalos-risco]] ·
revisão anterior [[docs/revisoes/2026-11-22-revisao-e1-e3]]

> Registro de trabalho, não previsão. Todos os números abaixo são realizações Monte
> Carlo com semente e N declarados, medidas pelo `testador_quant` sobre o commit
> `d1779b8` (código de simulação inalterado pelas correções posteriores, que só
> tocaram documentação, checagens de entrada e memória do bootstrap).

## Objetivo
Simular GBM multiativo correlacionado com incremento exato, calibrar a partir de
preços históricos de vários ativos, valorar carteiras buy-and-hold e decompor o CVaR
da carteira por ativo, reaproveitando `risk_summary` sem alterá-lo.

**Contrato:** o usuário aprovou em 2026-11-27 o formato 3D
`(steps + 1, n_sims, n_assets)` (§2.2), ao pedir explicitamente a execução do E6. O
formato de ativo único `(steps + 1, n_sims)` não mudou, e o valor da carteira volta a ele.

## Modelo e metodologia
- `dS_i = mu_i S_i dt + σ_i S_i dW_i` com `d⟨W_i,W_j⟩ = ρ_ij dt` (medida física P,
  parâmetros históricos; `mu` em ano⁻¹, `σ` em ano^-1/2, tempo em anos).
- Incremento exato: `Δln S = (mu − σ²/2)Δt + √Δt·σ ⊙ (Z Lᵀ)`, `LLᵀ = R`.
  - Usa Cholesky quando R é definida positiva. Para posto incompleto (ρ = ±1), usa
    `eigh` com recorte dos autovalores em [−1e-12, 0) e normalização das linhas de L.
    A fatoração usada fica registrada em `factorization`.
  - Matriz inválida é rejeitada. `nearest_correlation` (Higham 2002, projeções
    alternadas com correção de Dykstra) é uma função separada e explícita.
- RNG: `default_rng(seed)` local, com um único fluxo de choques. O bootstrap das
  contribuições usa o fluxo `SeedSequence(seed, spawn_key=(crc32("cvar_contribution_bootstrap"),))`.
- Calibração:
  - alinhamento por *inner join* nas datas comuns, com contagem de datas descartadas
    por ativo;
  - `mu_i = 252·mean_i + σ_i²/2`, `σ_i = √252·std_i` (ddof = 1), R pela correlação
    amostral dos log-retornos;
  - Ledoit–Wolf (2004, alvo de correlação constante) é uma função separada.
- Carteira: `V_t = Σ q_i S_i(t)`, sem rebalanceamento. Com `initial_weights`,
  `q_i = w_i·V0/S_i0`.
- Contribuição ao CVaR: `C_i = mean(L_i | L ≥ VaR)`, com o VaR e a máscara de
  `risk_summary` (percentil linear, empates incluídos).
  - Σ C_i = CVaR por linearidade, até o arredondamento do nocional.
  - A leitura de Euler (∂ES/∂u_i, Tasche 2002) exige L contínua perto do VaR. Com
    átomos, o estimador é a TCE.
- Fontes (citadas nas docstrings; **não lidas no original** nesta sessão, verificadas
  por derivação, por implementação independente e por Monte Carlo):
  Glasserman (2003) §3.2.3; Fisher (1915, 1921); Higham (2002) IMA JNA 22(3);
  Ledoit & Wolf (2004) JPM 30(4); Tasche (2002) JBF 26(7); McNeil, Frey & Embrechts (QRM).

## Processo
- O `pesquisador_quant` deu um parecer somente leitura com correções de transcrição
  do plano:
  - a covariância é dos terminais S(T);
  - Fisher-z se aplica aos **log**-retornos;
  - `scipy.linalg.ldl` (Bunch–Kaufman) não serve; a escolha foi Cholesky com fallback
    `eigh`;
  - Higham precisa de pós-processamento;
  - o alvo de LW tem de ser explícito.
- Sequência: `backend_quant` → `testador_quant` ∥ `revisor_quant` → `corretor` (só P3
  confirmados) → nova revisão aprovada.
- Todos atuaram por delegação por instruções, nunca com mais de dois especialistas ao
  mesmo tempo.
- O principal escreveu `src/calibration/data.py`, o README, a §15 e esta nota.

## Testes realmente executados
- `tests/test_multi_asset.py`, `tests/test_multi_asset_calibration.py`,
  `tests/test_risk_decomposition.py` e `tests/test_multi_asset_smoke.py`: 182 passed
  em 6,5 s, testes `slow` incluídos.
- Mutantes: 47 de 48 mortos. O sobrevivente é equivalente: o termo `−s_ii` do θ de LW
  tem média zero exata.
- Suíte completa (commit `6985e47`): `2895 passed, 51 subtests passed` em 255 s.
  `ruff check`: All checks passed. `mypy`: sem problemas em 91 arquivos.
  - Ambiente: venv isolada no scratchpad com `--system-site-packages`, scipy 1.18.1,
    numpy 2.5.3, pandas 3.0.5, ruff 0.17.6 e mypy 2.4.1.
  - O container não tinha scipy, ruff nem yfinance; nada foi instalado no projeto.

Valores medidos. Parâmetros: s0 = [100, 50, 20], mu = [0,05; 0,10; −0,02],
σ = [0,2; 0,35; 0,5], T = 1,5, R = [[1; 0,6; −0,4], [0,6; 1; −0,2], [−0,4; −0,2; 1]].

| Verificação | Semente, N | Resultado |
| --- | --- | --- |
| Fisher-z, log-terminais (paths) | 20261127, 1e5 | r = 0,5995 / −0,3961 / −0,2011; z = −0,24 / 1,48 / −0,36 |
| Covariância terminal vs. analítica | 314, 2e5 | Cov12 406,48 vs 407,17; Cov13 −122,79 vs −121,83; Var3 172,36 vs 171,40; \|z\| ≤ 1,15 (terminal ≤ 1,44) |
| `n_assets = 1` vs. `GbmSimulator` | 101/202, 5e4 | KS de 2 amostras p = 0,512; KS vs. lognormal p = 0,379 |
| Memória (`tracemalloc`) | (3000, 50, 2) | pico 2 594 240 B ≤ estimativa 2 675 136 B |
| Contribuições, 2 ativos i.i.d., pesos 0,5/0,5, V0 = 200, p = 0,95 | 11, 2e4 | C = 27,708 / 28,493; CVaR 56,201; VaR 45,665; erro da identidade 7,1e-15; diferença a 1,19 SE (SE 0,661) |
| SE do bootstrap vs. réplicas MC | 200 réplicas, n = 2000 | razão 0,98 / 1,12 |
| Recuperação da calibração | 2026, n = 20000 | z de mû ≤ 1,07; z de σ̂ ≤ 0,51 |

## Limitações
- Correlação e volatilidade constantes. O GBM conjunto é lognormal com cópula
  gaussiana: não há dependência de cauda nem saltos comuns.
- Carteira buy-and-hold, sem rebalanceamento; V0 ≤ 0 (pesos de soma zero) é rejeitado,
  sem métrica absoluta alternativa.
- A calibração usa só as datas comuns, o que descarta informação de ativos com
  calendários diferentes. O drift histórico é ruidoso e não é risco-neutro.
- Os intervalos e o bootstrap supõem trajetórias i.i.d.
- `validate_correlation` usa tolerância absoluta de −1e-12 no autovalor mínimo e
  rejeita R PSD válida de ordem ≈ 1000 (ex.: `ones((1000, 1000))`).
- A memória O(n²) da validação e da fatoração em `__init__` não entra na checagem de
  orçamento.
- No fallback `eigh_clipped`, as trajetórias de uma mesma semente dependem do LAPACK;
  em `generate_terminal`, `batch_size` muda os valores só pelo arredondamento do BLAS.

## Próximos passos
- **E9:** integrar ao CLI e ao relatório (schema v2, unidades e `factorization`).
  Isso ficou fora do escopo deste épico.
- Decisão do usuário: trocar a tolerância de autovalor por uma relativa (`tol·n`).
- Dependência de cauda (t-cópula), rebalanceamento periódico e confirmação das seções
  citadas nas fontes originais.
