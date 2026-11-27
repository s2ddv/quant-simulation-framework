---
registro_id: quant-simulation-framework-e5-volatilidade-estocastica-2026-11-24
projeto: quant-simulation-framework
data: 2026-11-24
tipo: nota-de-epico
epico: E5
branch: claude/serene-clarke-qqq11r
---

# E5 — Volatilidade estocástica (Heston) e GARCH(1,1)-t como simulador

Especificação: [plano de expansão, §E5](../../docs/plano-expansao.md).
Relacionadas: [E2 — redução de variância](2026-11-23-e2-reducao-variancia.md),
[E3 — precificação](2026-11-22-e3-precificacao.md).

A equipe de `AGENTS.md` trabalhou em **delegação por instruções**: não havia
perfis nativos, e as `developer_instructions` de `.codex/agents/*.toml` foram
copiadas em cada tarefa. Nunca houve mais de dois especialistas ativos, e cada
arquivo teve um único escritor por vez. O fluxo foi este:

1. `pesquisador_quant` (somente leitura) deu o parecer.
2. `backend_quant` implementou.
3. `testador_quant` e `revisor_quant` atuaram sobre um commit estável.
4. `corretor` corrigiu os achados confirmados.
5. O mesmo revisor fez nova revisão.

## Objetivo e hipóteses

- **E5.1:** simular o Heston (1993) sob P (ou sob Q com `mu = r − q`) com dois
  esquemas, QE de Andersen (2008) e full truncation de Lord et al. (2010),
  declarando o erro de discretização. Validar contra a teoria: momentos CIR,
  martingale e preço europeu semi-analítico pela função característica
  "little Heston trap" (Albrecher et al. 2007).
- **E5.2:** apenas parecer de calibração, sem implementação.
- **E5.3:** expor o GARCH(1,1)-t de `src/backtest/garch.py` como simulador de
  trajetórias `(steps+1, n_sims)`, sem duplicar nem alterar o MLE. O `mu` é a
  localização do log-retorno **diário**.
- **Hipóteses:**
  - O QE casa média e variância condicionais do CIR, então `E[v_t]` e
    `Var[v_t]` são exatos na grade.
  - A correção de martingale (K0*) dá `E[S_T] = S0·e^{mu·T}` exato.
  - O viés do QE-M em preços é pequeno (centavos) com Δt ≤ 1/4.
  - O viés do full truncation é grande e cai com Δt.

## Parecer do pesquisador (resumo)

**Fontes.** As fontes primárias **não foram lidas no original**. O proxy bloqueou
os PDFs de Andersen e do arXiv, e RFS, Wilmott e Risk.net exigem assinatura.
Por isso as fórmulas vieram de memória e foram verificadas numericamente. A
garantia matemática vem da derivação independente do revisor:
- EDO de Riccati para a função característica;
- quadratura de `E[exp(A v′)|v]` para K0*;
- Monte Carlo.

**Erros de transcrição no plano, corrigidos no parecer:**

1. `quad`/Gauss–Laguerre foi substituído por Gauss–Legendre composto em painéis
   geométricos com parada adaptativa. O backend acrescentou bissecção por
   painel, porque o GL fixo errava ~1e-5 fora do dinheiro com κ = 5, ξ = 3,
   ρ = −0,99.
2. Não é viável medir a redução de viés entre dois Δt no QE-M, porque o viés
   fica abaixo do SE. O teste de redução usa o full truncation. No QE-M só se
   exige a faixa `4·SE + viés tolerado` em dois Δt.
3. O limite `ξ → 0` precisa de um ξ mínimo (1e-4) e de tolerância O(ξ). `ξ = 0`
   tem ramo-limite exato.
4. `E[S_T] = S0·e^{muT}` é exato com QE-M e com o full truncation (log-Euler).
   Sem a correção, é só aproximado.
5. A matriz de variância do full truncation devolve `v⁺ = max(ṽ, 0)`.
6. **E5.3:** com `h1 = h̄` a variância incondicional vale em todo t. O teste
   exige `ν > 4` e a condição de quarto momento `β² + 2αβ + α²κ_z < 1`.
7. **E5.3:** `backtest.garch` depende de SciPy. Com inovações t,
   `E[S_T] = ∞`, porque a t não tem função geradora de momentos.

**Fórmulas conferidas como corretas:** K0–K4 com γ1 = γ2 = ½, o QE com
ψc = 1,5, K0* com as condições `A < 1/(2a)` e `A < β`, e `Var[v_t]` do plano.

## Implementação

| Arquivo | Conteúdo |
| --- | --- |
| `src/simulation/heston.py` | `HestonSimulator(BaseStochasticSimulator)`: `scheme ∈ {"andersen_qe", "full_truncation"}`, `martingale_correction`, ramo exato com ξ = 0, `generate_paths`, `generate_paths_and_variance`, `generate_terminal` em lotes, RNG `SeedSequence(seed).spawn(3)` ([0] normais da variância, [1] uniformes do QE, [2] normais do preço), `model_info()` com `exact_on_grid: false` e Feller reportado, `calibrate` → `NotImplementedError` |
| `src/validation/heston_theory.py` | Momentos CIR, `feller_condition`, CF "little trap" estável, `heston_price` (Lewis), `heston_price_gil_pelaez` (verificação), limite com ξ = 0 por BS e variância integrada, rejeição de 0 < ξ < 1e-4 |
| `src/simulation/garch.py` | `GarchTSimulator(BaseStochasticSimulator)`, que importa `fit_garch_t` e cuja recursão é idêntica à de `simulate_cumulative` (mesma ordem de sorteio). Inclui `from_days`, `generate_paths_and_variance`, `calibrate`, `model_info()` e o gancho `_drift_adjustment` (aditivo e previsível; insuficiente para E3.9 com inovações t) |

**Desvios aceitos em relação ao plano:**
- `return_variance=True` virou o método `generate_paths_and_variance()`. A
  base exige que `generate_paths` devolva um `ndarray`, e o mypy rejeita outro
  tipo de retorno.
- A quadratura em NumPy substitui `quad`; o SciPy só aparece nos testes.

## Testes executados e resultados

Ambiente: Python 3.13, NumPy 2.5.3, SciPy 1.18.1, `.venv` local com as dependências já
declaradas no `pyproject.toml` (instalação autorizada pelo usuário). Os valores abaixo
foram medidos pelo `testador_quant` em `321d90f` (Heston) e `2386cfe` (GARCH). A
correção `d581a89` muda a saída do QE em no máximo 4e-13 relativo para a mesma
semente. Todos os valores vêm dos próprios testes.

### Validação completa (commit `d581a89`)

- `MPLBACKEND=Agg .venv/bin/python -m pytest -q -p no:cacheprovider`:
  **2714 passed**, 51 subtests, em 348 s (antes do E5: 2432 passed em 185 s).
  Os testes `slow` rodam por padrão.
- `.venv/bin/python -m ruff check`: All checks passed.
- `.venv/bin/python -m mypy`: no issues found in 84 source files.

### Heston — teoria (determinístico)

- Casos I–III (S0 = 100, r = q = 0, v0 = θ, K = 70/100/140): Lewis contra a
  tabela, ≤ 4,3e-8. Contra `scipy.integrate.quad` (Gil-Pelaez escrito no
  teste), ≤ 9e-14. Lewis contra Gil-Pelaez e contra nodes = 64/tol = 1e-14,
  ≤ 1e-7 (asserção).

  | Caso | κ | θ = v0 | ξ | ρ | T | C(70) | C(100) | C(140) |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | I | 0,5 | 0,04 | 1,0 | −0,9 | 10 | 35,8497697 | 13,0846701 | 0,2957744 |
  | II | 0,3 | 0,04 | 0,9 | −0,5 | 15 | 37,1696647 | 16,6492229 | 5,1381905 |
  | III | 1,0 | 0,09 | 1,0 | −0,3 | 5 | 38,7720441 | 21,7952877 | 9,9830678 |

- Revisão independente:
  - A CF bate com a EDO de Riccati (DOP853, rtol 1e-12) a ≤ 3e-12 relativo
    em 12 conjuntos de parâmetros, nas retas u, u−i/2 e u−i.
  - Bate com uma referência em `clongdouble` a 2e-14.
  - Busca aleatória de 4000 conjuntos de parâmetros: max |arg| do quociente
    do log = 2,13 rad. O corte de ramo nunca foi cruzado; é evidência
    numérica, não prova.
  - Grade de estresse de 360 preços contra `quad`: desvio máximo 3,5e-9.
  - K0* conferido por quadratura de `E[exp(A v′)|v]` a 2,4e-15.
- Limite BS (v0 = θ = 0,04, κ = 1,5, ρ = −0,5, T = 1): erro O(ξ). Com
  ξ = 1e-2, até 0,023; com ξ = 1e-4, até 2,3e-4. Tolerância 5ξS0.

### Heston — Monte Carlo

Os preços usam variável de controle S_T, com média `S0·e^{(r−q)T}` exata no
QE-M e no FT. O SE é o do resíduo. Tolerância: 4·SE + viés declarado (0,15 com
Δt = 1/4, 0,10 com Δt ≤ 1/8).

**Preços QE-M, modo rápido** (N = 100k, sementes 9101–9106):

| Caso | Δt | erro K=70 (z) | erro K=100 (z) | erro K=140 (z) |
| --- | --- | --- | --- | --- |
| I | 1/4 | −0,0627 (−2,33) | −0,0212 (−0,78) | +0,0099 (1,20) |
| I | 1/8 | −0,0017 (−0,06) | −0,0039 (−0,14) | −0,0111 (−1,52) |
| II | 1/4 | −0,0984 (−2,29) | −0,1443 (−2,39) | −0,0960 (−1,45) |
| II | 1/8 | −0,0310 (−0,71) | −0,0181 (−0,29) | −0,0106 (−0,16) |
| III | 1/4 | +0,0746 (1,67) | +0,0454 (0,69) | +0,0711 (0,90) |
| III | 1/8 | +0,0342 (0,77) | +0,1078 (1,65) | +0,1374 (1,74) |

**Preços QE-M, modo lento** (N = 1e6 em lotes de 100k, sementes 9400–9421).
Todos ficam a |z| ≤ 2,2; SE ≈ 0,0085 no caso I e 0,014–0,026 nos casos II e III.

| Caso | Δt | erro K=70 (z) | erro K=100 (z) | erro K=140 (z) |
| --- | --- | --- | --- | --- |
| I | 1/8 | −0,0187 (−2,20) | −0,0154 (−1,79) | −0,0003 (−0,12) |
| I | 1/32 | +0,0114 (1,33) | +0,0068 (0,79) | +0,0051 (2,04) |
| II | 1/8 | −0,0065 (−0,46) | +0,0021 (0,10) | +0,0126 (0,54) |
| II | 1/32 | +0,0201 (1,37) | +0,0395 (1,84) | +0,0461 (1,79) |
| III | 1/8 | +0,0068 (0,48) | +0,0030 (0,14) | +0,0189 (0,75) |
| III | 1/32 | +0,0089 (0,63) | +0,0036 (0,18) | +0,0126 (0,51) |

**O viés do QE-M não foi resolvido estatisticamente:** ficou abaixo do SE
em todos os casos. No caso II com Δt = 1/32 os três erros são positivos, mas
dentro da tolerância.

**Caso com r ≠ 0 e q ≠ 0** (dinâmica do caso III, v0 = 0,06, r = 0,03,
q = 0,01, mu = r − q, desconto e^{−rT}). Todos com |z| ≤ 1,33, em Δt = 1/4
(semente 9201) e 1/8 (semente 9202):

| Opção | Referência |
| --- | --- |
| put K=80 | 8,57212 |
| call K=100 | 24,3955 |
| call K=130 | 13,7722 |

**Viés do full truncation** (caso I, K = 100):

| Δt | Viés | SE | N | Semente |
| --- | --- | --- | --- | --- |
| 1/4 | +2,0889 | 0,033 | 1e5 | 9301 |
| 1/8 | +1,1408 | 0,030 | 1e5 | 9302 |
| 1/4 | +2,0459 | 0,0105 | 1e6 | 9501 |
| 1/32 | +0,2567 | 0,0088 | 1e6 | 9502 |

A redução foi verificada com folga, sem exigir monotonicidade: 1/8 contra
1/4 < 0,75; 1/32 contra 1/4 ≈ 0,125 < 0,5.

**Momentos de v no QE** (N = 50k, 20 passos):
- Caso I (Feller violado, T = 2, semente 11), em t = 0,5 / 1 / 2:
  - média: z = −0,67 / −1,05 / −1,62;
  - variância: z = 0,32 / −0,40 / −1,56 (em t = 2: 0,03233 contra
    0,03459, SE 0,00144).
- κ = 2, θ = 0,04, ξ = 0,3, v0 = 0,09 (semente 12): |z| ≤ 1,60.
- No FT há viés real contra o CIR (z até 18 com Δt = 1/20). Ele fica
  coberto pelo termo declarado |Euler − CIR|; contra a recursão exata de
  Euler, |z| ≤ 1,55.

**Limite ξ → 0** (v0 = θ = 0,04, μ = 0,05, T = 1, 50 passos, N = 50k,
semente 2024):

| ξ | z da média de ln(S_T/S0) | z da variância | p do KS |
| --- | --- | --- | --- |
| 1e-3 | 0,08 | 1,20 | 0,955 |
| 1e-4 | 0,08 | 1,17 | 0,924 |
| 0 (ramo exato) | — | — | 0,147 |

No modo lento (ξ = 1e-3, N = 1e6, semente 5): KS p = 0,548 no QE e 0,559 no FT.

**Martingale E[S_T] = S0·e^{μT}** (T = 2, μ = 0,07, Δt = 1/4, N = 200k,
referência 115,027):

| Esquema | ρ | z |
| --- | --- | --- |
| QE-M | −0,9 | 0,84 |
| FT | −0,9 | −0,15 |
| QE-M | 0,5 | −1,35 |
| FT | 0,5 | −0,11 |

- ρ = ±1 (N = 200k, semente 42): |z| ≤ 2,14.
- Caso I com T = 10 e N = 1e6: QE-M z = 1,96; FT z = 0,26.
- Sem a correção de martingale (v0 = 0,2, κ = 2, θ = 0,04, ξ = 0,3,
  ρ = −0,9, Δt = 1): E[S] = 85,86, a −264 SE. Com QE-M: +0,003 (0,04 SE).
  A medição é do revisor.

**Contratos:**
- Feller violado: tudo finito e variância ≥ 0 nos dois esquemas.
- Um teste determinístico de um passo, com stubs dos três geradores, cobre
  os 4 regimes de ψ, ρ ∈ {−0,9; 0,5} e a correção ligada e desligada. Ele
  mata os mutantes K1 com sinal trocado, K0* omitido e ψc ∈ {0,5; 1,0}.
- Memória: pico/estimativa (tracemalloc) entre 0,80 e 0,999.

### GARCH(1,1)-t

- **Igualdade com `simulate_cumulative`:** bit a bit pelo log interno, em 3
  configurações (h0 incondicional, do usuário e calibrado). Nos preços,
  atol 1e-12.
- **Variância incondicional** (ω = 2e-6, α = 0,05, β = 0,90, ν = 8,
  μ = 2e-4; σ̄² = 4e-5 por dia). A condição de quarto momento vale:
  κ_z = 4,5 e β² + 2αβ + α²κ_z = 0,91125 < 1.

  | Teste | Semente | N | Dias | h0 | Estatística | Estimativa | SE | z |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | rápido | 101 | 12000 | 1000 | σ̄² | E[e²] | 4,1201e-5 | 7,155e-7 | +1,68 |
  | rápido | 101 | 12000 | 1000 | σ̄² | E[h] | 4,0119e-5 | 1,134e-7 | +1,05 |
  | rápido | 102 | 12000 | 1000 | 4σ̄² | E[e²] | 3,9530e-5 | 7,290e-7 | −0,64 |
  | rápido | 102 | 12000 | 1000 | 4σ̄² | E[h] | 4,0014e-5 | 1,165e-7 | +0,12 |
  | lento | 501–505 | 150000 | 400 | σ̄² | E[e²] | 4,00745e-5 | 2,053e-7 | +0,36 |
  | lento | 501–505 | 150000 | 400 | σ̄² | E[h] | 4,00132e-5 | 3,30e-8 | +0,40 |

- **Previsão de variância** E[h_{1+k}] = σ̄² + 0,95^k(h0 − σ̄²), com
  h0 = 6σ̄², semente 303, N = 20000: |z| ≤ 1,66.
- **Falso alarme da tolerância de 4·SE**, medido pelo revisor: 250 réplicas
  deram sd(z) = 1,00 em e² e 1,04 em h, com max|z| = 3,26.
- **Mutantes:** 27 na primeira revisão, 25 mortos (os 2 sobreviventes eram
  equivalentes ou ganharam teste). Na nova revisão, 7 de 7 mortos.
- **Rejeições:** ν ≤ 2, α + β ≥ 1, ω ≤ 0, valores bool, nan/inf e
  horizonte ≠ steps/252 levantam ValueError.

## Limitações

- **O Heston tem erro de discretização.** Não é exato na grade
  (`exact_on_grid: false`). O viés do QE-M ficou abaixo do SE em todos os
  casos medidos, mas a tolerância declarada (0,15 com Δt = 1/4 e 0,10 com
  Δt ≤ 1/8, em S0 = 100) é um valor de projeto, sem prova de limite. O full
  truncation tem viés grande: +2,0 no caso I com K = 100 e Δt = 1/4.
- **Os benchmarks de Andersen** (parâmetros e preços) foram transcritos de
  memória. Só o caso I com K = 100 (13,0847) foi conferido contra um valor
  publicado lembrado. Os preços de referência vêm do próprio pricer, que
  concorda com `scipy.integrate.quad` e com a EDO de Riccati.
- **A calibração do Heston (E5.2) não foi implementada.** Os parâmetros são
  fornecidos pelo usuário. `calibrate` levanta `NotImplementedError`.
- **No GARCH, `mu` é a localização diária do log-retorno**, não o drift anual
  do preço da §2.1. Com inovações t, `E[S_T] = ∞` (`model_info` traz
  `terminal_price_mean_finite: false`). VaR, CVaR e quantis estão bem
  definidos.
- **O gancho `_drift_adjustment`** só permite regras aditivas e previsíveis.
  A risco-neutralização (E3.9.0/E3.9.1) exige outro desenho, com revisão.
- **A memória** é estimada com uma folga medida em NumPy 2.5.3 e Python 3.13.
  A matriz do CI (Python 3.10/3.12 e outras versões do NumPy) não foi
  verificada.
- **Não há integração com a CLI nem com o relatório JSON**, que pertencem ao
  E9 (pendência).

## Pendências de revisão (P3 opcionais, não bloqueantes)

- A folga de memória do Heston em grade longa é fina (razão 0,999, cerca de
  6 vetores). Uma versão diferente do NumPy pode consumi-la; o teste acusa.
- O teste de Gil-Pelaez com κ < ρξ pode apertar a tolerância para cerca de
  1e-8, para distinguir `E = exp(−dT)` direto.
- `heston_price` aceita só `K` escalar. Com κ ≤ 1e-200 o QE levanta
  ValueError (underflow de m²).
- O registro no vault Obsidian não foi feito: `~/.codex/quant-obsidian.json`
  não existe neste ambiente. Esta nota é o rascunho.

## Próximos passos

- **E9:** subcomando ou flag de modelo, `model: "heston"` e `"garch11_t"` no
  schema v2, com `discretization` e `exact_on_grid`.
- **E3.6/E3.9:** precificação Heston sob Q, comparador com o GARCH-t e o
  desenho do gancho Q.
- **E5.2:** decisão do usuário (ver recomendação abaixo).
- **E8:** `generate_terminal` do Heston em fluxo já evita a matriz completa. Um
  benchmark pode ser feito depois.

## Recomendação E5.2 (parecer, sem implementação)

**Não implementar a calibração histórica do Heston agora.**
- A variância é latente: a verossimilhança diária exige filtragem.
- κ e ξ ficam fracamente identificados com dados diários e correlacionados
  entre si.
- Dados históricos dão parâmetros sob P, e o prêmio de risco de volatilidade
  só se identifica com opções.

**Caminhos possíveis:**
- **Precificação:** calibrar a cotações de opções (opção A), o que exige uma
  fonte de dados de opções.
- **Risco sob P com dados intradiários:** GMM com variância realizada
  (Bollerslev & Zhou 2002). É o caminho mais barato e defensável.
- **Alternativas:** a MLE simulada ou o filtro de partículas é cara e tem
  verossimilhança ruidosa. O MCMC é muito caro. O QMLE/Kalman é enviesado. O
  mapeamento GARCH → Heston é só heurístico.

**Recomendação de uso imediato:** Heston com parâmetros fornecidos pelo
usuário, mais análise de sensibilidade.
