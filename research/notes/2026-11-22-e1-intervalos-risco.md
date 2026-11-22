---
registro_id: quant-simulation-framework-e1-intervalos-risco-2026-11-22
projeto: quant-simulation-framework
data: 2026-11-22
tipo: nota-de-epico
epico: E1
branch: claude/serene-clarke-qqq11r
---

# E1 — Incerteza Monte Carlo das métricas de risco

Especificação: [plano de expansão, §E1](../../docs/plano-expansao.md).
Execução pela equipe de agentes de `AGENTS.md`, em **delegação por
instruções** (não havia perfis nativos): `backend_quant` implementou,
`testador_quant` e `revisor_quant` atuaram em paralelo sobre commits estáveis
e o `corretor` tratou apenas achados confirmados.

## Objetivo

`risk_summary` devolve estimativas pontuais de `prob_loss`, VaR e CVaR. O
épico acrescenta intervalos de confiança do **estimador Monte Carlo** (erro
amostral com `s0`, `mu`, `sigma` e horizonte fixos). Eles são distintos de:

- percentis da distribuição de perdas (o VaR já é um percentil);
- incerteza de calibração (`calibration_sensitivity`);
- erro de modelo (adequação do GBM).

`src/risk/statistics.py` não foi alterado. Os testes existentes passam sem
modificação.

## Implementação

| Tarefa | Função | Arquivo | Commits |
| --- | --- | --- | --- |
| E1.1 | `prob_loss_interval(terminal, s0, level=0.95, method="wilson"\|"clopper_pearson")` | `src/risk/intervals.py` | `764c1e7`, correção `3ca0d5d` |
| E1.2 | `var_interval(terminal, s0, confidence=0.95, level=0.95)`; `terminal_quantile(s0, mu, sigma, horizon, p)` | `src/risk/intervals.py`; `src/validation/statistics.py` (só adições) | `3c49052`, correções `1c6fa80`, `64eeb51` |
| E1.3 | `cvar_interval(..., method="asymptotic"\|"bootstrap", *, n_bootstrap=1000, seed=None, batch_size=None)` | `src/risk/intervals.py` | `6312d1c`, correção `64eeb51` |
| E1.4 | `results.risk_intervals` (campo opcional da v1) e linhas no resumo da CLI | `src/simulation/report.py`, `main.py` | `459a4d6` (outra sessão), `5530b79`, correção `f522b73` |

### Metodologia

Convenções idênticas a `risk_summary`: perda `L = s0 - S_T`; evento de perda
`S_T < s0` (estrito); `VaR = _linear_percentile(L, p)`; `CVaR = média de
L >= VaR` com empates. As funções importam os helpers de `risk.statistics`, e
as estimativas pontuais coincidem com `risk_summary` por igualdade exata.

- **Probabilidade de perda.** `k ~ Binom(n, p)`. Wilson (aproximado) ou
  Clopper-Pearson (exato e conservador), reutilizando
  `validation.tail_probability`.
- **VaR.** Intervalo livre de distribuição por estatísticas de ordem:
  `j = Binom.ppf(a/2, N, p)` e `k = Binom.ppf(1-a/2, N, p) + 1`, recortados
  para `[1, N]`. O resultado informa `lower_rank`, `upper_rank`, `saturated` e
  `coverage = F(k-1) - F(j-1)`, que é a cobertura exata para `L` contínua com
  os postos efetivamente usados, e é conservadora com átomos.
  - O recorte reduz a cobertura **nos dois lados**. O `backend_quant` corrigiu
    o briefing do principal, que dizia o contrário, e o revisor confirmou:
    `N=30`, `p=0,05` → postos `[1, 5]`, cobertura 0,770.
  - O ponto (interpolação linear) difere do posto de ordem em `1-p`, ou seja,
    O(1) em posto e O(1/N) em nível de probabilidade.
- **CVaR assintótico** (Brazauskas et al. 2008; Manistre e Hancock 2005):
  `Var ≈ [Var(L | L >= VaR) + p(CVaR - VaR)^2] / (N(1-p))`, com IC normal.
  - Com `n_tail < 2`, os limites ficam `null` e o resultado traz `note`.
  - O denominador `N(1-p)` segue o plano e é conservador com empates no VaR.
- **CVaR bootstrap.** Bootstrap percentil i.i.d. em lotes, com orçamento de
  aproximadamente 64 MiB e mínimo de uma linha O(N). Os índices são sorteados
  réplica a réplica, de modo que o resultado não depende de `batch_size`.
  - **Desvio do plano:** `SeedSequence(seed, spawn_key=("cvar_bootstrap",))`
    é rejeitado pelo NumPy (`ValueError: unrecognized seed string`). Usa-se
    `spawn_key=(zlib.crc32(b"cvar_bootstrap"),)`, documentado no código.
  - Com `n_tail < 2`, os limites são devolvidos acompanhados de `note`.
- **Relatório.** `results.risk_intervals` tem `interval_kind:
  "monte_carlo_estimator"`, `level` 0,95 (nível do IC, distinto de
  `confidence`, o nível do VaR), Wilson, estatísticas de ordem e CVaR
  assintótico. Não há bootstrap no relatório: o bloco é O(N), determinístico
  e sem RNG adicional.
  - `schema_version` continua 1.
  - `read_report` aceita relatórios v1 antigos sem o bloco.
  - Adulterações dentro do bloco são detectadas no replay.
  - `environment` registra a versão do SciPy.

## Testes realmente executados e resultados

Os testes rodaram em `.venv` local, criada nesta sessão com as dependências já
declaradas em `pyproject.toml` (sem dependência nova): Python 3.13, NumPy
2.5.5, SciPy 1.18.1.

- Suíte completa em `382c153` (antes da correção da CLI): **1755 passed, 51
  subtests passed** em 59,9 s. `ruff check` e `mypy` limpos.
- Validação final sobre `8607172`, que já inclui todas as correções, mais
  as edições do README e da §15:
  `MPLBACKEND=Agg .venv/bin/python -m pytest -q` → **1759 passed, 51 subtests
  passed** em 60,7 s; `ruff check` → "All checks passed!"; `mypy` →
  "Success: no issues found in 62 source files".
- Os arquivos congelados não foram tocados: `git diff 9eb63b1` em
  `src/risk/statistics.py`, `src/backtest`, `research/experiments` e
  `research/tail-risk` sai vazio. Também nenhum teste preexistente foi
  modificado; em `tests/`, o diff só traz arquivos novos.
- Revisão final (`f522b73`): o P2 da CLI foi resolvido. Resta um P3 cosmético
  pendente: a cobertura formatada com `.1%` arredonda 0,9496 para "95.0%".

Coberturas empíricas, com sementes fixas e réplicas formadas por blocos
disjuntos de um único fluxo (não prefixos):

| Estudo | Réplicas | Cobertura | Referência | Critério |
| --- | --- | --- | --- | --- |
| Wilson 95%, N=500, p=0,3 | 2.000 | 0,9485 | exata 0,9547 | [0,93; 0,97] |
| VaR por ordem, N=2.000, p=0,95 | 1.000 | 0,965 | exata 0,960 | ≥ 0,93 |
| VaR saturado, N=10, p=0,95 | 2.000 | 0,396 | informada 0,390 | ±4 SE |
| VaR saturado inferior, N=30, p=0,05 | 2.000 | 0,773 | informada 0,770 | ±4 SE |
| CVaR assintótico, N=2.000, p=0,95 | 500 | 0,948 | — | ≥ 0,90 |
| CVaR bootstrap, N=2.000, p=0,95, B=400 | 500 | 0,938 | — | ≥ 0,90 |

Outras evidências:

- O SE assintótico médio dividido pelo desvio padrão empírico do estimador,
  em 500 réplicas, deu 1,014.
- A fórmula fechada do CVaR do GBM foi conferida contra integração `quad` com
  tolerância relativa de 1e-9.
- A revisão reproduziu o desvio Monte Carlo do CVaR com 2.000 réplicas:
  0,7436 empírico contra 0,7409 de SE médio. Com N(1-p) = 5 pontos de cauda,
  a cobertura cai para 0,867, o que está documentado.

O bootstrap usa B = 400 no teste `slow` por custo: B = 1.000 levaria cerca de
45 s. Os estudos com muitas réplicas têm versão rápida sem marcador e versão
`@pytest.mark.slow`, marcador registrado em `pyproject.toml`. Por ora o
marcador roda por padrão; a exclusão no CI é a tarefa E10.1.

## Limitações

- Os ICs assumem amostras i.i.d. Técnicas antitéticas, de estratificação e de
  QMC (E2) invalidam essa hipótese; será preciso usar pares ou scrambles como
  unidade.
- Com cauda pequena (N(1-p) ≲ 10), os ICs assintótico e bootstrap do CVaR
  podem subcobrir. O IC percentil pode não conter a estimativa.
- Com N pequeno, o IC do VaR satura e sua cobertura real fica abaixo do nível.
  O JSON informa `coverage`, e a CLI marca "(saturado; cobertura real X%)".
- O bloco do relatório não traz `unit` nem `parameter_scope`. A recomendação
  P3 da revisão foi adiada porque dois arquivos de teste fixam o conjunto
  exato de chaves; a docstring e a CLI fazem a distinção.
- Versões anteriores do programa não reproduzem relatórios novos, porque a
  comparação de `results` é por igualdade total. O replay exato também depende
  da versão do SciPy (agora registrada).
- Remover o bloco de um relatório novo o faz passar como v1 antigo. Não há
  falsificação possível, porque o bloco é derivável, mas há omissão.
  Distinguir os dois casos exige `schema_version` 2 (E9.2).
- Outra sessão Claude escreveu em paralelo na mesma branch: a E1.4 em
  `report.py`/`main.py` e arquivos de teste duplicados
  (`tests/test_risk_intervals_cvar.py`, `tests/test_report_intervals.py`). A
  E1.4 foi adotada após teste e revisão independentes nesta sessão; os testes
  duplicados foram mantidos porque passam.
- O vault Obsidian (`~/.codex/quant-obsidian.json`) não existe neste
  ambiente. Esta nota é o registro local.

## Próximos passos

- E10.1: excluir `slow` no CI padrão e criar um job agendado.
- E2: ICs para amostras antitéticas e QMC, com pares ou scrambles como
  unidade.
- E9.2: schema v2 com `unit` e `parameter_scope` no bloco.
- E9.3: histograma de perdas com faixas de IC de VaR/CVaR.
