---
registro_id: quant-simulation-framework-atualizacao-notas-2026-11-19
projeto: quant-simulation-framework
data: 2026-11-19
tipo: nota-de-atualizacao
estado: local-sem-commit
---

# Atualização das notas de 17 e 18 de novembro de 2026

As notas abaixo são registros históricos e permanecem sem reescrita de conteúdo.
Esta nota indica o que delas foi superado pela `main` publicada
(`818914c5fcc6f6ebe4c163808790b5daa814fcbf`, conferido com `git ls-remote`
em 2026-11-19). Consulte sempre o código atual antes de usar um registro antigo.

- [2026-11-17-execution-scenarios.md](2026-11-17-execution-scenarios.md)
- [2026-11-18-execution-scenarios-results.md](2026-11-18-execution-scenarios-results.md)
- [2026-11-18-dark-chart-design.md](2026-11-18-dark-chart-design.md)

## Correspondência de commits

As notas citam a branch `codex/obsidian-quant-integration`. Os commits dela
foram reaplicados na `main` com o mesmo conteúdo, verificado por
`git patch-id --stable`:

| Branch de trabalho | `main` | Assunto |
| --- | --- | --- |
| `f5d0b9d` | `7709089` | docs(obsidian): connect simulation research context |
| `871808c` | `d142752` | feat(simulation): calculate 10000 parameter scenarios by default |

O commit `818914c` (leque colorido) existe apenas na `main`.

## Pontos superados

1. **Grade de sensibilidade.** As notas de execução descrevem 5 níveis por
   parâmetro (25 cenários) como padrão da CLI. Desde `d142752`, o padrão de
   `--scenario-levels` é 100 (100 x 100 = 10000 cenários analíticos); as opções
   são 3, 5, 7, 9 e 100 (`main.py`). A grade continua simétrica, com centro
   exato, mesma fórmula lognormal e sem RNG; o replay de grades legadas foi
   mantido. A matriz de Monte Carlo não é multiplicada pela grade.
2. **Gráfico de trajetórias.** A nota de design escuro descreve 50 trajetórias
   em quatro cores repetidas. Desde `818914c`, o padrão é `n_display=300`,
   índices uniformemente espaçados, cor pelo preço terminal na escala `turbo`,
   mediana ponto a ponto das trajetórias exibidas em branco e preço inicial em
   âmbar (`src/simulation/plotting.py`). O estilo escuro da nota não teve
   commit próprio: entrou na `main` junto com `818914c`, que também alterou
   os gráficos de distribuição e convergência. Na `main`, as barras acima ou
   no limiar `s0 - VaR` usam azul-petróleo (`_TEAL`), não verde.
3. **Contagens de testes.** As contagens registradas (309 e 311 testes) valem
   para as revisões da época. O registro de publicação de `818914c` observou
   332 testes e 51 subtestes; esta atualização não executou a suíte.

## Ajustes de privacidade nas notas

Caminhos absolutos do diretório pessoal foram substituídos: o interpretador da
venv do PyCharm aparece como `<venv-pycharm>/bin/python`, a raiz do projeto como
"diretório do repositório" e os PNGs sintéticos temporários sem caminho. Valores
numéricos, hashes, comandos e resultados registrados não foram alterados.

## Limites

Esta nota é documental: não houve nova simulação, download de dados, execução
de testes ou mudança de lógica financeira. Unidades e convenções seguem as
notas originais (tempo em anos, `mu` em ano^-1, `sigma` em ano^-1/2, preços e
VaR na moeda do ativo). Cenários não são previsões garantidas.
