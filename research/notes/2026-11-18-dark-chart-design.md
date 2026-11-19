---
registro_id: quant-simulation-framework-chart-design-bloomberg-style-2026-11-18
projeto: quant-simulation-framework
data: 2026-11-18T17:24:08.474511-03:00
tipo: registro-de-entrega
estado: implementado-testado-local-sem-commit
---

# Gráficos em estilo terminal financeiro escuro

[[Codex Quant/Projetos/quant-simulation-framework/Projeto]]
[[Codex Quant/Projetos/quant-simulation-framework/Registros/Execucao-repetida-e-25-cenarios-2026-11-18]]

## Objetivo e hipótese

Usuário pediu gráficos mais coloridos e em estilo Bloomberg. Hipótese de design:
uma paleta escura com acentos âmbar, verde, vermelho, azul e roxo pode melhorar
a distinção visual dos resultados sem alterar números nem contratos financeiros.
O objetivo é inspiração visual em terminal financeiro, sem uso de logotipo.

## Implementação e estado

Branch codex/obsidian-quant-integration; HEAD f5d0b9d8fab9b45fad06e3730c111ca1e2d59314;
trabalho local sem commit/push. Alteração .idea/monte-carlo-gbm.iml preexistente
foi preservada. Arquivos desta entrega: src/simulation/plotting.py,
tests/test_plotting_design.py e trecho Price charts de README.md.
SHA-256 do plotting.py final: 8408597c0e97954cb25a4f37c664a275231826a51180ee48f014a7fb7aff74b3.

- Trajetórias: painel grafite, quatro cores repetidas para as 50 trajetórias
  exibidas, preço inicial em âmbar, legenda informando 50/1000 no caso padrão.
- Distribuição: barras vermelhas para preços estritamente abaixo de s0 - VaR,
  verdes para preços iguais ou acima do limiar, com bins compartilhados; linhas
  do preço no VaR, preço inicial e média final.
- Convergência: cores por métrica, referência teórica da média tracejada e
  três painéis com unidades corretas. Eixo N logarítmico existente mantido.
- Estilo é local às figuras Matplotlib; sem alterar rcParams global, RNG,
  estimação, simulação, risco, JSON ou CLI. Gráficos continuam salvos em PNG.

## Validação observada

- Especialista frontend: 3 testes focados passaram e três PNG sintéticos
  (paths, distribution e convergence, gravados em diretório temporário
  fora do repositório) inspecionados visualmente.
- `MPLBACKEND=Agg MPLCONFIGDIR=/tmp/qsf-mpl .venv/bin/python -m pytest -q`:
  **311 passed, 51 subtests passed**, 22.70 s; um aviso externo de
  depreciação do plugin anyio CallSpec2. Nenhuma falha.
- `MPLBACKEND=Agg MPLCONFIGDIR=/tmp/qsf-mpl <venv-pycharm>/bin/python main.py --replay reports/simulation-20261117T204141372801Z/report.json --no-plot`:
  exit 0; criou reports/simulation-20261118T202227541711Z/ com 3 PNGs e report.json.
  JSON é idêntico byte a byte ao original, SHA-256
  d385ac59be359ff777a40de568fd33ce093af28035a7eef11c83275d856e8f42.
  Os três PNGs PETR4.SA foram inspecionados visualmente; sem corte de elementos
  observado no preview. `git diff --check`: passou.

## Dados, limites e próximos passos

Dados: replay offline de snapshot de 749 preços ajustados PETR4.SA, período
2023-11-17 a 2026-11-17. Simulação salva: horizonte 1 ano, 252 passos,
1000 trajetórias, seed 42, confiança VaR 0.95. Valores de cenário e dados não
foram recalculados para o design; matemática e contratos ficam na implementação
existente. Marcador vermelho é preço s0 - VaR, não a perda. Unidades de variância
são moeda do ativo ao quadrado. Gráficos de preços usam moeda do ativo.

Janela gráfica interativa não executada: validação cobriu saída PNG com Agg.
Visual inspirado em terminal financeiro não é interface Bloomberg real. Os
cenários e preços simulados não são garantia de previsão ou desempenho B3.
Próximo passo: abrir os PNGs da nova pasta para inspeção pelo usuário e
executar normalmente para gerar novos gráficos no mesmo estilo.
