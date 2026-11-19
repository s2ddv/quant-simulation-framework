---
registro_id: quant-simulation-framework-01a1eb38-e033-7fd2-a4bb-ebc250470db5-execution-scenarios
projeto: quant-simulation-framework
data: 2026-11-18T09:16:47.268389-03:00
tipo: registro-de-entrega
estado: implementado-testado-local-sem-commit
---

# Correção da execução repetida e ampliação dos cenários

[[Codex Quant/Projetos/quant-simulation-framework/Projeto]]

## Objetivo e hipótese

Corrigir o FileExistsError fornecido pelo usuário ao executar main.py no PyCharm,
causado pelo destino padrão reports/simulation já ocupado. Escolher um destino
livre implicitamente deve permitir execução repetida sem perder resultados.
Ampliar a análise de 9 para 25 cenários de parâmetros usando o GBM normal existente,
com opções de 49/81. Protocolo prévio: research/notes/2026-11-17-execution-scenarios.md.
Grade mais densa não implica maior precisão nem extremos mais amplos.

## Estado e implementação

Branch codex/obsidian-quant-integration; HEAD f5d0b9d8fab9b45fad06e3730c111ca1e2d59314.
Alterações locais sem commit/push. Arquivo .idea/monte-carlo-gbm.iml preexistente
preservado. SHA-256 do diff revisado (sete arquivos): 70646c8892609c428043b611e12d11d3e0590571a892cc017aba8884c0573d25.

- main.py: destino implícito livre, nome com timestamp UTC e sufixo em colisões;
  destino explícito ocupado continua recusado. Fronteira CLI explica erros operacionais.
  Sensibilidade ativa por padrão para novas execuções, 5 níveis/25 cenários;
  --scenario-levels 3/5/7/9 e --no-calibration-sensitivity. Tabela com média,
  P5/P50/P95 e VaR. Replay preserva opções gravadas salvo pedido explícito.
- src/validation/calibration_sensitivity.py: grade uniforme simétrica até 81 linhas,
  centro exato, mesma fórmula lognormal e sem RNG. API com padrão legado 3 níveis.
- src/simulation/report.py: restaura scenario_levels opcional no replay.
- tests/test_main.py, tests/test_calibration_sensitivity.py,
  tests/test_research_integration.py: regressões, referência SciPy e adulterações.
- README.md e research/notes/: instruções, protocolo e resultados.

Ambiente de teste criado em .venv com --system-site-packages; pytest 9.2.0 instalado
somente nessa venv após tentativa sem rede falhar. Dependências numéricas do sistema
reutilizadas. PyCharm e sua venv não foram modificados.

## Validação executada

1. Reprodução original: python3 main.py --ticker PETR4.SA --no-plot levantava
   FileExistsError antes do download. Traceback do usuário confirmou a mesma causa.
2. Especialista backend: 67 testes focados passaram e quatro comparações exatas
   com JSON da implementação anterior passaram.
3. `MPLBACKEND=Agg MPLCONFIGDIR=/tmp/qsf-mpl .venv/bin/python -m pytest -q`:
   **309 passed, 51 subtests passed**, 25.49 s. Um aviso de depreciação do plugin
   anyio (CallSpec2); nenhuma falha. Python 3.14.7, NumPy 2.4.6, pandas 3.0.6,
   SciPy 1.18.1, Matplotlib 3.11.2, yfinance 1.7.0.
4. `PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/qsf-mpl .venv/bin/python -m unittest discover -s tests -p 'test_statistics.py' -v`:
   **11 testes OK**, 0.047 s. `git diff --check`: sem erros.
5. Execução real com o Python indicado pelo usuário:
   `MPLBACKEND=Agg MPLCONFIGDIR=/tmp/qsf-mpl <venv-pycharm>/bin/python main.py --ticker PETR4.SA --no-plot`:
   exit 0; consulta pública yfinance, 749 fechamentos ajustados, 2023-11-17 a
   2026-11-17. Configuração: 1 ano, 252 passos, 1000 trajetórias, seed 42,
   alpha 0.95, mu_delta 0.02, sigma_relative_delta 0.2, 25 cenários.
   NumPy desse ambiente: 2.5.3. Report e três PNGs gerados.
6. Mesmo Python, `main.py --replay reports/simulation-20261117T204141372801Z/report.json --no-plot`:
   exit 0. JSON do replay **idêntico byte a byte**. Assinaturas dos seis PNGs
   verificadas; paths.png da execução real inspecionado visualmente.
7. Revisão independente somente leitura: nenhum achado confirmado no diff.
   Conferiu igualdade legado em quatro casos, fórmulas/centros em 16 grades,
   nove combinações de replay em memória e colisões de nomes. Não repetiu suíte.

## Artefatos

Raiz: diretório do repositório (caminhos abaixo são relativos a ele).

- reports/simulation-20261117T204141372801Z/report.json e convergence.png,
  paths.png, final-distribution.png.
- reports/simulation-20261118T121522846270Z/: replay com os mesmos quatro arquivos.
- SHA-256 do JSON novo/replay: d385ac59be359ff777a40de568fd33ce093af28035a7eef11c83275d856e8f42.
- SHA-256 do snapshot histórico novo: db4a4e32149d5a71e62f058e27baaecf2e4cde9e1b9506ec3a46a7a6593a018e.
- reports/simulation/report.json preservado:
  925389bcf8edce84b51e4a533958befad5742b4d4c038e7805c75f45776d89ac.

## Fontes, interpretação e limites

Fonte técnica consultada: NIST/SEMATECH e-Handbook of Statistical Methods,
seção 1.3.6.6.9, Lognormal Distribution, autoria institucional NIST/SEMATECH,
https://www.itl.nist.gov/div898/handbook/eda/section3/eda3669.htm.
Consulta HTML 2026-11-17: definição, percent point function e common statistics;
edição não indicada; páginas físicas/impressas não se aplicam; hash do original
não obtido. Essa fonte dá suporte à lei lognormal existente. Densidade da grade
é escolha local; nenhuma estratégia ou modelo financeiro novo foi implementado.

Resultados sintéticos/testes e download real não comprovam capacidade preditiva
ou desempenho B3. Não houve backtest, treino, seleção de estratégia ou holdout;
separação treino/teste, custos e universo com deslistados não se aplicam a esta
correção operacional/grade condicional e não foram certificados. Tempo em anos,
mu em ano^-1, sigma em ano^-1/2, preços/VaR na moeda do ativo. Sem mudança do RNG
ou do contrato de perdas. Janela gráfica interativa não testada (--no-plot).

O relatório anterior reports/simulation/report.json continua incompatível com
read_report: falta terminal_mean_confidence_interval e referências analíticas
diferem aproximadamente 1e-14. A reprodução foi rejeitada com Replay result
mismatch in results. Isso antecede o diff; não relaxamos a integridade nem
sobrescrevemos esse relatório. Compatibilidade de grades legadas foi testada,
mas não equivale a compatibilidade com todas as versões antigas de relatórios.

## Próximos passos

Executar main.py normalmente no PyCharm; escolher --scenario-levels 9 para 81
cenários. Se necessário posteriormente, planejar migração explícita e auditável
dos relatórios históricos incompatíveis, preservando originais e proveniência.
