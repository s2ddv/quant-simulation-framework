# Execução repetida e ampliação de cenários — protocolo

Data: 2026-11-17, America/Sao_Paulo. Projeto: quant-simulation-framework.
Base: branch codex/obsidian-quant-integration, commit f5d0b9d.

## Problema e hipótese local (antes da avaliação)

O traceback fornecido pelo usuário e a reprodução local mostram que o destino
padrão reports/simulation ocupado bloqueia main.py antes de consultar o ticker.
Hipótese: escolher automaticamente um destino livre quando --report é omitido
permite repetir a execução preservando todos os relatórios existentes. Nomes
explícitos continuam protegidos contra sobrescrita.

A análise atual tem 9 combinações (3 níveis de drift por 3 de volatilidade).
Ampliar a grade para 5 níveis por parâmetro fornece 25 cenários, incluindo os
9 anteriores e combinações intermediárias. Opções limitadas a 3, 5, 7 e 9 níveis
permitem até 81 cenários pequenos, sem multiplicar matrizes de Monte Carlo.
Não se espera que maior densidade da grade necessariamente amplie seus extremos.

## Método e baseline

Preservar o GBM normal e suas fórmulas analíticas existentes. Parâmetros: preço
na moeda do ativo; horizonte em anos; mu em ano^-1; sigma em ano^-1/2.
Grade simétrica entre mu +/- mu_delta e sigma*(1 +/- sigma_relative_delta),
incluindo exatamente o centro. Padrão da API/replay legado permanece 3 níveis;
CLI de sensibilidade passa a 5. Sensibilidade é determinística e não usa RNG.
O modelo Student-t da API permanece fora desta ampliação.

Fonte técnica consultada: NIST/SEMATECH e-Handbook of Statistical Methods,
seção 1.3.6.6.9, Lognormal Distribution, autoria institucional NIST/SEMATECH,
https://www.itl.nist.gov/div898/handbook/eda/section3/eda3669.htm.
Consulta HTML em 2026-11-17: definição, percent point function e common statistics.
Versão/edição não indicada na página; páginas físicas/impressas não se aplicam;
hash do documento não obtido (consulta web, sem download do original).
A fonte sustenta a lei lognormal; a escolha da grade é hipótese de sensibilidade
local, não um resultado acadêmico dos autores nem evidência de previsão.

## Critérios e validação planejada

- Reproduzir a falha; executar duas vezes sem --report com destino inicial ocupado;
  verificar novos bundles e bytes anteriores preservados.
- Manter rejeição de nomes explícitos ocupados antes de rede ou simulação.
- Comparar 25/49/81 cenários com referência SciPy independente; centro, limites,
  parâmetros inválidos e degenerações. Os 9 cenários antigos devem permanecer
  idênticos no modo legado, inclusive no replay.
- Validar JSON/replay e detectar adulteração da grade/configuração.
- Executar suíte offline uma vez após consolidação e revisão independente.
- Executar CLI/replay com saída nova e verificar JSON e PNGs.

Dados sintéticos nos testes, seeds explícitas nas simulações (42 por padrão).
Essas verificações não demonstram desempenho B3 nem disponibilidade do provedor.
Resultados e comandos observados serão registrados após a execução.
