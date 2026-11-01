# Interpretação dos resultados reais — protocolo v1

O experimento empírico foi concluído para os três ativos. Este texto interpreta
os resultados salvos; não altera o protocolo, parâmetros ou observações.
Dados: `frozen-network-v1/manifest.json`. Métricas e previsões: `frozen-network-v1/results/results.json`.
Relatório de tabelas e nove figuras: `frozen-network-v1/results/report.md`. Figuras inspecionadas visualmente.

## Resultado principal: teste sequencial 2019–2025

|Ativo|Previsões|MSE histórico|MSE EWMA|MSE ML|ML vs EWMA|IC 95% da diferença ML–EWMA, L=10|
|---|---:|---:|---:|---:|---:|---|
|SPY|1760|3.51208e-7|3.07040e-7|3.03205e-7|-1.25%|[-2.43506e-8, 1.76541e-8]|
|AAPL|1760|1.26502e-6|1.19034e-6|1.15252e-6|-3.18%|[-9.45070e-8, 5.10321e-9]|
|PETR4.SA|1742|1.47196e-5|1.34744e-5|1.28459e-5|-4.66%|[-1.59227e-6, 5.89615e-8]|

MSE em log-retorno^4. Percentuais = 100*(MSE_ML/MSE_EWMA - 1), calculados por ativo;
não são rentabilidade nem ponderação entre ativos. O ML teve menor MSE agregado
nos três ativos, mas TODOS os intervalos ML–EWMA do teste completo cruzaram zero
com L=10. Não há evidência robusta de superioridade geral frente ao EWMA.
A magnitude relativa foi modesta; relevância econômica não foi avaliada.

## Consistência por calendário

|Ativo|2019|2020–2021|2022–2023|2024–2025|
|---|---|---|---|---|
|SPY — menor MSE|ML|ML|EWMA|ML|
|AAPL — menor MSE|ML|ML|EWMA|ML|
|PETR4.SA — menor MSE|EWMA|ML|histórico|EWMA|

O ganho não se repete em todos os períodos. As curvas acumuladas mostram grande
concentração em 2020 e 2025 nos ativos americanos. Frente ao EWMA, 2025 contribui mais que 2020 tanto em SPY quanto em AAPL; em PETR4.SA somente 2020 favorece ML nessa soma anual.
PETR4.SA é o exemplo mais forte de reversão: ML vence no MSE de 2020–2021,
mas perde para EWMA em 2019 e 2024–2025, e para histórico em 2022–2023.
Em 2022–2023 e 2024–2025, seus ICs ML–EWMA ficam acima de zero nos três blocos,
favorecendo EWMA. Não se trata de um ganho distribuído uniformemente no tempo.

## Sensibilidade e comparações inconclusivas

SPY: ML–EWMA cruza zero no teste completo com L=5,10,20; ML–histórico cruza
com L=5,10, mas exclui zero com L=20. AAPL: ML–EWMA cruza zero no teste completo
com os três blocos; ML–histórico fica negativo com os três. PETR4.SA:
ML–EWMA exclui zero com L=5, mas cruza com L=10,20. ML–histórico exclui zero
com L=5,10, mas cruza com L=20. Portanto decisões binárias dependem do bloco
em alguns contrastes; não escolher o comprimento que torna o achado favorável.
AAPL 2019 e 2024–2025 também apresentam mudanças de inclusão de zero em alguns
contrastes; os detalhes de todas as sensibilidades constam das tabelas/JSON.

## Métricas secundárias e risco

No teste completo, ML tem menor MAE e pinball em SPY e AAPL. Em PETR4.SA,
EWMA vence MAE e pinball apesar do menor MSE do ML: há conflito de objetivo.
SPY 2022–2023 também diverge: EWMA vence MSE/pinball, histórico vence MAE.
AAPL 2022–2023: EWMA vence MSE, ML vence MAE/pinball.
Taxas de violação do VaR do ML: SPY 6.08%, AAPL 4.83%, PETR4.SA 3.44%,
frente à referência nominal de 5%. PETR4.SA EWMA: 4.82%; a menor taxa do ML
pode indicar conservadorismo e não prova melhor calibração. Wilson e bootstrap
das taxas estão no JSON, junto às diferenças pareadas secundárias.
Não comparar pinball em dólares com pinball em reais sem normalização explícita.

## Limitações e registro da execução

Os preços ajustados foram obtidos em 2026 e podem incorporar revisões posteriores:
não são dados autenticados disponíveis em cada origem histórica. O alvo r² é
ruidoso, a média condicional zero e a distribuição normal para VaR são hipóteses.
Bootstrap de perdas condiciona nos ajustes realizados, sem reestimar seleção ou
modelos, e supõe dependência local compatível com blocos; choques e mudanças
persistentes limitam essa interpretação. ICs são pontuais, não corrigidos por
múltiplas comparações. Cruzar zero não demonstra equivalência. Os três ativos
não representam todo o mercado; nenhuma estratégia ou rentabilidade foi testada.
Método de blocos: Künsch (1989), https://doi.org/10.1214/aos/1176347265;
as escolhas finitas desta implementação estão declaradas no protocolo.

A primeira tentativa `frozen-v1` ficou bloqueada por DNS no sandbox, registrada
em seu manifesto (não houve avaliação). Após autorização de rede, o download
teve sucesso em novo destino `frozen-network-v1`; essa é a única mudança operacional
de destino, sem alteração do desenho nem uso de resultados para reajustar parâmetros.
O fornecedor emitiu avisos de depreciação de pandas, sem invalidar os preços.
Base Git e estado modificado constam dos manifestos/execution.json; o commit que
contém este relatório identifica a versão final entregue. Os hashes do diff Git
não abrangem arquivos então não rastreados; o commit final contém esses arquivos.
