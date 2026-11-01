# Protocolo v1 — comparação sequencial de volatilidade

Registrado antes de consultar qualquer avaliação final. Base: 43b8238 (main);
árvore inicialmente limpa. Implementação estará modificada na execução; o commit
final identifica o código entregue. Nenhum ajuste de protocolo após os resultados.

## Dados e desenho
SPY (ETF amplo dos EUA), AAPL (ação americana) e PETR4.SA (ação brasileira):
contrastes de instrumento e mercado, sem representatividade estatística do mercado.
Yahoo Finance via yfinance, Close com auto_adjust=True, diário, start=2010-01-01
inclusivo, end=2026-01-01 exclusivo: sessões até 2025-12-31.
Uma chamada por ativo, sem substituição nem repetição automática. Rejeitar duplicatas,
preços não positivos/não finitos e datas fora da faixa; nenhuma interpolação, exclusão
ou preenchimento. Registrar lacunas de calendário >7 dias para inspeção, sem presumir
que sejam sessões faltantes. Snapshot JSON canônico com SHA-256, opções, versões,
instante de obtenção e datas efetivas. Avaliação e replay posteriores offline.
Preços ajustados atuais podem incorporar revisões: esta não é base point-in-time autenticada.

Treino: alvos disponíveis em 2010–2015 após lookback comum de 63 sessões.
Validação: alvos em [2016-01-01,2019-01-01). Teste: [2019-01-01,2026-01-01).
Segmentos fixos: 2019, 2020–2021, 2022–2023, 2024–2025. São calendários,
não regimes estimados. Não reiniciar avaliação nas fronteiras dos segmentos.

## Modelos e métricas
Reutilizar ForecastConfig: janelas históricas [5,22,63], EWMA [0.90,0.94,0.97],
features NNLS médias de r² de 1,5,22 sessões; intercepto e slopes não negativos.
Selecionar referências pelo MSE da validação, empate por ordem da grade; congelar no teste.
Refit NNLS expansivo a cada 22 origens e no início de validação/teste, somente com
alvos realizados até a origem. Teste sequencial admite alvos passados do próprio teste.
Horizonte próximo pregão observado; E[r|informação]=0; alvo r² é proxy ruidosa da variância.
Primária: média das diferenças pareadas de erros quadráticos, ML-histórico,
ML-EWMA e histórico-EWMA. Negativo favorece primeiro modelo. MSE em log-retorno^4;
MAE em log-retorno². Secundárias: diferença de erros absolutos e pinball de preço,
contagem/taxa de violações estritas, referência nominal 5%, Wilson 95% aproximado sob
independência e bootstrap da taxa. Pinball em moeda do ativo: não agregar entre ativos.
Análise por ativo; nenhum resumo quantitativo entre ativos. Relevância material será
descrita pela diferença MSE relativa ao MSE da segunda referência, sem limiar decisório.

## Incerteza
Moving-block bootstrap não circular de blocos sobrepostos: sortear uniformemente
inícios 0..n-L, concatenar ceil(n/L) blocos, truncar o último para n observações.
Mesmos índices para todas as colunas/modelos; diferenças pareadas antes de reamostrar.
2.000 réplicas, lotes de 64, NumPy default_rng/PCG64, seed=42 para cada série/segmento.
L principal=10; sensibilidades=5,20. IC percentil 95% (quantis lineares).
Menos de 10 blocos completos: alerta, sem conclusão forte; n<L: sem IC.
Memória estimada: para n<=1800 e 12 colunas, lote indexado <12 MiB, mais índices,
replicações e previsões; custo aproximado <100 milhões de valores por ativo.
Duração esperada de minutos; executar um experimento pesado por vez.
Referência primária: Künsch (1989), The Jackknife and the Bootstrap for General
Stationary Observations, Annals of Statistics 17, 1217–1241,
https://doi.org/10.1214/aos/1176347265 . Usamos sua família de blocos sobrepostos;
comprimentos fixos, truncamento e IC percentis são decisões deste protocolo, não
seleção ótima de bloco nem reprodução integral de seu procedimento assintótico.

ICs condicionais aos dados congelados, modelos ajustados e protocolo; quantificam
variabilidade temporal observada sob dependência fraca/estacionariedade aproximada.
Não refazer ajustes nas réplicas. Não cobrem toda incerteza de seleção/estimação,
revisões ou mudanças futuras. ICs pontuais sem correção de multiplicidade.
Cruzar zero é inconclusivo, não prova equivalência. Discutir sensibilidade ao bloco,
concentração temporal e conflitos entre métricas. Menos violações pode ser excesso
de conservadorismo; não implica melhor VaR, rentabilidade ou superioridade geral.

## Artefatos e falha de obtenção
Destino novo research/experiments/frozen-v1; nunca sobrescrever. Separar snapshots,
manifesto e resultados, salvando previsões individuais e fits, métricas, gráficos e
identidade Git. Replay deve verificar hashes e igualdade dos resultados offline.
Se um download falhar, registrar bloqueio; não usar sintéticos como dados empíricos.
