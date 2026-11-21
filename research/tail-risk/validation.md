# Fase 2 — Relatório de validação estatística

Evento: retorno simples de 5 pregões ≤ −8,6%; mu = 5% a.a.; T = 5/252; 5 passos diários; choques t padronizados (variância unitária). Parâmetros de cada linha estão em `validation_results.json`; reprodução: `python research/tail-risk/validation_experiments.py` (≈7 s). IC de 95% de Wilson (proporções); nenhum erro-padrão baseado em momentos dos preços foi usado.

## Implementação
- `src/validation/tail_probability.py`: IC de Wilson e Clopper–Pearson, probabilidade normal exata, estimador Monte Carlo em lotes (sementes filhas de `SeedSequence`).
- `tests/test_tail_probability.py` (11 testes): KS do log-retorno contra a normal teórica, quantis, cobertura do IC em 60 sementes, escala 1/√n da largura, determinismo, varredura de σ, ordem das caudas por df, entradas inválidas. Testes com semente fixa usam IC de 99,9% para não falhar por evento de 5% de probabilidade.
- Reaproveitados os testes existentes de momentos/quantis lognormais de `tests/test_statistics.py` e `src/validation/statistics.py` (média e variância do preço só valem para GBM normal).

## Resultados (1 milhão de trajetórias por célula, semente 20260101)
P(queda ≥ 8,6%) em %, com IC de Wilson 95%:

| σ a.a. | Normal | t df=8 | t df=5 | t df=4 | t df=3 | Normal analítico |
|---|---|---|---|---|---|---|
| 10% | 0,0000 | 0,0001 | 0,0033 | 0,0113 | 0,0349 | ≈0 |
| 15% | 0,0009 | 0,0076 | 0,0376 | 0,0703 | 0,1501 | 0,0009 |
| 20% | 0,0704 [0,0654; 0,0758] | 0,1344 | 0,2191 | 0,2981 [0,2876; 0,3090] | 0,4069 | 0,0656 |
| 30% | 1,6494 | 1,7648 | 1,8243 | 1,8502 | 1,7131 | 1,6571 |

- **Normal vs. analítico:** os valores simulados de σ = 15%, 20% e 30% ficam dentro do IC e compatíveis com a fórmula fechada.
- **Convergência (perda ≥ 5%, σ = 20%, analítico 3,2748%):** n = 10⁴: 3,16% [2,83; 3,52]; 10⁵: 3,24% [3,13; 3,35]; 10⁶: 3,28% [3,24; 3,31]; 4·10⁶: 3,28% [3,26; 3,30]. A largura cai com ≈1/√n. É uma única realização; os prefixos não são amostras independentes.
- **Sementes (t df=4, σ=20%, 10 sementes):** 0,2856%–0,3095%, compatível com o erro binomial (≈ ±0,011 p.p. de desvio de 95% por semente).
- **Heterogeneidade:** com variância fixa, a cauda t **não é uniformemente mais pesada**: a σ=30% o evento (≈2,2 desvios) é só ~10% mais provável com t4 e **menos** provável com t3 que o normal. O excesso de probabilidade cresce com a distância em desvios-padrão, a partir de ≈3σ. Para σ=10% (≈ 6σ) a probabilidade normal é da ordem de 10⁻¹⁰ (≈6σ) enquanto a t4 dá ≈10⁻⁴: diferença de várias ordens de magnitude.

## Limitações
- O sinal é do modelo, não do mercado: aqui as caudas são impostas por construção.
- As t(df≤2) estão fora do escopo (variância infinita). Com df finito E[S_T] é infinito.
- Os IC de Wilson/Clopper–Pearson cobrem só o erro de Monte Carlo; não cobrem erro do modelo nem incerteza de parâmetros.
- O simulador mantém trajetórias completas por lote; a memória por lote é ≈ 8·6·10⁵ bytes.

## Testes executados
`pytest tests/test_tail_probability.py`: 11 passed. A suíte completa é executada antes do commit.
