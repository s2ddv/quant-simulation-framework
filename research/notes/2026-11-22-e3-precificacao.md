---
registro_id: quant-simulation-framework-e3-precificacao-2026-11-22
projeto: quant-simulation-framework
data: 2026-11-22
tipo: nota-de-epico
epico: E3
estado: parcial (E3.0-E3.2 concluídos; E3.3-E3.8 bloqueados)
---

# E3 — Precificação de derivativos (parcial: E3.0, E3.1, E3.2)

Rascunho local. O vault Obsidian `AI Quant` não estava acessível nesta sessão
(`~/.codex/quant-obsidian.json` ausente no contêiner), por isso a nota fica
em `research/notes/` até ser sincronizada.

Especificação: [docs/plano-expansao.md](../../docs/plano-expansao.md), §2.1 e E3.

## Objetivo

Criar `src/pricing/` com precificação sob a medida risco-neutra Q, validada
contra fórmulas fechadas, sem nunca usar o drift histórico `mu`.

## Escopo executado e bloqueio

- **Executado:** E3.0 (parecer), E3.1 (BSM analítico e gregas) e E3.2
  (volatilidade implícita).
- **Bloqueado:** E3.3–E3.8 dependem de E2.1 (antitéticas) e E2.2 (variável
  de controle). Na abertura da tarefa, a §15 do plano marcava as duas como
  "pendente" e `src/simulation/variance_reduction.py` não existia. Por isso
  não foram implementados nem o Monte Carlo europeu, nem a asiática, as
  barreiras, Longstaff–Schwartz, as gregas por MC ou a integração ao CLI.

## Metodologia e fluxo da equipe

O fluxo seguiu `AGENTS.md`, sempre por delegação por instruções a partir de
`.codex/agents/*.toml`, sem papel nativo e com no máximo dois especialistas
ativos:

1. **`pesquisador_quant` (E3.0, somente leitura).** Entregou o parecer:
   - dinâmica `dS = (r−q)S dt + σS dW^Q`;
   - σ igual em P e Q (Girsanov altera só o drift);
   - dividendos discretos fora do escopo;
   - limites da volatilidade histórica como proxy;
   - fórmulas, gregas, limites de não arbitragem e o contrato das
     degenerações;
   - tolerâncias do Brent e passos de diferenças finitas.

   Também apontou três transcrições a corrigir no plano, já aplicadas no
   commit `e8ac17b`:
   - direção da correção BGK em E3.5;
   - T da Tabela 1 de Longstaff–Schwartz em E3.6 (1 e 2 anos, não 1/2);
   - Merton (1973) acrescentado à §16.
2. **`backend_quant`.** Implementou `src/pricing/black_scholes.py` e
   `tests/test_black_scholes.py`.
3. **`testador_quant` e `revisor_quant` em paralelo**, sobre o alvo fixado
   pelos sha256 dos arquivos e depois pelo commit `1223d8a`.
   - O testador escreveu `tests/test_black_scholes_independent.py` com
     referências independentes: integral numérica sob Q, PDE de BSM com q,
     simetria put-call de Bates/Haug, monotonicidade, convexidade e
     continuidade nos limites.
   - O revisor não achou erro de fórmula, sinal ou unidade. Apontou duas
     lacunas de teste (P2), confirmadas por mutação, encaminhadas ao testador.
     Apontou também itens de robustez em extremos (P3), deixados como
     pendência.

### Equações (risco-neutro, dividendo contínuo)

```text
d1 = [ln(S/K) + (r − q + σ²/2)T]/(σ√T),   d2 = d1 − σ√T
C = S e^{−qT}Φ(d1) − K e^{−rT}Φ(d2)
P = K e^{−rT}Φ(−d2) − S e^{−qT}Φ(−d1)
```

Gregas:

- vega e rho por 1,00 de σ e de r;
- theta = −∂V/∂T por ano.

Casos com `T = 0` ou `σ = 0`:

- o preço é o forward descontado `max(S e^{−qT} − K e^{−rT}, 0)`;
- as gregas são degraus;
- `ValueError` exatamente em F = K.

Volatilidade implícita:

- `brentq` com xtol=1e-12, rtol=4·eps e maxiter=200, verificando a
  convergência;
- rejeita `T = 0`, preço fora dos limites estritos e preço inalcançável no
  bracket.

### Fontes

- **Fontes primárias citadas nas docstrings:**
  - Merton (1973), *Bell J. Econ.* 4(1);
  - Hull, 9ª ed., caps. 15, 17 e 19;
  - Haug, 2ª ed., caps. 1–2.
- **Verificação:** nenhuma dessas fontes foi lida no texto original durante
  a tarefa. As docstrings marcam página e número do exemplo como "não
  verificado".
- **Exemplos numéricos:** os valores impressos de Hull usados nos testes
  vieram de fontes secundárias e foram reproduzidos numericamente:
  - 42/40 → 4,76/0,81;
  - 930/900 com q=0,03 → 51,83;
  - 49/50 → Δ 0,522 e Θ −4,31/ano.

## Testes realmente executados e resultados

Todos determinísticos e offline. Não houve simulação Monte Carlo, logo não
há SE nem semente neste escopo.

| Verificação | Resultado medido |
| --- | --- |
| Hull 42/40 | C = 4,759422392871532; P = 0,8085993729000958 (impressos 4,76/0,81, abs 5e-3) |
| Hull índice 930/900, q = 0,03 | C = 51,83295679649086 |
| Hull gregas 49/50, T = 20/52 | Δ 0,52160, Γ 0,065544, vega 12,1055, Θ −4,30533/ano, ρ 8,90696 |
| Paridade put-call | 90 pontos, 1e-10 relativo, com q ≠ 0 e r < 0 |
| Gregas vs. diferenças finitas centrais | 24 casos, com os passos do parecer |
| Ida e volta da volatilidade implícita | 120 casos: 116 bem condicionados com erro máximo 2,7e-13 (< 1e-8); 4 com vega ~1e-10 recuperam o preço a ≤ 1e-12·max(S,K), com erro em σ até ~2e-5 |
| Preço vs. integral sob Q (`quad`) | 48 pontos × call/put; maior erro relativo à tolerância 4,7e-6 |
| PDE de BSM com q | Fecha a 1e-9 do maior termo |
| Mutação (revisor) | 15 de 21 mutantes mortos pelo arquivo original; lacunas P2 em correção pelo `testador_quant` |

Os comandos e contagens da validação final estão no commit que fecha
E3.1/E3.2 e no relatório da sessão.

## Limitações

- **Volatilidade histórica não é implícita.** Ignora o prêmio de risco de
  volatilidade, o smile/skew e a estrutura a termo. Um preço com
  `sigma_source = "historical_proxy"` é um valor de modelo, não um preço
  justo ou de mercado, nem uma previsão.
- **Dividendos:** só rendimento contínuo `q`.
- **Precisão relativa:** perde-se precisão relativa em opções extremamente
  fora do dinheiro e com σ√T minúsculo, onde o erro absoluto é ≤ eps·S.
- **Extremos finitos (P3 do revisor):** alguns parâmetros geram aviso de
  overflow ou `ValueError` mesmo quando o resultado é representável. Nunca
  há nan/inf silencioso.
- **Viés das barreiras contínuas:** a discretização e a correção BGK
  (E3.5) não foram tratadas porque E3.5 está bloqueado.

## Próximos passos

1. Concluir e revisar E2.1/E2.2 e então executar E3.3 (`mc_european` com
   `mu = r − q` como reparametrização, 4·SE e cobertura do IC), seguido de
   E3.4–E3.8.
2. Conferir em exemplares de Hull, Haug, BGK (1997) e Longstaff–Schwartz
   (2001) os números de página e exemplo, além das duas correções feitas no
   plano.
3. Opcional: robustez em log-espaço para `S e^{−qT}` e `K T e^{−rT}` em
   extremos (P3-1 do revisor).
