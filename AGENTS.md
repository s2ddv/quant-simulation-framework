# Equipe quantitativa do Codex

## Projeto e comandos

Este repositório é um simulador de GBM para um ativo em Python >=3.10.
Leia `README.md` e `pyproject.toml`; confirme branch, estado do Git e arquivos
envolvidos antes de trabalhar. Preserve alterações existentes e outros worktrees.
Não assuma que funcionalidades de outra branch já foram integradas.

- `main.py`: interface de terminal, execução e replay.
- `src/calibration/`: dados históricos, estimação e fundamentos.
- `src/simulation/`: GBM, gráficos Matplotlib e relatórios JSON.
- `src/risk/`: estatísticas de perdas, VaR e CVaR.
- `src/validation/`: referências analíticas e convergência.
- `tests/`: testes offline com pytest e unittest.
- `research/report.md`: roteiro de pesquisa, não resultados experimentais.
- `notebooks/` e `figures/`: pesquisa e exemplos visuais.

Use a `.venv` existente quando disponível. Não instale dependências sem necessidade.
Comandos a partir da raiz:

```sh
.venv/bin/python main.py --help
MPLBACKEND=Agg .venv/bin/python -m pytest -q
PYTHONPATH=src MPLBACKEND=Agg .venv/bin/python -m unittest discover -s tests -p 'test_statistics.py' -v
```

Se necessário, configure `MPLCONFIGDIR` para um diretório temporário gravável.
Execuções novas usam yfinance e rede; replay de um relatório existente é offline.
`--no-plot` ainda grava JSON e PNG. Escolha um destino novo; não substitua artefatos.
Testes offline não comprovam disponibilidade do provedor nem interfaces gráficas.

## Orquestração

O agente principal coordena a equipe. Delegue quando uma subtarefa concreta puder
avançar de forma independente ou exigir uma revisão especializada. Não é necessário
acionar todos os especialistas; resolva tarefas triviais diretamente.

1. Interprete a solicitação, seus limites e critérios de aceitação. Planejamento
   pedido pelo usuário permanece somente leitura até autorização para implementar.
2. Inspecione o estado atual e identifique os especialistas necessários na tabela.
3. Distribua tarefas com objetivo, evidência de entrada, arquivos permitidos,
   arquivos proibidos, dependências e critérios de conclusão. Indique expressamente
   se a tarefa é somente leitura. O escopo da delegação nunca amplia a autorização.
4. Mantenha no máximo **dois especialistas ativos ao mesmo tempo**, inclusive em
   trabalhos leves. Não faça trabalho pesado no principal enquanto dois filhos
   estiverem executando trabalho pesado. Especialistas não criam outros agentes.
5. Atribua cada arquivo a um único escritor. Se houver sobreposição, serialize as
   tarefas. Arquivos compartilhados ficam com o principal até transferência explícita.
   Contextos separados de agentes não significam diretórios ou branches separados.
6. Consolide as mudanças antes da validação. Execute testes e revisão sobre uma
   versão estável identificada por commit ou diff. Não edite o alvo enquanto ele
   estiver sendo revisado. Testador e revisor podem atuar em paralelo se independentes.
7. Encaminhe ao corretor somente falhas confirmadas, com reprodução ou evidência
   concreta. Hipóteses vão para investigação; não devem virar correções especulativas.
8. Revalide o comportamento corrigido, a regressão correspondente e os testes
   afetados. Peça nova revisão das correções relevantes. Não repita tentativas
   idênticas indefinidamente: relate a causa e qualquer decisão necessária.
9. Entregue resultado, arquivos envolvidos, verificações realmente executadas,
   limitações e pendências. Não declare sucesso se faltam critérios de aceitação.

| Especialista | Quando delegar | Fronteira padrão |
| --- | --- | --- |
| `pesquisador_quant` | Modelos, SDEs, métodos numéricos e literatura | Pesquisa e parecer; não edita produção |
| `backend_quant` | Algoritmos, simulação, calibração, risco e infraestrutura | Implementação nos arquivos atribuídos |
| `frontend` | Gráficos, apresentação e interfaces quantitativas | Matplotlib/CLI atuais; dashboard apenas se solicitado |
| `testador_quant` | Testes unitários, estatísticos, integração e regressão | Testes atribuídos; não corrige produção |
| `revisor_quant` | Auditoria matemática, arquitetura, desempenho e segurança | Relatório independente; nenhuma edição durante revisão |
| `corretor` | Bugs confirmados e reproduzíveis | Correção mínima e regressão nos arquivos atribuídos |

As definições estão em `.codex/agents/*.toml`. Use o papel nativo quando a ferramenta
de delegação permitir selecioná-lo. Se a sessão expuser apenas delegação genérica,
leia o arquivo correspondente e inclua suas `developer_instructions` na tarefa
delegada, junto ao escopo. Identifique esse caminho como **delegação por instruções**;
não alegue que um perfil nativo foi carregado. Se não houver ferramenta multiagente,
informe a limitação e execute sequencialmente, sem inventar agentes independentes.

## Comunicação e conclusão

Toda resposta de especialista deve informar:

- **Papel e objetivo:** tarefa recebida e escopo atendido.
- **Resultado e evidência:** arquivos/linhas, fórmulas, fontes ou reprodução.
- **Alterações:** arquivos modificados ou declaração de que não houve edição.
- **Validação:** comandos, resultados e o que não foi verificado.
- **Pendências:** hipóteses, riscos concretos, bloqueios e próximo responsável.

Sinalize cedo conflitos de arquivos, orçamento de memória insuficiente, evidência
contraditória ou falta de informação essencial. Não mude o contrato silenciosamente.
O principal decide integração e encerramento; um especialista não aprova o próprio
trabalho como revisão independente.

## Padrões quantitativos

Use como referência, para parâmetros constantes:

```text
dS = mu*S*dt + sigma*S*dW
S(t) = S(0)*exp((mu - sigma^2/2)*t + sigma*W(t))
E[S(T)] = S(0)*exp(mu*T)
Var[S(T)] = S(0)^2*exp(2*mu*T)*(exp(sigma^2*T) - 1)
```

- O motor atual usa incrementos exatos lognormais, não Euler–Maruyama. A grade
  temporal não cria erro de discretização na lei do GBM nesses pontos; medições
  dependentes da trajetória entre pontos podem exigir outra análise.
- Declare equações, hipóteses, parâmetros e unidades: preço na moeda do ativo,
  tempo em anos, `mu` em ano^-1 e `sigma` em ano^-1/2. A calibração usa 252 pregões.
- `mu` é drift do preço; não confunda com a média dos log-retornos. Na calibração
  atual, `mu = 252*mean(log_returns) + sigma^2/2` e `sigma = sqrt(252)*std(log_returns)`.
- Drift histórico não é drift risco-neutro; volatilidade histórica não é implícita.
  Uma implementação futura de precificação deve declarar a medida e dividendos.
- Compare resultados com soluções analíticas, inclusive média, variância e quantis.
  Diferencie percentis da distribuição, intervalos de confiança de estimadores,
  erro Monte Carlo, erro numérico, incerteza de calibração e erro do modelo.
- Controle sementes e registre configuração, dados, versões e convenção de RNG.
  Mesma semente não promete igualdade entre versões ou formatos diferentes da amostra.
- Use tolerâncias justificadas e sementes fixas. Não exija queda monotônica do erro
  de uma realização Monte Carlo. Prefixos aninhados são amostras correlacionadas.
- Preserve contratos atuais: matriz `(steps + 1, n_sims)`, linha zero com preço
  inicial; perdas `s0 - terminal`; VaR/CVaR em unidades de preço, podendo ser negativos;
  CVaR inclui empates no limiar. Mudanças nesses contratos exigem escopo explícito.
- Considere finitude, overflow, underflow, degenerações e memória antes de otimizar.
  Não invente resultados de experimentos ou apresente cenários como previsões garantidas.

## Recursos, ferramentas e permissões

- Ferramentas, rede, sandbox e aprovações são herdados da sessão e continuam sujeitos
  às políticas do ambiente. A instrução de somente leitura não é um sandbox separado.
- Use busca e leitura local para inspeção, Python/NumPy para verificações autorizadas
  e fontes primárias para pesquisa. Não instale conectores nem exponha credenciais.
- Execute no máximo uma suíte completa, compilação pesada ou benchmark por vez.
  Estime as matrizes e temporários antes de simular: o motor atual mantém trajetórias
  completas. Reduza a amostra ou serialize quando a memória estiver limitada.
- Reutilize evidências válidas; não duplique downloads, pesquisas ou testes sem motivo.
- Não execute comandos destrutivos, publique, envie mensagens externas ou altere
  configurações globais sem autorização aplicável. Não faça commit/push por iniciativa
  de um especialista. Não enfraqueça sandbox ou aprovações para fazer um teste passar.

Consulte `docs/equipe-multiagente.md` para compatibilidade, ativação e testes da equipe.


<!-- BEGIN CODEX QUANT OBSIDIAN v1 -->
## Contexto e registro no Obsidian

Este projeto participa da integração persistente com o vault `AI Quant`.
Leia `~/.codex/quant-obsidian.json` e a seção "Base de conhecimento Quant no Obsidian"
de `~/.codex/AGENTS.md`. Antes de implementar modelos, consulte a página deste projeto,
as pesquisas, decisões e experimentos pertinentes, além dos contratos do repositório.
Confirme tudo contra o estado atual do Git e do código.

Após tarefas relevantes, documente automaticamente objetivo/hipótese, implementação,
arquivos, metodologia, testes realmente executados/resultados, limitações e próximos
passos. Organize por projeto/modelo/experimento/decisão e use links internos do Obsidian.
Não invente evidências, duplique registros nem documente alterações triviais. Crie
notas novas sem sobrescrever notas pessoais ou excluir arquivos; releia para verificar.
Se o vault estiver pendente ou inacessível, informe a limitação e guarde um rascunho.
A integração não concede permissão para alterar lógica financeira, fazer commit/push
ou publicar dados. As demais instruções deste arquivo continuam em vigor.
<!-- END CODEX QUANT OBSIDIAN v1 -->
