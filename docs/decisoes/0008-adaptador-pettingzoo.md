# 0008. Adaptador PettingZoo

- **Data:** 2026-11-04
- **Status:** aceita
- **Fase:** F2
- **Seções afetadas:** ARQUITETURA §2.1, §2.2, §2.4, §6.2, §7.5, §9.2

## Contexto

O §2.2 define o `GenomeParallelEnv`, mas deixa em aberto como ele obtém o que os contratos de camadas o impedem de importar, o que acontece com causas de morte simultâneas e a partir de que tick conta o incremento de fitness. Além disso, a leitura do `pettingzoo.test.parallel_api_test` (versão 1.27) mostrou um conflito com o reuso de slots do §2.2.

## Decisão

1. **Injeção em vez de importação.** O §2.2 delega `render()` a `genome.viz`, mas `genome.envs` só pode importar `genome.core` (PADROES §2). O ambiente recebe no construtor os operadores de genoma, os genomas dos fundadores e um `Renderer` (protocolo de `genome.envs`). `genome.lab.environments.make_env` faz a ligação. Os contratos não mudam.
2. **`founder_bodies` no núcleo.** A função passa de `genome.lab.ecology` para `genome.core.founders`, para que os dois caminhos sorteiem os corpos dos fundadores no mesmo ponto do fluxo.
3. **`options["controller"]`** é uma fábrica `(config, streams) -> Brain`, porque o controlador aleatório precisa do fluxo criaturas do mundo que o `reset` acabou de criar. A ação recebida em `step` tem precedência; quem não recebe ação segue o controlador. Sem controlador, faltar ação para um agente vivo é erro.
4. **Semente.** `reset(seed=None)` repete a última seed; sem nenhuma seed anterior, é erro. Não existe seed vinda do relógio (§2.5).
5. **Chaves desconhecidas em `options` são ignoradas**, porque o `parallel_api_test` chama `reset(options={"options": 1})`.
6. **Causas simultâneas.** `death_cause` é uma string única, com as causas na ordem da tabela de §7.5, unidas por `+` (por exemplo `starvation+dehydration`). Assim, o campo único do §2.2 registra todas as causas, como o §7.5 exige.
7. **Recompensa (§9.2).** `T_ref = t_max`. O termo de `F_base` conta nos ticks em que a criatura envelhece, de 1 a `t_vivo`: o recém-nascido e o morto do tick não somam `F_base`, mas o morto ainda recebe o bônus de um filho nascido naquele tick. O bônus usa `D = n_filhos + λ · n_netos` da própria criatura.
8. **`n_grandchildren`** é um novo atributo do estado, listado no §9.2. Um neto credita o avô só se ele ainda estiver vivo, porque um morto não tem linha na tabela. A soma por `founder_id` do fim da geração será tratada na F4.
9. **Observação de quem morreu.** Quem morre no tick não sente mais, então recebe a última observação que já tinha recebido.
10. **Sensores antecipados.** `Simulation.observe` guarda as entradas do tick seguinte, e o `step_batch` as reutiliza, para o adaptador não pagar os sensores duas vezes. A chuva passa a ser estendida no fim do tick, para que sentir antes não mude o estado numa fronteira de bloco.
11. **Conflito entre o `parallel_api_test` e o §2.2.** O teste oficial afirma que um `agent` terminado nunca volta, enquanto o §2.2 define agentes como slots reutilizados. Mantivemos o §2.2. O teste oficial roda em três cenários sem reuso: até o truncamento, até a extinção e com nascimentos em slots novos. O reuso tem teste próprio, e a equivalência com o caminho rápido cobre mundos com reuso.
12. **Ações externas.** Valores fora das faixas são limitados ao extremo mais próximo, como o §6.2 manda. `NaN` não tem extremo mais próximo e continuaria `NaN` depois do limite, então é erro.

## Consequências

- Um mundo com reuso de slot reprovaria o `parallel_api_test` por construção, e isso não indica defeito. Bibliotecas externas que assumam a mesma regra devem usar `infos[a]["uid"]` como identidade, como o §2.2 já pede.
- `state()` devolve um dicionário de arrays, não o array único do `ParallelEnv`, porque o §2.2 o define como snapshot para inspeção.
- `mode="continuous"` continua levantando `NotImplementedError` até a F3 (ADR 0004).
