# 0011. Desempenho antes da F3

- **Data:** 2026-11-05
- **Status:** aceita
- **Fase:** F2 (issue #14)
- **Seções afetadas:** ARQUITETURA §11.7, §11.8, §12.5, §12.6

## Contexto

O usuário decidiu otimizar antes da F3, para aproveitar todos os núcleos com uso de até ~80% do processador. Na `main` de partida (`42ab83f`), com 600 fundadores, o rendimento parava em ~191 ticks/s com 10 processos e não subia mais, com ~3800 page faults por tick. A bateria confirmatória levaria centenas de horas a esse ritmo.

Nenhuma regra do mundo pode mudar. O hash do estado tem que ficar idêntico ao da `main` de partida, com qualquer número de processos.

Este ADR registra o bloco 1, um tick mais barato e sem paralelismo, e o bloco 2, vários mundos em paralelo. O bloco 3 o completa quando chegar.

## Decisão

1. **Pares em grade uniforme toroidal.** Um kernel Numba substitui `sparse_distance_matrix` e o `argsort` nos sensores e grava os pares já ordenados por observador e alvo. A grade gera candidatos a mais, com uma folga relativa ao raio e ao lado do mundo, e aplica o mesmo filtro exato de distância. Assim o arredondamento nunca perde um par.
2. **Buffers reaproveitados.** Pares e pesos ficam em `PairBuffer`, que dura entre os ticks e dobra quando falta espaço. Eles não entram no hash nem nos checkpoints, e o que contêm só vale até a consulta seguinte.
3. **Retina sem array intermediário.** O kernel calcula e agrega os sinais no mesmo laço. `retina_signals` continua existindo para o inspetor.
4. **Água em kernels.** Rios, lagos e poças usam kernels escalares, em vez de arrays de todo ponto contra todo corpo. A árvore de vértices dos rios continua: trocá-la poderia mudar quais candidatos entram em caso de empate, e o teste bit a bit não garantiria isso em todo mundo.
5. **Sem cache de `centerline`.** O plano previa avaliar um cache de posições dentro do tick. Depois do kernel, cada chamada custa ~0,29 ms. O tato e a corrente são o único par que lê as mesmas posições, então a economia seria de ~2% de um tick de ~14 ms, ao preço de estado escondido que precisaria ser invalidado. Fica de fora.
6. **Comparação bit a bit como teste.** Cada kernel novo é comparado por igualdade exata com a forma em arrays que substituiu, além do hash em 35 pontos de referência (200, 600 e 1000 fundadores, duas seeds).

## Consequências

- Os kernels usam `np.hypot` e `%` do Numba. A igualdade com o NumPy foi verificada no Windows. Como os testes bit a bit fazem parte da suíte, o CI em Linux os cobre quando voltar a rodar.
- Em Numba, uma divisão por zero levanta `ZeroDivisionError`, em vez de produzir `NaN`. Um rio com segmento de comprimento zero, que antes propagava `NaN` em silêncio, agora falha na hora. Nenhuma configuração do repositório gera esse caso.
- As árvores continuam fora do caminho quente, exceto a de vértices dos rios. A de amostras serve ao olfato, e as de plantas, à irrigação e ao espaçamento, a cada `T_irr` ticks ou quando há nascimentos.
- **O que limitava a escala era sobretudo a memória nova por tick.** Sem os page faults, o rendimento volta a crescer com os núcleos. O que resta vem da topologia do processador: um núcleo de eficiência rende ~0,5 de um de desempenho, e o segundo thread de um núcleo acrescenta ~0,39. Os números ficam em `docs/relatorios/desempenho-pre-f3.md`, no fechamento da issue.

## Bloco 2: vários mundos em paralelo

### Decisão

1. **Um processo por mundo, em vez de `ProcessPoolExecutor`.** Nesse executor, um processo que morre de forma abrupta quebra o pool inteiro e derruba os outros mundos. `genome.lab.pool` abre um processo `spawn` por mundo, e cada um termina como `ok`, `failed` (exceção), `crashed` (código de saída) ou `not_started` (Ctrl+C antes da vez dele). O custo é carregar os módulos a cada mundo: um mundo de um tick leva ~2,5 s do início ao fim, com o cache do Numba já pronto.
2. **Teto de `floor(cpu_fraction × núcleos lógicos)` processos.** Com `lab.cpu_fraction: 0.8`, são 12 no i5-13400, que mediram ~72% de CPU. Com 600 fundadores, 12 processos rendem 535 ticks/s, contra 656 com 16. Quem quiser esses ~18% a mais sobe `cpu_fraction`; o padrão respeita o teto pedido pelo usuário.
3. **`lab.process_priority`, com `above_normal` nos presets.** Vale só para os processos do executor. No Windows, usa `SetPriorityClass`; em outros sistemas, reduz o `nice`, e sem permissão o processo fica em `normal`. `high` e `realtime` ficam de fora, porque podem deixar o resto da máquina sem resposta. O valor pedido e o aplicado de fato vão para o `batch.yaml`.
4. **Retomada de bateria.** `--resume` lê o manifesto de cada seed: sem manifesto, começa; `completed` ou `extinction`, pula; `interrupted` com checkpoint, retoma; `interrupted` sem checkpoint, falha com mensagem e não apaga nada. Sem `--resume`, encontrar uma seed já gravada é erro. Configuração, seed ou controlador diferentes do gravado são recusados antes de abrir qualquer processo.
5. **`batch.yaml` à parte.** O formato de `manifest.yaml` não muda, e ele continua sendo a fonte da retomada. O `batch.yaml` é regravado de forma atômica a cada mundo que termina.

### Medição

i5-13400 com 16 processadores lógicos, 15,7 GB de RAM, Chrome e Parsec abertos, no Windows.

**Bateria real:** `f1_ecology`, 20 seeds, 10 000 ticks cada, heurístico, com gravação, 12 processos em `above_normal`.

| Medida | Resultado |
| --- | --- |
| Tempo total | 264 s (757 ticks/s) |
| Desfecho | 20 seeds `completed`, 20 hashes distintos |
| CPU total | mediana de 74%, máximo de 83% |
| Pico de RAM (soma dos conjuntos de trabalho) | ~2,1 GB em 14 processos Python |
| RAM livre mínima | 4,4 GB |

Os 757 ticks/s não se comparam diretamente com o `bench` (1244 ticks/s, com 200 fundadores e 12 processos): a bateria inclui a partida de cada mundo e a gravação, e o `bench` mede 500 ticks. A diferença não foi decomposta.

**Prioridade:** 600 fundadores, 12 processos, 500 ticks, em ticks/s.

| Situação | `normal` | `above_normal` |
| --- | --- | --- |
| Máquina ociosa | 572, 492 | 574, 532 |
| 16 processos ocupados concorrendo em `normal` | 218, 199 | 331 |

Com a máquina ociosa, o ganho fica dentro do ruído: as duas medições em `normal` já diferem em ~16%. A prioridade só importa quando outro programa disputa a CPU, e aí rendeu ~59% a mais. Esse número vem de uma única medição em `above_normal`, porque a segunda foi interrompida pelo sistema por falta de memória; o valor indica a direção, não o tamanho exato do ganho.

**Determinismo e queda:**
- Os 35 pontos de referência seguem iguais aos da `main` de partida.
- O hash de uma seed é o mesmo dentro e fora do executor, com 1 e 2 processos, em `normal` e `above_normal`.
- Todos os processos de uma bateria de 4 seeds foram encerrados à força depois de 35 s. O `--resume` retomou as quatro do checkpoint do tick 2000 e chegou aos mesmos 4 hashes de uma bateria sem queda.

### Consequências

- **`config_hash` muda.** Os dois campos novos entram nele, como toda a configuração (§2.2). Execuções anteriores têm um `config.yaml` sem esses campos e deixam de ser retomáveis. Mudar `cpu_fraction` ou `process_priority` também muda o hash, embora o resultado da simulação seja o mesmo, e portanto impede a retomada com outro valor. Não há execução de valor a preservar antes da F3.
- **Encerrar à força só o processo pai** (pelo Gerenciador de Tarefas, por exemplo) deixa os filhos rodando sozinhos, e um `--resume` lançado nesse momento escreveria nos mesmos diretórios. Antes de retomar, é preciso confirmar que os processos antigos acabaram. Job Objects resolveriam isso no Windows, mas ficam de fora enquanto o caso não custar mais do que esse cuidado.
- **Ctrl+C dá 60 s aos mundos em andamento** para gravarem `interrupted` e encerra à força os que restarem. Um mundo encerrado assim mantém o desfecho `interrupted` com que nasceu e retoma do último checkpoint.
- **A retomada só devolve o que foi gravado.** Com o `T_ckpt` de 50 000 ticks do `f1_ecology`, um mundo de 10 000 ticks não grava checkpoint: se cair, o `--resume` informa que aquela seed não tem de onde voltar. Baterias longas precisam de um `T_ckpt` proporcional ao que se aceita perder.
- **Nada rodou em Linux.** A redução do `nice` e a medição acima valem para o que foi testado no Windows; o CI cobre os testes quando voltar a rodar.
