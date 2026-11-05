# 0010. Visualizador e medição de desempenho

- **Data:** 2026-11-04
- **Status:** aceita
- **Fase:** F2
- **Seções afetadas:** ARQUITETURA §2.1, §2.2, §11.5, §11.6, §12.6; ADR 0009

## Contexto

O §2.1 diz que o visualizador recebe snapshots e nunca escreve no estado, mas não define o formato desses snapshots, onde fica o laço interativo nem como a dependência opcional entra nas verificações. O §12.6 pede o desempenho medido com 1000 criaturas, num modo contínuo que só existe a partir da F3. A revisão do PR #12 também apontou falhas na retomada que mudam decisões do ADR 0009.

## Decisão

1. **Entrada do `genome.viz`.** O visualizador recebe o mapeamento de arrays nomeados que `GenomeParallelEnv.state` devolve, e não `SimulationSnapshot`, que também clonaria genomas e fluxos a cada quadro. O laboratório passa cópias, então o desenho não alcança os arrays do mundo.
2. **Duas camadas.** `viz/scene.py` e `viz/inspector.py` não importam pygame, e a lógica é testada sem tela. `viz/renderer.py` é o único módulo do projeto que importa pygame.
3. **Laço interativo em `genome.lab`.** O `viz` não pode importar o `lab`, e só o `lab` avança o mundo. O renderizador traduz teclas e cliques em `Command` e `Click`, e o `lab/viewer.py` decide o que eles significam.
4. **Entradas do inspetor.** O laboratório passa uma cópia do resultado de `Simulation.observe()` só com uma criatura selecionada. Os sensores não sorteiam nem alteram nada, e o tick seguinte reutiliza o original, que a tela não alcança. Pelo `render()` do ambiente, o inspetor não recebe entradas.
5. **Geometria vinda do núcleo.** Os lóbulos usam o kernel `retina_signals`, e as narinas e os raios da água usam `nostril_offsets` e `water_radius`, expostos neste bloco sem mudar nenhum resultado. O visualizador não tem cópia dessas fórmulas.
6. **Rios como discos.** Os arrays guardam só os vértices, sem dizer onde cada rio termina, então os rios são discos sobre os vértices, numa máscara refeita apenas quando a geografia muda.
7. **Laços de desenho.** Criaturas, lagos e poças são desenhados com um laço Python, fora do núcleo. A regra de não ter laço por criatura vale para o caminho quente da simulação, e o custo do quadro foi medido.
8. **`pygame-ce` no extra `viz`.** Segue PADROES §8. O grupo `dev` inclui `genome[viz]`, para os testes e os tipos rodarem em toda verificação sem mudar os comandos nem o CI. `make_env` importa o renderizador só quando há modo de renderização.
9. **Cenários do benchmark.** 200 e 600 criaturas usam o mundo de treino. 1000 criaturas usam a geografia de `configs/continuous.yaml` em `mode: training`, na densidade de §12.4, porque o modo contínuo só existe a partir da F3.
10. **Ajustes no ADR 0009, vindos da revisão do PR #12.**
    - O fim da execução junta as duas tabelas e grava o manifesto antes de apagar os blocos, para um fim interrompido poder ser retomado.
    - Retomar de um checkpoint apaga os posteriores, porque os blocos que eles contaram deixam de existir. Eles saem antes dos blocos, para uma queda no meio não deixar um checkpoint apontando para blocos apagados.
    - `restore` confere tudo antes da primeira escrita, e os blocos só são apagados depois dele.
    - A execução constrói o mundo antes de gravar qualquer arquivo.
    - O `dirty` do manifesto ignora o diretório da própria execução.

## Consequências

- Assistir não muda nenhum resultado: um teste compara o hash do estado com e sem visualizador, com criatura selecionada.
- A janela real não é coberta pelos testes automáticos, que rodam com `SDL_VIDEODRIVER=dummy`.
- Pelo `render()` do ambiente não há seleção de criatura; o inspetor só existe no visualizador interativo.
- O tick mediu de ~2 a ~8 vezes a estimativa de §12.6, conforme o cenário. Os números estão em `docs/relatorios/desempenho-f2.md`, com a decisão sobre otimizar antes da F3 em aberto.
- `fsync` nos checkpoints continua fora: uma queda de energia ainda pode corromper o último checkpoint, e não há recuo automático para o anterior.
