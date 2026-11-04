# 0009. Telemetria, checkpoints e manifesto

- **Data:** 2026-11-04
- **Status:** aceita
- **Fase:** F2
- **Seções afetadas:** ARQUITETURA §2.1, §11.1, §11.5, §11.6, §11.7

## Contexto

O §11 define o que registrar, mas deixa pontos em aberto: como guardar vários checkpoints com os nomes fixos de §11.1, o que fazer com `G_ckpt` antes de existirem gerações, o que uma linha de `ticks` resume e como uma queda afeta um arquivo Parquet, que fica ilegível sem o rodapé. Estas decisões foram aprovadas no plano do bloco F2.2.

## Decisão

1. **Um diretório por checkpoint.** `checkpoints/checkpoint_<tick>/` guarda `checkpoint_<tick>.npz`, `genomes.json` e `rng_state.json`. Ele é gravado à parte e renomeado no fim, então uma queda nunca deixa metade de um checkpoint com o nome final.
2. **Frequência até a F4.** Sem gerações, um mundo é a execução inteira, e os checkpoints seguem `T_ckpt` ticks em qualquer modo. O `G_ckpt` entra com o laço de gerações.
3. **Totais dos balanços.** O núcleo fecha os balanços a cada tick e não acumula totais, porque nada depende deles. O checkpoint guarda os acumuladores do intervalo atual da telemetria e quantos blocos valem.
4. **Linhas de `ticks`.** Há uma linha nos ticks com `(tick + 1) % T_log == 0`, mais uma no fim se sobrar intervalo. `births` e `deaths` somam o intervalo, e os erros de balanço guardam o pior valor dele, para um erro não sumir entre duas amostras. O resto é o estado no fim do tick. `mean_energy` e `mean_hydration` são nulos num mundo vazio.
5. **Fundadores em `events`.** Fundadores são eventos `birth` com `tick = 0`, `parent_a` nulo e `reproduction` nula, porque sem eles a genealogia não teria raiz.
6. **Colunas futuras.** `species_count`, `species_id` e `parent_b` existem e ficam nulas até a F3, para o esquema não mudar. `reproduction` é `asexual` nos filhos.
7. **Blocos durante a execução.** Uma execução grava `ticks.parquet.parts/` e `events.parquet.parts/`. O checkpoint força a gravação e registra quantos blocos valem, a retomada apaga os posteriores e o fim junta tudo em `ticks.parquet` e `events.parquet`, em blocos de `flush_rows`. Uma execução já juntada não pode ser retomada.
8. **Manifesto.** `manifest.yaml` nasce com `outcome: interrupted` e só o fim o substitui. Uma interrupção por teclado grava `finished_at`; uma queda o deixa nulo. A retomada mantém `started_at`. `protocol_version` e `import_divergences` ficam vazios até a F5 e a F3.
9. **Campos a mais no manifesto.** `controller` diz qual controlador a retomada deve recriar. `resumed` lista cada retomada com tick, horário, commit e `dirty`, porque o código pode ter mudado desde o início.
10. **Restauração sobre um mundo recém-criado.** `Simulation.restore` age sobre uma simulação criada com a mesma seed e configuração. Os fluxos voltam no lugar, porque o controlador aleatório e as plantas guardam referências a eles. A geografia é comparada, e um snapshot de outro mundo é erro.
11. **Genomas.** `genome.io` recebe um `GenomeCodec`, porque não pode importar `genome.lab`. O genoma de um controlador fixo não tem conteúdo; o formato real chega com o NEAT na F3.

## Consequências

- O `config.yaml` da execução é a fonte da retomada, e `load_config` o relê com o mesmo `config_hash`. Carregar um checkpoint com outra configuração é erro.
- Uma execução interrompida não tem `ticks.parquet` nem `events.parquet` até ser retomada e terminar; os blocos ficam legíveis em `*.parts/`.
- O `T_ckpt` de 50 000 ticks é maior que os 10 000 ticks do preset da F1, então esse preset não grava checkpoints sem um `t_ckpt` menor. O valor de §11.8 foi mantido, porque foi pensado para o modo contínuo.
- O checkpoint cobre o caminho rápido. O `GenomeParallelEnv` não é retomável: `_seen_descendants` e as ações carregadas não são salvos.
- Os blocos em disco acrescentam uma etapa ao fim da execução. O custo da telemetria entra no relatório de desempenho do F2.3.
