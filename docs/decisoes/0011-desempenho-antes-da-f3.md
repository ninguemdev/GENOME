# 0011. Desempenho antes da F3

- **Data:** 2026-11-05
- **Status:** aceita
- **Fase:** F2 (issue #14)
- **Seções afetadas:** ARQUITETURA §12.5, §12.6

## Contexto

O usuário decidiu otimizar antes da F3, para aproveitar todos os núcleos com uso de até ~80% do processador. Na `main` de partida (`42ab83f`), com 600 fundadores, o rendimento parava em ~191 ticks/s com 10 processos e não subia mais, com ~3800 page faults por tick. A bateria confirmatória levaria centenas de horas a esse ritmo.

Nenhuma regra do mundo pode mudar. O hash do estado tem que ficar idêntico ao da `main` de partida, com qualquer número de processos.

Este ADR registra o bloco 1, um tick mais barato e sem paralelismo. Os blocos 2 e 3 o completam quando chegarem.

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
