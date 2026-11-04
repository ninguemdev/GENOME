# 0005. Sentidos e estado inicial das criaturas

- **Data:** 2026-11-04
- **Status:** aceita
- **Fase:** F1
- **Seções afetadas:** ARQUITETURA §5.2, §5.3, §5.6, §5.7, §6.3

## Contexto

As seções de sentidos e locomoção descrevem os resultados esperados, mas deixam alguns detalhes em aberto: a tabela da visão não tem linha para criaturas, a fração de água por célula do olfato não tem algoritmo, a grade não fixa onde cada valor fica, e o estado inicial não diz como sortear as posições. As escolhas 1 a 6 foram aprovadas pelo usuário no plano do bloco 2 da F1. A escolha 7 surgiu na revisão do PR do bloco, ainda aguarda a confirmação dele e pode mudar sem afetar as demais.

## Decisão

1. **Distância às criaturas na visão (§5.2).** É medida até o centro da outra criatura e a direção também aponta para o centro, como nas plantas.
2. **Fração de água por célula (§5.3).** Cada célula é amostrada em `olf_water_subsamples` × `olf_water_subsamples` pontos, nos centros de uma subgrade regular. Um ponto vale o maior peso entre as águas que o cobrem: 1 nos rios, V/V\_max nos lagos e V/V\_p nas poças. A fração da célula é a média dos pontos. Os rios são rasterizados uma vez; lagos e poças são recalculados a cada `T_olf`, só numa janela em torno de cada corpo d'água.
3. **Depósito e leitura das grades (§5.3).** A biomassa de uma planta entra inteira na célula ⌊x / `cell_olf`⌋, ⌊y / `cell_olf`⌋. Na interpolação bilinear, o valor de uma célula fica no centro dela, e os índices são toroidais.
4. **Água dos fundadores (§5.6).** "Qualquer água" é a água existente em t = 0, medida por `Water.margin_distance`. As poças começam vazias (ADR 0004), então as posições de poça não restringem o nascimento.
5. **Posições dos fundadores (§5.6).** Pontos uniformes do fluxo criaturas, em lotes de tamanho fixo, são aceitos em ordem até completar a população. Depois, as orientações são sorteadas em U\[−π, π) do mesmo fluxo. Criaturas podem nascer sobrepostas (§6.3).
6. **Instante do comando (§6.3).** A velocidade-alvo usa a orientação φ\_t e a correnteza em P\_t. Depois são atualizadas a velocidade e, com ela, a posição e a orientação.
7. **Objeto na mesma posição (§5.2).** Um vetor nulo não tem direção, então o objeto conta como bem à frente (θ = 0), em qualquer orientação. Sem essa regra, `arctan2(0, 0)` daria θ = −φ, e o objeto mais próximo possível ficaria invisível para quem olhasse para o lado errado. Isso pode ocorrer: dois filhos de um genitor imóvel nascem no mesmo ponto (§7.4).

## Consequências

- Novo campo em `SensorConfig`: `olf_water_subsamples` (4), adicionado à tabela de §5.7.
- Em cada tick, o número de pontos testados na janela de um corpo d'água cresce com a área dele, não com o tamanho do mundo.
- Se os lotes de candidatos se esgotarem, a criação dos fundadores falha com erro explícito em vez de criar uma população incompleta.
