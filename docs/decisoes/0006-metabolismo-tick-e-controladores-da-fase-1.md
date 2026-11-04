# 0006. Metabolismo, tick e controladores da Fase 1

- **Data:** 2026-11-04
- **Status:** aceita
- **Fase:** F1
- **Seções afetadas:** ARQUITETURA §2.4, §7.1 a §7.5, §10.2, §13.1

## Contexto

O último bloco da F1 fecha o ecossistema. A especificação define as fórmulas e a ordem do tick, mas não fixa alguns instantes, desempates e o significado de "presa no teto" no critério de saída. Além disso, os controles de §10.2 foram descritos pensando no NEAT, que só existe a partir da F3. Todas as escolhas abaixo foram aprovadas pelo usuário no plano do bloco.

## Decisão

1. **Instante do metabolismo.** R\_H e C\_H usam H\_t, do início do tick, e I\_sol(t) e U\_t. Comer e beber entram no passo 5, numa única atualização com os custos, então o resultado de §7.1 e §7.2 não depende de ordem interna.
2. **Requisitos de reprodução (§7.4).** São avaliados com as reservas do fim do passo 3, antes de comer e beber, como as demais interações (§2.4).
3. **Genitor que morre no mesmo tick.** O filho nasce mesmo assim, porque o custo já foi pago no passo 5. Na prática só ocorre por velhice, pois as reservas exigidas superam o custo.
4. **Idade do recém-nascido.** Quem nasce no passo 7 não envelhece no passo 10. O filho sente pela primeira vez com idade 0, como §7.4 pede e como os fundadores.
5. **Teto da população e slots.** A vaga é contada no passo 4, antes das mortes. Os filhos são inseridos antes de liberar os slots dos mortos do tick, de modo que um slot nunca muda de ocupante no tick da morte (§2.2).
6. **Empate na fonte de água (§7.3).** Rio, depois lagos e, por último, poças, cada grupo por ID.
7. **Controlador aleatório na F1.** Sorteia as 7 saídas a cada tick, do fluxo criaturas. O controle de §10.2, com pesos sorteados em `dense_starter`, depende do NEAT e fica para a F3.
8. **Controlador heurístico.** Segue as regras de §10.2 só com as entradas de §5.5, sem olfato. Busca água quando H < 0,5 e há água visível, e comida caso contrário; come ao tocar planta com E < 0,9; explora andando e girando. A exploração usa o registrador de memória como fase de oscilação, então não consome aleatoriedade. Os parâmetros ficam em `LabConfig`.
9. **"Presa no teto" no critério de saída (§13.1).** A seed reprova se a população ficar em `max_population` em mais de 10% dos ticks.

## Consequências

- Corpo do filho clonal: é copiado do genitor. Quando houver mutação no mundo (F3), o corpo precisará vir do genoma.
- Mutação no mundo, reprodução sexuada e modo contínuo continuam levantando `NotImplementedError` até a F3 (ADR 0004).
- Os ajustes de calibração da F1 ficam em ADRs próprios, com os dados que os justificam.
