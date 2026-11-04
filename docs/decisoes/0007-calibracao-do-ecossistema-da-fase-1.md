# 0007. Calibração do ecossistema da Fase 1

- **Data:** 2026-11-04
- **Status:** aceita
- **Fase:** F1
- **Seções afetadas:** ARQUITETURA §4.7, §10.2, §10.6, §12.2, §12.3

## Contexto

Com o tick completo e os valores da especificação, o heurístico não passava no critério de saída de §13.1. Os dados abaixo vêm de execuções exploratórias no preset `f1_ecology` (10 000 ticks, 200 fundadores, teto de 600). As primeiras linhas foram medidas com o heurístico mirando a média dos receptores; as três últimas, com o código final, que mira o receptor mais forte, como §10.2 pede.

| Configuração | Seeds | Resultado |
| --- | --- | --- |
| Especificação, heurístico bebendo só com H < 0,5 | 1 a 3 | Nenhum nascimento: H para em ~0,57 e nunca alcança os 0,7 exigidos. Extinção em 3001 ticks; as plantas acabam em 2 das 3 seeds |
| Heurístico bebendo até H = 0,9 | 1 a 4 | Surto (~300 nascimentos) e extinção em ~1300 ticks |
| `p_sol` = 0,016 (×4), sem freio de densidade | 1, 2 | Teto em 78% a 86% dos ticks; extinção perto do tick 8000, com a onda de nascimentos após a morte de uma geração sincronizada |
| `p_sol` = 0,024, limite de densidade 0,02 | 1 a 5 | Sobrevive, mas seeds 1 e 5 ficam em 11% dos ticks no teto, por causa do surto dos fundadores |
| **Final:** `p_sol` = 0,008, limite 0,02, E ≥ 0,93 | 1, 5, 6, 7 | Extinção em todas, entre 2769 e 3874 ticks |
| **Final:** `p_sol` = 0,012, limite 0,02, E ≥ 0,93 | 1 a 12 | Seed 8 extinta no tick 3081; as outras 11 chegam ao fim |
| **Final:** `p_sol` = 0,016, limite 0,02, E ≥ 0,93 | 1 a 12 | **Sem extinção e 0% dos ticks no teto em todas**; população final de 176 a 330 |

O motivo do colapso: a produção das plantas é proporcional ao número de plantas, então o pasto é instável. Quando o consumo passa da produção, as plantas somem e não voltam, porque só uma planta viva propaga. A §12.3 previa N ≈ 0,35 · N\_pp ≈ 210 para 600 plantas, sem margem para os 200 fundadores.

## Decisão

1. **`p_sol` passa de 0,004 para 0,016.** O valor 0,012 passaria nas 5 seeds do critério, mas falha na seed 8. Escolhi o dobro do valor que colapsa em todas as seeds. Vale para todos os presets, e `p_chuva` não muda.
2. **O heurístico continua bebendo até H = 0,9 enquanto toca a água** (`heuristic_drink_until`). A busca por água continua começando em H < 0,5, como em §10.2. Sem isso, a regra de §10.2 e os limiares da ADR 0004 são incompatíveis.
3. **O heurístico só reproduz com E ≥ 0,93** (`heuristic_reproduce_energy`, antes 0,9), o que corta o surto dos fundadores. Como ele só come com E < 0,9, a reserva vem da última mordida; a validação da configuração recusa valores que uma mordida não alcança.
4. **O heurístico só reproduz com pouca densidade à vista**: a maior entrada de densidade fica em até 0,02 (`heuristic_crowding_limit`). Ele usa só a visão e torna os nascimentos dependentes da população.

Nenhuma mecânica nem fórmula mudou. Os itens 2 a 4 são regras do controlador, não do mundo.

## Consequências

- **A variação entre mundos é grande.** A seed 11 passa, mas com população de 112 a 287 e menos de 900 plantas. O critério de saída com 5 seeds não garante todos os mundos: execuções longas podem encontrar seeds que colapsam, e isso é um resultado válido (§1.1).
- **§10.2 e §12.3 ganham números novos.** A produção por planta irrigada sobe de ~0,046 para ~0,170 E/tick, então a capacidade de suporte passa de ~0,35 · N\_pp para ~1,3 · N\_pp. O valor real é menor, pois inclui plantas saturadas.
- **O pasto é uma propriedade também dos genomas.** Cérebros evoluídos na F3 podem pastar diferente do heurístico, e a calibração deve ser revista com o controle imóvel e o heurístico antes das baterias da F5 (§10.1).
- **O freio de densidade é uma ajuda do controle heurístico.** Ele mostra o que uma estratégia razoável pode alcançar, e o ambiente não impõe nada disso às redes.
