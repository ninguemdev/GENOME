# Relatório de desempenho da F2

Mede o custo por tick com 200, 600 e 1000 criaturas, para confirmar ou corrigir as estimativas de ARQUITETURA §12.6.

**Resultado:** o tick custa de ~2 a ~8 vezes o estimado, conforme o cenário, antes mesmo de existirem redes neurais. O critério de saída da F2 pede só a medição, então nada foi otimizado neste bloco, mas o prazo das baterias de §12.6 precisa ser revisto antes da F3.

## Como foi medido

- **Máquina:** Intel Core i5-13400, 16 GB de RAM, Windows 11. Um processo, sem paralelismo.
- **Código:** commit `8e5fffb`, com o `scripts/bench.py` deste PR.
- **Comando:** `uv run python scripts/bench.py --population 200 600 1000 --ticks 1000 --repeats 3`
- **Execução:** seed 1 e controlador heurístico, 1000 ticks por rodada (um ciclo inteiro de dia e noite), depois de 5 ticks de aquecimento fora do relógio. O total é a mediana de 3 rodadas.
- **Sistemas:** vêm de uma quarta rodada sob o profiler e valem como proporção do total.

## Custo por tick

| Criaturas fundadoras | Mundo | Rodadas (ms/tick) | Mediana (ms/tick) | Estimativa de §12.6 | Razão | População final |
| --- | --- | --- | --- | --- | --- | --- |
| 200 | 256 × 256 | 7,20 · 7,18 · 7,24 | **7,20** | ~2 a 4 ms | ~1,8 a 3,6× | 324 |
| 600 | 256 × 256 | 16,09 · 15,75 · 15,65 | **15,75** | ~2 a 4 ms | ~3,9 a 7,9× | 600 (teto) |
| 1000 | 512 × 512 | 30,89 · 30,58 · 31,12 | **30,89** | ~6 a 10 ms | ~3,1 a 5,1× | 1200 (teto) |

A variação entre rodadas ficou abaixo de 3%.

## Onde o tempo vai

| Sistema | 200 | 600 | 1000 |
| --- | --- | --- | --- |
| Sensores (total) | 64,2% (4,62 ms) | 79,7% (12,56 ms) | 72,7% (22,46 ms) |
| · visão | 31,2% (2,25 ms) | 40,7% (6,41 ms) | 36,1% (11,16 ms) |
| · pares (`build_scene`) | 18,3% (1,32 ms) | 24,1% (3,79 ms) | 21,4% (6,60 ms) |
| · toque | 10,3% (0,74 ms) | 11,8% (1,86 ms) | 12,9% (4,00 ms) |
| · olfato | 1,6% (0,11 ms) | 1,2% (0,18 ms) | 0,8% (0,26 ms) |
| Locomoção | 8,1% (0,58 ms) | 7,8% (1,23 ms) | 7,6% (2,36 ms) |
| Comer | 3,3% (0,24 ms) | 1,5% (0,24 ms) | 2,2% (0,68 ms) |
| Beber | 4,0% (0,29 ms) | 2,2% (0,34 ms) | 1,7% (0,54 ms) |
| Metabolismo | 1,1% (0,08 ms) | 0,6% (0,10 ms) | 0,4% (0,12 ms) |
| Ambiente | 12,7% (0,92 ms) | 3,3% (0,51 ms) | 12,0% (3,72 ms) |

Os sensores são o gargalo em todos os cenários. Visão e geração de pares somam de 50% a 65% do tick.

## Custos em torno do tick

| | 200 | 600 | 1000 |
| --- | --- | --- | --- |
| Tick com telemetria (ms/tick) | 6,97 | 15,34 | 31,14 |
| Diferença para o tick sem telemetria | −0,23 | −0,41 | +0,25 |
| Um quadro do visualizador, sem tela (ms) | 7,91 | 10,30 | 14,38 |

- **Telemetria:** o custo ficou dentro do ruído da medição, com diferenças negativas em dois cenários. Dá para afirmar que ele é pequeno diante do tick, mas não dá para dar um número. Isso inclui gravar os blocos e juntá-los no fim, sem checkpoints.
- **Visualizador:** cabe nos 33 ms de um quadro a 30 FPS em todos os cenários. Somando o tick, 1000 criaturas a 1 tick por quadro dão ~45 ms, ou ~22 FPS. O quadro foi medido em `rgb_array` e sem criatura selecionada; apresentar em janela real não foi medido.

## O que isso muda nas estimativas de §12.6

Contas lineares a partir das medianas acima, **sem o custo das redes e supondo que 14 processos escalam sem perda**, o que ainda não foi medido. São pisos, não previsões:

| Cenário de §12.6 | Estimativa | Com os números medidos |
| --- | --- | --- |
| Uma geração: 3 mundos em paralelo × 2000 ticks | ~4 a 8 s | ~14 a 32 s |
| 1 milhão de ticks com ~1000 criaturas | ~2 a 3 h | ~8,6 h |
| Bateria confirmatória: ~260 milhões de ticks em 14 processos | até ~15 h | ~37 a 81 h |
| Bateria exploratória: 5 seeds, até 50 gerações por fase (~1 h a ~3 ms) | ~1 h | ~2,4 a 5,3 h |

## Limitações

- **Controlador heurístico.** Ele não tem o custo de avaliar redes, que chega na F3, e o próprio controlador entra no tempo medido, sem ter sido medido em separado.
- **A população não fica fixa.** Os cenários são rotulados pelos fundadores: o de 200 termina em 324, e os outros dois chegam ao teto. O custo por tick reflete essa população em movimento.
- **1000 criaturas não é o modo contínuo real.** O cenário usa a geografia de `configs/continuous.yaml` em `mode: training`, porque o modo contínuo só existe a partir da F3 (ADR 0010).
- **Uma máquina, uma seed, uma medição.** Windows 11, sem repetir em Linux nem com outras seeds.
- **Sem paralelismo.** O custo de 14 processos disputando cache e memória não foi medido.

## Decisão pendente antes da F3

Vale decidir se a otimização dos sensores (visão e pares) entra antes do NEAT ou se os prazos de §12.6 são aceitos como estão. O caminho quente mais evidente é a geração e o filtro de pares, mas nenhuma otimização foi tentada, então o ganho possível é desconhecido.
