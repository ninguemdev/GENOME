# 0003. Parâmetros das manchas de plantas da Fase 1

- **Data:** 2026-11-03
- **Status:** aceita
- **Fase:** F0 (parâmetros); o comportamento é implementado na F4
- **Seções afetadas:** ARQUITETURA §4.6, §9.3, §9.6

## Contexto

Na Fase 1 do currículo, as plantas iniciais (×2) ficam em "manchas espaçadas" (§4.6, §9.3), para que a abundância exija deslocamento entre fontes de comida. A especificação não define quantas manchas, de que tamanho, nem a que distância.

## Decisão

Três campos novos em `CurriculumConfig`. São pontos de partida, editáveis no YAML antes de cada execução e calibrados na F4:

| Parâmetro | Valor | Descrição |
| --- | --- | --- |
| `phase1_patch_count` | 16 | Número de manchas; as plantas iniciais da Fase 1 são divididas igualmente entre elas |
| `phase1_patch_radius` | 22 | Raio de cada mancha |
| `phase1_patch_gap` | 24 | Distância mínima entre as bordas de duas manchas |

Motivos dos valores, no preset de treino (256 × 256, 2 rios, 3 lagos):

- **Plantas por mancha:** 600 × 2 = 1200 plantas iniciais em 16 manchas dão 75 plantas por mancha.
- **Raio 22:** a irrigação alcança cerca de 16 unidades a partir da margem da água (`r_irr` = 6, mais 3 saltos de 1,1 · `d_p`). Por isso, só a faixa irrigável do círculo recebe plantas, uma área útil de cerca de 600 unidades². Na rede cristalina de `d_p` = 3 (uma planta a cada ~7,8 unidades²), isso comporta cerca de 77 plantas.
- **Espaço 24 entre bordas:** é maior que o raio de visão à noite (`r_vis` · `m_noite` = 20) e menor que o de dia (40). De dia, a próxima mancha é visível da borda da atual; à noite, não.
- **Cabem todas:** com centros a pelo menos 2 · 22 + 24 = 68 unidades, 16 manchas ocupam cerca de 1100 unidades de margem, menos que as ~1300 de faixa irrigável ao longo dos rios e lagos.

## Consequências

- A F4 implementa a geração das manchas pelo fluxo plantas: centros na faixa irrigável e plantas nas posições válidas de §4.4 dentro do raio. O que fazer quando uma mancha não comporta sua cota (redistribuir ou perder o excedente) é decidido lá, com ADR.
- As estimativas acima são grosseiras: a faixa irrigável depende da geometria sorteada dos rios e lagos. A calibração da F4 deve medir a ocupação real das manchas.
- No modo contínuo esses campos existem, mas não têm efeito, porque o currículo só vale no modo treino.
