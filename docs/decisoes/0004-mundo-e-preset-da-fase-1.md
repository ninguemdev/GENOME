# 0004. Geração do mundo e preset da Fase 1

- **Data:** 2026-11-04
- **Status:** aceita
- **Fase:** F1
- **Seções afetadas:** ARQUITETURA §2.4, §3.2 a §3.7, §4.6, §5.6, §10.2, §13.1

## Contexto

A F1 implementa o mundo e o ecossistema com controladores fixos. Várias regras da ARQUITETURA descrevem o resultado esperado sem fixar o algoritmo, e o critério de saída da fase (10 000 ticks com reprodução clonal) não cabe em nenhum dos presets de §12.1: o treino exige `max_ticks = t_max = 2000`, e o modo contínuo aplica mutação a todo nascimento (§8.7), o que só existe a partir da F3. As escolhas 1 a 4 foram confirmadas pelo usuário.

## Decisão

1. **Preset `configs/f1_ecology.yaml`.** É o preset de treino com `t_max = max_ticks = 10 000`. Os nascimentos são clones (`in_world_mutation_rate = 0`, via `GenomeOperators.clone`). Mutação no mundo, reprodução sexuada e o modo contínuo levantam `NotImplementedError` até a F3.
2. **Rios em meandro periódico (§3.2).** Cada rio sorteia, do fluxo mundo, um eixo (horizontal ou vertical), um sentido de correnteza, a posição perpendicular inicial e, para cada harmônico k = 1…`river_harmonics`, uma amplitude em U(0, `river_amplitude` / k) e uma fase em U(0, 2π). O deslocamento perpendicular é a soma dos senos de período W/k (ou H/k), então a curva é suave e fecha exatamente na costura do toro. A linha poligonal tem um vértice a cada `river_vertex_step` ao longo do eixo.
3. **Distância ao rio.** A distância à linha central é a menor distância aos dois segmentos vizinhos do vértice mais próximo (consulta numa `cKDTree` dos vértices). Com `river_vertex_step` ≤ `s_rio` ≤ `w_rio`/2, o erro é desprezível em curvas suaves.
4. **Plantas iniciais por aceitação sequencial (§4.6).** Pontos uniformes do fluxo plantas são testados em ordem. Um ponto é aceito se cumpre a validade de §4.4 (espaçamento e fora das áreas máximas de água) e se seria irrigado pelo que já foi aceito: a até `r_irr` da margem da água (h = 0) ou a até 1,1 · `d_p` de uma planta aceita com h < `k_max` (h = menor h vizinho + 1). Depois, b ∈ {1, …, `b_max`} é sorteado do mesmo fluxo.
5. **Reprodução do controlador heurístico (§10.2).** Ele tenta reproduzir só com folga: E ≥ 0,9 · E\_max e H ≥ 0,7 · H\_max. Os limiares entram na configuração no bloco que implementa os controladores.
6. **Lagos e poças.** Centros por rejeição, do fluxo mundo: um lago fica fora dos rios (distância à linha central ≥ `w_rio`/2 + r\_max) e não sobrepõe o r\_max de outro lago; uma posição de poça fica fora dos rios, a pelo menos r\_max + `r_p_max` de cada lago e 2 · `r_p_max` de outra poça. V\_max ~ U(`v_max_min`, `v_max_max`). As poças começam vazias, porque o mundo começa sem chuva (§3.6).
7. **Instante do clima no passo de ambiente (§2.4, passo 8).** Lagos, poças, umidade e plantas usam o clima do tick t (L(t), chuva(t), I\_sol(t)); só depois o índice do clima avança para t + 1. A chuva(0) é seca, e a transição t → t + 1 consome o sorteio t da sequência pré-calculada.
8. **Sorteio de crescimento à noite.** Cada planta irrigada sorteia u em todo tick, mesmo com p(t) = 0, para que o consumo do fluxo plantas não dependa da hora.

## Consequências

- Novos campos em `WaterConfig`: `river_harmonics` (3), `river_amplitude` (16) e `river_vertex_step` (1), adicionados à tabela de §3.7.
- Tentativas de posicionamento por rejeição têm um limite técnico nomeado no módulo; se esgotado, a geração falha com erro explícito em vez de criar um mundo incompleto.
- O preset da F1 só serve à calibração e ao critério de saída; o fitness do treino (§9.2) não é usado nele.
- Com o meandro periódico, dois rios de eixos diferentes sempre se cruzam. Isso é aceito: o cruzamento é água contínua.
