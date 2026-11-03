# 0002. Parâmetros da configuração que não estão nas tabelas

- **Data:** 2026-11-03
- **Status:** aceita
- **Fase:** F0
- **Seções afetadas:** ARQUITETURA §2.3, §2.5, §3 a §12

## Contexto

AGENTS.md proíbe números mágicos: todo parâmetro vem da configuração (§2.3). Vários valores, porém, aparecem só no texto corrido ou nas fórmulas da ARQUITETURA, fora das tabelas de parâmetros. Além disso, §2.3 lista as seções da configuração, mas não diz em qual delas cada parâmetro fica, nem como se escrevem ângulos e seeds.

## Decisão

1. **Valores do texto viram campos**, com o mesmo valor da especificação. Nenhum valor foi alterado.

| Seção | Campo | Valor | Origem |
| --- | --- | --- | --- |
| `climate` | `rain_block_ticks` | 100 000 | §2.5 |
| `water` | `k_r_max` | 0,25 | §3.3, r\_max = 0,25 · √V\_max |
| `water` | `v_max_min`, `v_max_max` | 1000; 5000 | §3.7, faixa de `V_max` |
| `plants` | `b_max` | 3 | §4.1 |
| `plants` | `irrigation_hop_factor` | 1,1 | §4.2 |
| `plants` | `min_spacing_factor` | 0,9 | §4.4 |
| `creatures` | `mass_min`/`mass_max`, `v_max_min`/`v_max_max` | 0,5–2,0; 0,5–1,5 | §5.1 |
| `creatures` | `body_radius_exponent`, `e_max_per_mass`, `h_max_per_mass` | 0,5; 100; 60 | §5.1 |
| `creatures` | `t_vida_ref`, `t_vida_exponent`, `t_mat_fraction`, `touch_margin` | 3000; 0,25; 0,1; 0,5 | §5.1 |
| `creatures` | `founder_energy_fraction`, `founder_hydration_fraction` | 0,6; 0,8 | §5.6 |
| `sensors` | `fov_half_deg`, `edge_window_deg` | 90; 15 | §5.2 |
| `sensors` | `receptor_mu_deg`, `receptor_sigma_deg` | tabela de §5.2 | §5.2 |
| `sensors` | `nostril_body_factor`, `nostril_offset`, `eps_olf` | 2; 1; 10⁻⁶ | §5.3 |
| `metabolism` | `kleiber_exponent`, `surface_exponent` | 3/4; 2/3 | §7.1, §7.2 |
| `metabolism` | `eat_threshold`, `drink_threshold`, `reproduce_threshold` | 0,5; 0,5; 0,7 | §6.2, §7.4 |
| `metabolism` | `repr_min_energy_fraction`, `repr_min_hydration_fraction` | 0,8; 0,6 | §7.4 |
| `metabolism` | `child_energy_fraction`, `child_hydration_fraction` | 0,4; 0,4 | §7.4 |
| `neat` | `sparse_min_connections`, `sparse_max_connections` | 15; 20 | §8.1 |
| `neat` | `init_weight_sigma`, `reproduce_bias_init`, `founder_mass`, `founder_v_max` | 1; −2; 1; 1 | §6.1, §8.1 |
| `neat` | taxas e desvios dos operadores | tabela de §8.3 | §8.3 |
| `neat` | `crossover_disabled_prob` | 0,75 | §8.4 |
| `neat` | `delta_increase`, `delta_decrease`, `protected_top_species` | 1,05; 0,95; 2 | §8.5, §8.6 |
| `neat` | `import_mutation_rate` | 0 | §11.4 |
| `curriculum` | `phase1_plant_multiplier`, `phase1_p_sol_multiplier`, `phase1_cost_multiplier` | 2; 2; 0,7 | §9.3 |
| `lab` | `n_seeds_min` | 10 | §10.3 |
| `lab` | `immobile_death_fraction` | 0,95 | §9.3, §10.2 |
| `lab` | `ledger_tolerance`, `ledger_debug` | 10⁻⁹; falso | §7.6 |

2. **Nomes:** os campos das tabelas usam o nome da tabela em `snake_case` minúsculo (PADROES §4). Nomes que colidiriam com palavras reservadas ou ficariam ambíguos ganham sufixo: `lambda` vira `lambda_grandchildren`. Os novos campos têm nomes descritivos em inglês.
3. **Ângulos em graus no YAML** (`*_deg`), como na especificação. A configuração expõe propriedades `*_rad` para uso na simulação (§1.3).
4. **Seeds:** `root_seed` não faz parte da configuração. Ela é argumento da execução (§2.2, `reset(seed)`) e fica no manifesto (§11.7). Assim, o `config_hash` é o mesmo para todas as seeds de uma condição. `lab_seed` fica em `LabConfig`, por ser fixa por protocolo, com o valor 20261103, confirmado pelo usuário.
5. **Chaves dos fluxos aleatórios:** todo `spawn_key` tem a forma fixa `(domínio, g, k, fluxo)` (§2.5), com zero nas posições que o domínio não usa:
   - evolução: `(EVOLUTION, 0, 0, 0)`;
   - contínuo: `(CONTINUOUS, 0, 0, fluxo)`;
   - validação e teste: `(VALIDATION | TEST, 0, i, fluxo)`, com a entropia `lab_seed`.
6. **Validações entre seções no modo treino:** `run.max_ticks = training.t_max` e `world.initial_population = training.n` (§12.1).
7. **Obstáculos:** `world.n_obstacles` precisa ser 0 na v1 (§3.5). Os parâmetros geométricos dos obstáculos entram quando a extensão de §13.2 for especificada.

## Consequências

- Nenhum sistema da F1 em diante precisa de constante numérica própria: todo valor chega pela configuração.
- Mudar qualquer um desses valores muda o `config_hash`, como deve ser.
- `neat.topology_genesis` começa como `dense_starter`, porque a especificação não define um padrão. A escolha deve ser revista na F3.
- A regra de §9.3 para "plantas iniciais em manchas espaçadas" não tinha parâmetros numéricos na especificação. Eles foram definidos no ADR 0003.
