# GENOME — Documento de Arquitetura e Planejamento

**Genetic Evolution & Neural Organism Modeling Environment** · Ambiente de Modelagem de Organismos Neurais e Evolução Genética

Nov 3, 2026 · @dionel

## 1. Visão geral

O GENOME é um laboratório de vida artificial e neuroevolução em Python: criaturas com cérebros NEAT evoluem num mundo 2D toroidal com água, clima e plantas, exposto como `ParallelEnv` do PettingZoo. Ele sucede o EDEN, trocando a Unity por um núcleo de simulação próprio, determinístico e vetorizado.

### 1.1 Princípios

1. **Separação estrita.** O núcleo de simulação (NumPy, SciPy, Numba) não depende de bibliotecas de RL nem gráficas. O adaptador PettingZoo e o visualizador apenas leem e escrevem pela interface do núcleo. O visual nunca altera o estado.
2. **Determinismo absoluto.** Toda aleatoriedade deriva da seed raiz por fluxos independentes (2.5). O passo de tempo é fixo, e a velocidade de execução ou de visualização não altera nenhum resultado.
3. **Inspecionabilidade total.** Energia, água, sensores, ativações da rede, genealogia e IDs de gene de qualquer indivíduo são auditáveis. Qualquer instante pode ser reconstruído por re-simulação.
4. **Evolução avaliada contra controles.** Resultados só valem comparados a controles (aleatório, heurístico, imóvel, modelos nulos, sem currículo) em seeds pareadas (10).
5. **Preservação de dados.** Extinções, estagnações e colapsos são resultados válidos, registrados em telemetria e checkpoints.
6. **Conservação verificável.** Os balanços de energia e de água fecham a cada tick (7.6). Um desvio indica erro de simulação.
7. **Eficiência por construção.** Arrays por atributo, consultas espaciais vetorizadas e redes compiladas. Registra-se pouco e re-simula-se para inspecionar.

### 1.2 Escopo da v1

| Na v1 | Fora da v1 (13.2) |
| --- | --- |
| Mundo contínuo e toroidal; rios com correnteza, lagos e poças | Predação (Fase 4 do currículo) |
| Ciclo dia/noite, chuva markoviana, umidade do ar | Obstáculos perceptíveis (obstáculos ficam desativados) |
| Plantas com biomassa, irrigação por saltos e propagação cristalina | rtNEAT no modo contínuo |
| Criaturas com visão, olfato, toque, memória e locomoção com inércia | Mutação de funções de ativação |
| Genes corporais evoluíveis (massa, velocidade, cor) | Carcaças, crescimento corporal, comunicação |
| Reprodução assexuada e sexuada | Aceleração por GPU |
| NEAT próprio; modo treino geracional com currículo de 3 fases; modo contínuo |  |
| Controles, protocolo estatístico, telemetria Parquet, formatos de genoma |  |

### 1.3 Convenções

- **Tempo:** 1 tick = Δt = 1. Todas as taxas são por tick.
- **Espaço:** unidades de mundo, x para a direita e y para baixo (coordenadas de tela). Ângulos em radianos, crescendo no sentido horário a partir de +x. Em relação à criatura, ângulo positivo é à direita.
- **Criatura de referência:** massa M = 1 e velocidade máxima v\_max = 1 unidade/tick. Valores "de referência" se referem a ela.
- **Números:** todos os valores são pontos de partida de calibração, versionados pelo protocolo (10.1).
- **Idioma:** documento em português; código, chaves de arquivo e colunas de telemetria em inglês.

## 2. Arquitetura de software

O núcleo de simulação é independente do PettingZoo. O `ParallelEnv` é um adaptador fino sobre ele, e o laboratório chama o núcleo diretamente no caminho rápido, sem dicionários por agente.

&#91;embedded content: arquitetura · 7 pacotes e o que chama o quê\]

As setas indicam quem chama quem. O núcleo só depende de `genome.neat`, usado nos nascimentos; o visualizador recebe snapshots e nunca escreve no estado.

### 2.1 Regras de dependência

- `genome.core` importa apenas genome.neat (operadores de genoma e compilação de redes, usados nos nascimentos), e genome.neat não importa nenhum outro pacote do GENOME.
- `genome.envs`, `genome.lab` e `genome.viz` acessam o núcleo só pela sua interface pública (`reset`, `step_batch`, `snapshot`, `restore`).
- `genome.viz` é somente leitura: recebe snapshots e nunca escreve no estado.
- O estado das criaturas fica em arrays por atributo (posição `(N, 2)`, energia `(N,)` e assim por diante), pré-alocados com capacidade `max_population` e uma máscara de vivos. Vagas livres são reutilizadas por uma lista de slots.

### 2.2 Adaptador PettingZoo: `GenomeParallelEnv`

| Elemento | Definição |
| --- | --- |
| `metadata` | `{"name": "genome_v1", "render_modes": ["human", "rgb_array"]}` |
| `possible_agents` | Slots `creature_0000` … `creature_{max_population−1}`. Um slot só é reutilizado depois que o término do ocupante anterior foi reportado. |
| `agents` | Slots ocupados por criaturas vivas. A identidade permanente é `infos[a]["uid"]`. |
| `observation_space(a)` | `Box(shape=(42,), dtype=float32)` com os limites de cada entrada (5.5). |
| `action_space(a)` | `Box(shape=(7,))`, mínimos `[−1, −1, −1, 0, 0, 0, 0]`, máximos `1` (6.2). |
| `reset(seed, options)` | `seed` é a seed raiz. `options`: `mode`, `split` (`train`, `val`, `test`), `generation`, `world_index`, `founders` (genomas) ou `controller`. Devolve `(observations, infos)`. |
| `step(actions)` | Devolve `observations, rewards, terminations, truncations, infos`. |
| `render()` | Delegado a `genome.viz`. |
| `state()` | Snapshot global do mundo, para inspeção. |

**Semântica do passo:**

- Uma criatura nascida no tick t entra em `agents` e recebe sua primeira observação no retorno de `step` desse tick.
- `terminations[a] = True` no tick da morte, com `infos[a]["death_cause"]` (`starvation`, `dehydration`, `old_age`).
- `truncations[a] = True` para todos quando `max_ticks` é atingido.
- `rewards[a]` é o incremento de fitness da própria criatura no tick (9.2): a soma ao longo da vida dá F\_base mais o bônus de reprodução acumulado enquanto ela está viva. O fitness oficial do NEAT é sempre o calculado pelo laboratório no fim da geração.
- `infos[a]` inclui `uid`, `founder_id`, `born` e `death_cause`.

**Equivalência dos caminhos:** a mesma seed e as mesmas ações produzem estados idênticos pelo `GenomeParallelEnv` e pelo caminho rápido (`core.step_batch`). Um teste automatizado garante isso, e outro roda `pettingzoo.test.parallel_api_test`.

### 2.3 Configuração

A configuração é um conjunto de dataclasses imutáveis carregadas de YAML: `RunConfig`, `WorldConfig`, `ClimateConfig`, `WaterConfig`, `PlantConfig`, `CreatureConfig`, `SensorConfig`, `MetabolismConfig`, `NEATConfig`, `TrainingConfig`, `CurriculumConfig` e `LabConfig`. O `config_hash` é o SHA-256 do JSON canônico (chaves ordenadas) da configuração completa.

```python
from dataclasses import dataclass
from typing import Literal, Optional

@dataclass(frozen=True)
class RunConfig:
    mode: Literal["training", "continuous"]
    max_ticks: Optional[int]        # None = sem limite (modo contínuo)
    terminate_on_extinction: bool
    max_population: int
```

### 2.4 Ordem do tick

A ordem é fixa e faz parte do determinismo. As interações do passo 4 são resolvidas sobre o estado do fim do passo 3, sem depender da ordem das criaturas.

1. **Sensores:** observações calculadas com o estado e o clima do tick t.
2. **Redes:** avaliação em lote de todas as redes.
3. **Locomoção:** inércia, correnteza, integração da posição e da orientação.
4. **Interações simultâneas:** comer, beber e reproduzir (7.3, 7.4).
5. **Metabolismo:** custos, recuperação pela água, perdas de água.
6. **Mortes:** fome, desidratação, velhice.
7. **Nascimentos:** inserção dos filhos, que agem a partir do tick seguinte.
8. **Ambiente:** clima avança para t+1; umidade; lagos; poças; irrigação (a cada `T_irr`); plantas; campos de olfato (a cada `T_olf`).
9. **Contabilidade e telemetria:** fechamento dos balanços e registro.
10. **Avanço:** t ← t + 1 e idade de cada criatura + 1.

### 2.5 Determinismo e seeds

Toda aleatoriedade vem de quatro fluxos por mundo (mundo, clima, plantas, criaturas). O fluxo das criaturas tem um segundo nível por execução, `evolucao`, que guarda os operadores do NEAT entre gerações. As seeds de validação e teste vêm de uma seed de laboratório separada, fixa por protocolo, e por isso nunca coincidem com seeds de treino.

```text
seed_raiz (uma por execução)                   seed_laboratorio (uma por protocolo)
├── ("evolucao")  operadores do NEAT           ├── ("validacao", i)  i = 0..2
│                 entre gerações               └── ("teste", i)      i = 0..9
├── ("treino", g, k)  geração g, mundo k=0..2        mesmos 4 fluxos por mundo
│     ├── mundo      rios, lagos, obstáculos, posições de poças
│     ├── clima      sequência da chuva
│     ├── plantas    posições iniciais e sorteios de crescimento
│     └── criaturas  posições e orientações iniciais, mutação e crossover no mundo
└── ("continuo")      mesmos 4 fluxos
```

**Implementação:** `SeedSequence(entropia, spawn_key=(DOMINIO, g, k, FLUXO))`, com constantes inteiras para domínio e fluxo, e `default_rng` (PCG64).

**Regras obrigatórias:**

- `float64` em todo o núcleo. Numba sem `fastmath` e sem reduções paralelas (`prange`) em somas, que mudam a ordem de soma.
- Nenhuma iteração sobre `dict` ou `set` no núcleo. Arrays ordenados por `uid`; o contador de `uid` só cresce.
- A sequência da chuva é pré-calculada em blocos de 10⁵ ticks, sempre na mesma ordem.
- As plantas sorteiam um número por planta irrigada por tick, em ordem de ID.
- Checkpoints guardam `bit_generator.state` de cada fluxo.
- **Testes:** hash do estado após 1000 ticks igual em duas execuções e em processos diferentes; execução retomada de checkpoint igual à execução direta.

## 3. Mundo

O mundo é um toro contínuo com três tipos de água (rios infinitos, lagos finitos e poças temporárias), um ciclo dia/noite determinístico, chuva markoviana e umidade do ar. Dimensões e quantidades de cada elemento dependem do preset (12.1).

### 3.1 Espaço toroidal

O vetor entre dois pontos usa sempre a menor imagem toroidal:

```latex
\Delta x = \big((x_b - x_a + W/2) \bmod W\big) - W/2, \qquad \Delta y = \big((y_b - y_a + H/2) \bmod H\big) - H/2, \qquad d = \sqrt{\Delta x^2 + \Delta y^2}
```

- Depois de integrar posições: x ← x mod W e, se x = W por arredondamento, x ← 0. O mesmo vale para y. A `cKDTree` com `boxsize` exige coordenadas em \[0, W).
- Todo raio de percepção é menor que min(W, H)/2, para que nada seja visto "pelos dois lados".

### 3.2 Rios

- **Geometria:** estática, gerada pelo fluxo mundo. Cada rio é uma linha poligonal que atravessa o mapa (passeio com direção suavizada), com largura `w_rio`.
- **Água:** volume infinito. Criaturas podem entrar e beber livremente.
- **Correnteza:** campo estático. A direção é a tangente t̂ do segmento mais próximo, no sentido de geração; a intensidade é máxima no centro e zero nas margens:

```latex
\vec v_{\text{corr}} = v_{\text{corr,max}} \left( 1 - \left( \frac{2 d_c}{w_{\text{rio}}} \right)^2 \right) \hat t, \qquad d_c < \frac{w_{\text{rio}}}{2}
```

Aqui d\_c é a distância à linha central. A correnteza entra na locomoção (6.3).

- **Percepção:** a linha central é amostrada a cada `s_rio` ≤ `w_rio`/2. Cada ponto entra nos sensores com distância até a margem, max(0, d − w\_rio/2). Os pontos ficam numa árvore espacial estática.

### 3.3 Lagos

- Centros e volumes máximos `V_max` são sorteados pelo fluxo mundo. Os lagos começam cheios.
- O raio acompanha o volume atual, com área proporcional ao volume: r(t) = r\_max · √(V/V\_max), onde r\_max = 0,25 · √V\_max.
- Regeneração por tick, proporcional ao tamanho do lago e reforçada pela chuva:

```latex
V_{t+1} = \min\big(V_{\max},\ V_t + \rho_L \, V_{\max} \, \gamma(t)\big) - \text{consumo}, \qquad \gamma(t) = \begin{cases} \gamma_{\text{chuva}} & \text{com chuva} \\ 1 & \text{sem chuva} \end{cases}
```

- O consumo é dividido entre as criaturas (7.3).

### 3.4 Poças

- As posições elegíveis são sorteadas pelo fluxo mundo, em terra, fora de rios, lagos e obstáculos. Todas as poças são iguais.
- Durante a chuva, toda posição elegível fica com V = `V_p`.
- Fora da chuva, cada poça evapora ε\_p · I\_sol(t) por tick (3.6). Ela some em V = 0, ou antes, se for consumida.
- Raio: r = r\_p,max · √(V/V\_p).

### 3.5 Obstáculos

- Círculos estáticos gerados pelo fluxo mundo. Bloqueiam plantas e movimento: a posição é projetada para fora e a componente normal da velocidade é zerada.
- **Desativados por padrão na v1** (`n_obstaculos = 0`), porque nenhum sensor os percebe. A ativação depende da extensão de 13.2.

### 3.6 Clima

**Luminosidade:** determinística, com crepúsculos suaves para não criar descontinuidades nos sensores. Sendo τ = t mod T\_ciclo:

```latex
L(t) = \begin{cases} \min\left(1,\ \dfrac{\tau}{T_{\text{crep}}},\ \dfrac{T_{\text{dia}} - \tau}{T_{\text{crep}}}\right) & \tau < T_{\text{dia}} \\[6pt] 0 & \tau \ge T_{\text{dia}} \end{cases}
```

**Chuva:** cadeia de Markov de dois estados, independente do dia, com sequência pré-calculada do fluxo clima. P(seco → chuva) = `p_sc` e P(chuva → seco) = `p_cs`. Com os valores padrão, chove 25% do tempo, em episódios médios de 200 ticks separados por 600 ticks secos. Cada mundo começa sem chuva.

**Sol efetivo:** I\_sol(t) = L(t) · (1 − chuva(t)).

**Umidade do ar:** aproxima-se exponencialmente de um alvo que depende da chuva, começando em `U_seco`:

```latex
U_{t+1} = U_t + (U^* - U_t)\left(1 - e^{-1/\tau_U}\right), \qquad U^* = \begin{cases} U_{\text{chuva}} & \text{com chuva} \\ U_{\text{seco}} & \text{sem chuva} \end{cases}
```

| Fator | Afeta |
| --- | --- |
| L(t) | Crescimento das plantas (4.3), raio de visão (5.2) |
| Chuva | Regeneração dos lagos, enchimento das poças, crescimento das plantas |
| I\_sol(t) | Evaporação das poças, perda de água das criaturas (7.2) |
| U(t) | Perda de água das criaturas (7.2) |

### 3.7 Parâmetros

| Parâmetro | Valor | Descrição |
| --- | --- | --- |
| `w_rio` | 8 | Largura dos rios |
| `v_corr_max` | 0,3 | Correnteza no centro do rio (unidades/tick) |
| `s_rio` | 4 | Espaçamento da amostragem dos rios |
| `V_max` | 1000 a 5000 | Faixa sorteada do volume máximo dos lagos |
| `rho_L` | 0,0005 | Regeneração do lago por tick, como fração de V\_max |
| `gamma_chuva` | 3 | Multiplicador da regeneração na chuva |
| `V_p` | 30 | Volume de uma poça cheia |
| `r_p_max` | 3 | Raio de uma poça cheia |
| `eps_p` | 0,1 | Evaporação da poça por tick de sol pleno (\~300 ticks) |
| `T_ciclo`, `T_dia`, `T_crep` | 1000, 600, 50 | Ciclo, duração do dia e do crepúsculo (ticks) |
| `p_sc`, `p_cs` | 1/600, 1/200 | Transições da chuva por tick |
| `tau_U` | 100 | Constante de tempo da umidade (ticks) |
| `U_chuva`, `U_seco` | 1,0; 0,2 | Alvos da umidade |

## 4. Plantas

Cada planta guarda até 3 unidades de biomassa (stacks), cresce só quando irrigada e, cheia, propaga-se em rede cristalina. O espaço livre e o alcance da irrigação são o limite natural de densidade.

### 4.1 Estado

- **Biomassa** b ∈ (0, 3\]. O stack exibido é ⌈b⌉, e cada unidade de biomassa vale `E_stack` de energia.
- Posição fixa, ID e salto de irrigação h.
- Criaturas atravessam plantas. A planta morre quando b ≤ 0, por consumo ou murcha.

### 4.2 Irrigação por saltos

- **h = 0:** a planta está a até `r_irr` da margem de alguma água (rio, lago no raio atual, poça existente).
- **h = k + 1:** a planta está a até 1,1 · `d_p` de uma planta com h = k.
- Até `k_max`. Acima disso, a planta não é irrigada.
- O cálculo é uma busca em largura sobre os pares de vizinhança da `cKDTree`, refeita a cada `T_irr` ticks, porque lagos encolhem e poças secam.

O limite de saltos impede que o transporte entre plantas espalhe a irrigação pelo mapa inteiro.

### 4.3 Crescimento

A cada tick, cada planta irrigada sorteia u \~ U(0, 1) do fluxo plantas, em ordem de ID, e cresce se u < p(t):

```latex
p(t) = L(t) \cdot \begin{cases} p_{\text{sol}} & \text{sem chuva} \\ p_{\text{chuva}} & \text{com chuva} \end{cases}
```

As chances são fixas e dependem só da condição atual; o sorteio vem do fluxo com seed. À noite (L = 0) nada cresce.

**Evento de crescimento:**

- Se b < 3: b ← min(3, b + 1).
- Se b = 3: a planta propaga (4.4) e não muda.

Cada unidade de biomassa criada entra na contabilidade como energia solar (7.6).

### 4.4 Propagação cristalina

1. Os candidatos são 6 posições à distância `d_p` da planta, nos ângulos 60° · k.
2. Eles são testados na ordem k = (s + m) mod 6, com m = 0, 1, …, 5 e s = ID mod 6. O ponto de partida girado pelo ID evita que as plantas avancem sempre na mesma direção.
3. Um candidato é válido se:
   - não há planta a menos de 0,9 · `d_p`;
   - está fora de rios, do raio máximo dos lagos (r\_max), do raio máximo das posições de poça (r\_p,max) e de obstáculos;
   - fica irrigado (h\_pai < k\_max, ou a até `r_irr` de água).
4. O primeiro candidato válido recebe uma planta nova com b = 1. Se nenhum for válido, o evento é perdido.

Como nenhuma planta nasce na área máxima de um lago ou poça, nenhuma é afogada quando a água volta a subir. O teto `max_plants` existe só como proteção contra erros.

### 4.5 Murcha

Uma planta sem irrigação perde 1/`T_murcha` de biomassa por tick, um stack a cada `T_murcha` ticks. A biomassa perdida sai do sistema como murcha (7.6).

### 4.6 Estado inicial

- `n_plantas_0` plantas (preset, 12.1) em posições válidas sorteadas pelo fluxo plantas, uniformes sobre a área irrigável.
- Biomassa inicial b ∈ {1, 2, 3}, sorteada do mesmo fluxo.
- Na Fase 1 do currículo, as plantas iniciais ficam concentradas em manchas (9.3).

### 4.7 Parâmetros

| Parâmetro | Valor | Descrição |
| --- | --- | --- |
| `E_stack` | 25 | Energia por unidade de biomassa |
| `d_p` | 3 | Distância entre plantas na rede cristalina |
| `r_irr` | 6 | Alcance da irrigação direta |
| `k_max` | 3 | Saltos máximos de irrigação entre plantas |
| `T_irr` | 10 | Intervalo de recálculo da irrigação (ticks) |
| `p_sol` | 0,004 | Chance de crescer por tick sob sol pleno |
| `p_chuva` | 0,0015 | Chance de crescer por tick de dia com chuva |
| `T_murcha` | 200 | Ticks para perder um stack sem irrigação |

## 5. Criaturas: corpo e sentidos

O corpo vem de três genes (massa, velocidade máxima e cor), e todo o resto é derivado deles por alometria. A criatura percebe o mundo por 42 entradas: visão frontal com retina de 6 receptores, olfato de 360°, toque, propriocepção, memória e clima.

### 5.1 Genes corporais e alometria

| Gene | Faixa | Mutação |
| --- | --- | --- |
| `mass` (M) | \[0,5; 2,0\] | Log-normal multiplicativa, limitada à faixa (8.3) |
| `v_max` | \[0,5; 1,5\] | Gaussiana, limitada à faixa (8.3) |
| `color` (c) | \[0, 1) circular | Gaussiana com *wrap-around* (8.3) |

| Grandeza derivada | Fórmula | Referência (M = 1) | Motivo |
| --- | --- | --- | --- |
| Raio do corpo r\_c | M^(1/2) | 1,0 | Área proporcional à massa em 2D |
| Energia máxima E\_max | 100 · M | 100 | Reserva proporcional à massa |
| Água máxima H\_max | 60 · M | 60 | Reserva proporcional à massa |
| Expectativa de vida T\_vida | 3000 · M^(1/4) | 3000 | Alometria biológica; o mínimo, 2523, passa de t\_max |
| Maturidade T\_mat | 0,1 · T\_vida | 300 |  |
| Inércia linear τ | 3 · M | 3 ticks | Corpos maiores mudam de velocidade mais devagar |
| Inércia de giro τ\_rot | 2 · M | 2 ticks |  |
| Alcance de toque r\_t | r\_c + 0,5 | 1,5 | Comer, beber, achar parceiro |

A cor é um marcador de linhagem: não afeta o corpo, define a cor no visualizador (matiz HSV) e a afinidade entre criaturas.

### 5.2 Visão

**Raio dinâmico:** a visão piora à noite.

```latex
r_{\text{vis}}(t) = r_{\text{vis}} \big( m_{\text{noite}} + (1 - m_{\text{noite}})\, L(t) \big)
```

**Visibilidade:** um objeto é visto se d < r\_vis(t) e |θ| ≤ 90°, onde θ é o ângulo relativo à orientação, positivo à direita.

**Receptores:** cada canal visual tem 6 receptores sobrepostos. A sensibilidade é gaussiana, com uma janela que zera na borda do campo, e a distância atenua linearmente até zero no limite do raio:

```latex
g_j(\theta) = w(\theta)\, e^{-\frac{(\theta - \mu_j)^2}{2\sigma_j^2}}, \qquad w(\theta) = \min\left(1,\ \frac{90^\circ - |\theta|}{15^\circ}\right), \qquad f(d) = 1 - \frac{d}{r_{\text{vis}}(t)}
```

| Receptor j | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| Centro μ\_j | −70° | −35° | −10° | +10° | +35° | +70° |
| Largura σ\_j | 20° | 14° | 8° | 8° | 14° | 20° |

Os receptores centrais são estreitos (fóvea) e os periféricos, largos. Um objeto que cruza o campo passa de um receptor para o vizinho sem saltos, inclusive ao entrar pela borda.

**Canais:**

```latex
a^{\text{comida}}_j,\ a^{\text{água}}_j = \max_i \frac{W_i}{W_{\max}} f(d_i)\, g_j(\theta_i), \qquad a^{\text{dens}}_j = 1 - e^{-\sum_i f(d_i)\, g_j(\theta_i)}, \qquad a^{\text{afin}}_j = \max_i \operatorname{sim}(c, c_i)\, f(d_i)\, g_j(\theta_i)
```

- **Comida e água (máximo):** o receptor indica o melhor alvo na sua direção. O máximo independe da ordem dos objetos.
- **Densidade (soma saturada):** quantos indivíduos há na direção.
- **Afinidade (máximo):** o indivíduo mais aparentado e mais próximo na direção.
- A própria criatura é excluída dos canais de criaturas.

| Objeto | Peso W\_i / W\_max | Distância d\_i | Direção θ\_i |
| --- | --- | --- | --- |
| Planta | b / 3 | Até o centro | Até o centro |
| Rio | 1,0 | Até a margem, por ponto amostrado (3.2) | Até o ponto |
| Lago | V / V\_max | max(0, d\_centro − r(t)) | Até o centro |
| Poça | V / V\_p | max(0, d\_centro − r(t)) | Até o centro |

**Similaridade de cor:**

```latex
\Delta c = \min\big(|c_1 - c_2|,\ 1 - |c_1 - c_2|\big), \qquad \operatorname{sim}(c_1, c_2) = 1 - 2\,\Delta c \in [0, 1]
```

### 5.3 Olfato

**Campos em grade** (células de `cell_olf` unidades), recalculados a cada `T_olf` ticks e suavizados por filtro gaussiano com bordas toroidais (`scipy.ndimage.gaussian_filter(mode="wrap")`, desvio `sigma_olf`):

- **Comida:** biomassa das plantas depositada em cada célula.
- **Água:** fração da célula coberta por água, ponderada por V/V\_max nos lagos e V/V\_p nas poças (rios = 1). Esta grade de umidade do solo serve só ao olfato; a irrigação usa distância (4.2).

**Narinas:** dois pontos à distância δ = 2 · r\_c + 1 à frente do corpo, a ±`alpha_olf` da orientação, lidos por interpolação bilinear (C\_E, C\_D). Cada campo gera duas entradas, intensidade e direção:

```latex
o_{\text{int}} = 1 - e^{-k_{\text{olf}}\,(C_E + C_D)/2}, \qquad o_{\text{dir}} = \frac{C_E - C_D}{C_E + C_D + \epsilon} \in [-1, 1], \qquad \epsilon = 10^{-6}
```

Direção positiva indica mais cheiro à esquerda. A direção vem normalizada, independentemente da intensidade, o que torna o gradiente legível mesmo longe da fonte.

### 5.4 Toque

| Toque | Valor 1 quando |
| --- | --- |
| Comida | Há uma planta com centro a até r\_t |
| Água | A margem de alguma água está a até r\_t, ou a criatura está dentro da água |
| Parceiro | Há outra criatura com centro a até r\_t + r\_c,j, madura e com sim(c, c\_j) ≥ `s_min` |

### 5.5 Vetor de entradas (42)

| Índice | Canal | Faixa | Descrição |
| --- | --- | --- | --- |
| 0–5 | Visão: comida | \[0, 1\] | Melhor planta por receptor |
| 6–11 | Visão: água | \[0, 1\] | Melhor fonte de água por receptor |
| 12–17 | Visão: densidade de criaturas | \[0, 1\] | Quantidade de indivíduos por receptor |
| 18–23 | Visão: afinidade | \[0, 1\] | Indivíduo mais aparentado e próximo por receptor |
| 24 | Olfato: comida, intensidade | \[0, 1\] | o\_int do campo de comida |
| 25 | Olfato: comida, direção | \[−1, 1\] | o\_dir do campo de comida |
| 26 | Olfato: água, intensidade | \[0, 1\] | o\_int do campo de água |
| 27 | Olfato: água, direção | \[−1, 1\] | o\_dir do campo de água |
| 28 | Toque: comida | {0, 1} | 5.4 |
| 29 | Toque: água | {0, 1} | 5.4 |
| 30 | Toque: parceiro | {0, 1} | 5.4 |
| 31 | Energia | \[0, 1\] | E / E\_max |
| 32 | Hidratação | \[0, 1\] | H / H\_max |
| 33 | Velocidade frontal | \[−1, 1\] | Velocidade real na direção da frente / v\_max, limitada |
| 34 | Velocidade lateral | \[−1, 1\] | Velocidade real para a direita / v\_max, limitada |
| 35 | Velocidade angular | \[−1, 1\] | ω / ω\_max, limitada |
| 36 | Maturidade | \[0, 1\] | min(1, idade / T\_mat) |
| 37 | Idade relativa | \[0, 1\] | idade / T\_vida |
| 38 | Memória | \[0, 1\] | O\_mem do tick anterior (0 no nascimento) |
| 39 | Luminosidade | \[0, 1\] | L(t) |
| 40 | Chuva | {0, 1} | Estado da chuva |
| 41 | Umidade do ar | \[0, 1\] | U(t) |

As velocidades (33–35) são as reais, incluindo inércia e correnteza: com inércia, a criatura precisa saber como está se movendo para frear e manobrar. Este layout é o `sensor_model = "genome-v1"`; qualquer mudança nele cria um novo identificador (11.2).

### 5.6 Estado inicial

- Fundadores e população inicial nascem com E = 0,6 · E\_max, H = 0,8 · H\_max, idade 0, velocidade 0 e memória 0.
- Posição e orientação são sorteadas pelo fluxo criaturas, em terra, fora de obstáculos e a pelo menos `d_min_agua` de qualquer água. Isso evita que o fitness dependa de nascer já ao lado da água.

### 5.7 Parâmetros

| Parâmetro | Valor | Descrição |
| --- | --- | --- |
| `r_vis` | 40 | Raio de visão de dia |
| `m_noite` | 0,5 | Fração do raio de visão à noite |
| `cell_olf` | 4 | Tamanho da célula das grades de olfato |
| `sigma_olf` | 3 células | Desvio do filtro gaussiano |
| `T_olf` | 10 | Intervalo de recálculo dos campos (ticks) |
| `alpha_olf` | 45° | Ângulo das narinas |
| `k_olf_food`, `k_olf_water` | 0,5; 2,0 | Escalas de saturação do olfato |
| `s_min` | 0,8 | Similaridade de cor mínima para parceiro |
| `d_min_agua` | 5 | Distância mínima da água no nascimento dos fundadores |

## 6. Rede neural, saídas e locomoção

O cérebro é uma rede sem ciclos avaliada numa única passada por tick, com 7 saídas: três eixos de movimento (frente, lado, giro), três ações e um registrador de memória. A velocidade comandada é um alvo que o corpo alcança com inércia proporcional à massa.

### 6.1 Arquitetura da rede

- **Sem ciclos (feedforward):** nenhuma mutação ou crossover pode criar um ciclo (8.3, 8.4). A rede é avaliada em ordem topológica: cada nó calcula act(bias + Σ peso · valor de origem).
- **Memória explícita:** a única memória são os registradores. A saída O\_mem do tick t volta como entrada 38 no tick t + 1. `n_mem` = 1; mudar esse valor muda o `sensor_model`.
- **Bias por nó:** cada nó oculto e de saída tem um gene de bias, que evolui como os pesos. O bias não ocupa entradas.
- **Bias inicial:** 0 em todos os nós, exceto `reproduce`, com −2,0 (saída 0,12). Comer e beber só custam quando executados (7.1), então começar no limiar não desperdiça energia e dá sinal de seleção desde a geração 0. Reproduzir é caro e começa desligado.
- **Compilação:** cada genoma vira arrays (ordem topológica, conexões de entrada em formato CSR, pesos, bias, ativação). Todas as redes vivas são avaliadas juntas numa função Numba por tick.

| Nós | Ativação | Faixa | Motivo |
| --- | --- | --- | --- |
| Saídas de movimento | tanh | \[−1, 1\] | O sinal indica o sentido; zero é repouso |
| Saídas de ação e memória | sigmoide | \[0, 1\] | Ações disparadas por limiar |
| Ocultos | tanh | \[−1, 1\] | Limitada e capaz de inibir |

### 6.2 Saídas (7)

| Ação | ID do nó | Saída | Ativação | Efeito |
| --- | --- | --- | --- | --- |
| 0 | 42 | `forward` (O\_f) | tanh | Positivo avança; negativo recua |
| 1 | 43 | `lateral` (O\_l) | tanh | Positivo desloca para a direita; negativo, para a esquerda |
| 2 | 44 | `turn` (O\_g) | tanh | Positivo gira no sentido horário; negativo, anti-horário |
| 3 | 45 | `eat` | sigmoide | Tenta comer se > 0,5 (7.3) |
| 4 | 46 | `drink` | sigmoide | Tenta beber se > 0,5 (7.3) |
| 5 | 47 | `reproduce` | sigmoide | Tenta reproduzir se > 0,7 (7.4) |
| 6 | 48 | `memory` | sigmoide | Grava o registrador de memória |

No `GenomeParallelEnv`, ações de políticas externas são limitadas a essas faixas antes de entrar no núcleo.

### 6.3 Locomoção com inércia

**Comando.** As saídas de movimento definem uma velocidade-alvo no referencial da criatura. O vetor u = (O\_f, O\_l) é normalizado se ‖u‖ > 1, para que a diagonal não seja mais rápida. Recuar e andar de lado são mais lentos:

```latex
v^*_f = v_{\max}\, u_f \cdot \begin{cases} 1 & u_f \ge 0 \\ k_{\text{trás}} & u_f < 0 \end{cases}, \qquad v^*_l = v_{\max}\, k_{\text{lat}}\, u_l, \qquad \omega^* = \omega_{\max}\, O_g
```

**Referencial do mundo.** Com a frente ĥ = (cos φ, sin φ) e a direita r̂ = (−sin φ, cos φ) em coordenadas de tela, somando a correnteza (3.2):

```latex
\vec v^* = v^*_f\, \hat h + v^*_l\, \hat r + \vec v_{\text{corr}}(\vec P)
```

**Inércia.** A velocidade real se aproxima do alvo de forma exponencial, com constantes de tempo proporcionais à massa (5.1):

```latex
\vec v_{t+1} = \vec v^* + (\vec v_t - \vec v^*)\, e^{-1/\tau}, \qquad \omega_{t+1} = \omega^* + (\omega_t - \omega^*)\, e^{-1/\tau_{\text{rot}}}
```

**Integração**, com *wrap-around* (3.1):

```latex
\vec P_{t+1} = (\vec P_t + \vec v_{t+1}) \bmod (W, H), \qquad \phi_{t+1} = \operatorname{wrap}_{[-\pi,\, \pi)}(\phi_t + \omega_{t+1})
```

**Consequências:**

- Criaturas maiores têm τ maior: demoram mais para acelerar, frear e mudar de direção.
- A distância até parar é cerca de v · τ (3 unidades na criatura de referência, o dobro do alcance de comer): frear antes da comida é uma habilidade que a evolução precisa encontrar.
- A correnteza arrasta a criatura gradualmente, também com inércia.
- A atualização é exata para arrasto linear, estável para qualquer τ e determinística. O fator e^(−1/τ) é calculado uma vez por criatura, no nascimento.

**Propriocepção:** entradas 33 a 35 = v · ĥ / v\_max, v · r̂ / v\_max e ω / ω\_max, limitadas a \[−1, 1\].

**Custo:** cobrado sobre o comando, não sobre a velocidade resultante (7.1). Deslizar por inércia não custa; acelerar, frear ativamente e nadar contra a correnteza custam.

**Contato:** criaturas se sobrepõem livremente e atravessam plantas. Obstáculos, quando ativos, bloqueiam (3.5).

### 6.4 Parâmetros

| Parâmetro | Valor | Descrição |
| --- | --- | --- |
| `omega_max` | 0,2 rad/tick | Giro máximo (volta completa em \~31 ticks) |
| `k_back` | 0,5 | Fator de velocidade ao recuar (k\_trás) |
| `k_lat` | 0,5 | Fator de velocidade lateral |
| `k_inertia` | 3 | τ = k\_inertia · M |
| `k_inertia_rot` | 2 | τ\_rot = k\_inertia\_rot · M |

## 7. Metabolismo, interações e ciclo de vida

Energia e água são reservas separadas. A comida enche a energia; a água repõe a hidratação e recupera um pouco de energia, sempre menos que o custo basal. Toda ação custa só quando executada, e os dois balanços fecham a cada tick.

### 7.1 Energia

```latex
E_{t+1} = \min\big(E_{\max},\ E_t + G_{\text{comida}} + R_H - C_{\text{basal}} - C_{\text{mov}} - C_{\text{ações}} - C_{\text{repr}}\big)
```

**Custo basal:** lei de Kleiber, mais manutenção muscular proporcional à velocidade máxima, mais senescência no fim da vida:

```latex
C_{\text{basal}} = b_0\, M^{3/4}\, s(t) + b_v\, M\, v_{\max}^2, \qquad s(t) = 1 + k_{\text{sen}} \max\left(0,\ \frac{\text{idade}/T_{\text{vida}} - f_{\text{sen}}}{1 - f_{\text{sen}}}\right)
```

**Custo de movimento:** sobre o comando normalizado u (6.3), escalado por M · v\_max². Recuar e andar de lado custam o mesmo por tick que avançar, mas rendem metade da velocidade.

```latex
C_{\text{mov}} = c_m\, M\, v_{\max}^2\, (u_f^2 + u_l^2) + c_g\, M\, O_g^2
```

**Custo das ações:** cobrado em cada tick em que a ação é executada, havendo ou não comida ou água ao alcance.

```latex
C_{\text{ações}} = C_a\, M^{3/4} \Big( \mathbb{1}[O_{\text{eat}} > 0{,}5] + \mathbb{1}[O_{\text{drink}} > 0{,}5] \Big)
```

**Recuperação pela água:**

```latex
R_H = \rho_H\, b_0\, M^{3/4}\, \frac{H}{H_{\max}}
```

Como ρ\_H < 1 e s(t) ≥ 1, vale sempre R\_H < C\_basal: **água sozinha nunca sustenta a criatura**, apenas retarda a fome. Cada unidade de energia recuperada consome χ unidades de água (7.2).

**Custo de reprodução:** C\_repr só existe no tick de um nascimento bem-sucedido (7.4).

### 7.2 Água

```latex
H_{t+1} = \min\big(H_{\max},\ H_t + G_{\text{água}} - C_H - C_{\text{repr},H}\big), \qquad C_H = h_0\, M^{2/3} \big(1 + \kappa\, I_{\text{sol}}(t)\, (1 - U_t)\big) + \eta\, C_{\text{mov}} + \chi\, R_H
```

- **M^(2/3):** a perda é proporcional à área da superfície. Como a reserva cresce com M, o tempo até desidratar cresce com M^(1/3): ser grande custa mais comida, mas ajuda contra a sede.
- **Sol e ar seco** aceleram a perda.
- **Esforço físico** gasta água (η · C\_mov).
- **Conversão em energia** gasta água (χ · R\_H).

Com os valores padrão, a água esvazia cerca de 1,6 vez mais rápido que a energia, em fração da reserva (12.2).

### 7.3 Comer e beber

As interações são resolvidas de forma simultânea, sem sorteio e sem depender da ordem das criaturas.

**Comer:**

1. Cada criatura com O\_eat > 0,5 reivindica a planta mais próxima com centro a até r\_t. Empate de distância vai para a planta de menor ID.
2. A demanda individual, em biomassa, é q\_i = min(q\_eat · M^(3/4), (E\_max − E\_i) / E\_stack): ninguém come além do que cabe.
3. Cada planta j entrega proporcionalmente:

```latex
\Delta b_i = q_i \cdot \min\left(1,\ \frac{b_j}{\sum_{k \to j} q_k}\right), \qquad G_{\text{comida},i} = E_{\text{stack}}\, \Delta b_i
```

**Beber:**

1. Cada criatura com O\_drink > 0,5 reivindica a fonte cuja margem está mais próxima, até r\_t (dentro da água, distância 0). Empate vai para o rio, depois para o menor ID.
2. A demanda é q\_i = min(q\_drink · M^(3/4), H\_max − H\_i).
3. Rios atendem qualquer demanda. Lagos e poças dividem o volume disponível pela mesma regra proporcional.

### 7.4 Reprodução

**Requisitos**, todos no mesmo tick. Se algum falhar, nada acontece e nada é cobrado.

| Requisito | Valor |
| --- | --- |
| Saída `reproduce` | > 0,7 |
| Idade | ≥ T\_mat |
| Energia | ≥ 0,8 · E\_max |
| Água | ≥ 0,6 · H\_max |
| Tempo desde a última reprodução | ≥ `T_rec` |
| População | Abaixo de `max_population` |

**Modos:**

- **Treino:** só reprodução assexuada clonal, sem mutação no mundo (8.6).
- **Contínuo:** `reproduction_mode = "mixed"`. Há reprodução sexuada quando existe parceiro; sem parceiro, a criatura reproduz assexuadamente. Também aceita `"asexual"` e `"sexual"`.

**Formação de casais (modo contínuo):**

1. Candidatos são as criaturas que cumprem todos os requisitos no tick.
2. Cada candidato escolhe o candidato mais próximo com centro a até r\_t + r\_c,j e sim(c, c\_j) ≥ `s_min`. Empate vai para o menor `uid`.
3. Escolhas mútuas formam casais. Só então a compatibilidade genômica é verificada (d\_g < delta\_tax, 8.5 e 8.7), uma vez por casal.
4. Casal incompatível, ou candidato sem escolha mútua, reproduz assexuadamente.

**Nascimento:**

- O filho nasce com E\_0 = 0,4 · E\_max e H\_0 = 0,4 · H\_max, calculados com o corpo do próprio filho.
- O custo é E\_0 / ε\_repr em energia e H\_0 em água. Na assexuada, o genitor paga tudo; na sexuada, cada genitor paga metade. A perda (1 − ε\_repr) entra na contabilidade.
- Posição: na assexuada, atrás do genitor, à distância r\_c; na sexuada, no ponto médio do casal. Orientação: a do genitor (na sexuada, a do genitor de menor `uid`). Velocidade, idade e memória começam em 0.
- Genoma: cópia com mutação (assexuada) ou crossover com mutação (sexuada), pelo fluxo criaturas (8.3, 8.4).
- Se os nascimentos do tick passarem do teto da população, têm prioridade os de menor `uid` do genitor; os demais falham sem custo.

### 7.5 Morte

| Causa | Condição | Código |
| --- | --- | --- |
| Fome | E ≤ 0 | `starvation` |
| Desidratação | H ≤ 0 | `dehydration` |
| Velhice | idade ≥ T\_vida | `old_age` |

Causas simultâneas são todas registradas. A energia e a água restantes saem do sistema; não há carcaças na v1. O slot é liberado.

### 7.6 Contabilidade de energia e água

| Balanço | Entradas | Estoques | Saídas |
| --- | --- | --- | --- |
| Energia | Solar (biomassa criada × E\_stack); hídrica (Σ R\_H) | Plantas (Σ b · E\_stack); criaturas (Σ E) | Metabolismo (basal, movimento, ações); perda da reprodução; energia dos mortos; murcha; excedente acima de E\_max |
| Água | Regeneração dos lagos; enchimento das poças; água bebida dos rios | Lagos; poças; criaturas (Σ H) | Evaporação das poças; perdas das criaturas (C\_H); água dos mortos; excedente acima de H\_max |

A cada tick, a variação dos estoques deve igualar entradas menos saídas, com erro relativo abaixo de 10⁻⁹. No modo de depuração a simulação para no primeiro desvio; nos demais, o desvio é registrado como alerta.

### 7.7 Parâmetros

| Parâmetro | Valor | Descrição |
| --- | --- | --- |
| `b_0` | 0,06 | Custo basal (Kleiber) |
| `b_v` | 0,02 | Manutenção muscular por v\_max² |
| `k_sen`, `f_sen` | 0,5; 0,8 | Aumento do basal na senescência e início dela |
| `c_m` | 0,12 | Custo de deslocamento |
| `c_g` | 0,02 | Custo de giro |
| `C_a` | 0,01 | Custo por tentativa de comer ou beber |
| `rho_H` | 0,3 | Recuperação máxima pela água, como fração de b\_0 · M^(3/4) |
| `chi` | 1,0 | Água consumida por unidade de energia recuperada |
| `h_0` | 0,06 | Perda basal de água |
| `kappa` | 1,0 | Aumento da perda sob sol e ar seco |
| `eta` | 0,5 | Água gasta por unidade de energia de movimento |
| `q_eat` | 0,2 | Biomassa ingerida por tick · M^(3/4) |
| `q_drink` | 4,0 | Água bebida por tick · M^(3/4) |
| `T_rec` | 150 | Intervalo mínimo entre reproduções (ticks) |
| `eps_repr` | 0,8 | Eficiência da transferência ao filho |
| `reproduction_mode` | `mixed` | Modo contínuo: `mixed`, `asexual` ou `sexual` |

## 8. Genética e NEAT próprio

O GENOME usa uma implementação própria do NEAT, com genes identificados por ID calculado a partir da estrutura, o que dispensa o registro de inovações e permite trocar genomas entre execuções. Os operadores de genoma valem nos dois modos; o maquinário populacional (especiação com seleção, compartilhamento de fitness, cotas) só roda entre gerações, no modo treino.

### 8.1 Genoma

- **Corpo:** `mass`, `v_max`, `color` (5.1).
- **Nós:** `id`, `type` (`input`, `output`, `hidden`), `bias` e `activation` (ocultos e saídas). Os fixos são as entradas 0–41 e as saídas 42–48 do `sensor_model = "genome-v1"`.
- **Conexões:** identificadas pelo par (`from`, `to`), com `weight` e `enabled`.

**Gênese da topologia:**

| Modo | Conexões iniciais | Ocultos | Evolução estrutural |
| --- | --- | --- | --- |
| `dense_starter` | As 294 conexões entrada → saída (42 × 7) | 0 | Divisão de conexões; novas conexões só envolvendo ocultos |
| `sparse_starter` | 15 a 20 pares entrada → saída sorteados | 0 | Novas conexões e divisão de conexões |

Pesos iniciais \~ N(0, 1). Corpo inicial dos fundadores da geração 0: M = 1, v\_max = 1 e cor sorteada, para que as linhagens sejam distinguíveis desde o início.

### 8.2 IDs de gene por hash

- **Nó oculto:** todo oculto nasce da divisão de uma conexão (A, B), e seu ID é calculado dela: os 8 bytes de `blake2b` sobre o texto `"A:B"`, lidos como inteiro big-endian u (digest\_size = 8), dão ID = 2⁵² + (u mod 2⁵²). O intervalo \[2⁵², 2⁵³) nunca colide com os nós fixos e é exato em JSON e JavaScript. O `hash()` nativo do Python é proibido, porque muda entre execuções. Identificador: `hash_function = "blake2b64-mod52-v1"`.
- **Conexão:** não tem número próprio; o par (`from`, `to`) é a sua identidade.
- **Homologia:** a mesma divisão, em qualquer genoma, execução ou computador, gera o mesmo ID, e o crossover alinha esses genes. Consequência aceita: linhagens que dividem a mesma conexão de forma independente produzem nós homólogos, mesmo com pesos muito diferentes.
- **Colisões:** só ficam prováveis perto de 67 milhões de ocultos distintos (paradoxo do aniversário em 2⁵²). Desprezível para o projeto.
- **Exibição:** no visualizador e nos relatórios, os ocultos recebem apelidos curtos por genoma (`H1`, `H2`, …). Os IDs calculados ficam nos arquivos e na lógica.

### 8.3 Operadores de mutação

Aplicados nesta ordem a cada filho que sofre mutação.

| Operador | Probabilidade | Regra |
| --- | --- | --- |
| Pesos | 0,8 por genoma | Cada conexão: 90% perturba com N(0; 0,5), 10% sorteia N(0, 1). Limite \[−8, 8\] |
| Bias | 0,8 por genoma | O mesmo, nos ocultos e nas saídas |
| Nova conexão | 0,05 | Origem em entradas ou ocultos, destino em ocultos ou saídas, par inexistente e sem criar ciclo. Peso N(0, 1). Até 20 tentativas |
| Novo nó | 0,03 | Divide uma conexão ativa (A, B) ainda não dividida no genoma: desativa (A, B), cria o oculto H(A, B) com bias 0 e tanh, e as conexões (A, oculto) com peso 1 e (oculto, B) com o peso antigo |
| Alternar conexão | 0,01 | Ativa ou desativa uma conexão sorteada |
| Massa | sempre | M ← M · e^N(0; 0,05), limitada à faixa |
| Velocidade | sempre | v\_max ← v\_max + N(0; 0,05), limitada à faixa |
| Cor | sempre | c ← (c + N(0; 0,02)) mod 1 |

**Ciclos:** uma conexão (A, B) é proibida se A for alcançável a partir de B, considerando todas as conexões, ativas ou não. Assim, nenhuma reativação futura cria ciclo.

### 8.4 Crossover

1. **Genes correspondentes** (mesmo ID de nó ou mesmo par): peso e bias vêm de um dos pais, com 50% de chance cada.
2. **Genes sem correspondência:** vêm do pai mais apto (treino: maior fitness; contínuo: maior energia atual). Em empate, de ambos.
3. **Genes desativados:** se desativado em algum dos pais, o gene nasce desativado com 75% de chance.
4. **Nós:** os fixos, mais os ocultos referenciados pelas conexões herdadas.
5. **Reparo de ciclos:** as conexões herdadas entram em ordem de (`from`, `to`); a que criaria um ciclo é descartada.
6. **Corpo:** massa e velocidade são a média dos pais; a cor é a média circular. Depois, a mutação normal.

### 8.5 Distância genômica e especiação

Como os IDs não seguem ordem cronológica, genes excessivos e disjuntos são contados juntos:

```latex
d_g = c_1 \frac{N_{\text{dif}}}{N} + c_3\, \bar W + c_b \left( \frac{|\Delta M|}{1{,}5} + \frac{|\Delta v_{\max}|}{1{,}0} \right)
```

- N\_dif = ocultos e conexões presentes em só um dos genomas; N = ocultos e conexões do maior genoma.
- W̄ = diferença média absoluta dos pesos das conexões correspondentes e dos bias dos nós correspondentes.
- Os denominadores do termo de corpo são as larguras das faixas dos genes. A cor fica de fora por ser só marcador.
- **Atribuição:** cada genoma entra na primeira espécie (em ordem de ID) cujo representante esteja a d\_g < δ; sem nenhuma, funda uma espécie nova. O representante de cada espécie é um membro da geração anterior sorteado do fluxo `evolucao`.
- **Limiar dinâmico:** depois de cada geração, δ ← 1,05 · δ se houver mais de `S_alvo` espécies e δ ← 0,95 · δ se houver menos. Isso controla o número de espécies e compensa a escala diferente de d\_g entre `dense_starter` e `sparse_starter`.

### 8.6 Regime do modo treino

O treino combina duas escalas: reprodução clonal dentro do mundo, a cada tick, e NEAT geracional entre gerações.

**Ciclo da geração g:**

1. O NEAT entrega N fundadores.
2. São gerados 3 mundos, com seeds (`treino`, g, k), e todos os N fundadores entram em cada um.
3. Cada mundo roda até t\_max ou até a extinção, em processos paralelos.
4. O fitness de cada fundador é a média dos 3 mundos (9.2).
5. A cada `G_val` gerações, a população é avaliada nas seeds de validação (10.3).
6. O NEAT executa especiação, compartilhamento, alocação, seleção, crossover e mutação, gerando os N fundadores de g + 1.

**Reprodução dentro do mundo:** os filhos são clones exatos (`in_world_mutation_rate = 0`, configurável), herdam o `founder_id` e são creditados à linhagem do fundador. Existem só durante a geração e não entram na população do NEAT. Assim, o sucesso da linhagem mede só a qualidade do genoma fundador.

**Seleção e proteção de estruturas jovens:**

- **Compartilhamento:** F\_adj = F / tamanho da espécie.
- **Alocação:** descendentes por espécie proporcionais à soma de F\_adj, arredondados pelo método dos maiores restos (empates pelo ID da espécie).
- **Cota mínima:** `m_min` descendentes para espécies com menos de `A_prot` gerações. Se a soma das cotas passar de N, as cotas são reduzidas proporcionalmente.
- **Estagnação:** espécie sem melhora do seu melhor fitness por `G_estag` gerações é eliminada; as 2 melhores espécies nunca são.
- **Genitores:** os 50% melhores de cada espécie.
- **Elitismo:** o melhor genoma de cada espécie com pelo menos 5 membros passa sem alteração.
- **Reprodução entre gerações:** 75% dos descendentes por crossover seguido de mutação, 25% só por mutação; cruzamento entre espécies com chance 0,001.

### 8.7 Regime do modo contínuo

- Não há gerações nem fitness explícito: sobrevive e reproduz quem consegue.
- Todo nascimento aplica mutação (assexuada) ou crossover com mutação (sexuada), pelo fluxo criaturas.
- **Taxonomia:** a cada `T_tax` ticks, um censo agrupa a população com o limiar fixo `delta_tax`. Uma espécie mantém o ID enquanto tiver membros próximos ao seu representante, que é atualizado a cada censo. A especiação aqui é só classificação para telemetria.
- O maquinário populacional do NEAT não roda neste modo. rtNEAT fica como extensão futura (13.2).

### 8.8 Parâmetros

| Parâmetro | Valor | Descrição |
| --- | --- | --- |
| `N` | 200 | Fundadores por geração (treino) |
| `worlds_per_generation` | 3 | Mundos por geração |
| `c_1`, `c_3`, `c_b` | 1,0; 0,4; 0,5 | Coeficientes da distância genômica |
| `delta_0` | 3,0 | Limiar inicial de compatibilidade |
| `S_alvo` | 15 | Número-alvo de espécies |
| `m_min`, `A_prot` | 2; 10 | Cota mínima e idade de proteção (gerações) |
| `G_estag` | 20 | Gerações sem melhora até a eliminação |
| `survival_threshold` | 0,5 | Fração elegível como genitor |
| `elitism_min_size` | 5 | Tamanho mínimo da espécie para o elitismo |
| `p_crossover`, `p_interspecies` | 0,75; 0,001 | Reprodução entre gerações |
| `in_world_mutation_rate` | 0 | Mutação dos clones no treino |
| `T_tax`, `delta_tax` | 1000; 3,0 | Censo taxonômico do modo contínuo |

## 9. Fitness e currículo

Em todas as fases, o fitness mede a mesma coisa: viver muito, viver bem e deixar descendentes, com o bônus de reprodução limitado por um teto. O currículo endurece o mundo de fase em fase em vez de mudar a recompensa.

### 9.1 Princípios

1. **Objetivo único:** nenhuma fase recompensa um comportamento isolado (correr, girar, comer).
2. **O currículo muda o mundo, não a recompensa:** o que é útil numa fase continua útil na seguinte.
3. **Fitness derivado do estado interno:** só tempo de vida, reservas e descendentes. Novas mecânicas não exigem mudar a fórmula.
4. **Equilíbrio entre necessidades:** o bem-estar usa o mínimo entre energia e água; excesso de uma não compensa falta da outra.
5. **Retornos limitados:** o bônus de reprodução satura.
6. **Ações bloqueadas são neutras:** uma ação ainda não liberada não tem efeito nem custo, então não há pressão para suprimi-la antes da hora.
7. **Fitness não negativo:** obrigatório para o compartilhamento de fitness do NEAT, e vale para qualquer termo futuro.
8. **Coerência com o modo contínuo:** o fitness premia o mesmo que a seleção natural do ecossistema contínuo.

### 9.2 Função de fitness

```latex
F = F_{\text{base}} + B_{\text{repr}}, \qquad F_{\text{base}} = w_S\, S + w_B\, B, \qquad w_S + w_B = 1
```

**Tempo de referência:** T\_ref = t\_max. Como T\_vida ≥ 2523 > t\_max = 2000, ninguém morre de velhice no treino. Normalizar pela duração da geração também impede que a evolução favoreça vidas curtas.

```latex
S = \frac{t_{\text{vivo}}}{T_{\text{ref}}}, \qquad B = \frac{1}{T_{\text{ref}}} \sum_{t=1}^{t_{\text{vivo}}} \min\left(\frac{E_t}{E_{\max}},\ \frac{H_t}{H_{\max}}\right)
```

Por construção, B ≤ S: morrer cedo nunca compensa. S premia durar; B premia durar bem. Como a água esvazia mais rápido, B tende a medir sobretudo a gestão da água, o que é coerente com o projeto.

| Criatura (t\_max = 2000, w\_S = w\_B = 0,5) | S | B | F\_base |
| --- | --- | --- | --- |
| Viveu 2000 ticks, pior reserva \~70% | 1,00 | 0,70 | 0,85 |
| Viveu 2000 ticks, pior reserva \~20% | 1,00 | 0,20 | 0,60 |
| Pior reserva \~60%, morreu no tick 500 | 0,25 | 0,15 | 0,20 |
| Nasceu bem e morreu no tick 50 | 0,03 | 0,02 | 0,02 |

Quem só sobrevive, mesmo com dificuldade, mantém fitness significativo, o que preserva espaço para genomas ainda incompetentes.

**Bônus de reprodução:** curva saturante com teto k\_repr. Filhos e netos contam pelo que nasceu no mundo, nas linhagens clonais do fundador.

```latex
B_{\text{repr}} = k_{\text{repr}} \left( 1 - e^{-D / D_{\text{sat}}} \right), \qquad D = n_{\text{filhos}} + \lambda\, n_{\text{netos}}
```

| D | B\_repr (k\_repr = 0,5; D\_sat = 5) |
| --- | --- |
| 1 | 0,09 |
| 5 | 0,32 |
| 10 | 0,43 |
| Qualquer | < 0,50 |

O bônus nunca alcança o valor de sobreviver bem. Como reproduzir transfere energia e água e reduz B, reproduzir só compensa quando as reservas permitem. Nas fases sem reprodução, k\_repr = 0.

**Fitness do fundador:** média de F nos 3 mundos da geração.

**Acumuladores por criatura**, atualizados de forma vetorizada:

| Acumulador | Atualização |
| --- | --- |
| `ticks_alive` | +1 por tick vivo |
| `wellbeing_sum` | + min(E/E\_max, H/H\_max) por tick vivo |
| `founder_id` | Herdado no nascimento; nunca muda |
| `n_children` | +1 por filho nascido |
| `n_grandchildren` | +1 por neto nascido |

No fim da geração, filhos e netos são somados por `founder_id`. No `GenomeParallelEnv`, a recompensa por tick é o incremento correspondente (2.2).

**Comparabilidade:** o fitness só é comparável dentro da mesma fase. Comparações entre fases, condições ou experimentos usam a avaliação nas seeds de teste, nas condições do ecossistema final.

### 9.3 Fases do currículo

| Fase | Mundo | Ações liberadas | k\_repr |
| --- | --- | --- | --- |
| 1. Sobrevivência em abundância | Plantas iniciais ×2 em manchas espaçadas; `p_sol` ×2; sem chuva; lagos cheios; `b_0`, `c_m`, `c_g` e `h_0` ×0,7 | Movimento, comer, beber, memória | 0 |
| 2. Escassez | Valores padrão: chuva, lagos finitos, poças | As mesmas | 0 |
| 3. Reprodução | Igual à Fase 2 | Todas, inclusive reproduzir | 0,5 |
| 4. Reservada | Será definida com a predação (13.2) | — | — |

A abundância da Fase 1 vem de manchas densas e espaçadas, não de crescimento rápido em todo lugar, para que ficar parado nunca seja suficiente.

**Invariante de calibração:** em todas as fases, pelo menos 95% das criaturas do controle imóvel morrem antes de t\_max, nascendo pela mesma regra dos fundadores (5.6). É verificada automaticamente antes de cada bateria (10.2).

### 9.4 Avanço e fim do treino

A população avança de fase quando ocorre o primeiro destes eventos:

1. **Domínio:** a mediana de F\_base da população nas seeds de **validação**, em média móvel de `G_janela` gerações, atinge `F_avanco`.
2. **Limite:** a fase completa `G_max_fase` gerações.

O motivo do avanço é registrado. A Fase 3 termina em `G_max_fase` gerações ou quando a mediana de validação passa `G_plateau` gerações sem melhorar.

### 9.5 Transição entre fases

- População, espécies e histórico são mantidos integralmente.
- Durante `G_trans` gerações, cada parâmetro do mundo (e k\_repr) é interpolado linearmente entre o valor da fase antiga e o da nova, evitando extinções por choque.
- Ações novas são liberadas no início da transição.

### 9.6 Parâmetros

| Parâmetro | Valor | Descrição |
| --- | --- | --- |
| `t_max` | 2000 | Duração da geração (ticks) |
| `w_S`, `w_B` | 0,5; 0,5 | Pesos da sobrevivência e do bem-estar |
| `k_repr` | 0,5 | Teto do bônus de reprodução (Fase 3) |
| `D_sat` | 5 | Escala de saturação do bônus |
| `lambda` | 0,5 | Peso dos netos |
| `F_avanco` | 0,6 | Mediana de F\_base para avançar |
| `G_janela` | 10 | Janela da média móvel |
| `G_max_fase` | 200 | Limite de gerações por fase |
| `G_trans` | 10 | Gerações de transição |
| `G_plateau` | 50 | Gerações sem melhora para encerrar a Fase 3 |

## 10. Rigor experimental

Só execuções confirmatórias, com protocolo congelado antes de rodar, sustentam conclusões. Cada condição usa 20 seeds raiz pareadas, é comparada a seis controles e é avaliada em seeds de teste que nunca participam do treino nem das decisões durante ele.

### 10.1 Protocolo versionado

- Antes de cada bateria, o protocolo é registrado em `protocol.yaml`: hipóteses, condições, seeds, métricas, testes estatísticos e critérios de sucesso.
- O protocolo pode mudar após testes. Cada mudança incrementa a versão e registra justificativa e data.
- Todo resultado é publicado com a versão do protocolo que o gerou.
- **Exploratórias:** calibram parâmetros e testam ideias; não sustentam conclusões.
- **Confirmatórias:** rodadas com o protocolo congelado; só elas sustentam conclusões.

### 10.2 Controles

| Controle | Descrição | O que verifica |
| --- | --- | --- |
| Aleatório | `dense_starter` com pesos sorteados, sem evolução | Linha de base mínima |
| Heurístico | Regras fixas: ir à comida mais forte na visão; beber quando H < 0,5 e houver água visível; comer ao tocar planta quando E < 0,9; senão, explorar andando e girando | Se a evolução supera uma estratégia razoável |
| Imóvel | Nunca se move; tenta comer e beber em todo tick | Sanidade do mundo (invariante, 9.3) |
| Nulo (treino) | NEAT completo, mas os genitores são sorteados ao acaso, ignorando o fitness; mutação e deriva mantidas | Adaptação real contra deriva neutra |
| Nulo de reprodução (contínuo) | A cada nascimento, o genoma do filho vem de uma criatura viva sorteada; o genitor real paga o custo | Remove a seleção sobre a reprodução. É um nulo parcial: a seleção por sobrevivência continua |
| Sem currículo | Evolução iniciada direto na Fase 3 | Se o currículo realmente ajuda |

**Testes de sanidade**, automáticos, antes de cada bateria confirmatória. Se algum falhar, o mundo ou o fitness está mal calibrado e a bateria não roda.

- Pelo menos 95% do controle imóvel morre antes de t\_max, em todas as fases.
- O heurístico supera o aleatório nas seeds de teste (Wilcoxon, p < 0,05).
- Os testes de determinismo passam e os balanços de energia e água fecham.

### 10.3 Seeds, conjuntos e avaliação

| Conjunto | Origem | Uso |
| --- | --- | --- |
| Treino | `seed_raiz` da execução, um mundo novo por geração e índice | Seleção do NEAT |
| Validação | 3 mundos fixos da `seed_laboratorio` | Avanço de fase, fim do treino, escolha do campeão |
| Teste | 10 mundos fixos da `seed_laboratorio` | Só a avaliação final |

- **Execuções:** `n_seeds` = 20 seeds raiz por condição (mínimo 10). Todas as condições usam o mesmo conjunto, o que forma pares diretamente comparáveis.
- **Validação:** a cada `G_val` gerações, toda a população é avaliada nos 3 mundos de validação.
- **Campeão:** o genoma com maior F\_base médio na última validação.
- **Métrica primária:** mediana de F da população final, como fundadores nos 10 mundos de teste, nas regras da Fase 3. Secundárias: F do campeão nos mundos de teste e sobrevivência da população final transferida para o modo contínuo.

**Análise estatística:**

| Comparação | Teste | Tamanho de efeito |
| --- | --- | --- |
| Pareada (mesmas seeds raiz) | Wilcoxon de postos sinalizados | Correlação rank-biserial pareada |
| Não pareada | U de Mann-Whitney | Delta de Cliff |

- Correção para comparações múltiplas pelo método de Holm, com α = 0,05.
- Diferença de medianas com intervalo de confiança de 95% por *bootstrap* (10 000 reamostragens, sorteadas da `seed_laboratorio`).

### 10.4 Métricas

**Modo treino, por geração:**

| Métrica | Descrição |
| --- | --- |
| Fitness | Mediana e máximo de F, F\_base, S, B e B\_repr |
| Generalização | Diferença entre o fitness no treino e na validação |
| Reprodução | Filhos e netos por fundador |
| Eficiência de forrageio | Energia ingerida dividida pela energia gasta em movimento |
| Causas de morte | Proporções de fome, desidratação e velhice |
| Complexidade | Média de ocultos e de conexões ativas |
| Diversidade | Número de espécies, δ atual e distância genômica média entre amostras |

**Modo contínuo, por janela de ticks:**

| Métrica | Descrição |
| --- | --- |
| Dinâmica populacional | Tamanho da população; nascimentos e mortes por 1000 ticks |
| Reprodução | Fração de nascimentos sexuados e assexuados |
| Longevidade | Tempo de vida médio e causas de morte |
| Diversidade | Espécies e linhagens ativas; distribuição dos genes corporais |
| Atividade evolutiva | Estatísticas de Bedau-Packard: genes que persistem na população por mais tempo do que persistiriam por acaso, comparados a um modelo-sombra de genes neutros |

### 10.5 Detecção de trapaças

Estratégias inesperadas não são necessariamente ruins.

- **Aceitável:** funciona no ecossistema final (modo contínuo, seeds de teste). É uma descoberta legítima.
- **Inaceitável:** depende de um artefato do treino (uma fase, um parâmetro do currículo) ou de um erro de simulação. A regra ou o código é corrigido.

**Mecanismos:**

- **Painel comportamental por geração:** distância média percorrida, fração do tempo parado, taxa de tentativas de comer e beber e taxa de sucesso delas.
- **Alertas automáticos:** quando o fitness sobe e uma métrica comportamental cai de forma suspeita (por exemplo, fitness subindo com a distância percorrida tendendo a zero).
- **Teste de transferência:** campeões de cada fase são avaliados periodicamente nas condições do ecossistema final.
- **Contabilidade:** qualquer desvio nos balanços de energia ou água indica um erro explorável (7.6).

### 10.6 Parâmetros

| Parâmetro | Valor | Descrição |
| --- | --- | --- |
| `n_seeds` | 20 | Seeds raiz por condição |
| `n_val`, `n_test` | 3; 10 | Mundos de validação e de teste |
| `G_val` | 5 | Intervalo entre validações (gerações) |
| `alpha` | 0,05 | Nível de significância |
| `n_bootstrap` | 10 000 | Reamostragens do bootstrap |

## 11. Formatos de dados, telemetria e checkpoints

Genomas são JSON portáteis entre execuções; telemetria é Parquet colunar gravado em blocos; checkpoints guardam o estado completo, inclusive os geradores aleatórios. Como a simulação é determinística, registra-se pouco e qualquer instante é reconstruído por re-simulação.

### 11.1 Arquivos

| Dado | Formato | Arquivos |
| --- | --- | --- |
| Configuração, manifesto, protocolo | YAML | `config.yaml`, `manifest.yaml`, `protocol.yaml` |
| Genomas | JSON UTF-8, gzip opcional | `*.genome`, `*.species` (ou `.genome.gz`, `.species.gz`) |
| Telemetria | Parquet (`pyarrow`), em blocos de 10 000 linhas | `ticks.parquet`, `events.parquet`, `generations.parquet`, `species.parquet` |
| Checkpoints | `.npz` mais JSON | `checkpoint_<tick>.npz`, `genomes.json`, `rng_state.json` |

### 11.2 Arquivo `.genome`

Um único indivíduo: corpo e cérebro.

| Grupo | Campos |
| --- | --- |
| Compatibilidade | `format_version`, `sensor_model`, `num_inputs`, `num_outputs`, `hash_function` |
| Metadados | `name`, `author`, `created_at`, `source_mode` (`training` ou `continuous`), `topology_genesis`, `curriculum_stage`, `generation_reached` (treino) ou `birth_tick` (contínuo), `fitness`, `lineage` (uids dos pais) |
| Origem | `config_hash`, `config_summary` (dimensões, clima, metabolismo), `root_seed`, `code_version` |
| Corpo | `mass`, `v_max`, `color` |
| Cérebro | `nodes` (`id`, `type`, `label` nos fixos, `activation` e `bias` nos ocultos e saídas); `connections` (`from`, `to`, `weight`, `enabled`) |

```json
{
  "format_version": 1,
  "sensor_model": "genome-v1",
  "num_inputs": 42,
  "num_outputs": 7,
  "hash_function": "blake2b64-mod52-v1",
  "name": "Forrageador_Ribeirinho_g142",
  "source_mode": "training",
  "topology_genesis": "sparse_starter",
  "curriculum_stage": 3,
  "generation_reached": 142,
  "fitness": 1.08,
  "body": {"mass": 1.12, "v_max": 0.87, "color": 0.314},
  "brain": {
    "nodes": [
      {"id": 0, "type": "input", "label": "vision_food_r1"},
      {"id": 45, "type": "output", "label": "eat", "activation": "sigmoid", "bias": 0.12},
      {"id": 7039250718443391, "type": "hidden", "activation": "tanh", "bias": -0.40}
    ],
    "connections": [
      {"from": 0, "to": 7039250718443391, "weight": 1.42, "enabled": true},
      {"from": 7039250718443391, "to": 45, "weight": -0.85, "enabled": true}
    ]
  }
}
```

O exemplo omite campos de metadados e de origem por brevidade; no arquivo real todos são obrigatórios.

### 11.3 Arquivo `.species`

Um grupo de genomas de uma espécie (padrão: os 5 melhores), para povoar um ecossistema com variedade em vez de clones. Tem os mesmos campos de compatibilidade e origem do `.genome`, compartilhados pelo grupo, mais `species_name`, `species_size`, `representative` (índice do genoma representante) e `genomes` (lista com corpo, cérebro e metadados de cada um).

### 11.4 Regras de importação

1. **`format_version`:** versões antigas são convertidas quando há conversor; versões desconhecidas são recusadas.
2. **`sensor_model`:** precisa ser idêntico ao do destino, senão a importação é recusada, porque as entradas teriam outro significado.
3. **`hash_function`:** precisa ser idêntica, senão a importação é recusada.
4. **`config_hash`:** se diferente, a importação é permitida com aviso, e a divergência vai para o manifesto.
5. **Povoamento:** com menos genomas que vagas, os genomas se repetem em rodízio. `import_mutation_rate` (padrão 0) pode aplicar mutação às cópias.

### 11.5 Telemetria

| Tabela | Frequência | Colunas principais |
| --- | --- | --- |
| `ticks` | A cada `T_log` ticks | `tick`, `population`, `births`, `deaths`, `plants_total`, `plants_stack1`…`3`, `biomass_total`, `water_lakes`, `water_puddles`, `light`, `rain`, `humidity`, `mean_energy`, `mean_hydration`, `species_count`, `energy_ledger_error`, `water_ledger_error` |
| `events` | Todo nascimento e morte | `tick`, `event`, `uid`, `parent_a`, `parent_b`, `founder_id`, `reproduction` (`asexual`, `sexual`), `death_cause`, `age`, `mass`, `v_max`, `color`, `species_id` |
| `generations` | Por geração (treino) | `generation`, `phase`, `delta`, `species_count`, estatísticas de F, F\_base, S, B e B\_repr, `val_median`, painel comportamental, `advance_reason` |
| `species` | Por geração ou censo | `species_id`, `size`, `age`, `best_fitness`, `representative_uid`, médias dos genes corporais |

Os escritores acumulam linhas em memória e gravam a cada 10 000. A tabela `events` sozinha reconstrói a genealogia completa.

### 11.6 Checkpoints

- **Conteúdo:** arrays do núcleo (criaturas, plantas, água, clima, tick), genomas vivos, estado de cada fluxo aleatório (`bit_generator.state`), estado do NEAT (espécies, δ, contadores de estagnação, geração, fase) e totais dos balanços.
- **Frequência:** a cada `T_ckpt` ticks no contínuo e a cada `G_ckpt` gerações no treino.
- **Garantia:** retomar de um checkpoint produz exatamente o mesmo resultado que a execução direta (teste em 2.5).

### 11.7 Manifesto

Cada execução grava `manifest.yaml` com: configuração completa e `config_hash`; `seed_raiz` e `seed_laboratorio`; versão do protocolo; commit do repositório e indicação de alterações não commitadas; versões de Python, NumPy, SciPy, Numba e PettingZoo; sistema e CPU; horários de início e fim; desfecho (`completed`, `extinction`, `interrupted`); divergências de importação.

### 11.8 Parâmetros

| Parâmetro | Valor | Descrição |
| --- | --- | --- |
| `T_log` | 10 | Intervalo da tabela `ticks` |
| `T_ckpt` | 50 000 | Intervalo de checkpoint no contínuo (ticks) |
| `G_ckpt` | 10 | Intervalo de checkpoint no treino (gerações) |
| `flush_rows` | 10 000 | Linhas por bloco gravado |

## 12. Escala, calibração e desempenho

1000 criaturas cabem com folga num mundo 512 × 512: elas ocupam cerca de 1,2% da área, e o limite real é a produção de comida, definida pela quantidade de rios e lagos. O treino usa mundos menores com 200 fundadores, na mesma densidade.

### 12.1 Presets

| Parâmetro | Treino | Contínuo |
| --- | --- | --- |
| Mundo (W × H) | 256 × 256 | 512 × 512 |
| Fundadores / população inicial | 200 | 1000 |
| `max_population` | 600 | 1200 |
| Rios | 2 | 4 |
| Lagos | 3 | 10 |
| Posições de poças | 20 | 80 |
| Plantas iniciais | 600 | 2500 |
| `max_plants` (segurança) | 10 000 | 40 000 |
| Grades de olfato | 64 × 64 | 128 × 128 |
| `max_ticks` | 2000 (t\_max) | Sem limite |
| `terminate_on_extinction` | Sim | Sim (configurável) |

### 12.2 Calibração derivada (criatura de referência)

| Grandeza | Conta | Resultado |
| --- | --- | --- |
| Custo parado | C\_basal 0,08 − R\_H 0,018 | 0,062 E/tick |
| Fome parada, saindo cheia | 100 / 0,062 | \~1600 ticks |
| Custo de quem forrageia | basal 0,08 + movimento 0,062 + ações 0,004 − R\_H 0,014 | \~0,13 E/tick |
| Comida por geração | 0,13 × 2000 / 25 | \~10,5 stacks |
| Sede parada, sol pleno e ar seco | 60 / (0,06 × 1,8 + 0,018) | \~480 ticks |
| Sede parada à noite | 60 / (0,06 + 0,018) | \~770 ticks |
| Água de quem forrageia | clima médio 0,079 + movimento 0,031 + conversão 0,014 | \~0,124 H/tick (\~4 recargas por geração) |
| Esvaziamento relativo | água 0,21%/tick contra energia 0,13%/tick | água \~1,6× mais rápida |
| Prazo do fundador para achar água | 48 / 0,124 | \~390 ticks |
| Produção por planta irrigada | 0,55 × (0,75 × 0,004 + 0,25 × 0,0015) × 25 | \~0,046 E/tick |
| Velocidade ótima de cruzeiro | √(C\_basal / (c\_m · M)) | \~0,82 · v\_max |
| Distância de frenagem | v · τ | 3 unidades |

A velocidade ótima fica no meio da faixa do gene v\_max, não num extremo: a evolução tem um ótimo real para encontrar.

### 12.3 Capacidade de suporte

```latex
N_{\text{sustentável}} \approx \frac{N_{\text{pp}} \times 0{,}046}{0{,}13} \approx 0{,}35\, N_{\text{pp}}
```

N\_pp é o número de plantas irrigadas **não saturadas** (as saturadas e cercadas não produzem). A fórmula é um ponto de partida para dimensionar rios e lagos; a capacidade real é menor e depende de quanto as criaturas pastam.

### 12.4 1000 criaturas em 512 × 512

- **Espaço:** uma criatura a cada \~262 unidades², espaçamento médio de \~16 unidades. Os corpos ocupam \~1,2% do mapa.
- **Visão:** em média \~10 criaturas no campo de visão de dia e \~2,5 à noite, com 3 a 5 vezes mais perto da água.
- **Comida:** \~2850 plantas produtivas, ou cerca de 4000 a 6000 plantas no total. Com 4 rios atravessando o mapa, a faixa irrigável (\~15 unidades de cada margem) soma \~23% da área e comporta \~7900 posições cristalinas, sem contar os lagos.

### 12.5 Estratégias de desempenho

- Estado em arrays por atributo, com slots pré-alocados (2.1).
- `cKDTree` com `boxsize`: a das criaturas é refeita a cada tick, a das plantas só quando o conjunto muda, e as de rios e lagos uma única vez.
- Pares criatura–objeto gerados de uma vez (`sparse_distance_matrix(..., output_type="ndarray")`) e agregados em kernels Numba.
- Redes compiladas e avaliadas em lote no Numba, sem `fastmath` (2.5).
- Campos de olfato a cada `T_olf` ticks e irrigação a cada `T_irr`.
- **Paralelismo por processos:** um processo por mundo, seed ou condição. Os 3 mundos de uma geração rodam em paralelo. No i5-13400, até 14 processos. A RTX 2060 não é usada pelo núcleo, que é CPU.

### 12.6 Estimativas de custo

Estimativas grosseiras, a confirmar com profiling na fase F2 do roadmap.

| Cenário | Por tick | Total |
| --- | --- | --- |
| Mundo de treino (200 a 600 criaturas) | \~2 a 4 ms | Uma geração (3 mundos em paralelo × 2000 ticks): \~4 a 8 s |
| Mundo contínuo (\~1000 criaturas) | \~6 a 10 ms | 1 milhão de ticks: \~2 a 3 h |
| Bateria confirmatória: 3 condições evolutivas × 20 seeds × até 600 gerações × 3,6 mundos por geração | \~3 ms | \~260 milhões de ticks; até \~15 h com 14 processos |
| Bateria exploratória: 5 seeds, até 50 gerações por fase | \~3 ms | \~1 h com 14 processos |

## 13. Roadmap e extensões

O desenvolvimento segue seis fases, cada uma fechada por um critério verificável. O ecossistema mínimo, sustentado por um controlador heurístico, vem antes de qualquer evolução, porque o maior risco do projeto é o equilíbrio ecológico, não o código.

### 13.1 Fases de desenvolvimento

&#91;embedded content: roadmap · 6 fases e critérios de saída\]

Cada fase só começa quando o critério de saída da anterior foi verificado.

Os valores de calibração deste documento são pontos de partida: a fase F1 é onde eles são ajustados, e cada ajuste entra no protocolo versionado (10.1).

### 13.2 Extensões futuras

| Extensão | Depende de |
| --- | --- |
| Predação e Fase 4 do currículo | Atuador de ataque, modelo de dano, conversão de carne em energia |
| Obstáculos perceptíveis | Novo canal sensorial e novo `sensor_model` |
| rtNEAT no modo contínuo | NEAT estável no modo treino |
| Mutação de funções de ativação (tanh, ReLU, gaussiana) | Operador novo e nova versão do formato |
| Carcaças como recurso | Predação ou decomposição |
| Crescimento corporal (ontogenia) | Genes de desenvolvimento |
| Comunicação entre criaturas | Novo canal sensorial e nova saída |
| Comparação com PPO multiagente | `GenomeParallelEnv` estável |
| Aceleração por GPU (JAX) | Núcleo vetorizado e perfilado |

### 13.3 Questão em aberto

- **Percepção de obstáculos:** quando forem ativados, as criaturas os percebem pela visão (6 entradas novas) ou por um toque de colisão (1 entrada nova)?
