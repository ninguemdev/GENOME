# Padrões de engenharia do GENOME

Este documento define como o código é organizado, escrito e testado. A especificação do que o código faz está em `docs/ARQUITETURA.md`; aqui está o como.

## 1. Estrutura do repositório

```text
GENOME/
├── AGENTS.md · CLAUDE.md · README.md · LICENSE · pyproject.toml · uv.lock
├── .claude/settings.json
├── .github/            templates de issue e PR, workflows de CI
├── configs/            presets YAML (training.yaml, continuous.yaml)
├── docs/
│   ├── ARQUITETURA.md  especificação (fonte de verdade)
│   ├── PADROES.md      este documento
│   ├── FLUXO_GIT.md    fluxo de trabalho no git
│   └── decisoes/       ADRs numerados (0001-titulo.md)
├── scripts/            utilitários de linha de comando (benchmark, execuções)
├── src/genome/
│   ├── core/           simulação pura (sem dependência de outro pacote do GENOME)
│   │   ├── config.py   dataclasses imutáveis, carga YAML, validação, config_hash
│   │   ├── rng.py      hierarquia de seeds e fluxos (§2.5)
│   │   ├── space.py    geometria toroidal (§3.1)
│   │   ├── protocols.py  Brain, GenomeOperators e demais contratos
│   │   ├── state.py    estado em arrays por atributo e slots
│   │   ├── world/      água, clima, obstáculos (§3)
│   │   ├── plants.py   (§4)
│   │   ├── sensors/    visão, olfato, toque (§5)
│   │   ├── physics.py  locomoção com inércia (§6.3)
│   │   ├── metabolism.py · interactions.py · lifecycle.py (§7)
│   │   ├── ledger.py   contabilidade de energia e água (§7.6)
│   │   └── tick.py     orquestra a ordem do tick (§2.4)
│   ├── neat/           NEAT próprio (a partir da F3)
│   ├── envs/           GenomeParallelEnv (F2)
│   ├── lab/            modos de execução, controladores fixos, controles, protocolo
│   ├── io/             .genome, .species, Parquet, checkpoints, manifesto
│   └── viz/            visualizador e inspetor (F2)
└── tests/              espelha src/genome/ (tests/core/test_space.py etc.)
```

Os módulos acima são o alvo. Crie cada um somente na fase que o usa; os pacotes existem desde a F0 com um `__init__.py` que descreve sua responsabilidade.

## 2. Camadas e contratos

| Pacote | Pode importar | Não pode importar |
| --- | --- | --- |
| `genome.core` | biblioteca padrão, NumPy, SciPy, Numba, PyYAML | qualquer outro pacote do GENOME |
| `genome.neat` | `genome.core` (protocolos e tipos) | `envs`, `lab`, `io`, `viz` |
| `genome.envs` | `genome.core` | `neat`, `lab`, `io`, `viz` |
| `genome.io` | `genome.core`, `genome.neat` | `envs`, `lab`, `viz` |
| `genome.viz` | `genome.core`, `genome.io` | `neat`, `envs`, `lab` |
| `genome.lab` | todos | — |

Os contratos são verificados pelo `import-linter` (`uv run lint-imports`) no pre-commit e no CI. Exemplo de contrato no `pyproject.toml`:

```toml
[tool.importlinter]
root_package = "genome"

[[tool.importlinter.contracts]]
name = "core é independente"
type = "forbidden"
source_modules = ["genome.core"]
forbidden_modules = ["genome.neat", "genome.envs", "genome.lab", "genome.io", "genome.viz"]
```

**Inversão de dependência:** o núcleo declara protocolos (`typing.Protocol`) em `core/protocols.py`. Implementações concretas (controladores fixos, cérebros NEAT, operadores de genoma) vivem fora do núcleo e são injetadas pelo `genome.lab`.

## 3. Design do núcleo

- **Estado em arrays por atributo:** cada atributo das criaturas é um array NumPy pré-alocado com capacidade `max_population`, mais uma máscara de vivos e uma lista de slots livres (§2.1). Plantas seguem o mesmo padrão.
- **Sistemas como funções:** cada passo do tick é uma função que recebe estado, configuração e o fluxo aleatório de que precisa, e atualiza o estado. `tick.py` só chama esses sistemas na ordem de §2.4.
- **Sem objetos por criatura:** nada de classe `Creature` com métodos chamados num laço. Uma criatura é um índice nos arrays.
- **Pureza nos kernels:** kernels Numba recebem arrays e devolvem arrays ou escrevem em arrays de saída; não leem estado global.
- **Imutabilidade da configuração:** dataclasses `frozen=True`, validadas na carga. Nenhum sistema altera a configuração.

## 4. Clean code

- Uma responsabilidade por função; alvo de até ~40 linhas por função e ~400 por módulo. Passar disso é sinal para dividir.
- Nomes que dizem o que a coisa é (`hydration_cost`, não `hc`). Funções com verbo, valores com substantivo.
- Nomes de parâmetros iguais aos das tabelas da ARQUITETURA, em `snake_case` minúsculo (`rho_L` vira `rho_l`, `T_vida` vira `t_vida`).
- Nenhum número mágico. Constantes físicas e parâmetros vêm da configuração; constantes puramente técnicas (tolerâncias, tamanhos de bloco) ficam nomeadas no topo do módulo.
- Sem estado global mutável, sem singletons, sem variáveis de módulo alteradas em tempo de execução.
- Erros explícitos com mensagens claras; nunca `except: pass`.
- Comentários explicam o porquê, não o quê. Docstrings no estilo Google, em inglês, citando a seção da ARQUITETURA que a função implementa.
- Tipagem completa; `Any` só com justificativa em comentário.
- Código morto e código comentado não entram no repositório.

## 5. Determinismo

- Geradores só de `genome.core.rng`, que deriva cada fluxo por `SeedSequence(entropia, spawn_key=(DOMINIO, g, k, FLUXO))` (§2.5).
- O ruff bloqueia o módulo `random` e a API global do NumPy (regras `NPY002` e `TID251` em `pyproject.toml`).
- `float64` em todo o núcleo; tolerâncias de teste com `numpy.testing.assert_allclose`.
- Numba: `@njit(cache=True, fastmath=False)`; nada de `prange` em somas.
- Nenhuma dependência de ordem de `dict` ou `set`; resolução de conflitos por regras de §7.3, desempates por ID ou `uid`.
- Após integrar posições, aplique a correção de §3.1 (`x == W` vira `0`).

## 6. Desempenho

- Caminhos quentes (sensores, física, metabolismo, avaliação de redes) são vetorizados em NumPy ou escritos como kernels Numba. Nenhum laço Python por criatura.
- Otimize só depois de medir: `scripts/bench.py` roda um mundo de referência e reporta ms por tick por sistema. Registre a linha de base no PR que muda desempenho.
- Consultas espaciais com `scipy.spatial.cKDTree(boxsize=(W, H))`; pares gerados de uma vez.

## 7. Testes

| Tipo | O que cobre | Onde |
| --- | --- | --- |
| Unitário | Cada fórmula de §3–§9, com valores calculados à mão (§12.2) | `tests/<pacote>/` |
| Propriedade | Geometria toroidal, conservação, independência de ordem (Hypothesis) | idem |
| Determinismo | Hash do estado após N ticks igual em duas execuções e após restaurar checkpoint | `tests/determinism/` |
| Contabilidade | Balanços de energia e água fecham com erro relativo < 1e-9 | `tests/core/test_ledger.py` |
| Integração | Ecossistema roda N ticks sem erro; equivalência PettingZoo × caminho rápido (F2) | `tests/integration/` |

- Cobertura mínima de 90% em `genome.core` (`pytest --cov=genome.core --cov-fail-under=90`).
- Testes acima de ~2 s levam `@pytest.mark.slow`.
- Nenhum teste depende de rede, relógio ou ordem de execução.
- Bug corrigido ganha um teste de regressão no mesmo commit.

## 8. Dependências por fase

Adicione cada dependência só na fase que a usa. Qualquer outra exige pergunta ao usuário e um ADR.

| Fase | Execução | Desenvolvimento |
| --- | --- | --- |
| F0 | `numpy`, `scipy`, `pyyaml` | `pytest`, `pytest-cov`, `hypothesis`, `ruff`, `mypy`, `import-linter`, `pre-commit`, `types-PyYAML` |
| F1 | `numba` | — |
| F2 | `pettingzoo`, `gymnasium`, `pyarrow`; extra `viz`: `pygame-ce` | — |
| F3–F4 | — | — |
| F5 | extra `report`: `matplotlib` | — |

A estatística da F5 usa `scipy.stats` (Wilcoxon, Mann-Whitney, bootstrap), já disponível.

## 9. Ferramentas

- **Ambiente:** `uv` com Python 3.12 e `uv.lock` versionado.
- **Ruff:** formatação e lint. Regras mínimas: `E`, `F`, `W`, `I`, `N`, `UP`, `B`, `SIM`, `NPY`, `TID`, `RUF`, `D` (estilo Google). `TID251` bane `random` em todo o pacote.
- **Mypy:** `strict = true`, com plugin do NumPy quando disponível.
- **pre-commit:** ruff (format e check), mypy, import-linter e testes rápidos.
- **CI (GitHub Actions):** em todo push e PR: `uv sync`, ruff, mypy, import-linter, pytest com cobertura. Linux, Python 3.12.

## 10. Decisões e mudanças na especificação

- **ADR:** toda decisão que completa ou ajusta a ARQUITETURA vira um arquivo `docs/decisoes/NNNN-titulo.md` com contexto, decisão e consequências (modelo em `docs/decisoes/0000-modelo.md`).
- **Calibração (F1):** mudanças de valores de parâmetros são permitidas, com ADR e atualização da tabela da ARQUITETURA no mesmo PR, citando os dados que justificaram a mudança.
- **Mecânicas, fórmulas, interfaces e formatos:** só mudam com aprovação explícita do usuário. Pergunte antes de implementar.
- A `docs/ARQUITETURA.md` do repositório é a fonte de verdade. Mudanças nela entram por PR, com commit `docs(arquitetura): ...`.
