# GENOME

**Genetic Evolution & Neural Organism Modeling Environment**: um laboratório de vida artificial e neuroevolução em Python.

Criaturas com cérebros NEAT evoluem num mundo 2D toroidal com rios, lagos, poças, clima e plantas. O núcleo de simulação é próprio, determinístico e vetorizado (NumPy, SciPy, Numba). Ele é exposto como `ParallelEnv` do PettingZoo.

## Estado atual

O projeto está na **fase F0 (Fundação)**: repositório, ferramentas, configuração, hierarquia de seeds e geometria toroidal. As fases seguintes estão no roadmap das [issues](https://github.com/ninguemdev/GENOME/issues) e em `docs/ARQUITETURA.md` §13.

## Instalação

Requer [uv](https://docs.astral.sh/uv/). O uv instala o Python 3.12 se ele não estiver disponível.

```bash
uv sync
```

## Testes e verificações

```bash
uv run pytest -m "not slow"   # suíte rápida
uv run pytest                 # suíte completa, com cobertura
uv run ruff format . && uv run ruff check .
uv run mypy
uv run lint-imports
```

## Documentação

- [`docs/ARQUITETURA.md`](docs/ARQUITETURA.md): especificação completa (fonte de verdade).
- [`AGENTS.md`](AGENTS.md): regras de trabalho para agentes e colaboradores.
- [`docs/PADROES.md`](docs/PADROES.md) e [`docs/FLUXO_GIT.md`](docs/FLUXO_GIT.md): padrões de engenharia e fluxo git.

## Licença

[MIT](LICENSE) © 2026 Dionel Morais
