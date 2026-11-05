# AGENTS.md — GENOME

Laboratório de vida artificial e neuroevolução em Python, exposto como `ParallelEnv` do PettingZoo.
Repositório: https://github.com/ninguemdev/GENOME · Licença MIT.

## Fontes de verdade

1. `docs/ARQUITETURA.md`: especificação completa (fórmulas, parâmetros, ordem do tick, formatos). Leia as seções relevantes antes de implementar. Cite a seção no docstring de cada função que implementa uma regra (ex.: `Implements ARQUITETURA §7.1`).
2. `docs/PADROES.md`: arquitetura de código, clean code, testes e determinismo. Leia antes de escrever código.
3. `docs/FLUXO_GIT.md`: issues, branches, commits e PRs. Leia antes de qualquer operação git.
4. A issue da fase atual (`gh issue view <n>`): escopo e checklist.

Se o código precisar divergir da especificação, pare e pergunte (ver "Quando parar").

## Ordem do projeto: laboratório antes da IA

| Fase | Conteúdo | Redes neurais |
| --- | --- | --- |
| F0 | Fundação: repositório, ferramentas, configuração, seeds, toroide | Não |
| F1 | MVP ecológico: mundo, clima, plantas, criaturas, metabolismo, controladores fixos | Não |
| F2 | Inspeção e PettingZoo: visualizador, `GenomeParallelEnv`, telemetria, checkpoints | Não |
| F3 | NEAT próprio e modo contínuo | Sim |
| F4 | Modo treino geracional e currículo | Sim |
| F5 | Experimentos: protocolo, baterias, estatística | Sim |

- Nunca implemente código de uma fase futura. Uma fase só começa depois que o PR da anterior foi mergeado.
- `src/genome/neat/` fica só com `__init__.py` até a F3.
- Em F0–F2, todo cérebro é um controlador fixo (heurístico, aleatório, imóvel) que implementa o protocolo `Brain` do núcleo.

## Comandos

```bash
uv sync                                   # ambiente
uv run ruff format . && uv run ruff check .
uv run mypy                               # tipagem estrita
uv run lint-imports                       # contratos de camadas
uv run pytest -m "not slow"               # rápido
uv run pytest                             # completo, antes de PR
```

Antes de cada commit: formatação, lint, mypy, contratos e testes rápidos. Antes do PR: a suíte completa.

## Regras invioláveis do núcleo

- Toda aleatoriedade vem de `genome.core.rng`, nos fluxos de §2.5. Proibidos: módulo `random`, funções globais `np.random.*`, seeds baseadas em tempo e `hash()` nativo para IDs.
- `float64` no núcleo. Numba com `fastmath=False` e sem `prange` em reduções.
- Nenhuma iteração sobre `dict` ou `set` no núcleo; arrays ordenados por `uid`.
- Interações simultâneas e independentes da ordem das criaturas (§7.3).
- `genome.core` não importa nenhum outro pacote do GENOME. Ele define protocolos; os outros pacotes implementam e o laboratório injeta.
- Todo fluxo novo de energia ou água entra na contabilidade (§7.6), com teste.
- Parâmetros vêm só da configuração (§2.3). Nenhum número mágico no código.
- Nenhum laço Python por criatura em caminho quente: NumPy vetorizado ou kernel Numba.

## Código

- Python 3.12, layout `src/`, tipagem completa (`mypy --strict`).
- Identificadores, docstrings e comentários em inglês. Documentos, issues, PRs e commits em português do Brasil.
- Funções pequenas com uma responsabilidade, módulos coesos, dependências injetadas, sem estado global mutável.
- Erros explícitos; a configuração é validada ao carregar.
- Dependências novas só conforme a tabela de `docs/PADROES.md`, com justificativa no PR.

## Testes

- Toda regra de §3 a §9 tem teste unitário com valores calculados à mão (use §12.2).
- Hypothesis para geometria toroidal e conservação.
- Teste de determinismo (hash do estado) cobre qualquer mudança no núcleo.
- Cobertura mínima de 90% em `genome.core`.
- Testes lentos levam `@pytest.mark.slow`.

## Git (detalhes em `docs/FLUXO_GIT.md`)

- Uma issue por fase, com checklist. Um branch e um PR por fase; fases grandes podem ter até 3 blocos sequenciais. Nunca mais de um PR aberto.
- Branch `fase/fN-nome`. Nada de commits na `main`, exceto o bootstrap inicial.
- Conventional Commits em português: `tipo(escopo): descrição no imperativo`. Commits atômicos, cada um com testes passando, rodapé `Refs #N`.
- **Sem coautoria.** Nenhum trailer `Co-Authored-By`, nenhuma linha "Generated with Claude Code" ou similar em commits, PRs, issues ou comentários. Use a identidade git já configurada e nunca altere `user.name` ou `user.email`.
- O agente abre o PR, valida e faz o merge (`docs/FLUXO_GIT.md` §6). Nunca force-push na `main`; `--force-with-lease` só no próprio branch, depois de rebase.
- Ao fim de cada sessão, atualize o checklist editando o corpo da issue. Não comente a cada passo.

## Definição de pronto (cada item do checklist)

- Implementa a seção citada da ARQUITETURA, com o docstring referenciando-a.
- Testes novos e a suíte rápida passam.
- `ruff`, `mypy` e `lint-imports` limpos.
- Nenhum `TODO` sem issue correspondente.
- Item marcado no checklist da issue.

## Quando parar e perguntar

- A especificação é ambígua, contraditória ou parece errada.
- A implementação exigiria mudar uma mecânica, fórmula, interface ou formato da ARQUITETURA.
- Seria preciso uma dependência fora da tabela de `docs/PADROES.md`.
- Algo exigiria force-push, reescrever histórico publicado ou mexer na `main`.

Ajustes de valores de calibração na F1 são permitidos sem perguntar: registre-os num ADR em `docs/decisoes/` e atualize a tabela de parâmetros da ARQUITETURA no mesmo PR.

## Idioma

Responda ao usuário em português do Brasil.
