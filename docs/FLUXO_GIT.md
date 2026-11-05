# Fluxo de trabalho no git

O GENOME usa uma `main` sempre estável e um branch por fase do roadmap. O objetivo é histórico limpo e rastreável, com poucas issues, poucos branches e poucos PRs.

## 1. Visão geral

```text
issue da fase (#N, checklist)
   └── branch fase/fN-nome  (a partir da main atualizada)
         ├── commits atômicos, Conventional Commits, Refs #N
         └── PR "[FN] Nome da fase" → Closes #N → validação e merge pelo agente → branch apagado
```

- **Remoto:** `origin` = `https://github.com/ninguemdev/GENOME.git`.
- **Um PR aberto por vez.** A fase seguinte só começa depois do merge.
- **Exceção:** o bootstrap inicial (documentos, licença, templates) vai direto na `main`, uma única vez.

## 2. Issues

- **Uma issue por fase** (F0 a F5), criada no bootstrap a partir do template `fase.md`, com o checklist completo da fase e o critério de saída.
- **Labels:** `fase:F0` … `fase:F5`, `tipo:bug`, `tipo:decisao`. Milestone única: `v1`.
- **Bugs** fora do escopo da fase atual viram uma issue `tipo:bug`. Se não bloqueiam, entram no checklist da próxima fase afetada; se bloqueiam, são corrigidos no branch atual com um commit `fix`.
- **Progresso:** o checklist é atualizado editando o corpo da issue (`gh issue edit N --body-file ...`), uma vez por sessão. Comentários só para fatos relevantes: um bloqueio, uma decisão tomada, um número de calibração.

## 3. Branches

- Nome: `fase/fN-nome-curto` (ex.: `fase/f1-mvp-ecologico`).
- Sempre criado a partir da `main` atualizada: `git switch main && git pull --ff-only && git switch -c fase/f1-mvp-ecologico`.
- **Fases grandes:** se o diff passar de ~2000 linhas, a fase pode ser dividida em até 3 blocos sequenciais (`fase/f1-mvp-ecologico-1`, `-2`, `-3`). Cada bloco tem um PR e só começa depois do merge do anterior. A issue da fase continua única.
- Antes de abrir o PR: `git fetch origin && git rebase origin/main`. Depois do rebase, `git push --force-with-lease` só no próprio branch.

## 4. Commits

**Formato:** Conventional Commits, descrição em português, no imperativo, sem ponto final, até ~72 caracteres.

```text
tipo(escopo): descrição curta no imperativo

Corpo opcional explicando o porquê da mudança, não o quê.
Quebre linhas em ~72 colunas.

Refs #N
```

| Tipo | Uso |
| --- | --- |
| `feat` | Comportamento novo |
| `fix` | Correção de bug |
| `refactor` | Mudança interna sem alterar comportamento |
| `perf` | Ganho de desempenho, com números no corpo |
| `test` | Testes novos ou corrigidos, sem código de produção |
| `docs` | Documentação, ADRs, ARQUITETURA |
| `build` | `pyproject.toml`, dependências, `uv.lock` |
| `ci` | Workflows do GitHub Actions |
| `chore` | Manutenção que não se encaixa acima |

**Escopos:** `core`, `neat`, `envs`, `lab`, `io`, `viz`, `config`, `arquitetura`, `adr`, `repo`, `github`.

**Exemplos:**

```text
feat(core): implementa distância toroidal pela menor imagem
test(core): adiciona propriedades de simetria do toroide
fix(core): corrige posição igual a W após o wrap
docs(adr): registra calibração de p_sol na F1
build(repo): adiciona numba às dependências
```

**Regras:**

- Commits atômicos: uma mudança lógica, com testes passando em cada commit.
- Código e seus testes no mesmo commit.
- Nada de `WIP`, `ajustes`, `fix typo` no histórico enviado. Antes do push, reorganize localmente (`git rebase -i`) se necessário.
- Nunca `--no-verify`.

## 5. Sem coautoria

- Commits, PRs, issues e comentários **não** levam `Co-Authored-By`, "Generated with Claude Code" nem qualquer marca de ferramenta.
- A atribuição automática está desligada em `.claude/settings.json` (`attribution.commit` e `attribution.pr` vazios).
- O autor é a identidade git já configurada na máquina. O agente nunca altera `user.name` nem `user.email`.
- **Verificação antes do push:**

```bash
git log --format=%B origin/main..HEAD | grep -iE "co-authored-by|generated with" && echo "REMOVER ATRIBUIÇÃO" || echo "ok"
```

## 6. Pull requests

- **Quando abrir:** quando todos os itens do checklist do bloco estão prontos e o critério de saída da fase passa.
- **Título:** `[FN] Nome da fase` (ou `[FN.2] Nome — bloco 2`).
- **Corpo:** pelo template, com resumo, itens entregues, evidência do critério de saída (números e saídas de teste), ADRs criados, como testar e `Closes #N` (no último bloco da fase) ou `Refs #N`.
- **Checks obrigatórios:** CI verde, suíte completa, verificação de atribuição (§5).
- **Validação antes do merge:** feita pelo agente, sobre o SHA final do PR: suíte completa, `ruff`, `mypy`, `lint-imports`, verificação de atribuição (§5), leitura do diff contra a ARQUITETURA e CI verde. Se algo falhar, o agente corrige e valida de novo, sem fazer o merge.
- **Merge:** feito pelo agente, com merge commit, para preservar os commits atômicos, e preso ao SHA validado: `gh pr merge N --merge --delete-branch --match-head-commit <sha>`.
- **Depois do merge:** `git switch main && git pull --ff-only && git branch -d fase/...`. Ative em Settings do GitHub a opção "Automatically delete head branches".

## 7. Tags

Depois do merge de cada fase, o usuário pode criar uma tag anotada: `v0.0` (F0), `v0.1` (F1) … `v1.0` (F5). O agente só cria tags se pedido.

## 8. Proibido

- Force-push na `main` ou em branch alheio.
- Reescrever histórico já mergeado.
- Commits diretos na `main` (exceto o bootstrap).
- Mais de um PR aberto ao mesmo tempo.
- Abrir PRs ou branches para mudanças triviais isoladas: elas entram no branch da fase atual.
