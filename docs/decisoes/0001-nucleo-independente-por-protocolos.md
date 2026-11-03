# 0001. Núcleo independente por inversão de dependência

- **Data:** 2026-11-03
- **Status:** aceita
- **Fase:** F0
- **Seções afetadas:** ARQUITETURA §2 e §2.1; PADROES §2

## Contexto

A ARQUITETURA §2.1 dizia que `genome.core` importa `genome.neat` (operadores de genoma e compilação de redes, usados nos nascimentos) e que `genome.neat` não importa nenhum outro pacote. PADROES §2, AGENTS.md e as issues do roadmap dizem o contrário: o núcleo não importa nenhum pacote do GENOME, declara protocolos, e `genome.neat` importa `genome.core`. As duas regras não cabem juntas no `import-linter`.

## Decisão

Prevalece PADROES §2, por decisão do usuário. O núcleo declara `Brain` e `GenomeOperators` em `genome.core.protocols`; `genome.neat` (F3) e os controladores fixos de `genome.lab` (F1) implementam esses protocolos, e o `genome.lab` os injeta. A ARQUITETURA §2 e §2.1 foram reescritas para dizer o mesmo.

## Consequências

- O contrato "core é independente" do `import-linter` proíbe `genome.core` de importar `neat`, `envs`, `lab`, `io` e `viz`.
- Os nascimentos (§7.4) recebem um `GenomeOperators` por injeção. Na F1, a reprodução clonal usa uma implementação mínima fora do núcleo.
- O núcleo pode ser testado com dublês dos protocolos, sem NEAT.
