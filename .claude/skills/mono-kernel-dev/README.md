# MI355 mono-kernel development recipes

Start with [port](mi355/port/SKILL.md). Build a real layer fixture and independent reference, bring up staged bodies, fuse incrementally, and use the RSI loop to select the next experiment. Read supporting references only at the gate or symptom that needs them; the numerical evidence JSON is for recomputation, not default context.

| Skill | Use |
|---|---|
| [port](mi355/port/SKILL.md) | P0–P5 bring-up, R/B/C state, RSI loop, parallel plans, acceptance and delivery |
| [schedule](mi355/schedule/SKILL.md) | Tile/storage contracts, persistent workers, incremental fusion, reuse and progress |
| [numerics](mi355/numerics/SKILL.md) | Independent oracles, quantization/routing, cache/state, first-divergence diagnosis |
| [benchmark](mi355/benchmark/SKILL.md) | Paired timing, direct cases, bounded search, PMC/ATT interpretation |
| [atom](mi355/atom/SKILL.md) | Early integration seams, preparation, dispatch/fallback, Graph and decode/MTP lifecycle |

In a FlyDSL checkout, the five `mi355-mono-kernel-*` links in `.claude/skills` expose these skills directly to Claude Code. Start with `/mi355-mono-kernel-port`; no installation step is needed for this checkout. The relative links and the complete package are tracked together so references continue to work after cloning or moving the repository.

A **fusion island** is a connected subgraph compiled into one kernel. **Persistent workers** reuse CTAs for tasks within a declared lifetime; a cross-invocation service is a separate design. **R/B/C** means independent reference, best validated implementation, and construction branch. Historical B/S/C shape labels follow each case's request/token convention.

Keep all five sibling directories under `mi355`. Skill names remain `mi355-mono-kernel-*` for invocation. To install on Linux with GNU `ln`, set `MONO_PACKAGE_ROOT` to this package's resolved root and `MONO_SKILL_INSTALL_DIR` to a skill directory discovered by your agent, then link each leaf:

```bash
(
    set -eu
    : "${MONO_PACKAGE_ROOT:?Set MONO_PACKAGE_ROOT first}"
    : "${MONO_SKILL_INSTALL_DIR:?Set MONO_SKILL_INSTALL_DIR first}"
    mkdir -p "$MONO_SKILL_INSTALL_DIR"
    for part in port schedule numerics benchmark atom; do
        ln -sT "$MONO_PACKAGE_ROOT/mi355/$part" "$MONO_SKILL_INSTALL_DIR/mi355-mono-kernel-$part"
    done
)
```

The command stops with an error at an existing destination, including a directory or a valid/dangling symlink, without overwriting it or creating a link inside it. Earlier links may already have been created; inspect existing entries before rerunning. Reload the agent if its catalog has not refreshed. Keep the Chinese edition, `mono-kernel-dev-cn`, outside active discovery unless choosing it instead of English. Both editions have the same skill names and corresponding content; neither requires the other.

[Dependency notes](mi355/port/references/local-dependencies.md) distinguish embedded guidance from current-project runtime requirements. [Case provenance](mi355/port/references/sources.md) identifies historical evidence and limitations. The package includes no private-repository links or author-machine dependencies, full model weights, or generic executable kernel. New tile-boundary, EP, and service-loop rules are design guidance; historical measurements retain their original scope.
