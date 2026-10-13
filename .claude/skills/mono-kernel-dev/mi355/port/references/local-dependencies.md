# Sharing and runtime dependencies

Share the complete `mono-kernel-dev/mi355/{port,schedule,numerics,benchmark,atom}` suite. References, assets, scripts, and agent metadata travel with it. Cases and development methods are self-contained; current model parameters come from the current project.

## Removed historical dependencies

| Previous dependency | Replacement |
|---|---|
| External method documents or author-machine checkout | Methods are embedded; see the [case/method index](sources.md) |
| Historical source manifest with 31 machine paths | Removed from the bundle; case logic, numerical extracts, and evidence limits remain |
| Fixed bootstrap installation path | Locate the [initializer](../SKILL.md) relative to the loaded port skill; Python standard library only |
| Bootstrap provenance paths and `--source` | Removed; initialize only current state and an experiment log. Record actual identities during development using the [iteration rules](iteration-loop.md) |

Bootstrap is optional. Reuse existing records; do not delete or reinitialize old task evidence. Use contract/dependency/fusion templates only as needed. Read the workflow from the skill rather than copying an execution plan.

## Bundle structure and installation

All five short-named directories are siblings under `mi355`. Relative cross-skill links require that structure; copying one leaf alone omits shared workflows/cases. Keep the English and Chinese trees separate. Install only the desired language to avoid duplicate skill names; the package-level README describes portable installation.

The Chinese edition is a corresponding reviewed copy, not a runtime dependency of the English edition. No absolute installation path or external repository link is required by either bundle.

## Current-project execution dependencies

The initializer needs Python 3. Kernel work needs the current project's ROCm/HIP, GPU DSL/compiler, reference layer, quantization/synchronization/communication helpers, and target GPU. ATOM integration additionally needs the applicable ATOM project.

Discover interfaces/versions locally and check layout, memory scope, and progress semantics. A matching helper name does not establish a matching contract. These are execution dependencies, not bindings to historical nodes/checkouts. The bundle does not include full models/runtime or one executable kernel for every new model.

## Portability acceptance

After relocation, all relative links must resolve and the initializer must still create two records. Shared content must not depend on author-machine files or private repositories. Methods and extracts are independently readable; executing kernels still requires the task's project, inputs, and hardware. Unbundled historical raw experiments are not claimed to be fully reproducible from this package.
