# Reproducers for two bundled-LLVM issues in FlyDSL 0.4.0

- `bug-a/`: the same module builds into different code objects from process to process (ReassociatePass and allocation order).
- `bug-b/`: an SGPR spill restores lanes that were never stored (greedy live-range splitting).

See the linked ROCm/FlyDSL issues for how to run them.
