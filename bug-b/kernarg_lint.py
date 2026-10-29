"""Lint of a code object's kernel-argument handling (gfx950 ISA):

  python3 kernarg_lint.py CODE_OBJECT_OR_ISA... [--names a,b,c] [--json OUT]

Each argument is followed from its scalar kernarg load (``s_load_dword* sD, s[K:K+1]``
with K..K+1 the kernarg pointer) through ``s_mov`` copies, a buffer resource's masked
high word, SGPR spill lanes (``v_writelane`` / ``v_readlane``) and whole-VGPR stack slots
(``scratch_`` / ``buffer_`` dword stores and loads: spills to memory), as a forward dataflow
over the code's basic blocks (branch targets from ``s_branch`` / ``s_cbranch_*``). Two
checks:

- ``unread``: an argument some load brings in, but no instruction on any path reads it
  from any of its loads. Either the kernel does not use the argument (a load merged over
  it) or the code takes its value from somewhere else, such as a spill slot holding
  another argument.
- ``split``: a 64-bit address (a buffer resource's or a scalar load's base) whose low
  and high words come, on every path, from two different arguments.

Prints ``LINT <file> ok`` or ``LINT <file> HIT unread=[..] split=[..]`` per input (exit
status 1 on any hit). A ``.co`` is disassembled with ``$LLVM_BIN/llvm-objdump`` (default
/opt/rocm/llvm/bin); the argument offsets come from its metadata (for a ``.s``, from the
``.co`` beside it, else 8 bytes each). ``--names``: the arguments' names in order (the
launch signature), for the report."""

import argparse
import json
import os
import re
import subprocess
import sys

LLVM_BIN = os.environ.get("LLVM_BIN", "/opt/rocm/llvm/bin")
LOADS = {"s_load_dword": 1, "s_load_dwordx2": 2, "s_load_dwordx4": 4, "s_load_dwordx8": 8,
         "s_load_dwordx16": 16}
# opcodes whose first operand is not a destination
NO_DST = ("s_cmp", "s_bitcmp", "s_cbranch", "s_branch", "s_waitcnt", "s_barrier", "s_nop",
          "s_sleep", "s_setprio", "s_endpgm", "s_dcache", "s_icache", "s_trap", "s_sethalt",
          "s_set_gpr_idx", "s_setreg", "s_store", "s_buffer_store", "s_atomic",
          "buffer_store", "buffer_atomic", "buffer_wbl2", "buffer_inv", "global_store",
          "global_atomic", "flat_store", "flat_atomic", "scratch_store", "ds_write",
          "ds_store", "ds_add", "v_writelane", "s_ttrace", "s_wakeup", "s_mov_b32 exec",
          "s_or_b64 exec", "s_and_b64 exec", "s_xor_b64 exec", "s_andn2_b64 exec")


def disassemble(path):
    """(instructions [(offset, text)], argument (offset, size) list or None)."""
    args = None
    co = path[:-2] + ".co" if path.endswith(".s") else path
    if path.endswith(".s"):
        text = open(path).read()
    else:
        text = subprocess.run([f"{LLVM_BIN}/llvm-objdump", "-d", "--mcpu=gfx950", path],
                              capture_output=True, text=True, check=True).stdout
    if os.path.exists(co) and os.path.exists(f"{LLVM_BIN}/llvm-readelf"):
        notes = subprocess.run([f"{LLVM_BIN}/llvm-readelf", "--notes", co],
                               capture_output=True, text=True, check=True).stdout
        args = [(int(o), int(s)) for o, s in
                re.findall(r"\.offset:\s+(\d+)\s+\.size:\s+(\d+)", notes)] or None
    ins, start = [], None
    for ln in text.splitlines():
        m = re.match(r"^([0-9a-f]+) <([^>]+)>:", ln)
        if m:
            if start is None:
                start = int(m.group(1), 16)
            continue
        if start is None:
            continue
        m = re.match(r"^\s+(.*?)\s*//\s*([0-9A-Fa-f]+):", ln)
        if m:
            ins.append((int(m.group(2), 16) - start, m.group(1).strip()))
    return ins, args


def sregs(op):
    m = re.fullmatch(r"s(\d+)", op)
    if m:
        return [int(m.group(1))]
    m = re.fullmatch(r"s\[(\d+):(\d+)\]", op)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    return []


def split(text):
    parts = text.split(None, 1)
    if len(parts) < 2:
        return parts[0], []
    return parts[0], [o.strip() for o in parts[1].split(",")]


BRANCH = re.compile(r"s_(c?branch)(_\w+)? (-?\d+)$")
MEM_DWORD = re.compile(r"^(scratch|buffer)_(store|load)_dword$")


def blocks_of(ins):
    """Basic blocks [(first, last) instruction indices] and their successor blocks."""
    at = {o: i for i, (o, _) in enumerate(ins)}
    leaders, targets = {0}, {}
    for i, (o, t) in enumerate(ins):
        m = BRANCH.match(t)
        if m:
            simm = int(m.group(3))
            simm -= 65536 if simm >= 32768 else 0  # a signed 16-bit dword offset
            tgt = at.get(o + 4 + 4 * simm)
            targets[i] = tgt
            if tgt is not None:
                leaders.add(tgt)
            leaders.add(i + 1)
        elif t.startswith(("s_endpgm", "s_setpc", "s_swappc")):
            leaders.add(i + 1)
    starts = sorted(x for x in leaders if x < len(ins))
    index = {b: n for n, b in enumerate(starts)}
    blocks, succ = [], []
    for n, b in enumerate(starts):
        last = (starts[n + 1] if n + 1 < len(starts) else len(ins)) - 1
        t = ins[last][1]
        nxt = []
        if last in targets and targets[last] is not None:
            nxt.append(index[targets[last]])
        if not (t.startswith(("s_branch", "s_endpgm", "s_setpc")) and not t.startswith("s_cbranch")):
            if last + 1 < len(ins):
                nxt.append(index[last + 1])
        blocks.append((b, last))
        succ.append(nxt)
    return blocks, succ


def transfer(state, o, t, events):
    """One instruction on ``state`` ({SGPR or (VGPR, lane): frozenset of provenances}:
    ("K", kernarg dword, load offset) or ("P", half) for the kernarg pointer). ``events``
    collects (kind, ...): loads ("load", dword, at), reads ("read", provenance) and the
    address pairs used ("addr", lo provenances, hi provenances, at). An instruction's
    reads are taken before its writes."""
    op, ops = split(t)
    get = lambda r: state.get(r, frozenset())  # noqa: E731
    # the SGPRs it reads
    if op == "v_writelane_b32":
        srcs = sregs(ops[1]) if len(ops) >= 2 else []
    elif op == "v_readlane_b32":
        srcs = []
    elif op in LOADS or op in ("s_mov_b32", "s_mov_b64") or (
            op == "s_and_b32" and len(ops) == 3 and ops[2] == "0xffff"):
        srcs = [r for x in ops[1:] for r in sregs(x)]
    else:
        has_dst = bool(ops) and not t.startswith(NO_DST) and bool(sregs(ops[0]))
        srcs = [r for x in (ops[1:] if has_dst else ops) for r in sregs(x)]
    for r in srcs:
        for p in get(r):
            if p[0] == "K":
                events.append(("read", p))
    for x in ops[1:]:
        rs = sregs(x)
        if len(rs) in (2, 4) and (op.startswith(("buffer_", "s_load", "s_buffer_load"))
                                  or (op.startswith("global_") and len(rs) == 2)):
            events.append(("addr", get(rs[0]), get(rs[1]), o))
    # then its writes
    if op in LOADS and len(ops) >= 3:
        base = sregs(ops[1])
        if len(base) == 2 and get(base[0]) == {("P", 0)} and get(base[1]) == {("P", 1)}:
            off = int(ops[2], 16) if ops[2].startswith("0x") else int(ops[2])
            for k, r in enumerate(sregs(ops[0])):
                state[r] = frozenset({("K", off // 4 + k, o)})
                events.append(("load", off // 4 + k, o))
        else:
            for r in sregs(ops[0]):
                state.pop(r, None)
    elif MEM_DWORD.match(op) and len(ops) >= 3:
        # a whole VGPR to or from a stack slot (spills to memory): its lanes' provenances
        # follow it, keyed by the slot's base register and offset
        store = "_store_" in op
        if op.startswith("scratch_"):
            vreg, tail = (ops[1] if store else ops[0]), ops[2]
        else:
            vreg, tail = ops[0], (ops[3] if len(ops) > 3 else "")
        m = re.match(r"(\S+)(?:\s+offset:(\d+))?", tail)
        slot = ("M", m.group(1), int(m.group(2) or 0)) if m else ("M", tail, 0)
        src, dst = (vreg, slot) if store else (slot, vreg)
        for k in [k for k in state if isinstance(k, tuple) and k[0] == dst]:
            del state[k]
        for k, v in [(k, v) for k, v in state.items() if isinstance(k, tuple) and k[0] == src]:
            state[(dst, k[1])] = v
    elif op == "v_writelane_b32" and len(ops) >= 3:
        if ops[2].isdigit():
            key = (ops[0], int(ops[2]))
            if srcs and get(srcs[0]):
                state[key] = get(srcs[0])
            else:
                state.pop(key, None)
    elif op == "v_readlane_b32" and len(ops) >= 3:
        d = sregs(ops[0])
        if d:
            v = state.get((ops[1], int(ops[2]))) if ops[2].isdigit() else None
            if v:
                state[d[0]] = v
            else:
                state.pop(d[0], None)
    elif op in ("s_mov_b32", "s_mov_b64") and len(ops) == 2 and sregs(ops[0]):
        d, sr = sregs(ops[0]), sregs(ops[1])
        vals = [get(sr[i]) if i < len(sr) else frozenset() for i in range(len(d))]
        for r, v in zip(d, vals):
            if v:
                state[r] = v
            else:
                state.pop(r, None)
    elif op == "s_and_b32" and len(ops) == 3 and ops[2] == "0xffff" and sregs(ops[0]):
        # a pointer's high word masked into a buffer resource keeps its provenance
        d, sr = sregs(ops[0]), sregs(ops[1])
        v = get(sr[0]) if sr else frozenset()
        if v:
            state[d[0]] = v
        else:
            state.pop(d[0], None)
    else:
        dst = [] if t.startswith(NO_DST) else sregs(ops[0]) if ops else []
        for r in dst:
            state.pop(r, None)


def analyze(ins, must):
    """Forward dataflow over the basic blocks: ``must`` keeps a provenance only where
    every path into a block agrees; else every path's provenances are kept."""
    blocks, succ = blocks_of(ins)
    parsed = [(o, t) for o, t in ins]
    preds = [[] for _ in blocks]
    for b, nx in enumerate(succ):
        for n in nx:
            preds[n].append(b)
    out = [None] * len(blocks)
    entry = {0: frozenset({("P", 0)}), 1: frozenset({("P", 1)})}
    work = list(range(len(blocks)))
    queued = set(work)
    while work:
        b = work.pop(0)
        queued.discard(b)
        ins_states = [out[p] for p in preds[b] if out[p] is not None]
        if b == 0:
            ins_states.append(entry)
        if not ins_states:
            continue
        if must:
            st = dict(ins_states[0])
            for other in ins_states[1:]:
                for k in list(st):
                    if other.get(k) != st[k]:
                        del st[k]
        else:
            st = {}
            for other in ins_states:
                for k, v in other.items():
                    st[k] = st.get(k, frozenset()) | v
        ev = []
        first, last = blocks[b]
        for o, t in parsed[first:last + 1]:
            transfer(st, o, t, ev)
        if st != out[b]:
            out[b] = st
            for n in succ[b]:
                if n not in queued:
                    work.append(n)
                    queued.add(n)
    # one more pass with every block's settled input, collecting the events
    events = []
    for b, (first, last) in enumerate(blocks):
        ins_states = [out[p] for p in preds[b] if out[p] is not None]
        if b == 0:
            ins_states.append(entry)
        if not ins_states:
            continue
        if must:
            st = dict(ins_states[0])
            for other in ins_states[1:]:
                for k in list(st):
                    if other.get(k) != st[k]:
                        del st[k]
        else:
            st = {}
            for other in ins_states:
                for k, v in other.items():
                    st[k] = st.get(k, frozenset()) | v
        for o, t in parsed[first:last + 1]:
            transfer(st, o, t, events)
    return events


def lint(ins, arg_layout=None):
    """(unread argument indices, split address pairs [(lo arg, hi arg, offset)])."""
    may = analyze(ins, must=False)
    loaded = {}
    for e in may:
        if e[0] == "load":
            loaded.setdefault(e[1], set()).add(e[2])
    read = {(p[1], p[2]) for e in may if e[0] == "read" for p in [e[1]]}
    if arg_layout is None:
        top = max(loaded) if loaded else -1
        arg_layout = [(8 * a, 8) for a in range(top // 2 + 1)]
    unread = []
    for idx, (off, size) in enumerate(arg_layout):
        dws = range(off // 4, (off + size + 3) // 4)
        if not any(d in loaded for d in dws):
            continue
        if not any((d, ld) in read for d in dws for ld in loaded.get(d, ())):
            unread.append(idx)

    def arg_of(dword):
        for idx, (off, size) in enumerate(arg_layout):
            if off // 4 <= dword < (off + size + 3) // 4:
                return idx
        return None

    # an address whose two words certainly come from two different arguments
    split_ptrs = {}
    for e in analyze(ins, must=True):
        if e[0] == "addr" and len(e[1]) == 1 and len(e[2]) == 1:
            (lo,), (hi,) = e[1], e[2]
            if lo[0] == hi[0] == "K" and hi[1] != lo[1] + 1:
                split_ptrs.setdefault((arg_of(lo[1]), arg_of(hi[1])), e[3])
    return unread, [(lo, hi, at) for (lo, hi), at in sorted(split_ptrs.items(), key=lambda x: x[1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--names", default="", help="the arguments' names, comma-separated")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    names = [n for n in a.names.split(",") if n]
    hits = 0
    for path in a.inputs:
        ins, layout = disassemble(path)
        unread, split_ptrs = lint(ins, layout)
        nm = lambda i: names[i] if i is not None and i < len(names) else f"arg{i}"  # noqa: E731
        un = [nm(i) for i in unread]
        sp = [f"{nm(lo)}.lo+{nm(hi)}.hi@+{at}" for lo, hi, at in split_ptrs]
        ok = not un and not sp
        hits += not ok
        print(f"LINT {path} " + ("ok" if ok else f"HIT unread={un} split={sp}"), flush=True)
        if a.json:
            with open(a.json, "a") as f:
                f.write(json.dumps({"file": path, "ok": ok, "unread": un, "split": sp}) + "\n")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
