#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Optionally initialize mono-kernel workflow state and an empty experiment log."""

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model name for the task record")
    parser.add_argument("--output", required=True, type=Path, help="New task directory; never overwritten")
    args = parser.parse_args()
    if not args.model.strip() or any(ord(c) < 32 for c in args.model):
        parser.error("Model name must be nonempty and on one line")

    output = args.output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        parser.error(f"Output already exists; use existing records or choose a new directory: {output}")
    workflow = {
        "schema_version": 1,
        "model": args.model,
        "objective": None,
        "target_buckets": [],
        "inputs": [],
        "phase": "P0",
        "phase_status": "not_started",
        "reference_R": None,
        "parallel_plans": {},
        "active_parallel_plan": None,
        "fastest_validated_B_by_bucket": {},
        "functional_C": None,
        "current_partition": None,
        "active_candidate": None,
        "last_result": None,
        "next_action": {
            "action": "Select a real representative layer and implement snapshot/restore plus a reference runner; map its actual device DAG.",
            "exit_criterion": "The same initial inputs/state/cache can be replayed with defined output and side-effect checks.",
            "on_failure": "Isolate fixture/reference drift before writing fusion code.",
        },
        "rejected_hypotheses": [],
        "branch_limits": None,
        "live_jobs": [],
    }
    records = {
        "workflow-state.json": json.dumps(workflow, indent=2, ensure_ascii=False) + "\n",
        "experiments.jsonl": "",
    }

    try:
        output.mkdir(parents=True, exist_ok=False)
        for name, content in records.items():
            with (output / name).open("x", encoding="utf-8") as target:
                target.write(content)
    except OSError as error:
        parser.exit(1, f"Could not finish creating records: {error}\n")
    print(json.dumps({"output": str(output.resolve()), "files": sorted(records)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
