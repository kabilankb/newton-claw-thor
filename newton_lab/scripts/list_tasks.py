#!/usr/bin/env python3
"""List Newton tasks, or verify tasks_registry.json matches the code registry.

    python list_tasks.py            # print the registry (stdlib only)
    python list_tasks.py --check    # exit 1 if the JSON and newton_lab.envs disagree (needs the venv)
"""
import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
reg = json.loads((LAB / "tasks_registry.json").read_text())["environments"]

if "--check" in sys.argv:
    sys.path.insert(0, str(LAB))
    from newton_lab import envs
    a, b = {e["id"] for e in reg}, set(envs.task_ids())
    if a != b:
        sys.exit(f"MISMATCH json-only={sorted(a - b)} code-only={sorted(b - a)}")
    print(f"OK {len(a)} tasks in sync")
else:
    for e in reg:
        print(f"{e['id']:40s} {e['robot']:22s} {e['category']:10s} {e['notes']}")
