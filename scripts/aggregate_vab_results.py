"""Aggregate per-trial outcomes from a CaP-X output directory into a summary.

Usage:
    python scripts/aggregate_vab_results.py outputs/google_gemini-3.1-pro-preview

Reports per-task: trials_run, trials_succeeded, success_rate, avg_wallclock_seconds,
avg_code_blocks, avg_regenerations.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


_TRIAL_PATTERN = re.compile(
    r"^trial_(?P<trial>\d+)_sandboxrc_(?P<rc>\d+)"
    r"_reward_(?P<reward>[0-9.]+)_taskcompleted_(?P<completed>\d+)$"
)


def _summarize_task(task_dir: Path) -> dict[str, Any]:
    """Aggregate every `trial_*` directory under `task_dir`.

    For each trial id, take the *latest* sandbox-rc attempt (i.e., highest
    rc index) so retries that succeeded after a failed first pass are
    counted as the trial's outcome.
    """
    trials: dict[str, dict[str, Any]] = {}  # trial_id -> latest attempt info
    for sub in task_dir.iterdir():
        if not sub.is_dir():
            continue
        m = _TRIAL_PATTERN.match(sub.name)
        if not m:
            continue
        tid = m.group("trial")
        rc = int(m.group("rc"))
        existing = trials.get(tid)
        if existing is not None and existing["rc"] >= rc:
            continue
        info: dict[str, Any] = {
            "rc": rc,
            "reward": float(m.group("reward")),
            "completed": int(m.group("completed")),
            "dir": sub.name,
        }
        # Pull richer metadata from all_responses.json if it exists.
        ar = sub / "all_responses.json"
        if ar.exists():
            try:
                payload = json.loads(ar.read_text())
                if isinstance(payload, dict):
                    info.update(
                        {
                            "num_code_blocks": payload.get("num_code_blocks"),
                            "num_regenerations": payload.get("num_regenerations"),
                            "num_finishes": payload.get("num_finishes"),
                        }
                    )
            except Exception:
                pass
        trials[tid] = info

    n_trials = len(trials)
    successes = sum(1 for t in trials.values() if t["completed"] == 1)
    avg_blocks = (
        sum(t.get("num_code_blocks") or 0 for t in trials.values()) / max(n_trials, 1)
    )
    avg_regen = (
        sum(t.get("num_regenerations") or 0 for t in trials.values()) / max(n_trials, 1)
    )
    return {
        "task": task_dir.name,
        "trials_run": n_trials,
        "trials_succeeded": successes,
        "success_rate": successes / n_trials if n_trials else 0.0,
        "avg_code_blocks": avg_blocks,
        "avg_regenerations": avg_regen,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "root",
        type=Path,
        help="Path containing per-task output dirs (e.g., outputs/google_gemini-3.1-pro-preview)",
    )
    args = p.parse_args()

    if not args.root.exists():
        print(f"Path not found: {args.root}", file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = []
    for task_dir in sorted(args.root.iterdir()):
        if not task_dir.is_dir():
            continue
        if not any(task_dir.iterdir()):
            continue
        rows.append(_summarize_task(task_dir))

    if not rows:
        print("No task directories found.")
        return 0

    print(
        f"{'task':40s} {'trials':>7s} {'success':>8s} {'rate':>7s} "
        f"{'avg_blocks':>11s} {'avg_regen':>10s}"
    )
    print("-" * 90)
    for r in rows:
        print(
            f"{r['task']:40s} {r['trials_run']:>7d} {r['trials_succeeded']:>8d} "
            f"{r['success_rate']:>6.2%} "
            f"{r['avg_code_blocks']:>11.2f} {r['avg_regenerations']:>10.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
