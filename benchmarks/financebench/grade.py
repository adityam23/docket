#!/usr/bin/env python
"""Combine deterministic numeric passes with Claude's out-of-band verdicts into a
FinanceBench scorecard.

Split from generation on purpose: on a 6 GB box we can't hold a second judge
model beside the candidate, and a small-model judge is noisy on financial
free-form answers — so Claude grades the ``needs_judge`` rows directly (the user's
instruction). A row is CORRECT if the numeric matcher already resolved it
(``auto_numeric_ok``) OR Claude's verdict says so.

    # 1. export responses:   run_full.py ... --export responses.jsonl
    # 2. Claude writes verdicts.jsonl:  {"id": ..., "correct": true/false}
    # 3. grade:
    uv run python benchmarks/financebench/grade.py \
        --responses responses.jsonl --verdicts verdicts.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--responses", required=True)
    ap.add_argument("--verdicts", help="JSONL of {id, correct} from Claude's judging")
    ap.add_argument("--label")
    args = ap.parse_args()

    rows = _load(args.responses)
    verdicts = {v["id"]: bool(v["correct"]) for v in _load(args.verdicts)} if args.verdicts else {}

    by_type: dict[str, list[bool]] = defaultdict(list)
    correct = cited = citable = ungraded = 0
    for r in rows:
        if r["auto_numeric_ok"]:
            ok = True
        elif r["id"] in verdicts:
            ok = verdicts[r["id"]]
        else:
            ok = False
            ungraded += 1
        correct += int(ok)
        by_type[r.get("question_type") or "?"].append(ok)
        resp = r.get("response") or ""
        is_refusal = "do not contain" in resp.lower()
        if not is_refusal:
            citable += 1
            if "[" in resp and "]" in resp:
                cited += 1

    n = len(rows)
    label = args.label or Path(args.responses).stem
    print(f"\n=== {label} — FinanceBench (150-Q, Claude-judged) ===")
    print(f"  accuracy   {correct/n:.1%}  ({correct}/{n})")
    print(f"  citations  {cited}/{citable}")
    if ungraded:
        print(f"  ⚠ {ungraded} rows needed a judge verdict but had none (counted wrong)")
    print("  by question_type:")
    for qt, oks in sorted(by_type.items()):
        print(f"    {qt:<22} {sum(oks)}/{len(oks)}  ({sum(oks)/len(oks):.0%})")


if __name__ == "__main__":
    main()
