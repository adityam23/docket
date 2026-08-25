#!/usr/bin/env python
"""Build a grounded SFT dataset for the generator LoRA.

Sources FinanceBench (real 150-Q open subset) and/or the contained smoke set,
converts each into a chat example in the EXACT grounded format served in
production (system + numbered-context user turn + cited assistant answer), and
writes train/val JSONL ready for QLoRA (see ``train_qlora.py``).

    uv run python benchmarks/financebench/build_trainset.py \
        --source financebench --val-frac 0.1 --out-dir benchmarks/financebench/trainset

NOTE: FinanceBench's 150 open Qs are few for SFT — treat this as the seed. Scale
with synthetic grounded Q/A generated over your own ingested filings (same
format) before the real run; the builder is the single conversion point for both.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from docket.eval.financebench import (  # noqa: E402
    fb_to_item,
    load_financebench,
    to_sft_example,
)

HERE = Path(__file__).parent
SMOKE = HERE / "smoke_set.jsonl"
FB_CACHE = HERE / "data" / "financebench_merged.jsonl"


def _items(source: str) -> list[dict]:
    items: list[dict] = []
    if source in ("financebench", "both"):
        items += [fb_to_item(r) for r in load_financebench(str(FB_CACHE))]
    if source in ("smoke", "both"):
        items += [json.loads(l) for l in SMOKE.read_text().splitlines() if l.strip()]
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["financebench", "smoke", "both"], default="financebench")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out-dir", default=str(HERE / "trainset"))
    args = ap.parse_args()

    items = _items(args.source)
    examples = [to_sft_example(it) for it in items]
    # Deterministic split (no RNG — reproducible): every 1/val_frac-th to val.
    step = max(int(round(1 / args.val_frac)), 2) if args.val_frac else 0
    train, val = [], []
    for i, ex in enumerate(examples):
        (val if step and i % step == 0 else train).append(ex)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        p = out / f"{name}.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))
        print(f"wrote {p}  ({len(rows)} examples)")
    print(f"total {len(examples)} from source={args.source}")


if __name__ == "__main__":
    main()
