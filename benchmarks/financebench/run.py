#!/usr/bin/env python
"""Run the FinanceBench GENERATOR smoke test against one candidate, or compare
several saved scorecards.

The candidate only has to speak the OpenAI ``/v1`` contract, so a local
``llama-server -m model.gguf --port 8971`` is a drop-in target — we reuse the
production :class:`OpenAICompatProvider`, no candidate-specific inference code.

    # score one reachable candidate and save its card
    uv run python benchmarks/financebench/run.py \
        --base-url http://127.0.0.1:8971/v1 --model qwen3.5-4b \
        --label "Qwen3.5-4B Q4" --out benchmarks/financebench/results/qwen35-4b.json

    # print the head-to-head table once cards exist
    uv run python benchmarks/financebench/run.py --compare benchmarks/financebench/results/*.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

# Make the package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from docket.eval.financebench import Scorecard, run_smoke  # noqa: E402
from docket.providers.openai_compat import OpenAICompatProvider  # noqa: E402

SMOKE_SET = Path(__file__).with_name("smoke_set.jsonl")


def load_items(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def print_card(card: Scorecard) -> None:
    s = card.summary()
    print(f"\n=== {s['model']} ===")
    print(f"  accuracy      {s['accuracy']:.0%}  ({s['correct']}/{s['n']})")
    print(f"  citations     {s['cited']} (of answerable)")
    print(f"  refusal       {s['refusal_ok']} correct")
    print(f"  hallucinated  {s['hallucinated']}")
    print(f"  avg latency   {s['avg_latency_ms']} ms")
    for it in card.items:
        mark = "PASS" if it.ok else "FAIL"
        print(f"    [{mark}] {it.id:<20} {it.reason}")


def compare(paths: list[str]) -> None:
    cards = [json.loads(Path(p).read_text()) for p in paths]
    cols = ["model", "accuracy", "correct", "cited", "refusal_ok", "hallucinated", "avg_latency_ms"]
    widths = {c: max(len(c), *(len(str(card[c])) for card in cards)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for card in sorted(cards, key=lambda c: c["accuracy"], reverse=True):
        print("  ".join(str(card[c]).ljust(widths[c]) for c in cols))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", help="OpenAI-/v1 base URL of the candidate")
    ap.add_argument("--model", help="model id the endpoint serves")
    ap.add_argument("--label", help="display label for the scorecard")
    ap.add_argument("--out", help="write the scorecard JSON here")
    ap.add_argument("--set", default=str(SMOKE_SET), help="smoke-set jsonl path")
    ap.add_argument("--max-tokens", type=int, default=384,
                    help="response budget; raise for reasoning models that emit <think>")
    ap.add_argument("--compare", nargs="+", help="print a table from saved cards")
    args = ap.parse_args()

    if args.compare:
        paths = [p for pat in args.compare for p in glob.glob(pat)]
        compare(paths)
        return

    if not (args.base_url and args.model):
        ap.error("need --base-url and --model (or --compare)")

    items = load_items(Path(args.set))
    provider = OpenAICompatProvider(name="candidate", base_url=args.base_url, chat_model=args.model)
    card = run_smoke(provider, items, model_label=args.label or args.model, max_tokens=args.max_tokens)
    print_card(card)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        Path(args.out).write_text(json.dumps(card.summary(), indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
