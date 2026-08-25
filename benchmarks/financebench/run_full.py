#!/usr/bin/env python
"""Run the REAL FinanceBench open subset (150 Q) against a candidate.

Two modes, same generator + scoring path (only the CONTEXT source differs):

- **oracle-evidence** (default): each question is handed its gold evidence, so we
  measure the generator in isolation — the honest "how good is this model at
  grounded financial QA" ceiling, before retrieval enters the picture.

    uv run python benchmarks/financebench/run_full.py \
        --base-url http://127.0.0.1:11434/v1 --model qwen35-4b-distill \
        --max-tokens 2048 --limit 20            # quick slice

- **full-RAG** (`--rag`): builds a corpus from each filing's PDF and retrieves
  live (dense+sparse → optional cross-encoder rerank), measuring end-to-end
  retrieval+generation. Needs the source filings via `--docs-dir` (one
  `<doc_name>.pdf` per question's `doc_id`) and the ingest extra
  (`uv sync --extra ingest`). Point `--embed-url`/`--rerank-url` at a backend
  serving the finance embedder + reranker (T26). Add `--gap` to also run oracle
  and print the retrieval gap vs the ceiling.

Scoring is numeric-first (deterministic); pass --judge-url/--judge-model to grade
free-form answers with an LLM judge (the canonical FinanceBench method).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from docket.config import Settings  # noqa: E402
from docket.eval.financebench import (  # noqa: E402
    Scorecard,
    _gold_number,
    _match_value_unit_agnostic,
    fb_to_item,
    generate_financebench,
    load_financebench,
    make_rag_retrieve,
    run_financebench,
)
from docket.providers.openai_compat import OpenAICompatProvider  # noqa: E402

CACHE = Path(__file__).with_name("data") / "financebench_merged.jsonl"


def _print_card(card: Scorecard, items: list[dict], mode: str) -> None:
    """Overall + by-question_type (FinanceBench's own taxonomy)."""
    by_type: dict[str, list[bool]] = defaultdict(list)
    for it, r in zip(items, card.items):
        by_type[it["meta"].get("question_type") or "?"].append(r.ok)
    print(f"\n=== {card.model} — FinanceBench ({mode}) ===")
    print(f"  accuracy     {card.accuracy:.1%}  ({card.correct}/{card.n})")
    print(f"  citations    {card.cited}/{card.citable}")
    print(f"  avg latency  {card.avg_latency_ms} ms")
    print("  by question_type:")
    for qt, oks in sorted(by_type.items()):
        print(f"    {qt:<28} {sum(oks)}/{len(oks)}")


def _build_retriever_for(docs_dir: str, settings: Settings, *, verbose: bool = True,
                         corpus_cache: str | None = None):
    """Return a cached ``doc_id -> Retriever`` lookup for full-RAG.

    Each filing is ingested into its OWN corpus (FinanceBench questions target a
    single filing) via the production ingest pipeline, so the finance embedder /
    reranker configured in ``settings`` are exercised end to end — no parallel
    retrieval path (CLAUDE.md). A missing PDF yields ``None`` (→ empty context),
    reported rather than silently faked.

    ``corpus_cache`` (a dir) reuses the Phase-A sweep's per-filing embedded
    corpora (``sweep_recall._cached_corpus``, keyed by embedder) — so we load the
    vectors already computed for the recall sweep instead of re-embedding all 84
    filings, and Phase-B retrieval is byte-identical to what the sweep measured.
    Missing entries are chunked+embedded on the fly and cached like the sweep."""
    from docket.ingest.index import Corpus
    from docket.ingest.pipeline import ingest_paths
    from docket.service import load_retriever

    cached_corpus = None
    if corpus_cache:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from sweep_recall import _cached_corpus  # reuse the sweep's cache (one impl)
        cached_corpus = _cached_corpus

    cache: dict[str, object] = {}
    missing: set[str] = set()

    def retriever_for(doc_id: str):
        if doc_id in cache:
            return cache[doc_id]
        pdf = Path(docs_dir) / f"{doc_id}.pdf"
        if not pdf.is_file():
            if verbose and doc_id not in missing:
                missing.add(doc_id)
                print(f"  [full-RAG] missing filing PDF for {doc_id}: {pdf}")
            cache[doc_id] = None
            return None
        if cached_corpus is not None:
            corpus = cached_corpus(doc_id, pdf, settings, Path(corpus_cache), settings.embed_model)
        else:
            corpus = ingest_paths([str(pdf)], settings=settings, corpus=Corpus())
        retriever = load_retriever(settings, corpus=corpus)
        cache[doc_id] = retriever
        if verbose:
            print(f"  [full-RAG] {doc_id}: {len(corpus)} chunks"
                  f"{' (cached)' if cached_corpus is not None else ''}")
        return retriever

    return retriever_for


def _done_ids(path: str) -> set:
    """Ids already written to a JSONL output — lets a killed run resume by
    appending instead of restarting (mirrors the sweep's checkpoint design)."""
    import json
    import os

    done: set = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    try:
                        done.add(json.loads(line)["id"])
                    except Exception:  # noqa: BLE001 — tolerate a torn last line
                        pass
    return done


def _dump_contexts(items, retrieve, out_path: str, *, base_url: str) -> None:
    """Run retrieval ONCE for every item and persist the hits to JSONL.

    On a 6 GB card the generator, embedder and reranker don't all fit resident
    together, and the retrieved context is generator-independent — so we retrieve
    with the winning embedder+reranker served alone, cache the *real* hits here,
    then generate with each candidate served alone via ``--contexts`` (no faking:
    these are byte-for-byte the hits live retrieval produced). One retrieval pass
    feeds every generator instead of re-retrieving per model.

    Crash-recoverable like the Phase-A sweep: each item is retrieved through the
    shared ``_resilient`` wrapper (waits for the shared backend to come back after
    a restart, then retries), and completed ids are skipped on resume, so a
    backend death mid-run (observed on a big-filing rerank) never loses progress
    or restarts from zero."""
    import json

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sweep_recall import _resilient  # reuse the sweep's backend-wait+retry

    done = _done_ids(out_path)
    todo = [it for it in items if it["id"] not in done]
    if done:
        print(f"  [dump-contexts] resuming: {len(done)} already done, {len(todo)} to go")
    n = len(done)
    with open(out_path, "a") as f:
        for item in todo:
            hits = _resilient(lambda: retrieve(item), base_url, f"retrieve {item['id']}")
            f.write(json.dumps({"id": item["id"], "doc_id": item.get("doc_id"),
                                "n_hits": len(hits), "hits": hits}) + "\n")
            f.flush()
            n += 1
            if n % 10 == 0:
                print(f"  [dump-contexts] {n}/{len(items)} retrieved")
    print(f"\ndumped retrieved contexts for {n} items -> {out_path}")


def _cached_retrieve(path: str):
    """Build a ``retrieve(item) -> hits`` that replays contexts dumped by
    ``_dump_contexts`` — lets generation run with only the generator resident
    (the embedder/reranker aren't needed once the hits are cached)."""
    import json

    cache: dict[str, list] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # A crash mid-dump can leave a torn tail line; skip it rather than
                # abort the whole generation on resume (mirrors _dump_contexts /
                # _export_responses, which already tolerate a torn last line).
                continue
            cache[rec["id"]] = rec.get("hits", [])
    print(f"loaded cached contexts for {len(cache)} items <- {path}")
    return lambda item: cache.get(item["id"], [])


class _ResilientProvider:
    """Wrap a provider so ``.chat`` survives a backend restart: on a disconnect it
    waits for the (watchdog-relaunched) backend and retries, instead of yielding an
    ``[error]`` row. Reuses the sweep's ``_resilient`` — one retry policy for the
    whole benchmark. Transparent for everything else (name, etc.)."""

    def __init__(self, inner, base_url: str):
        self._inner = inner
        self._base_url = base_url
        self.name = getattr(inner, "name", "candidate")

    def __getattr__(self, key):
        return getattr(self._inner, key)

    def chat(self, *args, **kwargs):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from sweep_recall import _resilient
        return _resilient(lambda: self._inner.chat(*args, **kwargs), self._base_url, "chat")


def _export_responses(provider, items, out_path: str, mode: str, *,
                      max_tokens: int, retrieve=None) -> None:
    """Generate answers and write a JSONL for out-of-band (Claude) judging.

    Each row carries the question, gold answer, model response, and an
    ``auto_numeric_ok`` flag from the deterministic numeric matcher — so only the
    rows the matcher can't resolve (``needs_judge``) actually need a human/LLM
    verdict. Keeps the numeric path deterministic while letting Claude grade the
    free-form remainder without loading a second model on a 6 GB box."""
    import json
    import os

    # Resume: skip items already generated with a real (non-error) response, so a
    # killed/crashed run continues instead of restarting. Error rows are re-done.
    done: set = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if not str(row.get("response", "")).startswith("[error"):
                    done.add(row["id"])
        # Rewrite keeping only the good rows (drops error/torn lines), then append.
        good = [json.loads(l) for l in open(out_path) if l.strip()
                and json.loads(l)["id"] in done]
        with open(out_path, "w") as f:
            for row in good:
                f.write(json.dumps(row) + "\n")
    todo = [it for it in items if it["id"] not in done]
    if done:
        print(f"  [export] resuming: {len(done)} already done, {len(todo)} to go")

    # Write incrementally (one JSON line per item, flushed) so a killed run — e.g.
    # a co-tenant restarting a shared backend, or a slow CPU run interrupted — keeps
    # every response computed so far instead of losing the whole buffer.
    n = n_auto = need = 0
    with open(out_path, "a") as f:
        for item, text, latency_ms, hits in generate_financebench(
            provider, todo, max_tokens=max_tokens, retrieve=retrieve
        ):
            gnum = _gold_number(item["gold"]["answer"])
            auto_ok = gnum is not None and _match_value_unit_agnostic(text, gnum, tol=0.01)
            n += 1
            n_auto += int(auto_ok)
            need += int(not auto_ok)
            f.write(json.dumps({
                "id": item["id"],
                "question_type": item["meta"].get("question_type"),
                "company": item["meta"].get("company"),
                "doc_id": item.get("doc_id"),
                "question": item["question"],
                "gold": item["gold"]["answer"],
                "response": text,
                "n_hits": len(hits),
                "auto_numeric_ok": auto_ok,
                "needs_judge": not auto_ok,
                "latency_ms": latency_ms,
            }) + "\n")
            f.flush()
    print(f"\nexported {n} responses ({mode}) -> {out_path}")
    print(f"  auto numeric-correct: {n_auto}/{n}  |  needs Claude judge: {need}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--request-timeout", type=float, default=120.0,
                    help="per-request HTTP timeout (s). Raise for reasoning models whose "
                    "long reasoning_content at a high --max-tokens exceeds the 120s default.")
    ap.add_argument("--limit", type=int, help="run only the first N (quick check)")
    ap.add_argument("--judge-url", help="OpenAI-/v1 base URL for an LLM judge")
    ap.add_argument("--judge-model", help="judge model id")
    ap.add_argument("--cache", default=str(CACHE))
    # --- full-RAG (T26) ---
    ap.add_argument("--rag", action="store_true", help="full-RAG mode (live retrieval)")
    ap.add_argument("--gap", action="store_true", help="with --rag, also run oracle and print the gap")
    ap.add_argument("--docs-dir", help="folder of <doc_id>.pdf filings (required for --rag)")
    ap.add_argument("--embed-url", help="/v1 base serving the finance embedder")
    ap.add_argument("--embed-model", default="bge-m3:latest")
    ap.add_argument("--rerank-url", help="/v1 base serving the cross-encoder reranker")
    ap.add_argument("--rerank-model", default="bge-reranker-v2-m3:latest")
    ap.add_argument("--retrieval-k", type=int, default=6, help="chunks fed as context in full-RAG")
    ap.add_argument("--corpus-cache", help="reuse the Phase-A sweep's per-filing embedded corpora "
                    "(dir, e.g. benchmarks/financebench/.sweep_cache) instead of re-embedding")
    ap.add_argument("--dump-contexts", help="with --rag: retrieve every item ONCE and write the "
                    "hits to this JSONL (no generation), so generators can reuse them via --contexts")
    ap.add_argument("--contexts", help="generate/score using contexts previously dumped by "
                    "--dump-contexts (full-RAG without a live embedder/reranker resident)")
    ap.add_argument("--export", help="write per-item responses to this JSONL for out-of-band "
                    "(Claude) judging, instead of scoring with a small-model judge")
    args = ap.parse_args()

    if args.rag and not args.docs_dir:
        ap.error("--rag requires --docs-dir (folder of <doc_id>.pdf filings)")
    if args.dump_contexts and not args.rag:
        ap.error("--dump-contexts requires --rag (it caches live retrieval output)")
    if args.contexts and args.rag:
        ap.error("--contexts replays cached retrieval; do not combine with --rag")

    records = load_financebench(args.cache)
    items = [fb_to_item(r) for r in records]
    if args.limit:
        items = items[: args.limit]
    print(f"loaded {len(items)} FinanceBench items")

    provider = OpenAICompatProvider(name=args.label or args.model,
                                    base_url=args.base_url, chat_model=args.model,
                                    timeout_s=args.request_timeout)
    # Under the watchdog, survive a mid-run backend restart instead of emitting
    # [error] rows (the generator run is long; a transient disconnect shouldn't cost a row).
    provider = _ResilientProvider(provider, args.base_url)
    judge = None
    if args.judge_url and args.judge_model:
        judge = OpenAICompatProvider(name="judge", base_url=args.judge_url,
                                     chat_model=args.judge_model)

    # Context source: gold evidence (oracle) or live per-filing retrieval (full-RAG).
    retrieve = None
    mode = "oracle-evidence"
    if args.rag:
        mode = "full-RAG"
        s = Settings(
            backend_url=args.base_url,
            embed_url=args.embed_url or args.base_url,
            embed_model=args.embed_model,
            rerank_url=args.rerank_url,
            rerank_model=args.rerank_model,
        )
        retriever_for = _build_retriever_for(args.docs_dir, s, corpus_cache=args.corpus_cache)
        retrieve = make_rag_retrieve(retriever_for, k=args.retrieval_k)
        if args.dump_contexts:
            _dump_contexts(items, retrieve, args.dump_contexts, base_url=args.base_url)
            return
    elif args.contexts:
        mode = "full-RAG (cached ctx)"
        retrieve = _cached_retrieve(args.contexts)

    if args.export:
        _export_responses(provider, items, args.export, mode,
                          max_tokens=args.max_tokens, retrieve=retrieve)
        return

    card = run_financebench(provider, items, judge=judge,
                            max_tokens=args.max_tokens, retrieve=retrieve)
    _print_card(card, items, mode)

    if args.rag and args.gap:
        oracle_card = run_financebench(provider, items, judge=judge, max_tokens=args.max_tokens)
        _print_card(oracle_card, items, "oracle-evidence")
        delta = oracle_card.accuracy - card.accuracy
        print(f"\n  retrieval gap vs oracle ceiling: {delta:+.1%} "
              f"(oracle {oracle_card.accuracy:.1%} − full-RAG {card.accuracy:.1%})")


if __name__ == "__main__":
    main()
