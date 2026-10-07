#!/usr/bin/env python
"""Run the REAL FinanceBench open subset (150 Q) against a candidate.

Two modes, different generation paths:

- **oracle-evidence** (default): each question is handed its gold evidence, so we
  measure the generator in isolation — the honest "how good is this model at
  grounded financial QA" ceiling, before retrieval enters the picture. Stays on
  the direct grounded-prompt path (`agent.graph._grounded_messages`) by design:
  it deliberately isolates the generator from the aggregate router.

    uv run python benchmarks/financebench/run_full.py \
        --base-url http://127.0.0.1:11434/v1 --model qwen35-4b-distill \
        --max-tokens 2048 --limit 20            # quick slice

- **full-RAG** (`--rag`): builds a corpus from each filing's PDF, ingests it
  into a dedicated engine collection (`--collection`, default `financebench` —
  never the production `docket` collection), and retrieves live via the
  engine's hybrid search + rerank (mettle; retrieval lives engine-side — see
  the cutover note in README.md). Routes every question through the REAL
  production `agent.graph.answer()` path (the aggregate router included), not
  the oracle mode's direct-prompt shortcut — batched two-pass (retrieve
  everything, then generate everything) so a phase-switching engine pays the
  ~2s retrieval<->chat profile switch once for the whole run, not once per
  question (see `PrefetchedRetriever`/`run_rag_via_answer` in
  `docket/eval/financebench.py`). Needs the source filings via `--docs-dir`
  (one `<doc_name>.pdf` per question's `doc_id`) and the ingest extra
  (`uv sync --extra ingest`). `--rerank-model` selects the engine's reranker
  (embeddings are the collection's engine-side default — docket's own
  `embed_model` setting only affects `dk embed`, not engine-side search). Add
  `--gap` to also run oracle and print the retrieval gap vs the ceiling.
  `--contexts` (previously-dumped hits, see `--dump-contexts` below) also now
  routes through `answer()`, with no live engine to prefetch against.

`--max-tokens` only bounds generation on the oracle-evidence (direct) path;
full-RAG (`--rag`/`--contexts`) runs through `agent.graph.answer()`, which uses
its own fixed per-call generation budgets.

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
    PrefetchedRetriever,
    Scorecard,
    _gold_number,
    _match_value_unit_agnostic,
    fb_to_item,
    generate_financebench,
    generate_via_answer,
    load_financebench,
    make_rag_retrieve,
    run_financebench,
    run_rag_via_answer,
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


def _build_retriever_for(docs_dir: str, settings: Settings, *, verbose: bool = True):
    """Return a cached ``doc_id -> Retriever`` lookup for full-RAG.

    Each filing is ingested into the engine's collection via the production
    ingest pipeline and retrieved through the engine, scoped to that filing's
    chunks via the search allowlist (post-cutover: search lives engine-side;
    FinanceBench questions target a single filing). A missing PDF yields
    ``None`` (→ empty context), reported rather than silently faked."""
    from docket.ingest.index import Corpus
    from docket.ingest.pipeline import ingest_paths
    from docket.service import load_retriever

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
        corpus = ingest_paths([str(pdf)], settings=settings, corpus=Corpus())
        retriever = load_retriever(settings, corpus=corpus, scope_doc=doc_id)
        if verbose:
            print(f"  [full-RAG] {doc_id}: {len(corpus)} chunks")
        # Cache on success too (the bug: this branch never cached, so every
        # question against the SAME filing re-OCR'd and re-POSTed it from
        # scratch -- expensive, and a duplicate post of the same chunk ids
        # 409s against the engine).
        cache[doc_id] = retriever
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


def _resilient(fn, what: str, *, tries: int = 8):
    """Retry a backend call through transient 5xx / transport blips (the
    engine may be restarting under us). A persistent failure re-raises so the
    row stays un-done for resume."""
    import httpx
    import time as _t

    last: Exception | None = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — classify below
            retryable = isinstance(e, (httpx.TransportError, httpx.HTTPStatusError))
            last = e
            if not retryable or attempt == tries - 1:
                raise
            wait = min(2.0 * (attempt + 1), 10.0)
            print(f"  [{what}] {type(e).__name__} (attempt {attempt + 1}); retry in {wait:.0f}s")
            _t.sleep(wait)
    raise last  # pragma: no cover


def _dump_contexts(items, retrieve, out_path: str) -> None:
    """Run retrieval ONCE for every item and persist the hits to JSONL.

    On a 6 GB card the generator, embedder and reranker don't all fit resident
    together, and the retrieved context is generator-independent — so we retrieve
    with the winning embedder+reranker served alone, cache the *real* hits here,
    then generate with each candidate served alone via ``--contexts`` (no faking:
    these are byte-for-byte the hits live retrieval produced). One retrieval pass
    feeds every generator instead of re-retrieving per model.

    Crash-recoverable: each item is retrieved through the ``_resilient``
    wrapper (retries while the supervisor brings the engine back), and completed
    ids are skipped on resume, so a backend death mid-run never loses progress
    or restarts from zero."""
    import json


    done = _done_ids(out_path)
    todo = [it for it in items if it["id"] not in done]
    if done:
        print(f"  [dump-contexts] resuming: {len(done)} already done, {len(todo)} to go")
    n = len(done)
    with open(out_path, "a") as f:
        for item in todo:
            hits = _resilient(lambda: retrieve(item), f"retrieve {item['id']}")
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


class _StaticRetriever:
    """Adapter so ``--contexts`` replay (cached retrieval hits from a prior
    ``--dump-contexts`` run — Phase B's decoupled generation, no live
    embedder/reranker resident) can still drive the REAL
    ``agent.graph.answer()`` path: every call returns the SAME previously
    dumped hits (truncated to ``k``) — honest about there being no fresh
    retrieval available, never fabricating one. ``prefetch`` is a harmless
    no-op (there is nothing live to pre-run; the hits are already on disk)."""

    def __init__(self, hits: list[dict]) -> None:
        self._hits = hits

    def prefetch(self, query: str, k: int) -> list[dict]:
        return self.retrieve(query, k=k)

    def retrieve(self, query: str, *, k: int = 10, candidates: int | None = None) -> list[dict]:
        return self._hits[:k]

    def iterative_retrieve(self, queries, *, k: int = 10) -> list[dict]:
        return self._hits[:k]


def _cached_retriever_of(path: str):
    """Build a ``retriever_of(item) -> _StaticRetriever`` for the router path
    from ``--contexts``-dumped hits. Reuses ``_cached_retrieve``'s cache
    loading (one concept, one implementation) instead of re-parsing the
    dump file a second way."""
    retrieve = _cached_retrieve(path)
    return lambda item: _StaticRetriever(retrieve(item))


class _ResilientProvider:
    """Wrap a provider so ``.chat`` survives a backend restart: on a disconnect it
    retries while the supervisor restarts the engine, instead of yielding an
    ``[error]`` row. Transparent for everything else (name, etc.)."""

    def __init__(self, inner):
        self._inner = inner
        self.name = getattr(inner, "name", "candidate")

    def __getattr__(self, key):
        return getattr(self._inner, key)

    def chat(self, *args, **kwargs):
        return _resilient(lambda: self._inner.chat(*args, **kwargs), "chat")


def _export_responses(provider, items, out_path: str, mode: str, *,
                      max_tokens: int, retrieve=None, retriever_of=None,
                      settings: Settings | None = None) -> None:
    """Generate answers and write a JSONL for out-of-band (Claude) judging.

    Each row carries the question, gold answer, model response, and an
    ``auto_numeric_ok`` flag from the deterministic numeric matcher — so only the
    rows the matcher can't resolve (``needs_judge``) actually need a human/LLM
    verdict. Keeps the numeric path deterministic while letting Claude grade the
    free-form remainder without loading a second model on a 6 GB box.

    ``retriever_of`` (optional), when given, routes generation through the REAL
    production ``agent.graph.answer()`` path (:func:`generate_via_answer`)
    instead of the direct-prompt :func:`generate_financebench` (``retrieve``);
    the two are mutually exclusive callers (oracle-evidence/legacy-direct vs
    full-RAG-via-router) — see ``main()``."""
    import json
    import os

    # Resume: skip items already generated with a real (non-error) response, so a
    # killed/crashed run continues instead of restarting. Error rows are re-done.
    # A single pass builds BOTH `done` and the rewritten `good` rows -- the old
    # code re-read the file a SECOND time for `good` without the torn-line
    # guard the first pass already had, so a torn last line (the exact case
    # this comment claims to tolerate) raised instead of being dropped, and
    # `open(out_path)` there was never closed.
    done: set = set()
    good: list[dict] = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # drop a torn last line, as documented
                if not str(row.get("response", "")).startswith("[error"):
                    done.add(row["id"])
                    good.append(row)
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
    gen = (
        generate_via_answer(todo, retriever_of=retriever_of, provider=provider, settings=settings)
        if retriever_of is not None
        else generate_financebench(provider, todo, max_tokens=max_tokens, retrieve=retrieve)
    )
    with open(out_path, "a") as f:
        for item, text, latency_ms, hits in gen:
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
    ap.add_argument("--collection", default="financebench",
                    help="engine collection to ingest FinanceBench filings into for --rag "
                    "(a dedicated collection, never the production 'docket' collection)")
    ap.add_argument("--rerank-model", default="qwen3-reranker-0.6b:latest")
    ap.add_argument("--retrieval-k", type=int, default=6,
                    help="chunks fed as context in full-RAG (wired through as "
                    "Settings.context_chunks for the --rag router path)")
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
    provider = _ResilientProvider(provider)
    judge = None
    if args.judge_url and args.judge_model:
        judge = OpenAICompatProvider(name="judge", base_url=args.judge_url,
                                     chat_model=args.judge_model)

    # Generation path: oracle-evidence stays on the direct grounded-prompt
    # shortcut (`retriever_of`/`settings_for_router` stay None); full-RAG
    # (`--rag` or `--contexts`) routes through the REAL `agent.graph.answer()`
    # path via `retriever_of` -- see `generate_via_answer`/`run_rag_via_answer`
    # in docket/eval/financebench.py.
    retriever_of = None
    settings_for_router: Settings | None = None
    mode = "oracle-evidence"
    if args.rag:
        mode = "full-RAG"
        s = Settings(
            backend_url=args.base_url,
            rerank_model=args.rerank_model,
            engine_collection=args.collection,
            context_chunks=args.retrieval_k,
        )
        retriever_for = _build_retriever_for(args.docs_dir, s)
        if args.dump_contexts:
            # Retrieval-only: no generation, so the router split doesn't apply.
            _dump_contexts(items, make_rag_retrieve(retriever_for, k=args.retrieval_k),
                           args.dump_contexts)
            return

        wrapped: dict[str, object] = {}

        def retriever_of(item, _retriever_for=retriever_for, _wrapped=wrapped):
            doc_id = item.get("doc_id", "DOC")
            if doc_id not in _wrapped:
                base = _retriever_for(doc_id)
                _wrapped[doc_id] = PrefetchedRetriever(base) if base is not None else None
            return _wrapped[doc_id]

        settings_for_router = s
    elif args.contexts:
        mode = "full-RAG (cached ctx)"
        retriever_of = _cached_retriever_of(args.contexts)
        settings_for_router = Settings()

    if args.export:
        if retriever_of is not None:
            _export_responses(provider, items, args.export, mode, max_tokens=args.max_tokens,
                              retriever_of=retriever_of, settings=settings_for_router)
        else:
            _export_responses(provider, items, args.export, mode, max_tokens=args.max_tokens)
        return

    if retriever_of is not None:
        card = run_rag_via_answer(items, retriever_of=retriever_of, provider=provider,
                                  settings=settings_for_router, judge=judge)
    else:
        card = run_financebench(provider, items, judge=judge, max_tokens=args.max_tokens)
    _print_card(card, items, mode)

    if args.rag and args.gap:
        oracle_card = run_financebench(provider, items, judge=judge, max_tokens=args.max_tokens)
        _print_card(oracle_card, items, "oracle-evidence")
        delta = oracle_card.accuracy - card.accuracy
        print(f"\n  retrieval gap vs oracle ceiling: {delta:+.1%} "
              f"(oracle {oracle_card.accuracy:.1%} − full-RAG {card.accuracy:.1%})")


if __name__ == "__main__":
    main()
