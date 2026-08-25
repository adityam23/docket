#!/usr/bin/env python
"""Phase-A retrieval sweep: gold-evidence **recall@k** across embedder × reranker.

FinanceBench is retrieval-bound (the paper: basic RAG ~19% vs an oracle ceiling),
so before spending generator tokens we optimise the retriever directly against a
deterministic, generator-free signal: *does the retrieved context contain the
question's gold evidence span?* This isolates the retrieval lever (T26) and ranks
configs cheaply — only the winners go to end-to-end generation (Phase B).

Reuses the production stack verbatim — ``ingest_paths`` (chunk+embed), the
``Retriever`` (hybrid BM25+dense → RRF → rerank), and ``make_reranker`` — so a
config that wins here is the config that ships. No parallel retrieval path.

**Recoverable by design** (a co-tenant may restart the shared infengine backend
mid-run): every embedded corpus is cached to ``--cache-dir`` (reloaded instantly
on restart — the expensive embedding work is never redone), results are
checkpointed to ``--out`` after each config, and embed/rerank calls wait for the
backend to come back and retry rather than crash. Re-invoking with the same args
resumes: cached corpora + already-written configs are skipped.

    uv run python -u benchmarks/financebench/sweep_recall.py \
        --base-url http://127.0.0.1:11434/v1 \
        --docs-dir benchmarks/financebench/filings \
        --out benchmarks/financebench/results/recall_sweep.json

Recall is a *proxy* (token overlap-coefficient vs the gold evidence_text, not an
answer check) — honest about that: it ranks retrieval quality, it is not accuracy.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from docket.config import Settings  # noqa: E402
from docket.ingest.chunk import chunk_pages  # noqa: E402
from docket.ingest.embed import embed_texts  # noqa: E402
from docket.ingest.index import Corpus  # noqa: E402
from docket.ingest.ocr import pdf_to_pages  # noqa: E402
from docket.retrieval.rerank_client import make_reranker  # noqa: E402
from docket.retrieval.retriever import Retriever, tokenize  # noqa: E402

# pypdf extraction leaves control chars (esp. NUL \x00) in the text, and embed-gemma
# 500s on them. Strip control chars (keep \t\n\r) and cap length for the *embed
# input* — the stored chunk text stays intact for BM25 + recall scoring.
_EMBED_CHAR_CAP = 2000
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _embed_clean(t: str) -> str:
    return _CTRL.sub(" ", t)[:_EMBED_CHAR_CAP]

CACHE = Path(__file__).with_name("data") / "financebench_merged.jsonl"


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _wait_backend(base_url: str, timeout: float = 600.0) -> bool:
    """Block until the backend answers ``/v1/models`` (a co-tenant may be
    restarting it). Returns False if it never comes back within ``timeout``."""
    url = base_url.rstrip("/") + "/models"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:  # noqa: S310
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            pass
        print("    [backend] waiting for infengine to come back…", flush=True)
        time.sleep(5)
    return False


def _resilient(fn, base_url: str, what: str, tries: int = 6):
    """Run ``fn`` (an embed/rerank call), surviving a backend that goes away: on
    failure, wait for it to return, then retry with backoff. Raises after tries."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            print(f"    {what} attempt {attempt+1}/{tries} failed: {str(e)[:90]}", flush=True)
            _wait_backend(base_url)
            time.sleep(min(2 * (attempt + 1), 20))
    raise RuntimeError(f"{what}: exhausted {tries} retries")


def _gold_evidence(rec: dict) -> list[set]:
    out = []
    for e in rec.get("evidence") or []:
        if isinstance(e, dict) and e.get("evidence_text"):
            toks = set(tokenize(e["evidence_text"]))
            if toks:
                out.append(toks)
    return out


def _overlap_coeff(a: set, b: set) -> float:
    """|a∩b| / min(|a|,|b|) — size-robust: a chunk fully inside a big evidence
    table still scores 1.0, so table-shaped evidence isn't unfairly penalised."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _recall_hit(hit_token_sets: list[set], golds: list[set], thresh: float) -> bool:
    for cs in hit_token_sets:
        for g in golds:
            if _overlap_coeff(cs, g) >= thresh:
                return True
    return False


def _embed_chunks(texts: list[str], s: Settings, base_url: str) -> list[list[float] | None]:
    """Embed chunk texts, robust to BOTH a backend that goes away AND a single
    poison chunk that 500s the model.

    A batch failure is disambiguated: if the backend is unreachable we wait for it
    and retry the same batch (never drop data); if it's up (so the 500 is
    content-driven) we bisect down to the offending chunk and skip just that one
    (``None`` vector). Inputs are capped to ``_EMBED_CHAR_CAP`` up front to avoid
    the common over-long case entirely."""
    capped = [_embed_clean(t) for t in texts]
    out: list[list[float] | None] = [None] * len(capped)

    def do(lo: int, hi: int) -> None:
        # Retry the SAME batch first: on a shared box the 500s are mostly transient
        # (infengine swapping models under a co-tenant's load) and clear in <1s, so
        # fast retries absorb them cheaply. Only escalate to a backend-wait if they
        # persist; only bisect+skip a size-1 batch that still fails (poison chunk).
        last = ""
        for attempt in range(10):
            try:
                vecs = embed_texts(capped[lo:hi], settings=s)
                for j, v in enumerate(vecs):
                    out[lo + j] = v
                return
            except Exception as e:  # noqa: BLE001
                last = str(e)[:60]
                if attempt < 5:
                    time.sleep(0.3)                      # cheap fast-retry for swaps
                else:
                    _wait_backend(base_url, timeout=10)  # persistent → maybe a restart
                    time.sleep(min(1.0 * (attempt - 4), 5))
        if hi - lo <= 1:
            print(f"    embed: skipping persistently-failing chunk {lo} ({last})", flush=True)
            out[lo] = None
            return
        mid = (lo + hi) // 2
        do(lo, mid)
        do(mid, hi)

    for i in range(0, len(capped), 64):
        do(i, min(i + 64, len(capped)))
    return out


def _cached_corpus(doc: str, pdf: Path, s: Settings, cache_dir: Path, embedder: str) -> Corpus:
    """Chunk+embed a filing once, cached under ``cache_dir/<embedder>/<doc>``.

    Chunking is the production ``chunk_pages``; embedding uses the resilient helper
    so one bad chunk (or a backend restart) can't abort the filing. The expensive
    part (vectors) is persisted, so a restart only costs re-loading JSONL — never
    re-embedding. Reused across reranker/k."""
    path = cache_dir / _slug(embedder) / doc
    cached = Corpus.load(str(path))
    if len(cached) and cached.has_vectors:
        return cached
    # Retrieval sweep: FinanceBench gold evidence is born-digital text, so a stray
    # image page (chart/signature) must not drop the whole filing — skip such pages.
    pages = pdf_to_pages(str(pdf), on_missing_text="skip")
    chunks = chunk_pages(pages, doc_id=doc, source=str(pdf),
                         words=s.chunk_words, overlap=s.chunk_overlap)
    vectors = _embed_chunks([c.text for c in chunks], s, base_url=s.embed_url)
    # drop chunks whose embed was skipped (keep dense space clean; rare)
    keep = [(c, v) for c, v in zip(chunks, vectors) if v is not None]
    corpus = Corpus()
    if keep:
        corpus.add([c for c, _ in keep], [v for _, v in keep])
    corpus.save(str(path))
    return corpus


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    ap.add_argument("--docs-dir", default=str(Path(__file__).with_name("filings")))
    ap.add_argument("--cache-dir", default=str(Path(__file__).with_name(".sweep_cache")))
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--out", default=str(Path(__file__).with_name("results") / "recall_sweep.json"))
    ap.add_argument("--limit", type=int, help="use only the first N questions")
    ap.add_argument("--candidates", type=int, default=40)
    ap.add_argument("--ks", default="3,5,6,10,20")
    ap.add_argument("--thresholds", default="0.5,0.8")
    ap.add_argument("--embedders", default="embed-gemma:latest,bge-m3:latest,qwen3-embedding-0.6b:latest")
    ap.add_argument("--rerankers", default="none,bge-reranker-v2-m3:latest,qwen3-reranker-0.6b:latest")
    ap.add_argument("--chunk-words", type=int, default=220)
    ap.add_argument("--chunk-overlap", type=int, default=40)
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    thresholds = [float(x) for x in args.thresholds.split(",")]
    embedders = [e.strip() for e in args.embedders.split(",") if e.strip()]
    rerankers = [r.strip() for r in args.rerankers.split(",") if r.strip()]
    docs_dir = Path(args.docs_dir)
    cache_dir = Path(args.cache_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in open(args.cache) if l.strip()]
    if args.limit:
        records = records[: args.limit]

    present, missing = [], set()
    for r in records:
        if (docs_dir / f"{r['doc_name']}.pdf").is_file():
            present.append(r)
        else:
            missing.add(r["doc_name"])
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in present:
        by_doc[r["doc_name"]].append(r)
    print(f"{len(present)}/{len(records)} questions have their filing "
          f"({len(by_doc)} filings). missing filings: {len(missing)}", flush=True)

    golds = {r["financebench_id"]: _gold_evidence(r) for r in present}
    questions = {r["financebench_id"]: r["question"] for r in present}

    # Resume: keep result rows already written (same embedder+reranker key).
    done_keys: set[tuple] = set()
    results: list[dict] = []
    # A reranker whose model can't be loaded in the available VRAM (deterministic
    # CUDA OOM — it crashes the shared backend, and retrying just thrashes it all
    # night) is recorded here and skipped for EVERY embedder and every resume. This
    # is the one thing a wait-and-retry loop must NOT retry: a repeatable OOM.
    oom_rerankers: set[str] = set()
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            results = prev.get("results", [])
            done_keys = {(r["embedder"], r["reranker"]) for r in results}
            oom_rerankers = set(prev.get("oom_rerankers", []))
            if results:
                print(f"resuming: {len(results)} configs already done", flush=True)
            if oom_rerankers:
                print(f"skipping known-OOM rerankers: {', '.join(sorted(oom_rerankers))}", flush=True)
        except Exception:  # noqa: BLE001
            pass

    skipped_filings: set[str] = set()  # unreadable PDFs dropped mid-run (disclosed, not hidden)

    def checkpoint():
        out_path.write_text(json.dumps(
            {"n_questions": len(present), "missing_filings": sorted(missing),
             "skipped_filings": sorted(skipped_filings),
             "oom_rerankers": sorted(oom_rerankers),
             "ks": ks, "thresholds": thresholds, "results": results}, indent=2))

    for embedder in embedders:
        if all((embedder, rr) in done_keys for rr in rerankers):
            print(f"[{embedder}] all rerankers done — skipping", flush=True)
            continue
        s = Settings(backend_url=args.base_url, embed_url=args.base_url, embed_model=embedder,
                     chunk_words=args.chunk_words, chunk_overlap=args.chunk_overlap)
        t0 = time.perf_counter()
        corpus_by_doc: dict[str, Corpus] = {}
        failed_docs: list[str] = []
        n_chunks = 0
        for doc in by_doc:
            try:
                corpus = _cached_corpus(
                    doc, docs_dir / f"{doc}.pdf", s, cache_dir, embedder)
            except Exception as e:  # noqa: BLE001 — one unreadable PDF must not kill the run
                failed_docs.append(doc)
                print(f"    [skip] {doc}: {type(e).__name__}: {str(e)[:100]}", flush=True)
                continue
            if not len(corpus):
                failed_docs.append(doc)
                print(f"    [skip] {doc}: no usable chunks", flush=True)
                continue
            corpus_by_doc[doc] = corpus
            n_chunks += len(corpus)
        if failed_docs:
            skipped_filings.update(failed_docs)
            print(f"[{embedder}] skipped {len(failed_docs)} unreadable filings: "
                  f"{', '.join(sorted(failed_docs))}", flush=True)
        qids = [r["financebench_id"] for r in present if r["doc_name"] in corpus_by_doc]
        qvecs = _resilient(lambda: embed_texts([questions[i] for i in qids], settings=s),
                           s.embed_url, "query-embed")
        qvec = dict(zip(qids, qvecs))
        print(f"\n[{embedder}] {n_chunks} chunks over {len(by_doc)} filings, "
              f"ingest+embed {round(time.perf_counter()-t0,1)}s", flush=True)

        for rr_name in rerankers:
            if (embedder, rr_name) in done_keys:
                continue
            if rr_name in oom_rerankers:
                print(f"  [{rr_name:<26}] skipped — known to OOM this GPU (see oom_rerankers)", flush=True)
                continue
            try:
                # rerank_model is a required str in Settings; for the "none" arm the
                # reranker is disabled (rerank_url=None → base_reranker=None), so leave
                # rerank_model at its default rather than passing None (pydantic rejects it).
                rs = Settings(backend_url=args.base_url, embed_url=args.base_url, embed_model=embedder,
                              **({} if rr_name == "none"
                                 else {"rerank_url": args.base_url, "rerank_model": rr_name}))
                base_reranker = None if rr_name == "none" else make_reranker(rs)
                reranker = None
                if base_reranker is not None:
                    # On this 6 GB GPU the reranker model must be swapped in (evicting
                    # the resident embedder). Force that swap ONCE here, patiently, so
                    # the per-query loop below doesn't race a mid-load /v1/rerank (which
                    # errors while /v1/models already answers 200). Then wrap every call
                    # so a later backend blip is still survived.
                    #
                    # But a trivial 2-doc warmup that STILL can't complete after a bounded
                    # number of tries is not a transient blip — it means the model doesn't
                    # fit in VRAM (CUDA OOM crashes the backend on load). Record it as
                    # OOM-poison and skip it everywhere, rather than retrying into an
                    # all-night crash loop with the watchdog.
                    try:
                        _resilient(lambda: base_reranker("warmup", ["alpha", "beta"]),
                                   args.base_url, f"rerank-warmup({rr_name})", tries=8)
                    except RuntimeError:
                        oom_rerankers.add(rr_name)
                        checkpoint()
                        print(f"  [{rr_name:<26}] cannot load (backend OOM on this GPU) — "
                              f"recorded as oom, permanently skipped", flush=True)
                        continue
                    reranker = lambda q, docs: _resilient(
                        lambda: base_reranker(q, docs), args.base_url, f"rerank({rr_name})", tries=15)
                hits_at = {(k, t): 0 for k in ks for t in thresholds}
                best_overlap_sum = 0.0
                n_scored = 0
                t1 = time.perf_counter()
                for doc, recs in by_doc.items():
                    if doc not in corpus_by_doc:
                        continue
                    r = Retriever(corpus_by_doc[doc], embed_query=lambda q: None, reranker=reranker)
                    for rec in recs:
                        qid = rec["financebench_id"]
                        if qid not in qvec:
                            continue
                        n_scored += 1
                        r.embed_query = (lambda _q, _qid=qid: qvec[_qid])
                        hits = r.retrieve(questions[qid], k=max(ks), candidates=args.candidates)
                        hit_sets = [set(tokenize(h["text"])) for h in hits]
                        g = golds[qid]
                        if g:
                            best_overlap_sum += max(
                                (_overlap_coeff(cs, gg) for cs in hit_sets for gg in g), default=0.0)
                        for k in ks:
                            for t in thresholds:
                                if _recall_hit(hit_sets[:k], g, t):
                                    hits_at[(k, t)] += 1
                n = n_scored or 1
                row = {
                    "embedder": embedder, "reranker": rr_name, "n": n_scored, "chunks": n_chunks,
                    "mean_best_overlap": round(best_overlap_sum / n, 4),
                    "recall": {f"@{k}_t{t}": round(hits_at[(k, t)] / n, 4) for k in ks for t in thresholds},
                    "rerank_secs": round(time.perf_counter() - t1, 1),
                }
            except Exception as e:  # noqa: BLE001 — one flaky reranker config must not kill the sweep
                # Not added to done_keys → a later resume retries it when the backend is calmer.
                print(f"  [{rr_name:<26}] FAILED — skipping, will retry on resume: "
                      f"{type(e).__name__}: {str(e)[:120]}", flush=True)
                continue
            results.append(row)
            done_keys.add((embedder, rr_name))
            checkpoint()  # durable after every config
            head = "  ".join(f"@{k}={row['recall'][f'@{k}_t{thresholds[0]}']:.2f}" for k in ks)
            print(f"  [{rr_name:<26}] recall(t{thresholds[0]}) {head}  "
                  f"meanovl={row['mean_best_overlap']:.3f}  ({row['rerank_secs']}s)", flush=True)

    checkpoint()
    print(f"\nwrote {out_path}", flush=True)
    if results:
        kk = 6 if 6 in ks else ks[0]
        best = max(results, key=lambda r: r["recall"][f"@{kk}_t{thresholds[0]}"])
        print(f"best @{kk}(t{thresholds[0]}): {best['embedder']} + {best['reranker']} "
              f"= {best['recall'][f'@{kk}_t{thresholds[0]}']:.1%}", flush=True)


if __name__ == "__main__":
    main()
