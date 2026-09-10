#!/usr/bin/env python
"""DuckDB-path (exact cosine) vs mettle turbovec (ANN) — real FinanceBench.

Holds EVERYTHING constant except the index: same docket chunks, same
embed-gemma embeddings, same gold queries, same per-filing scoping, same
deterministic recall metric. Only the vector index differs:

  - "duckdb/exact": docket ``Corpus`` (brute-force cosine over in-memory
    vectors — the production lite path; DuckDB is the store swap, the search
    algorithm is identical exact similarity).
  - "turbovec": a mettle ``/v1/collections/{name}`` with the SAME chunks
    POSTed as documents; the engine embeds them (embed-gemma) into a turbovec
    ``IdMapIndex`` and answers ``/search`` (dense, ANN).

Output: recall@k (t0.5/t0.8) + mean query latency + index size for both, so
the verdict (is ANN parity with exact, and is it faster/smaller?) is measured,
not assumed.

Runs as a bounded smoke by default (--limit questions, --max-filings); pass
--limit 0 --max-filings 0 for the full 150-Q set.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from docket.config import Settings  # noqa: E402
from docket.retrieval.retriever import tokenize  # noqa: E402
from sweep_recall import (  # noqa: E402
    CACHE,
    _cached_corpus,
    _embed_clean,
    _gold_evidence,
    _overlap_coeff,
    _recall_hit,
)

BASE = "http://127.0.0.1:11434/v1"
EMBEDDER = "embed-gemma:latest"
KS = (3, 5, 6, 10)
THRESHOLDS = (0.5, 0.8)


def _turbovec_search(collection: str, query: str, k: int) -> list[dict]:
    r = httpx.post(
        f"{BASE}/collections/{collection}/search",
        json={"query": query, "k": k, "mode": "dense"},
        timeout=120.0,
    )
    r.raise_for_status()
    return r.json()


def _ensure_collection(collection: str) -> None:
    httpx.delete(f"{BASE}/collections/{collection}", timeout=30.0)
    r = httpx.post(
        f"{BASE}/collections",
        json={"name": collection, "embedding_model": EMBEDDER},
        timeout=30.0,
    )
    # 201 created (it did). 409 = already exists (race) → still fine.
    assert r.status_code in (201, 409), f"create {collection}: {r.status_code} {r.text[:200]}"


def _ingest_chunks(collection: str, doc: str, chunks: list[str]) -> list[str]:
    """POST each chunk as one document; the engine keeps a <512-token chunk as a
    single chunk. Returns the document ids (== chunk ids) for the allowlist."""
    ids = []
    for i, text in enumerate(chunks):
        cid = f"{doc}__{i}"
        r = httpx.post(
            f"{BASE}/collections/{collection}/documents",
            json={"id": cid, "content": _embed_clean(text), "metadata": {"doc_id": doc}},
            timeout=120.0,
        )
        r.raise_for_status()
        ids.append(cid)
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=24, help="questions to score (0 = all)")
    ap.add_argument("--max-filings", type=int, default=6, help="filings to index (0 = all)")
    ap.add_argument("--docs-dir", default="filings")
    ap.add_argument("--cache-dir", default=".sweep_cache")
    ap.add_argument("--candidates", type=int, default=40)
    args = ap.parse_args()

    docs_dir = Path(args.docs_dir)
    cache_dir = Path(args.cache_dir)

    records = [json.loads(l) for l in open(CACHE) if l.strip()]
    # Keep only records with a present filing and a gold evidence set.
    recs = []
    for r in records:
        pdf = docs_dir / f"{r['doc_name']}.pdf"
        if pdf.is_file() and _gold_evidence(r):
            recs.append(r)
    if args.limit:
        recs = recs[: args.limit]

    by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_doc[r["doc_name"]].append(r)
    docs = sorted(by_doc)
    if args.max_filings:
        docs = docs[: args.max_filings]
    n_docs = len(docs)
    qids = [r["financebench_id"] for d in docs for r in by_doc[d]]
    print(f"{len(qids)} questions across {n_docs} filings "
          f"({', '.join(docs[:10])}{'…' if n_docs > 10 else ''})", flush=True)

    s = Settings(backend_url=BASE, embed_url=BASE, embed_model=EMBEDDER,
                 chunk_words=220, chunk_overlap=40)

    # --- per-path accumulators ---
    hits_exact = {(k, t): 0 for k in KS for t in THRESHOLDS}
    hits_tv = {(k, t): 0 for k in KS for t in THRESHOLDS}
    lat_exact, lat_tv = [], []
    n_scored = 0
    bytes_exact = 0
    tv_measured = 0

    for doc in docs:
        # --- exact path: docket Corpus (cached) ---
        corpus = _cached_corpus(doc, docs_dir / f"{doc}.pdf", s, cache_dir, EMBEDDER)
        if not len(corpus):
            print(f"  [skip] {doc}: no chunks", flush=True)
            continue
        bytes_exact += sum(len(c.text) for c in corpus.chunks) * 4  # rough char→bytes
        # --- turbovec: per-filing collection with the SAME chunks ---
        tv = f"fb_{doc.lower()}"
        _ensure_collection(tv)
        chunk_texts = [c.text for c in corpus.chunks]
        doc_ids = _ingest_chunks(tv, doc, chunk_texts)
        print(f"[{doc}] {len(chunk_texts)} chunks ingested into {tv}", flush=True)
        # engine reports resident slot bytes as index memory proxy via /engine/status
        try:
            st = httpx.get("http://127.0.0.1:11434/engine/status", timeout=30.0).json()
            tv_measured = sum(sl.get("measured_bytes", 0) for sl in st.get("slots", []))
        except Exception:  # noqa: BLE001
            pass

        from docket.retrieval.retriever import Retriever
        retr = Retriever(corpus, embed_query=lambda q: None)

        for rec in by_doc[doc]:
            qid = rec["financebench_id"]
            q = rec["question"]
            golds = _gold_evidence(rec)
            if not golds:
                continue

            # exact: embed query, cosine over the corpus
            t0 = time.perf_counter()
            # embed via the backend so both paths use identical query vectors
            from docket.ingest.embed import embed_texts
            qv = embed_texts([q], settings=s)[0]
            retr.embed_query = (lambda _q, _qv=qv: _qv)
            hits = retr.retrieve(q, k=max(KS), candidates=args.candidates)
            lat_exact.append(time.perf_counter() - t0)
            hit_sets = [set(tokenize(h["text"])) for h in hits]
            for k in KS:
                for t in THRESHOLDS:
                    if _recall_hit(hit_sets[:k], golds, t):
                        hits_exact[(k, t)] += 1

            # turbovec: dense ANN over the same chunks, allowlist = this doc's ids
            t0 = time.perf_counter()
            tv_hits = _turbovec_search(tv, q, k=max(KS))
            lat_tv.append(time.perf_counter() - t0)
            tv_sets = [set(tokenize(h["text"])) for h in tv_hits]
            for k in KS:
                for t in THRESHOLDS:
                    if _recall_hit(tv_sets[:k], golds, t):
                        hits_tv[(k, t)] += 1
            n_scored += 1

    n = n_scored or 1
    def table(label, hits):
        row = {f"@{k}_t{t}": round(hits[(k, t)] / n, 4) for k in KS for t in THRESHOLDS}
        best = max(row, key=row.get)
        parts = "  ".join("@{k}t0.5={v:.2f}".format(k=k, v=row[("@%d_t0.5" % k)]) for k in KS)
        print("[{l}] recall {b}={best:.1%}  {parts}".format(
            l=label, b=best, best=row[best], parts=parts), flush=True)
        return row

    print(f"\n=== DuckDB/exact vs turbovec (real FinanceBench, n={n}) ===", flush=True)
    r_exact = table("duckdb/exact", hits_exact)
    r_tv = table("turbovec", hits_tv)

    ml = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
    print(f"mean query latency: exact={ml(lat_exact)*1000:.0f}ms  "
          f"turbovec={ml(lat_tv)*1000:.0f}ms", flush=True)
    print(f"index bytes: exact-cosine≈{bytes_exact//1024}KiB; "
          f"turbovec resident-bytes={tv_measured} (slot proxy)", flush=True)

    # verdict at the headline k (6, t0.5 — the harness's chosen operating point)
    k6 = ("@6_t0.5")
    exact6, tv6 = r_exact[k6], r_tv[k6]
    print(f"\nVERDICT @6_t0.5: exact={exact6:.3f}  turbovec={tv6:.3f}  "
          f"delta={tv6 - exact6:+.3f}  → "
          f"{'turbovec parity/wins' if tv6 >= exact6 - 0.01 else 'exact wins (ANN loses recall)'}",
          flush=True)


if __name__ == "__main__":
    main()
