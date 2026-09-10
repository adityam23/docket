#!/usr/bin/env python
"""Index smoke test: **duckdb exact cosine** vs **mettle/turbovec ANN** for dense
retrieval on the real FinanceBench corpus, with the **measured-best embedder
(qwen3-embedding)** + the winning combo reranker (qwen3-reranker). This answers
the question that gates docket's store choice: does the dense index need an ANN
(turbovec) or is exact cosine (duckdb) enough/better at this scale?

Method (index-isolated):
- One shared chunk set: docket's word-window chunks (220/40, page-exact) per
  filing, embedded with **qwen3-embedding-0.6b** on the running engine.
- **Path A (duckdb)**: the SAME chunks + vectors inserted into a duckdb table;
  per-question dense search = ``list_cosine_similarity`` exact cosine, scoped to
  the question's filing.
- **Path B (turbovec)**: the SAME chunk texts POSTed into a per-filing mettle
  collection (whose ``embedding_model`` = qwen3-embedding, so the engine embeds
  them with the same model); per-question search ``mode:"dense"`` = turbovec ANN.
- Both also get a **qwen3-reranker** row (the winning combo) over top-40.

Metric: gold-evidence recall@k (t=0.5/0.8), per-query latency, index on-disk
bytes. Reuses the proven helpers from ``sweep_recall`` so the scoring is
identical to the Phase-A sweep's.

Crash-resilient: waits for the backend, retries transient blips, checkpoints to
``--out`` after every path so a killed run resumes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import duckdb  # noqa: E402
import httpx  # noqa: E402

from docket.config import Settings  # noqa: E402
from docket.ingest.embed import embed_texts  # noqa: E402
from docket.retrieval.retriever import tokenize  # noqa: E402

from sweep_recall import (  # noqa: E402
    CACHE,
    _cached_corpus,
    _gold_evidence,
    _recall_hit,
    _resilient,
    _wait_backend,
)

_EMBEDDER = "qwen3-embedding-0.6b:latest"  # measured-best embedder (beats embed-gemma @10)
_RERANKER = "qwen3-reranker-0.6b:latest"
_KS = (3, 5, 6, 10, 20)
_THRESHOLDS = (0.5, 0.8)


# --- mettle /v1/collections client (docket-side; boundary respected) ----------


class MettleCollections:
    """Thin client for mettle's ``/v1/collections`` search surface. The engine
    owns chunking + embedding internally; we feed it the SAME chunk texts docket
    uses so the only difference between the two paths is the index (exact vs ANN)."""

    def __init__(self, base_url: str, bearer: str | None = None) -> None:
        self.base = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {bearer}"} if bearer else {}

    def _post(self, path: str, json_body: dict) -> httpx.Response:
        return httpx.post(f"{self.base}{path}", json=json_body, headers=self.headers, timeout=300.0)

    def ensure_collection(self, name: str, embedding_model: str) -> None:
        httpx.delete(f"{self.base}/v1/collections/{name}", headers=self.headers, timeout=60.0)
        r = self._post("/v1/collections", {"name": name, "embedding_model": embedding_model})
        if r.status_code not in (200, 201):  # noqa: PLR2004
            raise RuntimeError(f"create collection {name}: {r.status_code} {r.text[:200]}")

    def ingest(self, collection: str, doc_id: str, content: str) -> None:
        r = self._post(
            f"/v1/collections/{collection}/documents",
            {"id": doc_id, "content": content, "metadata": {}},
        )
        if r.status_code not in (200, 201):  # noqa: PLR2004
            raise RuntimeError(f"ingest {collection}/{doc_id}: {r.status_code} {r.text[:200]}")

    def search_dense(self, collection: str, query: str, k: int) -> tuple[list[dict], float]:
        """Return ``(hits, elapsed_s)`` for a ``mode:"dense"`` (turbovec ANN) search."""
        t0 = time.perf_counter()
        r = self._post(f"/v1/collections/{collection}/search", {"query": query, "k": k,
                                                               "mode": "dense"})
        elapsed = time.perf_counter() - t0
        if r.status_code != 200:  # noqa: PLR2004
            raise RuntimeError(f"search {collection}: {r.status_code} {r.text[:200]}")
        return r.json(), elapsed


# --- duckdb exact-cosine path -------------------------------------------------


class DuckDense:
    """Exact dense cosine over a duckdb table. Mirrors docket's future store:
    filter to a doc, ``list_cosine_similarity`` order, top-k."""

    def __init__(self, db_path: Path) -> None:
        self.con = duckdb.connect(str(db_path))
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS chunks("
            "doc_name VARCHAR, chunk_id VARCHAR, text VARCHAR, vector FLOAT[], "
            "PRIMARY KEY(chunk_id))"
        )

    def add(self, doc_name: str, chunks: list, vectors: list[list[float] | None]) -> None:
        rows = [
            (doc_name, ch.chunk_id, ch.text, vec) for ch, vec in zip(chunks, vectors) if vec is not None
        ]
        if rows:
            self.con.executemany("INSERT OR IGNORE INTO chunks VALUES (?, ?, ?, ?)", rows)

    def search(self, doc_name: str, qvec: list[float], k: int) -> tuple[list[str], float]:
        t0 = time.perf_counter()
        rows = self.con.execute(
            "SELECT text FROM chunks WHERE doc_name = ? AND vector IS NOT NULL "
            "ORDER BY list_cosine_similarity(vector, ?::FLOAT[]) DESC LIMIT ?",
            [doc_name, qvec, k],
        ).fetchall()
        return [r[0] for r in rows], time.perf_counter() - t0


# --- shared -------------------------------------------------------------------


def _make_settings(base_url: str, embedder: str) -> Settings:
    return Settings(backend_url=base_url, embed_url=base_url, embed_model=embedder,
                    rerank_url=base_url, rerank_model=_RERANKER)


def _rolled_recall(hit_texts: list[str], golds: list[set], reranker, query: str,
                   ks, thresh) -> dict:
    """Recall@k over hit_texts, optionally after the winning-combo reranker."""
    if reranker and hit_texts:
        try:
            scores = reranker(query, hit_texts[:40])
            order = sorted(range(len(hit_texts[:40])), key=lambda i: scores[i], reverse=True)
            ordered = [set(tokenize(hit_texts[i])) for i in order]
        except Exception:  # noqa: BLE001 - rerank is an optional bonus row
            ordered = [set(tokenize(t)) for t in hit_texts]
    else:
        ordered = [set(tokenize(t)) for t in hit_texts]
    return {f"@{k}_t{t}": float(_recall_hit(ordered[:k], golds, t)) for k in ks for t in thresh}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    ap.add_argument("--bearer", default=os.environ.get("DK_ENGINE_KEY"))
    ap.add_argument("--embedder", default=_EMBEDDER,
                    help="embedding model id served at the engine; default qwen3-embedding-0.6b")
    ap.add_argument("--docs-dir", default=str(Path(__file__).with_name("filings")))
    ap.add_argument("--cache-dir", default=str(Path(__file__).with_name(".sweep_cache")))
    ap.add_argument("--out", default=str(Path(__file__).with_name("results") / "index_sweep.json"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--db", default=str(Path(__file__).with_name("results") / "index_sweep.duckdb"))
    args = ap.parse_args()

    base_url, embedder = args.base_url, args.embedder
    if not _wait_backend(base_url, timeout=30.0):
        print(f"engine NOT up at {base_url} — start it, then re-run. (harness is ready.)",
              flush=True)
        return

    docs_dir = Path(args.docs_dir)
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in open(CACHE) if l.strip()]  # noqa: SIM115
    if args.limit:
        records = records[: args.limit]
    present = [r for r in records if (docs_dir / f"{r['doc_name']}.pdf").is_file()]
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in present:
        by_doc[r["doc_name"]].append(r)
    print(f"{len(present)}/{len(records)} questions have their filing "
          f"({len(by_doc)} filings)", flush=True)

    s = _make_settings(base_url, embedder)
    mc = MettleCollections(base_url, args.bearer)
    golds = {r["financebench_id"]: _gold_evidence(r) for r in present}
    questions = {r["financebench_id"]: r["question"] for r in present}

    duck = DuckDense(db_path)
    collection_names: dict[str, str] = {}
    n_chunks = 0
    for doc in by_doc:
        corpus = _cached_corpus(doc, docs_dir / f"{doc}.pdf", s, Path(args.cache_dir), embedder)
        if not len(corpus):
            print(f"    [skip] {doc}: no chunks", flush=True)
            continue
        duck.add(doc, corpus.chunks, corpus.vectors)
        coll = f"idx_{doc}"
        mc.ensure_collection(coll, embedder)
        for ch in corpus.chunks:
            _resilient(lambda c=coll, cid=ch.chunk_id, txt=ch.text:
                       mc.ingest(c, cid, txt), base_url, f"ingest {doc}")
        collection_names[doc] = coll
        n_chunks += len(corpus)
    duck.con.commit()
    print(f"built {len(collection_names)} filings / {n_chunks} chunks "
          f"(duckdb {db_path.stat().st_size/1e6:.1f} MB)", flush=True)

    qids = [r["financebench_id"] for r in present if r["doc_name"] in collection_names]
    qvecs = _resilient(lambda: embed_texts([questions[q] for q in qids], settings=s),
                       base_url, "query-embed")
    qvec = dict(zip(qids, qvecs))

    from docket.retrieval.rerank_client import make_reranker  # noqa: PLC0415
    reranker = None
    try:
        base_rer = make_reranker(s)
        if base_rer is not None:
            _resilient(lambda: base_rer("warmup", ["alpha", "beta"]), base_url,
                       "rerank-warmup", tries=6)
            reranker = lambda q, docs: _resilient(lambda: base_rer(q, docs), base_url,
                                                  "rerank", tries=12)
    except Exception:  # noqa: BLE001
        reranker = None

    results = {}
    for path in ("duckdb-exact", "turbovec-ann"):
        hits_at = {(k, t): 0 for k in _KS for t in _THRESHOLDS}
        hits_at_rr = {(k, t): 0 for k in _KS for t in _THRESHOLDS}
        lat, n = [], 0
        for doc, recs in by_doc.items():
            if doc not in collection_names:
                continue
            for rec in recs:
                qid = rec["financebench_id"]
                if qid not in qvec:
                    continue
                g, q = golds.get(qid) or [], questions[qid]
                if path == "duckdb-exact":
                    texts, dt = duck.search(doc, qvec[qid], max(_KS))
                else:
                    raw, dt = mc.search_dense(collection_names[doc], q, max(_KS))
                    texts = [h["text"] for h in raw]
                lat.append(dt)
                n += 1
                for k in _KS:
                    for t in _THRESHOLDS:
                        if _recall_hit([set(tokenize(x)) for x in texts[:k]], g, t):
                            hits_at[(k, t)] += 1
                rr = _rolled_recall(texts, g, reranker, q, _KS, _THRESHOLDS)
                for (k, t) in rr:
                    hits_at_rr[(k, t)] += int(rr[f"@{k}_t{t}"])
        n = n or 1
        results[path] = {
            "n": n,
            "recall": {f"@{k}_t{t}": round(hits_at[(k, t)] / n, 4)
                       for k in _KS for t in _THRESHOLDS},
            "recall_reranked": {f"@{k}_t{t}": round(hits_at_rr[(k, t)] / n, 4)
                                for k in _KS for t in _THRESHOLDS},
            "median_search_ms": round(float(sorted(lat)[len(lat) // 2]) * 1000, 1) if lat else None,
        }
        print(f"[{path:12}] recall@6(t0.5)={results[path]['recall']['@6_t0.5']:.3f}  "
              f"@10(t0.5)={results[path]['recall']['@10_t0.5']:.3f}  "
              f"reranked@6={results[path]['recall_reranked']['@6_t0.5']:.3f}  "
              f"med={results[path]['median_search_ms']}ms", flush=True)

    out_path.write_text(json.dumps({
        "embedder": embedder, "reranker": _RERANKER,
        "n_questions": len(present), "duckdb_bytes": db_path.stat().st_size,
        "results": results}, indent=2))
    print(f"\nwrote {out_path}", flush=True)
    a, b = results["duckdb-exact"], results["turbovec-ann"]
    diff = a["recall"]["@6_t0.5"] - b["recall"]["@6_t0.5"]
    print(f"duckdb - turbovec recall@6(t0.5) = {diff:+.3f} "
          f"({'duckdb' if diff >= 0 else 'turbovec'} advantages)", flush=True)


if __name__ == "__main__":
    main()
