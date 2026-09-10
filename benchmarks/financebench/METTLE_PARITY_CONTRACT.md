# Mettle ↔ docket retrieval parity contract (2026-08-31)

**Question being answered.** Does mettle's native retrieval stack — turbovec
ANN index + SQLite-FTS5 keyword/hybrid search + served rerankers — earn its
place in mettle, or should mettle shrink to a pure inference engine and leave
retrieval (BM25, exact-cosine dense, fusion, rerank orchestration) to docket?
This is decided by **measured, equivalent numbers on real FinanceBench data**,
not by argument. The mettle side is already measured
(`/personal-projects/mettle/benchmarks/retrieval_parity/results/mettle_results.json`,
harness `mettle_bench.py` in the same dir). **This document is the contract for
producing the docket-side numbers in a way that is exactly equivalent.**

**Authority.** Written by the mettle-side agent after mapping both codebases
(2026-08-31). The executing agent MUST follow §2 (equivalence rules) to the
letter; deviations MUST be disclosed in the output JSON's `deviations` field.
Do not modify docket production code. One new benchmark script is expected:
`benchmarks/financebench/docket_parity.py` (plus its results file).

## 1. What mettle measured (final, 2026-09-04)

Source: `/personal-projects/mettle/benchmarks/retrieval_parity/results/mettle_results.json`
(full per-question recall@k in `results/mettle_state.json`). 150 questions / 84
filings / 36,920 chunks, GPU-resident models, recall@k with t0.5 unless noted:

| embedder | arm | @3 | @6 | @10 | @20_t0.8 |
|---|---|---|---|---|---|
| embed-gemma | exact (local cosine) | 0.573 | 0.740 | 0.827 | 0.773 |
| embed-gemma | dense (turbovec ANN) | 0.587 | 0.747 | 0.833 | 0.780 |
| embed-gemma | hybrid (FTS5→dense) | 0.587 | 0.747 | 0.833 | 0.780 |
| embed-gemma | rerank dense (bge-rr) | 0.660 | 0.780 | 0.860 | 0.800 |
| embed-gemma | rerank dense (qwen3-rr) | 0.740 | 0.820 | 0.900 | 0.833 |
| bge-m3 | dense | 0.540 | 0.660 | 0.740 | 0.687 |
| bge-m3 | rerank dense (qwen3-rr) | 0.713 | 0.813 | 0.873 | 0.773 |
| qwen3-emb-0.6b | dense | 0.627 | 0.733 | 0.787 | 0.767 |
| qwen3-emb-0.6b | rerank dense (qwen3-rr) | 0.747 | 0.827 | 0.867 | 0.813 |

**keyword arm is 0.000 at every k** — mettle's FTS5 ANDs all sanitized query
terms and FinanceBench questions are long natural sentences; zero chunks
contain every term, so pure keyword returns nothing (hybrid unaffected: its
FTS5 leg finds no candidates and falls back to dense). Reproduced on a live
collection. Docket's BM25 arm (rank_bm25, additive scoring) is the comparable.

Latency (embed-gemma, per query, n=150): dense p50 42.9 ms (incl. 38.3 ms
query embed); exact-cosine local p50 27.6 ms; hybrid p50 37.2 ms; global
cross-filing search p50 7.3 s. Rerank call (≤40 docs): bge-rr p50 36.0 s;
qwen3-rr p50 12.9 s.

Throughput/memory: embed batch-64 3.74–4.83 ch/s per embedder; per-filing
ingest ~3.4 ch/s (per-chunk API cost, see §2 note); turbovec 4-bit index
15.1/20.0 MB vs raw f32 113/151 MB per embedder (~7.5× smaller); 5 models
GPU-resident 2.9 GB on a 6 GB card.

Provenance: mettle git head 1d37ec3 (dirty: retrieval.rs cosine-normalization
fix + parity harness), mettled built 2026-09-03 23:29 WITH `--features cuda`,
profile bench-retrieval n_ctx=8192. Global collection numbers
(36,920 chunks, cross-filing) in `global_collection(embed-gemma)`: dense/hybrid
0.886 recall-proxy mbo, keyword 0 (same AND-expansion issue).

---

## 2. Equivalence rules (binding)

Both sides MUST share, exactly:

1. **Questions + gold truth**: `benchmarks/financebench/data/financebench_merged.jsonl`
   (150 rows; all usable — every `doc_name` has its filing and gold evidence).
   Gold evidence = `evidence[].evidence_text` token sets (skip empties),
   exactly as `sweep_recall.py::_gold_evidence`.
2. **Chunks**: the frozen sweep cache `.sweep_cache/<embedder-slug>/<DOC>/corpus.jsonl`
   (36,920 chunks over 84 filings; byte-identical chunk text across the three
   embedder caches — verified 2026-08-31). Chunk text used for scoring = the RAW
   `text` field. Chunk text used as *embed input and as the mettle-stored text* =
   `embed_clean(text)` (strip control chars `[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]`
   → space, cap 2000 chars — `sweep_recall.py::_embed_clean`). Only 53/36,920
   chunks (0.1%) are affected by the cap.
3. **Metric** (verbatim from `sweep_recall.py`, mirrored in mettle's
   `parity_common.py`): `tokenize = re [a-z0-9]+ lowercase`; overlap coefficient
   `|a∩b|/min(|a|,|b|)`; recall hit iff any retrieved chunk overlaps any gold
   span at ≥ threshold. Report **recall@k** for `k ∈ {3,5,6,10,20}` ×
   `t ∈ {0.5, 0.8}` + `mean_best_overlap`. Recall is a proxy, not answer accuracy.
4. **Scoping**: per-filing — each question retrieves only within its own
   filing (docket's per-filing `Corpus`; mettle's per-filing collection).
5. **Candidates/k**: rerank arms rerank the **top-40** candidates
   (`Retriever` default `candidates=40`) and report top-k after rerank.
6. **Timing semantics**: query latency = wall time from question to ranked hits,
   **including query embedding** on both sides. Rerank latency = wall time of
   the `/v1/rerank` call with ≤40 documents.
7. **Embedders**: `embed-gemma:latest` (768d), `bge-m3:latest` (1024d),
   `qwen3-embedding-0.6b:latest` (1024d). **Rerankers**:
   `bge-reranker-v2-m3:latest` (logits), `qwen3-reranker-0.6b:latest` (P(yes)).
   All served by mettle over `/v1` — docket has no local model runtime, and both
   systems consume the *same* served models; that is the point.
8. **Backend**: mettle `mettled` built from `/personal-projects/mettle`
   (git head + binary mtime recorded in the mettle results provenance), started
   with the benchmark profile — see §5.
9. **Vector fidelity**: mettle verified the cached vectors are bit-for-bit what
   the current binary emits (`min_cos` ≥ 0.9998 for all three embedders, in
   `mettle_results.json.fidelity`). Docket arms that reuse cached vectors inherit
   this; if you re-embed, disclose it.
10. **Cosine space (fix 2026-09-01)**: mettle's `Collection` now L2-normalizes
    embeddings at index-add and query time (mettle-core `retrieval.rs`,
    `l2_normalize`). Before that fix turbovec's inner-product estimate made
    dense ranking norm-biased (long high-norm chunks won against true matches —
    caught by the AMCOR_2022_8K self-search sanity). All mettle dense/hybrid
    scores are therefore cosine estimates in [-1, 1], same ranking space as
    docket's exact cosine. Docket arms need no change (Corpus/Retriever already
    cosine).

## 3. Arms to run on the docket side

Naming: arm ids below map 1:1 to mettle arms so the final table pairs up.

| arm | what | mettle counterpart | source of truth |
|---|---|---|---|
| `exact` | dense exact-cosine over cached vectors (`Corpus.dense_search`) | mettle `exact` (same vectors, same algorithm) | **new script** |
| `bm25` | BM25-only ranking over `embed_clean(text)` tokenization | mettle `keyword` (SQLite FTS5, BM25 order) | **new script** |
| `hybrid` | production `Retriever` (BM25+dense+RRF, no rerank) | mettle `hybrid` (FTS5-top100 → dense) | **new script** (or cite `recall_sweep.json` `reranker=none` rows — same pipeline; disclose) |
| `rerank_exact(<rr>)` | rerank the `exact` top-40 via `/v1/rerank` | mettle `rerank_exact(<rr>)` | **new script** |
| `rerank_hybrid(<rr>)` | rerank the production fused top-40 via `/v1/rerank` | mettle `rerank_dense(<rr>)` (native-pipeline candidates) | cite `recall_sweep.json` reranker rows **only if re-validated**: re-run ≥ `embed-gemma` × both rerankers fresh and report delta vs the 2026-08-24 rows; if delta > 0.01 recall, re-run all |
| `embed_thruput` | `embed_texts` (batch=64) over the fixed 3-filing subset | mettle `embed_thruput` | **new script** |
| `memory` | raw f32 bytes `4×dim×36,920` + `corpus.jsonl` on-disk size per embedder | mettle turbovec `index.bin` bytes | **new script** |

Notes:

- **`exact` is the shared reference arm.** Both sides rank the *same* cached
  vectors by exact cosine; expect recall parity within ±0.005 (any larger gap
  is a scoring bug, not a system difference — investigate before reporting).
- **`bm25`**: build the BM25 index over `embed_clean(text)` (mettle's FTS5
  indexes the cleaned/stored text). Membership/overlap scoring stays on RAW text
  per rule 2. Use `rank_bm25` (installed with `--extra ingest`); the
  pure-Python fallback is acceptable but disclose which ran
  (`tests/test_pipeline.py::test_sparse_recall_consistent_across_bm25_backends`
  shows they agree).
- **`rerank_hybrid`** reuse condition: `results/recall_sweep.json` (2026-08-24)
  was produced against an older mettle binary; the current binary's vectors are
  verified identical, but rerank scoring may have shifted. Re-validation is
  mandatory for at least one embedder.
- **Optional arm `rerank_hf(<rr>)`**: the true "reranker lives in docket"
  alternative — run HF `BAAI/bge-reranker-v2-m3` via
  `uv run --with torch --with transformers python ...` (fp32, CUDA if it fits,
  else CPU — record device), score the same `exact` top-40 candidate lists, and
  report recall + latency + Spearman rank correlation of its scores vs mettle's
  `/v1/rerank` scores. This is the only arm where docket runs model weights
  locally; it answers "what would docket pay to stop calling mettle for
  rerank?". Skip only with a stated reason.

## 4. Output contract

Write `benchmarks/financebench/results/docket_parity.json`:

```json
{
  "generated": "<iso8601>",
  "n_questions": 150, "n_filings": 84,
  "arms": {
    "<embedder>": {
      "exact":            {"@3_t0.5": 0.0, "...": "...", "@20_t0.8": 0.0, "mean_best_overlap": 0.0},
      "bm25":             {"...": "..."},
      "hybrid":           {"...": "..."},
      "rerank_exact(bge-reranker-v2-m3:latest": {"...": "..."},
      "rerank_hybrid(...)": {"...": "..."},
      "latency": {"exact": {"n":0,"mean_ms":0,"p50_ms":0,"p95_ms":0}, "bm25": {"...": "..."},
                   "rerank_call(bge-reranker-v2-m3:latest)": {"...": "..."}}
    }
  },
  "embed_thruput": {"<embedder>": {"subset_filings": ["..."], "texts": 0, "batch": 64,
                                     "chunks_per_sec": 0.0, "per_text_ms": 0.0,
                                     "batch_latency": {"n":0,"mean_ms":0,"p50_ms":0,"p95_ms":0}}},
  "memory": {"<embedder>": {"raw_f32_bytes": 0, "corpus_jsonl_bytes": 0, "dim": 0}},
  "rerank_hf": {"<reranker>": {"device": "...", "recall": {"...": "..."},
                                 "latency": {"...": "..."}, "spearman_vs_mettle": 0.0}},
  "deviations": [],
  "provenance": {"docket_git_head": "...", "mettle_binary": "<from mettle_results.json>",
                  "bm25_backend": "rank_bm25|pure-python", "vectors": "cached|re-embedded"}
}
```

Same latency-stats fields as mettle (`n, mean_ms, p50_ms, p95_ms`). Report
recall at **all** k/threshold pairs even where mettle printed only t0.5.

## 5. Environment

1. mettle server (already measured with this exact setup):
   ```bash
   cd /personal-projects/mettle
   METTLE_CONFIG=/personal-projects/mettle/benchmarks/retrieval_parity/mettle-bench-profile.toml \
     target/release/mettled
   # then activate the profile (engine idles without one):
   curl -X PUT http://127.0.0.1:11434/engine/profiles/active \
     -H 'Content-Type: application/json' -d '{"name":"bench-retrieval"}'
   ```
   Config serves the 3 embedders + 2 rerankers GPU-resident at `n_ctx=8192`
   (131072 inflates per-slot VRAM estimates and the profile manager drops 4 of
   5 models — do not raise it). All 5 together use ~2.8 GB of the 6 GB card.
2. `uv sync --extra ingest` in `/personal-projects/docket`.
3. Long rerank arms: mettle's `/v1/rerank` decodes each (query, doc) pair in its
   own context (parity-mandated); ~11 s per 40-doc call. Use
   `engine_watchdog.sh`-style supervision if running unattended.

## 6. Deliverable

`results/docket_parity.json` + a short stdout summary. Then the mettle-side
agent (or the user) pairs the two JSONs into the final verdict:
**keep retrieval in mettle vs move it to docket**, on measured recall,
latency, memory, and engineering surface.
