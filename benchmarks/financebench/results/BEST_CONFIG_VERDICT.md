# BEST_CONFIG_VERDICT — docket-side mirror (2026-09-10)

Execution: mettle-side agent, per `BEST_CONFIG_CONTRACT.md` (canonical copy in
`/personal-projects/mettle/benchmarks/retrieval_parity/`; artifacts there:
`results/BEST_CONFIG_VERDICT.md`, `results/mettle_best_config.json`,
`results/turbovec_ablation.json`, per-question state files). Docket code and
data were read-only inputs. **This file is the docket-side mirror, written after
independent verification of every headline number** — the verifier's own metric
implementation (contract §2 spec, not the mettle harness code), numpy ranking
over the frozen sweep-cache vectors, own FTS5 SQL, and a live re-run of the
served arms on the same mettled binary.

## Verification result: numbers CONFIRMED

- **Aggregates**: all 6 arms × 10 recall@k/t flags + mbo + n recompute
  bit-for-bit from the per-question state file. Latencies reproduce under the
  harness's seconds→ms convention.
- **exact arm (150/150)**: independently re-ranked every filing's frozen
  vectors (numpy cosine, persisted query vectors) and re-scored with my own
  overlap-coefficient implementation — every per-question flag and
  best_overlap matches.
- **keyword_or arm (150/150)**: independently re-issued FTS5 OR-mode
  `MATCH 't1 OR t2 …' ORDER BY bm25()` over the collection DBs (own SQL,
  own tokenization) — every per-question row matches.
- **Served arms, live re-run (12 sampled questions)**: dense, hybrid,
  rerank_hybrid(qwen3-rr), rerank_exact(qwen3-rr) all reproduce exactly
  (rerank latency within ±0.5 s of published).
- **Structural rerank consistency (150 q × 2 arms)**: rerank arms inherit
  their candidate pool's `best_overlap` exactly (order-invariance over the
  same top-40) — confirms the candidate-pool story below.
- **turbovec bw4 (Part C), reproduced with the actual wheel**: index bytes
  **20,459,859 — exact match**; top-10 overlap vs exact **0.954 — exact
  match**; recall10 0.6933 with the published +0.0067 delta (1/150); global
  exact recall@10 reference 0.6867 recomputed exactly; latency 0.63 ms p50
  vs published 0.555 (mine ran with mettled resident — noise).

## Head-to-head (mettle vs docket, verified on both sides)

| arm | mettle @3/@6/@10/@20t8 | docket @10 | Δ@10 |
|---|---|---|---|
| exact | .627/.727/**.793**/.760 | .793 | **.000** |
| dense (served) | .620/.747/.787/.773 | — | sanity ≈ exact |
| hybrid (served) | .620/.747/.787/.773 | .667 | **+.120** |
| keyword_or (FTS5-OR+bm25) | .320/.433/**.487**/.500 | bm25 .380 | **+.107** |
| rerank_exact(qwen3-rr) | .760/.840/**.880**/.813 | .880 | **.000** |
| rerank_hybrid(qwen3-rr) | .747/.827/**.867**/.813 | **.913** | **−.046** |

**Decisions (Q1–Q3), as measured:**
1. Mettle matches docket on every shared arm to the third decimal, but
   docket's winner (.913 rerank_hybrid) does not transfer: docket's RRF hybrid
   pool is diverse (the reranker converts +.246 over its .667 hybrid base);
   mettle's hybrid collapses to dense (.787 == dense), so rerank yields only
   .867. Mettle's best is rerank_exact = .880@10 — closing the gap needs an
   RRF-style diverse candidate pool (fusion change), not a reranker change.
2. FTS5 keyword leg **salvageable** — OR-mode + bm25() = .487@10 vs docket
   bm25 .380 (exceeds; decision rule met). Stays in mettle with the named fix
   (OR-mode search mode; production mode is still AND → 0.000).
3. turbovec **stays in mettle-core**: 0.555 ms p50 vs numpy 6.58 ms (11.8×),
   7.4× smaller than f32, metric-invisible recall cost (+.0067), in-kernel
   allowlist, O(1) deletes, no train phase; FAISS-HNSW/hnswlib tie speed at
   8× memory with no deletes; IVFPQ loses .38 of exact overlap and is useless
   filtered.

## Verification caveats (gaps found — none change a verdict)

1. `mettle_best_config.json.deviations` is `[]` while the verdict §3 lists six
   deviations — the contract requires them in the output JSON. The Part C JSON
   carries its two (bw8 unsupported by the 1.0.0 wheel; sqlite_vec skipped).
2. `peak_rss_bytes` is 0 for every Part C engine — the contract's peak-RSS
   delta during build was not measured (ru_maxrss is a high-water mark that
   cannot be reset mid-process); index bytes on disk are the measured column.
3. Turbovec was measured via the PyPI wheel **1.0.0**, while mettle-core pins
   turbovec **0.8.1** — disclosed in provenance, but the verdict's
   "mettle-core" phrasing slightly overstates identity. The stay-decision's
   magnitudes (11.8×, 7.4× compression) are unlikely to flip across those
   minor versions, but the 0.8.1 build is what mettle actually ships.
4. Sanity gate relaxed "self-search top-1" to "top-10 containment" for the
   hybrid/keyword_or paths (dense kept strict top-1) — disclosed in the
   sanity note, not in deviations.
5. Search-arm latencies exclude the ~41 ms query embed (docket's per-arm
   latencies include ~44 ms) — add ~41 ms when comparing latency rows.
6. Contract quoted docket exact p50 88.3 ms; `docket_parity.json` says
   98.25 ms incl. 41.6 ms embed (53.47 ms ranking-only). Recall targets in
   the contract were verified accurate.

## Component-residence outcome (what this contract actually decides)

The contract's purpose was **component residence** — for each component,
decide by measurement which project it lives in, NOT "docket reimplements
what mettle provides". The measured outcome:

| Component | Lives in | Why (deciding number) |
|---|---|---|
| Chunking / ingestion | docket | shared frozen substrate; settled |
| Embedding-model serving | mettle | query embed 40.6 ms served; fidelity gated; one engine for all surfaces |
| Dense vector index (turbovec bw4) | mettle | 0.555 ms p50 vs 6.58 ms numpy brute force (11.8×), 7.4× smaller than f32, +0.0067 metric delta, in-kernel allowlist, O(1) deletes — the index over the embeddings, not an alternative to them |
| Keyword / BM25 leg | **mettle** (with OR-mode FTS5 fix) | FTS5-OR .487@10 **beats** docket's rank_bm25 .380@10 — docket should drop its weaker local keyword leg once mettle ships OR-mode, and consume it over `/v1` |
| Hybrid fusion (candidate pool) | mettle, after adopting RRF-style union | docket's .913 winner is rerank over a DIVERSE pool; mettle's FTS5→dense fusion collapses to dense (.787) — mettle must reproduce the diversity so the .913 winner lives engine-side |
| Reranker serving | mettle | scores identical (.880@10 / .813@20t8); batching fix separate |
| Orchestration / API | mettle `/v1` | consumer coupling exists |

Consequence: docket does not reimplement any of the retrieval primitives the
engine measures better — keyword (OR-mode), dense ANN index, rerank serving,
and fusion stay mettle-side; docket stays the application layer: ingestion,
agent, trust/citation layer, and the OpenAI-/v1 consumer.

Turbovec's purpose (it is not redundant with embeddings): embeddings produce
vectors; an index answers "top-k vectors among 36,920 per filing". The
ablation measured whether that index deserves to exist: yes — 4-bit
quantization gives 7.4× memory reduction and 11.8× lower query latency than
brute force at metric-invisible cost (+0.0067), with O(1) deletes and
in-kernel allowlist filtering that FAISS/HNSW cannot match without post-
filter at 5–28× the latency.
