# METTLE-ONLY CUTOVER CONTRACT — docket drops local retrieval, consumes the engine (2026-09-10)

**Status:** agreed with the user (grilling session, 2026-09-10). Implemented by
a docket-side agent. Evidence base: mettle-side
`benchmarks/retrieval_parity/results/BEST_CONFIG_VERDICT.md` (2026-09-10) and
the paired `agent-docs/BEST_CONFIG_ADOPTION_CONTRACT.md` in the mettle repo.

**Identity.** docket (= the product formerly named agent-assistant) is the only
consumer of mettle's `/v1` HTTP API. mettle is the only supported engine — all
providers, retrieval and chat. Per user decision: **hard mettle-only** — no
local retrieval fallback ever; mettled unreachable = docket features fail
loudly. Offline docket is dead (no chat, no retrieval): accepted.

**Sequencing (blocking).** This contract executes only AFTER mettle ships
`agent-docs/BEST_CONFIG_ADOPTION_CONTRACT.md`: the new `mode=hybrid`
(RRF-union), OR-mode `mode=keyword`, rerank batching, and qwen3 shipped
defaults are dependencies of every gate below.

**PWA context.** docket is headed for a PWA desktop app (stack decision belongs
to a separate grilling/agent). This contract must leave a clean boundary, not
pre-empt that work (§5).

---

## 1. Deletions — docket stops implementing what the engine provides

Everything below is measured-redundant; mettle's served equivalents are at
parity or better (hybrid .787@10 vs docket local .667; FTS5-OR keyword .487 vs
rank_bm25 .380; rerank bit-identical .880):

1. **Local sparse leg** — `_PurePythonBM25`, `rank_bm25` usage and the
   `rank_bm25` dependency (`retrieval/retriever.py`, `ingest` extra).
2. **Local dense leg** — cosine-over-Corpus-vectors numpy brute force in
   `retrieval/retriever.py`.
3. **Local fusion** — `rrf()` in `retrieval/retriever.py` (fusion moves into
   mettle's `mode=hybrid`, RRF k=60 over keyword∪dense top-40 pools).
4. **Local vector storage** — corpus vectors in `ingest/index.py`'s duckdb
   store. `Corpus` shrinks to what docket uniquely owns: chunk lineage
   (doc_id, page, form_type, company, ticker, fiscal_period) for citations,
   doc-level dedup, and ingest job state. Chunk TEXT for display comes back
   from mettle search hits.
5. **Non-mettle providers** — `providers/` drops Cerebras, Groq, llama-server,
   Ollama routing; `openai_compat.py` pointed at mettle's `/v1` becomes the
   only path, for chat and everything else.

docket's **chunking stays** (user decision): `ingest/ocr.py`, `ingest/chunk.py`
(220w/40 overlap, page-boundary-respecting, citation lineage), filing-metadata
extraction. Mettle's chunker is a different geometry and would invalidate every
parity number; docket posts its chunks to the engine.

## 2. New wiring — docket as a pure mettle client

| function | call | notes |
|---|---|---|
| embed | `POST /v1/embeddings` | already the case when `DK_EMBED_URL` set; becomes unconditional |
| dense search | `POST /v1/collections/{c}/search` `mode=dense` | |
| keyword search | same, `mode=keyword` | OR-mode + bm25 after mettle adoption |
| fused retrieval | same, `mode=hybrid` | RRF-union, k=60, engine-side |
| rerank | `POST /v1/rerank` (qwen3-reranker-0.6b:latest) | already `rerank_client.py`; batching lands engine-side (≤4 s/call) |
| chat | `providers/openai_compat.py` → mettle `/v1/chat/completions` | only provider |
| engine mgmt | `engine.py` `/engine/*` surface | stays as-is |

- Ingest: chunks (id, text, metadata incl. page/form_type/company/ticker/
  fiscal_period) → `POST /v1/collections/{c}/documents`. Metadata must round-trip
  (mettle persists it in `documents.metadata_json`) — citations depend on it.
- `iterative_retrieve` (multi-query merge) STAYS client-side — it is
  orchestration over multiple engine calls, not a duplicated engine feature.
- Collections re-ingested under the qwen3 default (1024-d); embed-gemma
  collections retired or kept only for comparison.

## 3. Engine lifecycle — docket supervises mettled

User decision: docket's backend owns mettled's process lifecycle:

- Settings gain the mettled binary path (+ port / METTLE_CONFIG location).
- On startup: if `/engine/status` (root surface, not `/v1`) doesn't answer,
  spawn mettled; health-probe until ready; restart on crash (bounded backoff).
- Surface engine state in the existing `engine_view()` (`available`/status/
  device already exist) — the PWA gets an honest "engine offline/starting"
  state instead of silent failure.
- Never import engine code (existing boundary rule, unchanged).

## 4. Configuration handoff

mettle ships the winning defaults (qwen3 embed + rerank aliases, 8192-ctx
retrieval-resident profile). docket's setup does NOT push engine config; it
points at the engine and relies on mettle's defaults. If settings must override
(profile selection via `PUT /engine/profiles/active`), that already exists via
`engine.py` — no new config surface.

## 5. Portability constraints (PWA pre-shaping, boundary only)

The PWA agent will grill the stack separately; this contract only guarantees it
inherits a clean shape:

- Engine access is HTTP-only, settings-driven (`backend_url`, binary path);
  no hard-coded localhost assumptions beyond defaults.
- No mettle imports, no direct reads of mettle's state/collection files —
  `duckdb`/collections under `.state` are mettle's, not docket's.
- No new local-FS dependencies in retrieval paths beyond docket's own
  ingest/corpus metadata store.
- All engine features degrade to explicit "engine offline" UI states, never
  silent local substitutes (there are none left).

## 6. Acceptance gate (blocking)

End-to-end parity re-run **through docket's own path** (its
`eval/financebench.py` harness) against the mettled engine, after re-ingest
under qwen3:

1. rerank over the engine's new `hybrid` pool **≥ .90@10 t0.5** (target
   .913 ± .02) — the winner now lives entirely in the engine;
2. keyword (`mode=keyword`) **.487 ± .02 @10 t0.5**;
3. chunking proven unchanged (same frozen corpus digest — chunking stayed);
4. supervision verified: kill mettled mid-run → docket detects, restarts,
   requests recover; no silent fallback paths;
5. rerank latency observed ≤ 4 s p50 per 40-doc call through the client;
6. providers: only mettle paths reachable; removed providers have no callers
   (grep-verified).

Delivery report: gate table, deletion list with file/symbol names, deviations.

## 7. Explicit non-goals

- PWA stack/setup decisions (other agent's grilling; §5 constraints only).
- Chunker changes (parity substrate frozen).
- New docket-side caching/indexing of any kind.
- mettle-side changes of any kind (they belong to the adoption contract).
