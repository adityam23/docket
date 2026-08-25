# Stack (SOTA verified Aug 2026 — re-verify before wiring; models drift monthly)

> Domain is **finance filings (SEC EDGAR)** and the **headline is the SAE
> interpretability layer**, not the small-vs-frontier benchmark — see
> [decisions.md](decisions.md) § Session 2. The interpretability row below is
> therefore **not** "roadmap fluff"; it's the differentiator (gated on beating the
> Tier-1 trust baseline on the eval harness).

| Layer | Choice | Notes |
|---|---|---|
| Vector index | **turbovec** | TurboQuant/ICLR-2026; 87% less memory than FAISS, no training. PyPI name/build unverified — pin in Phase 1. |
| OCR | **Baidu Unlimited-OCR** (3B MoE, MIT) | one-pass multi-page; needs ≥8 GB BF16 → quantize or fall back to **Surya** (650M) / **dots.ocr** (1.7B) to fit 6 GB. HF weights, not PyPI. |
| Embeddings | **EmbeddingGemma-300m** / **BGE-M3** (hybrid dense+sparse) | run as a dedicated `llama-server --embeddings` or via sentence-transformers. |
| Reranker | **BGE-reranker-v2** / **Qwen3-Reranker** | for needle-in-haystack recall. |
| Agent framework | **LangGraph** | 2026 production consensus (checkpointing/durability). Extra: `uv sync --extra agent`. |
| Serving model (6 GB) | **Gemma-4-E4B-it-Q4_K_M** | already on disk (5 G); fits the 6 GB target. |
| Providers | local `/v1` + **Cerebras** (~1M tok/day) + **Groq** (fast) | BYO key; verify model ids/limits (drift). |
| Eval | **RAGAS** + **DeepEval** (pytest CI gate) + **Promptfoo** (model compare/red-team) | Promptfoo is a Node tool, not pip. |
| Tracing/obs | **Langfuse** (MIT, OTel) | trace store + HITL annotation queue. |
| XAI auto-eval | **XGBoost/LightGBM + SHAP** | explainable hallucination classifier over trace features. |
| **Interpretability (HEADLINE)** | **Gemma Scope** SAEs (Gemma 2/3) via TransformerLens + sae-lens | Tier-3 "lab mode" glass-box feature-attribution/steering — the product's differentiator. Needs a local white-box backend (the reason the model is small+local). Ship as a real feature only if it beats the Tier-1 trust baseline on the eval harness (else honest showcase). |
| Benchmark | **FinanceBench** (headline, rigor/optimization loop) + **RULER** (multi-hop stress), NIAH-2 baseline | frontier baseline = **Muse Spark 1.2** (Meta MSL, cheap API); **Muse Glimmer 30B** open baseline. Report **honestly** vs the baseline — not "we beat GPT" (decisions § S2-Q9). |
| Platform DE | Redpanda/Kafka · Spark · dbt · DuckDB/BigQuery · Airflow | `platform` profile; the operator runs one live instance on the CPU-only VPS. Single live feed = **SEC EDGAR** (arXiv/EUR-Lex dropped). Derived-layer + rolling-window retention only (decisions § S2-Q9). |

## Phase-gated dependencies
Kept OUT of `pyproject.toml` until their phase lands (so `uv sync` stays fast on
a disk-constrained host). Add per phase:
- **ingest:** turbovec, an OCR model (HF), sentence-transformers/FlagEmbedding, pypdf, duckdb.
- **eval:** ragas, deepeval, langfuse, xgboost, shap (+ promptfoo via npm).
- **interp:** torch, transformer-lens, sae-lens.
- **platform:** confluent-kafka, pyspark, dbt-duckdb.

## Models already on the dev host (`/personal-projects/models-hermes-agent/`)
- `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` (16 G) — not currently serving (infengine's
  `gemma4:e2b` on `127.0.0.1:11434` replaced the 26B llama-server).
- `gemma-4-E4B-it-Q4_K_M.gguf` (5 G) — the 6 GB-fit product serving model.
- `mtp-gemma-4-26B-A4B-it.gguf` (441 M) — MTP/speculative draft.
- Embeddings: infengine serves `embed-gemma:latest` (EmbeddingGemma-300m,
  768-dim) on `127.0.0.1:11434` — `DK_EMBED_URL=http://127.0.0.1:11434/v1`.
