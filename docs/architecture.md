# Architecture

> **Read [case-study.md](case-study.md) first.** The product is a
> **finance-filings research assistant** (SEC EDGAR), not a domain-agnostic
> engine — see [decisions.md](decisions.md) § Session 2. This doc describes the
> machinery; the case study says what it's *for*.

## Data flow
```
Live feed: SEC EDGAR  ── platform profile ──► Redpanda ─┐
Any-PDF folder        ── lite profile ──► embedded queue ┤   (any-PDF = side effect;
                                                         ▼    finance is the identity)
                 Unlimited-OCR ► chunk ► embed (dedicated model) ► turbovec index
                 finance-aware sectioning: Item 1A / MD&A / financial statements
                                                                            ▼
   LangGraph agent  ◄─ hybrid retrieve + rerank + ITERATIVE re-query (needle-in-haystack)
     finance tools: retrieve · extract_financials(revenue/EPS/segment) · detect_guidance_change
                    · compare_across_docs(peers) · compute_metric · cite_page · generate_report
                                                                            ▼
   Answer / Report ─► Trust layer (logprob + semantic entropy → 🟢🟡🔴 reliability)   ◄── BASELINE
                   ─► Interpretability (SAE feature-attribution "lab mode")           ◄── HEADLINE
                       must BEAT the trust baseline on the eval harness to ship as a real feature
                                                                            ▼
   Eval: RAGAS + DeepEval(CI gate) + Promptfoo + XAI(XGBoost+SHAP) + Langfuse + HITL
         ↑ also the referee that decides SAE-vs-baseline
   Benchmark: FinanceBench (headline, rigor loop) + RULER (multi-hop stress); honest frontier baseline
              {Qwen3-0.6B · Gemma4-E4B · +tooling · Muse Glimmer 30B · Muse Spark 1.2}
Backend: OpenAI-/v1 — llama-server ‖ infengine ‖ Ollama ‖ Cerebras/Groq
```

The headline is the **interpretability**, not "small rivals frontier"; the small
model is a *means* (white-box access + hardware), and FinanceBench is a
rigor/optimization loop reported honestly — not a "we beat GPT" claim. Full
reasoning: decisions § S2-Q3/Q4/Q9.

## The backend boundary (and the infengine reconciliation)
The app depends on a model backend **only** across the OpenAI-compatible `/v1`
HTTP contract (`DK_BACKEND_URL`, default `http://127.0.0.1:11434/v1`). It never
imports backend code. Consequences:
- The stock llama.cpp **`llama-server`** (running on this host as `gemma4:26b`),
  the **infengine** Rust engine, and **Ollama** are interchangeable backends.
- **infengine can stay half-built** — we only need its already-working
  single-user chat/embed surface, reached over `/v1`. Two repos, one contract,
  zero code coupling.
- One client class (`providers/openai_compat.py`) also serves **Cerebras** and
  **Groq** — the router just swaps base_url/key.

### Embeddings
The chat server is not an embedding server (`/v1/embeddings` → 501 unless started
with `--embeddings`). Run a **dedicated** `llama-server --embeddings` (e.g.
EmbeddingGemma-300m) on a separate port and set `DK_EMBED_URL`. Never bolt
`--embeddings` onto the shared chat server.

## Profiles (Q13)
- **`lite`** — embedded queue + DuckDB, zero-config, any-PDF. The **default a
  downloader gets**: point it at *their own* PDFs, no Kafka/Spark. **Never wired
  to the operator's infrastructure** (decisions § S2-Q8).
- **`platform`** — live Redpanda → Spark → lakehouse (medallion, dbt,
  DuckDB/BigQuery, Airflow). Shipped as Docker/Helm/Ansible so users run it on
  their **own** hardware. The operator's public demo is *one* `platform`
  deployment he runs — decoupled from every install.

## Hosting topology (three decoupled layers)
The operator's live demo and the shipped product are **different deployments of
the same code**. Full rationale in [case-study.md](case-study.md) § Hosting and
decisions § S2-Q5–Q9.

- **GitHub** — all code (`lite` + `platform` + IaC). What "counts" for the
  portfolio and what users `curl | sh`-install onto their own box.
- **Operator's domain** (GitHub Pages, static, free) — case-study page + the
  **demo video** of the full local-GPU run + SAE interpretability lab-mode (the
  GPU parts that can't run live).
- **`demo.<domain>` → operator's existing CPU-only VPS** (free tiers) — ONE live
  `platform` deployment:
  - EDGAR → Kafka → Spark/dbt → index, and the dashboard (filings landing in real
    time, ingest stages, freshness/throughput) — **all live, no GPU**.
  - **Ask box = cached, read-only** Q&As rendered from a real local-GPU run,
    showing full citations + trust + **SAE interpretability** (the headline made
    visible without a GPU). No live inference, no public API key.

Two distinct "observability" surfaces — do not conflate: **pipeline** obs (stream
health/throughput; no GPU; live) vs **model/answer** obs (trust + SAE
interpretability; needs GPU; cached/video + reproducible-by-clone). A JS
*simulation* pretending to be the app is rejected — cached-with-disclosure is real
software with one offline feature; a fake pipeline would kill the anti-slop thesis.

## Retention (VPS, free-tier friendly)
Keep only the **derived** layer; never hoard raw filings (decisions § S2-Q9):
- **Bronze (raw filing blobs)** → *pointer only*; EDGAR is a permanent public
  archive, re-fetch by accession-id on demand.
- **Silver (chunk text + embeddings + index)** → keep; chunk text *is* the cited
  evidence, so citations survive without the raw blob.
- **Gold (extracted financials marts)** → keep; small.
- **10-year backfill** → a *demonstrated capability* (run once, capture, prune),
  not a resident. The VPS holds a rolling recent window + the fixed FinanceBench
  set.

## Capability tiers (graceful degradation)
`providers/base.py::Capability` advertises CHAT / EMBED / LOGPROBS /
HIDDEN_STATES / SAE. The trust + interpretability features light up per backend:
logprobs (local + most APIs) → Tier-1; hidden states (local weights) → Tier-2;
Gemma Scope SAEs (Gemma 2/3 via TransformerLens) → Tier-3. API-only backends
degrade to Tier-1.

**Tier-3 (SAE lab mode) is the product's headline** (decisions § S2-Q3/Q4), and
it needs a **local white-box backend** — this is *the* reason the model is small
and self-hosted, not a "small rivals frontier" thesis. On the public VPS demo the
lab-mode is shown via **cached** examples (no GPU there); it runs live only on a
GPU-backed local/clone install. Tier-1 is the always-on baseline the SAE detector
must *beat on the eval harness* to count as a real feature rather than a showcase.
