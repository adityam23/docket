# Roadmap

Ordered, not rigidly dated (deadlines are targets; scope-first). **Re-anchored
2026-08-19** to the finance case study + interpretability headline — see
[decisions.md](decisions.md) § Session 2 and [case-study.md](case-study.md).

## Phase 0 — scaffold ✅ (2026-08-18)
Package, config, provider `/v1` abstraction, Tier-1 trust primitive, CLI + web
health, offline tests. Verified against the live llama-server.

## Phase 1 — lite MVP  (target ~2026-08-24: GitHub + website write-up)
Pipeline plumbing ✅ (2026-08-18, offline-verified + live `dk ask`; see `docs/decisions.md`).
Remaining = wiring the heavy pieces behind seams already in the code, now
**finance-flavoured**:
- Dedicated embedding endpoint + `ingest/embed.py` E2E.  ← seam ready; sparse-only today
- OCR → chunk → turbovec index over a filings folder (`lite`).  ← chunk/index done; OCR wired via infengine `/v1/ocr` (`_ocr_model.py`, needs pymupdf + a scanned-PDF E2E); turbovec still a seam
- **Finance-native retrieval + tools:** Item 1A / MD&A / financial-statement
  sectioning; `extract_financials` / `detect_guidance_change` /
  `compare_across_docs(peers)`. ← generalise the existing tools to the domain
- Hybrid + rerank + iterative retrieval.  ← done except the cross-encoder reranker seam
- `POST /ask` → cited answer + reliability label; the non-technical web UI. ← ✅ done
- Publish repo + write-up (frame it as the finance case study, not a generic engine).

## Phase 2 — install + live ingestion + public demo  (target ~2026-08-31)
- `install.sh` (`curl | sh` prebuilt binaries) + `uninstall.sh` wired to real releases.
- **DE-first live SEC EDGAR spine** → Redpanda → OCR → index, running continuously
  — the single flagship feed. Derived-layer + rolling-window retention (pointer to
  EDGAR for raw); free-tier only.
- **Hosting topology (decisions § S2-Q5–Q9):**
  - GitHub Pages case-study page on the operator's domain + the demo video.
  - `demo.<domain>` → the CPU-only VPS running the live `platform` pipeline +
    dashboard (pipeline observability live), **Ask box cached/read-only**.

## Phase 2.5 — generator accuracy + FinanceBench loop  (prep DONE 2026-08-22)
Accuracy is the precondition for the interpretability headline to matter, so it
comes first (decisions § Session 4). **DONE:** the generator base was chosen on
measured data — **Qwen3.5-4B Claude-Opus-Reasoning-distill** (94% on the
discriminating smoke set) — and the whole measurement+training loop is built in
`benchmarks/financebench/` (smoke gate, real 150-Q eval `run_full.py`, SFT builder,
portable cloud QLoRA). **DONE (2026-08-24, 150-Q, Claude-judged, corrected scorer):**
- Real **FinanceBench 150-Q** on the chosen distill — **oracle 80.7%** vs gemma 55.3%
  (generator quality, **+25.4 pt**); confirms the Session-4 base choice on real data.
- **Finance-adapted retrieval (T26)** — embed-gemma + qwen3-reranker (recall@6 0.84)
  behind the existing seams, `run_full.py` flipped to full-RAG: **qwen 68.0%** vs gemma
  40.0% (**+28.0 pt**); retrieval cost oracle→RAG qwen −12.7 / gemma −15.3 pt. Retrieval
  is the dominant remaining lever for both. **Phase B complete.**

**REMAINING:**
- Grow the SFT set with synthetic grounded Q/A over ingested filings (150 open Qs
  are a seed), then execute the cloud QLoRA run and re-eval (the reward loop).
- Squeeze the last points: 5 qwen RAG rows run away in `reasoning_content` (empty
  answers) — `max_tokens`/stop tuning; and retrieval recall@6 0.84 is the ceiling on
  full-RAG (a bigger-k or better-embedder pass is the next retrieval lever).

## Phase 3 — the headline: novel interpretability  (over time, but it's the point)
- **SAE lab mode** (`eval/`/`interp` layer): Gemma Scope feature-attribution +
  steering over the local model; surface it in the answer/observability view.
  **DEFERRED behind Phase 2.5** (decisions § Session 4): the accuracy winner has no
  matching public SAE (Qwen-Scope = 2B/8B/9B, not 4B), so lab mode follows the
  accuracy push rather than gating the model choice. Still the differentiator, not
  cut — if it later beats the Tier-1 baseline (the S2-Q4 gate), an SAE-covered base
  model becomes a legitimate re-open.
- **The gate:** build the SAE-based hallucination/uncertainty detector and
  **measure it against the Tier-1 logprob/semantic-entropy baseline on the eval
  harness.** Beats baseline ⇒ ship as a real feature; else ⇒ keep as an
  honestly-labeled "lab mode" showcase (decisions § S2-Q4).
- Interpretability Tier-2 (white-box internal-state probe) as the stepping stone.

## Phase 3+ — depth (over time)
- Eval harness: RAGAS + DeepEval CI gate + Promptfoo + Langfuse traces; the
  referee for the SAE-vs-baseline gate above.
- **FinanceBench as a rigor/optimization *loop*** (iterate retrieval/prompts/tools
  against it), reported honestly vs Muse Spark 1.2 / Muse Glimmer 30B; RULER for
  multi-hop stress. Golden set built via deep research; Aditya sole judge; numeric
  answers. The eval harness that drives this loop is **built** (Phase 2.5,
  `benchmarks/financebench/`); this line is now the *continuous* iteration on top.
- XAI auto-eval + human-in-the-loop review flywheel.
- Lakehouse depth (Spark/medallion/dbt/warehouse/Airflow) + `platform` IaC (so
  users stand up their *own* stack) + the demonstrated 10-year backfill.

## Parked sibling projects (memory: portfolio-gap-projects)
NL grid-imbalance forecaster (DE showcase) and a crypto order-book alpha engine —
resurrect later; both are strong streaming-DE portfolio pieces.
