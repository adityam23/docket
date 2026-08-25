# CLAUDE.md

Guidance for Claude Code (and any agent) working in this repository.

## Guiding principle (overrides everything below)

**Clean, modular, secure code.** One concept → one implementation, lifted into a
shared module. Never write three near-identical helpers local to three files —
that is spaghetti. Before adding a function, look for an existing one to reuse or
extend; if two blocks resemble each other, extract the common core. Refactor
freely to keep it that way. Security is a first-class concern: validate inputs,
never trust external content, keep secrets in config/env, and preserve the
backend boundary. Prefer deleting duplication over adding cleverness.

## Dependencies (surface the trade-off; never auto-decide)

Don't reinvent the wheel: when a complex feature is cleanly provided by a
lightweight library (stdlib or third-party), prefer it over a bespoke
implementation — **but do not adopt it silently, and do not reflexively hand-roll
either.** Surface the choice to the user with the concrete trade-off: **how many
*other* places in the codebase could reuse that library vs. a heavy library
pulled in for a single feature.** A broadly-reusable dependency (e.g. one that
also covers crypto/TLS/serialization elsewhere) is easy to justify; a heavy,
single-purpose dependency is not. The user judges whether the dependency is worth
it — present the reuse-breadth-vs-weight call, then let them decide.

## Mission

**A self-hostable finance-filings research assistant** over SEC EDGAR:
live-ingested, cited, confidence-scored answers across an analyst's whole coverage
universe, with a **novel glass-box interpretability layer (SAE feature-attribution
"lab mode") as the headline differentiator.** The retrieval/agent/eval tooling is
what makes a **small, local** model good enough; the model is small because
(a) white-box interpretability is impossible over a hosted API and (b) the target
hardware is 6 GB VRAM — **not** because "small rivals frontier" is the pitch.

This is anchored to a concrete business case (see `docs/case-study.md`) so it is
**not a solution looking for a problem** — a buy-side analyst who must read every
filing before the alpha decays. Every capability maps to a line item of that
workflow; if a feature doesn't, it doesn't belong. Chosen from real hiring-market
demand (`agent` + `eval` dominate ~1000 analysed postings); **do not** recycle
prior projects, and **do not** drift back toward a domain-agnostic engine.

**Anti-slop rule (the operator's #1 concern):** the whole point is *real
engineering*, not "vibe-coded slop a C-suite could have prompted." Never fake a
capability to look done — a disclosed cached example is fine; a simulation
pretending to be the running system is forbidden (it destroys the entire thesis).

Read `docs/decisions.md` before making design changes — **Session 2 (2026-08-19)
is the current identity** and supersedes parts of Session 1's Q1–Q18. They were
reached through deliberate grilling; don't silently revisit them — if one needs
reopening, say so explicitly. `docs/case-study.md` is what the engine is *for*.
The frontend design language is settled in **Session 3 (2026-08-21)** and
specified in `docs/design-language.md` — do not drift back toward generic
dark-SaaS/"AI slop". **Session 4 (2026-08-22)** chose the generator base
(**Qwen3.5-4B Claude-Opus distill**, on measured FinanceBench data) and **deferred
the SAE "lab mode" behind an accuracy push** — this is *sequencing*, not a cut: the
SAE is still the headline differentiator (accuracy is banked first). The
FinanceBench measurement+training loop lives in `benchmarks/financebench/`; the
next dominant lever is **retrieval** (TODO T26). The former Ask conversation-context
gap is **resolved** (first-class Chat/Session, TODO T21 done).

## Always use uv

**Every Python invocation goes through `uv` — no bare `python`/`pip`.**

```bash
uv sync                      # base deps (fast/light)
uv sync --extra agent        # + LangGraph agent layer
uv run dk health             # CLI
uv run pytest                # tests
uv run uvicorn docket.web.app:app --port 8760
```

## Reusability (hard rule)

One concept → one implementation. This mirrors the existing design: there is
**one** OpenAI-`/v1` client (`providers/openai_compat.py`) reused for every
backend (local llama-server/infengine/Ollama **and** Cerebras/Groq) — the router
parameterises base_url/key, it does not fork a class per provider. Before writing
a second block that resembles an existing one, extract a shared helper.

## Backend boundary (the infengine reconciliation)

This app **never imports infengine's code.** It depends on a backend only across
the **OpenAI-compatible `/v1` HTTP contract** (`DK_BACKEND_URL`). So infengine
can stay half-finished forever — we only use the part that already works:
single-user chat with **per-token logprobs** (`/v1/chat/completions`), embeddings
(`/v1/embeddings`), and scanned-page OCR (`/v1/ocr` via `ingest/_ocr_model.py`).
The stock llama.cpp `llama-server` and Ollama speak the same `/v1`, so any of
them is a drop-in backend. Keep this boundary clean.

## Space & the shared server

The dev host is disk-constrained (~33 GB free) and runs a **persistent
infengine backend** (llama.cpp engine, `gemma4:e2b` MTP + `unlimited-ocr` +
`embed-gemma`) on `127.0.0.1:11434` — **do not kill it** and do not download
multi-GB models/deps without need. Embeddings for non-infengine backends need a
*separate* dedicated endpoint, not the chat server.

## Testing

Offline & deterministic — no network, no live backend, no real LLM in `pytest`
(a live backend is exercised by `dk health`/`dk chat` during handoff, not in the
suite). DB/artifacts (when added) use temp dirs via the real schema-init.

## Layout

`docket/` = `config` · `providers/` (the `/v1` abstraction) · `ingest/`
(ocr/chunk/embed/index + `feeds/`) · `retrieval/` · `agent/` (graph + tools) ·
`trust/` (reliability) · `eval/` (harness + xai) · `report/` · `web/` · `cli`.
`platform/` = self-host IaC (Kafka/Spark). `benchmarks/` = RULER/FinanceBench.
`docs/` = the finalised design; `docs/design-language.md` = the canonical
frontend design spec — authentic mobile Windows Phone "Metro" (Microsoft Design
Language), Apple-polished; read before touching `docket/web/frontend/`.
Backlog lives in `docs/roadmap.md`.
