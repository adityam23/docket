# docket

**A self-hostable finance-filings research assistant.** Live-ingest SEC filings,
ask anything across your whole coverage universe, and get a **cited, confidence-
scored** answer in seconds — running on your own hardware, with a **glass-box view
of *why* the model answered** (SAE feature-attribution "lab mode") that the SaaS
tools don't show.

> The headline is **transparency**: a novel interpretability layer, not a
> "small model beats frontier" claim. The model is small and local because
> white-box interpretability needs the weights (and because it fits 6 GB VRAM);
> the retrieval/agent/eval tooling is what makes it good enough. We benchmark on
> **FinanceBench** and report it *honestly* against a frontier baseline.

Anchored to a concrete business case — a buy-side analyst who must read every
filing before the alpha decays (see [`docs/case-study.md`](docs/case-study.md)) —
so it's a real product, not a tech demo. Driven by real hiring-market demand
(`agent` + `eval` are the two loudest signals across ~1000 analysed job postings);
**not** a rehash of prior work. See [`docs/`](docs/index.md) for the full design;
read **[`docs/decisions.md`](docs/decisions.md) § Session 2** for the reasoning.

## Status

**Phase 1 (lite MVP) plumbing DONE** — ingest → retrieve → agent → answer runs
end to end, offline-verified (20 pytest) and against a live backend (`dk chat`,
`dk ask` with citations + 🟢/🟡/🔴 reliability from real per-token logprobs).
Scanned-page OCR is wired through the backend's `/v1/ocr` (`ingest/_ocr_model.py`).
Remaining per-phase backlog (embedding endpoint E2E, scanned-PDF verification,
turbovec, reranker, eval harness, install/pages): `docs/roadmap.md`.

## Quickstart (dev)

```bash
uv sync                       # base deps only (fast/light)
uv run dk health              # check the local /v1 backend is reachable
uv run dk chat "hello"        # one-shot chat smoke test
uv run pytest                 # offline smoke tests
uv run dk serve               # dashboard + API on http://127.0.0.1:8760
```

Optional layers: `uv sync --extra agent` (LangGraph), `uv sync --extra ingest`.

## Dashboard (SvelteKit)

The primary surface is a SvelteKit observability dashboard (Svelte 5 runes,
built to a static SPA that FastAPI serves — no Node at runtime):

- **Overview** — document/chunk counts, embedding coverage, backend + model status.
- **Documents** — every indexed doc, the ingest **stage** it reached
  (queued → ocr → chunk → embed → index → embedded), coverage meters, and a live
  folder-ingest runner that streams per-document stage transitions.
- **Ask** — grounded, cited answers with the 🟢/🟡/🔴 reliability banner, token
  surprisal, hop count, and the full **retrieval trace** (ranked hits + scores).
- **Observability** — runtime config, retrieval knobs, embedding + backend health.

Build it once (needs Node ≥ 18); FastAPI then serves it at `/`:

```bash
cd docket/web/frontend
npm install && npm run build      # → build/ (served by `dk serve`)
npm run dev                       # optional: hot-reload dev, proxies /api → :8760
```

The JSON API lives under `/api/*` (`/api/overview`, `/api/corpus`, `/api/config`,
`/api/ask`, `/api/ingest` + `/api/ingest/status`) — usable headless too.

## Backend

Any OpenAI-compatible `/v1` server works — the stock llama.cpp `llama-server`,
the [infengine](../infengine) Rust engine, or Ollama — configured via
`DK_BACKEND_URL` (default `http://127.0.0.1:11434/v1`). Switch to a free-tier API
with `DK_PROVIDER=cerebras|groq` + the matching key. Embeddings need a **dedicated**
endpoint (`DK_EMBED_URL`) — the chat server is not an embedding server. See
`docs/architecture.md`.

## Docs

- `docs/decisions.md` — the finalised design in two grilling sessions; **read
  Session 2 (2026-08-19, the case-study pivot) first.**
- `docs/case-study.md` — the finance business case: user, pain, product mapping,
  hosting topology, demo script, golden set. **What the engine is *for*.**
- `docs/architecture.md` — components, the `/v1` boundary, the two profiles, the
  hosting topology + retention.
- `docs/stack.md` — the SOTA stack (verified Aug 2026) + phase-gated deps.
- `docs/roadmap.md` — phased build plan + backlog.

## License

GPL-3.0 — see [`LICENSE`](LICENSE).
