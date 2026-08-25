# infengine model profiles — consumer side (2026-08-24)

infengine is gaining **model profiles**: named, runtime-switchable collections
of models (chat / embed / rerank / ocr + draft attachments) that fit the
available VRAM, with graceful degradation instead of crashes. **Design is
agreed; not implemented yet.** The authoritative engine-side contract is
`infengine/agent-docs/profiles-contract.md`; this document is only what
docket must know and eventually adopt. Nothing here requires code
changes until the engine ships the surface.

Not related to the *ingestion* profiles (platform / lite) in
[architecture.md](architecture.md) - different word, different thing.

## What stays true

- The backend boundary is unchanged: `/v1` OpenAI shapes, `DK_BACKEND_URL`,
  one client class (`providers/openai_compat.py`). llama-server, Ollama and
  infengine remain interchangeable.
- `/v1` request/response shapes do not change. Three additive conveniences
  arrive (below); all are ignorable by older clients.

## What's new on `/v1` (additive)

1. **Role aliases**: `"model": "role:embed"` resolves to the active profile's
   highest-priority embed slot. Useful for "whatever embedder is active"
   without hardcoding ids. Explicit model ids keep working exactly as today.
2. **Structured unavailability**: a request targeting a slot the active
   profile dropped returns `503` with
   `{error: {code: "slot_unavailable", reason: "vram", profile_degraded: true, missing: [...]}}`.
   Clients should surface the reason, not retry blindly.
3. **`/v1/models` extension fields**: `loaded: bool` and `vram_bytes` per
   catalog entry. OpenAI SDKs ignore unknown fields; our model picker can use
   them to grey out unloaded models.

## The opt-in `/engine` surface (infengine-only)

Bearer-authed like `/v1`. **Any other backend (llama-server, Ollama, Cerebras,
Groq) 404s these paths** - capability-detect once (`GET /engine/profiles`) and
hide the whole feature when absent, same pattern as the `Capability` enum in
`providers/base.py`.

| Call | Use |
|------|-----|
| `GET /engine/profiles` | Profile definitions + active + per-profile fit preview (bytes) — the profile picker's data source |
| `PUT /engine/profiles/active` | Switch profile (async; completion arrives via events) |
| `GET /engine/status` | Per-slot state, est+measured VRAM — status page |
| `GET /engine/device` | VRAM totals/free — capacity display |
| `POST /engine/slots/{id}/load` / `unload` | Ad-hoc load/unload outside the active profile |
| `GET /engine/events` | SSE, replays last 100 on connect |

Events: `profile.activating|active|degraded{dropped, shrunk, reason}`,
`slot.loaded|unloaded|failed{reason}`, `vram.pressure{free_bytes}`.

## Consumer responsibilities when this ships

- **Profile picker UI**: list profiles with fit previews; activate; show
  switch progress from `profile.activating` → `profile.active`.
- **Degradation surfacing**: `profile.degraded` must become a visible state
  ("OCR unavailable in this profile"), never a silent capability loss.
- **503 handling**: on `slot_unavailable`, show the reason; offer switching to
  a profile that includes the missing slot.
- **Mid-stream failures**: an evicted slot's stream ends with an in-band SSE
  error event, not a silent close — the stream reader must parse terminal
  error events and preserve chat history (only the in-flight response is
  lost; the engine guarantees unrelated slots are never touched).
- **No polling loops**: subscribe to `/engine/events` once; it replays state
  on connect.

## SAE note

The engine's deferred `sae` role is unrelated to Tier-3 SAE lab mode
(architecture.md): that runs Gemma Scope SAEs via TransformerLens on local
weights in the app, not as an engine-served model. If the engine ever gains a
sae role, it will be a new capability announcement, not a change to Tier-3.
