"""Local web app (FastAPI): the primary surface.

Two layers:
- a JSON **observability API** under ``/api/*`` — corpus + per-document ingest
  stage, embedding state, redacted config, backend health, a retrieval-traced
  ``ask``, and a live ingest job runner; and
- the **SvelteKit dashboard** (``web/frontend/build``), served as a static SPA
  when built. Every endpoint is a thin transport over the shared service /
  observability functions, so the wiring lives exactly once (CLAUDE.md).
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import queue
import threading
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ..agent.trace import compact_step, hit_view
from ..config import load_settings
from ..engine import (
    engine_activate,
    engine_events_stream,
    engine_view,
    get_supervisor,
)
from ..config_write import (
    ConfigWriteError,
    clear_secret,
    secret_status,
    set_secret,
    update_config,
)
from ..ingest.samples import load_samples
from ..report.render import reliability_banner
from ..service import ask_traced as ask_traced_service
from ..service import chat_store, health as health_service
from ..service import remove_document
from ..service import trace_store
from .jobs import JOBS
from .observability import capacity_stats, config_view, corpus_stats

# Cap browser-uploaded ingests so a stray selection can't exhaust memory/disk.
# _MAX_FILE_BYTES bounds any SINGLE PDF (a huge scan shouldn't be ingestible);
# _MAX_UPLOAD_BYTES bounds the whole selection. The client mirrors _MAX_FILE_BYTES
# (see frontend api.js MAX_PDF_BYTES) for a friendly pre-check — keep them in sync.
_MAX_FILE_BYTES = 50 * 1024 * 1024
_MAX_UPLOAD_BYTES = 200 * 1024 * 1024

app = FastAPI(title="docket", version="0.2.0")


# --- Host/Origin protection (DNS rebinding + CSRF) ---------------------------
#
# The server binds 127.0.0.1 only, but that does not stop a browser tab on any
# other site from reaching it: DNS rebinding points an attacker-controlled
# hostname at 127.0.0.1 (defeats "it's localhost, nothing else can see it"),
# and even without rebinding a plain cross-site request can still hit a
# body-less endpoint (a cross-site <form> needs no JS to POST) or a plain GET
# (an <img>/fetch triggers `GET /api/ask/stream` — inference plus a persisted
# trace). Two independent checks, both needed, kept as ONE implementation each
# rather than duplicated per endpoint:
#
# 1. TrustedHostMiddleware pins the Host header to this server's real loopback
#    names. A rebound hostname fails here, before any route runs — closing the
#    vector that would otherwise let a rebinding page ride this server's
#    forwarded ``Authorization: Bearer <engine_key>`` to the engine.
# 2. `_csrf_guard` rejects state-changing requests (every method but
#    GET/HEAD/OPTIONS, plus the side-effecting `GET /api/ask/stream`) whose
#    declared origin is cross-site. It prefers the `Origin` header (sent by
#    every browser cross-origin fetch/form-post); when `Origin` is absent it
#    falls back to `Sec-Fetch-Site`. Both headers absent is treated as
#    same-origin: that combination is curl/the CLI/TestClient, not a browser
#    — nothing is enforcing same-origin on those callers for an attacker page
#    to spoof, so there is no signal to check.
#
# TestClient's default Host is "testserver" — not a real loopback name, so it
# would otherwise fail check 1 for the whole existing test suite. Tests opt it
# into the allowlist via DOCKET_TEST_TRUSTED_HOST, set once at collection time
# by tests/conftest.py (before any test module imports this module and bakes
# the middleware's allowed_hosts) — deliberately not a `DK_` Settings field, so
# production's allowlist is never configurable away from the real loopback set.
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
# A safe method that is nonetheless side-effecting (runs inference, persists a
# trace) and so needs the same cross-site guard as a state-changing method.
_CSRF_GUARDED_SAFE_PATHS = frozenset({"/api/ask/stream"})


def _is_cross_site(request: Request) -> bool:
    origin = request.headers.get("origin")
    if origin:
        return urlsplit(origin).netloc != request.headers.get("host", "")
    return request.headers.get("sec-fetch-site") == "cross-site"


@app.middleware("http")
async def _csrf_guard(request: Request, call_next):
    guarded = request.method not in _SAFE_METHODS or request.url.path in _CSRF_GUARDED_SAFE_PATHS
    if guarded and _is_cross_site(request):
        return PlainTextResponse("cross-site request blocked", status_code=403)
    return await call_next(request)


_allowed_hosts = ["127.0.0.1", "localhost", "::1", "[::1]"]
if _test_host := os.environ.get("DOCKET_TEST_TRUSTED_HOST"):
    _allowed_hosts.append(_test_host)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=_allowed_hosts)


@app.on_event("startup")
def _start_engine_supervisor() -> None:
    """§3: on startup, probe the engine and spawn mettled when configured and
    unreachable. The supervisor then keeps it alive (crash-restart, bounded
    backoff) for the process's lifetime."""
    get_supervisor(load_settings()).ensure()


def stop_engine_if_ephemeral() -> None:
    """Stop the engine supervisor's process group — but ONLY for the packaged
    app, which opts in via ``DK_ENGINE_EPHEMERAL=1`` (the Tauri sidecar sets it
    at spawn). On the dev path (`dk serve`, incl. its --reload restarts) the
    engine stays warm across reloads: killing mettled per edit would cost a
    full weight reload.

    One gated implementation, two callers: this process's own FastAPI
    ``shutdown`` hook below (the dev-path behaviour, and a safety net for the
    packaged app), and the frozen sidecar's early-signal path
    (``packaging/docket_server_main.py``), which calls this BEFORE uvicorn's
    graceful connection drain rather than after — see that module's docstring
    for why the ordering matters. ``EngineSupervisor.stop()`` is idempotent
    (a no-op once the engine is already stopped), so both callers racing or
    double-calling is harmless.
    """
    if os.environ.get("DK_ENGINE_EPHEMERAL") == "1":
        get_supervisor(load_settings()).stop()


@app.on_event("shutdown")
def _stop_engine_supervisor() -> None:
    """SIGTERM/SIGINT (uvicorn shutdown) stops the engine supervisor's process
    group. See ``stop_engine_if_ephemeral`` for the gating; this hook is the
    dev-path behaviour (and, for the packaged sidecar, a harmless second call
    after its own early-signal stop already ran)."""
    stop_engine_if_ephemeral()


_FRONTEND = Path(__file__).parent / "frontend" / "build"


class AskRequest(BaseModel):
    question: str
    session_id: str | None = None  # T21: run inside a Chat (prior turns threaded)


class IngestRequest(BaseModel):
    folder: str


class UploadFileItem(BaseModel):
    name: str
    content_b64: str  # base64-encoded PDF bytes (browser reads the file locally)


class UploadRequest(BaseModel):
    files: list[UploadFileItem]


class ConfigUpdate(BaseModel):
    # Only whitelisted non-secret settings; validated by config_write.update_config.
    updates: dict[str, object]


class KeyUpdate(BaseModel):
    provider: str            # "engine" (config_write.SECRET_ENV)
    value: str | None = None  # set when present; clear when null/empty


class EngineProfileRequest(BaseModel):
    name: str


def _answer_payload(question: str, *, session_id: str | None = None, on_step=None) -> tuple[dict, dict]:
    """Run the agent (persisting a trace) and shape the answer payload.

    Returns ``(payload, trace)``. The hit projection is shared with the trace via
    ``hit_view`` (one shape — CLAUDE.md). Used by ``/ask`` (back-compat),
    ``/api/ask``, and the SSE stream (which passes a live ``on_step`` sink).
    With ``session_id`` the ask runs inside a Chat: prior turns are threaded as
    model context and the trace nests under the chat (T21).
    """
    result, trace = ask_traced_service(question, on_step=on_step, session_id=session_id)
    payload = {
        "question": result.question,
        "answer": result.answer,
        "reliability": result.reliability,
        "banner": reliability_banner(result.reliability),
        "surprisal": trace["surprisal"],
        "detail": result.detail,  # aggregation breakdown (op/result/unit/per-item) or None
        "citations": [c.model_dump() for c in result.citations],
        "hits": [hit_view(h) for h in (result.hits or [])],
        "queries": result.queries or [question],
        "hops": len(result.queries or [question]),
        "elapsed_ms": trace["elapsed_ms"],
        "trace_id": trace["id"],
        "chat_id": trace.get("chat_id"),
    }
    return payload, trace


# --- Observability API -------------------------------------------------------

@app.get("/api/health")
def api_health() -> dict:
    return health_service()


@app.get("/api/config")
def api_config() -> dict:
    return config_view()


@app.get("/api/corpus")
def api_corpus() -> dict:
    return corpus_stats()


@app.get("/api/capacity")
def api_capacity() -> dict:
    """How many more documents fit given this device's disk (and its specs)."""
    return capacity_stats()


@app.delete("/api/corpus/{doc_id}")
def api_remove_document(doc_id: str) -> dict:
    """Remove one document (all its chunks + vectors) from the local index."""
    if JOBS.is_running():
        raise HTTPException(status_code=409, detail="an ingest job is running; try again once it finishes")
    removed = remove_document(doc_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"no such document: {doc_id}")
    return {"doc_id": doc_id, "removed_chunks": removed}


@app.post("/api/samples/load")
def api_load_samples() -> dict:
    """Opt-in: add the bundled synthetic sample documents to the index."""
    if JOBS.is_running():
        raise HTTPException(status_code=409, detail="an ingest job is running; try again once it finishes")
    load_samples()
    return corpus_stats()


@app.get("/api/overview")
def api_overview() -> dict:
    """One call for the dashboard landing: health + corpus totals + config."""
    stats = corpus_stats()
    return {
        "health": health_service(),
        "config": config_view(),
        "totals": stats["totals"],
        "retrieval": stats["retrieval"],
        "capacity": capacity_stats(),
        "index_dir": stats.get("index_dir"),
    }


def _require_session(session_id: str) -> None:
    """404 early for an unknown Chat session (before any inference starts)."""
    if session_id and chat_store().get(session_id) is None:
        raise HTTPException(status_code=404, detail=f"no such chat session: {session_id}")


@app.post("/api/ask")
def api_ask(req: AskRequest) -> dict:
    _require_session(req.session_id)
    payload, _ = _answer_payload(req.question, session_id=req.session_id)
    return payload


@app.get("/api/ask/stream")
def api_ask_stream(question: str, session_id: str | None = None) -> StreamingResponse:
    """Answer while streaming pipeline steps over SSE (T07 live timeline).

    The blocking agent runs on a worker thread; its ``on_step`` sink pushes compact
    step events onto a queue that this generator drains as ``event: step`` frames,
    then a final ``event: done`` carries the full answer payload (with ``trace_id``
    for the per-chat page). The non-streaming ``/api/ask`` stays for tests/compat.
    With ``session_id`` the question joins that Chat and prior turns are threaded
    as context server-side (T21 — history is never accepted from the client).
    """
    _require_session(session_id)

    events: "queue.Queue[tuple[str, dict] | None]" = queue.Queue()

    def on_step(kind: str, info: dict) -> None:
        events.put(("step", compact_step(kind, info)))

    def run() -> None:
        try:
            payload, _ = _answer_payload(question, session_id=session_id, on_step=on_step)
            events.put(("done", payload))
        except Exception as e:  # noqa: BLE001 - surface failure to the client as an event
            events.put(("error", {"detail": str(e)}))
        finally:
            events.put(None)

    threading.Thread(target=run, name="ask-stream", daemon=True).start()

    def frames():
        while True:
            item = events.get()
            if item is None:
                break
            event, data = item
            yield f"event: {event}\ndata: {json.dumps(data)}\n\n"

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/traces")
def api_traces() -> dict:
    """Recent per-chat trace summaries, newest first (T04/T05)."""
    return {"traces": trace_store().list()}


@app.get("/api/traces/{trace_id}")
def api_trace(trace_id: str) -> dict:
    """The full trace for one chat: every prompt, response and retrieval hop."""
    trace = trace_store().get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"no such trace: {trace_id}")
    return trace


@app.delete("/api/traces/{trace_id}")
def api_trace_delete(trace_id: str) -> dict:
    """Delete one trace (a one-shot question from the audit history)."""
    if not trace_store().delete(trace_id):
        raise HTTPException(status_code=404, detail=f"no such trace: {trace_id}")
    return {"deleted": trace_id}


@app.get("/api/chats")
def api_chats() -> dict:
    """First-class Chat/Session records (T21): each groups its ordered turns;
    every turn keeps its own trace (chat → turns → trace)."""
    return {"chats": chat_store().list()}


class ChatCreate(BaseModel):
    title: str | None = None  # defaults to the first turn's question


@app.post("/api/chats")
def api_chat_create(req: ChatCreate) -> dict:
    """Mint a new (empty) Chat session. The Ask surface calls this lazily before
    its first question; later questions thread their turns into it server-side."""
    from ..agent.trace import _utc_now

    store = chat_store()
    now = _utc_now()
    chat_id = store.next_id(now)
    # Untitled until its first turn names it (the first question).
    return store.create(chat_id, req.title or "", now=now)


@app.get("/api/chats/{chat_id}")
def api_chat(chat_id: str) -> dict:
    """One Chat with its ordered turn summaries."""
    chat = chat_store().get(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail=f"no such chat session: {chat_id}")
    return chat


@app.delete("/api/chats/{chat_id}")
def api_chat_delete(chat_id: str) -> dict:
    """Remove a Chat and cascade-delete every turn's trace, so a deleted chat
    leaves nothing behind (its turns would otherwise resurface as one-shots)."""
    if not chat_store().delete(chat_id):
        raise HTTPException(status_code=404, detail=f"no such chat session: {chat_id}")
    removed = trace_store().delete_by_chat(chat_id)
    return {"deleted": chat_id, "traces_removed": removed}


@app.patch("/api/config")
def api_config_update(req: ConfigUpdate) -> dict:
    """Persist non-secret runtime settings from the Settings page (T01).

    Written to ``.env.local`` (overlays ``.env``); the change is picked up on the
    next request — no restart. Returns the fresh redacted config view.
    """
    try:
        update_config(req.updates)
    except ConfigWriteError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return config_view()


@app.post("/api/config/keys")
def api_config_keys(req: KeyUpdate) -> dict:
    """Set or clear a provider API key (T02 BYOK). Never echoes the value back —
    returns only which keys are now set."""
    try:
        if req.value:
            set_secret(req.provider, req.value)
        else:
            clear_secret(req.provider)
    except ConfigWriteError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return {"api_keys": secret_status()}


# --- Engine control (mettle `/engine/*` consumer) ----------------------------


@app.get("/api/engine")
def api_engine() -> dict:
    """The SPA's window onto the engine's profile surface: definitions + active
    profile + per-slot status + device VRAM, or ``{"available": False}`` when the
    backend isn't a mettle-class engine (the feature is then hidden)."""
    return engine_view(load_settings())


@app.put("/api/engine/profile")
def api_engine_profile(req: EngineProfileRequest) -> dict:
    """Switch the engine's active profile. 202 = accepted (progress streams on
    ``/api/engine/events``); the engine maps the outcome to a profile event."""
    status, body = engine_activate(load_settings(), req.name)
    if status == 202:
        return {"accepted": True, **body}
    if status in (404, 409, 503):
        raise HTTPException(status_code=status, detail=body.get("error") or body)
    raise HTTPException(status_code=502, detail=body.get("error") or f"activation failed ({status})")


@app.get("/api/engine/supervision")
def api_engine_supervision() -> dict:
    """The supervisor's honest view of mettled's lifecycle: ``offline`` /
    ``starting`` / ``available`` / ``failed`` — the SPA renders engine state
    instead of guessing (cutover contract §3/§5: never a silent substitute)."""
    return get_supervisor(load_settings()).view()


@app.post("/api/engine/restart")
def api_engine_restart() -> dict:
    """Manual restart — the only exit from the supervisor's terminal
    ``failed`` state. Resets the backoff budget and kicks the loop; the loop's
    next cycle decides the state."""
    get_supervisor(load_settings()).restart()
    return get_supervisor(load_settings()).view()

@app.get("/api/engine/events")
def api_engine_events() -> StreamingResponse:
    """Proxy the engine's SSE event stream (replays the last 100 on connect) to
    the SPA, which drives live activation progress. The SPA never talks to the
    engine or holds its bearer token."""
    return StreamingResponse(engine_events_stream(load_settings()), media_type="text/event-stream")


@app.post("/api/ingest")
def api_ingest(req: IngestRequest) -> dict:
    """Kick off a background ingest of every PDF under ``folder`` and return the
    initial job state (poll ``/api/ingest/status`` for live per-document stages)."""
    try:
        job = JOBS.start(req.folder)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return job.as_dict()


@app.post("/api/ingest/upload")
def api_ingest_upload(req: UploadRequest) -> dict:
    """Ingest PDFs the user picked with the in-browser folder chooser.

    The browser reads files locally and sends them base64-encoded (no server-side
    filesystem browsing), keeping the path-agnostic, privacy-respecting UX.
    """
    decoded: list[tuple[str, bytes]] = []
    total = 0
    for item in req.files:
        try:
            data = base64.b64decode(item.content_b64, validate=True)
        except (binascii.Error, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"bad file encoding for {item.name}") from e
        if len(data) > _MAX_FILE_BYTES:
            mb = _MAX_FILE_BYTES // (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=f"{item.name} exceeds the {mb} MB per-file limit",
            )
        total += len(data)
        if total > _MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="selection exceeds the upload size limit")
        decoded.append((item.name, data))
    try:
        job = JOBS.start_uploads(decoded)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return job.as_dict()


@app.get("/api/ingest/status")
def api_ingest_status() -> dict:
    """State of the most recent ingest job (``null`` job if none has run)."""
    return {"running": JOBS.is_running(), "job": JOBS.latest()}


# --- Back-compat endpoints (kept for the CLI/tests and simple integrations) ---

@app.get("/health")
def health() -> dict:
    return health_service()


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    """Grounded, cited answer + Tier-1 reliability label for a question."""
    payload, _ = _answer_payload(req.question)
    return payload


# --- SvelteKit SPA (static) --------------------------------------------------

if (_FRONTEND / "_app").is_dir():
    app.mount("/_app", StaticFiles(directory=_FRONTEND / "_app"), name="assets")


@app.get("/", response_model=None)
@app.get("/{path:path}", response_model=None)
def spa(path: str = ""):
    """Serve the built dashboard; deep links fall back to the SPA entrypoint.

    Registered last so it never shadows the ``/api/*`` routes. When the frontend
    has not been built yet, returns a friendly page pointing at the build step.
    """
    index = _FRONTEND / "index.html"
    if index.is_file():
        candidate = (_FRONTEND / path).resolve()
        # Serve a real static file (favicon, etc.) if it exists and is inside build.
        if path and _FRONTEND.resolve() in candidate.parents and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index)
    return HTMLResponse(_UNBUILT_HTML)


_UNBUILT_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>docket</title>
<style>body{font:16px/1.6 system-ui,sans-serif;max-width:640px;margin:4rem auto;padding:0 1rem;color:#e8e8ea;background:#0b0c10}
code{background:#1a1c22;padding:.15rem .4rem;border-radius:5px;color:#8ad3ff}a{color:#8ad3ff}</style></head><body>
<h1>docket — dashboard not built</h1>
<p>The observability API is live at <code>/api/*</code>, but the SvelteKit dashboard
has not been built yet. Build it once:</p>
<pre><code>cd docket/web/frontend
npm install
npm run build</code></pre>
<p>Then reload. During development run <code>npm run dev</code> for hot reload
(it proxies <code>/api</code> to this server).</p>
</body></html>"""
