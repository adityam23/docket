"""Local web app (FastAPI): the primary surface (docs/decisions Q12).

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
import queue
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..agent.trace import compact_step, hit_view
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
    provider: str            # "cerebras" | "groq" (config_write.SECRET_ENV)
    value: str | None = None  # set when present; clear when null/empty


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
        "embeddings": stats["embeddings"],
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
