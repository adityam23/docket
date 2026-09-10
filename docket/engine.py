"""Consumer side of the engine's management surface (``/engine/*``).

docket only ever *consumes* the engine's HTTP API — profile definitions via
``GET /engine/profiles``, activation via ``PUT /engine/profiles/active`` (202,
completion arrives on the SSE bus at ``/engine/events``), and per-slot/device
state via ``GET /engine/status`` + ``GET /engine/device``. We never import or
edit engine code (the boundary): feature requests for the engine are written as
contract docs into ``../mettle/agent-docs/``, never code changes here.

Capability-detect once: a backend that answers ``GET /engine/profiles`` is a
mettle-class engine with a profile surface; anything else (llama-server, Ollama,
Cerebras, Groq) 404s it and the whole feature is hidden — the same pattern as
``Capability`` in ``providers/base.py``.

The ``engine`` surface is mounted at the server ROOT (``/engine/*``), NOT under
the OpenAI ``/v1`` base that ``backend_url`` points at — so the base is derived by
stripping the trailing ``/v1`` (see :func:`_engine_base`).
"""

from __future__ import annotations

from typing import Iterator

import httpx

from .config import Settings

_PROBE_TTL = 5.0  # seconds; a positive probe is cached so role: resolution doesn't hammer the engine
_positive_cache: dict[str, float] = {}


def _headers(settings: Settings) -> dict[str, str]:
    key = getattr(settings, "engine_key", None)
    return {"Authorization": f"Bearer {key}"} if key else {}


def _engine_base(settings: Settings) -> str:
    """The engine-management base. ``backend_url`` is the OpenAI-/v1 base
    (e.g. ``http://host/v1``), but mettle mounts the ``/engine/*`` surface at the
    server ROOT of the same host — so strip the trailing ``/v1``. A URL without
    a ``/v1`` suffix is used as-is."""
    url = settings.backend_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


def _get(settings: Settings, path: str, timeout: float = 5.0) -> httpx.Response | None:
    url = _engine_base(settings) + path
    try:
        return httpx.get(url, headers=_headers(settings), timeout=timeout)
    except httpx.HTTPError:
        return None


def _json(resp: httpx.Response | None, default: dict) -> dict:
    return default if resp is None else (resp.json() if resp.status_code == 200 else default)


def engine_available(settings: Settings) -> bool:
    """True when the backend exposes the engine-management surface. Positive
    results are cached briefly; negatives are re-probed (cheap GET)."""
    base = settings.backend_url
    if base in _positive_cache:
        return True
    resp = _get(settings, "/engine/profiles")
    if resp is not None and resp.status_code == 200:
        _positive_cache[base] = 0.0
        return True
    return False


def engine_view(settings: Settings) -> dict:
    """The SPA's window onto the engine: profile definitions + active + status +
    device, or ``{"available": False}`` when the backend has no engine surface."""
    profiles = _get(settings, "/engine/profiles")
    if profiles is None or profiles.status_code != 200:
        return {"available": False}
    out = profiles.json()
    out["available"] = True
    out["status"] = _json(_get(settings, "/engine/status"), {})
    out["device"] = _json(_get(settings, "/engine/device"), {})
    # Positive probe cached so the SPA's repeated GET /api/engine is cheap.
    _positive_cache[settings.backend_url] = 0.0
    return out


def engine_activate(settings: Settings, name: str) -> tuple[int, dict]:
    """Ask the engine to switch to ``name``. Returns ``(status, body)``:
    202 accepted (completion streams on ``engine_events``), 404 unknown profile,
    409 switch already in flight, 503 busy."""
    url = _engine_base(settings) + "/engine/profiles/active"
    try:
        r = httpx.put(url, json={"name": name}, headers=_headers(settings), timeout=10.0)
    except httpx.HTTPError as e:  # noqa: BLE001
        return 0, {"error": str(e)}
    body: dict = {}
    try:
        body = r.json()
    except ValueError:
        pass
    return r.status_code, body


def engine_events_stream(settings: Settings) -> Iterator[str]:
    """Replay the engine's SSE event stream (last 100 replayed on connect by the
    engine). Yields SSE-framed text lines so the web adapter can proxy them
    unchanged to the SPA — the SPA never holds the engine's bearer token."""
    url = _engine_base(settings) + "/engine/events"
    try:
        with httpx.stream("GET", url, headers=_headers(settings), timeout=None) as r:
            if r.status_code != 200:
                yield f'event: error\ndata: {{"status": {r.status_code}}}\n\n'
                return
            for line in r.iter_lines():
                # Drop `event:` lines so every payload is a default `data:`
                # message — the SPA's EventSource.onmessage then catches ALL
                # events (named events would need addEventListener per name).
                # The JSON `data` payload already carries the event type.
                if line.startswith("event:"):
                    continue
                if line:
                    yield line + "\n"
                else:
                    yield "\n"  # SSE event separator (blank line)
    except httpx.HTTPError as e:  # noqa: BLE001
        yield f'event: error\ndata: {{"message": "{e}"}}\n\n'


def resolve_model(settings: Settings, model_id: str) -> str:
    """Resolve a ``role:<kind>`` alias to the active profile's slot id so a
    profile switch automatically re-points chat/embed/rerank. Anything that is
    NOT a ``role:`` alias passes through unchanged, and when the engine surface
    is absent the alias is passed through too (the backend errors, surfaced
    properly) — no silent substitution."""
    if not model_id.startswith("role:"):
        return model_id
    role = model_id[len("role:"):]
    if not engine_available(settings):
        return model_id
    data = _get(settings, "/engine/profiles")
    if data is None or data.status_code != 200:
        return model_id
    active = data.json().get("active")
    slots: list[dict] = []
    for p in data.json().get("profiles", []):
        if p.get("name") == active:
            slots = p.get("slots", [])
            break
    role_slots = [s for s in slots if s.get("role") == role]
    if not role_slots:
        return model_id
    # Highest priority first (mettle orders by priority, then declaration order);
    # slots without a priority rank below all priority-bearing ones.
    role_slots.sort(key=lambda s: (s.get("priority") is None, -(s.get("priority") or 0)))
    return role_slots[0].get("id") or model_id
