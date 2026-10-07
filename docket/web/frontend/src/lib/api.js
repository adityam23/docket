// Thin client over the FastAPI observability API. Same-origin in production
// (FastAPI serves this bundle); dev proxies /api → :8760 (see vite.config.js).

// Per-file PDF ceiling — mirrors the server's _MAX_FILE_BYTES (web/app.py). Used
// for a friendly client-side pre-check so an oversize file is rejected before we
// waste memory base64-encoding it. Keep in sync with the server constant.
export const MAX_PDF_BYTES = 50 * 1024 * 1024;

// One error shape everywhere: every fetch helper below throws a plain Error
// whose `.message` is already the clean server/HTTP message — never a raw
// "Error: " toString(), never a dumped response blob. Call sites read it with
// `errMsg(e)` instead of each re-deriving it via their own String(e).replace(...).
export function errMsg(e) {
	return e?.message ?? String(e);
}

async function toJSON(r) {
	try {
		return await r.json();
	} catch {
		return null;
	}
}

// Shared request core (one concept, one implementation — GET and POST/PATCH/
// DELETE differ only in fetch() options). On a non-ok response, prefer the
// FastAPI HTTPException `{"detail": "<message>"}` shape; fall back to the raw
// HTTP status when the body isn't that shape (e.g. a 422 validation array).
async function request(path, opts) {
	const r = await fetch(path, opts);
	const data = await toJSON(r);
	if (!r.ok) {
		const detail = data && typeof data.detail === 'string' ? data.detail : null;
		throw new Error(detail || `${r.status} ${r.statusText}`.trim());
	}
	return data ?? {};
}

function jsonOpts(method, body) {
	return {
		method,
		headers: { 'Content-Type': 'application/json' },
		body: body === undefined ? undefined : JSON.stringify(body)
	};
}

const getJSON = (path) => request(path);
const postJSON = (path, body) => request(path, jsonOpts('POST', body));
const patchJSON = (path, body) => request(path, jsonOpts('PATCH', body));
const del = (path) => request(path, jsonOpts('DELETE'));

// One way to read a message out of a browser `error` Event dispatched by an
// EventSource (shared by every SSE consumer below — CLAUDE.md: one
// implementation). A server-sent `event: error` frame carries JSON `.data`
// (`{"detail"|"message": "..."}`, framed by web/sse.py's `sse()`); a bare
// network-level connection drop carries none, so callers get `fallback`
// instead of us inventing a reason we were never given.
function sseErrorMessage(e, fallback) {
	if (e?.data) {
		try {
			const d = JSON.parse(e.data);
			return d.detail || d.message || fallback;
		} catch {
			/* not JSON — keep fallback */
		}
	}
	return fallback;
}

// Read a File (from a folder picker) into a base64 payload for /api/ingest/upload.
function fileToB64(file) {
	return new Promise((resolve, reject) => {
		const reader = new FileReader();
		reader.onerror = () => reject(reader.error);
		reader.onload = () => {
			// result is a data URL: "data:...;base64,XXXX" — keep the payload.
			const s = String(reader.result);
			resolve({ name: file.name, content_b64: s.slice(s.indexOf(',') + 1) });
		};
		reader.readAsDataURL(file);
	});
}

export const api = {
	overview: () => getJSON('/api/overview'),
	corpus: () => getJSON('/api/corpus'),
	capacity: () => getJSON('/api/capacity'),
	config: () => getJSON('/api/config'),
	health: () => getJSON('/api/health'),
	// Non-secret settings (Settings page, T01) → .env.local overlay; returns the
	// fresh redacted config view. Provider API keys (BYOK) never echo the value
	// back — only which keys are now set (web/app.py PATCH /api/config, POST
	// /api/config/keys).
	updateConfig: (updates) => patchJSON('/api/config', { updates }),
	setKey: (provider, value) => postJSON('/api/config/keys', { provider, value }),
	ingestStatus: () => getJSON('/api/ingest/status'),
	removeDoc: (docId) => del(`/api/corpus/${encodeURIComponent(docId)}`),
	loadSamples: () => postJSON('/api/samples/load'),
	async ingestFiles(fileList) {
		const files = await Promise.all(Array.from(fileList).map(fileToB64));
		return postJSON('/api/ingest/upload', { files });
	},

	// --- traces + chat sessions (observability; T21 nesting) ---
	traces: () => getJSON('/api/traces'),
	trace: (id) => getJSON(`/api/traces/${encodeURIComponent(id)}`),
	deleteTrace: (id) => del(`/api/traces/${encodeURIComponent(id)}`),
	chats: () => getJSON('/api/chats'),
	deleteChat: (id) => del(`/api/chats/${encodeURIComponent(id)}`),
	// Mint an empty Chat session (T21); the first turn names it.
	createChat: () => postJSON('/api/chats', {}),

	// --- engine profile surface (mettle /engine/*; `available:false` when absent) ---
	// The SPA never talks to the engine directly — the server proxies it and holds
	// the bearer, so no CORS / secret exposure (see web/app.py).
	engine: () => getJSON('/api/engine'),
	engineProfile: (name) => request('/api/engine/profile', jsonOpts('PUT', { name })),
	// Supervisor's honest lifecycle view (F3): offline/starting/available/
	// degraded/incompatible/failed + phase/reason/detail. Polled by
	// stores/supervision.js — one poller, every consumer reads the store.
	engineSupervision: () => getJSON('/api/engine/supervision'),
	// The only exit from `failed` and from the port-held `offline` state.
	// Returns the fresh supervision view (same shape as engineSupervision()).
	engineRestart: () => postJSON('/api/engine/restart'),
	// Live profile-switch progress: replayed SSE (last 100 events on connect).
	// Returns a cancel fn; onEvent fires per SSE event (JSON data payloads).
	// onError gets a human reason (never silently invisible — F3): the proxy
	// sends one `event: error` frame with a message before closing on failure;
	// a bare connection drop reports a generic fallback instead.
	engineEvents(onEvent, onError) {
		const es = new EventSource('/api/engine/events');
		es.onmessage = (e) => { try { onEvent(JSON.parse(e.data)); } catch { onEvent(e.data); } };
		es.addEventListener('error', (e) => {
			onError?.(sseErrorMessage(e, 'connection lost'));
		});
		return () => es.close();
	},

	// Live answer over SSE (T07). With `sessionId` (T21) the question joins that
	// Chat and prior turns are threaded server-side. Calls onStep for each
	// pipeline step, onDone with the final payload (incl. trace_id/chat_id),
	// onError on failure. Returns a cancel fn.
	askStream(question, sessionId, { onStep, onDone, onError }) {
		const params = new URLSearchParams({ question });
		if (sessionId) params.set('session_id', sessionId);
		const es = new EventSource(`/api/ask/stream?${params}`);
		let finished = false;
		es.addEventListener('step', (e) => {
			try {
				onStep(JSON.parse(e.data));
			} catch {
				/* ignore malformed frame */
			}
		});
		es.addEventListener('done', (e) => {
			finished = true;
			try {
				onDone(JSON.parse(e.data));
			} finally {
				es.close();
			}
		});
		es.addEventListener('error', (e) => {
			// Ignore errors after we've already received `done` (the server closing
			// the stream post-done also fires `error`, which isn't a real failure).
			if (finished) return;
			finished = true;
			onError(sseErrorMessage(e, 'stream connection error'));
			es.close();
		});
		return () => {
			finished = true;
			es.close();
		};
	}
};
