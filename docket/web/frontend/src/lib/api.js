// Thin client over the FastAPI observability API. Same-origin in production
// (FastAPI serves this bundle); dev proxies /api → :8760 (see vite.config.js).

// Per-file PDF ceiling — mirrors the server's _MAX_FILE_BYTES (web/app.py). Used
// for a friendly client-side pre-check so an oversize file is rejected before we
// waste memory base64-encoding it. Keep in sync with the server constant.
export const MAX_PDF_BYTES = 50 * 1024 * 1024;

async function getJSON(path) {
	const r = await fetch(path);
	if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
	return r.json();
}

async function sendJSON(path, method, body) {
	const r = await fetch(path, {
		method,
		headers: { 'Content-Type': 'application/json' },
		body: body === undefined ? undefined : JSON.stringify(body)
	});
	const data = await r.json().catch(() => ({}));
	if (!r.ok) throw new Error(data.detail || `${r.status}`);
	return data;
}

const postJSON = (path, body) => sendJSON(path, 'POST', body);
const patchJSON = (path, body) => sendJSON(path, 'PATCH', body);
const del = (path) => sendJSON(path, 'DELETE');

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
	ask: (question, sessionId = null) =>
		postJSON('/api/ask', sessionId ? { question, session_id: sessionId } : { question }),
	ingest: (folder) => postJSON('/api/ingest', { folder }),
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
	chat: (id) => getJSON(`/api/chats/${encodeURIComponent(id)}`),
	deleteChat: (id) => del(`/api/chats/${encodeURIComponent(id)}`),
	// Mint an empty Chat session (T21); the first turn names it.
	createChat: () => postJSON('/api/chats', {}),

	// --- settings write path (T01) + BYOK keys (T02) ---
	updateConfig: (updates) => patchJSON('/api/config', { updates }),
	// value null/'' clears the key; the server never echoes the secret back.
	setKey: (provider, value) => postJSON('/api/config/keys', { provider, value }),

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
			// A server-sent `error` event carries JSON; a browser connection drop does
			// not. Ignore errors after we've already received `done`.
			if (finished) return;
			finished = true;
			let msg = 'stream connection error';
			if (e.data) {
				try {
					msg = JSON.parse(e.data).detail || msg;
				} catch {
					/* keep default */
				}
			}
			onError(msg);
			es.close();
		});
		return () => {
			finished = true;
			es.close();
		};
	}
};
