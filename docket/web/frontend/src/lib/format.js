// Presentation helpers shared across views (one place — reuse rule).
// No emoji as UI icons (design-language §8): step/stage glyphs are names into
// the shared Icon set; trust keeps coloured DOTS via chip tones only.

export function num(n) {
	if (n === null || n === undefined) return '—';
	return n.toLocaleString('en-US');
}

export function pct(x) {
	if (x === null || x === undefined) return '—';
	return `${Math.round(x * 100)}%`;
}

export function bytesish(chars) {
	if (!chars) return '0';
	if (chars < 1000) return `${chars}`;
	if (chars < 1_000_000) return `${(chars / 1000).toFixed(1)}k`;
	return `${(chars / 1_000_000).toFixed(1)}M`;
}

export function basename(p) {
	if (!p) return '';
	return p.split('/').pop();
}

// Filing-aware citation label. Backend citations/hits carry at least doc_id +
// page; when the corpus grows richer filing metadata (form type, company,
// ticker, fiscal period — TODO T22) we compose "10-K · Acme · FY2023 · p.42".
// Absent that metadata we degrade honestly to "<doc_id> · p.<page>" — never
// invent a form or period we weren't given. One place so Ask, the drawer and
// traces agree.
export function filingCite(c) {
	if (!c) return '';
	const parts = [];
	if (c.form) parts.push(c.form); // e.g. 10-K, 10-Q, 8-K
	if (c.company) parts.push(c.company);
	else if (!c.form && c.doc_id) parts.push(c.doc_id);
	if (c.fiscal) parts.push(c.fiscal); // e.g. FY2023
	if (c.page !== undefined && c.page !== null) parts.push(`p.${c.page}`);
	return parts.join(' · ');
}

// Bytes → gibibytes, one decimal (for disk/RAM device stats).
export function gb(bytes) {
	if (!bytes) return '0';
	return (bytes / 1024 ** 3).toFixed(1);
}

// Reliability label → tone + copy. The triad 🟢/🟡/🔴 exists ONLY as chip/dot
// colours on reliability surfaces — never decorative glyph text.
export const RELIABILITY = {
	high: { tone: 'ok', text: 'High reliability — well grounded in the sources.' },
	medium: { tone: 'warn', text: 'Medium reliability — check the citations.' },
	low: { tone: 'bad', text: 'Low reliability — the model may be guessing; verify before trusting.' },
	unknown: { tone: 'neutral', text: 'Reliability unknown — backend did not expose token probabilities.' }
};

export function reliability(label) {
	return RELIABILITY[label] || RELIABILITY.unknown;
}

// Asymmetric, caution-ONLY trust signal for the end-user answer view (T03):
// a warning only when the model may lack information or be wrong — never a
// positive "high reliability" affirmation. `high`/`unknown` → null → nothing.
export const CAUTION = {
	medium: {
		tone: 'warn',
		text: 'This may not be fully supported by the documents — check it against the cited sources below.'
	},
	low: {
		tone: 'bad',
		text: 'The model may not have enough information here, or may be guessing — verify against the sources before trusting this.'
	}
};

export function caution(label) {
	return CAUTION[label] || null;
}

// Trace step kind → line-glyph name + ALL-CAPS label + rail tone (shared by
// every timeline). Tones: accent marks model-touching steps; warn/refine is
// deliberately MONOCHROME (amber stays reserved for the reliability signal).
export const STEP = {
	plan: { label: 'Plan', icon: 'list', tone: 'neutral' },
	retrieve: { label: 'Retrieve', icon: 'search', tone: 'info' },
	context: { label: 'Assemble context', icon: 'layers', tone: 'neutral' },
	prompt: { label: 'Prompt model', icon: 'pen', tone: 'accent' },
	response: { label: 'Model response', icon: 'spark', tone: 'accent' },
	refine: { label: 'Re-query', icon: 'refresh', tone: 'warn' },
	extract_prompt: { label: 'Extraction prompt', icon: 'pen', tone: 'accent' },
	extract_response: { label: 'Extraction output', icon: 'spark', tone: 'accent' },
	extract_items: { label: 'Parsed values', icon: 'list', tone: 'info' },
	final: { label: 'Final answer', icon: 'check', tone: 'ok' }
};

export function stepMeta(kind) {
	return STEP[kind] || { label: kind, icon: 'dots', tone: 'neutral' };
}

// ISO timestamp → short local date-time (trace list / per-chat header).
export function when(iso) {
	if (!iso) return '';
	const d = new Date(iso);
	if (isNaN(d.getTime())) return iso;
	return d.toLocaleString(undefined, {
		month: 'short',
		day: 'numeric',
		hour: '2-digit',
		minute: '2-digit'
	});
}

// Ingest / document stage → chip tone + human label. Monochrome discipline:
// green/amber stay reserved for reliability — progress reads via info/accent,
// and only hard errors take red.
export const STAGE = {
	queued: { tone: 'neutral', label: 'Queued' },
	ocr: { tone: 'info', label: 'OCR / extract' },
	chunk: { tone: 'info', label: 'Chunking' },
	embed: { tone: 'accent', label: 'Embedding' },
	index: { tone: 'accent', label: 'Indexing' },
	indexed: { tone: 'neutral', label: 'Indexed (sparse)' },
	embedded: { tone: 'accent', label: 'Embedded (hybrid)' },
	embedding: { tone: 'accent', label: 'Embedding' },
	done: { tone: 'info', label: 'Done' },
	skipped: { tone: 'neutral', label: 'Skipped' },
	error: { tone: 'bad', label: 'Error' }
};

export function stage(s) {
	return STAGE[s] || { tone: 'neutral', label: s || 'unknown' };
}
