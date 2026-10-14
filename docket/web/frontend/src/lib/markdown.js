// Model answer → safe HTML. The one place model-authored markdown is rendered
// (Ask turns and the per-chat permalink share it via <Answer>).
//
// Security: model output is untrusted. Every line is HTML-escaped FIRST and only
// then are the markdown markers turned into a fixed set of tags — no raw HTML,
// links or attributes from the model ever reach the DOM.
//
// Deliberately a small subset (headings, bold/italic, inline code, blockquotes,
// lists, pipe tables, paragraphs) plus docket's `[n]` citation pills. If answers
// outgrow it, swap the body of renderAnswer for a real parser + sanitizer; the
// callers don't change.

const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };

function escapeHtml(s) {
	return s.replace(/[&<>"']/g, (c) => ESC[c]);
}

// Inline markers on an already-escaped string.
function inline(s) {
	return s
		.replace(/`([^`]+)`/g, '<code>$1</code>')
		.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
		.replace(/(^|[^*])\*([^*\s][^*]*)\*/g, '$1<em>$2</em>')
		.replace(/\[(\d+)\]/g, '<span class="cite-pill" title="source $1">$1</span>');
}

const RE_HEADING = /^(#{1,6})\s+(.*)$/;
const RE_QUOTE = /^&gt;\s?(.*)$/;
const RE_UL = /^\s*[-*+]\s+(.*)$/;
const RE_OL = /^\s*\d+[.)]\s+(.*)$/;
const RE_TABLE_SEP = /^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$/;

function tableCells(line) {
	return line.replace(/^\|/, '').replace(/\|$/, '').split('|').map((c) => inline(c.trim()));
}

export function renderAnswer(text) {
	const lines = escapeHtml(text || '').split('\n');
	const out = [];
	let para = [];

	const flushPara = () => {
		if (para.length) out.push(`<p>${para.map(inline).join('<br>')}</p>`);
		para = [];
	};

	for (let i = 0; i < lines.length; ) {
		const line = lines[i];
		const t = line.trim();
		let m;

		if (!t) {
			flushPara();
			i++;
		} else if ((m = t.match(RE_HEADING))) {
			flushPara();
			// Answers sit under the page's own title: every heading level
			// renders as one small section heading.
			out.push(`<h4>${inline(m[2])}</h4>`);
			i++;
		} else if (t.startsWith('|')) {
			flushPara();
			const rows = [];
			while (i < lines.length && lines[i].trim().startsWith('|')) rows.push(lines[i++].trim());
			const hasHead = rows.length > 1 && RE_TABLE_SEP.test(rows[1]);
			const head = hasHead ? `<thead><tr>${tableCells(rows[0]).map((c) => `<th>${c}</th>`).join('')}</tr></thead>` : '';
			const body = (hasHead ? rows.slice(2) : rows)
				.map((r) => `<tr>${tableCells(r).map((c) => `<td>${c}</td>`).join('')}</tr>`)
				.join('');
			out.push(`<table>${head}<tbody>${body}</tbody></table>`);
		} else if (RE_QUOTE.test(t)) {
			flushPara();
			const q = [];
			while (i < lines.length && (m = lines[i].trim().match(RE_QUOTE))) (q.push(inline(m[1])), i++);
			out.push(`<blockquote>${q.join('<br>')}</blockquote>`);
		} else if (RE_UL.test(line) || RE_OL.test(line)) {
			flushPara();
			const ordered = !RE_UL.test(line);
			const re = ordered ? RE_OL : RE_UL;
			const items = [];
			while (i < lines.length && (m = lines[i].match(re))) (items.push(`<li>${inline(m[1])}</li>`), i++);
			const tag = ordered ? 'ol' : 'ul';
			out.push(`<${tag}>${items.join('')}</${tag}>`);
		} else {
			para.push(t);
			i++;
		}
	}
	flushPara();
	return out.join('');
}
