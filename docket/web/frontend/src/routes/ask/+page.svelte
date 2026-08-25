<script>
	// Ask — a real chat surface over the corpus.
	// Metro: turns are CONTENT, not heavy bubbles — everything on one left line;
	// a turn is identified by its ALL-CAPS label ("you" / "answer") and type
	// weight. Enter sends, Shift+Enter adds a newline; the pipeline streams into
	// the pending turn (SSE), then the answer settles with filing-aware
	// citations, a caution-only trust glance, and an Explain drawer.
	//
	// T21 conversation context: this page holds a real Chat session. The first
	// question creates one server-side (the done payload returns `chat_id`);
	// every later question sends `session_id` so prior turns are threaded into
	// the model's context. "New chat" starts fresh.
	import { onDestroy, tick } from 'svelte';
	import { api } from '$lib/api.js';
	import { num, caution } from '$lib/format.js';
	import Chip from '$lib/components/Chip.svelte';
	import ExplainDrawer from '$lib/components/ExplainDrawer.svelte';
	import Icon from '$lib/components/Icon.svelte';
	import Timeline from '$lib/components/Timeline.svelte';

	let question = $state('');
	let scope = $state(null);
	let sessionId = $state(null); // T21: the Chat these turns belong to
	let chatTitle = $state('');
	let turns = $state([]); // { id, question, res, liveSteps, loading, error }
	let cancel = null;
	let seq = 0;
	let scroller; // conversation viewport

	const examples = [
		'What was total revenue and how did it change year over year?',
		'Summarize the key risk factors.',
		'What are the operating expenses?'
	];

	async function loadScope() {
		try {
			scope = (await api.corpus()).totals;
		} catch {
			scope = null;
		}
	}
	loadScope();

	onDestroy(() => cancel && cancel());

	async function toBottom() {
		await tick();
		if (scroller) scroller.scrollTop = scroller.scrollHeight;
	}

	function newChat() {
		if (cancel) cancel();
		cancel = null;
		sessionId = null;
		chatTitle = '';
		turns = [];
	}

	async function ask() {
		const q = question.trim();
		const busy = turns.some((t) => t.loading);
		if (!q || busy) return;
		question = '';
		if (cancel) cancel();

		const turn = { id: ++seq, question: q, res: null, liveSteps: [], loading: true, error: '' };
		turns = [...turns, turn];
		toBottom();

		const upd = (patch) => {
			turns = turns.map((t) => (t.id === turn.id ? { ...t, ...patch } : t));
		};

		// T21: the first question mints the Chat session; every later question
		// joins it so prior turns are threaded server-side as model context.
		if (!sessionId) {
			try {
				sessionId = (await api.createChat()).id;
			} catch (e) {
				upd({ error: String(e).replace(/^Error:\s*/, ''), loading: false });
				return;
			}
		}

		cancel = api.askStream(
			q,
			sessionId,
			{
				onStep: (s) => {
					const cur = turns.find((t) => t.id === turn.id);
					upd({ liveSteps: [...(cur?.liveSteps || []), s] });
					toBottom();
				},
				onDone: (payload) => {
					// The first answered turn names the chat; later turns join it.
					if (payload.chat_id) sessionId = payload.chat_id;
					if (!chatTitle && payload.question) chatTitle = payload.question;
					upd({ res: payload, loading: false });
					toBottom();
				},
				onError: (msg) => {
					upd({ error: msg, loading: false });
				}
			}
		);
	}

	function use(ex) {
		question = ex;
		ask();
	}

	function onKey(e) {
		// Enter sends; Shift+Enter (or the IME composing) inserts a newline.
		if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
			e.preventDefault();
			ask();
		}
	}

	// Split an answer into text + [n] citation pills.
	function splitCitations(text) {
		const parts = [];
		let last = 0;
		const re = /\[(\d+)\]/g;
		let m;
		while ((m = re.exec(text)) !== null) {
			if (m.index > last) parts.push({ t: 'text', v: text.slice(last, m.index) });
			parts.push({ t: 'cite', v: m[1] });
			last = re.lastIndex;
		}
		if (last < text.length) parts.push({ t: 'text', v: text.slice(last) });
		return parts;
	}

	let anyLoading = $derived(turns.some((t) => t.loading));
</script>

<div class="page-head">
	<div class="row spread wrap" style="align-items: flex-end">
		<div style="min-width: 0">
			<h1 class="page-title">ask</h1>
			<p class="page-sub">Grounded, cited answers with a reliability signal.</p>
		</div>
		<div class="row wrap">
			{#if scope}
				<Chip tone="neutral">{num(scope.documents)} filings · {num(scope.chunks)} chunks</Chip>
			{/if}
			{#if turns.length}
				<button class="btn ghost new-chat" onclick={newChat} disabled={anyLoading}>
					<Icon name="plus" size={13} /> New chat
				</button>
			{/if}
		</div>
	</div>
</div>

<div class="chat dense">
	<div class="stream" bind:this={scroller}>
		{#if !turns.length}
			<div class="intro">
				<div class="intro-glyph"><Icon name="chat" size={30} stroke={1.2} /></div>
				<div class="intro-copy">Ask anything about your filings. Answers cite the exact span they came from.</div>
				<div class="row wrap" style="justify-content: center; margin-top: 14px">
					{#each examples as ex}
						<button class="ex" onclick={() => use(ex)}>{ex}</button>
					{/each}
				</div>
			</div>
		{/if}

		{#each turns as t (t.id)}
			<!-- user turn: content, not chrome -->
			<div class="who caps">you</div>
			<div class="user-q">{t.question}</div>

			<!-- assistant turn -->
			{#if t.error}
				<div class="banner bad"><Icon name="alert" size={16} /> {t.error}</div>
			{:else if t.loading && !t.res}
				<div class="working fade-in">
					<div class="row" style="gap: 8px; margin-bottom: 8px">
						<span class="spin"></span><span class="mute2 small">Working through the filings…</span>
					</div>
					{#if t.liveSteps.length}<Timeline steps={t.liveSteps} live />{/if}
				</div>
			{:else if t.res}
				{@const warn = caution(t.res.reliability)}
				{@const parts = splitCitations(t.res.answer || '')}
				<div class="who caps">answer</div>
				<div class="answer">
					{#each parts as p}
						{#if p.t === 'cite'}<span class="cite-pill" title="source {p.v}">{p.v}</span>{:else}{p.v}{/if}
					{/each}
				</div>

				{#if warn}
					<div class="glance {warn.tone}">
						<Icon name="alert" size={15} /><span>{warn.text}</span>
					</div>
				{/if}

				<div class="row wrap meta">
					<Chip tone="neutral">{t.res.elapsed_ms} ms</Chip>
					{#if t.res.hops > 1}<Chip tone="accent">{t.res.hops} hops</Chip>{/if}
					{#if t.res.trace_id}<a class="small link" href={`/chat/${t.res.trace_id}`}>Permalink →</a>{/if}
				</div>

				<ExplainDrawer res={t.res} />
			{/if}
		{/each}
	</div>

	<!-- composer -->
	<div class="composer">
		<textarea
			rows="1"
			bind:value={question}
			placeholder="Ask about revenue, risks, guidance…  (Enter to send, Shift+Enter for a newline)"
			onkeydown={onKey}
		></textarea>
		<button class="send" onclick={ask} disabled={anyLoading || !question.trim()} aria-label="Send">
			{#if anyLoading}<span class="spin"></span>{:else}<Icon name="send" size={18} />{/if}
		</button>
	</div>
</div>

<style>
	.chat {
		display: flex; flex-direction: column; gap: 14px;
		height: calc(100vh - 220px); min-height: 420px;
	}
	.stream {
		flex: 1; overflow-y: auto; padding: 6px 2px;
		display: flex; flex-direction: column;
	}
	.stream > :first-child { margin-top: 0 !important; }

	.intro { text-align: center; color: var(--text-mute); margin: auto; max-width: 44ch; padding-bottom: 40px; }
	.intro-glyph { color: var(--text-faint); display: grid; place-items: center; }
	.intro-copy { margin-top: 10px; color: var(--text-dim); }

	/* Turns are content on the ONE left line — labels + weight do the work. */
	.who { font-size: 0.66rem; letter-spacing: 0.16em; margin: 26px 0 5px; color: var(--text-faint); }
	.user-q {
		font-size: var(--fs-md); font-weight: var(--w-semibold); letter-spacing: -0.01em;
		color: var(--text); white-space: pre-wrap; max-width: 72ch;
		padding-left: 12px; border-left: 3px solid var(--accent);
	}

	.answer { font-size: var(--fs-normal); line-height: 1.7; white-space: pre-wrap; max-width: 78ch; }
	.cite-pill {
		display: inline-grid; place-items: center; min-width: 19px; height: 19px; padding: 0 4px;
		background: var(--accent-soft); color: var(--accent-bright); border-radius: 2px;
		font: 600 0.72rem/1 var(--mono); vertical-align: baseline; margin: 0 1px;
		cursor: default;
	}
	.glance {
		display: flex; gap: 9px; align-items: flex-start; margin-top: 14px;
		padding: 11px 13px; border-radius: var(--radius-sm); font-size: 0.88rem; max-width: 78ch;
	}
	.glance.warn { background: var(--warn-soft); color: var(--warn); } /* reliability signal */
	.glance.bad { background: var(--bad-soft); color: var(--bad); }    /* reliability signal */
	.meta { margin-top: 12px; }

	.working { padding: 8px 0 4px; max-width: 720px; }

	.ex {
		background: var(--surface-2); border: none; color: var(--text-dim);
		font: inherit; font-size: 0.82rem; padding: 7px 13px; border-radius: 2px; cursor: pointer;
		transition: background var(--dur-fast), color var(--dur-fast);
	}
	.ex:hover { background: var(--surface-3); color: var(--text); }

	.composer {
		display: flex; gap: 10px; align-items: flex-end;
		background: var(--surface-2);
		padding: 10px;
	}
	.composer textarea {
		border: none; background: none; padding: 8px 6px; min-height: 24px; max-height: 200px;
		box-shadow: none;
	}
	.composer textarea:focus { box-shadow: none; }
	.send {
		flex-shrink: 0; width: 42px; height: 42px; padding: 0; border-radius: var(--radius);
		display: grid; place-items: center; cursor: pointer;
		background: var(--accent); color: #fff; border: none;
		transition: background var(--dur-fast);
	}
	.send:hover:not(:disabled) { background: var(--accent-bright); }
	.send:disabled { opacity: 0.35; cursor: not-allowed; }

	.new-chat { padding: 7px 12px; font-size: 0.8rem; }

	@media (max-width: 720px) {
		.user-q, .answer { max-width: 100%; }
	}
</style>
