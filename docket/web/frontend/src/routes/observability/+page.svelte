<script>
	// Observability — two pivots (docs/design-language.md §9, T18):
	//   Pipeline — live Kafka/Spark/dbt freshness+throughput. Config-gated: it
	//              does not exist until this instance points at its own stream,
	//              so it degrades to an honest empty state (never faked).
	//   Traces   — the audit surface, NESTED chat → turns → trace (T21): a chat
	//              groups its ordered turns; each turn keeps its own trace and
	//              links to the full /chat/<trace_id> page. One-shot questions
	//              (no session) list flat. Expanding a turn loads the verbatim
	//              timeline via the shared component.
	import { onMount } from 'svelte';
	import { api } from '$lib/api.js';
	import { when } from '$lib/format.js';
	import Chip from '$lib/components/Chip.svelte';
	import Icon from '$lib/components/Icon.svelte';
	import Pivots from '$lib/components/Pivots.svelte';
	import Timeline from '$lib/components/Timeline.svelte';

	let traces = $state(null);
	let chats = $state(null);
	let overview = $state(null);
	let error = $state('');
	let openId = $state('');
	let full = $state({});
	let loadingId = $state('');
	let tab = $state('traces');
	let confirming = $state(''); // row key awaiting a delete confirm (two-step)
	let busyDelete = $state(''); // row key mid-delete

	async function refresh() {
		try {
			[traces, chats, overview] = await Promise.all([
				api.traces().then((r) => r.traces),
				api.chats().then((r) => r.chats).catch(() => []),
				api.overview().catch(() => null)
			]);
		} catch (e) {
			error = String(e).replace(/^Error:\s*/, '');
		}
	}
	onMount(refresh);

	async function toggle(id) {
		if (openId === id) {
			openId = '';
			return;
		}
		openId = id;
		// Chat heads render their turns inline from `chats` — there is no trace under
		// a `chat:` id, so fetching one would 404. Only lone traces need a fetch.
		if (id.startsWith('chat:')) return;
		if (!full[id]) {
			loadingId = id;
			try {
				full[id] = await api.trace(id);
			} catch (e) {
				error = String(e).replace(/^Error:\s*/, '');
			} finally {
				loadingId = '';
			}
		}
	}

	function relTone(rel) {
		return rel === 'low' ? 'bad' : rel === 'medium' ? 'warn' : rel === 'high' ? 'ok' : 'neutral';
	}

	// Two-step delete: first click arms the confirm, second commits. Deleting a chat
	// cascades to its turns' traces server-side; mirror that in local state.
	async function removeChat(id) {
		busyDelete = 'chat:' + id;
		try {
			await api.deleteChat(id);
			chats = (chats || []).filter((c) => c.id !== id);
			traces = (traces || []).filter((t) => t.chat_id !== id);
			if (openId === 'chat:' + id) openId = '';
		} catch (e) {
			error = String(e).replace(/^Error:\s*/, '');
		} finally {
			busyDelete = '';
			confirming = '';
		}
	}
	async function removeTrace(id) {
		busyDelete = id;
		try {
			await api.deleteTrace(id);
			traces = (traces || []).filter((t) => t.id !== id);
			delete full[id];
			if (openId === id) openId = '';
		} catch (e) {
			error = String(e).replace(/^Error:\s*/, '');
		} finally {
			busyDelete = '';
			confirming = '';
		}
	}

	// The DE pipeline is present only if the backend reports it — otherwise absent.
	let pipeline = $derived(overview?.pipeline || null);

	// Traces that belong to a chat fold under that chat; one-shots stay flat.
	let loneTraces = $derived(
		(traces || []).filter((t) => !t.chat_id || !(chats || []).some((c) => c.id === t.chat_id))
	);
	let tabs = $derived([
		{ id: 'pipeline', label: 'pipeline' },
		{ id: 'traces', label: 'traces', badge: (traces?.length ?? 0) + (chats?.length ?? 0) || undefined }
	]);
</script>

<!-- One delete control, reused for chats and one-shot traces: trash → confirm. -->
{#snippet delCtl(key, run)}
	{#if confirming === key}
		<span class="confirm">
			<button class="mini danger" disabled={busyDelete === key} onclick={run}>
				{#if busyDelete === key}<span class="spin sm"></span>{:else}delete{/if}
			</button>
			<button class="mini" disabled={busyDelete === key} onclick={() => (confirming = '')}>cancel</button>
		</span>
	{:else}
		<button class="icon-btn" title="Delete" aria-label="Delete" onclick={() => (confirming = key)}>
			<Icon name="trash" size={14} />
		</button>
	{/if}
{/snippet}

<div class="page-head">
	<div class="row spread wrap" style="align-items: flex-end">
		<div>
			<h1 class="page-title">observability</h1>
			<p class="page-sub">How the data flows in, and exactly how each answer came out.</p>
		</div>
		<button class="btn ghost" onclick={refresh}><Icon name="refresh" size={13} /> Refresh</button>
	</div>
</div>

<div style="margin-bottom: 22px"><Pivots {tabs} bind:value={tab} /></div>

{#if error}
	<div class="banner bad"><Icon name="alert" size={16} /> {error}</div>
{/if}

{#if tab === 'pipeline'}
	{#if pipeline}
		<!-- Real pipeline metrics render here once the backend exposes them. -->
		<div class="panel panel-pad">
			<div class="panel-title">Ingestion stream</div>
			<div class="mute2 small">Live freshness &amp; throughput.</div>
		</div>
	{:else}
		<div class="degraded">
			<div class="dg-title">No live pipeline connected</div>
			This instance isn't pointed at a streaming ingestion pipeline, so there's
			nothing to chart here yet. On the self-host build, connect a Kafka/Spark/dbt
			source and its freshness and throughput appear on this pivot. Batch ingestion
			you run by hand is visible on <a href="/filings">Filings</a>.
		</div>
	{/if}
{:else}
	<!-- Traces: chats nested, then one-shot questions -->
	{#if !traces}
		<div class="empty"><div class="spin"></div></div>
	{:else if !traces.length && !(chats || []).length}
		<div class="empty">
			<div class="big"><Icon name="pulse" size={40} stroke={1.2} /></div>
			<div>No answers recorded yet.</div>
			<div class="small mute2" style="margin-top: 6px">
				Ask a question and its full trace appears here. <a class="link" href="/ask">Go to Ask →</a>
			</div>
		</div>
	{:else}
		<div class="list">
			{#each chats as c}
				<div class="entry">
					<div class="head-row">
						<button class="chat-head" onclick={() => toggle('chat:' + c.id)}>
							<span class="caret" class:open={openId === 'chat:' + c.id}><Icon name="down" size={12} /></span>
							<span class="q"><span class="chat-tag caps">chat</span>{c.title || 'Untitled chat'}</span>
							<span class="meta">
								<Chip tone="neutral">{c.turns.length} turns</Chip>
								<span class="mute2 small">{when(c.updated_at || c.created_at)}</span>
							</span>
						</button>
						{@render delCtl('chat:' + c.id, () => removeChat(c.id))}
					</div>
					{#if openId === 'chat:' + c.id}
						<div class="turns fade-in">
							{#each c.turns as turn}
								<div class="turn-row">
									<span class="rg {relTone(turn.reliability)}"></span>
									<a class="tq" href={`/chat/${turn.trace_id}`}>{turn.question}</a>
									<span class="mute2 small mono">{turn.trace_id}</span>
									<span class="mute2 small">{when(turn.created_at)}</span>
								</div>
							{/each}
						</div>
					{/if}
				</div>
			{/each}

			{#if (chats || []).length && loneTraces.length}
				<div class="group-header">one-shot questions</div>
			{/if}
			{#each loneTraces as t (t.id)}
				<div class="entry">
					<div class="head-row">
						<button class="chat-head" onclick={() => toggle(t.id)}>
							<span class="caret" class:open={openId === t.id}><Icon name="down" size={12} /></span>
							<span class="q">{t.question}</span>
							<span class="meta">
								<Chip tone={relTone(t.reliability)} dot>{t.reliability}</Chip>
								{#if t.path === 'aggregate'}<Chip tone="accent">aggregate</Chip>{/if}
								{#if t.hops > 1}<Chip tone="neutral">{t.hops} hops</Chip>{/if}
								<span class="mute2 small mono">{t.elapsed_ms} ms</span>
								<span class="mute2 small">{when(t.created_at)}</span>
							</span>
						</button>
						{@render delCtl(t.id, () => removeTrace(t.id))}
					</div>
					{#if openId === t.id}
						<div class="trace-body">
							{#if loadingId === t.id}
								<div class="empty"><div class="spin"></div></div>
							{:else if full[t.id]}
								<div class="row" style="justify-content: flex-end; margin-bottom: 10px">
									<a class="btn ghost new-chat" href={`/chat/${t.id}`}>Open full page →</a>
								</div>
								<Timeline steps={full[t.id].steps} />
							{/if}
						</div>
					{/if}
				</div>
			{/each}
		</div>
		<p class="small mute2" style="margin-top: 14px">
			Each trace shows which filings were mentioned, the exact context passed to the
			model, and — where the retriever emits rerank scores — whether a chunk was truly
			relevant, not merely vector-similar.
		</p>
	{/if}
{/if}

<style>
	.list { display: flex; flex-direction: column; }
	.entry { border-bottom: 1px solid var(--line); }
	/* Head + delete control share a row; the control reveals on hover/focus. */
	.head-row { display: flex; align-items: center; gap: 6px; padding-right: 8px; }
	.chat-head {
		display: flex; align-items: center; gap: 12px; flex: 1; min-width: 0; text-align: left;
		background: none; border: none; color: var(--text); font: inherit; cursor: pointer;
		padding: 15px 8px; transition: background var(--dur-fast);
	}
	.head-row:hover .chat-head { background: var(--surface-1); }
	.head-row .icon-btn { opacity: 0; transition: opacity var(--dur-fast); flex-shrink: 0; }
	.head-row:hover .icon-btn, .head-row:focus-within .icon-btn { opacity: 1; }
	.confirm { display: flex; gap: 6px; flex-shrink: 0; }
	.mini {
		background: var(--surface-2); color: var(--text-dim); border: 1px solid var(--line-strong);
		font: inherit; font-size: 0.78rem; padding: 5px 12px; cursor: pointer;
		transition: background var(--dur-fast), color var(--dur-fast), border-color var(--dur-fast);
	}
	.mini:hover:not(:disabled) { background: var(--surface-3); color: var(--text); }
	.mini.danger { border-color: var(--bad); color: var(--bad); }
	.mini.danger:hover:not(:disabled) { background: var(--bad); color: var(--bg); }
	.mini:disabled { opacity: 0.5; cursor: default; }
	.spin.sm { width: 12px; height: 12px; border-width: 2px; }
	.caret { color: var(--text-mute); transition: transform var(--dur-fast); flex-shrink: 0; display: inline-flex; }
	.caret.open { transform: rotate(180deg); }
	.q { font-weight: var(--w-regular); font-size: var(--fs-normal); flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
	.chat-tag { color: var(--accent-bright); margin-right: 10px; }
	.meta { display: flex; align-items: center; gap: 8px; flex-shrink: 0; }
	.trace-body { padding: 6px 8px 20px 30px; }

	.turns { padding: 4px 8px 16px 30px; display: flex; flex-direction: column; }
	.turn-row {
		display: grid; grid-template-columns: auto 1fr auto auto; gap: 12px; align-items: center;
		padding: 10px 4px; border-bottom: 1px solid var(--line);
	}
	.turn-row:last-child { border-bottom: none; }
	.rg { width: 8px; height: 8px; border-radius: 50%; background: var(--neutral); }
	.rg.ok { background: var(--ok); }     /* reliability dots only */
	.rg.warn { background: var(--warn); }
	.rg.bad { background: var(--bad); }
	.tq { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text); }
	.tq:hover { color: var(--accent-bright); }
	.new-chat { padding: 6px 12px; font-size: 0.8rem; }

	@media (max-width: 720px) {
		.q { white-space: normal; }
		.chat-head { flex-wrap: wrap; }
		.turn-row { grid-template-columns: auto 1fr; grid-auto-rows: auto; }
	}
</style>
