<script>
	// Per-chat timeline as its own page (T06). Deep-linkable /chat/<trace_id>;
	// SPA client-side route (web/app.py falls back to index.html). Renders the
	// full timeline for one trace via the shared component. When the trace
	// belongs to a Chat session (T21), a breadcrumb names the chat and turn.
	import { onMount } from 'svelte';
	import { page } from '$app/stores';
	import { api } from '$lib/api.js';
	import { when } from '$lib/format.js';
	import Chip from '$lib/components/Chip.svelte';
	import Icon from '$lib/components/Icon.svelte';
	import Timeline from '$lib/components/Timeline.svelte';

	let trace = $state(null);
	let error = $state('');
	let id = $derived($page.params.id);

	async function load() {
		trace = null;
		error = '';
		try {
			trace = await api.trace(id);
		} catch (e) {
			error = String(e).replace(/^Error:\s*/, '');
		}
	}
	// Re-fetch if the id changes (client-side nav between chats).
	$effect(() => {
		if (id) load();
	});

	function relTone(rel) {
		return rel === 'low' ? 'bad' : rel === 'medium' ? 'warn' : rel === 'high' ? 'ok' : 'neutral';
	}
</script>

<div class="page-head">
	<div class="row" style="gap: 18px; margin-bottom: 14px">
		<a class="back small" href="/observability"><Icon name="back" size={12} /> Observability</a>
		<a class="back small" href="/ask">Ask again</a>
	</div>
	<h1 class="page-title">chat trace</h1>
</div>

{#if error}
	<div class="banner bad"><Icon name="alert" size={16} /> {error}</div>
{:else if !trace}
	<div class="empty"><div class="spin"></div></div>
{:else}
	{#if trace.chat_id}
		<div class="caps crumb">
			part of chat <a class="link mono" href="/observability">{trace.chat_id}</a>
			{#if trace.turn_index !== null && trace.turn_index !== undefined}· turn {trace.turn_index + 1}{/if}
		</div>
	{/if}

	<div class="q-block fade-in">
		<div class="who caps">you</div>
		<div class="question">{trace.question}</div>

		<div class="who caps" style="margin-top: 22px">answer</div>
		<div class="answer">{trace.answer}</div>

		<div class="row wrap meta">
			<Chip tone={relTone(trace.reliability)} dot>{trace.reliability}</Chip>
			{#if trace.path === 'aggregate'}<Chip tone="accent">aggregate</Chip>{/if}
			{#if trace.hops > 1}<Chip tone="neutral">{trace.hops} hops</Chip>{/if}
			<Chip tone="neutral">{trace.elapsed_ms} ms</Chip>
			<span class="mute2 small">
				{when(trace.created_at)}
				{#if trace.surprisal !== null && trace.surprisal !== undefined}· σ {trace.surprisal}{/if}
				· <span class="mono">{trace.id}</span>
			</span>
		</div>
	</div>

	<div class="section-title">timeline</div>
	<Timeline steps={trace.steps} />
{/if}

<style>
	.crumb { margin: -6px 0 16px; }
	.back { color: var(--text-mute); display: inline-flex; align-items: center; gap: 5px; }
	.back:hover { color: var(--text); }
	.q-block { background: var(--surface-2); padding: var(--pad); margin-bottom: 10px; }
	.who { font-size: 0.66rem; letter-spacing: 0.16em; margin-bottom: 6px; color: var(--text-faint); }
	.question {
		font-size: var(--fs-ml); font-weight: var(--w-semibold); letter-spacing: -0.01em;
		padding-left: 12px; border-left: 3px solid var(--accent);
	}
	.answer { white-space: pre-wrap; line-height: 1.7; max-width: 78ch; }
	.meta { margin-top: 20px; }
</style>
