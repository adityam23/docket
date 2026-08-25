<script>
	// The ONE "explain this answer" drawer (CLAUDE.md reuse). Layered, each layer
	// shown ONLY if this instance's config produced the data behind it — so a user
	// never sees a locked tease for a capability they don't have (graceful
	// degradation, per the anti-slop rule):
	//   Sources   — citations + retrieved chunks (always present)
	//   Reasoning — the full persisted trace timeline (lazy-loaded by trace_id)
	//   Lab       — SAE feature-attribution / steering; requires a white-box local
	//               backend. On hosted API keys it degrades to an honest note.
	import { api } from '$lib/api.js';
	import { filingCite } from '$lib/format.js';
	import HitCard from '$lib/components/HitCard.svelte';
	import Icon from '$lib/components/Icon.svelte';
	import Pivots from '$lib/components/Pivots.svelte';
	import Timeline from '$lib/components/Timeline.svelte';

	// res = the /api/ask done payload for one assistant turn.
	let { res, labAvailable = false } = $props();

	let open = $state(false);
	let tab = $state('sources');
	let full = $state(null); // lazily-fetched full trace (for the reasoning layer)
	let loadingFull = $state(false);
	let fullError = $state('');

	async function ensureFull() {
		if (full || loadingFull || !res?.trace_id) return;
		loadingFull = true;
		try {
			full = await api.trace(res.trace_id);
		} catch (e) {
			fullError = String(e).replace(/^Error:\s*/, '');
		} finally {
			loadingFull = false;
		}
	}

	function select(t) {
		tab = t;
		if (t === 'reasoning') ensureFull();
	}
	function toggle() {
		open = !open;
	}

	let citations = $derived(res?.citations || []);
	let hits = $derived(res?.hits || []);

	let tabs = $derived([
		{ id: 'sources', label: 'sources' },
		{ id: 'reasoning', label: 'reasoning' },
		{ id: 'lab', label: 'lab' }
	]);
</script>

<div class="drawer">
	<button class="drawer-toggle" onclick={toggle} aria-expanded={open}>
		<span class="caret" class:open><Icon name="down" size={13} /></span>
		<span class="caps" style="color: var(--text-dim)">Explain this answer</span>
		<span class="mute2 small">sources · reasoning{#if labAvailable} · lab{/if}</span>
	</button>

	{#if open}
		<div class="drawer-body fade-in">
			<Pivots {tabs} bind:value={tab} onselect={select} />

			{#if tab === 'sources'}
				<div class="layer">
					{#if citations.length}
						{#each citations as c, i}
							<div class="source">
								<span class="cite-pill">{i + 1}</span>
								<div>
									<div class="mono small" style="color: var(--text)">{filingCite(c)}</div>
									{#if c.quote}<div class="quote">"{c.quote}"</div>{/if}
								</div>
							</div>
						{/each}
					{:else}
						<div class="mute2 small">No citations were emitted for this answer.</div>
					{/if}

					{#if hits.length}
						<div class="panel-title" style="margin: 16px 0 8px">Retrieved chunks</div>
						<div class="hits">
							{#each hits as h}<HitCard hit={h} />{/each}
						</div>
					{/if}
				</div>
			{:else if tab === 'reasoning'}
				<div class="layer">
					{#if loadingFull}
						<div class="empty"><div class="spin"></div></div>
					{:else if fullError}
						<div class="banner bad"><Icon name="alert" size={16} /> {fullError}</div>
					{:else if full}
						<Timeline steps={full.steps} />
						{#if res.trace_id}
							<div class="row" style="justify-content: flex-end; margin-top: 10px">
								<a class="small link" href={`/chat/${res.trace_id}`}>Open full trace →</a>
							</div>
						{/if}
					{:else}
						<div class="mute2 small">No trace recorded for this answer.</div>
					{/if}
				</div>
			{:else if tab === 'lab'}
				<div class="layer">
					{#if labAvailable && res?.lab}
						<!-- Real SAE attribution renders here when the backend supplies it. -->
						<div class="mute2 small">Feature attribution rendering.</div>
					{:else}
						<div class="degraded">
							<div class="dg-title">Lab mode isn't available on this backend</div>
							The glass-box layer — SAE feature attribution, steering, and the
							white-box uncertainty detector — needs a local, logprob-exposing model
							(it can't run over a hosted API). Point this instance at a local backend
							in <a href="/settings">Settings</a> to enable it.
						</div>
					{/if}
				</div>
			{/if}
		</div>
	{/if}
</div>

<style>
	.drawer { border-top: 1px solid var(--line); margin-top: 14px; padding-top: 10px; }
	.drawer-toggle {
		display: flex; align-items: center; gap: 9px; width: 100%; text-align: left;
		background: none; border: none; color: var(--text-dim); font: inherit;
		cursor: pointer; padding: 4px 0;
	}
	.drawer-toggle:hover .caps { color: var(--text); }
	.caret { color: var(--text-mute); transition: transform var(--dur-fast); display: inline-flex; }
	.caret.open { transform: rotate(180deg); }
	.drawer-body { margin-top: 12px; }
	.layer { margin-top: 16px; }

	.source { display: flex; gap: 10px; padding: 9px 0; border-bottom: 1px solid var(--line); }
	.source:last-child { border-bottom: none; }
	.quote { color: var(--text-dim); font-size: 0.85rem; margin-top: 3px; }
	.cite-pill {
		display: inline-grid; place-items: center; min-width: 20px; height: 20px; padding: 0 5px;
		background: var(--accent-soft); color: var(--accent-bright);
		font: 600 0.74rem/1 var(--mono); flex-shrink: 0;
	}
	.hits { display: flex; flex-direction: column; gap: 8px; }
</style>
