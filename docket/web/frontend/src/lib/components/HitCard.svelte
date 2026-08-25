<script>
	// The ONE retrieval hit / chunk card (CLAUDE.md reuse): doc · page · rank ·
	// score and expands to the FULL stored chunk text. The always-available "see
	// the referenced span" fallback (T13) — no retained PDF needed. Flat dark
	// field, square corners.
	import Icon from './Icon.svelte';

	let { hit, n = null, open = false } = $props();
	let expanded = $state(open);
	// A short chunk isn't worth a toggle — only clamp when there's more to reveal.
	let clampable = $derived((hit.text || '').length > 180);
</script>

<div class="hit">
	<div class="row spread">
		<span class="mono small head">
			<span class="rank">#{n ?? (hit.rank ?? 0) + 1}</span>
			{hit.doc_id} · p.{hit.page}
		</span>
		{#if hit.score !== undefined && hit.score !== null}
			<span class="mono small mute2">score {hit.score}</span>
		{/if}
	</div>
	<div class="hit-text" class:open={expanded || !clampable}>{hit.text}</div>
	{#if clampable}
		<button class="more" onclick={() => (expanded = !expanded)}>
			<Icon name={expanded ? 'down' : 'chevron'} size={12} />
			{expanded ? 'Show less' : 'Show full chunk'}
		</button>
	{/if}
</div>

<style>
	.hit { background: var(--surface-1); padding: 11px 13px; }
	.head { color: var(--text); }
	.rank { color: var(--accent-bright); }
	.hit-text {
		margin-top: 6px; color: var(--text-dim); font-size: 0.82rem; line-height: 1.5;
		display: -webkit-box; -webkit-line-clamp: 3; line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;
	}
	.hit-text.open {
		display: block; -webkit-line-clamp: unset; line-clamp: unset; overflow: visible; white-space: pre-wrap;
	}
	.more {
		margin-top: 6px; padding: 0; background: none; border: none; cursor: pointer;
		color: var(--accent-bright); font: inherit; font-size: 0.78rem;
		display: inline-flex; align-items: center; gap: 4px;
	}
	.more:hover { text-decoration: underline; }
</style>
