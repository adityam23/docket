<script>
	// The ONE top chrome (T18): a Pivot-style typographic strip — lowercase
	// headers, active bright + larger, the next header peeking right in faint
	// grey. No sidebar, no underline glow, no pill bar. App title = small
	// ALL-CAPS wordmark. Status is MONOCHROME (green stays reserved for the
	// reliability triad); red appears only for hard failures.
	import { page } from '$app/stores';
	import { ingest } from '$lib/stores/ingest.js';
	import Icon from './Icon.svelte';

	let { health = null } = $props();

	const nav = [
		{ href: '/', label: 'home' },
		{ href: '/ask', label: 'ask' },
		{ href: '/filings', label: 'filings' },
		{ href: '/observability', label: 'observability' },
		{ href: '/settings', label: 'settings' }
	];

	let path = $derived($page.url.pathname);
	let healthy = $derived(health?.ok === true);
	let healthUnknown = $derived(health === null);

	let job = $derived($ingest.job);
	let ingesting = $derived($ingest.running === true);

	function active(href) {
		return href === '/' ? path === '/' : path.startsWith(href);
	}
</script>

<header class="topnav">
	<div class="bar">
		<a href="/" class="brand caps">agent·assistant</a>

		<nav class="pivots" aria-label="Primary">
			{#each nav as item}
				<a href={item.href} class="pivot" class:active={active(item.href)}>{item.label}</a>
			{/each}
		</nav>

		<div class="status">
			{#if ingesting && job}
				<a class="ingest mono" href="/filings" title="Ingestion running in the background">
					<span class="spin"></span>
					<span>{job.completed}/{job.total}</span>
				</a>
			{/if}
			<span
				class="health"
				class:up={healthy}
				class:down={!healthy && !healthUnknown}
				title={health?.base_url || ''}
			>
				<span class="dot"></span>
				<span class="small">{#if healthUnknown}checking…{:else if healthy}backend online{:else}backend offline{/if}</span>
			</span>
		</div>
	</div>
</header>

<style>
	.topnav {
		position: sticky; top: 0; z-index: 20;
		background: rgba(0, 0, 0, 0.88);
		backdrop-filter: blur(10px);
	}
	.bar {
		max-width: 1220px; margin: 0 auto; padding: 0 var(--left-line);
		height: 58px; display: flex; align-items: center; gap: 34px;
	}
	.brand {
		color: var(--text); letter-spacing: 0.14em; font-size: 0.72rem;
		white-space: nowrap;
	}
	.brand .mute2 { color: var(--text-faint); }

	.pivots { flex: 1; gap: 26px; }
	/* Slightly smaller than in-page pivots so the page title dominates. */
	.pivots :global(.pivot) { font-size: 1.05rem; }
	.pivots :global(.pivot.active) { font-size: 1.25rem; }

	.status { display: flex; align-items: center; gap: 14px; }
	.ingest {
		display: flex; align-items: center; gap: 8px; padding: 5px 10px;
		background: var(--surface-2); color: var(--text-dim); font-size: 0.78rem;
		border-radius: var(--radius-sm);
	}
	.health { display: flex; align-items: center; gap: 8px; color: var(--text-mute); white-space: nowrap; }
	.health .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--text-faint); transition: background var(--dur-fast); }
	.health.up .dot { background: var(--text); }        /* online = bright white */
	.health.down .dot { background: var(--bad); }       /* hard failure may use red */

	@media (max-width: 720px) {
		.bar { padding: 0 16px; gap: 16px; height: auto; flex-wrap: wrap; padding-top: 10px; padding-bottom: 10px; }
		.pivots { order: 3; flex-basis: 100%; }
		.status .small { display: none; }
	}
</style>
