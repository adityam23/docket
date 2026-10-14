<script>
	// The ONE top chrome (T18): a Pivot-style typographic strip — lowercase
	// headers, active bright + larger, the next header peeking right in faint
	// grey. No sidebar, no underline glow, no pill bar. App title = small
	// ALL-CAPS wordmark. Status is MONOCHROME (green stays reserved for the
	// reliability triad); red appears only for hard failures.
	//
	// Engine status (F3) comes from the ONE supervision poller (stores/
	// supervision.js, attached app-wide by +layout.svelte) — this replaces the
	// old plain /health up/down dot, which couldn't tell "offline" apart from
	// a minutes-long first-run model download. Detail (phase/reason/full
	// detail) lives in EnginePanel; this stays a compact, honest one-liner.
	import { page } from '$app/stores';
	import { api, errMsg } from '$lib/api.js';
	import { ingest } from '$lib/stores/ingest.js';
	import { supervision } from '$lib/stores/supervision.js';
	import { supervisionTone, supervisionText } from '$lib/format.js';
	import Icon from './Icon.svelte';

	const nav = [
		{ href: '/', label: 'home' },
		{ href: '/ask', label: 'ask' },
		{ href: '/filings', label: 'filings' },
		{ href: '/observability', label: 'observability' },
		{ href: '/settings', label: 'settings' }
	];

	let path = $derived($page.url.pathname);

	let job = $derived($ingest.job);
	let ingesting = $derived($ingest.running === true);

	let view = $derived($supervision.view);
	// A failed poll means the docket server itself isn't answering: never keep
	// showing the last engine state as if it were current.
	let unreachable = $derived(!!$supervision.error);
	let tone = $derived(unreachable ? 'down' : supervisionTone(view?.state));
	let statusText = $derived(unreachable ? 'docket server unreachable' : supervisionText(view));
	// The ONLY two exits a manual restart can take (backend facts, F3): the
	// terminal `failed` state, and the port-held variant of `offline` (its
	// `reason` is always set — a bare unconfigured/unprobed offline is not).
	let canRestart = $derived(!unreachable && (view?.state === 'failed' || (view?.state === 'offline' && !!view?.reason)));
	let restarting = $state(false);
	let restartError = $state('');

	async function restart() {
		if (restarting) return;
		restarting = true;
		restartError = '';
		try {
			await api.engineRestart();
		} catch (e) {
			restartError = errMsg(e);
		} finally {
			restarting = false;
			supervision.refresh();
		}
	}

	function active(href) {
		return href === '/' ? path === '/' : path.startsWith(href);
	}
</script>

<header class="topnav">
	<div class="bar">
		<a href="/" class="brand">Docket</a>

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
			<span class="health {tone}" title={view?.detail ? JSON.stringify(view.detail) : ''}>
				<span class="dot"></span>
				<span class="small status-text">{statusText}</span>
				{#if view?.active}<span class="small mute2">· {view.active}</span>{/if}
				{#if view?.switching}<span class="small mute2">switching…</span>{/if}
			</span>
			{#if canRestart}
				<button class="restart-btn" onclick={restart} disabled={restarting}>
					<Icon name="refresh" size={12} />
					{restarting ? 'restarting…' : 'restart engine'}
				</button>
			{/if}
			{#if restartError}<span class="small restart-err">{restartError}</span>{/if}
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
		color: var(--text); font-size: 1.05rem; font-weight: 600;
		white-space: nowrap;
	}

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
	.health.up .dot { background: var(--text); }        /* available = bright white */
	.health.busy .dot { background: var(--text-dim); } /* starting (incl. download/load progress) */
	.health.warn .dot { background: var(--text-mute); } /* degraded/incompatible — stays monochrome */
	.health.down .dot { background: var(--bad); }       /* offline/failed — the only red states */
	.status-text {
		display: inline-block; max-width: 320px; overflow: hidden;
		text-overflow: ellipsis; white-space: nowrap; vertical-align: bottom;
	}

	.restart-btn {
		display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px;
		background: var(--surface-2); color: var(--text); font: inherit; font-size: 0.78rem;
		border: 1px solid var(--line-strong); border-radius: var(--radius-sm); cursor: pointer;
		transition: background var(--dur-fast), border-color var(--dur-fast);
	}
	.restart-btn:hover:not(:disabled) { border-color: var(--bad); color: var(--bad); }
	.restart-btn:disabled { opacity: 0.5; cursor: not-allowed; }
	.restart-err { color: var(--bad); }

	@media (max-width: 720px) {
		.bar { padding: 0 16px; gap: 16px; height: auto; flex-wrap: wrap; padding-top: 10px; padding-bottom: 10px; }
		.pivots { order: 3; flex-basis: 100%; }
		.status .small { display: none; }
	}
</style>
