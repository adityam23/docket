<script>
	// Home — the Metro hub / Start board.
	// A big lowercase title bleeding off the right edge; below it, a tile WALL on
	// the real grid (small/medium/wide mix): the hero Ask tile boldly
	// accent-filled, live tiles flipping only when their datum changes. A recent-
	// answers strip overflows the right edge with a peek. Capabilities the
	// backend doesn't serve (a live DE pipeline, FinanceBench) degrade to an
	// honest "not configured" face — never a fake number.
	import { onMount } from 'svelte';
	import { api } from '$lib/api.js';
	import { num, pct, bytesish, when } from '$lib/format.js';
	import Chip from '$lib/components/Chip.svelte';
	import Icon from '$lib/components/Icon.svelte';
	import Tile from '$lib/components/Tile.svelte';

	let ov = $state(null);
	let traces = $state(null);
	let error = $state('');

	async function load() {
		try {
			const [o, t] = await Promise.all([api.overview(), api.traces()]);
			ov = o;
			traces = t.traces || [];
		} catch (e) {
			error = String(e).replace(/^Error:\s*/, '');
		}
	}
	onMount(() => {
		load();
		const id = setInterval(load, 12000); // gentle live refresh; tiles flip on change
		return () => clearInterval(id);
	});

	let totals = $derived(ov?.totals);
	let cap = $derived(ov?.capacity);
	let healthy = $derived(ov?.health?.ok === true);

	// A live DE pipeline (Kafka/Spark/dbt) is config-gated: it exists only once the
	// operator points this instance at their own stream. No endpoint yet → absent.
	let pipelineConfigured = $derived(!!ov?.pipeline);

	// Trust summary across recorded answers — derived, honestly labelled as history.
	let trust = $derived.by(() => {
		const t = traces || [];
		const c = { high: 0, medium: 0, low: 0, unknown: 0 };
		for (const x of t) c[x.reliability in c ? x.reliability : 'unknown']++;
		return { total: t.length, ...c };
	});
	let recents = $derived((traces || []).slice(0, 6));

	function relDot(r) {
		return r === 'low' ? 'bad' : r === 'medium' ? 'warn' : r === 'high' ? 'ok' : 'neutral';
	}
</script>

<div class="hub">
	<h1 class="bleed">coverage</h1>
	<p class="page-sub" style="--i: 1">
		Your filings, freshly indexed — ask across the whole corpus and see exactly how confident each answer is.
	</p>
</div>

{#if error}
	<div class="banner bad"><Icon name="alert" size={16} /> Could not reach the API: {error}</div>
{:else if !ov}
	<div class="empty"><div class="spin"></div></div>
{:else}
	<div class="tile-wall cascade" style="margin-top: 28px">
		<!-- Hero: Ask — boldly accent-filled -->
		<Tile href="/ask" variant="hero" size="s-wide" badge={num(totals.documents)}>
			<div class="hero-copy" style="--i: 0">
				<div class="tile-label">ask your filings</div>
				<div class="hero-line">Grounded, cited answers across every document in scope.</div>
				<div class="row wrap" style="margin-top: 10px">
					<span class="hero-chip">{num(totals.chunks)} chunks in scope</span>
				</div>
			</div>
		</Tile>

		<!-- Corpus -->
		<Tile href="/filings" label="corpus" size="s-medium" flipKey={totals.documents + ':' + totals.chunks} glyph="db">
			<div class="tile-big">{num(totals.documents)}</div>
			<div class="tile-sub">{num(totals.chunks)} chunks · {bytesish(totals.chars)} chars</div>
			<div class="row spread small mute2" style="margin-top: auto">
				<span>embedding coverage</span>
				<span class="mono">{pct(totals.embedding_coverage)}</span>
			</div>
		</Tile>

		<!-- Trust summary — the ONLY place the triad appears -->
		<Tile href="/observability" label="trust" size="s-medium" flipKey={trust.total + ':' + trust.high + ':' + trust.low}>
			{#if trust.total}
				<div class="trust-bars">
					{#each [['high', trust.high], ['medium', trust.medium], ['low', trust.low], ['unknown', trust.unknown]] as [k, v]}
						{#if v}
							<span class="tb {relDot(k)}" style="flex: {v}" title="{v} {k}"></span>
						{/if}
					{/each}
				</div>
				<div class="tile-big" style="font-size: var(--fs-ml)">{pct(trust.total ? trust.high / trust.total : 0)}</div>
				<div class="tile-sub">{num(trust.total)} answers · share high-confidence</div>
			{:else}
				<div class="tile-sub">No answers recorded yet.</div>
			{/if}
			<div class="small mute2" style="margin-top: auto">Reliability, not correctness — verify citations.</div>
		</Tile>

		<!-- Backend -->
		<Tile href="/settings" label="backend" size="s-small" flipKey={healthy ? 'up' : 'down'}>
			<div class="row" style="gap: 9px; margin-top: 2px">
				<span class="hdot" class:up={healthy}></span>
				<span class="tile-big" style="font-size: var(--fs-md)">{healthy ? 'Online' : 'Offline'}</span>
			</div>
			<div class="tile-sub">
				{healthy ? `${(ov.health.models || []).length} models served` : (ov.health.error || 'unreachable')}
			</div>
		</Tile>

		<!-- Capacity -->
		{#if cap}
			<Tile href="/filings" label="room to grow" size="s-small" flipKey={cap.remaining_documents_est}>
				<div class="tile-big">~{num(cap.remaining_documents_est)}</div>
				<div class="tile-sub">more filings fit · limited by {cap.binding_constraint === 'ram' ? 'RAM' : 'disk'}</div>
			</Tile>
		{/if}

		<!-- Pulse (DE pipeline) — config-gated: honest empty until pointed at a stream -->
		<Tile label="pipeline pulse" size="s-small" flipKey={pipelineConfigured ? 'on' : 'off'} glyph={!pipelineConfigured ? 'stream' : null}>
			{#if pipelineConfigured}
				<div class="tile-big">live</div>
				<div class="tile-sub">ingestion stream connected</div>
			{:else}
				<div class="tile-sub">Not connected.</div>
				<div class="small mute2">The live Kafka/Spark pipeline appears once this instance is pointed at your own stream.</div>
			{/if}
		</Tile>

		<!-- FinanceBench — honest empty (no eval run wired to this instance) -->
		<Tile label="financebench" size="s-small" flipKey="pending" glyph="chart">
			<div class="tile-sub">No eval run recorded.</div>
			<div class="small mute2">Accuracy scores surface here after a benchmark run.</div>
		</Tile>
	</div>

	<!-- Recent answers — overflows the right edge with a peek (panorama affordance) -->
	{#if recents.length}
		<div class="strip-head caps" style="--i: 2">recent answers</div>
		<div class="strip" style="--i: 3">
			{#each recents as r}
				<a class="recent" href={`/chat/${r.id}`}>
					<span class="rg {relDot(r.reliability)}"></span>
					<span class="rq">{r.question}</span>
					<span class="rt mute2 small">{when(r.created_at)}</span>
				</a>
			{/each}
		</div>
	{/if}
{/if}

<style>
	.hub { overflow: hidden; }
	/* The defining Metro cue: the title clips off the right edge, inviting a pan. */
	.bleed {
		font-size: clamp(4rem, 14vw, 8rem);
		font-weight: var(--w-semilight);
		line-height: 1;
		letter-spacing: -0.03em;
		white-space: nowrap;
		margin: 0 -2px calc(var(--u) * 1.5);
	}

	.hero-copy { display: flex; flex-direction: column; gap: 6px; height: 100%; }
	.hero-line { font-size: var(--fs-ml); font-weight: var(--w-light); line-height: 1.3; max-width: 40ch; }
	.hero-chip {
		font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em;
		background: rgba(255, 255, 255, 0.16); padding: 3px 8px;
	}

	.trust-bars { display: flex; gap: 3px; height: 12px; margin: 4px 0 8px; }
	.tb { min-width: 5px; }
	.tb.ok { background: var(--ok); }
	.tb.warn { background: var(--warn); }
	.tb.bad { background: var(--bad); }
	.tb.neutral { background: var(--neutral); }

	.hdot { width: 10px; height: 10px; border-radius: 50%; background: var(--bad); }
	.hdot.up { background: var(--text); } /* online = bright white, not green */

	.strip-head { margin: calc(var(--block) * 1.25) 0 10px; }
	/* Horizontal strip that visibly continues past the right edge. Cards are sized
	   to exactly one 2-column tile so they land on the same grid lines as the tile
	   wall above — the strip reads as the same field, then bleeds off the right. */
	.strip {
		--wall-cols: 6;
		display: flex; gap: var(--gutter);
		overflow-x: auto; scrollbar-width: none;
		padding-bottom: 6px;
		margin-right: calc(-1 * var(--left-line));
		padding-right: var(--left-line);
	}
	.strip::-webkit-scrollbar { display: none; }
	.recent {
		flex: 0 0 calc((100% - (var(--wall-cols) - 1) * var(--gutter)) / var(--wall-cols) * 2 + var(--gutter));
		background: var(--surface-2); padding: 16px;
		display: flex; flex-direction: column; gap: 8px; min-width: 0;
	}
	.recent:hover { background: var(--surface-3); }
	.rg { width: 9px; height: 9px; border-radius: 50%; background: var(--neutral); }
	.rg.ok { background: var(--ok); }
	.rg.warn { background: var(--warn); }
	.rg.bad { background: var(--bad); }
	.rq {
		color: var(--text); font-size: var(--fs-normal); font-weight: var(--w-regular);
		line-height: 1.35; display: -webkit-box; -webkit-line-clamp: 2; line-clamp: 2;
		-webkit-box-orient: vertical; overflow: hidden;
	}
	.rt { white-space: nowrap; }

	/* Track the tile-wall's column count so cards stay one 2-col tile wide. */
	@media (max-width: 900px) { .strip { --wall-cols: 4; } }
	@media (max-width: 720px) {
		.strip { --wall-cols: 2; }
		.recent { flex-basis: 82%; } /* 2 cols == full width; show a peek instead */
	}
</style>
