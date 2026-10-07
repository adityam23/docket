<script>
	// Engine profile control + status (mettle /engine/* consumer).
	// Shows the active profile, the switchable profile list (slots + VRAM fit),
	// device VRAM, and live activation progress over SSE. When the backend has no
	// engine surface (`available: false`) it degrades to an honest empty state —
	// never a fake panel. Monochrome state labels: the green/amber/red triad is
	// reserved for the reliability signal (design-language §3), so slot states
	// read as neutral text; only a hard error takes `.bad`.
	import { onMount, onDestroy } from 'svelte';
	import { api, errMsg } from '$lib/api.js';
	import { gb, supervisionTone, supervisionText } from '$lib/format.js';
	import { supervision } from '$lib/stores/supervision.js';
	import Icon from '$lib/components/Icon.svelte';

	let eng = $state(null);       // GET /api/engine payload
	let err = $state('');
	let busy = $state('');        // profile name being switched
	let lastEvt = $state(null);   // latest SSE event (activation progress line)
	let closeEvents = $state(null);
	let streamError = $state(''); // honest reason when the SSE stream dies (F3)

	// Supervisor's lifecycle view (F3) — this panel is the detail home: the
	// compact TopNav dot shows the same state, this shows the full phase/
	// reason/detail the engine reported. One poller (stores/supervision.js);
	// this component just attaches for as long as it's mounted.
	let sup = $derived($supervision.view);
	let supTone = $derived(supervisionTone(sup?.state));
	let supText = $derived(supervisionText(sup));
	let detachSupervision;
	onMount(() => (detachSupervision = supervision.attach()));
	onDestroy(() => detachSupervision?.());

	function load() {
		api.engine()
			.then((d) => {
				eng = d;
				// Start streaming activation progress once we know the surface
				// exists — an engine without a surface has no events to replay.
				if (d.available && !closeEvents) startEvents();
			})
			.catch((e) => (err = errMsg(e)));
	}
	function startEvents() {
		streamError = '';
		closeEvents = api.engineEvents(
			(ev) => {
				lastEvt = ev;
				streamError = '';
				// An activation settling (or a slot change) invalidates the cached
				// view — refresh so the active profile + states are live.
				if (ev && (ev.profile_active || ev.slot_unavailable || ev.slot_available)) {
					load();
				}
			},
			(reason) => {
				// Honest, not invisible: say the stream is down instead of just
				// freezing the last progress line with no explanation. The
				// EventSource retries on its own (we don't recreate it) — a
				// subsequent event clears this via the onEvent handler above.
				streamError = reason;
			}
		);
	}
	onMount(load);
	onDestroy(() => closeEvents?.());

	async function switchTo(name) {
		if (busy || name === eng?.active) return;
		busy = name;
		err = '';
		try {
			const res = await api.engineProfile(name);
			if (res && !res.accepted) err = res.error || res.detail || 'activation rejected';
		} catch (e) {
			err = errMsg(e);
		} finally {
			busy = '';
			load(); // re-read active immediately; SSE continues the progress line
		}
	}

	function stateTone(s) {
		// Monochrome by default; only an explicit slot failure reads `.bad`.
		return /error|unavailable|fail/i.test(s || '') ? 'bad' : 'neutral';
	}
	function evtText(ev) {
		if (ev == null) return '';
		if (typeof ev === 'string') return ev;
		return ev.text || ev.type || JSON.stringify(ev).slice(0, 140);
	}
	// `detail` is whatever the engine's /engine/ready reported verbatim — a
	// string (a mettle reason like "dropped_vram") or a structured body.
	// Render it honestly either way rather than assuming a shape.
	function detailText(d) {
		if (d == null || d === '') return '';
		if (typeof d === 'string') return d;
		try {
			return JSON.stringify(d);
		} catch {
			return String(d);
		}
	}
</script>

<div class="engine">
	{#if sup}
		<div class="cap">SUPERVISION</div>
		<div class="sup-row">
			<span class="r-dot {supTone === 'down' ? 'bad' : supTone === 'up' ? 'up' : ''}"></span>
			<span>{supText}</span>
			{#if sup.pid}<span class="mute2 small mono">pid {sup.pid}</span>{/if}
		</div>
		{#if detailText(sup.detail)}
			<div class="small mute2 sup-detail">{detailText(sup.detail)}</div>
		{/if}
	{/if}

	{#if !eng}
		<div class="empty"><div class="spin"></div></div>
	{:else if !eng.available}
		<div class="degraded">
			<Icon name="alert" size={16} />
			<div>
				<div class="dg-title">No engine profile surface</div>
				<p class="small mute2">
					The configured backend doesn't expose <code>/engine/profiles</code>. Profile
					switching and VRAM-aware slot management are available only against a
					mettle-class engine.
				</p>
			</div>
		</div>
	{:else}
		{#if eng.device && eng.device.available}
			<div class="device cap">DEVICE</div>
			<div class="device-row">
				<span class="mute2">{eng.device.backend || 'gpu'}</span>
				<span class="fig">{gb(eng.device.free_bytes)} free</span>
				<span class="mute2">/ {gb(eng.device.total_bytes)} total</span>
			</div>
		{/if}

		<div class="list">
			{#each eng.profiles || [] as p (p.name)}
				<button
					class="profile"
					class:active={p.name === eng.active}
					onclick={() => switchTo(p.name)}
					disabled={!!busy || p.name === eng.active}
				>
					<span class="p-name">{p.name}</span>
					{#if p.name === eng.active}
						<span class="active-chip">ACTIVE</span>
					{:else}
						<span class="p-switch">{busy === p.name ? 'switching…' : 'switch'}</span>
					{/if}
					<span class="p-bytes">{p.estimated_bytes != null ? gb(p.estimated_bytes) : '—'}</span>

					{#if p.slots && p.slots.length}
						<span class="slots">
							{#each p.slots as s (s.id)}
								<span class="slot">
									<span class="r-dot {stateTone(s.state)}"></span>
									<span class="s-cap">{s.role}</span>
									<span class="s-id">{s.id}</span>
									{#if s.state}
										<span class="s-st mute2">{s.state}</span>
									{/if}
								</span>
							{/each}
						</span>
					{/if}
				</button>
			{/each}
		</div>

		{#if lastEvt}
			<div class="evt">
				<Icon name="stream" size={14} />
				<span class="small mute2">{evtText(lastEvt)}</span>
			</div>
		{/if}
		{#if streamError}
			<div class="evt">
				<Icon name="alert" size={14} />
				<span class="small">live updates unavailable — {streamError}</span>
			</div>
		{/if}
	{/if}

	{#if err}
		<div class="banner bad"><Icon name="alert" size={16} /> {err}</div>
	{/if}
</div>

<style>
	.engine { display: flex; flex-direction: column; gap: 6px; }
	.cap { color: var(--text-mute); font-size: 0.72rem; letter-spacing: 0.06em; }
	.device { margin-top: 14px; }
	.device-row { display: flex; align-items: baseline; gap: 8px; font-size: 0.92rem; }
	.fig { font-variant-numeric: tabular-nums; color: var(--text); }

	.list { display: flex; flex-direction: column; gap: 6px; margin-top: 12px; }
	.profile {
		display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px;
		width: 100%; text-align: left; color: var(--text); font: inherit;
		background: none; border: none; border-bottom: 1px solid var(--line);
		padding: 14px 8px; cursor: pointer; transition: background var(--dur-fast);
	}
	.profile:hover:not(:disabled) { background: var(--surface-1); }
	.profile.active { cursor: default; }
	.p-name { font-size: 1.05rem; font-weight: var(--w-semibold); }
	.profile.active .p-name { color: var(--accent-bright); }
	.active-chip { color: var(--accent-bright); font-size: 0.68rem; letter-spacing: 0.08em; }
	.p-switch { color: var(--text-dim); font-size: 0.78rem; }
	.p-bytes { margin-left: auto; font-variant-numeric: tabular-nums; color: var(--text-dim); font-size: 0.82rem; }

	.slots { display: flex; flex-direction: column; gap: 3px; width: 100%; margin-top: 2px; }
	.slot { display: flex; align-items: center; gap: 8px; font-size: 0.82rem; color: var(--text-dim); }
	.r-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--neutral); flex-shrink: 0; }
	.r-dot.bad { background: var(--bad); }
	.r-dot.up { background: var(--text); }
	.s-cap { color: var(--text-mute); font-size: 0.7rem; letter-spacing: 0.06em; min-width: 42px; }
	.s-id { color: var(--text); font-variant-numeric: tabular-nums; }
	.s-st { margin-left: auto; }

	.sup-row { display: flex; align-items: center; gap: 8px; margin-top: 4px; }
	.sup-detail { margin-top: 2px; word-break: break-word; }

	.evt { display: flex; align-items: center; gap: 8px; margin-top: 12px; color: var(--text-mute); }

	@media (max-width: 720px) {
		.p-bytes { margin-left: 0; }
	}
</style>
