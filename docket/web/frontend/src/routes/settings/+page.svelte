<script>
	// Settings — the one place configuration is CHANGED, as a text list-menu
	// (Metro settings style: open rows on black, ALL-CAPS section captions, no
	// boxes). Non-secret knobs persist via PATCH /api/config (→ .env.local
	// overlay); provider API keys via POST /api/config/keys (0600, never echoed).
	// Appearance (density / motion) is a client-side preference in the ui store.
	import { onMount } from 'svelte';
	import { api, errMsg } from '$lib/api.js';
	import { ui } from '$lib/stores/ui.js';
	import Chip from '$lib/components/Chip.svelte';

	let cfg = $state(null);
	let health = $state(null);
	let error = $state('');

	let form = $state({});
	let original = {};
	let saving = $state(false);
	let saved = $state(false);
	let saveError = $state('');

	let keyInput = $state({ engine: '' });
	let keyBusy = $state('');

	const NUMS = ['request_timeout_s', 'retrieval_k', 'context_chunks', 'max_hops'];

	// Mirrors Settings.engine_collection's server-side pattern (config.py
	// _COLLECTION_NAME_PATTERN) — client-side so a bad name is caught before
	// the round trip; the server validates for real and is authoritative.
	const COLLECTION_PATTERN = /^[A-Za-z0-9_-][A-Za-z0-9._-]{0,127}$/;

	function initForm() {
		const r = cfg.retrieval;
		form = {
			backend_url: cfg.backend_url,
			chat_model: cfg.chat_model,
			rerank_model: cfg.rerank_model,
			engine_binary: cfg.supervision?.engine_binary || '',
			mettle_config: cfg.supervision?.mettle_config || '',
			engine_collection: cfg.retrieval.engine_collection,
			retrieval_mode: cfg.retrieval.retrieval_mode,
			retrieval_k: r.retrieval_k,
			context_chunks: r.context_chunks,
			max_hops: r.max_hops,
			request_timeout_s: cfg.request_timeout_s
		};
		original = { ...form };
	}

	let collectionValid = $derived(COLLECTION_PATTERN.test(form.engine_collection ?? ''));

	async function load() {
		try {
			[cfg, health] = await Promise.all([api.config(), api.health()]);
			initForm();
		} catch (e) {
			error = errMsg(e);
		}
	}
	onMount(load);

	async function save() {
		const updates = {};
		for (const k of Object.keys(form)) {
			let v = form[k];
			if (NUMS.includes(k)) v = Number(v);
			if (v !== original[k]) updates[k] = v;
		}
		if (!Object.keys(updates).length) {
			saved = true;
			setTimeout(() => (saved = false), 2000);
			return;
		}
		saving = true;
		saveError = '';
		try {
			cfg = await api.updateConfig(updates);
			initForm();
			health = await api.health();
			saved = true;
			setTimeout(() => (saved = false), 2500);
		} catch (e) {
			saveError = errMsg(e);
		} finally {
			saving = false;
		}
	}

	async function submitKey(provider, value) {
		keyBusy = provider;
		saveError = '';
		try {
			const r = await api.setKey(provider, value || null);
			cfg = { ...cfg, api_keys: r.api_keys };
			keyInput[provider] = '';
		} catch (e) {
			saveError = errMsg(e);
		} finally {
			keyBusy = '';
		}
	}

</script>

<h1 class="page-title">settings</h1>
<p class="page-sub" style="margin-bottom: 8px">
	Engine, models and retrieval — saved to a local
	<span class="mono">.env.local</span> overlay and applied on the next request, no restart.
</p>

{#if error}
	<div class="banner bad">{error}</div>
{:else if !cfg}
	<div class="empty"><div class="spin"></div></div>
{:else}
	{#if saveError}
		<div class="banner bad" style="margin-bottom: 16px">{saveError}</div>
	{/if}

	<!-- Engine & model -->
	<div class="list-section-title">Engine &amp; model</div>
	<div class="list-menu">
		<div class="list-row">
			<div>
				<div class="lr-label">Engine</div>
				<div class="lr-hint">mettle — one <span class="mono">/v1</span> base serves chat, embeddings, rerank and collections.
					<Chip tone={health?.ok ? 'accent' : 'bad'} dot>{health?.ok ? 'online' : 'offline'}</Chip>
				</div>
			</div>
		</div>
		<div class="list-row">
			<div>
				<div class="lr-label">Engine URL</div>
				<div class="lr-hint">OpenAI-compatible <span class="mono">/v1</span> base.</div>
			</div>
			<input type="text" bind:value={form.backend_url} spellcheck="false" style="max-width: 320px" />
		</div>
		<div class="list-row">
			<div>
				<div class="lr-label">Chat model</div>
				<div class="lr-hint">Served models: {(health?.models || []).join(', ') || 'none reported'}</div>
			</div>
			<input type="text" bind:value={form.chat_model} spellcheck="false" style="max-width: 320px" />
		</div>
	</div>

	<!-- Reranker. Embeddings + chunking (chunk_words/chunk_overlap) are a frozen
	     parity substrate and embed_model only affects the offline `dk embed`
	     path — all three are env-only developer config, not web-writable
	     (config_write.py WRITABLE_FIELDS), so there's nothing to edit here. -->
	<div class="list-section-title">Reranker</div>
	<div class="list-menu">
		<div class="list-row">
			<div>
				<div class="lr-label">Reranker</div>
				<div class="lr-hint">Cross-encoder served at <span class="mono">/v1/rerank</span>; the reranked order is what ships.</div>
			</div>
			<input type="text" bind:value={form.rerank_model} spellcheck="false" style="max-width: 320px" />
		</div>
	</div>

	<!-- Retrieval -->
	<div class="list-section-title">Retrieval</div>
	<div class="list-menu">
		<div class="list-row">
			<div>
				<div class="lr-label">Search mode</div>
				<div class="lr-hint">Engine search strategy; <span class="mono">hybrid</span> unions keyword + dense.</div>
			</div>
			<select bind:value={form.retrieval_mode} style="max-width: 200px">
				<option value="hybrid">hybrid</option>
				<option value="dense">dense</option>
				<option value="keyword">keyword</option>
			</select>
		</div>
		<div class="list-row">
			<div>
				<div class="lr-label">Collection</div>
				<div class="lr-hint">
					Engine collection filings are ingested into. 1–128 chars: letters, digits,
					<span class="mono">. _ -</span>, never starting with a dot (server-enforced).
				</div>
				{#if form.engine_collection && !collectionValid}
					<div class="lr-hint" style="color: var(--bad)">Doesn't match the allowed pattern — the save will be rejected.</div>
				{/if}
			</div>
			<input
				type="text"
				bind:value={form.engine_collection}
				spellcheck="false"
				aria-invalid={form.engine_collection ? !collectionValid : false}
				style="max-width: 280px"
			/>
		</div>
		<div class="list-row">
			<div><div class="lr-label">Candidates (k)</div><div class="lr-hint">Candidates pulled before rerank (default 40).</div></div>
			<input type="number" min="1" bind:value={form.retrieval_k} style="max-width: 140px" />
		</div>
		<div class="list-row">
			<div><div class="lr-label">Context chunks</div><div class="lr-hint">Chunks placed in the prompt.</div></div>
			<input type="number" min="1" bind:value={form.context_chunks} style="max-width: 140px" />
		</div>
		<div class="list-row">
			<div><div class="lr-label">Max agentic hops</div></div>
			<input type="number" min="1" bind:value={form.max_hops} style="max-width: 140px" />
		</div>
		<div class="list-row">
			<div><div class="lr-label">Request timeout (s)</div></div>
			<input type="number" min="1" step="1" bind:value={form.request_timeout_s} style="max-width: 140px" />
		</div>
	</div>

	<div class="row" style="margin: 22px 0 8px; gap: 12px">
		<button class="btn" onclick={save} disabled={saving || !collectionValid}>
			{#if saving}<span class="spin"></span> Saving…{:else}Save settings{/if}
		</button>
		{#if saved}<Chip tone="neutral" dot>Saved</Chip>{/if}
		<span class="mute2 small">Index: <span class="mono">{cfg.index_dir}</span></span>
	</div>

	<!-- Engine supervision -->
	<div class="list-section-title">Engine supervision</div>
	<div class="list-menu">
		<div class="list-row">
			<div>
				<div class="lr-label">mettled binary</div>
				<div class="lr-hint">
					Path to the engine binary. Set: docket spawns, health-probes and crash-restarts it.
					Blank: probe-only — an unreachable engine shows <span class="mono">offline</span> and docket features fail loudly.
				</div>
			</div>
			<input type="text" bind:value={form.engine_binary} spellcheck="false" placeholder="/path/to/mettled" style="max-width: 320px" />
		</div>
		<div class="list-row">
			<div>
				<div class="lr-label">Engine config</div>
				<div class="lr-hint">Passed to the engine as <span class="mono">METTLE_CONFIG</span> when docket spawns it. Blank = engine default.</div>
			</div>
			<input type="text" bind:value={form.mettle_config} spellcheck="false" style="max-width: 320px" />
		</div>
	</div>

	<!-- Engine key -->
	<div class="list-section-title">Engine key</div>
	<div class="list-menu">
		<p class="small mute2" style="margin: 0 0 6px">
			Stored locally in <span class="mono">.env.local</span> at <span class="mono">0600</span> and never shown again.
			Needed only when the engine enforces auth on <span class="mono">/v1</span> / <span class="mono">/engine/*</span>.
		</p>
		{#each ['engine'] as p}
			<div class="list-row">
				<div>
					<div class="lr-label mono">{p}
						<Chip tone={cfg.api_keys[p] ? 'accent' : 'neutral'} dot={false}>{cfg.api_keys[p] ? 'set' : 'not set'}</Chip>
					</div>
				</div>
				<div class="row" style="gap: 8px">
					<input
						type="password"
						placeholder={cfg.api_keys[p] ? 'Enter a new key…' : 'Paste API key…'}
						bind:value={keyInput[p]}
						autocomplete="off"
						spellcheck="false"
						style="max-width: 240px"
					/>
					<button class="btn" onclick={() => submitKey(p, keyInput[p])} disabled={keyBusy === p || !keyInput[p]}>
						{#if keyBusy === p}<span class="spin"></span>{:else}Save{/if}
					</button>
					{#if cfg.api_keys[p]}
						<button class="btn ghost" onclick={() => submitKey(p, '')} disabled={keyBusy === p}>Clear</button>
					{/if}
				</div>
			</div>
		{/each}
	</div>

	<!-- Pipeline connection — honest: configured out-of-band on the self-host build -->
	<div class="list-section-title">Pipeline connection</div>
	<div class="list-menu">
		<div class="list-row">
			<div>
				<div class="lr-label">Streaming ingestion</div>
				<div class="lr-hint">
					The live Kafka/Spark/dbt pipeline is wired via environment on the self-host build;
					when connected, its metrics appear on <a class="link" href="/observability">Observability → pipeline</a>.
				</div>
			</div>
			<Chip tone="neutral">not connected</Chip>
		</div>
	</div>

	<!-- Appearance (client-side prefs) -->
	<div class="list-section-title">Appearance</div>
	<div class="list-menu">
		<div class="list-row">
			<div>
				<div class="lr-label">Density</div>
				<div class="lr-hint">Airy board, or a tighter layout for dense work.</div>
			</div>
			<select value={$ui.density} onchange={(e) => ui.set({ density: e.currentTarget.value })} style="max-width: 200px">
				<option value="comfortable">Comfortable</option>
				<option value="dense">Dense</option>
			</select>
		</div>
		<div class="list-row">
			<div>
				<div class="lr-label">Motion</div>
				<div class="lr-hint">Tile flips and transitions. Reduced also honours your OS setting.</div>
			</div>
			<select value={$ui.motion} onchange={(e) => ui.set({ motion: e.currentTarget.value })} style="max-width: 200px">
				<option value="full">Full</option>
				<option value="reduced">Reduced</option>
			</select>
		</div>
	</div>
{/if}
