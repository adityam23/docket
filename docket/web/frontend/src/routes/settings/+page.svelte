<script>
	// Settings — the one place configuration is CHANGED, as a text list-menu
	// (Metro settings style: open rows on black, ALL-CAPS section captions, no
	// boxes). Non-secret knobs persist via PATCH /api/config (→ .env.local
	// overlay); provider API keys via POST /api/config/keys (0600, never echoed).
	// Appearance (density / motion) is a client-side preference in the ui store.
	import { onMount } from 'svelte';
	import { api } from '$lib/api.js';
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

	let keyInput = $state({ cerebras: '', groq: '' });
	let keyBusy = $state('');

	const NUMS = ['request_timeout_s', 'chunk_words', 'chunk_overlap', 'retrieval_k', 'context_chunks', 'max_hops'];

	function initForm() {
		const r = cfg.retrieval;
		form = {
			provider: cfg.provider,
			backend_url: cfg.backend_url,
			chat_model: cfg.chat_model,
			embed_url: cfg.embed.url || '',
			embed_model: cfg.embed.model,
			chunk_words: r.chunk_words,
			chunk_overlap: r.chunk_overlap,
			retrieval_k: r.retrieval_k,
			context_chunks: r.context_chunks,
			max_hops: r.max_hops,
			request_timeout_s: cfg.request_timeout_s
		};
		original = { ...form };
	}

	async function load() {
		try {
			[cfg, health] = await Promise.all([api.config(), api.health()]);
			initForm();
		} catch (e) {
			error = String(e).replace(/^Error:\s*/, '');
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
			saveError = String(e).replace(/^Error:\s*/, '');
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
			saveError = String(e).replace(/^Error:\s*/, '');
		} finally {
			keyBusy = '';
		}
	}

	let embedEnabled = $derived(!!(form.embed_url && form.embed_url.trim()));
</script>

<h1 class="page-title">settings</h1>
<p class="page-sub" style="margin-bottom: 8px">
	Backend, models, embeddings and retrieval — saved to a local
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

	<!-- Backend / model -->
	<div class="list-section-title">Backend &amp; model</div>
	<div class="list-menu">
		<div class="list-row">
			<div>
				<div class="lr-label">Provider</div>
				<div class="lr-hint">Where inference runs.
					<Chip tone={health?.ok ? 'neutral' : 'bad'} dot>{health?.ok ? 'online' : 'offline'}</Chip>
				</div>
			</div>
			<select bind:value={form.provider} style="max-width: 320px">
				<option value="local">local (llama-server / infengine / Ollama)</option>
				<option value="cerebras">cerebras</option>
				<option value="groq">groq</option>
			</select>
		</div>
		<div class="list-row">
			<div>
				<div class="lr-label">Backend URL</div>
				<div class="lr-hint">OpenAI-compatible <span class="mono">/v1</span> endpoint.</div>
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

	<!-- Embeddings & chunking -->
	<div class="list-section-title">Embeddings &amp; chunking</div>
	<div class="list-menu">
		<div class="list-row">
			<div>
				<div class="lr-label">Embeddings URL</div>
				<div class="lr-hint">Blank = sparse-only (BM25).</div>
			</div>
			<div class="row" style="gap: 10px; max-width: 380px">
				<Chip tone={embedEnabled ? 'accent' : 'neutral'}>{embedEnabled ? 'dense on' : 'sparse only'}</Chip>
				<input type="text" bind:value={form.embed_url} spellcheck="false" placeholder="http://127.0.0.1:11434/v1" style="max-width: 280px" />
			</div>
		</div>
		<div class="list-row">
			<div><div class="lr-label">Embedding model</div></div>
			<input type="text" bind:value={form.embed_model} spellcheck="false" style="max-width: 320px" />
		</div>
		<div class="list-row">
			<div><div class="lr-label">Chunk size (words)</div></div>
			<input type="number" min="20" bind:value={form.chunk_words} style="max-width: 140px" />
		</div>
		<div class="list-row">
			<div><div class="lr-label">Chunk overlap</div></div>
			<input type="number" min="0" bind:value={form.chunk_overlap} style="max-width: 140px" />
		</div>
	</div>

	<!-- Retrieval -->
	<div class="list-section-title">Retrieval</div>
	<div class="list-menu">
		<div class="list-row">
			<div><div class="lr-label">Candidates (k)</div><div class="lr-hint">Chunks pulled before ranking.</div></div>
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
		<button class="btn" onclick={save} disabled={saving}>
			{#if saving}<span class="spin"></span> Saving…{:else}Save settings{/if}
		</button>
		{#if saved}<Chip tone="neutral" dot>Saved</Chip>{/if}
		<span class="mute2 small">Index: <span class="mono">{cfg.index_dir}</span></span>
	</div>

	<!-- API keys (BYOK) -->
	<div class="list-section-title">Provider API keys (BYOK)</div>
	<div class="list-menu">
		<p class="small mute2" style="margin: 0 0 6px">
			Stored locally in <span class="mono">.env.local</span> at <span class="mono">0600</span> and never shown again.
			Needed only for the Cerebras / Groq providers.
		</p>
		{#each ['cerebras', 'groq'] as p}
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
