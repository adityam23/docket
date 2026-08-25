<script>
	// Filings — browse · fuzzy-search · add · remove, plus corpus telemetry
	// (retrieval mode, embedding coverage) and capacity. The general "what's in
	// the corpus" numbers live here; per-answer telemetry lives in Observability.
	// Ingest add/remove reuse the app-wide ingest store (T12) so a job stays
	// visible across navigation. Dense work surface; typographic rows, tabular
	// figures, no chrome.
	import { onMount } from 'svelte';
	import { api, MAX_PDF_BYTES } from '$lib/api.js';
	import { ingest } from '$lib/stores/ingest.js';
	import { num, pct, bytesish, gb, stage as stageInfo } from '$lib/format.js';
	import Chip from '$lib/components/Chip.svelte';
	import Icon from '$lib/components/Icon.svelte';
	import Meter from '$lib/components/Meter.svelte';
	import StatCard from '$lib/components/StatCard.svelte';

	let corpus = $state(null);
	let capacity = $state(null);
	let overview = $state(null);
	let error = $state('');
	let query = $state('');

	let ingestError = $state('');
	let busy = $state(false);
	let removing = $state('');
	let dragging = $state(false);
	let picker;

	let job = $derived($ingest.job);
	let running = $derived($ingest.running === true);

	async function loadCorpus() {
		try {
			[corpus, capacity, overview] = await Promise.all([api.corpus(), api.capacity(), api.overview()]);
		} catch (e) {
			error = String(e).replace(/^Error:\s*/, '');
		}
	}

	let wasRunning = false;
	$effect(() => {
		const r = $ingest.running;
		if (wasRunning && !r) loadCorpus();
		wasRunning = r;
	});

	const mb = Math.round(MAX_PDF_BYTES / (1024 * 1024));

	async function onFiles(fileList) {
		ingestError = '';
		let pdfs = Array.from(fileList || []).filter((f) => f.name.toLowerCase().endsWith('.pdf'));
		if (!pdfs.length) {
			ingestError = 'Please choose PDF files.';
			return;
		}
		const oversize = pdfs.filter((f) => f.size > MAX_PDF_BYTES);
		pdfs = pdfs.filter((f) => f.size <= MAX_PDF_BYTES);
		let warn = '';
		if (oversize.length) {
			warn = `Skipped ${oversize.length} file(s) over the ${mb} MB limit: ${oversize.map((f) => f.name).join(', ')}.`;
		}
		if (!pdfs.length) {
			ingestError = warn || 'No PDF files under the size limit.';
			return;
		}
		try {
			busy = true;
			const started = await api.ingestFiles(pdfs);
			ingest.kick(started);
			ingestError = warn;
		} catch (err) {
			ingestError = String(err).replace(/^Error:\s*/, '');
		} finally {
			busy = false;
		}
	}

	function onPick(e) {
		const files = e.target.files;
		e.target.value = '';
		onFiles(files);
	}
	function onDrop(e) {
		e.preventDefault();
		dragging = false;
		if (running || busy) return;
		onFiles(e.dataTransfer?.files);
	}

	async function loadSamples() {
		ingestError = '';
		try {
			busy = true;
			await api.loadSamples();
			await loadCorpus();
		} catch (err) {
			ingestError = String(err).replace(/^Error:\s*/, '');
		} finally {
			busy = false;
		}
	}

	async function removeDoc(docId) {
		if (!confirm(`Remove "${docId}" from the index? This cannot be undone.`)) return;
		try {
			removing = docId;
			await api.removeDoc(docId);
			await loadCorpus();
		} catch (err) {
			ingestError = String(err).replace(/^Error:\s*/, '');
		} finally {
			removing = '';
		}
	}

	onMount(loadCorpus);

	let docs = $derived(corpus?.documents ?? []);
	let totals = $derived(corpus?.totals);
	let emb = $derived(overview?.embeddings);

	// Simple, dependency-free fuzzy filter: subsequence match on doc id / source.
	function fuzzy(needle, hay) {
		if (!needle) return true;
		needle = needle.toLowerCase();
		hay = (hay || '').toLowerCase();
		let i = 0;
		for (const ch of hay) if (ch === needle[i]) i++;
		return i === needle.length;
	}
	let shown = $derived(docs.filter((d) => fuzzy(query, `${d.doc_id} ${d.source}`)));
</script>

<h1 class="page-title">filings</h1>
<p class="page-sub" style="margin-bottom: 24px">
	Everything indexed on this machine — search it, add PDFs, or remove a filing. Retrieval reads only from here.
</p>

<!-- Corpus telemetry -->
{#if totals}
	<div class="cards dense">
		<StatCard label="Filings" value={num(totals.documents)} sub="{num(totals.chunks)} chunks · {bytesish(totals.chars)} chars" />
		<StatCard
			label="Embedding coverage"
			value={pct(totals.embedding_coverage)}
			sub="{num(totals.vectorized_chunks)} / {num(totals.chunks)} vectorized"
		/>
		<div class="stat">
			<div class="label">Retrieval</div>
			<div class="value" style="font-size: var(--fs-md); margin-top: 10px">
				<Chip tone="neutral">{emb?.enabled ? 'hybrid' : 'sparse-only'}</Chip>
			</div>
			<div class="sub">{emb?.retrieval_mode || '—'}{#if emb?.dim} · {emb.dim}-dim{/if}</div>
		</div>
		{#if capacity}
			<StatCard
				label="Room to grow"
				value="~{num(capacity.remaining_documents_est)}"
				sub="more filings · limited by {capacity.binding_constraint === 'ram' ? 'RAM' : 'disk'}"
			/>
		{/if}
	</div>
{/if}

<!-- Add documents -->
<div class="section-title" style="margin-top: 32px">add filings</div>
<p class="small mute2" style="margin: -6px 0 12px">
	Pick PDFs or drop them below. They are sent to your local server and indexed on this machine.
	Files over {mb} MB are skipped, and an already-indexed filing is skipped (remove it first to re-ingest).
</p>
<input bind:this={picker} type="file" multiple accept="application/pdf" style="display: none" onchange={onPick} />
<div
	class="dropzone"
	class:dragging
	role="button"
	tabindex="0"
	ondragover={(e) => {
		e.preventDefault();
		dragging = true;
	}}
	ondragleave={() => (dragging = false)}
	ondrop={onDrop}
	onclick={() => !(running || busy) && picker.click()}
	onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && !(running || busy) && picker.click()}
>
	{#if running || busy}
		<span class="spin"></span><span>Working…</span>
	{:else}
		<div class="dz-icon"><Icon name="upload" size={30} stroke={1.3} /></div>
		<div><strong>Choose PDFs</strong> or drop them here</div>
		<div class="small mute2">multi-select · {mb} MB per file</div>
	{/if}
</div>
<div class="row" style="margin-top: 14px">
	<button class="btn ghost" onclick={loadSamples} disabled={running || busy}>Load sample filings</button>
</div>

{#if capacity}
	<div class="small mute2" style="margin-top: 12px">
		About <strong>{num(capacity.remaining_documents_est)}</strong> more filings fit — limited by
		{capacity.binding_constraint === 'ram' ? 'RAM' : 'disk'}
		({gb(capacity.device.disk_free_bytes)} GB disk free{#if capacity.device.ram_total_bytes}, {gb(capacity.device.ram_total_bytes)} GB RAM{/if}).
		The corpus stays in memory for retrieval, so RAM bounds growth too.
	</div>
{/if}
{#if ingestError}
	<div class="banner bad" style="margin-top: 12px"><Icon name="alert" size={16} /> {ingestError}</div>
{/if}

{#if job}
	<div style="margin-top: 18px">
		<div class="row spread small muted" style="margin-bottom: 8px">
			<span>Job <span class="mono">{job.id}</span> · {job.folder}</span>
			<span>{job.completed} / {job.total} filings</span>
		</div>
		<Meter value={job.total ? job.completed / job.total : 0} accent={job.status !== 'done'} />
		<div class="job-docs cascade">
			{#each job.docs as d, di}
				{@const si = stageInfo(d.stage)}
				<div class="job-doc" style="--i: {di}">
					<span class="mono jd-name">{d.doc_id}</span>
					<Chip tone={si.tone} dot={['ocr', 'chunk', 'embed', 'index'].includes(d.stage)}>{si.label}</Chip>
					{#if d.chunks}<span class="mute2 small">{d.chunks} chunks</span>{/if}
					{#if d.error}<span class="small err">{d.error}</span>{/if}
				</div>
			{/each}
		</div>
	</div>
{/if}

<!-- Browse + fuzzy search -->
<div class="section-title" style="margin-top: 36px">indexed filings</div>
{#if error}
	<div class="banner bad"><Icon name="alert" size={16} /> {error}</div>
{:else if !corpus}
	<div class="empty"><div class="spin"></div></div>
{:else if !docs.length}
	<div class="empty">
		<div class="big"><Icon name="doc" size={40} stroke={1.2} /></div>
		<div>No filings indexed yet.</div>
		<div class="small mute2" style="margin-top: 6px">Add PDFs above to get started.</div>
	</div>
{:else}
	<div class="row spread wrap" style="gap: 12px; margin-bottom: 6px">
		<span class="caps">{num(totals.documents)} in the index</span>
		<input
			type="search"
			placeholder="Fuzzy search filings…"
			bind:value={query}
			style="max-width: 260px"
			spellcheck="false"
		/>
	</div>
	<table class="table">
		<thead>
			<tr>
				<th>Filing</th>
				<th>Stage</th>
				<th class="num">Pages</th>
				<th class="num">Chunks</th>
				<th style="width: 180px">Embedding coverage</th>
				<th class="num">Remove</th>
			</tr>
		</thead>
		<tbody>
			{#each shown as d}
				{@const si = stageInfo(d.stage)}
				<tr>
					<td class="doc-id">{d.doc_id}</td>
					<td><Chip tone={si.tone} dot>{si.label}</Chip></td>
					<td class="num">{d.pages}</td>
					<td class="num">{d.chunks}</td>
					<td>
						<div class="row" style="gap: 10px">
							<div style="flex: 1"><Meter value={d.coverage} ok={d.coverage >= 0.999} /></div>
							<span class="mono small mute2" style="width: 34px">{pct(d.coverage)}</span>
						</div>
					</td>
					<td class="num">
						<button
							class="icon-btn"
							title="Remove this filing from the index"
							aria-label="Remove {d.doc_id}"
							disabled={running || busy || removing === d.doc_id}
							onclick={() => removeDoc(d.doc_id)}
						>
							{#if removing === d.doc_id}<span class="spin"></span>{:else}<Icon name="trash" size={14} />{/if}
						</button>
					</td>
				</tr>
			{/each}
			{#if !shown.length}
				<tr><td colspan="6" class="mute2 small" style="text-align: center; padding: 22px">No filings match "{query}".</td></tr>
			{/if}
		</tbody>
	</table>
{/if}

<style>
	.dropzone {
		display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px;
		padding: 30px 20px; border: 1.5px dashed var(--line-strong); border-radius: var(--radius-tile);
		background: var(--surface-1); color: var(--text-dim); cursor: pointer; text-align: center;
		transition: border-color var(--dur-fast), background var(--dur-fast), color var(--dur-fast);
	}
	.dropzone:hover { border-color: var(--text-mute); }
	.dropzone.dragging { border-color: var(--accent); background: var(--accent-soft); color: var(--text); }
	.dz-icon { color: var(--text-faint); }

	.job-docs { margin-top: 14px; display: flex; flex-direction: column; gap: 8px; }
	.job-doc {
		display: flex; align-items: center; gap: 12px; padding: 9px 12px;
		background: var(--surface-2);
	}
	.jd-name { min-width: 160px; }
	.err { color: var(--bad); }
</style>
