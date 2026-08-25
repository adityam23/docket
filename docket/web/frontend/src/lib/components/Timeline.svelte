<script>
	// The ONE timeline renderer (CLAUDE.md reuse): turns a trace's ordered steps
	// into a vertical, expandable timeline. Used verbatim by Observability (T05),
	// the per-chat page (T06) and — in `live` mode with compact steps — inline on
	// Ask (T07). Steps render with the shared line-glyph set; expandable bodies
	// show verbatim prompts / context / hits. Monochrome rail; the accent marks
	// model-touching steps.
	import { stepMeta } from '$lib/format.js';
	import Chip from '$lib/components/Chip.svelte';
	import HitCard from '$lib/components/HitCard.svelte';
	import Icon from '$lib/components/Icon.svelte';

	let { steps = [], live = false } = $props();

	let openIdx = $state({});
	function toggle(i) {
		openIdx[i] = !openIdx[i];
	}

	// Full hits (array of objects) get HitCards; a compact step carries only a
	// count. Prompts/context/response text expand to <pre>. Everything else is a
	// one-line summary.
	function hasBody(s) {
		if (Array.isArray(s.hits) && s.hits.length) return true;
		if (Array.isArray(s.messages) && s.messages.length) return true;
		if (Array.isArray(s.items) && s.items.length) return true;
		if (typeof s.context === 'string' && s.context) return true;
		if (typeof s.text === 'string' && s.text) return true;
		return false;
	}

	function summary(s) {
		const bits = [];
		if (typeof s.hop === 'number') bits.push(`hop ${s.hop + 1}`);
		if (s.path) bits.push(`${s.path} path`);
		if (s.op) bits.push(s.op);
		if (s.query) bits.push(`“${s.query}”`);
		if (typeof s.hits === 'number') bits.push(`${s.hits} chunks`);
		else if (Array.isArray(s.hits)) bits.push(`${s.hits.length} chunks`);
		if (typeof s.items === 'number') bits.push(`${s.items} values`);
		else if (Array.isArray(s.items)) bits.push(`${s.items.length} values`);
		if (typeof s.chars === 'number') bits.push(`${s.chars} chars`);
		return bits.join(' · ');
	}
</script>

<div class="timeline" class:live>
	{#each steps as s, i}
		{@const m = stepMeta(s.kind)}
		{@const body = hasBody(s)}
		<div class="node">
			<div class="rail">
				<span class="dot {m.tone}"><Icon name={m.icon} size={13} stroke={1.8} /></span>
				{#if i < steps.length - 1}<span class="line"></span>{/if}
			</div>
			<div class="body">
				<button class="head" class:plain={!body} onclick={() => body && toggle(i)}>
					<span class="lbl caps">{m.label}</span>
					{#if summary(s)}<span class="sum mono">{summary(s)}</span>{/if}
					{#if s.reliability}<Chip tone={s.reliability === 'low' ? 'bad' : s.reliability === 'medium' ? 'warn' : 'ok'} dot>{s.reliability}</Chip>{/if}
					{#if body}<span class="caret" class:open={openIdx[i]}><Icon name="down" size={12} /></span>{/if}
				</button>

				{#if body && openIdx[i]}
					<div class="detail">
						{#if Array.isArray(s.hits) && s.hits.length}
							<div class="hits">
								{#each s.hits as h, hi}<HitCard hit={h} n={hi + 1} />{/each}
							</div>
						{/if}
						{#if Array.isArray(s.messages)}
							{#each s.messages as msg}
								<div class="msg">
									<div class="role">{msg.role}</div>
									<pre>{msg.content}</pre>
								</div>
							{/each}
						{/if}
						{#if typeof s.context === 'string' && s.context}
							<pre>{s.context}</pre>
						{/if}
						{#if Array.isArray(s.items) && s.items.length}
							<ul class="items">
								{#each s.items as it}
									<li><span class="mono">{it.value}{it.unit}</span> — {it.label}</li>
								{/each}
							</ul>
						{/if}
						{#if typeof s.text === 'string' && s.text && !Array.isArray(s.messages)}
							<pre>{s.text}</pre>
						{/if}
					</div>
				{/if}
			</div>
		</div>
	{/each}
	{#if !steps.length}
		<div class="mute2 small">No steps recorded.</div>
	{/if}
</div>

<style>
	.timeline { display: flex; flex-direction: column; }
	.node { display: grid; grid-template-columns: 30px 1fr; gap: 12px; }
	.rail { display: flex; flex-direction: column; align-items: center; }
	.dot {
		width: 26px; height: 26px; border-radius: 0; display: grid; place-items: center;
		background: var(--surface-3); color: var(--text-dim); flex-shrink: 0;
	}
	.dot.accent { background: var(--accent-soft); color: var(--accent-bright); }
	.dot.info { background: var(--info-soft); color: var(--info); }
	.dot.warn { background: var(--neutral-soft); color: var(--text-dim); } /* non-trust: monochrome */
	.dot.ok { background: var(--surface-3); color: var(--text); }
	.line { flex: 1; width: 1px; background: var(--line-strong); margin: 2px 0; min-height: 12px; }
	.body { padding-bottom: 14px; min-width: 0; }
	.head {
		display: flex; align-items: center; gap: 10px; width: 100%; text-align: left;
		background: none; border: none; color: var(--text); font: inherit; cursor: pointer;
		padding: 4px 0;
	}
	.head.plain { cursor: default; }
	.lbl { font-size: 0.72rem; }
	.sum { color: var(--text-mute); font-size: 0.78rem; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
	.caret { color: var(--text-mute); margin-left: auto; transition: transform 0.15s; display: inline-flex; }
	.caret.open { transform: rotate(180deg); }

	.detail { margin-top: 8px; display: flex; flex-direction: column; gap: 8px; }
	.hits { display: flex; flex-direction: column; gap: 8px; }
	.msg { background: var(--surface-1); }
	.role {
		font: 400 0.68rem/1 var(--mono); text-transform: uppercase; letter-spacing: 0.1em;
		color: var(--text-mute); padding: 7px 10px; border-bottom: 1px solid var(--line);
	}
	pre {
		margin: 0; padding: 10px 12px; background: var(--surface-1);
		font: 0.8rem/1.5 var(--mono); color: var(--text-dim);
		white-space: pre-wrap; word-break: break-word; max-height: 340px; overflow: auto;
	}
	.msg pre { max-height: none; }
	.items { margin: 0; padding-left: 18px; color: var(--text-dim); font-size: 0.85rem; }
	.items li { margin: 3px 0; }
	.live .dot { width: 22px; height: 22px; }
	.live .body { padding-bottom: 10px; }
</style>
