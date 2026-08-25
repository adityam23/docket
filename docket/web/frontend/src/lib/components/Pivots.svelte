<script>
	// The ONE in-page pivot tab strip (reused by Observability, ExplainDrawer, and
	// anywhere a page has typographic sub-tabs). Purely client state — no routing.
	// Metro pivot: active header bright + larger, the next header peeks to the
	// right in faint grey; headers slide via the shared .pivots styles.
	let { tabs = [], value = $bindable(tabs[0]?.id), onselect = null } = $props();

	function pick(id) {
		value = id;
		onselect?.(id);
	}
</script>

<div class="pivots" role="tablist">
	{#each tabs as t}
		<button
			class="pivot"
			class:active={value === t.id}
			role="tab"
			aria-selected={value === t.id}
			onclick={() => pick(t.id)}
		>
			{t.label}{#if t.badge != null}<span class="badge">{t.badge}</span>{/if}
		</button>
	{/each}
</div>
