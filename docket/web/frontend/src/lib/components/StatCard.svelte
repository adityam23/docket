<script>
	// The ONE stat face: ALL-CAPS label over a big Light tabular numeral on a
	// flat dark field. No borders, no shadows.
	let { label, value, unit = '', sub = '', accent = false } = $props();
</script>

<div class="stat" class:accent>
	<div class="label">{label}</div>
	<div class="value">{value}{#if unit}<small> {unit}</small>{/if}</div>
	{#if sub}<div class="sub">{sub}</div>{/if}
</div>

<style>
	.stat.accent .value { color: var(--accent-bright); }
</style>
