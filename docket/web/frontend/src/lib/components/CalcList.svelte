<script>
	// The ONE rendering of the calculator audit trail ([{expression, result}]),
	// shared by the Ask answer and the trace Timeline's calculate step.
	let { calculations = [] } = $props();
</script>

<ul class="calcs">
	{#each calculations as c}
		<li class="mono">
			<span class="expr">{c.expression}</span>
			<span class="eq">=</span>
			<span class="res" class:err={String(c.result).startsWith('error')}>{c.result}</span>
		</li>
	{/each}
</ul>

<style>
	.calcs { margin: 0; padding: 0; list-style: none; font-size: 0.85rem; }
	.calcs li { margin: 3px 0; color: var(--text-dim); font-variant-numeric: tabular-nums; }
	.eq { margin: 0 6px; color: var(--text-faint); }
	.res { color: var(--text); }
	.res.err { color: var(--bad); }
</style>
