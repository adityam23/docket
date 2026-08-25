<script>
	// The ONE meter: flat bar, square ends. Neutral by default; `accent` for
	// brand-tinted progress. (Green is reserved for reliability — never here.)
	let { value = 0, ok = false, accent = false } = $props();
	let width = $derived(`${Math.max(0, Math.min(1, value)) * 100}%`);
</script>

<div class="meter" class:ok class:accent>
	<span style="width: {width}"></span>
</div>
