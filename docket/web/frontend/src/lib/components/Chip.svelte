<script>
	// The ONE chip primitive. Metro: a tiny flat field with ALL-CAPS lettering —
	// square-ish (pills are wrong). `ok`/`warn` tones are reserved for the
	// reliability signal; everything else is monochrome or the accent.
	let { tone = 'neutral', dot = false, mono = true, children } = $props();
</script>

<span class="chip {tone}" class:mono>
	{#if dot}<span class="dot"></span>{/if}
	{@render children?.()}
</span>
