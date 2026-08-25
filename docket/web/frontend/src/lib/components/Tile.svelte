<script>
	// The ONE live-tile primitive (CLAUDE.md reuse).
	// A tile is a FLAT SOLID FILL on the wall grid — no border, no shadow, square
	// corners. It flips its face ONLY when its underlying datum changes ("calm
	// alive"), via the shared WAAPI flip (motion.js) so the restart is always
	// clean. Variants: `hero` = boldly accent-filled (the few); everything else
	// is a dark monochrome field.
	import { onMount } from 'svelte';
	import { flipFace, tilt } from '$lib/motion.js';
	import Icon from './Icon.svelte';

	let {
		href = null,
		label = '',
		variant = '',      // '' | 'hero'
		size = 's-small',  // s-small (2×1) | s-medium (2×2) | s-wide (4×2)
		badge = null,      // count badge, top-right
		glyph = null,      // centred line-glyph face (iconic template)
		flipKey = undefined,
		children,
		foot
	} = $props();

	let el = $state(null);
	let mounted = false;
	let prev;

	// Flip on change, never on first render (that's the entrance cascade).
	$effect(() => {
		const k = flipKey;
		if (!mounted) {
			prev = k;
			return;
		}
		if (k !== prev && el) {
			prev = k;
			// The DOM updates reactively underneath; the WAAPI dip reads as the
			// Metro face-change without cloning nodes.
			flipFace(el, () => {});
		}
	});
	onMount(() => (mounted = true));

	const cls = $derived(['tile', size, variant].filter(Boolean).join(' '));
</script>

<svelte:element this={href ? 'a' : 'div'} {href} class={cls} bind:this={el} use:tilt>
	{#if badge !== null}<span class="badge-n">{badge}</span>{/if}
	{#if label}
		<div class="tile-label">
			{#if glyph}<span class="tile-glyph"><Icon name={glyph} size={14} stroke={1.7} /></span>{/if}
			{label}
		</div>
	{/if}
	<div class="tile-body">
		{@render children?.()}
	</div>
	{#if foot}<div class="tile-foot">{@render foot()}</div>{/if}
</svelte:element>

<style>
	.tile-body { display: flex; flex-direction: column; gap: 6px; flex: 1; min-width: 0; }
	.tile-glyph { display: inline-flex; margin-right: 7px; vertical-align: -2px; color: var(--text-dim); }
	.tile.hero .tile-glyph { color: rgba(255, 255, 255, 0.8); }
</style>
