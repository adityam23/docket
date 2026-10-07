<script>
	import '../app.css';
	import { onMount } from 'svelte';
	import { page } from '$app/stores';
	import TopNav from '$lib/components/TopNav.svelte';
	import { ingest } from '$lib/stores/ingest.js';
	import { supervision } from '$lib/stores/supervision.js';
	import { ui } from '$lib/stores/ui.js';

	let { children } = $props();

	onMount(() => {
		ui.hydrate(); // apply persisted density / motion prefs to <html>
		// App-wide pollers (T12 / F3): keep the header indicators live on every
		// route — ingestion progress and engine state both survive navigation
		// instead of each component re-polling its own copy.
		const detachIngest = ingest.attach();
		const detachSupervision = supervision.attach();
		return () => {
			detachIngest();
			detachSupervision();
		};
	});

	// Turnstile page-enter (T19): re-keying the page container on navigation
	// replays the staggered .turn entrance on each route's content.
	let path = $derived($page.url.pathname);
</script>

<div class="app">
	<TopNav />
	{#key path}
		<main class="main turn">
			{@render children()}
		</main>
	{/key}
</div>
