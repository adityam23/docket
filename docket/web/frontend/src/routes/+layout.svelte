<script>
	import '../app.css';
	import { onMount } from 'svelte';
	import { page } from '$app/stores';
	import TopNav from '$lib/components/TopNav.svelte';
	import { api } from '$lib/api.js';
	import { ingest } from '$lib/stores/ingest.js';
	import { ui } from '$lib/stores/ui.js';

	let { children } = $props();
	let health = $state(null);

	async function refresh() {
		try {
			health = await api.health();
		} catch {
			health = { ok: false, error: 'unreachable' };
		}
	}

	onMount(() => {
		ui.hydrate(); // apply persisted density / motion prefs to <html>
		refresh();
		const t = setInterval(refresh, 15000);
		// App-wide ingest poller (T12): keeps the header indicator live on every
		// route, so background ingestion stays visible after navigating away.
		const detach = ingest.attach();
		return () => {
			clearInterval(t);
			detach();
		};
	});

	// Turnstile page-enter (T19): re-keying the page container on navigation
	// replays the staggered .turn entrance on each route's content.
	let path = $derived($page.url.pathname);
</script>

<div class="app">
	<TopNav {health} />
	{#key path}
		<main class="main turn">
			{@render children()}
		</main>
	{/key}
</div>
