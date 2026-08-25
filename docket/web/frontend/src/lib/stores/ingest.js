// One app-wide ingest poller (T12). The server already keeps the job registry
// process-global, so a job survives navigation; this store makes its status
// visible everywhere (the sidebar indicator + the Documents view) from a SINGLE
// poller instead of one per component (CLAUDE.md: one implementation).
//
// The layout mounts `attach()` for the app's lifetime, so status is tracked no
// matter which route is showing. Components read `$ingest`; the Documents view
// calls `kick()` right after starting a job so the indicator updates instantly.

import { writable } from 'svelte/store';
import { api } from '$lib/api.js';

function createIngest() {
	const { subscribe, set } = writable({ running: false, job: null });
	let timer = null;
	let attached = 0;

	async function tick() {
		try {
			set(await api.ingestStatus());
		} catch {
			/* keep the last known status on a transient error */
		}
	}

	function poll() {
		if (timer) return;
		tick();
		timer = setInterval(tick, 1000);
	}

	function stop() {
		if (timer) {
			clearInterval(timer);
			timer = null;
		}
	}

	return {
		subscribe,
		/** Keep the poller alive while any component is listening (ref-counted). */
		attach() {
			attached += 1;
			poll();
			return () => {
				attached -= 1;
				if (attached <= 0) stop();
			};
		},
		/** Seed state from a freshly-started job and ensure polling is live. */
		kick(job) {
			if (job) set({ running: true, job });
			poll();
		},
		refresh: tick
	};
}

export const ingest = createIngest();
