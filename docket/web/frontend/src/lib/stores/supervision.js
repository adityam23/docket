// One app-wide engine-supervision poller (F3), mirroring the ingest store's
// ref-counted pattern (CLAUDE.md: one implementation) — a SINGLE poller feeds
// both the TopNav indicator (always visible) and EnginePanel (detail), not one
// each. Polls GET /api/engine/supervision every 2 s while the tab is visible;
// backs off to a slow heartbeat while hidden so a backgrounded tab doesn't
// hammer the server during a minutes-long first-run model download.

import { writable } from 'svelte/store';
import { api, errMsg } from '$lib/api.js';

const ACTIVE_INTERVAL_MS = 2000;
const HIDDEN_INTERVAL_MS = 15000;

function createSupervision() {
	const { subscribe, update } = writable({ view: null, error: '' });
	let timer = null;
	let attached = 0;

	async function tick() {
		try {
			const view = await api.engineSupervision();
			update((s) => ({ ...s, view, error: '' }));
		} catch (e) {
			// Keep the last known view on a transient fetch error — surface the
			// error rather than silently going stale forever.
			update((s) => ({ ...s, error: errMsg(e) }));
		}
	}

	function hidden() {
		return typeof document !== 'undefined' && document.hidden;
	}

	function schedule() {
		if (timer) clearInterval(timer);
		timer = setInterval(tick, hidden() ? HIDDEN_INTERVAL_MS : ACTIVE_INTERVAL_MS);
	}

	function onVisibility() {
		if (attached <= 0) return;
		// Coming back into view: refresh immediately rather than waiting out
		// whatever's left of the slow hidden interval.
		if (!hidden()) tick();
		schedule();
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
			if (attached === 1) {
				tick();
				schedule();
				if (typeof document !== 'undefined') {
					document.addEventListener('visibilitychange', onVisibility);
				}
			}
			return () => {
				attached -= 1;
				if (attached <= 0) {
					stop();
					if (typeof document !== 'undefined') {
						document.removeEventListener('visibilitychange', onVisibility);
					}
				}
			};
		},
		/** Force an immediate re-read (e.g. right after a restart POST). */
		refresh: tick
	};
}

export const supervision = createSupervision();
