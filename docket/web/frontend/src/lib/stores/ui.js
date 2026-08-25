// One store for user presentation preferences (Appearance settings): board
// density and a motion toggle. Persisted to localStorage and mirrored onto
// <html> data-attributes so app.css can react without prop drilling. One place,
// one implementation (CLAUDE.md reuse) — every view reads the same store.

import { writable } from 'svelte/store';

const KEY = 'docket.ui';
const DEFAULTS = { density: 'comfortable', motion: 'full' }; // motion: full | reduced

function read() {
	if (typeof localStorage === 'undefined') return { ...DEFAULTS };
	try {
		return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(KEY) || '{}') };
	} catch {
		return { ...DEFAULTS };
	}
}

// Reflect prefs onto <html> so the density/motion CSS hooks apply globally.
// `comfortable` is the CSS default (airy board) — only the dense override needs
// an attribute; individual work surfaces still opt into `.dense` locally.
function apply(v) {
	if (typeof document === 'undefined') return;
	const el = document.documentElement;
	if (v.density === 'dense') el.setAttribute('data-density', 'dense');
	else el.removeAttribute('data-density');
	if (v.motion === 'reduced') el.setAttribute('data-motion', 'reduced');
	else el.removeAttribute('data-motion');
}

function createUi() {
	const { subscribe, set, update } = writable(read());
	return {
		subscribe,
		/** Apply persisted prefs to <html>. Called once on app mount. */
		hydrate() {
			const v = read();
			set(v);
			apply(v);
		},
		set(patch) {
			update((v) => {
				const next = { ...v, ...patch };
				try {
					localStorage.setItem(KEY, JSON.stringify(next));
				} catch {
					/* private mode / disabled storage — keep in-memory only */
				}
				apply(next);
				return next;
			});
		}
	};
}

export const ui = createUi();
