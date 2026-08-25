// The ONE Metro motion implementation (docs/design-language.md §7, TODO T19).
// CSS classes in app.css carry the staggered entrances (cascade / turnstile —
// `--i` × ~100ms is the reusable heart); this module carries what needs JS:
//
//   • tilt      — Svelte action: press feedback, rotateX/Y toward the touch +
//                 slight scale-down, ~100ms sine settle (GPU transforms only).
//   • flipFace  — reliable tile-flip via the Web Animations API: first half
//                 rotates to 90°, the caller swaps content, second half settles
//                 back. WAAPI restarts cleanly on every call (pass 1's
//                 rAF/class-toggle restart felt broken — that's fixed here).
//
// Every effect checks motionAllowed() so BOTH prefers-reduced-motion and the
// in-app toggle (ui store → <html data-motion>) silence it.

const REDUCED_QUERY = '(prefers-reduced-motion: reduce)';

export function motionAllowed() {
	if (typeof document === 'undefined') return false;
	if (document.documentElement.hasAttribute('data-motion')) {
		return !document.documentElement.hasAttribute('data-motion'); // attr present == reduced
	}
	if (typeof matchMedia === 'function') return !matchMedia(REDUCED_QUERY).matches;
	return true;
}

const MAX_TILT_DEG = 5;

/** Svelte action: tactile tilt-press on flat Metro surfaces. */
export function tilt(node) {
	function settle() {
		node.style.transform = '';
	}

	function onDown(e) {
		if (!motionAllowed() || e.button > 0) return;
		const r = node.getBoundingClientRect();
		const px = (e.clientX - r.left) / r.width - 0.5;
		const py = (e.clientY - r.top) / r.height - 0.5;
		node.style.transform = `perspective(900px) rotateX(${(-py * MAX_TILT_DEG).toFixed(2)}deg) rotateY(${(px * MAX_TILT_DEG).toFixed(2)}deg) scale(0.985)`;

		function onUp() {
			settle();
			window.removeEventListener('pointerup', onUp);
			window.removeEventListener('pointercancel', onUp);
		}
		window.addEventListener('pointerup', onUp);
		window.addEventListener('pointercancel', onUp);
	}

	function onLeave() {
		if (node.style.transform) settle();
	}

	node.addEventListener('pointerdown', onDown);
	node.addEventListener('pointerleave', onLeave);
	return {
		destroy() {
			node.removeEventListener('pointerdown', onDown);
			node.removeEventListener('pointerleave', onLeave);
		}
	};
}

/**
 * Flip a tile to reveal new content: rotateX 0→90° (ease-in), then the caller's
 * `swap()` runs at the hinge, then 90°→0° (spring out). Resolves when settled.
 * No-ops (calls swap immediately) under reduced motion.
 */
export function flipFace(node, swap) {
	if (!motionAllowed()) {
		swap();
		return Promise.resolve();
	}
	const half = (deg, dur, easing) =>
		new Promise((done) => {
			const anim = node.animate(
				[{ transform: 'perspective(700px) rotateX(0deg)' }, { transform: `perspective(700px) rotateX(${deg}deg)` }],
				{ duration: dur, easing, fill: 'forwards' }
			);
			anim.onfinish = () => done();
			// Safety: never leave a tile stuck if onfinish is swallowed.
			setTimeout(done, dur + 80);
		});

	return half(-90, 250, 'cubic-bezier(0.55, 0.06, 0.68, 0.19)').then(() => {
		swap();
		return half(0, 350, 'cubic-bezier(0.34, 1.32, 0.5, 1)');
	});
}
