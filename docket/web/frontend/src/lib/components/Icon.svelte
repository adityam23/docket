<script>
	// The ONE monochrome line-glyph set.
	// Simple geometric strokes, currentColor, centred — no emoji anywhere in the
	// UI. The trust triad stays coloured DOTS (semantic signal), never glyphs.
	let { name, size = 18, stroke = 1.6 } = $props();

	// 24×24 viewBox, stroke-based geometry.
	const G = {
		chat: 'M4 5h16v11H9l-5 4z', // speech bubble
		doc: 'M6 3h8l4 4v14H6zM14 3v4h4M9 12h6M9 16h6',
		pulse: 'M3 12h4l2-6 4 12 2-6h6', // activity
		gear: 'M12 8.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7zM12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M19.1 4.9L17 7M7 17l-2.1 2.1',
		search: 'M10.5 4a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13zM15.5 15.5L21 21',
		trash: 'M4 7h16M9 7V4h6v3M7 7l1 14h8l1-14M10 11v6M14 11v6',
		upload: 'M12 16V4M7 9l5-5 5 5M4 20h16',
		refresh: 'M20 12a8 8 0 1 1-2.34-5.66M20 3v5h-5',
		send: 'M4 12h15M13 6l6 6-6 6',
		check: 'M4 12.5l5.5 5.5L20 6.5',
		alert: 'M12 3L2 21h20zM12 10v5M12 18.2v.1',
		clock: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM12 7v5l3.5 2',
		db: 'M12 3c-4.4 0-8 1.3-8 3s3.6 3 8 3 8-1.3 8-3-3.6-3-8-3zM4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3',
		grow: 'M4 19L10 13l3.5 3.5L20 9M20 9h-5.5M20 9v5.5',
		stream: 'M3 7h18M3 12h18M3 17h18', // layers of feed lines
		chart: 'M4 20V10M10 20V4M16 20v-8M22 20H2',
		flask: 'M10 3h4M10 3v6l-5.5 9.5A1.8 1.8 0 0 0 6 21h12a1.8 1.8 0 0 0 1.5-2.5L14 9V3M7.5 15h9',
		back: 'M14 6l-6 6 6 6',
		chevron: 'M9 6l6 6-6 6',
		down: 'M6 9l6 6 6-6',
		x: 'M6 6l12 12M18 6L6 18',
		dots: 'M5 12h.01M12 12h.01M19 12h.01',
		open: 'M14 4h6v6M20 4l-9 9M20 14v6H4V4h6', // external
		plus: 'M12 5v14M5 12h14',
		layers: 'M12 3l9 5-9 5-9-5zM3 13l9 5 9-5',
		pen: 'M4 20l1-4L16.5 4.5a2.1 2.1 0 0 1 3 3L8 19zM14 7l3 3',
		spark: 'M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z',
		list: 'M8 6h13M8 12h13M8 18h13M3.5 6h.01M3.5 12h.01M3.5 18h.01',
		shield: 'M12 3l8 3v6c0 4.4-3.4 7.8-8 9-4.6-1.2-8-4.6-8-9V6z',
		key: 'M14.5 4a5.5 5.5 0 1 0-4.9 8L8 13.6 9.4 15 8 16.4 10 18.4l1.6-1.6 1.4 1.4 2.4-2.4A5.5 5.5 0 0 0 14.5 4zM15 9h.01'
	};

	let d = $derived(G[name] || G.dots);
</script>

<svg
	width={size}
	height={size}
	viewBox="0 0 24 24"
	fill="none"
	stroke="currentColor"
	stroke-width={stroke}
	stroke-linecap="square"
	stroke-linejoin="miter"
	aria-hidden="true"
	class="icon"
>
	<path {d} />
</svg>

<style>
	.icon { display: block; flex-shrink: 0; }
</style>
