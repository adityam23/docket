<script>
	// A model answer rendered as formatted, citation-pilled text. Shared by the
	// Ask view and the per-chat permalink so both read identically.
	import { renderAnswer } from '$lib/markdown.js';

	let { text = '' } = $props();
</script>

<!-- renderAnswer escapes all model text before adding its fixed tag set. -->
<div class="answer">{@html renderAnswer(text)}</div>

<style>
	.answer { font-size: var(--fs-normal); line-height: 1.7; max-width: 78ch; }
	.answer :global(p) { margin: 0 0 0.8em; }
	.answer :global(h4) {
		margin: 1.1em 0 0.4em; font-size: 0.78rem; font-weight: 600;
		letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-mute);
	}
	.answer :global(h4:first-child) { margin-top: 0; }
	.answer :global(strong) { color: var(--text); font-weight: 600; }
	.answer :global(code) { font: 0.88em var(--mono); }
	.answer :global(blockquote) {
		margin: 0 0 0.8em; padding: 2px 0 2px 14px;
		border-left: 2px solid var(--accent); color: var(--text-mute);
	}
	.answer :global(ul), .answer :global(ol) { margin: 0 0 0.8em; padding-left: 1.3em; }
	.answer :global(table) {
		border-collapse: collapse; margin: 0 0 0.9em; font: 0.85rem/1.5 var(--mono);
	}
	.answer :global(th), .answer :global(td) {
		padding: 5px 14px 5px 0; text-align: left; border-bottom: 1px solid var(--line);
	}
	.answer :global(th) { color: var(--text-mute); font-weight: 600; }
	@media (max-width: 720px) { .answer { max-width: 100%; } }
</style>
