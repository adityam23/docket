import adapter from '@sveltejs/adapter-static';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [
		sveltekit({
			// Single-page app: FastAPI serves the built bundle and falls back to
			// index.html for client-side routes (see web/app.py). No Node at runtime.
			adapter: adapter({
				pages: 'build',
				assets: 'build',
				fallback: 'index.html',
				precompress: false,
				strict: false
			})
		})
	],
	server: {
		// `npm run dev` proxies the observability API to the FastAPI server so the
		// dashboard has hot reload while talking to the real backend.
		proxy: {
			'/api': 'http://127.0.0.1:8760'
		}
	}
});
