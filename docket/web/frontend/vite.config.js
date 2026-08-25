import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		// `npm run dev` proxies the observability API to the FastAPI server so the
		// dashboard has hot reload while talking to the real backend.
		proxy: {
			'/api': 'http://127.0.0.1:8760'
		}
	}
});
