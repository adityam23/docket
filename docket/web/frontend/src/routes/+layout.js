// Pure client-side SPA: FastAPI serves the static bundle and the observability
// API. No SSR, no prerendered data — every view fetches live from /api.
export const ssr = false;
export const prerender = false;
