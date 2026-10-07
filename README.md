# docket

Research assistant over SEC EDGAR filings: retrieval, rerank, embeddings and
grounded generation all live in the **mettle** engine, consumed over its
OpenAI-compatible `/v1` HTTP API; docket owns the UI, the agent, the lineage
and the trust layer.

Self-contained. The release wheel ships the dashboard; no Node, no external
services required to start. The only supported engine is **mettle** — any
server that speaks the same OpenAI-compatible `/v1` contract (chat with
per-token logprobs, embeddings, rerank, collections) is a drop-in backend.

## Install

Linux and macOS:

```sh
curl -fsSL https://raw.githubusercontent.com/adityam23/docket/main/install.sh -o install.sh
sh install.sh
```

The installer downloads the latest release wheel, installs it as an isolated
[uv tool](https://docs.astral.sh/uv/concepts/tools/), and puts `dk` on your
PATH. State lands in XDG dirs (`~/.local/share/docket`, `~/.config/docket`).

Optional local engine:

```sh
sh install.sh --with-engine /path/to/mettle
```

This copies the binary next to `dk` as `docket-engine` and records it in the
manifest. It is not started or configured for you.

Uninstall:

```sh
curl -fsSL https://raw.githubusercontent.com/adityam23/docket/main/uninstall.sh -o uninstall.sh
sh uninstall.sh          # add --data to also delete the ingested index
```

## Desktop app (AppImage / DMG)
Click-to-run builds are produced by CI (`.github/workflows/release-desktop.yml`,
on version tags): a thin Tauri shell spawns the frozen `docket-server` sidecar
on a free port, points it at the bundled `mettled` engine, and opens the OS
webview at the dashboard. Model weights are downloaded by the engine on first
run (mettle's own first-run flow). Artifacts are **unsigned** by design:
macOS shows a one-time right-click → Open warning; Linux AppImages need
`libfuse2` (or run with `--appimage-extract`) and the system
`webkit2gtk-4.1` webview.

Local Linux build (everything except the engine binary, which you point at via
`DK_ENGINE_BINARY`):

```sh
packaging/build-sidecar.sh                       # PyInstaller sidecar → dist/docket-server/
cp -a dist/docket-server desktop/src-tauri/resources/server/
mkdir -p desktop/src-tauri/resources/engine      # put (or symlink) a mettled binary here
cd desktop/src-tauri && npx --yes @tauri-apps/cli build
```

The CLI/uv-tool install below remains the developer path.

## Quickstart (dev)

```sh
git clone https://github.com/adityam23/docket && cd docket
uv sync                  # base deps only
uv run dk serve          # dashboard + API on http://127.0.0.1:8760
uv run dk chat "hello"   # one-shot generation smoke test
uv run pytest            # offline suite, no backend needed
```

Extras: `uv sync --extra ingest` (PDF text extraction), `--extra agent`
(LangGraph). The dashboard is a SvelteKit (Svelte 5) SPA: rebuild it with
`cd docket/web/frontend && npm ci && npm run build`; releases
ship it prebuilt.

## Configuration

Every knob is a `DK_`-prefixed env var or a line in `.env`. All retrieval
capability is engine-side: the engine serves chat with per-token logprobs,
embeddings, `/v1/rerank` and the collections surface on ONE base URL — there is
no separate embed or rerank endpoint. If the engine is unreachable, docket
features fail loudly (no silent local fallback).

| Variable | Purpose |
|---|---|
| `DK_BACKEND_URL` | `/v1` base of the mettle engine. Default `http://127.0.0.1:11434/v1` |
| `DK_CHAT_MODEL` | Chat model id (or `role:<kind>` resolved from the engine's active profile) |
| `DK_EMBED_MODEL` | Embedding model id for the offline `dk embed` path only; ingest POSTs chunks to the engine's collection, which embeds with its own default model |
| `DK_RERANK_MODEL` | Cross-encoder reranker id. Default `qwen3-reranker-0.6b:latest` |
| `DK_ENGINE_BINARY` | Path to the `mettled` binary; set to let docket supervise (spawn + crash-restart) it |
| `DK_METTLE_CONFIG` | Passed to the engine as `METTLE_CONFIG` when spawning |
| `DK_ENGINE_KEY` | Bearer for the engine-management surface, when the engine enforces auth |
| `DK_ENGINE_COLLECTION` | Collection name filings are ingested into |

## How it works

```
ingest:  pdf -> ocr -> chunk -> POST chunks to the engine collection
answer:  query -> engine search (dense/keyword/hybrid) -> cross-encoder rerank
         -> grounded LLM hops -> cited answer + per-token confidence label
```

docket stores only chunk **lineage** locally (doc ids, pages, sources, filing
metadata) — chunk text and vectors live engine-side. Search, rerank, embed and
chat are engine capabilities reached through the one OpenAI-`/v1` client; the
offline test suite runs the full pipeline against a fake provider.

## Benchmarks

FinanceBench, 150 questions, production grounded prompt, k=10 context. The
full-RAG accuracy runs used `embed-gemma` + `qwen3-reranker-0.6b`; the shipped
default embedder is now `qwen3-embedding-0.6b` (higher recall@10 below), and the
full-RAG numbers have not yet been re-measured under it. Two metrics,
never mixed: **answer accuracy** (numeric match within tolerance for metrics
questions, Claude-judged equivalence for free-form ones) and **gold-evidence
recall@k** (token overlap of retrieved chunks vs the gold evidence span).
Reproduction commands:
[`benchmarks/financebench/README.md`](benchmarks/financebench/README.md).

Answer accuracy:

| Generator | Oracle evidence | Full RAG @6 | Full RAG @10 (shipped) |
|---|---|---|---|
| Qwen3.5-4B Opus-distill (default chat model) | 80.7% | 68.0% | **82.0%** |
| gemma4:e2b | 55.3% | 40.0% | — |

Retrieval recall@6 / recall@10 at threshold 0.5 (36,920 chunks):

| Embedder | no reranker | + bge-reranker-v2-m3 | + qwen3-reranker-0.6b |
|---|---|---|---|
| embed-gemma (768d) | 0.527 / 0.613 | 0.773 / 0.840 | **0.840 / 0.900** |
| bge-m3 (1024d) | 0.513 / 0.600 | 0.747 / 0.807 | 0.793 / 0.847 |
| qwen3-embedding (1024d) | 0.540 / 0.667 | 0.793 / 0.860 | **0.840 / 0.913** |

Raising the context from 6 to 10 chunks recovers +14 pt of answer accuracy,
closing the gap to the oracle ceiling; at k=10 retrieval is no longer the
distill's bottleneck. The cross-encoder reranker remains the dominant retrieval
lever (+31 pts recall@6 over the fused order).

## License

GPL-3.0. See [LICENSE](LICENSE).
