# docket

Local RAG engine for long documents: hybrid retrieval, cross-encoder rerank,
grounded generation, evaluated on FinanceBench.

Self-contained. The release wheel ships the dashboard; no engine install, no
Node, no external services required to start. Works with any OpenAI-compatible
`/v1` server (llama-server, Ollama) or a free-tier API key (Cerebras, Groq).

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
sh install.sh --with-engine /path/to/llama-server
```

This copies the binary next to `dk` as `docket-engine` and records it in the
manifest. It is not started or configured for you.

Uninstall:

```sh
curl -fsSL https://raw.githubusercontent.com/adityam23/docket/main/uninstall.sh -o uninstall.sh
sh uninstall.sh          # add --data to also delete the ingested index
```

## Quickstart (dev)

```sh
git clone https://github.com/adityam23/docket && cd docket
uv sync                  # base deps only
uv run dk serve          # dashboard + API on http://127.0.0.1:8760
uv run dk chat "hello"   # one-shot generation smoke test
uv run pytest            # offline suite, no backend needed
```

Extras: `uv sync --extra ingest` (PDF text extraction), `--extra agent`
(LangGraph). The dashboard rebuilds with `cd docket/web/frontend && npm ci &&
npm run build`; releases ship it prebuilt.

## Configuration

Every knob is a `DK_`-prefixed env var or a line in `.env`. Defaults are safe:
without an embedding endpoint retrieval degrades to sparse BM25; without a
reranker the fused RRF order stands. Nothing errors on missing backends.

| Variable | Purpose |
|---|---|
| `DK_BACKEND_URL` | `/v1` base of any OpenAI-compatible server. Default `http://127.0.0.1:11434/v1` |
| `DK_CHAT_MODEL` | Model id served there. Default `gemma4:e2b` |
| `DK_PROVIDER` | `local`, `cerebras`, or `groq` |
| `DK_CEREBRAS_API_KEY` / `DK_GROQ_API_KEY` | Free-tier keys when `DK_PROVIDER != local` |
| `DK_EMBED_URL` | Dedicated embeddings endpoint. Unset keeps sparse-only |
| `DK_RERANK_URL` | Endpoint serving `/v1/rerank` (Cohere shape). Unset disables reranking |

## How it works

```
ingest:  pdf -> ocr? -> chunk -> embed? -> index (BM25 + optional vectors)
answer:  query -> hybrid recall -> cross-encoder rerank? -> grounded LLM hops
         -> cited answer + per-token confidence label
```

One `/v1` client talks to every backend. Seams are explicit: `Corpus.dense_search`
swaps vector stores, `Retriever(reranker=...)` swaps rankers, `Provider` swaps
llama.cpp for hosted APIs without code changes. Degradation is graceful at each
step, so the offline test suite runs the full pipeline against fakes.

## Benchmarks

Measured on FinanceBench with the production grounded prompt
(`benchmarks/financebench/README.md` has reproduction commands):

| Setup | Result |
|---|---|
| Oracle evidence (retrieval ceiling) | ~89% accuracy |
| Full RAG, sparse-only, no reranker | ~19% accuracy, recall@6 ~0.52 |
| + qwen3-reranker-0.6b | recall@6 ~0.79-0.84 across embedders |

The reranker is the dominant lever. Embedding choice is secondary at k=6.

## License

GPL-3.0. See [LICENSE](LICENSE).
