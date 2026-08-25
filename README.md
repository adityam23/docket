# docket

Local RAG engine for long documents: hybrid retrieval, cross-encoder rerank,
grounded generation, evaluated on FinanceBench.

Self-contained. The release wheel ships the dashboard; no engine install, no
Node, no external services required to start. Works with any OpenAI-compatible
`/v1` server or a free-tier API key (Cerebras, Groq); the supported engine is
**mettle**.

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
ingest:  pdf -> ocr -> chunk -> embed -> index (BM25 + optional vectors)
answer:  query -> hybrid recall -> RRF fusion -> cross-encoder rerank
         -> grounded LLM hops -> cited answer + per-token confidence label
```

Stages marked without qualifiers are always on; `ocr`, `embed`, and `rerank`
activate only when their endpoint is configured, degrading gracefully otherwise.

One `/v1` client talks to every backend. Seams are explicit: `Corpus.dense_search`
swaps vector stores, `Retriever(reranker=...)` swaps rankers, `Provider` swaps
llama.cpp for hosted APIs without code changes. Degradation is graceful at each
step, so the offline test suite runs the full pipeline against fakes.

## Benchmarks

FinanceBench, 150 questions, production grounded prompt, shipped retrieval
winner (`embed-gemma` + `qwen3-reranker-0.6b`, k=10 context). Two metrics,
never mixed: **answer accuracy** (numeric match within tolerance for metrics
questions, Claude-judged equivalence for free-form ones) and **gold-evidence
recall@k** (token overlap of retrieved chunks vs the gold evidence span).
Reproduction commands:
[`benchmarks/financebench/README.md`](benchmarks/financebench/README.md).

Answer accuracy:

| Generator | Oracle evidence | Full RAG @6 | Full RAG @10 (shipped) |
|---|---|---|---|
| Qwen3.5-4B Opus-distill | 80.7% | 68.0% | **82.0%** |
| gemma4:e2b (default chat model) | 55.3% | 40.0% | — |

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
