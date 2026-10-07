# FinanceBench — eval + generator fine-tune pipeline

The measurement + training loop for the local generator. One conversion point,
one grounded prompt (`agent.graph._grounded_messages`), reused everywhere so
training matches inference.

## 1. Pick a base model (smoke gate)

`smoke_set.jsonl` — 34 self-contained, evidence-grounded items (extraction,
echo-proof derived arithmetic, distractors, table/unit traps, refusals). Serve a
candidate on any `/v1` endpoint and score it:

```bash
# serve a GGUF via any OpenAI-compatible server's prebuilt binary (its normal /v1 role)
engine-cli serve --config cfg.toml          # [[model]] id + local gguf path, kv_cache="fp8"
uv run python benchmarks/financebench/run.py \
    --base-url http://127.0.0.1:11434/v1 --model <id> --label "<name>" \
    --max-tokens 2048 --out results/<name>.json
uv run python benchmarks/financebench/run.py --compare results/*.json
```

Result (2026-08-22, Q4_K_M, RTX 3060): **Qwen3.5-4B Claude-Opus-distill 94% (32/34)**
≫ Gemma-4 E2B 56% (cites only 14/29) ≈ Qwen3.5-2B base 53%. Chosen base:
**Qwen3.5-4B Claude-Opus-Reasoning-distill** (SAE/lab-mode deferred — see decisions).

**Scoring fix disclosure (2026-10-07).** `docket/eval/financebench.py`'s magnitude-suffix
matcher (`_numbers`) had a word-boundary bug: a trailing word's first letter could be
misread as a `k`/`m`/`b`/`t` magnitude suffix with no boundary check (e.g. `"5 thousand"`
→ 5e12 instead of 5000, `"12 months"` → 1.2e7 instead of 12). Provider error strings
(`"[error: ...]"`) could also be numerically matched against a gold by accident (a stray
digit, a port number, or an HTTP status code landing inside the ×1000-tolerant matcher's
band). Both are fixed and regression-tested. This can shift `run.py`'s smoke-gate numbers
(and `run_full.py`'s numeric-match results) slightly versus earlier runs that hit either
bug — disclosed here, not hidden; no number already recorded under `results/` has been
retroactively edited, only future runs use the corrected scorer.

## 2. Real FinanceBench (the actual benchmark) — Claude-judged

`run_full.py` loads the real 150-Q open subset (`PatronusAI/financebench`,
auto-cached to `data/`) and scores in **oracle-evidence** mode (generator in
isolation, fed the gold evidence). Scoring is numeric-first (deterministic).

**Grading is Claude, not a small model.** The 6 GB target GPU serves ~one model
at a time, and a small-model judge is noisy on financial free-form answers, so
generation and grading are **decoupled**:

```bash
# 1. generate + export responses (numeric matcher auto-resolves numeric rows)
uv run python benchmarks/financebench/run_full.py \
    --base-url <candidate-/v1> --model <id> --max-tokens 1024 \
    --export results/responses/<name>.jsonl          # --limit N for a quick slice
# 2. Claude reads the needs_judge rows and writes results/responses/verdicts_<name>.jsonl
#    ({"id":..., "correct": true/false}) by the FinanceBench criterion.
# 3. combine into a scorecard
uv run python benchmarks/financebench/grade.py \
    --responses results/responses/<name>.jsonl \
    --verdicts  results/responses/verdicts_<name>.jsonl
```

**Results (150-Q, Claude-judged, corrected scorer 2026-08-24).** All four runs graded
with one consistent scorer after fixing a year-first auto-pass bug (a prose gold leading
with a fiscal year was auto-matched on the *year* and skipped the judge; the numeric path
now fires only on **bare-figure** golds — see `docket/eval/financebench.py`
`_PURE_NUM_RE`, regression-tested):

| generator | oracle (gold context) | full-RAG @6 |
|---|---|---|
| **qwen35-4b Opus-distill** (deployed default) | **80.7%** (121/150) | **68.0%** (102/150) |
| `gemma4:e2b` | **55.3%** (83/150) | **40.0%** (60/150) |

Generator quality gap **+25.4 pt** (oracle); full-RAG gap **+28.0 pt**; retrieval cost
(oracle→RAG) qwen −12.7 / gemma −15.3 pt. Anchor: the paper's shared-vault retrieval
setting is ~19%; oracle is the upper-bound reference. Pre-fix figures (86/61.3/73.3/46.7)
were inflated 5–7 pt and are superseded.

**Router disclosure (2026-10-07).** The 80.7%/68.0% figures above (and the 82.0%
full-RAG@10 figure in §5) were measured on the *direct* grounded-prompt path
(`agent.graph._grounded_messages` called straight from the harness) — not the production
`agent.graph.answer()` path, which also runs the aggregate router (`plan_operation`) for
cross-document questions. ORACLE-evidence mode stays on the direct path by design (it
deliberately isolates the generator from the router). Full-RAG no longer does: `run_full.py
--rag`/`--contexts` now route every question through the real `answer()` call, batched
two-pass (`PrefetchedRetriever`/`run_rag_via_answer` in `docket/eval/financebench.py`) so a
phase-switching engine (docket/phase.py) pays the retrieval↔chat profile switch once for the
whole run, not once per question. A re-run through the router is pending a live mettle
session — until then, treat 80.7%/68.0%/82.0% as the pre-router ceiling, not the deployed
number; no new number is claimed here.

### Filings (needed for full-RAG / the sweep)

```bash
uv sync --extra ingest
uv run python benchmarks/financebench/download_filings.py \
    --out-dir benchmarks/financebench/filings      # 84 filings, GitHub-raw source
```

## 3. Build the SFT set

```bash
uv run python benchmarks/financebench/build_trainset.py --source financebench \
    --val-frac 0.1 --out-dir trainset
```

Emits chat JSONL (`{"messages":[system,user,assistant]}`) in the exact served
grounded format. FinanceBench's 150 Qs are a *seed* — scale with synthetic
grounded Q/A over your own ingested filings (same converter).

## 4. QLoRA fine-tune (cloud)

`train_qlora.py` runs where GPU torch exists (Kaggle free / Vast 4090 ~$2-3/run):

```bash
pip install -U transformers peft trl bitsandbytes accelerate datasets
python train_qlora.py --base Qwen/Qwen3.5-4B \
    --train trainset/train.jsonl --val trainset/val.jsonl --epochs 3
```

Then merge → `convert_hf_to_gguf.py` → `llama-quantize Q4_K_M` → your engine's
`[[model]]` → re-run `run_full.py`. That last step closes the loop: the eval is
the training reward signal.

## 5. Retrieval (the dominant FinanceBench lever)

Measured on the shipped stack (post-scoring-fix, 2026-08-25): FinanceBench is
retrieval-bound — oracle ceiling 80.7% vs 68.0% at full-RAG @6 for the distill — so
the retriever was optimised *before* spending generator tokens.

> **Cutover note (2026-09-10).** Retrieval lives engine-side now (mettle
>`/v1`): the docket-local recall-sweep machinery (`sweep_recall.py` and the
>parity scripts) was **removed** at the mettle-only cutover; their measurement
>documents (`METTLE_PARITY_CONTRACT.md`, `BEST_CONFIG_CONTRACT.md`, verdict)
>and results JSONs remain as the historical record. Full-RAG reproduction is
>`run_full.py --rag` below — retrieval now goes through the engine's collection
>search + `/v1/rerank` via the production pipeline.


> **6 GB VRAM note (2026-08-24).** All three embedders and all three rerankers are
> sub-1 GB and fit on GPU together — **but only once the 128K-context chat model is
> out of the way**. `gemma4:e2b` + MTP at `n_ctx=131072` reserve ~4.3 GB of the 6 GB
> card (KV cache), leaving no room and OOM-crashing the backend when the sweep loads
> a reranker. The sweep therefore runs against a **benchmark profile** of
> `~/.engine/config.toml` (chat + OCR disabled, every embedder/reranker
> `n_gpu_layers=999`); restore live chat with `cp
> ~/.engine/config.toml.chat-profile.bak ~/.engine/config.toml` + restart. On this
> host the VRAM lever is **context length, not model weights** (2026-08-24).

**Phase-A result (2026-08-24, full 3×3 grid, GPU-resident, recall@6 t0.5 / @10):**

| embedder | none | +bge-reranker | +qwen3-reranker |
|---|---|---|---|
| embed-gemma (768d)      | 0.527 / 0.613 | 0.773 / 0.840 | **0.840 / 0.900** |
| bge-m3 (1024d)          | 0.513 / 0.600 | 0.747 / 0.807 | 0.793 / 0.847 |
| qwen3-embedding (1024d) | 0.540 / 0.667 | 0.793 / 0.860 | **0.840 / 0.913** |

**Winner: `embed-gemma` + `qwen3-reranker-0.6b` = recall@6 0.840** (@10 0.900). The
**reranker is the dominant lever** (+~30 pts over no-rerank for every embedder);
qwen3-reranker beats bge-reranker across all three embedders; embedder choice barely
matters once a good reranker is in place. `qwen3-embedding + qwen3-reranker` ties at
@6 (0.840) and edges ahead at @10 (0.913) if a larger k helps.

**Phase B — end-to-end on the winner.** Take the top embedder×reranker and run
full-RAG (context = live retrieval instead of gold evidence), Claude-judged as in
§2, reporting the gap vs the oracle ceiling:

```bash
uv run python benchmarks/financebench/run_full.py \
    --base-url <candidate-/v1> --model <id> --rag --gap \
    --docs-dir benchmarks/financebench/filings --collection financebench \
    --rerank-model qwen3-reranker-0.6b:latest \
    --retrieval-k 10 --export results/responses/rag_<name>.jsonl     # shipped k
```

`--collection` (default `financebench`) is a dedicated engine collection for the
benchmark run — never the production `docket` collection; the embedder is the
collection's engine-side default (`embed-gemma` per the Phase-A result above), not a
docket flag — `embed_model` in docket's own config only affects `dk embed`, unrelated
to engine-side collection search.

**Phase-B result (2026-08-25, distill, winner retriever):** k=10 ships.
`context_chunks=10` scores **82.0% (123/150)** full-RAG vs 68.0% at k=6 (+14 pt).
That is statistically indistinguishable from the 80.7% oracle ceiling on this run
pair (independent generations: 18 items flip in k10's favour, 16 in oracle's,
mostly free-form refusals and empty-response failures) — at k=10 retrieval is no
longer the distill's bottleneck. Verdicts:
`results/responses/verdicts_rag_qwen35_rag_k10.jsonl`.

The generator fine-tune (§3–4) only raises the ceiling retrieval can reach.

