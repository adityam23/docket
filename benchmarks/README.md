# benchmarks/ — proving the thesis (Q14)

The head-to-head that proves a small local model + tooling rivals frontier models.

**Datasets:** FinanceBench (SEC-domain) + RULER (multi-needle/multi-hop), with
NIAH-2 as the single-needle baseline. Re-verify these are current before wiring.

**Baseline table (same harness for every row):**

| Row | Model | Mode |
|---|---|---|
| floor | Qwen3-0.6B | raw |
| local | Gemma-4-E4B | raw (6 GB) |
| **ours** | Gemma-4-E4B | **+ full tooling** |
| open | Muse Glimmer 30B | API |
| frontier | Muse Spark 1.2 | API (Meta MSL, cheap) |

**Headline framing:** above ~200–400K tokens on non-Gemini models, RAG over
focused chunks beats naive long-context — so *ours* can **beat** the frontier
model's naive long-context, not merely match it.

TODO (Phase 3): harness runner, per-row metrics, and the results table/plot.
