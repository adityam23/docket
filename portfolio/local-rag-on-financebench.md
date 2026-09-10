---
title: What Moved the Number, and What Didn't
description: Building a local RAG stack for long financial filings and chasing FinanceBench accuracy from a naive baseline to oracle parity, one measured lever at a time.
date: 2026-09-02
repo_url: https://github.com/adityam23/docket
permalink: /projects/docket/local-rag-on-financebench/
---

# What Moved the Number, and What Didn't

The problem I kept coming back to was a person with a job: a buy-side analyst covering a few dozen companies who cannot read every filing before the market prices it. Ten-Ks land by the hundred during earnings season, the value of a guidance cut buried in an 8-K decays in hours, and the questions worth asking are the ones that span documents — how does gross margin trend across five competitors' last three 10-Qs, did any filing this morning change its guidance. The category exists and pays; desks already buy AlphaSense and Hebbia for it. What I wanted to build was a self-hosted version that runs on a 6 GB GPU, keeps the analyst's queries and theses on their own hardware, and answers with citations you can check.

Everything in this essay is about the retrieval-and-generation stack that makes a small local model good enough for that job. I did not build a frontier model. I built a pipeline — hybrid retrieval, fusion, cross-encoder reranking, grounded generation — and then I did something unfashionable: I measured each piece of it against a real benchmark, FinanceBench, and let the numbers decide what shipped. The headline results are that a 4B quantized model reaches 82% answer accuracy on FinanceBench's 150-question open set with full retrieval (against ~19% for the benchmark's own shared-vault retrieval setting, and an 80.7% oracle ceiling where the model is handed the gold evidence), that the single biggest retrieval lever was worth ~31 points of recall, and that the last four points of accuracy came not from a better model but from letting ten chunks into the context instead of six. The more useful result is how many of my assumptions about where the gains were did not survive contact with measurement.

[docket](https://github.com/adityam23/docket) is the Python application: ingest, the retriever, the agent loop, a small web dashboard. The model-serving side is a separate Rust engine, [mettle](https://github.com/adityam23/mettle), which I built in parallel and which this project treats as a drop-in backend. The seam between them is deliberately boring — everything docket needs from a model provider is the OpenAI-compatible `/v1` HTTP contract: chat with per-token logprobs, embeddings, and a Cohere-shaped rerank endpoint. Any server that speaks that contract works, which is what makes the whole evaluation story possible: I could benchmark candidate models by pointing the same harness at different backends, and the offline test suite runs the entire pipeline against fakes with no model resident at all.

## A benchmark that punishes the two mistakes retrieval makes

FinanceBench is a set of 150 real analyst questions over 84 actual SEC filings, with gold answers and gold evidence spans. Two properties make it the right stress test for long-document RAG rather than another trivia benchmark.

First, the answers are numbers. "What was Netflix's free cash flow in 2022?" — the metric is a numeric match within tolerance, not a paraphrase-similarity score a model can talk its way around. Second, the correctness of an answer depends on the retrieval actually surfacing the one span that contains the figure, out of a corpus of 36,920 chunks across filings that run to hundreds of pages. FinanceBench therefore scores exactly the two things my pipeline could get wrong: whether the right evidence was retrieved (gold-evidence recall at k) and whether the model then answered correctly from it (answer accuracy). I treated them as two metrics and never mixed them — recall is a proxy for answer quality, not the thing itself, and I kept saying so in every results table.

The paper's own reference points put the task in perspective: with shared-vault retrieval over the whole corpus, the paper reports roughly 19% accuracy. Oracle evidence — the model given the gold span — is the upper-bound reference. Everything I did sits in the 40-point corridor between those two, and the entire exercise was about which end of the corridor my system lands on and why.

## A stack that degrades instead of failing

The retrieval design started as a sequence of bets, each one additive over the last. Sparse BM25 keyword search is always on — it is what makes the base install work with no embedding model at all, and its token-overlap matching catches exact phrases like "net cash provided by operating activities" that vector search can smear. When an embedding endpoint is configured, dense cosine search joins it and the two rankings are merged with reciprocal-rank fusion, which is the rare fusion method that does not need calibrated scores from either side. When a reranker is configured, the top fused candidates are reordered by a cross-encoder that scores each candidate pair against the query directly. Each layer is optional and the pipeline degrades gracefully: no embedding endpoint means sparse-only retrieval, no reranker means the fused order stands, and nothing anywhere errors on a missing backend.

That last property was a design decision before it was a convenience. Because every stage has a well-defined degraded mode, the whole answer path can run offline in tests against a fake provider, no GPU and no network — which meant the benchmark harness could be built and trusted before a single expensive run happened, and the seam where each new model slots in was exercised by the suite rather than by prayer. It also matched the product constraint: a user who just downloaded the thing and has no models configured should get working, if sparser, answers — not a stack trace.

The one-backend rule deserves emphasis because it is the reason the measurements in this essay were possible at all. One OpenAI-compatible client class talks to every provider — the local engine, Ollama, a hosted API. The router only swaps base URL and key. There is no per-provider fork of the client, and there is no second retrieval path used for benchmarking; the sweep that picked the shipped configuration runs the production `ingest_paths`, the production `Retriever`, and the production reranker seam verbatim. The configuration that wins the sweep is the configuration that ships, because it is the same code.

## The easy test set was lying to me

Before touching the real benchmark I built a smoke gate: 34 self-contained, evidence-grounded items designed to discriminate between candidate generator models — extraction, arithmetic that cannot be echoed from the prompt, distractors, table and unit traps, and refusal cases where the honest answer is "not derivable". I served candidate models over the engine's normal `/v1` role and scored them with the same grounded prompt the production system uses, so the measurement reflected the real system.

The first version of that set was useless. On an easy 8-item subset every candidate scored between 75% and 100%, which suggested the field was level. It was not. When I rebuilt the set to be discriminating — harder derived arithmetic, distractors, trap answer choices — a ~40-point spread opened up between the models. The winner was a community fine-tune, Qwen3.5-4B Claude-Opus-Reasoning-distill: 94% (32/34), citing 28 of 29 answerable items and refusing all 5 unanswerable ones. The incumbent Gemma-4 E2B scored 56% and cited only 14 of 29 answerable items; Qwen3.5-2B base scored 53%. The gap had been there all along; the easy set simply never asked the questions that exposed it.

The lesson is mundane and I keep re-learning it: a gate only discriminates if its items are hard for the reasons the real task is hard. FinanceBench's real questions are not short trivia; they are multi-step computations over numbers that appear nowhere else in the prompt. An easy smoke set measures whether a model can chat, which is not the property I needed to measure.

## The first real number: oracle 80.7%, and where the ceiling was

Grading introduced its own constraint. The 6 GB target GPU serves roughly one model at a time, and a small-model judge is too noisy on free-form financial answers to be a reliable referee. So generation and grading are decoupled: the harness generates answers over the real 150 questions and exports them with per-item metadata; a deterministic numeric matcher auto-resolves rows whose gold is a bare figure; and the remaining free-form rows are graded by a frontier model against the FinanceBench criterion, verdicts merged into a scorecard. This is worth stating plainly because it decided how every number below was produced: the thing being measured (a 4B local model) and the thing doing the measuring (a frontier judge) are different systems, and neither one grades itself.

Running the two generators in oracle mode — gold evidence fed in, retrieval removed — isolated generator quality from everything else. The distill scored 80.7% (121/150). The deployed default, Gemma-4 E2B, scored 55.3%. That +25.4-point gap confirmed on the real benchmark what the smoke set had suggested: the generator swap was the largest single jump available, and it was a decision about which weights to serve, not about prompt engineering.

Then the same generators ran in full-RAG mode at six chunks of context — live retrieval instead of gold evidence. The distill fell to 68.0% and Gemma to 40.0%. The retrieval gap, oracle to RAG, was −12.7 points for the distill and −15.3 for Gemma. That number reframed the project: for both generators, the gap between what the model could do with perfect evidence and what it did with my retrieval was bigger than I wanted it to be. Retrieval was the bottleneck, and it was the lever I had not yet pulled.

## The OOM thrash that was never about the models

The retrieval sweep measured gold-evidence recall at k across a full grid: three embedders (embed-gemma at 768 dimensions, bge-m3, qwen3-embedding at 1024) crossed with three rerankers (none, bge-reranker-v2-m3, qwen3-reranker-0.6b). The harness was built to run unattended and crash-recoverable on a shared 6 GB card: every embedded corpus cached to disk so it reloaded rather than re-embedded, results checkpointed after every configuration, and embed/rerank calls waiting for the backend and retrying instead of dying — because the box was shared and a co-tenant could restart the engine mid-run. It survived exactly that, sitting through a real backend kill and losing no work.

The first night did not go well. The sweep kept crash-looping the engine with CUDA out-of-memory errors, and I spent hours assuming the embedder or reranker weights were the problem. They were not, and `nvidia-smi` rather than intuition finally said so. The culprit was the always-on chat model: the 128K-context generator plus its speculative-draft model reserved about 4.3 GB of the 6 GB card in KV cache alone, leaving 0.6–1.5 GB free. Every embedder and reranker in the sweep was under a gigabyte — cheap. Loading any second model next to the chat reservation was what OOM'd. On this card the VRAM lever was context length, not model weights, and the fix was a benchmark profile of the engine that disabled the chat and OCR models the sweep never used and put every embedder and reranker on the GPU. Five models then sat GPU-resident in about 3.5 GB with headroom. The sweep completed 9/9 configurations with zero OOM.

## The reranker was the lever

The full grid, recall at k=6 and k=10 with a 0.5 overlap threshold:

| embedder | no reranker | + bge-reranker-v2-m3 | + qwen3-reranker-0.6b |
|---|---|---|---|
| embed-gemma (768d) | 0.527 / 0.613 | 0.773 / 0.840 | **0.840 / 0.900** |
| bge-m3 (1024d) | 0.513 / 0.600 | 0.747 / 0.807 | 0.793 / 0.847 |
| qwen3-embedding (1024d) | 0.540 / 0.667 | 0.793 / 0.860 | **0.840 / 0.913** |

Three findings, each of which contradicted something I believed going in.

The reranker is the dominant retrieval lever: every embedder sat at 0.51–0.54 recall@6 unreranked, and the qwen3 reranker lifted all three to 0.79–0.84 — roughly +30 points. BM25-plus-dense fusion gets the right neighborhood; the cross-encoder is what actually separates the right chunk from the similar-looking ones. I had expected the embedder to matter most — it is the piece that turns text into the space where similarity lives — and it mattered least once a good reranker was in place.

Second, qwen3-reranker beat bge-reranker across every embedder. This is the model I had earlier written off as "won't fit" on the card, based on a size estimate I never verified against the benchmark profile. It fit fine, and it was better. That is the same failure as the OOM night in miniature: reasoning from an unmeasured assumption about a resource constraint instead of measuring.

Third, the winner — embed-gemma plus qwen3-reranker at 0.840 recall@6 — tied with qwen3-embedding plus the same reranker, which edged ahead at k=10 (0.913). I shipped the embed-gemma combination anyway, for a boring reason: embed-gemma was the incumbent 768-dimensional embedder, so switching cost a full re-embed of the live index for no measured gain at the operating point. qwen3-embedding stayed on the bench, noted as the move to make if a larger k proved valuable. This was the first time in the project that the answer to "which model" was "the one you already have, because the difference is inside the noise."

## My evaluation harness was lying to me too

The corrected numbers above exist because of a bug that inflated every early result by 5–7 points, and the story of finding it is the most instructive dead end in the project.

The deterministic numeric auto-scorer extracted the first number it found in a gold answer and matched responses against it. That is fine when the gold is a bare figure — "$1,616.00", "24.26%". It is catastrophically wrong when a prose gold leads with a fiscal year: "…improving as of FY2022. 4.8% → 5.3%". The first number is 2022, so any response that mentioned 2022 auto-passed and, worse, skipped the human judge entirely. A model that cited the year and said nothing else scored as correct.

The fix was to gate auto-scoring behind a regex that matches a bare figure only, and route every prose gold and every zero gold to the judge. The change re-routed 26, 56, and 33 auto-passed rows per run — all hand-graded, verdicts merged with full coverage. Every number I had reported before the fix — the distill's 86.0% oracle, the 73.3% RAG, Gemma's 61.3% and 46.7% — was inflated and had to be re-derived. The corrected scorecard, produced by one consistent scorer across all four runs:

| run | buggy scorer | corrected |
|---|---|---|
| qwen35-4b distill — oracle | 86.0% | **80.7%** (121/150) |
| qwen35-4b distill — full RAG @6 | 73.3% | **68.0%** (102/150) |
| gemma4:e2b — oracle | 61.3% | **55.3%** (83/150) |
| gemma4:e2b — full RAG @6 | 46.7% | **40.0%** (60/150) |

The conclusions survived — the distill beat Gemma by roughly the same margin under either scorer, and retrieval was still the dominant lever. But the absolute numbers were wrong by more than most of the individual improvements I was chasing, which is the part that stings. I had built a careful measurement apparatus and then trusted one of its internal shortcuts. The regression tests I added for the pure-number path are now the reason I trust any scorecard the harness produces. An eval harness bug does not look like a bug; it looks like results.

A second bug of the same family: the retriever's `score` field kept the fused RRF score even after the reranker had reordered the hits, so any downstream consumer reading scores would have seen values that contradicted the actual order. Fixed so the score reflects the ranking that produced it.

## Ten chunks instead of six

With the winning retriever fixed in place, the end-to-end full-RAG runs at six chunks of context put the distill at 68.0% against its 80.7% oracle ceiling — a 12.7-point retrieval cost. The sweep's k=10 recall numbers hinted that the ceiling was not being reached at k=6, so I raised the shipped context from six chunks to ten and reran the full 150 questions. Answer accuracy went from 68.0% to 82.0% (123/150) — a +14-point jump from changing a single integer — and landed statistically indistinguishable from the 80.7% oracle ceiling on that run pair. Eighteen items flipped in the k=10 run's favor and sixteen in the oracle's, mostly free-form refusals and empty-response failures moving in both directions.

The interpretation is the interesting part. At six chunks, the system was still retrieval-bound: the model was failing because the figure it needed was in the seventh or eighth chunk. At ten, retrieval stopped being the bottleneck for the distill, and the residual errors were the generator's own — including five questions where the model burned its entire token budget inside a reasoning field and returned an empty answer, and free-form questions where it refused answers that were in fact derivable. Those are generator problems, and the generator's ceiling is what a fine-tune is supposed to raise. The retrieval stack, after the reranker and the context-length fix, is no longer the thing holding the system back — which is exactly the property you want before you spend training compute.

There is a second reading of the +14 points that I keep coming back to. The gap between 68.0% and 82.0% was recovered by adding four chunks of context, not by any improvement in ranking quality. The reranker had already done its work at the top of the list; the failure at k=6 was that the correct chunk was ranked seventh, not that the reranker was wrong. "Better retrieval" and "more context" are different levers, and in this regime the cheaper one — letting the model see more of an already-good ranking — was the one that closed the gap. The practical lesson for long documents is that your recall@k curve has a knee, and shipping the context size to the right of it can be worth more than the next ranking improvement.

## What the failures had in common

Across the project, the expensive mistakes shared a shape: I reasoned from an assumption about a system instead of measuring the system. The easy smoke set hid a 40-point generator gap because I assumed the items were hard. The OOM thrash ate a night because I assumed the model weights were the problem and never checked what the chat model's KV cache was doing to a 6 GB card. The reranker I wrote off as "won't fit" was the best one in the grid. And the scorer bug inflated every number for days because I trusted an auto-pass shortcut inside the harness I had built specifically to be trustworthy. In each case the fix was the same move: make the measurement direct, make it visible, and let it override the story I had already told myself.

The other consistent theme is that graceful degradation turned out to be a measurement strategy, not just a robustness nicety. Because every stage of the pipeline has a defined degraded mode, I could run the whole evaluation loop against fakes offline, swap backends without forking code, and benchmark candidate models on the production retrieval path rather than a toy version of it. The single `/v1` seam, the seam that keeps docket engine-agnostic, is also the seam that made the benchmark honest: the config that wins the sweep is the config that ships, because it is literally the same code path.

[docket](https://github.com/adityam23/docket) ships with `embed-gemma` for embeddings, `qwen3-reranker-0.6b` for reranking, and ten context chunks as the default retrieval configuration, and it still runs sparser and less accurate but fully functional if you take either model away. The engine that serves those models locally — [mettle](https://github.com/adityam23/mettle) — treats a 6 GB card as a scheduler rather than a cliff, which is the story of the other half of this project. The numbers here were produced by the two of them talking over a contract any OpenAI-compatible server can speak. The lesson I will keep is narrower: decide what you are optimizing, measure it directly, and then measure your measurement. The reranker was worth thirty points. Trusting my eval harness was worth more than that, in the other direction, until I stopped.
