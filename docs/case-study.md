# Case study — the finance-filings research assistant

This is the concrete, business-real use case the whole engine serves. It exists
because the project was drifting as a *solution looking for a problem* (see
[decisions.md](decisions.md) § Session 2). Everything the codebase does is
justified here, against one workflow a real desk actually pays for.

> **One line:** *Read every SEC filing the moment it drops, ask it anything
> across your whole coverage universe, and get a cited answer with a confidence
> score in seconds — on your own hardware, with a glass-box view of why the model
> answered the way it did.*

## The real-world analog (this market exists and pays)
Agentic search over financial filings is a funded category: **AlphaSense**,
**Hebbia**, **BamSEC**, Bloomberg's filing AI. So the *need* is not invented —
this is a self-hostable, small-model, **provably-transparent** take on a thing
desks already buy. Our wedge is (1) it runs on your own box, (2) it shows *why*
it believes an answer at a depth the SaaS tools don't, and (3) it's cheap because
the model is small and local.

## Who it's for
- **Primary user:** a **buy-side research analyst / PM** at a small quant or
  fundamental fund, a boutique research shop, or a family office — someone
  covering dozens of names who can't read every filing before the information is
  priced in.
- **Secondary:** a **credit/risk analyst** cross-checking a borrower's filings
  against internal memos; a **corporate-strategy** analyst tracking competitors.

## The pain (concrete, and why it's worth paying to remove)
1. **Earnings season floods more filings than anyone can read** — 10-K / 10-Q /
   8-K arrive faster than a human covering 30 names can process, and the value of
   an insight (a guidance cut buried in an 8-K) **decays in minutes to hours** as
   the market prices it. This is the clock that makes *real-time* ingestion worth
   building — a compliance/legal reader has weeks; an analyst has minutes.
2. **Cross-filing questions take hours manually** — "gross-margin trend across
   these 5 competitors' last 3 10-Qs", "did any 8-K this morning mention a
   guidance change?" — exactly the multi-hop, compare-across-documents work the
   agent automates.
3. **You can't blindly trust an extracted number** — a hallucinated revenue
   figure is worse than no answer. Every figure must be cited to filing + page,
   and flagged when the model is unsure.
4. **Cost + IP at scale** — running thousands of filings/day through a frontier
   API is expensive, and a fund's *queries and thesis* are alpha it won't leak.
   A small local model answers both.

## How each capability is the *means to that end*
Every part of the engine maps to a line item of the pain above — nothing is here
"because it's cool":

| Capability (in the code) | The job it does for the analyst |
|---|---|
| **Live SEC EDGAR → Kafka/Spark/dbt spine** (`platform`) | Ingests a filing within minutes of it being filed — the real-time clock in pain #1. This is also the DE-employability showcase. |
| **Finance-native retrieval** (Item 1A / MD&A / financial-statement aware) + hybrid + rerank + **iterative agentic** re-query | Needle-in-haystack recall across huge filings; multi-hop for the cross-filing questions in pain #2. |
| **Finance-native tools** — extract revenue/EPS/segment, detect guidance *changes*, compare margins across peers, `cite_page` | Turns "read it yourself" into a structured, cited answer. |
| **Trust layer** (Tier-1 logprob + semantic entropy → 🟢/🟡/🔴) | Never trade on a guess — pain #3. Low-confidence routes to human review. |
| **Novel interpretability** (SAE feature-attribution "lab mode") | The *headline differentiator*: a glass-box "why did it answer this" that the SaaS tools don't show. **Must beat the Tier-1 baseline on the eval harness to count as a real feature** (else it's an honest showcase — see decisions § S2-Q4). |
| **Small local model (6 GB)** | Cheap at scale + keeps queries/thesis private (pain #4) + *makes the interpretability possible at all* (white-box needs the weights). |
| **Eval harness** (RAGAS + DeepEval gate + FinanceBench) | Proves the finance-tuned system is accurate, and referees whether the interpretability earns its headline. |

## The interface is part of the workflow (why Metro)
The analyst is scanning *many* filings fast against a decaying clock (pain #1), so
the UI has to be **glanceable and figure-dense, not chrome-heavy**. The frontend
language is the mobile Windows Phone **Metro** aesthetic, finished to Apple's
standard of craft — chosen precisely because it is **typography-first,
content-over-chrome, glanceable, and data/figure-friendly**: its transit-signage
heritage means it stays legible at a glance and uses **tabular/lining numerals**,
which is exactly what an analyst reading columns of financials across a coverage
board needs. Content (a filing's title, a figure, a cited answer) *is* the
tappable object; the chrome gets out of the way. Full spec:
[design-language.md](design-language.md).

The redesign also surfaced a real **workflow requirement**: an analyst doesn't ask
one question — they **ask follow-ups** ("...and how does that compare to last
year?", "...which segment drove it?"). So **Ask must be a genuine multi-turn
conversation with memory.** Today it isn't: each question is sent to the backend
**standalone** (no prior turns as context) and recorded as its **own isolated
trace**, so "a chat" isn't a real entity and the model has no conversational
memory — the UI implies a conversation the backend doesn't have. This is a **known
gap the build backlog addresses** via a first-class **Chat/Session** concept that
groups turns (each turn keeping its own trace, nested under the chat), with
context-window management on the `/api/ask` path (see `design-language.md` §12 and
TODO **T21**). The **anti-slop discipline still holds**: the public showcase Ask
box is **cached, read-only, and disclosed as such** — a real local-GPU run made
visible without a GPU, never a faked or simulated one.

## What "finance-specific" changes (vs a generic any-PDF tool)
The Session-2 decision (S2-Q8) is to **optimize hard for this one domain** rather
than be a mediocre generalist:
- **Extraction schema is financial** — revenue, EPS, segment data, guidance,
  risk-factor deltas — not a generic "extract table".
- **Retrieval is filing-structure-aware** — sections like Item 1A (Risk Factors),
  MD&A, and the financial statements are first-class.
- **The eval loop is FinanceBench** — the system is *iterated against it*, not
  just scored once.
- **`lite`/any-PDF still works**, but it's a side effect, not the identity.

## Hosting topology — three decoupled layers
The operator's demo and the shipped product are **different deployments of the
same code**. A downloaded app is **never** wired to the operator's infrastructure.

```
┌─ GitHub ─────────────────────────────────────────────────────────────┐
│ all code (lite + platform + IaC). This is what "counts" for the       │
│ portfolio and what users install. `curl | sh` → runs on THEIR box.    │
└───────────────────────────────────────────────────────────────────────┘
┌─ operator's domain  (GitHub Pages, static — free) ───────────────────┐
│ the case-study page: writeup, architecture, and the DEMO VIDEO of the │
│ full local-GPU run + SAE interpretability lab-mode (the GPU parts that │
│ can't run live).                                                       │
└───────────────────────────────────────────────────────────────────────┘
┌─ demo.<domain>  → operator's existing CPU-only VPS  (free tiers) ─────┐
│ ONE live `platform` deployment, running for real:                     │
│   • EDGAR → Kafka/Redpanda → Spark/dbt → index      ✅ live (no GPU)   │
│   • Dashboard: filings landing in real time, ingest  ✅ live (no GPU)  │
│     stages, freshness/throughput metrics                              │
│   • Ask box:  CACHED, read-only Q&As rendered from a  ⚠️ not live GPU  │
│     real local-GPU run — shows full citations + trust                 │
│     + SAE interpretability (the headline, made visible                │
│     without a GPU). No live inference, no public API key.             │
└───────────────────────────────────────────────────────────────────────┘
```

Rationale (full chain in decisions § S2-Q5–Q9):
- **Only GPU inference is offline** — a disclosed hardware limit — so the Ask box
  is cached. **Everything else is genuinely live.** A JS *simulation* pretending
  to be the app is explicitly rejected: if a recruiter finds a faked pipeline,
  the "real engineering, not slop" thesis dies. Cached-with-disclosure ≠ fake.
- **The live thing recruiters click is the DE pipeline + dashboard** — which runs
  fine CPU-only and is the strongest DE-employability signal. A live pipeline with
  moving real-world timestamps is the one artifact no front-end can fake.
- **Two "observability" surfaces, don't conflate them:** *pipeline* observability
  (stream health/throughput — live, no GPU) vs *model/answer* observability
  (trust + SAE interpretability — GPU, so cached/video + reproducible-by-clone).

## Retention on the VPS (free-tier friendly)
Keep only the *derived* layer; never hoard raw filings (decisions § S2-Q9):

| Medallion layer | On the VPS? | Why |
|---|---|---|
| Raw filing blobs (bronze) | **pointer only** | EDGAR is a permanent *public* archive — store accession-id/URL, re-fetch on demand. |
| Chunk text + embeddings (silver) + vector index | **keep** | Powers retrieval; the chunk text *is* the cited evidence, so citations work with no raw blob. |
| Extracted financials (gold marts) | **keep** | Small; the actual "data". |
| 10-year historical backfill | **capability, not resident** | Run the Spark batch once to prove it, capture it, prune. VPS holds a rolling recent window + the fixed FinanceBench set. |

EDGAR API is free (User-Agent + rate-limit respect); cached-example inference runs
on the local GPU or free Cerebras/Groq tiers. The live demo costs nothing beyond
the already-paid VPS.

## The demo script (what a recruiter experiences)
1. Land on the case-study page (domain) → read the one-liner + architecture,
   watch the video of the local-GPU run with SAE lab-mode.
2. Click through to `demo.<domain>` → watch **real filings streaming in now**,
   with timestamps that will be newer if they come back tomorrow.
3. Open a **cached** analyst question ("What changed in NVDA's guidance vs last
   quarter?") → see the cited answer, the 🟢/🟡/🔴 trust score, the retrieval
   trace, and the **SAE feature attribution** for why the model answered.
4. (For the technical reader) clone the repo, `curl | sh`, point it at their own
   filings or any PDF folder, and run the *full* thing — live inference + lab-mode
   — on their own GPU.

## The golden set (evaluation)
Built via deep research; **Aditya is the sole human judge**, biased to
**deterministic numeric answers** pulled from filings (revenue/EPS/segment) so
judgment is objective and the LLM never both proposes and grades. This *is* the
FinanceBench-style task and doubles as the headline benchmark — reported honestly
against a frontier baseline (a respectable gap is fine; an unreproducible "we beat
GPT" claim is not).

## Explicitly out of scope (v1)
- No live public GPU inference / no publicly-exposed API keys.
- No design-partner fund, no real client data, no SLAs to anyone.
- No non-finance domain as a *first-class* product surface (any-PDF is a side
  effect only).
- No code-gen / sandboxed tools (deterministic Python tools only).
