# Design decisions (locked)

Reached through deliberate grilling sessions. Don't silently reopen; flag if one
needs revisiting. **Four sessions so far:**

- **Session 1 (2026-08-18)** — chose the project and locked its shape (Q1–Q18).
- **Session 2 (2026-08-19)** — the **case-study pivot**: reframed the project
  around a concrete, business-real use case because Session 1's framing was a
  *solution looking for a problem*. Session 2 **supersedes** several Session-1
  lock-ins; where it does, the Session-1 entry is marked `⚠️ SUPERSEDED` and
  points here.
- **Session 3 (2026-08-21)** — the **Metro-authentic frontend redesign**: locked
  the frontend design language to a faithfully-reproduced Windows Phone "Metro"
  experience finished to Apple's craft standard, after a first redesign pass
  drifted into generic dark-SaaS slop (reopened as **T16**). Also surfaced a real
  backend gap: the Ask chat-vs-trace / conversation-context model is broken and
  needs a first-class Chat/Session entity.
- **Session 4 (2026-08-22)** — **generator base model + FinanceBench sequencing**:
  chose the local generator on measured data (Qwen3.5-4B Opus distill) and
  **deferred the SAE "lab mode"** to prioritise accuracy now. This adjusts the
  *sequencing* of the Session-2 headline (interpretability), not the identity —
  the SAE remains the differentiator on the roadmap; it is not being cut.

**If you read only one section, read Session 2 — it is the current identity of
the project.** Session 1 is kept in full because the reasoning still explains
*why* the machinery exists; Session 2 changed *what it is for*.

---

# Session 4 (2026-08-22) — generator base model + FinanceBench sequencing

## What was decided
1. **Generator base = `Qwen3.5-4B` Claude-Opus-Reasoning-distill** (a community
   fine-tune), chosen on **measured** data, not vibes. It won a 34-item
   discriminating smoke set at **94% (32/34)**, cites 28/29 answerable items, and
   nails 5/5 refusals — decisively ahead of the current Gemma-4 E2B (56%, cites
   only 14/29) and Qwen3.5-2B base (53%). The easy 8-item set had hidden the gap
   (all three 75–100%); a harder set with distractors, echo-proof *derived*
   arithmetic, CAGR, and table/unit traps opened a ~40-point spread.
2. **Use community fine-tunes, do not train from scratch** (operator: "the point is
   not to waste energy training from scratch, it is to create a top-of-the-line
   product — use the best possible chance"). Fine-tuning layers on top via QLoRA.
3. **Defer the SAE / white-box "lab mode"; ship accuracy first.** The performance
   winner (4B distill) has **no matching public SAE** — Qwen-Scope covers 2B/8B/9B,
   not 4B, and Gemma-4 SAE coverage is unconfirmed. Rather than sacrifice accuracy
   to keep a model with an SAE, we take the accurate model now and treat lab mode
   as the next headline milestone once accuracy is banked.
4. **Scope for the accuracy push = generator + retrieval** (operator's pick).
   Compute for the eventual QLoRA run is **decided later** — the pipeline is
   authored to be portable (Kaggle-free / Vast-4090), not host-locked.

## Why this does NOT reopen Session 2
Session 2 made **interpretability the headline** and the small local model a
*means*. Session 4 changes neither: the SAE is still the differentiator on the
roadmap (Phase 3), and the model is still small for the same reasons (6 GB
hardware + white-box access needs weights). What changed is **order of
operations** — accuracy is a precondition for the interpretability story to matter
(an SAE hallucination-detector is only interesting on a model whose answers are
worth trusting). If the SAE later proves it beats the Tier-1 baseline (the
Session-2 S2-Q4 gate), revisiting the base model to one with SAE coverage is a
legitimate re-open — flag it then.

## The honesty guardrail held (anti-slop)
No results were fabricated. Every number came from serving the actual GGUFs through
infengine's **prebuilt** binary (its normal `/v1` role — infengine code was never
modified or rebuilt) and scoring with `docket/eval/financebench.py`, which
reuses the **production** grounded prompt so the measurement reflects the real
system. A known limitation is recorded: the numeric auto-scorer can be gamed by a
model that dumps every evidence figure, which is why the hard items use *derived*
golds and the real 150-Q run keeps an optional LLM `--judge`.

## What was built (see `benchmarks/financebench/README.md`, TODO T25/T26)
The full measurement+training loop: smoke gate (`run.py` + `smoke_set.jsonl`), real
150-Q eval (`run_full.py`, `PatronusAI/financebench`, oracle-evidence, numeric-first
+ judge), SFT builder (`build_trainset.py` → `to_sft_example`, exact served prompt
format), and a portable cloud QLoRA trainer (`train_qlora.py`). **Next and dominant
lever: retrieval** — FinanceBench is retrieval-bound (perfect retrieval ≈ 89% vs
basic RAG ≈ 19%), so a finance-adapted embedder + reranker and a full-RAG eval mode
(T26) matter more than further generator tuning.

---

# Session 3 (2026-08-21) — Metro-authentic frontend redesign

## What was decided
The frontend design language is the **mobile Windows Phone "Metro" (Microsoft
Design Language / "Modern UI") experience, reproduced faithfully, then finished to
Apple's standard of craft.** Concretely: **typography *is* the interface**
(big Light/SemiLight lowercase titles that clip and bleed off the right edge,
ALL-CAPS small utility labels, everything flush to one left line); **content over
chrome** (no card frames, borders, or panel shadows — the content is the tappable
object); **flat solid fields on a true-black canvas**; **one bold accent used
sparingly** (monochrome everywhere else); **live tiles that flip** on data change
in a staggered wave; and **kinetic, *staggered* motion** — the real Metro motion
system (turnstile, continuum, tilt, list-cascade, tile-flip, panorama parallax),
built on GPU transforms with the `index × ~100 ms` stagger as its reusable heart.
Navigation is Panorama (Home hub) + Pivot (Observability, in-page tabs), both
long horizontal canvases with edge-peek. **Apple polish is a *finish on top of*
authentic Metro — 60 fps, optical precision, restraint — not a replacement for
it.** The full, canonical spec is **[design-language.md](design-language.md)**;
read it before touching `docket/web/frontend/`.

## Why we reopened (T16)
An *earlier pass in this same session* redesigned the frontend and drifted into
generic dark-SaaS **"AI slop"**: rounded (16–20 px) floating graphite cards with
borders, shadows and hover-lift; big gutters; a **timid mauve accent smeared over
every surface**; generic semibold type with no edge-bleed; emoji icons
(📄🗑🕑◆); and ad-hoc, janky animation (rAF/class-toggle flips that felt broken,
animating whole pages as one block). The operator rejected it flat — **"not at
all metro"** — and supplied Windows Phone reference images (Start screen, People/
Music hubs, Mail Pivot, last.fm's Metro site). **Process lesson to record:
research a named design language thoroughly (look it up — Wikipedia, the MSDN/
Microsoft Learn theme-resource archives, the Metro-in-Motion write-ups) *before*
implementing. Do not assume you know what "Metro" means from the name.** The
failure was building first from a vague mental model instead of the real spec.

## Binding constraints carried forward (from the earlier grilling — still true)
- **Dark only.** True-black canvas is the identity (OLED heritage + deep contrast).
- **Accent: not blue, not super-green**, and it must read as **decoration, NEVER
  as "this is correct."** Pick one bold WP-palette hue (Magenta, Violet, Amber,
  Mauve, distinct-Teal are candidates); flat and used at scale on a few tiles, not
  spread over every surface. **The final accent hue is an OPEN decision for the
  build session.**
- **The trust triad 🟢 / 🟡 / 🔴 is reserved *exclusively* for the reliability
  signal** and must not collide with the accent.
- **Hybrid tile colouring:** most tiles are dark monochrome fields; a **few hero
  tiles are boldly accent-filled** (flat, solid — lean bolder than pass 1's faint
  tint, as real WP tiles are fully accent-filled).
- **Corners default to square** (Metro, 0 radius); a **tiny ≤6 px radius is the
  maximum Apple concession** — still open, to confirm in the build session. Pill
  cards are explicitly wrong.

## A real backend decision surfaced (beyond the re-skin)
The Ask **conversation-context / chat-vs-trace model is broken and must be fixed
for the product to make sense** — this is not cosmetic. Today each question is
sent to `/api/ask` (+`/stream`) **standalone**, with **no prior turns as context**
(so the model has **no conversational memory**), and each question is recorded as
**its own independent trace** (so "a chat" is not a real entity — every question
is a one-shot). The UX therefore implies a conversation the backend doesn't have.
**Decision: introduce a first-class Chat/Session entity** — a Chat groups an
ordered list of **turns**, each turn keeping its own trace **nested under the
chat**; `/api/ask` + `/api/ask/stream` accept **conversation history / a
`session_id`** with **context-window management/truncation**; the trace store
represents the chat→turns→trace hierarchy; Ask, Observability, and the
`chat/[id]` permalink reflect the nesting. This touches `web/app.py`, the
`agent/` graph + trace store, and the frontend — it is a **backlog item beyond
the frontend re-skin** (§12 of design-language.md).

## Kept from pass 1 (only the visual skin + motion are redone)
The **route/IA restructure and honest-degradation logic are KEPT** — Home / Ask /
Filings (renamed from Documents) / Observability / Settings / `chat/[id]`;
sidebar and `/documents` removed; the `ui` prefs store (density + motion),
`filingCite()`, Filings fuzzy-search, and **config-gating + honest degradation**
for pipeline / lab-mode / FinanceBench (required by the anti-slop rule — never
fake a capability; a disclosed "not configured" empty state is correct). The
`/api/*` contract and `web/observability.py` read-models are untouched by the
skin. **Only the visual skin and motion system are redone.**

---

# Session 2 (2026-08-19) — the case-study pivot

## Why we reopened
The Session-1 project was described by its *mechanism* ("self-hostable agentic
document engine; a small model rivals frontier via tooling; trust + eval"), not
by a *job a business needs done*. Every noun was a feature. The operator's own
words: **"the project right now is a solution looking for a problem."** The goal
of this session: anchor the whole thing to **one real, painful, business-useful
workflow**, and let that workflow justify every capability — so the engine is a
*means to an end*, not an end in itself.

The method was a grilling session (see `.claude/skills/grill-me`): map the
decision tree, ask the frontier one round at a time, make the operator defend
each choice. What follows is the full chain, in the order it was settled, with
the reasoning that survived the grilling — **not** a tidy after-the-fact
rationalization. Read the reasoning, not just the verdicts; the *why* is the
point.

## The seven pivots (summary)
1. **Product identity → a finance-filings research tool**, optimized hard for
   that one use case. (Was: a domain-agnostic any-PDF assistant with finance as
   one of several showcase corpora.)
2. **Headline novelty → a novel transparency/interpretability method.** (Was:
   "a 6 GB model rivals frontier, proven on a benchmark.")
3. **The small local model → a *means*, not the thesis.** It exists because (a)
   white-box interpretability is impossible over a hosted API, and (b) the
   operator's hardware is a 6 GB RTX 3060. It is no longer sold as "small beats
   big."
4. **Self-host rationale → relocated.** Justified by *needing model internals for
   interpretability* + *cost at scale*, not by "the buyer's documents are
   confidential" (the original story was soft — SEC filings are public).
5. **Hosting topology → three explicit layers**, with a *cached, read-only* Ask
   surface and an emphatic **no fake/simulated app**.
6. **DE spine → real and live on the operator's existing (CPU-only) VPS**,
   holding only the *derived* layer + a rolling window (never a raw-filing
   archive), runnable entirely on free tiers.
7. **FinanceBench → headline as a *rigor/optimization loop*, reported honestly**
   against a frontier baseline — never as an unreproducible "we beat GPT" claim.

## The grilling chain (round by round)

### Round 1 — the three roots

**S2-Q1 — Altitude: what is this case study *for*?**
Options were (a) a portfolio *narrative* (a convincing demo + writeup),
(b) a genuinely *deployable* tool a real analyst could run, still judged as
portfolio, (c) a *design-partner* play (get a real fund to use it — scope
explosion, blows the milestones).
**Decision: (b) — a real, working, full-fledged demo.** The operator's framing:
the differentiator between *"vibe-coded slop a C-suite could have prompted"* and
*real engineering effort (AI-assisted)*. A faked or scripted demo is the exact
thing this project must not be. Do **not** chase a real fund as a user — you
don't need a paying customer to make the portfolio case, and pursuing one
detonates the timeline.

**S2-Q2 — The buyer, and the real reason to self-host.**
Pushed to name a buyer who would run a self-hosted tool instead of buying
AlphaSense / Hebbia / Bloomberg. This question **exposed a conflation**: the
operator had mixed two unrelated "hostings" — the *buyer's* deployment vs *his
own* portfolio hosting — and his CPU-only VPS only constrains the second. It
also surfaced that the "buyer's documents are confidential" argument is weak for
public SEC filings. **Outcome:** the self-host justification was relocated to
interpretability + cost (see S2-Q3 and pivot 4), and the two hostings were
separated (see S2-Q5/Q6).

**S2-Q3 — Does the case study *serve* the "6 GB rivals frontier" thesis, or
*replace* it?**
The pivotal answer. The operator: the small model size is **only** a consequence
of his host limits, *and* of the fact that **you cannot bring interpretability
research into API calls** — white-box methods need the weights/activations.
Commodity tracing (tool-call/thinking traces) is "done by everyone anyway"; he
wants **a novel transparency/observability method people have not really seen.**
**Decision:** interpretability is the **headline**; the small local model is the
*means* to it, not a benchmark thesis. This inverts Session 1's Q-thesis.

### Round 2 — novelty + hosting

**S2-Q4 — What *is* the novel transparency, and who is it for?**
The plan's Tier-3 (**Gemma Scope SAE feature-attribution + steering**, "lab
mode") is the genuinely-rare piece and matches the operator's SAE-Lens
background. But a finance end-user will never read an SAE feature — so the
audience must be named. **Decision (operator's exact framing):** *"if it works
better than baseline → an actual feature; if not → just a novelty showcase for
my profile."* Concretely: build an SAE/white-box **hallucination/uncertainty
detector** and **measure it against the Tier-1 logprob/semantic-entropy baseline
on the eval harness.** Beats baseline ⇒ real buyer value *and* a novel method.
Doesn't ⇒ keep it, but label it honestly as "lab mode" and don't oversell buyer
value. **The eval harness is what keeps the interpretability from being a flex —
it forces the novelty to earn its headline.**

**S2-Q5 — The Ask box on the public instance.**
Given no GPU on the VPS, options were cached-only, live-cloud-API, or both.
**Decision: cached-only, read-only.** Kill the live cloud key (cost/abuse not
worth it). But **keep the Ask surface** — it is where the *interpretability* (the
headline) is visible without a GPU. Cached Q&As are rendered from a real
local-GPU run and show the full SAE/trust/trace. Removing it would bury the one
thing that separates this from slop.

### Round 3 — "then why host a real app at all?"

**S2-Q6 — If the Ask answers are cached, why not just fake the whole app?**
The operator's sharp challenge: *"why even host a real app with fake features?
Couldn't it just be a JS-heavy simulation on GitHub Pages pretending to be the
app?"* **Answer — a hard no, and it's the most important fork in the session:**
- **"Cached answers" is not "a fake app."** Exactly *one* feature is cached (GPU
  inference), for a disclosed hardware reason. Everything else on the instance —
  Kafka stream, Spark transforms, dbt marts, freshness/throughput metrics, real
  SEC filings landing in real time — is **genuinely live**.
- **A simulation pretending to be the app is the "vibe-coded slop" from S2-Q1.**
  The failure mode: a recruiter pokes it, finds the "pipeline" is a hardcoded
  animation, and the entire "real engineering, not slop" thesis dies — worse than
  no demo. The moment any part is fake-pretending-to-be-real, you *are* the slop.
- **GitHub Pages is static** — it physically cannot run Kafka/Spark. "Just put it
  on Pages" isn't a cheaper same-thing; it's a categorically fake or recorded
  thing.

This clarified that **only the GPU model inference is offline; the DE system is
real.** Two meanings of "observability" were also separated here:
(1) **pipeline/DE observability** (stream health, throughput, ingest stages) —
no GPU, fully live-able; (2) **model/answer observability** (trust, SAE
interpretability, reasoning trace) — needs GPU + white-box, so it lives in the
recorded walkthrough + is reproducible by clone.

**S2-Q7 — How to present the (real) DE pipeline: live vs recorded.**
Both are *real software*; the choice is live-hosting vs a recorded video — never
a JS simulation. **Decision: live on the VPS.** The operator already pays for and
maintains the VPS; if it runs on free tiers he's happy to spend a few resources
on it. For a **DE-targeted** portfolio, a genuinely-live pipeline with moving
real-world timestamps is the single most convincing artifact and the one thing
no front-end can fake; the maintenance burden *is* itself a DE flex.

### Round 4 — identity, wiring, retention

**S2-Q8 — Is finance the product's *domain* or just the *demo skin*? And is a
download wired to the operator's Kafka?**
First the wiring, because it resolves the architecture: **a downloaded app is
never wired to the operator's Kafka.** That would break the self-hostable
premise. Downloads are self-contained (`lite` profile: point at your own PDFs,
embedded index, no infra). A user who wants streaming stands up *their own*
Kafka/Spark from the shipped IaC (`platform` profile). The operator's public VPS
is just *one* `platform` deployment he runs as a showcase — decoupled from every
install.
Then the identity fork: (a) domain-agnostic any-PDF product, finance as a demo
skin; (b) a finance product specifically. **Decision: (b) — a finance-filings
research tool.** Operator's reasoning: *"I could not get good performance on a
domain-agnostic thing, but I could optimize the hell out of it for one use
case."* This is the answer that finally kills "solution looking for a problem":
a focused, benchmarkable finance tool is useful to a business; a mediocre
generalist is a tech demo. `lite`/any-PDF survives as a side effect, not the
identity. **FinanceBench stays the headline regardless** (S2-Q9).

**S2-Q9 — Live retention on the VPS.**
The operator: *"I don't have to keep historical PDFs of all filings, right? I can
just get the data and keep that."* Correct, and it's what keeps this tiny-VPS +
free-tier friendly. Keep/drop, mapped to the medallion layers:
- **Raw filing blobs (bronze) — pointer only.** EDGAR is a permanent *public*
  archive; store the accession-number/URL and re-fetch on demand.
- **Chunk text + embeddings (silver) + vector index — keep.** The chunk text *is*
  the cited evidence, so citations work with no raw blob.
- **Extracted structured financials (gold marts) — keep.** Small; this is "the
  data" he meant.
- **10-year historical backfill — a demonstrated *capability*, not a resident.**
  Run the Spark batch once to prove it, capture it, prune. The live VPS holds
  only a **rolling recent window** + the **fixed FinanceBench filing set** as a
  stable demo/benchmark corpus.
EDGAR API is free (needs a User-Agent + rate-limit respect); inference for the
cached examples runs on the local GPU or the free Cerebras/Groq tiers. **The
whole live demo runs on free tiers + the already-paid VPS.**

**Honesty guardrail on FinanceBench (agreed with S2-Q9):** report it as a
*rigor* signal — the finance-tuned system's score **next to** the frontier
baseline and the tooling delta — **not** as a "we beat GPT" claim. Since S2-Q3
demoted the small-rivals-frontier thesis, FinanceBench doesn't need the small
model to *win*; it needs to show a domain-optimized system, honestly benchmarked.
A tuned small model + tooling landing *respectably close* is a great, credible
story; an unreproducible "we beat frontier" claim reads as the slop we're
avoiding.

## Session-2 settled tree (the current identity)
1. **Altitude:** a real, working demo — real engineering, AI-assisted; the
   anti-slop signal is the whole point.
2. **Product identity:** a **finance-filings research tool**, optimized hard for
   that one use case (any-PDF survives as a side effect).
3. **Buyer/self-host reason:** relocated — self-host justified by needing
   white-box access for the novel interpretability, plus cost at scale; *not* by
   filing confidentiality.
4. **Headline novelty:** a **novel transparency method** (SAE
   feature-attribution/steering "lab mode"). Beats the logprob-entropy baseline
   on the eval harness ⇒ a real feature; else ⇒ an honestly-labeled showcase.
5. **Small local model:** a *means* to #4 (interpretability needs internals) +
   the hardware limit — not the thesis.
6. **DE spine:** live SEC EDGAR → Kafka/Spark/dbt, **real and live on the
   CPU-only VPS**, derived-layer + rolling window only, free-tier.
7. **FinanceBench:** headline **as a rigor/optimization loop**, reported honestly
   against a frontier baseline.
8. **Hosting (three layers):** GitHub = code + install; the operator's domain
   (GitHub Pages, static) = case-study page + demo video; a `demo.` subdomain →
   VPS = the *live, real* DE pipeline + dashboard; **Ask = cached, read-only**
   showcase carrying the full interpretability (no live GPU, no public API key).
   A JS simulation is off the table.
9. **Two profiles:** `lite` (default download, self-contained, own PDFs/feed —
   never wired to the operator's infra) + `platform` (opt-in; ship
   docker-compose/helm for the user's *own* Kafka/Spark).

See **[case-study.md](case-study.md)** for the fully-defined business case,
personas, product mapping, hosting topology, and demo script.

---

# Session 1 (2026-08-18) — original lock-ins

Kept in full: the reasoning still explains *why* each capability exists. Where
Session 2 changed the intent, the entry is marked `⚠️ SUPERSEDED`.

## Thesis (⚠️ SUPERSEDED by Session-2 pivots 1–3)
> A small, self-hostable model + retrieval/eval **tooling** that rivals frontier
> models on long-document QA, extraction, and reporting. Chosen because `agent`
> (874) and `eval` (758) dominate ~1000 analysed job postings.

Still true that the market drove the project and that the tooling is what makes a
small model *good enough*. **But** the headline is now the interpretability, not
"rivals frontier"; the model's smallness is a means (hardware + white-box
access), not the selling point; and the product is finance-specific, not
domain-agnostic. The market signals (`agent`/`eval`) still hold — finance is the
domain that makes them concrete and benchmarkable.

## The lock-ins
- **Q1 Project:** the agentic document assistant (over grid-forecasting and a
  crypto-trading engine, which are parked — see memory `portfolio-gap-projects`).
- **Q2 Cadence:** GitHub code + website write-up by ~2026-08-24; simple Linux/Mac
  install + GitHub Pages by ~2026-08-31; the rest iterated over time.
- **Q4 Shape:** a document-intelligence agent **with a first-class eval +
  observability harness** (not a plain RAG chatbot; not an eval tool alone).
  *(Session 2: the "observability harness" is promoted to the headline via the
  novel interpretability layer — see S2-Q3/Q4.)*
- **Q5 Corpus (⚠️ SUPERSEDED by S2-Q8):** was "SEC EDGAR + EU/NL regulation as
  showcases; any-PDF-folder is the actual product." Now: **the product is
  finance-specific**; SEC EDGAR is the domain, any-PDF is a side effect, EU/NL
  regulation is dropped as a showcase.
- **Q6 Hosting:** must genuinely fit **6 GB VRAM** (Gemma-4-E4B). *(Session 2:
  still true, but the reason is the operator's hardware + the interpretability
  requirement, not a "small rivals frontier" claim. Optional online API keys for
  the operator's own testing/backfill only — never exposed publicly, see S2-Q5.)*
- **Q8 Agent:** SOTA needle-in-haystack recall via hybrid + rerank + **iterative
  agentic retrieval**; deterministic tools (`retrieve`, `compare_across_docs`,
  `extract_table`, `compute_metric`, `cite_page`, `generate_report`); outputs are
  cited **reports**. *(Session 2: tools become finance-native — extract
  revenue/EPS/segment, detect guidance changes across filings, compare margins
  across peers — and retrieval is tuned to filing structure: Item 1A / MD&A /
  financial statements.)*
- **Q9 Trust/interpretability (⚠️ PROMOTED by S2-Q3/Q4):** was "Tier-1 only for
  now; Tier-2/Tier-3 are roadmap." Now: **interpretability is the headline.**
  Tier-1 (logprob + semantic entropy → reliability score) is the always-on
  baseline *and the thing the SAE detector must beat*; Tier-2 (white-box probe)
  and Tier-3 (**Gemma Scope SAE lab mode**) are the novelty, no longer "someday."
- **Q10 Data engineering:** **DE-first / live** — the live streaming ingestion
  spine on real feeds → Kafka/Redpanda → OCR → chunk → embed → index, running
  continuously; lakehouse depth (Spark backfill, medallion, dbt, DuckDB/BigQuery,
  Airflow) layers on. The `lite` profile ships an embedded, zero-config, any-PDF
  path so a non-technical user needs no Kafka/Spark. *(Session 2: feeds narrow to
  **SEC EDGAR only** for the flagship; arXiv/EUR-Lex dropped. Retention is
  derived-layer + rolling window — see S2-Q9. The spine runs live on the VPS.)*
- **Q11 Eval (the novelty):** RAGAS (reference-free RAG metrics) for batch
  golden-set scoring + DeepEval (pytest) as the CI deploy gate + Promptfoo for
  the small-vs-frontier comparison; traced in **Langfuse**. Plus **XAI auto-eval**
  (XGBoost/LightGBM + SHAP over trace features → explainable hallucination flag)
  and **human-in-the-loop** review (stratified sample; human judges *before*
  seeing the machine's answer to avoid anchoring/circularity). *(Session 2: the
  harness gains a specific job — it is the referee that decides whether the SAE
  interpretability layer beats the Tier-1 baseline, S2-Q4.)*
- **Q12 Surface:** a **local web app** for a **non-technical end user** (primary);
  CLI/library secondary; API underneath. *(Session 2: the same web app is also
  the live public showcase on the VPS; the Ask surface is cached/read-only there,
  S2-Q5.)*
- **Q13 Profiles:** one codebase, two profiles — **`lite`** (embedded + DuckDB,
  zero-config, any-PDF) and **`platform`** (live Redpanda→Spark→lakehouse). Ship
  Docker/Helm/Ansible so users stand up their **own** streaming stack; never hook
  into the operator's infra. *(Session 2 confirmed this emphatically: a download
  is never wired to the operator's Kafka, S2-Q8.)*
- **Q14 Benchmark:** FinanceBench + RULER (multi-needle/multi-hop), NIAH-2
  baseline. Table = {Qwen3-0.6B raw · Gemma-4-E4B raw · E4B + full tooling ·
  **Muse Glimmer 30B** (open, API) · **Muse Spark 1.2** (Meta MSL, frontier,
  cheap API = the frontier baseline)}. *(Session 2: re-framed from "small can
  *beat* frontier" to a **rigor/optimization loop reported honestly** — S2-Q9.
  FinanceBench is the headline benchmark; RULER stays as the multi-hop stress
  test. Re-verify the models/benchmarks are current before wiring.)*
- **Q15 Report:** full structured, fully-cited report with a **reliability score**
  surfaced to the end user (🟢/🟡/🔴 + detail). *(Session 2: finance-native — a
  filing-analysis report; the reliability score is where the SAE detector, if it
  beats baseline, feeds in.)*
- **Q16 Tools:** deterministic typed **Python tools only** in v1 (no code-gen, no
  sandbox). Code-interpreter is a later, deliberate addition.
- **Q17 Golden set:** built via deep research; **Aditya is the sole human judge**.
  Bias to deterministic numeric answers so judgment is objective; the LLM never
  both proposes and grades. *(Session 2: numeric-answer bias is doubly apt for
  finance — reported figures are objective ground truth; aligns with
  FinanceBench.)*
- **Q18 Install:** **`curl | sh` prebuilt binaries first** (no Rust compile for
  users), **built-in uninstaller from day one**, Docker secondary. Auto-detect
  CUDA/Metal/CPU; fall back to a free-tier API on weak hardware.

## Method note
Always verify current SOTA online before locking a model/tool; never assume a
named tool is still current (it drifts monthly). The operator grills hard and is
right to — recompute the design-tree frontier honestly; don't declare it empty to
wrap up early.
