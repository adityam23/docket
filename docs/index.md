# docket — design docs

The finalised design for the project. These are the source of truth; the code
implements them; `docs/roadmap.md` tracks the phased plan against them.

**What this is (one line):** a self-hostable **finance-filings research
assistant** over SEC EDGAR — live-ingested, cited, confidence-scored answers, with
a **novel glass-box interpretability** layer as the headline differentiator.
Chosen from real hiring-market demand and anchored to a concrete business case so
it isn't a *solution looking for a problem*.

- **[decisions.md](decisions.md)** — the settled design, in three grilling sessions.
  **Read Session 2 first** (the 2026-08-19 case-study pivot — the current
  identity); Session 3 (2026-08-21) settles the frontend design language; Session 1
  (Q1–Q18) explains why the machinery exists.
- **[design-language.md](design-language.md)** — the canonical frontend spec:
  authentic mobile Windows Phone **Metro** (Microsoft Design Language),
  Apple-polished. **Read before touching `docket/web/frontend/`.**
- **[case-study.md](case-study.md)** — the finance business case: user, pain,
  product mapping, hosting topology, demo script, golden set. **Read this to
  understand what the engine is *for*.**
- **[architecture.md](architecture.md)** — components, the OpenAI-`/v1` backend
  boundary, the infengine reconciliation, the two profiles, the hosting topology
  + retention, the data flow.
- **[stack.md](stack.md)** — the SOTA stack (verified Aug 2026) + phase-gated
  dependencies + the models available on the dev host.
- **[engine-profiles.md](engine-profiles.md)** — the infengine model-profiles
  contract from the consumer side: the opt-in `/engine` surface, the additive
  `/v1` conveniences, and what the UI must do when it ships. (Engine-side
  design agreed 2026-08-24, not implemented yet.)
- **[roadmap.md](roadmap.md)** — phased build plan, milestones, and backlog.

Cross-repo memory (this project also lives in the operator's persistent memory):
`docket-plan.md` and `portfolio-gap-projects.md`.
