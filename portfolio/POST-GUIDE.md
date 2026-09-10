# How to write the docket portfolio post

This is a **brief**, not the post. It tells you how to write the one long-form
essay about docket that goes on the portfolio site
(<https://adityam23.github.io>). Do not commit prose here — write the actual
post in a *separate* file (see "Where the post file goes" below).

> This guide file is deliberately safe to leave in `portfolio/`: the site
> aggregator only publishes `portfolio/*.md` files that **start with `---` YAML
> front matter**. This file starts with a `#` heading, so it is skipped and
> never appears on the site. Keep it that way — never put front matter at the
> top of this guide.

---

## 1. The voice you're matching

The site is styled as Unix **man pages**, and every post is a long-form
**essay / postmortem**, not a tutorial or a README. Before writing, read the two
reference posts to internalize the register:

- `_posts/2026-05-17-9-Classes-5-Weeks.md` — a competition postmortem.
- `_posts/2026-08-03-ai-safety-rounding-error.md` — a researched argument.

Rules distilled from them:

- **First person, narrative.** "I entered as…", "The shift that produced the
  largest jump was…". You are telling the story of building the thing.
- **Section headings are sentences about the work, not labels.** "Features
  carried more weight than models", "The slow grind from 0.66 to 0.7168" — not
  "Architecture" or "Results".
- **Concrete numbers, honestly.** Real benchmark deltas, what *didn't* work and
  got reverted, the dead ends. The 9-classes post spends whole paragraphs on
  experiments that failed. Do the same for docket.
- **Essay, not how-to.** The home page literally says "all entries below are
  essays, not tutorials." Don't write install steps; write about *why the design
  is the way it is* and what you learned.
- Don't open with a `---` horizontal rule or the layout will read it as front
  matter. Open with a Markdown `#` H1 that repeats the title.

## 2. The angle for docket (fill these in — don't just describe features)

docket is a local RAG engine for long documents: hybrid retrieval → RRF fusion →
cross-encoder rerank → grounded generation, benchmarked on FinanceBench. The
README is the *spec*; the post is the *story*. Pick a spine and commit to it. The
strongest candidate spine, judging by the reference posts, is:

**"What actually moved the number, and what didn't."** FinanceBench answer
accuracy and gold-evidence recall@k are your leaderboard. Walk the reader from a
naive baseline to the shipped config (`embed-gemma` + `qwen3-reranker-0.6b`,
k=10) and show which levers mattered.

Questions the post should answer (turn each into narrative, not bullets):

- Why long-document RAG is its own problem, and what FinanceBench specifically
  punishes (numeric answers within tolerance; recall of a gold evidence span).
- The retrieval stack as a sequence of bets: BM25 baseline → dense recall → RRF
  fusion → cross-encoder rerank. Which bet paid off most? (The README says the
  reranker is the dominant lever: +31 pts recall@6. Tell that story — how you
  found it, what you expected.)
- The "+context vs +better-retrieval" tension: raising k from 6→10 recovered
  +14 pts answer accuracy and closed the gap to the oracle ceiling. What does
  that tell you about where the bottleneck really was?
- **Graceful degradation as a design principle.** No embedding endpoint → sparse
  BM25; no reranker → fused order stands; nothing errors on a missing backend.
  Why you built it that way, and how it let the offline test suite run the full
  pipeline against fakes. This is a genuinely interesting engineering choice —
  give it a section.
- The mettle relationship: docket is engine-agnostic (any OpenAI-compatible
  `/v1`) but the supported local engine is mettle. If mettle also gets a post,
  cross-link them (plain Markdown link).
- At least one honest dead end — a retrieval or generation idea that *didn't*
  beat the fused order, with the number that killed it.

Keep any tables you include small and in service of the argument (see the
README's benchmark tables for the shape).

## 3. Where the post file goes — the two publishing paths

There are **two ways** to publish, and you must pick one. Both are documented in
full in the site repo's `PUBLISHING.md`; the summary:

### Path B — source lives in THIS repo (recommended for docket)

Use this so the post travels with the code. Write the post as a **new** file:

```
portfolio/<name>.md      e.g. portfolio/local-rag-on-financebench.md
```

The filename is slugified for the URL (`Local RAG.md` → `local-rag`). Give it
front matter (this is the real post, so front matter IS at the top):

```yaml
---
title: The Title Of The Post
description: One-line summary shown on the index and in the NAME section.
date: 2026-09-02
repo_url: https://github.com/adityam23/docket
permalink: /projects/docket/<slug>/
---
```

- `date:` is mandatory and must be `YYYY-MM-DD` (10 chars). Without it the
  aggregator stamps the build day and the URL changes on every build.
- Then register this repo once in the **site repo's** `projects.yml`:

  ```yaml
  projects:
    - repo: adityam23/music-gen
    - repo: adityam23/docket
  #    branch: master   # only if the default branch isn't main
  ```

- **Publishing is triggered by the site repo, not this one.** Pushing to docket
  alone does nothing. Push a change to `adityam23.github.io`, OR hit "Run
  workflow" on its Actions tab, OR re-enable the daily cron in its `build.yml`.
  The aggregator then clones docket, copies `portfolio/*.md` into `_posts/`, and
  the post goes live.

### Path A — source lives in the SITE repo instead

If you'd rather keep this as a standalone essay decoupled from the docket repo,
skip `portfolio/` entirely and create the file directly in the site repo at
`_posts/YYYY-MM-DD-<slug>.md` with the same front matter (drop `repo_url` if you
don't want the source link). The aggregator never touches hand-authored
`_posts/` files. Pushing the site repo publishes it. Use Path A only if you
*don't* want the post to live with docket's code — for docket, Path B is the
natural choice.

## 4. Images

If the post needs figures (a pipeline diagram, a recall curve):

- Put them in `portfolio/assets/` in **this** repo and reference them
  **relatively**: `![alt](assets/pipeline.png)` or `![alt](./assets/pipeline.png)`.
  The aggregator copies them into `assets/projects/docket/` on the site and
  rewrites the paths automatically.
- Absolute `https://` image URLs are left untouched, so both styles can coexist.
- Do **not** hand-place images in the site's `assets/projects/` — that folder is
  wiped (`rm -rf`) on every build.

## 5. Comments (giscus) — linking giscus.app to the right repo

Comments are [giscus](https://giscus.app): each thread is stored as a GitHub
Discussion. The site's `post` layout already contains the giscus `<script>`; it
only renders when the ID fields are present. You provide four values that the
layout injects into the script tag's `data-repo`, `data-repo-id`,
`data-category`, and `data-category-id` attributes.

**Which repo holds the discussions?** Recommended: centralize on the site repo
(`adityam23/adityam23.github.io`) so every post's comments live in one place and
are moderated together. giscus maps threads by page `pathname`, so each post
still gets its own thread even when they share one repo. (You *could* instead
point docket's posts at `adityam23/docket`'s own Discussions — same procedure,
just a different repo.)

### One-time setup on the repo that will hold discussions

1. **Enable Discussions**: repo **Settings → General → Features → check
   "Discussions"**. (giscus requires this — it will refuse otherwise.)
2. **Install the giscus GitHub App**: <https://github.com/apps/giscus> → Install
   → grant it access to that repo. (Required for giscus to read/write
   Discussions.)
3. The repo must be **public** (it is).
4. Make sure a Discussion **category** exists to post into. An
   "Announcements"-type category (maintainer-only new threads) is ideal so only
   giscus opens threads. `Announcements` is the layout default.

### Get the IDs from giscus.app

Go to <https://giscus.app> and fill the **Configuration** form:

| giscus.app field | What to enter |
|---|---|
| **Repository** | `adityam23/adityam23.github.io` (type `owner/repo`). giscus.app validates it live and shows a green check only if the repo is public, has the app installed, and has Discussions on — if you see an error, fix the step above it names. |
| **Page ↔ Discussions Mapping** | Choose **"Discussion title contains page `pathname`"**. The layout is hardcoded to `data-mapping="pathname"`, so this must match. |
| **Discussion Category** | `Announcements` (or the category you made). |
| Features / theme | Leave default — the layout already sets reactions, theme, input position. |

giscus.app then generates a full `<script>` snippet. **Do not copy the whole
snippet.** From it, take only the two computed IDs:

- `data-repo-id="R_…"`     → this is your **`giscus_repo_id`**
- `data-category-id="DIC_…"` → this is your **`giscus_category_id`**

(`data-repo` = the `owner/repo` slug you typed; `data-category` = the category
name — you already know both.)

### Put the four values where the layout reads them

**Recommended — set once for all posts** in the site repo's `_config.yml`
`defaults` block, so no post repeats them:

```yaml
defaults:
  - scope:
      path: ""
      type: "posts"
    values:
      layout: "post"
      giscus_repo: "adityam23/adityam23.github.io"
      giscus_repo_id: "R_xxxxxxxxxxxx"          # data-repo-id from giscus.app
      giscus_category: "Announcements"
      giscus_category_id: "DIC_xxxxxxxxxxxx"     # data-category-id from giscus.app
```

**Or per-post**, if you only want comments on some posts: put the same four keys
in this post's front matter instead.

The IDs are **not secrets** — they're visible in page source by design, safe to
commit. If the site owner has already done this one-time setup once, you don't
repeat it; comments will just work.

## 6. Final checklist

- [ ] Wrote the post as a *new* file (not this guide), front matter at the top.
- [ ] `date:` set to a real `YYYY-MM-DD`.
- [ ] `permalink:` under `/projects/docket/…`, `repo_url:` set (Path B).
- [ ] Registered `adityam23/docket` in the site repo's `projects.yml`.
- [ ] Numbers are real; at least one honest dead end included.
- [ ] Triggered a site-repo build (push there / Run workflow) — pushing docket
      alone does nothing.
- [ ] Confirmed it's live at `https://adityam23.github.io/projects/docket/<slug>/`.
