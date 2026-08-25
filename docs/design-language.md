# Design language — Metro (Windows Phone), as if Apple built it

> **Status:** canonical spec for the frontend. Supersedes the first redesign pass
> (Session-3 grilling, 2026-08-21), which drifted into generic dark-SaaS and must
> be redone. Read this **before** touching `docket/web/frontend/`.
> The `/api/*` backend contract is unchanged by presentation work — but note the
> **conversation-context issue** in §12, which *is* a backend change.

## 0. One-paragraph intent

The look is the **mobile Windows Phone "Metro" experience** (Microsoft Design
Language / "Modern UI"), reproduced faithfully — **typography as the interface,
content over chrome, flat solid fields on a true-black canvas, one bold accent,
live tiles that flip, and kinetic, staggered motion** — and then finished to
**Apple's standard of craft**: 60 fps GPU-composited motion, optical precision,
restraint, and cohesion. Apple polish is a *finish on top of* authentic Metro,
**not** a replacement for it. The first pass inverted that (rounded floating
graphite cards, timid mauve accent, big gutters, generic semibold type, ad-hoc
animation) and read as AI-slop dark dashboard. That is the failure mode to avoid.

Reference: real Metro is the Windows Phone 7/8 Start screen, the People/Music
hubs (Panorama), the Mail app (Pivot + Continuum), and last.fm's Metro website —
huge lowercase headers bleeding off the right edge, ghosted secondary titles,
flat colour tiles, and content that swings/cascades in.

## 1. The five principles (and what each means *here*)

1. **Authentically digital.** No skeuomorphism, no faux depth. Flat solid fills;
   no gradients, bevels, drop-shadows, or glossy highlights. Richness comes from
   **type, flat colour, live data, and motion** — the things only a screen does.
2. **Content, not chrome.** The content *is* the UI. Strip borders, card frames,
   panel shadows, and ornamental buttons. A filing's title, a figure, an answer —
   these are the tappable objects. Push controls to a minimal action bar so the
   canvas stays pure content. (This is the single biggest correction vs. pass 1,
   which wrapped everything in bordered rounded cards.)
3. **Fast and fluid — motion is meaning.** Every transition and touch is
   acknowledged with choreographed, **staggered** motion that conveys hierarchy
   and continuity (see §8). Motion is a first-class material, not decoration.
4. **Pride in craftsmanship.** Pixel-precise alignment (everything on one left
   line), exact type weights, correct spacing rhythm. This is where "Apple"
   lives — Metro's bones, obsessively finished.
5. **Win as one.** One token system, one set of primitives, reused everywhere
   (mirrors CLAUDE.md's "one concept → one implementation"). No per-screen forks.

## 2. Typography — the heart of it

Type carries hierarchy and navigation; **boxes and borders do not.** Big
light-weight headers over small regular body; scale and whitespace create
structure. This is Swiss/International-Typographic + transit-signage DNA.

**Typeface.** Segoe WP is the original; on web use a close humanist-geometric
sans via the system stack (`-apple-system, "Segoe UI", Inter, Roboto, sans-serif`).
The *behaviour* matters more than the exact face.

**Weights.** Large display/titles → **Light / SemiLight**. Body → **Regular**.
Emphasis/accent → **SemiBold**. The signature is **thin type at large size**.
Pass 1 used semibold everywhere — wrong; it kills the Metro elegance.

**Type ramp** (from WP `PhoneFontSize` theme resources, in px; adapt to rem on
web — the *ratios* and role mapping are the point):

| Role | WP px | Weight | Notes |
|---|---|---|---|
| Huge display | 186.7 | Light | hero numerals (e.g. a big count/date on a tile) |
| Page/hero title | 72 | SemiLight | the big lowercase title that **bleeds off the right edge** |
| Extra-large | 42.7 | SemiLight | section titles |
| Large / group header | 32 | SemiLight | list-group headers (often "subtle"/dim) |
| Medium-large | 25.3 | Regular | |
| Medium | 22.7 | Regular | |
| Normal (body) | 20 | Regular | default body |
| Small | 18.7 | Regular | captions, subtle detail — use sparingly |

**Signature conventions (reproduce these exactly):**
- **Lowercase headers.** Section / pivot / page titles are lowercase by default
  ("ask", "filings", "coverage"). Quiet, modern, authentically-digital.
- **ALL-CAPS small labels** with letter-spacing for utility captions / action-bar
  labels / field captions — the counterpoint to the large lowercase display type.
- **Oversized titles that clip and bleed off the right edge — never wrap.** A
  title wider than the viewport is *clipped on the right*; the clipping signals
  "there is more horizontally" and invites the pan. This is a defining Metro cue.
- **Alignment is king.** Everything flush-left on one crisp vertical line
  (WP: 24 px in). Ragged-right. No jagged/mixed indents.
- **Generous negative space.** Emptiness is an active element; low density on
  hubs; hierarchy from scale + space, not lines/boxes.
- **Tabular / lining numerals** for all figures (`font-variant-numeric:
  tabular-nums`) — essential for the finance-data tables/citations.

## 3. Colour & theme

- **True-black canvas (#000 / near-black).** Dark only (WP chose true black for
  OLED battery + deep contrast; here it's the identity). Surfaces are black/very
  dark flat fields.
- **One accent colour, applied sparingly** — tiles, toggles, selection,
  links, active pivot. Everything else is monochrome (white/greys on black).
  This starkness is what makes it read as Metro. Pass 1 spread a timid mauve
  everywhere over graphite and lost it. The WP accent palette is bold and
  saturated (e.g. Cobalt `#0050EF`, Cyan `#1BA1E2`, Magenta `#D80073`, Lime
  `#A4C400`, Teal `#00ABA9`, Emerald, Violet, Amber, Mauve `#76608A`, …).
- **Constraint from the product grilling (still binding):** the accent must be
  **not blue, not super-green**, and must read as **decoration, never "this is
  correct."** The **trust triad — 🟢 green / 🟡 amber / 🔴 red — is reserved
  exclusively for the reliability signal** and must not collide with the accent.
  **DECIDED in the 2026-08-21 build session: the accent is WP Magenta
  `#D80073`** (bold, saturated, unmistakably decoration; clear of the triad at
  tile scale). Amber was rejected (collides with 🟡), Mauve was pass-1's timid
  hue, Teal sits too near "green". Whatever it is, it's flat and used at scale
  on a few tiles, not smeared over every surface.
- **Hybrid tile colouring (grilled decision):** most tiles are dark monochrome
  fields; a **few hero tiles are boldly accent-filled** (flat, solid — real WP
  tiles are fully accent-filled, so lean bolder than pass 1's faint tint).
- **Flat.** No gradients/shadows/borders as structure. Depth is implied by
  **motion and layering**, not by drop shadows. Hierarchy = type size + accent.

## 4. Layout, grid & tiles

- **Spacing unit ≈ 12 px** (WP "golden number"); block spacing ≈ 24 px; content
  on one left line. Use a small consistent scale; keep two density modes (airy
  hub vs dense work) on **one** token system.
- **Tiles are a real grid, edge-to-edge, with a small *uniform* gutter** — tiles
  read as one aligned field, not floating cards with big gaps. WP tile sizes:
  **small 159², medium 336², wide 691×336** (1×1 / 2×2 / 4×2 cells). Reproduce
  the size mix (small/medium/wide) so the board has Metro rhythm.
- **No card borders or shadows.** A tile is a flat solid fill (accent or dark),
  optional monochrome glyph, a count badge (top-right), a bottom-left label.
- **Corners:** WP tiles are **square** (0 radius). **DECIDED in the 2026-08-21
  build session: tiles/cards stay square; the ≤4px Apple concession is spent
  only on controls (inputs/buttons/chips).** Pill cards (pass 1's 16–20 px) are
  explicitly wrong.

## 5. Live tiles (the Start-screen board)

- Flat solid fill; centred **monochrome line glyph** (not emoji); **count badge**
  top-right; title bottom-left (medium/wide).
- **Flip animation:** tiles periodically flip front↔back on a 3D rotation to
  reveal a second face (title/count/back content), in a **staggered wave** across
  the board. Here, keep the discipline from pass 1: **a tile flips only when its
  underlying datum changes** (calm-alive), *and* an initial staggered entrance —
  but implement it with real, reliable motion (pass 1's flip felt broken; see §8).
- **Iconic template** (Metro-purest) is the model for our tiles: single centred
  monochrome glyph + count + up to three short content lines on a flat fill.

## 6. Navigation — Panorama & Pivot, adapted to the web app

Both are **long horizontal canvases that extend past the edge**; the hand pans
sideways. The **edge-peek** (a sliver of the next section/header showing at the
right) is the core affordance that invites the swipe.

- **Panorama / Hub** — an *explore/discover* entry point: one **big lowercase
  title bleeding off-screen**, optional full-bleed background with **parallax**
  (background pans slowest, title mid, sections at finger speed), sections that
  **peek** at the right edge, wrap-around, ≤5 sections. → **Use for Home** (the
  coverage board / hub) and potentially as the shell metaphor.
- **Pivot** — a *filter/switch-view* tabbed control: small typographic headers,
  **active header bright, next header peeking right**, headers slide with content,
  wrap-around, homogeneous pages only (never a wizard). → **Use for
  Observability (Pipeline · Traces)** and any in-page tabs; the top-level app nav
  is a Pivot-style typographic strip (no glowing sidebar).
- **Never nest** a Pivot inside a Panorama section; never put a horizontally-
  swiping control inside a Pivot page.
- **Chrome:** minimal. App-title = small ALL-CAPS; page-title = large light
  lowercase. No drawn back button (web: browser/router back). A minimal bottom or
  contextual **action bar** (few monochrome round icon buttons + a "…" overflow)
  is the Metro home for actions — keep the canvas content-pure.

## 7. Motion system (implement precisely; this is what pass 1 botched)

Principles: **motion = meaning + continuity; the *content* moves, not a
container; everything is staggered** (`index × ~100 ms` offsets); animate only
GPU-composited transforms (`transform`, `opacity`) for 60 fps; keep frequent
micro-feedback ≤100 ms and page transitions ≈350 ms; **always honour
`prefers-reduced-motion` and the in-app motion toggle.**

| Effect | Transform | Duration | From → To | Easing | Purpose |
|---|---|---|---|---|---|
| **Turnstile** (page enter) | `rotateY`, origin left edge | ~350 ms/el, staggered | 75°→0° in, 0°→−90° out | exponential (sine ok) | forward/back page transition; header then rows swing in one-by-one |
| **Continuum** | tapped element translates/continues into next page while rest turnstiles | ~350 ms | element flies across | exponential | carries the selected object across navigation for continuity |
| **Tilt** (press) | `rotateX/Y` toward touch + slight `scale` down | ~100 ms settle | tilt→rest | sine | tactile feedback on flat elements |
| **List cascade** | `translateX` | 800 ms/item, stagger `i×100 ms + 100` | ±80 px→0 | sine | rows wave in on load |
| **Tile flip** | `rotateX/Y` 3D | ~350–600 ms | front↔back, staggered | spring/exponential | live-tile face change (on data change) |
| **Panorama parallax** | `translateX` at layer-specific rates | follows finger | bg slowest → content 1:1 | inertial | depth on the hub |
| **Pivot header slide** | `translateX` + active-header brighten | ~300 ms | headers slide, next peeks | ease-out | tab switch |

The **stagger formula `index × ~100 ms`** is the reusable heart of every Metro
cascade. Most imitations (and pass 1) animate the whole page as one block — that
is the tell of fake Metro. Fix the flip trigger to be reliable (pass 1's
rAF/class-toggle restart felt broken).

## 8. Iconography

Monochrome **line glyphs** (white/greys), simple geometric, centred in tiles and
action buttons. **No emoji** as UI icons (pass 1 used 📄🗑🕑◆ — replace with a
consistent line-icon set). The trust triad may keep coloured dots, but as
semantic signal, not decorative emoji.

## 9. How it maps to docket's screens

- **Home** = the Metro **hub / Start board**: big lowercase title ("coverage")
  bleeding off-edge; a tile wall (hero **Ask** tile boldly accent-filled; live
  tiles Corpus / Trust / Recents / Backend / Capacity; honest-empty Pulse &
  FinanceBench). Airy. Staggered turnstile/cascade entrance; tiles flip on data
  change.
- **Ask** = a real chat surface (see §12 for the context bug). Big lowercase
  header; message turns as content (not heavy bubbles); Enter sends / Shift+Enter
  newline; streaming; filing-aware citations; inline trust glance; **Explain**
  layer (Sources → Reasoning → Lab) degrading per config. Dense.
- **Filings** = list/hub: fuzzy search, add (drop), remove, corpus telemetry,
  capacity. Typographic list with group headers; tabular figures.
- **Observability** = **Pivot** (Pipeline · Traces). Pipeline config-gated with an
  honest empty state; Traces = full timeline + reranker/relevance lens.
- **Settings** = text **list-menu** (Metro settings style), including Appearance
  (density, motion). Keep all config knobs here.
- **chat/[id]** = permalink to a real (or frozen-for-showcase) trace.

## 10. What the first pass got wrong (checklist to reverse)

1. Rounded (16–20 px) floating cards with borders + hover-lift + big gutters →
   **flat, edge-to-edge tiles, uniform small gutter, no borders/shadows, square
   (or ≤6 px) corners.**
2. Graphite-everywhere + faint mauve accent smeared over all surfaces →
   **true-black canvas + ONE bold accent used sparingly**; monochrome elsewhere.
3. Generic semibold type, no edge-bleed → **big Light/SemiLight lowercase titles
   that clip off the right edge; ALL-CAPS small labels; one left line.**
4. Top nav that's a plain link bar → **Pivot-style typographic nav with active-
   bright + edge-peek**; Home behaves like a Panorama hub.
5. Ad-hoc, janky flip (rAF/class toggle) animating blocks as one → **the real
   Metro motion system (§7), staggered, GPU transforms, reduced-motion honoured.**
6. Emoji icons → **monochrome line-glyph set.**

## 11. Keep from the first pass (IA/logic is largely sound; only the skin is wrong)

The route restructure and non-visual logic are worth keeping and re-skinning:
- Routes: Home / Ask / Filings (renamed from Documents) / Observability / Settings
  / chat/[id]; Sidebar and /documents removed.
- `ui` prefs store (density + motion, persisted); `filingCite()` helper;
  fuzzy-search on Filings; config-gating + **honest degradation** for pipeline /
  lab-mode / FinanceBench (this is *correct* and required by the anti-slop rule —
  never fake a capability; a disclosed empty/"not configured" state is right).
- The `/api/*` contract and `web/observability.py` read-models are untouched by
  the skin.

## 12. Open functional issue — conversation context & the chat/trace model (BACKEND)

**This is not cosmetic and must be fixed for the product to make sense.** Today
the Ask UI presents a multi-turn conversation, but:
- each question is sent to `/api/ask` / `/api/ask/stream` **standalone**, with **no
  prior turns as context** — so the model has **no conversational memory**; and
- each question is recorded as **its own independent trace**, so "a chat" isn't a
  real entity — traceability treats every question as a separate one-shot.

The UX therefore implies a conversation the backend doesn't actually have. Fix
requires a first-class **Chat/Session** concept:
- a **Chat** groups an ordered list of **turns**; each turn keeps its own trace,
  nested under the chat (Observability + `chat/[id]` reflect this nesting);
- `/api/ask` + `/api/ask/stream` accept **conversation history** (or a server-
  threaded `session_id`), with **context-window management/truncation** and clear
  rules for what prior context is sent;
- the trace store / model represent the chat→turns→trace hierarchy;
- Ask, Observability, and the permalink page are updated accordingly.

This touches `web/app.py`, the `agent/` graph + trace store, and the frontend —
it is a **real backlog item beyond the frontend re-skin** (see docs/roadmap.md).
Related gap: **filing metadata** (form type / company / ticker / fiscal period)
is not in the corpus model, so filing-aware citations degrade to `doc_id`; Metro
surfaces such metadata prominently, so consider adding it to ingest/corpus.

## 13. Sources (Metro research, 2026-08-21)

- Wikipedia — *Metro (design language)*.
- Microsoft Learn / MSDN archives — *Theme resources for Windows Phone* (font
  families, `PhoneFontSize` ramp, text styles, 12 px thickness resources);
  *Panorama control design guidelines*; *Pivot control for WP8*; *Iconic Tile
  template*; *App bar for Windows Phone*; *Start tile layout (WP8.1)*; *Theme
  design decisions / accent colours*; *UX guidance: Animations & Transitions*.
- Blink UX — *Guide to Metro Design* (principles; 25 px grid / 12 px gutter;
  Segoe WP weights).
- Smashing Magazine — *Designing for Windows Phone 7 and Metro*; *WP Design for
  Developers*.
- Jeff Wilcox — *"Metro" design guide for developers* (12 px golden number, 24 px
  left line, "alignment is king").
- Scott Logic — *Metro in Motion* #1 (fluid list: 80 px, 800 ms, sine, `i×100 ms`
  stagger), #3 (flying titles), #4 (tilt: `PlaneProjection`+scale, ~100 ms sine).
- Clarity Consulting / Ben Cull / Visually Located — Turnstile & Continuum
  storyboards (~350 ms, `RotationY` about left edge, ExponentialEase).
- Kodyaz — WP8 accent colour hex/RGB list.
