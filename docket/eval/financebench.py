"""FinanceBench-style GENERATOR smoke test (small, contained).

Isolates a candidate model's grounded-reasoning ability by handing it GOLD
evidence as context — the *same* grounded prompt the agent uses in production
(``agent.graph._grounded_messages``) — and scoring whether it:

  1. **finds** a figure stated in the evidence (extraction),
  2. **computes** a derived figure (numeric reasoning),
  3. **cites** its source with a ``[n]`` bracket, and
  4. **refuses** ("The documents do not contain this.") when the answer is absent.

This is NOT the full 150-question FinanceBench run (that loads the real dataset
and drives *live retrieval*). It is the "do the basics work?" gate an operator
uses to pick a base model *before* committing to a fine-tune — exactly the
contained comparison requested. Because every gold answer is derivable purely
from the provided evidence, ground truth is exact and no real filing figures are
fabricated (see ``benchmarks/financebench/smoke_set.jsonl``).

Dependency-light on purpose (pure Python + the OpenAI-/v1 ``Provider``) so it
runs against any reachable engine (mettle is the supported one).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from ..agent.graph import REFUSAL, _format_context, _grounded_messages
from ..agent.graph import answer as graph_answer
from ..agent.graph import plan_operation
from ..config import Settings
from ..providers.base import Provider

# Substring of the system-prompt refusal sentinel ("The documents do not contain
# this.") — matched loosely so paraphrases still count as a refusal.
_REFUSAL = "do not contain"

# The ``"[error: ...]"`` marker ``generate_financebench``/``generate_via_answer``
# emit when the provider call itself raised (transport failure, HTTP error,
# SlotUnavailable, ...). Must never reach the numeric/judge scorers below: its
# text is an exception message (a URL, a status code, a stray digit) that can
# accidentally numeric-match a gold under the ×1000-tolerant matcher -- which
# would silently credit a request that never even reached the model.
_ERROR_PREFIX = "[error"


def _is_error_response(text: str) -> bool:
    return text.strip().lower().startswith(_ERROR_PREFIX)


# Magnitude words / suffixes → multiplier, so "$3.2 billion", "3.2B" and
# "3,200 million" all normalise to the same value for comparison.
#
# NOT unified with ``agent.tools.parse_number``'s ``_MULT`` table even though
# both now use the identical word-boundary-anchored matching approach (see
# below): ``parse_number`` is the production extraction path (model-extracted
# values for ``aggregate()``/``compute_metric``) and deliberately recognises
# only k/m/b-family suffixes; this scorer additionally has to recognise
# trillion/kk/tn/bn forms that show up in FinanceBench's free-form gold/model
# text. Widening ``parse_number``'s table would silently change what the
# production aggregation tool accepts; narrowing this one would drop
# trillion-scale support from the benchmark. Two small tables, ONE matching
# algorithm (mirrored, not duplicated-with-drift) is the safer trade-off here.
_MAG = {
    "thousand": 1e3, "k": 1e3,
    "million": 1e6, "mm": 1e6, "m": 1e6,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "trillion": 1e12, "tn": 1e12, "t": 1e12,
}
# A bare number, with an optional "%" captured as its own group (percent is a
# unit, not a magnitude multiplier). The magnitude suffix is matched
# SEPARATELY below, against the text immediately following the number --
# never as a branch of this alternation. An alternation tries its branches in
# LISTED order and accepts the first that matches a *prefix*, not the longest
# candidate, so "thousand"/"total"/"months" were silently swallowed by the
# "t"/"m" single-letter branches tried earlier (the bug: "5 thousand" → 5e12,
# "1,577 total" → 1.577e15, "12 months" → 1.2e7).
_NUM_RE = re.compile(r"(-?\$?\s*\d[\d,]*(?:\.\d+)?)\s*(%)?", re.IGNORECASE)
# Anchored as a WHOLE WORD (mirrors ``tools.parse_number``'s ``\b`` anchoring):
# "m\b" cannot match inside "mm" or "months" because the character right
# after the "m" is itself a word character, so no boundary exists there. This
# makes the match correct regardless of alternation order (unlike the old
# ``_NUM_RE``), since every candidate must consume a full word, never a prefix
# of a longer unrelated one.
_SUFFIX_RE = re.compile(
    r"(?:" + "|".join(re.escape(s) for s in sorted(_MAG, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _numbers(text: str) -> list[float]:
    """Extract numeric magnitudes from free text, normalising $, commas, percent
    and magnitude suffixes (thousand/k, million/mm/m, billion/bn/b,
    trillion/tn/t) to plain floats. A suffix only applies when it is the very
    next WHOLE WORD after the number -- so "1,577 total" or "12 months" are
    left unscaled instead of misreading "total"/"months" as the "t"/"m"
    magnitude word they happen to start with, while "5 thousand" correctly
    scales to 5000 (see ``_SUFFIX_RE``)."""
    out: list[float] = []
    for m in _NUM_RE.finditer(text):
        raw, pct = m.group(1), m.group(2)
        digits = raw.replace("$", "").replace(",", "").strip()
        if digits in ("", "-", "."):
            continue
        try:
            val = float(digits)
        except ValueError:
            continue
        if pct:
            out.append(val)
            continue
        tail = text[m.end():].lstrip()
        sm = _SUFFIX_RE.match(tail)
        if sm:
            val *= _MAG[sm.group(0).lower()]
        out.append(val)
    return out


def _match_value(response: str, gold: float, tol: float = 0.02) -> bool:
    """True if any number in the response is within ``tol`` (relative) of gold.
    Falls back to a small absolute epsilon for values near zero."""
    eps = max(abs(gold) * tol, 1e-9)
    return any(abs(n - gold) <= eps for n in _numbers(response))


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _match_text(response: str, answer: str, aliases: list[str]) -> bool:
    """True if the gold answer (or any accepted alias) appears in the response."""
    hay = _norm(response)
    return any(_norm(a) in hay for a in [answer, *aliases] if a)


@dataclass
class ItemResult:
    id: str
    kind: str
    ok: bool
    answered: bool
    cited: bool
    hallucinated: bool
    latency_ms: int
    response: str
    reason: str


@dataclass
class Scorecard:
    model: str
    items: list[ItemResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.items)

    @property
    def correct(self) -> int:
        return sum(1 for i in self.items if i.ok)

    @property
    def cited(self) -> int:
        # citation only expected on items that should be answered
        return sum(1 for i in self.items if i.cited and i.kind != "refuse")

    @property
    def citable(self) -> int:
        return sum(1 for i in self.items if i.kind != "refuse")

    @property
    def hallucinated(self) -> int:
        return sum(1 for i in self.items if i.hallucinated)

    @property
    def refuse_ok(self) -> int:
        return sum(1 for i in self.items if i.kind == "refuse" and i.ok)

    @property
    def refuse_total(self) -> int:
        return sum(1 for i in self.items if i.kind == "refuse")

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def avg_latency_ms(self) -> int:
        return round(sum(i.latency_ms for i in self.items) / self.n) if self.n else 0

    def summary(self) -> dict:
        return {
            "model": self.model,
            "n": self.n,
            "accuracy": round(self.accuracy, 3),
            "correct": self.correct,
            "cited": f"{self.cited}/{self.citable}",
            "refusal_ok": f"{self.refuse_ok}/{self.refuse_total}",
            "hallucinated": self.hallucinated,
            "avg_latency_ms": self.avg_latency_ms,
        }


def _as_hits(evidence: list[str], doc_id: str) -> list[dict]:
    """Shape plain evidence snippets like retrieval hits so we can reuse the
    production context formatter (one concept, one implementation)."""
    return [{"doc_id": doc_id, "page": i, "text": t} for i, t in enumerate(evidence, 1)]


def score_item(item: dict, response: str, latency_ms: int) -> ItemResult:
    kind = item["kind"]
    gold = item["gold"]
    if _is_error_response(response):
        # A provider/transport failure, not an answer -- never numerically or
        # textually matched (score_item 0 for it, flagged, not hallucinated:
        # it never reached the model, so it isn't a confident wrong answer).
        return ItemResult(item["id"], kind, ok=False, answered=False, cited=False,
                          hallucinated=False, latency_ms=latency_ms, response=response,
                          reason="provider error (not scored)")
    refused = _REFUSAL in response.lower()
    cited = bool(re.search(r"\[\d+\]", response))
    answered = bool(response.strip()) and not refused

    if kind == "refuse":
        ok = refused
        reason = "refused as required" if ok else "answered when it should refuse"
        return ItemResult(item["id"], kind, ok, answered, cited,
                          hallucinated=not ok, latency_ms=latency_ms,
                          response=response, reason=reason)

    if kind == "value":
        ok = _match_value(response, float(gold["value"]), gold.get("tol", 0.02))
        reason = "value within tolerance" if ok else f"expected ~{gold['value']}"
    else:  # "text"
        ok = _match_text(response, gold["answer"], gold.get("aliases", []))
        reason = "gold phrase present" if ok else f"missing '{gold['answer']}'"

    # A confident-but-wrong answer on a real question is a (soft) hallucination.
    hallucinated = answered and not ok
    return ItemResult(item["id"], kind, ok, answered, cited, hallucinated,
                      latency_ms, response, reason)


def run_smoke(provider: Provider, items: list[dict], *, model_label: str,
              max_tokens: int = 384) -> Scorecard:
    """Run every item through the grounded prompt on ``provider`` and score it.

    Reuses :func:`generate_financebench`'s timed generation loop (its default,
    ``retrieve=None``, is exactly "format the item's own evidence as context" --
    what this smoke test always did) instead of re-implementing it (CLAUDE.md:
    one concept, one implementation)."""
    card = Scorecard(model=model_label)
    for item, text, latency_ms, _hits in generate_financebench(
        provider, items, max_tokens=max_tokens
    ):
        card.items.append(score_item(item, text, latency_ms))
    return card


# --- Real FinanceBench (150-Q open subset) -----------------------------------
# The smoke set above is a contained gate. This is the actual benchmark: the
# public open subset (PatronusAI/financebench). We run it in ORACLE-EVIDENCE mode
# by default (feed each question its gold evidence) to measure the *generator*
# in isolation; a full-RAG mode swaps the evidence for live retrieval hits.

FINANCEBENCH_URL = (
    "https://huggingface.co/datasets/PatronusAI/financebench/"
    "resolve/main/financebench_merged.jsonl"
)


def load_financebench(cache_path: str) -> list[dict]:
    """Return the FinanceBench open-subset records, downloading to ``cache_path``
    on first use. Network only when the cache is cold — never called in tests."""
    import json as _json
    import os
    import urllib.request

    if not os.path.exists(cache_path):
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        urllib.request.urlretrieve(FINANCEBENCH_URL, cache_path)  # noqa: S310
    with open(cache_path) as f:
        return [_json.loads(line) for line in f if line.strip()]


def fb_to_item(rec: dict) -> dict:
    """Map a FinanceBench record to a harness item (oracle-evidence). The gold
    ``answer`` is free-form, so scoring is numeric-first with a judge fallback."""
    evidence = [
        e.get("evidence_text", "")
        for e in (rec.get("evidence") or [])
        if isinstance(e, dict) and e.get("evidence_text")
    ]
    return {
        "id": rec.get("financebench_id", "?"),
        "kind": "financebench",
        "doc_id": rec.get("doc_name", "DOC"),
        "question": rec.get("question", ""),
        "evidence": evidence or ["(no evidence provided in dataset)"],
        "gold": {"answer": rec.get("answer", "")},
        "meta": {
            "question_type": rec.get("question_type"),
            "company": rec.get("company"),
        },
    }


def _match_value_unit_agnostic(response: str, gold: float, tol: float = 0.01) -> bool:
    """Like ``_match_value`` but tolerant of ×1000 unit shifts — FinanceBench gold
    answers state a figure in whatever unit the question named (e.g. 1577 meaning
    $1,577 million), while a model may write '$1.577 billion'. Match if any
    response number equals gold times some power of 1000."""
    import math

    if gold == 0:
        return any(abs(n) <= tol for n in _numbers(response))
    for n in _numbers(response):
        if n == 0 or (n < 0) != (gold < 0):
            continue
        ratio = abs(n) / abs(gold)
        scale = 1000.0 ** round(math.log10(ratio) / 3)
        if abs(ratio - scale) <= tol * scale:
            return True
    return False


# A gold answer we can auto-score is a *bare* figure — what metrics questions
# produce ('$1577.00', '0.83', '24.26%', '9.5x'). The whole string is one number
# (optional sign/currency/percent/magnitude/'x'|'times', trailing period).
_PURE_NUM_RE = re.compile(
    r"^\s*-?\$?\s*\d[\d,]*(?:\.\d+)?\s*"
    r"(?:%|x|times|billion|million|thousand|bn|b|mm|m|k)?\s*\.?\s*$",
    re.IGNORECASE,
)


def _gold_number(answer: str) -> float | None:
    """The gold's numeric value, but ONLY when the gold answer is a bare figure
    (what metrics questions produce, e.g. '$1577.00', '0.83', '24.26%'). Returns
    None — deferring to the judge — for any *prose* gold, even one that embeds a
    figure. Rationale: the deterministic matcher cannot tell the substantive figure
    from an incidental one, so a prose gold that leads with a fiscal year (e.g.
    'improving ... as of FY2022. 4.8%->5.3%') would otherwise auto-pass any response
    merely citing 2022 — a spurious correct that also skips the judge. It also
    returns None for a gold of exactly 0 ('none'/'0' answers): the near-zero match
    is too loose to trust, so a judge decides."""
    if not _PURE_NUM_RE.match(answer.strip()):
        return None
    nums = _numbers(answer)
    if not nums or nums[0] == 0:
        return None
    return nums[0]


_JUDGE_SYSTEM = (
    "You grade financial QA. Given a QUESTION, the GOLD answer, and a MODEL "
    "answer, reply with exactly 'CORRECT' if the model answer conveys the same "
    "key fact/figure as gold (ignore rounding within ~1%, wording, and units that "
    "mean the same), otherwise reply 'INCORRECT'."
)


def judge_correct(judge: Provider, question: str, gold: str, response: str) -> bool:
    """LLM-as-judge for free-form answers. Kept separate + optional so the
    numeric path stays deterministic and testable without a model."""
    msg = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": f"QUESTION: {question}\nGOLD: {gold}\nMODEL: {response}"},
    ]
    verdict = (judge.chat(msg, max_tokens=8, temperature=0.0).text or "").strip().upper()
    return verdict.startswith("CORRECT")


def score_financebench(item: dict, response: str, *, judge: Provider | None = None) -> ItemResult:
    """Score one real-FinanceBench item: exact-ish numeric match when the gold is
    numeric, else defer to the judge (or a lenient text match if no judge)."""
    gold = item["gold"]["answer"]
    if _is_error_response(response):
        # Never numeric/judge-matched -- a provider error's text (a URL, an
        # HTTP status code) can land inside the ×1000-tolerant matcher's band
        # purely by accident, which would silently credit a request that
        # never reached the model.
        return ItemResult(item["id"], item["kind"], ok=False, answered=False, cited=False,
                          hallucinated=False, latency_ms=0, response=response,
                          reason="provider error (not scored)")
    cited = bool(re.search(r"\[\d+\]", response))
    answered = bool(response.strip()) and _REFUSAL not in response.lower()
    gnum = _gold_number(gold)
    if gnum is not None and _match_value_unit_agnostic(response, gnum, tol=0.01):
        ok, reason = True, "numeric match"
    elif judge is not None:
        ok = judge_correct(judge, item["question"], gold, response)
        reason = "judge: correct" if ok else "judge: incorrect"
    else:
        ok = _match_text(response, gold, [])
        reason = "text match" if ok else "no numeric/text match (no judge)"
    return ItemResult(item["id"], item["kind"], ok, answered, cited,
                      hallucinated=answered and not ok, latency_ms=0,
                      response=response, reason=reason)


def make_rag_retrieve(retriever_for, *, k: int = 6):
    """Build a ``retrieve(item) -> hits`` for full-RAG mode: live retrieval scoped
    to the item's own filing.

    ``retriever_for(doc_id)`` returns a :class:`Retriever` over that filing's
    corpus (the caller builds + caches these from the source PDFs). The returned
    hits are the SAME shape ``_format_context`` consumes, so the generator and
    scoring paths are byte-for-byte identical to oracle mode — only the *source*
    of the context changes (gold evidence → retrieved chunks). A filing with no
    corpus yields no hits (honest empty context, never gold fallback)."""
    def retrieve(item: dict) -> list[dict]:
        r = retriever_for(item.get("doc_id", "DOC"))
        if r is None:
            return []
        return r.retrieve(item["question"], k=k)

    return retrieve


def generate_financebench(provider: Provider, items: list[dict], *,
                          max_tokens: int = 512, retrieve=None):
    """Yield ``(item, response_text, latency_ms, hits)`` for each item.

    The single generation loop shared by scoring (:func:`run_financebench`) and
    response export (so Claude can judge free-form answers out-of-band) — one
    concept, one implementation (CLAUDE.md). ``retrieve`` (optional) sources the
    context from live retrieval (full-RAG) instead of the gold evidence (oracle);
    everything else is identical, so the two modes differ only in context source."""
    for item in items:
        hits = retrieve(item) if retrieve is not None else _as_hits(
            item["evidence"], item.get("doc_id", "DOC")
        )
        context = _format_context(hits)
        messages = _grounded_messages(item["question"], context)
        t0 = time.perf_counter()
        try:
            text = provider.chat(messages, max_tokens=max_tokens, temperature=0.0).text or ""
        except Exception as e:
            text = f"[error: {e}]"
        latency_ms = round((time.perf_counter() - t0) * 1000)
        yield item, text, latency_ms, hits


def run_financebench(provider: Provider, items: list[dict], *,
                     judge: Provider | None = None, max_tokens: int = 512,
                     retrieve=None) -> Scorecard:
    """Run the real FinanceBench items through the grounded prompt and score them.

    ``retrieve`` (optional) turns this into **full-RAG** mode: for each item it
    returns live retrieval hits used as context instead of the gold evidence
    (oracle mode). Everything downstream — prompt, generation, scoring — is
    unchanged, so the accuracy delta between the two modes isolates the retrieval
    gap vs the oracle ceiling (T26)."""
    card = Scorecard(model=getattr(provider, "name", "candidate"))
    for item, text, latency_ms, _hits in generate_financebench(
        provider, items, max_tokens=max_tokens, retrieve=retrieve
    ):
        r = score_financebench(item, text, judge=judge)
        r.latency_ms = latency_ms
        card.items.append(r)
    return card


# --- full-RAG through the REAL production path (B7) --------------------------
#
# ``run_financebench``/``generate_financebench`` above call ``_grounded_messages``
# directly -- a *shortcut* that skips ``agent.graph.answer()``'s aggregate router
# (``plan_operation``) and its multi-hop re-query loop. That shortcut is fine for
# ORACLE mode (it deliberately isolates the generator), but full-RAG must measure
# what a user actually gets, so it has to go through ``answer()`` itself.
#
# Routing through ``answer()`` means retrieval and generation now share one
# engine that can't hold both the embed/rerank and chat profiles resident at
# once (docket/phase.py): naively calling ``answer()`` once per question would
# pay a ~2s profile switch TWICE per question (retrieve -> generate -> next
# question's retrieve -> ...). ``PrefetchedRetriever`` + ``run_rag_via_answer``
# below batch this into two passes over the whole question set instead: every
# question's retrieval first (one held RETRIEVAL window), then every question's
# generation (one held CHAT window) -- so the switch is paid twice TOTAL, not
# twice per question. A genuine re-query hop (needs the model's refined query
# first) is the one case that still falls through to a live retrieval call
# mid-generation; that is an inherent extra switch, not a batching failure.


class PrefetchedRetriever:
    """Wrap a per-filing :class:`~docket.retrieval.retriever.Retriever` (or the
    ``--contexts`` replay stand-in) so a BATCH of FinanceBench questions can
    retrieve everything up front, then generate everything, without paying a
    phase switch per question.

    ``prefetch(query, k)`` runs the real retrieval call for that exact
    ``(query, k)`` pair and caches it. ``retrieve`` then serves that exact pair
    from the cache with NO further call to the inner retriever at all -- so no
    engine call, so no phase entry, which is the entire point. Anything NOT
    prefetched (chiefly a re-query hop: ``agent.graph.answer()`` only knows the
    refined query after seeing a low-confidence first answer, so it can never
    be prefetched) falls through to the live inner retriever, which legitimately
    enters RETRIEVAL again.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self._cache: dict[tuple[str, int], list[dict]] = {}

    def prefetch(self, query: str, k: int) -> list[dict]:
        hits = self._inner.retrieve(query, k=k)
        self._cache[(query, k)] = hits
        return hits

    def retrieve(self, query: str, *, k: int = 10, candidates: int | None = None) -> list[dict]:
        cached = self._cache.get((query, k))
        if cached is not None:
            return cached
        return self._inner.retrieve(query, k=k, candidates=candidates)

    def iterative_retrieve(self, queries, *, k: int = 10) -> list[dict]:
        # A re-query hop is never prefetched (it needs the model's refined
        # query first) -- always a legitimate live fall-through.
        return self._inner.iterative_retrieve(queries, k=k)


def _answer_k(question: str, settings: Settings) -> int:
    """The exact ``k`` ``agent.graph.answer()`` will request for ``question``'s
    FIRST retrieval call -- mirrors ``plan_operation``'s routing and
    ``_aggregate_answer``'s ``k`` formula exactly, so a pass-1 prefetch using
    this ``k`` is guaranteed to be the cache hit pass 2 needs."""
    if plan_operation(question) is not None:
        return max(settings.context_chunks, 8) * 2
    return settings.context_chunks


def generate_via_answer(
    items: list[dict],
    *,
    retriever_of,
    provider: Provider,
    settings: Settings,
    prefetch: bool = True,
):
    """Yield ``(item, response_text, latency_ms, hits)`` for each item, generated
    through the REAL production path -- ``agent.graph.answer()`` (the aggregate
    router + grounded QA loop) -- instead of :func:`generate_financebench`'s
    direct-to-``_grounded_messages`` shortcut.

    ``retriever_of(item) -> retriever | None`` supplies a per-question retriever
    (``None`` means no corpus for that filing -> an honest refusal, never a gold
    fallback). When it is a :class:`PrefetchedRetriever` (the live full-RAG
    case) and ``prefetch`` is true (default), pass 1 retrieves for every item
    BEFORE pass 2 generates for any of them -- see the module note above for why.
    The ``--contexts`` replay stand-in has nothing live to prefetch; its
    ``prefetch`` is a harmless no-op either way.
    """
    if prefetch:
        for item in items:
            r = retriever_of(item)
            if r is not None:
                r.prefetch(item["question"], _answer_k(item["question"], settings))

    for item in items:
        r = retriever_of(item)
        t0 = time.perf_counter()
        if r is None:
            text, hits = REFUSAL, []
        else:
            try:
                ans = graph_answer(item["question"], retriever=r, provider=provider, settings=settings)
                text, hits = ans.answer, ans.hits
            except Exception as e:  # noqa: BLE001 - a candidate that errors on an item scores 0 for it
                text, hits = f"[error: {e}]", []
        latency_ms = round((time.perf_counter() - t0) * 1000)
        yield item, text, latency_ms, hits


def run_rag_via_answer(
    items: list[dict],
    *,
    retriever_of,
    provider: Provider,
    settings: Settings,
    judge: Provider | None = None,
    prefetch: bool = True,
) -> Scorecard:
    """Full-RAG FinanceBench, scored through the REAL production
    ``agent.graph.answer()`` path (see the module note above) instead of
    :func:`run_financebench`'s direct-prompt shortcut. Batches retrieval ahead
    of generation via :class:`PrefetchedRetriever` so a phase-switching engine
    pays the switch once for the whole set, not once per question."""
    card = Scorecard(model=getattr(provider, "name", "candidate"))
    for item, text, latency_ms, _hits in generate_via_answer(
        items, retriever_of=retriever_of, provider=provider, settings=settings, prefetch=prefetch
    ):
        r = score_financebench(item, text, judge=judge)
        r.latency_ms = latency_ms
        card.items.append(r)
    return card


# --- SFT training-data builder (generator fine-tune) --------------------------
# Convert items into supervised chat examples in the EXACT grounded format used
# at inference (`_grounded_messages`), so the LoRA learns the deployed contract:
# answer only from numbered context, cite with [n], refuse when absent.

def _target_answer(item: dict) -> str:
    """The supervised assistant target for an item — a concise, cited answer (or
    the refusal sentinel), matching the production system prompt's rules."""
    kind, gold = item["kind"], item["gold"]
    if kind == "refuse":
        return REFUSAL  # the ONE refusal sentinel (CLAUDE.md), not a repeated literal
    if kind == "financebench":
        ans = str(gold.get("answer", "")).strip()
        return f"{ans} [1]" if "[" not in ans else ans
    if kind == "value":
        return f"{gold['value']} [1]"
    return f"{gold.get('answer', '')} [1]"


def to_sft_example(item: dict) -> dict:
    """One item -> a chat SFT record: {"messages": [system, user, assistant]}.
    Built from :func:`_grounded_messages` -- the SAME system+user construction
    used at inference -- instead of rebuilding ``[system, user]`` by hand and
    reaching for the private ``_SYSTEM`` constant (CLAUDE.md: one implementation
    of the grounded-prompt contract)."""
    context = _format_context(_as_hits(item["evidence"], item.get("doc_id", "DOC")))
    messages = _grounded_messages(item["question"], context)
    return {"messages": [*messages, {"role": "assistant", "content": _target_answer(item)}]}
