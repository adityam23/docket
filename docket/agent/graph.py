"""The agent: grounded, cited answering with iterative agentic re-query and an
always-on Tier-1 reliability label.

``answer()`` is the single orchestration implementation — a deterministic loop
that both the CLI and the web ``/ask`` endpoint call, and that ``build_graph()``
wraps in LangGraph for the ``agent`` extra (so there is exactly one control flow,
not two — CLAUDE.md reusability). It never imports LangGraph itself, so it runs
under the base install and in the offline test suite.

Flow per hop:
  retrieve → format numbered context → grounded chat (logprobs on) →
  reliability_label(logprobs). If the answer looks low-confidence and hops
  remain, ask the model for better search terms and re-query (multi-hop recall).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..config import Settings, load_settings
from ..providers.base import Provider
from ..report.schema import Citation
from ..retrieval.retriever import Retriever
from ..trust.reliability import reliability_score
from .tools import aggregate, parse_number
from .trace import StepSink, hit_view

_CITE = re.compile(r"\[(\d+)\]")


def _emit(on_step: StepSink | None, kind: str, **info: object) -> None:
    """Fire a trace/step event if a sink is attached; a no-op otherwise, so the
    loop is unchanged for the CLI and the offline tests (mirrors pipeline.emit)."""
    if on_step is not None:
        on_step(kind, dict(info))

_SYSTEM = (
    "You are a careful document analyst. Answer ONLY from the numbered context. "
    "Cite every claim with the bracketed source number, e.g. [2]. If the context "
    "does not contain the answer, say exactly: 'The documents do not contain this.' "
    "Do not use outside knowledge."
)


@dataclass
class Answer:
    question: str
    answer: str
    reliability: str
    surprisal: float
    citations: list[Citation] = field(default_factory=list)
    hits: list[dict] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    # Structured breakdown for aggregation answers (op/result/unit/per-item),
    # None for plain QA. Lets the report/web show an auditable line-item table.
    detail: dict | None = None


def _format_context(hits: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] ({h['doc_id']} p.{h['page']}) {h['text']}"
        for i, h in enumerate(hits, start=1)
    )


def _grounded_messages(
    question: str, context: str, history: list[dict] | None = None
) -> list[dict]:
    """Build the grounded chat messages.

    With a multi-turn session (T21) the bounded prior turns are threaded in as
    alternating user/assistant messages AFTER the system prompt and BEFORE the
    final grounded user message — so the model keeps conversational memory while
    the current question still anchors on the freshly retrieved context.
    """
    user = f"Context:\n{context}\n\nQuestion: {question}"
    msgs: list[dict] = [{"role": "system", "content": _SYSTEM}]
    if history:
        msgs.extend(history)
    msgs.append({"role": "user", "content": user})
    return msgs


# Context-window management for threaded conversations (T21): keep the most
# recent turns and drop oldest-first until the total fits the character budget.
# Deliberately conservative constants rather than config knobs — they bound a
# prompt that already carries retrieved chunks; revisit with the eval harness.
_MAX_HISTORY_TURNS = 6       # Q&A pairs kept
_MAX_HISTORY_CHARS = 4_000   # total budget for the threaded history


def trim_history(history: list[dict]) -> list[dict]:
    """Bound conversation history to the recent window.

    Keeps whole user/assistant pairs from the END of the conversation (most
    recent first), never exceeding ``_MAX_HISTORY_TURNS`` pairs or
    ``_MAX_HISTORY_CHARS`` characters total. Returns turns in chronological
    order; malformed entries are dropped rather than sent to the model.
    """
    if not history:
        return []
    pairs: list[list[dict]] = []
    i = 0
    while i < len(history):
        msg = history[i]
        if not isinstance(msg, dict) or msg.get("role") not in ("user", "assistant"):
            i += 1  # skip garbage instead of forwarding it
            continue
        nxt = history[i + 1] if i + 1 < len(history) else None
        if msg["role"] == "user" and isinstance(nxt, dict) and nxt.get("role") == "assistant":
            pairs.append([msg, nxt])
            i += 2
        else:
            pairs.append([msg])
            i += 1

    kept: list[list[dict]] = []
    total = 0
    for pair in reversed(pairs[-_MAX_HISTORY_TURNS:]):
        cost = sum(len(m.get("content") or "") for m in pair)
        if kept and total + cost > _MAX_HISTORY_CHARS:
            break
        total += cost
        kept.insert(0, pair)
    return [m for pair in kept for m in pair]


def _refine_query(provider: Provider, question: str, context: str) -> str:
    """Ask the model for better retrieval keywords when a hop came up short."""
    msg = [
        {
            "role": "user",
            "content": (
                "The retrieved context did not answer the question. Give a short "
                "alternative search query (keywords only, no punctuation) likely to "
                f"find the answer.\n\nQuestion: {question}"
            ),
        }
    ]
    try:
        return provider.chat(msg, temperature=0.0, max_tokens=32).text.strip() or question
    except Exception:  # noqa: BLE001 - refinement is best-effort
        return question


def _citations(text: str, hits: list[dict]) -> list[Citation]:
    """Resolve [n] markers in the answer to source hits (cite_page semantics).

    Falls back to the top hit when the model answered without explicit markers,
    so a grounded answer is never left uncited.
    """
    used = {int(n) for n in _CITE.findall(text)}
    idxs = sorted(i for i in used if 1 <= i <= len(hits))
    if not idxs and hits:
        idxs = [1]
    return [Citation.from_hit(hits[i - 1]) for i in idxs]


# --- deterministic aggregation path (typed tools, no code-gen)
#
# A cross-document "total/average/max across a bunch of docs" question fails the
# plain QA path because the model has to both find every value AND do arithmetic
# in its head. Here the arithmetic is deterministic (:func:`aggregate`); the model
# only EXTRACTS each cited value. The planner below is a heuristic seam — with a
# native function-calling backend the model would pick the op/tool itself, but the
# extract→aggregate→cite mechanics stay exactly the same.


@dataclass
class Operation:
    name: str   # one of tools.AGG_OPS
    target: str  # what to extract for (the raw question today)


# Checked in order; first hit wins. count/extremum/mean are matched before the
# broad "sum" cues so "average ... total" reads as mean, not sum.
_OP_TRIGGERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("count", ("how many", "number of", "count of")),
    ("max", ("highest", "largest", "maximum", "most expensive", "biggest", "greatest")),
    ("min", ("lowest", "smallest", "minimum", "cheapest", "least expensive")),
    ("mean", ("average", "on average", "mean value", "typical")),
    ("sum", ("total", "combined", "altogether", "in total", "overall", "added up",
             "add up", "sum of", "aggregate", "across all")),
)
_PERIOD_CUES = ("per month", "per year", "monthly", "each month", "annually")


def plan_operation(question: str) -> Operation | None:
    """Classify a question as an aggregation op, or None for plain QA.

    Deterministic and offline-testable. Returns None (→ QA path) for anything
    that isn't clearly a cross-value roll-up, so single-fact recall is untouched.
    """
    q = question.lower()
    for name, kws in _OP_TRIGGERS:
        if any(kw in q for kw in kws):
            return Operation(name=name, target=question)
    # "... across ... per month/monthly ..." implies summation even without "total".
    if "across" in q and any(c in q for c in _PERIOD_CUES):
        return Operation(name="sum", target=question)
    return None


_EXTRACT_SYSTEM = (
    "You extract numbers from numbered context. For the requested quantity, output "
    "ONLY a JSON array. For each context item that states a relevant value, emit "
    '{"i": <item number>, "value": "<number with unit, e.g. $1,200 or 5%>"}. '
    "Omit items without the quantity. Do NOT add, average, or otherwise compute. "
    "Return JSON only, no prose."
)


def _extract_messages(target: str, context: str) -> list[dict]:
    user = f"Context:\n{context}\n\nExtract this quantity from every relevant item: {target}"
    return [{"role": "system", "content": _EXTRACT_SYSTEM}, {"role": "user", "content": user}]


def _json_array(text: str) -> list:
    """Tolerantly pull the first JSON array out of a model response."""
    a, b = text.find("["), text.rfind("]")
    if a == -1 or b <= a:
        return []
    try:
        val = json.loads(text[a : b + 1])
    except (json.JSONDecodeError, ValueError):
        return []
    return val if isinstance(val, list) else []


def _parse_extraction(text: str, hits: list[dict]) -> list[dict]:
    """Turn the model's extraction JSON into cited ``{value, unit, citation}``
    items. Every value is anchored to the hit it was pulled from, so the total is
    auditable line-by-line. Nulls / unparseable / out-of-range entries are dropped
    (a value we can't parse is safer omitted than summed wrong)."""
    items: list[dict] = []
    for entry in _json_array(text):
        if not isinstance(entry, dict):
            continue
        try:
            i = int(entry["i"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (1 <= i <= len(hits)) or entry.get("value") is None:
            continue
        try:
            value, unit = parse_number(entry["value"])
        except ValueError:
            continue
        h = hits[i - 1]
        items.append(
            {
                "value": value,
                "unit": unit,
                "citation": Citation.from_hit(h),
                "label": f"{h['doc_id']} p.{h['page']}",
            }
        )
    return _dedup_items(items)


def _dedup_items(items: list[dict]) -> list[dict]:
    """Drop values that are the same figure counted twice via overlapping chunks.

    Retrieval chunks overlap by ``chunk_overlap`` words, so one figure can surface
    in two adjacent retrieved chunks; extracting from both would sum/average it
    twice and inflate the total (the reported bug). Two items are the *same
    figure* when they share document, page, value and unit — collapse them to one,
    keeping the first (its citation). This is deliberately conservative: two
    genuinely distinct line items with the identical value on the same page also
    collapse; that under-counts rather than invents a larger number, which is the
    safer failure for a trust-first tool. Order is preserved for stable [n]
    markers.
    """
    seen: set[tuple] = set()
    out: list[dict] = []
    for it in items:
        c = it["citation"]
        key = (c.doc_id, c.page, round(float(it["value"]), 6), it["unit"])
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


_OP_LABEL = {"sum": "Total", "mean": "Average", "min": "Minimum", "max": "Maximum", "count": "Count"}


def _fmt(value: float, unit: str) -> str:
    if unit == "USD":
        return f"${value:,.2f}"
    if unit == "%":
        return f"{value:g}%"
    return f"{value:g}"


def _render_aggregate(op: Operation, agg: dict) -> str:
    """Deterministic prose with per-item [n] markers matching the breakdown order."""
    lines = [
        f"{_OP_LABEL[op.name]} = {_fmt(agg['result'], agg['unit'])} across "
        f"{agg['count']} value(s)."
    ]
    for j, it in enumerate(agg["items"], start=1):
        lines.append(f"- {_fmt(it['value'], it['unit'])} [{j}] ({it['label']})")
    return "\n".join(lines)


def _coverage(agg: dict, hits: list[dict]) -> dict:
    """How complete the roll-up is: distinct documents that contributed a value
    vs. distinct documents present in the retrieved scope. ``partial`` flags that
    some in-scope document yielded no value — the total may be undercounting, and
    we say so rather than present a falsely precise number."""
    contributing = {it["citation"].doc_id for it in agg["items"]}
    in_scope = {h["doc_id"] for h in hits}
    return {
        "documents_contributing": len(contributing),
        "documents_in_scope": len(in_scope),
        "partial": bool(in_scope) and len(contributing) < len(in_scope),
    }


def _public_detail(agg: dict, coverage: dict) -> dict:
    """JSON-serialisable view of the aggregate (Citation → dict) for web/report."""
    return {
        "op": agg["op"],
        "result": agg["result"],
        "unit": agg["unit"],
        "count": agg["count"],
        "coverage": coverage,
        "breakdown": [
            {
                "value": it["value"],
                "unit": it["unit"],
                "label": it["label"],
                "citation": it["citation"].model_dump(),
            }
            for it in agg["items"]
        ],
    }


def _aggregate_answer(
    question: str,
    op: Operation,
    *,
    retriever: Retriever,
    provider: Provider,
    settings: Settings,
    on_step: StepSink | None = None,
) -> Answer:
    _emit(on_step, "plan", path="aggregate", op=op.name, question=question)
    # Broader recall than QA: an aggregate must see *every* contributing value,
    # so pull more chunks (they may be spread across many documents).
    k = max(settings.context_chunks, 8) * 2
    hits = retriever.retrieve(question, k=k)
    _emit(on_step, "retrieve", hop=0, query=question, hits=[hit_view(h) for h in hits])
    if not hits:
        _emit(on_step, "final", text="The documents do not contain this.", reliability="unknown")
        return Answer(question, "The documents do not contain this.", "unknown", float("nan"))

    extract_msgs = _extract_messages(op.target, _format_context(hits))
    _emit(on_step, "extract_prompt", hop=0, messages=extract_msgs)
    res = provider.chat(extract_msgs, temperature=0.0, max_tokens=700)
    _emit(on_step, "extract_response", hop=0, text=res.text)
    items = _parse_extraction(res.text, hits)
    _emit(
        on_step,
        "extract_items",
        items=[{"value": it["value"], "unit": it["unit"], "label": it["label"]} for it in items],
    )
    reliability, surprisal = reliability_score(res.logprobs or [])
    try:
        agg = aggregate(op.name, items)
    except ValueError as e:
        # Refuse to invent a number: no values, or incomparable units.
        msg = (
            "The documents do not contain this."
            if not items
            else f"Could not compute a single {op.name}: {e}."
        )
        _emit(on_step, "final", text=msg, reliability="low" if items else "unknown")
        return Answer(question, msg, "low" if items else "unknown", surprisal, hits=hits)

    coverage = _coverage(agg, hits)
    body = _render_aggregate(op, agg)
    if coverage["partial"]:
        body += (
            f"\n\nNote: only {coverage['documents_contributing']} of "
            f"{coverage['documents_in_scope']} documents in scope contributed a "
            "value, so this may be incomplete."
        )
    _emit(on_step, "final", text=body, reliability=reliability)
    return Answer(
        question=question,
        answer=body,
        reliability=reliability,
        surprisal=surprisal,
        citations=[it["citation"] for it in agg["items"]],
        hits=hits,
        queries=[question],
        detail=_public_detail(agg, coverage),
    )


def answer(
    question: str,
    *,
    retriever: Retriever,
    provider: Provider,
    settings: Settings | None = None,
    on_step: StepSink | None = None,
    history: list[dict] | None = None,
) -> Answer:
    """Answer over the corpus. ``on_step`` (optional) receives a trace event per
    stage for Observability / the live timeline; when None the loop is unchanged
    (CLI + offline tests). One control flow, instrumented once.

    ``history`` (optional) is prior conversation context as ``[{role, content}]``
    user/assistant turns (T21): it is trimmed to the recent window and threaded
    into the grounded prompt so follow-up questions keep their thread. The
    deterministic aggregation path ignores it — extraction must stay anchored to
    the retrieved documents, not the chat.
    """
    s = settings or load_settings()
    op = plan_operation(question)
    if op is not None:
        return _aggregate_answer(
            question, op, retriever=retriever, provider=provider, settings=s, on_step=on_step
        )
    _emit(on_step, "plan", path="qa", question=question)
    bounded_history = trim_history(history or [])
    if bounded_history:
        _emit(on_step, "history", turns=len(bounded_history), chars=sum(len(m.get("content") or "") for m in bounded_history))
    queries = [question]
    hits = retriever.retrieve(question, k=s.context_chunks)
    _emit(on_step, "retrieve", hop=0, query=question, hits=[hit_view(h) for h in hits])
    res = None
    reliability, surprisal = "unknown", float("nan")
    for hop in range(max(1, s.max_hops)):
        context = _format_context(hits)
        _emit(on_step, "context", hop=hop, context=context)
        messages = _grounded_messages(question, context, history=bounded_history)
        _emit(on_step, "prompt", hop=hop, messages=messages)
        res = provider.chat(messages, temperature=0.0, max_tokens=700)
        reliability, surprisal = reliability_score(res.logprobs or [])
        _emit(on_step, "response", hop=hop, text=res.text, reliability=reliability)
        last_hop = hop == max(1, s.max_hops) - 1
        # Re-query only when the label is genuinely low AND we have a signal to
        # trust; without logprobs the label is 'unknown' and re-querying is noise.
        if reliability != "low" or last_hop or not res.logprobs:
            break
        nq = _refine_query(provider, question, context)
        queries.append(nq)
        _emit(on_step, "refine", hop=hop, query=nq)
        hits = retriever.iterative_retrieve(queries, k=s.context_chunks)
        _emit(on_step, "retrieve", hop=hop + 1, query=nq, hits=[hit_view(h) for h in hits])

    text = res.text if res else "The documents do not contain this."
    _emit(on_step, "final", text=text, reliability=reliability)
    return Answer(
        question=question,
        answer=text,
        reliability=reliability,
        surprisal=surprisal,
        citations=_citations(text, hits),
        hits=hits,
        queries=queries,
    )


def build_graph(retriever: Retriever, provider: Provider):
    """LangGraph wrapper around the same ``answer()`` flow (needs the agent extra:
    ``uv sync --extra agent``). A single-node graph today; the iterative loop and
    per-tool nodes expand here without changing the orchestration semantics."""
    try:
        from langgraph.graph import END, START, StateGraph  # type: ignore
    except ImportError as e:  # pragma: no cover - needs the agent extra
        raise RuntimeError("install the agent extra: uv sync --extra agent") from e

    def _node(state: dict) -> dict:  # pragma: no cover - exercised with langgraph
        ans = answer(state["question"], retriever=retriever, provider=provider)
        return {"result": ans}

    g = StateGraph(dict)
    g.add_node("answer", _node)
    g.add_edge(START, "answer")
    g.add_edge("answer", END)
    return g.compile()
