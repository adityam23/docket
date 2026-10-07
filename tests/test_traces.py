"""Offline tests for the per-chat trace store (T04): the on_step sink threaded
through the agent loop, the JSONL ring-buffer store, and the traced service path.
No network — a fake provider stands in for inference."""

from __future__ import annotations

import json

from docket.agent.graph import answer
from docket.agent.trace import TraceCollector, TraceStore, compact_step, hit_view
from docket.config import Settings

from .fakes import FakeEngine, retriever_over, acme_chunks


def test_answer_emits_ordered_trace_steps():
    steps: list[tuple[str, dict]] = []
    r = retriever_over(acme_chunks())
    answer(
        "What was revenue?",
        retriever=r,
        provider=FakeEngine(),
        settings=Settings(max_hops=1),
        on_step=lambda k, i: steps.append((k, i)),
    )
    kinds = [k for k, _ in steps]
    assert kinds[0] == "plan"
    assert kinds[-1] == "final"
    assert "retrieve" in kinds and "prompt" in kinds and "response" in kinds

    # The prompt step carries the VERBATIM system+user messages (T05 shows these).
    prompt = next(i for k, i in steps if k == "prompt")
    assert prompt["messages"][0]["role"] == "system"
    assert "Context:" in prompt["messages"][1]["content"]

    # The retrieve step carries full hit text (the always-present verification
    # fallback — T13) with score/rank/page.
    retr = next(i for k, i in steps if k == "retrieve")
    assert retr["hits"] and retr["hits"][0]["text"]
    assert "page" in retr["hits"][0]


def test_answer_without_sink_is_unchanged():
    r = retriever_over(acme_chunks())
    res = answer("What was revenue?", retriever=r, provider=FakeEngine(), settings=Settings())
    assert res.answer and res.citations  # no on_step passed → loop behaves as before


def test_hit_view_shape():
    v = hit_view({"chunk_id": "c#0", "doc_id": "d", "page": 2, "score": 0.5, "rank": 1, "text": "hi"})
    assert v == {"chunk_id": "c#0", "doc_id": "d", "page": 2, "score": 0.5, "rank": 1,
                 "form_type": "", "company": "", "ticker": "", "fiscal_period": "",
                 "text": "hi"}
    assert hit_view({})["text"] == ""  # missing text → empty, never None


def test_compact_step_is_light():
    c = compact_step("retrieve", {"hop": 0, "query": "q", "hits": [{"text": "x"}, {"text": "y"}]})
    assert c == {"kind": "retrieve", "hop": 0, "query": "q", "hits": 2}
    c2 = compact_step("response", {"hop": 0, "text": "hello", "reliability": "high"})
    assert c2["kind"] == "response" and c2["chars"] == 5


def test_trace_store_ring_buffer(tmp_path):
    store = TraceStore(str(tmp_path), cap=3)
    for n in range(5):
        store.append(
            {"id": f"t{n}", "question": f"q{n}", "created_at": "x", "steps": [{"kind": "final"}]}
        )
    ids = [t["id"] for t in store.list()]
    assert ids == ["t4", "t3", "t2"]            # newest first, oldest pruned past cap
    assert store.get("t4")["question"] == "q4"
    assert store.get("t0") is None             # pruned
    assert "steps" not in store.list()[0]      # summaries omit the heavy steps


def test_trace_store_delete_and_cascade(tmp_path):
    store = TraceStore(str(tmp_path), cap=10)
    store.append({"id": "solo", "question": "q", "created_at": "x"})
    store.append({"id": "c1t1", "question": "q", "created_at": "x", "chat_id": "chat-1"})
    store.append({"id": "c1t2", "question": "q", "created_at": "x", "chat_id": "chat-1"})

    assert store.delete("solo") is True
    assert store.get("solo") is None
    assert store.delete("solo") is False        # already gone

    # cascade: every trace of a chat drops together, others untouched
    assert store.delete_by_chat("chat-1") == 2
    assert store.delete_by_chat("chat-1") == 0
    assert [t["id"] for t in store.list()] == []


def test_ids_unique_at_cap_within_same_second():
    """B6: the id's suffix used to be `len(records) + 1` — pinned at `cap + 1`
    once the ring buffer filled (next id was always "count+1"), and two ids
    minted within the same one-second timestamp resolution collided outright.
    Minting many ids from the SAME created_at (a frozen/same-second clock)
    must still produce unique ids, with no disk access needed to do it."""
    store = TraceStore("/does/not/exist", cap=5)
    created_at = "2026-08-21T10:00:00+00:00"
    ids = [store.next_id(created_at) for _ in range(50)]
    assert len(set(ids)) == 50
    assert all(i.startswith("trace-") for i in ids)  # existing prefix format kept


def test_trace_store_get_delete_correct_after_cap_with_same_timestamp(tmp_path):
    """Regression for the id-collision bug end to end: fill past the cap with
    records minted from the identical timestamp, then confirm get/delete each
    resolve to exactly the record asked for — never "the wrong one"."""
    store = TraceStore(str(tmp_path), cap=5)
    created_at = "2026-08-21T10:00:00+00:00"
    ids = [store.next_id(created_at) for _ in range(10)]
    for i, tid in enumerate(ids):
        store.append({"id": tid, "question": f"q{i}", "created_at": created_at})

    kept_ids = [t["id"] for t in store.list()]
    assert len(kept_ids) == 5
    for tid in kept_ids:
        assert store.get(tid)["id"] == tid   # resolves to itself, not a collided sibling

    victim = kept_ids[0]
    assert store.delete(victim) is True
    assert store.get(victim) is None
    assert len(store.list()) == 4             # only the one victim removed


def test_trace_store_concurrent_appends_no_loss_or_raise(tmp_path):
    """B6 race: `_write_jsonl` used to write a FIXED `path + ".tmp"`, so two
    concurrent appends (two tabs, or the SSE stream racing the POST fallback —
    FastAPI runs sync endpoints concurrently in a threadpool) could both
    truncate the same tmp file and have the loser's `os.replace` raise
    `FileNotFoundError`, or silently lose one writer's record. Every one of N
    concurrent appends must land, and none may raise."""
    import threading

    store = TraceStore(str(tmp_path), cap=200)
    n = 40
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            store.append(
                {"id": store.next_id("2026-08-21T10:00:00+00:00"), "question": f"q{i}", "created_at": "x"}
            )
        except BaseException as e:  # noqa: BLE001 - captured, not raised, from a thread
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    listed = store.list()
    assert len(listed) == n                      # none lost to a lost-update race
    assert len({t["id"] for t in listed}) == n   # and none collided either


def test_collector_build_record():
    from docket.report.schema import Citation

    class _Ans:  # minimal Answer-shaped object
        answer = "A [1]."
        reliability = "high"
        surprisal = 0.0123
        citations = [Citation(doc_id="d", page=1, quote="q")]
        queries = ["q1", "q2"]
        detail = None

    col = TraceCollector("q1", now="2026-08-19T12:00:00+00:00")
    col("plan", {"path": "qa"})
    col("final", {"text": "A [1]."})
    rec = col.build(_Ans(), trace_id="trace-1", elapsed_ms=7)
    assert rec["id"] == "trace-1" and rec["hops"] == 2 and rec["path"] == "qa"
    assert rec["surprisal"] == 0.0123 and rec["elapsed_ms"] == 7
    assert rec["steps"][0]["kind"] == "plan"


def test_ask_traced_persists_and_leaks_no_secret(tmp_path, monkeypatch):
    import docket.service as svc

    monkeypatch.setattr(svc, "get_provider", lambda s: FakeEngine())
    s = Settings(index_dir=str(tmp_path))
    res, trace = svc.ask_traced("What was revenue?", settings=s, retriever=retriever_over(acme_chunks()))

    assert trace["id"] and trace["question"] == "What was revenue?"
    assert trace["steps"] and trace["steps"][-1]["kind"] == "final"
    assert trace["elapsed_ms"] >= 0

    # Persisted and retrievable via the same store.
    got = svc.trace_store(s).get(trace["id"])
    assert got and got["answer"] == res.answer

    # No secret material ever lands in a trace.
    blob = json.dumps(trace).lower()
    assert "api_key" not in blob and "authorization" not in blob


# --- T21: first-class Chat/Session (chat → turns → trace) --------------------


class CapturingProvider(FakeEngine):
    """Records every chat() call so tests can assert the exact messages."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.calls: list[list[dict]] = []

    def chat(self, messages, **kw):
        self.calls.append(messages)
        return super().chat(messages, **kw)


def test_trim_history_bounds_window():
    from docket.agent.graph import trim_history

    # Six pairs fit exactly; a seventh drops the OLDEST.
    hist = []
    for n in range(7):
        hist += [{"role": "user", "content": f"q{n}"}, {"role": "assistant", "content": f"a{n}"}]
    kept = trim_history(hist)
    assert kept[0] == {"role": "user", "content": "q1"}  # q0 dropped
    assert kept[-1] == {"role": "assistant", "content": "a6"}

    # A huge recent pair pushes older ones out via the character budget.
    big = [{"role": "user", "content": "x" * 3900}, {"role": "assistant", "content": "y" * 90}]
    kept2 = trim_history(hist + big)
    assert {"role": "user", "content": "x" * 3900} in kept2
    assert sum(len(m["content"]) for m in kept2) <= 4000

    # Garbage entries are dropped, never forwarded.
    assert trim_history([{"nope": 1}, {"role": "user", "content": "q"}]) == [
        {"role": "user", "content": "q"}
    ]
    assert trim_history([]) == []


def test_answer_threads_history_into_prompt():
    r = retriever_over(acme_chunks())
    p = CapturingProvider()
    history = [{"role": "user", "content": "What was revenue?"}, {"role": "assistant", "content": "$10M [1]."}]
    answer(
        "And vs last year?",
        retriever=r,
        provider=p,
        settings=Settings(max_hops=1),
        history=history,
    )
    msgs = p.calls[0]
    assert msgs[0]["role"] == "system"
    # Prior turns sit between system and the final grounded user message.
    assert msgs[1]["role"] == "user" and msgs[1]["content"] == "What was revenue?"
    assert msgs[2]["role"] == "assistant" and msgs[2]["content"] == "$10M [1]."
    assert msgs[-1]["role"] == "user" and "Question: And vs last year?" in msgs[-1]["content"]

    # No history → identical shape to before (back-compat).
    p2 = CapturingProvider()
    answer("What was revenue?", retriever=r, provider=p2, settings=Settings(max_hops=1))
    assert [m["role"] for m in p2.calls[0]] == ["system", "user"]


def test_chat_store_roundtrip_and_cap(tmp_path):
    from docket.agent.trace import ChatStore

    store = ChatStore(str(tmp_path), cap=3)
    c1 = store.next_id("2026-08-21T10:00:00+00:00")
    store.append_turn(c1, {"trace_id": "t1", "question": "first", "reliability": "high",
                           "created_at": "2026-08-21T10:00:00+00:00"})
    store.append_turn(c1, {"trace_id": "t2", "question": "follow-up", "reliability": "medium",
                           "created_at": "2026-08-21T10:01:00+00:00"})

    chat = store.get(c1)
    assert chat["title"] == "first"          # first turn names the chat
    assert [t["trace_id"] for t in chat["turns"]] == ["t1", "t2"]

    # Ring buffer caps whole chats.
    ids = []
    for n in range(5):
        cid = store.next_id(f"2026-08-21T11:{n:02d}:00+00:00")
        store.append_turn(cid, {"trace_id": f"x{n}", "question": "q", "created_at": "now"})
        ids.append(cid)
    listed = store.list()
    assert len(listed) == 3
    assert store.get(c1) is None             # oldest pruned

    # Delete removes only that record.
    assert store.delete(ids[-1]) is True
    assert store.get(ids[-1]) is None
    assert len(store.list()) == 2


def test_ask_traced_session_nests_turns(tmp_path, monkeypatch):
    import docket.service as svc

    monkeypatch.setattr(svc, "get_provider", lambda s: FakeEngine())
    s = Settings(index_dir=str(tmp_path))

    chats = svc.chat_store(s)
    session_id = chats.next_id("2026-08-21T12:00:00+00:00")
    chats.create(session_id, "", now="2026-08-21T12:00:00+00:00")  # minted via POST /api/chats (untitled)

    res1, t1 = svc.ask_traced("What was revenue?", settings=s, retriever=retriever_over(acme_chunks()), session_id=session_id)
    res2, t2 = svc.ask_traced("Summarize the key risk factors.", settings=s, retriever=retriever_over(acme_chunks()), session_id=session_id)

    # Traces nest under the chat with an ordered turn index.
    assert t1["chat_id"] == session_id and t1["turn_index"] == 0
    assert t2["chat_id"] == session_id and t2["turn_index"] == 1

    chat = chats.get(session_id)
    assert [t["trace_id"] for t in chat["turns"]] == [t1["id"], t2["id"]]
    assert chat["title"] == "What was revenue?"   # first turn names it

    # History derives user/assistant turns from the stored traces (one source of truth).
    hist = svc.session_history(session_id, settings=s)
    roles = [(m["role"], m["content"]) for m in hist]
    assert ("user", "What was revenue?") in roles
    assert ("assistant", res1.answer) in roles
    assert ("assistant", res2.answer) in roles


def test_ask_traced_without_session_unchanged(tmp_path, monkeypatch):
    """Back-compat: no session_id → standalone one-shot exactly as before."""
    import docket.service as svc

    monkeypatch.setattr(svc, "get_provider", lambda s: FakeEngine())
    s = Settings(index_dir=str(tmp_path))
    _, trace = svc.ask_traced("What was revenue?", settings=s, retriever=retriever_over(acme_chunks()))

    assert trace["chat_id"] is None and trace["turn_index"] is None
    assert svc.chat_store(s).list() == []       # no chat records created
    try:
        svc.session_history("missing", settings=s)
        raise AssertionError("expected ValueError for unknown session")
    except ValueError:
        pass


def test_chat_store_new_mints_id_and_equal_timestamps(monkeypatch, tmp_path):
    """B6: ``create`` used to default ``now`` via two SEPARATE ``_utc_now()``
    calls (one for created_at, one for updated_at) when the caller (the old
    ``/api/chats`` route) didn't pass one — they could disagree by whatever
    elapsed between the calls. ``ChatStore.new`` mints the id and both
    timestamps from exactly one clock read."""
    import docket.agent.trace as trace_mod

    calls = {"n": 0}

    def fake_now() -> str:
        calls["n"] += 1
        return f"2026-08-21T10:00:0{calls['n']}+00:00"  # a DIFFERENT value per call

    monkeypatch.setattr(trace_mod, "_utc_now", fake_now)
    store = trace_mod.ChatStore(str(tmp_path))
    chat = store.new("hello")

    assert calls["n"] == 1                               # exactly one clock read
    assert chat["created_at"] == chat["updated_at"]       # so they can't disagree
    assert chat["title"] == "hello" and chat["turns"] == []
    assert chat["id"].startswith("chat-")                 # existing prefix format kept
    assert store.get(chat["id"]) == chat                  # minted AND persisted
