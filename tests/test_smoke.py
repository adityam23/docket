"""Offline smoke tests — no network, no backend required; tests stay deterministic and offline. A live backend is exercised separately by
`dk health` / `dk chat` in the handoff verification, not in pytest."""

from docket.config import Profile, load_settings
from docket.providers.base import Capability
from docket.providers.router import get_provider
from docket.trust.reliability import reliability_label


def test_defaults():
    s = load_settings()
    assert s.profile == Profile.lite
    assert s.backend_url.endswith("/v1")   # mettle's OpenAI-compatible surface


def test_router_builds_local_provider():
    prov = get_provider(load_settings())
    assert prov.name == "mettle"
    assert Capability.CHAT in prov.capabilities


def test_slot_unavailable_becomes_graceful_answer(monkeypatch):
    """A mettle ``slot_unavailable`` 503 from provider.chat must surface as a
    user-facing Answer (reason + missing model), not a bare 500."""
    from docket.agent.graph import Answer
    from docket.providers.base import SlotUnavailable
    import docket.service as service

    class _Retriever:
        def retrieve(self, query, k=0):
            return []
        def iterative_retrieve(self, queries, k=0):
            return []

    class _Provider:
        capabilities = Capability.CHAT | Capability.LOGPROBS
        def chat(self, messages, **kw):
            raise SlotUnavailable(reason="vram", missing=["qwen3:latest"], profile_degraded=True)

    monkeypatch.setattr(service, "get_provider", lambda s: _Provider())
    res = service.ask("what does the filing say?", retriever=_Retriever())
    assert isinstance(res, Answer)
    assert "isn't resident" in res.answer
    assert "qwen3:latest" in res.answer
    assert res.detail == {"reason": "vram", "missing": ["qwen3:latest"], "profile_degraded": True}


def test_reliability_label_ordering():
    confident = [-0.05, -0.02, -0.10]  # low surprisal
    unsure = [-1.5, -2.0, -1.2]        # high surprisal
    assert reliability_label(confident) == "high"
    assert reliability_label(unsure) == "low"
    assert reliability_label([]) == "unknown"
