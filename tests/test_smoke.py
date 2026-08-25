"""Offline smoke tests — no network, no backend required (per docs/decisions.md:
tests stay deterministic and offline). A live backend is exercised separately by
`dk health` / `dk chat` in the handoff verification, not in pytest."""

from docket.config import Profile, Provider, load_settings
from docket.providers.base import Capability
from docket.providers.router import get_provider
from docket.trust.reliability import reliability_label


def test_defaults():
    s = load_settings()
    assert s.profile == Profile.lite
    assert s.provider == Provider.local
    assert s.backend_url.endswith("/v1")


def test_router_builds_local_provider():
    prov = get_provider(load_settings())
    assert prov.name == "local"
    assert Capability.CHAT in prov.capabilities


def test_reliability_label_ordering():
    confident = [-0.05, -0.02, -0.10]  # low surprisal
    unsure = [-1.5, -2.0, -1.2]        # high surprisal
    assert reliability_label(confident) == "high"
    assert reliability_label(unsure) == "low"
    assert reliability_label([]) == "unknown"
