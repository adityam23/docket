"""Engine bearer auth, shared by the /v1 provider and the /engine supervisor
(a neutral module: ``providers`` imports ``engine``, so neither can host it)."""

from __future__ import annotations


def bearer_headers(key: str | None) -> dict[str, str]:
    """The ONE place an ``Authorization: Bearer`` header gets built (CLAUDE.md
    'Reusability') -- every /v1 route, including /engine/*, takes the same
    bearer auth. Returns {} when no key is configured (local dev)."""
    return {"Authorization": f"Bearer {key}"} if key else {}
