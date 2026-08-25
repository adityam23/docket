"""The ONE place runtime settings are persisted from the dashboard (Settings page
T01 + BYOK keys T02). Everything is written to ``.env.local`` — the machine-local
overlay that layers over ``.env`` (see ``config._ENV_LOCAL_FILE``) — so a change
takes effect on the next ``load_settings()`` and survives a restart, without ever
mutating the hand-authored, comment-rich ``.env``.

Security (CLAUDE.md — secrets are first-class):
- The file is written ``0600`` (it may hold API keys).
- Only an allowlist of keys is writable; unknown keys are rejected.
- Values are type-validated through the real ``Settings`` model before landing.
- Values may not contain newlines (no ``KEY=…\\nINJECTED=…`` line injection).
- Secrets are never read back out — the read models keep redacting them to bools.

One writer, reused by the knobs endpoint and the keys endpoint (no second
``.env`` mutation implementation anywhere).
"""

from __future__ import annotations

from enum import Enum

from pydantic import ValidationError

from .config import _ENV_LOCAL_FILE, Settings, load_settings

# Non-secret runtime settings the Settings page may change. Names are the
# ``Settings`` field names; persisted as ``DK_<UPPER>`` with pydantic's prefix.
WRITABLE_FIELDS: frozenset[str] = frozenset(
    {
        "provider",
        "backend_url",
        "chat_model",
        "embed_url",
        "embed_model",
        "request_timeout_s",
        "chunk_words",
        "chunk_overlap",
        "retrieval_k",
        "context_chunks",
        "max_hops",
    }
)

# UI key name -> the env var it persists to. Extendable as providers are added.
SECRET_ENV: dict[str, str] = {
    "cerebras": "DK_CEREBRAS_API_KEY",
    "groq": "DK_GROQ_API_KEY",
}

_PREFIX = "DK_"


class ConfigWriteError(ValueError):
    """A rejected write (unknown key, bad type, or unsafe value)."""


def _envstr(value: object) -> str:
    """Serialise a validated Settings value to a ``.env`` scalar."""
    if isinstance(value, Enum):
        value = value.value
    s = str(value)
    if "\n" in s or "\r" in s:
        raise ConfigWriteError("values may not contain newlines")
    return s


def _read_env_local() -> dict[str, str]:
    """Parse the existing ``.env.local`` into a dict (empty if absent).

    Deliberately minimal — we only ever wrote it ourselves (plain ``KEY=value``
    lines, no quoting/interpolation), so a full dotenv parser is unwarranted.
    Comments and blanks are preserved by being dropped from the dict and never
    re-emitted; this file is machine-owned, so that is fine.
    """
    out: dict[str, str] = {}
    try:
        with open(_ENV_LOCAL_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    return out


def _write_env_local(env: dict[str, str]) -> None:
    """Atomically rewrite ``.env.local`` from ``env`` with 0600 perms.

    Written to a sibling temp file then ``os.replace``d in, so a crash mid-write
    never leaves a half-written secrets file. Created 0600 from the start (never
    a window where the key is world-readable).
    """
    import os
    import tempfile

    body = (
        "# Machine-local overrides written by the dashboard (Settings + API keys).\n"
        "# Layers over .env; gitignored; do not commit. Edit via the app.\n"
        + "".join(f"{k}={v}\n" for k, v in sorted(env.items()))
    )
    d = os.path.dirname(_ENV_LOCAL_FILE) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".env.local.", suffix=".tmp")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body)
        os.replace(tmp, _ENV_LOCAL_FILE)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def update_config(updates: dict[str, object]) -> None:
    """Persist non-secret runtime settings. Rejects unknown keys and bad types.

    Type-validates by constructing a real ``Settings`` (init kwargs take priority
    and are validated); the rest of the model loads normally, so we only read
    back the fields we set. Clearing ``embed_url`` (None/"") removes the override.
    """
    if not updates:
        return
    unknown = set(updates) - WRITABLE_FIELDS
    if unknown:
        raise ConfigWriteError(f"unknown or read-only setting(s): {', '.join(sorted(unknown))}")
    try:
        validated = Settings(**updates)  # type: ignore[arg-type]
    except ValidationError as e:
        raise ConfigWriteError(_short_validation(e)) from e

    env = _read_env_local()
    for field in updates:
        env_key = f"{_PREFIX}{field.upper()}"
        val = getattr(validated, field)
        if val is None or val == "":
            env.pop(env_key, None)  # unset -> fall back to .env / default
        else:
            env[env_key] = _envstr(val)
    _write_env_local(env)


def set_secret(name: str, value: str) -> None:
    """Persist a provider API key to ``.env.local`` (0600). Never echoed back."""
    env_key = SECRET_ENV.get(name)
    if env_key is None:
        raise ConfigWriteError(f"unknown provider key: {name}")
    value = (value or "").strip()
    if not value:
        raise ConfigWriteError("key value is empty")
    env = _read_env_local()
    env[env_key] = _envstr(value)  # rejects newlines (injection guard)
    _write_env_local(env)


def clear_secret(name: str) -> None:
    """Remove a stored provider API key."""
    env_key = SECRET_ENV.get(name)
    if env_key is None:
        raise ConfigWriteError(f"unknown provider key: {name}")
    env = _read_env_local()
    if env.pop(env_key, None) is not None:
        _write_env_local(env)


def secret_status() -> dict[str, bool]:
    """Which provider keys are currently set (booleans only — never the value)."""
    s = load_settings()
    return {
        "cerebras": bool(s.cerebras_api_key),
        "groq": bool(s.groq_api_key),
    }


def _short_validation(e: ValidationError) -> str:
    parts = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err.get("loc", ())) or "value"
        parts.append(f"{loc}: {err.get('msg', 'invalid')}")
    return "; ".join(parts) or "invalid settings"
