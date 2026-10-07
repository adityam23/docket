"""The ONE place runtime settings are persisted from the dashboard (Settings page
T01 + BYOK keys T02). Everything is written to ``.env.local`` — the machine-local
overlay that layers over ``.env`` (see ``config._ENV_LOCAL_FILE``) — so a change
takes effect on the next ``load_settings()`` and survives a restart, without ever
mutating the hand-authored, comment-rich ``.env``.

Security (CLAUDE.md — secrets are first-class):
- The file is written ``0600`` (it may hold API keys).
- Only an allowlist of keys is writable; unknown keys are rejected.
- Values are type-validated through the real ``Settings`` model before landing.
- Values may not contain newlines (no ``KEY=…\\nINJECTED=…`` line injection) or
  ``${`` (python-dotenv expands it unconditionally and it cannot be escaped).
- Every value is single-quoted on write (``_quote_env_value``) so python-dotenv
  (what pydantic-settings reads this file with) never treats it as a comment,
  a second quote style, or a shell-interpolated token.
- Secrets are never read back out — the read models keep redacting them to bools.

One writer, reused by the knobs endpoint and the keys endpoint (no second
``.env`` mutation implementation anywhere).
"""

from __future__ import annotations

from enum import Enum

from pydantic import ValidationError

from .atomic_write import atomic_write_text
from .config import _ENV_LOCAL_FILE, Settings, load_settings

# Non-secret runtime settings the Settings page may change. Names are the
# ``Settings`` field names; persisted as ``DK_<UPPER>`` with pydantic's prefix.
#
# ``embed_model``, ``chunk_words`` and ``chunk_overlap`` are deliberately NOT
# here even though they are ``Settings`` fields: ``chunk_words``/``chunk_overlap``
# (220/40) are a frozen parity substrate (changing them would silently shift
# every future ingest's chunk boundaries relative to past ones), and
# ``embed_model`` only affects the offline ``dk embed`` path, not anything the
# dashboard drives live. They stay env-only developer config.
WRITABLE_FIELDS: frozenset[str] = frozenset(
    {
        "backend_url",
        "chat_model",
        "rerank_model",
        "retrieval_mode",
        "engine_collection",
        "engine_binary",
        "mettle_config",
        "request_timeout_s",
        "retrieval_k",
        "context_chunks",
        "max_hops",
    }
)

# UI key name -> the env var it persists to. Post-cutover there is exactly one
SECRET_ENV: dict[str, str] = {
    "engine": "DK_ENGINE_KEY",
}
_PREFIX = "DK_"


class ConfigWriteError(ValueError):
    """A rejected write (unknown key, bad type, or unsafe value)."""


def _envstr(value: object) -> str:
    """Validate + stringify a Settings value to its RAW logical form (never
    quoted here — quoting is ``_write_env_local``'s one job, so there is a
    single place that knows the on-disk encoding).

    Two guards, both because this is what python-dotenv (what pydantic-settings
    reads ``.env.local`` with) does on the read side, verified against the
    installed parser:
    - no newlines: a bare ``\\n``/``\\r`` would start a second, attacker-chosen
      line (``KEY=a\\nINJECTED=1``) — classic line-injection guard.
    - no ``${``: python-dotenv POSIX-expands ``${NAME}`` / ``${NAME:-default}``
      UNCONDITIONALLY, even inside single quotes — quoting does not protect it
      (confirmed: ``'${HOME}'`` reads back as the *expanded* value, same as
      unquoted or double-quoted). There is no escape that survives it either
      (a backslash before ``${`` is left as a literal character, the expansion
      still fires). A value containing it would silently read back as something
      else entirely next launch, so it is rejected outright rather than quietly
      corrupted.
    """
    if isinstance(value, Enum):
        value = value.value
    s = str(value)
    if "\n" in s or "\r" in s:
        raise ConfigWriteError("values may not contain newlines")
    if "${" in s:
        raise ConfigWriteError(
            "values may not contain '${' — python-dotenv expands it unconditionally "
            "(even single-quoted) and it cannot be escaped"
        )
    return s


def _quote_env_value(value: str) -> str:
    """Single-quote ``value`` for a ``.env.local`` line, escaping it exactly as
    python-dotenv's single-quote decoder expects so the round trip is byte-exact.

    Verified against the installed parser (``dotenv/parser.py``): inside single
    quotes, only ``\\\\`` -> ``\\`` and ``\\'`` -> ``'`` are decoded (any other
    backslash sequence passes through literally) — so backslashes must be
    escaped FIRST, then quotes, or an escape introduced for a quote could be
    re-escaped by the backslash pass. Single-quoting (vs. double/unquoted) is
    what stops python-dotenv from treating a trailing `` #comment`` as a
    comment, from erroring on an embedded ``"``, and from re-interpreting ``=``
    or whitespace — the three failure modes this format replaces.
    """
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _unquote_env_value(raw: str) -> str:
    """Reverse of ``_quote_env_value`` — every line this module has ever
    written is single-quoted, so ``raw`` is always ``'...'``. Unescape in the
    opposite order to the writer (quotes, then backslashes). Falls back to the
    literal text for a line that somehow isn't quoted (defensive only; nothing
    in this module writes that)."""
    if len(raw) >= 2 and raw.startswith("'") and raw.endswith("'"):
        inner = raw[1:-1]
        return inner.replace("\\'", "'").replace("\\\\", "\\")
    return raw


def _read_env_local() -> dict[str, str]:
    """Parse the existing ``.env.local`` into ``{key: raw logical value}``
    (empty if absent). Deliberately minimal line scanning (we only ever wrote
    it ourselves) but DOES understand the single-quoted value format this
    module writes, via ``_unquote_env_value`` — so a value round-trips through
    read-modify-write (e.g. updating one field while carrying the rest forward)
    without ever double-quoting or corrupting an untouched key.
    """
    out: dict[str, str] = {}
    try:
        with open(_ENV_LOCAL_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                out[k.strip()] = _unquote_env_value(v.strip())
    except FileNotFoundError:
        pass
    return out


def _write_env_local(env: dict[str, str]) -> None:
    """Atomically rewrite ``.env.local`` from ``env`` (raw logical values —
    quoting happens here, the one place that applies it) with 0600 perms, via
    the shared ``atomic_write_text`` (a sibling temp file + ``os.replace``, so
    a crash mid-write never leaves a half-written secrets file, and it is
    created 0600 from the start — never a window where the key is
    world-readable).
    """
    body = (
        "# Machine-local overrides written by the dashboard (Settings + API keys).\n"
        "# Layers over .env; gitignored; do not commit. Edit via the app.\n"
        + "".join(f"{k}={_quote_env_value(v)}\n" for k, v in sorted(env.items()))
    )
    atomic_write_text(_ENV_LOCAL_FILE, body, mode=0o600)


def update_config(updates: dict[str, object]) -> None:
    """Persist non-secret runtime settings. Rejects unknown keys and bad types.

    Clearing a value (None/"") removes the override and falls back to .env /
    the field default — it is never itself type/pattern-validated (an empty
    string is the "unset" sentinel, not a candidate value for e.g.
    ``engine_collection``'s non-empty collection-name pattern). Every value
    actually being SET is type-validated by constructing a real ``Settings``
    (init kwargs take priority and are validated; the rest of the model loads
    normally, so only the fields being set need to round-trip through it).
    """
    if not updates:
        return
    unknown = set(updates) - WRITABLE_FIELDS
    if unknown:
        raise ConfigWriteError(f"unknown or read-only setting(s): {', '.join(sorted(unknown))}")

    clears = {field for field, val in updates.items() if val is None or val == ""}
    sets = {field: val for field, val in updates.items() if field not in clears}
    try:
        validated = Settings(**sets)  # type: ignore[arg-type]
    except ValidationError as e:
        raise ConfigWriteError(_short_validation(e)) from e

    env = _read_env_local()
    for field in clears:
        env.pop(f"{_PREFIX}{field.upper()}", None)
    for field in sets:
        env[f"{_PREFIX}{field.upper()}"] = _envstr(getattr(validated, field))
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
    return {"engine": bool(load_settings().engine_key)}

def _short_validation(e: ValidationError) -> str:
    parts = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err.get("loc", ())) or "value"
        parts.append(f"{loc}: {err.get('msg', 'invalid')}")
    return "; ".join(parts) or "invalid settings"
