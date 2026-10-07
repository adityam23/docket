"""Runtime configuration. Every field is env-overridable with the ``DK_`` prefix
(e.g. ``DK_BACKEND_URL``, ``DK_ENGINE_BINARY``, ``DK_CHAT_MODEL``) or via a local
``.env`` file.

The engine is mettle — the only supported backend, via its OpenAI-compatible
``/v1`` plus the collection/management surface. That single HTTP contract is
the whole backend boundary (cutover contract, 2026-09-10).
"""

from __future__ import annotations

import os
import re
import sys
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Mettle's own collection-name contract (adoption contract, 2026-09-10): alnum
# plus ``. _ -``, 1-128 chars, never leading with a dot. Validated here too so
# a bad name is rejected at the config boundary instead of 404ing against the
# engine on the next ask. Written without look-around: pydantic-core's regex
# engine (Rust `regex` crate) does not support ``(?!...)`` — so "no leading
# dot" is instead expressed by excluding ``.`` from the first character's own
# class, with the full ``. _ -`` set allowed only from the second character on.
_COLLECTION_NAME_PATTERN = r"^[A-Za-z0-9_-][A-Za-z0-9._-]{0,127}$"


def _project_root() -> Path | None:
    """The directory that owns this checkout (holds ``pyproject.toml``).

    Config must not depend on the process working directory: ``uv run dk serve``,
    the installed ``dk`` console script, and ``uvicorn --reload`` can each start
    with a different CWD, and a CWD-relative ``.env`` / ``.docket_index`` silently
    vanishes there (embeddings read as "unconfigured", corpus reads as empty).
    Anchoring to the checkout root keeps a single source of truth. Returns None
    for an installed (wheel) copy — then state falls back to XDG dirs, which
    env vars can still override.
    """
    if getattr(sys, "frozen", False):
        # A PyInstaller build's __file__ lives under sys._MEIPASS, not the
        # checkout. If the frozen binary happens to run from inside the repo's
        # own dist/ dir, the parent walk below would still find the repo's
        # pyproject.toml and silently adopt the developer's .env/.env.local/
        # .docket_index — a packaged app must never see checkout-local state,
        # so a frozen process always falls back to XDG dirs.
        return None
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return None


_ROOT = _project_root()
# Installed runs (uv tool / pip wheel) have no checkout root; fall back to XDG
# dirs so state lands in stable per-user locations instead of the CWD du jour.
_DATA_DIR = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "docket"
_CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "docket"
_ENV_FILE = str(_ROOT / ".env") if _ROOT else str(_CONFIG_DIR / "env")
# `.env.local` is the single machine-local override the UI writes to (Settings
# page + BYOK keys). It layers *over* `.env`: pydantic-settings loads the files
# in order and later files win, so a value set from the dashboard takes effect on
# the next `load_settings()` without touching the hand-authored, comment-rich
# `.env`. It is gitignored and written 0600 (it may hold API keys). One writer
# for both knobs and secrets lives in `config_write.py` (CLAUDE.md: one impl).
_ENV_LOCAL_FILE = str(_ROOT / ".env.local") if _ROOT else str(_CONFIG_DIR / "env.local")
_DEFAULT_INDEX_DIR = str(_ROOT / ".docket_index") if _ROOT else str(_DATA_DIR / "index")

class Profile(str, Enum):
    lite = "lite"          # embedded, any-PDF, engine-backed
    platform = "platform"  # live Kafka/Spark lakehouse ingestion (self-host IaC)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DK_", env_file=(_ENV_FILE, _ENV_LOCAL_FILE), extra="ignore"
    )

    profile: Profile = Profile.lite
    # OpenAI-/v1 base URL of the engine (mettle). The ONLY supported backend:
    # chat, embeddings, rerank and collection search all ride this one base.
    backend_url: str = "http://127.0.0.1:11434/v1"
    chat_model: str = "qwen35-4b-opus-distill"  # the measured FinanceBench
    #   winner (82.0% full-RAG @ 10 chunks) and the `chat` profile's only
    #   slot (residency contract D5) -- chat and embed/rerank no longer need
    #   to co-reside: mettle's 6 GB config switches between a `retrieval`
    #   profile (embed + rerank) and this `chat` profile per phase, see
    #   engine_retrieval_profile/engine_chat_profile below and docket/phase.py.

    # Embeddings and rerank are UNCONDITIONAL engine features — mettle serves
    # them on the same /v1 base as chat. Only the model aliases are tunable;
    # qwen3 is the shipped engine default (adoption contract, 2026-09-10).
    embed_model: str = "qwen3-embedding-0.6b:latest"
    rerank_model: str = "qwen3-reranker-0.6b:latest"

    # Bearer for the engine (both /v1 and /engine/*). Set only when the engine
    # enforces auth; unset => no auth header (the default localhost dev case).
    engine_key: str | None = None

    # Engine process supervision (§3 of the cutover contract): docket's backend
    # owns mettled's lifecycle. Binary path unset => never spawn, only probe.
    engine_binary: str | None = None   # e.g. ~/.cargo/bin/mettled
    mettle_config: str | None = None   # METTLE_CONFIG override (config.toml)
    # How long a spawned-but-not-yet-answering engine may stay alive before we
    # treat it as hung (kill the group, count the attempt, respawn). A first
    # run legitimately downloads GB-scale weights for minutes, hence the long
    # default — this ceiling exists so a binary that NEVER answers can't hold
    # "starting" forever.
    engine_startup_window_s: float = Field(900.0, gt=0)

    # Engine phase coordination (residency contract D5): mettle's 6 GB config
    # can't keep embed+rerank and the chat generator resident together, so it
    # ships two profiles and docket switches between them per phase via the
    # ONE coordinator (docket/phase.py), applied per-call by every provider
    # built through providers/router.py. Empty ("") for EITHER name disables
    # switching entirely -- a single-profile engine, coordinator is a no-op.
    # Validated with the same pattern mettle enforces for collection names
    # (empty is the one exception). Not web-writable: an engine-topology
    # decision, not a Settings-page knob (config_write.WRITABLE_FIELDS).
    engine_retrieval_profile: str = "retrieval"
    engine_chat_profile: str = "chat"
    # How long a phase switch (PUT + poll /engine/ready + verify /engine/status
    # active/!switching) may take before the coordinator gives up and raises
    # loudly (docket/engine.py EngineSwitchError) instead of hanging forever.
    # Measured switch cost on the reference card is ~2 s; this is a generous
    # ceiling, not the expected latency.
    engine_switch_timeout_s: float = Field(120.0, gt=0)

    request_timeout_s: float = Field(120.0, gt=0)

    # Ingest/retrieval knobs. The persisted corpus (chunk lineage + doc dedup +
    # job state, NO chunk text — display text comes back from engine hits)
    # lives under index_dir; zero-config, embedded.
    index_dir: str = _DEFAULT_INDEX_DIR
    chunk_words: int = Field(220, ge=1)   # target words per chunk (frozen parity
    #   substrate — env-only developer config, not web-writable)
    chunk_overlap: int = Field(40, ge=0)  # word overlap between adjacent chunks;
    #   must be < chunk_words (see the model validator below)
    retrieval_mode: Literal["dense", "keyword", "hybrid"] = "hybrid"  # engine modes
    engine_collection: str = Field("docket", pattern=_COLLECTION_NAME_PATTERN)
    #   mettle collection docket ingests into
    retrieval_k: int = Field(40, ge=1)    # candidates pulled before rerank
    context_chunks: int = Field(10, ge=1)  # top chunks fed to the model
    #   (FinanceBench: 82.0% @10 vs 68.0% @6)
    max_hops: int = Field(2, ge=1)        # iterative agentic re-query budget

    @model_validator(mode="after")
    def _validate_chunk_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_words:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_words ({self.chunk_words})"
            )
        return self

    @model_validator(mode="after")
    def _validate_engine_phase_profiles(self) -> "Settings":
        """Same name rule as ``engine_collection`` (mettle's collection-name
        contract), with one exception: empty is the documented opt-out that
        disables phase switching (docket/phase.py), so it can't share the
        field's own ``pattern=`` (which requires 1-128 chars)."""
        for field in ("engine_retrieval_profile", "engine_chat_profile"):
            value = getattr(self, field)
            if value and not re.fullmatch(_COLLECTION_NAME_PATTERN, value):
                raise ValueError(
                    f"{field} ({value!r}) must match {_COLLECTION_NAME_PATTERN!r} "
                    "or be empty (disables phase switching)"
                )
        return self


def load_settings() -> Settings:
    return Settings()
