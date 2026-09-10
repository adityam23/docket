"""Runtime configuration. Every field is env-overridable with the ``DK_`` prefix
(e.g. ``DK_BACKEND_URL``, ``DK_ENGINE_BINARY``, ``DK_CHAT_MODEL``) or via a local
``.env`` file.

The engine is mettle — the only supported backend, via its OpenAI-compatible
``/v1`` plus the collection/management surface. That single HTTP contract is
the whole backend boundary (cutover contract, 2026-09-10).
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


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
    chat_model: str = "gemma4:e2b"  # e2b (not e4b) so chat + embed + OCR
    #                                 co-reside in 6 GB VRAM on an all-in-one
    #                                 engine.

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
    engine_startup_window_s: float = 900.0

    request_timeout_s: float = 120.0

    # Ingest/retrieval knobs. The persisted corpus (chunk lineage + doc dedup +
    # job state, NO chunk text — display text comes back from engine hits)
    # lives under index_dir; zero-config, embedded.
    index_dir: str = _DEFAULT_INDEX_DIR
    chunk_words: int = 220        # target words per chunk
    chunk_overlap: int = 40       # word overlap between adjacent chunks
    retrieval_mode: str = "hybrid"  # dense | keyword | hybrid (engine modes)
    engine_collection: str = "docket"  # mettle collection docket ingests into
    retrieval_k: int = 20         # candidates pulled before rerank
    context_chunks: int = 10      # top chunks fed to the model (FinanceBench: 82.0% @10 vs 68.0% @6)
    max_hops: int = 2             # iterative agentic re-query budget

def load_settings() -> Settings:
    return Settings()
