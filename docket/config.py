"""Runtime configuration. Every field is env-overridable with the ``DK_`` prefix
(e.g. ``DK_BACKEND_URL``, ``DK_PROVIDER``, ``DK_CHAT_MODEL``) or via a local
``.env`` file.

The local backend is anything that speaks the OpenAI-compatible ``/v1`` API:
the stock llama.cpp ``llama-server``, or Ollama. That single HTTP contract is the whole backend boundary.
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
    lite = "lite"          # embedded, zero-config, any-PDF (simple local install)
    platform = "platform"  # live Kafka/Spark lakehouse ingestion (self-host IaC)


class Provider(str, Enum):
    local = "local"        # OpenAI-/v1 backend: llama-server | Ollama
    cerebras = "cerebras"  # free-tier API (~1M tok/day)
    groq = "groq"          # free-tier API (fast)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DK_", env_file=(_ENV_FILE, _ENV_LOCAL_FILE), extra="ignore"
    )

    profile: Profile = Profile.lite
    provider: Provider = Provider.local

    # OpenAI-compatible /v1 base URL for the local backend.
    backend_url: str = "http://127.0.0.1:11434/v1"
    chat_model: str = "gemma4:e2b"  # e2b (not e4b) so chat + embed-gemma + OCR
    #                                 co-reside in 6 GB VRAM on an all-in-one
    #                                 server; stock llama-server ignores this.

    # Embeddings need a DEDICATED model/endpoint — the shared chat `llama-server`
    # is NOT started with --embeddings (and a chat model's embeddings are poor
    # anyway). Default None => embeddings unconfigured and retrieval degrades to
    # sparse-only (graceful; keeps the backend boundary generic). Point this at a
    # separate `llama-server --embeddings` (e.g. EmbeddingGemma) via DK_EMBED_URL.
    # An all-in-one server serves chat+embeddings on one /v1 base — there set
    # DK_EMBED_URL = DK_BACKEND_URL to turn dense retrieval on.
    embed_url: str | None = None
    embed_model: str = "embed-gemma:latest"

    # Cross-encoder reranker over the fused candidates (the dominant FinanceBench
    # lever — Phase-A sweep: recall@6 0.840 vs 0.527 unreranked, n=150; T26). Same backend
    # boundary as chat/embed: a POST to the OpenAI-style ``/v1/rerank`` on an
    # all-in-one server (Cohere shape). Default None => no reranker and the
    # fused RRF order stands (graceful, offline-safe). Point DK_RERANK_URL at the
    # ``/v1`` base (e.g. = DK_BACKEND_URL) to turn reranking on; the seam only
    # uses rank ORDER, never an absolute score, so any served reranker fits.
    rerank_url: str | None = None
    rerank_model: str = "qwen3-reranker-0.6b:latest"

    # BYO keys for the free-tier API providers.
    cerebras_api_key: str | None = None
    groq_api_key: str | None = None

    request_timeout_s: float = 120.0

    # Lite-profile ingest/retrieval knobs. The persisted corpus (chunks + optional
    # dense vectors) lives under index_dir; zero-config, embedded.
    index_dir: str = _DEFAULT_INDEX_DIR
    chunk_words: int = 220        # target words per chunk
    chunk_overlap: int = 40       # word overlap between adjacent chunks
    retrieval_k: int = 20         # candidates pulled before rerank
    context_chunks: int = 10      # top chunks fed to the model (FinanceBench: 82.0% @10 vs 68.0% @6)
    max_hops: int = 2             # iterative agentic re-query budget

def load_settings() -> Settings:
    return Settings()
