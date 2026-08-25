"""Runtime configuration. Every field is env-overridable with the ``DK_`` prefix
(e.g. ``DK_BACKEND_URL``, ``DK_PROVIDER``, ``DK_CHAT_MODEL``) or via a local
``.env`` file.

The local backend is anything that speaks the OpenAI-compatible ``/v1`` API:
the stock llama.cpp ``llama-server`` (what runs on this box today), the infengine
Rust server, or Ollama. That single HTTP contract is the whole reconciliation —
see docs/architecture.md.
"""

from __future__ import annotations

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
    for a pip-installed package (no pyproject alongside the code) — then we fall
    back to CWD-relative defaults, which env vars can still override.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return None


_ROOT = _project_root()
_ENV_FILE = str(_ROOT / ".env") if _ROOT else ".env"
# `.env.local` is the single machine-local override the UI writes to (Settings
# page + BYOK keys). It layers *over* `.env`: pydantic-settings loads the files
# in order and later files win, so a value set from the dashboard takes effect on
# the next `load_settings()` without touching the hand-authored, comment-rich
# `.env`. It is gitignored and written 0600 (it may hold API keys). One writer
# for both knobs and secrets lives in `config_write.py` (CLAUDE.md: one impl).
_ENV_LOCAL_FILE = str(_ROOT / ".env.local") if _ROOT else ".env.local"
_DEFAULT_INDEX_DIR = str(_ROOT / ".docket_index") if _ROOT else ".docket_index"


class Profile(str, Enum):
    lite = "lite"          # embedded, zero-config, any-PDF (simple local install)
    platform = "platform"  # live Kafka/Spark lakehouse ingestion (self-host IaC)


class Provider(str, Enum):
    local = "local"        # OpenAI-/v1 backend: llama-server | infengine | Ollama
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
    #                                 infengine; stock llama-server ignores this.

    # Embeddings need a DEDICATED model/endpoint — the shared chat `llama-server`
    # is NOT started with --embeddings (and a chat model's embeddings are poor
    # anyway). Default None => embeddings unconfigured and retrieval degrades to
    # sparse-only (graceful; keeps the backend boundary generic). Point this at a
    # separate `llama-server --embeddings` (e.g. EmbeddingGemma) via DK_EMBED_URL.
    # An all-in-one infengine serves chat+embeddings on one /v1 base — there set
    # DK_EMBED_URL = DK_BACKEND_URL to turn dense retrieval on.
    embed_url: str | None = None
    embed_model: str = "embed-gemma:latest"

    # Cross-encoder reranker over the fused candidates (the dominant FinanceBench
    # lever — perfect retrieval ≈ 89% vs basic RAG ≈ 19%; T26). Same backend
    # boundary as chat/embed: a POST to the OpenAI-style ``/v1/rerank`` on an
    # all-in-one infengine (Cohere shape). Default None => no reranker and the
    # fused RRF order stands (graceful, offline-safe). Point DK_RERANK_URL at the
    # ``/v1`` base (e.g. = DK_BACKEND_URL) to turn reranking on; the seam only
    # uses rank ORDER, never an absolute score, so any served reranker fits.
    rerank_url: str | None = None
    rerank_model: str = "bge-reranker-v2-m3:latest"

    # BYO keys for the free-tier API providers.
    cerebras_api_key: str | None = None
    groq_api_key: str | None = None

    request_timeout_s: float = 120.0

    # Lite-profile ingest/retrieval knobs. The persisted corpus (chunks + optional
    # dense vectors) lives under index_dir; zero-config, embedded (docs Q13).
    index_dir: str = _DEFAULT_INDEX_DIR
    chunk_words: int = 220        # target words per chunk
    chunk_overlap: int = 40       # word overlap between adjacent chunks
    retrieval_k: int = 20         # candidates pulled before rerank
    context_chunks: int = 6       # top chunks fed to the model as grounded context
    max_hops: int = 2             # iterative agentic re-query budget (docs Q8)


def load_settings() -> Settings:
    return Settings()
