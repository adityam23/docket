"""Corpus store for the lite profile: chunks + optional dense vectors, with
brute-force cosine search and zero-config JSON persistence.

Design (docs/decisions Q13, docs/stack.md):
- **Lite** is embedded and honest about scale — an any-PDF folder is small, so a
  dependency-free brute-force cosine over stored vectors is correct and keeps
  `uv sync` tiny. `turbovec` (TurboQuant) is a drop-in dense backend for larger
  corpora, loaded lazily behind the same `dense_search` seam when installed.
- Metadata/lineage lives beside the vectors; DuckDB (the `platform` path) can
  replace the JSON store later without touching callers.

One concept, one implementation: this is the *only* corpus store; retrieval and
the agent tools read chunks through it.
"""

from __future__ import annotations

import json
import math
import os

from .chunk import Chunk


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class Corpus:
    """Chunks + parallel optional dense vectors, addressable by ``chunk_id``."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.vectors: list[list[float] | None] = []
        self._by_id: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], vectors: list[list[float]] | None = None) -> None:
        if vectors is not None and len(vectors) != len(chunks):
            raise ValueError("vectors length must match chunks length")
        for i, ch in enumerate(chunks):
            if ch.chunk_id in self._by_id:
                continue  # idempotent re-ingest
            self._by_id[ch.chunk_id] = len(self.chunks)
            self.chunks.append(ch)
            self.vectors.append(vectors[i] if vectors is not None else None)

    def get(self, chunk_id: str) -> Chunk:
        return self.chunks[self._by_id[chunk_id]]

    def doc_ids(self) -> list[str]:
        """Distinct document ids in stable first-seen order."""
        seen: dict[str, None] = {}
        for ch in self.chunks:
            seen.setdefault(ch.doc_id, None)
        return list(seen)

    def remove(self, doc_id: str) -> int:
        """Drop every chunk (and its vector) belonging to ``doc_id``.

        Returns the number of chunks removed. Rebuilds the id index; O(n), which
        is correct for the small lite corpus. The inverse of ``add`` — the one
        place a document leaves the index (docs/decisions Q13; CLAUDE.md reuse).
        """
        keep_chunks: list[Chunk] = []
        keep_vecs: list[list[float] | None] = []
        removed = 0
        for ch, vec in zip(self.chunks, self.vectors):
            if ch.doc_id == doc_id:
                removed += 1
                continue
            keep_chunks.append(ch)
            keep_vecs.append(vec)
        if removed:
            self.chunks = keep_chunks
            self.vectors = keep_vecs
            self._by_id = {ch.chunk_id: i for i, ch in enumerate(self.chunks)}
        return removed

    @property
    def has_vectors(self) -> bool:
        return any(v is not None for v in self.vectors)

    def dense_search(self, query_vec: list[float], k: int = 10) -> list[tuple[str, float]]:
        """Cosine over stored vectors → ``[(chunk_id, score), ...]`` desc."""
        scored = [
            (self.chunks[i].chunk_id, _cosine(query_vec, v))
            for i, v in enumerate(self.vectors)
            if v is not None
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]

    # --- persistence (zero-config JSON; DuckDB is the platform-profile swap) ---

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "corpus.jsonl"), "w") as f:
            for ch, vec in zip(self.chunks, self.vectors):
                f.write(json.dumps({**ch.as_dict(), "vector": vec}) + "\n")

    @classmethod
    def load(cls, path: str) -> "Corpus":
        c = cls()
        fp = os.path.join(path, "corpus.jsonl")
        if not os.path.exists(fp):
            return c
        chunks: list[Chunk] = []
        vecs: list[list[float] | None] = []
        with open(fp) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                vecs.append(row.pop("vector", None))
                chunks.append(Chunk(**row))
        c.add(chunks, None)
        c.vectors = vecs  # restore vectors positionally
        return c


# Backwards-compatible thin alias for the phase-0 placeholder name.
VectorIndex = Corpus
