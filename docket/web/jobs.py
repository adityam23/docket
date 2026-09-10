"""In-memory ingest-job registry for the web dashboard.

Ingestion is otherwise a blocking CLI call; the dashboard needs to kick it off
and *watch* each document walk ocr → chunk → embed → index. This runs one job at
a time on a background thread and records per-document stage transitions, which
the frontend polls via ``GET /api/ingest/status``.

Single-user, local, one process — so a process-local
registry is correct; no broker or DB. It reuses ``ingest_paths`` verbatim via its
``on_event`` hook, so there is exactly one ingest implementation.
"""

from __future__ import annotations

import os
import shutil
import threading
from dataclasses import dataclass, field

from ..config import Settings, load_settings
from ..ingest.index import Corpus
from ..ingest.pipeline import ingest_paths


@dataclass
class DocProgress:
    doc_id: str
    source: str
    stage: str = "queued"   # queued|ocr|chunk|embed|index|done|skipped|error
    pages: int = 0
    chunks: int = 0
    vectorized: bool = False
    error: str = ""

    def as_dict(self) -> dict:
        return self.__dict__


@dataclass
class Job:
    id: str
    folder: str
    status: str = "running"          # running|done|error
    started_monotonic: float = 0.0
    error: str = ""
    docs: list[DocProgress] = field(default_factory=list)
    _index: dict[str, DocProgress] = field(default_factory=dict, repr=False)

    def doc(self, doc_id: str, source: str = "") -> DocProgress:
        dp = self._index.get(doc_id)
        if dp is None:
            dp = DocProgress(doc_id=doc_id, source=source)
            self._index[doc_id] = dp
            self.docs.append(dp)
        return dp

    def as_dict(self) -> dict:
        done = sum(1 for d in self.docs if d.stage in ("done", "skipped", "error"))
        return {
            "id": self.id,
            "folder": self.folder,
            "status": self.status,
            "error": self.error,
            "total": len(self.docs),
            "completed": done,
            "docs": [d.as_dict() for d in self.docs],
        }


def _discover_pdfs(folder: str) -> list[str]:
    return sorted(
        os.path.join(root, f)
        for root, _, files in os.walk(folder)
        for f in files
        if f.lower().endswith(".pdf")
    )


class IngestJobs:
    """Serialises ingest runs (one worker) and exposes their live state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._worker: threading.Thread | None = None
        self._seq = 0

    def latest(self) -> dict | None:
        with self._lock:
            if not self._order:
                return None
            return self._jobs[self._order[-1]].as_dict()

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.as_dict() if job else None

    def is_running(self) -> bool:
        return bool(self._worker and self._worker.is_alive())

    def start(self, folder: str, *, settings: Settings | None = None) -> Job:
        """Validate the folder, register a job, and run it on a worker thread."""
        folder = os.path.abspath(os.path.expanduser(folder))
        if not os.path.isdir(folder):
            raise ValueError(f"not a directory: {folder}")
        pdfs = _discover_pdfs(folder)
        if not pdfs:
            raise ValueError(f"no PDF files found under {folder}")
        return self._launch(folder, pdfs, settings=settings)

    def start_uploads(
        self, files: list[tuple[str, bytes]], *, settings: Settings | None = None
    ) -> Job:
        """Ingest browser-selected PDFs (name, bytes) chosen via a folder picker.

        The bytes are written to a throwaway temp dir under ``index_dir`` and
        ingested by the SAME worker as the folder path; each chunk cites its
        original filename (not the temp path), and the temp copies are deleted
        once the job settles — the extracted text persists only in the index.
        """
        if not files:
            raise ValueError("no files were provided")
        s = settings or load_settings()
        with self._lock:
            seq = self._seq + 1  # tentative; _launch increments authoritatively
        upload_dir = os.path.join(s.index_dir, ".uploads", f"job-{seq}")
        os.makedirs(upload_dir, exist_ok=True)
        paths: list[str] = []
        display: dict[str, str] = {}
        for name, data in files:
            safe = os.path.basename(name)  # never trust the client filename
            if not safe.lower().endswith(".pdf"):
                continue
            path = os.path.join(upload_dir, safe)
            with open(path, "wb") as f:
                f.write(data)
            paths.append(path)
            display[path] = safe
        if not paths:
            shutil.rmtree(upload_dir, ignore_errors=True)
            raise ValueError("no PDF files among the selected items")
        return self._launch(
            f"{len(paths)} uploaded file(s)", paths, settings=s,
            source_of=display.get, cleanup_dir=upload_dir,
        )

    def _launch(
        self,
        folder: str,
        pdfs: list[str],
        *,
        settings: Settings | None = None,
        source_of=None,
        cleanup_dir: str | None = None,
    ) -> Job:
        """Register a job over already-resolved paths and run it on the worker."""
        if self.is_running():
            if cleanup_dir:
                shutil.rmtree(cleanup_dir, ignore_errors=True)
            raise RuntimeError("an ingest job is already running")
        s = settings or load_settings()
        with self._lock:
            self._seq += 1
            job = Job(id=f"job-{self._seq}", folder=folder)
            for p in pdfs:
                job.doc(os.path.splitext(os.path.basename(p))[0],
                        source=source_of(p) if source_of else p)
            self._jobs[job.id] = job
            self._order.append(job.id)

        self._worker = threading.Thread(
            target=self._run, args=(job, pdfs, s),
            kwargs={"source_of": source_of, "cleanup_dir": cleanup_dir},
            name=f"ingest-{job.id}", daemon=True,
        )
        self._worker.start()
        return job

    def _run(self, job: Job, pdfs: list[str], s: Settings, *,
             source_of=None, cleanup_dir: str | None = None) -> None:
        def on_event(doc_id: str, stage: str, info: dict) -> None:
            with self._lock:
                dp = job.doc(doc_id)
                dp.stage = stage
                if "pages" in info:
                    dp.pages = int(info["pages"])
                if "chunks" in info:
                    dp.chunks = int(info["chunks"])
                if "vectorized" in info:
                    dp.vectorized = bool(info["vectorized"])
                if stage == "skipped":
                    dp.error = str(info.get("reason", ""))

        try:
            corpus = Corpus.load(s.index_dir)          # incremental extend
            ingest_paths(pdfs, settings=s, corpus=corpus,
                         on_event=on_event, source_of=source_of)
            corpus.save(s.index_dir)
            with self._lock:
                job.status = "done"
        except Exception as e:  # noqa: BLE001 - surface failure as job state
            with self._lock:
                job.status = "error"
                job.error = str(e)
        finally:
            if cleanup_dir:  # drop temp upload copies; index already persisted
                shutil.rmtree(cleanup_dir, ignore_errors=True)


# Process-wide singleton (one ingest worker per server process).
JOBS = IngestJobs()
