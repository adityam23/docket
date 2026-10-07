"""In-memory ingest-job registry for the web dashboard.

Ingestion is otherwise a blocking CLI call; the dashboard needs to kick it off
and *watch* each document walk ``ocr → chunk → post → done/skipped/failed``.
This runs one job at a time on a background thread and records per-document
stage transitions, which the frontend polls via ``GET /api/ingest/status``.

Single-user, local, one process — so a process-local
registry is correct; no broker or DB. It reuses ``pipeline.ingest_into_index``
verbatim via its ``on_event`` hook, so there is exactly one ingest
implementation.
"""

from __future__ import annotations

import os
import shutil
import threading
from dataclasses import dataclass, field

from ..config import Settings, load_settings
from ..ingest import pipeline as pipe


@dataclass
class DocProgress:
    doc_id: str
    source: str
    stage: str = "queued"   # queued|ocr|chunk|post|done|skipped|failed
    pages: int = 0
    chunks: int = 0
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
        done = sum(1 for d in self.docs if d.stage in ("done", "skipped", "error", "failed"))
        failed = sum(1 for d in self.docs if d.stage == "failed")
        return {
            "id": self.id,
            "folder": self.folder,
            "status": self.status,
            "error": self.error,
            "total": len(self.docs),
            "completed": done,
            "failed": failed,
            "docs": [d.as_dict() for d in self.docs],
        }


def _dedupe_name(name: str, used: set[str]) -> str:
    """Give an uploaded file a disk-safe, collision-free basename within its
    job's upload dir: two picked files sharing a basename (e.g. two folders
    each containing "report.pdf") must not overwrite each other on disk — and
    since the doc id is the filename stem, they genuinely ARE two different
    documents and should end up with two different ids."""
    stem, ext = os.path.splitext(name)
    candidate = name
    i = 1
    while candidate in used:
        candidate = f"{stem}_{i}{ext}"
        i += 1
    used.add(candidate)
    return candidate


def sweep_stale_uploads(settings: Settings | None = None) -> None:
    """Delete any ``.uploads/job-N`` temp dirs left behind by a process that
    died mid-ingest (B4): normally cleaned up in ``_run``'s ``finally``, but a
    crash before that point leaves the decoded PDF bytes on disk forever
    otherwise. Safe to call on every startup — this process's job registry
    always starts empty, so there is never a live job this early."""
    s = settings or load_settings()
    uploads_root = os.path.join(s.index_dir, ".uploads")
    if os.path.isdir(uploads_root):
        shutil.rmtree(uploads_root, ignore_errors=True)


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

    def is_running(self) -> bool:
        return bool(self._worker and self._worker.is_alive())

    def _next_seq(self) -> int:
        """Atomically reserve the next job/upload-dir number. Uploads need
        this BEFORE ``_launch`` runs (to name their temp dir); peeking
        ``self._seq`` without reserving it let two concurrent uploads compute
        the SAME number and share (and ``rmtree``-race) one directory."""
        with self._lock:
            self._seq += 1
            return self._seq

    def start(self, folder: str, *, settings: Settings | None = None) -> Job:
        """Validate the folder, register a job, and run it on a worker thread."""
        folder = os.path.abspath(os.path.expanduser(folder))
        if not os.path.isdir(folder):
            raise ValueError(f"not a directory: {folder}")
        pdfs = pipe.discover_pdfs(folder)
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
        Two picked files sharing a basename get disambiguated on disk (and so
        get distinct doc ids) rather than overwriting each other.
        """
        if not files:
            raise ValueError("no files were provided")
        s = settings or load_settings()
        seq = self._next_seq()  # reserved atomically; also this job's id
        job_id = f"job-{seq}"
        upload_dir = os.path.join(s.index_dir, ".uploads", job_id)
        os.makedirs(upload_dir, exist_ok=True)
        paths: list[str] = []
        display: dict[str, str] = {}
        used_names: set[str] = set()
        for name, data in files:
            safe = os.path.basename(name)  # never trust the client filename
            if not safe.lower().endswith(".pdf"):
                continue
            safe = _dedupe_name(safe, used_names)
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
            source_of=display.get, cleanup_dir=upload_dir, job_id=job_id,
        )

    def _launch(
        self,
        folder: str,
        pdfs: list[str],
        *,
        settings: Settings | None = None,
        source_of=None,
        cleanup_dir: str | None = None,
        job_id: str | None = None,
    ) -> Job:
        """Register a job over already-resolved paths and run it on the worker.

        Acquires ``pipeline.CORPUS_LOCK`` for the WHOLE run — released by
        ``_run``'s ``finally``, not here — so this job's load→ingest→save never
        interleaves with another ingest job, a samples load, or a document
        removal (B4). The previous ``is_running()`` pre-check happened BEFORE
        this method claimed anything, so two concurrent callers could both see
        "idle" and both launch.
        """
        if not pipe.CORPUS_LOCK.acquire(blocking=False):
            if cleanup_dir:
                shutil.rmtree(cleanup_dir, ignore_errors=True)
            raise RuntimeError("an ingest job is already running")
        try:
            s = settings or load_settings()
            jid = job_id or f"job-{self._next_seq()}"
            with self._lock:
                job = Job(id=jid, folder=folder)
                for p in pdfs:
                    job.doc(pipe.doc_id_of(p), source=source_of(p) if source_of else p)
                self._jobs[job.id] = job
                self._order.append(job.id)

            self._worker = threading.Thread(
                target=self._run, args=(job, pdfs, s),
                kwargs={"source_of": source_of, "cleanup_dir": cleanup_dir},
                name=f"ingest-{job.id}", daemon=True,
            )
            self._worker.start()
        except Exception:
            pipe.CORPUS_LOCK.release()
            raise
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
                if stage in ("skipped", "failed"):
                    dp.error = str(info.get("reason", ""))

        try:
            pipe.ingest_into_index(pdfs, settings=s, on_event=on_event, source_of=source_of)
            with self._lock:
                job.status = "done"
        except Exception as e:  # noqa: BLE001 - surface failure as job state
            with self._lock:
                job.status = "error"
                job.error = str(e)
        finally:
            if cleanup_dir:  # drop temp upload copies; index already persisted
                shutil.rmtree(cleanup_dir, ignore_errors=True)
            pipe.CORPUS_LOCK.release()


# Process-wide singleton (one ingest worker per server process).
JOBS = IngestJobs()
