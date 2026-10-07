"""docket: a self-hostable finance-filings research assistant over SEC EDGAR —
live-ingested, cited, confidence-scored answers across an analyst's whole
coverage universe, with a trust layer that flags when it's guessing. The
retrieval/agent/eval tooling is what makes a small, local model good enough;
the model is small because white-box interpretability is impossible over a
hosted API and the target hardware is 6 GB VRAM, not because small is meant
to rival frontier models.

"""

__version__ = "0.0.1"
