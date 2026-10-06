"""Isolate every test from the developer's real environment: Settings must not
read the checkout's ``.env``/``.env.local`` (anchored there deliberately — see
``config._project_root`` — so a dev's ``DK_BACKEND_URL``/``DK_ENGINE_BINARY``
would otherwise leak in). Patching the module constants ``_ENV_FILE``/
``_ENV_LOCAL_FILE`` does NOT work: ``Settings.model_config["env_file"]`` bakes
their values into a tuple at class-definition time, so it must be patched
directly (verified manually: a temp ``.env`` with a bogus ``DK_CHAT_MODEL`` was
picked up by ``Settings()`` without this patch, and was not picked up with it).
"""

from __future__ import annotations

import os

import pytest

from docket.config import Settings


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    # No real env file loads during a test. A per-call override (e.g.
    # ``Settings(_env_file=...)``, used by test_config_write.py) still wins.
    monkeypatch.setitem(Settings.model_config, "env_file", ())

    for key in list(os.environ):  # clear any DK_* the dev's shell exports
        if key.startswith("DK_"):
            monkeypatch.delenv(key, raising=False)

    # A private index dir, not the real checkout's .docket_index.
    monkeypatch.setenv("DK_INDEX_DIR", str(tmp_path))
