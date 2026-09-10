"""Throwaway smoke test for the engine-supervision lifecycle (cutover contract
§3) — run manually, not part of pytest. Exercises the REAL supervisor loop
against a real fake engine process:

1. engine answering /engine/status, binary unset   -> probe-only: available
2. engine dies, binary unset                       -> offline (loud, no spawn)
3. binary configured, engine dies mid-run          -> respawn + recover

Run: uv run python benchmarks/smoke_supervision.py
"""
import http.server
import os
import signal
import socket
import tempfile
import textwrap
import threading
import time

_here = os.path.dirname(os.path.abspath(__file__))
sys_path = os.path.join(_here, "..")
import sys  # noqa: E402

sys.path.insert(0, sys_path)

from docket.config import Settings  # noqa: E402
from docket.engine import EngineSupervisor  # noqa: E402


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _StatusHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # silence
        pass


def fake_engine(port: int) -> http.server.ThreadingHTTPServer:
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), _StatusHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def wait_state(sup: EngineSupervisor, want: str, timeout: float = 45.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if sup.view()["state"] == want:
            return True
        time.sleep(0.2)
    return False


def main() -> int:
    failures = []

    # --- 1. probe-only available ------------------------------------------------
    port = free_port()
    srv = fake_engine(port)
    sup = EngineSupervisor(Settings(backend_url=f"http://127.0.0.1:{port}/v1", engine_binary=None))
    sup.ensure()
    if wait_state(sup, "available"):
        print("[ok] 1: probe-only -> available")
    else:
        failures.append(f"1: state={sup.view()}")
        print("[FAIL] 1: probe-only never became available")

    # --- 2. engine dies, binary unset -> offline, no spawn ----------------------
    srv.shutdown()
    srv.server_close()
    if wait_state(sup, "offline", timeout=15.0):
        v = sup.view()
        assert v["pid"] is None, "probe-only must never spawn"
        print("[ok] 2: engine died -> offline, no spawn (loud)")
    else:
        failures.append(f"2: state={sup.view()}")
        print("[FAIL] 2: never went offline after death")
    sup.stop()

    # --- 3. configured binary: spawn, crash, respawn -----------------------------
    port = free_port()
    tmp = tempfile.mkdtemp()
    engine_py = os.path.join(tmp, "fake_engine.py")
    fake = os.path.join(tmp, "fake-mettled")
    with open(engine_py, "w") as f:
        f.write(textwrap.dedent(f"""\
            import http.server, threading, time

            class H(http.server.BaseHTTPRequestHandler):
                def do_GET(self):
                    body = b"ok"
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def log_message(self, *a):
                    pass

            srv = http.server.ThreadingHTTPServer(("127.0.0.1", {port}), H)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            time.sleep(3600)
            """))
    # SINGLE PROCESS with the project venv interpreter as shebang: mirrors the
    # real shape (one mettled binary, supervisor must notice THE SPAWNED PID
    # dying) and sidesteps `uv run`'s cwd sensitivity — the supervisor runs
    # children with cwd=dirname(binary). sys.executable here is the uv venv's
    # python, so invoking python stays uv-managed per the project rule.
    with open(fake, "w") as f:
        f.write(f"#!{sys.executable}\n{open(engine_py).read()}")
    os.chmod(fake, 0o755)
    sup2 = EngineSupervisor(Settings(backend_url=f"http://127.0.0.1:{port}/v1", engine_binary=fake))
    sup2.ensure()
    if wait_state(sup2, "available"):
        pid1 = sup2.view()["pid"]
        print(f"[ok] 3a: spawned fake mettled pid={pid1} -> available")
    else:
        failures.append("3a: spawn never became available")
        print("[FAIL] 3a:", sup2.view())

    # Crash it: kill THE SPAWNED PID (real mettled is a single process, and
    # the fake now is too — the supervisor must notice THE SPAWNED PID dying).
    # First wait for the supervisor to NOTICE (state leaves 'available'), then
    # wait for the respawn to become available again.
    try:
        os.kill(pid1, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    left_available = wait_state(sup2, "starting", timeout=15.0)
    if left_available and wait_state(sup2, "available", timeout=60.0):
        pid2 = sup2.view()["pid"]
        if pid2 != pid1:
            print(f"[ok] 3b: crash -> respawned pid={pid2} -> available")
        else:
            failures.append(f"3b: same pid {pid2} after crash (no respawn)")
            print("[FAIL] 3b: pid unchanged after crash")
    else:
        failures.append(f"3b: no recovery after crash; state={sup2.view()}")
        print("[FAIL] 3b:", sup2.view())
    # 4. stop() must take the whole engine process group with it — an orphaned
    # mettled holds its weights in VRAM and OOMs the next launch.
    pid_before = sup2.view()["pid"]
    alive_before = pid_before is not None and os.path.exists(f"/proc/{pid_before}")
    sup2.stop()
    time.sleep(1.0)
    still_alive = pid_before is not None and os.path.exists(f"/proc/{pid_before}")
    if alive_before and not still_alive:
        print(f"[ok] 4: stop() killed the spawned engine (pid {pid_before})")
    else:
        failures.append(f"4: engine still alive after stop() (pid={pid_before})")
        print("[FAIL] 4: orphaned engine after stop")

    port3 = free_port()
    slow = os.path.join(tempfile.mkdtemp(), "slow-mettled")
    with open(slow, "w") as f:
        f.write("#!/bin/sh\nsleep 60\n")  # alive, never answers the probe
    os.chmod(slow, 0o755)
    sup3 = EngineSupervisor(Settings(backend_url=f"http://127.0.0.1:{port3}/v1", engine_binary=slow))
    sup3.ensure()
    time.sleep(4.0)
    v = sup3.view()
    if v["state"] == "starting" and v["attempts"] == 1:
        print("[ok] 5: alive-but-slow child stays patient (attempts stays 1)")
    else:
        failures.append(f"5: budget consumed during patient start: {v}")
        print("[FAIL] 5:", v)
    sup3.stop()
    time.sleep(1.0)
    reaped = v["pid"] is None or not os.path.exists(f"/proc/{v['pid']}")
    if reaped:
        print("[ok] 5b: stop() reaped the slow engine")
    else:
        failures.append(f"5b: slow engine orphaned (pid={v['pid']})")
        print("[FAIL] 5b: slow engine still alive")

    print()
    if failures:
        print("FAILURES:", failures)
        return 1
    print("ALL SUPERVISION SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
