"""Throwaway smoke test for the engine-supervision lifecycle (cutover contract
§3, residency contract D1/D2) — run manually, not part of pytest. Exercises the
REAL supervisor loop, probing ``GET /engine/ready``, against real fake engine
processes/servers:

1. engine answering /engine/ready 200, binary unset -> probe-only: available
2. engine dies, binary unset                        -> offline (loud, no spawn)
3. binary configured, engine dies mid-run           -> respawn + recover
4. stop() kills the spawned engine's process group
5. alive-but-slow child stays patient (no budget spent) until it answers
6. startup ceiling kills a child that never answers; the attempt counts
7. mettled exit 98 (port held by a foreign engine) -> offline, terminal
8. /engine/ready 503 with a downloading phase       -> starting, phase surfaced
9. /engine/ready 404 (old engine)                   -> incompatible, never ready
10. restart() un-sticks the 'failed' terminal state (the D1 restart() bug)

Run: uv run python benchmarks/smoke_supervision.py
"""
import http.server
import json
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


def wait_reason(sup: EngineSupervisor, timeout: float = 45.0) -> bool:
    """Wait for view()['reason'] to be set. Unlike ``wait_state(sup,
    "offline", ...)``, this can't false-positive on the constructor's initial
    ``state == "offline"`` default (observable before the supervisor thread
    has done anything) -- ``reason`` starts (and stays) ``None`` until a real
    code path (e.g. exit-98 handling) sets it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if sup.view()["reason"]:
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

    # --- 6. startup ceiling: a child that NEVER answers must not hold
    # "starting" forever. Short window; the hung group gets killed, the
    # attempt counts, a fresh child respawns. This is the production path
    # for a binary that binds nothing.
    port4 = free_port()
    hung = os.path.join(tempfile.mkdtemp(), "hung-mettled")
    with open(hung, "w") as f:
        f.write("#!/bin/sh\nsleep 600\n")  # alive, never answers
    os.chmod(hung, 0o755)
    sup4 = EngineSupervisor(Settings(
        backend_url=f"http://127.0.0.1:{port4}/v1",
        engine_binary=hung, engine_startup_window_s=1.5))
    sup4.ensure()
    pid4 = None
    for _ in range(100):  # wait for the first (hung) spawn to exist
        pid4 = sup4.view()["pid"]
        if pid4 is not None:
            break
        time.sleep(0.1)
    time.sleep(7.0)  # 1.5 s ceiling + 3 s TERM wait + 1 s backoff + respawn
    v4 = sup4.view()
    gone = pid4 is not None and not os.path.exists(f"/proc/{pid4}")
    if gone and v4["attempts"] >= 2:
        print(f"[ok] 6: hung child killed past ceiling, attempt counted "
              f"(pid {pid4} gone, attempts={v4['attempts']})")
    else:
        failures.append(f"6: hung engine not handled: pid4={pid4} gone={gone} v={v4}")
        print("[FAIL] 6:", v4)
    sup4.stop()
    time.sleep(1.0)
    if v4["pid"] is None or not os.path.exists(f"/proc/{v4['pid']}"):
        print("[ok] 6b: stop() reaped the respawned engine")
    else:
        failures.append(f"6b: respawned engine orphaned (pid={v4['pid']})")
        print("[FAIL] 6b: respawned engine still alive")

    # --- 7. mettled exit 98 (§1.6): a foreign engine holds the port --------
    # -> offline, terminal: no respawn loop, ever (another attempt would just
    # collide with the same foreign engine again).
    port7 = free_port()
    held98 = os.path.join(tempfile.mkdtemp(), "held98-mettled")
    with open(held98, "w") as f:
        f.write("#!/bin/sh\nexit 98\n")
    os.chmod(held98, 0o755)
    sup7 = EngineSupervisor(Settings(backend_url=f"http://127.0.0.1:{port7}/v1", engine_binary=held98))
    sup7.ensure()
    if wait_reason(sup7, timeout=15.0):
        v7 = sup7.view()
        if v7["state"] == "offline" and v7["reason"] and "98" in v7["reason"] and v7["pid"] is None:
            print(f"[ok] 7a: exit 98 -> offline, reason={v7['reason']!r}")
        else:
            failures.append(f"7a: unexpected view after exit 98: {v7}")
            print("[FAIL] 7a:", v7)
        time.sleep(3.0)  # no respawn loop: state/pid must not move afterwards
        v7b = sup7.view()
        if v7b["state"] == "offline" and v7b["pid"] is None:
            print("[ok] 7b: no respawn loop after exit 98")
        else:
            failures.append(f"7b: respawn loop after exit 98: {v7b}")
            print("[FAIL] 7b:", v7b)
    else:
        failures.append(f"7: never went offline after exit 98: {sup7.view()}")
        print("[FAIL] 7:", sup7.view())
    sup7.stop()

    # --- 8. /engine/ready 503 downloading -> starting, phase surfaced -------
    class _ReadyDownloadingHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = json.dumps({
                "ready": False,
                "phase": {"phase": "downloading", "model": "m", "bytes_done": 1, "bytes_total": 10},
                "reason": "loading",
            }).encode()
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # silence
            pass

    port8 = free_port()
    srv8 = http.server.ThreadingHTTPServer(("127.0.0.1", port8), _ReadyDownloadingHandler)
    threading.Thread(target=srv8.serve_forever, daemon=True).start()
    sup8 = EngineSupervisor(Settings(backend_url=f"http://127.0.0.1:{port8}/v1", engine_binary=None))
    sup8.ensure()
    if wait_state(sup8, "starting", timeout=10.0):
        v8 = sup8.view()
        if v8["phase"].get("phase") == "downloading":
            print("[ok] 8: 503 downloading -> starting, phase surfaced")
        else:
            failures.append(f"8: phase not surfaced: {v8}")
            print("[FAIL] 8:", v8)
    else:
        failures.append(f"8: never reached starting for 503: {sup8.view()}")
        print("[FAIL] 8:", sup8.view())
    sup8.stop()
    srv8.shutdown()
    srv8.server_close()

    # --- 9. /engine/ready 404 (old engine) -> incompatible, never ready -----
    class _ReadyMissingHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = b'{"error":"not found"}'
            self.send_response(404)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # silence
            pass

    port9 = free_port()
    srv9 = http.server.ThreadingHTTPServer(("127.0.0.1", port9), _ReadyMissingHandler)
    threading.Thread(target=srv9.serve_forever, daemon=True).start()
    sup9 = EngineSupervisor(Settings(backend_url=f"http://127.0.0.1:{port9}/v1", engine_binary=None))
    sup9.ensure()
    if wait_state(sup9, "incompatible", timeout=10.0):
        v9 = sup9.view()
        if v9["reason"] and "too old" in v9["reason"]:
            print("[ok] 9: 404 on /engine/ready -> incompatible, never ready")
        else:
            failures.append(f"9: reason missing/wrong: {v9}")
            print("[FAIL] 9:", v9)
    else:
        failures.append(f"9: never reached incompatible for 404: {sup9.view()}")
        print("[FAIL] 9:", sup9.view())
    sup9.stop()
    srv9.shutdown()
    srv9.server_close()

    # --- 10. restart() un-sticks 'failed' (the D1 restart() bug) ------------
    # Before the fix, restart() reset `attempts` outside the lock and never
    # set `_wake`, so the `failed` loop (which waits only on `_wake`/`_stop`)
    # never left `failed`. The respawn-after-restart cycle can complete faster
    # than any sampling interval can reliably observe a transient non-'failed'
    # state, so the binary itself records each invocation to a counter file --
    # that's the actual signal: did restart() cause ANOTHER spawn attempt.
    port10 = free_port()  # nothing ever listens here -- the binary always crashes first
    crash_dir = tempfile.mkdtemp()
    crash = os.path.join(crash_dir, "always-crash-mettled")
    counter = os.path.join(crash_dir, "invocations")
    with open(crash, "w") as f:
        f.write(f"#!/bin/sh\necho x >> {counter}\nexit 1\n")
    os.chmod(crash, 0o755)
    sup10 = EngineSupervisor(Settings(backend_url=f"http://127.0.0.1:{port10}/v1", engine_binary=crash))
    sup10._BACKOFF_S = (0.1,)  # instance override: tiny budget, reach 'failed' fast
    sup10.ensure()

    def invocations() -> int:
        try:
            with open(counter) as f:
                return sum(1 for _ in f)
        except FileNotFoundError:
            return 0

    if wait_state(sup10, "failed", timeout=15.0):
        print("[ok] 10a: exhausted budget -> failed")
        before = invocations()
        sup10.restart()
        respawned = False
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if invocations() > before:
                respawned = True
                break
            time.sleep(0.02)
        if respawned:
            print("[ok] 10b: restart() un-stuck the 'failed' loop (binary re-invoked)")
        else:
            failures.append("10b: restart() did not unstick 'failed' (regression)")
            print("[FAIL] 10b: still stuck in failed after restart()")
    else:
        failures.append(f"10a: never reached failed: {sup10.view()}")
        print("[FAIL] 10a:", sup10.view())
    sup10.stop()

    print()
    if failures:
        print("FAILURES:", failures)
        return 1
    print("ALL SUPERVISION SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
