//! docket desktop shell — thin Tauri layer.
//!
//! Responsibilities (METTLE_ONLY_CUTOVER_CONTRACT.md §8):
//! 1. spawn the frozen `docket-server` sidecar on a FREE port (a dev
//!    `dk serve` may hold 8760; never hardcode);
//! 2. point it at the bundled `mettled` via DK_ENGINE_BINARY (engine config
//!    and weights stay mettle's business — first-run download is engine-side,
//!    see mettle repo agent-docs/ENGINE_FIRST_RUN_SURFACE_CONTRACT.md);
//! 3. wait for /api/health, then open the webview at the server URL;
//! 4. quit-on-close: SIGTERM the sidecar (its uvicorn shutdown stops the
//!    engine's process group), SIGKILL after a grace period — on EVERY exit
//!    path, not just the window's close button:
//!      - `RunEvent::Exit` (fired once, at the very end of the event loop,
//!        regardless of whether the exit started from the window's close
//!        button, macOS Cmd+Q / app-menu Quit, or a programmatic
//!        `AppHandle::exit` — all of those funnel through
//!        `RunEvent::ExitRequested` then `ControlFlow::Exit` then this) —
//!        one teardown call here instead of forking logic per path;
//!      - a `setup()` failure (health probe timeout): torn down inline,
//!        before returning the error, since a failed `build()` never reaches
//!        `RunEvent::Exit`;
//!      - the shell process itself receiving SIGTERM/SIGINT directly (not
//!        via any window/menu event) — a raw libc signal handler, since
//!        that's the only thing safe to install outside Tauri's own loop;
//!      - the shell crashing without delivering any signal at all — a
//!        Linux `PR_SET_PDEATHSIG` kernel backstop set on the child at
//!        spawn time.
//! The sidecar itself now stops mettled's process group FIRST, before
//! uvicorn's own connection drain (packaging/docket_server_main.py) — see
//! `GRACE_PERIOD` below for why that changes this shell's kill budget.

use parking_lot::Mutex;
use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::atomic::{AtomicI32, Ordering};
use std::time::{Duration, Instant};

use tauri::Manager;

/// Handle to the sidecar so the one teardown path (`RunEvent::Exit` / setup
/// failure) can take it down.
struct ServerHandle(Mutex<Option<Child>>);

/// The sidecar's pid, mirrored outside any lock for the raw OS signal handler
/// below. A `parking_lot::Mutex` (or anything that allocates) is not safe to
/// touch from inside a signal handler — that path can only use this atomic
/// plus bare libc calls, never `ServerHandle`.
static CHILD_PID: AtomicI32 = AtomicI32::new(0);

/// Shell grace budget before SIGKILL-ing the sidecar.
///
/// Budget math: mettled's own worst-case kill path is TERM-wait 3 s +
/// KILL/reap-wait 5 s = 8 s (`docket/engine.py` `EngineSupervisor._kill_group`).
/// The sidecar (`packaging/docket_server_main.py`) now runs that kill
/// synchronously on the FIRST signal it receives, BEFORE uvicorn's own
/// connection drain even starts counting — so the only thing this grace period
/// must clear is that 8 s engine-kill path, not uvicorn's full HTTP drain
/// window. 12 s leaves a 4 s margin over the 8 s worst case; if the sidecar is
/// still draining an open SSE connection when that fires, SIGKILL-ing it here
/// is harmless — mettled is already dead by then.
const GRACE_PERIOD: Duration = Duration::from_secs(12);

fn free_port() -> std::io::Result<u16> {
    // Bind port 0, read it, drop the listener: the OS guarantees it was free
    // a moment ago; the server rebinds immediately after.
    let l = TcpListener::bind(("127.0.0.1", 0))?;
    let port = l.local_addr()?.port();
    drop(l);
    Ok(port)
}

/// Minimal HTTP/1.0 GET to /api/health over a raw TcpStream — the response we
/// need is one status line; a client crate would be the only other
/// dependency. Every socket op is time-bounded so a hung engine behind the
/// health endpoint can't block past this call's ~2 s budget (previously
/// unbounded, able to ride the backend's 120 s request timeout and blow
/// through the 30 s setup deadline in `setup()`).
fn health_ok(port: u16) -> bool {
    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    let Ok(mut s) = TcpStream::connect_timeout(&addr, Duration::from_secs(2)) else {
        return false;
    };
    let _ = s.set_read_timeout(Some(Duration::from_secs(2)));
    let _ = s.set_write_timeout(Some(Duration::from_secs(2)));
    if s.write_all(b"GET /api/health HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
        .is_err()
    {
        return false;
    }
    let mut buf = [0u8; 16];
    let n = s.read(&mut buf).unwrap_or(0);
    let status = std::str::from_utf8(&buf[..n]).unwrap_or("");
    status.starts_with("HTTP/1.0 200") || status.starts_with("HTTP/1.1 200")
}

/// Spawn the sidecar with the engine wired to the bundled mettled.
fn spawn_server(resources: &PathBuf) -> std::io::Result<(Child, u16)> {
    let port = free_port()?;
    let mut bin = resources.join("server/docket-server/docket-server");
    if !bin.exists() {
        bin = resources.join("server/docket-server/docket-server.exe");
    }
    // Tauri resource copies must be executable; enforce (resources keep mode
    // on most targets, but the AppImage squashfs is not guaranteed to).
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755));
    }
    let mettled = resources.join("engine/mettled");
    let mut cmd = Command::new(bin);
    cmd.arg("--port").arg(port.to_string())
        .env("DK_ENGINE_BINARY", mettled)
        // Packaged contract: quitting the app kills the engine group (the
        // sidecar's shutdown hook honours this flag; `dk serve` stays warm).
        .env("DK_ENGINE_EPHEMERAL", "1");
    // The sidecar is its own session leader: quitting cannot leave the webview
    // pointing at a server orphaned behind an already-exited shell.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
        #[cfg(target_os = "linux")]
        unsafe {
            cmd.pre_exec(|| {
                // Kernel backstop: if THIS shell process dies without ever
                // delivering a signal to the child (SIGKILL, a crash), the
                // kernel sends SIGTERM here directly, so the sidecar (and in
                // turn mettled, via its own engine-first SIGTERM handling —
                // packaging/docket_server_main.py) still dies with the app.
                // Linux-only; macOS/BSD have no PR_SET_PDEATHSIG.
                libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM as libc::c_ulong);
                Ok(())
            });
        }
    }
    let child = cmd.spawn()?;
    Ok((child, port))
}

/// `spawn_server`, retried once with a freshly probed port.
///
/// `free_port()` binds, reads the port, and drops the listener before the
/// sidecar binds that same port itself — a narrow window in which another
/// process can steal it. That race surfaces as the sidecar exiting almost
/// immediately instead of ever becoming healthy; detect that and retry once.
fn spawn_server_retrying(resources: &PathBuf) -> std::io::Result<(Child, u16)> {
    let (mut child, mut port) = spawn_server(resources)?;
    std::thread::sleep(Duration::from_millis(300));
    if matches!(child.try_wait(), Ok(Some(_))) {
        let (c2, p2) = spawn_server(resources)?;
        child = c2;
        port = p2;
    }
    CHILD_PID.store(child.id() as i32, Ordering::SeqCst);
    Ok((child, port))
}

/// SIGTERM the sidecar, wait up to `GRACE_PERIOD`, then SIGKILL+reap.
///
/// The one teardown implementation — called from every normal exit path
/// (`RunEvent::Exit`, a setup/health failure) so neither the sidecar nor the
/// engine it may have spawned ever outlives the app. See `GRACE_PERIOD`'s doc
/// for why this budget no longer needs to cover uvicorn's full graceful
/// drain. (The raw-signal and PDEATHSIG backstops below are necessarily
/// separate: a signal handler can't call into this — `Child` methods and the
/// allocator are not signal-safe.)
fn stop_server(child: &mut Child) {
    unsafe {
        libc::kill(child.id() as i32, libc::SIGTERM);
    }
    let deadline = Instant::now() + GRACE_PERIOD;
    while Instant::now() < deadline {
        if matches!(child.try_wait(), Ok(Some(_))) {
            CHILD_PID.store(0, Ordering::SeqCst);
            return;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    let _ = child.kill();
    let _ = child.wait();
    CHILD_PID.store(0, Ordering::SeqCst);
}

/// Raw OS signal handler: the shell's own last-resort path for when the
/// process receives SIGTERM/SIGINT directly (`kill <pid>`, a terminal
/// Ctrl-C) rather than through any Tauri window/menu event.
///
/// Signal-handler-safe only: no Rust std allocation, no locks, no Tauri
/// API — just the raw process-group kill and an immediate exit. Deliberately
/// best-effort (one TERM, no wait/KILL follow-up): the process is exiting
/// right now either way, and the sidecar's own engine-first shutdown (see
/// `stop_server`'s doc) still gets its chance to run once it receives this
/// TERM.
extern "C" fn handle_term_signal(_sig: i32) {
    let pid = CHILD_PID.load(Ordering::SeqCst);
    if pid > 0 {
        unsafe {
            libc::kill(-pid, libc::SIGTERM);
        }
    }
    unsafe {
        libc::_exit(143);
    }
}

fn main() {
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGTERM, handle_term_signal as libc::sighandler_t);
        libc::signal(libc::SIGINT, handle_term_signal as libc::sighandler_t);
    }

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            // Second launch: focus the existing window instead of spawning a
            // second server + engine (they'd collide on state and VRAM).
            if let Some(w) = app.get_webview_window("main") {
                let _ = w.set_focus();
            }
        }))
        .manage(ServerHandle(Mutex::new(None)))
        .setup(|app| {
            let resources = app.path().resource_dir()?;
            let (mut child, port) = spawn_server_retrying(&resources)
                .map_err(|e| format!("failed to spawn docket-server sidecar: {e}"))?;

            let deadline = Instant::now() + Duration::from_secs(30);
            while Instant::now() < deadline {
                if health_ok(port) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(250));
            }
            if !health_ok(port) {
                // A `setup()` error makes `build()` return Err without ever
                // reaching `RunEvent::Exit`, so tear the still-local child
                // down right here before propagating — otherwise it (and any
                // engine it spawned) would be dropped unkilled while the app
                // silently vanishes.
                stop_server(&mut child);
                let msg = format!("docket-server did not become healthy on port {port}");
                // No dialog surface is wired in: tauri's dialog API lives in
                // a separate plugin crate (tauri-plugin-dialog) in Tauri v2,
                // not the core `tauri` crate, and this ticket's scope is "no
                // new crates beyond libc" — so stderr is the only surface for
                // this failure today (flagged in the ticket report).
                eprintln!("docket: {msg}");
                return Err(msg.into());
            }
            *app.state::<ServerHandle>().0.lock() = Some(child);

            let url = format!("http://127.0.0.1:{port}/")
                .parse()
                .expect("localhost URL");
            tauri::WebviewWindowBuilder::new(app, "main", tauri::WebviewUrl::External(url))
                .title("docket")
                .inner_size(1280.0, 860.0)
                .build()?;
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building docket desktop shell");

    app.run(|app_handle, event| {
        // Quit-on-close (v1 decision): no tray, nothing stays resident.
        // `RunEvent::Exit` is the single event every normal exit path
        // funnels through (window close, macOS Cmd+Q / app-menu Quit, a
        // programmatic `AppHandle::exit`) — one teardown call here instead
        // of forking logic per path.
        if let tauri::RunEvent::Exit = event {
            if let Some(mut child) = app_handle.state::<ServerHandle>().0.lock().take() {
                stop_server(&mut child);
            }
        }
    });
}
