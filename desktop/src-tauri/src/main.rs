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
//!    engine's process group), SIGKILL after a grace period.

use parking_lot::Mutex;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use tauri::Manager;

/// Handle to the sidecar so quit can take it down.
struct ServerHandle(Mutex<Option<Child>>);

fn free_port() -> std::io::Result<u16> {
    // Bind port 0, read it, drop the listener: the OS guarantees it was free
    // a moment ago; the server rebinds immediately after.
    let l = TcpListener::bind(("127.0.0.1", 0))?;
    let port = l.local_addr()?.port();
    drop(l);
    Ok(port)
}

/// Minimal HTTP/1.0 GET to /api/health over a raw TcpStream — the response we
/// need is one status line; a client crate would be the only other dependency.
fn health_ok(port: u16) -> bool {
    let Ok(mut s) = std::net::TcpStream::connect(("127.0.0.1", port)) else {
        return false;
    };
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
        .env("DK_ENGINE_BINARY", mettled);
    // The sidecar is its own session leader: quitting cannot leave the webview
    // pointing at a server orphaned behind an already-exited shell.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let child = cmd.spawn()?;
    Ok((child, port))
}

/// SIGTERM the sidecar, wait, then SIGKILL. The Python shutdown hook kills
/// mettled's process group on uvicorn's SIGTERM path.
fn stop_server(child: &mut Child) {
    unsafe {
        libc::kill(child.id() as i32, libc::SIGTERM);
    }
    for _ in 0..50 {
        if child.try_wait().map(|s| s.is_some()).unwrap_or(true) {
            return;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    let _ = child.kill();
    let _ = child.wait();
}

fn main() {
    tauri::Builder::default()
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
            let (child, port) = spawn_server(&resources)
                .map_err(|e| format!("failed to spawn docket-server sidecar: {e}"))?;
            *app.state::<ServerHandle>().0.lock() = Some(child);

            let deadline = Instant::now() + Duration::from_secs(30);
            while Instant::now() < deadline {
                if health_ok(port) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(250));
            }
            if !health_ok(port) {
                return Err(
                    format!("docket-server did not become healthy on port {port}").into(),
                );
            }

            let url = format!("http://127.0.0.1:{port}/")
                .parse()
                .expect("localhost URL");
            tauri::WebviewWindowBuilder::new(app, "main", tauri::WebviewUrl::External(url))
                .title("docket")
                .inner_size(1280.0, 860.0)
                .build()?;
            Ok(())
        })
        .on_window_event(|window, event| {
            // Quit-on-close (v1 decision): no tray, nothing stays resident.
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                let app = window.app_handle();
                if let Some(mut child) = app.state::<ServerHandle>().0.lock().take() {
                    stop_server(&mut child);
                }
                app.exit(0);
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running docket desktop shell");
}