"""Colab-side process/port helper for the per-backend policy-server notebooks.

Every notebook in control/vla_bridge/notebooks/ writes this file to
/content/colab_server_utils.py with a %%writefile cell (byte-identical to this file, checked
by a sha256 cell and by control/test_colab_notebooks.py) and imports it with
sys.path.insert(0, '/content'). Local control code never imports it.

What it guarantees for a launch cell:
- stop_servers() clears any earlier server: every process whose cmdline contains a known
  server marker, plus whatever is listening on the notebook's port.
- start_server() starts the server unbuffered (PYTHONUNBUFFERED=1) with its log in a file.
- wait_until_ready() only succeeds when the process this cell started (or one of its
  children) owns the port. A stale server that answers on the port is an error, not a
  success. It also fails on an early exit and, for patched launchers, on a missing banner.

The diagnostic helpers (log_report, process_report, gpu_report, health_report) print what a
partner needs to debug a run, and stop_server() is the teardown.

Imports: stdlib + psutil only (psutil is preinstalled on Colab).
"""

import json
import os
import re
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import psutil

SERVER_MARKERS = (
    "lerobot.async_inference.policy_server",
    "diffusion_policy_server.py",
    "groot_n17_server.py",
    "fastfs_server.py",
    "openvla_oft_server.py",
    "openvla_server.py",
    "octo_server.py",
    "wallx_server.py",
    "rdt1b_server.py",
    "groot_n15_server.py",
)

# Case-sensitive on purpose: "Error" covers KeyError, NotImplementedError, GatedRepoError and
# "Error in StreamActions"; lowercase prose such as "error budget" is not a match.
ERROR_PATTERN = re.compile(
    r"ERROR|Error|Traceback|Exception|OutOfMemory|out of memory|Killed|beginAllocateToPool"
)

_ACCESS_ERRORS = (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess)


def _cmdline(proc) -> str:
    try:
        return " ".join(proc.cmdline())
    except _ACCESS_ERRORS:
        return ""
    except OSError:
        return ""


def _describe(pid: int) -> str:
    try:
        return f"PID {pid} ({_cmdline(psutil.Process(pid)) or '?'})"
    except _ACCESS_ERRORS:
        return f"PID {pid} (gone)"


def _tcp_connections(proc):
    # Colab's psutil may predate 6.0, where net_connections() replaced connections().
    method = getattr(proc, "net_connections", None) or proc.connections
    return method(kind="tcp")


def _read_lines(log_path) -> list:
    path = Path(log_path)
    if not path.exists():
        return []
    return path.read_text(errors="replace").splitlines()


def _tail(log_path, lines: int = 40) -> str:
    return "\n".join(_read_lines(log_path)[-lines:])


def listening_pids(port: int) -> set:
    """PIDs of every visible process with a TCP socket in LISTEN state on `port`."""
    pids = set()
    for proc in psutil.process_iter():
        try:
            for conn in _tcp_connections(proc):
                if conn.status == psutil.CONN_LISTEN and conn.laddr and conn.laddr.port == port:
                    pids.add(proc.pid)
        except _ACCESS_ERRORS:
            continue
        except OSError:
            continue
    return pids


def stop_servers(port: int, patterns: Sequence[str] = (), timeout_s: float = 15.0) -> list:
    """Stop every process matching a pattern plus the port's listener; return what was stopped."""
    me = os.getpid()
    targets = {}
    for proc in psutil.process_iter():
        if proc.pid == me:
            continue
        cmdline = _cmdline(proc)
        if cmdline and any(pattern and pattern in cmdline for pattern in patterns):
            targets[proc.pid] = proc
    for pid in listening_pids(port):
        if pid != me and pid not in targets:
            try:
                targets[pid] = psutil.Process(pid)
            except _ACCESS_ERRORS:
                continue
    stopped = []
    for pid, proc in sorted(targets.items()):
        description = f"PID {pid} ({_cmdline(proc) or '?'})"
        try:
            proc.terminate()
            stopped.append(description)
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            print(f"Cannot stop {description}: access denied")
    _, alive = psutil.wait_procs(list(targets.values()), timeout=5)
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=5)
    for line in stopped:
        print(f"Stopped {line}")
    deadline = time.monotonic() + timeout_s
    while True:
        holders = listening_pids(port) - {me}
        if not holders:
            return stopped
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"port {port} is still held by "
                + ", ".join(_describe(pid) for pid in sorted(holders))
                + f" after {timeout_s:.0f} s; stop it by hand before launching"
            )
        time.sleep(0.2)


def start_server(argv: Sequence[str], log_path,
                 env_overrides: Optional[Mapping[str, str]] = None,
                 cwd=None) -> subprocess.Popen:
    """Start `argv` unbuffered with stdout+stderr in `log_path` (truncated)."""
    env = os.environ.copy()
    env.update(env_overrides or {})
    env["PYTHONUNBUFFERED"] = "1"
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "wb", buffering=0)
    try:
        proc = subprocess.Popen(list(argv), stdout=log, stderr=subprocess.STDOUT, env=env,
                                cwd=None if cwd is None else str(cwd))
    finally:
        log.close()
    print(f"Started PID {proc.pid}: {' '.join(argv)}")
    print(f"Log → {log_path}")
    return proc


def _probe(port: int, health_url: Optional[str], ready: Optional[Callable[[dict], bool]]):
    """Return (answered, health): answered is True once the server counts as ready."""
    if health_url:
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                health = json.loads(response.read().decode())
        except Exception:
            return False, None
        if not isinstance(health, dict):
            return False, None
        ok = ready(health) if ready is not None else health.get("status") == "ready"
        return bool(ok), health
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True, None
    except OSError:
        return False, None


def _own_pids(proc) -> set:
    pids = {proc.pid}
    try:
        pids.update(child.pid for child in psutil.Process(proc.pid).children(recursive=True))
    except _ACCESS_ERRORS:
        pass
    return pids


def wait_until_ready(proc, port: int, log_path, timeout_s: float,
                     health_url: Optional[str] = None,
                     ready: Optional[Callable[[dict], bool]] = None,
                     banner: Optional[str] = None,
                     settle_s: float = 2.0, poll_s: float = 1.0):
    """Wait until the server this cell started owns `port` and is ready.

    Returns the /health dict in health mode, None in TCP mode. Raises RuntimeError on an
    early exit, a foreign/stale owner or a missing launcher banner, and TimeoutError (after
    terminating `proc`) when the deadline passes.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        code = proc.poll()
        if code is not None:
            raise RuntimeError(
                f"server PID {proc.pid} exited with code {code} before it was ready. "
                f"Last log lines of {log_path}:\n{_tail(log_path)}"
            )
        answered, health = _probe(port, health_url, ready)
        if answered:
            owners = listening_pids(port)
            mine = _own_pids(proc)
            if not owners or not owners <= mine:
                foreign = ", ".join(_describe(pid) for pid in sorted(owners - mine))
                foreign = foreign or "no visible process"
                raise RuntimeError(
                    f"port {port} answered but is owned by {foreign}, not the server this cell "
                    f"started (PID {proc.pid}). That is a stale server; re-run the launch cell, "
                    "which stops old servers first."
                )
            time.sleep(settle_s)
            code = proc.poll()
            if code is not None:
                raise RuntimeError(
                    f"server PID {proc.pid} exited with code {code} right after binding port "
                    f"{port}. Last log lines of {log_path}:\n{_tail(log_path)}"
                )
            if banner is not None:
                text = Path(log_path).read_text(errors="replace") if Path(log_path).exists() else ""
                if banner not in text:
                    raise RuntimeError(
                        f"expected launcher banner '{banner}' not found in {log_path}, so this "
                        "is not the patched launcher. Re-run the embed cells, then the launch cell."
                    )
            print(f"READY: PID {proc.pid} ({_cmdline(psutil.Process(proc.pid))}) owns port "
                  f"{port}; log → {log_path}")
            return health
        time.sleep(poll_s)
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    raise TimeoutError(
        f"server PID {proc.pid} was not ready on port {port} within {timeout_s:.0f} s "
        f"(terminated). Last log lines of {log_path}:\n{_tail(log_path)}"
    )


def log_report(log_path, tail: int = 80, max_matches: int = 40) -> None:
    """Print the last `tail` log lines and up to `max_matches` numbered error lines."""
    path = Path(log_path)
    if not path.exists():
        print(f"{path} does not exist (has the launch cell run on this runtime?)")
        return
    lines = _read_lines(path)
    print(f"--- last {tail} lines of {path} ({len(lines)} lines total) ---")
    for line in lines[-tail:]:
        print(line)
    matches = [(number, line) for number, line in enumerate(lines, start=1)
               if ERROR_PATTERN.search(line)]
    print(f"--- {len(matches)} lines matching ERROR_PATTERN (showing up to {max_matches}) ---")
    for number, line in matches[:max_matches]:
        print(f"{number}: {line}")
    if not matches:
        print("(none)")


def process_report(port: int, expected_marker: str, proc=None) -> None:
    """Print every known server process, then check the right server owns `port`."""
    print("--- running server processes ---")
    found = False
    now = time.time()
    for candidate in psutil.process_iter():
        cmdline = _cmdline(candidate)
        if not cmdline or not any(marker in cmdline for marker in SERVER_MARKERS):
            continue
        try:
            elapsed = now - candidate.create_time()
            rss_mb = candidate.memory_info().rss / 1e6
        except _ACCESS_ERRORS:
            continue
        found = True
        print(f"PID {candidate.pid}  up {elapsed:7.0f} s  RSS {rss_mb:8.1f} MB  {cmdline}")
    if not found:
        print("(no process with a known server marker)")
    owners = listening_pids(port)
    if not owners:
        raise RuntimeError(f"nothing is listening on port {port}; run the launch cell")
    for pid in sorted(owners):
        try:
            cmdline = _cmdline(psutil.Process(pid))
        except _ACCESS_ERRORS:
            cmdline = ""
        if expected_marker not in cmdline:
            raise RuntimeError(
                f"WRONG server on port {port}: PID {pid} ({cmdline or '?'}) does not contain "
                f"'{expected_marker}'. Run the teardown cell, then the launch cell."
            )
    if proc is not None:
        mine = _own_pids(proc)
        if not owners <= mine:
            raise RuntimeError(
                f"port {port} is owned by "
                + ", ".join(_describe(pid) for pid in sorted(owners))
                + f", not by the server this notebook started (PID {proc.pid})"
            )
    print(f"OK: {expected_marker} (PID {min(owners)}) owns port {port}")


def gpu_report() -> None:
    """Print GPU memory/utilisation and the processes using the GPU."""
    queries = (
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
         "--format=csv"],
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"],
    )
    for argv in queries:
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        except FileNotFoundError:
            print("no nvidia-smi (CPU runtime?): pick a GPU runtime for this notebook")
            return
        print(result.stdout.rstrip() or result.stderr.rstrip())


def health_report(port: int) -> dict:
    """GET /health on the local server, pretty-print it and return it."""
    url = f"http://127.0.0.1:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            health = json.loads(response.read().decode())
    except Exception as exc:
        print(f"GET {url} failed: {type(exc).__name__}: {exc}")
        raise
    print(json.dumps(health, indent=2, sort_keys=True))
    return health


def stop_server(proc, port: int, patterns: Sequence[str] = ()) -> None:
    """Teardown: stop `proc` (if alive) and any leftover server, then confirm the port is free."""
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        print(f"Stopped PID {proc.pid}")
    stop_servers(port, patterns)
    print(f"port {port} is free")
