"""Tests for vla_bridge/colab_server_utils.py, the Colab-side process/port helper.

Every notebook in vla_bridge/notebooks/ writes that module to /content and uses it to
stop stale servers, start its own server and prove the server it started owns the port.
These tests use real `sys.executable -u` child processes (socket listeners, sleepers,
an exiter and a stdlib http.server) on free ephemeral ports. Each test uses a unique
cmdline marker so stop_servers() can never match an unrelated process.
"""

import ast
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

psutil = pytest.importorskip("psutil")

from vla_bridge import colab_server_utils as csu  # noqa: E402

MODULE_PATH = Path(__file__).resolve().parent / "vla_bridge" / "colab_server_utils.py"

LISTENER = """\
import socket, sys, time
port, delay, marker = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
for line in sys.argv[4:]:
    print(line)
print("hello from child", marker)
time.sleep(delay)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", port))
s.listen(8)
print("listening on", port)
while True:
    time.sleep(0.2)
"""

SLEEPER = """\
import sys, time
print("sleeper", sys.argv[1])
while True:
    time.sleep(0.2)
"""

EXITER = """\
import sys
print("starting", sys.argv[1])
print("boom")
sys.exit(3)
"""

HEALTH_SERVER = """\
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port, status, marker = int(sys.argv[1]), sys.argv[2], sys.argv[3]

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"status": status, "protocol_version": 1}).encode()
        self.send_response(200 if self.path == "/health" else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass

print("health server", marker)
HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def scripts(tmp_path):
    paths = {}
    for name, text in (("listener", LISTENER), ("sleeper", SLEEPER),
                       ("exiter", EXITER), ("health", HEALTH_SERVER)):
        path = tmp_path / f"{name}.py"
        path.write_text(text)
        paths[name] = str(path)
    return paths


@pytest.fixture
def marker(tmp_path):
    return f"csu-marker-{tmp_path.name}-{os.getpid()}"


@pytest.fixture
def children():
    procs = []
    yield procs
    for proc in procs:
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def spawn(children, argv):
    """Start a child outside the helper (a decoy / pre-existing process)."""
    proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    children.append(proc)
    return proc


def wait_listening(port, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.05)
    raise AssertionError(f"nothing listening on {port}")


def started(children, *args, **kwargs):
    proc = csu.start_server(*args, **kwargs)
    children.append(proc)
    return proc


FAST = dict(poll_s=0.1, settle_s=0.2)


def test_listening_pids_finds_child_and_is_empty_for_free_port(scripts, marker, children):
    port = free_port()
    assert csu.listening_pids(port) == set()
    proc = spawn(children, [sys.executable, "-u", scripts["listener"], str(port), "0", marker])
    wait_listening(port)
    assert proc.pid in csu.listening_pids(port)


def test_start_server_and_wait_until_ready_tcp_mode(scripts, marker, children, tmp_path):
    port = free_port()
    log = tmp_path / "server.log"
    proc = started(children, [sys.executable, "-u", scripts["listener"], str(port), "0.5", marker],
                   log, env_overrides={"CSU_TEST_ENV": "1"})
    result = csu.wait_until_ready(proc, port, log, timeout_s=10, **FAST)
    assert result is None
    assert proc.poll() is None
    text = log.read_text()
    assert "hello from child" in text
    assert f"listening on {port}" in text


def test_start_server_sets_unbuffered_env_and_overrides(tmp_path, children):
    script = tmp_path / "env_probe.py"
    script.write_text("import os\nprint('PYU', os.environ.get('PYTHONUNBUFFERED'),"
                      " 'X', os.environ.get('CSU_X'))\n")
    log = tmp_path / "env.log"
    proc = started(children, [sys.executable, str(script)], log, env_overrides={"CSU_X": "42"})
    proc.wait(timeout=10)
    assert "PYU 1 X 42" in log.read_text()


def test_stale_owner_is_rejected_even_though_port_answers(scripts, marker, children, tmp_path):
    port = free_port()
    decoy = spawn(children, [sys.executable, "-u", scripts["listener"], str(port), "0", marker])
    wait_listening(port)
    log = tmp_path / "server.log"
    proc = started(children, [sys.executable, "-u", scripts["sleeper"], marker], log)
    with pytest.raises(RuntimeError, match="owned by") as excinfo:
        csu.wait_until_ready(proc, port, log, timeout_s=5, **FAST)
    assert str(decoy.pid) in str(excinfo.value)
    assert str(proc.pid) in str(excinfo.value)


def test_early_exit_raises_with_exit_code_and_log_tail(scripts, marker, children, tmp_path):
    port = free_port()
    log = tmp_path / "server.log"
    proc = started(children, [sys.executable, "-u", scripts["exiter"], marker], log)
    with pytest.raises(RuntimeError) as excinfo:
        csu.wait_until_ready(proc, port, log, timeout_s=10, **FAST)
    message = str(excinfo.value)
    assert "3" in message
    assert "boom" in message


def test_banner_found_anywhere_in_log_passes(scripts, marker, children, tmp_path):
    port = free_port()
    log = tmp_path / "server.log"
    banner = "Patched Thing.method: unit test banner"
    proc = started(children, [sys.executable, "-u", scripts["listener"], str(port), "0.2", marker,
                              "first line", "second line", banner, "after banner"], log)
    assert csu.wait_until_ready(proc, port, log, timeout_s=10, banner=banner, **FAST) is None
    assert not log.read_text().startswith(banner)


def test_missing_banner_raises(scripts, marker, children, tmp_path):
    port = free_port()
    log = tmp_path / "server.log"
    proc = started(children, [sys.executable, "-u", scripts["listener"], str(port), "0.2", marker,
                              "a stock server line"], log)
    with pytest.raises(RuntimeError, match="expected launcher banner"):
        csu.wait_until_ready(proc, port, log, timeout_s=10,
                             banner="Patched Missing.banner", **FAST)


def test_health_mode_returns_health_dict(scripts, marker, children, tmp_path):
    port = free_port()
    log = tmp_path / "server.log"
    proc = started(children, [sys.executable, "-u", scripts["health"], str(port), "ready", marker],
                   log)
    health = csu.wait_until_ready(proc, port, log, timeout_s=10,
                                  health_url=f"http://127.0.0.1:{port}/health",
                                  ready=lambda h: h.get("status") == "ready"
                                  and h.get("protocol_version") == 1, **FAST)
    assert health == {"status": "ready", "protocol_version": 1}


def test_health_mode_default_ready_uses_status(scripts, marker, children, tmp_path):
    port = free_port()
    log = tmp_path / "server.log"
    proc = started(children, [sys.executable, "-u", scripts["health"], str(port), "loading",
                              marker], log)
    with pytest.raises(TimeoutError):
        csu.wait_until_ready(proc, port, log, timeout_s=2,
                             health_url=f"http://127.0.0.1:{port}/health", **FAST)
    assert proc.wait(timeout=10) is not None


def test_health_mode_ready_predicate_false_times_out_and_terminates(scripts, marker, children,
                                                                     tmp_path):
    port = free_port()
    log = tmp_path / "server.log"
    proc = started(children, [sys.executable, "-u", scripts["health"], str(port), "ready", marker],
                   log)
    with pytest.raises(TimeoutError):
        csu.wait_until_ready(proc, port, log, timeout_s=2,
                             health_url=f"http://127.0.0.1:{port}/health",
                             ready=lambda h: False, **FAST)
    assert proc.wait(timeout=10) is not None


def test_stop_servers_stops_port_listener_and_marker_process(scripts, marker, children):
    port = free_port()
    listener = spawn(children, [sys.executable, "-u", scripts["listener"], str(port), "0",
                                "other-marker"])
    sleeper = spawn(children, [sys.executable, "-u", scripts["sleeper"], marker])
    wait_listening(port)
    stopped = csu.stop_servers(port, patterns=[marker], timeout_s=10)
    assert listener.wait(timeout=10) is not None
    assert sleeper.wait(timeout=10) is not None
    assert csu.listening_pids(port) == set()
    joined = "\n".join(stopped)
    assert str(listener.pid) in joined
    assert str(sleeper.pid) in joined


def test_stop_servers_on_idle_port_with_no_match_is_a_no_op(marker):
    assert csu.stop_servers(free_port(), patterns=[marker], timeout_s=2) == []


def test_stop_servers_never_stops_the_calling_process(scripts, children, monkeypatch):
    # Restrict the process scan to this process and one sleeper, then use a pattern
    # (the interpreter path) that both cmdlines contain. Only the sleeper may stop.
    sleeper = spawn(children, [sys.executable, "-u", scripts["sleeper"], "self-exclusion"])
    me = psutil.Process(os.getpid())
    other = psutil.Process(sleeper.pid)
    monkeypatch.setattr(csu.psutil, "process_iter", lambda *a, **k: iter([me, other]))
    stopped = csu.stop_servers(free_port(), patterns=[sys.executable], timeout_s=5)
    assert sleeper.wait(timeout=10) is not None
    assert me.is_running()
    assert len(stopped) == 1 and str(sleeper.pid) in stopped[0]


def test_process_report_ok_for_right_owner(scripts, marker, children, tmp_path, capsys):
    port = free_port()
    log = tmp_path / "server.log"
    proc = started(children, [sys.executable, "-u", scripts["listener"], str(port), "0", marker],
                   log)
    wait_listening(port)
    csu.process_report(port, marker, proc)
    out = capsys.readouterr().out
    assert f"OK: {marker} (PID {proc.pid}) owns port {port}" in out


def test_process_report_wrong_marker_raises(scripts, marker, children):
    port = free_port()
    proc = spawn(children, [sys.executable, "-u", scripts["listener"], str(port), "0", marker])
    wait_listening(port)
    with pytest.raises(RuntimeError, match="WRONG server on port"):
        csu.process_report(port, "fastfs_server.py", proc)


def test_process_report_foreign_owner_raises(scripts, marker, children):
    port = free_port()
    spawn(children, [sys.executable, "-u", scripts["listener"], str(port), "0", marker])
    mine = spawn(children, [sys.executable, "-u", scripts["sleeper"], marker])
    wait_listening(port)
    with pytest.raises(RuntimeError, match=str(mine.pid)):
        csu.process_report(port, marker, mine)


def test_process_report_nothing_listening_raises(marker):
    with pytest.raises(RuntimeError, match="nothing is listening"):
        csu.process_report(free_port(), marker)


def test_log_report_prints_tail_and_numbered_error_matches(tmp_path, capsys):
    log = tmp_path / "server.log"
    lines = [f"info line {i}" for i in range(50)]
    lines[3] = "Traceback (most recent call last):"
    lines[10] = "ERROR:root:Error in StreamActions: bad shape"
    lines[20] = "RuntimeError: beginAllocateToPool: already recording to mempool_id"
    lines[30] = "huggingface_hub.errors.GatedRepoError: 401"
    log.write_text("\n".join(lines) + "\n")
    csu.log_report(log, tail=5, max_matches=40)
    out = capsys.readouterr().out
    for i in range(45, 50):
        assert f"info line {i}" in out
    assert "info line 44" not in out
    assert "4: Traceback (most recent call last):" in out
    assert "11: ERROR:root:Error in StreamActions: bad shape" in out
    assert "21: RuntimeError: beginAllocateToPool" in out
    assert "31: huggingface_hub.errors.GatedRepoError" in out


def test_log_report_missing_log(tmp_path, capsys):
    csu.log_report(tmp_path / "nope.log")
    assert "does not exist" in capsys.readouterr().out


@pytest.mark.parametrize("line", [
    "ERROR something", "KeyError: 'x'", "NotImplementedError", "GatedRepoError",
    "Error in StreamActions", "Traceback (most recent call last):", "Exception: x",
    "torch.OutOfMemoryError: CUDA", "CUDA out of memory", "Killed",
    "beginAllocateToPool: already recording",
])
def test_error_pattern_matches(line):
    assert csu.ERROR_PATTERN.search(line)


@pytest.mark.parametrize("line", ["INFO server ready", "error budget (lowercase)", "killed it"])
def test_error_pattern_is_case_sensitive(line):
    assert not csu.ERROR_PATTERN.search(line)


def test_server_markers_cover_every_notebook_server():
    assert set(csu.SERVER_MARKERS) == {
        "lerobot.async_inference.policy_server", "diffusion_policy_server.py",
        "groot_n17_server.py", "fastfs_server.py", "openvla_oft_server.py",
        "openvla_server.py", "octo_server.py", "wallx_server.py", "rdt1b_server.py",
        "groot_n15_server.py",
    }


def test_stop_server_teardown_frees_port(scripts, marker, children, tmp_path, capsys):
    port = free_port()
    log = tmp_path / "server.log"
    proc = started(children, [sys.executable, "-u", scripts["listener"], str(port), "0", marker],
                   log)
    wait_listening(port)
    csu.stop_server(proc, port, [marker])
    assert proc.poll() is not None
    assert csu.listening_pids(port) == set()
    assert f"port {port} is free" in capsys.readouterr().out


def test_stop_server_with_no_process(marker, capsys):
    port = free_port()
    csu.stop_server(None, port, [marker])
    assert f"port {port} is free" in capsys.readouterr().out


def test_module_imports_only_stdlib_and_psutil():
    tree = ast.parse(MODULE_PATH.read_text())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative imports are not allowed"
            roots.add(node.module.split(".")[0])
    allowed = set(sys.stdlib_module_names) | {"psutil", "__future__"}
    assert roots <= allowed, roots - allowed
    assert "vla_bridge" not in roots
