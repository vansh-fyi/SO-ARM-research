"""Generate the per-backend Colab notebooks in control/vla_bridge/notebooks/.

One self-contained notebook per Colab-hosted model or service (13), plus README.md, all
rendered from scripts/colab_notebook_specs.py and the control/vla_bridge sources. Every .py
a notebook runs is embedded as a %%writefile cell whose body is byte-identical to the repo
file, followed by a sha256 verify cell, so nothing has to be uploaded to Colab.

Usage (from the repo root):
    python scripts/build_colab_notebooks.py           # write the notebooks + README
    python scripts/build_colab_notebooks.py --check   # exit 1 if anything drifted

control/test_colab_notebooks.py runs the same comparison in the test suite.
Stdlib only.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import colab_notebook_specs as specs  # noqa: E402

VLA_BRIDGE = ROOT / "control" / "vla_bridge"
NOTEBOOK_DIR = VLA_BRIDGE / "notebooks"
HELPER = "colab_server_utils.py"
CELL_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# --- Shared cells, lifted verbatim from the former combined policy-server notebook -------

OPENMP_GUARDS = r'''import os

# Test 7 UAT crash investigation (Phase 12, 2026-09-27): Colab sessions that
# load multiple ML libraries (torch, opencv-headless, Fast-FoundationStereo's
# own deps) commonly end up with more than one bundled OpenMP runtime
# (libgomp vs libiomp) loaded into the same process -- the classic
# "OMP: Error #15" failure mode, which manifests as an abrupt, tracebackless
# kernel death the first time a *new* library's compute path runs for the
# first time (matches "crashes right as inference starts"). Set before any
# heavy imports so it applies process-wide to every cell below.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
print("OpenMP duplicate-runtime guards set.")'''

PYNGROK_NOTE = r'''**pyngrok legitimacy gate:** the next cell installs `pyngrok`, flagged
`SUS` in `11-RESEARCH.md`'s Package Legitimacy Audit. Do **not** run that
install cell until a human has completed the verification in this plan's
Task 3 `checkpoint:human-verify` (visit pypi.org/project/pyngrok, confirm
the `ngrok`/`alexdlaird` maintainer namespace and that `8.1.2` is a real
published version).'''

PYNGROK_INSTALL = r'''# Colab notebook cell — DO NOT RUN until the human has completed the
# pyngrok legitimacy check in this plan's Task 3 checkpoint:human-verify
!pip install pyngrok==8.1.2'''

TOKEN_CELL = r'''# One-time token setup: HF Hub (for authenticated, faster/rate-limit-free checkpoint
# downloads) + ngrok (for the TCP tunnel). Checks for saved token files on Google Drive
# first (same MyDrive/SoARM-Research/.hf_token / .ngrok_token convention as other
# cells in this project), else falls back to a one-time getpass() prompt so tokens
# are never echoed or left in cell output/notebook history.
import getpass
from pathlib import Path

_hf_token = None
_ngrok_token = None
try:
    from google.colab import drive
    drive.mount("/content/drive")
    _hf_token_path = Path("/content/drive/MyDrive/SoARM-Research/.hf_token")
    _ngrok_token_path = Path("/content/drive/MyDrive/SoARM-Research/.ngrok_token")
    if _hf_token_path.exists():
        _hf_token = _hf_token_path.read_text().strip()
        print(f"HF token loaded from {_hf_token_path}")
    if _ngrok_token_path.exists():
        _ngrok_token = _ngrok_token_path.read_text().strip()
        print(f"ngrok token loaded from {_ngrok_token_path}")
except Exception as _exc:
    print(f"Drive mount/token read skipped ({type(_exc).__name__}: {_exc})")

if not _hf_token:
    _hf_token = getpass.getpass("HF Hub token (paste here): ")
if not _ngrok_token:
    _ngrok_token = getpass.getpass("ngrok authtoken (paste here): ")

from huggingface_hub import login
login(token=_hf_token)
print("Logged into HF Hub.")

from pyngrok import ngrok
ngrok.set_auth_token(_ngrok_token)
print("ngrok authtoken set.")

del _hf_token, _ngrok_token'''

GRPCIO_NOTE = r'''Adds `grpcio` — already audited and approved in `11-RESEARCH.md`'s Package
Legitimacy Audit as a required transitive dependency of `lerobot`'s own
`async` extra. No separate checkpoint needed for this install.'''

TRANSFORMERS_NOTE = r'''**Why the pins (live session, 2026-10-18).** Colab now ships `transformers` 5.13.1.
`transformers` >= 5.9 removed the `cache_position` argument from `create_causal_mask`, and
lerobot 0.6.1's `pi_gemma` still passes it, so pi0 and pi05 fail every inference on Colab's
default. The pin cell installs `transformers==5.5.4` after lerobot, and the version cells
before and after the installs prove it: they fail if `transformers` is not 5.5.4, if
`lerobot` is not 0.6.1, or if pip changed `torch`. `torch` must stay Colab's
2.11.0+cu128.'''

TORCHAUDIO_NOTE = r'''`lerobot`'s installs (the `lerobot[async]` and `lerobot[dataset]` installs above) pull in a `torchaudio` build whose CUDA version doesn't always match Colab's preinstalled `torch` -- confirmed live (Phase 12 Test 7 UAT) to crash the launch cell's `from lerobot.async_inference.policy_server import serve` with `RuntimeError: Detected that PyTorch and TorchAudio were compiled with different CUDA versions`, which cascades into a `ModuleNotFoundError` for `transformers.AutoProcessor` and can leave the kernel needing a full restart. `torchaudio` isn't actually used by anything in this notebook -- uninstalling it removes the mismatched import path entirely (`transformers`' own `is_torchaudio_available()` check then correctly skips it).'''

TORCHAUDIO_UNINSTALL = "!pip uninstall -y torchaudio"

CLOUDFLARED_NOTE = r'''Copy the printed `host:port` (the `tcp://` prefix is already stripped) for the local commands below.

If ngrok's free tier no longer supports a TCP tunnel endpoint by the time
this is run (Assumption A4 in `11-RESEARCH.md`), fall back to `cloudflared`
with a configured Zero Trust tunnel for raw TCP forwarding (heavier setup
than ngrok TCP — cloudflared's zero-config "quick tunnels" are HTTP-only
and will not work here).'''

TEARDOWN_NOTE = r'''## Teardown (every session, no exceptions)

An unattended open tunnel into a Colab-hosted {server} is an
elevation-of-privilege risk (T-11-08 in this plan's threat register) —
stop both processes at the end of every session, not just when finished
for the day: the cell below disconnects this notebook's tunnel, stops the
ngrok agent, stops the server and confirms port {port} is free. Run it only
after the local episode or smoke run has finished.'''

GRPC_MARKERS = (
    "lerobot.async_inference.policy_server",
    "diffusion_policy_server.py",
    "groot_n17_server.py",
)


# --- Cell helpers ---------------------------------------------------------------------------

def _cell(slug, cell_type, text):
    if not CELL_ID.match(slug):
        raise ValueError(f"invalid cell id {slug!r}")
    cell = {"cell_type": cell_type}
    if cell_type == "code":
        cell["execution_count"] = None
    cell["id"] = slug
    cell["metadata"] = {}
    if cell_type == "code":
        cell["outputs"] = []
    cell["source"] = text.splitlines(keepends=True)
    return cell


def md(slug, text):
    return _cell(slug, "markdown", text.strip("\n"))


def code(slug, text):
    return _cell(slug, "code", text.strip("\n"))


def _source(name):
    path = VLA_BRIDGE / name
    text = path.read_text()
    if not text.endswith("\n") or text.endswith("\n\n"):
        raise ValueError(f"{path} must end with exactly one newline to embed byte-identically")
    # IPython dedents a cell before running %%writefile, which empties whitespace-only lines.
    if re.search(r"^[ \t]+$", text, re.MULTILINE):
        raise ValueError(f"{path} has whitespace-only lines that %%writefile would not preserve")
    return text


def _slug(name):
    return Path(name).stem.replace("_", "-")


def embed_cells(name):
    """%%writefile cell (byte-identical body) + sha256 verify cell for control/vla_bridge/<name>."""
    text = _source(name)
    digest = hashlib.sha256((VLA_BRIDGE / name).read_bytes()).hexdigest()
    embed_id = "resident-fastfs-source" if name == "fastfs_server.py" else f"embed-{_slug(name)}"
    verify = f'''import hashlib
from pathlib import Path

EXPECTED_SHA256 = '{digest}'
actual = hashlib.sha256(Path('/content/{name}').read_bytes()).hexdigest()
if actual != EXPECTED_SHA256:
    raise RuntimeError(
        f'/content/{name} has sha256 {{actual[:12]}}, expected {{EXPECTED_SHA256[:12]}}: '
        're-run the %%writefile cell above. To change the file, edit control/vla_bridge/{name} '
        'and regenerate the notebooks (scripts/build_colab_notebooks.py).'
    )
print(f'/content/{name} matches control/vla_bridge/{name} (sha256 {{actual[:12]}})')'''
    return [
        _cell(embed_id, "code", f"%%writefile /content/{name}\n" + text),
        code(f"verify-{_slug(name)}", verify),
    ]


def one_line(command):
    return re.sub(r"\s*\\\n\s*", " ", command.strip())


def _helper_import(*names):
    return ("import sys\n\nsys.path.insert(0, '/content')\n"
            f"from colab_server_utils import {', '.join(sorted(names))}\n")


# --- Header ---------------------------------------------------------------------------------

def _licence_text(spec):
    if not spec["licences"]:
        return "none"
    return "; ".join(f"[{repo}](https://huggingface.co/{repo}): accept with the token's account "
                     "first" for repo in spec["licences"])


def title_md(spec):
    tunnel = "ngrok TCP" if spec["tunnel"] == "tcp" else "ngrok HTTP"
    embedded = ", ".join(f"`{name}`" for name in [HELPER, *spec["embed"]])
    rows = [
        ("Lane", spec["lane_label"]),
        ("Port → tunnel", f"{spec['port']} → {tunnel}"),
        ("GPU runtime", spec["gpu"]),
        ("Gated HF licences", _licence_text(spec)),
        ("Sign-offs before running", spec["prerequisites"] or "none"),
        ("Default checkpoint", f"`{spec['checkpoint']}`"),
        ("Server log", f"`{spec['log']}`"),
        ("Embedded files (written to `/content`)", embedded),
    ]
    table = "\n".join(f"| {key} | {value} |" for key, value in rows)
    return f"""# {spec['title']}

{spec['serves'].strip()}

| | |
|---|---|
{table}

Run top to bottom on a fresh runtime. Works from the VS Code Colab extension: nothing to upload.
Every file this notebook runs is written into `/content` by a cell below and checked against
its sha256. Run the teardown cell at the end of every session. The index of all notebooks,
with the local smoke and episode commands, is `control/vla_bridge/notebooks/README.md`.

Generated by `scripts/build_colab_notebooks.py` from `scripts/colab_notebook_specs.py` +
`control/vla_bridge` sources, so edit those and regenerate, not this file."""


def gated_access_cells(spec):
    repos = ", ".join(repr(repo) for repo in spec["licences"])
    check = f'''from huggingface_hub import hf_hub_download

GATED_REPOS = [{repos}]
for repo in GATED_REPOS:
    try:
        hf_hub_download(repo, 'config.json')
    except Exception as exc:
        raise RuntimeError(
            f'{{repo}}: {{type(exc).__name__}}: {{exc}}. The HF token cannot read this gated repo: '
            f'accept the licence at https://huggingface.co/{{repo}} with the token\\'s account, '
            'then re-run this cell.'
        ) from exc
    print(f'{{repo}}: gated access OK')'''
    return [
        md("gated-access-heading", "## Gated licence check\n\nFails now, with the repo "
           "name, instead of minutes later inside the server's model load."),
        code("gated-access-check", check),
    ]


def header_cells(spec):
    cells = [
        md("title", title_md(spec)),
        code("openmp-guards", OPENMP_GUARDS),
        md("pyngrok-legitimacy", PYNGROK_NOTE),
        code("pyngrok-install", PYNGROK_INSTALL),
        code("tokens", TOKEN_CELL),
    ]
    if spec["licences"]:
        cells += gated_access_cells(spec)
    cells.append(md("helper-heading", "## Server helper\n\nThe next cell writes "
                    f"`control/vla_bridge/{HELPER}` into `/content/{HELPER}`; the cell after it "
                    "checks its sha256. The launch, diagnostic and teardown cells import it to stop "
                    "stale servers, prove this notebook's server owns its port, and report "
                    "logs, processes and GPU memory."))
    cells += embed_cells(HELPER)
    return cells


# --- gRPC lane ------------------------------------------------------------------------------

VERSIONS_BEFORE = r'''# Record torch + transformers before any install (fresh interpreter, not this kernel's
# imports). The versions cell after the installs fails if pip changed torch.
import subprocess
import sys
from pathlib import Path

TORCH_BEFORE = Path('/content/torch_version_before.txt')
probe = "import torch, transformers; print(torch.__version__); print(transformers.__version__)"
torch_version, transformers_version = subprocess.run(
    [sys.executable, '-c', probe], capture_output=True, text=True, check=True).stdout.split()[-2:]
print('before installs: torch', torch_version, '| transformers', transformers_version)
if TORCH_BEFORE.exists():
    print(f'Keeping the first recorded torch {TORCH_BEFORE.read_text().strip()} ({TORCH_BEFORE})')
else:
    TORCH_BEFORE.write_text(torch_version + '\n')
    print(f'Recorded torch {torch_version} → {TORCH_BEFORE}')'''


def versions_after(spec):
    expected = {"transformers": "5.5.4", "lerobot": "0.6.1"}
    if spec["install"].get("diffusers"):
        expected["diffusers"] = "0.39.0"
    return f'''# Fresh interpreter: this kernel may still hold pre-install packages in memory.
import json
import subprocess
import sys
from pathlib import Path

EXPECTED = {expected!r}
VERIFIED_TORCH = '2.11.0+cu128'
probe = (
    "import importlib.metadata as m, json, torch\\n"
    "def v(p):\\n"
    "    try:\\n"
    "        return m.version(p)\\n"
    "    except m.PackageNotFoundError:\\n"
    "        return None\\n"
    "print(json.dumps({{'torch': torch.__version__, 'cuda': torch.version.cuda,\\n"
    "                  'cudnn': torch.backends.cudnn.version(), 'transformers': v('transformers'),\\n"
    "                  'lerobot': v('lerobot'), 'diffusers': v('diffusers'), 'grpcio': v('grpcio')}}))\\n"
)
result = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True)
if result.returncode != 0:
    raise RuntimeError(f'version probe failed:\\n{{result.stderr[-3000:]}}')
versions = json.loads(result.stdout.strip().splitlines()[-1])
for name, version in versions.items():
    print(f'{{name:13s}} {{version}}')
problems = [f'{{name}} is {{versions[name]}}, expected {{want}}'
            for name, want in EXPECTED.items() if versions[name] != want]
torch_before = Path('/content/torch_version_before.txt').read_text().strip()
if versions['torch'] != torch_before:
    problems.append(f'pip changed torch: {{torch_before}} → {{versions["torch"]}}; '
                    'start a fresh runtime and run the cells in order')
if problems:
    raise RuntimeError('; '.join(problems))
if versions['torch'] != VERIFIED_TORCH:
    print(f'WARNING: torch is {{versions["torch"]}}, but the pins were verified live with '
          f'{{VERIFIED_TORCH}} on 2026-10-18. Expect surprises.')
print('Versions OK.')'''


def grpc_install_cells(spec):
    extras = spec["install"]["extras"]
    install = (f"# lerobot 0.6.1 with the `{extras}` extra(s), the dataset extra and num2words.\n"
               f"!pip install 'lerobot[{extras}]==0.6.1'\n"
               "!pip install 'lerobot[dataset]==0.6.1'\n"
               "!pip install num2words")
    pin = ("# Pin transformers AFTER lerobot: Colab's 5.13.1 breaks lerobot 0.6.1 (see above).\n"
           "!pip install 'transformers==5.5.4'")
    if spec["install"].get("diffusers"):
        pin += ("\n# lerobot's `diffusion` extra (diffusers>=0.38.0,<0.40.0), pinned.\n"
                "!pip install 'diffusers==0.39.0'")
    return [
        md("install-heading", "## Install lerobot 0.6.1 and pin transformers\n\n" + GRPCIO_NOTE
           + "\n\n" + TRANSFORMERS_NOTE),
        code("versions-before", VERSIONS_BEFORE),
        code("install-lerobot", install),
        code("pin-versions", pin),
        md("torchaudio-note", TORCHAUDIO_NOTE),
        code("torchaudio-uninstall", TORCHAUDIO_UNINSTALL),
        code("versions-after", versions_after(spec)),
    ]


def _placeholder_cell(spec, address_expr):
    return (f"SMOKE = {spec['smoke_command']!r}\n"
            f"EPISODE = {one_line(spec['episode_command'])!r}\n"
            f"for placeholder in {tuple(spec['placeholders'])!r}:\n"
            f"    SMOKE = SMOKE.replace(placeholder, {address_expr})\n"
            f"    EPISODE = EPISODE.replace(placeholder, {address_expr})\n"
            "print('Local smoke (run in control/):')\n"
            "print(SMOKE)\n"
            "print('Local episode (run in control/):')\n"
            "print(EPISODE)")


def grpc_tunnel_cells(spec):
    tunnel = f'''from pyngrok import ngrok

# Explicitly request a TCP tunnel — gRPC needs HTTP/2 semantics that a
# default HTTP tunnel silently breaks (Pitfall 5 in 11-RESEARCH.md).
tunnel = ngrok.connect({spec['port']}, "tcp")
print(f"PolicyServer reachable at: {{tunnel.public_url}}")
# public_url looks like tcp://0.tcp.ngrok.io:12345 — strip the "tcp://"
# prefix; the local RobotClient's --server_address flag wants host:port.
SERVER_ADDRESS = tunnel.public_url.removeprefix("tcp://")
print(f"--server-address {{SERVER_ADDRESS}}")
''' + _placeholder_cell(spec, "SERVER_ADDRESS")
    return [
        md("tunnel-heading", f"## Open a TCP tunnel (not HTTP) to port {spec['port']}"),
        code("tunnel", tunnel),
        md("tunnel-fallback", CLOUDFLARED_NOTE),
    ]


# --- Launch ---------------------------------------------------------------------------------

def launch_md(spec):
    launch = spec["launch"]
    patterns = ", ".join(f"`{p}`" for p in spec["stop_patterns"])
    lines = [
        "## Launch",
        "",
        "The launch cell first stops any earlier server: every process whose command line "
        f"contains one of {patterns}, and whatever listens on port {spec['port']}.",
        "It then starts this notebook's server as a separate OS process (unbuffered, log at "
        f"`{spec['log']}`).",
        f"It only succeeds once **its own** still-running process owns port {spec['port']}.",
        "A stale server answering on the port is an error, never a successful launch (in the "
        "2026-10-18 live session a stale stock server made a diffusion launch look successful).",
    ]
    if launch.get("banner"):
        lines.append(f"For this patched launcher it also requires the banner `{launch['banner']}` "
                     "somewhere in the log, so the stock server can never pass for it.")
    return "\n".join(lines)


def launch_cell(spec):
    launch = spec["launch"]
    names = ["start_server", "stop_servers", "wait_until_ready"]
    if spec["lane"] == "grpc":
        names.append("gpu_report")
    lines = [f"{statement}\n" for statement in launch.get("imports", ())]
    lines.append(_helper_import(*names))
    lines.append(f"\nPORT = {spec['port']}\n{spec['log_var']} = {spec['log']!r}\n"
                 f"STOP_PATTERNS = {list(spec['stop_patterns'])!r}\n")
    if launch.get("constants"):
        lines.append(launch["constants"].strip("\n") + "\n")
    if launch.get("comment"):
        lines.append("\n" + launch["comment"].strip("\n") + "\n")
    if launch.get("pre"):
        lines.append("\n" + launch["pre"].strip("\n") + "\n")
    lines.append("\nstop_servers(PORT, STOP_PATTERNS)\n")
    if spec["lane"] == "grpc":
        lines.append("print('--- GPU memory before starting the server ---')\ngpu_report()\n")
    cwd = f",\n    cwd={launch['cwd']}" if launch.get("cwd") else ""
    lines.append(f"{spec['process_var']} = start_server(\n    {launch['argv']},\n"
                 f"    {spec['log_var']},\n    env_overrides={launch['env']}{cwd},\n)\n")
    if launch.get("before_wait"):
        lines.append(launch["before_wait"].strip("\n") + "\n")
    wait_args = [spec["process_var"], "PORT", spec["log_var"], f"timeout_s={launch['timeout_s']}"]
    if spec["tunnel"] == "http":
        wait_args.append(f"health_url='http://127.0.0.1:{spec['port']}/health'")
    if launch.get("ready"):
        wait_args.append(f"ready={launch['ready']}")
    if launch.get("banner"):
        wait_args.append(f"banner={launch['banner']!r}")
    lines.append(f"health = wait_until_ready(\n    {', '.join(wait_args[:3])},\n"
                 f"    {', '.join(wait_args[3:])},\n)\n")
    if launch.get("post"):
        lines.append(launch["post"].strip("\n") + "\n")
    return "".join(lines)


# --- HTTP lane ------------------------------------------------------------------------------

def http_tunnel_cells(spec):
    var = spec["tunnel_var"]
    head = spec.get("tunnel_code") or (
        f"from pyngrok import ngrok\n\n"
        f"{var} = ngrok.connect({spec['port']}, \"http\")  # plain JSON-over-HTTP\n"
        f"print(f\"{spec['endpoint_label']} endpoint: {{{var}.public_url}}\")")
    tunnel = head.strip("\n") + "\n" + _placeholder_cell(spec, f"{var}.public_url")
    heading = spec.get("tunnel_heading") or f"## Open an HTTP tunnel to port {spec['port']}"
    return [md("tunnel-heading", heading), code("tunnel", tunnel)]


# --- Diagnostics and teardown ---------------------------------------------------------------

def diagnostics_cells(spec):
    port, marker, log = spec["port"], spec["marker"], spec["log"]
    cells = [
        md("diagnostics-heading", f"""## Diagnostics

Run any of these after the launch cell, also from a restarted kernel: each cell re-declares
what it needs.
- Log: the tail of `{log}`, then every line matching `ERROR`, `Error`, `Traceback`,
  `Exception`, `OutOfMemory`, `out of memory`, `Killed` or `beginAllocateToPool`.
- Processes: every known server process on this runtime (stale servers from other notebooks
  show up here); fails unless `{marker}` owns port {port}.
- Package versions, GPU memory{', and `/health`' if spec['tunnel'] == 'http' else ''}."""),
        code("diag-log", _helper_import("log_report") + f"\nlog_report({log!r})"),
        code("diag-process", _helper_import("process_report")
             + f"\nprocess_report({port}, {marker!r}, globals().get({spec['process_var']!r}))"),
        code("diag-versions", spec["version_check"]),
        code("diag-gpu", _helper_import("gpu_report") + "\ngpu_report()"),
    ]
    if spec["tunnel"] == "http":
        cells.append(code("diag-health", _helper_import("health_report")
                          + f"\nhealth_report({port})"))
    return cells


def teardown_cells(spec):
    var = spec["tunnel_var"]
    server = "`PolicyServer`" if spec["lane"] == "grpc" else "server"
    cell = (f"# {spec['teardown_comment']}\n" + _helper_import("stop_server")
            + "from pyngrok import ngrok\n\n"
            f"if {var!r} in globals():\n"
            "    try:\n"
            f"        ngrok.disconnect({var}.public_url)\n"
            f"        print(f'Disconnected {{{var}.public_url}}')\n"
            "    except Exception as exc:\n"
            "        print(f'Tunnel already closed ({type(exc).__name__}: {exc})')\n"
            "ngrok.kill()\n"
            "print('ngrok agent stopped.')\n"
            f"stop_server(globals().get({spec['process_var']!r}), {spec['port']}, "
            f"{list(spec['stop_patterns'])!r})")
    return [md("teardown-heading", TEARDOWN_NOTE.format(server=server, port=spec["port"])),
            code("teardown", cell)]


# --- Notebook assembly ----------------------------------------------------------------------

def _extra(cells):
    built = []
    for slug, cell_type, text in cells:
        built.append(md(slug, text) if cell_type == "markdown" else code(slug, text))
    return built


def build_cells(spec):
    cells = header_cells(spec)
    cells.append(md("intro", spec["intro_md"]))
    if spec["lane"] == "grpc":
        cells += grpc_install_cells(spec)
    else:
        cells += _extra(spec["install"])
    cells += _extra(spec["extra_cells"])
    if spec["embed"]:
        cells.append(md("launch-notes", spec["launch_notes"]))
        for name in spec["embed"]:
            cells += embed_cells(name)
    cells.append(md("launch-heading", launch_md(spec)))
    cells.append(code("launch", launch_cell(spec)))
    cells += _extra(spec.get("after_launch", ()))
    if spec["lane"] == "grpc":
        cells += grpc_tunnel_cells(spec)
    else:
        cells += http_tunnel_cells(spec)
    cells.append(md("local-run", spec["local_md"]))
    cells += _extra(spec.get("after_local", ()))
    cells += diagnostics_cells(spec)
    cells += teardown_cells(spec)
    ids = [cell["id"] for cell in cells]
    duplicates = sorted({cell_id for cell_id in ids if ids.count(cell_id) > 1})
    if duplicates:
        raise ValueError(f"{spec['name']}: duplicate cell ids {duplicates}")
    return cells


def render_notebook(spec):
    colab = {"gpuType": spec["gpu_type"], "provenance": []}
    if spec["high_ram"]:
        colab["machine_shape"] = "hm"
    notebook = {
        "cells": build_cells(spec),
        "metadata": {
            "accelerator": "GPU",
            "colab": colab,
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


def render_readme():
    rows = []
    for spec in specs.NOTEBOOKS:
        tunnel = "ngrok TCP" if spec["tunnel"] == "tcp" else "ngrok HTTP"
        licences = ", ".join(f"`{repo}`" for repo in spec["licences"]) or "none"
        if spec["prerequisites"]:
            licences += f"; {spec['prerequisites']}"
        backend = (f"`{spec['backend']}`" if spec["lane"] != "depth"
                   else "(depth service, for `--depth-endpoint`)")
        rows.append(
            f"| [`{spec['name']}.ipynb`]({spec['name']}.ipynb) | {backend} | "
            f"{spec['lane_label']} | {spec['port']} → {tunnel} | {spec['gpu']} | {licences} | "
            f"`{spec['checkpoint']}` | `{spec['smoke_command']}` | "
            f"`{one_line(spec['episode_command'])}` |"
        )
    table = "\n".join(rows)
    return f"""<!-- Generated by scripts/build_colab_notebooks.py from scripts/colab_notebook_specs.py; edit those and regenerate. -->
# Colab policy-server notebooks

One self-contained notebook per Colab-hosted model or service. Each one installs its own
dependencies, writes every file it runs into `/content` (byte-identical to
`control/vla_bridge/`, checked by sha256), launches its server, proves that server owns its
port, opens its tunnel, prints the ready-to-paste local commands, and has diagnostic and
teardown cells. They replace the former combined policy-server notebook.

## How to run

1. Open the notebook from the VS Code Colab extension (or colab.research.google.com) on a
   **fresh runtime** of the GPU type in the table. Nothing has to be uploaded.
2. Use one notebook per runtime and run it **top to bottom**. The token cell reads
   `MyDrive/SoARM-Research/.hf_token` / `.ngrok_token` from Drive, or prompts for them.
3. Accept every gated licence in the table with the HF account whose token you use, and
   complete any listed sign-off, before running the notebook.
4. Run the local smoke command from `control/` (add `2>&1 | grep -v objc` to hide the
   harmless cv2/av duplicate-class warnings), then the episode command.
5. Run the teardown cell at the end of **every** session: an open tunnel into a Colab
   server is an elevation-of-privilege risk (T-11-08).

## Index

| notebook | `--backend` | lane | port → tunnel | GPU runtime | gated licences / sign-offs | default checkpoint | local smoke command | local episode command |
|---|---|---|---|---|---|---|---|---|
{table}

Gemini backends are not Colab-hosted (no notebook): `gemini-robotics` and `gemini-robotics-2`
call the Gemini Robotics-ER API directly from the laptop.

## Live-session facts (2026-10-18)

- **transformers pin.** Colab ships `transformers` 5.13.1. Since 5.9, `create_causal_mask`
  no longer takes `cache_position`, which lerobot 0.6.1's `pi_gemma` passes, so pi0/pi05
  fail every inference. Every gRPC notebook installs lerobot 0.6.1, then pins
  `transformers==5.5.4` (diffusion and groot-n17 also pin `diffusers==0.39.0`), and fails if
  pip changed `torch` (verified with `2.11.0+cu128`).
- **pi0 without torch.compile.** The pi0 checkpoint ships `compile_model: true`
  (max-autotune): a 465 s first inference, then `beginAllocateToPool: already recording to
  mempool_id` after every client reconnect. `grpc_pi0.ipynb` downloads it at its verified
  revision into `/content/ckpt/pi0_so101_nocompile` with `compile_model` set to false, and
  the local commands pass `--checkpoint /content/ckpt/pi0_so101_nocompile`. The pi05
  checkpoint already has `compile_model: false`.
- **Stale-server check.** A stock server left on port 5173 made a diffusion launch look
  successful. Every launch cell now stops known server processes and the port's listener
  first, and only succeeds when its own process owns the port (plus the `Patched ...` banner
  for the diffusion and groot-n17 launchers).
- **groot-n17 replacement.** The previous default checkpoint was removed from the Hub (HTTP
  401). The default is now `tsangb34/groot-n17-so101-soccer-red_bowl-40episodes`, verified
  live; details in `control/vla_bridge/backend_configs/groot_n17.json`.

## Camera mapping

The rig has two physical cameras: the IMX335 wrist camera (cv2 index 0, native 1920x1080) and
the AR0144 stereo camera (cv2 index 1, captured at 1600x600, 800x600 per half). The AR0144
delivers **one physical side-by-side stereo frame, not two independent feeds**. It is split
into left/right halves, giving three genuinely distinct real views:

| Checkpoint camera key | Real source | Crop |
| --- | --- | --- |
| `camera1` | IMX335 wrist | as-is, full 1920x1080 frame |
| `camera2` | AR0144 stereo — LEFT half | columns `[0:800]` of the 1600x600 frame |
| `camera3` | AR0144 stereo — RIGHT half | columns `[800:1600]` of the 1600x600 frame |

This is the SmolVLA wiring (`victorvanhalst/smolvla_so101_cube` expects `camera1`-`camera3`)
and the default when a backend declares no `camera_map`; the other backends map their own
camera names onto the sources `wrist`, `overhead_left` and `overhead_right` through
`backend_configs/<backend>.json` or `--camera-map`. Stock lerobot camera configs open each
named camera with its own `cv2` device and cannot crop one wider frame, so there were two
implementation options:

1. A wrapper camera process that opens cv2 index 1 once, splits each frame and republishes
   the halves as two separate virtual camera sources (most webcam drivers do not allow two
   concurrent opens of one index).
2. The **observation-dict patch**: open cv2 index 1 once in the local `RobotClient` process,
   crop each 1600x600 frame into `camera2`/`camera3` arrays in the observation dict before it
   is sent to the server. This patches the stock `robot_client.py` observation-building step
   rather than using CLI flags; `vla_bridge/robot_client.py` (`_wire_stereo_split_cameras()`)
   implements it.

## Regenerating

The notebooks and this README are generated. Edit `scripts/colab_notebook_specs.py`,
`scripts/build_colab_notebooks.py` or the embedded `control/vla_bridge/*.py` sources, then run
from the repo root:

```bash
python scripts/build_colab_notebooks.py          # rewrite every notebook + this README
python scripts/build_colab_notebooks.py --check  # exit 1 if anything drifted
```

`control/test_colab_notebooks.py` fails on any drift between the committed files, the generator
and the embedded sources.
"""


def render_all():
    """Every generated file: {path: text}. Deterministic."""
    names = [spec["name"] for spec in specs.NOTEBOOKS]
    if len(set(names)) != len(names):
        raise ValueError("duplicate notebook names")
    for spec in specs.NOTEBOOKS:
        banner = spec["launch"].get("banner")
        if banner and banner not in _source(spec["embed"][0]):
            raise ValueError(f"{spec['name']}: banner {banner!r} is not in {spec['embed'][0]}")
    files = {NOTEBOOK_DIR / f"{spec['name']}.ipynb": render_notebook(spec)
             for spec in specs.NOTEBOOKS}
    files[NOTEBOOK_DIR / "README.md"] = render_readme()
    return files


def write_all():
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for path, text in render_all().items():
        if not path.exists() or path.read_text() != text:
            path.write_text(text)
        print(f"Saved → {path.relative_to(ROOT)}")


def check():
    """Drifted, missing or stale files in the notebooks directory (empty = in sync)."""
    expected = render_all()
    problems = []
    for path, text in expected.items():
        if not path.exists():
            problems.append(f"missing: {path.relative_to(ROOT)}")
        elif path.read_text() != text:
            problems.append(f"drifted: {path.relative_to(ROOT)}")
    if NOTEBOOK_DIR.exists():
        for path in sorted(NOTEBOOK_DIR.iterdir()):
            if path not in expected:
                problems.append(f"stale extra file: {path.relative_to(ROOT)}")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if any generated file differs from the committed one")
    args = parser.parse_args(argv)
    if args.check:
        problems = check()
        for problem in problems:
            print(problem)
        if problems:
            print("Run: python scripts/build_colab_notebooks.py")
            return 1
        print("Colab notebooks are in sync.")
        return 0
    write_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
