"""Supply-chain pins for the Phase 15 MLLM provider SDKs (Plan 15-02).

These tests pin the supply-chain decision taken at the Plan 15-02 human
checkpoint: `anthropic==1.11.0` (Claude backend) and `openai==3.23.0` (GPT
backend and the HF Inference Providers router) are exactly pinned in
`control/requirements.txt` with a provenance comment block, and those exact
versions are what `control/.venv` has installed. The provider plans (15-06,
15-07, 15-08) verify their request kwargs against the signatures of these
installed SDKs, so a silent version drift would invalidate those checks.

No test here makes a network call; importing the SDKs only loads modules.
"""

import importlib
import importlib.metadata
import re
import sys
from pathlib import Path

import pytest

REQUIREMENTS_PATH = Path(__file__).resolve().parent / "requirements.txt"

SDK_PINS = {"anthropic": "1.11.0", "openai": "3.23.0"}

SDK_PROVENANCE = {
    "anthropic": {
        "repo": "github.com/anthropics/anthropic-sdk-python",
        "license": "MIT",
        "released": "2026-09-30",
        "wheel_sha256": "52f97b2c485cca7ac66058374f5073a3febb7e6849b16989c145d602a3efee21",
    },
    "openai": {
        "repo": "github.com/openai/openai-python",
        "license": "Apache-2.0",
        "released": "2026-10-01",
        "wheel_sha256": "7fbec2e50a05ac0fa858629505688e5998fd82245058390ac1b6d98960174bd5",
    },
}

TRANSITIVE_PINS = {
    "docstring_parser": "0.18.0",
    "httpx2": "2.13.1",
    "httpcore2": "2.13.1",
    "jiter": "0.17.0",
    "truststore": "0.10.4",
}

HUMAN_APPROVAL_DATE = "2026-10-02"

# name, optional [extras], then exactly one `==` pin with no further specifiers.
EXACT_PIN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9._,-]+\])?==[A-Za-z0-9._+!-]+$")
SHA256 = re.compile(r"\b[0-9a-f]{64}\b")


def _requirement_lines():
    lines = REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def _plan_15_02_block():
    """Return the comment block that starts at `# Plan 15-02` and the pins after it."""
    lines = REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("# Plan 15-02")]
    assert len(starts) == 1, "expected exactly one `# Plan 15-02` provenance block"
    block = []
    for line in lines[starts[0]:]:
        if not line.strip():
            break
        block.append(line)
    return block


def test_sdk_pins_present_as_exact_lines():
    lines = _requirement_lines()
    for name, version in SDK_PINS.items():
        assert f"{name}=={version}" in lines


def test_every_requirement_line_is_an_exact_pin():
    for line in _requirement_lines():
        assert EXACT_PIN.match(line), f"not an exact == pin: {line!r}"


def test_sdk_pins_follow_their_provenance_block():
    block = _plan_15_02_block()
    comments = [line for line in block if line.startswith("#")]
    pins = [line.strip() for line in block if not line.startswith("#")]
    assert comments, "provenance comments must precede the pins"
    assert pins == [f"{name}=={version}" for name, version in SDK_PINS.items()]
    assert block[: len(comments)] == comments, "comment lines must come before the pin lines"


@pytest.mark.parametrize("name", sorted(SDK_PROVENANCE))
def test_provenance_block_records_project_repo_license_date_and_sha(name):
    text = "\n".join(line for line in _plan_15_02_block() if line.startswith("#"))
    facts = SDK_PROVENANCE[name]
    assert f"`{name}`" in text, f"PyPI project name `{name}` missing"
    assert facts["repo"] in text
    assert facts["license"] in text
    assert facts["released"] in text
    assert facts["wheel_sha256"] in text


def test_provenance_block_has_one_distinct_sha256_per_sdk():
    text = "\n".join(line for line in _plan_15_02_block() if line.startswith("#"))
    digests = SHA256.findall(text)
    assert len(digests) == len(SDK_PINS)
    assert len(set(digests)) == len(SDK_PINS)


def test_provenance_block_records_transitives_and_human_approval():
    text = "\n".join(line for line in _plan_15_02_block() if line.startswith("#"))
    for name, version in TRANSITIVE_PINS.items():
        assert f"{name} {version}" in text, f"transitive {name} {version} not recorded"
    assert HUMAN_APPROVAL_DATE in text
    assert "approved" in text.lower()


@pytest.mark.parametrize("name,version", sorted(SDK_PINS.items()))
def test_installed_sdk_version_matches_pin(name, version):
    assert importlib.metadata.version(name) == version


@pytest.mark.parametrize("name", sorted(SDK_PINS))
def test_sdk_installed_in_the_running_venv(name):
    assert sys.prefix != sys.base_prefix, "tests must run inside control/.venv, not a base interpreter"
    location = Path(importlib.metadata.distribution(name).locate_file("")).resolve()
    assert Path(sys.prefix).resolve() in location.parents


def test_sdks_import_and_expose_their_clients():
    anthropic = importlib.import_module("anthropic")
    openai = importlib.import_module("openai")
    assert callable(anthropic.Anthropic)
    assert callable(openai.OpenAI)
