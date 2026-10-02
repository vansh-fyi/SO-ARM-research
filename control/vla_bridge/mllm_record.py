"""Per-call audit layer for MLLM backends (Phase 15, MLLM-06, PIPE-02, D-20..D-22).

What this is
============
Helpers that turn one MLLM provider call into an auditable, secret-free
artifact set:

  - `redact_text` / `redact`: the ONE redaction pass applied to every record,
    sidecar text and error string (D-00b, D-22, T-15-10). Every occurrence of
    each secret value (and its JSON-escaped form) becomes `<redacted>`, and
    the VALUE of any `Authorization` / `x-api-key` / `api-key` / `api_key` /
    `apikey` key (any case) is replaced regardless of its content, in dicts
    and in header-style free text alike.
  - `to_json_safe`: SDK objects (`model_dump(mode="json")`), numpy scalars and
    arrays, paths and bytes become plain strict JSON (research Pitfall 6).
  - `estimate_cost_usd`: cost from provider token counts and a dated per-model
    price table taken from config (D-22c).
  - `write_call_sidecar`: the path-safe atomic writer for the per-call sidecar
    directory (D-22d, ASVS V5).
  - `CallRecorder`: plugs into `MllmActionSourceBase(recorder=...)` by
    dependency injection. Its `finalize()` adds `cost_usd_est` +
    `price_table`, writes the sidecar, sets `sidecar_dir`, `sidecar_files` and
    `images[i].path`, and never raises: a disk or naming failure becomes a
    redacted `sidecar_error` in the record instead of reaching the control loop.

Sidecar layout (D-22d)
----------------------
One directory per provider call, next to `episode.jsonl`::

    <out>/mllm_calls/<chunk_id:04d>/
        prompt.txt           system prompt + user text + image list (send order)
        request_meta.json    settings requested/applied/omitted, hashes, schema, ...
        response_raw.json    verbatim provider payload, secret-redacted
        thinking.txt         FULL native thinking text (the record keeps a capped copy)
        depth.npy  aligned_left.png  depth_heatmap.png  depth_grid.txt   (from the collector)
        images/<role>.<png|jpg>   the exact image bytes sent

Large payloads live here, never inline in `episode.jsonl`. Paths are built only
from an integer chunk id and fixed whitelists (`SIDECAR_FILE_NAMES`,
`SIDECAR_IMAGE_ROLES`, `IMAGE_EXTENSIONS`); nothing a model returns can choose
a directory or filename. Every name is validated before anything is written.

numpy is imported only inside functions, so importing this module is cheap.
"""

import copy
import hashlib
import json
import math
import numbers
import re
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path, PurePath

REDACTED = "<redacted>"
SECRET_HEADER_KEYS = ("authorization", "x-api-key", "api-key", "api_key", "apikey")
MIN_SECRET_LEN = 4
MAX_REPR_CHARS = 2000
MAX_ERROR_CHARS = 2000
_MAX_DEPTH = 64

SIDECAR_FILE_NAMES = (
    "prompt.txt",
    "request_meta.json",
    "response_raw.json",
    "thinking.txt",
    "depth.npy",
    "aligned_left.png",
    "depth_heatmap.png",
    "depth_grid.txt",
)
SIDECAR_IMAGE_ROLES = ("wrist", "left", "right", "depth")
IMAGE_EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg"}
IMAGES_SUBDIR = "images"

_HEADER_KEY_SET = frozenset(SECRET_HEADER_KEYS) | {"x_api_key", "proxy-authorization"}
# Header-style "key: value" / "key=value" / '"key": "value"' in free text.
_HEADER_VALUE_RE = re.compile(
    r"(?i)(?P<key>\b(?:proxy-)?authorization|\bx[-_]api[-_]key|\bapi[-_]?key)"
    r"(?P<sep>[\"']?\s*[:=]\s*[\"']?)"
    r"(?P<value>(?:bearer\s+|basic\s+)?[^\s\"',;}\]]+)"
)


# --- Redaction (D-00b, D-22, T-15-10) ---------------------------------------------------


def _usable_secrets(secrets) -> list[str]:
    usable = {s for s in (secrets or ()) if isinstance(s, str) and len(s) >= MIN_SECRET_LEN}
    variants = set()
    for secret in usable:
        variants.add(secret)
        variants.add(json.dumps(secret)[1:-1])
        variants.add(json.dumps(secret, ensure_ascii=False)[1:-1])
    # Longest first, so a secret containing another is removed whole.
    return sorted((v for v in variants if len(v) >= MIN_SECRET_LEN), key=len, reverse=True)


def _redact_text(text: str, usable: list[str]) -> str:
    for secret in usable:
        if secret in text:
            text = text.replace(secret, REDACTED)
    return _HEADER_VALUE_RE.sub(lambda m: f"{m.group('key')}{m.group('sep')}{REDACTED}", text)


def redact_text(text, secrets) -> str:
    """`text` with every secret (length >= 4) and every header-style secret
    value replaced by `<redacted>`. Empty or too-short secrets are ignored."""
    if not isinstance(text, str):
        text = str(text)
    return _redact_text(text, _usable_secrets(secrets))


def _is_secret_header_key(key) -> bool:
    return isinstance(key, str) and key.strip().lower() in _HEADER_KEY_SET


def _redact(obj, usable: list[str], depth: int = 0):
    if depth > _MAX_DEPTH:
        return REDACTED
    if isinstance(obj, str):
        return _redact_text(obj, usable)
    if isinstance(obj, Mapping):
        out = {}
        for key, value in obj.items():
            new_key = _redact_text(key, usable) if isinstance(key, str) else key
            out[new_key] = REDACTED if _is_secret_header_key(key) else _redact(value, usable, depth + 1)
        return out
    if isinstance(obj, list):
        return [_redact(item, usable, depth + 1) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_redact(item, usable, depth + 1) for item in obj)
    return obj


def redact(obj, secrets):
    """Walk dicts, lists and tuples; redact secret substrings in every string
    (keys included) and the whole VALUE of any secret-header key."""
    return _redact(obj, _usable_secrets(secrets))


def error_text(error, secrets) -> str | None:
    """A redacted, truncated `"<Type>: <message>"` for an exception (or the
    redacted string itself); None when there is no error."""
    if error is None:
        return None
    if isinstance(error, BaseException):
        text = f"{type(error).__name__}: {error}"
    else:
        text = str(error)
    return redact_text(text, secrets)[:MAX_ERROR_CHARS]


# --- JSON safety and hashing ------------------------------------------------------------


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(bytes(data)).hexdigest()


def _truncated_repr(obj) -> str:
    try:
        text = repr(obj)
    except Exception as exc:  # a broken __repr__ must not break logging
        text = f"<unrepresentable {type(obj).__name__}: {type(exc).__name__}>"
    if len(text) > MAX_REPR_CHARS:
        text = text[:MAX_REPR_CHARS] + "...(truncated)"
    return text


def to_json_safe(obj, _depth: int = 0):
    """Plain strict-JSON version of `obj` (json.dumps(..., allow_nan=False) works).

    SDK objects -> `model_dump(mode="json")`; numpy scalars -> `item()`; numpy
    arrays -> `tolist()`; paths -> str; bytes -> `{"bytes": n, "sha256": hex}`;
    non-finite floats -> their str; anything unknown -> a truncated repr.
    """
    if _depth > _MAX_DEPTH:
        return _truncated_repr(obj)
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, numbers.Integral) and type(obj).__module__ == "builtins":
        return int(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, (bytes, bytearray, memoryview)):
        data = bytes(obj)
        return {"bytes": len(data), "sha256": sha256_hex(data)}
    if isinstance(obj, PurePath):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Mapping):
        return {str(key): to_json_safe(value, _depth + 1) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_json_safe(item, _depth + 1) for item in obj]

    if type(obj).__module__.split(".")[0] == "numpy":
        import numpy as np

        if isinstance(obj, np.ndarray):
            return to_json_safe(obj.tolist(), _depth + 1)
        if isinstance(obj, np.generic):
            return to_json_safe(obj.item(), _depth + 1)

    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            return to_json_safe(model_dump(mode="json"), _depth + 1)
        except Exception:
            pass
    if isinstance(obj, numbers.Integral):
        return int(obj)
    if isinstance(obj, numbers.Real):
        return to_json_safe(float(obj), _depth + 1)
    return _truncated_repr(obj)


# --- Cost (D-22c) ----------------------------------------------------------------------------


def _count(value):
    """A non-negative token count, or None."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return value


def _price(value):
    """A non-negative finite USD-per-million-tokens price, or None."""
    return _count(value)


def estimate_cost_usd(usage, prices) -> float | None:
    """Estimated USD cost of one call.

    Billing rule: `(input_tokens - cached_input_tokens)` at `input_per_mtok`,
    cached tokens at `cached_input_per_mtok` (falls back to the input price
    when absent), `output_tokens` at `output_per_mtok`. Output tokens already
    include reasoning tokens, so reasoning is never added again. An unknown
    (None) cached count means nothing is known to be cached. Returns None when
    input/output counts or the two required prices are missing or invalid.
    """
    if not isinstance(usage, Mapping) or not isinstance(prices, Mapping):
        return None
    input_tokens = _count(usage.get("input_tokens"))
    output_tokens = _count(usage.get("output_tokens"))
    if input_tokens is None or output_tokens is None:
        return None
    raw_cached = usage.get("cached_input_tokens")
    if raw_cached is None:
        cached = 0
    else:
        cached = _count(raw_cached)
        if cached is None or cached > input_tokens:
            return None

    input_price = _price(prices.get("input_per_mtok"))
    output_price = _price(prices.get("output_per_mtok"))
    if input_price is None or output_price is None:
        return None
    cached_price = input_price
    if prices.get("cached_input_per_mtok") is not None:
        cached_price = _price(prices.get("cached_input_per_mtok"))
        if cached_price is None:
            return None

    total = ((input_tokens - cached) * input_price + cached * cached_price
             + output_tokens * output_price)
    return total / 1e6


# --- Sidecar writer (D-22d, ASVS V5, T-15-11) ---------------------------------------------


def _validate_chunk_id(chunk_id) -> None:
    if type(chunk_id) is not int or chunk_id < 0:
        raise ValueError(f"sidecar chunk_id must be a non-negative int, got {type(chunk_id).__name__}")


def _items(mapping, what: str) -> list[tuple]:
    if mapping is None:
        return []
    if isinstance(mapping, Mapping):
        return list(mapping.items())
    try:
        pairs = [tuple(pair) for pair in mapping]
    except TypeError:
        raise ValueError(f"sidecar {what} must be a mapping or (name, content) pairs") from None
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError(f"sidecar {what} must be (name, content) pairs")
    return pairs


def _check_content(content, what: str) -> None:
    if not isinstance(content, (str, bytes, bytearray, memoryview)):
        raise TypeError(f"sidecar {what} content must be str or bytes, got {type(content).__name__}")


def _atomic_write(path: Path, content) -> None:
    """Temporary sibling then `Path.replace` (io_logger.write_depth_result idiom)."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        if isinstance(content, str):
            temporary.write_text(content, encoding="utf-8")
        else:
            temporary.write_bytes(bytes(content))
        temporary.replace(path)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def write_call_sidecar(root, chunk_id, *, files, images, image_media_type) -> dict:
    """Write one call's sidecar files under `<root>/<chunk_id:04d>/`.

    `files`: whitelisted name -> str (written verbatim, UTF-8) or bytes.
    `images`: role -> bytes (or (role, bytes) pairs), written to
    `images/<role>.<ext>` with ext from `image_media_type`.

    Every chunk id, name, role, media type and content type is validated
    BEFORE any directory or file is created (ValueError / TypeError).

    Returns `{"sidecar_dir": "<root name>/<NNNN>", "files": [names relative to
    the sidecar dir], "images": {role: path relative to root's parent}}`; the
    image paths join directly onto the episode output directory.
    """
    _validate_chunk_id(chunk_id)
    file_items = _items(files, "files")
    image_items = _items(images, "images")

    seen = set()
    for name, content in file_items:
        if not isinstance(name, str) or name not in SIDECAR_FILE_NAMES:
            raise ValueError("sidecar file name is not whitelisted")
        if name in seen:
            raise ValueError(f"duplicate sidecar file name {name}")
        seen.add(name)
        _check_content(content, name)

    extension = None
    seen_roles = set()
    for role, content in image_items:
        if not isinstance(role, str) or role not in SIDECAR_IMAGE_ROLES:
            raise ValueError("sidecar image role is not whitelisted")
        if role in seen_roles:
            raise ValueError(f"duplicate sidecar image role {role}")
        seen_roles.add(role)
        if not isinstance(content, (bytes, bytearray, memoryview)):
            raise TypeError(f"sidecar image {role} must be bytes, got {type(content).__name__}")
    if image_items:
        extension = IMAGE_EXTENSIONS.get(image_media_type) if isinstance(image_media_type, str) else None
        if extension is None:
            raise ValueError("sidecar image media type is not whitelisted")

    root = Path(root)
    dir_name = f"{chunk_id:04d}"
    chunk_dir = root / dir_name
    sidecar_dir = f"{root.name}/{dir_name}"

    written = []
    if file_items:
        chunk_dir.mkdir(parents=True, exist_ok=True)
    for name, content in file_items:
        _atomic_write(chunk_dir / name, content)
        written.append(name)

    image_paths = {}
    if image_items:
        images_dir = chunk_dir / IMAGES_SUBDIR
        images_dir.mkdir(parents=True, exist_ok=True)
        for role, content in image_items:
            filename = f"{role}.{extension}"
            _atomic_write(images_dir / filename, content)
            image_paths[role] = f"{sidecar_dir}/{IMAGES_SUBDIR}/{filename}"

    return {"sidecar_dir": sidecar_dir, "files": written, "images": image_paths}


# --- The recorder hook (D-20..D-22) ------------------------------------------------------------

# Names the recorder itself writes; a collector can never replace them.
RECORDER_FILE_NAMES = ("prompt.txt", "request_meta.json", "response_raw.json", "thinking.txt")
COLLECTOR_FILE_NAMES = tuple(name for name in SIDECAR_FILE_NAMES if name not in RECORDER_FILE_NAMES)
_META_RECORD_KEYS = (
    "backend", "provider", "model_requested", "model_returned", "request_id", "response_id",
    "sdk", "prompt_template_version", "prompt_sha256", "prompt_template_sha256", "schema_sha256",
    "include_urdf", "chunk_len", "step_dt_s", "depth", "capture", "prev_chunk_tick_period_s",
)


def _prompt_text(request) -> str:
    lines = [
        request.system_prompt,
        "",
        "===== USER MESSAGE =====",
        "",
        request.user_text,
        "",
        "===== IMAGES (send order; each sent after the label 'Image: <role>', "
        f"before the user text; {request.image_media_type}) =====",
    ]
    for role, data in request.images:
        lines.append(f"{role}: sha256 {sha256_hex(data)}, {len(data)} bytes")
    return "\n".join(lines) + "\n"


class CallRecorder:
    """Recorder for `MllmActionSourceBase(recorder=...)`: adds cost and writes
    the per-call sidecar directory under `root` (normally `<out>/mllm_calls`).

    `prices` is the dated per-model price table from the backend config
    (`input_per_mtok`, `output_per_mtok`, optional `cached_input_per_mtok`,
    plus any provenance keys such as `verified_on`); a deep-copied snapshot
    goes into every record. `finalize()` never raises.
    """

    def __init__(self, root, *, secrets=(), prices=None):
        self.root = Path(root)
        self._secrets = tuple(s for s in (secrets or ()) if isinstance(s, str) and s)
        self.prices = copy.deepcopy(prices) if prices is not None else None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(root={str(self.root)!r}, prices={self.prices is not None})"

    def _request_meta(self, record: dict, chunk_id: int, request, error) -> dict:
        meta = {"chunk_id": chunk_id}
        meta.update({key: record.get(key) for key in _META_RECORD_KEYS})
        meta.update({
            "schema_name": request.schema_name,
            "schema": request.schema,
            "settings": record.get("settings"),
            "image_roles": [role for role, _data in request.images],
            "image_media_type": request.image_media_type,
            "images": record.get("images"),
            "usage": record.get("usage"),
            "stop": record.get("stop"),
            "cost_usd_est": record.get("cost_usd_est"),
            "price_table": record.get("price_table"),
            "error": error_text(error, self._secrets),
        })
        return meta

    def _json(self, obj) -> str:
        return json.dumps(redact(to_json_safe(obj), self._secrets), indent=2, allow_nan=False) + "\n"

    def _sidecar_files(self, record: dict, chunk_id: int, request, result, inputs, error) -> dict:
        files = {
            "prompt.txt": redact_text(_prompt_text(request), self._secrets),
            "request_meta.json": self._json(self._request_meta(record, chunk_id, request, error)),
        }
        if result is not None and getattr(result, "raw_payload", None) is not None:
            payload = result.raw_payload
            if isinstance(payload, str):  # the provider wire body, verbatim
                files["response_raw.json"] = redact_text(payload, self._secrets)
            else:
                files["response_raw.json"] = self._json(payload)
        thinking = getattr(result, "native_thinking_text", None) if result is not None else None
        if isinstance(thinking, str) and thinking:
            files["thinking.txt"] = redact_text(thinking, self._secrets)
        extra = getattr(inputs, "sidecar_files", None) if inputs is not None else None
        if isinstance(extra, Mapping):
            for name, content in extra.items():
                if not isinstance(name, str) or name not in COLLECTOR_FILE_NAMES:
                    continue  # never written: not whitelisted, or a recorder-owned name
                files[name] = redact_text(content, self._secrets) if isinstance(content, str) else content
        return files

    def finalize(self, record, *, chunk_id, request, result, inputs, error) -> dict:
        """The augmented, redacted record (cost, price snapshot, sidecar paths)."""
        try:
            record = copy.deepcopy(record)
        except Exception:
            record = dict(record)
        try:
            # (1) cost + dated price-table snapshot (D-22c)
            record["cost_usd_est"] = estimate_cost_usd(record.get("usage"), self.prices)
            record["price_table"] = copy.deepcopy(self.prices)
            record["sidecar_dir"] = None
            record["sidecar_files"] = []
            # (2)+(3) sidecar files and the exact images sent (D-22d)
            try:
                files = self._sidecar_files(record, chunk_id, request, result, inputs, error)
                written = write_call_sidecar(
                    self.root, chunk_id, files=files, images=list(request.images),
                    image_media_type=request.image_media_type,
                )
                record["sidecar_dir"] = written["sidecar_dir"]
                record["sidecar_files"] = written["files"]
                for image in record.get("images") or []:
                    if isinstance(image, dict):
                        image["path"] = written["images"].get(image.get("role"))
            except (OSError, ValueError, TypeError) as exc:  # (4) disk full / bad name: never reach the loop
                record["sidecar_error"] = error_text(exc, self._secrets)
        except Exception as exc:  # defence in depth: finalize never raises
            record["recorder_error"] = error_text(exc, self._secrets)
        # (5) one last redaction pass over the whole record
        return redact(to_json_safe(record), self._secrets)


__all__ = [
    "COLLECTOR_FILE_NAMES",
    "CallRecorder",
    "RECORDER_FILE_NAMES",
    "IMAGE_EXTENSIONS",
    "IMAGES_SUBDIR",
    "MAX_ERROR_CHARS",
    "REDACTED",
    "SECRET_HEADER_KEYS",
    "SIDECAR_FILE_NAMES",
    "SIDECAR_IMAGE_ROLES",
    "error_text",
    "estimate_cost_usd",
    "redact",
    "redact_text",
    "sha256_hex",
    "to_json_safe",
    "write_call_sidecar",
]
