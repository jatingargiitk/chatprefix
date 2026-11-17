"""Load a model's chat template, special tokens, tokenizer and stop tokens.

A source is a Hugging Face Hub repo id (``Qwen/Qwen3-0.6B``), a local directory holding the
tokenizer files, or a single ``.jinja`` file.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

SPECIAL_TOKEN_KEYS = ("bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token")
HUB_FILES = (
    "tokenizer_config.json",
    "chat_template.jinja",
    "chat_template.json",
    "special_tokens_map.json",
    "generation_config.json",
    "tokenizer.json",
)


class SourceError(Exception):
    pass


@dataclass
class ModelFiles:
    name: str
    templates: Dict[str, str]  # name -> jinja source; "default" is the one apply_chat_template uses
    special_tokens: Dict[str, str] = field(default_factory=dict)
    tokenizer_path: Optional[Path] = None
    stop_token_ids: List[int] = field(default_factory=list)

    def template_for(self, tools: bool = False) -> str:
        if tools and "tool_use" in self.templates:
            return self.templates["tool_use"]
        return self.templates["default"]


def _token_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return value["content"]
    return None


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _templates_from(value: Any) -> Dict[str, str]:
    if isinstance(value, str):
        return {"default": value}
    if isinstance(value, list):
        out = {}
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("name"), str) and isinstance(item.get("template"), str):
                out[item["name"]] = item["template"]
        return out
    if isinstance(value, dict):
        return {k: v for k, v in value.items() if isinstance(v, str)}
    return {}


def load_dir(path: Path, name: Optional[str] = None) -> ModelFiles:
    cfg = _read_json(path / "tokenizer_config.json") or {}
    templates: Dict[str, str] = {}
    jinja_file = path / "chat_template.jinja"
    if jinja_file.is_file():
        templates["default"] = jinja_file.read_text(encoding="utf-8")
    else:
        ct_json = _read_json(path / "chat_template.json")
        if ct_json and "chat_template" in ct_json:
            templates = _templates_from(ct_json["chat_template"])
        elif "chat_template" in cfg:
            templates = _templates_from(cfg["chat_template"])
    if not templates:
        raise SourceError(f"no chat template found in {path} (looked for chat_template.jinja, chat_template.json, tokenizer_config.json)")
    if "default" not in templates:
        raise SourceError(f"{path}: several chat templates ({', '.join(sorted(templates))}) and none named 'default'")

    special: Dict[str, str] = {}
    for key in SPECIAL_TOKEN_KEYS:
        tok = _token_str(cfg.get(key))
        if tok:
            special[key] = tok
    stm = _read_json(path / "special_tokens_map.json") or {}
    for key in SPECIAL_TOKEN_KEYS:
        if key not in special:
            tok = _token_str(stm.get(key))
            if tok:
                special[key] = tok

    gen = _read_json(path / "generation_config.json") or {}
    eos = gen.get("eos_token_id")
    stop_ids = [eos] if isinstance(eos, int) else [i for i in (eos or []) if isinstance(i, int)]

    tok_path = path / "tokenizer.json"
    return ModelFiles(
        name=name or str(path),
        templates=templates,
        special_tokens=special,
        tokenizer_path=tok_path if tok_path.is_file() else None,
        stop_token_ids=stop_ids,
    )


def load_jinja(path: Path, special_tokens: Optional[Dict[str, str]] = None) -> ModelFiles:
    return ModelFiles(name=str(path), templates={"default": path.read_text(encoding="utf-8")}, special_tokens=dict(special_tokens or {}))


REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def _cache_root() -> Path:
    base = os.environ.get("CHATPREFIX_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "chatprefix")
    return Path(base)


def fetch_hub(repo: str, revision: str = "main", tokenizer: bool = True, endpoint: Optional[str] = None) -> Path:
    """Download the tokenizer files of a Hub repo into the cache and return the directory."""
    endpoint = (endpoint or os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
    token = os.environ.get("HF_TOKEN")
    dest = _cache_root() / repo.replace("/", "--") / revision
    dest.mkdir(parents=True, exist_ok=True)
    pinned = re.fullmatch(r"[0-9a-f]{40}", revision) is not None
    for fname in HUB_FILES:
        if fname == "tokenizer.json" and not tokenizer:
            continue
        target = dest / fname
        missing_marker = dest / (fname + ".missing")
        if pinned and (target.is_file() or missing_marker.is_file()):
            continue
        url = f"{endpoint}/{repo}/resolve/{revision}/{fname}"
        req = urllib.request.Request(url, headers={"User-Agent": "chatprefix"})
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                if target.exists():
                    target.unlink()
                missing_marker.touch()
                continue
            if e.code in (401, 403):
                raise SourceError(f"{repo}: HTTP {e.code} for {fname} (gated or private repo? set HF_TOKEN)") from e
            raise SourceError(f"{repo}: HTTP {e.code} for {fname}") from e
        except urllib.error.URLError as e:
            raise SourceError(f"{repo}: could not reach {endpoint}: {e.reason}") from e
        target.write_bytes(data)
        if missing_marker.exists():
            missing_marker.unlink()
    return dest


def load(source: str, revision: str = "main", tokenizer: bool = True) -> ModelFiles:
    p = Path(source)
    if p.is_dir():
        files = load_dir(p)
    elif p.is_file():
        files = load_jinja(p)
    elif REPO_ID.match(source):
        rev_label = "" if revision == "main" else f"@{revision[:12]}"
        files = load_dir(fetch_hub(source, revision, tokenizer=tokenizer), name=f"{source}{rev_label}")
    else:
        raise SourceError(f"{source!r} is not a directory, a file, or a Hub repo id like org/name")
    if not tokenizer:
        files.tokenizer_path = None
    return files

