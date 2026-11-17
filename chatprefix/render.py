"""Render a chat template the way transformers' apply_chat_template(tokenize=False) does.

Mirrors transformers/utils/chat_template_utils.py: an ImmutableSandboxedEnvironment with
trim_blocks/lstrip_blocks, the loopcontrols extension, a `{% generation %}` tag, a tojson filter that
does not HTML-escape, raise_exception, strftime_now, and the reasoning-field aliasing. Keys whose value
is None are dropped from messages first, like apply_chat_template does.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

REASONING_FIELDS = ("reasoning_content", "thinking", "reasoning")


class TemplateRenderError(Exception):
    """The template raised (raise_exception) or failed to render."""


class _GenerationTag(jinja2.ext.Extension):
    # `{% generation %}...{% endgeneration %}` marks assistant tokens for return_assistant_tokens_mask.
    # For plain rendering it is a pass-through, which is what transformers does when the tracker is off.
    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return jinja2.nodes.CallBlock(self.call_method("_passthrough"), [], [], body).set_lineno(lineno)

    def _passthrough(self, caller):
        return caller()


def _raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


class _TemplateDict:
    def __call__(self, *args, **kwargs):
        return dict(*args, **kwargs)

    def fromkeys(self, *args, **kwargs):
        return dict.fromkeys(*args, **kwargs)


def compile_template(source: str, now: Optional[datetime] = None) -> jinja2.Template:
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[_GenerationTag, jinja2.ext.loopcontrols]
    )
    fixed = now

    def strftime_now(fmt):
        return (fixed or datetime.now()).strftime(fmt)

    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise_exception
    env.globals["strftime_now"] = strftime_now
    env.globals["dict"] = _TemplateDict()
    return env.from_string(source)


def _alias_reasoning(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for m in messages:
        reasoning = next((m[k] for k in REASONING_FIELDS if m.get(k) is not None), None)
        if reasoning is not None:
            m = {"thinking": reasoning, **m, "reasoning_content": reasoning}
        out.append(m)
    return out


def prepare_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = [{k: v for k, v in json.loads(json.dumps(m)).items() if v is not None} for m in messages]
    return _alias_reasoning(cleaned)


def render(
    template: jinja2.Template,
    messages: List[Dict[str, Any]],
    *,
    special_tokens: Optional[Dict[str, str]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    add_generation_prompt: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    kwargs: Dict[str, Any] = dict(special_tokens or {})
    kwargs.update(extra or {})
    try:
        return template.render(
            messages=prepare_messages(messages),
            tools=json.loads(json.dumps(tools)) if tools is not None else None,
            documents=None,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
    except jinja2.exceptions.TemplateError as e:
        raise TemplateRenderError(str(e)) from e
    except Exception as e:  # templates can fail in many ways (e.g. indexing None)
        raise TemplateRenderError(f"{type(e).__name__}: {e}") from e
