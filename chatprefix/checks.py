"""The checks. Each returns Findings; FAIL means the template breaks a property callers rely on."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from chatprefix.render import TemplateRenderError, compile_template, render
from chatprefix.source import ModelFiles

FAIL, WARN, INFO, OK = "FAIL", "WARN", "INFO", "ok"

DEFAULT_CONVERSATION: List[Dict[str, Any]] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
    {"role": "user", "content": "And of Italy?"},
    {"role": "assistant", "content": "Rome."},
]

REASONING = ["The user wants the capital of France.", "Now the capital of Italy."]

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}

GENERATION_TAG = re.compile(r"\{%-?\s*generation\s*-?%\}")


def with_reasoning(conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    conv = copy.deepcopy(conversation)
    n = 0
    for m in conv:
        if m.get("role") == "assistant" and "reasoning_content" not in m:
            m["reasoning_content"] = REASONING[n] if n < len(REASONING) else f"Reasoning for turn {n + 1}."
            n += 1
    return conv


@dataclass
class Finding:
    check: str
    status: str
    message: str
    details: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"check": self.check, "status": self.status, "message": self.message, "details": self.details}


def _snip(s: str, limit: int = 60) -> str:
    if len(s) > limit:
        s = s[:limit] + "..."
    return repr(s)


def divergence(earlier: str, later: str) -> List[str]:
    i = 0
    n = min(len(earlier), len(later))
    while i < n and earlier[i] == later[i]:
        i += 1
    head = earlier[max(0, i - 30):i]
    return [
        f"first difference at char {i}, after {_snip(head)}",
        f"shorter render has: {_snip(earlier[i:])}",
        f"longer render has:  {_snip(later[i:])}",
    ]


class Checker:
    def __init__(
        self,
        files: ModelFiles,
        conversation: Optional[List[Dict[str, Any]]] = None,
        now: Optional[datetime] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.files = files
        self.conversation = conversation if conversation is not None else DEFAULT_CONVERSATION
        self.now = now
        self.extra = extra or {}
        self.template = compile_template(files.template_for(tools=False), now=now)
        self.tool_template = compile_template(files.template_for(tools=True), now=now)
        self._tokenizer = None

    def render(self, messages, add_generation_prompt=False, tools=None) -> str:
        tpl = self.tool_template if tools else self.template
        return render(
            tpl,
            messages,
            special_tokens=self.files.special_tokens,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            extra=self.extra,
        )

    # -- prefix stability -------------------------------------------------------------------
    def check_prefix(self) -> List[Finding]:
        conv = self.conversation
        out: List[Finding] = []
        skipped = []
        for k in range(1, len(conv)):
            try:
                shorter = self.render(conv[:k])
            except TemplateRenderError as e:
                skipped.append(f"first {k} message(s) don't render on their own: {e}")
                continue
            try:
                longer = self.render(conv[: k + 1])
            except TemplateRenderError as e:
                skipped.append(f"first {k + 1} message(s) don't render: {e}")
                continue
            if not longer.startswith(shorter):
                out.append(
                    Finding(
                        "prefix",
                        FAIL,
                        f"adding message {k + 1} ({conv[k].get('role')}) rewrites the render of messages 1-{k}",
                        divergence(shorter, longer),
                    )
                )
        if not out:
            n = len(conv) - 1 - len(skipped)
            msg = f"each of {n} longer conversation(s) starts with the render of the shorter one"
            out.append(Finding("prefix", OK if n else INFO, msg if n else "nothing could be compared", skipped))
        elif skipped:
            out[-1].details.extend(skipped)
        return out

    # -- generation prompt -----------------------------------------------------------------
    def check_genprompt(self) -> List[Finding]:
        conv = self.conversation
        out: List[Finding] = []
        compared = 0
        for i, m in enumerate(conv):
            if m.get("role") != "assistant" or i == 0:
                continue
            try:
                prompt = self.render(conv[:i], add_generation_prompt=True)
                full = self.render(conv[: i + 1])
            except TemplateRenderError as e:
                out.append(Finding("genprompt", INFO, f"could not render around message {i + 1}: {e}"))
                continue
            compared += 1
            if not full.startswith(prompt):
                out.append(
                    Finding(
                        "genprompt",
                        FAIL,
                        f"the generation prompt before message {i + 1} is not how the template writes that turn",
                        divergence(prompt, full),
                    )
                )
        if not any(f.status == FAIL for f in out) and compared:
            out.insert(0, Finding("genprompt", OK, f"add_generation_prompt=True matches the rendered turn for {compared} assistant message(s)"))
        return out

    # -- system role -------------------------------------------------------------------------
    def check_system(self) -> List[Finding]:
        conv = self.conversation
        system = next((m for m in conv if m.get("role") == "system"), None)
        if system is None:
            return [Finding("system", INFO, "conversation has no system message; not checked")]
        try:
            text = self.render(conv)
        except TemplateRenderError as e:
            without = [m for m in conv if m.get("role") != "system"]
            try:
                self.render(without)
            except TemplateRenderError:
                return [Finding("system", INFO, f"conversation does not render: {e}")]
            return [Finding("system", WARN, f"template rejects a system message: {e}")]
        content = system.get("content")
        if isinstance(content, str) and content and content not in text:
            return [Finding("system", WARN, "system message is silently dropped from the render")]
        return [Finding("system", OK, "system message is rendered")]

    # -- tools ---------------------------------------------------------------------------------
    def check_tools(self) -> List[Finding]:
        msgs = [m for m in self.conversation[:2] if m.get("role") in ("system", "user")] or [
            {"role": "user", "content": "What is the weather in Paris?"}
        ]
        name = WEATHER_TOOL["function"]["name"]
        try:
            text = self.render(msgs, add_generation_prompt=True, tools=[WEATHER_TOOL])
        except TemplateRenderError as e:
            return [Finding("tools", WARN, f"template raises when tools= is passed: {e}")]
        if name not in text:
            return [Finding("tools", WARN, "tools= is silently ignored (the tool name never appears in the render)")]
        return [Finding("tools", OK, "tools= is rendered into the prompt")]

    # -- assistant mask ------------------------------------------------------------------------
    def check_mask(self) -> List[Finding]:
        if GENERATION_TAG.search(self.files.template_for()):
            return [Finding("mask", OK, "template has {% generation %} blocks, so return_assistant_tokens_mask works")]
        return [
            Finding(
                "mask",
                INFO,
                "no {% generation %} block: return_assistant_tokens_mask (and TRL assistant_only_loss) can't be used",
            )
        ]

    # -- tokenizer-based checks -----------------------------------------------------------------
    def tokenizer(self):
        if self._tokenizer is None and self.files.tokenizer_path is not None:
            from tokenizers import Tokenizer

            self._tokenizer = Tokenizer.from_file(str(self.files.tokenizer_path))
        return self._tokenizer

    def check_bos(self) -> List[Finding]:
        bos = self.files.special_tokens.get("bos_token")
        tok = self.tokenizer()
        if tok is None:
            return [Finding("bos", INFO, "no tokenizer.json; BOS handling not checked")]
        if not bos:
            return [Finding("bos", OK, "no bos_token configured")]
        bos_id = tok.token_to_id(bos)
        try:
            text = self.render(self.conversation)
        except TemplateRenderError as e:
            return [Finding("bos", INFO, f"conversation does not render: {e}")]
        ids = tok.encode(text).ids  # add_special_tokens=True, like tokenizer(text)
        starts = text.startswith(bos)
        if bos_id is not None and starts and ids[:2] == [bos_id, bos_id]:
            return [
                Finding(
                    "bos",
                    WARN,
                    f"double BOS: the template writes {bos!r} and the tokenizer adds another",
                    [
                        f"tokenizer(text) starts with ids {ids[:3]}; {bos!r} is id {bos_id}",
                        "apply_chat_template(tokenize=True) is fine (it passes add_special_tokens=False); "
                        "tokenizing the rendered string yourself is not",
                    ],
                )
            ]
        if bos_id is not None and not starts and ids[:1] == [bos_id]:
            return [
                Finding(
                    "bos",
                    WARN,
                    f"the template doesn't write {bos!r} but the tokenizer adds it",
                    ["apply_chat_template(tokenize=True) gives no BOS while tokenizer(text) gives one"],
                )
            ]
        msg = f"the template writes {bos!r} and the tokenizer does not add another" if starts else f"{bos!r} is never doubled"
        return [Finding("bos", OK, msg)]

    def stop_strings(self) -> List[str]:
        stops = []
        eos = self.files.special_tokens.get("eos_token")
        if eos:
            stops.append(eos)
        tok = self.tokenizer()
        if tok is not None:
            for i in self.files.stop_token_ids:
                s = tok.id_to_token(i)
                if s and s not in stops:
                    stops.append(s)
        return stops

    def check_eos(self) -> List[Finding]:
        stops = self.stop_strings()
        if not stops:
            return [Finding("eos", INFO, "no eos_token or generation_config stop ids; not checked")]
        out: List[Finding] = []
        checked = 0
        for i, m in enumerate(self.conversation):
            content = m.get("content")
            if m.get("role") != "assistant" or not isinstance(content, str) or not content:
                continue
            try:
                text = self.render(self.conversation[: i + 1])
            except TemplateRenderError:
                continue
            pos = text.rfind(content)
            if pos < 0:
                continue
            checked += 1
            tail = text[pos + len(content):]
            if not any(tail.lstrip(" \n").startswith(s) for s in stops):
                out.append(
                    Finding(
                        "eos",
                        WARN,
                        f"assistant message {i + 1} is not followed by a stop token",
                        [f"after the content the render has {_snip(tail)}; stop tokens: {', '.join(map(repr, stops))}"],
                    )
                )
        if not out:
            if checked:
                out.append(Finding("eos", OK, f"assistant turns end with a stop token ({', '.join(map(repr, stops))})"))
            else:
                out.append(Finding("eos", INFO, "no assistant content found in the render; not checked"))
        return out

    def run(self, only: Optional[List[str]] = None) -> List[Finding]:
        order = ["prefix", "genprompt", "system", "tools", "bos", "eos", "mask"]
        findings: List[Finding] = []
        for name in order:
            if only and name not in only:
                continue
            findings.extend(getattr(self, f"check_{name}")())
        return findings


CHECKS = ["prefix", "genprompt", "system", "tools", "bos", "eos", "mask"]
