"""Command line: chatprefix SOURCE [options]."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from typing import List, Optional

from chatprefix import __version__
from chatprefix.checks import CHECKS, DEFAULT_CONVERSATION, FAIL, WARN, Checker, with_reasoning
from chatprefix.render import TemplateRenderError
from chatprefix.source import SourceError, load


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chatprefix",
        description=(
            "Check a Hugging Face chat template: does each turn render the same once the conversation grows "
            "(prefix/KV-cache reuse, multi-turn training), does add_generation_prompt match the rendered turn, "
            "and are system messages, tools, BOS and stop tokens handled the way callers assume."
        ),
    )
    p.add_argument("source", help="Hub repo id (org/name), a local tokenizer directory, or a .jinja file")
    p.add_argument("--revision", default="main", help="Hub revision (branch, tag or commit sha); default main")
    p.add_argument("--conversation", metavar="FILE", help="JSON list of messages to test with (default: a 5-message chat)")
    p.add_argument("--reasoning", action="store_true", help="give each assistant message a reasoning_content field")
    p.add_argument("--check", action="append", choices=CHECKS, help="run only this check (repeatable)")
    p.add_argument("--date", help="date strftime_now() returns, YYYY-MM-DD (default: today)")
    p.add_argument("--kwarg", action="append", default=[], metavar="KEY=JSON", help="extra template variable, e.g. enable_thinking=false")
    p.add_argument("--bos", help="bos_token to use with a bare .jinja file")
    p.add_argument("--eos", help="eos_token to use with a bare .jinja file")
    p.add_argument("--no-tokenizer", action="store_true", help="don't download/use tokenizer.json (skips token-level checks)")
    p.add_argument("--show", action="store_true", help="print the full render of the conversation")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--strict", action="store_true", help="exit 1 on WARN as well as FAIL")
    p.add_argument("--version", action="version", version=f"chatprefix {__version__}")
    return p


def _parse_kwargs(items: List[str]) -> dict:
    out = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"chatprefix: --kwarg expects KEY=JSON, got {item!r}")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        files = load(args.source, revision=args.revision, tokenizer=not args.no_tokenizer)
    except SourceError as e:
        print(f"chatprefix: {e}", file=sys.stderr)
        return 2
    if args.bos:
        files.special_tokens["bos_token"] = args.bos
    if args.eos:
        files.special_tokens["eos_token"] = args.eos

    conversation = DEFAULT_CONVERSATION
    if args.conversation:
        with open(args.conversation, encoding="utf-8") as f:
            conversation = json.load(f)
        if not isinstance(conversation, list) or not all(isinstance(m, dict) for m in conversation):
            print("chatprefix: --conversation must be a JSON list of message objects", file=sys.stderr)
            return 2
    if args.reasoning:
        conversation = with_reasoning(conversation)
    now = datetime.strptime(args.date, "%Y-%m-%d") if args.date else None

    checker = Checker(files, conversation=conversation, now=now, extra=_parse_kwargs(args.kwarg))
    findings = checker.run(only=args.check)
    fails = sum(f.status == FAIL for f in findings)
    warns = sum(f.status == WARN for f in findings)

    if args.json:
        doc = {
            "source": files.name,
            "findings": [f.to_dict() for f in findings],
            "fail": fails,
            "warn": warns,
        }
        if args.show:
            doc["render"] = _safe_render(checker, conversation)
        print(json.dumps(doc, indent=2, ensure_ascii=False))
    else:
        print(f"chatprefix {files.name}")
        for f in findings:
            print(f"  {f.status:<4} {f.check:<9} {f.message}")
            for d in f.details:
                print(f"       {'':<9} {d}")
        print(f"{fails} fail, {warns} warn")
        if args.show:
            print("--- render ---")
            print(_safe_render(checker, conversation))
    return 1 if fails or (args.strict and warns) else 0


def _safe_render(checker: Checker, conversation) -> str:
    try:
        return checker.render(conversation)
    except TemplateRenderError as e:
        return f"<render error: {e}>"
