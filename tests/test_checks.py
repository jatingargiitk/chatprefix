"""Findings on real templates vendored from the Hub (see tests/fixtures/templates/SOURCES.md)."""

from datetime import datetime

from chatprefix.checks import FAIL, INFO, OK, WARN, Checker, with_reasoning, DEFAULT_CONVERSATION
from chatprefix.source import load_dir

from conftest import TEMPLATES

NOW = datetime(2026, 11, 17)


def run(name, conversation=None, **kw):
    files = load_dir(TEMPLATES / name)
    return Checker(files, conversation=conversation, now=NOW, **kw).run()


def by_check(findings, check):
    return [f for f in findings if f.check == check]


def statuses(findings, check):
    return [f.status for f in by_check(findings, check)]


def test_qwen25_is_clean():
    f = run("qwen2.5-0.5b-instruct")
    assert [x.status for x in f if x.status in (FAIL, WARN)] == []
    assert statuses(f, "prefix") == [OK]
    assert statuses(f, "genprompt") == [OK]


def test_qwen3_rewrites_the_last_assistant_turn():
    f = run("qwen3-0.6b")
    prefix = by_check(f, "prefix")
    assert [p.status for p in prefix] == [FAIL]
    assert "adding message 4 (user)" in prefix[0].message
    assert "<think>" in prefix[0].details[1]
    assert statuses(f, "genprompt") == [OK]


def test_qwen3_drops_reasoning_from_history():
    f = run("qwen3-0.6b", conversation=with_reasoning(DEFAULT_CONVERSATION))
    prefix = by_check(f, "prefix")
    assert [p.status for p in prefix] == [FAIL]
    assert "The user wants the capital of France." in prefix[0].details[1]
    assert "The user wants the capital of France." not in prefix[0].details[2]


def test_deepseek_generation_prompt_opens_think():
    f = run("deepseek-r1-distill-qwen-1.5b")
    gp = by_check(f, "genprompt")
    assert [g.status for g in gp] == [FAIL, FAIL]
    assert "'<think>\\n'" in gp[0].details[1]
    assert statuses(f, "prefix") == [OK]
    assert statuses(f, "tools") == [WARN]


def test_phi4_mini_appends_eos_to_every_render():
    f = run("phi-4-mini-instruct")
    prefix = by_check(f, "prefix")
    assert [p.status for p in prefix] == [FAIL] * 4
    assert all("endoftext" in p.details[1] for p in prefix)
    assert statuses(f, "tools") == [WARN]


def test_mistral_moves_system_to_last_user_turn():
    f = run("mistral-7b-instruct-v0.3")
    assert statuses(f, "system") == [WARN]
    assert statuses(f, "prefix") == [FAIL, FAIL]
    assert statuses(f, "genprompt") == [FAIL, FAIL]
    assert statuses(f, "tools") == [OK]


def test_gpt_oss_final_turn_uses_return():
    f = run("gpt-oss-20b")
    prefix = by_check(f, "prefix")
    assert [p.status for p in prefix] == [FAIL]
    assert "return|>" in prefix[0].details[1]
    assert "end|>" in prefix[0].details[2]


def test_smollm3_has_generation_blocks():
    f = run("smollm3-3b")
    assert statuses(f, "mask") == [OK]
    assert [x.status for x in f if x.status in (FAIL, WARN)] == []


def test_no_tokenizer_means_bos_skipped():
    f = run("qwen3-0.6b")
    assert statuses(f, "bos") == [INFO]


def test_eos_uses_template_eos_without_tokenizer():
    f = run("qwen2.5-0.5b-instruct")
    eos = by_check(f, "eos")
    assert [e.status for e in eos] == [OK]
    assert "<|im_end|>" in eos[0].message


def test_only_selected_checks_run():
    files = load_dir(TEMPLATES / "qwen3-0.6b")
    f = Checker(files, now=NOW).run(only=["mask", "system"])
    assert [x.check for x in f] == ["system", "mask"]


def test_kwargs_reach_the_template():
    # enable_thinking=False makes Qwen3 write an empty think block in the generation prompt
    files = load_dir(TEMPLATES / "qwen3-0.6b")
    c = Checker(files, now=NOW, extra={"enable_thinking": False})
    text = c.render(DEFAULT_CONVERSATION[:2], add_generation_prompt=True)
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
