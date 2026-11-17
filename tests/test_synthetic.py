"""Hand-written templates and a tiny tokenizer, for the checks whose real-world cases need big files."""

import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from chatprefix.checks import FAIL, INFO, OK, WARN, Checker, divergence
from chatprefix.render import TemplateRenderError, compile_template, render
from chatprefix.source import ModelFiles, SourceError, load, load_dir

CHATML = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
BOS_CHATML = "{{ bos_token }}" + CHATML


def make_tokenizer(tmp_path, add_bos):
    vocab = {"<s>": 0, "<|im_start|>": 1, "<|im_end|>": 2, "[UNK]": 3, "x": 4}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.add_special_tokens(["<s>", "<|im_start|>", "<|im_end|>"])
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    if add_bos:
        tok.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 0)])
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    return path


def files(template, tmp_path=None, add_bos=None, **special):
    f = ModelFiles(name="t", templates={"default": template}, special_tokens=special)
    if add_bos is not None:
        f.tokenizer_path = make_tokenizer(tmp_path, add_bos)
    return f


def statuses(findings, check):
    return [x.status for x in findings if x.check == check]


def test_clean_chatml_passes_everything_but_tools():
    f = Checker(files(CHATML, eos_token="<|im_end|>")).run()
    assert statuses(f, "prefix") == [OK]
    assert statuses(f, "genprompt") == [OK]
    assert statuses(f, "system") == [OK]
    assert statuses(f, "tools") == [WARN]
    assert statuses(f, "eos") == [OK]
    assert statuses(f, "mask") == [INFO]


def test_double_bos(tmp_path):
    f = Checker(files(BOS_CHATML, tmp_path, add_bos=True, bos_token="<s>")).run(only=["bos"])
    assert [x.status for x in f] == [WARN]
    assert "double BOS" in f[0].message
    assert "[0, 0, 1]" in f[0].details[0]


def test_single_bos_from_template(tmp_path):
    f = Checker(files(BOS_CHATML, tmp_path, add_bos=False, bos_token="<s>")).run(only=["bos"])
    assert [x.status for x in f] == [OK]


def test_bos_only_from_tokenizer(tmp_path):
    f = Checker(files(CHATML, tmp_path, add_bos=True, bos_token="<s>")).run(only=["bos"])
    assert [x.status for x in f] == [WARN]
    assert "doesn't write" in f[0].message


def test_missing_stop_token():
    tpl = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}\n{% endfor %}"
    f = Checker(files(tpl, eos_token="<|im_end|>")).run(only=["eos"])
    assert [x.status for x in f] == [WARN, WARN]


def test_template_that_rejects_system():
    tpl = "{% if messages[0].role == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}" + CHATML
    f = Checker(files(tpl)).run(only=["system"])
    assert f[0].status == WARN
    assert "System role not supported" in f[0].message


def test_template_that_raises_on_tools():
    tpl = "{% if tools %}{{ raise_exception('no tools') }}{% endif %}" + CHATML
    f = Checker(files(tpl)).run(only=["tools"])
    assert f[0].status == WARN and "raises" in f[0].message


def test_tool_use_template_is_used_for_tools():
    f = ModelFiles(name="t", templates={"default": CHATML, "tool_use": "{{ tools[0].function.name }}" + CHATML})
    out = Checker(f).run(only=["tools"])
    assert out[0].status == OK


def test_prefix_partial_renders_are_skipped_not_failed():
    tpl = "{% if messages|length < 2 %}{{ raise_exception('need two') }}{% endif %}" + CHATML
    f = Checker(files(tpl)).run(only=["prefix"])
    assert [x.status for x in f] == [OK]
    assert any("don't render on their own" in d for d in f[0].details)


def test_divergence_points_at_first_difference():
    d = divergence("abcXYZ", "abcdef")
    assert d[0].startswith("first difference at char 3")
    assert "'XYZ'" in d[1] and "'def'" in d[2]


def test_render_drops_none_and_aliases_reasoning():
    tpl = compile_template("{% for m in messages %}{{ m.keys()|list|sort|join(',') }};{% endfor %}")
    out = render(tpl, [{"role": "assistant", "content": "a", "tool_calls": None, "reasoning": "r"}])
    assert out == "content,reasoning,reasoning_content,role,thinking;"


def test_tojson_does_not_escape_html():
    tpl = compile_template("{{ x|tojson }}")
    assert tpl.render(x={"a": "<b>é"}) == '{"a": "<b>é"}'


def test_render_error_wraps_raise_exception():
    tpl = compile_template("{{ raise_exception('boom') }}")
    with pytest.raises(TemplateRenderError, match="boom"):
        render(tpl, [])


def test_sandbox_blocks_attribute_escape():
    tpl = compile_template("{{ ''.__class__.__mro__ }}")
    with pytest.raises(TemplateRenderError):
        render(tpl, [])


def test_load_dir_reads_template_list_and_special_tokens(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": [{"name": "default", "template": CHATML}, {"name": "tool_use", "template": "T"}],
                "bos_token": {"content": "<s>", "__type": "AddedToken"},
                "eos_token": None,
            }
        )
    )
    (tmp_path / "special_tokens_map.json").write_text(json.dumps({"eos_token": "</s>", "bos_token": "<ignored>"}))
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [2, 7]}))
    f = load_dir(tmp_path)
    assert f.templates["tool_use"] == "T"
    assert f.special_tokens == {"bos_token": "<s>", "eos_token": "</s>"}
    assert f.stop_token_ids == [2, 7]


def test_jinja_file_beats_tokenizer_config(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "OLD"}))
    (tmp_path / "chat_template.jinja").write_text("NEW")
    assert load_dir(tmp_path).templates == {"default": "NEW"}


def test_errors(tmp_path):
    with pytest.raises(SourceError, match="no chat template"):
        load_dir(tmp_path)
    with pytest.raises(SourceError, match="not a directory"):
        load("not a repo id")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": [{"name": "a", "template": "x"}]}))
    with pytest.raises(SourceError, match="none named 'default'"):
        load_dir(tmp_path)
