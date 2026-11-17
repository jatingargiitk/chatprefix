import json

from chatprefix.cli import main

from conftest import TEMPLATES


def test_text_output_and_exit_code(capsys):
    code = main([str(TEMPLATES / "qwen3-0.6b"), "--date", "2026-11-17"])
    out = capsys.readouterr().out
    assert code == 1
    assert "FAIL prefix" in out
    assert out.strip().endswith("1 fail, 0 warn")


def test_clean_template_exits_zero(capsys):
    assert main([str(TEMPLATES / "qwen2.5-0.5b-instruct")]) == 0


def test_strict_fails_on_warn(capsys):
    assert main([str(TEMPLATES / "smollm3-3b"), "--check", "tools"]) == 0
    assert main([str(TEMPLATES / "phi-4-mini-instruct"), "--check", "tools"]) == 0
    assert main([str(TEMPLATES / "phi-4-mini-instruct"), "--check", "tools", "--strict"]) == 1


def test_json_output(capsys):
    main([str(TEMPLATES / "deepseek-r1-distill-qwen-1.5b"), "--json", "--check", "genprompt"])
    doc = json.loads(capsys.readouterr().out)
    assert doc["fail"] == 2
    assert {f["check"] for f in doc["findings"]} == {"genprompt"}


def test_conversation_file_and_show(tmp_path, capsys):
    conv = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    p = tmp_path / "c.json"
    p.write_text(json.dumps(conv))
    main([str(TEMPLATES / "qwen2.5-0.5b-instruct"), "--conversation", str(p), "--show"])
    out = capsys.readouterr().out
    assert "--- render ---" in out
    assert "<|im_start|>user\nhi<|im_end|>" in out


def test_jinja_file_with_eos(tmp_path, capsys):
    p = tmp_path / "t.jinja"
    p.write_text("{% for m in messages %}[{{ m.role }}] {{ m.content }}{% endfor %}")
    code = main([str(p), "--eos", "</s>", "--check", "eos"])
    out = capsys.readouterr().out
    assert code == 0
    assert "WARN eos" in out


def test_bad_source(capsys):
    assert main(["nope nope"]) == 2
    assert "not a directory" in capsys.readouterr().err


def test_kwarg_parsing(capsys):
    code = main([str(TEMPLATES / "qwen3-0.6b"), "--kwarg", "enable_thinking=false", "--check", "genprompt"])
    assert code == 0
