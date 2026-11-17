"""Our renderer against transformers' own render_jinja_template (skipped if transformers isn't installed)."""

from datetime import datetime

import pytest

from chatprefix.checks import DEFAULT_CONVERSATION, WEATHER_TOOL, with_reasoning
from chatprefix.render import compile_template, render
from chatprefix.source import load_dir

from conftest import MODELS, TEMPLATES

cu = pytest.importorskip("transformers.utils.chat_template_utils")

CONVERSATIONS = [
    DEFAULT_CONVERSATION,
    with_reasoning(DEFAULT_CONVERSATION),
    DEFAULT_CONVERSATION[1:],
    DEFAULT_CONVERSATION[:2],
]


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("tools", [None, [WEATHER_TOOL]])
@pytest.mark.parametrize("gen", [False, True])
def test_render_matches_transformers(name, tools, gen, monkeypatch):
    files = load_dir(TEMPLATES / name)
    source = files.template_for(tools=bool(tools))
    fixed = datetime(2020, 1, 2)  # not today, so a missed patch would show up

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed

    monkeypatch.setattr(cu, "datetime", FixedDatetime)
    ours = compile_template(source, now=fixed)
    compared = 0
    for conv in CONVERSATIONS:
        try:
            expected = cu.render_jinja_template(
                conversations=[conv], tools=tools, chat_template=source, add_generation_prompt=gen, **files.special_tokens
            )[0][0]
        except Exception as e:  # the template rejects this input; we must reject it too
            with pytest.raises(Exception):
                render(ours, conv, special_tokens=files.special_tokens, tools=tools, add_generation_prompt=gen)
            continue
        assert render(ours, conv, special_tokens=files.special_tokens, tools=tools, add_generation_prompt=gen) == expected
        compared += 1
    assert compared
