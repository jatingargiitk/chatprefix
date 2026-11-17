from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
TEMPLATES = FIXTURES / "templates"
MODELS = sorted(p.name for p in TEMPLATES.iterdir() if p.is_dir())


@pytest.fixture
def template_dir():
    return lambda name: TEMPLATES / name
