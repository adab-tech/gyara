"""The golden vectors shared with the TypeScript port must match this implementation."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "gen_vectors.py"


def test_vectors_file_is_current(capsys):
    spec = importlib.util.spec_from_file_location("gen_vectors", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import json

    expected = json.loads(module.OUT.read_text(encoding="utf-8"))
    assert module.build() == expected, "tests/vectors.json is stale: run scripts/gen_vectors.py"
