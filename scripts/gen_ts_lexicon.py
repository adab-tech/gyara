"""Generate js/src/lexicon.ts from the Python tone lexicon (single source of truth).

python scripts/gen_ts_lexicon.py          # rewrite js/src/lexicon.ts
python scripts/gen_ts_lexicon.py --check  # exit 1 if the file is stale
"""

import json
import sys
from pathlib import Path

from gyara.tone import LEXICON

OUT = Path(__file__).resolve().parent.parent / "js" / "src" / "lexicon.ts"


def render() -> str:
    rows = ",\n".join(
        f"  [{json.dumps(k, ensure_ascii=False)}, {json.dumps(v)}]" for k, v in LEXICON.items()
    )
    return (
        "// GENERATED from src/gyara/tone.py by scripts/gen_ts_lexicon.py. Do not edit by hand.\n"
        'type Tone = "H" | "L";\n'
        "export const LEXICON: ReadonlyMap<string, readonly Tone[]> = new Map<string, Tone[]>([\n"
        f"{rows},\n"
        "]);\n"
    )


if __name__ == "__main__":
    text = render()
    if "--check" in sys.argv:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            print("js/src/lexicon.ts is stale: run `python scripts/gen_ts_lexicon.py`")
            sys.exit(1)
        sys.exit(0)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}")
