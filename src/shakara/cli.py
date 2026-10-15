"""`shakara` command line interface."""

import argparse
import json
import sys
from typing import Iterable, List, Optional

from . import __version__
from .audit import audit_text, dataset_stats
from .numbers import spell_out_numbers
from .orthography import normalize
from .syllables import syllabify
from .tone import annotate_tones


def _read(path: Optional[str]) -> str:
    if path in (None, "-"):
        return sys.stdin.read()
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _emit(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def _cmd_fix(a) -> int:
    sys.stdout.write(normalize(_read(a.file), digits=not a.keep_digits))
    return 0


def _cmd_spell(a) -> int:
    sys.stdout.write(spell_out_numbers(normalize(_read(a.file))))
    return 0


def _cmd_syllables(a) -> int:
    for word in a.words:
        parts = syllabify(normalize(word))
        print(f"{word}\t" + "-".join(s.text for s in parts) + "\t" + " ".join(s.weight for s in parts))
    return 0


def _cmd_tone(a) -> int:
    sys.stdout.write(annotate_tones(normalize(_read(a.file)), guess=a.guess))
    return 0


def _cmd_audit(a) -> int:
    report = audit_text(_read(a.file))
    if a.json:
        _emit(report.to_dict())
    else:
        print(f"words: {report.words}   hooked letters: {report.hooked_letters}")
        for issue in report.issues:
            print(f"- {issue}")
        print("OK" if report.ok else "FAIL (run `shakara fix` to repair)")
    return 0 if report.ok else 1


def _cmd_stats(a) -> int:
    stats = dataset_stats(_read(a.file).splitlines())
    if a.json:
        _emit(stats)
    else:
        for key, value in stats.items():
            print(f"{key}: {value}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="shakara", description="Hausa text toolkit.")
    p.add_argument("--version", action="version", version=f"shakara {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, fn, help_, file_arg=True):
        s = sub.add_parser(name, help=help_)
        if file_arg:
            s.add_argument("file", nargs="?", help="input file (default: stdin)")
        s.set_defaults(func=fn)
        return s

    add("fix", _cmd_fix, "restore ɓ ɗ ƙ ƴ from apostrophe spellings").add_argument(
        "--keep-digits", action="store_true", help="do not fold Arabic-Indic digits to 0-9")
    add("spell", _cmd_spell, "spell digits as Hausa words (for speech)")
    s = add("syllables", _cmd_syllables, "syllabify words with weights", file_arg=False)
    s.add_argument("words", nargs="+")
    add("tone", _cmd_tone, "heuristic tone marks").add_argument(
        "--guess", action="store_true", help="also mark unknown words (low confidence)")
    add("audit", _cmd_audit, "check text; exit 1 if problems found").add_argument(
        "--json", action="store_true")
    add("stats", _cmd_stats, "corpus statistics (one sentence per line)").add_argument(
        "--json", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
