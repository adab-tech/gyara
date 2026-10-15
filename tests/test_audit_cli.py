import json

from gyara import audit_text, dataset_stats
from gyara.cli import main


def test_audit_clean():
    r = audit_text("Ina ƙasa mai kyau")
    assert r.ok and r.hooked_letters == 1


def test_audit_flags_problems():
    r = audit_text("k'asa ٢٠١٥")
    assert not r.ok
    assert r.ascii_hook_spellings == 1 and r.foreign_digits == 4


def test_stats():
    s = dataset_stats(["Ina ƙasa", "Ina ƙasa", "", "ina 5 gida"])
    assert s["lines"] == 4 and s["empty_lines"] == 1 and s["duplicate_lines"] == 1
    assert s["lines_with_digits"] == 1
    assert s["tokens"] == 7 and s["types"] == 4
    assert s["hooked_letters"] == 2


def test_cli_fix_and_audit(tmp_path, capsys):
    f = tmp_path / "in.txt"
    f.write_text("k'asa b'aki\n", encoding="utf-8")
    assert main(["audit", str(f)]) == 1
    assert main(["fix", str(f)]) == 0
    assert capsys.readouterr().out.endswith("ƙasa ɓaki\n")


def test_cli_audit_json_ok(tmp_path, capsys):
    f = tmp_path / "ok.txt"
    f.write_text("ƙasa\n", encoding="utf-8")
    assert main(["audit", "--json", str(f)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_cli_syllables(capsys):
    assert main(["syllables", "k'asa"]) == 0
    assert "ƙa-sa" in capsys.readouterr().out


def test_audit_warns_on_quoted_lookalikes_without_failing():
    r = audit_text("Ya ce 'yaro' ne")
    assert r.ok and r.warnings and "'yaro'" in r.warnings[0]
    assert r.to_dict()["warnings"] == r.warnings


def test_cli_audit_prints_warning(tmp_path, capsys):
    f = tmp_path / "q.txt"
    f.write_text("'yaro'\n", encoding="utf-8")
    assert main(["audit", str(f)]) == 0
    assert "warning:" in capsys.readouterr().out
