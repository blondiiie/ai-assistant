from __future__ import annotations

import hashlib
from pathlib import Path

from app.sync import scanner
from app.sync.scanner import _sha256


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    f = tmp_path / "x.md"
    f.write_bytes(b"hello world")
    assert _sha256(f) == hashlib.sha256(b"hello world").hexdigest()


def test_iter_files_filters_by_extension_and_ignores(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "c.png").write_text("c", encoding="utf-8")
    (tmp_path / ".hidden.md").write_text("h", encoding="utf-8")
    ignored = tmp_path / ".obsidian"
    ignored.mkdir()
    (ignored / "plugin.md").write_text("p", encoding="utf-8")

    monkeypatch.setattr(scanner.settings, "ignore_dirs", ".obsidian,.trash")

    found = [p.name for p in scanner._iter_files(tmp_path)]
    assert set(found) == {"a.md", "b.txt"}
