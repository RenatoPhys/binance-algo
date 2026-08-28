from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from binance_algo.research.experiments.models import CodeFingerprint, ProvenanceQuality
from binance_algo.research.experiments.provenance import (
    build_code_fingerprint,
    source_tree_sha256,
)


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(("git", *arguments), cwd=root, check=True, capture_output=True)


def test_git_clean_dirty_and_untracked_code_fingerprints(tmp_path: Path) -> None:
    source = tmp_path / "src" / "package" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "research@example.invalid")
    _git(tmp_path, "config", "user.name", "Research Test")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "initial")

    clean = build_code_fingerprint(tmp_path)
    assert clean.provenance_quality is ProvenanceQuality.GIT_CLEAN
    assert clean.git_commit and not clean.git_dirty and clean.git_diff_sha256 is None

    source.write_text("VALUE = 2\n", encoding="utf-8")
    dirty = build_code_fingerprint(tmp_path)
    assert dirty.provenance_quality is ProvenanceQuality.GIT_DIRTY
    assert dirty.git_commit == clean.git_commit
    assert dirty.git_dirty and dirty.git_diff_sha256
    assert build_code_fingerprint(tmp_path) == dirty

    untracked = tmp_path / "src" / "package" / "new.py"
    untracked.write_text("NEW = True\n", encoding="utf-8")
    with_untracked = build_code_fingerprint(tmp_path)
    assert with_untracked.git_diff_sha256 != dirty.git_diff_sha256


def test_source_tree_fallback_is_deterministic_and_excludes_runtime_state(tmp_path: Path) -> None:
    source = tmp_path / "src" / "package" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")

    first = build_code_fingerprint(tmp_path)
    assert first.provenance_quality is ProvenanceQuality.FALLBACK_SOURCE_HASH
    assert first.source_tree_sha256 == source_tree_sha256(tmp_path)

    runtime = tmp_path / "var" / "state" / "runtime.yaml"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("ignored: true\n", encoding="utf-8")
    assert build_code_fingerprint(tmp_path) == first

    source.write_text("VALUE = 2\n", encoding="utf-8")
    assert build_code_fingerprint(tmp_path).source_tree_sha256 != first.source_tree_sha256


def test_source_tree_fallback_rejects_mixed_git_provenance() -> None:
    with pytest.raises(ValidationError, match="requires only a source-tree checksum"):
        CodeFingerprint(
            git_commit=None,
            git_dirty=False,
            git_diff_sha256="d" * 64,
            source_tree_sha256="s" * 64,
            provenance_quality=ProvenanceQuality.FALLBACK_SOURCE_HASH,
        )
