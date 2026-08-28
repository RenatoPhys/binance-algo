"""Deterministic Git provenance with an explicit source-tree fallback."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from binance_algo.common.errors import ResearchError
from binance_algo.research.experiments.models import CodeFingerprint, ProvenanceQuality

_SOURCE_SUFFIXES = frozenset({".py", ".pyi", ".toml", ".yaml", ".yml", ".md"})
_IGNORED_PARTS = frozenset({".git", ".venv", ".mypy_cache", ".pytest_cache", ".ruff_cache", "var"})


def _git(project_root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ("git", *arguments),
        cwd=project_root,
        check=False,
        capture_output=True,
    )


def _hash_untracked(project_root: Path, digest: hashlib._Hash) -> None:
    result = _git(project_root, "ls-files", "--others", "--exclude-standard", "-z")
    if result.returncode != 0:
        raise ResearchError("cannot enumerate untracked files for code fingerprint")
    paths = sorted(item for item in result.stdout.split(b"\0") if item)
    for raw_path in paths:
        relative = raw_path.decode("utf-8", errors="strict")
        path = project_root / relative
        if not path.is_file():
            continue
        digest.update(b"untracked\0")
        digest.update(relative.replace("\\", "/").encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())


def _dirty_diff_sha256(project_root: Path) -> str:
    result = _git(project_root, "diff", "HEAD", "--binary", "--no-ext-diff")
    if result.returncode != 0:
        raise ResearchError("cannot read Git diff for code fingerprint")
    digest = hashlib.sha256()
    digest.update(result.stdout)
    _hash_untracked(project_root, digest)
    return digest.hexdigest()


def source_tree_sha256(project_root: Path) -> str:
    """Hash source/configuration text while excluding state, artifacts, and tool caches."""

    project_root = project_root.resolve()
    candidates = [
        path
        for path in project_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in _SOURCE_SUFFIXES
        and not _IGNORED_PARTS.intersection(path.relative_to(project_root).parts)
    ]
    if not candidates:
        raise ResearchError(f"no source files available for fallback provenance: {project_root}")
    digest = hashlib.sha256()
    for path in sorted(candidates, key=lambda item: item.relative_to(project_root).as_posix()):
        relative = path.relative_to(project_root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_code_fingerprint(project_root: Path) -> CodeFingerprint:
    """Prefer Git commit/diff identity and fall back explicitly when Git is unavailable."""

    project_root = project_root.resolve()
    try:
        commit = _git(project_root, "rev-parse", "HEAD")
    except OSError:
        commit = None
    if commit is not None and commit.returncode == 0:
        commit_sha = commit.stdout.decode().strip()
        status = _git(project_root, "status", "--porcelain=v1", "--untracked-files=all")
        if status.returncode != 0:
            raise ResearchError("cannot read Git status for code fingerprint")
        dirty = bool(status.stdout)
        return CodeFingerprint(
            git_commit=commit_sha,
            git_dirty=dirty,
            git_diff_sha256=_dirty_diff_sha256(project_root) if dirty else None,
            source_tree_sha256=None,
            provenance_quality=(
                ProvenanceQuality.GIT_DIRTY if dirty else ProvenanceQuality.GIT_CLEAN
            ),
        )
    return CodeFingerprint(
        git_commit=None,
        git_dirty=False,
        git_diff_sha256=None,
        source_tree_sha256=source_tree_sha256(project_root),
        provenance_quality=ProvenanceQuality.FALLBACK_SOURCE_HASH,
    )


__all__ = ["build_code_fingerprint", "source_tree_sha256"]
