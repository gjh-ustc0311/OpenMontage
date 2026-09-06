"""Small storage helpers shared by the Phase 2 state modules."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from lib.replication_preprocess.storage import (
    atomic_write_json,
    canonical_json_bytes,
    resolve_under,
    sha256_file,
    sha256_json,
)

from .errors import SceneReplacementError


REVISION_RE = re.compile(r"^r[0-9]{4}$")


def _safe_project_directory(path: Path, project_dir: Path, label: str) -> Path:
    """Create *path* only when its canonical location is the requested project path."""
    expected = Path(os.path.abspath(path))
    try:
        before = resolve_under(expected, project_dir)
    except (OSError, ValueError) as exc:
        raise SceneReplacementError(f"{label} must stay inside the project workspace") from exc
    if path.is_symlink() or before != expected:
        raise SceneReplacementError(f"{label} must not contain symlink indirection")
    expected.mkdir(parents=True, exist_ok=True)
    try:
        after = resolve_under(expected, project_dir, must_exist=True)
    except (OSError, ValueError) as exc:
        raise SceneReplacementError(f"{label} is unsafe after creation") from exc
    if after != expected or expected.is_symlink() or not expected.is_dir():
        raise SceneReplacementError(f"{label} must be a canonical project directory")
    return expected


def relative_ref(path: Path, project_dir: Path) -> dict[str, str]:
    resolved = resolve_under(path, project_dir, must_exist=True)
    return {
        "path": resolved.relative_to(project_dir).as_posix(),
        "sha256": sha256_file(resolved),
    }


def load_json(path: Path, project_dir: Path, *, reject_symlink: bool = True) -> dict[str, Any]:
    try:
        resolved = resolve_under(path, project_dir, must_exist=True)
        if reject_symlink and path.is_symlink():
            raise SceneReplacementError(f"JSON artifact must not be a symlink: {path}")
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except SceneReplacementError:
        raise
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SceneReplacementError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise SceneReplacementError(f"JSON artifact must be an object: {path}")
    return value


def verify_ref(reference: dict[str, Any], project_dir: Path, *, label: str) -> Path:
    if not isinstance(reference, dict):
        raise SceneReplacementError(f"{label} reference is missing")
    path_value = reference.get("path")
    digest = reference.get("sha256")
    if not isinstance(path_value, str) or not isinstance(digest, str):
        raise SceneReplacementError(f"{label} reference is incomplete")
    try:
        path = resolve_under(project_dir / path_value, project_dir, must_exist=True)
    except (OSError, ValueError) as exc:
        raise SceneReplacementError(f"{label} reference is missing or unsafe") from exc
    if (project_dir / path_value).is_symlink() or not path.is_file():
        raise SceneReplacementError(f"{label} reference must be a regular independent file")
    if sha256_file(path) != digest:
        raise SceneReplacementError(f"{label} hash mismatch")
    return path


def next_revision(revisions_dir: Path) -> str:
    numbers = []
    if revisions_dir.is_dir():
        for child in revisions_dir.iterdir():
            if child.is_dir() and REVISION_RE.fullmatch(child.name):
                numbers.append(int(child.name[1:]))
    return f"r{max(numbers, default=0) + 1:04d}"


def write_revision_directory(
    revisions_dir: Path,
    revision: str,
    documents: dict[str, Any],
    *,
    project_dir: Path,
) -> dict[str, dict[str, str]]:
    """Atomically publish a directory of immutable JSON documents."""

    if not REVISION_RE.fullmatch(revision):
        raise SceneReplacementError("Invalid replacement revision")
    revisions_dir = _safe_project_directory(
        revisions_dir, project_dir, "Scene-replacement revisions root"
    )
    destination = revisions_dir / revision
    if destination.exists() or destination.is_symlink():
        raise SceneReplacementError(f"Replacement revision already exists: {revision}")
    staging = Path(tempfile.mkdtemp(prefix=f".{revision}.", dir=revisions_dir))
    references: dict[str, dict[str, str]] = {}
    try:
        for name, value in documents.items():
            if Path(name).name != name or not name.endswith(".json"):
                raise SceneReplacementError(f"Invalid revision document name: {name}")
            staged_path = staging / name
            atomic_write_json(staged_path, value)
            references[name] = {
                "path": (destination / name).resolve().relative_to(project_dir).as_posix(),
                "sha256": sha256_file(staged_path),
            }
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return references


__all__ = [
    "REVISION_RE",
    "atomic_write_json",
    "canonical_json_bytes",
    "load_json",
    "next_revision",
    "relative_ref",
    "resolve_under",
    "sha256_file",
    "sha256_json",
    "verify_ref",
    "write_revision_directory",
]
