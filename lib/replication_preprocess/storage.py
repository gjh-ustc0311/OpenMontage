"""Filesystem and hashing helpers for immutable preprocessing revisions."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical JSON representation used for fingerprints."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    """Write JSON atomically without exposing a partial manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def resolve_under(path: Path, root: Path, *, must_exist: bool = False) -> Path:
    """Resolve *path* and reject paths escaping *root*, including symlinks."""

    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=must_exist)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Path must stay under project workspace: {path}") from exc
    return resolved


def next_revision(revisions_dir: Path) -> str:
    existing: list[int] = []
    if revisions_dir.is_dir():
        for child in revisions_dir.iterdir():
            if child.is_dir() and child.name.startswith("r") and child.name[1:].isdigit():
                existing.append(int(child.name[1:]))
    return f"r{(max(existing, default=0) + 1):04d}"
