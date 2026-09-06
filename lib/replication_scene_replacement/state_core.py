"""Shared persistence primitives for the direct scene-replacement engine."""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .errors import SceneReplacementError
from .storage import resolve_under, sha256_json

try:  # pragma: no cover - platform-specific import
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None
try:  # pragma: no cover - platform-specific import
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None


PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class SceneReplacementStateCore:
    """Path-safe revision locking and optimistic concurrency helpers."""

    @contextmanager
    def _exclusive_lock(self):
        artifact_root = self._ensure_safe_directory(
            self.artifact_root, "Scene-replacement artifact root"
        )
        lock_path = artifact_root / ".state.lock"
        if lock_path.is_symlink():
            raise SceneReplacementError("Scene-replacement lock must not be a symlink")
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(lock_path, flags, 0o600)
        try:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            elif msvcrt is not None:  # pragma: no cover - Windows
                if os.fstat(descriptor).st_size == 0:
                    os.write(descriptor, b"\0")
                    os.fsync(descriptor)
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
            else:  # pragma: no cover - unsupported platform
                raise SceneReplacementError("No supported project-lock implementation")
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            elif msvcrt is not None:  # pragma: no cover - Windows
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            os.close(descriptor)

    @staticmethod
    def validate_project_id(project_id: str) -> None:
        if not isinstance(project_id, str) or not PROJECT_ID_RE.fullmatch(project_id):
            raise SceneReplacementError(
                "project_id must contain only letters, digits, dot, underscore, or hyphen"
            )

    def relative(self, path: Path) -> str:
        try:
            return resolve_under(path, self.project_dir).relative_to(self.project_dir).as_posix()
        except (OSError, ValueError) as exc:
            raise SceneReplacementError("Path must stay inside the project workspace") from exc

    def _ensure_safe_directory(self, path: Path, label: str) -> Path:
        expected = Path(os.path.abspath(path))
        try:
            before = resolve_under(expected, self.project_dir)
        except (OSError, ValueError) as exc:
            raise SceneReplacementError(f"{label} must stay inside the project workspace") from exc
        if path.is_symlink() or before != expected:
            raise SceneReplacementError(f"{label} must not contain symlink indirection")
        expected.mkdir(parents=True, exist_ok=True)
        try:
            after = resolve_under(expected, self.project_dir, must_exist=True)
        except (OSError, ValueError) as exc:
            raise SceneReplacementError(f"{label} is unsafe after creation") from exc
        if after != expected or expected.is_symlink() or not expected.is_dir():
            raise SceneReplacementError(f"{label} must be a canonical project directory")
        return expected

    @staticmethod
    def _fingerprint(operation: str, inputs: dict[str, Any]) -> str:
        payload = {key: value for key, value in inputs.items() if key != "output_path"}
        return sha256_json({"operation": operation, "inputs": payload})

    @staticmethod
    def _idempotent(state: dict[str, Any], fingerprint: str) -> bool:
        return any(
            item.get("fingerprint") == fingerprint
            for item in state.get("operation_history", [])
        )

    @staticmethod
    def _require_parent(inputs: dict[str, Any], index: dict[str, Any]) -> None:
        if inputs.get("parent_replacement_revision") != index.get("replacement_revision"):
            raise SceneReplacementError("parent_replacement_revision is stale or missing")


__all__ = ["PROJECT_ID_RE", "SceneReplacementStateCore"]
