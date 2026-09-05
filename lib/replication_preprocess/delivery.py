"""Readable, byte-preserving delivery copies of an accepted replication package."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

from .storage import atomic_write_json, resolve_under, sha256_file, sha256_json
from .selection import validate_document


NAMING_VERSION = "sequence-v1"


class DeliveryError(ValueError):
    """Delivery failed without invalidating the already exported source media."""


def _time(point: dict) -> Fraction:
    return point["pts"] * Fraction(point["time_base"]["num"], point["time_base"]["den"])


def build_manifest(state: dict, index: dict, report: dict, variant: str | None = None) -> dict:
    """Derive all names and bindings from canonical times, never directory order."""
    if (state.get("plan_status") != "ready" or state.get("anchor_status") != "ready"
            or (index.get("export") or {}).get("status") != "validated"
            or report.get("status") != "validated"):
        raise DeliveryError("Delivery requires a ready timeline, accepted anchors and validated export")
    clips = sorted(state["generation_clips"], key=lambda c: (_time(c["start"]), c["clip_id"]))
    exported = {c["clip_id"]: c for c in report["clips"]}
    if (len(exported) != len(report["clips"])
            or set(exported) != {c["clip_id"] for c in clips}):
        raise DeliveryError("Exported clips do not match the current plan")
    clip_width = max(2, len(str(len(clips))))
    frame_width = max(2, len(str(max((len(c["keyframes"]) for c in clips), default=0))))
    rows = []
    for number, clip in enumerate(clips, 1):
        display_id = f"S{number:0{clip_width}d}"
        media = exported[clip["clip_id"]]
        if media.get("full_decode") != "passed" or media.get("status") != "validated":
            raise DeliveryError(f"Clip has not passed export validation: {clip['clip_id']}")
        row = {k: deepcopy(clip[k]) for k in (
            "clip_id", "start", "end", "duration_s", "atomic_segment_ids",
        )}
        row.update(sequence=number, display_id=display_id, source_path=media["path"],
                   sha256=media["sha256"], keyframes=[])
        anchors = sorted(clip["keyframes"], key=lambda a: (_time(a["source_time"]), a["anchor_id"]))
        for frame_number, anchor in enumerate(anchors, 1):
            frame = {k: deepcopy(anchor[k]) for k in (
                "anchor_id", "atomic_segment_id", "role", "source_time",
                "atomic_segment_offset", "generation_clip_offset", "sha256",
            )}
            frame.update(sequence=frame_number, display_id=f"{display_id}_K{frame_number:0{frame_width}d}",
                         clip_id=clip["clip_id"], source_path=anchor["path"])
            row["keyframes"].append(frame)
        rows.append(row)
    manifest = {
        "schema_version": "1.0", "naming_version": NAMING_VERSION,
        "project_id": state["project_id"], "plan_revision": state["plan_revision"],
        "source_sha256": state["source"]["file_sha256"],
        "plan_state_ref": deepcopy(index["manifest_index"]["plan_state.json"]),
        "export_ref": deepcopy(index["export"]), "clips": rows,
    }
    fingerprint = sha256_json(manifest)
    variant = variant or fingerprint[:16]
    if not re.fullmatch(re.escape(fingerprint[:16]) + r"(?:-repair-[1-9][0-9]*)?", variant):
        raise DeliveryError("Delivery directory does not match the current package")
    suffix = f"replication/delivery/{state['plan_revision']}/{variant}"
    for row in rows:
        row["path"] = f"assets/video/{suffix}/{row['display_id']}.mp4"
        for frame in row["keyframes"]:
            frame["path"] = f"assets/images/{suffix}/{frame['display_id']}.png"
    manifest["delivery_fingerprint"] = fingerprint
    validate_document("replication_delivery_manifest", manifest)
    return manifest


def _assets(manifest: dict):
    for clip in manifest["clips"]:
        yield clip
        yield from clip["keyframes"]


def delivery_fields(reference: dict | None, manifest: dict | None = None) -> dict:
    return {
        "delivery": reference,
        "delivery_status": "ready" if reference else "pending",
        "delivery_clip_paths": [c["path"] for c in manifest["clips"]] if manifest else [],
        "delivery_keyframe_paths": [a["path"] for c in manifest["clips"] for a in c["keyframes"]] if manifest else [],
    }


class DeliveryPackage:
    def __init__(self, project_dir: Path, index_path: Path):
        self.project_dir = project_dir
        self.index_path = index_path

    def _path(self, relative: str, *, exists: bool = False) -> Path:
        return resolve_under(self.project_dir / relative, self.project_dir, must_exist=exists)

    def _report(self, index: dict) -> dict:
        reference = index["export"]
        path = self._path(reference["report_path"], exists=True)
        if sha256_file(path) != reference["report_sha256"]:
            raise DeliveryError("Export report hash mismatch")
        return json.loads(path.read_text(encoding="utf-8"))

    def _check(self, index: dict, state: dict, reference: dict) -> dict:
        path = self._path(reference["manifest_path"], exists=True)
        if sha256_file(path) != reference["manifest_sha256"]:
            raise DeliveryError("Delivery manifest hash mismatch")
        expected = build_manifest(state, index, self._report(index), path.parent.name)
        expected_path = f"artifacts/replication/delivery/{state['plan_revision']}/{path.parent.name}/manifest.json"
        if (reference["manifest_path"] != expected_path
                or reference["delivery_fingerprint"] != expected["delivery_fingerprint"]
                or json.loads(path.read_text(encoding="utf-8")) != expected):
            raise DeliveryError("Delivery naming, timing or ownership differs from the current plan")
        for asset in _assets(expected):
            copy_path = self._path(asset["path"], exists=True)
            if (self.project_dir / asset["path"]).is_symlink():
                raise DeliveryError(f"Delivery file must be an independent copy: {asset['path']}")
            if sha256_file(copy_path) != asset["sha256"]:
                raise DeliveryError(f"Delivery file hash mismatch: {asset['path']}")
        return expected

    def validate(self, index: dict, state: dict) -> tuple[dict, list[str]]:
        reference = index.get("delivery")
        if not reference:  # Old packages and pending reviews remain valid.
            return delivery_fields(None), []
        try:
            manifest = self._check(index, state, reference)
            return delivery_fields(reference, manifest), []
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return {**delivery_fields(None), "delivery_status": "failed"}, [f"delivery_invalid:{exc}"]

    def _publish_index(self, index: dict, reference: dict | None) -> None:
        current = json.loads(self.index_path.read_text(encoding="utf-8"))
        if current != index:
            raise DeliveryError("Project index changed before delivery publication; retry with current state")
        updated = deepcopy(index)
        previous = updated.get("delivery")
        if previous and previous != reference:
            history = updated.setdefault("delivery_history", [])
            if previous not in history:
                history.append(previous)
        updated["delivery"] = reference
        if updated != current:
            atomic_write_json(self.index_path, updated)
        index.clear()
        index.update(updated)

    def publish(self, index: dict, state: dict) -> tuple[dict, list[str]]:
        report = self._report(index)
        manifest = build_manifest(state, index, report)
        if index.get("delivery"):
            fields, issues = self.validate(index, state)
            if not issues:
                return fields, []
            self._publish_index(index, None)

        fingerprint = manifest["delivery_fingerprint"]
        revision = state["plan_revision"]
        variant = fingerprint[:16]
        repair = 0
        while True:
            directories = {
                kind: self._path(f"{root}/replication/delivery/{revision}/{variant}")
                for kind, root in (("video", "assets/video"), ("images", "assets/images"), ("manifest", "artifacts"))
            }
            manifest_path = directories["manifest"] / "manifest.json"
            if manifest_path.is_file():
                reference = {
                    "delivery_fingerprint": fingerprint,
                    "manifest_path": manifest_path.relative_to(self.project_dir).as_posix(),
                    "manifest_sha256": sha256_file(manifest_path),
                }
                try:  # Recover a completed package whose index publication was interrupted.
                    existing = self._check(index, state, reference)
                except (OSError, ValueError, KeyError, TypeError):
                    pass
                else:
                    self._publish_index(index, reference)
                    return delivery_fields(reference, existing), []
            if not any(path.exists() for path in directories.values()):
                break
            repair += 1
            variant = f"{fingerprint[:16]}-repair-{repair}"
        manifest = build_manifest(state, index, report, variant)
        staging: dict[str, Path] = {}
        created: list[Path] = []
        try:
            for kind, final in directories.items():
                final.parent.mkdir(parents=True, exist_ok=True)
                staging[kind] = Path(tempfile.mkdtemp(prefix=".delivery-", dir=final.parent))
            for asset in _assets(manifest):
                source = self._path(asset["source_path"], exists=True)
                if sha256_file(source) != asset["sha256"]:
                    raise DeliveryError(f"Canonical media hash mismatch: {asset['source_path']}")
                kind = "video" if asset["path"].endswith(".mp4") else "images"
                target = staging[kind] / Path(asset["path"]).name
                shutil.copyfile(source, target)
                if sha256_file(target) != asset["sha256"]:
                    raise DeliveryError(f"Delivery copy hash mismatch: {asset['path']}")
            atomic_write_json(staging["manifest"] / "manifest.json", manifest)
            for kind, final in directories.items():
                if final.exists():
                    raise DeliveryError("Delivery destination appeared during publication; retry")
                os.rename(staging[kind], final)
                created.append(final)
            reference = {
                "delivery_fingerprint": fingerprint,
                "manifest_path": manifest_path.relative_to(self.project_dir).as_posix(),
                "manifest_sha256": sha256_file(manifest_path),
            }
            self._publish_index(index, reference)
        except Exception:
            for path in created:
                shutil.rmtree(path)
            raise
        finally:
            for path in staging.values():
                if path.exists():
                    shutil.rmtree(path)
        artifacts = [str(manifest_path), *[str(self.project_dir / a["path"]) for a in _assets(manifest)]]
        return delivery_fields(reference, manifest), artifacts
