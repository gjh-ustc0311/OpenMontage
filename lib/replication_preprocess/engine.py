"""High-level deterministic engine for the replication preprocessing tool."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from typing import Any

from .analysis import (
    analyze_frame_quality,
    apply_separator_rules,
    build_atomic_segments,
    build_overview_contact_sheet,
    collect_runtime_versions,
    detect_scene_candidates,
    evidence_targets,
    extract_evidence_images,
    probe_source,
    select_boundaries,
)
from .config import compute_export_fingerprint, compute_plan_fingerprint
from .delivery import DeliveryPackage, delivery_fields
from .keyframe_workflow import KeyframeWorkflow
from .media_assets import legacy_trim_equivalent, media_fingerprint, media_signature
from .selection import SELECTION_ACTIONS, bind_anchors, validate_document
from .models import load_config, stable_id, time_point
from .planner import build_generation_plan, build_scene_groups
from .review import (
    ReviewValidationError,
    apply_review_submission,
    build_review_request,
)
from .storage import (
    atomic_write_json,
    next_revision,
    resolve_under,
    sha256_file,
    sha256_json,
)


class ReplicationPreprocessError(RuntimeError):
    pass


_PROJECT_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")


class ReplicationEngine(KeyframeWorkflow):
    """Create, review, export, and validate preprocessing revisions."""

    def __init__(self, project_dir: Path, index_path: Path):
        self.project_dir = project_dir.resolve(strict=True)
        self.index_path = resolve_under(index_path, self.project_dir)
        if self.index_path.exists() and not self.index_path.is_file():
            raise ReplicationPreprocessError("output_path must be a JSON file path")
        if self.index_path.suffix.lower() != ".json":
            raise ReplicationPreprocessError("output_path must use a .json extension")
        self.artifact_root = self.project_dir / "artifacts" / "replication"
        self.revisions_dir = self.artifact_root / "revisions"
        self.image_root = self.project_dir / "assets" / "images" / "replication"
        self.video_root = self.project_dir / "assets" / "video" / "replication"

    @staticmethod
    def validate_project_id(project_id: str) -> None:
        if not _PROJECT_ID_RE.fullmatch(project_id):
            raise ReplicationPreprocessError(
                "project_id must contain only letters, digits, dot, underscore, or hyphen"
            )

    def _relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.project_dir).as_posix()

    def _revision_dir(self, revision: str) -> Path:
        return self.revisions_dir / revision

    def _load_json(self, path: Path) -> dict[str, Any]:
        resolved = resolve_under(path, self.project_dir, must_exist=True)
        try:
            loaded = json.loads(resolved.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ReplicationPreprocessError(f"Invalid JSON artifact: {path}") from exc
        if not isinstance(loaded, dict):
            raise ReplicationPreprocessError(f"JSON artifact must be an object: {path}")
        return loaded

    def _load_state(self, revision: str) -> dict[str, Any]:
        if not re.fullmatch(r"r\d{4}", revision):
            raise ReplicationPreprocessError("Invalid parent_plan_revision")
        path = self._revision_dir(revision) / "plan_state.json"
        if not path.is_file():
            raise ReplicationPreprocessError(f"Unknown parent plan revision: {revision}")
        state = self._load_json(path)
        for field, filename in (("review_request", "boundary_review_request.json"),
                                ("review_submission", "boundary_review_submission.json"),
                                ("keyframe_request", "keyframe_selection_request.json"),
                                ("keyframe_submission", "keyframe_selection_submission.json")):
            entry = state.get("manifest_index", {}).get(filename)
            if entry:
                artifact = self.project_dir / entry["path"]
                if sha256_file(artifact) != entry["sha256"]:
                    raise ReviewValidationError(f"Manifest hash mismatch: {filename}")
                state[field] = self._load_json(artifact)
        return state

    def _find_idempotent(self, fingerprint: str) -> dict[str, Any] | None:
        if not self.index_path.is_file():
            return None
        index = self._load_json(self.index_path)
        self._assert_index_integrity(index)
        revision = index.get("plan_revision")
        if not isinstance(revision, str):
            return None
        state = self._load_state(revision)
        if state.get("input_fingerprint") != fingerprint:
            return None
        state["executed_stages"] = []
        state["reused_stages"] = ["analysis", "review", "planning"]
        state["invalidated_stages"] = []
        return state

    def _apply_topology_overrides(
        self,
        boundaries: list[dict[str, Any]],
        overrides: list[dict[str, Any]],
        ledger_pts: set[int],
        source: dict[str, Any],
        config: dict[str, Any],
        time_base: Fraction,
    ) -> list[dict[str, Any]]:
        result = deepcopy(boundaries)
        by_id = {item["boundary_id"]: item for item in result}
        for override in overrides:
            action = override.get("action")
            if action not in {"insert_boundary", "remove_boundary"}:
                continue
            if not override.get("actor") or not override.get("reason"):
                raise ReplicationPreprocessError("Manual overrides require actor and reason")
            if action == "remove_boundary":
                boundary_id = override.get("boundary_id")
                if boundary_id not in by_id:
                    raise ReplicationPreprocessError(f"Unknown boundary override target: {boundary_id}")
                result = [item for item in result if item["boundary_id"] != boundary_id]
                by_id.pop(boundary_id)
                continue
            pts = override.get("pts")
            if not isinstance(pts, int) or pts not in ledger_pts:
                raise ReplicationPreprocessError("Inserted boundary PTS must match a decoded video frame")
            if not source["start"]["pts"] < pts < source["end"]["pts"]:
                raise ReplicationPreprocessError("Inserted boundary must be inside the source interval")
            boundary_id = stable_id(
                "bnd",
                source["file_sha256"],
                pts,
                config["config_fingerprints"]["analysis"],
            )
            if any(item["time"]["pts"] == pts for item in result):
                raise ReplicationPreprocessError("A boundary already exists at the inserted PTS")
            hard = bool(override.get("hard", False))
            result.append({
                "schema_version": "1.0",
                "boundary_id": boundary_id,
                "time": time_point(pts, time_base),
                "frame_ordinal": None,
                "boundary_type": "manual",
                "accepted_threshold": None,
                "content_val": 0.0,
                "components": {"delta_hue": 0.0, "delta_sat": 0.0, "delta_lum": 0.0, "delta_edges": 0.0},
                "relationship": "human_hard" if hard else "human_soft",
                "hard": hard,
                "hard_reason": "manual_insert" if hard else None,
                "diagnostic_status": "accepted",
                "manual_override": deepcopy(override),
            })
        return sorted(result, key=lambda item: item["time"]["pts"])

    def _apply_non_topology_overrides(
        self,
        boundaries: list[dict[str, Any]],
        segments: list[dict[str, Any]],
        overrides: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> set[str]:
        boundary_by_id = {item["boundary_id"]: item for item in boundaries}
        segment_by_id = {item["segment_id"]: item for item in segments}
        drops: set[str] = set()
        for override in overrides:
            action = override.get("action")
            if action not in {"force_hard", "force_soft", "drop"}:
                continue
            if not override.get("actor") or not override.get("reason"):
                raise ReplicationPreprocessError("Manual overrides require actor and reason")
            if action in {"force_hard", "force_soft"}:
                boundary_id = override.get("boundary_id")
                if boundary_id not in boundary_by_id:
                    raise ReplicationPreprocessError(f"Unknown boundary override target: {boundary_id}")
                boundary = boundary_by_id[boundary_id]
                boundary["hard"] = action == "force_hard"
                boundary["relationship"] = "human_hard" if boundary["hard"] else "human_soft"
                boundary["hard_reason"] = "human_override" if boundary["hard"] else None
                boundary["manual_override"] = deepcopy(override)
            else:
                segment_id = override.get("segment_id")
                if segment_id not in segment_by_id:
                    raise ReplicationPreprocessError(f"Unknown segment drop target: {segment_id}")
                if not config["regroup"]["allow_drop_under_1s"]:
                    raise ReplicationPreprocessError("Dropping segments is disabled by configuration")
                if Fraction(segment_by_id[segment_id]["duration_s"]) >= Fraction(
                    str(config["regroup"]["drop_threshold_s"])
                ):
                    raise ReplicationPreprocessError("Only segments below the drop threshold can be dropped")
                drops.add(segment_id)
        return drops

    def _verify_request_evidence(self, request: dict[str, Any]) -> None:
        claimed_request_hash = request.get("request_sha256")
        request_body = deepcopy(request)
        request_body.pop("request_sha256", None)
        if claimed_request_hash != sha256_json(request_body):
            raise ReviewValidationError("Review request hash mismatch")
        for item in request.get("items", []):
            for evidence in item.get("evidence", {}).values():
                relative = evidence.get("path")
                digest = evidence.get("sha256")
                if not relative or not digest:
                    raise ReviewValidationError("Review request contains incomplete evidence")
                path = resolve_under(self.project_dir / relative, self.project_dir, must_exist=True)
                if sha256_file(path) != digest:
                    raise ReviewValidationError(f"Review evidence hash mismatch: {relative}")

    def _build_review_artifacts(
        self,
        *,
        source_path: Path,
        source: dict[str, Any],
        ledger: list[Any],
        time_base: Fraction,
        segments: list[dict[str, Any]],
        pending: list[dict[str, Any]],
        config: dict[str, Any],
        revision: str,
        review_round: int,
    ) -> tuple[dict[str, Any], list[str], str | None]:
        offset_key = "context_offset_s" if review_round == 1 else "expanded_context_offset_s"
        context = Fraction(str(config["review"][offset_key]))
        segment_by_id = {item["segment_id"]: item for item in segments}
        targets = {
            boundary["boundary_id"]: evidence_targets(
                boundary, segment_by_id, ledger, time_base, context
            )
            for boundary in pending
        }
        evidence_dir = self.image_root / "revisions" / revision / f"review-round-{review_round}"
        evidence, boards = extract_evidence_images(
            path=source_path,
            time_base=time_base,
            rotation=int(source.get("rotation") or 0),
            boundary_targets=targets,
            boundary_metadata={item["boundary_id"]: item for item in pending},
            output_dir=evidence_dir,
            project_dir=self.project_dir,
            review_config=config["review"],
        )
        overview_path = evidence_dir / "contact-sheet.jpg"
        overview = build_overview_contact_sheet(
            boards,
            overview_path,
            jpeg_quality=int(config["review"]["contact_sheet_jpeg_quality"]),
        )
        items = []
        for boundary in pending:
            items.append({
                "review_item_id": stable_id(
                    "bri", source["file_sha256"], boundary["boundary_id"], review_round
                ),
                "boundary_id": boundary["boundary_id"],
                "left_segment_id": boundary["left_segment_id"],
                "right_segment_id": boundary["right_segment_id"],
                "time": boundary["time"],
                "content_val": boundary["content_val"],
                "accepted_threshold": boundary["accepted_threshold"],
                "components": boundary["components"],
                "evidence": evidence[boundary["boundary_id"]],
            })
        request = build_review_request(
            source_sha256=source["file_sha256"],
            parent_plan_revision=revision,
            config_fingerprint=config["config_fingerprint"],
            analysis_fingerprint=config["config_fingerprints"]["analysis"],
            review_fingerprint=config["config_fingerprints"]["review"],
            protocol_version="boundary-visual-review-v2",
            review_round=review_round,
            items=items,
        )
        artifacts = [path for path in boards]
        if overview:
            artifacts.append(overview)
        return request, artifacts, self._relative(overview_path) if overview else None

    def _validate_plan_state(self, state: dict[str, Any]) -> None:
        """Reject incomplete or internally inconsistent plans before publication."""

        source = state["source"]
        segments = state["atomic_segments"]
        boundaries = state["boundaries"]
        if state.get("review_request"):
            self._verify_request_evidence(state["review_request"])
        if state.get("schema_version") == "3.0":
            self._validate_anchor_state(state)
        if not segments:
            raise ReplicationPreprocessError("Plan has no atomic segments")
        if segments[0]["start"]["pts"] != source["start"]["pts"]:
            raise ReplicationPreprocessError("Atomic timeline does not start at source start")
        if segments[-1]["end"]["pts"] != source["end"]["pts"]:
            raise ReplicationPreprocessError("Atomic timeline does not end at source end")
        if len(boundaries) != len(segments) - 1:
            raise ReplicationPreprocessError("Boundary and atomic segment counts are inconsistent")
        for sequence, segment in enumerate(segments):
            if segment.get("sequence") != sequence:
                raise ReplicationPreprocessError("Atomic segment sequence is inconsistent")
            if segment["start"]["pts"] >= segment["end"]["pts"]:
                raise ReplicationPreprocessError("Atomic segment must have positive duration")
            if sequence and segments[sequence - 1]["end"]["pts"] != segment["start"]["pts"]:
                raise ReplicationPreprocessError("Atomic timeline has a gap or overlap")
            keyframe = segment.get("keyframe", {})
            if keyframe.get("path"):
                keyframe_path = resolve_under(
                    self.project_dir / keyframe["path"], self.project_dir, must_exist=True
                )
                if sha256_file(keyframe_path) != keyframe.get("sha256"):
                    raise ReplicationPreprocessError("Keyframe hash mismatch")
        for index, boundary in enumerate(boundaries):
            left, right = segments[index], segments[index + 1]
            if (
                boundary.get("left_segment_id") != left["segment_id"]
                or boundary.get("right_segment_id") != right["segment_id"]
                or boundary["time"]["pts"] != left["end"]["pts"]
                or boundary["time"]["pts"] != right["start"]["pts"]
            ):
                raise ReplicationPreprocessError("Boundary references are inconsistent")

        if state["plan_status"] != "ready":
            if state.get("generation_clips"):
                raise ReplicationPreprocessError("Non-ready plan cannot expose generation clips")
            return

        clips = state.get("generation_clips", [])
        dropped = state.get("dropped_intervals", [])
        if not clips and not dropped:
            raise ReplicationPreprocessError("Ready plan has no timeline outputs")
        segment_by_id = {item["segment_id"]: item for item in segments}
        boundary_by_pair = {
            (item["left_segment_id"], item["right_segment_id"]): item
            for item in boundaries
        }
        consumed_ids: list[str] = []
        minimum = Fraction(str(state["config"]["profile"]["min_duration_s"]))
        maximum = Fraction(str(state["config"]["profile"]["max_duration_s"]))
        for clip in clips:
            member_ids = clip["atomic_segment_ids"]
            if not 1 <= len(member_ids) <= int(state["config"]["regroup"]["max_atomic_segments"]):
                raise ReplicationPreprocessError("Generation clip member count is invalid")
            members = [segment_by_id[item] for item in member_ids]
            if any(
                right["sequence"] != left["sequence"] + 1
                for left, right in zip(members, members[1:])
            ):
                raise ReplicationPreprocessError("Generation clip members are not adjacent")
            for left, right in zip(members, members[1:]):
                boundary = boundary_by_pair[(left["segment_id"], right["segment_id"])]
                if boundary.get("hard") is True or boundary.get("relationship") not in {
                    "same_scene", "forced_same_scene", "human_soft"
                }:
                    raise ReplicationPreprocessError("Generation clip crosses an unsafe boundary")
            duration = Fraction(clip["duration_s"])
            if duration != sum((Fraction(item["duration_s"]) for item in members), Fraction(0)):
                raise ReplicationPreprocessError("Generation clip duration is inconsistent")
            if duration < minimum or duration >= maximum:
                raise ReplicationPreprocessError("Generation clip duration violates profile")
            if sum(Fraction(item["duration_s"]) >= minimum for item in members) > 1:
                raise ReplicationPreprocessError("Generation clip merges multiple normal segments")
            if (
                clip["start"]["pts"] != members[0]["start"]["pts"]
                or clip["end"]["pts"] != members[-1]["end"]["pts"]
                or (state.get("schema_version") != "3.0" and len(clip.get("keyframes", [])) != len(members))
            ):
                raise ReplicationPreprocessError("Generation clip anchors are inconsistent")
            consumed_ids.extend(member_ids)
        for item in dropped:
            segment = segment_by_id[item["segment_id"]]
            if (
                not state["config"]["regroup"]["allow_drop_under_1s"]
                or Fraction(segment["duration_s"])
                >= Fraction(str(state["config"]["regroup"]["drop_threshold_s"]))
            ):
                raise ReplicationPreprocessError("Dropped interval is not approved by the profile")
            consumed_ids.append(item["segment_id"])
        ordered_consumed = sorted(consumed_ids, key=lambda item: segment_by_id[item]["sequence"])
        if len(set(consumed_ids)) != len(consumed_ids) or ordered_consumed != [
            item["segment_id"] for item in segments
        ]:
            raise ReplicationPreprocessError("Plan does not cover each atomic segment exactly once")

    def _validate_anchor_state(self, state: dict[str, Any]) -> None:
        if state.get("keyframe_request"):
            self._verify_keyframe_request(state)
        for segment in state["atomic_segments"]:
            selection = segment.get("anchor_selection", {})
            anchors = selection.get("anchors", [])
            if selection.get("status") == "ready":
                if sum(a.get("role") == "primary_representative" for a in anchors) != 1:
                    raise ReplicationPreprocessError("Each ready segment requires one primary representative")
            elif anchors:
                raise ReplicationPreprocessError("Unreviewed selection cannot expose adopted anchors")
            pts_seen = set()
            base = Fraction(segment["start"]["time_base"]["num"], segment["start"]["time_base"]["den"])
            for anchor in anchors:
                pts = anchor["pts"]
                if (pts in pts_seen or not segment["start"]["pts"] <= pts < segment["end"]["pts"]
                    or anchor["atomic_segment_id"] != segment["segment_id"]
                    or anchor["source_time"] != time_point(pts, base)
                    or anchor["atomic_segment_offset"] != time_point(pts - segment["start"]["pts"], base)):
                    raise ReplicationPreprocessError("Anchor time or segment binding is inconsistent")
                pts_seen.add(pts)
                path = resolve_under(self.project_dir / anchor["path"], self.project_dir, must_exist=True)
                if sha256_file(path) != anchor["sha256"] or anchor.get("format") != "png":
                    raise ReplicationPreprocessError("Anchor working PNG hash or format mismatch")
        expected = deepcopy(state)
        bind_anchors(expected)
        if expected["anchor_status"] != state.get("anchor_status"):
            raise ReplicationPreprocessError("Anchor readiness is inconsistent")
        for actual, rebound in zip(state.get("generation_clips", []), expected.get("generation_clips", [])):
            if actual.get("keyframes") != rebound.get("keyframes") or actual.get("anchor_status") != rebound.get("anchor_status"):
                raise ReplicationPreprocessError("Generation clip anchor bindings are inconsistent")

    def _set_next_actions(self, state: dict[str, Any]) -> None:
        actions = []
        revision = state["plan_revision"]
        if state.get("review_request") and state["review_status"] in {"pending_agent", "needs_more_evidence"}:
            actions.append({"type": "agent_boundary_review", "skill": "skills/meta/replication-boundary-review.md",
                            "request_path": self._relative(self._revision_dir(revision) / "boundary_review_request.json")})
        by_status: dict[str, list[str]] = {}
        dropped = {item["segment_id"] for item in state.get("dropped_intervals", [])}
        for segment in state["atomic_segments"]:
            if segment["segment_id"] not in dropped:
                by_status.setdefault(segment["anchor_selection"]["status"], []).append(segment["segment_id"])
        active = by_status.get("pending_agent", []) + by_status.get("needs_more_evidence", [])
        if active:
            actions.append({"type": "agent_keyframe_selection", "skill": "skills/meta/replication-keyframe-selection.md",
                            "request_path": self._relative(self._revision_dir(revision) / "keyframe_selection_request.json"),
                            "segment_ids": active})
        if state["plan_status"] != "ready" and not any(a["type"] == "agent_boundary_review" for a in actions):
            actions.append({"type": "human_plan_review", "reason": "timeline_unresolved"})
        if by_status.get("needs_human"):
            actions.append({"type": "human_keyframe_review", "segment_ids": by_status["needs_human"],
                            "request_path": self._relative(self._revision_dir(revision) / "keyframe_selection_request.json")})
        state["next_actions"] = actions
        state["next_action"] = actions[0] if actions else None

    def _write_revision(self, state: dict[str, Any], extra_artifacts: list[str]) -> dict[str, Any]:
        if state.get("parent_plan_revision") and self._load_json(self.index_path).get("plan_revision") != state["parent_plan_revision"]:
            raise ReviewValidationError("Parent revision changed before publication")
        if sha256_file(Path(state["source"]["path"])) != state["source"]["file_sha256"]:
            raise ReviewValidationError("Source changed before revision publication")
        state["schema_version"] = "3.0"
        bind_anchors(state)
        self._set_next_actions(state)
        selections = [s["anchor_selection"] for s in state["atomic_segments"]]
        state["quality_report"]["representative_selection"] = {
            "strategy": "representative-selection-v1",
            "anchor_status": state["anchor_status"],
            "candidate_count": sum(len(s["candidates"]) for s in selections),
            "primary_count": sum(a["role"] == "primary_representative" for s in selections for a in s["anchors"]),
            "supplementary_count": sum(a["role"] == "supplementary_anchor" for s in selections for a in s["anchors"]),
        }
        delivery_history = []
        if self.index_path.is_file():
            previous = self._load_json(self.index_path)
            delivery_history = deepcopy(previous.get("delivery_history", []))
            if previous.get("delivery") and previous["delivery"] not in delivery_history:
                delivery_history.append(deepcopy(previous["delivery"]))
            history = state.setdefault("export_history", [])
            for entry in [*previous.get("export_history", []), previous.get("export")]:
                if entry:
                    entry = deepcopy(entry)
                    if entry == previous.get("export"):
                        entry.setdefault("plan_state_ref", previous["manifest_index"]["plan_state.json"])
                    if entry not in history:
                        history.append(entry)
        self._validate_plan_state(state)
        revision = state["plan_revision"]
        revision_dir = self._revision_dir(revision)
        if revision_dir.exists():
            raise ReplicationPreprocessError(f"Plan revision already exists: {revision}")
        self.revisions_dir.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(tempfile.mkdtemp(prefix=f".{revision}.", dir=self.revisions_dir))
        manifests: dict[str, Any] = {
            "video.json": state["source"],
            "boundaries.json": {"schema_version": "1.0", "boundaries": state["boundaries"], "suppressed_candidates": state.get("suppressed_candidates", [])},
            "atomic_segments.json": {"schema_version": "3.0", "atomic_segments": state["atomic_segments"]},
            "scene_groups.json": {"schema_version": "1.0", "scene_groups": state.get("scene_groups", [])},
            "generation_clips.json": {"schema_version": "3.0", "generation_clips": state.get("generation_clips", [])},
            "dropped_intervals.json": {"schema_version": "1.0", "dropped_intervals": state.get("dropped_intervals", [])},
            "timeline_map.json": {"schema_version": "1.0", "entries": state.get("timeline_map", [])},
            "quality_report.json": state["quality_report"],
            "review_items.json": {"schema_version": "1.0", "review_items": state.get("review_items", [])},
        }
        if state.get("review_request"):
            manifests["boundary_review_request.json"] = state["review_request"]
        if state.get("review_submission"):
            manifests["boundary_review_submission.json"] = state["review_submission"]
        if state.get("keyframe_request"):
            validate_document("keyframe_selection_request", state["keyframe_request"])
            manifests["keyframe_selection_request.json"] = state["keyframe_request"]
        if state.get("keyframe_submission"):
            manifests["keyframe_selection_submission.json"] = state["keyframe_submission"]
        result = {"schema_version": "1.0", "anchor_status": state["anchor_status"], "segments": [
            {"segment_id": segment["segment_id"], "selection": segment["anchor_selection"]}
            for segment in state["atomic_segments"]
        ]}
        validate_document("keyframe_selection_result", result)
        manifests["keyframe_selection.json"] = result
        manifest_index: dict[str, dict[str, str]] = {}
        artifacts = list(extra_artifacts)
        try:
            for name, payload in manifests.items():
                staged_path = staging_dir / name
                final_path = revision_dir / name
                atomic_write_json(staged_path, payload)
                manifest_index[name] = {
                    "path": self._relative(final_path),
                    "sha256": sha256_file(staged_path),
                }
                artifacts.append(str(final_path))
            state_to_write = deepcopy(state)
            state_to_write.pop("review_request", None)
            state_to_write.pop("review_submission", None)
            state_to_write.pop("keyframe_request", None)
            state_to_write.pop("keyframe_submission", None)
            state_to_write["manifest_index"] = deepcopy(manifest_index)
            staged_state_path = staging_dir / "plan_state.json"
            atomic_write_json(staged_state_path, state_to_write)
            state_path = revision_dir / "plan_state.json"
            manifest_index["plan_state.json"] = {
                "path": self._relative(state_path),
                "sha256": sha256_file(staged_state_path),
            }
            artifacts.append(str(state_path))
            os.replace(staging_dir, revision_dir)
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
        index = {
            "schema_version": "3.0",
            "tool": "replication_preprocess",
            "plan_status": state["plan_status"],
            "review_status": state["review_status"],
            "project_id": state["project_id"],
            "plan_revision": revision,
            "parent_plan_revision": state.get("parent_plan_revision"),
            "source_sha256": state["source"]["file_sha256"],
            "config_fingerprint": state["config"]["config_fingerprint"],
            "config_fingerprints": deepcopy(state["config"]["config_fingerprints"]),
            "config_sources": deepcopy(state["config"]["config_sources"]),
            "plan_fingerprint": state.get("plan_fingerprint"),
            "runtime_versions": state["runtime_versions"],
            "review_digest": state.get("review_digest", "none"),
            "manifest_index": manifest_index,
            "contact_sheet_path": state.get("contact_sheet_path"),
            "next_action": state.get("next_action"),
            "next_actions": state.get("next_actions", []),
            "anchor_status": state["anchor_status"],
            "selection_strategy": "representative-selection-v1",
            "export_history": state.get("export_history", []),
            "export": None,
            "delivery": None,
            "delivery_history": delivery_history,
        }
        atomic_write_json(self.index_path, index)
        artifacts.append(str(self.index_path))
        state_to_write["artifacts"] = artifacts
        state_to_write["index"] = index
        return state_to_write

    def _timeline_map(
        self, clips: list[dict[str, Any]], dropped: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        entries = []
        for clip in clips:
            entries.append({
                "schema_version": "1.0",
                "source_start": clip["start"],
                "source_end": clip["end"],
                "target_type": "generation_clip",
                "target_id": clip["clip_id"],
                "target_offset_s": "0",
            })
        for item in dropped:
            entries.append({
                "schema_version": "1.0",
                "source_start": item["start"],
                "source_end": item["end"],
                "target_type": "dropped_interval",
                "target_id": item["segment_id"],
                "target_offset_s": None,
            })
        return sorted(entries, key=lambda item: item["source_start"]["pts"])

    def _finish_state(
        self,
        state: dict[str, Any],
        approved_drop_ids: set[str],
    ) -> None:
        state["plan_fingerprint"] = compute_plan_fingerprint(
            state["config"], state.get("review_digest", "none")
        )
        state["config"]["plan_fingerprint"] = state["plan_fingerprint"]
        pending = [item for item in state["boundaries"] if item["relationship"] == "pending_agent"]
        uncertain = [item for item in state["boundaries"] if item["relationship"] == "uncertain"]
        state["approved_drop_ids"] = sorted(set(state.get("approved_drop_ids", [])) | approved_drop_ids
                                            | {i["segment_id"] for i in state.get("dropped_intervals", [])})
        approved_drop_ids = set(state["approved_drop_ids"])
        if pending or uncertain:
            state["generation_clips"] = []
            state["scene_groups"] = []
            state["dropped_intervals"] = []
            state["timeline_map"] = []
            state["plan_status"] = "needs_review"
            state["review_items"] = [{
                "schema_version": "1.0",
                "review_item_id": stable_id("ri", state["source"]["file_sha256"], boundary["boundary_id"]),
                "reason": "boundary_relationship_unresolved",
                "affected_boundary_ids": [boundary["boundary_id"]],
                "candidate_actions": ["review_evidence", "human_force_hard", "human_force_soft"],
            } for boundary in pending + uncertain]
            return
        groups = build_scene_groups(
            state["atomic_segments"], state["boundaries"], state["source"]["file_sha256"]
        )
        clips, dropped, review_items = build_generation_plan(
            state["atomic_segments"],
            state["boundaries"],
            state["config"],
            source_hash=state["source"]["file_sha256"],
            review_digest=state.get("review_digest", "none"),
            approved_drop_ids=approved_drop_ids,
        )
        state["scene_groups"] = groups
        state["generation_clips"] = clips
        state["dropped_intervals"] = dropped
        state["timeline_map"] = self._timeline_map(clips, dropped)
        state["review_items"] = review_items
        state["plan_status"] = "needs_review" if review_items else "ready"
        if review_items and state.get("next_action") is None:
            state["next_action"] = {
                "type": "human_plan_review",
                "reason": "no_legal_generation_plan",
            }

    def plan(self, inputs: dict[str, Any]) -> dict[str, Any]:
        source_path = Path(str(inputs.get("source_path", ""))).expanduser().resolve()
        if not source_path.is_file():
            raise ReplicationPreprocessError("plan requires an existing source_path")
        requested_profile = inputs.get("profile", "default-v1")
        config = load_config(inputs.get("config_path"), profile=requested_profile)
        selection_actions = [key for key in SELECTION_ACTIONS if inputs.get(key) is not None]
        if selection_actions:
            if len(selection_actions) != 1 or inputs.get("review_submission") is not None or inputs.get("manual_overrides"):
                raise ReviewValidationError("Submit exactly one selection action separately from boundary changes")
            return self._selection_revision(inputs, config)
        runtime_versions = collect_runtime_versions()
        parent_revision = inputs.get("parent_plan_revision")
        review_submission = inputs.get("review_submission")
        manual_overrides = inputs.get("manual_overrides") or []
        if not isinstance(manual_overrides, list):
            raise ReplicationPreprocessError("manual_overrides must be a list")
        if any(not isinstance(item, dict) for item in manual_overrides):
            raise ReplicationPreprocessError("Each manual override must be an object")
        allowed_override_actions = {
            "force_hard", "force_soft", "insert_boundary", "remove_boundary", "drop"
        }
        if any(item.get("action") not in allowed_override_actions for item in manual_overrides):
            raise ReplicationPreprocessError("Unknown manual override action")
        if manual_overrides and not parent_revision:
            raise ReplicationPreprocessError("Manual overrides require a parent plan revision")
        for item in manual_overrides:
            if item.get("parent_plan_revision") != parent_revision:
                raise ReplicationPreprocessError(
                    "Manual override parent_plan_revision does not match the request"
                )
            expected_analysis = config["config_fingerprints"]["analysis"]
            supplied_analysis = item.get("analysis_fingerprint")
            legacy_config = item.get("config_fingerprint")
            if supplied_analysis != expected_analysis and legacy_config != config["config_fingerprint"]:
                raise ReplicationPreprocessError(
                    "Manual override analysis_fingerprint does not match the active configuration"
                )
        topology_change = any(
            item.get("action") in {"insert_boundary", "remove_boundary"}
            for item in manual_overrides
        )
        if review_submission is not None and topology_change:
            raise ReplicationPreprocessError(
                "Topology overrides invalidate review evidence; submit them in a separate revision"
            )
        parent: dict[str, Any] | None = None
        if parent_revision:
            parent = self._load_state(str(parent_revision))
            if review_submission is None and not manual_overrides:
                raise ReplicationPreprocessError(
                    "A parent revision requires a review submission or manual overrides"
                )
        source_digest = sha256_file(source_path)
        if parent is not None:
            if parent["source"]["file_sha256"] != source_digest:
                raise ReplicationPreprocessError("Source file does not match parent revision")
            parent_analysis = parent.get("config", {}).get("config_fingerprints", {}).get("analysis")
            if parent_analysis != config["config_fingerprints"]["analysis"]:
                raise ReplicationPreprocessError("Analysis configuration does not match parent revision")
            if (
                review_submission is not None
                and parent.get("config", {}).get("config_fingerprints", {}).get("review")
                != config["config_fingerprints"]["review"]
            ):
                raise ReplicationPreprocessError(
                    "Review configuration changed; run without a submission to regenerate evidence"
                )
            if parent.get("runtime_versions") != runtime_versions:
                raise ReplicationPreprocessError(
                    "Media runtime versions do not match parent revision; create a fresh plan"
                )
        input_fingerprint = sha256_json({
            "source_sha256": source_digest,
            "config_fingerprint": config["config_fingerprint"],
            "parent_plan_revision": parent_revision,
            "review_submission": review_submission,
            "manual_overrides": manual_overrides,
            "runtime_versions": runtime_versions,
        })
        existing = self._find_idempotent(input_fingerprint)
        if existing:
            existing["artifacts"] = [str(self.index_path)]
            existing["index"] = self._load_json(self.index_path)
            return existing
        if parent_revision:
            if not self.index_path.is_file():
                raise ReplicationPreprocessError("Parent revision has no canonical plan index")
            current_index = self._load_json(self.index_path)
            if current_index.get("plan_revision") != parent_revision:
                raise ReplicationPreprocessError(
                    "Parent revision is stale; use the current canonical plan revision"
                )

        revision = next_revision(self.revisions_dir)
        extra_artifacts: list[str] = []
        if parent_revision and not topology_change:
            assert parent is not None
            state = deepcopy(parent)
            state.update({
                "plan_revision": revision,
                "parent_plan_revision": parent_revision,
                "input_fingerprint": input_fingerprint,
                "config": config,
                "review_request": None,
                "review_submission": None,
                "contact_sheet_path": None,
                "next_action": None,
                "executed_stages": ["review", "planning"],
                "reused_stages": ["analysis"],
                "invalidated_stages": ["review", "planning", "export"],
            })
            request_path = self._revision_dir(str(parent_revision)) / "boundary_review_request.json"
            if review_submission is not None:
                if not request_path.is_file():
                    raise ReplicationPreprocessError("Parent revision has no pending boundary review request")
                request = self._load_json(request_path)
                self._verify_request_evidence(request)
                boundaries, digest, unresolved = apply_review_submission(
                    state["boundaries"], request, review_submission
                )
                state["boundaries"] = boundaries
                state["review_digest"] = digest
                state["review_submission"] = deepcopy(review_submission)
                review_round = int(request["review_round"])
                if unresolved and review_round < int(config["review"]["max_agent_rounds"]):
                    source, ledger, time_base = probe_source(source_path)
                    pending = [item for item in boundaries if item["boundary_id"] in unresolved]
                    request2, artifacts, contact_sheet = self._build_review_artifacts(
                        source_path=source_path,
                        source=source,
                        ledger=ledger,
                        time_base=time_base,
                        segments=state["atomic_segments"],
                        pending=pending,
                        config=config,
                        revision=revision,
                        review_round=review_round + 1,
                    )
                    state["review_request"] = request2
                    state["review_status"] = "needs_more_evidence"
                    state["contact_sheet_path"] = contact_sheet
                    state["next_action"] = {
                        "type": "agent_boundary_review",
                        "skill": "skills/meta/replication-boundary-review.md",
                        "request_path": f"artifacts/replication/revisions/{revision}/boundary_review_request.json",
                    }
                    extra_artifacts.extend(artifacts)
                elif unresolved:
                    state["review_status"] = "needs_human"
                else:
                    state["review_status"] = "accepted"
            approved_drops = self._apply_non_topology_overrides(
                state["boundaries"], state["atomic_segments"], manual_overrides, config
            )
            self._finish_state(state, approved_drops)
            if review_submission is None and any(
                item["relationship"] in {"pending_agent", "uncertain"}
                for item in state["boundaries"]
            ):
                raise ReplicationPreprocessError(
                    "Manual overrides must resolve every pending boundary in the revision"
                )
            if manual_overrides and not any(
                item["relationship"] in {"pending_agent", "uncertain"}
                for item in state["boundaries"]
            ):
                state["review_status"] = "accepted"
        else:
            source, ledger, time_base = probe_source(source_path)
            qualities = analyze_frame_quality(
                source_path,
                time_base,
                int(config["keyframe"]["analysis_width_px"]),
                config,
                include_descriptors=True,
            )
            candidates = detect_scene_candidates(
                source_path,
                ledger,
                time_base,
                float(config["scene_detection"]["minimum_threshold"]),
            )
            boundaries, suppressed = select_boundaries(
                source=source,
                ledger=ledger,
                candidates=candidates,
                qualities=qualities,
                time_base=time_base,
                config=config,
            )
            apply_separator_rules(boundaries, ledger, qualities, time_base, config)
            boundaries = self._apply_topology_overrides(
                boundaries, manual_overrides, {item.pts for item in ledger}, source, config, time_base
            )
            segments = build_atomic_segments(source, boundaries, time_base, config)
            approved_drops = self._apply_non_topology_overrides(
                boundaries, segments, manual_overrides, config
            )
            pending = [item for item in boundaries if item["relationship"] == "pending_agent"]
            state = {
                "schema_version": "3.0",
                "project_id": inputs["project_id"],
                "plan_revision": revision,
                "parent_plan_revision": parent_revision,
                "input_fingerprint": input_fingerprint,
                "source": source,
                "config": config,
                "runtime_versions": runtime_versions,
                "executed_stages": ["analysis", "review", "planning"],
                "reused_stages": [],
                "invalidated_stages": ["analysis", "review", "planning", "export"],
                "boundaries": boundaries,
                "suppressed_candidates": suppressed,
                "atomic_segments": segments,
                "review_digest": "none",
                "review_submission": None,
                "review_request": None,
                "contact_sheet_path": None,
                "next_action": None,
                "review_status": "not_required",
                "quality_report": {
                    "schema_version": "1.0",
                    "keyframes_passed": sum(item.get("keyframe", {}).get("quality_status") == "passed" for item in segments),
                    "keyframes_fallback": sum(item.get("keyframe", {}).get("quality_status") == "low_quality_fallback" for item in segments),
                    "keyframes_failed": sum(item.get("keyframe", {}).get("quality_status") == "failed" for item in segments),
                    "candidate_boundary_count": len(candidates),
                    "suppressed_boundary_count": len(suppressed),
                },
            }
            self._prepare_keyframes(state, ledger=ledger, qualities=qualities)
            if pending:
                request, artifacts, contact_sheet = self._build_review_artifacts(
                    source_path=source_path,
                    source=source,
                    ledger=ledger,
                    time_base=time_base,
                    segments=segments,
                    pending=pending,
                    config=config,
                    revision=revision,
                    review_round=1,
                )
                state["review_request"] = request
                state["review_status"] = "pending_agent"
                state["contact_sheet_path"] = contact_sheet
                state["next_action"] = {
                    "type": "agent_boundary_review",
                    "skill": "skills/meta/replication-boundary-review.md",
                    "request_path": f"artifacts/replication/revisions/{revision}/boundary_review_request.json",
                }
                extra_artifacts.extend(artifacts)
            self._finish_state(state, approved_drops)
            if manual_overrides and not pending:
                state["review_status"] = "accepted"
        if any("anchor_selection" not in segment for segment in state["atomic_segments"]):
            self._prepare_keyframes(state)
        return self._write_revision(state, extra_artifacts)

    def _current_v2_state(self) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Return the canonical v2 state; v1 artifacts remain immutable and ignored."""

        if not self.index_path.is_file():
            return None
        index = self._load_json(self.index_path)
        revision = index.get("plan_revision")
        if not isinstance(revision, str):
            return None
        state = self._load_state(revision)
        fingerprints = state.get("config", {}).get("config_fingerprints")
        if state.get("schema_version") not in {"2.0", "3.0"} or not isinstance(fingerprints, dict):
            return None
        self._assert_index_integrity(index)
        return index, state

    def _auto_review_revision(
        self,
        *,
        current: dict[str, Any],
        config: dict[str, Any],
        runtime_versions: dict[str, str],
        source_path: Path,
    ) -> dict[str, Any]:
        """Reuse boundaries/keyframe choices while regenerating review evidence."""

        source, ledger, time_base = probe_source(source_path)
        revision = next_revision(self.revisions_dir)
        state = deepcopy(current)
        for boundary in state["boundaries"]:
            if (
                boundary.get("boundary_type") == "scenedetect"
                and boundary.get("relationship") != "auto_separator"
                and not boundary.get("manual_override")
            ):
                boundary["relationship"] = "pending_agent"
                boundary["hard"] = None
                boundary["hard_reason"] = None
                boundary.pop("review", None)
        state.update({
            "schema_version": "2.0",
            "plan_revision": revision,
            "parent_plan_revision": current["plan_revision"],
            "source": source,
            "config": config,
            "runtime_versions": runtime_versions,
            "review_digest": "none",
            "review_request": None,
            "review_submission": None,
            "contact_sheet_path": None,
            "next_action": None,
            "review_status": "not_required",
            "generation_clips": [],
            "scene_groups": [],
            "dropped_intervals": [],
            "timeline_map": [],
            "executed_stages": ["review"],
            "reused_stages": ["analysis"],
            "invalidated_stages": ["review", "planning", "export"],
        })
        state["input_fingerprint"] = sha256_json({
            "source_sha256": source["file_sha256"],
            "config_fingerprint": config["config_fingerprint"],
            "auto_reuse": "review",
            "parent_plan_revision": current["plan_revision"],
            "runtime_versions": runtime_versions,
        })
        artifacts: list[str] = []
        pending = [
            item for item in state["boundaries"]
            if item.get("relationship") == "pending_agent"
        ]
        if pending:
            request, review_artifacts, contact_sheet = self._build_review_artifacts(
                source_path=source_path,
                source=source,
                ledger=ledger,
                time_base=time_base,
                segments=state["atomic_segments"],
                pending=pending,
                config=config,
                revision=revision,
                review_round=1,
            )
            state["review_request"] = request
            state["review_status"] = "pending_agent"
            state["contact_sheet_path"] = contact_sheet
            state["next_action"] = {
                "type": "agent_boundary_review",
                "skill": "skills/meta/replication-boundary-review.md",
                "request_path": (
                    f"artifacts/replication/revisions/{revision}/"
                    "boundary_review_request.json"
                ),
            }
            artifacts.extend(review_artifacts)
        if any("anchor_selection" not in segment for segment in state["atomic_segments"]):
            self._prepare_keyframes(state)
        self._finish_state(state, set())
        if state["plan_status"] == "ready":
            state["executed_stages"].append("planning")
        return self._write_revision(state, artifacts)

    def _auto_planning_revision(
        self,
        *,
        current: dict[str, Any],
        config: dict[str, Any],
        runtime_versions: dict[str, str],
    ) -> dict[str, Any]:
        """Reuse accepted analysis/review results and rebuild only the clip DAG."""

        revision = next_revision(self.revisions_dir)
        state = deepcopy(current)
        state.update({
            "schema_version": "2.0",
            "plan_revision": revision,
            "parent_plan_revision": current["plan_revision"],
            "config": config,
            "runtime_versions": runtime_versions,
            "review_request": None,
            "review_submission": None,
            "next_action": None,
            "executed_stages": ["planning"],
            "reused_stages": ["analysis", "review"],
            "invalidated_stages": ["planning", "export"],
        })
        state["input_fingerprint"] = sha256_json({
            "source_sha256": state["source"]["file_sha256"],
            "config_fingerprint": config["config_fingerprint"],
            "auto_reuse": "planning",
            "parent_plan_revision": current["plan_revision"],
            "runtime_versions": runtime_versions,
        })
        request_path = (
            self._revision_dir(current["plan_revision"]) / "boundary_review_request.json"
        )
        if current.get("review_status") in {"pending_agent", "needs_more_evidence"}:
            if not request_path.is_file():
                raise ReplicationPreprocessError(
                    "Current pending review has no boundary review request"
                )
            previous_request = self._load_json(request_path)
            request = deepcopy(previous_request)
            state["review_request"] = request
            state["next_action"] = {
                "type": "agent_boundary_review",
                "skill": "skills/meta/replication-boundary-review.md",
                "request_path": (
                    f"artifacts/replication/revisions/{revision}/"
                    "boundary_review_request.json"
                ),
            }
        approved_drops = {
            item["segment_id"] for item in current.get("dropped_intervals", [])
        }
        if any("anchor_selection" not in segment for segment in state["atomic_segments"]):
            self._prepare_keyframes(state)
        self._finish_state(state, approved_drops)
        return self._write_revision(state, [])

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute with stage-local invalidation and automatic upstream reuse."""

        source_path = Path(str(inputs.get("source_path", ""))).expanduser().resolve()
        if not source_path.is_file():
            raise ReplicationPreprocessError("run requires an existing source_path")
        requested_profile = inputs.get("profile", "default-v1")
        config = load_config(inputs.get("config_path"), profile=requested_profile)

        explicit_continuation = bool(
            inputs.get("parent_plan_revision")
            or inputs.get("review_submission") is not None
            or inputs.get("manual_overrides")
            or any(inputs.get(key) is not None for key in SELECTION_ACTIONS)
        )
        plan_result: dict[str, Any]
        if explicit_continuation:
            plan_result = self.plan(inputs)
        else:
            current_pair = self._current_v2_state()
            current = current_pair[1] if current_pair else None
            runtime_versions = collect_runtime_versions()
            source_digest = sha256_file(source_path)
            if (
                current is None
                or current["source"]["file_sha256"] != source_digest
                or current.get("runtime_versions") != runtime_versions
            ):
                plan_result = self.plan(inputs)
            else:
                old = current["config"]["config_fingerprints"]
                new = config["config_fingerprints"]
                if old["analysis"] != new["analysis"]:
                    plan_result = self.plan(inputs)
                elif old["review"] != new["review"]:
                    plan_result = self._auto_review_revision(
                        current=current,
                        config=config,
                        runtime_versions=runtime_versions,
                        source_path=source_path,
                    )
                elif old["planning"] != new["planning"]:
                    plan_result = self._auto_planning_revision(
                        current=current,
                        config=config,
                        runtime_versions=runtime_versions,
                    )
                elif current.get("schema_version") != "3.0" or old.get("selection") != new["selection"]:
                    plan_result = self._selection_revision(inputs, config, current=current)
                else:
                    plan_result = deepcopy(current)
                    plan_result["config"] = config
                    plan_result["executed_stages"] = []
                    plan_result["reused_stages"] = ["analysis", "review", "planning"]
                    plan_result["invalidated_stages"] = (
                        ["export"] if old["export"] != new["export"] else []
                    )
                    plan_result["artifacts"] = [str(self.index_path)]
                    plan_result["index"] = current_pair[0]

        if plan_result["plan_status"] != "ready":
            return plan_result
        plan_executed = list(plan_result.get("executed_stages", []))
        plan_reused = list(plan_result.get("reused_stages", []))
        invalidated = list(plan_result.get("invalidated_stages", []))
        exported = self.export(inputs, config=config)
        exported["executed_stages"] = list(dict.fromkeys(
            [*plan_executed, *exported.get("executed_stages", [])]
        ))
        exported["reused_stages"] = list(dict.fromkeys(
            [*plan_reused, *exported.get("reused_stages", [])]
        ))
        exported["invalidated_stages"] = invalidated
        validation = self.validate(inputs)
        exported["validation_status"] = validation["validation_status"]
        exported["issues"] = validation["issues"]
        return exported

    def _load_current(self, plan_path: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        path = Path(plan_path).resolve() if plan_path else self.index_path
        index = self._load_json(path)
        if index.get("schema_version") not in {"2.0", "3.0"}:
            raise ReplicationPreprocessError("Unsupported replication package version")
        revision = index.get("plan_revision")
        if not revision:
            raise ReplicationPreprocessError("Plan index has no revision")
        return index, self._load_state(revision)

    def _assert_index_integrity(self, index: dict[str, Any]) -> None:
        for name, entry in index.get("manifest_index", {}).items():
            try:
                path = resolve_under(
                    self.project_dir / entry["path"], self.project_dir, must_exist=True
                )
                if sha256_file(path) != entry["sha256"]:
                    raise ReplicationPreprocessError(f"Manifest hash mismatch: {name}")
            except (KeyError, OSError, ValueError) as exc:
                raise ReplicationPreprocessError(
                    f"Manifest is missing or unsafe: {name}"
                ) from exc

    def _reusable_media(self, state: dict[str, Any], encoding: dict[str, Any]) -> tuple[dict, list]:
        wanted = {media_fingerprint(state, clip, encoding): clip for clip in state["generation_clips"]}
        assets, issues = {}, []
        index = self._load_json(self.index_path)
        references = [*state.get("export_history", []), *index.get("export_history", [])]
        if index.get("export"):
            entry = deepcopy(index["export"])
            entry.setdefault("plan_state_ref", index["manifest_index"]["plan_state.json"])
            references.append(entry)
        for reference in references:
            try:
                report_path = resolve_under(self.project_dir / reference["report_path"], self.project_dir, must_exist=True)
                if sha256_file(report_path) != reference["report_sha256"]:
                    raise ValueError("export report hash mismatch")
                report = self._load_json(report_path)
                state_ref = reference["plan_state_ref"]
                old_path = resolve_under(self.project_dir / state_ref["path"], self.project_dir, must_exist=True)
                if sha256_file(old_path) != state_ref["sha256"]:
                    raise ValueError("export source state hash mismatch")
                old_state = self._load_json(old_path)
                old_clips = {c["clip_id"]: c for c in old_state.get("generation_clips", [])}
                for asset in report.get("clips", []):
                    old_clip = old_clips.get(asset["clip_id"])
                    if not old_clip:
                        continue
                    if not asset.get("media_signature") and not legacy_trim_equivalent(old_state, old_clip):
                        continue
                    signature = media_signature(old_state, old_clip, report["encoding"])
                    fingerprint = sha256_json(signature)
                    if fingerprint not in wanted or fingerprint in assets:
                        continue
                    if asset.get("media_signature") and (asset["media_signature"] != signature or asset.get("media_fingerprint") != fingerprint):
                        issues.append({"path": asset.get("path"), "reason": "media signature mismatch"})
                        continue
                    try:
                        path = resolve_under(self.project_dir / asset["path"], self.project_dir, must_exist=True)
                        if sha256_file(path) != asset["sha256"] or asset.get("full_decode") != "passed" or asset.get("status") != "validated":
                            raise ValueError("media hash or prior decode validation mismatch")
                        probe = _probe_export(path)
                        clip = wanted[fingerprint]
                        tolerance = max(Fraction(state["source"]["nominal_frame_duration_s"]), Fraction(1, 1000))
                        if probe["video_streams"] != 1 or bool(probe["audio_streams"]) != bool(state["source"]["audio_present"]) or abs(Fraction(str(probe["duration_seconds"])) - Fraction(clip["duration_s"])) > tolerance:
                            raise ValueError("media stream or duration mismatch")
                        assets[fingerprint] = {**deepcopy(asset), "media_signature": signature, "media_fingerprint": fingerprint}
                    except (OSError, ValueError, KeyError, ReplicationPreprocessError) as exc:
                        issues.append({"path": asset.get("path"), "reason": str(exc)})
            except (OSError, ValueError, KeyError, ReplicationPreprocessError) as exc:
                issues.append({"path": reference.get("report_path"), "reason": str(exc)})
        return assets, issues

    def _cached_export_report(self, path: Path, index: dict, fingerprint: str) -> dict | None:
        try:
            references = [*index.get("export_history", []), index.get("export")]
            reference = next((r for r in references if r and r["report_path"] == self._relative(path)), None)
            if not reference or sha256_file(path) != reference["report_sha256"]:
                return None
            report = self._load_json(path)
            if report.get("export_fingerprint") != fingerprint:
                return None
            for asset in report["clips"]:
                clip = resolve_under(self.project_dir / asset["path"], self.project_dir, must_exist=True)
                if sha256_file(clip) != asset["sha256"] or asset.get("full_decode") != "passed":
                    return None
            return report
        except (OSError, ValueError, KeyError, ReplicationPreprocessError):
            return None

    def _publish_export_reference(self, index: dict, reference: dict) -> None:
        previous = index.get("export")
        history = index.setdefault("export_history", [])
        if previous and previous != reference:
            previous = deepcopy(previous)
            previous.setdefault("plan_state_ref", index["manifest_index"]["plan_state.json"])
            if previous not in history:
                history.append(previous)
        if previous != reference and index.get("delivery"):
            delivery_history = index.setdefault("delivery_history", [])
            if index["delivery"] not in delivery_history:
                delivery_history.append(deepcopy(index["delivery"]))
            index["delivery"] = None
        index["export"] = reference
        atomic_write_json(self.index_path, index)

    def export(
        self,
        inputs: dict[str, Any],
        *,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Export/cache canonical media, then publish its readable delivery copies."""
        data = self._export_media(inputs, config=config)
        index, state = self._load_current()
        data["index"] = index
        data.update(delivery_fields(None))
        if state.get("anchor_status") != "ready":
            return data
        try:
            fields, artifacts = DeliveryPackage(self.project_dir, self.index_path).publish(index, state)
            data.update(fields)
            data["artifacts"].extend(artifacts)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # Export status and canonical files stay valid; expose the separate failure.
            data.update(delivery_status="failed", delivery_error=str(exc),
                        error_code="delivery_generation_failed")
        return data

    def _export_media(
        self,
        inputs: dict[str, Any],
        *,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        index, state = self._load_current(inputs.get("plan_path"))
        self._assert_index_integrity(index)
        self._validate_plan_state(state)
        if state.get("plan_status") != "ready":
            raise ReplicationPreprocessError("Only a ready plan can be exported")
        source_path = Path(state["source"]["path"])
        if sha256_file(source_path) != state["source"]["file_sha256"]:
            raise ReplicationPreprocessError("Source file changed after planning")
        if config is None and inputs.get("config_path"):
            config = load_config(
                inputs.get("config_path"), profile=inputs.get("profile", "default-v1")
            )
        effective_config = config or state["config"]
        if config is not None:
            state_fingerprints = state["config"]["config_fingerprints"]
            for stage in ("analysis", "review", "planning"):
                if state_fingerprints[stage] != config["config_fingerprints"][stage]:
                    raise ReplicationPreprocessError(
                        f"{stage} configuration changed; run the workflow before exporting"
                    )
        plan_fingerprint = state.get("plan_fingerprint") or compute_plan_fingerprint(
            state["config"], state.get("review_digest", "none")
        )
        export_fingerprint = compute_export_fingerprint(
            effective_config, plan_fingerprint
        )
        variant = export_fingerprint[:16]
        output_dir = self.video_root / "clips" / state["plan_revision"] / variant
        report_dir = self.artifact_root / "exports" / state["plan_revision"] / variant
        report_path = report_dir / "export_report.json"
        if (index.get("export") or {}).get("export_fingerprint") == export_fingerprint:
            report_path = self.project_dir / index["export"]["report_path"]
        report = self._cached_export_report(report_path, index, export_fingerprint)
        if report is not None:
            export_reference = {
                "status": "validated",
                "export_fingerprint": export_fingerprint,
                "export_config_fingerprint": effective_config["config_fingerprints"]["export"],
                "config_sources": deepcopy(effective_config["config_sources"]),
                "report_path": self._relative(report_path),
                "report_sha256": sha256_file(report_path),
                "plan_state_ref": deepcopy(index["manifest_index"]["plan_state.json"]),
            }
            self._publish_export_reference(index, export_reference)
            return {
                "plan_status": "ready",
                "review_status": state["review_status"],
                "plan_revision": state["plan_revision"],
                "export_status": "validated",
                "clip_paths": [item["path"] for item in report.get("clips", [])],
                "export_report_path": self._relative(report_path),
                "export_fingerprint": export_fingerprint,
                "config_fingerprints": deepcopy(effective_config["config_fingerprints"]),
                "config_sources": deepcopy(effective_config["config_sources"]),
                "executed_stages": [],
                "reused_stages": ["export"],
                "invalidated_stages": [],
                "artifacts": [
                    str(report_path),
                    *[str(self.project_dir / item["path"]) for item in report.get("clips", [])],
                    str(self.index_path),
                ],
                "index": index,
            }
        source_path = Path(state["source"]["path"])
        if sha256_file(source_path) != state["source"]["file_sha256"]:
            raise ReplicationPreprocessError("Source file changed after planning")
        if report_path.exists():
            recovery = 1
            while (self.artifact_root / "exports" / state["plan_revision"] / f"{variant}-repair-{recovery}" / "export_report.json").exists():
                recovery += 1
            variant = f"{variant}-repair-{recovery}"
            output_dir = self.video_root / "clips" / state["plan_revision"] / variant
            report_path = self.artifact_root / "exports" / state["plan_revision"] / variant / "export_report.json"
        reusable, reuse_issues = self._reusable_media(state, effective_config["export"])
        output_dir.mkdir(parents=True, exist_ok=True)
        export_config = effective_config["export"]
        reports: list[dict[str, Any]] = []
        artifacts: list[str] = []
        created: list[Path] = []
        staged: list[tuple[Path, Path]] = []
        try:
            for clip in state["generation_clips"]:
                fingerprint = media_fingerprint(state, clip, export_config)
                if fingerprint in reusable:
                    reused = deepcopy(reusable[fingerprint])
                    reused.update(clip_id=clip["clip_id"], reused=True,
                                  anchor_ids=[a["anchor_id"] for a in clip.get("keyframes", [])])
                    reports.append(reused)
                    artifacts.append(str(self.project_dir / reused["path"]))
                    continue
                final_path = output_dir / f"{clip['clip_id']}.mp4"
                temp_path = output_dir / f".{clip['clip_id']}.part.mp4"
                start = clip["start"]["seconds"]
                end = clip["end"]["seconds"]
                video_filter = (
                    f"trim=start_pts={clip['start']['pts']}:end_pts={clip['end']['pts']},setpts=PTS-{clip['start']['pts']},"
                    "scale=trunc(iw/2)*2:trunc(ih/2)*2"
                )
                command = ["ffmpeg", "-v", "error", "-y", "-copyts", "-i", str(source_path)]
                if state["source"]["audio_present"]:
                    audio_filter = f"atrim=start={start}:end={end},asetpts=PTS-{start}/TB"
                    audio_stream_index = int(state["source"]["audio_stream_index"])
                    command += [
                        "-filter_complex",
                        f"[0:v:0]{video_filter}[v];[0:{audio_stream_index}]{audio_filter}[a]",
                        "-map", "[v]", "-map", "[a]",
                    ]
                else:
                    command += ["-vf", video_filter, "-map", "0:v:0"]
                command += [
                    "-c:v", export_config["video_codec"],
                    "-crf", str(export_config["crf"]),
                    "-preset", export_config["preset"],
                    "-pix_fmt", export_config["pixel_format"],
                    "-fps_mode", "passthrough",
                ]
                if state["source"]["audio_present"]:
                    command += ["-c:a", export_config["audio_codec"], "-b:a", export_config["audio_bitrate"]]
                command += ["-metadata:s:v:0", "rotate=0", "-movflags", "+faststart", str(temp_path)]
                completed = subprocess.run(command, capture_output=True, text=True, check=False)
                if completed.returncode != 0:
                    raise ReplicationPreprocessError(completed.stderr.strip() or "FFmpeg export failed")
                decode = subprocess.run(
                    ["ffmpeg", "-v", "error", "-i", str(temp_path), "-f", "null", "-"],
                    capture_output=True, text=True, check=False,
                )
                if decode.returncode != 0:
                    raise ReplicationPreprocessError(f"Exported clip failed full decode: {clip['clip_id']}")
                probe = _probe_export(temp_path)
                duration = Fraction(str(probe["duration_seconds"]))
                minimum = Fraction(str(state["config"]["profile"]["min_duration_s"]))
                maximum = Fraction(str(state["config"]["profile"]["max_duration_s"]))
                if duration < minimum or duration >= maximum:
                    raise ReplicationPreprocessError(
                        f"Exported clip duration violates profile: {clip['clip_id']}={duration}"
                    )
                expected_duration = Fraction(clip["duration_s"])
                tolerance = max(
                    Fraction(str(state["source"]["nominal_frame_duration_s"])),
                    Fraction(1, 1000),
                )
                if abs(duration - expected_duration) > tolerance:
                    raise ReplicationPreprocessError(
                        f"Exported clip duration does not match its plan: {clip['clip_id']}"
                    )
                if probe["video_streams"] != 1 or probe["audio_streams"] not in {0, 1}:
                    raise ReplicationPreprocessError("Exported clip has an unexpected stream layout")
                report = {
                    "media_fingerprint": fingerprint,
                    "media_signature": media_signature(state, clip, export_config),
                    "anchor_ids": [a["anchor_id"] for a in clip.get("keyframes", [])],
                    "reused": False,
                    "clip_id": clip["clip_id"],
                    "path": self._relative(final_path),
                    "sha256": sha256_file(temp_path),
                    "duration_seconds": probe["duration_seconds"],
                    "video_streams": probe["video_streams"],
                    "audio_streams": probe["audio_streams"],
                    "full_decode": "passed",
                    "status": "validated",
                }
                if bool(probe["audio_streams"]) != bool(state["source"]["audio_present"]):
                    raise ReplicationPreprocessError("Exported clip audio presence does not match source")
                reports.append(report)
                artifacts.append(str(final_path))
                staged.append((temp_path, final_path))
            for temp_path, final_path in staged:
                os.replace(temp_path, final_path)
                created.append(final_path)
        except Exception:
            for path in output_dir.glob("*.part.mp4"):
                path.unlink(missing_ok=True)
            for path in created:
                path.unlink(missing_ok=True)
            raise
        export_report = {
            "schema_version": "2.0",
            "plan_revision": state["plan_revision"],
            "plan_fingerprint": plan_fingerprint,
            "export_fingerprint": export_fingerprint,
            "export_config_fingerprint": effective_config["config_fingerprints"]["export"],
            "config_sources": deepcopy(effective_config["config_sources"]),
            "status": "validated",
            "encoding": deepcopy(export_config),
            "reuse_issues": reuse_issues,
            "clips": reports,
        }
        atomic_write_json(report_path, export_report)
        artifacts.append(str(report_path))
        export_reference = {
            "status": "validated",
            "export_fingerprint": export_fingerprint,
            "export_config_fingerprint": effective_config["config_fingerprints"]["export"],
            "config_sources": deepcopy(effective_config["config_sources"]),
            "report_path": self._relative(report_path),
            "report_sha256": sha256_file(report_path),
            "plan_state_ref": deepcopy(index["manifest_index"]["plan_state.json"]),
        }
        self._publish_export_reference(index, export_reference)
        artifacts.append(str(self.index_path))
        return {
            "plan_status": "ready",
            "review_status": state["review_status"],
            "plan_revision": state["plan_revision"],
            "export_status": "validated",
            "clip_paths": [item["path"] for item in reports],
            "export_report_path": self._relative(report_path),
            "export_fingerprint": export_fingerprint,
            "config_fingerprints": deepcopy(effective_config["config_fingerprints"]),
            "config_sources": deepcopy(effective_config["config_sources"]),
            "executed_stages": ["export"] if any(not r.get("reused") for r in reports) else [],
            "reused_stages": ["export"] if any(r.get("reused") for r in reports) else [],
            "invalidated_stages": [],
            "artifacts": artifacts,
            "index": index,
        }

    def validate(self, inputs: dict[str, Any]) -> dict[str, Any]:
        index, state = self._load_current(inputs.get("plan_path"))
        issues: list[str] = []
        try:
            if sha256_file(Path(state["source"]["path"])) != state["source"]["file_sha256"]:
                issues.append("source_hash_mismatch")
        except OSError:
            issues.append("source_missing")
        for name, entry in index.get("manifest_index", {}).items():
            try:
                path = resolve_under(self.project_dir / entry["path"], self.project_dir, must_exist=True)
                if sha256_file(path) != entry["sha256"]:
                    issues.append(f"manifest_hash_mismatch:{name}")
            except (KeyError, OSError, ValueError):
                issues.append(f"manifest_missing_or_unsafe:{name}")
        try:
            self._validate_plan_state(state)
        except (ReplicationPreprocessError, OSError, ValueError, KeyError):
            issues.append("plan_state_invalid")
        request_entry = index.get("manifest_index", {}).get("boundary_review_request.json")
        if request_entry:
            try:
                request = self._load_json(self.project_dir / request_entry["path"])
                self._verify_request_evidence(request)
            except (ReviewValidationError, OSError, ValueError, KeyError):
                issues.append("review_evidence_invalid")
        segments = state.get("atomic_segments", [])
        for left, right in zip(segments, segments[1:]):
            if left["end"]["pts"] != right["start"]["pts"]:
                issues.append("atomic_timeline_gap_or_overlap")
                break
        clip_ids = {item["clip_id"] for item in state.get("generation_clips", [])}
        if len(clip_ids) != len(state.get("generation_clips", [])):
            issues.append("duplicate_generation_clip_id")
        export_info = index.get("export")
        if export_info:
            try:
                report_path = resolve_under(
                    self.project_dir / export_info["report_path"], self.project_dir, must_exist=True
                )
                if sha256_file(report_path) != export_info["report_sha256"]:
                    issues.append("export_report_hash_mismatch")
                else:
                    report = self._load_json(report_path)
                    for clip in report.get("clips", []):
                        clip_path = resolve_under(
                            self.project_dir / clip["path"], self.project_dir, must_exist=True
                        )
                        if sha256_file(clip_path) != clip["sha256"]:
                            issues.append(f"clip_hash_mismatch:{clip['clip_id']}")
            except (KeyError, OSError, ValueError):
                issues.append("export_artifact_missing_or_unsafe")
        delivery, delivery_issues = DeliveryPackage(self.project_dir, self.index_path).validate(index, state)
        issues.extend(delivery_issues)
        return {
            "plan_status": state["plan_status"],
            "review_status": state["review_status"],
            "plan_revision": state["plan_revision"],
            "plan_fingerprint": state.get("plan_fingerprint"),
            "export_fingerprint": (index.get("export") or {}).get("export_fingerprint"),
            "config_fingerprints": deepcopy(
                state.get("config", {}).get("config_fingerprints", {})
            ),
            "config_sources": deepcopy(
                state.get("config", {}).get("config_sources", [])
            ),
            "executed_stages": [],
            "reused_stages": ["analysis", "review", "planning"],
            "invalidated_stages": [],
            "validation_status": "passed" if not issues else "failed",
            "issues": issues,
            "artifacts": [str(self.index_path)],
            "index": index,
            **delivery,
        }


def _probe_export(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        raise ReplicationPreprocessError("ffprobe failed for exported clip")
    data = json.loads(completed.stdout)
    streams = data.get("streams", [])
    duration = data.get("format", {}).get("duration")
    if duration is None:
        video = next((item for item in streams if item.get("codec_type") == "video"), None)
        duration = video.get("duration") if video else None
    if duration is None:
        raise ReplicationPreprocessError("Exported clip has no measurable duration")
    return {
        "duration_seconds": str(duration),
        "video_streams": sum(item.get("codec_type") == "video" for item in streams),
        "audio_streams": sum(item.get("codec_type") == "audio" for item in streams),
    }
