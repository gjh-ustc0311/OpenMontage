"""V2 immutable state machine for direct scene generation.

The engine never calls a generation provider.  It freezes the exact prompt and
ordered image references that the coding Agent must pass to the built-in
``image_gen`` tool, imports that tool's full-canvas result, and persists the
Agent/human review decisions.  Masks and compositing are deliberately absent.
"""

from __future__ import annotations

import hashlib
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable

from lib.replication_preprocess.models import stable_id
from schemas.artifacts import validate_artifact
from schemas.replication import validate_phase2_document

from .errors import SceneReplacementError
from .imaging import (
    DIRECT_RESULT_MAX_ASPECT_RATIO_DIFFERENCE_PERCENT,
    DIRECT_RESULT_TARGET_SIZE,
    ImageAssetError,
    copy_image_asset,
    import_image_asset,
    normalize_direct_result,
)
from .quality import (
    MULTIFRAME_RENDERER_VERSION,
    QualityEvidenceError,
    generate_direct_comparison,
    generate_direct_review_crops,
    generate_multiframe_comparison,
)
from .storage import (
    REVISION_RE,
    atomic_write_json,
    load_json,
    next_revision,
    resolve_under,
    sha256_file,
    sha256_json,
    verify_ref,
    write_revision_directory,
)
from .state_core import SceneReplacementStateCore
from .upstream import (
    anchor_fingerprint,
    build_source_snapshot,
    presentation_map_from_delivery,
    verify_source_snapshot,
)


OPERATIONS = {
    "initialize",
    "apply_scene_plan",
    "record_direction_selection",
    "prepare_edit",
    "mark_dispatched",
    "import_result",
    "reconcile_execution",
    "prepare_scene_review",
    "prepare_global_review",
    "apply_review",
    "publish",
    "validate",
}

PRE_GENERATION_RULES = {
    "source_snapshot_binding",
    "anchor_binding",
    "time_binding",
    "scene_design_binding",
    "reference_binding",
    "functional_space",
    "contact",
    "occlusion",
    "perspective",
    "lighting",
    "prompt_role_distinction",
    "north_american_context",
    "target_consumer_fit",
    "authentic_consumer_environment",
    "non_studio_aesthetic",
    "rework_mode_binding",
}

TARGET_SIZE_PROMPT_RE = re.compile(r"(?<!\d)720\s*(?:x|X|×|\*)\s*1280(?!\d)")

GENERATED_RESULT_RULES = {
    "result_file_integrity",
    "request_binding",
    "subject_preservation",
    "product_detail_fidelity",
    "composition_correspondence",
    "functional_space",
    "contact",
    "occlusion",
    "perspective",
    "lighting",
    "significant_difference",
    "scene_consistency",
    "north_american_context",
    "target_consumer_fit",
    "authentic_consumer_environment",
    "non_studio_aesthetic",
    "artifact_free",
}

HARD_RESULT_RULES = {
    "subject_preservation",
    "product_detail_fidelity",
    "composition_correspondence",
    "functional_space",
    "contact",
    "occlusion",
    "perspective",
    "lighting",
    "significant_difference",
    "scene_consistency",
    "north_american_context",
    "target_consumer_fit",
    "authentic_consumer_environment",
    "non_studio_aesthetic",
}

SAMPLE_RULES = {
    "main_reference_quality",
    "subject_preservation",
    "functional_space",
    "contact",
    "occlusion",
    "perspective",
    "lighting",
    "north_american_context",
    "target_consumer_fit",
    "authentic_consumer_environment",
    "non_studio_aesthetic",
}

GROUP_RULES = {
    "all_anchors_adopted",
    "scene_consistency",
    "shared_constraints",
    "north_american_context",
    "target_consumer_fit",
    "authentic_consumer_environment",
    "non_studio_aesthetic",
}

GLOBAL_RULES = {
    "all_scenes_human_approved",
    "all_anchors_present",
    "chronological_order",
    "source_result_binding",
    "readable_naming",
}

SEVERITY_RANK = {"minor": 1, "major": 2, "critical": 3}
AUDIENCE_PROFILE = {
    "market": "north_america",
    "age_min": 30,
    "age_max": 50,
    "environment_policy": "ordinary_consumer_real_use",
    "staging_policy": "non_studio",
}

STATE_SCHEMA_VERSION = "2.1"
LEGACY_DIRECT_STATE_VERSION = "2.0"
PROMPT_BUILDER_VERSION = "direct-scene-v1"


def _human(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("kind") == "human"
        and bool(str(value.get("actor") or value.get("name") or "").strip())
    )


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SceneReplacementError(f"{label} must be a non-empty string")
    return value.strip()


def _submission(inputs: dict[str, Any]) -> dict[str, Any]:
    value = inputs.get("submission")
    if not isinstance(value, dict):
        raise SceneReplacementError("This operation requires an object submission")
    return deepcopy(value)


def _validate_document(name: str, value: dict[str, Any]) -> None:
    try:
        validate_phase2_document(name, value)
    except Exception as exc:
        raise SceneReplacementError(f"{name} contract violation: {exc}") from exc


def _reference(value: dict[str, Any]) -> dict[str, str]:
    return {"path": value["path"], "sha256": value["sha256"]}


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _rules(review: dict[str, Any]) -> dict[str, str]:
    outcomes: dict[str, str] = {}
    for item in review.get("rules", []):
        rule_id = item.get("rule_id") if isinstance(item, dict) else None
        if not isinstance(rule_id, str) or rule_id in outcomes:
            raise SceneReplacementError("Review rule IDs must be non-empty and unique")
        outcomes[rule_id] = str(item.get("status"))
    return outcomes


def _require_rules(review: dict[str, Any], required: set[str], label: str) -> None:
    outcomes = _rules(review)
    missing = sorted(required - {key for key, value in outcomes.items() if value == "pass"})
    if missing:
        raise SceneReplacementError(
            f"{label} review is missing passing required rules: {', '.join(missing)}"
        )


def _require_rule_coverage(review: dict[str, Any], required: set[str], label: str) -> None:
    missing = sorted(required - set(_rules(review)))
    if missing:
        raise SceneReplacementError(
            f"{label} review did not evaluate required rules: {', '.join(missing)}"
        )


class SceneReplacementV2Engine(SceneReplacementStateCore):
    """Persist the sole supported direct-generation scene-replacement workflow."""

    def __init__(self, project_dir: Path, output_path: Path):
        self.project_dir = project_dir.expanduser().resolve(strict=True)
        if not self.project_dir.is_dir():
            raise SceneReplacementError("project_dir must be an existing directory")
        try:
            self.output_path = resolve_under(output_path.expanduser(), self.project_dir)
        except (OSError, ValueError) as exc:
            raise SceneReplacementError("output_path must stay inside the project workspace") from exc
        expected = (
            self.project_dir
            / "artifacts"
            / "replication"
            / "scene-replacement-v2"
            / "index.json"
        )
        if self.output_path != expected:
            raise SceneReplacementError(
                "output_path must be artifacts/replication/scene-replacement-v2/index.json"
            )
        if self.output_path.exists() and (
            not self.output_path.is_file() or self.output_path.is_symlink()
        ):
            raise SceneReplacementError("output_path must be a regular JSON file")
        self.artifact_root = expected.parent
        self.revisions_dir = self.artifact_root / "revisions"
        self.asset_root = (
            self.project_dir / "assets" / "images" / "replication" / "scene-replacement-v2"
        )

    def execute(self, operation: str, inputs: dict[str, Any]) -> dict[str, Any]:
        if operation not in OPERATIONS:
            raise SceneReplacementError(f"Unsupported V2 operation: {operation}")
        project_id = inputs.get("project_id")
        self.validate_project_id(project_id)
        if self.project_dir.name != project_id:
            raise SceneReplacementError("project_id does not match project_dir")
        if operation == "validate":
            return self.validate(inputs)
        return getattr(self, operation)(inputs)

    def _read_index(self) -> dict[str, Any]:
        if not self.output_path.is_file():
            raise SceneReplacementError("Scene-replacement V2 package has not been initialized")
        index = load_json(self.output_path, self.project_dir)
        if (
            index.get("schema_version") not in {LEGACY_DIRECT_STATE_VERSION, STATE_SCHEMA_VERSION}
            or index.get("tool") != "replication_scene_replacement"
            or index.get("project_id") != self.project_dir.name
            or not REVISION_RE.fullmatch(str(index.get("replacement_revision", "")))
        ):
            raise SceneReplacementError("Invalid scene-replacement V2 canonical index")
        manifests = index.get("manifest_index")
        if not isinstance(manifests, dict) or set(manifests) < {"state.json", "source_snapshot.json"}:
            raise SceneReplacementError("Scene-replacement V2 manifest index is incomplete")
        for name, reference in manifests.items():
            verify_ref(reference, self.project_dir, label=f"scene-replacement-v2 {name}")
        return index

    def _load_current(self) -> tuple[dict[str, Any], dict[str, Any]]:
        index = self._read_index()
        state = load_json(
            verify_ref(index["manifest_index"]["state.json"], self.project_dir, label="V2 state"),
            self.project_dir,
        )
        if (
            state.get("replacement_revision") != index["replacement_revision"]
            or state.get("project_id") != index["project_id"]
            or state.get("tool") != index["tool"]
        ):
            raise SceneReplacementError("Canonical V2 index and immutable state differ")
        snapshot = load_json(
            verify_ref(
                index["manifest_index"]["source_snapshot.json"],
                self.project_dir,
                label="V2 source snapshot",
            ),
            self.project_dir,
        )
        if snapshot != state.get("source_snapshot"):
            raise SceneReplacementError("V2 state embeds a different source snapshot")
        _validate_document("scene_replacement_source_snapshot", snapshot)
        _validate_document("scene_replacement_v2_state", state)
        return index, state

    def _write_state(
        self,
        state: dict[str, Any],
        *,
        operation: str,
        fingerprint: str,
        expected_index: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if expected_index is None:
            if self.output_path.exists():
                raise SceneReplacementError("Canonical V2 index appeared during initialization")
        elif self._read_index() != expected_index:
            raise SceneReplacementError("Canonical V2 index changed before publication; retry")
        state.setdefault("operation_history", []).append(
            {
                "operation": operation,
                "fingerprint": fingerprint,
                "parent_replacement_revision": state.get("parent_replacement_revision"),
                "replacement_revision": state["replacement_revision"],
            }
        )
        _validate_document("scene_replacement_v2_state", state)
        references = write_revision_directory(
            self.revisions_dir,
            state["replacement_revision"],
            {"source_snapshot.json": state["source_snapshot"], "state.json": state},
            project_dir=self.project_dir,
        )
        index = {
            "schema_version": STATE_SCHEMA_VERSION,
            "tool": "replication_scene_replacement",
            "project_id": state["project_id"],
            "replacement_revision": state["replacement_revision"],
            "parent_replacement_revision": state.get("parent_replacement_revision"),
            "package_status": self._package_status(state),
            "source_snapshot_id": state["source_snapshot"]["snapshot_id"],
            "source_snapshot_fingerprint": state["source_snapshot"]["fingerprints"]["snapshot"],
            "manifest_index": references,
            "last_operation": deepcopy(state["operation_history"][-1]),
            "delivery": deepcopy(state.get("delivery")),
        }
        if expected_index is not None and self._read_index() != expected_index:
            raise SceneReplacementError("Canonical V2 index changed before publication; retry")
        atomic_write_json(self.output_path, index)
        return self._result(state, index)

    def _result(self, state: dict[str, Any], index: dict[str, Any]) -> dict[str, Any]:
        value = deepcopy(state)
        value.update(
            state=deepcopy(state),
            index=deepcopy(index),
            index_path=self.relative(self.output_path),
            package_status=self._package_status(state),
            validation_status="not_run",
            issues=[],
            next_actions=self._next_actions(state),
            checkpoint_artifacts=self._checkpoint_artifacts(state, index),
            artifacts=[
                str(self.output_path),
                *[
                    str(self.project_dir / reference["path"])
                    for reference in index["manifest_index"].values()
                ],
            ],
        )
        prepared = [
            request
            for request in state.get("edit_requests", {}).values()
            if request.get("status") == "prepared"
            and request.get("dependency_status") == "current"
        ]
        if prepared:
            request = max(prepared, key=lambda item: item["preparation_ordinal"])
            value["generation_packet"] = self._generation_packet(request)
        return value

    @staticmethod
    def _generation_packet(request: dict[str, Any]) -> dict[str, Any]:
        return {
            "request_id": request["request_id"],
            "attempt_number": request["attempt_number"],
            "generation_mode": request["generation_mode"],
            "prompt": request["prompt"],
            "prompt_sha256": request["prompt_sha256"],
            "prompt_builder_version": request.get("prompt_builder_version"),
            "ordered_image_references": deepcopy(
                request.get("ordered_image_references", request.get("ordered_references", []))
            ),
            "context_refs": deepcopy(request.get("context_refs", [])),
            "requested_capability": "imagegen2",
            "required_actual_tool": "image_gen",
            "output_requirement": {
                "width": DIRECT_RESULT_TARGET_SIZE[0],
                "height": DIRECT_RESULT_TARGET_SIZE[1],
                "aspect_ratio": "9:16",
                "max_aspect_ratio_difference_percent": (
                    DIRECT_RESULT_MAX_ASPECT_RATIO_DIFFERENCE_PERCENT
                ),
            },
        }

    def _checkpoint_artifacts(
        self, state: dict[str, Any], index: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        index_ref = {"path": self.relative(self.output_path), "sha256": sha256_file(self.output_path)}
        snapshot_ref = deepcopy(index["manifest_index"]["source_snapshot.json"])
        snapshot = state["source_snapshot"]
        artifacts: dict[str, dict[str, Any]] = {
            "replication_source_snapshot": {
                "version": STATE_SCHEMA_VERSION,
                "project_id": state["project_id"],
                "replacement_revision": state["replacement_revision"],
                "status": "ready",
                "current_index_ref": index_ref,
                "source_snapshot_ref": snapshot_ref,
                "source_snapshot_fingerprint": snapshot["fingerprints"]["snapshot"],
                "upstream_plan_revision": snapshot["upstream"]["plan_revision"],
                "anchor_count": len(snapshot["anchors"]),
            }
        }
        plan = state.get("scene_plan")
        if plan:
            scenes = []
            blockers = self._blockers(state)
            for scene in plan["physical_scenes"]:
                group_review = self._review_by_id(state, scene.get("group_review_id"))
                scenes.append(
                    {
                        "scene_id": scene["scene_id"],
                        "status": self._checkpoint_scene_status(state, scene),
                        "anchor_ids": deepcopy(scene["anchor_ids"]),
                        "group_human_approved": bool(
                            group_review
                            and group_review.get("status") == "pass"
                            and group_review.get("dependency_status") == "current"
                            and _human(group_review.get("human_confirmation"))
                        ),
                    }
                )
            package_status = self._package_status(state)
            if package_status == "published":
                package_status = "ready"
            artifacts["scene_replacement_package"] = {
                "version": STATE_SCHEMA_VERSION,
                "project_id": state["project_id"],
                "replacement_revision": state["replacement_revision"],
                "status": "blocked" if blockers else package_status,
                "current_index_ref": index_ref,
                "source_snapshot_ref": snapshot_ref,
                "counts": {
                    "scenes_total": len(scenes),
                    "scenes_approved": sum(item["group_human_approved"] for item in scenes),
                    "anchors_total": len(snapshot["anchors"]),
                    "anchors_adopted": len(state.get("adopted_results", {})),
                },
                "scenes": scenes,
                "blockers": blockers,
                "global_comparison_ref": _reference(
                    state["global_review_candidate"]["global_comparison"]
                )
                if state.get("global_review_candidate")
                and state["global_review_candidate"].get("dependency_status") == "current"
                else None,
                "global_review_id": state.get("global_review_id"),
                "global_human_approved": bool(
                    state.get("global_review_candidate")
                    and (
                        self._review_by_id(state, state.get("global_review_id"))
                        or {}
                    ).get("status")
                    == "pass"
                    and (
                        self._review_by_id(state, state.get("global_review_id"))
                        or {}
                    ).get("scope_id")
                    == state["global_review_candidate"].get("candidate_id")
                ),
            }
        if state.get("delivery"):
            delivery = state["delivery"]
            artifacts["scene_replacement_delivery"] = {
                "version": STATE_SCHEMA_VERSION,
                "project_id": state["project_id"],
                "replacement_revision": state["replacement_revision"],
                "status": "ready",
                "delivery_manifest_ref": _reference(delivery),
                "delivery_fingerprint": delivery["delivery_fingerprint"],
                "physical_scene_count": len(plan["physical_scenes"]),
                "accepted_result_count": len(state["adopted_results"]),
                "downstream_adaptation_status": "unverified",
            }
        for name, artifact in artifacts.items():
            validate_artifact(name, artifact)
        return artifacts

    @staticmethod
    def _review_by_id(state: dict[str, Any], review_id: str | None) -> dict[str, Any] | None:
        if not review_id:
            return None
        return next((item for item in state.get("reviews", []) if item["review_id"] == review_id), None)

    def _blockers(self, state: dict[str, Any]) -> list[str]:
        blockers = []
        plan = state.get("scene_plan") or {}
        if plan.get("unassigned_anchor_ids"):
            blockers.append("Scene plan has unassigned anchors")
        for request in state.get("edit_requests", {}).values():
            if request.get("dependency_status") == "current" and request.get("execution_status") == "unknown":
                blockers.append(f"Unknown image_gen execution: {request['request_id']}")
        for task in state.get("logical_tasks", {}).values():
            if (
                task.get("dependency_status") == "current"
                and task.get("status") == "exhausted"
                and task["anchor_id"] not in state.get("adopted_results", {})
            ):
                blockers.append(f"Attempt budget exhausted: {task['task_id']}")
        return sorted(set(blockers))

    @staticmethod
    def _checkpoint_scene_status(state: dict[str, Any], scene: dict[str, Any]) -> str:
        if scene.get("dependency_status") != "current":
            return "blocked"
        if not scene.get("selection_receipt_id"):
            return "awaiting_direction"
        main = scene.get("main_anchor_id")
        if not main or main not in state.get("adopted_results", {}):
            task = next(
                (
                    item
                    for item in state.get("logical_tasks", {}).values()
                    if item.get("anchor_id") == main and item.get("dependency_status") == "current"
                ),
                None,
            )
            if task and task.get("latest_result_id"):
                result = state.get("results", {}).get(task["latest_result_id"], {})
                if result.get("status") == "rejected":
                    return "needs_rework"
            return "awaiting_generation"
        if not scene.get("sample_review_id"):
            return "awaiting_sample"
        difficult = scene.get("difficult_view", {})
        if difficult.get("status") == "required" and difficult.get("anchor_id") not in state.get("adopted_results", {}):
            return "needs_difficult_view"
        if any(anchor not in state.get("adopted_results", {}) for anchor in scene["anchor_ids"]):
            return "expanding"
        candidate = state.get("scene_review_candidates", {}).get(scene["scene_id"])
        if not candidate or candidate.get("dependency_status") != "current":
            return "awaiting_group_evidence"
        if not scene.get("group_review_id"):
            return "awaiting_group_approval"
        return "ready"

    def _package_status(self, state: dict[str, Any]) -> str:
        if state.get("delivery"):
            return "published"
        if self._blockers(state):
            return "blocked"
        plan = state.get("scene_plan")
        if not plan:
            return "source_locked"
        if plan.get("status") != "ready":
            return "awaiting_human"
        anchors = {item["anchor_id"] for item in state["source_snapshot"]["anchors"]}
        if set(state.get("adopted_results", {})) == anchors and all(
            scene.get("sample_review_id") and scene.get("group_review_id")
            for scene in plan["physical_scenes"]
        ):
            candidate = state.get("global_review_candidate")
            if not candidate or candidate.get("dependency_status") != "current":
                return "in_progress"
            review = self._review_by_id(state, state.get("global_review_id"))
            if (
                review
                and review.get("status") == "pass"
                and review.get("dependency_status") == "current"
                and review.get("scope_id") == candidate.get("candidate_id")
                and review.get("candidate_fingerprint")
                == candidate.get("dependency_fingerprint")
                and _human(review.get("human_confirmation"))
            ):
                return "ready"
            return "awaiting_global_approval"
        return "in_progress"

    def _next_actions(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        if not state.get("scene_plan"):
            return [{"type": "agent_scene_analysis", "direction_count": 3}]
        actions: list[dict[str, Any]] = []
        for scene in state["scene_plan"]["physical_scenes"]:
            if scene.get("dependency_status") != "current":
                actions.append({"type": "reconfirm_scene_plan", "scene_id": scene["scene_id"]})
                continue
            if not scene.get("selection_receipt_id"):
                actions.append({"type": "human_direction_selection", "scene_id": scene["scene_id"]})
        if state["scene_plan"].get("status") != "ready":
            return actions

        def generation_action(anchor_id: str) -> dict[str, Any]:
            task = self._active_task(state, anchor_id, required=False)
            if task and task.get("status") == "exhausted":
                return {
                    "type": "blocked_attempt_budget_exhausted",
                    "anchor_id": anchor_id,
                    "logical_task_id": task["task_id"],
                    "human_decision_required": True,
                }
            if task:
                current = state["edit_requests"].get(task.get("current_request_id"))
                if current and current.get("execution_status") == "confirmed_failed":
                    return {
                        "type": "prepare_rework",
                        "anchor_id": anchor_id,
                        "strategy": "source_regeneration",
                        "trigger_execution_request_id": current["request_id"],
                    }
                if task.get("latest_result_id"):
                    result = state["results"].get(task["latest_result_id"], {})
                    if result.get("status") == "rejected":
                        review = self._latest_review(
                            state, "generated_result", result["result_id"]
                        )
                        return {
                            "type": "prepare_rework",
                            "anchor_id": anchor_id,
                            "strategy": (review or {})
                            .get("rework_decision", {})
                            .get("strategy"),
                        }
            return {"type": "prepare_generation", "anchor_id": anchor_id}

        for scene in state["scene_plan"]["physical_scenes"]:
            main = scene["main_anchor_id"]
            if main not in state["adopted_results"]:
                actions.append(generation_action(main))
                continue
            sample = self._review_by_id(state, scene.get("sample_review_id"))
            if not sample or sample.get("status") != "pass":
                actions.append({"type": "human_sample_review", "scene_id": scene["scene_id"]})
                continue
            difficult = scene["difficult_view"]
            difficult_anchor = difficult.get("anchor_id")
            if difficult.get("status") == "required" and difficult_anchor not in state["adopted_results"]:
                actions.append(generation_action(difficult_anchor))
                continue
            remaining = [
                anchor_id
                for anchor_id in scene["anchor_ids"]
                if anchor_id not in state["adopted_results"]
            ]
            if remaining:
                actions.extend(generation_action(anchor_id) for anchor_id in remaining)
                continue
            candidate = state.get("scene_review_candidates", {}).get(scene["scene_id"])
            if not candidate or candidate.get("dependency_status") != "current":
                actions.append({"type": "prepare_scene_review", "scene_id": scene["scene_id"]})
                continue
            group = self._review_by_id(state, scene.get("group_review_id"))
            if not group or group.get("status") != "pass":
                actions.append(
                    {
                        "type": "human_group_review",
                        "scene_id": scene["scene_id"],
                        "candidate_id": candidate["candidate_id"],
                    }
                )
        if not actions:
            candidate = state.get("global_review_candidate")
            if not candidate or candidate.get("dependency_status") != "current":
                actions.append({"type": "prepare_global_review"})
            else:
                review = self._review_by_id(state, state.get("global_review_id"))
                if not review or review.get("status") != "pass":
                    actions.append(
                        {
                            "type": "human_global_review",
                            "candidate_id": candidate["candidate_id"],
                        }
                    )
                elif not state.get("delivery"):
                    actions.append({"type": "publish"})
        return actions

    def _mutate(
        self,
        operation: str,
        inputs: dict[str, Any],
        callback: Callable[[dict[str, Any], str], None],
    ) -> dict[str, Any]:
        with self._exclusive_lock():
            index, current = self._load_current()
            fingerprint = self._fingerprint(operation, inputs)
            if self._idempotent(current, fingerprint):
                return self._result(current, index)
            self._require_parent(inputs, index)
            state = deepcopy(current)
            self._upgrade_state(state)
            revision = next_revision(self.revisions_dir)
            state["parent_replacement_revision"] = current["replacement_revision"]
            state["replacement_revision"] = revision
            state["delivery"] = None
            self._mutation_parent_index = index
            try:
                callback(state, revision)
            finally:
                self._mutation_parent_index = None
            self._validate_state(state)
            return self._write_state(
                state, operation=operation, fingerprint=fingerprint, expected_index=index
            )

    def _upgrade_state(self, state: dict[str, Any]) -> None:
        """Upgrade an existing direct-generation 2.0 state without editing old revisions."""

        if state.get("schema_version") == STATE_SCHEMA_VERSION:
            return
        if state.get("schema_version") != LEGACY_DIRECT_STATE_VERSION:
            raise SceneReplacementError("Unsupported scene-replacement state version")
        snapshot = state["source_snapshot"]
        presentation = snapshot.get("presentation_map")
        if not presentation:
            delivery_path = verify_ref(
                snapshot["upstream"]["delivery_ref"],
                self.project_dir,
                label="Phase 1 delivery for presentation migration",
            )
            presentation = presentation_map_from_delivery(
                load_json(delivery_path, self.project_dir)
            )
        state["schema_version"] = STATE_SCHEMA_VERSION
        state["presentation_map"] = deepcopy(presentation)
        state.setdefault("scene_review_candidates", {})
        state.setdefault("global_review_candidate", None)
        state.setdefault("global_review_id", None)
        for request in state.get("edit_requests", {}).values():
            if request.get("status") == "prepared":
                request["dependency_status"] = "stale"
                request["status"] = "superseded"
        for review in state.get("reviews", []):
            if review.get("stage") == "group":
                review["dependency_status"] = "stale"
        for scene in (state.get("scene_plan") or {}).get("physical_scenes", []):
            for field in (
                "selection_receipt_id",
                "selection_options_sha256",
                "selected_direction_sha256",
                "scene_review_candidate_id",
                "group_review_id",
                "group_evidence_review_id",
                "contact_sheet",
            ):
                scene.pop(field, None)
        if state.get("scene_plan"):
            state["scene_plan"]["status"] = "awaiting_human"

    def initialize(self, inputs: dict[str, Any]) -> dict[str, Any]:
        with self._exclusive_lock():
            upstream = inputs.get("upstream_index_path")
            if not isinstance(upstream, str):
                raise SceneReplacementError("initialize requires upstream_index_path")
            upstream_path = Path(upstream).expanduser()
            if not upstream_path.is_absolute():
                upstream_path = self.project_dir / upstream_path
            canonical = self.project_dir / "artifacts" / "replication" / "index.json"
            if Path(os.path.abspath(upstream_path)) != canonical or upstream_path.is_symlink():
                raise SceneReplacementError("initialize requires canonical artifacts/replication/index.json")
            try:
                upstream_path = resolve_under(upstream_path, self.project_dir, must_exist=True)
            except (OSError, ValueError) as exc:
                raise SceneReplacementError("Phase 1 index must be inside the project workspace") from exc
            fingerprint_inputs = deepcopy(inputs)
            fingerprint_inputs["observed_upstream_index_sha256"] = sha256_file(upstream_path)
            fingerprint = self._fingerprint("initialize", fingerprint_inputs)
            if self.output_path.exists():
                index, current = self._load_current()
                if self._idempotent(current, fingerprint):
                    verify_source_snapshot(current["source_snapshot"], self.project_dir)
                    return self._result(current, index)
                self._require_parent(inputs, index)
                revision = next_revision(self.revisions_dir)
                snapshot = build_source_snapshot(
                    project_dir=self.project_dir,
                    upstream_index_path=upstream_path,
                    replacement_revision=revision,
                    expected_project_id=inputs["project_id"],
                    upstream_plan_revision=inputs.get("upstream_plan_revision"),
                    supplementary_anchor_confirmations=inputs.get("supplementary_anchor_confirmations"),
                    inherited_supplementary_anchor_confirmations=current["source_snapshot"].get(
                        "supplementary_anchor_confirmations", []
                    ),
                )
                state = deepcopy(current)
                self._upgrade_state(state)
                old = state["source_snapshot"]
                state.update(
                    replacement_revision=revision,
                    parent_replacement_revision=current["replacement_revision"],
                    source_snapshot=snapshot,
                    presentation_map=deepcopy(snapshot["presentation_map"]),
                    delivery=None,
                )
                self._apply_rebase(state, old, snapshot)
                self._validate_state(state)
                return self._write_state(
                    state,
                    operation="initialize",
                    fingerprint=fingerprint,
                    expected_index=index,
                )

            revision = next_revision(self.revisions_dir)
            snapshot = build_source_snapshot(
                project_dir=self.project_dir,
                upstream_index_path=upstream_path,
                replacement_revision=revision,
                expected_project_id=inputs["project_id"],
                upstream_plan_revision=inputs.get("upstream_plan_revision"),
                supplementary_anchor_confirmations=inputs.get("supplementary_anchor_confirmations"),
            )
            _validate_document("scene_replacement_source_snapshot", snapshot)
            state = {
                "schema_version": STATE_SCHEMA_VERSION,
                "tool": "replication_scene_replacement",
                "project_id": inputs["project_id"],
                "replacement_revision": revision,
                "parent_replacement_revision": None,
                "source_snapshot": snapshot,
                "presentation_map": deepcopy(snapshot["presentation_map"]),
                "scene_plan": None,
                "logical_tasks": {},
                "edit_requests": {},
                "results": {},
                "reviews": [],
                "adopted_results": {},
                "scene_review_candidates": {},
                "global_review_candidate": None,
                "global_review_id": None,
                "invalidations": [],
                "rebase_history": [],
                "operation_history": [],
                "delivery": None,
            }
            self._validate_state(state)
            return self._write_state(
                state,
                operation="initialize",
                fingerprint=fingerprint,
                expected_index=None,
            )

    def _apply_rebase(
        self, state: dict[str, Any], old: dict[str, Any], new: dict[str, Any]
    ) -> None:
        old_by_id = {item["anchor_id"]: item for item in old["anchors"]}
        new_by_id = {item["anchor_id"]: item for item in new["anchors"]}
        affected = sorted(
            anchor_id
            for anchor_id in set(old_by_id) | set(new_by_id)
            if old_by_id.get(anchor_id) != new_by_id.get(anchor_id)
        )
        if old["fingerprints"]["snapshot"] != new["fingerprints"]["snapshot"] and not affected:
            affected = sorted(new_by_id)
        state["rebase_history"].append(
            {
                "from_snapshot_id": old["snapshot_id"],
                "to_snapshot_id": new["snapshot_id"],
                "from_snapshot_fingerprint": old["fingerprints"]["snapshot"],
                "to_snapshot_fingerprint": new["fingerprints"]["snapshot"],
                "status": "invalidated" if affected else "reused",
                "affected_anchor_ids": affected,
            }
        )
        if affected:
            self._invalidate(state, affected, "upstream_source_changed", new["snapshot_id"])
            if state.get("scene_plan"):
                state["scene_plan"]["status"] = "needs_reconfirmation"
                for scene in state["scene_plan"]["physical_scenes"]:
                    if set(scene["anchor_ids"]) & set(affected):
                        scene["dependency_status"] = "needs_reconfirmation"

    def apply_scene_plan(self, inputs: dict[str, Any]) -> dict[str, Any]:
        submitted = _submission(inputs)

        def apply(state: dict[str, Any], revision: str) -> None:
            _validate_document("scene_replacement_v2_plan", submitted)
            plan = self._validate_scene_plan(submitted, state)
            plan_id = stable_id(
                "srp2", state["source_snapshot"]["fingerprints"]["anchor_set"], plan
            )
            previous = state.get("scene_plan")
            if previous:
                changed = sorted(
                    anchor
                    for anchor in self._anchor_ids(state)
                    if self._plan_binding(previous, anchor) != self._plan_binding(plan, anchor)
                )
                if changed:
                    self._invalidate(state, changed, "scene_plan_changed", plan_id)
            plan["scene_plan_id"] = plan_id
            plan["replacement_revision"] = revision
            plan["unassigned_anchor_ids"] = []
            for scene in plan["physical_scenes"]:
                scene["dependency_status"] = "current"
            plan["status"] = "awaiting_human"
            state["scene_plan"] = plan
            state["scene_review_candidates"] = {}
            state["global_review_candidate"] = None
            state["global_review_id"] = None

        return self._mutate("apply_scene_plan", inputs, apply)

    def _validate_scene_plan(
        self, submitted: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        value = deepcopy(submitted)
        if value.get("schema_version") != STATE_SCHEMA_VERSION:
            raise SceneReplacementError(
                f"Scene plan requires schema_version {STATE_SCHEMA_VERSION}"
            )
        snapshot_fingerprint = state["source_snapshot"]["fingerprints"]["snapshot"]
        if value.get("source_snapshot_fingerprint") != snapshot_fingerprint:
            raise SceneReplacementError("Scene plan source snapshot is stale")
        if value.get("audience_profile") != AUDIENCE_PROFILE:
            raise SceneReplacementError(
                "Scene plan must target North America, ages 30-50, ordinary consumer use, non-studio"
            )
        known = self._anchor_ids(state)
        assigned: list[str] = []
        scene_ids: set[str] = set()
        for scene in value["physical_scenes"]:
            scene_id = _nonempty(scene.get("scene_id"), "scene_id")
            if scene_id in scene_ids:
                raise SceneReplacementError("Physical scene IDs must be unique")
            scene_ids.add(scene_id)
            anchors = scene.get("anchor_ids")
            if not isinstance(anchors, list) or not anchors or len(anchors) != len(set(anchors)):
                raise SceneReplacementError(f"Physical scene {scene_id} requires unique anchor_ids")
            if not set(anchors) <= known:
                raise SceneReplacementError(f"Physical scene {scene_id} references an unknown anchor")
            assigned.extend(anchors)
            if set(scene.get("per_anchor_constraints", {})) != set(anchors):
                raise SceneReplacementError(f"Physical scene {scene_id} needs constraints for every anchor")
            directions = scene.get("directions", [])
            if len(directions) != 3:
                raise SceneReplacementError(f"Physical scene {scene_id} requires exactly three directions")
            direction_ids = [item.get("direction_id") for item in directions]
            setting_types = [item.get("consumer_context", {}).get("setting_type") for item in directions]
            if len(set(direction_ids)) != 3 or len(set(setting_types)) != 3:
                raise SceneReplacementError(
                    f"Physical scene {scene_id} requires three distinct direction IDs and setting types"
                )
            selected = scene.get("selected_direction_id")
            confirmation = scene.get("selection_confirmation")
            if selected is not None or confirmation is not None:
                raise SceneReplacementError(
                    "Scene proposals must be submitted unselected; record the human choice separately"
                )
            if scene.get("main_anchor_id") is not None and scene["main_anchor_id"] not in anchors:
                raise SceneReplacementError("Main anchor must belong to its physical scene")
            difficult = scene.get("difficult_view", {})
            if difficult.get("status") == "required" and difficult.get("anchor_id") not in anchors:
                raise SceneReplacementError("Required difficult view must identify an anchor in the scene")
            if not scene.get("main_anchor_id") or difficult.get("status") == "pending":
                raise SceneReplacementError(
                    "Every scene proposal requires a main anchor and final difficult-view decision"
                )
            if (
                difficult.get("status") == "required"
                and difficult.get("anchor_id") == scene.get("main_anchor_id")
            ):
                raise SceneReplacementError(
                    "A required difficult view must differ from the main anchor"
                )
        if len(assigned) != len(set(assigned)) or set(assigned) != known:
            raise SceneReplacementError("Every current anchor must belong to exactly one physical scene")
        return value

    def record_direction_selection(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Bind a human choice to the exact three directions shown for one scene."""

        submitted = _submission(inputs)

        def apply(state: dict[str, Any], revision: str) -> None:
            plan = state.get("scene_plan")
            if not plan:
                raise SceneReplacementError("A current three-direction scene plan is required")
            if submitted.get("scene_plan_id") != plan.get("scene_plan_id"):
                raise SceneReplacementError("Direction selection must bind the current scene plan")
            scene = self._scene_by_id(
                state, _nonempty(submitted.get("scene_id"), "scene_id")
            )
            directions = scene["directions"]
            direction_ids = [item["direction_id"] for item in directions]
            if submitted.get("presented_direction_ids") != direction_ids:
                raise SceneReplacementError(
                    "Direction selection must bind the three directions in their presented order"
                )
            options_sha256 = sha256_json(directions)
            if submitted.get("directions_sha256") != options_sha256:
                raise SceneReplacementError(
                    "Presented direction digest differs from the scene plan"
                )
            selected_id = _nonempty(
                submitted.get("selected_direction_id"), "selected_direction_id"
            )
            selected = next(
                (item for item in directions if item["direction_id"] == selected_id), None
            )
            if selected is None:
                raise SceneReplacementError(
                    "Selected direction is not one of the three proposals"
                )
            confirmation = deepcopy(submitted.get("human_confirmation"))
            if not _human(confirmation):
                raise SceneReplacementError(
                    "Direction selection requires human confirmation"
                )
            previous = scene.get("selected_direction_id")
            if previous and previous != selected_id:
                self._invalidate(
                    state,
                    scene["anchor_ids"],
                    "direction_selection_changed",
                    selected_id,
                )
            receipt_body = {
                "scene_plan_id": plan["scene_plan_id"],
                "scene_id": scene["scene_id"],
                "presented_direction_ids": direction_ids,
                "directions_sha256": options_sha256,
                "selected_direction_id": selected_id,
                "selected_direction_sha256": sha256_json(selected),
                "human_confirmation": confirmation,
            }
            receipt_id = stable_id("sselect2", receipt_body)
            if (
                previous == selected_id
                and scene.get("selection_receipt_id")
                and scene["selection_receipt_id"] != receipt_id
            ):
                old_candidate = state.get("scene_review_candidates", {}).get(
                    scene["scene_id"]
                )
                if old_candidate:
                    old_candidate["dependency_status"] = "stale"
                    old_candidate["stale_reason"] = "direction_selection_reconfirmed"
                for field in (
                    "scene_review_candidate_id",
                    "group_review_id",
                    "group_evidence_review_id",
                    "contact_sheet",
                ):
                    scene.pop(field, None)
                state["global_review_candidate"] = None
                state["global_review_id"] = None
            scene.update(
                selected_direction_id=selected_id,
                selection_confirmation=confirmation,
                selection_receipt_id=receipt_id,
                selection_options_sha256=options_sha256,
                selected_direction_sha256=sha256_json(selected),
            )
            for task in state["logical_tasks"].values():
                if (
                    task.get("dependency_status") == "current"
                    and task.get("anchor_id") in scene["anchor_ids"]
                    and task.get("approved_target", {}).get("id") == selected_id
                ):
                    task["approved_target"].update(
                        confirmation=deepcopy(confirmation),
                        selection_receipt_id=scene["selection_receipt_id"],
                        selected_direction_sha256=scene[
                            "selected_direction_sha256"
                        ],
                    )
            plan["replacement_revision"] = revision
            plan["status"] = (
                "ready"
                if all(
                    item.get("selection_receipt_id")
                    and item.get("selected_direction_id")
                    for item in plan["physical_scenes"]
                )
                else "awaiting_human"
            )

        return self._mutate("record_direction_selection", inputs, apply)

    @staticmethod
    def _anchor_ids(state: dict[str, Any]) -> set[str]:
        return {item["anchor_id"] for item in state["source_snapshot"]["anchors"]}

    @staticmethod
    def _plan_binding(plan: dict[str, Any], anchor_id: str) -> str | None:
        for scene in plan.get("physical_scenes", []):
            if anchor_id in scene.get("anchor_ids", []):
                return sha256_json(
                    {
                        "scene_id": scene["scene_id"],
                        "shared_constraints": scene["shared_constraints"],
                        "per_anchor_constraints": scene["per_anchor_constraints"][anchor_id],
                        "directions": scene["directions"],
                        "selected_direction_id": scene.get("selected_direction_id"),
                        "main_anchor_id": scene.get("main_anchor_id"),
                        "difficult_view": scene["difficult_view"],
                    }
                )
        return None

    def _anchor(self, state: dict[str, Any], anchor_id: str) -> dict[str, Any]:
        anchor = next(
            (item for item in state["source_snapshot"]["anchors"] if item["anchor_id"] == anchor_id),
            None,
        )
        if anchor is None:
            raise SceneReplacementError(f"Unknown anchor: {anchor_id}")
        return anchor

    def _scene_for_anchor(self, state: dict[str, Any], anchor_id: str) -> dict[str, Any]:
        plan = state.get("scene_plan")
        if not plan:
            raise SceneReplacementError("A scene plan is required")
        if plan.get("status") != "ready":
            raise SceneReplacementError(
                "All physical scenes require recorded human direction selections before generation"
            )
        scene = next(
            (item for item in plan["physical_scenes"] if anchor_id in item["anchor_ids"]),
            None,
        )
        if scene is None:
            raise SceneReplacementError(f"Anchor is not assigned to a physical scene: {anchor_id}")
        if scene.get("dependency_status") != "current":
            raise SceneReplacementError("Physical scene needs dependency reconfirmation")
        if (
            not scene.get("selected_direction_id")
            or not _human(scene.get("selection_confirmation"))
            or not scene.get("selection_receipt_id")
            or scene.get("selection_options_sha256") != sha256_json(scene["directions"])
        ):
            raise SceneReplacementError("Physical scene is awaiting human direction selection")
        return scene

    def _invalidate(
        self, state: dict[str, Any], anchor_ids: Iterable[str], reason: str, dependency: str
    ) -> None:
        affected = sorted(set(anchor_ids))
        if not affected:
            return
        invalidation_id = stable_id("sri2", reason, dependency, affected)
        entry = {
            "invalidation_id": invalidation_id,
            "reason": reason,
            "dependency": dependency,
            "anchor_ids": affected,
        }
        if entry not in state["invalidations"]:
            state["invalidations"].append(entry)
        for task in state["logical_tasks"].values():
            if task["anchor_id"] in affected:
                task["dependency_status"] = "stale"
        for request in state["edit_requests"].values():
            if request["anchor_id"] in affected:
                request["dependency_status"] = "stale"
                request["dependency_invalidation"] = {
                    "reason": reason,
                    "dependency": dependency,
                    "invalidation_id": invalidation_id,
                }
                if request["execution_status"] == "not_started":
                    request["status"] = "superseded"
        for result in state["results"].values():
            if result["anchor_id"] in affected:
                result["dependency_status"] = "stale"
                if result.get("status") == "adopted":
                    result["status"] = "rejected"
                result.setdefault("stale_reasons", []).append(
                    {
                        "reason": reason,
                        "dependency": dependency,
                        "invalidation_id": invalidation_id,
                    }
                )
        for review in state["reviews"]:
            if set(review.get("anchor_ids", [])) & set(affected):
                review["dependency_status"] = "stale"
        for anchor_id in affected:
            state["adopted_results"].pop(anchor_id, None)
        for scene_id, candidate in state.get("scene_review_candidates", {}).items():
            if set(candidate.get("anchor_ids", [])) & set(affected):
                candidate["dependency_status"] = "stale"
                candidate["stale_reason"] = reason
        candidate = state.get("global_review_candidate")
        if candidate:
            candidate["dependency_status"] = "stale"
            candidate["stale_reason"] = reason
        state["global_review_id"] = None
        for scene in (state.get("scene_plan") or {}).get("physical_scenes", []):
            if set(scene["anchor_ids"]) & set(affected):
                for field in (
                    "main_reference_result_id",
                    "sample_review_id",
                    "group_review_id",
                    "group_evidence_review_id",
                    "contact_sheet",
                    "scene_review_candidate_id",
                    "difficult_view_result_id",
                ):
                    scene.pop(field, None)

    def _active_task(
        self, state: dict[str, Any], anchor_id: str, *, required: bool = True
    ) -> dict[str, Any] | None:
        candidates = [
            item
            for item in state["logical_tasks"].values()
            if item["anchor_id"] == anchor_id
            and item["dependency_status"] == "current"
            and item["status"] != "superseded"
        ]
        if len(candidates) > 1:
            raise SceneReplacementError(f"Anchor has multiple active V2 tasks: {anchor_id}")
        if not candidates and required:
            raise SceneReplacementError(f"Anchor has no active V2 task: {anchor_id}")
        return candidates[0] if candidates else None

    def _latest_review(
        self, state: dict[str, Any], stage: str, scope_id: str
    ) -> dict[str, Any] | None:
        matches = [
            item
            for item in state["reviews"]
            if item["stage"] == stage
            and item["scope_id"] == scope_id
            and item["dependency_status"] == "current"
        ]
        return matches[-1] if matches else None

    @staticmethod
    def _selected_direction(scene: dict[str, Any]) -> dict[str, Any]:
        selected = next(
            (
                item
                for item in scene["directions"]
                if item["direction_id"] == scene.get("selected_direction_id")
            ),
            None,
        )
        if selected is None or sha256_json(selected) != scene.get(
            "selected_direction_sha256"
        ):
            raise SceneReplacementError("Selected direction binding is missing or stale")
        return selected

    def _build_generation_prompt(
        self,
        *,
        scene: dict[str, Any],
        anchor: dict[str, Any],
        mode: str,
        trigger_review: dict[str, Any] | None,
        creative_instructions: str | None,
    ) -> str:
        direction = self._selected_direction(scene)
        context = direction["consumer_context"]
        lines = [
            "Directly generate the complete replacement image from the supplied image references.",
            "Output exactly 720x1280 pixels in a 9:16 portrait frame.",
            "Do not use masks, compositing, source-pixel restoration, cropping, padding, or an advertising-studio look.",
            f"Selected scene direction ({direction['direction_id']}): {direction['description']}",
            "Visible scene changes: " + "; ".join(direction["visual_changes"]),
            "Functional preservation: " + direction["function_preservation"],
            "Cross-view strategy: " + direction["cross_view_strategy"],
            (
                "Audience and use context: ordinary North American consumers aged 30-50; "
                + context["ownership_or_access_rationale"]
                + "; "
                + context["usage_habit_rationale"]
            ),
            "Authenticity cues: " + "; ".join(context["authenticity_cues"]),
            "Anti-studio constraints: " + "; ".join(context["anti_studio_constraints"]),
            "Shared scene constraints: " + "; ".join(scene["shared_constraints"]),
            "This keyframe's constraints: "
            + "; ".join(scene["per_anchor_constraints"][anchor["anchor_id"]]),
            (
                "Preserve subject and product identity, geometry, color, labels, parts, pose, "
                "position, contact, occlusion, perspective, lighting compatibility, and interaction."
            ),
        ]
        if mode == "local_adjustment":
            lines.append(
                "Edit only the localized failed region in the generated-result reference; use the source frame only as an invariant reference."
            )
        elif mode == "source_regeneration":
            lines.append(
                "Regenerate from the source keyframe; do not use the failed generated image as a reference."
            )
        if trigger_review:
            actions = [item["rework_action"] for item in trigger_review.get("issues", [])]
            if actions:
                lines.append("Required corrections: " + "; ".join(actions))
        if creative_instructions:
            lines.append("Additional creative detail: " + creative_instructions.strip())
        lines.append(f"Prompt builder: {PROMPT_BUILDER_VERSION}.")
        prompt = "\n".join(lines)
        if not TARGET_SIZE_PROMPT_RE.search(prompt) or "9:16" not in prompt:
            raise SceneReplacementError(
                "Final prompt must explicitly require 720x1280 and 9:16"
            )
        return prompt

    def prepare_edit(self, inputs: dict[str, Any]) -> dict[str, Any]:
        submitted = _submission(inputs)

        def apply(state: dict[str, Any], revision: str) -> None:
            forbidden = {
                "approved_target",
                "prompt",
                "ordered_references",
                "ordered_image_references",
                "context_refs",
            } & set(submitted)
            if forbidden:
                raise SceneReplacementError(
                    "prepare_edit derives the selected target, final prompt, and references; "
                    f"remove caller-supplied fields: {', '.join(sorted(forbidden))}"
                )
            anchor_id = _nonempty(submitted.get("anchor_id"), "anchor_id")
            anchor = self._anchor(state, anchor_id)
            scene = self._scene_for_anchor(state, anchor_id)
            direction = self._selected_direction(scene)
            target = {
                "id": direction["direction_id"],
                "description": direction["description"],
                "confirmation": deepcopy(scene["selection_confirmation"]),
                "selection_receipt_id": scene["selection_receipt_id"],
                "selected_direction_sha256": scene["selected_direction_sha256"],
            }
            mode = submitted.get("generation_mode")
            if mode not in {"initial", "local_adjustment", "source_regeneration"}:
                raise SceneReplacementError("Invalid generation_mode")

            main_anchor_id = scene["main_anchor_id"]
            if anchor_id != main_anchor_id:
                sample = self._review_by_id(state, scene.get("sample_review_id"))
                if (
                    not sample
                    or sample.get("dependency_status") != "current"
                    or sample.get("status") != "pass"
                    or sample.get("reviewer", {}).get("kind") != "human"
                    or not _human(sample.get("human_confirmation"))
                ):
                    raise SceneReplacementError(
                        "Expansion requires explicit human approval of the main sample"
                    )
                difficult = scene["difficult_view"]
                difficult_anchor = difficult.get("anchor_id")
                if (
                    difficult.get("status") == "required"
                    and anchor_id != difficult_anchor
                    and difficult_anchor not in state["adopted_results"]
                ):
                    raise SceneReplacementError(
                        "The difficult view must be adopted before ordinary expansion"
                    )

            task = self._active_task(state, anchor_id, required=False)
            lineage = [
                state["source_snapshot"]["snapshot_id"],
                state["scene_plan"]["scene_plan_id"],
                anchor_id,
            ]
            if task is None:
                if mode != "initial":
                    raise SceneReplacementError("A fresh V2 task must start with generation_mode initial")
                task_id = stable_id("srt2", lineage, target)
                task = {
                    "schema_version": STATE_SCHEMA_VERSION,
                    "task_id": task_id,
                    "anchor_id": anchor_id,
                    "anchor_fingerprint": anchor_fingerprint(anchor),
                    "anchor_lineage": lineage,
                    "approved_target": target,
                    "status": "active",
                    "dependency_status": "current",
                    "request_ids": [],
                    "current_request_id": None,
                    "counted_dispatches": 0,
                    "latest_result_id": None,
                    "related_task_ids": [],
                }
                state["logical_tasks"][task_id] = task
            elif task["approved_target"].get("selection_receipt_id") != scene.get(
                "selection_receipt_id"
            ):
                raise SceneReplacementError(
                    "Logical task target differs from the current direction selection"
                )
            if task["counted_dispatches"] >= 3:
                task["status"] = "exhausted"
                raise SceneReplacementError("Logical task has exhausted its three image_gen calls")
            current_request = (
                state["edit_requests"].get(task.get("current_request_id"))
                if task.get("current_request_id")
                else None
            )
            if current_request:
                if current_request["execution_status"] == "unknown":
                    raise SceneReplacementError("Unknown image_gen execution must be reconciled first")
                if current_request["status"] == "prepared":
                    current_request["status"] = "superseded"

            latest_result: dict[str, Any] | None = None
            trigger_review: dict[str, Any] | None = None
            trigger_execution_request_id: str | None = None
            if mode == "initial":
                if task["counted_dispatches"] or task.get("latest_result_id"):
                    raise SceneReplacementError("Only the first V2 attempt may use generation_mode initial")
                if submitted.get("trigger_review_id") or submitted.get("base_result_id"):
                    raise SceneReplacementError("Initial generation cannot bind a previous review or result")
            else:
                latest_id = task.get("latest_result_id")
                latest_result = state["results"].get(latest_id) if latest_id else None
                if (
                    mode == "source_regeneration"
                    and current_request
                    and current_request.get("execution_status") == "confirmed_failed"
                    and submitted.get("trigger_execution_request_id")
                    == current_request["request_id"]
                ):
                    trigger_execution_request_id = current_request["request_id"]
                    if submitted.get("trigger_review_id") or submitted.get("base_result_id"):
                        raise SceneReplacementError(
                            "Execution retry cannot bind a result review or generated base image"
                        )
                elif latest_result and latest_result.get("status") == "rejected":
                    trigger_review = self._latest_review(
                        state, "generated_result", latest_result["result_id"]
                    )
                    if (
                        not trigger_review
                        or trigger_review.get("status") != "fail"
                        or trigger_review["review_id"]
                        != submitted.get("trigger_review_id")
                    ):
                        raise SceneReplacementError(
                            "Rework must bind the latest failed self-review"
                        )
                    strategy = trigger_review.get("rework_decision", {}).get("strategy")
                    if mode != strategy:
                        raise SceneReplacementError(
                            "generation_mode must match the self-review rework strategy"
                        )
                    if mode == "local_adjustment":
                        if submitted.get("base_result_id") != latest_result["result_id"]:
                            raise SceneReplacementError(
                                "Local adjustment must edit the reviewed generated result"
                            )
                    elif submitted.get("base_result_id") is not None:
                        raise SceneReplacementError(
                            "Source regeneration must not include a base generated result"
                        )
                else:
                    raise SceneReplacementError(
                        "Rework requires a rejected result review or a confirmed failed execution"
                    )

            parent_state_ref = deepcopy(self._mutation_parent_index["manifest_index"]["state.json"])
            image_references: list[dict[str, Any]] = []
            if mode == "local_adjustment":
                assert latest_result is not None
                editable = latest_result["generated_image"]
                image_references.extend(
                    [
                        {
                            "role": "generated_result",
                            "usage": "edit_target",
                            "reference": deepcopy(editable),
                            "anchor_id": anchor_id,
                            "result_id": latest_result["result_id"],
                        },
                        {
                            "role": "source_frame",
                            "usage": "invariant_reference",
                            "reference": deepcopy(anchor["image"]),
                            "anchor_id": anchor_id,
                        },
                    ]
                )
            else:
                image_references.append(
                    {
                        "role": "source_frame",
                        "usage": "edit_target",
                        "reference": deepcopy(anchor["image"]),
                        "anchor_id": anchor_id,
                    }
                )
            if anchor_id != scene.get("main_anchor_id"):
                main_result_id = scene.get("main_reference_result_id")
                main_result = state["results"].get(main_result_id)
                if (
                    not main_result
                    or main_result.get("status") != "adopted"
                    or main_result.get("anchor_id") != scene["main_anchor_id"]
                ):
                    raise SceneReplacementError("Expansion generation requires the adopted main reference")
                image_references.append(
                    {
                        "role": "main_reference",
                        "usage": "continuity_reference",
                        "reference": _reference(main_result["normalization"]["image"]),
                        "anchor_id": main_result["anchor_id"],
                        "result_id": main_result_id,
                    }
                )
                difficult_result_id = scene.get("difficult_view_result_id")
                if difficult_result_id and anchor_id != scene["difficult_view"].get("anchor_id"):
                    difficult_result = state["results"].get(difficult_result_id)
                    if (
                        difficult_result
                        and difficult_result.get("status") == "adopted"
                        and difficult_result.get("anchor_id")
                        == scene["difficult_view"].get("anchor_id")
                    ):
                        image_references.append(
                            {
                                "role": "supplementary_reference",
                                "usage": "continuity_reference",
                                "reference": _reference(
                                    difficult_result["normalization"]["image"]
                                ),
                                "anchor_id": difficult_result["anchor_id"],
                                "result_id": difficult_result_id,
                            }
                        )

            creative = submitted.get("creative_instructions")
            if creative is not None:
                creative = _nonempty(creative, "creative_instructions")
            prompt = self._build_generation_prompt(
                scene=scene,
                anchor=anchor,
                mode=mode,
                trigger_review=trigger_review,
                creative_instructions=creative,
            )

            regions = deepcopy(submitted.get("inspection_regions", []))
            names: set[str] = set()
            for region in regions:
                name = _nonempty(region.get("name"), "inspection region name")
                if name in names:
                    raise SceneReplacementError("Inspection region names must be unique")
                names.add(name)
                left, top, right, bottom = region["bbox"]
                if not (
                    0 <= left < right <= anchor["image"]["width"]
                    and 0 <= top < bottom <= anchor["image"]["height"]
                ):
                    raise SceneReplacementError("Inspection region must stay inside the source canvas")

            attempt = task["counted_dispatches"] + 1
            preparation_ordinal = 1 + max(
                [item["preparation_ordinal"] for item in state["edit_requests"].values()],
                default=0,
            )
            request_body = {
                "schema_version": STATE_SCHEMA_VERSION,
                "logical_task_id": task["task_id"],
                "anchor_id": anchor_id,
                "anchor_fingerprint": anchor_fingerprint(anchor),
                "physical_scene_id": scene["scene_id"],
                "scene_plan_id": state["scene_plan"]["scene_plan_id"],
                "selected_direction_id": scene["selected_direction_id"],
                "selection_receipt_id": scene["selection_receipt_id"],
                "selected_direction_sha256": scene["selected_direction_sha256"],
                "approved_target": target,
                "source_snapshot_fingerprint": state["source_snapshot"]["fingerprints"]["snapshot"],
                "upstream_plan_revision": state["source_snapshot"]["upstream"]["plan_revision"],
                "source_image": deepcopy(anchor["image"]),
                "generation_mode": mode,
                "prompt": prompt,
                "prompt_sha256": _prompt_hash(prompt),
                "prompt_builder_version": PROMPT_BUILDER_VERSION,
                "ordered_image_references": image_references,
                "context_refs": [
                    {
                        "role": "scene_design",
                        "usage": "design_record",
                        "reference": parent_state_ref,
                    }
                ],
                "inspection_regions": regions,
                "requested_capability": "imagegen2",
                "required_actual_tool": "image_gen",
                "attempt_number": attempt,
                "preparation_ordinal": preparation_ordinal,
                "status": "prepared",
                "execution_status": "not_started",
                "dependency_status": "current",
                "prepared_in_revision": revision,
            }
            if trigger_review:
                request_body["trigger_review_id"] = trigger_review["review_id"]
            if trigger_execution_request_id:
                request_body["trigger_execution_request_id"] = (
                    trigger_execution_request_id
                )
            if mode == "local_adjustment":
                request_body["base_result_id"] = submitted["base_result_id"]
            request_fingerprint = sha256_json(request_body)
            request_id = stable_id("sreq2", request_fingerprint)
            request = {
                **request_body,
                "request_id": request_id,
                "request_fingerprint": request_fingerprint,
            }
            _validate_document("scene_replacement_v2_edit_request", request)
            state["edit_requests"][request_id] = request
            task["request_ids"].append(request_id)
            task["current_request_id"] = request_id

        return self._mutate("prepare_edit", inputs, apply)

    def mark_dispatched(self, inputs: dict[str, Any]) -> dict[str, Any]:
        submitted = _submission(inputs)

        def apply(state: dict[str, Any], revision: str) -> None:
            request_id = _nonempty(submitted.get("request_id"), "request_id")
            request = state["edit_requests"].get(request_id)
            if not request or request.get("dependency_status") != "current":
                raise SceneReplacementError("Dispatch request is missing or stale")
            if request.get("status") != "prepared" or request.get("execution_status") != "not_started":
                raise SceneReplacementError("Only a prepared, not-started request can be dispatched")
            task = state["logical_tasks"][request["logical_task_id"]]
            if task.get("current_request_id") != request_id:
                raise SceneReplacementError("Only the logical task's current request can be dispatched")
            review = self._latest_review(state, "pre_generation", request_id)
            if not review or review.get("status") != "pass":
                raise SceneReplacementError("Passing pre-generation self-review is required")
            _require_rules(review, PRE_GENERATION_RULES, "pre-generation")
            if task["counted_dispatches"] >= 3:
                raise SceneReplacementError("Logical task has exhausted its three image_gen calls")
            request.update(
                status="dispatched",
                execution_status="unknown",
                dispatched_in_revision=revision,
                counts_toward_limit=True,
            )
            task["counted_dispatches"] += 1
            if task["counted_dispatches"] >= 3:
                task["status"] = "exhausted"

        return self._mutate("mark_dispatched", inputs, apply)

    def import_result(self, inputs: dict[str, Any]) -> dict[str, Any]:
        submitted = _submission(inputs)

        def apply(state: dict[str, Any], revision: str) -> None:
            request_id = _nonempty(submitted.get("request_id"), "request_id")
            request = state["edit_requests"].get(request_id)
            if not request:
                raise SceneReplacementError("Unknown generation request")
            if request.get("status") != "dispatched" or request.get("execution_status") not in {
                "unknown",
                "abandoned_unknown",
            }:
                raise SceneReplacementError("Result can only resolve a dispatched generation request")
            execution = deepcopy(submitted.get("execution"))
            if not isinstance(execution, dict) or execution.get("requested_capability") != "imagegen2" or execution.get("actual_tool") != "image_gen":
                raise SceneReplacementError("Execution must prove use of built-in image_gen")
            if submitted.get("success") is not True:
                request.update(
                    status="failed",
                    execution_status="confirmed_failed",
                    execution=execution,
                    error=_nonempty(submitted.get("error"), "generation error"),
                    resolved_in_revision=revision,
                )
                task = state["logical_tasks"][request["logical_task_id"]]
                if task["counted_dispatches"] < 3:
                    task["status"] = "active"
                return
            result_path = _nonempty(submitted.get("result_path"), "result_path")
            supplied_result = Path(result_path).expanduser()
            if not supplied_result.is_absolute():
                supplied_result = self.project_dir / supplied_result
            try:
                supplied_result = resolve_under(
                    supplied_result, self.project_dir, must_exist=True
                )
            except (OSError, ValueError) as exc:
                raise SceneReplacementError(
                    "image_gen result must be saved as a project-local file before import"
                ) from exc
            if supplied_result.is_symlink() or not supplied_result.is_file():
                raise SceneReplacementError("image_gen result must be a regular project-local file")
            result_seed = sha256_json(
                {
                    "request_id": request_id,
                    "path": self.relative(supplied_result),
                    "sha256": sha256_file(supplied_result),
                    "execution": execution,
                }
            )
            result_id = stable_id("sres2", result_seed)
            raw_suffix = Path(result_path).suffix.lower() or ".bin"
            result_dir = self.asset_root / "results" / result_id
            try:
                generated = import_image_asset(
                    supplied_result,
                    self.project_dir,
                    result_dir / "generated.png",
                    "direct_generated_result",
                    raw_destination_path=result_dir / f"raw{raw_suffix}",
                )
            except ImageAssetError as exc:
                request.update(
                    status="failed",
                    execution_status="confirmed_failed",
                    execution=execution,
                    error=f"result_import_failed:{exc}",
                    resolved_in_revision=revision,
                )
                task = state["logical_tasks"][request["logical_task_id"]]
                if task["counted_dispatches"] < 3:
                    task["status"] = "active"
                return
            anchor = self._anchor(state, request["anchor_id"])
            try:
                normalization = normalize_direct_result(
                    generated["path"],
                    self.project_dir,
                    result_dir / "direct-result.png",
                )
            except ImageAssetError as exc:
                normalization = {"status": "blocked", "reason": str(exc)}

            response_path = self.artifact_root / "responses" / f"{result_id}.json"
            atomic_write_json(
                response_path,
                {
                    "schema_version": STATE_SCHEMA_VERSION,
                    "request_id": request_id,
                    "request_fingerprint": request["request_fingerprint"],
                    "success": True,
                    "execution": execution,
                    "generated_image": _reference(generated),
                },
            )
            response_ref = {"path": self.relative(response_path), "sha256": sha256_file(response_path)}
            reviewed_image = (
                normalization["image"]
                if normalization["status"] == "ready"
                else generated
            )
            comparison = _reference(
                generate_direct_comparison(
                    anchor["image"]["path"],
                    reviewed_image["path"],
                    result_dir / "source-vs-direct-result.png",
                    self.project_dir,
                )
            )
            crops = [
                _reference(item)
                for item in generate_direct_review_crops(
                    anchor["image"]["path"],
                    reviewed_image["path"],
                    request["inspection_regions"],
                    result_dir / "review-crops",
                    self.project_dir,
                )
            ]
            late = request.get("execution_status") == "abandoned_unknown" or request.get("dependency_status") != "current"
            result = {
                "schema_version": STATE_SCHEMA_VERSION,
                "result_id": result_id,
                "request_id": request_id,
                "request_fingerprint": request["request_fingerprint"],
                "logical_task_id": request["logical_task_id"],
                "anchor_id": request["anchor_id"],
                "source_anchor_fingerprint": request["anchor_fingerprint"],
                "scene_plan_id": request["scene_plan_id"],
                "generation_mode": request["generation_mode"],
                "status": "late" if late else "imported",
                "dependency_status": "stale" if late else "current",
                "generated_image": generated,
                "normalization": normalization,
                "raw_response": response_ref,
                "execution": execution,
                "comparison": comparison,
                "review_crops": crops,
                "imported_in_revision": revision,
                "late_arrival": late,
            }
            if request.get("base_result_id"):
                result["base_result_id"] = request["base_result_id"]
            if request.get("trigger_review_id"):
                result["trigger_review_id"] = request["trigger_review_id"]
            state["results"][result_id] = result
            request.setdefault("late_result_ids", [])
            if late:
                request["late_result_ids"].append(result_id)
                return
            request.update(
                status="succeeded",
                execution_status="succeeded",
                execution=execution,
                result_id=result_id,
                resolved_in_revision=revision,
            )
            task = state["logical_tasks"][request["logical_task_id"]]
            task["latest_result_id"] = result_id

        return self._mutate("import_result", inputs, apply)

    def reconcile_execution(self, inputs: dict[str, Any]) -> dict[str, Any]:
        submitted = _submission(inputs)

        def apply(state: dict[str, Any], revision: str) -> None:
            request_id = _nonempty(submitted.get("request_id"), "request_id")
            request = state["edit_requests"].get(request_id)
            if not request or request.get("execution_status") != "unknown":
                raise SceneReplacementError("Only an unknown dispatched request can be reconciled")
            resolution = submitted.get("resolution")
            if resolution not in {
                "success",
                "confirmed_failed",
                "confirmed_not_run",
                "abandoned_unknown",
            }:
                raise SceneReplacementError("Invalid reconciliation resolution")
            task = state["logical_tasks"][request["logical_task_id"]]
            if resolution == "success":
                result_id = submitted.get("result_id")
                if request.get("result_id") != result_id or result_id not in state["results"]:
                    raise SceneReplacementError("Success reconciliation requires an already imported result")
                request["status"] = "succeeded"
                request["execution_status"] = "succeeded"
            elif resolution == "confirmed_not_run":
                request["status"] = "failed"
                request["execution_status"] = "confirmed_not_run"
                if request.get("counts_toward_limit"):
                    task["counted_dispatches"] -= 1
                    request["counts_toward_limit"] = False
                task["status"] = "active"
            elif resolution == "confirmed_failed":
                request["status"] = "failed"
                request["execution_status"] = "confirmed_failed"
                if task["counted_dispatches"] < 3:
                    task["status"] = "active"
            else:
                request["execution_status"] = "abandoned_unknown"
            reconciliation = {
                "request_id": request_id,
                "resolution": resolution,
                "replacement_revision": revision,
            }
            for field in ("evidence", "reason", "result_id"):
                if submitted.get(field) is not None:
                    reconciliation[field] = submitted[field]
            request["reconciliation"] = reconciliation
            request["resolved_in_revision"] = revision

        return self._mutate("reconcile_execution", inputs, apply)

    @staticmethod
    def _presentation_by_anchor(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
        rows = state.get("presentation_map") or state["source_snapshot"].get(
            "presentation_map", []
        )
        mapping = {item["anchor_id"]: item for item in rows}
        if set(mapping) != {
            item["anchor_id"] for item in state["source_snapshot"]["anchors"]
        }:
            raise SceneReplacementError(
                "Readable presentation map differs from the current anchor set"
            )
        return mapping

    def _comparison_items(
        self, state: dict[str, Any], anchor_ids: Iterable[str]
    ) -> list[dict[str, Any]]:
        presentation = self._presentation_by_anchor(state)
        rows: list[dict[str, Any]] = []
        for anchor_id in anchor_ids:
            result_id = state["adopted_results"].get(anchor_id)
            result = state["results"].get(result_id)
            if (
                not result
                or result.get("status") != "adopted"
                or result.get("dependency_status") != "current"
                or result.get("normalization", {}).get("status") != "ready"
            ):
                raise SceneReplacementError(
                    f"Comparison requires a current normalized adopted result: {anchor_id}"
                )
            anchor = self._anchor(state, anchor_id)
            scene = self._scene_for_anchor(state, anchor_id)
            row = presentation[anchor_id]
            rows.append(
                {
                    **deepcopy(row),
                    "scene_id": scene["scene_id"],
                    "direction_id": scene["selected_direction_id"],
                    "source_path": anchor["image"]["path"],
                    "source_sha256": anchor["image"]["sha256"],
                    "result_id": result_id,
                    "result_path": result["normalization"]["image"]["path"],
                    "result_sha256": result["normalization"]["image"]["sha256"],
                }
            )
        rows.sort(
            key=lambda item: (
                item["clip_sequence"],
                item["keyframe_sequence"],
                item["anchor_id"],
            )
        )
        return rows

    def prepare_scene_review(self, inputs: dict[str, Any]) -> dict[str, Any]:
        submitted = _submission(inputs)

        def apply(state: dict[str, Any], revision: str) -> None:
            scene = self._scene_by_id(
                state, _nonempty(submitted.get("scene_id"), "scene_id")
            )
            sample = self._review_by_id(state, scene.get("sample_review_id"))
            if (
                not sample
                or sample.get("status") != "pass"
                or sample.get("dependency_status") != "current"
                or not _human(sample.get("human_confirmation"))
            ):
                raise SceneReplacementError(
                    "Scene review evidence requires an approved main sample"
                )
            rows = self._comparison_items(state, scene["anchor_ids"])
            dependency = {
                "scene_plan_id": state["scene_plan"]["scene_plan_id"],
                "scene_id": scene["scene_id"],
                "selection_receipt_id": scene["selection_receipt_id"],
                "items": [
                    {
                        "anchor_id": item["anchor_id"],
                        "display_id": item["display_id"],
                        "source_sha256": item["source_sha256"],
                        "result_id": item["result_id"],
                        "result_sha256": item["result_sha256"],
                    }
                    for item in rows
                ],
                "renderer_version": MULTIFRAME_RENDERER_VERSION,
            }
            fingerprint = sha256_json(dependency)
            candidate_id = stable_id("sscene2", dependency)
            output_dir = (
                self.asset_root
                / "review-candidates"
                / "scenes"
                / scene["scene_id"]
                / fingerprint[:16]
            )
            rendered = generate_multiframe_comparison(
                rows,
                output_dir / "source-vs-generated.png",
                self.project_dir,
                title=f"{state['project_id']} - {scene['scene_id']} REVIEW",
            )
            bindings = rendered.pop("bindings")
            binding_path = (
                self.artifact_root
                / "review-candidates"
                / "scenes"
                / scene["scene_id"]
                / fingerprint[:16]
                / "bindings.json"
            )
            atomic_write_json(
                binding_path,
                {
                    "schema_version": STATE_SCHEMA_VERSION,
                    "candidate_id": candidate_id,
                    "dependency_fingerprint": fingerprint,
                    "ordered_display_ids": [item["display_id"] for item in rows],
                    "bindings": bindings,
                },
            )
            previous = state["scene_review_candidates"].get(scene["scene_id"])
            if previous and previous.get("candidate_id") != candidate_id:
                previous["dependency_status"] = "stale"
            candidate = {
                "schema_version": STATE_SCHEMA_VERSION,
                "candidate_id": candidate_id,
                "scene_id": scene["scene_id"],
                "anchor_ids": [item["anchor_id"] for item in rows],
                "ordered_display_ids": [item["display_id"] for item in rows],
                "dependency_fingerprint": fingerprint,
                "dependency_status": "current",
                "comparison": rendered,
                "binding_manifest": {
                    "path": self.relative(binding_path),
                    "sha256": sha256_file(binding_path),
                },
                "prepared_in_revision": revision,
            }
            state["scene_review_candidates"][scene["scene_id"]] = candidate
            scene["scene_review_candidate_id"] = candidate_id
            scene["contact_sheet"] = deepcopy(rendered)
            scene.pop("group_review_id", None)
            state["global_review_candidate"] = None
            state["global_review_id"] = None

        return self._mutate("prepare_scene_review", inputs, apply)

    def prepare_global_review(self, inputs: dict[str, Any]) -> dict[str, Any]:
        _ = deepcopy(inputs.get("submission") or {})

        def apply(state: dict[str, Any], revision: str) -> None:
            plan = state.get("scene_plan")
            if not plan or plan.get("status") != "ready":
                raise SceneReplacementError("A ready scene plan is required")
            group_review_ids: list[str] = []
            for scene in plan["physical_scenes"]:
                candidate = state["scene_review_candidates"].get(scene["scene_id"])
                group = self._review_by_id(state, scene.get("group_review_id"))
                if (
                    not candidate
                    or candidate.get("dependency_status") != "current"
                    or scene.get("scene_review_candidate_id")
                    != candidate.get("candidate_id")
                    or not group
                    or group.get("status") != "pass"
                    or group.get("dependency_status") != "current"
                    or not _human(group.get("human_confirmation"))
                ):
                    raise SceneReplacementError(
                        f"Scene {scene['scene_id']} requires current paired evidence and human approval"
                    )
                group_review_ids.append(group["review_id"])
            rows = self._comparison_items(state, self._anchor_ids(state))
            dependency = {
                "scene_plan_id": plan["scene_plan_id"],
                "group_review_ids": group_review_ids,
                "items": [
                    {
                        "anchor_id": item["anchor_id"],
                        "display_id": item["display_id"],
                        "source_sha256": item["source_sha256"],
                        "result_id": item["result_id"],
                        "result_sha256": item["result_sha256"],
                    }
                    for item in rows
                ],
                "renderer_version": MULTIFRAME_RENDERER_VERSION,
                "naming_version": "sequence-v1",
            }
            fingerprint = sha256_json(dependency)
            candidate_id = stable_id("sglobal2", dependency)
            output_root = (
                self.asset_root / "delivery" / revision / fingerprint[:16]
            )
            aliases: list[dict[str, Any]] = []
            for row in rows:
                readable = copy_image_asset(
                    row["result_path"],
                    self.project_dir,
                    output_root / f"{row['display_id']}.png",
                )
                aliases.append(
                    {
                        **{
                            key: deepcopy(row[key])
                            for key in (
                                "anchor_id",
                                "clip_id",
                                "clip_sequence",
                                "clip_display_id",
                                "keyframe_sequence",
                                "display_id",
                                "timeline_sequence",
                                "scene_id",
                                "direction_id",
                                "result_id",
                            )
                        },
                        "source_image": {
                            "path": row["source_path"],
                            "sha256": row["source_sha256"],
                        },
                        "image": readable,
                    }
                )
            comparison_rows = [
                {
                    **row,
                    "result_path": alias["image"]["path"],
                    "result_sha256": alias["image"]["sha256"],
                }
                for row, alias in zip(rows, aliases)
            ]
            rendered = generate_multiframe_comparison(
                comparison_rows,
                output_root / "comparisons" / "all-anchors-comparison.png",
                self.project_dir,
                title=f"{state['project_id']} - FINAL MULTI-FRAME COMPARISON",
            )
            bindings = rendered.pop("bindings")
            candidate_dir = (
                self.artifact_root
                / "review-candidates"
                / "global"
                / revision
                / fingerprint[:16]
            )
            assets_path = candidate_dir / "accepted-assets.json"
            binding_path = candidate_dir / "comparison-bindings.json"
            atomic_write_json(
                assets_path,
                {
                    "schema_version": STATE_SCHEMA_VERSION,
                    "candidate_id": candidate_id,
                    "dependency_fingerprint": fingerprint,
                    "naming_version": "sequence-v1",
                    "assets": aliases,
                },
            )
            atomic_write_json(
                binding_path,
                {
                    "schema_version": STATE_SCHEMA_VERSION,
                    "candidate_id": candidate_id,
                    "dependency_fingerprint": fingerprint,
                    "ordered_display_ids": [item["display_id"] for item in rows],
                    "bindings": bindings,
                },
            )
            previous = state.get("global_review_candidate")
            if previous and previous.get("candidate_id") != candidate_id:
                previous["dependency_status"] = "stale"
            state["global_review_candidate"] = {
                "schema_version": STATE_SCHEMA_VERSION,
                "candidate_id": candidate_id,
                "dependency_fingerprint": fingerprint,
                "dependency_status": "current",
                "accepted_assets_manifest": {
                    "path": self.relative(assets_path),
                    "sha256": sha256_file(assets_path),
                },
                "output_root": self.relative(output_root),
                "global_comparison": rendered,
                "binding_manifest": {
                    "path": self.relative(binding_path),
                    "sha256": sha256_file(binding_path),
                },
                "item_count": len(rows),
                "ordered_display_ids": [item["display_id"] for item in rows],
                "renderer_version": MULTIFRAME_RENDERER_VERSION,
                "prepared_in_revision": revision,
            }
            state["global_review_id"] = None

        return self._mutate("prepare_global_review", inputs, apply)

    def apply_review(self, inputs: dict[str, Any]) -> dict[str, Any]:
        submitted = _submission(inputs)

        def apply(state: dict[str, Any], revision: str) -> None:
            _validate_document("scene_replacement_v2_review_submission", submitted)
            for reference in submitted["evidence_refs"]:
                verify_ref(reference, self.project_dir, label="review evidence")
            for rule in submitted["rules"]:
                for reference in rule["evidence_refs"]:
                    verify_ref(reference, self.project_dir, label=f"review rule {rule['rule_id']}")
            for issue in submitted["issues"]:
                for reference in issue["evidence_refs"]:
                    verify_ref(reference, self.project_dir, label=f"review issue {issue['rule_id']}")
            stage = submitted["stage"]
            if stage == "pre_generation":
                anchors = self._apply_pre_review(state, submitted)
            elif stage == "generated_result":
                anchors = self._apply_result_review(state, submitted)
            elif stage == "sample":
                anchors = self._apply_sample_review(state, submitted)
            elif stage == "group":
                anchors = self._apply_group_review(state, submitted)
            else:
                anchors = self._apply_global_review(state, submitted)
            review_id = stable_id("srv2", revision, submitted)
            review = {
                "schema_version": STATE_SCHEMA_VERSION,
                "review_id": review_id,
                **deepcopy(submitted),
                "anchor_ids": sorted(set(anchors)),
                "dependency_status": "current",
                "replacement_revision": revision,
            }
            state["reviews"].append(review)
            if stage == "generated_result":
                result = state["results"][submitted["result_id"]]
                if submitted["status"] == "pass":
                    self._clear_scene_acceptance(state, result["anchor_id"])
                    result["status"] = "adopted"
                    result["adoption_review_id"] = review_id
                    state["adopted_results"][result["anchor_id"]] = result["result_id"]
                    scene = self._scene_for_anchor(state, result["anchor_id"])
                    if result["anchor_id"] == scene.get("main_anchor_id"):
                        scene["main_reference_result_id"] = result["result_id"]
                    if result["anchor_id"] == scene.get("difficult_view", {}).get(
                        "anchor_id"
                    ):
                        scene["difficult_view_result_id"] = result["result_id"]
                else:
                    result["status"] = "rejected"
                    state["adopted_results"].pop(result["anchor_id"], None)
                    self._clear_scene_acceptance(state, result["anchor_id"])
            elif stage == "sample" and submitted["status"] == "pass":
                scene = self._scene_by_id(state, submitted["scene_id"])
                scene["sample_review_id"] = review_id
            elif stage == "sample":
                scene = self._scene_by_id(state, submitted["scene_id"])
                for field in (
                    "sample_review_id",
                    "scene_review_candidate_id",
                    "group_review_id",
                    "group_evidence_review_id",
                    "contact_sheet",
                ):
                    scene.pop(field, None)
                state["global_review_candidate"] = None
                state["global_review_id"] = None
            elif stage == "group":
                scene = self._scene_by_id(state, submitted["scene_id"])
                if submitted["status"] == "pass":
                    scene["group_review_id"] = review_id
                else:
                    scene.pop("group_review_id", None)
                state["global_review_candidate"] = None
                state["global_review_id"] = None
            elif stage == "global" and submitted["status"] == "pass":
                state["global_review_id"] = review_id

        return self._mutate("apply_review", inputs, apply)

    def _apply_pre_review(self, state: dict[str, Any], review: dict[str, Any]) -> list[str]:
        request = state["edit_requests"].get(review["scope_id"])
        if not request or request.get("status") != "prepared" or request.get("dependency_status") != "current":
            raise SceneReplacementError("Pre-generation review must bind a current prepared request")
        if review.get("anchor_id") != request["anchor_id"]:
            raise SceneReplacementError("Pre-generation review anchor binding differs")
        if review.get("reviewer", {}).get("kind") != "ai_coding_assistant":
            raise SceneReplacementError(
                "Pre-generation review must be performed by the AI coding assistant"
            )
        if review["status"] == "pass":
            if review.get("issues") or review.get("rework_decision"):
                raise SceneReplacementError(
                    "Passing pre-generation review cannot contain issues or a rework decision"
                )
            _require_rules(review, PRE_GENERATION_RULES, "pre-generation")
        return [request["anchor_id"]]

    def _result_and_digest(
        self, state: dict[str, Any], review: dict[str, Any]
    ) -> tuple[dict[str, Any], str]:
        result = state["results"].get(review.get("result_id"))
        if not result or result.get("dependency_status") != "current":
            raise SceneReplacementError("Review result is missing or stale")
        digest = (
            result["normalization"]["image"]["sha256"]
            if result["normalization"]["status"] == "ready"
            else result["generated_image"]["sha256"]
        )
        if review.get("result_sha256") != digest:
            raise SceneReplacementError("Review result hash does not match the imported direct image")
        return result, digest

    def _apply_result_review(self, state: dict[str, Any], review: dict[str, Any]) -> list[str]:
        result, _ = self._result_and_digest(state, review)
        if review["scope_id"] != result["result_id"] or review.get("anchor_id") != result["anchor_id"]:
            raise SceneReplacementError("Generated-result self-review binding differs")
        _require_rule_coverage(review, GENERATED_RESULT_RULES, "generated-result")
        if review.get("reviewer", {}).get("kind") != "ai_coding_assistant":
            raise SceneReplacementError(
                "Generated-result self-review must be performed by the AI coding assistant"
            )
        if review["status"] == "pass":
            if result["normalization"]["status"] != "ready":
                raise SceneReplacementError("A normalization-blocked result cannot pass self-review")
            if review.get("adopt_result_id") != result["result_id"]:
                raise SceneReplacementError("Passing self-review must explicitly adopt its result")
            if review.get("issues") or review.get("rework_decision"):
                raise SceneReplacementError(
                    "Passing generated-result review cannot contain issues or rework"
                )
            _require_rules(review, GENERATED_RESULT_RULES, "generated-result")
        elif review["status"] == "fail":
            outcomes = _rules(review)
            failed_rule_ids = {
                rule_id for rule_id, status in outcomes.items() if status == "fail"
            }
            issue_rule_ids = {item["rule_id"] for item in review["issues"]}
            if issue_rule_ids != failed_rule_ids:
                raise SceneReplacementError(
                    "Failed generated-result rules and issue rule IDs must match exactly"
                )
            highest = max(
                (item["severity"] for item in review["issues"]),
                key=SEVERITY_RANK.__getitem__,
            )
            hard_failure = bool(failed_rule_ids & HARD_RESULT_RULES)
            normalization_blocked = result["normalization"]["status"] != "ready"
            if normalization_blocked and "result_file_integrity" not in failed_rule_ids:
                raise SceneReplacementError(
                    "A normalization-blocked result must fail result_file_integrity"
                )
            decision = review["rework_decision"]
            expected_strategy = (
                "local_adjustment"
                if highest == "minor" and not hard_failure and not normalization_blocked
                else "source_regeneration"
            )
            if decision["highest_severity"] != highest or decision["strategy"] != expected_strategy:
                raise SceneReplacementError(
                    "Rework strategy must be local_adjustment only for exclusively minor non-hard failures; "
                    "major, critical, and hard-rule failures require source_regeneration"
                )
        else:
            raise SceneReplacementError("Generated-result self-review must pass or fail")
        return [result["anchor_id"]]

    def _apply_sample_review(self, state: dict[str, Any], review: dict[str, Any]) -> list[str]:
        result, _ = self._result_and_digest(state, review)
        scene = self._scene_by_id(state, review["scene_id"])
        if result["anchor_id"] != scene.get("main_anchor_id") or review["scope_id"] != result["result_id"]:
            raise SceneReplacementError("Sample review must bind the adopted main result")
        if state["adopted_results"].get(result["anchor_id"]) != result["result_id"]:
            raise SceneReplacementError("Sample review requires an adopted main result")
        if review["status"] == "pass":
            if not _human(review.get("human_confirmation")) or review["reviewer"].get("kind") != "human":
                raise SceneReplacementError("Sample approval must be a human review and confirmation")
            _require_rules(review, SAMPLE_RULES, "sample")
        return [result["anchor_id"]]

    def _apply_group_review(self, state: dict[str, Any], review: dict[str, Any]) -> list[str]:
        scene = self._scene_by_id(state, review["scene_id"])
        candidate = state.get("scene_review_candidates", {}).get(scene["scene_id"])
        if (
            not candidate
            or candidate.get("dependency_status") != "current"
            or candidate.get("candidate_id") != scene.get("scene_review_candidate_id")
        ):
            raise SceneReplacementError(
                "Group review requires current paired scene evidence prepared first"
            )
        if (
            review["scope_id"] != candidate["candidate_id"]
            or review.get("candidate_id") != candidate["candidate_id"]
            or review.get("candidate_fingerprint")
            != candidate["dependency_fingerprint"]
            or set(review.get("anchor_ids", [])) != set(scene["anchor_ids"])
        ):
            raise SceneReplacementError("Group review must bind every anchor in the physical scene")
        expected_evidence = {
            (item["path"], item["sha256"])
            for item in (
                _reference(candidate["comparison"]),
                candidate["binding_manifest"],
            )
        }
        supplied_evidence = {
            (item["path"], item["sha256"]) for item in review["evidence_refs"]
        }
        if not expected_evidence <= supplied_evidence:
            raise SceneReplacementError(
                "Group review must cite the prepared comparison and binding manifest"
            )
        if review["status"] == "pass":
            if not _human(review.get("human_confirmation")) or review["reviewer"].get("kind") != "human":
                raise SceneReplacementError("Group approval must be a human review and confirmation")
            if review.get("issues") or review.get("rework_decision"):
                raise SceneReplacementError("Passing group review cannot contain issues")
            _require_rules(review, GROUP_RULES, "group")
        return list(scene["anchor_ids"])

    def _apply_global_review(self, state: dict[str, Any], review: dict[str, Any]) -> list[str]:
        candidate = state.get("global_review_candidate")
        if not candidate or candidate.get("dependency_status") != "current":
            raise SceneReplacementError(
                "Global review requires a current prepared multi-frame candidate"
            )
        expected_anchor_ids = [
            item["anchor_id"]
            for item in sorted(
                self._presentation_by_anchor(state).values(),
                key=lambda item: item["timeline_sequence"],
            )
        ]
        if (
            review["scope_id"] != candidate["candidate_id"]
            or review.get("candidate_id") != candidate["candidate_id"]
            or review.get("candidate_fingerprint")
            != candidate["dependency_fingerprint"]
            or review.get("comparison_sha256")
            != candidate["global_comparison"]["sha256"]
            or review.get("anchor_ids") != expected_anchor_ids
        ):
            raise SceneReplacementError(
                "Global review must bind the exact ordered current candidate"
            )
        expected_evidence = {
            (item["path"], item["sha256"])
            for item in (
                _reference(candidate["global_comparison"]),
                candidate["binding_manifest"],
                candidate["accepted_assets_manifest"],
            )
        }
        supplied_evidence = {
            (item["path"], item["sha256"]) for item in review["evidence_refs"]
        }
        if not expected_evidence <= supplied_evidence:
            raise SceneReplacementError(
                "Global review must cite the total comparison, bindings, and accepted assets"
            )
        if review["status"] != "pass":
            raise SceneReplacementError(
                "The final global review must explicitly pass before publication"
            )
        if (
            review.get("reviewer", {}).get("kind") != "human"
            or not _human(review.get("human_confirmation"))
        ):
            raise SceneReplacementError(
                "Global approval must be a human review and confirmation"
            )
        if review.get("issues") or review.get("rework_decision"):
            raise SceneReplacementError("Passing global review cannot contain issues")
        _require_rules(review, GLOBAL_RULES, "global")
        return expected_anchor_ids

    def _scene_by_id(self, state: dict[str, Any], scene_id: str) -> dict[str, Any]:
        scene = next(
            (item for item in state["scene_plan"]["physical_scenes"] if item["scene_id"] == scene_id),
            None,
        )
        if not scene or scene.get("dependency_status") != "current":
            raise SceneReplacementError("Physical scene is missing or stale")
        return scene

    @staticmethod
    def _clear_scene_acceptance(state: dict[str, Any], anchor_id: str) -> None:
        for scene in state["scene_plan"]["physical_scenes"]:
            if anchor_id in scene["anchor_ids"]:
                for field in (
                    "group_review_id",
                    "group_evidence_review_id",
                    "contact_sheet",
                    "scene_review_candidate_id",
                ):
                    scene.pop(field, None)
                candidate = state.get("scene_review_candidates", {}).get(
                    scene["scene_id"]
                )
                if candidate:
                    candidate["dependency_status"] = "stale"
                    candidate["stale_reason"] = "adopted_result_changed"
                global_candidate = state.get("global_review_candidate")
                if global_candidate:
                    global_candidate["dependency_status"] = "stale"
                    global_candidate["stale_reason"] = "adopted_result_changed"
                state["global_review_id"] = None
                if scene.get("main_anchor_id") == anchor_id:
                    scene.pop("main_reference_result_id", None)
                    scene.pop("sample_review_id", None)
                if scene.get("difficult_view", {}).get("anchor_id") == anchor_id:
                    scene.pop("difficult_view_result_id", None)

    def publish(self, inputs: dict[str, Any]) -> dict[str, Any]:
        submitted = deepcopy(inputs.get("submission") or {})

        def apply(state: dict[str, Any], revision: str) -> None:
            plan = state.get("scene_plan")
            if not plan or plan.get("status") != "ready":
                raise SceneReplacementError("A ready V2 scene plan is required")
            anchors = self._anchor_ids(state)
            if set(state["adopted_results"]) != anchors:
                raise SceneReplacementError("Every anchor must have an adopted direct result")
            candidate = state.get("global_review_candidate")
            if not candidate or candidate.get("dependency_status") != "current":
                raise SceneReplacementError(
                    "A current prepared global multi-frame comparison is required"
                )
            global_review = self._review_by_id(state, state.get("global_review_id"))
            if (
                not global_review
                or global_review.get("stage") != "global"
                or global_review.get("scope_id") != candidate["candidate_id"]
                or global_review.get("candidate_fingerprint")
                != candidate["dependency_fingerprint"]
                or global_review.get("status") != "pass"
                or global_review.get("dependency_status") != "current"
                or not _human(global_review.get("human_confirmation"))
            ):
                raise SceneReplacementError(
                    "Publication requires human approval of the current global comparison"
                )
            _require_rules(global_review, GLOBAL_RULES, "global")
            confirmation = submitted.get("confirmation")
            if confirmation is not None and not _human(confirmation):
                raise SceneReplacementError("Publish confirmation must identify a human")

            assets_path = verify_ref(
                candidate["accepted_assets_manifest"],
                self.project_dir,
                label="accepted readable assets manifest",
            )
            assets_manifest = load_json(assets_path, self.project_dir)
            if (
                assets_manifest.get("candidate_id") != candidate["candidate_id"]
                or assets_manifest.get("dependency_fingerprint")
                != candidate["dependency_fingerprint"]
                or assets_manifest.get("naming_version") != "sequence-v1"
            ):
                raise SceneReplacementError(
                    "Accepted readable assets manifest differs from the approved candidate"
                )
            aliases = assets_manifest.get("assets")
            if not isinstance(aliases, list):
                raise SceneReplacementError("Accepted readable assets manifest is invalid")
            aliases_by_anchor = {item.get("anchor_id"): item for item in aliases}
            if len(aliases_by_anchor) != len(aliases) or set(aliases_by_anchor) != anchors:
                raise SceneReplacementError(
                    "Accepted readable assets must bind every anchor exactly once"
                )
            for anchor_id, alias in aliases_by_anchor.items():
                verify_ref(
                    alias["image"],
                    self.project_dir,
                    label=f"readable accepted image {anchor_id}",
                )
            verify_ref(
                candidate["global_comparison"],
                self.project_dir,
                label="global multi-frame comparison",
            )
            verify_ref(
                candidate["binding_manifest"],
                self.project_dir,
                label="global comparison bindings",
            )

            accepted = []
            comparisons = []
            delivery_dir = self.artifact_root / "deliveries" / revision
            for scene in plan["physical_scenes"]:
                sample = self._review_by_id(state, scene.get("sample_review_id"))
                group = self._review_by_id(state, scene.get("group_review_id"))
                scene_candidate = state.get("scene_review_candidates", {}).get(
                    scene["scene_id"]
                )
                if not sample or sample.get("status") != "pass" or not _human(sample.get("human_confirmation")):
                    raise SceneReplacementError(f"Scene {scene['scene_id']} lacks human sample approval")
                if not group or group.get("status") != "pass" or not _human(group.get("human_confirmation")):
                    raise SceneReplacementError(f"Scene {scene['scene_id']} lacks human group approval")
                _require_rules(sample, SAMPLE_RULES, "sample")
                _require_rules(group, GROUP_RULES, "group")
                if (
                    not scene_candidate
                    or scene_candidate.get("dependency_status") != "current"
                    or scene.get("scene_review_candidate_id")
                    != scene_candidate.get("candidate_id")
                    or group.get("scope_id") != scene_candidate.get("candidate_id")
                ):
                    raise SceneReplacementError(
                        f"Scene {scene['scene_id']} lacks current approved comparison evidence"
                    )
                verify_ref(
                    scene_candidate["comparison"],
                    self.project_dir,
                    label=f"scene comparison {scene['scene_id']}",
                )
                verify_ref(
                    scene_candidate["binding_manifest"],
                    self.project_dir,
                    label=f"scene comparison bindings {scene['scene_id']}",
                )
                for anchor_id in scene["anchor_ids"]:
                    anchor = self._anchor(state, anchor_id)
                    alias = aliases_by_anchor[anchor_id]
                    presentation = self._presentation_by_anchor(state)[anchor_id]
                    for key in (
                        "clip_id",
                        "clip_sequence",
                        "clip_display_id",
                        "keyframe_sequence",
                        "display_id",
                        "timeline_sequence",
                    ):
                        if alias.get(key) != presentation[key]:
                            raise SceneReplacementError(
                                f"Readable asset presentation binding differs: {anchor_id}"
                            )
                    result_id = state["adopted_results"][anchor_id]
                    result = state["results"][result_id]
                    request = state["edit_requests"][result["request_id"]]
                    review = self._review_by_id(state, result.get("adoption_review_id"))
                    if (
                        result.get("dependency_status") != "current"
                        or result.get("status") != "adopted"
                        or result["normalization"]["status"] != "ready"
                        or not review
                        or review.get("status") != "pass"
                    ):
                        raise SceneReplacementError(f"Adopted result is stale or unreviewed: {result_id}")
                    if request["prompt_sha256"] != _prompt_hash(request["prompt"]):
                        raise SceneReplacementError(f"Prompt hash mismatch: {request['request_id']}")
                    image = result["normalization"]["image"]
                    verify_ref(image, self.project_dir, label=f"accepted direct result {result_id}")
                    if (
                        alias.get("result_id") != result_id
                        or alias["image"]["sha256"] != image["sha256"]
                    ):
                        raise SceneReplacementError(
                            f"Readable image differs from adopted direct result: {anchor_id}"
                        )
                    attempts = []
                    task = state["logical_tasks"][result["logical_task_id"]]
                    for request_id in task["request_ids"]:
                        item = state["edit_requests"][request_id]
                        if item.get("counts_toward_limit"):
                            attempts.append(
                                {
                                    "request_id": request_id,
                                    "generation_mode": item["generation_mode"],
                                    "attempt_number": item["attempt_number"],
                                    "prompt": item["prompt"],
                                    "prompt_sha256": item["prompt_sha256"],
                                    "prompt_builder_version": item.get(
                                        "prompt_builder_version"
                                    ),
                                    "execution_status": item["execution_status"],
                                }
                            )
                    accepted.append(
                        {
                            "anchor_id": anchor_id,
                            "scene_id": scene["scene_id"],
                            **{
                                key: deepcopy(presentation[key])
                                for key in (
                                    "clip_id",
                                    "clip_sequence",
                                    "clip_display_id",
                                    "keyframe_sequence",
                                    "display_id",
                                    "timeline_sequence",
                                )
                            },
                            "role": anchor["role"],
                            "purpose": anchor["purpose"],
                            "source_time": deepcopy(anchor["source_time"]),
                            "source_image": deepcopy(anchor["image"]),
                            "image": deepcopy(alias["image"]),
                            "scene_plan_id": plan["scene_plan_id"],
                            "selected_direction_id": scene["selected_direction_id"],
                            "selection_receipt_id": scene["selection_receipt_id"],
                            "logical_task_id": result["logical_task_id"],
                            "request_id": request["request_id"],
                            "result_id": result_id,
                            "review_id": review["review_id"],
                            "generation_mode": result["generation_mode"],
                            "prompt": request["prompt"],
                            "prompt_sha256": request["prompt_sha256"],
                            "prompt_builder_version": request.get(
                                "prompt_builder_version"
                            ),
                            "attempt_history": attempts,
                            "human_confirmation": deepcopy(group["human_confirmation"]),
                        }
                    )
                comparisons.append(
                    {
                        "scene_id": scene["scene_id"],
                        "candidate_id": scene_candidate["candidate_id"],
                        "dependency_fingerprint": scene_candidate[
                            "dependency_fingerprint"
                        ],
                        "anchor_ids": deepcopy(scene_candidate["anchor_ids"]),
                        "ordered_display_ids": deepcopy(
                            scene_candidate["ordered_display_ids"]
                        ),
                        "comparison_ref": _reference(
                            scene_candidate["comparison"]
                        ),
                        "binding_manifest_ref": deepcopy(
                            scene_candidate["binding_manifest"]
                        ),
                    }
                )
            accepted.sort(key=lambda item: item["timeline_sequence"])
            current_state_ref = deepcopy(self._mutation_parent_index["manifest_index"]["state.json"])
            delivery = {
                "schema_version": STATE_SCHEMA_VERSION,
                "tool": "replication_scene_replacement",
                "project_id": state["project_id"],
                "replacement_revision": revision,
                "naming_version": "sequence-v1",
                "source_snapshot_id": state["source_snapshot"]["snapshot_id"],
                "source_snapshot_fingerprint": state["source_snapshot"]["fingerprints"]["snapshot"],
                "source_snapshot_ref": deepcopy(
                    self._mutation_parent_index["manifest_index"]["source_snapshot.json"]
                ),
                "accepted_state_ref": current_state_ref,
                "upstream_plan_revision": state["source_snapshot"]["upstream"]["plan_revision"],
                "scene_plan_id": plan["scene_plan_id"],
                "audience_profile": deepcopy(plan["audience_profile"]),
                "status": "ready",
                "downstream_adaptation_status": "unverified",
                "physical_scenes": [
                    {
                        "scene_id": scene["scene_id"],
                        "anchor_ids": deepcopy(scene["anchor_ids"]),
                        "selected_direction_id": scene["selected_direction_id"],
                        "selection_receipt_id": scene["selection_receipt_id"],
                        "main_reference_result_id": scene["main_reference_result_id"],
                        "sample_review_id": scene["sample_review_id"],
                        "scene_review_candidate_id": scene[
                            "scene_review_candidate_id"
                        ],
                        "group_review_id": scene["group_review_id"],
                    }
                    for scene in plan["physical_scenes"]
                ],
                "accepted_results": accepted,
                "comparison_assets": comparisons,
                "global_review_candidate_id": candidate["candidate_id"],
                "global_review_id": global_review["review_id"],
                "global_human_confirmation": deepcopy(
                    global_review["human_confirmation"]
                ),
                "accepted_assets_manifest_ref": deepcopy(
                    candidate["accepted_assets_manifest"]
                ),
                "global_comparison_ref": _reference(
                    candidate["global_comparison"]
                ),
                "global_binding_manifest_ref": deepcopy(
                    candidate["binding_manifest"]
                ),
            }
            delivery["delivery_fingerprint"] = sha256_json(delivery)
            _validate_document("scene_replacement_v2_delivery_manifest", delivery)
            delivery_path = delivery_dir / "delivery.json"
            atomic_write_json(delivery_path, delivery)
            state["delivery"] = {
                "path": self.relative(delivery_path),
                "sha256": sha256_file(delivery_path),
                "delivery_fingerprint": delivery["delivery_fingerprint"],
                "status": "ready",
                "confirmation": deepcopy(confirmation),
            }

        return self._mutate("publish", inputs, apply)

    def _validate_state(self, state: dict[str, Any]) -> None:
        _validate_document("scene_replacement_v2_state", state)
        verify_source_snapshot(state["source_snapshot"], self.project_dir)
        if state.get("schema_version") == STATE_SCHEMA_VERSION:
            snapshot_presentation = state["source_snapshot"].get("presentation_map")
            if (
                snapshot_presentation is not None
                and state.get("presentation_map") != snapshot_presentation
            ):
                raise SceneReplacementError(
                    "State presentation map differs from the source snapshot"
                )
            self._presentation_by_anchor(state)
        for request_id, request in state["edit_requests"].items():
            if request_id != request["request_id"] or request["prompt_sha256"] != _prompt_hash(request["prompt"]):
                raise SceneReplacementError("Persisted request key or prompt hash differs")
            _validate_document("scene_replacement_v2_edit_request", request)
            image_references = request.get(
                "ordered_image_references", request.get("ordered_references", [])
            )
            for item in image_references:
                verify_ref(item["reference"], self.project_dir, label=f"request reference {request_id}")
            for item in request.get("context_refs", []):
                verify_ref(
                    item["reference"],
                    self.project_dir,
                    label=f"request context {request_id}",
                )
        for result_id, result in state["results"].items():
            if result_id != result["result_id"]:
                raise SceneReplacementError("Persisted result key differs")
            verify_ref(result["generated_image"], self.project_dir, label=f"generated image {result_id}")
            verify_ref(result["raw_response"], self.project_dir, label=f"raw response {result_id}")
            if result["normalization"]["status"] == "ready":
                verify_ref(result["normalization"]["image"], self.project_dir, label=f"direct image {result_id}")
            if result.get("comparison"):
                verify_ref(result["comparison"], self.project_dir, label=f"comparison {result_id}")
            for item in result["review_crops"]:
                verify_ref(item, self.project_dir, label=f"review crop {result_id}")
        for review in state["reviews"]:
            for reference in review["evidence_refs"]:
                verify_ref(reference, self.project_dir, label=f"review evidence {review['review_id']}")
            for rule in review["rules"]:
                for reference in rule["evidence_refs"]:
                    verify_ref(
                        reference,
                        self.project_dir,
                        label=f"review rule evidence {review['review_id']}",
                    )
            for issue in review["issues"]:
                for reference in issue["evidence_refs"]:
                    verify_ref(
                        reference,
                        self.project_dir,
                        label=f"review issue evidence {review['review_id']}",
                    )
        for scene_id, candidate in state.get("scene_review_candidates", {}).items():
            if candidate.get("scene_id") != scene_id:
                raise SceneReplacementError("Scene review candidate index differs")
            verify_ref(
                candidate["comparison"],
                self.project_dir,
                label=f"scene review comparison {scene_id}",
            )
            verify_ref(
                candidate["binding_manifest"],
                self.project_dir,
                label=f"scene review bindings {scene_id}",
            )
        candidate = state.get("global_review_candidate")
        if candidate:
            verify_ref(
                candidate["accepted_assets_manifest"],
                self.project_dir,
                label="global accepted assets",
            )
            verify_ref(
                candidate["global_comparison"],
                self.project_dir,
                label="global comparison",
            )
            verify_ref(
                candidate["binding_manifest"],
                self.project_dir,
                label="global comparison bindings",
            )
        for anchor_id, result_id in state["adopted_results"].items():
            result = state["results"].get(result_id)
            if not result or result["anchor_id"] != anchor_id or result["status"] != "adopted":
                raise SceneReplacementError("Adopted-result index differs from persisted result")

    def validate(self, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        index, state = self._load_current()
        issues: list[str] = []
        try:
            self._validate_state(state)
            if state.get("delivery"):
                delivery_path = verify_ref(state["delivery"], self.project_dir, label="V2 delivery")
                delivery = load_json(delivery_path, self.project_dir)
                _validate_document("scene_replacement_v2_delivery_manifest", delivery)
                if delivery["delivery_fingerprint"] != state["delivery"]["delivery_fingerprint"]:
                    raise SceneReplacementError("Delivery fingerprint differs from state")
                for field in (
                    "accepted_assets_manifest_ref",
                    "global_comparison_ref",
                    "global_binding_manifest_ref",
                ):
                    verify_ref(delivery[field], self.project_dir, label=field)
                for accepted in delivery["accepted_results"]:
                    verify_ref(
                        accepted["image"],
                        self.project_dir,
                        label=f"delivered keyframe {accepted['display_id']}",
                    )
                    if accepted["prompt_sha256"] != _prompt_hash(
                        accepted["prompt"]
                    ):
                        raise SceneReplacementError(
                            f"Delivered prompt hash differs: {accepted['display_id']}"
                        )
                    for attempt in accepted["attempt_history"]:
                        if attempt["prompt_sha256"] != _prompt_hash(
                            attempt["prompt"]
                        ):
                            raise SceneReplacementError(
                                "Delivered attempt prompt hash differs: "
                                f"{attempt['request_id']}"
                            )
                for comparison in delivery["comparison_assets"]:
                    verify_ref(
                        comparison["comparison_ref"],
                        self.project_dir,
                        label=f"scene comparison {comparison['scene_id']}",
                    )
                    verify_ref(
                        comparison["binding_manifest_ref"],
                        self.project_dir,
                        label=f"scene comparison bindings {comparison['scene_id']}",
                    )
        except Exception as exc:
            issues.append(str(exc))
        result = self._result(state, index)
        result["validation_status"] = "failed" if issues else "passed"
        result["issues"] = issues
        if issues:
            result["package_status"] = "blocked"
        if state.get("delivery"):
            result["delivery"] = load_json(
                self.project_dir / state["delivery"]["path"], self.project_dir
            )
        return result


__all__ = [
    "AUDIENCE_PROFILE",
    "GENERATED_RESULT_RULES",
    "GLOBAL_RULES",
    "GROUP_RULES",
    "HARD_RESULT_RULES",
    "OPERATIONS",
    "PRE_GENERATION_RULES",
    "SAMPLE_RULES",
    "SceneReplacementV2Engine",
]
