"""Deterministic persistence for direct replication scene generation.

The tool deliberately does not call an image model.  It prepares immutable
requests for the coding agent's built-in ImageGen capability and imports the
result after that external call has completed.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from jsonschema.exceptions import ValidationError

from lib.paths import PROJECTS_DIR
from lib.replication_scene_replacement.engine_v2 import SceneReplacementV2Engine
from lib.replication_scene_replacement.errors import SceneReplacementError
from schemas.replication import load_phase2_schema, validate_phase2_document
from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    ResumeSupport,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolTier,
)


OPERATIONS = (
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
)

# Retain the public symbol used by integrations/tests while routing it to V2.
SceneReplacementEngine = SceneReplacementV2Engine


class ReplicationSceneReplacement(BaseTool):
    """Manage Phase 2 scene-replacement state without invoking generation."""

    name = "replication_scene_replacement"
    version = "0.4.0"
    tier = ToolTier.ENHANCE
    capability = "enhancement"
    provider = "openmontage"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL
    resume_support = ResumeSupport.FROM_CHECKPOINT

    dependencies = ["python:PIL"]
    install_instructions = "Install the base OpenMontage dependencies (Pillow is required)."
    capabilities = [
        "freeze_replication_source_revision",
        "plan_noncontiguous_physical_scenes",
        "record_one_of_three_human_scene_direction_selections",
        "prepare_direct_builtin_imagegen_requests",
        "expose_exact_generation_prompt",
        "reconcile_external_generation_results",
        "persist_multimodal_self_review",
        "route_rework_by_issue_severity",
        "render_scene_and_global_source_result_comparisons",
        "publish_readable_sequence_named_keyframes",
        "publish_scene_replacement_package",
    ]
    supports = {
        "generation": "external_agent_handshake_only",
        "requested_capability": "imagegen2",
        "session_tool": "image_gen",
        "max_calls_per_logical_task": 3,
        "target_size": [720, 1280],
        "max_aspect_ratio_difference_percent": 5,
    }
    best_for = [
        "traceable direct scene replacement of Phase 1 replication anchors",
        "North American consumer-realistic product scenes with self-review and human gates",
    ]
    not_good_for = [
        "calling an image-generation provider from Python",
        "identity replacement, video generation, audio processing, or final stitching",
    ]
    resource_profile = ResourceProfile(
        cpu_cores=2,
        ram_mb=2048,
        vram_mb=0,
        disk_mb=4096,
        network_required=False,
    )
    idempotency_key_fields = [
        "operation",
        "project_id",
        "output_path",
        "parent_replacement_revision",
        "upstream_index_path",
        "upstream_plan_revision",
        "supplementary_anchor_confirmations",
        "submission",
    ]
    side_effects = [
        "writes immutable Phase 2 revisions and imported assets under the project workspace",
        "publishes only after scene-level and final global human approval",
    ]
    fallback_tools: list[str] = []
    agent_skills: list[str] = []
    user_visible_verification = [
        "Inspect the exact prompt and ordered references before image_gen dispatch",
        "Inspect paired source/direct-result scene boards and the final multi-frame board before approval",
    ]

    # Load the published schema rather than maintaining a second, drifting
    # inline copy in the Python class.  Its relative refs are resolved by the
    # offline registry in ``schemas.replication`` during execution.
    input_schema = load_phase2_schema("replication_scene_replacement_v2")
    output_schema = {
        "type": "object",
        "required": ["replacement_revision", "package_status", "index_path"],
        "properties": {
            "replacement_revision": {"type": "string"},
            "package_status": {
                "enum": [
                    "source_locked",
                    "in_progress",
                    "awaiting_human",
                    "awaiting_global_approval",
                    "ready",
                    "blocked",
                    "published",
                ]
            },
            "index_path": {"type": "string"},
            "validation_status": {"enum": ["passed", "failed", "not_run"]},
            "issues": {"type": "array", "items": {"type": "string"}},
            "next_actions": {"type": "array", "items": {"type": "object"}},
            "checkpoint_artifacts": {
                "type": "object",
                "properties": {
                    "replication_source_snapshot": {"type": "object"},
                    "scene_replacement_package": {"type": "object"},
                    "scene_replacement_delivery": {"type": "object"},
                },
                "additionalProperties": False,
            },
            "state": {"type": "object"},
            "delivery": {"type": ["object", "null"]},
            "generation_packet": {
                "type": "object",
                "required": [
                    "request_id", "attempt_number", "generation_mode", "prompt",
                    "prompt_sha256", "ordered_image_references", "context_refs", "requested_capability",
                    "required_actual_tool", "output_requirement"
                ],
                "properties": {
                    "request_id": {"type": "string"},
                    "attempt_number": {"type": "integer", "minimum": 1, "maximum": 3},
                    "generation_mode": {"enum": ["initial", "local_adjustment", "source_regeneration"]},
                    "prompt": {"type": "string", "minLength": 1},
                    "prompt_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
                    "prompt_builder_version": {"type": ["string", "null"]},
                    "ordered_image_references": {"type": "array", "minItems": 1},
                    "context_refs": {"type": "array", "minItems": 1},
                    "requested_capability": {"const": "imagegen2"},
                    "required_actual_tool": {"const": "image_gen"},
                    "output_requirement": {
                        "type": "object",
                        "required": [
                            "width", "height", "aspect_ratio",
                            "max_aspect_ratio_difference_percent"
                        ],
                        "properties": {
                            "width": {"const": 720},
                            "height": {"const": 1280},
                            "aspect_ratio": {"const": "9:16"},
                            "max_aspect_ratio_difference_percent": {"const": 5}
                        },
                        "additionalProperties": False
                    }
                },
                "additionalProperties": False
            },
        },
    }

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        started = time.monotonic()
        operation = inputs.get("operation")
        project_id = inputs.get("project_id")
        output_path = inputs.get("output_path")
        if operation not in OPERATIONS:
            return ToolResult(success=False, error="Invalid operation")
        if not isinstance(project_id, str) or not isinstance(output_path, str):
            return ToolResult(
                success=False,
                error="project_id and output_path are required",
            )

        try:
            validate_phase2_document("replication_scene_replacement_v2", inputs)
            SceneReplacementEngine.validate_project_id(project_id)
            projects_root = PROJECTS_DIR.expanduser().resolve(strict=True)
            project_leaf = PROJECTS_DIR.expanduser() / project_id
            if project_leaf.is_symlink():
                raise SceneReplacementError(
                    "Project workspace leaf must not be a symlink"
                )
            try:
                project_dir = project_leaf.resolve(strict=True)
            except FileNotFoundError as exc:
                raise SceneReplacementError(
                    "Project workspace must be created before running "
                    "replication_scene_replacement"
                ) from exc
            if (
                not project_dir.is_dir()
                or project_dir.parent != projects_root
                or not project_dir.is_relative_to(projects_root)
            ):
                raise SceneReplacementError(
                    "Project workspace must be a real direct child of projects/"
                )
            supplied_output = Path(output_path).expanduser()
            if not supplied_output.is_absolute():
                supplied_output = (Path.cwd() / supplied_output).resolve()
            else:
                supplied_output = supplied_output.resolve()
            if not supplied_output.is_relative_to(project_dir):
                raise SceneReplacementError(
                    "output_path must stay inside the project workspace"
                )
            engine = SceneReplacementEngine(project_dir, supplied_output)
            data = engine.execute(operation, inputs)
            data.setdefault("index_path", engine.relative(supplied_output))
            data.setdefault("replacement_revision", "none")
            data.setdefault("package_status", "blocked")
            data.setdefault("validation_status", "not_run")
            data.setdefault("issues", [])
            data.setdefault("next_actions", [])
            artifacts = list(dict.fromkeys(data.pop("artifacts", [str(supplied_output)])))
            return ToolResult(
                success=data.get("validation_status") != "failed" and data.get("package_status") != "blocked",
                data=data,
                artifacts=artifacts,
                error=data.get("error"),
                duration_seconds=round(time.monotonic() - started, 3),
            )
        except (
            SceneReplacementError,
            ValidationError,
            OSError,
            ValueError,
            KeyError,
            TypeError,
        ) as exc:
            return ToolResult(
                success=False,
                data={
                    "replacement_revision": inputs.get("parent_replacement_revision") or "none",
                    "package_status": "blocked",
                    "validation_status": "failed",
                    "issues": [str(exc)],
                    "next_actions": [],
                },
                error=str(exc),
                duration_seconds=round(time.monotonic() - started, 3),
            )
