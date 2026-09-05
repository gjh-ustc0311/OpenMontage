"""SceneDetect + agent-review preprocessing for short-video replication."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from lib.paths import PROJECTS_DIR
from lib.replication_preprocess.analysis import MediaAnalysisError
from lib.replication_preprocess.engine import (
    ReplicationEngine,
    ReplicationPreprocessError,
)
from lib.replication_preprocess.review import ReviewValidationError
from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolTier,
)


class ReplicationPreprocess(BaseTool):
    """Build exact, reviewable generation clips without embedded AI models."""

    name = "replication_preprocess"
    version = "0.1.0"
    tier = ToolTier.ANALYZE
    capability = "analysis"
    provider = "openmontage"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL

    dependencies = [
        "cmd:ffmpeg",
        "cmd:ffprobe",
        "python:av",
        "python:cv2",
        "python:scenedetect",
    ]
    install_instructions = (
        "Install the CPU-only replication profile: make install-replication. "
        "No GPU, Torch, Transformers, or model weights are required."
    )
    agent_skills = ["ffmpeg"]
    capabilities = [
        "plan_replication_segments",
        "build_boundary_review_evidence",
        "apply_agent_boundary_review",
        "export_generation_clips",
        "validate_replication_package",
    ]
    best_for = [
        "PTS-accurate preprocessing of short product videos",
        "agent-reviewed scene grouping before video replication",
    ]
    not_good_for = [
        "semantic scene classification without an AI coding assistant",
        "HDR inputs or videos with multiple video streams",
    ]
    resource_profile = ResourceProfile(
        cpu_cores=4,
        ram_mb=2048,
        vram_mb=0,
        disk_mb=4096,
        network_required=False,
    )
    idempotency_key_fields = [
        "operation",
        "project_id",
        "source_path",
        "config_path",
        "profile",
        "parent_plan_revision",
        "review_submission",
        "manual_overrides",
    ]
    side_effects = [
        "writes immutable manifests and review images under the project workspace",
        "exports validated MP4 clips only for ready plans",
    ]
    fallback_tools: list[str] = []
    user_visible_verification = [
        "Inspect the boundary contact sheet before submitting scene relationships",
        "Verify exported clips preserve source order, timing, and audio presence",
    ]

    input_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["operation", "project_id", "output_path"],
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["plan", "export", "validate", "run"],
            },
            "project_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "output_path": {"type": "string", "minLength": 1},
            "source_path": {"type": "string", "minLength": 1},
            "plan_path": {"type": ["string", "null"]},
            "config_path": {"type": ["string", "null"]},
            "profile": {"type": "string", "default": "default-v1"},
            "parent_plan_revision": {
                "type": ["string", "null"],
                "pattern": "^r[0-9]{4}$",
            },
            "review_submission": {
                "type": ["object", "null"],
            },
            "manual_overrides": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "action", "actor", "reason", "parent_plan_revision",
                        "config_fingerprint",
                    ],
                    "properties": {
                        "action": {
                            "enum": [
                                "force_hard", "force_soft", "insert_boundary",
                                "remove_boundary", "drop",
                            ]
                        },
                        "actor": {"type": "string", "minLength": 1},
                        "reason": {"type": "string", "minLength": 1},
                        "parent_plan_revision": {
                            "type": "string", "pattern": "^r[0-9]{4}$"
                        },
                        "config_fingerprint": {
                            "type": "string", "pattern": "^[a-f0-9]{64}$"
                        },
                    },
                },
                "default": [],
            },
        },
    }
    output_schema = {
        "type": "object",
        "required": ["plan_status", "review_status", "plan_revision"],
        "properties": {
            "plan_status": {"enum": ["ready", "needs_review", "blocked"]},
            "review_status": {
                "enum": [
                    "not_required",
                    "pending_agent",
                    "accepted",
                    "needs_more_evidence",
                    "needs_human",
                ]
            },
            "plan_revision": {"type": "string"},
            "index_path": {"type": "string"},
            "contact_sheet_path": {"type": ["string", "null"]},
            "clip_paths": {"type": "array", "items": {"type": "string"}},
        },
    }

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        started = time.monotonic()
        operation = inputs.get("operation")
        project_id = inputs.get("project_id")
        output_path = inputs.get("output_path")
        if operation not in {"plan", "export", "validate", "run"}:
            return ToolResult(success=False, error="Invalid operation")
        if not isinstance(project_id, str) or not isinstance(output_path, str):
            return ToolResult(success=False, error="project_id and output_path are required")
        try:
            ReplicationEngine.validate_project_id(project_id)
            project_dir = (PROJECTS_DIR / project_id).resolve()
            if not project_dir.is_dir():
                raise ReplicationPreprocessError(
                    "Project workspace must be created before running replication_preprocess"
                )
            supplied_output = Path(output_path).expanduser()
            if not supplied_output.is_absolute():
                supplied_output = (Path.cwd() / supplied_output).resolve()
            engine = ReplicationEngine(project_dir, supplied_output)
            if operation == "plan":
                data = engine.plan(inputs)
            elif operation == "export":
                data = engine.export(inputs)
            elif operation == "validate":
                data = engine.validate(inputs)
            else:
                data = engine.plan(inputs)
                if data["plan_status"] == "ready":
                    data = engine.export({**inputs, "plan_path": str(supplied_output)})
                    validation = engine.validate({**inputs, "plan_path": str(supplied_output)})
                    data["validation_status"] = validation["validation_status"]
                    data["issues"] = validation["issues"]
            data.setdefault("index_path", engine._relative(supplied_output))
            data.setdefault("contact_sheet_path", data.get("index", {}).get("contact_sheet_path"))
            data.setdefault("clip_paths", [])
            artifacts = list(dict.fromkeys(data.pop("artifacts", [str(supplied_output)])))
            return ToolResult(
                success=data.get("validation_status") != "failed",
                data=data,
                artifacts=artifacts,
                duration_seconds=round(time.monotonic() - started, 3),
            )
        except (
            ReplicationPreprocessError,
            ReviewValidationError,
            MediaAnalysisError,
            OSError,
            ValueError,
        ) as exc:
            message = str(exc)
            review_recoverable = isinstance(exc, ReviewValidationError) or any(
                needle in message.lower()
                for needle in (
                    "review submission",
                    "parent revision",
                    "parent plan revision",
                    "manual override",
                    "only a ready plan",
                )
            )
            return ToolResult(
                success=False,
                data={
                    "plan_status": "needs_review" if review_recoverable else "blocked",
                    "review_status": "pending_agent" if review_recoverable else "not_required",
                    "plan_revision": inputs.get("parent_plan_revision") or "none",
                    "error_code": "review_required" if review_recoverable else "dependency_or_input_blocked",
                },
                error=message,
                duration_seconds=round(time.monotonic() - started, 3),
            )
