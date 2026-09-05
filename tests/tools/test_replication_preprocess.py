from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.analysis import replication_preprocess as tool_module
from tools.analysis.replication_preprocess import ReplicationPreprocess


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available",
)


def _make_fixture(path: Path, *, with_audio: bool = False) -> None:
    command = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=red:s=160x120:r=10:d=3.5",
        "-f", "lavfi", "-i", "color=c=blue:s=160x120:r=10:d=3.5",
    ]
    if with_audio:
        command += [
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=7",
        ]
    command += [
        "-filter_complex", "[0:v:0][1:v:0]concat=n=2:v=1:a=0[v]",
        "-map", "[v]",
    ]
    if with_audio:
        command += ["-map", "2:a:0", "-c:a", "aac", "-b:a", "96k"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _write_config(path: Path) -> None:
    path.write_text(
        json.dumps({
            "scene_detection": {
                "initial_threshold": 20,
                "minimum_threshold": 20,
                "threshold_step": 5,
            },
            "export": {"preset": "ultrafast"},
        }),
        encoding="utf-8",
    )


def _submission(request: dict) -> dict:
    return {
        "schema_version": "1.0",
        "request_id": request["request_id"],
        "request_sha256": request["request_sha256"],
        "parent_plan_revision": request["parent_plan_revision"],
        "review_protocol_version": request["review_protocol_version"],
        "reviewer": {"kind": "ai_coding_assistant", "name": "pytest-reviewer"},
        "decisions": [
            {
                "review_item_id": item["review_item_id"],
                "boundary_id": item["boundary_id"],
                "relationship": "different_scene",
                "confidence": "high",
                "evidence_refs": ["left_edge", "right_edge", "evidence_board"],
                "rationale": "The full-frame color and visual context change at the cut.",
            }
            for item in request["items"]
        ],
    }


def test_tool_requires_caller_created_workspace(tmp_path: Path, monkeypatch) -> None:
    projects = tmp_path / "projects"
    projects.mkdir()
    monkeypatch.setattr(tool_module, "PROJECTS_DIR", projects)

    result = ReplicationPreprocess().execute({
        "operation": "plan",
        "project_id": "missing-project",
        "source_path": str(tmp_path / "missing.mp4"),
        "output_path": str(projects / "missing-project" / "replication.json"),
    })

    assert result.success is False
    assert "must be created" in (result.error or "")


@pytest.mark.parametrize("with_audio", [False, True])
def test_plan_review_export_validate_round_trip(
    tmp_path: Path, monkeypatch, with_audio: bool
) -> None:
    pytest.importorskip("av")
    pytest.importorskip("scenedetect")
    pytest.importorskip("cv2")

    projects = tmp_path / "projects"
    project = projects / "demo"
    project.mkdir(parents=True)
    source = tmp_path / "source.mp4"
    config = tmp_path / "replication.json"
    index = project / "artifacts" / "replication" / "index.json"
    _make_fixture(source, with_audio=with_audio)
    _write_config(config)
    monkeypatch.setattr(tool_module, "PROJECTS_DIR", projects)
    tool = ReplicationPreprocess()
    base = {
        "project_id": "demo",
        "source_path": str(source),
        "config_path": str(config),
        "output_path": str(index),
    }

    first = tool.execute({"operation": "plan", **base})
    assert first.success is True, first.error
    assert first.data["plan_status"] == "needs_review"
    assert first.data["review_status"] == "pending_agent"
    assert first.data["plan_revision"] == "r0001"

    repeated = tool.execute({"operation": "plan", **base})
    assert repeated.success is True, repeated.error
    assert repeated.data["plan_revision"] == "r0001"
    assert len(list((project / "artifacts" / "replication" / "revisions").iterdir())) == 1

    request_path = project / first.data["index"]["next_action"]["request_path"]
    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert request["items"]
    for item in request["items"]:
        assert set(item["evidence"]) >= {
            "left_keyframe", "left_context", "left_edge", "right_edge",
            "right_context", "right_keyframe", "left_background",
            "right_background", "evidence_board",
        }
        for evidence in item["evidence"].values():
            assert (project / evidence["path"]).is_file()

    if not with_audio:
        evidence_path = project / request["items"][0]["evidence"]["left_edge"]["path"]
        original_evidence = evidence_path.read_bytes()
        evidence_path.write_bytes(original_evidence + b"tampered")
        rejected = tool.execute({
            "operation": "plan",
            **base,
            "parent_plan_revision": "r0001",
            "review_submission": _submission(request),
        })
        assert rejected.success is False
        assert "evidence hash mismatch" in (rejected.error or "").lower()
        evidence_path.write_bytes(original_evidence)

    reviewed = tool.execute({
        "operation": "plan",
        **base,
        "parent_plan_revision": "r0001",
        "review_submission": _submission(request),
    })
    assert reviewed.success is True, reviewed.error
    assert reviewed.data["plan_status"] == "ready"
    assert reviewed.data["review_status"] == "accepted"
    assert reviewed.data["plan_revision"] == "r0002"
    assert len(reviewed.data["generation_clips"]) == 2

    exported = tool.execute({
        "operation": "export",
        "project_id": "demo",
        "output_path": str(index),
    })
    assert exported.success is True, exported.error
    assert exported.data["export_status"] == "validated"
    assert len(exported.data["clip_paths"]) == 2
    assert all((project / path).is_file() for path in exported.data["clip_paths"])
    report = json.loads(
        (project / exported.data["export_report_path"]).read_text(encoding="utf-8")
    )
    assert all(item["audio_streams"] == int(with_audio) for item in report["clips"])

    repeated_export = tool.execute({
        "operation": "export",
        "project_id": "demo",
        "output_path": str(index),
    })
    assert repeated_export.success is True, repeated_export.error
    assert repeated_export.data["clip_paths"] == exported.data["clip_paths"]

    validated = tool.execute({
        "operation": "validate",
        "project_id": "demo",
        "output_path": str(index),
    })
    assert validated.success is True, validated.error
    assert validated.data["validation_status"] == "passed"
    assert validated.data["issues"] == []

    if not with_audio:
        state_path = project / "artifacts" / "replication" / "revisions" / "r0002" / "plan_state.json"
        original_state = state_path.read_bytes()
        state_path.write_bytes(original_state + b" ")
        tampered = tool.execute({
            "operation": "validate",
            "project_id": "demo",
            "output_path": str(index),
        })
        assert tampered.success is False
        assert "manifest_hash_mismatch:plan_state.json" in tampered.data["issues"]


def test_two_uncertain_rounds_escalate_then_manual_override_resolves(
    tmp_path: Path, monkeypatch
) -> None:
    pytest.importorskip("av")
    pytest.importorskip("scenedetect")
    pytest.importorskip("cv2")
    projects = tmp_path / "projects"
    project = projects / "uncertain"
    project.mkdir(parents=True)
    source = tmp_path / "source.mp4"
    config = tmp_path / "replication.json"
    index = project / "artifacts" / "replication" / "index.json"
    _make_fixture(source)
    _write_config(config)
    monkeypatch.setattr(tool_module, "PROJECTS_DIR", projects)
    tool = ReplicationPreprocess()
    base = {
        "project_id": "uncertain",
        "source_path": str(source),
        "config_path": str(config),
        "output_path": str(index),
    }

    first = tool.execute({"operation": "plan", **base})
    request_path = project / first.data["index"]["next_action"]["request_path"]
    request = json.loads(request_path.read_text(encoding="utf-8"))

    def uncertain_submission(value: dict) -> dict:
        submission = _submission(value)
        for decision in submission["decisions"]:
            decision.update({
                "relationship": "uncertain",
                "confidence": "low",
                "rationale": "The available views do not establish spatial continuity.",
            })
        return submission

    second = tool.execute({
        "operation": "plan",
        **base,
        "parent_plan_revision": "r0001",
        "review_submission": uncertain_submission(request),
    })
    assert second.success is True, second.error
    assert second.data["review_status"] == "needs_more_evidence"
    request2_path = project / second.data["index"]["next_action"]["request_path"]
    request2 = json.loads(request2_path.read_text(encoding="utf-8"))
    assert request2["review_round"] == 2

    third = tool.execute({
        "operation": "plan",
        **base,
        "parent_plan_revision": "r0002",
        "review_submission": uncertain_submission(request2),
    })
    assert third.success is True, third.error
    assert third.data["plan_status"] == "needs_review"
    assert third.data["review_status"] == "needs_human"
    assert third.data["generation_clips"] == []

    boundary_id = third.data["boundaries"][0]["boundary_id"]
    resolved = tool.execute({
        "operation": "plan",
        **base,
        "parent_plan_revision": "r0003",
        "manual_overrides": [{
            "action": "force_hard",
            "boundary_id": boundary_id,
            "actor": "human-reviewer",
            "reason": "The locations are visibly different on expanded evidence.",
            "parent_plan_revision": "r0003",
            "config_fingerprint": third.data["config"]["config_fingerprint"],
        }],
    })
    assert resolved.success is True, resolved.error
    assert resolved.data["plan_status"] == "ready"
    assert resolved.data["review_status"] == "accepted"


def test_output_path_cannot_escape_project_workspace(tmp_path: Path, monkeypatch) -> None:
    projects = tmp_path / "projects"
    project = projects / "safe"
    project.mkdir(parents=True)
    monkeypatch.setattr(tool_module, "PROJECTS_DIR", projects)

    result = ReplicationPreprocess().execute({
        "operation": "plan",
        "project_id": "safe",
        "source_path": str(tmp_path / "source.mp4"),
        "output_path": str(tmp_path / "outside.json"),
    })

    assert result.success is False
    assert result.data["plan_status"] == "blocked"
    assert "under project workspace" in (result.error or "")
