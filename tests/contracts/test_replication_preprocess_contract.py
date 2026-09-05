from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.replication_preprocess.models import time_point
from lib.replication_preprocess.review import build_review_request
from tools.analysis.replication_preprocess import ReplicationPreprocess
from tools.tool_registry import ToolRegistry


ROOT = Path(__file__).resolve().parents[2]
jsonschema = pytest.importorskip("jsonschema")


def _load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_tool_contract_is_cpu_only_and_registry_discoverable() -> None:
    tool = ReplicationPreprocess()
    dependency_text = " ".join(tool.dependencies).lower()

    assert tool.resource_profile.vram_mb == 0
    assert "torch" not in dependency_text
    assert "transformers" not in dependency_text
    assert "clip" not in dependency_text

    registry = ToolRegistry()
    registry.discover("tools.analysis")
    assert registry.get("replication_preprocess") is not None


def test_replication_json_schemas_are_valid() -> None:
    schemas = [
        _load("schemas/tools/replication_preprocess.schema.json"),
        _load("schemas/replication/boundary_review_request.schema.json"),
        _load("schemas/replication/boundary_review_submission.schema.json"),
    ]
    for schema in schemas:
        jsonschema.Draft202012Validator.check_schema(schema)


def test_generated_review_request_matches_schema() -> None:
    item = {
        "review_item_id": "bri_" + "1" * 16,
        "boundary_id": "bnd_" + "2" * 16,
        "left_segment_id": "seg_left",
        "right_segment_id": "seg_right",
        "time": time_point(100, __import__("fractions").Fraction(1, 100)),
        "content_val": 37.5,
        "accepted_threshold": 30,
        "components": {"delta_lum": 10.0},
        "evidence": {
            role: {"path": f"evidence/{role}.jpg", "sha256": "3" * 64}
            for role in (
                "left_keyframe", "left_context", "left_edge", "right_edge",
                "right_context", "right_keyframe", "evidence_board",
            )
        },
    }
    request = build_review_request(
        source_sha256="4" * 64,
        parent_plan_revision="r0001",
        config_fingerprint="5" * 64,
        protocol_version="boundary-visual-review-v1",
        review_round=1,
        items=[item],
    )

    jsonschema.Draft202012Validator(
        _load("schemas/replication/boundary_review_request.schema.json")
    ).validate(request)
