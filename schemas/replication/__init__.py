"""Domain schema helpers for replication workflows.

Phase 2 schemas use relative references but must never resolve them over the
network.  ``phase2_schema_registry`` registers every local Phase 2 ``$id`` so
callers can validate the public tool and persisted-document contracts offline.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from referencing import Registry, Resource


SCHEMA_DIR = Path(__file__).resolve().parent
TOOLS_SCHEMA_DIR = SCHEMA_DIR.parent / "tools"

PHASE2_SCHEMA_PATHS = (
    SCHEMA_DIR / "scene_replacement_common.schema.json",
    SCHEMA_DIR / "scene_replacement_source_snapshot.schema.json",
    SCHEMA_DIR / "scene_replacement_edit_result_submission.schema.json",
    SCHEMA_DIR / "scene_replacement_v2_plan.schema.json",
    SCHEMA_DIR / "scene_replacement_v2_edit_request.schema.json",
    SCHEMA_DIR / "scene_replacement_v2_review_submission.schema.json",
    SCHEMA_DIR / "scene_replacement_v2_state.schema.json",
    SCHEMA_DIR / "scene_replacement_v2_delivery_manifest.schema.json",
    TOOLS_SCHEMA_DIR / "replication_scene_replacement_v2.schema.json",
)


def _schema_name(path: Path) -> str:
    return path.name.removesuffix(".schema.json")


@lru_cache(maxsize=None)
def _load_path(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_phase2_schema(name: str) -> dict[str, Any]:
    """Load a Phase 2 schema by its filename stem."""

    matches = [path for path in PHASE2_SCHEMA_PATHS if _schema_name(path) == name]
    if not matches:
        raise FileNotFoundError(f"Unknown Phase 2 schema: {name}")
    return _load_path(matches[0])


@lru_cache(maxsize=1)
def phase2_schema_registry() -> Registry:
    """Return an offline registry for every local Phase 2 schema ``$id``."""

    registry = Registry()
    for path in PHASE2_SCHEMA_PATHS:
        schema = _load_path(path)
        Draft202012Validator.check_schema(schema)
        schema_id = schema.get("$id")
        if not isinstance(schema_id, str) or not schema_id:
            raise ValueError(f"Phase 2 schema has no $id: {path}")
        registry = registry.with_resource(schema_id, Resource.from_contents(schema))
    return registry


def validate_phase2_document(name: str, document: dict[str, Any]) -> None:
    """Validate a Phase 2 document using local references only."""

    schema = load_phase2_schema(name)
    Draft202012Validator(schema, registry=phase2_schema_registry()).validate(document)


__all__ = [
    "PHASE2_SCHEMA_PATHS",
    "load_phase2_schema",
    "phase2_schema_registry",
    "validate_phase2_document",
]
