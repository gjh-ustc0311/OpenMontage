"""Deterministic preprocessing primitives for reference-video replication.

The package deliberately contains no LLM or model integration.  PySceneDetect
produces boundary evidence, and the calling agent supplies reviewed boundary
relationships as ordinary, versioned input data.
"""

from .planner import build_generation_plan, build_scene_groups
from .review import ReviewValidationError, apply_review_submission

__all__ = [
    "ReviewValidationError",
    "apply_review_submission",
    "build_generation_plan",
    "build_scene_groups",
]
