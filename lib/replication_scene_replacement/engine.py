"""Compatibility import for the sole supported direct-generation engine."""

from .engine_v2 import SceneReplacementV2Engine


SceneReplacementEngine = SceneReplacementV2Engine

__all__ = ["SceneReplacementEngine", "SceneReplacementV2Engine"]
