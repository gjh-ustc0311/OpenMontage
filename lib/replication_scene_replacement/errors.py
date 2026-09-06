"""Domain errors raised by the scene-replacement persistence layer."""


class SceneReplacementError(RuntimeError):
    """A request violated the immutable scene-replacement contract."""

