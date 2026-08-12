from __future__ import annotations

from .base import PersonDetector


class HailoAdapterNotImplemented(NotImplementedError):
    pass


class HailoPersonDetector(PersonDetector):
    """Deliberately incomplete until the target HailoRT/HEF output API is verified on hardware."""

    def __init__(self, hef_path: str, conf_threshold: float = 0.4) -> None:
        raise HailoAdapterNotImplemented(
            "Hailo detector is an unverified adapter. Validate the installed HailoRT API, target chip, "
            "HEF input/output tensors, and post-processing before implementing it."
        )

    def detect(self, frame):  # pragma: no cover
        raise HailoAdapterNotImplemented
