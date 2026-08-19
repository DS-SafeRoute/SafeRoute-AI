from __future__ import annotations

from typing import Optional

from .models import CongestionLevel, EventDetectionSettings


class CongestionEventDetector:
    """Turns per-frame local levels into the three BE event types."""

    def __init__(self) -> None:
        self.current_level = CongestionLevel.NORMAL
        self._candidate: Optional[CongestionLevel] = None
        self._candidate_count = 0
        self._normal_count = 0
        self._last_started_at_ms: Optional[int] = None

    def reset(self) -> None:
        self.__init__()

    def observe(self, level: CongestionLevel, now_ms: int,
                settings: EventDetectionSettings) -> Optional[str]:
        if level == CongestionLevel.NORMAL:
            self._candidate = None
            self._candidate_count = 0
            if self.current_level == CongestionLevel.NORMAL:
                self._normal_count = 0
                return None
            self._normal_count += 1
            if self._normal_count >= settings.recovery_consecutive_frames:
                self.current_level = CongestionLevel.NORMAL
                self._normal_count = 0
                return "CONGESTION_ENDED"
            return None

        self._normal_count = 0
        if level.rank <= self.current_level.rank:
            self._candidate = None
            self._candidate_count = 0
            return None
        if self._candidate != level:
            self._candidate = level
            self._candidate_count = 1
        else:
            self._candidate_count += 1
        if self._candidate_count < settings.required_consecutive_frames:
            return None

        previous = self.current_level
        self._candidate = None
        self._candidate_count = 0
        if previous != CongestionLevel.NORMAL:
            self.current_level = level
            return "CONGESTION_LEVEL_UP"
        cooldown_ms = int(settings.cooldown_sec * 1000)
        if self._last_started_at_ms is not None and now_ms - self._last_started_at_ms < cooldown_ms:
            return None
        self.current_level = level
        self._last_started_at_ms = now_ms
        return "CONGESTION_STARTED"
