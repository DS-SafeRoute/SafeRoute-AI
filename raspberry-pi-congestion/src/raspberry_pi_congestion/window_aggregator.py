from __future__ import annotations

import time
from typing import Callable, Optional

from .models import WindowSummary


class WindowAggregator:
    def __init__(self, window_sec: float = 5.0, clock: Callable[[], float] = time.monotonic) -> None:
        if window_sec <= 0:
            raise ValueError("window_sec must be positive")
        self.window_sec = window_sec
        self._clock = clock
        self._started_monotonic = clock()
        self._window_start_ms: Optional[int] = None
        self._counts: list[int] = []

    def add_sample(self, person_count: int, now_ms: int) -> None:
        if person_count < 0:
            raise ValueError("person_count must not be negative")
        if self._window_start_ms is None:
            self._window_start_ms = now_ms
        self._counts.append(person_count)

    def should_flush(self) -> bool:
        return self._clock() - self._started_monotonic >= self.window_sec

    def flush(self, now_ms: int) -> Optional[WindowSummary]:
        if not self._counts:
            self._reset()
            return None
        peak = max(self._counts)
        average = sum(self._counts) / len(self._counts)
        summary = WindowSummary(
            window_start_ms=self._window_start_ms if self._window_start_ms is not None else now_ms,
            window_end_ms=now_ms,
            sample_count=len(self._counts),
            avg_headcount=min(float(average), peak),
            peak_headcount=peak,
        )
        self._reset()
        return summary

    def _reset(self) -> None:
        self._counts.clear()
        self._window_start_ms = None
        self._started_monotonic = self._clock()

    def reconfigure(self, window_sec: float) -> None:
        if window_sec <= 0:
            raise ValueError("window_sec must be positive")
        self.window_sec = window_sec
        self._reset()
