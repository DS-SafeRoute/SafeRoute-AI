from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.x <= 1.0 and 0.0 <= self.y <= 1.0):
            raise ValueError("ROI coordinates must be normalized to [0, 1]")


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0
    class_name: str = "person"

    def bottom_center(self, frame_width: int, frame_height: int) -> Point:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        return Point(
            max(0.0, min(1.0, ((self.x1 + self.x2) / 2.0) / frame_width)),
            max(0.0, min(1.0, self.y2 / frame_height)),
        )


@dataclass(frozen=True)
class WindowSummary:
    window_start_ms: int
    window_end_ms: int
    sample_count: int
    avg_headcount: int
    peak_headcount: int


@dataclass(frozen=True)
class CongestionObservation:
    event_id: str
    cctv_code: str
    avg_headcount: int
    peak_headcount: int
    sample_count: int
    window_start: int
    window_end: int
    captured_at: int
    s3_image_key: Optional[str] = None

    @classmethod
    def from_summary(cls, event_id: str, cctv_code: str, summary: WindowSummary) -> "CongestionObservation":
        return cls(
            event_id=event_id,
            cctv_code=cctv_code,
            avg_headcount=summary.avg_headcount,
            peak_headcount=summary.peak_headcount,
            sample_count=summary.sample_count,
            window_start=summary.window_start_ms,
            window_end=summary.window_end_ms,
            captured_at=summary.window_end_ms,
        )

    def to_json(self) -> dict:
        return {
            "eventId": self.event_id,
            "cctvCode": self.cctv_code,
            "avgHeadcount": self.avg_headcount,
            "peakHeadcount": self.peak_headcount,
            "sampleCount": self.sample_count,
            "windowStart": self.window_start,
            "windowEnd": self.window_end,
            "capturedAt": self.captured_at,
            "s3ImageKey": self.s3_image_key,
        }
