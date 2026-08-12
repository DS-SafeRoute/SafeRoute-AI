from __future__ import annotations

from typing import Sequence

from .models import Detection, Point


class RoiCounter:
    def __init__(self, roi: Sequence[Point]) -> None:
        if len(roi) < 3:
            raise ValueError("ROI requires at least three points")
        self.roi = list(roi)

    def count_inside(self, detections: Sequence[Detection], frame_width: int, frame_height: int) -> int:
        return len(self.filter_inside(detections, frame_width, frame_height))

    def filter_inside(self, detections: Sequence[Detection], frame_width: int, frame_height: int) -> list[Detection]:
        return [d for d in detections if self._point_in_polygon(d.bottom_center(frame_width, frame_height), self.roi)]

    @staticmethod
    def _point_in_polygon(point: Point, polygon: Sequence[Point], epsilon: float = 1e-9) -> bool:
        x, y = point.x, point.y
        inside = False
        previous = polygon[-1]
        for current in polygon:
            cross = (x - previous.x) * (current.y - previous.y) - (y - previous.y) * (current.x - previous.x)
            if abs(cross) <= epsilon and min(previous.x, current.x) - epsilon <= x <= max(previous.x, current.x) + epsilon and min(previous.y, current.y) - epsilon <= y <= max(previous.y, current.y) + epsilon:
                return True
            if (current.y > y) != (previous.y > y):
                intersection_x = (previous.x - current.x) * (y - current.y) / (previous.y - current.y) + current.x
                if x < intersection_x:
                    inside = not inside
            previous = current
        return inside
