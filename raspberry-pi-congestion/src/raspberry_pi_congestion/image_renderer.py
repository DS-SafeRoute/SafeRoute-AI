from __future__ import annotations

from typing import Sequence

from .models import Detection, Point


class OpenCvDetectionRenderer:
    """원본 프레임을 변경하지 않고 검출 결과가 그려진 복사본을 만든다."""

    def __init__(self, roi: Sequence[Point] = ()) -> None:
        # roi 인자는 기존 호출부 호환을 위해 받지만 더 이상 표시하지 않는다.
        self.roi = []

    def render(self, frame, detections: Sequence[Detection],
               inside_detections: Sequence[Detection]):
        import cv2
        rendered = frame.copy()
        height, width = rendered.shape[:2]

        for detection in detections:
            color = (0, 255, 0)
            x1 = min(width - 1, max(0, round(detection.x1)))
            y1 = min(height - 1, max(0, round(detection.y1)))
            x2 = min(width - 1, max(0, round(detection.x2)))
            y2 = min(height - 1, max(0, round(detection.y2)))
            cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                rendered,
                f"person {detection.confidence:.2f}",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            rendered,
            f"headcount: {len(detections)}",
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return rendered
