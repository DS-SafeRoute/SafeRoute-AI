from __future__ import annotations

from typing import Sequence

from .models import Detection, Point


class OpenCvPreview:
    """개발 PC에서만 사용하는 추론 미리보기 창."""

    def __init__(self, roi: Sequence[Point], window_name: str = "SafeRoute Congestion Preview") -> None:
        self.roi = list(roi)
        self.window_name = window_name
        self._cv2 = None
        self._window_opened = False

    def show(
            self,
            frame,
            detections: Sequence[Detection],
            inside_detections: Sequence[Detection],
    ) -> bool:
        cv2 = self._get_cv2()
        preview = frame.copy()
        height, width = preview.shape[:2]
        inside_ids = {id(detection) for detection in inside_detections}

        roi_points = [
            [
                min(width - 1, max(0, round(point.x * width))),
                min(height - 1, max(0, round(point.y * height))),
            ]
            for point in self.roi
        ]
        import numpy as np
        cv2.polylines(
            preview,
            [np.asarray(roi_points, dtype=np.int32)],
            True,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        for detection in detections:
            is_inside = id(detection) in inside_ids
            color = (0, 255, 0) if is_inside else (0, 128, 255)
            x1 = min(width - 1, max(0, round(detection.x1)))
            y1 = min(height - 1, max(0, round(detection.y1)))
            x2 = min(width - 1, max(0, round(detection.x2)))
            y2 = min(height - 1, max(0, round(detection.y2)))
            cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
            bottom_x = round((x1 + x2) / 2)
            cv2.circle(preview, (bottom_x, y2), 4, color, -1)
            cv2.putText(
                preview,
                f"person {detection.confidence:.2f}",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            preview,
            f"ROI headcount: {len(inside_detections)} / detected: {len(detections)}",
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            "Q or ESC: quit",
            (16, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(self.window_name, preview)
        self._window_opened = True
        key = cv2.waitKey(1) & 0xFF
        return key not in (ord("q"), ord("Q"), 27)

    def close(self) -> None:
        if self._cv2 is None or not self._window_opened:
            return
        try:
            self._cv2.destroyWindow(self.window_name)
        except Exception:
            self._cv2.destroyAllWindows()
        finally:
            self._window_opened = False

    def _get_cv2(self):
        if self._cv2 is None:
            import cv2
            self._cv2 = cv2
        return self._cv2
