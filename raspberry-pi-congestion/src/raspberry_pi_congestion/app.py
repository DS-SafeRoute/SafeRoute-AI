from __future__ import annotations

import logging
import time
import uuid
from typing import Callable, Optional

from .api_client import CongestionReporter
from .models import CongestionObservation
from .offline_queue import OfflineQueue
from .roi_counter import RoiCounter
from .window_aggregator import WindowAggregator

logger = logging.getLogger(__name__)


class CongestionPipeline:
    def __init__(self, video_source, detector, roi_counter: RoiCounter,
                 aggregator: WindowAggregator, reporter: CongestionReporter,
                 cctv_code: str, offline_queue: Optional[OfflineQueue] = None,
                 target_fps: float = 5.0, flush_interval_sec: float = 30.0,
                 monotonic: Callable[[], float] = time.monotonic,
                 epoch_ms: Callable[[], int] = lambda: int(time.time() * 1000)) -> None:
        self.video_source = video_source
        self.detector = detector
        self.roi_counter = roi_counter
        self.aggregator = aggregator
        self.reporter = reporter
        self.cctv_code = cctv_code
        self.offline_queue = offline_queue
        self.target_fps = target_fps
        self.flush_interval_sec = flush_interval_sec
        self._monotonic = monotonic
        self._epoch_ms = epoch_ms
        self._last_inference = float("-inf")
        self._last_queue_flush = monotonic()
        self._last_inference_error_log = float("-inf")
        self._closed = False

    def run(self) -> None:
        try:
            for frame in self.video_source.frames():
                now = self._monotonic()
                if self.target_fps > 0 and now - self._last_inference < 1.0 / self.target_fps:
                    continue
                self._last_inference = now
                self.process_frame(frame)
                self._maybe_flush_queue(now)
        finally:
            self.close()

    def process_frame(self, frame) -> Optional[CongestionObservation]:
        try:
            detections = self.detector.detect(frame)
        except Exception as exc:
            now = self._monotonic()
            if now - self._last_inference_error_log >= 10.0:
                self._last_inference_error_log = now
                logger.error("Person inference failed (further errors suppressed for 10s): %s", type(exc).__name__)
            return None
        height, width = frame.shape[:2]
        count = self.roi_counter.count_inside(detections, width, height)
        now_ms = self._epoch_ms()
        self.aggregator.add_sample(count, now_ms)
        if not self.aggregator.should_flush():
            return None
        summary = self.aggregator.flush(now_ms)
        if summary is None:
            return None
        observation = CongestionObservation.from_summary(str(uuid.uuid4()), self.cctv_code, summary)
        if not self.reporter.report(observation) and self.offline_queue is not None:
            self.offline_queue.enqueue(observation.event_id, observation.to_json())
        return observation

    def _maybe_flush_queue(self, now: float) -> None:
        if self.offline_queue is None or now - self._last_queue_flush < self.flush_interval_sec:
            return
        self._last_queue_flush = now
        for item in self.offline_queue.peek_oldest(limit=5):
            if self.reporter.report_json(item.payload):
                self.offline_queue.mark_success(item.id)
            else:
                self.offline_queue.mark_failed_attempt(item.id)
                break

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.video_source.close()
        self.detector.close()
        if self.offline_queue is not None:
            self.offline_queue.close()
