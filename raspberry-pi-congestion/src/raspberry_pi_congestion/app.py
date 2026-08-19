from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from .api_client import CongestionReporter
from .event_detector import CongestionEventDetector
from .models import (
    CongestionEvent, CongestionObservation, CongestionThresholds,
    DeviceCongestionConfig, EventDetectionSettings,
)
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
                 epoch_ms: Callable[[], int] = lambda: int(time.time() * 1000),
                 config_provider=None, config_poll_active_sec: float = 5.0,
                 config_poll_inactive_sec: float = 15.0) -> None:
        self.video_source = video_source
        self.detector = detector
        self.roi_counter = roi_counter
        self.aggregator = aggregator
        self.reporter = reporter
        self.config_provider = config_provider
        self.cctv_code = cctv_code
        self.offline_queue = offline_queue
        self.target_fps = target_fps
        self.flush_interval_sec = flush_interval_sec
        self.config_poll_active_sec = config_poll_active_sec
        self.config_poll_inactive_sec = config_poll_inactive_sec
        self._monotonic = monotonic
        self._epoch_ms = epoch_ms
        self._last_inference = float("-inf")
        self._last_queue_flush = monotonic()
        self._last_config_poll = float("-inf")
        self._last_inference_error_log = float("-inf")
        self._closed = False
        self._event_detector = CongestionEventDetector()
        # Only local/test pipelines use this fallback. Server modes always poll BE.
        self._config = None if config_provider else DeviceCongestionConfig(
            True, "local-dry-run", cctv_code, 1, 1.0, aggregator.window_sec,
            target_fps, CongestionThresholds(2.0, 3.0, 5.0), EventDetectionSettings(3, 5, 30),
        )

    def run(self) -> None:
        try:
            for frame in self.video_source.frames():
                now = self._monotonic()
                self._maybe_refresh_config(now)
                if self._config is None or not self._config.training_active:
                    continue
                if self.target_fps > 0 and now - self._last_inference < 1.0 / self.target_fps:
                    continue
                self._last_inference = now
                self.process_frame(frame)
                self._maybe_flush_queue(now)
        finally:
            self.close()

    def process_frame(self, frame) -> Optional[CongestionObservation]:
        config = self._config
        if config is None or not config.training_active or not config.training_session_id:
            return None
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
        self._process_local_event(frame, count, now_ms, config)
        self.aggregator.add_sample(count, now_ms)
        if not self.aggregator.should_flush():
            return None
        summary = self.aggregator.flush(now_ms)
        if summary is None:
            return None
        event_id = str(uuid.uuid4())
        image_key = self._upload_snapshot(frame, "MONITORING", event_id, now_ms, config)
        observation = CongestionObservation.from_summary(
            event_id, config.training_session_id, self.cctv_code,
            config.config_version, summary, image_key,
        )
        reported = self.reporter.report(observation)
        retryable = not hasattr(self.reporter, "should_queue_failure") or self.reporter.should_queue_failure(observation.event_id)
        if not reported and retryable and self.offline_queue is not None:
            self.offline_queue.enqueue(observation.event_id, observation.to_json(), "observation")
        return observation

    def _process_local_event(self, frame, count: int, now_ms: int,
                             config: DeviceCongestionConfig) -> None:
        if config.monitored_area_m2 is None or config.thresholds is None or config.event_detection is None:
            return
        density = count / config.monitored_area_m2
        level = config.thresholds.classify(density)
        event_type = self._event_detector.observe(level, now_ms, config.event_detection)
        if event_type is None or not config.training_session_id:
            return
        event = CongestionEvent(
            str(uuid.uuid4()), config.training_session_id, self.cctv_code,
            event_type, now_ms, count, density, level, config.config_version,
        )
        client = self.config_provider
        if client is None or not hasattr(client, "report_event"):
            logger.info("congestion event: %s", event.to_json())
            return

        with ThreadPoolExecutor(max_workers=2) as executor:
            report_future = executor.submit(client.report_event, event)
            upload_future = executor.submit(
                self._upload_snapshot, frame, "CONGESTION_EVENT", event.event_id, now_ms, config
            )
            reported = report_future.result()
            image_key = upload_future.result()
        retryable = not hasattr(client, "should_queue_failure") or client.should_queue_failure(event.event_id)
        if not reported and retryable and self.offline_queue is not None:
            self.offline_queue.enqueue(event.event_id, event.to_json(), "event")
        if reported and image_key:
            attached = client.attach_event_image(event.event_id, image_key, self._epoch_ms())
            retryable_image = not hasattr(client, "should_queue_failure") or client.should_queue_failure(f"image:{event.event_id}")
            if not attached and retryable_image and self.offline_queue is not None:
                payload = {"eventId": event.event_id, "eventImageKey": image_key, "uploadedAt": self._epoch_ms()}
                self.offline_queue.enqueue(f"image:{event.event_id}", payload, "event_image")

    def _upload_snapshot(self, frame, image_type: str, reference_id: str,
                         captured_at: int, config: DeviceCongestionConfig) -> Optional[str]:
        client = self.config_provider
        if client is None or not hasattr(client, "request_image_upload") or not config.training_session_id:
            return None
        try:
            import cv2
            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                return None
            target = client.request_image_upload(
                training_session_id=config.training_session_id,
                cctv_code=self.cctv_code,
                image_type=image_type,
                reference_id=reference_id,
                captured_at=captured_at,
            )
            if target is None or target["expiresAt"] <= self._epoch_ms():
                return None
            return target["objectKey"] if client.upload_jpeg(target["uploadUrl"], encoded.tobytes()) else None
        except Exception as exc:
            logger.warning("Snapshot processing failed: %s", type(exc).__name__)
            return None

    def _maybe_refresh_config(self, now: float) -> None:
        if self.config_provider is None:
            return
        forced = (hasattr(self.config_provider, "consume_config_refresh_request")
                  and self.config_provider.consume_config_refresh_request())
        interval = self.config_poll_active_sec if self._config and self._config.training_active else self.config_poll_inactive_sec
        if not forced and now - self._last_config_poll < interval:
            return
        self._last_config_poll = now
        new_config = self.config_provider.fetch_config(self.cctv_code)
        if new_config is None:
            return
        previous = self._config
        self._config = new_config
        session_changed = previous is None or previous.training_session_id != new_config.training_session_id
        if (previous is None or previous.config_version != new_config.config_version
                or previous.training_active != new_config.training_active or session_changed):
            if new_config.training_active:
                self.aggregator.reconfigure(new_config.snapshot_interval_sec)
                self.target_fps = new_config.target_inference_fps
            else:
                self.aggregator.reconfigure(previous.snapshot_interval_sec if previous else 5.0)
            logger.info("Applied congestion config version %d", new_config.config_version)
        if session_changed or not new_config.training_active:
            self._event_detector.reset()

    def _maybe_flush_queue(self, now: float) -> None:
        if self.offline_queue is None or now - self._last_queue_flush < self.flush_interval_sec:
            return
        self._last_queue_flush = now
        for item in self.offline_queue.peek_oldest(limit=5):
            if item.operation == "event" and hasattr(self.reporter, "report_event_json"):
                success = self.reporter.report_event_json(item.payload)
            elif item.operation == "event_image" and hasattr(self.reporter, "attach_event_image"):
                success = self.reporter.attach_event_image(item.payload["eventId"], item.payload["eventImageKey"], item.payload["uploadedAt"])
            else:
                success = self.reporter.report_json(item.payload)
            if success:
                self.offline_queue.mark_success(item.id)
            else:
                retryable = not hasattr(self.reporter, "should_queue_failure") or self.reporter.should_queue_failure(item.event_id)
                if not retryable:
                    logger.error("Dropping terminally rejected queued %s %s", item.operation, item.event_id)
                    self.offline_queue.mark_success(item.id)
                    continue
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
