from __future__ import annotations

import base64
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

from .api_client import ImageUploadResult
from .models import CongestionEvent, CongestionObservation, WindowSummary

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Snapshot:
    frame: object
    detections: tuple
    inside_detections: tuple


@dataclass(frozen=True)
class MonitoringDelivery:
    event_id: str
    training_session_id: str
    cctv_code: str
    config_version: int
    summary: WindowSummary
    snapshot: Snapshot


@dataclass(frozen=True)
class EventDelivery:
    event: CongestionEvent
    snapshot: Snapshot


class DeliveryQueue:
    """Bounded, event-first network delivery worker.

    Frames remain unencoded until the worker handles them. When capacity is
    exhausted, the oldest monitoring snapshot is discarded before an event.
    """

    def __init__(self, client, renderer, offline_queue=None, max_items: int = 32,
                 shutdown_timeout_sec: float = 5.0,
                 max_presigned_refreshes: int = 1,
                 epoch_ms: Callable[[], int] = lambda: int(time.time() * 1000)) -> None:
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        if shutdown_timeout_sec < 0:
            raise ValueError("shutdown_timeout_sec must not be negative")
        self.client = client
        self.renderer = renderer
        self.offline_queue = offline_queue
        self.max_items = max_items
        self.shutdown_timeout_sec = shutdown_timeout_sec
        self.max_presigned_refreshes = max_presigned_refreshes
        self._epoch_ms = epoch_ms
        self._events: deque[EventDelivery] = deque()
        self._monitoring: deque[MonitoringDelivery] = deque()
        self._condition = threading.Condition()
        self._active_session_id: Optional[str] = None
        self._accepting = True
        self._stop = False
        self._busy = False
        self._current_job = None
        self._flush_requested = False
        self._thread = threading.Thread(
            target=self._run, name="congestion-delivery", daemon=True
        )
        self._thread.start()

    def set_session(self, session_id: Optional[str]) -> int:
        with self._condition:
            if session_id == self._active_session_id:
                return 0
            self._active_session_id = session_id
            discarded = len(self._events) + len(self._monitoring)
            self._events.clear()
            self._monitoring.clear()
            self._condition.notify_all()
            return discarded

    def submit_monitoring(self, job: MonitoringDelivery) -> bool:
        with self._condition:
            if not self._can_accept(job.training_session_id):
                return False
            if self._size() >= self.max_items:
                if self._monitoring:
                    self._monitoring.popleft()
                    logger.warning("Dropped oldest monitoring snapshot from full delivery queue")
                else:
                    logger.warning("Dropped monitoring snapshot because delivery queue contains only events")
                    return False
            self._monitoring.append(job)
            self._condition.notify()
            return True

    def submit_event(self, job: EventDelivery) -> bool:
        session_id = job.event.training_session_id
        with self._condition:
            if not self._can_accept(session_id):
                return False
            if self._size() >= self.max_items:
                if self._monitoring:
                    self._monitoring.popleft()
                    logger.warning("Dropped oldest monitoring snapshot to preserve congestion event")
                else:
                    logger.error("Delivery queue is saturated with congestion events")
                    return False
            self._events.append(job)
            self._condition.notify()
            return True

    def request_offline_flush(self) -> None:
        with self._condition:
            self._flush_requested = True
            self._condition.notify()

    def wait_idle(self, timeout_sec: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout_sec
        with self._condition:
            while (self._size() or self._busy or self._flush_requested) and time.monotonic() < deadline:
                self._condition.wait(max(0.0, deadline - time.monotonic()))
            return not self._size() and not self._busy

    def close(self) -> None:
        with self._condition:
            if not self._accepting:
                return
            self._accepting = False
            self._condition.notify_all()
        self.wait_idle(self.shutdown_timeout_sec)
        with self._condition:
            pending = [*self._events, *self._monitoring]
            if self._current_job is not None:
                pending.insert(0, self._current_job)
            self._events.clear()
            self._monitoring.clear()
            self._stop = True
            self._condition.notify_all()
        for job in pending:
            self._persist_pending(job)
        self._thread.join(timeout=0.1)

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    @property
    def pending_count(self) -> int:
        with self._condition:
            return self._size()

    def _can_accept(self, session_id: str) -> bool:
        return (self._accepting and bool(session_id)
                and session_id == self._active_session_id)

    def _size(self) -> int:
        return len(self._events) + len(self._monitoring)

    def _run(self) -> None:
        while True:
            job = None
            with self._condition:
                while not self._stop and not self._size() and not self._flush_requested:
                    self._condition.wait()
                if self._stop:
                    return
                if self._events:
                    job = self._events.popleft()
                elif self._flush_requested:
                    self._flush_requested = False
                else:
                    job = self._monitoring.popleft()
                self._busy = True
                self._current_job = job
            try:
                if job is None:
                    self._flush_offline()
                elif self._job_session(job) == self._active_session_id:
                    if isinstance(job, EventDelivery):
                        self._deliver_event(job)
                    else:
                        self._deliver_monitoring(job)
            except Exception as exc:
                logger.exception("Background delivery failed: %s", type(exc).__name__)
                if job is not None:
                    self._persist_pending(job)
            finally:
                with self._condition:
                    self._busy = False
                    self._current_job = None
                    self._condition.notify_all()

    def _flush_offline(self) -> None:
        if self.offline_queue is None or not self._active_session_id:
            return
        for item in self.offline_queue.peek_oldest(
                limit=5, training_session_id=self._active_session_id):
            payload = item.payload
            success = False
            if item.operation == "event" and hasattr(self.client, "report_event_json"):
                event_payload = payload.get("eventPayload", payload)
                success = self.client.report_event_json(event_payload)
                if success and self._session_active(item.training_session_id):
                    image_key = payload.get("eventImageKey")
                    jpeg = self._decode_jpeg(payload)
                    if not image_key and jpeg is not None:
                        image_key = self._upload(
                            jpeg, item.training_session_id, event_payload["cctvCode"],
                            "CONGESTION_EVENT", item.event_id, event_payload["detectedAt"],
                        )
                        if image_key is None:
                            self.offline_queue.mark_failed_attempt(item.id)
                            break
                    if image_key:
                        attached = self.client.attach_event_image(
                            item.event_id, image_key,
                            int(payload.get("uploadedAt", self._epoch_ms())),
                        )
                        if not attached and self._retryable(f"image:{item.event_id}"):
                            self._enqueue_offline(
                                f"image:{item.event_id}",
                                {"eventId": item.event_id, "eventImageKey": image_key,
                                 "uploadedAt": int(payload.get("uploadedAt", self._epoch_ms()))},
                                "event_image", item.training_session_id,
                            )
                        # POST completed. A failed PATCH now has its own retry item.
                        success = True
            elif item.operation == "event_image" and hasattr(self.client, "attach_event_image"):
                success = self.client.attach_event_image(
                    payload["eventId"], payload["eventImageKey"], payload["uploadedAt"]
                )
            elif item.operation == "pending_observation":
                observation_payload = dict(payload["observationPayload"])
                jpeg = self._decode_jpeg(payload)
                if jpeg is not None:
                    observation_payload["monitoringImageKey"] = self._upload(
                        jpeg, item.training_session_id, observation_payload["cctvCode"],
                        "MONITORING", item.event_id, observation_payload["capturedAt"],
                    )
                success = self.client.report_json(observation_payload)
            else:
                success = self.client.report_json(payload)
            if success:
                self.offline_queue.mark_success(item.id)
                continue
            if not self._retryable(item.event_id):
                logger.error("Dropping terminally rejected queued %s %s", item.operation, item.event_id)
                self.offline_queue.mark_success(item.id)
                continue
            self.offline_queue.mark_failed_attempt(item.id)
            break

    def _deliver_monitoring(self, job: MonitoringDelivery) -> None:
        rendered = self.renderer.render(
            job.snapshot.frame, job.snapshot.detections, job.snapshot.inside_detections
        )
        jpeg = self._encode(rendered)
        image_key = self._upload(
            jpeg, job.training_session_id, job.cctv_code, "MONITORING",
            job.event_id, job.summary.captured_at_ms,
        ) if jpeg is not None else None
        if not self._session_active(job.training_session_id):
            return
        observation = CongestionObservation.from_summary(
            job.event_id, job.training_session_id, job.cctv_code,
            job.config_version, job.summary, image_key,
        )
        if not self.client.report(observation) and self._retryable(job.event_id):
            self._enqueue_offline(
                job.event_id, observation.to_json(), "observation", job.training_session_id
            )

    def _deliver_event(self, job: EventDelivery) -> None:
        event = job.event
        if not self.client.report_event(event):
            if self._retryable(event.event_id):
                self._persist_pending(job)
            return
        if not self._session_active(event.training_session_id):
            return
        rendered = self.renderer.render(
            job.snapshot.frame, job.snapshot.detections, job.snapshot.inside_detections
        )
        jpeg = self._encode(rendered)
        image_key = self._upload(
            jpeg, event.training_session_id, event.cctv_code, "CONGESTION_EVENT",
            event.event_id, event.detected_at,
        ) if jpeg is not None else None
        if image_key is None and jpeg is not None and self._session_active(event.training_session_id):
            self._persist_pending(job)
            return
        if image_key and self._session_active(event.training_session_id):
            uploaded_at = self._epoch_ms()
            if (not self.client.attach_event_image(event.event_id, image_key, uploaded_at)
                    and self._retryable(f"image:{event.event_id}")):
                self._enqueue_offline(
                    f"image:{event.event_id}",
                    {"eventId": event.event_id, "eventImageKey": image_key,
                     "uploadedAt": uploaded_at},
                    "event_image", event.training_session_id,
                )

    def _upload(self, jpeg: bytes, session_id: str, cctv_code: str,
                image_type: str, reference_id: str, captured_at: int) -> Optional[str]:
        if not hasattr(self.client, "request_image_upload"):
            return None
        for _ in range(self.max_presigned_refreshes + 1):
            if not self._session_active(session_id):
                return None
            target = self.client.request_image_upload(
                training_session_id=session_id, cctv_code=cctv_code,
                image_type=image_type, reference_id=reference_id,
                captured_at=captured_at,
            )
            if target is None:
                return None
            if target["expiresAt"] <= self._epoch_ms():
                continue
            if not self._session_active(session_id):
                return None
            result = self.client.upload_jpeg(target["uploadUrl"], jpeg)
            if result is True or result == ImageUploadResult.SUCCESS:
                return target["objectKey"]
            if result != ImageUploadResult.EXPIRED:
                return None
        return None

    def _persist_pending(self, job) -> None:
        if self.offline_queue is None:
            return
        if isinstance(job, EventDelivery):
            rendered = self.renderer.render(
                job.snapshot.frame, job.snapshot.detections, job.snapshot.inside_detections
            )
            jpeg = self._encode(rendered)
            payload = {"eventPayload": job.event.to_json()}
            operation = "event"
            event_id = job.event.event_id
            session_id = job.event.training_session_id
        else:
            # Keep the stable observation payload and its encoded snapshot together.
            observation = CongestionObservation.from_summary(
                job.event_id, job.training_session_id, job.cctv_code,
                job.config_version, job.summary,
            )
            rendered = self.renderer.render(
                job.snapshot.frame, job.snapshot.detections, job.snapshot.inside_detections
            )
            jpeg = self._encode(rendered)
            payload = {"observationPayload": observation.to_json()}
            operation = "pending_observation"
            event_id = job.event_id
            session_id = job.training_session_id
        if jpeg is not None:
            payload["jpegBase64"] = base64.b64encode(jpeg).decode("ascii")
        self._enqueue_offline(event_id, payload, operation, session_id)

    def _enqueue_offline(self, event_id: str, payload: dict, operation: str,
                         session_id: str) -> None:
        if self.offline_queue is not None:
            self.offline_queue.enqueue(event_id, payload, operation, session_id)

    def _retryable(self, event_id: str) -> bool:
        return (not hasattr(self.client, "should_queue_failure")
                or self.client.should_queue_failure(event_id))

    def _session_active(self, session_id: str) -> bool:
        with self._condition:
            return session_id == self._active_session_id

    @staticmethod
    def _job_session(job) -> str:
        return (job.event.training_session_id if isinstance(job, EventDelivery)
                else job.training_session_id)

    @staticmethod
    def _encode(frame) -> Optional[bytes]:
        import cv2
        ok, encoded = cv2.imencode(".jpg", frame)
        return encoded.tobytes() if ok else None

    @staticmethod
    def _decode_jpeg(payload: dict) -> Optional[bytes]:
        encoded = payload.get("jpegBase64")
        return base64.b64decode(encoded) if encoded else None
