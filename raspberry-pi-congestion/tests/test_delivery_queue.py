import threading

import numpy as np

from raspberry_pi_congestion.delivery import (
    DeliveryQueue, EventDelivery, MonitoringDelivery, Snapshot,
)
from raspberry_pi_congestion.models import (
    CongestionEvent, CongestionLevel, WindowSummary,
)


SESSION = "550e8400-e29b-41d4-a716-446655440000"


class BlockingRenderer:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def render(self, frame, detections, inside_detections):
        self.calls += 1
        if self.calls == 1:
            self.entered.set()
            assert self.release.wait(2)
        return frame


class Client:
    def __init__(self):
        self.delivered = []

    def report(self, observation):
        self.delivered.append(("observation", observation.event_id))
        return True

    def report_event(self, event):
        self.delivered.append(("event", event.event_id))
        return True


def snapshot(value):
    return Snapshot(np.full((4, 4, 3), value, dtype=np.uint8), (), ())


def monitoring(event_id):
    return MonitoringDelivery(
        event_id, SESSION, "CCTV_001", 1,
        WindowSummary(0, 5_000, 4_000, 1, 1.0, 1), snapshot(1),
    )


def event(event_id):
    return EventDelivery(
        CongestionEvent(
            event_id, SESSION, "CCTV_001", "CONGESTION_STARTED",
            4_000, 3, 3.0, CongestionLevel.CROWDED, 1,
        ),
        snapshot(2),
    )


def test_full_queue_drops_oldest_monitoring_and_sends_event_first():
    renderer = BlockingRenderer()
    client = Client()
    queue = DeliveryQueue(client, renderer, max_items=2)
    queue.set_session(SESSION)

    assert queue.submit_monitoring(monitoring("active"))
    assert renderer.entered.wait(1)
    assert queue.submit_monitoring(monitoring("old"))
    assert queue.submit_monitoring(monitoring("new"))
    assert queue.submit_event(event("important"))
    assert queue.pending_count == 2

    renderer.release.set()
    assert queue.wait_idle()
    assert client.delivered == [
        ("observation", "active"),
        ("event", "important"),
        ("observation", "new"),
    ]
    queue.close()


def test_session_change_discards_waiting_and_suppresses_inflight_delivery():
    renderer = BlockingRenderer()
    client = Client()
    queue = DeliveryQueue(client, renderer, max_items=3)
    queue.set_session(SESSION)
    queue.submit_monitoring(monitoring("active"))
    assert renderer.entered.wait(1)
    queue.submit_monitoring(monitoring("waiting"))

    changed = threading.Event()
    discarded = []

    def change_session():
        discarded.append(queue.set_session("new-session"))
        changed.set()

    transition = threading.Thread(target=change_session)
    transition.start()
    with queue._condition:
        while queue._active_session_id != "new-session":
            queue._condition.wait(1)
    assert not changed.is_set()
    renderer.release.set()
    assert changed.wait(1)
    transition.join(1)
    assert discarded == [1]
    assert client.delivered == []
    queue.close()


class PassthroughRenderer:
    def render(self, frame, detections, inside_detections):
        return frame


class BlockingReportClient(Client):
    def __init__(self):
        super().__init__()
        self.report_started = threading.Event()
        self.release_report = threading.Event()

    def report(self, observation):
        self.report_started.set()
        assert self.release_report.wait(2)
        return False


def test_session_change_waits_for_inflight_report_and_skips_stale_retry(tmp_path):
    from raspberry_pi_congestion.offline_queue import OfflineQueue

    client = BlockingReportClient()
    offline = OfflineQueue(str(tmp_path / "offline.db"))
    queue = DeliveryQueue(client, PassthroughRenderer(), offline_queue=offline)
    queue.set_session(SESSION)
    queue.submit_monitoring(monitoring("in-flight"))
    assert client.report_started.wait(1)

    changed = threading.Event()
    transition = threading.Thread(
        target=lambda: (queue.set_session("new-session"), changed.set())
    )
    transition.start()
    with queue._condition:
        while queue._active_session_id != "new-session":
            queue._condition.wait(1)
    assert not changed.is_set()

    client.release_report.set()
    assert changed.wait(1)
    transition.join(1)
    assert offline.size() == 0
    queue.close()
    offline.close()


class SlowUploadClient(Client):
    def __init__(self):
        super().__init__()
        self.upload_started = threading.Event()
        self.release_upload = threading.Event()

    def request_image_upload(self, **kwargs):
        self.upload_started.set()
        assert self.release_upload.wait(2)
        return None


def test_network_wait_does_not_block_inference_submission():
    from raspberry_pi_congestion.app import CongestionPipeline
    from raspberry_pi_congestion.detectors.fake_detector import FakePersonDetector
    from raspberry_pi_congestion.models import (
        CongestionThresholds, DeviceCongestionConfig, EventDetectionSettings, Point,
    )
    from raspberry_pi_congestion.roi_counter import RoiCounter
    from raspberry_pi_congestion.window_aggregator import WindowAggregator

    class Source:
        def close(self):
            pass

    client = SlowUploadClient()
    pipeline = CongestionPipeline(
        Source(), FakePersonDetector([]),
        RoiCounter([Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]),
        WindowAggregator(5), client, "CCTV_001", config_provider=client,
    )
    pipeline._config = DeviceCongestionConfig(
        True, SESSION, "CCTV_001", 1, 100,
        5, 5, CongestionThresholds(1, 2, 3), EventDetectionSettings(2, 2, 0),
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    submission_done = threading.Event()

    def submit_frames():
        pipeline.process_frame(frame, 1_000)
        pipeline.process_frame(frame, 2_000)
        pipeline.process_frame(frame, 5_000)
        submission_done.set()

    submission = threading.Thread(target=submit_frames)
    submission.start()
    assert client.upload_started.wait(1)
    assert submission_done.wait(1)
    client.release_upload.set()
    submission.join(1)
    pipeline.close()


def test_shutdown_timeout_persists_only_jobs_not_owned_by_worker(tmp_path):
    from raspberry_pi_congestion.offline_queue import OfflineQueue

    client = SlowUploadClient()
    offline = OfflineQueue(str(tmp_path / "offline.db"))
    queue = DeliveryQueue(
        client, BlockingRenderer(), offline_queue=offline,
        shutdown_timeout_sec=0.01,
    )
    # This test blocks in the client, not in the renderer.
    queue.renderer.release.set()
    queue.set_session(SESSION)
    queue.submit_monitoring(monitoring("in-flight"))
    assert client.upload_started.wait(1)
    queue.submit_monitoring(monitoring("waiting"))

    queue.close()

    items = offline.peek_oldest()
    assert len(items) == 1
    item = items[0]
    assert item.event_id == "waiting"
    assert item.operation == "pending_observation"
    assert item.payload["observationPayload"]["capturedAt"] == 4_000
    assert item.payload["jpegBase64"]
    client.release_upload.set()
    queue._thread.join(1)
    offline.close()
