import numpy as np

from raspberry_pi_congestion.api_client import LoggingCongestionReporter
from raspberry_pi_congestion.app import CongestionPipeline
from raspberry_pi_congestion.models import DeviceCongestionConfig, Point
from raspberry_pi_congestion.roi_counter import RoiCounter
from raspberry_pi_congestion.window_aggregator import WindowAggregator


class Source:
    def __init__(self):
        self.closed = False

    def frames(self):
        yield np.zeros((10, 10, 3), dtype=np.uint8)

    def close(self):
        self.closed = True


class Detector:
    def __init__(self):
        self.calls = 0
        self.closed = False

    def detect(self, frame):
        self.calls += 1
        return []

    def close(self):
        self.closed = True


class InactiveProvider:
    def __init__(self):
        self.calls = 0
    def fetch_config(self, code):
        self.calls += 1
        return DeviceCongestionConfig(
            False,
            None,
            code,
            1,
            snapshot_interval_sec=7,
            target_inference_fps=2,
        )


def test_inactive_training_stops_inference_and_upload_work():
    source, detector, provider = Source(), Detector(), InactiveProvider()
    aggregator = WindowAggregator()
    pipeline = CongestionPipeline(
        source, detector, RoiCounter([Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]),
        aggregator, LoggingCongestionReporter(), "CCTV_001", config_provider=provider,
    )
    pipeline.run()
    assert provider.calls == 1
    assert detector.calls == 0
    assert aggregator.window_sec == 7
    assert pipeline.target_fps == 2
    assert source.closed and detector.closed
