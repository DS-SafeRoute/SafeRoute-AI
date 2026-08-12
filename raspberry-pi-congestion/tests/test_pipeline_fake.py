import numpy as np

from raspberry_pi_congestion.api_client import CongestionReporter
from raspberry_pi_congestion.app import CongestionPipeline
from raspberry_pi_congestion.detectors.fake_detector import FakePersonDetector
from raspberry_pi_congestion.models import Detection, Point
from raspberry_pi_congestion.roi_counter import RoiCounter
from raspberry_pi_congestion.window_aggregator import WindowAggregator


class Clock:
    value = 0.0
    def __call__(self): return self.value


class Source:
    def __init__(self, frame): self.frame = frame; self.closed = False
    def frames(self): yield self.frame
    def close(self): self.closed = True


class Reporter(CongestionReporter):
    def __init__(self): self.items = []
    def report(self, item): self.items.append(item); return True


def test_fake_detector_pipeline_and_resource_cleanup():
    clock = Clock()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    source = Source(frame)
    detector = FakePersonDetector([Detection(30, 20, 50, 60, .9)])
    detector.closed = False
    detector.close = lambda: setattr(detector, "closed", True)
    reporter = Reporter()
    aggregator = WindowAggregator(5, clock)
    pipeline = CongestionPipeline(source, detector, RoiCounter([Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]),
                                  aggregator, reporter, "CCTV_TEST", target_fps=0,
                                  monotonic=clock, epoch_ms=lambda: 6000)
    aggregator.add_sample(1, 1000)
    clock.value = 5
    pipeline.run()
    assert reporter.items[0].sample_count == 2
    assert reporter.items[0].cctv_code == "CCTV_TEST"
    assert source.closed and detector.closed
