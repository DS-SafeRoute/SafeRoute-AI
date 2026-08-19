from raspberry_pi_congestion.event_detector import CongestionEventDetector
from raspberry_pi_congestion.models import CongestionLevel, EventDetectionSettings


def test_started_level_up_and_recovery_consecutive_frames():
    detector = CongestionEventDetector()
    settings = EventDetectionSettings(3, 5, 30)
    assert [detector.observe(CongestionLevel.CROWDED, i, settings) for i in range(3)] == [None, None, "CONGESTION_STARTED"]
    assert [detector.observe(CongestionLevel.VERY_CROWDED, i, settings) for i in range(3, 6)] == [None, None, "CONGESTION_LEVEL_UP"]
    assert [detector.observe(CongestionLevel.NORMAL, i, settings) for i in range(6, 11)] == [None, None, None, None, "CONGESTION_ENDED"]


def test_level_up_bypasses_started_cooldown():
    detector = CongestionEventDetector()
    settings = EventDetectionSettings(1, 1, 30)
    assert detector.observe(CongestionLevel.CAUTION, 1000, settings) == "CONGESTION_STARTED"
    assert detector.observe(CongestionLevel.CROWDED, 1001, settings) == "CONGESTION_LEVEL_UP"


def test_suppressed_restart_can_fire_after_cooldown_expires():
    detector = CongestionEventDetector()
    settings = EventDetectionSettings(1, 1, 30)
    assert detector.observe(CongestionLevel.CAUTION, 1000, settings) == "CONGESTION_STARTED"
    assert detector.observe(CongestionLevel.NORMAL, 2000, settings) == "CONGESTION_ENDED"
    assert detector.observe(CongestionLevel.CAUTION, 3000, settings) is None
    assert detector.observe(CongestionLevel.CAUTION, 31001, settings) == "CONGESTION_STARTED"
