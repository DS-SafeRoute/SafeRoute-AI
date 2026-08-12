import pytest

from raspberry_pi_congestion.detectors import FakePersonDetector, create_detector
from raspberry_pi_congestion.detectors.hailo_detector import HailoAdapterNotImplemented, HailoPersonDetector


class Config:
    detector_backend = "fake"
    model_path = None
    detector_conf_threshold = .4


def test_fake_detector_factory():
    assert isinstance(create_detector(Config()), FakePersonDetector)


def test_hailo_adapter_is_explicitly_incomplete():
    with pytest.raises(HailoAdapterNotImplemented):
        HailoPersonDetector("model.hef")
