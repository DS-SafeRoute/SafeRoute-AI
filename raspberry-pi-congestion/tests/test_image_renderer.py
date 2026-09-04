import numpy as np

from raspberry_pi_congestion.image_renderer import OpenCvDetectionRenderer
from raspberry_pi_congestion.models import Detection, Point


def test_renderer_draws_person_bbox_without_mutating_source_frame():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    original = frame.copy()
    detection = Detection(40, 40, 100, 100, 0.91)
    renderer = OpenCvDetectionRenderer(
        [Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]
    )

    rendered = renderer.render(frame, [detection], [detection])

    assert np.array_equal(frame, original)
    assert not np.array_equal(rendered, original)
    assert rendered[40, 40].tolist() == [0, 255, 0]


def test_renderer_does_not_draw_roi_or_mark_detections_as_outside():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    detection = Detection(40, 40, 100, 100, 0.91)
    renderer = OpenCvDetectionRenderer(
        [Point(0, 0), Point(0.1, 0), Point(0.1, 0.1), Point(0, 0.1)]
    )

    rendered = renderer.render(frame, [detection], [])

    assert rendered[40, 40].tolist() == [0, 255, 0]
    assert rendered[0, 0].tolist() == [0, 0, 0]
