import math

import pytest

from raspberry_pi_congestion.video_source import FileVideoSource, RtspVideoSource


class Capture:
    def __init__(self, frames=(), opened=True, fps=30.0):
        self.frames = list(frames)
        self.opened = opened
        self.released = False
        self.fps = fps

    def isOpened(self):
        return self.opened

    def read(self):
        return (True, self.frames.pop(0)) if self.frames else (False, None)

    def get(self, _):
        return self.fps

    def release(self):
        self.released = True


class Clock:
    def __init__(self):
        self.value = 0.0
        self.sleeps = []

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.value += seconds


def test_file_eof_stops_and_releases():
    cap = Capture(["frame"])
    source = FileVideoSource("video.mp4", realtime=False, capture_factory=lambda _: cap)
    assert list(source.frames()) == ["frame"]
    source.close()
    assert cap.released


def test_file_frames_follow_source_fps_in_realtime_mode():
    cap = Capture(["first", "second", "third"], fps=2.0)
    clock = Clock()
    source = FileVideoSource(
        "video.mp4", capture_factory=lambda _: cap,
        monotonic=clock, sleeper=clock.sleep,
    )

    assert list(source.frames()) == ["first", "second", "third"]
    assert clock.sleeps == [0.5, 0.5]
    source.close()


def test_file_uses_fallback_when_source_fps_is_invalid():
    cap = Capture(["first", "second"], fps=0)
    clock = Clock()
    source = FileVideoSource(
        "video.mp4", fallback_fps=10, capture_factory=lambda _: cap,
        monotonic=clock, sleeper=clock.sleep,
    )

    assert list(source.frames()) == ["first", "second"]
    assert clock.sleeps == [0.1]
    source.close()


@pytest.mark.parametrize("fallback_fps", [math.nan, math.inf, -math.inf, 0, -1])
def test_file_rejects_non_finite_or_non_positive_fallback_fps(fallback_fps):
    with pytest.raises(ValueError, match="finite and positive"):
        FileVideoSource("video.mp4", fallback_fps=fallback_fps)


def test_file_loop_reopens_capture_and_resets_pacing():
    captures = iter([Capture(["first"], fps=2), Capture(["second"], fps=2)])
    clock = Clock()
    source = FileVideoSource(
        "video.mp4", loop=True, capture_factory=lambda _: next(captures),
        monotonic=clock, sleeper=clock.sleep,
    )
    frames = source.frames()

    assert next(frames) == "first"
    assert next(frames) == "second"
    source.close()
    assert clock.sleeps == [0.5]


def test_file_keeps_frames_that_are_older_than_playback_clock():
    cap = Capture(list(range(10)), fps=10)
    clock = Clock()
    source = FileVideoSource(
        "video.mp4", capture_factory=lambda _: cap,
        monotonic=clock, sleeper=clock.sleep,
    )
    frames = source.frames()

    assert next(frames) == 0
    clock.value = 0.35
    assert list(frames) == list(range(1, 10))
    assert source.current_position_ms == pytest.approx(900)
    source.close()


def test_rtsp_reconnect_limit_and_backoff():
    captures = []
    def factory(_):
        cap = Capture()
        captures.append(cap)
        return cap
    sleeps = []
    source = RtspVideoSource("rtsp://user:password@camera/stream", max_reconnects=2,
                             base_delay_sec=.5, capture_factory=factory, sleeper=sleeps.append)
    assert list(source.frames()) == []
    assert sleeps == [.5, 1.0]
    assert len(captures) == 3
    source.close()
