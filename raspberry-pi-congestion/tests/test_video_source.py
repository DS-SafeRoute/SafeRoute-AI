from raspberry_pi_congestion.video_source import FileVideoSource, RtspVideoSource


class Capture:
    def __init__(self, frames=(), opened=True):
        self.frames = list(frames); self.opened = opened; self.released = False
    def isOpened(self): return self.opened
    def read(self): return (True, self.frames.pop(0)) if self.frames else (False, None)
    def release(self): self.released = True


def test_file_eof_stops_and_releases():
    cap = Capture(["frame"])
    source = FileVideoSource("video.mp4", capture_factory=lambda _: cap)
    assert list(source.frames()) == ["frame"]
    source.close()
    assert cap.released


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
