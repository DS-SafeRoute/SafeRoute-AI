from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Callable, Iterator, Optional

logger = logging.getLogger(__name__)


class VideoSource(ABC):
    @abstractmethod
    def frames(self) -> Iterator[object]: ...

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class FileVideoSource(VideoSource):
    def __init__(self, path: str, loop: bool = False, capture_factory: Optional[Callable] = None) -> None:
        self.path = path
        self.loop = loop
        self._capture_factory = capture_factory or _opencv_capture
        self._cap = self._capture_factory(path)
        if not self._cap.isOpened():
            self.close()
            raise RuntimeError(f"Cannot open video file: {path}")

    def frames(self) -> Iterator[object]:
        while self._cap is not None:
            ok, frame = self._cap.read()
            if ok:
                yield frame
            elif self.loop:
                self._cap.release()
                self._cap = self._capture_factory(self.path)
                if not self._cap.isOpened():
                    return
            else:
                return

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class RtspVideoSource(VideoSource):
    def __init__(self, url: str, max_reconnects: int = 5, base_delay_sec: float = 1.0,
                 max_delay_sec: float = 30.0, capture_factory: Optional[Callable] = None,
                 sleeper: Callable[[float], None] = time.sleep) -> None:
        if max_reconnects < 0:
            raise ValueError("max_reconnects must not be negative")
        self.url = url
        self.max_reconnects = max_reconnects
        self.base_delay_sec = base_delay_sec
        self.max_delay_sec = max_delay_sec
        self._capture_factory = capture_factory or _opencv_capture
        self._sleep = sleeper
        self._cap = self._capture_factory(url)

    def frames(self) -> Iterator[object]:
        reconnects = 0
        while self._cap is not None:
            ok, frame = self._cap.read() if self._cap.isOpened() else (False, None)
            if ok:
                reconnects = 0
                yield frame
                continue
            if reconnects >= self.max_reconnects:
                logger.error("RTSP reconnect limit reached (%d)", self.max_reconnects)
                return
            delay = min(self.base_delay_sec * (2 ** reconnects), self.max_delay_sec)
            reconnects += 1
            logger.warning("RTSP read failed; reconnect %d/%d in %.1fs", reconnects, self.max_reconnects, delay)
            self._sleep(delay)
            self._cap.release()
            self._cap = self._capture_factory(self.url)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def create_video_source(source: str, file_loop: bool = False, **rtsp_options) -> VideoSource:
    if source.lower().startswith("rtsp://"):
        return RtspVideoSource(source, **rtsp_options)
    return FileVideoSource(source, loop=file_loop)


def _opencv_capture(source: str):
    import cv2
    return cv2.VideoCapture(source)
