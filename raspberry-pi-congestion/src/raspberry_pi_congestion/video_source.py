from __future__ import annotations

import logging
import math
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
    def __init__(self, path: str, loop: bool = False, realtime: bool = True,
                 fallback_fps: float = 30.0, capture_factory: Optional[Callable] = None,
                 monotonic: Callable[[], float] = time.monotonic,
                 sleeper: Callable[[float], None] = time.sleep) -> None:
        if not math.isfinite(fallback_fps) or fallback_fps <= 0:
            raise ValueError("fallback_fps must be finite and positive")
        self.path = path
        self.loop = loop
        self.realtime = realtime
        self.fallback_fps = fallback_fps
        self._capture_factory = capture_factory or _opencv_capture
        self._monotonic = monotonic
        self._sleep = sleeper
        self._cap = self._capture_factory(path)
        if not self._cap.isOpened():
            self.close()
            raise RuntimeError(f"Cannot open video file: {path}")
        self._frame_interval_sec = self._resolve_frame_interval()

    def frames(self) -> Iterator[object]:
        next_frame_at = self._monotonic()
        while self._cap is not None:
            ok, frame = self._cap.read()
            if ok:
                if self.realtime:
                    # 파일을 디스크 속도로 읽으면 5초 집계가 생성되기 전에 영상이 끝난다.
                    # 원본 FPS 간격만큼 기다려 실제 CCTV 스트림과 같은 시간 흐름을 만든다.
                    delay = next_frame_at - self._monotonic()
                    if delay > 0:
                        self._sleep(delay)
                yield frame
                if self.realtime:
                    next_frame_at += self._frame_interval_sec
                    now = self._monotonic()
                    if next_frame_at < now - self._frame_interval_sec:
                        # 추론이 오래 걸린 경우 지난 프레임 시간까지 연속 sleep하지 않도록 기준을 재설정한다.
                        next_frame_at = now
            elif self.loop:
                self._cap.release()
                self._cap = self._capture_factory(self.path)
                if not self._cap.isOpened():
                    return
                self._frame_interval_sec = self._resolve_frame_interval()
                next_frame_at = self._monotonic()
            else:
                return

    def _resolve_frame_interval(self) -> float:
        if not self.realtime:
            return 0.0
        try:
            # OpenCV CAP_PROP_FPS의 숫자 값은 5다. 테스트 대역에서도 cv2 import 없이 조회한다.
            source_fps = float(self._cap.get(5))
        except (AttributeError, TypeError, ValueError):
            source_fps = 0.0
        if not math.isfinite(source_fps) or source_fps <= 0:
            logger.warning(
                "영상 FPS를 읽지 못해 fallback FPS %.2f를 사용합니다: %s",
                self.fallback_fps,
                self.path,
            )
            source_fps = self.fallback_fps
        return 1.0 / source_fps

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
