from __future__ import annotations

import argparse
import logging
import sys
import threading

from .api_client import AuthHeaderProvider, LoggingCongestionReporter, SafeRouteDeviceClient
from .app import CongestionPipeline
from .config import AppConfig, ConfigError
from .detectors import create_detector
from .offline_queue import OfflineQueue
from .preview import OpenCvPreview
from .relay import LightCommandExecutor, RelayController, RelayControllerError
from .roi_counter import RoiCounter
from .video_source import FileVideoSource, create_video_source
from .window_aggregator import WindowAggregator

logger = logging.getLogger(__name__)


def _start_light_command_executor(config: AppConfig, device_client: SafeRouteDeviceClient) -> None:
    """유도등 명령 폴링을 이 프로세스 안에서 백그라운드 스레드로 돌린다.

    혼잡도 감지(pipeline.run())와 완전히 독립된 관심사라, 릴레이 초기화가
    실패해도 혼잡도 감지 자체는 계속 돼야 한다 - 여기서 예외를 흡수한다.
    """
    try:
        relay = RelayController(host=config.relay_host, port=config.relay_port)
        relay.refresh_status()
    except RelayControllerError as exc:
        logger.warning("릴레이 초기화 실패 - 유도등 명령 폴링 없이 계속 진행: %s", exc)
        return
    executor = LightCommandExecutor(device_client, relay, config.cctv_code)
    thread = threading.Thread(
        target=executor.run_forever,
        kwargs={"interval_sec": config.relay_poll_interval_sec},
        name="light-command-executor",
        daemon=True,
    )
    thread.start()
    logger.info("유도등 명령 폴링 시작: host=%s, port=%s, interval=%ss",
                config.relay_host, config.relay_port, config.relay_poll_interval_sec)


def main(argv=None) -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", choices=["dry-run", "file", "rtsp", "test"])
    args = parser.parse_args(argv)
    try:
        config = AppConfig.from_env(mode=args.mode)
    except ConfigError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2
    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    device_client = None if config.mode in {"dry-run", "test"} else SafeRouteDeviceClient(
        config.server_base_url or "",
        AuthHeaderProvider(config.device_auth_token, config.auth_header_name, config.auth_header_prefix),
        config.request_timeout_sec, config.max_http_retries,
    )
    reporter = LoggingCongestionReporter() if device_client is None else device_client
    source = create_video_source(config.video_source, file_loop=config.video_loop,
                                 max_reconnects=config.rtsp_max_reconnects,
                                 base_delay_sec=config.rtsp_reconnect_base_delay_sec) if config.mode == "rtsp" else FileVideoSource(
        config.video_source,
        config.video_loop,
        realtime=config.file_realtime,
        fallback_fps=config.file_fallback_fps,
    )
    if config.mode == "test":
        from .detectors import FakePersonDetector
        detector = FakePersonDetector()
    else:
        detector = create_detector(config)
    queue = None if config.mode in {"dry-run", "test"} else OfflineQueue(config.offline_queue_db_path, config.offline_queue_max_age_sec, config.offline_queue_max_items)
    preview = OpenCvPreview(()) if config.show_preview else None
    pipeline = CongestionPipeline(source, detector, RoiCounter(),
                                  WindowAggregator(config.window_sec), reporter, config.cctv_code, queue,
                                  config.target_inference_fps, config.offline_flush_interval_sec,
                                  config_provider=device_client,
                                  config_poll_active_sec=config.config_poll_active_sec,
                                  config_poll_inactive_sec=config.config_poll_inactive_sec,
                                  preview=preview)
    if device_client is not None and config.relay_host:
        _start_light_command_executor(config, device_client)
    pipeline.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
