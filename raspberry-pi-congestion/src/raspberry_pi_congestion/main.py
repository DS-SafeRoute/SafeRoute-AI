from __future__ import annotations

import argparse
import logging
import sys

from .api_client import AuthHeaderProvider, LoggingCongestionReporter, SpringCongestionReporter
from .app import CongestionPipeline
from .config import AppConfig, ConfigError
from .detectors import create_detector
from .offline_queue import OfflineQueue
from .roi_counter import RoiCounter
from .roi_provider import InteractiveRoiSelector, JsonRoiProvider
from .video_source import FileVideoSource, create_video_source
from .window_aggregator import WindowAggregator


def _setup_roi(config: AppConfig) -> None:
    source = FileVideoSource(config.video_source)
    try:
        frame = next(source.frames())
        JsonRoiProvider(config.roi_config_path).save(InteractiveRoiSelector().select(frame))
    finally:
        source.close()


def main(argv=None) -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", choices=["setup-roi", "dry-run", "file", "rtsp", "test"])
    args = parser.parse_args(argv)
    try:
        config = AppConfig.from_env(mode=args.mode)
    except ConfigError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2
    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if config.mode == "setup-roi":
        _setup_roi(config)
        return 0
    reporter = LoggingCongestionReporter() if config.mode in {"dry-run", "test"} else SpringCongestionReporter(
        config.server_base_url or "", config.observation_path,
        AuthHeaderProvider(config.device_auth_token, config.auth_header_name, config.auth_header_prefix),
        config.request_timeout_sec, config.max_http_retries,
    )
    source = create_video_source(config.video_source, file_loop=config.video_loop,
                                 max_reconnects=config.rtsp_max_reconnects,
                                 base_delay_sec=config.rtsp_reconnect_base_delay_sec) if config.mode == "rtsp" else FileVideoSource(config.video_source, config.video_loop)
    if config.mode == "test":
        from .detectors import FakePersonDetector
        detector = FakePersonDetector()
    else:
        detector = create_detector(config)
    queue = None if config.mode in {"dry-run", "test"} else OfflineQueue(config.offline_queue_db_path, config.offline_queue_max_age_sec, config.offline_queue_max_items)
    pipeline = CongestionPipeline(source, detector, RoiCounter(JsonRoiProvider(config.roi_config_path).load()),
                                  WindowAggregator(config.window_sec), reporter, config.cctv_code, queue,
                                  config.target_inference_fps, config.offline_flush_interval_sec)
    pipeline.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
