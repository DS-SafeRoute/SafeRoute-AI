from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Mapping, Optional


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class AppConfig:
    mode: str
    video_source: str
    roi_config_path: str
    cctv_code: str
    detector_backend: str
    model_path: Optional[str]
    detector_conf_threshold: float
    target_inference_fps: float
    window_sec: float
    server_base_url: Optional[str]
    observation_path: str
    config_poll_active_sec: float
    config_poll_inactive_sec: float
    device_auth_token: Optional[str]
    auth_header_name: str
    auth_header_prefix: str
    request_timeout_sec: float
    max_http_retries: int
    offline_queue_db_path: str
    offline_queue_max_age_sec: float
    offline_queue_max_items: int
    offline_flush_interval_sec: float
    rtsp_max_reconnects: int
    rtsp_reconnect_base_delay_sec: float
    video_loop: bool
    file_realtime: bool
    file_fallback_fps: float
    log_level: str

    @staticmethod
    def from_env(env: Optional[Mapping[str, str]] = None, mode: Optional[str] = None) -> "AppConfig":
        e = env if env is not None else os.environ
        selected_mode = mode or e.get("RUN_MODE", "dry-run")
        def required(key: str) -> str:
            value = e.get(key)
            if not value:
                raise ConfigError(f"Missing required environment variable: {key}")
            return value
        def boolean(key: str, default: bool) -> bool:
            return e.get(key, str(default)).lower() in {"1", "true", "yes", "on"}
        def positive_float(key: str, default: str) -> float:
            try:
                value = float(e.get(key, default))
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{key} must be a positive number") from exc
            if not math.isfinite(value) or value <= 0:
                raise ConfigError(f"{key} must be a positive number")
            return value
        server = e.get("SAFEROUTE_SERVER_BASE_URL")
        cctv_code = required("CCTV_CODE")
        config_poll_active_sec = positive_float("CONFIG_POLL_ACTIVE_SEC", "5")
        config_poll_inactive_sec = positive_float("CONFIG_POLL_INACTIVE_SEC", "15")
        file_fallback_fps = positive_float("FILE_FALLBACK_FPS", "30")
        if cctv_code not in {"CCTV_001", "CCTV_002"} and selected_mode not in {"dry-run", "test", "setup-roi"}:
            raise ConfigError("CCTV_CODE must be CCTV_001 or CCTV_002")
        if selected_mode in {"file", "rtsp"} and not server:
            raise ConfigError("SAFEROUTE_SERVER_BASE_URL is required for server reporting modes")
        if selected_mode in {"file", "rtsp"} and not e.get("DEVICE_AUTH_TOKEN"):
            raise ConfigError("DEVICE_AUTH_TOKEN is required for server reporting modes")
        return AppConfig(
            mode=selected_mode, video_source=required("VIDEO_SOURCE"),
            roi_config_path=e.get("ROI_CONFIG_PATH", f"./config/roi/{cctv_code}.json"), cctv_code=cctv_code,
            detector_backend=e.get("DETECTOR_BACKEND", "ultralytics"), model_path=e.get("MODEL_PATH"),
            detector_conf_threshold=float(e.get("DETECTOR_CONF_THRESHOLD", "0.4")),
            target_inference_fps=float(e.get("TARGET_INFERENCE_FPS", "5")), window_sec=float(e.get("WINDOW_SEC", "5")),
            server_base_url=server, observation_path=e.get("CONGESTION_OBSERVATION_PATH", "/api/v1/device/congestion-observations"),
            config_poll_active_sec=config_poll_active_sec,
            config_poll_inactive_sec=config_poll_inactive_sec,
            device_auth_token=e.get("DEVICE_AUTH_TOKEN"), auth_header_name=e.get("AUTH_HEADER_NAME", "Authorization"),
            auth_header_prefix=e.get("AUTH_HEADER_PREFIX", "Bearer"), request_timeout_sec=float(e.get("REQUEST_TIMEOUT_SEC", "5")),
            max_http_retries=int(e.get("MAX_HTTP_RETRIES", "2")), offline_queue_db_path=e.get("OFFLINE_QUEUE_DB_PATH", "./offline_queue.sqlite3"),
            offline_queue_max_age_sec=float(e.get("OFFLINE_QUEUE_MAX_AGE_SEC", "86400")), offline_queue_max_items=int(e.get("OFFLINE_QUEUE_MAX_ITEMS", "1000")),
            offline_flush_interval_sec=float(e.get("OFFLINE_FLUSH_INTERVAL_SEC", "30")), rtsp_max_reconnects=int(e.get("RTSP_MAX_RECONNECTS", "5")),
            rtsp_reconnect_base_delay_sec=float(e.get("RTSP_RECONNECT_BASE_DELAY_SEC", "1")), video_loop=boolean("VIDEO_LOOP", False),
            file_realtime=boolean("FILE_REALTIME", True), file_fallback_fps=file_fallback_fps,
            log_level=e.get("LOG_LEVEL", "INFO"),
        )
