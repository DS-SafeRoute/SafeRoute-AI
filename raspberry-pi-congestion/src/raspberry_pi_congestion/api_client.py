from __future__ import annotations

import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from typing import Callable, Mapping, Optional

from .models import CongestionEvent, CongestionObservation, DeviceCongestionConfig

logger = logging.getLogger(__name__)


class CongestionReporter(ABC):
    @abstractmethod
    def report(self, observation: CongestionObservation) -> bool: ...

    def report_json(self, payload: dict) -> bool:
        raise NotImplementedError


class LoggingCongestionReporter(CongestionReporter):
    def report(self, observation: CongestionObservation) -> bool:
        logger.info("congestion observation: %s", observation.to_json())
        return True

    def report_json(self, payload: dict) -> bool:
        logger.info("queued congestion payload: %s", payload)
        return True


class AuthHeaderProvider:
    def __init__(self, token: Optional[str], header_name: str = "Authorization", prefix: str = "Bearer") -> None:
        self._token = token
        self.header_name = header_name
        self.prefix = prefix.strip()

    def headers(self) -> dict[str, str]:
        if not self._token:
            return {}
        value = f"{self.prefix} {self._token}" if self.prefix else self._token
        return {self.header_name: value}


class SafeRouteDeviceClient(CongestionReporter):
    CONFIG_PATH = "/api/v1/device/congestion-config"
    OBSERVATION_PATH = "/api/v1/device/congestion-observations"
    EVENT_PATH = "/api/v1/device/congestion-events"
    PRESIGNED_PATH = "/api/v1/device/congestion-images/presigned-url"
    RETRYABLE_STATUSES = {429}

    def __init__(self, base_url: str, auth_header_provider: AuthHeaderProvider,
                 timeout_sec: float = 5.0, max_retries: int = 2,
                 backoff_base_sec: float = 0.5, request: Optional[Callable] = None,
                 sleeper: Callable[[float], None] = time.sleep) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth = auth_header_provider
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.backoff_base_sec = backoff_base_sec
        self._request = request
        self._sleep = sleeper
        self._request_state = threading.local()
        self._delivery_outcomes: dict[str, bool] = {}
        self._outcomes_lock = threading.Lock()
        self._config_refresh_requested = threading.Event()

    def fetch_config(self, cctv_code: str) -> Optional[DeviceCongestionConfig]:
        response = self._send("GET", self.CONFIG_PATH, params={"cctvCode": cctv_code})
        if response is None:
            return None
        try:
            config = DeviceCongestionConfig.from_json(response.json())
            if config.cctv_code != cctv_code:
                raise ValueError("response cctvCode does not match this device")
            return config
        except (KeyError, TypeError, ValueError) as exc:
            logger.error("Invalid congestion config response: %s", exc)
            return None

    def report(self, observation: CongestionObservation) -> bool:
        success = self._send("POST", self.OBSERVATION_PATH, json=observation.to_json()) is not None
        self._remember_outcome(observation.event_id, success)
        return success

    def report_json(self, payload: dict) -> bool:
        success = self._send("POST", self.OBSERVATION_PATH, json=payload) is not None
        self._remember_outcome(str(payload["eventId"]), success)
        return success

    def report_event(self, event: CongestionEvent) -> bool:
        return self.report_event_json(event.to_json())

    def report_event_json(self, payload: dict) -> bool:
        success = self._send("POST", self.EVENT_PATH, json=payload) is not None
        self._remember_outcome(str(payload["eventId"]), success)
        return success

    def _remember_outcome(self, event_id: str, success: bool) -> None:
        with self._outcomes_lock:
            self._delivery_outcomes[event_id] = bool(not success and getattr(self._request_state, "retryable", True))

    def should_queue_failure(self, event_id: str) -> bool:
        with self._outcomes_lock:
            return self._delivery_outcomes.pop(event_id, True)

    def consume_config_refresh_request(self) -> bool:
        requested = self._config_refresh_requested.is_set()
        self._config_refresh_requested.clear()
        return requested

    def request_image_upload(self, *, training_session_id: str, cctv_code: str,
                             image_type: str, reference_id: str,
                             captured_at: int) -> Optional[dict]:
        payload = {
            "requestId": str(uuid.uuid4()),
            "trainingSessionId": training_session_id,
            "cctvCode": cctv_code,
            "imageType": image_type,
            "referenceId": reference_id,
            "capturedAt": captured_at,
            "contentType": "image/jpeg",
        }
        response = self._send("POST", self.PRESIGNED_PATH, json=payload)
        if response is None:
            return None
        try:
            body = response.json()
            return {"objectKey": str(body["objectKey"]), "uploadUrl": str(body["uploadUrl"]), "expiresAt": int(body["expiresAt"])}
        except (KeyError, TypeError, ValueError):
            logger.error("Invalid presigned URL response")
            return None

    def upload_jpeg(self, upload_url: str, jpeg: bytes) -> bool:
        request = self._request_fn()
        try:
            response = request("PUT", upload_url, data=jpeg, headers={"Content-Type": "image/jpeg"},
                               timeout=(self.timeout_sec, self.timeout_sec))
            return 200 <= response.status_code < 300
        except Exception as exc:
            logger.warning("Image upload failed: %s", type(exc).__name__)
            return False

    def attach_event_image(self, event_id: str, object_key: str, uploaded_at: int) -> bool:
        path = f"{self.EVENT_PATH}/{event_id}/image"
        success = self._send("PATCH", path, json={"eventImageKey": object_key, "uploadedAt": uploaded_at},
                             retry_statuses={404, 409}) is not None
        self._remember_outcome(f"image:{event_id}", success)
        return success

    def _request_fn(self) -> Callable:
        if self._request is not None:
            return self._request
        import requests
        return requests.request

    def _send(self, method: str, path: str, **kwargs):
        request = self._request_fn()
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = {**self.auth.headers(), **kwargs.pop("headers", {})}
        retry_statuses = kwargs.pop("retry_statuses", set())
        if "json" in kwargs:
            headers["Content-Type"] = "application/json"
        for attempt in range(self.max_retries + 1):
            try:
                response = request(method, url, headers=headers,
                                   timeout=(self.timeout_sec, self.timeout_sec), **kwargs)
                status = response.status_code
                if 200 <= status < 300:
                    self._request_state.retryable = False
                    return response
                if status == 409:
                    self._config_refresh_requested.set()
                retryable = status >= 500 or status in self.RETRYABLE_STATUSES or status == 409 or status in retry_statuses
                if not retryable:
                    self._request_state.retryable = False
                    logger.error("Device API %s %s rejected with HTTP %d", method, path, status)
                    return None
            except Exception as exc:
                logger.warning("Device API %s %s failed: %s", method, path, type(exc).__name__)
            if attempt >= self.max_retries:
                self._request_state.retryable = True
                return None
            self._sleep(self.backoff_base_sec * (2 ** attempt))
        return None


class SpringCongestionReporter(SafeRouteDeviceClient):
    """Backward-compatible observation-only constructor."""

    def __init__(self, base_url: str, path: str = SafeRouteDeviceClient.OBSERVATION_PATH,
                 auth_header_provider: Optional[AuthHeaderProvider] = None,
                 timeout_sec: float = 5.0, max_retries: int = 2,
                 backoff_base_sec: float = 0.5, request: Optional[Callable] = None,
                 sleeper: Callable[[float], None] = time.sleep) -> None:
        super().__init__(base_url, auth_header_provider or AuthHeaderProvider(None), timeout_sec,
                         max_retries, backoff_base_sec, request, sleeper)
        self.OBSERVATION_PATH = path
