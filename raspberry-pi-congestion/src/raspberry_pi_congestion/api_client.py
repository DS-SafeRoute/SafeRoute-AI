from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Callable, Mapping, Optional

from .models import CongestionObservation

logger = logging.getLogger(__name__)


class CongestionReporter(ABC):
    @abstractmethod
    def report(self, observation: CongestionObservation) -> bool: ...

    def report_json(self, payload: dict) -> bool:
        return self.report(CongestionObservation(
            event_id=payload["eventId"], cctv_code=payload["cctvCode"],
            avg_headcount=payload["avgHeadcount"], peak_headcount=payload["peakHeadcount"],
            sample_count=payload["sampleCount"], window_start=payload["windowStart"],
            window_end=payload["windowEnd"], captured_at=payload["capturedAt"],
            s3_image_key=payload.get("s3ImageKey"),
        ))


class LoggingCongestionReporter(CongestionReporter):
    def report(self, observation: CongestionObservation) -> bool:
        logger.info("congestion observation: %s", observation.to_json())
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


class SpringCongestionReporter(CongestionReporter):
    RETRYABLE_STATUSES = {429}
    NON_RETRYABLE_STATUSES = {400, 401, 403, 404}

    def __init__(self, base_url: str, path: str = "/api/v1/device/congestion-observations",
                 auth_header_provider: Optional[AuthHeaderProvider] = None,
                 timeout_sec: float = 5.0, max_retries: int = 2,
                 backoff_base_sec: float = 0.5, request: Optional[Callable] = None,
                 sleeper: Callable[[float], None] = time.sleep) -> None:
        self.url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
        self.auth = auth_header_provider or AuthHeaderProvider(None)
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.backoff_base_sec = backoff_base_sec
        self._request = request
        self._sleep = sleeper

    def report(self, observation: CongestionObservation) -> bool:
        return self._post(observation.to_json())

    def report_json(self, payload: dict) -> bool:
        return self._post(payload)

    def _post(self, payload: Mapping) -> bool:
        if self._request is None:
            import requests
            request = requests.post
            network_errors = (requests.RequestException,)
        else:
            request = self._request
            network_errors = (TimeoutError, OSError)
        headers = {"Content-Type": "application/json", **self.auth.headers()}
        for attempt in range(self.max_retries + 1):
            retry = False
            try:
                response = request(
                    self.url,
                    json=dict(payload),
                    headers=headers,
                    timeout=(self.timeout_sec, self.timeout_sec),
                )
                status = response.status_code
                if 200 <= status < 300:
                    return True
                if status in self.NON_RETRYABLE_STATUSES or status < 500 and status not in self.RETRYABLE_STATUSES:
                    logger.error("Observation rejected with HTTP %d", status)
                    return False
                retry = True
            except Exception as exc:
                logger.warning("Observation request failed: %s", type(exc).__name__)
                retry = True
            if not retry or attempt >= self.max_retries:
                return False
            self._sleep(self.backoff_base_sec * (2 ** attempt))
        return False
