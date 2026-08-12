import pytest

from raspberry_pi_congestion.api_client import AuthHeaderProvider, SpringCongestionReporter
from raspberry_pi_congestion.models import CongestionObservation


class Response:
    def __init__(self, status): self.status_code = status


def observation():
    return CongestionObservation("event-1", "CCTV_A", 5, 8, 25, 1000, 6000, 6000)


def test_request_contains_only_allowed_fields():
    body = observation().to_json()
    assert set(body) == {"eventId", "cctvCode", "avgHeadcount", "peakHeadcount", "sampleCount", "windowStart", "windowEnd", "capturedAt", "s3ImageKey"}
    assert not ({"density", "congestionLevel", "edgeId", "sessionId", "monitoredAreaM2"} & set(body))


@pytest.mark.parametrize("status", [200, 201, 204])
def test_2xx_success(status):
    reporter = SpringCongestionReporter("http://server", request=lambda *a, **k: Response(status))
    assert reporter.report(observation())


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_client_errors_are_not_retried(status):
    calls = []
    reporter = SpringCongestionReporter("http://server", max_retries=3, request=lambda *a, **k: calls.append(k["json"]) or Response(status), sleeper=lambda _: None)
    assert not reporter.report(observation())
    assert len(calls) == 1


@pytest.mark.parametrize("outcomes", [[429, 200], [500, 200], [TimeoutError(), 200]])
def test_retryable_failures_are_bounded_and_payload_is_stable(outcomes):
    calls = []
    values = iter(outcomes)
    def request(*args, **kwargs):
        calls.append(kwargs["json"].copy())
        value = next(values)
        if isinstance(value, Exception): raise value
        return Response(value)
    reporter = SpringCongestionReporter("http://server", max_retries=1, request=request, sleeper=lambda _: None)
    assert reporter.report(observation())
    assert len(calls) == 2 and calls[0] == calls[1]


def test_configurable_auth_header():
    auth = AuthHeaderProvider("secret", "X-Custom-Token", "")
    assert auth.headers() == {"X-Custom-Token": "secret"}
