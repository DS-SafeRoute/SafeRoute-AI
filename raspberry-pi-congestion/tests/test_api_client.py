import pytest

from raspberry_pi_congestion.api_client import AuthHeaderProvider, SafeRouteDeviceClient, SpringCongestionReporter
from raspberry_pi_congestion.models import CongestionObservation, DeviceCongestionConfig


class Response:
    def __init__(self, status, body=None):
        self.status_code = status
        self._body = body or {}

    def json(self):
        return self._body


def observation():
    return CongestionObservation("event-1", "session-uuid", "CCTV_001", 4.75, 8, 25, 1000, 6000, 6000, 1)


def test_request_contains_only_allowed_fields():
    body = observation().to_json()
    assert set(body) == {"eventId", "trainingSessionId", "cctvCode", "avgHeadcount", "peakHeadcount", "sampleCount", "windowStart", "windowEnd", "capturedAt", "monitoringImageKey", "configVersion"}
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
    assert not reporter.should_queue_failure("event-1")


def test_conflict_is_queued_with_same_id_and_requests_config_refresh():
    reporter = SpringCongestionReporter("http://server", max_retries=0,
                                        request=lambda *a, **k: Response(409), sleeper=lambda _: None)
    item = observation()
    assert not reporter.report(item)
    assert reporter.should_queue_failure(item.event_id)
    assert reporter.consume_config_refresh_request()


@pytest.mark.parametrize("outcomes", [[429, 200], [500, 200], [TimeoutError(), 200]])
def test_retryable_failures_are_bounded_and_payload_is_stable(outcomes):
    calls = []
    values = iter(outcomes)
    def request(*args, **kwargs):
        calls.append(kwargs["json"].copy())
        value = next(values)
        if isinstance(value, Exception):
            raise value
        return Response(value)
    reporter = SpringCongestionReporter("http://server", max_retries=1, request=request, sleeper=lambda _: None)
    assert reporter.report(observation())
    assert len(calls) == 2 and calls[0] == calls[1]


def test_configurable_auth_header():
    auth = AuthHeaderProvider("secret", "X-Custom-Token", "")
    assert auth.headers() == {"X-Custom-Token": "secret"}


def test_fetches_and_validates_backend_config():
    body = {
        "trainingActive": True, "trainingSessionId": "550e8400-e29b-41d4-a716-446655440000", "cctvCode": "CCTV_001",
        "monitoredAreaM2": 2.0, "configVersion": 3, "snapshotIntervalSec": 5,
        "targetInferenceFps": 5,
        "congestionThresholds": {"CAUTION_FROM": 2.0, "CROWDED_FROM": 3.0, "VERY_CROWDED_FROM": 5.0},
        "eventDetection": {"requiredConsecutiveFrames": 3, "recoveryConsecutiveFrames": 5, "cooldownSec": 30},
    }
    calls = []
    client = SafeRouteDeviceClient("http://server", AuthHeaderProvider("token"),
                                   request=lambda *a, **k: calls.append((a, k)) or Response(200, body))
    config = client.fetch_config("CCTV_001")
    assert config and config.training_session_id == "550e8400-e29b-41d4-a716-446655440000" and config.config_version == 3
    assert calls[0][1]["params"] == {"cctvCode": "CCTV_001"}
    assert calls[0][1]["headers"]["Authorization"] == "Bearer token"


def test_fetches_light_commands():
    body = {"commands": [{"commandId": "cmd-1", "lightCode": "LIGHT_001", "direction": "LEFT"}]}
    calls = []
    client = SafeRouteDeviceClient("http://server", AuthHeaderProvider("token"),
                                   request=lambda *a, **k: calls.append((a, k)) or Response(200, body))

    commands = client.fetch_light_commands("CCTV_001")

    assert commands == [{"commandId": "cmd-1", "lightCode": "LIGHT_001", "direction": "LEFT"}]
    assert calls[0][1]["params"] == {"cctvCode": "CCTV_001"}


def test_fetch_light_commands_returns_empty_list_on_failure():
    client = SafeRouteDeviceClient("http://server", AuthHeaderProvider("token"), max_retries=0,
                                   request=lambda *a, **k: Response(500), sleeper=lambda _: None)

    assert client.fetch_light_commands("CCTV_001") == []


def test_acks_light_command_success():
    calls = []
    client = SafeRouteDeviceClient("http://server", AuthHeaderProvider("token"),
                                   request=lambda *a, **k: calls.append(k) or Response(200))

    assert client.ack_light_command("cmd-1", success=True) is True
    assert calls[0]["json"] == {"success": True}


def test_acks_light_command_failure_includes_reason():
    calls = []
    client = SafeRouteDeviceClient("http://server", AuthHeaderProvider("token"),
                                   request=lambda *a, **k: calls.append(k) or Response(200))

    client.ack_light_command("cmd-1", success=False, fail_reason="relay timeout")

    assert calls[0]["json"] == {"success": False, "failReason": "relay timeout"}


@pytest.mark.parametrize("field", ["snapshotIntervalSec", "targetInferenceFps"])
def test_inactive_config_rejects_non_positive_runtime_values(field):
    payload = {
        "trainingActive": False,
        "trainingSessionId": None,
        "cctvCode": "CCTV_001",
        "configVersion": 1,
        field: 0,
    }

    with pytest.raises(ValueError, match="snapshot interval and target FPS"):
        DeviceCongestionConfig.from_json(payload)
