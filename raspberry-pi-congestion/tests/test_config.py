import pytest

from raspberry_pi_congestion.config import AppConfig, ConfigError


def test_default_paths_follow_project_directory_structure():
    config = AppConfig.from_env(
        {
            "RUN_MODE": "dry-run",
            "VIDEO_SOURCE": "./sample_videos/test.mp4",
            "CCTV_CODE": "CCTV_ENTRANCE_01",
        }
    )

    assert config.model_path is None
    assert config.file_realtime
    assert config.file_fallback_fps == 30
    assert not config.show_preview


def test_show_preview_can_be_enabled_from_env():
    config = AppConfig.from_env(
        {
            "RUN_MODE": "dry-run",
            "VIDEO_SOURCE": "./sample_videos/test.mp4",
            "CCTV_CODE": "CCTV_001",
            "SHOW_PREVIEW": "true",
        }
    )

    assert config.show_preview


def test_relay_is_disabled_by_default():
    config = AppConfig.from_env(
        {
            "RUN_MODE": "dry-run",
            "VIDEO_SOURCE": "./sample_videos/test.mp4",
            "CCTV_CODE": "CCTV_001",
        }
    )

    assert config.relay_host is None
    assert config.relay_port is None


def test_relay_host_and_port_are_parsed():
    config = AppConfig.from_env(
        {
            "RUN_MODE": "dry-run",
            "VIDEO_SOURCE": "./sample_videos/test.mp4",
            "CCTV_CODE": "CCTV_001",
            "RELAY_HOST": "192.168.0.81",
            "RELAY_PORT": "5000",
        }
    )

    assert config.relay_host == "192.168.0.81"
    assert config.relay_port == 5000
    assert config.relay_poll_interval_sec == 2


def test_relay_host_without_port_raises():
    env = {
        "RUN_MODE": "dry-run",
        "VIDEO_SOURCE": "./sample_videos/test.mp4",
        "CCTV_CODE": "CCTV_001",
        "RELAY_HOST": "192.168.0.81",
    }

    with pytest.raises(ConfigError, match="RELAY_PORT"):
        AppConfig.from_env(env)


@pytest.mark.parametrize("key", ["CONFIG_POLL_ACTIVE_SEC", "CONFIG_POLL_INACTIVE_SEC"])
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_poll_intervals_must_be_finite_and_positive(key, value):
    env = {
        "RUN_MODE": "dry-run",
        "VIDEO_SOURCE": "./sample_videos/test.mp4",
        "CCTV_CODE": "CCTV_001",
        key: value,
    }

    with pytest.raises(ConfigError, match=key):
        AppConfig.from_env(env)
