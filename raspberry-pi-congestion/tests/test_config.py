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

    assert config.roi_config_path == "./config/roi/CCTV_ENTRANCE_01.json"
    assert config.model_path is None
    assert config.file_realtime
    assert config.file_fallback_fps == 30


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
