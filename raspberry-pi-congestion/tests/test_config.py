from raspberry_pi_congestion.config import AppConfig


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
