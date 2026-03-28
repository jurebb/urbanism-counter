import pytest
from pathlib import Path
from unittest.mock import patch
from batch_run import get_videos, build_jobs


DUMMY_CONFIG = {
    "output_dir": "output/",
    "max_parallel": 3,
    "variants": [
        {"name": "paths", "flags": ["--paths", "--no-boxes"]},
        {"name": "features_seating", "flags": ["--features", "--features-filter", "Seating"], "features_only": True},
        {"name": "plain", "flags": []},
    ],
}


class TestGetVideos:
    def test_videos_dir_picks_up_mp4s(self, tmp_path):
        (tmp_path / "a.mp4").touch()
        (tmp_path / "b.mp4").touch()
        (tmp_path / "ignore.txt").touch()
        config = {"videos_dir": str(tmp_path), "videos": []}
        videos = get_videos(config)
        names = [Path(v["path"]).name for v in videos]
        assert sorted(names) == ["a.mp4", "b.mp4"]

    def test_features_frame_override_applied(self, tmp_path):
        mp4 = tmp_path / "clip.mp4"
        mp4.touch()
        config = {
            "videos_dir": str(tmp_path),
            "videos": [{"path": str(mp4), "features_frame": 99}],
        }
        videos = get_videos(config)
        assert videos[0]["features_frame"] == 99

    def test_unlisted_video_defaults_to_frame_zero(self, tmp_path):
        (tmp_path / "clip.mp4").touch()
        config = {"videos_dir": str(tmp_path), "videos": []}
        videos = get_videos(config)
        assert videos[0]["features_frame"] == 0

    def test_no_videos_dir_uses_explicit_list(self, tmp_path):
        mp4 = tmp_path / "clip.mp4"
        mp4.touch()
        config = {"videos": [{"path": str(mp4), "features_frame": 5}]}
        videos = get_videos(config)
        assert len(videos) == 1
        assert videos[0]["features_frame"] == 5


class TestBuildJobs:
    def test_job_count_is_videos_times_variants(self, tmp_path):
        (tmp_path / "a.mp4").touch()
        (tmp_path / "b.mp4").touch()
        config = {**DUMMY_CONFIG, "videos_dir": str(tmp_path), "videos": []}
        videos = get_videos(config)
        jobs = build_jobs(config, videos)
        assert len(jobs) == 2 * len(DUMMY_CONFIG["variants"])

    def test_features_only_flag_added_for_features_only_variant(self, tmp_path):
        (tmp_path / "clip.mp4").touch()
        config = {**DUMMY_CONFIG, "videos_dir": str(tmp_path), "videos": []}
        jobs = build_jobs(config, get_videos(config))
        features_job = next(j for j in jobs if j["variant"] == "features_seating")
        assert "--features-only" in features_job["flags"]

    def test_features_only_flag_absent_for_normal_variant(self, tmp_path):
        (tmp_path / "clip.mp4").touch()
        config = {**DUMMY_CONFIG, "videos_dir": str(tmp_path), "videos": []}
        jobs = build_jobs(config, get_videos(config))
        paths_job = next(j for j in jobs if j["variant"] == "paths")
        assert "--features-only" not in paths_job["flags"]

    def test_features_frame_added_when_nonzero(self, tmp_path):
        mp4 = tmp_path / "clip.mp4"
        mp4.touch()
        config = {**DUMMY_CONFIG, "videos_dir": str(tmp_path),
                  "videos": [{"path": str(mp4), "features_frame": 42}]}
        jobs = build_jobs(config, get_videos(config))
        for job in jobs:
            assert "--features-frame" in job["flags"]
            idx = job["flags"].index("--features-frame")
            assert job["flags"][idx + 1] == "42"

    def test_features_frame_absent_when_zero(self, tmp_path):
        (tmp_path / "clip.mp4").touch()
        config = {**DUMMY_CONFIG, "videos_dir": str(tmp_path), "videos": []}
        jobs = build_jobs(config, get_videos(config))
        for job in jobs:
            assert "--features-frame" not in job["flags"]

    def test_output_dir_per_variant(self, tmp_path):
        (tmp_path / "clip.mp4").touch()
        config = {**DUMMY_CONFIG, "output_dir": str(tmp_path / "out"),
                  "videos_dir": str(tmp_path), "videos": []}
        jobs = build_jobs(config, get_videos(config))
        paths_job = next(j for j in jobs if j["variant"] == "paths")
        assert "paths" in paths_job["output_dir"]
        assert "clip" in paths_job["output_dir"]
