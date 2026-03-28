import pytest
from pathlib import Path
from unittest.mock import patch
from datetime import datetime
from batch_run import get_videos, build_jobs, format_duration, print_summary


DUMMY_CONFIG = {
    "output_dir": "output/",
    "max_parallel": 3,
    "variants": [
        {"name": "paths", "flags": ["--paths", "--no-boxes"]},
        {"name": "features_seating", "flags": ["--features", "--features-filter", "Seating"], "features_only": True},
        {"name": "plain", "flags": []},
    ],
}


class TestFormatDuration:
    def test_seconds_only(self):
        assert format_duration(45) == "45s"

    def test_minutes_and_seconds(self):
        assert format_duration(125) == "2m 5s"

    def test_hours_minutes_seconds(self):
        assert format_duration(3661) == "1h 1m 1s"

    def test_long_batch_12_hours(self):
        assert format_duration(12 * 3600) == "12h 0m 0s"


class TestPrintSummary:
    def test_prints_start_finish_duration(self, capsys):
        start = datetime(2026, 3, 28, 10, 0, 0)
        end   = datetime(2026, 3, 28, 10, 5, 30)
        results = [{"label": "vid/paths", "status": "OK", "elapsed": 330}]
        print_summary(results, start, end)
        out = capsys.readouterr().out
        assert "2026-03-28 10:00:00" in out
        assert "2026-03-28 10:05:30" in out
        assert "5m 30s" in out

    def test_counts_successes(self, capsys):
        start = datetime(2026, 3, 28, 9, 0, 0)
        end   = datetime(2026, 3, 28, 9, 1, 0)
        results = [
            {"label": "a/paths", "status": "OK",      "elapsed": 30},
            {"label": "b/paths", "status": "FAIL(1)", "elapsed": 10},
        ]
        print_summary(results, start, end)
        out = capsys.readouterr().out
        assert "1/2" in out


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
        assert len(videos) == 1, "override entry must not duplicate the glob result"
        assert videos[0]["features_frame"] == 99

    def test_unlisted_video_defaults_to_frame_zero(self, tmp_path):
        (tmp_path / "clip.mp4").touch()
        config = {"videos_dir": str(tmp_path), "videos": []}
        videos = get_videos(config)
        assert videos[0]["features_frame"] == 0

    def test_glob_pattern_traverses_subdirs(self, tmp_path):
        sub1 = tmp_path / "park1"
        sub2 = tmp_path / "park2"
        sub1.mkdir(); sub2.mkdir()
        (sub1 / "clip_75fps.mp4").touch()
        (sub2 / "clip_75fps.mp4").touch()
        (sub1 / "clip_other.mp4").touch()  # should be excluded
        config = {"videos_dir": str(tmp_path / "**" / "*75fps.mp4"), "videos": []}
        videos = get_videos(config)
        names = [Path(v["path"]).name for v in videos]
        assert sorted(names) == ["clip_75fps.mp4", "clip_75fps.mp4"]
        assert all("other" not in n for n in names)

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
