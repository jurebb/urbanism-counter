"""
Batch runner — processes all videos in batch.yaml against all variants in parallel.

Usage:
    python batch_run.py batch.yaml
    python batch_run.py batch.yaml --dry-run   # print jobs without running
"""

import os
import sys
import yaml
import subprocess
import concurrent.futures
from pathlib import Path
from datetime import datetime


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_videos(config: dict) -> list[dict]:
    """Return list of {path, features_frame} dicts from videos_dir + per-video overrides."""
    overrides = {
        Path(v["path"]).expanduser().resolve(): v
        for v in config.get("videos", [])
    }

    import glob as _glob
    videos = []
    videos_glob = config.get("videos_dir")
    if videos_glob:
        videos_glob = str(Path(videos_glob).expanduser())
        # Plain directory → match all mp4s in it
        if not any(c in videos_glob for c in ("*", "?", "[")):
            videos_glob = videos_glob.rstrip("/") + "/*.mp4"
        for mp4_str in sorted(_glob.glob(videos_glob, recursive=True)):
            mp4 = Path(mp4_str).resolve()
            override = overrides.get(mp4, {})
            videos.append({
                "path": str(mp4),
                "features_frame": override.get("features_frame", 0),
            })

    # Include explicitly listed videos not already picked up by the glob
    added = {v["path"] for v in videos}
    for v in config.get("videos", []):
        p = Path(v["path"]).expanduser().resolve()
        if str(p) not in added:
            videos.append({
                "path": str(p),
                "features_frame": v.get("features_frame", 0),
            })

    return videos


def build_jobs(config: dict, videos: list[dict]) -> list[dict]:
    """Build the full job list: one entry per (video, variant) pair."""
    output_root = Path(config["output_dir"]).expanduser()
    jobs = []

    for video in videos:
        video_path = Path(video["path"])
        video_name = video_path.stem
        features_frame = video.get("features_frame", 0)

        for variant in config["variants"]:
            variant_dir = output_root / video_name / variant["name"]
            flags = [str(f) for f in variant.get("flags", [])]

            if features_frame > 0:
                flags += ["--features-frame", str(features_frame)]

            if variant.get("features_only"):
                flags.append("--features-only")

            flags += ["--output-dir", str(variant_dir)]

            jobs.append({
                "video":      str(video_path),
                "variant":    variant["name"],
                "video_name": video_name,
                "output_dir": str(variant_dir),
                "flags":      flags,
            })

    return jobs


def format_duration(seconds: int) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def print_summary(results: list, start: datetime, end: datetime) -> None:
    elapsed = int((end - start).total_seconds())
    ok = sum(1 for r in results if r["status"] == "OK")
    print(f"\n{'='*50}")
    print(f"Started:  {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished: {end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {format_duration(elapsed)}")
    print(f"Jobs:     {ok}/{len(results)} succeeded")
    for r in sorted(results, key=lambda r: r["label"]):
        print(f"  [{r['status']:12}] {r['label']} ({format_duration(r['elapsed'])})")


def run_job(job: dict) -> dict:
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "tracker_ocsort.py", job["video"]] + job["flags"]
    label = f"{job['video_name']}/{job['variant']}"

    start = datetime.now()
    print(f"[START] {label}", flush=True)

    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    elapsed = int((datetime.now() - start).total_seconds())
    status = "OK" if result.returncode == 0 else f"FAIL({result.returncode})"
    print(f"[{status}] {label} — {format_duration(elapsed)}", flush=True)
    return {"label": label, "status": status, "elapsed": elapsed}


def main():
    dry_run = "--dry-run" in sys.argv
    config_path = next((a for a in sys.argv[1:] if not a.startswith("--")), None)
    if not config_path:
        print("Usage: python batch_run.py batch.yaml [--dry-run]")
        sys.exit(1)

    config = load_config(config_path)
    videos = get_videos(config)
    jobs = build_jobs(config, videos)
    max_parallel = config.get("max_parallel", 3)

    print(f"\nBatch: {len(jobs)} jobs ({len(videos)} videos × {len(config['variants'])} variants), {max_parallel} parallel")
    if dry_run:
        print("\n--- DRY RUN ---")
        for job in jobs:
            print(f"  {job['video_name']}/{job['variant']}")
            print(f"    cmd: tracker_ocsort.py {job['video']} {' '.join(job['flags'])}")
        return

    print()
    batch_start = datetime.now()
    print(f"Started:  {batch_start.strftime('%Y-%m-%d %H:%M:%S')}")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = {pool.submit(run_job, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                job = futures[future]
                label = f"{job['video_name']}/{job['variant']}"
                print(f"[ERROR] {label}: {e}", flush=True)
                results.append({"label": label, "status": f"ERROR", "elapsed": 0})

    batch_end = datetime.now()
    print_summary(results, batch_start, batch_end)


if __name__ == "__main__":
    main()
