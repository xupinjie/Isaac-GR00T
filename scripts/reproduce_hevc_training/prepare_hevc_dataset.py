#!/usr/bin/env python3
"""Convert the LeRobot videos to HEVC while preserving training-relevant properties."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


VIDEO_SUFFIX = ".mp4"


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"Required executable is not on PATH: {name}")
    return path


def probe_video(path: Path) -> dict:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_frames,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        capture=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"Expected one video stream in {path}, found {len(streams)}")
    stream = streams[0]
    frames = stream.get("nb_read_frames") or stream.get("nb_frames")
    if frames in {None, "N/A"}:
        raise RuntimeError(f"Could not determine frame count for {path}")
    stream["frames"] = int(frames)

    return stream


def probe_gop(path: Path) -> int:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=key_frame",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture=True,
    )
    flags = []
    for line in result.stdout.splitlines():
        value = line.split(",", 1)[0].strip()
        if value in {"0", "1"}:
            flags.append(int(value))
    keyframes = [index for index, flag in enumerate(flags) if flag == 1]
    if not flags or not keyframes or keyframes[0] != 0:
        raise RuntimeError(f"Could not determine a valid GOP structure for {path}")
    if len(keyframes) == 1:
        return len(flags)
    return max(b - a for a, b in zip(keyframes, keyframes[1:]))


def copy_non_video_content(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name in {".git", "videos"}:
            continue
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def same_rate(left: str, right: str) -> bool:
    return Fraction(left) == Fraction(right)


def validate_pair(source: Path, destination: Path, expected_gop: int) -> dict:
    source_info = probe_video(source)
    output_info = probe_video(destination)
    errors = []
    if output_info["codec_name"] not in {"hevc", "h265"}:
        errors.append(f"codec={output_info['codec_name']}")
    for key in ("width", "height", "pix_fmt", "frames"):
        if source_info.get(key) != output_info.get(key):
            errors.append(f"{key}: {source_info.get(key)} -> {output_info.get(key)}")
    if not same_rate(source_info["avg_frame_rate"], output_info["avg_frame_rate"]):
        errors.append(
            f"fps: {source_info['avg_frame_rate']} -> {output_info['avg_frame_rate']}"
        )
    output_gop = probe_gop(destination)
    if output_gop != expected_gop:
        errors.append(f"GOP={output_gop}, expected {expected_gop}")
    if errors:
        raise RuntimeError(f"Validation failed for {destination}: " + "; ".join(errors))
    return {
        "frames": output_info["frames"],
        "bytes": destination.stat().st_size,
    }


def transcode_one(
    source: Path,
    destination: Path,
    *,
    gop_size: int,
    preset: str,
    crf: int,
    x265_pools: int,
) -> tuple[str, dict]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            return "reused", validate_pair(source, destination, gop_size)
        except Exception:
            destination.unlink()

    temporary = destination.with_name(destination.name + ".partial.mp4")
    temporary.unlink(missing_ok=True)
    x265_params = ":".join(
        [
            f"keyint={gop_size}",
            f"min-keyint={gop_size}",
            "scenecut=0",
            "open-gop=0",
            "bframes=0",
            f"pools={x265_pools}",
            "frame-threads=2",
            "log-level=error",
        ]
    )
    try:
        run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-an",
                "-map_metadata",
                "0",
                "-fps_mode",
                "passthrough",
                "-c:v",
                "libx265",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-g",
                str(gop_size),
                "-keyint_min",
                str(gop_size),
                "-sc_threshold",
                "0",
                "-x265-params",
                x265_params,
                "-tag:v",
                "hvc1",
                "-movflags",
                "+faststart",
                str(temporary),
            ]
        )
        result = validate_pair(source, temporary, gop_size)
        temporary.replace(destination)
        return "encoded", result
    finally:
        temporary.unlink(missing_ok=True)


def update_metadata(destination: Path) -> None:
    info_path = destination / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    for feature in info.get("features", {}).values():
        if feature.get("dtype") == "video":
            feature.setdefault("info", {})["video.codec"] = "hevc"
    temporary = info_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(info, indent=4) + "\n")
    temporary.replace(info_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--jobs", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--gop-size", type=int, default=2)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--x265-pools", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    if source == destination or source in destination.parents:
        raise ValueError("Destination must not be the source or a child of the source")
    if not (source / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset: {source}")
    require_tool("ffmpeg")
    require_tool("ffprobe")

    videos = sorted((source / "videos").rglob(f"*{VIDEO_SUFFIX}"))
    if not videos:
        raise RuntimeError(f"No MP4 files found under {source / 'videos'}")
    first_gop = probe_gop(videos[0])
    if first_gop != args.gop_size:
        raise RuntimeError(
            f"Source GOP is {first_gop}, but --gop-size={args.gop_size}. "
            "Pass the source GOP explicitly; do not silently change it."
        )

    copy_non_video_content(source, destination)
    started = time.time()
    totals = {"encoded": 0, "reused": 0, "frames": 0, "bytes": 0}
    failures = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {}
        for video in videos:
            relative = video.relative_to(source)
            future = executor.submit(
                transcode_one,
                video,
                destination / relative,
                gop_size=args.gop_size,
                preset=args.preset,
                crf=args.crf,
                x265_pools=args.x265_pools,
            )
            futures[future] = relative
        for completed, future in enumerate(as_completed(futures), start=1):
            relative = futures[future]
            try:
                status, result = future.result()
                totals[status] += 1
                totals["frames"] += result["frames"]
                totals["bytes"] += result["bytes"]
            except Exception as exc:
                failures.append(f"{relative}: {exc}")
            if completed == 1 or completed % 25 == 0 or completed == len(videos):
                print(
                    f"[{completed}/{len(videos)}] encoded={totals['encoded']} "
                    f"reused={totals['reused']} failed={len(failures)}",
                    flush=True,
                )
    if failures:
        print("\n".join(failures), file=sys.stderr)
        raise RuntimeError(f"{len(failures)} video(s) failed")

    update_metadata(destination)
    ffmpeg_version = run(["ffmpeg", "-version"], capture=True).stdout.splitlines()[0]
    manifest_path = destination / "hevc_transcode_manifest.json"
    manifest = {
        "source": str(source),
        "destination": str(destination),
        "video_count": len(videos),
        "codec": "hevc",
        "encoder": "libx265",
        "preset": args.preset,
        "crf": args.crf,
        "gop_size": args.gop_size,
        "pixel_format": "yuv420p",
        "b_frames": 0,
        "jobs": args.jobs,
        "x265_pools_per_job": args.x265_pools,
        "elapsed_seconds": time.time() - started,
        "encoded_videos": totals["encoded"],
        "reused_videos": totals["reused"],
        "total_decoded_frames_across_cameras": totals["frames"],
        "output_video_bytes": totals["bytes"],
        "ffmpeg_version": ffmpeg_version,
        "validation": "codec, resolution, pixel format, frame rate, frame count, and GOP size",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
