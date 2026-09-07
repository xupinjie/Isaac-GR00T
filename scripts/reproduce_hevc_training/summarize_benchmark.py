#!/usr/bin/env python3
"""Combine Trainer metrics and per-step timing into a compact comparison."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def trainer_metrics(path: Path) -> dict:
    matches = []
    for raw_line in path.read_text(errors="replace").splitlines():
        line = ANSI.sub("", raw_line)
        if "'train_runtime':" not in line:
            continue
        start = line.find("{")
        if start < 0:
            continue
        try:
            payload = ast.literal_eval(line[start:])
        except (SyntaxError, ValueError):
            continue
        if isinstance(payload, dict) and "train_runtime" in payload:
            matches.append(payload)
    if not matches:
        raise RuntimeError(f"No final Trainer metrics found in {path}")
    return matches[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument("--cpu-variant", default="cpu_torchcodec")
    parser.add_argument("--async-variant", default="async_gop")
    args = parser.parse_args()

    step_summaries = {
        item["variant"]: item
        for item in json.loads((args.benchmark_root / "summary.json").read_text())
    }
    variants = [args.cpu_variant, args.async_variant]
    combined = {}
    for variant in variants:
        process_wall_path = args.benchmark_root / f"{variant}_process_wall_seconds.txt"
        combined[variant] = {
            "trainer": trainer_metrics(args.benchmark_root / "logs" / f"{variant}.log"),
            "step_timing": step_summaries[variant],
            "process_wall_seconds": (
                int(process_wall_path.read_text().strip()) if process_wall_path.exists() else None
            ),
        }

    cpu = combined[args.cpu_variant]
    asynchronous = combined[args.async_variant]
    comparison = {
        "trainer_runtime_speedup": (
            cpu["trainer"]["train_runtime"] / asynchronous["trainer"]["train_runtime"]
        ),
        "throughput_speedup": (
            asynchronous["trainer"]["train_samples_per_second"]
            / cpu["trainer"]["train_samples_per_second"]
        ),
        "steady_step_speedup_excluding_first10": (
            cpu["step_timing"]["step_time_mean_s_excluding_first10"]
            / asynchronous["step_timing"]["step_time_mean_s_excluding_first10"]
        ),
        "max_rank_data_wait_speedup_excluding_first10": (
            cpu["step_timing"]["max_data_wait_mean_s_excluding_first10"]
            / asynchronous["step_timing"]["max_data_wait_mean_s_excluding_first10"]
        ),
        "train_loss_absolute_difference": abs(
            cpu["trainer"]["train_loss"] - asynchronous["trainer"]["train_loss"]
        ),
    }
    output = {"variants": combined, "comparison": comparison}
    target = args.benchmark_root / "reproduction_summary.json"
    target.write_text(json.dumps(output, indent=2) + "\n")

    print("| Metric | CPU TorchCodec | Async GOP | Speedup |")
    print("| --- | ---: | ---: | ---: |")
    print(
        f"| Trainer runtime | {cpu['trainer']['train_runtime']:.2f} s | "
        f"{asynchronous['trainer']['train_runtime']:.2f} s | "
        f"{comparison['trainer_runtime_speedup']:.2f}x |"
    )
    print(
        f"| Throughput | {cpu['trainer']['train_samples_per_second']:.2f} samples/s | "
        f"{asynchronous['trainer']['train_samples_per_second']:.2f} samples/s | "
        f"{comparison['throughput_speedup']:.2f}x |"
    )
    print(
        "| Mean step, steps 11-200 | "
        f"{cpu['step_timing']['step_time_mean_s_excluding_first10']:.4f} s | "
        f"{asynchronous['step_timing']['step_time_mean_s_excluding_first10']:.4f} s | "
        f"{comparison['steady_step_speedup_excluding_first10']:.2f}x |"
    )
    print(
        f"| Train loss | {cpu['trainer']['train_loss']:.6f} | "
        f"{asynchronous['trainer']['train_loss']:.6f} | - |"
    )


if __name__ == "__main__":
    main()
