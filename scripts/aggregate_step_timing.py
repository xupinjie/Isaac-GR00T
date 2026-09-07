#!/usr/bin/env python3
"""Validate and aggregate per-rank step timing files from the benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def _load_rank(path: Path) -> list[dict]:
    json_path = path.with_suffix(".json")
    jsonl_path = path.with_suffix(".jsonl")
    if json_path.exists():
        return json.loads(json_path.read_text())["records"]
    if jsonl_path.exists():
        return [json.loads(line) for line in jsonl_path.read_text().splitlines() if line]
    raise FileNotFoundError(f"missing {json_path.name} and {jsonl_path.name}")


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def aggregate_variant(raw_root: Path, variant: str) -> tuple[list[dict], dict]:
    ranks = [_load_rank(raw_root / variant / f"rank_{rank:02d}") for rank in range(8)]
    expected_steps = list(range(1, 201))
    for rank, records in enumerate(ranks):
        steps = [int(record["step"]) for record in records]
        if steps != expected_steps:
            raise ValueError(
                f"{variant} rank {rank}: expected steps 1..200, got {len(steps)} records "
                f"from {steps[:1]} to {steps[-1:] if steps else []}"
            )

    aggregated: list[dict] = []
    cumulative_s = 0.0
    for index, step in enumerate(expected_steps):
        rank_records = [records[index] for records in ranks]
        critical_rank = max(range(8), key=lambda rank: rank_records[rank]["total_s"])
        data_wait_rank = max(range(8), key=lambda rank: rank_records[rank]["data_wait_s"])
        train_rank = max(range(8), key=lambda rank: rank_records[rank]["train_s"])
        critical = rank_records[critical_rank]
        loss = statistics.fmean(float(record["loss"]) for record in rank_records)
        cumulative_s += float(critical["total_s"])
        aggregated.append(
            {
                "variant": variant,
                "step": step,
                "critical_rank": critical_rank,
                "data_wait_s": float(critical["data_wait_s"]),
                "train_s": float(critical["train_s"]),
                "total_s": float(critical["total_s"]),
                "max_data_wait_rank": data_wait_rank,
                "max_data_wait_s": float(rank_records[data_wait_rank]["data_wait_s"]),
                "max_train_rank": train_rank,
                "max_train_s": float(rank_records[train_rank]["train_s"]),
                "loss": loss,
                "cumulative_s": cumulative_s,
            }
        )

    steady = aggregated[1:]
    totals = [row["total_s"] for row in steady]
    waits = [row["data_wait_s"] for row in steady]
    trains = [row["train_s"] for row in steady]
    steady_after_10 = aggregated[10:]
    totals_after_10 = [row["total_s"] for row in steady_after_10]
    max_waits_after_10 = [row["max_data_wait_s"] for row in steady_after_10]
    max_trains_after_10 = [row["max_train_s"] for row in steady_after_10]
    summary = {
        "variant": variant,
        "steps": len(aggregated),
        "cumulative_200_s": aggregated[-1]["cumulative_s"],
        "step_time_p50_s_excluding_step1": statistics.median(totals),
        "step_time_p95_s_excluding_step1": _percentile(totals, 0.95),
        "data_wait_p50_s_excluding_step1": statistics.median(waits),
        "data_wait_p95_s_excluding_step1": _percentile(waits, 0.95),
        "train_p50_s_excluding_step1": statistics.median(trains),
        "train_p95_s_excluding_step1": _percentile(trains, 0.95),
        "step1_total_s": aggregated[0]["total_s"],
        "step1_max_data_wait_s": aggregated[0]["max_data_wait_s"],
        "step_time_mean_s_excluding_first10": statistics.fmean(totals_after_10),
        "max_data_wait_mean_s_excluding_first10": statistics.fmean(max_waits_after_10),
        "max_data_wait_p50_s_excluding_first10": statistics.median(max_waits_after_10),
        "max_data_wait_p95_s_excluding_first10": _percentile(max_waits_after_10, 0.95),
        "max_train_mean_s_excluding_first10": statistics.fmean(max_trains_after_10),
        "final_loss_mean_8_ranks": aggregated[-1]["loss"],
    }
    return aggregated, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["cpu_torchcodec", "old_accv", "new_async_gop"],
    )
    args = parser.parse_args()

    rows: list[dict] = []
    summaries: list[dict] = []
    for variant in args.variants:
        variant_rows, summary = aggregate_variant(args.benchmark_root / "raw", variant)
        rows.extend(variant_rows)
        summaries.append(summary)

    args.benchmark_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.benchmark_root / "step_timing_comparison.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.benchmark_root / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
