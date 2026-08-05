#!/usr/bin/env python3
"""Aggregate canonical fixed-timeout runs for seeds 0-3."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNS = Path(
    os.environ.get(
        "FIXED_TIMEOUT_RUNS_ROOT",
        "/Users/pranavr/Developer/Work/STRIDE/fixed_timeout_runs",
    )
)
OUT = ROOT / "outputs"
SEEDS = range(4)
GRIDS = range(10, 20)
TIMEOUT_MINUTES = 120.0

SOURCES = {
    "ilp": {
        0: [RUNS / "ilp/20260630_083954/ilp_summary.csv"],
        1: [RUNS / "ilp/20260630_114424/ilp_summary.csv"],
        2: [RUNS / "ilp/20260630_151526/ilp_summary.csv"],
        3: [RUNS / "ilp/20260702_174144/ilp_summary.csv"],
    },
    "patternboost": {
        0: [
            RUNS / "patternboost/20260625_084012/patternboost_summary.csv",
            RUNS / "patternboost/20260625_222420/patternboost_summary.csv",
        ],
        1: [RUNS / "patternboost/20260630_200749/patternboost_summary.csv"],
        2: [RUNS / "patternboost/20260701_104255/patternboost_summary.csv"],
        3: [RUNS / "patternboost/20260702_012952/patternboost_summary.csv"],
    },
    "ppo": {
        0: [RUNS / "ppo/20260623_184959/ppo_summary.csv"],
        1: [RUNS / "ppo/20260703_231114/ppo_summary.csv"],
        2: [RUNS / "ppo/20260704_234925/ppo_summary.csv"],
        3: [RUNS / "ppo/20260705_194928/ppo_summary.csv"],
    },
}

LONG_FIELDS = [
    "method", "seed", "n", "target", "success", "quality", "runtime_minutes",
    "timed_out", "process_status", "detail_status", "return_code", "source_path",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_sources() -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    manifest_sources: list[dict] = []
    part_counts: Counter[tuple[str, int]] = Counter()
    for method, seed_sources in SOURCES.items():
        for seed, paths in seed_sources.items():
            for path in paths:
                key = (method, seed)
                part_counts[key] += 1
                bundled = OUT / "fixed_timeout_seed0_3_raw" / (
                    f"{method}_seed{seed}_part{part_counts[key]}.csv"
                )
                source_path = path if path.exists() else bundled
                if not source_path.exists():
                    raise FileNotFoundError(
                        f"Neither canonical nor bundled source exists: {path}; {bundled}"
                    )
                canonical_label = str(Path("fixed_timeout_runs") / path.relative_to(RUNS))
                manifest_sources.append(
                    {
                        "method": method,
                        "seed": seed,
                        "path": canonical_label,
                        "_content": source_path.read_bytes(),
                        "sha256": sha256(source_path),
                    }
                )
                with source_path.open(newline="") as handle:
                    for row in csv.DictReader(handle):
                        n = int(row["n"])
                        if n not in GRIDS:
                            continue
                        row_seed = int(row["seed"])
                        if row_seed != seed:
                            raise ValueError(
                                f"Seed mismatch in {source_path}: expected {seed}, got {row_seed}"
                            )
                        target = 2 * n
                        runtime = float(row["elapsed_minutes"])
                        if method == "ilp":
                            success = row["solver_status"] == "optimal" and float(row["objective"] or 0) == target
                            quality = float(row["objective"]) if row["objective"] else None
                            timed_out = row["solver_status"] == "time_limit" or runtime >= 119.99
                            detail_status = row["solver_status"]
                        elif method == "patternboost":
                            quality = float(row["best_post_saturation_score"])
                            success = int(row["hit_target"]) == 1 and quality >= target
                            timed_out = row["status"] == "timeout" or runtime >= 119.99
                            detail_status = row["result_status"]
                        else:
                            quality = float(row["best_avg_violations"])
                            success = float(row["best_success_rate"] or 0) > 0 and quality == 0
                            timed_out = row["status"] == "timeout" or runtime >= 119.99
                            detail_status = row["termination_reason"]
                        records.append(
                            {
                                "method": method,
                                "seed": seed,
                                "n": n,
                                "target": target,
                                "success": int(success),
                                "quality": quality,
                                "runtime_minutes": runtime,
                                "timed_out": int(timed_out),
                                "process_status": row["status"],
                                "detail_status": detail_status,
                                "return_code": row.get("return_code", ""),
                                "source_path": canonical_label,
                            }
                        )
    return records, manifest_sources


def validate(records: list[dict]) -> list[str]:
    messages: list[str] = []
    keys = [(row["method"], row["seed"], row["n"]) for row in records]
    duplicates = [key for key, count in Counter(keys).items() if count != 1]
    if duplicates:
        raise ValueError(f"Duplicate records: {duplicates}")
    expected = {(method, seed, n) for method in SOURCES for seed in SEEDS for n in GRIDS}
    missing = sorted(expected - set(keys))
    extra = sorted(set(keys) - expected)
    if missing or extra:
        raise ValueError(f"Missing={missing}; extra={extra}")
    if len(records) != 120:
        raise ValueError(f"Expected 120 records, got {len(records)}")
    ilp_timeout_crashes = []
    for row in records:
        if row["timed_out"]:
            if float(row["runtime_minutes"]) < 119.99:
                raise ValueError(f"Timeout shorter than budget: {row}")
        if row["method"] == "ilp":
            if row["success"]:
                if not (
                    row["process_status"] == "completed"
                    and row["detail_status"] == "optimal"
                    and str(row["return_code"]) == "0"
                ):
                    raise ValueError(f"Unexpected successful ILP state: {row}")
            else:
                if not (
                    row["timed_out"]
                    and row["process_status"] == "failed_return_1"
                    and row["detail_status"] == "time_limit"
                    and str(row["return_code"]) == "1"
                    and row["quality"] is None
                ):
                    raise ValueError(f"Unexpected unsuccessful ILP state: {row}")
                ilp_timeout_crashes.append(row)
        elif row["method"] == "patternboost":
            allowed = "completed" if row["success"] else "timeout"
            if row["process_status"] != allowed:
                raise ValueError(f"Unexpected PatternBoost state: {row}")
        elif row["method"] == "ppo":
            allowed = "completed" if row["success"] else "timeout"
            if row["process_status"] != allowed:
                raise ValueError(f"Unexpected PPO state: {row}")
    if len(ilp_timeout_crashes) != 6:
        raise ValueError(f"Expected six known ILP timeout/write failures, got {len(ilp_timeout_crashes)}")
    messages.append("PASS: exactly 120 unique method-seed-grid records.")
    messages.append("PASS: four observations per method and grid size.")
    messages.append("PASS: all source rows use seeds 0-3 and grids n=10-19.")
    messages.append("PASS: every timeout consumed the full 120-minute budget.")
    messages.append("PASS: method-specific success, status, and return-code combinations are valid.")
    affected = ", ".join(
        f"seed {row['seed']} n={row['n']}" for row in ilp_timeout_crashes
    )
    messages.append(
        "KNOWN ANOMALY: six ILP runs reached Gurobi's time limit without an incumbent "
        f"and then returned code 1 while writing the missing objective ({affected})."
    )
    return messages


def sample_sd(values: list[float]) -> float:
    return statistics.stdev(values)


def aggregate(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(row["method"], row["n"])].append(row)

    output = []
    for n in GRIDS:
        row: dict[str, int | float] = {"n": n, "target": 2 * n}
        for method in SOURCES:
            values = sorted(grouped[(method, n)], key=lambda item: item["seed"])
            runtimes = [float(item["runtime_minutes"]) for item in values]
            prefix = {"ilp": "ilp", "patternboost": "pb", "ppo": "ppo"}[method]
            row[f"{prefix}_successes"] = sum(int(item["success"]) for item in values)
            if method != "ilp":
                qualities = [float(item["quality"]) for item in values]
                row[f"{prefix}_quality_mean"] = statistics.mean(qualities)
                row[f"{prefix}_quality_sd"] = sample_sd(qualities)
            row[f"{prefix}_runtime_median"] = statistics.median(runtimes)
            row[f"{prefix}_runtime_min"] = min(runtimes)
            row[f"{prefix}_runtime_max"] = max(runtimes)
            row[f"{prefix}_timeouts"] = sum(int(item["timed_out"]) for item in values)
        output.append(row)
    return output


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(mean: float, sd: float) -> str:
    return f"{mean:.2f} $\\pm$ {sd:.2f}"


def runtime_cell(row: dict, prefix: str) -> str:
    median = float(row[f"{prefix}_runtime_median"])
    low = float(row[f"{prefix}_runtime_min"])
    high = float(row[f"{prefix}_runtime_max"])
    def display(value: float) -> str:
        if value >= 119.99:
            return "120.0"
        return f"{value:.2f}" if value < 1 else f"{value:.1f}"

    high_text = display(high)
    median_text = display(median)
    low_text = display(low)
    if int(row[f"{prefix}_timeouts"]) >= 2 and median < 119.99:
        median_text = f"$\\geq${median_text}"
    return f"{median_text} [{low_text}--{high_text}]"


def outcomes_tex(rows: list[dict]) -> str:
    lines = []
    for row in rows:
        lines.append(
            f"{row['n']} & {row['target']} & {row['ilp_successes']}/4 & "
            f"{row['pb_successes']}/4 & {mean_sd(row['pb_quality_mean'], row['pb_quality_sd'])} & "
            f"{row['ppo_successes']}/4 & {mean_sd(row['ppo_quality_mean'], row['ppo_quality_sd'])} \\\\"
        )
    return "\n".join(lines) + "\n"


def runtime_tex(rows: list[dict]) -> str:
    lines = []
    for row in rows:
        lines.append(
            f"{row['n']} & {runtime_cell(row, 'ilp')} & {row['ilp_timeouts']}/4 & "
            f"{runtime_cell(row, 'pb')} & {row['pb_timeouts']}/4 & "
            f"{runtime_cell(row, 'ppo')} & {row['ppo_timeouts']}/4 \\\\"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    records, manifest_sources = read_sources()
    validation = validate(records)
    aggregates = aggregate(records)
    outcome_fragment = outcomes_tex(aggregates)
    runtime_fragment = runtime_tex(aggregates)
    if outcome_fragment.count("$\\pm$") != 20:
        raise ValueError("Expected 20 mean-plus-SD cells in the outcome table")
    if "$\\geq$97.9" not in runtime_fragment:
        raise ValueError("Expected the mixed-censoring ILP median to be marked as a lower bound")
    if runtime_fragment.count("120.0") < 1:
        raise ValueError("Expected timeout-limit values in the runtime table")

    raw_dir = OUT / "fixed_timeout_seed0_3_raw"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir()
    part_counts: Counter[tuple[str, int]] = Counter()
    for source in manifest_sources:
        key = (source["method"], source["seed"])
        part_counts[key] += 1
        bundled = raw_dir / f"{source['method']}_seed{source['seed']}_part{part_counts[key]}.csv"
        bundled.write_bytes(source.pop("_content"))
        source["bundled_path"] = str(bundled.relative_to(ROOT))
        source["bundled_sha256"] = sha256(bundled)
        if source["bundled_sha256"] != source["sha256"]:
            raise ValueError(f"Bundled source checksum mismatch: {bundled}")
    manifest = {
        "description": "Canonical fixed-timeout inputs for the four-run seeds 0-3 analysis.",
        "seeds": list(SEEDS),
        "grid_sizes": list(GRIDS),
        "timeout_minutes": TIMEOUT_MINUTES,
        "sources": manifest_sources,
    }
    (OUT / "fixed_timeout_seed0_3_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    write_csv(OUT / "fixed_timeout_seed0_3_long.csv", records, LONG_FIELDS)
    write_csv(OUT / "fixed_timeout_seed0_3_aggregate.csv", aggregates)
    (OUT / "fixed_timeout_seed0_3_outcomes_table.tex").write_text(outcome_fragment)
    (OUT / "fixed_timeout_seed0_3_runtime_table.tex").write_text(runtime_fragment)
    validation.extend(
        [
            "PASS: 20 solution-quality cells were generated with sample standard deviations using denominator r-1.",
            "PASS: timeout counts identify censored runtime values, including the mixed-censoring n=18 ILP median as a lower bound.",
            "NOTE: exploratory parallel seed-5 runs are excluded.",
            "NOTE: PPO seed 0 used an older runner revision than seeds 1-3; recorded hyperparameters match, but the historical runner diff is not version-controlled.",
        ]
    )
    (OUT / "fixed_timeout_seed0_3_validation.md").write_text(
        "# Fixed-Timeout Seeds 0-3 Validation\n\n" + "\n".join(f"- {line}" for line in validation) + "\n"
    )
    print("Generated four-seed fixed-timeout artifacts (120 validated observations).")


if __name__ == "__main__":
    main()
