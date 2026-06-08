from __future__ import annotations

import csv
import json
import os
import statistics
from collections import defaultdict
from dataclasses import fields, replace
from pathlib import Path
from typing import Sequence

from benchmark_models import (
    BenchmarkMetadata,
    BenchmarkRun,
    BenchmarkRunDict,
    BenchmarkSummary,
    BenchmarkSummaryDict,
)


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of no values")
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[
        int(quantile * 100) - 1
    ]


def relative_change(current: float, baseline: float) -> float:
    if baseline == 0:
        raise ValueError("baseline must be non-zero")
    return (current / baseline - 1.0) * 100.0


def _optional_median(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.median(present) if present else None


def _optional_p95(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return percentile(present, 0.95) if present else None


def aggregate_runs(runs: Sequence[BenchmarkRun]) -> list[BenchmarkSummary]:
    groups: dict[
        tuple[str, str, str, str], list[BenchmarkRun]
    ] = defaultdict(list)
    for run in runs:
        if not run.warmup:
            groups[(run.scenario, run.format, run.backend, run.runner)].append(run)

    summaries = []
    for key, group in sorted(groups.items()):
        successful = [run for run in group if run.success]
        durations = [run.duration_seconds for run in successful]
        summaries.append(
            BenchmarkSummary(
                scenario=key[0],
                format=key[1],
                backend=key[2],
                runner=key[3],
                successful_runs=len(successful),
                failed_runs=len(group) - len(successful),
                median_duration_seconds=(
                    statistics.median(durations) if durations else 0.0
                ),
                p95_duration_seconds=(
                    percentile(durations, 0.95) if durations else 0.0
                ),
                median_rows_per_second=_optional_median(
                    [run.rows_per_second for run in successful]
                ),
                p95_rows_per_second=_optional_p95(
                    [run.rows_per_second for run in successful]
                ),
                median_mib_per_second=_optional_median(
                    [run.mib_per_second for run in successful]
                ),
                relative_throughput_change_pct=None,
                relative_latency_change_pct=None,
            )
        )

    baselines = {
        (summary.scenario, summary.backend, summary.runner): summary
        for summary in summaries
        if summary.format == "parquet" and summary.successful_runs
    }
    paired = []
    for summary in summaries:
        baseline = baselines.get(
            (summary.scenario, summary.backend, summary.runner)
        )
        if (
            summary.format != "vortex"
            or not summary.successful_runs
            or baseline is None
        ):
            paired.append(summary)
            continue
        throughput = None
        if (
            summary.median_rows_per_second is not None
            and baseline.median_rows_per_second not in (None, 0)
        ):
            throughput = relative_change(
                summary.median_rows_per_second,
                baseline.median_rows_per_second,
            )
        latency = None
        if baseline.median_duration_seconds:
            latency = relative_change(
                summary.median_duration_seconds,
                baseline.median_duration_seconds,
            )
        paired.append(
            replace(
                summary,
                relative_throughput_change_pct=throughput,
                relative_latency_change_pct=latency,
            )
        )
    return paired


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text)
    temporary.replace(path)


def write_raw_json(
    runs: Sequence[BenchmarkRun],
    metadata: BenchmarkMetadata,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summaries = aggregate_runs(runs)
    payload = {
        "metadata": dict(metadata),
        "runs": [run.to_dict() for run in runs],
        "summaries": [summary.to_dict() for summary in summaries],
    }
    _atomic_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_csv(
    path: Path,
    rows: Sequence[BenchmarkRunDict | BenchmarkSummaryDict],
    columns: list[str],
) -> None:
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _empty_or_bar_chart(
    path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    ylabel: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    if values:
        positions = range(len(values))
        axis.bar(positions, values)
        axis.set_xticks(list(positions), labels, rotation=30, ha="right")
    else:
        axis.text(
            0.5,
            0.5,
            "No successful runs",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_xticks([])
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _storage_values(
    metadata: BenchmarkMetadata,
) -> tuple[list[str], list[float]]:
    tables = metadata.get("tables")
    if tables is None:
        return [], []
    labels = []
    values = []
    for format_name, table in (
        ("parquet", tables["parquet"]),
        ("vortex", tables["vortex"]),
    ):
        size = table["physical_bytes"]
        if isinstance(size, (int, float)):
            labels.append(format_name)
            values.append(float(size) / 1024 / 1024)
    return labels, values


def _summary_chart_data(
    summaries: Sequence[BenchmarkSummary], scenarios: set[str]
) -> tuple[list[str], list[float]]:
    labels = []
    values = []
    for summary in summaries:
        if (
            summary.scenario in scenarios
            and summary.successful_runs
            and summary.median_rows_per_second is not None
        ):
            labels.append(
                f"{summary.scenario}\n{summary.format}/{summary.backend}/{summary.runner}"
            )
            values.append(summary.median_rows_per_second)
    return labels, values


def write_reports(
    runs: Sequence[BenchmarkRun],
    metadata: BenchmarkMetadata,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = aggregate_runs(runs)
    results_json = output_dir / "results.json"
    results_csv = output_dir / "results.csv"
    summary_csv = output_dir / "summary.csv"
    write_raw_json(runs, metadata, results_json)
    _write_csv(
        results_csv,
        [run.to_dict() for run in runs],
        [field.name for field in fields(BenchmarkRun)],
    )
    _write_csv(
        summary_csv,
        [summary.to_dict() for summary in summaries],
        [field.name for field in fields(BenchmarkSummary)],
    )

    chart_specs = []
    storage_labels, storage_values = _storage_values(metadata)
    chart_specs.append(
        (
            output_dir / "storage-size.png",
            "LakeSoul Physical Storage Size",
            storage_labels,
            storage_values,
            "MiB",
        )
    )
    for filename, title, scenarios in [
        (
            "scan-throughput.png",
            "Scan Throughput",
            {"metadata_scan", "full_scan", "filtered_scan", "blob_scan"},
        ),
        ("decode-throughput.png", "Decode Throughput", {"decode_images"}),
        (
            "backend-throughput.png",
            "Backend Processing Throughput",
            {"expand_captions", "batch_samples"},
        ),
    ]:
        labels, values = _summary_chart_data(summaries, scenarios)
        chart_specs.append(
            (output_dir / filename, title, labels, values, "rows/s")
        )
    for path, title, labels, values, ylabel in chart_specs:
        _empty_or_bar_chart(path, title, labels, values, ylabel)

    return [
        results_json,
        results_csv,
        summary_csv,
        *(spec[0] for spec in chart_specs),
    ]
