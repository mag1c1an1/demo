from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import pyarrow.dataset as ds

from benchmark_models import (
    BenchmarkConfigMetadata,
    BenchmarkMetadata,
    BenchmarkRun,
    BenchmarkTablesMetadata,
    OperationMetrics,
    TableReportMetadata,
)
from benchmark_report import aggregate_runs, write_raw_json, write_reports
from data_backends import (
    BackendConfig,
    choose_records,
    init_ray,
    iter_records,
    iter_samples,
    run_daft_worker,
)
from import_data import validate_table
from multimodal_data import decode_rgb, select_filenames

SCAN_COLUMNS = {
    "metadata_scan": ["filename", "width", "height", "captions"],
    "full_scan": None,
    "filtered_scan": ["filename", "image_blob", "captions"],
    "blob_scan": ["filename", "image_blob"],
}


@dataclass(frozen=True)
class BenchmarkConfig:
    parquet_table: str = "flickr30k_parquet"
    vortex_table: str = "flickr30k_vortex"
    limit: int = 1000
    seed: int = 7
    batch_size: int = 32
    warmups: int = 1
    runs: int = 3
    ray_address: str = "local"
    backends: tuple[str, ...] = ("native", "ray", "daft-native", "daft-ray")
    output_dir: Path = Path("benchmark-results")
    allow_missing_optional: bool = False

    def __post_init__(self) -> None:
        if self.limit < 0:
            raise ValueError("limit must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch size must be positive")
        if self.warmups < 0 or self.runs <= 0:
            raise ValueError("warmups must be non-negative and runs must be positive")


@dataclass(frozen=True)
class BenchmarkCase:
    scenario: str
    format_name: str
    backend: str
    runner: str
    operation: Callable[[], OperationMetrics]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark matching Parquet and Vortex LakeSoul tables"
    )
    parser.add_argument("--parquet-table", default="flickr30k_parquet")
    parser.add_argument("--vortex-table", default="flickr30k_vortex")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--ray-address", default="local")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["native", "ray", "daft-native", "daft-ray"],
        default=["native", "ray", "daft-native", "daft-ray"],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results"))
    parser.add_argument("--allow-missing-optional", action="store_true")
    return parser


def execute_case(
    scenario: str,
    format_name: str,
    backend: str,
    runner: str,
    warmup: bool,
    operation: Callable[[], OperationMetrics],
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> BenchmarkRun:
    started = clock_ns()
    try:
        metrics = operation()
        duration = (clock_ns() - started) / 1_000_000_000
        rows = metrics["input_rows"]
        output_rows = metrics["output_rows"]
        input_bytes = metrics["input_bytes"]
        return BenchmarkRun(
            run_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            scenario=scenario,
            format=format_name,
            backend=backend,
            runner=runner,
            warmup=warmup,
            success=True,
            duration_seconds=duration,
            input_rows=rows,
            output_rows=output_rows,
            input_bytes=input_bytes,
            rows_per_second=rows / duration if duration else None,
            output_rows_per_second=output_rows / duration if duration else None,
            mib_per_second=input_bytes / duration / 1024 / 1024 if duration else None,
            decode_errors=metrics.get("decode_errors", 0),
            error_type=None,
            error_message=None,
        )
    except Exception as exc:
        duration = (clock_ns() - started) / 1_000_000_000
        return BenchmarkRun(
            str(uuid.uuid4()),
            datetime.now(timezone.utc).isoformat(),
            scenario,
            format_name,
            backend,
            runner,
            warmup,
            False,
            duration,
            0,
            0,
            0,
            None,
            None,
            None,
            0,
            type(exc).__name__,
            str(exc)[:1000],
        )


def _blob_bytes(batch) -> int:
    index = batch.schema.get_field_index("image_blob")
    if index < 0:
        return 0
    return sum(
        len(value.as_py()) for value in batch.column(index) if value.is_valid
    )


def scan_operation(
    table_name: str,
    scenario: str,
    filenames: Sequence[str],
    *,
    batch_size: int = 1024,
) -> OperationMetrics:
    from lakesoul.arrow.dataset import lakesoul_dataset

    if scenario not in SCAN_COLUMNS:
        raise ValueError(f"unknown scan scenario: {scenario}")
    dataset = lakesoul_dataset(table_name, batch_size=batch_size)
    filter_expression = None
    if scenario == "filtered_scan":
        filter_expression = ds.field("filename").isin(list(filenames))
    rows = 0
    input_bytes = 0
    for batch in dataset.to_batches(
        columns=SCAN_COLUMNS[scenario],
        filter=filter_expression,
        batch_size=batch_size,
    ):
        rows += batch.num_rows
        input_bytes += _blob_bytes(batch)
    return {
        "input_rows": rows,
        "output_rows": rows,
        "input_bytes": input_bytes,
        "decode_errors": 0,
    }


def decode_operation(config: BackendConfig) -> OperationMetrics:
    if config.backend == "daft":
        return _daft_metrics(config)
    records = choose_records(iter_records(config), config)
    decoded = 0
    errors = 0
    encoded_bytes = 0
    for record in records:
        encoded_bytes += len(record.image_bytes)
        try:
            image = decode_rgb(record.image_bytes)
            if image.size != (record.width, record.height):
                raise ValueError(f"decoded size mismatch for {record.filename}")
            decoded += 1
        except ValueError:
            errors += 1
            if not config.skip_corrupt:
                raise
    return {
        "input_rows": len(records),
        "output_rows": decoded,
        "input_bytes": encoded_bytes,
        "decode_errors": errors,
    }


def expand_operation(config: BackendConfig) -> OperationMetrics:
    if config.backend == "daft":
        return _daft_metrics(config)
    records = choose_records(iter_records(config), config)
    return {
        "input_rows": len(records),
        "output_rows": sum(len(record.captions) for record in records),
        "input_bytes": sum(len(record.image_bytes) for record in records),
        "decode_errors": 0,
    }


def batch_operation(config: BackendConfig) -> OperationMetrics:
    if config.backend == "daft":
        return _daft_metrics(config)
    samples = iter(iter_samples(config))
    output_rows = 0
    input_bytes = 0
    errors = 0
    batch = []
    for sample in samples:
        try:
            image = decode_rgb(sample.image_bytes)
        except ValueError:
            errors += 1
            if not config.skip_corrupt:
                raise
            continue
        batch.append((image, sample.caption))
        input_bytes += len(sample.image_bytes)
        if len(batch) == config.batch_size:
            output_rows += len(batch)
            batch = []
    output_rows += len(batch)
    return {
        "input_rows": output_rows,
        "output_rows": output_rows,
        "input_bytes": input_bytes,
        "decode_errors": errors,
    }


def _daft_metrics(config: BackendConfig) -> OperationMetrics:
    with tempfile.TemporaryDirectory(prefix="lakesoul-daft-benchmark-") as directory:
        root = Path(directory)
        payload = run_daft_worker(
            config, root / "samples.arrow", root / "status.json"
        )
    return {
        "input_rows": payload["input_rows"],
        "output_rows": payload["output_rows"],
        "input_bytes": payload["input_bytes"],
        "decode_errors": payload["decode_errors"],
    }


def _backend_config(
    name: str, table_name: str, config: BenchmarkConfig
) -> BackendConfig:
    if name.startswith("daft-"):
        backend = "daft"
        runner = name.removeprefix("daft-")
    else:
        backend = name
        runner = "native"
    return BackendConfig(
        backend=backend,
        table_name=table_name,
        split="all",
        batch_size=config.batch_size,
        seed=config.seed,
        limit=config.limit,
        ray_address=config.ray_address,
        daft_runner=runner,
        skip_corrupt=False,
    )


def build_cases(
    config: BenchmarkConfig, selected_filenames: Sequence[str]
) -> list[BenchmarkCase]:
    cases = []
    tables = {
        "parquet": config.parquet_table,
        "vortex": config.vortex_table,
    }
    for format_name, table_name in tables.items():
        for scenario in SCAN_COLUMNS:
            cases.append(
                BenchmarkCase(
                    scenario,
                    format_name,
                    "native",
                    "native",
                    lambda table_name=table_name, scenario=scenario: scan_operation(
                        table_name,
                        scenario,
                        selected_filenames,
                        batch_size=config.batch_size,
                    ),
                )
            )
        for backend_name in config.backends:
            backend_config = _backend_config(backend_name, table_name, config)
            runner = backend_config.daft_runner
            for scenario, operation in [
                ("decode_images", decode_operation),
                ("expand_captions", expand_operation),
                ("batch_samples", batch_operation),
            ]:
                cases.append(
                    BenchmarkCase(
                        scenario,
                        format_name,
                        backend_config.backend,
                        runner,
                        lambda operation=operation, backend_config=backend_config: operation(
                            backend_config
                        ),
                    )
                )
    return cases


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for name in [
        "daft",
        "lakesoul",
        "matplotlib",
        "pyarrow",
        "ray",
        "torch",
        "transformers",
    ]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or result.stderr.strip() or None


def _config_metadata(config: BenchmarkConfig) -> BenchmarkConfigMetadata:
    return {
        "parquet_table": config.parquet_table,
        "vortex_table": config.vortex_table,
        "limit": config.limit,
        "seed": config.seed,
        "batch_size": config.batch_size,
        "warmups": config.warmups,
        "runs": config.runs,
        "ray_address": config.ray_address,
        "backends": list(config.backends),
        "output_dir": str(config.output_dir),
        "allow_missing_optional": config.allow_missing_optional,
    }


def capture_environment(config: BenchmarkConfig) -> BenchmarkMetadata:
    try:
        memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError):
        memory_bytes = None
    return {
        "benchmark_config": _config_metadata(config),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "memory_bytes": memory_bytes,
        "gpu": _command_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
        ),
        "java": _command_output(["java", "-version"]),
        "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_command_output(["git", "status", "--porcelain"])),
        "packages": _package_versions(),
    }


def _table_report_metadata(table_name: str) -> TableReportMetadata:
    validation = validate_table(table_name)
    return {
        "row_count": validation["row_count"],
        "filenames": validation["filenames"],
        "digest": validation["digest"],
        "logical_blob_bytes": validation["logical_blob_bytes"],
        "physical_bytes": validation["physical_bytes"],
    }


def _validate_tables(
    config: BenchmarkConfig,
) -> tuple[BenchmarkTablesMetadata, tuple[str, ...]]:
    from lakesoul.arrow.dataset import lakesoul_dataset

    parquet = _table_report_metadata(config.parquet_table)
    vortex = _table_report_metadata(config.vortex_table)
    parquet_schema = str(lakesoul_dataset(config.parquet_table).schema)
    vortex_schema = str(lakesoul_dataset(config.vortex_table).schema)
    if parquet_schema != vortex_schema:
        raise RuntimeError("Parquet and Vortex schemas differ")
    for key in ("row_count", "filenames", "digest"):
        if parquet[key] != vortex[key]:
            raise RuntimeError(f"Parquet and Vortex {key} values differ")
    selected = select_filenames(
        parquet["filenames"], limit=config.limit, seed=config.seed
    )
    return {
        "parquet": parquet,
        "vortex": vortex,
        "schema": parquet_schema,
    }, selected


def _config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    return BenchmarkConfig(
        parquet_table=args.parquet_table,
        vortex_table=args.vortex_table,
        limit=args.limit,
        seed=args.seed,
        batch_size=args.batch_size,
        warmups=args.warmups,
        runs=args.runs,
        ray_address=args.ray_address,
        backends=tuple(args.backends),
        output_dir=args.output_dir,
        allow_missing_optional=args.allow_missing_optional,
    )


def _print_summary(runs: Sequence[BenchmarkRun]) -> None:
    print("scenario                 format   backend runner  ok/fail   rows/s")
    for summary in aggregate_runs(runs):
        throughput = (
            f"{summary.median_rows_per_second:.1f}"
            if summary.median_rows_per_second is not None
            else "-"
        )
        print(
            f"{summary.scenario:24} {summary.format:8} "
            f"{summary.backend:7} {summary.runner:7} "
            f"{summary.successful_runs}/{summary.failed_runs:<6} {throughput}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    config = _config_from_args(build_parser().parse_args(argv))
    metadata = capture_environment(config)
    tables, selected = _validate_tables(config)
    metadata["tables"] = tables

    needs_ray = any(
        backend in {"ray", "daft-ray"} for backend in config.backends
    )
    if needs_ray:
        try:
            init_ray(config.ray_address)
        except Exception:
            if not config.allow_missing_optional:
                raise
            config = replace(
                config,
                backends=tuple(
                    backend
                    for backend in config.backends
                    if backend not in {"ray", "daft-ray"}
                ),
            )

    runs: list[BenchmarkRun] = []
    for case in build_cases(config, selected):
        for index in range(config.warmups + config.runs):
            result = execute_case(
                scenario=case.scenario,
                format_name=case.format_name,
                backend=case.backend,
                runner=case.runner,
                warmup=index < config.warmups,
                operation=case.operation,
            )
            runs.append(result)
            write_raw_json(
                runs, metadata, config.output_dir / "results.json"
            )
            state = "ok" if result.success else f"failed: {result.error_type}"
            print(
                f"{case.scenario} {case.format_name} "
                f"{case.backend}/{case.runner}: {state}"
            )
    write_reports(runs, metadata, config.output_dir)
    _print_summary(runs)
    failures = [run for run in runs if not run.warmup and not run.success]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
