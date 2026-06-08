from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TypedDict, cast


CsvValue = str | int | float | bool | None


class BenchmarkRunDict(TypedDict):
    run_id: str
    timestamp: str
    scenario: str
    format: str
    backend: str
    runner: str
    warmup: bool
    success: bool
    duration_seconds: float
    input_rows: int
    output_rows: int
    input_bytes: int
    rows_per_second: float | None
    output_rows_per_second: float | None
    mib_per_second: float | None
    decode_errors: int
    error_type: str | None
    error_message: str | None


class BenchmarkSummaryDict(TypedDict):
    scenario: str
    format: str
    backend: str
    runner: str
    successful_runs: int
    failed_runs: int
    median_duration_seconds: float
    p95_duration_seconds: float
    median_rows_per_second: float | None
    p95_rows_per_second: float | None
    median_mib_per_second: float | None
    relative_throughput_change_pct: float | None
    relative_latency_change_pct: float | None


class BenchmarkConfigMetadata(TypedDict):
    parquet_table: str
    vortex_table: str
    limit: int
    seed: int
    batch_size: int
    warmups: int
    runs: int
    ray_address: str
    backends: list[str]
    output_dir: str
    allow_missing_optional: bool


class TableReportMetadata(TypedDict):
    row_count: int
    filenames: list[str]
    digest: str
    logical_blob_bytes: int
    physical_bytes: int | None


class BenchmarkTablesMetadata(TypedDict):
    parquet: TableReportMetadata
    vortex: TableReportMetadata
    schema: str


class BenchmarkMetadata(TypedDict, total=False):
    benchmark_config: BenchmarkConfigMetadata
    timestamp: str
    host: str
    platform: str
    python: str
    cpu_count: int | None
    memory_bytes: int | None
    gpu: str | None
    java: str | None
    git_commit: str | None
    git_dirty: bool
    packages: dict[str, str | None]
    tables: BenchmarkTablesMetadata
    integration: bool


class OperationMetrics(TypedDict):
    input_rows: int
    output_rows: int
    input_bytes: int
    decode_errors: int


@dataclass(frozen=True)
class BenchmarkRun:
    run_id: str
    timestamp: str
    scenario: str
    format: str
    backend: str
    runner: str
    warmup: bool
    success: bool
    duration_seconds: float
    input_rows: int
    output_rows: int
    input_bytes: int
    rows_per_second: float | None
    output_rows_per_second: float | None
    mib_per_second: float | None
    decode_errors: int
    error_type: str | None
    error_message: str | None

    def to_dict(self) -> BenchmarkRunDict:
        return cast(BenchmarkRunDict, asdict(self))


@dataclass(frozen=True)
class BenchmarkSummary:
    scenario: str
    format: str
    backend: str
    runner: str
    successful_runs: int
    failed_runs: int
    median_duration_seconds: float
    p95_duration_seconds: float
    median_rows_per_second: float | None
    p95_rows_per_second: float | None
    median_mib_per_second: float | None
    relative_throughput_change_pct: float | None
    relative_latency_change_pct: float | None

    def to_dict(self) -> BenchmarkSummaryDict:
        return cast(BenchmarkSummaryDict, asdict(self))
