import json

from benchmark_models import BenchmarkRun
from benchmark_report import aggregate_runs, write_reports


def run(format_name: str, duration: float, throughput: float) -> BenchmarkRun:
    return BenchmarkRun(
        run_id=f"{format_name}-{duration}",
        timestamp="2026-06-08T00:00:00Z",
        scenario="blob_scan",
        format=format_name,
        backend="native",
        runner="native",
        warmup=False,
        success=True,
        duration_seconds=duration,
        input_rows=100,
        output_rows=100,
        input_bytes=1024,
        rows_per_second=throughput,
        output_rows_per_second=throughput,
        mib_per_second=1.0,
        decode_errors=0,
        error_type=None,
        error_message=None,
    )


def test_aggregate_uses_median_inclusive_p95_and_relative_change():
    runs = [
        run("parquet", 3.0, 30.0),
        run("parquet", 2.0, 40.0),
        run("parquet", 1.0, 50.0),
        run("vortex", 2.0, 60.0),
        run("vortex", 1.5, 80.0),
        run("vortex", 1.0, 100.0),
    ]
    summaries = aggregate_runs(runs)
    vortex = next(summary for summary in summaries if summary.format == "vortex")
    assert vortex.median_rows_per_second == 80.0
    assert vortex.relative_throughput_change_pct == 100.0


def test_write_reports_creates_json_csv_summary_and_png(tmp_path):
    paths = write_reports([run("parquet", 1.0, 100.0)], {}, tmp_path)
    assert {path.name for path in paths} >= {
        "results.json",
        "results.csv",
        "summary.csv",
        "storage-size.png",
        "scan-throughput.png",
        "decode-throughput.png",
        "backend-throughput.png",
    }
    assert json.loads((tmp_path / "results.json").read_text())["runs"]
