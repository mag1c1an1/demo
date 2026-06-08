from benchmark import build_parser, execute_case


def test_benchmark_cli_defaults():
    args = build_parser().parse_args([])
    assert args.limit == 1000
    assert args.warmups == 1
    assert args.runs == 3
    assert args.ray_address == "local"


def test_execute_case_calculates_rates():
    result = execute_case(
        scenario="blob_scan",
        format_name="vortex",
        backend="native",
        runner="native",
        warmup=False,
        operation=lambda: {
            "input_rows": 10,
            "output_rows": 10,
            "input_bytes": 1024 * 1024,
            "decode_errors": 0,
        },
        clock_ns=iter([0, 1_000_000_000]).__next__,
    )
    assert result.success
    assert result.rows_per_second == 10.0
    assert result.mib_per_second == 1.0


def test_execute_case_records_failure_without_throughput():
    def fail():
        raise RuntimeError("boom")

    result = execute_case(
        "decode_images", "parquet", "native", "native", False, fail
    )
    assert not result.success
    assert result.rows_per_second is None
    assert result.error_type == "RuntimeError"
