# Repository Guidelines

## Project Structure & Module Organization

The repository is a Python 3.10 LakeSoul multimodal benchmark:

- `multimodal_data.py`: immutable records, deterministic selection/splits, image decoding, and canonical digests.
- `import_data.py`: paired Parquet/Vortex LakeSoul import CLI.
- `data_backends.py` and `daft_worker.py`: native, Ray, and isolated Daft data paths.
- `benchmark.py`, `benchmark_models.py`, and `benchmark_report.py`: benchmark execution, statistics, and report generation.
- `train_retrieval.py`: CLIP retrieval training through the shared backend API.
- `tests/`: unit tests; `tests/integration/` contains Spark/LakeSoul end-to-end coverage.
- `data/`, `spark-warehouse/`, model outputs, and benchmark outputs are local artifacts and must remain untracked.

## Build, Test, and Development Commands

Install or refresh dependencies:

```bash
uv sync
```

Run fast tests and syntax checks:

```bash
.venv/bin/python -m compileall import_data.py multimodal_data.py data_backends.py \
  daft_worker.py benchmark_models.py benchmark_report.py benchmark.py \
  train_retrieval.py tests
.venv/bin/python -m pytest -m "not integration" -q
```

Run the external-service integration test:

```bash
RUN_LAKESOUL_INTEGRATION=1 .venv/bin/python -m pytest \
  tests/integration/test_dual_format_pipeline.py -v --timeout=180
```

Use `import_data.py --overwrite`, `benchmark.py`, and
`train_retrieval.py --dry-run-batches 1` for local workflow smoke tests.

## Coding Style & Naming Conventions

Use four-space indentation, type hints, `snake_case` functions/variables, and
`PascalCase` dataclasses. Prefer immutable `@dataclass(frozen=True)` contracts.
Keep backend-specific behavior behind the shared interfaces and avoid loading
all image blobs into memory. Use `pathlib.Path`, explicit exceptions, and
atomic replacement for generated JSON/CSV files. No formatter is configured;
keep code PEP 8 compliant and run `compileall` plus `git diff --check`.

## Testing Guidelines

Tests use pytest. Name files `test_<module>.py` and tests `test_<behavior>`.
Add focused unit tests for deterministic rules, CLI defaults, failure records,
and report outputs. Mark tests requiring Spark, LakeSoul metadata, or Ray with
`@pytest.mark.integration`; default test runs must not require those services.

## Commit & Pull Request Guidelines

History favors short Conventional Commit subjects such as `feat:`, `test:`,
`docs:`, `build:`, and `chore:`. Keep commits scoped to one behavioral change.
Pull requests should describe the change, list exact verification commands,
note required external services, and identify generated artifacts or benchmark
results. Include screenshots only when report chart appearance changes.

## Configuration & Data Safety

Never commit Flickr30k data, Spark warehouses, credentials, trained models, or
benchmark outputs. Existing LakeSoul tables must not be dropped without an
explicit `--overwrite` request.
