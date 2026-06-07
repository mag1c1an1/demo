# LakeSoul Multimodal Benchmark and Data Pipeline Design

## Purpose

Extend the Flickr30k demo into a reproducible LakeSoul multimodal showcase that:

1. Imports identical data into Parquet-backed and Vortex-backed LakeSoul tables.
2. Compares storage, scan, filtering, blob transfer, image decoding, and training-batch preparation end to end.
3. Demonstrates local and distributed processing with Ray.
4. Demonstrates multimodal transformation with Daft using both its native and Ray runners.
5. Lets CLIP retrieval training select `native`, `ray`, or `daft` data backends without changing model code.
6. Produces reusable JSON, CSV, and PNG benchmark reports.

The default run uses 1,000 Flickr30k images for a practical demo. Passing `--limit 0` uses the complete dataset.

## Scope

### Included

- Flickr30k image blobs, dimensions, captions, and bounding boxes.
- Two LakeSoul tables with identical logical contents and different physical file formats.
- Local Spark import.
- LakeSoul Arrow, Ray Data, and Daft data preparation paths.
- Local Ray and existing remote Ray cluster execution.
- End-to-end benchmark metrics and static report charts.
- Refactoring retrieval training around a common sample-stream contract.
- Automated unit tests and small local integration tests.

### Excluded

- A native Daft LakeSoul connector.
- Distributed Spark import.
- Ray Train, distributed model training, hyperparameter tuning, or model serving.
- URI-based image storage, vector indexes, or embedding persistence.
- Automated provisioning of a Ray cluster, PostgreSQL metadata service, or object storage.
- Full Flickr30k and remote-cluster runs in the default automated test suite.

## Logical Schema

Both LakeSoul tables use the same schema:

```text
filename: string, non-null
image_blob: binary, non-null
width: int32, non-null
height: int32, non-null
captions: list<string>, non-null
bboxes: list<struct<
  chain_ids: list<string>,
  xmin: int32,
  ymin: int32,
  xmax: int32,
  ymax: int32
>>, nullable
```

The default table names are:

```text
flickr30k_parquet
flickr30k_vortex
```

The physical format is not stored as a logical column because it is a table-level property and is already present in benchmark metadata.

## Components

### `import_data.py`

The import command owns source discovery, deterministic sampling, XML parsing, Spark setup, dual-table writes, and post-write validation.

Its default behavior is:

```bash
python import_data.py \
  --limit 1000 \
  --parquet-table flickr30k_parquet \
  --vortex-table flickr30k_vortex
```

Important options:

```text
--limit N                  Number of images; 0 means all images
--batch-size N             Number of source records per Spark write
--seed N                   Deterministic sampling seed
--parquet-table NAME       Parquet-backed LakeSoul table
--vortex-table NAME        Vortex-backed LakeSoul table
--overwrite                Explicitly replace existing target tables
--output PATH              Import metrics JSON path
```

The command refuses to modify either target table when a target exists and `--overwrite` is absent. With `--overwrite`, it drops both targets before writing so a run cannot silently compare a new table with an old table.

### `multimodal_data.py`

This module contains framework-independent data contracts and deterministic data rules:

```python
@dataclass(frozen=True)
class Sample:
    filename: str
    image_bytes: bytes
    caption: str


@dataclass(frozen=True)
class ImageRecord:
    filename: str
    image_bytes: bytes
    captions: tuple[str, ...]
```

It owns:

- Stable source ordering and seeded limiting.
- Image-level train, validation, and test splitting.
- Caption expansion after splitting.
- Canonical sample hashing.
- Corrupt-image validation helpers.

Splitting happens by filename before captions are expanded. This prevents captions for one image from appearing in multiple splits.

### `data_backends.py`

This module exposes one backend-neutral interface:

```python
def iter_samples(
    backend: str,
    table_name: str,
    split: str,
    *,
    batch_size: int,
    seed: int,
    limit: int,
    ray_address: str | None,
    daft_runner: str,
    skip_corrupt: bool,
) -> Iterator[Sample]:
    ...
```

Backends:

- `native`: LakeSoul Arrow batches are projected in bulk and converted directly to samples.
- `ray`: `lakesoul.ray.read_lakesoul` creates a Ray Dataset; Ray tasks expand captions and prepare samples.
- `daft`: LakeSoul Arrow record batches enter Daft through `daft.from_arrow`; Daft performs caption expansion and image decoding or validation. `--daft-runner native` uses local threads and `--daft-runner ray` uses Ray.

All backends implement the same deterministic split and limit semantics. Backend conformance tests compare canonical sample hashes, not incidental partition arrival order.

### `benchmark.py`

This is the benchmark orchestrator. A typical local run is:

```bash
python benchmark.py \
  --limit 1000 \
  --warmups 1 \
  --runs 3 \
  --ray-address local \
  --output-dir benchmark-results
```

A remote run uses:

```bash
python benchmark.py \
  --ray-address ray://ray-head.example:10001 \
  --daft-runner ray
```

The orchestrator:

1. Validates both tables and their logical equality.
2. Captures environment and table metadata.
3. Runs one warm-up by default.
4. Runs each measured case three times by default.
5. Writes raw run records before aggregation.
6. Calls the reporting module after all requested scenarios finish.
7. Returns a non-zero exit code if required scenarios fail or table equality validation fails.

Each scenario creates a fresh LakeSoul scan or framework plan. Materialized data from one measured run is not reused by another measured run.

### `benchmark_report.py`

This module reads normalized benchmark records and creates:

```text
benchmark-results/results.json
benchmark-results/results.csv
benchmark-results/summary.csv
benchmark-results/storage-size.png
benchmark-results/scan-throughput.png
benchmark-results/decode-throughput.png
benchmark-results/backend-throughput.png
```

Charts use successful, validated measured runs only. Warm-ups and failed runs remain in `results.json` for diagnosis but are excluded from summaries.

### `train_retrieval.py`

The training command retains CLIP model, loss, optimizer, evaluation, and checkpoint behavior while replacing direct DuckDB point queries with the common backend contract.

Key options:

```text
--backend native|ray|daft
--table flickr30k_parquet|flickr30k_vortex
--limit N
--seed N
--batch-size N
--epochs N
--ray-address local|auto|ray://...
--daft-runner native|ray
--skip-corrupt
```

The default backend is `native`. The model and collator receive the same `Sample` values regardless of backend.

## Import Data Flow

1. Validate the dataset JSON, image directory, annotation directory, Spark JAR, and target-table state.
2. Sort source filenames and apply deterministic seeded sampling when `--limit` is non-zero.
3. Parse each selected image and annotation once.
4. Compute a source SHA-256 digest from canonical records containing filename, image SHA-256, dimensions, captions, and bounding boxes.
5. Build one Spark DataFrame for each source batch.
6. Time the Parquet and Vortex write actions separately while excluding source parsing and digest computation.
7. Alternate which format is written first on successive batches to reduce systematic first-write cache bias.
8. Append subsequent batches to their corresponding table.
9. Read each completed table through LakeSoul Arrow and compute its canonical digest.
10. Require equal schema, row count, filename set, and canonical digest across source, Parquet, and Vortex.
11. Record write time, file count, physical bytes, logical blob bytes, rows, and digest.

If one table write fails, the command stops and marks the import incomplete. It does not report comparative results until both tables pass validation. A later retry requires `--overwrite`.

## Benchmark Matrix

Every storage scenario runs once for each table format.

### Storage and Import

- Total measured Spark write time.
- Number of data files.
- Total physical data-file bytes.
- Logical encoded-image bytes.
- Physical-to-logical size ratio.

### LakeSoul Scan

`metadata_scan`

- Project `filename`, `width`, `height`, and `captions`.
- Consume all selected rows.
- Report rows/s and serialized MiB/s where byte accounting is available.

`full_scan`

- Project all columns, including `image_blob` and bounding boxes.
- Consume all selected rows and all binary values.
- Report rows/s and blob MiB/s.

`filtered_scan`

- Select a deterministic filename subset generated from the shared seed.
- Project `filename`, `image_blob`, and `captions`.
- Report matched rows, latency, rows/s, and blob MiB/s.

`blob_scan`

- Project `filename` and `image_blob`.
- Consume image bytes without decoding.
- Report rows/s and blob MiB/s.

### Multimodal Processing

`decode_images`

- Decode each selected image into RGB.
- Validate decoded width and height.
- Report images/s, encoded MiB/s, and `decode_errors`.

`expand_captions`

- Expand each image row into one row per caption.
- Report source images/s and output samples/s.

`prepare_training_batches`

- Expand captions, decode images, and form fixed-size batches matching the CLIP collator input boundary.
- Exclude model forward and backward time.
- Report images/s, samples/s, batches/s, and encoded MiB/s.

The processing matrix includes:

```text
native
ray
daft-native
daft-ray
```

Ray reads LakeSoul through the LakeSoul Ray datasource. Daft receives LakeSoul Arrow data and owns downstream transformation. `daft-ray` demonstrates distributed Daft transformation after the Arrow bridge; it is not described as a native distributed Daft LakeSoul scan.

## Timing and Statistics

Timing uses `time.perf_counter_ns()` around only the operation under test. Framework initialization, Ray connection, table validation, model loading, report generation, and source parsing are measured separately or excluded.

Defaults:

```text
warm-up runs: 1
measured runs: 3
```

Each raw run records:

```text
run_id
timestamp
scenario
format
backend
runner
warmup
success
duration_seconds
input_rows
output_rows
input_bytes
rows_per_second
output_rows_per_second
mib_per_second
decode_errors
error_type
error_message
```

Each summary group reports:

- Median duration and throughput.
- P95 duration and throughput using an inclusive percentile calculation.
- Minimum and maximum.
- Successful and failed run counts.
- Vortex relative change versus Parquet:

```text
throughput_change_pct = (vortex_throughput / parquet_throughput - 1) * 100
latency_change_pct = (vortex_latency / parquet_latency - 1) * 100
size_change_pct = (vortex_size / parquet_size - 1) * 100
```

With only three measured runs, P95 is useful as a directional tail indicator, not a statistically strong latency claim. The report states the run count beside every summary.

## Ray Execution

`--ray-address` semantics:

```text
local       Start or reuse a local Ray runtime
auto        Connect to an existing runtime discovered by Ray
ray://...   Connect through Ray Client
host:port   Connect to an existing Ray cluster
```

The driver initializes Ray once before measured scenarios. Initialization time is excluded. The remote runtime must provide:

- A matching Python minor version.
- The same LakeSoul wheel.
- Ray and, for Daft-Ray scenarios, Daft.
- Access to LakeSoul metadata and all table data files.
- Equivalent object-store and metadata environment variables.

The benchmark emits the detected Ray cluster resources and package versions into result metadata.

## Daft Execution

Daft is an optional demo dependency but a requested benchmark capability. If it is absent:

- Import, native, and Ray-only commands remain usable.
- Explicitly requested Daft scenarios fail fast with an installation message.
- A run requesting `--backends all` records Daft scenarios as unavailable and exits non-zero unless `--allow-missing-optional` is set.

Daft native and Ray runner selection occurs before constructing any Daft DataFrame because Daft locks its runner after initialization. Benchmarking both runners in one command therefore uses isolated child processes, one runner per process.

## Error Handling

### Prerequisite Failures

Commands validate prerequisites before destructive or expensive work. Messages name the missing file, table, package, service, or environment setting and include the relevant option that can override the default.

### Corrupt Images

Benchmark mode counts image decode failures and continues by default so data-quality behavior is visible. Failed rows do not count as successfully decoded images.

Training mode fails on the first corrupt image by default. `--skip-corrupt` logs and omits corrupt samples while recording the count.

### Partial Benchmark Failures

Every scenario writes a failure record with exception type and a bounded error message. No throughput is calculated for failed runs. Summary and charts exclude them.

### Result Durability

Raw result JSON is updated atomically after each scenario by writing a temporary file and replacing the destination. A process interruption therefore preserves all completed scenario records.

## Reproducibility Metadata

`results.json` includes:

- Command-line arguments.
- Git commit and dirty-worktree flag.
- UTC timestamp.
- Hostname, OS, CPU count, memory, and GPU availability.
- Python, Java, Spark, LakeSoul, PyArrow, Ray, Daft, DuckDB, Torch, and Transformers versions when installed.
- Table names, physical formats, row counts, file counts, physical bytes, and canonical digests.
- Seed, limit, batch sizes, warm-up count, measured-run count, and Ray address mode.

Credentials and secret environment-variable values are never written.

## Testing Strategy

### Unit Tests

Tests cover:

- Caption cleanup and XML parsing.
- Stable sampling for a fixed seed.
- Image-level split isolation.
- Caption expansion.
- Canonical digest stability and sensitivity.
- CLI defaults and validation.
- Median, inclusive P95, throughput, and relative-change calculations.
- Failure exclusion from summaries.
- CSV/JSON schema and chart file creation.
- Backend sample normalization and corrupt-image policy.

### Integration Tests

A small fixture generates several in-memory JPEG images, captions, and annotations. When Spark and the LakeSoul JAR are available, integration tests:

1. Write the fixture into temporary Parquet-backed and Vortex-backed LakeSoul tables.
2. Validate equal schema, rows, filenames, and canonical digests.
3. Run each available backend against both tables.
4. Compare canonical output sample hashes.
5. Run one warm-up and one measured benchmark iteration.
6. Verify JSON, CSV, and PNG outputs.

Ray integration uses a local runtime with bounded CPU resources. Daft tests are skipped only when the optional dependency is not installed and are required in the documented full demo environment.

### Manual Smoke Tests

Manual validation covers:

- Complete Flickr30k import with `--limit 0`.
- Complete local benchmark.
- Remote Ray Data benchmark.
- Daft Ray runner against the remote cluster.
- One short training epoch for every backend and both storage formats.

## Documentation

`README.md` will explain:

- Environment prerequisites.
- Dataset and LakeSoul metadata setup.
- Dual-format import commands.
- Quick and full benchmark commands.
- Local and remote Ray examples.
- Daft native and Ray examples.
- Training backend examples.
- Metric definitions and correct interpretation.
- The Arrow-bridge limitation for Daft.

## Acceptance Criteria

The work is complete when:

1. One import command creates both target tables from the same deterministic records.
2. Import validation proves schema, row, filename, and canonical-content equality.
3. Benchmark output includes storage, scan, filter, blob, decode, caption expansion, and batch-preparation results.
4. Results include raw JSON, raw CSV, summary CSV, and four readable PNG charts.
5. Ray works locally and accepts a remote address.
6. Daft works with native and Ray runners through the documented LakeSoul Arrow bridge.
7. Retrieval training accepts `native`, `ray`, and `daft` while retaining one model/training implementation.
8. Image-level splits prevent caption leakage.
9. Automated unit and small integration tests pass in the full demo environment.
10. README commands are sufficient to reproduce the 1,000-image demo.
