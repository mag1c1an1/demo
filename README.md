# LakeSoul Multimodal Demo

## What It Demonstrates

This project imports the same Flickr30k image records into Parquet-backed and
Vortex-backed LakeSoul tables, validates their logical equality, benchmarks
storage and multimodal processing, and feeds the same backend contract into
CLIP retrieval training.

The benchmark emits raw JSON, raw CSV, aggregate CSV, and four PNG charts.
Filename limiting and train/validation/test splits are deterministic and happen
at image level before caption expansion.

## Prerequisites

- Python 3.10
- Java compatible with Spark 3.3
- `lakesoul-spark-3.3-3.0.0-SNAPSHOT.jar` in the project root
- Flickr30k data under `data/`:
  - `dataset_flickr30k_allEN.json`
  - `flickr30k-images/`
  - `Annotations/`
- Local LakeSoul metadata/storage configuration available to both Spark and
  the Python LakeSoul reader

## Install

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync
```

## Import Matching Parquet and Vortex Tables

The default import selects 1,000 images and writes both formats from each
materialized source batch. Existing targets are preserved unless
`--overwrite` is supplied.

```bash
.venv/bin/python import_data.py --overwrite
```

Import metrics are written to `benchmark-results/import.json`. The command
fails if source, Parquet, and Vortex row sets or canonical digests differ.

## Run the 1,000-Image Benchmark

```bash
.venv/bin/python benchmark.py --output-dir benchmark-results
```

The complete quick path is:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync
.venv/bin/python import_data.py --overwrite
.venv/bin/python benchmark.py --output-dir benchmark-results
.venv/bin/python train_retrieval.py \
  --backend native \
  --table flickr30k_vortex \
  --limit 1000 \
  --epochs 1
```

## Run the Full Flickr30k Benchmark

`--limit 0` means all available images.

```bash
.venv/bin/python import_data.py --limit 0 --overwrite
.venv/bin/python benchmark.py \
  --limit 0 \
  --warmups 1 \
  --runs 3 \
  --output-dir benchmark-results-full
```

## Use a Remote Ray Cluster

Use `auto` for an existing Ray environment or provide a Ray address:

```bash
.venv/bin/python benchmark.py \
  --ray-address ray://head.example:10001 \
  --backends ray daft-ray
```

The same option is available in training:

```bash
.venv/bin/python train_retrieval.py \
  --backend ray \
  --ray-address ray://head.example:10001
```

## Run Daft Native and Daft-Ray

Daft modes are isolated in worker processes because runner selection is
process-global.

```bash
.venv/bin/python benchmark.py --backends daft-native daft-ray

.venv/bin/python train_retrieval.py \
  --backend daft \
  --daft-runner native \
  --dry-run-batches 1

.venv/bin/python train_retrieval.py \
  --backend daft \
  --daft-runner ray \
  --ray-address local \
  --dry-run-batches 1
```

## Train CLIP Retrieval

```bash
.venv/bin/python train_retrieval.py \
  --backend native \
  --table flickr30k_vortex \
  --limit 1000 \
  --epochs 5 \
  --batch-size 64
```

Available data backends are `native`, `ray`, and `daft`. Use
`--dry-run-batches 1` to verify decoded image/text batches without loading the
CLIP model.

## Understand the Metrics

- `duration_seconds`: wall-clock duration for one isolated case.
- `rows_per_second`: source image rows divided by duration.
- `output_rows_per_second`: expanded or decoded output rows divided by duration.
- `mib_per_second`: encoded image bytes processed per second.
- `relative_*_change_pct`: Vortex change relative to a matching Parquet case.
- p95 uses the inclusive percentile method.

Warm-ups and failures remain in `results.json` but are excluded from numeric
summaries. A failed required case makes the benchmark exit non-zero.

## Known Boundary: Daft Uses the LakeSoul Arrow Bridge

Daft does not use a native LakeSoul connector here. LakeSoul performs the
projected Arrow scan, then Daft or Daft-Ray performs caption expansion and
image decoding after Arrow ingestion. Daft-Ray therefore distributes
processing after LakeSoul Arrow ingestion.

## Tests

```bash
.venv/bin/python -m compileall \
  import_data.py multimodal_data.py data_backends.py daft_worker.py \
  benchmark_models.py benchmark_report.py benchmark.py train_retrieval.py tests

.venv/bin/python -m pytest -m "not integration" -v
```

The integration test creates temporary LakeSoul tables and exercises all four
processing modes:

```bash
RUN_LAKESOUL_INTEGRATION=1 \
  .venv/bin/python -m pytest \
  tests/integration/test_dual_format_pipeline.py -v --timeout=180
```
