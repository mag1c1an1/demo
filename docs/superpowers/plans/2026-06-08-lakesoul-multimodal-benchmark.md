# LakeSoul Multimodal Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Flickr30k demo that imports matching Parquet and Vortex LakeSoul tables, benchmarks storage and multimodal processing through native/Ray/Daft backends, emits JSON/CSV/PNG reports, and feeds the same backends into CLIP retrieval training.

**Architecture:** Keep LakeSoul storage access, framework-specific processing, benchmark statistics, reporting, and model training behind focused modules. The shared `Sample` and `ImageRecord` contracts define deterministic limiting, image-level splits, caption expansion, and canonical hashing. Ray reads through LakeSoul's bundled datasource; Daft consumes LakeSoul Arrow streams and runs in isolated native or Ray-runner processes.

**Tech Stack:** Python 3.10, LakeSoul, Spark, PyArrow, Ray 2.55, Daft, PyTorch, Transformers, Pillow, Matplotlib, pytest.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `multimodal_data.py` | Immutable data contracts, deterministic selection/splits, caption expansion, image validation, canonical hashes |
| `import_data.py` | CLI, source parsing, Spark setup, paired writes, import timing, table validation |
| `data_backends.py` | Common `native`, `ray`, and `daft` sample/image iteration API |
| `daft_worker.py` | One-process Daft native/Ray execution entry point |
| `benchmark_models.py` | Raw run, summary, and environment metadata dataclasses |
| `benchmark_report.py` | Aggregation, JSON/CSV persistence, and PNG charts |
| `benchmark.py` | Scenario definitions, timing, subprocess orchestration, CLI |
| `train_retrieval.py` | CLIP training/evaluation using the common backend API |
| `tests/conftest.py` | Small deterministic JPEG and annotation fixtures |
| `tests/test_runtime_capabilities.py` | LakeSoul, Ray, and Daft runtime capability smoke tests |
| `tests/test_multimodal_data.py` | Data-rule unit tests |
| `tests/test_import_data.py` | Import CLI and paired-write unit tests |
| `tests/test_data_backends.py` | Backend contract tests |
| `tests/test_benchmark_report.py` | Statistics and report-output tests |
| `tests/test_benchmark.py` | Scenario timing and failure-record tests |
| `tests/test_train_retrieval.py` | Training CLI and dataset-adapter tests |
| `tests/integration/test_dual_format_pipeline.py` | Optional Spark/LakeSoul/Ray/Daft end-to-end smoke test |
| `README.md` | Reproduction, interpretation, local/remote Ray, Daft, and training instructions |

## Task 1: Establish Test Tooling and Runtime Capabilities

**Files:**
- Modify: `pyproject.toml:1-19`
- Modify: `.gitignore:1-27`
- Modify: `uv.lock`
- Create: `tests/test_runtime_capabilities.py`

- [ ] **Step 1: Add the runtime capability test**

```python
# tests/test_runtime_capabilities.py
def test_runtime_capabilities_are_available():
    import daft
    import ray

    assert callable(daft.from_arrow)
    assert callable(daft.set_runner_native)
    assert callable(daft.set_runner_ray)
    decode_expr = daft.col("image_blob").image.decode(
        on_error="null",
        mode="RGB",
    )
    assert decode_expr is not None
    assert ray.__version__


def test_daft_native_runner_smoke():
    import daft

    daft.set_runner_native(num_threads=1)
    assert daft.from_pydict({"value": [1, 2]}).to_arrow().num_rows == 2
```

- [ ] **Step 2: Run the test and verify the missing dependency failure**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache .venv/bin/python -m pytest \
  tests/test_runtime_capabilities.py -v
```

Expected: FAIL because `pytest` and `daft` are not installed.

- [ ] **Step 3: Add demo and test dependencies**

Update `pyproject.toml` to:

```toml
[project]
name = "multi"
version = "0.1.0"
description = "LakeSoul multimodal storage and training benchmark"
readme = "README.md"
requires-python = ">=3.10,<3.11"
dependencies = [
    "daft[ray]",
    "httpx[socks]>=0.28.1",
    "lakesoul[all]",
    "matplotlib>=3.9,<4",
    "pillow>=12.2.0",
    "transformers>=4.45,<5.0",
]

[dependency-groups]
dev = [
    "pytest>=8.3,<9",
    "pytest-timeout>=2.3,<3",
]

[tool.pytest.ini_options]
markers = [
    "integration: requires Spark, LakeSoul metadata, and local table storage",
]
testpaths = ["tests"]

[tool.uv.sources]
lakesoul = { path = "lakesoul-1.2.0.dev0-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" }

[[tool.uv.index]]
url = "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
default = true
```

Append to `.gitignore`:

```gitignore

# Demo outputs
benchmark-results/
clip-flickr30k-finetuned/
.pytest_cache/
```

Resolve dependencies:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync
```

Expected: dependency resolution installs a Daft build compatible with the Ray 2.55 LakeSoul runtime, plus Matplotlib and pytest. Do not add exact Ray or Daft version assertions.

- [ ] **Step 4: Run native and Ray capability smoke tests**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache .venv/bin/python -m pytest \
  tests/test_runtime_capabilities.py -v
```

Then run:

```bash
RAY_DEDUP_LOGS=0 .venv/bin/python -c \
  'import daft, ray; ray.init(num_cpus=1, include_dashboard=False); daft.set_runner_ray(noop_if_initialized=True); assert daft.from_pydict({"x":[1]}).to_arrow().num_rows == 1; ray.shutdown()'
```

Expected: both commands PASS against the LakeSoul Ray 2.55 environment. If the smoke test fails, report the missing runtime capability rather than introducing old-version pins.

- [ ] **Step 5: Commit the runtime baseline**

```bash
git add pyproject.toml uv.lock .gitignore tests/test_runtime_capabilities.py
git commit -m "build: add benchmark runtime dependencies"
```

## Task 2: Implement Deterministic Multimodal Data Contracts

**Files:**
- Create: `multimodal_data.py`
- Create: `tests/conftest.py`
- Create: `tests/test_multimodal_data.py`

- [ ] **Step 1: Write failing tests for selection, split isolation, expansion, hashes, and corrupt images**

```python
# tests/test_multimodal_data.py
import pytest

from multimodal_data import (
    ImageRecord,
    Sample,
    canonical_record_digest,
    canonical_sample_digest,
    decode_rgb,
    expand_captions,
    select_filenames,
    split_filenames,
)


def test_select_filenames_is_stable_and_limit_zero_means_all():
    names = ["c.jpg", "a.jpg", "b.jpg", "d.jpg"]
    assert select_filenames(names, limit=0, seed=7) == tuple(sorted(names))
    assert select_filenames(names, limit=2, seed=7) == select_filenames(
        reversed(names), limit=2, seed=7
    )


def test_split_happens_by_filename_before_caption_expansion():
    names = [f"{index}.jpg" for index in range(20)]
    splits = split_filenames(names, seed=11)
    assert set(splits["train"]).isdisjoint(splits["val"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert set(splits["val"]).isdisjoint(splits["test"])
    assert set().union(*map(set, splits.values())) == set(names)


def test_expand_captions_preserves_filename_and_blob():
    record = ImageRecord("1.jpg", b"jpeg", 10, 20, ("one", "two"), ())
    assert expand_captions(record) == (
        Sample("1.jpg", b"jpeg", "one"),
        Sample("1.jpg", b"jpeg", "two"),
    )


def test_canonical_digests_are_order_independent_and_content_sensitive():
    left = ImageRecord("1.jpg", b"a", 10, 20, ("one",), ())
    right = ImageRecord("2.jpg", b"b", 20, 10, ("two",), ())
    assert canonical_record_digest([left, right]) == canonical_record_digest(
        [right, left]
    )
    assert canonical_sample_digest(expand_captions(left)) != canonical_sample_digest(
        (Sample("1.jpg", b"a", "changed"),)
    )


def test_decode_rgb_raises_for_corrupt_bytes():
    with pytest.raises(ValueError, match="cannot decode image"):
        decode_rgb(b"not-an-image")
```

`tests/conftest.py` provides reusable encoded images:

```python
import io

import pytest
from PIL import Image


@pytest.fixture
def jpeg_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), color=(20, 40, 60)).save(buffer, format="JPEG")
    return buffer.getvalue()
```

- [ ] **Step 2: Run tests and verify RED**

```bash
.venv/bin/python -m pytest tests/test_multimodal_data.py -v
```

Expected: FAIL with `ModuleNotFoundError: multimodal_data`.

- [ ] **Step 3: Implement the shared contracts and deterministic rules**

Create `multimodal_data.py` with:

```python
from __future__ import annotations

import hashlib
import io
import json
import random
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

from PIL import Image

DEFAULT_SPLIT_RATIO = (0.937, 0.0315, 0.0315)


@dataclass(frozen=True)
class BoundingBox:
    chain_ids: tuple[str, ...]
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass(frozen=True)
class ImageRecord:
    filename: str
    image_bytes: bytes
    width: int
    height: int
    captions: tuple[str, ...]
    bboxes: tuple[BoundingBox, ...]


@dataclass(frozen=True)
class Sample:
    filename: str
    image_bytes: bytes
    caption: str


def select_filenames(
    filenames: Iterable[str], *, limit: int, seed: int
) -> tuple[str, ...]:
    if limit < 0:
        raise ValueError("limit must be non-negative")
    ordered = sorted(set(filenames))
    if limit == 0 or limit >= len(ordered):
        return tuple(ordered)
    rng = random.Random(seed)
    selected = rng.sample(ordered, limit)
    return tuple(sorted(selected))


def split_filenames(
    filenames: Iterable[str],
    *,
    seed: int,
    ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIO,
) -> Mapping[str, tuple[str, ...]]:
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("split ratios must contain three values summing to 1")
    ordered = sorted(set(filenames))
    random.Random(seed).shuffle(ordered)
    train_end = int(len(ordered) * ratios[0])
    val_end = train_end + int(len(ordered) * ratios[1])
    return {
        "train": tuple(ordered[:train_end]),
        "val": tuple(ordered[train_end:val_end]),
        "test": tuple(ordered[val_end:]),
    }


def expand_captions(record: ImageRecord) -> tuple[Sample, ...]:
    return tuple(
        Sample(record.filename, record.image_bytes, caption)
        for caption in record.captions
    )


def decode_rgb(image_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
        return image.convert("RGB")
    except Exception as exc:
        raise ValueError("cannot decode image") from exc


def _record_payload(record: ImageRecord) -> dict[str, object]:
    return {
        "filename": record.filename,
        "image_sha256": hashlib.sha256(record.image_bytes).hexdigest(),
        "width": record.width,
        "height": record.height,
        "captions": list(record.captions),
        "bboxes": [asdict(box) for box in record.bboxes],
    }


def canonical_record_digest(records: Iterable[ImageRecord]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.filename):
        payload = json.dumps(
            _record_payload(record), sort_keys=True, separators=(",", ":")
        ).encode()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def canonical_sample_digest(samples: Iterable[Sample]) -> str:
    digest = hashlib.sha256()
    keys = sorted(
        (
            sample.filename,
            hashlib.sha256(sample.image_bytes).hexdigest(),
            sample.caption,
        )
        for sample in samples
    )
    for key in keys:
        digest.update(json.dumps(key, separators=(",", ":")).encode())
    return digest.hexdigest()
```

- [ ] **Step 4: Run tests and verify GREEN**

```bash
.venv/bin/python -m pytest tests/test_multimodal_data.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit the shared data layer**

```bash
git add multimodal_data.py tests/conftest.py tests/test_multimodal_data.py
git commit -m "feat: add deterministic multimodal data contracts"
```

## Task 3: Rewrite Import as a Safe Paired Parquet/Vortex Operation

**Files:**
- Modify: `import_data.py:1-170`
- Create: `tests/test_import_data.py`

- [ ] **Step 1: Write failing tests for CLI defaults, source parsing, alternating write order, and overwrite safety**

```python
# tests/test_import_data.py
from pathlib import Path

import pytest

from import_data import (
    ImportConfig,
    alternating_formats,
    build_parser,
    ensure_targets_are_writable,
)


def test_import_cli_defaults_to_paired_tables_and_1000_rows():
    args = build_parser().parse_args([])
    assert args.limit == 1000
    assert args.parquet_table == "flickr30k_parquet"
    assert args.vortex_table == "flickr30k_vortex"
    assert args.overwrite is False


def test_alternating_formats_switches_first_writer_each_batch():
    assert alternating_formats(0) == ("parquet", "vortex")
    assert alternating_formats(1) == ("vortex", "parquet")


def test_existing_target_requires_explicit_overwrite():
    with pytest.raises(RuntimeError, match="already exist"):
        ensure_targets_are_writable(
            existing={"flickr30k_parquet"}, overwrite=False
        )


def test_import_config_rejects_same_table_name(tmp_path: Path):
    with pytest.raises(ValueError, match="must be different"):
        ImportConfig(
            data_dir=tmp_path,
            jar_path=tmp_path / "lake.jar",
            parquet_table="same",
            vortex_table="same",
            limit=1000,
            batch_size=100,
            seed=7,
            overwrite=False,
            output=tmp_path / "import.json",
        )
```

- [ ] **Step 2: Run tests and verify RED**

```bash
.venv/bin/python -m pytest tests/test_import_data.py -v
```

Expected: FAIL because the new import API does not exist.

- [ ] **Step 3: Refactor parsing and configuration without performing writes**

Replace global format/table constants with:

```python
@dataclass(frozen=True)
class ImportConfig:
    data_dir: Path
    jar_path: Path
    parquet_table: str
    vortex_table: str
    limit: int
    batch_size: int
    seed: int
    overwrite: bool
    output: Path

    def __post_init__(self) -> None:
        if self.parquet_table == self.vortex_table:
            raise ValueError("parquet and vortex table names must be different")
        if self.limit < 0:
            raise ValueError("limit must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch size must be positive")


def alternating_formats(batch_index: int) -> tuple[str, str]:
    return ("parquet", "vortex") if batch_index % 2 == 0 else ("vortex", "parquet")


def ensure_targets_are_writable(
    *, existing: set[str], overwrite: bool
) -> None:
    if existing and not overwrite:
        names = ", ".join(sorted(existing))
        raise RuntimeError(f"target tables already exist: {names}; use --overwrite")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--jar-path", type=Path, default=PROJECT_DIR / JAR_NAME)
    parser.add_argument("--parquet-table", default="flickr30k_parquet")
    parser.add_argument("--vortex-table", default="flickr30k_vortex")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark-results/import.json")
    )
    return parser
```

Convert `parse_bboxes` to return `BoundingBox` values and add:

```python
def load_source_records(config: ImportConfig) -> Iterator[ImageRecord]:
    captions_path = config.data_dir / "dataset_flickr30k_allEN.json"
    captions_data = json.loads(captions_path.read_text())
    selected = select_filenames(
        captions_data.keys(), limit=config.limit, seed=config.seed
    )
    for filename in selected:
        image_path = config.data_dir / "flickr30k-images" / filename
        xml_path = config.data_dir / "Annotations" / f"{Path(filename).stem}.xml"
        if not image_path.is_file() or not xml_path.is_file():
            raise FileNotFoundError(f"missing source files for {filename}")
        width, height = parse_image_size(xml_path)
        yield ImageRecord(
            filename=filename,
            image_bytes=image_path.read_bytes(),
            width=width,
            height=height,
            captions=tuple(clean_caption(c) for c in captions_data[filename]),
            bboxes=tuple(parse_bboxes(xml_path)),
        )
```

- [ ] **Step 4: Run unit tests and verify GREEN before Spark code**

```bash
.venv/bin/python -m pytest tests/test_import_data.py -v
```

Expected: PASS.

- [ ] **Step 5: Add paired Spark writes, validation, and atomic import metrics**

Implement these concrete functions in `import_data.py`:

```python
def record_to_row(record: ImageRecord) -> tuple:
    return (
        record.filename,
        record.image_bytes,
        record.width,
        record.height,
        list(record.captions),
        [
            {
                "chain_ids": list(box.chain_ids),
                "xmin": box.xmin,
                "ymin": box.ymin,
                "xmax": box.xmax,
                "ymax": box.ymax,
            }
            for box in record.bboxes
        ],
    )


def write_batch(
    spark,
    schema,
    rows: list[tuple],
    *,
    table_name: str,
    file_format: str,
    first_batch: bool,
) -> float:
    started = time.perf_counter_ns()
    writer = spark.createDataFrame(rows, schema).write.format("lakesoul")
    writer = writer.option("file_format", file_format)
    if not first_batch:
        writer = writer.mode("append")
    writer.saveAsTable(table_name)
    return (time.perf_counter_ns() - started) / 1_000_000_000


def validate_table(table_name: str) -> dict[str, object]:
    records = list(iter_lakesoul_records(table_name))
    return {
        "row_count": len(records),
        "filenames": sorted(record.filename for record in records),
        "digest": canonical_record_digest(records),
        "logical_blob_bytes": sum(len(record.image_bytes) for record in records),
    }
```

`iter_lakesoul_records` must scan projected LakeSoul Arrow batches and convert Arrow rows back to `ImageRecord`. `main()` must:

1. Validate paths.
2. Start Spark.
3. Detect existing target tables.
4. Drop both tables only when `--overwrite`.
5. Materialize each source batch once.
6. Call `write_batch` in `alternating_formats(batch_index)` order.
7. Accumulate write seconds per format.
8. Stop Spark in `finally`.
9. Validate source, Parquet, and Vortex digests.
10. Atomically write import JSON through `temporary_path.replace(output)`.

- [ ] **Step 6: Run focused and regression tests**

```bash
.venv/bin/python -m pytest tests/test_import_data.py tests/test_multimodal_data.py -v
```

Expected: PASS.

- [ ] **Step 7: Run a ten-image manual import smoke test**

```bash
.venv/bin/python import_data.py \
  --limit 10 \
  --batch-size 5 \
  --parquet-table flickr30k_parquet_smoke \
  --vortex-table flickr30k_vortex_smoke \
  --overwrite \
  --output /tmp/lakesoul-import-smoke.json
```

Expected: both tables have 10 rows and identical canonical digests; JSON records separate write times and physical sizes.

- [ ] **Step 8: Commit paired import**

```bash
git add import_data.py tests/test_import_data.py
git commit -m "feat: import matching parquet and vortex tables"
```

## Task 4: Add Native and Ray Data Backends

**Files:**
- Create: `data_backends.py`
- Create: `tests/test_data_backends.py`

- [ ] **Step 1: Write failing backend contract tests using an injected record source**

```python
# tests/test_data_backends.py
from multimodal_data import ImageRecord, canonical_sample_digest
from data_backends import BackendConfig, iter_samples_from_records


def records():
    return [
        ImageRecord("a.jpg", b"a", 1, 1, ("a1", "a2"), ()),
        ImageRecord("b.jpg", b"b", 1, 1, ("b1",), ()),
        ImageRecord("c.jpg", b"c", 1, 1, ("c1",), ()),
    ]


def test_native_backend_splits_before_caption_expansion():
    config = BackendConfig(
        backend="native",
        table_name="unused",
        split="train",
        batch_size=2,
        seed=3,
        limit=0,
        ray_address="local",
        daft_runner="native",
        skip_corrupt=False,
    )
    samples = tuple(iter_samples_from_records(records(), config))
    selected_names = {sample.filename for sample in samples}
    assert all(
        (record.filename in selected_names) == bool(
            set(record.captions) & {sample.caption for sample in samples}
        )
        for record in records()
    )


def test_backend_digest_does_not_depend_on_input_order():
    config = BackendConfig(
        "native", "unused", "all", 2, 3, 0, "local", "native", False
    )
    forward = iter_samples_from_records(records(), config)
    reverse = iter_samples_from_records(reversed(records()), config)
    assert canonical_sample_digest(forward) == canonical_sample_digest(reverse)
```

- [ ] **Step 2: Run tests and verify RED**

```bash
.venv/bin/python -m pytest tests/test_data_backends.py -v
```

Expected: FAIL with `ModuleNotFoundError: data_backends`.

- [ ] **Step 3: Implement backend configuration and native iteration**

```python
@dataclass(frozen=True)
class BackendConfig:
    backend: str
    table_name: str
    split: str
    batch_size: int
    seed: int
    limit: int
    ray_address: str | None
    daft_runner: str
    skip_corrupt: bool


def choose_records(
    records: Iterable[ImageRecord], config: BackendConfig
) -> list[ImageRecord]:
    materialized = sorted(records, key=lambda item: item.filename)
    selected_names = set(
        select_filenames(
            (record.filename for record in materialized),
            limit=config.limit,
            seed=config.seed,
        )
    )
    selected = [r for r in materialized if r.filename in selected_names]
    if config.split == "all":
        return selected
    splits = split_filenames(
        (record.filename for record in selected), seed=config.seed
    )
    allowed = set(splits[config.split])
    return [record for record in selected if record.filename in allowed]


def iter_samples_from_records(
    records: Iterable[ImageRecord], config: BackendConfig
) -> Iterator[Sample]:
    for record in choose_records(records, config):
        yield from expand_captions(record)


def iter_native_records(
    table_name: str, *, batch_size: int
) -> Iterator[ImageRecord]:
    dataset = lakesoul_dataset(table_name, batch_size=batch_size)
    for batch in dataset.to_batches(
        columns=["filename", "image_blob", "width", "height", "captions", "bboxes"]
    ):
        yield from arrow_batch_to_records(batch)
```

`arrow_batch_to_records` must normalize Arrow list/struct values into immutable tuples.

- [ ] **Step 4: Add Ray initialization and LakeSoul Ray record iteration**

```python
def init_ray(address: str | None) -> None:
    import ray

    if ray.is_initialized():
        return
    if address in (None, "local"):
        ray.init(include_dashboard=False)
    elif address == "auto":
        ray.init(address="auto")
    else:
        ray.init(address=address)


def iter_ray_records(config: BackendConfig) -> Iterator[ImageRecord]:
    init_ray(config.ray_address)
    import lakesoul.ray  # registers ray.data.read_lakesoul
    import ray

    dataset = ray.data.read_lakesoul(
        config.table_name, batch_size=config.batch_size
    )
    for batch in dataset.iter_batches(batch_format="pyarrow"):
        yield from arrow_batch_to_records(batch)
```

Dispatch with:

```python
def iter_records(config: BackendConfig) -> Iterator[ImageRecord]:
    if config.backend == "native":
        yield from iter_native_records(
            config.table_name, batch_size=config.batch_size
        )
    elif config.backend == "ray":
        yield from iter_ray_records(config)
    else:
        raise ValueError(
            f"backend does not expose raw ImageRecord values: {config.backend}"
        )


def iter_samples(config: BackendConfig) -> Iterator[Sample]:
    if config.backend == "daft":
        yield from iter_daft_samples(config)
    else:
        yield from iter_samples_from_records(iter_records(config), config)
```

- [ ] **Step 5: Run unit tests and local Ray smoke**

```bash
.venv/bin/python -m pytest tests/test_data_backends.py -v
```

Then:

```bash
.venv/bin/python -c \
  'from data_backends import BackendConfig, init_ray; init_ray("local"); import ray; assert ray.is_initialized(); ray.shutdown()'
```

Expected: PASS.

- [ ] **Step 6: Commit native and Ray backends**

```bash
git add data_backends.py tests/test_data_backends.py
git commit -m "feat: add native and ray lakesoul backends"
```

## Task 5: Add Daft Native and Daft-Ray Processing

**Files:**
- Modify: `data_backends.py`
- Create: `daft_worker.py`
- Modify: `tests/test_data_backends.py`

- [ ] **Step 1: Add failing tests for Daft conversion and runner command construction**

```python
def test_build_daft_worker_command_contains_runner_and_table():
    from data_backends import build_daft_worker_command

    command = build_daft_worker_command(
        BackendConfig(
            "daft", "flickr30k_vortex", "train", 32, 7, 1000,
            "ray://head:10001", "ray", False
        )
    )
    assert command[-4:] == [
        "--runner", "ray", "--table", "flickr30k_vortex"
    ]
    assert "--ray-address" in command


def test_daft_arrow_transform_decodes_and_expands(jpeg_bytes):
    import pyarrow as pa
    from daft_worker import transform_arrow

    table = pa.table(
        {
            "filename": ["a.jpg"],
            "image_blob": [jpeg_bytes],
            "width": [8],
            "height": [6],
            "captions": [["one", "two"]],
            "bboxes": [[]],
        }
    )
    batches = list(transform_arrow(table, skip_corrupt=False))
    rows = pa.Table.from_batches(batches).to_pylist()
    assert [row["caption"] for row in rows] == ["one", "two"]
    assert all(row["decoded_width"] == 8 for row in rows)
    assert all(row["decoded_height"] == 6 for row in rows)
```

- [ ] **Step 2: Run tests and verify RED**

```bash
.venv/bin/python -m pytest \
  tests/test_data_backends.py::test_build_daft_worker_command_contains_runner_and_table \
  tests/test_data_backends.py::test_daft_arrow_transform_decodes_and_expands -v
```

Expected: FAIL because Daft worker functions do not exist.

- [ ] **Step 3: Implement the Daft Arrow transformation**

Create `daft_worker.py`:

```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator

import daft
import pyarrow as pa


def configure_runner(runner: str, ray_address: str | None) -> None:
    if runner == "native":
        daft.set_runner_native()
    elif runner == "ray":
        address = None if ray_address in (None, "local") else ray_address
        daft.set_runner_ray(address=address, noop_if_initialized=True)
    else:
        raise ValueError(f"unsupported Daft runner: {runner}")


def transform_arrow(
    data: pa.Table | Iterable[pa.RecordBatch], *, skip_corrupt: bool
) -> Iterator[pa.RecordBatch]:
    arrow_input = (
        data
        if isinstance(data, pa.Table)
        else (pa.Table.from_batches([batch]) for batch in data)
    )
    frame = daft.from_arrow(arrow_input)
    frame = frame.explode("captions").with_column_renamed("captions", "caption")
    decoded = frame["image_blob"].image.decode(
        on_error="null" if skip_corrupt else "raise",
        mode="RGB",
    )
    frame = frame.with_column("decoded_image", decoded)
    if skip_corrupt:
        frame = frame.where(frame["decoded_image"].not_null())
    for batch in frame.select(
        "filename",
        "image_blob",
        "caption",
        "width",
        "height",
        "decoded_image",
    ).to_arrow_iter():
        output_rows = []
        for row in batch.to_pylist():
            image = row["decoded_image"]
            if image is None:
                continue
            if image.shape[:2] != (row["height"], row["width"]):
                raise ValueError(
                    f"decoded size mismatch for {row['filename']}: "
                    f"{image.shape[:2]} != {(row['height'], row['width'])}"
                )
            output_rows.append(
                {
                    "filename": row["filename"],
                    "image_blob": row["image_blob"],
                    "caption": row["caption"],
                    "decoded_width": image.shape[1],
                    "decoded_height": image.shape[0],
                }
            )
        if output_rows:
            yield pa.RecordBatch.from_pylist(output_rows)
```

The CLI configures the runner before constructing a DataFrame. It performs a metadata-only LakeSoul scan to choose the deterministic filename set with `select_filenames` and `split_filenames`, then performs a fresh projected scan for `filename`, `image_blob`, `captions`, `width`, and `height`. It filters batches to the selected filename set before passing the Arrow iterator to `transform_arrow`.

The CLI writes expanded sample rows to an Arrow IPC stream with this schema:

```text
filename: string
image_blob: binary
caption: string
decoded_width: int64
decoded_height: int64
```

It also writes a JSON status file containing rows, encoded bytes, decode errors, duration, canonical sample digest, and bounded exception details.

- [ ] **Step 4: Implement subprocess isolation in `data_backends.py`**

```python
def build_daft_worker_command(config: BackendConfig) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("daft_worker.py")),
        "--batch-size", str(config.batch_size),
        "--seed", str(config.seed),
        "--limit", str(config.limit),
        "--split", config.split,
    ]
    if config.skip_corrupt:
        command.append("--skip-corrupt")
    if config.ray_address:
        command.extend(["--ray-address", config.ray_address])
    command.extend(
        ["--runner", config.daft_runner, "--table", config.table_name]
    )
    return command


def iter_daft_samples(config: BackendConfig) -> Iterator[Sample]:
    with tempfile.TemporaryDirectory(prefix="lakesoul-daft-") as directory:
        output = Path(directory) / "samples.arrow"
        status = Path(directory) / "status.json"
        command = build_daft_worker_command(config)
        command.extend(["--output", str(output), "--status", str(status)])
        completed = subprocess.run(command, check=False, text=True)
        if completed.returncode:
            message = json.loads(status.read_text()).get(
                "error_message", "Daft worker failed"
            )
            raise RuntimeError(message)
        with pa.memory_map(output, "r") as source:
            reader = pa.ipc.open_stream(source)
            for batch in reader:
                for row in batch.to_pylist():
                    yield Sample(
                        filename=row["filename"],
                        image_bytes=row["image_blob"],
                        caption=row["caption"],
                    )
```

`iter_samples` dispatches directly to `iter_daft_samples` for the Daft backend. Raw `iter_records` remains intentionally limited to native and Ray, because Daft owns caption expansion and image decoding rather than exposing an imitation native record scan.

- [ ] **Step 5: Verify Daft native and Daft-Ray**

```bash
.venv/bin/python -m pytest tests/test_data_backends.py -v
```

Run both smoke modes:

```bash
.venv/bin/python daft_worker.py \
  --runner native \
  --table flickr30k_vortex_smoke \
  --split all \
  --limit 2 \
  --batch-size 2 \
  --output /tmp/daft-native.arrow \
  --status /tmp/daft-native.json
```

```bash
.venv/bin/python daft_worker.py \
  --runner ray \
  --ray-address local \
  --table flickr30k_vortex_smoke \
  --split all \
  --limit 2 \
  --batch-size 2 \
  --output /tmp/daft-ray.arrow \
  --status /tmp/daft-ray.json
```

Expected: both status files report success and identical canonical sample digests.

- [ ] **Step 6: Commit Daft backends**

```bash
git add data_backends.py daft_worker.py tests/test_data_backends.py
git commit -m "feat: add daft multimodal backends"
```

## Task 6: Add Benchmark Models, Statistics, and Reports

**Files:**
- Create: `benchmark_models.py`
- Create: `benchmark_report.py`
- Create: `tests/test_benchmark_report.py`

- [ ] **Step 1: Write failing statistics and output tests**

```python
# tests/test_benchmark_report.py
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
```

- [ ] **Step 2: Run tests and verify RED**

```bash
.venv/bin/python -m pytest tests/test_benchmark_report.py -v
```

Expected: FAIL because benchmark model/report modules do not exist.

- [ ] **Step 3: Implement normalized dataclasses**

```python
# benchmark_models.py
from dataclasses import asdict, dataclass


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

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


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
```

- [ ] **Step 4: Implement aggregation, atomic JSON/CSV, and deterministic charts**

Use:

```python
def percentile(values: Sequence[float], quantile: float) -> float:
    return statistics.quantiles(values, n=100, method="inclusive")[
        int(quantile * 100) - 1
    ] if len(values) > 1 else values[0]


def relative_change(current: float, baseline: float) -> float:
    if baseline == 0:
        raise ValueError("baseline must be non-zero")
    return (current / baseline - 1.0) * 100.0
```

`aggregate_runs` must group by `(scenario, format, backend, runner)`, exclude warm-ups and failures from numeric summaries, retain failed counts, then pair Vortex with matching Parquet groups for relative changes.

`write_reports` must:

1. Create the output directory.
2. Write `results.json.tmp`, then replace `results.json`.
3. Write flattened raw and summary CSV files.
4. Force Matplotlib's `Agg` backend.
5. Create four PNGs even when a category has no data; the empty chart must say "No successful runs".
6. Close every figure.

- [ ] **Step 5: Run report tests**

```bash
.venv/bin/python -m pytest tests/test_benchmark_report.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit benchmark reporting**

```bash
git add benchmark_models.py benchmark_report.py tests/test_benchmark_report.py
git commit -m "feat: add benchmark statistics and reports"
```

## Task 7: Implement Benchmark Scenarios and Orchestration

**Files:**
- Create: `benchmark.py`
- Create: `tests/test_benchmark.py`

- [ ] **Step 1: Write failing tests for timing, failure records, warm-ups, and CLI defaults**

```python
# tests/test_benchmark.py
from benchmark import BenchmarkConfig, build_parser, execute_case


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
```

- [ ] **Step 2: Run tests and verify RED**

```bash
.venv/bin/python -m pytest tests/test_benchmark.py -v
```

Expected: FAIL because `benchmark.py` does not exist.

- [ ] **Step 3: Implement CLI, timing, and scenario result normalization**

```python
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


def execute_case(
    scenario: str,
    format_name: str,
    backend: str,
    runner: str,
    warmup: bool,
    operation: Callable[[], Mapping[str, int]],
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
```

- [ ] **Step 4: Implement concrete operations**

Add operations with exact projections:

```python
SCAN_COLUMNS = {
    "metadata_scan": ["filename", "width", "height", "captions"],
    "full_scan": None,
    "filtered_scan": ["filename", "image_blob", "captions"],
    "blob_scan": ["filename", "image_blob"],
}
```

Implement:

- `scan_operation(table, scenario, filenames)` using fresh `lakesoul_dataset` instances.
- `decode_operation(config)` using `iter_records`, `decode_rgb`, dimension checks, and error counting for native/Ray.
- `expand_operation(config)` counting source records and caption samples for native/Ray.
- `batch_operation(config)` decoding and grouping samples into `batch_size` chunks without loading CLIP for native/Ray.
- Daft decode, expansion, and batching operations through `daft_worker.py`, using worker status metrics because Daft runner selection is process-global.

The filtered filename list must come from the same seeded selected filename set for both tables.

- [ ] **Step 5: Implement environment capture, table validation, run loop, and atomic checkpoints**

The orchestrator must:

```python
for case in cases:
    for index in range(config.warmups + config.runs):
        warmup = index < config.warmups
        result = execute_case(
            scenario=case.scenario,
            format_name=case.format_name,
            backend=case.backend,
            runner=case.runner,
            warmup=warmup,
            operation=case.operation,
        )
        runs.append(result)
        write_raw_json(runs, metadata, config.output_dir / "results.json")
```

Before the loop:

1. Validate both tables with the canonical digest code from import.
2. Fail if row counts, schemas, filename sets, or digests differ.
3. Initialize Ray once when a requested backend needs it.
4. Check Daft availability and honor `--allow-missing-optional`.
5. Capture Git state, host, CPU, memory, GPU, Java, Python, and package versions without secret values.

After the loop:

1. Call `write_reports`.
2. Print a compact console table.
3. Exit non-zero if required cases failed.

- [ ] **Step 6: Run benchmark unit tests**

```bash
.venv/bin/python -m pytest tests/test_benchmark.py tests/test_benchmark_report.py -v
```

Expected: PASS.

- [ ] **Step 7: Run a small local benchmark**

```bash
.venv/bin/python benchmark.py \
  --parquet-table flickr30k_parquet_smoke \
  --vortex-table flickr30k_vortex_smoke \
  --limit 10 \
  --warmups 1 \
  --runs 1 \
  --ray-address local \
  --output-dir /tmp/lakesoul-benchmark-smoke
```

Expected: JSON, two CSVs, and four PNGs exist; successful storage and processing scenarios contain non-zero duration and throughput.

- [ ] **Step 8: Commit benchmark orchestration**

```bash
git add benchmark.py tests/test_benchmark.py
git commit -m "feat: benchmark lakesoul multimodal pipelines"
```

## Task 8: Refactor Retrieval Training Around the Backend Contract

**Files:**
- Modify: `train_retrieval.py:1-261`
- Create: `tests/test_train_retrieval.py`

- [ ] **Step 1: Write failing tests for CLI and iterable sample decoding**

```python
# tests/test_train_retrieval.py
from multimodal_data import Sample
from train_retrieval import LakeSoulSampleDataset, build_parser


def test_training_cli_selects_backend_and_table():
    args = build_parser().parse_args(
        ["--backend", "ray", "--table", "flickr30k_vortex", "--epochs", "1"]
    )
    assert args.backend == "ray"
    assert args.table == "flickr30k_vortex"
    assert args.epochs == 1


def test_dataset_decodes_samples(jpeg_bytes):
    dataset = LakeSoulSampleDataset(
        sample_factory=lambda: iter([Sample("a.jpg", jpeg_bytes, "caption")])
    )
    image, caption = next(iter(dataset))
    assert image.mode == "RGB"
    assert image.size == (8, 6)
    assert caption == "caption"
```

- [ ] **Step 2: Run tests and verify RED**

```bash
.venv/bin/python -m pytest tests/test_train_retrieval.py -v
```

Expected: FAIL because the CLI and iterable adapter do not exist.

- [ ] **Step 3: Replace DuckDB point queries with an iterable backend adapter**

Remove `duckdb`, `random`, the map-style `Flickr30kDataset`, and debug prints. Add:

```python
class LakeSoulSampleDataset(torch.utils.data.IterableDataset):
    def __init__(self, sample_factory):
        self._sample_factory = sample_factory

    def __iter__(self):
        for sample in self._sample_factory():
            yield decode_rgb(sample.image_bytes), sample.caption


def make_dataset(args, split: str) -> LakeSoulSampleDataset:
    config = BackendConfig(
        backend=args.backend,
        table_name=args.table,
        split=split,
        batch_size=args.data_batch_size,
        seed=args.seed,
        limit=args.limit,
        ray_address=args.ray_address,
        daft_runner=args.daft_runner,
        skip_corrupt=args.skip_corrupt,
    )
    return LakeSoulSampleDataset(lambda: iter_samples(config))
```

Add CLI options matching the spec:

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["native", "ray", "daft"], default="native")
    parser.add_argument("--table", default="flickr30k_vortex")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--ray-address", default="local")
    parser.add_argument("--daft-runner", choices=["native", "ray"], default="native")
    parser.add_argument("--skip-corrupt", action="store_true")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--output-dir", default="clip-flickr30k-finetuned")
    return parser
```

Keep model, loss, optimizer, evaluation, and save logic shared across backends.

- [ ] **Step 4: Correct data-loader and evaluation semantics**

Use separate iterable datasets for each split:

```python
train_loader = DataLoader(
    make_dataset(args, "train"),
    batch_size=args.batch_size,
    collate_fn=lambda batch: collate_fn(batch, processor),
    num_workers=0,
)
val_loader = DataLoader(
    make_dataset(args, "val"),
    batch_size=args.batch_size,
    collate_fn=lambda batch: collate_fn(batch, processor),
    num_workers=0,
)
```

Do not call `shuffle=True` for an `IterableDataset`; deterministic backend selection already controls sample order. Guard empty loaders and avoid dividing by zero in `train_one_epoch`.

- [ ] **Step 5: Run training unit tests and existing loss smoke**

```bash
.venv/bin/python -m pytest tests/test_train_retrieval.py -v
```

Then:

```bash
.venv/bin/python -c \
  'import torch; from train_retrieval import contrastive_loss; x=torch.eye(2); assert contrastive_loss(x,x).item() >= 0'
```

Expected: PASS.

- [ ] **Step 6: Run one-batch data-only smoke for every backend**

Add `--dry-run-batches 1` to stop after collating one batch without loading or training the model. Run:

```bash
for backend in native ray daft; do
  .venv/bin/python train_retrieval.py \
    --backend "$backend" \
    --table flickr30k_vortex_smoke \
    --limit 10 \
    --batch-size 2 \
    --dry-run-batches 1
done
```

Expected: every backend reports one valid image/text batch.

- [ ] **Step 7: Commit training backend integration**

```bash
git add train_retrieval.py tests/test_train_retrieval.py
git commit -m "feat: train retrieval from selectable data backends"
```

## Task 9: Add End-to-End Integration Coverage

**Files:**
- Create: `tests/integration/test_dual_format_pipeline.py`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Write the marked integration test**

```python
# tests/integration/test_dual_format_pipeline.py
import os

import pytest

from benchmark_report import write_reports
from data_backends import BackendConfig, iter_samples
from import_data import import_records
from multimodal_data import canonical_sample_digest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.environ.get("RUN_LAKESOUL_INTEGRATION") != "1",
    reason="set RUN_LAKESOUL_INTEGRATION=1 to run LakeSoul integration",
)
def test_dual_format_backends_and_reports(
    spark_session, tiny_image_records, tmp_path
):
    suffix = os.getpid()
    parquet_table = f"flickr30k_parquet_test_{suffix}"
    vortex_table = f"flickr30k_vortex_test_{suffix}"
    result = import_records(
        spark_session,
        tiny_image_records,
        parquet_table=parquet_table,
        vortex_table=vortex_table,
        batch_size=2,
    )
    assert result["parquet"]["digest"] == result["vortex"]["digest"]

    digests = {}
    for backend, runner in [
        ("native", "native"),
        ("ray", "native"),
        ("daft", "native"),
        ("daft", "ray"),
    ]:
        config = BackendConfig(
            backend,
            vortex_table,
            "all",
            2,
            7,
            0,
            "local",
            runner,
            False,
        )
        digests[f"{backend}-{runner}"] = canonical_sample_digest(
            iter_samples(config)
        )
    assert len(set(digests.values())) == 1
    assert write_reports([], {"integration": True}, tmp_path)
```

- [ ] **Step 2: Run default tests and verify integration is skipped**

```bash
.venv/bin/python -m pytest -m "not integration" -v
```

Expected: all unit tests PASS; integration test is not selected.

- [ ] **Step 3: Add Spark and tiny-record fixtures**

`tests/conftest.py` must:

- Generate four JPEG `ImageRecord` values with deterministic sizes/captions.
- Start a local Spark session using the project JAR.
- Configure the same LakeSoul catalog and extensions as `import_data.py`.
- Stop Spark after the session.
- Skip with a precise message if the JAR is absent.

- [ ] **Step 4: Run the full local integration test**

```bash
RUN_LAKESOUL_INTEGRATION=1 \
  .venv/bin/python -m pytest \
  tests/integration/test_dual_format_pipeline.py -v --timeout=180
```

Expected: PASS with matching Parquet/Vortex content and all four backend digests.

- [ ] **Step 5: Commit integration coverage**

```bash
git add tests/conftest.py tests/integration/test_dual_format_pipeline.py
git commit -m "test: cover dual format multimodal pipeline"
```

## Task 10: Document Reproduction and Run Final Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write README sections with executable commands**

Use this structure:

```markdown
# LakeSoul Multimodal Demo

## What It Demonstrates
## Prerequisites
## Install
## Import Matching Parquet and Vortex Tables
## Run the 1,000-Image Benchmark
## Run the Full Flickr30k Benchmark
## Use a Remote Ray Cluster
## Run Daft Native and Daft-Ray
## Train CLIP Retrieval
## Understand the Metrics
## Known Boundary: Daft Uses the LakeSoul Arrow Bridge
## Tests
```

Include the exact quick path:

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

State explicitly that Daft-Ray is distributed processing after LakeSoul Arrow ingestion, not a native Daft LakeSoul connector.

- [ ] **Step 2: Run formatting and static syntax checks**

```bash
.venv/bin/python -m compileall \
  import_data.py multimodal_data.py data_backends.py daft_worker.py \
  benchmark_models.py benchmark_report.py benchmark.py train_retrieval.py tests
```

Expected: all files compile.

- [ ] **Step 3: Run all unit tests**

```bash
.venv/bin/python -m pytest -m "not integration" -v
```

Expected: PASS.

- [ ] **Step 4: Run the integration suite**

```bash
RUN_LAKESOUL_INTEGRATION=1 \
  .venv/bin/python -m pytest -m integration -v --timeout=180
```

Expected: PASS.

- [ ] **Step 5: Run the documented quick benchmark**

```bash
.venv/bin/python import_data.py \
  --limit 1000 \
  --overwrite \
  --output benchmark-results/import.json

.venv/bin/python benchmark.py \
  --limit 1000 \
  --warmups 1 \
  --runs 3 \
  --ray-address local \
  --output-dir benchmark-results
```

Expected:

- Both tables validate with equal logical digests.
- Required scenarios have three successful measured runs.
- `results.json`, `results.csv`, `summary.csv`, and four PNGs exist.
- Console summary reports Parquet/Vortex size, latency, and throughput changes.

- [ ] **Step 6: Inspect repository state and benchmark artifacts**

```bash
git status --short
find benchmark-results -maxdepth 1 -type f -printf "%f %s bytes\n" | sort
```

Expected: only intentional source/documentation changes are tracked; benchmark outputs remain ignored.

- [ ] **Step 7: Commit documentation**

```bash
git add README.md
git commit -m "docs: explain multimodal benchmark workflow"
```

- [ ] **Step 8: Perform final verification before completion**

Run:

```bash
git status --short
git log --oneline -10
.venv/bin/python -m pytest -m "not integration" -q
```

Expected: clean worktree and all unit tests PASS. Report integration and 1,000-image benchmark evidence separately, including any scenarios skipped because external cluster services were unavailable.
