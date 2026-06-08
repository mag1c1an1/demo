from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal, TypedDict, cast

import pyarrow as pa

from multimodal_data import (
    BoundingBox,
    ImageRecord,
    Sample,
    expand_captions,
    select_filenames,
    split_filenames,
)


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

    def __post_init__(self) -> None:
        if self.backend not in {"native", "ray", "daft"}:
            raise ValueError(f"unsupported backend: {self.backend}")
        if self.split not in {"all", "train", "val", "test"}:
            raise ValueError(f"unsupported split: {self.split}")
        if self.batch_size <= 0:
            raise ValueError("batch size must be positive")
        if self.limit < 0:
            raise ValueError("limit must be non-negative")
        if self.daft_runner not in {"native", "ray"}:
            raise ValueError(f"unsupported Daft runner: {self.daft_runner}")


class BoundingBoxRow(TypedDict):
    chain_ids: list[str] | None
    xmin: int
    ymin: int
    xmax: int
    ymax: int


class DaftWorkerSuccess(TypedDict):
    success: Literal[True]
    input_rows: int
    output_rows: int
    input_bytes: int
    decode_errors: int
    duration_seconds: float
    digest: str


class DaftWorkerFailure(TypedDict):
    success: Literal[False]
    error_type: str
    error_message: str


DaftWorkerStatus = DaftWorkerSuccess | DaftWorkerFailure


def _normalize_box(value: BoundingBoxRow) -> BoundingBox:
    return BoundingBox(
        chain_ids=tuple(value["chain_ids"] or []),
        xmin=value["xmin"],
        ymin=value["ymin"],
        xmax=value["xmax"],
        ymax=value["ymax"],
    )


def arrow_batch_to_records(
    batch: pa.RecordBatch | pa.Table,
) -> Iterator[ImageRecord]:
    for row in batch.to_pylist():
        image_blob = row["image_blob"]
        yield ImageRecord(
            filename=str(row["filename"]),
            image_bytes=bytes(image_blob),
            width=int(row["width"]),
            height=int(row["height"]),
            captions=tuple(str(value) for value in (row["captions"] or [])),
            bboxes=tuple(_normalize_box(value) for value in (row["bboxes"] or [])),
        )


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
    selected = [record for record in materialized if record.filename in selected_names]
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
    from lakesoul.arrow.dataset import lakesoul_dataset

    dataset = lakesoul_dataset(table_name, batch_size=batch_size)
    columns = [
        "filename",
        "image_blob",
        "width",
        "height",
        "captions",
        "bboxes",
    ]
    for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
        yield from arrow_batch_to_records(batch)


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
    import lakesoul.ray  # noqa: F401
    import ray

    read_lakesoul = cast(
        Callable[..., Any], getattr(ray.data, "read_lakesoul")
    )
    dataset = read_lakesoul(
        config.table_name, batch_size=config.batch_size
    )
    for batch in dataset.iter_batches(
        batch_size=config.batch_size, batch_format="pyarrow"
    ):
        yield from arrow_batch_to_records(batch)


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


def build_daft_worker_command(config: BackendConfig) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("daft_worker.py")),
        "--batch-size",
        str(config.batch_size),
        "--seed",
        str(config.seed),
        "--limit",
        str(config.limit),
        "--split",
        config.split,
    ]
    if config.skip_corrupt:
        command.append("--skip-corrupt")
    if config.ray_address:
        command.extend(["--ray-address", config.ray_address])
    command.extend(["--runner", config.daft_runner, "--table", config.table_name])
    return command


def run_daft_worker(
    config: BackendConfig, output: Path, status: Path
) -> DaftWorkerSuccess:
    command = build_daft_worker_command(config)
    command.extend(["--output", str(output), "--status", str(status)])
    completed = subprocess.run(command, check=False, text=True)
    raw: Any = json.loads(status.read_text()) if status.is_file() else {}
    if completed.returncode:
        message = (
            raw.get("error_message", "Daft worker failed")
            if isinstance(raw, dict)
            else "Daft worker failed"
        )
        raise RuntimeError(str(message))
    if not isinstance(raw, dict) or raw.get("success") is not True:
        raise RuntimeError("Daft worker returned an invalid status payload")
    required_ints = ("input_rows", "output_rows", "input_bytes", "decode_errors")
    if any(not isinstance(raw.get(key), int) for key in required_ints):
        raise RuntimeError("Daft worker status contains invalid metrics")
    duration = raw.get("duration_seconds")
    digest = raw.get("digest")
    if not isinstance(duration, (int, float)) or not isinstance(digest, str):
        raise RuntimeError("Daft worker status contains invalid summary values")
    return {
        "success": True,
        "input_rows": raw["input_rows"],
        "output_rows": raw["output_rows"],
        "input_bytes": raw["input_bytes"],
        "decode_errors": raw["decode_errors"],
        "duration_seconds": float(duration),
        "digest": digest,
    }


def iter_daft_samples(config: BackendConfig) -> Iterator[Sample]:
    with tempfile.TemporaryDirectory(prefix="lakesoul-daft-") as directory:
        output = Path(directory) / "samples.arrow"
        status = Path(directory) / "status.json"
        run_daft_worker(config, output, status)
        with pa.memory_map(str(output), "r") as source:
            reader = pa.ipc.open_stream(source)
            for batch in reader:
                for row in batch.to_pylist():
                    yield Sample(
                        filename=row["filename"],
                        image_bytes=bytes(row["image_blob"]),
                        caption=row["caption"],
                    )


def iter_samples(config: BackendConfig) -> Iterator[Sample]:
    if config.backend == "daft":
        yield from iter_daft_samples(config)
    else:
        yield from iter_samples_from_records(iter_records(config), config)
