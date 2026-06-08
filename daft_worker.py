from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import daft
import pyarrow as pa

from data_backends import (
    DaftWorkerFailure,
    DaftWorkerStatus,
    DaftWorkerSuccess,
)
from multimodal_data import (
    canonical_sample_digest_from_keys,
    select_filenames,
    split_filenames,
)

OUTPUT_SCHEMA = pa.schema(
    [
        ("filename", pa.string()),
        ("image_blob", pa.binary()),
        ("caption", pa.string()),
        ("decoded_width", pa.int64()),
        ("decoded_height", pa.int64()),
    ]
)


def configure_runner(runner: str, ray_address: str | None) -> None:
    if runner == "native":
        daft.set_runner_native()
    elif runner == "ray":
        address = None if ray_address in (None, "local") else ray_address
        daft.set_runner_ray(address=address, noop_if_initialized=True)
    else:
        raise ValueError(f"unsupported Daft runner: {runner}")


def _transform_table(
    table: pa.Table, *, skip_corrupt: bool
) -> Iterator[pa.RecordBatch]:
    if table.num_rows == 0:
        return
    frame = daft.from_arrow(table)
    frame = frame.explode("captions").with_column_renamed("captions", "caption")
    decoded = frame["image_blob"].decode_image(
        on_error="null" if skip_corrupt else "raise",
        mode="RGB",
    )
    frame = frame.with_column("decoded_image", decoded)
    if skip_corrupt:
        frame = frame.where(frame["decoded_image"].not_null())
    frame = frame.with_columns(
        {
            "decoded_width": frame["decoded_image"].image_width(),
            "decoded_height": frame["decoded_image"].image_height(),
        }
    )
    selected = frame.select(
        "filename",
        "image_blob",
        "caption",
        "width",
        "height",
        "decoded_width",
        "decoded_height",
    )
    for batch in selected.to_arrow_iter():
        rows = []
        for row in batch.to_pylist():
            actual = (row["decoded_width"], row["decoded_height"])
            expected = (row["width"], row["height"])
            if actual != expected:
                raise ValueError(
                    f"decoded size mismatch for {row['filename']}: "
                    f"{actual} != {expected}"
                )
            rows.append(
                {
                    "filename": row["filename"],
                    "image_blob": row["image_blob"],
                    "caption": row["caption"],
                    "decoded_width": row["decoded_width"],
                    "decoded_height": row["decoded_height"],
                }
            )
        if rows:
            yield pa.RecordBatch.from_pylist(rows, schema=OUTPUT_SCHEMA)


def transform_arrow(
    data: pa.Table | Iterable[pa.RecordBatch], *, skip_corrupt: bool
) -> Iterator[pa.RecordBatch]:
    if isinstance(data, pa.Table):
        yield from _transform_table(data, skip_corrupt=skip_corrupt)
        return
    for batch in data:
        yield from _transform_table(
            pa.Table.from_batches([batch]), skip_corrupt=skip_corrupt
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", choices=["native", "ray"], required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="all")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ray-address", default="local")
    parser.add_argument("--skip-corrupt", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    return parser


def _selected_names(
    table_name: str, *, limit: int, seed: int, split: str, batch_size: int
) -> set[str]:
    from lakesoul.arrow.dataset import lakesoul_dataset

    names = []
    dataset = lakesoul_dataset(table_name, batch_size=batch_size)
    for batch in dataset.to_batches(columns=["filename"], batch_size=batch_size):
        names.extend(batch.column("filename").to_pylist())
    selected = select_filenames(names, limit=limit, seed=seed)
    if split == "all":
        return set(selected)
    return set(split_filenames(selected, seed=seed)[split])


def _filtered_batches(
    table_name: str, selected: set[str], batch_size: int
) -> Iterator[pa.RecordBatch]:
    from lakesoul.arrow.dataset import lakesoul_dataset

    columns = ["filename", "image_blob", "captions", "width", "height"]
    dataset = lakesoul_dataset(table_name, batch_size=batch_size)
    for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
        mask = pa.array(
            [name in selected for name in batch.column("filename").to_pylist()]
        )
        filtered = batch.filter(mask)
        if filtered.num_rows:
            yield filtered


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run(args: argparse.Namespace) -> DaftWorkerSuccess:
    configure_runner(args.runner, args.ray_address)
    selected = _selected_names(
        args.table,
        limit=args.limit,
        seed=args.seed,
        split=args.split,
        batch_size=args.batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    encoded_bytes = 0
    source_rows = 0
    sample_keys: list[tuple[str, str, str]] = []
    source_names: set[str] = set()
    output_names: set[str] = set()
    started = time.perf_counter_ns()
    with pa.OSFile(str(args.output), "wb") as sink:
        with pa.ipc.new_stream(sink, OUTPUT_SCHEMA) as writer:
            for source_batch in _filtered_batches(
                args.table, selected, args.batch_size
            ):
                source_rows += source_batch.num_rows
                source_names.update(
                    source_batch.column("filename").to_pylist()
                )
                encoded_bytes += sum(
                    len(value.as_py())
                    for value in source_batch.column("image_blob")
                    if value.is_valid
                )
                for output_batch in transform_arrow(
                    pa.Table.from_batches([source_batch]),
                    skip_corrupt=args.skip_corrupt,
                ):
                    writer.write_batch(output_batch)
                    rows += output_batch.num_rows
                    for row in output_batch.to_pylist():
                        output_names.add(row["filename"])
                        sample_keys.append(
                            (
                                row["filename"],
                                hashlib.sha256(row["image_blob"]).hexdigest(),
                                row["caption"],
                            )
                        )
    duration = (time.perf_counter_ns() - started) / 1_000_000_000
    return {
        "success": True,
        "input_rows": source_rows,
        "output_rows": rows,
        "input_bytes": encoded_bytes,
        "decode_errors": len(source_names - output_names),
        "duration_seconds": duration,
        "digest": canonical_sample_digest_from_keys(sample_keys),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload: DaftWorkerStatus
    try:
        payload = run(args)
    except Exception as exc:
        failure: DaftWorkerFailure = {
            "success": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:1000],
        }
        payload = failure
        _atomic_json(args.status, payload)
        return 1
    _atomic_json(args.status, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
