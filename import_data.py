from __future__ import annotations

import argparse
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TypedDict
from urllib.parse import urlparse

from multimodal_data import (
    BoundingBox,
    ImageRecord,
    canonical_record_digest_from_payloads,
    canonical_record_payload,
    select_filenames,
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
JAR_NAME = "lakesoul-spark-3.3-3.0.0-SNAPSHOT.jar"


class SourceValidation(TypedDict):
    row_count: int
    filenames: list[str]
    digest: str
    logical_blob_bytes: int


class TableValidation(SourceValidation):
    physical_bytes: int | None


class WrittenTableValidation(TableValidation):
    write_seconds: float


class ImportResult(TypedDict):
    source: SourceValidation
    parquet: WrittenTableValidation
    vortex: WrittenTableValidation


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
    return ("parquet", "vortex") if batch_index % 2 == 0 else (
        "vortex",
        "parquet",
    )


def ensure_targets_are_writable(*, existing: set[str], overwrite: bool) -> None:
    if existing and not overwrite:
        names = ", ".join(sorted(existing))
        raise RuntimeError(f"target tables already exist: {names}; use --overwrite")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import matching Parquet and Vortex LakeSoul tables"
    )
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


def get_int_text(parent, tag: str) -> int:
    if parent is None:
        raise ValueError(f"missing parent for {tag}")
    element = parent.find(tag)
    if element is None or element.text is None:
        raise ValueError(f"missing {tag}")
    return int(element.text)


def parse_bboxes(xml_path: Path) -> list[BoundingBox]:
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.findall("object"):
        bounds = obj.find("bndbox")
        if bounds is None:
            continue
        chain_ids = tuple(
            element.text
            for element in obj.findall("name")
            if element.text is not None
        )
        boxes.append(
            BoundingBox(
                chain_ids=chain_ids,
                xmin=get_int_text(bounds, "xmin"),
                ymin=get_int_text(bounds, "ymin"),
                xmax=get_int_text(bounds, "xmax"),
                ymax=get_int_text(bounds, "ymax"),
            )
        )
    return boxes


def parse_image_size(xml_path: Path) -> tuple[int, int]:
    size = ET.parse(xml_path).getroot().find("size")
    return get_int_text(size, "width"), get_int_text(size, "height")


def clean_caption(text: str) -> str:
    return re.sub(r"</?start>|</?end>", "", text).strip()


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


def lake_soul_schema():
    from pyspark.sql.types import (
        ArrayType,
        BinaryType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )

    bbox_schema = StructType(
        [
            StructField("chain_ids", ArrayType(StringType()), False),
            StructField("xmin", IntegerType(), False),
            StructField("ymin", IntegerType(), False),
            StructField("xmax", IntegerType(), False),
            StructField("ymax", IntegerType(), False),
        ]
    )
    return StructType(
        [
            StructField("filename", StringType(), False),
            StructField("image_blob", BinaryType(), False),
            StructField("width", IntegerType(), False),
            StructField("height", IntegerType(), False),
            StructField("captions", ArrayType(StringType()), False),
            StructField("bboxes", ArrayType(bbox_schema), True),
        ]
    )


def create_spark_session(jar_path: Path):
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[4]")
        .appName("lakesoul-multimodal-import")
        .config("spark.jars", str(jar_path))
        .config(
            "spark.sql.extensions",
            "com.dmetasoul.lakesoul.sql.LakeSoulSparkSessionExtension",
        )
        .config(
            "spark.sql.catalog.lakesoul",
            "org.apache.spark.sql.lakesoul.catalog.LakeSoulCatalog",
        )
        .config("spark.sql.defaultCatalog", "lakesoul")
        .config("spark.sql.warehouse.dir", str(PROJECT_DIR / "spark-warehouse"))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


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


def iter_lakesoul_records(
    table_name: str, *, batch_size: int = 1024
) -> Iterator[ImageRecord]:
    from data_backends import arrow_batch_to_records
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


def _physical_table_bytes(table_name: str) -> int | None:
    from lakesoul.arrow.dataset import lakesoul_dataset

    total = 0
    found = False
    for group in lakesoul_dataset(table_name).file_urls():
        for url in group:
            parsed = urlparse(url)
            path = Path(parsed.path if parsed.scheme == "file" else url)
            if parsed.scheme not in ("", "file") or not path.is_file():
                continue
            total += path.stat().st_size
            found = True
    return total if found else None


def validate_table(table_name: str) -> TableValidation:
    row_count = 0
    filenames = []
    logical_blob_bytes = 0
    payloads = []
    for record in iter_lakesoul_records(table_name):
        row_count += 1
        filenames.append(record.filename)
        logical_blob_bytes += len(record.image_bytes)
        payloads.append(
            (record.filename, canonical_record_payload(record))
        )
    return {
        "row_count": row_count,
        "filenames": sorted(filenames),
        "digest": canonical_record_digest_from_payloads(payloads),
        "logical_blob_bytes": logical_blob_bytes,
        "physical_bytes": _physical_table_bytes(table_name),
    }


def _with_write_seconds(
    validation: TableValidation, write_seconds: float
) -> WrittenTableValidation:
    return {
        "row_count": validation["row_count"],
        "filenames": validation["filenames"],
        "digest": validation["digest"],
        "logical_blob_bytes": validation["logical_blob_bytes"],
        "physical_bytes": validation["physical_bytes"],
        "write_seconds": write_seconds,
    }


def _batched(records: Iterable[ImageRecord], size: int) -> Iterator[list[ImageRecord]]:
    batch: list[ImageRecord] = []
    for record in records:
        batch.append(record)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def import_records(
    spark,
    records: Iterable[ImageRecord],
    *,
    parquet_table: str,
    vortex_table: str,
    batch_size: int,
) -> ImportResult:
    if parquet_table == vortex_table:
        raise ValueError("parquet and vortex table names must be different")
    schema = lake_soul_schema()
    source_filenames: list[str] = []
    source_payloads: list[tuple[str, bytes]] = []
    source_blob_bytes = 0
    write_seconds = {"parquet": 0.0, "vortex": 0.0}
    tables = {"parquet": parquet_table, "vortex": vortex_table}
    wrote_any = False

    for batch_index, batch in enumerate(_batched(records, batch_size)):
        source_filenames.extend(record.filename for record in batch)
        source_payloads.extend(
            (record.filename, canonical_record_payload(record))
            for record in batch
        )
        source_blob_bytes += sum(len(record.image_bytes) for record in batch)
        rows = [record_to_row(record) for record in batch]
        for file_format in alternating_formats(batch_index):
            write_seconds[file_format] += write_batch(
                spark,
                schema,
                rows,
                table_name=tables[file_format],
                file_format=file_format,
                first_batch=batch_index == 0,
            )
        wrote_any = True

    if not wrote_any:
        raise ValueError("no source records selected")

    source: SourceValidation = {
        "row_count": len(source_filenames),
        "filenames": sorted(source_filenames),
        "digest": canonical_record_digest_from_payloads(source_payloads),
        "logical_blob_bytes": source_blob_bytes,
    }
    parquet = _with_write_seconds(
        validate_table(parquet_table), write_seconds["parquet"]
    )
    vortex = _with_write_seconds(
        validate_table(vortex_table), write_seconds["vortex"]
    )
    if not (
        source["digest"] == parquet["digest"] == vortex["digest"]
        and source["filenames"] == parquet["filenames"] == vortex["filenames"]
    ):
        raise RuntimeError("source, Parquet, and Vortex table contents differ")
    return {"source": source, "parquet": parquet, "vortex": vortex}


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _config_from_args(args: argparse.Namespace) -> ImportConfig:
    return ImportConfig(
        data_dir=args.data_dir,
        jar_path=args.jar_path,
        parquet_table=args.parquet_table,
        vortex_table=args.vortex_table,
        limit=args.limit,
        batch_size=args.batch_size,
        seed=args.seed,
        overwrite=args.overwrite,
        output=args.output,
    )


def _validate_paths(config: ImportConfig) -> None:
    required = [
        config.jar_path,
        config.data_dir / "dataset_flickr30k_allEN.json",
        config.data_dir / "flickr30k-images",
        config.data_dir / "Annotations",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("required paths are absent: " + ", ".join(missing))


def main(argv: Sequence[str] | None = None) -> int:
    config = _config_from_args(build_parser().parse_args(argv))
    _validate_paths(config)
    spark = create_spark_session(config.jar_path)
    try:
        targets = {config.parquet_table, config.vortex_table}
        existing = {name for name in targets if spark.catalog.tableExists(name)}
        ensure_targets_are_writable(existing=existing, overwrite=config.overwrite)
        if config.overwrite:
            for table_name in targets:
                spark.sql(f"DROP TABLE IF EXISTS `{table_name}`")
        result = import_records(
            spark,
            load_source_records(config),
            parquet_table=config.parquet_table,
            vortex_table=config.vortex_table,
            batch_size=config.batch_size,
        )
    finally:
        spark.stop()

    payload = {
        "config": {
            **asdict(config),
            "data_dir": str(config.data_dir),
            "jar_path": str(config.jar_path),
            "output": str(config.output),
        },
        "tables": result,
    }
    _atomic_write_json(config.output, payload)
    print(
        f"Imported {result['source']['row_count']} images into "
        f"{config.parquet_table} and {config.vortex_table}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
