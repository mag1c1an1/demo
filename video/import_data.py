from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence, TypedDict

from lakesoul.metadata import create_table
from lakesoul.metadata.meta_ops import get_arrow_schema_by_table_name
from lakesoul.ray import LakeSoulDatasink

VIDEO_DIR = Path(__file__).resolve().parent
PROJECT_DIR = VIDEO_DIR.parent
DEFAULT_SOURCE_DIR = VIDEO_DIR / "data" / "UCF101_subset"
DEFAULT_THUMBNAIL_DIR = VIDEO_DIR / "data" / "UCF101_thumbnails"
JAR_NAME = "lakesoul-spark-3.3-3.0.0-SNAPSHOT.jar"

DAFT_COLUMNS = (
    "video_id",
    "split",
    "label",
    "video_path",
    "codec",
    "width",
    "height",
    "fps",
    "duration_sec",
    "thumbnail_path",
    "thumbnail_blob",
    "probe_error",
    "thumbnail_error",
)

TABLE_COLUMNS = (
    "video_id",
    "split",
    "label",
    "video_path",
    "codec",
    "width",
    "height",
    "fps",
    "duration_sec",
    "thumbnail_path",
    "thumbnail_blob",
)


class ImportSummary(TypedDict):
    discovered_rows: int
    imported_rows: int
    skipped_rows: int
    thumbnail_bytes: int
    digest: str
    write_seconds: float


@dataclass(frozen=True)
class ImportConfig:
    source_dir: Path
    thumbnail_dir: Path
    jar_path: Path
    table: str
    file_format: str
    limit: int
    batch_size: int
    overwrite: bool
    thumbnail_second: float
    media_timeout: float
    daft_runner: str
    ray_address: str | None
    concurrency: int
    ffprobe_bin: str
    ffmpeg_bin: str
    skip_corrupt: bool
    output: Path

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.table):
            raise ValueError(f"invalid LakeSoul table name: {self.table}")
        if self.file_format not in {"parquet", "vortex"}:
            raise ValueError(f"unsupported file format: {self.file_format}")
        if self.limit < 0:
            raise ValueError("limit must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch size must be positive")
        if self.thumbnail_second < 0:
            raise ValueError("thumbnail second must be non-negative")
        if self.media_timeout <= 0:
            raise ValueError("media timeout must be positive")
        if self.daft_runner not in {"native", "ray"}:
            raise ValueError(f"unsupported Daft runner: {self.daft_runner}")
        if self.concurrency <= 0:
            raise ValueError("concurrency must be positive")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Probe UCF101 videos and import metadata plus thumbnails into LakeSoul"
        )
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--thumbnail-dir", type=Path, default=DEFAULT_THUMBNAIL_DIR)
    parser.add_argument("--jar-path", type=Path, default=PROJECT_DIR / JAR_NAME)
    parser.add_argument("--table", default="ucf101_video")
    parser.add_argument(
        "--file-format", choices=["parquet", "vortex"], default="parquet"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum sorted videos to import; 0 imports all videos",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--thumbnail-second", type=float, default=1.0)
    parser.add_argument("--media-timeout", type=float, default=30.0)
    parser.add_argument("--daft-runner", choices=["native", "ray"], default="native")
    parser.add_argument("--ray-address", default="local")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument(
        "--skip-corrupt",
        action="store_true",
        help="Skip videos that cannot be probed or decoded",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "benchmark-results" / "video-import.json",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ImportConfig:
    return ImportConfig(
        source_dir=args.source_dir,
        thumbnail_dir=args.thumbnail_dir,
        jar_path=args.jar_path,
        table=args.table,
        file_format=args.file_format,
        limit=args.limit,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        thumbnail_second=args.thumbnail_second,
        media_timeout=args.media_timeout,
        daft_runner=args.daft_runner,
        ray_address=args.ray_address,
        concurrency=args.concurrency,
        ffprobe_bin=args.ffprobe_bin,
        ffmpeg_bin=args.ffmpeg_bin,
        skip_corrupt=args.skip_corrupt,
        output=args.output,
    )


def discover_videos(source_dir: Path, *, limit: int = 0) -> list[dict[str, str]]:
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"video source directory is absent: {source_dir}")

    paths = sorted(path for path in source_dir.glob("*/*/*.avi") if path.is_file())
    if limit:
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"no AVI videos found below {source_dir}")

    rows: list[dict[str, str]] = []
    video_ids: set[str] = set()
    for path in paths:
        split, label, filename = path.relative_to(source_dir).parts
        video_id = Path(filename).stem
        if video_id in video_ids:
            raise ValueError(f"duplicate UCF101 video id: {video_id}")
        video_ids.add(video_id)
        rows.append(
            {
                "video_id": video_id,
                "split": split,
                "label": label,
                "video_path": str(path.resolve()),
            }
        )
    return rows


def _process_error(completed: subprocess.CompletedProcess[Any]) -> str:
    stderr = completed.stderr
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors="replace")
    message = str(stderr or "").strip()
    return message[-1000:] or f"process exited with status {completed.returncode}"


def _parse_frame_rate(value: object) -> float | None:
    if value in (None, "", "0/0"):
        return None
    numerator, separator, denominator = str(value).partition("/")
    if not separator:
        return float(numerator)
    denominator_value = float(denominator)
    if denominator_value == 0:
        return None
    return float(numerator) / denominator_value


def probe_video(path: str, ffprobe_bin: str, timeout: float) -> dict[str, object]:
    """Return codec, dimensions, frame rate, and duration from ffprobe."""
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,codec_name",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        path,
    ]
    empty: dict[str, object] = {
        "codec": None,
        "width": None,
        "height": None,
        "fps": None,
        "duration_sec": None,
    }
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode:
            return {**empty, "error": _process_error(completed)}
        payload = json.loads(completed.stdout)
        streams = payload.get("streams") or []
        if not streams:
            raise ValueError("ffprobe returned no video stream")
        stream = streams[0]
        codec = stream.get("codec_name")
        if not codec:
            raise ValueError("ffprobe returned no video codec")
        duration = (payload.get("format") or {}).get("duration")
        return {
            "codec": codec,
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "fps": _parse_frame_rate(stream.get("r_frame_rate")),
            "duration_sec": None if duration is None else float(duration),
            "error": None,
        }
    except Exception as exc:
        return {**empty, "error": f"{type(exc).__name__}: {exc}"[:1000]}


def _thumbnail_command(path: str, second: float, ffmpeg_bin: str) -> list[str]:
    return [
        ffmpeg_bin,
        "-v",
        "error",
        "-ss",
        str(second),
        "-i",
        path,
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
    ]


def extract_thumbnail(
    path: str,
    second: float,
    ffmpeg_bin: str,
    timeout: float,
) -> dict[str, object]:
    attempts = [second]
    if second > 0:
        attempts.append(0.0)
    last_error = "ffmpeg returned no image"
    for seek_second in attempts:
        try:
            completed = subprocess.run(
                _thumbnail_command(path, seek_second, ffmpeg_bin),
                check=False,
                capture_output=True,
                timeout=timeout,
            )
            if completed.returncode:
                last_error = _process_error(completed)
                continue
            if completed.stdout:
                return {"thumbnail_blob": completed.stdout, "error": None}
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"[:1000]
    return {"thumbnail_blob": None, "error": last_error}


def process_video(
    path: str,
    video_id: str,
    split: str,
    label: str,
    ffprobe_bin: str,
    ffmpeg_bin: str,
    thumbnail_second: float,
    timeout: float,
    thumbnail_dir: str,
    skip_corrupt: bool,
) -> dict[str, object]:
    metadata = probe_video(path, ffprobe_bin, timeout)
    thumbnail = extract_thumbnail(path, thumbnail_second, ffmpeg_bin, timeout)
    errors = [str(error) for error in (metadata["error"], thumbnail["error"]) if error]
    if errors and not skip_corrupt:
        raise RuntimeError(f"media processing failed for {path}: " + "; ".join(errors))

    result = {
        "codec": metadata["codec"],
        "width": metadata["width"],
        "height": metadata["height"],
        "fps": metadata["fps"],
        "duration_sec": metadata["duration_sec"],
        "thumbnail_path": None,
        "thumbnail_blob": thumbnail["thumbnail_blob"],
        "probe_error": metadata["error"],
        "thumbnail_error": thumbnail["error"],
    }
    if not errors:
        result["thumbnail_path"] = _write_thumbnail(
            Path(thumbnail_dir),
            {
                "video_id": video_id,
                "split": split,
                "label": label,
                "thumbnail_blob": thumbnail["thumbnail_blob"],
            },
        )
    return result


def build_daft_frame(config: ImportConfig):
    import daft
    from daft import DataType, col

    source_rows = discover_videos(config.source_dir, limit=config.limit)
    df = daft.from_pydict(
        {
            name: [row[name] for row in source_rows]
            for name in ("video_id", "split", "label", "video_path")
        }
    )
    result_type = DataType.struct(
        {
            "codec": DataType.string(),
            "width": DataType.int64(),
            "height": DataType.int64(),
            "fps": DataType.float64(),
            "duration_sec": DataType.float64(),
            "thumbnail_path": DataType.string(),
            "thumbnail_blob": DataType.binary(),
            "probe_error": DataType.string(),
            "thumbnail_error": DataType.string(),
        }
    )

    @daft.cls(max_concurrency=config.concurrency)
    class VideoProcessor:
        def __init__(
            self,
            ffprobe_bin: str,
            ffmpeg_bin: str,
            thumbnail_second: float,
            timeout: float,
            thumbnail_dir: str,
            skip_corrupt: bool,
        ) -> None:
            self.ffprobe_bin = ffprobe_bin
            self.ffmpeg_bin = ffmpeg_bin
            self.thumbnail_second = thumbnail_second
            self.timeout = timeout
            self.thumbnail_dir = thumbnail_dir
            self.skip_corrupt = skip_corrupt

        @daft.method(return_dtype=result_type)
        def process(
            self,
            path: str,
            video_id: str,
            split: str,
            label: str,
        ) -> dict[str, object]:
            return process_video(
                path,
                video_id,
                split,
                label,
                self.ffprobe_bin,
                self.ffmpeg_bin,
                self.thumbnail_second,
                self.timeout,
                self.thumbnail_dir,
                self.skip_corrupt,
            )

    processor = VideoProcessor(
        config.ffprobe_bin,
        config.ffmpeg_bin,
        config.thumbnail_second,
        config.media_timeout,
        str(config.thumbnail_dir.resolve()),
        config.skip_corrupt,
    )
    df = df.with_column(
        "processed",
        processor.process(
            col("video_path"),
            col("video_id"),
            col("split"),
            col("label"),
        ),
    )
    df = (
        df.with_column("codec", col("processed").get("codec"))
        .with_column("width", col("processed").get("width"))
        .with_column("height", col("processed").get("height"))
        .with_column("fps", col("processed").get("fps"))
        .with_column("duration_sec", col("processed").get("duration_sec"))
        .with_column("thumbnail_path", col("processed").get("thumbnail_path"))
        .with_column("thumbnail_blob", col("processed").get("thumbnail_blob"))
        .with_column("probe_error", col("processed").get("probe_error"))
        .with_column("thumbnail_error", col("processed").get("thumbnail_error"))
        .select(*DAFT_COLUMNS)
    )
    if config.skip_corrupt:
        df = df.where(col("probe_error").is_null() & col("thumbnail_error").is_null())
    return df.select(*TABLE_COLUMNS)


def create_spark_session(jar_path: Path):
    from pyspark.sql import SparkSession

    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    spark = (
        SparkSession.builder.master("local[4]")
        .appName("lakesoul-video-import")
        .config("spark.jars", str(jar_path))
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
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


def lake_soul_schema():
    from pyspark.sql.types import (
        BinaryType,
        DoubleType,
        LongType,
        StringType,
        StructField,
        StructType,
    )

    return StructType(
        [
            StructField("video_id", StringType(), False),
            StructField("split", StringType(), False),
            StructField("label", StringType(), False),
            StructField("video_path", StringType(), False),
            StructField("codec", StringType(), False),
            StructField("width", LongType(), False),
            StructField("height", LongType(), False),
            StructField("fps", DoubleType(), True),
            StructField("duration_sec", DoubleType(), True),
            StructField("thumbnail_path", StringType(), False),
            StructField("thumbnail_blob", BinaryType(), False),
        ]
    )


def _write_thumbnail(thumbnail_dir: Path, row: dict[str, object]) -> str:
    output = (
        thumbnail_dir / str(row["split"]) / str(row["label"]) / f"{row['video_id']}.jpg"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp-{os.getpid()}")
    temporary.write_bytes(bytes(row["thumbnail_blob"]))  # type: ignore
    temporary.replace(output)
    return str(output.resolve())


def _row_payload(row: dict[str, object]) -> bytes:
    thumbnail = bytes(row["thumbnail_blob"])  # type: ignore
    payload = {name: row[name] for name in TABLE_COLUMNS if name != "thumbnail_blob"}
    payload["thumbnail_sha256"] = hashlib.sha256(thumbnail).hexdigest()
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _digest_payloads(payloads: list[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for _, payload in sorted(payloads):
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _normalize_ray_batch(batch, schema):
    return batch.cast(schema)


def _validate_table(table: str, *, batch_size: int) -> tuple[int, int, str]:
    from lakesoul.arrow.dataset import lakesoul_dataset

    row_count = 0
    thumbnail_bytes = 0
    payloads: list[tuple[str, bytes]] = []
    dataset = lakesoul_dataset(table, batch_size=batch_size)
    for batch in dataset.to_batches(columns=list(TABLE_COLUMNS), batch_size=batch_size):
        for row in batch.to_pylist():
            row_count += 1
            thumbnail_bytes += len(row["thumbnail_blob"])
            payloads.append((str(row["video_id"]), _row_payload(row)))
    return row_count, thumbnail_bytes, _digest_payloads(payloads)


def import_videos(config: ImportConfig) -> ImportSummary:
    df = build_daft_frame(config)
    # TODO(jiax): create/overrite table in write
    schema = df.schema().to_pyarrow_schema()
    create_table(
        config.table,
        table_schema=schema,
        table_path=str(PROJECT_DIR / "spark-warehouse" / "default" / config.table),
        properties={"file_format": config.file_format},
    )
    target_schema = get_arrow_schema_by_table_name(config.table)
    discovered_rows = len(discover_videos(config.source_dir, limit=config.limit))
    thumbnail_bytes = 0
    payloads: list[tuple[str, bytes]] = []

    ds = (
        df.to_ray_dataset()
        .map_batches(
            _normalize_ray_batch,
            batch_size=config.batch_size,
            batch_format="pyarrow",
            fn_kwargs={"schema": target_schema},
        )
        .materialize()
    )
    imported_rows = ds.count()
    if not imported_rows:
        raise ValueError("no videos were successfully processed")

    for batch in ds.iter_batches(
        batch_size=config.batch_size,
        batch_format="pyarrow",
    ):
        for row in batch.to_pylist():
            thumbnail_bytes += len(row["thumbnail_blob"])
            payloads.append((str(row["video_id"]), _row_payload(row)))

    digest = _digest_payloads(payloads)
    sink = LakeSoulDatasink(
        config.table,
        format=config.file_format,  # type: ignore
        batch_size=config.batch_size,
        thread_num=config.concurrency,
    )
    started = time.perf_counter_ns()
    ds.write_datasink(
        sink,
        ray_remote_args={"max_retries": 0},
        concurrency=config.concurrency,
    )
    write_seconds = (time.perf_counter_ns() - started) / 1_000_000_000

    table_rows, table_thumbnail_bytes, table_digest = _validate_table(
        config.table, batch_size=config.batch_size
    )
    if (
        table_rows != imported_rows
        or table_thumbnail_bytes != thumbnail_bytes
        or table_digest != digest
    ):
        raise RuntimeError("source and LakeSoul table contents differ")

    return {
        "discovered_rows": discovered_rows,
        "imported_rows": imported_rows,
        "skipped_rows": discovered_rows - imported_rows,
        "thumbnail_bytes": thumbnail_bytes,
        "digest": digest,
        "write_seconds": write_seconds,
    }


def _validate_executable(name: str) -> None:
    path = Path(name)
    if path.parent != Path("."):
        if not path.is_file() or not os.access(path, os.X_OK):
            raise FileNotFoundError(
                f"media executable is absent or not executable: {name}"
            )
        return
    if shutil.which(name) is None:
        raise FileNotFoundError(f"media executable is not on PATH: {name}")


def validate_environment(config: ImportConfig) -> None:
    missing = [
        str(path) for path in (config.source_dir, config.jar_path) if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("required paths are absent: " + ", ".join(missing))
    _validate_executable(config.ffprobe_bin)
    _validate_executable(config.ffmpeg_bin)


def _table_exists(table: str) -> bool:
    from lakesoul.metadata.meta_ops import get_table_info_by_name

    try:
        get_table_info_by_name(table, "default")
    except RuntimeError as error:
        if "not found" in str(error):
            return False
        raise
    return True


def prepare_lakesoul_table(config: ImportConfig) -> None:
    spark = create_spark_session(config.jar_path)
    try:
        if config.overwrite:
            spark.sql(f"DROP TABLE IF EXISTS `{config.table}`")
        elif _table_exists(config.table):
            raise RuntimeError(
                f"target table already exists: {config.table}; use --overwrite"
            )
        pass
        # (
        #     spark.createDataFrame([], lake_soul_schema())
        #     .write.format("lakesoul")
        #     .option("file_format", config.file_format)
        #     .saveAsTable(config.table)
        # )
    finally:
        spark.stop()


def configure_daft(config: ImportConfig) -> None:
    import daft
    import ray

    address = None if config.ray_address in (None, "local") else config.ray_address
    if not ray.is_initialized():
        ray.init(address=address, include_dashboard=False)
    if config.daft_runner == "native":
        daft.set_runner_native()
        return
    daft.set_runner_ray(address=address, noop_if_initialized=True)


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    config = config_from_args(build_parser().parse_args(argv))
    validate_environment(config)
    prepare_lakesoul_table(config)
    configure_daft(config)
    summary = import_videos(config)

    config_payload = asdict(config)
    for name, value in list(config_payload.items()):
        if isinstance(value, Path):
            config_payload[name] = str(value)
    _atomic_write_json(
        config.output,
        {"config": config_payload, "summary": summary},
    )
    print(
        f"Imported {summary['imported_rows']} videos into {config.table} "
        f"as {config.file_format} ({summary['skipped_rows']} skipped)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
