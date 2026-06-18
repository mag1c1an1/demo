from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence, TypedDict

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = Path(__file__).resolve().parent / "lerobot-pusht"
DEFAULT_OUTPUT = PROJECT_DIR / "benchmark-results" / "pusht-lakesoul-import.json"
TABLE_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
DEFAULT_TABLE_PREFIX = "lerobot_pusht_vortex"
DEFAULT_FILE_FORMAT = "vortex"
VIDEO_KEY = "observation.image"


class TableSummary(TypedDict):
    rows: int
    digest: str
    logical_blob_bytes: int
    write_seconds: float


class ImportSummary(TypedDict):
    frames: TableSummary
    episodes: TableSummary
    videos: TableSummary
    selected_episodes: int
    selected_frames: int
    source_video_bytes: int
    segment_video_bytes: int


@dataclass(frozen=True)
class TableNames:
    frames: str
    episodes: str
    videos: str

    def values(self) -> tuple[str, str, str]:
        return (self.frames, self.episodes, self.videos)


@dataclass(frozen=True)
class ImportConfig:
    source_dir: Path
    table_prefix: str
    batch_size: int
    episode_limit: int
    overwrite: bool
    output: Path
    ffmpeg_bin: str
    ffprobe_bin: str
    media_timeout: float
    ray_address: str | None
    concurrency: int
    file_format: str = DEFAULT_FILE_FORMAT

    def __post_init__(self) -> None:
        if not TABLE_NAME_PATTERN.fullmatch(self.table_prefix):
            raise ValueError(f"invalid LakeSoul table prefix: {self.table_prefix}")
        if self.batch_size <= 0:
            raise ValueError("batch size must be positive")
        if self.episode_limit < 0:
            raise ValueError("episode limit must be non-negative")
        if self.media_timeout <= 0:
            raise ValueError("media timeout must be positive")
        if self.concurrency <= 0:
            raise ValueError("concurrency must be positive")
        if self.file_format != "vortex":
            raise ValueError("only vortex output is supported")

    @property
    def tables(self) -> TableNames:
        return TableNames(
            frames=f"{self.table_prefix}_frames",
            episodes=f"{self.table_prefix}_episodes",
            videos=f"{self.table_prefix}_videos",
        )


@dataclass(frozen=True)
class SourceBundle:
    info: dict[str, Any]
    tasks: dict[int, str]
    episodes: list[dict[str, Any]]
    frames: list[dict[str, Any]]
    videos: list[Path]


SegmentExtractor = Callable[[Path, float, int, float, str, float], bytes]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import LeRobot PushT into LakeSoul/Vortex frames, episodes, videos tables"
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--table-prefix", default=DEFAULT_TABLE_PREFIX)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--episode-limit",
        type=int,
        default=0,
        help="Maximum sorted episodes to import; 0 imports all episodes",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--media-timeout", type=float, default=30.0)
    parser.add_argument("--ray-address", default="local")
    parser.add_argument("--concurrency", type=int, default=4)
    return parser


def config_from_args(args: argparse.Namespace) -> ImportConfig:
    return ImportConfig(
        source_dir=args.source_dir,
        table_prefix=args.table_prefix,
        batch_size=args.batch_size,
        episode_limit=args.episode_limit,
        overwrite=args.overwrite,
        output=args.output,
        ffmpeg_bin=args.ffmpeg_bin,
        ffprobe_bin=args.ffprobe_bin,
        media_timeout=args.media_timeout,
        ray_address=args.ray_address,
        concurrency=args.concurrency,
    )


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
    required = [
        config.source_dir,
        config.source_dir / "meta" / "info.json",
        config.source_dir / "meta" / "tasks.parquet",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("required paths are absent: " + ", ".join(missing))
    if not _find_parquet_files(config.source_dir / "data"):
        raise FileNotFoundError(
            f"no frame parquet files below {config.source_dir / 'data'}"
        )
    if not _find_parquet_files(config.source_dir / "meta" / "episodes"):
        raise FileNotFoundError(
            f"no episode parquet files below {config.source_dir / 'meta' / 'episodes'}"
        )
    if not _find_video_files(config.source_dir):
        raise FileNotFoundError(
            f"no MP4 video files below {config.source_dir / 'videos'}"
        )
    _validate_executable(config.ffmpeg_bin)
    _validate_executable(config.ffprobe_bin)


def _find_parquet_files(path: Path) -> list[Path]:
    return sorted(item for item in path.glob("chunk-*/*.parquet") if item.is_file())


def _find_video_files(source_dir: Path) -> list[Path]:
    return sorted(
        item for item in (source_dir / "videos").glob("*/*/*.mp4") if item.is_file()
    )


def _read_parquet_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def _load_tasks(path: Path) -> dict[int, str]:
    rows = pq.read_table(path).to_pylist()
    tasks: dict[int, str] = {}
    for row in rows:
        task_index = int(row["task_index"])
        task_text = None
        for name, value in row.items():
            if name != "task_index" and value is not None:
                task_text = str(value)
                break
        if task_text is None:
            raise ValueError(f"task {task_index} has no text")
        tasks[task_index] = task_text
    return tasks


def load_source_bundle(
    source_dir: Path | str, *, episode_limit: int = 0
) -> SourceBundle:
    source_dir = Path(source_dir)
    info = json.loads((source_dir / "meta" / "info.json").read_text())
    tasks = _load_tasks(source_dir / "meta" / "tasks.parquet")
    episodes = sorted(
        _read_parquet_rows(_find_parquet_files(source_dir / "meta" / "episodes")),
        key=lambda row: int(row["episode_index"]),
    )
    if episode_limit:
        episodes = episodes[:episode_limit]
    selected_episode_ids = {int(row["episode_index"]) for row in episodes}
    if not selected_episode_ids:
        raise ValueError("no episodes selected")

    frames = [
        row
        for row in _read_parquet_rows(_find_parquet_files(source_dir / "data"))
        if int(row["episode_index"]) in selected_episode_ids
    ]
    frames.sort(key=lambda row: int(row["index"]))
    if not frames:
        raise ValueError("no frames selected")

    return SourceBundle(
        info=info,
        tasks=tasks,
        episodes=episodes,
        frames=frames,
        videos=_find_video_files(source_dir),
    )


def frame_source_row_to_lakesoul(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "observation_state": [float(value) for value in row["observation.state"]],
        "action": [float(value) for value in row["action"]],
        "timestamp": float(row["timestamp"]),
        "frame_index": int(row["frame_index"]),
        "episode_index": int(row["episode_index"]),
        "index": int(row["index"]),
        "task_index": int(row["task_index"]),
        "next_reward": float(row["next.reward"]),
        "next_done": bool(row["next.done"]),
        "next_success": bool(row["next.success"]),
    }


def _parse_frame_rate(value: Any) -> float | None:
    if value in (None, "", "0/0"):
        return None
    text = str(value)
    numerator, separator, denominator = text.partition("/")
    if not separator:
        return float(numerator)
    denominator_value = float(denominator)
    if denominator_value == 0:
        return None
    return float(numerator) / denominator_value


def probe_video_path(path: Path, ffprobe_bin: str, timeout: float) -> dict[str, Any]:
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode:
        raise RuntimeError(_process_error(completed))
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise ValueError(f"ffprobe returned no video stream for {path}")
    stream = streams[0]
    duration = (payload.get("format") or {}).get("duration") or stream.get("duration")
    size = (payload.get("format") or {}).get("size")
    nb_frames = stream.get("nb_frames")
    return {
        "codec": str(stream["codec_name"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": _parse_frame_rate(stream.get("avg_frame_rate"))
        or _parse_frame_rate(stream.get("r_frame_rate")),
        "duration_sec": None if duration is None else float(duration),
        "num_frames": None if nb_frames in (None, "N/A") else int(nb_frames),
        "file_size_bytes": None if size is None else int(size),
    }


def _process_error(completed: subprocess.CompletedProcess[Any]) -> str:
    stderr = completed.stderr
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors="replace")
    message = str(stderr or "").strip()
    return message[-1000:] or f"process exited with status {completed.returncode}"


def _parse_numbered_name(name: str, prefix: str) -> int:
    match = re.fullmatch(rf"{re.escape(prefix)}-(\d+)", name)
    if match is None:
        raise ValueError(f"invalid {prefix} name: {name}")
    return int(match.group(1))


def build_video_rows(
    source_dir: Path,
    video_paths: Iterable[Path],
    *,
    ffprobe_bin: str,
    timeout: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(video_paths):
        relative = path.relative_to(source_dir)
        parts = relative.parts
        if len(parts) != 4 or parts[0] != "videos":
            raise ValueError(f"unexpected video path: {path}")
        video_blob = path.read_bytes()
        metadata = probe_video_path(path, ffprobe_bin, timeout)
        rows.append(
            {
                "camera_angle": parts[1],
                "chunk_index": _parse_numbered_name(parts[2], "chunk"),
                "file_index": _parse_numbered_name(path.stem, "file"),
                "relative_path": str(relative),
                "filename": path.name,
                "file_size_bytes": len(video_blob),
                "sha256": hashlib.sha256(video_blob).hexdigest(),
                "codec": metadata["codec"],
                "width": metadata["width"],
                "height": metadata["height"],
                "fps": metadata["fps"],
                "duration_sec": metadata["duration_sec"],
                "num_frames": metadata["num_frames"],
                "video_blob": video_blob,
            }
        )
    return rows


def extract_episode_segment(
    video_path: Path,
    from_timestamp: float,
    length: int,
    fps: float,
    ffmpeg_bin: str,
    timeout: float,
) -> bytes:
    command = [
        ffmpeg_bin,
        "-v",
        "error",
        "-nostdin",
        "-ss",
        f"{from_timestamp:.6f}",
        "-i",
        str(video_path),
        "-frames:v",
        str(length),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-r",
        f"{fps:g}",
        "-movflags",
        "frag_keyframe+empty_moov+default_base_moof",
        "-f",
        "mp4",
        "pipe:1",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        timeout=timeout,
    )
    if completed.returncode:
        raise RuntimeError(_process_error(completed))
    if not completed.stdout:
        raise RuntimeError("ffmpeg returned an empty episode segment")
    return completed.stdout


def _group_frame_rows(
    frame_rows: Iterable[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in frame_rows:
        grouped.setdefault(int(row["episode_index"]), []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["frame_index"]))
    return grouped


def _video_path_by_chunk_file(
    video_paths: Iterable[Path],
) -> dict[tuple[int, int], Path]:
    indexed: dict[tuple[int, int], Path] = {}
    for path in video_paths:
        relative = path.parts
        chunk_index = _parse_numbered_name(path.parent.name, "chunk")
        file_index = _parse_numbered_name(path.stem, "file")
        key = (chunk_index, file_index)
        if key in indexed:
            raise ValueError(f"duplicate video chunk/file key: {key}")
        if VIDEO_KEY not in relative:
            raise ValueError(f"unexpected video key in path: {path}")
        indexed[key] = path
    return indexed


def build_episode_rows(
    episode_meta_rows: Iterable[dict[str, Any]],
    frame_rows: Iterable[dict[str, Any]],
    tasks: dict[int, str],
    video_paths: Iterable[Path],
    *,
    fps: float,
    ffmpeg_bin: str,
    media_timeout: float,
    segment_extractor: SegmentExtractor = extract_episode_segment,
) -> list[dict[str, Any]]:
    grouped_frames = _group_frame_rows(frame_rows)
    videos = _video_path_by_chunk_file(video_paths)
    rows: list[dict[str, Any]] = []
    for meta in sorted(episode_meta_rows, key=lambda row: int(row["episode_index"])):
        episode_index = int(meta["episode_index"])
        frames = grouped_frames.get(episode_index, [])
        length = int(meta["length"])
        if len(frames) != length:
            raise ValueError(
                f"episode {episode_index} has {len(frames)} frames, expected {length}"
            )
        task_index = int(frames[0]["task_index"])
        from_timestamp = float(meta[f"videos/{VIDEO_KEY}/from_timestamp"])
        to_timestamp = float(meta[f"videos/{VIDEO_KEY}/to_timestamp"])
        video_key = (
            int(meta[f"videos/{VIDEO_KEY}/chunk_index"]),
            int(meta[f"videos/{VIDEO_KEY}/file_index"]),
        )
        if video_key not in videos:
            raise FileNotFoundError(f"missing video for chunk/file key: {video_key}")
        segment = segment_extractor(
            videos[video_key],
            from_timestamp,
            length,
            fps,
            ffmpeg_bin,
            media_timeout,
        )
        row_tasks = meta.get("tasks") or [tasks[task_index]]
        rows.append(
            {
                "episode_index": episode_index,
                "task_index": task_index,
                "fps": int(round(fps)),
                "length": length,
                "dataset_from_index": int(meta["dataset_from_index"]),
                "dataset_to_index": int(meta["dataset_to_index"]),
                "timestamps": [float(row["timestamp"]) for row in frames],
                "actions": [
                    [float(value) for value in row["action"]] for row in frames
                ],
                "observation_state": [
                    [float(value) for value in row["observation.state"]]
                    for row in frames
                ],
                "next_reward": [float(row["next.reward"]) for row in frames],
                "next_done": [bool(row["next.done"]) for row in frames],
                "next_success": [bool(row["next.success"]) for row in frames],
                "tasks": [str(value) for value in row_tasks],
                "observation_image_video_blob": segment,
                "observation_image_from_timestamp": from_timestamp,
                "observation_image_to_timestamp": to_timestamp,
                "observation_image_video_sha256": hashlib.sha256(segment).hexdigest(),
                "observation_image_video_bytes": len(segment),
                "observation_image_video_codec": "h264",
            }
        )
    return rows


def frames_schema():
    return pa.schema(
        [
            pa.field("observation_state", pa.list_(pa.float32()), nullable=False),
            pa.field("action", pa.list_(pa.float32()), nullable=False),
            pa.field("timestamp", pa.float32(), nullable=False),
            pa.field("frame_index", pa.int64(), nullable=False),
            pa.field("episode_index", pa.int64(), nullable=False),
            pa.field("index", pa.int64(), nullable=False),
            pa.field("task_index", pa.int64(), nullable=False),
            pa.field("next_reward", pa.float32(), nullable=False),
            pa.field("next_done", pa.bool_(), nullable=False),
            pa.field("next_success", pa.bool_(), nullable=False),
        ]
    )


def episodes_schema():
    vector = pa.list_(pa.float32())
    return pa.schema(
        [
            pa.field("episode_index", pa.int64(), nullable=False),
            pa.field("task_index", pa.int64(), nullable=False),
            pa.field("fps", pa.int32(), nullable=False),
            pa.field("length", pa.int64(), nullable=False),
            pa.field("dataset_from_index", pa.int64(), nullable=False),
            pa.field("dataset_to_index", pa.int64(), nullable=False),
            pa.field("timestamps", pa.list_(pa.float32()), nullable=False),
            pa.field("actions", pa.list_(vector), nullable=False),
            pa.field("observation_state", pa.list_(vector), nullable=False),
            pa.field("next_reward", pa.list_(pa.float32()), nullable=False),
            pa.field("next_done", pa.list_(pa.bool_()), nullable=False),
            pa.field("next_success", pa.list_(pa.bool_()), nullable=False),
            pa.field("tasks", pa.list_(pa.string()), nullable=False),
            pa.field("observation_image_video_blob", pa.binary(), nullable=False),
            pa.field("observation_image_from_timestamp", pa.float64(), nullable=False),
            pa.field("observation_image_to_timestamp", pa.float64(), nullable=False),
            pa.field("observation_image_video_sha256", pa.string(), nullable=False),
            pa.field("observation_image_video_bytes", pa.int64(), nullable=False),
            pa.field("observation_image_video_codec", pa.string(), nullable=False),
        ]
    )


def videos_schema():
    return pa.schema(
        [
            pa.field("camera_angle", pa.string(), nullable=False),
            pa.field("chunk_index", pa.int32(), nullable=False),
            pa.field("file_index", pa.int32(), nullable=False),
            pa.field("relative_path", pa.string(), nullable=False),
            pa.field("filename", pa.string(), nullable=False),
            pa.field("file_size_bytes", pa.int64(), nullable=False),
            pa.field("sha256", pa.string(), nullable=False),
            pa.field("codec", pa.string(), nullable=False),
            pa.field("width", pa.int32(), nullable=False),
            pa.field("height", pa.int32(), nullable=False),
            pa.field("fps", pa.float64(), nullable=True),
            pa.field("duration_sec", pa.float64(), nullable=True),
            pa.field("num_frames", pa.int64(), nullable=True),
            pa.field("video_blob", pa.binary(), nullable=False),
        ]
    )


def _batched(
    rows: Iterable[dict[str, Any]], size: int
) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _normalize_ray_batch(batch, schema):
    return batch.cast(schema)


def _table_path(table_name: str) -> str:
    return str(PROJECT_DIR / "spark-warehouse" / "default" / table_name)


def write_lakesoul_table(
    table_name: str,
    rows: list[dict[str, Any]],
    schema: pa.Schema,
    *,
    file_format: str,
    batch_size: int,
    concurrency: int,
) -> float:
    import ray
    from lakesoul.metadata import create_table
    from lakesoul.metadata.meta_ops import get_arrow_schema_by_table_name
    from lakesoul.ray import LakeSoulDatasink

    if not rows:
        raise ValueError(f"no rows to write for {table_name}")
    create_table(
        table_name,
        table_schema=schema,
        table_path=_table_path(table_name),
        properties={"file_format": file_format},
    )
    target_schema = get_arrow_schema_by_table_name(table_name)
    table = pa.Table.from_pylist(rows, schema=schema)
    dataset = (
        ray.data.from_arrow(table)
        .map_batches(
            _normalize_ray_batch,  # type: ignore
            batch_size=batch_size,
            batch_format="pyarrow",
            fn_kwargs={"schema": target_schema},
        )
        .materialize()
    )
    if dataset.count() != len(rows):
        raise RuntimeError(f"Ray dataset row count changed for {table_name}")
    sink = LakeSoulDatasink(
        table_name,
        format=file_format,  # type: ignore[arg-type]
        batch_size=batch_size,
        thread_num=concurrency,
    )
    started = time.perf_counter_ns()
    dataset.write_datasink(
        sink,
        ray_remote_args={"max_retries": 0},
        concurrency=concurrency,
    )
    return (time.perf_counter_ns() - started) / 1_000_000_000


def _json_safe(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def canonical_row_payload(row: dict[str, Any]) -> bytes:
    return json.dumps(
        _json_safe(row),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def digest_rows(rows: Iterable[dict[str, Any]], key_column: str) -> str:
    payloads = [(str(row[key_column]), canonical_row_payload(row)) for row in rows]
    digest = hashlib.sha256()
    for _, payload in sorted(payloads):
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def logical_blob_bytes(rows: Iterable[dict[str, Any]], blob_column: str | None) -> int:
    if blob_column is None:
        return 0
    return sum(len(bytes(row[blob_column])) for row in rows)


def read_lakesoul_rows(
    table_name: str,
    columns: list[str],
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    from lakesoul.arrow.dataset import lakesoul_dataset

    rows: list[dict[str, Any]] = []
    dataset = lakesoul_dataset(table_name, batch_size=batch_size)
    for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
        rows.extend(batch.to_pylist())
    return rows


def _summarize_expected(
    rows: list[dict[str, Any]],
    *,
    key_column: str,
    blob_column: str | None,
    write_seconds: float,
) -> TableSummary:
    return {
        "rows": len(rows),
        "digest": digest_rows(rows, key_column),
        "logical_blob_bytes": logical_blob_bytes(rows, blob_column),
        "write_seconds": write_seconds,
    }


def validate_lakesoul_table(
    table_name: str,
    expected: TableSummary,
    columns: list[str],
    *,
    key_column: str,
    blob_column: str | None,
    batch_size: int,
) -> None:
    rows = read_lakesoul_rows(table_name, columns, batch_size=batch_size)
    actual = _summarize_expected(
        rows,
        key_column=key_column,
        blob_column=blob_column,
        write_seconds=expected["write_seconds"],
    )
    if (
        actual["rows"] != expected["rows"]
        or actual["digest"] != expected["digest"]
        or actual["logical_blob_bytes"] != expected["logical_blob_bytes"]
    ):
        raise RuntimeError(
            f"source and LakeSoul table contents differ for {table_name}"
        )


def probe_video_bytes(
    video_blob: bytes,
    *,
    ffprobe_bin: str,
    timeout: float,
) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(suffix=".mp4") as handle:
        handle.write(video_blob)
        handle.flush()
        return probe_video_path(Path(handle.name), ffprobe_bin, timeout)


def validate_episode_segments(
    episode_rows: Iterable[dict[str, Any]],
    *,
    ffprobe_bin: str,
    timeout: float,
) -> None:
    for row in episode_rows:
        metadata = probe_video_bytes(
            bytes(row["observation_image_video_blob"]),
            ffprobe_bin=ffprobe_bin,
            timeout=timeout,
        )
        expected_length = int(row["length"])
        actual_frames = metadata["num_frames"]
        if actual_frames is not None and actual_frames != expected_length:
            raise RuntimeError(
                f"episode {row['episode_index']} segment has {actual_frames} frames, "
                f"expected {expected_length}"
            )
        if (
            metadata["fps"] is not None
            and abs(float(metadata["fps"]) - row["fps"]) > 0.01
        ):
            raise RuntimeError(
                f"episode {row['episode_index']} segment fps differs: {metadata['fps']}"
            )
        duration = metadata["duration_sec"]
        expected_duration = expected_length / float(row["fps"])
        if duration is not None and abs(float(duration) - expected_duration) > 0.15:
            raise RuntimeError(
                f"episode {row['episode_index']} segment duration differs: {duration}"
            )


def _table_columns(schema) -> list[str]:
    return list(schema.names)


def _table_exists(table_name: str) -> bool:
    from lakesoul.metadata.meta_ops import get_table_info_by_name

    try:
        get_table_info_by_name(table_name, "default")
    except RuntimeError as error:
        if "not found" in str(error).lower():
            return False
        raise
    return True


def ensure_targets_are_absent(tables: TableNames, *, overwrite: bool) -> None:
    existing = {name for name in tables.values() if _table_exists(name)}
    if existing:
        names = ", ".join(sorted(existing))
        suffix = (
            "; pure Python LakeSoul writer cannot drop tables, "
            "use a new --table-prefix or clean them externally"
            if overwrite
            else "; use a new --table-prefix or clean them externally"
        )
        raise RuntimeError(f"target tables already exist: {names}{suffix}")


def configure_ray(ray_address: str | None) -> None:
    import ray

    address = None if ray_address in (None, "local") else ray_address
    if not ray.is_initialized():
        ray.init(address=address, include_dashboard=False)


def import_lerobot_pusht(config: ImportConfig) -> ImportSummary:
    bundle = load_source_bundle(config.source_dir, episode_limit=config.episode_limit)
    fps = float(bundle.info["fps"])
    frame_rows = [frame_source_row_to_lakesoul(row) for row in bundle.frames]
    video_rows = build_video_rows(
        config.source_dir,
        bundle.videos,
        ffprobe_bin=config.ffprobe_bin,
        timeout=config.media_timeout,
    )
    episode_rows = build_episode_rows(
        bundle.episodes,
        bundle.frames,
        bundle.tasks,
        bundle.videos,
        fps=fps,
        ffmpeg_bin=config.ffmpeg_bin,
        media_timeout=config.media_timeout,
    )
    validate_episode_segments(
        episode_rows,
        ffprobe_bin=config.ffprobe_bin,
        timeout=config.media_timeout,
    )

    tables = config.tables
    frame_schema = frames_schema()
    episode_schema = episodes_schema()
    video_schema = videos_schema()
    ensure_targets_are_absent(tables, overwrite=config.overwrite)
    configure_ray(config.ray_address)
    frame_write_seconds = write_lakesoul_table(
        tables.frames,
        frame_rows,
        frame_schema,
        file_format=config.file_format,
        batch_size=config.batch_size,
        concurrency=config.concurrency,
    )
    episode_write_seconds = write_lakesoul_table(
        tables.episodes,
        episode_rows,
        episode_schema,
        file_format=config.file_format,
        batch_size=config.batch_size,
        concurrency=config.concurrency,
    )
    video_write_seconds = write_lakesoul_table(
        tables.videos,
        video_rows,
        video_schema,
        file_format=config.file_format,
        batch_size=config.batch_size,
        concurrency=config.concurrency,
    )

    frame_summary = _summarize_expected(
        frame_rows,
        key_column="index",
        blob_column=None,
        write_seconds=frame_write_seconds,
    )
    episode_summary = _summarize_expected(
        episode_rows,
        key_column="episode_index",
        blob_column="observation_image_video_blob",
        write_seconds=episode_write_seconds,
    )
    video_summary = _summarize_expected(
        video_rows,
        key_column="relative_path",
        blob_column="video_blob",
        write_seconds=video_write_seconds,
    )

    validate_lakesoul_table(
        tables.frames,
        frame_summary,
        _table_columns(frame_schema),
        key_column="index",
        blob_column=None,
        batch_size=config.batch_size,
    )
    validate_lakesoul_table(
        tables.episodes,
        episode_summary,
        _table_columns(episode_schema),
        key_column="episode_index",
        blob_column="observation_image_video_blob",
        batch_size=config.batch_size,
    )
    validate_lakesoul_table(
        tables.videos,
        video_summary,
        _table_columns(video_schema),
        key_column="relative_path",
        blob_column="video_blob",
        batch_size=config.batch_size,
    )

    return {
        "frames": frame_summary,
        "episodes": episode_summary,
        "videos": video_summary,
        "selected_episodes": len(bundle.episodes),
        "selected_frames": len(bundle.frames),
        "source_video_bytes": video_summary["logical_blob_bytes"],
        "segment_video_bytes": episode_summary["logical_blob_bytes"],
    }


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _config_payload(config: ImportConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    payload["tables"] = asdict(config.tables)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    config = config_from_args(build_parser().parse_args(argv))
    validate_environment(config)
    summary = import_lerobot_pusht(config)
    _atomic_write_json(
        config.output,
        {"config": _config_payload(config), "summary": summary},
    )
    print(
        f"Imported {summary['selected_frames']} frames and "
        f"{summary['selected_episodes']} episodes into "
        f"{config.tables.frames}, {config.tables.episodes}, {config.tables.videos}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
