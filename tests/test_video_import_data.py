import json
import subprocess
from pathlib import Path

import pyarrow as pa
import pytest

import video.import_data as video_import
from video.import_data import (
    ImportConfig,
    TABLE_COLUMNS,
    _normalize_ray_batch,
    _write_thumbnail,
    build_daft_frame,
    build_parser,
    discover_videos,
    extract_thumbnail,
    import_videos,
    probe_video,
)


def make_config(tmp_path: Path) -> ImportConfig:
    return ImportConfig(
        source_dir=tmp_path,
        thumbnail_dir=tmp_path / "thumbnails",
        jar_path=tmp_path / "lake.jar",
        table="ucf101_video",
        file_format="parquet",
        limit=0,
        batch_size=8,
        overwrite=False,
        thumbnail_second=1.0,
        media_timeout=10.0,
        daft_runner="native",
        ray_address="local",
        concurrency=2,
        ffprobe_bin="ffprobe",
        ffmpeg_bin="ffmpeg",
        skip_corrupt=False,
        output=tmp_path / "result.json",
    )


def test_cli_defaults_to_ucf101_parquet_table():
    args = build_parser().parse_args([])

    assert args.table == "ucf101_video"
    assert args.file_format == "parquet"
    assert args.limit == 0
    assert args.daft_runner == "native"
    assert args.thumbnail_second == 1.0


def test_config_rejects_unsafe_table_name(tmp_path: Path):
    config = make_config(tmp_path)

    with pytest.raises(ValueError, match="invalid LakeSoul table"):
        ImportConfig(**{**config.__dict__, "table": "bad; DROP TABLE"})


def test_discover_videos_extracts_split_label_and_id(tmp_path: Path):
    first = tmp_path / "test" / "Archery" / "v_Archery_g01_c01.avi"
    second = tmp_path / "train" / "BaseballPitch" / "v_Baseball_g02_c03.avi"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    rows = discover_videos(tmp_path)

    assert rows == [
        {
            "video_id": "v_Archery_g01_c01",
            "split": "test",
            "label": "Archery",
            "video_path": str(first.resolve()),
        },
        {
            "video_id": "v_Baseball_g02_c03",
            "split": "train",
            "label": "BaseballPitch",
            "video_path": str(second.resolve()),
        },
    ]
    assert discover_videos(tmp_path, limit=1) == rows[:1]


def test_probe_video_parses_ffprobe_json(monkeypatch: pytest.MonkeyPatch):
    payload = {
        "streams": [
            {
                "codec_name": "mpeg4",
                "width": 320,
                "height": 240,
                "r_frame_rate": "30000/1001",
            }
        ],
        "format": {"duration": "4.25"},
    }

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = probe_video("clip.avi", "ffprobe", 10.0)

    assert result["codec"] == "mpeg4"
    assert result["width"] == 320
    assert result["height"] == 240
    assert result["fps"] == pytest.approx(29.97002997)
    assert result["duration_sec"] == 4.25
    assert result["error"] is None


def test_extract_thumbnail_falls_back_to_first_frame(
    monkeypatch: pytest.MonkeyPatch,
):
    commands = []
    results = iter(
        [
            subprocess.CompletedProcess([], 1, b"", b"seek failed"),
            subprocess.CompletedProcess([], 0, b"jpeg", b""),
        ]
    )

    def fake_run(command, **kwargs):
        commands.append(command)
        return next(results)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = extract_thumbnail("clip.avi", 1.0, "ffmpeg", 10.0)

    assert result == {"thumbnail_blob": b"jpeg", "error": None}
    assert commands[0][4] == "1.0"
    assert commands[1][4] == "0.0"


def test_write_thumbnail_uses_split_and_label_directories(tmp_path: Path):
    row = {
        "video_id": "v_Archery_g01_c01",
        "split": "train",
        "label": "Archery",
        "thumbnail_blob": b"jpeg",
    }

    output = Path(_write_thumbnail(tmp_path, row))

    assert output == (
        tmp_path / "train" / "Archery" / "v_Archery_g01_c01.jpg"
    ).resolve()
    assert output.read_bytes() == b"jpeg"


def test_build_daft_frame_has_expected_schema(tmp_path: Path):
    video = tmp_path / "train" / "Archery" / "v_Archery_g01_c01.avi"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")

    frame = build_daft_frame(make_config(tmp_path))

    assert frame.schema().column_names() == list(TABLE_COLUMNS)


def test_normalize_ray_batch_uses_target_lakesoul_schema():
    source = pa.table(
        {
            "video_id": pa.array(["video"], type=pa.string()),
            "thumbnail_blob": pa.array([b"jpeg"], type=pa.binary()),
        }
    )
    target = pa.schema(
        [
            pa.field("video_id", pa.large_string()),
            pa.field("thumbnail_blob", pa.large_binary()),
        ]
    )

    normalized = _normalize_ray_batch(source, target)

    assert normalized.schema == target


def test_import_videos_uses_ray_lakesoul_datasink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    video = tmp_path / "train" / "Archery" / "v_Archery_g01_c01.avi"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    config = make_config(tmp_path)
    row = {
        "video_id": "v_Archery_g01_c01",
        "split": "train",
        "label": "Archery",
        "video_path": str(video.resolve()),
        "codec": "mpeg4",
        "width": 320,
        "height": 240,
        "fps": 25.0,
        "duration_sec": 3.5,
        "thumbnail_path": str(tmp_path / "thumbnail.jpg"),
        "thumbnail_blob": b"jpeg",
    }
    table = pa.Table.from_pylist([row])
    calls = {}

    class FakeDataset:
        def map_batches(self, fn, **kwargs):
            calls["map_batches"] = {"fn": fn, **kwargs}
            return self

        def materialize(self):
            calls["materialized"] = True
            return self

        def count(self):
            return 1

        def iter_batches(self, **kwargs):
            calls["iter_batches"] = kwargs
            yield table

        def write_datasink(self, sink, **kwargs):
            calls["sink"] = sink
            calls["write_datasink"] = kwargs

    class FakeFrame:
        class Schema:
            def to_pyarrow_schema(self):
                return table.schema

        def schema(self):
            return self.Schema()

        def to_ray_dataset(self):
            calls["to_ray_dataset"] = True
            return FakeDataset()

    class FakeSink:
        def __init__(self, table_name, **kwargs):
            self.table_name = table_name
            self.options = kwargs

    monkeypatch.setattr(video_import, "build_daft_frame", lambda _: FakeFrame())
    monkeypatch.setattr(video_import, "create_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        video_import,
        "get_arrow_schema_by_table_name",
        lambda _: table.schema,
    )
    monkeypatch.setattr(video_import, "LakeSoulDatasink", FakeSink)
    monkeypatch.setattr(
        video_import,
        "_validate_table",
        lambda table_name, batch_size: (1, 4, video_import._digest_payloads(
            [(row["video_id"], video_import._row_payload(row))]
        )),
    )

    summary = import_videos(config)

    assert calls["materialized"] is True
    assert calls["map_batches"]["batch_size"] == 8
    assert calls["map_batches"]["batch_format"] == "pyarrow"
    assert calls["map_batches"]["fn_kwargs"] == {"schema": table.schema}
    assert calls["sink"].table_name == "ucf101_video"
    assert calls["sink"].options == {
        "format": "parquet",
        "batch_size": 8,
        "thread_num": 2,
    }
    assert calls["write_datasink"] == {
        "ray_remote_args": {"max_retries": 0},
        "concurrency": 2,
    }
    assert summary["imported_rows"] == 1
    assert summary["skipped_rows"] == 0
    assert summary["thumbnail_bytes"] == 4
