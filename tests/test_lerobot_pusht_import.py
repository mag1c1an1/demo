import hashlib
from pathlib import Path

import pytest

from robot import import_lerobot_pusht as pusht


def test_cli_defaults_to_vortex_three_table_import():
    args = pusht.build_parser().parse_args([])
    config = pusht.config_from_args(args)

    assert config.source_dir == pusht.DEFAULT_SOURCE_DIR
    assert config.table_prefix == "lerobot_pusht_vortex"
    assert config.file_format == "vortex"
    assert config.episode_limit == 0
    assert config.tables.values() == (
        "lerobot_pusht_vortex_frames",
        "lerobot_pusht_vortex_episodes",
        "lerobot_pusht_vortex_videos",
    )


def test_import_config_rejects_unsafe_table_prefix(tmp_path: Path):
    with pytest.raises(ValueError, match="invalid LakeSoul table prefix"):
        pusht.ImportConfig(
            source_dir=tmp_path,
            table_prefix="bad;DROP",
            batch_size=1024,
            episode_limit=0,
            overwrite=False,
            output=tmp_path / "out.json",
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
            media_timeout=30.0,
            ray_address="local",
            concurrency=4,
        )


def test_frame_source_row_to_lakesoul_renames_dotted_columns():
    row = {
        "observation.state": [1.5, 2.5],
        "action": [3, 4],
        "timestamp": 0.25,
        "frame_index": 2,
        "episode_index": 7,
        "index": 99,
        "task_index": 0,
        "next.reward": 0.75,
        "next.done": False,
        "next.success": True,
    }

    assert pusht.frame_source_row_to_lakesoul(row) == {
        "observation_state": [1.5, 2.5],
        "action": [3.0, 4.0],
        "timestamp": 0.25,
        "frame_index": 2,
        "episode_index": 7,
        "index": 99,
        "task_index": 0,
        "next_reward": 0.75,
        "next_done": False,
        "next_success": True,
    }


def test_build_video_rows_extracts_provenance_and_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    video_path = (
        tmp_path / "videos" / "observation.image" / "chunk-000" / "file-000.mp4"
    )
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"mp4-bytes")

    monkeypatch.setattr(
        pusht,
        "probe_video_path",
        lambda path, ffprobe_bin, timeout: {
            "codec": "av1",
            "width": 96,
            "height": 96,
            "fps": 10.0,
            "duration_sec": 2.0,
            "num_frames": 20,
        },
    )

    rows = pusht.build_video_rows(
        tmp_path,
        [video_path],
        ffprobe_bin="ffprobe",
        timeout=30.0,
    )

    assert rows[0]["camera_angle"] == "observation.image"
    assert rows[0]["chunk_index"] == 0
    assert rows[0]["file_index"] == 0
    assert rows[0]["relative_path"] == (
        "videos/observation.image/chunk-000/file-000.mp4"
    )
    assert rows[0]["file_size_bytes"] == len(b"mp4-bytes")
    assert rows[0]["sha256"] == hashlib.sha256(b"mp4-bytes").hexdigest()
    assert rows[0]["video_blob"] == b"mp4-bytes"


def test_build_episode_rows_aggregates_trajectory_and_segment(
    tmp_path: Path,
):
    video_path = (
        tmp_path / "videos" / "observation.image" / "chunk-000" / "file-000.mp4"
    )
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"source")
    episode_meta = [
        {
            "episode_index": 0,
            "length": 2,
            "dataset_from_index": 0,
            "dataset_to_index": 2,
            "videos/observation.image/chunk_index": 0,
            "videos/observation.image/file_index": 0,
            "videos/observation.image/from_timestamp": 0.0,
            "videos/observation.image/to_timestamp": 0.2,
            "tasks": ["Push the T-shaped block onto the T-shaped target."],
        }
    ]
    frames = [
        {
            "observation.state": [1.0, 2.0],
            "action": [3.0, 4.0],
            "timestamp": 0.0,
            "frame_index": 0,
            "episode_index": 0,
            "index": 0,
            "task_index": 0,
            "next.reward": 0.1,
            "next.done": False,
            "next.success": False,
        },
        {
            "observation.state": [5.0, 6.0],
            "action": [7.0, 8.0],
            "timestamp": 0.1,
            "frame_index": 1,
            "episode_index": 0,
            "index": 1,
            "task_index": 0,
            "next.reward": 0.2,
            "next.done": True,
            "next.success": False,
        },
    ]
    calls = {}

    def fake_extract(path, from_timestamp, length, fps, ffmpeg_bin, timeout):
        calls["args"] = (path, from_timestamp, length, fps, ffmpeg_bin, timeout)
        return b"segment"

    rows = pusht.build_episode_rows(
        episode_meta,
        frames,
        {0: "fallback task"},
        [video_path],
        fps=10.0,
        ffmpeg_bin="ffmpeg",
        media_timeout=30.0,
        segment_extractor=fake_extract,
    )

    assert calls["args"] == (video_path, 0.0, 2, 10.0, "ffmpeg", 30.0)
    assert rows[0]["episode_index"] == 0
    assert rows[0]["timestamps"] == [0.0, 0.1]
    assert rows[0]["actions"] == [[3.0, 4.0], [7.0, 8.0]]
    assert rows[0]["observation_state"] == [[1.0, 2.0], [5.0, 6.0]]
    assert rows[0]["next_done"] == [False, True]
    assert rows[0]["observation_image_video_blob"] == b"segment"
    assert rows[0]["observation_image_video_bytes"] == 7


def test_ensure_targets_are_absent_rejects_existing_tables(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        pusht,
        "_table_exists",
        lambda name: name.endswith("_frames"),
    )

    with pytest.raises(RuntimeError, match="already exist"):
        pusht.ensure_targets_are_absent(
            pusht.TableNames("demo_frames", "demo_episodes", "demo_videos"),
            overwrite=False,
        )


def test_digest_rows_hashes_binary_by_sha_and_length():
    rows = [{"id": 1, "blob": b"abc"}, {"id": 2, "blob": b"def"}]

    assert pusht.digest_rows(rows, "id") == pusht.digest_rows(reversed(rows), "id")
    assert pusht.digest_rows(rows, "id") != pusht.digest_rows(
        [{"id": 1, "blob": b"abc"}, {"id": 2, "blob": b"changed"}],
        "id",
    )
