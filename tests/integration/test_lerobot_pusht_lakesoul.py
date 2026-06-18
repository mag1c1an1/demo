import os
import uuid
from pathlib import Path

import pytest

from robot.import_lerobot_pusht import (
    ImportConfig,
    import_lerobot_pusht,
    read_lakesoul_rows,
)

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.environ.get("RUN_LAKESOUL_INTEGRATION") != "1",
    reason="set RUN_LAKESOUL_INTEGRATION=1 to run LakeSoul integration",
)
def test_lerobot_pusht_imports_three_vortex_tables(tmp_path):
    project_dir = Path(__file__).resolve().parents[2]
    source_dir = project_dir / "robot" / "lerobot-pusht"
    if not source_dir.is_dir():
        pytest.skip(f"LeRobot PushT source directory is absent: {source_dir}")

    prefix = f"lerobot_pusht_it_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    config = ImportConfig(
        source_dir=source_dir,
        table_prefix=prefix,
        batch_size=64,
        episode_limit=2,
        overwrite=True,
        output=tmp_path / "import.json",
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
        media_timeout=60.0,
        ray_address="local",
        concurrency=1,
    )
    tables = config.tables
    summary = import_lerobot_pusht(config)

    assert summary["selected_episodes"] == 2
    assert summary["selected_frames"] == 279
    assert summary["frames"]["rows"] == 279
    assert summary["episodes"]["rows"] == 2
    assert summary["videos"]["rows"] == 1
    assert summary["videos"]["logical_blob_bytes"] > 0
    assert summary["episodes"]["logical_blob_bytes"] > 0

    episode_rows = read_lakesoul_rows(
        tables.episodes,
        [
            "episode_index",
            "length",
            "observation_image_video_blob",
            "observation_image_video_sha256",
        ],
        batch_size=2,
    )
    assert [
        row["length"] for row in sorted(episode_rows, key=lambda r: r["episode_index"])
    ] == [161, 118]
    assert all(row["observation_image_video_blob"] for row in episode_rows)
