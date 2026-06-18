import os

import pytest

from robot.import_lerobot_pusht import DEFAULT_TABLE_PREFIX
from robot.lerobot_frame_demo import (
    build_demo_frames,
    lakesoul_tables_exist,
    render_contact_sheet_html,
)

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.environ.get("RUN_LAKESOUL_INTEGRATION") != "1",
    reason="set RUN_LAKESOUL_INTEGRATION=1 to run LakeSoul integration",
)
def test_lerobot_frame_demo_decodes_default_tables():
    if not lakesoul_tables_exist(DEFAULT_TABLE_PREFIX):
        pytest.skip(
            "default LeRobot PushT LakeSoul tables are absent; "
            "run robot/import_lerobot_pusht.py first"
        )

    frames = build_demo_frames(DEFAULT_TABLE_PREFIX, max_episodes=2, batch_size=1024)
    html = render_contact_sheet_html(frames)

    assert len(frames) == 6
    assert all(frame.jpeg_bytes.startswith(b"\xff\xd8") for frame in frames)
    assert "LakeSoul LeRobot PushT Frame Locator" in html
    assert "episodes[" in html
