import io

import av
import numpy as np
import pytest

from robot.lerobot_frame_demo import (
    FrameSpec,
    encode_rgb_jpeg,
    locate_frames,
    render_contact_sheet_html,
    select_fixed_frame_specs,
    table_names_for_prefix,
)


def make_test_video_blob() -> bytes:
    output = io.BytesIO()
    container = av.open(output, mode="w", format="mp4")
    try:
        stream = container.add_stream("libx264", rate=10)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for index in range(3):
            rgb = np.full((16, 16, 3), index * 80, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    return output.getvalue()


def test_table_names_for_prefix_matches_import_tables():
    assert table_names_for_prefix("demo").values() == (
        "demo_frames",
        "demo_episodes",
        "demo_videos",
    )


def test_select_fixed_frame_specs_uses_start_middle_end_per_episode():
    rows = [
        {"episode_index": 2, "length": 2},
        {"episode_index": 0, "length": 5},
        {"episode_index": 1, "length": 1},
    ]

    specs = select_fixed_frame_specs(rows, max_episodes=2)

    assert specs == [
        FrameSpec(0, 0, "start"),
        FrameSpec(0, 2, "middle"),
        FrameSpec(0, 4, "end"),
        FrameSpec(1, 0, "start"),
    ]


def test_encode_rgb_jpeg_returns_jpeg_bytes():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)

    jpeg = encode_rgb_jpeg(rgb)

    assert jpeg.startswith(b"\xff\xd8")
    assert len(jpeg) > 100


def test_locate_frames_decodes_episode_blob_and_metadata():
    video_blob = make_test_video_blob()
    episode_rows = [
        {
            "episode_index": 0,
            "task_index": 0,
            "fps": 10,
            "length": 3,
            "tasks": ["Push the T-shaped block onto the T-shaped target."],
            "observation_image_video_blob": video_blob,
            "observation_image_from_timestamp": 12.0,
            "observation_image_to_timestamp": 12.3,
        }
    ]
    frame_rows = [
        {
            "observation_state": [1.0, 2.0],
            "action": [3.0, 4.0],
            "timestamp": 0.1,
            "frame_index": 1,
            "episode_index": 0,
            "index": 11,
            "task_index": 0,
            "next_reward": 0.5,
            "next_done": False,
            "next_success": False,
        }
    ]

    located = locate_frames(frame_rows, episode_rows, [(0, 1, "middle")])

    assert len(located) == 1
    assert located[0].locator == "episodes[0].observation_image_video_blob[1]"
    assert located[0].source_timestamp == pytest.approx(12.1)
    assert located[0].jpeg_bytes.startswith(b"\xff\xd8")


def test_render_contact_sheet_html_contains_locator_and_image_data():
    video_blob = make_test_video_blob()
    located = locate_frames(
        [
            {
                "observation_state": [1.0, 2.0],
                "action": [3.0, 4.0],
                "timestamp": 0.0,
                "frame_index": 0,
                "episode_index": 0,
                "index": 10,
                "task_index": 0,
                "next_reward": 0.25,
                "next_done": False,
                "next_success": False,
            }
        ],
        [
            {
                "episode_index": 0,
                "task_index": 0,
                "fps": 10,
                "length": 3,
                "tasks": ["task <escaped>"],
                "observation_image_video_blob": video_blob,
                "observation_image_from_timestamp": 0.0,
                "observation_image_to_timestamp": 0.3,
            }
        ],
        [(0, 0, "start")],
    )

    html = render_contact_sheet_html(located)

    assert "data:image/jpeg;base64," in html
    assert "episodes[0].observation_image_video_blob[0]" in html
    assert "task &lt;escaped&gt;" in html
    assert "index=10, episode=0, frame=0" in html
