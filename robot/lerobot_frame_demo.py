from __future__ import annotations

import base64
import html
import io
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import av

from import_lerobot_pusht import (
    DEFAULT_TABLE_PREFIX,
    TableNames,
    _table_exists,
    read_lakesoul_rows,
)


FRAME_COLUMNS = [
    "observation_state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
    "next_reward",
    "next_done",
    "next_success",
]

EPISODE_COLUMNS = [
    "episode_index",
    "task_index",
    "fps",
    "length",
    "tasks",
    "observation_image_video_blob",
    "observation_image_from_timestamp",
    "observation_image_to_timestamp",
]


@dataclass(frozen=True)
class FrameSpec:
    episode_index: int
    frame_index: int
    label: str


@dataclass(frozen=True)
class LocatedFrame:
    spec: FrameSpec
    frame: dict[str, Any]
    episode: dict[str, Any]
    jpeg_bytes: bytes
    source_timestamp: float

    @property
    def locator(self) -> str:
        return (
            f"episodes[{self.spec.episode_index}]"
            f".observation_image_video_blob[{self.spec.frame_index}]"
        )


def table_names_for_prefix(prefix: str = DEFAULT_TABLE_PREFIX) -> TableNames:
    return TableNames(
        frames=f"{prefix}_frames",
        episodes=f"{prefix}_episodes",
        videos=f"{prefix}_videos",
    )


def lakesoul_tables_exist(prefix: str = DEFAULT_TABLE_PREFIX) -> bool:
    tables = table_names_for_prefix(prefix)
    return all(_table_exists(table_name) for table_name in tables.values())


def select_fixed_frame_specs(
    episode_rows: Sequence[dict[str, Any]], *, max_episodes: int = 2
) -> list[FrameSpec]:
    if max_episodes <= 0:
        raise ValueError("max episodes must be positive")
    specs: list[FrameSpec] = []
    for episode in sorted(episode_rows, key=lambda row: int(row["episode_index"]))[
        :max_episodes
    ]:
        episode_index = int(episode["episode_index"])
        length = int(episode["length"])
        if length <= 0:
            raise ValueError(f"episode {episode_index} has non-positive length")
        seen: set[int] = set()
        for label, frame_index in (
            ("start", 0),
            ("middle", length // 2),
            ("end", length - 1),
        ):
            if frame_index in seen:
                continue
            seen.add(frame_index)
            specs.append(FrameSpec(episode_index, frame_index, label))
    return specs


def normalize_frame_specs(
    frame_specs: Iterable[FrameSpec | tuple[int, int] | tuple[int, int, str]]
) -> list[FrameSpec]:
    normalized: list[FrameSpec] = []
    for spec in frame_specs:
        if isinstance(spec, FrameSpec):
            normalized.append(spec)
            continue
        if len(spec) == 2:
            episode_index, frame_index = spec
            label = "manual"
        elif len(spec) == 3:
            episode_index, frame_index, label = spec
        else:
            raise ValueError("frame specs must contain 2 or 3 values")
        normalized.append(FrameSpec(int(episode_index), int(frame_index), str(label)))
    return normalized


def decode_video_frame_rgb(video_blob: bytes, frame_index: int):
    if frame_index < 0:
        raise ValueError("frame index must be non-negative")
    with av.open(io.BytesIO(video_blob)) as container:
        for decoded_index, frame in enumerate(container.decode(video=0)):
            if decoded_index == frame_index:
                return frame.to_ndarray(format="rgb24")
    raise IndexError(f"video blob does not contain frame {frame_index}")


def encode_rgb_jpeg(rgb_array: Any) -> bytes:
    output = io.BytesIO()
    container = av.open(output, mode="w", format="mjpeg")
    try:
        stream = container.add_stream("mjpeg", rate=1)
        stream.width = int(rgb_array.shape[1])
        stream.height = int(rgb_array.shape[0])
        stream.pix_fmt = "yuvj420p"
        frame = av.VideoFrame.from_ndarray(rgb_array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    jpeg = output.getvalue()
    if not jpeg:
        raise RuntimeError("PyAV returned an empty JPEG")
    return jpeg


def _index_by_episode(rows: Iterable[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        indexed[int(row["episode_index"])] = row
    return indexed


def _index_by_episode_frame(
    rows: Iterable[dict[str, Any]],
) -> dict[tuple[int, int], dict[str, Any]]:
    indexed: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        indexed[(int(row["episode_index"]), int(row["frame_index"]))] = row
    return indexed


def locate_frames(
    frame_rows: Sequence[dict[str, Any]],
    episode_rows: Sequence[dict[str, Any]],
    frame_specs: Iterable[FrameSpec | tuple[int, int] | tuple[int, int, str]],
) -> list[LocatedFrame]:
    frames = _index_by_episode_frame(frame_rows)
    episodes = _index_by_episode(episode_rows)
    located: list[LocatedFrame] = []
    for spec in normalize_frame_specs(frame_specs):
        key = (spec.episode_index, spec.frame_index)
        if key not in frames:
            raise KeyError(f"missing frame row for episode/frame {key}")
        if spec.episode_index not in episodes:
            raise KeyError(f"missing episode row for episode {spec.episode_index}")
        frame = frames[key]
        episode = episodes[spec.episode_index]
        length = int(episode["length"])
        if spec.frame_index >= length:
            raise IndexError(
                f"frame {spec.frame_index} is outside episode {spec.episode_index} "
                f"length {length}"
            )
        rgb = decode_video_frame_rgb(
            bytes(episode["observation_image_video_blob"]),
            spec.frame_index,
        )
        fps = float(episode["fps"])
        source_timestamp = (
            float(episode["observation_image_from_timestamp"])
            + spec.frame_index / fps
        )
        located.append(
            LocatedFrame(
                spec=spec,
                frame=frame,
                episode=episode,
                jpeg_bytes=encode_rgb_jpeg(rgb),
                source_timestamp=source_timestamp,
            )
        )
    return located


def build_demo_frames(
    table_prefix: str = DEFAULT_TABLE_PREFIX,
    *,
    max_episodes: int = 2,
    frame_specs: Iterable[FrameSpec | tuple[int, int] | tuple[int, int, str]]
    | None = None,
    batch_size: int = 1024,
) -> list[LocatedFrame]:
    tables = table_names_for_prefix(table_prefix)
    episode_rows = read_lakesoul_rows(
        tables.episodes,
        EPISODE_COLUMNS,
        batch_size=batch_size,
    )
    specs = (
        select_fixed_frame_specs(episode_rows, max_episodes=max_episodes)
        if frame_specs is None
        else normalize_frame_specs(frame_specs)
    )
    frame_rows = read_lakesoul_rows(
        tables.frames,
        FRAME_COLUMNS,
        batch_size=batch_size,
    )
    return locate_frames(frame_rows, episode_rows, specs)


def _format_vector(values: Iterable[Any]) -> str:
    return "[" + ", ".join(f"{float(value):.2f}" for value in values) + "]"


def _image_data_uri(jpeg: bytes) -> str:
    encoded = base64.b64encode(jpeg).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def render_contact_sheet_html(frames: Sequence[LocatedFrame]) -> str:
    cards = []
    for item in frames:
        task = ", ".join(str(value) for value in item.episode.get("tasks") or [])
        cards.append(
            f"""
            <article class="frame-card">
              <img src="{_image_data_uri(item.jpeg_bytes)}" alt="{html.escape(item.locator)}" />
              <table>
                <tr><th>label</th><td>{html.escape(item.spec.label)}</td></tr>
                <tr><th>locator</th><td><code>{html.escape(item.locator)}</code></td></tr>
                <tr><th>frame key</th><td>index={int(item.frame["index"])}, episode={item.spec.episode_index}, frame={item.spec.frame_index}</td></tr>
                <tr><th>timestamp</th><td>local={float(item.frame["timestamp"]):.3f}s, source={item.source_timestamp:.3f}s</td></tr>
                <tr><th>state</th><td><code>{html.escape(_format_vector(item.frame["observation_state"]))}</code></td></tr>
                <tr><th>action</th><td><code>{html.escape(_format_vector(item.frame["action"]))}</code></td></tr>
                <tr><th>reward</th><td>{float(item.frame["next_reward"]):.4f}</td></tr>
                <tr><th>done</th><td>{bool(item.frame["next_done"])}</td></tr>
                <tr><th>task</th><td>{html.escape(task)}</td></tr>
              </table>
            </article>
            """
        )
    body = "\n".join(cards)
    return f"""
    <style>
      .lakesoul-frame-demo {{
        font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: #17202a;
      }}
      .lakesoul-frame-demo h2 {{
        font-size: 20px;
        margin: 0 0 12px;
      }}
      .frame-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 12px;
      }}
      .frame-card {{
        border: 1px solid #d7dde5;
        border-radius: 6px;
        padding: 10px;
        background: #ffffff;
      }}
      .frame-card img {{
        width: 100%;
        image-rendering: pixelated;
        border: 1px solid #eef1f5;
        border-radius: 4px;
        background: #f8fafc;
      }}
      .frame-card table {{
        width: 100%;
        margin-top: 8px;
        border-collapse: collapse;
        font-size: 12px;
      }}
      .frame-card th {{
        width: 74px;
        text-align: left;
        vertical-align: top;
        color: #536273;
        font-weight: 600;
        padding: 3px 6px 3px 0;
      }}
      .frame-card td {{
        padding: 3px 0;
        word-break: break-word;
      }}
      .frame-card code {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 11px;
      }}
    </style>
    <section class="lakesoul-frame-demo">
      <h2>LakeSoul LeRobot PushT Frame Locator</h2>
      <div class="frame-grid">
        {body}
      </div>
    </section>
    """
