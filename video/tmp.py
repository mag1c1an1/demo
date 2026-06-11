import json
import subprocess
from pathlib import Path

import daft
from daft import col
from daft.datatype import DataType


def probe_video(path: str) -> dict:
    cmd = [
        "ffprobe",
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

    try:
        out = subprocess.check_output(cmd, text=True)
        data = json.loads(out)

        stream = data["streams"][0]
        fmt = data["format"]

        fps_raw = stream.get("r_frame_rate", "0/1")
        num, den = fps_raw.split("/")
        fps = float(num) / float(den) if float(den) != 0 else None

        return {
            "codec": stream.get("codec_name"),
            "width": int(stream.get("width")),
            "height": int(stream.get("height")),
            "fps": fps,
            "duration_sec": float(fmt.get("duration")),
        }
    except Exception:
        return {
            "codec": None,
            "width": None,
            "height": None,
            "fps": None,
            "duration_sec": None,
        }


def extract_thumbnail(path: str) -> bytes | None:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-ss",
        "1",
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

    try:
        return subprocess.check_output(cmd)
    except Exception:
        return None


probe_return_type = DataType.struct(
    {
        "codec": DataType.string(),
        "width": DataType.int64(),
        "height": DataType.int64(),
        "fps": DataType.float64(),
        "duration_sec": DataType.float64(),
    }
)

probe_video_udf = daft.udf(
    probe_video,
    return_dtype=probe_return_type,
)

thumbnail_udf = daft.udf(
    extract_thumbnail,
    return_dtype=DataType.binary(),
)


df = daft.read_csv("ucf101_metadata.csv")

df = df.with_column(
    "video_meta",
    probe_video_udf(col("video_path")),
)

df = (
    df.with_column("codec", col("video_meta").struct.get("codec"))
    .with_column("width", col("video_meta").struct.get("width"))
    .with_column("height", col("video_meta").struct.get("height"))
    .with_column("fps", col("video_meta").struct.get("fps"))
    .with_column("duration_sec", col("video_meta").struct.get("duration_sec"))
    .exclude("video_meta")
)

df = df.with_column(
    "thumbnail",
    thumbnail_udf(col("video_path")),
)

df.show(5)

df.write_parquet("ucf101_lakesoul_demo_parquet")


import json
import subprocess
import ray
import ray.data

ray.init()


def probe_and_thumbnail(row: dict) -> dict:
    path = row["video_path"]

    # ffprobe: metadata
    try:
        probe_cmd = [
            "ffprobe",
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
        out = subprocess.check_output(probe_cmd, text=True)
        data = json.loads(out)

        stream = data["streams"][0]
        fmt = data["format"]

        fps_raw = stream.get("r_frame_rate", "0/1")
        num, den = fps_raw.split("/")
        fps = float(num) / float(den) if float(den) != 0 else None

        row["codec"] = stream.get("codec_name")
        row["width"] = int(stream.get("width"))
        row["height"] = int(stream.get("height"))
        row["fps"] = fps
        row["duration_sec"] = float(fmt.get("duration"))
    except Exception:
        row["codec"] = None
        row["width"] = None
        row["height"] = None
        row["fps"] = None
        row["duration_sec"] = None

    # ffmpeg: thumbnail binary
    try:
        thumb_cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-ss",
            "1",
            "-i",
            path,
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-vcodec", "mjpeg",
            "pipe:1",
        ]
        row["thumbnail"] = subprocess.check_output(thumb_cmd)
    except Exception:
        row["thumbnail"] = None

    return row


ds = ray.data.read_csv("ucf101_metadata.csv")

ds = ds.map(
    probe_and_thumbnail,
    concurrency=8,  # 并发处理 8 个视频，可按机器调整
)

ds.show(5)

ds.write_parquet("ucf101_lakesoul_demo_parquet")
