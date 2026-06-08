from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from multimodal_data import ImageRecord


def make_jpeg(width: int, height: int, color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def jpeg_bytes():
    return make_jpeg(8, 6, (20, 40, 60))


@pytest.fixture
def tiny_image_records():
    return [
        ImageRecord(
            f"{index}.jpg",
            make_jpeg(8 + index, 6 + index, (20 * index, 40, 60)),
            8 + index,
            6 + index,
            (f"caption {index} a", f"caption {index} b"),
            (),
        )
        for index in range(1, 5)
    ]


@pytest.fixture(scope="session")
def spark_session():
    jar_path = Path(__file__).resolve().parents[1] / (
        "lakesoul-spark-3.3-3.0.0-SNAPSHOT.jar"
    )
    if not jar_path.is_file():
        pytest.skip(f"LakeSoul Spark JAR is absent: {jar_path}")

    from import_data import create_spark_session

    spark = create_spark_session(jar_path)
    try:
        yield spark
    finally:
        spark.stop()
