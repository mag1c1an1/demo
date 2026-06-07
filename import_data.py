import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    BinaryType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
IMAGES_DIR = DATA_DIR / "flickr30k-images"
ANNOTATIONS_DIR = DATA_DIR / "Annotations"
FILE_FORMAT = "vortex"


def get_int_text(parent, tag: str) -> int:
    elem = parent.find(tag)
    if elem is None or elem.text is None:
        raise ValueError(f"missing {tag}")
    return int(elem.text)


def parse_bboxes(xml_path: Path) -> list[dict]:
    """Parse a Flickr30k Entities XML annotation file into a list of bbox dicts."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bboxes = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue  # skip scene/noobject entries
        chain_ids = [n.text for n in obj.findall("name")]

        bboxes.append(
            {
                "chain_ids": chain_ids,
                "xmin": int(get_int_text(bndbox, "xmin")),
                "ymin": int(get_int_text(bndbox, "ymin")),
                "xmax": int(get_int_text(bndbox, "xmax")),
                "ymax": int(get_int_text(bndbox, "ymax")),
            }
        )
    return bboxes


def parse_image_size(xml_path: Path) -> tuple[int, int]:
    """Return (width, height) from the XML annotation."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    return int(get_int_text(size, "width")), int(get_int_text(size, "height"))


def clean_caption(text: str) -> str:
    """Strip <start>/<end> tags from a caption string."""
    return re.sub(r"</?start>|</?end>", "", text).strip()


def write_table(spark, schema):
    """Write Flickr30k data to LakeSoul table in batches to avoid OOM."""
    with open(DATA_DIR / "dataset_flickr30k_allEN.json") as f:
        captions_data = json.load(f)

    BATCH_SIZE = 100
    batch = []
    total = 0
    first_batch = True

    for fname, raw_captions in captions_data.items():
        img_path = IMAGES_DIR / fname
        xml_path = ANNOTATIONS_DIR / f"{fname[:-4]}.xml"

        if not img_path.exists() or not xml_path.exists():
            continue

        image_blob = img_path.read_bytes()
        captions = [clean_caption(c) for c in raw_captions]
        bboxes = parse_bboxes(xml_path)
        width, height = parse_image_size(xml_path)

        batch.append((fname, image_blob, width, height, captions, bboxes))
        total += 1

        if len(batch) >= BATCH_SIZE:
            df = spark.createDataFrame(batch, schema)
            if first_batch:
                df.write.format("lakesoul").option(
                    "file_format", FILE_FORMAT
                ).saveAsTable("flickr30k")
                first_batch = False
            else:
                df.write.mode("append").format("lakesoul").option(
                    "file_format", FILE_FORMAT
                ).saveAsTable("flickr30k")
            print(f"Written {total} rows")
            batch = []

    # Write remaining
    if batch:
        df = spark.createDataFrame(batch, schema)
        if first_batch:
            df.write.format("lakesoul").option("file_format", FILE_FORMAT).saveAsTable(
                "flickr30k"
            )
        else:
            df.write.mode("append").format("lakesoul").option(
                "file_format", FILE_FORMAT
            ).saveAsTable("flickr30k")
        print(f"Written {total} rows (done)")


if __name__ == "__main__":
    PROJECT_DIR = Path(__file__).resolve().parent
    jar_path = str(PROJECT_DIR / "lakesoul-spark-3.3-3.0.0-SNAPSHOT.jar")
    print(jar_path)
    spark = (
        SparkSession.builder.master("local[4]")
        .config("spark.jars", jar_path)
        .config(
            "spark.sql.extensions",
            "com.dmetasoul.lakesoul.sql.LakeSoulSparkSessionExtension",
        )
        .config(
            "spark.sql.catalog.lakesoul",
            "org.apache.spark.sql.lakesoul.catalog.LakeSoulCatalog",
        )
        .config("spark.sql.defaultCatalog", "lakesoul")
        .config("spark.hadoop.fs.s3.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.buffer.dir", "/tmp/opt/spark/work-dir/s3a")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000")
        .config("spark.hadoop.fs.s3a.access.key", "rustfsadmin")
        .config("spark.hadoop.fs.s3a.secret.key", "rustfsadmin")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # tablePath = "s3://lakesoul-test-bucket/titanic_raw"
    # trainFilePath = "/opt/spark/work-dir/titanic/dataset/train.csv"
    print("Debug -- Show tables before importing data")
    spark.sql("drop table if exists flickr30k").show()

    bbox_schema = StructType(
        [
            StructField("chain_ids", ArrayType(StringType()), False),
            StructField("xmin", IntegerType(), False),
            StructField("ymin", IntegerType(), False),
            StructField("xmax", IntegerType(), False),
            StructField("ymax", IntegerType(), False),
        ]
    )
    schema = StructType(
        [
            StructField("filename", StringType(), False),
            StructField("image_blob", BinaryType(), False),
            StructField("width", IntegerType(), False),
            StructField("height", IntegerType(), False),
            StructField("captions", ArrayType(StringType()), False),
            StructField("bboxes", ArrayType(bbox_schema), True),
        ]
    )

    write_table(spark, schema)
    spark.stop()
