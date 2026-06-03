# from datasets import load_dataset


# 先加载全部数据
# def main():
#     print("Hello from multi!")

import random
import string
from pathlib import Path

from lakesoul.spark import LakeSoulTable
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit


def __generate_random_string(length):
    characters = string.ascii_lowercase + string.digits
    return "".join(random.choice(characters) for _ in range(length))


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
    spark.sql("drop table if exists test_table").show()
    datalist = [("a", 1), ("b", 2), ("c", 3)]
    df = spark.createDataFrame(datalist, ["key", "value"])
    df.write.format("lakesoul").option("file_format", "vortex").saveAsTable(
        "test_table"
    )
    spark.sql("show tables").show()

    # trainDf = spark.read.format("csv").option("header", "true").load(trainFilePath)
    # trainDf = trainDf.withColumn("split", lit("train"))
    # print("Debug -- Load data into dataframe")
    # trainDf.show()

    # spark.sql("drop table if exists titanic_raw")
    # trainDf.write.mode("append").format("lakesoul").option(
    #     "rangePartitions", "split"
    # ).option("shortTableName", "titanic_raw").save(tablePath)
    # print("Debug -- Show tables after importing data")
    # spark.sql("show tables").show()
    # LakeSoulTable.forName(spark, "titanic_raw").toDF().show()

    spark.stop()


# if __name__ == "__main__":
#     main()
