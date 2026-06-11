# import daft
# from daft import col, DataType
# from daft.functions import encode_image

# video_path = "/home/maji/data/Projects/multi/video/data/UCF101_subset/train/Basketball/v_Basketball_g01_c01.avi"

# df = daft.read_video_frames(
#     path=video_path,
#     image_height=480,
#     image_width=640,
#     is_key_frame=True,
#     sample_interval_seconds=1.0,
# )
# df = df.with_column( "video_path", daft.lit(video_path)).with_column("data", encode_image(col("data"), "JPEG"))

# # df = (
# #     df
# #     .with_column("video_path", daft.lit(video_path))
# #     .with_column("path", col("path").cast(DataType.string()))
# #     .with_column("frame_time_base", col("frame_time_base").cast(DataType.string()))
# #     .with_column("video_path", col("video_path").cast(DataType.string()))
# #     .with_column("data", encode_image(col("data"), "JPEG"))
# # )
# from lakesoul.metadata import create_table
# from lakesoul.ray import LakeSoulDatasink
# schema = df.schema().to_pyarrow_schema()
# create_table(
#     "video_frames_table",
#     table_schema=schema,
#     table_path="/tmp/lakesoul/video_frames_table",
# )
# ds = df.to_ray_dataset()
# import ray
# # print(ds.schema())
# # schema = ds.schema().to_pyarrow_schema()
# # print(schema)


# sink = LakeSoulDatasink("video_frames_table")
# ds.write_datasink(sink)

import ray
import lakesoul.ray
df = ray.data.read_lakesoul("video_frames_table").to_daft()

from daft import col
from daft.functions import decode_image

df = df.with_column(
    "data",
    decode_image(col("data"))
)

df.show()