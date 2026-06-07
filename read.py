import duckdb
import lakesoul._lib  # 或导入会触发 _lib 加载的上层接口
from lakesoul.arrow import lakesoul_dataset
from lakesoul.logging import init_logger

# init_logger("lakesoul_io=debug")
conn = duckdb.connect()
flickr30k = lakesoul_dataset("flickr30k")
conn.sql("SELECT * FROM flickr30k").show()
conn.sql("SELECT image_blob FROM flickr30k where filename = '3323.jpg'").show()
