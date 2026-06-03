是的，这个判断挺现实的。

如果按“AI 数据湖 / 多模态数据湖”的标准来看，**LakeSoul 现在更像是一个偏传统 Lakehouse / 表格式数据湖存储层**，离“AI 数据湖”还有明显距离。尤其是你说的：

> 多模态文件写入和读取还没测试完

这其实说明现在还处在 **基础 I/O 能力验证阶段**，还没到真正的“AI 数据管理能力”阶段。

我觉得可以分三层看。

## 1. 现在 LakeSoul 比较强的地方

LakeSoul 目前更核心的能力应该还是：

```text
表管理
元数据管理
增量写入
Upsert / CDC
分区管理
快照读取
Arrow / DataFusion / Spark / Flink 集成
对象存储读写
```

这些能力对传统数据湖是有价值的，尤其是：

```text
结构化数据
半结构化数据
CDC 数据
实时入湖
批流一体
```

比如订单表、用户行为表、日志表、业务事实表，这些 LakeSoul 是有比较明确定位的。

但是 AI 数据湖要求的不只是：

```text
把数据按表存起来
```

而是：

```text
把训练数据、非结构化文件、embedding、标注、版本、血缘、质量指标统一管理起来
```

这就不只是存储格式的问题了。

## 2. 多模态能力至少要先打通这条链路

多模态文件写入读取，其实只是第一步。比如图片、PDF、音频、视频，最基本要能做到：

```text
写入文件
记录文件 URI
记录文件元数据
读取文件
批量扫描文件
和表数据 join
在训练框架中消费
```

比如一张训练样本表：

```sql
CREATE TABLE image_samples (
  id STRING,
  image_uri STRING,
  caption STRING,
  label STRING,
  image_embedding ARRAY<FLOAT>,
  quality_score DOUBLE,
  version STRING
);
```

真正训练时流程可能是：

```text
LakeSoul 表扫描样本元数据
        ↓
根据 image_uri 读取对象存储文件
        ↓
解码图片 / 文本 / 音频
        ↓
组成 batch
        ↓
喂给 PyTorch / Ray / Spark / DataFusion
```

如果现在连“多模态文件写入和读取”都还没完全测试完，那说明目前最多只能说：

> LakeSoul 有潜力管理多模态数据的元数据，但还不能说已经具备完整多模态数据湖能力。

这个表述会更稳。

## 3. 真正 AI 数据湖还需要更上层的能力

比如你前面提到的模型训练数据管理，至少还要有：

```text
数据集版本
样本筛选
标注管理
质量检查
embedding 管理
向量索引
训练集 / 验证集 / 测试集切分
数据血缘
数据可复现
和 Ray / PyTorch / Spark ML 的集成
```

LakeSoul 现在如果只是做到：

```text
表 + 文件 URI + 快照版本
```

那它只能覆盖其中一小部分：

```text
数据版本：部分支持，靠快照/commit
样本筛选：部分支持，靠 SQL filter
标注管理：可以建表存，但没有专门系统
质量检查：需要外部任务做
embedding 管理：可以存向量字段，但没有向量检索能力
多模态读取：还需要完善
训练数据集成：需要 Ray/PyTorch Dataset 适配
```

所以你说“能力太弱了”并不夸张。更准确地说：

> LakeSoul 现在有数据湖表格式和增量存储能力，但距离 AI 数据湖所需的多模态数据管理、训练数据治理、向量检索和 ML 框架集成还有不少缺口。

## 我觉得当前比较现实的路线

不要一上来就宣传“AI 数据湖”。可以先把目标降成：

> LakeSoul 支持 AI 训练场景下的多模态数据集管理。

这个目标更可落地。

第一阶段先做：

```text
图片 / 文本 / PDF / 音频 / 视频文件的 URI 管理
文件元数据 schema
批量写入测试
批量读取测试
对象存储兼容测试
Ray Dataset 读取
PyTorch Dataset 读取
```

第二阶段做：

```text
样本表版本管理
训练集筛选 SQL
数据集 manifest 导出
质量分数字段
label / annotation 表
embedding 字段存储
```

第三阶段再做：

```text
向量索引
语义检索
自动标注
数据质量 profile
训练数据 lineage
和 MLflow / Ray Train / TorchData 集成
```

这样讲会更像一个可信 roadmap。

一句话总结：

> 现在 LakeSoul 还不应该直接对标完整 AI 数据湖，比较合理的定位是：先把 Lakehouse 的快照、增量、元数据能力扩展到多模态训练数据集管理上。多模态文件 I/O 测完之前，只能说它在补 AI 数据湖的地基。

我推荐你先用 **Flickr30k**，不要一上来用 COCO。

理由很简单：**Flickr30k 足够多模态，但规模没那么变态，适合先验证 LakeSoul 的能力边界。**

## 首选：Flickr30k

Flickr30k 是经典的 **图文多模态数据集**：大约 3 万张图片，每张图片有多条 caption，适合做 image-caption、图文检索、embedding、训练样本管理等实验。Hugging Face 上的 `nlphuji/flickr30k` 包含 `image`、`caption`、`split`、`img_id`、`filename` 等字段，结构很适合落到 LakeSoul 表里做管理。([Hugging Face][1])

你可以把它建成这种表：

```sql
CREATE TABLE multimodal_samples (
  sample_id STRING,
  image_uri STRING,
  filename STRING,
  captions ARRAY<STRING>,
  split STRING,
  image_width INT,
  image_height INT,
  image_embedding ARRAY<FLOAT>,
  text_embedding ARRAY<FLOAT>,
  quality_score DOUBLE,
  dataset_version STRING
);
```

然后测试这些能力：

```text
1. 图片文件写入对象存储
2. LakeSoul 表记录 image_uri + caption + split
3. 按 split 读取 train/val/test
4. 批量读取 image_uri 对应图片
5. 生成 image/text embedding 后回写
6. 用 SQL 筛选训练样本
7. 验证 snapshot/version 是否能复现同一批训练数据
```

这比 Food-101 更适合验证“AI 数据湖”，因为 Food-101 主要是 **图像分类**，而 Flickr30k 是 **图像 + 文本 caption**，天然就是多模态。

## 为什么不首推 COCO？

COCO 更完整，也更像工业级 benchmark。它是大规模目标检测、分割、captioning 数据集，官方说明它支持 object segmentation、recognition in context、captioning 等任务。([COCO Dataset][2])

但问题是：**COCO 太大、结构也更复杂**。有 images、annotations、captions、instances、categories、segmentation、bbox 等等。对于你现在这种“多模态文件写入读取还没测完”的阶段，COCO 会把问题复杂化。

我的建议是：

```text
第一阶段：Flickr30k
第二阶段：Food-101
第三阶段：COCO
```

## 可选：Food-101

Food-101 也不错。它有 101 个食物类别，共 101,000 张图片，每类包含 750 张训练图片和 250 张测试图片；训练集故意没有完全清洗，存在一些噪声和错误标签。([Hugging Face][3])

它适合测：

```text
图片文件批量写入
分类 label 管理
train/test split
质量检查
噪声标签处理
样本筛选
```

但它缺少文本 caption，所以多模态味道弱一些。

## 我给你的最终推荐

你现在最应该用：

```text
nlphuji/flickr30k
```

它最适合 LakeSoul 当前阶段，因为它刚好覆盖：

```text
图片文件
文本 caption
结构化 metadata
train/test split
样本版本
embedding 回写
样本筛选
质量检查
```

最小实验目标可以定成：

> 用 LakeSoul 管理 Flickr30k 的图片 URI、caption、split、embedding 和数据集版本，并验证从 LakeSoul 表中稳定读取同一批训练样本。

这比空泛地说“支持 AI 数据湖”更有说服力。

[1]: https://huggingface.co/datasets/nlphuji/flickr30k?utm_source=chatgpt.com "nlphuji/flickr30k · Datasets at Hugging Face"
[2]: https://cocodataset.org/?utm_source=chatgpt.com "COCO dataset"
[3]: https://huggingface.co/datasets/ethz/food101?utm_source=chatgpt.com "ethz/food101 · Datasets at Hugging Face"

你 leader 说的 **“直接用 blob 存”**，大概率是指：不要只在 LakeSoul 表里存 `image_uri` / `file_path`，而是把图片、音频、PDF 这类二进制内容本身作为字段写进 LakeSoul 表。

比如不是这样：

```sql
image_uri STRING
caption STRING
label STRING
```

而是这样：

```sql
image_blob BINARY
caption STRING
label STRING
```

或者 Arrow 里类似：

```text
image: binary / large_binary
caption: string
label: string
```

这个思路可以做，但我建议你要分清楚：**blob 存储适合做能力验证，不一定适合作为最终生产方案。**

## blob 存的好处

对 LakeSoul 来说，直接 blob 存有几个明显优点：

第一，**测试链路更直接**。

你可以验证 LakeSoul 是否真的支持：

```text
binary 字段写入
binary 字段读取
大字段 batch scan
Arrow binary array 转换
Python/Rust/Spark/Flink 端一致性
```

如果目标是证明“LakeSoul 能管理多模态数据”，那 blob 是最直观的。

第二，**数据和元数据强一致**。

如果图片内容本身就在表里，那 snapshot 版本天然包含图片内容。你读某个版本的 LakeSoul 表，就能拿到当时那一版图片和 caption。

这对训练数据复现很有吸引力：

```text
model_v1 使用 table snapshot 100
snapshot 100 里包含图片 blob + caption + label
以后重新读取 snapshot 100，数据内容不会变
```

第三，**不依赖外部文件路径**。

如果只存 URI：

```text
s3://bucket/images/001.jpg
```

那外部对象可能被删、被覆盖、权限变了、路径迁移了。blob 存在表里就少了这类问题。

## blob 存的问题

但 blob 也有很明显的坑。

最大的问题是：**大二进制文件会拖垮表格式读写路径**。

Lakehouse 表格式本质上更擅长处理：

```text
int
string
timestamp
decimal
struct
array
map
small binary
```

但如果每一行塞一张图片、一段音频、一个 PDF，问题就来了：

```text
Parquet/Arrow batch 变大
内存压力变大
shuffle 成本变高
projection/filter 的收益变差
小文件/大字段管理复杂
读取 caption 时也可能被 blob 拖累
```

比如你只是想查：

```sql
SELECT caption, label
FROM samples
WHERE split = 'train';
```

理论上不需要读图片。但如果文件布局、编码、reader 实现不好，blob 字段可能仍然造成 I/O 或内存压力。

第二个问题是：**更新成本高**。

如果你想修一个 caption 或 label，但图片 blob 跟同一行数据绑定在一起，写入放大可能更严重。

第三个问题是：**训练框架未必喜欢从表里读大 blob**。

PyTorch / Ray / WebDataset / Hugging Face Datasets 很多时候更习惯：

```text
metadata table + object storage file
```

也就是表里存路径，训练时按路径流式读取文件。

## 所以我建议你做两个版本

不要和 leader 的方向冲突。可以这样拆：

## A. blob 模式：证明 LakeSoul 支持多模态字段

这是最小验证。

表结构可以是：

```sql
CREATE TABLE multimodal_blob_samples (
  sample_id STRING,
  filename STRING,
  image_blob BINARY,
  caption STRING,
  label STRING,
  split STRING,
  image_format STRING,
  image_size BIGINT,
  dataset_version STRING
);
```

测试目标：

```text
1. 把图片读成 bytes
2. 写入 LakeSoul binary/blob 字段
3. 从 LakeSoul 读回 bytes
4. 校验 md5/sha256 一致
5. 用 PIL / OpenCV 解码读回的 bytes
6. caption/label/split 正常读取
7. batch scan 不 OOM
```

这个可以作为 **LakeSoul 多模态 blob I/O demo**。

## B. URI 模式：更接近生产 AI 数据湖

表结构可以是：

```sql
CREATE TABLE multimodal_uri_samples (
  sample_id STRING,
  image_uri STRING,
  caption STRING,
  label STRING,
  split STRING,
  image_sha256 STRING,
  image_size BIGINT,
  image_format STRING,
  dataset_version STRING
);
```

测试目标：

```text
1. 图片存在 S3/RustFS/本地对象存储
2. LakeSoul 只管理 image_uri 和 metadata
3. 读取表后根据 image_uri 拉取图片
4. 用 sha256 校验内容
5. 支持训练样本筛选
6. 支持 dataset version/snapshot 复现
```

这个更像真正的 AI 数据湖。

## 你可以跟 leader 这样解释

比较稳的说法是：

> 我们可以先按 blob 方式做最小闭环，验证 LakeSoul 对多模态二进制字段的写入、读取、快照和跨引擎兼容性。但生产场景下，大文件长期直接塞进表里可能会带来读写放大、内存压力和训练框架适配问题。所以建议同时保留 URI + checksum 的方案，后续对比 blob 模式和 external object 模式的性能与可维护性。

核心不是“blob 对不对”，而是：

```text
blob 模式适合验证 LakeSoul 存储能力；
URI 模式适合真实大规模多模态数据湖；
两者都可以支持，但使用场景不同。
```

## 数据集也可以相应调整

如果 leader 要 blob，那我更建议你先别用 Flickr30k 全量，先做一个小规模 subset：

```text
Flickr30k 取 1000 张
每张图片 bytes 写入 binary 字段
caption 存 string / array<string>
split 存 string
sha256 存 string
```

先把闭环跑通：

```text
image file -> bytes -> LakeSoul BINARY -> read bytes -> decode image -> compare checksum
```

这就是最有说服力的第一版。

然后再扩展到：

```text
1k samples
10k samples
full dataset
```

最后做 blob vs URI 对比。这样既符合 leader 的要求，也不会把自己锁死在一个可能不适合生产的设计里。
