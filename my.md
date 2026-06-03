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
