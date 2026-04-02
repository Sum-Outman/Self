# Self AGI 示例多模态数据集

这是一个用于测试Self AGI系统真实数据加载功能的示例数据集。

## 数据集结构
```
data/multimodal/
├── images/                # 图像文件
│   └── image_*.png
├── annotations/           # 标注文件
│   └── annotations.jsonl
└── dataset_info.json      # 数据集信息
```

## 标注格式
标注文件使用JSONL格式（每行一个JSON对象），每个对象包含以下字段：

```json
{
  "id": "样本ID",
  "image_path": "图像相对路径",
  "text": "文本描述",
  "labels": {
    "category": "类别名称",
    "category_id": 类别ID,
    "multilabel": [类别ID],
    "is_example": true
  },
  "metadata": {
    "source": "数据集来源",
    "generated_by": "创建脚本",
    "image_size": [宽度, 高度],
    "file_size": 文件大小
  }
}
```

## 使用示例

### 在RealMultimodalDataset中使用
```python
from training.real_multimodal_dataset import RealMultimodalDataset, DataSourceType

config = {
    "data_root": "data/multimodal",
    "annotations_path": "annotations/annotations.jsonl",
    "strict_real_data": True
}

dataset = RealMultimodalDataset(config, mode="train", data_source=DataSourceType.REAL_IMAGE_TEXT)
print(f"数据集大小: {len(dataset)}")
```

## 注意事项
1. 此数据集仅用于测试和演示目的
2. 图像为程序生成，但标注格式符合真实数据集要求
3. 数据集兼容"严格模式"，可用于验证真实数据加载功能
