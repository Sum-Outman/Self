# Dataset Preparation | 数据集准备

This guide covers preparing datasets for training AGI models in the Self AGI system, including data collection, preprocessing, formatting, and quality assurance.

本指南涵盖为 Self AGI 系统中训练 AGI 模型准备数据集，包括数据收集、预处理、格式化和质量保证。

## Dataset Types | 数据集类型

### Text Datasets | 文本数据集
- **Plain Text**: Raw text files (.txt) with one document per line or file
- **Structured Text**: JSON, CSV, XML files with structured text data
- **Conversational Data**: Dialog datasets with multiple turns and speakers
- **Code Data**: Source code files in various programming languages
- **Scientific Text**: Research papers, technical documents, mathematical text

- **纯文本**: 原始文本文件 (.txt)，每行或每个文件一个文档
- **结构化文本**: 带有结构化文本数据的 JSON、CSV、XML 文件
- **对话数据**: 具有多轮对话和多个说话者的对话数据集
- **代码数据**: 各种编程语言的源代码文件
- **科学文本**: 研究论文、技术文档、数学文本

### Image Datasets | 图像数据集
- **Classification Datasets**: Images with single class labels
- **Object Detection**: Images with bounding boxes and object labels
- **Segmentation**: Images with pixel-level masks
- **Captioning**: Images with descriptive text captions
- **Multiview Images**: Multiple views of the same object/scene

- **分类数据集**: 带有单个类别标签的图像
- **目标检测**: 带有边界框和目标标签的图像
- **分割**: 带有像素级掩码的图像
- **描述生成**: 带有描述性文本描述的图像
- **多视图图像**: 同一对象/场景的多个视图

### Audio Datasets | 音频数据集
- **Speech Recognition**: Audio recordings with transcriptions
- **Speaker Identification**: Audio with speaker labels
- **Music Data**: Music recordings with genre/instrument labels
- **Environmental Sounds**: Non-speech audio with sound type labels
- **Multilingual Audio**: Audio in multiple languages

- **语音识别**: 带有转录的音频录制
- **说话者识别**: 带有说话者标签的音频
- **音乐数据**: 带有流派/乐器标签的音乐录制
- **环境声音**: 带有声音类型标签的非语音音频
- **多语言音频**: 多种语言的音频

### Video Datasets | 视频数据集
- **Action Recognition**: Videos with action labels
- **Temporal Segmentation**: Videos with temporal boundaries
- **Video Captioning**: Videos with descriptive captions
- **Multimodal Video**: Videos with associated audio and text
- **3D Video**: Depth-aware or stereo video data

- **动作识别**: 带有动作标签的视频
- **时间分割**: 带有时间边界的视频
- **视频描述生成**: 带有描述性描述的视频
- **多模态视频**: 带有相关音频和文本的视频
- **3D视频**: 深度感知或立体视频数据

### Multimodal Datasets | 多模态数据集
- **Text-Image Pairs**: Images with corresponding text descriptions
- **Audio-Visual Data**: Video with synchronized audio
- **Text-Code-Image**: Combined text, code, and image data
- **Robot Sensor Data**: Combined camera, LIDAR, IMU, and joint state data
- **Embodied AI Data**: Data from physical or simulated robot interactions

- **文本-图像对**: 带有对应文本描述的图像
- **音频-视觉数据**: 带有同步音频的视频
- **文本-代码-图像**: 组合的文本、代码和图像数据
- **机器人传感器数据**: 组合的摄像头、激光雷达、IMU 和关节状态数据
- **具身AI数据**: 来自物理或模拟机器人交互的数据

## Data Collection | 数据收集

### Collection Methods | 收集方法

#### Web Scraping | 网络爬取
```python
from data_collection.web_scraper import AGIWebScraper

# Initialize web scraper
scraper = AGIWebScraper(
    domains=["example.com", "research.org"],
    content_types=["text", "images", "pdf"],
    rate_limit=1.0  # Requests per second
)

# Scrape data
scraped_data = scraper.scrape(
    max_pages=1000,
    save_dir="./collected_data"
)
```

#### API Collection | API收集
```python
from data_collection.api_collector import APIDataCollector

# Initialize API collector
collector = APIDataCollector(
    api_keys={"openai": "key1", "huggingface": "key2"},
    rate_limit=10  # Requests per minute
)

# Collect data from APIs
api_data = collector.collect_from_apis(
    apis=["arxiv", "wikipedia", "common_crawl"],
    query="AGI research",
    max_results=1000
)
```

#### Simulation Data Generation | 模拟数据生成
```python
from data_collection.simulator import DataSimulator

# Initialize data simulator
simulator = DataSimulator(
    robot_model="humanoid_v2",
    environment="home_environment",
    sensors=["camera", "lidar", "imu", "microphone"]
)

# Generate simulated robot data
simulated_data = simulator.generate_data(
    num_episodes=1000,
    episode_length=1000,  # Time steps
    tasks=["navigation", "manipulation", "dialogue"]
)
```

#### Real-World Data Collection | 真实世界数据收集
```python
from data_collection.real_world_collector import RealWorldDataCollector

# Initialize real-world collector
collector = RealWorldDataCollector(
    robot_interface="ros2",
    sensors=["rgb_camera", "depth_camera", "microphone_array", "force_torque"],
    storage_path="./real_world_data"
)

# Collect real-world data
real_data = collector.collect(
    duration_hours=24,
    activities=["household_tasks", "conversations", "object_manipulation"],
    annotation_mode="semi_automatic"
)
```

### Data Licensing | 数据许可
- **Open Datasets**: Use datasets with permissive licenses (CC-BY, MIT, Apache 2.0)
- **Commercial Use**: Ensure datasets allow commercial use if needed
- **Attribution Requirements**: Follow attribution requirements for each dataset
- **Privacy Compliance**: Ensure compliance with privacy regulations (GDPR, CCPA)
- **Ethical Considerations**: Avoid datasets with biased or harmful content

- **开放数据集**: 使用具有宽松许可证的数据集 (CC-BY, MIT, Apache 2.0)
- **商业用途**: 确保数据集允许商业用途（如果需要）
- **署名要求**: 遵循每个数据集的署名要求
- **隐私合规**: 确保符合隐私法规 (GDPR, CCPA)
- **伦理考虑**: 避免使用带有偏见或有害内容的数据集

## Data Preprocessing | 数据预处理

### Text Preprocessing | 文本预处理

#### Cleaning | 清洗
```python
from preprocessing.text import TextCleaner

cleaner = TextCleaner()

# Clean text data
cleaned_text = cleaner.clean(
    raw_text,
    remove_html=True,
    remove_urls=True,
    remove_emojis=True,
    normalize_whitespace=True
)
```

#### Tokenization | 分词
```python
from preprocessing.text import Tokenizer

# Initialize tokenizer
tokenizer = Tokenizer(
    vocab_size=50257,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

# Tokenize text
tokens = tokenizer.tokenize(
    text_data,
    max_length=512,
    truncation=True,
    padding=True
)
```

#### Normalization | 归一化
```python
from preprocessing.text import TextNormalizer

normalizer = TextNormalizer()

# Normalize text
normalized_text = normalizer.normalize(
    text_data,
    lowercase=True,
    remove_accents=True,
    expand_contractions=True,
    normalize_numbers=True
)
```

### Image Preprocessing | 图像预处理

#### Resizing and Cropping | 调整大小和裁剪
```python
from preprocessing.image import ImageProcessor

processor = ImageProcessor()

# Process images
processed_images = processor.process(
    image_data,
    target_size=(224, 224),
    crop_mode="center",  # or "random", "resize"
    maintain_aspect_ratio=True
)
```

#### Augmentation | 增强
```python
from preprocessing.image import ImageAugmentor

augmentor = ImageAugmentor()

# Augment images
augmented_images = augmentor.augment(
    image_data,
    augmentations=[
        "random_flip",
        "random_rotation",
        "color_jitter",
        "random_crop"
    ],
    augmentation_probability=0.5
)
```

#### Normalization | 归一化
```python
from preprocessing.image import ImageNormalizer

normalizer = ImageNormalizer()

# Normalize images
normalized_images = normalizer.normalize(
    image_data,
    mean=[0.485, 0.456, 0.406],  # ImageNet mean
    std=[0.229, 0.224, 0.225]     # ImageNet std
)
```

### Audio Preprocessing | 音频预处理

#### Waveform Processing | 波形处理
```python
from preprocessing.audio import AudioProcessor

processor = AudioProcessor()

# Process audio
processed_audio = processor.process(
    audio_data,
    target_sample_rate=16000,
    target_duration=5.0,  # seconds
    normalize_volume=True
)
```

#### Feature Extraction | 特征提取
```python
from preprocessing.audio import FeatureExtractor

extractor = FeatureExtractor()

# Extract features
features = extractor.extract(
    audio_data,
    features=["mfcc", "mel_spectrogram", "chroma"],
    n_mfcc=13,
    n_mels=64
)
```

#### Augmentation | 增强
```python
from preprocessing.audio import AudioAugmentor

augmentor = AudioAugmentor()

# Augment audio
augmented_audio = augmentor.augment(
    audio_data,
    augmentations=[
        "time_stretch",
        "pitch_shift",
        "add_noise",
        "reverb"
    ]
)
```

### Video Preprocessing | 视频预处理

#### Frame Extraction | 帧提取
```python
from preprocessing.video import VideoProcessor

processor = VideoProcessor()

# Extract frames
frames = processor.extract_frames(
    video_data,
    frame_rate=30,  # Frames per second
    max_frames=300,
    resize=(224, 224)
)
```

#### Temporal Processing | 时间处理
```python
from preprocessing.video import TemporalProcessor

processor = TemporalProcessor()

# Process temporal sequences
processed_video = processor.process(
    video_frames,
    segment_length=16,  # Frames per segment
    overlap=8,          # Overlap between segments
    temporal_augmentation=True
)
```

### Multimodal Alignment | 多模态对齐

#### Temporal Alignment | 时间对齐
```python
from preprocessing.multimodal import TemporalAligner

aligner = TemporalAligner()

# Align multimodal data
aligned_data = aligner.align(
    modalities=["video", "audio", "text"],
    timestamps=timestamps,
    alignment_strategy="dynamic_time_warping"
)
```

#### Feature Fusion | 特征融合
```python
from preprocessing.multimodal import FeatureFuser

fuser = FeatureFuser()

# Fuse multimodal features
fused_features = fuser.fuse(
    features={
        "text": text_features,
        "image": image_features,
        "audio": audio_features
    },
    fusion_method="attention",  # or "concatenation", "cross_attention"
    fusion_dimension=768
)
```

## Dataset Formatting | 数据集格式化

### Standard Formats | 标准格式

#### JSONL Format | JSONL 格式
```json
{
  "id": "example_001",
  "text": "This is an example text.",
  "image_path": "images/example_001.jpg",
  "audio_path": "audio/example_001.wav",
  "metadata": {
    "source": "web",
    "language": "en",
    "license": "CC-BY-4.0"
  },
  "annotations": {
    "text_labels": ["example", "sample"],
    "image_labels": ["object", "scene"],
    "audio_labels": ["speech", "english"]
  }
}
```

#### TFRecord Format | TFRecord 格式
```python
from preprocessing.formats import TFRecordConverter

converter = TFRecordConverter()

# Convert to TFRecord
converter.convert_to_tfrecord(
    dataset=dataset,
    output_path="./data/tfrecords",
    shard_size=1000  # Examples per shard
)
```

#### HuggingFace Dataset Format | HuggingFace 数据集格式
```python
from datasets import Dataset

# Create HuggingFace dataset
hf_dataset = Dataset.from_dict({
    "text": text_data,
    "image": image_paths,
    "audio": audio_paths
})

# Save dataset
hf_dataset.save_to_disk("./data/hf_dataset")
```

### Custom Formats | 自定义格式

#### AGI Dataset Format | AGI 数据集格式
```python
from preprocessing.formats import AGIDatasetFormat

# Create AGI dataset format
agi_format = AGIDatasetFormat(
    modalities=["text", "image", "audio", "video", "sensor"],
    compression="zstd",
    indexing=True
)

# Save dataset
agi_format.save_dataset(
    dataset=multimodal_data,
    output_path="./data/agi_dataset"
)
```

#### Robot Dataset Format | 机器人数据集格式
```python
from preprocessing.formats import RobotDatasetFormat

# Create robot dataset format
robot_format = RobotDatasetFormat(
    sensor_types=["rgb", "depth", "imu", "joint_state"],
    action_space="continuous",
    episode_structure=True
)

# Save robot data
robot_format.save_dataset(
    robot_data=robot_experiences,
    output_path="./data/robot_dataset"
)
```

## Data Quality Assurance | 数据质量保证

### Quality Checks | 质量检查

#### Data Validation | 数据验证
```python
from quality.data_validator import DataValidator

validator = DataValidator()

# Validate dataset
validation_results = validator.validate(
    dataset=dataset,
    checks=[
        "format_check",
        "completeness_check",
        "consistency_check",
        "duplicate_check"
    ]
)

# Get validation report
report = validator.generate_report(validation_results)
```

#### Statistical Analysis | 统计分析
```python
from quality.statistical_analyzer import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()

# Analyze dataset statistics
statistics = analyzer.analyze(
    dataset=dataset,
    metrics=[
        "class_distribution",
        "sequence_lengths",
        "feature_distributions",
        "correlations"
    ]
)

# Visualize statistics
analyzer.visualize_statistics(statistics)
```

### Bias Detection | 偏见检测
```python
from quality.bias_detector import BiasDetector

detector = BiasDetector()

# Detect biases in dataset
bias_report = detector.detect(
    dataset=dataset,
    bias_types=["demographic", "linguistic", "cultural", "representation"]
)

# Mitigate biases
mitigated_dataset = detector.mitigate(
    dataset=dataset,
    bias_report=bias_report,
    mitigation_strategy="rebalancing"
)
```

### Annotation Quality | 标注质量

#### Inter-annotator Agreement | 标注者间一致性
```python
from quality.annotation_quality import AnnotationQuality

quality_checker = AnnotationQuality()

# Check annotation quality
agreement_scores = quality_checker.calculate_agreement(
    annotations=annotations,
    annotators=["annotator_1", "annotator_2", "annotator_3"],
    metric="cohens_kappa"  # or "fleiss_kappa", "krippendorff_alpha"
)
```

#### Annotation Verification | 标注验证
```python
from quality.annotation_quality import AnnotationVerifier

verifier = AnnotationVerifier()

# Verify annotations
verification_results = verifier.verify(
    dataset=dataset,
    ground_truth=ground_truth,
    verification_sample=0.1  # Verify 10% of annotations
)

# Correct annotations
corrected_dataset = verifier.correct(
    dataset=dataset,
    verification_results=verification_results
)
```

## Dataset Splitting | 数据集划分

### Split Strategies | 划分策略

#### Random Split | 随机划分
```python
from splitting.dataset_splitter import DatasetSplitter

splitter = DatasetSplitter()

# Random split
train_set, val_set, test_set = splitter.random_split(
    dataset=dataset,
    ratios=[0.8, 0.1, 0.1],  # train, val, test
    seed=42
)
```

#### Stratified Split | 分层划分
```python
# Stratified split by class
train_set, val_set, test_set = splitter.stratified_split(
    dataset=dataset,
    strata_column="label",
    ratios=[0.8, 0.1, 0.1],
    seed=42
)
```

#### Temporal Split | 时间划分
```python
# Temporal split for time-series data
train_set, val_set, test_set = splitter.temporal_split(
    dataset=dataset,
    time_column="timestamp",
    split_points=["2024-01-01", "2024-06-01"]  # Train before, val between, test after
)
```

### Cross-Validation | 交叉验证
```python
from splitting.cross_validation import CrossValidator

validator = CrossValidator()

# K-fold cross-validation
folds = validator.k_fold_split(
    dataset=dataset,
    k=5,
    shuffle=True,
    seed=42
)

# Stratified K-fold
stratified_folds = validator.stratified_k_fold_split(
    dataset=dataset,
    k=5,
    strata_column="label",
    seed=42
)
```

## Dataset Versioning | 数据集版本控制

### Version Control | 版本控制
```python
from versioning.dataset_versioning import DatasetVersionManager

version_manager = DatasetVersionManager(
    storage_backend="s3",  # or "local", "gcs", "azure"
    version_schema="semantic"  # or "timestamp", "hash"
)

# Create new version
new_version = version_manager.create_version(
    dataset=dataset,
    version_name="v1.2.0",
    changelog="Added new multimodal examples",
    tags=["multimodal", "high_quality"]
)

# List versions
versions = version_manager.list_versions()

# Load specific version
loaded_dataset = version_manager.load_version("v1.1.0")
```

### Dataset Cards | 数据集卡片
```python
from versioning.dataset_cards import DatasetCardGenerator

generator = DatasetCardGenerator()

# Generate dataset card
dataset_card = generator.generate_card(
    dataset=dataset,
    card_template="standard",  # or "detailed", "minimal"
    include_statistics=True,
    include_license=True,
    include_citation=True
)

# Save dataset card
generator.save_card(dataset_card, "./dataset_cards/agi_dataset_card.md")
```

## Dataset Storage | 数据集存储

### Storage Options | 存储选项

#### Local Storage | 本地存储
```python
from storage.local_storage import LocalDatasetStorage

local_storage = LocalDatasetStorage(
    base_path="./datasets",
    compression="zstd",
    encryption=False
)

# Save dataset locally
local_storage.save(
    dataset=dataset,
    dataset_name="agi_multimodal",
    version="v1.0.0"
)
```

#### Cloud Storage | 云存储
```python
from storage.cloud_storage import CloudDatasetStorage

cloud_storage = CloudDatasetStorage(
    provider="aws",  # or "gcp", "azure"
    bucket_name="agi-datasets",
    region="us-east-1"
)

# Upload dataset to cloud
cloud_storage.upload(
    dataset=dataset,
    dataset_name="agi_multimodal",
    version="v1.0.0",
    make_public=False
)
```

#### Distributed Storage | 分布式存储
```python
from storage.distributed_storage import DistributedDatasetStorage

distributed_storage = DistributedDatasetStorage(
    storage_nodes=["node1:9000", "node2:9000", "node3:9000"],
    replication_factor=3,
    consistency_level="strong"
)

# Store dataset distributedly
distributed_storage.store(
    dataset=dataset,
    dataset_name="agi_multimodal",
    shard_size=1000  # Examples per shard
)
```

## Best Practices | 最佳实践

### Data Preparation Best Practices | 数据准备最佳实践
1. **Start with Quality**: Begin with high-quality, well-annotated data
2. **Document Everything**: Document data sources, preprocessing steps, and decisions
3. **Maintain Version Control**: Use version control for datasets and preprocessing code
4. **Ensure Reproducibility**: Make data preparation pipeline reproducible
5. **Consider Ethics**: Consider ethical implications of data collection and use
6. **Plan for Scale**: Design pipelines that can scale with dataset size
7. **Validate Continuously**: Continuously validate data quality throughout pipeline
8. **Backup Regularly**: Regularly backup datasets and preprocessing artifacts

1. **从质量开始**: 从高质量、标注良好的数据开始
2. **记录一切**: 记录数据来源、预处理步骤和决策
3. **维护版本控制**: 对数据集和预处理代码使用版本控制
4. **确保可复现性**: 使数据准备流程可复现
5. **考虑伦理**: 考虑数据收集和使用的伦理影响
6. **规划可扩展性**: 设计可随数据集大小扩展的流程
7. **持续验证**: 在整个流程中持续验证数据质量
8. **定期备份**: 定期备份数据集和预处理产物

### Multimodal Data Best Practices | 多模态数据最佳实践
1. **Align Modalities**: Ensure proper temporal and semantic alignment between modalities
2. **Balance Modalities**: Ensure balanced representation across all modalities
3. **Handle Missing Data**: Develop strategies for handling missing modalities
4. **Optimize Storage**: Use efficient storage formats for multimodal data
5. **Standardize Formats**: Use standardized formats for each modality
6. **Validate Cross-modal Consistency**: Ensure consistency between modalities

1. **对齐模态**: 确保模态间正确的时间和语义对齐
2. **平衡模态**: 确保所有模态的平衡表示
3. **处理缺失数据**: 制定处理缺失模态的策略
4. **优化存储**: 对多模态数据使用高效的存储格式
5. **标准化格式**: 对每种模态使用标准化格式
6. **验证跨模态一致性**: 确保模态间的一致性

## Next Steps | 后续步骤

After preparing datasets:

数据集准备完成后：

1. **Train Models**: Use prepared datasets to train AGI models
2. **Evaluate Data Quality**: Evaluate impact of data quality on model performance
3. **Iterate**: Iterate on data collection and preparation based on model performance
4. **Share Datasets**: Share datasets with the community (if permitted)
5. **Maintain Datasets**: Maintain and update datasets over time

1. **训练模型**: 使用准备好的数据集训练 AGI 模型
2. **评估数据质量**: 评估数据质量对模型性能的影响
3. **迭代**: 基于模型性能迭代数据收集和准备
4. **共享数据集**: 与社区共享数据集（如果允许）
5. **维护数据集**: 随时间维护和更新数据集

---

*Last Updated: March 30, 2026*  
*最后更新: 2026年3月30日*