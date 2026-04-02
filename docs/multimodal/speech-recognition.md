# 语音识别模块 / Speech Recognition Module

## 概述 / Overview

### 中文
语音识别模块提供从零开始实现的工业级语音识别功能，包含语音特征提取、声学建模、CTC解码、语言模型集成等完整组件。不依赖任何预训练模型，所有功能均为自主实现。

### English
The Speech Recognition module provides industrial-grade speech recognition functionality implemented from scratch, including complete components such as speech feature extraction, acoustic modeling, CTC decoding, and language model integration. No pre-trained models are used, all functionality is independently implemented.

---

## 核心功能 / Core Features

### 中文
1. **语音特征提取**：MFCC、梅尔频谱图、预加重、分帧加窗
2. **声学模型**：基于Transformer的声学编码器
3. **CTC解码**：连接时序分类解码，支持束搜索
4. **语言模型**：n-gram语言模型重打分
5. **多语言支持**：中文、英文、数字、标点符号识别
6. **可训练**：完整的训练接口，从零开始训练

### English
1. **Speech Feature Extraction**: MFCC, Mel spectrogram, pre-emphasis, framing and windowing
2. **Acoustic Model**: Transformer-based acoustic encoder
3. **CTC Decoding**: Connectionist Temporal Classification decoding with beam search support
4. **Language Model**: n-gram language model rescoring
5. **Multilingual Support**: Chinese, English, numbers, punctuation recognition
6. **Trainable**: Complete training interface, train from scratch

---

## 核心类 / Core Classes

### SpeechRecognitionConfig
```python
@dataclass
class SpeechRecognitionConfig:
    # 音频参数 / Audio parameters
    sample_rate: int = 16000
    frame_length: int = 25  # 毫秒 / ms
    frame_shift: int = 10   # 毫秒 / ms
    num_mels: int = 80
    
    # 模型参数 / Model parameters
    encoder_dim: int = 512
    encoder_layers: int = 12
    encoder_heads: int = 8
    decoder_dim: int = 512
    decoder_layers: int = 6
    
    # 词汇表 / Vocabulary
    vocab_size: int = 5000  # 中文字符 + 英文字母 + 数字 + 标点
    blank_id: int = 0
    
    # 训练参数 / Training parameters
    dropout_rate: float = 0.1
    ctc_weight: float = 0.5
    lm_weight: float = 0.3
    beam_width: int = 10
    
    # 频谱图参数 / Spectrogram parameters
    spectrogram_size: int = 128
    patch_size: int = 16
```

### SpeechFeatureExtractor
```python
class SpeechFeatureExtractor(nn.Module):
    """语音特征提取器 - 从零开始的MFCC/频谱图提取"""
```
- 预加重滤波器 / Pre-emphasis filter
- 汉明窗 / Hamming window
- FFT变换 / FFT transformation
- 梅尔滤波器组 / Mel filter bank
- 对数压缩 / Logarithmic compression

### AcousticEncoder
```python
class AcousticEncoder(nn.Module):
    """声学编码器 - 基于Transformer的音频特征编码"""
```
- 输入投影层 / Input projection layer
- 位置编码 / Positional encoding
- Transformer编码器层 / Transformer encoder layers
- 自注意力机制 / Self-attention mechanism

### CTCDecoder
```python
class CTCDecoder:
    """CTC解码器 - 连接时序分类解码"""
```
- 贪婪解码 / Greedy decoding
- 束搜索解码 / Beam search decoding
- 语言模型重打分 / Language model rescoring
- 空白标签处理 / Blank label handling

### LanguageModel
```python
class LanguageModel:
    """n-gram语言模型 - 用于解码重打分"""
```
- n-gram统计 / n-gram statistics
- 概率计算 / Probability calculation
- 平滑处理 / Smoothing

---

## 主要方法 / Main Methods

### 特征提取 / Feature Extraction

#### 提取梅尔频谱图 / Extract Mel Spectrogram
```python
def forward(audio: torch.Tensor) -> torch.Tensor
```
从原始音频波形提取梅尔频谱图特征。

Extract Mel spectrogram features from raw audio waveform.

### 声学编码 / Acoustic Encoding

#### 编码音频特征 / Encode Audio Features
```python
def encode(mel_spec: torch.Tensor) -> torch.Tensor
```
将梅尔频谱图编码为声学特征。

Encode Mel spectrogram into acoustic features.

### CTC解码 / CTC Decoding

#### 贪婪解码 / Greedy Decode
```python
def greedy_decode(logits: torch.Tensor) -> List[int]
```
使用贪婪算法解码CTC输出。

Decode CTC output using greedy algorithm.

#### 束搜索解码 / Beam Search Decode
```python
def beam_search_decode(logits: torch.Tensor, beam_width: int = 10) -> List[int]
```
使用束搜索算法解码CTC输出。

Decode CTC output using beam search algorithm.

### 端到端识别 / End-to-End Recognition

#### 语音识别 / Speech Recognition
```python
def recognize(audio: torch.Tensor) -> str
```
端到端语音识别，从音频到文本。

End-to-end speech recognition from audio to text.

---

## 使用示例 / Usage Examples

### 中文
```python
from models.multimodal.speech_recognition import (
    SpeechRecognitionConfig,
    SpeechFeatureExtractor,
    AcousticEncoder,
    CTCDecoder
)

# 创建配置 / Create config
config = SpeechRecognitionConfig(
    sample_rate=16000,
    num_mels=80,
    encoder_dim=512,
    vocab_size=5000
)

# 创建组件 / Create components
feature_extractor = SpeechFeatureExtractor(config)
acoustic_encoder = AcousticEncoder(config)
decoder = CTCDecoder(config)

# 假设有音频数据 / Assume we have audio data
# audio = torch.randn(1, 16000)  # 1秒音频 / 1 second audio

# 特征提取 / Feature extraction
# mel_spec = feature_extractor(audio)

# 声学编码 / Acoustic encoding
# acoustic_features = acoustic_encoder(mel_spec)

# CTC解码 / CTC decoding
# text = decoder.greedy_decode(acoustic_features)

# print(f"识别结果: {text}")
```

### English
```python
from models.multimodal.speech_recognition import (
    SpeechRecognitionConfig,
    SpeechFeatureExtractor,
    AcousticEncoder,
    CTCDecoder
)

# Create config
config = SpeechRecognitionConfig(
    sample_rate=16000,
    num_mels=80,
    encoder_dim=512,
    vocab_size=5000
)

# Create components
feature_extractor = SpeechFeatureExtractor(config)
acoustic_encoder = AcousticEncoder(config)
decoder = CTCDecoder(config)

# Assume we have audio data
# audio = torch.randn(1, 16000)  # 1 second audio

# Feature extraction
# mel_spec = feature_extractor(audio)

# Acoustic encoding
# acoustic_features = acoustic_encoder(mel_spec)

# CTC decoding
# text = decoder.greedy_decode(acoustic_features)

# print(f"Recognition result: {text}")
```

---

## 相关模块 / Related Modules

- [语音合成](./speech-synthesis.md) - 语音合成
- [多模态融合](./fusion-networks.md) - 多模态融合
- [视觉编码](./vision-encoder.md) - 视觉编码
