# 语音合成模块 / Speech Synthesis Module

## 概述 / Overview

### 中文
语音合成模块提供从零开始实现的文本转语音（TTS）功能，包含文本编码、声学模型、声码器等完整组件。不依赖任何预训练模型，所有功能均为自主实现。

### English
The Speech Synthesis module provides text-to-speech (TTS) functionality implemented from scratch, including complete components such as text encoding, acoustic model, and vocoder. No pre-trained models are used, all functionality is independently implemented.

---

## 核心功能 / Core Features

### 中文
1. **文本编码**：字符级和音素级文本编码
2. **声学模型**：基于Transformer的梅尔频谱图生成
3. **声码器**：从零开始的声码器实现
4. **多语言支持**：中文、英文、数字、标点符号合成
5. **可训练**：完整的训练接口，从零开始训练
6. **语音控制**：语速、音调、音量调节

### English
1. **Text Encoding**: Character-level and phoneme-level text encoding
2. **Acoustic Model**: Transformer-based Mel spectrogram generation
3. **Vocoder**: Vocoder implementation from scratch
4. **Multilingual Support**: Chinese, English, numbers, punctuation synthesis
5. **Trainable**: Complete training interface, train from scratch
6. **Speech Control**: Speed, pitch, volume adjustment

---

## 核心类 / Core Classes

### SpeechSynthesisConfig
```python
@dataclass
class SpeechSynthesisConfig:
    # 音频参数 / Audio parameters
    sample_rate: int = 22050
    frame_length: int = 50  # 毫秒 / ms
    frame_shift: int = 12.5 # 毫秒 / ms
    num_mels: int = 80
    fft_size: int = 1024
    
    # 模型参数 / Model parameters
    encoder_dim: int = 512
    encoder_layers: int = 6
    encoder_heads: int = 8
    decoder_dim: int = 512
    decoder_layers: int = 6
    decoder_heads: int = 8
    
    # 词汇表 / Vocabulary
    vocab_size: int = 5000  # 中文字符 + 英文字母 + 数字 + 标点
    
    # 声码器参数 / Vocoder parameters
    vocoder_layers: int = 30
    vocoder_channels: int = 512
    
    # 训练参数 / Training parameters
    dropout_rate: float = 0.1
    learning_rate: float = 1e-4
```

### TextEncoder
```python
class TextEncoder(nn.Module):
    """文本编码器 - 将文本转换为特征表示"""
```
- 字符嵌入 / Character embedding
- 位置编码 / Positional encoding
- Transformer编码器 / Transformer encoder
- 自注意力机制 / Self-attention mechanism

### AcousticDecoder
```python
class AcousticDecoder(nn.Module):
    """声学解码器 - 生成梅尔频谱图"""
```
- 文本到声学特征映射 / Text to acoustic feature mapping
- 自回归生成 / Autoregressive generation
- 注意力机制 / Attention mechanism
- 持续时间预测 / Duration prediction

### Vocoder
```python
class Vocoder(nn.Module):
    """声码器 - 将梅尔频谱图转换为音频波形"""
```
- 上采样网络 / Upsampling network
- 波形生成 / Waveform generation
- 后处理 / Post-processing

---

## 主要方法 / Main Methods

### 文本编码 / Text Encoding

#### 编码文本 / Encode Text
```python
def encode(text: str) -> torch.Tensor
```
将文本编码为特征表示。

Encode text into feature representation.

### 声学解码 / Acoustic Decoding

#### 生成梅尔频谱图 / Generate Mel Spectrogram
```python
def decode(text_features: torch.Tensor) -> torch.Tensor
```
从文本特征生成梅尔频谱图。

Generate Mel spectrogram from text features.

### 声码器 / Vocoder

#### 合成音频 / Synthesize Audio
```python
def synthesize(mel_spec: torch.Tensor) -> torch.Tensor
```
从梅尔频谱图合成音频波形。

Synthesize audio waveform from Mel spectrogram.

### 端到端合成 / End-to-End Synthesis

#### 语音合成 / Speech Synthesis
```python
def synthesize_speech(text: str) -> torch.Tensor
```
端到端语音合成，从文本到音频。

End-to-end speech synthesis from text to audio.

---

## 使用示例 / Usage Examples

### 中文
```python
from models.multimodal.speech_synthesis import (
    SpeechSynthesisConfig,
    TextEncoder,
    AcousticDecoder,
    Vocoder
)

# 创建配置 / Create config
config = SpeechSynthesisConfig(
    sample_rate=22050,
    num_mels=80,
    encoder_dim=512,
    vocab_size=5000
)

# 创建组件 / Create components
text_encoder = TextEncoder(config)
acoustic_decoder = AcousticDecoder(config)
vocoder = Vocoder(config)

# 文本合成 / Text synthesis
# text = "你好，世界！"
# text_features = text_encoder.encode(text)
# mel_spec = acoustic_decoder.decode(text_features)
# audio = vocoder.synthesize(mel_spec)

# 保存音频 / Save audio
# import soundfile as sf
# sf.write('output.wav', audio.numpy(), config.sample_rate)
```

### English
```python
from models.multimodal.speech_synthesis import (
    SpeechSynthesisConfig,
    TextEncoder,
    AcousticDecoder,
    Vocoder
)

# Create config
config = SpeechSynthesisConfig(
    sample_rate=22050,
    num_mels=80,
    encoder_dim=512,
    vocab_size=5000
)

# Create components
text_encoder = TextEncoder(config)
acoustic_decoder = AcousticDecoder(config)
vocoder = Vocoder(config)

# Text synthesis
# text = "Hello, World!"
# text_features = text_encoder.encode(text)
# mel_spec = acoustic_decoder.decode(text_features)
# audio = vocoder.synthesize(mel_spec)

# Save audio
# import soundfile as sf
# sf.write('output.wav', audio.numpy(), config.sample_rate)
```

---

## 相关模块 / Related Modules

- [语音识别](./speech-recognition.md) - 语音识别
- [多模态融合](./fusion-networks.md) - 多模态融合
- [音频编码](./audio-encoder.md) - 音频编码
