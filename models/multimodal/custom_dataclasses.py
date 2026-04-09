#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态处理器基础数据结构模块

包含：
1. MultimodalInput - 多模态输入数据类
2. ProcessedModality - 处理后的模态数据类

工业级AGI系统要求：从零开始设计，不使用预训练模型依赖
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class MultimodalInput:
    """多模态输入数据"""

    text: Optional[str] = None
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    audio_path: Optional[str] = None
    audio_base64: Optional[str] = None
    video_path: Optional[str] = None
    video_base64: Optional[str] = None
    sensor_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ProcessedModality:
    """处理后的模态数据"""

    modality_type: str  # text, image, audio, video, sensor
    embeddings: List[float]
    features: Dict[str, Any]
    confidence: float
    metadata: Dict[str, Any]
