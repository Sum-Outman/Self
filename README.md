# Self AGI System - 自主通用人工智能系统
# Self AGI System - Autonomous General Intelligence System

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.0-blue.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18-blue.svg)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue.svg)](https://www.typescriptlang.org/)

## 概述 | Overview

**Self AGI** 是一个从零开始设计的自主通用人工智能系统，采用统一的 Transformer 架构。系统提供完整的长期和短期记忆管理、多模态处理能力、人形机器人控制、知识库系统以及自主学习功能。

**Self AGI** is an autonomous general intelligence system designed from scratch with a unified Transformer architecture. It provides complete long-term and short-term memory management, multimodal processing capabilities, humanoid robot control, knowledge base systems, and autonomous learning functionality.

本系统中模型为未训练模型需训练后使用。可以根据自己需求和已有资源自主进行部分开发直接移植替换接入预训练模型、ollama本地模型、外部API使用。推荐使用本地未训练模型进行多模态训练使用才能具备完善完整的多模态神经反馈系统。（预训练模型版本、ollama版本、外接API版本，将在后续推出）

The models in this system are untrained models and need to be trained before use. You can independently develop and directly transplant or replace to access pre-trained models, ollama local models, or external APIs according to your needs and existing resources. It is recommended to use local untrained models for multimodal training to have a complete multimodal neural feedback system. (Pre-trained model version, ollama version, external API version will be launched in the future)

本项目的核心特点 | Core Features:
- 统一的模型架构，集成多种能力模块 | Unified model architecture with integrated capability modules
- 支持 GPU 和 CPU 运行 | Supports both GPU and CPU execution
- 完整的拉普拉斯图学习机制集成 | Complete Laplacian graph learning mechanism integration
- 人形机器人控制和仿真能力 | Humanoid robot control and simulation capabilities
- 多模态数据处理和融合 | Multimodal data processing and fusion
- 自我演化、自我修正、自学习能力 | Self-evolution, self-correction, and self-learning capabilities
- AttnRes动态深度聚合注意力机制 | AttnRes dynamic depthwise aggregation attention mechanism
- 四元数空间感知增强 | Quaternion spatial perception enhancement
- 混合专家系统 | Mixture of Experts system
- 状态空间模型 | State space models

## 核心特色功能 | Core Special Features

### 1. 自我学习功能 | Self-Learning Capability

系统实现了完整的自主学习管理器，具备以下核心能力 | The system implements a complete autonomous learning manager with the following core capabilities:

#### 知识缺口检测 | Knowledge Gap Detection
- 基于任务失败分析检测知识缺口 | Detect knowledge gaps based on task failure analysis
- 能力自我评估 | Capability self-assessment
- 外部需求分析 | External requirement analysis
- 知识图谱完整性检查 | Knowledge graph completeness check

#### 学习目标自动生成 | Automatic Learning Goal Generation
- 智能学习目标设定 | Intelligent learning goal setting
- 优先级排序 | Priority ranking
- 先决条件分析 | Prerequisite analysis
- 学习时间估算 | Learning time estimation

#### 学习资源发现 | Learning Resource Discovery
- 自动发现和收集学习材料 | Automatic discovery and collection of learning materials
- 资源缓存管理 | Resource cache management
- 多类型资源支持（理论、实践、案例、评估）| Multi-type resource support (theory, practice, case studies, assessment)

#### 学习进度监控 | Learning Progress Monitoring
- 实时跟踪学习进度 | Real-time learning progress tracking
- 效果评估 | Effectiveness evaluation
- 掌握程度计算 | Mastery level calculation
- 性能指标统计 | Performance metric statistics

#### 自我改进循环 | Self-Improvement Loop
- 基于反馈的学习策略优化 | Feedback-based learning strategy optimization
- 学习效果验证 | Learning effect verification
- 策略自适应调整 | Strategy adaptive adjustment

#### 知识整合 | Knowledge Integration
- 将新知识整合到现有知识体系中 | Integrate new knowledge into existing knowledge systems
- 知识库更新 | Knowledge base updates
- 记忆系统同步 | Memory system synchronization
- 概念关联建立 | Concept association establishment

### 2. 自我修正功能 | Self-Correction Capability

系统集成了完整的自我修正模块，包含以下核心组件 | The system integrates a complete self-correction module with the following core components:

#### 错误检测子系统 | Error Detection Subsystem
- 模式匹配器 | Pattern matcher
- 一致性检查器 | Consistency checker
- 规则验证器 | Rule validator
- 错误分类器 | Error classifier
- 严重性评估器 | Severity evaluator

#### 原因分析子系统 | Cause Analysis Subsystem
- 因果分析器 | Causal analyzer
- 故障树分析器 | Fault tree analyzer
- 根因分析器 | Root cause analyzer
- 影响分析器 | Impact analyzer

#### 改正生成子系统 | Correction Generation Subsystem
- 知识检索器 | Knowledge retriever
- 改正规则应用器 | Correction rule applier
- 优化生成器 | Optimization generator
- 策略选择器 | Strategy selector
- 改正生成器 | Correction generator

#### 验证子系统 | Verification Subsystem
- 有效性检查器 | Effectiveness checker
- 一致性验证器 | Consistency verifier
- 测试模拟器 | Test simulator
- 回归测试器 | Regression tester
- 验证网络 | Verification network

### 3. 自我演化功能 | Self-Evolution Capability

系统具备自主演化能力，支持 | The system has autonomous evolution capability, supporting:

- 神经架构搜索(NAS) | Neural Architecture Search (NAS)
- 参数优化 | Parameter optimization
- 适应度评估 | Fitness evaluation
- 进化策略 | Evolutionary strategies
- 动态架构调整 | Dynamic architecture adjustment
- 架构搜索集成 | Architecture search integration

### 4. AttnRes - 动态深度聚合注意力机制 | AttnRes - Dynamic Depthwise Aggregation Attention Mechanism

AttnRes是基于Kimi 2026年3月16日论文《Attention Residuals: Dynamic Depthwise Aggregation for Large Language Models》的最新技术集成 | AttnRes is the latest technology integration based on Kimi's March 16, 2026 paper "Attention Residuals: Dynamic Depthwise Aggregation for Large Language Models"

#### 核心特性 | Core Features
- 动态聚合层 | Dynamic aggregation layer
- 残差缩放参数 | Residual scale parameter
- 注意力缩放参数 | Attention scale parameter
- 深度聚合机制 | Depthwise aggregation mechanism
- 可配置启用/禁用 | Configurable enable/disable

#### 技术优势 | Technical Advantages
- 提升注意力机制效率 | Improve attention mechanism efficiency
- 增强长程依赖建模 | Enhance long-range dependency modeling
- 改善梯度流动 | Improve gradient flow
- 支持灵活配置 | Support flexible configuration

### 5. 四元数集成 | Quaternion Integration

系统全面引入四元数表示和运算，显著提升训练效率、模型计算能力、多模态能力和AGI机器人能力 | The system fully introduces quaternion representation and operations, significantly improving training efficiency, model computing capability, multimodal capability, and AGI robot capability

#### 四元数核心功能 | Quaternion Core Features
- 完整的四元数数据结构（NumPy 和 PyTorch 双版本）| Complete quaternion data structure (NumPy and PyTorch dual versions)
- 四元数基本运算（乘法、共轭、归一化、点积、SLERP 等）| Quaternion basic operations (multiplication, conjugate, normalization, dot product, SLERP, etc.)
- 四元数转换函数（欧拉角、旋转矩阵、轴角表示）| Quaternion conversion functions (Euler angles, rotation matrix, axis-angle representation)
- 四元数距离度量和损失函数 | Quaternion distance metrics and loss functions

#### 四元数神经网络层 | Quaternion Neural Network Layers
- QuaternionLinear：四元数线性层，支持复数权重 | Quaternion linear layer supporting complex weights
- QuaternionAttention：四元数注意力机制，增强空间感知 | Quaternion attention mechanism enhancing spatial perception
- QuaternionTransformerBlock：四元数 Transformer 块，集成四元数运算 | Quaternion Transformer block integrating quaternion operations
- QuaternionConv1D：四元数卷积层，处理时序数据 | Quaternion convolutional layer for sequential data
- QuaternionEmbedding：四元数嵌入层，用于旋转表示 | Quaternion embedding layer for rotation representation

#### 四元数集成优势 | Quaternion Integration Advantages
- 训练效率提升：四元数表示减少参数冗余，加快收敛速度 | Training efficiency improvement: Quaternion representation reduces parameter redundancy and accelerates convergence
- 避免万向节锁，提高旋转学习的稳定性 | Avoiding gimbal lock improves stability of rotation learning
- 模型计算能力增强：四元数神经网络层提高空间推理能力 | Model computing capability enhancement: Quaternion neural network layers improve spatial reasoning capabilities
- 四元数注意力机制增强长期依赖建模 | Quaternion attention mechanism enhances long-range dependency modeling

### 6. 混合专家系统 (MoE) | Mixture of Experts System (MoE)

基于最新MoE研究的高效实现 | Efficient implementation based on latest MoE research

#### 关键特性 | Key Features
- 稀疏激活：每个token只激活少数专家 | Sparse activation: Each token only activates a few experts
- 负载平衡：确保专家使用均衡 | Load balancing: Ensure balanced expert usage
- 容量因子：处理专家容量限制 | Capacity factor: Handle expert capacity limits
- 可扩展性：支持大量专家 | Scalability: Support large number of experts

#### 路由器类型 | Router Types
- Top-k选择 | Top-k selection
- 带噪声的Top-k | Noisy Top-k
- 学习路由器 | Learned router

### 7. 状态空间模型 | State Space Models

基于Mamba和RetNet架构的高效序列建模 | Efficient sequence modeling based on Mamba and RetNet architectures

#### 核心特性 | Core Features
- 选择性状态空间：输入依赖的状态转移 | Selective state space: Input-dependent state transitions
- 高效扫描算法：线性时间复杂度的序列建模 | Efficient scanning algorithm: Linear time complexity sequence modeling
- 硬件感知设计：高效GPU实现 | Hardware-aware design: Efficient GPU implementation
- 长程依赖：处理超长序列 | Long-range dependency: Process very long sequences

#### RetNet保留机制 | RetNet Retention Mechanism
- 并行形式训练 | Parallel form training
- 递归形式推理 | Recursive form inference
- 衰减机制 | Decay mechanism
- 门控函数 | Gating function

### 8. 模仿学习能力 | Imitation Learning Capability

系统集成了完整的模仿学习框架，支持从人类演示中学习复杂技能 | The system integrates a complete imitation learning framework, supporting learning complex skills from human demonstrations

#### 核心功能 | Core Features
- **行为克隆 | Behavior Cloning**: 直接从演示数据学习策略映射 | Directly learn policy mappings from demonstration data
- **逆强化学习 | Inverse Reinforcement Learning (IRL)**: 从演示中推断奖励函数 | Infer reward functions from demonstrations
- **生成对抗模仿学习 | Generative Adversarial Imitation Learning (GAIL)**: 通过对抗训练学习策略 | Learn policies through adversarial training
- **从演示中学习 | Learning from Demonstration (LfD)**: 完整的演示学习管道 | Complete demonstration learning pipeline
- **多模态演示处理 | Multimodal Demonstration Processing**: 支持视频、运动捕捉、遥操作等多种演示方式 | Supports video, motion capture, teleoperation, and other demonstration methods

#### 技术优势 | Technical Advantages
- **演示数据采集 | Demonstration Data Collection**: 统一的演示数据采集接口，支持多种输入设备 | Unified demonstration data collection interface supporting multiple input devices
- **轨迹对齐 | Trajectory Alignment**: 动态时间规整(DTW)等算法实现演示轨迹对齐 | Dynamic Time Warping (DTW) and other algorithms for demonstration trajectory alignment
- **技能泛化 | Skill Generalization**: 支持在新环境中泛化学习到的技能 | Supports generalization of learned skills to new environments
- **增量学习 | Incremental Learning**: 支持从多个演示中持续学习和改进技能 | Supports continuous learning and improvement from multiple demonstrations

### 9. 真实世界交互能力 | Real-World Interaction Capability

系统具备完整的真实世界感知、决策和执行能力，支持与物理环境的无缝交互 | The system has complete real-world perception, decision-making, and execution capabilities, supporting seamless interaction with the physical environment

#### 感知系统 | Perception System
- **多模态传感器融合 | Multimodal Sensor Fusion**: 融合视觉、听觉、触觉、力觉等多种传感器数据 | Fuses visual, auditory, tactile, force, and other sensor data
- **环境建模 | Environment Modeling**: 实时构建3D环境地图，支持SLAM和场景理解 | Real-time 3D environment mapping with SLAM and scene understanding
- **物体识别与跟踪 | Object Recognition and Tracking**: 识别、定位和跟踪真实世界中的物体 | Recognize, localize, and track objects in the real world
- **人体姿态估计 | Human Pose Estimation**: 理解人类动作和意图 | Understand human actions and intentions

#### 决策与规划 | Decision and Planning
- **任务级规划 | Task-Level Planning**: 高层任务分解和执行规划 | High-level task decomposition and execution planning
- **运动规划 | Motion Planning**: 无碰撞路径规划和轨迹优化 | Collision-free path planning and trajectory optimization
- **反应式控制 | Reactive Control**: 实时环境变化响应和应急处理 | Real-time environmental change response and emergency handling
- **风险评估 | Risk Assessment**: 评估操作安全性和潜在风险 | Evaluate operational safety and potential risks

#### 执行系统 | Execution System
- **硬件抽象层 | Hardware Abstraction Layer**: 统一的硬件接口，支持多种机器人平台 | Unified hardware interface supporting multiple robot platforms
- **力控制 | Force Control**: 精细的力反馈控制，支持柔性交互 | Fine force feedback control for compliant interaction
- **安全机制 | Safety Mechanisms**: 紧急停止、碰撞检测、限力保护等安全功能 | Emergency stop, collision detection, force limiting, and other safety features
- **实时通信 | Real-Time Communication**: 低延迟的硬件控制和数据传输 | Low-latency hardware control and data transmission

## 系统功能 | System Features

### 1. 核心模型架构 | Core Model Architecture

- **Transformer 核心 | Transformer Core**: 12 层 Transformer 架构，768 隐藏维度，12 个注意力头 | 12-layer Transformer architecture, 768 hidden dimensions, 12 attention heads
- **记忆系统 | Memory System**: 长期和短期记忆管理，支持记忆重要性学习和压缩检索 | Long-term and short-term memory management with importance learning and compression retrieval
- **多模态融合 | Multimodal Fusion**: 文本、图像、音频、视频、传感器数据的统一处理 | Unified processing of text, image, audio, video, and sensor data
- **推理引擎 | Reasoning Engine**: 逻辑推理、数学推理、因果推理等能力 | Logical reasoning, mathematical reasoning, causal reasoning capabilities

### 2. 拉普拉斯图学习机制 | Laplacian Graph Learning Mechanism

系统集成了完整的拉普拉斯图学习框架 | The system integrates a complete Laplacian graph learning framework:

- **图拉普拉斯矩阵计算 | Graph Laplacian Matrix Computation**: 支持无向图和有向图的拉普拉斯矩阵计算 | Supports Laplacian matrix computation for undirected and directed graphs
- **标准化拉普拉斯 | Normalized Laplacian**: 对称标准化和随机游走标准化 | Symmetric normalization and random walk normalization
- **稀疏矩阵优化 | Sparse Matrix Optimization**: 大规模图的高效存储和计算 | Efficient storage and computation for large-scale graphs
- **特征值分解 | Eigenvalue Decomposition**: 拉普拉斯矩阵的特征值和特征向量计算 | Eigenvalue and eigenvector computation for Laplacian matrices
- **拉普拉斯正则化 | Laplacian Regularization**: 图拉普拉斯正则化、流形正则化、半监督学习 | Graph Laplacian regularization, manifold regularization, semi-supervised learning
- **拉普拉斯增强训练 | Laplacian-Enhanced Training**: 与主训练框架的无缝集成 | Seamless integration with the main training framework

### 3. 机器人 AGI 功能 | Robot AGI Capabilities

系统提供完整的机器人控制和仿真能力 | The system provides complete robot control and simulation capabilities:

- **硬件抽象层 | Hardware Abstraction Layer**: 统一的硬件接口，支持仿真和真实硬件 | Unified hardware interface supporting both simulation and real hardware
- **人形机器人控制 | Humanoid Robot Control**: 关节位置、速度、扭矩控制 | Joint position, velocity, torque control
- **仿真环境 | Simulation Environments**: PyBullet 和 Gazebo 仿真支持 | PyBullet and Gazebo simulation support
- **传感器集成 | Sensor Integration**: IMU、相机、激光雷达、力扭矩传感器等 | IMU, cameras, LiDAR, force-torque sensors, etc.
- **运动规划 | Motion Planning**: 路径规划、避障、轨迹生成 | Path planning, obstacle avoidance, trajectory generation
- **学习控制 | Learning Control**: 模仿学习、强化学习、从演示中学习 | Imitation learning, reinforcement learning, learning from demonstration
- **模仿学习集成 | Imitation Learning Integration**: 完整的行为克隆、IRL、GAIL等模仿学习算法集成 | Complete integration of behavior cloning, IRL, GAIL, and other imitation learning algorithms
- **真实世界感知 | Real-World Perception**: 多模态传感器融合、环境建模、物体识别跟踪 | Multimodal sensor fusion, environment modeling, object recognition and tracking
- **物理交互 | Physical Interaction**: 力控制、柔性操作、安全机制 | Force control, compliant manipulation, safety mechanisms
- **人机协作 | Human-Robot Collaboration**: 人体姿态估计、意图理解、安全协作 | Human pose estimation, intention understanding, safe collaboration

### 4. 知识库系统 | Knowledge Base System

- **知识存储 | Knowledge Storage**: 结构化和非结构化知识存储 | Structured and unstructured knowledge storage
- **语义搜索 | Semantic Search**: 基于向量的相似性搜索 | Vector-based similarity search
- **知识图谱 | Knowledge Graph**: 实体关系图构建和查询 | Entity-relationship graph construction and querying
- **知识验证 | Knowledge Validation**: 质量验证和一致性检查 | Quality verification and consistency checking
- **知识推理 | Knowledge Reasoning**: 基于知识图谱的逻辑推理 | Knowledge graph-based logical reasoning
- **知识类型 | Knowledge Types**: 事实、规则、过程、概念、关系、事件、问题解决方案、经验 | Facts, rules, procedures, concepts, relationships, events, problem solutions, experiences

### 5. 系统控制模块 | System Control Modules

- **文件系统管理器 | File System Manager**: 跨平台文件操作、磁盘管理、文件监控、压缩解压 | Cross-platform file operations, disk management, file monitoring, compression/decompression
- **进程管理器 | Process Manager**: 进程创建、监控、终止、资源限制、进程间通信 | Process creation, monitoring, termination, resource limits, inter-process communication
- **网络控制器 | Network Controller**: TCP/UDP/HTTP服务器、网络配置、网络监控诊断 | TCP/UDP/HTTP servers, network configuration, network monitoring and diagnostics
- **服务管理器 | Service Manager**: Windows/Linux/macOS服务管理、计划任务、启动项管理 | Windows/Linux/macOS service management, scheduled tasks, startup item management

### 6. 硬件控制系统 | Hardware Control System

- **内存管理器 | Memory Manager**: 内存分配、内存池、垃圾回收、内存泄漏检测 | Memory allocation, memory pools, garbage collection, memory leak detection
- **硬件管理器 | Hardware Manager**: 硬件设备枚举、状态监控、热插拔检测 | Hardware device enumeration, status monitoring, hot-plug detection
- **电机控制器 | Motor Controller**: 多轴电机控制、PID控制、运动规划、急停保护 | Multi-axis motor control, PID control, motion planning, emergency stop protection
- **传感器接口 | Sensor Interface**: IMU、摄像头、激光雷达、触觉传感器、数据融合 | IMU, cameras, LiDAR, tactile sensors, data fusion
- **串口控制器 | Serial Controller**: 跨平台串口通信、波特率配置、数据包收发 | Cross-platform serial communication, baud rate configuration, data packet transmission/reception
- **系统健康管理器 | System Health Manager**: 健康状态评估、异常检测、故障诊断 | Health status assessment, anomaly detection, fault diagnosis
- **系统监控器 | System Monitor**: CPU/内存/磁盘/网络监控、进程监控、温度监控 | CPU/memory/disk/network monitoring, process monitoring, temperature monitoring
- **自主模式管理器 | Autonomous Mode Manager**: 全自主模式、任务执行模式、模式切换管理 | Full autonomous mode, task execution mode, mode switching management

### 7. 多模态处理 | Multimodal Processing

- **语音识别 | Speech Recognition**: 从零开始的ASR、特征提取、声学模型、CTC解码 | ASR from scratch, feature extraction, acoustic model, CTC decoding
- **语音合成 | Speech Synthesis**: 从零开始的TTS、文本编码、声学模型、声码器 | TTS from scratch, text encoding, acoustic model, vocoder
- **视觉编码器 | Vision Encoder**: CNN/ViT、图像特征提取、目标检测、语义分割 | CNN/ViT, image feature extraction, object detection, semantic segmentation
- **音频编码器 | Audio Encoder**: 梅尔频谱图提取、音频特征编码、声音事件检测 | Mel spectrogram extraction, audio feature encoding, sound event detection
- **文本编码器 | Text Encoder**: 字符/词嵌入、位置编码、自注意力、上下文编码 | Character/word embedding, positional encoding, self-attention, contextual encoding
- **多模态融合网络 | Multimodal Fusion Networks**: 早期融合、晚期融合、协同注意力融合、门控融合 | Early fusion, late fusion, co-attention fusion, gated fusion
- **嵌入模块 | Embedding Module**: 从零开始的文本/图像/音频嵌入器 | Text/image/audio embedders from scratch

### 8. 高级功能 | Advanced Features

- **Transformer模型 | Transformer Model**: 12层Transformer架构、自注意力、多头注意力、位置编码 | 12-layer Transformer architecture, self-attention, multi-head attention, positional encoding
- **物理模拟 | Physics Simulation**: 基于PINN的物理模拟、机器人动力学、环境模拟、多物理场耦合 | PINN-based physics simulation, robot dynamics, environment simulation, multi-physics coupling
- **进化管理器 | Evolution Manager**: 神经架构搜索(NAS)、参数优化、适应度评估、进化策略 | Neural Architecture Search (NAS), parameter optimization, fitness evaluation, evolutionary strategies
- **机器人多模态控制 | Robot Multimodal Control**: 多模态数据同步、跨模态特征融合、多模态推理、感知-运动协调 | Multimodal data synchronization, cross-modal feature fusion, multimodal reasoning, sensor-motor coordination

### 9. 训练系统 | Training System

- **从零开始训练 | From-Scratch Training**: 完整的训练流程，从随机初始化开始 | Complete training pipeline starting from random initialization
- **分布式训练 | Distributed Training**: 多 GPU 和多节点训练支持 | Multi-GPU and multi-node training support
- **训练监控 | Training Monitoring**: 实时指标可视化和检查点管理 | Real-time metrics visualization and checkpoint management
- **超参数优化 | Hyperparameter Optimization**: 自动化超参数搜索 | Automated hyperparameter search

## 技术架构 | Technical Architecture

### 后端技术栈 | Backend Technology Stack
- **Web 框架 | Web Framework**: FastAPI 0.104.0 (Python 3.9+)
- **AI 框架 | AI Framework**: PyTorch 2.9.0+
- **数据库 | Database**: SQLite (开发 development), PostgreSQL (生产可选 production optional)
- **缓存 | Cache**: Redis (可选 optional)
- **向量搜索 | Vector Search**: FAISS

### 前端技术栈 | Frontend Technology Stack
- **框架 | Framework**: React 18 + TypeScript 5.0
- **构建工具 | Build Tool**: Vite 5.0
- **样式 | Styling**: Tailwind CSS 3.4
- **状态管理 | State Management**: Zustand

### 模型技术 | Model Technology
- **核心架构 | Core Architecture**: Transformer (12 层 layers, 768 隐藏维度 hidden dimensions)
- **图学习 | Graph Learning**: 拉普拉斯矩阵、图神经网络 | Laplacian matrix, graph neural networks
- **物理建模 | Physical Modeling**: PINN (物理信息神经网络 Physics-Informed Neural Networks)
- **优化 | Optimization**: 拉普拉斯正则化、自适应学习率 | Laplacian regularization, adaptive learning rate
- **注意力机制 | Attention Mechanism**: AttnRes动态深度聚合、FlashAttention-2 | AttnRes dynamic depthwise aggregation, FlashAttention-2
- **序列建模 | Sequence Modeling**: Mamba状态空间模型、RetNet保留机制 | Mamba state space models, RetNet retention mechanism
- **空间感知 | Spatial Perception**: 四元数表示与运算 | Quaternion representation and operations

## 快速开始 | Quick Start

### 环境要求 | Environment Requirements
- Python 3.9+ (推荐 3.10+ recommended)
- Node.js 18.0+ (推荐 20.0+ recommended)
- CUDA 11.8+ (可选，用于 GPU 加速 optional, for GPU acceleration)
- 内存 | RAM: 最少 16GB (推荐 32GB 用于训练 recommended 32GB for training)
- 磁盘空间 | Disk Space: 最少 50GB

### 安装步骤 | Installation Steps

1. **克隆仓库 | Clone Repository**
```bash
git clone https://github.com/Sum-Outman/Self.git
cd Self
```

2. **创建 Python 虚拟环境 | Create Python Virtual Environment**
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python -m venv .venv
source .venv/bin/activate
```

3. **安装 Python 依赖 | Install Python Dependencies**
```bash
pip install -e .
```

4. **安装前端依赖 | Install Frontend Dependencies**
```bash
cd frontend
npm install
cd ..
```

5. **初始化数据库 | Initialize Database**
```bash
python backend/create_missing_tables.py
python backend/create_admin.py
```
**安全配置提示 | Security Configuration Note**: 生产环境请通过环境变量设置安全密码：
```bash
# Windows (PowerShell)
$env:ADMIN_PASSWORD="your_secure_password"
$env:DEMO_PASSWORD="another_secure_password"

# Linux/macOS
export ADMIN_PASSWORD="your_secure_password"
export DEMO_PASSWORD="another_secure_password"
```
或编辑 `.env` 文件中的用户账户配置部分。

6. **启动后端服务 | Start Backend Service**
```bash
python backend/main.py
```
后端将在 http://localhost:8000 启动，API 文档在 http://localhost:8000/docs | Backend starts at http://localhost:8000, API documentation at http://localhost:8000/docs

7. **启动前端开发服务器 | Start Frontend Development Server**
```bash
cd frontend
npm run dev
```
前端将在 http://localhost:3000 启动 | Frontend starts at http://localhost:3000

8. **访问系统 | Access System**
打开浏览器访问 http://localhost:3000 | Open browser and visit http://localhost:3000
- 默认管理员账号 | Default admin account: admin (邮箱: admin@selfagi.com)
- 默认演示账号 | Default demo account: demo (邮箱: demo@selfagi.com)
- 安全提示 | Security Note: 默认密码已隐藏，生产环境请通过环境变量设置安全密码 (ADMIN_PASSWORD, DEMO_PASSWORD)

## 拉普拉斯机制介绍 | Laplacian Mechanism Introduction

### 基本概念 | Basic Concepts

拉普拉斯矩阵是图论中的核心概念，在本系统中用于 | Laplacian matrix is a core concept in graph theory, used in this system for:

1. **图结构学习 | Graph Structure Learning**: 学习数据中的图结构关系 | Learning graph structure relationships in data
2. **特征平滑 | Feature Smoothing**: 通过拉普拉斯正则化实现特征平滑 | Feature smoothing via Laplacian regularization
3. **流形学习 | Manifold Learning**: 利用数据的流形结构进行半监督学习 | Semi-supervised learning using manifold structure of data
4. **多模态融合 | Multimodal Fusion**: 在多模态特征空间中构建图结构 | Building graph structure in multimodal feature space

### 数学原理 | Mathematical Principles

**非标准化拉普拉斯 | Unnormalized Laplacian**:
```
L = D - A
```
其中 | where:
- D 是度矩阵（对角矩阵，对角线上的元素是各节点的度） | D is the degree matrix (diagonal matrix with node degrees on the diagonal)
- A 是邻接矩阵（描述节点之间的连接关系） | A is the adjacency matrix (describes connections between nodes)

**对称标准化拉普拉斯 | Symmetric Normalized Laplacian**:
```
L_sym = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}
```

**随机游走标准化拉普拉斯 | Random Walk Normalized Laplacian**:
```
L_rw = D^{-1} L = I - D^{-1} A
```

### 在系统中的应用 | Applications in the System

1. **训练正则化 | Training Regularization**: 使用拉普拉斯正则化提高模型泛化能力 | Using Laplacian regularization to improve model generalization
2. **多模态融合 | Multimodal Fusion**: 在不同模态特征间构建图结构 | Building graph structures between different modality features
3. **记忆关联 | Memory Association**: 在记忆系统中构建关联图 | Building association graphs in the memory system
4. **知识图谱 | Knowledge Graph**: 知识库中的实体关系建模 | Entity relationship modeling in the knowledge base

## 机器人 AGI 功能 | Robot AGI Capabilities

### 硬件控制架构 | Hardware Control Architecture

系统采用四层硬件控制架构 | The system uses a four-layer hardware control architecture:

1. **AGI 智能层 | AGI Intelligence Layer**: 高层规划和推理 | High-level planning and reasoning
2. **硬件抽象层 | Hardware Abstraction Layer**: 统一的 API 接口 | Unified API interface
3. **协议层 | Protocol Layer**: 通信协议和驱动 | Communication protocols and drivers
4. **物理硬件 | Physical Hardware**: 实际机器人和传感器 | Actual robots and sensors

### 串口数据服务架构 | Serial Data Service Architecture

系统实现了统一的串口数据接收和解码服务，采用"接收后交给后台解码"的架构理念 | The system implements a unified serial data reception and decoding service, adopting the "receive then decode by backend" architecture concept:

```
硬件层（各种串口设备）
    ↓ 发送原始数据
串口接收服务（统一接收）
    ↓ 传递原始数据
后台解码服务（统一解码）
    ↓ 输出结构化数据
AGI系统（直接使用）
```

**核心特性 | Core Features**:

1. **统一数据接收 | Unified Data Reception**:
   - 前端/硬件层只需接收原始串口数据 | Frontend/hardware layer only needs to receive raw serial data
   - 支持多种数据格式：RAW、ASCII、HEX、JSON、二进制 | Supports multiple data formats: RAW, ASCII, HEX, JSON, Binary
   - 自动协议检测和格式转换 | Automatic protocol detection and format conversion

2. **智能解码服务 | Intelligent Decoding Service**:
   - `SerialDecoder`: 串口数据解码器，支持多种协议解码 | Serial data decoder supporting multiple protocol decoding
   - `SerialDataService`: 串口数据服务，统一管理数据接收、解码和分发 | Serial data service for unified data reception, decoding and distribution
   - 支持自定义解码规则和目的地处理器 | Supports custom decoding rules and destination handlers

3. **集成API端点 | Integrated API Endpoints**:
   - `POST /api/hardware/serial/receive`: 接收原始串口数据 | Receive raw serial data
   - `POST /api/hardware/serial/command`: 发送串口命令 | Send serial commands
   - 支持Base64、HEX、UTF-8等多种编码格式 | Supports multiple encoding formats: Base64, HEX, UTF-8

4. **实时数据流 | Real-time Data Stream**:
   - 支持WebSocket实时数据推送 | Supports WebSocket for real-time data push
   - 数据缓存和历史查询 | Data caching and historical queries
   - 统计信息和监控 | Statistics and monitoring

**设计优势 | Design Advantages**:
- **解耦合 | Decoupling**: 硬件接收与数据处理分离 | Hardware reception separated from data processing
- **可扩展性 | Scalability**: 易于添加新的设备协议 | Easy to add new device protocols
- **容错性 | Fault Tolerance**: 支持模拟模式和数据验证 | Supports simulation mode and data validation
- **统一性 | Unification**: 所有串口数据统一处理管道 | Unified processing pipeline for all serial data

### 支持的硬件 | Supported Hardware

**机器人平台 | Robot Platforms**:
- 人形机器人 (双足、四足) | Humanoid robots (bipedal, quadrupedal)
- 机械臂 (6-DOF、7-DOF) | Robotic arms (6-DOF, 7-DOF)
- 移动机器人 (轮式、履带式) | Mobile robots (wheeled, tracked)

**传感器 | Sensors**:
- 视觉传感器 (RGB、深度、立体相机) | Vision sensors (RGB, depth, stereo cameras)
- 惯性传感器 (IMU、陀螺仪、加速度计) | Inertial sensors (IMU, gyroscopes, accelerometers)
- 距离传感器 (激光雷达、超声波、红外) | Range sensors (LiDAR, ultrasonic, infrared)
- 力/扭矩传感器 | Force/torque sensors

### 仿真环境 | Simulation Environments

**PyBullet 仿真 | PyBullet Simulation**:
- 基于物理的真实仿真 | Physics-based realistic simulation
- 精确的碰撞检测 | Accurate collision detection
- 支持 URDF 机器人描述格式 | Supports URDF robot description format

**Gazebo 仿真 | Gazebo Simulation**:
- ROS 2 集成 | ROS 2 integration
- 高级传感器模型 | Advanced sensor models
- 多机器人仿真支持 | Multi-robot simulation support

## 项目结构 | Project Structure

```
Self/
├── backend/                 # FastAPI 后端 | FastAPI backend
│   ├── routes/             # API 路由 | API routes
│   ├── services/           # 业务逻辑服务 | Business logic services
│   │   ├── serial_decoder.py        # 串口数据解码器 | Serial data decoder
│   │   ├── serial_data_service.py   # 串口数据服务 | Serial data service
│   │   ├── hardware_service.py      # 硬件服务 | Hardware service
│   │   └── base_service.py          # 服务基类 | Service base class
│   ├── db_models/          # 数据库模型 | Database models
│   ├── core/               # 核心工具 | Core utilities
│   │   └── error_handler.py         # 统一错误处理 | Unified error handling
│   └── main.py             # 应用入口 | Application entry point
├── frontend/               # React 前端 | React frontend
│   └── src/
│       ├── pages/          # 页面组件 | Page components
│       ├── components/     # UI 组件 | UI components
│       └── services/       # API 客户端 | API clients
├── models/                 # AGI 模型实现 | AGI model implementations
│   ├── transformer/        # Transformer 核心 | Transformer core
│   ├── graph/              # 图学习 (拉普拉斯) | Graph learning (Laplacian)
│   ├── multimodal/         # 多模态处理 | Multimodal processing
│   ├── memory/             # 记忆系统 | Memory system
│   └── physics/            # 物理建模 | Physical modeling
├── training/               # 训练系统 | Training system
│   ├── laplacian/          # 拉普拉斯增强训练 | Laplacian-enhanced training
│   ├── self_learning_manager.py  # 自主学习管理器 | Self-learning manager
│   └── trainer.py          # 主训练器 | Main trainer
├── hardware/               # 硬件集成 | Hardware integration
│   ├── robot_controller.py # 机器人控制 | Robot control
│   ├── simulation.py       # 仿真环境 | Simulation environment
│   └── unified_interface.py # 统一接口 | Unified interface
├── models/                 # 系统控制模块 | System control modules
│   └── system_control/     # 系统控制 | System control
│       ├── serial_controller.py      # 串口控制器 | Serial controller
│       └── real_hardware/            # 真实硬件接口 | Real hardware interface
│           ├── base_interface.py     # 硬件接口基类 | Hardware interface base class
│           ├── motor_controller.py   # 电机控制器 | Motor controller
│           ├── sensor_interface.py   # 传感器接口 | Sensor interface
│           ├── naoqi_controller.py   # NAOqi机器人控制器 | NAOqi robot controller
│           ├── real_hardware_manager.py # 真实硬件管理器 | Real hardware manager
│           └── __init__.py           # 模块初始化 | Module initialization
├── docs/                   # 文档 | Documentation
└── tests/                  # 测试 | Tests
    ├── unit/                # 单元测试 | Unit tests
    │   ├── services/        # 服务测试 | Service tests
    │   │   └── test_serial_data_service.py # 串口数据服务测试 | Serial data service tests
    │   └── hardware/        # 硬件测试 | Hardware tests
    └── integration/         # 集成测试 | Integration tests
```

## 开发指南 | Development Guide

### 代码风格 | Code Style

- **Python**: 遵循 PEP 8，使用 Black 格式化 | Follow PEP 8, use Black formatting
- **TypeScript**: 使用严格模式，ESLint + Prettier | Use strict mode, ESLint + Prettier
- **文档 | Documentation**: 更新 README 和 docs 目录 | Update README and docs directory

### 运行测试 | Running Tests

```bash
# 运行所有测试 | Run all tests
pytest tests/

# 运行特定测试 | Run specific tests
pytest tests/unit/

# 运行串口数据服务测试 | Run serial data service tests
pytest tests/unit/services/test_serial_data_service.py -v

# 运行硬件相关测试 | Run hardware-related tests
pytest tests/unit/hardware/ -v

# 生成测试覆盖率报告 | Generate test coverage report
pytest tests/ --cov=backend --cov=hardware --cov=models --cov-report=html
```

### 串口数据服务测试示例 | Serial Data Service Test Example

串口数据服务包含完整的测试覆盖，确保数据接收和解码的可靠性 | Serial data service includes comprehensive test coverage ensuring reliable data reception and decoding:

1. **解码器功能测试 | Decoder Function Tests**:
   - RAW、ASCII、HEX、JSON协议解码 | RAW, ASCII, HEX, JSON protocol decoding
   - 协议自动检测 | Automatic protocol detection
   - 错误处理和统计 | Error handling and statistics

2. **数据服务测试 | Data Service Tests**:
   - 服务启动/停止 | Service start/stop
   - 数据接收和处理 | Data reception and processing
   - 目的地处理器集成 | Destination handler integration

3. **集成测试 | Integration Tests**:
   - 与机器人控制器集成 | Integration with robot controller
   - API端点功能验证 | API endpoint functionality verification

## 许可证 | License

本项目采用 Apache License 2.0 开源协议 - 查看 [LICENSE](LICENSE) 文件了解详情。 | This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 联系方式 | Contact

- **邮箱 | Email**: silencecrowtom@qq.com
- **GitHub 仓库 | GitHub Repository**: https://github.com/Sum-Outman/Self
- **Issues**: https://github.com/Sum-Outman/Self/issues

---

## 更新与修复 | Updates and Fixes

### 2026年4月1日 | April 1, 2026

#### 新增功能介绍 | New Feature Introductions
1. **模仿学习能力 | Imitation Learning Capability**: 
   - 添加行为克隆(Behavior Cloning)、逆强化学习(IRL)、生成对抗模仿学习(GAIL)等核心功能介绍
   - 包含演示数据采集、轨迹对齐、技能泛化、增量学习等技术优势说明
   - 支持多模态演示处理（视频、运动捕捉、遥操作等）

2. **真实世界交互能力 | Real-World Interaction Capability**:
   - 添加感知系统（多模态传感器融合、环境建模、物体识别跟踪、人体姿态估计）
   - 添加决策与规划（任务级规划、运动规划、反应式控制、风险评估）
   - 添加执行系统（硬件抽象层、力控制、安全机制、实时通信）

3. **机器人AGI功能扩展 | Robot AGI Feature Expansion**:
   - 在系统功能部分扩展模仿学习集成说明
   - 添加真实世界感知、物理交互、人机协作等功能介绍

#### README文档更新 | README Documentation Updates
1. **完整中英文对照 | Complete Chinese-English Bilingual**: 所有内容均提供中英文对照版本 | All content provided in Chinese-English bilingual format
2. **自我学习功能介绍 | Self-Learning Feature Introduction**: 添加完整的自主学习管理器功能说明 | Added complete self-learning manager feature description
3. **自我修正功能介绍 | Self-Correction Feature Introduction**: 添加完整的自我修正模块功能说明 | Added complete self-correction module feature description
4. **自我演化功能介绍 | Self-Evolution Feature Introduction**: 添加自主演化能力说明 | Added autonomous evolution capability description
5. **AttnRes特色功能 | AttnRes Special Feature**: 添加动态深度聚合注意力机制介绍 | Added dynamic depthwise aggregation attention mechanism introduction
6. **四元数集成 | Quaternion Integration**: 添加四元数表示与运算功能说明 | Added quaternion representation and operations feature description
7. **混合专家系统 | Mixture of Experts**: 添加MoE系统功能说明 | Added MoE system feature description
8. **状态空间模型 | State Space Models**: 添加Mamba和RetNet架构说明 | Added Mamba and RetNet architecture description

#### 安全改进 | Security Improvements
1. **管理员用户创建脚本** (`backend/create_admin.py`):
   - 移除硬编码密码，支持通过环境变量配置（ADMIN_PASSWORD, ADMIN_EMAIL, DEMO_PASSWORD, DEMO_EMAIL, DEMO_IS_ADMIN）
   - 移除控制台打印密码的安全风险
   - 添加密码强度检查（最小8字符）
   - 演示用户默认不是管理员（可通过环境变量启用）

2. **启动脚本更新** (`start_self_agi.bat`):
   - 更新默认登录信息显示，不再显示密码
   - 添加安全配置提示和环境变量说明
   - 在.env模板中添加用户账户配置选项

3. **README文档更新**:
   - 更新默认登录信息和安全提示
   - 添加环境变量配置说明

#### 硬件接口修复 | Hardware Interface Fixes
1. **训练器硬件接口** (`training/trainer.py`):
   - 将`simulation_mode`默认值从`True`改为`False`，强制要求真实硬件连接
   - 重写`_init_robot_hardware_interface`函数，移除模拟实现
   - 要求连接真实硬件，否则抛出异常
   - 符合项目"禁止使用虚拟数据"的要求

#### 代码缺陷修复 | Code Defect Fixes
1. **导入错误修复** (`backend/main.py`):
   - 修复缺失的数据库模型导入（PasswordResetToken, EmailVerificationToken等）
   - 修复JWT错误处理（`jwt.PyJWTError`改为`jwt.JWTError`）

2. **数据类字段顺序修复** (`training/external_api_framework.py`):
   - 修复`APIRequest`类字段顺序错误（非默认参数不能在默认参数之后）
   - 添加缺失的`url`字段

#### 项目要求符合性 | Project Requirement Compliance
- 严格执行"禁止使用虚拟数据"要求
- 硬件接口在真实硬件不可用时显示警告而非提供模拟数据
- 移除所有模拟回退机制，要求真实硬件连接
- 保持系统完整性和安全性标准

---

*最后更新 | Last Updated: 2026年4月1日 | April 1, 2026*
