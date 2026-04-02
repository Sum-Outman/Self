# Contrastive Learning and Cross-Modal Alignment | 对比学习和跨模态对齐

This document provides detailed documentation of the contrastive learning and cross-modal alignment capabilities in Self AGI.

本文档详细介绍 Self AGI 中的对比学习和跨模态对齐功能。

## Overview | 概述

The Contrastive Alignment Model provides cross-modal representation learning using contrastive learning, inspired by CLIP and ALIGN architectures.

对比对齐模型使用对比学习提供跨模态表示学习，受 CLIP 和 ALIGN 架构启发。

### Core Features | 核心特性

1. **Contrastive Loss | 对比损失**: InfoNCE loss function for aligning different modalities
2. **Shared Embedding Space | 共享嵌入空间**: All modalities mapped to the same space
3. **Temperature Parameter | 温度参数**: Learnable temperature for contrastive learning
4. **Hard Negative Mining | 硬负样本挖掘**: Difficult negative examples for enhanced training

## Contrastive Alignment Model | 对比对齐模型

```python
from models.multimodal.contrastive_learning import ContrastiveAlignmentModel

# Initialize contrastive alignment model
config = {
    'text_embedding_dim': 768,
    'image_embedding_dim': 768,
    'audio_embedding_dim': 256,
    'sensor_embedding_dim': 256,
    'vocab_size': 100000,
    'num_layers': 12,
    'max_position_embeddings': 2048,
    'image_size': 224,
    'patch_size': 16,
    'projection_dim': 512,
    'temperature': 0.07,
    'learnable_temperature': True
}

contrastive_model = ContrastiveAlignmentModel(config)
```

## InfoNCE Loss | InfoNCE 损失

```python
# The model uses the InfoNCE (Noise Contrastive Estimation) loss:
#
# L = -log [ exp(sim(z_i, z_j) / τ) / Σ_{k=1 to 2N} exp(sim(z_i, z_k) / τ) ]
#
# where:
# - sim is cosine similarity
# - τ is temperature parameter
# - z_i and z_j are positive pairs
# - z_k are negative samples

# Compute contrastive loss
text_features = contrastive_model.encode_text(text_input)
image_features = contrastive_model.encode_image(image_input)

loss = contrastive_model.compute_contrastive_loss(
    features_1=text_features,
    features_2=image_features,
    temperature=0.07
)

print(f"Contrastive Loss: {loss.item():.4f}")
```

## Cross-Modal Retrieval | 跨模态检索

```python
# Text-to-Image retrieval
query_text = "A photo of a cat sitting on a mat"
text_embedding = contrastive_model.encode_text(query_text)

# Search in image database
top_k_images = contrastive_model.retrieve(
    query=text_embedding,
    database=image_database,
    modality='image',
    top_k=5
)

print("Text-to-Image Retrieval Results:")
for i, result in enumerate(top_k_images):
    print(f"{i+1}: {result['id']} (Score: {result['score']:.3f})")

# Image-to-Text retrieval
image_embedding = contrastive_model.encode_image(query_image)
top_k_texts = contrastive_model.retrieve(
    query=image_embedding,
    database=text_database,
    modality='text',
    top_k=5
)
```

## Training Pipeline | 训练流程

```python
from models.multimodal.contrastive_learning import ContrastiveAlignmentTrainer

# Initialize trainer
trainer = ContrastiveAlignmentTrainer(
    model=contrastive_model,
    learning_rate=3e-4,
    batch_size=256,
    temperature=0.07,
    hard_negative_mining=True,
    num_hard_negatives=4
)

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0.0
    
    for batch in dataloader:
        # Get paired data
        text_batch = batch['text']
        image_batch = batch['image']
        audio_batch = batch.get('audio')
        sensor_batch = batch.get('sensor')
        
        # Forward pass
        loss = trainer.train_step(
            text=text_batch,
            image=image_batch,
            audio=audio_batch,
            sensor=sensor_batch
        )
        
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader):.4f}")
    
    # Validation
    if (epoch + 1) % 5 == 0:
        recall_at_k = trainer.evaluate_recall(val_dataloader, k=[1, 5, 10])
        print(f"Recall@1: {recall_at_k[1]:.2%}")
        print(f"Recall@5: {recall_at_k[5]:.2%}")
```

## Shared Embedding Space | 共享嵌入空间

```python
# All modalities are projected to the same embedding space
text_emb = contrastive_model.encode_text(text_input)  # Shape: [batch, 512]
image_emb = contrastive_model.encode_image(image_input)  # Shape: [batch, 512]
audio_emb = contrastive_model.encode_audio(audio_input)  # Shape: [batch, 512]
sensor_emb = contrastive_model.encode_sensor(sensor_input)  # Shape: [batch, 512]

# Compute cross-modal similarities
text_image_sim = torch.cosine_similarity(text_emb, image_emb, dim=-1)
text_audio_sim = torch.cosine_similarity(text_emb, audio_emb, dim=-1)

print(f"Text-Image Similarity: {text_image_sim.mean():.3f}")
print(f"Text-Audio Similarity: {text_audio_sim.mean():.3f}")
```

*Last Updated: March 31, 2026*  
*最后更新: 2026年3月31日*
