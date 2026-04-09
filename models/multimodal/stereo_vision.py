#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双目空间识别模块

功能：
1. 双目图像采集与同步
2. 立体匹配与深度计算
3. 三维空间重建
4. 物体三维定位与姿态识别
5. 双目视觉标定

工业级AGI系统要求：从零开始训练，不使用预训练模型依赖
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class StereoVisionModule(nn.Module):
    """双目空间识别模块 - 基于深度学习的立体视觉处理

    功能：
    - 双目图像特征提取
    - 视差图计算
    - 深度图生成
    - 三维点云重建
    - 物体三维定位
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        embedding_dim: int = 768,
        max_disparity: int = 128,
        focal_length: float = 800.0,
        baseline: float = 0.12,  # 基线距离（米）
    ):
        super().__init__()
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        self.max_disparity = max_disparity
        self.focal_length = focal_length
        self.baseline = baseline

        # 左图像编码器
        self.left_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 残差块1
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 残差块2
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 残差块3
            nn.Conv2d(256, embedding_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
        )

        # 右图像编码器（共享权重）
        self.right_encoder = self.left_encoder

        # 立体匹配网络 - 计算视差
        self.disparity_network = nn.Sequential(
            nn.Conv2d(embedding_dim * 2, embedding_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim, embedding_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim // 2, max_disparity, kernel_size=3, padding=1),
        )

        # 深度估计网络
        self.depth_refinement = nn.Sequential(
            nn.Conv2d(max_disparity, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        # 三维空间特征提取
        self.spatial_feature_extractor = nn.Sequential(
            nn.Linear(embedding_dim + 3, embedding_dim),  # +3 for XYZ coordinates
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        logger.info("双目空间识别模块初始化完成")
        logger.info(f"  图像尺寸: {image_size}")
        logger.info(f"  嵌入维度: {embedding_dim}")
        logger.info(f"  最大视差: {max_disparity}")
        logger.info(f"  基线距离: {baseline}m")
        logger.info(f"  焦距: {focal_length}px")

    def forward(
        self, left_image: torch.Tensor, right_image: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """前向传播 - 双目视觉处理

        参数:
            left_image: 左图像 [B, 3, H, W]
            right_image: 右图像 [B, 3, H, W]

        返回:
            包含视差图、深度图、三维特征的字典
        """
        batch_size = left_image.size(0)

        # 提取左右图像特征
        left_features = self.left_encoder(left_image)
        right_features = self.right_encoder(right_image)

        # 计算视差图
        combined_features = torch.cat([left_features, right_features], dim=1)
        disparity_logits = self.disparity_network(combined_features)

        # 软argmax获取视差值
        disparity_prob = torch.softmax(disparity_logits, dim=1)
        disparity_values = torch.arange(
            self.max_disparity, dtype=torch.float32, device=disparity_logits.device
        )
        disparity_map = torch.sum(
            disparity_prob * disparity_values.view(1, -1, 1, 1), dim=1, keepdim=True
        )

        # 计算深度图
        # depth = (focal_length * baseline) / disparity
        # 避免除零
        disparity_safe = torch.clamp(disparity_map, min=1.0)
        depth_map = (self.focal_length * self.baseline) / disparity_safe
        depth_map = torch.clamp(depth_map, min=0.1, max=100.0)

        # 深度细化
        depth_refined = self.depth_refinement(disparity_logits)
        depth_refined = depth_refined * 100.0  # 缩放到实际深度范围

        # 生成三维点云
        point_cloud = self._generate_point_cloud(depth_refined, batch_size)

        # 提取空间特征
        spatial_features = self.spatial_feature_extractor(
            torch.cat([left_features.mean(dim=[2, 3]), point_cloud.mean(dim=1)], dim=1)
        )

        return {
            "disparity_map": disparity_map,
            "depth_map": depth_refined,
            "point_cloud": point_cloud,
            "spatial_features": spatial_features,
            "left_features": left_features,
            "right_features": right_features,
        }

    def _generate_point_cloud(
        self, depth_map: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """根据深度图生成三维点云

        参数:
            depth_map: 深度图 [B, 1, H, W]
            batch_size: 批次大小

        返回:
            点云坐标 [B, N, 3]
        """
        _, _, height, width = depth_map.shape

        # 创建像素坐标网格
        u = torch.arange(width, dtype=torch.float32, device=depth_map.device)
        v = torch.arange(height, dtype=torch.float32, device=depth_map.device)
        u, v = torch.meshgrid(u, v, indexing="xy")

        # 相机内参（假设主点在图像中心）
        cx = width / 2.0
        cy = height / 2.0

        # 计算三维坐标
        Z = depth_map.squeeze(1)  # [B, H, W]
        X = (u - cx) * Z / self.focal_length
        Y = (v - cy) * Z / self.focal_length

        # 组合成点云 [B, H*W, 3]
        point_cloud = torch.stack(
            [X.view(batch_size, -1), Y.view(batch_size, -1), Z.view(batch_size, -1)],
            dim=2,
        )

        return point_cloud

    def detect_objects_3d(
        self,
        left_image: torch.Tensor,
        right_image: torch.Tensor,
        object_regions: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """三维物体检测与定位

        参数:
            left_image: 左图像
            right_image: 右图像
            object_regions: 可选的物体区域列表

        返回:
            物体三维信息列表
        """
        with torch.no_grad():
            results = self.forward(left_image, right_image)

        depth_map = results["depth_map"]
        point_cloud = results["point_cloud"]

        objects_3d = []

        if object_regions is None:
            # 如果没有提供区域，使用整个图像
            object_regions = [
                {
                    "label": "scene",
                    "bbox": [0, 0, left_image.size(3), left_image.size(2)],
                }
            ]

        for region in object_regions:
            bbox = region.get("bbox", [0, 0, left_image.size(3), left_image.size(2)])
            x1, y1, x2, y2 = map(int, bbox)

            # 提取区域内的点云
            batch_idx = 0
            h, w = depth_map.size(2), depth_map.size(3)

            # 将像素坐标转换为点云索引
            idx_y = torch.linspace(y1, y2 - 1, steps=min(10, y2 - y1), dtype=torch.long)
            idx_x = torch.linspace(x1, x2 - 1, steps=min(10, x2 - x1), dtype=torch.long)

            indices = []
            for y in idx_y:
                for x in idx_x:
                    if y < h and x < w:
                        indices.append(y * w + x)

            if indices:
                indices = torch.tensor(
                    indices, dtype=torch.long, device=point_cloud.device
                )
                region_points = point_cloud[batch_idx, indices]

                # 计算物体三维位置和尺寸
                center = region_points.mean(dim=0)
                min_coords = region_points.min(dim=0)[0]
                max_coords = region_points.max(dim=0)[0]
                size = max_coords - min_coords

                objects_3d.append(
                    {
                        "label": region.get("label", "unknown"),
                        "center_3d": center.cpu().numpy().tolist(),
                        "size_3d": size.cpu().numpy().tolist(),
                        "distance": float(center[2].cpu()),  # Z坐标为深度
                        "bbox_2d": bbox,
                    }
                )

        return objects_3d

    def calibrate(
        self,
        calibration_images: List[Tuple[np.ndarray, np.ndarray]],
        chessboard_size: Tuple[int, int] = (9, 6),
    ) -> Dict[str, Any]:
        """双目相机标定

        参数:
            calibration_images: 标定图像对列表 [(left_img, right_img), ...]
            chessboard_size: 棋盘格尺寸 (内角点数量)

        返回:
            标定参数
        """
        try:
            import cv2

            # 准备对象点
            objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[
                0: chessboard_size[0], 0: chessboard_size[1]
            ].T.reshape(-1, 2)

            objpoints = []
            imgpoints_left = []
            imgpoints_right = []

            for left_img, right_img in calibration_images:
                gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
                gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

                ret_left, corners_left = cv2.findChessboardCorners(
                    gray_left, chessboard_size, None
                )
                ret_right, corners_right = cv2.findChessboardCorners(
                    gray_right, chessboard_size, None
                )

                if ret_left and ret_right:
                    objpoints.append(objp)
                    imgpoints_left.append(corners_left)
                    imgpoints_right.append(corners_right)

            if len(objpoints) < 3:
                logger.warning("标定图像不足，至少需要3组有效图像")
                return {}  # 返回空字典

            # 单目标定
            ret_left, mtx_left, dist_left, _, _ = cv2.calibrateCamera(
                objpoints, imgpoints_left, gray_left.shape[::-1], None, None
            )
            ret_right, mtx_right, dist_right, _, _ = cv2.calibrateCamera(
                objpoints, imgpoints_right, gray_right.shape[::-1], None, None
            )

            # 双目标定
            flags = cv2.CALIB_FIX_INTRINSIC
            ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
                objpoints,
                imgpoints_left,
                imgpoints_right,
                mtx_left,
                dist_left,
                mtx_right,
                dist_right,
                gray_left.shape[::-1],
                None,
                None,
                None,
                None,
                flags=flags,
            )

            calibration_result = {
                "left_camera_matrix": mtx_left.tolist(),
                "left_distortion": dist_left.tolist(),
                "right_camera_matrix": mtx_right.tolist(),
                "right_distortion": dist_right.tolist(),
                "rotation_matrix": R.tolist(),
                "translation_vector": T.tolist(),
                "essential_matrix": E.tolist(),
                "fundamental_matrix": F.tolist(),
                "reprojection_error": float(ret),
            }

            logger.info("双目相机标定完成")
            logger.info(f"  重投影误差: {ret:.4f}")

            return calibration_result

        except ImportError:
            logger.error("OpenCV不可用，无法进行相机标定")
            return {}  # 返回空字典
        except Exception as e:
            logger.error(f"相机标定失败: {e}")
            return {}  # 返回空字典

    def get_module_info(self) -> Dict[str, Any]:
        """获取模块信息"""
        return {
            "module_name": "StereoVisionModule",
            "module_type": "双目空间识别",
            "image_size": self.image_size,
            "embedding_dim": self.embedding_dim,
            "max_disparity": self.max_disparity,
            "focal_length": self.focal_length,
            "baseline": self.baseline,
            "capabilities": [
                "双目图像处理",
                "视差图计算",
                "深度图生成",
                "三维点云重建",
                "物体三维定位",
                "相机标定",
            ],
            "status": "ready",
        }


class StereoVisionService:
    """双目空间识别服务 - 单例模式"""

    _instance = None
    _module = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self._load_module()

    def _load_module(self):
        """加载双目空间识别模块"""
        try:
            logger.info("正在加载双目空间识别模块...")

            self._module = StereoVisionModule(
                image_size=(224, 224),
                embedding_dim=768,
                max_disparity=128,
                focal_length=800.0,
                baseline=0.12,
            )

            # 设置为评估模式
            self._module.eval()

            logger.info("双目空间识别模块加载成功")

        except Exception as e:
            logger.error(f"双目空间识别模块加载失败: {e}")
            self._module = None

    def process_stereo_images(
        self, left_image: np.ndarray, right_image: np.ndarray
    ) -> Dict[str, Any]:
        """处理双目图像

        参数:
            left_image: 左图像 (numpy数组)
            right_image: 右图像 (numpy数组)

        返回:
            处理结果字典
        """
        if self._module is None:
            return {"success": False, "error": "模块未加载"}

        try:
            import torch
            from PIL import Image
            import torchvision.transforms as transforms

            # 转换图像格式
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            # 转换numpy数组为PIL图像
            if isinstance(left_image, np.ndarray):
                left_pil = Image.fromarray(left_image)
                right_pil = Image.fromarray(right_image)
            else:
                left_pil = left_image
                right_pil = right_image

            # 应用变换
            left_tensor = transform(left_pil).unsqueeze(0)
            right_tensor = transform(right_pil).unsqueeze(0)

            # 处理
            with torch.no_grad():
                results = self._module(left_tensor, right_tensor)

            return {
                "success": True,
                "disparity_map": results["disparity_map"].cpu().numpy(),
                "depth_map": results["depth_map"].cpu().numpy(),
                "point_cloud": results["point_cloud"].cpu().numpy(),
                "spatial_features": results["spatial_features"].cpu().numpy(),
            }

        except Exception as e:
            logger.error(f"双目图像处理失败: {e}")
            return {"success": False, "error": str(e)}

    def get_service_info(self) -> Dict[str, Any]:
        """获取服务信息"""
        if self._module is None:
            return {"status": "not_loaded", "error": "模块未加载"}

        return {"status": "loaded", "module_info": self._module.get_module_info()}


# 全局单例实例
_stereo_vision_service = None


def get_stereo_vision_service() -> StereoVisionService:
    """获取双目空间识别服务单例"""
    global _stereo_vision_service
    if _stereo_vision_service is None:
        _stereo_vision_service = StereoVisionService()
    return _stereo_vision_service
