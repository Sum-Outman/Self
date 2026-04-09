"""
资源预测器
预测训练资源需求，优化资源配置

功能：
1. 训练时间预测算法
2. 资源需求预测（GPU/内存/存储）
3. 优化建议自动生成
4. 成本估算和效益分析
"""

import json
import math
import os
import statistics
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, asdict, field
import logging
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """资源类型"""

    GPU = "gpu"
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    POWER = "power"


class PredictionMethod(Enum):
    """预测方法"""

    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    TIME_SERIES = "time_series"
    RULE_BASED = "rule_based"
    NEURAL_NETWORK = "neural_network"
    HYBRID = "hybrid"


class OptimizationGoal(Enum):
    """优化目标"""

    MINIMIZE_TIME = "minimize_time"
    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_RESOURCES = "minimize_resources"
    MAXIMIZE_PERFORMANCE = "maximize_performance"
    BALANCED = "balanced"


@dataclass
class ResourceProfile:
    """资源配置"""

    resource_type: ResourceType
    quantity: float
    unit: str
    specification: str = ""
    cost_per_hour: float = 0.0
    availability: float = 1.0  # 可用性（0-1）
    power_consumption_watts: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result["resource_type"] = self.resource_type.value
        return result


@dataclass
class TrainingProfile:
    """训练配置"""

    model_parameters: int
    batch_size: int
    sequence_length: int = 512  # Transformer相关
    hidden_size: int = 768  # Transformer相关
    num_layers: int = 12  # Transformer相关
    num_heads: int = 12  # Transformer相关
    vocab_size: int = 50257  # Transformer相关
    dataset_size: int
    epochs: int
    learning_rate: float = 0.001
    optimizer: str = "adam"
    mixed_precision: bool = True
    gradient_accumulation: int = 1
    distributed_training: bool = False
    num_gpus: int = 1
    num_nodes: int = 1
    checkpoint_frequency: int = 1000
    validation_frequency: int = 100

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    def compute_flops(self) -> float:
        """计算理论FLOPs（浮点运算次数）"""
        # Transformer模型的FLOPs估计公式
        # 参考：https://arxiv.org/abs/2001.08361
        L = self.num_layers
        H = self.hidden_size
        self.num_heads
        self.vocab_size
        S = self.sequence_length
        B = self.batch_size

        # 每个Transformer层的FLOPs
        # 自注意力：2 * B * S * S * H
        # 前馈网络：2 * B * S * H * 4H (假设中间层是4H)
        # 层归一化和残差连接：忽略，相对较小

        attention_flops = 2 * B * S * S * H
        ffn_flops = 2 * B * S * H * 4 * H

        per_layer_flops = attention_flops + ffn_flops

        # 总FLOPs（前向传播）
        forward_flops = per_layer_flops * L

        # 反向传播大约是前向传播的2-3倍
        backward_flops = forward_flops * 2.5

        # 总FLOPs per iteration
        total_flops_per_iter = forward_flops + backward_flops

        # 乘以总迭代次数
        iterations = self.dataset_size / B * self.epochs
        total_flops = total_flops_per_iter * iterations

        return total_flops

    def compute_memory_requirements(self) -> Dict[str, float]:
        """计算内存需求"""
        # 模型参数内存（字节）
        param_memory_bytes = self.model_parameters * 4  # 假设float32

        # 梯度内存（与参数相同）
        grad_memory_bytes = param_memory_bytes

        # 优化器状态内存
        # Adam优化器：参数 * 2 * 4（momentum + variance）
        optimizer_memory_bytes = self.model_parameters * 2 * 4

        # 激活内存（近似估计）
        # Transformer激活内存 ≈ batch_size * sequence_length * hidden_size * layers * 2
        activation_memory_bytes = (
            self.batch_size
            * self.sequence_length
            * self.hidden_size
            * 4
            * self.num_layers
            * 2
        )

        # 总内存需求
        total_memory_bytes = (
            param_memory_bytes
            + grad_memory_bytes
            + optimizer_memory_bytes
            + activation_memory_bytes
        )

        # 转换为GB
        total_memory_gb = total_memory_bytes / (1024**3)

        return {
            "parameters_gb": param_memory_bytes / (1024**3),
            "gradients_gb": grad_memory_bytes / (1024**3),
            "optimizer_gb": optimizer_memory_bytes / (1024**3),
            "activations_gb": activation_memory_bytes / (1024**3),
            "total_gb": total_memory_gb,
            "minimum_gpu_memory_gb": total_memory_gb * 1.2,  # 增加20%安全边际
        }


@dataclass
class HistoricalData:
    """历史训练数据"""

    training_id: str
    training_profile: TrainingProfile
    actual_duration_hours: float
    actual_resource_usage: Dict[str, float]
    actual_cost: float
    success: bool
    failure_reason: str = ""
    start_time: str = ""
    end_time: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result["training_profile"] = self.training_profile.to_dict()
        return result


@dataclass
class PredictionResult:
    """预测结果"""

    training_duration_hours: float
    confidence_interval: Tuple[float, float]  # 95%置信区间
    resource_requirements: Dict[str, float]  # 资源需求
    estimated_cost: float
    bottlenecks: List[str]
    optimization_suggestions: List[Dict[str, Any]]
    risk_assessment: Dict[str, float]
    alternative_configs: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            **asdict(self),
            "confidence_interval": list(self.confidence_interval),
        }


class ResourcePredictor:
    """资源预测器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化资源预测器"""
        self.config = config or {
            "prediction_method": PredictionMethod.HYBRID.value,
            "optimization_goal": OptimizationGoal.BALANCED.value,
            "confidence_level": 0.95,
            "historical_data_limit": 1000,
            "enable_ml_predictions": True,
            "enable_rule_based_predictions": True,
            "enable_cost_estimation": True,
            "resource_costs": {
                "gpu_hour": 2.0,  # 美元/GPU小时
                "cpu_hour": 0.1,  # 美元/CPU小时
                "memory_gb_hour": 0.02,  # 美元/GB小时
                "storage_gb_hour": 0.001,  # 美元/GB小时
            },
            "performance_models": {
                "gpu_flops": 19.5e12,  # NVIDIA A100 19.5 TFLOPS
                "cpu_flops": 0.5e12,  # 现代CPU 0.5 TFLOPS
                "memory_bandwidth_gbps": 1555,  # A100 1555 GB/s
                "network_bandwidth_gbps": 10,  # 典型网络 10 Gbps
            },
        }

        self.logger = logging.getLogger(f"{__name__}.ResourcePredictor")

        # 历史数据存储
        self.historical_data: List[HistoricalData] = []

        # ML模型
        self.ml_models: Dict[str, Any] = {}
        self.scaler = StandardScaler()

        # 规则库
        self.rules = self._initialize_rules()

        # 加载历史数据
        self._load_historical_data()

        self.logger.info("资源预测器初始化完成")

    def _initialize_rules(self) -> Dict[str, Any]:
        """初始化规则库"""
        return {
            "memory_rules": [
                {
                    "condition": lambda tp: tp.model_parameters > 10e9,
                    "action": "require_large_memory",
                    "message": "模型参数超过100亿，需要大内存配置",
                    "min_memory_gb": 128,
                },
                {
                    "condition": lambda tp: tp.batch_size > 1024,
                    "action": "require_high_memory",
                    "message": "批量大小超过1024，需要高内存带宽",
                    "min_memory_gb": 64,
                },
            ],
            "gpu_rules": [
                {
                    "condition": lambda tp: tp.model_parameters > 1e9,
                    "action": "require_multiple_gpus",
                    "message": "模型参数超过10亿，建议使用多GPU",
                    "min_gpus": 4,
                },
                {
                    "condition": lambda tp: tp.dataset_size > 1e9,
                    "action": "require_gpu_cluster",
                    "message": "数据集超过10亿样本，建议使用GPU集群",
                    "min_gpus": 8,
                },
            ],
            "time_rules": [
                {
                    "condition": lambda tp: tp.compute_flops() > 1e21,
                    "action": "long_training_warning",
                    "message": "理论FLOPs超过10^21，训练时间可能很长",
                    "estimated_days": 30,
                },
            ],
            "optimization_rules": [
                {
                    "condition": lambda tp: tp.batch_size < 32,
                    "action": "increase_batch_size",
                    "message": "批量大小小于32，GPU利用率可能不足",
                    "suggestion": "增加批量大小到32或更大",
                },
                {
                    "condition": lambda tp: not tp.mixed_precision,
                    "action": "enable_mixed_precision",
                    "message": "未启用混合精度训练",
                    "suggestion": "启用混合精度训练以节省内存和加速训练",
                },
            ],
        }

    def _load_historical_data(self):
        """加载历史数据"""
        # 这里可以从数据库或文件加载
        # 暂时使用空列表
        self.historical_data = []

        # 如果有历史数据文件，加载它
        historical_file = "historical_training_data.json"
        try:
            if os.path.exists(historical_file):
                with open(historical_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        # 转换回HistoricalData对象
                        profile = TrainingProfile(**item["training_profile"])
                        historical = HistoricalData(
                            training_id=item["training_id"],
                            training_profile=profile,
                            actual_duration_hours=item["actual_duration_hours"],
                            actual_resource_usage=item["actual_resource_usage"],
                            actual_cost=item["actual_cost"],
                            success=item["success"],
                            failure_reason=item.get("failure_reason", ""),
                            start_time=item.get("start_time", ""),
                            end_time=item.get("end_time", ""),
                            metrics=item.get("metrics", {}),
                        )
                        self.historical_data.append(historical)

                self.logger.info(f"加载历史数据: {len(self.historical_data)} 条记录")
        except Exception as e:
            self.logger.error(f"加载历史数据失败: {e}")

    def _save_historical_data(self):
        """保存历史数据"""
        try:
            data = [h.to_dict() for h in self.historical_data]
            historical_file = "historical_training_data.json"
            with open(historical_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"历史数据已保存: {len(data)} 条记录")
        except Exception as e:
            self.logger.error(f"保存历史数据失败: {e}")

    def add_historical_data(self, historical: HistoricalData):
        """添加历史数据"""
        self.historical_data.append(historical)

        # 限制历史数据大小
        limit = self.config.get("historical_data_limit", 1000)
        if len(self.historical_data) > limit:
            self.historical_data = self.historical_data[-limit:]

        # 保存到文件
        self._save_historical_data()

        # 重新训练ML模型
        if self.config.get("enable_ml_predictions", True):
            self._retrain_ml_models()

    def _retrain_ml_models(self):
        """重新训练ML模型"""
        if len(self.historical_data) < 10:
            self.logger.info("历史数据不足，跳过模型训练")
            return

        try:
            # 准备特征和标签
            features = []
            labels_duration = []
            labels_memory = []
            labels_gpu = []

            for historical in self.historical_data:
                if not historical.success:
                    continue

                tp = historical.training_profile

                # 特征
                feature = [
                    math.log10(tp.model_parameters + 1),
                    math.log10(tp.batch_size + 1),
                    math.log10(tp.dataset_size + 1),
                    tp.epochs,
                    tp.learning_rate,
                    1 if tp.mixed_precision else 0,
                    1 if tp.distributed_training else 0,
                    tp.num_gpus,
                    tp.num_nodes,
                ]

                features.append(feature)

                # 标签
                labels_duration.append(math.log10(historical.actual_duration_hours + 1))

                # 资源使用
                if "memory_gb" in historical.actual_resource_usage:
                    labels_memory.append(
                        math.log10(historical.actual_resource_usage["memory_gb"] + 1)
                    )

                if "gpu_count" in historical.actual_resource_usage:
                    labels_gpu.append(historical.actual_resource_usage["gpu_count"])

            if len(features) < 5:
                return

            # 标准化特征
            X = np.array(features)
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)

            # 训练持续时间预测模型
            if len(labels_duration) >= 5:
                y_duration = np.array(labels_duration)
                model_duration = RandomForestRegressor(
                    n_estimators=100, random_state=42
                )
                model_duration.fit(X_scaled, y_duration)
                self.ml_models["duration"] = model_duration

            # 训练内存预测模型
            if len(labels_memory) >= 5:
                y_memory = np.array(labels_memory)
                model_memory = RandomForestRegressor(n_estimators=100, random_state=42)
                model_memory.fit(X_scaled, y_memory)
                self.ml_models["memory"] = model_memory

            # 训练GPU预测模型
            if len(labels_gpu) >= 5:
                y_gpu = np.array(labels_gpu)
                model_gpu = RandomForestRegressor(n_estimators=100, random_state=42)
                model_gpu.fit(X_scaled, y_gpu)
                self.ml_models["gpu"] = model_gpu

            self.logger.info("ML模型训练完成")

        except Exception as e:
            self.logger.error(f"训练ML模型失败: {e}")

    def predict(
        self,
        training_profile: TrainingProfile,
        available_resources: Optional[List[ResourceProfile]] = None,
        optimization_goal: Optional[OptimizationGoal] = None,
    ) -> PredictionResult:
        """预测训练资源需求

        参数:
            training_profile: 训练配置
            available_resources: 可用资源列表
            optimization_goal: 优化目标

        返回:
            PredictionResult: 预测结果
        """
        self.logger.info(
            f"开始预测: 模型参数={                 training_profile.model_parameters:,}, 数据集大小={                 training_profile.dataset_size:,}"
        )

        # 确定优化目标
        goal = optimization_goal or OptimizationGoal(
            self.config.get("optimization_goal", OptimizationGoal.BALANCED.value)
        )

        # 计算理论需求
        theoretical_requirements = self._compute_theoretical_requirements(
            training_profile
        )

        # 应用规则库
        rule_based_suggestions = self._apply_rules(training_profile)

        # ML预测
        ml_predictions = (
            self._ml_predict(training_profile)
            if self.config.get("enable_ml_predictions", True)
            else {}
        )

        # 结合预测结果
        combined_predictions = self._combine_predictions(
            theoretical_requirements, rule_based_suggestions, ml_predictions
        )

        # 考虑可用资源
        if available_resources:
            adjusted_predictions = self._adjust_for_available_resources(
                combined_predictions, available_resources
            )
        else:
            adjusted_predictions = combined_predictions

        # 优化建议
        optimization_suggestions = self._generate_optimization_suggestions(
            training_profile, adjusted_predictions, goal
        )

        # 风险评估
        risk_assessment = self._assess_risks(training_profile, adjusted_predictions)

        # 替代配置
        alternative_configs = self._generate_alternative_configs(
            training_profile, adjusted_predictions, goal
        )

        # 计算置信区间
        confidence_interval = self._calculate_confidence_interval(
            training_profile, adjusted_predictions
        )

        # 成本估算
        estimated_cost = self._estimate_cost(adjusted_predictions)

        # 构建结果
        result = PredictionResult(
            training_duration_hours=adjusted_predictions.get("duration_hours", 0),
            confidence_interval=confidence_interval,
            resource_requirements=adjusted_predictions.get("resource_requirements", {}),
            estimated_cost=estimated_cost,
            bottlenecks=adjusted_predictions.get("bottlenecks", []),
            optimization_suggestions=optimization_suggestions,
            risk_assessment=risk_assessment,
            alternative_configs=alternative_configs,
        )

        self.logger.info(
            f"预测完成: 预计时长={result.training_duration_hours:.1f} 小时"
        )

        return result

    def _compute_theoretical_requirements(
        self, training_profile: TrainingProfile
    ) -> Dict[str, Any]:
        """计算理论资源需求"""
        requirements = {}

        # 计算FLOPs
        total_flops = training_profile.compute_flops()
        requirements["total_flops"] = total_flops

        # 计算内存需求
        memory_req = training_profile.compute_memory_requirements()
        requirements["memory_requirements"] = memory_req

        # 估计训练时间
        gpu_flops = self.config["performance_models"]["gpu_flops"]
        gpu_count = training_profile.num_gpus * training_profile.num_nodes
        effective_flops = gpu_flops * gpu_count * 0.3  # 30% 利用率

        if effective_flops > 0:
            training_time_seconds = total_flops / effective_flops
            training_time_hours = training_time_seconds / 3600
        else:
            training_time_hours = 0

        requirements["duration_hours"] = training_time_hours
        requirements["effective_gpu_count"] = gpu_count

        # 存储需求
        # 检查点存储：模型大小 * 检查点频率
        model_size_gb = memory_req["parameters_gb"]
        checkpoint_storage_gb = (
            model_size_gb * training_profile.checkpoint_frequency * 2
        )  # 包含优化器状态

        # 数据集存储：近似估计
        dataset_storage_gb = training_profile.dataset_size * 0.001  # 假设每个样本1KB

        requirements["storage_requirements"] = {
            "model_gb": model_size_gb,
            "checkpoints_gb": checkpoint_storage_gb,
            "dataset_gb": dataset_storage_gb,
            "total_gb": model_size_gb + checkpoint_storage_gb + dataset_storage_gb,
        }

        # 网络需求（分布式训练）
        if training_profile.distributed_training:
            # 梯度同步数据量
            grad_sync_gb = memory_req["gradients_gb"] * training_profile.num_gpus

            # 每轮同步次数
            syncs_per_epoch = (
                training_profile.dataset_size / training_profile.batch_size
            )

            requirements["network_requirements"] = {
                "gradient_sync_gb_per_epoch": grad_sync_gb,
                "total_gradient_sync_gb": grad_sync_gb
                * syncs_per_epoch
                * training_profile.epochs,
                "estimated_bandwidth_requirement_gbps": 10,  # 假设10Gbps
            }

        return requirements

    def _apply_rules(self, training_profile: TrainingProfile) -> Dict[str, Any]:
        """应用规则库"""
        suggestions = []
        bottlenecks = []
        requirements = {}

        # 应用内存规则
        for rule in self.rules["memory_rules"]:
            if rule["condition"](training_profile):
                suggestions.append(
                    {
                        "type": "memory",
                        "action": rule["action"],
                        "message": rule["message"],
                        "min_memory_gb": rule.get("min_memory_gb", 0),
                    }
                )
                requirements["min_memory_gb"] = max(
                    requirements.get("min_memory_gb", 0), rule.get("min_memory_gb", 0)
                )

        # 应用GPU规则
        for rule in self.rules["gpu_rules"]:
            if rule["condition"](training_profile):
                suggestions.append(
                    {
                        "type": "gpu",
                        "action": rule["action"],
                        "message": rule["message"],
                        "min_gpus": rule.get("min_gpus", 0),
                    }
                )
                requirements["min_gpus"] = max(
                    requirements.get("min_gpus", 0), rule.get("min_gpus", 0)
                )

        # 应用时间规则
        for rule in self.rules["time_rules"]:
            if rule["condition"](training_profile):
                suggestions.append(
                    {
                        "type": "time",
                        "action": rule["action"],
                        "message": rule["message"],
                        "estimated_days": rule.get("estimated_days", 0),
                    }
                )
                bottlenecks.append("训练时间过长")

        # 应用优化规则
        for rule in self.rules["optimization_rules"]:
            if rule["condition"](training_profile):
                suggestions.append(
                    {
                        "type": "optimization",
                        "action": rule["action"],
                        "message": rule["message"],
                        "suggestion": rule.get("suggestion", ""),
                    }
                )

        return {
            "suggestions": suggestions,
            "requirements": requirements,
            "bottlenecks": bottlenecks,
        }

    def _ml_predict(self, training_profile: TrainingProfile) -> Dict[str, Any]:
        """使用ML模型预测"""
        if not self.ml_models:
            return {}  # 返回空字典

        try:
            # 准备特征
            feature = [
                math.log10(training_profile.model_parameters + 1),
                math.log10(training_profile.batch_size + 1),
                math.log10(training_profile.dataset_size + 1),
                training_profile.epochs,
                training_profile.learning_rate,
                1 if training_profile.mixed_precision else 0,
                1 if training_profile.distributed_training else 0,
                training_profile.num_gpus,
                training_profile.num_nodes,
            ]

            X = np.array([feature])
            X_scaled = self.scaler.transform(X)

            predictions = {}

            # 持续时间预测
            if "duration" in self.ml_models:
                y_pred_log = self.ml_models["duration"].predict(X_scaled)[0]
                predictions["duration_hours"] = 10**y_pred_log - 1

            # 内存预测
            if "memory" in self.ml_models:
                y_pred_log = self.ml_models["memory"].predict(X_scaled)[0]
                predictions["memory_gb"] = 10**y_pred_log - 1

            # GPU预测
            if "gpu" in self.ml_models:
                predictions["gpu_count"] = int(
                    self.ml_models["gpu"].predict(X_scaled)[0]
                )

            return predictions

        except Exception as e:
            self.logger.error(f"ML预测失败: {e}")
            return {}  # 返回空字典

    def _combine_predictions(
        self,
        theoretical: Dict[str, Any],
        rule_based: Dict[str, Any],
        ml_predictions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """结合多种预测方法"""
        combined = {
            "resource_requirements": {},
            "bottlenecks": rule_based.get("bottlenecks", []),
        }

        # 持续时间：优先使用ML预测，其次理论计算
        if "duration_hours" in ml_predictions:
            combined["duration_hours"] = ml_predictions["duration_hours"]
        elif "duration_hours" in theoretical:
            combined["duration_hours"] = theoretical["duration_hours"]

        # 内存需求：取最大值
        memory_values = []

        if "memory_requirements" in theoretical:
            memory_values.append(
                theoretical["memory_requirements"]["minimum_gpu_memory_gb"]
            )

        if "min_memory_gb" in rule_based.get("requirements", {}):
            memory_values.append(rule_based["requirements"]["min_memory_gb"])

        if "memory_gb" in ml_predictions:
            memory_values.append(ml_predictions["memory_gb"])

        if memory_values:
            combined["resource_requirements"]["memory_gb"] = max(memory_values)

        # GPU数量：取最大值
        gpu_values = []

        if "effective_gpu_count" in theoretical:
            gpu_values.append(theoretical["effective_gpu_count"])

        if "min_gpus" in rule_based.get("requirements", {}):
            gpu_values.append(rule_based["requirements"]["min_gpus"])

        if "gpu_count" in ml_predictions:
            gpu_values.append(ml_predictions["gpu_count"])

        if gpu_values:
            combined["resource_requirements"]["gpu_count"] = max(gpu_values)

        # 存储需求
        if "storage_requirements" in theoretical:
            combined["resource_requirements"]["storage_gb"] = theoretical[
                "storage_requirements"
            ]["total_gb"]

        # 网络需求
        if "network_requirements" in theoretical:
            combined["resource_requirements"]["network_gbps"] = theoretical[
                "network_requirements"
            ]["estimated_bandwidth_requirement_gbps"]

        # 添加理论FLOPs
        if "total_flops" in theoretical:
            combined["total_flops"] = theoretical["total_flops"]

        return combined

    def _adjust_for_available_resources(
        self, predictions: Dict[str, Any], available_resources: List[ResourceProfile]
    ) -> Dict[str, Any]:
        """根据可用资源调整预测"""
        adjusted = predictions.copy()

        # 分析可用资源
        available = {}
        for resource in available_resources:
            if resource.resource_type == ResourceType.GPU:
                available["gpu"] = available.get("gpu", 0) + resource.quantity
            elif resource.resource_type == ResourceType.MEMORY:
                available["memory_gb"] = (
                    available.get("memory_gb", 0) + resource.quantity
                )
            elif resource.resource_type == ResourceType.STORAGE:
                available["storage_gb"] = (
                    available.get("storage_gb", 0) + resource.quantity
                )
            elif resource.resource_type == ResourceType.NETWORK:
                available["network_gbps"] = max(
                    available.get("network_gbps", 0), resource.quantity
                )

        # 检查资源不足
        bottlenecks = predictions.get("bottlenecks", [])
        requirements = predictions.get("resource_requirements", {})

        # GPU检查
        if "gpu_count" in requirements:
            if available.get("gpu", 0) < requirements["gpu_count"]:
                bottlenecks.append(
                    f"GPU不足: 需要 {                         requirements['gpu_count']}，可用 {                         available.get(                             'gpu', 0)}"
                )
                # 调整训练时间（假设线性缩放）
                if available.get("gpu", 0) > 0:
                    scale_factor = requirements["gpu_count"] / available["gpu"]
                    if "duration_hours" in adjusted:
                        adjusted["duration_hours"] *= scale_factor

        # 内存检查
        if "memory_gb" in requirements:
            if available.get("memory_gb", 0) < requirements["memory_gb"]:
                bottlenecks.append(
                    f"内存不足: 需要 {                         requirements['memory_gb']:.1f} GB，可用 {                         available.get(                             'memory_gb',                             0):.1f} GB"
                )
                # 内存不足可能导致OOM，无法训练
                adjusted["duration_hours"] = float("inf")

        # 存储检查
        if "storage_gb" in requirements:
            if available.get("storage_gb", 0) < requirements["storage_gb"]:
                bottlenecks.append(
                    f"存储不足: 需要 {                         requirements['storage_gb']:.1f} GB，可用 {                         available.get(                             'storage_gb', 0):.1f} GB"
                )

        # 网络检查
        if "network_gbps" in requirements:
            if available.get("network_gbps", 0) < requirements["network_gbps"]:
                bottlenecks.append(
                    f"网络带宽不足: 需要 {                         requirements['network_gbps']} Gbps，可用 {                         available.get(                             'network_gbps', 0)} Gbps"
                )
                # 网络瓶颈可能增加训练时间
                if "duration_hours" in adjusted:
                    adjusted["duration_hours"] *= 1.2  # 增加20%

        adjusted["bottlenecks"] = bottlenecks

        return adjusted

    def _generate_optimization_suggestions(
        self,
        training_profile: TrainingProfile,
        predictions: Dict[str, Any],
        goal: OptimizationGoal,
    ) -> List[Dict[str, Any]]:
        """生成优化建议"""
        suggestions = []

        duration = predictions.get("duration_hours", 0)
        requirements = predictions.get("resource_requirements", {})

        # 根据优化目标生成建议
        if goal == OptimizationGoal.MINIMIZE_TIME:
            # 最小化时间
            if duration > 24:
                suggestions.append(
                    {
                        "type": "time_optimization",
                        "title": "训练时间过长",
                        "description": f"预计训练时间 {duration:.1f} 小时",
                        "suggestions": [
                            "增加GPU数量",
                            "使用混合精度训练",
                            "增加批量大小",
                            "使用梯度累积",
                            "优化数据加载管道",
                        ],
                        "priority": "high",
                    }
                )

            if requirements.get("memory_gb", 0) > 32:
                suggestions.append(
                    {
                        "type": "memory_optimization",
                        "title": "高内存需求",
                        "description": f"预计需要 {requirements['memory_gb']:.1f} GB 内存",
                        "suggestions": [
                            "使用梯度检查点",
                            "减少批量大小",
                            "使用模型并行",
                            "优化模型架构",
                        ],
                        "priority": "medium",
                    }
                )

        elif goal == OptimizationGoal.MINIMIZE_COST:
            # 最小化成本
            gpu_count = requirements.get("gpu_count", 1)
            gpu_hour_cost = self.config["resource_costs"]["gpu_hour"]
            estimated_cost = duration * gpu_count * gpu_hour_cost

            if estimated_cost > 1000:
                suggestions.append(
                    {
                        "type": "cost_optimization",
                        "title": "训练成本过高",
                        "description": f"预计成本 ${estimated_cost:.2f}",
                        "suggestions": [
                            "使用更便宜的GPU实例",
                            "使用竞价实例",
                            "优化训练效率减少时间",
                            "使用更小的模型",
                            "使用迁移学习",
                        ],
                        "priority": "high",
                    }
                )

        elif goal == OptimizationGoal.MINIMIZE_RESOURCES:
            # 最小化资源
            if requirements.get("gpu_count", 1) > 4:
                suggestions.append(
                    {
                        "type": "resource_optimization",
                        "title": "GPU资源需求高",
                        "description": f"需要 {requirements['gpu_count']} 个GPU",
                        "suggestions": [
                            "使用模型压缩",
                            "使用知识蒸馏",
                            "使用量化训练",
                            "使用更高效的模型架构",
                        ],
                        "priority": "high",
                    }
                )

        # 通用优化建议
        if not training_profile.mixed_precision:
            suggestions.append(
                {
                    "type": "general_optimization",
                    "title": "启用混合精度训练",
                    "description": "混合精度训练可减少内存使用并加速训练",
                    "suggestions": ["启用混合精度训练"],
                    "priority": "medium",
                }
            )

        if training_profile.batch_size < 32:
            suggestions.append(
                {
                    "type": "general_optimization",
                    "title": "增加批量大小",
                    "description": f"当前批量大小 {training_profile.batch_size} 较小，可能影响GPU利用率",
                    "suggestions": ["增加批量大小到32或更大"],
                    "priority": "low",
                }
            )

        return suggestions

    def _assess_risks(
        self, training_profile: TrainingProfile, predictions: Dict[str, Any]
    ) -> Dict[str, float]:
        """风险评估"""
        risks = {}

        duration = predictions.get("duration_hours", 0)
        bottlenecks = predictions.get("bottlenecks", [])

        # 时间风险
        if duration > 168:  # 超过一周
            risks["time_risk"] = 0.9
        elif duration > 72:  # 超过3天
            risks["time_risk"] = 0.7
        elif duration > 24:  # 超过1天
            risks["time_risk"] = 0.5
        else:
            risks["time_risk"] = 0.2

        # 资源风险
        if any("不足" in b for b in bottlenecks):
            risks["resource_risk"] = 0.8
        else:
            risks["resource_risk"] = 0.3

        # 成本风险
        gpu_count = predictions.get("resource_requirements", {}).get("gpu_count", 1)
        gpu_hour_cost = self.config["resource_costs"]["gpu_hour"]
        estimated_cost = duration * gpu_count * gpu_hour_cost

        if estimated_cost > 10000:
            risks["cost_risk"] = 0.9
        elif estimated_cost > 1000:
            risks["cost_risk"] = 0.7
        elif estimated_cost > 100:
            risks["cost_risk"] = 0.4
        else:
            risks["cost_risk"] = 0.1

        # 失败风险（基于历史数据）
        if len(self.historical_data) > 0:
            failure_rate = sum(1 for h in self.historical_data if not h.success) / len(
                self.historical_data
            )
            risks["failure_risk"] = failure_rate
        else:
            risks["failure_risk"] = 0.2

        # 总体风险（加权平均）
        weights = {
            "time_risk": 0.3,
            "resource_risk": 0.3,
            "cost_risk": 0.2,
            "failure_risk": 0.2,
        }

        overall_risk = sum(risks.get(k, 0) * weights.get(k, 0) for k in weights)
        risks["overall_risk"] = overall_risk

        return risks

    def _generate_alternative_configs(
        self,
        training_profile: TrainingProfile,
        predictions: Dict[str, Any],
        goal: OptimizationGoal,
    ) -> List[Dict[str, Any]]:
        """生成替代配置"""
        alternatives = []

        original_duration = predictions.get("duration_hours", 0)
        original_gpu_count = predictions.get("resource_requirements", {}).get(
            "gpu_count", 1
        )

        # 配置1：增加GPU减少时间
        if original_gpu_count < 8:
            alt_gpu_count = min(original_gpu_count * 2, 8)
            alt_duration = original_duration * original_gpu_count / alt_gpu_count

            alternatives.append(
                {
                    "name": "增加GPU配置",
                    "description": f"增加GPU到 {alt_gpu_count} 个以加速训练",
                    "changes": {
                        "num_gpus": alt_gpu_count,
                    },
                    "estimated_duration_hours": alt_duration,
                    "estimated_cost_change": (
                        alt_duration * alt_gpu_count
                        - original_duration * original_gpu_count
                    )
                    * self.config["resource_costs"]["gpu_hour"],
                    "suitability": (
                        "minimize_time"
                        if goal == OptimizationGoal.MINIMIZE_TIME
                        else "balanced"
                    ),
                }
            )

        # 配置2：减少批量大小降低内存
        if training_profile.batch_size > 32:
            alt_batch_size = training_profile.batch_size // 2
            # 粗略估计：批量大小减半，迭代次数加倍
            alt_duration = original_duration * 2

            alternatives.append(
                {
                    "name": "减少批量大小",
                    "description": f"减少批量大小到 {alt_batch_size} 以降低内存需求",
                    "changes": {
                        "batch_size": alt_batch_size,
                    },
                    "estimated_duration_hours": alt_duration,
                    "estimated_cost_change": (alt_duration - original_duration)
                    * original_gpu_count
                    * self.config["resource_costs"]["gpu_hour"],
                    "suitability": (
                        "minimize_resources"
                        if goal == OptimizationGoal.MINIMIZE_RESOURCES
                        else "balanced"
                    ),
                }
            )

        # 配置3：使用混合精度
        if not training_profile.mixed_precision:
            # 混合精度可减少内存并可能加速
            alt_duration = original_duration * 0.7  # 估计加速30%

            alternatives.append(
                {
                    "name": "启用混合精度",
                    "description": "启用混合精度训练以减少内存使用并加速",
                    "changes": {
                        "mixed_precision": True,
                    },
                    "estimated_duration_hours": alt_duration,
                    "estimated_cost_change": (alt_duration - original_duration)
                    * original_gpu_count
                    * self.config["resource_costs"]["gpu_hour"],
                    "suitability": "balanced",
                }
            )

        return alternatives

    def _calculate_confidence_interval(
        self, training_profile: TrainingProfile, predictions: Dict[str, Any]
    ) -> Tuple[float, float]:
        """计算置信区间"""
        duration = predictions.get("duration_hours", 0)

        if len(self.historical_data) < 5:
            # 数据不足，使用宽泛的置信区间
            return (duration * 0.5, duration * 2.0)

        # 计算相似历史训练的误差
        errors = []
        for historical in self.historical_data:
            if historical.success:
                # 简单相似度计算
                similarity = self._compute_similarity(
                    training_profile, historical.training_profile
                )
                if similarity > 0.7:
                    # 计算相对误差
                    if (
                        "duration_hours" in predictions
                        and historical.actual_duration_hours > 0
                    ):
                        error = (
                            abs(
                                predictions["duration_hours"]
                                - historical.actual_duration_hours
                            )
                            / historical.actual_duration_hours
                        )
                        errors.append(error)

        if errors:
            mean_error = statistics.mean(errors)
            std_error = (
                statistics.stdev(errors) if len(errors) > 1 else mean_error * 0.5
            )

            self.config.get("confidence_level", 0.95)
            z_score = 1.96  # 95%置信度

            margin = z_score * std_error
            lower = duration * (1 - margin)
            upper = duration * (1 + margin)

            return (max(0, lower), upper)
        else:
            return (duration * 0.7, duration * 1.3)

    def _compute_similarity(self, tp1: TrainingProfile, tp2: TrainingProfile) -> float:
        """计算两个训练配置的相似度"""
        similarities = []

        # 模型参数相似度
        if tp1.model_parameters > 0 and tp2.model_parameters > 0:
            param_sim = min(tp1.model_parameters, tp2.model_parameters) / max(
                tp1.model_parameters, tp2.model_parameters
            )
            similarities.append(param_sim)

        # 数据集大小相似度
        if tp1.dataset_size > 0 and tp2.dataset_size > 0:
            data_sim = min(tp1.dataset_size, tp2.dataset_size) / max(
                tp1.dataset_size, tp2.dataset_size
            )
            similarities.append(data_sim)

        # 批量大小相似度
        if tp1.batch_size > 0 and tp2.batch_size > 0:
            batch_sim = min(tp1.batch_size, tp2.batch_size) / max(
                tp1.batch_size, tp2.batch_size
            )
            similarities.append(batch_sim)

        # GPU数量相似度
        gpu_sim = (
            min(tp1.num_gpus, tp2.num_gpus) / max(tp1.num_gpus, tp2.num_gpus)
            if max(tp1.num_gpus, tp2.num_gpus) > 0
            else 1.0
        )
        similarities.append(gpu_sim)

        # 平均值
        return statistics.mean(similarities) if similarities else 0.0

    def _estimate_cost(self, predictions: Dict[str, Any]) -> float:
        """估算成本"""
        duration = predictions.get("duration_hours", 0)
        requirements = predictions.get("resource_requirements", {})

        total_cost = 0.0

        # GPU成本
        gpu_count = requirements.get("gpu_count", 1)
        gpu_hour_cost = self.config["resource_costs"]["gpu_hour"]
        total_cost += duration * gpu_count * gpu_hour_cost

        # CPU成本
        cpu_count = requirements.get("cpu_count", 8)  # 默认8个CPU
        cpu_hour_cost = self.config["resource_costs"]["cpu_hour"]
        total_cost += duration * cpu_count * cpu_hour_cost

        # 内存成本
        memory_gb = requirements.get("memory_gb", 32)  # 默认32GB
        memory_hour_cost = self.config["resource_costs"]["memory_gb_hour"]
        total_cost += duration * memory_gb * memory_hour_cost

        # 存储成本
        storage_gb = requirements.get("storage_gb", 100)  # 默认100GB
        storage_hour_cost = self.config["resource_costs"]["storage_gb_hour"]
        total_cost += duration * storage_gb * storage_hour_cost

        return total_cost


# 全局实例
_resource_predictor_instance = None


def get_resource_predictor(
    config: Optional[Dict[str, Any]] = None,
) -> ResourcePredictor:
    """获取资源预测器单例

    参数:
        config: 配置字典

    返回:
        ResourcePredictor: 资源预测器实例
    """
    global _resource_predictor_instance

    if _resource_predictor_instance is None:
        _resource_predictor_instance = ResourcePredictor(config)

    return _resource_predictor_instance


__all__ = [
    "ResourcePredictor",
    "get_resource_predictor",
    "ResourceProfile",
    "TrainingProfile",
    "HistoricalData",
    "PredictionResult",
    "ResourceType",
    "PredictionMethod",
    "OptimizationGoal",
]
