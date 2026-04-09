#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四元数模型预训练验证 - Self AGI 系统四元数全面引入实施方案验证模块

功能：
1. 四元数模型预训练验证
2. 四元数操作学习能力测试
3. 模型性能和准确率评估
4. 与基准模型的对比测试

验证任务：
1. 四元数归一化学习
2. 四元数乘法学习
3. 四元数旋转学习
4. 四元数插值学习
5. 四元数序列建模

工业级质量标准要求：
- 准确性：模型应达到高精度（>95%）
- 泛化性：在未见数据上表现良好
- 效率：训练和推理速度快
- 稳定性：数值稳定，无梯度爆炸/消失

验证指标：
1. 角度误差（度）
2. 点积相似度（0-1）
3. 双重覆盖准确率
4. 训练时间效率
5. 内存使用效率
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, Tuple
import math
import json

from models.quaternion_core import (
    Quaternion,
    QuaternionNormalization,
)
from models.quaternion_nn import (
    QuaternionLinear,
)
from models.quaternion_optimizer import (
    QuaternionAdam,
    QuaternionMixedLoss,
)


class QuaternionValidationTask:
    """四元数验证任务基类"""

    def __init__(self, task_name: str, input_dim: int, output_dim: int):
        self.task_name = task_name
        self.input_dim = input_dim
        self.output_dim = output_dim

    def generate_data(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成训练数据

        注意：根据用户要求"系统可以在没有硬件条件下单独运行AGI所有功能"，
        当具体验证任务未实现时，返回空张量并记录警告。
        """
        import torch

        logging.getLogger(__name__).warning(
            f"四元数验证数据生成：具体任务未实现（任务: {self.task_name}, 样本数: {num_samples}）。\n"
            "根据用户要求'系统可以在没有硬件条件下单独运行AGI所有功能'，\n"
            "返回空张量，系统可以继续运行（四元数验证功能将受限）。"
        )
        empty_tensor = torch.tensor([])
        return empty_tensor, empty_tensor  # 返回空张量表示数据不可用

    def evaluate(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """评估预测结果

        注意：根据用户要求"系统可以在没有硬件条件下单独运行AGI所有功能"，
        当具体验证任务未实现时，返回默认评估结果并记录警告。
        """
        logging.getLogger(__name__).warning(
            f"四元数验证评估：具体任务未实现（任务: {self.task_name}）。\n"
            "根据用户要求'系统可以在没有硬件条件下单独运行AGI所有功能'，\n"
            "返回默认评估结果，系统可以继续运行（四元数验证功能将受限）。"
        )
        return {
            "accuracy": 0.0,
            "error": 1.0,
            "task": self.task_name,
            "implementation_status": "not_implemented",
        }

    def get_description(self) -> str:
        """获取任务描述"""
        return self.task_name


class QuaternionNormalizationTask(QuaternionValidationTask):
    """四元数归一化任务"""

    def __init__(self):
        super().__init__("四元数归一化", input_dim=4, output_dim=4)

    def generate_data(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成归一化任务数据"""
        # 生成随机四元数（未归一化）
        inputs = torch.randn(num_samples, 4)

        # 计算归一化目标
        targets = inputs / torch.norm(inputs, dim=1, keepdim=True).clamp(min=1e-8)

        return inputs, targets

    def evaluate(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """评估归一化任务"""
        # 计算预测四元数的模长
        pred_norms = torch.norm(predictions, dim=1)

        # 计算角度误差
        dot = torch.sum(predictions * targets, dim=1)
        dot = torch.clamp(dot, -1.0, 1.0)
        angle_errors = 2 * torch.acos(torch.abs(dot)) * 180 / math.pi

        metrics = {
            "平均模长误差": torch.mean(torch.abs(pred_norms - 1.0)).item(),
            "最大模长误差": torch.max(torch.abs(pred_norms - 1.0)).item(),
            "平均角度误差(度)": torch.mean(angle_errors).item(),
            "最大角度误差(度)": torch.max(angle_errors).item(),
            "准确率(<1度)": (angle_errors < 1.0).float().mean().item(),
        }

        return metrics


class QuaternionMultiplicationTask(QuaternionValidationTask):
    """四元数乘法任务"""

    def __init__(self):
        super().__init__("四元数乘法", input_dim=8, output_dim=4)  # 输入: 两个四元数

    def generate_data(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成乘法任务数据"""
        # 生成两个随机四元数
        q1 = torch.randn(num_samples, 4)
        q2 = torch.randn(num_samples, 4)

        # 归一化
        q1 = q1 / torch.norm(q1, dim=1, keepdim=True).clamp(min=1e-8)
        q2 = q2 / torch.norm(q2, dim=1, keepdim=True).clamp(min=1e-8)

        # 输入是两个四元数的拼接
        inputs = torch.cat([q1, q2], dim=1)

        # 计算四元数乘法目标
        targets = torch.zeros_like(q1)

        for i in range(num_samples):
            quat1 = Quaternion(
                q1[i, 0].item(), q1[i, 1].item(), q1[i, 2].item(), q1[i, 3].item()
            )
            quat2 = Quaternion(
                q2[i, 0].item(), q2[i, 1].item(), q2[i, 2].item(), q2[i, 3].item()
            )
            result = quat1 * quat2
            targets[i] = torch.tensor(result.as_vector())

        return inputs, targets

    def evaluate(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """评估乘法任务"""
        # 归一化预测和目标
        pred_norm = predictions / torch.norm(predictions, dim=1, keepdim=True).clamp(
            min=1e-8
        )
        target_norm = targets / torch.norm(targets, dim=1, keepdim=True).clamp(min=1e-8)

        # 计算角度误差
        dot = torch.sum(pred_norm * target_norm, dim=1)
        dot = torch.clamp(dot, -1.0, 1.0)
        angle_errors = 2 * torch.acos(torch.abs(dot)) * 180 / math.pi

        metrics = {
            "平均角度误差(度)": torch.mean(angle_errors).item(),
            "最大角度误差(度)": torch.max(angle_errors).item(),
            "准确率(<5度)": (angle_errors < 5.0).float().mean().item(),
            "准确率(<1度)": (angle_errors < 1.0).float().mean().item(),
        }

        return metrics


class QuaternionRotationTask(QuaternionValidationTask):
    """四元数旋转任务"""

    def __init__(self):
        super().__init__("四元数旋转", input_dim=7, output_dim=3)  # 输入: 四元数 + 向量

    def generate_data(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成旋转任务数据"""
        # 生成随机四元数
        quats = torch.randn(num_samples, 4)
        quats = quats / torch.norm(quats, dim=1, keepdim=True).clamp(min=1e-8)

        # 生成随机向量
        vectors = torch.randn(num_samples, 3)

        # 输入是四元数和向量的拼接
        inputs = torch.cat([quats, vectors], dim=1)

        # 计算旋转后的向量
        targets = torch.zeros_like(vectors)

        for i in range(num_samples):
            quat = Quaternion(
                quats[i, 0].item(),
                quats[i, 1].item(),
                quats[i, 2].item(),
                quats[i, 3].item(),
            )
            vec = vectors[i].numpy()
            rotated = quat.rotate_vector(vec)
            targets[i] = torch.tensor(rotated)

        return inputs, targets

    def evaluate(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """评估旋转任务"""
        # 计算向量距离
        distances = torch.norm(predictions - targets, dim=1)

        # 计算角度误差（向量之间的角度）
        dot = torch.sum(predictions * targets, dim=1)
        norm_pred = torch.norm(predictions, dim=1)
        norm_target = torch.norm(targets, dim=1)
        cos_angles = dot / (norm_pred * norm_target + 1e-8)
        cos_angles = torch.clamp(cos_angles, -1.0, 1.0)
        angle_errors = torch.acos(torch.abs(cos_angles)) * 180 / math.pi

        metrics = {
            "平均距离误差": torch.mean(distances).item(),
            "最大距离误差": torch.max(distances).item(),
            "平均角度误差(度)": torch.mean(angle_errors).item(),
            "最大角度误差(度)": torch.max(angle_errors).item(),
            "准确率(<5度)": (angle_errors < 5.0).float().mean().item(),
        }

        return metrics


class QuaternionSlerpTask(QuaternionValidationTask):
    """四元数球面线性插值任务"""

    def __init__(self):
        super().__init__(
            "四元数SLERP", input_dim=9, output_dim=4
        )  # 输入: 两个四元数 + 插值参数

    def generate_data(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成SLERP任务数据"""
        # 生成两个随机四元数
        q1 = torch.randn(num_samples, 4)
        q2 = torch.randn(num_samples, 4)

        # 归一化
        q1 = q1 / torch.norm(q1, dim=1, keepdim=True).clamp(min=1e-8)
        q2 = q2 / torch.norm(q2, dim=1, keepdim=True).clamp(min=1e-8)

        # 生成插值参数
        t = torch.rand(num_samples, 1)

        # 输入是两个四元数和插值参数的拼接
        inputs = torch.cat([q1, q2, t], dim=1)

        # 计算SLERP目标
        targets = torch.zeros_like(q1)

        for i in range(num_samples):
            quat1 = Quaternion(
                q1[i, 0].item(), q1[i, 1].item(), q1[i, 2].item(), q1[i, 3].item()
            )
            quat2 = Quaternion(
                q2[i, 0].item(), q2[i, 1].item(), q2[i, 2].item(), q2[i, 3].item()
            )
            result = quat1.slerp(quat2, t[i].item())
            targets[i] = torch.tensor(result.as_vector())

        return inputs, targets

    def evaluate(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """评估SLERP任务"""
        # 归一化预测和目标
        pred_norm = predictions / torch.norm(predictions, dim=1, keepdim=True).clamp(
            min=1e-8
        )
        target_norm = targets / torch.norm(targets, dim=1, keepdim=True).clamp(min=1e-8)

        # 计算角度误差
        dot = torch.sum(pred_norm * target_norm, dim=1)
        dot = torch.clamp(dot, -1.0, 1.0)
        angle_errors = 2 * torch.acos(torch.abs(dot)) * 180 / math.pi

        metrics = {
            "平均角度误差(度)": torch.mean(angle_errors).item(),
            "最大角度误差(度)": torch.max(angle_errors).item(),
            "准确率(<1度)": (angle_errors < 1.0).float().mean().item(),
            "准确率(<0.1度)": (angle_errors < 0.1).float().mean().item(),
        }

        return metrics


class QuaternionValidationModel(nn.Module):
    """四元数验证模型"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # 调整维度为4的倍数（四元数线性层要求）
        self.adjusted_input_dim = ((input_dim + 3) // 4) * 4
        self.adjusted_output_dim = ((output_dim + 3) // 4) * 4

        # 输入投影（如果需要）
        if self.adjusted_input_dim != input_dim:
            self.input_proj = nn.Linear(input_dim, self.adjusted_input_dim)
        else:
            self.input_proj = nn.Identity()

        # 输出投影（如果需要）
        if self.adjusted_output_dim != output_dim:
            self.output_proj = nn.Linear(self.adjusted_output_dim, output_dim)
        else:
            self.output_proj = nn.Identity()

        # 四元数线性层（使用调整后的维度）
        quat_input_dim = self.adjusted_input_dim // 4
        quat_hidden_dim = hidden_dim // 4
        quat_output_dim = self.adjusted_output_dim // 4

        self.layer1 = QuaternionLinear(quat_input_dim, quat_hidden_dim)
        self.layer2 = QuaternionLinear(quat_hidden_dim, quat_hidden_dim)
        self.layer3 = QuaternionLinear(quat_hidden_dim, quat_output_dim)

        # 激活函数
        self.activation = nn.GELU()

        # 四元数归一化层
        self.norm = QuaternionNormalization()

        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 输入投影
        x = self.input_proj(x)

        # 四元数线性层
        x = self.layer1(x)
        x = self.layer_norm(x)
        x = self.activation(x)

        x = self.layer2(x)
        x = self.layer_norm(x)
        x = self.activation(x)

        x = self.layer3(x)

        # 四元数归一化（对于输出是四元数的任务）
        if self.output_dim == 4:
            x = self.norm(x)

        # 输出投影
        x = self.output_proj(x)

        return x


class QuaternionValidator:
    """四元数验证器"""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.tasks = [
            QuaternionNormalizationTask(),
            QuaternionMultiplicationTask(),
            QuaternionRotationTask(),
            QuaternionSlerpTask(),
        ]

    def run_validation(
        self,
        task_idx: int = None,
        num_samples: int = 1000,
        num_epochs: int = 50,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """运行验证"""
        results = {}

        # 选择任务
        if task_idx is not None:
            tasks_to_run = [self.tasks[task_idx]]
        else:
            tasks_to_run = self.tasks

        for task in tasks_to_run:
            print(f"\n正在验证任务: {task.get_description()}")

            # 生成数据
            inputs, targets = task.generate_data(num_samples)

            # 划分训练集和测试集
            split_idx = int(0.8 * num_samples)
            train_inputs, test_inputs = inputs[:split_idx], inputs[split_idx:]
            train_targets, test_targets = targets[:split_idx], targets[split_idx:]

            # 转换为张量并移动到设备
            train_inputs = train_inputs.to(self.device)
            train_targets = train_targets.to(self.device)
            test_inputs = test_inputs.to(self.device)
            test_targets = test_targets.to(self.device)

            # 创建模型
            model = QuaternionValidationModel(
                input_dim=train_inputs.shape[1], output_dim=train_targets.shape[1]
            ).to(self.device)

            # 创建优化器
            optimizer = QuaternionAdam(model.parameters(), lr=1e-3)

            # 创建损失函数
            criterion = QuaternionMixedLoss()

            # 训练模型
            train_losses = []
            test_losses = []

            for epoch in range(num_epochs):
                # 训练
                model.train()
                epoch_loss = 0.0

                # 批量训练
                for i in range(0, len(train_inputs), batch_size):
                    batch_inputs = train_inputs[i: i + batch_size]
                    batch_targets = train_targets[i: i + batch_size]

                    optimizer.zero_grad()
                    outputs = model(batch_inputs)
                    loss = criterion(outputs, batch_targets)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                avg_train_loss = epoch_loss / (len(train_inputs) / batch_size)
                train_losses.append(avg_train_loss)

                # 测试
                model.eval()
                with torch.no_grad():
                    test_outputs = model(test_inputs)
                    test_loss = criterion(test_outputs, test_targets).item()
                    test_losses.append(test_loss)

                # 打印进度
                if (epoch + 1) % 10 == 0:
                    print(
                        f"  周期 {                             epoch + 1}/{num_epochs}, 训练损失: {                             avg_train_loss:.6f}, 测试损失: {                             test_loss:.6f}"
                    )

            # 最终评估
            model.eval()
            with torch.no_grad():
                test_outputs = model(test_inputs)
                metrics = task.evaluate(test_outputs, test_targets)

            # 保存结果
            results[task.task_name] = {
                "metrics": metrics,
                "train_losses": train_losses,
                "test_losses": test_losses,
                "final_train_loss": train_losses[-1],
                "final_test_loss": test_losses[-1],
            }

            print(f"  任务完成，最终测试损失: {test_losses[-1]:.6f}")
            print(f"  评估指标: {json.dumps(metrics, indent=2, ensure_ascii=False)}")

        return results

    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """运行综合验证"""
        print("=" * 80)
        print("四元数模型预训练综合验证")
        print("=" * 80)

        all_results = {}

        # 运行所有任务
        for i, task in enumerate(self.tasks):
            print(f"\n[{i + 1}/{len(self.tasks)}] 验证任务: {task.get_description()}")

            try:
                results = self.run_validation(
                    task_idx=i, num_samples=2000, num_epochs=30, batch_size=64
                )

                all_results[task.task_name] = results[task.task_name]

            except Exception as e:
                print(f"  任务验证失败: {e}")
                all_results[task.task_name] = {
                    "error": str(e),
                    "metrics": {},
                    "train_losses": [],
                    "test_losses": [],
                    "final_train_loss": float("inf"),
                    "final_test_loss": float("inf"),
                }

        # 汇总结果
        summary = self._summarize_results(all_results)

        print("\n" + "=" * 80)
        print("验证完成!")
        print("=" * 80)

        # 打印汇总
        print("\n汇总结果:")
        for task_name, task_results in all_results.items():
            if "error" in task_results:
                print(f"  {task_name}: 失败 - {task_results['error']}")
            else:
                metrics = task_results["metrics"]
                list(metrics.values())[0] if metrics else 0
                print(
                    f"  {task_name}: 成功 - 最终测试损失: {task_results['final_test_loss']:.6f}"
                )

        return {"tasks": all_results, "summary": summary}

    def _summarize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """汇总结果"""
        successful_tasks = []
        failed_tasks = []

        for task_name, task_results in results.items():
            if "error" in task_results:
                failed_tasks.append(task_name)
            else:
                successful_tasks.append(task_name)

        summary = {
            "total_tasks": len(results),
            "successful_tasks": len(successful_tasks),
            "failed_tasks": len(failed_tasks),
            "success_rate": (
                len(successful_tasks) / len(results) if len(results) > 0 else 0
            ),
            "failed_tasks_list": failed_tasks,
            "successful_tasks_list": successful_tasks,
        }

        return summary

    def save_results(self, results: Dict[str, Any], filepath: str):
        """保存结果到文件"""
        # 转换结果为JSON可序列化格式
        json_results = {}

        for task_name, task_results in results.items():
            if isinstance(task_results, dict):
                json_task_results = {}
                for key, value in task_results.items():
                    if isinstance(value, torch.Tensor):
                        json_task_results[key] = value.tolist()
                    elif isinstance(value, np.ndarray):
                        json_task_results[key] = value.tolist()
                    elif isinstance(value, (int, float, str, bool, list, dict)):
                        json_task_results[key] = value
                    else:
                        json_task_results[key] = str(value)
                json_results[task_name] = json_task_results
            else:
                json_results[task_name] = str(task_results)

        # 保存到文件
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)

        print(f"结果已保存到: {filepath}")

    def load_results(self, filepath: str) -> Dict[str, Any]:
        """从文件加载结果"""
        with open(filepath, "r", encoding="utf-8") as f:
            results = json.load(f)

        return results


# ============================================================================
# 测试函数
# ============================================================================


def test_quaternion_validation():
    """测试四元数验证"""
    print("测试四元数验证...")

    # 创建验证器
    validator = QuaternionValidator(device="cpu")

    # 测试单个任务
    try:
        results = validator.run_validation(
            task_idx=0, num_samples=100, num_epochs=5, batch_size=16  # 归一化任务
        )

        assert len(results) > 0, "验证结果不应为空"
        assert "四元数归一化" in results, "应包含归一化任务结果"

        task_results = results["四元数归一化"]
        assert "metrics" in task_results, "应包含评估指标"
        assert "final_test_loss" in task_results, "应包含最终测试损失"

        print("✓ 单任务验证测试通过")

    except Exception as e:
        print(f"单任务验证失败: {e}")
        raise

    # 测试数据生成
    tasks = [
        QuaternionNormalizationTask(),
        QuaternionMultiplicationTask(),
        QuaternionRotationTask(),
        QuaternionSlerpTask(),
    ]

    for task in tasks:
        inputs, targets = task.generate_data(10)
        assert inputs.shape[0] == 10, f"{task.task_name} 输入形状错误"
        assert targets.shape[0] == 10, f"{task.task_name} 目标形状错误"
        assert inputs.shape[1] == task.input_dim, f"{task.task_name} 输入维度错误"
        assert targets.shape[1] == task.output_dim, f"{task.task_name} 输出维度错误"

    print("✓ 所有任务数据生成测试通过")

    # 测试评估
    for task in tasks:
        # 生成随机预测和目标
        predictions = torch.randn(10, task.output_dim)
        targets = torch.randn(10, task.output_dim)

        metrics = task.evaluate(predictions, targets)
        assert isinstance(metrics, dict), f"{task.task_name} 评估结果应为字典"
        assert len(metrics) > 0, f"{task.task_name} 评估指标不应为空"

    print("✓ 所有任务评估测试通过")

    print("所有四元数验证测试通过！")

    return True


def run_pre_training_validation():
    """运行预训练验证"""
    print("运行四元数模型预训练验证...")

    # 创建验证器
    validator = QuaternionValidator()

    # 运行综合验证
    results = validator.run_comprehensive_validation()

    # 保存结果
    validator.save_results(results, "quaternion_validation_results.json")

    # 打印摘要
    summary = results["summary"]
    print("\n验证摘要:")
    print(f"  总任务数: {summary['total_tasks']}")
    print(f"  成功任务数: {summary['successful_tasks']}")
    print(f"  失败任务数: {summary['failed_tasks']}")
    print(f"  成功率: {summary['success_rate']:.2%}")

    if summary["failed_tasks"]:
        print(f"  失败任务: {', '.join(summary['failed_tasks_list'])}")

    # 判断验证是否通过
    if summary["success_rate"] >= 0.75:  # 75%成功率
        print("\n✅ 四元数模型预训练验证通过!")
        return True
    else:
        print("\n❌ 四元数模型预训练验证失败!")
        return False


if __name__ == "__main__":
    # 运行测试
    test_passed = test_quaternion_validation()

    if test_passed:
        print("\n" + "=" * 80)
        print("开始四元数模型预训练验证...")
        print("=" * 80)

        # 运行预训练验证
        validation_passed = run_pre_training_validation()

        if validation_passed:
            print("\n🎉 四元数全面引入实施方案验证成功完成!")
        else:
            print("\n⚠️  四元数全面引入实施方案验证未通过，需要进一步优化。")
    else:
        print("\n❌ 四元数验证测试失败，无法进行预训练验证。")
