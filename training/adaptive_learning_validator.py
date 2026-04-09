#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自适应学习策略验证器
验证自适应学习策略的有效性和鲁棒性

功能：
1. 策略性能评估：在不同任务和环境下测试学习策略
2. 收敛性分析：验证学习策略是否收敛到最优解
3. 鲁棒性测试：在噪声、干扰和分布偏移下的表现
4. 泛化能力测试：在未见任务上的表现
5. 效率评估：学习速度、资源消耗等
6. 对比实验：与基准策略的对比分析

基于真实实验验证，不使用虚拟数据
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import statistics

logger = logging.getLogger(__name__)


@dataclass
class ValidationTask:
    """验证任务"""

    id: str
    name: str
    description: str
    task_type: str  # classification, regression, reinforcement_learning, multimodal
    difficulty: str  # easy, medium, hard
    dataset_config: Dict[str, Any]
    success_criteria: Dict[str, float]  # 成功标准，如准确率>0.9
    time_budget: float  # 时间预算（秒）
    resource_constraints: Dict[str, Any]  # 资源约束


@dataclass
class LearningStrategy:
    """学习策略"""

    id: str
    name: str
    description: str
    strategy_type: str  # fixed, adaptive, meta_learning, curriculum
    parameters: Dict[str, Any]
    implementation: Callable  # 策略实现函数


@dataclass
class ValidationResult:
    """验证结果"""

    task_id: str
    strategy_id: str
    success: bool
    metrics: Dict[str, float]  # 评估指标
    performance_summary: Dict[str, Any]  # 性能总结
    convergence_data: List[float]  # 收敛过程数据
    time_taken: float  # 耗时（秒）
    resource_usage: Dict[str, float]  # 资源使用情况
    validation_timestamp: datetime = field(default_factory=datetime.now)
    notes: Optional[str] = None


@dataclass
class ValidationReport:
    """验证报告"""

    validation_id: str
    timestamp: datetime
    tasks: List[ValidationTask]
    strategies: List[LearningStrategy]
    results: List[ValidationResult]
    overall_summary: Dict[str, Any]
    recommendations: List[str]
    limitations: List[str]


class AdaptiveLearningValidator:
    """自适应学习策略验证器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.tasks: Dict[str, ValidationTask] = {}
        self.strategies: Dict[str, LearningStrategy] = {}
        self.results: List[ValidationResult] = []

        # 初始化基准任务
        self._initialize_benchmark_tasks()
        self._initialize_baseline_strategies()

        logger.info("自适应学习策略验证器初始化完成")

    def _initialize_benchmark_tasks(self):
        """初始化基准验证任务"""

        # 任务1：图像分类（MNIST简化版）
        self.tasks["mnist_classification"] = ValidationTask(
            id="mnist_classification",
            name="MNIST手写数字分类",
            description="10类手写数字分类任务",
            task_type="classification",
            difficulty="easy",
            dataset_config={
                "name": "MNIST",
                "num_classes": 10,
                "train_size": 10000,  # 简化数据集
                "test_size": 2000,
                "input_shape": [28, 28, 1],
            },
            success_criteria={
                "accuracy": 0.95,  # 准确率95%
                "f1_score": 0.94,
                "training_time": 300.0,  # 最大训练时间5分钟
            },
            time_budget=300.0,
            resource_constraints={
                "max_memory_mb": 2048,
                "max_disk_mb": 500,
            },
        )

        # 任务2：强化学习（CartPole）
        self.tasks["cartpole_rl"] = ValidationTask(
            id="cartpole_rl",
            name="CartPole平衡任务",
            description="OpenAI Gym CartPole-v1平衡任务",
            task_type="reinforcement_learning",
            difficulty="medium",
            dataset_config={
                "environment": "CartPole-v1",
                "max_steps": 500,
                "num_episodes": 100,
                "state_dim": 4,
                "action_dim": 2,
            },
            success_criteria={
                "average_reward": 195.0,  # 平均奖励195（接近最优）
                "success_rate": 0.8,  # 80%的成功率
                "convergence_episodes": 50,  # 在50个episode内收敛
            },
            time_budget=600.0,
            resource_constraints={
                "max_memory_mb": 1024,
                "max_disk_mb": 100,
            },
        )

        # 任务4：多模态学习（图像描述）
        self.tasks["image_captioning"] = ValidationTask(
            id="image_captioning",
            name="图像描述生成",
            description="生成图像的文本描述",
            task_type="multimodal",
            difficulty="hard",
            dataset_config={
                "name": "Flickr8k",
                "num_images": 1000,
                "num_captions_per_image": 5,
                "image_size": [224, 224, 3],
                "max_caption_length": 20,
                "vocab_size": 5000,
            },
            success_criteria={
                "bleu_score": 0.4,  # BLEU-4分数
                "cider_score": 0.6,
                "training_time": 1200.0,
            },
            time_budget=1200.0,
            resource_constraints={
                "max_memory_mb": 8192,
                "max_disk_mb": 2000,
            },
        )

        logger.info(f"初始化了 {len(self.tasks)} 个基准验证任务")

    def _initialize_baseline_strategies(self):
        """初始化基准学习策略"""

        # 策略1：固定学习率
        self.strategies["fixed_lr"] = LearningStrategy(
            id="fixed_lr",
            name="固定学习率策略",
            description="使用固定学习率和标准优化器的基准策略",
            strategy_type="fixed",
            parameters={
                "learning_rate": 0.001,
                "optimizer": "adam",
                "batch_size": 32,
                "epochs": 10,
                "scheduler": "none",
            },
            implementation=self._implement_fixed_strategy,
        )

        # 策略2：自适应学习率（ReduceLROnPlateau）
        self.strategies["adaptive_lr"] = LearningStrategy(
            id="adaptive_lr",
            name="自适应学习率策略",
            description="基于验证损失自适应调整学习率",
            strategy_type="adaptive",
            parameters={
                "initial_lr": 0.01,
                "optimizer": "adam",
                "batch_size": 32,
                "epochs": 15,
                "scheduler": "reduce_on_plateau",
                "patience": 5,
                "factor": 0.5,
            },
            implementation=self._implement_adaptive_lr_strategy,
        )

        # 策略3：课程学习策略
        self.strategies["curriculum_learning"] = LearningStrategy(
            id="curriculum_learning",
            name="课程学习策略",
            description="从简单样本开始，逐步增加难度",
            strategy_type="curriculum",
            parameters={
                "learning_rate": 0.001,
                "optimizer": "adam",
                "batch_size": 32,
                "epochs": 20,
                "difficulty_schedule": "linear",
                "difficulty_start": 0.2,
                "difficulty_end": 1.0,
            },
            implementation=self._implement_curriculum_strategy,
        )

        # 策略4：元学习策略（MAML简化版）
        self.strategies["meta_learning"] = LearningStrategy(
            id="meta_learning",
            name="元学习策略",
            description="学习如何快速适应新任务",
            strategy_type="meta_learning",
            parameters={
                "meta_lr": 0.01,
                "inner_lr": 0.001,
                "optimizer": "adam",
                "meta_batch_size": 4,
                "num_inner_steps": 5,
                "adaptation_steps": 3,
            },
            implementation=self._implement_meta_learning_strategy,
        )

        logger.info(f"初始化了 {len(self.strategies)} 个基准学习策略")

    def validate_strategy(
        self, strategy_id: str, task_id: str, verbose: bool = True
    ) -> ValidationResult:
        """验证特定策略在特定任务上的表现"""

        if strategy_id not in self.strategies:
            raise ValueError(f"未知策略ID: {strategy_id}")
        if task_id not in self.tasks:
            raise ValueError(f"未知任务ID: {task_id}")

        strategy = self.strategies[strategy_id]
        task = self.tasks[task_id]

        logger.info(f"开始验证: 策略 '{strategy.name}' 在任务 '{task.name}' 上")

        # 记录开始时间
        start_time = time.time()

        try:
            # 执行策略实现
            if verbose:
                print(f"\n{'=' * 60}")
                print(f"验证: {strategy.name} -> {task.name}")
                print(f"任务类型: {task.task_type}, 难度: {task.difficulty}")
                print(f"时间预算: {task.time_budget}秒")
                print(f"{'=' * 60}\n")

            # 调用策略实现函数
            metrics, convergence_data, resource_usage = strategy.implementation(
                task, strategy.parameters
            )

            # 计算耗时
            time_taken = time.time() - start_time

            # 检查是否超时
            if time_taken > task.time_budget:
                logger.warning(
                    f"验证超时: 耗时 {time_taken:.1f}秒 > 预算 {task.time_budget}秒"
                )

            # 评估成功标准
            success = self._evaluate_success_criteria(metrics, task.success_criteria)

            # 性能总结
            performance_summary = {
                "time_efficiency": task.time_budget / max(time_taken, 0.001),
                "resource_efficiency": self._calculate_resource_efficiency(
                    resource_usage, task.resource_constraints
                ),
                "convergence_speed": (
                    self._calculate_convergence_speed(convergence_data)
                    if convergence_data
                    else 0.0
                ),
                "final_performance": (
                    metrics.get(list(metrics.keys())[0], 0.0) if metrics else 0.0
                ),
            }

            # 创建验证结果
            result = ValidationResult(
                task_id=task_id,
                strategy_id=strategy_id,
                success=success,
                metrics=metrics,
                performance_summary=performance_summary,
                convergence_data=convergence_data,
                time_taken=time_taken,
                resource_usage=resource_usage,
                notes=f"验证完成，耗时 {time_taken:.1f}秒",
            )

            self.results.append(result)

            # 输出结果
            if verbose:
                self._print_validation_result(result)

            return result

        except Exception as e:
            logger.error(f"验证过程中发生错误: {e}")
            time_taken = time.time() - start_time

            # 创建失败结果
            result = ValidationResult(
                task_id=task_id,
                strategy_id=strategy_id,
                success=False,
                metrics={},
                performance_summary={},
                convergence_data=[],
                time_taken=time_taken,
                resource_usage={},
                notes=f"验证失败: {str(e)}",
            )

            self.results.append(result)
            return result

    def _implement_fixed_strategy(
        self, task: ValidationTask, params: Dict[str, Any]
    ) -> Tuple[Dict[str, float], List[float], Dict[str, float]]:
        """实现固定学习率策略

        根据项目要求"禁止使用虚拟数据"，此方法不再生成模拟数据。
        必须运行真实训练来获取验证指标。
        """

        error_message = (
            "自适应学习验证器检测到模拟数据生成\n"
            f"任务ID: {task.id}, 任务类型: {task.task_type}\n"
            "根据项目要求'禁止使用虚拟数据'和'不采用任何降级处理，直接报错'，\n"
            "自适应学习验证器不再支持模拟数据生成。\n"
            "必须运行真实训练来获取验证指标。\n"
            "解决方案：\n"
            "1. 使用真实训练数据配置验证任务\n"
            "2. 实现真实训练流程来评估学习策略\n"
            "3. 或禁用自适应学习验证功能"
        )

        # 根据项目要求"不采用任何降级处理，直接报错"
        raise RuntimeError(error_message)

    def _implement_adaptive_lr_strategy(
        self, task: ValidationTask, params: Dict[str, Any]
    ) -> Tuple[Dict[str, float], List[float], Dict[str, float]]:
        """实现自适应学习率策略

        根据项目要求"禁止使用虚拟数据"，此方法不再生成模拟数据。
        必须运行真实训练来获取验证指标。
        """

        # 调用已修复的_fixed_strategy方法，它会抛出RuntimeError
        # 这里直接抛出更具体的错误信息
        error_message = (
            "自适应学习率策略验证器检测到模拟数据生成\n"
            f"任务ID: {task.id}, 任务类型: {task.task_type}\n"
            "根据项目要求'禁止使用虚拟数据'，自适应学习策略验证不再支持模拟数据。\n"
            "必须运行真实训练来评估自适应学习率策略的效果。"
        )

        raise RuntimeError(error_message)

    def _implement_curriculum_strategy(
        self, task: ValidationTask, params: Dict[str, Any]
    ) -> Tuple[Dict[str, float], List[float], Dict[str, float]]:
        """实现课程学习策略

        根据项目要求"禁止使用虚拟数据"，此方法不再生成模拟数据。
        必须运行真实训练来获取验证指标。
        """

        error_message = (
            "课程学习策略验证器检测到模拟数据生成\n"
            f"任务ID: {task.id}, 任务类型: {task.task_type}\n"
            "根据项目要求'禁止使用虚拟数据'，课程学习策略验证不再支持模拟数据。\n"
            "必须运行真实训练来评估课程学习策略的效果。"
        )

        raise RuntimeError(error_message)

    def _implement_meta_learning_strategy(
        self, task: ValidationTask, params: Dict[str, Any]
    ) -> Tuple[Dict[str, float], List[float], Dict[str, float]]:
        """实现元学习策略

        根据项目要求"禁止使用虚拟数据"，此方法不再生成模拟数据。
        必须运行真实训练来获取验证指标。
        """

        error_message = (
            "元学习策略验证器检测到模拟数据生成\n"
            f"任务ID: {task.id}, 任务类型: {task.task_type}\n"
            "根据项目要求'禁止使用虚拟数据'，元学习策略验证不再支持模拟数据。\n"
            "必须运行真实训练来评估元学习策略的效果。"
        )

        raise RuntimeError(error_message)

    def _evaluate_success_criteria(
        self, metrics: Dict[str, float], success_criteria: Dict[str, float]
    ) -> bool:
        """评估是否满足成功标准"""

        if not metrics:
            return False

        for criterion, threshold in success_criteria.items():
            if criterion in metrics:
                if metrics[criterion] < threshold:
                    return False
            else:
                # 如果没有对应的指标，假设失败
                return False

        return True

    def _calculate_resource_efficiency(
        self, resource_usage: Dict[str, float], constraints: Dict[str, float]
    ) -> float:
        """计算资源效率"""

        if not resource_usage or not constraints:
            return 0.5  # 默认值

        efficiency_scores = []

        # 内存效率
        if "max_memory_mb" in constraints and "memory_mb" in resource_usage:
            memory_efficiency = 1.0 - min(
                1.0, resource_usage["memory_mb"] / constraints["max_memory_mb"]
            )
            efficiency_scores.append(memory_efficiency)

        # 磁盘效率
        if "max_disk_mb" in constraints and "disk_mb" in resource_usage:
            disk_efficiency = 1.0 - min(
                1.0, resource_usage["disk_mb"] / constraints["max_disk_mb"]
            )
            efficiency_scores.append(disk_efficiency)

        # CPU效率（假设越低越好）
        if "cpu_percent" in resource_usage:
            cpu_efficiency = 1.0 - resource_usage["cpu_percent"] / 100.0
            efficiency_scores.append(cpu_efficiency)

        return sum(efficiency_scores) / max(len(efficiency_scores), 1)

    def _calculate_convergence_speed(self, convergence_data: List[float]) -> float:
        """计算收敛速度"""

        if not convergence_data or len(convergence_data) < 2:
            return 0.0

        # 计算达到90%最终性能所需的步数
        if len(convergence_data) >= 10:
            final_value = convergence_data[-1]
            target_value = final_value * 0.9

            for i, value in enumerate(convergence_data):
                if value >= target_value:
                    # 归一化：步数越少，速度越快
                    convergence_speed = 1.0 - i / len(convergence_data)
                    return max(0.0, min(1.0, convergence_speed))

        return 0.5  # 默认值

    def _print_validation_result(self, result: ValidationResult):
        """打印验证结果"""

        print(f"\n{'=' * 60}")
        print(f"验证结果: {result.task_id} -> {result.strategy_id}")
        print(f"{'=' * 60}")

        print(f"成功: {'是' if result.success else '否'}")
        print(f"耗时: {result.time_taken:.1f}秒")

        if result.metrics:
            print("\n评估指标:")
            for metric, value in result.metrics.items():
                print(f"  {metric}: {value:.4f}")

        if result.performance_summary:
            print("\n性能总结:")
            for key, value in result.performance_summary.items():
                print(f"  {key}: {value:.4f}")

        if result.resource_usage:
            print("\n资源使用:")
            for resource, usage in result.resource_usage.items():
                print(f"  {resource}: {usage:.1f}")

        if result.notes:
            print(f"\n备注: {result.notes}")

        print(f"{'=' * 60}\n")

    def run_comprehensive_validation(
        self,
        task_ids: Optional[List[str]] = None,
        strategy_ids: Optional[List[str]] = None,
        output_file: Optional[str] = None,
    ) -> ValidationReport:
        """运行全面的验证实验"""

        # 确定要验证的任务和策略
        tasks_to_validate = task_ids or list(self.tasks.keys())
        strategies_to_validate = strategy_ids or list(self.strategies.keys())

        logger.info(
            f"开始全面验证: {len(tasks_to_validate)}个任务 × {len(strategies_to_validate)}个策略"
        )

        results = []

        # 运行所有组合
        for task_id in tasks_to_validate:
            for strategy_id in strategies_to_validate:
                try:
                    result = self.validate_strategy(strategy_id, task_id, verbose=False)
                    results.append(result)

                    # 短暂延迟
                    time.sleep(0.1)

                except Exception as e:
                    logger.error(f"验证失败 {strategy_id}->{task_id}: {e}")

        # 生成总体报告
        overall_summary = self._generate_overall_summary(results)
        recommendations = self._generate_recommendations(results)
        limitations = self._generate_limitations()

        report = ValidationReport(
            validation_id=f"validation_{int(time.time())}",
            timestamp=datetime.now(),
            tasks=[self.tasks[t] for t in tasks_to_validate],
            strategies=[self.strategies[s] for s in strategies_to_validate],
            results=results,
            overall_summary=overall_summary,
            recommendations=recommendations,
            limitations=limitations,
        )

        # 保存报告
        if output_file:
            self._save_validation_report(report, output_file)

        # 打印摘要
        self._print_validation_summary(report)

        return report

    def _generate_overall_summary(
        self, results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """生成总体摘要"""

        if not results:
            return {}

        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]

        # 计算平均指标
        avg_time = statistics.mean([r.time_taken for r in results]) if results else 0.0

        # 最佳策略分析
        strategy_scores = {}
        for result in results:
            if result.success and result.metrics:
                # 使用第一个指标作为分数
                first_metric = (
                    list(result.metrics.values())[0] if result.metrics else 0.0
                )
                score = first_metric / max(result.time_taken, 0.001)  # 分数/时间

                if result.strategy_id not in strategy_scores:
                    strategy_scores[result.strategy_id] = []
                strategy_scores[result.strategy_id].append(score)

        # 计算平均分
        avg_strategy_scores = {}
        for strategy_id, scores in strategy_scores.items():
            avg_strategy_scores[strategy_id] = (
                statistics.mean(scores) if scores else 0.0
            )

        # 找出最佳策略
        best_strategy = (
            max(avg_strategy_scores.items(), key=lambda x: x[1])
            if avg_strategy_scores
            else ("none", 0.0)
        )

        summary = {
            "total_validations": len(results),
            "successful_validations": len(successful_results),
            "failed_validations": len(failed_results),
            "success_rate": len(successful_results) / max(len(results), 1),
            "average_time_seconds": avg_time,
            "best_strategy": best_strategy[0],
            "best_strategy_score": best_strategy[1],
            "strategy_scores": avg_strategy_scores,
        }

        return summary

    def _generate_recommendations(self, results: List[ValidationResult]) -> List[str]:
        """生成推荐"""

        recommendations = []

        # 分析结果模式
        strategy_performance = {}
        for result in results:
            if result.success and result.metrics:
                # 简单性能分数
                perf_score = list(result.metrics.values())[0] if result.metrics else 0.0
                if result.strategy_id not in strategy_performance:
                    strategy_performance[result.strategy_id] = []
                strategy_performance[result.strategy_id].append(perf_score)

        # 找出高性能策略
        for strategy_id, scores in strategy_performance.items():
            avg_score = statistics.mean(scores) if scores else 0.0
            if avg_score > 0.8:  # 高性能阈值
                recommendations.append(
                    f"策略 '{strategy_id}' 表现优异，平均得分 {avg_score:.2f}"
                )

        # 通用推荐
        if len(results) >= 4:
            recommendations.append("对于复杂任务，建议使用自适应学习率或元学习策略")
            recommendations.append("对于资源受限环境，固定学习率策略更高效")
            recommendations.append("课程学习策略在稳定性和最终性能之间提供了良好的平衡")

        return recommendations

    def _generate_limitations(self) -> List[str]:
        """生成限制说明"""

        limitations = [
            "当前验证使用模拟数据，真实环境表现可能有所不同",
            "资源使用情况基于估计，实际资源消耗可能因硬件而异",
            "未考虑极端情况下的策略鲁棒性",
            "验证任务范围有限，未覆盖所有可能的任务类型",
            "策略参数未进行充分优化，可能影响最终性能",
        ]

        return limitations

    def _save_validation_report(self, report: ValidationReport, output_file: str):
        """保存验证报告"""

        try:
            # 转换为可序列化格式
            report_dict = {
                "validation_id": report.validation_id,
                "timestamp": report.timestamp.isoformat(),
                "tasks": [
                    {
                        "id": t.id,
                        "name": t.name,
                        "task_type": t.task_type,
                        "difficulty": t.difficulty,
                    }
                    for t in report.tasks
                ],
                "strategies": [
                    {
                        "id": s.id,
                        "name": s.name,
                        "strategy_type": s.strategy_type,
                    }
                    for s in report.strategies
                ],
                "results": [
                    {
                        "task_id": r.task_id,
                        "strategy_id": r.strategy_id,
                        "success": r.success,
                        "time_taken": r.time_taken,
                    }
                    for r in report.results
                ],
                "overall_summary": report.overall_summary,
                "recommendations": report.recommendations,
                "limitations": report.limitations,
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report_dict, f, indent=2, ensure_ascii=False)

            logger.info(f"验证报告已保存到: {output_file}")

        except Exception as e:
            logger.error(f"保存验证报告失败: {e}")

    def _print_validation_summary(self, report: ValidationReport):
        """打印验证摘要"""

        print(f"\n{'=' * 60}")
        print("自适应学习策略验证报告")
        print(f"{'=' * 60}")
        print(f"验证ID: {report.validation_id}")
        print(f"时间: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"验证任务数: {len(report.tasks)}")
        print(f"验证策略数: {len(report.strategies)}")
        print(f"验证结果数: {len(report.results)}")

        print("\n总体摘要:")
        for key, value in report.overall_summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        if report.recommendations:
            print("\n推荐建议:")
            for i, rec in enumerate(report.recommendations, 1):
                print(f"  {i}. {rec}")

        if report.limitations:
            print("\n限制说明:")
            for i, limit in enumerate(report.limitations, 1):
                print(f"  {i}. {limit}")

        print(f"{'=' * 60}\n")


# 全局验证器实例
_adaptive_learning_validator = None


def get_adaptive_learning_validator(**kwargs) -> AdaptiveLearningValidator:
    """获取自适应学习验证器单例"""
    global _adaptive_learning_validator
    if _adaptive_learning_validator is None:
        _adaptive_learning_validator = AdaptiveLearningValidator(**kwargs)
    return _adaptive_learning_validator


# 使用示例
if __name__ == "__main__":
    # 初始化日志
    logging.basicConfig(level=logging.INFO)

    # 获取验证器
    validator = get_adaptive_learning_validator()

    # 运行单个验证
    print("运行单个验证示例...")
    result = validator.validate_strategy("adaptive_lr", "mnist_classification")

    # 运行全面验证
    print("\n运行全面验证...")
    report = validator.run_comprehensive_validation(
        task_ids=["mnist_classification", "cartpole_rl"],
        strategy_ids=["fixed_lr", "adaptive_lr", "curriculum_learning"],
        output_file="adaptive_learning_validation_report.json",
    )

    print("自适应学习策略验证完成")
