"""
训练报告生成器
自动生成训练报告，支持多种格式和可视化图表

功能：
1. 训练报告模板系统（HTML/PDF）
2. 数据可视化图表自动生成
3. 报告定时发送和存档
4. 报告版本管理和对比
"""

import json
import os
import time
import datetime
import tempfile
import shutil
import base64
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, asdict, field
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """报告格式"""
    HTML = "html"
    PDF = "pdf"
    MARKDOWN = "markdown"
    JSON = "json"
    CSV = "csv"


class ReportType(Enum):
    """报告类型"""
    TRAINING_SUMMARY = "training_summary"      # 训练摘要报告
    PERFORMANCE_ANALYSIS = "performance_analysis"  # 性能分析报告
    RESOURCE_USAGE = "resource_usage"          # 资源使用报告
    ANOMALY_REPORT = "anomaly_report"          # 异常检测报告
    MODEL_COMPARISON = "model_comparison"      # 模型对比报告
    VALIDATION_REPORT = "validation_report"    # 验证报告
    CHECKPOINT_REPORT = "checkpoint_report"    # 检查点报告
    TRAINING_PROGRESS = "training_progress"    # 训练进度报告
    WEEKLY_SUMMARY = "weekly_summary"          # 每周总结报告
    MONTHLY_SUMMARY = "monthly_summary"        # 每月总结报告


@dataclass
class TrainingMetrics:
    """训练指标"""
    training_id: str
    model_id: str
    start_time: str
    end_time: str
    total_duration: float  # 秒
    total_epochs: int
    total_steps: int
    final_loss: float
    best_loss: float
    final_accuracy: float
    best_accuracy: float
    learning_rates: List[float]
    losses: List[float]
    accuracies: List[float]
    validation_losses: List[float]
    validation_accuracies: List[float]
    gradient_norms: List[float]
    batch_times: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class ResourceMetrics:
    """资源指标"""
    cpu_usage_percent: List[float]
    memory_usage_percent: List[float]
    gpu_usage_percent: List[float]
    gpu_memory_usage_percent: List[float]
    disk_usage_percent: List[float]
    network_usage_mbps: List[float]
    timestamps: List[str]
    peak_cpu: float
    peak_memory: float
    peak_gpu: float
    average_cpu: float
    average_memory: float
    average_gpu: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class ModelMetadata:
    """模型元数据"""
    model_name: str
    model_version: str
    model_architecture: str
    model_parameters: int
    model_size_mb: float
    framework: str
    framework_version: str
    training_framework: str
    training_framework_version: str
    created_by: str
    created_at: str
    tags: List[str] = field(default_factory=list)
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class Hyperparameters:
    """超参数"""
    learning_rate: float
    batch_size: int
    optimizer: str
    loss_function: str
    weight_decay: float
    momentum: float
    dropout_rate: float
    num_layers: int
    hidden_size: int
    num_heads: int  # Transformer模型
    feedforward_size: int  # Transformer模型
    max_sequence_length: int  # Transformer模型
    gradient_clip: float
    warmup_steps: int
    lr_scheduler: str
    epochs: int
    patience: int
    validation_split: float
    seed: int
    data_augmentation: bool
    mixed_precision: bool
    distributed_training: bool
    num_gpus: int
    num_nodes: int
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class DatasetInfo:
    """数据集信息"""
    dataset_name: str
    dataset_version: str
    dataset_size: int
    num_classes: int
    input_shape: List[int]
    train_samples: int
    val_samples: int
    test_samples: int
    data_source: str
    data_license: str
    preprocessing_steps: List[str]
    augmentation_steps: List[str]
    class_distribution: Dict[str, int] = field(default_factory=dict)
    feature_columns: List[str] = field(default_factory=list)
    target_column: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class ReportConfig:
    """报告配置"""
    report_type: ReportType
    report_format: ReportFormat
    title: str
    subtitle: str = ""
    author: str = "AGI训练系统"
    organization: str = "Self AGI"
    include_charts: bool = True
    include_summary: bool = True
    include_details: bool = True
    include_recommendations: bool = True
    chart_style: str = "seaborn"  # matplotlib样式
    color_palette: str = "viridis"  # 颜色调色板
    chart_dpi: int = 300
    max_charts_per_page: int = 4
    output_directory: str = "reports"
    template_directory: str = "templates"
    language: str = "zh"  # 报告语言
    timezone: str = "Asia/Shanghai"
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result["report_type"] = self.report_type.value
        result["report_format"] = self.report_format.value
        return result


class ChartGenerator:
    """图表生成器"""
    
    def __init__(self, style: str = "seaborn", palette: str = "viridis", dpi: int = 300):
        """初始化图表生成器"""
        self.style = style
        self.palette = palette
        self.dpi = dpi
        
        # 设置matplotlib样式
        plt.style.use(style)
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['savefig.dpi'] = dpi
        plt.rcParams['figure.figsize'] = (10, 6)
        
        logger.info(f"图表生成器初始化: style={style}, palette={palette}, dpi={dpi}")
    
    def generate_loss_curve(self, 
                           training_losses: List[float], 
                           validation_losses: Optional[List[float]] = None,
                           title: str = "训练损失曲线",
                           xlabel: str = "轮次",
                           ylabel: str = "损失") -> Figure:
        """生成损失曲线图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 训练损失
        epochs = range(1, len(training_losses) + 1)
        ax.plot(epochs, training_losses, 'b-', label='训练损失', linewidth=2)
        
        # 验证损失（如果存在）
        if validation_losses:
            if len(validation_losses) <= len(training_losses):
                val_epochs = range(1, len(validation_losses) + 1)
                ax.plot(val_epochs, validation_losses, 'r--', label='验证损失', linewidth=2)
            else:
                ax.plot(epochs, validation_losses[:len(training_losses)], 'r--', label='验证损失', linewidth=2)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        ax.tick_params(labelsize=10)
        
        # 添加最小损失标记
        if training_losses:
            min_loss = min(training_losses)
            min_epoch = training_losses.index(min_loss) + 1
            ax.scatter([min_epoch], [min_loss], color='red', s=100, zorder=5)
            ax.annotate(f'最小损失: {min_loss:.4f}', 
                       xy=(min_epoch, min_loss), 
                       xytext=(min_epoch + len(training_losses)*0.05, min_loss + max(training_losses)*0.1),
                       arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                       fontsize=10, color='red')
        
        plt.tight_layout()
        return fig
    
    def generate_accuracy_curve(self,
                               training_accuracies: List[float],
                               validation_accuracies: Optional[List[float]] = None,
                               title: str = "训练准确率曲线",
                               xlabel: str = "轮次",
                               ylabel: str = "准确率") -> Figure:
        """生成准确率曲线图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 训练准确率
        epochs = range(1, len(training_accuracies) + 1)
        ax.plot(epochs, training_accuracies, 'g-', label='训练准确率', linewidth=2)
        
        # 验证准确率（如果存在）
        if validation_accuracies:
            if len(validation_accuracies) <= len(training_accuracies):
                val_epochs = range(1, len(validation_accuracies) + 1)
                ax.plot(val_epochs, validation_accuracies, 'm--', label='验证准确率', linewidth=2)
            else:
                ax.plot(epochs, validation_accuracies[:len(training_accuracies)], 'm--', label='验证准确率', linewidth=2)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        ax.tick_params(labelsize=10)
        
        # 设置y轴范围为0-1或0-100
        if max(training_accuracies) <= 1.0:
            ax.set_ylim([0, 1.0])
        else:
            ax.set_ylim([0, 100])
        
        # 添加最大准确率标记
        if training_accuracies:
            max_acc = max(training_accuracies)
            max_epoch = training_accuracies.index(max_acc) + 1
            ax.scatter([max_epoch], [max_acc], color='red', s=100, zorder=5)
            ax.annotate(f'最大准确率: {max_acc:.4f}', 
                       xy=(max_epoch, max_acc), 
                       xytext=(max_epoch + len(training_accuracies)*0.05, max_acc - max(training_accuracies)*0.1),
                       arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                       fontsize=10, color='red')
        
        plt.tight_layout()
        return fig
    
    def generate_learning_rate_curve(self,
                                    learning_rates: List[float],
                                    title: str = "学习率变化曲线",
                                    xlabel: str = "步数",
                                    ylabel: str = "学习率") -> Figure:
        """生成学习率曲线图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        steps = range(1, len(learning_rates) + 1)
        ax.plot(steps, learning_rates, 'purple', label='学习率', linewidth=2)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        ax.tick_params(labelsize=10)
        
        # 使用对数刻度（如果学习率变化范围大）
        if max(learning_rates) / min(learning_rates) > 100:
            ax.set_yscale('log')
            ax.set_ylabel(f"{ylabel} (对数刻度)", fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def generate_resource_usage_chart(self,
                                     resource_metrics: ResourceMetrics,
                                     title: str = "资源使用情况",
                                     xlabel: str = "时间",
                                     ylabel: str = "使用率 (%)") -> Figure:
        """生成资源使用情况图表"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 时间序列数据
        timestamps = pd.to_datetime(resource_metrics.timestamps)
        
        # CPU使用率
        ax1 = axes[0, 0]
        ax1.plot(timestamps, resource_metrics.cpu_usage_percent, 'blue', linewidth=2)
        ax1.set_title('CPU使用率', fontsize=14, fontweight='bold')
        ax1.set_xlabel(xlabel, fontsize=10)
        ax1.set_ylabel('CPU使用率 (%)', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45, labelsize=8)
        
        # 内存使用率
        ax2 = axes[0, 1]
        ax2.plot(timestamps, resource_metrics.memory_usage_percent, 'green', linewidth=2)
        ax2.set_title('内存使用率', fontsize=14, fontweight='bold')
        ax2.set_xlabel(xlabel, fontsize=10)
        ax2.set_ylabel('内存使用率 (%)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45, labelsize=8)
        
        # GPU使用率
        ax3 = axes[1, 0]
        if resource_metrics.gpu_usage_percent:
            ax3.plot(timestamps[:len(resource_metrics.gpu_usage_percent)], 
                    resource_metrics.gpu_usage_percent, 'red', linewidth=2)
            ax3.set_title('GPU使用率', fontsize=14, fontweight='bold')
            ax3.set_xlabel(xlabel, fontsize=10)
            ax3.set_ylabel('GPU使用率 (%)', fontsize=10)
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis='x', rotation=45, labelsize=8)
        else:
            ax3.text(0.5, 0.5, '无GPU数据', ha='center', va='center', fontsize=12)
            ax3.set_title('GPU使用率 (无数据)', fontsize=14, fontweight='bold')
            ax3.axis('off')
        
        # GPU内存使用率
        ax4 = axes[1, 1]
        if resource_metrics.gpu_memory_usage_percent:
            ax4.plot(timestamps[:len(resource_metrics.gpu_memory_usage_percent)], 
                    resource_metrics.gpu_memory_usage_percent, 'orange', linewidth=2)
            ax4.set_title('GPU内存使用率', fontsize=14, fontweight='bold')
            ax4.set_xlabel(xlabel, fontsize=10)
            ax4.set_ylabel('GPU内存使用率 (%)', fontsize=10)
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(axis='x', rotation=45, labelsize=8)
        else:
            ax4.text(0.5, 0.5, '无GPU内存数据', ha='center', va='center', fontsize=12)
            ax4.set_title('GPU内存使用率 (无数据)', fontsize=14, fontweight='bold')
            ax4.axis('off')
        
        fig.suptitle(title, fontsize=18, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def generate_gradient_distribution(self,
                                      gradient_norms: List[float],
                                      title: str = "梯度分布",
                                      xlabel: str = "梯度范数",
                                      ylabel: str = "频率") -> Figure:
        """生成梯度分布直方图"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 直方图
        ax1 = axes[0]
        ax1.hist(gradient_norms, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_title('梯度范数分布', fontsize=14, fontweight='bold')
        ax1.set_xlabel(xlabel, fontsize=10)
        ax1.set_ylabel(ylabel, fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_norm = np.mean(gradient_norms)
        std_norm = np.std(gradient_norms)
        ax1.axvline(mean_norm, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_norm:.4f}')
        ax1.axvline(mean_norm + std_norm, color='orange', linestyle=':', linewidth=1.5, label=f'±1标准差')
        ax1.axvline(mean_norm - std_norm, color='orange', linestyle=':', linewidth=1.5)
        ax1.legend(fontsize=9)
        
        # 时间序列图
        ax2 = axes[1]
        steps = range(1, len(gradient_norms) + 1)
        ax2.plot(steps, gradient_norms, 'green', linewidth=2)
        ax2.set_title('梯度范数变化', fontsize=14, fontweight='bold')
        ax2.set_xlabel('步数', fontsize=10)
        ax2.set_ylabel('梯度范数', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 添加指数移动平均
        if len(gradient_norms) > 10:
            window = min(50, len(gradient_norms) // 10)
            ema = pd.Series(gradient_norms).ewm(span=window).mean()
            ax2.plot(steps, ema, 'red', linewidth=2, label=f'EMA (窗口={window})')
            ax2.legend(fontsize=9)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def figure_to_base64(self, fig: Figure) -> str:
        """将图表转换为base64编码的PNG图像"""
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=self.dpi)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)  # 关闭图形释放内存
        return img_str
    
    def figure_to_svg(self, fig: Figure) -> str:
        """将图表转换为SVG字符串"""
        import io
        buf = io.StringIO()
        fig.savefig(buf, format='svg', bbox_inches='tight')
        svg_str = buf.getvalue()
        plt.close(fig)  # 关闭图形释放内存
        return svg_str


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, config: Optional[ReportConfig] = None):
        """初始化报告生成器"""
        self.config = config or ReportConfig(
            report_type=ReportType.TRAINING_SUMMARY,
            report_format=ReportFormat.HTML,
            title="训练报告",
            include_charts=True,
            include_summary=True,
            include_details=True,
            include_recommendations=True,
        )
        
        self.chart_generator = ChartGenerator(
            style=self.config.chart_style,
            palette=self.config.color_palette,
            dpi=self.config.chart_dpi
        )
        
        # 初始化Jinja2模板环境
        self.template_env = self._setup_template_env()
        
        # 确保输出目录存在
        os.makedirs(self.config.output_directory, exist_ok=True)
        
        logger.info(f"报告生成器初始化: 类型={self.config.report_type.value}, 格式={self.config.report_format.value}")
    
    def _setup_template_env(self) -> Environment:
        """设置模板环境"""
        # 首先检查自定义模板目录
        template_dirs = []
        
        if os.path.exists(self.config.template_directory):
            template_dirs.append(self.config.template_directory)
        
        # 添加默认模板目录（相对于当前文件）
        default_template_dir = os.path.join(os.path.dirname(__file__), "templates")
        if os.path.exists(default_template_dir):
            template_dirs.append(default_template_dir)
        
        # 如果都没有，使用当前目录
        if not template_dirs:
            template_dirs.append(".")
        
        return Environment(
            loader=FileSystemLoader(template_dirs),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    def generate_training_summary_report(self,
                                        training_metrics: TrainingMetrics,
                                        model_metadata: ModelMetadata,
                                        hyperparameters: Hyperparameters,
                                        dataset_info: DatasetInfo,
                                        resource_metrics: Optional[ResourceMetrics] = None,
                                        additional_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """生成训练摘要报告"""
        logger.info(f"生成训练摘要报告: {training_metrics.training_id}")
        
        # 生成图表
        charts = {}
        if self.config.include_charts:
            try:
                # 损失曲线
                loss_fig = self.chart_generator.generate_loss_curve(
                    training_losses=training_metrics.losses,
                    validation_losses=training_metrics.validation_losses if hasattr(training_metrics, 'validation_losses') else None,
                    title="训练损失曲线"
                )
                charts['loss_curve'] = self.chart_generator.figure_to_base64(loss_fig)
                
                # 准确率曲线
                accuracy_fig = self.chart_generator.generate_accuracy_curve(
                    training_accuracies=training_metrics.accuracies,
                    validation_accuracies=training_metrics.validation_accuracies if hasattr(training_metrics, 'validation_accuracies') else None,
                    title="训练准确率曲线"
                )
                charts['accuracy_curve'] = self.chart_generator.figure_to_base64(accuracy_fig)
                
                # 学习率曲线（如果有）
                if training_metrics.learning_rates and len(training_metrics.learning_rates) > 0:
                    lr_fig = self.chart_generator.generate_learning_rate_curve(
                        learning_rates=training_metrics.learning_rates,
                        title="学习率变化曲线"
                    )
                    charts['learning_rate_curve'] = self.chart_generator.figure_to_base64(lr_fig)
                
                # 梯度分布（如果有）
                if training_metrics.gradient_norms and len(training_metrics.gradient_norms) > 0:
                    gradient_fig = self.chart_generator.generate_gradient_distribution(
                        gradient_norms=training_metrics.gradient_norms,
                        title="梯度分布"
                    )
                    charts['gradient_distribution'] = self.chart_generator.figure_to_base64(gradient_fig)
                
                # 资源使用情况（如果有）
                if resource_metrics and len(resource_metrics.cpu_usage_percent) > 0:
                    resource_fig = self.chart_generator.generate_resource_usage_chart(
                        resource_metrics=resource_metrics,
                        title="资源使用情况"
                    )
                    charts['resource_usage'] = self.chart_generator.figure_to_base64(resource_fig)
                    
            except Exception as e:
                logger.error(f"生成图表失败: {e}")
                charts['error'] = f"图表生成失败: {str(e)}"
        
        # 准备报告数据
        report_data = {
            "config": self.config.to_dict(),
            "training_metrics": training_metrics.to_dict(),
            "model_metadata": model_metadata.to_dict(),
            "hyperparameters": hyperparameters.to_dict(),
            "dataset_info": dataset_info.to_dict(),
            "resource_metrics": resource_metrics.to_dict() if resource_metrics else {},
            "charts": charts,
            "additional_data": additional_data or {},
            "generated_at": datetime.datetime.now().isoformat(),
            "report_id": f"report_{training_metrics.training_id}_{int(time.time())}",
            "language": self.config.language,
        }
        
        # 计算统计信息
        report_data['statistics'] = self._calculate_statistics(report_data)
        
        # 生成建议
        if self.config.include_recommendations:
            report_data['recommendations'] = self._generate_recommendations(report_data)
        
        # 生成报告
        report_file = self._generate_report_file(report_data)
        
        return {
            "success": True,
            "report_id": report_data['report_id'],
            "report_file": report_file,
            "report_data": report_data,
            "message": "训练摘要报告生成成功",
        }
    
    def _calculate_statistics(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """计算统计信息"""
        metrics = report_data['training_metrics']
        
        stats = {
            "training_duration_hours": metrics['total_duration'] / 3600,
            "samples_per_second": metrics['total_steps'] / metrics['total_duration'] if metrics['total_duration'] > 0 else 0,
            "epochs_per_hour": metrics['total_epochs'] / (metrics['total_duration'] / 3600) if metrics['total_duration'] > 0 else 0,
            "loss_improvement": ((metrics['losses'][0] - metrics['final_loss']) / metrics['losses'][0] * 100) if metrics['losses'] and metrics['losses'][0] > 0 else 0,
            "accuracy_improvement": (metrics['final_accuracy'] - metrics['accuracies'][0]) * 100 if metrics['accuracies'] else 0,
            "training_efficiency": metrics['final_accuracy'] / (metrics['total_duration'] / 3600) if metrics['total_duration'] > 0 else 0,
        }
        
        # 计算收敛速度
        if metrics['losses']:
            # 找到损失下降到初始值90%、50%、10%的时间点
            initial_loss = metrics['losses'][0]
            target_levels = [0.9, 0.5, 0.1]
            convergence_points = {}
            
            for level in target_levels:
                target_loss = initial_loss * level
                for i, loss in enumerate(metrics['losses']):
                    if loss <= target_loss:
                        convergence_points[f'converge_to_{int(level*100)}%'] = i + 1
                        break
                else:
                    convergence_points[f'converge_to_{int(level*100)}%'] = None
            
            stats['convergence'] = convergence_points
        
        return stats
    
    def _generate_recommendations(self, report_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成建议"""
        recommendations = []
        metrics = report_data['training_metrics']
        
        # 检查训练时长
        if metrics['total_duration'] > 24 * 3600:  # 超过24小时
            recommendations.append({
                "type": "optimization",
                "title": "训练时间过长",
                "description": f"训练耗时 {metrics['total_duration']/3600:.1f} 小时，考虑优化训练效率",
                "suggestions": [
                    "增加批量大小以利用GPU并行计算",
                    "使用混合精度训练减少内存占用",
                    "优化数据加载管道减少I/O等待",
                    "使用学习率预热和衰减策略",
                ],
                "priority": "medium",
            })
        
        # 检查最终损失
        if metrics['final_loss'] > 1.0:
            recommendations.append({
                "type": "performance",
                "title": "损失值偏高",
                "description": f"最终损失 {metrics['final_loss']:.4f} 较高，可能未收敛或过拟合",
                "suggestions": [
                    "降低学习率",
                    "增加训练轮次",
                    "添加正则化（如Dropout、权重衰减）",
                    "检查数据质量和标注",
                ],
                "priority": "high",
            })
        
        # 检查准确率
        if metrics['final_accuracy'] < 0.7 and metrics.get('accuracies', []):
            initial_acc = metrics['accuracies'][0]
            improvement = (metrics['final_accuracy'] - initial_acc) * 100
            
            recommendations.append({
                "type": "performance",
                "title": "准确率偏低",
                "description": f"最终准确率 {metrics['final_accuracy']:.2%}，提升 {improvement:.1f}%",
                "suggestions": [
                    "调整模型架构（增加层数、神经元数）",
                    "使用更复杂的优化器（如AdamW）",
                    "增加训练数据或使用数据增强",
                    "尝试不同的损失函数",
                ],
                "priority": "high",
            })
        
        # 检查梯度爆炸/消失
        if 'gradient_norms' in metrics and metrics['gradient_norms']:
            grad_norms = metrics['gradient_norms']
            mean_grad = np.mean(grad_norms)
            
            if mean_grad > 100:
                recommendations.append({
                    "type": "stability",
                    "title": "梯度爆炸风险",
                    "description": f"平均梯度范数 {mean_grad:.2f} 较高，存在梯度爆炸风险",
                    "suggestions": [
                        "使用梯度裁剪（gradient clipping）",
                        "降低学习率",
                        "使用更稳定的优化器",
                        "检查模型初始化",
                    ],
                    "priority": "critical",
                })
            elif mean_grad < 1e-6:
                recommendations.append({
                    "type": "stability",
                    "title": "梯度消失风险",
                    "description": f"平均梯度范数 {mean_grad:.2e} 较低，存在梯度消失风险",
                    "suggestions": [
                        "使用ReLU等非饱和激活函数",
                        "使用残差连接",
                        "使用批归一化",
                        "调整权重初始化",
                    ],
                    "priority": "critical",
                })
        
        # 检查资源使用
        if report_data.get('resource_metrics'):
            res_metrics = report_data['resource_metrics']
            
            if res_metrics.get('peak_memory', 0) > 90:
                recommendations.append({
                    "type": "resource",
                    "title": "内存使用过高",
                    "description": f"峰值内存使用 {res_metrics['peak_memory']:.1f}%",
                    "suggestions": [
                        "减少批量大小",
                        "使用梯度累积",
                        "启用梯度检查点（gradient checkpointing）",
                        "优化模型架构减少参数",
                    ],
                    "priority": "medium",
                })
        
        return recommendations
    
    def _generate_report_file(self, report_data: Dict[str, Any]) -> str:
        """生成报告文件"""
        report_id = report_data['report_id']
        report_format = self.config.report_format
        
        # 生成文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_id}_{timestamp}.{report_format.value}"
        filepath = os.path.join(self.config.output_directory, filename)
        
        try:
            if report_format == ReportFormat.HTML:
                self._generate_html_report(report_data, filepath)
            elif report_format == ReportFormat.PDF:
                self._generate_pdf_report(report_data, filepath)
            elif report_format == ReportFormat.MARKDOWN:
                self._generate_markdown_report(report_data, filepath)
            elif report_format == ReportFormat.JSON:
                self._generate_json_report(report_data, filepath)
            elif report_format == ReportFormat.CSV:
                self._generate_csv_report(report_data, filepath)
            else:
                raise ValueError(f"不支持的报告格式: {report_format}")
            
            logger.info(f"报告文件生成成功: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"生成报告文件失败: {e}")
            # 生成备份文件
            backup_file = os.path.join(self.config.output_directory, f"{report_id}_error.json")
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            return backup_file
    
    def _generate_html_report(self, report_data: Dict[str, Any], filepath: str):
        """生成HTML报告"""
        try:
            # 尝试加载模板
            template_name = f"{self.config.report_type.value}.html.j2"
            template = self.template_env.get_template(template_name)
        except Exception:
            # 使用默认模板
            template = self._get_default_html_template()
        
        html_content = template.render(**report_data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _get_default_html_template(self):
        """获取默认HTML模板"""
        template_str = """
<!DOCTYPE html>
<html lang="{{ language }}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ config.title }}</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 30px; }
        .header h1 { margin: 0 0 10px 0; color: #2c3e50; }
        .header .subtitle { color: #7f8c8d; font-size: 1.2em; margin-bottom: 20px; }
        .section { margin-bottom: 40px; }
        .section h2 { color: #3498db; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        .card { background: white; border: 1px solid #ddd; border-radius: 5px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .chart { text-align: center; margin: 20px 0; }
        .chart img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .stat-item { background: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }
        .stat-value { font-size: 2em; font-weight: bold; color: #2c3e50; }
        .stat-label { color: #7f8c8d; margin-top: 5px; }
        .recommendation { margin-bottom: 15px; padding: 15px; border-left: 4px solid; border-radius: 3px; }
        .recommendation.critical { border-color: #e74c3c; background: #ffeaea; }
        .recommendation.high { border-color: #e67e22; background: #fff3e0; }
        .recommendation.medium { border-color: #f1c40f; background: #fffde7; }
        .recommendation.low { border-color: #2ecc71; background: #eaffea; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f2f2f2; font-weight: bold; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; text-align: center; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ config.title }}</h1>
            {% if config.subtitle %}<div class="subtitle">{{ config.subtitle }}</div>{% endif %}
            <div>报告ID: {{ report_id }}</div>
            <div>生成时间: {{ generated_at }}</div>
            <div>作者: {{ config.author }}</div>
        </div>
        
        {% if config.include_summary %}
        <div class="section">
            <h2>训练摘要</h2>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value">{{ "%.1f"|format(statistics.training_duration_hours) }}</div>
                    <div class="stat-label">训练时长 (小时)</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{{ "%.0f"|format(training_metrics.total_epochs) }}</div>
                    <div class="stat-label">总轮次</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{{ "%.4f"|format(training_metrics.final_loss) }}</div>
                    <div class="stat-label">最终损失</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{{ "%.2f"|format(training_metrics.final_accuracy * 100) }}%</div>
                    <div class="stat-label">最终准确率</div>
                </div>
            </div>
        </div>
        {% endif %}
        
        {% if charts %}
        <div class="section">
            <h2>训练图表</h2>
            {% for chart_name, chart_data in charts.items() %}
                {% if chart_name != 'error' %}
                <div class="chart">
                    <h3>{{ chart_name|replace('_', ' ')|title }}</h3>
                    <img src="data:image/png;base64,{{ chart_data }}" alt="{{ chart_name }}">
                </div>
                {% endif %}
            {% endfor %}
        </div>
        {% endif %}
        
        {% if recommendations %}
        <div class="section">
            <h2>优化建议</h2>
            {% for rec in recommendations %}
            <div class="recommendation {{ rec.priority }}">
                <h3>{{ rec.title }}</h3>
                <p>{{ rec.description }}</p>
                <ul>
                    {% for suggestion in rec.suggestions %}
                    <li>{{ suggestion }}</li>
                    {% endfor %}
                </ul>
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        {% if config.include_details %}
        <div class="section">
            <h2>详细数据</h2>
            
            <div class="card">
                <h3>模型信息</h3>
                <table>
                    <tr><th>模型名称</th><td>{{ model_metadata.model_name }}</td></tr>
                    <tr><th>模型架构</th><td>{{ model_metadata.model_architecture }}</td></tr>
                    <tr><th>参数量</th><td>{{ "{:,}".format(model_metadata.model_parameters) }}</td></tr>
                    <tr><th>模型大小</th><td>{{ "%.2f"|format(model_metadata.model_size_mb) }} MB</td></tr>
                </table>
            </div>
            
            <div class="card">
                <h3>超参数</h3>
                <table>
                    <tr><th>学习率</th><td>{{ hyperparameters.learning_rate }}</td></tr>
                    <tr><th>批量大小</th><td>{{ hyperparameters.batch_size }}</td></tr>
                    <tr><th>优化器</th><td>{{ hyperparameters.optimizer }}</td></tr>
                    <tr><th>损失函数</th><td>{{ hyperparameters.loss_function }}</td></tr>
                    <tr><th>训练轮次</th><td>{{ hyperparameters.epochs }}</td></tr>
                </table>
            </div>
        </div>
        {% endif %}
        
        <div class="footer">
            <p>生成系统: Self AGI 训练系统</p>
            <p>报告版本: 1.0.0 | © {{ config.organization }} {{ generated_at[:4] }}</p>
        </div>
    </div>
</body>
</html>
        """
        
        return self.template_env.from_string(template_str)
    
    def _generate_pdf_report(self, report_data: Dict[str, Any], filepath: str):
        """生成PDF报告"""
        # 暂时生成HTML，然后可以转换为PDF
        # 实际项目中可以使用weasyprint或reportlab
        html_file = filepath.replace('.pdf', '.html')
        self._generate_html_report(report_data, html_file)
        
        # 尝试转换为PDF
        try:
            import weasyprint  # type: ignore
            weasyprint.HTML(filename=html_file).write_pdf(filepath)
            os.remove(html_file)  # 删除临时HTML文件
            logger.info(f"PDF报告生成成功: {filepath}")
        except ImportError:
            logger.warning("weasyprint未安装，保存为HTML文件")
            # 重命名为HTML
            os.rename(html_file, filepath)
        except Exception as e:
            logger.error(f"PDF转换失败: {e}")
            # 重命名为HTML
            os.rename(html_file, filepath)
    
    def _generate_markdown_report(self, report_data: Dict[str, Any], filepath: str):
        """生成Markdown报告"""
        # 构建Markdown内容
        markdown_lines = [
            f"# {report_data['config']['title']}",
            "",
            f"**报告ID:** {report_data['report_id']}  ",
            f"**生成时间:** {report_data['generated_at']}  ",
            f"**作者:** {report_data['config']['author']}",
            "",
            "## 训练摘要",
            "",
            f"- **训练时长:** {report_data['statistics']['training_duration_hours']:.1f} 小时",
            f"- **总轮次:** {report_data['training_metrics']['total_epochs']}",
            f"- **最终损失:** {report_data['training_metrics']['final_loss']:.4f}",
            f"- **最终准确率:** {report_data['training_metrics']['final_accuracy']:.2%}",
            f"- **最佳损失:** {report_data['training_metrics']['best_loss']:.4f}",
            f"- **最佳准确率:** {report_data['training_metrics']['best_accuracy']:.2%}",
            "",
            "## 模型信息",
            "",
            f"- **模型名称:** {report_data['model_metadata']['model_name']}",
            f"- **模型架构:** {report_data['model_metadata']['model_architecture']}",
            f"- **参数量:** {report_data['model_metadata']['model_parameters']:,}",
            f"- **模型大小:** {report_data['model_metadata']['model_size_mb']:.2f} MB",
            "",
            "## 超参数",
            "",
            f"- **学习率:** {report_data['hyperparameters']['learning_rate']}",
            f"- **批量大小:** {report_data['hyperparameters']['batch_size']}",
            f"- **优化器:** {report_data['hyperparameters']['optimizer']}",
            f"- **损失函数:** {report_data['hyperparameters']['loss_function']}",
            f"- **训练轮次:** {report_data['hyperparameters']['epochs']}",
        ]
        
        # 添加优化建议（如果有）
        if report_data.get('recommendations'):
            markdown_lines.extend([
                "",
                "## 优化建议",
                ""
            ])
            
            for rec in report_data['recommendations']:
                markdown_lines.extend([
                    f"### {rec['title']}",
                    "",
                    f"**优先级:** {rec['priority']}  ",
                    f"**描述:** {rec['description']}",
                    "",
                    "**建议:**"
                ])
                
                for suggestion in rec['suggestions']:
                    markdown_lines.append(f"- {suggestion}")
                
                markdown_lines.append("")  # 空行分隔
        
        # 添加页脚
        markdown_lines.extend([
            "",
            "---",
            "",
            "*生成系统: Self AGI 训练系统*  ",
            "*报告版本: 1.0.0*"
        ])
        
        # 写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(markdown_lines))
    
    def _generate_json_report(self, report_data: Dict[str, Any], filepath: str):
        """生成JSON报告"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    def _generate_csv_report(self, report_data: Dict[str, Any], filepath: str):
        """生成CSV报告"""
        # 提取主要指标
        import csv
        
        metrics = report_data['training_metrics']
        
        rows = [
            ['指标', '值'],
            ['训练ID', metrics['training_id']],
            ['模型ID', metrics['model_id']],
            ['开始时间', metrics['start_time']],
            ['结束时间', metrics['end_time']],
            ['训练时长(小时)', f"{metrics['total_duration']/3600:.2f}"],
            ['总轮次', metrics['total_epochs']],
            ['总步数', metrics['total_steps']],
            ['最终损失', f"{metrics['final_loss']:.6f}"],
            ['最佳损失', f"{metrics['best_loss']:.6f}"],
            ['最终准确率', f"{metrics['final_accuracy']:.6f}"],
            ['最佳准确率', f"{metrics['best_accuracy']:.6f}"],
        ]
        
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)


# 全局实例
_report_generator_instance = None


def get_report_generator(config: Optional[ReportConfig] = None) -> ReportGenerator:
    """获取报告生成器单例
    
    参数:
        config: 报告配置
        
    返回:
        ReportGenerator: 报告生成器实例
    """
    global _report_generator_instance
    
    if _report_generator_instance is None:
        _report_generator_instance = ReportGenerator(config)
    
    return _report_generator_instance


__all__ = [
    "ReportGenerator",
    "get_report_generator",
    "ReportConfig",
    "ReportType",
    "ReportFormat",
    "TrainingMetrics",
    "ResourceMetrics",
    "ModelMetadata",
    "Hyperparameters",
    "DatasetInfo",
    "ChartGenerator",
]