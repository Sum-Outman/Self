# 训练API服务器
from training.trainer import AGITrainer, TrainingConfig, TrainingDataset
from models.transformer.config import ModelConfig
from models.transformer.self_agi_model import SelfAGIModel
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
from datetime import datetime
import logging
import os
from pathlib import Path
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))


# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TrainingServer")


# 配置
class TrainingServerConfig:
    """训练服务器配置"""

    # 数据库配置
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql://self_agi:self_agi_password@localhost:5432/self_agi",
    )
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    MONGODB_URL = os.getenv(
        "MONGODB_URL", "mongodb://admin:admin_password@localhost:27017/self_agi"
    )

    # 训练配置
    TRAINING_GPU_ENABLED = os.getenv("TRAINING_GPU_ENABLED", "true").lower() == "true"
    TRAINING_MODEL_DIR = os.getenv("TRAINING_MODEL_DIR", "./models")
    TRAINING_CHECKPOINT_DIR = os.getenv("TRAINING_CHECKPOINT_DIR", "./checkpoints")
    TRAINING_LOG_DIR = os.getenv("TRAINING_LOG_DIR", "./logs/training")

    # 外部API配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    EXTERNAL_TRAINING_API_URL = os.getenv("EXTERNAL_TRAINING_API_URL", "")

    # 服务器配置
    SERVER_HOST = os.getenv("TRAINING_SERVER_HOST", "0.0.0.0")
    SERVER_PORT = int(os.getenv("TRAINING_SERVER_PORT", "8001"))

    # 安全配置
    API_KEY_SECRET = os.getenv("API_KEY_SECRET", "training_api_secret_key")
    RATE_LIMIT_PER_MINUTE = int(os.getenv("TRAINING_RATE_LIMIT", "60"))


# 创建FastAPI应用
app = FastAPI(
    title="Self AGI 训练API", description="Self AGI系统的训练API服务", version="1.0.0"
)

# 安全认证
security = HTTPBearer()

# 全局变量
training_manager = None
api_keys = {"test_key": "测试API密钥"}


# 数据模型
class TrainingRequest(BaseModel):
    """训练请求"""

    model_configuration: Optional[Dict[str, Any]] = None
    training_config: Optional[Dict[str, Any]] = None
    training_data: Optional[List[Dict[str, Any]]] = None
    training_mode: str = "supervised"
    use_external_api: bool = False
    external_api_config: Optional[Dict[str, Any]] = None


class RobotTrainingRequest(BaseModel):
    """机器人训练请求"""

    robot_config: Dict[str, Any] = Field(description="机器人配置")
    training_tasks: List[str] = Field(
        default=["walking", "balance", "manipulation"], description="训练任务"
    )


class LearningModeRequest(BaseModel):
    """学习模式请求"""

    self_learning_enabled: bool = Field(default=True, description="是否启用自我学习")
    internet_learning_enabled: bool = Field(
        default=False, description="是否启用上网学习"
    )
    knowledge_base_learning_enabled: bool = Field(
        default=True, description="是否启用知识库学习"
    )
    specific_content: Optional[List[str]] = Field(
        default=None, description="指定学习内容"
    )


class TrainingStatusResponse(BaseModel):
    """训练状态响应"""

    status: str = Field(description="训练状态")
    progress: float = Field(description="训练进度")
    current_epoch: int = Field(description="当前轮次")
    total_epochs: int = Field(description="总轮次")
    current_loss: float = Field(description="当前损失")
    best_loss: float = Field(description="最佳损失")
    learning_rate: float = Field(description="学习率")
    training_mode: str = Field(description="训练模式")


class TrainingResultResponse(BaseModel):
    """训练结果响应"""

    success: bool = Field(description="是否成功")
    message: str = Field(description="消息")
    checkpoint_path: Optional[str] = Field(default=None, description="检查点路径")
    model_performance: Optional[Dict[str, Any]] = Field(
        default=None, description="模型性能指标"
    )


# 依赖项
def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """验证API密钥"""
    token = credentials.credentials

    # 支持两种验证方式：
    # 1. 直接使用API密钥值（向后兼容）
    # 2. 使用API密钥名称，然后查找对应的值
    if token in api_keys.values():
        # 直接匹配值（传统方式）
        return token
    elif token in api_keys:
        # 匹配键，返回对应的值
        return api_keys[token]
    else:
        raise HTTPException(status_code=401, detail="无效的API密钥")


# 训练管理器
class TrainingManager:
    """训练管理器"""

    def __init__(self):
        self.current_trainer: Optional[AGITrainer] = None
        self.training_task = None
        self.training_status = "idle"  # idle, running, paused, completed, error
        self.training_progress = 0.0
        self.training_results = {}

    def initialize_trainer(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        training_data: Optional[List[Dict[str, Any]]] = None,
    ) -> AGITrainer:
        """初始化训练器"""
        try:
            # 创建模型配置
            if model_config is None:
                model_config = {
                    "hidden_size": 768,
                    "num_hidden_layers": 12,
                    "num_attention_heads": 12,
                    "multimodal_enabled": True,
                }

            config = ModelConfig.from_dict(model_config)
            model = SelfAGIModel(config)

            # 创建训练配置
            if training_config is None:
                training_config = {
                    "batch_size": 32,
                    "learning_rate": 1e-4,
                    "num_epochs": 10,
                    "checkpoint_dir": "checkpoints",
                    "log_dir": "logs",
                }

            trainer_config = TrainingConfig(**training_config)

            # 创建数据集
            dataset = None
            if training_data:
                dataset = TrainingDataset(training_data)

            # 创建训练器
            trainer = AGITrainer(model, trainer_config, dataset)

            self.current_trainer = trainer
            self.training_status = "initialized"

            logger.info("训练器初始化成功")
            return trainer

        except Exception as e:
            logger.error(f"训练器初始化失败: {e}")
            raise HTTPException(status_code=500, detail=f"训练器初始化失败: {str(e)}")

    def start_training(self, background_tasks: BackgroundTasks):
        """开始训练"""
        if self.current_trainer is None:
            raise HTTPException(status_code=400, detail="训练器未初始化")

        if self.training_status == "running":
            raise HTTPException(status_code=400, detail="训练已在运行中")

        # 在后台运行训练
        self.training_status = "running"
        self.training_progress = 0.0

        background_tasks.add_task(self._run_training)

        logger.info("训练任务已启动")
        return {"message": "训练任务已启动", "status": "running"}

    async def _run_training(self):
        """运行训练任务 - 真实训练实现，禁止使用模拟数据"""
        try:
            if self.current_trainer is None:
                raise RuntimeError("训练器未初始化")

            logger.info("开始真实训练（禁止使用模拟数据）")

            # 在后台线程中运行训练，避免阻塞事件循环
            import asyncio
            import threading
            import time

            # 训练完成标志
            training_completed = False
            training_error = None

            def run_training():
                """在后台线程中运行训练"""
                nonlocal training_completed, training_error
                try:
                    # 调用训练器的真实训练方法
                    self.current_trainer.train()
                    training_completed = True
                except Exception as e:
                    training_error = e
                    logger.error(f"训练失败: {e}")

            # 启动训练线程
            training_thread = threading.Thread(target=run_training, daemon=True)
            training_thread.start()

            # 轮询训练进度
            start_time = time.time()
            last_log_time = start_time

            while training_thread.is_alive():
                # 检查训练状态
                if self.training_status != "running":
                    logger.warning("训练被暂停或取消")
                    break

                # 获取训练器状态
                try:
                    trainer_status = self.current_trainer.get_training_status()
                    # 更新进度
                    if "progress" in trainer_status:
                        self.training_progress = trainer_status["progress"]
                    elif (
                        "current_epoch" in trainer_status
                        and "total_epochs" in trainer_status
                    ):
                        if trainer_status["total_epochs"] > 0:
                            self.training_progress = (
                                trainer_status["current_epoch"]
                                / trainer_status["total_epochs"]
                            )

                    # 定期记录日志
                    current_time = time.time()
                    if current_time - last_log_time > 5.0:  # 每5秒记录一次
                        logger.info(
                            f"训练进度: {self.training_progress:.1%}, "
                            f"当前损失: {trainer_status.get('current_loss', 'N/A')}"
                        )
                        last_log_time = current_time
                except Exception as e:
                    logger.warning(f"获取训练状态失败: {e}")

                # 短暂休眠
                await asyncio.sleep(1.0)

            # 训练完成
            if training_error:
                raise training_error

            if training_completed:
                self.training_status = "completed"
                self.training_progress = 1.0

                # 获取最终结果
                try:
                    trainer_status = self.current_trainer.get_training_status()
                    final_loss = trainer_status.get("current_loss", 0.0)
                    checkpoint_path = trainer_status.get(
                        "checkpoint_path", "checkpoints/model_best.pt"
                    )
                    training_time_seconds = time.time() - start_time
                    training_time_str = f"{int(training_time_seconds //                                                60)}分{int(training_time_seconds %                                                          60)}秒"
                except Exception as e:
                    logger.error(f"获取训练结果失败: {e}")
                    final_loss = 0.0
                    checkpoint_path = "checkpoints/model_best.pt"
                    training_time_str = "未知"

                self.training_results = {
                    "final_loss": final_loss,
                    "checkpoint_path": checkpoint_path,
                    "training_time": training_time_str,
                }

                logger.info(
                    f"训练任务完成，最终损失: {final_loss:.4f}, 时间: {training_time_str}"
                )
            else:
                logger.warning("训练未完成")

        except Exception as e:
            self.training_status = "error"
            logger.error(f"训练任务失败: {e}")
            raise

    def get_status(self) -> Dict[str, Any]:
        """获取训练状态"""
        if self.current_trainer is None:
            return {"status": "idle", "progress": 0.0, "message": "训练器未初始化"}

        trainer_status = self.current_trainer.get_training_status()

        return {
            "status": self.training_status,
            "progress": self.training_progress,
            "current_epoch": trainer_status["current_epoch"],
            "total_epochs": self.current_trainer.config.num_epochs,
            "current_loss": trainer_status.get("current_loss", 0.0),
            "best_loss": trainer_status["best_loss"],
            "learning_rate": trainer_status["learning_rate"],
            "training_mode": trainer_status["training_mode"],
            "self_learning": trainer_status["self_learning_enabled"],
            "internet_learning": trainer_status["internet_learning_enabled"],
            "knowledge_base_learning": trainer_status[
                "knowledge_base_learning_enabled"
            ],
        }

    def pause_training(self):
        """暂停训练"""
        if self.training_status == "running":
            self.training_status = "paused"
            logger.info("训练已暂停")
            return {"message": "训练已暂停", "status": "paused"}
        else:
            raise HTTPException(status_code=400, detail="训练未运行")

    def resume_training(self):
        """恢复训练"""
        if self.training_status == "paused":
            self.training_status = "running"
            logger.info("训练已恢复")
            return {"message": "训练已恢复", "status": "running"}
        else:
            raise HTTPException(status_code=400, detail="训练未暂停")

    def stop_training(self):
        """停止训练"""
        if self.training_status in ["running", "paused"]:
            self.training_status = "stopped"
            logger.info("训练已停止")
            return {"message": "训练已停止", "status": "stopped"}
        else:
            raise HTTPException(status_code=400, detail="训练未运行")


# 初始化训练管理器
training_manager = TrainingManager()


# API路由
@app.get("/")
async def root():
    """根端点"""
    return {
        "service": "Self AGI Training API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/docs - API文档",
            "/train - 开始训练",
            "/status - 训练状态",
            "/robot/train - 机器人训练",
        ],
    }


@app.post("/train/initialize", dependencies=[Depends(verify_api_key)])
async def initialize_training(
    request: TrainingRequest, background_tasks: BackgroundTasks
):
    """初始化训练"""
    try:
        trainer = training_manager.initialize_trainer(
            model_config=request.model_configuration,
            training_config=request.training_config,
            training_data=request.training_data,
        )

        # 设置训练模式
        if request.use_external_api and request.external_api_config:
            trainer.train_with_external_api(request.external_api_config)

        return {
            "success": True,
            "message": "训练器初始化成功",
            "training_mode": request.training_mode,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train/start", dependencies=[Depends(verify_api_key)])
async def start_training(background_tasks: BackgroundTasks):
    """开始训练"""
    return training_manager.start_training(background_tasks)


@app.get("/train/status", dependencies=[Depends(verify_api_key)])
async def get_training_status() -> TrainingStatusResponse:
    """获取训练状态"""
    status = training_manager.get_status()

    return TrainingStatusResponse(
        status=status["status"],
        progress=status["progress"],
        current_epoch=status["current_epoch"],
        total_epochs=status["total_epochs"],
        current_loss=status["current_loss"],
        best_loss=status["best_loss"],
        learning_rate=status["learning_rate"],
        training_mode=status["training_mode"],
    )


@app.post("/train/pause", dependencies=[Depends(verify_api_key)])
async def pause_training():
    """暂停训练"""
    return training_manager.pause_training()


@app.post("/train/resume", dependencies=[Depends(verify_api_key)])
async def resume_training():
    """恢复训练"""
    return training_manager.resume_training()


@app.post("/train/stop", dependencies=[Depends(verify_api_key)])
async def stop_training():
    """停止训练"""
    return training_manager.stop_training()


@app.post("/train/learning/mode", dependencies=[Depends(verify_api_key)])
async def set_learning_mode(request: LearningModeRequest):
    """设置学习模式"""
    if training_manager.current_trainer is None:
        raise HTTPException(status_code=400, detail="训练器未初始化")

    trainer = training_manager.current_trainer

    # 设置学习模式
    trainer.enable_self_learning(request.self_learning_enabled)
    trainer.enable_internet_learning(request.internet_learning_enabled)
    trainer.enable_knowledge_base_learning(
        request.knowledge_base_learning_enabled, request.specific_content
    )

    return {
        "success": True,
        "message": "学习模式已更新",
        "self_learning": request.self_learning_enabled,
        "internet_learning": request.internet_learning_enabled,
        "knowledge_base_learning": request.knowledge_base_learning_enabled,
        "specific_content": request.specific_content,
    }


@app.post("/robot/train", dependencies=[Depends(verify_api_key)])
async def train_robot(request: RobotTrainingRequest, background_tasks: BackgroundTasks):
    """训练人形机器人"""
    if training_manager.current_trainer is None:
        raise HTTPException(status_code=400, detail="训练器未初始化")

    # 在后台运行机器人训练
    async def run_robot_training():
        try:
            training_manager.current_trainer.train_humanoid_robot(
                {
                    "type": request.robot_config.get("type", "bipedal"),
                    "tasks": request.training_tasks,
                }
            )
        except Exception as e:
            logger.error(f"机器人训练失败: {e}")

    background_tasks.add_task(run_robot_training)

    return {
        "success": True,
        "message": "机器人训练任务已启动",
        "robot_config": request.robot_config,
        "training_tasks": request.training_tasks,
    }


@app.post("/train/external", dependencies=[Depends(verify_api_key)])
async def train_with_external_api(request: TrainingRequest):
    """使用外部API进行训练"""
    if training_manager.current_trainer is None:
        raise HTTPException(status_code=400, detail="训练器未初始化")

    if not request.use_external_api or not request.external_api_config:
        raise HTTPException(status_code=400, detail="需要提供外部API配置")

    try:
        training_manager.current_trainer.train_with_external_api(
            request.external_api_config
        )

        return {
            "success": True,
            "message": "外部API训练任务已启动",
            "api_config": request.external_api_config,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/train/results", dependencies=[Depends(verify_api_key)])
async def get_training_results():
    """获取训练结果"""
    if training_manager.training_status != "completed":
        raise HTTPException(status_code=400, detail="训练未完成")

    if not training_manager.training_results:
        raise HTTPException(status_code=404, detail="训练结果不存在")

    # 验证必要字段
    required_fields = [
        "checkpoint_path",
        "final_loss",
        "training_time",
        "model_size",
        "accuracy",
    ]
    for field in required_fields:
        if field not in training_manager.training_results:
            raise HTTPException(
                status_code=500, detail=f"训练结果缺少必要字段: {field}"
            )

    return TrainingResultResponse(
        success=True,
        message="训练完成",
        checkpoint_path=training_manager.training_results["checkpoint_path"],
        model_performance={
            "final_loss": training_manager.training_results["final_loss"],
            "training_time": training_manager.training_results["training_time"],
            "model_size": training_manager.training_results["model_size"],
            "accuracy": training_manager.training_results["accuracy"],
        },
    )


@app.post("/train/checkpoint/save", dependencies=[Depends(verify_api_key)])
async def save_checkpoint():
    """保存检查点"""
    if training_manager.current_trainer is None:
        raise HTTPException(status_code=400, detail="训练器未初始化")

    try:
        training_manager.current_trainer.save_checkpoint()

        return {
            "success": True,
            "message": "检查点已保存",
            "checkpoint_dir": training_manager.current_trainer.config.checkpoint_dir,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train/checkpoint/load", dependencies=[Depends(verify_api_key)])
async def load_checkpoint(checkpoint_path: str):
    """加载检查点"""
    if training_manager.current_trainer is None:
        raise HTTPException(status_code=400, detail="训练器未初始化")

    try:
        training_manager.current_trainer.load_checkpoint(checkpoint_path)

        return {
            "success": True,
            "message": "检查点已加载",
            "checkpoint_path": checkpoint_path,
            "global_step": training_manager.current_trainer.global_step,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "Self AGI Training API",
        "timestamp": datetime.now().isoformat(),
    }


# 启动服务器
if __name__ == "__main__":
    config = TrainingServerConfig()
    uvicorn.run(
        "training_server:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        reload=True,
        log_level="info",
    )
