"""
专业领域能力路由模块
处理专业领域能力管理和测试API请求
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from backend.dependencies import get_db, get_current_user
from backend.db_models.user import User
from backend.schemas.response import SuccessResponse
from backend.services.professional_capabilities_service import get_professional_capabilities_service

# 创建日志记录器
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/professional/capabilities", tags=["专业领域能力"])

# 专业领域能力定义（已弃用，现在通过服务获取）
PROFESSIONAL_CAPABILITIES = [
    {
        "id": "programming",
        "name": "编程能力",
        "description": "代码生成、代码分析、代码调试、代码优化",
        "icon": "code",
        "enabled": True,
        "status": "active",
        "performance": 85,
        "last_tested": "2026-03-09T08:59:04Z",
        "test_results": {
            "passed": 12,
            "failed": 1,
            "total": 13,
        },
        "capabilities": ["代码生成", "代码分析", "代码调试", "代码优化", "代码重构"],
    },
    {
        "id": "mathematics",
        "name": "数学能力",
        "description": "数学问题求解、符号计算、数学证明、数值计算",
        "icon": "calculator",
        "enabled": True,
        "status": "active",
        "performance": 92,
        "last_tested": "2026-03-09T08:59:04Z",
        "test_results": {
            "passed": 15,
            "failed": 0,
            "total": 15,
        },
        "capabilities": ["代数", "几何", "微积分", "概率统计", "符号计算"],
    },
    {
        "id": "physics",
        "name": "物理模拟",
        "description": "物理引擎集成、运动模拟、碰撞检测、流体模拟",
        "icon": "atom",
        "enabled": False,
        "status": "inactive",
        "performance": 45,
        "last_tested": "2026-03-09T08:59:04Z",
        "test_results": {
            "passed": 5,
            "failed": 2,
            "total": 7,
        },
        "capabilities": ["运动模拟", "碰撞检测", "刚体动力学", "粒子系统"],
    },
    {
        "id": "medical",
        "name": "医学推理",
        "description": "医学知识库、疾病诊断、治疗方案推理、医学文献分析",
        "icon": "stethoscope",
        "enabled": False,
        "status": "inactive",
        "performance": 60,
        "last_tested": "2026-03-09T08:59:04Z",
        "test_results": {
            "passed": 8,
            "failed": 3,
            "total": 11,
        },
        "capabilities": ["疾病诊断", "症状分析", "治疗方案", "药物交互"],
    },
    {
        "id": "finance",
        "name": "金融分析",
        "description": "金融数据建模、风险评估、投资策略、投资组合优化",
        "icon": "trending-up",
        "enabled": True,
        "status": "active",
        "performance": 78,
        "last_tested": "2026-03-09T08:59:04Z",
        "test_results": {
            "passed": 10,
            "failed": 2,
            "total": 12,
        },
        "capabilities": ["风险分析", "投资组合", "市场预测", "技术指标"],
    },
]

@router.get("", response_model=SuccessResponse[List[Dict[str, Any]]])
async def get_capabilities(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取专业领域能力列表"""
    try:
        logger.info(f"用户 {user.id} 获取专业领域能力列表")
        
        # 获取专业领域能力服务实例
        service = get_professional_capabilities_service()
        
        # 从服务获取能力列表
        capabilities = service.get_capabilities()
        
        return SuccessResponse.create(
            data=capabilities,
            message="专业领域能力列表获取成功"
        )
        
    except Exception as e:
        logger.error(f"获取专业领域能力列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取专业领域能力列表失败: {str(e)}"
        )

@router.get("/{capability_id}", response_model=SuccessResponse[Dict[str, Any]])
async def get_capability(
    capability_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取特定专业领域能力详情"""
    try:
        logger.info(f"用户 {user.id} 获取专业领域能力详情: {capability_id}")
        
        # 获取专业领域能力服务实例
        service = get_professional_capabilities_service()
        
        # 从服务获取能力详情
        capability = service.get_capability(capability_id)
        
        if not capability:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"未找到ID为 {capability_id} 的专业领域能力"
            )
        
        return SuccessResponse.create(
            data=capability,
            message="专业领域能力详情获取成功"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取专业领域能力详情失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取专业领域能力详情失败: {str(e)}"
        )

@router.post("/{capability_id}/test", response_model=SuccessResponse[Dict[str, Any]])
async def test_capability(
    capability_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """测试特定专业领域能力"""
    try:
        logger.info(f"用户 {user.id} 测试专业领域能力: {capability_id}")
        
        # 获取专业领域能力服务实例
        service = get_professional_capabilities_service()
        
        # 通过服务执行测试
        test_result = service.test_capability(capability_id)
        
        return SuccessResponse.create(
            data=test_result,
            message="专业领域能力测试请求已处理"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"测试专业领域能力失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"测试专业领域能力失败: {str(e)}"
        )

@router.get("/overall/status", response_model=SuccessResponse[Dict[str, Any]])
async def get_overall_status(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取专业领域能力整体状态"""
    try:
        logger.info(f"用户 {user.id} 获取专业领域能力整体状态")
        
        # 获取专业领域能力服务实例
        service = get_professional_capabilities_service()
        
        # 从服务获取能力列表
        capabilities = service.get_capabilities()
        
        # 计算整体状态
        enabled_capabilities = len([c for c in capabilities if c.get("enabled", False)])
        average_performance = 0
        if capabilities:
            average_performance = sum(c.get("performance", 0) for c in capabilities) / len(capabilities)
        
        overall_status = {
            "total_capabilities": len(capabilities),
            "enabled_capabilities": enabled_capabilities,
            "average_performance": round(average_performance, 2),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        
        return SuccessResponse.create(
            data=overall_status,
            message="专业领域能力整体状态获取成功"
        )
        
    except Exception as e:
        logger.error(f"获取专业领域能力整体状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取专业领域能力整体状态失败: {str(e)}"
        )