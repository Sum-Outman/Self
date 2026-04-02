"""
诊断路由模块 - 提供系统诊断API

功能：
1. 运行全面系统诊断
2. 获取诊断报告
3. 获取诊断历史
4. 管理自动诊断

基于修复计划三中的P1优先级问题："系统诊断功能实现验证"
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone
import logging

from backend.dependencies import get_db, get_current_user, get_current_admin
from backend.db_models.user import User
from backend.schemas.response import StandardResponse
from backend.services.system_diagnostic_service import (
    get_system_diagnostic_service,
    DiagnosticCategory,
    DiagnosticStatus,
)

router = APIRouter(prefix="/api/diagnostic", tags=["系统诊断"])

# 配置日志
logger = logging.getLogger(__name__)


@router.post("/run", response_model=Dict[str, Any])
async def run_system_diagnosis(
    categories: Optional[List[str]] = Body(
        default=None,
        description="诊断类别列表（可选），如 ['system_health', 'hardware', 'software', 'performance', 'security', 'network', 'database', 'api']"
    ),
    user: User = Depends(get_current_admin),  # 仅管理员可运行诊断
):
    """运行系统诊断
    
    运行全面的系统诊断，包括系统健康、硬件、软件、性能、安全等多个方面
    返回诊断报告ID，可用于查询诊断结果
    """
    try:
        logger.info(f"管理员 {user.username} 请求运行系统诊断，类别: {categories}")
        
        # 获取诊断服务实例
        diagnostic_service = get_system_diagnostic_service()
        
        # 运行诊断
        report_id = diagnostic_service.run_diagnosis(categories)
        
        logger.info(f"诊断启动成功，报告ID: {report_id}")
        
        return {
            "success": True,
            "report_id": report_id,
            "message": "系统诊断已启动",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "estimated_time": "诊断可能需要几分钟时间完成",
        }
        
    except RuntimeError as e:
        logger.warning(f"诊断正在运行中: {e}")
        return {
            "success": False,
            "report_id": None,
            "message": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "busy",
            "error": "DIAGNOSIS_IN_PROGRESS",
        }
        
    except Exception as e:
        logger.error(f"运行系统诊断失败: {e}")
        return {
            "success": False,
            "report_id": None,
            "message": f"运行系统诊断失败: {str(e)}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "error": "DIAGNOSIS_FAILED",
        }


@router.get("/report/{report_id}", response_model=Dict[str, Any])
async def get_diagnostic_report(
    report_id: str,
    user: User = Depends(get_current_admin),  # 仅管理员可查看诊断报告
):
    """获取诊断报告
    
    根据报告ID获取详细的诊断报告
    """
    try:
        logger.info(f"管理员 {user.username} 请求获取诊断报告: {report_id}")
        
        # 获取诊断服务实例
        diagnostic_service = get_system_diagnostic_service()
        
        # 获取诊断报告
        report = diagnostic_service.get_diagnostic_report(report_id)
        
        if not report:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"诊断报告不存在: {report_id}"
            )
        
        return {
            "success": True,
            "report": report,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"获取诊断报告失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取诊断报告失败: {str(e)}"
        )


@router.get("/history", response_model=Dict[str, Any])
async def get_diagnostic_history(
    limit: int = Query(default=10, ge=1, le=100, description="数量限制 (1-100)"),
    user: User = Depends(get_current_admin),  # 仅管理员可查看诊断历史
):
    """获取诊断历史
    
    获取最近的诊断报告历史
    """
    try:
        logger.info(f"管理员 {user.username} 请求获取诊断历史，限制: {limit}")
        
        # 获取诊断服务实例
        diagnostic_service = get_system_diagnostic_service()
        
        # 获取诊断历史
        history = diagnostic_service.get_diagnostic_history(limit)
        
        return {
            "success": True,
            "history": history,
            "count": len(history),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
    except Exception as e:
        logger.error(f"获取诊断历史失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取诊断历史失败: {str(e)}"
        )


@router.get("/categories", response_model=Dict[str, Any])
async def get_diagnostic_categories(
    user: User = Depends(get_current_user),  # 普通用户也可查看诊断类别
):
    """获取诊断类别
    
    获取可用的诊断类别和描述
    """
    try:
        categories = []
        
        for category in DiagnosticCategory:
            categories.append({
                "id": category.value,
                "name": category.name,
                "description": _get_category_description(category),
                "enabled_by_default": _is_category_enabled_by_default(category),
            })
        
        return {
            "success": True,
            "categories": categories,
            "count": len(categories),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
    except Exception as e:
        logger.error(f"获取诊断类别失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取诊断类别失败: {str(e)}"
        )


@router.get("/status", response_model=Dict[str, Any])
async def get_diagnostic_service_status(
    user: User = Depends(get_current_user),  # 普通用户也可查看诊断服务状态
):
    """获取诊断服务状态
    
    获取诊断服务的当前状态和统计信息
    """
    try:
        # 获取诊断服务实例
        diagnostic_service = get_system_diagnostic_service()
        
        # 获取诊断历史
        history = diagnostic_service.get_diagnostic_history(limit=5)
        
        # 分析状态
        total_reports = len(history)
        running_count = sum(1 for report in history if report.get("status") == DiagnosticStatus.RUNNING.value)
        completed_count = sum(1 for report in history if report.get("status") == DiagnosticStatus.COMPLETED.value)
        failed_count = sum(1 for report in history if report.get("status") == DiagnosticStatus.FAILED.value)
        
        # 检查是否正在运行诊断
        diagnostic_service_instance = get_system_diagnostic_service()
        diagnosis_in_progress = getattr(diagnostic_service_instance, "diagnosis_in_progress", False)
        
        return {
            "success": True,
            "status": {
                "diagnosis_in_progress": diagnosis_in_progress,
                "auto_diagnosis_enabled": getattr(diagnostic_service_instance, "running", False),
                "total_reports": total_reports,
                "running_reports": running_count,
                "completed_reports": completed_count,
                "failed_reports": failed_count,
                "service_health": "healthy" if total_reports > 0 else "unknown",
            },
            "recent_reports": history,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
    except Exception as e:
        logger.error(f"获取诊断服务状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取诊断服务状态失败: {str(e)}"
        )


@router.post("/auto/start", response_model=Dict[str, Any])
async def start_auto_diagnosis(
    user: User = Depends(get_current_admin),  # 仅管理员可管理自动诊断
):
    """启动自动诊断
    
    启动定期自动诊断服务
    """
    try:
        logger.info(f"管理员 {user.username} 请求启动自动诊断")
        
        # 获取诊断服务实例
        diagnostic_service = get_system_diagnostic_service()
        
        # 启动自动诊断
        diagnostic_service.start_auto_diagnosis()
        
        return {
            "success": True,
            "message": "自动诊断已启动",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "interval_seconds": diagnostic_service.config.get("diagnosis_interval", 300.0),
        }
        
    except Exception as e:
        logger.error(f"启动自动诊断失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"启动自动诊断失败: {str(e)}"
        )


@router.post("/auto/stop", response_model=Dict[str, Any])
async def stop_auto_diagnosis(
    user: User = Depends(get_current_admin),  # 仅管理员可管理自动诊断
):
    """停止自动诊断
    
    停止定期自动诊断服务
    """
    try:
        logger.info(f"管理员 {user.username} 请求停止自动诊断")
        
        # 获取诊断服务实例
        diagnostic_service = get_system_diagnostic_service()
        
        # 停止自动诊断
        diagnostic_service.stop_auto_diagnosis()
        
        return {
            "success": True,
            "message": "自动诊断已停止",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
    except Exception as e:
        logger.error(f"停止自动诊断失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"停止自动诊断失败: {str(e)}"
        )


@router.get("/quick", response_model=Dict[str, Any])
async def run_quick_diagnosis(
    user: User = Depends(get_current_user),  # 普通用户也可运行快速诊断
):
    """运行快速诊断
    
    运行快速系统诊断，只检查关键系统组件
    """
    try:
        logger.info(f"用户 {user.username} 请求运行快速诊断")
        
        # 获取诊断服务实例
        diagnostic_service = get_system_diagnostic_service()
        
        # 运行快速诊断（只检查关键类别）
        quick_categories = ["system", "resource", "performance"]
        report_id = diagnostic_service.run_diagnosis(quick_categories)
        
        return {
            "success": True,
            "report_id": report_id,
            "message": "快速诊断已启动",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "categories": quick_categories,
        }
        
    except RuntimeError as e:
        logger.warning(f"诊断正在运行中: {e}")
        return {
            "success": False,
            "report_id": None,
            "message": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "busy",
            "error": "DIAGNOSIS_IN_PROGRESS",
        }
        
    except Exception as e:
        logger.error(f"运行快速诊断失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"运行快速诊断失败: {str(e)}"
        )


@router.get("/export/json/{report_id}", response_model=Dict[str, Any])
async def export_diagnostic_report_json(
    report_id: str,
    user: User = Depends(get_current_admin),  # 仅管理员可导出诊断报告
):
    """导出诊断报告为JSON格式
    
    将诊断报告导出为JSON格式，可直接下载或用于进一步分析
    """
    try:
        logger.info(f"管理员 {user.username} 请求导出诊断报告JSON: {report_id}")
        
        # 获取诊断服务实例
        diagnostic_service = get_system_diagnostic_service()
        
        # 获取诊断报告
        report = diagnostic_service.get_diagnostic_report(report_id)
        
        if not report:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"诊断报告不存在: {report_id}"
            )
        
        # 导出为JSON
        json_content = diagnostic_service.export_report_to_json(report)
        
        # 创建临时文件名
        import tempfile
        import os
        temp_dir = tempfile.gettempdir()
        filename = f"diagnostic_report_{report_id}.json"
        filepath = os.path.join(temp_dir, filename)
        
        # 保存到临时文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(json_content)
        
        return {
            "success": True,
            "message": "诊断报告已导出为JSON",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "report_id": report_id,
            "format": "json",
            "download_url": f"/api/diagnostic/download/{filename}",
            "filename": filename,
        }
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"导出诊断报告JSON失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"导出诊断报告JSON失败: {str(e)}"
        )


@router.get("/export/html/{report_id}", response_model=Dict[str, Any])
async def export_diagnostic_report_html(
    report_id: str,
    user: User = Depends(get_current_admin),  # 仅管理员可导出诊断报告
):
    """导出诊断报告为HTML格式
    
    将诊断报告导出为HTML格式，适合在浏览器中查看和打印
    """
    try:
        logger.info(f"管理员 {user.username} 请求导出诊断报告HTML: {report_id}")
        
        # 获取诊断服务实例
        diagnostic_service = get_system_diagnostic_service()
        
        # 获取诊断报告
        report = diagnostic_service.get_diagnostic_report(report_id)
        
        if not report:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"诊断报告不存在: {report_id}"
            )
        
        # 创建临时文件名
        import tempfile
        import os
        temp_dir = tempfile.gettempdir()
        filename = f"diagnostic_report_{report_id}.html"
        filepath = os.path.join(temp_dir, filename)
        
        # 导出为HTML
        html_content = diagnostic_service.export_report_to_html(report, filepath)
        
        return {
            "success": True,
            "message": "诊断报告已导出为HTML",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "report_id": report_id,
            "format": "html",
            "download_url": f"/api/diagnostic/download/{filename}",
            "filename": filename,
        }
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"导出诊断报告HTML失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"导出诊断报告HTML失败: {str(e)}"
        )


@router.get("/export/text/{report_id}", response_model=Dict[str, Any])
async def export_diagnostic_report_text(
    report_id: str,
    user: User = Depends(get_current_admin),  # 仅管理员可导出诊断报告
):
    """导出诊断报告为文本格式
    
    将诊断报告导出为纯文本格式，适合日志记录和简单查看
    """
    try:
        logger.info(f"管理员 {user.username} 请求导出诊断报告文本: {report_id}")
        
        # 获取诊断服务实例
        diagnostic_service = get_system_diagnostic_service()
        
        # 获取诊断报告
        report = diagnostic_service.get_diagnostic_report(report_id)
        
        if not report:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"诊断报告不存在: {report_id}"
            )
        
        # 创建临时文件名
        import tempfile
        import os
        temp_dir = tempfile.gettempdir()
        filename = f"diagnostic_report_{report_id}.txt"
        filepath = os.path.join(temp_dir, filename)
        
        # 导出为文本
        text_content = diagnostic_service.export_report_to_text(report, filepath)
        
        return {
            "success": True,
            "message": "诊断报告已导出为文本",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "report_id": report_id,
            "format": "text",
            "download_url": f"/api/diagnostic/download/{filename}",
            "filename": filename,
        }
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"导出诊断报告文本失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"导出诊断报告文本失败: {str(e)}"
        )


@router.get("/download/{filename}", response_model=None)
async def download_diagnostic_report(
    filename: str,
    user: User = Depends(get_current_admin),  # 仅管理员可下载诊断报告
):
    """下载导出的诊断报告文件
    
    下载通过导出接口生成的诊断报告文件
    """
    try:
        import tempfile
        import os
        from fastapi.responses import FileResponse
        
        # 验证文件名
        if not filename.startswith("diagnostic_report_"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="无效的文件名"
            )
        
        # 检查文件是否存在
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)
        
        if not os.path.exists(filepath):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"文件不存在: {filename}"
            )
        
        logger.info(f"管理员 {user.username} 下载诊断报告文件: {filename}")
        
        # 返回文件
        return FileResponse(
            path=filepath,
            filename=filename,
            media_type="application/octet-stream"
        )
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"下载诊断报告文件失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"下载诊断报告文件失败: {str(e)}"
        )


def _get_category_description(category: DiagnosticCategory) -> str:
    """获取诊断类别描述"""
    descriptions = {
        DiagnosticCategory.SYSTEM: "系统健康诊断 - 检查系统整体健康状态、活跃问题、修复历史",
        DiagnosticCategory.HARDWARE: "硬件诊断 - 检查硬件设备连接、状态和性能",
        DiagnosticCategory.RESOURCE: "资源诊断 - 检查CPU、内存、磁盘等系统资源使用情况",
        DiagnosticCategory.PERFORMANCE: "性能诊断 - 检查系统性能指标、资源使用和瓶颈",
        DiagnosticCategory.SECURITY: "安全诊断 - 检查安全配置、认证状态和潜在风险",
        DiagnosticCategory.NETWORK: "网络诊断 - 检查网络连接、API端点和通信状态",
        DiagnosticCategory.DATABASE: "数据库诊断 - 检查数据库连接、性能和完整性",
        DiagnosticCategory.API: "API诊断 - 检查API服务可用性、响应时间和错误率",
    }
    return descriptions.get(category, "未知诊断类别")


def _is_category_enabled_by_default(category: DiagnosticCategory) -> bool:
    """检查诊断类别是否默认启用"""
    default_enabled_categories = [
        DiagnosticCategory.SYSTEM,
        DiagnosticCategory.RESOURCE,
        DiagnosticCategory.PERFORMANCE,
        DiagnosticCategory.SECURITY,
    ]
    return category in default_enabled_categories