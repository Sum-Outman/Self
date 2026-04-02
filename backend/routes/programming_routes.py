"""
编程能力路由模块
处理代码分析、代码生成、代码调试等编程相关功能
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import json

from backend.dependencies import get_db, get_current_user, get_current_admin
from backend.core.config import Config

# 导入专业领域能力管理器
try:
    from training.professional_domain_capabilities import (
        get_global_professional_domain_manager,
        ProgrammingLanguage,
    )
    PROFESSIONAL_CAPABILITIES_AVAILABLE = True
except ImportError as e:
    PROFESSIONAL_CAPABILITIES_AVAILABLE = False
    print(f"警告: 无法导入专业领域能力管理器: {e}")
    # 创建虚拟函数和枚举以避免导入错误
    def get_global_professional_domain_manager():
        raise ImportError("专业领域能力管理器不可用")
    
    # 定义替代的编程语言枚举
    from enum import Enum
    class ProgrammingLanguage(Enum):
        PYTHON = "python"
        JAVASCRIPT = "javascript"
        JAVA = "java"
        CPP = "c++"
        RUST = "rust"
        GO = "go"
        SQL = "sql"

router = APIRouter(prefix="/api/programming", tags=["编程能力"])

# 安全
security = HTTPBearer()


# 数据模型
class CodeAnalysisRequest(BaseModel):
    """代码分析请求"""
    code: str = Field(..., description="要分析的源代码")
    language: str = Field(default="python", description="编程语言 (python, javascript, java, c++, rust, go, sql)")
    analyze_complexity: bool = Field(default=True, description="是否分析代码复杂度")
    analyze_quality: bool = Field(default=True, description="是否分析代码质量")
    detect_bugs: bool = Field(default=True, description="是否检测bug")
    detect_security: bool = Field(default=True, description="是否检测安全漏洞")
    detect_performance: bool = Field(default=True, description="是否检测性能问题")


class CodeGenerationRequest(BaseModel):
    """代码生成请求"""
    description: str = Field(..., description="代码功能描述")
    language: str = Field(default="python", description="编程语言")
    code_type: str = Field(default="function", description="代码类型 (function, class, algorithm, test)")
    include_analysis: bool = Field(default=True, description="是否包含代码分析")


class CodeDebugRequest(BaseModel):
    """代码调试请求"""
    code: str = Field(..., description="要调试的源代码")
    language: str = Field(default="python", description="编程语言")
    error_message: Optional[str] = Field(default=None, description="错误信息（如果有）")
    include_fixed_code: bool = Field(default=True, description="是否包含修复后的代码")


class ProgrammingReportResponse(BaseModel):
    """编程能力报告响应"""
    success: bool
    timestamp: str
    capabilities: Dict[str, Any]
    overall_status: Dict[str, bool]
    message: str


class CodeAnalysisResponse(BaseModel):
    """代码分析响应"""
    success: bool
    timestamp: str
    analysis_result: Dict[str, Any]
    suggestions: List[str]
    message: str


class CodeGenerationResponse(BaseModel):
    """代码生成响应"""
    success: bool
    timestamp: str
    generated_code: str
    analysis_result: Optional[Dict[str, Any]]
    quality_score: float
    message: str


class CodeDebugResponse(BaseModel):
    """代码调试响应"""
    success: bool
    timestamp: str
    original_code: str
    fixed_code: Optional[str]
    bugs_detected: List[Dict[str, Any]]
    suggested_fixes: List[Dict[str, Any]]
    message: str


@router.get("/capabilities", response_model=ProgrammingReportResponse)
async def get_programming_capabilities(
    db: Session = Depends(get_db),
    user = Depends(get_current_user),
):
    """获取编程能力报告"""
    if not PROFESSIONAL_CAPABILITIES_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="编程能力服务不可用，请检查依赖安装"
        )
    
    try:
        # 获取专业领域能力管理器
        manager = get_global_professional_domain_manager()
        
        # 获取编程能力报告
        programming_report = manager.programming_manager.get_programming_report()
        
        # 获取综合报告中的状态信息
        comprehensive_report = manager.get_comprehensive_report()
        
        return ProgrammingReportResponse(
            success=True,
            timestamp=datetime.now(timezone.utc).isoformat(),
            capabilities=programming_report,
            overall_status=comprehensive_report.get("overall_status", {}),
            message="编程能力报告获取成功"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取编程能力报告失败: {str(e)}"
        )


@router.post("/analyze", response_model=CodeAnalysisResponse)
async def analyze_code(
    request: CodeAnalysisRequest,
    db: Session = Depends(get_db),
    user = Depends(get_current_user),
):
    """分析代码"""
    if not PROFESSIONAL_CAPABILITIES_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="编程能力服务不可用，请检查依赖安装"
        )
    
    try:
        # 获取专业领域能力管理器
        manager = get_global_professional_domain_manager()
        
        # 转换语言枚举
        try:
            language_enum = ProgrammingLanguage(request.language.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的编程语言: {request.language}。支持的语言: {', '.join([lang.value for lang in ProgrammingLanguage])}"
            )
        
        # 分析代码
        analysis_result = manager.programming_manager.analyze_code(
            code=request.code,
            language=language_enum
        )
        
        # 转换为字典
        result_dict = analysis_result.to_dict()
        
        # 提取建议
        suggestions = []
        if result_dict.get("optimization_suggestions"):
            suggestions.extend(result_dict["optimization_suggestions"])
        if result_dict.get("refactoring_suggestions"):
            suggestions.extend(result_dict["refactoring_suggestions"])
        
        return CodeAnalysisResponse(
            success=True,
            timestamp=datetime.now(timezone.utc).isoformat(),
            analysis_result=result_dict,
            suggestions=suggestions,
            message="代码分析完成"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"代码分析失败: {str(e)}"
        )


@router.post("/generate", response_model=CodeGenerationResponse)
async def generate_code(
    request: CodeGenerationRequest,
    db: Session = Depends(get_db),
    user = Depends(get_current_user),
):
    """生成代码"""
    if not PROFESSIONAL_CAPABILITIES_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="编程能力服务不可用，请检查依赖安装"
        )
    
    try:
        # 获取专业领域能力管理器
        manager = get_global_professional_domain_manager()
        
        # 转换语言枚举
        try:
            language_enum = ProgrammingLanguage(request.language.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的编程语言: {request.language}。支持的语言: {', '.join([lang.value for lang in ProgrammingLanguage])}"
            )
        
        # 验证代码类型
        valid_code_types = ["function", "class", "algorithm", "test"]
        if request.code_type not in valid_code_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的代码类型: {request.code_type}。支持的代码类型: {', '.join(valid_code_types)}"
            )
        
        # 生成代码
        result = manager.programming_manager.generate_code(
            description=request.description,
            language=language_enum,
            code_type=request.code_type
        )
        
        return CodeGenerationResponse(
            success=True,
            timestamp=datetime.now(timezone.utc).isoformat(),
            generated_code=result["generated_code"],
            analysis_result=result["analysis_result"] if request.include_analysis else None,
            quality_score=result["quality_score"],
            message="代码生成完成"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"代码生成失败: {str(e)}"
        )


@router.post("/debug", response_model=CodeDebugResponse)
async def debug_code(
    request: CodeDebugRequest,
    db: Session = Depends(get_db),
    user = Depends(get_current_user),
):
    """调试代码"""
    if not PROFESSIONAL_CAPABILITIES_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="编程能力服务不可用，请检查依赖安装"
        )
    
    try:
        # 获取专业领域能力管理器
        manager = get_global_professional_domain_manager()
        
        # 转换语言枚举
        try:
            language_enum = ProgrammingLanguage(request.language.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的编程语言: {request.language}。支持的语言: {', '.join([lang.value for lang in ProgrammingLanguage])}"
            )
        
        # 调试代码
        result = manager.programming_manager.debug_code(
            code=request.code,
            language=language_enum,
            error_message=request.error_message
        )
        
        return CodeDebugResponse(
            success=True,
            timestamp=datetime.now(timezone.utc).isoformat(),
            original_code=result["original_code"],
            fixed_code=result["fixed_code"] if request.include_fixed_code else None,
            bugs_detected=result["bugs_detected"],
            suggested_fixes=result["suggested_fixes"],
            message="代码调试完成"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"代码调试失败: {str(e)}"
        )


@router.get("/languages", response_model=Dict[str, Any])
async def get_supported_languages(
    db: Session = Depends(get_db),
    user = Depends(get_current_user),
):
    """获取支持的编程语言"""
    if not PROFESSIONAL_CAPABILITIES_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="编程能力服务不可用，请检查依赖安装"
        )
    
    try:
        # 返回支持的编程语言列表
        languages = [lang.value for lang in ProgrammingLanguage]
        
        return {
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "languages": languages,
            "count": len(languages),
            "message": "支持的编程语言列表"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取支持的编程语言失败: {str(e)}"
        )


@router.get("/health", response_model=Dict[str, Any])
async def programming_health_check():
    """编程能力健康检查"""
    # 检查实际依赖项可用性
    try:
        import jedi
        jedi_available = True
    except ImportError:
        jedi_available = False
    
    try:
        import sympy
        sympy_available = True
    except ImportError:
        sympy_available = False
    
    try:
        import radon
        radon_available = True
    except ImportError:
        radon_available = False
    
    health_status = {
        "service": "Self AGI 编程能力API",
        "status": "healthy" if PROFESSIONAL_CAPABILITIES_AVAILABLE else "unavailable",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dependencies": {
            "professional_capabilities_available": PROFESSIONAL_CAPABILITIES_AVAILABLE,
            "ast_available": True,  # Python标准库
            "jedi_available": jedi_available,
            "sympy_available": sympy_available,
            "radon_available": radon_available,
        },
        "capabilities": {
            "code_analysis": PROFESSIONAL_CAPABILITIES_AVAILABLE,
            "code_generation": PROFESSIONAL_CAPABILITIES_AVAILABLE,
            "code_debugging": PROFESSIONAL_CAPABILITIES_AVAILABLE,
        }
    }
    
    return health_status