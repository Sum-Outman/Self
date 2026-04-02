#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self AGI 外部API集成框架
支持与外部服务（LLM API、图像识别、机器人控制等）的集成

功能：
1. 多API提供商支持（OpenAI、Anthropic、Google、Azure等）
2. 统一API调用接口
3. 认证和密钥管理
4. 速率限制和错误处理
5. 结果缓存和日志记录
6. 插件式架构，支持自定义API
"""

import sys
import os
import logging
import json
import time
import hashlib
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import requests
from urllib.parse import urlparse

# 导入异步库（可选）
try:
    import aiohttp  # type: ignore
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


class APIType(Enum):
    """API类型枚举"""
    LLM = "llm"  # 大语言模型
    VISION = "vision"  # 计算机视觉
    SPEECH = "speech"  # 语音识别/合成
    ROBOT_CONTROL = "robot_control"  # 机器人控制
    KNOWLEDGE_GRAPH = "knowledge_graph"  # 知识图谱
    DATABASE = "database"  # 数据库
    CUSTOM = "custom"  # 自定义API


class AuthMethod(Enum):
    """认证方法枚举"""
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    BASIC_AUTH = "basic_auth"
    BEARER_TOKEN = "bearer_token"
    NONE = "none"


@dataclass
class APIConfig:
    """API配置"""
    
    api_type: APIType
    provider: str
    base_url: str
    auth_method: AuthMethod
    credentials: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    max_retries: int = 3
    rate_limit_per_minute: int = 60
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIRequest:
    """API请求"""
    
    endpoint: str  # API端点路径
    url: Optional[str] = None  # 完整URL，如果提供则优先使用
    method: str = "POST"
    params: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: Optional[int] = None


@dataclass
class APIResponse:
    """API响应"""
    
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    headers: Dict[str, str] = field(default_factory=dict)
    request_time_ms: float = 0.0


class APIProvider(ABC):
    """API提供者基类"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.logger = logging.getLogger(f"APIProvider.{config.provider}")
        
        # 请求统计
        self.request_count = 0
        self.error_count = 0
        self.total_request_time = 0.0
        
        # 速率限制
        self.last_request_time = 0
        self.request_timestamps = []
        
        self.logger.info(f"初始化API提供者: {config.provider}")
    
    def prepare_headers(self) -> Dict[str, str]:
        """准备请求头（包含认证信息）"""
        headers = {
            "User-Agent": "Self-AGI/1.0",
            "Content-Type": "application/json"
        }
        
        # 添加认证头
        auth_method = self.config.auth_method
        credentials = self.config.credentials
        
        if auth_method == AuthMethod.API_KEY:
            api_key = credentials.get("api_key")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        
        elif auth_method == AuthMethod.BEARER_TOKEN:
            token = credentials.get("bearer_token")
            if token:
                headers["Authorization"] = f"Bearer {token}"
        
        elif auth_method == AuthMethod.BASIC_AUTH:
            username = credentials.get("username")
            password = credentials.get("password")
            if username and password:
                import base64
                auth_str = f"{username}:{password}"
                auth_bytes = auth_str.encode("utf-8")
                auth_b64 = base64.b64encode(auth_bytes).decode("utf-8")
                headers["Authorization"] = f"Basic {auth_b64}"
        
        return headers
    
    def check_rate_limit(self) -> bool:
        """检查速率限制"""
        if self.config.rate_limit_per_minute <= 0:
            return True
        
        current_time = time.time()
        
        # 清理超过1分钟的时间戳
        one_minute_ago = current_time - 60
        self.request_timestamps = [t for t in self.request_timestamps if t > one_minute_ago]
        
        # 检查是否超过限制
        if len(self.request_timestamps) >= self.config.rate_limit_per_minute:
            self.logger.warning(f"速率限制: {len(self.request_timestamps)}/{self.config.rate_limit_per_minute}")
            return False
        
        return True
    
    def wait_for_rate_limit(self, max_wait_seconds: int = 60):
        """等待速率限制"""
        if not self.check_rate_limit():
            current_time = time.time()
            oldest_timestamp = min(self.request_timestamps) if self.request_timestamps else current_time
            
            # 计算需要等待的时间
            wait_time = max(0, oldest_timestamp + 60 - current_time)
            
            if wait_time > max_wait_seconds:
                wait_time = max_wait_seconds
            
            if wait_time > 0:
                self.logger.info(f"等待速率限制: {wait_time:.2f}秒")
                time.sleep(wait_time)
    
    def make_request(self, request: APIRequest) -> APIResponse:
        """执行API请求
        
        默认实现：使用HTTP请求发送API调用
        子类可以重写此方法以实现特定API提供商的逻辑
        
        参数:
            request: API请求对象
            
        返回:
            API响应对象
        """
        import time as time_module
        
        start_time = time_module.time()
        success = False
        error_msg = None
        status_code = None
        response_data = None
        response_headers = {}
        
        try:
            # 准备请求参数
            headers = self.prepare_headers()
            headers.update(request.headers or {})
            
            # 构建请求URL
            url = request.url
            if not url:
                # 使用配置中的基础URL
                if self.config.base_url:
                    url = self.config.base_url.rstrip("/") + "/" + request.endpoint.lstrip("/")
                else:
                    raise ValueError("请求URL未指定且配置中无base_url")
            
            # 检查速率限制
            if not self.check_rate_limit():
                self.wait_for_rate_limit()
            
            # 记录请求开始
            self.logger.debug(f"发送API请求到: {url}, 方法: {request.method}, 超时: {request.timeout}s")
            
            # 执行HTTP请求
            request_kwargs = {
                "headers": headers,
                "timeout": request.timeout or self.config.default_timeout
            }
            
            if request.params:
                request_kwargs["params"] = request.params
            
            if request.data:
                request_kwargs["json"] = request.data
            
            if request.method.upper() == "GET":
                response = requests.get(url, **request_kwargs)
            elif request.method.upper() == "POST":
                response = requests.post(url, **request_kwargs)
            elif request.method.upper() == "PUT":
                response = requests.put(url, **request_kwargs)
            elif request.method.upper() == "DELETE":
                response = requests.delete(url, **request_kwargs)
            else:
                raise ValueError(f"不支持的HTTP方法: {request.method}")
            
            # 处理响应
            status_code = response.status_code
            response_headers = dict(response.headers)
            
            if response.headers.get("Content-Type", "").startswith("application/json"):
                response_data = response.json()
            else:
                response_data = response.text
            
            # 检查HTTP状态码
            if 200 <= status_code < 300:
                success = True
                self.logger.debug(f"API请求成功: {status_code}")
            else:
                error_msg = f"HTTP错误 {status_code}: {response_data}"
                self.logger.warning(f"API请求失败: {error_msg}")
        
        except requests.exceptions.Timeout as e:
            error_msg = f"请求超时: {e}"
            self.logger.error(error_msg)
        
        except requests.exceptions.ConnectionError as e:
            error_msg = f"连接错误: {e}"
            self.logger.error(error_msg)
        
        except requests.exceptions.RequestException as e:
            error_msg = f"请求异常: {e}"
            self.logger.error(error_msg)
        
        except Exception as e:
            error_msg = f"未知错误: {e}"
            self.logger.error(error_msg)
        
        finally:
            # 更新统计信息
            end_time = time_module.time()
            request_time_ms = (end_time - start_time) * 1000
            self.request_count += 1
            self.total_request_time += request_time_ms
            
            if not success:
                self.error_count += 1
            
            # 记录请求时间戳用于速率限制
            self.request_timestamps.append(start_time)
        
        return APIResponse(
            success=success,
            data=response_data,
            error=error_msg,
            status_code=status_code,
            headers=response_headers,
            request_time_ms=request_time_ms
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        avg_request_time = 0.0
        if self.request_count > 0:
            avg_request_time = self.total_request_time / self.request_count
        
        return {
            "provider": self.config.provider,
            "api_type": self.config.api_type.value,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "success_rate": 1.0 - (self.error_count / max(self.request_count, 1)),
            "avg_request_time_ms": avg_request_time,
            "enabled": self.config.enabled
        }


class RESTAPIProvider(APIProvider):
    """REST API提供者"""
    
    def make_request(self, request: APIRequest) -> APIResponse:
        """执行REST API请求"""
        start_time = time.time()
        
        # 检查速率限制
        if not self.check_rate_limit():
            self.wait_for_rate_limit()
        
        # 准备URL和参数
        url = f"{self.config.base_url.rstrip('/')}/{request.endpoint.lstrip('/')}"
        
        # 准备请求参数
        headers = self.prepare_headers()
        headers.update(request.headers)
        
        timeout = request.timeout or self.config.timeout
        
        # 记录请求
        self.request_count += 1
        self.request_timestamps.append(time.time())
        
        try:
            # 执行请求
            if request.method.upper() == "GET":
                response = requests.get(
                    url,
                    params=request.params,
                    headers=headers,
                    timeout=timeout
                )
            elif request.method.upper() == "POST":
                response = requests.post(
                    url,
                    params=request.params,
                    json=request.data,
                    headers=headers,
                    timeout=timeout
                )
            elif request.method.upper() == "PUT":
                response = requests.put(
                    url,
                    params=request.params,
                    json=request.data,
                    headers=headers,
                    timeout=timeout
                )
            elif request.method.upper() == "DELETE":
                response = requests.delete(
                    url,
                    params=request.params,
                    headers=headers,
                    timeout=timeout
                )
            else:
                raise ValueError(f"不支持的HTTP方法: {request.method}")
            
            # 计算请求时间
            request_time_ms = (time.time() - start_time) * 1000
            self.total_request_time += request_time_ms
            
            # 处理响应
            if response.status_code >= 200 and response.status_code < 300:
                # 成功
                try:
                    data = response.json()
                except ValueError:
                    data = response.text
                
                return APIResponse(
                    success=True,
                    data=data,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    request_time_ms=request_time_ms
                )
            else:
                # 错误
                self.error_count += 1
                
                error_msg = f"API请求失败: {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f", {error_data}"
                except ValueError:
                    error_msg += f", {response.text}"
                
                self.logger.error(error_msg)
                
                return APIResponse(
                    success=False,
                    error=error_msg,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    request_time_ms=request_time_ms
                )
        
        except requests.exceptions.Timeout:
            # 超时错误
            self.error_count += 1
            request_time_ms = (time.time() - start_time) * 1000
            
            error_msg = f"API请求超时: {timeout}秒"
            self.logger.error(error_msg)
            
            return APIResponse(
                success=False,
                error=error_msg,
                request_time_ms=request_time_ms
            )
        
        except requests.exceptions.ConnectionError:
            # 连接错误
            self.error_count += 1
            request_time_ms = (time.time() - start_time) * 1000
            
            error_msg = "API连接错误"
            self.logger.error(error_msg)
            
            return APIResponse(
                success=False,
                error=error_msg,
                request_time_ms=request_time_ms
            )
        
        except Exception as e:
            # 其他错误
            self.error_count += 1
            request_time_ms = (time.time() - start_time) * 1000
            
            error_msg = f"API请求异常: {e}"
            self.logger.error(error_msg)
            
            return APIResponse(
                success=False,
                error=error_msg,
                request_time_ms=request_time_ms
            )


class LLMAPIProvider(RESTAPIProvider):
    """LLM API提供者（大语言模型）"""
    
    def complete(self, 
                 prompt: str, 
                 max_tokens: int = 1000,
                 temperature: float = 0.7,
                 **kwargs) -> APIResponse:
        """完成文本生成"""
        
        # 准备请求数据
        data = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        # 根据提供商调整数据格式
        provider = self.config.provider.lower()
        
        if provider == "openai":
            data = {
                "model": self.config.metadata.get("model", "gpt-3.5-turbo"),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
            endpoint = "v1/chat/completions"
        
        elif provider == "anthropic":
            data = {
                "model": self.config.metadata.get("model", "claude-3-haiku-20240307"),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
                **kwargs
            }
            endpoint = "v1/messages"
        
        elif provider == "google":
            data = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": temperature,
                    **kwargs.get("generation_config", {})
                }
            }
            endpoint = f"v1/models/{self.config.metadata.get('model', 'gemini-pro')}:generateContent"
        
        else:
            # 通用格式
            endpoint = "v1/completions"
        
        request = APIRequest(
            endpoint=endpoint,
            method="POST",
            data=data
        )
        
        return self.make_request(request)
    
    def embed(self, texts: List[str], **kwargs) -> APIResponse:
        """生成文本嵌入"""
        
        # 准备请求数据
        data = {
            "input": texts,
            **kwargs
        }
        
        # 根据提供商调整
        provider = self.config.provider.lower()
        
        if provider == "openai":
            endpoint = "v1/embeddings"
            data["model"] = self.config.metadata.get("embedding_model", "text-embedding-3-small")
        
        elif provider == "google":
            endpoint = f"v1/models/{self.config.metadata.get('embedding_model', 'text-embedding-004')}:embedContent"
            # Google API需要不同的数据格式
            contents = []
            for text in texts:
                contents.append({
                    "parts": [{"text": text}]
                })
            data = {
                "requests": [
                    {
                        "model": self.config.metadata.get("embedding_model", "text-embedding-004"),
                        "content": content
                    } for content in contents
                ]
            }
        
        else:
            endpoint = "v1/embeddings"
        
        request = APIRequest(
            endpoint=endpoint,
            method="POST",
            data=data
        )
        
        return self.make_request(request)


class VisionAPIProvider(RESTAPIProvider):
    """计算机视觉API提供者"""
    
    def analyze_image(self, 
                     image_url: Optional[str] = None,
                     image_base64: Optional[str] = None,
                     features: List[str] = None,
                     **kwargs) -> APIResponse:
        """分析图像"""
        
        if not image_url and not image_base64:
            return APIResponse(
                success=False,
                error="必须提供image_url或image_base64"
            )
        
        features = features or ["objects", "faces", "labels"]
        
        # 准备请求数据
        data = {
            "features": features,
            **kwargs
        }
        
        if image_url:
            data["image"] = {"url": image_url}
        elif image_base64:
            data["image"] = {"base64": image_base64}
        
        # 根据提供商调整
        provider = self.config.provider.lower()
        
        if provider == "google":
            endpoint = "v1/images:annotate"
            # Google Vision API需要不同的数据格式
            requests = []
            for feature in features:
                requests.append({
                    "image": {"source": {"imageUri": image_url}} if image_url else {"content": image_base64},
                    "features": [{"type": feature.upper()}]
                })
            data = {"requests": requests}
        
        elif provider == "azure":
            endpoint = "vision/v3.2/analyze"
            params = {"visualFeatures": ",".join(features)}
            if image_url:
                data = {"url": image_url}
            else:
                # Azure需要二进制图像数据
                headers = {"Content-Type": "application/octet-stream"}
                request = APIRequest(
                    endpoint=endpoint,
                    method="POST",
                    params=params,
                    data=image_base64.encode() if image_base64 else b"",
                    headers=headers
                )
                return self.make_request(request)
        
        else:
            endpoint = "v1/vision/analyze"
        
        request = APIRequest(
            endpoint=endpoint,
            method="POST",
            data=data
        )
        
        return self.make_request(request)


class APIManager:
    """API管理器"""
    
    def __init__(self):
        self.providers: Dict[str, APIProvider] = {}
        self.provider_configs: Dict[str, APIConfig] = {}
        self.cache: Dict[str, Tuple[APIResponse, float]] = {}  # 缓存: key -> (response, timestamp)
        self.cache_ttl: int = 300  # 缓存TTL（秒）
        
        self.logger = logging.getLogger("APIManager")
        
        # 加载默认配置
        self._load_default_configs()
        
        self.logger.info("API管理器初始化完成")
    
    def _load_default_configs(self):
        """加载默认配置
        
        从环境变量加载API配置，遵循项目'禁止使用虚拟数据'要求。
        如果没有找到有效的API密钥，不创建任何虚拟配置。
        """
        self.logger.info("正在从环境变量加载API配置...")
        
        # 检查常见的API密钥环境变量
        api_configs = []
        
        # OpenAI配置
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if openai_api_key and openai_api_key.strip():
            self.logger.info("检测到OpenAI API密钥")
            openai_config = APIConfig(
                api_type=APIType.LLM,
                provider="openai",
                base_url="https://api.openai.com/v1",
                auth_method=AuthMethod.BEARER_TOKEN,
                credentials={"api_key": openai_api_key.strip()},
                timeout=60,
                max_retries=3,
                rate_limit_per_minute=60,
                enabled=True,
                metadata={"model": "gpt-4"}
            )
            api_configs.append(openai_config)
        
        # Anthropic配置
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_api_key and anthropic_api_key.strip():
            self.logger.info("检测到Anthropic API密钥")
            anthropic_config = APIConfig(
                api_type=APIType.LLM,
                provider="anthropic",
                base_url="https://api.anthropic.com/v1",
                auth_method=AuthMethod.BEARER_TOKEN,
                credentials={"api_key": anthropic_api_key.strip()},
                timeout=60,
                max_retries=3,
                rate_limit_per_minute=60,
                enabled=True,
                metadata={"model": "claude-3-opus-20240229"}
            )
            api_configs.append(anthropic_config)
        
        # Google AI配置
        google_ai_api_key = os.environ.get("GOOGLE_AI_API_KEY")
        if google_ai_api_key and google_ai_api_key.strip():
            self.logger.info("检测到Google AI API密钥")
            google_config = APIConfig(
                api_type=APIType.LLM,
                provider="google",
                base_url="https://generativelanguage.googleapis.com/v1",
                auth_method=AuthMethod.API_KEY,
                credentials={"api_key": google_ai_api_key.strip()},
                timeout=60,
                max_retries=3,
                rate_limit_per_minute=60,
                enabled=True,
                metadata={"model": "gemini-pro"}
            )
            api_configs.append(google_config)
        
        # Azure OpenAI配置
        azure_openai_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        azure_openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if azure_openai_api_key and azure_openai_api_key.strip() and azure_openai_endpoint:
            self.logger.info("检测到Azure OpenAI配置")
            azure_config = APIConfig(
                api_type=APIType.LLM,
                provider="azure_openai",
                base_url=azure_openai_endpoint.strip(),
                auth_method=AuthMethod.API_KEY,
                credentials={"api_key": azure_openai_api_key.strip()},
                timeout=60,
                max_retries=3,
                rate_limit_per_minute=60,
                enabled=True,
                metadata={"deployment": "gpt-4", "api_version": "2024-02-01"}
            )
            api_configs.append(azure_config)
        
        # 注册所有找到的配置
        registered_count = 0
        for config in api_configs:
            try:
                success = self.register_provider(config)
                if success:
                    registered_count += 1
                    self.logger.info(f"已注册API提供者: {config.provider} ({config.api_type.value})")
                else:
                    self.logger.warning(f"注册API提供者失败: {config.provider}")
            except Exception as e:
                self.logger.error(f"注册API提供者时出错 {config.provider}: {e}")
        
        if registered_count > 0:
            self.logger.info(f"成功加载并注册了 {registered_count} 个API配置")
        else:
            self.logger.warning("未找到有效的API配置。系统将以无外部API模式运行。")
            self.logger.info("要启用外部API功能，请设置相应的环境变量:")
            self.logger.info("  - OPENAI_API_KEY: OpenAI API密钥")
            self.logger.info("  - ANTHROPIC_API_KEY: Anthropic API密钥")
            self.logger.info("  - GOOGLE_AI_API_KEY: Google AI API密钥")
            self.logger.info("  - AZURE_OPENAI_API_KEY 和 AZURE_OPENAI_ENDPOINT: Azure OpenAI配置")
    
    def register_provider(self, config: APIConfig) -> bool:
        """注册API提供者"""
        try:
            provider_id = f"{config.provider}_{config.api_type.value}"
            
            # 创建提供者实例
            if config.api_type == APIType.LLM:
                provider = LLMAPIProvider(config)
            elif config.api_type == APIType.VISION:
                provider = VisionAPIProvider(config)
            else:
                provider = RESTAPIProvider(config)
            
            self.providers[provider_id] = provider
            self.provider_configs[provider_id] = config
            
            self.logger.info(f"注册API提供者: {provider_id}")
            return True
        
        except Exception as e:
            self.logger.error(f"注册API提供者失败: {e}")
            return False
    
    def get_provider(self, 
                    api_type: APIType, 
                    provider: Optional[str] = None) -> Optional[APIProvider]:
        """获取API提供者"""
        
        # 如果没有指定提供者，返回第一个匹配的
        if provider is None:
            for provider_id, p in self.providers.items():
                if p.config.api_type == api_type and p.config.enabled:
                    return p
            return None  # 返回None
        
        # 查找特定提供者
        provider_id = f"{provider}_{api_type.value}"
        return self.providers.get(provider_id)
    
    def make_request(self,
                    api_type: APIType,
                    endpoint: str,
                    method: str = "POST",
                    data: Optional[Dict[str, Any]] = None,
                    params: Optional[Dict[str, Any]] = None,
                    provider: Optional[str] = None,
                    use_cache: bool = False,
                    cache_key: Optional[str] = None) -> APIResponse:
        """执行API请求"""
        
        # 获取提供者
        api_provider = self.get_provider(api_type, provider)
        
        if api_provider is None:
            error_msg = f"没有可用的API提供者: {api_type.value}"
            if provider:
                error_msg += f", 提供者: {provider}"
            self.logger.error(error_msg)
            return APIResponse(success=False, error=error_msg)
        
        # 检查缓存
        if use_cache:
            if cache_key is None:
                # 生成缓存键
                cache_data = {
                    "provider": api_provider.config.provider,
                    "api_type": api_type.value,
                    "endpoint": endpoint,
                    "method": method,
                    "data": data,
                    "params": params
                }
                cache_key = hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()
            
            if cache_key in self.cache:
                cached_response, timestamp = self.cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    self.logger.debug(f"使用缓存响应: {cache_key}")
                    return cached_response
                else:
                    # 缓存过期
                    del self.cache[cache_key]
        
        # 执行请求
        request = APIRequest(
            endpoint=endpoint,
            method=method,
            data=data or {},
            params=params or {}
        )
        
        response = api_provider.make_request(request)
        
        # 缓存成功响应
        if use_cache and response.success:
            self.cache[cache_key] = (response, time.time())
        
        return response
    
    def llm_complete(self,
                    prompt: str,
                    provider: Optional[str] = None,
                    **kwargs) -> APIResponse:
        """LLM文本完成（完整接口）"""
        
        api_provider = self.get_provider(APIType.LLM, provider)
        
        if api_provider is None or not isinstance(api_provider, LLMAPIProvider):
            error_msg = f"没有可用的LLM API提供者"
            if provider:
                error_msg += f": {provider}"
            self.logger.error(error_msg)
            return APIResponse(success=False, error=error_msg)
        
        return api_provider.complete(prompt, **kwargs)
    
    def vision_analyze(self,
                      image_url: Optional[str] = None,
                      image_base64: Optional[str] = None,
                      provider: Optional[str] = None,
                      **kwargs) -> APIResponse:
        """视觉分析（完整接口）"""
        
        api_provider = self.get_provider(APIType.VISION, provider)
        
        if api_provider is None or not isinstance(api_provider, VisionAPIProvider):
            error_msg = f"没有可用的视觉API提供者"
            if provider:
                error_msg += f": {provider}"
            self.logger.error(error_msg)
            return APIResponse(success=False, error=error_msg)
        
        return api_provider.analyze_image(
            image_url=image_url,
            image_base64=image_base64,
            **kwargs
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取所有提供者的统计信息"""
        stats = {}
        
        for provider_id, provider in self.providers.items():
            stats[provider_id] = provider.get_stats()
        
        overall_stats = {
            "total_providers": len(self.providers),
            "enabled_providers": sum(1 for p in self.providers.values() if p.config.enabled),
            "total_requests": sum(s["request_count"] for s in stats.values()),
            "total_errors": sum(s["error_count"] for s in stats.values()),
            "cache_size": len(self.cache)
        }
        
        return {
            "overall": overall_stats,
            "providers": stats
        }
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        self.logger.info("API缓存已清空")


class ExternalAPILearningManager:
    """外部API学习管理器
    
    功能：
    1. 从外部API获取数据用于学习
    2. 转换API响应为训练数据
    3. 管理学习进度和状态
    4. 与训练系统集成
    """
    
    def __init__(self, api_manager: Optional[APIManager] = None):
        """初始化外部API学习管理器"""
        self.api_manager = api_manager or get_global_api_manager()
        self.logger = logging.getLogger("ExternalAPILearningManager")
        
        # 学习状态跟踪
        self.learning_sessions: Dict[str, Dict[str, Any]] = {}
        self.learning_data_cache: Dict[str, List[Any]] = {}
        self.active_learning_tasks: List[str] = []
        
        # 学习配置
        self.max_data_points_per_session = 1000
        self.min_confidence_threshold = 0.7
        self.data_validation_enabled = True
        
        self.logger.info("外部API学习管理器初始化完成")
    
    def create_learning_session(self, 
                               session_id: str,
                               learning_objective: str,
                               api_types: List[APIType],
                               data_sources: List[Dict[str, Any]]) -> bool:
        """创建学习会话
        
        参数:
            session_id: 会话ID
            learning_objective: 学习目标描述
            api_types: 使用的API类型列表
            data_sources: 数据源配置列表
            
        返回:
            是否成功创建
        """
        try:
            # 检查必要的API提供者
            available_providers = []
            for api_type in api_types:
                provider = self.api_manager.get_provider(api_type)
                if provider:
                    available_providers.append((api_type.value, provider.config.provider))
                else:
                    self.logger.warning(f"没有可用的API提供者: {api_type.value}")
            
            # 如果没有可用的API提供者，仍然创建会话但标记为受限模式
            operational_mode = "full" if available_providers else "limited"
            
            if not available_providers:
                self.logger.warning("没有可用的API提供者，会话将在受限模式下运行")
                self.logger.warning("注意：需要配置有效的API密钥才能进行实际的数据收集")
            
            # 创建学习会话
            session = {
                "session_id": session_id,
                "learning_objective": learning_objective,
                "api_types": [t.value for t in api_types],
                "data_sources": data_sources,
                "available_providers": available_providers,
                "operational_mode": operational_mode,
                "created_at": time.time(),
                "status": "created",
                "data_points_collected": 0,
                "last_update": time.time(),
                "errors": [],
                "progress": 0.0
            }
            
            self.learning_sessions[session_id] = session
            self.active_learning_tasks.append(session_id)
            
            self.logger.info(f"创建学习会话: {session_id}, 目标: {learning_objective}")
            self.logger.info(f"运行模式: {operational_mode}, 可用API提供者: {available_providers}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"创建学习会话失败: {e}")
            return False
    
    def collect_learning_data(self, 
                             session_id: str,
                             data_type: str,
                             query: Optional[str] = None,
                             count: int = 10) -> List[Dict[str, Any]]:
        """收集学习数据
        
        参数:
            session_id: 会话ID
            data_type: 数据类型 (text, image, audio, knowledge, etc.)
            query: 查询字符串（可选）
            count: 要收集的数据点数量
            
        返回:
            收集到的数据列表
        """
        if session_id not in self.learning_sessions:
            self.logger.error(f"学习会话不存在: {session_id}")
            return []  # 返回空列表
        
        session = self.learning_sessions[session_id]
        collected_data = []
        
        try:
            # 检查会话运行模式
            operational_mode = session.get("operational_mode", "full")
            
            if operational_mode == "limited":
                # 受限模式：没有可用的API提供者
                self.logger.warning(f"会话 {session_id} 在受限模式下运行，没有可用的API提供者")
                self.logger.warning("需要配置有效的API密钥才能进行实际的数据收集")
                # 根据项目要求"禁止使用虚拟数据"，不返回模拟数据
                return []  # 返回空列表
                
            else:
                # 完整模式：使用实际API
                # 根据数据类型选择API
                if data_type == "text":
                    data = self._collect_text_data(session_id, query, count)
                    collected_data.extend(data)
                    
                elif data_type == "image":
                    data = self._collect_image_data(session_id, query, count)
                    collected_data.extend(data)
                    
                elif data_type == "knowledge":
                    data = self._collect_knowledge_data(session_id, query, count)
                    collected_data.extend(data)
                    
                elif data_type == "code":
                    data = self._collect_code_data(session_id, query, count)
                    collected_data.extend(data)
                    
                else:
                    self.logger.warning(f"不支持的数据类型: {data_type}")
                    return []  # 返回空列表
            
            # 更新会话状态
            session["data_points_collected"] += len(collected_data)
            session["last_update"] = time.time()
            session["progress"] = min(1.0, session["data_points_collected"] / self.max_data_points_per_session)
            
            # 缓存数据
            cache_key = f"{session_id}_{data_type}"
            if cache_key not in self.learning_data_cache:
                self.learning_data_cache[cache_key] = []
            self.learning_data_cache[cache_key].extend(collected_data)
            
            self.logger.info(f"为会话 {session_id} 收集了 {len(collected_data)} 个{data_type}数据点")
            
            return collected_data
            
        except Exception as e:
            error_msg = f"收集学习数据失败: {e}"
            self.logger.error(error_msg)
            session["errors"].append({"time": time.time(), "error": error_msg})
            return []  # 返回空列表
    
    def _collect_text_data(self, session_id: str, query: Optional[str], count: int) -> List[Dict[str, Any]]:
        """收集文本数据"""
        session = self.learning_sessions[session_id]
        text_data = []
        
        # 获取LLM API提供者
        llm_provider = self.api_manager.get_provider(APIType.LLM)
        if not llm_provider or not isinstance(llm_provider, LLMAPIProvider):
            self.logger.warning("没有可用的LLM API提供者，无法收集文本数据")
            return text_data
        
        # 如果没有查询，使用学习目标作为基础
        if not query:
            query = f"请提供关于'{session['learning_objective']}'的学习资料和示例"
        
        try:
            # 调用LLM API获取文本数据
            prompt = f"""作为学习数据收集的一部分，请提供以下信息：
            
学习目标: {session['learning_objective']}
具体要求: {query}

请以结构化的格式提供{count}个相关的学习数据点，每个数据点包含：
1. 主题/标题
2. 详细描述
3. 关键概念
4. 相关示例
5. 学习要点
"""
            response = llm_provider.complete(
                prompt=prompt,
                max_tokens=4000,
                temperature=0.7
            )
            
            if response.success and response.data:
                # 解析响应数据
                text_content = response.data
                if isinstance(text_content, dict):
                    # 尝试提取文本内容
                    if "choices" in text_content and len(text_content["choices"]) > 0:
                        text_content = text_content["choices"][0].get("message", {}).get("content", "")
                    elif "text" in text_content:
                        text_content = text_content["text"]
                
                # 创建数据点
                data_point = {
                    "data_type": "text",
                    "content": str(text_content),
                    "source": f"llm_{llm_provider.config.provider}",
                    "query": query,
                    "collected_at": time.time(),
                    "confidence": 0.8,  # 默认置信度
                    "metadata": {
                        "provider": llm_provider.config.provider,
                        "model": llm_provider.config.metadata.get("model", "unknown"),
                        "response_length": len(str(text_content))
                    }
                }
                
                text_data.append(data_point)
                self.logger.info(f"从{llm_provider.config.provider}收集到文本数据，长度: {len(str(text_content))}")
            
            else:
                self.logger.warning(f"LLM API调用失败: {response.error}")
                
        except Exception as e:
            self.logger.error(f"收集文本数据异常: {e}")
        
        return text_data
    
    def _collect_image_data(self, session_id: str, query: Optional[str], count: int) -> List[Dict[str, Any]]:
        """收集图像数据"""
        session = self.learning_sessions[session_id]
        image_data = []
        
        # 获取视觉API提供者
        vision_provider = self.api_manager.get_provider(APIType.VISION)
        if not vision_provider or not isinstance(vision_provider, VisionAPIProvider):
            self.logger.warning("没有可用的视觉API提供者，无法收集图像数据")
            return image_data
        
        # 如果没有查询，使用学习目标作为基础
        if not query:
            query = f"关于'{session['learning_objective']}'的图像"
        
        try:
            # 尝试获取图像数据
            # 在实际实现中，这里可以：
            # 1. 调用图像搜索API（如Unsplash、Pexels、Google Images）
            # 2. 从本地图像库加载
            # 3. 使用用户上传的图像
            
            # 示例：尝试从公开API获取相关图像（需要实现实际的API集成）
            self.logger.info(f"尝试收集与'{query}'相关的图像数据")
            
            # 检查是否有配置的图像搜索API
            image_search_enabled = self.config.get("enable_image_search", False) if hasattr(self, 'config') else False
            image_search_api_key = self.config.get("image_search_api_key", "") if hasattr(self, 'config') else ""
            
            if image_search_enabled and image_search_api_key:
                # 如果有图像搜索API配置，可以调用实际API
                # 示例：调用Unsplash API
                # unsplash_url = f"https://api.unsplash.com/search/photos?query={query}&per_page={count}&client_id={image_search_api_key}"
                # response = requests.get(unsplash_url)
                # if response.status_code == 200:
                #     data = response.json()
                #     for photo in data.get('results', [])[:count]:
                #         image_url = photo.get('urls', {}).get('regular')
                #         if image_url:
                #             # 使用视觉API分析图像
                #             analysis_result = vision_provider.analyze_image(image_url=image_url)
                #             if analysis_result.success:
                #                 image_data.append({
                #                     "data_type": "image",
                #                     "content": analysis_result.data,
                #                     "source": image_url,
                #                     "query": query,
                #                     "collected_at": time.time(),
                #                     "confidence": 0.7,
                #                     "metadata": {
                #                         "provider": "unsplash",
                #                         "analysis_provider": vision_provider.config.provider
                #                     }
                #                 })
                self.logger.info("图像搜索API已配置但未实现，需要集成实际的图像搜索服务")
            else:
                # 没有图像搜索配置，记录信息性日志
                self.logger.info("图像数据收集需要配置图像搜索API或提供图像源")
                self.logger.info("建议：配置Unsplash、Pexels或Google Images API密钥以启用图像数据收集")
            
        except Exception as e:
            self.logger.error(f"收集图像数据异常: {e}")
        
        return image_data
    
    def _collect_knowledge_data(self, session_id: str, query: Optional[str], count: int) -> List[Dict[str, Any]]:
        """收集知识数据"""
        session = self.learning_sessions[session_id]
        knowledge_data = []
        
        # 获取LLM API提供者（如果没有知识图谱API，使用LLM）
        llm_provider = self.api_manager.get_provider(APIType.LLM)
        if not llm_provider or not isinstance(llm_provider, LLMAPIProvider):
            self.logger.warning("没有可用的LLM API提供者，无法收集知识数据")
            return knowledge_data
        
        # 如果没有查询，使用学习目标作为基础
        if not query:
            query = f"关于'{session['learning_objective']}'的知识点和概念"
        
        try:
            # 调用LLM API获取知识数据
            prompt = f"""作为知识学习数据收集的一部分，请提供以下信息：
            
学习目标: {session['learning_objective']}
具体要求: {query}

请以结构化的格式提供{count}个相关的知识数据点，每个数据点包含：
1. 概念/术语
2. 详细定义
3. 关键特性
4. 相关示例
5. 应用场景
"""
            response = llm_provider.complete(
                prompt=prompt,
                max_tokens=4000,
                temperature=0.7
            )
            
            if response.success and response.data:
                # 解析响应数据
                knowledge_content = response.data
                if isinstance(knowledge_content, dict):
                    # 尝试提取文本内容
                    if "choices" in knowledge_content and len(knowledge_content["choices"]) > 0:
                        knowledge_content = knowledge_content["choices"][0].get("message", {}).get("content", "")
                    elif "text" in knowledge_content:
                        knowledge_content = knowledge_content["text"]
                
                # 创建数据点
                data_point = {
                    "data_type": "knowledge",
                    "content": str(knowledge_content),
                    "source": f"llm_{llm_provider.config.provider}",
                    "query": query,
                    "collected_at": time.time(),
                    "confidence": 0.8,  # 默认置信度
                    "metadata": {
                        "provider": llm_provider.config.provider,
                        "model": llm_provider.config.metadata.get("model", "unknown"),
                        "response_length": len(str(knowledge_content))
                    }
                }
                
                knowledge_data.append(data_point)
                self.logger.info(f"从{llm_provider.config.provider}收集到知识数据，长度: {len(str(knowledge_content))}")
            
            else:
                self.logger.warning(f"LLM API调用失败: {response.error}")
                
        except Exception as e:
            self.logger.error(f"收集知识数据异常: {e}")
        
        return knowledge_data
    
    def _collect_code_data(self, session_id: str, query: Optional[str], count: int) -> List[Dict[str, Any]]:
        """收集代码数据"""
        session = self.learning_sessions[session_id]
        code_data = []
        
        # 获取LLM API提供者
        llm_provider = self.api_manager.get_provider(APIType.LLM)
        if not llm_provider or not isinstance(llm_provider, LLMAPIProvider):
            self.logger.warning("没有可用的LLM API提供者，无法收集代码数据")
            return code_data
        
        # 如果没有查询，使用学习目标作为基础
        if not query:
            query = f"关于'{session['learning_objective']}'的编程示例和代码片段"
        
        try:
            # 调用LLM API获取代码数据
            prompt = f"""作为编程学习数据收集的一部分，请提供以下信息：
            
学习目标: {session['learning_objective']}
具体要求: {query}

请提供{count}个相关的代码示例，每个示例包含：
1. 编程语言
2. 代码片段
3. 功能说明
4. 使用示例
5. 注意事项
"""
            response = llm_provider.complete(
                prompt=prompt,
                max_tokens=4000,
                temperature=0.7
            )
            
            if response.success and response.data:
                # 解析响应数据
                code_content = response.data
                if isinstance(code_content, dict):
                    # 尝试提取文本内容
                    if "choices" in code_content and len(code_content["choices"]) > 0:
                        code_content = code_content["choices"][0].get("message", {}).get("content", "")
                    elif "text" in code_content:
                        code_content = code_content["text"]
                
                # 创建数据点
                data_point = {
                    "data_type": "code",
                    "content": str(code_content),
                    "source": f"llm_{llm_provider.config.provider}",
                    "query": query,
                    "collected_at": time.time(),
                    "confidence": 0.8,  # 默认置信度
                    "metadata": {
                        "provider": llm_provider.config.provider,
                        "model": llm_provider.config.metadata.get("model", "unknown"),
                        "response_length": len(str(code_content))
                    }
                }
                
                code_data.append(data_point)
                self.logger.info(f"从{llm_provider.config.provider}收集到代码数据，长度: {len(str(code_content))}")
            
            else:
                self.logger.warning(f"LLM API调用失败: {response.error}")
                
        except Exception as e:
            self.logger.error(f"收集代码数据异常: {e}")
        
        return code_data
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取学习会话状态"""
        if session_id not in self.learning_sessions:
            return None  # 返回None
        
        session = self.learning_sessions[session_id].copy()
        
        # 计算缓存数据统计
        cache_stats = {}
        total_data_points = 0
        
        for cache_key in list(self.learning_data_cache.keys()):
            if cache_key.startswith(session_id + "_"):
                data_type = cache_key.replace(session_id + "_", "")
                data_list = self.learning_data_cache[cache_key]
                cache_stats[data_type] = len(data_list)
                total_data_points += len(data_list)
        
        session["cache_stats"] = cache_stats
        session["total_cached_data"] = total_data_points
        
        return session
    
    def get_learning_data(self, 
                         session_id: str,
                         data_type: Optional[str] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """获取学习数据
        
        参数:
            session_id: 会话ID
            data_type: 数据类型（可选）
            limit: 返回的数据点数量限制
            
        返回:
            学习数据列表
        """
        result_data = []
        
        if data_type:
            cache_key = f"{session_id}_{data_type}"
            if cache_key in self.learning_data_cache:
                result_data = self.learning_data_cache[cache_key][:limit]
        else:
            # 返回所有类型的数据
            for cache_key in list(self.learning_data_cache.keys()):
                if cache_key.startswith(session_id + "_"):
                    result_data.extend(self.learning_data_cache[cache_key][:limit])
        
        return result_data
    
    def prepare_training_data(self, 
                             session_id: str,
                             data_type: str,
                             format: str = "text") -> Dict[str, Any]:
        """准备训练数据
        
        参数:
            session_id: 会话ID
            data_type: 数据类型
            format: 输出格式 (text, json, dataset)
            
        返回:
            格式化的训练数据
        """
        cache_key = f"{session_id}_{data_type}"
        
        if cache_key not in self.learning_data_cache:
            return {"error": f"没有找到{data_type}类型的数据"}
        
        raw_data = self.learning_data_cache[cache_key]
        
        if format == "text":
            # 将数据转换为纯文本格式
            text_data = []
            for item in raw_data:
                text_item = f"数据类型: {item.get('data_type', 'unknown')}\n"
                text_item += f"来源: {item.get('source', 'unknown')}\n"
                text_item += f"内容: {item.get('content', '')}\n"
                text_item += f"收集时间: {time.ctime(item.get('collected_at', 0))}\n"
                text_item += "-" * 40
                text_data.append(text_item)
            
            return {
                "format": "text",
                "data_type": data_type,
                "item_count": len(text_data),
                "data": "\n\n".join(text_data)
            }
        
        elif format == "json":
            # 返回原始JSON数据
            return {
                "format": "json",
                "data_type": data_type,
                "item_count": len(raw_data),
                "data": raw_data
            }
        
        elif format == "dataset":
            # 转换为数据集格式（用于训练）
            dataset = []
            for item in raw_data:
                dataset_item = {
                    "text": str(item.get("content", "")),
                    "metadata": {
                        "source": item.get("source", ""),
                        "data_type": item.get("data_type", ""),
                        "confidence": item.get("confidence", 0.0),
                        "collected_at": item.get("collected_at", 0),
                        **item.get("metadata", {})
                    }
                }
                dataset.append(dataset_item)
            
            return {
                "format": "dataset",
                "data_type": data_type,
                "item_count": len(dataset),
                "data": dataset
            }
        
        else:
            return {"error": f"不支持的格式: {format}"}
    
    def end_learning_session(self, session_id: str, save_data: bool = True) -> bool:
        """结束学习会话
        
        参数:
            session_id: 会话ID
            save_data: 是否保存数据
            
        返回:
            是否成功结束
        """
        if session_id not in self.learning_sessions:
            self.logger.error(f"学习会话不存在: {session_id}")
            return False
        
        try:
            session = self.learning_sessions[session_id]
            session["status"] = "completed"
            session["completed_at"] = time.time()
            
            # 从活动任务中移除
            if session_id in self.active_learning_tasks:
                self.active_learning_tasks.remove(session_id)
            
            # 生成会话摘要
            total_data_points = 0
            data_types = set()
            
            for cache_key in list(self.learning_data_cache.keys()):
                if cache_key.startswith(session_id + "_"):
                    data_list = self.learning_data_cache[cache_key]
                    total_data_points += len(data_list)
                    data_type = cache_key.replace(session_id + "_", "")
                    data_types.add(data_type)
            
            session["summary"] = {
                "total_data_points": total_data_points,
                "data_types": list(data_types),
                "duration_seconds": time.time() - session["created_at"],
                "error_count": len(session["errors"])
            }
            
            self.logger.info(f"学习会话 {session_id} 已结束")
            self.logger.info(f"摘要: {session['summary']}")
            
            if not save_data:
                # 清理缓存数据
                cache_keys_to_remove = []
                for cache_key in list(self.learning_data_cache.keys()):
                    if cache_key.startswith(session_id + "_"):
                        cache_keys_to_remove.append(cache_key)
                
                for cache_key in cache_keys_to_remove:
                    del self.learning_data_cache[cache_key]
            
            return True
            
        except Exception as e:
            self.logger.error(f"结束学习会话失败: {e}")
            return False
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """获取所有学习会话"""
        sessions_list = []
        
        for session_id, session in self.learning_sessions.items():
            session_info = {
                "session_id": session_id,
                "learning_objective": session.get("learning_objective", ""),
                "status": session.get("status", "unknown"),
                "created_at": session.get("created_at", 0),
                "data_points_collected": session.get("data_points_collected", 0),
                "progress": session.get("progress", 0.0)
            }
            
            if "completed_at" in session:
                session_info["completed_at"] = session["completed_at"]
            
            sessions_list.append(session_info)
        
        return sessions_list
    
    def get_active_sessions(self) -> List[str]:
        """获取活动中的学习会话ID"""
        return self.active_learning_tasks.copy()
    
    def clear_old_sessions(self, max_age_hours: int = 24) -> int:
        """清理旧的学习会话
        
        参数:
            max_age_hours: 最大保留时间（小时）
            
        返回:
            清理的会话数量
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        sessions_to_remove = []
        
        for session_id, session in self.learning_sessions.items():
            created_at = session.get("created_at", 0)
            
            # 检查是否过期
            if current_time - created_at > max_age_seconds:
                # 检查是否已完成
                if session.get("status") == "completed":
                    sessions_to_remove.append(session_id)
        
        # 清理会话
        removed_count = 0
        for session_id in sessions_to_remove:
            # 清理缓存数据
            cache_keys_to_remove = []
            for cache_key in list(self.learning_data_cache.keys()):
                if cache_key.startswith(session_id + "_"):
                    cache_keys_to_remove.append(cache_key)
            
            for cache_key in cache_keys_to_remove:
                del self.learning_data_cache[cache_key]
            
            # 从活动任务中移除
            if session_id in self.active_learning_tasks:
                self.active_learning_tasks.remove(session_id)
            
            # 从会话字典中移除
            del self.learning_sessions[session_id]
            removed_count += 1
        
        if removed_count > 0:
            self.logger.info(f"清理了 {removed_count} 个旧的学习会话")
        
        return removed_count


# 全局API管理器实例
_global_api_manager = None

def get_global_api_manager() -> APIManager:
    """获取全局API管理器实例（单例模式）"""
    global _global_api_manager
    if _global_api_manager is None:
        _global_api_manager = APIManager()
    return _global_api_manager


# 全局外部API学习管理器实例
_global_external_api_learning_manager = None

def get_global_external_api_learning_manager() -> ExternalAPILearningManager:
    """获取全局外部API学习管理器实例（单例模式）"""
    global _global_external_api_learning_manager
    if _global_external_api_learning_manager is None:
        _global_external_api_learning_manager = ExternalAPILearningManager()
    return _global_external_api_learning_manager


if __name__ == "__main__":
    # 测试外部API集成框架
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== 测试外部API集成框架 ===")
    
    # 创建API管理器
    api_manager = get_global_api_manager()
    
    # 创建测试API配置
    test_llm_config = APIConfig(
        api_type=APIType.LLM,
        provider="openai",
        base_url="https://api.openai.com",
        auth_method=AuthMethod.API_KEY,
        credentials={"api_key": "test-key-123"},  # 测试用密钥
        timeout=30,
        max_retries=3,
        rate_limit_per_minute=60,
        metadata={"model": "gpt-3.5-turbo"}
    )
    
    test_vision_config = APIConfig(
        api_type=APIType.VISION,
        provider="google",
        base_url="https://vision.googleapis.com",
        auth_method=AuthMethod.API_KEY,
        credentials={"api_key": "test-key-456"},  # 测试用密钥
        timeout=30,
        max_retries=3,
        rate_limit_per_minute=60,
        metadata={"model": "vision-v1"}
    )
    
    # 注册提供者
    print("注册API提供者...")
    api_manager.register_provider(test_llm_config)
    api_manager.register_provider(test_vision_config)
    
    # 获取统计信息
    print("\nAPI统计信息:")
    stats = api_manager.get_stats()
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    # 测试通用请求
    print("\n测试通用API请求...")
    
    # 注意：由于测试配置使用无效密钥，实际请求会失败
    # 这里主要测试框架功能
    response = api_manager.make_request(
        api_type=APIType.LLM,
        endpoint="v1/models",
        method="GET",
        provider="openai",
        use_cache=True
    )
    
    print(f"API响应: 成功={response.success}")
    if response.success:
        print(f"数据: {response.data}")
    else:
        print(f"错误: {response.error}")
    
    # 测试完整接口
    print("\n测试完整接口...")
    
    llm_response = api_manager.llm_complete(
        prompt="你好，世界！",
        provider="openai",
        max_tokens=50
    )
    
    print(f"LLM响应: 成功={llm_response.success}")
    if llm_response.success:
        print(f"数据: {llm_response.data}")
    else:
        print(f"错误: {llm_response.error}")
    
    # 再次获取统计信息
    print("\n更新后的API统计信息:")
    stats = api_manager.get_stats()
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    print("\n外部API集成框架测试完成!")
    print("注意：测试使用了无效的API密钥，实际使用时需要配置有效的密钥")
    
    # 测试外部API学习管理器
    print("\n=== 测试外部API学习管理器 ===")
    
    # 获取学习管理器实例
    learning_manager = get_global_external_api_learning_manager()
    
    # 创建学习会话
    session_id = f"test_session_{int(time.time())}"
    learning_objective = "测试外部API学习功能"
    api_types = [APIType.LLM]
    data_sources = [
        {
            "type": "text",
            "description": "文本学习数据",
            "query_template": "关于{objective}的{count}个示例"
        }
    ]
    
    print(f"创建学习会话: {session_id}")
    created = learning_manager.create_learning_session(
        session_id=session_id,
        learning_objective=learning_objective,
        api_types=api_types,
        data_sources=data_sources
    )
    
    if created:
        print("✓ 学习会话创建成功")
        
        # 获取会话状态
        session_status = learning_manager.get_session_status(session_id)
        print(f"会话状态: {json.dumps(session_status, indent=2, ensure_ascii=False)}")
        
        # 收集文本数据（由于API密钥无效，实际不会调用API，但测试框架功能）
        print("\n测试数据收集功能...")
        collected_data = learning_manager.collect_learning_data(
            session_id=session_id,
            data_type="text",
            query="Python编程基础",
            count=3
        )
        
        print(f"收集到 {len(collected_data)} 个数据点")
        
        if collected_data:
            print("✓ 数据收集成功")
            
            # 获取学习数据
            learning_data = learning_manager.get_learning_data(
                session_id=session_id,
                data_type="text",
                limit=5
            )
            print(f"学习数据数量: {len(learning_data)}")
            
            # 准备训练数据
            training_data = learning_manager.prepare_training_data(
                session_id=session_id,
                data_type="text",
                format="json"
            )
            
            if "error" not in training_data:
                print(f"训练数据准备成功，包含 {training_data.get('item_count', 0)} 个项目")
            else:
                print(f"训练数据准备失败: {training_data.get('error')}")
        
        else:
            print("⚠ 数据收集失败（可能是由于API密钥无效）")
            print("注意：实际使用时需要配置有效的API密钥")
        
        # 结束学习会话
        ended = learning_manager.end_learning_session(session_id, save_data=True)
        if ended:
            print("✓ 学习会话结束成功")
        
        # 获取所有会话
        all_sessions = learning_manager.get_all_sessions()
        print(f"\n所有学习会话: {len(all_sessions)} 个")
        for session_info in all_sessions:
            print(f"  - {session_info['session_id']}: {session_info['learning_objective']} ({session_info['status']})")
        
        # 清理测试会话
        cleaned = learning_manager.clear_old_sessions(max_age_hours=0)  # 清理所有已完成会话
        print(f"清理了 {cleaned} 个旧会话")
        
    else:
        print("✗ 学习会话创建失败")
    
    print("\n外部API学习管理器测试完成!")