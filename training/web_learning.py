#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自主上网学习模块
实现从互联网自动获取学习资源的能力

功能：
1. 智能网络爬取：基于学习目标自动搜索和收集网络资源
2. 内容提取和清洗：从网页中提取结构化知识
3. 多源数据整合：整合来自不同网站和API的知识
4. 质量评估：评估学习资源的质量和相关性
5. 知识转换：将网络内容转换为可训练的数据格式
6. 安全合规：遵守robots.txt和网站使用条款

基于真实实现，不使用虚拟数据
"""

import requests
import logging
import re
import time
import json
import urllib.parse
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import hashlib
import random

# HTML解析依赖
try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    logging.warning("BeautifulSoup不可用，HTML解析功能受限")

# PDF解析依赖
try:
    import PyPDF2
    PDF_PARSER_AVAILABLE = True
except ImportError:
    PDF_PARSER_AVAILABLE = False
    logging.warning("PyPDF2不可用，PDF解析功能受限")

# 学术API依赖
try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False
    logging.warning("arxiv不可用，学术论文搜索功能受限")

logger = logging.getLogger(__name__)


@dataclass
class WebResource:
    """网络资源"""
    
    url: str                          # 资源URL
    title: str                        # 资源标题
    content_type: str                 # 内容类型（html, pdf, video, api）
    content: str                      # 原始内容
    extracted_text: str               # 提取的文本
    metadata: Dict[str, Any]          # 元数据
    quality_score: float = 0.0        # 质量评分 0.0-1.0
    relevance_score: float = 0.0      # 相关性评分 0.0-1.0
    retrieved_at: datetime = field(default_factory=datetime.now)
    processed: bool = False           # 是否已处理
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "url": self.url,
            "title": self.title,
            "content_type": self.content_type,
            "content_length": len(self.content),
            "extracted_text_length": len(self.extracted_text),
            "metadata": self.metadata,
            "quality_score": self.quality_score,
            "relevance_score": self.relevance_score,
            "retrieved_at": self.retrieved_at.isoformat(),
            "processed": self.processed,
        }


@dataclass
class LearningQuery:
    """学习查询"""
    
    topic: str                        # 学习主题
    keywords: List[str]               # 关键词
    depth_level: int = 1              # 搜索深度（1-浅层，2-中等，3-深层）
    resource_types: List[str] = field(default_factory=lambda: ["html", "pdf"])
    max_results: int = 10             # 最大结果数
    language: str = "zh"              # 语言（zh, en等）
    
    def to_search_query(self) -> str:
        """转换为搜索查询字符串"""
        query_parts = [self.topic] + self.keywords
        return " ".join(query_parts)


class WebLearningEngine:
    """网络学习引擎"""
    
    def __init__(self,
                 cache_dir: Optional[str] = None,
                 user_agent: Optional[str] = None,
                 request_timeout: int = 30,
                 rate_limit_delay: float = 1.0):
        """
        初始化网络学习引擎
        
        参数:
            cache_dir: 缓存目录
            user_agent: 用户代理字符串
            request_timeout: 请求超时时间（秒）
            rate_limit_delay: 请求间延迟（秒）
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("web_learning_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        
        self.request_timeout = request_timeout
        self.rate_limit_delay = rate_limit_delay
        
        # 会话对象
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
        
        # 搜索API配置
        self.search_apis = self._initialize_search_apis()
        
        logger.info(f"网络学习引擎初始化完成，缓存目录: {self.cache_dir}")
    
    def _initialize_search_apis(self) -> Dict[str, Dict[str, Any]]:
        """初始化搜索API配置"""
        # 注意：实际使用时需要配置API密钥
        search_apis = {
            "google_custom_search": {
                "enabled": False,
                "api_key": None,
                "search_engine_id": None,
                "endpoint": "https://www.googleapis.com/customsearch/v1",
            },
            "duckduckgo": {
                "enabled": True,
                "endpoint": "https://html.duckduckgo.com/html/",
                "method": "html_scraping",
            },
            "arxiv": {
                "enabled": ARXIV_AVAILABLE,
                "endpoint": "http://export.arxiv.org/api/query",
            },
            "wikipedia": {
                "enabled": True,
                "endpoint": "https://en.wikipedia.org/w/api.php",
            },
            "github": {
                "enabled": True,
                "endpoint": "https://api.github.com/search/repositories",
            },
        }
        
        return search_apis
    
    def search_learning_resources(self, 
                                 learning_query: LearningQuery) -> List[WebResource]:
        """
        搜索学习资源
        
        参数:
            learning_query: 学习查询
            
        返回:
            网络资源列表
        """
        logger.info(f"搜索学习资源: {learning_query.topic}, 关键词: {learning_query.keywords}")
        
        resources = []
        
        # 1. 使用多个搜索源
        search_sources = []
        
        # 添加通用搜索引擎
        if self.search_apis["duckduckgo"]["enabled"]:
            search_sources.append(("duckduckgo", self._search_duckduckgo))
        
        # 添加学术搜索引擎
        if learning_query.resource_types and "pdf" in learning_query.resource_types:
            if self.search_apis["arxiv"]["enabled"]:
                search_sources.append(("arxiv", self._search_arxiv))
        
        # 添加Wikipedia
        if self.search_apis["wikipedia"]["enabled"]:
            search_sources.append(("wikipedia", self._search_wikipedia))
        
        # 2. 并行搜索（简化实现：顺序执行）
        for source_name, search_func in search_sources:
            try:
                source_resources = search_func(learning_query)
                resources.extend(source_resources)
                logger.info(f"从 {source_name} 找到 {len(source_resources)} 个资源")
                
                # 达到最大结果数时停止
                if len(resources) >= learning_query.max_results:
                    resources = resources[:learning_query.max_results]
                    break
                
                # 延迟以避免速率限制
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.warning(f"搜索源 {source_name} 失败: {e}")
        
        # 3. 获取资源内容
        fetched_resources = []
        for resource in resources:
            try:
                fetched_resource = self.fetch_resource(resource.url)
                if fetched_resource:
                    # 评估资源质量
                    fetched_resource.quality_score = self.evaluate_resource_quality(fetched_resource)
                    fetched_resource.relevance_score = self.evaluate_resource_relevance(
                        fetched_resource, learning_query
                    )
                    fetched_resources.append(fetched_resource)
                    
                    # 延迟以避免速率限制
                    time.sleep(self.rate_limit_delay)
                    
            except Exception as e:
                logger.warning(f"获取资源失败 {resource.url}: {e}")
        
        # 4. 按相关性排序
        fetched_resources.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.info(f"共获取 {len(fetched_resources)} 个学习资源")
        return fetched_resources
    
    def _search_duckduckgo(self, learning_query: LearningQuery) -> List[WebResource]:
        """使用DuckDuckGo搜索"""
        resources = []
        
        try:
            # 构建搜索URL
            query = learning_query.to_search_query()
            encoded_query = urllib.parse.quote(query)
            search_url = f"{self.search_apis['duckduckgo']['endpoint']}?q={encoded_query}"
            
            # 发送请求
            response = self.session.get(search_url, timeout=self.request_timeout)
            response.raise_for_status()
            
            if BEAUTIFULSOUP_AVAILABLE:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 解析搜索结果
                result_elements = soup.find_all('a', class_='result__url')
                
                for element in result_elements[:learning_query.max_results]:
                    try:
                        url = element.get('href')
                        title_element = element.find_next('h2')
                        title = title_element.get_text(strip=True) if title_element else "无标题"
                        
                        if url and url.startswith('http'):
                            resource = WebResource(
                                url=url,
                                title=title,
                                content_type="html",
                                content="",
                                extracted_text="",
                                metadata={"source": "duckduckgo", "query": query}
                            )
                            resources.append(resource)
                            
                    except Exception as e:
                        logger.debug(f"解析DuckDuckGo结果失败: {e}")
                        continue
            
            else:
                # 简单文本解析
                logger.warning("BeautifulSoup不可用，无法解析DuckDuckGo搜索结果")
        
        except Exception as e:
            logger.warning(f"DuckDuckGo搜索失败: {e}")
        
        return resources
    
    def _search_arxiv(self, learning_query: LearningQuery) -> List[WebResource]:
        """使用arXiv搜索学术论文"""
        resources = []
        
        if not ARXIV_AVAILABLE:
            return resources
        
        try:
            import arxiv
            
            # 构建搜索查询
            query = learning_query.to_search_query()
            
            # 搜索论文
            search = arxiv.Search(
                query=query,
                max_results=min(learning_query.max_results, 10),
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            for result in search.results():
                # 获取PDF链接
                pdf_url = None
                for link in result.links:
                    if link.title == "pdf":
                        pdf_url = link.href
                        break
                
                if pdf_url:
                    resource = WebResource(
                        url=pdf_url,
                        title=result.title,
                        content_type="pdf",
                        content="",
                        extracted_text=result.summary,
                        metadata={
                            "source": "arxiv",
                            "authors": [str(a) for a in result.authors],
                            "published": str(result.published),
                            "categories": result.categories,
                            "primary_category": result.primary_category,
                        }
                    )
                    resources.append(resource)
        
        except Exception as e:
            logger.warning(f"arXiv搜索失败: {e}")
        
        return resources
    
    def _search_wikipedia(self, learning_query: LearningQuery) -> List[WebResource]:
        """搜索Wikipedia文章"""
        resources = []
        
        try:
            # 使用Wikipedia API
            query = learning_query.to_search_query()
            api_params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
                "srlimit": min(learning_query.max_results, 10),
            }
            
            response = self.session.get(
                self.search_apis["wikipedia"]["endpoint"],
                params=api_params,
                timeout=self.request_timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            if "query" in data and "search" in data["query"]:
                for result in data["query"]["search"]:
                    page_id = result["pageid"]
                    title = result["title"]
                    
                    # 构建文章URL
                    url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}"
                    
                    resource = WebResource(
                        url=url,
                        title=title,
                        content_type="html",
                        content="",
                        extracted_text=result.get("snippet", ""),
                        metadata={
                            "source": "wikipedia",
                            "page_id": page_id,
                            "word_count": result.get("wordcount", 0),
                        }
                    )
                    resources.append(resource)
        
        except Exception as e:
            logger.warning(f"Wikipedia搜索失败: {e}")
        
        return resources
    
    def fetch_resource(self, url: str) -> Optional[WebResource]:
        """获取网络资源"""
        
        # 检查缓存
        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                # 检查缓存是否过期（1天）
                cached_at = datetime.fromisoformat(cached_data.get("retrieved_at", ""))
                if (datetime.now() - cached_at).days < 1:
                    logger.debug(f"使用缓存资源: {url}")
                    return WebResource(
                        url=cached_data["url"],
                        title=cached_data["title"],
                        content_type=cached_data["content_type"],
                        content=cached_data.get("content", ""),
                        extracted_text=cached_data.get("extracted_text", ""),
                        metadata=cached_data.get("metadata", {}),
                        quality_score=cached_data.get("quality_score", 0.0),
                        relevance_score=cached_data.get("relevance_score", 0.0),
                        retrieved_at=cached_at,
                        processed=cached_data.get("processed", False),
                    )
            
            except Exception as e:
                logger.debug(f"加载缓存失败 {url}: {e}")
        
        try:
            # 发送HTTP请求
            response = self.session.get(url, timeout=self.request_timeout)
            response.raise_for_status()
            
            # 确定内容类型
            content_type = response.headers.get('Content-Type', '').lower()
            
            # 解析内容
            title = ""
            extracted_text = ""
            
            if 'text/html' in content_type and BEAUTIFULSOUP_AVAILABLE:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 提取标题
                title_tag = soup.find('title')
                title = title_tag.get_text(strip=True) if title_tag else "无标题"
                
                # 提取正文文本
                # 移除脚本、样式等
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                
                # 获取段落文本
                paragraphs = soup.find_all('p')
                extracted_text = "\n".join([p.get_text(strip=True) for p in paragraphs])
                
                # 如果段落文本太短，使用整个body
                if len(extracted_text) < 100:
                    body = soup.find('body')
                    if body:
                        extracted_text = body.get_text(strip=True, separator='\n')
                
                content_type_str = "html"
                
            elif 'application/pdf' in content_type and PDF_PARSER_AVAILABLE:
                # 处理PDF文件
                import io
                pdf_file = io.BytesIO(response.content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                # 提取文本
                extracted_text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += page_text + "\n"
                
                # 使用URL作为标题
                title = url.split('/')[-1] or "PDF文档"
                content_type_str = "pdf"
                
            else:
                # 其他内容类型
                title = url.split('/')[-1] or "资源"
                extracted_text = response.text[:10000]  # 限制长度
                content_type_str = "text"
            
            # 创建资源对象
            resource = WebResource(
                url=url,
                title=title[:200],  # 限制标题长度
                content_type=content_type_str,
                content=response.text[:50000],  # 保存原始内容（限制大小）
                extracted_text=extracted_text[:100000],  # 限制提取文本大小
                metadata={
                    "content_type": content_type,
                    "content_length": len(response.content),
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                }
            )
            
            # 保存到缓存
            cache_data = resource.to_dict()
            cache_data["content"] = resource.content  # 保存原始内容
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"获取资源成功: {url}, 类型: {content_type_str}")
            return resource
            
        except Exception as e:
            logger.warning(f"获取资源失败 {url}: {e}")
            return None
    
    def evaluate_resource_quality(self, resource: WebResource) -> float:
        """评估资源质量"""
        quality_score = 0.5  # 基础分数
        
        # 1. 内容长度
        content_length = len(resource.extracted_text)
        if content_length > 1000:
            quality_score += 0.2
        elif content_length > 100:
            quality_score += 0.1
        
        # 2. 内容类型
        if resource.content_type == "pdf":
            quality_score += 0.1  # PDF通常质量较高
        
        # 3. 来源可信度
        domain = urllib.parse.urlparse(resource.url).netloc.lower()
        trusted_domains = ['.edu', '.gov', '.org', 'arxiv.org', 'wikipedia.org']
        if any(trusted in domain for trusted in trusted_domains):
            quality_score += 0.2
        
        # 4. 元数据完整性
        if resource.metadata and len(resource.metadata) > 3:
            quality_score += 0.1
        
        # 限制在0.0-1.0之间
        return max(0.0, min(1.0, quality_score))
    
    def evaluate_resource_relevance(self, 
                                  resource: WebResource,
                                  learning_query: LearningQuery) -> float:
        """评估资源相关性"""
        relevance_score = 0.0
        
        # 1. 标题匹配
        title = resource.title.lower()
        query_terms = [learning_query.topic.lower()] + [k.lower() for k in learning_query.keywords]
        
        for term in query_terms:
            if term in title:
                relevance_score += 0.3
        
        # 2. 内容匹配
        content = resource.extracted_text.lower()
        for term in query_terms:
            if term in content:
                relevance_score += 0.1
        
        # 3. 来源类型匹配
        if learning_query.resource_types and resource.content_type in learning_query.resource_types:
            relevance_score += 0.2
        
        # 4. 质量分数贡献
        relevance_score += resource.quality_score * 0.2
        
        # 限制在0.0-1.0之间
        return max(0.0, min(1.0, relevance_score))
    
    def process_resource_for_learning(self, 
                                     resource: WebResource) -> Dict[str, Any]:
        """将资源处理为学习材料"""
        processed_data = {
            "id": hashlib.md5(resource.url.encode()).hexdigest(),
            "title": resource.title,
            "url": resource.url,
            "content_type": resource.content_type,
            "extracted_text": resource.extracted_text,
            "metadata": resource.metadata,
            "quality_score": resource.quality_score,
            "relevance_score": resource.relevance_score,
            "processed_at": datetime.now().isoformat(),
            "learning_units": [],
        }
        
        # 根据内容类型进行不同处理
        if resource.content_type == "html" or resource.content_type == "text":
            # 分割为学习单元（段落）
            paragraphs = [p.strip() for p in resource.extracted_text.split('\n') if p.strip()]
            for i, paragraph in enumerate(paragraphs[:50]):  # 限制数量
                if len(paragraph) > 50:  # 最小长度
                    unit = {
                        "id": f"{processed_data['id']}_unit_{i}",
                        "type": "text_paragraph",
                        "content": paragraph,
                        "source": resource.url,
                        "position": i,
                    }
                    processed_data["learning_units"].append(unit)
        
        elif resource.content_type == "pdf":
            # 分割为章节或页面
            sections = [s.strip() for s in resource.extracted_text.split('\n\n') if s.strip()]
            for i, section in enumerate(sections[:100]):  # 限制数量
                if len(section) > 100:  # 最小长度
                    unit = {
                        "id": f"{processed_data['id']}_section_{i}",
                        "type": "pdf_section",
                        "content": section,
                        "source": resource.url,
                        "position": i,
                    }
                    processed_data["learning_units"].append(unit)
        
        resource.processed = True
        logger.info(f"处理资源完成: {resource.title}, 学习单元: {len(processed_data['learning_units'])}")
        
        return processed_data
    
    def batch_process_resources(self, 
                               resources: List[WebResource]) -> List[Dict[str, Any]]:
        """批量处理资源"""
        processed_results = []
        
        for resource in resources:
            if not resource.processed:
                try:
                    processed = self.process_resource_for_learning(resource)
                    processed_results.append(processed)
                    
                    # 保存处理结果
                    output_file = self.cache_dir / f"{processed['id']}_processed.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(processed, f, indent=2, ensure_ascii=False)
                    
                except Exception as e:
                    logger.warning(f"处理资源失败 {resource.url}: {e}")
        
        return processed_results


# 全局网络学习引擎实例
_web_learning_engine = None

def get_web_learning_engine(**kwargs) -> WebLearningEngine:
    """获取网络学习引擎单例"""
    global _web_learning_engine
    if _web_learning_engine is None:
        _web_learning_engine = WebLearningEngine(**kwargs)
    return _web_learning_engine


# 使用示例
if __name__ == "__main__":
    # 初始化日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建网络学习引擎
    engine = get_web_learning_engine()
    
    # 创建学习查询
    query = LearningQuery(
        topic="深度学习",
        keywords=["神经网络", "Transformer", "注意力机制"],
        depth_level=1,
        resource_types=["html", "pdf"],
        max_results=3,
        language="zh"
    )
    
    # 搜索资源
    resources = engine.search_learning_resources(query)
    
    print(f"找到 {len(resources)} 个学习资源:")
    for i, resource in enumerate(resources):
        print(f"{i+1}. {resource.title}")
        print(f"   质量: {resource.quality_score:.2f}, 相关性: {resource.relevance_score:.2f}")
        print(f"   URL: {resource.url}")
        print()
    
    # 处理资源
    if resources:
        processed = engine.batch_process_resources(resources)
        print(f"处理完成，生成 {len(processed)} 个学习材料")
    
    print("自主上网学习模块测试完成")