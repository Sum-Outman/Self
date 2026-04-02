"""知识版本控制模块

提供知识项的版本控制功能，包括：
1. 版本创建和存储
2. 版本历史查询
3. 版本回滚
4. 版本差异比较
"""

import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging


class KnowledgeVersion:
    """知识版本"""
    
    def __init__(
        self,
        knowledge_id: str,
        version_number: int,
        content: Dict[str, Any],
        created_at: datetime,
        created_by: str = "system",
        change_description: str = "",
        change_type: str = "update",  # create, update, delete, merge
        parent_version: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.knowledge_id = knowledge_id
        self.version_number = version_number
        self.content = content
        self.created_at = created_at
        self.created_by = created_by
        self.change_description = change_description
        self.change_type = change_type
        self.parent_version = parent_version
        self.metadata = metadata or {}
        
        # 计算内容哈希
        self.content_hash = self._calculate_hash()
    
    def _calculate_hash(self) -> str:
        """计算内容哈希值"""
        content_str = json.dumps(self.content, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content_str.encode('utf-8')).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "knowledge_id": self.knowledge_id,
            "version_number": self.version_number,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "change_description": self.change_description,
            "change_type": self.change_type,
            "parent_version": self.parent_version,
            "content_hash": self.content_hash,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeVersion':
        """从字典创建版本对象"""
        # 解析日期时间
        created_at = data["created_at"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        
        return cls(
            knowledge_id=data["knowledge_id"],
            version_number=data["version_number"],
            content=data["content"],
            created_at=created_at,
            created_by=data.get("created_by", "system"),
            change_description=data.get("change_description", ""),
            change_type=data.get("change_type", "update"),
            parent_version=data.get("parent_version"),
            metadata=data.get("metadata", {})
        )


class KnowledgeVersionControl:
    """知识版本控制器"""
    
    def __init__(self, storage_path: str, logger: Optional[logging.Logger] = None):
        self.storage_path = storage_path
        self.logger = logger or logging.getLogger(__name__)
        
        # 版本存储: {knowledge_id: [KnowledgeVersion, ...]}
        self._versions: Dict[str, List[KnowledgeVersion]] = {}
        
        # 加载现有版本
        self._load_versions()
    
    def _get_version_file_path(self) -> str:
        """获取版本存储文件路径"""
        import os
        return os.path.join(self.storage_path, "knowledge_versions.json")
    
    def _load_versions(self) -> None:
        """从文件加载版本数据"""
        import os
        
        version_file = self._get_version_file_path()
        if not os.path.exists(version_file):
            self.logger.info(f"版本文件不存在，创建新文件: {version_file}")
            return
        
        try:
            with open(version_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._versions = {}
            for knowledge_id, version_list in data.items():
                self._versions[knowledge_id] = [
                    KnowledgeVersion.from_dict(v) for v in version_list
                ]
            
            self.logger.info(f"加载版本数据: {sum(len(v) for v in self._versions.values())} 个版本")
        except Exception as e:
            self.logger.error(f"加载版本数据失败: {e}")
            self._versions = {}
    
    def _save_versions(self) -> None:
        """保存版本数据到文件"""
        import os
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self._get_version_file_path()), exist_ok=True)
            
            # 准备数据
            data = {}
            for knowledge_id, version_list in self._versions.items():
                data[knowledge_id] = [v.to_dict() for v in version_list]
            
            with open(self._get_version_file_path(), 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self.logger.debug(f"版本数据已保存")
        except Exception as e:
            self.logger.error(f"保存版本数据失败: {e}")
    
    def create_version(
        self,
        knowledge_id: str,
        content: Dict[str, Any],
        created_by: str = "system",
        change_description: str = "",
        change_type: str = "update"
    ) -> KnowledgeVersion:
        """创建新版本
        
        Args:
            knowledge_id: 知识项ID
            content: 知识内容
            created_by: 创建者
            change_description: 变更描述
            change_type: 变更类型 (create, update, delete, merge)
            
        Returns:
            创建的版本对象
        """
        # 获取当前最新版本号
        current_versions = self._versions.get(knowledge_id, [])
        current_version_number = current_versions[-1].version_number if current_versions else 0
        
        # 计算父版本（上一个版本）
        parent_version = current_version_number if current_versions else None
        
        # 创建新版本
        new_version = KnowledgeVersion(
            knowledge_id=knowledge_id,
            version_number=current_version_number + 1,
            content=content,
            created_at=datetime.now(),
            created_by=created_by,
            change_description=change_description,
            change_type=change_type,
            parent_version=parent_version
        )
        
        # 添加到版本列表
        if knowledge_id not in self._versions:
            self._versions[knowledge_id] = []
        
        self._versions[knowledge_id].append(new_version)
        
        # 保存到文件
        self._save_versions()
        
        self.logger.info(
            f"创建版本: {knowledge_id} v{new_version.version_number} "
            f"({change_type}) - {change_description[:50]}..."
        )
        
        return new_version
    
    def get_version_history(
        self,
        knowledge_id: str,
        limit: int = 20,
        offset: int = 0
    ) -> List[KnowledgeVersion]:
        """获取版本历史
        
        Args:
            knowledge_id: 知识项ID
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            版本历史列表（按版本号降序排列）
        """
        if knowledge_id not in self._versions:
            return []
        
        versions = self._versions[knowledge_id]
        # 按版本号降序排列
        sorted_versions = sorted(versions, key=lambda v: v.version_number, reverse=True)
        
        # 分页
        start_idx = offset
        end_idx = offset + limit
        return sorted_versions[start_idx:end_idx]
    
    def get_version(
        self,
        knowledge_id: str,
        version_number: int
    ) -> Optional[KnowledgeVersion]:
        """获取特定版本
        
        Args:
            knowledge_id: 知识项ID
            version_number: 版本号
            
        Returns:
            版本对象，如果不存在则返回None
        """
        if knowledge_id not in self._versions:
            return None
        
        for version in self._versions[knowledge_id]:
            if version.version_number == version_number:
                return version
        
        return None
    
    def get_latest_version(self, knowledge_id: str) -> Optional[KnowledgeVersion]:
        """获取最新版本
        
        Args:
            knowledge_id: 知识项ID
            
        Returns:
            最新版本对象，如果不存在则返回None
        """
        if knowledge_id not in self._versions:
            return None
        
        versions = self._versions[knowledge_id]
        return max(versions, key=lambda v: v.version_number)
    
    def revert_to_version(
        self,
        knowledge_id: str,
        version_number: int,
        created_by: str = "system",
        change_description: str = ""
    ) -> Optional[KnowledgeVersion]:
        """回滚到指定版本
        
        Args:
            knowledge_id: 知识项ID
            version_number: 要回滚到的版本号
            created_by: 创建者
            change_description: 变更描述
            
        Returns:
            新创建的版本对象，如果回滚失败则返回None
        """
        # 获取目标版本
        target_version = self.get_version(knowledge_id, version_number)
        if not target_version:
            self.logger.error(f"版本不存在: {knowledge_id} v{version_number}")
            return None
        
        # 获取当前版本
        current_version = self.get_latest_version(knowledge_id)
        if not current_version:
            self.logger.error(f"知识项不存在: {knowledge_id}")
            return None
        
        # 检查是否已经是当前版本
        if current_version.version_number == version_number:
            self.logger.warning(f"已经是最新版本: {knowledge_id} v{version_number}")
            return current_version
        
        # 创建回滚版本（复制目标版本的内容）
        revert_version = self.create_version(
            knowledge_id=knowledge_id,
            content=target_version.content.copy(),
            created_by=created_by,
            change_description=f"回滚到版本 {version_number}: {change_description}",
            change_type="revert"
        )
        
        self.logger.info(
            f"回滚完成: {knowledge_id} 从 v{current_version.version_number} "
            f"回滚到 v{version_number} (新版本: v{revert_version.version_number})"
        )
        
        return revert_version
    
    def compare_versions(
        self,
        knowledge_id: str,
        version_a: int,
        version_b: int
    ) -> Dict[str, Any]:
        """比较两个版本的差异
        
        Args:
            knowledge_id: 知识项ID
            version_a: 版本A
            version_b: 版本B
            
        Returns:
            差异信息
        """
        version_a_obj = self.get_version(knowledge_id, version_a)
        version_b_obj = self.get_version(knowledge_id, version_b)
        
        if not version_a_obj or not version_b_obj:
            return {"error": "版本不存在"}
        
        # 简单差异比较
        diff_result = {
            "knowledge_id": knowledge_id,
            "version_a": version_a,
            "version_b": version_b,
            "content_hash_same": version_a_obj.content_hash == version_b_obj.content_hash,
            "fields_changed": [],
            "change_summary": f"从版本 {version_a} 到版本 {version_b}"
        }
        
        # 比较内容字段
        content_a = version_a_obj.content
        content_b = version_b_obj.content
        
        # 收集所有字段
        all_fields = set(content_a.keys()) | set(content_b.keys())
        
        for field in all_fields:
            value_a = content_a.get(field)
            value_b = content_b.get(field)
            
            if value_a != value_b:
                diff_result["fields_changed"].append({
                    "field": field,
                    "value_a": value_a,
                    "value_b": value_b
                })
        
        return diff_result
    
    def delete_knowledge_versions(self, knowledge_id: str) -> bool:
        """删除知识项的所有版本
        
        Args:
            knowledge_id: 知识项ID
            
        Returns:
            是否成功删除
        """
        if knowledge_id in self._versions:
            del self._versions[knowledge_id]
            self._save_versions()
            self.logger.info(f"删除知识项版本: {knowledge_id}")
            return True
        
        return False
    
    def cleanup_old_versions(
        self,
        max_versions_per_knowledge: int = 50,
        max_total_versions: int = 1000
    ) -> Dict[str, int]:
        """清理旧版本
        
        Args:
            max_versions_per_knowledge: 每个知识项保留的最大版本数
            max_total_versions: 保留的最大总版本数
            
        Returns:
            清理统计信息
        """
        total_deleted = 0
        knowledge_deleted = {}
        
        # 清理每个知识项的旧版本
        for knowledge_id, versions in list(self._versions.items()):
            if len(versions) > max_versions_per_knowledge:
                # 按版本号排序，保留最新的
                sorted_versions = sorted(versions, key=lambda v: v.version_number)
                versions_to_keep = sorted_versions[-max_versions_per_knowledge:]
                versions_to_delete = len(versions) - len(versions_to_keep)
                
                self._versions[knowledge_id] = versions_to_keep
                total_deleted += versions_to_delete
                knowledge_deleted[knowledge_id] = versions_to_delete
        
        # 清理完成后保存
        if total_deleted > 0:
            self._save_versions()
            self.logger.info(f"清理了 {total_deleted} 个旧版本")
        
        return {
            "total_deleted": total_deleted,
            "knowledge_deleted": knowledge_deleted
        }


# 全局版本控制器实例
_version_controller: Optional[KnowledgeVersionControl] = None


def get_version_controller(
    storage_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> KnowledgeVersionControl:
    """获取版本控制器单例实例"""
    global _version_controller
    
    if _version_controller is None:
        if storage_path is None:
            # 默认存储路径
            import os
            storage_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "versions")
        
        _version_controller = KnowledgeVersionControl(storage_path, logger)
    
    return _version_controller