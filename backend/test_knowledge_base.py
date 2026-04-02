#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识库管理系统测试脚本
"""

import sys
import os
import logging
from typing import Dict

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.knowledge_base import KnowledgeManager, KnowledgeType  # noqa: E402

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_knowledge_manager_initialization():
    """测试知识管理器初始化"""
    logger.info("测试知识管理器初始化...")

    try:
        # 创建知识管理器
        km = KnowledgeManager()

        # 检查组件
        assert km.store is not None, "知识存储器未初始化"
        assert km.retriever is not None, "知识检索器未初始化"
        assert km.graph is not None, "知识图谱未初始化"
        assert km.validator is not None, "知识验证器未初始化"

        logger.info("✓ 知识管理器初始化测试通过")
        return km

    except Exception as e:
        logger.error(f"知识管理器初始化测试失败: {e}")
        raise


def test_add_knowledge(km: KnowledgeManager):
    """测试添加知识"""
    logger.info("测试添加知识...")

    # 测试事实知识
    fact_content = {
        "statement": "水的沸点是100摄氏度",
        "evidence": ["物理实验", "科学文献"],
        "confidence": 0.95,
    }

    result = km.add_knowledge(
        knowledge_type=KnowledgeType.FACT,
        content=fact_content,
        metadata={"domain": "physics", "language": "zh"},
    )

    assert result["success"], f"添加事实失败: {result.get('error')}"
    fact_id = result["id"]
    logger.info(f"✓ 添加事实成功: ID={fact_id}")

    # 测试规则知识
    rule_content = {
        "condition": "如果物体受到力的作用",
        "conclusion": "那么物体会产生加速度",
        "exceptions": ["在真空中", "当力为零时"],
    }

    result = km.add_knowledge(
        knowledge_type=KnowledgeType.RULE,
        content=rule_content,
        metadata={"domain": "physics", "law": "牛顿第二定律"},
    )

    assert result["success"], f"添加规则失败: {result.get('error')}"
    rule_id = result["id"]
    logger.info(f"✓ 添加规则成功: ID={rule_id}")

    # 测试概念知识
    concept_content = {
        "name": "人工智能",
        "definition": "人工智能是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统",
        "examples": ["机器学习", "自然语言处理", "计算机视觉"],
        "properties": {"type": "技术", "field": "计算机科学"},
    }

    result = km.add_knowledge(
        knowledge_type=KnowledgeType.CONCEPT,
        content=concept_content,
        metadata={"domain": "computer_science", "importance": "high"},
    )

    assert result["success"], f"添加概念失败: {result.get('error')}"
    concept_id = result["id"]
    logger.info(f"✓ 添加概念成功: ID={concept_id}")

    return {"fact_id": fact_id, "rule_id": rule_id, "concept_id": concept_id}


def test_query_knowledge(km: KnowledgeManager, knowledge_ids: Dict[str, str]):
    """测试查询知识"""
    logger.info("测试查询知识...")

    # 查询所有类型的知识
    results = km.query_knowledge("人工智能", limit=5)
    assert len(results) > 0, "查询无结果"
    logger.info(f"✓ 查询成功，找到 {len(results)} 条相关知识")

    # 按类型查询
    concept_results = km.query_knowledge(
        "定义", knowledge_type=KnowledgeType.CONCEPT, limit=3
    )
    logger.info(f"✓ 概念查询成功，找到 {len(concept_results)} 条概念知识")

    # 根据ID获取知识
    fact = km.get_knowledge_by_id(knowledge_ids["fact_id"])
    assert fact is not None, "根据ID获取知识失败"
    assert fact["type"] == "fact", f"知识类型错误: {fact['type']}"
    logger.info(f"✓ 根据ID获取知识成功: {fact['content']['statement']}")

    return results


def test_update_knowledge(km: KnowledgeManager, knowledge_ids: Dict[str, str]):
    """测试更新知识"""
    logger.info("测试更新知识...")

    # 更新事实知识
    updates = {
        "content": {"statement": "在标准大气压下，水的沸点是100摄氏度"},
        "metadata": {"precision": "high", "verified": True},
    }

    result = km.update_knowledge(knowledge_ids["fact_id"], updates)
    assert result["success"], f"更新知识失败: {result.get('error')}"

    # 验证更新
    updated_fact = km.get_knowledge_by_id(knowledge_ids["fact_id"])
    assert "标准大气压" in updated_fact["content"]["statement"], "更新未生效"
    logger.info(f"✓ 更新知识成功: {updated_fact['content']['statement']}")

    return result


def test_delete_knowledge(km: KnowledgeManager, knowledge_ids: Dict[str, str]):
    """测试删除知识"""
    logger.info("测试删除知识...")

    # 删除规则知识
    result = km.delete_knowledge(knowledge_ids["rule_id"])
    assert result["success"], f"删除知识失败: {result.get('error')}"

    # 验证删除
    deleted_rule = km.get_knowledge_by_id(knowledge_ids["rule_id"])
    assert deleted_rule is None, "知识删除未生效"
    logger.info(f"✓ 删除知识成功: ID={knowledge_ids['rule_id']}")

    return result


def test_knowledge_graph(km: KnowledgeManager, knowledge_ids: Dict[str, str]):
    """测试知识图谱"""
    logger.info("测试知识图谱...")

    if km.graph is None:
        logger.warning("知识图谱未启用，跳过测试")
        return None  # 返回None

    # 添加关系
    km.graph.add_edge(
        source_id=knowledge_ids["concept_id"],
        target_id=knowledge_ids["fact_id"],
        relation_type="related_to",
        weight=0.8,
        metadata={"relation": "concept_example"},
    )

    # 查询邻居
    neighbors = km.graph.get_neighbors(knowledge_ids["concept_id"])
    assert len(neighbors) > 0, "未找到邻居节点"
    logger.info(f"✓ 知识图谱邻居查询成功，找到 {len(neighbors)} 个邻居")

    # 获取图谱统计
    stats = km.graph.get_stats()
    assert stats["num_nodes"] > 0, "图谱节点数为0"
    logger.info(
        f"✓ 知识图谱统计: {stats['num_nodes']} 个节点, {stats['num_edges']} 条边"
    )

    return stats


def test_knowledge_validation(km: KnowledgeManager):
    """测试知识验证"""
    logger.info("测试知识验证...")

    if km.validator is None:
        logger.warning("知识验证器未启用，跳过测试")
        return None  # 返回None

    # 创建一个有问题的知识（缺少必需字段）
    problematic_content = {
        "name": "测试概念",
        # 缺少'definition'字段
    }

    # 验证知识
    validation_result = km.validator.validate(
        {
            "id": "test_id",
            "type": "concept",
            "content": problematic_content,
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }
    )

    assert not validation_result["valid"], "有问题的知识应该验证失败"
    assert len(validation_result["errors"]) > 0, "应该有验证错误"
    logger.info(f"✓ 知识验证成功，发现 {len(validation_result['errors'])} 个错误")

    # 测试修正建议
    corrections = km.validator.suggest_corrections(
        knowledge_item={"type": "concept", "content": problematic_content},
        validation_result=validation_result,
    )

    assert len(corrections) > 0, "应该有修正建议"
    logger.info(f"✓ 生成 {len(corrections)} 条修正建议")

    return validation_result


def test_knowledge_stats(km: KnowledgeManager):
    """测试知识统计"""
    logger.info("测试知识统计...")

    stats = km.get_stats()
    assert "total_knowledge" in stats, "统计信息缺少total_knowledge"
    assert "by_type" in stats, "统计信息缺少by_type"

    logger.info(f"✓ 知识统计: 总共 {stats['total_knowledge']} 条知识")
    for k_type, count in stats["by_type"].items():
        logger.info(f"  - {k_type}: {count} 条")

    return stats


def main():
    """主测试函数"""
    logger.info("开始知识库管理系统测试...")

    try:
        # 1. 测试知识管理器初始化
        km = test_knowledge_manager_initialization()

        # 2. 测试添加知识
        knowledge_ids = test_add_knowledge(km)

        # 3. 测试查询知识
        _ = test_query_knowledge(km, knowledge_ids)

        # 4. 测试更新知识
        _ = test_update_knowledge(km, knowledge_ids)

        # 5. 测试知识图谱
        _ = test_knowledge_graph(km, knowledge_ids)

        # 6. 测试知识验证
        _ = test_knowledge_validation(km)

        # 7. 测试知识统计
        _ = test_knowledge_stats(km)

        # 8. 测试删除知识
        _ = test_delete_knowledge(km, knowledge_ids)

        logger.info("✅ 所有知识库测试通过！")

        # 清理测试数据
        for knowledge_id in knowledge_ids.values():
            try:
                km.delete_knowledge(knowledge_id)
            except Exception:
                pass  # 已实现

        return True

    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
