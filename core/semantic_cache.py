# ================================================================
# semantic_cache.py：语义缓存层
# ================================================================
#
# 【核心职责】
#   用向量相似度判断用户问题是否被回答过，如果相似则直接返回缓存结果。
#   避免重复调用模型，降低成本、提升响应速度。
#
# 【设计原则】
#   1. 缓存命中率由阈值（threshold）控制，阈值越低命中率越高，但误判风险也越高。
#   2. 缓存存储使用 ChromaDB（与知识库共用基础设施，但不污染业务数据）。
#   3. 异步设计，不阻塞主请求。
#   4. 可插拔：用户可在 config.yaml 中开关缓存功能。
#
# 【为什么需要语义缓存】
#   高盛预测 Token 消耗将增长 24 倍（2026-2030）。
#   个人/企业场景中，用户经常问高度相似的问题（如“总结昨日会议”）。
#   语义缓存可将成本降低 80%，延迟降低 90%。
# ================================================================

import hashlib
import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import chromadb
import numpy as np

logger = logging.getLogger(__name__)


# ================================================================
# 数据结构
# ================================================================

@dataclass
class CacheEntry:
    """缓存条目"""
    prompt_hash: str
    prompt: str
    response: str
    embedding: List[float]
    created_at: datetime
    ttl_seconds: int = 86400  # 默认 24 小时

    def is_expired(self) -> bool:
        return datetime.now() - self.created_at > timedelta(seconds=self.ttl_seconds)


# ================================================================
# 语义缓存主类
# ================================================================

class SemanticCache:
    """
    语义缓存：基于向量相似度的缓存层。

    工作流程：
        1. 用户提问 → 生成向量
        2. 在缓存库中检索最相似的 1 条记录
        3. 如果相似度 > threshold → 命中缓存，直接返回
        4. 如果相似度 ≤ threshold → 未命中，调用模型，并将结果存入缓存

    配置项（config.yaml）：
        cache:
          enabled: true
          threshold: 0.92        # 相似度阈值，0.85-0.95 之间
          ttl_seconds: 86400     # 缓存有效期（秒）
          max_entries: 10000     # 最大缓存条数（LRU 策略）
    """

    def __init__(
        self,
        chroma_path: str = "./chroma_db",
        collection_name: str = "cache",
        threshold: float = 0.92,
        ttl_seconds: int = 86400,
        max_entries: int = 10000,
        embedder=None,  # 注入 Embedder 实例，避免循环依赖
    ):
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.embedder = embedder

        # 初始化 ChromaDB
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(f"语义缓存已初始化: threshold={threshold}, ttl={ttl_seconds}s")

    # ================================================================
    # 核心接口
    # ================================================================

    async def get(self, prompt: str) -> Optional[str]:
        """
        从缓存中检索。

        Args:
            prompt: 用户输入

        Returns:
            命中缓存返回缓存内容，否则返回 None
        """
        if not self.embedder:
            logger.warning("语义缓存未配置 embedder，跳过检索")
            return None

        # 生成查询向量
        query_embedding = self.embedder.embed(prompt)

        # 在缓存库中检索最相似的一条
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=1,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"缓存检索失败: {e}")
            return None

        # 检查是否有结果
        if not results["ids"] or not results["ids"][0]:
            return None

        # 检查相似度是否超过阈值
        distance = results["distances"][0][0]  # cosine distance，范围 [0, 2]
        similarity = 1 - distance  # 转为相似度 [0, 1]

        if similarity < self.threshold:
            logger.debug(f"缓存未命中: similarity={similarity:.3f} < threshold={self.threshold}")
            return None

        # 检查是否过期
        metadata = results["metadatas"][0][0]
        created_at = datetime.fromisoformat(metadata.get("created_at", "1970-01-01T00:00:00"))
        if datetime.now() - created_at > timedelta(seconds=self.ttl_seconds):
            logger.debug("缓存已过期")
            # 异步删除过期条目（可选）
            return None

        logger.info(f"✅ 缓存命中: similarity={similarity:.3f}")
        return results["documents"][0][0]

    async def set(self, prompt: str, response: str) -> None:
        """
        将结果存入缓存。

        Args:
            prompt: 用户输入
            response: 模型回答
        """
        if not self.embedder:
            return

        # 检查缓存数量是否超过上限
        if self.collection.count() >= self.max_entries:
            # 简单 LRU：删除最早的 10% 条目
            self._evict_oldest(int(self.max_entries * 0.1))

        # 生成向量
        embedding = self.embedder.embed(prompt)

        # 生成唯一 ID（基于 prompt 的哈希）
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

        # 元数据
        metadata = {
            "prompt_hash": prompt_hash,
            "created_at": datetime.now().isoformat(),
            "ttl_seconds": str(self.ttl_seconds),
        }

        try:
            self.collection.add(
                ids=[prompt_hash],
                embeddings=[embedding],
                documents=[response],
                metadatas=[metadata],
            )
            logger.debug(f"缓存已存储: {prompt_hash}")
        except Exception as e:
            logger.error(f"缓存存储失败: {e}")

    async def clear(self) -> None:
        """清空所有缓存"""
        try:
            self.client.delete_collection("cache")
            self.collection = self.client.get_or_create_collection(
                name="cache",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("缓存已清空")
        except Exception as e:
            logger.error(f"清空缓存失败: {e}")

    # ================================================================
    # 内部管理方法
    # ================================================================

    def _evict_oldest(self, count: int) -> None:
        """删除最早的 count 条缓存"""
        try:
            results = self.collection.get(include=["metadatas"])
            if not results["ids"]:
                return

            # 按创建时间排序
            ids_with_time = []
            for i, meta in enumerate(results["metadatas"]):
                created_at = meta.get("created_at", "1970-01-01T00:00:00")
                ids_with_time.append((created_at, results["ids"][i]))

            ids_with_time.sort(key=lambda x: x[0])
            to_delete = [item[1] for item in ids_with_time[:count]]

            if to_delete:
                self.collection.delete(ids=to_delete)
                logger.info(f"已淘汰 {len(to_delete)} 条缓存")
        except Exception as e:
            logger.error(f"淘汰缓存失败: {e}")

    def stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            "total_entries": self.collection.count(),
            "threshold": self.threshold,
            "ttl_seconds": self.ttl_seconds,
            "max_entries": self.max_entries,
        }


# ================================================================
# 工厂函数：根据配置创建缓存实例
# ================================================================

def get_cache(
    config: Dict[str, Any],
    embedder=None,
    chroma_path: str = "./chroma_db",
) -> Optional[SemanticCache]:
    """
    根据配置创建语义缓存实例。

    Args:
        config: config.yaml 中的 cache 配置块
        embedder: Embedder 实例（用于生成向量）
        chroma_path: ChromaDB 存储路径

    Returns:
        SemanticCache 实例，如果 enabled=False 则返回 None
    """
    if not config.get("enabled", False):
        logger.info("语义缓存已禁用（config.yaml 中 cache.enabled=false）")
        return None

    return SemanticCache(
        chroma_path=chroma_path,
        collection_name=config.get("collection_name", "cache"),
        threshold=config.get("threshold", 0.92),
        ttl_seconds=config.get("ttl_seconds", 86400),
        max_entries=config.get("max_entries", 10000),
        embedder=embedder,
    )


# ================================================================
# 快速测试入口
# ================================================================

if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    # 模拟 embedder（实际使用时注入真实 embedder）
    class MockEmbedder:
        def embed(self, text: str) -> List[float]:
            # 简单模拟：用文本长度生成伪向量（仅供测试）
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            return [float(b) / 255.0 for b in h[:10]]

    async def test():
        cache = SemanticCache(
            chroma_path="./chroma_db",
            collection_name="cache",
            threshold=0.85,
            ttl_seconds=3600,
            embedder=MockEmbedder(),
        )

        # 第一次：未命中
        result = await cache.get("什么是DSPy？")
        print(f"第一次查询: {result}")

        # 存入缓存
        await cache.set("什么是DSPy？", "DSPy是斯坦福大学开发的声明式编程框架。")

        # 第二次：命中
        result = await cache.get("什么是DSPy？")
        print(f"第二次查询: {result}")

        # 清空
        await cache.clear()
        print("缓存已清空")

    asyncio.run(test())