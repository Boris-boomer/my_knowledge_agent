# ================================================================
# vector_store.py：向量存储（封装 ChromaDB）
# ================================================================
#
# 【核心职责】
#   封装 ChromaDB 的增删改查操作，让业务层（build_kb、agent）不直接依赖 ChromaDB API。
#
# 【为什么需要这一层】
#   1. 隔离外部依赖：如果以后从 ChromaDB 换成 FAISS / PGVector，只需改这个文件。
#   2. 统一错误处理：所有向量操作都经过这里，异常被捕获并转换为业务层可理解的错误。
#   3. 简化调用：业务层只需要 `add()` 和 `query()`，不需要关心 ChromaDB 的细节。
#
# 【对外暴露】
#   VectorStore            - 主类
#   get_vector_store()     - 工厂函数，从配置创建实例
#
# 【使用示例】
#   from vector_store import VectorStore
#   store = VectorStore(path="./chroma_db", collection_name="docs")
#   store.add(ids=["doc_1"], embeddings=[emb], documents=["内容"], metadatas=[{"source": "test.txt"}])
#   results = store.query(query_embedding=emb, top_k=3)
# ================================================================

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import chromadb

logger = logging.getLogger(__name__)


# ================================================================
# 向量存储主类
# ================================================================

class VectorStore:
    """
    向量存储：封装 ChromaDB 的增删改查操作。

    设计原则：
        1. 只暴露业务需要的方法（add / query / delete / count）
        2. 所有方法都有明确的错误处理
        3. 返回结果统一为业务层可用的格式
    """

    def __init__(
        self,
        path: str = "./chroma_db",
        collection_name: str = "docs",
        distance_metric: str = "cosine",
    ):
        """
        初始化向量存储。

        Args:
            path: ChromaDB 持久化路径
            collection_name: 集合名称
            distance_metric: 距离度量（cosine / l2 / ip）
        """
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.path))
        self.collection_name = collection_name
        self.distance_metric = distance_metric

        self._collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_metric},
        )

        logger.info(f"向量存储已初始化: path={path}, collection={collection_name}, count={self.count()}")

    # ================================================================
    # 核心接口
    # ================================================================

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """
        批量添加向量。

        Args:
            ids: 文档 ID 列表（必须唯一）
            embeddings: 向量列表
            documents: 原始文本列表
            metadatas: 元数据列表（可选）

        Returns:
            成功添加的文档数量

        Raises:
            ValueError: 如果 ids / embeddings / documents 长度不一致
            RuntimeError: 如果添加失败
        """
        if not (len(ids) == len(embeddings) == len(documents)):
            raise ValueError(f"ids, embeddings, documents 长度不一致: {len(ids)}, {len(embeddings)}, {len(documents)}")

        if not ids:
            logger.warning("空列表，跳过添加")
            return 0

        try:
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas or [{}] * len(ids),
            )
            logger.debug(f"已添加 {len(ids)} 条记录")
            return len(ids)

        except Exception as e:
            logger.error(f"添加失败: {e}")
            raise RuntimeError(f"向量添加失败: {e}")

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        filter_condition: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        向量检索。

        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            filter_condition: 元数据过滤条件（如 {"source": "test.txt"}）

        Returns:
            包含以下字段的字典：
                - ids: 文档 ID 列表
                - documents: 原始文本列表
                - metadatas: 元数据列表
                - distances: 距离列表

            如果无结果，所有字段为空列表。
        """
        if not query_embedding:
            logger.warning("查询向量为空")
            return {"ids": [], "documents": [], "metadatas": [], "distances": []}

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_condition,
                include=["documents", "metadatas", "distances"],
            )

            return {
                "ids": results.get("ids", [[]])[0],
                "documents": results.get("documents", [[]])[0],
                "metadatas": results.get("metadatas", [[]])[0],
                "distances": results.get("distances", [[]])[0],
            }

        except Exception as e:
            logger.error(f"查询失败: {e}")
            return {"ids": [], "documents": [], "metadatas": [], "distances": []}

    def delete(self, ids: List[str]) -> int:
        """
        删除指定 ID 的文档。

        Args:
            ids: 要删除的文档 ID 列表

        Returns:
            成功删除的数量
        """
        if not ids:
            return 0

        try:
            self._collection.delete(ids=ids)
            logger.debug(f"已删除 {len(ids)} 条记录")
            return len(ids)

        except Exception as e:
            logger.error(f"删除失败: {e}")
            return 0

    def delete_by_filter(self, filter_condition: Dict[str, Any]) -> int:
        """
        按条件删除文档。

        Args:
            filter_condition: 元数据过滤条件

        Returns:
            成功删除的数量
        """
        try:
            # 先获取符合条件的所有 ID
            results = self._collection.get(where=filter_condition)
            ids = results.get("ids", [])
            if ids:
                self._collection.delete(ids=ids)
            return len(ids)
        except Exception as e:
            logger.error(f"按条件删除失败: {e}")
            return 0

    def count(self) -> int:
        """获取集合中的文档总数"""
        try:
            return self._collection.count()
        except Exception as e:
            logger.error(f"获取计数失败: {e}")
            return 0

    def clear(self) -> None:
        """清空整个集合（删除后重建）"""
        try:
            self.client.delete_collection(self.collection_name)
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_metric},
            )
            logger.info(f"集合已清空: {self.collection_name}")
        except Exception as e:
            logger.error(f"清空失败: {e}")
            raise RuntimeError(f"清空集合失败: {e}")

    def get_all(self, limit: int = 1000) -> Dict[str, Any]:
        """
        获取集合中的所有文档（用于调试或导出）。

        Args:
            limit: 返回数量上限

        Returns:
            包含 ids, documents, metadatas 的字典
        """
        try:
            results = self._collection.get(limit=limit, include=["documents", "metadatas"])
            return {
                "ids": results.get("ids", []),
                "documents": results.get("documents", []),
                "metadatas": results.get("metadatas", []),
            }
        except Exception as e:
            logger.error(f"获取全部失败: {e}")
            return {"ids": [], "documents": [], "metadatas": []}

    def stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        return {
            "total_documents": self.count(),
            "collection_name": self.collection_name,
            "distance_metric": self.distance_metric,
            "path": str(self.path),
        }


# ================================================================
# 工厂函数
# ================================================================

def get_vector_store(config: Dict[str, Any]) -> VectorStore:
    """
    根据配置创建 VectorStore 实例。

    Args:
        config: 从 config.yaml 读取的 vector_store 配置块

    Returns:
        VectorStore 实例

    示例配置:
        vector_store:
          enabled: true
          backend: chromadb
          params:
            path: ./chroma_db
            collection_name: docs
    """
    if not config.get("enabled", True):
        logger.warning("vector_store 被禁用，返回空存储（无法使用）")
        # 返回一个可用的实例，但后续操作会报错
        return VectorStore(path="./chroma_db_disabled", collection_name="disabled")

    params = config.get("params", {})
    return VectorStore(
        path=params.get("path", "./chroma_db"),
        collection_name=params.get("collection_name", "docs"),
        distance_metric=params.get("distance_metric", "cosine"),
    )


# ================================================================
# 快速测试入口
# ================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 测试
    store = VectorStore(path="./test_chroma_db", collection_name="test")

    # 添加
    store.add(
        ids=["doc_1", "doc_2"],
        embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        documents=["内容1", "内容2"],
        metadatas=[{"source": "test1.txt"}, {"source": "test2.txt"}],
    )

    print(f"文档数量: {store.count()}")

    # 查询
    results = store.query(query_embedding=[0.1, 0.2, 0.3], top_k=1)
    print(f"查询结果: {results}")

    # 统计
    print(f"统计: {store.stats()}")

    # 清理
    store.clear()
    print(f"清空后数量: {store.count()}")