# ================================================================
# reranker.py：重排序器
# ================================================================
#
# 【核心职责】
#   对向量检索的候选结果进行二次精排，提升检索精度。
#
# 【为什么需要 Reranker】
#   - 向量检索（Embedding）是"粗筛"，速度快但精度有限
#   - Reranker 是"精排"，用 Cross-Encoder 对每个候选重新打分
#   - 结合两者：先粗筛 10-20 个候选，再精排取前 3-5 个
#   - 实验数据：Reranker 可提升检索精度 15%-30%
#
# 【工作原理】
#   1. 向量检索返回 top_k * 3 个候选（粗筛）
#   2. 对每个 (query, doc) 对，用 Cross-Encoder 计算相关性分数
#   3. 按相关性分数重排序，取前 top_k 个
#   4. 可选的混合打分：0.4 * 向量相似度 + 0.6 * Reranker 分数
#
# 【使用方式】
#   from core.reranker import Reranker
#   reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
#   results = reranker.rerank(query, candidates, top_k=3)
# ================================================================

import os
# 设置 Hugging Face 镜像，解决国内访问官方源超时问题
# 环境变量必须在导入 sentence_transformers 前设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# 尝试导入 CrossEncoder；若未安装则降级处理
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logger.warning(
        "sentence-transformers 未安装，Reranker 功能不可用。"
        "安装命令: uv pip install sentence-transformers"
    )


class Reranker:
    """
    重排序器：对检索结果进行二次精排。

    使用 Cross-Encoder 对 (query, document) 对进行相关性打分，
    比向量检索更精确，但计算量更大，因此只对少量候选进行精排。

    Attributes:
        model_name: Cross-Encoder 模型名称
        enabled: Reranker 是否可用（由依赖安装情况和配置共同决定）
        _model: 延迟加载的模型实例，首次使用时加载
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        enabled: bool = True,
    ):
        """
        初始化 Reranker。

        Args:
            model_name: Cross-Encoder 模型名称
                - "cross-encoder/ms-marco-MiniLM-L-6-v2" (推荐，轻量)
                - "cross-encoder/ms-marco-MiniLM-L-12-v2" (更准，更大)
                - 中文场景: "BAAI/bge-reranker-base" (需下载)
            enabled: 是否启用 Reranker（可由配置文件控制）
        """
        self.model_name = model_name
        self.enabled = enabled and CROSS_ENCODER_AVAILABLE
        self._model = None

        if self.enabled:
            logger.info(f"Reranker 已启用，模型: {model_name}")
        else:
            if not enabled:
                logger.info("Reranker 已禁用（配置文件关闭）")
            else:
                logger.warning(
                    "sentence-transformers 未安装，Reranker 不可用。"
                    "安装命令: uv pip install sentence-transformers"
                )

    @property
    def model(self):
        """
        延迟加载模型，仅在实际使用时加载，避免启动时占用资源。

        加载失败时自动禁用 Reranker，并记录错误日志。

        Returns:
            CrossEncoder 实例，如果加载失败则返回 None
        """
        if self._model is None and self.enabled:
            try:
                # 直接加载，使用环境变量中设置的镜像源
                # 不加额外参数，避免版本兼容性问题
                self._model = CrossEncoder(self.model_name)
                logger.info(f"Reranker 模型加载成功: {self.model_name}")
            except Exception as e:
                logger.error(f"Reranker 模型加载失败: {e}")
                # 加载失败时禁用 Reranker，后续调用直接降级
                self.enabled = False
                self._model = None
        return self._model

    def is_available(self) -> bool:
        """检查 Reranker 是否可用（依赖已安装、配置启用、模型加载成功）"""
        return self.enabled and CROSS_ENCODER_AVAILABLE and self.model is not None

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 3,
        weight_vector: float = 0.4,
        weight_rerank: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """
        对候选结果进行重排序。

        Args:
            query: 用户查询
            candidates: 候选结果列表，每项需包含 "content" 字段
            top_k: 返回前 k 个结果
            weight_vector: 向量相似度权重（默认 0.4）
            weight_rerank: Reranker 分数权重（默认 0.6）

        Returns:
            重排序后的结果列表（按最终分数降序排列）；若不可用则返回前 top_k 个原始候选
        """
        # 边界情况：空输入
        if not candidates:
            return []

        # 若 Reranker 不可用，直接返回前 top_k 个候选（降级）
        if not self.is_available():
            logger.debug("Reranker 不可用，返回原始候选")
            return candidates[:top_k]

        try:
            # 提取文档内容，忽略空内容
            documents = [c.get("content", "") for c in candidates if c.get("content")]
            if not documents:
                return candidates[:top_k]

            # 构造 (query, doc) 对
            pairs = [(query, doc) for doc in documents]

            # 调用 CrossEncoder 打分
            scores = self.model.predict(pairs)

            # 合并分数，更新候选结果
            for i, candidate in enumerate(candidates):
                vector_score = candidate.get("similarity_score", 0.0)
                # 确保分数索引有效
                rerank_score = float(scores[i]) if i < len(scores) else 0.0

                # 归一化 Reranker 分数（通常 CrossEncoder 输出 -1 到 1，映射到 0-1）
                normalized_rerank = (rerank_score + 1) / 2

                # 计算最终分数：向量相似度与 Reranker 分数的加权和
                candidate["rerank_score"] = round(rerank_score, 4)
                candidate["final_score"] = round(
                    weight_vector * vector_score + weight_rerank * normalized_rerank * 100,
                    2,
                )

            # 按最终分数降序排列，取前 top_k 个
            sorted_candidates = sorted(
                candidates,
                key=lambda x: x.get("final_score", 0),
                reverse=True,
            )
            return sorted_candidates[:top_k]

        except Exception as e:
            # 任何异常都记录日志并返回原始候选（降级）
            logger.error(f"Reranker 重排序失败: {e}，返回原始候选")
            return candidates[:top_k]


def get_reranker(config: Optional[Dict[str, Any]] = None) -> Reranker:
    """
    工厂函数：根据配置创建 Reranker 实例。

    Args:
        config: config.yaml 中的 reranker 配置块

    Returns:
        Reranker 实例
    """
    if config is None:
        config = {}
    enabled = config.get("enabled", False)
    model_name = config.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    return Reranker(model_name=model_name, enabled=enabled)


# ================================================================
# 快速测试入口
# ================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("===== Reranker 测试 =====")

    # 创建 Reranker（使用轻量模型）
    reranker = get_reranker({"enabled": True})

    if not reranker.is_available():
        print("❌ Reranker 不可用，请安装 sentence-transformers 或检查网络")
        print("   安装命令: uv pip install sentence-transformers")
        exit(1)

    # 模拟候选结果
    candidates = [
        {"content": "DSPy 是斯坦福大学开发的声明式编程框架。", "similarity_score": 0.75},
        {"content": "DSPy 是数字信号处理（DSP）的 Python 扩展库。", "similarity_score": 0.82},
        {"content": "DSPy 是一个用于自动优化 LLM 提示词的编译器。", "similarity_score": 0.70},
        {"content": "DSPy 是深度学习的模型优化工具。", "similarity_score": 0.65},
    ]

    query = "DSPy 是什么？"

    print(f"\n查询: {query}")
    print("\n排序前:")
    for i, c in enumerate(candidates):
        print(f"  {i+1}. score={c['similarity_score']} - {c['content'][:30]}...")

    # 重排序
    results = reranker.rerank(query, candidates, top_k=3)

    print("\n排序后:")
    for i, r in enumerate(results):
        print(f"  {i+1}. rerank={r.get('rerank_score', 0):.4f} - {r['content'][:30]}...")

    print("\n✅ Reranker 测试完成")