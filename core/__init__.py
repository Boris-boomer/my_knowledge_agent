# ================================================================
# core/__init__.py：核心库的公共接口
# ================================================================
#
# 这个文件定义了 core/ 层的对外暴露接口。
# 外部模块（如 src/）只需要 from core import ... 即可使用所有功能。
#
# 【使用方式】
#   from core import ModelRouter, SemanticCache, CostCalculator
#
#   router = ModelRouter(llm_config=config["llm"], embedder_config=config["embedder"])
#   response = router.generate("你好")
#   vector = router.embed("文本")
# ================================================================

__version__ = "1.0.0"


# ================================================================
# 1. 基础数据结构（base.py）
# ================================================================

from .base import (
    # ---- LLM 适配器基类 ----
    BaseLLM,
    # ---- 响应对象 ----
    LLMResponse,
    TokenChunk,
    # ---- 工具调用 ----
    Tool,
    ToolCall,
    # ---- 多 Agent 协作（P2 预留） ----
    AgentCapable,
    SubTask,
    AgentResult,
)


# ================================================================
# 2. 语义缓存（semantic_cache.py）
# ================================================================

from .semantic_cache import (
    SemanticCache,
    get_cache,
)


# ================================================================
# 3. 可观测性（observability.py）
# ================================================================

from .observability import (
    TokenCounter,
    CostCalculator,
    StructuredLogger,
    new_trace,
    get_token_counter,
    get_cost_calculator,
    get_logger,
)


# ================================================================
# 4. 模型路由器（llm_router.py）
# ================================================================

from .llm_router import (
    ModelRouter,
    get_adapter,
)


# ================================================================
# 5. Reranker 重排序器（reranker.py）- 🟡 P1
# ================================================================

from .reranker import (
    Reranker,
    get_reranker,
)


# ================================================================
# 6. MIPROv2 优化器（mipro_optimizer.py）- 🟢 P2 预留
# ================================================================

from .mipro_optimizer import (
    MIPROv2Optimizer,
    MIPROv2Config,
    get_mipro_optimizer,
)


# ================================================================
# 7. 公共 API 列表
# ================================================================

__all__ = [
    # ---- 基础数据结构 ----
    "BaseLLM",
    "LLMResponse",
    "TokenChunk",
    "Tool",
    "ToolCall",
    # ---- 多 Agent 协作（P2 预留） ----
    "AgentCapable",
    "SubTask",
    "AgentResult",
    # ---- 语义缓存 ----
    "SemanticCache",
    "get_cache",
    # ---- 可观测性 ----
    "TokenCounter",
    "CostCalculator",
    "StructuredLogger",
    "new_trace",
    "get_token_counter",
    "get_cost_calculator",
    "get_logger",
    # ---- 模型路由器 ----
    "ModelRouter",
    "get_adapter",
    # ---- Reranker（P1） ----
    "Reranker",
    "get_reranker",
    # ---- MIPROv2 优化器（P2 预留） ----
    "MIPROv2Optimizer",
    "MIPROv2Config",
    "get_mipro_optimizer",
]