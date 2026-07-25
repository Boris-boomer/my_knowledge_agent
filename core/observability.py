# ================================================================
# observability.py：可观测性模块（Token 计数 + 成本追踪 + 结构化日志）
# ================================================================
#
# 【核心职责】
#   1. Token 计数：使用 tiktoken（OpenAI 标准）或回退算法精确计数
#   2. 成本估算：基于模型定价表，实时计算每次调用的成本
#   3. 结构化日志：输出 JSON 格式日志，可直接导入 ELK / Loki / Grafana
#   4. 链路追踪（Trace）：为每个请求生成 trace_id，关联所有操作
#
# 【为什么需要这个模块】
#   - Agent 系统 Token 消耗将增长 24 倍（高盛 2026 预测）
#   - 用户需要知道“每次回答花了多少钱”，才能做成本控制
#   - 生产环境必须结构化日志，而不是 print() 或 logging 的纯文本
# ================================================================

import json
import time
import uuid
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

# 尝试导入 tiktoken（OpenAI 官方分词器）
try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

logger = logging.getLogger(__name__)


# ================================================================
# 1. 定价表（单位：美元 / 百万 Token）
# ================================================================
# 价格来源：各厂商官网（2026 年 7 月）
# 更新策略：每季度手动更新一次，或从 model_registry 读取
# ================================================================

PRICING_TABLE = {
    # ---- OpenAI ----
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # ---- DeepSeek ----
    "deepseek-chat": {"input": 0.14, "output": 0.28},
    "deepseek-coder": {"input": 0.14, "output": 0.28},
    # ---- 阿里通义 ----
    "qwen-max": {"input": 0.20, "output": 0.60},
    "qwen-plus": {"input": 0.04, "output": 0.12},
    "qwen-turbo": {"input": 0.02, "output": 0.06},
    # ---- 智谱 GLM ----
    "glm-4-plus": {"input": 0.50, "output": 0.50},
    "glm-4": {"input": 0.10, "output": 0.10},
    "glm-4-flash": {"input": 0.01, "output": 0.01},
    # ---- 本地 Ollama（成本为 0）----
    "ollama": {"input": 0.00, "output": 0.00},
}


def get_pricing(model: str) -> Dict[str, float]:
    """获取模型的定价信息，如果不存在则返回默认值 0.5 和 1.5"""
    # 尝试精确匹配
    if model in PRICING_TABLE:
        return PRICING_TABLE[model]

    # 尝试模糊匹配（例如 "deepseek-r1:14b" 匹配 "deepseek-chat"）
    for key in PRICING_TABLE:
        if key in model or model in key:
            return PRICING_TABLE[key]

    # 兜底：返回通用价格（安全起见偏保守）
    return {"input": 0.50, "output": 1.50}


# ================================================================
# 2. Token 计数器
# ================================================================

class TokenCounter:
    """Token 计数器，支持 tiktoken 精确计数和回退算法"""

    def __init__(self, encoding_name: str = "cl100k_base"):
        """初始化计数器

        Args:
            encoding_name: tiktoken 编码名称，默认 cl100k_base（GPT-4 / 3.5 通用）
        """
        self.encoding_name = encoding_name
        self._encoding = None

        if TIKTOKEN_AVAILABLE:
            try:
                self._encoding = tiktoken.get_encoding(encoding_name)
            except Exception:
                logger.warning(f"获取 tiktoken 编码失败，将使用回退计数")

    def count(self, text: str) -> int:
        """计算文本的 Token 数量

        Args:
            text: 输入文本

        Returns:
            Token 数量
        """
        if not text:
            return 0

        if self._encoding:
            try:
                return len(self._encoding.encode(text))
            except Exception:
                logger.debug("tiktoken 计数失败，使用回退算法")

        # 回退算法：中英文混合的近似计数
        # 英文：约 4 字符 = 1 Token，中文：约 1.5 字 = 1 Token
        # 通用近似：总字符数 / 3（对中英文混合较为合理）
        return max(1, len(text) // 3)


# ================================================================
# 3. 成本计算器
# ================================================================

class CostCalculator:
    """成本计算器"""

    def __init__(self, token_counter: Optional[TokenCounter] = None):
        self.token_counter = token_counter or TokenCounter()

    def calculate(
        self,
        prompt: str,
        response: str,
        model: str,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """计算单次调用的成本

        Args:
            prompt: 用户输入
            response: 模型输出
            model: 模型名称
            input_tokens: 如果已知，直接传入，否则自动计数
            output_tokens: 如果已知，直接传入，否则自动计数

        Returns:
            包含 tokens 和 cost 的字典
        """
        # 计数
        if input_tokens is None:
            input_tokens = self.token_counter.count(prompt)
        if output_tokens is None:
            output_tokens = self.token_counter.count(response)

        # 获取定价
        pricing = get_pricing(model)

        # 计算成本（美元）
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        return {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6),
            "currency": "USD",
        }


# ================================================================
# 4. 链路追踪（Trace）
# ================================================================

@dataclass
class Span:
    """追踪的单个操作片段"""
    name: str
    start_time: float
    end_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


@dataclass
class Trace:
    """完整的链路追踪"""
    trace_id: str
    spans: List[Span] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def add_span(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> "Span":
        """添加一个 Span（开始计时）"""
        span = Span(name=name, start_time=time.time(), end_time=0.0, metadata=metadata or {})
        self.spans.append(span)
        return span

    def end_span(self, span: Span) -> None:
        """结束一个 Span（记录结束时间）"""
        span.end_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，用于 JSON 序列化"""
        return {
            "trace_id": self.trace_id,
            "duration_ms": (time.time() - self.start_time) * 1000,
            "spans": [
                {
                    "name": s.name,
                    "duration_ms": s.duration_ms,
                    "metadata": s.metadata,
                }
                for s in self.spans
            ],
        }


def new_trace() -> Trace:
    """创建一个新的 Trace"""
    return Trace(trace_id=str(uuid.uuid4()))


# ================================================================
# 5. 结构化日志
# ================================================================

class StructuredLogger:
    """结构化日志记录器（输出 JSON 格式）"""

    def __init__(self, service_name: str = "my_knowledge_agent"):
        self.service_name = service_name

    def log_llm_call(
        self,
        trace_id: str,
        prompt: str,
        response: str,
        model: str,
        cost_info: Dict[str, Any],
        success: bool,
        error: Optional[str] = None,
        **extra,
    ) -> None:
        """记录一次 LLM 调用（JSON 格式）"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "service": self.service_name,
            "event": "llm_call",
            "trace_id": trace_id,
            "model": model,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "success": success,
            "cost": cost_info,
        }

        if error:
            log_entry["error"] = error

        # 合并额外字段
        log_entry.update(extra)

        # 输出 JSON（INFO 级别）
        logger.info(json.dumps(log_entry, ensure_ascii=False))

    def log_cache_hit(
        self,
        trace_id: str,
        prompt: str,
        similarity: float,
        **extra,
    ) -> None:
        """记录缓存命中事件"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "service": self.service_name,
            "event": "cache_hit",
            "trace_id": trace_id,
            "prompt_length": len(prompt),
            "similarity": round(similarity, 4),
        }
        log_entry.update(extra)
        logger.info(json.dumps(log_entry, ensure_ascii=False))

    def log_error(
        self,
        trace_id: str,
        error: str,
        module: str,
        **extra,
    ) -> None:
        """记录错误事件"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "service": self.service_name,
            "event": "error",
            "trace_id": trace_id,
            "module": module,
            "error": error,
        }
        log_entry.update(extra)
        logger.error(json.dumps(log_entry, ensure_ascii=False))


# ================================================================
# 6. 全局单例（方便使用）
# ================================================================

_TOKEN_COUNTER: Optional[TokenCounter] = None
_COST_CALCULATOR: Optional[CostCalculator] = None
_STRUCTURED_LOGGER: Optional[StructuredLogger] = None


def get_token_counter() -> TokenCounter:
    global _TOKEN_COUNTER
    if _TOKEN_COUNTER is None:
        _TOKEN_COUNTER = TokenCounter()
    return _TOKEN_COUNTER


def get_cost_calculator() -> CostCalculator:
    global _COST_CALCULATOR
    if _COST_CALCULATOR is None:
        _COST_CALCULATOR = CostCalculator(get_token_counter())
    return _COST_CALCULATOR


def get_logger() -> StructuredLogger:
    global _STRUCTURED_LOGGER
    if _STRUCTURED_LOGGER is None:
        _STRUCTURED_LOGGER = StructuredLogger()
    return _STRUCTURED_LOGGER


# ================================================================
# 快速测试入口
# ================================================================

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)

    print("===== Token 计数测试 =====")
    counter = TokenCounter()
    text = "DSPy 是斯坦福大学开发的声明式编程框架，用于自动优化 LLM 提示词。"
    tokens = counter.count(text)
    print(f"文本: {text}")
    print(f"Token 数量: {tokens}")

    print("\n===== 成本计算测试 =====")
    calc = CostCalculator()
    result = calc.calculate(
        prompt="什么是 DSPy？",
        response="DSPy 是斯坦福大学开发的声明式编程框架，用于自动优化 LLM 提示词。",
        model="gpt-4o-mini",
    )
    print(f"成本详情: {json.dumps(result, indent=2)}")

    print("\n===== 结构化日志测试 =====")
    logger = get_logger()
    trace = new_trace()
    logger.log_llm_call(
        trace_id=trace.trace_id,
        prompt="什么是 DSPy？",
        response="DSPy 是斯坦福大学开发的声明式编程框架，用于自动优化 LLM 提示词。",
        model="gpt-4o-mini",
        cost_info=result,
        success=True,
        user_id="test_user",
    )

    print("\n===== 链路追踪测试 =====")
    trace = new_trace()
    span1 = trace.add_span("embedding_generation", {"model": "qwen3-embedding"})
    time.sleep(0.01)
    trace.end_span(span1)

    span2 = trace.add_span("llm_generation", {"model": "gpt-4o-mini"})
    time.sleep(0.02)
    trace.end_span(span2)

    print(json.dumps(trace.to_dict(), indent=2))

    print("\n✅ 可观测性模块测试完成")