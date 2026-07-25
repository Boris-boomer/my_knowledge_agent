# ================================================================
# base.py：LLM 核心基类与数据结构
# ================================================================
#
# 这个文件定义了整个 core/ 层的基础类型和抽象接口。
# 所有适配器（Ollama、OpenAI 等）都实现这里定义的接口，
# 所有上层代码（src/、其他项目）都使用这里定义的数据结构。
#
# 【核心职责】
#   1. 定义 LLM 适配器的抽象基类（BaseLLM）
#   2. 定义标准化数据结构（LLMResponse、TokenChunk、ToolCall）
#   3. 定义工具调用相关类型（Tool）
#   4. 预留多 Agent 协作接口（AgentCapable、SubTask、AgentResult）
#
# 【为什么需要这个文件】
#   - 统一接口：所有适配器实现同一套接口，上层代码不依赖具体实现
#   - 标准化数据：确保不同厂商的返回格式被统一为同一种数据结构
#   - 可扩展性：新增厂商只需继承 BaseLLM，不影响上层代码
#   - 未来兼容：预留多 Agent 协作接口，便于后续扩展
#
# 【设计原则】
#   - 接口与实现分离：上层只依赖抽象接口，不依赖具体适配器
#   - 数据不可变：所有数据结构使用 dataclass，避免意外修改
#   - 类型安全：使用类型注解，便于 IDE 提示和静态检查
#
# 【对外暴露】
#   BaseLLM              - 所有适配器的抽象基类（必须实现 generate / stream_generate）
#   LLMResponse          - 非流式响应的标准化结构
#   TokenChunk           - 流式响应的每个数据块
#   Tool                 - 工具定义（函数调用的参数结构）
#   ToolCall             - 模型返回的工具调用指令
#   AgentCapable         - 多 Agent 协作能力接口（预留）
#   SubTask              - 子任务定义（多 Agent 协作）
#   AgentResult          - Agent 执行结果（多 Agent 协作）
# ================================================================

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncIterator, Literal, TypedDict
from dataclasses import dataclass, field


# ================================================================
# 第一部分：工具调用（Tool Calling）定义
# ================================================================
#
# 工具调用是 Agent 系统的核心能力：模型可以返回结构化的指令，
# 告诉上层代码"请调用某个函数"，而不是直接输出文本。
#
# 为什么这样设计：
#   - 标准化的工具定义，兼容 OpenAI、智谱、DeepSeek 等主流厂商
#   - 上层代码通过 ToolCall 对象可以安全地执行外部操作（搜索、计算等）
# ================================================================


class Tool(TypedDict, total=False):
    """
    工具定义（符合 OpenAI / 智谱 / DeepSeek 通用标准）。

    使用示例：
        Tool(
            type="function",
            function={
                "name": "search",
                "description": "搜索互联网获取实时信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索关键词"}
                    },
                    "required": ["query"]
                }
            }
        )
    """
    type: Literal["function"]
    function: Dict[str, Any]  # 包含 name, description, parameters


@dataclass
class ToolCall:
    """
    模型返回的工具调用指令。

    当模型决定调用工具时，会返回此类实例。
    上层代码需要根据 name 调用对应的函数，并将 arguments 作为参数传入。

    Attributes:
        id: 调用唯一标识（用于追踪）
        name: 要调用的工具名称
        arguments: 调用参数（已解析为 Python 字典）
    """
    id: str
    name: str
    arguments: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，便于序列化"""
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


# ================================================================
# 第二部分：流式输出数据块
# ================================================================
#
# 流式输出是 2026 年 LLM 应用的标配，用户体验要求"逐字显示"。
# 每个 TokenChunk 代表一个输出片段，上层可以实时显示。
#
# 为什么要有 is_final 字段：
#   - 让上层知道"这个 chunk 是最后一块了"，可以关闭连接或更新状态
#   - 流式响应结束时，finish_reason 会包含停止原因（stop / length / tool_calls）
# ================================================================


@dataclass
class TokenChunk:
    """
    流式输出的每一个数据块。

    使用方式：
        async for chunk in adapter.stream_generate("你好"):
            print(chunk.text, end="", flush=True)
            if chunk.is_final:
                print(f"\n完成原因: {chunk.finish_reason}")

    Attributes:
        text: 当前块的文本内容（可能为空，比如最后一块）
        is_final: 是否为最后一块（True 表示流式输出结束）
        finish_reason: 结束原因（"stop" / "length" / "tool_calls" / "cache"）
        token_count: 本块的 Token 数量（仅在 is_final=True 时有值）
        reasoning: 推理过程（仅当模型输出 <think> 标签时才有）
    """
    text: str
    is_final: bool = False
    finish_reason: Optional[str] = None
    token_count: Optional[int] = None
    reasoning: Optional[str] = None


# ================================================================
# 第三部分：标准化响应对象
# ================================================================
#
# 为什么要有 LLMResponse：
#   - 各厂商 API 返回格式不一致（Ollama 返回 JSON，OpenAI 返回 ChatCompletion）
#   - 上层代码需要一个统一的格式来处理所有厂商的响应
#   - 分离 reasoning（推理过程）和 content（最终答案），便于分别处理
#
# 为什么把 reasoning 和 content 分开：
#   - DeepSeek-R1 等模型会在 <think> 标签中输出推理过程
#   - 上层可能想展示推理过程（提升透明度），也可能只想展示最终答案
#   - 分离后，上层可以自由选择显示哪些内容
# ================================================================


@dataclass
class LLMResponse:
    """
    完整的非流式模型响应。

    这是 generate() 方法的标准返回格式。
    所有适配器必须返回此类型的对象。

    Attributes:
        content: 纯净的最终答案（已剥离推理过程）
        reasoning: 推理过程（如果模型输出过 <think> 标签）
        tool_calls: 工具调用列表（如果模型请求调用工具）
        finish_reason: 结束原因（"stop" / "length" / "tool_calls"）
        usage: Token 使用统计 {"prompt_tokens": 10, "completion_tokens": 20}
    """
    content: str
    reasoning: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: Optional[str] = None
    usage: Dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """总 Token 数（输入 + 输出）"""
        return self.usage.get("prompt_tokens", 0) + self.usage.get("completion_tokens", 0)

    @property
    def has_tool_calls(self) -> bool:
        """是否包含工具调用"""
        return len(self.tool_calls) > 0

    @property
    def is_complete(self) -> bool:
        """是否正常完成（不是被截断）"""
        return self.finish_reason in ("stop", "tool_calls")


# ================================================================
# 第四部分：抽象基类
# ================================================================
#
# 所有 LLM 适配器必须继承 BaseLLM 并实现其抽象方法。
# 上层代码（如 ModelRouter）只依赖这个抽象接口，不依赖具体实现。
#
# 为什么要有抽象基类：
#   - 强制所有适配器实现相同的方法签名
#   - 便于单元测试（可以用 Mock 实现）
#   - 新增厂商时，只需实现 BaseLLM，无需改动上层逻辑
#
# 为什么要有同步和异步两个版本：
#   - generate(): 同步版本，简单场景使用（如命令行工具）
#   - stream_generate(): 异步版本，Web 界面必须（流式输出用户体验更好）
#   - 两个方法都提供，由调用者按需选择
# ================================================================


class BaseLLM(ABC):
    """
    所有 LLM 适配器的抽象基类。

    这是整个 core/ 层的核心接口。所有适配器（Ollama、OpenAI、Qwen 等）
    都必须实现此类中的所有抽象方法。

    实现者注意：
        1. generate() 必须返回 LLMResponse 对象
        2. stream_generate() 必须返回 AsyncIterator[TokenChunk]
        3. health_check() 用于路由器的故障转移判断
        4. estimate_cost() 用于成本追踪
    """

    # ================================================================
    # 同步文本生成
    # ================================================================

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
        tools: Optional[List[Tool]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        生成回答（非流式，同步）。

        这是最基础的方法，适用于不需要流式输出的场景。
        所有适配器必须实现此方法。

        Args:
            prompt: 用户输入的提示词
            reasoning_effort: 推理强度（low/medium/high），仅部分模型支持（如 o1、R1）
            tools: 可用的工具列表（函数调用），支持 Agent 场景
            temperature: 温度参数（0-1），覆盖默认值
            max_tokens: 最大输出 Token 数，覆盖默认值
            **kwargs: 厂商特定参数（用于扩展）

        Returns:
            LLMResponse 对象，包含内容、推理、工具调用和 Token 统计
        """
        pass

    # ================================================================
    # 异步流式文本生成
    # ================================================================

    @abstractmethod
    async def stream_generate(
        self,
        prompt: str,
        *,
        reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
        tools: Optional[List[Tool]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[TokenChunk]:
        """
        生成回答（流式，异步）。

        适用于需要逐字显示的场景（如 Web 聊天界面）。
        所有适配器必须实现此方法。

        使用方式：
            async for chunk in adapter.stream_generate("你好"):
                print(chunk.text, end="", flush=True)

        Args:
            参数同 generate()

        Yields:
            TokenChunk 对象，每个包含一小段文本和状态信息
        """
        yield TokenChunk(text="")  # 占位，子类必须实现

    # ================================================================
    # 健康检查（用于路由器的故障转移）
    # ================================================================

    @abstractmethod
    async def health_check(self, deep: bool = False) -> bool:
        """
        检查模型是否可用（异步）。

        路由器在调用前会先执行健康检查，如果不可用则自动切换到备选模型。

        Args:
            deep: 是否进行深度检查
                  - False: 只检查服务是否存活（ping）
                  - True: 发送真实短 prompt 验证推理能力

        Returns:
            True 表示可用，False 表示不可用
        """
        pass

    # ================================================================
    # 成本估算（用于可观测性）
    # ================================================================

    @abstractmethod
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        估算本次调用的成本（美元）。

        用于 Token 成本追踪，帮助用户控制费用。

        Args:
            prompt_tokens: 输入 Token 数量
            completion_tokens: 输出 Token 数量

        Returns:
            估算成本（美元）
        """
        pass


# ================================================================
# 第五部分：多 Agent 协作接口（预留，🟢 P2）
# ================================================================
#
# 参考 2026 年最新架构（DeerFlow 2.0 / Hyra-1.0）：
#   - 复杂任务由多个 Agent 协作完成
#   - 规划 Agent 拆解任务 → 多个执行 Agent 并行工作 → 合成 Agent 合并结果
#
# 此接口为预留设计，不影响当前功能。
# 未来实现时可参考以下模式：
#   1. 任务分解：plan_agent.decompose(question) -> List[SubTask]
#   2. 并行执行：executor_agents.run(sub_tasks) -> List[AgentResult]
#   3. 结果合成：synthesis_agent.merge(results) -> FinalAnswer
# ================================================================


@dataclass
class SubTask:
    """
    子任务定义（用于多 Agent 协作）。

    Attributes:
        id: 子任务唯一标识
        description: 任务描述
        agent_type: 需要的 Agent 类型（"search" / "reason" / "summarize" / "code" / etc.）
        dependencies: 依赖的其他子任务 ID 列表（用于执行顺序控制）
        params: 任务参数（灵活扩展）
    """
    id: str
    description: str
    agent_type: str
    dependencies: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """
    Agent 执行结果。

    Attributes:
        task_id: 对应的子任务 ID
        success: 是否执行成功
        content: 执行结果内容
        metadata: 额外元数据（如执行耗时、Token 使用等）
        error: 错误信息（仅当 success=False 时有值）
    """
    task_id: str
    success: bool
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class AgentCapable(ABC):
    """
    多 Agent 协作能力接口（预留）。

    实现此接口的类可以参与多 Agent 协作系统。
    当前版本仅提供接口定义，不包含完整实现。
    """

    @abstractmethod
    async def execute_task(self, task: SubTask) -> AgentResult:
        """
        执行单个子任务（预留）。

        Args:
            task: 子任务定义

        Returns:
            AgentResult: 执行结果
        """
        pass

    @abstractmethod
    async def decompose(self, question: str) -> List[SubTask]:
        """
        将复杂问题拆解为子任务列表（预留）。

        Args:
            question: 用户问题

        Returns:
            List[SubTask]: 子任务列表
        """
        pass

    @abstractmethod
    async def synthesize(self, results: List[AgentResult]) -> str:
        """
        合并多个子任务的结果（预留）。

        Args:
            results: 子任务执行结果列表

        Returns:
            str: 合并后的最终回答
        """
        pass


# ================================================================
# 快速测试入口
# ================================================================
# 直接运行此文件可验证基础数据结构是否正常。
# ================================================================

if __name__ == "__main__":
    print("===== 测试数据结构 =====")

    # 测试 LLMResponse
    response = LLMResponse(
        content="DSPy 是斯坦福大学开发的声明式编程框架。",
        reasoning="用户询问 DSPy 的定义，我根据知识库回答。",
        finish_reason="stop",
        usage={"prompt_tokens": 10, "completion_tokens": 20},
    )
    print(f"响应内容: {response.content}")
    print(f"推理过程: {response.reasoning}")
    print(f"总 Token: {response.total_tokens}")
    print(f"是否完成: {response.is_complete}")

    # 测试 ToolCall
    tool_call = ToolCall(
        id="call_123",
        name="search",
        arguments={"query": "DSPy 最新版本"},
    )
    print(f"\n工具调用: {tool_call.name} -> {tool_call.arguments}")

    # 测试 TokenChunk
    chunk = TokenChunk(
        text="你好",
        is_final=False,
    )
    print(f"\n流式块: '{chunk.text}' (final={chunk.is_final})")

    # 测试多 Agent 数据结构（P2 预留）
    subtask = SubTask(
        id="task_001",
        description="搜索 DSPy 相关信息",
        agent_type="search",
        params={"query": "DSPy"},
    )
    print(f"\n子任务: {subtask.id} - {subtask.agent_type}")

    print("\n✅ 所有数据结构测试通过")