# ================================================================
# vectorless_rag.py：无向量检索（文档树导航）
# ================================================================
import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from core.llm_router import ModelRouter

logger = logging.getLogger(__name__)


class DocumentNode:
    """文档树节点"""
    def __init__(self, title: str, content: str, level: int, source: str = ""):
        self.title = title
        self.content = content
        self.level = level
        self.source = source
        self.children: List["DocumentNode"] = []

    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "content": self.content[:500],
            "level": self.level,
            "source": self.source,
            "children": [c.to_dict() for c in self.children]
        }


class VectorlessRAG:
    """
    无向量检索：基于文档树结构的 LLM 导航检索。
    不使用向量数据库，不使用分块，直接理解文档结构。
    """

    def __init__(self, router: ModelRouter, doc_paths: List[str] = None):
        self.router = router
        self.trees: List[DocumentNode] = []
        self.max_depth = 5
        self._all_docs_cache = []  # 用于加速构建

        if doc_paths:
            self.build_trees(doc_paths)

    def build_trees(self, doc_paths: List[str]) -> None:
        """从文档路径构建文档树"""
        for path in doc_paths:
            if Path(path).exists():
                tree = self._build_tree_from_file(path)
                if tree:
                    self.trees.append(tree)
                    logger.info(f"构建文档树: {path}，{len(tree.children)} 个子章节")
        logger.info(f"共构建 {len(self.trees)} 个文档树")

    def _build_tree_from_file(self, file_path: str) -> Optional[DocumentNode]:
        """从单个文件构建树"""
        try:
            # 尝试使用 docling 解析（如果可用）
            try:
                from docling.document_converter import DocumentConverter
                converter = DocumentConverter()
                result = converter.convert(file_path)
                text = result.document.export_to_text()
                return self._parse_text_to_tree(text, file_path)
            except ImportError:
                # 降级：直接读取纯文本
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                return self._parse_text_to_tree(text, file_path)
        except Exception as e:
            logger.warning(f"构建文档树失败: {file_path}，错误: {e}")
            return None

    def _parse_text_to_tree(self, text: str, source: str) -> DocumentNode:
        """将文本解析为树结构（按标题层级）"""
        root = DocumentNode("root", "", 0, source)

        # 常见标题标记：#, ##, ###, 或者 一、, (一), 1.
        lines = text.split("\n")
        current_level = 0
        current_node = root
        stack = [root]

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # 检测标题级别
            level = self._detect_heading_level(stripped)

            if level > 0:
                # 是标题行
                title = stripped
                node = DocumentNode(title, "", level, source)
                # 找到正确父节点
                while len(stack) > level:
                    stack.pop()
                stack[-1].children.append(node)
                stack.append(node)
                current_node = node
            else:
                # 是正文内容
                if current_node and current_node.content:
                    current_node.content += "\n" + stripped
                elif current_node:
                    current_node.content = stripped

        return root

    def _detect_heading_level(self, line: str) -> int:
        """检测标题级别"""
        # Markdown: #, ##, ###
        if line.startswith("# "):
            return 1
        if line.startswith("## "):
            return 2
        if line.startswith("### "):
            return 3
        if line.startswith("#### "):
            return 4

        # 中文数字: 一、, 二、, (一), (1)
        if re.match(r"^[一二三四五六七八九十]、", line):
            return 1
        if re.match(r"^（[一二三四五六七八九十]）", line):
            return 2
        if re.match(r"^[0-9]+\.", line):
            return 2
        if re.match(r"^[0-9]+\.[0-9]+\.", line):
            return 3

        return 0

    def navigate(self, query: str, tree: DocumentNode, depth: int = 0) -> Tuple[Optional[str], int]:
        """
        LLM 在文档树上导航，返回 (答案, 消耗步数)

        核心逻辑：
        1. 当前节点有内容 → 让 LLM 判断能否回答
        2. 能回答 → 返回答案
        3. 不能回答 → 让 LLM 选择进入哪个子节点
        4. 子节点为空 → 返回 None
        """
        if depth > self.max_depth:
            return None, depth

        # 构建导航 Prompt
        child_titles = [f"{c.title[:50]}" for c in tree.children]

        navigate_prompt = f"""你是一个文档导航助手。你的任务是根据用户问题，决定在当前文档章节中如何行动。

【当前章节标题】{tree.title}
【当前章节内容】{tree.content[:800] if tree.content else '(无内容)'}
【子章节列表】{', '.join(child_titles) if child_titles else '(无子章节)'}

【用户问题】{query}

请决定下一步行动，只输出以下三种格式之一：
1. 如果当前章节内容足以回答用户问题，输出: ANSWER: <你的回答>
2. 如果需要进入某个子章节继续搜索，输出: GO: <子章节完整标题>
3. 如果当前及子章节都与问题无关，输出: SKIP

注意：如果内容不足以回答，不要强行回答。"""

        try:
            response = self.router.generate(navigate_prompt).strip()
            logger.debug(f"导航响应 (深度{depth}): {response[:100]}...")
        except Exception as e:
            logger.warning(f"LLM 导航失败: {e}")
            return None, depth

        # 解析响应
        if response.startswith("ANSWER:"):
            answer = response[7:].strip()
            if len(answer) > 10:
                logger.info(f"导航找到答案 (深度{depth})")
                return answer, depth + 1
            return None, depth + 1

        if response.startswith("GO:"):
            target = response[3:].strip()
            for child in tree.children:
                if child.title.strip() == target or child.title.startswith(target):
                    logger.info(f"进入子章节: {target}")
                    return self.navigate(query, child, depth + 1)
            # 模糊匹配
            for child in tree.children:
                if target in child.title or child.title in target:
                    logger.info(f"模糊匹配进入: {child.title}")
                    return self.navigate(query, child, depth + 1)
            logger.warning(f"未找到子章节: {target}")
            return None, depth + 1

        return None, depth + 1

    def retrieve(self, query: str) -> Dict[str, Any]:
        """
        对当前加载的所有文档树执行导航，返回最佳结果。
        """
        if not self.trees:
            return {"success": False, "answer": "", "steps": 0, "source": ""}

        best_answer = ""
        best_steps = 999
        best_source = ""

        for tree in self.trees:
            answer, steps = self.navigate(query, tree)
            if answer and steps < best_steps:
                best_answer = answer
                best_steps = steps
                best_source = getattr(tree, "source", "未知文档")

        if best_answer:
            return {
                "success": True,
                "answer": best_answer,
                "steps": best_steps,
                "source": best_source,
            }

        return {
            "success": False,
            "answer": "未找到相关内容",
            "steps": self.max_depth,
            "source": "",
        }