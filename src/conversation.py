# ================================================================
# conversation.py：对话引擎（多路检索 + 深度思考 + 自一致性投票 + 融合）
# ================================================================
import asyncio
import logging
import re
import time
import os
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from src.config_loader import load_config
from src.vector_store import get_vector_store
from core.llm_router import ModelRouter
from core.reranker import get_reranker

# BM25 相关（可选）
try:
    from rank_bm25 import BM25Okapi
    import jieba
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

logger = logging.getLogger(__name__)


# ================================================================
# 辅助：无向量检索（Vectorless RAG）
# ================================================================
class DocumentNode:
    """文档树节点"""
    def __init__(self, title: str, content: str, level: int, source: str = ""):
        self.title = title
        self.content = content
        self.level = level
        self.source = source
        self.children: List["DocumentNode"] = []


class VectorlessRAG:
    """
    基于文档树结构的 LLM 导航检索，不使用向量数据库。
    现在支持传入文件夹路径，自动递归扫描所有支持的文件。
    """

    def __init__(self, router: ModelRouter, doc_paths: List[str] = None):
        self.router = router
        self.trees: List[DocumentNode] = []
        self.max_depth = 5

        if doc_paths:
            self.build_trees(doc_paths)

    def build_trees(self, root_paths: List[str]) -> None:
        """从多个根路径递归扫描所有支持的文件，构建文档树"""
        supported_exts = {'.txt', '.pdf', '.docx'}
        file_paths = []

        for root_path in root_paths:
            if not os.path.exists(root_path):
                logger.warning(f"Vectorless: 路径不存在，跳过 {root_path}")
                continue
            if os.path.isfile(root_path):
                # 单个文件
                if Path(root_path).suffix.lower() in supported_exts:
                    file_paths.append(root_path)
                else:
                    logger.warning(f"Vectorless: 不支持的文件格式，跳过 {root_path}")
            else:
                # 文件夹：递归遍历
                for dirpath, _, filenames in os.walk(root_path):
                    for f in filenames:
                        if Path(f).suffix.lower() in supported_exts:
                            file_paths.append(os.path.join(dirpath, f))

        logger.info(f"Vectorless: 找到 {len(file_paths)} 个文档文件")
        for path in file_paths:
            tree = self._build_tree_from_file(path)
            if tree:
                self.trees.append(tree)
                logger.info(f"Vectorless: 构建文档树 {path}，{len(tree.children)} 个子章节")
        logger.info(f"Vectorless: 共构建 {len(self.trees)} 个文档树")

    def _build_tree_from_file(self, file_path: str):
        """从单个文件构建树，失败返回 None"""
        try:
            # 优先用 docling 解析
            try:
                from docling.document_converter import DocumentConverter
                converter = DocumentConverter()
                result = converter.convert(file_path)
                text = result.document.export_to_text()
            except ImportError:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except Exception as e:
                logger.warning(f"Vectorless: docling 解析失败 {file_path}: {e}，尝试直接读取")
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            if not text or len(text.strip()) < 10:
                return None
            return self._parse_text_to_tree(text, file_path)
        except Exception as e:
            logger.warning(f"Vectorless: 构建失败 {file_path}: {e}")
            return None

    def _parse_text_to_tree(self, text: str, source: str) -> DocumentNode:
        root = DocumentNode("root", "", 0, source)
        lines = text.split("\n")
        stack = [root]
        current_node = root

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            level = self._detect_heading_level(stripped)
            if level > 0:
                node = DocumentNode(stripped, "", level, source)
                while len(stack) > level:
                    stack.pop()
                stack[-1].children.append(node)
                stack.append(node)
                current_node = node
            else:
                if current_node and current_node.content:
                    current_node.content += "\n" + stripped
                elif current_node:
                    current_node.content = stripped
        return root

    def _detect_heading_level(self, line: str) -> int:
        if line.startswith("# "): return 1
        if line.startswith("## "): return 2
        if line.startswith("### "): return 3
        if line.startswith("#### "): return 4
        if re.match(r"^[一二三四五六七八九十]、", line): return 1
        if re.match(r"^（[一二三四五六七八九十]）", line): return 2
        if re.match(r"^[0-9]+\.", line): return 2
        if re.match(r"^[0-9]+\.[0-9]+\.", line): return 3
        return 0

    # ========== 改为异步 ==========
    async def navigate(self, query: str, tree: DocumentNode, depth: int = 0) -> Tuple[Optional[str], int]:
        if depth > self.max_depth:
            return None, depth

        child_titles = [c.title[:50] for c in tree.children]
        prompt = f"""你是一个文档导航助手。根据用户问题决定行动。

【当前章节】{tree.title}
【内容】{tree.content[:800] if tree.content else '(无)'}
【子章节】{', '.join(child_titles) if child_titles else '(无)'}

【问题】{query}

输出格式（只输出以下之一）：
- ANSWER: <你的回答>
- GO: <子章节标题>
- SKIP
"""
        try:
            # 使用异步的 generate_full
            response_obj = await self.router.generate_full(prompt)
            response = response_obj.content.strip()
        except Exception as e:
            logger.warning(f"导航生成失败: {e}")
            return None, depth

        if response.startswith("ANSWER:"):
            ans = response[7:].strip()
            if len(ans) > 10:
                return ans, depth + 1
            return None, depth + 1
        if response.startswith("GO:"):
            target = response[3:].strip()
            for child in tree.children:
                if child.title.strip() == target or child.title.startswith(target):
                    return await self.navigate(query, child, depth + 1)
            # 模糊匹配
            for child in tree.children:
                if target in child.title or child.title in target:
                    return await self.navigate(query, child, depth + 1)
            return None, depth + 1
        return None, depth + 1

    # ========== 改为异步 ==========
    async def retrieve(self, query: str) -> Dict[str, Any]:
        if not self.trees:
            return {"success": False, "answer": "", "steps": 0, "source": ""}
        best_answer = ""
        best_steps = 999
        best_source = ""
        for tree in self.trees:
            answer, steps = await self.navigate(query, tree)
            if answer and steps < best_steps:
                best_answer = answer
                best_steps = steps
                best_source = getattr(tree, "source", "未知文档")
        if best_answer:
            return {"success": True, "answer": best_answer, "steps": best_steps, "source": best_source}
        return {"success": False, "answer": "未找到相关内容", "steps": self.max_depth, "source": ""}


# ================================================================
# 对话引擎
# ================================================================
class ConversationEngine:
    """整合：混合检索 + Query Planning + 并行 Vectorless RAG + 深度思考 + 自一致性投票 + 融合"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = load_config(config_path)

        # 核心组件
        self.vector_store = get_vector_store(self.config.get("vector_store", {}))
        self.router = ModelRouter(
            llm_config=self.config.get("llm", {}),
            embedder_config=self.config.get("embedder", {}),
        )
        reranker_config = self.config.get("reranker", {})
        self.reranker = get_reranker(reranker_config)

        self.top_k = self.config.get("agent", {}).get("top_k", 3)
        self.doc_count = self.vector_store.count()

        # 功能开关
        self.enable_deep_think = True
        self.enable_self_consistency = True
        self.num_samples = 3
        self.enable_hybrid_search = True
        self.enable_query_planning = True
        self.retrieval_mode = "parallel"  # hybrid / vectorless / parallel

        # 对话历史
        self.conversation_history: List[Dict[str, str]] = []

        # BM25 索引
        self.bm25_index = None
        self.bm25_corpus = []
        if self.doc_count > 0 and BM25_AVAILABLE:
            self._build_bm25_index()
        elif self.doc_count > 0:
            logger.warning("BM25 不可用，仅使用向量检索")

        # Vectorless RAG（修复：正确扫描文件夹）
        self.vectorless_rag = None
        if self.retrieval_mode in ("vectorless", "parallel"):
            try:
                kb_paths = self.config.get("knowledge_base", {}).get("paths", [])
                if kb_paths:
                    self.vectorless_rag = VectorlessRAG(self.router, kb_paths)
                    logger.info(f"Vectorless RAG 初始化成功，共 {len(self.vectorless_rag.trees)} 个文档树")
                else:
                    logger.warning("无 knowledge_base.paths，Vectorless RAG 跳过")
            except Exception as e:
                logger.error(f"Vectorless RAG 初始化失败: {e}")
                self.vectorless_rag = None

        # 日志
        if self.doc_count == 0:
            logger.warning("向量库为空，请先构建知识库")
        else:
            logger.info(f"知识库已加载，共 {self.doc_count} 条文档")
            logger.info(f"深度思考: {'启用' if self.enable_deep_think else '禁用'}")
            logger.info(f"自一致性投票: {'启用' if self.enable_self_consistency else '禁用'} (采样数: {self.num_samples})")
            logger.info(f"混合检索: {'启用' if self.enable_hybrid_search else '禁用'}")
            logger.info(f"Query Planning: {'启用' if self.enable_query_planning else '禁用'}")
            logger.info(f"检索模式: {self.retrieval_mode}")

    # ========== BM25 索引构建 ==========
    def _build_bm25_index(self):
        try:
            all_docs = self.vector_store.get_all(limit=10000)
            if not all_docs.get("documents"):
                return
            self.bm25_corpus = all_docs["documents"]
            tokenized_corpus = [list(jieba.cut_for_search(doc)) for doc in self.bm25_corpus]
            self.bm25_index = BM25Okapi(tokenized_corpus)
            logger.info(f"BM25 索引构建完成，共 {len(self.bm25_corpus)} 条文档")
        except Exception as e:
            logger.error(f"BM25 索引构建失败: {e}")
            self.bm25_index = None

    def _bm25_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        if self.bm25_index is None:
            return []
        tokenized_query = list(jieba.cut_for_search(query))
        scores = self.bm25_index.get_scores(tokenized_query)
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:top_k]

    # ========== Query Planning ==========
    async def _plan_queries(self, query: str) -> List[str]:
        if not self.enable_query_planning:
            return [query]
        prompt = f"""用户问题：{query}
        生成 {self.num_samples} 个不同角度的检索词/问法，每行一个。"""
        try:
            response_obj = await self.router.generate_full(prompt)
            response = response_obj.content.strip()
            planned = [p.strip() for p in response.split("\n") if p.strip()]
            unique = []
            seen = set()
            for p in planned:
                if p not in seen and p != query:
                    seen.add(p)
                    unique.append(p)
            if unique:
                logger.info(f"Query Planning: {query} → {unique}")
                return [query] + unique[:3]
            return [query]
        except Exception:
            return [query]

    # ========== 混合检索 ==========
    def search_hybrid(self, query: str) -> List[Dict[str, Any]]:
        all_results = []
        seen_ids = set()
        recall_k = self.top_k * 2

        # 向量检索
        query_embedding = self.router.embed(query)
        if query_embedding:
            vec_results = self.vector_store.query(query_embedding, top_k=recall_k)
            if vec_results.get("ids"):
                for i in range(len(vec_results["ids"])):
                    doc_id = vec_results["ids"][i]
                    if doc_id in seen_ids:
                        continue
                    seen_ids.add(doc_id)
                    all_results.append({
                        "id": doc_id,
                        "content": vec_results["documents"][i],
                        "filename": vec_results["metadatas"][i].get("filename", "未知"),
                        "source": vec_results["metadatas"][i].get("source", ""),
                        "similarity_score": round(1 - vec_results["distances"][i], 4),
                        "vector_score": round(1 - vec_results["distances"][i], 4),
                        "bm25_score": 0.0,
                        "final_score": 0.0,
                    })

        # BM25 检索
        if self.enable_hybrid_search and self.bm25_index is not None:
            bm25_results = self._bm25_search(query, top_k=recall_k)
            if bm25_results:
                scores = [s for _, s in bm25_results]
                min_s = min(scores) if scores else 0
                max_s = max(scores) if scores else 1
                range_s = max_s - min_s if max_s > min_s else 1
                for idx, raw_score in bm25_results:
                    if idx >= len(self.bm25_corpus):
                        continue
                    doc_id = f"bm25_{idx}"
                    if doc_id in seen_ids:
                        continue
                    seen_ids.add(doc_id)
                    norm = (raw_score - min_s) / range_s if range_s > 0 else 0
                    all_results.append({
                        "id": doc_id,
                        "content": self.bm25_corpus[idx],
                        "filename": "BM25检索结果",
                        "source": "",
                        "similarity_score": round(norm, 4),
                        "vector_score": 0.0,
                        "bm25_score": round(norm, 4),
                        "final_score": 0.0,
                    })

        if not all_results:
            return []

        # 加权合并
        for r in all_results:
            vec = r.get("vector_score", 0.0)
            bm = r.get("bm25_score", 0.0)
            if self.enable_hybrid_search:
                r["final_score"] = round(0.6 * vec + 0.4 * bm, 4)
            else:
                r["final_score"] = vec

        all_results.sort(key=lambda x: x["final_score"], reverse=True)

        # Reranker
        if self.reranker.is_available() and all_results:
            all_results = self.reranker.rerank(query, all_results, top_k=self.top_k)
        else:
            all_results = all_results[:self.top_k]

        return all_results

    # ========== 上下文合并 ==========
    async def contextualize_query(self, current_query: str) -> str:
        if len(self.conversation_history) < 2:
            return current_query
        recent = self.conversation_history[-6:]
        context_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent])
        prompt = f"""对话历史：\n{context_str}\n\n当前问题：{current_query}\n请改写为完整独立的查询，只输出改写后的查询。"""
        try:
            response_obj = await self.router.generate_full(prompt)
            result = response_obj.content.strip()
            if result and len(result) > 2:
                logger.info(f"上下文改写: {current_query} → {result}")
                return result
        except Exception:
            pass
        return current_query

    # ========== 深度思考 ==========
    def _build_think_prompt(self, context: str, query: str) -> str:
        return f"""你是一个知识库助手。请按步骤思考并回答。

【思考步骤】
1. 理解问题
2. 分析信息（知识库+你的知识）
3. 推理
4. 得出结论

【输出格式】
<think>（思考过程）</think>
<answer>（最终回答）</answer>

【知识库内容】
{context}

【问题】{query}
"""

    def _parse_think_response(self, raw: str) -> Tuple[str, str]:
        think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
        thinking = think_match.group(1).strip() if think_match else ""
        answer = answer_match.group(1).strip() if answer_match else raw.strip()
        if not answer_match and think_match:
            parts = re.split(r"</think>", raw, maxsplit=1)
            if len(parts) > 1:
                answer = parts[1].strip()
                answer = re.sub(r"</?answer>", "", answer).strip()
        return thinking, answer

    async def _generate_with_think(self, context: str, query: str) -> Tuple[str, str, float]:
        start = time.time()
        prompt = self._build_think_prompt(context, query)
        full = ""
        async for chunk in self.router.stream_generate(prompt):
            full += chunk.text
        thinking, answer = self._parse_think_response(full)
        if not thinking:
            thinking = full
        elapsed = time.time() - start
        return thinking, answer, elapsed

    # ========== 自一致性投票 ==========
    async def _self_consistency_vote(self, context: str, query: str) -> Tuple[str, str, List[str]]:
        tasks = [self._generate_with_think(context, query) for _ in range(self.num_samples)]
        results = await asyncio.gather(*tasks)
        think_list = [r[0] for r in results]
        answer_list = [r[1] for r in results]
        logger.info(f"投票完成，{len(answer_list)} 个候选")

        counter = Counter(answer_list)
        most_common = counter.most_common(1)
        if most_common and most_common[0][1] >= 2:
            winner = most_common[0][0]
            idx = answer_list.index(winner)
            logger.info(f"投票胜出（重复 {most_common[0][1]} 次）")
            return think_list[idx], winner, answer_list

        # ================================================================
        # 法官择优（动态支持任意数量的候选，修复 IndexError）
        # ================================================================
        if len(answer_list) >= 2:
            logger.info("候选答案不同，启动法官模型择优")
            # 动态构建候选列表，而不是硬编码 3 个
            candidate_lines = "\n".join([f"{i+1}. {answer_list[i]}" for i in range(len(answer_list))])
            judge_prompt = f"""用户问题：{query}
候选答案：
{candidate_lines}
只输出数字（1-{len(answer_list)}），不要解释。"""
            try:
                judge_resp = self.router.generate(judge_prompt).strip()
                digits = re.findall(r"\d", judge_resp)
                if digits:
                    idx = int(digits[0]) - 1
                    if 0 <= idx < len(answer_list):
                        logger.info(f"法官选择: {idx+1}")
                        return think_list[idx], answer_list[idx], answer_list
            except Exception:
                pass

        # 兜底：选最长的
        longest_idx = max(range(len(answer_list)), key=lambda i: len(answer_list[i]))
        logger.info(f"兜底选择最长的答案: {longest_idx+1}")
        return think_list[longest_idx], answer_list[longest_idx], answer_list

    # ========== 并行检索 + 融合 ==========
    async def _async_hybrid_search(self, query: str) -> List[Dict[str, Any]]:
        return self.search_hybrid(query)

    async def _async_vectorless_search(self, query: str) -> Dict[str, Any]:
        if not self.vectorless_rag:
            return {"success": False, "answer": "", "steps": 0, "source": ""}
        return await self.vectorless_rag.retrieve(query)

    async def _merge_results(self, query: str, rag_results: List[Dict], vectorless_result: Dict) -> Optional[str]:
        if not rag_results and not vectorless_result.get("success"):
            return None
        rag_context = ""
        if rag_results:
            rag_context = "\n".join([f"【来源 {r['filename']}】\n{r['content']}\n" for r in rag_results[:2]])
        vectorless_context = vectorless_result.get("answer", "") if vectorless_result.get("success") else ""

        if not rag_context and not vectorless_context:
            return None

        merge_prompt = f"""请综合以下两份信息回答用户问题。

【信息A：向量检索】
{rag_context if rag_context else '（无）'}

【信息B：文档树导航】
{vectorless_context if vectorless_context else '（无）'}

【用户问题】{query}

要求：综合两者，若有冲突指出，以详细者为准。回答清晰有条理。
请回答："""
        try:
            response_obj = await self.router.generate_full(merge_prompt)
            return response_obj.content
        except Exception as e:
            logger.error(f"融合失败: {e}")
            return None

    # ================================================================
    # 主对话流（优化格式版：思考与回答清晰区分）
    # ================================================================
    async def answer_stream(self, query: str, use_vectorless: bool = True):
        # 1. 上下文改写（改为 await）
        contextualized_query = await self.contextualize_query(query)
        self.conversation_history.append({"role": "user", "content": query})

        # 2. 并行检索（根据模式）
        rag_results = None
        vectorless_result = None

        if self.retrieval_mode == "parallel":
            hybrid_task = asyncio.create_task(self._async_hybrid_search(contextualized_query))
            if use_vectorless and self.vectorless_rag:
                vectorless_task = asyncio.create_task(self._async_vectorless_search(contextualized_query))
                rag_results = await hybrid_task
                vectorless_result = await vectorless_task

                # 尝试融合
                if rag_results or vectorless_result.get("success"):
                    merged = await self._merge_results(contextualized_query, rag_results or [], vectorless_result or {})
                    if merged:
                        yield merged
                        self.conversation_history.append({"role": "assistant", "content": merged})
                        return
            else:
                # 只走混合检索
                rag_results = await hybrid_task
                # 不融合，继续往下走单路生成

        elif self.retrieval_mode == "hybrid":
            rag_results = self.search_hybrid(contextualized_query)
        elif self.retrieval_mode == "vectorless":
            if use_vectorless and self.vectorless_rag:
                vectorless_result = await self.vectorless_rag.retrieve(contextualized_query)

        # 3. 构造上下文
        if rag_results:
            context_parts = [f"【来自 {r['filename']}】\n{r['content']}\n" for r in rag_results[:3]]
            context = "\n".join(context_parts) if context_parts else "（无相关内容）"
        elif vectorless_result and vectorless_result.get("success"):
            context = vectorless_result["answer"]
        else:
            context = "（知识库中没有找到相关内容）"

        # ============================================================
        # 4. 深度思考 + 投票（优化格式版）
        # ============================================================
        final_answer = ""  # 确保变量始终存在

        if self.enable_self_consistency and (rag_results or (vectorless_result and vectorless_result.get("success"))):
            yield f"\n🧠 正在深度思考（{self.num_samples} 次采样）...\n\n"
            thinking, final_answer, _ = await self._self_consistency_vote(context, contextualized_query)

            if thinking:
                # 用分隔线和引用块区分思考过程
                yield "---\n"
                yield "**💭 思考过程：**\n"
                yield "> " + thinking.replace("\n", "\n> ") + "\n\n"
                yield "---\n\n"

            # 最终回答用醒目标题 + 普通正文
            yield "**📝 最终回答：**\n\n"
            if final_answer:
                for char in final_answer:
                    yield char
                yield "\n\n"
            else:
                yield "（模型未生成有效答案，请重试）\n\n"

        elif self.enable_deep_think:
            yield "\n🧠 正在深度思考...\n\n"
            thinking, final_answer, _ = await self._generate_with_think(context, contextualized_query)

            if thinking:
                yield "---\n"
                yield "**💭 思考过程：**\n"
                yield "> " + thinking.replace("\n", "\n> ") + "\n\n"
                yield "---\n\n"

            yield "**📝 最终回答：**\n\n"
            if final_answer:
                for char in final_answer:
                    yield char
                yield "\n\n"
            else:
                yield "（模型未生成有效答案，请重试）\n\n"

        else:
            # 传统模式（无强制思考）
            prompt = f"你是一个知识库助手。请根据以下内容回答问题：\n{context}\n\n问题：{contextualized_query}\n回答："
            async for chunk in self.router.stream_generate(prompt):
                yield chunk.text
            final_answer = ""  # 传统模式下没有 final_answer

        # 5. 来源引用（如果有检索结果）
        if rag_results:
            sources = [f"📄 {r['filename']}" for r in rag_results[:3]]
            yield f"\n\n---\n📎 参考来源: {', '.join(sources)}"
        elif vectorless_result and vectorless_result.get("success"):
            yield f"\n\n---\n📎 参考来源: {vectorless_result.get('source', '未知文档')}"

        # 保存回答到历史
        self.conversation_history.append({"role": "assistant", "content": final_answer if final_answer else ""})

    # ========== 管理方法 ==========
    def get_stats(self) -> Dict[str, Any]:
        return {
            "doc_count": self.vector_store.count(),
            "top_k": self.top_k,
            "reranker_enabled": self.reranker.is_available(),
            "history_length": len(self.conversation_history),
            "deep_think_enabled": self.enable_deep_think,
            "self_consistency_enabled": self.enable_self_consistency,
            "hybrid_search_enabled": self.enable_hybrid_search,
            "query_planning_enabled": self.enable_query_planning,
            "retrieval_mode": self.retrieval_mode,
        }

    def reload(self):
        self.config = load_config(self.config_path)
        self.vector_store = get_vector_store(self.config.get("vector_store", {}))
        self.doc_count = self.vector_store.count()
        if BM25_AVAILABLE and self.doc_count > 0:
            self._build_bm25_index()
        # 重建 Vectorless RAG
        if self.retrieval_mode in ("vectorless", "parallel"):
            kb_paths = self.config.get("knowledge_base", {}).get("paths", [])
            if kb_paths:
                self.vectorless_rag = VectorlessRAG(self.router, kb_paths)
        return self.doc_count

    def clear_history(self):
        self.conversation_history = []