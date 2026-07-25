# ================================================================
# chunker.py：文本分块器（升级版：语义边界感知的动态分块）
# ================================================================
#
# 【核心职责】
#   把长文本切成小块，保持语义完整性，用于向量化存储和检索。
#
# 【为什么需要动态分块】
#   - 固定大小分块（如 500 字）会在句子中间切断，破坏语义
#   - 短文档被切成无意义的碎片，长文档在中间被截断
#   - 动态分块根据段落、句子等语义边界智能切分，每个块都是语义单元
#
# 【分块策略（优先级从高到低）】
#   1. 段落边界（\n\n）：最高优先级，保持主题完整
#   2. 句子边界（。！？）：段落太长时按句子切分
#   3. 字数上限（max_chunk_size）：兜底，防止块过大
#
# 【核心算法】
#   1. 按空行切分为段落
#   2. 如果段落大小在 [min, max] 之间 → 直接作为一块
#   3. 如果段落小于 min → 与下一段落合并（保持语义连贯）
#   4. 如果段落大于 max → 按句子切分成多个块
#   5. 相邻块之间保留重叠文字（overlap），防止边界信息丢失
#
# 【对外暴露】
#   chunk_text(text, min_chunk_size, max_chunk_size, overlap) -> List[str]
#   chunk_text_fixed(text, chunk_size, overlap) -> List[str]  # 兼容旧版
#
# 【使用示例】
#   from chunker import chunk_text
#   chunks = chunk_text(long_text, min_chunk_size=200, max_chunk_size=800, overlap=50)
# ================================================================

import re
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


# ================================================================
# 常量
# ================================================================

# 中文句子结束符（用于按句子切分）
SENTENCE_ENDINGS = r'[。！？\n]'

# 默认分块参数
DEFAULT_MIN_CHUNK_SIZE = 200
DEFAULT_MAX_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 50


# ================================================================
# 主函数：动态分块
# ================================================================

def chunk_text(
    text: str,
    min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[str]:
    """
    语义边界感知的动态分块。

    Args:
        text: 待切分的文本
        min_chunk_size: 每块最小字数（短段落会合并）
        max_chunk_size: 每块最大字数（长段落会切分）
        overlap: 块与块之间的重叠字数（防止边界信息丢失）

    Returns:
        文本块列表，如果文本为空则返回空列表

    Example:
        >>> chunks = chunk_text("很长的文档内容...", min_chunk_size=200, max_chunk_size=800)
        >>> print(f"切成 {len(chunks)} 块")
    """
    if not text or not text.strip():
        logger.warning("输入文本为空，返回空列表")
        return []

    # ----------------------------------------------------------------
    # 步骤 1：按空行切分为段落
    # ----------------------------------------------------------------
    # 大多数正式文档用空行分隔不同主题。
    # 按段落切分，每个块的内容更集中。
    # ----------------------------------------------------------------
    paragraphs = _split_by_paragraphs(text)

    if not paragraphs:
        return []

    # ----------------------------------------------------------------
    # 步骤 2：段落合并与切分
    # ----------------------------------------------------------------
    chunks = []
    buffer = ""

    for para in paragraphs:
        para_len = len(para)

        # 如果缓冲区 + 当前段落不超过 max，合并
        if len(buffer) + para_len <= max_chunk_size:
            buffer = buffer + "\n" + para if buffer else para
        else:
            # 缓冲区不为空，先保存
            if buffer:
                chunks.append(buffer)
                buffer = ""

            # 段落本身在范围内，直接作为一块
            if min_chunk_size <= para_len <= max_chunk_size:
                chunks.append(para)

            # 段落太长，按句子切分
            elif para_len > max_chunk_size:
                sub_chunks = _split_by_sentences(para, max_chunk_size)
                chunks.extend(sub_chunks)

            # 段落太短，留到下次合并
            else:
                buffer = para

    # 处理剩余缓冲区
    if buffer:
        # 如果缓冲区太小，合并到上一块
        if len(buffer) < min_chunk_size and chunks:
            chunks[-1] = chunks[-1] + "\n" + buffer
        else:
            chunks.append(buffer)

    # ----------------------------------------------------------------
    # 步骤 3：应用重叠（overlap）
    # ----------------------------------------------------------------
    # 重叠的目的是防止关键信息被切在块边界上。
    # 比如“张三在2024年...”如果“张三”在上一块末尾，“在2024年”在下一块开头，
    # 检索时可能只召回其中一块，导致信息不完整。
    # 重叠让边界信息同时出现在相邻两块中。
    # ----------------------------------------------------------------
    if overlap > 0 and len(chunks) > 1:
        chunks = _apply_overlap(chunks, overlap)

    # 过滤空块
    chunks = [c.strip() for c in chunks if c.strip()]

    logger.debug(f"动态分块完成：原文 {len(text)} 字 → {len(chunks)} 块")
    return chunks


# ================================================================
# 内部实现函数
# ================================================================

def _split_by_paragraphs(text: str) -> List[str]:
    """按空行（\n\n）或连续换行切分文本为段落"""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


def _split_by_sentences(text: str, max_chunk_size: int) -> List[str]:
    """
    将长段落按句子切分成更小的块。

    Args:
        text: 长段落文本
        max_chunk_size: 每块最大字数

    Returns:
        切分后的块列表，每块尽量接近 max_chunk_size 但不超过
    """
    # 按句子结束符切分，保留结束符
    sentences = re.split(f'({SENTENCE_ENDINGS})', text)

    chunks = []
    current = ""

    for i in range(0, len(sentences), 2):
        # 每个句子的内容是 sentences[i]，结束符是 sentences[i+1]
        sentence = sentences[i]
        if i + 1 < len(sentences):
            sentence += sentences[i + 1]

        # 如果当前块加上新句子超过 max_chunk_size，且当前块已有内容
        if len(current) + len(sentence) > max_chunk_size and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current += sentence

    # 最后一块
    if current.strip():
        chunks.append(current.strip())

    return chunks


def _apply_overlap(chunks: List[str], overlap: int) -> List[str]:
    """
    在相邻块之间应用重叠。

    Args:
        chunks: 原始块列表
        overlap: 重叠字数

    Returns:
        重叠处理后的块列表

    Note:
        重叠方式：从上一块的末尾取 overlap 个字，拼到下一块的开头。
        如果上一块本身就比 overlap 短，则取上一块的全部。
    """
    if overlap <= 0:
        return chunks

    result = [chunks[0]]

    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        curr = chunks[i]

        # 从上一块末尾取 overlap 个字
        overlap_text = prev[-overlap:] if len(prev) > overlap else prev

        # 拼到当前块开头
        merged = overlap_text + curr
        result.append(merged)

    return result


# ================================================================
# 兼容旧版：固定大小分块
# ================================================================

def chunk_text_fixed(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    固定大小分块（保留旧版，用于对比测试或迁移过渡）。

    策略：
        1. 按空行切段落
        2. 段落太长则按固定大小切分

    注意：
        这个函数在 2026 年已不推荐使用。
        推荐使用 chunk_text() 动态分块版本。
    """
    if not text or not text.strip():
        return []

    paragraphs = _split_by_paragraphs(text)
    if not paragraphs:
        return []

    chunks = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            # 按固定大小切分（不考虑句子边界）
            for i in range(0, len(para), chunk_size - overlap):
                chunk = para[i:i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk.strip())

    if overlap > 0 and len(chunks) > 1:
        chunks = _apply_overlap(chunks, overlap)

    chunks = [c.strip() for c in chunks if c.strip()]
    logger.debug(f"固定分块完成：原文 {len(text)} 字 → {len(chunks)} 块")
    return chunks


# ================================================================
# 便捷函数
# ================================================================

def estimate_chunk_count(text: str, max_chunk_size: int = 800) -> int:
    """
    估算文本会被切成多少块（用于进度显示）。

    Args:
        text: 原始文本
        max_chunk_size: 每块最大字数

    Returns:
        估算的块数
    """
    if not text:
        return 0
    # 粗略估算：总字数 / 块大小 * 1.2（考虑重叠）
    total_chars = len(text)
    estimated = int(total_chars / max_chunk_size * 1.2)
    return max(1, estimated)


# ================================================================
# 快速测试入口
# ================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("===== 动态分块测试 =====")

    test_text = """
    这是第一段内容。它包含一些中文文本，用于测试分块算法。

    这是第二段内容。它比第一段长一些。我们希望通过分块算法，在保持语义完整的同时，把长文本切成合适的块。
    这个段落比较长，可能会触发句子级别的切分。用于测试长段落的分块效果。
    
    这是第三段。每段由空行分隔，程序会自动识别。
    """

    print("原始文本长度:", len(test_text), "字符")
    print("-" * 60)

    # 测试动态分块
    chunks = chunk_text(test_text, min_chunk_size=50, max_chunk_size=100, overlap=10)

    print(f"\n动态分块结果：{len(chunks)} 块")
    for i, chunk in enumerate(chunks):
        print(f"  块 {i+1} (长度: {len(chunk)} 字): {chunk[:40]}...")

    # 对比固定分块
    print("\n" + "-" * 60)
    fixed_chunks = chunk_text_fixed(test_text, chunk_size=100, overlap=10)
    print(f"固定分块结果：{len(fixed_chunks)} 块")
    for i, chunk in enumerate(fixed_chunks):
        print(f"  块 {i+1} (长度: {len(chunk)} 字): {chunk[:40]}...")

    print("\n✅ 动态分块测试完成")