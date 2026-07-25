# ================================================================
# build_kb.py：构建知识库主脚本（支持断点续传）
# ================================================================
#
# 【核心职责】
#   1. 扫描 config.yaml 中指定的文件夹路径
#   2. 对每个文件，检查是否已存在于向量库中（通过 source 字段）
#   3. 如果已存在，跳过；否则解析、分块、向量化、存入
#   4. 支持追加新文件，不会覆盖已有数据
#
# 【使用方式】
#   uv run python src/build_kb.py          # 增量构建（推荐）
#   uv run python src/build_kb.py --force  # 强制重建（清空所有数据）
#
# 【断点续传原理】
#   每次运行前，从向量库中读取所有已存在的文件路径（source 字段），
#   形成一个集合。处理文件时，如果该文件路径已经在集合中，则跳过。
#   这样即使之前处理到一半中断，重新运行后会跳过已完成的文件。
# ================================================================

import os
import sys
import logging
import time
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path

# 添加项目根目录到 Python 路径（如果直接运行此脚本）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config_loader import load_config
from src.parser import parse_file, is_supported_file
from src.chunker import chunk_text
from src.vector_store import get_vector_store

# 从 core 导入模型路由器（包含 embed 能力）
from core.llm_router import ModelRouter

logger = logging.getLogger(__name__)


# ================================================================
# 主函数
# ================================================================

def build_knowledge_base(
    config_path: str = "config.yaml",
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    """
    构建知识库：扫描文件夹 → 解析文档 → 分块 → 向量化 → 存入向量库

    Args:
        config_path: 配置文件路径
        force_rebuild: 是否强制重建（清空现有向量库）

    Returns:
        包含构建统计信息的字典
    """
    start_time = time.time()

    # 统计信息
    stats = {
        "total_files": 0,
        "parsed_files": 0,
        "total_chunks": 0,
        "stored_chunks": 0,
        "skipped_files": [],       # 已存在于向量库中，跳过的文件
        "failed_files": [],        # 处理失败的文件
        "duration_seconds": 0,
        "vector_store_count": 0,
    }

    # ----------------------------------------------------------------
    # 步骤 1：加载配置
    # ----------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("开始构建知识库")
    logger.info("=" * 60)

    try:
        config = load_config(config_path)
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        stats["duration_seconds"] = time.time() - start_time
        return stats

    # ----------------------------------------------------------------
    # 步骤 2：读取配置
    # ----------------------------------------------------------------
    kb_config = config.get("knowledge_base", {})
    root_paths = kb_config.get("paths", [])
    exclude_patterns = kb_config.get("exclude_patterns", [])
    recursive = kb_config.get("recursive", True)

    if not root_paths:
        logger.error("knowledge_base.paths 为空，请在 config.yaml 中配置文档路径")
        stats["duration_seconds"] = time.time() - start_time
        return stats

    # 分块参数（动态分块）
    chunker_config = config.get("chunker", {})
    min_chunk_size = chunker_config.get("min_chunk_size", 200)
    max_chunk_size = chunker_config.get("max_chunk_size", 800)
    overlap = chunker_config.get("overlap", 50)

    # 初始化 LLM 路由器（用于向量化）
    llm_config = config.get("llm", {})
    embedder_config = config.get("embedder", {})
    router = ModelRouter(
        llm_config=llm_config,
        embedder_config=embedder_config,
    )

    # 初始化向量库
    vector_store_config = config.get("vector_store", {})
    vector_store = get_vector_store(vector_store_config)

    # 如果强制重建，清空向量库
    if force_rebuild:
        try:
            vector_store.clear()
            logger.info("已清空向量库（强制重建模式）")
        except Exception as e:
            logger.warning(f"清空向量库失败: {e}")

    # ================================================================
    # 【新增】读取已有文件列表（断点续传）
    # ================================================================
    # 从向量库中读取所有已存在的文件路径，用于跳过已处理的文件
    # ================================================================
    existing_files = set()
    if not force_rebuild:
        try:
            # 获取向量库中所有元数据
            all_data = vector_store._collection.get(include=["metadatas"])
            if all_data and all_data.get("metadatas"):
                for meta in all_data["metadatas"]:
                    # 从 source 或 filename 字段获取文件路径
                    source = meta.get("source", "")
                    if source:
                        existing_files.add(source)
                    # 如果 source 是相对路径，尝试用 filename 补充
                    filename = meta.get("filename", "")
                    if filename and filename not in existing_files:
                        # 有些文件可能只存了 filename，没有完整路径
                        # 这种情况我们无法精确判断，存疑的文件不跳过
                        pass
                logger.info(f"📊 向量库中已有 {len(existing_files)} 个文件的记录，将跳过已处理文件")
            else:
                logger.info("📊 向量库为空，将处理所有文件")
        except Exception as e:
            logger.warning(f"读取已有文件列表失败: {e}，将处理所有文件")

    # ----------------------------------------------------------------
    # 步骤 3：扫描文件
    # ----------------------------------------------------------------
    logger.info(f"正在扫描文件夹: {root_paths}")
    all_files = []

    for root_path in root_paths:
        if not os.path.exists(root_path):
            logger.warning(f"路径不存在，已跳过: {root_path}")
            continue

        for root, dirs, files in os.walk(root_path):
            # 检查是否要排除当前目录
            should_exclude_dir = False
            for pattern in exclude_patterns:
                if pattern in root or pattern in dirs:
                    should_exclude_dir = True
                    break
            if should_exclude_dir:
                continue

            for file in files:
                full_path = os.path.join(root, file)

                # 检查是否匹配排除模式
                should_exclude = False
                for pattern in exclude_patterns:
                    if pattern in full_path or pattern in file:
                        should_exclude = True
                        break
                if should_exclude:
                    continue

                # 检查是否支持的文件类型
                if is_supported_file(full_path):
                    all_files.append(full_path)

    stats["total_files"] = len(all_files)
    logger.info(f"✅ 扫描完成，找到 {stats['total_files']} 个支持的文件")

    if stats["total_files"] == 0:
        logger.warning("没有找到可处理的文件，请检查：")
        logger.warning("  1. 文件夹路径是否正确")
        logger.warning("  2. 是否包含 .txt / .pdf / .docx 文件")
        stats["duration_seconds"] = time.time() - start_time
        return stats

    # ----------------------------------------------------------------
    # 步骤 4：逐个文件处理
    # ----------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("开始处理文件...")

    doc_id = vector_store.count()
    processed_files = 0

    for file_idx, file_path in enumerate(all_files, 1):
        filename = os.path.basename(file_path)
        logger.info(f"[{file_idx}/{stats['total_files']}] 处理: {filename}")

        # ================================================================
        # 【新增】检查是否已处理（断点续传核心逻辑）
        # ================================================================
        if not force_rebuild and file_path in existing_files:
            logger.info(f"  ⏭️ 已存在于向量库中，跳过")
            stats["skipped_files"].append(filename)
            continue

        # 如果文件路径不在 existing_files 中，但可能之前处理时用的是相对路径
        # 尝试用文件名模糊匹配（更保守的做法，避免误跳过）
        # 这里暂不实现模糊匹配，避免误判

        # ----------------------------------------------------------------
        # 4.1 解析文件
        # ----------------------------------------------------------------
        content, error = parse_file(file_path)
        if error:
            logger.warning(f"  跳过: {error}")
            stats["failed_files"].append(filename)
            continue

        if not content or len(content.strip()) < 10:
            logger.warning("  跳过: 内容为空或少于 10 个字符")
            stats["failed_files"].append(filename)
            continue

        # ----------------------------------------------------------------
        # 4.2 动态分块
        # ----------------------------------------------------------------
        chunks = chunk_text(
            content,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            overlap=overlap,
        )

        if not chunks:
            logger.warning("  跳过: 分块后为空")
            stats["failed_files"].append(filename)
            continue

        logger.info(f"  分块: {len(chunks)} 块")

        # ----------------------------------------------------------------
        # 4.3 生成向量并存入向量库
        # ----------------------------------------------------------------
        batch_ids = []
        batch_embeddings = []
        batch_documents = []
        batch_metadatas = []

        for chunk_idx, chunk in enumerate(chunks):
            try:
                # 使用 ModelRouter.embed() 生成向量
                embedding = router.embed(chunk)

                if not embedding:
                    logger.warning(f"  块 {chunk_idx + 1} 向量生成失败，跳过")
                    continue

                doc_id += 1
                batch_ids.append(f"doc_{doc_id}")
                batch_embeddings.append(embedding)
                batch_documents.append(chunk)
                batch_metadatas.append({
                    "source": file_path,
                    "filename": filename,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                })

            except Exception as e:
                logger.warning(f"  块 {chunk_idx + 1} 处理失败: {e}")
                continue

        # 批量写入
        if batch_ids:
            try:
                vector_store.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                )
                stored_count = len(batch_ids)
                stats["stored_chunks"] += stored_count
                stats["total_chunks"] += len(chunks)
                processed_files += 1
                # 更新已处理文件集合（实时更新，避免同一文件重复处理）
                existing_files.add(file_path)
                logger.info(f"  ✅ 已存入 {stored_count} 块")
            except Exception as e:
                logger.error(f"  存入失败: {e}")
                stats["failed_files"].append(filename)
        else:
            stats["failed_files"].append(filename)

    stats["parsed_files"] = processed_files
    stats["vector_store_count"] = vector_store.count()

    # ----------------------------------------------------------------
    # 步骤 5：完成统计
    # ----------------------------------------------------------------
    stats["duration_seconds"] = round(time.time() - start_time, 2)

    logger.info("=" * 60)
    logger.info("🎉 知识库构建完成！")
    logger.info(f"  总文件数: {stats['total_files']}")
    logger.info(f"  成功解析: {stats['parsed_files']}")
    logger.info(f"  跳过文件: {len(stats['skipped_files'])} (已存在于向量库)")
    logger.info(f"  失败文件: {len(stats['failed_files'])}")
    logger.info(f"  总块数: {stats['total_chunks']}")
    logger.info(f"  存入向量库: {stats['stored_chunks']} 块")
    logger.info(f"  向量库总记录: {vector_store.count()}")
    logger.info(f"  总耗时: {stats['duration_seconds']} 秒")
    logger.info("=" * 60)

    if stats["failed_files"] and len(stats["failed_files"]) <= 20:
        logger.info("失败文件列表:")
        for f in stats["failed_files"]:
            logger.info(f"  - {f}")
    elif stats["failed_files"]:
        logger.info(f"失败文件列表: {len(stats['failed_files'])} 个文件（太多，请查看日志）")

    return stats


# ================================================================
# 命令行入口
# ================================================================

def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="构建知识库 - 扫描、解析、分块、向量化文档（支持断点续传）"
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config.yaml",
        help="配置文件路径（默认: config.yaml）",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="强制重建（清空现有向量库，从头开始）",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="显示详细日志（DEBUG 级别）",
    )

    args = parser.parse_args()

    # 配置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    # 执行构建
    stats = build_knowledge_base(args.config, args.force)

    # 退出码：如果有文件失败，返回 1
    if stats.get("failed_files") and len(stats["failed_files"]) > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()