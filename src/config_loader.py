# ================================================================
# config_loader.py：配置加载器
# ================================================================
#
# 这个模块只做一件事：读取并校验 config.yaml 文件。
#
# 【核心职责】
#   1. 读取 config.yaml 文件内容，转换成 Python 字典
#   2. 校验用户填写的路径是否存在
#   3. 如果配置有误，打印友好的错误提示并退出
#   4. 如果配置正确，返回配置字典供其他模块使用
#
# 【对外暴露】
#   load_config() -> dict
#
# 【使用示例】
#   from config_loader import load_config
#   config = load_config()
#   print(config["knowledge_base"]["paths"])
# ================================================================

import os
import sys
import logging
from typing import List, Dict, Any
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    加载并校验 config.yaml 配置文件。

    Args:
        config_path: 配置文件路径，默认当前目录下的 config.yaml

    Returns:
        校验通过后的配置字典

    Raises:
        SystemExit: 如果配置文件不存在、格式错误或校验失败，打印错误后退出
    """
    # ----------------------------------------------------------------
    # 步骤 1：检查配置文件是否存在
    # ----------------------------------------------------------------
    if not os.path.exists(config_path):
        logger.error(f"找不到配置文件: {config_path}")
        logger.error(f"请确保 {config_path} 文件存在于当前目录。")
        logger.error(f"当前工作目录: {os.getcwd()}")
        sys.exit(1)

    # ----------------------------------------------------------------
    # 步骤 2：读取 YAML 文件内容
    # ----------------------------------------------------------------
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"配置文件格式错误: {e}")
        logger.error("请检查 config.yaml 的语法是否正确。")
        logger.error("常见问题：缩进使用了 Tab（应使用空格）、漏了冒号、引号不匹配。")
        sys.exit(1)

    if config is None:
        logger.error("配置文件为空，请检查 config.yaml 是否有内容。")
        sys.exit(1)

    # ----------------------------------------------------------------
    # 步骤 3：校验核心配置项是否存在
    # ----------------------------------------------------------------
    knowledge_base = config.get("knowledge_base")
    if not knowledge_base:
        logger.error("config.yaml 中缺少 'knowledge_base' 配置项。")
        logger.error("请参考示例配置文件，添加知识库路径配置。")
        sys.exit(1)

    paths = knowledge_base.get("paths")
    if not paths:
        logger.error("config.yaml 中 'knowledge_base.paths' 为空。")
        logger.error("请至少填写一个文件夹路径，例如：")
        logger.error('  paths:')
        logger.error('    - "D:/MyDocuments"')
        sys.exit(1)

    if isinstance(paths, str):
        config["knowledge_base"]["paths"] = [paths]
        logger.info("已将 knowledge_base.paths 从字符串自动转换为列表")

    # ----------------------------------------------------------------
    # 步骤 4：设置默认值
    # ----------------------------------------------------------------
    config.setdefault("app", {})
    config["app"].setdefault("log_level", "INFO")
    config["app"].setdefault("validate_paths_on_start", True)

    config.setdefault("llm", {})
    config["llm"].setdefault("enabled", True)
    config["llm"].setdefault("backend", "ollama")
    config["llm"].setdefault("params", {})
    config["llm"]["params"].setdefault("model", "deepseek-r1:14b")
    config["llm"]["params"].setdefault("base_url", "http://localhost:11434")
    config["llm"]["params"].setdefault("temperature", 0.3)
    config["llm"]["params"].setdefault("max_tokens", 512)
    config["llm"]["params"].setdefault("timeout", 60)

    config.setdefault("embedder", {})
    config["embedder"].setdefault("enabled", True)
    config["embedder"].setdefault("backend", "ollama")
    config["embedder"].setdefault("params", {})
    config["embedder"]["params"].setdefault("model", "qwen3-embedding:0.6b")
    config["embedder"]["params"].setdefault("base_url", "http://localhost:11434")

    config.setdefault("vector_store", {})
    config["vector_store"].setdefault("enabled", True)
    config["vector_store"].setdefault("backend", "chromadb")
    config["vector_store"].setdefault("params", {})
    config["vector_store"]["params"].setdefault("path", "./chroma_db")
    config["vector_store"]["params"].setdefault("collection_name", "docs")

    config.setdefault("chunker", {})
    config["chunker"].setdefault("chunk_size", 500)
    config["chunker"].setdefault("overlap", 50)

    config.setdefault("agent", {})
    config["agent"].setdefault("enabled", True)
    config["agent"].setdefault("max_steps", 5)
    config["agent"].setdefault("stream", True)
    config["agent"].setdefault("top_k", 3)

    config.setdefault("cache", {})
    config["cache"].setdefault("enabled", False)

    config.setdefault("observability", {})
    config["observability"].setdefault("enabled", True)

    # ----------------------------------------------------------------
    # 步骤 5：路径校验
    # ----------------------------------------------------------------
    if config["app"].get("validate_paths_on_start", True):
        invalid_paths = _validate_paths(config["knowledge_base"]["paths"])
        if invalid_paths:
            logger.warning("以下路径不存在，将跳过：")
            for p in invalid_paths:
                logger.warning(f"  - {p}")

    _setup_logging(config["app"].get("log_level", "INFO"))
    return config


def _validate_paths(paths: List[str]) -> List[str]:
    """校验路径列表，返回不存在的路径列表"""
    return [p for p in paths if not os.path.exists(p)]


def _setup_logging(level: str) -> None:
    """根据配置设置日志级别"""
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    log_level = log_level_map.get(level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


if __name__ == "__main__":
    config = load_config()
    print("\n✅ 配置加载成功！")
    print(f"   知识库路径: {config['knowledge_base']['paths']}")
    print(f"   LLM 后端: {config['llm']['backend']}")
    print(f"   Embedder 后端: {config['embedder']['backend']}")