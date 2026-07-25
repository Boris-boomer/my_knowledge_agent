# ================================================================
# parser.py：文档解析器
# ================================================================
#
# 这个模块负责把各种格式的文档转换成纯文本。
# 支持格式：.txt、.pdf、.docx
#
# 【核心职责】
#   1. 根据文件扩展名自动选择合适的解析器
#   2. 提取文档中的文本内容
#   3. 解析失败时返回明确的错误信息，不崩溃
#
# 【对外暴露】
#   parse_file(file_path: str) -> Tuple[str, str]
#     返回: (文件内容, 错误信息) ｜ 成功时错误信息为空字符串
#
# 【使用示例】
#   from parser import parse_file
#   content, error = parse_file("D:/文档/报告.pdf")
#   if error:
#       print(f"解析失败: {error}")
#   else:
#       print(content)
# ================================================================

import os
import logging
from pathlib import Path
from typing import Tuple

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def parse_file(file_path: str) -> Tuple[str, str]:
    """
    解析文档文件，提取纯文本内容。

    Args:
        file_path: 文档的完整路径

    Returns:
        (content, error_message)
        - 成功: (文本内容, "")
        - 失败: ("", "错误描述")
    """
    if not os.path.exists(file_path):
        error_msg = f"文件不存在: {file_path}"
        logger.error(error_msg)
        return "", error_msg

    ext = Path(file_path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        error_msg = f"不支持的文件格式: {ext}，支持的格式: {', '.join(SUPPORTED_EXTENSIONS)}"
        logger.warning(error_msg)
        return "", error_msg

    try:
        if ext == ".pdf":
            content, error = _parse_with_docling(file_path, InputFormat.PDF)
        elif ext == ".docx":
            content, error = _parse_with_docling(file_path, InputFormat.DOCX)
        elif ext == ".txt":
            content, error = _parse_txt(file_path)
        else:
            return "", f"未处理的格式: {ext}"

        if error:
            return "", error

        if not content or not content.strip():
            logger.warning(f"文件解析后内容为空: {file_path}")
            return "", ""

        return content, ""

    except Exception as e:
        error_msg = f"解析失败: {str(e)}"
        logger.error(f"{error_msg} (文件: {file_path})")
        return "", error_msg


def _parse_with_docling(file_path: str, input_format: InputFormat) -> Tuple[str, str]:
    """使用 docling 解析 PDF 或 DOCX 文件"""
    try:
        converter = DocumentConverter()

        if input_format == InputFormat.PDF:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True
            converter = DocumentConverter(pipeline_options=pipeline_options)

        result = converter.convert(file_path)
        content = result.document.export_to_text()
        return content, ""

    except Exception as e:
        error_msg = f"docling 解析失败: {str(e)}"
        logger.error(error_msg)
        return "", error_msg


def _parse_txt(file_path: str) -> Tuple[str, str]:
    """解析 TXT 文本文件（自动检测编码）"""
    encodings = ["utf-8", "gbk", "gb2312", "latin-1"]

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read(), ""
        except UnicodeDecodeError:
            continue
        except Exception as e:
            error_msg = f"读取文件失败: {str(e)}"
            logger.error(error_msg)
            return "", error_msg

    error_msg = f"无法解码文件: {file_path}，尝试过的编码: {', '.join(encodings)}"
    logger.error(error_msg)
    return "", error_msg


def is_supported_file(file_path: str) -> bool:
    """检查文件是否为支持的类型"""
    return Path(file_path).suffix.lower() in SUPPORTED_EXTENSIONS


def get_supported_extensions() -> list:
    """返回支持的扩展名列表"""
    return list(SUPPORTED_EXTENSIONS)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys

    if len(sys.argv) < 2:
        print("用法: python parser.py <文件路径>")
        sys.exit(0)

    file_path = sys.argv[1]
    content, error = parse_file(file_path)

    if error:
        print(f"❌ 解析失败: {error}")
    else:
        print(f"✅ 解析成功，共 {len(content)} 个字符")
        print("-" * 60)
        print(content[:500])
        if len(content) > 500:
            print("...(内容过长，已截断)")