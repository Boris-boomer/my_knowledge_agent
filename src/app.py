# ================================================================
# app.py：Gradio 界面入口（适配 Gradio 5.x tuples 格式）
# ================================================================
import os
import sys
import logging
import tempfile
import shutil
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gradio as gr
import yaml

from src.build_kb import build_knowledge_base
from src.conversation import ConversationEngine

logger = logging.getLogger(__name__)


def update_knowledge_base_from_files(file_paths, engine: ConversationEngine) -> str:
    """上传文件并构建知识库"""
    if not file_paths:
        return "❌ 请至少选择一个文件"
    temp_dir = tempfile.mkdtemp()
    try:
        for path in file_paths:
            shutil.copy2(path, os.path.join(temp_dir, os.path.basename(path)))
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config["knowledge_base"]["paths"] = [temp_dir]
        with open("config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        stats = build_knowledge_base("config.yaml")
        engine.reload()
        return (
            f"✅ 知识库构建完成！\n"
            f"   上传文件数: {len(file_paths)}\n"
            f"   成功解析: {stats.get('parsed_files', 0)}\n"
            f"   存入向量库: {stats.get('stored_chunks', 0)} 块\n"
            f"   总耗时: {stats.get('duration_seconds', 0)} 秒"
        )
    except Exception as e:
        logger.error(f"构建失败: {e}")
        traceback.print_exc()
        return f"❌ 构建失败: {e}"
    finally:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


def clear_database(engine: ConversationEngine) -> str:
    try:
        engine.vector_store.clear()
        engine.reload()
        return "✅ 知识库已清空！"
    except Exception as e:
        return f"❌ 清空失败: {e}"


def create_ui(engine: ConversationEngine):
    """创建 Gradio 界面，使用 tuples 格式 (Gradio 5.x)"""

    # ---------- 响应函数（tuples 格式）----------
    # 增加 vectorless_enabled 参数
    async def respond_stream(message, chat_history, vectorless_enabled):
        # chat_history 是列表，每个元素为 [user, bot]
        chat_history.append([message, ""])
        yield chat_history

        full_response = ""
        try:
            # 把开关状态传给 answer_stream
            async for chunk in engine.answer_stream(message, use_vectorless=vectorless_enabled):
                full_response += chunk
                chat_history[-1][1] = full_response
                yield chat_history
        except Exception as e:
            logger.error(f"流式错误: {e}")
            traceback.print_exc()
            chat_history[-1][1] = f"❌ 生成错误: {e}"
            yield chat_history

    def update_kb_wrapper(files):
        return update_knowledge_base_from_files(files, engine)

    def clear_wrapper():
        return clear_database(engine)

    def clear_chat():
        engine.clear_history()
        return []

    with gr.Blocks(title="个人知识库问答 Agent") as demo:
        gr.Markdown("# 📚 个人知识库问答 Agent")
        with gr.Row():
            doc_count = engine.vector_store.count()
            reranker_status = "✅ 已启用" if engine.reranker.is_available() else "❌ 未启用"
            gr.Markdown(f"**知识库状态：** {doc_count} 条文档 | **Reranker：** {reranker_status}")

        with gr.Tab("💬 对话"):
            # 显式指定 type="tuples"
            chatbot = gr.Chatbot(label="对话", height=500)

            # ===== 新增 Vectorless RAG 开关 =====
            with gr.Row():
                vectorless_toggle = gr.Checkbox(
                    label="🔍 启用深度检索（Vectorless RAG）",
                    value=False,
                    info="开启后会在文档树中导航检索，更准但更慢。关闭后只使用向量检索，速度更快。"
                )

            with gr.Row():
                msg = gr.Textbox(label="输入问题", placeholder="例如：DSPy 是什么？", scale=4)
                submit_btn = gr.Button("发送", variant="primary", scale=1)
            clear_btn = gr.Button("清空对话", variant="secondary")
            gr.Examples(examples=["DSPy 是什么？", "如何安装 DSPy？", "学代码"], inputs=[msg])

            # 事件绑定：传入 vectorless_toggle 的值
            msg.submit(respond_stream, [msg, chatbot, vectorless_toggle], [chatbot])
            submit_btn.click(respond_stream, [msg, chatbot, vectorless_toggle], [chatbot])
            msg.submit(lambda: "", None, [msg])
            submit_btn.click(lambda: "", None, [msg])
            clear_btn.click(clear_chat, None, [chatbot])

        with gr.Tab("📤 上传文档"):
            gr.Markdown("### 上传文档构建知识库\n支持 .txt / .pdf / .docx，可多选。")
            with gr.Row():
                upload_btn = gr.File(
                    label="选择文档（可多选）",
                    file_count="multiple",
                    file_types=[".txt", ".pdf", ".docx"],
                    type="filepath",
                    interactive=True,
                )
                build_btn = gr.Button("🚀 导入并构建知识库", variant="primary")
            kb_status = gr.Textbox(label="构建状态", lines=8, interactive=False, placeholder="等待操作...")
            with gr.Row():
                clear_db_btn = gr.Button("🗑️ 清空知识库", variant="stop")
            build_btn.click(update_kb_wrapper, inputs=[upload_btn], outputs=[kb_status])
            clear_db_btn.click(clear_wrapper, outputs=[kb_status])

    return demo


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    engine = ConversationEngine("config.yaml")
    demo = create_ui(engine)

    # 队列控制，保证顺序
    demo.queue(default_concurrency_limit=1)

    print("\n🚀 启动服务 http://localhost:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()