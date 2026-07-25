@echo off
chcp 65001 >nul
title my_knowledge_agent

echo ============================================================
echo       个人知识库问答 Agent - 启动脚本 (Windows)
echo ============================================================
echo.

REM ============================================================
REM 1. 切换到项目根目录
REM ============================================================
cd /d D:\ai_hands_on\idea\my_knowledge_agent
if errorlevel 1 (
    echo ❌ 无法切换到项目目录: D:\ai_hands_on\idea\my_knowledge_agent
    pause
    exit /b 1
)
echo ✅ 已切换到项目目录

REM ============================================================
REM 2. 设置 Hugging Face 镜像（必须，解决国内网络问题）
REM ============================================================
set HF_ENDPOINT=https://hf-mirror.com
echo ✅ 已设置 Hugging Face 镜像: %HF_ENDPOINT%

REM ============================================================
REM 3. 检查 uv 是否存在
REM ============================================================
echo [1/4] 检查 uv 包管理器...
uv --version >nul 2>&1
if errorlevel 1 (
    echo ⚠️ uv 未安装，正在安装...
    pip install uv
    if errorlevel 1 (
        echo ❌ uv 安装失败，请手动安装: pip install uv
        pause
        exit /b 1
    )
)
echo ✅ uv 已就绪

REM ============================================================
REM 4. 检查并创建虚拟环境
REM ============================================================
echo [2/4] 检查虚拟环境...
if not exist ".venv" (
    echo 创建虚拟环境...
    uv venv
)
echo ✅ 虚拟环境已就绪

REM ============================================================
REM 5. 安装依赖（尽量静默，失败时显示详情）
REM ============================================================
echo [3/4] 安装依赖...
uv pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ⚠️ 依赖安装失败，尝试详细模式...
    uv pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ 依赖安装失败，请检查网络或手动安装
        pause
        exit /b 1
    )
)
echo ✅ 依赖已就绪

REM ============================================================
REM 6. 启动服务
REM ============================================================
echo [4/4] 启动服务...
echo.
echo ============================================================
echo    🚀 服务启动中...
echo    访问地址: http://localhost:7860
echo    按 Ctrl+C 停止服务
echo ============================================================
echo.

REM 使用 uv run python 启动，无需手动激活虚拟环境
uv run python src/app.py

pause