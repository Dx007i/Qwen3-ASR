@echo off
cd /d "%~dp0"
set PYTHONUTF8=1

REM 可选环境变量(也可在系统中永久设置):
REM   set ASR_MODEL_DIR=E:\models\Qwen3-ASR-1.7B-hf
REM   set ASR_DEVICE=cuda:0
REM   set PORT=8003

if "%ASR_MODEL_DIR%"=="" (
    echo [ERROR] 未设置 ASR_MODEL_DIR 环境变量,请指向 Qwen/Qwen3-ASR-1.7B-hf 模型目录
    echo         模型下载方式见 README.md
    pause
    exit /b 1
)

echo ============================================
echo   Qwen3-ASR Service  port=%PORT%  device=%ASR_DEVICE%
echo   模型: %ASR_MODEL_DIR%
echo ============================================
echo.

python asr_server.py
pause
