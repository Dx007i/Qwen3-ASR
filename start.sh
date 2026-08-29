#!/usr/bin/env bash
cd "$(dirname "$0")"

# 可选环境变量:ASR_DEVICE(默认 cuda:0)、PORT(默认 8003)、ASR_LOCK_PATH
: "${ASR_MODEL_DIR:?未设置 ASR_MODEL_DIR 环境变量,请指向 Qwen/Qwen3-ASR-1.7B-hf 模型目录(见 README.md)}"

echo "============================================"
echo "  Qwen3-ASR Service  port=${PORT:-8003}  device=${ASR_DEVICE:-cuda:0}"
echo "  模型: $ASR_MODEL_DIR"
echo "============================================"

exec python asr_server.py
