#!/usr/bin/env bash

# Cấu hình đường dẫn và biến môi trường
CONDA_ENV="/home/zeus/miniconda3/envs/cloudspace"
EXPORT_PATH="export PATH=\"$CONDA_ENV/bin:\$PATH\""
MODEL_NAME="/home/zeus/content/models/Qwen3.5-4B-AWQ-4bit"
PORT=8000
LOG_FILE="/home/zeus/content/vllm_server.log"

start() {
    echo "Checking if vLLM server is already running on port $PORT..."
    if python3 -c "import socket; s = socket.socket(); s.connect(('127.0.0.1', $PORT))" >/dev/null 2>&1; then
        echo "⚠️ Server is already running on port $PORT."
        exit 0
    fi

    echo "⚙️ Auto-detecting GPU configuration..."
    # Lấy thông tin GPU (số lượng và VRAM) dùng python
    eval "$EXPORT_PATH"
    NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
    VRAM_GB=$(python3 -c "import torch; print(round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1))" 2>/dev/null || echo "16.0")
    
    GPU_MEM_UTIL="0.80"
    MAX_MODEL_LEN=4096
    
    # Nếu VRAM nhỏ (e.g. <= 16.5GB như T4), dùng cấu hình an toàn
    if python3 -c "import sys; sys.exit(0 if float('$VRAM_GB') <= 16.5 else 1)"; then
        GPU_MEM_UTIL="0.75"
        MAX_MODEL_LEN=3072
    fi
    
    echo "🚀 Hardware Config: GPUs=$NUM_GPUS, VRAM=$VRAM_GB GB"
    echo "⚙️ Auto-configured Server parameters: --gpu-memory-utilization=$GPU_MEM_UTIL, --max-model-len=$MAX_MODEL_LEN, --tensor-parallel-size=$NUM_GPUS"

    echo "🚀 Starting vLLM API Server in background..."
    # Chạy lệnh khởi động vLLM
    eval "$EXPORT_PATH && nohup python -m vllm.entrypoints.openai.api_server \
        --model $MODEL_NAME \
        --port $PORT \
        --trust-remote-code \
        --served-model-name Qwen/Qwen3.5-4B \
        --enforce-eager \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --tensor-parallel-size $NUM_GPUS > $LOG_FILE 2>&1 &"

    echo "⏳ Server is starting. Checking logs..."
    sleep 5
    tail -n 15 $LOG_FILE
    echo "Run './manage_server.sh status' to monitor startup progress."
}

stop() {
    echo "Stopping vLLM server..."
    PID=$(pgrep -f "vllm.entrypoints.openai.api_server")
    if [ -n "$PID" ]; then
        kill $PID
        echo "✅ Killed vLLM process $PID."
    else
        echo "ℹ️ No vLLM server process found."
    fi
}

status() {
    # Check if port is open
    if python3 -c "import socket; s = socket.socket(); s.connect(('127.0.0.1', $PORT))" >/dev/null 2>&1; then
        PID=$(pgrep -f "vllm.entrypoints.openai.api_server")
        echo "🟢 vLLM Server is RUNNING (PID: $PID) on port $PORT."
    else
        PID_VLLM=$(pgrep -f "vllm.entrypoints.openai.api_server")
        if [ -n "$PID_VLLM" ]; then
            echo "🟡 vLLM Server is active (PID: $PID_VLLM) but port $PORT is not open/ready yet."
        else
            echo "🔴 vLLM Server is STOPPED."
        fi
    fi
    
    echo "--- Last 15 lines of log ($LOG_FILE) ---"
    if [ -f "$LOG_FILE" ]; then
        tail -n 15 $LOG_FILE
    else
        echo "Log file not found."
    fi
}

case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 {start|stop|status}"
        exit 1
        ;;
esac
