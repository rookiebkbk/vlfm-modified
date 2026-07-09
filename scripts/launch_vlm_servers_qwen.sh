#!/usr/bin/env bash
# Copyright [2023] Boston Dynamics AI Institute, Inc.
#
# 启动 YOLOv8 + Qwen3-VL (vLLM) + MobileSAM 三个服务。
#
# Qwen3-VL 替代了原有的 BLIP2 (VQA) 和 BLIP2ITM (image-text matching)，
# 使用 vLLM 的 OpenAI 兼容 API 提供服务。
#
# 显存策略（RTX 4090 24GB 单卡）:
#   - Qwen3-VL-8B 先启动，占 ~16GB 模型权重 + ~3GB KV cache
#   - YOLOv8x 后启动，占 ~400MB
#   - MobileSAM 后启动，占 ~200MB
#   - 总计约 19.6GB / 24GB
#
# 前置条件:
#   conda create -n vllm_service python=3.10 -y
#   conda activate vllm_service
#   pip install vllm qwen-vl-utils modelscope
#   modelscope download --model Qwen/Qwen3-VL-8B-Instruct --local_dir /root/objnav/vlfm/qwen

# ---- 环境变量 ----

# Python 解释器（用于 YOLOv8 和 MobileSAM）
export VLFM_PYTHON=${VLFM_PYTHON:-`which python`}

# vLLM 可执行文件路径（vllm_service 环境）
export VLLM_EXEC=${VLLM_EXEC:-/root/anaconda3/envs/vllm_service/bin/vllm}

# Qwen3-VL 模型路径
export QWEN_MODEL_PATH=${QWEN_MODEL_PATH:-/root/objnav/vlfm/qwen}

# MobileSAM 模型权重
export MOBILE_SAM_CHECKPOINT=${MOBILE_SAM_CHECKPOINT:-data/mobile_sam.pt}

# YOLOv8 权重
export YOLOV8_WEIGHTS=${YOLOV8_WEIGHTS:-yolov8x.pt}

# ---- 端口配置 ----
export YOLOV8_PORT=${YOLOV8_PORT:-12186}
export QWEN_PORT=${QWEN_PORT:-12182}      # Qwen3-VL 替代 BLIP2ITM (原端口 12182)
export SAM_PORT=${SAM_PORT:-12183}

# ---- GPU 配置 ----
# vLLM 的 GPU 显存使用比例。RTX 4090 24GB: 0.80 = 19.2GB 给 vLLM
export QWEN_GPU_MEMORY=${QWEN_GPU_MEMORY:-0.80}

session_name=vlm_servers_qwen_${RANDOM}

echo "============================================"
echo "  Starting VLM Servers (Qwen3-VL edition)"
echo "============================================"
echo ""
echo "Session:  ${session_name}"
echo ""
echo "Services (start order for GPU memory):"
echo "  1. Qwen3-VL    port ${QWEN_PORT}  (vLLM, GPU mem: ${QWEN_GPU_MEMORY})"
echo "  2. YOLOv8      port ${YOLOV8_PORT}"
echo "  3. MobileSAM   port ${SAM_PORT}"
echo ""

# ---- 检查前置条件 ----
if [ ! -f "${MOBILE_SAM_CHECKPOINT}" ]; then
    echo "WARNING: MobileSAM checkpoint not found at ${MOBILE_SAM_CHECKPOINT}"
fi

if [ ! -f "${YOLOV8_WEIGHTS}" ]; then
    echo "WARNING: YOLOv8 weights not found at ${YOLOV8_WEIGHTS}"
fi

if [ ! -d "${QWEN_MODEL_PATH}" ]; then
    echo "ERROR: Qwen3-VL model not found at ${QWEN_MODEL_PATH}"
    echo "  Download it first:"
    echo "  modelscope download --model Qwen/Qwen3-VL-8B-Instruct --local_dir ${QWEN_MODEL_PATH}"
    exit 1
fi

if [ ! -f "${VLLM_EXEC}" ]; then
    echo "ERROR: vLLM not found at ${VLLM_EXEC}"
    echo "  Install it first:"
    echo "  conda create -n vllm_service python=3.10 -y"
    echo "  conda activate vllm_service"
    echo "  pip install vllm qwen-vl-utils"
    exit 1
fi

# ---- 创建 tmux 会话 ----
tmux new-session -d -s ${session_name}

# Split the window vertically (creates pane 0.1)
tmux split-window -v -t ${session_name}:0

# Split the top pane horizontally (creates pane 0.1, old 0.1 becomes 0.2)
tmux split-window -h -t ${session_name}:0.0

# ---- 启动服务（Qwen 先启动以抢占显存） ----

# Pane 0.0: Qwen3-VL via vLLM (先启动，占 ~16GB 模型 + ~3GB KV cache)
tmux send-keys -t ${session_name}:0.0 \
    "echo '=== Qwen3-VL via vLLM (port ${QWEN_PORT}) ===' && ${VLLM_EXEC} serve ${QWEN_MODEL_PATH} --port ${QWEN_PORT} --gpu-memory-utilization ${QWEN_GPU_MEMORY} --max-model-len 4096 --enforce-eager" C-m

# Pane 0.1: YOLOv8 (等 Qwen 加载后再启动)
#          vLLM 加载模型需要 ~30s，加 60s 延迟确保 vLLM 先占稳显存
tmux send-keys -t ${session_name}:0.1 \
    "echo 'Waiting 60s for vLLM to load Qwen3-VL first...' && sleep 60 && echo '=== YOLOv8 (port ${YOLOV8_PORT}) ===' && ${VLFM_PYTHON} -m myon.vlm.yolov8 --port ${YOLOV8_PORT} --weights ${YOLOV8_WEIGHTS}" C-m

# Pane 0.2: MobileSAM (同样延迟启动)
tmux send-keys -t ${session_name}:0.2 \
    "echo 'Waiting 60s for vLLM to load Qwen3-VL first...' && sleep 60 && echo '=== MobileSAM (port ${SAM_PORT}) ===' && ${VLFM_PYTHON} -m vlfm.vlm.sam --port ${SAM_PORT}" C-m

echo ""
echo "Created tmux session '${session_name}'. 3 servers launching:"
echo "  [0.0] Qwen3-VL     (port ${QWEN_PORT}) -- loading first (~30s)"
echo "  [0.1] YOLOv8       (port ${YOLOV8_PORT}) -- starts after 60s delay"
echo "  [0.2] MobileSAM    (port ${SAM_PORT}) -- starts after 60s delay"
echo ""
echo "Total startup: ~90 seconds. Monitor with:"
echo "  tmux attach-session -t ${session_name}"
echo ""
echo "To kill all servers:"
echo "  tmux kill-session -t ${session_name}"
