#!/usr/bin/env bash
# Copyright [2023] Boston Dynamics AI Institute, Inc.
#
# 启动 YOLOv8 + Qwen3-VL (transformers/Flask) + MobileSAM 三个服务。
#
# Qwen3-VL 替代了原有的 BLIP2 (VQA) 和 BLIP2ITM (image-text matching)，
# 使用 transformers 直接加载 + Flask 提供服务（与 BLIP2/ITM 相同架构）。
#
# 显存策略（RTX 4090 24GB 单卡）:
#   - Qwen3-VL-8B 先启动，占 ~16GB (bf16)
#   - YOLOv8x 后启动，占 ~400MB
#   - MobileSAM 后启动，占 ~200MB
#   - 总计约 16.6GB / 24GB

# ---- 环境变量 ----

# Python 解释器（用于 YOLOv8 和 MobileSAM，vlfm 环境）
export VLFM_PYTHON=${VLFM_PYTHON:-`which python`}

# Python 解释器（用于 Qwen3-VL，vllm_service 环境提供新版 torch/transformers）
export QWEN_PYTHON=${QWEN_PYTHON:-/root/anaconda3/envs/vllm_service/bin/python}

# Qwen3-VL 模型路径
export QWEN_MODEL_PATH=${QWEN_MODEL_PATH:-/root/objnav/vlfm/qwen/qwen3-vl-8B}

# MobileSAM 模型权重
export MOBILE_SAM_CHECKPOINT=${MOBILE_SAM_CHECKPOINT:-data/mobile_sam.pt}

# YOLOv8 权重
export YOLOV8_WEIGHTS=${YOLOV8_WEIGHTS:-yolov8x.pt}

# ---- 端口配置 ----
export YOLOV8_PORT=${YOLOV8_PORT:-12186}
export QWEN_PORT=${QWEN_PORT:-12182}      # Qwen3-VL 替代 BLIP2ITM (原端口 12182)
export SAM_PORT=${SAM_PORT:-12183}

session_name=vlm_servers_qwen_${RANDOM}

echo "============================================"
echo "  Starting VLM Servers (Qwen3-VL edition)"
echo "============================================"
echo ""
echo "Session:  ${session_name}"
echo ""
echo "Services (Qwen starts first for GPU memory):"
echo "  1. Qwen3-VL    port ${QWEN_PORT}  (transformers/Flask)"
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

if [ ! -f "${QWEN_PYTHON}" ]; then
    echo "ERROR: QWEN_PYTHON not found at ${QWEN_PYTHON}"
    echo "  Ensure the vllm_service conda environment exists with transformers installed."
    exit 1
fi

for service_port in "${QWEN_PORT}" "${YOLOV8_PORT}" "${SAM_PORT}"; do
    if ss -lnt | grep -q ":${service_port} "; then
        echo "ERROR: port ${service_port} is already in use. Stop the old VLM server session first."
        exit 1
    fi
done

# ---- 创建 tmux 会话 ----
tmux new-session -d -s ${session_name}

# Split the window vertically (creates pane 0.1)
tmux split-window -v -t ${session_name}:0

# Split the top pane horizontally (creates pane 0.1, old 0.1 becomes 0.2)
tmux split-window -h -t ${session_name}:0.0

# ---- 启动服务（Qwen 先启动以抢占显存） ----

# Pane 0.0: Qwen3-VL via transformers/Flask (先启动，占 ~16GB)
tmux send-keys -t ${session_name}:0.0 \
    "echo '=== Qwen3-VL (port ${QWEN_PORT}) ===' && ${QWEN_PYTHON} -m vlfm.vlm.qwen_vl --port ${QWEN_PORT} --model-path ${QWEN_MODEL_PATH}" C-m

# Pane 0.1: YOLOv8 (Qwen 健康检查通过后再启动，避免显存争抢)
tmux send-keys -t ${session_name}:0.1 \
    "echo 'Waiting for Qwen3-VL...' && until curl -fsS http://localhost:${QWEN_PORT}/qwen_vl/health >/dev/null; do sleep 5; done; echo '=== YOLOv8 (port ${YOLOV8_PORT}) ===' && ${VLFM_PYTHON} -m myon.vlm.yolov8 --port ${YOLOV8_PORT} --weights ${YOLOV8_WEIGHTS}" C-m

# Pane 0.2: MobileSAM (同样等待 Qwen 就绪)
tmux send-keys -t ${session_name}:0.2 \
    "echo 'Waiting for Qwen3-VL...' && until curl -fsS http://localhost:${QWEN_PORT}/qwen_vl/health >/dev/null; do sleep 5; done; echo '=== MobileSAM (port ${SAM_PORT}) ===' && ${VLFM_PYTHON} -m vlfm.vlm.sam --port ${SAM_PORT}" C-m

echo ""
echo "Created tmux session '${session_name}'. 3 servers launching:"
echo "  [0.0] Qwen3-VL     (port ${QWEN_PORT}) -- loading first"
echo "  [0.1] YOLOv8       (port ${YOLOV8_PORT}) -- waits for Qwen health check"
echo "  [0.2] MobileSAM    (port ${SAM_PORT}) -- waits for Qwen health check"
echo ""
echo "Monitor startup with:"
echo "  tmux attach-session -t ${session_name}"
echo ""
echo "To kill all servers:"
echo "  tmux kill-session -t ${session_name}"
