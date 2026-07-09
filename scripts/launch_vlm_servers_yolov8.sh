#!/usr/bin/env bash
# Copyright [2023] Boston Dynamics AI Institute, Inc.
#
# 基于 launch_vlm_servers.sh，将 YOLOv7 替换为 YOLOv8。
# 去掉了 GroundingDINO（精简版不需要），如需要可取消注释。

export VLFM_PYTHON=${VLFM_PYTHON:-`which python`}
export MOBILE_SAM_CHECKPOINT=${MOBILE_SAM_CHECKPOINT:-data/mobile_sam.pt}
export BLIP2ITM_PORT=${BLIP2ITM_PORT:-12182}
export SAM_PORT=${SAM_PORT:-12183}
export YOLOV8_PORT=${YOLOV8_PORT:-12186}
export YOLOV8_WEIGHTS=${YOLOV8_WEIGHTS:-yolov8x.pt}

session_name=vlm_servers_yolov8_${RANDOM}

# Create a detached tmux session
tmux new-session -d -s ${session_name}

# Split the window: 3 panes for 3 servers (YOLOv8 + SAM + BLIP2ITM)
tmux split-window -v -t ${session_name}:0

# Split the top pane horizontally
tmux split-window -h -t ${session_name}:0.0

# Run commands in each pane
tmux send-keys -t ${session_name}:0.0 "${VLFM_PYTHON} -m myon.vlm.yolov8 --port ${YOLOV8_PORT} --weights ${YOLOV8_WEIGHTS}" C-m
tmux send-keys -t ${session_name}:0.1 "${VLFM_PYTHON} -m vlfm.vlm.blip2itm --port ${BLIP2ITM_PORT}" C-m
tmux send-keys -t ${session_name}:0.2 "${VLFM_PYTHON} -m vlfm.vlm.sam --port ${SAM_PORT}" C-m

echo "Created tmux session '${session_name}'. 3 servers launched:"
echo "  [0.0] YOLOv8       (port ${YOLOV8_PORT})"
echo "  [0.1] BLIP2ITM     (port ${BLIP2ITM_PORT})"
echo "  [0.2] MobileSAM    (port ${SAM_PORT})"
echo ""
echo "Wait ~90 seconds for models to load, then monitor with:"
echo "  tmux attach-session -t ${session_name}"
