#!/usr/bin/env bash
# Copyright [2023] Boston Dynamics AI Institute, Inc.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

default_vlfm_python=/root/anaconda3/envs/vlfm/bin/python
if [[ ! -x "${default_vlfm_python}" ]]; then
    default_vlfm_python=$(which python)
fi
VLFM_PYTHON=${VLFM_PYTHON:-${default_vlfm_python}}
QWEN_PYTHON=${QWEN_PYTHON:-/root/anaconda3/envs/vllm_service/bin/python}
MOBILE_SAM_CHECKPOINT=${MOBILE_SAM_CHECKPOINT:-${REPO_ROOT}/data/mobile_sam.pt}
YOLOV8_WEIGHTS=${YOLOV8_WEIGHTS:-${REPO_ROOT}/yolov8x.pt}

BLIP2ITM_PORT=${BLIP2ITM_PORT:-12182}
SAM_PORT=${SAM_PORT:-12183}
QWEN_VQA_PORT=${QWEN_VQA_PORT:-12184}
YOLOV8_PORT=${YOLOV8_PORT:-12186}

qwen_model=${QWEN_VQA_MODEL:-${QWEN_MODEL_PATH:-2b}}
session_name=${VLM_SESSION_NAME:-vlm_servers_itm_vqa_${RANDOM}}
config_path=${VLM_CONFIG_PATH:-${REPO_ROOT}/config/experiments/myon_objectnav_hm3d.yaml}
dry_run=false

usage() {
    cat <<'EOF'
Usage: launch_vlm_servers_itm_vqa.sh [options]

Options:
  --qwen-model MODEL   2b, 8b, or a local checkpoint path (default: 2b)
  --qwen-python PATH   Python executable containing transformers and torch
  --config PATH        Experiment YAML used to determine whether VQA is enabled
  --session-name NAME  tmux session name
  --dry-run            Validate and print resolved settings without launching
  -h, --help           Show this help

Ports are configured with environment variables, not Hydra YAML:
  BLIP2ITM_PORT=12182 SAM_PORT=12183 QWEN_VQA_PORT=12184 YOLOV8_PORT=12186
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --qwen-model)
            qwen_model=$2
            shift 2
            ;;
        --qwen-python)
            QWEN_PYTHON=$2
            shift 2
            ;;
        --config)
            config_path=$2
            shift 2
            ;;
        --session-name)
            session_name=$2
            shift 2
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1"
            usage
            exit 2
            ;;
    esac
done

case "${qwen_model,,}" in
    2b|qwen3.5-2b|qwen3_5-2b)
        QWEN_MODEL_PATH=${REPO_ROOT}/qwen/qwen3.5-2B
        ;;
    8b|qwen3-vl-8b|qwen3_vl-8b)
        QWEN_MODEL_PATH=${REPO_ROOT}/qwen/qwen3-vl-8B
        ;;
    *)
        if [[ "${qwen_model}" = /* ]]; then
            QWEN_MODEL_PATH=${qwen_model}
        else
            QWEN_MODEL_PATH=${REPO_ROOT}/${qwen_model}
        fi
        ;;
esac

if [[ "${config_path}" != /* ]]; then
    config_path=${REPO_ROOT}/${config_path}
fi

for required_file in \
    "${VLFM_PYTHON}" \
    "${MOBILE_SAM_CHECKPOINT}" \
    "${YOLOV8_WEIGHTS}" \
    "${config_path}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "ERROR: required file not found: ${required_file}"
        exit 1
    fi
done

use_vqa=$("${VLFM_PYTHON}" -c '
import sys
from omegaconf import OmegaConf

config = OmegaConf.load(sys.argv[1])
enabled = OmegaConf.select(
    config,
    "habitat_baselines.rl.policy.use_vqa_verification",
    default=False,
)
print("true" if enabled else "false")
' "${config_path}")

if [[ "${use_vqa}" == "true" ]]; then
    for required_file in "${QWEN_PYTHON}" "${QWEN_MODEL_PATH}/config.json"; do
        if [[ ! -f "${required_file}" ]]; then
            echo "ERROR: VQA is enabled but required file was not found: ${required_file}"
            exit 1
        fi
    done
fi

if [[ "${dry_run}" == "true" ]]; then
    if [[ "${use_vqa}" == "true" ]]; then
        dry_run_qwen="Qwen model: ${QWEN_MODEL_PATH}
Qwen Python: ${QWEN_PYTHON}
Qwen VQA port: ${QWEN_VQA_PORT}"
    else
        dry_run_qwen="Qwen VQA: disabled; model and Python checks skipped"
    fi
    cat <<EOF
VLFM Python: ${VLFM_PYTHON}
Config: ${config_path}
VQA enabled: ${use_vqa}
${dry_run_qwen}
Core ports: BLIP2ITM=${BLIP2ITM_PORT} SAM=${SAM_PORT} YOLOV8=${YOLOV8_PORT}
EOF
    exit 0
fi

service_ports=("${BLIP2ITM_PORT}" "${SAM_PORT}" "${YOLOV8_PORT}")
if [[ "${use_vqa}" == "true" ]]; then
    service_ports+=("${QWEN_VQA_PORT}")
fi

for service_port in "${service_ports[@]}"; do
    if ss -lnt | grep -q ":${service_port} "; then
        echo "ERROR: port ${service_port} is already in use. Stop the old VLM server session first."
        exit 1
    fi
done

tmux new-session -d -s "${session_name}" -n services
tmux split-window -h -t "${session_name}:0"
tmux split-window -v -t "${session_name}:0.0"
if [[ "${use_vqa}" == "true" ]]; then
    tmux split-window -v -t "${session_name}:0.1"
fi
tmux select-layout -t "${session_name}:0" tiled

tmux send-keys -t "${session_name}:0.0" \
    "cd '${REPO_ROOT}' && echo '=== BLIP2ITM :${BLIP2ITM_PORT} ===' && BLIP2ITM_PORT=${BLIP2ITM_PORT} '${VLFM_PYTHON}' -m vlfm.vlm.blip2itm --port ${BLIP2ITM_PORT}" C-m

if [[ "${use_vqa}" == "true" ]]; then
    tmux send-keys -t "${session_name}:0.1" \
        "cd '${REPO_ROOT}' && echo '=== Qwen VQA :${QWEN_VQA_PORT} (${QWEN_MODEL_PATH}) ===' && QWEN_VQA_PORT=${QWEN_VQA_PORT} '${QWEN_PYTHON}' -m vlfm.vlm.qwen_vl --port ${QWEN_VQA_PORT} --model-path '${QWEN_MODEL_PATH}'" C-m
    wait_for_core="until curl -fsS http://localhost:${BLIP2ITM_PORT}/blip2itm/health >/dev/null && curl -fsS http://localhost:${QWEN_VQA_PORT}/qwen_vl/health >/dev/null; do sleep 5; done"
    wait_message="Waiting for BLIP2ITM and Qwen VQA..."
    yolo_pane=0.2
    sam_pane=0.3
else
    wait_for_core="until curl -fsS http://localhost:${BLIP2ITM_PORT}/blip2itm/health >/dev/null; do sleep 5; done"
    wait_message="Waiting for BLIP2ITM..."
    yolo_pane=0.1
    sam_pane=0.2
fi

tmux send-keys -t "${session_name}:${yolo_pane}" \
    "cd '${REPO_ROOT}' && echo '${wait_message}' && ${wait_for_core}; echo '=== YOLOv8 :${YOLOV8_PORT} ===' && YOLOV8_PORT=${YOLOV8_PORT} '${VLFM_PYTHON}' -m myon.vlm.yolov8 --port ${YOLOV8_PORT} --weights '${YOLOV8_WEIGHTS}'" C-m

tmux send-keys -t "${session_name}:${sam_pane}" \
    "cd '${REPO_ROOT}' && echo '${wait_message}' && ${wait_for_core}; echo '=== MobileSAM :${SAM_PORT} ===' && SAM_PORT=${SAM_PORT} MOBILE_SAM_CHECKPOINT='${MOBILE_SAM_CHECKPOINT}' '${VLFM_PYTHON}' -m vlfm.vlm.sam --port ${SAM_PORT}" C-m

if [[ "${use_vqa}" == "true" ]]; then
    qwen_service_line="  Qwen VQA  http://localhost:${QWEN_VQA_PORT}/qwen_vl"
    qwen_model_line="Qwen model: ${QWEN_MODEL_PATH}"
    eval_ports="BLIP2ITM_PORT=${BLIP2ITM_PORT} SAM_PORT=${SAM_PORT} QWEN_VQA_PORT=${QWEN_VQA_PORT} YOLOV8_PORT=${YOLOV8_PORT}"
else
    qwen_service_line="  Qwen VQA  disabled by ${config_path}"
    qwen_model_line="Qwen model: not loaded"
    eval_ports="BLIP2ITM_PORT=${BLIP2ITM_PORT} SAM_PORT=${SAM_PORT} YOLOV8_PORT=${YOLOV8_PORT}"
fi

cat <<EOF
Created tmux session: ${session_name}

Services:
  BLIP2ITM  http://localhost:${BLIP2ITM_PORT}/blip2itm
${qwen_service_line}
  YOLOv8    http://localhost:${YOLOV8_PORT}/yolov8
  MobileSAM http://localhost:${SAM_PORT}/mobile_sam

${qwen_model_line}

Monitor:
  tmux attach-session -t ${session_name}

Run evaluation with matching environment ports:
  ${eval_ports} python myon/run.py

Stop the service session:
  tmux kill-session -t ${session_name}
EOF
