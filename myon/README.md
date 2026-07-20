# myon ObjectNav 复现说明

本目录是当前仓库中基于 Habitat 的 Myon 版本 ObjectNav 实现。它在 VLFM 的深度占据图、目标检测、MobileSAM 分割和前沿探索基础上，加入了更稳定的前沿选择与 Qwen 视觉问答验证。

## 当前版本

- 评测入口：`python myon/run.py`
- 默认配置：`config/experiments/myon_objectnav_hm3d.yaml`
- 数据集：HM3D ObjectNav，默认评测 split 为 `val_challenging`
- 并行环境：1 个（便于单 GPU 复现）
- 策略：`HabitatSimplePolicyV2`
- 目标类别：HM3D 的 `chair`、`bed`、`potted plant`、`toilet`、`tv`、`couch`
- 目标检测：Ultralytics YOLOv8
- 图像分割：MobileSAM
- 图文匹配：BLIP-2 ITM 服务（端口 `12182`）
- 目标可见性验证：Qwen 视觉语言模型（端口 `12184`，配置中默认开启）
- 前沿策略：最小坚持步数、防抖切换、卡住前沿冷却黑名单、环路抑制
- 记录：评测日志会写入 Habitat 的输出目录；设置 `VLFM_RECORD_ACTIONS_DIR` 可额外保存逐步动作和 VQA 信息

Qwen 策略 `QwenPolicyV3` 也保留在代码中：它对每个观测分别预测目标相关性和可探索性，并将两者等权融合为前沿价值。默认 HM3D 配置使用 `HabitatSimplePolicyV2` 的 BLIP-2 ITM 探索价值，并使用 Qwen 做最终目标可见性验证。

## 环境

推荐 Linux、CUDA GPU（至少 16 GB 显存；Qwen3.5-2B 更省显存）和 conda。仓库原始依赖锁定在 Python 3.9、PyTorch 1.12.1、CUDA 11.3 生态；不要在同一个环境中升级这些核心版本。

### 1. 主环境 `vlfm`

```bash
conda create -n vlfm python=3.9 -y
conda activate vlfm
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
  -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/IDEA-Research/GroundingDINO.git@eeba084341aaa454ce13cb32fa7fd9282fc73a67 \
  salesforce-lavis==1.0.2
pip install -e ".[habitat]"
pip install ultralytics mobile-sam frontier_exploration depth_camera_filtering
sudo apt-get install tmux curl
```

`pyproject.toml` 中的核心版本包括：`numpy==1.26.4`、`transformers==4.26.0`、`timm==0.4.12`、`opencv-python==4.5.5.64`、`flask>=2.3.2`、`open3d>=0.17.0`。其中 `transformers==4.26.0` 是 BLIP-2/LAVIS 兼容性要求。

### 2. Qwen 服务环境 `vllm_service`

Qwen 服务使用独立 Python 解释器，避免与主环境的旧 PyTorch/LAVIS 依赖冲突。脚本默认查找 `/root/anaconda3/envs/vllm_service/bin/python`，也可以用 `--qwen-python` 或 `QWEN_PYTHON` 指定其他解释器。

```bash
conda create -n vllm_service python=3.10 -y
conda activate vllm_service
# 按本机 CUDA 版本安装兼容的 PyTorch
pip install torch torchvision transformers accelerate flask pillow requests
```

Qwen 代码通过 `AutoModelForImageTextToText` 和 `AutoProcessor` 加载本地 checkpoint，不再依赖 vLLM。建议使用 Qwen3.5-2B；若使用 Qwen3-VL-8B，需要约 16 GB 以上显存，并确保 `transformers` 支持该 checkpoint 的模型类。

## 模型和文件

大文件不随 Git 提交，请自行下载到以下位置：

| 文件 | 默认路径 | 用途 |
| --- | --- | --- |
| PointNav 权重 | `data/pointnav_weights.pth` | 深度点导航控制器（仓库已包含） |
| Dummy policy | `data/dummy_policy.pth` | Habitat 评测占位 checkpoint；缺失时运行 `python -m vlfm.utils.generate_dummy_policy` |
| BLIP-2 权重 | LAVIS 默认缓存，或 `data/blip2_pretrained.pth` | ITM 服务 |
| MobileSAM | `data/mobile_sam.pt` | 实例分割 |
| YOLOv8 | `yolov8x.pt` | COCO 目标检测 |
| Qwen | `qwen/qwen3.5-2B` 或 `qwen/qwen3-vl-8B` | VQA 目标验证 |

同时需要克隆 YOLOv7（部分旧版 VLFM 代码仍会导入）：

```bash
git clone https://github.com/WongKinYiu/yolov7.git
```

## Habitat 数据集

需要 Habitat-Sim `v0.2.4`、Habitat-Lab/Baselines `0.2.4-20230405`，以及 Matterport/HM3D 账号下载场景。数据目录建议设为仓库下的 `data/`，最终至少包含：

```text
data/
  scene_datasets/hm3d/...
  datasets/objectnav/hm3d/v1/...
```

下载方式沿用仓库根目录 `README.md` 的 HM3D 章节。下载完成后，确认 `config/experiments/myon_objectnav_hm3d.yaml` 中的数据集路径与本地目录一致。

## 启动服务与评测

先激活主环境并在仓库根目录执行：

```bash
conda activate vlfm
bash scripts/launch_vlm_servers_itm_vqa.sh --qwen-model 2b
```

启动脚本会创建 tmux 会话并按依赖启动四个服务：

| 服务 | 默认端口 |
| --- | ---: |
| BLIP2 ITM | `12182` |
| MobileSAM | `12183` |
| Qwen VQA | `12184` |
| YOLOv8 | `12186` |

查看服务日志：

```bash
tmux ls
tmux attach-session -t <脚本输出的会话名>
```

服务健康检查通过后，在另一个终端运行完整 HM3D 评测：

```bash
conda activate vlfm
cd /root/objnav/vlfm
BLIP2ITM_PORT=12182 SAM_PORT=12183 QWEN_VQA_PORT=12184 YOLOV8_PORT=12186 \
  python myon/run.py
```

只检查路径、配置和端口而不启动服务：

```bash
bash scripts/launch_vlm_servers_itm_vqa.sh --dry-run --qwen-model 2b
```

评测结束后释放 GPU：

```bash
tmux kill-session -t <会话名>
```

如需关闭 Qwen 验证，可在配置中将 `use_vqa_verification` 改为 `False`；脚本会根据配置跳过 Qwen 服务，但 YOLOv8、MobileSAM 和 BLIP-2 ITM 仍需运行。

## 测试

```bash
conda activate vlfm
pytest -q test/test_acyclic_enforcer.py test/test_qwen_policy.py test/test_qwen_vqa.py test/test_vqa_logging.py
```

这些测试覆盖前沿环路抑制、Qwen 分数融合/响应解析和 VQA 日志。完整 Habitat 评测属于 GPU、数据集和模型服务集成测试，不能仅通过单元测试替代。

## 复现注意事项

1. 所有服务端口必须空闲；启动脚本会在创建 tmux 会话前检查端口。
2. Qwen 服务和主环境可以使用不同的 Python，但必须共享本地模型目录和 CUDA 驱动。
3. `val_challenging` 是当前配置的默认 split；复现旧结果时请显式切换到对应 split，并记录 checkpoint、模型版本和端口配置。
4. 本仓库不提交模型、场景数据、评测输出和 TensorBoard 文件；这些内容应通过下载或运行生成。
