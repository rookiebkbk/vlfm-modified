"""展示 run.py 中 cfg 传入 execute_exp 前的完整状态。

用法:
    cd /root/objnav/vlfm && python myon/show_info.py

输出:
    - cfg 的完整 YAML 内容（传入 execute_exp 前的状态）
    - 各主要配置块的键列表
"""

import os
import random
import torch
import numpy as np

# 注册 habitat 及 vlfm 的自定义类，使其可被 Hydra 发现
import frontier_exploration  # noqa: F401
import hydra  # noqa: F401
from habitat import get_config  # noqa: F401
from habitat.config import read_write
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.run import execute_exp
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from omegaconf import DictConfig, OmegaConf
from habitat_baselines.utils.common import get_action_space_info  # noqa: F401

import vlfm.measurements.traveled_stairs  # noqa: F401
import vlfm.obs_transformers.resize  # noqa: F401
import vlfm.policy.action_replay_policy  # noqa: F401
import vlfm.policy.habitat_policies  # noqa: F401
# import vlfm.utils.vlfm_trainer  # noqa: F401


from myon.final.utils import final_trainer  # noqa: F401

class HabitatConfigPlugin(SearchPathPlugin):
    """向 Hydra 注册 habitat 自带的 config/ 搜索路径。"""
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="habitat", path="config/")

register_hydra_plugin(HabitatConfigPlugin)



@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="experiments/vlfm_objectnav_hm3d",
)
def main(cfg: DictConfig) -> None:

    # ---- 以下与 run.py 中的处理完全一致 ----
    assert os.path.isdir("data"), "Missing 'data/' directory!"
    if not os.path.isfile("data/dummy_policy.pth"):
        print("Dummy policy weights not found! Please run the following command first:")
        print("python -m vlfm.utils.generate_dummy_policy")
        exit(1)

    # 在 patch_config 锁定之前替换 trainer
    cfg.habitat_baselines.trainer_name = "final"
    # 设置不打印完整配置
    cfg.habitat_baselines.verbose = False


    cfg = patch_config(cfg)
    with read_write(cfg):
        try:
            cfg.habitat.simulator.agents.main_agent.sim_sensors.pop("semantic_sensor")
        except KeyError:
            pass
    # ---- 到这里 cfg 就是传入 execute_exp 前的最终状态 ----
    execute_exp(cfg, "eval" if cfg.habitat_baselines.evaluate else "train")

if __name__ == "__main__":
    main()
