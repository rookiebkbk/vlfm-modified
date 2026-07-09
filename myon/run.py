# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

"""myon 精简版 Policy 入口。

用法:
    cd /root/objnav/vlfm && python myon/run.py

启动前需要先运行模型服务:
    python -m myon.vlm.yolov8 --port 12186 &
    python -m vlfm.vlm.sam --port 12183 &
    python -m vlfm.vlm.blip2itm --port 12182 &
"""

import os

# 注册 habitat 及 vlfm 的自定义类，使其可被 Hydra 发现
import frontier_exploration  # noqa: F401
import hydra  # noqa: F401
from habitat.config import read_write
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.run import execute_exp
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from omegaconf import DictConfig

import vlfm.measurements.traveled_stairs  # noqa: F401
import vlfm.obs_transformers.resize  # noqa: F401
import vlfm.policy.action_replay_policy  # noqa: F401
import vlfm.policy.habitat_policies  # noqa: F401

# 注册 myon 的自定义 policy 和 trainer
import myon.policy.habitat_policies  # noqa: F401
import myon.final.utils.final_trainer  # noqa: F401


class HabitatConfigPlugin(SearchPathPlugin):
    """向 Hydra 注册 habitat 自带的 config/ 搜索路径。"""

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="habitat", path="config/")


register_hydra_plugin(HabitatConfigPlugin)


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="experiments/myon_objectnav_hm3d",
)
def main(cfg: DictConfig) -> None:
    assert os.path.isdir("data"), "Missing 'data/' directory!"
    if not os.path.isfile("data/dummy_policy.pth"):
        print("Dummy policy weights not found! Please run:")
        print("  python -m vlfm.utils.generate_dummy_policy")
        exit(1)

    cfg.habitat_baselines.verbose = False

    cfg = patch_config(cfg)
    with read_write(cfg):
        try:
            cfg.habitat.simulator.agents.main_agent.sim_sensors.pop("semantic_sensor")
        except KeyError:
            pass

    execute_exp(cfg, "eval" if cfg.habitat_baselines.evaluate else "train")


if __name__ == "__main__":
    main()
