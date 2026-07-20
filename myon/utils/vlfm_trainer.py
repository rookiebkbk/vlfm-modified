# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
import tqdm
from habitat import VectorEnv, logger
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat_baselines import PPOTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.common.tensorboard_utils import (
    TensorboardWriter,
)
from habitat_baselines.rl.ddppo.algo import DDPPO  # noqa: F401.
from habitat_baselines.rl.ppo.single_agent_access_mgr import (  # noqa: F401.
    SingleAgentAccessMgr,
)
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import (
    extract_scalars_from_info as extract_scalars_from_info_habitat,
)
from omegaconf import OmegaConf


def extract_scalars_from_info(info: Dict[str, Any]) -> Dict[str, float]:
    info_filtered = {
        k: v
        for k, v in info.items()
        if not isinstance(v, list) and k != "vqa_step"
    }
    return extract_scalars_from_info_habitat(info_filtered)


@baseline_registry.register_trainer(name="vlfm")
class VLFMTrainer(PPOTrainer):
    envs: VectorEnv

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Some configurations require not to load the checkpoint, like when using
        # a hierarchial policy
        if self.config.habitat_baselines.eval.should_load_ckpt:
            # map_location="cpu" is almost always better than mapping to a CUDA device.
            ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
            step_id = ckpt_dict["extra_state"]["step"]
            print(step_id)
        else:
            ckpt_dict = {"config": None}
        # 若要严格复现训练的结果，则从ckpt中加载配置，否则使用当前配置：
        config = self._get_resume_state_config_or_new_config(ckpt_dict["config"])

        with read_write(config):
            config.habitat.dataset.split = config.habitat_baselines.eval.split

        if len(self.config.habitat_baselines.eval.video_option) > 0:
            agent_config = get_agent_config(config.habitat.simulator)
            agent_sensors = agent_config.sim_sensors
            extra_sensors = config.habitat_baselines.eval.extra_sim_sensors
            with read_write(agent_sensors):
                agent_sensors.update(extra_sensors)
            with read_write(config):
                if config.habitat.gym.obs_keys is not None:
                    for render_view in extra_sensors.values():
                        if render_view.uuid not in config.habitat.gym.obs_keys:
                            config.habitat.gym.obs_keys.append(render_view.uuid)
                config.habitat.simulator.debug_render = True

        if config.habitat_baselines.verbose:
            logger.info(f"env config: {OmegaConf.to_yaml(config)}")

        self._init_envs(config, is_eval=True)

        self._agent = self._create_agent(None)
        action_shape, discrete_actions = get_action_space_info(self._agent.policy_action_space)

        if self._agent.actor_critic.should_load_agent_state:
            self._agent.load_state_dict(ckpt_dict)

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device="cpu")

        test_recurrent_hidden_states = torch.zeros(
            (
                self.config.habitat_baselines.num_environments,
                *self._agent.hidden_state_shape,
            ),
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.habitat_baselines.num_environments,
            *action_shape,
            device=self.device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            self.config.habitat_baselines.num_environments,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[Any, Any] = {}  # dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        rgb_frames: List[List[np.ndarray]] = [[] for _ in range(self.config.habitat_baselines.num_environments)]
        if len(self.config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(self.config.habitat_baselines.video_dir, exist_ok=True)

        # 确定要跑多少episodes，-1表示跑完数据集里的所有episodes
        number_of_eval_episodes = self.config.habitat_baselines.test_episode_count # -1
        evals_per_ep = self.config.habitat_baselines.eval.evals_per_ep # 1
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes, dataset only has {{total_num_eps}}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert number_of_eval_episodes > 0, "You must specify a number of evaluation episodes with test_episode_count"

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        self._agent.eval()

        from vlfm.utils.habitat_visualizer import HabitatVis

        num_successes = 0
        num_total = 0
        hab_vis = HabitatVis() #用于 video 制作
        if "VLFM_RECORD_ACTIONS_DIR" in os.environ:
            episode_step_buffers = {i: [] for i in range(self.config.habitat_baselines.num_environments)}
        while len(stats_episodes) < (number_of_eval_episodes * evals_per_ep) and self.envs.num_envs > 0:
            current_episodes_info = self.envs.current_episodes()

            with inference_mode():# 更激进的torch.no_grad()，会完全禁止梯度计算和autocast
                action_data = self._agent.actor_critic.act( # HabitatITMPolicyV2.act()
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )

                # Buffer detailed per-step action info for analysis
                if "VLFM_RECORD_ACTIONS_DIR" in os.environ:
                    policy_info = action_data.policy_info[0] if action_data.policy_info else {}
                    action_id = int(action_data.actions.cpu()[0].item())
                    nav_goal_raw = policy_info.get("nav_goal", np.zeros(0))
                    rho_theta_raw = policy_info.get("rho_theta", np.zeros(0))
                    step_data = {
                        "step": len(episode_step_buffers[0]),
                        "action": action_id,
                        "mode": policy_info.get("mode", "unknown"),
                        "target_object": policy_info.get("target_object", ""),
                        "gps": policy_info.get("gps", ""),
                        "yaw": policy_info.get("yaw", None),
                        "target_detected": bool(policy_info.get("target_detected", False)),
                        "nav_goal": [float(x) for x in nav_goal_raw] if hasattr(nav_goal_raw, '__len__') and len(nav_goal_raw) > 0 else None,
                        "rho_theta": [float(x) for x in rho_theta_raw] if hasattr(rho_theta_raw, '__len__') and len(rho_theta_raw) > 0 else None,
                        "stop_called": bool(policy_info.get("stop_called", False)),
                    }
                    if "vqa_step" in policy_info:
                        step_data["vqa"] = policy_info["vqa_step"]
                    episode_step_buffers[0].append(step_data)

                #更新点导航策略的隐藏状态和 prev_actions
                if action_data.should_inserts is None:
                    test_recurrent_hidden_states = action_data.rnn_hidden_states
                    prev_actions.copy_(action_data.actions)  # type: ignore
                else:
                    for i, should_insert in enumerate(action_data.should_inserts):
                        if should_insert.item():
                            test_recurrent_hidden_states[i] = action_data.rnn_hidden_states[i]
                            prev_actions[i].copy_(action_data.actions[i])  # type: ignore

            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            # action_data.env_actions 是经过转换器转换后的动作，适用于环境的动作空间
            # action_data.actions 是原始动作，适用于策略网络的输入
            # 离散动作的情况下，action_data.env_actions 和 action_data.actions 通常是一样的。
            # 但真机情况下，action_data.env_actions 是连续动作，需要截断到合法范围。
            if is_continuous_action_space(self._env_spec.action_space):
                # Clipping actions to the specified limits
                step_data = [
                    np.clip(
                        a.numpy(),
                        self._env_spec.action_space.low,
                        self._env_spec.action_space.high,
                    )
                    for a in action_data.env_actions.cpu()
                ]
            else:
                step_data = [a.item() for a in action_data.env_actions.cpu()]
            
            
            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]
            policy_infos = self._agent.actor_critic.get_extra(action_data, infos, dones)
            for i in range(len(policy_infos)):
                infos[i].update(policy_infos[i])

            # new step 的观测
            batch = batch_obs(  # type: ignore
                observations,
                device=self.device,
            )
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            rewards = torch.tensor(rewards_l, dtype=torch.float, device="cpu").unsqueeze(1)
            current_episode_reward += rewards

            # Backfill reward into the last buffered step
            if "VLFM_RECORD_ACTIONS_DIR" in os.environ:
                num_envs = self.envs.num_envs
                for i in range(num_envs):
                    if i in episode_step_buffers and len(episode_step_buffers[i]) > 0:
                        episode_step_buffers[i][-1]["reward"] = float(rewards_l[i])
                        episode_step_buffers[i][-1]["done"] = bool(dones[i])
            next_episodes_info = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):

                # 如果一个episode已经评估了evals_per_ep次了，就暂停这个环境。
                # 或者如果episode_id是123123123，也暂停（用来测试的，表示这个episode不应该被评估）
                if (
                    ep_eval_count[
                        (
                            next_episodes_info[i].scene_id,
                            next_episodes_info[i].episode_id,
                        )
                    ]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)
                elif int(next_episodes_info[i].episode_id) == 123123123:
                    envs_to_pause.append(i)

                if len(self.config.habitat_baselines.eval.video_option) > 0:
                    hab_vis.collect_data(batch, infos, action_data.policy_info)

                # episode ended
                if not not_done_masks[i].item():
                    pbar.update()
                    episode_stats = {"reward": current_episode_reward[i].item()}
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )
                    ep_eval_count[k] += 1
                    # use scene_id + episode_id as unique id for storing stats
                    # 当前episode完成后，env.step() 会自动加载下一个episode。
                    # 完成一个episode后，len(stats_episodes) 会增加1，直到达到 number_of_eval_episodes * evals_per_ep
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats

                    if episode_stats["success"] == 1:
                        num_successes += 1
                    num_total += 1
                    print(f"Success rate: {num_successes / num_total * 100:.2f}% ({num_successes} out of {num_total})")

                    from vlfm.utils.episode_stats_logger import log_episode_stats

                    try:
                        failure_cause = log_episode_stats(
                            current_episodes_info[i].episode_id,
                            current_episodes_info[i].scene_id,
                            infos[i],
                        )
                    except Exception:
                        failure_cause = "Unknown"

                    # Flush per-episode step buffer to JSON
                    if "VLFM_RECORD_ACTIONS_DIR" in os.environ:
                        import json

                        ep_id = current_episodes_info[i].episode_id
                        scene = os.path.basename(current_episodes_info[i].scene_id).split(".")[0]
                        record = {
                            "episode_id": ep_id,
                            "scene_id": scene,
                            "success": int(episode_stats["success"]),
                            "failure_cause": failure_cause,
                            "num_steps": len(episode_step_buffers[i]),
                            "steps": episode_step_buffers[i],
                        }
                        out_dir = os.environ["VLFM_RECORD_ACTIONS_DIR"]
                        os.makedirs(out_dir, exist_ok=True)
                        fname = f"{int(ep_id):04d}_{scene}.json"
                        with open(os.path.join(out_dir, fname), "w") as f:
                            json.dump(record, f, indent=2)
                        episode_step_buffers[i] = []

                    if len(self.config.habitat_baselines.eval.video_option) > 0:
                        rgb_frames[i] = hab_vis.flush_frames(failure_cause)
                        generate_video(
                            video_option=self.config.habitat_baselines.eval.video_option,
                            video_dir=self.config.habitat_baselines.video_dir,
                            images=rgb_frames[i],
                            episode_id=current_episodes_info[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=extract_scalars_from_info(infos[i]),
                            fps=self.config.habitat_baselines.video_fps,
                            tb_writer=writer,
                            keys_to_include_in_name=self.config.habitat_baselines.eval_keys_to_include_in_name,
                        )

                        rgb_frames[i] = []

                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str,
                            self.config.habitat.task,
                            current_episodes_info[i].episode_id,
                        )

            # 将完成了所有分配到的episode的环境从self.envs中移除，直到所有环境都完成了分配的episode退出while循环
            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        pbar.close()

        if "ZSOS_DONE_PATH" in os.environ:
            # Create an empty file at ZSOS_DONE_PATH to signal that the
            # evaluation is done
            done_path = os.environ["ZSOS_DONE_PATH"]
            with open(done_path, "w") as f:
                f.write("")

        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."

        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = np.mean([v[stat_key] for v in stats_episodes.values()])

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalar("eval_reward/average_reward", aggregated_stats["reward"], step_id)

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        self.envs.close()
