from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import SimConfig, GPUMemoryConfig
from ..robot import AbilityHandRight, XHandRight


@register_env("CustomEnv-v1", max_episode_steps=200)
class CustomEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda", "fetch", "ability_hand_right", "xhand_right"]
    agent: Union[Panda, Fetch]

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.robot_uids = robot_uids
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # def reset(self, **kwargs):
    #     with torch.no_grad():
    #         self.agent.reset(qpos)
    # super().reset()

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**18,
                max_rigid_contact_count=2**25,
            ),
            spacing=20,
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1)

    # def _load_agent(self, options: dict):
    #     super()._load_agent(options, sapien.Pose(p=[0, 0, 0]))

    def _load_scene(self, options: dict):
        pass

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # pass
        if self.robot_uids == "ability_hand_right":
            init_qpos = torch.tensor([0, 0, 0, 0, 0, 0, 0.7235, 0.7235, 0.7235, 0.7235], device=self.device) # ability hand
        elif self.robot_uids == "xhand_right":
            init_qpos = torch.tensor([0.7235, 0.7235, 0.7235, 0.7235, 0.7235, 0.7235, 0.7235, 0.7235, 0.7235, 0.7235, 0.7235, 0.7235], device=self.device) # xhand 
        else:
            raise NotImplementedError(f"Robot {self.robot_uids} not implemented")
        with torch.device(self.device):
            self.agent.reset(init_qpos)

    def evaluate(self):
        return {
            "success": torch.zeros(self.num_envs, device=self.device, dtype=bool),
            "fail": torch.zeros(self.num_envs, device=self.device, dtype=bool),
        }

    def _get_obs_extra(self, info: Dict):
        return dict()

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
