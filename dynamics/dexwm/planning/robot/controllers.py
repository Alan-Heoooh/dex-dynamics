from dataclasses import dataclass, field
from typing import Sequence, Union

import numpy as np
import torch
from gymnasium import spaces
from mani_skill.agents.controllers import (
    PDJointPosController,
    PDJointPosControllerConfig,
)
from mani_skill.utils import common
from mani_skill.utils.structs.types import Array, DriveMode


class MSParameterizedMimicJoint(PDJointPosController):
    config: "MSParameterizedMimicJointConfig"  # type: ignore [override maniskill]

    def get_mimic_info(self):
        if not hasattr(self, "action_idx"):
            mimic = {}
            if self.config.mimic_target is None:
                n = len(self.joints) // 2
                for i in range(n):
                    mimic[self.config.joint_names[i + n]] = self.config.joint_names[i]
            else:
                mimic = self.config.mimic_target

            n = len(mimic)

            action_idx = []
            joint_in_action_idx = {}
            joint_idx = {}
            for idx, i in enumerate(self.config.joint_names):
                joint_idx[i] = idx
                if i not in mimic:
                    action_idx.append(idx)  # action will be maped to idx joint
                    joint_in_action_idx[i] = len(action_idx) - 1

            mimic_in_qpos = []
            mimic_target_in_action = []
            for k, v in mimic.items():
                mimic_in_qpos.append(joint_idx[k])
                mimic_target_in_action.append(joint_in_action_idx[v])

            self.action_idx = torch.tensor(
                action_idx, device=self.device, dtype=torch.long
            )
            self.mimic_in_qpos = torch.tensor(
                mimic_in_qpos, device=self.device, dtype=torch.long
            )
            self.mimic_target_in_action = torch.tensor(
                mimic_target_in_action, device=self.device, dtype=torch.long
            )

    def get_fixed_joints_info(self):
        joint_idx = {}
        for idx, i in enumerate(self.config.joint_names):
            joint_idx[i] = idx

        self.fixed_joints = {}
        if self.config.fixed_joints is not None:
            for k, v in self.config.fixed_joints.items():
                self.fixed_joints[joint_idx[k]] = v

    def reset(self):
        super().reset()
        self.get_mimic_info()
        n = len(self.mimic_in_qpos)

        self.get_fixed_joints_info()

        self.coeffient = torch.tensor(
            np.broadcast_to(self.config.coefficient, n),
            device=self.device,
            dtype=torch.float32,
        )
        self.offeset = torch.tensor(
            np.broadcast_to(self.config.offset, n),
            device=self.device,
            dtype=torch.float32,
        )

    def _get_joint_limits(self):
        joint_limits = super()._get_joint_limits()
        self.get_mimic_info()
        return joint_limits[self.action_idx.detach().cpu().numpy()]

    def set_action(self, action: Array):
        # print("???:", self.action_space_low, self.action_space_high)
        action = self._preprocess_action(action)
        action = common.to_tensor(action)  # type: ignore
        assert isinstance(action, torch.Tensor)
        self._step = 0
        self._start_qpos = self.qpos
        if self.config.use_delta:
            if self.config.use_target:
                cur = self._target_qpos
                # raise NotImplementedError("Delta mode is not implemented yet for mimic..")
            else:
                cur = self._start_qpos
            active_target = cur[..., self.action_idx] + action

        else:
            active_target = action

        passive = (
            active_target[..., self.mimic_target_in_action] * self.coeffient[None, :]
            + self.offeset[None, :]
        )

        target_qpos = torch.zeros_like(self.qpos)
        target_qpos[..., self.action_idx] = active_target
        target_qpos[..., self.mimic_in_qpos] = passive

        for k, v in self.fixed_joints.items():
            target_qpos[..., k] = v

        self._target_qpos = target_qpos

        if self.config.interpolate:
            self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
        else:
            self.set_drive_targets(self._target_qpos)


@dataclass
class MSParameterizedMimicJointConfig(PDJointPosControllerConfig):
    coefficient: list[float] = field(default_factory=lambda: [1.0])
    offset: list[float] = field(default_factory=lambda: [0.0])
    mimic_target: dict | None = None
    fixed_joints: dict | None = None
    controller_cls = MSParameterizedMimicJoint
    # which to mimic which
    # str to str
    # when mimic_target is None, it means the second half of the joints mimic the first half
