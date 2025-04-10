import os
from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import sapien.physx as physx
import torch
from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import vectorize_pose

from .controllers import MSParameterizedMimicJointConfig

# assets_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./meshes"))
from dexwm.utils.macros import ASSETS_DIR


@register_agent()
class XHandRight(BaseAgent):
    disable_self_collisions = True
    uid = "xhand_right"
    # retargeting_name = "ability_hand"
    urdf_path = f"{ASSETS_DIR}/xhand/xhand_right.urdf"
    action_names = [
        "right_hand_thumb_bend_joint",
        "right_hand_thumb_rota_joint1",
        "right_hand_thumb_rota_joint2",
        "right_hand_index_bend_joint",
        "right_hand_index_joint1",
        "right_hand_index_joint2",
        "right_hand_mid_joint1",
        "right_hand_mid_joint2",
        "right_hand_ring_joint1",
        "right_hand_ring_joint2",
        "right_hand_pinky_joint1",
        "right_hand_pinky_joint2",
    ]
    vis_joint_names = [
        "right_hand_thumb_bend_joint",
        "right_hand_thumb_rota_joint1",
        "right_hand_thumb_rota_joint2",
        # "right_hand_thumb_rota_joint3",
        "right_hand_index_bend_joint",
        "right_hand_index_joint1",
        "right_hand_index_joint2",
        # "right_hand_index_joint3",
        "right_hand_mid_joint1",
        "right_hand_mid_joint2",
        # "right_hand_mid_joint3",
        "right_hand_ring_joint1",
        "right_hand_ring_joint2",
        # "right_hand_ring_joint3",
        "right_hand_pinky_joint1",
        "right_hand_pinky_joint2",
        # "right_hand_pinky_joint3",
    ]

    finger_link_names = [
        "right_hand_thumb_bend_link",
        "right_hand_thumb_rota_link1",
        "right_hand_thumb_rota_link2",
        "right_hand_index_bend_link",
        "right_hand_index_rota_link1",
        "right_hand_index_rota_link2",
        "right_hand_mid_link1",
        "right_hand_mid_link2",
        "right_hand_ring_link1",
        "right_hand_ring_link2",
        "right_hand_pinky_link1",
        "right_hand_pinky_link2",
    ]

    mimic_joint_map = {}

    def __init__(self, *args, **kwargs):
        self.hand_joint_names = [
            "right_hand_thumb_bend_joint",
            "right_hand_thumb_rota_joint1",
            "right_hand_thumb_rota_joint2",
            "right_hand_index_bend_joint",
            "right_hand_index_joint1",
            "right_hand_index_joint2",
            "right_hand_mid_joint1",
            "right_hand_mid_joint2",
            "right_hand_ring_joint1",
            "right_hand_ring_joint2",
            "right_hand_pinky_joint1",
            "right_hand_pinky_joint2",
        ]
        self.hand_stiffness = 2e4
        self.hand_damping = 300
        self.hand_force_limit = 20

        self.ee_link_name = "right_hand_ee_link"

        super().__init__(*args, **kwargs)

    # @property
    # def _sensor_configs(self):
    #     return [
    #         CameraConfig(
    #             uid="hand_camera",
    #             pose=Pose.create_from_pq([0.0, 0.08, 0.02], [0, -1, 0, 1]),
    #             width=128,
    #             height=128,
    #             fov=np.pi / 2,
    #             near=0.01,
    #             far=100,
    #             mount=sapien_utils.get_obj_by_name(
    #                 self.robot.get_links(), "thumb_base"
    #             ),
    #         )
    #     ]

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        # arm_pd_joint_pos = PDJointPosControllerConfig(
        #     self.arm_joint_names,
        #     None,
        #     None,
        #     self.arm_stiffness,
        #     self.arm_damping,
        #     self.arm_force_limit,
        #     normalize_action=False,
        # )
        # arm_pd_joint_delta_pos = PDJointPosControllerConfig(
        #     self.arm_joint_names,
        #     -0.1,
        #     0.1,
        #     self.arm_stiffness,
        #     self.arm_damping,
        #     self.arm_force_limit,
        #     use_delta=True,
        # )
        # arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        # arm_pd_joint_target_delta_pos.use_target = True

        # # PD ee position
        # arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
        #     self.arm_joint_names,
        #     -0.1,
        #     0.1,
        #     0.1,
        #     self.arm_stiffness,
        #     self.arm_damping,
        #     self.arm_force_limit,
        #     ee_link=self.ee_link_name,
        #     urdf_path=self.urdf_path,
        #     # frame="ee_align",
        #     # use_delta=False,
        #     # normalize_action=False,
        # )

        # arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        # arm_pd_ee_target_delta_pose.use_target = True

        # -------------------------------------------------------------------------- #
        # Hand
        # -------------------------------------------------------------------------- #
        hand_pd_joint_pos = PDJointPosControllerConfig(
            self.hand_joint_names,
            None,
            None,
            self.hand_stiffness,
            self.hand_damping,
            self.hand_force_limit,
            drive_mode="acceleration",
            use_delta=False,
        )

        hand_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.hand_joint_names,
            -0.1,
            0.1,
            self.hand_stiffness,
            self.hand_damping,
            self.hand_force_limit,
            drive_mode="acceleration",
            use_delta=True,
        )
        hand_pd_joint_target_pos = deepcopy(hand_pd_joint_pos)
        hand_pd_joint_target_pos.use_target = True

        # hand_pd_joint_mimic_pos = MSParameterizedMimicJointConfig(
        #     self.hand_joint_names,
        #     None,
        #     None,
        #     self.hand_stiffness,
        #     self.hand_damping,
        #     self.hand_force_limit,
        #     drive_mode="acceleration",
        #     use_delta=False,
        #     coefficient=[1.05851325, 1.05851325, 1.05851325, 1.05851325],
        #     offset=[0.72349796, 0.72349796, 0.72349796, 0.72349796],
        #     mimic_target={
        #         "index_q2": "index_q1",
        #         "middle_q2": "middle_q1",
        #         "ring_q2": "ring_q1",
        #         "pinky_q2": "pinky_q1",
        #     },
        # )
        # hand_pd_joint_mimic_delta_pos = MSParameterizedMimicJointConfig(
        #     self.hand_joint_names,
        #     -0.1,
        #     0.1,
        #     self.hand_stiffness,
        #     self.hand_damping,
        #     self.hand_force_limit,
        #     drive_mode="acceleration",
        #     use_delta=True,
        #     coefficient=[1.05851325, 1.05851325, 1.05851325, 1.05851325],
        #     offset=[0.72349796, 0.72349796, 0.72349796, 0.72349796],
        #     mimic_target={
        #         "index_q2": "index_q1",
        #         "middle_q2": "middle_q1",
        #         "ring_q2": "ring_q1",
        #         "pinky_q2": "pinky_q1",
        #     },
        # )
        # hand_target_mimic_delta_pos = deepcopy(hand_pd_joint_mimic_delta_pos)
        # hand_target_mimic_delta_pos.use_target = True

        controller_configs = dict(
            pd_joint_mimic_pos=hand_pd_joint_target_pos,  # dict(arm=arm_pd_joint_pos, hand=hand_pd_joint_mimic_pos),
            # pd_joint_delta_pos=dict(arm=arm_pd_joint_delta_pos, hand=hand_delta_pos),
            # pd_joint_mimic_delta_pos=hand_pd_joint_mimic_delta_pos,  # dict(
            #     arm=arm_pd_joint_delta_pos, hand=hand_pd_joint_mimic_delta_pos
            # ),
            # pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=hand_target_delta_pos),
            # pd_ee_delta_pose=dict(
            #     arm=arm_pd_ee_delta_pose, gripper=hand_target_delta_pos
            # ),
            # pd_ee_target_delta_pose=dict(
            #     arm=arm_pd_ee_target_delta_pose, gripper=hand_target_delta_pos
            # ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def _after_init(self):
        # hand_front_link_names = [
        #     "thumb_L2",
        #     "index_L2",
        #     "middle_L2",
        #     "ring_L2",
        #     "pinky_L2",
        # ]
        # self.hand_front_links = sapien_utils.get_objs_by_names(
        #     self.robot.get_links(), hand_front_link_names
        # )

        # finger_tip_link_names = [
        #     "thumb_tip",
        #     "index_tip",
        #     "middle_tip",
        #     "ring_tip",
        #     "pinky_tip",
        # ]
        # self.finger_tip_links = sapien_utils.get_objs_by_names(
        #     self.robot.get_links(), finger_tip_link_names
        # )

        # self.palm_link_name = "base"
        # self.palm_link = sapien_utils.get_obj_by_name(
        #     self.robot.get_links(), self.palm_link_name
        # )

        # self.tcp = sapien_utils.get_obj_by_name(
        #     self.robot.get_links(), self.ee_link_name
        # )

        self.queries: Dict[str, Tuple[physx.PhysxGpuContactQuery, Tuple[int]]] = dict()

        # self.color = {
        #     "thumb_base": [0.1, 0.9, 0.9],
        #     "thumb_L1": [0.9, 0.1, 0.9],
        #     "index_L1": [0.9, 0.1, 0.9],
        #     "middle_L1": [0.9, 0.1, 0.9],
        #     "ring_L1": [0.9, 0.1, 0.9],
        #     "pinky_L1": [0.9, 0.1, 0.9],
        #     "thumb_L2": [0.9, 0.9, 0.1],
        #     "index_L2": [0.9, 0.9, 0.1],
        #     "middle_L2": [0.9, 0.9, 0.1],
        #     "ring_L2": [0.9, 0.9, 0.1],
        #     "pinky_L2": [0.9, 0.9, 0.1],
        # }

    def _before_reset(self, num_envs=0):
        init_state = torch.tensor(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        )
        self.init_state = init_state

        if num_envs == 0:
            num_envs = self.robot.get_state().shape[0]
        self.robot.set_state(init_state.repeat(num_envs, 1))
        # self.controller.reset()

    # def is_grasping(self, object: Actor = None):
    #     # TODO: Implement this function
    #     pass

    # def is_static(self, threshold: float = 0.2):
    #     qvel = self.robot.get_qvel()[..., :-2]
    #     return torch.max(torch.abs(qvel), 1)[0] <= threshold

    # def is_contact(self, object: Actor, min_force: float = 0.5):
    #     contacts = torch.zeros(self.scene.num_envs, device=self.device)
    #     for idx in range(len(self.hand_front_links)):
    #         contact_forces = self.scene.get_pairwise_contact_forces(
    #             self.hand_front_links[idx], object
    #         )

    #         forces = torch.linalg.norm(contact_forces, dim=-1)

    #         flag = forces > min_force
    #         # direction to open fingers
    #         # direction = -self.finger_lilink_poses(self, link_names, tag):
    #     """
    #     Get the pose of the links specified by the link names
    #     """
    #     if getattr(self, "_link_tags", None) is None:
    #         self._link_tags = dict()
    #     if tag not in self._link_tags:
    #         self._link_tags[tag] = sapien_utils.get_objs_by_names(
    #             self.robot.get_links(), link_names
    #         )

    #     links = self._link_tags[tag]
    #     return torch.stack(
    #         [link.pose.to_transformation_matrix() for link in links], dim=1
    #     )

    #         # angle = common.compute_angle_between(direction, contact_forces)
    #         # flag = torch.logical_and(
    #         #     forces > min_force, torch.rad2deg(angle) < max_angle
    #         # )
    #         contacts += flag
    #     return contacts >= 2

    # @property
    # def palm_pose(self):
    #     return vectorize_pose(self.palm_link.pose, device=self.device)

    # @property
    # def pulp_poses(self):
    #     pulp_poses = [
    #         vectorize_pose(link.pose, device=self.device)
    #         for link in self.hand_front_links
    #     ]
    #     return torch.stack(pulp_poses, dim=-2)

    # @property
    # def tip_poses(self):
    #     tip_poses = [
    #         vectorize_pose(link.pose, device=self.device)
    #         for link in self.finger_tip_links
    #     ]
    #     return torch.stack(tip_poses, dim=-2)

    def get_link_poses(self, link_names, tag):
        """
        Get the pose of the links specified by the link names
        """
        if getattr(self, "_link_tags", None) is None:
            self._link_tags = dict()
        if tag not in self._link_tags:
            self._link_tags[tag] = sapien_utils.get_objs_by_names(
                self.robot.get_links(), link_names
            )

        links = self._link_tags[tag]
        return torch.stack(
            [link.pose.to_transformation_matrix() for link in links], dim=1
        )
