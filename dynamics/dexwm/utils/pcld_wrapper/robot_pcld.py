import os
import sys

from pathlib import Path
import warnings
import numpy as np
import torch
from .mano_wrapper import ManoWrapper, ManoConfig
from .robot_utils import sample_robot_point_cloud
from .mimic_joint import MimicJointForward
from dexwm.utils.macros import ASSETS_DIR, EXTERNAL_DIR
import dexwm.planning.env  # import *
import dexwm.planning.robot
import gymnasium as gym
from .hand_pcld import PcldWrapper

from manopth.manolayer import ManoLayer
from manopth.tensutils import (
    make_list,
    subtract_flat_id,
    th_pack,
    th_posemap_axisang,
    th_with_zeros,
)
from pytorch3d.transforms import matrix_to_axis_angle

class RobotPcldWrapper(PcldWrapper):
    def __init__(
        self,
        robot_uids,
        num_samples,
        particles_per_hand,
        device="cuda",
    ):
        self.device = device
        self.num_samples = num_samples
        self.particles_per_hand = particles_per_hand

        # create robot environment
        env = gym.make(
            "CustomEnv-v1",
            robot_uids=robot_uids,
            num_envs=num_samples,
            render_mode="human",
            control_mode="pd_joint_mimic_pos",
            sim_backend=device,
        )
        robot = env.agent
        self.env = env
        self.env.reset()

        self.robot = robot

        urdf_path = robot.urdf_path
        self.urdf_path = urdf_path
        action_names = robot.action_names
        mimic_joint_map = robot.mimic_joint_map
        qpos_names = (
            robot.vis_joint_names
        )  # [j.name for j in robot.robot.get_active_joints()]

        self.mimic_forward = MimicJointForward(
            action_names=action_names,
            qpos_names=qpos_names,
            mimic_joint_map=mimic_joint_map,
            device=device,
        )

        init_action = torch.zeros(1, len(action_names)).to(device)
        init_qpos = self.mimic_forward.forward(init_action)  # (1, n_joints)

        # obtain initial point clouds
        pclds = sample_robot_point_cloud(
            urdf_path,
            num_samples=particles_per_hand,
            qpos=init_qpos.squeeze(0).cpu().numpy(),
        )  # dict[str, torch.Tensor]

        query_link_names = list(pclds.keys())
        self.query_link_names = query_link_names
        pcld_list = [pclds[name] for name in query_link_names]
        pcld_len = torch.cat(
            [
                torch.tensor([0]),
                torch.cumsum(torch.tensor([len(pc) for pc in pcld_list]), dim=0),
            ]
        )
        pclds = torch.cat(pcld_list, dim=0).to(self.device).float()  # (n_points, 3)
        pcld_indices = torch.zeros((len(pclds)), dtype=torch.long).to(self.device)
        for i in range(len(pcld_len) - 1):
            pcld_indices[pcld_len[i] : pcld_len[i + 1]] = i

        finger_names = robot.finger_link_names
        is_finger_masks = torch.zeros(len(pclds), dtype=bool, device=self.device)
        for i, name in enumerate(query_link_names):
            if name in finger_names:
                is_finger_masks[pcld_len[i] : pcld_len[i + 1]] = True
        self.is_finger_masks = is_finger_masks

        self.pclds = pclds
        self.pcld_indices = pcld_indices
        self.link_names = query_link_names

        # retargeting config
        self.mano_layer_default = ManoLayer(
            mano_root=f"{EXTERNAL_DIR}/mano/models",
            side="right",
            use_pca=True,
            ncomps=45,
            flat_hand_mean=False,
        ).to(self.device)

        self.mano_layer = ManoWrapper(config=ManoConfig()).to(self.device)

        # retarget_robot_name, retarget_hand_type = retargeting_robot_name_map[
        #     self.robot.uid
        # ]
        # retargeting_config_path = get_default_config_path(
        #     retarget_robot_name, RetargetingType.position, retarget_hand_type
        # )
        # robot_dir = f"{EXTERNAL_DIR}/dex-retargeting/assets/robots/hands"
        # RetargetingConfig.set_default_urdf_dir(robot_dir)
        # retargeting = RetargetingConfig.load_from_file(retargeting_config_path).build()

        # indices = retargeting.optimizer.target_link_human_indices
        # self.indices = indices
        # self.retargeting = retargeting
        # self.retargeting_to_action_idx = [
        #     retargeting.optimizer.robot.dof_joint_names.index(name)
        #     for name in action_names
        # ]
        # self.retargeting_translate_matrix = torch.tensor(
        #     [[0, -1, 0], [0, 0, 1], [-1, 0, 0]], device=self.device, dtype=torch.float32
        # )
        # self.joint_limits = torch.from_numpy(
        #     self.retargeting.optimizer.robot.joint_limits[
        #         self.retargeting_to_action_idx
        #     ]
        # ).to(self.device).to(torch.float32)

    @property
    def qpos(self):
        # robot_qpos = torch.clip(
        #     self.action[:, 6:], self.joint_limits[:, 0], self.joint_limits[:, 1]
        # )
        robot_qpos = self.action[:, 6:]
        return self.mimic_forward.forward(robot_qpos)

    @property
    def root_pose(self):
        """
        root pose computed from action: (B, 0:3) translation, (B, 3:6) rotation
        return:
            (B, 4, 4) root pose
        """
        return self.xyz_rpg_to_pose(self.action[:, :6])

    def set_init_params(self, action_vec): # init_qpos=None:
        """
        input:
            self.action: torch.Tensor (num_samples, 12)
        """
        # if init_qpos is not None:
        #     self.action[:, 6:] = (
        #         torch.FloatTensor(init_qpos).unsqueeze(0).to(self.device)
        #     )
        # self.action_init = self.action.detach().clone()


        if action_vec.dim() == 1:
            action_vec = action_vec.unsqueeze(0)

        # self.action = action_vec.repeat(self.num_samples, 1).to(self.device)
        self.action = action_vec.to(self.device)
        self.action_init = self.action.detach().clone()

    def reset(self):
        self.action = self.action_init.detach().clone()

    def forward(self):
        """
        return current point cloud
        """
        # with torch.device(self.device):

        self.robot.robot.set_qpos(self.qpos)
        # self.robot.robot.set_state(torch.cat([torch.zeros(1, 13).to(self.device), self.qpos, torch.zeros(1, 10).to(self.device)], dim = -1))

        # self.robot.reset(self.qpos)
        # for i in range(10):
        # self.env.step(self.action[:, 6:])
        # self.env.step(None)
        # self.env.update_scene()
        if self.env.gpu_sim_enabled:
            """
            Warning: Really Important!!!!
            """
            self.env.scene._gpu_apply_all()
            self.robot.scene.px.gpu_update_articulation_kinematics()
            self.env.scene._gpu_fetch_all()
        link_poses = self.robot.get_link_poses(
            self.link_names, tag="link"
        )  # (B, n_links, 4, 4)
        pcld_poses = link_poses[:, self.pcld_indices]  # (B, n_points, 4, 4)

        pcld_poses = torch.einsum("bij,bnjk->bnik", self.root_pose, pcld_poses)

        pcld = (
            torch.einsum("ni, bnji->bnj", self.pclds, pcld_poses[..., :3, :3])
            + pcld_poses[..., :3, 3]
        )  # (B, n_points, 3)

        return pcld

    def state_action_to_pcld_action(self, action):
        """
        input:
            action: torch.Tensor (B, n_actions)
        output:
            current point cloud  (B, n_points, 3)
        """
        self.action += action
        return self.forward()

    def convert(self, action):
        """
        input:
            action: torch.Tensor (B, T, len(action_names)) relative action
        output:
            delta point_cloud: torch.Tensor (B, T, n_points, 3)
            qpos: torch.Tensor (B, T+1, n_joints)
            # is_finger_masks: torch.Tensor (n_points)
        """
        B, T, action_dim = action.shape
        assert B == self.num_samples, "B should be equal to the number of samples"

        verts = torch.zeros(
            B,
            T + 1,
            self.particles_per_hand,
            3,
            device=self.device,
        )
        qposs = torch.zeros(
            B,
            T + 1,
            action_dim,
            device=self.device,
        )

        init_pcld = self.forward()
        verts[:, 0] = init_pcld
        qposs[:, 0] = self.action

        for t in range(T):
            verts[:, t + 1] = self.state_action_to_pcld_action(action[:, t])
            qposs[:, t + 1] = self.action

        return verts[:, 1:] - verts[:, :-1], qposs

    def visualize(self, state, wis3d, name=""):
        """
        input:
            state: torch.Tensor (1, 12)
        """
        assert state.shape[0] == 1, "only support batch size 1"

        qpos = (self.mimic_forward.forward(state[:, 6:])).squeeze(0)
        pose = self.xyz_rpg_to_pose(state[:, :6]).squeeze(0)
        # from dexwm.utils.wis3d_new import Wis3D
        # ERROR:
        wis3d.add_robot(self.urdf_path, qpos.cpu().numpy(), Tw_w2B=pose.cpu().numpy(), name=name)


if __name__ == "__main__":
    data = torch.load("/hdd/yulin/dynamics/Dex-World-Model/dexwm/tester/test.pt").to(
        "cuda"
    )

    from dexwm.utils.wis3d_new import Wis3D
    from manopth.manolayer import ManoLayer

    # robot_pcld_wrapper = RobotPcldWrapper(urdf_path="")
    wis3d = Wis3D(
        out_folder="wis3d",
        sequence_name="robot_pcld_wrapper",
        xyz_pattern=("x", "-y", "-z"),
    )

    mano_layer = ManoLayer(
        mano_root=f"{EXTERNAL_DIR}/mano/models",
        side="right",
        use_pca=True,
        ncomps=45,
        flat_hand_mean=False,
    ).to("cuda")

    T, _ = data.mano_pose.shape

    hand_verts, _ = mano_layer.forward(
        th_betas=data.mano_shape.unsqueeze(0).repeat(T, 1),
        th_pose_coeffs=data.mano_pose[:, :48],
        th_trans=data.mano_pose[:, 48:],
    )
    hand_verts = hand_verts / 1000

    wis3d.add_mesh(hand_verts[0], mano_layer.th_faces, name="hand_mesh")

    robot = RobotPcldWrapper(
        robot_uids="ability_hand_right",
        num_samples=1,
        particles_per_hand=256,
        device="cuda",  # "cuda",
    )

    robot.set_init_params(
        mano_pose=data.mano_pose[0],
        mano_shape=data.mano_shape,
    )

    # robot.action[:, 7] = 0.3
    # while True:
    #     robot.env.render()

    # add initialization
    # robot.action = torch.zeros((robot.num_samples, 12)).to(robot.device)
    # robot.action[:, 7:12] = 0.4

    # print(robot.qpos)

    wis3d.add_robot(robot.urdf_path, robot.qpos[0].cpu().numpy(), name="robot")
    pcld = robot.forward()
    wis3d.add_point_cloud(pcld[0], name="robot_pcld")
    wis3d.add_point_cloud(pcld[0][robot.is_finger_masks], name="finger_pcld")

    robot.visualize(robot.action, wis3d, name="robot_vis")