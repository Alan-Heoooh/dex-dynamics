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

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from manopth.manolayer import ManoLayer
from manopth.tensutils import (
    make_list,
    subtract_flat_id,
    th_pack,
    th_posemap_axisang,
    th_with_zeros,
)
from pytorch3d.transforms import matrix_to_axis_angle
from sapien import Pose

urdf_path = f"{ASSETS_DIR}/ability_hand/ability_hand_right.urdf"

retargeting_robot_name_map = {
    "ability_hand_right": (RobotName.ability, HandType.right),
}


# def keypoint_to_pose(keypoint: np.ndarray) -> Pose:
#     x_axis = (keypoint[9] - keypoint[0]) / np.linalg.norm(
#         keypoint[9] - keypoint[0], keepdims=True
#     )
#     y = keypoint[0] - keypoint[17]
#     y_axis = y - y.dot(x_axis) * x_axis
#     y_axis = y_axis / np.linalg.norm(y_axis)
#     z_axis = np.cross(x_axis, y_axis)
#     rot_matrix = np.stack([x_axis, y_axis, z_axis, keypoint[0]], axis=-1)
#     trans_matrix = np.concatenate([rot_matrix, np.ones((1, 4))], axis=0).astype(
#         np.float32
#     )
#     return Pose(trans_matrix)


def keypoint_to_pose(keypoint: torch.Tensor) -> torch.Tensor:
    """
    input:
        keypoint: torch.Tensor (B, n_keypoints, 3)
    output:
        pose: torch.Tensor (B, 4, 4)
    """
    x_axis = (keypoint[:, 9] - keypoint[:, 0]) / torch.linalg.norm(
        keypoint[:, 9] - keypoint[:, 0], keepdims=True
    )
    y = keypoint[:, 0] - keypoint[:, 17]  # (B, 3)
    y_axis = y - torch.einsum("bi,bi->b", y, x_axis).unsqueeze(1) * x_axis
    y_axis = y_axis / torch.linalg.norm(y_axis, keepdims=True)
    z_axis = torch.cross(x_axis, y_axis, dim=-1)
    rot_matrix = torch.stack(
        [x_axis, y_axis, z_axis, keypoint[:, 0]], dim=-1
    )  # (B, 3, 4)
    trans_matrix = torch.cat(
        [
            rot_matrix,
            torch.ones((1, 1, 4))
            .repeat(rot_matrix.shape[0], 1, 1)
            .to(rot_matrix.device),
        ],
        dim=-2,
    ).float()
    return trans_matrix


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

        # self.action_limits = robot.action_limits

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

        # import pdb; pdb.set_trace()
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

        retarget_robot_name, retarget_hand_type = retargeting_robot_name_map[
            self.robot.uid
        ]
        retargeting_config_path = get_default_config_path(
            retarget_robot_name, RetargetingType.position, retarget_hand_type
        )
        robot_dir = f"{EXTERNAL_DIR}/dex-retargeting/assets/robots/hands"
        RetargetingConfig.set_default_urdf_dir(robot_dir)
        retargeting = RetargetingConfig.load_from_file(retargeting_config_path).build()

        indices = retargeting.optimizer.target_link_human_indices
        # if retargeting_type == "POSITION":
        self.indices = indices
        self.retargeting = retargeting
        # self.retargeting_to_vis_idx = [
        #     retargeting.optimizer.robot.dof_joint_names.index(name)
        #     for name in self.robot.vis_joint_names
        # ]
        self.retargeting_to_action_idx = [
            retargeting.optimizer.robot.dof_joint_names.index(name)
            for name in action_names
        ]
        self.retargeting_translate_matrix = torch.tensor(
            [[0, -1, 0], [0, 0, 1], [-1, 0, 0]], device=self.device, dtype=torch.float32
        )
        self.joint_limits = torch.from_numpy(
            self.retargeting.optimizer.robot.joint_limits[
                self.retargeting_to_action_idx
            ]
        ).to(self.device).to(torch.float32)

    @property
    def qpos(self):
        robot_qpos = torch.clip(
            self.action[:, 6:], self.joint_limits[:, 0], self.joint_limits[:, 1]
        )
        return self.mimic_forward.forward(robot_qpos)

    @property
    def root_pose(self):
        """
        root pose computed from action: (B, 0:3) translation, (B, 3:6) rotation
        return:
            (B, 4, 4) root pose
        """
        return self.xyz_rpg_to_pose(self.action[:, :6])

    def set_init_params(self, action_vec): # init_qpos=None):
        """
        input:
            mano_pose: torch.Tensor (1, 51)
            mano_shape: torch.Tensor (1, 10)
        self.action:
            torch.Tensor (num_samples, 12)
        """
        # if init_qpos is not None:
        #     self.action[:, 6:] = (
        #         torch.FloatTensor(init_qpos).unsqueeze(0).to(self.device)
        #     )
        # self.action_init = self.action.detach().clone()


        if action_vec.dim() == 1:
            action_vec = action_vec.unsqueeze(0)

        self.action = action_vec.repeat(self.num_samples, 1).to(self.device)
        self.action_init = self.action.detach().clone()

    def retarget_from_hand_action(self, action, mano_shape):
        """
        input:
            action: torch.Tensor (1, 12) or (12)
        output:
            matrix: (4, 4), qpos: (n_joints)
        """
        if mano_shape.dim() == 1:
            mano_shape = mano_shape.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        B = action.shape[0]

        if mano_shape.shape[0] != B:
            mano_shape = mano_shape.repeat(B, 1)

        # from wis3d import Wis3D

        # wis3d = Wis3D(
        #     out_folder="wis3d",
        #     sequence_name="robot_pcld_wrapper",
        #     xyz_pattern=("x", "-y", "-z"),
        # )

        # hand_verts, _, _ = self.mano_layer_hand.forward(
        #     th_betas=mano_shape,
        #     th_pose_coeffs=action[:, 3:].detach().clone(),
        #     th_trans=action[:, :3].detach().clone(),
        # )
        # wis3d.add_mesh(
        #     hand_verts.squeeze(0),
        #     self.mano_layer_hand.th_faces,
        #     name="origin_hand_mesh",
        # )

        mano_pose = action[:, 3:].detach().clone()
        mano_pose[:, :3] = 0
        hand_verts, joint_pos, _ = self.mano_layer_hand.forward(
            th_betas=mano_shape,
            th_pose_coeffs=mano_pose,
            th_trans=torch.zeros((B, 3), device=self.device),
        )
        # assert (
        #     mano_shape.shape[0] == 1
        # ), "only support batch size 1 for hand shape retargeting"
        # assert (
        #     action.shape[0] == self.num_samples
        # ), f"action batch size {action.shape[0]} should be equal to num_samples {self.num_samples}"

        # retarget
        # action = self.retarget_from_hand_action(action, mano_shape)
        joint_pos = joint_pos.squeeze(0)
        trans_joint_pos = torch.einsum(
            "ij,bj->bi", self.retargeting_translate_matrix, joint_pos
        )

        trans_hand_verts = torch.einsum(
            "ij,bj->bi", self.retargeting_translate_matrix, hand_verts.squeeze(0)
        )
        # wis3d.add_mesh(
        #     trans_hand_verts.squeeze(0),
        #     self.mano_layer_hand.th_faces,
        #     name="rotated_centered_hand_mesh",
        # )

        # wis3d.add_mesh(
        #     hand_verts.squeeze(0), self.mano_layer.th_faces, name="centered_hand_mesh"
        # )

        ref_value = trans_joint_pos.squeeze(0)[self.indices, :]
        # else:
        #     origin_indices = indices[0, :]
        #     task_indices = indices[1, :]
        #     ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

        qpos = torch.from_numpy(self.retargeting.retarget(ref_value.cpu().numpy())).to(
            self.device
        )
        vis_qpos = torch.cat(
            [
                torch.zeros(6, device=self.device),
                (
                    self.mimic_forward.forward(
                        qpos[self.retargeting_to_action_idx].unsqueeze(0).float()
                    )
                ).squeeze(0),
            ]
        )

        # vis_qpos = self.mimic_forward.forward(
        #     qpos[self.retargeting_to_action_idx].unsqueeze(0).float()
        # ).squeeze(0)
        # robot_action = torch.zeros(1, 12).to(self.device)
        # robot_action[:, 6:] = qpos[self.retargeting_to_action_idx]

        # self.visualize(robot_action, wis3d, name="origin retargeted_hand")
        # robot_action[:, :6] = qpos[:6]
        # self.visualize(robot_action, wis3d, name="rotated_hand")

        _, rot = th_posemap_axisang(qpos.float().unsqueeze(0)[:, 3:6])
        rot = torch.einsum(
            "bij,jk->bik", rot.view(-1, 3, 3), self.retargeting_translate_matrix.T
        )
        # robot_action[:, 3:6] = matrix_to_axis_angle(rot)
        # robot_action[:, :3] = torch.einsum(
        #     "bij, bj->bi", rot.view(-1, 3, 3), robot_action[:, :3]
        # )
        # self.visualize(robot_action, wis3d, name="translated_hand")

        hand_verts_origin, hand_joints_origin, _ = self.mano_layer_hand.forward(
            th_betas=mano_shape,
            th_pose_coeffs=action[:, 3:],
            th_trans=action[:, :3],
        )
        # wis3d.add_mesh(
        #     hand_verts_origin.squeeze(0),
        #     self.mano_layer_hand.th_faces,
        #     name="hand_mesh",
        # )

        _, rot_1 = th_posemap_axisang(action[:, 3:6])
        rot_final = torch.einsum(
            "bij,bjk->bik",
            rot_1.view(-1, 3, 3),
            rot,  # self.retargeting_translate_matrix.T
        )
        rpg = matrix_to_axis_angle(rot_final)
        # self.visualize(robot_action, wis3d, name="rotated_hand")

        xyz = hand_joints_origin[:, 0] + torch.einsum(
            "bij, j->bi", rot_final, qpos.float()[:3] - trans_joint_pos[0]
        )
        # self.visualize(robot_action, wis3d, name="final_robot")

        # action[:, :6] = xyz_rpg
        # rpg = matrix_to_axis_angle(rot)
        # action[:, :3] = hand_joints_origin[0, :3]
        # wrist_matrix = torch.cat([])
        wrist_matrix = self.xyz_rpg_to_pose(
            torch.cat([xyz, rpg], dim=-1)
            # torch.cat([hand_joints_origin[:, 0, :3], rpg], dim=-1)
        )

        # self.visualize(action, wis3d, name="retargeted_robot")
        return wrist_matrix.squeeze(0), vis_qpos

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
        # self.env.update_scene()# step(None)
        if self.env.gpu_sim_enabled:
            """
            Warning: Really Important!!!!
            """
            self.env.scene._gpu_apply_all()
            self.robot.scene.px.gpu_update_articulation_kinematics()
            self.env.scene._gpu_fetch_all()
        # torch.cuda.synchronize()
        # while True:
        #     self.env.render()

        # self.env.base_env.viewer.paused = True
        # self.
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
        B, T, _ = action.shape
        assert B == self.num_samples, "B should be equal to the number of samples"

        verts = torch.zeros(
            B,
            T + 1,
            self.particles_per_hand,
            3,
            device=self.device,
        )
        qposs = torch.zeros(B, T + 1, 12, device=self.device)

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