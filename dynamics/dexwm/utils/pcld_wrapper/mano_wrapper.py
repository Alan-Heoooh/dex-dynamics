import os

import mano
import numpy as np
import torch
from mano.webuser.smpl_handpca_wrapper_HAND_only import ready_arguments
from manopth import rodrigues_layer, rot6d, rotproj
from manopth.manolayer import ManoLayer
from manopth.tensutils import (
    make_list,
    subtract_flat_id,
    th_pack,
    th_posemap_axisang,
    th_with_zeros,
)

# from matplotlib import axis
from pytorch3d.transforms import matrix_to_axis_angle
from torch.nn import Module
from dexwm.utils.macros import EXTERNAL_DIR
from dataclasses import dataclass


@dataclass
class ManoConfig:
    flat_hand_mean: bool = False
    use_pca: bool = True
    robust_rot: bool = False
    side: str = "right"
    ncomps: int = 6
    root_rot_mode: str = "axisang"
    joint_rot_mode: str = "axisang"
    # n_pick: int = 20
    clip_range: float = 2.0
    center_idx: int | None = None


class ManoWrapper(ManoLayer):
    def __init__(self, config: ManoConfig):
        super().__init__(
            center_idx=config.center_idx,  # (None,)
            flat_hand_mean=config.flat_hand_mean,  # (True,)
            ncomps=config.ncomps,  # (6,)
            side=config.side,  # ("right",)
            mano_root=f"{EXTERNAL_DIR}/mano/models",  # ("mano/models",)
            use_pca=config.use_pca,  # (True,)
            root_rot_mode=config.root_rot_mode,  # ("axisang",)
            joint_rot_mode=config.joint_rot_mode,  # ("axisang",)
            robust_rot=config.robust_rot,
        )  # (False,)

        selected_components = self.th_selected_comps.cpu().numpy()
        selected_components_inv = selected_components.T / np.diag(
            selected_components @ selected_components.T
        )
        self.register_buffer(
            "th_selected_comps_inv", torch.Tensor(selected_components_inv)
        )

    def forward(
        self,
        th_pose_coeffs,
        th_betas=torch.zeros(1),
        th_trans=torch.zeros(1),
        # is_right=torch.Tensor([1.0]),
        root_palm=torch.Tensor([0]),
        share_betas=torch.Tensor([0]),
    ):
        
        
        verts, joints = super().forward(
            th_pose_coeffs, th_betas, torch.zeros(1), root_palm, share_betas
        )
        verts = verts / 1000
        joints = joints / 1000

        origin_joints = joints + th_trans.unsqueeze(1)

        # verts[..., 0] = (2 * is_right.unsqueeze(1) - 1) * verts[..., 0]
        # joints[..., 0] = (2 * is_right.unsqueeze(1) - 1) * joints[..., 0]

        verts = verts + th_trans.unsqueeze(1) # - joints[:, 0:1, :]
        joints = joints + th_trans.unsqueeze(1) # - joints[:, 0:1, :]
        return verts, joints, origin_joints

    def convert_to_axisangle(self, th_pose_coeffs: torch.Tensor, input_format="rotmat"):
        """
        Convert pose coefficients to axis-angle representation
        th_pose_coeffs: [batch_size, 16, 3, 3] or [batch_size, 48]
        """
        if input_format == "rotmat":
            axis_angle = matrix_to_axis_angle(th_pose_coeffs).reshape(-1, 48)
            root_rot = axis_angle[:, :3]
            hand_pose = (axis_angle[..., 3:] - self.th_hands_mean).mm(
                self.th_selected_comps_inv
            )
        elif input_format == "axisang":
            axis_angle = th_pose_coeffs
            root_rot = axis_angle[:, :3]
            hand_pose = axis_angle[..., 3:].mm(self.th_selected_comps_inv)
        else:
            raise ValueError("Invalid input format")

        return torch.cat([root_rot, hand_pose], 1)

    @property
    def hand_faces(self) -> torch.Tensor:
        return self.th_faces

    # @property
    def hand_faces_rol(self, right) -> torch.Tensor:
        return self.th_faces if right else self.th_faces[..., [0, 2, 1]]


if __name__ == "__main__":
    from wis3d import Wis3D
    from yacs.config import CfgNode as CN

    from dexrl.utils.path import PACKAGE_EXTERNAL_DIR

    _C = CN()
    _C.mano_root = f"{PACKAGE_EXTERNAL_DIR}/mano/models"
    _C.flat_hand_mean = False
    _C.side = "right"
    _C.ncomps = 6
    _C.root_rot_mode = "axisang"
    _C.joint_rot_mode = "axisang"
    _C.center_idx = None
    _C.use_pca = True
    _C.robust_rot = False

    mano_layer = ManoWrapper(_C).to("cuda")

    params = torch.load("/hdd/yulin/manohand/dexrl/mano_params.pt")
    pred = torch.load("/hdd/yulin/manohand/dexrl/mano_pred.pt")

    wis3d = Wis3D(
        out_folder="wis3d", sequence_name="debug_mano", xyz_pattern=("x", "-y", "-z")
    )
    wis3d.add_mesh(
        pred["j3d"][0],
        mano_layer.th_faces,
        name="mano_mesh",
    )

    # params: {"betas": (1, 10), "hand_pose": (1, 15, 3, 3), "global_orient": (1, 1, 3, 3)"
    shape_vector = params["betas"]  # .cpu()
    hand_pose = params["hand_pose"]  # .cpu()
    global_orient = params["global_orient"]  # .cpu()

    hand_pose_rotmat = torch.cat([global_orient, hand_pose], 1)  # .view(1, 15, 3, 3)

    axis_angle = mano_layer.convert_to_axisangle(hand_pose_rotmat)

    verts, joints = mano_layer(axis_angle, shape_vector)

    wis3d.add_mesh(
        verts[0],
        mano_layer.th_faces,
        name="forward mano_mesh",
    )
