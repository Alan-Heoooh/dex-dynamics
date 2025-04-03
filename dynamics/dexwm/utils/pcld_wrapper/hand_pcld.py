import torch
import numpy as np

# from manopth.manolayer import ManoLayer
from .mano_wrapper import ManoWrapper, ManoConfig
from dexwm.utils.macros import EXTERNAL_DIR, ASSETS_DIR
from dexwm.utils.sample import furthest_point_sampling
import pickle
from manopth.tensutils import th_posemap_axisang
from pytorch3d.transforms import matrix_to_axis_angle


class PcldWrapper:
    is_finger_masks: torch.BoolTensor = None
    # is_inner_masks: torch.BoolTensor = None
    is_hand_part_masks: list[torch.BoolTensor] = None
    is_palm_masks: torch.BoolTensor = None

    def __init__(self, **kwargs):
        pass

    def sample(self, **kwargs):
        raise NotImplementedError

    def set_init_params(self, **kwargs):
        raise NotImplementedError

    def retarget(self, **kwargs):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def forward(self):
        """
        return current point cloud
        """
        raise NotImplementedError

    def state_action_to_pcld_action(self, action):
        """
        convert state action to point cloud action
        input:
            action: torch.Tensor (B, len(action_names))
        output:
            pcld_action: torch.Tensor (B, T, n_points, 3)
            qpos: torch.Tensor (B, T, n_joints)
        """
        raise NotImplementedError

    def convert(self, action):
        """
        input:
            action: torch.Tensor (B, T, len(action_names))
        output:
            delta point_cloud: torch.Tensor (B, T, n_points, 3)
        """
        raise NotImplementedError

    def visualize(self, state, wis3d):
        raise NotImplementedError

    @staticmethod
    def xyz_rpg_to_pose(xyz_rpg):
        """
        input:
            xyz_rpg: torch.Tensor (B, 6)
        output:
            pose: torch.Tensor (B, 4, 4)
        """
        xyz = xyz_rpg[:, :3]
        rpg = xyz_rpg[:, 3:]
        _, rot = th_posemap_axisang(rpg)
        pose = torch.cat(
            [
                torch.cat([rot.view(-1, 3, 3), xyz.unsqueeze(-1)], dim=-1),
                torch.tensor([[0, 0, 0, 1]], device=xyz_rpg.device)
                .unsqueeze(0)
                .repeat(xyz_rpg.shape[0], 1, 1),
            ],
            dim=-2,
        )
        return pose

    @staticmethod
    def pose_to_xyz_rpg(pose):
        """
        input:
            pose: torch.Tensor (B, 4, 4)
        output:
            xyz_rpg: torch.Tensor (B, 6)
        """
        xyz = pose[:, :3, 3]
        rot = pose[:, :3, :3]
        rpg = matrix_to_axis_angle(rot)  # (B, 3)
        xyz_rpg = torch.cat([xyz, rpg], dim=-1)
        return xyz_rpg


class HandPcldWrapper(PcldWrapper):
    # _initial_can_pose = torch.tensor(
    #     [1.3571, -0.3359, 0.3821, 1.1535, -0.1231, -0.3064]
    # ).reshape(-1, 6)
    finger_name = ["thumb", "index", "middle", "ring", "pinky"]

    def __init__(
        self,
        particles_per_hand,
        num_samples,
        device="cuda",
    ):
        super().__init__()

        self.device = device
        self.num_samples = num_samples
        self.particles_per_hand = particles_per_hand

        self.mano_layer = ManoWrapper(config=ManoConfig()).to(self.device)

        # sample initial hand point cloud
        mano_flat_layer = ManoWrapper(config=ManoConfig(flat_hand_mean=True)).to(
            self.device
        )
        # hand_verts_0, _ = self.mano_forward(torch.zeros(1, 12, device=self.device))
        hand_verts_0, hand_joints, _ = mano_flat_layer.forward(
            th_betas=None,
            th_pose_coeffs=torch.zeros(1, 9, device=self.device),
            th_trans=torch.zeros(1, 3, device=self.device),
        )
        self.hand_verts_0 = hand_verts_0

        self.finger_region = np.load(f"{ASSETS_DIR}/fingers.npy")

        # with open(f"{EXTERNAL_DIR}/obman_train/assets/contact_zones.pkl", "rb") as p_f:
        #     contact_data = pickle.load(p_f)
        # palm_region = np.array([i for i in range(96) if i not in contact_region])
        # inner_region = np.load(f"{ASSETS_DIR}/inner.npy")
        # is_inner_mask = np.isin(hand_sample_indices, inner_region)

        # palm_region = np.load(f"{ASSETS_DIR}/palm.npy")
        # is_palm_mask = np.isin(hand_sample_indices, palm_region)

        contact_region = torch.load(f"{ASSETS_DIR}/hand_part.pt")
        self.contact_region = contact_region
        self.palm_region = contact_region["base_link"]
        self.separate_finger_region = [
            contact_region[f"{finger_name}_link"] for finger_name in self.finger_name
        ]

        self.sample()

        # finger_names = ["thumb", "index",]
        # self.extract_finger_region(finger_names)

        assert self.hand_sample_indices.shape[0] == self.particles_per_hand, "hand sample indices should be equal to particles per hand"

    def single_finger_region(self, finger_name):
        return self.contact_region[f"{finger_name}_link"]
    
    def extract_finger_region(self, finger_names):

        hand_sample_indices = []
        for finger_name in finger_names:
            hand_sample_indices.append(self.single_finger_region(finger_name))

        self.hand_sample_indices = np.concatenate(hand_sample_indices)

        self.hand_sample_indices = torch.IntTensor(self.hand_sample_indices).to(self.device)

    def sample(self):
        """sample initial hand point cloud"""
        _, hand_sample_indices = furthest_point_sampling(
            self.hand_verts_0.cpu().numpy().reshape(-1, 3), self.particles_per_hand
        )

        is_finger_masks = np.isin(hand_sample_indices, self.finger_region)
        self.is_finger_masks = torch.BoolTensor(is_finger_masks).to(self.device)

        # self.is_inner_masks = torch.BoolTensor(is_inner_mask).to(self.device)
        is_palm_mask = np.isin(hand_sample_indices, self.palm_region)
        self.is_palm_masks = torch.BoolTensor(is_palm_mask).to(self.device)

        separate_finger_masks = [is_palm_mask]
        for i in range(5):
            # finger_name = self.finger_name[i]
            separate_finger_masks.append(
                np.isin(hand_sample_indices, self.separate_finger_region[i])
            )

        self.is_hand_part_masks = [
            torch.BoolTensor(mask).to(self.device) for mask in separate_finger_masks
        ]

        self.hand_sample_indices = torch.IntTensor(hand_sample_indices).to(self.device)

    @property
    def hand_faces(self):
        return self.mano_layer.th_faces

    def set_init_params(self, mano_shape, action_vec):  # mano_pose, mano_shape):
        """
        input:
            mano_shape: torch.Tensor (1, 10)
            mano_pose: torch.Tensor (1, 51)
        """
        if mano_shape.dim() == 1:
            mano_shape = mano_shape.unsqueeze(0)
        if action_vec.dim() == 1:
            action_vec = action_vec.unsqueeze(0)

        # self.shape_vec = mano_shape
        # # self.trans_vec = mano_pose[:, 48:]
        # self.pose_vec = self.retarget(mano_pose, mano_shape).repeat(
        #     self.num_samples, 1
        # )  # (1, 12)
        # self.pose_vec[:, 6:] = self._initial_can_pose.to(self.device)

        self.pose_vec = action_vec.repeat(self.num_samples, 1).to(self.device)
        self.shape_vec = mano_shape.to(self.device)

        # self.init_trans_vec = self.trans_vec
        self.init_pose_vec = self.pose_vec.detach().clone()
        self.init_shape_vec = self.shape_vec.detach().clone()

        # sample initial hand point cloud
        # hand_verts = self.mano_forward()

        # _, hand_sample_indices = furthest_point_sampling(
        #     hand_verts.cpu().numpy().reshape(-1, 3), self.particles_per_hand
        # )
        # self.hand_sample_indices = torch.IntTensor(hand_sample_indices).to(self.device)

    def retarget(self, mano_pose, mano_shape, transform_pose):
        """
        input:
            mano_pose: torch.Tensor (B, 51)
            mano_shape: torch.Tensor (B, 10)
            transform_pose: torch.Tensor (4, 4)

        return:
            action: torch.Tensor (B, 12)
        """
        if mano_shape.dim() == 1:
            mano_shape = mano_shape.unsqueeze(0)
        if mano_pose.dim() == 1:
            mano_pose = mano_pose.unsqueeze(0)

        pose_vec = torch.cat(
            [
                mano_pose[:, 48:],
                self.mano_layer.convert_to_axisangle(
                    mano_pose[:, :48], input_format="axisang"
                ),
            ],
            dim=-1,
        )  # .repeat(self.num_samples, 1)  # (1, 12)

        # if transform_pose is not None:
        origin_verts, _, _ = self.mano_layer.forward(
            th_betas=mano_shape,
            th_pose_coeffs=pose_vec[:, 3:],
            th_trans=pose_vec[:, :3],
        )
        trans_verts = torch.einsum(
            "ij, kj->ik", origin_verts[:, 0], transform_pose[:3, :3]
        ) + transform_pose[:3, 3].unsqueeze(0)

        # wrist_rot = pose_vec[:, 3:6]
        # _, rot_mat = th_posemap_axisang(wrist_rot)
        # wrist_rot = torch.einsum("ij,njk->nik", transform_pose[:3, :3], rot_mat.view(-1, 3, 3))
        # pose_vec[:, 3:6] = matrix_to_axis_angle(wrist_rot)
        # pose_vec[:, :3] += transform_pose[:3, 3]

        wrist_pose = self.xyz_rpg_to_pose(pose_vec[:, :6])
        wrist_pose = torch.einsum("ij, kjl->kil", transform_pose, wrist_pose)
        pose_vec[:, 3:6] = self.pose_to_xyz_rpg(wrist_pose)[:, 3:]

        new_verts, _, _ = self.mano_layer.forward(
            th_betas=mano_shape,
            th_pose_coeffs=pose_vec[:, 3:],
            th_trans=pose_vec[:, :3],
        )
        pose_vec[:, :3] += trans_verts - new_verts[:, 0, :]

        return pose_vec

    def reset(self):
        self.shape_vec = self.init_shape_vec.detach().clone()
        # self.trans_vec = self.init_trans_vec
        self.pose_vec = self.init_pose_vec.detach().clone()

    def mano_forward(self, pose_vec=None):
        if pose_vec is None:
            pose_vec = self.pose_vec

        if getattr(self, "shape_vec", None) is None:
            shape_vec = None
        else:
            shape_vec = self.shape_vec.repeat(pose_vec.shape[0], 1)

        hand_verts, hand_joints, _ = self.mano_layer.forward(
            th_betas=shape_vec,
            th_pose_coeffs=pose_vec[:, 3:],
            th_trans=pose_vec[:, :3],
        )
        return hand_verts, hand_joints

    def forward(self):
        return self.mano_forward()[0][:, self.hand_sample_indices]

    def state_action_to_pcld_action(self, action):
        """
        input:
            action: torch.Tensor (B, len(action))
        """
        self.pose_vec += action
        # self.pose_vec = self.pose_vec + action[:, 3:]  # (B, 9)
        # self.trans_vec = self.trans_vec + action[:, :3]  # (B, 3)
        # if self.shape_vec.shape[0] != action.shape[0]:
        #     self.shape_vec = self.shape_vec.repeat(action.shape[0], 1)

        return self.mano_forward()

    def convert(self, action):
        """
        input:
            action: torch.Tensor (B, T, len(action))
        return:
            pcld_action: torch.Tensor(B, T, n_points, 3)
            hand_verts: torch.Tensor (B, T+1, 778, 3)
            # is_finger_masks: torch.Tensor (n_points)

        """
        B, T, _ = action.shape

        # hand_pcld = torch.zeros(B, T, self.particles_per_hand, 3, device=self.device)
        hand_verts = torch.zeros(B, T + 1, 778, 3, device=self.device)
        hand_poses = torch.zeros(B, T + 1, 12, device=self.device)
        hand_joints = torch.zeros(B, T + 1, 21, 3, device=self.device)

        # init_hand_verts = self.forward()  # (1, 778, 3)
        init_hand_verts, init_hand_joints = self.mano_forward()
        hand_verts[:, 0] = init_hand_verts
        hand_poses[:, 0] = self.pose_vec
        hand_joints[:, 0] = init_hand_joints

        for t in range(T):
            hand_verts[:, t + 1], hand_joints[:, t + 1] = (
                self.state_action_to_pcld_action(action[:, t])
            )
            hand_poses[:, t + 1] = self.pose_vec

        return (
            (hand_verts[:, 1:] - hand_verts[:, :-1])[:, :, self.hand_sample_indices],
            {
                "states": hand_poses,
                "joints": hand_joints,
                "verts": hand_verts,
            },
        )

    def visualize(self, state, wis3d, name=""):
        """
        input:
            state: torch.Tensor (1, 12)
        """
        assert state.shape[0] == 1, "only support batch size 1"

        hand_verts, _ = self.mano_forward(state)  # .squeeze(0)  # (778, 3)
        hand_verts = hand_verts.squeeze(0)
        # print(hand_verts.shape)
        wis3d.add_mesh(hand_verts, self.hand_faces, name=name)
        # wis3d.add_point_cloud(hand_verts[0][self.hand_sample_indices], name="hand_pcld")

    def convert_to_mano_parameters(self, action):
        """
        input:
            action: torch.Tensor (B, 12)
        output:
            mano_parameters: torch.Tensor (B, 51)
        """
        trans = action[:, :3]
        root_rot = action[:, 3:6]
        hand_pose = action[:, 6:].mm(self.mano_layer.th_selected_comps)
        return torch.cat([root_rot, hand_pose, trans], dim=-1)
        # pose_vec = torch.cat(
        #     [
        #         mano_pose[:, 48:],
        #         self.mano_layer.convert_to_axisangle(
        #             mano_pose[:, :48], input_format="axisang"
        #         ),
        #     ],
        #     dim=-1,
        # )  # .repeat(self.num_samples, 1)  # (1, 12)
        # return pose_vec


if __name__ == "__main__":
    data = torch.load("/hdd/yulin/dynamics/Dex-World-Model/dexwm/tester/test.pt").to(
        "cuda"
    )
    hand_pcld_wrapper = HandPcldWrapper(particles_per_hand=40, num_samples=1)
    from manopth.manolayer import ManoLayer
    import trimesh
    from wis3d import Wis3D

    wis3d = Wis3D(
        out_folder="wis3d",
        sequence_name="hand pcld wrapper",
        xyz_pattern=("x", "-y", "-z"),
    )

    mano_layer = ManoLayer(
        mano_root=f"{EXTERNAL_DIR}/mano/models",
        side="right",
        use_pca=True,
        ncomps=45,
        flat_hand_mean=False,
    )
    mano_layer = mano_layer.to("cuda")

    mesh = trimesh.load(data.obj_mesh_path[0])
    obj_verts = torch.tensor(mesh.vertices).float().to("cuda")

    T, _ = data.mano_pose.shape

    hand_verts, _ = mano_layer.forward(
        th_betas=data.mano_shape.unsqueeze(0).repeat(T, 1),
        th_pose_coeffs=data.mano_pose[:, :48],
        th_trans=data.mano_pose[:, 48:],
    )
    hand_verts = hand_verts / 1000

    hand_pcld_wrapper.set_init_params(
        mano_pose=data.mano_pose[0], mano_shape=data.mano_shape
    )

    hand_verts_simplified = hand_pcld_wrapper.mano_forward()

    obj_pose = data.obj_pose
    obj_pcld = torch.einsum("ij, nkj->nik", obj_verts, obj_pose[:, :3, :3]) + obj_pose[
        :, :3, 3
    ].unsqueeze(1)

    wis3d.add_mesh(hand_verts[0], mano_layer.th_faces, name="hand_mesh_origin")

    wis3d.add_mesh(
        hand_verts_simplified.reshape(-1, 3),
        mano_layer.th_faces,
        name="hand_mesh_simpliefied",
    )

    wis3d.add_point_cloud(
        hand_verts_simplified.reshape(-1, 3)[hand_pcld_wrapper.hand_sample_indices],
        name="hand_pcld",
    )

    wis3d.add_point_cloud(
        hand_verts_simplified.reshape(-1, 3)[hand_pcld_wrapper.hand_sample_indices][
            hand_pcld_wrapper.is_finger_masks
        ],
        name="hand_fingers",
    )

    wis3d.add_point_cloud(
        hand_verts_simplified.reshape(-1, 3)[hand_pcld_wrapper.hand_sample_indices][
            hand_pcld_wrapper.is_inner_masks
        ],
        name="hand_palm",
    )

    hand_pcld_wrapper.visualize(
        hand_pcld_wrapper.pose_vec, wis3d, name="hand_visualzie"
    )

    with open(f"{EXTERNAL_DIR}/obman_train/assets/contact_zones.pkl", "rb") as p_f:
        contact_data = pickle.load(p_f)

    contact_region = contact_data["contact_zones"]
    for i in range(len(contact_region)):
        wis3d.add_point_cloud(
            hand_verts_simplified.reshape(-1, 3)[contact_region[i]],
            name=f"contact_region_{i}",
        )

    # obtain a good initialize hand pose for hand pose searching
    pose = np.load("/hdd/yulin/dexrl/dexrl/model/hand.npy")  # (1, )
    rot = np.load("/hdd/yulin/dexrl/dexrl/model/rot.npy")
    init_hand_parameters = torch.from_numpy(np.concatenate([rot, pose], axis=1)).to(
        "cuda"
    )  # (1, 16, 3, 3)
    mano_pose = hand_pcld_wrapper.mano_layer.convert_to_axisangle(
        init_hand_parameters, input_format="rotmat"
    )
    hand_pcld_wrapper.pose_vec[:, 3:] = mano_pose
    hand_pcld_wrapper.visualize(
        hand_pcld_wrapper.pose_vec, wis3d, name="hand_visualzie"
    )
    print(mano_pose)
    # for i in range(T):
    #     wis3d.add_mesh(
    #         hand_verts[i],
    #         mano_layer.th_faces,
    #         name="hand_mesh",
    #     )
    #     wis3d.add_mesh(
    #         obj_pcld[i],
    #         torch.tensor(mesh.faces),
    #         name="obj_mesh",
    #     )
    #     wis3d.increase_scene_id()