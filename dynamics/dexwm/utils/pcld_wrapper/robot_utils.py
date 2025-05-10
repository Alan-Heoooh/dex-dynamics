# from wis3d import Wis3D, utils_3d
import numpy as np
import warnings
import trimesh
import torch
import os
import transforms3d
import warnings

import numpy as np
import sapien

from dexwm.utils.sample import furthest_point_sampling
from dexwm.utils.wis3d_new import Wis3D
from dexwm.utils import utils_3d

try:
    import sapien

    from dexwm.utils.wis3d_new import (
        SAPIENKinematicsModelStandalone,
    )
except ImportError:
    print("SAPIEN is not installed, SAPIENKinematicsModelStandalone is not available.")

    raise ImportError("Please install sapien first.")


def sample_surface(
    mesh: trimesh.Trimesh, counts: int, return_index: bool = False
) -> np.ndarray:
    samples = trimesh.sample.sample_surface(mesh, counts)
    if return_index:
        return samples[0], samples[1]
    return samples[0]


def sample_robot_point_cloud(
    urdf_path, num_samples, link_names=None, qpos=None, save_path=None
):
    meshes, robot_poses = return_mesh(
        urdf_path, link_names=link_names, qpos=qpos
    )  # dict[str, trimesh.Trimesh]

    # sample uniformly on the surface of the meshes, and then run furthest point sampling
    sampled_pcld = []
    transformed_pcld = []
    sampled_indices = {}

    n_samples = 0
    for link_name, mesh in meshes.items():
        sampled_indices[link_name] = np.arange(n_samples, n_samples + 128)
        n_samples += 128

        robot_pose = robot_poses[link_name]
        # mesh.vertices = utils_3d.transform_points(mesh.vertices, robot_pose)
        # trans
        sampled_pcld.append(sample_surface(mesh, 128))
        transformed_pcld.append(utils_3d.transform_points(sampled_pcld[-1], robot_pose))

    sampled_pcld = np.concatenate(sampled_pcld, axis=0)
    transformed_pcld = np.concatenate(transformed_pcld, axis=0)  # (N, 3)
    selected_pcld, selected_indices = furthest_point_sampling(
        transformed_pcld, num_samples
    )

    # find the corresponding link name from selected_indices
    pclds = {
        link_name: sampled_pcld[selected_indices[np.isin(selected_indices, indices)]]
        for link_name, indices in sampled_indices.items()
    }

    # for k, v in pclds.items():
    #     wis3d.add_point_cloud(v, name=k)
    pclds = {k: torch.from_numpy(v) for k, v in pclds.items()}
    if save_path is not None:
        save_path = os.path.join(save_path, f"{num_samples}_pc.pt")
        torch.save(pclds, save_path)

    return pclds


def return_mesh(
    urdf_path: str,
    link_names: list[str],
    qpos: list[float] = None,
    Tw_w2B=None,
    name="",
):
    # if urdf_path not in Wis3D.urdf_caches:
    #     Wis3D.urdf_caches[urdf_path] =
    if Tw_w2B is None:
        Tw_w2B = np.eye(4)

    sk: SAPIENKinematicsModelStandalone = SAPIENKinematicsModelStandalone(
        urdf_path
    )  # Wis3D.urdf_caches[urdf_path]
    links = sk.robot.get_links()
    if qpos is None:
        warnings.warn("qpos is not provided, using default qpos.")
        qpos = np.zeros(len(sk.robot.get_active_joints()))
    if len(qpos) < len(sk.robot.get_active_joints()):
        warnings.warn(
            f"qpos is not complete {len(qpos)} < {len(sk.robot.get_active_joints())}, filling the rest with 0."
        )
        qpos = np.asarray(qpos).tolist() + [0] * (
            len(sk.robot.get_active_joints()) - len(qpos)
        )
    # local_pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]) * 0.05
    # joint_mesh = None

    return_meshes = {}
    robot_poses = {}
    if link_names is None:
        link_names = [link.name for link in links]
    for link_index, link in enumerate(links):
        link_name = link.name
        if link_name not in link_names:
            continue
        pq = sk.compute_forward_kinematics(qpos, link_index)
        link_pose = utils_3d.Rt_to_pose(transforms3d.quaternions.quat2mat(pq.q), pq.p)
        robot_poses[link_name] = Tw_w2B @ link_pose

        # pose = utils_3d.Rt_to_pose(transforms3d.quaternions.quat2mat(pq.q), pq.p)
        pose = np.eye(4)
        components = link.get_entity().get_components()
        for component in components:
            if isinstance(component, sapien.pysapien.render.RenderBodyComponent):
                local_pose = component.render_shapes[
                    0
                ].local_pose.to_transformation_matrix()
                pose = pose @ local_pose
                if hasattr(component.render_shapes[0], "filename"):
                    mesh_path = component.render_shapes[0].filename
                    mesh = trimesh.load(mesh_path, force="mesh")
                    # add_name = link_name if name == "" else name + "_" + link_name
                    m = trimesh.Trimesh(
                        utils_3d.transform_points(mesh.vertices, pose),
                        mesh.faces,
                    )
                    return_meshes[link_name] = m
                    # if joint_mesh is None:
                    #     joint_mesh = m
                    # else:
                    #     joint_mesh = trimesh.util.concatenate(joint_mesh, m)
                    # wis3d.add_mesh(m, name=add_name)
                    # axis_in_base = utils_3d.transform_points(local_pts, Tw_w2B @ pose)
                    # if add_local_coord:
                    #     wis3d.add_coordinate_transformation(
                    #         Tw_w2B @ pose, name=f"{link_name}_coord"
                    #     )
                    # self.add_lines(axis_in_base[0], axis_in_base[1], name=f'{link_name}_x')
                    # self.add_lines(axis_in_base[0], axis_in_base[2], name=f'{link_name}_y')
                    # self.add_lines(axis_in_base[0], axis_in_base[3], name=f'{link_name}_z')
    return return_meshes, robot_poses


def get_robot_poses(
    urdf_path: str,
    link_names: list[str] = None,
    qpos: np.ndarray | list[float] = None,
    Tw_w2B: np.array = None,
) -> np.array:
    """
    Get the poses of the robot links in the world frame.
    return:
        pose: (num_links, 4, 4)
    """
    if urdf_path not in Wis3D.urdf_caches:
        Wis3D.urdf_caches[urdf_path] = SAPIENKinematicsModelStandalone(urdf_path)
    if Tw_w2B is None:
        Tw_w2B = np.eye(4)

    sk: SAPIENKinematicsModelStandalone = Wis3D.urdf_caches[urdf_path]
    links = sk.robot.get_links()
    if qpos is None:
        warnings.warn("qpos is not provided, using default qpos.")
        qpos = np.zeros(len(sk.robot.get_active_joints()))
    if len(qpos) < len(sk.robot.get_active_joints()):
        warnings.warn(
            f"qpos is not complete {len(qpos)} < {len(sk.robot.get_active_joints())}, filling the rest with 0."
        )
        qpos = np.asarray(qpos).tolist() + [0] * (
            len(sk.robot.get_active_joints()) - len(qpos)
        )

    robot_poses = {}
    if link_names is None:
        link_names = [link.name for link in links]

    for link_index, link in enumerate(links):
        link_name = link.name
        if link_name not in link_names:
            continue
        pq = sk.compute_forward_kinematics(qpos, link_index)

        pose = utils_3d.Rt_to_pose(transforms3d.quaternions.quat2mat(pq.q), pq.p)
        robot_poses[link_name] = Tw_w2B @ pose
        # robot_poses.append(Tw_w2B @ pose)
        # robot_poses.append(Tw_w2B @ pose)
    return robot_poses


if __name__ == "__main__":
    from dexwm.utils.macros import ASSETS_DIR
    from dexwm.utils.pcld_wrapper.mimic_joint import MimicJointForward

    wis3d = Wis3D(
        out_folder="wis3d",
        sequence_name="robot_pcld_wrapper",
        xyz_pattern=("x", "-y", "-z"),
    )
    urdf_path = f"{ASSETS_DIR}/ability_hand/ability_hand_right.urdf"

    wis3d.add_robot(urdf_path)
    pclds = sample_robot_point_cloud(urdf_path, 128)

    # if urdf_path not in Wis3D.urdf_caches:
    #     Wis3D.urdf_caches[urdf_path] =
    robot: SAPIENKinematicsModelStandalone = SAPIENKinematicsModelStandalone(urdf_path)

    qpos_names = [j.name for j in robot.robot.get_active_joints()]

    mimic_joints = MimicJointForward(
        action_names=[
            "thumb_q1",
            "index_q1",
            "middle_q1",
            "ring_q1",
            "pinky_q1",
            "thumb_q2",
        ],
        qpos_names=qpos_names,
        mimic_joint_map={
            "index_q2": dict(
                mimic="index_q1", multiplier=1.05851325, offset=0.72349796
            ),
            "middle_q2": dict(
                mimic="middle_q1", multiplier=1.05851325, offset=0.72349796
            ),
            "ring_q2": dict(mimic="ring_q1", multiplier=1.05851325, offset=0.72349796),
            "pinky_q2": dict(
                mimic="pinky_q1", multiplier=1.05851325, offset=0.72349796
            ),
        },
        device="cpu",
    )

    qpos = [0.2, 0.2, 0.2, 0.0, 0.2, 0.2]  # 0.2, 0.0, 0.0, 0.0]
    qpos = (
        mimic_joints.forward(torch.tensor(qpos).float().unsqueeze(0)).squeeze(0).numpy()
    )
    link_poses = get_robot_poses(
        urdf_path, link_names=pclds.keys(), qpos=qpos
    )  # ).float()

    # for k in pclds.keys():
    #     wis3d.add_point_cloud(pclds[k].float(), name=k)

    pcld_ordered = [pclds[k].float() for k in link_poses.keys()]
    pcld_len = torch.cat(
        [
            torch.tensor([0]),
            torch.cumsum(torch.tensor([len(pc) for pc in pcld_ordered]), dim=0),
        ]
    )
    sampled_pclds = torch.cat(pcld_ordered, dim=0)  # (num_samples, 3)
    pcld_indices = torch.zeros((len(sampled_pclds))).long()
    for i in range(len(pcld_len) - 1):
        pcld_indices[pcld_len[i] : pcld_len[i + 1]] = i

    link_poses = torch.stack(
        [torch.from_numpy(pose) for pose in link_poses.values()]
    ).float()
    pcld_pose = link_poses[pcld_indices, :, :]  # (num_samples, 4, 4)

    verts = (
        torch.einsum("ni, nji->nj", sampled_pclds, pcld_pose[:, :3, :3])
        + pcld_pose[:, :3, 3]
    )  # .unsqueeze(1)

    wis3d.add_robot(urdf_path, qpos=qpos)
    wis3d.add_point_cloud(verts, name="point clouds")