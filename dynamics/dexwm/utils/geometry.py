import torch
import numpy as np
from pytorch3d.ops import corresponding_points_alignment
from pytorch3d.structures import Meshes, Pointclouds
from manopth.tensutils import th_posemap_axisang


def aligned_points_to_transformation_matrix(p1, p2, estimate_scale=False):
    """
    Compute the transformation matrix that aligns p1 to p2.

    Parameters:
    - p1: torch.Tensor of shape (B, N, 3), source points.
    - p2: torch.Tensor of shape (B, N, 3), target points.
    - estimate_scale: bool, whether to estimate the scale.

    Returns:
    - transformation_matrix: torch.Tensor of shape (B, 4, 4), transformation matrix.
    """
    if p1.ndim == 2:
        p1 = p1.unsqueeze(0)

    if p1.shape[0] != p2.shape[0]:
        assert p1.shape[0] == 1
        p1 = p1.repeat(p2.shape[0], 1, 1)

    B, N, _ = p1.shape

    result = corresponding_points_alignment(
        Pointclouds(p1), Pointclouds(p2), estimate_scale=estimate_scale
    )

    return torch.cat(
        [
            torch.cat([torch.linalg.inv(result.R), result.T.unsqueeze(-1)], dim=-1),
            torch.tensor([[0, 0, 0, 1]], device=p1.device, dtype=p1.dtype)
            .reshape(-1, 1, 4)
            .repeat(B, 1, 1),
        ],
        dim=-2,
    )


# def generate_random_rotation(num_samples, device="cuda"):
#     """
#     generate a random rotation matrix
#     return: torch.Tensor of shape (B, 3, 3)
#     """
#     random_vec = torch.randn(num_samples, 3, device=device)
#     random_vec = random_vec / torch.norm(random_vec, dim=-1, keepdim=True)
#     theta = torch.rand(num_samples, 1, device=device) * 2 * torch.pi
#     _, rot = th_posemap_axisang(random_vec * theta)
#     return rot.view(-1, 3, 3)


def generate_random_rotation(num_samples, along_z=False, device="cuda"):
    """
    generate a random rotation matrix
    return: torch.Tensor of shape (B, 3, 3)
    """
    if along_z:
        random_vec = torch.tensor([0, 0, 1], device=device).repeat(num_samples, 1)
    else:
        random_vec = torch.randn(num_samples, 3, device=device)
        random_vec = random_vec / torch.norm(random_vec, dim=-1, keepdim=True)
    theta = torch.rand(num_samples, 1, device=device) * 2 * torch.pi
    _, rot = th_posemap_axisang(random_vec * theta)
    return rot.view(-1, 3, 3)


"""
To be honest pytorch3d implementation works better than spicy implementation
"""
'''
from scipy.linalg import orthogonal_procrustes
def aligned_points_to_transformation_matrix(
    p1: torch.Tensor, p2: torch.Tensor, estimate_scale=False
):
    """
    Compute the transformation matrix that aligns p1 to p2.

    Parameters:
    - p1: np.array of shape (N, 3), source points.
    - p2: np.array of shape (N, 3), target points.
    - estimate_scale: bool, whether to estimate the scale.

    Returns:
    - transformation_matrix: np.array of shape (4, 4), transformation matrix.
    """
    assert p1.shape == p2.shape and p1.ndim == 2, "Invalid input shapes."
    transformation_matrix = np.zeros((4, 4))

    transformation_matrix[:3, 3] = np.mean(p2, axis=0) - np.mean(p1, axis=0)

    # translate data to the origin
    mx1 = p1 - p1.mean(axis=0)
    mx2 = p2 - p2.mean(axis=0)

    # normalize the data
    # mx1 = mx1 / (torch.norm(mx1) + 1e-8)
    # mx2 = mx2 / (torch.norm(mx2) + 1e-8)

    R, scale = orthogonal_procrustes(mx1, mx2)
    transformation_matrix[:3, :3] = np.linalg.inv(R)

    transformation_matrix[3, 3] = 1

    return transformation_matrix
'''

if __name__ == "__main__":
    p1 = np.load("/hdd/yulin/Dex-World-Model/p1.npy")
    p2 = np.load("/hdd/yulin/Dex-World-Model/p2.npy")

    from wis3d import Wis3D

    wis3d = Wis3D(
        out_folder="wis3d",
        sequence_name="rotation",
        xyz_pattern=("x", "-y", "-z"),
    )

    wis3d.add_point_cloud(p1, name="p1")
    wis3d.add_point_cloud(p2, name="p2")

    # p1 = torch.from_numpy(p1).float().unsqueeze(0)
    # p2 = torch.from_numpy(p2).float().unsqueeze(0)

    transformation_matrix = aligned_points_to_transformation_matrix(p1, p2)

    # p1_trans = torch.einsum(
    #     "bni,bji->bnj", p1, transformation_matrix[:, :3, :3]
    # ) + transformation_matrix[:, :3, 3].unsqueeze(-2)
    p1_trans = (
        np.matmul(p1, transformation_matrix[:3, :3].T) + transformation_matrix[:3, 3]
    )
    wis3d.add_point_cloud(p1_trans.reshape(-1, 3), name="p1_trans")
