"""scripts to convert mesh to sdf"""

import os
import sys
import time

import mesh2sdf
import numpy as np
import torch

# import torchcumesh2sdf
# from diso import DiffMC


def get_sdf_from_normalized_mesh(
    vertices: np.array, faces: np.array, resolution: int = 128
) -> np.array:
    """
    :param vertices: vertices of the mesh, (N, 3) normalized to [-scale, scale] (preprocessed)
    :param faces: faces of the mesh, (M, 3)
    :param resolution: resolution of the sdf, default is 128
    :return: sdf, np.array, (R, R, R), where R = resolution
    """
    sdf = mesh2sdf.compute(
        vertices, faces, resolution, fix=True, level=2 / resolution, return_mesh=False
    )
    return sdf


# import trimesh
# def get_sdf_from_normalized_mesh_cu(vertices: np.array, faces: np.array, resolution: int = 128) -> torch.Tensor:
#     """
#     :param vertices: vertices of the mesh, (N, 3) normalized to [-scale, scale] (preprocessed)
#     :param faces: faces of the mesh, (M, 3)
#     :param resolution: resolution of the sdf, default is 128
#     :return: sdf, torch.Tensor, (R, R, R), where R = resolution
#     """
#     # size = args.size  # resolution of SDF
#     vertices = torch.from_numpy(vertices).float().cuda()
#     faces = torch.from_numpy(faces).int().cuda()

#     band = 3 / resolution  # band value

#     tris = vertices[faces]
#     tris = tris * 0.5 + 0.5
#     udf = torchcumesh2sdf.get_udf(tris, resolution, band)

#     diffmc = DiffMC().cuda()
#     with torch.no_grad():
#         vertices, faces = diffmc((udf - 1.0 / resolution))

#     new_mesh = trimesh.Trimesh(vertices=vertices.cpu().numpy(), faces=faces.cpu().numpy())
#     assert new_mesh.is_watertight

#     tris = vertices[faces]
#     sdf = torchcumesh2sdf.get_sdf(tris, resolution, band)

#     sdf[sdf > 1 / 10] = 1 / 10
#     sdf[sdf < -1 / 10] = -1 / 10
#     return sdf


def normalize_mesh(
    vertices: np.array, faces: np.array, scale: float = 0.8
) -> tuple[np.array, np.array, np.array, float]:
    """
    :param vertices: vertices of the mesh, (N, 3)
    :param faces: faces of the mesh, (M, 3)
    :param scale: scale of the mesh, default is 0.8, range [0, 1]
    :return: normalized vertices, faces, center, scale
    """
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * scale / (bbmax - bbmin).max()  # scale to fit in [-1, 1], scalar
    vertices = (vertices - center) * scale
    return (
        vertices.astype(np.float32),
        faces.astype(np.int32),
        center.astype(np.float32),
        scale.astype(np.float32),
    )


def get_sdf_from_mesh(
    vertices: np.array,
    faces: np.array,
    resolution: int = 128,
    scale: float = 0.8,
    use_mesh2sdf=True,
) -> tuple[np.array, np.array, float]:
    """
    :param vertices: vertices of the mesh, (N, 3)
    :param faces: faces of the mesh, (M, 3)
    :param resolution: resolution of the sdf, default is 128
    :param scale: scale of the mesh, default is 0.8, range [0, 1]
    :return: sdf, center, scale, torch.Tensor, torch.Tensor, torch.Tensor
        sdf: signed distance field, (R, R, R), where R = resolution
        center: center of the mesh, (3,)
        scale: scale of the mesh, scalar
        both on cuda
    """
    normalized_vertices, normalized_faces, center, scale = normalize_mesh(
        vertices, faces, scale
    )
    if use_mesh2sdf:
        sdf = get_sdf_from_normalized_mesh(
            normalized_vertices, normalized_faces, resolution
        )
    else:
        raise NotImplementedError
        # sdf = (
        #     get_sdf_from_normalized_mesh_cu(normalized_vertices, normalized_faces, resolution).cpu().numpy()
        # )  # torch cuda

    return sdf, center, scale


def query_sdf(
    sdf: torch.Tensor, points: torch.Tensor, scale: torch.Tensor, center: torch.Tensor
) -> torch.Tensor:
    """
    :param sdf: signed distance field, (R, R, R)
    :param points: points to query, (B, N, 3), or (N, 3)
    :param scale: scale of the mesh, scalar
    :param center: center of the mesh, (3,)
    :return: sdf values at the queried points, (B, N,)
    """
    normalized_points = (points - center) * scale  # normalize points to [-1, 1]
    return query_sdf_from_normalized_points(sdf, normalized_points)


def query_sdf_from_normalized_points(
    sdf: torch.Tensor, points: torch.Tensor
) -> torch.Tensor:
    """
    :param sdf: signed distance field, (R, R, R)
    :param points: points to query, (a, b, c, 3) normalized to [-1, 1]
    :param scale: scale of the mesh, scalar
    :param center: center of the mesh, (3,)
    :return: sdf values at the queried points, (a, b, c,)
    """
    points = points[..., [2, 1, 0]]
    points = points.clamp(-1, 1)  # clamp to [-1, 1]

    original_dim = points.ndim
    while points.ndim < 5:
        points = points.unsqueeze(0)
    # points[.]
    sampled_sdf = torch.nn.functional.grid_sample(
        sdf.unsqueeze(0).unsqueeze(0),  # (1, 1, R, R, R)
        points,  # (1, a, b, c, 3)
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )  # , align_corners=True)

    for _ in range(6 - original_dim):
        sampled_sdf = sampled_sdf.squeeze(0)
    return sampled_sdf  # (a, b, c)


if __name__ == "__main__":
    obj_path = "/hdd/data/dexrl/mesh/banana/object.obj"
    import trimesh

    from dexrl.visualize.sdf_visualizer import visualize_grid_sdf, visualize_point_sdf
    from wis3d import Wis3D

    wis3d = Wis3D(
        out_folder="wis3d",
        sequence_name="nocs",
        xyz_pattern=("x", "y", "z"),
    )

    mesh = trimesh.load(obj_path)
    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.int32)

    # vertices = torch.tensor(mesh.vertices, dtype=torch.float32).to("cuda")
    # faces = torch.tensor(mesh.faces, dtype=torch.int64).to("cuda")

    sdf, scale, center = get_sdf_from_mesh(vertices, faces, resolution=128, scale=0.8)
    # print(sdf.shape, scale, center)
    # visualize_grid_sdf(wis3d, sdf, truncation=2.0, name="gt sdf")

    # sdf = torch.from_numpy(sdf).to("cuda")
    vertices = torch.from_numpy(vertices).to("cuda")
    # center = torch.from_numpy(center).to("cuda")
    # scale = torch.tensor(scale).to("cuda")
    queried_sdf = query_sdf(sdf, vertices.unsqueeze(0), scale, center)
    # visualize_point_sdf(
    #     wis3d,
    #     vertices.cpu().numpy(),
    #     queried_sdf.cpu().numpy(),
    #     truncation=2.0,
    #     name="queried vertices sdf",
    # )
    # print(sdf.shape)

    N = 128
    x = torch.linspace(-1, 1, N)
    y = torch.linspace(-1, 1, N)
    z = torch.linspace(-1, 1, N)

    # Create the 3D meshgrid for x, y, z coordinates
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")

    # Stack the 3D grids along a new dimension to create (N, N, N, 3)
    sample_grid = torch.stack((grid_x, grid_y, grid_z), dim=-1).to("cuda")
    queried_grid_sdf = query_sdf_from_normalized_points(sdf, sample_grid)

    from skimage import measure

    verts, faces, _, _ = measure.marching_cubes(sdf.cpu().numpy(), level=0)
    wis3d.add_mesh(verts, faces, name="gt mesh")

    verts, faces, _, _ = measure.marching_cubes(queried_grid_sdf.cpu().numpy(), level=0)
    wis3d.add_mesh(verts, faces, name="queried grid sdf mesh")
    # print(" ")
    # visualize_point_sdf(
    #     wis3d,
    #     sample_grid.reshape(-1, 3).cpu().numpy(),
    #     queried_grid_sdf.reshape(-1).cpu().numpy(),
    #     truncation=2.0,
    #     name="queried grid sdf",
    # )
    # visualize_grid_sdf(
    #     wis3d,
    #     queried_grid_sdf.cpu().numpy(),
    #     truncation=2.0,
    #     name="queried grid sdf via grid vis",
    # )
