import torch
import numpy as np
from multipledispatch import dispatch
from transforms3d.euler import euler2axangle
from transforms3d.quaternions import quat2mat


def Rt_to_pose(R, t=np.zeros(3)):
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose


def cart_to_hom(pts):
    """
    :param pts: (N, 3 or 2)
    :return pts_hom: (N, 4 or 3)
    """
    if isinstance(pts, np.ndarray):
        pts_hom = np.concatenate(
            (pts, np.ones([*pts.shape[:-1], 1], dtype=np.float32)), -1
        )
    else:
        ones = torch.ones([*pts.shape[:-1], 1], dtype=torch.float32, device=pts.device)
        pts_hom = torch.cat((pts, ones), dim=-1)
    return pts_hom


def hom_to_cart(pts):
    return pts[..., :-1] / pts[..., -1:]


@dispatch(np.ndarray, np.ndarray)
def transform_points(pts, pose):
    """
    :param pts: Nx3
    :param pose: 4x4
    :param calib:Calibration
    :return:
    """
    pts = cart_to_hom(pts)
    pts = pts @ pose.T
    pts = hom_to_cart(pts)
    return pts


@dispatch(torch.Tensor, torch.Tensor)
def transform_points(pts, pose):
    pts = cart_to_hom(pts)
    pts = pts @ pose.transpose(-1, -2)
    pts = hom_to_cart(pts)
    return pts


def depth_to_cam(K, depth_map):
    """

    :param K: np.ndarray or torch.Tensor, 3x3
    :param depth_map: H,W
    :return:
    """
    if isinstance(K, torch.Tensor):
        K = K.detach().cpu().numpy()
    if isinstance(depth_map, np.ndarray):
        x_range = np.arange(0, depth_map.shape[1])
        y_range = np.arange(0, depth_map.shape[0])
        x_idxs, y_idxs = np.meshgrid(x_range, y_range)
    else:
        x_range = torch.arange(0, depth_map.shape[1]).to(device=depth_map.device)
        y_range = torch.arange(0, depth_map.shape[0]).to(device=depth_map.device)
        y_idxs, x_idxs = torch.meshgrid(y_range, x_range, indexing="ij")
    x_idxs, y_idxs = x_idxs.reshape(-1), y_idxs.reshape(-1)
    depth = depth_map[y_idxs, x_idxs]
    pts_rect = img_to_cam(K, x_idxs + 0.5, y_idxs + 0.5, depth)
    return pts_rect


def img_to_cam(K, u, v, depth_rect):
    """
    :param u: (N)
    :param v: (N)
    :param depth_rect: (N)
    :return: pts_rect:(N, 3)
    """
    # check_type(u)
    # check_type(v)

    fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if isinstance(depth_rect, np.ndarray):
        x = ((u - cu) * depth_rect) / fu
        y = ((v - cv) * depth_rect) / fv
        pts_rect = np.concatenate(
            (x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1
        )
    else:
        x = ((u.float() - cu) * depth_rect) / fu
        y = ((v.float() - cv) * depth_rect) / fv
        pts_rect = torch.cat(
            (x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), dim=1
        )
    return pts_rect


def rot_euler2axangle(rotation, axes="sxyz"):
    """
    Convert Euler angles [roll, pitch, yaw] -> axis-angle vector, for either
    a PyTorch tensor or a NumPy array (shape = (3,)).

    :param rotation: 3D rotation angles [rx, ry, rz] in radians.
                     Accepts torch.Tensor or np.ndarray.
    :param axes: The axes convention to use. Default is "sxyz".
    :return: A vector [ax, ay, az] where
             axis = [ax, ay, az]/angle, angle = norm([ax, ay, az]).
    """
    # --- 1) Pull out the 3 angles as floats, regardless of input type ---
    if isinstance(rotation, torch.Tensor):
        ai = rotation[0].item()
        aj = rotation[1].item()
        ak = rotation[2].item()
    elif isinstance(rotation, np.ndarray):
        ai = float(rotation[0])
        aj = float(rotation[1])
        ak = float(rotation[2])
    else:
        raise TypeError(
            f"rotation must be torch.Tensor or np.ndarray, got {type(rotation)}"
        )

    # --- 2) Convert Euler angles -> (axis, angle) using transforms3d ---
    axis, angle = euler2axangle(ai, aj, ak, axes=axes)  # axis is a NumPy array, angle is float

    # --- 3) Build the final axis-angle vector = axis * angle ---
    if isinstance(rotation, torch.Tensor):
        # Make a PyTorch tensor on the same device/dtype as 'rotation'
        axis_angle = torch.tensor(
            axis,
            dtype=rotation.dtype,
            device=rotation.device
        ) * angle
    else:
        # Make a NumPy array
        axis_angle = axis * angle
        axis_angle = np.asarray(axis_angle, dtype=rotation.dtype)
        
    return axis_angle