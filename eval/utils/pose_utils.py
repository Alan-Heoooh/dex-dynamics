# import pinocchio as pin
import numpy as np
from sapien import Pose

convention = np.array([
    [-1, 0., 0., 0.],
    [0., -1., 0., 0.],
    [0., 0., 1., 0.],
    [0., 0., 0., 1]
], dtype=np.float64)

opencv_to_ros = np.array([
    [0, 0., 1., 0.],
    [-1., 0., 0., 0.],
    [0., -1., 0., 0.],
    [0., 0., 0., 1]
], dtype=np.float64
)

def xyzw_to_wxyz(quat):
    quat = quat.copy()
    return np.roll(quat, 1)

def opencv_to_sapien_pose(pose):
    pose = Pose(pose @ convention)
    return np.concatenate([pose.p, pose.q])

def sapien_pose_to_opencv(pose):
    return Pose(p=pose[:3], q = pose[3:]).to_transformation_matrix() @ convention


def matrix_to_sapien_pose(pose):
    # pose = pin.SE3ToXYZQUAT(pin.SE3(pose))
    # pose[3:] = xyzw_to_wxyz(pose[3:])

    pose = Pose(pose)
    return np.concatenate([pose.p, pose.q])

def sapien_pose_to_matrix(pose):
    return Pose(p=pose[:3], q = pose[3:]).to_transformation_matrix() #@ convention

def rotation_deg(rotation1, rotation2):
    """
    rotation1: (3, 3)
    rotation2: (3, 3)
    """
    rot_mat = rotation1.T @ rotation2
    return np.arccos(np.clip((np.trace(rot_mat) -1) / 2, -1, 1)) * 180 / np.pi


if __name__ == "__main__":
    pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    print(matrix_to_sapien_pose(pose))
    print(sapien_pose_to_matrix(matrix_to_sapien_pose(pose)))