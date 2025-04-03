from scipy.spatial.transform import Rotation as R
import numpy as np


def to_transformation_matrix(pose: np.ndarray) -> np.ndarray:
    """
    covert (n, 7) pose to (n, 4, 4) transformation matrix
    """
    rot = R.from_quat(pose[:, :4])
    trans = pose[:, 4:]
    return np.concatenate(
        [
            np.concatenate([rot.as_matrix(), trans[:, :, None]], axis=-1),
            np.array([0, 0, 0, 1]).reshape(1, 1, 4).repeat(pose.shape[0], axis=0),
        ],
        axis=-2,
    )


def to_pos_quaternion(transformation_matrix: np.ndarray) -> np.ndarray:
    """
    convert (n, 4, 4) transformation matrix to (n, 7) pose
    """
    rot = R.from_matrix(transformation_matrix[:, :3, :3])
    trans = transformation_matrix[:, :3, 3]
    return np.concatenate([rot.as_quat(), trans], axis=-1)


def find_longest_true_sequence(occlusion):
    longest_start = -1
    longest_end = -1
    max_length = 0

    current_start = -1
    current_length = 0

    for i, value in enumerate(occlusion):
        if value:  # If the current value is True
            if current_length == 0:  # Starting a new sequence
                current_start = i
            current_length += 1
        else:  # If the current value is False
            if current_length > max_length:  # End the previous sequence
                max_length = current_length
                longest_start = current_start
                longest_end = i - 1  # End index is one before the current index
            current_length = 0  # Reset the current sequence

    # Handle the case where the longest sequence ends at the last element
    if current_length > max_length:
        max_length = current_length
        longest_start = current_start
        longest_end = len(occlusion) - 1

    return longest_start, longest_end, max_length
