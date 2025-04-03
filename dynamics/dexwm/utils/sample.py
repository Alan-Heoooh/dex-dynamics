import torch
import numpy as np
from tqdm import tqdm

total_iterations = 4000


# def furthest_point_sampling(points, num_samples):
#     """
#     points: (N, 3)
#     num_samples: int
#     """
#     sampled_points = points[0][None, ...]  # (3, )
#     sampled_point = points[0]  # (3, )
#     distance = np.linalg.norm(points - sampled_point, axis=-1)  # (N, )
#     sampled_indices = [0]
#     # print(points.shape)
#     # points = np.tile(points, (total_iterations, 1))  # (3,)
#     # print(points.shape)

#     for i in range(num_samples):  # tqdm(range(1, total_iterations), desc="Processing"):
#         # find the largest distance to the existing points in the point cloud
#         # sample points (N, 3)
#         # points (m, 3)
#         # distance (N, m)
#         distance_ = np.linalg.norm(points - sampled_point, axis=-1)  # (N, m)
#         distance = np.where(distance < distance_, distance, distance_)  # (N, m)
#         # distances = np.linalg.norm(sampled_points[:, None, :] - points[None, :, :], axis=-1) # (N, m)
#         # distances_min = np.min(distance, axis=-1) # (N)
#         distance_argmax = np.argmax(distance)  # ()
#         point = points[distance_argmax]  # (3,)
#         # points[i] = point # (m+1, 3)
#         sampled_points = np.concatenate(
#             [sampled_points, point[None]], axis=0
#         )  # (m+1, 3)
#         sampled_indices.append(distance_argmax)
#     return points, sampled_indices


def furthest_point_sampling(points, num_samples):
    """
    Perform Furthest Point Sampling (FPS) on a set of points.

    Parameters:
    - points: np.ndarray of shape (N, 3), input point cloud.
    - num_samples: int, number of points to sample.

    Returns:
    - sampled_points: np.ndarray of shape (num_samples, 3), sampled points.
    """
    N = points.shape[0]
    sampled_indices = []
    distances = np.full(N, np.inf)

    # Start with a random point
    sampled_indices.append(np.random.randint(N))

    for _ in range(num_samples - 1):
        # Update distances to the sampled set
        last_sampled = points[sampled_indices[-1]]
        dist_to_last = np.linalg.norm(points - last_sampled, axis=1)
        distances = np.minimum(distances, dist_to_last)

        # Choose the furthest point
        next_index = np.argmax(distances)
        sampled_indices.append(next_index)

    return points[sampled_indices], np.array(sampled_indices)
