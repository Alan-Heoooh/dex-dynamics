from copy import deepcopy
import os
import numpy as np
import torch
import yaml

class Projector:
    def __init__(self, calib_path: str):
        with open(os.path.join(calib_path, "extrinsics.yml"), "r") as f:
            self.extrinsics = yaml.safe_load(f)
        with open(os.path.join(calib_path, "transform_data.yml"), "r") as f:
            self.transform_data = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Transformation from ROS frame to optical frame
        self.ros_to_optical = torch.tensor(
            [
                [-0., -1., -0., -0.],
                [-0., -0., -1., -0.],
                [ 1.,  0.,  0.,  0.],
                [ 0.,  0.,  0.,  1.]
            ],
            dtype=torch.float64,
            device=self.device,
        )
        self.optical_to_ros = torch.linalg.inv(self.ros_to_optical)

        # Store transformations from marker frame to camera frame (optical and ROS)
        self.marker_to_color_optical = {}
        self.marker_to_color = {}
        self.marker_to_link = {}
        for camera_sn, value in self.extrinsics.items():
            # The extrinsic matrix is the pose of the camera in the marker frame,
            # so we need its inverse to get the transformation from marker to camera.
            camera_link_to_marker = torch.tensor(value["transformation"], dtype=torch.float64, device=self.device)
            marker_to_camera_link = torch.linalg.inv(camera_link_to_marker)

            color_to_link = torch.tensor(self.transform_data[f'{camera_sn}_color_frame_to_{camera_sn}_link']["transformation"], dtype=torch.float64, device=self.device)

            link_to_color = torch.linalg.inv(color_to_link)

            optical_to_color = torch.tensor(self.transform_data[f'{camera_sn}_color_optical_frame_to_{camera_sn}_color_frame']["transformation"], dtype=torch.float64, device=self.device)

            color_to_optical = torch.linalg.inv(optical_to_color)

            # Store the transformation from marker to ROS frame
            self.marker_to_link[camera_sn] = marker_to_camera_link

            self.marker_to_color[camera_sn] = link_to_color @ marker_to_camera_link

            self.marker_to_color_optical[camera_sn] = color_to_optical @ self.marker_to_color[camera_sn]

    def project_point_cloud_groups(
        self,
        point_cloud: torch.Tensor,  # [N, P, 3]
        src_camera_sn: str,
        tgt_camera_sn: str,
    ) -> torch.Tensor:
        """
        Projects point cloud from the source camera's optical frame to
        the target camera's optical frame entirely in PyTorch.

        Args:
            point_cloud (torch.Tensor): The point cloud of shape [N, P, 3].
            src_camera_sn (str): Source camera serial number.
            tgt_camera_sn (str): Target camera serial number.

        Returns:
            torch.Tensor: Transformed point cloud [N, P, 3] in the target camera's frame.
        """
        # 1) Get the transformation matrices from marker to source and target camera frames
        marker_to_src_camera = self.marker_to_color_optical[src_camera_sn]  # shape [4, 4]
        marker_to_tgt_camera = self.marker_to_color_optical[tgt_camera_sn]  # shape [4, 4]

        # 2) Compute src->tgt transform:
        #    src_camera_to_tgt_camera = inv(marker_to_tgt_camera) @ marker_to_src_camera
        src_camera_to_tgt_camera = torch.matmul(
            marker_to_tgt_camera,
            torch.linalg.inv(marker_to_src_camera)
        ) # shape [4,4]

        point_cloud = point_cloud.to(self.device)

        # 3) Expand point cloud to homogeneous coordinates: [N, P, 3] -> [N, P, 4]
        N, P, _ = point_cloud.shape
        device = point_cloud.device
        dtype = point_cloud.dtype

        ones = torch.ones((N, P, 1), device=device, dtype=dtype)
        point_cloud_hom = torch.cat([point_cloud, ones], dim=2)  # [N, P, 4]

        # 4) Apply the transform
        point_cloud_hom_transformed = point_cloud_hom @ src_camera_to_tgt_camera.T  # [N, P, 4]

        # 5) Discard the last dimension (the homogeneous "1")
        transformed_points = point_cloud_hom_transformed[..., :3]  # [N, P, 3]

        return transformed_points
    
    def project_point_cloud_to_marker(
        self,
        point_cloud: torch.Tensor,  # [N, P, 3]
        camera_sn: str,
    ) -> torch.Tensor:
        """
        Projects point cloud from the camera's optical frame to
        the marker frame entirely in PyTorch.

        Args:
            point_cloud (torch.Tensor): The point cloud of shape [N, P, 3].
            camera_sn (str): Camera serial number.

        Returns:
            torch.Tensor: Transformed point cloud [N, P, 3] in the marker frame.
        """

        # 1) Get the transformation matrix from marker to camera frame
        marker_to_camera = self.marker_to_color_optical[camera_sn]

        # 2) Compute camera->marker transform:
        #    camera_to_marker = inv(marker_to_camera)

        camera_to_marker = torch.linalg.inv(marker_to_camera)

        point_cloud = point_cloud.to(self.device)

        # 3) Expand point cloud to homogeneous coordinates: [N, P, 3] -> [N, P, 4]

        N, P, _ = point_cloud.shape
        device = point_cloud.device
        dtype = point_cloud.dtype

        ones = torch.ones((N, P, 1), device=device, dtype=dtype)
        point_cloud_hom = torch.cat([point_cloud, ones], dim=2)

        # 4) Apply the transform
        point_cloud_hom_transformed = point_cloud_hom @ camera_to_marker.T

        # 5) Discard the last dimension (the homogeneous "1")
        transformed_points = point_cloud_hom_transformed[..., :3]

        return transformed_points