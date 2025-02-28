from copy import deepcopy
import os
import numpy as np
import yaml

class Projector:
    def __init__(self, calib_path: str):
        with open(os.path.join(calib_path, "extrinsics.yml"), "r") as f:
            self.extrinsics = yaml.safe_load(f)
        with open(os.path.join(calib_path, "transform_data.yml"), "r") as f:
            self.transform_data = yaml.safe_load(f)

        self.intrinsics = {}
        self.intrinsics['cam_0'] = np.array([[909.22192383,   0.        , 634.7142334 ,   0.        ],
                                            [  0.        , 909.41491699, 352.74398804,   0.        ],
                                            [  0.        ,   0.        ,   1.        ,   0.        ]])
        self.intrinsics['cam_1'] = np.array([[912.91558838,   0.        , 661.25982666,   0.        ],
                                    [  0.        , 912.52545166, 373.5128479 ,   0.        ],
                                    [  0.        ,   0.        ,   1.        ,   0.        ]])
        self.intrinsics['cam_2'] = np.array([[916.56665039,   0.        , 648.18109131,   0.        ],
                                            [  0.        , 916.77130127, 358.43869019,   0.        ],
                                            [  0.        ,   0.        ,   1.        ,   0.        ]])
        self.intrinsics['cam_3'] = np.array([[910.8637085 ,   0.        , 619.1239624 ,   0.        ],
                                    [  0.        , 910.2946167 , 351.13458252,   0.        ],
                                    [  0.        ,   0.        ,   1.        ,   0.        ]])

        # Store transformations from marker frame to camera frame (optical and ROS)
        self.marker_to_color_optical = {}
        self.marker_to_color = {}
        self.marker_to_link = {}
        for camera_sn, value in self.extrinsics.items():
            # The extrinsic matrix is the pose of the camera in the marker frame,
            # so we need its inverse to get the transformation from marker to camera.
            camera_link_to_marker = np.array(value["transformation"], dtype=np.float64)
            marker_to_camera_link = np.linalg.inv(camera_link_to_marker)

            color_to_link = np.array(self.transform_data[f'{camera_sn}_color_frame_to_{camera_sn}_link']["transformation"], dtype=np.float64)
            link_to_color = np.linalg.inv(color_to_link)

            optical_to_color = np.array(self.transform_data[f'{camera_sn}_color_optical_frame_to_{camera_sn}_color_frame']["transformation"], dtype=np.float64)
            color_to_optical = np.linalg.inv(optical_to_color)

            # Store the transformation from marker to ROS frame
            self.marker_to_link[camera_sn] = marker_to_camera_link
            self.marker_to_color[camera_sn] = link_to_color @ marker_to_camera_link
            self.marker_to_color_optical[camera_sn] = color_to_optical @ self.marker_to_color[camera_sn]

    def project_point_cloud_groups(
        self,
        point_cloud: np.ndarray,  # [N, P, 3]
        src_camera_sn: str,
        tgt_camera_sn: str,
    ) -> np.ndarray:
        """
        Projects point cloud from the source camera's optical frame to
        the target camera's optical frame entirely in NumPy.

        Args:
            point_cloud (np.ndarray): The point cloud of shape [N, P, 3].
            src_camera_sn (str): Source camera serial number.
            tgt_camera_sn (str): Target camera serial number.

        Returns:
            np.ndarray: Transformed point cloud [N, P, 3] in the target camera's frame.
        """
        # 1) Get the transformation matrices from marker to source and target camera frames
        marker_to_src_camera = self.marker_to_color_optical[src_camera_sn]  # shape [4, 4]
        marker_to_tgt_camera = self.marker_to_color_optical[tgt_camera_sn]  # shape [4, 4]

        # 2) Compute src->tgt transform:
        #    src_camera_to_tgt_camera = inv(marker_to_tgt_camera) @ marker_to_src_camera
        src_camera_to_tgt_camera = marker_to_tgt_camera @ np.linalg.inv(marker_to_src_camera)  # shape [4, 4]

        # 3) Expand point cloud to homogeneous coordinates: [N, P, 3] -> [N, P, 4]
        N, P, _ = point_cloud.shape
        ones = np.ones((N, P, 1), dtype=np.float64)
        point_cloud_hom = np.concatenate([point_cloud, ones], axis=2)  # [N, P, 4]

        # 4) Apply the transform
        point_cloud_hom_transformed = point_cloud_hom @ src_camera_to_tgt_camera.T  # [N, P, 4]

        # 5) Discard the last dimension (the homogeneous "1")
        transformed_points = point_cloud_hom_transformed[..., :3]  # [N, P, 3]

        return transformed_points

    def project_point_cloud_to_marker(
        self,
        point_cloud: np.ndarray,  # [N, P, 3]
        camera_sn: str,
    ) -> np.ndarray:
        """
        Projects point cloud from the camera's optical frame to
        the marker frame entirely in NumPy.

        Args:
            point_cloud (np.ndarray): The point cloud of shape [N, P, 3].
            camera_sn (str): Camera serial number.

        Returns:
            np.ndarray: Transformed point cloud [N, P, 3] in the marker frame.
        """
        # 1) Get the transformation matrix from marker to camera frame
        marker_to_camera = self.marker_to_color_optical[camera_sn]

        # 2) Compute camera->marker transform:
        #    camera_to_marker = inv(marker_to_camera)
        camera_to_marker = np.linalg.inv(marker_to_camera)

        # 3) Expand point cloud to homogeneous coordinates: [N, P, 3] -> [N, P, 4]
        P, _ = point_cloud.shape
        ones = np.ones((P, 1), dtype=np.float64)
        point_cloud_hom = np.concatenate([point_cloud, ones], axis=1)  # [N, P, 4]

        # 4) Apply the transform
        point_cloud_hom_transformed = point_cloud_hom @ camera_to_marker.T  # [N, P, 4]

        # 5) Discard the last dimension (the homogeneous "1")
        transformed_points = point_cloud_hom_transformed[..., :3]  # [N, P, 3]

        return transformed_points
    
    def intrinsic_matrix(self, camera_sn: str):
        return self.intrinsics[camera_sn]