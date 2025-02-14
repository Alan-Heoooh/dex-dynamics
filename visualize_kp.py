import os
import numpy as np
import open3d as o3d
import torch
from projector import Projector

color = '/home/coolbot/data/hand_perception/record_scene_5/cam_3/color'
depth = '/home/coolbot/data/hand_perception/record_scene_5/cam_3/depth'

intrinsics = np.array([[910.8637085 ,   0.        , 619.1239624 ,   0.        ],
                        [  0.        , 910.2946167 , 351.13458252,   0.        ],
                        [  0.        ,   0.        ,   1.        ,   0.        ]])

color_img = o3d.io.read_image(os.path.join(color, 'color_000000.png'))
depth_img = o3d.io.read_image(os.path.join(depth, 'depth_000000.png'))


color_img = np.array(color_img)
depth_img = np.array(depth_img)
color_img = o3d.geometry.Image(color_img)
depth_img = o3d.geometry.Image(depth_img)
rgbd_cam_3 = o3d.geometry.RGBDImage.create_from_color_and_depth(color_img, depth_img, depth_scale=1000.0, convert_rgb_to_intensity=False)

# get camera intrinsic
fx, fy = intrinsics[0, 0], intrinsics[1, 1]
cx, cy = intrinsics[0, 2], intrinsics[1, 2]
cam_intrinsic = o3d.camera.PinholeCameraIntrinsic(1280, 720, fx, fy, cx, cy)

# axis
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

# create point cloud in each camera frame
pcd_cam_3 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_cam_3, cam_intrinsic)

kp = np.array([[0.2101, 0.1583, 0.7618],
         [0.1828, 0.1525, 0.7880],
         [0.1630, 0.1426, 0.8090],
         [0.1480, 0.1257, 0.8239],
         [0.1373, 0.1049, 0.8493],
         [0.1535, 0.0950, 0.7913],
         [0.1390, 0.0688, 0.8047],
         [0.1379, 0.0570, 0.8234],
         [0.1334, 0.0483, 0.8445],
         [0.1687, 0.0767, 0.7837],
         [0.1619, 0.0488, 0.7968],
         [0.1634, 0.0351, 0.8154],
         [0.1656, 0.0223, 0.8381],
         [0.1945, 0.0757, 0.7777],
         [0.1871, 0.0512, 0.7907],
         [0.1897, 0.0382, 0.8115],
         [0.1895, 0.0266, 0.8323],
         [0.2159, 0.0785, 0.7766],
         [0.2175, 0.0592, 0.7847],
         [0.2208, 0.0472, 0.7989],
         [0.2220, 0.0342, 0.8136]])

pcd_kp = o3d.geometry.PointCloud()
pcd_kp.points = o3d.utility.Vector3dVector(kp)
pcd_kp.paint_uniform_color([1, 0, 0])

# visualize the point cloud
o3d.visualization.draw_geometries([pcd_cam_3, axis, pcd_kp])


intrinsics = np.array([[909.22192383,   0.        , 634.7142334 ,   0.        ],
                                            [  0.        , 909.41491699, 352.74398804,   0.        ],
                                            [  0.        ,   0.        ,   1.        ,   0.        ]])

color = '/home/coolbot/data/hand_perception/record_scene_5/cam_0/color'
depth = '/home/coolbot/data/hand_perception/record_scene_5/cam_0/depth'

color_img = o3d.io.read_image(os.path.join(color, 'color_000000.png'))
depth_img = o3d.io.read_image(os.path.join(depth, 'depth_000000.png'))


color_img = np.array(color_img)
depth_img = np.array(depth_img)
color_img = o3d.geometry.Image(color_img)
depth_img = o3d.geometry.Image(depth_img)
rgbd_cam_0 = o3d.geometry.RGBDImage.create_from_color_and_depth(color_img, depth_img, depth_scale=1000.0, convert_rgb_to_intensity=False)

# get camera intrinsic
fx, fy = intrinsics[0, 0], intrinsics[1, 1]
cx, cy = intrinsics[0, 2], intrinsics[1, 2]
cam_intrinsic = o3d.camera.PinholeCameraIntrinsic(1280, 720, fx, fy, cx, cy)

# axis
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

# create point cloud in each camera frame
pcd_cam_0 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_cam_0, cam_intrinsic)

# filter out the far away points
# pcd_cam_0 = pcd_cam_0.crop([0.0, 0.0, 0.0], [0.5, 0.5, 1.0])

kp = kp[None, :, :]

# transform to torch tensor
kp = torch.tensor(kp, dtype=torch.float64)

projector = Projector(calib_path='/home/coolbot/data/calib')

kp_cam_3_to_cam_1 = projector.project_point_cloud_groups(kp, "cam_3", "cam_0")

print(kp_cam_3_to_cam_1)

kp_cam_3_to_cam_1 = kp_cam_3_to_cam_1.cpu().numpy()

kp_cam_3_to_cam_1 = kp_cam_3_to_cam_1[0]

pcd_kp_cam_3_to_cam_1 = o3d.geometry.PointCloud()

pcd_kp_cam_3_to_cam_1.points = o3d.utility.Vector3dVector(kp_cam_3_to_cam_1)

# visualize the point cloud

# o3d.visualization.draw_geometries([pcd_kp_cam_3_to_cam_1, axis, pcd_cam_0])

pcd_cam_0_to_marker = o3d.geometry.PointCloud()

pcd_cam_0_points = np.asarray(pcd_cam_0.points)
pcd_cam_0_points = pcd_cam_0_points[None, :, :]
pcd_cam_0_points = torch.tensor(pcd_cam_0_points, dtype=torch.float64)

pcd_cam_0_to_marker_points = projector.project_point_cloud_to_marker(pcd_cam_0_points, "cam_0")

pcd_cam_0_to_marker_points = pcd_cam_0_to_marker_points.cpu().numpy()

pcd_cam_0_to_marker_points = pcd_cam_0_to_marker_points[0]

pcd_cam_0_to_marker.points = o3d.utility.Vector3dVector(pcd_cam_0_to_marker_points)

pcd_cam_3_to_marker = o3d.geometry.PointCloud()

pcd_cam_3_points = np.asarray(pcd_cam_3.points)
pcd_cam_3_points = pcd_cam_3_points[None, :, :]
pcd_cam_3_points = torch.tensor(pcd_cam_3_points, dtype=torch.float64)

pcd_cam_3_to_marker_points = projector.project_point_cloud_to_marker(pcd_cam_3_points, "cam_3")

pcd_cam_3_to_marker_points = pcd_cam_3_to_marker_points.cpu().numpy()

pcd_cam_3_to_marker_points = pcd_cam_3_to_marker_points[0]

pcd_cam_3_to_marker.points = o3d.utility.Vector3dVector(pcd_cam_3_to_marker_points)

o3d.visualization.draw_geometries([axis, pcd_cam_3_to_marker, pcd_cam_0_to_marker])
