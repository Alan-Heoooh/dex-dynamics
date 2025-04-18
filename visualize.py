import os 
import numpy as np
import torch
import yaml
import open3d as o3d
import trimesh
from object_perception.projector_np import Projector

from wis3d import Wis3D


FILTER_MIN = np.array([-0.2, -0.2, -0.2])
FILTER_MAX = np.array([0.2, 0.2, 0.2])


def filter_point_cloud(pcd):
    pcd_points = np.asarray(pcd.points)

    x_filter = (pcd_points.T[0] > FILTER_MIN[0]) & (pcd_points.T[0] < FILTER_MAX[0])
    y_filter = (pcd_points.T[1] > FILTER_MIN[1]) & (pcd_points.T[1] < FILTER_MAX[1])
    z_filter = (pcd_points.T[2] > FILTER_MIN[2]) & (pcd_points.T[2] < FILTER_MAX[2])

    filter = x_filter & y_filter & z_filter

    pcd.points = o3d.utility.Vector3dVector(pcd_points[filter])
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[filter])

    return pcd

root_dir = '/home/coolbot/data'
camera_sn_list = [ 'cam_0', 'cam_1', 'cam_2', 'cam_3']
master_camera_sn = camera_sn_list[-1]

calib_dir = os.path.join(root_dir, 'calib')
# train_dir = os.path.join(root_dir, 'train')

projector = Projector(calib_dir)

intrinsics = {}
intrinsics['cam_0'] = np.array([[909.22192383,   0.        , 634.7142334 ,   0.        ],
                                            [  0.        , 909.41491699, 352.74398804,   0.        ],
                                            [  0.        ,   0.        ,   1.        ,   0.        ]]).astype(np.float32)
intrinsics['cam_1'] = np.array([[912.91558838,   0.        , 661.25982666,   0.        ],
                            [  0.        , 912.52545166, 373.5128479 ,   0.        ],
                            [  0.        ,   0.        ,   1.        ,   0.        ]]).astype(np.float32)
# ERROR
intrinsics['cam_2'] = np.array([[916.56665039,   0.        , 648.18109131,   0.        ],
                                            [  0.        , 916.77130127, 358.43869019,   0.        ],
                                            [  0.        ,   0.        ,   1.        ,   0.        ]])
intrinsics['cam_3'] = np.array([[910.8637085 ,   0.        , 619.1239624 ,   0.        ],
                            [  0.        , 910.2946167 , 351.13458252,   0.        ],
                            [  0.        ,   0.        ,   1.        ,   0.        ]]).astype(np.float32)

data_dir = os.path.join(root_dir, 'hand_object_perception')

pred_file = os.path.join(data_dir, 'pred_hand_data_0316_cam3')

train_dir = os.path.join(data_dir, 'train_0313')

scene_id = 300

pred_data = np.load(os.path.join(pred_file, f'pred_scene_{scene_id:04d}.npy'), allow_pickle=True)
scene_len = len(pred_data)
scene_dir = os.path.join(train_dir, f'scene_{scene_id:04d}_0')

wis3d = Wis3D(out_folder="/home/coolbot/Documents/git/dex-dynamics/wis3d_exp",
              sequence_name="hand_trajectory_310",
              xyz_pattern=("x", "-y", "-z"),
            )

wis3d.set_scene_id(0)

scene_list = [24, 48]


# for i in range(0, scene_len, 1):
for i in scene_list:
    cloud_marker_all = o3d.geometry.PointCloud()
    for camera_sn in camera_sn_list:

        color_file = os.path.join(scene_dir, camera_sn, 'color', f'color_{i:06d}.png')
        depth_file = os.path.join(scene_dir, camera_sn, 'depth', f'depth_{i:06d}.png')

        color = o3d.io.read_image(color_file)
        depth = o3d.io.read_image(depth_file)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_scale=1000.0, depth_trunc=1.0, convert_rgb_to_intensity=False)
        
        cam_intrinsics = intrinsics[camera_sn]
        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(1280, 720, cam_intrinsics[0, 0], cam_intrinsics[1, 1], cam_intrinsics[0, 2], cam_intrinsics[1, 2])

        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsic)

        cloud_points = np.array(cloud.points)
        cloud_colors = np.array(cloud.colors)

        cloud_marker_points = projector.project_point_cloud_to_marker(cloud_points, camera_sn)
        cloud_marker = o3d.geometry.PointCloud()
        cloud_marker.points = o3d.utility.Vector3dVector(cloud_marker_points)
        cloud_marker.colors = o3d.utility.Vector3dVector(cloud_colors)
        cloud_marker_all += cloud_marker

    # pred data
    data = pred_data[i]
    if data is None:
        continue
    pred_joints_3d = np.array(data['pred_joints_3d']).astype(np.float32)
    pred_verts_3d = np.array(data['pred_verts_3d']).astype(np.float32)
    pred_ref_joints_3d = np.array(data['pred_ref_joints_3d']).astype(np.float32)

    j3d_pcd = o3d.geometry.PointCloud()
    j3d_pcd.points = o3d.utility.Vector3dVector(pred_joints_3d[0])
    verts_pcd = o3d.geometry.PointCloud()
    verts_pcd.points = o3d.utility.Vector3dVector(pred_verts_3d[0])
    # ref_j3d_pcd = o3d.geometry.PointCloud()
    # ref_j3d_pcd.points = o3d.utility.Vector3dVector(pred_ref_joints_3d[0])
    # ref_j3d_pcd.paint_uniform_color([0.0, 0.0, 1.0])

    verts_marker_points = projector.project_point_cloud_to_marker(pred_verts_3d[0],  master_camera_sn)
    verts_marker = o3d.geometry.PointCloud()
    verts_marker.points = o3d.utility.Vector3dVector(verts_marker_points)
    verts_marker.paint_uniform_color([1.0, 0.0, 0.0])

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    hand_mesh = o3d.geometry.TriangleMesh()
    hand_mesh.vertices = o3d.utility.Vector3dVector(verts_marker_points)
    hand_faces = np.load("mano_faces.npy")
    hand_mesh.triangles = o3d.utility.Vector3iVector(hand_faces)
    hand_mesh.compute_vertex_normals()
    # change to color blue
    hand_mesh.paint_uniform_color([0.0, 0.0, 1.0])

    print(f"frame {i}")
    cloud_marker_all_bbox = filter_point_cloud(cloud_marker_all)
    cloud_marker_all = cloud_marker_all_bbox
    o3d.visualization.draw_geometries([ axis, cloud_marker_all, hand_mesh])
    o3d.io.write_point_cloud(f"pcd_{i}.ply", cloud_marker_all)

    # wis3d visualization
    hand_mesh_vertices = np.array(hand_mesh.vertices)
    hand_mesh_faces = np.array(hand_mesh.triangles)
    hand_mesh_colors = np.array(hand_mesh.vertex_colors)
    wis3d.add_mesh(vertices=hand_mesh_vertices, faces=hand_mesh_faces, vertex_colors=hand_mesh_colors, name="hand_mesh")
    # wis3d.add_mesh(hand_mesh, name="hand_mesh")
    cloud_marker_all_points = np.array(cloud_marker_all.points)
    cloud_marker_all_colors = np.array(cloud_marker_all.colors)
    wis3d.add_point_cloud(vertices=cloud_marker_all_points, colors=cloud_marker_all_colors, name="cloud_marker_all")
    wis3d.increase_scene_id()