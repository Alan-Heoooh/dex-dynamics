import os 
import numpy as np
import torch
import yaml
import open3d as o3d
import trimesh

# file = '/home/coolbot/data/hand_obj_ret/new/hand_obj_post_processing_ret_0101.npy'

# data = np.load(file, allow_pickle=True).item()
# object_init = data['object_init_pcd']
# object_final = data['object_final_pcd']
# hand_init = data['hand_init_pcd'][0]
# hand_final = data['hand_final_pcd'][0]

# hand_init = np.array(hand_init)
# hand_final = np.array(hand_final)

# object_init_pcd = o3d.geometry.PointCloud()
# object_init_pcd.points = o3d.utility.Vector3dVector(object_init)
# # object_init_pcd.paint_uniform_color([0.1, 0.1, 0.7])
# # purple
# object_init_pcd.paint_uniform_color([1, 0, 1])
# object_final_pcd = o3d.geometry.PointCloud()
# object_final_pcd.points = o3d.utility.Vector3dVector(object_final)
# object_final_pcd.paint_uniform_color([0.1, 0.1, 0.7])
# hand_init_pcd = o3d.geometry.PointCloud()
# hand_init_pcd.points = o3d.utility.Vector3dVector(hand_init)
# # hand_init_pcd.paint_uniform_color([0.1, 0.7, 0.1])
# # red
# hand_init_pcd.paint_uniform_color([1, 0, 0])
# hand_final_pcd = o3d.geometry.PointCloud()
# hand_final_pcd.points = o3d.utility.Vector3dVector(hand_final)
# hand_final_pcd.paint_uniform_color([0.1, 0.7, 0.1])

# axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=[0, 0, 0])

# o3d.visualization.draw_geometries([object_init_pcd, hand_init_pcd, axis])

# o3d.visualization.draw_geometries([object_init_pcd, object_final_pcd, axis])

camera_sn = 'cam_0'

scene_file = '/home/coolbot/data/hand_object_perception/train/scene_0101_0'
ret_file = '/home/coolbot/data/exps/scene_0101_0'
camera_wilor_file = os.path.join(ret_file, f'{camera_sn}_wilor_data.pt')

camera_wilor_data = torch.load(camera_wilor_file)

camera_dir = os.path.join(scene_file, camera_sn)

camera_color_dir = os.path.join(camera_dir, 'color')
camera_depth_dir = os.path.join(camera_dir, 'depth')

camera_wilor = os.path.join(scene_file, f'{camera_sn}_wilor')

camera_color_files = sorted(os.listdir(camera_color_dir))
camera_depth_files = sorted(os.listdir(camera_depth_dir))

intrinsics = {}
intrinsics['cam_0'] = np.array([[909.22192383,   0.        , 634.7142334 ,   0.        ],
                                    [  0.        , 909.41491699, 352.74398804,   0.        ],
                                    [  0.        ,   0.        ,   1.        ,   0.        ]])
intrinsics['cam_1'] = np.array([[912.91558838,   0.        , 661.25982666,   0.        ],
                            [  0.        , 912.52545166, 373.5128479 ,   0.        ],
                            [  0.        ,   0.        ,   1.        ,   0.        ]])
intrinsics['cam_3'] = np.array([[910.8637085 ,   0.        , 619.1239624 ,   0.        ],
                            [  0.        , 910.2946167 , 351.13458252,   0.        ],
                            [  0.        ,   0.        ,   1.        ,   0.        ]])

for scene_id in range(0, len(camera_color_files), 5):
    color_file = os.path.join(camera_color_dir, camera_color_files[scene_id])
    depth_file = os.path.join(camera_depth_dir, camera_depth_files[scene_id])
    wilor_file = os.path.join(camera_wilor, camera_color_files[scene_id].replace('.png', '_0.obj'))

    color = o3d.io.read_image(color_file)
    depth = o3d.io.read_image(depth_file)
    # wilor = o3d.io.read_triangle_mesh(wilor_file)
    # hand_mesh = trimesh.load(wilor_file)
    # # hand_mesh = o3d.geometry.TriangleMesh.create_from_trimesh(hand_mesh)


    # hand_mesh_o3d = o3d.geometry.TriangleMesh()
    # hand_mesh_o3d.vertices = o3d.utility.Vector3dVector(hand_mesh.vertices)
    # hand_mesh_o3d.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)

    verts = camera_wilor_data['verts'][scene_id][0]

    verts = np.array(verts)
    verts_pcd = o3d.geometry.PointCloud()
    verts_pcd.points = o3d.utility.Vector3dVector(verts)
    verts_pcd.paint_uniform_color([0.1, 0.1, 0.7])

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_scale=1000.0, depth_trunc=1.0, convert_rgb_to_intensity=False)

    fx, fy, cx, cy = intrinsics[camera_sn][0, 0], intrinsics[camera_sn][1, 1], intrinsics[camera_sn][0, 2], intrinsics[camera_sn][1, 2]
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(1280, 720, fx, fy, cx, cy))
    
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])


    o3d.visualization.draw_geometries([pcd, verts_pcd, axis])

    # import pdb; pdb.set_trace()