import os
import numpy as np
import yaml
import open3d as o3d
import torch
import cv2 as cv

from PIL import Image

from projector_np import Projector
from pcd_utils import *
from utils.visualize import *
from utils.constants import *
from loss import *

from wis3d import Wis3D

def rgbd_image_to_point_cloud(color, depth, intrinsics):
    _h = color.shape[0]
    _w = color.shape[1]
    color = o3d.geometry.Image(color)
    depth = o3d.geometry.Image(depth)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_scale=1000.0, convert_rgb_to_intensity=False)

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    cam_intrinsic = o3d.camera.PinholeCameraIntrinsic(_w, _h, fx, fy, cx, cy)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam_intrinsic)

    return pcd

def project_point_cloud_to_marker(pcd, camera_sn, projector=None):
    pcd_marker = o3d.geometry.PointCloud()

    pcd_color = np.asarray(pcd.colors)
    pcd_points = np.asarray(pcd.points)

    pcd_marker_points = projector.project_point_cloud_to_marker(pcd_points, camera_sn)

    pcd_marker.points = o3d.utility.Vector3dVector(pcd_marker_points)
    pcd_marker.colors = o3d.utility.Vector3dVector(pcd_color)

    return pcd_marker

def filter_point_cloud(pcd, filter_min=FILTER_MIN, filter_max=FILTER_MAX):
    pcd_points = np.asarray(pcd.points)

    x_filter = (pcd_points.T[0] > filter_min[0]) & (pcd_points.T[0] < filter_max[0])
    y_filter = (pcd_points.T[1] > filter_min[1]) & (pcd_points.T[1] < filter_max[1])
    z_filter = (pcd_points.T[2] > filter_min[2]) & (pcd_points.T[2] < filter_max[2])

    filter = x_filter & y_filter & z_filter

    pcd.points = o3d.utility.Vector3dVector(pcd_points[filter])
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[filter])

    return pcd

def merge_point_clouds(frame_id, camera_list, object_path, projector=None):
    pcd_all_list = []
    for camera_sn in camera_list:
        color_image_path = f'{object_path}/{camera_sn}/color'
        depth_image_path = f'{object_path}/{camera_sn}/depth'

        color_image_files = sorted(os.listdir(color_image_path))
        depth_image_files = sorted(os.listdir(depth_image_path))

        color = np.array(Image.open(f'{color_image_path}/{color_image_files[frame_id]}'))
        depth = np.array(Image.open(f'{depth_image_path}/{depth_image_files[frame_id]}'))

        intrinsics = INTRINSICS[camera_sn]

        pcd = rgbd_image_to_point_cloud(color, depth, intrinsics)
        pcd_marker = project_point_cloud_to_marker(pcd, camera_sn, projector=projector)
        pcd_marker = filter_point_cloud(pcd_marker)
        pcd_all_list.append(pcd_marker)

    # o3d.visualization.draw_geometries(pcd_all)

    pcd_all = o3d.geometry.PointCloud()
    for point_id in range(len(pcd_all_list)):
        pcd_all += pcd_all_list[point_id]

    return pcd_all


# @profile
def preprocess_raw_pcd(pcd_all, rm_stats_outliers=2, visualize=False):

    # color segmentation
    pcd_colors = np.asarray(pcd_all.colors, dtype=np.float32)
    # bgr
    pcd_rgb = pcd_colors[None, :, :]

    pcd_hsv = cv.cvtColor(pcd_rgb, cv.COLOR_RGB2HSV)
    hsv_lower = np.array([0, 0.5, 0], dtype=np.float32)
    hsv_upper = np.array([360, 1, 1], dtype=np.float32)
    mask = cv.inRange(pcd_hsv, hsv_lower, hsv_upper)
    cube_label = np.where(mask[0] == 255)

    # cube_label = np.where((pcd_colors[:, 0] > 0.7) & (pcd_colors[:, 1] > 0.7) & (pcd_colors[:, 2] > 0.7))
    cube = pcd_all.select_by_index(cube_label[0])
    # rest_label = np.where(np.logical_and(pcd_colors[:, 0] > 0.5, pcd_colors[:, 2] > 0.2))
    # rest = pcd_all.select_by_index(rest_label[0])
    rest = pcd_all.select_by_index(cube_label[0], invert=True)

    if visualize:
        visualize_o3d([cube], title='selected_dough')

    if visualize:
        visualize_o3d([rest], title='discarded_part')

    cube = cube.voxel_down_sample(voxel_size=0.001)

    if rm_stats_outliers:
        rm_iter = 0
        outliers = None
        outlier_stat = None
        # remove until there's no new outlier
        while rm_iter < 1: # outlier_stat is None or len(outlier_stat.points) > 0:
            cl, inlier_ind_cube_stat = cube.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.5+0.5*rm_iter)
            cube_stat = cube.select_by_index(inlier_ind_cube_stat)
            outlier_stat = cube.select_by_index(inlier_ind_cube_stat, invert=True)
            if outliers is None:
                outliers = outlier_stat
            else:
                outliers += outlier_stat

            # print(len(outlier.points))
            
            # cl, inlier_ind_cube_stat = cube.remove_radius_outlier(nb_points=50, radius=0.05)
            # cube_stat = cube.select_by_index(inlier_ind_cube_stat)
            # outliers = cube.select_by_index(inlier_ind_cube_stat, invert=True)

            cube = cube_stat
            rm_iter += 1

            # press needs those points
            # if 'press' in args.env or rm_stats_outliers == 1: break

        # cl, inlier_ind_rest_stat = rest.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.5)
        # rest_stat = rest.select_by_index(inlier_ind_rest_stat)
        # rest = rest_stat

        if visualize:
            outliers.paint_uniform_color([0.0, 0.8, 0.0])
            visualize_o3d([cube, outliers], title='cleaned_dough')

    # rest = rest.voxel_down_sample(voxel_size=0.004)

    # if visualize:
    #     visualize_o3d([cube, rest], title='downsampled_dough_and_tool')

    # n_bins = 30
    # cube_points = np.asarray(cube.points)
    # cube_colors = np.asarray(cube.colors)
    # cube_z_hist_array = np.histogram(cube_points[:, 2], bins=n_bins)
    # # cube_z_count_max = np.argmax(cube_z_hist_array[0])
    # cube_z_cap = cube_z_hist_array[1][n_bins - 1]
    # selected_idx = cube_points[:, 2] < cube_z_cap
    # cube_points = cube_points[selected_idx]
    # cube_colors = cube_colors[selected_idx]

    # cube = o3d.geometry.PointCloud()
    # cube.points = o3d.utility.Vector3dVector(cube_points)
    # cube.colors = o3d.utility.Vector3dVector(cube_colors)

    # if visualize:
    #     visualize_o3d([cube], title='cube_under_ceiling')

    return cube, rest


def inside_hand_filter(sampled_points, hand_mesh, in_d=0.0, close_d=-0.01, visualize=False):
    sdf_all = np.full(sampled_points.shape[0], True)

    n_points_close = 0
    # for tool_mesh, _ in tool_list:
    f = SDF(hand_mesh.vertices, hand_mesh.triangles)
    sdf = f(sampled_points)
    sdf_all &= sdf < in_d
    n_points_close += np.sum(sdf > close_d)
    
    out_tool_points = sampled_points[sdf_all, :]
    in_tool_points = sampled_points[~sdf_all, :]

    if visualize:
        out_tool_pcd = o3d.geometry.PointCloud()
        out_tool_pcd.points = o3d.utility.Vector3dVector(out_tool_points)
        out_tool_pcd.paint_uniform_color([0.6, 0.6, 0.6])
        
        in_tool_pcd = o3d.geometry.PointCloud()
        in_tool_pcd.points = o3d.utility.Vector3dVector(in_tool_points)
        in_tool_pcd.paint_uniform_color([0.0, 0.0, 0.0])

        visualize_o3d([hand_mesh, out_tool_pcd, in_tool_pcd], title='inside_hand_filter')

    return out_tool_points, in_tool_points

# @profile
def sample(pcd, pcd_dense_prev, pcd_sparse_prev, hand_mesh, is_moving_back, patch=False, visualize=False):
    if pcd_dense_prev is not None and is_moving_back: 
        return pcd_dense_prev, pcd_sparse_prev
    
    n_particles = 300

    cube, rest = preprocess_raw_pcd(pcd, visualize=visualize)
    # is_close = check_if_close(cube, tool_list)

    # if pcd_dense_prev is not None and not is_close: 
    #     ##### 5.a apply temporal prior: copy the soln from the last frame #####
    #     return pcd_dense_prev, pcd_sparse_prev
    
    cube_colors = np.asarray(cube.colors)
    color_avg = list(np.mean(cube_colors, axis=0))

    # if visualize:
    #     visualize_o3d([hand_mesh], title='hand_mesh')
    
    ##### 1. random sample 100x points in the bounding box #####
    lower = cube.get_min_bound()
    upper = cube.get_max_bound()
    sample_size = 50 * n_particles
    sampled_points = np.random.rand(sample_size, 3) * (upper - lower) + lower

    # if patch and is_close:
    #     cube = inside_cube_filter(cube, tool_list, visualize=visualize)

    # #### 2.a use SDF to filter out points OUTSIDE the convex hull #####

    # f = SDF(convex_hull.vertices, convex_hull.triangles)
    # sdf = f(sampled_points)
    # sampled_points = sampled_points[sdf > 0, :]

    # cube = cube.voxel_down_sample(voxel_size=0.004)

    selected_mesh = poisson_mesh_reconstruct(cube, depth=6, mesh_fix=True, visualize=visualize)
    f = SDF(selected_mesh.points, selected_mesh.faces.reshape(selected_mesh.n_faces, -1)[:, 1:])
    
    sdf = f(sampled_points)
    sampled_points = sampled_points[sdf > 0]

    ##### 3. use SDF to filter out points INSIDE the tool mesh #####
    if hand_mesh is not None:
        sampled_points, _ = inside_hand_filter(sampled_points, hand_mesh=hand_mesh, visualize=visualize)
    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)

    if visualize:
        visualize_o3d([sampled_pcd, cube, hand_mesh], title='sampled_points')

    ##### 6. filter out the noise #####
    cl, inlier_ind_stat = sampled_pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.5)
    sampled_pcd_stat = sampled_pcd.select_by_index(inlier_ind_stat)
    outliers_stat = sampled_pcd.select_by_index(inlier_ind_stat, invert=True)

    sampled_pcd = sampled_pcd_stat
    outliers = outliers_stat

    if visualize:
        sampled_pcd.paint_uniform_color([0.0, 0.8, 0.0])
        outliers.paint_uniform_color([0.8, 0.0, 0.0])
        visualize_o3d([cube, sampled_pcd, outliers], title='cleaned_point_cloud', pcd_color=color_avg)

    ##### 7. farthest point sampling to downsample the pcd to 300 points #####
    fps_points = fps(np.asarray(sampled_pcd.points), n_particles)
    fps_pcd = o3d.geometry.PointCloud()
    fps_pcd.points = o3d.utility.Vector3dVector(fps_points)

    if visualize:
        visualize_o3d([fps_pcd, hand_mesh], title='fps_point_cloud', pcd_color=color_avg)

    selected_pcd = fps_pcd

    return sampled_pcd, selected_pcd, selected_mesh


def load_hand_mesh(hand_ret, frame_id, master_camera_sn='cam_0', projector=None):
    hand_verts = hand_ret['pred_verts_3d'][0]
    hand_faces = np.load('object_perception/mano_faces.npy', allow_pickle=True)

    # change to numpy array
    hand_verts = np.array(hand_verts).astype(np.float32)
    hand_verts = projector.project_point_cloud_to_marker(hand_verts, master_camera_sn)
    hand_faces = np.array(hand_faces)

    hand_mesh = o3d.geometry.TriangleMesh()
    hand_mesh.vertices = o3d.utility.Vector3dVector(hand_verts)
    hand_mesh.triangles = o3d.utility.Vector3iVector(hand_faces)

    return hand_mesh

def hand_obj_sample(hand_idx_list, object_idx_list, hand_ret_list, scene_path, camera_list, master_camera_sn, visualize=False, negative_data=False, projector=None):
    pcd_dense_prev = None
    pcd_sparse_prev = None
    is_moving_back = False
    hand_obj_ret = {}
    pcd_init = merge_point_clouds(frame_id=object_idx_list[0], camera_list=camera_list, object_path=scene_path, projector=projector)
    pcd_final = merge_point_clouds(frame_id=object_idx_list[1], camera_list=camera_list, object_path=scene_path, projector=projector)

    if visualize:
        visualize_o3d([pcd_init], title='object_init')
        visualize_o3d([pcd_final], title='object_final')

    hand_ret_init, hand_ret_final = hand_ret_list[hand_idx_list[0]], hand_ret_list[hand_idx_list[1]]
    hand_mesh_init = load_hand_mesh(hand_ret_init, frame_id=hand_idx_list[0], master_camera_sn=master_camera_sn, projector=projector)
    hand_mesh_final = load_hand_mesh(hand_ret_final, frame_id=hand_idx_list[1], master_camera_sn=master_camera_sn, projector=projector)
    pcd_dense_init, pcd_sparse_init, _ = sample(pcd_init, pcd_dense_prev, pcd_sparse_prev, hand_mesh_init, is_moving_back, patch=False, visualize=visualize)
    if negative_data:
        pcd_sparse_final = pcd_sparse_init
        pcd_dense_final = pcd_dense_init
    else:
        pcd_dense_final, pcd_sparse_final, _ = sample(pcd_final, pcd_dense_prev, pcd_sparse_prev, hand_mesh_final, is_moving_back, patch=False, visualize=visualize)

    hand_obj_ret['object_init_pcd'] = np.asarray(pcd_sparse_init.points)
    hand_obj_ret['object_final_pcd'] = np.asarray(pcd_sparse_final.points)
    hand_obj_ret['hand_init_pcd'] = np.asarray(hand_mesh_init.vertices)
    hand_obj_ret['hand_final_pcd'] = np.asarray(hand_mesh_final.vertices)
    hand_obj_ret['object_init_pcd_dense'] = np.asarray(pcd_dense_init.points)
    hand_obj_ret['object_final_pcd_dense'] = np.asarray(pcd_dense_final.points)

    return hand_obj_ret

def random_hand_init_frame(hand_idx_list_all, n_hand_samples):
    # Generate unique random indices
    available_indices = np.arange(hand_idx_list_all[0], hand_idx_list_all[1] + 1)
    n_available = len(available_indices)

    # Ensure we don't request more samples than available
    if n_hand_samples > n_available:
        hand_init_list = np.random.randint(hand_idx_list_all[0], hand_idx_list_all[1] + 1, n_hand_samples)
    else:
        hand_init_list = np.random.choice(
            available_indices,
            size=n_hand_samples,
            replace=False  # Ensures no duplicates
        )

    return hand_init_list


def main(pcd_dense_prev=None, pcd_sparse_prev=None):
    visualize = False

    n_hand_samples = 4 # 4
    n_negative_hand_samples = 4 # 4
    master_camera_sn = 'cam_0'
    camera_list = ['cam_0', 'cam_1', 'cam_2', 'cam_3']
    calib_path = '/home/coolbot/data/calib'
    data_path = '/home/coolbot/data/hand_object_perception'
    hand_ret_file = 'pred_hand_data_0413_thumb_press'
    # train_dir = os.path.join(data_path, 'train_0313')
    # train_dir  = '/media/coolbot/Extreme Pro/data/train_0412_palm_press_finger_press'
    train_dir  = '/media/coolbot/Extreme Pro/data/train_0413_thumb_press'
    save_ret_dir = '/home/coolbot/data/hand_obj_ret_0413_thumb_press'
    os.makedirs(save_ret_dir, exist_ok=True)
    
    with open('/home/coolbot/Documents/git/dex-dynamics/object_perception/hand_perception_label_integrate.yaml' , 'r') as f:
        hand_perception_label = yaml.load(f, Loader=yaml.FullLoader)

    # chamfer = Chamfer()
    # emd = EMDCPU()

    projector = Projector(calib_path=calib_path)

    BAD_SCENE = [301, 302, 307, 308, 311, 314, 316, 325, 326, 344, 359, 362, 364, 366, 395, 397,
                702, 704, 708, 716, 717, 726, 728, 734, 750]

    for scene_dir in sorted(os.listdir(train_dir)):
        scene_path = os.path.join(train_dir, scene_dir)
        scene_len = len(os.listdir(os.path.join(scene_path, 'cam_0', 'color')))
        scene_idx = int(scene_dir[6:10])

        # if scene_idx in BAD_SCENE:
        #     print(f'Skipping scene {scene_idx}.')
        #     continue

        # hand perception frames
        hand_ret_path = os.path.join(data_path, hand_ret_file, f'pred_scene_{scene_idx:04d}.npy')
        hand_ret_list = np.load(hand_ret_path, allow_pickle=True)
        hand_idx_list_all = hand_perception_label["hand_idx"][scene_idx]
        print(f'initial frame: {hand_idx_list_all[0]}, contact frame: {hand_idx_list_all[1]}, hand end frame: {hand_idx_list_all[2]}')

        hand_init_list = random_hand_init_frame(hand_idx_list_all, n_hand_samples)

        # object perception frames
        for i, hand_init_idx in enumerate(hand_init_list):
            object_idx_list = [1, scene_len - 1]
            hand_idx_list = [hand_init_idx, hand_idx_list_all[2]]
            print(f'hand_idx_list: {hand_idx_list}')

            hand_obj_ret = hand_obj_sample(hand_idx_list, object_idx_list, hand_ret_list, scene_path, camera_list, master_camera_sn, visualize=visualize, projector=projector)
            np.save(os.path.join(save_ret_dir, f'hand_obj_ret_{scene_idx:04d}-{i}.npy'), hand_obj_ret)
            print(f'Saved hand object ret for scene {scene_idx}-{i}.')
            
        # negative data
        for i in range(n_negative_hand_samples):
            object_idx_list = [1, 1]
            hand_idx_list = np.random.randint(hand_idx_list_all[0], hand_idx_list_all[1] + 1, 2)
            # hand_idx_list = sorted(hand_idx_list)
            print(f'negative hand_idx_list: {hand_idx_list}')
            hand_obj_ret = hand_obj_sample(hand_idx_list, object_idx_list, hand_ret_list, scene_path, camera_list, master_camera_sn, visualize=visualize, negative_data=True, projector=projector)
            np.save(os.path.join(save_ret_dir, f'hand_obj_ret_neg_{scene_idx:04d}-{i}.npy'), hand_obj_ret)
            print(f'Saved negative hand object ret for scene {scene_idx}-{i}.')

            # obj_init = hand_obj_ret['object_init_pcd']
            # obj_final = hand_obj_ret['object_final_pcd']
            # obj_init = torch.tensor(obj_init).float().unsqueeze(0)
            # obj_final = torch.tensor(obj_final).float().unsqueeze(0)


        # visualize the point cloud and mesh
        # import pyvista as pv
        # plt = pv.Plotter()
        # # import pdb; pdb.set_trace()
        # point_cloud_init = pv.PolyData(obj_init.squeeze().numpy())
        # point_cloud_final = pv.PolyData(obj_final.squeeze().numpy())
        # plt.add_mesh(point_cloud_init, color='red', point_size=10)
        # plt.add_mesh(point_cloud_final, color='blue', point_size=10)
        # plt.add_mesh(mesh_init, color='red', opacity=0.2)
        # plt.add_mesh(mesh_final, color='blue', opacity=0.2)
        # plt.show()

        # print(f"sdf loss of {scene_dir}: {differentiable_sdf_loss(obj_init.squeeze(), obj_final.squeeze())}")
        # print(f"occupancy loss of {scene_dir}: {differentiable_occupancy_loss(obj_init.squeeze(), obj_final.squeeze())}")
        # print(f"chamfer loss of {scene_dir}: {chamfer(obj_init, obj_final)}")
        # print(f"chamfer loss (20%) of {scene_dir}: {chamfer(obj_init, obj_final, probability=0.2)}")
        # print(f"emd loss of {scene_dir}: {emd(obj_init, obj_final)}")
        # print("*********************************************************")

if __name__ == '__main__':
    main()