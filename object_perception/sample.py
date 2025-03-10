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
from loss import *


calib_path = '/home/coolbot/data/calib'
# object_path = '/home/coolbot/data/hand_object_perception/train/scene_0101_0'

# hand_ret_path = '/home/coolbot/data/hand_obj_ret/new/hand_obj_ret_0101.npy'


n_particles = 300

FILTER_MIN = np.array([0.0 - 0.1, 0.0 - 0.1, 0.0 - 0.07])
FILTER_MAX = np.array([0.0 + 0.07, 0.0 + 0.07, 0.0 + 0.07])



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

def project_point_cloud_to_marker(pcd, camera_sn):
    projector = Projector(calib_path=calib_path)

    pcd_marker = o3d.geometry.PointCloud()

    pcd_color = np.asarray(pcd.colors)
    pcd_points = np.asarray(pcd.points)

    pcd_marker_points = projector.project_point_cloud_to_marker(pcd_points, camera_sn)

    pcd_marker.points = o3d.utility.Vector3dVector(pcd_marker_points)
    pcd_marker.colors = o3d.utility.Vector3dVector(pcd_color)

    return pcd_marker

def filter_point_cloud(pcd):
    pcd_points = np.asarray(pcd.points)

    x_filter = (pcd_points.T[0] > FILTER_MIN[0]) & (pcd_points.T[0] < FILTER_MAX[0])
    y_filter = (pcd_points.T[1] > FILTER_MIN[1]) & (pcd_points.T[1] < FILTER_MAX[1])
    z_filter = (pcd_points.T[2] > FILTER_MIN[2]) & (pcd_points.T[2] < FILTER_MAX[2])

    filter = x_filter & y_filter & z_filter

    pcd.points = o3d.utility.Vector3dVector(pcd_points[filter])
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[filter])

    return pcd

def merge_point_clouds(frame_id, camera_list, object_path):
    projector = Projector(calib_path=calib_path)
    pcd_all_list = []
    for camera_sn in camera_list:
        color_image_path = f'{object_path}/{camera_sn}/color'
        depth_image_path = f'{object_path}/{camera_sn}/depth'

        color_image_files = sorted(os.listdir(color_image_path))
        depth_image_files = sorted(os.listdir(depth_image_path))

        color = np.array(Image.open(f'{color_image_path}/{color_image_files[frame_id]}'))
        depth = np.array(Image.open(f'{depth_image_path}/{depth_image_files[frame_id]}'))

        intrinsics = projector.intrinsics[camera_sn]

        pcd = rgbd_image_to_point_cloud(color, depth, intrinsics)

        pcd_marker = project_point_cloud_to_marker(pcd, camera_sn)

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

# def check_if_close(cube, tool_list):
#     cube_hull, _ = cube.compute_convex_hull()
#     f = SDF(cube_hull.vertices, cube_hull.triangles)
    
#     # a sparse pass
#     for _, tool_surface in tool_list:
#         tool_surface_sparse = tool_surface.voxel_down_sample(voxel_size=0.01)
#         sdf = f(np.asarray(tool_surface_sparse.points))
#         n_points_close = np.sum(sdf > 0)
#         if n_points_close > 0:
#             return True

#     return False


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

    # visualize = True

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
    # sampled_points, _ = inside_hand_filter(sampled_points, hand_mesh=hand_mesh, visualize=visualize)
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

    ##### (optional) 8. surface sampling #####
    surface_sample = False
    if surface_sample:
        ##### 7. farthest point sampling to downsample the pcd to 300 points #####
        # fps_points = fps(np.asarray(sampled_pcd.points), args.n_particles * 10)
        # fps_pcd = o3d.geometry.PointCloud()
        # fps_pcd.points = o3d.utility.Vector3dVector(fps_points)

        selected_mesh = alpha_shape_mesh_reconstruct(sampled_pcd, alpha=0.005, visualize=visualize)
        
        if not args.correspondance or pcd_dense_prev is None:
            selected_surface = o3d.geometry.TriangleMesh.sample_points_poisson_disk(selected_mesh, args.n_particles)
            surface_points = np.asarray(selected_surface.points)
        else:
            tri_mesh = trimesh.Trimesh(np.asarray(selected_mesh.vertices), np.asarray(selected_mesh.triangles),
                                        vertex_normals=np.asarray(selected_mesh.vertex_normals))
            mesh_q = trimesh.proximity.ProximityQuery(tri_mesh)
            prox_points, distance, triangle_id = mesh_q.on_surface(np.asarray(pcd_sparse_prev.points))
            selector = (distance > 0.0)[..., None]
            surface_points = prox_points * selector + np.asarray(pcd_sparse_prev.points) * (1 - selector)
        
        surface_pcd = o3d.geometry.PointCloud()
        surface_pcd.points = o3d.utility.Vector3dVector(surface_points)

        if visualize:
            visualize_o3d([surface_pcd], title='surface_point_cloud', pcd_color=color_avg)

        selected_pcd = surface_pcd
    else:
        ##### 7. farthest point sampling to downsample the pcd to 300 points #####
        fps_points = fps(np.asarray(sampled_pcd.points), n_particles)
        fps_pcd = o3d.geometry.PointCloud()
        fps_pcd.points = o3d.utility.Vector3dVector(fps_points)

        if visualize:
            visualize_o3d([fps_pcd, hand_mesh], title='fps_point_cloud', pcd_color=color_avg)

        selected_pcd = fps_pcd

    return sampled_pcd, selected_pcd


def load_hand_mesh(hand_ret, frame_id):
    hand_verts = hand_ret['pred_verts_3d'][0]
    # hand_faces = hand_ret['hand_faces'][0]
    hand_faces = np.load('/home/coolbot/Documents/git/dex-dynamics/mano_faces.npy', allow_pickle=True)

    # change to numpy array
    hand_verts = np.array(hand_verts).astype(np.float32)
    projector = Projector(calib_path=calib_path)
    hand_verts = projector.project_point_cloud_to_marker(hand_verts, 'cam_0')
    hand_faces = np.array(hand_faces)

    hand_mesh = o3d.geometry.TriangleMesh()
    hand_mesh.vertices = o3d.utility.Vector3dVector(hand_verts)
    hand_mesh.triangles = o3d.utility.Vector3iVector(hand_faces)

    return hand_mesh

BAD_SCENE = [2, 3, 4, 5, 6, 7,]

def main(pcd_dense_prev=None, pcd_sparse_prev=None):
    # camera_list = ['cam_0', 'cam_1', 'cam_2', 'cam_3']
    camera_list = ['cam_0', 'cam_1', 'cam_3']

    # data_path = '/home/coolbot/data/hand_object_perception'
    data_path = '/home/coolbot/data'

    # train_dir = os.path.join(data_path, 'train_4cameras')
    train_dir = os.path.join(data_path, 'train')
    
    visualize = True
    is_moving_back = False
    # Step 1: Get point cloud from each camera, project to marker frame, and filter.

    with open('/home/coolbot/Documents/git/dex-dynamics/object_perception/hand_perception_label.yaml' , 'r') as f:
        hand_perception_label = yaml.load(f, Loader=yaml.FullLoader)

    chamfer = Chamfer()
    
    for scene_dir in sorted(os.listdir(train_dir)):
        scene_path = os.path.join(train_dir, scene_dir)
        scene_len = len(os.listdir(os.path.join(scene_path, 'cam_0', 'color')))
        scene_idx = int(scene_dir[6:10])
        
        if scene_idx in BAD_SCENE:
            continue

        if scene_idx != 105:
            continue

        object_idx_list = [3, scene_len - 1]
        # object_idx_list = [3, 5]

        hand_idx_list = hand_perception_label["hand_idx"][scene_idx]

        hand_ret_path = os.path.join(data_path, 'pred_wo_cam2', f'pred_scene_{scene_idx:04d}.npy')
        hand_ret_list = np.load(hand_ret_path, allow_pickle=True)

        hand_obj_ret = {}

        for object_idx, hand_idx in zip(object_idx_list, hand_idx_list):
            pcd = merge_point_clouds(frame_id=object_idx, camera_list=camera_list, object_path=scene_path)
            
            if visualize:
                visualize_o3d([pcd], title='merged_point_cloud')
            
            hand_ret = hand_ret_list[hand_idx]

            hand_mesh = load_hand_mesh(hand_ret, frame_id=hand_idx)

            pcd_dense, pcd_sparse = sample(pcd, pcd_dense_prev, pcd_sparse_prev, hand_mesh,
                is_moving_back, patch=False, visualize=visualize)

            if object_idx == object_idx_list[0]:
                hand_obj_ret['object_init_pcd'] = np.asarray(pcd_sparse.points)
                hand_obj_ret['hand_init_pcd'] = np.asarray(hand_mesh.vertices)
            else:
                hand_obj_ret['object_final_pcd'] = np.asarray(pcd_sparse.points)
                hand_obj_ret['hand_final_pcd'] = np.asarray(hand_mesh.vertices)

                # Step 2: Save the point cloud data to a file.
                # np.save(f'/home/coolbot/data/hand_obj_ret_inside_hand_filter/hand_obj_ret_{scene_idx:04d}.npy', hand_obj_ret)
                # print(f'Saved hand object ret for scene {scene_idx}.')
        obj_init = hand_obj_ret['object_init_pcd']
        obj_final = hand_obj_ret['object_final_pcd']
        obj_init = torch.tensor(obj_init).float().unsqueeze(0)
        obj_final = torch.tensor(obj_final).float().unsqueeze(0)

        print(f"chamfer loss of {scene_dir}: {chamfer(obj_init, obj_final)}")


if __name__ == '__main__':
    main()