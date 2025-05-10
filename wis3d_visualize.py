from wis3d import Wis3D
import numpy as np
import os
import torch
import open3d as o3d

# wis3d_dir = "/home/coolbot/data/0402_111413/wis3d/mppi_step_1_iter0029_plan"

# for ret_file in sorted(os.listdir(wis3d_dir)):
#     pcd_file = os.path.join(wis3d_dir, ret_file, "point_clouds")
#     obj_pcd_file = os.path.join(pcd_file, "obj_pcld.ply")

#     obj_pcd = o3d.io.read_point_cloud(obj_pcd_file)
#     obj_pcd_points = np.asarray(obj_pcd.points)

#     print(obj_pcd_points.shape)

#     # store the object point cloud
#     file_number = int(ret_file)
#     np.save(f"target_obj_pcd_{file_number:04d}", obj_pcd_points)

wis3d = Wis3D(
    out_folder="wis3d",
    sequence_name="a",
    xyz_pattern=("x", "-y", "-z"),
)

