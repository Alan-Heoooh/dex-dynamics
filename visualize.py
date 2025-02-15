import os 
import numpy as np
import torch
import yaml
import open3d as o3d


file = '/home/coolbot/data/hand_obj_ret/hand_obj_ret_0.npy'

data_list = np.load(file, allow_pickle=True)

for data in data_list:
    hand_verts = data['verts']
    hand_faces = data['hand_faces']
    # convert to numpy array
    hand_verts = np.array(hand_verts[0])
    hand_faces = np.array(hand_faces[0])
    obj_pcd_sparse = data['pcd_sparse']

    hand_mesh = o3d.geometry.TriangleMesh()
    hand_mesh.vertices = o3d.utility.Vector3dVector(hand_verts)
    hand_mesh.triangles = o3d.utility.Vector3iVector(hand_faces)

    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(obj_pcd_sparse)

    o3d.visualization.draw_geometries([hand_mesh, obj_pcd])