import numpy as np
import os
import matplotlib.pyplot as plt
from io import BytesIO
import imageio
import open3d as o3d

plt.switch_backend('Agg')  # Headless mode

data_dir = '/mnt/dynamics/exps/aug_on_the_fly_hand_sampling_augment/wis3d/validation/debug_200'
save_dir = 'visualizations/aug_on_the_fly_hand_sampling_augment/validation'

os.makedirs(save_dir, exist_ok=True)

def create_multi_view_frame(hand_pcd, obj_pcd, pred_obj, title, views, axis_limits):
    """Create a frame with multiple viewing angles"""
    fig = plt.figure(figsize=(15, 8))
    fig.suptitle(title, y=0.95, fontsize=14)
    
    # Create 4 subplots with different views
    for i, (elev, azim, subplot_title) in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, i, projection='3d')
        ax.scatter(hand_pcd[:, 0], hand_pcd[:, 1], hand_pcd[:, 2], 
                  c='red', s=3, label='Hand')
        ax.scatter(obj_pcd[:, 0], obj_pcd[:, 1], obj_pcd[:, 2], 
                  c='blue', s=3, label='Object')
        if pred_obj is not None:
            ax.scatter(pred_obj[:, 0], pred_obj[:, 1], pred_obj[:, 2], 
                    c='green', s=3, label='Predicted Object')
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(subplot_title)
        ax.set_xlim(*axis_limits['x'])
        ax.set_ylim(*axis_limits['y'])
        ax.set_zlim(*axis_limits['z'])
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    return imageio.imread(buf)

# Define multiple viewing angles (elevation, azimuth, title)
view_config = [
    (30, 45, "Front-Isometric View"),
    (90, 0, "Top View"),
    (0, 0, "Front View"),
    (0, 90, "Side View")
]

for ret_path in sorted(os.listdir(data_dir)):
    # ret_file_path = os.path.join(data_dir, ret_file)
    # ret_data = np.load(ret_file_path, allow_pickle=True).item()
    
    # hand_init = ret_data["hand_init_pcd"]
    # hand_final = ret_data["hand_final_pcd"]
    # obj_init = ret_data["object_init_pcd"]
    # obj_final = ret_data["object_final_pcd"]

    hand_init = o3d.io.read_point_cloud(os.path.join(data_dir, ret_path, 'point_clouds', 'current_hand_pos.ply')).points
    hand_final = o3d.io.read_point_cloud(os.path.join(data_dir, ret_path, 'point_clouds', 'gt_hand_pos.ply')).points
    obj_init = o3d.io.read_point_cloud(os.path.join(data_dir, ret_path, 'point_clouds', 'current_obj_pos.ply')).points
    obj_final = o3d.io.read_point_cloud(os.path.join(data_dir, ret_path, 'point_clouds', 'gt_obj_pos.ply')).points
    pred_obj = o3d.io.read_point_cloud(os.path.join(data_dir, ret_path, 'point_clouds', 'pred_obj_pos.ply')).points

    hand_init = np.asarray(hand_init)
    hand_final = np.asarray(hand_final)
    obj_init = np.asarray(obj_init)
    obj_final = np.asarray(obj_final)
    pred_obj = np.asarray(pred_obj)

    # # transfer z axis
    # hand_init[:, 2] = -hand_init[:, 2]
    # hand_final[:, 2] = -hand_final[:, 2]
    # obj_init[:, 2] = -obj_init[:, 2]
    # obj_final[:, 2] = -obj_final[:, 2]
    # pred_obj[:, 2] = -pred_obj[:, 2]
    
    # Calculate axis limits from all data
    all_points = np.concatenate([hand_init, hand_final, obj_init, obj_final, pred_obj], axis=0)
    axis_limits = {
        'x': (np.min(all_points[:, 0]), np.max(all_points[:, 0])),
        'y': (np.min(all_points[:, 1]), np.max(all_points[:, 1])),
        'z': (np.min(all_points[:, 2]), np.max(all_points[:, 2]))
    }
    
    # Generate frames
    frame_initial = create_multi_view_frame(hand_init, obj_init, None,
                                           "Initial State", view_config, axis_limits)
    frame_final = create_multi_view_frame(hand_final, obj_final, pred_obj,
                                         "Final State", view_config, axis_limits)
    
    # Save as GIF
    gif_path = os.path.join(save_dir, f'{os.path.splitext(ret_path)[0]}_multiview.gif')
    imageio.mimsave(gif_path, [frame_initial, frame_final,frame_final], duration=2000, loop=0)

    print(f"Saved {gif_path}")

print(f"Generated {len(os.listdir(data_dir))} GIFs in {save_dir}")