import numpy as np
import os
import matplotlib.pyplot as plt
from io import BytesIO
import imageio
import open3d as o3d

plt.switch_backend('Agg')  # Headless mode

data_dir = '/home/coolbot/Documents/git/dex-dynamics/dynamics/exps/all_skills/0424_122516/best_action'
save_dir = '/home/coolbot/data/visualizations/all_skills/0424_122516/best_action'
os.makedirs(save_dir, exist_ok=True)

def create_multi_view_frame(hand_pcd, obj_pcd, gt_obj, title, views, axis_limits):
    """Create a frame with multiple viewing angles"""
    fig = plt.figure(figsize=(15, 8))
    fig.suptitle(title, y=0.95, fontsize=14)

    for i, (elev, azim, subplot_title) in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, i, projection='3d')
        ax.scatter(hand_pcd[:, 0], hand_pcd[:, 1], hand_pcd[:, 2],
                  c='red', s=3, label='Hand')
        ax.scatter(obj_pcd[:, 0], obj_pcd[:, 1], obj_pcd[:, 2],
                  c='blue', s=3, label='Object')
        if gt_obj is not None:
            ax.scatter(gt_obj[:, 0], gt_obj[:, 1], gt_obj[:, 2],
                    c='green', s=3, label='Ground Truth Object', alpha=0.3)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(subplot_title)
        # Use fixed axis limits passed as argument
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

# --- Step 1: Load all data first and collect points for global limits ---
all_step_data = []
all_points_list = []
sorted_ret_paths = sorted(os.listdir(data_dir))
last_valid_ret_path = None # Keep track of the last successfully processed path for filename

print(f"Loading data from {len(sorted_ret_paths)} directories...")
for ret_path in sorted_ret_paths:
    step_dir = os.path.join(data_dir, ret_path, 'point_clouds')
    if not os.path.isdir(step_dir):
        print(f"Skipping {ret_path}, not a directory or point_clouds subdir missing.")
        continue
    try:
        hand_init_pcd = o3d.io.read_point_cloud(os.path.join(step_dir, 'init_hand_pcld.ply'))
        hand_final_pcd = o3d.io.read_point_cloud(os.path.join(step_dir, 'pred_hand_pcld.ply'))
        obj_init_pcd = o3d.io.read_point_cloud(os.path.join(step_dir, 'init_obj_pcld.ply'))
        obj_final_pcd = o3d.io.read_point_cloud(os.path.join(step_dir, 'target_obj_pcld.ply'))
        pred_obj_pcd = o3d.io.read_point_cloud(os.path.join(step_dir, 'pred_obj_pcld.ply'))

        # Check if point clouds are empty
        if not hand_init_pcd.has_points() or not hand_final_pcd.has_points() or \
           not obj_init_pcd.has_points() or not obj_final_pcd.has_points() or \
           not pred_obj_pcd.has_points():
            print(f"Warning: Empty point cloud found in {ret_path}. Skipping this step.")
            continue

        hand_init = np.asarray(hand_init_pcd.points)
        hand_final = np.asarray(hand_final_pcd.points)
        obj_init = np.asarray(obj_init_pcd.points)
        obj_final = np.asarray(obj_final_pcd.points)
        pred_obj = np.asarray(pred_obj_pcd.points)

        step_data = {
            "hand_init": hand_init,
            "hand_final": hand_final,
            "obj_init": obj_init,
            "obj_final": obj_final,
            "pred_obj": pred_obj,
            "ret_path": ret_path # Store original path if needed later
        }
        all_step_data.append(step_data)
        last_valid_ret_path = ret_path # Update last valid path

        # Collect points for global axis calculation
        all_points_list.extend([hand_init, hand_final, obj_init, obj_final, pred_obj])

    except Exception as e:
        print(f"Error loading data for {ret_path}: {e}. Skipping this step.")
        continue # Skip this step if loading fails

if not all_points_list:
    print("No valid point cloud data loaded. Exiting.")
    exit()

print("Calculating global axis limits...")
# --- Step 2: Calculate global axis limits ---
all_points_combined = np.concatenate(all_points_list, axis=0)
min_x, max_x = np.min(all_points_combined[:, 0]), np.max(all_points_combined[:, 0])
min_y, max_y = np.min(all_points_combined[:, 1]), np.max(all_points_combined[:, 1])
min_z, max_z = np.min(all_points_combined[:, 2]), np.max(all_points_combined[:, 2])

# Calculate ranges and find the maximum range
range_x = max_x - min_x
range_y = max_y - min_y
range_z = max_z - min_z
max_range = max(range_x, range_y, range_z)

# Calculate centers
center_x = (max_x + min_x) / 2
center_y = (max_y + min_y) / 2
center_z = (max_z + min_z) / 2

# Set new limits based on max_range, centered
new_min_x = center_x - max_range / 2
new_max_x = center_x + max_range / 2
new_min_y = center_y - max_range / 2
new_max_y = center_y + max_range / 2
new_min_z = center_z - max_range / 2
new_max_z = center_z + max_range / 2

global_axis_limits = {
    'x': (new_min_x, new_max_x),
    'y': (new_min_y, new_max_y),
    'z': (new_min_z, new_max_z)
}
print(f"Global limits calculated (equal range): {global_axis_limits}")

# --- Step 3: Generate frames using loaded data and global limits ---
frame_list = []
print(f"Generating {len(all_step_data)} steps for GIF...")
for i, step_data in enumerate(all_step_data):
    hand_init = step_data["hand_init"]
    hand_final = step_data["hand_final"]
    obj_init = step_data["obj_init"]
    obj_final = step_data["obj_final"]
    pred_obj = step_data["pred_obj"]

    # Generate frames using the global axis limits
    frame_initial = create_multi_view_frame(hand_init, obj_init, obj_final,
                                           f"Step {i} Initial State", view_config, global_axis_limits)
    frame_final = create_multi_view_frame(hand_final, pred_obj, obj_final,
                                         f"Step {i} Final State", view_config, global_axis_limits)

    # Save frames to list
    frame_list.append(frame_initial)
    frame_list.append(frame_final)
    frame_list.append(frame_final) # Add final frame twice for pause effect

# --- Step 4: Save as GIF ---
if frame_list and last_valid_ret_path:
    # Use the name part of the last successfully processed directory for the GIF filename
    gif_filename_base = os.path.splitext(last_valid_ret_path)[0]
    gif_path = os.path.join(save_dir, f'{gif_filename_base}_multiview_fixed_axes.gif')
    print(f"Saving GIF to {gif_path}...")
    imageio.mimsave(gif_path, frame_list, duration=2000, loop=0) # duration in ms
    print(f"Saved {gif_path}")
    print(f"Generated GIF with {len(all_step_data)} steps.")
elif not frame_list:
    print("No frames were generated.")
else:
    print("No valid data directories were processed, cannot determine GIF filename.")