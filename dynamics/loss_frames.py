import os
import numpy as np
import torch
from dexwm.dynamics.loss import PositionLoss, Chamfer, EMDCPU, MSE, RMSE


action_per_frames = 20
dataset_dir = "/home/coolbot/data/rewarped_softness_50/pinch_trajectories_ability_hand_urdf"

scene_list = sorted(os.listdir(dataset_dir))
scene_list = [scene for scene in scene_list if scene.endswith(".npy")]


chamfer_loss = Chamfer()
emd_loss = EMDCPU()
mse_loss = MSE()
rmse_loss = RMSE()

chamfer_loss_list = []
emd_loss_list = []
mse_loss_list = []
rmse_loss_list = []


object_start_list = []
object_end_list = []
for scene in scene_list[:20]:
    hand_obj_ret = np.load(os.path.join(dataset_dir, scene), allow_pickle=True).item()
    object_sampled_particles = hand_obj_ret["object_sampled_particles"] # (T, n_object, 3)
    object_sampled_particles = torch.FloatTensor(object_sampled_particles) # (T, n_object, 3)
    
    sequence_length, n_points_object, _ = object_sampled_particles.shape
    object_sampled_particles = object_sampled_particles.permute(1, 0, 2) # (n_object, T, 3)
    
    for seq_start in range(sequence_length - (action_per_frames + 1)):
        seq_end = seq_start + action_per_frames
        object_start = object_sampled_particles[:, seq_start, :] # (n_object, 3)
        object_end = object_sampled_particles[:, seq_end, :] # (n_object, 3)

        object_start_list.append(object_start)
        object_end_list.append(object_end)

object_start_list = torch.stack(object_start_list, dim=0) # (n_object, n_seq, 3)
object_end_list = torch.stack(object_end_list, dim=0) # (n_object, n_seq, 3)

# chamfer loss
chamfer_loss_value = chamfer_loss(object_start_list, object_end_list)
print("Chamfer Loss: ", chamfer_loss_value.item())
# emd loss
emd_loss_value = emd_loss(object_start_list, object_end_list)
print("EMD Loss: ", emd_loss_value.item())
# mse loss
mse_loss_value = mse_loss(object_start_list, object_end_list)
print("MSE Loss: ", mse_loss_value.item())
# rmse loss
rmse_loss_value = rmse_loss(object_start_list, object_end_list)
print("RMSE Loss: ", rmse_loss_value.item())
