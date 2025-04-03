import numpy as np
import torch
import time
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
# import threading
# from queue import Queue
# from wis3d import Wis3D
from dexwm.utils.wis3d_new import Wis3D
from torch_geometric.data import Batch
from dexwm.utils.geometry import aligned_points_to_transformation_matrix

# from perception.utils_cv import find_point_on_line
import trimesh
from dexwm.utils.pcld_wrapper import PcldWrapper


def convert_state_dict_to_state_list(predictions_dict):
    """
    Convert a dictionary of predictions to a list of predictions, where each prediction is a dictionary
    :param predictions_dict: A dictionary where each key is a string and each value is a numpy array
        of shape [batch_size, ...] or [time, ...]
    :return: A list of dictionaries, where each dictionary has the same keys as the input dictionary and each value
        is a numpy array of shape [...]
    """
    # assert values have the same length
    length = None
    for key, values in predictions_dict.items():
        if length is None:
            length = values.shape[0]
        else:
            assert (
                length == values.shape[0]
            ), f"length of values for key {key} is not the same as other keys {length}"

    state_list = []
    for t in range(length):
        state_dict = {
            k: v[t] for k, v in predictions_dict.items()
        }  # the first dim is batch
        state_list.append(state_dict)

    return state_list


class MPPIOptimizer:
    def __init__(
        self,
        sampler,
        point_cloud_wrapper,
        model,
        objective,
        a_dim,
        horizon,
        num_samples,
        gamma,
        config,
        logger,
        num_iters=3,
        init_std=0.5,
        log_every=1,
        device="cuda",
    ):
        self.obj_fn = objective
        self.sampler = sampler
        self.point_cloud_wrapper: PcldWrapper = point_cloud_wrapper
        self.model = model
        self.horizon = horizon
        self.a_dim = a_dim
        self.num_samples = num_samples
        self.gamma = gamma
        self.num_iters = num_iters

        self.update_std = config["update_std"]
        self.config = config
        self.logger = logger

        self.device = device

        self.debug = config["debug"]

        action_limit = np.array(self.config["action_limit"])
        action_limit[3:6] = np.deg2rad(action_limit[3:6])

        self.init_std = np.array(init_std) * action_limit

        # if len(self.init_std.shape) == 0:  # a single number
        #     self.init_std = (
        #         self.init_std[None, None]
        #         .repeat(self.horizon, axis=0)
        #         .repeat(self.a_dim, axis=1)
        #     )
        assert len(self.init_std.shape) == 1 and self.init_std.shape[0] == self.a_dim
        self.init_std = np.expand_dims(self.init_std, 0).repeat(self.horizon, axis=0)
        # else:
        #     raise NotImplementedError(f"Unknow std shape {self.init_std.shape}")

        self.log_every = log_every
        self._model_prediction_times = list()

    def update_dist(self, samples, scores):
        # actions: array with shape [num_samples, time, action_dim]
        # scores: array with shape [num_samples]
        scaled_rews = self.gamma * (scores - np.max(scores))  # all positive

        # exponentiated scores
        exp_rews = np.exp(scaled_rews)

        # weigh samples by exponentiated scores to bias sampling for future iterations
        softmax_prob = exp_rews / (np.sum(exp_rews, axis=0) + 1e-10)
        mu = np.sum(softmax_prob * samples, axis=0)
        print(f"max prob in the softmax prob list: {softmax_prob.max()}")
        # mu = np.sum(exp_rews * samples, axis=0) / (np.sum(exp_rews, axis=0) + 1e-10)
        # pdb.set_trace()

        # prior knowledge: the end effector should not move along z axis
        # mu[:, -1] = 0
        if self.update_std:
            std = np.sqrt(np.sum(softmax_prob * (samples - mu) ** 2, axis=0))
        else:
            std = self.init_std

        return mu, std  # self.init_std

    def plan(
        self,
        t,
        log_dir,
        observation_batch,
        # action_history,
        # goal,
        init_mean=None,
        visualize_k=False,
        return_best=False,
    ):
        start_start_time = time.time()

        os.makedirs(log_dir, exist_ok=True)

        # predict dataset: obtain initial and final object pose from dataset
        init_obj_pos = observation_batch.obj_init_pcd # (n_points, 3)
        target_obj_pos = observation_batch.obj_final_pcd # (n_points, 3)

        init_obj_pos = torch.from_numpy(init_obj_pos[0]).to(self.device).to(torch.float32)
        target_obj_pos = torch.from_numpy(target_obj_pos[0]).to(self.device).to(torch.float32)

        if self.debug:
            gt_init_hand_pos = observation_batch.hand_init_pcd # (n_hand_points, 3)
            gt_target_hand_pos = observation_batch.hand_final_pcd # (n_hand_points, 3)
        
        """initialize point cloud wrapper"""
        initial_action = torch.FloatTensor(self.config["init_action"]).to(self.device)
        # convert the degree to radian
        initial_action[3:6] = torch.deg2rad(initial_action[3:6])
        self.point_cloud_wrapper.set_init_params(action_vec=initial_action)

        """sample object point cloud and obtain the initial graph"""
        init_hand_pos = self.point_cloud_wrapper.forward()[0]  # (n_hands, 3)

        # import pdb; pdb.set_trace()
        init_pos = torch.cat((init_obj_pos, init_hand_pos), dim=-2)

        is_hand_masks = self.point_cloud_wrapper.is_hand_part_masks

        particle_type = torch.cat(
            [
                torch.full((n, 1), i, dtype=torch.int)
                for i, n in enumerate(
                    [
                        self.config["particles_per_obj"],
                        self.config["particles_per_hand"],
                    ]
                )
            ],
            dim=0,
        ).to(self.device)

        # if self.config["mask_fingertips"]:
        #     is_finger_masks = self.point_cloud_wrapper.is_finger_masks
        #     particle_type[self.config["particles_per_obj"] :][is_finger_masks] = 2

        mu = np.zeros((self.horizon, self.a_dim))
        std = self.init_std

        best_action, best_action_prediction = None, None
        self._model_prediction_times = []
        for iter in range(self.num_iters):
            # latent_action_samples: relative actions
            latent_action_samples = self.sampler.sample_actions(
                self.num_samples, mu, std
            )  # (B, T, n_actions)

            # action_samples = np.clip(latent_action_samples, -0.006, 0.006)

            self.point_cloud_wrapper.reset()
            
            pcld_action_samples, sample_states = self.point_cloud_wrapper.convert(
                torch.FloatTensor(latent_action_samples).to(self.device)
            )

            if self.debug:
                # add the ground truth action to the samples
                gt_pcld_action_sample = gt_target_hand_pos - gt_init_hand_pos
                gt_pcld_action_sample = gt_pcld_action_sample[None, None]
                pcld_action_samples = torch.cat([pcld_action_samples, gt_pcld_action_sample], dim=0)

            B, T, _, _ = pcld_action_samples.shape

            action_samples = torch.cat(
                [
                    torch.zeros(
                        (B, T, self.config["particles_per_obj"], 3),
                        device=pcld_action_samples.device,
                    ),
                    pcld_action_samples,
                ],
                dim=2,
            )  # (B, T, n_particles, 3)

            pred_start_time = time.time()
            with torch.no_grad():
                # Split action_samples into batches along the batch dimension (dim=0)
                n_sample_batch = self.config["sample_batch_size"]
                action_batches = torch.split(action_samples, n_sample_batch, dim=0)
                
                # Initialize lists to collect predictions
                obj_obs_list = []
                inhand_list = []
                
                # Process each batch sequentially
                for action_batch in action_batches:
                    batch_pred = self.model.predict_step(
                        init_pos,          # (n_particles, 3)
                        action_batch,      # (current_batch_size, T, n_particles, 3)
                        particle_type      # (n_particles,)
                    )
                    obj_obs_list.append(batch_pred['object_obs'])
                    inhand_list.append(batch_pred['inhand'])
                
                # Concatenate all batch predictions
                predictions = {
                    'object_obs': torch.cat(obj_obs_list, dim=0),
                    'inhand': torch.cat(inhand_list, dim=0)
                }

            prediction_time = time.time() - pred_start_time
            self._model_prediction_times.append(prediction_time)

            predict_obj_pose = predictions["object_obs"] # (n_samples, T, n_obj, 3)

            # NOTE: actually cost function and we should take the least cost
            rewards, reward_lists = self.obj_fn.forward(
                predictions,
                target_obj_pos,
                last_state_only=True,
                # finger_mask=is_finger_masks,
                # inner_mask=is_inner_masks,
                hand_mask=is_hand_masks,
                # hand_joints=sample_states.get("joints", None),
                obj_pose=predict_obj_pose,
                # mesh_file_path=observation_batch.obj_mesh_path[0],
                action=torch.from_numpy(latent_action_samples).to(self.device),
            )  # shape [num_samples, 1, 1]

            # import pdb; pdb.set_trace()

            if self.debug:
                wis3d = Wis3D(
                    out_folder=f"{log_dir}/wis3d_debug",
                    sequence_name="planner_prediction",
                    xyz_pattern=("x", "-y", "-z"),
                )
                wis3d.set_scene_id(0)
                for i in range(B):
                    wis3d.add_point_cloud(init_obj_pos, name="init_obj_pos")
                    wis3d.add_point_cloud(target_obj_pos, name="target_obj_pos")
                    wis3d.add_point_cloud(predictions["object_obs"][i, 0], name="obj_pred")
                    if i == B - 1:
                        wis3d.add_point_cloud(gt_init_hand_pos, name="gt_hand_pos")
                        wis3d.add_point_cloud(gt_target_hand_pos, name="gt_target_hand_pos")
                    else:
                        wis3d.add_point_cloud(init_hand_pos, name="init_hand_pos")
                        wis3d.add_point_cloud(predictions["inhand"][i, 0], name="hand_pred")
                    wis3d.increase_scene_id()

            rewards = rewards.cpu().numpy()[:, None, None]

            top_k = max(visualize_k, 1)

            best_prediction_inds = np.argsort(rewards.flatten())[
                :: self.num_samples // top_k
            ]
            print(f"prediction indices: {best_prediction_inds}")

            best_rewards = [rewards[i] for i in best_prediction_inds]
            # best_actions = [new_action_samples[x] for x in best_prediction_inds]
            # print("best rewards:", best_rewards)
            # print('best actions:', best_actions)

            last_state_rewards = (
                self.obj_fn.eval(predictions, target_obj_pos, last_state_only=True)
                .cpu()
                .numpy()[:, None, None]
            )  # shape [num_samples, 1, 1]
            best_last_state_indices = np.argsort(last_state_rewards.flatten())[:top_k]
            best_last_state_rewards = [
                last_state_rewards[i] for i in best_last_state_indices
            ]
            print(f"last state rewards: {best_last_state_rewards}")

            self.logger.add_scalar("train/mean_rewards", rewards.mean(), step=iter)
            self.logger.add_scalar(
                "train/best_rewards", rewards[best_prediction_inds[0], 0, 0], step=iter
            )
            for idx in range(len(reward_lists)):
                self.logger.add_scalar(
                    f"train/mean_{self.obj_fn.fun_names[idx]}",
                    reward_lists[idx].mean(),
                    step=iter,
                )
                self.logger.add_scalar(
                    f"train/best_{self.obj_fn.fun_names[idx]}",
                    reward_lists[idx][best_prediction_inds[0]],
                    step=iter,
                )
            self.logger.add_scalar(
                "eval/last_state_rewards", last_state_rewards.mean(), step=iter
            )
            self.logger.add_scalar(
                "eval/best_last_state_rewards", best_last_state_rewards[0], step=iter
            )

            # print(f'Computing rewards takes {time.time() - start_time}')

            if iter == self.num_iters - 1:
                end_end_time = time.time()

            if (iter % self.log_every == 0 and visualize_k > 0):
                wis3d = Wis3D(
                    out_folder=f"{log_dir}/wis3d",
                    sequence_name=f"mppi_step_{t}_iter{iter:04d}_plan",
                    xyz_pattern=("x", "-y", "-z"),
                )

                wis3d.add_point_cloud(init_obj_pos, colors=torch.tensor([[255, 0, 0]]).repeat(self.config["particles_per_obj"], 1), name="obj_pcld")
                wis3d.add_point_cloud(init_hand_pos, colors=torch.tensor([[255, 0, 0]]).repeat(self.config["particles_per_hand"], 1), name="hand_pcld")
                wis3d.add_point_cloud(target_obj_pos, colors=torch.tensor([[0, 0, 255]]).repeat(self.config["particles_per_obj"], 1), name="target_obj_pcld")

                for best_inds in best_prediction_inds:  # + [5]:
                    wis3d.set_scene_id(0)
                    # object point cloud prediction in best sampling trajectory
                    obj_pcld = predictions["object_obs"][best_inds]  # (T, n_obj, 3) 
                    self.point_cloud_wrapper.visualize(
                        sample_states[best_inds, 0:1],
                        wis3d,
                        name=f"hand_{best_inds}",
                    )

                    hand_pcld = predictions["inhand"][best_inds]  # (T, n_hand, 3)
                    wis3d.increase_scene_id()

                    for vis_i in range(T):
                        wis3d.add_point_cloud(hand_pcld[vis_i], colors=torch.tensor([[0, 255, 0]]).repeat(self.config["particles_per_hand"], 1),  name=f"hand_pcld")
                        self.point_cloud_wrapper.visualize(
                            sample_states[best_inds, vis_i + 1 : vis_i + 2],
                            wis3d,
                            name=f"hand_{best_inds}",
                        )
                        wis3d.add_point_cloud(obj_pcld[vis_i], colors=torch.tensor([[0, 255, 0]]).repeat(self.config["particles_per_obj"], 1), name=f"obj_pcld")
                        wis3d.add_point_cloud(target_obj_pos, colors=torch.tensor([[0, 0, 255]]).repeat(self.config["particles_per_obj"], 1), name="target_obj_pcld")

                        wis3d.increase_scene_id()

            if self.debug and best_prediction_inds.__len__() > 1:
                for reward in reward_lists:
                    # print(best_prediction_inds[0], best)
                    print(
                        f"traj {best_prediction_inds[0]}",
                        reward[best_prediction_inds[0]].item(),
                        f"traj {best_prediction_inds[1]}",
                        reward[best_prediction_inds[1]].item(),
                    )
            else:
                for reward in reward_lists:
                    print(f"best reward: {reward[best_prediction_inds[0]]}")

            start_time = time.time()
            mu, std = self.update_dist(latent_action_samples, -last_state_rewards)
            # print(f"mu shape = {mu.shape}, means over all steps = {mu.mean(0)}")
            # print(f'Updating distribution takes {time.time() - start_time}')
            # print(f'total time for the iteration: {time.time() - start_start_time}\n')

            # best_action = latent_action_samples[best_prediction_inds[0]]
            best_action_prediction = {
                # k: v[best_prediction_inds[0]] for k, v in predictions.items()
                # "obj_pose": obj_pose,  # (T, 4, 4)
                # "hand_state": sample_states["states"][best_prediction_inds[0]],  # (T, 12)
                "hand_state": sample_states[best_last_state_indices[0]],  # (T, 12)
            }

            # check shape of the best action
            for key, value in best_action_prediction.items():
                assert (value.shape[0] == self.horizon + 1), f"key {key} has wrong shape {value.shape}"
            # best_action_prediction = convert_state_dict_to_state_list(
            #     best_action_prediction
            # )
            # best_action_prediction["obj_mesh_path"] = observation_batch.obj_mesh_path[0]
            # best_action_prediction["mano_shape"] = observation_batch.mano_shape

            os.makedirs(f"{log_dir}/best_plans", exist_ok=True)
            torch.save(best_action_prediction, f"{log_dir}/best_plans/best_action_{iter}.pt")

        print(f"Total time for planning: {end_end_time - start_start_time}")

        if return_best:
            return mu, best_action, best_action_prediction
        else:
            return mu