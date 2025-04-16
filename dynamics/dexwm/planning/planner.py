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
from dexwm.utils.geometry import aligned_points_to_transformation_matrix
from dexwm.utils.pcld_wrapper import PcldWrapper
from dexwm.utils.utils_3d import rot_euler2axangle

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

        self.robot_type = config["robot_type"]
        self.robot_config = config["robots"][self.robot_type]
        # self.init_std = robot_config["init_std"]
        # if len(self.init_std.shape) == 0:  # a single number
        #     self.init_std = (
        #         self.init_std[None, None]
        #         .repeat(self.horizon, axis=0)
        #         .repeat(self.a_dim, axis=1)
        #     )
        
        # else:
        #     raise NotImplementedError(f"Unknow std shape {self.init_std.shape}")

        """initialize point cloud wrapper"""
        # self.init_action = np.array(self.robot_config["init_action"]).astype(np.float32)
        # self.init_action[3:6] = np.deg2rad(self.init_action[3:6])

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
        # print(f"max prob in the softmax prob list: {softmax_prob.max()}")
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
        init_mean=None,
        visualize_k=False,
        return_best=False,
        num_skills_in_sequence=2  # Parameter to define sequence length
    ):
        start_start_time = time.time()
        os.makedirs(log_dir, exist_ok=True)

        # Get initial and target object poses from dataset
        init_obj_pos = observation_batch.obj_init_pcd  # (n_points, 3)
        init_obj_pos = torch.from_numpy(init_obj_pos[0]).to(self.device).to(torch.float32)

        target_obj_pos = observation_batch.obj_final_pcd  # (n_points, 3)
        target_obj_pos = torch.from_numpy(target_obj_pos[0]).to(self.device).to(torch.float32)

        # target_obj_pos = np.load("/home/coolbot/data/target_shape/K.npy")
        # target_obj_pos_center = np.mean(target_obj_pos, axis=0)
        # target_obj_pos = target_obj_pos - target_obj_pos_center
        # target_obj_pos = target_obj_pos * 1.4
        # target_obj_pos = torch.from_numpy(target_obj_pos).to(self.device).to(torch.float32)

        if self.debug:
            gt_init_hand_pos = observation_batch.hand_init_pcd  # (n_hand_points, 3)
            gt_target_hand_pos = observation_batch.hand_final_pcd  # (n_hand_points, 3)

        """Define the Z-axis rotation angles to sample"""
        num_z_rotations = self.config["num_z_rotations"]
        z_rotations = np.linspace(0, 2 * np.pi, num_z_rotations, endpoint=False)

        # Track best results for all skill sequences
        all_sequence_rewards = []
        all_sequence_actions = []
        all_sequence_predictions = []
        all_sequence_metadata = []

        SKILLS = self.config["SKILLS"]

        print(f"Starting hierarchical optimization for sequences of {num_skills_in_sequence} skills...")
        
        # Define a recursive function to explore skill sequences
        def optimize_skill_sequence(sequence_idx=0, current_obj_pos=init_obj_pos, 
                                   sequence_metadata=[], sequence_actions=[], sequence_predictions=[]):
            if sequence_idx >= num_skills_in_sequence:
                # We've completed a full sequence, evaluate and return the result
                sequence_reward = sum([pred.get("reward", float('inf')) for pred in sequence_predictions])
                return sequence_reward, sequence_actions, sequence_predictions, sequence_metadata
            
            # Tracking best results for this position in the sequence
            best_sequence_reward = float('inf')
            best_sequence_actions = None
            best_sequence_predictions = None
            best_sequence_metadata = None
            
            # Loop through skills for this position in the sequence
            for skill_idx, skill in enumerate(SKILLS):
                print(f"\n===== Sequence position {sequence_idx+1}/{num_skills_in_sequence}, "
                      f"Skill {skill_idx+1}/{len(SKILLS)}: {skill} =====")
                
                # Get skill-specific parameters
                init_std = np.array(self.robot_config["init_std"][skill]).astype(np.float32)
                init_std[3:6] = np.deg2rad(init_std[3:6])
                init_std[3:6] = rot_euler2axangle(init_std[3:6], axes="sxyz")
                
                # Store original init_std for this skill
                self.init_std = init_std
                self.init_std = np.expand_dims(self.init_std, 0).repeat(self.horizon, axis=0)
                
                init_action = np.array(self.robot_config["init_action"][skill]).astype(np.float32)
                init_action[3:6] = np.deg2rad(init_action[3:6])
                
                # Track best results for this skill across Z-axis rotations
                skill_best_reward = float('inf')
                skill_best_action = None
                skill_best_prediction = None
                skill_best_z_rotation = None
                
                # Loop through Z-axis rotations
                for z_idx, z_rotation in enumerate(z_rotations):
                    print(f"------ Z-rotation {z_idx+1}/{num_z_rotations}: {z_rotation:.2f}rad ({np.degrees(z_rotation):.1f}°) ------")
                    
                    # Apply Z-axis rotation to initial action
                    rotated_init_action = init_action.copy()
                    rotated_init_action[:6] = rotate_around_world_z_np(pose=rotated_init_action[:6], theta=z_rotation)
                    rotated_init_action[3:6] = rot_euler2axangle(rotated_init_action[3:6], axes="sxyz")
                    rotated_init_action = torch.from_numpy(rotated_init_action).to(self.device).to(torch.float32)
                    
                    # Setup point cloud wrapper with the rotated initial pose
                    self.point_cloud_wrapper.set_init_params(action_vec=rotated_init_action)
                    self.point_cloud_wrapper.reset()
                    init_hand_pos = self.point_cloud_wrapper.forward()[0]  # (n_hands, 3)
                    
                    # Initial configuration for this skill+rotation combo
                    # If we're not at the first skill in sequence, use the current object position
                    if sequence_idx > 0:
                        # Use the object state from previous skill
                        # Note: current_obj_pos is already a tensor from previous step
                        init_pos = torch.cat((current_obj_pos, init_hand_pos), dim=-2)
                    else:
                        # First skill in sequence, use the initial object position
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

                    # Initialize MPPI parameters for this skill+rotation
                    mu = np.zeros((self.horizon, self.a_dim))
                    std = self.init_std

                    best_action = None
                    best_action_prediction = None
                    best_reward = float('inf')
                    
                    # Run MPPI iterations for current skill and Z-axis rotation
                    for iter in range(self.num_iters):
                        # Sample actions around current mean
                        latent_action_samples = self.sampler.sample_actions(
                            self.num_samples, mu, std
                        )  # (B, T, n_actions)

                        self.point_cloud_wrapper.reset()
                        
                        # Convert sampled actions to point cloud space
                        pcld_action_samples, sample_states = self.point_cloud_wrapper.convert(
                            torch.FloatTensor(latent_action_samples).to(self.device)
                        )

                        if self.debug:
                            # Add ground truth action to samples for debugging
                            gt_pcld_action_sample = gt_target_hand_pos - gt_init_hand_pos
                            gt_pcld_action_sample = gt_pcld_action_sample[None, None]
                            pcld_action_samples = torch.cat([pcld_action_samples, gt_pcld_action_sample], dim=0)

                        B, T, _, _ = pcld_action_samples.shape

                        # Construct full action samples including object actions (zeros)
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

                        # Run forward prediction with the model
                        pred_start_time = time.time()
                        with torch.no_grad():
                            # Split samples into batches to avoid GPU memory issues
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
                            
                            # Combine batch predictions
                            predictions = {
                                'object_obs': torch.cat(obj_obs_list, dim=0),
                                'inhand': torch.cat(inhand_list, dim=0)
                            }

                        prediction_time = time.time() - pred_start_time
                        self._model_prediction_times.append(prediction_time)

                        predict_obj_pose = predictions["object_obs"] # (n_samples, T, n_obj, 3)

                        # Evaluate rewards using the objective function
                        rewards, reward_lists = self.obj_fn.forward(
                            predictions,
                            # target_obj_pos if sequence_idx == num_skills_in_sequence - 1 else None,  # Only use target for last skill
                            target_obj_pos,
                            last_state_only=True,
                            hand_mask=is_hand_masks,
                            obj_pose=predict_obj_pose,
                            action=torch.from_numpy(latent_action_samples).to(self.device),
                        )  # shape [num_samples, 1, 1]

                        # Debug visualization
                        if self.debug:
                            debug_dir = os.path.join(log_dir, f"seq{sequence_idx+1}_skill_{skill}_z_rot_{z_idx}_debug")
                            os.makedirs(debug_dir, exist_ok=True)
                            wis3d = Wis3D(
                                out_folder=debug_dir,
                                sequence_name=f"iter_{iter}",
                                xyz_pattern=("x", "-y", "-z"),
                            )
                            wis3d.set_scene_id(0)
                            for i in range(min(B, 5)):  # Limit to 5 samples to avoid too many visualizations
                                wis3d.add_point_cloud(current_obj_pos, name="init_obj_pos")
                                wis3d.add_point_cloud(target_obj_pos, name="target_obj_pos")
                                wis3d.add_point_cloud(predictions["object_obs"][i, 0], name="obj_pred")
                                wis3d.add_point_cloud(init_hand_pos, name="init_hand_pos")
                                wis3d.add_point_cloud(predictions["inhand"][i, 0], name="hand_pred")
                                wis3d.increase_scene_id()

                        rewards = rewards.cpu().numpy()[:, None, None]
                        top_k = max(visualize_k, 1)

                        # Find best samples based on rewards
                        best_prediction_inds = np.argsort(rewards.flatten())[:: self.num_samples // top_k]
                        print(f"Seq{sequence_idx+1}, Skill: {skill}, Z-rot {z_idx}, iter {iter} - Best reward: {rewards[best_prediction_inds[0], 0, 0]:.4f}")

                        # Evaluate rewards at the last state (final outcome)
                        last_state_rewards = (
                            # self.obj_fn.eval(predictions, target_obj_pos if sequence_idx == num_skills_in_sequence - 1 else None, last_state_only=True)
                            self.obj_fn.eval(predictions, target_obj_pos, last_state_only=True)
                            .cpu()
                            .numpy()[:, None, None]
                        )  # shape [num_samples, 1, 1]
                        best_last_state_indices = np.argsort(last_state_rewards.flatten())[:top_k]
                        best_last_state_rewards = [last_state_rewards[i] for i in best_last_state_indices]
                        print(f"Seq{sequence_idx+1}, Skill: {skill}, Z-rot {z_idx}, iter {iter} - Last state rewards: {best_last_state_rewards[0][0][0]:.4f}")

                        # Visualization for planning results
                        if (iter % self.log_every == 0 and visualize_k > 0):
                            vis_dir = os.path.join(log_dir, f"seq{sequence_idx+1}_skill_{skill}_z_rot_{z_idx}")
                            os.makedirs(vis_dir, exist_ok=True)
                            wis3d = Wis3D(
                                out_folder=vis_dir,
                                sequence_name=f"iter_{iter}",
                                xyz_pattern=("x", "-y", "-z"),
                            )

                            wis3d.add_point_cloud(current_obj_pos, colors=torch.tensor([[255, 0, 0]]).repeat(self.config["particles_per_obj"], 1), name="obj_pcld")
                            wis3d.add_point_cloud(init_hand_pos, colors=torch.tensor([[255, 0, 0]]).repeat(self.config["particles_per_hand"], 1), name="hand_pcld")
                            wis3d.add_point_cloud(target_obj_pos, colors=torch.tensor([[0, 0, 255]]).repeat(self.config["particles_per_obj"], 1), name="target_obj_pcld")

                            for best_inds in best_prediction_inds[:1]:  # Just visualize the best sample
                                wis3d.set_scene_id(0)
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

                        # Update distribution parameters for next iteration
                        mu, std = self.update_dist(latent_action_samples, -last_state_rewards)
                        
                        # Save the best action for this skill+rotation
                        current_best_reward = best_last_state_rewards[0][0][0]
                        if current_best_reward < best_reward:
                            best_reward = current_best_reward
                            best_action = latent_action_samples[best_last_state_indices[0]]
                            best_action_prediction = {
                                "hand_state": sample_states[best_last_state_indices[0]],  # (T+1, 12)
                                "object_obs": predictions["object_obs"][best_last_state_indices[0]],  # (T, n_obj, 3)
                                "inhand": predictions["inhand"][best_last_state_indices[0]],  # (T, n_hand, 3)
                                "z_rotation": z_rotation,
                                "skill": skill,
                                "reward": best_reward
                            }
                
                    # After all MPPI iterations, save results for this skill+rotation
                    print(f"Best reward for Seq{sequence_idx+1}, Skill: {skill}, Z-rot: {best_action_prediction["z_rotation"]:.2f}rad: {best_action_prediction["reward"]:.4f}")
                
                    # Update best for this skill if this rotation is better
                    if best_reward < skill_best_reward:
                        skill_best_reward = best_reward
                        skill_best_action = best_action
                        skill_best_prediction = best_action_prediction
                        skill_best_z_rotation = z_rotation
                
                # After MPPI optimization for this skill+rotation combo
                # Get the final object state to pass to the next skill in sequence
                final_obj_state = skill_best_prediction["object_obs"][-1]  # Last state
                
                # Recursively optimize the rest of the sequence
                next_sequence_metadata = sequence_metadata + [{"skill": skill, "z_rotation": skill_best_z_rotation}]
                next_sequence_actions = sequence_actions + [skill_best_action]
                next_sequence_predictions = sequence_predictions + [skill_best_prediction]
                
                seq_reward, seq_actions, seq_predictions, seq_metadata = optimize_skill_sequence(
                    sequence_idx + 1, 
                    final_obj_state,
                    next_sequence_metadata,
                    next_sequence_actions, 
                    next_sequence_predictions
                )
                
                # Update best sequence if this is better
                if seq_reward < best_sequence_reward:
                    best_sequence_reward = seq_reward
                    best_sequence_actions = seq_actions
                    best_sequence_predictions = seq_predictions
                    best_sequence_metadata = seq_metadata
                    
                    # Save best sequence so far for this position
                    seq_dir = os.path.join(log_dir, f"best_so_far_seq{sequence_idx+1}")
                    os.makedirs(seq_dir, exist_ok=True)
                    for i, (action, prediction, metadata) in enumerate(zip(seq_actions, seq_predictions, seq_metadata)):
                        sk = metadata["skill"]
                        z_rot = metadata["z_rotation"]
                        os.makedirs(f"{seq_dir}/step_{i+1}_skill_{sk}", exist_ok=True)
                        torch.save({
                            "action": action,
                            "prediction": prediction,
                            "metadata": metadata
                        }, f"{seq_dir}/step_{i+1}_skill_{sk}/data.pt")
                    
                    print(f"New best sequence found at position {sequence_idx+1}:")
                    for i, meta in enumerate(seq_metadata):
                        print(f"  Step {i+1}: Skill {meta['skill']}, Z-rot: {meta['z_rotation']:.2f}rad")
                    print(f"  Total reward: {seq_reward:.4f}")
            
            # After trying all Z-rotations for this skill
            print(f"Best for Skill {skill} at position {sequence_idx+1}: Z-rot {skill_best_z_rotation:.2f}rad, Reward: {skill_best_reward:.4f}")
            
            # Save best sequence for this skill at this position
            all_sequence_rewards.append(best_sequence_reward)
            all_sequence_actions.append(best_sequence_actions)
            all_sequence_predictions.append(best_sequence_predictions)
            all_sequence_metadata.append(best_sequence_metadata)
        
            # After trying all skills for this position
            if len(all_sequence_rewards) == 0:
                # This should not happen in normal execution
                print("Warning: No valid sequences found!")
                return (float('inf'), [], [], [])
                
            # Return the best sequence found at this position
            best_idx = np.argmin(all_sequence_rewards)
            return (all_sequence_rewards[best_idx], all_sequence_actions[best_idx],
                    all_sequence_predictions[best_idx], all_sequence_metadata[best_idx])
    
        # Start the recursive optimization
        best_reward, best_actions, best_predictions, best_metadata = optimize_skill_sequence()
        
        # Save the best sequence results
        os.makedirs(f"{log_dir}/best_sequence", exist_ok=True)
        for i, (action, prediction, metadata) in enumerate(zip(best_actions, best_predictions, best_metadata)):
            skill = metadata["skill"]
            z_rotation = metadata["z_rotation"]
            os.makedirs(f"{log_dir}/best_sequence/step_{i+1}_skill_{skill}", exist_ok=True)
            torch.save({
                "action": action,
                "prediction": prediction,
                "metadata": metadata
            }, f"{log_dir}/best_sequence/step_{i+1}_skill_{skill}/data.pt")
        
        print(f"\n=== Optimization complete for {num_skills_in_sequence}-skill sequence ===")
        print(f"Best sequence:")
        for i, metadata in enumerate(best_metadata):
            print(f"  Step {i+1}: Skill {metadata['skill']}, "
                f"Z-rotation: {metadata['z_rotation']:.2f}rad ({np.degrees(metadata['z_rotation']):.1f}°)")
        print(f"Total reward: {best_reward:.4f}")
        
        end_end_time = time.time()
        print(f"Total planning time: {end_end_time - start_start_time:.2f} seconds")

        if return_best:
            return best_metadata, best_actions, best_predictions
        else:
            return best_metadata
            

def rotate_around_world_z_np(pose: np.ndarray, theta: float) -> np.ndarray:
    """
    Rotate a 3D pose [x, y, z, roll, pitch, yaw] around the WORLD's Z-axis by angle theta.
    
    Parameters
    ----------
    pose : np.ndarray, shape (6,)
        Original pose [x, y, z, roll, pitch, yaw].
    theta : float
        Rotation angle around the WORLD's Z-axis (in radians).

    Returns
    -------
    rotated_pose : np.ndarray, shape (6,)
        The rotated pose [x_new, y_new, z_new, roll_new, pitch_new, yaw_new].
    """
    # Unpack the incoming pose
    x, y, z, roll, pitch, yaw = pose
    
    # Compute new position
    x_new = x * np.cos(theta) - y * np.sin(theta)
    y_new = x * np.sin(theta) + y * np.cos(theta)
    z_new = z  # unchanged by rotation around Z-axis
    
    # Orientation changes: only yaw is incremented by theta
    roll_new = roll
    pitch_new = pitch
    yaw_new = yaw + theta
    
    # Return as a NumPy array
    return np.array([x_new, y_new, z_new, roll_new, pitch_new, yaw_new])