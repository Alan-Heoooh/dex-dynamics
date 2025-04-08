import pytorch_lightning as pl
import torch_geometric as pyg
import os
import torch
from dexwm.utils.sample import furthest_point_sampling
import numpy as np
from .dataset import connect_edges
from collections import OrderedDict

from dexwm.utils.pcld_wrapper import HandPcldWrapper
from dexwm.utils.geometry import generate_random_rotation


class DeformableDataset(pyg.data.InMemoryDataset):
    def __init__(self, config, split="train", transform=None):
        self.config = config
        self.split = split
        self.root = self.config["data_dir"]
        # os.path.join(DATA_DIR, self.config["data_dir_prefix"])
        assert os.path.exists(self.root), f"Dataset root {self.root} does not exist"

        super().__init__(
            self.root,
            transform=transform,
        )

        self.data_augment = self.config["data_augment"]

        if config["rebuild_dataset"]:
            print(f"DeformableDataset: Rebuild dataset. Root directory: {self.root}")
            self.process()

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f"deformable_{self.split}"]

    def process(self):
        hand_pcld_sampler = HandPcldWrapper(
            particles_per_hand=self.config["particles_per_hand"],
            num_samples=1,
            device="cpu",
        )

        train_val_split = lambda sid: sid % 10 != 0

        if self.split == "train" or self.split == "test":
            dataset_dir = self.root
            scene_list = sorted(os.listdir(dataset_dir))

            scene_list = [scene for scene in scene_list if scene.endswith(".npy")]
            
            if self.split == "train":
                scene_list = [scene for scene in scene_list if train_val_split(int(scene.split("_")[-1].split("-")[0]))]

            elif self.split == "test":
                scene_list = [scene for scene in scene_list if not train_val_split(int(scene.split("_")[-1].split("-")[0]))]
                # print(f"Test scene list: {scene_list}")

            # positive and negative data
            pos_scene_list = [scene for scene in scene_list if "neg" not in scene]
            neg_scene_list = [scene for scene in scene_list if "neg" in scene]
            if self.config['negative_data']:
                negative_ratio = self.config["negative_data_ratio"]
                neg_scene_list = np.random.choice(neg_scene_list, int(len(pos_scene_list) * negative_ratio), replace=False)
                scene_list = np.concatenate([pos_scene_list, neg_scene_list])
            else:
                scene_list = pos_scene_list

            
            data_list = []
            for scene in scene_list:
                hand_obj_ret = np.load(os.path.join(dataset_dir, scene), allow_pickle=True).item()
                # sequence_length = len(hand_obj_ret_list)

                # n_points_hand = self.config["particles_per_hand"]
                # n_points_object = self.config["particles_per_obj"]
                
                # load hand mesh, assume one hand
                hand_verts_seq = [torch.FloatTensor(hand_obj_ret["hand_init_pcd"]), torch.FloatTensor(hand_obj_ret["hand_final_pcd"])] # (T, 778, 3)
                hand_verts_seq = torch.stack(hand_verts_seq, dim=0) # (T, 778, 3)
                hand_verts = hand_verts_seq.permute(1, 0, 2)  # (778, T, 3)

                if not self.config["hand_sampling_augment"] or self.split == "test":
                    hand_pcld_sampler.sample()
                    hand_sample_indices = hand_pcld_sampler.hand_sample_indices
                    hand_pcd = hand_verts[hand_sample_indices]  # (particles_per_hand, T, 3)
                else:
                    hand_pcd = hand_verts

                if not self.config["object_sampling_augment"] or self.split == "test":
                    # load object pcd
                    obj_pcd_seq = [torch.FloatTensor(hand_obj_ret["object_init_pcd"]), torch.FloatTensor(hand_obj_ret["object_final_pcd"])] # (T, n, 3)
                else:
                    obj_pcd_seq = [torch.FloatTensor(hand_obj_ret["object_init_pcd_dense"]), torch.FloatTensor(hand_obj_ret["object_final_pcd_dense"])]

                # obj_pcd_seq = torch.stack(obj_pcd_seq, dim=0) # (T, n, 3)
                # obj_pcd = obj_pcd_seq.permute(1, 0, 2)  # (n, T,  3)


                # is_finger_tips = hand_pcld_sampler.is_finger_masks
                # _, hand_sample_indices = furthest_point_sampling(
                #     hand_verts[:, 0], self.config["particles_per_hand"]
                # )

                # n_points_object = obj_pcd.size(0)
                # n_points_hand = hand_pcd.size(0)

                # # classifying the particles
                # particle_type = torch.cat(
                #     [torch.full((n, 1), i, dtype=torch.int) for i, n in enumerate([n_points_object, n_points_hand])], dim=0,
                # )

                # if self.config["mask_fingertips"]:
                #     particle_type[n_points_object:][is_finger_tips] = 2

                # total_pcd = torch.cat([obj_pcd, hand_pcd], dim=0)

                for seq_start in range(1):
                    # action = torch.cat(
                    #     (
                    #         torch.zeros_like(obj_pcd[:, seq_start]),
                    #         hand_pcd[:, seq_start + 1] - hand_pcd[:, seq_start],
                    #     ),
                    #     dim=0,
                    # )

                    # node_features = torch.cat([particle_type, action], dim=-1)  # (n_objects + n_particles_hand, 4)

                    # construct the graph based on the distances between particles

                    # if self.config["use_final_graph"]:
                    #     pos_dict = OrderedDict(
                    #         object_obs=(0, obj_pcd[:, seq_start]),  # format: (start_index, position)
                    #         hand=(n_points_object, hand_pcd[:, seq_start + 1]),
                    #     )
                    # else:
                    #     pos_dict = OrderedDict(
                    #         object_obs=(0, obj_pcd[:, seq_start]),  # format: (start_index, position)
                    #         hand=(n_points_object, hand_pcd[:, seq_start]),
                    #     )

                    # edge_index, edge_attr = connect_edges(self.config, pos_dict)

                    graph_data = pyg.data.Data(
                        # x=node_features,
                        # edge_index=edge_index,
                        # edge_attr=edge_attr,
                        # pos=total_pcd[:, seq_start : seq_start + self.config[f"{self.split}_sequence_length"] + 1], # point cloud position of object and hand, shape: (n_objects + n_particles_hand, T, 3)
                        hand_pcd=hand_pcd,
                        # obj_pcd=obj_pcd,
                        obj_pcd_seq=obj_pcd_seq,

                        # Store particle counts as attributes
                        # n_points_object=n_points_object,
                        # n_points_hand=self.config["particles_per_hand"],
                        # n_points_hand=n_points_hand,
                        # Store original positions for edge recalculation
                        # original_pos=total_pcd[:, seq_start].clone(),
                        scene_id=scene  # Optional: for debugging
                    )

                    data_list.append(graph_data)
                
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])

        elif self.split == "predict":
            dataset_dir = self.root
            scene_list = sorted(os.listdir(dataset_dir))

            scene_list = [scene for scene in scene_list if scene.endswith(".npy")]
            scene_list = [scene for scene in scene_list if not train_val_split(int(scene.split("_")[-1].split("-")[0]))]

            # print(f"Predict scene list: {scene_list}")

            data_list = []

            for scene in scene_list:
                hand_obj_ret = np.load(os.path.join(dataset_dir, scene), allow_pickle=True).item()
                # sequence_length = len(hand_obj_ret_list)

                n_points_hand = self.config["particles_per_hand"]
                # n_points_object = self.config["particles_per_obj"]

                # load object pcd
                obj_pcd_seq = [torch.FloatTensor(hand_obj_ret["object_init_pcd"]), torch.FloatTensor(hand_obj_ret["object_final_pcd"])] # (T, n, 3)
                # load hand mesh, assume one hand
                hand_verts_seq = [torch.FloatTensor(hand_obj_ret["hand_init_pcd"]), torch.FloatTensor(hand_obj_ret["hand_final_pcd"])] # (T, 778, 3)

                obj_init_pcd = hand_obj_ret["object_init_pcd"]
                obj_init_pcd_center = obj_init_pcd.mean(axis=0) # (3,)
                obj_pcd_seq = [obj_pcd - obj_init_pcd_center for obj_pcd in obj_pcd_seq]
                hand_verts_seq = [hand_verts - obj_init_pcd_center for hand_verts in hand_verts_seq]


                obj_pcd_seq = torch.stack(obj_pcd_seq, dim=0) # (T, n, 3)
                hand_verts_seq = torch.stack(hand_verts_seq, dim=0) # (T, 778, 3)

                obj_pcd = obj_pcd_seq.permute(1, 0, 2)  # (n, T,  3)
                hand_verts = hand_verts_seq.permute(1, 0, 2)  # (778, T, 3)

                # sample hand particles
                hand_sample_indices = None
                _, hand_sample_indices = furthest_point_sampling(
                    hand_verts[:, 0], self.config["particles_per_hand"]
                )
                hand_pcd = hand_verts[hand_sample_indices]  # (particles_per_hand, T, 3)

                n_points_object = obj_pcd.size(0)

                # classifying the particles
                particle_type = torch.cat(
                    [torch.full((n, 1), i, dtype=torch.int) for i, n in enumerate([n_points_object, n_points_hand])], dim=0,
                )

                total_pcd = torch.cat([obj_pcd, hand_pcd], dim=0) # (n_objects + n_particles_hand, T, 3)
                for seq_start in range(1):
                    action = torch.cat(
                        (
                            torch.zeros_like(obj_pcd[:, seq_start]),
                            hand_pcd[:, seq_start + 1] - hand_pcd[:, seq_start],
                        ),
                        dim=0,
                    )

                    node_features = torch.cat([particle_type, action], dim=-1)  # (n_objects + n_particles_hand, 4)

                    # construct the graph based on the distances between particles
                    pos_dict = OrderedDict(
                        object_obs=(0, obj_pcd[:, seq_start]),  # format: (start_index, position)
                        hand=(n_points_object, hand_pcd[:, seq_start]),
                    )

                    edge_index, edge_attr = connect_edges(self.config, pos_dict)

                    graph_data = pyg.data.Data(
                        x=node_features,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        pos=total_pcd[:, seq_start : seq_start + self.config[f"{self.split}_sequence_length"] + 1], # point cloud position of object and hand

                        obj_init_pcd = obj_pcd_seq[0].cpu().numpy(),
                        obj_final_pcd = obj_pcd_seq[1].cpu().numpy(),
                        # hand_init_pcd = hand_pcd[:, 0],
                        # hand_final_pcd = hand_pcd[:, 1],
                    )

                    data_list.append(graph_data)
                data, slices = self.collate(data_list)
                torch.save((data, slices), self.processed_paths[0])

        else:
            raise ValueError(f"Invalid split {self.split}")
        
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        if self.split == "train" :
            if self.data_augment:
                data = self.apply_augmentation(data, hand_sampling_augment=self.config["hand_sampling_augment"], object_sampling_augment=self.config["object_sampling_augment"], apply_rotation=True)
            else:
                data = self.apply_augmentation(data, hand_sampling_augment=False, object_sampling_augment=False, apply_rotation=False)
        elif self.split == "test":
            data = self.apply_augmentation(data, hand_sampling_augment=False, object_sampling_augment=False, apply_rotation=False)
        elif self.split == "predict":
            return data
        else:
            raise ValueError(f"Invalid split {self.split}")
            
        return data
    
    def apply_augmentation(self, data, hand_sampling_augment=False, object_sampling_augment=False, apply_rotation=False):
        data = data.clone()
        
        # 1. Resample hand points from full mesh
        n_points_hand = self.config["particles_per_hand"]
        n_points_object = self.config["particles_per_obj"]
        
        if hand_sampling_augment:
            # Sample new indices using FPS
            _, hand_sample_indices = furthest_point_sampling(
                data.hand_pcd[:, 0], n_points_hand
            )
            hand_pcd = data.hand_pcd[hand_sample_indices]  # (n_points_hand, T, 3)
        else:
            hand_pcd = data.hand_pcd

        # 2. Resample object points from dense pcd
        obj_pcd_seq = data.obj_pcd_seq
        if object_sampling_augment:
            # Sample new indices using FPS
            obj_pcd_list = []
            for obj_pcd in obj_pcd_seq:
                _, obj_sample_indices = furthest_point_sampling(
                    obj_pcd, n_points_object
                )
                obj_pcd = obj_pcd[obj_sample_indices]
                obj_pcd_list.append(obj_pcd)
            obj_pcd = torch.stack(obj_pcd_list, dim=0)
        else:
            obj_pcd = torch.stack(obj_pcd_seq, dim=0)
        obj_pcd = obj_pcd.permute(1, 0, 2)  # (n_points_object, T, 3)

        # 2. Rebuild positions with new hand points
        new_pos = torch.cat([obj_pcd, hand_pcd], dim=0)
        
        # 3. Apply rotation to entire point cloud
        if apply_rotation:
            rand_rot = generate_random_rotation(
                num_samples=1,
                along_z=self.config['augment_along_z_axis'],
                device=new_pos.device
            ).squeeze(0)
            
            # Rotate positions
            rotated_pos = torch.einsum('ij,ntj->nti', rand_rot, new_pos)
        else:
            rotated_pos = new_pos
        
        # 4. Recompute action vectors
        hand_action = hand_pcd[:, 1] - hand_pcd[:, 0]  # Unrotated difference
        action = torch.cat([
            torch.zeros(n_points_object, 3, device=hand_action.device),
            hand_action
        ], dim=0)
        
        # Rotate action vectors
        if apply_rotation:
            rotated_action = action @ rand_rot.T
        else:
            rotated_action = action
        
        # 5. Update node features
        particle_type = torch.cat(
                    [torch.full((n, 1), i, dtype=torch.int) for i, n in enumerate([n_points_object, n_points_hand])], dim=0,
                )
        data.x = torch.cat([particle_type, rotated_action], dim=-1)
        
        # 6. Update positions
        data.pos = rotated_pos
        
        # 7. Recompute edges
        pos_dict = OrderedDict(
            object_obs=(0, rotated_pos[:n_points_object, 0]),
            hand=(n_points_object, rotated_pos[n_points_object:, 0])
        )
        edge_index, edge_attr = connect_edges(self.config, pos_dict)
        data.edge_index = edge_index
        data.edge_attr = edge_attr

        return data
    
class DeformableDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

    def setup(self, stage):
        if stage == "fit":
            train_set = DeformableDataset(self.config, "train")
            val_set = DeformableDataset(self.config, "test")

            self.train, self.val = train_set, val_set

        elif stage == "test":
            raise NotImplementedError

        elif stage == "predict":
            self.predict = DeformableDataset(self.config, "predict")
        else:
            raise ValueError(f"Invalid stage {stage}")

    def train_dataloader(self):
        return pyg.loader.DataLoader(
            self.train,
            batch_size=self.config["train_batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=True,
        )

    def val_dataloader(self):
        return pyg.loader.DataLoader(
            self.val,
            batch_size=self.config["train_batch_size"],
            num_workers=self.config["num_workers"],
        )

    def test_dataloader(self):
        return pyg.loader.DataLoader(
            self.test,
            batch_size=self.config["test_batch_size"],
            num_workers=self.config["num_workers"],
        )

    def predict_dataloader(self):
        return pyg.loader.DataLoader(
            self.predict,
            batch_size=self.config["test_batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=True,
        )
