import pytorch_lightning as pl
import torch_geometric as pyg
import os
import torch
from .YCB_loader import DexYCBVideoDataset
from dexwm.utils.macros import EXTERNAL_DIR, ASSETS_DIR
from dexwm.utils.sample import furthest_point_sampling
from .ycb_utils import to_transformation_matrix
from manopth.manolayer import ManoLayer
import trimesh
import numpy as np
from .dataset import connect_edges
from collections import OrderedDict
import pickle
from dexwm.utils.geometry import generate_random_rotation


class DexYCBDataset(pyg.data.InMemoryDataset):
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
        self.num_augment = self.config["num_augment"]

        if config["rebuild_dataset"]:
            print(f"DexYCBDataset: Rebuild dataset. Root directory: {self.root}")
            self.process()

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f"dexycb_{self.split}"]

    def process(self):
        tips_region = np.load(f"{ASSETS_DIR}/fingers.npy")

        if self.split == "train":  # or self.split == "test":
            dataset_dir = self.config["data_dir"]
            # os.path.join(DATA_DIR, self.config["data_dir_prefix"])
            dataset = DexYCBVideoDataset(dataset_dir, "right", mode=self.split)
            mano_layer = ManoLayer(
                mano_root=f"{EXTERNAL_DIR}/mano/models",
                side="right",
                use_pca=True,
                ncomps=45,
                flat_hand_mean=False,
            )
            # obj_mesh = trimesh.load(dataset[0]["object_mesh_file"])

            data_list = []
            for data in dataset:
                # load object mesh
                obj_mesh = trimesh.load(data["object_mesh_file"])
                obj_verts = trimesh.sample.sample_surface(
                    obj_mesh, self.config["particles_per_obj"]
                )[0]

                # load hand mesh
                mano_pose = data["hand_pose"]
                mano_shape = data["hand_shape"]
                length = data["length"]

                obj_pose = to_transformation_matrix(data["object_pose"])
                obj_pcld = np.einsum(
                    "ni,kji->knj", obj_verts, obj_pose[:, :3, :3]
                ) + obj_pose[:, :3, 3].reshape(-1, 1, 3)
                obj_pcld = torch.FloatTensor(obj_pcld)

                hand_verts, _ = mano_layer.forward(
                    th_betas=torch.FloatTensor(mano_shape)
                    .reshape(-1, 10)
                    .repeat(length, 1),
                    th_pose_coeffs=torch.FloatTensor(mano_pose[:, :48]),
                    th_trans=torch.FloatTensor(mano_pose[:, 48:]),
                )
                hand_verts = hand_verts / 1000

                obj_pcld = obj_pcld.permute(1, 0, 2)  # (particles_per_obj, T,  3)
                hand_verts = hand_verts.permute(1, 0, 2)  # (778, T, 3)

                # hand_sample_indices = None
                # for his_len in range(length - 1):
                # if hand_sample_indices is None:
                _, hand_sample_indices = furthest_point_sampling(
                    hand_verts[:, 0], self.config["particles_per_hand"]
                )

                # is_on_finger =
                hand_pcld = hand_verts[
                    hand_sample_indices
                ]  # (particles_per_hand, n, 3)

                n_points_object = self.config["particles_per_obj"]
                n_points_hand = self.config["particles_per_hand"]
                pos = torch.cat([obj_pcld, hand_pcld], dim=0)

                is_finger_tips = np.isin(hand_sample_indices, tips_region)

                particle_type = torch.cat(
                    [
                        torch.full((n, 1), i, dtype=torch.int)
                        for i, n in enumerate([n_points_object, n_points_hand])
                    ],
                    dim=0,
                )
                if self.config["mask_fingertips"]:
                    particle_type[n_points_object:][is_finger_tips] = 2

                for seq_start in range(
                    length
                    - (self.config[f"{self.split}_sequence_length"] + 1)
                    * self.config["action_per_frames"]
                ):
                    if self.data_augment:
                        for i in range(self.num_augment):
                            """random sample rotation"""
                            rand_rot = generate_random_rotation(
                                num_samples=1, device="cpu"
                            ).squeeze(0)
                            rand_offset = torch.rand((1, 1, 3), device="cpu") * 2 - 1

                            obj_pcld_aug = (
                                torch.einsum("ij, ntj->nti", rand_rot, obj_pcld)
                                + rand_offset
                            )
                            hand_pcld_aug = (
                                torch.einsum("ij, ntj->nti", rand_rot, hand_pcld)
                                + rand_offset
                            )
                            action = torch.cat(
                                (
                                    torch.zeros_like(obj_pcld_aug[:, seq_start]),
                                    hand_pcld_aug[
                                        :, seq_start + self.config["action_per_frames"]
                                    ]
                                    - hand_pcld_aug[:, seq_start],
                                ),
                                dim=0,
                            )
                            pos = torch.cat([obj_pcld_aug, hand_pcld_aug], dim=0)

                            node_features = torch.cat(
                                [particle_type, action], dim=-1
                            )  # (n_objects + n_particles, 4)

                            # construct the graph based on the distances between particles
                            pos_dict = OrderedDict(
                                object_obs=(
                                    0,
                                    obj_pcld_aug[:, seq_start],
                                ),  # format: (start_index, position)
                                hand=(
                                    n_points_object,
                                    hand_pcld_aug[:, seq_start],
                                ),
                            )

                            # the create graph is for the first time step,
                            # and the following step will create graph on the fly
                            edge_index, edge_attr = connect_edges(
                                self.config,
                                pos_dict,
                            )

                            # next_pos = torch.cat(
                            #     [obj_pcld[his_len + 1], hand_verts[his_len + 1]],
                            #     dim=0,
                            # )  # (n_objects + n_particles, 3)
                            # (n_objects + n_particles, T， 3)

                            graph_data = pyg.data.Data(
                                x=node_features,
                                edge_index=edge_index,
                                edge_attr=edge_attr,
                                pos=pos[
                                    :,
                                    seq_start : seq_start
                                    + (self.config[f"{self.split}_sequence_length"] + 1)
                                    * self.config["action_per_frames"] : self.config[
                                        "action_per_frames"
                                    ],
                                ],  # current position
                                # next_pos=next_pos,  # ground truth for loss computation
                                # object_cls=object_cls,
                                obj_samples=torch.from_numpy(obj_verts).float(),
                                obj_mesh_path=data["object_mesh_file"],
                                # hand_sample_indices =
                            )

                            data_list.append(graph_data)
                    else:
                        action = torch.cat(
                            (
                                torch.zeros_like(obj_pcld[:, seq_start]),
                                hand_pcld[
                                    :, seq_start + self.config["action_per_frames"]
                                ]
                                - hand_pcld[:, seq_start],
                            ),
                            dim=0,
                        )

                        node_features = torch.cat(
                            [particle_type, action], dim=-1
                        )  # (n_objects + n_particles, 4)

                        # construct the graph based on the distances between particles
                        pos_dict = OrderedDict(
                            object_obs=(
                                0,
                                obj_pcld[:, seq_start],
                            ),  # format: (start_index, position)
                            hand=(
                                n_points_object,
                                hand_pcld[:, seq_start],
                            ),
                        )

                        # the create graph is for the first time step,
                        # and the following step will create graph on the fly
                        edge_index, edge_attr = connect_edges(
                            self.config,
                            pos_dict,
                        )

                        # next_pos = torch.cat(
                        #     [obj_pcld[his_len + 1], hand_verts[his_len + 1]],
                        #     dim=0,
                        # )  # (n_objects + n_particles, 3)
                        # (n_objects + n_particles, T， 3)

                        graph_data = pyg.data.Data(
                            x=node_features,
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                            pos=pos[
                                :,
                                seq_start : seq_start
                                + (self.config[f"{self.split}_sequence_length"] + 1)
                                * self.config["action_per_frames"] : self.config[
                                    "action_per_frames"
                                ],
                            ],  # current position
                            # next_pos=next_pos,  # ground truth for loss computation
                            # object_cls=object_cls,
                            obj_samples=torch.from_numpy(obj_verts).float(),
                            obj_mesh_path=data["object_mesh_file"],
                            # hand_sample_indices =
                        )

                        data_list.append(graph_data)
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
        elif self.split == "val" or self.split == "test":
            dataset_dir = self.config["data_dir"]
            # os.path.join(DATA_DIR, self.config["data_dir_prefix"])
            dataset = DexYCBVideoDataset(dataset_dir, "right", mode=self.split)
            mano_layer = ManoLayer(
                mano_root=f"{EXTERNAL_DIR}/mano/models",
                side="right",
                use_pca=True,
                ncomps=45,
                flat_hand_mean=False,
            )
            # obj_mesh = trimesh.load(dataset[0]["object_mesh_file"])

            data_list = []
            for data in dataset:
                obj_mesh = trimesh.load(data["object_mesh_file"])
                obj_verts = trimesh.sample.sample_surface(
                    obj_mesh, self.config["particles_per_obj"]
                )[0]

                mano_pose = data["hand_pose"]
                mano_shape = data["hand_shape"]
                length = data["length"]

                obj_pose = to_transformation_matrix(data["object_pose"])
                obj_pcld = np.einsum(
                    "ni,kji->knj", obj_verts, obj_pose[:, :3, :3]
                ) + obj_pose[:, :3, 3].reshape(-1, 1, 3)
                obj_pcld = torch.FloatTensor(obj_pcld)

                hand_verts, _ = mano_layer.forward(
                    th_betas=torch.FloatTensor(mano_shape)
                    .reshape(-1, 10)
                    .repeat(length, 1),
                    th_pose_coeffs=torch.FloatTensor(mano_pose[:, :48]),
                    th_trans=torch.FloatTensor(mano_pose[:, 48:]),
                )
                hand_verts = hand_verts / 1000

                obj_pcld = obj_pcld.permute(1, 0, 2)  # (particles_per_obj, T,  3)
                hand_verts = hand_verts.permute(1, 0, 2)  # (778, T, 3)

                # hand_sample_indices = None
                # for his_len in range(length - 1):
                # if hand_sample_indices is None:
                _, hand_sample_indices = furthest_point_sampling(
                    hand_verts[:, 0], self.config["particles_per_hand"]
                )
                hand_pcld = hand_verts[ 
                    hand_sample_indices
                ]  # (particles_per_hand, n, 3)

                n_points_object = self.config["particles_per_obj"]
                n_points_hand = self.config["particles_per_hand"]
                pos = torch.cat([obj_pcld, hand_pcld], dim=0)

                is_finger_tips = np.isin(hand_sample_indices, tips_region)

                particle_type = torch.cat(
                    [
                        torch.full((n, 1), i, dtype=torch.int)
                        for i, n in enumerate([n_points_object, n_points_hand])
                    ],
                    dim=0,
                )
                if self.config["mask_fingertips"]:
                    particle_type[n_points_object:][is_finger_tips] = 2

                seq_start = (
                    -(self.config[f"{self.split}_sequence_length"] + 1)
                    * self.config["action_per_frames"]
                )
                action = torch.cat(
                    (
                        torch.zeros_like(obj_pcld[:, seq_start]),
                        hand_pcld[:, seq_start + self.config["action_per_frames"]]
                        - hand_pcld[:, seq_start],
                    ),
                    dim=0,
                )

                node_features = torch.cat(
                    [particle_type, action], dim=-1
                )  # (n_objects + n_particles, 4)

                # construct the graph based on the distances between particles
                pos_dict = OrderedDict(
                    object_obs=(
                        0,
                        obj_pcld[:, seq_start],
                    ),  # format: (start_index, position)
                    hand=(
                        n_points_object,
                        hand_pcld[:, seq_start],
                    ),
                )

                # the create graph is for the first time step,
                # and the following step will create graph on the fly
                edge_index, edge_attr = connect_edges(
                    self.config,
                    pos_dict,
                )

                graph_data = pyg.data.Data(
                    x=node_features,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    pos=pos[
                        :,
                        seq_start :: self.config["action_per_frames"],
                    ],  # current position
                    # next_pos=next_pos,  # ground truth for loss computation
                    # object_cls=object_cls,
                    mano_shape=torch.from_numpy(mano_shape).float(),  # (10)
                    mano_pose=torch.from_numpy(
                        mano_pose[
                            -(self.config["test_sequence_length"] + 1)
                            * self.config["action_per_frames"] :: self.config[
                                "action_per_frames"
                            ]
                        ]
                    ).float(),  # (T, 48)
                    obj_pose=torch.from_numpy(
                        obj_pose[
                            -(self.config["test_sequence_length"] + 1)
                            * self.config["action_per_frames"] :: self.config[
                                "action_per_frames"
                            ]
                        ]
                    ).float(),  # (T, 4, 4)
                    obj_samples=torch.from_numpy(obj_verts).float(),
                    obj_mesh_path=data["object_mesh_file"],
                    # hand_sample_indices =
                )

                data_list.append(graph_data)
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
        elif self.split == "predict":
            dataset_dir = self.config[
                "data_dir"
            ]  # os.path.join(DATA_DIR, self.config["data_dir_prefix"])
            dataset = DexYCBVideoDataset(
                dataset_dir,
                "right",
                mode="test",
                filter_objects=self.config["filter_objects"],
                file_path=self.config["test_file_path"],
            )
            mano_layer = ManoLayer(
                mano_root=f"{EXTERNAL_DIR}/mano/models",
                side="right",
                use_pca=True,
                ncomps=45,
                flat_hand_mean=False,
            )

            data_list = []
            for data in dataset:
                obj_mesh = trimesh.load(dataset[0]["object_mesh_file"])
                obj_verts = trimesh.sample.sample_surface(
                    obj_mesh, self.config["particles_per_obj"]
                )[0]

                mano_pose = data["hand_pose"]
                mano_shape = data["hand_shape"]
                length = data["length"]

                obj_pose = to_transformation_matrix(data["object_pose"])
                obj_pcld = np.einsum(
                    "ni,kji->knj", obj_verts, obj_pose[:, :3, :3]
                ) + obj_pose[:, :3, 3].reshape(-1, 1, 3)
                obj_pcld = torch.FloatTensor(obj_pcld)

                hand_verts, _ = mano_layer.forward(
                    th_betas=torch.FloatTensor(mano_shape)
                    .reshape(-1, 10)
                    .repeat(length, 1),
                    th_pose_coeffs=torch.FloatTensor(mano_pose[:, :48]),
                    th_trans=torch.FloatTensor(mano_pose[:, 48:]),
                )
                hand_verts = hand_verts / 1000

                obj_pcld = obj_pcld.permute(1, 0, 2)  # (particles_per_obj, T,  3)
                hand_verts = hand_verts.permute(1, 0, 2)  # (778, T, 3)

                _, hand_sample_indices = furthest_point_sampling(
                    hand_verts[:, 0], self.config["particles_per_hand"]
                )
                hand_pcld = hand_verts[hand_sample_indices]

                n_points_object = self.config["particles_per_obj"]
                n_points_hand = self.config["particles_per_hand"]
                is_finger_tips = np.isin(hand_sample_indices, tips_region)

                particle_type = torch.cat(
                    [
                        torch.full((n, 1), i, dtype=torch.int)
                        for i, n in enumerate([n_points_object, n_points_hand])
                    ],
                    dim=0,
                )
                if self.config["mask_fingertips"]:
                    particle_type[n_points_object:][is_finger_tips] = 2

                pos_dict = OrderedDict(
                    object_obs=(0, obj_pcld[:, 0]),
                    hand=(n_points_object, hand_pcld[:, 0]),
                )

                edge_index, edge_attr = connect_edges(
                    self.config,
                    pos_dict,
                )

                graph_data = pyg.data.Data(
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    x=particle_type,
                    obj_pos=obj_pcld[
                        :,
                        -(self.config["test_sequence_length"] + 1)
                        * self.config["action_per_frames"],
                    ],  # (n_objects, 3)
                    mano_shape=torch.from_numpy(mano_shape).float(),  # (10)
                    mano_pose=torch.from_numpy(
                        mano_pose[
                            -(self.config["test_sequence_length"] + 1)
                            * self.config["action_per_frames"] :: self.config[
                                "action_per_frames"
                            ]
                        ]
                    ).float(),  # (T, 48)
                    obj_pose=torch.from_numpy(
                        obj_pose[
                            -(self.config["test_sequence_length"] + 1)
                            * self.config["action_per_frames"] :: self.config[
                                "action_per_frames"
                            ]
                        ]
                    ).float(),  # (T, 4, 4)
                    obj_samples=torch.from_numpy(obj_verts).float(),  # (n_objects, 3)
                    obj_mesh_path=data["object_mesh_file"],
                )
                data_list.append(graph_data)

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
        else:
            raise ValueError(f"Invalid split {self.split}")


class DexYCBDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

    def setup(self, stage):
        if stage == "fit":
            train_set = DexYCBDataset(self.config, "train")
            val_set = DexYCBDataset(self.config, "test")

            self.train, self.val = train_set, val_set

        elif stage == "test":
            raise NotImplementedError

        elif stage == "predict":
            self.predict = DexYCBDataset(self.config, "predict")
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
        )
