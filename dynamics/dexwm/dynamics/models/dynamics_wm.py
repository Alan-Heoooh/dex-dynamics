from collections import OrderedDict

import time
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
import torch_geometric as pyg
import random

from torch.optim.lr_scheduler import ReduceLROnPlateau

from dexwm.dynamics.dataset.dataset import (
    connect_edges,
    connect_edges_batched,
    compute_slice_indices,
)
from dynamics.models.dpi_net import DPINet
from dynamics.loss import PositionLoss, Chamfer, EMDCPU, MSE
# from dynamics.models.autoencoder import AutoEncoder

from ..utils_general import AverageMeter
from wis3d import Wis3D
from utils.geometry import aligned_points_to_transformation_matrix
import os
import trimesh

class DynamicsPredictor(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        # Extract configuration parameters
        self.config = config

        self.debug = config["debug"]
        self.points_per_obj = config["n_points"]
        self.cumul_points = np.cumsum([0] + self.points_per_obj)
        self.type_feat_len = 1
        # self.vis_feat_len = config["feature_dim_vision"]
        # self.act_feat_len = config["feature_dim_action"]
        self.pos_len = 3  # vector length of position (xyz)
        self.his_len = config["history_length"]
        self.train_seq_len = config["train_sequence_length"]
        self.test_seq_len = config["test_sequence_length"]

        print(f"train seq len: {self.train_seq_len}, test seq len: {self.test_seq_len}")
        self.n_object_points = config["n_points"][0]
        self.N = sum(config["n_points"])  # total number of points
        self.lr = config["optimizer"]["lr"]
        # self.recurr_T = config["recurr_T"]
        self.teacher_forcing_thres = config["teacher_forcing_thres"]

        # Initialize the GNN layer
        self.layers = DPINet(config)

        # Training time: initialize loss functions
        self.position_loss = PositionLoss(
            config["loss_type"],
            config["chamfer_emd_weights"],
            config["loss_by_n_points"],
            self.points_per_obj,
            config["object_weights"],
        )
        self.pos_loss_weight = config["pos_loss_weight"]

        # Test time: initialize different loss types
        self.chamfer_loss = Chamfer()
        self.emd_loss = EMDCPU()
        self.mse_loss = MSE()

        # Test time: initialize placeholders for predicted and ground truth state sequences for visualization
        # self.loss_dicts = []
        self.error_seq = []
        self.pred_state_seqs = []
        (
            self.total_emd_loss,
            self.total_chamfer_loss,
            self.total_mse_loss,
        ) = AverageMeter(), AverageMeter(), AverageMeter()

        # Save hyperparameters
        self.save_hyperparameters()

        self.save_dir = config["exp_name"]

        if self.debug:
            self.wis3d_graph = Wis3D(
                    out_folder=f"{self.save_dir}/wis3d_graph_3",
                    sequence_name=f"debug_knn",
                    xyz_pattern=("x", "-y", "-z"),
                )

            print(self.save_dir)
            self.wis3d_graph.set_scene_id(0)

    # @profile
    def forward(
        self,
        data,
        t,
        pred_pos_prev,
        train,
    ):  # pred_pos_prev, pred_tac_prev, train):
        # forward_start_t = time.time()
        # DataBatch(x=(B * N), edge_index=(2, E), edge_attr=(E, 4), y=[B * N_p, 3],
        # pos=[B * N, (his_len + seq_len) * pos_len], batch=[B * N], ptr=[?])
        B = data.num_graphs
        N = data.num_nodes // B

        assert self.his_len == 1, "only support history length = 1"
        if t == 0:
            # last history state is the graph built when loading the data
            window = data.clone("edge_index", "edge_attr", "batch", "ptr")
            pos_prev = data.pos[:, t]
            window.pos = pos_prev

            action = window.x[:, self.type_feat_len :].view(B, N, -1)

            # import pdb; pdb.set_trace()
        else:
            pos_prev = pred_pos_prev.view(B * N, -1)

            window = data.clone("edge_index", "edge_attr", "y", "batch", "ptr")
            edge_index, edge_attr = connect_edges_batched(
                self.config,
                pos_prev,
                N,
                data.num_graphs,
                self.cumul_points,
            )
            # sort the edge_index by the first row so that we can unbatch them if desired
            # this is optional, but you need it for visualization to compute edge_slice_indices
            edge_index_indices = torch.argsort(edge_index[0])
            edge_index = edge_index[:, edge_index_indices]
            edge_attr = edge_attr[edge_index_indices]

            window.edge_index = edge_index
            window.edge_attr = edge_attr
            window.pos = pos_prev

            node_type = data.x[:, : self.type_feat_len]
            action = (data.pos[:, t + 1] - data.pos[:, t]).view(B, N, -1)  # (B, N, 3)
            action[:, : self.n_object_points] = 0  # only take hand action

            window.x = torch.cat([node_type, action.view(B * N, -1)], dim=-1)
            # compute the first index in edge_index that corresponds to the first edge of each graph
            # this can be determined by finding the first index greater than or equal to N, then 2N, then 3N, etc.
            # this is because the edge_index is sorted by the first row
            edge_slice_indices = compute_slice_indices(
                window.edge_index[0], N, data.num_graphs
            )
            window._slice_dict["edge_index"] = edge_slice_indices
            window._slice_dict["edge_attr"] = edge_slice_indices

        # Normalize the position in the state
        # window.pos = (window.pos - self.pos_mean.repeat(self.his_len)) / self.pos_scale
        if self.debug:
            print("Debug...")
            print("Store the graph")
            # Visualize the graph
            import matplotlib.pyplot as plt
            import networkx as nx
            from datetime import datetime

            from wis3d import Wis3D

            os.makedirs(f"{self.save_dir}/scratch", exist_ok=True)
            for batch_idx in range(B):
                # import pdb; pdb.set_trace()

                # print(t, window[batch_idx])
                # g = pyg.utils.to_networkx(window[batch_idx], to_undirected=True)
                # pos_dict = {
                #     t: pos
                #     # for t, pos in enumerate(window[batch_idx].pos[:, :2].tolist())
                #     for t, pos in enumerate(window[batch_idx].pos.tolist())

                # }
                # nx.draw_networkx(g, pos_dict, with_labels=False, node_size=10)
                # plt.savefig(
                #     f'{self.save_dir}/scratch/{datetime.now().strftime("%d-%b-%Y-%H_%M_%S_%f")}_s{t}_b{batch_idx}.png'
                # )
                # plt.close()

                g = pyg.utils.to_networkx(window[batch_idx], to_undirected=True)

                # wis3d add point cloud
                self.wis3d_graph.add_point_cloud(
                    window[batch_idx].pos[: self.config["particles_per_obj"]],
                    torch.tensor([[255, 0, 0]]).repeat(self.config["particles_per_obj"], 1),
                    name="current_obj_pos",
                )
                self.wis3d_graph.add_point_cloud(
                    window[batch_idx].pos[self.config["particles_per_obj"]:],
                    torch.tensor([[0, 255, 0]]).repeat(self.config["particles_per_hand"], 1),
                    name="current_hand_pos",
                )

                # wis3d add lines
                # get the start and end indices of the rod
                start_points = []
                end_points = []
                for edge in g.edges():
                    start_points.append(window[batch_idx].pos[edge[0]].detach().cpu().numpy())
                    # print(edge)
                    end_points.append(window[batch_idx].pos[edge[1]].detach().cpu().numpy())

                self.wis3d_graph.add_lines(
                    start_points,
                    end_points,
                    torch.tensor([[0, 0, 255]]).repeat(len(start_points), 1),
                    name="current_edges",
                )
                self.wis3d_graph.increase_scene_id()


        # Use DPI network to predict non-rigid and rigid motion norms
        rigid_norm, non_rigid_norm = self.layers(window)

        # Process predictions depending on if we want rigid transformation or not
        # rigid_norm_mask = torch.cat([torch.ones(B, self.n_object_points, 3), torch.zeros(B, N - self.n_object_points, 3)], dim=1).to(self.device)
        if self.config["has_rigid_motion"]:
            rigid_norm[:, self.n_object_points:] = 0
        non_rigid_norm[:, self.n_object_points:] = 0

        # Denormalize the motion norms and compute the predicted state
        non_rigid = non_rigid_norm  # * self.pos_scale

        rigid = (rigid_norm + action) if self.config["has_rigid_motion"] else action

        # rigid:
        #   0 - n_object_points: object rigid motion (normalized by object center)
        #   n_object_points - N: hand relative action
        pred_pos_rigid = pos_prev.view(B, N, -1) + rigid

        pred_pos = (pred_pos_rigid if self.config["rigid_only"] else (pred_pos_rigid + non_rigid))

        # Calculate loss
        loss = {}

        # gt_pos = data.next_pos
        gt_pos = data.pos[:, t + 1].view(B, N, -1)
        if train:
            position_loss, position_losses = self.position_loss(
                pred_pos, gt_pos, return_losses=True
            )
            # position_loss, position_losses, chamfer_loss, emd_loss = self.position_loss(
            #     pred_pos, gt_pos, return_losses=True
            # )

            loss["train_pos"] = position_loss
            loss["train_pos_losses"] = position_losses

            # loss["chamfer"] = chamfer_loss
            # loss["emd"] = emd_loss

        else:
            loss["chamfer"] = self.chamfer_loss(pred_pos, gt_pos)
            loss["emd"] = self.emd_loss(pred_pos, gt_pos)
            loss["mse"] = self.mse_loss(pred_pos, gt_pos)
            loss["rmse"] = torch.sqrt(loss["mse"])

        return loss, pred_pos, gt_pos

    def dynamics_prediction_run_forward(self, action, pos_prev, node_type):
        B, N, _ = action.shape

        node_feat = torch.cat([node_type, action], dim=-1)

        window = pyg.data.Batch(
            x=None,
            edge_index=None,
            edge_attr=None,
            # y=target_state,
            pos=None,  # ground truth for loss computation
            forces=None,
            flows=None,
            # pressure=pressure,
            object_cls=None,
            # rand_index = torch.cat([torch.from_numpy(d['rand_index']) for d in state_list], dim=0)
        )
        edge_index, edge_attr = connect_edges_batched(
            self.config,
            pos_prev,
            N,
            B,  # data.num_graphs,
            self.cumul_points,
        )

        # sort the edge_index by the first row so that we can unbatch them if desired
        # this is optional, but you need it for visualization to compute edge_slice_indices
        edge_index_indices = torch.argsort(edge_index[0])
        edge_index = edge_index[:, edge_index_indices]
        edge_attr = edge_attr[edge_index_indices]

        # # sort the edge_index by the first row so that we can unbatch them if desired
        # # this is optional, but you need it for visualization to compute edge_slice_indices
        edge_index_indices = torch.argsort(edge_index[0])
        edge_index = edge_index[:, edge_index_indices]
        edge_attr = edge_attr[edge_index_indices]

        window.edge_index = edge_index
        window.edge_attr = edge_attr
        window.pos = pos_prev.view(B * N, -1)
        window.x = node_feat.view(B * N, -1)
        edge_slice_indices = compute_slice_indices(window.edge_index[0], N, B)
        window._slice_dict = dict()
        window._slice_dict["edge_index"] = edge_slice_indices
        window._slice_dict["edge_attr"] = edge_slice_indices
        window._num_graphs = B

        # # Use DPI network to predict non-rigid and rigid motion norms
        rigid_norm, non_rigid_norm = self.layers(window)
        # rigid_norm[:, self.n_object_points :] = 0
        # rigid = rigid_norm + action  # * self.pos_scale
        # non_rigid_norm[:, self.n_object_points :] = 0
        # pred_pos = pos_prev.view(B, N, -1) + rigid

        if self.config["has_rigid_motion"]:
            rigid_norm[:, self.n_object_points:] = 0
        non_rigid_norm[:, self.n_object_points:] = 0

        # Denormalize the motion norms and compute the predicted state
        non_rigid = non_rigid_norm  # * self.pos_scale

        rigid = (rigid_norm + action) if self.config["has_rigid_motion"] else action

        # rigid:
        #   0- n_object_points: object rigid motion (normalized by object center)
        #   n_object_points - N: hand relative action
        pred_pos_rigid = pos_prev.view(B, N, -1) + rigid

        pred_pos = (pred_pos_rigid if self.config["rigid_only"] else (pred_pos_rigid + non_rigid))
        return pred_pos

    def training_step(self, batch, batch_idx):
        train_pos_loss = 0
        train_chamfer_loss = 0
        train_emd_loss = 0
        pred_pos = None
        box_losses = torch.tensor([0, 0], dtype=torch.float32)

        B = batch.num_graphs

        # import pdb; pdb.set_trace()

        if batch_idx == 0:
            if getattr(self, "train_seq", None) is None:
                self.train_seq = 0
            self.train_seq += 1

        if self.train_seq % self.config["visualize_every"] == 0 and batch_idx == 0:
            self.wis3d_train = Wis3D(
                out_folder=f"{self.save_dir}/wis3d/train",
                sequence_name=f"debug_{self.train_seq}",
                xyz_pattern=("x", "-y", "-z"),
            )

            self.wis3d_train.set_scene_id(0)
                
        # TODO: add augmentation here instead of in the dataset

        for i in range(self.train_seq_len):
            loss, pred_pos, gts = self.forward(batch, i, pred_pos, True)

            train_pos_loss += loss["train_pos"]
            # train_chamfer_loss += loss["chamfer"]
            # train_emd_loss += loss["emd"]

            box_losses += torch.tensor(loss["train_pos_losses"])

            if loss["train_pos"] > self.teacher_forcing_thres:
                pred_pos = gts

            if (self.train_seq % self.config["visualize_every"] == 0):
                # for vis_idx in range(B):
                if batch_idx == 0:
                    for vis_idx in range(1):
                        # import pdb; pdb.set_trace()
                        self.wis3d_train.add_point_cloud(
                            batch.pos[:, i].view(B, -1, 3)[vis_idx][
                                : self.config["particles_per_obj"]
                            ],
                            torch.tensor([[255, 0, 0]]).repeat(
                                self.config["particles_per_obj"], 1
                            ),
                            name="current_obj_pos",
                        )
                        self.wis3d_train.add_point_cloud(
                            batch.pos[:, i].view(B, -1, 3)[vis_idx][
                                self.config["particles_per_obj"] :
                            ],
                            torch.tensor([[255, 0, 0]]).repeat(
                                self.config["particles_per_hand"], 1
                            ),
                            name="current_hand_pos",
                        )

                        self.wis3d_train.add_point_cloud(
                            pred_pos.view(B, -1, 3)[vis_idx][
                                : self.config["particles_per_obj"]
                            ],
                            torch.tensor([[0, 255, 0]]).repeat(
                                self.config["particles_per_obj"], 1
                            ),
                            name="pred_obj_pos",
                        )
                        self.wis3d_train.add_point_cloud(
                            pred_pos.view(B, -1, 3)[vis_idx][
                                self.config["particles_per_obj"] :
                            ],
                            torch.tensor([[0, 255, 0]]).repeat(
                                self.config["particles_per_hand"], 1
                            ),
                            name="pred_hand_pos",
                        )

                        self.wis3d_train.add_point_cloud(
                            gts.view(B, -1, 3)[vis_idx][: self.config["particles_per_obj"]],
                            torch.tensor([[0, 0, 255]]).repeat(
                                self.config["particles_per_obj"], 1
                            ),
                            name="gt_obj_pos",
                        )
                        self.wis3d_train.add_point_cloud(
                            gts.view(B, -1, 3)[vis_idx][self.config["particles_per_obj"] :],
                            torch.tensor([[0, 0, 255]]).repeat(
                                self.config["particles_per_hand"], 1
                            ),
                            name="gt_hand_pos",
                        )
                        self.wis3d_train.increase_scene_id()

        # normalize the loss by the sequence length
        train_pos_loss /= self.train_seq_len
        # train_chamfer_loss /= self.train_seq_len
        # train_emd_loss /= self.train_seq_len

        box_losses /= self.train_seq_len

        self.log("train_loss", train_pos_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch.num_graphs, )

        # self.log("train_chamfer_loss", train_chamfer_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch.num_graphs,)

        # self.log("train_emd_loss", train_emd_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch.num_graphs,)

        # for i, loss in enumerate(box_losses.tolist()):
        #     self.log(f"train_obj_loss_{i}",loss,prog_bar=False, on_epoch=True, on_step=False,batch_size=batch.num_graphs,)

        self.log("total_loss",train_pos_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch.num_graphs, )

        self.log("learning_rate", self.optimizer.param_groups[0]["lr"], prog_bar=False, on_epoch=True, on_step=False, batch_size=batch.num_graphs,)

        return train_pos_loss

    def validation_step(self, batch, batch_idx):
        train_pos_loss = 0
        pred_pos = None

        # tac_feat_bubbles = self.autoencoder.encode_structured(batch)
        box_losses = torch.tensor([0, 0], dtype=torch.float32)
        B = batch.num_graphs
        if batch_idx == 0:
            if getattr(self, "val_seq", None) is None:
                self.val_seq = 0
            self.val_seq += 1

        if self.val_seq % self.config["visualize_every"] == 0 and batch_idx == 0:
            self.wis3d_val = Wis3D(
                out_folder=f"{self.save_dir}/wis3d/validation",
                sequence_name=f"debug_{self.val_seq}",
                xyz_pattern=("x", "-y", "-z"),
            )
            self.wis3d_val.set_scene_id(0)

        # Calculate total loss over the sequence

        for i in range(self.test_seq_len):
            loss, pred_pos, gts = self.forward(batch, i, pred_pos, True)
            train_pos_loss += loss["train_pos"]
            box_losses += torch.tensor(loss["train_pos_losses"])

            if (self.val_seq % self.config["visualize_every"] == 0):
                for vis_idx in range(B):
                    self.wis3d_val.add_point_cloud(
                        batch.pos[:, i].view(B, -1, 3)[vis_idx][
                            : self.config["particles_per_obj"]
                        ],
                        torch.tensor([[255, 0, 0]]).repeat(
                            self.config["particles_per_obj"], 1
                        ),
                        name="current_obj_pos",
                    )
                    self.wis3d_val.add_point_cloud(
                        batch.pos[:, i].view(B, -1, 3)[vis_idx][
                            self.config["particles_per_obj"] :
                        ],
                        torch.tensor([[255, 0, 0]]).repeat(
                            self.config["particles_per_hand"], 1
                        ),
                        name="current_hand_pos",
                    )

                    self.wis3d_val.add_point_cloud(
                        pred_pos.view(B, -1, 3)[vis_idx][
                            : self.config["particles_per_obj"]
                        ],
                        torch.tensor([[0, 255, 0]]).repeat(
                            self.config["particles_per_obj"], 1
                        ),
                        name="pred_obj_pos",
                    )
                    self.wis3d_val.add_point_cloud(
                        pred_pos.view(B, -1, 3)[vis_idx][
                            self.config["particles_per_obj"] :
                        ],
                        torch.tensor([[0, 255, 0]]).repeat(
                            self.config["particles_per_hand"], 1
                        ),
                        name="pred_hand_pos",
                    )
                    self.wis3d_val.add_point_cloud(
                        gts.view(B, -1, 3)[vis_idx][: self.config["particles_per_obj"]],
                        torch.tensor([[0, 0, 255]]).repeat(
                            self.config["particles_per_obj"], 1
                        ),
                        name="gt_obj_pos",
                    )
                    self.wis3d_val.add_point_cloud(
                        gts.view(B, -1, 3)[vis_idx][self.config["particles_per_obj"] :],
                        torch.tensor([[0, 0, 255]]).repeat(
                            self.config["particles_per_hand"], 1
                        ),
                        name="gt_hand_pos",
                    )
                    self.wis3d_val.increase_scene_id()


        # normalize the loss by the sequence length
        train_pos_loss /= self.test_seq_len
        box_losses /= self.test_seq_len

        # print(f"Validation loss: {train_pos_loss}, box losses: {box_losses}, ")

        for i, loss in enumerate(box_losses.tolist()):
            self.log(
                f"val_obj_loss_{i}",
                loss,
                prog_bar=False,
                on_epoch=True,
                on_step=False,
                batch_size=batch.num_graphs,
            )

        self.log(
            "val_loss",  # instead of val_pos_loss in order to be comparable to prev results
            train_pos_loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            batch_size=batch.num_graphs,
        )

    def test_step(self, batch, batch_idx):
        test_mse_loss = 0
        test_emd_loss = 0
        test_chamfer_loss = 0
        pred_pos = None

        B = batch.num_graphs

        for i in range(self.test_seq_len):
            loss, pred_pos, gts = self.forward(batch, i, pred_pos, False)
            test_mse_loss += loss["mse"]
            test_emd_loss += loss["emd"]
            test_chamfer_loss += loss["chamfer"]
            test_rmse_loss = loss["rmse"]

        # normalize the loss by the sequence length
        test_mse_loss /= self.test_seq_len
        test_emd_loss /= self.test_seq_len
        test_chamfer_loss /= self.test_seq_len
        test_rmse_loss /= self.test_seq_len


        self.log(
            "test_mse_loss",  
            test_mse_loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            batch_size=batch.num_graphs,
        )

        self.log(
            "test_emd_loss",
            test_emd_loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            batch_size=batch.num_graphs,
        )
        self.log(
            "test_chamfer_loss",
            test_chamfer_loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            batch_size=batch.num_graphs,
        )

        self.log(
            "test_rmse_loss",
            test_rmse_loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            batch_size=batch.num_graphs,
        )

        return test_mse_loss, test_emd_loss, test_chamfer_loss, test_rmse_loss


    def predict_step(self, init_pos, action_samples, node_type):
        """
        init_pos: (N, 3)
        action_samples: (B, T, N, 3)
        node_type: (N)
        """

        B, T, _, _ = action_samples.shape
        # B = new_action_samples.shape[0]
        # N = node_pos.shape[1]
        # observed_his_len = node_pos.shape[2]

        start_time = time.time()

        pos_seq_object = []

        if len(init_pos.shape) == 2:
            particle_pos = init_pos.unsqueeze(0).repeat(B, 1, 1)
        else:
            particle_pos = init_pos # (B, N, 3)
        node_type = node_type.unsqueeze(0).repeat(B, 1, 1)
        start_time = time.time()

        # import pdb; pdb.set_trace()
        for t in range(T):
            action = action_samples[:, t].float()  # shape (B, N, 3)
            particle_pos_new = self.dynamics_prediction_run_forward(
                action, # (B, N, 3)
                particle_pos, # (B, N, 3)
                node_type, # (B, N, 1)
            )

            particle_pos = particle_pos_new

            pos_seq_object.append(particle_pos.clone())

        # print(
        #     f"Prediction completed, taken {time.time() - start_time}s, "
        #     f"for sequence length {action_samples.shape[1]}"
        # )

        pos_seq_object = torch.stack(pos_seq_object, dim=1)

        pred_state_seq = dict(
            object_obs=pos_seq_object[:, :, : self.cumul_points[1]],
            inhand=pos_seq_object[:, :, self.cumul_points[1] : self.cumul_points[2]],
            # finger_mask=node_type[:, self.n_object_points :] == 2,
        )

        return pred_state_seq

    def predict_step_v1(self, batch, batch_idx):
        B = batch.num_graphs
        N = batch.num_nodes // B
        S = batch.pos.shape[-1] // self.pos_len
        n_bubble_points = (N - self.n_object_points) // 2

        # Get the ground truth state
        gt_pos_seq = batch.pos.view(B, N, S, self.pos_len).transpose(1, 2)

        # Get the state history for each object in the batch
        pos_seq_object = [gt_pos_seq[:, : self.his_len, : self.cumul_points[-1]]]

        pred_pos, pred_tactile = None, None
        seq_tac_loss_logger = AverageMeter()

        tac_feat_bubbles = self.autoencoder.encode_structured(batch)
        # For each state in the sequence (S), calculate the predicted state and losses
        for i in range(S - self.his_len):
            loss, pred_pos, pred_tactile, gts = self.forward(
                batch, i, tac_feat_bubbles, pred_pos, pred_tactile, False
            )

            seq_tac_loss_logger.update(loss["tac"].item())
            self.total_tac_loss.update(loss["tac"].item())
            self.total_emd_loss.update(loss["emd"].item())
            self.total_chamfer_loss.update(loss["chamfer"].item())
            self.total_mse_loss.update(loss["mse"].item())
            # Append the predicted state to the object state sequence
            pos_seq_object.append(pred_pos[:, None])

        # print(f'last-state error: tactile = {loss["tac"].item()}, mse = {loss["mse"].item()}')
        # print(
        #     f'count: {self.total_tac_loss.count}, seq tac loss avg: {seq_tac_loss_logger.avg}, mse loss avg: {self.total_mse_loss.avg} '
        #     f'total tac loss avg: {self.total_tac_loss.avg}. ')

        self.error_seq.append((loss["mse"].item(), seq_tac_loss_logger.avg))

        # Get the pos sequence
        pos_seq_object = torch.cat(pos_seq_object, dim=1)

        # Get the vision feature sequence
        vision_feature_object = batch.x[
            ..., self.type_feat_len : self.type_feat_len + self.vis_feat_len
        ].view(B, 1, N, self.vis_feat_len)[:, :, : self.cumul_points[-1]]

        state_seq_object = torch.cat(
            (
                pos_seq_object,
                vision_feature_object.repeat(1, S, 1, 1),
            ),
            dim=-1,
        )

        pos_seq_tool = gt_pos_seq[:, :, self.cumul_points[-1] :]

        red = torch.tensor([1.0, 0.0, 0.0], device=gt_pos_seq.device)
        state_seq_tool = torch.cat(
            (
                pos_seq_tool,
                red.repeat(B, S, N - self.cumul_points[-1], 1),
            ),
            dim=-1,
        )

        pred_state_seq = dict(
            object_obs=state_seq_object[:, :, : self.cumul_points[1]]
            .detach()
            .cpu()
            .numpy(),
            inhand=state_seq_object[:, :, self.cumul_points[1] :]
            .detach()
            .cpu()
            .numpy(),
            bubble=state_seq_tool[:, :].detach().cpu().numpy(),
        )

        # self.pred_tac_seqs.append(tac_pred_seqs)
        return pred_state_seq

    def configure_optimizers(self):
        # Freeze the weights of the AE
        # for name, param in self.autoencoder.named_parameters():
        #     param.requires_grad = False
        # print(f"Autoencoder frozen. ")

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, "min", factor=0.8, patience=3, verbose=True
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
            "monitor": "total_loss",
        }

    def to_device(self, device):
        self.autoencoder = self.autoencoder.to(device)
        self.autoencoder.set_statistics(device)
        self.config.device = device
        self = self.to(device)
        return self
