import pdb

import torch
import numpy as np
import scipy

from torch.autograd import Function
import emd
from torch import nn
import torch.nn.functional as F


class PositionLoss(torch.nn.Module):
    def __init__(self, loss_type, loss_weights, loss_by_n_points, N_list, object_weights=None):
        super(PositionLoss, self).__init__()

        self.loss_type = loss_type

        if "chamfer_emd" in self.loss_type:
            self.loss_weights = loss_weights

        self.loss_by_n_points = loss_by_n_points
        self.object_weights = object_weights
        self.N_list = N_list
        if object_weights is not None:
            assert len(object_weights) == len(N_list), f"object_weights {object_weights} should have the same length as N_list {N_list}"
        
        self.N_cum = np.cumsum([0] + self.N_list)

        self.chamfer = Chamfer()
        self.emd_cpu = EMDCPU()
        self.emd_cuda = EMDCPU()  # EMDCUDA()
        self.mse = MSE()
        
        print(f"loss type {loss_type} instantiated, with object losses weighted {object_weights}")

    # @profile
    def __call__(self, x, y, return_losses=False):
        # check self.N_cum is valid
        assert self.N_cum[-1] == x.shape[1], \
            f'x.shape is {x.shape} while self.N_cum is {self.N_cum}: x.shape[1] should equal to self.N_cum[-1]'

        # [object, hand]
        x_list = [
            x[:, self.N_cum[i] : self.N_cum[i + 1]] for i in range(len(self.N_cum) - 1)
        ]
        y_list = [
            y[:, self.N_cum[i] : self.N_cum[i + 1]] for i in range(len(self.N_cum) - 1)
        ]

        # import pdb; pdb.set_trace()

        loss = 0
        losses = []
        for i, (x, y, n) in enumerate(zip(x_list, y_list, self.N_list)):
            # compute loss for each object

            if self.loss_type == "mse":
                object_loss = self.mse(x, y)
                
            elif self.loss_type == "chamfer":
                object_loss = self.chamfer(x, y)
            elif self.loss_type == "emd_cpu":
                # loss += self.emd_cpu(x, y)
                object_loss = self.emd_cpu(x, y)
            elif self.loss_type == "emd_cuda":
                # loss += self.emd_cuda(x, y)
                object_loss = self.emd_cuda(x, y)
            elif "chamfer_emd" in self.loss_type:
                if self.loss_weights["chamfer"] > 0:
                    chamfer_loss = self.chamfer(x, y)
                    object_loss += self.loss_weights["chamfer"] * chamfer_loss

                if self.loss_weights["emd"] > 0:
                    if "cpu" in self.loss_type:
                        emd_loss = self.emd_cpu(x, y)
                    else:
                        emd_loss = self.emd_cuda(x, y)

                    object_loss += self.loss_weights["emd"] * emd_loss
            else:
                raise NotImplementedError("Only MSE is supported now")

            if self.object_weights is not None:
                loss += object_loss * self.object_weights[i] 
            else:
                loss += object_loss 

            # the loss recorded for each object is not affected
            # by the re-weighting procedure
            losses.append(object_loss)

            if self.loss_by_n_points:
                loss = loss * (self.N_list[0] / n)

        if return_losses:
            return loss, losses
        else:
            return loss


# class Chamfer:
#     @staticmethod
#     def compute(x, y, keep_dim=False):
#         # x: [B, M, D]
#         # y: [B, N, D]
#         M = x.shape[1]
#         N = y.shape[1]

#         # x: [B, M, N, D]
#         x_repeat = x[:, :, None, :].repeat(1, 1, N, 1)
#         # y: [B, M, N, D]
#         y_repeat = y[:, None, :, :].repeat(1, M, 1, 1)
#         # dis: [B, M, N]
#         dis_pos = torch.norm(x_repeat - y_repeat, dim=-1)

#         dis_x_to_nearest_y = torch.min(dis_pos, dim=2)[0] 
#         dis_y_to_nearest_x = torch.min(dis_pos, dim=1)[0]
        
#         if keep_dim:
#             return dis_x_to_nearest_y, dis_y_to_nearest_x
#         else:
#             return torch.max(dis_x_to_nearest_y) + torch.max(dis_y_to_nearest_x)
        
#     # @profile
#     def __call__(self, x, y):
#         return self.compute(x, y)
    
class Chamfer:
    @staticmethod
    def compute(x, y, probability, keep_dim=False):
        # x: [B, M, D]
        # y: [B, N, D]
        B, M, D = x.shape
        B_y, N, D_y = y.shape
        
        # Ensure x and y have the same batch size and feature dimensions
        assert B == B_y and D == D_y, "Batch size or feature dimension mismatch"
        
        # x: [B, M, N, D]
        x_repeat = x[:, :, None, :].repeat(1, 1, N, 1)
        # y: [B, M, N, D]
        y_repeat = y[:, None, :, :].repeat(1, M, 1, 1)
        # dis: [B, M, N]
        dis_pos = torch.norm(x_repeat - y_repeat, dim=-1)

        dis_x_to_nearest_y = torch.min(dis_pos, dim=2)[0]  # [B, M]
        dis_y_to_nearest_x = torch.min(dis_pos, dim=1)[0]  # [B, N]

        # probability = 1
        
        if keep_dim:
            return dis_x_to_nearest_y, dis_y_to_nearest_x
        else:
            # Flatten the distance tensors to select top 20% globally
            x_distances = dis_x_to_nearest_y.view(-1)  # Flatten to [B*M]
            y_distances = dis_y_to_nearest_x.view(-1)  # Flatten to [B*N]
            
            # Calculate top 20% for x to y distances
            k_x = max(1, int(round(probability * len(x_distances))))
            sorted_x = torch.sort(x_distances, descending=True)[0]
            top_x = sorted_x[:k_x]
            mean_x = torch.mean(top_x)
            
            # Calculate top 20% for y to x distances
            k_y = max(1, int(round(probability * len(y_distances))))
            sorted_y = torch.sort(y_distances, descending=True)[0]
            top_y = sorted_y[:k_y]
            mean_y = torch.mean(top_y)
            
            return mean_x + mean_y
        
    # @profile
    def __call__(self, x, y, probability=1):
        return self.compute(x, y, probability)

import torch
import torch.nn.functional as F

def differentiable_sdf_loss(pred_points, gt_points, grid_resolution=64, padding=0.1):
    """
    Fully differentiable SDF loss between two point clouds using trilinear interpolation.
    
    Args:
        pred_points: (N, 3) torch.Tensor on CUDA
        gt_points: (M, 3) torch.Tensor on CUDA
        grid_resolution: int, resolution of the SDF grid
        padding: float, padding around the bounding box

    Returns:
        loss: scalar tensor
    """
    device = pred_points.device
    all_points = torch.cat([pred_points, gt_points], dim=0)
    min_bound = all_points.min(dim=0)[0] - padding
    max_bound = all_points.max(dim=0)[0] + padding

    # Generate voxel grid
    lin = [torch.linspace(min_bound[i], max_bound[i], grid_resolution, device=device) for i in range(3)]
    grid_x, grid_y, grid_z = torch.meshgrid(*lin, indexing='ij')
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (R, R, R, 3)
    flat_grid = grid.reshape(-1, 3)  # (R^3, 3)

    # Compute SDF: min distance from grid point to GT points
    dists = torch.cdist(flat_grid.unsqueeze(0), gt_points.unsqueeze(0)).squeeze(0)  # (R^3, M)
    sdf_vals = dists.min(dim=1)[0]  # (R^3,)
    sdf_grid = sdf_vals.reshape(grid_resolution, grid_resolution, grid_resolution)

    # Normalize predicted points to [-1, 1] for grid_sample
    norm_pred = (pred_points - min_bound) / (max_bound - min_bound) * 2 - 1  # (N, 3)
    norm_pred = norm_pred.clamp(-1 + 1e-4, 1 - 1e-4)  # avoid edge issues

    # Interpolate SDF values at pred points
    sdf_grid = sdf_grid[None, None]  # (1, 1, D, H, W)
    coords = norm_pred.view(1, -1, 1, 1, 3)  # (1, N, 1, 1, 3)
    sampled_sdf = F.grid_sample(sdf_grid, coords, mode='bilinear', align_corners=True).squeeze()

    return (sampled_sdf ** 2).mean()



def differentiable_occupancy_loss(pred_points, gt_points, grid_resolution=100, sigma=0.05, padding=0.1):
    """
    Differentiable soft occupancy loss using Gaussian splatting.

    Args:
        pred_points: (N, 3) torch.Tensor on CUDA
        gt_points: (M, 3) torch.Tensor on CUDA
        grid_resolution: int
        sigma: float, Gaussian spread
        padding: float

    Returns:
        loss: scalar tensor
    """
    device = pred_points.device
    all_points = torch.cat([pred_points, gt_points], dim=0)
    min_bound = all_points.min(dim=0)[0] - padding
    max_bound = all_points.max(dim=0)[0] + padding

    # Create grid
    lin = [torch.linspace(min_bound[i], max_bound[i], grid_resolution, device=device) for i in range(3)]
    grid_x, grid_y, grid_z = torch.meshgrid(*lin, indexing='ij')
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (R, R, R, 3)
    grid = grid.unsqueeze(0)  # (1, R, R, R, 3)

    def splat(points):
        points = points.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # (N, 1, 1, 1, 3)
        dists = ((grid - points) ** 2).sum(-1)  # (N, R, R, R)
        gauss = torch.exp(-dists / (2 * sigma ** 2))  # (N, R, R, R)
        occupancy = gauss.max(dim=0)[0]  # (R, R, R)
        return occupancy

    pred_occ = splat(pred_points)
    gt_occ = splat(gt_points)

    # BCE loss over soft occupancy grids
    bce = F.binary_cross_entropy(pred_occ, gt_occ)
    return bce


class EMDCPU:
    # @profile
    def __call__(self, x, y):
        B = x.shape[0]

        x_ = x.detach().cpu().numpy()
        y_ = y.detach().cpu().numpy()

        y_ind_list = []
        for i in range(B):
            cost_matrix = scipy.spatial.distance.cdist(x_[i], y_[i])
            try:
                ind1, ind2 = scipy.optimize.linear_sum_assignment(
                    cost_matrix, maximize=False
                )
            except:
                print("Error in linear sum assignment!")

            y_ind_list.append(ind2)

        y_ind = np.stack(y_ind_list)
        batch_ind = torch.arange(B, device=x.device)[:, None]

        emd_pos = torch.mean(torch.norm(x - y[batch_ind, y_ind], dim=-1))

        return emd_pos


class emdFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2, eps, iters):

        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()

        assert(n == m)
        assert(xyz1.size()[0] == xyz2.size()[0])
        assert(n % 1024 == 0)
        assert(batchsize <= 512)

        xyz1 = xyz1.contiguous().float().cuda()
        xyz2 = xyz2.contiguous().float().cuda()
        dist = torch.zeros(batchsize, n, device='cuda').contiguous()
        assignment = torch.zeros(batchsize, n, device='cuda', dtype=torch.int32).contiguous() - 1
        assignment_inv = torch.zeros(batchsize, m, device='cuda', dtype=torch.int32).contiguous() - 1
        price = torch.zeros(batchsize, m, device='cuda').contiguous()
        bid = torch.zeros(batchsize, n, device='cuda', dtype=torch.int32).contiguous()
        bid_increments = torch.zeros(batchsize, n, device='cuda').contiguous()
        max_increments = torch.zeros(batchsize, m, device='cuda').contiguous()
        unass_idx = torch.zeros(batchsize * n, device='cuda', dtype=torch.int32).contiguous()
        max_idx = torch.zeros(batchsize * m, device='cuda', dtype=torch.int32).contiguous()
        unass_cnt = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()
        unass_cnt_sum = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()
        cnt_tmp = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()

        emd.forward(xyz1, xyz2, dist, assignment, price, assignment_inv, bid, bid_increments, max_increments, unass_idx, unass_cnt, unass_cnt_sum, cnt_tmp, max_idx, eps, iters)

        ctx.save_for_backward(xyz1, xyz2, assignment)
        return dist, assignment

    @staticmethod
    def backward(ctx, graddist, gradidx):
        xyz1, xyz2, assignment = ctx.saved_tensors
        graddist = graddist.contiguous()

        gradxyz1 = torch.zeros(xyz1.size(), device='cuda').contiguous()
        gradxyz2 = torch.zeros(xyz2.size(), device='cuda').contiguous()

        emd.backward(xyz1, xyz2, gradxyz1, graddist, assignment)
        return gradxyz1, gradxyz2, None, None


class emdModule(nn.Module):
    def __init__(self):
        super(emdModule, self).__init__()

    def forward(self, input1, input2, eps, iters):
        return emdFunction.apply(input1, input2, eps, iters)


class EMDCUDA:
    def __init__(self):
        self.emd = emdModule()

    # @profile
    def __call__(self, x, y):
        B = x.shape[0]
        N = x.shape[1]

        y_min, _ = torch.min(y, dim=1, keepdim=True)
        y_max, _ = torch.max(y, dim=1, keepdim=True)

        x_norm = (x - y_min) / (y_max - y_min)
        y_norm = (y - y_min) / (y_max - y_min)

        if N % 1024 == 0:
            # 0.005, 50 for training
            _, assignment = self.emd(x_norm, y_norm, 0.002, 50)
        else:
            n_repeat = 2 if N > 1024 else 1024 // N + 1
            x_repeat = x_norm.repeat(1, n_repeat, 1)[:, : 1024 * (N // 1024 + 1)]
            y_repeat = y_norm.repeat(1, n_repeat, 1)[:, : 1024 * (N // 1024 + 1)]

            _, assignment = self.emd(x_repeat, y_repeat, 0.002, 50)
            assignment = torch.remainder(assignment[:, :N], N)

        assignment = assignment.to(dtype=torch.int64, device=x.device)
        batch_ind = torch.arange(B, device=x.device)[:, None]

        emd_pos = torch.mean(torch.norm(x - y[batch_ind, assignment], dim=-1))
        return emd_pos


class MSE:
    def __call__(self, x, y):
        mse_pos = F.mse_loss(x, y)

        return mse_pos
