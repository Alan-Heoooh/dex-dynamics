import os 
import sys
import numpy as np
import open3d as o3d
import torch
import time
import argparse

from ..object_perception.projector_np import Projector
from ..object_perception.sample import *
from eval_agent import Agent
from ..dynamics.config_parser import ConfigParser
from ..dynamics.models import DynamicsPredictor
from ..dynamics.dexwm.planning.plan import test_planning
from ..utils.macros import PLANNING_DIR


# TODO:
FILTER_MIN = np.array([0.0 - 0.1, 0.0 - 0.1, 0.0 - 0.07])
FILTER_MAX = np.array([0.0 + 0.07, 0.0 + 0.07, 0.0 + 0.07])

def get_object_pcld(agent, camera_serials, projector):
    pcld_total = o3d.geometry.PointCloud()
    for camera in camera_serials:
        pcld = agent.get_pcld(camera)
        pcld_marker = project_point_cloud_to_marker(pcld, camera, projector)
        pcld_total += pcld_marker
    pcld_total = filter_point_cloud(pcld_total, FILTER_MIN, FILTER_MAX)
    _, object_initial_pcld,  object_initial_mesh = sample(pcld_total, pcd_dense_prev=None, pcd_sparse_prev=None, hand_mesh=None, is_moving_back=None, visualize=False) # (300, 3)

    return object_initial_pcld.points # (300, 3)

def eval(config, save_dir, model):

    camera_serials=["102422074156", "337322071340", "013422062309", "102122060842"]
    calib_path = "/home/coolbot/Documents/git/dex-dynamics/calib"
    agent = Agent(
                    control_mode="position",
                    arm_ip="192.168.1.212",
                    hand_tty_index="ttyUSB0",
                    camera_serials=camera_serials,
                    hand_type="ability",
                    arm_init=[0, 0, 0, 0, -180, 90, 270],
                    hand_init=[-15, 15, 15, 15, 15, 15],
                    arm_speed_limit=20,
                )
    
    projector = Projector(calib_path)

    # get object point cloud
    scale = 1.3
    obj_init_pcld = get_object_pcld(agent, camera_serials, projector)
    obj_target_pcld = np.load("/home/coolbot/data/target_shape/K.npy")
    obj_target_pcld_center = np.mean(obj_target_pcld, axis=0)
    obj_target_pcld -= obj_target_pcld_center
    obj_target_pcld *= scale

    observation_batch = {
        "obj_init_pcd": obj_init_pcld,
        "obj_target_pcd": obj_target_pcld,
    }

    # relative pose with respect to last pose
    best_action = test_planning(config, save_dir, model, observation_batch)

    # execute action

    agent.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="planning")
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join(PLANNING_DIR, "deformable_planning_config.yaml"),
        type=str,
        help="",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="0",
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    config = ConfigParser.from_dynamics_args(parser)
    save_dir = config["exp_name"]

    # load model
    pretrained_path = config["pretrained_path"]
    model = DynamicsPredictor.load_from_checkpoint(pretrained_path)
    device = model.config.device
    model = model.to(device)

    eval(config, save_dir, model)


    ready_pose = np.array([547.592529, 208.72522, 581.037903, -65.47246, -74.465778, -87.507443])
    pinch_pose = np.array([436.899994, 217.199997, 310.299988, -162.571159, -57.70447, 75])