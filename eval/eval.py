import os 
import sys
import numpy as np
import open3d as o3d
import torch
import time

from ..object_perception.projector_np import Projector
from eval_agent import Agent


def eval():

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

    ready_pose = np.array([547.592529, 208.72522, 581.037903, -65.47246, -74.465778, -87.507443])
    pinch_pose = np.array([436.899994, 217.199997, 310.299988, -162.571159, -57.70447, 75])

    pcld_total = []
    for camera in camera_serials:
        pcld = agent.get_pcld(camera)
        
    

    agent.stop()


if __name__ == "__main__":
    eval()