import sys
from tqdm import tqdm
import os
from device import XArm7Ability
import numpy as np
import torch
import time
from eval_agent import Agent

camera_serials = ["102422074156", "337322071340", "013422062309", "102122060842"]

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

agent.get_hand_pose()
agent.get_tcp_pose()


agent.stop()
