import threading
import time
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pinocchio as pin
from .ability import AbilityHand
from .xarm import XarmRobot
from .robot_base import RobotBase

class XArm7Ability(RobotBase):
    def __init__(
        self,
        name: str,
        control_mode: str,
        use_arm=True,
        use_hand=True,
        arm_ip="192.168.1.212",
        hand_tty_index=0,
        arm_init: list[float] = [0, 0, 0, 0, -180, 90, 270],
        hand_init: list[float] = [-15, 15, 15, 15, 15, 15],
        arm_speed_limit: float = 0.8,
    ):
        self.name = name
        self.use_arm = use_arm
        self.use_hand = use_hand

        # Setup arm
        assert control_mode == "position", "Only support position control now"

        if use_arm:
            self.arm = XarmRobot("xarm", arm_ip, control_mode, robot_init=arm_init, speed_limit=arm_speed_limit)

        if use_hand:
            self.hand = AbilityHand("ability", hand_tty_index, control_mode, robot_init=hand_init)

        if use_arm and use_hand:
            self.joint_names = self.arm.joint_names.update(self.hand.joint_names)
        elif use_arm:
            self.joint_names = self.arm.joint_names
        elif use_hand:
            self.joint_names = self.hand.joint_names
        else:
            raise ValueError("At least one of use_hand or use_arm should be True.")
    
        
    def reset(self):
        if self.use_arm:
            self.arm.reset()

        if self.use_hand:
            self.hand.reset()

    def set_action(self, action: np.ndarray, ee_control=False) -> None:
        """
        @param: action: (N,)
        """
        if ee_control:
            assert len(action) == 12, f"incorrect dimension {len(action)}"
            arm_action = action[:6]
            hand_pos = action[6:]
        else:
            assert len(action) == 13, f"incorrect dimension {len(action)}"
            arm_action = action[:7]
            hand_pos = action[7:]

        if self.use_arm:
            self.arm.set_action(arm_action, ee_control)
        if self.use_hand:
            self.hand.set_action(hand_pos)

    def set_arm_ee_pos(self, pos: np.ndarray, is_radian=None, wait=False) -> None:
        """
        @param: pos: (6), end effector position (x, y, z, roll, pitch, yaw)
        """
        if self.use_arm:
            self.arm.set_ee_pos(pos, is_radian, wait)
        else:
            raise ValueError("Arm is not used")
    
    def get_arm_ee_pos(self) -> np.ndarray:
        if self.use_arm:
            return self.arm.get_ee_pos()
        else:
            raise ValueError("Arm is not used")
        
    def set_hand_qpos(self, hand_pos: np.ndarray, is_radian=None) -> None:
        if self.use_hand:
            self.hand.set_action(hand_pos, is_radian=None)
        else:
            raise ValueError("Hand is not used")
        
    def get_hand_qpos(self) -> np.ndarray:
        if self.use_hand:
            return self.hand.get_qpos()
        else:
            raise ValueError("Hand is not used")

    def get_qpos(self) -> np.ndarray:
        qpos = np.zeros(13)

        if self.use_arm:
            qpos[:7] = self.arm.get_qpos()

        if self.use_hand:
            qpos[7:] = self.hand.get_qpos()
        
        return qpos

    def stop(self):
        if self.use_arm:
            self.arm.stop()

        if self.use_hand:
            self.hand.stop()

    @property
    def active_joint_names(self) -> list[str]:
        if self.use_arm and self.use_hand:
            return self.arm.active_joint_names + self.hand.active_joint_names
        elif self.use_arm:
            return self.arm.active_joint_names
        elif self.use_hand:
            return self.hand.active_joint_names
        else:
            raise ValueError("No active joint names")
