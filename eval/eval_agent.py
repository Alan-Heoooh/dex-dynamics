"""
Evaluation Agent.
"""

import time
import numpy as np
# from utils.transformation import xyz_rot_transform
import open3d as o3d
from device.camera.realsense import RealSenseRGBDCamera
from device.ability import AbilityHand
from device.xarm import XarmRobot 
# from device.xhand import XHand
from utils.projector import Projector
from device.camera.constant import *

# TODO: check hand qpos unit

class Agent:
    """
    Evaluation agent with Xarm, Ability hand and Intel RealSense RGB-D camera.

    Follow the implementation here to create your own real-world evaluation agent.
    """
    def __init__(
        self,
        # robot_ip,
        # pc_ip,
        control_mode="position",
        arm_ip="192.168.1.212",
        hand_tty_index="ttyUSB0",
        camera_serials=["102422074156", "337322071340", "013422062309", "102122060842"],
        hand_type="ability",
        arm_init: list[float] = [0, 0, 0, 0, -180, 90, 270],
        hand_init: list[float] = [-15, 15, 15, 15, 15, 15],
        arm_speed_limit: float = 20.0,
        **kwargs
    ): 
        self.camera_serials = camera_serials

        print("Init robot, hand, sensor, and camera.")
        self.robot = XarmRobot("xarm", arm_ip, control_mode, robot_init=arm_init, speed_limit=arm_speed_limit)
        self.robot.reset()
        print("Robot initialized.")
        time.sleep(1.5)

        if hand_type.lower() == "ability":
            self.hand = AbilityHand("ability", hand_tty_index, control_mode, robot_init=hand_init)
        # elif hand_type.lower() == "xhand":
        #     self.hand = XHand("xhand", control_mode=control_mode)
        else:
            raise ValueError(f"Unsupported hand type: {hand_type}. Supported types: ability, xhand.")
        self.hand.reset()
        print("Hand initialized.")
        time.sleep(1.5)

        # camera initialization
        self.cameras = {}
        for camera_serial in self.camera_serials:
            self.cameras["camera_serial"] = RealSenseRGBDCamera(serial = camera_serial)
            for _ in range(30): 
                self.cameras["camera_serial"].get_rgbd_image()
        print("Camera initialized.")
        print("Initialization Finished.")
        
    
    # @property
    # def ready_pose(self):
    #     return np.array([0.5, 0.0, 0.17, 0.0, 0.0, 1.0, 0.0])

    # @property
    # def ready_rot_6d(self):
    #     return np.array([-1, 0, 0, 0, 1, 0])
    
    def get_rgbd(self, camera_serial):
        colors, depths = self.cameras[camera_serial].get_rgbd_image()
        return colors, depths
    
    def get_pcld(self, camera_serial):
        color, depth = self.cameras[camera_serial].get_rgbd_image()
        _h = color.shape[0]
        _w = color.shape[1]
        color = o3d.geometry.Image(color)
        depth = o3d.geometry.Image(depth)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_scale=1000.0, convert_rgb_to_intensity=False)

        intrinsics = INTRINSICS[camera_serial]
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        cam_intrinsic = o3d.camera.PinholeCameraIntrinsic(_w, _h, fx, fy, cx, cy)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam_intrinsic)
        return pcd
    
    def get_tcp_pose(self):
        return self.robot.get_ee_pos()
    
    def get_hand_pose(self):
        """
        @return: hand joint angles (xhand: 12, ability: 6)
        """
        return self.hand.get_qpos()

    def set_tcp_pose(self, action, is_radian=None, wait=False):
        """
        @param: action: (6), end effector position (x, y, z, roll, pitch, yaw)
        """
        assert len(action) == 6, f"incorrect tcp pose dimension {len(action)}"
        self.robot.set_ee_pos(action, is_radian=is_radian, wait=wait)
        if wait:
            time.sleep(0.5)

    def set_hand_pose(self, action, is_radian=None):
        """
        @param: action: (6), hand joint angles (xhand: 12, ability: 6)
        """
        assert len(action) == 6, f"incorrect hand action dimension {len(action)}"
        self.hand.set_action(action, is_radian=is_radian)
    
    def stop(self):
        self.robot.stop()
        self.hand.stop()
    