import numpy as np
from .robot_base import RobotBase
from xarm.wrapper import XArmAPI


class XarmRobot(RobotBase):
    def __init__(
        self,
        name: str,
        ip: str,
        control_mode: str,
        robot_init: list[float] = [0, 0, 0, 0, -180, 90, 270],
        speed_limit: float = 0.8,
    ):
        super().__init__(name, control_mode)

        self.robot = XArmAPI(ip)

        if control_mode == "position":
            self.use_delta = {"arm": False, "gripper": False}
            self.normalize_action = {"arm": False, "gripper": False}
        else:
            raise NotImplementedError("Only support position control now")

        self.robot_init = robot_init
        self.arm_speed_limit = speed_limit
        self.joint_names = {
            "arm": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"],
            # "gripper": ["drive_joint"],
        }

    def set_action(self, action: np.ndarray, ee_control=False, is_radian=None, wait=False) -> None:
        """
        @param: action
        """
        # action = action[0].cpu().numpy()
        # qpos = action[:7]

        # print(qpos)
        if self.control_mode == "position":
            if ee_control:
                assert len(action) == 6, f"incorrect dimension {len(action)}"
                self.set_ee_pos(action, is_radian=is_radian, wait=wait)
            else:
                assert len(action) == 7, f"incorrect dimension {len(action)}"
                self.robot.set_servo_angle(angle=action, is_radian=is_radian, speed=self.arm_speed_limit, wait=wait)
            # self.robot.set_gripper_position(action[-1] * 1000, wait=False)
        else:
            raise NotImplementedError("Only support position control now")
        
    def set_ee_pos(self, pos: np.ndarray, is_radian=None, wait=False) -> None:
        """
        @param: pos: (6), end effector position (x, y, z, roll, pitch, yaw)
        """
        assert len(pos) == 6, f"incorrect dimension {len(pos)}"
        if self.control_mode == "position":
            code = self.robot.set_position(*pos, speed=self.arm_speed_limit, is_radian=is_radian, wait=wait)
            assert code == 0, "Cannot set robot pose"
        else:
            raise NotImplementedError("Only support position control now")

    def set_state(self, state: np.ndarray) -> None:
        qpos = state[:7]
        assert len(qpos) == 7, f"incorrect dimension len(qpos)"
        self.robot.set_servo_angle(angle=qpos, is_radian=True, speed=0.1, wait=False)
        # self.robot.set_gripper_position(state[-1] * 1000, wait=True)

    def clean_warning_error(self) -> None:
        code, (error_code, warn_code) = self.robot.get_err_warn_code(show=True)

        assert code == 0, "Failed to get_err_warn_code"
        if warn_code != 0:
            self.robot.clean_warn()
        if error_code != 0:
            self.robot.clean_error()

        self.robot.motion_enable(enable=True)
        self.robot.set_mode(6)
        self.robot.set_state(state=0)

    def get_qpos(self) -> np.ndarray:
        # qpos = np.zeros(
        #     13,
        # )
        qpos = self.robot.get_joint_states(is_radian=True)[1][0]
        # qpos[7:] = self.robot.get_gripper_position()[1] / 1000
        return qpos

    def get_obs(self) -> dict:
        states = self.robot.get_joint_states(is_radian=True)[1]
        # gripper = self.robot.get_gripper_position()[1] / 1000

        # qpos = np.zeros(
        #     13,
        # )
        qpos = states[0]
        # qpos[7:] = gripper

        return {
            "qpos": qpos,  # for both arm and
            "qvel": np.array(states[1]),  # only for arm
            "qacc": np.array(states[2]),  # only for arm
            # "gripper": np.array(gripper),
        }
    
    def get_ee_pos(self, is_radian=None) -> np.ndarray:
        """
        @return: (6), end effector position (x, y, z, roll, pitch, yaw)
        """
        code, pose = self.robot.get_position(is_radian=is_radian)
        assert code == 0, "Failed to get_position"
        return pose


    def reset(self) -> None:
        self.robot.motion_enable(enable=True)
        self.robot.set_mode(0)
        self.robot.set_state(0)
        # self.set_state(np.deg2rad(np.array(self.robot_init)))

        init_qpos = np.array(self.robot_init) * np.pi / 180
        
        error_threshold = np.deg2rad(2)
        error = 1e5

        print("start initionalize arm !")

        while error > error_threshold:

            # print("set state: ")
            self.set_action(init_qpos, is_radian=True)

            cur_qpos = self.get_qpos()
            # print("cur_qpos", cur_qpos)
            # print("init_qpos", init_qpos)

            error = np.linalg.norm(cur_qpos - init_qpos)
            # print("current error", error, "threshold", error_threshold)
        # self.robot.set_position(*self.robot_init, speed=50, wait=True)
        # self.robot.set_position(*self.robot_init, speed=50, wait=True)
        # self.robot.set_gripper_mode(0)
        # self.robot.set_gripper_enable(True)
        # self.robot.set_gripper_speed(2000)
        # self.robot.set_gripper_position(650, wait=True)

        self.robot.set_mode(7)
        self.robot.set_state(0)

        super().reset()

    def stop(self) -> None:
        super().stop()

    @property
    def joint_limits(self) -> np.ndarray:
        return np.array([[-np.pi, np.pi]] * 7 + [[0, 0.85]])

    @property
    def active_joint_names(self) -> list[str]:
        return self.joint_names["arm"]  # + self.joint_names["gripper"]
