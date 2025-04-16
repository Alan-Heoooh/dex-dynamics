from .robot_base import RobotBase
import numpy as np
from .ability_hand import RealAbilityHand


class AbilityHand(RobotBase):
    def __init__(
        self,
        name: str,
        hand_tty_index: int,
        control_mode: str,
        robot_init: list[float],
        verbose=False,
    ):  # TODO):
        super().__init__(name, control_mode)


        self.robot = RealAbilityHand(
            # usb_port=f"/dev/{hand_tty_index}",
            reply_mode=0x10,
            hand_address=0x50,
            plot_touch=False,
            verbose=verbose,
        )
        self.robot_init = robot_init
        self.joint_names = {"hand": ["thumb_q1", "index_q1", "middle_q1", "ring_q1", "little_q1", "thumb_q2"]}

    def set_action(self, action: np.ndarray, is_radian=None) -> None:
        assert len(action) == 6, f"incorrect dimension {len(action)}"
        if is_radian == None:
            action = np.deg2rad(action)
        self.robot.set_joint_angle(action, reply_mode=0x10)
        # raise NotImplementedError

    def set_state(self, state: np.ndarray) -> None:
        raise NotImplementedError

    def clean_warning_error(self) -> None:
        raise NotImplementedError

    def get_qpos(self) -> np.ndarray:
        """
        @return: (N, )
        """
        return np.array(self.robot.get_data().tolist()[:6])
        # raise NotImplementedError

    def get_obs(self) -> dict:
        """
        @return: observation dict
            {
                "qpos": (N),
                "qvel": (N),
                "qacc": (N),
            }
        """
        return {"qpos": self.get_qpos()}

    def reset(self) -> None:
        """
        add reset logic here
        """
        # self.init_obs = self.get_obs()
        self.robot.start_process()
        joints = np.deg2rad(np.array(self.robot_init))
        self.robot.set_joint_angle(joints, reply_mode=0x10)
        
        super().reset()

    def stop(self) -> None:
        self.robot.stop_process()

    @property
    def active_joint_names(self) -> list[str]:
        """should coordinate with digital twin"""
        return self.joint_names["hand"]

    @property
    def joint_limits(self) -> np.ndarray:
        raise NotImplementedError("joint_limits is not implemented")
