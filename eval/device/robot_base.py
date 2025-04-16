import numpy as np

class RobotBase:
    def __init__(
        self,
        name: str,
        control_mode: str,
    ):
        self.name = name
        self.control_mode = control_mode

        self.joint_names: dict[str, list[str]] = {}

    def set_action(self, action: np.ndarray) -> None:
        """
        @param: action: (N,)
        """
        raise NotImplementedError

    def set_state(self, state: np.ndarray) -> None:
        """
        @param: state (qpos): (N,)
        """
        raise NotImplementedError

    def clean_warning_error(self) -> None:
        raise NotImplementedError

    def get_qpos(self) -> np.ndarray:
        """
        @return: (N, )
        """
        raise NotImplementedError

    def get_obs(self) -> dict:
        """
        @return: observation dict
            {
                "qpos": (N),
                "qvel": (N),
                "qacc": (N),
            }
        """
        return {}

    def reset(self) -> None:
        """
        add reset logic here
        """
        self.init_obs = self.get_obs()

    def stop(self) -> None:
        pass

    @property
    def active_joint_names(self) -> list[str]:
        """should coordinate with digital twin"""
        raise NotImplementedError("active_joint_names is not implemented")

    @property
    def joint_limits(self) -> np.ndarray:
        raise NotImplementedError("joint_limits is not implemented")

    def get_controlled_joint_names(self, name: str) -> list[str]:
        return self.joint_names.get(name, [])
