import torch 

class MimicJointForward:
    def __init__(self, action_names, qpos_names, mimic_joint_map, device = "cuda"):
        
        self.action_names = action_names
        self.qpos_names = qpos_names
        self.device = device
        
        self.action_indices = [qpos_names.index(name) for name in action_names]

        
        self.mimic_target_names = []
        self.mimic_source_names = []  # target = source * multiplier + offset
        self.mimic_source_indices = []  # indices source joints in action_names
        self.mimic_target_indices = []  # indices target joints in qpos names
        self.mimic_multipliers = []
        self.mimic_offsets = []

        if mimic_joint_map is not None:
            for k, v in mimic_joint_map.items():
                assert k not in action_names, f"{k} not in action_names"
                assert (
                    v["mimic"] in action_names
                ), f"{v['mimic']} should be in action_names"
                self.mimic_target_names.append(k)
                self.mimic_source_names.append(v["mimic"])
                self.mimic_multipliers.append(v["multiplier"])
                self.mimic_offsets.append(v["offset"])
            self.mimic_source_indices = [
                action_names.index(name) for name in self.mimic_source_names
            ]
            self.mimic_target_indices = [
                qpos_names.index(name) for name in self.mimic_target_names
            ]
            self.mimic_multipliers = torch.tensor(self.mimic_multipliers).to(self.device)
            self.mimic_offsets = torch.tensor(self.mimic_offsets).to(self.device)

        assert len(qpos_names) == len(self.action_names) + len(
            self.mimic_target_names
        ), "action_names and mimic_target_names should cover all active joints"

    def forward(self, action):
        """
        input: action: torch.Tensor (B, n_actions)
        output: action: torch.Tensor (B, n_joints)
        """
        assert action.ndim ==2 and action.shape[1] == len(self.action_names), "action shape should be (B, n_actions)"
        qpos = torch.zeros(action.shape[0], len(self.qpos_names)).to(self.device)
        # import pdb; pdb.set_trace()
        qpos[:, self.action_indices] = action
        qpos[:, self.mimic_target_indices] = (action[:, self.mimic_source_indices] * self.mimic_multipliers + self.mimic_offsets)
        return qpos