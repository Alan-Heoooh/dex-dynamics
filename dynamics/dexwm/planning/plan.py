import os
import sys
import time
import argparse
import wandb

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from planning.cost_functions import CostFunction
from utils.macros import PLANNING_DIR
from planning.samplers import *
from planning.planner import MPPIOptimizer
from dynamics.dataset import DeformableDataModule
from dynamics.config_parser import ConfigParser
from dynamics.models import DynamicsPredictor
from utils.utils import *
from utils.visualizer import *

from dexwm.utils.pcld_wrapper import HandPcldWrapper
from dexwm.utils.pcld_wrapper.robot_pcld import RobotPcldWrapper
from torch.utils.tensorboard import SummaryWriter

torch.set_float32_matmul_precision("medium")


class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb

    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        self.writer.close()


def test_planning(config, save_dir):
    run_name = time.strftime("%m%d_%H%M%S")
    if not config["debug"]:
        wandb.init(
            project="planning",
            sync_tensorboard=False,
            config=config,
            save_code=True,
            group="planning",
            name=run_name,
        )
    writer = SummaryWriter(log_dir=f"{save_dir}/{run_name}")
    logger = Logger(log_wandb=not config["debug"], tensorboard=writer)

    pretrained_path = config["pretrained_path"]
    model = DynamicsPredictor.load_from_checkpoint(pretrained_path)
    # model.config.device = torch.device("cuda:1")
    device = model.config.device
    model = model.to(device)
    # config = ConfigParser(dict())
    debug = config["debug"]
    config.update_from_yaml(config["dynamics_config_path"])
    config._config["debug"] = debug

    assert config["test_batch_size"] == 1, "test batch size must be 1"

    # save config
    write_yaml(config, f"{save_dir}/{run_name}/config.yaml")

    data_module = DeformableDataModule(config)
    data_module.prepare_data()
    data_module.setup("predict")
    dataloader = data_module.predict_dataloader()

    # test model
    # get a batch of data
    data_iter = iter(dataloader)

    batch = next(data_iter).to(device)

    horizon = config["horizon"]
    action_dim = config["action_dim"]

    # build a planner
    # first build a sampler
    sampler = CorrelatedNoiseSampler(a_dim=action_dim, beta=config["beta"], horizon=horizon, num_repeat=1)
    # sampler = GaussianSampler(horizon=horizon, a_dim=action_dim)

    if config["predict_type"] == "hand":
        point_cloud_wrapper = HandPcldWrapper(
            particles_per_hand=config["particles_per_hand"],
            num_samples=config["num_samples"],
        )
    elif config["predict_type"] == "ability_hand_right" or config["predict_type"] == "xhand_right":
        point_cloud_wrapper = RobotPcldWrapper(
            config["predict_type"],
            particles_per_hand=config["particles_per_hand"],
            num_samples=config["num_samples"],
        )
    else:
        raise ValueError(f"Unknown predict type: {config['predict_type']}")

    cost_function = CostFunction(
        config=config,
        obj_funcs=config["loss"]["names"],
        weights=config["loss"]["weights"],
        eval_func=config["loss"]["eval"],
        last_states=config["loss"]["last_states"],
    )

    # then build a planner
    planner = MPPIOptimizer(
        sampler=sampler,
        point_cloud_wrapper=point_cloud_wrapper,
        model=model,
        objective=cost_function,
        a_dim=action_dim,
        horizon=horizon,
        num_samples=config["num_samples"],
        gamma=config["gamma"],  # the larger the more exploitation
        num_iters=config["num_execution_iterations"],
        init_std=config["init_std"],
        config=config,
        log_every=config["log_every"],
        logger=logger,
    )

    # then plan
    if config["debug"]:
        log_dir = f"{save_dir}"
    else:
        log_dir = f"{save_dir}/{run_name}"

    mu, _, best_actions = planner.plan(
        t=1,
        log_dir=log_dir,  # /{run_name}",
        observation_batch=batch,
        # object_type_feat,
        # vision_feature,
        # node_pos,
        # action_history=action_hist,
        # goal=goal,
        visualize_k=config["visualize_k"],
        return_best=True,
    )

    return best_actions


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
    best_actions = test_planning(config, save_dir)