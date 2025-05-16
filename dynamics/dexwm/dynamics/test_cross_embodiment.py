import sys
import os

# add project directory to PATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import *
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from dynamics.config_parser import ConfigParser
from utils.utils import *
from utils.visualizer import *
from dynamics.models import DynamicsPredictor
# from dynamics.dataset import DexYCBDataModule
from dynamics.dataset import DeformableDataModule, SimulationDataModule


def test(config, save_dir, model, data_module):

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[config.device.index] if config.device.index is not None else 1,
        default_root_dir=save_dir,
        log_every_n_steps=1,
        # logger=logger,
        num_sanity_val_steps=0,
        deterministic=True,
    )

    losses = trainer.test(model, data_module)

    return losses[0] # 1 epoch

HANDS = ["ability_hand", "allegro_hand", "leap_hand", "shadow_hand", "xhand"]

HAND_DATA_DIR = {
    "ability_hand": "/zihao-fast-vol/rewarped_softness_50/pinch_trajectories_ability_hand_urdf",
    "allegro_hand": "/zihao-fast-vol/rewarped_softness_50/pinch_trajectories_allegro_hand_urdf",
    "leap_hand": "/zihao-fast-vol/rewarped_softness_50/pinch_trajectories_leap_hand",
    "shadow_hand": "/zihao-fast-vol/rewarped_softness_50/pinch_trajectories_shadow_urdf",
    "xhand": "/zihao-fast-vol/rewarped_softness_50/pinch_trajectories_xhand_urdf",
}

HAND_CKPT_DIR = {
    "ability_hand": "/zihao-fast-vol/ckpts/ability_hand-epoch=54-step=11165-val_loss=0.00001.ckpt",
    "allegro_hand": "/zihao-fast-vol/ckpts/allegro_hand-epoch=67-step=13804-val_loss=0.00001.ckpt",
    "leap_hand": "/zihao-fast-vol/ckpts/leap_hand-epoch=65-step=13398-val_loss=0.00001.ckpt",
    "shadow_hand": "/zihao-fast-vol/ckpts/shadow_hand-epoch=66-step=13601-val_loss=0.00001.ckpt",
    "xhand": "/zihao-fast-vol/ckpts/xhand_epoch=62-step=12789-val_loss=0.00001.ckpt",
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="dynamics")
    print("dynamics dir:", DYNAMICS_DIR)
    parser.add_argument("-c", "--config", default=os.path.join(DYNAMICS_DIR, "simulation_dynamics_config.yaml"),type=str, help="config file path (default: dynamics_config.yaml)",)
    parser.add_argument("-r", "--resume",default=None, type=str, help="path to latest checkpoint (default: None)", )
    parser.add_argument("-d", "--device", default="0", type=str, help="indices of GPUs to enable (default: all)",)

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="data directory",
    )

    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="experiment name",
    )

    parser.add_argument(
        "--train_batch_size",
        type=str,
        default=None,
        help="train batch size",
    )

    parser.add_argument(
        "--log_name",
        type=str,
        default=None,
        help="log name",
    )

    parser.add_argument(
        "--num_workers",
        type=str,
        default=None,
        help="number of workers",
    )
    parser.add_argument(
        "--action_per_frames",
        type=str,
        default=None,
        help="number of actions per frame",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="path to the checkpoint file",
    )

    args = parser.parse_args()

    config = ConfigParser.from_dynamics_args(parser)

    if args.data_dir is not None:
        config.config["data_dir"] = args.data_dir
    if args.exp_name is not None:
        config.config["exp_name"] = args.exp_name  
    if args.train_batch_size is not None:
        config.config["train_batch_size"] = int(args.train_batch_size)
    if args.log_name is not None:
        config.config["log_name"] = args.log_name
    if args.num_workers is not None:
        config.config["num_workers"] = int(args.num_workers)
    if args.action_per_frames is not None:
        config.config["action_per_frames"] = int(args.action_per_frames)
    if args.ckpt_path is not None:
        config.config["ckpt_path"] = args.ckpt_path

    exps_dir = "/zihao-fast-vol/exps/cross_embodiment_test"

    loss_dict = {}

    for from_hand in HANDS:
        for to_hand in HANDS:
            print(f"Testing Cross-embodiment of dynamics model from {from_hand} to {to_hand}...")
            config.config["ckpt_path"] = HAND_CKPT_DIR[from_hand]
            config.config["data_dir"] = HAND_DATA_DIR[to_hand]
            config.config["exp_name"] = os.path.join(exps_dir, f"cross_embodiment_test_{from_hand}_to_{to_hand}")
            config.config["log_name"] = f"cross_embodiment_test_{from_hand}_to_{to_hand}"

            model = DynamicsPredictor.load_from_checkpoint(checkpoint_path=config.config["ckpt_path"])
            data_module = SimulationDataModule(config)
        
            save_dir = config["exp_name"]
            losses = test(config, save_dir, model, data_module)
            loss_dict[f"{from_hand}_to_{to_hand}"] = losses

    # Save the loss_dict to a file
    loss_dict_path = os.path.join(exps_dir, "cross_embodiment_test_loss_dict.npy")
    np.save(loss_dict_path, loss_dict)
    print("Cross-embodiment test loss dict saved to:", loss_dict_path)

