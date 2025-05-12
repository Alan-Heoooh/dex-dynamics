import sys
import os

# add project directory to PATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import *
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from dynamics.config_parser import ConfigParser

# from dynamics.dataset import DynamicsDataModule
from utils.utils import *
from utils.visualizer import *
from dynamics.models import DynamicsPredictor
# from dynamics.dataset import DexYCBDataModule
from dynamics.dataset import DeformableDataModule, SimulationDataModule

input_dim, encoding_dim = 5, 5


def train(config, save_dir):
    # data_module = DynamicsDataModule(config)
    model = DynamicsPredictor(config)
    
    data_module = SimulationDataModule(config)

    best_checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        filename="{epoch}-{step}-{val_loss:.5f}",
    )
    latest_checkpoint_callback = ModelCheckpoint(
        save_top_k=1,  # Keep only the latest checkpoint
        filename="latest-{epoch}-{step}",  # Name the file as "latest"
    )
    tensorboard_logger = TensorBoardLogger(
        os.path.join(save_dir, "train_tb_logs"), name=config["log_name"]
    )

    if not config["debug"]:
        wandb_logger = WandbLogger(project="dexwm", name=config['log_name'])
        logger = [tensorboard_logger, wandb_logger]
    else:
        logger = [tensorboard_logger]

    # import pdb; pdb.set_trace()

    # train model
    trainer = pl.Trainer(
        max_epochs=config["optimizer"]["max_epoch"],
        accelerator="gpu",
        devices=[config.device.index] if config.device.index is not None else 1,
        callbacks=[
            TQDMProgressBar(refresh_rate=20),
            # EarlyStopping(monitor="val_loss", mode="min", patience=20),
            best_checkpoint_callback,
            latest_checkpoint_callback,
        ],
        default_root_dir=save_dir,
        logger=logger,  # , wandb_logger],
        num_sanity_val_steps=0,
        log_every_n_steps=config["log_every_n_steps"],
        deterministic=True,
    )

    trainer.fit(model=model, datamodule=data_module)

    return best_checkpoint_callback.best_model_path


def test(config, stats, save_dir, best_model_path):
    model = DynamicsPredictor.load_from_checkpoint(checkpoint_path=best_model_path)

    data_module = SimulationDataModule(config)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[config.device.index] if config.device.index is not None else 1,
        default_root_dir=save_dir,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        deterministic=True,
    )

    trainer.test(model, data_module)

    tb_log_dir = os.path.dirname(os.path.dirname(best_model_path))
    loss_array_path = os.path.join(tb_log_dir, "metrics.npy")
    np.save(loss_array_path, model.test_losses)
    print(f"AE testing losses saved to {loss_array_path}")


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


    save_dir = config["exp_name"]

    best_model_path = train(config, save_dir)

    # test(config, stats, save_dir, best_model_path)
