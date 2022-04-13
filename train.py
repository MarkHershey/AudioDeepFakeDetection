"""This file is a simple example script training a CNN model on mel spectrograms."""
import argparse
import logging
from copy import deepcopy
from pathlib import Path
from typing import Callable, List, Optional, Union

import torch
from puts import timestamp_seconds
from torch.utils.data import ConcatDataset, DataLoader
from torchaudio.functional import compute_deltas

from DataLoader import lfcc, load_directory_split_train_test, mfcc
from models.simple import SimpleModel
from trainer import ModelTrainer
from utils import set_seed_all

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.DEBUG)


def init_logger(log_file: Union[Path, str]) -> None:
    # create file handler
    fh = logging.FileHandler(log_file)
    # create console handler
    ch = logging.StreamHandler()
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    LOGGER.addHandler(fh)
    LOGGER.addHandler(ch)
    return None


def train(
    real_dir: Union[Path, str],
    fake_dir: List[Union[Path, str]],
    amount_to_use: int = None,
    epochs: int = 20,
    device: str = "cuda" if torch.cuda.is_available else "cpu",
    batch_size: int = 32,
    save_dir: Union[str, Path] = None,
    test_size: float = 0.2,
) -> None:

    LOGGER.info("Loading data...")

    real_dataset_train, real_dataset_test = load_directory_split_train_test(
        path=real_dir,
        feature_fn=lfcc,
        feature_kwargs={},
        test_size=test_size,
        use_double_delta=True,
        phone_call=False,
        pad=True,
        label=1,
        amount_to_use=amount_to_use,
    )

    fake_dataset_train, fake_dataset_test = load_directory_split_train_test(
        path=fake_dir,
        feature_fn=lfcc,
        feature_kwargs={},
        test_size=test_size,
        use_double_delta=True,
        phone_call=False,
        pad=True,
        label=0,
        amount_to_use=amount_to_use,
    )

    dataset_train = ConcatDataset([real_dataset_train, fake_dataset_train])
    dataset_test = ConcatDataset([real_dataset_test, fake_dataset_test])
    LOGGER.info(f"Training model on {len(dataset_train)} audio files.")

    pos_weight = torch.Tensor([len(real_dataset_train) / len(fake_dataset_train)]).to(
        device
    )

    model = SimpleModel(feat_dim=40, time_dim=972, mid_dim=30, out_dim=1).to(device)

    ModelTrainer(
        batch_size=batch_size,
        epochs=epochs,
        device=device,
        optimizer_kwargs={
            "lr": 0.0001,
            "weight_decay": 0.0001,
        },
    ).train(
        model=model,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        save_dir=save_dir,
        pos_weight=pos_weight,
    )


def main(experiment_name: str = "debug"):

    root_save_dir = Path("saved")
    save_dir = root_save_dir / experiment_name
    log_file = save_dir / f"{timestamp_seconds()}.log"

    set_seed_all(42)
    init_logger(log_file)
    train(
        real_dir="/home/markhuang/Data/WaveFake/real",
        fake_dir="/home/markhuang/Data/WaveFake/fake/ljspeech_melgan",
        amount_to_use=None,
        epochs=50,
        device="cuda:1",
        batch_size=128,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main(experiment_name="in_dist_melgan_exp1")
