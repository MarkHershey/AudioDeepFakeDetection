import argparse
import logging
from copy import deepcopy
from pathlib import Path
from typing import Callable, List, Optional, Union

import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchaudio.functional import compute_deltas

from dataloader import load_directory_split_train_test, mel_spectrogram, mfcc
from models import ShallowCNN
from trainer import GDTrainer
from utils import set_seed_all

LOGGER = logging.getLogger()


def init_logger(log_file):
    LOGGER.setLevel(logging.INFO)

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


def save_model(
    model: torch.nn.Module,
    model_class: str,
    model_dir: Union[Path, str],
    name: str,
) -> None:
    full_model_dir = Path(f"{model_dir}/{model_class}/{name}")
    if not full_model_dir.exists():
        full_model_dir.mkdir(parents=True)

    torch.save(model.state_dict(), f"{full_model_dir}/model.pth")


def train(
    real_dir: Union[Path, str],
    fake_dir: List[Union[Path, str]],
    amount_to_use: int = None,
    epochs: int = 20,
    device: str = "cuda" if torch.cuda.is_available else "cpu",
    batch_size: int = 32,
    model_dir: Optional[str] = None,
    test_size: float = 0.2,
) -> None:

    LOGGER.info("Loading data...")

    real_dataset_train, real_dataset_test = load_directory_split_train_test(
        path=real_dir,
        feature_fn=mfcc,
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
        feature_fn=mfcc,
        feature_kwargs={},
        test_size=test_size,
        use_double_delta=True,
        phone_call=False,
        pad=True,
        label=0,
        amount_to_use=amount_to_use,
    )

    dataset_train = ConcatDataset([real_dataset_train, fake_dataset_train])
    LOGGER.info(f"Training model on {len(dataset_train)} audio files...")

    pos_weight = torch.Tensor([len(real_dataset_train) / len(fake_dataset_train)]).to(
        device
    )

    model = ShallowCNN(1, 1).to(device)

    model = GDTrainer(
        device=device,
        batch_size=batch_size,
        epochs=epochs,
        optimizer_kwargs={
            "lr": 0.0001,
            "weight_decay": 0.0001,
        },
    ).train(
        dataset=dataset_train,
        model=model,
        test_len=test_size,
        pos_weight=pos_weight,
    )

    if model_dir is not None:
        save_model(
            model=model,
            model_class="shallow_cnn",
            model_dir=model_dir,
            name=str(fake_dir).strip("/").replace("/", "_"),
        )


def main():
    # Fix all seeds
    set_seed_all(42)
    init_logger("experiment.log")
    LOGGER.setLevel(logging.DEBUG)
    train(
        real_dir="/media/jamestiotio/Data/LJ_Speech",
        fake_dir="/media/jamestiotio/Data/ljspeech_melgan_large",
        amount_to_use=None,
        epochs=10,
        batch_size=128,
        model_dir="saved",
    )


if __name__ == "__main__":
    main()