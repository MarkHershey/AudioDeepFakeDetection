import json
import logging
import warnings
from pathlib import Path
from typing import Callable, List, Optional, Union

import torch
from puts import timestamp_seconds
from torch.utils.data import ConcatDataset

from DataLoader import lfcc, load_directory_split_train_test, mfcc
from models.cnn import ShallowCNN
from models.lstm import SimpleLSTM, WaveLSTM
from models.mlp import MLP
from models.rnn import WaveRNN
from models.tssd import TSSD
from trainer import ModelTrainer
from utils import set_seed_all

warnings.filterwarnings("ignore")
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.DEBUG)


KWARGS_MAP = {
    "SimpleLSTM": {
        "lfcc": {"feat_dim": 40, "time_dim": 972, "mid_dim": 30, "out_dim": 1},
        "mfcc": {"feat_dim": 40, "time_dim": 972, "mid_dim": 30, "out_dim": 1},
    },
    "ShallowCNN": {
        "lfcc": {"in_features": 1, "out_dim": 1},
        "mfcc": {"in_features": 1, "out_dim": 1},
    },
    "MLP": {
        "lfcc": {"in_dim": 40 * 972, "out_dim": 1},
        "mfcc": {"in_dim": 40 * 972, "out_dim": 1},
    },
    "TSSD": {
        "wave": {"in_dim": 64600},
    },
    "WaveRNN": {
        "wave": {"num_frames": 10, "input_length": 64600},
    },
    "WaveLSTM": {
        "wave": {
            "num_frames": 10,
            "input_len": 64600,
            "hidden_dim": 30,
            "out_dim": 1,
        }
    },
}


def init_logger(log_file: Union[Path, str]) -> None:
    # create file handler
    fh = logging.FileHandler(log_file)
    # create console handler
    ch = logging.StreamHandler()
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # clear handlers
    LOGGER.handlers = []
    # add the handlers to the logger
    LOGGER.addHandler(fh)
    LOGGER.addHandler(ch)
    return None


def train(
    real_dir: Union[Path, str],
    fake_dir: Union[Path, str],
    amount_to_use: int = None,
    epochs: int = 20,
    device: str = "cuda" if torch.cuda.is_available else "cpu",
    batch_size: int = 32,
    save_dir: Union[str, Path] = None,
    test_size: float = 0.2,
    feature_classname: str = "wave",
    model_classname: str = "SimpleLSTM",
    in_distribution: bool = True,
) -> None:
    """
    Train a model on WaveFake data.

    Args:
        real_dir:
            path to LJSpeech dataset directory
        fake_dir:
            path to WaveFake dataset directory
        amount_to_use:
            amount of data to use (if None, use all) (default: None)
        epochs:
            number of epochs to train for (default: 20)
        device:
            device to use (default: "cuda" if available)
        batch_size:
            batch size (default: 32)
        save_dir:
            directory to save model checkpoints to (default: None)
        test_size:
            ratio of test set / whole dataset (default: 0.2)
        feature_classname:
            classname of feature extractor (possible: "wave", "mfcc", "lfcc")
        model_classname:
            classname of model (possible: "SimpleLSTM", "ShallowCNN", "WaveLSTM", "MLP")
        in_distribution:
            whether to use in-distribution data (default: True)
                - True: use 1:1 real:fake data (split melgan for training and test)
                - False: use 1:7 real:fake data (use melgan for test only, others for training)

    Returns:
        None
    """
    feature_classname = feature_classname.lower()
    assert feature_classname in ("wave", "lfcc", "mfcc")
    assert model_classname in (
        "SimpleLSTM",
        "ShallowCNN",
        "WaveLSTM",
        "MLP",
        "TSSD",
        "WaveRNN",
    )

    # get feature transformation function
    feature_fn = None if feature_classname == "wave" else eval(feature_classname)
    assert feature_fn in (None, lfcc, mfcc)
    # get model constructor
    Model = eval(model_classname)
    assert Model in (SimpleLSTM, ShallowCNN, WaveLSTM, MLP, TSSD, WaveRNN)

    model_kwargs: dict = KWARGS_MAP.get(model_classname).get(feature_classname)
    if model_kwargs is None:
        raise ValueError(
            f"model_kwargs not found for {model_classname} and {feature_classname}"
        )

    LOGGER.info(f"Training model: {model_classname}")
    LOGGER.info(f"Input feature : {feature_classname}")
    LOGGER.info(f"Model kwargs  : {json.dumps(model_kwargs, indent=2)}")

    ###########################################################################

    real_dir = Path(real_dir)
    fake_dir = Path(fake_dir)
    assert real_dir.is_dir()
    assert fake_dir.is_dir()
    melgan_dir = fake_dir / "ljspeech_melgan"
    melganLarge_dir = fake_dir / "ljspeech_melgan_large"
    assert melgan_dir.is_dir()
    assert melganLarge_dir.is_dir()

    LOGGER.info("Loading data...")

    real_dataset_train, real_dataset_test = load_directory_split_train_test(
        path=real_dir,
        feature_fn=feature_fn,
        feature_kwargs={},
        test_size=test_size,
        use_double_delta=True,
        phone_call=False,
        pad=True,
        label=1,
        amount_to_use=amount_to_use,
    )

    fake_melgan_train, fake_melgan_test = load_directory_split_train_test(
        path=melgan_dir,
        feature_fn=feature_fn,
        feature_kwargs={},
        test_size=test_size,
        use_double_delta=True,
        phone_call=False,
        pad=True,
        label=0,
        amount_to_use=amount_to_use,
    )

    dataset_train, dataset_test = None, None
    if in_distribution:
        # ljspeech (real) and melgan (fake) are split into train and test
        dataset_train = ConcatDataset([real_dataset_train, fake_melgan_train])
        dataset_test = ConcatDataset([real_dataset_test, fake_melgan_test])
        pos_weight = len(real_dataset_train) / len(fake_melgan_train)
    else:
        fake_dirs = list(fake_dir.glob("ljspeech_*"))
        assert len(fake_dirs) == 7
        # remove melgan from training data
        fake_dirs.remove(melgan_dir)
        # create datasets for each fake directory
        fake_dataset_train = list(
            map(
                lambda _dir: load_directory_split_train_test(
                    path=_dir,
                    feature_fn=feature_fn,
                    feature_kwargs={},
                    test_size=0.01,
                    use_double_delta=True,
                    phone_call=False,
                    pad=True,
                    label=0,
                    amount_to_use=amount_to_use,
                )[0],
                fake_dirs,
            )
        )
        # all fake audio (except melgan) are used for training
        fake_dataset_train = ConcatDataset(fake_dataset_train)
        pos_weight = len(real_dataset_train) / len(fake_dataset_train)
        # melgan is used for testing only
        dataset_train = ConcatDataset([real_dataset_train, fake_dataset_train])
        dataset_test = ConcatDataset([real_dataset_test, fake_melgan_test])

    ###########################################################################

    LOGGER.info(f"Training model on {len(dataset_train)} audio files.")
    LOGGER.info(f"Testing model on  {len(dataset_test)} audio files.")
    LOGGER.info(f"Train/Test ratio: {len(dataset_train) / len(dataset_test)}")
    LOGGER.info(f"Real/Fake ratio in training: {round(pos_weight, 3)} (pos_weight)")

    pos_weight = torch.Tensor([pos_weight]).to(device)

    model = Model(**model_kwargs).to(device)

    ModelTrainer(
        batch_size=batch_size,
        epochs=epochs,
        device=device,
        optimizer_kwargs={
            "lr": 0.0005,
            "weight_decay": 0.0001,
        },
    ).train(
        model=model,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        save_dir=save_dir,
        pos_weight=pos_weight,
    )


def experiment(
    name: str,
    seed: int,
    epochs: int,
    batch_size: int,
    feature_classname: str,
    model_classname: str,
    in_distribution: bool,
    real_dir="/home/markhuang/Data/WaveFake/real",
    fake_dir="/home/markhuang/Data/WaveFake/fake",
    amount_to_use=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
):

    root_save_dir = Path("saved")
    save_dir = root_save_dir / name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = save_dir / f"{timestamp_seconds()}.log"

    set_seed_all(seed)
    init_logger(log_file)

    LOGGER.info(f"Batch size: {batch_size}, seed: {seed}, epochs: {epochs}")

    train(
        real_dir=real_dir,
        fake_dir=fake_dir,
        amount_to_use=amount_to_use,
        epochs=epochs,
        device=device,
        batch_size=batch_size,
        save_dir=save_dir,
        feature_classname=feature_classname,
        model_classname=model_classname,
        in_distribution=in_distribution,
    )


def debug():
    for model_classname in KWARGS_MAP.keys():
        for feature_classname in KWARGS_MAP[model_classname].keys():
            for in_distribution in [True, False]:
                exp_setup = "I" if in_distribution else "O"
                exp_name = f"{model_classname}_{feature_classname}_{exp_setup}"
                try:
                    print(f">>>>> DEBUGGING: {exp_name}")
                    experiment(
                        name="debug",
                        seed=0,
                        epochs=3,
                        batch_size=16,
                        feature_classname=feature_classname,
                        model_classname=model_classname,
                        in_distribution=in_distribution,
                        real_dir="/home/markhh/Documents/DeepFakeAudioDetection/LJ_Speech",
                        fake_dir="/home/markhh/Documents/DeepFakeAudioDetection/WaveFake_generated_audio",
                        amount_to_use=160,
                    )
                    print(f">>>>> DEBUGGING Done: {exp_name}\n\n")
                except Exception as e:
                    print(f">>>>> DEBUGGING Failed: {exp_name}\n\n")
                    LOGGER.exception(e)


def main():
    for in_distribution in [True, False]:
        for model_classname in ["WaveRNN", "WaveLSTM"]:
            for feature_classname in ["wave"]:
                exp_setup = "I" if in_distribution else "O"
                exp_name = f"{model_classname}_{feature_classname}_{exp_setup}"
                try:
                    print(f">>>>> Starting experiment: {exp_name}")
                    experiment(
                        name=exp_name,
                        seed=42,
                        epochs=20,
                        batch_size=256,
                        feature_classname=feature_classname,
                        model_classname=model_classname,
                        in_distribution=in_distribution,
                    )
                    print(f">>>>> Experiment Done: {exp_name}\n\n")
                except Exception as e:
                    print(f">>>>> Experiment Failed: {exp_name}\n\n")
                    LOGGER.exception(e)


if __name__ == "__main__":
    # debug()
    main()
