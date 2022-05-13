import logging
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from metrics import alt_compute_eer
from utils import save_checkpoint, save_pred, set_learning_rate

LOGGER = logging.getLogger(__name__)


class Trainer(object):
    """This is a lightweight wrapper for training models with gradient descent.

    Its main function is to store information about the training process.

    Args:
        epochs (int): The amount of training epochs.
        batch_size (int): The batch size for training.
        device (str): The device to train on.
        optimizer_fn (Callable): Function for constructing the optimzer (Default: Adam).
        optimizer_kwargs (dict): Kwargs for the optimzer.
    """

    def __init__(
        self,
        epochs: int,
        batch_size: int,
        device: str,
        lr: float = 1e-3,
        optimizer_fn: Callable = torch.optim.Adam,
        optimizer_kwargs: Optional[dict] = {},
    ) -> None:
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.device = device
        self.lr = lr
        self.optimizer_fn = optimizer_fn
        self.optimizer_kwargs = optimizer_kwargs

        assert self.epochs > 0
        assert self.batch_size > 0
        assert isinstance(optimizer_fn, Callable)
        assert isinstance(optimizer_kwargs, dict)
        self.optimizer_kwargs["lr"] = self.lr


class ModelTrainer(Trainer):
    """A model trainer for binary classification"""

    def train(
        self,
        model: nn.Module,
        dataset_train: Dataset,
        dataset_test: Dataset,  # test or validation
        save_dir: Union[str, Path] = None,  # directory to save model checkpoints
        pos_weight: Optional[torch.FloatTensor] = None,
        checkpoint: dict = None,
    ) -> None:
        if save_dir:
            save_dir: Path = Path(save_dir)
            if not save_dir.exists():
                save_dir.mkdir(parents=True)

        train_loader = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        test_loader = DataLoader(
            dataset_test,
            batch_size=self.batch_size,
            drop_last=False,
        )

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optim = self.optimizer_fn(model.parameters(), **self.optimizer_kwargs)
        start_epoch = 0

        #######################################################################

        if checkpoint is not None:
            model.load_state_dict(checkpoint["state_dict"])
            optim.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"] + 1
            LOGGER.info(f"Loaded checkpoint from epoch {start_epoch - 1}")

        set_learning_rate(self.lr, optim)

        #######################################################################

        best_model = None
        best_acc = 0
        for epoch in range(start_epoch, self.epochs):
            ###################################################################
            # train
            model.train()
            total_loss = 0
            num_correct = 0.0
            num_total = 0.0

            for _, (batch_x, _, _, batch_y) in enumerate(train_loader):
                # get actual batch size
                curr_batch_size = batch_x.size(0)
                num_total += curr_batch_size
                # get batch input x
                batch_x = batch_x.to(self.device)
                # make batch label y a vector
                batch_y = batch_y.unsqueeze(1).type(torch.float32).to(self.device)
                # forward
                batch_out = model(batch_x)  # (B, 1)
                # compute loss
                batch_loss = criterion(batch_out, batch_y)  # (1, )
                # get binary prediction {0, 1}
                batch_pred = (torch.sigmoid(batch_out) + 0.5).int()
                # count number of correct predictions
                num_correct += (batch_pred == batch_y.int()).sum(dim=0).item()
                # accumulate loss
                total_loss += batch_loss.item() * curr_batch_size
                # backwards
                optim.zero_grad()  # reset gradient
                batch_loss.backward()  # compute gradient
                optim.step()  # update params

            # get loss for this epoch
            total_loss /= num_total
            # get training accuracy for this epoch
            train_acc = (num_correct / num_total) * 100

            ###################################################################
            # evaluation
            model.eval()
            num_correct = 0.0
            num_total = 0.0
            # save test label and predictions
            y_true = []
            y_pred = []

            for batch_x, _, _, batch_y in test_loader:
                # get actual batch size
                curr_batch_size = batch_x.size(0)
                num_total += curr_batch_size
                # get batch input x
                batch_x = batch_x.to(self.device)
                # make batch label y a vector
                batch_y = batch_y.unsqueeze(1).type(torch.float32).to(self.device)
                y_true.append(batch_y.clone().detach().int().cpu())
                # forward / inference
                batch_out = model(batch_x)
                # get binary prediction {0, 1}
                batch_pred = (torch.sigmoid(batch_out) + 0.5).int()
                y_pred.append(batch_pred.clone().detach().cpu())
                # count number of correct predictions
                num_correct += (batch_pred == batch_y.int()).sum(dim=0).item()

            # get test accuracy
            test_acc = (num_correct / num_total) * 100
            # get all labels and predictions
            y_true: np.ndarray = torch.cat(y_true, dim=0).numpy()
            y_pred: np.ndarray = torch.cat(y_pred, dim=0).numpy()
            # get auc and eer
            test_eer = alt_compute_eer(y_true, y_pred)

            LOGGER.info(
                f"[{epoch:03d}]: loss: {round(total_loss, 4)} - train acc: {round(train_acc, 2)} - test acc: {round(test_acc, 2)} - test eer : {round(test_eer, 4)}"
            )

            if test_acc > best_acc:
                best_acc = test_acc
                LOGGER.info(f"Best Test Accuracy: {round(best_acc, 3)}")

                if save_dir:
                    # save model checkpoint
                    save_path = save_dir / "best.pt"
                    save_checkpoint(
                        epoch=epoch,
                        model=model,
                        optimizer=optim,
                        model_kwargs=self.__dict__,
                        filename=save_path,
                    )
                    LOGGER.info(f"Best Model Saved: {save_path}")
                    # save labels and predictions
                    save_path = save_dir / "best_pred.json"
                    save_pred(y_true, y_pred, save_path)
                    LOGGER.info(f"Prediction Saved: {save_path}")

        return None

    def eval(
        self,
        model: nn.Module,
        dataset_test: Dataset,
        save_dir: Union[str, Path] = None,  # directory to save model predictions
        checkpoint: dict = None,
    ) -> None:
        if save_dir:
            save_dir: Path = Path(save_dir)
            if not save_dir.exists():
                save_dir.mkdir(parents=True)

        test_loader = DataLoader(
            dataset_test,
            batch_size=self.batch_size,
            drop_last=False,
        )
        #######################################################################

        if checkpoint is not None:
            model.load_state_dict(checkpoint["state_dict"])
            # optim.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"] + 1
            LOGGER.info(f"Loaded checkpoint from epoch {start_epoch - 1}")

        ###################################################################
        # evaluation
        model.eval()
        num_correct = 0.0
        num_total = 0.0
        # save test label and predictions
        y_true = []
        y_pred = []

        for batch_x, _, _, batch_y in test_loader:
            # get actual batch size
            curr_batch_size = batch_x.size(0)
            num_total += curr_batch_size
            # get batch input x
            batch_x = batch_x.to(self.device)
            # make batch label y a vector
            batch_y = batch_y.unsqueeze(1).type(torch.float32).to(self.device)
            y_true.append(batch_y.clone().detach().int().cpu())
            # forward / inference
            batch_out = model(batch_x)
            # get binary prediction {0, 1}
            batch_pred = (torch.sigmoid(batch_out) + 0.5).int()
            y_pred.append(batch_pred.clone().detach().cpu())
            # count number of correct predictions
            num_correct += (batch_pred == batch_y.int()).sum(dim=0).item()

        # get test accuracy
        test_acc = (num_correct / num_total) * 100
        # get all labels and predictions
        y_true: np.ndarray = torch.cat(y_true, dim=0).numpy()
        y_pred: np.ndarray = torch.cat(y_pred, dim=0).numpy()
        # get auc and eer
        test_eer = alt_compute_eer(y_true, y_pred)

        LOGGER.info(f"test acc: {round(test_acc, 2)} - test eer : {round(test_eer, 4)}")

        if save_dir:
            # save labels and predictions
            save_path = save_dir / "best_pred.json"
            save_pred(y_true, y_pred, save_path)
            LOGGER.info(f"Prediction Saved: {save_path}")
