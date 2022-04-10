import logging
from copy import deepcopy
from typing import Callable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

LOGGER = logging.getLogger(__name__)


class Trainer(object):
    """This is a lightweight wrapper for training models with gradient descent.

    Its main function is to store information about the training process.

    Args:
        epochs (int): The amount of training epochs.
        batch_size (int): Amount of audio files to use in one batch.
        device (str): The device to train on (Default 'cpu').
        batch_size (int): The amount of audio files to consider in one batch (Default: 32).
        optimizer_fn (Callable): Function for constructing the optimzer.
        optimizer_kwargs (dict): Kwargs for the optimzer.
    """

    def __init__(
        self,
        epochs: int = 20,
        batch_size: int = 32,
        device: str = "cpu",
        optimizer_fn: Callable = torch.optim.Adam,
        optimizer_kwargs: dict = {"lr": 1e-3},
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.optimizer_fn = optimizer_fn
        self.optimizer_kwargs = optimizer_kwargs
        self.epoch_test_losses: List[float] = []


class GDTrainer(Trainer):
    def train(
        self,
        dataset: torch.utils.data.Dataset,
        model: torch.nn.Module,
        test_len: float,
        pos_weight: Optional[torch.FloatTensor] = None,
    ):

        test_len = int(len(dataset) * test_len)
        train_len = len(dataset) - test_len
        lengths = [train_len, test_len]
        train, test = torch.utils.data.random_split(dataset, lengths)

        train_loader = DataLoader(
            train, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        test_loader = DataLoader(test, batch_size=self.batch_size, drop_last=True)

        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optim = self.optimizer_fn(model.parameters(), **self.optimizer_kwargs)

        best_model = None
        best_acc = 0
        for epoch in range(self.epochs):
            running_loss = 0
            num_correct = 0.0
            num_total = 0.0
            model.train()

            for i, (batch_x, _, batch_y) in enumerate(train_loader):
                batch_size = batch_x.size(0)
                num_total += batch_size

                batch_x = batch_x.to(self.device)
                batch_y = batch_y.unsqueeze(1).type(torch.float32).to(self.device)

                batch_out = model(batch_x)
                batch_loss = criterion(batch_out, batch_y)

                batch_pred = (torch.sigmoid(batch_out) + 0.5).int()
                num_correct += (batch_pred == batch_y.int()).sum(dim=0).item()

                running_loss += batch_loss.item() * batch_size

                optim.zero_grad()
                batch_loss.backward()
                optim.step()

            running_loss /= num_total
            train_accuracy = (num_correct / num_total) * 100

            num_correct = 0.0
            num_total = 0.0
            model.eval()
            for batch_x, _, batch_y in test_loader:

                batch_size = batch_x.size(0)
                num_total += batch_size
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.unsqueeze(1).type(torch.float32).to(self.device)
                batch_out = model(batch_x)

                batch_pred = (torch.sigmoid(batch_out) + 0.5).int()
                num_correct += (batch_pred == batch_y.int()).sum(dim=0).item()

            test_acc = 100 * (num_correct / num_total)

            if best_model is None or test_acc > best_acc:
                best_acc = test_acc
                best_model = deepcopy(model.state_dict())

            LOGGER.info(
                f"[{epoch:04d}]: {running_loss} - train acc: {train_accuracy} - test_acc: {test_acc}"
            )

        model.load_state_dict(best_model)
        return model
