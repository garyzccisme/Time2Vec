import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import train_test_split


class Model:
    def __init__(self, network, loss=nn.MSELoss(), max_epoch=1000):
        self.network = network
        self.loss = loss
        self.optimizer = torch.optim.Adam(network.parameters(), lr=1e-2)
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=100, verbose=True, min_lr=1e-3)
        self.max_epoch = max_epoch
        self.dataloader = None
        self.train_dataset = None
        self.valid_dataset = None

        self.train_loss = []
        self.valid_loss = []
        self.best_params = None
        self.best_loss = float('inf')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

    def get_tensor(self, array):
        if isinstance(array, tuple) or isinstance(array, list):
            return tuple(torch.as_tensor(a, dtype=torch.float32, device=self.device) for a in array)
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def generate_dataloader(self, x, y):
        if isinstance(x, tuple):
            train_set, valid_set = train_test_split(*x, y, test_size=0.2)
        else:
            train_set, valid_set = train_test_split(x, y, test_size=0.2)
        self.train_dataset = TensorDataset(*train_set)
        self.valid_dataset = TensorDataset(*valid_set)
        self.dataloader = DataLoader(self.train_dataset, batch_size=1024, shuffle=False)

    def train(self, *x, y):
        """
        Note: the X input is marked as *x, which allows multiple x input like (x1, x2, x3).
            When calling this function, must assert the `y` parameter: model.train(..., y=y).
        """
        x = self.get_tensor(x)
        y = self.get_tensor(y)
        self.generate_dataloader(x, y)
        for epoch in range(self.max_epoch):
            self.batch_train()

            train_loss = self.get_loss(self.train_dataset)
            valid_loss = self.get_loss(self.valid_dataset)
            self.train_loss.append(train_loss.item())
            self.valid_loss.append(valid_loss.item())
            self.scheduler.step(valid_loss)

            if valid_loss.item() < self.best_loss:
                self.best_params = copy.deepcopy(self.network.state_dict())
                self.best_loss = valid_loss.item()

            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Train Loss: {train_loss.item()}, Valid Loss: {valid_loss.item()}")

        self.network.load_state_dict(self.best_params)
        train_loss = self.get_loss(self.train_dataset)
        valid_loss = self.get_loss(self.valid_dataset)
        print(f"Best Train Loss: {train_loss.item()}, Best Valid Loss: {valid_loss.item()}")

    def get_loss(self, dataset):
        data = dataset[:]
        if len(data) > 2:
            x, y = data[:-1], data[-1]
        else:
            x, y = data
        pred = self.predict(x)
        loss = self.loss(pred, y)
        return loss

    def batch_train(self):
        for batch in self.dataloader:
            if len(batch) > 2:
                x, y = batch[:-1], batch[-1]
            else:
                x, y = batch

            self.optimizer.zero_grad()
            self.network.train()
            pred = self.network(x)
            loss = self.loss(pred, y)
            loss.backward()
            self.optimizer.step()

    def predict(self, x):
        self.network.eval()
        return self.network(self.get_tensor(x))
