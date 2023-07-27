import torch
from torch import nn
import torch.nn.functional as F
import lightning as pl

import numpy as np
import torch.utils.data as data_utils

class ChessMLP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(6, 16, 1), nn.Conv1d(16, 32, 1), nn.Conv1d(32, 4, 1)])
        self.fc = nn.ModuleList([nn.Linear(64 * 4 + 5, 32), nn.Linear(32, 32), nn.Linear(32, 32), nn.Linear(32, 3)])

    def forward(self, x):
        info = x[:, -5:]
        board = x[:, :-5].reshape(-1, 6, 64)
        # apply conv1d to each cell
        for i in range(len(self.convs) - 1):
            board = F.relu(self.convs[i](board))
        board = self.convs[-1](board)
        # flatten
        board = board.reshape(x.shape[0], -1)
        # concat info
        x = torch.cat((board, info), dim=1)

        for i in range(len(self.fc) - 1):
            x = F.relu(self.fc[i](x))
        x = self.fc[-1](x)
        return x

    def training_step(self, batch, batch_idx):
        # clip weight size to -1, 1
        for param in self.parameters():
            param.data.clamp_(-1, 1)

        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        
        # acc
        preds = torch.argmax(y_hat, dim=1)
        acc = torch.sum(preds == y).float() / len(preds)
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer

if __name__ == "__main__":
    train = np.load("data/train.npy")
    test = np.load("data/test.npy")
    # last value is target
    train_x, train_y = torch.tensor(train[:, :-1]).float(), torch.tensor(train[:, -1]).int().add(1)
    test_x, test_y = torch.tensor(test[:, :-1]).float(), torch.tensor(test[:, -1]).int().add(1)

    # print distribution of labels using torch.unique
    # print("train_y", torch.unique(train_y, return_counts=True))
    dist = torch.unique(test_y, return_counts=True)
    for i in range(len(dist)):
        print("test dist", dist[i] / len(test_y))

    # convert to dataloaders
    train_dataset = data_utils.TensorDataset(train_x, train_y)
    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=2048, shuffle=True)
    val_dataset = data_utils.TensorDataset(test_x, test_y)
    val_dataloader = data_utils.DataLoader(val_dataset, batch_size=2048, shuffle=False)
    
    model = ChessMLP()
    trainer = pl.Trainer(accelerator="mps", devices = 1,  max_epochs=50)  # set number of epochs
    trainer.fit(model, train_dataloader, val_dataloader)
