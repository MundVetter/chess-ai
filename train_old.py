import torch
from torch import nn
import torch.nn.functional as F
import lightning as pl

import numpy as np
import torch.utils.data as data_utils

import pandas as pd
from torch.utils.data import Dataset, DataLoader
import chess

SIGMA2 = 1627

import math
import torch
from torch import nn

class SineLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, is_first=False, w0=1.0):
        super().__init__()
        self.in_dim = in_dim
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.is_first = is_first
        self.w0 = w0

        # Compute standard deviation
        self.w_std = (1 / in_dim) if is_first else (math.sqrt(1.0 / in_dim) / w0)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights
        self.linear.weight.data.uniform_(-self.w_std, self.w_std)

    def forward(self, input):
        return torch.sin(self.w0 * self.linear(input))

class SineMLP(nn.Module):
    def __init__(self, layer_dims, w0=1.0, max_dim = None):
        super().__init__()
        layers = []
        if max_dim is None:
            # to device
            max_dim = torch.tensor([32, 32])
        else:
            self.max_dim = max_dim

        num_layers = len(layer_dims)
        for i in range(num_layers - 1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            is_first = i == 0
            layers.append(SineLayer(in_dim, out_dim, is_first=is_first, w0=w0))
            
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x / self.max_dim)

class ChessMLP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(13, 4)
        self.fc = nn.ModuleList([nn.Linear(64 * 4 + 5, 32), nn.Linear(32, 32), nn.Linear(32, 1)])
        self.SIREN = SineMLP([2, 32, 32, 1], max_dim=torch.tensor([64 * 4 + 5, 32]))
        # self.fc = nn.ModuleList([nn.Linear(64 * 4 + 5, 512), nn.Linear(512, 512), nn.Linear(512, 512), nn.Linear(512, 1)])

    def forward(self, x):
        # first generate the network    

        info = x[:, -5:]
        if self.training:
            # create dropout mask for one value
            mask = torch.bernoulli(torch.full((x.shape[0], 1), 0.95)).cuda()
            # apply on turn
            info[:, 0] = info[:, 0] * mask.squeeze(1)
        board = x[:, :-5].reshape(-1, 64)
        # if self.training:
        #     # create dropout mask for one value
        #     mask = torch.bernoulli(torch.full((x.shape[0], 1), 0.95)).cuda()
        #     # apply on turn
        #     info[:, 0] = info[:, 0] * mask.squeeze(1)
        # use an embedding layer for the board
        board = self.embedding(board)

        # flatten
        board = board.reshape(x.shape[0], -1)
        # concat info
        x = torch.cat((board, info), dim=1)

        for i in range(len(self.fc) - 1):
            x = F.relu(self.fc[i](x))
        return F.sigmoid(self.fc[-1](x)) * 2 - 1

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # use mse loss
        loss = F.mse_loss(y_hat, y.unsqueeze(1))
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y.unsqueeze(1))

        # log position eval
        if batch_idx == 0:
            for i, pred in enumerate(y_hat):
                self.log(f"position {i} eval:", pred * SIGMA2)
                if i == 10:
                    break

        #     self.log(f"position eval", y_hat)
        self.log('val_loss', loss, on_epoch=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

def bitboard_to_array(bb: int) -> np.ndarray:
    s = 8 * np.arange(7, -1, -1, dtype=np.uint64)
    b = (bb >> s).astype(np.uint8)
    b = np.unpackbits(b, bitorder="little")
    return b.reshape(64)

piece_dict = {
    None: 0,
    chess.Piece.from_symbol('P'): 1,
    chess.Piece.from_symbol('N'): 2,
    chess.Piece.from_symbol('B'): 3,
    chess.Piece.from_symbol('R'): 4,
    chess.Piece.from_symbol('Q'): 5,
    chess.Piece.from_symbol('K'): 6,
    chess.Piece.from_symbol('p'): 7,
    chess.Piece.from_symbol('n'): 8,
    chess.Piece.from_symbol('b'): 9,
    chess.Piece.from_symbol('r'): 10,
    chess.Piece.from_symbol('q'): 11,
    chess.Piece.from_symbol('k'): 12
}

def transform(fen):
    board = chess.Board(fen)
    board_state_copy = np.zeros(64 + 5, dtype=int)
    for i in range(64):
        piece = board.piece_at(i)
        board_state_copy[i] = piece_dict[piece]

    board_state_copy[64] = 1 if board.turn else -1
    board_state_copy[64 + 1] = board.has_kingside_castling_rights(chess.WHITE)
    board_state_copy[64 + 2] = board.has_queenside_castling_rights(chess.WHITE)
    board_state_copy[64 + 3] = board.has_kingside_castling_rights(chess.BLACK)
    board_state_copy[64 + 4] = board.has_queenside_castling_rights(chess.BLACK)

    return board_state_copy
    
class ChessDataset(Dataset):
    def __init__(self, csv_file, transform, validation=False):
        data = pd.read_parquet(csv_file)
        # self.data["Evaluation"] = pd.to_numeric(self.data["Evaluation"], errors="coerce").dropna()
        # self.data["Evaluation"] = self.data["Evaluation"] / SIGMA2 # 2 sigma
        # # cap values at -5 and 5
        # self.data["Evaluation"] = self.data["Evaluation"].clip(-1, 1).dropna()

        self.fens = data.iloc[:, 0].values
        self.evaluations = data.iloc[:, 1].values

        self.transform = transform
        self.validation = validation

    def __len__(self):
        if self.validation:
            return 4096
        return len(self.fens)

    def __getitem__(self, idx):
        fen = self.fens[idx]
        evaluation = self.evaluations[idx]

        return self.transform(fen), np.float32(evaluation)

if __name__ == "__main__":

    # train = np.load("data/train.npy")
    # test = np.load("data/test.npy")
    # # last value is target
    # train_x, train_y = torch.tensor(train[:, :-1]).float(), torch.tensor(train[:, -1]).int().add(1)
    # test_x, test_y = torch.tensor(test[:, :-1]).float(), torch.tensor(test[:, -1]).int().add(1)

    # # print distribution of labels using torch.unique
    # # print("train_y", torch.unique(train_y, return_counts=True))
    # dist = torch.unique(test_y, return_counts=True)
    # for i in range(len(dist)):
    #     print("test dist", dist[i] / len(test_y))

    # # convert to dataloaders
    # train_dataset = data_utils.TensorDataset(train_x, train_y)
    train_dataloader = data_utils.DataLoader(ChessDataset("data/chess_evals", transform), batch_size=2048, shuffle=True)
    # val_dataset = data_utils.TensorDataset(test_x, test_y)
    val_dataloader = data_utils.DataLoader(ChessDataset("data/chess_evals", transform, True), batch_size=2048)
    
    model = ChessMLP()
    trainer = pl.Trainer(accelerator="gpu", devices = 1,  max_epochs=3, max_steps = 1_000_000, check_val_every_n_epoch = None, val_check_interval = 500)  # set number of epochs
    trainer.fit(model, train_dataloader, val_dataloader)
