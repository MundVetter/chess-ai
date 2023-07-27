import torch
from torch import nn
import torch.nn.functional as F
import lightning as pl

import numpy as np
import torch.utils.data as data_utils

import pandas as pd
from torch.utils.data import Dataset, DataLoader
import chess

class ChessMLP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(6, 16, 1), nn.Conv1d(16, 32, 1), nn.Conv1d(32, 4, 1)])
        self.fc = nn.ModuleList([nn.Linear(64 * 4 + 5, 32), nn.Linear(32, 32), nn.Linear(32, 32), nn.Linear(32, 1)])

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
        x = F.softmax(self.fc[-1](x))
        return x * 2 - 1

    def training_step(self, batch, batch_idx):
        # clip weight size to -1, 1
        for param in self.parameters():
            param.data.clamp_(-1, 1)

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
        
        # acc
        preds = torch.argmax(y_hat, dim=1)
        acc = torch.sum(preds == y).float() / len(preds)
        # if batch_idx == 0:
        #     self.log(f"position eval", y_hat)
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer

def bitboard_to_array(bb: int) -> np.ndarray:
    s = 8 * np.arange(7, -1, -1, dtype=np.uint64)
    b = (bb >> s).astype(np.uint8)
    b = np.unpackbits(b, bitorder="little")
    return b.reshape(64)

def transform(fen):
    board = chess.Board(fen)
    board_state_copy = np.zeros(6*64 + 5, dtype=float)
    for j in range(6):
            board_state_copy[j*64:(j+1)*64] += bitboard_to_array(int(board.pieces(chess.Piece.from_symbol('PNBRQK'[j]).piece_type, chess.WHITE)))
            board_state_copy[j*64:(j+1)*64] -= bitboard_to_array(int(board.pieces(chess.Piece.from_symbol('PNBRQK'[j]).piece_type, chess.BLACK)))

    board_state_copy[6*64] = float(board.turn)
    board_state_copy[6*64 + 1] = float(board.has_kingside_castling_rights(chess.WHITE))
    board_state_copy[6*64 + 2] = float(board.has_queenside_castling_rights(chess.WHITE))
    board_state_copy[6*64 + 3] = float(board.has_kingside_castling_rights(chess.BLACK))
    board_state_copy[6*64 + 4] = float(board.has_queenside_castling_rights(chess.BLACK))

    return board_state_copy
    
class ChessDataset(Dataset):
    def __init__(self, csv_file, transform, validation=False):
        self.data = pd.read_csv(csv_file)
        self.data["Evaluation"] = pd.to_numeric(self.data["Evaluation"], errors="coerce").dropna()
        self.data["Evaluation"] = self.data["Evaluation"] / 1627 # 2 sigma
        # cap values at -5 and 5
        self.data["Evaluation"] = self.data["Evaluation"].clip(-1, 1)

        self.transform = transform
        self.validation = validation

    def __len__(self):
        if self.validation:
            return 4096
        return len(self.data)

    def __getitem__(self, idx):
        fen = self.data.iloc[idx, 0]  # Assuming FEN is the first column
        evaluation = self.data.iloc[idx, 1]  # Assuming Evaluation is the second column

        return np.float32(self.transform(fen)), evaluation
    
    def get_game(self, idx):
        return self.data.iloc[idx, 0]

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
    train_dataloader = data_utils.DataLoader(ChessDataset("data/chessData.csv", transform), batch_size=2048, shuffle=True)
    # val_dataset = data_utils.TensorDataset(test_x, test_y)
    val_dataloader = data_utils.DataLoader(ChessDataset("data/chessData.csv", transform, True), batch_size=2048)
    
    model = ChessMLP()
    trainer = pl.Trainer(accelerator="gpu", devices = 1,  max_epochs=1, max_steps = 1_000_000, check_val_every_n_epoch = None, val_check_interval = 500)  # set number of epochs
    trainer.fit(model, train_dataloader, val_dataloader)
