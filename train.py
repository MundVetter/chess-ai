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

class ChessMLP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(65, 8)
        self.fc = nn.ModuleList([nn.Linear(32 * 8 + 5, 32), nn.Linear(32, 32), nn.Linear(32, 32), nn.Linear(32, 1)])
        # self.bn = nn.ModuleList([nn.BatchNorm1d(32) for _ in range(3)])  # Define BatchNorm layers for the hidden layers
        # self.fc = nn.ModuleList([nn.Linear(64 * 4 + 5, 512), nn.Linear(512, 512), nn.Linear(512, 512), nn.Linear(512, 1)])

    def forward(self, x):
        info = x[:, -5:]
        if self.training:
            # create dropout mask for one value
            mask = torch.bernoulli(torch.full((x.shape[0], 1), 0.95)).cuda()
            # apply on turn
            info[:, 0] = info[:, 0] * mask.squeeze(1)

        board = x[:, :-5].reshape(-1, 32)

        # use an embedding layer for the board
        board = self.embedding(board)

        # flatten
        board = board.reshape(x.shape[0], -1)
        # concat info
        x = torch.cat((board, info), dim=1)

        for i in range(len(self.fc) - 1):
            x = self.fc[i](x)
            # x = self.bn[i](x)  # Apply BatchNorm
            x = F.relu(x)
        x = F.sigmoid(self.fc[-1](x))
        return x * 2 - 1

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

class PieceTracker:
    def __init__(self):
        self.piece_order = [
            chess.Piece.from_symbol('P'),  # white pawns
            chess.Piece.from_symbol('N'),  # white knights
            chess.Piece.from_symbol('B'),  # white bishops
            chess.Piece.from_symbol('R'),  # white rooks
            chess.Piece.from_symbol('Q'),  # white queen
            chess.Piece.from_symbol('K'),  # white king
            chess.Piece.from_symbol('p'),  # black pawns
            chess.Piece.from_symbol('n'),  # black knights
            chess.Piece.from_symbol('b'),  # black bishops
            chess.Piece.from_symbol('r'),  # black rooks
            chess.Piece.from_symbol('q'),  # black queen
            chess.Piece.from_symbol('k')   # black king
        ]

        self.piece_counts = {piece: 0 for piece in self.piece_order}

        self.piece_max_counts = {
            chess.Piece.from_symbol('P'): 8, chess.Piece.from_symbol('N'): 2,
            chess.Piece.from_symbol('B'): 2, chess.Piece.from_symbol('R'): 2,
            chess.Piece.from_symbol('Q'): 1, chess.Piece.from_symbol('K'): 1,
            chess.Piece.from_symbol('p'): 8, chess.Piece.from_symbol('n'): 2,
            chess.Piece.from_symbol('b'): 2, chess.Piece.from_symbol('r'): 2,
            chess.Piece.from_symbol('q'): 1, chess.Piece.from_symbol('k'): 1
        }

    def update_count(self, piece):
        if piece in self.piece_counts:
            self.piece_counts[piece] += 1

    def get_position(self, piece):
        piece_index = self.piece_order.index(piece)
        position = sum(self.piece_max_counts[p] for p in self.piece_order[:piece_index]) + self.piece_counts[piece]
        return position - 1  # -1 since indexing starts from 0


def transform(fen):
    board = chess.Board(fen)
    tracker = PieceTracker()
    board_state_copy = np.full(32 + 5, 64, dtype=int)
    
    for i in range(64):
        piece = board.piece_at(i)
        if piece is not None:
            if tracker.piece_counts[piece] < tracker.piece_max_counts[piece]:
                tracker.update_count(piece)
                position = tracker.get_position(piece)
                board_state_copy[position] = i

    board_state_copy[32] = int(board.turn)
    board_state_copy[32 + 1] = int(board.has_kingside_castling_rights(chess.WHITE))
    board_state_copy[32 + 2] = int(board.has_queenside_castling_rights(chess.WHITE))
    board_state_copy[32 + 3] = int(board.has_kingside_castling_rights(chess.BLACK))
    board_state_copy[32 + 4] = int(board.has_queenside_castling_rights(chess.BLACK))

    return board_state_copy
    
class ChessDataset(Dataset):
    def __init__(self, csv_file, transform, validation=False):
        data = pd.read_parquet(csv_file)
        # self.data["Evaluation"] = pd.to_numeric(self.data["Evaluation"], errors="coerce").dropna()
        # self.data["Evaluation"] = self.data["Evaluation"] / SIGMA2 # 2 sigma

        # self.data["Evaluation"] = self.data["Evaluation"].clip(-1, 1)

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
    train_dataloader = data_utils.DataLoader(ChessDataset("data/chess_evals", transform), batch_size=4096, shuffle=True)
    # val_dataset = data_utils.TensorDataset(test_x, test_y)
    val_dataloader = data_utils.DataLoader(ChessDataset("data/chess_evals", transform, True), batch_size=4096)
    
    model = ChessMLP()
    trainer = pl.Trainer(accelerator="gpu", devices = 1,  max_epochs=4, max_steps = 4_000_000, check_val_every_n_epoch = None, val_check_interval = 500)  # set number of epochs
    trainer.fit(model, train_dataloader, val_dataloader)
