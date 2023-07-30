import lightning as pl
from train import ChessMLP, ChessDataset, transform
import torch.utils.data as data_utils
import torch
import chess
from copy import deepcopy

def zeropoint_quantize(X):
    # Calculate value range (denominator)
    x_range = torch.max(X) - torch.min(X)
    x_range = 1 if x_range == 0 else x_range

    # Calculate scale
    scale = 255 / x_range

    # Shift by zero-point
    zeropoint = (-scale * torch.min(X) - 128).round()

    # Scale and round the inputs
    X_quant = torch.clip((X * scale + zeropoint).round(), -128, 127)

    # Dequantize

    return X_quant.to(torch.int8), zeropoint, scale

def dequant(x, zeropoint, scale):
    return (x - zeropoint) / scale

if __name__ == "__main__":
    FENs = ["rnbqkbnr/1pp1pppp/p7/3p4/3P4/3Q4/PPP1PPPP/RNB1KBNR w KQkq - 0 3", "rnbqkbnr/1pp1pppp/p7/3p4/3P4/7Q/PPP1PPPP/RNB1KBNR b KQkq - 1 3", "rn1qkbnr/1pp1pppp/p7/3p4/3P4/7b/PPP1PPPP/RNB1KBNR w KQkq - 0 4", "rn1qkbnr/1pp1pppp/p7/3p4/3P4/7P/PPP1PP1P/RNB1KBNR b KQkq - 0 4", "r2qkbnr/1pp1pppp/p1n5/3p4/3P4/7P/PPP1PP1P/RNB1KBNR w KQkq - 1 5", "r2qkbnr/1pp1pppp/p1n5/3p4/3P4/P6P/1PP1PP1P/RNB1KBNR b KQkq - 0 5", "r2qkbnr/1pp1pppp/p1n5/3p4/3P4/P6P/1PP1PP1P/RNB1KBNR w KQkq - 0 5"]
    # FENs = []

    model = ChessMLP.load_from_checkpoint("lightning_logs/version_58/checkpoints/epoch=0-step=3000.ckpt")
    model.eval()
    model.freeze()
    quants = []
    model_zp = deepcopy(model)
    embedding_quant = zeropoint_quantize(model_zp.embedding.weight.data)
    model_zp.embedding.weight.data = dequant(embedding_quant[0], embedding_quant[1], embedding_quant[2])
    quants.append(embedding_quant)
    for param in model_zp.fc.parameters():
        quant = zeropoint_quantize(param.data)
        quants.append(quant)
        param.data = dequant(quant[0], quant[1], quant[2])
    
    if len(FENs) == 0:
        trainer = pl.Trainer(accelerator="gpu", devices = 1)
        val_dataloader = data_utils.DataLoader(ChessDataset("data/chess_evals", transform, True), batch_size=2048)
        trainer.validate(model_zp, val_dataloader)
    else:
        SIGMA2 = 1627
        for FEN in FENs:
            input = torch.tensor(transform(FEN)).unsqueeze(0).cuda()
            board = chess.Board(FEN)
            output_quantized = model_zp(input) * SIGMA2
            output_original = model(input) * SIGMA2
            print("output_quantized", output_quantized)
            print("output_original", output_original)
            print("diff", output_quantized - output_original)
            print(board)
            print("")

    # save quants
    torch.save(quants, "data/quants.pt")