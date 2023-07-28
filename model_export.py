import lightning as pl
from train import ChessMLP, ChessDataset, transform
import torch.utils.data as data_utils
import torch
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
    model = ChessMLP.load_from_checkpoint("lightning_logs/version_27/checkpoints/epoch=0-step=6000.ckpt")
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
    
    trainer = pl.Trainer(accelerator="gpu", devices = 1)
    val_dataloader = data_utils.DataLoader(ChessDataset("data/chess_evals", transform, True), batch_size=2048)
    trainer.validate(model_zp, val_dataloader)

    # save quants
    torch.save(quants, "data/quants.pt")