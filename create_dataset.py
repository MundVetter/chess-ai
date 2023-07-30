import pandas as pd
import numpy as np
from train import transform # returns 37 long numpy array
SIGMA2 = 1627

if __name__ == "__main__":
    data = pd.read_csv("data/chessData.csv")
    # transform #n to 1 and #-n to -1
    data.replace(r'#\+\d+', 100_000, regex=True, inplace=True)
    data.replace(r'#-\d+', -100_000, regex=True, inplace=True)

    data["Evaluation"] = pd.to_numeric(data["Evaluation"], errors="coerce").dropna()

    # drop all remaining rows with NaN
    data = data.dropna()

    data["Evaluation"] = data["Evaluation"] / SIGMA2
    data["Evaluation"] = data["Evaluation"].clip(-1, 1)



    # apply transform on FEN
    data = data['FEN'].apply(transform)
    # transform to numpy array
    data = data.to_numpy()
    # save
    np.save("data/chess_evals.npy", data)



