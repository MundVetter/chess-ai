import lightning as pl
import torch
from train_old import ChessMLP
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # load model from lightning checkpoint

    model = ChessMLP.load_from_checkpoint("lightning_logs/version_40/checkpoints/epoch=4-step=11505.ckpt")
    # plot the weights  of fc layers
    # in one figure 
# Iterate over each layer
    for i, layer in enumerate(model.parameters()):
        # Only consider fully connected layers (i.e., weight layers)
        if len(layer.size()) == 2:
            # Convert to numpy for easier handling
            weight_matrix = layer.detach().cpu().numpy()

            # Iterate over each neuron in the layer
            for j, neuron_weights in enumerate(weight_matrix):
                # Plot neuron's weights
                plt.figure(figsize=(10, 5))
                plt.plot(neuron_weights)
                plt.title(f'Weights of Layer {i+1}, Neuron {j+1}')
                plt.xlabel('Weight Index')
                plt.ylabel('Weight Value')
                plt.show()
    x = 0