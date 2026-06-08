"""
Artificial Neural Network (ANN) architecture for cover crop biomass prediction.

Architecture
------------
Input Layer
    ↓
Hidden Layer (variable size)
    ↓
ReLU
    ↓
Dropout
    ↓
Hidden Layer (32 neurons)
    ↓
ReLU
    ↓
Output Layer (1 neuron)
"""

import torch.nn as nn


class ANNModel(nn.Module):
    """
    Feed-forward neural network for biomass estimation.

    Parameters
    ----------
    input_size : int
        Number of input features.

    hidden_size : int
        Number of neurons in the first hidden layer.

    dropout_rate : float
        Dropout probability applied after the first
        hidden layer.
    """

    def __init__(self, input_size, hidden_size, dropout_rate):
        super().__init__()

        # Network architecture
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """
        Perform a forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor.

        Returns
        -------
        torch.Tensor
            Predicted biomass values.
        """
        return self.net(x)