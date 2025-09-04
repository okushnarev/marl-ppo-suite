from typing import List, Tuple, Type

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """
    A flexible encoder that combines a configurable CNN block for feature extraction
    from sequence data, followed by a configurable MLP block for further processing.

    The model automatically calculates the intermediate feature dimensions
    between the CNN and MLP blocks.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            mlp_layer_configs: List[int],
            activation_fn: Type[nn.Module] = nn.ReLU
    ):
        """
        Initializes the flexible encoder network.

        Args:
            input_dim (Tuple[int, int]): The shape of the input observations,
                                           as (num_input_features, sequence_length).
            output_dim (int): The final output dimension of the network.
            mlp_layer_configs (List[int]): A list of hidden dimensions for the MLP block.
                                         The final output layer is added automatically.
            activation_fn (Type[nn.Module]): The activation function to use after each
                                             convolutional and hidden linear layer.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mlp_layer_configs = mlp_layer_configs
        self.activation_fn = activation_fn

        #  Build MLP Block
        mlp_layers = []
        current_features = input_dim
        for hidden_dim in self.mlp_layer_configs:
            mlp_layers += [
                nn.Linear(current_features, self.hidden_size),
                self.activation_fn(),
                nn.LayerNorm(self.hidden_size),
            ]
            current_features = hidden_dim

        # Add the final output layer. Note: No activation is applied here,
        # as it typically depends on the downstream task (e.g., logits for a policy,
        # or a raw value for a critic).
        mlp_layers.append(nn.Linear(current_features, self.output_dim))

        self.mlp_block = nn.Sequential(*mlp_layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_input_features, sequence_length).

        Returns:
            torch.Tensor: The encoded output tensor of shape (batch_size, output_dim).
        """
        output = self.mlp_block(x)
        return output

