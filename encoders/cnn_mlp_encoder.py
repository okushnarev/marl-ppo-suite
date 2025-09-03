import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Any, Type


class CNNMLPEncoder(nn.Module):
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
            cnn_layer_configs: List[Dict[str, Any]] | Dict[str, List[Any]],
            mlp_hidden_dims: List[int],
            activation_fn: Type[nn.Module] = nn.ReLU
    ):
        """
        Initializes the flexible encoder network.

        Args:
            input_dim (Tuple[int, int]): The shape of the input observations,
                                           as (num_input_features, sequence_length).
            output_dim (int): The final output dimension of the network.
            cnn_layer_configs (List[Dict[str, Any]]): A list of dictionaries, where each
                                                      dictionary configures one Conv1d layer.
                                                      Example: [{'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 'same'}, ...]
            mlp_hidden_dims (List[int]): A list of hidden dimensions for the MLP block.
                                         The final output layer is added automatically.
            activation_fn (Type[nn.Module]): The activation function to use after each
                                             convolutional and hidden linear layer.
        """
        super().__init__()
        
        # Check for proper CNN layer configs
        if isinstance(cnn_layer_configs, dict):
            k, v = zip(*cnn_layer_configs.items())
            layer_count = len(cnn_layer_configs[k[0]])
            _cfg = []
            for idx in range(layer_count):
                _cfg.append(dict(zip(k, [val[idx] for val in v])))
            cnn_layer_configs = _cfg

        #  Build CNN Block 
        cnn_layers = []
        current_channels = 1
        for config in cnn_layer_configs:
            cnn_layers.append(
                nn.Conv1d(in_channels=current_channels, **config)
            )
            cnn_layers.append(activation_fn())
            current_channels = config['out_channels']

        self.cnn_block = nn.Sequential(*cnn_layers)

        #  Automatically determine the input size for the MLP block
        mlp_input_dim = self._get_cnn_output_dim(input_dim)

        #  Build MLP Block
        mlp_layers = []
        current_features = mlp_input_dim

        for hidden_dim in mlp_hidden_dims:
            mlp_layers.append(
                nn.Linear(current_features, hidden_dim)
            )
            mlp_layers.append(activation_fn())
            current_features = hidden_dim

        # Add the final output layer. Note: No activation is applied here,
        # as it typically depends on the downstream task (e.g., logits for a policy,
        # or a raw value for a critic).
        mlp_layers.append(nn.Linear(current_features, output_dim))

        self.mlp_block = nn.Sequential(*mlp_layers)

    def _get_cnn_output_dim(self, input_dim: int) -> int:
        """
        Calculates the flattened output dimension of the CNN block by performing
        a single forward pass with a dummy tensor.
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_dim)
            output = self.cnn_block(dummy_input)
            return output.flatten(1).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_input_features, sequence_length).

        Returns:
            torch.Tensor: The encoded output tensor of shape (batch_size, output_dim).
        """
        # 1. Pass through CNN layers for feature extraction
        cnn_out = self.cnn_block(x)

        # 2. Flatten the output for the MLP
        flattened = cnn_out.flatten(start_dim=1)

        # 3. Pass through MLP layers for final processing
        output = self.mlp_block(flattened)

        return output

