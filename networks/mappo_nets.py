"""
Unified Code for the networks of the mappo implementation:
- working with vectorized environments
- supporting both MLP and RNN networks
"""
from argparse import Namespace

import torch
import torch.nn as nn
from gymnasium.spaces import Box, Discrete
from torch.distributions import Categorical

from cores.srmt_core import TransformerCore
from encoders.mlp_encoder import MLPEncoder
from networks.modules.rnn import GRUModule

from encoders.cnn_mlp_encoder import CNNMLPEncoder

from utils.env_tools import get_shape_from_obs_space


def _orthogonal_init(layer, gain=1.0, bias_const=0.0):
    """Enhanced orthogonal initialization with configurable gain and bias."""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, bias_const)
    elif isinstance(layer, nn.LayerNorm):
        nn.init.constant_(layer.weight, 1.0)
        nn.init.constant_(layer.bias, 0.0)


class Actor(nn.Module):
    """
    Actor network for MAPPO.
    """

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """Initialize the actor network.

        Args:
            args (argparse.Namespace): Arguments containing training hyperparameters
            obs_space (gymnasium.spaces.Box): Observation space for individual agents (Box)
            action_space (gymnasium.spaces.Discrete): Action space (Discrete)
            device (torch.device): Device to run the agent on
        """

        super(Actor, self).__init__()
        self.hidden_size = args.hidden_size
        self.use_rnn = args.use_rnn
        # self.use_layer_norm = args.use_layer_norm
        self.rnn_layers = args.rnn_layers
        self.fc_layers = args.fc_layers
        self.use_feature_normalization = args.use_feature_normalization
        self.actor_gain = args.actor_gain

        obs_shape = get_shape_from_obs_space(obs_space)
        obs_dim = obs_shape[0]

        action_type = action_space.__class__.__name__
        if action_type == "Discrete":
            self.action_dim = action_space.n
        else:
            raise NotImplementedError("Only discrete action space is supported")

        # Feature Normalization
        if self.use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        # MLP Layers
        layers = []
        in_dim = obs_dim
        for _ in range(self.fc_layers):
            layers += [
                nn.Linear(in_dim, self.hidden_size),
                nn.ReLU(),
                nn.LayerNorm(self.hidden_size),
            ]
            in_dim = self.hidden_size
        self.mlp = nn.Sequential(*layers)

        # SRMT Goes Here
        # RNN Layer (GRU)
        if self.use_rnn:
            self.rnn = GRUModule(self.hidden_size,
                                 self.hidden_size,
                                 num_layers=self.rnn_layers)

        # Output Layer
        self.output = nn.Linear(self.hidden_size, self.action_dim)

        self.apply(lambda module: _orthogonal_init(module, gain=nn.init.calculate_gain('relu')))
        _orthogonal_init(self.output, gain=self.actor_gain)

        self.to(device)

    def forward(self, x, rnn_states=None, masks=None):
        """
        Forward pass of the actor network.
        Batch size is n_agents * n_rollout_threads

        Args:
            x (torch.Tensor): Input tensor (batch_size, input_dim) or (seq_len, batch_size, input_dim)
            rnn_states (torch.Tensor, optional): RNN hidden state tensor. Required when use_rnn=True,
                                            ignored otherwise. Shape: (batch_size, num_layers, hidden_size)
            masks (torch.Tensor, optional): Mask tensor. Required when use_rnn=True, ignored otherwise.
                                       Shape: (batch_size, 1) or (seq_len, batch_size, 1)
        Returns:
            logits: action logits
            rnn_states_out: updated RNN states if use_rnn=True, None otherwise
        """
        if self.use_rnn and (rnn_states is None or masks is None):
            raise ValueError("rnn_states and masks must be provided when use_rnn=True")

        if self.use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)
        if self.use_rnn:
            x, rnn_states_out = self.rnn(x, rnn_states, masks)
        else:
            rnn_states_out = None
        logits = self.output(x)  # [seq_len, batch_size, action_dim]

        return logits, rnn_states_out

    def get_actions(self, obs, rnn_states=None, masks=None, available_actions=None, deterministic=False):
        """Get actions from the actor network.
        Batch size is n_agents * n_rollout_threads

        Args:
            obs: tensor of shape [batch_size, input_dim]
            rnn_states: tensor of shape [batch_size, num_layers, rnn_hidden_size],
                required when use_rnn=True, can be None otherwise
            masks: tensor of shape [batch_size, 1], required when use_rnn=True,
                can be None otherwise
            available_actions: tensor of shape [batch_size, action_dim]
            deterministic: bool, whether to use deterministic actions

        Returns:
            actions: tensor of shape [n_agents, 1]
            action_log_probs: tensor of shape [batch_size, 1]
            next_rnn_states: tensor of shape [batch_size, num_layers, rnn_hidden_size]
                if use_rnn=True, None otherwise
        """
        # Forward pass to get logits
        logits, rnn_states_out = self.forward(obs, rnn_states, masks)

        # Apply mask for available actions if provided
        if available_actions is not None:
            # Set unavailable actions to have a very small probability
            logits[available_actions == 0] = -1e10

        if deterministic:
            actions = torch.argmax(logits, dim=-1, keepdim=True)
            action_log_probs = None
        else:
            # Convert logits to action probabilities
            action_dist = Categorical(logits=logits)
            actions = action_dist.sample().unsqueeze(-1)  # (batch_size, 1)
            action_log_probs = action_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)  # (batch_size, 1)

        return actions, action_log_probs, rnn_states_out

    def evaluate_actions(self, obs, actions, rnn_states=None, masks=None, available_actions=None):
        """Evaluate actions for training.

        Args:
            obs_seq: tensor of shape [seq_len, batch_size, input_dim] or [batch_size, input_dim]
            actions: tensor of shape [seq_len, batch_size, 1] or [batch_size, 1]
            rnn_states: tensor of shape [batch_size, num_layers, hidden_size] - initial hidden state
                required when use_rnn=True, can be None otherwise
            masks: tensor of shape [seq_len, batch_size, 1] or [batch_size, 1]
                required when use_rnn=True, can be None otherwise
            available_actions: tensor of shape [seq_len, batch_size, action_dim] or [batch_size, action_dim]
                can be None, when all actions are available
        Returns:
            action_log_probs: log probabilities of actions [seq_len,batch_size, 1] or [batch_size, 1]
            dist_entropy: entropy of action distribution [seq_len, batch_size, 1] or [batch_size, 1]
            rnn_states_out: updated RNN states [batch_size, num_layers, hidden_size] or None
        """
        logits, rnn_states_out = self.forward(obs, rnn_states, masks)

        if available_actions is not None:
            # Set unavailable actions to have a very small probability
            logits[available_actions == 0] = -1e10

        action_dist = Categorical(logits=logits)
        action_log_probs = action_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)  # [seq_len, batch_size, 1]
        dist_entropy = action_dist.entropy().unsqueeze(-1)  # [seq_len, batch_size, 1]

        return action_log_probs, dist_entropy, rnn_states_out


class ActorSRMT(Actor):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super().__init__(args, obs_space, action_space, device=device)
        self.use_rnn = False
        self.rnn = None

        # SRMT specific params
        self.srmt_core = args.srmt_core
        self.use_agent_memory = args.use_agent_memory
        self.use_global_memory = args.use_global_memory
        self.data_chunk_length = args.data_chunk_length

        self.core = TransformerCore(args, self.hidden_size, device)

    def forward(self, x, rnn_states=None, masks=None, history_seq=None, agent_memory=None, global_memory=None):
        """
        Forward pass of the actor network.
        Batch size is n_agents * n_rollout_threads

        Args:
            x (torch.Tensor): Input tensor (batch_size, input_dim) or (seq_len, batch_size, input_dim)
            rnn_states (torch.Tensor, optional): RNN hidden state tensor. Required when use_rnn=True,
                                            ignored otherwise. Shape: (batch_size, num_layers, hidden_size)
            masks (torch.Tensor, optional): Mask tensor. Required when use_rnn=True, ignored otherwise.
                                       Shape: (batch_size, 1) or (seq_len, batch_size, 1)
        Returns:
            logits: action logits
            rnn_states_out: updated RNN states if use_rnn=True, None otherwise
        """

        if self.use_rnn and (rnn_states is None or masks is None):
            raise ValueError("rnn_states and masks must be provided when use_rnn=True")

        if self.use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        additional_outputs = {}
        if self.srmt_core:
            x, additional_outputs = self.core(x, history_seq, agent_memory, global_memory)

        if self.use_rnn:
            x, rnn_states_out = self.rnn(x, rnn_states, masks)
        else:
            rnn_states_out = None
        logits = self.output(x)  # [seq_len, batch_size, action_dim]

        return logits, rnn_states_out, additional_outputs

    def get_actions(self, obs, rnn_states=None, masks=None, available_actions=None, history_seq=None,
                    agent_memory=None, global_memory=None, deterministic=False):
        """Get actions from the actor network.
        Batch size is n_agents * n_rollout_threads

        Args:
            obs: tensor of shape [batch_size, input_dim]
            rnn_states: tensor of shape [batch_size, num_layers, rnn_hidden_size],
                required when use_rnn=True, can be None otherwise
            masks: tensor of shape [batch_size, 1], required when use_rnn=True,
                can be None otherwise
            available_actions: tensor of shape [batch_size, action_dim]
            deterministic: bool, whether to use deterministic actions

        Returns:
            actions: tensor of shape [n_agents, 1]
            action_log_probs: tensor of shape [batch_size, 1]
            next_rnn_states: tensor of shape [batch_size, num_layers, rnn_hidden_size]
                if use_rnn=True, None otherwise
        """
        # Forward pass to get logits
        logits, rnn_states_out, additional_outputs = self.forward(obs, rnn_states, masks, history_seq, agent_memory,
                                                                  global_memory)

        # Apply mask for available actions if provided
        if available_actions is not None:
            # Set unavailable actions to have a very small probability
            logits[available_actions == 0] = -1e10

        if deterministic:
            actions = torch.argmax(logits, dim=-1, keepdim=True)
            action_log_probs = None
        else:
            # Convert logits to action probabilities
            action_dist = Categorical(logits=logits)
            actions = action_dist.sample().unsqueeze(-1)  # (batch_size, 1)
            action_log_probs = action_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)  # (batch_size, 1)

        return actions, action_log_probs, rnn_states_out, additional_outputs

    def evaluate_actions(self, obs, actions, rnn_states=None, masks=None, available_actions=None, history_seq=None,
                         agent_memory=None, global_memory=None):
        """Evaluate actions for training.

        Args:
            obs_seq: tensor of shape [seq_len, batch_size, input_dim] or [batch_size, input_dim]
            actions: tensor of shape [seq_len, batch_size, 1] or [batch_size, 1]
            rnn_states: tensor of shape [batch_size, num_layers, hidden_size] - initial hidden state
                required when use_rnn=True, can be None otherwise
            masks: tensor of shape [seq_len, batch_size, 1] or [batch_size, 1]
                required when use_rnn=True, can be None otherwise
            available_actions: tensor of shape [seq_len, batch_size, action_dim] or [batch_size, action_dim]
                can be None, when all actions are available
        Returns:
            action_log_probs: log probabilities of actions [seq_len,batch_size, 1] or [batch_size, 1]
            dist_entropy: entropy of action distribution [seq_len, batch_size, 1] or [batch_size, 1]
            rnn_states_out: updated RNN states [batch_size, num_layers, hidden_size] or None
        """
        logits, rnn_states_out, additional_outputs = self.forward(obs, rnn_states, masks, history_seq, agent_memory,
                                                                  global_memory)

        if available_actions is not None:
            # Set unavailable actions to have a very small probability
            logits[available_actions == 0] = -1e10

        action_dist = Categorical(logits=logits)
        action_log_probs = action_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)  # [seq_len, batch_size, 1]
        dist_entropy = action_dist.entropy().unsqueeze(-1)  # [seq_len, batch_size, 1]

        return action_log_probs, dist_entropy, rnn_states_out


class Critic(nn.Module):
    """
    Critic network for MAPPO.
    """

    def __init__(self, args, centralized_obs_space, device=torch.device("cpu")):
        """Initialize the actor network.

        Args:
            args (argparse.Namespace): Arguments containing training hyperparameters
            centralized_obs_space (gymnasium.spaces.Box): Centralized observation space for critic (Box)
            device (torch.device): Device to run the agent on
        """
        super(Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self.use_rnn = args.use_rnn
        self.use_feature_normalization = args.use_feature_normalization
        self.rnn_layers = args.rnn_layers
        self.fc_layers = args.fc_layers

        cent_obs_shape = get_shape_from_obs_space(centralized_obs_space)
        cent_obs_dim = cent_obs_shape[0]

        # Feature Normalization
        if self.use_feature_normalization:
            self.feature_norm = nn.LayerNorm(cent_obs_dim)

        # MLP Layers
        layers = []
        in_dim = cent_obs_dim
        for _ in range(self.fc_layers):
            layers += [
                nn.Linear(in_dim, self.hidden_size),
                nn.ReLU(),
                nn.LayerNorm(self.hidden_size),
            ]
            in_dim = self.hidden_size
        self.mlp = nn.Sequential(*layers)

        # RNN Layer (GRU)
        if self.use_rnn:
            self.rnn = GRUModule(self.hidden_size,
                                 self.hidden_size,
                                 num_layers=self.rnn_layers)

        # Output Layer
        self.output = nn.Linear(self.hidden_size, 1)

        # Initialize weights
        self.apply(lambda module: _orthogonal_init(module, gain=nn.init.calculate_gain('relu')))
        _orthogonal_init(self.output, gain=1.0)

        self.to(device)

    def forward(self, x, rnn_states=None, masks=None):
        """Forward pass for critic network.

        Args:
            x (torch.Tensor): Input tensor (batch_size, input_dim) or (seq_len, batch_size, input_dim)
            rnn_states (torch.Tensor, optional): RNN hidden state tensor. Required when use_rnn=True,
                ignored when use_rnn=False. Shape: (batch_size, num_layers, hidden_size)
            masks (torch.Tensor, optional): Mask tensor. Required when use_rnn=True,
                ignored when use_rnn=False. Shape: (batch_size, 1) or (seq_len, batch_size, 1)
        Returns:
            values (torch.Tensor): Value predictions, shape (batch_size, 1) or (seq_len, batch_size, 1).
            rnn_states_out (torch.Tensor): updated RNN states if use_rnn=True, None otherwise
        """
        # Validate inputs when RNN is used
        if self.use_rnn and (rnn_states is None or masks is None):
            raise ValueError("rnn_states and masks must be provided when use_rnn=True")

        if self.use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        if self.use_rnn:
            x, rnn_states_out = self.rnn(x, rnn_states, masks)
        else:
            rnn_states_out = None

        values = self.output(x)  # [seq_len, batch_size, 1]

        return values, rnn_states_out


class ActorCriticSharedWeights(nn.Module):
    """
    Shared weights Actor Critic network
    Without RNN for now
    """

    def __init__(self, args: Namespace, obs_space: Box, action_space: Discrete, device=torch.device('cpu')):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.use_rnn = args.use_rnn and self.actor_rnn is not None
        self.use_feature_normalization = args.use_feature_normalization
        self.rnn_layers = args.rnn_layers

        self.actor_gain = args.actor_gain
        self.critic_gain = args.critic_gain

        obs_shape = get_shape_from_obs_space(obs_space)
        obs_dim = obs_shape[0]

        action_type = action_space.__class__.__name__
        if action_type == "Discrete":
            self.action_dim = action_space.n
        else:
            raise NotImplementedError("Only discrete action space is supported")

        # Feature Normalization
        if self.use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)
        """
        self.cnn_layer_configs = {
            'out_channels': [64, 128, 256],
            'kernel_size':  [3, 3, 3],
            'stride':       [2, 1, 1],
            'padding':      ['valid', 'valid', 'valid'],
        }
        """
        self.mlp_layer_configs = [self.hidden_size] * 2

        # TODO: take Encoder config out of class, just pass self.encoder = CNNMLPEncoder(*EncoderConfig)
        """
        self.encoder = CNNMLPEncoder(
            input_dim=obs_dim,
            output_dim=self.hidden_size,
            cnn_layer_configs=self.cnn_layer_configs,
            mlp_layer_configs=self.mlp_layer_configs,
        )
        """

        self.encoder = MLPEncoder(
            input_dim=obs_dim,
            output_dim=self.hidden_size,
            mlp_layer_configs=self.mlp_layer_configs,
        )


        # SRMT specific params
        self.srmt_core = args.srmt_core
        self.use_agent_memory = args.use_agent_memory
        self.use_global_memory = args.use_global_memory
        self.data_chunk_length = args.data_chunk_length
        self.core = TransformerCore(args, self.hidden_size, device) if self.srmt_core else None

        self.actor_decoder = nn.Linear(self.hidden_size, self.action_dim)
        self.critic_decoder = nn.Linear(self.hidden_size, 1)

        if self.use_rnn:
            self.actor_rnn = None
            self.critic_rnn = GRUModule(self.hidden_size,
                                        self.hidden_size,
                                        num_layers=self.rnn_layers)



        self.apply(lambda module: _orthogonal_init(module, gain=nn.init.calculate_gain('relu')))
        _orthogonal_init(self.actor_decoder, gain=self.actor_gain)
        _orthogonal_init(self.actor_decoder, gain=self.critic_gain)

        self.to(device)

    def forward(self,
                x,
                actor_rnn_states=None,
                critic_rnn_states=None,
                masks=None,
                available_actions=None,
                history_seq=None,
                agent_memory=None,
                global_memory=None,
                actor=True,
                critic=True,
                eval=False,
                actions=None,
                deterministic=False):
        # TODO: check program behavior. Maybe add actions as an input for action evaluation

        if self.use_feature_normalization:
            x = self.feature_norm(x)

        x = self.encoder(x)

        additional_outputs = {}
        if self.srmt_core:
            x, additional_outputs = self.core(x, history_seq, agent_memory, global_memory)

        # Forward actor
        dist_entropy = None
        action_log_probs = None
        actor_rnn_states_out = None
        x_actor = x
        if actor:
            if self.use_rnn and self.actor_rnn is not None:
                if actor_rnn_states is None or masks is None:
                    raise ValueError("rnn_states and masks must be provided when use_rnn=True")
                x_actor, actor_rnn_states_out = self.actor_rnn(x, actor_rnn_states, masks)

            logits = self.actor_decoder(x_actor)
            # Apply mask for available actions if provided
            if available_actions is not None:
                # Set unavailable actions to have a very small probability
                logits[available_actions == 0] = -1e10

            if deterministic:
                actions = torch.argmax(logits, dim=-1, keepdim=True)
                action_log_probs = None
            else:
                # Convert logits to action probabilities
                action_dist = Categorical(logits=logits)
                actions = action_dist.sample().unsqueeze(-1) if actions is None else actions  # (batch_size, 1)
                action_log_probs = action_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)  # (batch_size, 1)
                if eval:
                    dist_entropy = action_dist.entropy().unsqueeze(-1)  # [seq_len, batch_size, 1]

        else:
            actions = None


        # Forward critic
        values = None
        critic_rnn_states_out = None
        x_critic = x
        if critic:
            if self.use_rnn and self.critic_rnn is not None:
                if critic_rnn_states is None or masks is None:
                    raise ValueError("rnn_states and masks must be provided when use_rnn=True")
                x_critic, critic_rnn_states_out = self.critic_rnn(x, critic_rnn_states, masks)
            values = self.critic_decoder(x_critic)

        results = {
            # Actor part
            'actions':              actions,
            'action_log_probs':     action_log_probs,
            'dist_entropy':         dist_entropy,
            'actor_rnn_states_out':  actor_rnn_states_out,
            # Critic part
            'values':               values,
            'critic_rnn_states_out': critic_rnn_states_out,
            # Anything else
            'additional_outputs':   additional_outputs
        }

        return results

    def get_actions(self,
                    obs,
                    actor_rnn_states=None,
                    masks=None,
                    available_actions=None,
                    history_seq=None,
                    agent_memory=None,
                    global_memory=None,
                    deterministic=False, ):
        """Get actions from the actor network.
        Batch size is n_agents * n_rollout_threads

        Args:
            obs: tensor of shape [batch_size, input_dim]
            rnn_states: tensor of shape [batch_size, num_layers, rnn_hidden_size],
                required when use_rnn=True, can be None otherwise
            masks: tensor of shape [batch_size, 1], required when use_rnn=True,
                can be None otherwise
            available_actions: tensor of shape [batch_size, action_dim]
            deterministic: bool, whether to use deterministic actions

        Returns:
            actions: tensor of shape [n_agents, 1]
            action_log_probs: tensor of shape [batch_size, 1]
            next_rnn_states: tensor of shape [batch_size, num_layers, rnn_hidden_size]
                if use_rnn=True, None otherwise
        """

        results = self.forward(obs,
                               actor_rnn_states=actor_rnn_states,
                               masks=masks,
                               available_actions=available_actions,
                               history_seq=history_seq,
                               agent_memory=agent_memory,
                               global_memory=global_memory,
                               deterministic=deterministic,
                               critic=False)

        actions = results['actions']
        action_log_probs = results['action_log_probs']
        actor_rnn_states_out = results['actor_rnn_states_out']
        additional_outputs = results['additional_outputs']

        if self.srmt_core:
            return actions, action_log_probs, actor_rnn_states_out, additional_outputs
        else:
            return actions, action_log_probs, actor_rnn_states_out

    def get_values(self,
                   obs,
                   critic_rnn_states=None,
                   masks=None,
                   history_seq=None,
                   agent_memory=None,
                   global_memory=None,):
        """Get actions from the actor network.
        Batch size is n_agents * n_rollout_threads

        Args:
            obs: tensor of shape [batch_size, input_dim]
            rnn_states: tensor of shape [batch_size, num_layers, rnn_hidden_size],
                required when use_rnn=True, can be None otherwise
            masks: tensor of shape [batch_size, 1], required when use_rnn=True,
                can be None otherwise
            available_actions: tensor of shape [batch_size, action_dim]
            deterministic: bool, whether to use deterministic actions

        Returns:
            actions: tensor of shape [n_agents, 1]
            action_log_probs: tensor of shape [batch_size, 1]
            next_rnn_states: tensor of shape [batch_size, num_layers, rnn_hidden_size]
                if use_rnn=True, None otherwise
        """

        results = self.forward(obs,
                               critic_rnn_states=critic_rnn_states,
                               masks=masks,
                               history_seq=history_seq,
                               agent_memory=agent_memory,
                               global_memory=global_memory,
                               actor=False)

        values = results['values']
        critic_rnn_states_out = results['critic_rnn_states_out']

        return values, critic_rnn_states_out

    def evaluate_actions(self,
                         obs,
                         actions,
                         actor_rnn_states=None,
                         critic_rnn_states=None,
                         masks=None,
                         available_actions=None,
                         history_seq=None,
                         agent_memory=None,
                         global_memory=None,
                         deterministic=False):
        """Evaluate actions for training.

        Args:
            obs_seq: tensor of shape [seq_len, batch_size, input_dim] or [batch_size, input_dim]
            actions: tensor of shape [seq_len, batch_size, 1] or [batch_size, 1]
            rnn_states: tensor of shape [batch_size, num_layers, hidden_size] - initial hidden state
                required when use_rnn=True, can be None otherwise
            masks: tensor of shape [seq_len, batch_size, 1] or [batch_size, 1]
                required when use_rnn=True, can be None otherwise
            available_actions: tensor of shape [seq_len, batch_size, action_dim] or [batch_size, action_dim]
                can be None, when all actions are available
        Returns:
            action_log_probs: log probabilities of actions [seq_len,batch_size, 1] or [batch_size, 1]
            dist_entropy: entropy of action distribution [seq_len, batch_size, 1] or [batch_size, 1]
            rnn_states_out: updated RNN states [batch_size, num_layers, hidden_size] or None
        """
        results = self.forward(obs,
                               actor_rnn_states=actor_rnn_states,
                               critic_rnn_states=critic_rnn_states,
                               masks=masks,
                               available_actions=available_actions,
                               history_seq=history_seq,
                               agent_memory=agent_memory,
                               global_memory=global_memory,
                               deterministic=deterministic,
                               eval=True,
                               actions=actions,
                               )

        values = results['values']
        action_log_probs = results['action_log_probs']
        dist_entropy = results['dist_entropy']

        return values, action_log_probs, dist_entropy
