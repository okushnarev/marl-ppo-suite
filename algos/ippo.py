from algos.mappo import MAPPO, MAPPO_SRMT
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium

from networks.mappo_nets import Actor, Critic
from utils.scheduler import LinearScheduler
from utils.value_normalizers import create_value_normalizer
from typing import Optional


class IPPO(MAPPO):

    def _init_networks(self,
                       obs_space: gymnasium.spaces.Box,
                       state_space: gymnasium.spaces.Box,
                       action_space: gymnasium.spaces.Discrete) -> None:
        super()._init_networks(obs_space, state_space, action_space)

        self.critic = Critic(
            self.args,
            obs_space,
            self.device
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.lr,
            eps=self.args.optimizer_eps
        )

    def get_values(self,
                   state: torch.Tensor,
                   obs: torch.Tensor,
                   active_masks: torch.Tensor,
                   rnn_states: torch.Tensor = None,
                   masks: torch.Tensor = None):

        with torch.no_grad():

            if self.state_type == "AS":
                # Concatenate observation and state spaces for AS state type
                state = state * active_masks  # (batch_size, n_state) # Mask out inactive agents
                state = torch.cat([obs, state], dim=-1)

                # Handle RNN states and masks based on whether RNN is enabled
            if self.use_rnn:
                if rnn_states is None or masks is None:
                    raise ValueError("rnn_states and masks must be provided when RNN is enabled")

            # Get values and states
            values, rnn_states_out = self.critic(
                obs,
                rnn_states,
                masks
            )

            return values, rnn_states_out

    def evaluate_actions(self, state, obs, actions, available_actions, masks, active_masks, actor_h0=None,
                         critic_h0=None):
        """
        Evaluate actions for training.

        Args:
            state (torch.Tensor): State tensor #(seq_len, batch_size, n_state) or #(batch_size, n_state)
            obs (torch.Tensor): Observation tensor #(seq_len, batch_size, n_obs) or #(batch_size, n_obs)
            actions (torch.Tensor): Actions tensor #(seq_len, batch_size, 1) or #(batch_size, 1)
            available_actions (torch.Tensor): Available actions tensor #(seq_len, batch_size, action_dim) or #(batch_size, action_dim)
            masks (torch.Tensor): Masks tensor #(seq_len, batch_size, 1) or #(batch_size, 1)
            active_masks (torch.Tensor): Active masks tensor #(seq_len, batch_size, 1) or #(batch_size, 1)
            actor_h0 (torch.Tensor): Initial actor RNN states tensor #(num_layers, batch_size, hidden_size) or None
            critic_h0 (torch.Tensor): Initial critic RNN states tensor #(num_layers, batch_size, hidden_size) or None

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (values, action_log_probs, dist_entropy)
        """
        action_log_probs, dist_entropy, _ = self.actor.evaluate_actions(
            obs,
            actions,
            actor_h0,
            masks,
            available_actions)

        values, _ = self.critic(obs, critic_h0, masks)
        return values, action_log_probs, dist_entropy


class IPPO_SRMT(MAPPO_SRMT):

    def _init_networks(self,
                       obs_space: gymnasium.spaces.Box,
                       state_space: gymnasium.spaces.Box,
                       action_space: gymnasium.spaces.Discrete) -> None:
        super()._init_networks(obs_space, state_space, action_space)

        self.critic = Critic(
            self.args,
            obs_space,
            self.device
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.lr,
            eps=self.args.optimizer_eps
        )

    def get_values(self,
                   state: torch.Tensor,
                   obs: torch.Tensor,
                   active_masks: torch.Tensor,
                   rnn_states: torch.Tensor = None,
                   masks: torch.Tensor = None):

        with torch.no_grad():
            # Handle RNN states and masks based on whether RNN is enabled
            if self.use_rnn:
                if rnn_states is None or masks is None:
                    raise ValueError("rnn_states and masks must be provided when RNN is enabled")

            # Get values and states
            values, rnn_states_out = self.critic(
                obs,
                rnn_states,
                masks
            )

            return values, rnn_states_out

    def evaluate_actions(self, state, obs, actions, available_actions, masks, active_masks, actor_h0=None,
                         critic_h0=None, history_seq=None, agent_memory=None, global_memory=None):
        """
        Evaluate actions for training.

        Args:
            state (torch.Tensor): State tensor #(seq_len, batch_size, n_state) or #(batch_size, n_state)
            obs (torch.Tensor): Observation tensor #(seq_len, batch_size, n_obs) or #(batch_size, n_obs)
            actions (torch.Tensor): Actions tensor #(seq_len, batch_size, 1) or #(batch_size, 1)
            available_actions (torch.Tensor): Available actions tensor #(seq_len, batch_size, action_dim) or #(batch_size, action_dim)
            masks (torch.Tensor): Masks tensor #(seq_len, batch_size, 1) or #(batch_size, 1)
            active_masks (torch.Tensor): Active masks tensor #(seq_len, batch_size, 1) or #(batch_size, 1)
            actor_h0 (torch.Tensor): Initial actor RNN states tensor #(num_layers, batch_size, hidden_size) or None
            critic_h0 (torch.Tensor): Initial critic RNN states tensor #(num_layers, batch_size, hidden_size) or None

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (values, action_log_probs, dist_entropy)
        """
        action_log_probs, dist_entropy, _ = self.actor.evaluate_actions(
            obs,
            actions,
            actor_h0,
            masks,
            available_actions,
            history_seq,
            agent_memory,
            global_memory,
        )

        values, _ = self.critic(obs, critic_h0, masks)
        return values, action_log_probs, dist_entropy
