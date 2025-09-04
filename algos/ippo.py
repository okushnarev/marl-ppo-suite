from algos.mappo import MAPPO, MAPPO_SRMT
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium

from networks.mappo_nets import Actor, ActorCriticSRMT, Critic
from utils.scheduler import LinearScheduler
from utils.value_normalizers import create_value_normalizer
from typing import Optional


class IPPO(MAPPO):

    def _init_networks(self,
                       obs_space: gymnasium.spaces.Box,
                       state_space: gymnasium.spaces.Box,
                       action_space: gymnasium.spaces.Discrete) -> None:

        self.actor_critic = ActorCriticSRMT(
            self.args,
            obs_space,
            action_space,
            self.device
        )
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=self.lr,
            eps=self.args.optimizer_eps
        )

        self.actor = self.actor_critic
        self.critic = self.actor_critic

        if self.args.use_linear_lr_decay:
            self.scheduler = LinearScheduler(
                self.lr,
                self.args.min_lr,
                self.args.max_steps
            )

    def save(self, save_path, save_args=False):
        """Save both actor and critic networks."""
        # Save model weights and optimizer states
        model_path = save_path

        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict':    self.optimizer.state_dict(),
        }, model_path)

        # Save args separately
        if save_args:
            args_path = save_path + '.args'
            torch.save({'args': self.args}, args_path)

    def load(self, model_path):
        """Load both actor and critic networks."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

        # Load network states
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])

        # Load optimizer states
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load args separately if they exist
        args_path = model_path + '.args'
        if os.path.exists(args_path):
            args_dict = torch.load(args_path, weights_only=False)
            self.args = args_dict['args']

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
            values, rnn_states_out = self.actor_critic.forward_critic(
                obs,
                rnn_states,
                masks
            )

            return values, rnn_states_out

    def evaluate_actions(self, state, obs, actions, available_actions, masks, active_masks, actor_h0=None,
                         critic_h0=None,):
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
        action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            obs,
            actions,
            actor_h0,
            masks,
            available_actions)

        values, _ = self.actor_critic.forward_critic(obs, critic_h0, masks)
        return values, action_log_probs, dist_entropy

    def update(self, mini_batch):
        """
        Update policy using a mini-batch of experiences.

        Args:
            mini_batch (dict): Dictionary containing mini-batch data

        Returns:
            tuple: (value_loss, policy_loss, dist_entropy)
        """
        metrics = {}
        # Extract data from mini-batch
        (obs_batch,
         global_state_batch,
         actor_h0_batch,
         critic_h0_batch,
         actions_batch,
         values_batch,
         returns_batch,
         masks_batch,
         active_masks_batch,
         old_action_log_probs_batch,
         advantages_batch,
         available_actions_batch,
         ) = mini_batch

        # Evaluate actions
        values, action_log_probs, dist_entropy = self.evaluate_actions(
            global_state_batch, obs_batch, actions_batch,
            available_actions_batch, masks_batch, active_masks_batch,
            actor_h0_batch, critic_h0_batch,
        )

        # Calculate PPO ratio and KL divergence
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
        clip_ratio = (torch.abs(ratio - 1) > self.clip_param).float().mean().item()

        # Actor Loss
        surr1 = ratio * advantages_batch
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
        policy_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = -self.entropy_coef * torch.mean(dist_entropy)
        actor_loss = policy_loss + entropy_loss

        #  Critic loss
        critic_loss = self.compute_value_loss(values, values_batch, returns_batch)

        # Overall loss
        overall_loss = actor_loss + critic_loss

        # Update Actor Critic
        self.optimizer.zero_grad()
        overall_loss.backward()
        grad_norm = self._clip_gradients(self.actor_critic)
        self.optimizer.step()

        # Update metrics
        metrics.update({
            'critic_loss':      critic_loss.item(),
            'actor_loss':       actor_loss.item(),
            'entropy_loss':     entropy_loss.item(),
            'approx_kl':        approx_kl,
            'clip_ratio':       clip_ratio,
            'actor_grad_norm':  grad_norm,
            'critic_grad_norm': grad_norm
        })

        return metrics


class IPPO_SRMT(MAPPO_SRMT):

    def _init_networks(self,
                       obs_space: gymnasium.spaces.Box,
                       state_space: gymnasium.spaces.Box,
                       action_space: gymnasium.spaces.Discrete) -> None:

        self.actor_critic = ActorCriticSRMT(
            self.args,
            obs_space,
            action_space,
            self.device
        )
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=self.lr,
            eps=self.args.optimizer_eps
        )

        self.actor = self.actor_critic
        self.critic = self.actor_critic

        if self.args.use_linear_lr_decay:
            self.scheduler = LinearScheduler(
                self.lr,
                self.args.min_lr,
                self.args.max_steps
            )

    def save(self, save_path, save_args=False):
        """Save both actor and critic networks."""
        # Save model weights and optimizer states
        model_path = save_path

        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict':    self.optimizer.state_dict(),
        }, model_path)

        # Save args separately
        if save_args:
            args_path = save_path + '.args'
            torch.save({'args': self.args}, args_path)

    def load(self, model_path):
        """Load both actor and critic networks."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

        # Load network states
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])

        # Load optimizer states
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load args separately if they exist
        args_path = model_path + '.args'
        if os.path.exists(args_path):
            args_dict = torch.load(args_path, weights_only=False)
            self.args = args_dict['args']

    def get_values(self,
                   state: torch.Tensor,
                   obs: torch.Tensor,
                   active_masks: torch.Tensor,
                   rnn_states: torch.Tensor = None,
                   masks: torch.Tensor = None,
                   history_seq=None,
                   agent_memory=None,
                   global_memory=None):

        with torch.no_grad():
            # Handle RNN states and masks based on whether RNN is enabled
            if self.use_rnn:
                if rnn_states is None or masks is None:
                    raise ValueError("rnn_states and masks must be provided when RNN is enabled")

            # Get values and states
            values, rnn_states_out = self.actor_critic.forward_critic(
                obs,
                rnn_states,
                masks,
                history_seq,
                agent_memory,
                global_memory,
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
        action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            obs,
            actions,
            actor_h0,
            masks,
            available_actions,
            history_seq,
            agent_memory,
            global_memory,
        )

        values, _ = self.actor_critic.forward_critic(obs, critic_h0, masks, history_seq, agent_memory, global_memory)
        return values, action_log_probs, dist_entropy

    def update(self, mini_batch):
        """
        Update policy using a mini-batch of experiences.

        Args:
            mini_batch (dict): Dictionary containing mini-batch data

        Returns:
            tuple: (value_loss, policy_loss, dist_entropy)
        """
        metrics = {}
        # Extract data from mini-batch
        (obs_batch,
         global_state_batch,
         actor_h0_batch,
         critic_h0_batch,
         actions_batch,
         values_batch,
         returns_batch,
         masks_batch,
         active_masks_batch,
         old_action_log_probs_batch,
         advantages_batch,
         available_actions_batch,
         history_seq,
         agent_memory,
         global_memory,
         ) = mini_batch

        # Evaluate actions
        values, action_log_probs, dist_entropy = self.evaluate_actions(
            global_state_batch, obs_batch, actions_batch,
            available_actions_batch, masks_batch, active_masks_batch,
            actor_h0_batch, critic_h0_batch,
            history_seq, agent_memory, global_memory,
        )

        # Calculate PPO ratio and KL divergence
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
        clip_ratio = (torch.abs(ratio - 1) > self.clip_param).float().mean().item()

        # Actor Loss
        surr1 = ratio * advantages_batch
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
        policy_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = -self.entropy_coef * torch.mean(dist_entropy)
        actor_loss = policy_loss + entropy_loss

        #  Critic loss
        critic_loss = self.compute_value_loss(values, values_batch, returns_batch)

        # Overall loss
        overall_loss = actor_loss + critic_loss

        # Update Actor Critic
        self.optimizer.zero_grad()
        overall_loss.backward()
        grad_norm = self._clip_gradients(self.actor_critic)
        self.optimizer.step()

        # Update metrics
        metrics.update({
            'critic_loss':      critic_loss.item(),
            'actor_loss':       actor_loss.item(),
            'entropy_loss':     entropy_loss.item(),
            'approx_kl':        approx_kl,
            'clip_ratio':       clip_ratio,
            'actor_grad_norm':  grad_norm,
            'critic_grad_norm': grad_norm
        })

        return metrics

