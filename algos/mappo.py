import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium

from networks.mappo_nets import Actor, ActorSRMT, Critic
from utils.scheduler import LinearScheduler
from utils.value_normalizers import create_value_normalizer
from typing import Optional


class MAPPO:
    """
    Multi-Agent Proximal Policy Optimization (MAPPO) agent implementation with RNN networks.

    This agent implements the MAPPO algorithm for multi-agent reinforcement learning
    with centralized training and decentralized execution.
    """

    def __init__(self, args, obs_space, state_space, action_space, device=torch.device("cpu")):
        """
        Initialize the MAPPO agent.

        Args:
            args: Arguments containing training hyperparameters
            obs_space: Observation space for individual agents (Box)
            state_space: Centralized observation space for critic (Box)
            action_space: Action space (Discrete)
            device (torch.device): Device to run the agent on
        """
        # Extract dimensions from spaces
        obs_dim = obs_space.shape[0]
        state_dim = state_space.shape[0]
        action_dim = action_space.n

        # Input validation
        self._validate_inputs(args, obs_dim, state_dim, action_dim)

        self.args = args
        self.state_type = args.state_type  # [FP, EP, AS]
        self.n_agents = args.n_agents
        self.device = device

        # Initialize core components
        self._init_hyperparameters()
        self._init_networks(obs_space, state_space, action_space)

        if self.use_value_norm:
            self.value_normalizer = create_value_normalizer(
                normalizer_type=self.args.value_norm_type,
                device=device
            )
        else:
            self.value_normalizer = None

    def _validate_inputs(self, args, obs_dim: int, state_dim: int, action_dim: int) -> None:
        """Validate input parameters."""
        if obs_dim <= 0 or state_dim <= 0 or action_dim <= 0:
            raise ValueError(
                f"Dimensions must be positive integers: obs_dim={obs_dim}, state_dim={state_dim}, action_dim={action_dim}")
        required_attrs = ['n_agents', 'use_rnn', 'state_type', 'lr', 'clip_param', 'ppo_epoch',
                          'num_mini_batch']
        missing = [attr for attr in required_attrs if not hasattr(args, attr)]
        if missing:
            raise AttributeError(f"args missing required attributes: {missing}")

    def _init_hyperparameters(self) -> None:
        """Initialize training hyperparameters."""
        self.use_rnn = self.args.use_rnn
        self.clip_param = self.args.clip_param
        self.ppo_epoch = self.args.ppo_epoch
        self.num_mini_batch = self.args.num_mini_batch
        self.data_chunk_length = self.args.data_chunk_length
        self.entropy_coef = self.args.entropy_coef
        self.critic_coef = self.args.critic_coef
        self.max_grad_norm = self.args.max_grad_norm
        self.use_max_grad_norm = self.args.use_max_grad_norm
        self.use_clipped_value_loss = self.args.use_clipped_value_loss
        self.use_value_norm = self.args.use_value_norm
        self.use_huber_loss = self.args.use_huber_loss
        self.huber_delta = self.args.huber_delta

        # Training parameters
        self.lr = self.args.lr
        self.gamma = self.args.gamma
        self.use_gae = self.args.use_gae
        self.gae_lambda = self.args.gae_lambda

    def _init_networks(self,
                       obs_space: gymnasium.spaces.Box,
                       state_space: gymnasium.spaces.Box,
                       action_space: gymnasium.spaces.Discrete) -> None:
        """Initialize actor and critic networks with proper weight initialization."""

        self.actor = Actor(
            self.args,
            obs_space,
            action_space,
            self.device
        )

        if self.state_type == "AS":
            # Concatenate observation and state spaces for AS state type
            concat_dim = obs_space.shape[0] + state_space.shape[0]
            state_space = gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(concat_dim,),
                dtype=np.float32
            )

        self.critic = Critic(
            self.args,
            state_space,
            self.device
        )

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.args.optimizer_eps
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.lr,
            eps=self.args.optimizer_eps
        )

        if self.args.use_linear_lr_decay:
            self.scheduler = LinearScheduler(
                self.lr,
                self.args.min_lr,
                self.args.max_steps
            )

    def _clip_gradients(self, network: nn.Module) -> Optional[float]:
        """Clip gradients and return gradient norm."""
        if self.use_max_grad_norm:
            return nn.utils.clip_grad_norm_(
                network.parameters(),
                self.max_grad_norm
            )
        return None

    def get_actions(
            self,
            obs: torch.Tensor,
            rnn_states: torch.Tensor = None,
            masks: torch.Tensor = None,
            available_actions: torch.Tensor = None,
            deterministic: bool = False
    ):
        """
        Get actions from the policy network.
        Batched version -> Batch(B) = (n_rollout_threads, n_agents) = n_rollout_threads * n_agents

        Args:
            obs (torch.Tensor): Observation tensor #(batch_size, obs_shape)
            rnn_states (torch.Tensor, optional): RNN states tensor. Required when RNN is enabled,
                can be None otherwise. #(batch_size, num_layers hidden_size)
            masks (torch.Tensor, optional): Masks tensor. Required when RNN is enabled,
                can be None otherwise. #(batch_size, 1)
            available_actions (torch.Tensor, optional): Available actions tensor #(batch_size, n_actions)
            deterministic (bool): Whether to use deterministic actions

        Returns:
            actions (torch.Tensor): Actions tensor #(batch_size, 1)
            action_log_probs (torch.Tensor): Action log probabilities tensor #(batch_size, 1)
            rnn_states_out (torch.Tensor): Updated RNN states tensor #(batch_size, num_layers, hidden_size)
                                        or None if RNN is disabled
        """
        with torch.no_grad():
            # Handle RNN states and masks based on whether RNN is enabled
            if self.use_rnn:
                if rnn_states is None or masks is None:
                    raise ValueError("rnn_states and masks must be provided when RNN is enabled")

            # Get actions
            actions, action_log_probs, rnn_states_out = self.actor.get_actions(
                obs, rnn_states, masks, available_actions, deterministic
            )

        return actions, action_log_probs, rnn_states_out

    def get_values(self,
                   state: torch.Tensor,
                   obs: torch.Tensor,
                   active_masks: torch.Tensor,
                   rnn_states: torch.Tensor = None,
                   masks: torch.Tensor = None):
        """
        Get values from the critic network.
        Batched version -> Batch(B) = (n_rollout_threads, n_agents) = n_rollout_threads * n_agents

        Args:
            state (torch.Tensor): State tensor #(B, state_shape)
            obs (torch.Tensor): Observation tensor #(B, obs_shape)
            active_masks (torch.Tensor): Active masks tensor #(B, 1) - used for death masking in (AS)
            rnn_states (torch.Tensor, optional): RNN states tensor. Required when RNN is enabled,
                                          can be None otherwise. #(B, n_layers, hidden_size)
            masks (torch.Tensor, optional): Masks tensor. Required when RNN is enabled,
                                     can be None otherwise. #(B, 1)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (values, rnn_states_out) where rnn_states_out is None
                                      if RNN is disabled
        """
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
                state,
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

        if self.state_type == "AS":
            state = state * active_masks  # (seq_len, batch_size, n_state) # Mask out inactive agents
            # Concatenate observation and state spaces for AS state type
            state = torch.cat([obs, state], dim=-1)

        values, _ = self.critic(state, critic_h0, masks)
        return values, action_log_probs, dist_entropy

    def compute_value_loss(self, values, value_preds_batch, returns_batch):
        """
        Compute value function loss with normalization.

        Args:
            values: Current value predictions
            value_preds_batch: Old value predictions
            return_batch: Return targets
        """
        if self.use_value_norm:
            returns = self.value_normalizer.normalize(returns_batch, update=True)
            # values = self.value_normalizer.normalize(values, update=False)
            # value_preds_batch = self.value_normalizer.normalize(value_preds_batch, update=False)
        else:
            returns = returns_batch

        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + torch.clamp(
                values - value_preds_batch,
                -self.clip_param,
                self.clip_param
            )
            if self.use_huber_loss:
                # Compute Huber loss for clipped and unclipped predictions
                value_losses = F.huber_loss(values, returns, delta=self.huber_delta, reduction='none')
                value_losses_clipped = F.huber_loss(value_pred_clipped, returns, delta=self.huber_delta,
                                                    reduction='none')
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_losses = (values - returns).pow(2)
                value_losses_clipped = (value_pred_clipped - returns).pow(2)
                value_loss = self.critic_coef * torch.max(value_losses, value_losses_clipped).mean()
        else:
            if self.use_huber_loss:
                value_loss = F.huber_loss(values, returns, delta=self.huber_delta, reduction='mean')
            else:
                value_loss = self.critic_coef * (values - returns).pow(2).mean()

        return value_loss

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
         available_actions_batch) = mini_batch

        # Evaluate actions
        values, action_log_probs, dist_entropy = self.evaluate_actions(
            global_state_batch, obs_batch, actions_batch,
            available_actions_batch, masks_batch, active_masks_batch,
            actor_h0_batch, critic_h0_batch
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

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = self._clip_gradients(self.actor)
        self.actor_optimizer.step()

        #  Critic loss
        critic_loss = self.compute_value_loss(values, values_batch, returns_batch)

        # Update Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = self._clip_gradients(self.critic)
        self.critic_optimizer.step()

        # Update metrics
        metrics.update({
            'critic_loss':      critic_loss.item(),
            'actor_loss':       actor_loss.item(),
            'entropy_loss':     entropy_loss.item(),
            'approx_kl':        approx_kl,
            'clip_ratio':       clip_ratio,
            'actor_grad_norm':  actor_grad_norm,
            'critic_grad_norm': critic_grad_norm
        })

        return metrics

    def train(self, buffer):
        """
        Train the policy using experiences from the buffer.
        """
        train_info = {
            'critic_loss':      0,
            'actor_loss':       0,
            'entropy_loss':     0,
            'approx_kl':        0,
            'clip_ratio':       0,
            'actor_grad_norm':  0,
            'critic_grad_norm': 0,
        }

        # Train for ppo_epoch iterations
        for _ in range(self.ppo_epoch):

            if self.use_rnn:
                # Generate mini-batches
                mini_batches = buffer.get_minibatches_seq_first(
                    self.num_mini_batch,
                    data_chunk_length=self.data_chunk_length)
            else:
                # Generate mini-batches without RNN
                mini_batches = buffer.get_minibatches(self.num_mini_batch)

            # Update for each mini-batch
            for mini_batch in mini_batches:
                metrics = self.update(mini_batch)

                # Update training info
                for k, v in metrics.items():
                    if k in train_info:
                        train_info[k] += v

        # Calculate means
        num_updates = self.ppo_epoch * self.num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def update_learning_rate(self, current_step):
        """Update the learning rate based on the current step."""
        lr_now = self.scheduler.get_lr(current_step)
        for p in self.actor_optimizer.param_groups:
            p['lr'] = lr_now
        for p in self.critic_optimizer.param_groups:
            p['lr'] = lr_now

        return {
            'actor_lr':  self.actor_optimizer.param_groups[0]['lr'],
            'critic_lr': self.critic_optimizer.param_groups[0]['lr']
        }

    def save(self, save_path, save_args=False):
        """Save both actor and critic networks."""
        # Save model weights and optimizer states
        model_path = save_path

        torch.save({
            'actor_state_dict':            self.actor.state_dict(),
            'critic_state_dict':           self.critic.state_dict(),
            'actor_optimizer_state_dict':  self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, model_path)

        # Save args separately
        if save_args:
            args_path = save_path + '.args'
            torch.save({'args': self.args}, args_path)

    def load(self, model_path):
        """Load both actor and critic networks."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

        # Load network states
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])

        # Load optimizer states
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        # Load args separately if they exist
        args_path = model_path + '.args'
        if os.path.exists(args_path):
            args_dict = torch.load(args_path, weights_only=False)
            self.args = args_dict['args']


class MAPPO_SRMT(MAPPO):
    def _init_networks(self,
                       obs_space: gymnasium.spaces.Box,
                       state_space: gymnasium.spaces.Box,
                       action_space: gymnasium.spaces.Discrete) -> None:
        super()._init_networks(obs_space, state_space, action_space)

        self.actor = ActorSRMT(
            self.args,
            obs_space,
            action_space,
            self.device
        )

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.args.optimizer_eps
        )

    def get_actions(
            self,
            obs: torch.Tensor,
            rnn_states: torch.Tensor = None,
            masks: torch.Tensor = None,
            available_actions: torch.Tensor = None,
            history_seq: torch.Tensor = None,
            agent_memory: torch.Tensor = None,
            global_memory: torch.Tensor = None,
            deterministic: bool = False,
    ):
        with torch.no_grad():
            # Handle RNN states and masks based on whether RNN is enabled
            if self.use_rnn:
                if rnn_states is None or masks is None:
                    raise ValueError("rnn_states and masks must be provided when RNN is enabled")

            # Get actions
            actions, action_log_probs, rnn_states_out, additional_outputs = self.actor.get_actions(
                obs, rnn_states, masks, available_actions, history_seq, agent_memory, global_memory, deterministic
            )

        return actions, action_log_probs, rnn_states_out, *additional_outputs.values()

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

        if self.state_type == "AS":
            state = state * active_masks  # (seq_len, batch_size, n_state) # Mask out inactive agents
            # Concatenate observation and state spaces for AS state type
            state = torch.cat([obs, state], dim=-1)

        values, _ = self.critic(state, critic_h0, masks)
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

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = self._clip_gradients(self.actor)
        self.actor_optimizer.step()

        #  Critic loss
        critic_loss = self.compute_value_loss(values, values_batch, returns_batch)

        # Update Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = self._clip_gradients(self.critic)
        self.critic_optimizer.step()

        # Update metrics
        metrics.update({
            'critic_loss':      critic_loss.item(),
            'actor_loss':       actor_loss.item(),
            'entropy_loss':     entropy_loss.item(),
            'approx_kl':        approx_kl,
            'clip_ratio':       clip_ratio,
            'actor_grad_norm':  actor_grad_norm,
            'critic_grad_norm': critic_grad_norm
        })

        return metrics
