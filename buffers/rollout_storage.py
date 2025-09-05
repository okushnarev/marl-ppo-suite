import numpy as np
import torch

from utils.env_tools import get_shape_from_act_space, get_shape_from_obs_space
from utils.transform_tools import flatten_time_batch, to_tensor


def _transform_data(data_np: np.ndarray, device: torch.device, sequence_first: bool = False) -> torch.Tensor:
    """
    Transform data to be used in RNN.

    Args:
        data_np: Input numpy array with shape [T, N, M, feat_dim] or [T, N, num_layers, M, feat_dim] for RNN states
        device: Target device for tensor
        sequence_first: If True, keeps sequence dimension first [T, N, M, feat_dim] -> [T, N, M, feat_dim]
                      If False, transposes sequence and batch [T, N, M, feat_dim] -> [N, M, T, feat_dim] -> [N*M*T, feat_dim]
                      So we get steps sorted by rollout threads and agents.

    Returns:
        Transformed torch tensor
    """
    if sequence_first:
        # Keep original sequence ordering
        return to_tensor(data_np, device=device, copy=True)
    else:
        # Original behavior: transpose and flatten
        # Use copy() to ensure contiguous memory layout in NumPy
        tail_indices = range(3, len(data_np.shape))
        reshaped_data = data_np.transpose(1, 2, 0, *tail_indices).reshape(-1, *data_np.shape[3:])
        return to_tensor(reshaped_data, device=device, copy=True)


class RolloutStorage:
    """
    Rollout storage for collecting multi-agent experiences during training.
    Designed for MAPPO with n-step returns and RNN-based policies.
    Support for agent-specific global state and multiple parallel environments.
    """

    def __init__(self, args, n_agents, obs_space, action_space, state_space, device='cpu'):
        """
        Initialize rollout storage for collecting experiences.

        Args:
            args: Arguments containing training hyperparameters
            args.n_steps: Number of steps to collect before update (can be different from episode length)
            args.n_rollout_threads: Number of parallel environments
            args.use_rnn: Whether to use RNN networks (default: False)
            args.rnn_layers: Number of layers in the RNN
            args.hidden_size: Dimension of hidden state
            n_agents (int): Number of agents in the environment
            obs_space: Observation space
            action_space: Action space
            state_space: State space
            device (str): Device for storage ('cpu' for numpy-based implementation)
        """
        self.n_steps = args.n_steps
        self.n_rollout_threads = args.n_rollout_threads
        self.n_agents = n_agents
        self.use_rnn = args.use_rnn
        self.num_rnn_layers = args.rnn_layers
        self.hidden_size = args.hidden_size
        self.is_fp_state = (args.state_type == "FP")
        self.device = torch.device(device)

        # Current position in the buffer
        self.step = 0
        # (obs_0, state_0) → action_0 / action_log_prob_0 → (reward_0, obs_1, state_1, mask_1, trunc_1)
        # delta = reward_0 + gamma * value_1 * mask_1 - value_0

        obs_shape = get_shape_from_obs_space(obs_space)
        action_shape = get_shape_from_act_space(action_space)
        state_shape = get_shape_from_obs_space(state_space)

        # Core storage buffers - using numpy arrays for efficiency
        self.obs = np.zeros((self.n_steps + 1, self.n_rollout_threads, n_agents, *obs_shape), dtype=np.float32)
        self.rewards = np.zeros((self.n_steps, self.n_rollout_threads, n_agents, 1), dtype=np.float32)
        self.actions = np.zeros((self.n_steps, self.n_rollout_threads, n_agents, action_shape), dtype=np.int64)
        self.action_log_probs = np.zeros((self.n_steps, self.n_rollout_threads, n_agents, 1), dtype=np.float32)
        self.values = np.zeros((self.n_steps + 1, self.n_rollout_threads, n_agents, 1), dtype=np.float32)
        self.masks = np.ones((self.n_steps + 1, self.n_rollout_threads, n_agents, 1),
                             dtype=np.float32)  # 0 if episode done, 1 otherwise
        self.active_masks = np.ones((self.n_steps + 1, self.n_rollout_threads, n_agents, 1),
                                    dtype=np.float32)  # 0 if agent dead, 1 otherwise
        self.truncated = np.zeros((self.n_steps + 1, self.n_rollout_threads, n_agents, 1),
                                  dtype=np.bool_)  # 1 if episode truncated, 0 otherwise

        # Handle global state and critic RNN states based on state type
        if self.is_fp_state:  # Agent-specific state (FP)
            self.agent_state = np.zeros(
                (self.n_steps + 1, self.n_rollout_threads, n_agents, *state_shape),
                dtype=np.float32)
            if self.use_rnn:
                self.agent_critic_rnn = np.zeros(
                    (self.n_steps + 1, self.n_rollout_threads, n_agents, self.num_rnn_layers, self.hidden_size),
                    dtype=np.float32)
        else:  # Environment-central state (EP)
            self.env_state = np.zeros((self.n_steps + 1, self.n_rollout_threads, *state_shape), dtype=np.float32)
            if self.use_rnn:
                self.env_critic_rnn = np.zeros(
                    (self.n_steps + 1, self.n_rollout_threads, self.num_rnn_layers, self.hidden_size),
                    dtype=np.float32)

        # Initialize available actions buffer if shape is Discrete
        if action_space.__class__.__name__ == "Discrete":
            self.available_actions = np.ones((self.n_steps + 1, self.n_rollout_threads, n_agents, action_space.n),
                                             dtype=np.float32)
        else:
            self.available_actions = None

        # RNN hidden states
        if self.use_rnn:
            # Shape: [n_steps + 1, n_rollout_threads, n_agents, num_layers, hidden_size]
            self.actor_rnn_states = np.zeros(
                (self.n_steps + 1, self.n_rollout_threads, n_agents, self.num_rnn_layers, self.hidden_size),
                dtype=np.float32)
        else:
            self.actor_rnn_states = None

        # Extra buffers for the algorithm
        self.returns = np.zeros((self.n_steps + 1, self.n_rollout_threads, n_agents, 1), dtype=np.float32)
        self.advantages = np.zeros((self.n_steps, self.n_rollout_threads, n_agents, 1), dtype=np.float32)

    def for_agent(self, idx: int):
        """
        Return a view of the rollout storage for a specific agent.

        Args:
            idx (int): Index of the agent to create view for, must be in range [0, n_agents)

        Returns:
            AgentRolloutView: Zero-copy view of the storage data for the specified agent
        """
        from buffers.agent_rollout_view import AgentRolloutView

        if not 0 <= idx < self.n_agents:
            raise IndexError(f"Agent index {idx} out of range [0, {self.n_agents})")
        return AgentRolloutView(self, idx)

    def get_state(self, t=slice(None), *, replicate=False):
        """
        Get state at time t, optionally replicating for each agent.

        Args:
            t : int | slice | ndarray
                Time index/indices (0 … T).  Default 'slice(None)' = all steps.
            replicate : bool
                If True, broadcast central state to shape (..., N_agents, feat_dim).

        Returns:
            np.ndarray
                FP / AS         → shape (..., N_env, N_agents, feat)
                EP, replicate   → shape (..., N_env, N_agents, feat)
                EP, no rep      → shape (..., N_env, feat)
        """
        # ---------- centralised state ------------------------------------------
        if hasattr(self, "env_state"):  # EP
            s = self.env_state[t]  # Shape depends on t: (N,feat) if t is int, (T,N,feat) if t is slice
            if not replicate:
                return s  # 2-D or 3-D, zero-copy slice

            # add a size-1 'agent' axis right before the feature dim
            s = np.expand_dims(s, axis=-2)  # (N,1,F) or (T,N,1,F)
            # broadcast that axis to n_agents WITHOUT materialising copies
            shape = s.shape[:-2] + (self.n_agents, s.shape[-1])
            return np.broadcast_to(s, shape)  # view, zero-copy
        # ---------- agent-specific state ---------------------------------------
        return self.agent_state[t]  # already (T,N,M,F) or (N,M,F) if t is int

    def get_critic_rnn(self, t=slice(None), *, replicate=False):
        """Get critic RNN states at time t, optionally replicating for each agent.

        Args:
            t : int | slice | ndarray
                Time index/indices (0 … T).  Default 'slice(None)' = all steps.
            replicate : bool
                If True, broadcast central state to shape (..., N_agents, feat_dim).

        Returns:
            np.ndarray
                FP / AS         → shape (..., N_env, N_agents, layers, feat)
                EP, replicate   → shape (..., N_env, N_agents, layers, feat)
                EP, no rep      → shape (..., N_env, layers, feat)
        """
        if not self.use_rnn:  # Early exit if not using RNN
            return None
        # ---------- centralised state ------------------------------------------
        if hasattr(self, "env_critic_rnn"):
            h = self.env_critic_rnn[t]  # Shape depends on t: (N,L,H) if t is int, (T,N,L,H) if t is slice
            if not replicate:
                return h  # 3-D or 4-D, zero-copy slice

            # add a size-1 'agent' axis right before the layers dim
            h = np.expand_dims(h, axis=-3)  # (N,1,L,H) or (T,N,1,L,H)
            # broadcast that axis to n_agents WITHOUT materialising copies
            shape = h.shape[:-3] + (self.n_agents, h.shape[-2], h.shape[-1])
            return np.broadcast_to(h, shape)  # view, zero-copy

        # ---------- agent-specific state ---------------------------------------
        return self.critic_rnn_states[t]  # (T,N,M,L,H) or (N,M,L,H) if t is int

    def insert(self, obs, global_state, actions,
               action_log_probs, values, rewards,
               masks, truncates, actor_rnn_states=None, critic_rnn_states=None,
               active_masks=None, available_actions=None):
        """
        Insert a new transition into the buffer.

        Args:
            obs: Agent observations [n_rollout_threads, n_agents, obs_shape]
            global_state: Global state if available [n_rollout_threads, n_agents, state_shape]
            actions: Actions taken by agents [n_rollout_threads, n_agents, action_shape]
            action_log_probs: Log probs of actions [n_rollout_threads, n_agents, 1]
            values: Value predictions [n_rollout_threads, n_agents, 1]
            rewards: Rewards received [n_rollout_threads, n_agents, 1]
            masks: Episode termination masks [n_rollout_threads, n_agents, 1], 0 if episode done, 1 otherwise
            truncates: Boolean array indicating if episode was truncated (e.g., due to time limit)
                      rather than terminated [n_rollout_threads, n_agents, 1]
            actor_rnn_states: RNN states [n_rollout_threads, num_layers, n_agents, hidden_size]
            critic_rnn_states: RNN states [n_rollout_threads, num_layers, n_agents, hidden_size]
            active_masks: Agent active masks [n_rollout_threads, n_agents, 1], 0 if agent dead, 1 otherwise
            available_actions: Available actions mask [n_rollout_threads, n_agents, n_actions]
        """
        self.obs[self.step + 1] = obs.copy()

        # Store state based on type
        if self.is_fp_state:
            self.agent_state[
                self.step + 1] = global_state.copy()  # global_state is [n_rollout_threads, n_agents, state_dim]
        else:
            self.env_state[self.step + 1] = global_state.copy()  # global_state is [n_rollout_threads, state_dim]

        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.values[self.step] = values.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        self.truncated[self.step + 1] = truncates.copy()

        if self.use_rnn:
            if actor_rnn_states is not None:
                self.actor_rnn_states[self.step + 1] = actor_rnn_states.copy()

            # Store critic RNN states based on type
            if self.is_fp_state:
                # [n_rollout_threads, n_agents, n_layers, hidden_size]
                self.agent_critic_rnn[self.step + 1] = critic_rnn_states.copy()
            else:
                # For environment-central RNN states, take the first agent's RNN state
                # since all agents in the same environment have the same state
                # [n_rollout_threads, n_layers, hidden_size]
                self.env_critic_rnn[self.step + 1] = critic_rnn_states[:, 0].copy()

        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()

        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step += 1

    def compute_returns_and_advantages(self, next_values, gamma=0.99, lambda_=0.95, use_gae=True):
        """
        Compute returns and advantages using GAE (Generalized Advantage Estimation).
        Properly handles truncated episodes by incorporating next state values.

        Args:
            next_values: Value estimates for the next observations [n_rollout_threads, n_agents, 1]
            gamma: Discount factor
            lambda_: GAE lambda parameter for advantage weighting
            use_gae: Whether to use GAE or just n-step returns
        """
        # Set the value of the next observation
        self.values[-1] = next_values

        # Create arrays for storing returns and advantages
        advantages = np.zeros_like(self.rewards)
        returns = np.zeros_like(self.returns)

        if use_gae:
            # GAE advantage computation with vectorized operations for better performance
            gae = 0
            for step in reversed(range(self.n_steps)):
                # For truncated episodes, we adjust rewards directly
                adjusted_rewards = self.rewards[step].copy()  # [n_agents]

                # Identify truncated episodes (done but not terminated)
                truncated_mask = (self.masks[step + 1] == 0) & (self.truncated[step + 1] == 1)  # [n_agents]
                if np.any(truncated_mask):
                    # Add bootstrapped value only for truncated episodes
                    adjusted_rewards[truncated_mask] += gamma * self.values[step + 1][truncated_mask]

                # Calculate delta (TD error) with adjusted rewards
                delta = (
                        adjusted_rewards +
                        gamma * self.values[step + 1] * self.masks[step + 1] -
                        self.values[step]
                )  # [n_agents]

                # Standard GAE calculation
                gae = delta + gamma * lambda_ * self.masks[step + 1] * gae  # [n_agents]
                advantages[step] = gae  # [n_agents]

            # Compute returns as advantages + values
            returns[:-1] = advantages + self.values[:-1]  # [n_agents]
            returns[-1] = next_values  # [n_agents]

        else:
            # N-step returns without GAE (more efficient calculation)
            returns[-1] = next_values
            for step in reversed(range(self.n_steps)):
                # Adjust rewards for truncated episodes
                adjusted_rewards = self.rewards[step].copy()

                # Identify truncated episodes
                truncated_mask = (self.masks[step + 1] == 0) & (self.truncated[step + 1] == 1)

                # For truncated episodes, add discounted bootstrapped value directly to rewards
                if np.any(truncated_mask):
                    adjusted_rewards[truncated_mask] += gamma * returns[step + 1][truncated_mask]

                # Calculate returns with proper masking
                returns[step] = adjusted_rewards + gamma * returns[step + 1] * self.masks[step + 1]

            # Calculate advantages
            advantages = returns[:-1] - self.values[:-1]

        # Store results
        self.returns = returns

        # Normalize advantages (helps with training stability)
        # Use stable normalization with small epsilon
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        self.advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        return self.advantages, self.returns

    def after_update(self):
        """Copy the last observation and masks to the beginning for the next update."""
        self.obs[0] = self.obs[-1].copy()

        # Copy state based on type
        if hasattr(self, "env_state"):
            self.env_state[0] = self.env_state[-1].copy()
        else:
            self.agent_state[0] = self.agent_state[-1].copy()

        self.masks[0] = self.masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        self.truncated[0] = self.truncated[-1].copy()

        # Copy RNN states
        if self.use_rnn:
            self.actor_rnn_states[0] = self.actor_rnn_states[-1].copy()

            # Copy critic RNN states based on type
            if hasattr(self, "env_critic_rnn"):
                self.env_critic_rnn[0] = self.env_critic_rnn[-1].copy()
            else:
                self.agent_critic_rnn[0] = self.agent_critic_rnn[-1].copy()

        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

        # Reset step counter
        self.step = 0

    def get_minibatches(self, num_mini_batch, mini_batch_size=None):
        """
        Create minibatches for training - not RNN version

        Args:
            num_mini_batch (int): Number of minibatches to create
            mini_batch_size (int, optional): Size of each minibatch, if None will be calculated
                                            based on num_mini_batch

        Returns:
            Generator yielding minibatches for training
        """
        if self.use_rnn:
            raise ValueError("RNN is enabled, cannot use get_minibatches")

        batch_size = self.n_steps * self.n_rollout_threads * self.n_agents  # e.g., T * N * M = 400 * 8 * 5 = 16000

        if mini_batch_size is None:
            mini_batch_size = batch_size // num_mini_batch

        # Create random indices for minibatches
        batch_inds = np.random.permutation(batch_size)

        # Preshape data to improve performance (only do this once)
        # Batch size is [T, N, M, feat_dim] -> [T*N*M, feat_dim]
        data = {
            'obs':                  self.obs[:-1].reshape(-1, *self.obs.shape[3:]),
            'actions':              self.actions.reshape(-1, self.actions.shape[-1]),
            'values':               self.values[:-1].reshape(-1, 1),
            'returns':              self.returns[:-1].reshape(-1, 1),
            'masks':                self.masks[:-1].reshape(-1, 1),
            'active_masks':         self.active_masks[:-1].reshape(-1, 1),
            'old_action_log_probs': self.action_log_probs.reshape(-1, 1),
            'advantages':           self.advantages.reshape(-1, 1),
        }
        # Get state with replication if needed
        # This will handle both FP and non-FP states automatically
        state_data = self.get_state(slice(0, self.n_steps), replicate=True)
        data['global_state'] = state_data.reshape(-1, state_data.shape[-1])

        # Handle available actions specially
        if self.available_actions is not None:
            data['available_actions'] = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])

        # Yield minibatches
        start_ind = 0
        for _ in range(num_mini_batch):
            end_ind = min(start_ind + mini_batch_size, batch_size)
            if end_ind - start_ind < 1:  # Skip empty batches
                continue

            batch_inds_subset = batch_inds[start_ind:end_ind]

            batch = {
                key: to_tensor(data[key][batch_inds_subset], device=self.device, copy=True)
                for key in data.keys()
            }

            # Yield the minibatch as a tuple
            yield (
                batch['obs'],
                batch['global_state'],
                None,  # actor_rnn_states,
                None,  # critic_rnn_states,
                batch['actions'],
                batch['values'],
                batch['returns'],
                batch['masks'],
                batch['active_masks'],
                batch['old_action_log_probs'],
                batch['advantages'],
                batch['available_actions'] if 'available_actions' in batch else None  # Handle optional key
            )

            start_ind = end_ind

    def get_minibatches_seq_first(self, num_mini_batch=1, data_chunk_length=10):
        """
        Create minibatches for training RNN, flattening num_agents into total steps.
        Returns sequences with shape [seq_len*batch_size, feat_dim] and RNN states
        with shape [num_layers, batch_size, hidden_dim].

        Args:
            num_mini_batch (int): Number of minibatches to create
            data_chunk_length (int): Length of data chunk to use for training, default is 10

        Returns:
            Generator yielding minibatches as tuples with the following keys:
            - obs: [seq_len*batch_size, obs_dim]
            - global_state: [seq_len*batch_size, state_dim]
            - actor_rnn_states: [num_layers, batch_size, hidden_size]
            - critic_rnn_states: [num_layers, batch_size, hidden_size]
            - actions, values, returns, etc.: [seq_len*batch_size, dim]
        """
        if not self.use_rnn:
            raise ValueError("RNN is not enabled, cannot use get_minibatches_seq_first")

        total_steps = self.n_steps  # e.g., [T, N, M, feat_dim] -> T
        if total_steps < data_chunk_length:
            raise ValueError(f"n_steps ({total_steps}) must be >= data_chunk_length ({data_chunk_length})")

        # Calculate chunks and batch sizes
        rollout_threads = self.n_rollout_threads
        num_agents = self.n_agents
        total_agent_steps = total_steps * num_agents * rollout_threads  # e.g., T * M * N = 400 * 5 * 8 = 16000
        max_data_chunks = total_agent_steps // data_chunk_length  # e.g., 16000 // 10 = 1600

        # Adjust mini_batch count if needed
        num_mini_batch = min(num_mini_batch, max_data_chunks)  # e.g., 1
        mini_batch_size = max(1, max_data_chunks // num_mini_batch)  # e.g., 1600 // 1 = 1600

        # Pre-convert and flatten data, collapsing num_agents into the sequence
        # [T, N, M, feat_dim] -> [N*M*T, feat_dim]
        data = {
            'obs':                  _transform_data(self.obs[:-1], self.device),
            'actions':              _transform_data(self.actions, self.device),
            'values':               _transform_data(self.values[:-1], self.device),
            'returns':              _transform_data(self.returns[:-1], self.device),
            'masks':                _transform_data(self.masks[:-1], self.device),
            'active_masks':         _transform_data(self.active_masks[:-1], self.device),
            'old_action_log_probs': _transform_data(self.action_log_probs, self.device),
            'advantages':           _transform_data(self.advantages, self.device),
        }

        # Get state with replication if needed for all time steps
        # This will handle both FP and non-FP states automatically
        data['global_state'] = _transform_data(
            self.get_state(slice(0, self.n_steps), replicate=True),
            self.device)

        # Handle available actions specially
        if self.available_actions is not None:
            data['available_actions'] = _transform_data(self.available_actions[:-1], self.device)

        # Process RNN states - maintain num_layers dimension while flattening agents
        # [T, N, M, num_layers, hidden_size] -> [N, M, T, num_layers, hidden_size] -> [N*M*T, num_layers, hidden_size]
        actor_rnn_states = self.actor_rnn_states[:-1].transpose(1, 2, 0, 3, 4).reshape(-1,
                                                                                       self.actor_rnn_states.shape[-2],
                                                                                       self.actor_rnn_states.shape[-1])
        # This will handle both FP and non-FP states automatically
        critic_rnn_states = self.get_critic_rnn(slice(0, self.n_steps), replicate=True)
        critic_rnn_states = critic_rnn_states.transpose(1, 2, 0, 3, 4).reshape(-1,
                                                                               critic_rnn_states.shape[-2],
                                                                               critic_rnn_states.shape[-1])

        data['actor_rnn_states'] = to_tensor(actor_rnn_states, device=self.device, copy=True)
        data['critic_rnn_states'] = to_tensor(critic_rnn_states, device=self.device, copy=True)

        # Generate chunk start indices over total_agent_steps
        all_starts = np.arange(0, total_agent_steps - data_chunk_length + 1, data_chunk_length)
        if len(all_starts) > max_data_chunks:
            all_starts = all_starts[:max_data_chunks]
        np.random.shuffle(all_starts)

        # Generate minibatches
        for batch_start in range(0, len(all_starts), mini_batch_size):  # (0, 1600, 1600)
            batch_chunk_starts = all_starts[batch_start:batch_start + mini_batch_size]  # all_starts[0:1600]
            if not batch_chunk_starts.size:
                continue

            # Collect sequences
            sequences = {key: [] for key in data.keys()}

            for start_idx in batch_chunk_starts:
                end_idx = start_idx + data_chunk_length

                # Collect sequences for each data type
                for key in data.keys():
                    if key not in ['actor_rnn_states', 'critic_rnn_states']:
                        sequences[key].append(data[key][start_idx:end_idx])

                # Get initial RNN states
                sequences['actor_rnn_states'].append(data['actor_rnn_states'][start_idx])
                sequences['critic_rnn_states'].append(data['critic_rnn_states'][start_idx])

            stack_dims = {
                'actor_rnn_states':  0,
                'critic_rnn_states': 0,
            }

            # Stack sequences into proper shapes (seq_len, batch_size, feat_dim) or (batch_size, num_layers, hidden_dim)
            batch = {
                key: torch.stack(sequences[key], dim=stack_dims.get(key, 1))
                for key in sequences
            }

            T, N = data_chunk_length, mini_batch_size

            # Yield the minibatch as a tuple
            yield (
                flatten_time_batch(T, N, batch['obs']),
                flatten_time_batch(T, N, batch['global_state']),
                batch['actor_rnn_states'],
                batch['critic_rnn_states'],
                flatten_time_batch(T, N, batch['actions']),
                flatten_time_batch(T, N, batch['values']),
                flatten_time_batch(T, N, batch['returns']),
                flatten_time_batch(T, N, batch['masks']),
                flatten_time_batch(T, N, batch['active_masks']),
                flatten_time_batch(T, N, batch['old_action_log_probs']),
                flatten_time_batch(T, N, batch['advantages']),
                flatten_time_batch(T, N, batch['available_actions']) if 'available_actions' in batch else None
                # Handle optional key
            )


class RolloutStorageSRMT(RolloutStorage):
    def __init__(self, args, n_agents, obs_space, action_space, state_space, device='cpu'):
        super().__init__(args, n_agents, obs_space, action_space, state_space, device)

        self.obs_shape = get_shape_from_obs_space(obs_space)

        # SRMT specific params
        self.srmt_core = args.srmt_core
        self.use_agent_memory = args.use_agent_memory
        self.use_global_memory = args.use_global_memory
        self.data_chunk_length = args.data_chunk_length

        self._init_srmt_memory()

    def _init_srmt_memory(self):
        # SRMT specific storage buffer
        if self.srmt_core:
            self.history_seq = np.zeros(
                (self.n_steps + 1, self.n_rollout_threads, self.n_agents, self.data_chunk_length, self.hidden_size),
                dtype=np.float32)
            if self.use_agent_memory:
                self.agent_memory = np.zeros((self.n_steps + 1, self.n_rollout_threads, self.n_agents, self.hidden_size),
                                             dtype=np.float32)
            if self.use_global_memory:
                self.global_memory = np.zeros(
                    (self.n_steps + 1, self.n_rollout_threads, self.n_agents, self.n_agents, self.hidden_size),
                    dtype=np.float32)

    def after_update(self):
        super().after_update()
        self._init_srmt_memory()

    def get_minibatches(self, num_mini_batch, mini_batch_size=None):
        """
        Create minibatches for training - not RNN version

        Args:
            num_mini_batch (int): Number of minibatches to create
            mini_batch_size (int, optional): Size of each minibatch, if None will be calculated
                                            based on num_mini_batch

        Returns:
            Generator yielding minibatches for training
        """
        if self.use_rnn:
            raise ValueError("RNN is enabled, cannot use get_minibatches")

        batch_size = self.n_steps * self.n_rollout_threads * self.n_agents  # e.g., T * N * M = 400 * 8 * 5 = 16000

        if mini_batch_size is None:
            mini_batch_size = batch_size // num_mini_batch

        # Create random indices for minibatches
        batch_inds = np.random.permutation(batch_size)

        # Preshape data to improve performance (only do this once)
        # Batch size is [T, N, M, feat_dim] -> [T*N*M, feat_dim]
        data = {
            'obs':                  self.obs[:-1].reshape(-1, *self.obs.shape[3:]),
            'actions':              self.actions.reshape(-1, self.actions.shape[-1]),
            'values':               self.values[:-1].reshape(-1, 1),
            'returns':              self.returns[:-1].reshape(-1, 1),
            'masks':                self.masks[:-1].reshape(-1, 1),
            'active_masks':         self.active_masks[:-1].reshape(-1, 1),
            'old_action_log_probs': self.action_log_probs.reshape(-1, 1),
            'advantages':           self.advantages.reshape(-1, 1),
        }
        # Get state with replication if needed
        # This will handle both FP and non-FP states automatically
        state_data = self.get_state(slice(0, self.n_steps), replicate=True)
        data['global_state'] = state_data.reshape(-1, state_data.shape[-1])

        # Handle SRMT specific params separately
        if self.srmt_core:
            data['history_seq'] = self.history_seq[:-1].reshape(-1, self.data_chunk_length, self.hidden_size)
            if self.use_agent_memory:
                data['agent_memory'] = self.agent_memory[:-1].reshape(-1, self.hidden_size)
            if self.use_global_memory:
                data['global_memory'] = self.global_memory[:-1].reshape(-1, self.n_agents, self.hidden_size)

        # Handle available actions specially
        if self.available_actions is not None:
            data['available_actions'] = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])

        # Yield minibatches
        start_ind = 0
        for _ in range(num_mini_batch):
            end_ind = min(start_ind + mini_batch_size, batch_size)
            if end_ind - start_ind < 1:  # Skip empty batches
                continue

            batch_inds_subset = batch_inds[start_ind:end_ind]

            batch = {
                key: to_tensor(data[key][batch_inds_subset], device=self.device, copy=True)
                for key in data.keys()
            }

            # Yield the minibatch as a tuple
            yield (
                batch['obs'],
                batch['global_state'],
                None,  # actor_rnn_states,
                None,  # critic_rnn_states,
                batch['actions'],
                batch['values'],
                batch['returns'],
                batch['masks'],
                batch['active_masks'],
                batch['old_action_log_probs'],
                batch['advantages'],
                batch['available_actions'] if 'available_actions' in batch else None,  # Handle optional key
                batch['history_seq'] if 'history_seq' in batch else None,
                batch['agent_memory'] if 'agent_memory' in batch else None,
                batch['global_memory'] if 'global_memory' in batch else None,
            )

            start_ind = end_ind

    def insert(self,
               obs,
               global_state,
               actions,
               action_log_probs,
               values,
               rewards,
               masks,
               truncates,
               actor_rnn_states=None,
               critic_rnn_states=None,
               active_masks=None,
               available_actions=None,
               history_seq=None,
               agent_memory=None,
               global_memory=None,
               ):


        self.history_seq[self.step + 1] = history_seq.copy()
        self.agent_memory[self.step + 1] = agent_memory.copy()
        self.global_memory[self.step + 1] = global_memory.copy()

        super().insert(obs,
                       global_state,
                       actions,
                       action_log_probs,
                       values,
                       rewards,
                       masks,
                       truncates,
                       actor_rnn_states,
                       critic_rnn_states,
                       active_masks,
                       available_actions)

    def get_minibatches_seq_first(self, num_mini_batch=1, data_chunk_length=10):
        """
        Create minibatches for training RNN, flattening num_agents into total steps.
        Returns sequences with shape [seq_len*batch_size, feat_dim] and RNN states
        with shape [num_layers, batch_size, hidden_dim].

        Args:
            num_mini_batch (int): Number of minibatches to create
            data_chunk_length (int): Length of data chunk to use for training, default is 10

        Returns:
            Generator yielding minibatches as tuples with the following keys:
            - obs: [seq_len*batch_size, obs_dim]
            - global_state: [seq_len*batch_size, state_dim]
            - actor_rnn_states: [num_layers, batch_size, hidden_size]
            - critic_rnn_states: [num_layers, batch_size, hidden_size]
            - actions, values, returns, etc.: [seq_len*batch_size, dim]
        """
        if not self.use_rnn:
            raise ValueError("RNN is not enabled, cannot use get_minibatches_seq_first")

        total_steps = self.n_steps  # e.g., [T, N, M, feat_dim] -> T
        if total_steps < data_chunk_length:
            raise ValueError(f"n_steps ({total_steps}) must be >= data_chunk_length ({data_chunk_length})")

        # Calculate chunks and batch sizes
        rollout_threads = self.n_rollout_threads
        num_agents = self.n_agents
        total_agent_steps = total_steps * num_agents * rollout_threads  # e.g., T * M * N = 400 * 5 * 8 = 16000
        max_data_chunks = total_agent_steps // data_chunk_length  # e.g., 16000 // 10 = 1600

        # Adjust mini_batch count if needed
        num_mini_batch = min(num_mini_batch, max_data_chunks)  # e.g., 1
        mini_batch_size = max(1, max_data_chunks // num_mini_batch)  # e.g., 1600 // 1 = 1600

        # Pre-convert and flatten data, collapsing num_agents into the sequence
        # [T, N, M, feat_dim] -> [N*M*T, feat_dim]
        data = {
            'obs':                  _transform_data(self.obs[:-1], self.device),
            'actions':              _transform_data(self.actions, self.device),
            'values':               _transform_data(self.values[:-1], self.device),
            'returns':              _transform_data(self.returns[:-1], self.device),
            'masks':                _transform_data(self.masks[:-1], self.device),
            'active_masks':         _transform_data(self.active_masks[:-1], self.device),
            'old_action_log_probs': _transform_data(self.action_log_probs, self.device),
            'advantages':           _transform_data(self.advantages, self.device),
        }

        # Get state with replication if needed for all time steps
        # This will handle both FP and non-FP states automatically
        data['global_state'] = _transform_data(
            self.get_state(slice(0, self.n_steps), replicate=True),
            self.device)

        # Hande SRMT specific params separately
        if self.srmt_core:
            data['history_seq'] = _transform_data(self.history_seq[:-1], self.device)
            if self.use_agent_memory:
                data['agent_memory'] = _transform_data(self.agent_memory[:-1], self.device)
            if self.use_global_memory:
                data['global_memory'] = _transform_data(self.global_memory[:-1], self.device)

        # Handle available actions specially
        if self.available_actions is not None:
            data['available_actions'] = _transform_data(self.available_actions[:-1], self.device)

        # Process RNN states - maintain num_layers dimension while flattening agents
        # [T, N, M, num_layers, hidden_size] -> [N, M, T, num_layers, hidden_size] -> [N*M*T, num_layers, hidden_size]
        actor_rnn_states = self.actor_rnn_states[:-1].transpose(1, 2, 0, 3, 4).reshape(-1,
                                                                                       self.actor_rnn_states.shape[-2],
                                                                                       self.actor_rnn_states.shape[-1])
        # This will handle both FP and non-FP states automatically
        critic_rnn_states = self.get_critic_rnn(slice(0, self.n_steps), replicate=True)
        critic_rnn_states = critic_rnn_states.transpose(1, 2, 0, 3, 4).reshape(-1,
                                                                               critic_rnn_states.shape[-2],
                                                                               critic_rnn_states.shape[-1])

        data['actor_rnn_states'] = to_tensor(actor_rnn_states, device=self.device, copy=True)
        data['critic_rnn_states'] = to_tensor(critic_rnn_states, device=self.device, copy=True)

        # Generate chunk start indices over total_agent_steps
        all_starts = np.arange(0, total_agent_steps - data_chunk_length + 1, data_chunk_length)
        if len(all_starts) > max_data_chunks:
            all_starts = all_starts[:max_data_chunks]
        np.random.shuffle(all_starts)

        # Generate minibatches
        for batch_start in range(0, len(all_starts), mini_batch_size):  # (0, 1600, 1600)
            batch_chunk_starts = all_starts[batch_start:batch_start + mini_batch_size]  # all_starts[0:1600]
            if not batch_chunk_starts.size:
                continue

            # Collect sequences
            sequences = {key: [] for key in data.keys()}

            for start_idx in batch_chunk_starts:
                end_idx = start_idx + data_chunk_length

                # Collect sequences for each data type
                for key in data.keys():
                    if key not in ['actor_rnn_states', 'critic_rnn_states']:
                        sequences[key].append(data[key][start_idx:end_idx])

                # Get initial RNN states
                sequences['actor_rnn_states'].append(data['actor_rnn_states'][start_idx])
                sequences['critic_rnn_states'].append(data['critic_rnn_states'][start_idx])

            stack_dims = {
                'actor_rnn_states':  0,
                'critic_rnn_states': 0,
            }

            # Stack sequences into proper shapes (seq_len, batch_size, feat_dim) or (batch_size, num_layers, hidden_dim)
            batch = {
                key: torch.stack(sequences[key], dim=stack_dims.get(key, 1))
                for key in sequences
            }

            T, N = data_chunk_length, mini_batch_size

            # Yield the minibatch as a tuple
            yield (
                flatten_time_batch(T, N, batch['obs']),
                flatten_time_batch(T, N, batch['global_state']),
                batch['actor_rnn_states'],
                batch['critic_rnn_states'],
                flatten_time_batch(T, N, batch['actions']),
                flatten_time_batch(T, N, batch['values']),
                flatten_time_batch(T, N, batch['returns']),
                flatten_time_batch(T, N, batch['masks']),
                flatten_time_batch(T, N, batch['active_masks']),
                flatten_time_batch(T, N, batch['old_action_log_probs']),
                flatten_time_batch(T, N, batch['advantages']),
                # Handle optional key
                flatten_time_batch(T, N, batch['available_actions']) if 'available_actions' in batch else None,
                flatten_time_batch(T, N, batch['history_seq']) if 'history_seq' in batch else None,
                flatten_time_batch(T, N, batch['agent_memory']) if 'agent_memory' in batch else None,
                flatten_time_batch(T, N, batch['global_memory']) if 'global_memory' in batch else None,
            )

