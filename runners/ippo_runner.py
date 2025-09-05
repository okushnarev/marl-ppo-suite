from functools import partial

import numpy as np

from algos.ippo import IPPO, IPPO_SRMT
from runners.mappo_runner import MAPPORunner, MAPPO_SRMTRunner
from utils.reward_normalization_new import normalise_shared_reward
from utils.transform_tools import flatten_first_dims, to_tensor, unflatten_first_dim


class IPPORunner(MAPPORunner):
    def __init__(self, args, device):
        super().__init__(args, device)

        # Create agent
        self.agent = IPPO(args,
                          self.envs.observation_space,
                          self.envs.share_observation_space,
                          self.envs.action_space,
                          self.device)


class IPPO_SRMTRunner(MAPPO_SRMTRunner):
    def __init__(self, args, device):
        super().__init__(args, device)

        # Create agent
        self.agent = IPPO_SRMT(args,
                               self.envs.observation_space,
                               self.envs.share_observation_space,
                               self.envs.action_space,
                               self.device)

    def prep_data(self, data, step):
        return flatten_first_dims(
            to_tensor(data[step], device=self.device)
        ) if data is not None else None

    def collect_rollouts(self):
        """
        Collect trajectories by interacting with the environment.

        Returns:
            np.ndarray: Information from the last step of the rollout
        """
        # Start timing for rollout collection if performance metrics are enabled
        rollout_data = {
            'episode_lengths': [],
            'episode_rewards': []
        }

        for step in range(self.args.n_steps):
            # Get actions and values

            results = self.agent.get_actions_values(
                *map(partial(self.prep_data, step=step),
                     (
                         self.buffer.obs,
                         self.buffer.actor_rnn_states if self.args.use_rnn else None,
                         self.buffer.ctitic_rnn_states if self.args.use_rnn else None,
                         self.buffer.masks,
                         self.buffer.available_actions,
                         self.buffer.history_seq,
                         self.buffer.agent_memory,
                         self.buffer.global_memory,
                     )
                     ),
                deterministic=False
            )
            actions_t = results['actions']
            action_log_probs_t = results['action_log_probs']
            actor_rnn_states_t = results['actor_rnn_states_out']
            history_seq_t = results['history_seq']
            agent_memory_t = results['agent_memory']
            global_memory_t = results['global_memory']
            values_t = results['values']
            critic_rnn_states_t = results['critic_rnn_states_out']

            # Reshape actions and values
            shape = (self.args.n_rollout_threads, self.envs.n_agents)
            actions = unflatten_first_dim(actions_t, shape).cpu().numpy()
            action_log_probs = unflatten_first_dim(action_log_probs_t, shape).cpu().numpy()
            values = unflatten_first_dim(values_t, shape).cpu().numpy()

            # Reshape RNN states if using RNN
            actor_rnn_states = None
            critic_rnn_states = None
            if self.args.use_rnn:
                critic_rnn_states = unflatten_first_dim(critic_rnn_states_t, shape).cpu().numpy()

            # Execute actions in environment
            obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)
            # obs: (n_threads, n_agents, obs_dim)
            # share_obs: (n_threads, n_agents, share_obs_dim)
            # rewards: (n_threads, n_agents, 1)
            # dones: (n_threads, n_agents)
            # infos: (n_threads)
            # available_actions: None or (n_threads, n_agents, action_number)

            # Unflatten SRMT specific data
            history_seq = None
            agent_memory = None
            global_memory = None
            if self.args.srmt_core:
                history_seq = unflatten_first_dim(history_seq_t, shape).cpu().numpy()
                if self.args.use_agent_memory:
                    agent_memory = unflatten_first_dim(agent_memory_t, shape).cpu().numpy()
                if self.args.use_global_memory:
                    global_memory = unflatten_first_dim(global_memory_t, shape).cpu().numpy()

            # Freeze memory for dead agents
            dead_agents = np.array([info.get('dead_agents', [0] * self.args.n_agents) for info in infos])
            if dead_agents.any():
                dead_agents_idx = np.where(dead_agents == 1)
                if self.args.srmt_core:
                    history_seq[dead_agents_idx] = self.buffer.history_seq[step][dead_agents_idx]
                    if self.args.use_agent_memory:
                        agent_memory[dead_agents_idx] = self.buffer.agent_memory[step][dead_agents_idx]
                    if self.args.use_global_memory:
                        global_memory[dead_agents_idx] = self.buffer.global_memory[step][dead_agents_idx]

            # Update episode stats
            self.episode_length += 1
            self.episode_rewards += rewards[:, 0, 0]

            # Normalize rewards if enabled
            if self.args.use_reward_norm:
                rewards = normalise_shared_reward(rewards, self.reward_norm)

            # Handle episode termination
            done_envs = np.all(dones, axis=1)
            if np.any(done_envs):
                # self._check_episode_outcome(done_envs, self.total_steps + step*self.args.n_rollout_threads)
                done_indices = np.where(done_envs)[0]
                rollout_data['episode_lengths'].extend(self.episode_length[done_indices].tolist())
                rollout_data['episode_rewards'].extend(self.episode_rewards[done_indices].tolist())
                self.episode_length[done_indices] = 0
                self.episode_rewards[done_indices] = 0

            # Insert collected data
            data = (
                obs, share_obs, rewards, dones,
                infos, available_actions, values, actions,
                action_log_probs, actor_rnn_states, critic_rnn_states, history_seq, agent_memory,
                global_memory,
            )
            self.insert(data)

        return infos, rollout_data

    def compute_returns(self):
        """
        Compute returns and advantages for the collected trajectories.
        """
        next_value, _ = self.agent.get_values(
            flatten_first_dims(
                to_tensor(self.buffer.get_state(-1, replicate=True), device=self.device)
            ),
            flatten_first_dims(
                to_tensor(self.buffer.obs[-1], device=self.device)
            ),
            flatten_first_dims(
                to_tensor(self.buffer.active_masks[-1], device=self.device)
            ),
            flatten_first_dims(
                to_tensor(self.buffer.get_critic_rnn(-1, replicate=True), device=self.device)
            ) if self.args.use_rnn else None,
            flatten_first_dims(
                to_tensor(self.buffer.masks[-1], device=self.device)
            ),
            flatten_first_dims(
                to_tensor(self.buffer.history_seq[-1], device=self.device)
            ) if self.buffer.history_seq is not None else None,
            flatten_first_dims(
                to_tensor(self.buffer.agent_memory[-1], device=self.device)
            ) if self.buffer.agent_memory is not None else None,
            flatten_first_dims(
                to_tensor(self.buffer.global_memory[-1], device=self.device)
            ) if self.buffer.global_memory is not None else None,
        )

        self.buffer.compute_returns_and_advantages(
            unflatten_first_dim(
                next_value,
                (self.args.n_rollout_threads, self.envs.n_agents)
            ).cpu().numpy(),
            self.args.gamma,
            self.args.gae_lambda
        )
