import os
import time
import imageio
import numpy as np
import torch

from buffers.rollout_storage import RolloutStorage, RolloutStorageSRMT
from envs import make_vec_envs
from algos.mappo import MAPPO, MAPPO_SRMT
from utils.env_tools import get_shape_from_obs_space

from utils.logger import Logger
from utils.reward_normalization_new import StandardNormalizer, EMANormalizer, normalise_shared_reward
from utils.transform_tools import flatten_first_dims, unflatten_first_dim, to_tensor
from utils.video_utils import save_video, get_latest_sc2_replay


class MAPPORunner:
    """
    Runner class to handle environment interactions and training for MAPPO with agent-specific states.

    This class manages the environment with agent-specific states, agent, buffer, and training process,
    collecting trajectories and updating the policy.
    """

    def __init__(self, args, device):
        """
        Initialize the runner.

        Args:
            args: Arguments containing training parameters
            device: Torch device to use for computations
        """
        self.args = args
        self.device = device

        self.is_train = args.mode == "train"
        need_eval = args.mode in ("train", "eval")

        # Create training environment using the factory function
        self.envs = make_vec_envs(args, is_eval=not self.is_train,
                                  num_processes=1 if args.mode == "render"
                                  else args.n_rollout_threads)

        if need_eval:
            self.eval_envs = make_vec_envs(args, is_eval=True,
                                           num_processes=args.n_eval_rollout_threads)

        # Store args for creating evaluation environment later
        self.args = args
        self.args.n_agents = self.envs.n_agents

        # Get environment info
        print(f"n_agents: {self.envs.n_agents}, "
              f"action_space: {self.envs.action_space}, "
              f"state_space: {self.envs.share_observation_space}, "
              f"obs_space: {self.envs.observation_space}, "
              f"episode_limit: {self.envs.episode_limit}")

        # Create agent
        self.agent = MAPPO(args,
                           self.envs.observation_space,
                           self.envs.share_observation_space,
                           self.envs.action_space,
                           self.device)

        if self.is_train:

            # Create buffer
            self.buffer = RolloutStorage(
                args,
                self.envs.n_agents,
                self.envs.observation_space,
                self.envs.action_space,
                self.envs.share_observation_space,
                self.device)

            # Normalize rewards
            if args.use_reward_norm:
                if args.reward_norm_type == "ema":
                    self.reward_norm = EMANormalizer(clip=None)
                else:
                    self.reward_norm = StandardNormalizer(clip=None)

            # Initialize logger
            run_name = (
                f"lr{args.lr}_nenvs{args.n_rollout_threads}_nsteps{args.n_steps}_"
                f"minibatch{args.num_mini_batch}_epochs{args.ppo_epoch}_"
                f"gamma{args.gamma}_gae{args.gae_lambda}_"
                f"clip{args.clip_param}_state{args.state_type}_"
                f"aid{args.use_agent_id}_dmask{args.use_death_masking}_"
                f"rnn{args.use_rnn}_{int(time.time())}"
            )

            run_name = "".join(run_name)
            env_name = args.env_name + "_" + args.map_name
            hyperparams = vars(args)
            self.logger = Logger(
                run_name=run_name,
                env=env_name,
                algo=args.algo.upper(),
                use_wandb=args.use_wandb,
                config=hyperparams)

    def run(self):
        """
        Run the training process.
        """
        # Training stats
        self.total_steps = 0
        self.best_win_rate = 0
        self.last_battles_game = np.zeros((self.args.n_rollout_threads), dtype=np.float32)
        self.last_battles_won = np.zeros((self.args.n_rollout_threads), dtype=np.float32)
        self.episode_rewards = np.zeros((self.args.n_rollout_threads), dtype=np.float32)
        self.episode_length = np.zeros((self.args.n_rollout_threads), dtype=np.float32)
        evaluate_num = -1
        capture_num = -1
        log_num = 0

        # Warmup
        self.warmup()

        # Training loop
        while self.total_steps < self.args.max_steps:
            # Decay learning rate
            if self.args.use_linear_lr_decay:
                self.agent.update_learning_rate(self.total_steps)

            # Evaluate agent
            if self.total_steps // self.args.eval_interval > evaluate_num:
                should_capture = (self.total_steps // self.args.capture_video_interval > capture_num)
                self.evaluate(self.args.eval_episodes, should_capture and self.args.capture_video)
                evaluate_num += 1
                if should_capture:
                    capture_num += 1

            # Collect trajectories
            last_infos, rollout_data = self.collect_rollouts()
            self.total_steps += self.args.n_steps * self.args.n_rollout_threads

            # Compute returns and advantages
            self.compute_returns()

            # Train agent
            train_info = self.agent.train(self.buffer)

            # Log training information
            if self.total_steps // self.args.log_interval > log_num:
                self._log_rollout_outcome(last_infos, rollout_data, train_info, self.total_steps)
                log_num += 1

            # Reset buffer for next rollout
            self.buffer.after_update()

            # Final evaluation
        if self.args.use_eval:
            print(f"Final evaluation at {self.total_steps}/{self.args.max_steps}")
            self.evaluate(self.args.eval_episodes)

        # Save final model
        save_path = os.path.join(self.logger.dir_name, f"final-torch.model")
        self.agent.save(save_path)
        self.logger.log_model(
            file_path=save_path,
            name="final-model",
            artifact_type="model",
            metadata={"step": self.total_steps},
            alias="latest"
        )
        print(f"Saved final model to {save_path}")

        self.envs.close()
        self.eval_envs.close()
        self.logger.close()

    def warmup(self):
        """
        Warmup the agent.
        """
        # Reset environment
        obs, states, available_actions = self.envs.reset()

        # Store initial observations and states
        self.buffer.obs[0] = np.array(obs)  # (rollout_threads, n_agents, obs_shape)

        # Store state based on type
        if hasattr(self.buffer, "env_state"):
            self.buffer.env_state[0] = np.array(states)  # (rollout_threads, state_shape)
        else:
            self.buffer.agent_state[0] = np.array(states)  # (rollout_threads, n_agents, state_shape)

        self.buffer.available_actions[0] = np.array(available_actions)  # (rollout_threads, n_agents, n_actions)

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
            actions_t, action_log_probs_t, actor_rnn_states_t = self.agent.get_actions(
                flatten_first_dims(
                    to_tensor(self.buffer.obs[step], device=self.device)
                ),
                flatten_first_dims(
                    to_tensor(self.buffer.actor_rnn_states[step], device=self.device)
                ) if self.args.use_rnn else None,
                flatten_first_dims(
                    to_tensor(self.buffer.active_masks[step], device=self.device)
                ),
                flatten_first_dims(
                    to_tensor(self.buffer.available_actions[step], device=self.device)
                ) if self.buffer.available_actions is not None else None,
                deterministic=False
            )

            values_t, critic_rnn_states_t = self.agent.get_values(
                flatten_first_dims(
                    to_tensor(self.buffer.get_state(step, replicate=True), device=self.device)
                ),
                flatten_first_dims(
                    to_tensor(self.buffer.obs[step], device=self.device)
                ),
                flatten_first_dims(
                    to_tensor(self.buffer.active_masks[step], device=self.device)
                ),
                flatten_first_dims(
                    to_tensor(self.buffer.get_critic_rnn(step, replicate=True), device=self.device)
                ) if self.args.use_rnn else None,
                flatten_first_dims(
                    to_tensor(self.buffer.masks[step], device=self.device)
                )
            )

            # Reshape actions and values
            shape = (self.args.n_rollout_threads, self.envs.n_agents)
            actions = unflatten_first_dim(actions_t, shape).cpu().numpy()
            action_log_probs = unflatten_first_dim(action_log_probs_t, shape).cpu().numpy()
            values = unflatten_first_dim(values_t, shape).cpu().numpy()

            # Reshape RNN states if using RNN
            if self.args.use_rnn:
                actor_rnn_states = unflatten_first_dim(actor_rnn_states_t, shape).cpu().numpy()
                critic_rnn_states = unflatten_first_dim(critic_rnn_states_t, shape).cpu().numpy()
            else:
                actor_rnn_states = None
                critic_rnn_states = None

            # Execute actions in environment
            obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)
            # obs: (n_threads, n_agents, obs_dim)
            # share_obs: (n_threads, n_agents, share_obs_dim)
            # rewards: (n_threads, n_agents, 1)
            # dones: (n_threads, n_agents)
            # infos: (n_threads)
            # available_actions: None or (n_threads, n_agents, action_number)

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
                action_log_probs, actor_rnn_states, critic_rnn_states,
            )
            self.insert(data)

        return infos, rollout_data

    def insert(self, data):
        """
        Insert a new transition into the buffer.

        Args:
           data (tuple): Transition data containing:
            - obs: Agent observations (n_rollout_threads, n_agents, obs_dim)
            - share_obs: Shared observations (n_rollout_threads, n_agents, share_obs_dim)
            - rewards: Agent rewards (n_rollout_threads, n_agents, 1)
            - dones: Done flags (n_rollout_threads, n_agents)
            - infos: Environment info dicts [n_rollout_threads]
            - available_actions: Available actions mask (n_rollout_threads, n_agents, action_dim) or None
            - values: Value estimates (n_rollout_threads, n_agents, 1)
            - actions: Taken actions (n_rollout_threads, n_agents, action_dim)
            - action_log_probs: Action log probs (n_rollout_threads, n_agents, action_dim)
            - actor_rnn_states: Actor RNN states (n_rollout_threads, n_agents, num_layers, hidden_size)
            - critic_rnn_states: Critic RNN states (n_rollout_threads, n_agents, num_layers, hidden_size)
        """
        # Unpack transition data
        (obs, share_obs, rewards, dones, infos, available_actions,
         values, actions, action_log_probs, actor_rnn_states, critic_rnn_states) = data

        # Handle episode terminations
        done_envs = np.all(dones, axis=1)  # Check which environments are done
        # n_done_envs = done_envs.sum()
        done_env_mask = done_envs == True

        # Reset RNN states for done environments
        if self.args.use_rnn:
            actor_rnn_states[done_env_mask] = 0.0
            critic_rnn_states[done_env_mask] = 0.0

        # Create masks
        shape = (self.args.n_rollout_threads, self.envs.n_agents, 1)
        masks = np.ones(shape, dtype=np.float32)
        active_masks = np.ones(shape, dtype=np.float32)

        # Update masks for done environments and agents
        masks[done_env_mask, :, :] = 0.0  # broadcast across agents and last dim
        active_masks[dones, :] = 0.0  # for individual agent deaths
        active_masks[done_env_mask, :, :] = 1.0  # for full environment termination

        # Create truncation masks from environment infos
        truncates = np.array([[info['truncated']] for info in infos])  # (n_rollout_threads, 1)
        truncates = np.repeat(truncates, self.envs.n_agents, axis=1).reshape(*shape)  # (n_rollout_threads, n_agents, 1)

        # Store trajectory in buffer
        self.buffer.insert(
            obs=obs,  # (n_rollout_threads, n_agents, n_obs)
            global_state=share_obs,  # (n_rollout_threads, n_agents, n_state) or (n_rollout_threads, n_state)
            actions=actions,  # (n_rollout_threads, n_agents, 1)
            action_log_probs=action_log_probs,  # (n_rollout_threads, n_agents, 1)
            values=values,  # (n_rollout_threads, n_agents, 1)
            rewards=rewards,  # (n_rollout_threads, n_agents, 1)
            masks=masks,  # (n_rollout_threads, n_agents, 1)
            active_masks=active_masks,  # (n_rollout_threads, n_agents, 1)
            truncates=truncates,  # (n_rollout_threads, n_agents, 1)
            available_actions=available_actions,  # (n_rollout_threads, n_agents, n_actions) or None
            actor_rnn_states=actor_rnn_states,  # (n_rollout_threads, n_agents, num_layers, hidden_size)
            critic_rnn_states=critic_rnn_states,  # (n_rollout_threads, n_agents, num_layers, hidden_size)
        )

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
            )
        )

        self.buffer.compute_returns_and_advantages(
            unflatten_first_dim(
                next_value,
                (self.args.n_rollout_threads, self.envs.n_agents)
            ).cpu().numpy(),
            self.args.gamma,
            self.args.gae_lambda
        )

    def _check_episode_outcome(self, done_envs, current_step):
        """
        Check the outcome of the episode, and log the results.

        Args:
            done_envs (np.ndarray): Array of booleans indicating whether each environment is done
            current_step (int): Current step in the rollout
        """
        done_indices = np.where(done_envs)[0]

        # Extract episode lengths and rewards for done environments
        episode_lens = self.episode_length[done_indices]
        episode_rews = self.episode_rewards[done_indices]

        # Reset episode lengths and rewards for done environments
        self.episode_length[done_indices] = 0
        self.episode_rewards[done_indices] = 0

        # Log metrics
        self.logger.add_scalar('train/length', np.mean(episode_lens), current_step)
        self.logger.add_scalar('train/rewards', np.mean(episode_rews), current_step)

    def _log_rollout_outcome(self, last_infos, rollout_data, train_info, current_step):
        """
        Log the outcome of the rollout.

        Args:
            last_infos (list): List of dictionaries containing environment information for each thread
            rollout_data (dict): Dictionary containing episode-specific data
            train_info (dict): Dictionary containing training information
            current_step (int): Current step in the rollout
        """
        # Log episode-specific data
        self.logger.add_scalar('train/length', np.mean(rollout_data['episode_lengths']), current_step)
        self.logger.add_scalar('train/rewards', np.mean(rollout_data['episode_rewards']), current_step)

        battles_won = []
        battles_game = []
        incre_battles_won = []
        incre_battles_game = []

        for i, info in enumerate(last_infos):
            if "battles_won" in info.keys():
                battles_won.append(info["battles_won"])
                incre_battles_won.append(
                    info["battles_won"] - self.last_battles_won[i]
                )
            if "battles_game" in info.keys():
                battles_game.append(info["battles_game"])
                incre_battles_game.append(
                    info["battles_game"] - self.last_battles_game[i]
                )

        win_rate = (
            np.sum(incre_battles_won) / np.sum(incre_battles_game)
            if np.sum(incre_battles_game) > 0
            else 0.0
        )
        self.logger.add_scalar("train/win_rate", win_rate, current_step)

        self.last_battles_game = battles_game
        self.last_battles_won = battles_won
        self.logger.log_training(
            {
                "train/critic/critic_loss":      train_info['critic_loss'],
                "train/agent0/actor_loss":       train_info['actor_loss'],
                "train/agent0/entropy_loss":     train_info['entropy_loss'],
                "train/agent0/approx_kl":        train_info['approx_kl'],
                "train/agent0/clip_ratio":       train_info['clip_ratio'],
                "train/agent0/actor_grad_norm":  train_info['actor_grad_norm'],
                "train/critic/critic_grad_norm": train_info['critic_grad_norm'],
            }
        )

    @torch.no_grad()
    def evaluate(self, num_episodes=10, capture_video=False, model_path=None):
        """
        Evaluate the current policy, using vec envs.

        Args:
            num_episodes (int): Number of episodes to evaluate
            capture_video (bool): Whether to capture video of the evaluation
            model_path (str): Path to the model to evaluate

        Returns:
            tuple: (mean_rewards, win_rate)
        """
        # Load model if provided
        if model_path is not None:
            print(f"Loading model from {model_path} for evaluation...")
            self.agent.load(model_path)

        # Evaluation stats
        all_episode_rewards = []
        all_episode_lengths = []
        all_win_rates = []
        current_episode = 0
        capture_episodes = 0

        frames = [] if capture_video else None  # For video capture
        obs, _, available_actions = self.eval_envs.reset()

        # Episode tracking
        episode_rewards = np.zeros((self.args.n_eval_rollout_threads), dtype=np.float32)
        episode_length = np.zeros((self.args.n_eval_rollout_threads), dtype=np.float32)

        # Initialize RNN states
        if self.args.use_rnn:
            eval_rnn_states = np.zeros(
                (
                    self.args.n_eval_rollout_threads,
                    self.eval_envs.n_agents,
                    self.args.rnn_layers,
                    self.args.hidden_size
                ),
                dtype=np.float32)
        else:
            eval_rnn_states = None

        # Initialize masks
        eval_masks = np.ones(
            (self.args.n_eval_rollout_threads, self.eval_envs.n_agents, 1),
            dtype=np.float32)

        while True:
            # Get actions
            actions, _, eval_rnn_states = self.agent.get_actions(
                flatten_first_dims(
                    to_tensor(obs, device=self.device, copy=True)
                ),
                flatten_first_dims(
                    to_tensor(eval_rnn_states, device=self.device, copy=True)
                ) if self.args.use_rnn else None,
                flatten_first_dims(
                    to_tensor(eval_masks, device=self.device, copy=True)
                ),
                flatten_first_dims(
                    to_tensor(available_actions, device=self.device, copy=True)
                ) if available_actions is not None else None,
                deterministic=True
            )

            # Reshape actions and values
            shape = (self.args.n_eval_rollout_threads, self.eval_envs.n_agents)
            actions = unflatten_first_dim(actions, shape).cpu().numpy()
            eval_rnn_states = (
                unflatten_first_dim(eval_rnn_states, shape).cpu().numpy()
            ) if self.args.use_rnn else None

            # Execute actions in environment
            obs, share_obs, rewards, dones, infos, available_actions = self.eval_envs.step(actions)

            # Update episode stats
            episode_rewards += rewards[:, 0, 0]
            episode_length += 1

            # Handle episode termination
            done_envs = np.all(dones, axis=1)
            done_env_mask = done_envs == True

            # Reset RNN states and masks for done environments
            if self.args.use_rnn:
                eval_rnn_states[done_env_mask] = 0.0
            eval_masks = np.ones(
                (self.args.n_eval_rollout_threads, self.eval_envs.n_agents, 1),
                dtype=np.float32,
            )
            eval_masks[done_env_mask] = 0.0

            if capture_video and capture_episodes <= 3:
                # Render and capture frame of first 3 episodes
                frame = self.eval_envs.render(mode="rgb_array", env_id=0)
                frames.append(frame)

            # Update episode stats
            for i in range(self.args.n_eval_rollout_threads):
                if done_envs[i]:
                    all_episode_rewards.append(episode_rewards[i])
                    all_episode_lengths.append(episode_length[i])
                    episode_rewards[i] = 0
                    episode_length[i] = 0
                    current_episode += 1
                    if i == 0:
                        capture_episodes += 1
                    # Check if episode was won
                    all_win_rates.append(infos[i]["battle_won"])

            if current_episode >= num_episodes:
                break

        # Calculate statistics
        mean_rewards = np.mean(all_episode_rewards)
        mean_length = np.mean(all_episode_lengths)
        win_rate = np.mean(all_win_rates)

        # Log evaluation stats
        if self.is_train:
            self.logger.add_scalar('eval/rewards', mean_rewards, self.total_steps)
            self.logger.add_scalar('eval/win_rate', win_rate, self.total_steps)
            self.logger.add_scalar('eval/length', mean_length, self.total_steps)
            print(
                f"{self.total_steps}/{self.args.max_steps} Evaluation: Mean rewards: {mean_rewards:.2f},  Mean length: {mean_length:.2f}, Win rate: {win_rate:.2f}")
        else:
            print(f"Mean rewards: {mean_rewards:.2f},  Mean length: {mean_length:.2f}, Win rate: {win_rate:.2f}")

        if capture_video:
            video = np.stack(frames, axis=0)
            self.logger.add_video("eval/render", video, self.total_steps)

        # Update best win rate
        if self.is_train and win_rate > self.best_win_rate:
            self.best_win_rate = win_rate
            save_path = os.path.join(self.logger.dir_name, f"best-torch.model")
            self.agent.save(save_path)
            self.logger.log_model(
                file_path=save_path,
                name="best-model",
                artifact_type="model",
                metadata={"win_rate": win_rate,
                          "step":     self.total_steps},
                alias="latest"
            )
            print(f"Saved best model with win rate {win_rate:.2f}")

        return mean_rewards, win_rate

    @torch.no_grad()
    def render(self, model_path=None, num_episodes=10, render_mode="human"):
        """
        Render the model 

        Args:
            model_path (str, optional): Path to the model. Defaults to None.
            num_episodes (int, optional): Number of episodes to render. Defaults to 10.
            render_mode (str, optional): Rendering mode. Defaults to "human".
        """
        # Load model if provided
        if model_path is not None:
            print(f"Loading model from {model_path} for rendering...")
            self.agent.load(model_path)

        obs, _, available_actions = self.envs.reset()

        # Initialize RNN states
        if self.args.use_rnn:
            render_rnn_states = np.zeros(
                (
                    1,
                    self.envs.n_agents,
                    self.args.rnn_layers,
                    self.args.hidden_size
                ),
                dtype=np.float32)
        else:
            render_rnn_states = None

        # Initialize masks
        render_masks = np.ones((1, self.envs.n_agents, 1), dtype=np.float32)

        episode_rewards = 0
        episode_length = 0
        episode = 0

        frames = [] if render_mode == "rgb_array" else None

        # Render loop
        while True:
            # Get actions
            actions_t, _, render_rnn_states_t = self.agent.get_actions(
                flatten_first_dims(
                    to_tensor(obs, device=self.device, copy=True)
                ),
                flatten_first_dims(
                    to_tensor(render_rnn_states, device=self.device, copy=True)
                ) if self.args.use_rnn else None,
                flatten_first_dims(
                    to_tensor(render_masks, device=self.device, copy=True)
                ),
                flatten_first_dims(
                    to_tensor(available_actions, device=self.device, copy=True)
                ) if available_actions is not None else None,
                deterministic=True
            )

            # Reshape actions and values
            shape = (1, self.envs.n_agents)
            actions = unflatten_first_dim(actions_t, shape).cpu().numpy()
            render_rnn_states = (
                unflatten_first_dim(render_rnn_states_t, shape).cpu().numpy()
            ) if self.args.use_rnn else None

            # Execute actions in environment
            obs, _, rewards, dones, _, available_actions = self.envs.step(actions)

            # Update episode stats
            episode_rewards += rewards[0, 0, 0]
            episode_length += 1

            # Handle episode termination
            done_envs = np.all(dones, axis=1)
            done_env_mask = done_envs == True

            # Reset RNN states and masks for done environments
            if self.args.use_rnn:
                render_rnn_states[done_env_mask] = 0.0

            render_masks = np.ones((1, self.envs.n_agents, 1), dtype=np.float32, )
            render_masks[done_env_mask] = 0.0

            if self.args.env_name == "smacv2" and render_mode == "rgb_array":
                # Render and capture frame
                frame = self.envs.render(mode=render_mode, env_id=0)
                frames.append(frame)

            # Update episode stats
            if done_envs[0]:
                print(f"Episode rewards: {episode_rewards}, Episode length: {episode_length}")
                episode_rewards = 0
                episode_length = 0
                episode += 1
                if episode >= num_episodes:
                    break

        if render_mode == "rgb_array":
            save_video(frames, self.args.env_name, self.args.map_name, self.args.algo)

        if "smac" in self.args.env_name:
            print("Saving Starcraft II replay...")
            self.envs.envs[0].save_replay()
            # Get the latest replay from SC2's replay folder
            latest_replay = get_latest_sc2_replay()
            if latest_replay:
                print(f"Latest replay saved at: {latest_replay}")

    def close(self):
        """
        Close environments and logger.
        
        This method ensures proper cleanup of resources by:
        1. Closing the render environment if rendering was enabled
        2. Closing training and evaluation environments
        3. Closing the logger to ensure all metrics are properly saved
        
        Should be called when training is complete or if an exception occurs.
        """
        self.envs.close()
        if self.args.mode == "train":
            self.logger.close()
        if self.args.mode in ("train", "eval"):
            self.eval_envs.close()


class MAPPO_SRMTRunner(MAPPORunner):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.agent = MAPPO_SRMT(args,
                                self.envs.observation_space,
                                self.envs.share_observation_space,
                                self.envs.action_space,
                                self.device)

        self.obs_shape = get_shape_from_obs_space(self.envs.observation_space)

        if self.is_train:
            # Create buffer
            self.buffer = RolloutStorageSRMT(
                args,
                self.envs.n_agents,
                self.envs.observation_space,
                self.envs.action_space,
                self.envs.share_observation_space,
                self.device)

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
            (actions_t,
             action_log_probs_t,
             actor_rnn_states_t,
             history_seq_t,
             agent_memory_t,
             global_memory_t,) = self.agent.get_actions(
                flatten_first_dims(
                    to_tensor(self.buffer.obs[step], device=self.device)
                ),
                flatten_first_dims(
                    to_tensor(self.buffer.actor_rnn_states[step], device=self.device)
                ) if self.args.use_rnn else None,
                flatten_first_dims(
                    to_tensor(self.buffer.active_masks[step], device=self.device)
                ),
                flatten_first_dims(
                    to_tensor(self.buffer.available_actions[step], device=self.device)
                ) if self.buffer.available_actions is not None else None,
                flatten_first_dims(
                    to_tensor(self.buffer.history_seq[step], device=self.device)
                ) if self.buffer.history_seq is not None else None,
                flatten_first_dims(
                    to_tensor(self.buffer.agent_memory[step], device=self.device)
                ) if self.buffer.agent_memory is not None else None,
                flatten_first_dims(
                    to_tensor(self.buffer.global_memory[step], device=self.device)
                ) if self.buffer.global_memory is not None else None,
                deterministic=False,
            )

            values_t, critic_rnn_states_t = self.agent.get_values(
                flatten_first_dims(
                    to_tensor(self.buffer.get_state(step, replicate=True), device=self.device)
                ),
                flatten_first_dims(
                    to_tensor(self.buffer.obs[step], device=self.device)
                ),
                flatten_first_dims(
                    to_tensor(self.buffer.active_masks[step], device=self.device)
                ),
                flatten_first_dims(
                    to_tensor(self.buffer.get_critic_rnn(step, replicate=True), device=self.device)
                ) if self.args.use_rnn else None,
                flatten_first_dims(
                    to_tensor(self.buffer.masks[step], device=self.device)
                )
            )

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

    def insert(self, data):
        """
        Insert a new transition into the buffer.

        Args:
           data (tuple): Transition data containing:
            - obs: Agent observations (n_rollout_threads, n_agents, obs_dim)
            - share_obs: Shared observations (n_rollout_threads, n_agents, share_obs_dim)
            - rewards: Agent rewards (n_rollout_threads, n_agents, 1)
            - dones: Done flags (n_rollout_threads, n_agents)
            - infos: Environment info dicts [n_rollout_threads]
            - available_actions: Available actions mask (n_rollout_threads, n_agents, action_dim) or None
            - values: Value estimates (n_rollout_threads, n_agents, 1)
            - actions: Taken actions (n_rollout_threads, n_agents, action_dim)
            - action_log_probs: Action log probs (n_rollout_threads, n_agents, action_dim)
            - actor_rnn_states: Actor RNN states (n_rollout_threads, n_agents, num_layers, hidden_size)
            - critic_rnn_states: Critic RNN states (n_rollout_threads, n_agents, num_layers, hidden_size)
            - history_seq: SRMT History sequence ()
            - agent_memory: SRMT agent memory ()
            - global_memory: SRMT global memory ()
        """
        # Unpack transition data
        (obs, share_obs, rewards, dones, infos, available_actions,
         values, actions, action_log_probs, actor_rnn_states, critic_rnn_states,
         history_seq, agent_memory, global_memory,) = data

        # Handle episode terminations
        done_envs = np.all(dones, axis=1)  # Check which environments are done
        # n_done_envs = done_envs.sum()
        done_env_mask = done_envs == True

        # Reset RNN states for done environments
        if self.args.use_rnn:
            critic_rnn_states[done_env_mask] = 0.0

        # Create masks
        shape = (self.args.n_rollout_threads, self.envs.n_agents, 1)
        masks = np.ones(shape, dtype=np.float32)
        active_masks = np.ones(shape, dtype=np.float32)

        # Update masks for done environments and agents
        masks[done_env_mask, :, :] = 0.0  # broadcast across agents and last dim
        active_masks[dones, :] = 0.0  # for individual agent deaths
        active_masks[done_env_mask, :, :] = 1.0  # for full environment termination

        # Create truncation masks from environment infos
        truncates = np.array([[info['truncated']] for info in infos])  # (n_rollout_threads, 1)
        truncates = np.repeat(truncates, self.envs.n_agents, axis=1).reshape(*shape)  # (n_rollout_threads, n_agents, 1)

        # Store trajectory in buffer
        self.buffer.insert(
            obs=obs,  # (n_rollout_threads, n_agents, n_obs)
            global_state=share_obs,  # (n_rollout_threads, n_agents, n_state) or (n_rollout_threads, n_state)
            actions=actions,  # (n_rollout_threads, n_agents, 1)
            action_log_probs=action_log_probs,  # (n_rollout_threads, n_agents, 1)
            values=values,  # (n_rollout_threads, n_agents, 1)
            rewards=rewards,  # (n_rollout_threads, n_agents, 1)
            masks=masks,  # (n_rollout_threads, n_agents, 1)
            active_masks=active_masks,  # (n_rollout_threads, n_agents, 1)
            truncates=truncates,  # (n_rollout_threads, n_agents, 1)
            available_actions=available_actions,  # (n_rollout_threads, n_agents, n_actions) or None
            actor_rnn_states=actor_rnn_states,  # (n_rollout_threads, n_agents, num_layers, hidden_size)
            critic_rnn_states=critic_rnn_states,  # (n_rollout_threads, n_agents, num_layers, hidden_size)
            history_seq=history_seq,
            agent_memory=agent_memory,
            global_memory=global_memory,
        )

    @torch.no_grad()
    def evaluate(self, num_episodes=10, capture_video=False, model_path=None):
        """
        Evaluate the current policy, using vec envs.

        Args:
            num_episodes (int): Number of episodes to evaluate
            capture_video (bool): Whether to capture video of the evaluation
            model_path (str): Path to the model to evaluate

        Returns:
            tuple: (mean_rewards, win_rate)
        """
        # Load model if provided
        if model_path is not None:
            print(f"Loading model from {model_path} for evaluation...")
            self.agent.load(model_path)

        # Evaluation stats
        all_episode_rewards = []
        all_episode_lengths = []
        all_win_rates = []
        current_episode = 0
        capture_episodes = 0

        frames = [] if capture_video else None  # For video capture
        obs, _, available_actions = self.eval_envs.reset()

        # Init SRMT components for evaluation
        history_seq = None
        agent_memory = None
        global_memory = None
        if self.args.srmt_core:
            history_seq = np.zeros(
                (self.args.n_eval_rollout_threads, self.eval_envs.n_agents, self.args.data_chunk_length,
                 self.args.hidden_size),
                dtype=np.float32)
            if self.args.use_agent_memory:
                agent_memory = np.zeros(
                    (self.args.n_eval_rollout_threads, self.eval_envs.n_agents, self.args.hidden_size),
                    dtype=np.float32)
            if self.args.use_global_memory:
                global_memory = np.zeros(
                    (self.args.n_eval_rollout_threads, self.eval_envs.n_agents, self.args.n_agents,
                     self.args.hidden_size),
                    dtype=np.float32)

        # Episode tracking
        episode_rewards = np.zeros((self.args.n_eval_rollout_threads), dtype=np.float32)
        episode_length = np.zeros((self.args.n_eval_rollout_threads), dtype=np.float32)

        # Initialize RNN states
        eval_rnn_states = None

        # Initialize masks
        eval_masks = np.ones(
            (self.args.n_eval_rollout_threads, self.eval_envs.n_agents, 1),
            dtype=np.float32)

        while current_episode < num_episodes:
            # Get actions
            (actions,
             action_log_probs,
             actor_rnn_states,
             history_seq,
             agent_memory,
             global_memory,) = self.agent.get_actions(
                flatten_first_dims(
                    to_tensor(obs, device=self.device)
                ),
                flatten_first_dims(
                    to_tensor(eval_rnn_states, device=self.device)
                ) if eval_rnn_states is not None else None,
                flatten_first_dims(
                    to_tensor(eval_masks, device=self.device)
                ),
                flatten_first_dims(
                    to_tensor(available_actions, device=self.device)
                ) if available_actions is not None else None,
                flatten_first_dims(
                    to_tensor(history_seq, device=self.device)
                ) if history_seq is not None else None,
                flatten_first_dims(
                    to_tensor(agent_memory, device=self.device)
                ) if agent_memory is not None else None,
                flatten_first_dims(
                    to_tensor(global_memory, device=self.device)
                ) if global_memory is not None else None,
                deterministic=False,
            )

            # Reshape actions and values
            shape = (self.args.n_eval_rollout_threads, self.eval_envs.n_agents)
            actions = unflatten_first_dim(actions, shape).cpu().numpy()
            eval_rnn_states = (
                unflatten_first_dim(eval_rnn_states, shape).cpu().numpy()
            ) if eval_rnn_states is not None else None

            # Unfflatten SRMT data
            history_seq = unflatten_first_dim(history_seq, shape).cpu().numpy() if history_seq is not None else None
            agent_memory = unflatten_first_dim(agent_memory, shape).cpu().numpy() if agent_memory is not None else None
            global_memory = unflatten_first_dim(global_memory,
                                                shape).cpu().numpy() if global_memory is not None else None

            # Execute actions in environment
            obs, share_obs, rewards, dones, infos, available_actions = self.eval_envs.step(actions)

            # Update episode stats
            episode_rewards += rewards[:, 0, 0]
            episode_length += 1

            # Handle episode termination
            done_envs = np.all(dones, axis=1)
            done_env_mask = done_envs == True

            eval_masks = np.ones(
                (self.args.n_eval_rollout_threads, self.eval_envs.n_agents, 1),
                dtype=np.float32,
            )
            eval_masks[done_env_mask] = 0.0

            if capture_video and capture_episodes <= 3:
                # Render and capture frame of first 3 episodes
                frame = self.eval_envs.render(mode="rgb_array", env_id=0)
                frames.append(frame)

            # Update episode stats
            for i in range(self.args.n_eval_rollout_threads):
                if done_envs[i]:
                    all_episode_rewards.append(episode_rewards[i])
                    all_episode_lengths.append(episode_length[i])
                    episode_rewards[i] = 0
                    episode_length[i] = 0
                    current_episode += 1
                    if i == 0:
                        capture_episodes += 1
                    # Check if episode was won
                    all_win_rates.append(infos[i]["battle_won"])

        # Calculate statistics
        mean_rewards = np.mean(all_episode_rewards)
        mean_length = np.mean(all_episode_lengths)
        win_rate = np.mean(all_win_rates)

        # Log evaluation stats
        if self.is_train:
            self.logger.add_scalar('eval/rewards', mean_rewards, self.total_steps)
            self.logger.add_scalar('eval/win_rate', win_rate, self.total_steps)
            self.logger.add_scalar('eval/length', mean_length, self.total_steps)
            print(
                f"{self.total_steps}/{self.args.max_steps} Evaluation: Mean rewards: {mean_rewards:.2f},  Mean length: {mean_length:.2f}, Win rate: {win_rate:.2f}")
        else:
            print(f"Mean rewards: {mean_rewards:.2f},  Mean length: {mean_length:.2f}, Win rate: {win_rate:.2f}")

        if capture_video:
            video = np.stack(frames, axis=0)
            self.logger.add_video("eval/render", video, self.total_steps)

        # Update best win rate
        if self.is_train and win_rate > self.best_win_rate:
            self.best_win_rate = win_rate
            save_path = os.path.join(self.logger.dir_name, f"best-torch.model")
            self.agent.save(save_path)
            self.logger.log_model(
                file_path=save_path,
                name="best-model",
                artifact_type="model",
                metadata={"win_rate": win_rate,
                          "step":     self.total_steps},
                alias="latest"
            )
            print(f"Saved best model with win rate {win_rate:.2f}")

        return mean_rewards, win_rate
