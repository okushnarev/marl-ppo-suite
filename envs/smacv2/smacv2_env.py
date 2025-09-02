from __future__ import absolute_import, division, print_function

import numpy as np
from absl import logging
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

logging.set_verbosity(logging.ERROR)
import os.path as osp
import yaml

from gymnasium.spaces import Box, Discrete


class SMACv2Env:
    def __init__(self, args, seed=None):
        """
        Initialize the SMACv2 environment.

        Args:
            args (dict): Dictionary containing environment configuration parameters
            seed (int, optional): Random seed for the environment. Defaults to None.
        """

        self.map_config = self._load_map_config(args.map_name)
        self.use_agent_id = args.use_agent_id
        self.use_death_masking = args.use_death_masking

        # Store seed if provided
        self._seed = args.seed
        if self._seed is None:
            raise ValueError("SMACv2Env requires a seed to be set.")

        self.map_config['seed'] = self._seed

        self.env = StarCraftCapabilityEnvWrapper(**self.map_config)

        env_info = self.env.get_env_info()
        n_actions = env_info["n_actions"]
        state_shape = env_info["state_shape"]
        obs_shape = env_info["obs_shape"]

        self.episode_limit = env_info['episode_limit']
        self.n_agents = env_info["n_agents"]

        # Get properties from the environment
        self.timeouts = self.env.env.timeouts

        # Define observation and action spaces for vectorization
        self.share_observation_space = Box(
           low=-np.inf,
           high=np.inf,
           shape=(state_shape,),
           dtype=np.float32
        )

        # If using agent IDs, add the agent ID dimensions to the observation space
        if self.use_agent_id:
            obs_shape_with_id = obs_shape + self.n_agents
            self.observation_space = Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_shape_with_id,),
                dtype=np.float32
            )
        else:
            self.observation_space = Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_shape,),
                dtype=np.float32
            )

        self.action_space = Discrete(n_actions)

    def reset(self):
        """
        Reset the environment and return initial observations, states, and available actions.

        Returns:
            obs: List of observations for each agent
            state: State of the environment
            available_actions: List of available actions for each agent
        """
        # Reset the environment
        self.env.reset()

        # Get observations, state, and available actions
        obs = self.env.get_obs()
        state = self.env.get_state()
        available_actions = self.env.get_avail_actions()

        # Add agent IDs to observations if enabled
        if self.use_agent_id:
            obs = self._add_agent_id_to_obs(obs)

        return obs, state, available_actions

    def step(self, actions):
        """
        Take a step in the environment with the given actions
        and return observations, states, rewards, dones, infos, and available actions.

        Args:
            actions: Actions to take for each agent

        Returns:
            obs: List of observations for each agent
            state: State of the environment
            rewards: List of rewards for each agent
            dones: List of done flags for each agent
            infos: List of info dictionaries for each agent
            available_actions: List of available actions for each agent
        """
        # Take a step in the environment
        reward, terminated, info = self.env.step(actions)

        # Get observations, state, and available actions
        obs = self.env.get_obs()
        state = self.env.get_state()
        available_actions = self.env.get_avail_actions()

        # Add agent IDs to observations if enabled
        if self.use_agent_id:
            obs = self._add_agent_id_to_obs(obs)

        # Format rewards for each agent
        rewards = [[reward]] * self.n_agents

        # Pass additional info
        info["truncated"] = False

        # Format dones for each agent
        if terminated:
            # If the episode is terminated, all agents are done
            dones = [True] * self.n_agents
            if self.env.env.timeouts > self.timeouts:
                assert (
                    self.env.env.timeouts - self.timeouts == 1
                ), "Change of timeouts unexpected."
                info["truncated"] = True
                self.timeouts = self.env.env.timeouts
        elif self.use_death_masking:
            # Create a list of done flags for each agent based on death status
            dones = [bool(self.env.env.death_tracker_ally[agent_id]) for agent_id in range(self.n_agents)]
        else:
            # If use_death_masking is False, all agents are not done
            dones = [False] * self.n_agents

        # Dead agents IDs
        dead_agents = [action[0] for action in self.env.get_avail_actions()]

        # Pass additional info
        info.update({
            "win": self.env.env.win_counted,
            "lost": self.env.env.defeat_counted,
            "battles_game": self.env.env.battles_game,
            "battles_won": self.env.env.battles_won,
            "battle_won": self.env.env.win_counted,  # Add this key for compatibility with mappo_runner.py
            "dead_agents": dead_agents,
        })

        return obs, state, rewards, dones, info, available_actions

    def close(self):
        self.env.close()

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def save_replay(self):
        self.env.save_replay()

    def _add_agent_id_to_obs(self, obs):
        """
        Add agent ID as a one-hot encoding to each agent's observation.

        Args:
            obs: List of observations for each agent

        Returns:
            List of observations with agent IDs added
        """
        obs = np.asarray(obs, dtype=np.float32)            # (n_agents, obs_dim)
        eye = np.eye(self.n_agents, dtype=np.float32)      # (n_agents, n_agents)
        return np.concatenate([obs, eye], axis=1)          # (n_agents, obs_dim + n_agents)

    def _load_map_config(self, map_name):
        """
        Load map configuration from YAML file.

        Args:
            map_name: The name of the map configuration to load (e.g., "terran_5_vs_5")

        Returns:
            Dictionary containing the map configuration

        Raises:
            FileNotFoundError: If the configuration file cannot be found
        """
        # Load only from the local config directory
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(current_dir, "config", f"{map_name}.yaml")

        if not osp.exists(config_path):
            raise FileNotFoundError(f"Map config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as file:
            map_config = yaml.load(file, Loader=yaml.FullLoader)

        return map_config
