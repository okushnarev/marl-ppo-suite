"""
Environment vectorization for running multiple environments in parallel.

This module provides classes for running multiple environments in parallel,
either in separate processes (SubprocVecEnv) or in a single process (DummyVecEnv).
The implementation is based on the HARL codebase but simplified and optimized.
"""

import numpy as np
from multiprocessing import Process, Pipe
import multiprocessing as mp
from abc import ABC, abstractmethod
import copy


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image.

    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.

    Args:
        img_nhwc: list or array of images, ndim=4 once turned into array
                 n = batch index, h = height, w = width, c = channel

    Returns:
        bigim_HWc: ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N) / H))

    # Fill in missing images with zeros
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(N, H * W)])

    # Reshape and transpose to tile images
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H * h, W * w, c)

    return img_Hh_Ww_c


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle).

    This is needed for passing environment creation functions to subprocesses.
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class VecEnv(ABC):
    """
    An abstract vectorized environment.

    Used to batch data from multiple copies of an environment, so that
    each observation becomes a batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """

    closed = False
    viewer = None
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, num_envs, observation_space, share_observation_space, action_space, n_agents=None, episode_limit=None):
        """
        Initialize the vectorized environment.

        Args:
            num_envs: Number of environments to run in parallel
            observation_space: Observation space of a single environment
            share_observation_space: Shared observation space (for CTDE algorithms)
            action_space: Action space of a single environment
            n_agents: Number of agents in the environment (default: None)
            episode_limit: Maximum number of steps per episode (default: None)
        """
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space
        self.n_agents = n_agents
        self.episode_limit = episode_limit

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of observations, states, and available actions.

        Returns:
            obs: Observations from all environments
            states: States from all environments (for CTDE)
            available_actions: Available actions for all environments
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step with the given actions.

        Call step_wait() to get the results of the step.

        Args:
            actions: Actions to take in each environment
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns:
            obs: Observations from all environments
            states: States from all environments (for CTDE)
            rewards: Rewards from all environments
            dones: Done flags from all environments
            infos: Info dictionaries from all environments
            available_actions: Available actions for all environments
        """
        pass

    def close_extras(self):
        """
        Clean up the extra resources, beyond what's in this base class.

        Only runs when not self.closed.
        """
        pass

    def close(self):
        """Close all environments and release resources."""
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.

        Args:
            actions: Actions to take in each environment

        Returns:
            Results from step_wait()
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode="human", env_id=None):
        """
        Render the environments.

        Args:
            mode: Rendering mode ('human' or 'rgb_array')

        Returns:
            If mode is 'rgb_array', returns the rendered image
            If mode is 'human', returns whether the viewer is open
        """
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        if mode == "human":
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == "rgb_array":
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        """
        Return RGB images from each environment.

        Returns:
            List of RGB images from each environment
        """
        raise NotImplementedError

    @property
    def unwrapped(self):
        """Return the base environment."""
        return self

    def get_viewer(self):
        """Get or create the viewer for rendering."""
        if self.viewer is None:
            try:
                from gymnasium.envs.classic_control import rendering
                self.viewer = rendering.SimpleImageViewer()
            except ImportError:
                # Fallback if gymnasium rendering is not available
                self.viewer = None
                print("Warning: gymnasium.envs.classic_control.rendering not available. Rendering not supported.")
        return self.viewer

    def get_env_info(self):
        """
        Get information about the environment.

        Returns:
            Dictionary containing environment information
        """
        info = {
            'num_envs': self.num_envs,
            'observation_space': self.observation_space,
            'share_observation_space': self.share_observation_space,
            'action_space': self.action_space,
            'n_agents': self.n_agents,
            'episode_limit': self.episode_limit
        }
        return info


def worker(remote, parent_remote, env_fn_wrapper):
    """
    Worker function for SubprocVecEnv.

    Args:
        remote: Pipe connection to the parent process
        parent_remote: Pipe connection to the child process (to be closed)
        env_fn_wrapper: Wrapped environment creation function
    """
    parent_remote.close()
    env = env_fn_wrapper.x()

    while True:
        cmd, data = remote.recv()

        if cmd == "step":
            ob, s_ob, reward, done, info, available_actions = env.step(data)

            # Handle episode termination
            if type(done) is bool:  # done is a bool
                if done:  # if done, save the original obs, state, and available actions in info, and then reset
                    info["final_obs"] = copy.deepcopy(ob)
                    info["final_state"] = copy.deepcopy(s_ob)
                    info["final_avail_actions"] = copy.deepcopy(available_actions)
                    ob, s_ob, available_actions = env.reset()
            else:
                if np.all(done):  # if all agents are done, reset the environment
                    info["final_obs"] = copy.deepcopy(ob)
                    info["final_state"] = copy.deepcopy(s_ob)
                    info["final_avail_actions"] = copy.deepcopy(available_actions)
                    ob, s_ob, available_actions = env.reset()

            remote.send((ob, s_ob, reward, done, info, available_actions))

        elif cmd == "reset":
            ob, s_ob, available_actions = env.reset()
            remote.send((ob, s_ob, available_actions))

        elif cmd == "render":
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)

        elif cmd == "close":
            env.close()
            remote.close()
            break

        elif cmd == "get_spaces":
            remote.send((env.observation_space, env.share_observation_space, env.action_space))

        elif cmd == "get_num_agents":
            remote.send((env.n_agents))

        elif cmd == "get_episode_limit":
            remote.send(env.episode_limit)

        else:
            raise NotImplementedError(f"Unknown command: {cmd}")


class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in subprocesses.

    This allows for parallel execution of environments, which can significantly
    speed up training, especially for computationally intensive environments.
    """

    def __init__(self, env_fns):
        """
        Initialize the SubprocVecEnv.

        Args:
            env_fns: List of functions that create environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)

        # Check the current start method
        current_method = mp.get_start_method(allow_none=True)
        print(f"Current multiprocessing start method: {current_method}")

        # Create pipes for communication with subprocesses
        ctx = mp.get_context()
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(nenvs)])

        # Start subprocesses
        self.ps = [
            Process(
                target=worker,
                args=(work_remote, remote, CloudpickleWrapper(env_fn)),
            )
            for (work_remote, remote, env_fn) in zip(
                self.work_remotes, self.remotes, env_fns
            )
        ]

        for p in self.ps:
            p.daemon = True  # If the main process crashes, we should not cause things to hang
            p.start()

        for remote in self.work_remotes:
            remote.close()

        # Get number of agents
        self.remotes[0].send(("get_num_agents", None))
        self.n_agents = self.remotes[0].recv()

        # Get episode limit
        self.remotes[0].send(("get_episode_limit", None))
        self.episode_limit = self.remotes[0].recv()

        # Get spaces
        self.remotes[0].send(("get_spaces", None))
        self.observation_space, self.share_observation_space, self.action_space = self.remotes[0].recv()

        VecEnv.__init__(
            self, len(env_fns), self.observation_space, self.share_observation_space, self.action_space,
            self.n_agents, self.episode_limit
        )

    def step_async(self, actions):
        """
        Tell all the environments to start taking a step with the given actions.

        Args:
            actions: Actions to take in each environment
        """
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns:
            obs: Observations from all environments
            states: States from all environments (for CTDE)
            rewards: Rewards from all environments
            dones: Done flags from all environments
            infos: Info dictionaries from all environments
            available_actions: Available actions for all environments
        """
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, share_obs, rews, dones, infos, available_actions = zip(*results)

        return (
            np.stack(obs),
            np.stack(share_obs),
            np.stack(rews),
            np.stack(dones),
            infos,
            np.stack(available_actions),
        )

    def reset(self):
        """
        Reset all the environments.

        Returns:
            obs: Observations from all environments
            states: States from all environments (for CTDE)
            available_actions: Available actions for all environments
        """
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        obs, share_obs, available_actions = zip(*results)

        return np.stack(obs), np.stack(share_obs), np.stack(available_actions)

    def close(self):
        """Close all environments and terminate subprocesses."""
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True

    def render(self, mode="human", env_id=None):
        """
        Render the environments.

        Args:
            mode: Rendering mode ('human' or 'rgb_array')

        Returns:
            If mode is 'rgb_array', returns the rendered image
            If mode is 'human', returns whether the viewer is open
        """
        if env_id is not None:
            self.remotes[env_id].send(("render", mode))
            if mode == "rgb_array":
                return self.remotes[env_id].recv()
    
        for remote in self.remotes:
            remote.send(("render", mode))
            if mode == "rgb_array":
                imgs = [remote.recv() for remote in self.remotes]
                bigimg = tile_images(imgs)
                return bigimg
        return None


class DummyVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments sequentially in a single process.

    This is useful for debugging or when subprocessing is not desired.
    """

    def __init__(self, env_fns):
        """
        Initialize the DummyVecEnv.

        Args:
            env_fns: List of functions that create environments
        """
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]

        # Get n_agents and episode_limit from the environment
        self.n_agents = getattr(env, 'n_agents', None)
        self.episode_limit = getattr(env, 'episode_limit', None)

        VecEnv.__init__(
            self,
            len(env_fns),
            env.observation_space,
            env.share_observation_space,
            env.action_space,
            self.n_agents,
            self.episode_limit
        )

        self.actions = None

    def step_async(self, actions):
        """
        Store actions for step_wait.

        Args:
            actions: Actions to take in each environment
        """
        self.actions = actions

    def step_wait(self):
        """
        Execute stored actions in each environment.

        Returns:
            obs: Observations from all environments
            states: States from all environments (for CTDE)
            rewards: Rewards from all environments
            dones: Done flags from all environments
            infos: Info dictionaries from all environments
            available_actions: Available actions for all environments
        """
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, share_obs, rews, dones, infos, available_actions = map(
            np.array, zip(*results)
        )

        # Handle episode termination
        for i, done in enumerate(dones):
            if isinstance(done, bool):  # done is a bool
                if done:  # if done, save the original obs, state, and available actions in info, and then reset
                    infos[i]["final_obs"] = copy.deepcopy(obs[i])
                    infos[i]["final_state"] = copy.deepcopy(share_obs[i])
                    infos[i]["final_avail_actions"] = copy.deepcopy(
                        available_actions[i]
                    )
                    obs[i], share_obs[i], available_actions[i] = self.envs[i].reset()
            else:
                if np.all(done):  # if all agents are done, reset the environment
                    infos[i]["final_obs"] = copy.deepcopy(obs[i])
                    infos[i]["final_state"] = copy.deepcopy(share_obs[i])
                    infos[i]["final_avail_actions"] = copy.deepcopy(
                        available_actions[i]
                    )
                    obs[i], share_obs[i], available_actions[i] = self.envs[i].reset()

        self.actions = None

        return obs, share_obs, rews, dones, infos, available_actions

    def reset(self):
        """
        Reset all the environments.

        Returns:
            obs: Observations from all environments
            states: States from all environments (for CTDE)
            available_actions: Available actions for all environments
        """
        results = [env.reset() for env in self.envs]
        obs, share_obs, available_actions = map(np.array, zip(*results))

        return obs, share_obs, available_actions

    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()

    def render(self, mode="human", env_id=None):
        """
        Render the environments.

        Args:
            mode: Rendering mode ('human' or 'rgb_array')

        Returns:
            If mode is 'rgb_array', returns the rendered images
            If mode is 'human', renders to the screen
        """
        if env_id is not None:
            return self.envs[env_id].render(mode=mode)
        
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError
