import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Pipe
from envs.env_vectorization import VecEnv, CloudpickleWrapper, tile_images
import multiprocessing.shared_memory as shared_memory
from gymnasium.spaces import Discrete
import copy

def worker_shared_memory(remote, parent_remote, env_fn_wrapper, 
                         obs_shm_name, share_obs_shm_name, avail_actions_shm_name,
                         reward_shm_name, done_shm_name,
                         obs_shape, share_obs_shape, avail_actions_shape,
                         reward_shape, done_shape,
                         env_idx):
    """
    Worker function for SharedMemoryVecEnv.

    Args:
        remote: Pipe connection to the parent process
        parent_remote: Pipe connection to the child process (to be closed)
        env_fn_wrapper: Wrapped environment creation function
        *_shm_name: Names of shared memory blocks
        *_shape: Shapes of arrays in shared memory
        env_idx: Index of this environment in the shared memory arrays
    """
    parent_remote.close()
    env = env_fn_wrapper.x()
    
    # Connect to shared memory blocks
    obs_shm = shared_memory.SharedMemory(name=obs_shm_name)
    share_obs_shm = shared_memory.SharedMemory(name=share_obs_shm_name)
    avail_actions_shm = shared_memory.SharedMemory(name=avail_actions_shm_name)
    reward_shm = shared_memory.SharedMemory(name=reward_shm_name)
    done_shm = shared_memory.SharedMemory(name=done_shm_name)
    
    # Create numpy arrays using the shared memory buffers
    obs_buf = np.ndarray(obs_shape, dtype=np.float32, buffer=obs_shm.buf)
    share_obs_buf = np.ndarray(share_obs_shape, dtype=np.float32, buffer=share_obs_shm.buf)
    avail_actions_buf = np.ndarray(avail_actions_shape, dtype=np.float32, buffer=avail_actions_shm.buf)
    reward_buf = np.ndarray(reward_shape, dtype=np.float32, buffer=reward_shm.buf)
    done_buf = np.ndarray(done_shape, dtype=np.bool_, buffer=done_shm.buf)
    
    try:
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
                
                # Write data to shared memory
                obs_buf[env_idx] = ob
                share_obs_buf[env_idx] = s_ob
                avail_actions_buf[env_idx] = available_actions
                reward_buf[env_idx] = reward
                done_buf[env_idx] = done
                
                # Send only the info dict through the pipe
                remote.send(info)
                
            elif cmd == "reset":
                ob, s_ob, available_actions = env.reset()
                
                # Write data to shared memory
                obs_buf[env_idx] = ob
                share_obs_buf[env_idx] = s_ob
                avail_actions_buf[env_idx] = available_actions
                
                # Signal that reset is complete
                remote.send(True)
                
            elif cmd == "render":
                if data == "rgb_array":
                    fr = env.render(mode=data)
                    remote.send(fr)
                elif data == "human":
                    env.render(mode=data)
                    
            elif cmd == "close":
                env.close()
                obs_shm.close()
                share_obs_shm.close()
                avail_actions_shm.close()
                reward_shm.close()
                done_shm.close()
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
    except Exception as e:
        print(f"Exception in worker process {env_idx}: {e}")
        raise

class SharedMemoryVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in subprocesses and uses shared memory
    for efficient data transfer of observations and other large arrays.
    
    This implementation significantly reduces the interprocess communication overhead
    compared to the standard SubprocVecEnv.
    """
    
    def __init__(self, env_fns):
        """
        Initialize the SharedMemoryVecEnv.
        
        Args:
            env_fns: List of functions that create environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_fns)
        
        # Create pipes for communication with subprocesses
        ctx = mp.get_context("fork")
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        
        # Get environment information from the first environment
        env = env_fns[0]()
        self.n_agents = getattr(env, 'n_agents', None)
        self.episode_limit = getattr(env, 'episode_limit', None)
        observation_space = env.observation_space
        share_observation_space = env.share_observation_space
        action_space = env.action_space
        env.close()
        
        # Initialize the base VecEnv
        VecEnv.__init__(
            self, self.num_envs, observation_space, share_observation_space, action_space,
            self.n_agents, self.episode_limit
        )
        
        # Determine array shapes
        obs_shape = (self.num_envs, self.n_agents) + tuple(observation_space.shape)
        # TODO: Define support for cases when we have FP
        share_obs_shape = (self.num_envs,) + tuple(share_observation_space.shape)
        
        # For discrete action spaces
        if isinstance(action_space, Discrete):
            avail_actions_shape = (self.num_envs, self.n_agents, action_space.n)
        else:
            # For continuous action spaces, adjust as needed
            avail_actions_shape = (self.num_envs, self.n_agents, 1)
            
        reward_shape = (self.num_envs, self.n_agents, 1)
        done_shape = (self.num_envs, self.n_agents)
        
        # Create shared memory blocks
        self.obs_shm = shared_memory.SharedMemory(
            create=True, 
            size=int(np.prod(obs_shape) * np.dtype(np.float32).itemsize)
        )
        self.share_obs_shm = shared_memory.SharedMemory(
            create=True, 
            size=int(np.prod(share_obs_shape) * np.dtype(np.float32).itemsize)
        )
        self.avail_actions_shm = shared_memory.SharedMemory(
            create=True, 
            size=int(np.prod(avail_actions_shape) * np.dtype(np.float32).itemsize)
        )
        self.reward_shm = shared_memory.SharedMemory(
            create=True, 
            size=int(np.prod(reward_shape) * np.dtype(np.float32).itemsize)
        )
        self.done_shm = shared_memory.SharedMemory(
            create=True, 
            size=int(np.prod(done_shape) * np.dtype(np.bool_).itemsize)
        )
        
        # Create numpy arrays using the shared memory
        self.obs_buf = np.ndarray(obs_shape, dtype=np.float32, buffer=self.obs_shm.buf)
        self.share_obs_buf = np.ndarray(share_obs_shape, dtype=np.float32, buffer=self.share_obs_shm.buf)
        self.avail_actions_buf = np.ndarray(avail_actions_shape, dtype=np.float32, buffer=self.avail_actions_shm.buf)
        self.reward_buf = np.ndarray(reward_shape, dtype=np.float32, buffer=self.reward_shm.buf)
        self.done_buf = np.ndarray(done_shape, dtype=np.bool_, buffer=self.done_shm.buf)
        
        # Start subprocesses
        self.ps = [
            Process(
                target=worker_shared_memory,
                args=(
                    work_remote, remote, CloudpickleWrapper(env_fn),
                    self.obs_shm.name, self.share_obs_shm.name, self.avail_actions_shm.name,
                    self.reward_shm.name, self.done_shm.name,
                    obs_shape, share_obs_shape, avail_actions_shape,
                    reward_shape, done_shape,
                    i
                ),
            )
            for i, (work_remote, remote, env_fn) in enumerate(zip(
                self.work_remotes, self.remotes, env_fns
            ))
        ]
        
        for p in self.ps:
            p.daemon = True  # If the main process crashes, we should not cause things to hang
            p.start()
            
        for remote in self.work_remotes:
            remote.close()
            
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
        # Only receive the info dicts through the pipes
        infos = [remote.recv() for remote in self.remotes]
        self.waiting = False
        
        # Data is already in shared memory, just need to return views
        return (
            self.obs_buf.copy(),
            self.share_obs_buf.copy(),
            self.reward_buf.copy(),
            self.done_buf.copy(),
            infos,
            self.avail_actions_buf.copy(),
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
            
        # Wait for all environments to complete reset
        [remote.recv() for remote in self.remotes]
        
        # Data is already in shared memory
        return (
            self.obs_buf.copy(),
            self.share_obs_buf.copy(),
            self.avail_actions_buf.copy()
        )
        
    def close(self):
        """Close all environments, terminate subprocesses, and release shared memory."""
        if self.closed:
            return
            
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
                
        for remote in self.remotes:
            remote.send(("close", None))
            
        for p in self.ps:
            p.join()
            
        # Clean up shared memory
        self.obs_shm.close()
        self.share_obs_shm.close()
        self.avail_actions_shm.close()
        self.reward_shm.close()
        self.done_shm.close()
        
        self.obs_shm.unlink()
        self.share_obs_shm.unlink()
        self.avail_actions_shm.unlink()
        self.reward_shm.unlink()
        self.done_shm.unlink()
        
        self.closed = True
        
    def render(self, mode="human"):
        """
        Render the environments.
        
        Args:
            mode: Rendering mode ('human' or 'rgb_array')
            
        Returns:
            If mode is 'rgb_array', returns the rendered image
            If mode is 'human', returns whether the viewer is open
        """
        for remote in self.remotes:
            remote.send(("render", mode))
            if mode == "rgb_array":
                imgs = [remote.recv() for remote in self.remotes]
                bigimg = tile_images(imgs)
                return bigimg
        return None
