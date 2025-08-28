"""
Environment factory for creating StarCraft 2 environments with various wrappers.
"""
import platform
import multiprocessing as mp
import random
import numpy as np
import torch
from envs.env_vectorization import SubprocVecEnv, DummyVecEnv

# Configure multiprocessing start method based on platform
# This is done at module import time to ensure it's set before any MP operations
if mp.get_start_method(allow_none=True) is None:
    if platform.system() == "Darwin":  # macOS
        # On macOS, fork works well with thread limiting
        mp.set_start_method("forkserver", force=True)
    else:
        # On Linux, forkserver is generally safer
        mp.set_start_method("fork", force=True)

def create_env(args, is_eval=False):
    """
    LightMappo version only
    Create a StarCraft 2 environment with the specified wrappers based on arguments.

    Args:
        args: Arguments containing environment configuration
        is_eval: Whether this is an evaluation environment (default: False).
               Currently not used, but kept for potential future differentiation
               between training and evaluation environments.
    Returns:
        env: The wrapped environment
    """


    from smac.env import StarCraft2Env
    from envs.wrappers import AgentIDWrapper, DeathMaskingWrapper
    # Create base StarCraft2Env with safe defaults
    env = StarCraft2Env(map_name=args.map_name, difficulty=args.difficulty, obs_last_action=args.obs_last_actions)


    # Apply wrappers in the correct order (innermost to outermost)

    # 1. Always apply death masking first (innermost wrapper)
    env = DeathMaskingWrapper(env, use_death_masking=args.use_death_masking)

    # 2. Apply agent ID wrapper
    env = AgentIDWrapper(env,
                        use_agent_id=args.use_agent_id)

    return env


def make_env(args, base_seed, rank, is_eval=False):
    """
    Helper function to create an environment with a given seed.

    Args:
        args: Arguments object containing environment configuration
        base_seed: Base seed for all environments
        rank: Environment rank (for seeding)
        is_eval: Whether this is an evaluation environment

    Returns:
        A function that creates the environment when called
    """
    # Calculate a unique seed for this environment
    env_seed = base_seed + rank * 10000

    def _thunk():

        random.seed(env_seed)             # only affects this process
        np.random.seed(env_seed)
        torch.manual_seed(env_seed)
        # print(f"Creating environment with seed: {env_seed} (rank {rank})")

        env_name = args.env_name

        if env_name == "smacv1":
            if hasattr(args, "use_fp_wrapper") and args.use_fp_wrapper:
                # Legacy Version using wrapper.
                from smac.env import StarCraft2Env
                from envs.wrappers import FeaturePrunedStateWrapper

                env = StarCraft2Env(map_name=args.map_name, seed=env_seed)

                env = FeaturePrunedStateWrapper(
                    env,
                    use_agent_specific_state=args.use_agent_specific_state,
                    add_distance_state=args.add_distance_state,
                    add_xy_state=args.add_xy_state,
                    add_visible_state=args.add_visible_state,
                    add_center_xy=args.add_center_xy,
                    add_enemy_action_state=args.add_enemy_action_state,
                    use_mustalive=args.use_mustalive,
                    use_agent_id=args.use_agent_id
                )
            else:
                # New Version using SMACv1Env.
                from envs.smacv1.Starcraft2_Env import StarCraft2Env as SMACv1Env

                env = SMACv1Env(
                    map_name=args.map_name,
                    state_type=args.state_type,
                    seed=env_seed
                )
        elif env_name == "smacv2":
            from envs.smacv2 import SMACv2Env

            _orig_shuffle = random.shuffle

            def _patched_shuffle(seq, *args, **kwargs):
                # drop any extra positional args, but pass through a keyword "random" if given
                if args:
                    # args[0] is the old “random” function pysc2 passed;
                    # shuffle() will call random() from the module if no keyword is provided.
                    return _orig_shuffle(seq, **kwargs)
                return _orig_shuffle(seq, **kwargs)

            random.shuffle = _patched_shuffle
            env = SMACv2Env(args, seed=env_seed)
        else:
            raise ValueError(f"Unknown environment name: {env_name}")

        return env

    return _thunk

def make_vec_envs(args, num_processes, is_eval=False):
    """
    Create vectorized environments.

    Args:
        args: Arguments object containing environment configuration
        num_processes: Number of parallel environments
        is_eval: Whether these are evaluation environments

    Returns:
        Vectorized environments
    """
    # eval envs have a different base seed,
    base_seed = args.seed + 1000000 if is_eval else args.seed
    # Create environment thunks
    envs = [make_env(args, base_seed, rank=i, is_eval=is_eval) for i in range(num_processes)]

    # If only one environment, use DummyVecEnv for simplicity
    if len(envs) == 1:
        return DummyVecEnv(envs)
    else:
        # Choose the appropriate vectorization method
        # You can uncomment alternatives if you want to experiment

        # Option 1: SubprocVecEnv (reliable and now optimized)
        return SubprocVecEnv(envs)

        # Option 2: SharedMemoryVecEnv (potentially faster but more complex)
        # from envs.env_shared_memory_vec import SharedMemoryVecEnv
        # return SharedMemoryVecEnv(envs)

        # Option 3: RayVecEnv (for distributed training)
        # from envs.env_ray_vec import RayVecEnv
        # return RayVecEnv(envs)