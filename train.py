import os

from runners.ippo_runner import IPPORunner, IPPO_SRMTRunner

if "MKL_NUM_THREADS" not in os.environ:
    os.environ["MKL_NUM_THREADS"] = "1"
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import torch

from runners.mappo_runner import MAPPORunner, MAPPO_SRMTRunner
from runners.happo_runner import HAPPORunner
from utils.env_tools import set_global_seeds
from utils.sc2_utils import kill_sc2_processes


# Register cleanup function to kill SC2 processes on exit
# def cleanup_sc2_processes():
#     print("\nCleaning up any lingering SC2 processes...")
#     killed = kill_sc2_processes()
#     if killed > 0:
#         print(f"Killed {killed} SC2 processes during cleanup")
#     else:
#         print("No SC2 processes needed to be cleaned up")

# # Register the cleanup function to run when the script exits
# atexit.register(cleanup_sc2_processes)

def parse_args():
    """
    Parse command line arguments for MAPPO&HAPPO training.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser("MAPPO&HAPPO for StarCraft")

    # Algorithm parameters
    parser.add_argument("--algo", type=str, default="mappo",
                        choices=["mappo", "happo", "ippo"],
                        help="Which algorithm to use")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument("--cuda", action='store_false', default=True,
                        help="by default True, will use GPU to train; or else will use CPU;")
    parser.add_argument("--cuda_deterministic", action='store_false', default=True,
                        help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--torch_threads", type=int, default=None,
                        help="Set PyTorch/OMP/MKL threads (default None))")
    parser.add_argument("--max_steps", type=int, default=1_000_000,
                        help="Number of environment steps to train on")
    parser.add_argument("--n_rollout_threads", type=int, default=8,
                        help="Number of parallel environments (default: 8)")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1,
                        help="Number of threads for evaluation (default: 1)")

    # Environment parameters
    parser.add_argument("--env_name", type=str, default="smacv1",
                        choices=["smacv1", "smacv2"],
                        help="Which environment to run on")
    parser.add_argument("--map_name", type=str, default="3m",
                        help="Which SMAC map to run on")

    # State parameters
    parser.add_argument("--state_type", type=str, default="EP", choices=["FP", "EP", "AS"],
                        help="Type of state to use in critic: 'FP' (Feature Pruned AS - only Smacv1) or "
                             "'EP' (Environment Provided) or 'AS' (Agent-Specific - observation + state / not implemented)")

    # SMACv2 state parameters
    parser.add_argument("--use_death_masking", action="store_true", default=False,
                        help="Whether to use SMACv2 death masking (default: False)")  # will make sure critic see zeros too
    parser.add_argument("--use_agent_id", action="store_true", default=False,
                        help="Whether to use SMACv2 agent ID (default: False)")  # Doesn't make sense, because Randomness.

    # Optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Initial learning rate")
    parser.add_argument("--optimizer_eps", type=float, default=1e-5,
                        help="Epsilon for optimizer")
    parser.add_argument("--use_linear_lr_decay", action='store_true',
                        help='Use a linear schedule on the learning rate')
    parser.add_argument("--min_lr", type=float, default=1e-5,
                        help="Minimum learning rate")

    # Network parameters
    parser.add_argument("--fixed_order", action="store_true", default=False,
                        help="Whether to use fixed order for HAPPO (default: False)")
    parser.add_argument("--use_rnn", action="store_true", default=False,
                        help="Whether to use RNN networks (default: False)")
    parser.add_argument("--rnn_layers", type=int, default=1,
                        help="Number of RNN layers")
    parser.add_argument("--data_chunk_length", type=int, default=10,
                        help="Time length of chunks used to train a recurrent_policy")
    parser.add_argument("--fc_layers", type=int, default=2,
                        help="Number of fc layers in actor/critic network")
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="Dimension of hidden layers")
    parser.add_argument("--actor_gain", type=float, default=0.01,
                        help="Gain of the actor final linear layer")
    parser.add_argument("--critic_gain", type=float, default=0.01,
                        help="Gain of the critic final linear layer")
    parser.add_argument("--use_feature_normalization", action='store_false',
                        help="Apply layernorm to the inputs (default: True)")
    parser.add_argument("--use_value_norm", action="store_true",
                        help="Use running mean and std to normalize returns (default: False)")
    parser.add_argument("--value_norm_type", type=str, default="welford", choices=["welford", "ema"],
                        help="Type of value normalizer to use: 'welford' (original) or 'ema' (exponential moving average)")
    parser.add_argument("--use_reward_norm", action="store_false",
                        help="Use running mean and std to normalize rewards (default: True)")
    parser.add_argument("--reward_norm_type", type=str, default="efficient", choices=["efficient", "ema"],
                        help="Type of reward normalizer to use: 'efficient' (standard) or 'ema' (exponential moving average)")

    # SRMT specific parameters
    parser.add_argument("--srmt_core", action="store_true", default=False,
                        help="Use SRMT core in Actor (default: False)")
    parser.add_argument("--use_agent_memory", action="store_true", default=False,
                        help="Use agent memory in SRMT core (default: False)")
    parser.add_argument("--use_global_memory", action="store_true", default=False,
                        help="Use global memory in SRMT core (default: False)")
    parser.add_argument("--num_transformer_layers", default=1,
                        help="How many GPT2Blocks to use in core transformer (default: 1)")

    # PPO parameters
    parser.add_argument("--n_steps", type=int, default=400,
                        help="Number of steps to run in each environment per policy rollout (default: 400)")
    parser.add_argument("--ppo_epoch", type=int, default=15,
                        help="Number of epochs for PPO")
    parser.add_argument("--use_clipped_value_loss", action="store_false",
                        help="Use clipped value loss (default: True)")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help="PPO clip parameter")
    parser.add_argument("--num_mini_batch", type=int, default=1,
                        help="Number of mini-batches for PPO")
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument("--critic_coef", type=float, default=0.5,
                        help="Critic coefficient")
    parser.add_argument("--use_gae", action="store_false",
                        help="Use Generalized Advantage Estimation (default: True)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help="Lambda for Generalized Advantage Estimation")
    parser.add_argument("--use_proper_time_limits", action="store_false",
                        help="Use proper time limits (default: True)")
    parser.add_argument("--use_max_grad_norm", action="store_false",
                        help="Use max gradient norm (default: True)")
    parser.add_argument("--max_grad_norm", type=float, default=10,
                        help="Max gradient norm")
    parser.add_argument("--use_huber_loss", action="store_true",
                        help="Use huber loss (default: False)")
    parser.add_argument("--huber_delta", type=float, default=10.0,
                        help="Delta for huber loss")

    # Evaluation parameters
    parser.add_argument("--log_interval", type=int, default=10000,
                        help="Log interval (~5 rollouts 8 envs 400 steps)")
    parser.add_argument("--use_eval", action="store_false",
                        help="Evaluate the model during training (default: True)")
    parser.add_argument("--eval_interval", type=int, default=20000,
                        help="Evaluate the model every eval_interval steps (~25 rollouts 8 envs 400 steps)")
    parser.add_argument("--eval_episodes", type=int, default=32,
                        help="Number of episodes for evaluation")
    parser.add_argument("--capture_video", action="store_true", default=False,
                        help="Capture video during training (default: False)")
    parser.add_argument("--capture_video_interval", type=int, default=100000,
                        help="Capture video every capture_video_interval steps (~30 rollouts 8 envs 400 steps))")

    # Wandb parameters
    parser.add_argument('--use_wandb', action='store_true',
                        help='Track experiment with Weights & Biases (default: False)')

    # Tensorbord parameters
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name to show in tensorboard')
    parser.add_argument('--write_hyperparams', action='store_false', default=True,
                        help='Write config to json file (default: True)')

    # Rendering parameters
    parser.add_argument("--mode", choices=["train", "eval", "render"],
                        default="train",
                        help="train (default), eval (no learning), or render")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to the configuration file for rendering and evaluation")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to the model to render")
    parser.add_argument("--render_episodes", type=int, default=10,
                        help="Number of episodes to render")
    parser.add_argument("--render_mode", type=str, default="human", choices=["human", "rgb_array"],
                        help="Render mode: 'human' or 'rgb_array'")

    return parser.parse_args()


def load_render_config(args):
    import json
    with open(args.config, 'r') as f:
        config = json.load(f)

    protected_flags = {
        'mode':                   args.mode,
        'model':                  args.model,
        'render_episodes':        args.render_episodes,
        'render_mode':            args.render_mode,
        'eval_episodes':          args.eval_episodes,
        'capture_video':          args.capture_video,
        'n_eval_rollout_threads': args.n_eval_rollout_threads
        # Add any other flags that should be protected
    }

    # Create a copy of the original args
    original_args_dict = vars(args).copy()

    # Update with render config
    original_args_dict.update(config)

    # Restore protected flags
    for key, value in protected_flags.items():
        if value is not None:  # Only restore if the flag was set
            original_args_dict[key] = value

    # Create new Namespace with updated values
    args = argparse.Namespace(**original_args_dict)
    return args


def main():
    args = parse_args()

    if args.mode in ("eval", "render") and args.config:
        args = load_render_config(args)

    runner = None

    print("=============================")
    print(f"Training {args.algo.upper()} for StarCraft")
    print("=============================")
    print(f"Map: {args.map_name}")
    print(f"Seed: {args.seed}")
    print(f"Algorithm: {args.algo}")
    print(f"Reward Norm: {args.use_reward_norm} ({args.reward_norm_type})")
    print(f"Value Norm: {args.use_value_norm} ({args.value_norm_type})")
    print(f"State Type: {args.state_type}")

    print("\nTraining Parameters:")
    print(f" Rollout Threads: {args.n_rollout_threads}")
    print(f"  Steps: {args.n_steps}")
    print(f"  Epochs: {args.ppo_epoch}")
    print(f"  Mini-Batches: {args.num_mini_batch}")

    print(f"Using {'CUDA' if args.cuda and torch.cuda.is_available() else 'CPU'}")
    print(f"Using {'RNN' if args.use_rnn else 'MLP'} policy")
    print("=============================")

    # Set device and thread configuration centrally
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set thread configuration
    cpu_threads = args.torch_threads or 1
    torch.set_num_threads(cpu_threads)
    # os.environ["OMP_NUM_THREADS"]  = str(cpu_threads)
    # os.environ["MKL_NUM_THREADS"]  = str(cpu_threads)

    # Set global seeds and deterministic mode
    set_global_seeds(args.seed, deterministic_cuda=args.cuda_deterministic)

    # Print thread configuration
    print(f"Thread settings:")
    print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'Not set')}")
    print(f"  MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'Not set')}")
    print(f"  PyTorch threads: {torch.get_num_threads()}")

    try:
        if args.srmt_core:
            match args.algo:
                case "mappo":
                    runner = MAPPO_SRMTRunner(args, device)
                case "ippo":
                    runner = IPPO_SRMTRunner(args, device)
                case _:
                    raise ValueError(f"Invalid algorithm: {args.algo}")
        else:
            match args.algo:
                case "mappo":
                    runner = MAPPORunner(args, device)
                case "happo":
                    runner = HAPPORunner(args, device)
                case "ippo":
                    runner = IPPORunner(args, device)
                case _:
                    raise ValueError(f"Invalid algorithm: {args.algo}")

        if args.mode == "train":
            runner.run()
        elif args.mode == "eval":
            runner.evaluate(
                num_episodes=args.eval_episodes,
                capture_video=args.capture_video,
                model_path=args.model)
        else:
            runner.render(
                num_episodes=args.render_episodes,
                model_path=args.model,
                render_mode=args.render_mode)
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up environments even if an error occurs
        if runner is not None:
            try:
                runner.close()
            except Exception as e:
                print(f"Error closing environments: {e}")

        # # Force kill any remaining SC2 processes
        # print("Ensuring all SC2 processes are terminated...")
        # kill_sc2_processes()


if __name__ == "__main__":
    main()
