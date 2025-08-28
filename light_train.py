import argparse
from runners.light_mappo_runner import LightMAPPORunner
from runners.light_rnn_mappo_runner import LightRMAPPORunner

def parse_args():
    """
    Parse command line arguments for Light MAPPO training.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser("MAPPO for StarCraft")

    # Algorithm parameters
    parser.add_argument("--algo", type=str, default="mappo",
                        choices=["mappo", "mappo_rnn"],
                        help="Which algorithm to use")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument("--cuda", action='store_false', default=True,
                        help="by default True, will use GPU to train; or else will use CPU;")
    parser.add_argument("--cuda_deterministic",
                        action='store_false', default=True,
                        help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--max_steps", type=int, default=1_000_000,
                        help="Number of environment steps to train on")

    # Environment parameters
    parser.add_argument("--map_name", type=str, default="3m",
                        help="Which SMAC map to run on")
    parser.add_argument("--difficulty", type=str, default="7",
                        help="Difficulty of the SMAC map")
    parser.add_argument("--obs_last_actions", action="store_true", default=False,
                        help="Whether to include last actions in observations (default: False)")

    # Agent ID parameters
    parser.add_argument("--use_agent_id", action="store_false", default=True,
        help="Whether to add agent_id to observation (default: True)")

    # Death masking parameters
    parser.add_argument("--use_death_masking", action="store_true", default=False,
        help="Whether to use death masking to exclude dead agents from training (default: True)")
    
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
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="Dimension of hidden layers")
    parser.add_argument("--rnn_layers", type=int, default=1,
                        help="Number of RNN layers")
    parser.add_argument("--data_chunk_length", type=int, default=10,
                        help="Time length of chunks used to train a recurrent_policy")
    parser.add_argument("--fc_layers", type=int, default=1,
                        help="Number of fc layers in actor/critic network")
    parser.add_argument("--actor_gain", type=float, default=0.01,
                        help="Gain of the actor final linear layer")
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
    parser.add_argument("--use_eval", action="store_false",
                        help="Evaluate the model during training (default: True)")
    parser.add_argument("--eval_interval", type=int, default=5000,
                        help="Evaluate the model every eval_interval steps")
    parser.add_argument("--eval_episodes", type=int, default=32,
                        help="Number of episodes for evaluation")

    return parser.parse_args()


def main():
    args = parse_args()
    runner = None

    print("=============================")
    print(f"Training {args.algo.upper()} for StarCraft")
    print("=============================")
    print(f"Map: {args.map_name}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Seed: {args.seed}")
    print(f"Algorithm: {args.algo}")
    print(f"Reward Norm: {args.use_reward_norm} ({args.reward_norm_type})")
    print(f"Value Norm: {args.use_value_norm} ({args.value_norm_type})")

    # print(f"Using {'CUDA' if args.cuda and torch.cuda.is_available() else 'CPU'}")
    # print(f"Using {'recurrent' if args.use_recurrent_policy else 'MLP'} policy")
    print("=============================")

   
    if args.algo == "mappo":
        runner = LightMAPPORunner(args)
    elif args.algo == "mappo_rnn":
        runner = LightRMAPPORunner(args)
    else:
        raise ValueError(f"Invalid algorithm: {args.algo}")

    runner.run()


if __name__ == "__main__":
    main()