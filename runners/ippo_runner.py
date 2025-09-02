from algos.ippo import IPPO, IPPO_SRMT
from buffers.rollout_storage import RolloutStorageSRMT
from runners.mappo_runner import MAPPORunner, MAPPO_SRMTRunner


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
