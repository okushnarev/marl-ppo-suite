import torch
from pydantic import BaseModel
from transformers import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from torch import nn as nn
from copy import deepcopy


class CoreConfig(BaseModel):
    num_attention_heads: int = 4
    core_hidden_size: int = 64
    mem: bool = True
    max_position_embeddings: int = 16384
    add_cross_attention: bool = True


class TransformerCore(nn.Module):
    def __init__(self, cfg, input_size: int, device='cpu'):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.device = device
        self.core_cfg: CoreConfig = CoreConfig()
        self.core_cfg.core_hidden_size = self.cfg.hidden_size
        self.use_memory = cfg.use_agent_memory
        self.use_global_memory = cfg.use_global_memory
        self.num_agents = cfg.n_agents
        self.remember_time_steps = 8
        core_cfg_copy = deepcopy(self.core_cfg).__dict__
        core_cfg_copy['hidden_size'] = core_cfg_copy.pop('core_hidden_size')

        # self.encoder = nn.Linear(input_size, self.core_cfg.core_hidden_size)
        self.core_transformer = GPT2Block(GPT2Config(**core_cfg_copy))
        self.wpe = nn.Embedding(core_cfg_copy['max_position_embeddings'],
                                self.core_cfg.core_hidden_size)
        # self.decoder = nn.Linear(self.core_cfg.core_hidden_size, input_size)
        if self.use_memory:
            self.mem_head = nn.Linear(self.core_cfg.core_hidden_size,
                                      self.core_cfg.core_hidden_size,
                                      bias=False)

        self.ln_f = nn.LayerNorm(self.core_cfg.core_hidden_size, eps=1e-5)

        self.to(self.device)

    def forward(self,
                head_output,
                history_seq=None,
                agent_memory=None,
                global_memory=None,
                **kwargs):
        is_seq = not torch.is_tensor(head_output)
        if not is_seq:
            head_output = head_output.unsqueeze(1)
            first_time_mem = False
            if self.use_memory:
                # first pass with empty memory
                if not agent_memory.abs().sum().is_nonzero():
                    agent_memory = None
                    first_time_mem = True
                else:
                    agent_memory_batch = agent_memory.unsqueeze(1)
        if history_seq is not None:
            inputs = torch.cat([history_seq, head_output], dim=1)
        else:
            inputs = head_output.contiguous()

        if agent_memory is not None:
            inputs = torch.cat([agent_memory_batch, inputs], dim=1)

        position_ids = torch.arange(0, inputs.size(1), dtype=torch.long).to('cuda')
        position_ids = position_ids.unsqueeze(0)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs + position_embeds

        encoder_hidden_states = None
        if agent_memory is not None:
            if self.use_global_memory:
                encoder_hidden_states = global_memory.contiguous()
        x = self.core_transformer(hidden_states=hidden_states.contiguous(),
                                  encoder_hidden_states=encoder_hidden_states,
                                  )[0]
        x = self.ln_f(x)
        core_out = x[:, -1:]

        if self.use_memory:
            if first_time_mem:
                my_new_mem = core_out.contiguous()
            else:
                my_new_mem, _ = torch.split(x, [1, x.size()[1] - 1], dim=1)
            my_new_mem = self.mem_head(my_new_mem)

        # update history with current head_output
        if history_seq is not None:
            new_history_seq = torch.cat([history_seq[:, 1:], head_output], dim=1)

        if not is_seq:
            core_out = core_out.squeeze(1)
            if self.use_memory:
                global_memory = my_new_mem.repeat(1, self.num_agents, 1)
                my_new_mem = my_new_mem.squeeze(1)

        additional_outputs = {}

        additional_outputs['history_seq'] = new_history_seq if history_seq is not None else None
        additional_outputs['agent_memory'] = my_new_mem if self.use_memory else None
        additional_outputs['global_memory'] = global_memory if self.use_memory else None

        return core_out, additional_outputs

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_out_size(self) -> int:
        return self.core_cfg.core_hidden_size
