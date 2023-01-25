# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import math
import warnings

import torch
from torch import Tensor
from torch import distributed as dist
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

from tutel import net

from tutel.impls import communicate as C
from tutel.impls.fast_dispatch import fast_encode, fast_decode
from tutel.impls.overlap import a2a_ffn_overlap_forward


class FusedExpertsNetwork(torch.nn.Module):
    __constants__ = ['in_features', 'hidden_features', 'out_features', 'num_global_experts']
    in_features: int
    hidden_features: int
    out_features: int
    num_global_experts: int
    batched_fc1_weight: Tensor
    batched_fc2_weight: Tensor

    def __init__(
        self,
        in_features,
        hidden_features,
        num_global_experts=None,
        out_features=None,
        bias=True,
        activation_fn=None,
        a2a_ffn_overlap_degree=1,
        is_postscore=True,
        parallel_type='auto',
        use_2dh=False,
        group=None,
        scan_expert_func=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        if in_features % 2:
            raise ValueError(f'in_features ({in_features}) must be an even value.')

        self.group = group or dist.group.WORLD
        world_size = C.get_world_size(group)

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features or in_features

        self.world_size = world_size
        if num_global_experts is None:
            warnings.warn(f'initializing MoE expert layer with num_global_experts={world_size=}')
            num_global_experts = world_size

        if num_global_experts > world_size:
            if num_global_experts % world_size:
                raise ValueError(
                    f'Attempting to init {num_global_experts=} on {world_size=},'
                    f'if num_global_experts > world_size, num_global_experts must be divisible by world_size'
                )
        else:
            if world_size % num_global_experts:
                raise ValueError(
                    f'Attempting to init {num_global_experts=} on {world_size=},'
                    f'if world_size > num_global_experts, world_size must be divisible by num_global_experts'
                )

        self.num_global_experts = num_global_experts
        self.num_local_experts = num_global_experts // world_size or 1
        self.num_shards = world_size // num_global_experts or 1

        assert self.hidden_features % self.num_shards == 0, f"Can't evenly divide hidden_features ({self.hidden_features}) to {self.num_shards} shards."
        local_hidden_features = self.hidden_features // self.num_shards
        self.batched_fc1_weight = Parameter(torch.empty(self.num_local_experts, local_hidden_features, self.in_features, **factory_kwargs))
        self.batched_fc2_weight = Parameter(torch.empty(self.num_local_experts, local_hidden_features, self.out_features, **factory_kwargs))
        if bias:
            self.batched_fc1_bias = Parameter(torch.empty(self.num_local_experts, 1, local_hidden_features, **factory_kwargs))
            self.batched_fc2_bias = Parameter(torch.empty(self.num_local_experts, 1, (self.out_features + self.num_shards - 1) // self.num_shards, **factory_kwargs))
        else:
            self.register_parameter('batched_fc1_bias', None)
            self.register_parameter('batched_fc2_bias', None)
        self.reset_parameters()

        if activation_fn is None:
            activation_fn = lambda x: F.relu(x)
        self.activation_fn = activation_fn

        self.set_parallel_strategy(parallel_type)

        self.is_postscore = is_postscore

        self.a2a_ffn_overlap_degree = a2a_ffn_overlap_degree
        self.use_2dh = use_2dh

        if scan_expert_func is not None:
            for n, p in self.named_parameters():
                scan_expert_func(n, p)
        for n, p in self.named_parameters():
            setattr(p, '_tutel_expert', True)


    def extra_repr(self) -> str:
        num_local_experts = self.num_local_experts
        num_shards = self.num_shards
        in_features = self.in_features
        hidden_features = self.hidden_features
        out_features = self.out_features
        return (
            f'{self.num_global_experts} experts running on {self.world_size} devices '
            f'({num_local_experts=}; {num_shards=}) with '
            f'{in_features=}, {hidden_features=}, {out_features=}'
        )

    def reset_parameters(self) -> None:
        # same as nn.Linear except bias is set to 0
        init.kaiming_uniform_(self.batched_fc1_weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.batched_fc2_weight, a=math.sqrt(5))
        if self.batched_fc1_bias is not None:
            torch.nn.init.zeros_(self.batched_fc1_bias)
        if self.batched_fc1_bias is not None:
            torch.nn.init.zeros_(self.batched_fc2_bias)

    def set_parallel_strategy(self, parallel_type):
        # set up parallelization strategy
        self.force_data_parallel, self.force_adaptive, self.adaptive_degree = False, False, self.num_shards
        if parallel_type.startswith('adaptive:'):
            self.adaptive_degree = int(parallel_type[parallel_type.index(':') + 1:])
            if self.adaptive_degree == 0:
                self.force_data_parallel = True
            else:
                if self.adaptive_degree < 0 or self.num_shards % self.adaptive_degree != 0:
                    valids = [i for i in range(1, self.num_shards + 1) if self.num_shards % i == 0]
                    raise Exception("Unexpected value of adaptive_degree: %d, expecting a candidate within %s." % (self.adaptive_degree, valids))
                self.force_adaptive = True
            self.auto_parallel, self.use_model_parallel = False, True
        elif self.num_shards == 1:
            self.auto_parallel, self.use_model_parallel = False, False
        elif parallel_type in ('data', 'model'):
            self.auto_parallel, self.use_model_parallel = False, (parallel_type == 'model')
        elif parallel_type == 'auto':
            self.auto_parallel, self.use_model_parallel = True, False
        else:
            raise Exception('Unrecognized parallel type specified: %s' % parallel_type)

    def apply_parallel_strategy_to_weights(self):
        batched_fc1_weight = self.batched_fc1_weight
        batched_fc2_weight = self.batched_fc2_weight
        batched_fc1_bias = self.batched_fc1_bias
        batched_fc2_bias = self.batched_fc2_bias

        if self.force_data_parallel:
            batched_fc1_weight = net.zero_gather(batched_fc1_weight, group=self.group).view(self.num_global_experts, -1, batched_fc1_weight.size(2))
            batched_fc2_weight = net.zero_gather(batched_fc2_weight, group=self.group).view(self.num_global_experts, -1, batched_fc2_weight.size(2))
            batched_fc1_bias = net.zero_gather(batched_fc1_bias, group=self.group).view(self.num_global_experts, 1, -1)
            batched_fc2_bias = net.zero_gather(batched_fc2_bias, group=self.group).view(self.num_global_experts, 1, -1)
        elif self.force_adaptive:
            if self.num_shards > 1:
                group_size = self.num_shards // self.adaptive_degree
                if group_size > 1:
                    ffn_zero_group = net.create_groups_from_world(group_count=-group_size).model_group
                    batched_fc1_weight = net.zero_gather(batched_fc1_weight, group=ffn_zero_group).view(1, -1, self.in_features)
                    batched_fc2_weight = net.zero_gather(batched_fc2_weight, group=ffn_zero_group).view(1, -1, self.out_features)
                    batched_fc1_bias = net.zero_gather(batched_fc1_bias, group=ffn_zero_group).view(1, 1, -1)

                ffn_zero_group2 = net.create_groups_from_world(group_count=self.num_global_experts).model_group
                batched_fc2_bias = net.zero_gather(batched_fc2_bias, group=ffn_zero_group2)
                batched_fc2_bias = batched_fc2_bias.view(1, 1, -1)

                if self.adaptive_degree > 1:
                    batched_fc2_bias = torch.mul(batched_fc2_bias, 1.0 / self.adaptive_degree)
        else:
            if self.num_shards > 1:
                ffn_zero_group = net.create_groups_from_world(group_count=self.num_global_experts).model_group
                if not self.use_model_parallel:
                    batched_fc1_weight = net.zero_gather(batched_fc1_weight, group=ffn_zero_group).view(1, -1, self.in_features)
                    batched_fc2_weight = net.zero_gather(batched_fc2_weight, group=ffn_zero_group).view(1, -1, self.out_features)
                    batched_fc1_bias = net.zero_gather(batched_fc1_bias, group=ffn_zero_group).view(1, 1, -1)

                batched_fc2_bias = net.zero_gather(batched_fc2_bias, group=ffn_zero_group)
                batched_fc2_bias = batched_fc2_bias.view(self.batched_fc2_bias.size(0), 1, -1)

                if self.use_model_parallel:
                    batched_fc2_bias = torch.mul(batched_fc2_bias, 1.0 / self.num_shards)

        if batched_fc2_bias.size(-1) != self.out_features:
            batched_fc2_bias = batched_fc2_bias[:, :, :self.out_features]

        return (
            batched_fc1_weight.permute(0, 2, 1),
            batched_fc2_weight,
            batched_fc1_bias,
            batched_fc2_bias
        )

    def expert_fwd(self, x, dim):
        batched_fc1_weight, batched_fc2_weight, batched_fc1_bias, batched_fc2_bias = self.apply_parallel_strategy_to_weights()

        y = x.view(x.size(0), x.size(1), dim)

        y = torch.add(torch.matmul(y, batched_fc1_weight), batched_fc1_bias)
        y = self.activation_fn(y)
        y = torch.add(torch.matmul(y, batched_fc2_weight), batched_fc2_bias)

        return y

    def forward(self, x, logits_dtype, crit):
        shape = x.shape
        x = x.view(-1, shape[-1])

        y = fast_encode(x.to(logits_dtype), crit, self.is_postscore).to(x.dtype)

        if self.force_data_parallel:
            y = self.expert_fwd(y, shape[-1])
        else:
            if self.auto_parallel:
                self.use_model_parallel = (y.numel() * (self.num_shards - 1) * 2 < sum([x.numel() for x in self.parameters()]))

            if self.num_global_experts < self.world_size:
                if self.use_model_parallel:
                    y = y.repeat(1, self.adaptive_degree, 1).view(self.world_size, -1, y.size(2))
                else:
                    y = y.view(self.world_size, -1, y.size(2))

            if self.a2a_ffn_overlap_degree > 1 and y.is_cuda:
                def expert_fn(expert_input):
                    return self.expert_fwd(expert_input, shape[-1])
                y = a2a_ffn_overlap_forward(y, expert_fn=expert_fn, a2a_ffn_overlap_degree=self.a2a_ffn_overlap_degree, use_2dh=self.use_2dh, group=self.group)
            else:
                y = C.all_to_all(y, 1, 0, use_2dh=self.use_2dh, group=self.group)
                y = self.expert_fwd(y, shape[-1])
                y = C.all_to_all(y, 0, 1, use_2dh=self.use_2dh, group=self.group)

            if self.num_global_experts < self.world_size:
                if self.use_model_parallel:
                    y = torch.sum(y.view(self.num_global_experts, self.adaptive_degree, -1, y.size(2)), dim=1)
                else:
                    y = y.view(self.num_global_experts, -1, y.size(2))

        y = fast_decode(y.to(logits_dtype), crit, self.is_postscore)

        y = y.view(list(shape[:-1]) + [self.out_features])

        return y


ExpertModule = FusedExpertsNetwork 
