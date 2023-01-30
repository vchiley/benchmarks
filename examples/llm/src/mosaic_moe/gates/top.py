# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn.functional as F
import torch.distributed as dist

from tutel.impls.fast_dispatch import extract_critical
from tutel.impls import losses

class TopKGate(torch.nn.Module):
    def __init__(
        self,
        num_global_experts,
        k=1,
        fp32_gate=False,
        capacity_factor=1.,
        gate_noise=0.,
        inequivalent_tokens=False,
        batch_prioritized_routing=False,
        normalize_gate=True,
        is_gshard_loss=True,
        group=None,
    ):
        super().__init__()
        self.group = group or dist.group.WORLD

        self.num_global_experts = num_global_experts
        self.top_k = min(num_global_experts, int(k))
        self.fp32_gate = fp32_gate
        self.capacity_factor = capacity_factor
        self.gate_noise = gate_noise

        self.inequivalent_tokens = inequivalent_tokens
        self.batch_prioritized_routing = batch_prioritized_routing
        self.normalize_gate = normalize_gate
        self.is_gshard_loss = is_gshard_loss

    def extra_repr(self) -> str:
        top_k = self.top_k
        capacity_factor = self.capacity_factor
        gate_noise = self.gate_noise

        repr_str = f'{top_k=} gate with {capacity_factor=} and {gate_noise=}'
        repr_str += ' in fp32.' if self.fp32_gate else ' in default precision.'

        return repr_str

    def routing(self, logits, num_shards, a2a_ffn_overlap_degree=1):
        if self.training and self.gate_noise > 0:
            logits_w_noise = logits + self.gate_noise * torch.randn_like(logits) / self.num_global_experts
        else:
            logits_w_noise = logits

        scores = F.softmax(logits_w_noise, dim=1)
        if self.is_gshard_loss:
            _loss_fn = lambda gates, topk_ids: losses.gshard_loss(gates, topk_ids)
        else:
            _loss_fn = lambda gates, topk_ids: losses.load_importance_loss(
                F.softmax(logits, dim=1), logits_w_noise.gather(index=topk_ids, dim=1),
                self.num_global_experts, self.gate_noise)
        return extract_critical(
            scores,
            top_k=self.top_k,
            loss_fn=_loss_fn,
            capacity_factor=self.capacity_factor,
            batch_prioritized_routing=self.batch_prioritized_routing,
            normalize_gate=self.normalize_gate,
            group=self.group,
            alignment=num_shards * a2a_ffn_overlap_degree,
            inequivalent_tokens=self.inequivalent_tokens,
        )

    def forward(self, x, num_shards, a2a_ffn_overlap_degree=1):
        shape = x.shape
        assert len(shape) >= 2, "Input data must be at least 2D tensor: (s)amples, .., (m)odel_dim"
        x = x.view(-1, shape[-1])

        logits = self.gate_fwd(x)

        if x.is_cuda:
            with torch.cuda.amp.autocast(enabled=False):
                crit, l_aux = self.routing(logits, num_shards, a2a_ffn_overlap_degree=a2a_ffn_overlap_degree)
        else:
            crit, l_aux = self.routing(logits, num_shards, a2a_ffn_overlap_degree=a2a_ffn_overlap_degree)

        return logits.dtype, (crit, l_aux)


class LinearTopKGate(TopKGate):
    def __init__(
        self,
        in_features,
        num_global_experts,
        k=1,
        fp32_gate=False,
        capacity_factor=1.,
        gate_noise=0.,
        inequivalent_tokens=False,
        batch_prioritized_routing=False,
        normalize_gate=True,
        is_gshard_loss=True,
        group=None,
        device=None,
        dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': torch.float32 if fp32_gate else dtype}
        super().__init__(
            num_global_experts=num_global_experts,
            k=k,
            fp32_gate=fp32_gate,
            capacity_factor=capacity_factor,
            gate_noise=gate_noise,
            inequivalent_tokens=inequivalent_tokens,
            batch_prioritized_routing=batch_prioritized_routing,
            normalize_gate=normalize_gate,
            is_gshard_loss=is_gshard_loss,
            group=group,
        )
        self.gate = torch.nn.Linear(in_features, num_global_experts, bias=False, **factory_kwargs)

    def extra_repr(self) -> str:
        return 'Linear ' + super().extra_repr()

    def gate_fwd(self, x):
        gate = self.gate.float() if self.fp32_gate else self.gate
        return gate(x.float() if self.fp32_gate else x)


Gate = TopKGate
