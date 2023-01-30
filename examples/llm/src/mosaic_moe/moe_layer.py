# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torch.nn import Module


class MOELayer(Module):
    def __init__(self, gate, expert):
        super().__init__()
        # set gate
        self.gate = gate

        # set expert
        self.expert = expert

    def forward(self, x):
        logits_dtype, (crit, l_aux) = self.gate(x, self.expert.num_shards, a2a_ffn_overlap_degree=self.expert.a2a_ffn_overlap_degree)
        self.l_aux = l_aux  # enable loss_fn to access auxilary loss

        y = self.expert(x, logits_dtype, crit)
        y.l_aux = l_aux  # enable loss_fn to access auxilary loss

        return y


moe_layer = MOELayer
