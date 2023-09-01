import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

##############################################
#
#            GNN - Modules
#
##############################################
"""
N : number of nodes
D : number of features per node
E : number of classes

@ input :
    - adjacency matrix (N x N)
    - feature matrix (N x D)
    - label matrix (N x E)

"""


class GraphConv(Module):
    """
        GCN layer, refer:  https://arxiv.org/abs/1609.02907
    """

    def __init__(self, input_dim: int, output_dim: int, bias=True, type: str = 'xavier'):
        super(GraphConv, self).__init__()
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.weight = Parameter(torch.FloatTensor(self.in_dim, self.out_dim))
        if bias:
            self.bias = Parameter(torch.FloatTensor(self.out_dim))
        else:
            self.register_parameter('bias', None)
        if type == 'uniform':
            self.init_uniform()
        elif type == 'xavier':
            self.init_xavier()
        elif type == 'kaiming':
            self.init_kaiming()
        else:
            raise NotImplementedError

    def init_uniform(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform(-stdv, stdv)

    def init_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02)  # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def init_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, input: torch.Tensor, adj: torch.Tensor):
        # H_{l+1} = W * H_{l}
        output = torch.mm(input, self.weight)
        if not isinstance(adj, (float, int)):
            output = torch.spmm(adj, output)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + \
               str(self.in_dim) + ' -> ' + str(self.out_dim) + ')'
