import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.gru = nn.GRU(input_size, hidden_size, batch_first=self.batch_first)

    def forward(self, input):
        output, hn = self.gru(input)
        return output, hn


class LinearAttention(nn.Module):
    def __init__(self, input_size, intermediate_size, weights_size):
        super(LinearAttention, self).__init__()
        self.input_size = input_size
        self.intermediate_size = intermediate_size
        self.weights_size = weights_size

        self.linear_1 = nn.Linear(self.input_size, self.intermediate_size, bias=True)
        self.linear_2 = nn.Linear(self.intermediate_size, self.weights_size, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, input, mask=None):
        intermediate = self.tanh(self.linear_1(input))
        pre_attention = self.linear_2(intermediate)
        if mask is not None:
            zero_vec = -9e15*torch.ones_like(pre_attention)
            pre_attention = torch.where(mask > 0, pre_attention, zero_vec)
        attention_weights = self.softmax(pre_attention)
        attention_weights = attention_weights.permute(0, 2, 1)
        output_features = torch.bmm(attention_weights, input)

        return output_features


class Blend(nn.Module):
    def __init__(self, left_size, right_size, output_size):
        super(Blend, self).__init__()
        self.left_size = left_size
        self.right_size = right_size
        self.output_size = output_size

        self.bilinear = nn.Bilinear(self.left_size, self.right_size, output_size, bias=True)
        self.relu = nn.ReLU()

    def forward(self, left, right):
        output = self.relu(self.bilinear(left, right))

        return output


# https://github.com/Diego999/pyGAT/blob/master/layers.py
class SGAT(nn.Module):
    def __init__(self, input_size, output_size, leakyrelu_slope=0.01):
        super(SGAT, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.leakyrelu_slope = leakyrelu_slope

        self.W = nn.Parameter(torch.empty(size=(input_size, output_size)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*output_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.leakyrelu_slope)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        h_prime = torch.matmul(attention, Wh)

        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0] # number of nodes

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)

        return all_combinations_matrix.view(N, N, 2 * self.output_size)
