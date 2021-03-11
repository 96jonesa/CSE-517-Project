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

    def forward(self, input):
        intermediate = self.tanh(self.linear_1(input))
        attention_weights = self.softmax(self.linear_2(intermediate))
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


class MANSF(nn.Module):
    def __init__(self, T, num_stocks, gru_hidden_size, attn_inter_size, use_embed_size,
                 blend_size, gat_1_inter_size, gat_2_inter_size, leakyrelu_slope, elu_alpha, U):
        super(MANSF, self).__init__()
        self.T = T
        self.num_stocks = num_stocks
        self.gru_hidden_size = gru_hidden_size
        self.attn_inter_size = attn_inter_size
        self.use_embed_size = use_embed_size
        self.blend_size = blend_size
        self.gat_1_inter_size = gat_1_inter_size
        self.gat_2_inter_size = gat_2_inter_size
        self.leakyrelu_slope = leakyrelu_slope
        self.elu_alpha = elu_alpha
        self.U = U

        self.gru_p = GRU(3, gru_hidden_size, batch_first=True)
        self.gru_m = GRU(use_embed_size, gru_hidden_size, batch_first=True)
        self.gru_s = GRU(gru_hidden_size, gru_hidden_size, batch_first=True)
        self.attn_p = LinearAttention(gru_hidden_size, attn_inter_size, 1)
        self.attn_m = LinearAttention(gru_hidden_size, attn_inter_size, 1)
        self.attn_s = LinearAttention(gru_hidden_size, attn_inter_size, 1)
        self.blend = Blend(gru_hidden_size, gru_hidden_size, blend_size)
        self.mgat_1 = nn.ModuleList([SGAT(blend_size, gat_1_inter_size, leakyrelu_slope=leakyrelu_slope) for u in range(U)])
        self.mgat_2 = nn.ModuleList([SGAT(U * gat_1_inter_size, gat_2_inter_size, leakyrelu_slope=leakyrelu_slope) for u in range(U)])
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU(elu_alpha)
        self.final_linear = nn.Linear(U * gat_2_inter_size, 1, bias=True)

    # p is price data tensor of shape (num_stocks, T, 3), for the day under consideration
    # m is smi data list of tensors of shape (num_stocks, K, use_embed_size) of length T,
    #       where K is the number of tweets for the given stock on the day under consideration
    # neighorhoods is a list of adjacency lists, where each stock is indexed with the same
    #       indices they have in p and m
    def forward(self, p, m, neighborhoods):
        ## price encoding
        h_p, _ = self.gru_p(p)
        q = self.attn_p(h_p)

        ## smi encoding (day level)
        r = torch.zeros(self.num_stocks, 0, self.gru_hidden_size)
        r = r.to(device)
        for t in range(self.T):
            h_m, _ = self.gru_m(m[t])
            r_t = self.attn_m(h_m)
            r = torch.cat((r, r_t), 1)

        ## smi encoding (aggregate)
        h_s, _ = self.gru_s(r)
        c = self.attn_s(h_s)

        ## blending
        x = self.blend(q, c)

        ## reshaping (eliminating superfluous dimension)
        x = x.view(x.shape[0], x.shape[2])

        ## first gat layer
        #  first head
        sgat = self.mgat_1[0]
        z = sgat(x, neighborhoods)
        z = self.elu(z)

        #  remaining heads
        for u in range(1, self.U):
            sgat = self.mgat_1[u]
            z_u = sgat(x, neighborhoods)
            z_u = self.elu(z_u)

            z = torch.cat((z, z_u), 1)

        ## second gat layer
        #  first head
        sgat = self.mgat_2[0]
        new_z = sgat(z, neighborhoods)
        new_z = self.sigmoid(new_z)

        #  remaining heads
        for u in range(1, self.U):
            sgat = self.mgat_2[u]
            new_z_u = sgat(z, neighborhoods)
            new_z_u = self.sigmoid(new_z_u)

            new_z = torch.cat((new_z, new_z_u), 1)

        ## final layer
        y = self.sigmoid(self.final_linear(new_z))

        ## return result
        return y
