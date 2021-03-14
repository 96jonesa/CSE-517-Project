import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from layers import GRU, LinearAttention, Blend, SGAT


class MANSF(nn.Module):
    def __init__(self, T, gru_hidden_size, attn_inter_size, use_embed_size,
                 blend_size, gat_1_inter_size, gat_2_inter_size, leakyrelu_slope, elu_alpha, U):
        super(MANSF, self).__init__()
        self.T = T
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
    def forward(self, p, m, m_mask, neighborhoods):
        ## price encoding
        h_p, _ = self.gru_p(p)
        q = self.attn_p(h_p)

        ## smi encoding (day level)
        r = torch.zeros(p.shape[0], 0, self.gru_hidden_size)
        r = r.to(device)
        for t in range(self.T):
            h_m, _ = self.gru_m(m[t])
            r_t = self.attn_m(h_m, m_mask[t])
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
