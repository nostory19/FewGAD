import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from utils import *
from sklearn.neighbors import KernelDensity
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        # pdb.set_trace()
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)


class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq, 1).values


class MinReadout(nn.Module):
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values


class WSReadout(nn.Module):
    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq, sim)
        out = torch.sum(out, 1)
        return out


class Discriminator1(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Discriminator1, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

        self.negsamp_round = 1

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, c_neg, cur_neg_index):
        scs = []
        # positive
        scs.append(self.f_k(h_pl, c))

        # neg_scores = []
        for i in range(c_neg.shape[0]):
            neg_sample = c_neg[i].expand(h_pl.shape[0], -1)
            scs.append(self.f_k(h_pl, neg_sample))
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-2:-1, :], c_mi[:-1, :]), 0)
            if cur_neg_index.numel() > 0:
                c_mi[cur_neg_index] = c[cur_neg_index]
            scs.append(self.f_k(h_pl, c_mi))

        logits = torch.cat(tuple(scs))

        return logits

class DiffusionDenoiser(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DiffusionDenoiser, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        x_recon = F.relu(self.fc2(h))

        return x_recon


class Model(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout):
        super(Model, self).__init__()
        self.read_mode = readout
        self.gcn = GCN(n_in, n_h, activation)
        self.gcn_dec = GCN(n_h, n_in, activation)
        self.gcn_dec2 = GCN(n_h*2, n_in, activation)
        self.gcn_dec_sub = nn.Sequential(
            nn.Linear(n_h, n_h),
            nn.PReLU(),
            nn.Linear(n_h, n_h),
            nn.PReLU(),
            nn.Linear(n_h, n_in),
            nn.PReLU()
        )
        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()

        self.disc1 = Discriminator1(n_h, negsamp_round)

        self.denoiser = DiffusionDenoiser(input_dim=n_h, hidden_dim=32)
    def forward(self, seq1, adj, seq2, adj2, raw_bf1, cur_neg_index, sparse=False):
        h_1 = self.gcn(seq1, adj, sparse) # (200, 16, 64)
        h_2 = self.gcn(seq2, adj2, sparse)
        h_raw1 = self.gcn(raw_bf1, adj, sparse)
        f_1 = self.gcn_dec(h_raw1, adj, sparse)


        if self.read_mode != 'weighted_sum':
            c = self.read(h_1[:, : -1, :])
            h_mv = h_1[:, -1, :]
            c_neg = self.read(h_2[:, :, :])
            kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
            kde.fit(c_neg.detach().cpu().numpy())
            new_data = kde.sample(100)
            new_data = torch.tensor(new_data, dtype=torch.float32).to(device)
            c_neg = torch.cat([c_neg, new_data], dim=0)
        else:
            h_mv = h_1[:, -1, :]
            c = self.read(h_1[:, : -1, :], h_1[:, -2: -1, :])

        ret = self.disc1(c, h_mv, c_neg, cur_neg_index)

        return ret, f_1



    def analyse(self, features, adj, sparse=False):
        emd = self.gcn(features, adj, sparse)

        return emd