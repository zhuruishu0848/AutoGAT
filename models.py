import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *





class GNN(nn.Module):
    def __init__(self, args, input_dim, output_dim, hidden_dim, gru_step, dropout_p):
        super(GNN,self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.nhid = 64
        self.nclass = 32
        self.dropout_p = dropout_p
        self.gru_step = gru_step
        self.LSTM = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,#96
            num_layers=1,#lstm层数
            batch_first=True,#(batch,seq,dim)
            bidirectional=False
        )

        # self.SpGATLayer = SpGATLayer(
        #     nfeat=self.input_dim,
        #     nhid=self.nhid,
        #     nclass=self.hidden_dim,
        #     dropout=dropout_p,
        #     alpha=0.2,
        #     nheads=2,
        # )
        self.GSLLayer = GSLLayer(
            nfeat=self.input_dim,
            nhid=self.nhid,
            nclass=self.nclass,
            dropout=dropout_p,
            alpha=0.2,
            nheads=2,
        )
        self.GraphLayer = GraphLayer(
            args = args,
            input_dim = self.hidden_dim,
            output_dim = self.nhid,
            # output_dim = self.hidden_dim,
            act = torch.nn.Tanh(),
            dropout_p = self.dropout_p,
            gru_step = self.gru_step
        )
        self.ReadoutLayer = ReadoutLayer(
            args=args,
            input_dim = self.nhid,
            output_dim = self.output_dim,
            act = torch.nn.Tanh(),
            dropout_p = self.dropout_p
        )
        self.layers = [
            # self.SpGATLayer,
            self.GraphLayer,
            self.ReadoutLayer]


    def forward(self, feature, support, mask):#train_feature[idx], train_adj[idx], train_mask[idx]、
        # dv = 'cuda' if feature.is_cuda else 'cpu'
        # N = support.size()[-1]
        # support += torch.eye(N, device=dv)
        support = torch.mul(support, self.GSLLayer(feature, support))
        # support = self.GSLLayer(feature, support)
        feature, _ = self.LSTM(feature, None)
        activations = [feature]
        for index, layer in enumerate(self.layers):
            hidden = layer(activations[-1], support, mask)
            activations.append(hidden)
        embeddings = activations[-2]
        outputs = activations[-1]
        return outputs,embeddings

