import torch
import torch.nn as nn
import torch.nn.functional as F
from inits import glorot,xavier

class GAT_unit(nn.Module):
    """
    copy from https://github.com/Diego999/pyGAT
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):

        super(GAT_unit, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input_h, adj):
        Wh = torch.matmul(input_h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)

        onemask_vec = -9e15*torch.ones_like(e)
        e = torch.where(adj > 0, e, onemask_vec)
        attention = F.softmax(e, dim=1)

        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        temp = torch.transpose(Wh2, 1, -1)
        e = Wh1 + temp
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGAT_unit(nn.Module):
    """
    copy from https://github.com/Diego999/pyGAT
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGAT_unit, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.alphaelu = 1
        self.elu = nn.ELU(self.alphaelu)

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        assert not torch.isnan(h).any()

        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        e_rowsum = e_rowsum + torch.full((N, 1), 1e-25).cuda()

        edge_e = self.dropout(edge_e)

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()

        h_prime = h_prime.div(e_rowsum)
        assert not torch.isnan(h_prime).any()

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GSL_unit(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha):
        super(GSL_unit, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.alphaelu = 1
        self.elu = nn.ELU(self.alphaelu)

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]

        adj2 = torch.matmul(adj, adj.transpose(1, -1))
        adj2 = F.normalize(adj2, dim=1)

        edge = adj2.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        edge_e = torch.exp(-self.elu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        e_rowsum = e_rowsum + torch.full((N, 1), 1e-25, device=dv)

        edge_e = self.dropout(edge_e)

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), b=torch.eye(N, device=dv))
        assert not torch.isnan(h_prime).any()

        pij = h_prime.div(e_rowsum)
        assert not torch.isnan(pij).any()

        return pij

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class gru_unit(nn.Module):
    """
        copy from https://arxiv.org/abs/2004.13826
    """
    def __init__(self, output_dim, act, dropout_p):
        super(gru_unit,self).__init__()
        self.output_dim = output_dim
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(1-self.dropout_p)
        self.act = act
        self.z0_weight = glorot([self.output_dim, self.output_dim])
        self.z1_weight = glorot([self.output_dim, self.output_dim])
        self.r0_weight = glorot([self.output_dim, self.output_dim])
        self.r1_weight = glorot([self.output_dim, self.output_dim])
        self.h0_weight = glorot([self.output_dim, self.output_dim])
        self.h1_weight = glorot([self.output_dim, self.output_dim])
        self.z0_bias = nn.Parameter(torch.zeros(self.output_dim))
        self.z1_bias = nn.Parameter(torch.zeros(self.output_dim))
        self.r0_bias = nn.Parameter(torch.zeros(self.output_dim))
        self.r1_bias = nn.Parameter(torch.zeros(self.output_dim))
        self.h0_bias = nn.Parameter(torch.zeros(self.output_dim))
        self.h1_bias = nn.Parameter(torch.zeros(self.output_dim))

    def forward(self,support, x, mask):
        support = self.dropout(support)
        a = torch.matmul(support, x)
        # updata gate
        z0 = torch.matmul(a, self.z0_weight) + self.z0_bias
        z1 = torch.matmul(x, self.z1_weight) + self.z1_bias
        z = torch.sigmoid(z0+z1)
        # reset gate
        r0 = torch.matmul(a, self.r0_weight) + self.r0_bias
        r1 = torch.matmul(x, self.r1_weight) + self.r1_bias
        r = torch.sigmoid(r0+r1)
        # update embeddings
        h0 = torch.matmul(a, self.h0_weight) + self.h0_bias
        h1 = torch.matmul(r*x, self.h1_weight) + self.h1_bias
        h = self.act(mask * (h0 + h1))
        return h*z + x*(1-z)

class GATlayer(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GATlayer, self).__init__()
        self.dropout = dropout

        self.attentions = [GAT_unit(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]#multi-head
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GAT_unit(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    # def forward(self, x, adj):
    def forward(self, x, adj, mask):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        x = mask * x
        return F.log_softmax(x, dim=1)


class SpGATLayer(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGATLayer, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGAT_unit(nfeat,
                                      nhid,
                                      dropout=dropout,
                                      alpha=alpha,
                                      concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGAT_unit(nhid * nheads,
                                  nclass,
                                  dropout=dropout,
                                  alpha=alpha,
                                  concat=False)

    def forward(self, x, adj, mask):
        x_list = torch.unbind(x, 0)
        adj_list = torch.unbind(adj, 0)
        mask_list = torch.unbind(mask, 0)
        X_out_list = []
        for i in range(len(x_list)):
            x = x_list[i]
            adj = adj_list[i]
            mask = mask_list[i]
            x = F.dropout(x, self.dropout, training=self.training)
            x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)  # 拼接多个h
            x = mask * x
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.elu(self.out_att(x, adj))
            x = mask * x
            x = F.log_softmax(x, dim=1)
            X_out_list.append(x)

        out = torch.stack(X_out_list, dim=0)
        return out

class GSLLayer(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GSLLayer, self).__init__()
        self.dropout = dropout

        self.attention = GSL_unit(nfeat,
                                nhid,
                                dropout=dropout,
                                alpha=alpha)

    def forward(self, x, adj):
        x_list = torch.unbind(x, 0)
        adj_list = torch.unbind(adj, 0)
        adj_out_list = []
        for i in range(len(x_list)):
            x = x_list[i]
            adj = adj_list[i]
            pij = self.attention(x, adj)
            adj_out_list.append(pij)

        out = torch.stack(adj_out_list, dim=0)
        return out

class GraphLayer(nn.Module):
    """Graph layer."""
    def __init__(self, args,
                      input_dim,
                      output_dim,
                      act=nn.Tanh(),
                      dropout_p = 0.,
                      gru_step = 2):
        super(GraphLayer, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(1-self.dropout_p)
        self.gru_step = gru_step
        self.gru_unit = gru_unit(output_dim = self.output_dim,
                                 act = self.act,
                                 dropout_p = self.dropout_p)
        # self.dropout
        self.encode_weight = glorot([self.input_dim, self.output_dim])
        self.encode_bias = nn.Parameter(torch.zeros(self.output_dim))



    def forward(self, feature, support, mask):
        feature = self.dropout(feature)
        # encode inputs
        encoded_feature = torch.matmul(feature, self.encode_weight) + self.encode_bias
        output = mask * self.act(encoded_feature)
        # convolve
        for _ in range(self.gru_step):
            output = self.gru_unit(support, output, mask)
        return output

class ReadoutLayer(nn.Module):
    """Graph Readout Layer."""
    def __init__(self, args,
                 input_dim,
                 output_dim,
                 act=nn.ReLU(),
                 dropout_p=0.):
        super(ReadoutLayer, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(1-self.dropout_p)
        self.att_weight = glorot([self.input_dim, 1])
        self.emb_weight = glorot([self.input_dim, self.input_dim])
        self.mlp_weight = glorot([self.input_dim, self.output_dim])
        self.att_bias = nn.Parameter(torch.zeros(1))
        self.emb_bias = nn.Parameter(torch.zeros(self.input_dim))
        self.mlp_bias = nn.Parameter(torch.zeros(self.output_dim))

    def forward(self,x,_,mask):  # _ not used
        # soft attention
        att = torch.sigmoid(torch.matmul(x, self.att_weight)+self.att_bias)
        emb = self.act(torch.matmul(x, self.emb_weight)+self.emb_bias)
        N = torch.sum(mask, dim=1)
        M = (mask - 1) * 1e9
        # graph summation
        g = mask * att * emb
        g = torch.sum(g, dim=1)/N + torch.max(g+M,dim=1)[0]
        g = self.dropout(g)
        # classification
        output = torch.matmul(g,self.mlp_weight)+self.mlp_bias
        return output
