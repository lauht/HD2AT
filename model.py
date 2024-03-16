import math
import torch
import torch.nn as nn
import torch.nn.functional as F

inf = math.inf

class GraphAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GraphAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        support = torch.mm(input_feature, self.weight)
        output = torch.matmul(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.input_dim) + ' -> ' \
            + str(self.output_dim) + ')'
    
class RespEmb(nn.Module):
    def __init__(self, args, n_feat, v_dim, rnn_layers, dropout):
        super(RespEmb, self).__init__()
        self.args = args
        self.trans = nn.LSTM(
            input_size=n_feat,
            hidden_size=v_dim,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, X):
        timespan = []
        for i in range(self.args.window_size_o+1):
            x_hidden, _ = self.trans(X[:, i:i+self.args.window_size, :])
            x_hidden = x_hidden[:, -1, :]
            timespan.append(x_hidden)

        V = self.leaky_relu(torch.stack(timespan, dim=0))
        return V
    
class NodeProp(nn.Module):
    def __init__(self, args, k, v_dim, h_dim):
        super(NodeProp, self).__init__()
        ## aggregator
        self.args = args
        self.agg_list = nn.ModuleList()
        for _ in range(k):
            self.agg_list.append(GraphAggregator(v_dim, h_dim, use_bias=False))

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, V, adj):
        H = []
        for i, agg in enumerate(self.agg_list):
            h = []
            for j in range(self.args.window_size_o):
                ht = agg(adj[j, i, :, :], V[j, :, :])
                h.append(ht)
            h = torch.stack(h, dim=0) 
            H.append(h)
        H = torch.stack(H, dim=0) 
        H = self.leaky_relu(H) 
        return H
    
class TimeDynAtt(nn.Module):
    def __init__(self, k, v_dim, f_dim, h_dim):
        super(TimeDynAtt, self).__init__()

        self.f_dim = f_dim
        self.theta = nn.Parameter(torch.Tensor(1, k))
        self.Wfv = nn.Parameter(torch.Tensor(v_dim, f_dim))
        self.Wfh = nn.Parameter(torch.Tensor(k, h_dim, f_dim))
        self.Whh = nn.Parameter(torch.Tensor(k, h_dim, h_dim))
        self.reset_parameters()

        self.leaky_relu = nn.LeakyReLU()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta)
        nn.init.xavier_uniform_(self.Wfv)
        nn.init.xavier_uniform_(self.Wfh)
        nn.init.xavier_uniform_(self.Whh)

    def forward(self, V, H):
        To = H.size(1)
        N = H.size(2)
        vt = V[-1, :, :]

        f_key = vt.mm(self.Wfv) 
        f_query = H.permute(1,0,2,3).matmul(self.Wfh) 
        f = f_query.unsqueeze(3).matmul(f_key.unsqueeze(2)).squeeze() 
        
        timedecay = torch.exp(torch.linspace(1, To, To).unsqueeze(1).matmul(self.theta)).unsqueeze(2).repeat(1,1,N)
        f = self.leaky_relu(-f) / torch.sqrt(torch.Tensor([self.f_dim])) * timedecay
        f = torch.exp(f)
        alpha = (f / f.sum(dim=0).repeat(To,1,1)).permute(1,2,0) 

        H_trans = H.permute(1,0,2,3).matmul(self.Whh) 
        H_tilde = alpha.unsqueeze(2).matmul(H_trans.permute(1,2,0,3)).squeeze()
        H_tilde = self.leaky_relu(H_tilde)

        return H_tilde
    
class GraphHeteroAtt(nn.Module):
    def __init__(self, k, v_dim, h_dim, g_dim, u_dim):
        super(GraphHeteroAtt, self).__init__()
        self.k = k
        self.g_dim = g_dim
        self.Wgv = nn.Parameter(torch.Tensor(v_dim, g_dim))
        self.Wgh = nn.Parameter(torch.Tensor(k, h_dim, g_dim))
        self.Wuh = nn.Parameter(torch.Tensor(k, h_dim, u_dim))
        self.reset_parameters()

        self.leaky_relu = nn.LeakyReLU()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.Wgv)
        nn.init.xavier_uniform_(self.Wgh)
        nn.init.xavier_uniform_(self.Wuh)
    
    def forward(self, vt, H):
        g_key = vt.mm(self.Wgv) 
        g_query = H.matmul(self.Wgh)
        g = g_query.unsqueeze(2).matmul(g_key.unsqueeze(2)).squeeze()
        g = self.leaky_relu(g) / torch.sqrt(torch.Tensor([self.g_dim]))
        g = torch.exp(g)
        beta = g / g.sum(dim=0).repeat(self.k,1)

        H_trans = H.matmul(self.Wuh)
        U = beta.permute(1,0).unsqueeze(1).matmul(H_trans.permute(1,0,2)).squeeze()
        U = self.leaky_relu(U)

        return U

class HDAT(nn.Module):
    def __init__(self, n_feat, args, emb_dim=64, n_out=1, n_heads=6, dropout=0.3):
        super(HDAT, self).__init__()
        self.sample_weight = 1
        self.k = 12

        self.rnn_hid = emb_dim
        self.rnn_layers = 2

        self.v_dim = 64
        self.u_dim = 32
        self.d_dim = 32
        self.h_dim = 96
        self.f_dim = 128
        self.u_dim = 96
        self.g_dim = 64

        self.num_node = 505

        self.RespEncoder = RespEmb(args, n_feat, self.v_dim, self.rnn_layers, dropout)
        self.NodeProp = NodeProp(args, self.k, self.v_dim, self.h_dim)
        self.TimeDynAtt = TimeDynAtt(self.k, self.v_dim, self.f_dim, self.h_dim)
        self.GraphHeteroAtt = GraphHeteroAtt(self.k, self.v_dim, self.h_dim, self.g_dim, self.u_dim)
        self.A_fusion = nn.Parameter(torch.Tensor(self.num_node, self.v_dim+self.u_dim, 1))
        self.b_fusion = nn.Parameter(torch.Tensor(self.num_node, 1))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.A_fusion)
        nn.init.xavier_uniform_(self.b_fusion)

    def forward(self, input, adj):
        adj = adj.to(input.device) 

        V = self.RespEncoder(input)
        H = self.NodeProp(V, adj) 
        H_tilde = self.TimeDynAtt(V, H) 
        U = self.GraphHeteroAtt(V[-1, :, :], H_tilde) 
        
        # Signal Fusion
        uv = torch.cat([V[-1, :, :], U], dim=-1)
        y_hat = uv.unsqueeze(1).matmul(self.A_fusion).squeeze() + self.b_fusion.squeeze()
        y_hat = torch.tanh(y_hat)
        return y_hat
    
## ----------------- Loss Function --------------------
    
class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()

    def forward(self, x, y):
        pred_loss = torch.sqrt(F.mse_loss(input=x, target=y))
        return pred_loss
