import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, input_feats, output_feats, nodes):
        super(GraphConvolution, self).__init__()
        self.input_feats = input_feats
        self.output_feats = output_feats
        self.nodes = nodes
        self.W = nn.Parameter(torch.FloatTensor(input_feats, output_feats))
        self.A = nn.Parameter(torch.FloatTensor(nodes, nodes))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.constant_(self.A, 1e-3)  # small init for stability

    def forward(self, x):
        Wx = torch.matmul(x, self.W)  # (B, N, F_out)
        output = torch.matmul(self.A, Wx)  # (B, N, F_out)
        return output


class GCNLayer(nn.Module):
    def __init__(self, input_feats, output_feats, nodes, dropout):
        super(GCNLayer, self).__init__()
        self.gc = GraphConvolution(input_feats, output_feats, nodes)
        self.bn = nn.BatchNorm1d(nodes * output_feats)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        y = self.gc(x)  # (B, N, F)
        B, N, F_out = y.shape
        y = self.bn(y.view(B, -1)).view(B, N, F_out)
        y = self.activation(y)
        y = self.dropout(y)
        if residual.shape == y.shape:
            y = y + residual
        return y


class GCN_in(nn.Module):
    def __init__(self, input_feats, model_dim, joint_num, dropout=0.2, num_layers=3):
        super(GCN_in, self).__init__()
        self.joint_num = joint_num
        self.input_feats = input_feats
        self.model_dim = model_dim
        self.hidden_dim = model_dim

        layers = []
        in_dim = input_feats
        for i in range(num_layers - 1):
            layers.append(GCNLayer(in_dim, self.hidden_dim, joint_num, dropout))
            in_dim = self.hidden_dim
        layers.append(GCNLayer(in_dim, model_dim, joint_num, dropout))
        self.gcn_layers = nn.Sequential(*layers)

        self.to_time_embed = nn.Sequential(
            nn.Flatten(start_dim=1),  # (B*T, J, D) → (B*T, J*D)
            nn.Linear(joint_num * model_dim, model_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, T, D = x.shape
        x = x.view(B * T, self.joint_num, self.input_feats)
        y = self.gcn_layers(x)
        y = self.to_time_embed(y)  # (B*T, model_dim)
        y = y.view(B, T, self.model_dim)
        return y


class GCN_out(nn.Module):
    def __init__(self, model_dim, output_feats, joint_num, dropout=0.2, num_layers=3):
        super(GCN_out, self).__init__()
        self.joint_num = joint_num
        self.output_feats = output_feats
        self.model_dim = model_dim
        self.hidden_dim = model_dim

        self.to_joint_embed = nn.Sequential(
            nn.Linear(model_dim, joint_num * self.hidden_dim),
            nn.Dropout(dropout)
        )

        layers = []
        in_dim = self.hidden_dim
        for i in range(num_layers - 1):
            layers.append(GCNLayer(in_dim, self.hidden_dim, joint_num, dropout))
        layers.append(GCNLayer(self.hidden_dim, output_feats, joint_num, dropout))
        self.gcn_layers = nn.Sequential(*layers)

    def forward(self, x):
        B, T, D = x.shape
        y = self.to_joint_embed(x).view(B * T, self.joint_num, -1)
        y = self.gcn_layers(y)  # (B*T, J, 3)
        y = y.view(B, T, self.joint_num, self.output_feats)

        y = y.view(B, T, -1)  # -> (B, T, J*3)

        return y




if __name__ == '__main__':
    x = torch.randn(16, 25, 17 * 3)
    nodes = 17
    gcn_in = GCN_in(3, 512, nodes, 0.2)
    y = gcn_in(x)
    print(y.shape)
    gcn_out = GCN_out(512, 3, nodes, 0.2)
    y = gcn_out(y)
    print(y.shape)
