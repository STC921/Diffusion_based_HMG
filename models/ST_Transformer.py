import torch
import torch.nn as nn
import numpy as np
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from models.temporal_transformer import TemporalMotionTransformer
from models.spatial_transformer import SpatialMotionTransformer

class MotionTransformer(nn.Module):
    def __init__(self,
                 input_feats=3,
                 num_frames=240,
                 sequence_len=125,
                 joint_num=17,
                 t_latent_dim=512,
                 s_latent_dim=64,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.2,
                 activation="gelu"):
        super(MotionTransformer, self).__init__()

        self.input_feats = input_feats
        self.num_frames = num_frames
        self.sequence_len = sequence_len
        self.joint_num = joint_num
        self.t_latent_dim = t_latent_dim
        self.s_latent_dim = s_latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation

        self.t_model = TemporalMotionTransformer(
            input_feats=self.input_feats*self.joint_num,
            num_frames=self.num_frames,
            joint_num=self.joint_num,
            latent_dim=self.t_latent_dim,
            ff_size=self.ff_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            activation=self.activation)
        self.s_model = SpatialMotionTransformer(
            input_feats=self.input_feats*self.sequence_len,
            joint_num=self.joint_num,
            latent_dim=self.s_latent_dim,
            ff_size=self.ff_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            activation=self.activation)

        t_params = filter(lambda p: p.requires_grad, self.t_model.parameters())
        s_params = filter(lambda p: p.requires_grad, self.s_model.parameters())
        nparams = sum([np.prod(p.size()) for p in t_params]) + sum([np.prod(p.size()) for p in s_params])
        print('[INFO] ({}) MotionTransformer has {} params!'.format(self.__class__.__name__, nparams))

    def forward(self, x, timesteps=None):
        B, T, J, D = x.shape
        x_t = x.reshape(B, T, -1).contiguous()
        x_s = x.permute(0, 2, 1, 3).reshape(B, J, -1).contiguous()

        self.t_out = self.t_model(x_t, timesteps=timesteps)
        self.s_out = self.s_model(x_s, timesteps=timesteps)

        self.t_out = self.t_out.reshape(B, T, J, D).contiguous()
        self.s_out = self.s_out.reshape(B, J, T, D).permute(0, 2, 1, 3).contiguous()

        return self.t_out + self.s_out


if __name__ == '__main__':
    model = MotionTransformer()
    x = torch.randn(16, 125, 17, 3)
    t = torch.randint(0, 1000, (16,))
    out = model(x, t)
    print(out.shape)